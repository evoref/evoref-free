"""

``agent_trace*.jsonl`` (日付付きファイル含む) をエピソード単位でグルー
ピングし、終了 (``end``) イベントから SemanticFact 候補を生成する。
``AgentTraceStore`` は ``agent_trace_YYYY-MM-DD.jsonl`` という日付付きファイル
名で出力するため、本 extractor は ``ctx.agent_trace_dir`` 配下の
``agent_trace*.jsonl`` をグロブで横断する

抽出されるのは:

- ``failure_pattern`` — outcome に ``fail``/``error`` を含むエピソード。
  subject は ``loop.failure.<sha1[:12]>`` (owner は EvorefLoop のため
  ``loop.*`` 名前空間を使う)。``failure_signature`` フィールドを設定。
- ``decision`` — 成功エピソード。最後の ``action`` を要約として保存。
  subject は ``mem.decision.<episode_id>`` (owner は EvorefMem のため
  ``mem.*`` 名前空間を使う)。

スコープは常に ``project:<project_id>``。``ctx.project_id`` が ``None`` の場合
extractor は no-op となる。

設計原則:

- LLM 不要 (失敗シグネチャは SHA1 で機械的に算出)
- ``private`` の判定は ``begin`` イベントの ``private`` 印 (一次情報) と、行の
  ``trace_id`` が STM の private ノートの ``trace_id`` に一致するかの 2 本
  (``MDPIngester`` と同じ ``EpisodeRecord`` の述語を使う)。以前は episode_id を
  STM の trace_id と比べており一度も一致しなかった。
- 記憶の読み出しだけで完結したエピソード (``search_history`` のみ) は Step 7.5
  と同じく抽出しない (注入テキストが decision として蘇る循環を作らない)。
- fact / provenance の ``trace_id`` はリクエストの trace_id (行の ``trace_id``、
  無ければ episode_id) — episodic LTM ノート (Step 7.5) と同じ値になり、
  ファクトとノートを trace_id で連結できる。
- 読み手は ``MDPIngester`` (別 state ``local/memory/mdp_extract_state.json``)。
  ファイル別オフセットで差分だけ読み、処理済み episode_id を永続化する。
  in-memory だけだと再起動後に 30 日分の failure_pattern をもう一度
  occurrences 加算してしまう (decision は subject dedup で守られるが
  failure_pattern は signature 単位の加算なので二重計上になる)。

ファイル形式 (``debug_logger.log_agent_trace_event`` 由来)::

    {"event": "begin", "episode_id": "ep_xxx", "conversation_id": "...", "mode": "create", ...}
    {"event": "step", "episode_id": "ep_xxx", "step_index": 0, "action": "...", ...}
    {"event": "end", "episode_id": "ep_xxx", "outcome": "success" | "failure: ...", ...}
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from backend.free.core.session_mode import is_valid_session_mode
from backend.free.memory.extractors.base import (
    BaseExtractor,
    ExtractionContext,
    ExtractionResult,
)
from backend.free.memory.stores.short_term import MemoryNote
from backend.free.memory.types import SemanticFact
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.free.memory.notes.mdp_ingester import MDPIngester

logger = get_logger("memory.extractors.mdp_trace")


LOOP_FAILURE_PREFIX = "loop.failure."
"""failure_pattern ファクトの subject prefix (owner は EvorefLoop)"""

MEM_DECISION_PREFIX = "mem.decision."
"""decision ファクトの subject prefix (owner は EvorefMem)"""

_FAILURE_KEYWORDS = ("fail", "error", "exception", "abort")


def _is_failure_outcome(outcome: str, steps: list[dict[str, Any]] | None = None) -> bool:
    """エピソード outcome が失敗かを判定する。

    outcome 文字列のキーワードに加え、``partial`` で全ステップ reward 0 の
    エピソードも失敗として扱う (2026-07-15: write 不発の誤ルーティング 2 件が
    outcome=partial / reward=0.0 で decision ファクト化され、failure_pattern が
    1 件も生成されなかった)。
    """
    if not outcome:
        return False
    s = outcome.lower()
    if any(k in s for k in _FAILURE_KEYWORDS):
        return True
    if s == "partial" and steps:
        return all(not (st.get("reward") or 0) for st in steps)
    return False


#: ホスト計測ブロック (``system_hardware_info`` / spec 系 ``run_command_readonly``)
#: の見出し。``OS:`` で始まり ``Cores:`` を含む形は両者に共通。
_HOST_MEASUREMENT_RE = re.compile(r"\AOS:.*\n(?:.*\n)*?Cores:", re.MULTILINE)

#: ホスト計測ブロックのうち **時間で変わる値**。エピソード記憶に残すと、
#: 後日の検索でその数値が「現在の値」として注入される。
#:
#: 実インシデント (2026-08-19 ライブ監査 再検証): 「このPCのスペックを教えて」
#: に spec コマンド (OS/CPU/Cores/Disk のみ、RAM を一切出さない) が走った直後、
#: 27 分前の別ターンの計測を貼り付けた LTM チャンク
#: ``[mdp_trace] ... result=... RAM: 63.3 GB total (64795 MB), 23.6 GB available;
#: CPU usage: 10.5%`` が RAG 1 位 (0.3966) で注入され、モデルが
#: 「空き 23.6 GB」を **今回の計測値として** 回答した (実際の当該時点の空きは
#: 23.0 GB / 22.4 GB)。
#:
#: 静的な値 (OS / CPU / コア数 / 総 RAM / ディスク総量) は変わらないので残す。
#: 値を消すだけでラベルは残すのは、``read_file`` のメタ行だけ残す扱いと同じ理由 —
#: 「何を測ったか」は記憶に値するが「そのときいくつだったか」は値しない。
_VOLATILE_FIELD_SUBS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^CPU usage:.*$", re.MULTILINE), "CPU usage: (not recorded)"),
    (re.compile(r"^(GPU[^\n:]*:).*$", re.MULTILINE), r"\1 (not recorded)"),
    (re.compile(r"(^RAM:[^\n]*?),\s*[\d.]+\s*GB available", re.MULTILINE), r"\1"),
    (re.compile(r"(^Disk:[^\n]*?GB total),?\s+[\d.]+\s*GB free", re.MULTILINE), r"\1"),
)


def strip_volatile_measurements(text: str) -> str:
    """ホスト計測ブロックから時間で変わる値だけを落とす (純粋関数)。

    計測ブロックでなければ入力をそのまま返す。
    """
    if not _HOST_MEASUREMENT_RE.search(text):
        return text
    for pattern, replacement in _VOLATILE_FIELD_SUBS:
        text = pattern.sub(replacement, text)
    return text


def episode_task_and_result(steps: list[dict[str, Any]]) -> tuple[str, str]:
    """エピソードの最終ステップから ``state.task`` と ``observation`` を取り出す。

    agent_tracer (meta_cognitive.py) は step に ``state.task`` (ユーザー要求) と
    ``observation`` (生成結果サマリ) を記録する。``action`` は tool 呼び出し名で、
    単発生成では ``"none"`` になる。decision/failure ファクトの object は ``action``
    だけでは無内容になるため、task/observation を併用して意味のある要約にする。
    episodic LTM ノート (mdp_ingester) でも同じ enrich を共有する。
    """
    if not steps:
        return "", ""
    last = steps[-1]
    state = last.get("state")
    task = ""
    if isinstance(state, dict):
        task = str(state.get("task") or "")
    observation = str(last.get("observation") or "")
    return task, observation


def compute_failure_signature(
    *,
    error_type: str,
    normalized_file_path: str,
    last_actions: list[str],
) -> str:
    """``(error_type, normalized_file_path, last_3_step_actions)`` から
    SHA1 先頭 12 桁の ``failure_signature`` を作る
    """
    payload = "|".join(
        [
            (error_type or "").strip().lower(),
            (normalized_file_path or "").strip().lower(),
            "/".join((a or "").strip() for a in last_actions[-3:]),
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


#: 常設ストアが出力する日付付きファイルにマッチするグロブパターン
AGENT_TRACE_GLOB = "agent_trace*.jsonl"


class MDPTraceExtractor(BaseExtractor):
    """``agent_trace*.jsonl`` から failure_pattern / decision を抽出する。

    ファイルの読み方 (ファイル別オフセット / 書きかけ行の繰り越し / 未終了
    エピソードの保留 / private・記憶読み出しのみの除外 / 処理済みの永続化) は
    Step 7.5 と同じ :class:`~backend.free.memory.notes.mdp_ingester.MDPIngester`
    に委ね、本クラスは **完了エピソード → SemanticFact** の変換だけを持つ。
    Step 7.5 とは別の state ファイル (``mdp_extract_state.json``) を使うので、
    embedder 不在で 7.5 が走らなくても 8 は独立に進む。以前は毎サイクル全
    ファイルを全文パースし、除外規則も別実装で食い違っていた。
    """

    mode = "create"

    def __init__(self, state_path: Path | None = None) -> None:
        self.state_path = Path(state_path) if state_path is not None else None
        self._ingester: "MDPIngester | None" = None
        self._ingester_dir: Path | None = None

    def _ingester_for(self, directory: Path) -> "MDPIngester":
        """``directory`` 用の読み手 (ディレクトリが変わったら作り直す)。"""
        from backend.free.memory.notes.mdp_ingester import MDPIngester

        if self._ingester is None or self._ingester_dir != directory:
            state = self.state_path or (directory / "mdp_extract_state.json")
            self._ingester = MDPIngester(
                directory, state, file_pattern=AGENT_TRACE_GLOB,
            )
            self._ingester_dir = directory
        return self._ingester

    def extract(
        self,
        notes: Iterable[MemoryNote],
        ctx: ExtractionContext,
    ) -> ExtractionResult:
        """``ctx.agent_trace_dir`` の未処理エピソードを SemanticFact 候補にする。

        ``notes`` は private トレース除外に使う (note.trace_id / note.private を
        集計し、行の ``trace_id`` と照合する。``begin.private`` の印は
        ``MDPIngester`` 側が見る)。
        """
        result = ExtractionResult()
        if not ctx.project_id:
            logger.debug("MDPTraceExtractor: no project_id, skipping")
            return result
        directory = ctx.agent_trace_dir
        if directory is None:
            logger.debug("MDPTraceExtractor: no agent_trace_dir configured")
            return result
        directory = Path(directory)
        if not directory.exists() or not directory.is_dir():
            logger.debug(
                "MDPTraceExtractor: agent_trace_dir not found: %s", directory,
            )
            return result

        private_trace_ids: set[str] = {
            n.trace_id for n in notes if n.private and n.trace_id
        }
        ingester = self._ingester_for(directory)
        processed_before = ingester.processed_count
        episodes = ingester.collect_episodes(private_trace_ids=private_trace_ids)
        result.notes_skipped = max(
            0, ingester.processed_count - processed_before - len(episodes),
        )

        scope = SemanticFact.make_project_scope(ctx.project_id)
        for record in episodes:
            result.episodes_seen += 1
            episode_id = record.episode_id
            steps = record.steps
            outcome = record.outcome()
            fact_trace_id = record.trace_id or episode_id
            last_actions = [str(s.get("action") or "") for s in steps[-3:]]
            task_desc, last_observation = episode_task_and_result(steps)
            # begin イベントの実行モード (chat/create) を fact の mode_origin へ
            # 反映する。クラス既定 "create" のままだとチャットセッション由来の
            # decision ファクトが create パーティションを汚染する (2026-07-15:
            # 21 件全件が mode_origin="create" で取り込まれた)。
            episode_mode = record.mode() or self.mode
            if not is_valid_session_mode(episode_mode):
                episode_mode = self.mode
            if _is_failure_outcome(outcome, steps):
                fact = self._build_failure_fact(
                    outcome=outcome,
                    steps=steps,
                    last_actions=last_actions,
                    scope=scope,
                    ctx=ctx,
                    trace_id=fact_trace_id,
                )
            else:
                fact = self._build_decision_fact(
                    episode_id=episode_id,
                    outcome=outcome,
                    last_actions=last_actions,
                    task_desc=task_desc,
                    observation=last_observation,
                    scope=scope,
                    ctx=ctx,
                    trace_id=fact_trace_id,
                )
            fact.mode_origin = episode_mode  # type: ignore[assignment]
            # provenance 側も同じモードへ揃える。``_make_fact`` はクラス既定
            # (``self.mode`` = "create") を入れるため、mode_origin だけ直しても
            # 由来の記録は create のまま残る (2026-08-19 ライブ監査:
            # decision ファクト 62/62 が mode_origin="chat" /
            # provenance.mode="create" で保存されていた)。provenance は経路を
            # 追うための唯一の記録なので、食い違うと調査が誤誘導される。
            for prov in fact.provenances:
                prov.mode = episode_mode
            result.facts.append(fact)

        logger.debug(
            "MDPTraceExtractor: episodes_seen=%d facts=%d skipped=%d",
            result.episodes_seen,
            len(result.facts),
            result.notes_skipped,
        )
        return result

    # ─── ファクト構築 ──────────────────────────────────────────────────

    def _build_failure_fact(
        self,
        *,
        trace_id: str,
        outcome: str,
        steps: list[dict[str, Any]],
        last_actions: list[str],
        scope: str,
        ctx: ExtractionContext,
    ) -> SemanticFact:
        # outcome 文字列から error_type / file path をベストエフォートで推定
        error_type = ""
        normalized_file_path = ""
        # 末尾ステップの observation から "Error: ... in path/to/file" を拾う
        last_obs = ""
        if steps:
            last_obs = str(steps[-1].get("observation") or "")
        candidate = (outcome + " " + last_obs).strip()
        if ":" in candidate:
            head, _, _rest = candidate.partition(":")
            error_type = head.strip().split()[-1] if head.strip() else ""
        # file path 抽出は雑にスペース区切りで `.py` 含むトークンを拾う
        for tok in candidate.split():
            if "." in tok and "/" in tok and len(tok) < 200:
                normalized_file_path = tok
                break

        signature = compute_failure_signature(
            error_type=error_type,
            normalized_file_path=normalized_file_path,
            last_actions=last_actions,
        )
        subject = f"{LOOP_FAILURE_PREFIX}{signature}"
        # object は loop.failure_note.FailurePayload と同じ JSON 形式で書く。
        # plain text だと Step 13 consolidation (LoopFactView) や
        # loop/report の集計が _safe_json_loads でパースできず occurrences /
        # outcomes_history を取りこぼす。同一 signature の再発は別ファクトとして
        # 作られ、Step 13 が occurrences を合算して 1 件に統合する。
        outcome_label = (outcome or "")[:200]
        object_payload = {
            "error_type": error_type,
            "normalized_file_path": normalized_file_path,
            "last_actions": list(last_actions)[-3:],
            "occurrences": 1,
            "outcomes_history": [outcome_label] if outcome_label else [],
        }
        object_text = json.dumps(
            object_payload, ensure_ascii=False, sort_keys=True,
        )
        return self.make_fact(
            subject=subject,
            predicate="failed_with",
            object_text=object_text,
            fact_type="failure_pattern",
            scope=scope,
            note=None,
            ctx=ctx,
            confidence=0.6,
            trace_id=trace_id,
            failure_signature=signature,
        )

    def _build_decision_fact(
        self,
        *,
        episode_id: str,
        trace_id: str,
        outcome: str,
        last_actions: list[str],
        scope: str,
        ctx: ExtractionContext,
        task_desc: str = "",
        observation: str = "",
    ) -> SemanticFact:
        # episode_id は ep_xxx 形式なので ``mem.<kind>.<parts>`` の parts 要件を満たす
        subject = f"{MEM_DECISION_PREFIX}{episode_id}"
        # object は task (要求) / result (生成結果) を主とし、意味のある action のみ併記。
        # action は単発生成で "none" になるため、それ単体だと無内容になる。
        parts = [f"outcome={outcome}"]
        if task_desc:
            parts.append(f"task={task_desc}")
        if observation:
            # 揮発する計測値 (空き RAM / CPU 使用率 / 空きディスク) は落とす。
            # SemMem 経由の注入は LTM とは独立した供給経路で、片方だけ塞いでも
            # もう一方から同じ古い値が「現在の値」として出てくる。
            parts.append(f"result={strip_volatile_measurements(observation)}")
        meaningful_actions = [a for a in last_actions if a and a != "none"]
        if meaningful_actions:
            parts.append(f"actions={'/'.join(meaningful_actions)}")
        object_text = self.truncate("; ".join(parts), self.MAX_OBJECT_LEN)
        return self.make_fact(
            subject=subject,
            predicate="resolved",
            object_text=object_text,
            fact_type="decision",
            scope=scope,
            note=None,
            ctx=ctx,
            confidence=0.6,
            trace_id=trace_id,
        )
