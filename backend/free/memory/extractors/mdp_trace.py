"""

``agent_trace*.jsonl`` (日付付きファイル含む) をエピソード単位でグルー
ピングし、終了 (``end``) イベントから SemanticFact 候補を生成する。
``DebugLogger`` は ``agent_trace_YYYY-MM-DD.jsonl`` という日付付きファイル
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
- 既処理エピソードを ``self._processed_episode_ids`` でメモ化することで
  二重抽出を防ぐ。プロセス再起動後は (まれに) 再抽出されるが add_fact 側で
  fact_id は新発番されるため idempotent ではない点に留意 (
  failure_pattern 統合の再排除を行う想定)。
- ``private`` トレースのフィルタは入力側 (Step 8 メソッド) で行う。

ファイル形式 (``debug_logger.log_agent_trace_event`` 由来)::

    {"event": "begin", "episode_id": "ep_xxx", "conversation_id": "...", "mode": "coding", ...}
    {"event": "step", "episode_id": "ep_xxx", "step_index": 0, "action": "...", ...}
    {"event": "end", "episode_id": "ep_xxx", "outcome": "success" | "failure: ...", ...}
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from backend.free.memory.extractors.base import (
    BaseExtractor,
    ExtractionContext,
    ExtractionResult,
)
from backend.free.memory.stores.short_term import MemoryNote
from backend.free.memory.types import SemanticFact
from backend.log_config import get_logger

logger = get_logger("memory.extractors.mdp_trace")


LOOP_FAILURE_PREFIX = "loop.failure."
"""failure_pattern ファクトの subject prefix (owner は EvorefLoop)"""

MEM_DECISION_PREFIX = "mem.decision."
"""decision ファクトの subject prefix (owner は EvorefMem)"""

_FAILURE_KEYWORDS = ("fail", "error", "exception", "abort")


def _is_failure_outcome(outcome: str) -> bool:
    if not outcome:
        return False
    s = outcome.lower()
    return any(k in s for k in _FAILURE_KEYWORDS)


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


#: ``DebugLogger`` が出力する日付付きファイルにマッチするグロブパターン
AGENT_TRACE_GLOB = "agent_trace*.jsonl"


class MDPTraceExtractor(BaseExtractor):
    """``agent_trace*.jsonl`` から failure_pattern / decision を抽出する。"""

    mode = "coding"

    def __init__(self) -> None:
        # プロセス内の二重抽出防止用エピソード集合 (再起動でリセット可)
        self._processed_episode_ids: set[str] = set()

    # 単体テスト用フック (オーバーライド可能)
    def _open_lines(self, path: Path) -> Iterable[str]:
        with path.open("r", encoding="utf-8") as f:
            yield from f

    def _iter_trace_files(self, directory: Path) -> list[Path]:
        """``directory`` 配下の agent_trace*.jsonl をソート済みリストで返す。

        日付付きファイル名 (``agent_trace_YYYY-MM-DD.jsonl``) を辞書順で
        ソートすると時系列順になる。エピソードが日付をまたいでも
        ``begin``/``end`` を episode_id でグルーピングするので順序非依存だが、
        ロギングの再現性のためソートしておく。
        """
        return sorted(directory.glob(AGENT_TRACE_GLOB))

    def extract(
        self,
        notes: Iterable[MemoryNote],
        ctx: ExtractionContext,
    ) -> ExtractionResult:
        """STM ノートではなく ``ctx.agent_trace_dir`` 配下の
        ``agent_trace*.jsonl`` をすべて読む。

        ``notes`` 引数は ``private`` トレース除外に使う (note.trace_id /
        note.private を集計)。STM に対応する private トレースがあれば、
        その episode は完全にスキップする。
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
        trace_files = self._iter_trace_files(directory)
        if not trace_files:
            logger.debug(
                "MDPTraceExtractor: no agent_trace*.jsonl in %s", directory,
            )
            return result

        private_trace_ids: set[str] = {
            n.trace_id for n in notes if n.private and n.trace_id
        }

        episodes: dict[str, dict[str, Any]] = {}
        for path in trace_files:
            for raw in self._open_lines(path):
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "MDPTraceExtractor: malformed jsonl line in %s: %s",
                        path, exc,
                    )
                    continue
                episode_id = obj.get("episode_id")
                if not episode_id:
                    continue
                ep = episodes.setdefault(
                    episode_id,
                    {"steps": [], "begin": None, "end": None},
                )
                event = obj.get("event")
                if event == "begin":
                    ep["begin"] = obj
                elif event == "step":
                    ep["steps"].append(obj)
                elif event == "end":
                    ep["end"] = obj

        scope = SemanticFact.make_project_scope(ctx.project_id)
        for episode_id, ep in episodes.items():
            if episode_id in self._processed_episode_ids:
                continue
            if episode_id in private_trace_ids:
                self._processed_episode_ids.add(episode_id)
                result.notes_skipped += 1
                continue
            end = ep.get("end")
            if end is None:
                # 未終了エピソードは次回に持ち越し
                continue
            result.episodes_seen += 1
            outcome = str(end.get("outcome") or "")
            steps = ep["steps"]
            last_actions = [str(s.get("action") or "") for s in steps[-3:]]
            if _is_failure_outcome(outcome):
                fact = self._build_failure_fact(
                    episode_id=episode_id,
                    outcome=outcome,
                    steps=steps,
                    last_actions=last_actions,
                    scope=scope,
                    ctx=ctx,
                )
            else:
                fact = self._build_decision_fact(
                    episode_id=episode_id,
                    outcome=outcome,
                    last_actions=last_actions,
                    scope=scope,
                    ctx=ctx,
                )
            result.facts.append(fact)
            self._processed_episode_ids.add(episode_id)

        logger.debug(
            "MDPTraceExtractor: episodes_seen=%d facts=%d skipped_private=%d",
            result.episodes_seen,
            len(result.facts),
            result.notes_skipped,
        )
        return result

    # ─── ファクト構築 ──────────────────────────────────────────────────

    def _build_failure_fact(
        self,
        *,
        episode_id: str,
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
        object_text = self.truncate(
            f"outcome={outcome}; last_actions={'/'.join(last_actions)}",
            self.MAX_OBJECT_LEN,
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
            trace_id=episode_id,
            failure_signature=signature,
        )

    def _build_decision_fact(
        self,
        *,
        episode_id: str,
        outcome: str,
        last_actions: list[str],
        scope: str,
        ctx: ExtractionContext,
    ) -> SemanticFact:
        # episode_id は ep_xxx 形式なので ``mem.<kind>.<parts>`` の parts 要件を満たす
        subject = f"{MEM_DECISION_PREFIX}{episode_id}"
        object_text = self.truncate(
            f"outcome={outcome}; last_actions={'/'.join(last_actions)}",
            self.MAX_OBJECT_LEN,
        )
        return self.make_fact(
            subject=subject,
            predicate="resolved",
            object_text=object_text,
            fact_type="decision",
            scope=scope,
            note=None,
            ctx=ctx,
            confidence=0.6,
            trace_id=episode_id,
        )
