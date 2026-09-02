"""

``local/memory/agent_trace/agent_trace*.jsonl`` (``AgentTraceStore``) を
エピソード単位で読み出し、エピソード記憶 (LongTermMemory / 通常 RAG ベクトル
DB) に取り込むためのアダプタ。

設計の主旨

- **入力**: ``local_paths.agent_trace_dir`` 配下の ``agent_trace*.jsonl``
  (日付ベースのファイル名 ``agent_trace_YYYY-MM-DD.jsonl`` なのでグロブで
  横断する)。develop フラグには依存しない (以前は DebugLogger の evolve 限定
  JSONL を読んでいたため通常運用ではエピソード記憶が生成されなかった)。
- **粒度**: 1 エピソード = 1 ``MemoryNote``。``begin``/``step``/``end`` 3 種の
  イベントをグルーピングし、``end`` まで揃ったエピソードのみエピソード記憶に
  昇格させる。``end`` がまだ来ていないエピソードはステートに保留して次回呼び
  出しでマージする (``begin`` だけ先行して書き込まれた場合のロールフォワード)。
- **trace_id 連結**: agent_tracer 由来の各 JSONL 行には ``AgentTraceStore`` が
  ``contextvars`` の ``trace_id`` を自動付与する。本アダプタは行から
  ``trace_id`` を読み出し、``MemoryNote.trace_id`` にそのまま伝播させる。これに
  より B-1 の ``MDPTraceExtractor`` / Step 8 の Chat/Create extractor が同じ
  ``trace_id`` を ``SemanticFact.trace_id`` / ``Provenance.trace_id`` として
  受け継ぎ、ファクトと episodic LTM が連結可能になる。
- **private 防御**: ``private_trace_ids`` (現 STM の private ノートに紐づく
  trace_id 集合) に該当するエピソードはエピソード記憶昇格をスキップする。
  これにより ``private=True`` のトレースは scope=global / project どちらにも
  昇格しない
- **オフセット管理**: ファイル末尾までのバイトオフセットを
  ``local/memory/mdp_ingest_state.json`` に永続化する。次回呼び出しはオフセッ
  ト以降のみ読む。``processed_episode_ids`` も保持し、念のため二重昇格を防ぐ
  (上限 ``max_processed_ids`` で FIFO)。
- **副作用ゼロ (基本)**: 本クラスはストレージ書き込みを行わない。
  呼び出し側 (``SleepTimeWorker._step7_5_ingest_mdp_traces``) が
  ``MemoryNote`` を受け取り、埋め込み計算と
  ``LongTermMemory.absorb_from_short_term`` を実行する。これにより本クラスは
  embedder / vector store に依存せず、ユニットテストで完結する。

設計原則 (CLAUDE.md / .claude/rules/backend.md):

- フレームワーク非依存。ファイル I/O と JSON のみで完結
- LLM 呼び出しなし
- 50 行以内の関数 / ネスト 3 段以内
"""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.free.constants import READ_FILE_META_PREFIX
from backend.free.core.session_mode import normalize_session_mode
from backend.free.memory.extractors.mdp_trace import (
    episode_task_and_result,
    strip_volatile_measurements,
)
from backend.free.memory.stores.short_term import MemoryNote
from backend.io import atomic_write_text
from backend.log_config import get_logger

logger = get_logger("memory.mdp_ingester")


#: 記憶から読み出すだけのツール。これらだけで完結したエピソードはエピソード記憶へ
#: 昇格させない (``EpisodeRecord.is_memory_read_only`` 参照)。外界を観測する
#: ツール (read_file / run_command_readonly / search_code / fetch_url 等) は
#: 記憶する価値があるので含めない。
_MEMORY_READ_ACTIONS: frozenset[str] = frozenset({"search_history"})

#: エピソード記憶に載せる観測値の上限。
_OBSERVATION_MAX_CHARS = 300


def _summarize_observation(observation: str) -> str:
    """観測値をエピソード記憶用に要約する (純粋関数)。

    エピソード記憶が記録すべきは「**何をしたか**」であって、ツールが返した
    ペイロードそのものではない。``read_file`` の結果は先頭にメタ行
    ``[file: <path> | lines: N | chars: M]`` が付き、その後ろにファイル本文が
    続く。本文まで載せると:

    - 別セッションで読んだファイルの中身が ``[参考情報]`` として提示され、
      現在の話題と無関係な内容へ引きずられる
    - 同じ内容が LTM に重複して溜まり、注入予算を食う

    実データ (2026-08-16 再測定): ``result=[file: E:\\...\\README.md | lines: 121
    | chars: 3331 | showing lines 1-5] # evoref — 自己進化型ローカル LLM ア...``
    のように本文が貼られ、``[参考情報]`` のトークンを占有していた。

    メタ行だけ残せば「そのファイルを読んだ / 何行何文字だった」は保てる。
    メタ行を持たない観測値は従来どおり先頭 300 文字で切る。
    """
    text = observation.strip()
    if text.startswith(READ_FILE_META_PREFIX):
        end = text.find("]")
        if end != -1:
            return text[: end + 1]
    return strip_volatile_measurements(text)[:_OBSERVATION_MAX_CHARS]


# ──────────────────────────────────────────────────────────────────────────
# データクラス
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class EpisodeRecord:
    """1 エピソード分の生イベント集合。

    ``begin`` イベントが無いまま ``step``/``end`` だけが届いた場合は
    ``begin is None`` のままになる (debug_logger の書き出しが途中で
    クラッシュしたケース等を想定)。
    """

    episode_id: str
    begin: dict[str, Any] | None = None
    steps: list[dict[str, Any]] = field(default_factory=list)
    end: dict[str, Any] | None = None
    trace_id: str | None = None
    """``trace_id`` は最初に観測されたイベントから引き継ぐ"""

    def is_complete(self) -> bool:
        return self.end is not None

    def actions(self) -> list[str]:
        """ステップの ``action`` (ツール名) を順に返す。空 / ``none`` は除く。"""
        out: list[str] = []
        for step in self.steps:
            action = str(step.get("action") or "")
            if action and action != "none":
                out.append(action)
        return out

    def is_memory_read_only(self) -> bool:
        """記憶の読み出しだけで完結したエピソードか。

        ``search_history`` は**記憶から読み出した結果を整形して返すツール**で、
        その出力は新しい観測ではない。これをエピソード記憶へ書き戻すと、
        注入用に組み立てたテキストが次のセッションで「参考情報」として蘇る
        循環になる。

        実害 (2026-08-16 再測定): 位置指定の自己参照回答 (PR #435) の出力
        ``[以下は**今回の会話**の記録です] [この会話の最初の user 発話 3 件…]``
        が LTM に取り込まれていた。**「今回の会話」というラベルごと別セッションへ
        持ち越される**ため、読んだモデルは他人の会話を自分の会話として帰属する。

        外界を観測するツール (read_file / run_command_readonly / search_code /
        fetch_url など) は記憶する価値があるので対象外。``search_history`` と
        実際の操作が混在するエピソードも、操作の側に記録価値があるので残す。
        """
        actions = self.actions()
        return bool(actions) and set(actions) <= _MEMORY_READ_ACTIONS

    def conversation_id(self) -> str | None:
        if self.begin and isinstance(self.begin, dict):
            cid = self.begin.get("conversation_id")
            if isinstance(cid, str) and cid:
                return cid
        return None

    def mode(self) -> str:
        if self.begin and isinstance(self.begin, dict):
            m = self.begin.get("mode")
            if isinstance(m, str) and m:
                return m
        return "create"

    def outcome(self) -> str:
        if self.end and isinstance(self.end, dict):
            return str(self.end.get("outcome") or "")
        return ""

    def is_private(self) -> bool:
        """begin イベントに private フラグが刻まれているか。

        ``private_trace_ids`` (STM の private ノート由来) は、Full が走る前に
        当該ノートが押し出されていると空になる。begin イベント自体の印は
        ノートの寿命に依らないので、こちらを一次情報として併用する。
        """
        return bool(
            isinstance(self.begin, dict) and self.begin.get("private"),
        )

    def absorb_event(self, obj: dict[str, Any]) -> None:
        """1 行ぶんのイベントを取り込む。"""
        ev = obj.get("event")
        # trace_id は最初に観測したものを採用 (後続イベントのほうが空でも上書きしない)
        if not self.trace_id:
            tid = obj.get("trace_id")
            if isinstance(tid, str) and tid:
                self.trace_id = tid
        if ev == "begin":
            self.begin = obj
        elif ev == "step":
            self.steps.append(obj)
        elif ev == "end":
            self.end = obj


# ──────────────────────────────────────────────────────────────────────────
# 状態ファイル
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class IngestState:
    """``mdp_ingest_state.json`` のメモリ表現。"""

    file_offsets: dict[str, int] = field(default_factory=dict)
    processed_episode_ids: list[str] = field(default_factory=list)
    pending_episodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    """未終了エピソードの保留 (episode_id → serialized EpisodeRecord)"""

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_offsets": dict(self.file_offsets),
            "processed_episode_ids": list(self.processed_episode_ids),
            "pending_episodes": dict(self.pending_episodes),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> IngestState:
        return cls(
            file_offsets={
                str(k): int(v) for k, v in (d.get("file_offsets") or {}).items()
            },
            processed_episode_ids=list(d.get("processed_episode_ids") or []),
            pending_episodes=dict(d.get("pending_episodes") or {}),
        )


def _serialize_episode(ep: EpisodeRecord) -> dict[str, Any]:
    return {
        "episode_id": ep.episode_id,
        "begin": ep.begin,
        "steps": ep.steps,
        "end": ep.end,
        "trace_id": ep.trace_id,
    }


def _deserialize_episode(d: dict[str, Any]) -> EpisodeRecord:
    return EpisodeRecord(
        episode_id=str(d.get("episode_id") or ""),
        begin=d.get("begin"),
        steps=list(d.get("steps") or []),
        end=d.get("end"),
        trace_id=d.get("trace_id"),
    )


# ──────────────────────────────────────────────────────────────────────────
# Ingester 本体
# ──────────────────────────────────────────────────────────────────────────


class MDPIngester:
    """``agent_trace*.jsonl`` をエピソード単位で episodic LTM 用 ``MemoryNote``
    に変換するアダプタ。

    Args:
        log_dir: ``AgentTraceStore`` の常設ディレクトリ (``local_paths.agent_trace_dir``)。
            ``None`` の場合 ``collect_episodes`` は常に空リストを返す。
        state_path: オフセット永続化先 (``local/memory/mdp_ingest_state.json``)
        file_pattern: グロブパターン。デフォルトで日付ファイルを含む
        max_processed_ids: ``processed_episode_ids`` を FIFO 上限で切り詰める
        max_pending_episodes: 保留エピソードの上限 (それ以上は古い順に廃棄)
    """

    def __init__(
        self,
        log_dir: Path | None,
        state_path: Path | None,
        *,
        file_pattern: str = "agent_trace*.jsonl",
        max_processed_ids: int = 10000,
        max_pending_episodes: int = 1000,
    ) -> None:
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.state_path = Path(state_path) if state_path is not None else None
        self.file_pattern = file_pattern
        self.max_processed_ids = max_processed_ids
        self.max_pending_episodes = max_pending_episodes
        self._state = self._load_state()

    @property
    def processed_count(self) -> int:
        """処理済み (昇格 / 除外を問わず読み終えた) episode_id の数。"""
        return len(self._state.processed_episode_ids)

    # ─── state I/O ────────────────────────────────────────────────────────

    def _load_state(self) -> IngestState:
        if self.state_path is None or not self.state_path.exists():
            return IngestState()
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return IngestState.from_dict(data)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("MDPIngester: failed to load state %s: %s", self.state_path, exc)
        return IngestState()

    def _save_state(self) -> None:
        if self.state_path is None:
            return
        try:
            atomic_write_text(
                self.state_path,
                json.dumps(self._state.to_dict(), ensure_ascii=False, indent=2),
            )
        except OSError as exc:
            logger.warning("MDPIngester: failed to save state %s: %s", self.state_path, exc)

    # ─── 公開 API ─────────────────────────────────────────────────────────

    def collect_episodes(
        self,
        *,
        private_trace_ids: Iterable[str] | None = None,
    ) -> list[EpisodeRecord]:
        """新規バイトを読み出し、終了済みエピソードを返す。

        - ``private_trace_ids`` に該当するエピソードは ``processed_episode_ids``
          に登録した上で破棄する (再昇格防止 + 後続イベントの捨て読み)。
        - ``processed_episode_ids`` に既登録のエピソードはスキップする。
        - 未終了エピソードは ``pending_episodes`` に保留する。
        """
        if self.log_dir is None or not self.log_dir.exists():
            return []

        private_set: set[str] = {t for t in (private_trace_ids or []) if t}
        already_processed = set(self._state.processed_episode_ids)
        pending = self._restore_pending()
        new_events = self._read_new_events()
        for obj in new_events:
            self._merge_event(obj, pending, already_processed)

        completed: list[EpisodeRecord] = []
        for ep_id in list(pending.keys()):
            ep = pending[ep_id]
            if ep.is_complete():
                pending.pop(ep_id, None)
                if ep_id in already_processed:
                    continue
                if ep.is_private() or (ep.trace_id and ep.trace_id in private_set):
                    self._mark_processed(ep_id, already_processed)
                    continue
                if ep.is_memory_read_only():
                    logger.debug(
                        "MDPIngester: skipping memory-read episode %s "
                        "(actions=%s)", ep_id, ep.actions(),
                    )
                    self._mark_processed(ep_id, already_processed)
                    continue
                completed.append(ep)
                self._mark_processed(ep_id, already_processed)

        self._enforce_pending_cap(pending)
        self._state.pending_episodes = {
            k: _serialize_episode(v) for k, v in pending.items()
        }
        self._save_state()
        if completed:
            logger.info(
                "MDPIngester: collected %d episodes from %s",
                len(completed), self.log_dir,
            )
        return completed

    def to_memory_note(
        self,
        episode: EpisodeRecord,
        *,
        project_id: str | None,
        now: float | None = None,
    ) -> MemoryNote:
        """``EpisodeRecord`` を ``MemoryNote`` に変換する。

        - ``id`` は ``mdp_<episode_id>`` (重複した場合は LTM 側で chunk_id が
          一意化されるためここでは episode 毎に固定)
        - ``mode`` は begin event の値 (デフォルト create)
        - ``trace_id`` はイベントから引き継ぎ (contextvars 由来)
        - ``content`` は ``outcome`` + task/observation + 直近 3 アクションの
          1 行サマリ (ツール無しの単発生成でも無内容にならないよう enrich)
        """
        ts = now if now is not None else time.time()
        steps = episode.steps
        last_actions = [str(s.get("action") or "") for s in steps[-3:]]
        outcome = episode.outcome() or "unknown"
        # ツール無しの単発生成では action が "none" になり、outcome + actions だけ
        # では無内容なノート (`actions=none`) が LTM/RAG を汚染する。decision ファクト
        # と同じ enrich を共有し、最終ステップの task (ユーザー要求) と
        # observation (生成結果) を content に含める。
        task_desc, observation = episode_task_and_result(steps)
        content_parts = [
            f"[mdp_trace] episode={episode.episode_id}",
            f"outcome={outcome}",
        ]
        if task_desc:
            content_parts.append(f"task={task_desc[:300]}")
        if observation:
            content_parts.append(f"result={_summarize_observation(observation)}")
        if last_actions:
            content_parts.append("actions=" + " > ".join(last_actions))
        if conv := episode.conversation_id():
            content_parts.append(f"conversation={conv}")
        content = "; ".join(content_parts)

        note = MemoryNote(
            id=f"mdp_{episode.episode_id}",
            content=content,
            # "none" (ツール無し単発生成のプレースホルダ action) は検索語として
            # 無意味なので keywords から除く。
            keywords=[a for a in last_actions if a and a != "none"][:5],
            tags=["mdp_trace"],
            created_at=ts,
            accessed_at=ts,
            session_id=episode.episode_id,
            source="system",
            confidence=0.6,
            mode=normalize_session_mode(episode.mode(), default="create"),  # type: ignore[arg-type]
            project_id=project_id,
            trace_id=episode.trace_id,
        )
        return note

    # ─── 内部ヘルパ ────────────────────────────────────────────────────

    def _restore_pending(self) -> dict[str, EpisodeRecord]:
        return {
            k: _deserialize_episode(v)
            for k, v in (self._state.pending_episodes or {}).items()
        }

    def _read_new_events(self) -> list[dict[str, Any]]:
        """設定された log_dir の各ファイルを offset 以降だけ読む。

        ファイルが消滅・再生成された (offset > size) 場合はオフセットをリセット
        して全文を読み直す。
        """
        events: list[dict[str, Any]] = []
        if self.log_dir is None:
            return events
        for path in sorted(self.log_dir.glob(self.file_pattern)):
            try:
                size = path.stat().st_size
            except OSError:
                continue
            offset = self._state.file_offsets.get(path.name, 0)
            if offset > size:
                offset = 0
            if offset == size:
                continue
            try:
                with path.open("rb") as f:
                    f.seek(offset)
                    raw = f.read(size - offset)
                # 書きかけの末尾行 (改行なし) は次回に回す。offset を size まで
                # 進めると、その行は warning 1 回で恒久に失われる。
                cut = raw.rfind(b"\n")
                if cut < 0:
                    continue
                raw = raw[: cut + 1]
                self._state.file_offsets[path.name] = offset + len(raw)
            except OSError as exc:
                logger.warning("MDPIngester: failed to read %s: %s", path, exc)
                continue
            events.extend(self._parse_lines(raw))
        return events

    @staticmethod
    def _parse_lines(raw: bytes) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for line in raw.splitlines():
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as exc:
                logger.warning("MDPIngester: malformed jsonl line: %s", exc)
                continue
            if isinstance(obj, dict):
                out.append(obj)
        return out

    @staticmethod
    def _merge_event(
        obj: dict[str, Any],
        pending: dict[str, EpisodeRecord],
        already_processed: set[str],
    ) -> None:
        episode_id = obj.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            return
        if episode_id in already_processed:
            return
        ep = pending.get(episode_id)
        if ep is None:
            ep = EpisodeRecord(episode_id=episode_id)
            pending[episode_id] = ep
        ep.absorb_event(obj)

    def _mark_processed(self, episode_id: str, already_processed: set[str]) -> None:
        if episode_id in already_processed:
            return
        already_processed.add(episode_id)
        self._state.processed_episode_ids.append(episode_id)
        if len(self._state.processed_episode_ids) > self.max_processed_ids:
            drop = len(self._state.processed_episode_ids) - self.max_processed_ids
            self._state.processed_episode_ids = self._state.processed_episode_ids[drop:]

    def _enforce_pending_cap(self, pending: dict[str, EpisodeRecord]) -> None:
        if len(pending) <= self.max_pending_episodes:
            return
        # FIFO で古い順に切り捨て (Python 3.7+ dict は挿入順)
        excess = len(pending) - self.max_pending_episodes
        ordered = OrderedDict(pending)
        for _ in range(excess):
            try:
                ordered.popitem(last=False)
            except KeyError:
                break
        pending.clear()
        pending.update(ordered)
