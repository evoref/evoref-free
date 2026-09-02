"""

``ChatExtractor`` / ``CreateExtractor`` / ``MDPTraceExtractor`` 共通の
データクラスとヘルパを提供する。

設計の核となる考え方:

- **入力**: ``ShortTermMemory`` の ``MemoryNote`` 群 (chat/create 抽出器) または
  ``agent_trace*.jsonl`` ファイル群 (MDPTraceExtractor、日付付きファイル含む)。
- **出力**: ``SemanticFact`` のリスト + 統計 (``ExtractionResult``)。
  ``ExtractionResult`` には skip 件数や cap 当たり件数も含め、
  ``SleepTimeWorker._step8_extract_facts`` がログに残せるようにする。
- **副作用**: 抽出した fact の ID を ``MemoryNote.extracted_fact_ids`` へ書き戻し、
  次回の sleep-time で同じノートから二重抽出されないようにする。
- **永続化**: ``SemanticFactStore`` への書き込みは呼び出し側 (Step 8 メソッド)
  が行い、本クラスはストアに依存しない (ユニットテスト容易性のため)。
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.free.memory.stores.short_term import MemoryNote
from backend.free.memory.notes.subject_canonicalizer import SubjectCanonicalizer
from backend.free.memory.types import (
    FactType,
    MemoryMode,
    Provenance,
    SemanticFact,
)
from backend.log_config import get_logger

logger = get_logger("memory.extractors.base")


# ──────────────────────────────────────────────────────────────────────────
# 入出力データクラス
# ──────────────────────────────────────────────────────────────────────────


def _utterance_time(note: object, fallback: float) -> float:
    """ノートの発話時刻 (無ければ ``fallback``)。

    :meth:`BaseExtractor.make_fact` が ``created_at`` に使う。抽出時刻を
    入れると 1 バッチ分のファクトが全部同じ秒になり、世代の前後が
    ストアから失われる (詳細は呼出側のコメント)。
    """
    created_at = float(getattr(note, "created_at", 0.0) or 0.0)
    return created_at if created_at > 0 else fallback


@dataclass
class ExtractionContext:
    """Extractor 共通のコンテキスト。

    Attributes:
        project_id: クリエイトモードのプロジェクト ID。``None`` の場合
            CreateExtractor / MDPTraceExtractor は no-op (project スコープ
            必須のため)。
        agent_trace_dir: ``agent_trace*.jsonl`` を格納するディレクトリ
            (通常は ``debug_logger.log_dir``)。``MDPTraceExtractor`` は
            このディレクトリ配下の日付付きファイル
            (``agent_trace_YYYY-MM-DD.jsonl``) をグロブで横断する。
            ``None`` または存在しない場合は no-op。
        max_per_session: モード別セッション上限
            ``{"chat": 10, "create": 5}`` を期待
        max_pinned_per_session: pinned ノート由来の上限。``-1`` で無制限
        canonicalizer: subject 正規化器 (``None`` ならバイパスのみ)
        now: テスト容易性のための時刻注入
    """

    project_id: str | None = None
    agent_trace_dir: Path | None = None
    max_per_session: dict[str, int] = field(
        default_factory=lambda: {"chat": 10, "create": 5},
    )
    max_pinned_per_session: int = -1
    canonicalizer: SubjectCanonicalizer | None = None
    now: float | None = None
    #: ``{(fact_type, 属性スロット): (現在値, ...)}``。属性語を落とした訂正の
    #: 宛先を「既存スロットの現在値を名指しているか」で決めるために使う
    #: (:func:`~backend.free.memory.extractors.chat.
    #: resolve_value_anchored_attributes`)。空なら値アンカーは働かず、
    #: 従来どおり属性語と継承だけで解決する。
    live_attribute_values: dict[tuple[str, str], tuple[str, ...]] = field(
        default_factory=dict,
    )

    def current_time(self) -> float:
        return self.now if self.now is not None else time.time()


@dataclass
class ExtractionResult:
    """Extractor の実行結果。

    Attributes:
        facts: 生成された ``SemanticFact`` のリスト (永続化前)
        notes_processed: 走査した eligible ノート数
        notes_skipped: スキップしたノート数 (private/code_block 等)
        cap_dropped: 上限超過で破棄した候補数
        already_extracted: ``extracted_fact_ids`` で既処理だったノート数
        episodes_seen: ``MDPTraceExtractor`` で走査したエピソード数
    """

    facts: list[SemanticFact] = field(default_factory=list)
    notes_processed: int = 0
    notes_skipped: int = 0
    cap_dropped: int = 0
    already_extracted: int = 0
    episodes_seen: int = 0


# ──────────────────────────────────────────────────────────────────────────
# 共通基底
# ──────────────────────────────────────────────────────────────────────────


class BaseExtractor:
    """Extractor 共通基底。

    サブクラスは ``mode`` (``chat`` / ``create``) と ``extract`` を実装する。
    ``MDPTraceExtractor`` のように STM を入力に取らない抽出器は ``extract``
    を完全にオーバーライドする。
    """

    mode: MemoryMode = "chat"

    #: subject の最大長 (object と区別するため短めに)
    MAX_SUBJECT_LEN: int = 64

    #: object テキストの最大長 (Tier 注入時の予算節約のため)
    MAX_OBJECT_LEN: int = 280

    def extract(
        self,
        notes: Iterable[MemoryNote],  # noqa: ARG002
        ctx: ExtractionContext,  # noqa: ARG002
    ) -> ExtractionResult:
        """ノート列から SemanticFact 候補を抽出する。

        サブクラスでオーバーライドする。基底実装は何も返さない。
        """
        return ExtractionResult()

    # ─── ノートフィルタ ─────────────────────────────────────────────────

    @classmethod
    def is_eligible(cls, note: MemoryNote, mode: MemoryMode) -> bool:
        """Step 8 の抽出対象としてノートが適格か判定する。

        除外条件:

        - ``private=True``
        - ``is_code_block=True`` (コードブロックは完全スキップ)
        - ``is_tool_output=True`` (ツール出力は STM 以降に来ない想定だが
          念のため二重ガード)
        - ``extraction_skipped=True``
        - モード不一致 (chat extractor が create ノートを取らないなど)
        - ``content`` が空
        """
        if note.private:
            return False
        if note.is_code_block:
            return False
        if note.is_tool_output:
            return False
        if note.extraction_skipped:
            return False
        if (note.mode or "chat") != mode:
            return False
        if not (note.content or "").strip():
            return False
        return True

    # ─── ヘルパ ────────────────────────────────────────────────────────

    @classmethod
    def truncate(cls, text: str, max_len: int) -> str:
        """テキストを ``max_len`` で切り詰める。改行 1 個に正規化"""
        normalized = " ".join((text or "").split())
        if len(normalized) <= max_len:
            return normalized
        return normalized[: max_len - 1] + "…"

    def make_fact(
        self,
        *,
        subject: str,
        predicate: str,
        object_text: str,
        fact_type: FactType,
        scope: str,
        note: MemoryNote | None,
        ctx: ExtractionContext,
        confidence: float = 0.5,
        trace_id: str | None = None,
        **overrides: Any,
    ) -> SemanticFact:
        """共通フィールドを埋めた ``SemanticFact`` を作る。

        - subject は ``ctx.canonicalizer`` で正規化 (バイパスは尊重)
        - provenance を 1 件付与 (note 由来 or trace 由来)
        - pinned はノートの ``pin_flag`` を継承
        - ``from_correction`` はノートの ``is_correction`` を継承
        - confidence はデフォルト 0.5
        """
        canonical = subject.strip()
        if ctx.canonicalizer is not None:
            canonical = ctx.canonicalizer(canonical)
        canonical = self.truncate(canonical, self.MAX_SUBJECT_LEN)
        clipped_obj = self.truncate(object_text, self.MAX_OBJECT_LEN)
        now = ctx.current_time()

        # ノート由来の trace_id を最優先で fact / provenance に伝播する
        #。明示的に渡された trace_id (MDPTraceExtractor の
        # episode_id) はそれを優先する。
        effective_trace_id = trace_id or getattr(note, "trace_id", None)
        prov = Provenance(
            note_id=getattr(note, "id", None),
            session_id=getattr(note, "session_id", None) or None,
            trace_id=effective_trace_id,
            mode=getattr(note, "mode", self.mode) or self.mode,
            project_id=getattr(note, "project_id", None) or ctx.project_id,
            source=getattr(note, "source", None),
            captured_at=now,
        )

        fact = SemanticFact(
            id=SemanticFact.new_id(),
            subject=canonical or "unknown",
            predicate=predicate,
            object=clipped_obj,
            type=fact_type,
            scope=scope,
            mode_origin=self.mode,
            provenances=[prov],
            confidence=confidence,
            pinned=bool(getattr(note, "pin_flag", False)),
            # 訂正ターン由来か。ノートから引き継ぎ、競合解決が「同一セッション
            # だから微妙ケース」として pending へ落とすのを免除する
            # (SemanticFact.from_correction の説明を参照)。
            from_correction=bool(getattr(note, "is_correction", False)),
            # **発話時刻**を継ぐ (抽出時刻ではない)。sleep-time は 1 回の
            # バッチで会話全体を抽出するため ``now`` を入れると全ファクトの
            # ``created_at`` が同一秒になり、「新しい方を採る」判定が原理的に
            # 成立しない。実データ (2026-08-27 ライブ監査) では 12 件すべてが
            # ``1787814691.47〜.48`` に潰れており、訂正と初出の前後関係が
            # ストアから失われていた。``(N日前の記録)`` ラベルの根拠でもある。
            created_at=_utterance_time(note, now),
            accessed_at=now,
            session_ids=(
                {getattr(note, "session_id", "")} if note and note.session_id else set()
            ),
            private=False,  # private ノートはここに来ない
            trace_id=effective_trace_id,
        )
        for key, value in overrides.items():
            setattr(fact, key, value)
        return fact

    # ─── セッション別キャップ ──────────────────────────────────────────

    def apply_session_caps(
        self,
        candidates: list[tuple[MemoryNote, SemanticFact]],
        ctx: ExtractionContext,
    ) -> tuple[list[tuple[MemoryNote, SemanticFact]], int]:
        """セッションごとにモード別上限を適用する。

        - ``ctx.max_per_session[self.mode]`` (例: chat=10) を超えた候補は破棄
        - ``pin_flag=True`` のノート由来候補は pinned 別カウントで管理
        - ``ctx.max_pinned_per_session = -1`` (デフォルト) なら pinned は無制限

        副作用として、候補を出したノートの ``extraction_deferred`` を更新する:
        1 件も採用されなかったノートは ``True`` (次サイクルへ見送り。eviction
        保護の対象になる)、採用されたノートは ``False``。

        Returns:
            (採用された (note, fact) のリスト, 破棄された候補数)
        """
        kept, dropped = self._apply_session_caps(candidates, ctx)
        kept_notes = {id(note) for note, _ in kept if note is not None}
        for note, _ in candidates:
            if note is not None:
                note.extraction_deferred = id(note) not in kept_notes
        return kept, dropped

    def _apply_session_caps(
        self,
        candidates: list[tuple[MemoryNote, SemanticFact]],
        ctx: ExtractionContext,
    ) -> tuple[list[tuple[MemoryNote, SemanticFact]], int]:
        """:meth:`apply_session_caps` の本体 (副作用なし)。"""
        cap = ctx.max_per_session.get(self.mode, 10)
        pinned_cap = ctx.max_pinned_per_session
        per_session_count: dict[str, int] = {}
        per_session_pinned: dict[str, int] = {}
        kept: list[tuple[MemoryNote, SemanticFact]] = []
        dropped = 0
        for note, fact in candidates:
            sid = (note.session_id or "_no_session_") if note else "_no_session_"
            if note and note.pin_flag:
                if pinned_cap < 0:
                    kept.append((note, fact))
                    per_session_pinned[sid] = per_session_pinned.get(sid, 0) + 1
                    continue
                if per_session_pinned.get(sid, 0) >= pinned_cap:
                    dropped += 1
                    continue
                per_session_pinned[sid] = per_session_pinned.get(sid, 0) + 1
                kept.append((note, fact))
                continue

            if per_session_count.get(sid, 0) >= cap:
                dropped += 1
                continue
            per_session_count[sid] = per_session_count.get(sid, 0) + 1
            kept.append((note, fact))
        return kept, dropped
