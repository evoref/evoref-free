"""

``ChatExtractor`` / ``CreateExtractor`` / ``MDPTraceExtractor`` 共通の
データクラスとヘルパを提供する。

設計の核となる考え方:

- **入力**: エピソード記憶の ``MemoryNote`` 群 (chat/create 抽出器) または
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
from datetime import datetime, timezone
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.free.memory.episodic.note import MemoryNote
from backend.free.memory.notes.subject_canonicalizer import SubjectCanonicalizer
from backend.free.memory.note_facts import origin_of
from backend.free.memory.types import (
    FactType,
    MemoryMode,
    Provenance,
    SemanticFact,
)
from backend.free.core.correction_verdict import (
    POINTING_TARGETS as _POINTING_TARGETS,
)
from backend.free.core.relative_date import annotate_relative_dates
from backend.free.core.text_quality import detect_lang
from backend.log_config import get_logger

logger = get_logger("memory.extractors.base")


# ──────────────────────────────────────────────────────────────────────────
# 入出力データクラス
# ──────────────────────────────────────────────────────────────────────────


def note_is_verified_correction(note: object) -> bool:
    """ノートが **検証済みの訂正** か (純粋関数)。

    ``MemoryNote.is_correction`` は字句で立てた **候補** に過ぎない。訂正の力
    (``from_correction`` による時刻を跨いだ supersede / 値アンカーでのスロット
    決定 / 直前スロットの継承) を持たせてよいのは、
    ``sleep.correction_curator`` (Step 8.0) が「過去の発言の誤りを指している」
    と判定した (``correction_verdict`` が ``assistant`` / ``self``) ものだけ。

    実インシデント (2026-09-08 夜のライブ監査): 物理の計算に対する訂正
    「違います。最初の答えを計算し直してください。…」が
    ``mem.personal.family`` として書かれ、``from_correction`` の力で本物の
    家族ファクト 4 件を supersede した (G-01)。9 時間前のノートが引用中の
    「色が違う」で訂正候補になり、より新しい occupation を supersede した
    例もある (G-04)。

    aux が使えず検証できなかった (``correction_verified_at is None``) ノートは
    False — 訂正でないと決めつけるのではなく、**通常の再言明として扱う**
    (単値スロットなら発話時刻順の置換だけが働く)。
    """
    return str(getattr(note, "correction_verdict", "") or "") in _POINTING_TARGETS


def note_verification_rejected(note: object) -> bool:
    """検証済みで **訂正ではないと判定された** ノートか (純粋関数)。

    ``same_value`` / ``premise_change`` / ``none`` / ``third_party`` /
    ``no_context`` … が入る。未検証 (``correction_verified_at is None``) は
    False — まだ判定していないだけなので、フォールバック経路には乗せる。
    """
    if getattr(note, "correction_verified_at", None) is None:
        return False
    return not note_is_verified_correction(note)


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
            (``local_paths.agent_trace_dir``、``AgentTraceStore`` の常設
            出力先)。``MDPTraceExtractor`` は
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
    #: 未検証の訂正候補を据え置くか (Step 8.0 の補助タスクが使える構成で真)。
    #: 偽 (degraded) なら従来どおり通常の再言明として消費する。
    defer_unverified_corrections: bool = False

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
    #: 検証待ちで据え置いた訂正候補 (次サイクルで再検討する)。
    notes_deferred: int = 0
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

    #: 抽出規則の版。``Provenance.extractor_version`` に刻む。抽出の意味を
    #: 変えたら上げる (どの版が作ったファクトかで選り分けて再導出するため)。
    EXTRACTOR_VERSION: int = 1

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
        - ``from_correction`` は **検証済みの訂正** から継承
          (:func:`note_is_verified_correction`)
        - confidence はデフォルト 0.5
        """
        canonical = subject.strip()
        if ctx.canonicalizer is not None:
            canonical = ctx.canonicalizer(canonical)
        canonical = self.truncate(canonical, self.MAX_SUBJECT_LEN)
        now = ctx.current_time()
        # 相対日付 (「来週の金曜日」「明後日」) は発話時刻と組で初めて意味を
        # 持つ。絶対日付を併記して残す (core.relative_date、H-04)。
        utterance_ts = _utterance_time(note, now)
        clipped_obj = self.truncate(
            annotate_relative_dates(
                object_text,
                datetime.fromtimestamp(utterance_ts, tz=timezone.utc).astimezone(),
            ),
            self.MAX_OBJECT_LEN,
        )

        # ノート由来の trace_id を最優先で fact / provenance に伝播する
        #。明示的に渡された trace_id (MDPTraceExtractor の
        # episode_id) はそれを優先する。
        effective_trace_id = trace_id or getattr(note, "trace_id", None)
        prov = Provenance(
            note_id=getattr(note, "id", None),
            session_id=getattr(note, "session_id", None) or None,
            turn_id=getattr(note, "turn_id", "") or None,
            trace_id=effective_trace_id,
            mode=getattr(note, "mode", self.mode) or self.mode,
            project_id=getattr(note, "project_id", None) or ctx.project_id,
            source=getattr(note, "source", None),
            captured_at=now,
            extractor=type(self).__name__,
            extractor_version=self.EXTRACTOR_VERSION,
        )

        fact = SemanticFact(
            id=SemanticFact.new_id(),
            subject=canonical or "unknown",
            predicate=predicate,
            object=clipped_obj,
            type=fact_type,
            scope=scope,
            mode_origin=self.mode,
            # 誰が述べたか (c_16 §3)。``note_facts.origin_of`` と同じ規則
            # — 片方だけ変えると経路によって注入可否が食い違う。
            origin=origin_of(note),  # type: ignore[arg-type]
            provenances=[prov],
            confidence=confidence,
            pinned=bool(getattr(note, "pin_flag", False)),
            # 訂正ターン由来か。**検証済み** (Step 8.0 が assistant / self と
            # 判定した) ノートからだけ引き継ぎ、競合解決が「同一セッション
            # だから微妙ケース」として pending へ落とすのを免除する
            # (:func:`note_is_verified_correction` / SemanticFact.from_correction)。
            from_correction=note_is_verified_correction(note),
            # **発話時刻**を継ぐ (抽出時刻ではない)。sleep-time は 1 回の
            # バッチで会話全体を抽出するため ``now`` を入れると全ファクトの
            # ``created_at`` が同一秒になり、「新しい方を採る」判定が原理的に
            # 成立しない。実データ (2026-08-27 ライブ監査) では 12 件すべてが
            # ``1787814691.47〜.48`` に潰れており、訂正と初出の前後関係が
            # ストアから失われていた。``(N日前の記録)`` ラベルの根拠でもある。
            created_at=utterance_ts,
            accessed_at=now,
            session_ids=(
                {getattr(note, "session_id", "")} if note and note.session_id else set()
            ),
            private=False,  # private ノートはここに来ない
            trace_id=effective_trace_id,
            # 言語はノートから継ぐ (無ければ本文から決定論判定)。
            lang=getattr(note, "lang", "") or detect_lang(clipped_obj),
        )
        for key, value in overrides.items():
            setattr(fact, key, value)
        # ``statement`` (規則で命題化した本文) は ``fact.text`` で object より
        # 優先されるので、相対日付の併記もこちらに掛ける (A/B 2026-09-10 で
        # ``mem.personal.schedule`` の「来週の水曜日」が併記されずに残った)。
        statement = getattr(fact, "statement", None)
        if statement:
            fact.statement = annotate_relative_dates(
                statement,
                datetime.fromtimestamp(utterance_ts, tz=timezone.utc).astimezone(),
            )
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
