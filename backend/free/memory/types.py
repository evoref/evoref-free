"""EvorefMem 共通型

含まれるもの:
- 型 Literal: `NoteSource` / `MemoryMode` / `FactType` / `TaskStatus`
- `Provenance` データクラス — SemanticFact の出処トレース
- `SemanticFact` データクラス — 意味記憶の **作業用** 表現

永続形は `Evidence` (kind=`fact`) 1 本で、対応は
`backend.free.memory.semantic.fact` が持つ (c_16 §3 / §4.2)。本モジュールは
JSONL シリアライザを持たない — レコードの読み書きは `EvidenceStore` の
事象ログと snapshot に閉じる。

設計原則 (CLAUDE.md / .claude/rules/backend.md):
- Python 3.12+ の型表現 (`X | None`, `Literal[...]`)
- フレームワーク非依存 (pydantic 不使用、純粋 dataclass)
- 後方互換不要
- ベクトル列は numpy のみで扱い、**永続化しない**

`MemoryNote` は `backend.free.memory.episodic.note` に置く (エピソード記憶の
永続形 `Evidence` と対で読む方が分かりやすいため)。本モジュールは
`MemoryNote` を再エクスポートしない (循環依存防止)。
ただし `MemoryNote` 用の型 Literal (`NoteSource` / `MemoryMode` /
`TaskStatus`) は本モジュールで一元管理する。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from backend.free.rag.evidence import new_evidence_id
from backend.free.rag.evidence.types import Origin, Veracity

# ──────────────────────────────────────────────────────────────────────────
# 型 Literal
# ──────────────────────────────────────────────────────────────────────────

NoteSource = Literal["user", "assistant", "system", "rag"]
"""MemoryNote の発生源"""

MemoryMode = Literal["chat", "create"]
"""モード (チャット / クリエイト)"""

TaskStatus = Literal["open", "in_progress", "done", "failed"]
"""task ファクト / MemoryNote のタスク状態"""

FactType = Literal[
    "personal_fact",
    "world_fact",
    "preference",
    "emotion",
    "opinion",
    "belief",
    "decision",
    "commitment",
    "project",
    "policy",
    "fewshot",         # policy subtype から独立昇格 (EvorefLearn owned)
    "failure_pattern",
    "learned_failure_pattern",  # PolicyAdjuster 由来の集約失敗パターン (EvorefLearn owned)
    "progress_marker",
    "task",
    "create_task",
    "artifact",        # ラルフループの編集成果物トレース
    "create",
    "model",
    "claim",           # know.* の取得単位由来の主張 (c_16 §3.5 / §4.2)
]
"""SemanticFact の type タグ (永続形では ``Evidence.attrs.fact_type``)。

`policy` / `failure_pattern` / `progress_marker` は統合済。`artifact` は
ラルフループの成果物 (ファイルパス / diff SHA1 / 行数) を追跡する。
`create_task` と `fewshot` は: 前者は Extractor 由来と LoopDriver 由来の
構造差を明示、後者は policy subtype から意味的に独立した FactType に昇格。
`learned_failure_pattern` は LogIngestor + PolicyAdjuster で追加:
develop=evolve で出力される decision/outcome JSONL を集約した結果、失敗率
閾値を超えた (decision_point, chosen) パターンを EvorefLearn pillar が
SemMem に書き戻す。loop owned の `failure_pattern` (quality_gate 由来) と
origin / namespace を分離して共存させる。`claim` は `know.<domain>.<topic>`
の世界知識で、取得単位 (`items.jsonl`) を provenance に持つ。
"""


# ──────────────────────────────────────────────────────────────────────────
# Provenance
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class Provenance:
    """SemanticFact の出処メタデータ。

    1 つのファクトは複数 Provenance を持ちうる (同一事実が複数セッションで
    観測された場合など)。独立出所数 (裏取り件数) は保存せず、読み込み時に
    `source_id` / `session_id` のユニーク数として数える (c_16 §3.1)。
    """

    note_id: str | None = None
    session_id: str | None = None
    turn_id: str | None = None
    """元になった WM ターンの ID。セッション内の順序と、そのターンの
    ログ / experience への連結点 (c_05 §0.6)。"""

    trace_id: str | None = None
    mode: MemoryMode | None = None
    project_id: str | None = None
    source: NoteSource | None = None
    captured_at: float = 0.0

    extractor: str | None = None
    """このファクトを作った抽出器 / キュレータのクラス名。

    ファクトの生産者は 4 系統以上 (ChatExtractor / CreateExtractor /
    MDPTraceExtractor と LLM キュレータ 3 種) あるのに、どれが作ったかを
    記録していなかった。抽出ロジックを変えた後、既存ファクトを「どの版が
    作ったか」で選り分けて再導出できない (2026-09-05 監査)。
    """

    extractor_version: int | None = None
    """抽出器の版。抽出規則を変えたら上げる。"""

    model: str | None = None
    """LLM を使って作られた場合の生成モデル名 (決定論抽出は ``None``)。"""

    source_id: str | None = None
    """文書 / 取得単位の識別子 (``doc:<package>/<doc>`` / ``item:ki_…``)。

    `know.*` の claim は取得単位を必ずここに持つ (c_16 §3.1 / §4.2)。裏取り
    件数はこの値のユニーク数で数える。
    """

    def to_dict(self) -> dict[str, Any]:
        return {
            "note_id": self.note_id,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "trace_id": self.trace_id,
            "mode": self.mode,
            "project_id": self.project_id,
            "source": self.source,
            "captured_at": self.captured_at,
            "extractor": self.extractor,
            "extractor_version": self.extractor_version,
            "model": self.model,
            "source_id": self.source_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Provenance:
        return cls(
            note_id=d.get("note_id"),
            session_id=d.get("session_id"),
            turn_id=d.get("turn_id"),
            trace_id=d.get("trace_id"),
            mode=d.get("mode"),
            project_id=d.get("project_id"),
            source=d.get("source"),
            captured_at=float(d.get("captured_at", 0.0)),
            extractor=d.get("extractor"),
            extractor_version=d.get("extractor_version"),
            model=d.get("model"),
            source_id=d.get("source_id"),
        )


# ──────────────────────────────────────────────────────────────────────────
# SemanticFact
# ──────────────────────────────────────────────────────────────────────────


@dataclass(eq=False)
class SemanticFact:
    """意味記憶 1 件の **作業用** 表現 (永続形は ``Evidence``)。

    `subject` / `predicate` / `object` の 3 つ組で意味を表現し、`scope`
    (`global` / `project:<id>`) と `type` で優先度を制御する。**スコープは
    フィールドであってディレクトリではない** — ストアは 1 つ (c_16 §4.2)。

    ``eq=False`` (同一性比較) — ファクトは ``id`` で識別する設計なので値比較に
    意味が無く、しかも ``embedding: np.ndarray`` を持つため生成される
    ``__eq__`` は危険。dataclass の ``__eq__`` はフィールドのタプル比較で、
    numpy 配列同士の比較は ``bool()`` で ``ValueError`` を投げる。

    **旧実装が動いていたのは偶然**だった: 第 1 フィールドが ``id: str`` で、
    id が違えばタプル比較がそこで False を返して打ち切られ ``embedding`` まで
    到達しない。つまりフィールドの並び順が守っていただけで、``id`` を後ろへ
    動かす / 別のフィールドを先頭に足す、といった無関係な変更で
    ``list.index()`` や ``in`` が突然例外を投げるようになる (2026-09-01 監査)。
    """

    # ── 識別子・本体 ────────────────────────────────────────────────────
    id: str
    subject: str
    predicate: str
    object: str
    type: FactType
    scope: str  # "global" or "project:<project_id>"

    statement: str | None = None
    """正規化済みの命題。``None`` は未正規化 (``object`` をそのまま使う)。

    ``object`` には発話原文がそのまま入る (抽出器は原文を切り出すだけ)。
    そのため ``[関連する記憶]`` には会話の足場や一人称がついた行が並び、
    値としての比較もできない。実データ (2026-08-16 監査時点):

        mem.personal.user states:
        「コーヒー派？紅茶派？私はコーヒーを1日3杯は飲んじゃう。」

    ここに正規化後の命題を **別フィールドで** 持ち、``object`` は証拠として
    残す。上書きしないのは、正規化が誤ったときに復旧できるようにするため
    (未検証の生成物が権威ある事実として永続化される事故を、このリポジトリは
    繰り返し踏んでいる)。消費側は ``fact.text`` を使う。

    永続形では ``Evidence.text`` がこの値 (無ければ ``object``)。
    """

    # ── メタ ────────────────────────────────────────────────────────────
    mode_origin: MemoryMode = "chat"
    lang: str = ""
    """本文の言語 (``ja`` / ``en`` / 未判定は空)。決定論判定で埋める。

    横断検索の順位付けで **同点時のタイブレークと重み付け** にだけ使う。
    フィルタには使わない — GUI ロケールで入力照合を排他にすると片方の言語が
    死ぬ前例がある (2026-09-03)。
    """
    provenances: list[Provenance] = field(default_factory=list)
    confidence: float = 0.5
    pinned: bool = False
    pin_locked_until: float | None = None
    profile_id: str = "default"

    origin: Origin = "user"
    """誰が述べたか (c_16 §3 / §4.2)。

    ``user`` = ユーザーの言明、``tool`` = ツール出力から導いた事実、
    ``assistant`` = アシスタントの発話由来、``document`` / ``web`` =
    取り込んだ文書・取得器。``mem.*`` の競合は「``origin=user`` かつ
    ``as_of`` が新しい方が勝つ」で解く (c_16 §4.2)。``assistant`` 由来は
    既定で注入しない (``memory.evidence.ranking.allow_assistant_origin_injection``)。
    """

    veracity: Veracity = "stated"
    """真偽状態 (c_16 §3)。競合中は ``disputed``、取り下げは ``retracted``。"""

    contradicts: list[str] = field(default_factory=list)
    """矛盾する相手のファクト id (c_16 §3 / §4.2)。競合解決が結ぶ。"""

    # ── supersession ────────────────────────────────────────────────────
    superseded_by: str | None = None

    # ── 検索・観測 ──────────────────────────────────────────────────────
    embedding: np.ndarray | None = None
    """**永続化しない** 作業用ベクトル。

    ベクトルの正は ``embeddings/<model_id>/`` (c_16 §6.1)。注入の関連度ゲート
    と sleep-time の競合検出が使うぶんは
    :meth:`SemanticStore.vectors_for` が snapshot から復元して載せる。
    """

    embed_as_query: bool = False
    """このファクトを **query 側** で埋め込むか (永続形は ``attrs.embed_as_query``)。

    既定は document 側。``idx.command.*`` のような内部索引は「過去の質問文」を
    溜めて読み手 (``ToolCallJudge``) が ``embed_query`` で引くので、書く側も
    query 側で揃えないと instruction-aware なモデルでは自己類似度が 0.78 程度
    まで落ちる。側の決定は
    :func:`~backend.free.memory.sleep._curator_common.index_embed_fields` が
    SSOT で、実際に埋め込むのは snapshot 生成 (c_16 §6.1)。"""

    embed_mode: str = "chat"
    """``embed_as_query`` のときの ``mode`` (``embedding.instructions`` の鍵)。

    読み手の ``embed_query(query, mode=mode)`` と揃える。document 側では無視。"""

    created_at: float = 0.0
    """**発話時刻** (抽出時刻ではない)。永続形の ``as_of`` (c_16 §3)。"""

    accessed_at: float = 0.0
    """最後に使われた時刻。永続形の ``last_used_at``。"""

    session_ids: set[str] = field(default_factory=set)
    private: bool = False

    # ── 統合追加フィールド ────────────────────────────────────────
    trace_id: str | None = None
    """MDP トレース連結用 (agent_tracer 由来)。永続形は ``provenance[0].trace_id``。"""

    auto_evolved: bool = False
    """PolicyEvolver により自動進化したファクトか
    (`conflict.auto_for_evolved_policies` の判定に使用)"""

    from_correction: bool = False
    """ユーザーが自分の値を言い直したターン由来か。

    判定は :func:`backend.free.agent.feedback.restates_a_value` で、
    チャット応答パス → ``WorkingMemory.add_turn(correction=...)`` →
    ``MemoryNote.is_correction`` → 抽出器、と伝播する。

    ``SemanticConflictResolver._decide`` がこれを見て、同一スロットの旧値との
    競合を **disputed にせず即 supersede** する。``_is_borderline`` は
    「同 ``session_id``」または「``confirm_window_hours`` 以内」を微妙ケース
    として disputed にするが、**会話中の訂正はその両方を必ず満たす**ため、
    印が無いといちばん確度の高い訂正がいちばん自動解決されなかった。"""

    failure_signature: str | None = None
    """failure_pattern の照合用ハッシュ
    (error_type, normalized_file_path, last_3_step_actions) の SHA1 先頭 12 桁"""

    eval_metric: dict[str, float] | None = None
    """policy ファクトの評価値 (fitness / accuracy / latency 等)"""

    # ── 前方互換 round-trip ───────────────────────────
    _extra: dict[str, Any] = field(default_factory=dict)
    """未知キーの退避先 (``Evidence._extra`` と往復する)。

    利用者は EvorefMem 内部に限定し、pillar 境界を越えて直接参照しないこと。"""

    # ── ヘルパ ──────────────────────────────────────────────────────────

    @property
    def text(self) -> str:
        """提示・比較・埋め込みに使う本文。

        正規化済みの :attr:`statement` があればそれを、無ければ ``object``
        (発話原文) を返す。永続形の ``Evidence.text`` と同じ値。
        """
        return self.statement or self.object

    @staticmethod
    def new_id() -> str:
        """新規ファクト用の ID を発番する。

        ``Evidence.id`` と同一の体系 (``ev_`` + hex12) — ファクトの永続形は
        ``Evidence`` なので、別体系の id を持つと 1 レコード 2 名前になる。
        """
        return new_evidence_id()

    @staticmethod
    def make_global_scope() -> str:
        return "global"

    @staticmethod
    def make_project_scope(project_id: str) -> str:
        return f"project:{project_id}"

    def is_project_scoped(self) -> bool:
        return self.scope.startswith("project:")

    def project_id(self) -> str | None:
        if self.is_project_scoped():
            return self.scope.split(":", 1)[1]
        return None


def make_fact(
    subject: str,
    predicate: str,
    object_: str,
    type: FactType,
    scope: str,
    *,
    mode_origin: MemoryMode = "chat",
    confidence: float = 0.5,
    now: float | None = None,
    **overrides: Any,
) -> SemanticFact:
    """テスト・呼び出し側の利便性のための簡易ファクトリ。

    必須項目だけを位置引数で受け取り、残りはデフォルト値で埋める。
    `overrides` で任意フィールドを追加上書きできる。
    """
    if now is None:
        now = time.time()
    fact = SemanticFact(
        id=SemanticFact.new_id(),
        subject=subject,
        predicate=predicate,
        object=object_,
        type=type,
        scope=scope,
        mode_origin=mode_origin,
        confidence=confidence,
        created_at=now,
        accessed_at=now,
    )
    for key, value in overrides.items():
        setattr(fact, key, value)
    return fact


__all__ = [
    "FactType",
    "MemoryMode",
    "NoteSource",
    "Provenance",
    "SemanticFact",
    "TaskStatus",
    "make_fact",
]
