"""EvorefMem 共通型

EvorefMem 統合仕様 に基づく共通型・データクラスを定義する

含まれるもの:
- 型 Literal: `NoteSource` / `MemoryMode` / `FactType` / `Scope` / `TaskStatus` /
  `ReviewStatus`
- `Provenance` データクラス — SemanticFact の出処トレース
- `SemanticFact` データクラス — 意味記憶の最小単位
- `serialize_fact` / `deserialize_fact` — JSONL 行レベルのシリアライザ

設計原則 (CLAUDE.md / .claude/rules/backend.md):
- Python 3.12+ の型表現 (`X | None`, `Literal[...]`)
- フレームワーク非依存 (pydantic 不使用、純粋 dataclass)
- 後方互換不要
- ベクトル列は numpy のみで扱い、JSONL では list 化する

`MemoryNote` は履歴的な事情で `backend.free.memory.stores.short_term` に置く。
本モジュールは `MemoryNote` を再エクスポートしない (循環依存防止)。
ただし `MemoryNote` 用の型 Literal (`NoteSource` / `MemoryMode` /
`TaskStatus`) は本モジュールで一元管理する。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4

import numpy as np

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
]
"""SemanticFact の type タグ。`policy` / `failure_pattern` / `progress_marker`
は統合済。`artifact` はラルフループの成果物 (ファイルパス / diff SHA1 /
行数) を追跡する。`create_task` と `fewshot` は: 前者は Extractor 由来と
LoopDriver 由来の構造差を明示、後者は policy subtype から意味的に独立した
FactType に昇格。`learned_failure_pattern` は LogIngestor + PolicyAdjuster
で追加: develop=evolve で出力される decision/outcome JSONL を集約した結果、
失敗率閾値を超えた
(decision_point, chosen) パターンを EvorefLearn pillar が SemMem に書き戻す。
loop owned の `failure_pattern` (quality_gate 由来) と origin / namespace を分離して
共存させる。"""

ReviewStatus = Literal[
    "none",
    "pending",
    "resolved_keep_old",
    "resolved_keep_new",
    "resolved_merged",
]
"""コンフリクト解消ワークフローの状態"""


# ──────────────────────────────────────────────────────────────────────────
# Provenance
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class Provenance:
    """SemanticFact の出処メタデータ。

    1 つのファクトは複数 Provenance を持ちうる (同一事実が複数セッションで
    観測された場合など)。
    """

    note_id: str | None = None
    session_id: str | None = None
    trace_id: str | None = None
    mode: MemoryMode | None = None
    project_id: str | None = None
    source: NoteSource | None = None
    captured_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "note_id": self.note_id,
            "session_id": self.session_id,
            "trace_id": self.trace_id,
            "mode": self.mode,
            "project_id": self.project_id,
            "source": self.source,
            "captured_at": self.captured_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Provenance:
        return cls(
            note_id=d.get("note_id"),
            session_id=d.get("session_id"),
            trace_id=d.get("trace_id"),
            mode=d.get("mode"),
            project_id=d.get("project_id"),
            source=d.get("source"),
            captured_at=float(d.get("captured_at", 0.0)),
        )


# ──────────────────────────────────────────────────────────────────────────
# SemanticFact
# ──────────────────────────────────────────────────────────────────────────


@dataclass(eq=False)
class SemanticFact:
    """

    `subject` / `predicate` / `object` の 3 つ組で意味を表現し、`scope`
    (`global` / `project:<id>`) と `type` で物理分離・優先度制御する。
    `policy` / `failure_pattern` / `progress_marker` は統合された
    永続化先として用いられる。

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
    """

    # ── メタ ────────────────────────────────────────────────────────────
    subject_aliases: list[str] = field(default_factory=list)
    scope_locked: bool = False
    mode_origin: MemoryMode = "chat"
    provenances: list[Provenance] = field(default_factory=list)
    confidence: float = 0.5
    pinned: bool = False
    pin_locked_until: float | None = None
    profile_id: str = "default"

    # ── supersession ────────────────────────────────────────────────────
    superseded_by: str | None = None
    supersedes: list[str] = field(default_factory=list)

    # ── レビュー ────────────────────────────────────────────────────────
    requires_user_review: bool = False
    review_status: ReviewStatus = "none"

    # ── 検索・観測 ──────────────────────────────────────────────────────
    embedding: np.ndarray | None = None
    created_at: float = 0.0
    accessed_at: float = 0.0
    access_count: int = 0
    session_ids: set[str] = field(default_factory=set)
    private: bool = False

    # ── 統合追加フィールド ────────────────────────────────────────
    trace_id: str | None = None
    """MDP トレース連結用 (agent_tracer 由来)"""

    credit_score: float | None = None
    """credit_assigner 由来の貢献度スコア"""

    auto_evolved: bool = False
    """PolicyEvolver により自動進化したファクトか
    (`conflict.auto_for_evolved_policies` の判定に使用)"""

    from_correction: bool = False
    """ユーザーが自分の値を言い直したターン由来か。

    判定は :func:`backend.free.agent.feedback.restates_a_value` で、
    チャット応答パス → ``WorkingMemory.add_turn(correction=...)`` →
    ``MemoryNote.is_correction`` → 抽出器、と伝播する。

    ``SemanticConflictResolver._decide`` がこれを見て、同一スロットの旧値との
    競合を **pending にせず即 supersede** する。``_is_borderline`` は
    「同 ``session_id``」または「``confirm_window_hours`` 以内」を微妙ケース
    として pending にするが、**会話中の訂正はその両方を必ず満たす**ため、
    印が無いといちばん確度の高い訂正がいちばん自動解決されなかった。"""

    failure_signature: str | None = None
    """failure_pattern の照合用ハッシュ
    (error_type, normalized_file_path, last_3_step_actions) の SHA1 先頭 12 桁"""

    eval_metric: dict[str, float] | None = None
    """policy ファクトの評価値 (fitness / accuracy / latency 等)"""

    # ── 前方互換 round-trip ───────────────────────────
    _version: int = 1
    """fact record のバージョン。`manifest.component_versions.fact` と一致させる。
    schema_version=1 の現状では常に 1。将来 fact record レイアウトを変更した際に
    SchemaMigrator が in-place rewrite で書き換える。省略時は 1 とみなす。"""

    _extra: dict[str, Any] = field(default_factory=dict)
    """JSONL round-trip 時に未知フィールドを保持するバッファ。
    v1 バックエンドが未知のキーを受け取ったとき、dataclass には吸収できない
    トップレベルキーをここに退避し、serializer が再度トップレベルに復元する。
    ステップ移行中の Free / Pro 混在や SchemaMigrator の in-place rewrite で
    データ欠損を起こさないための土台
    既知フィールドと同名キーを持つ場合は serializer で既知フィールド優先で
    上書きする (決定論)。利用者は EvorefMem 内部に限定し、pillar 境界を越えて
    直接参照しないこと。"""

    # ── ヘルパ ──────────────────────────────────────────────────────────

    @property
    def text(self) -> str:
        """提示・比較・埋め込みに使う本文。

        正規化済みの :attr:`statement` があればそれを、無ければ ``object``
        (発話原文) を返す。正規化が入っていない環境・古いファクトでも従来と
        同じ値になるので、消費側はこれを使えば分岐が要らない。
        """
        return self.statement or self.object

    @staticmethod
    def new_id() -> str:
        """新規ファクト用の短縮 ID を生成する"""
        return f"sf_{uuid4().hex[:12]}"

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


# ──────────────────────────────────────────────────────────────────────────
# シリアライザ (JSONL 行レベル)
# ──────────────────────────────────────────────────────────────────────────


# serialize / deserialize で "既知" として扱うトップレベル JSON キー集合。
# ここに含まれないキーは `_extra` に退避され、次回 serialize で原形のまま
# トップレベルに復元される
_KNOWN_FACT_KEYS: frozenset[str] = frozenset({
    "id",
    "subject",
    "subject_aliases",
    "predicate",
    "object",
    "statement",
    "type",
    "scope",
    "scope_locked",
    "mode_origin",
    "provenances",
    "confidence",
    "pinned",
    "pin_locked_until",
    "profile_id",
    "superseded_by",
    "supersedes",
    "requires_user_review",
    "review_status",
    "embedding",
    "created_at",
    "accessed_at",
    "access_count",
    "session_ids",
    "private",
    "trace_id",
    "credit_score",
    "auto_evolved",
    "from_correction",
    "failure_signature",
    "eval_metric",
    "_version",
})


def serialize_fact(
    fact: SemanticFact, *, include_embedding: bool = True,
) -> dict[str, Any]:
    """`SemanticFact` を JSON-serializable な dict に変換する。

    embedding は `tolist()` で list 化する。`session_ids` は set のため
    sorted list 化して決定的にする。ストレージ層から呼ばれる想定だが、
    ユニットテスト・デバッグ用途でも使う。

    `_extra` に保持された未知フィールドはトップレベルに復元する。
    既知フィールドと同名キーを持つ場合は既知フィールド側を優先する

    Args:
        include_embedding: ``embedding`` をペイロードに含めるか。永続化
            (``facts.jsonl``) では ``False`` を渡す — ベクトルの正は
            ``embeddings/<model_id>/vectors.npy`` 側で、JSON へ二重に持つと
            **実測でファイルの 92% がベクトルのテキスト表現** になる
            (94 ファクトで 3.03MB、同じ 90 本の npy は 0.37MB)。しかも
            ``facts.jsonl`` は追記式なので ``update_fact`` のたびに 1024 個の
            float をもう 1 行足す。既定を ``True`` のままにしてあるのは、
            export / デバッグダンプが従来どおり自己完結した dict を得られる
            ようにするため。
    """
    # _extra を先に展開し、既知フィールドで上書きする (既知フィールド優先)。
    out: dict[str, Any] = dict(fact._extra)
    out.update({
        "id": fact.id,
        "subject": fact.subject,
        "subject_aliases": list(fact.subject_aliases),
        "predicate": fact.predicate,
        "object": fact.object,
        "statement": fact.statement,
        "type": fact.type,
        "scope": fact.scope,
        "scope_locked": fact.scope_locked,
        "mode_origin": fact.mode_origin,
        "provenances": [p.to_dict() for p in fact.provenances],
        "confidence": fact.confidence,
        "pinned": fact.pinned,
        "pin_locked_until": fact.pin_locked_until,
        "profile_id": fact.profile_id,
        "superseded_by": fact.superseded_by,
        "supersedes": list(fact.supersedes),
        "requires_user_review": fact.requires_user_review,
        "review_status": fact.review_status,
        "embedding": (
            fact.embedding.tolist()
            if include_embedding and fact.embedding is not None
            else None
        ),
        "created_at": fact.created_at,
        "accessed_at": fact.accessed_at,
        "access_count": fact.access_count,
        "session_ids": sorted(fact.session_ids),
        "private": fact.private,
        # 統合追加
        "trace_id": fact.trace_id,
        "credit_score": fact.credit_score,
        "auto_evolved": fact.auto_evolved,
        "from_correction": fact.from_correction,
        "failure_signature": fact.failure_signature,
        "eval_metric": dict(fact.eval_metric) if fact.eval_metric is not None else None,
        "_version": int(fact._version),
    })
    return out


def deserialize_fact(d: dict[str, Any]) -> SemanticFact:
    """dict から `SemanticFact` を再構築する。

    後方互換は提供しないが、追加した統合フィールドが欠損して
    いてもデフォルト値で復元できる (内部で書き込んだファイルを
    内部で読み戻すケース等のため)

    `_KNOWN_FACT_KEYS` に含まれないトップレベルキーは `_extra` に退避し、
    次回 `serialize_fact` で原形のまま復元される
    `_version` 省略時は 1 として扱う (現行 schema_version=1 デフォルト)。
    """
    emb = d.get("embedding")
    eval_metric_raw = d.get("eval_metric")
    eval_metric = (
        {k: float(v) for k, v in eval_metric_raw.items()}
        if isinstance(eval_metric_raw, dict)
        else None
    )
    provenances_raw = d.get("provenances", [])
    extra = {k: v for k, v in d.items() if k not in _KNOWN_FACT_KEYS}
    return SemanticFact(
        id=d["id"],
        subject=d["subject"],
        subject_aliases=list(d.get("subject_aliases", [])),
        predicate=d["predicate"],
        object=d["object"],
        statement=d.get("statement"),
        type=d["type"],
        scope=d["scope"],
        scope_locked=bool(d.get("scope_locked", False)),
        mode_origin=d.get("mode_origin", "chat"),
        provenances=[Provenance.from_dict(p) for p in provenances_raw],
        confidence=float(d.get("confidence", 0.5)),
        pinned=bool(d.get("pinned", False)),
        pin_locked_until=d.get("pin_locked_until"),
        profile_id=d.get("profile_id", "default"),
        superseded_by=d.get("superseded_by"),
        supersedes=list(d.get("supersedes", [])),
        requires_user_review=bool(d.get("requires_user_review", False)),
        review_status=d.get("review_status", "none"),
        embedding=np.array(emb, dtype=np.float32) if emb is not None else None,
        created_at=float(d.get("created_at", 0.0)),
        accessed_at=float(d.get("accessed_at", 0.0)),
        access_count=int(d.get("access_count", 0)),
        session_ids=set(d.get("session_ids", [])),
        private=bool(d.get("private", False)),
        trace_id=d.get("trace_id"),
        credit_score=d.get("credit_score"),
        auto_evolved=bool(d.get("auto_evolved", False)),
        from_correction=bool(d.get("from_correction", False)),
        failure_signature=d.get("failure_signature"),
        eval_metric=eval_metric,
        _version=int(d.get("_version", 1)),
        _extra=extra,
    )


def serialize_fact_jsonl(
    fact: SemanticFact, *, include_embedding: bool = False,
) -> str:
    """1 ファクトを JSONL 1 行 (改行無し) にエンコードする

    **既定で ``embedding`` を書かない。** ベクトルの正は
    ``embeddings/<model_id>/vectors.npy`` で、``facts.jsonl`` へ二重に持つ
    意味は無い。読み戻しは ``SemanticFactStore._load`` が EmbeddingStore から
    hydrate する。旧形式 (embedding 入り) の行はそのまま読めるので、移行は
    「読めるが書かない」で足りる — 既存行は次の compact / rewrite で落ちる。
    """
    return json.dumps(
        serialize_fact(fact, include_embedding=include_embedding),
        ensure_ascii=False,
    )


def deserialize_fact_jsonl(line: str) -> SemanticFact:
    """JSONL 1 行から `SemanticFact` を復元する

    旧形式 (``embedding`` を含む行) も読める。新形式では ``embedding`` は
    ``None`` になり、``SemanticFactStore._load`` が EmbeddingStore から埋める。
    """
    return deserialize_fact(json.loads(line))


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
