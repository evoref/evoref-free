"""成長の台帳 (``learning.growth_ledger``、docs/f_04 §6.2 / §6.6)。

Level 2 (成長) の提案・採用・却下・期限切れ・巻き戻し・置き換えを 1 行ずつ記録し、
「なぜこの版か」を ID で辿れるようにする (c_05 §0.6 の ID 連鎖)。

- 行は **ID だけ** を持ち、問いや候補の本文は持たない (候補の本文は提案ファイル
  ``learning.growth_proposal`` が持ち、採用されたものだけが ``history/`` /
  ``generation_deltas.json`` に残る)。
- model_key 単位 (``store/learning/<mk>/growth/ledger.jsonl``)。書き手は Level 1 (背景処理) と
  利用者の採用・却下・巻き戻し (API / CLI) だけで、チャット応答パスでは書かない
  (書き手スレッドの対象外。採用台帳 ``learning.prompt_adoption`` と同じ通常の追記)。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from backend.io import JSONLAppendStore
from backend.io.codec import CodecError, codec_for, persisted
from backend.io.format_registry import FormatSpec, register_format
from backend.io.id_registry import new_id
from backend.log_config import get_logger
from backend.utils import utc_now

logger = get_logger("learning.growth.ledger")

#: 行 ID と提案 ID の接頭辞 (ID 台帳に登録済み)。
ROW_ID_PREFIX = "gro_"
PROPOSAL_ID_PREFIX = "gpp_"

GrowthOp = Literal["propose", "adopt", "reject", "expire", "rollback", "supersede"]
OPS: tuple[str, ...] = ("propose", "adopt", "reject", "expire", "rollback", "supersede")

AssetKind = Literal["prompt", "generation_params", "recall_keys", "exemplar"]
ASSET_PROMPT: AssetKind = "prompt"
ASSET_GENERATION_PARAMS: AssetKind = "generation_params"
ASSET_KINDS: tuple[str, ...] = ("prompt", "generation_params", "recall_keys", "exemplar")
#: 提案・採用の対象にする資産 (§6.1。``recall_keys`` は段 3、``exemplar`` は語彙の予約)。
PROPOSABLE_KINDS: tuple[str, ...] = (ASSET_PROMPT, ASSET_GENERATION_PARAMS)

DecidedBy = Literal["auto", "user", "monitor"]
DECIDED_BY: tuple[str, ...] = ("auto", "user", "monitor")

#: ``reason`` の語彙 (open。行の意味を変えずに足せる)。
REASON_PROPOSED = "proposed"
REASON_TTL = "ttl"
REASON_MAX_PENDING = "max_pending"
REASON_SUPERSEDED = "superseded"
REASON_GUARD_REGRESSION = "guard_regression"
REASON_USER = "user"

LEDGER_FILE = "ledger.jsonl"


@persisted()
@dataclass(kw_only=True)
class GrowthGate:
    """採用ゲートの結果 (§4.5 の一対比較ゲートをそのまま。測っていない値は null)。"""

    name: str = ""
    wins: int | None = None
    losses: int | None = None
    ties: int | None = None
    noise_floor: int | None = None
    sample_wins: int | None = None
    sample_losses: int | None = None
    sample_ties: int | None = None
    guards_passed: int | None = None
    guards_failed: int | None = None
    case_digest: str = ""
    _extra: dict[str, Any] | None = None


@persisted()
@dataclass(kw_only=True)
class GrowthLedgerRow:
    """``ledger.jsonl`` の 1 行。``_v`` は行の版 (形式の版と同じ、c_05 §0.5.1)。"""

    _v: int = 1
    written_at: str = ""
    row_id: str
    proposal_id: str = ""
    op: GrowthOp
    asset_kind: AssetKind = ASSET_PROMPT
    mode: str = ""
    #: 提案時点の現行版 (プロンプトは ``meta.version`` の文字列、生成パラメータは
    #: ``generation_deltas`` の ``delta_version``)。
    parent_version: str = ""
    #: 候補本文の blake2b (本文そのものは持たない)。
    candidate_digest: str = ""
    #: ``propose`` / ``supersede`` / ``expire`` は ``None`` (誰も採否を決めていない)。
    decided_by: DecidedBy | None = None
    gate: GrowthGate | None = None
    reason: str = ""
    #: 採用台帳 ``learning.prompt_adoption`` の ``pad_`` (プロンプトの採用・巻き戻しの行だけ)。
    adoption_id: str | None = None
    regression_case_ids: list[str] = field(default_factory=list)
    experience_ids: list[str] = field(default_factory=list)
    level1_session_id: str | None = None
    model_key: str = ""
    _extra: dict[str, Any] | None = None


GROWTH_LEDGER_FORMAT = register_format(FormatSpec(
    format_id="learning.growth_ledger",
    version=1,
    klass="sot",
    writers=frozenset({"free"}),
    path_key=f"store/learning/<mk>/growth/{LEDGER_FILE}",
    retention="unbounded (one line per propose/adopt/reject/expire/rollback/supersede)",
    export=True,
    encodings=("jsonl",),
    enums={"op": "closed", "asset_kind": "closed", "decided_by": "closed"},
    records=(GrowthLedgerRow,),
))


_ROW_CODEC = codec_for(GrowthLedgerRow)
_GATE_CODEC = codec_for(GrowthGate)


def gate_from_verdict(verdict: dict[str, Any] | None, *, case_digest: str = "") -> GrowthGate | None:
    """採用ゲートの verdict / 生成パラメータの結果辞書から ``gate`` を作る (無ければ ``None``)。"""
    if not verdict:
        return None

    def _int(key: str) -> int | None:
        value = verdict.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    return GrowthGate(
        name=str(verdict.get("gate") or ("pairwise" if verdict.get("wins") is not None else "")),
        wins=_int("wins"), losses=_int("losses"), ties=_int("ties"),
        noise_floor=_int("noise_floor"),
        sample_wins=_int("sample_wins"), sample_losses=_int("sample_losses"),
        sample_ties=_int("sample_ties"),
        guards_passed=_int("guards_passed"), guards_failed=_int("guards_failed"),
        case_digest=case_digest or str(verdict.get("case_digest") or ""),
    )


def encode_gate(gate: GrowthGate | None) -> dict[str, Any] | None:
    return None if gate is None else _GATE_CODEC.encode(gate)


def decode_gate(raw: Any) -> GrowthGate | None:
    if not isinstance(raw, dict):
        return None
    try:
        return _GATE_CODEC.decode(raw)
    except CodecError:
        return None


class GrowthLedger:
    """成長の台帳の読み書き (:class:`JSONLAppendStore` の上の薄い層)。

    版が新しい行を持つファイルへは書かない (``DataReadonlyError``、c_05 §0.4.5)。readonly
    起動でも同じ例外になる (呼出側が扱う)。
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        #: 直近の読み込みで飛ばした行 (必須キーの欠け・型の合わない値)。
        self.skipped_rows = 0
        self._store: JSONLAppendStore[GrowthLedgerRow] = JSONLAppendStore(
            self.path,
            serialize=lambda row: json.dumps(_ROW_CODEC.encode(row), ensure_ascii=False),
            deserialize=self._deserialize,
            key_of=lambda row: row.row_id,
            row_version=GROWTH_LEDGER_FORMAT.version,
        )

    def _deserialize(self, line: str) -> GrowthLedgerRow:
        try:
            return _ROW_CODEC.decode(json.loads(line))
        except CodecError:
            self.skipped_rows += 1
            raise

    def append(self, **fields: Any) -> GrowthLedgerRow:
        """1 行追記して、その行を返す (``row_id`` と ``written_at`` はここで付ける)。"""
        row = GrowthLedgerRow(written_at=utc_now(), row_id=new_id(ROW_ID_PREFIX), **fields)
        if not row.model_key:
            row.model_key = self.path.parent.parent.name
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._store.append(row)
        return row

    def extend(self, rows: list[GrowthLedgerRow]) -> None:
        """読んだ行をそのまま積む (import / 往復の検査。ID と時刻は変えない)。"""
        if rows:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._store.append_many(rows)

    def load(self) -> list[GrowthLedgerRow]:
        """書いた順の行 (版が新しい行・読めない行・未知の ``op`` の行は飛ばして数える)。"""
        self.skipped_rows = 0
        rows = [r for r in self._store.load_all().values() if r.op in OPS]
        if self.skipped_rows:
            logger.warning(
                "Skipped %d growth ledger row(s) without required keys or with bad values in %s",
                self.skipped_rows, self.path,
            )
        return rows

    def tail(self, n: int = 20) -> list[GrowthLedgerRow]:
        """末尾 ``n`` 行 (新しい順ではなく書いた順)。"""
        rows = self.load()
        return rows[-max(0, n):] if n else []

    def newer_rows(self) -> int:
        """版がこの台帳より新しい行の数 (1 以上なら書かない)。"""
        return self._store.newer_rows()


def encode_row(row: GrowthLedgerRow) -> dict[str, Any]:
    """API / CLI へ出す形 (台帳の 1 行そのもの。本文は元から持たない)。"""
    return _ROW_CODEC.encode(row)


__all__ = [
    "ASSET_GENERATION_PARAMS",
    "ASSET_KINDS",
    "ASSET_PROMPT",
    "DECIDED_BY",
    "GROWTH_LEDGER_FORMAT",
    "LEDGER_FILE",
    "OPS",
    "PROPOSABLE_KINDS",
    "PROPOSAL_ID_PREFIX",
    "REASON_GUARD_REGRESSION",
    "REASON_MAX_PENDING",
    "REASON_PROPOSED",
    "REASON_SUPERSEDED",
    "REASON_TTL",
    "REASON_USER",
    "ROW_ID_PREFIX",
    "GrowthGate",
    "GrowthLedger",
    "GrowthLedgerRow",
    "decode_gate",
    "encode_gate",
    "encode_row",
    "gate_from_verdict",
]
