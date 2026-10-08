"""採用台帳 (``learning.prompt_adoption``、c_05 §0.6 の「経験 → プロンプト版」の逆向き)。

経験 → 版の向きは ``gen_config.prompt_version`` で辿れるが、版 → 採用の根拠になった
経験の向きは辿れなかった (``<mode>.meta.json`` は次の採用で上書きされ、
``level1_history/`` は volatile で最新 N 件だけ)。Level 1 が版を上げた (採用) /
採用後監視が巻き戻した (rollback) ときに 1 行ずつ追記する。

- 行は ID だけを持ち、問いの本文は持たない (プライバシー)。
- ``cases[].experience_ids`` は **根拠となった候補群** であって、再生成に使った
  厳密なターンではない。``case_id`` は問いのハッシュ (``prompt_eval._case_id``) で、
  同じ問いの別ターンを 1 ケースに畳むので、その全ターンの経験 ID の和集合を持つ。
- 経験は 1000 件/ファイルでローテーションするので ``experience_ids`` は弱い参照
  (宙に浮いた ID は ``evoref doctor`` が数えるだけ)。
- 書き手は Level 1 (背景処理) だけで、チャット応答パスではない (c_05 §0.5.9 の
  書き手スレッドの対象外。``candidates.jsonl`` と同じ通常の追記)。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from backend.io import JSONLAppendStore
from backend.io.codec import codec_for, persisted
from backend.io.format_registry import FormatSpec, register_format
from backend.io.id_registry import new_id
from backend.utils import utc_now

#: ``adoption_id`` の接頭辞 (ID 台帳に登録済み)。
ADOPTION_ID_PREFIX = "pad_"

AdoptionOp = Literal["adopt", "rollback"]


@persisted()
@dataclass(kw_only=True)
class AdoptionCaseRef:
    """採用ゲートの評価ケース 1 件への参照 (問いの本文は持たない)。"""

    case_id: str
    kind: str = ""
    #: 根拠となった経験の ID (訂正ケースは (宛先, 訂正した発話) の順、同じ case_id の和集合)。
    experience_ids: list[str] = field(default_factory=list)
    _extra: dict[str, Any] | None = None


@persisted()
@dataclass(kw_only=True)
class PromptAdoptionRecord:
    """``adoptions.jsonl`` の 1 行。``_v`` は行の版 (形式の版と同じ、c_05 §0.5.1)。"""

    _v: int = 1
    written_at: str = ""
    adoption_id: str
    op: AdoptionOp = "adopt"
    mode: str = ""
    #: この操作で出来た版 (``<mode>.meta.json`` の ``version``)。
    version: int = 0
    #: 採用なら置き換えた版、巻き戻しなら戻した先の版。
    parent_version: int | None = None
    level1_session_id: str | None = None
    model_key: str = ""
    cases: list[AdoptionCaseRef] = field(default_factory=list)
    #: ``case_id`` を並べ替えて連結した blake2b (ケース集合の同一性の鍵)。
    case_digest: str = ""
    #: 一対比較ゲートの勝敗 (絶対採点・巻き戻しでは null)。
    wins: int | None = None
    losses: int | None = None
    ties: int | None = None
    _extra: dict[str, Any] | None = None


ADOPTIONS_FILE = "adoptions.jsonl"
PROMPT_ADOPTION_FORMAT = register_format(FormatSpec(
    format_id="learning.prompt_adoption",
    version=1,
    klass="sot",
    writers=frozenset({"free"}),
    path_key=f"store/learning/<mk>/prompts/{ADOPTIONS_FILE}",
    retention="unbounded (one line per adopted/rolled-back version)",
    export=True,
    encodings=("jsonl",),
    enums={"op": "closed"},
    records=(PromptAdoptionRecord,),
))


def adoption_store(path: Path) -> JSONLAppendStore[PromptAdoptionRecord]:
    """``adoptions.jsonl`` の追記ストア (版が新しい行があれば追記を拒否する)。"""
    codec = codec_for(PromptAdoptionRecord)
    return JSONLAppendStore(
        path,
        serialize=lambda record: json.dumps(codec.encode(record), ensure_ascii=False),
        deserialize=lambda line: codec.decode(json.loads(line)),
        key_of=lambda record: record.adoption_id,
        row_version=PROMPT_ADOPTION_FORMAT.version,
    )


def ledger_path(prompt_dir: Path) -> Path:
    """プロンプト版と同じパーティション (``store/learning/<mk>/prompts/``) の台帳。"""
    return Path(prompt_dir) / ADOPTIONS_FILE


def case_digest(case_ids: list[str]) -> str:
    """ケース集合の鍵 (順序に依らない)。"""
    joined = "\n".join(sorted(case_ids))
    return hashlib.blake2b(joined.encode("utf-8"), digest_size=8).hexdigest()


def case_refs(cases: list[Any]) -> list[AdoptionCaseRef]:
    """``PromptEvalCase`` の列から参照を作る (本文は落とす)。"""
    return [
        AdoptionCaseRef(
            case_id=c.case_id, kind=c.kind, experience_ids=list(c.experience_ids),
        )
        for c in cases
    ]


def append_adoption(
    prompt_dir: Path,
    *,
    op: AdoptionOp,
    mode: str,
    version: int,
    parent_version: int | None,
    level1_session_id: str | None = None,
    cases: list[AdoptionCaseRef] | None = None,
    wins: int | None = None,
    losses: int | None = None,
    ties: int | None = None,
) -> PromptAdoptionRecord:
    """台帳へ 1 行追記して、その行を返す。``model_key`` はパーティションの名前。"""
    refs = list(cases or ())
    record = PromptAdoptionRecord(
        written_at=utc_now(),
        adoption_id=new_id(ADOPTION_ID_PREFIX),
        op=op,
        mode=mode,
        version=int(version),
        parent_version=None if parent_version is None else int(parent_version),
        level1_session_id=level1_session_id,
        model_key=Path(prompt_dir).parent.name,
        cases=refs,
        case_digest=case_digest([r.case_id for r in refs]),
        wins=wins,
        losses=losses,
        ties=ties,
    )
    adoption_store(ledger_path(prompt_dir)).append(record)
    return record


def load_adoptions(prompt_dir: Path) -> list[PromptAdoptionRecord]:
    """台帳の行を書いた順に返す (版が新しい行・読めない行は飛ばす)。"""
    rows = adoption_store(ledger_path(prompt_dir)).load_all()
    return list(rows.values())


__all__ = [
    "ADOPTIONS_FILE",
    "ADOPTION_ID_PREFIX",
    "PROMPT_ADOPTION_FORMAT",
    "AdoptionCaseRef",
    "PromptAdoptionRecord",
    "adoption_store",
    "append_adoption",
    "case_digest",
    "case_refs",
    "ledger_path",
    "load_adoptions",
]
