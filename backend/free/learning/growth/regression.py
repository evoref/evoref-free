"""回帰台帳 (``learning.regression_case``) と回帰ケースの状態 (``learning.regression_status``)。

Level 2 (成長、docs/f_04 §6) の部品。実際に起きた失敗 (検証済みの訂正・👎・決定論の検証器・
会話の整合性) と同梱の種ケースを、**問いの本文ごと** 再生できる形で持つ。撤去した eval_core の
後継。

- 台帳はモデル非依存 (``store/learning/shared/growth/regression_cases.jsonl``)。1 行 1 ケースで、
  状態の変更 (quarantine / retire) は同じ ``case_id`` の行を追記する (後勝ち)。削除は tombstone
  の直後に compaction して本文を物理的に消す (forget / 件数上限)。
- 書き手は sleep-time Step 13.6 (収穫) と、後続の Level 1 の段・API (状態の変更) だけ。
  チャット応答パスでは書かない (書き手スレッドの対象外。採用台帳と同じ通常の追記)。
- 状態ファイル (model_key 単位の derived) は、ケースの役割 (guard / pressure / unknown) と
  収穫済みの経験 (``harvested``: 経験 ID → 収穫した源) を持つ。消えても台帳の再生と収穫の
  やり直しで作り直せる (やり直しで増えるのは重複の件数だけ — dedupe_key で畳む)。
- 同梱の種ケース (``_defaults/regression_seed.jsonl``、``source=seed``) は台帳へ写さず、読むとき
  に足す (``learning.growth.regression.use_seed``)。ガード専用で、件数上限には数えない。
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from backend.free.core.verifier_events import decided_by_shortcut
from backend.free.learning.case_filters import (
    MODEL_ORIGIN_PREFIXES,
    REASON_SHORTCUT_TOOL,
    is_model_origin,
    regeneration_mismatch_reason,
    replay_context,
)
from backend.io import JSONLAppendStore
from backend.io.codec import CodecError, codec_for, persisted
from backend.io.format_registry import FormatSpec, register_format
from backend.io.id_registry import new_id
from backend.io.versioned import VersionedPayloadFile
from backend.log_config import get_logger
from backend.utils import parse_utc, utc_now

if TYPE_CHECKING:
    from backend.free.optimizer.prompt_eval import PromptEvalCase

logger = get_logger("learning.growth.regression")

#: ``case_id`` の接頭辞 (ID 台帳に登録済み)。
CASE_ID_PREFIX = "rgc_"

SOURCE_VERIFIED_CORRECTION = "verified_correction"
SOURCE_USER_NEGATIVE = "user_negative"
SOURCE_VERIFIER_FAILURE = "verifier_failure"
SOURCE_CONSISTENCY = "consistency"
SOURCE_SEED = "seed"
#: 収穫源 (``seed`` は同梱の種ケースだけ)。読むときは未知の値も残す (open 列挙)。
SOURCES: tuple[str, ...] = (
    SOURCE_VERIFIED_CORRECTION, SOURCE_USER_NEGATIVE, SOURCE_VERIFIER_FAILURE,
    SOURCE_CONSISTENCY, SOURCE_SEED,
)

CaseStatus = Literal["active", "quarantined", "retired"]
STATUS_ACTIVE: CaseStatus = "active"
STATUS_QUARANTINED: CaseStatus = "quarantined"
STATUS_RETIRED: CaseStatus = "retired"
#: 件数上限を超えたときに消す順 (それぞれ古い行から)。
EVICTION_ORDER: tuple[CaseStatus, ...] = (STATUS_RETIRED, STATUS_QUARANTINED, STATUS_ACTIVE)
_KNOWN_STATUSES = frozenset(EVICTION_ORDER)

#: ``consistency`` のケースに持たせる直前の会話の上限 (ターン数・文字数)。
PRIOR_TURNS_MAX = 4
PRIOR_TURNS_MAX_CHARS = 2000

#: モデル由来の失敗理由 (``MODEL_ORIGIN_PREFIXES``、表は case_filters が持つ)。プロンプトでは
#: 直らない崩れなので pressure にせず ``quarantined`` で登録する。
#: ``scripts/growth_signal_report.py`` も同じ表を使う。

#: 再生できない理由 (:mod:`backend.free.learning.case_filters` の理由に足すもの)。
REASON_NO_QUERY = "no_query"
REASON_QUERY_TRUNCATED = "query_truncated"

#: 同梱の種ケース。
DEFAULT_SEED_PATH = Path(__file__).resolve().parent / "_defaults" / "regression_seed.jsonl"

#: 回帰ケースの役割 (f_04 §6.1)。
CaseRole = Literal["guard", "pressure", "unknown"]


# ── 形式 ──


@persisted()
@dataclass(kw_only=True)
class PriorTurn:
    """``consistency`` のケースが前置する直前の 1 ターン。"""

    query: str = ""
    response: str = ""
    _extra: dict[str, Any] | None = None


@persisted()
@dataclass(kw_only=True)
class RegressionCase:
    """``regression_cases.jsonl`` の 1 行。``_v`` は行の版 (形式の版と同じ、c_05 §0.5.1)。"""

    _v: int = 1
    written_at: str = ""
    case_id: str
    mode: str = "chat"
    source: str
    #: 再生成に使う問い (本文。切らない — 切れた問いは収穫しない)。
    query: str
    #: 記録したツール結果ブロック (≤ ``TOOL_CONTEXT_CAP``、切られたものは収穫しない)。
    tool_context: str = ""
    prior_turns: list[PriorTurn] = field(default_factory=list)
    #: 検証済みの訂正の正しい値 / 誤った値 (不変則 #12: 検証済みでない span は入れない)。
    expected_value: str = ""
    wrong_value: str = ""
    calculate_unverified: bool = False
    hint: str = ""
    #: 根拠の経験 (弱い参照。ローテーションで宙に浮いたら ``evoref doctor`` が数える)。
    experience_ids: list[str] = field(default_factory=list)
    session_id: str = ""
    dedupe_key: str = ""
    status: CaseStatus = STATUS_ACTIVE
    _extra: dict[str, Any] | None = None

    def to_eval_case(self) -> "PromptEvalCase | None":
        """採用ゲートの評価ケース (``PromptEvalCase``) へ変換する (f_04 §6.5)。

        ``consistency`` は直前の会話を前置して再生する必要があり、``PromptEvalCase`` では
        運べないので ``None`` (再生は ReplayHarness が受け持つ)。
        """
        from backend.free.optimizer.prompt_eval import (
            CASE_KIND_CORRECTION,
            CASE_KIND_FAILED,
            CASE_KIND_SAMPLE,
            CASE_KIND_USER_NEGATIVE,
            PromptEvalCase,
        )

        if self.source == SOURCE_CONSISTENCY or self.prior_turns:
            return None
        kind = {
            SOURCE_VERIFIED_CORRECTION: CASE_KIND_CORRECTION,
            SOURCE_USER_NEGATIVE: CASE_KIND_USER_NEGATIVE,
            SOURCE_VERIFIER_FAILURE: CASE_KIND_FAILED,
            SOURCE_SEED: CASE_KIND_SAMPLE,
        }.get(self.source, CASE_KIND_FAILED)
        return PromptEvalCase(
            case_id=self.case_id,
            query=self.query,
            kind=kind,
            hint=self.hint,
            reference=self.expected_value,
            mode=self.mode,
            expected_value=self.expected_value,
            wrong_value=self.wrong_value,
            tool_context=self.tool_context,
            calculate_unverified=self.calculate_unverified,
            experience_ids=tuple(self.experience_ids),
        )


REGRESSION_CASE_FORMAT = register_format(FormatSpec(
    format_id="learning.regression_case",
    version=1,
    klass="sot",
    writers=frozenset({"free"}),
    path_key="store/learning/shared/growth/regression_cases.jsonl",
    retention=(
        "learning.growth.regression.max_cases (seed cases not counted); "
        "evicts retired, then quarantined, then active, oldest first"
    ),
    export=True,
    encodings=("jsonl",),
    enums={"source": "open", "status": "closed"},
    records=(RegressionCase,),
))


@persisted()
@dataclass(kw_only=True)
class RegressionCaseStatus:
    """1 ケースの再生の記録 (model_key 単位)。"""

    role: CaseRole = "unknown"
    pass_streak: int = 0
    fail_streak: int = 0
    #: 同じ資産の版で合否が反転した (立っている間は guard にも pressure にも使わない)。
    flaky: bool = False
    #: ``pass`` / ``fail`` (未実行は null)。
    last_result: str | None = None
    last_run_at: str | None = None
    _extra: dict[str, Any] | None = None


@persisted()
@dataclass(kw_only=True)
class RegressionStatusFile:
    """``regression_status.json`` のペイロード。"""

    #: ``case_id`` → 再生の記録 (ReplayHarness が書く、PR4)。
    cases: dict[str, RegressionCaseStatus] = field(default_factory=dict)
    #: 収穫済みの経験 ID → 収穫した源 (Step 13.6 の再処理を避ける。経験が消えたら刈る)。
    harvested: dict[str, list[str]] = field(default_factory=dict)
    _extra: dict[str, Any] | None = None


REGRESSION_STATUS_FORMAT = register_format(FormatSpec(
    format_id="learning.regression_status",
    version=1,
    klass="derived",
    writers=frozenset({"free"}),
    path_key="store/learning/<mk>/growth/regression_status.json",
    retention="one per partition; rebuilt by replaying the ledger and re-harvesting",
    export=False,
    records=(RegressionStatusFile,),
))


def _status_file(path: Path) -> VersionedPayloadFile:
    codec = codec_for(RegressionStatusFile)
    return VersionedPayloadFile(
        REGRESSION_STATUS_FORMAT, path, component="RegressionStatus", state_logger=logger,
        decode=codec.decode, encode=codec.encode,
    )


def load_status(path: Path) -> RegressionStatusFile:
    """状態ファイルを読む (無い・読めない・新しい版なら空から。derived なので作り直せる)。"""
    f = _status_file(Path(path))
    if f.load() and isinstance(f.payload, RegressionStatusFile):
        return f.payload
    return RegressionStatusFile()


def save_status(path: Path, status: RegressionStatusFile) -> bool:
    """状態ファイルを書く (原子的。書けたら ``True``)。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    f = _status_file(target)
    f.payload = status
    return f.save()


# ── 鍵と判定 ──


def normalize_text(text: str) -> str:
    """重複の鍵に使う正規化 (NFKC + 空白の畳み込み)。"""
    return " ".join(unicodedata.normalize("NFKC", text or "").split())


def dedupe_key(mode: str, query: str, expected_value: str = "") -> str:
    """``mode`` + 正規化した ``query`` + ``expected_value`` の blake2b (16 hex)。"""
    joined = f"{mode}\n{normalize_text(query)}\n{normalize_text(expected_value)}"
    return hashlib.blake2b(joined.encode("utf-8"), digest_size=8).hexdigest()


def replay_exclusion(exp: Mapping[str, Any]) -> str | None:
    """経験 (dict) を問いだけで再生できない理由を返す (再生できれば ``None``)。

    述語は採用ゲートと同じ :mod:`~backend.free.learning.case_filters` を使う (不変則 #14 (a))。
    近道 (recall / learned) がツールを選んだターンはツール結果の有無に依らず外す (不変則 #15)。
    """
    if not str(exp.get("query") or "").strip():
        return REASON_NO_QUERY
    if exp.get("query_truncated"):
        return REASON_QUERY_TRUNCATED
    signals = exp.get("signals") or {}
    decided_by = signals.get("decided_by") if isinstance(signals, Mapping) else None
    if decided_by_shortcut(decided_by if isinstance(decided_by, dict) else None):
        return REASON_SHORTCUT_TOOL
    return regeneration_mismatch_reason(dict(exp)) or replay_context(dict(exp))[1]


def make_case(
    *,
    source: str,
    query: str,
    mode: str = "chat",
    expected_value: str = "",
    wrong_value: str = "",
    tool_context: str = "",
    calculate_unverified: bool = False,
    prior_turns: Iterable[PriorTurn] = (),
    hint: str = "",
    experience_ids: Iterable[str] = (),
    session_id: str = "",
    status: CaseStatus = STATUS_ACTIVE,
) -> RegressionCase:
    """新しいケースを作る (ID を発番し、重複の鍵を計算する)。"""
    return RegressionCase(
        written_at=utc_now(),
        case_id=new_id(CASE_ID_PREFIX),
        mode=mode,
        source=source,
        query=query,
        tool_context=tool_context,
        prior_turns=list(prior_turns),
        expected_value=expected_value,
        wrong_value=wrong_value,
        calculate_unverified=bool(calculate_unverified),
        hint=hint,
        experience_ids=[i for i in experience_ids if i],
        session_id=session_id,
        dedupe_key=dedupe_key(mode, query, expected_value),
        status=status,
    )


# ── 台帳 ──


_CASE_CODEC = codec_for(RegressionCase)


def _serialize(case: RegressionCase) -> str:
    return json.dumps(_CASE_CODEC.encode(case), ensure_ascii=False)


def load_seed_cases(path: Path = DEFAULT_SEED_PATH) -> list[RegressionCase]:
    """同梱の種ケースを読む (``source=seed`` 以外の行・読めない行は飛ばして件数を WARNING)。"""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    seeds: list[RegressionCase] = []
    skipped = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            case = _CASE_CODEC.decode(json.loads(line))
        except (ValueError, CodecError):
            skipped += 1
            continue
        if case.source != SOURCE_SEED:
            skipped += 1
            continue
        seeds.append(case)
    if skipped:
        logger.warning("Skipped %d unreadable or non-seed row(s) in %s", skipped, path)
    return seeds


class RegressionLedger:
    """回帰台帳の読み書き (:class:`JSONLAppendStore` の上の薄い層)。

    版が新しい行を持つファイルへは書かない (``DataReadonlyError``、c_05 §0.4.5)。readonly 起動
    でも同じ例外になる (呼出側が 0 件で返す)。
    """

    def __init__(
        self,
        path: Path | str,
        *,
        use_seed: bool = True,
        seed_path: Path | str = DEFAULT_SEED_PATH,
        max_cases: int = 200,
    ) -> None:
        self.path = Path(path)
        self.use_seed = use_seed
        self.seed_path = Path(seed_path)
        self.max_cases = max(1, int(max_cases))
        #: 直近の読み込みで飛ばした行 (必須キーの欠け・型の合わない値)。
        self.skipped_rows = 0
        #: 直近の読み込みで使わなかった行 (状態が未知、新しい版が足した値)。
        self.ignored_rows = 0
        self._store: JSONLAppendStore[RegressionCase] = JSONLAppendStore(
            self.path,
            serialize=_serialize,
            deserialize=self._deserialize,
            key_of=lambda case: case.case_id,
            row_version=REGRESSION_CASE_FORMAT.version,
        )

    def _deserialize(self, line: str) -> RegressionCase:
        try:
            return _CASE_CODEC.decode(json.loads(line))
        except CodecError:
            self.skipped_rows += 1
            raise

    # ── 読む ──

    def _all_rows(self) -> list[RegressionCase]:
        """台帳の生きている行すべて (未知の状態の行を含む)。読めない行は飛ばして数える。"""
        self.skipped_rows = 0
        rows = list(self._store.load_all().values())
        if self.skipped_rows:
            logger.warning(
                "Skipped %d regression case row(s) without required keys or with bad values in %s",
                self.skipped_rows, self.path,
            )
        return rows

    def load(self) -> list[RegressionCase]:
        """台帳の生きている行 (種ケースを含まない)。版が新しい行・読めない行は飛ばす。

        状態 (closed 列挙) が未知の行は使わずに数える (``ignored_rows``)。行そのものは
        ストアに残り、compaction でも原形のまま書き戻される (c_05 §0.5.3)。
        """
        rows = self._all_rows()
        known = [c for c in rows if c.status in _KNOWN_STATUSES]
        self.ignored_rows = len(rows) - len(known)
        if self.ignored_rows:
            logger.warning(
                "Kept %d regression case row(s) with an unknown status in %s unchanged; they are not used",
                self.ignored_rows, self.path,
            )
        return known

    def seeds(self) -> list[RegressionCase]:
        """同梱の種ケース (``use_seed`` が偽なら空)。"""
        return load_seed_cases(self.seed_path) if self.use_seed else []

    def cases(self, *, include_seed: bool = True) -> list[RegressionCase]:
        """台帳の行 (+ 種ケース)。"""
        rows = self.load()
        return [*rows, *self.seeds()] if include_seed else rows

    def newer_rows(self) -> int:
        """版がこの台帳より新しい行の数 (1 以上なら書かない)。"""
        return self._store.newer_rows()

    # ── 書く ──

    def add(self, candidates: Iterable[RegressionCase]) -> tuple[list[RegressionCase], int]:
        """重複の鍵で畳んで追記する。``(追加したケース, 重複で畳んだ件数)`` を返す。

        既存の行 (状態を問わず) と種ケース、同じ呼出の中の先の候補と同じ鍵は追加しない。
        追加の後で件数上限を掛ける (:meth:`enforce_limit`)。
        """
        known = {c.dedupe_key for c in (*self._all_rows(), *self.seeds()) if c.dedupe_key}
        added: list[RegressionCase] = []
        duplicates = 0
        for case in candidates:
            key = case.dedupe_key or dedupe_key(case.mode, case.query, case.expected_value)
            if key in known:
                duplicates += 1
                continue
            known.add(key)
            added.append(case if case.dedupe_key else replace(case, dedupe_key=key))
        if added:
            self._store.append_many(added)
            self.enforce_limit()
        return added, duplicates

    def set_status(self, case_id: str, status: CaseStatus) -> bool:
        """ケースの状態を変える (同じ ``case_id`` の行を追記、後勝ち)。変えたら ``True``。"""
        current = next((c for c in self.load() if c.case_id == case_id), None)
        if current is None or current.status == status:
            return False
        self._store.append(replace(current, status=status, written_at=utc_now()))
        return True

    def quarantine(self, case_id: str) -> bool:
        """guard / pressure に使わない状態にする (モデル由来の失敗など)。"""
        return self.set_status(case_id, STATUS_QUARANTINED)

    def retire(self, case_id: str) -> bool:
        """利用者が外したケース (件数上限で最初に消える)。"""
        return self.set_status(case_id, STATUS_RETIRED)

    def _delete(self, case_ids: list[str], reason: str) -> int:
        if not case_ids:
            return 0
        for case_id in case_ids:
            self._store.tombstone(case_id, reason=reason)
        # 問いの本文を持つので tombstone だけで残さず、その場で物理的に消す
        self._store.compact()
        return len(case_ids)

    def enforce_limit(self) -> int:
        """件数上限 (種ケースは数えない) を超えた分を消す。消した件数を返す。

        ``retired`` → ``quarantined`` → ``active`` の順に、それぞれ ``written_at`` の古い行から。
        """
        rows = self.load()
        excess = len(rows) - self.max_cases
        if excess <= 0:
            return 0
        victims: list[str] = []
        for status in EVICTION_ORDER:
            same = [c for c in rows if c.status == status]
            same.sort(key=_written_order)
            for case in same:
                if len(victims) >= excess:
                    break
                victims.append(case.case_id)
        removed = self._delete(victims, "max_cases")
        if removed:
            logger.info("Regression ledger: evicted %d case(s) over max_cases=%d", removed, self.max_cases)
        return removed

    def forget(
        self,
        *,
        case_ids: Iterable[str] = (),
        experience_ids: Iterable[str] = (),
        session_ids: Iterable[str] = (),
    ) -> int:
        """指定したケース・経験・セッションに結び付くケースを消す (forget、f_04 §1.2.0)。"""
        ids, exps, sessions = set(case_ids), set(experience_ids), set(session_ids)
        victims = [
            c.case_id for c in self._all_rows()  # 状態が未知の行も対象 (c_05 §0.5.3)
            if c.case_id in ids or exps.intersection(c.experience_ids)
            or (c.session_id and c.session_id in sessions)
        ]
        return self._delete(victims, "forget")


def _written_order(case: RegressionCase) -> float:
    """``written_at`` の古い順 (読めない時刻は最も古い扱い)。"""
    at = parse_utc(case.written_at)
    return float("-inf") if at is None else at.timestamp()


__all__ = [
    "CASE_ID_PREFIX",
    "DEFAULT_SEED_PATH",
    "EVICTION_ORDER",
    "MODEL_ORIGIN_PREFIXES",
    "PRIOR_TURNS_MAX",
    "PRIOR_TURNS_MAX_CHARS",
    "REASON_NO_QUERY",
    "REASON_QUERY_TRUNCATED",
    "REGRESSION_CASE_FORMAT",
    "REGRESSION_STATUS_FORMAT",
    "SOURCES",
    "SOURCE_CONSISTENCY",
    "SOURCE_SEED",
    "SOURCE_USER_NEGATIVE",
    "SOURCE_VERIFIED_CORRECTION",
    "SOURCE_VERIFIER_FAILURE",
    "STATUS_ACTIVE",
    "STATUS_QUARANTINED",
    "STATUS_RETIRED",
    "PriorTurn",
    "RegressionCase",
    "RegressionCaseStatus",
    "RegressionLedger",
    "RegressionStatusFile",
    "dedupe_key",
    "is_model_origin",
    "load_seed_cases",
    "load_status",
    "make_case",
    "normalize_text",
    "replay_exclusion",
    "save_status",
]
