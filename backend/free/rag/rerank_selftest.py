"""リランカーの自己テスト (c_16 §7.2.1) — PC が変わったときだけ測る。

旧リランカーはスコアが全候補 0.0 / ~1e-22 に退化したまま毎回 2〜3 秒を空費して撤去された
(PR #30/#36 → #105)。再導入では起動したリランカーに既知の日本語の組を送り、

1. 退化していない (関連文書が 1 位・関連 − 無関係の最大 ≥ :data:`MIN_MARGIN` logit・
   全部 0 / 同値でない)
2. 1 件あたりの ms (ウォーム 1 回 + 計測 3 回の中央値)
3. 締切に収まる候補数 (``min(max_candidates, floor(deadline_ms × CANDIDATE_HEADROOM / ms_per_doc))``、
   ``min_candidates`` 未満なら無効)

を決めて ``cache/rerank_selftest.json`` (volatile、c_05 §0.7.1) に残す。

実行するのは起動スクリプト (``scripts/launch_llama.py`` / ``evoref serve``) で、backend は
結果を読むだけ。判定は純関数 (:func:`judge_scores` / :func:`decide_candidates` /
:func:`evaluate_measurement` / :func:`selftest_needed`) で、HTTP は :func:`measure_rerank_server`
の薄い層だけ。

**PC の指紋にモデルや llama.cpp の版は含めない** (PC が変わったときだけ測る)。結果には
モデルの ``model_key`` と llama-server の版を記録だけする。
"""

from __future__ import annotations

import hashlib
import math
import os
import platform
import socket
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx

from backend.free.rag.rerank_llamacpp import (
    RERANK_PATH,
    build_rerank_payload,
    degenerate_reason,
    parse_rerank_response,
)
from backend.io.codec import codec_for, persisted
from backend.io.format_registry import FormatSpec, register_format
from backend.io.versioned import VersionedJsonFile
from backend.log_config import get_logger

logger = get_logger("rag.rerank_selftest")

#: 関連文書のスコアが無関係文書の最大をこの幅 (logit) 以上上回ること。
MIN_MARGIN = 1.0
#: 候補数を決めるときに使う締切の割合。1 回の時間は中央値で ms/件 × 件数に乗るが、p90 は
#: 中央値の約 1.2 倍に伸びる (2026-09-30 実測、GPU・docs のチャンク 360 トークン前後で
#: 候補 10 件 = 締切 1000 ms の 91% を中央値で使い、32% が締切を超えた)。0.8 なら同じ条件で 8 件・超過 1%。
CANDIDATE_HEADROOM = 0.8
#: ウォームアップの回数 (計測に入れない)。
WARM_RUNS = 1
#: 暖機 1 回目が打ち切り時間を超えたときに暖機をやり直す回数 (一時的な遅さで too_slow を保存しない)。
WARM_RETRIES = 1
#: 暖機をやり直す前の待機 (秒)。GPU の解放待ち等の一時的な遅さが抜けるのを待つ。
WARM_RETRY_WAIT_SEC = 3.0
#: 計測の回数 (最小値を採る)。
MEASURE_RUNS = 3
#: 1 回の自己テスト要求の HTTP timeout (秒)。CPU 配置の最悪 (20 件 14 秒) より長く取る。
REQUEST_TIMEOUT_SEC = 60.0

#: 環境起因の失敗 (``server_unhealthy`` / ``http_error:*``) がこの回数続いたら、無効を保存して
#: 次の起動から測り直さない (手動の --rerank-selftest と PC の変化では測り直す)。埋め込みの
#: ``embed_placement.GPU_UNHEALTHY_LIMIT`` と同じ考え方・同じ値。
ENVIRONMENTAL_FAILURE_LIMIT = 3
#: 利用可能な物理メモリが足りなくて自己テストを見送った理由 (保存しない・数えない状態)。
LOW_MEMORY = "low_memory"

Placement = Literal["gpu", "cpu", ""]


# ── 既知の組 (クエリ 1 + 関連 1 + 無関係 4)。関連文書は実際のチャンク長 (300 トークン前後) に寄せる ──

SELFTEST_QUERY = "Python の asyncio で複数のコルーチンを並行に動かし、全部の結果がそろうまで待つにはどうすればよいですか？"

SELFTEST_DOCUMENTS: tuple[str, ...] = (
    # 0: 関連 (これが 1 位でなければ退化)
    "asyncio で複数のコルーチンを同時に進めて、すべての結果を待つ代表的な方法は asyncio.gather です。"
    "results = await asyncio.gather(fetch(a), fetch(b), fetch(c)) のように書くと、三つのコルーチンは"
    "同じイベントループの上で並行に実行され、戻り値は渡した順番のリストとして返ります。どれか一つが例外を"
    "送出すると gather はその例外をそのまま伝えますが、return_exceptions=True を指定すると例外も結果の"
    "一要素として受け取れるので、失敗した処理だけを後から調べられます。Python 3.11 以降では "
    "asyncio.TaskGroup も使えます。async with asyncio.TaskGroup() as tg: の中で tg.create_task() を"
    "呼んでタスクを登録すると、ブロックを抜けるときに全タスクの完了を待ち、どれかが失敗した場合は残りの"
    "タスクを取り消して ExceptionGroup にまとめて送出します。個々の結果は作成したタスクの result() で"
    "取り出します。完了した順に結果を処理したい場合は asyncio.as_completed を、最初の一つだけを待ちたい"
    "場合は asyncio.wait に return_when=FIRST_COMPLETED を渡します。いずれも時間のかかる処理が CPU を"
    "占有しない I/O 待ちであることが前提で、重い計算はスレッドやプロセスに逃がす必要があります。",
    # 1〜4: 無関係 (速度が実際のチャンク長で測れるよう、関連文書と同程度の長さに揃える)
    "味噌汁をおいしく作るこつは、だしを沸騰させすぎないことと、味噌を火を止める直前に溶き入れることです。"
    "昆布は水から入れて弱火でゆっくり温め、沸く直前に取り出します。かつお節は火を止めてから一握り加え、"
    "二分ほど置いて沈んだらこします。煮干しを使う場合は頭と腹わたを取り、三十分ほど水に浸してから火に"
    "かけると苦みが出にくくなります。具は火の通りにくい大根やにんじんなどの根菜から順に入れ、豆腐や"
    "わかめ、ねぎは最後に加えると食感と香りが残ります。味噌は一種類だけでなく、赤味噌と白味噌を合わせると"
    "味に奥行きが出ます。溶き入れた後に再び沸騰させると香りが飛ぶので、温め直すときも煮立たせないように"
    "します。塩分を控えたいときは、だしを濃いめに取って味噌の量を減らし、具を多めにして満足感を補います。"
    "作り置きする場合は、味噌を入れる前の状態で冷蔵し、食べる分だけ温めてから味噌を溶くのがおすすめです。"
    "季節の野菜を取り入れるのも楽しみの一つで、春は菜の花や新たまねぎ、夏はなすやみょうが、秋はきのこや"
    "さつまいも、冬は白菜やかぶがよく合います。仕上げに七味唐辛子や粉山椒、柚子の皮を少し添えると香りが"
    "引き立ちます。だしを取った後の昆布やかつお節は、細かく刻んで醤油とみりんで炒り煮にすれば佃煮やふりかけ"
    "として無駄なく使い切れます。",
    "日本の梅雨は、オホーツク海高気圧と太平洋高気圧の間にできる停滞前線が本州付近にとどまることで起こります。"
    "例年は五月上旬に沖縄で始まり、六月上旬に九州南部、七月中旬ごろに関東甲信で明けますが、年によって"
    "数週間前後します。前線に向かって南から暖かく湿った空気が流れ込むと、積乱雲が次々に発生して同じ場所に"
    "かかり続ける線状降水帯ができ、短時間に記録的な大雨になることがあります。梅雨の終わりごろは特に雨量が"
    "多く、河川の氾濫や土砂災害への警戒が必要です。一方で、梅雨の雨は夏の水資源を支える大切な雨でもあり、"
    "降水量が少ない年は取水制限が行われることもあります。気象庁は梅雨入りと梅雨明けを速報として発表しますが、"
    "秋に天候の経過を振り返って確定値を見直すため、速報とは日付が変わる年も少なくありません。"
    "梅雨の時期は湿度が高く、カビやダニが繁殖しやすいので、室内の換気や除湿機の利用、押し入れの通気に気を"
    "配ると過ごしやすくなります。食品も傷みやすくなるため、作り置きの料理は早めに冷蔵し、弁当には十分に"
    "冷ましてから詰めるのが安全です。北海道には明確な梅雨が無いとされますが、近年は同じ時期に雨が続く"
    "「蝦夷梅雨」と呼ばれる天候が見られることもあります。",
    "サッカーのオフサイドは、味方がボールを蹴った瞬間に、攻撃側の選手が相手陣内でボールと後ろから二人目の"
    "守備側選手よりもゴールラインに近い位置にいて、その後のプレーに関与した場合に反則となります。"
    "オフサイドの位置にいること自体は反則ではなく、プレーに干渉したり、相手の動きを妨げたり、その位置に"
    "いたことで利益を得たりしたときに初めて罰せられます。自陣にいる場合や、ゴールキック・スローイン・"
    "コーナーキックから直接ボールを受けた場合は適用されません。判定は副審が旗で合図し、反則があった地点から"
    "守備側の間接フリーキックで再開します。近年はビデオ・アシスタント・レフェリーが導入され、肩や膝など"
    "得点できる体の部位の位置を映像で確認して判定するようになりましたが、数センチの差で得点が取り消される"
    "ことへの賛否は今も続いています。"
    "オフサイドの規則は十九世紀のイングランドで生まれ、当初は攻撃側の選手より前に三人の守備側選手がいる"
    "ことを求めていましたが、一九二五年に二人へ改められ、得点が大きく増えました。守備側は最終ラインを"
    "そろえて押し上げ、相手をオフサイドの位置に置き去りにする戦術を使い、攻撃側は二列目から飛び出す"
    "動きで対抗します。規則をよく知ると、ラインの駆け引きという試合の見どころが一段と面白くなります。",
    "確定申告の医療費控除は、一年間に自分や生計を一にする家族のために支払った医療費の合計が十万円"
    "(総所得金額等が二百万円未満の人はその五%) を超えた部分について、最高二百万円まで所得から差し引ける"
    "制度です。保険金や高額療養費などで補てんされた金額は差し引いて計算します。対象になるのは診療費や"
    "治療のための医薬品の購入費、通院のための交通費などで、健康診断の費用や美容目的の施術は原則として"
    "対象外です。領収書の提出は不要ですが、医療費控除の明細書を作成して申告書に添付し、領収書は自宅で五年間"
    "保存します。健康保険組合から届く医療費通知を使えば明細の記入を省略できます。また、特定の市販薬の購入費"
    "が一万二千円を超える場合に使えるセルフメディケーション税制もありますが、通常の医療費控除とはどちらか"
    "一方しか選べません。"
    "申告は翌年の二月十六日から三月十五日までが原則ですが、会社員など確定申告の義務が無い人が医療費控除の"
    "ためだけに行う還付申告は、その年の翌年一月一日から五年間提出できます。電子申告を使えば自宅から"
    "手続きでき、マイナポータルと連携すると医療費の情報を自動で取り込めます。控除によって所得税が戻る"
    "だけでなく、翌年の住民税も軽くなるので、該当しそうな年は領収書を捨てずにまとめておくとよいでしょう。",
)
SELFTEST_RELEVANT_INDEX = 0


# ── PC の指紋 ─────────────────────────────────────────────


@persisted()
@dataclass
class PcInfo:
    """指紋の材料。モデルと llama.cpp の版は含めない。"""

    hostname: str = ""
    cpu: str = ""
    logical_cores: int = 0
    memory_gb: int = 0
    gpus: list[str] = field(default_factory=list)
    _extra: dict[str, Any] | None = None

    @property
    def digest(self) -> str:
        """材料から決まる sha256 (hex)。GPU の並び順には依存しない。"""
        parts = [
            self.hostname.strip().lower(),
            self.cpu.strip(),
            str(self.logical_cores),
            str(self.memory_gb),
            *sorted(g.strip() for g in self.gpus),
        ]
        return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def _windows_memory_status() -> tuple[int, int] | None:
    """Windows の (物理メモリの総量, 利用可能量) をバイトで。取れなければ ``None``。"""
    import ctypes

    class _MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
        return None
    return int(status.ullTotalPhys), int(status.ullAvailPhys)


def _read_meminfo() -> str:
    """``/proc/meminfo`` の本文 (テストで差し替える)。"""
    with open("/proc/meminfo", encoding="utf-8") as f:
        return f.read()


def _physical_total_bytes() -> int | None:
    """物理メモリの総量 (バイト)。取れなければ ``None``。"""
    try:
        if sys.platform == "win32":
            mem = _windows_memory_status()
            return mem[0] if mem else None
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None


def _physical_available_bytes() -> int | None:
    """いま利用可能な物理メモリ (バイト)。総量とは別に取る (片方の失敗でもう片方を失わない)。

    Linux は ``/proc/meminfo`` の ``MemAvailable`` (キャッシュの回収分を含む)。読めなければ
    ``SC_AVPHYS_PAGES`` (macOS 等では無い)。どちらも無ければ ``None``。
    """
    try:
        if sys.platform == "win32":
            mem = _windows_memory_status()
            return mem[1] if mem else None
    except (AttributeError, OSError, ValueError):
        return None
    if sys.platform.startswith("linux"):
        try:
            for line in _read_meminfo().splitlines():
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            pass
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_AVPHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None


def _physical_memory_gb() -> int:
    """物理メモリ (GB、四捨五入)。取れなければ 0。"""
    total = _physical_total_bytes()
    return round(total / (1024 ** 3)) if total else 0


def available_physical_memory_mb() -> int:
    """いま利用可能な物理メモリ (MiB)。取れなければ 0 (= 判定しない)。"""
    avail = _physical_available_bytes()
    return avail // (1024 * 1024) if avail else 0


def _cpu_name() -> str:
    """CPU 名 (Windows はレジストリの製品名、無ければ ``platform.processor()``)。"""
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return (platform.processor() or platform.machine() or "").strip()


def collect_pc_info(gpus: Sequence[str]) -> PcInfo:
    """この PC の指紋の材料。``gpus`` は llama-server ``--list-devices`` のデバイス名。"""
    return PcInfo(
        hostname=socket.gethostname(),
        cpu=_cpu_name(),
        logical_cores=os.cpu_count() or 0,
        memory_gb=_physical_memory_gb(),
        gpus=list(gpus),
    )


# ── 判定 (純関数) ─────────────────────────────────────────


@dataclass(frozen=True)
class ScoreVerdict:
    """スコア列の判定。``reason`` は失敗の理由 (成功なら空)。"""

    ok: bool
    reason: str
    margin: float | None


def judge_scores(
    scores: Sequence[float],
    *,
    relevant_index: int = SELFTEST_RELEVANT_INDEX,
    min_margin: float = MIN_MARGIN,
    n_docs: int = len(SELFTEST_DOCUMENTS),
) -> ScoreVerdict:
    """既知の組のスコアが退化していないか。

    失敗の理由: ``bad_response`` (件数違い・非有限値) / ``degenerate_all_zero`` /
    ``degenerate_flat`` / ``relevant_not_top`` / ``margin_too_small``。
    """
    if len(scores) != n_docs or not all(math.isfinite(s) for s in scores):
        return ScoreVerdict(False, "bad_response", None)
    bad = degenerate_reason(scores)
    if bad is not None:
        return ScoreVerdict(False, f"degenerate_{bad}", None)
    relevant = scores[relevant_index]
    others = [s for i, s in enumerate(scores) if i != relevant_index]
    margin = relevant - max(others)
    if margin <= 0:
        return ScoreVerdict(False, "relevant_not_top", margin)
    if margin < min_margin:
        return ScoreVerdict(False, "margin_too_small", margin)
    return ScoreVerdict(True, "", margin)


def ms_per_doc(timings_ms: Sequence[float], n_docs: int) -> float:
    """計測した 1 要求の時間 (ms) の最小値を文書数で割る (起動直後の遅い外れ値で候補数を過小にしない)。"""
    if not timings_ms or n_docs <= 0:
        raise ValueError("no timings to summarize")
    return min(timings_ms) / n_docs


def decide_candidates(
    per_doc_ms: float, *, deadline_ms: int, max_candidates: int, min_candidates: int,
) -> tuple[int, str]:
    """締切に収まる候補数と、無効にする理由 (有効なら空)。

    ``candidates = min(max_candidates, floor(deadline_ms × CANDIDATE_HEADROOM / per_doc_ms))``。
    ``min_candidates`` 未満なら ``(候補数, "too_slow")`` — 候補数は記録のために返す。
    """
    if per_doc_ms <= 0:
        return max_candidates, ""
    fit = math.floor(deadline_ms * CANDIDATE_HEADROOM / per_doc_ms)
    candidates = min(max_candidates, fit)
    if candidates < min_candidates:
        return candidates, "too_slow"
    return candidates, ""


@dataclass
class Measurement:
    """rerank サーバへの自己テスト要求の生の結果 (HTTP 層の出力)。"""

    scores: list[float] = field(default_factory=list)
    timings_ms: list[float] = field(default_factory=list)
    prompt_tokens: int | None = None
    #: 要求そのものが失敗した理由 (``http_500`` 等)。成功なら空。
    error: str = ""
    #: 暖機 1 回が打ち切り時間を超えたので計測を省いた (``timings_ms`` は暖機の 1 回だけ)。
    aborted_slow: bool = False


@dataclass(frozen=True)
class SelftestVerdict:
    """1 つの配置での自己テストの判定。"""

    enabled: bool
    reason: str
    ms_per_doc: float | None
    candidates: int
    margin: float | None
    #: CPU で 1 回だけ再試行する価値がある失敗か (退化・サーバ失敗。遅すぎは再試行しない)。
    retry_on_cpu: bool


def evaluate_measurement(
    measurement: Measurement, *, deadline_ms: int, max_candidates: int, min_candidates: int,
    n_docs: int = len(SELFTEST_DOCUMENTS),
) -> SelftestVerdict:
    """測った結果を判定する (退化 → 速度 → 候補数の順)。"""
    if measurement.error:
        return SelftestVerdict(False, measurement.error, None, 0, None, retry_on_cpu=True)
    if measurement.aborted_slow:
        per_doc = ms_per_doc(measurement.timings_ms, n_docs)
        candidates, _ = decide_candidates(
            per_doc, deadline_ms=deadline_ms,
            max_candidates=max_candidates, min_candidates=min_candidates,
        )
        return SelftestVerdict(False, "too_slow", per_doc, candidates, None, retry_on_cpu=False)
    verdict = judge_scores(measurement.scores, n_docs=n_docs)
    if not verdict.ok:
        return SelftestVerdict(False, verdict.reason, None, 0, verdict.margin, retry_on_cpu=True)
    per_doc = ms_per_doc(measurement.timings_ms, n_docs)
    candidates, reason = decide_candidates(
        per_doc, deadline_ms=deadline_ms,
        max_candidates=max_candidates, min_candidates=min_candidates,
    )
    return SelftestVerdict(
        not reason, reason, per_doc, candidates, verdict.margin, retry_on_cpu=False,
    )


# ── 結果の永続化 ──────────────────────────────────────────


@persisted()
@dataclass
class RerankSelftestResult:
    """自己テスト 1 回の結果 (``cache/rerank_selftest.json`` の payload)。"""

    fingerprint: str
    enabled: bool
    #: 無効の理由 (``too_slow`` / ``degenerate_flat`` / ``server_unhealthy`` 等)。有効なら空。
    reason: str = ""
    placement: Placement = ""
    gpu_layers: int = 0
    threads: int = 0
    ms_per_doc: float | None = None
    candidates: int = 0
    margin: float | None = None
    scores: list[float] = field(default_factory=list)
    prompt_tokens: int | None = None
    tested_at: str = ""
    #: 環境起因の失敗 (:func:`is_environmental_failure`) が同じ PC で続けて起きた回数 (この回を含む)。
    #: :data:`ENVIRONMENTAL_FAILURE_LIMIT` に届くまで次の起動でまた測り、届いたら無効を確定する。
    environmental_streak: int = 0
    #: 記録だけ (指紋には入れない)。
    model_key: str = ""
    model_file: str = ""
    llama_server_version: str = ""
    pc: PcInfo = field(default_factory=PcInfo)
    _extra: dict[str, Any] | None = None


#: 形式 (c_05 §0.7.1)。PC 固有の測定値で、測り直せる (derived) が高価 (起動し直して測る)
#: ので ``evoref reset`` でも残す (``--include-cache`` まで)。
RERANK_SELFTEST_FORMAT = register_format(FormatSpec(
    format_id="cache.rerank_selftest",
    version=1,
    klass="derived",
    writers=frozenset({"free"}),
    path_key="cache/rerank_selftest.json",
    retention="one record, rewritten when the PC fingerprint or an explicit gpu_layers changes (or on a manual re-run)",
    export=False,
    keep_on_reset=True,
    records=(RerankSelftestResult,),
))


class RerankSelftestFile(VersionedJsonFile):
    """``cache/rerank_selftest.json`` の読み書き (volatile: 読めなければ捨てて測り直す)。"""

    FORMAT = RERANK_SELFTEST_FORMAT
    _state_logger = logger

    def __init__(self, path: Path | str) -> None:
        super().__init__(path)
        self.result: RerankSelftestResult | None = None

    def _to_payload(self) -> Any:
        if self.result is None:
            raise ValueError("no rerank selftest result to save")
        return codec_for(RerankSelftestResult).encode(self.result)

    def _from_payload(self, payload: Any) -> None:
        self.result = codec_for(RerankSelftestResult).decode(payload)


def load_selftest_result(path: Path | str) -> tuple[RerankSelftestResult | None, str]:
    """保存済みの結果と、読んだときの分類 (``absent`` / ``current`` / ``corrupt`` 等)。"""
    store = RerankSelftestFile(path)
    if store.load():
        return store.result, str(store.last_status or "current")
    return None, str(store.last_status or "absent")


def save_selftest_result(path: Path | str, result: RerankSelftestResult) -> bool:
    """結果を書く (AtomicWriter)。書けたら ``True``。"""
    store = RerankSelftestFile(path)
    store.result = result
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return store.save()


def selftest_needed(
    saved: RerankSelftestResult | None,
    fingerprint: str,
    *,
    force: bool = False,
    explicit_gpu_layers: int | None = None,
) -> tuple[bool, str]:
    """自己テストを走らせるか、とその理由。

    理由: ``forced`` / ``no_result`` / ``fingerprint_changed`` / ``placement_setting_changed`` /
    ``environmental_retry`` (環境起因の失敗を数えている途中。上限に届いたら測り直さない)。
    読めない結果 (新しい版・壊れ) は ``saved=None`` で渡る (``no_result``)。

    **再テストは PC が変わったときだけ** — 唯一の例外が明示の ``gpu_layers`` (整数) で、
    保存済みの配置と食い違えば測り直す (明示の設定は常に優先する、c_16 §7.2.1)。
    ``auto`` (``explicit_gpu_layers=None``) なら保存済みの配置をそのまま使う。
    """
    if force:
        return True, "forced"
    if saved is None:
        return True, "no_result"
    if saved.fingerprint != fingerprint:
        return True, "fingerprint_changed"
    if explicit_gpu_layers is not None and saved.gpu_layers != explicit_gpu_layers:
        return True, "placement_setting_changed"
    if (
        not saved.enabled and is_environmental_failure(saved.reason)
        and saved.environmental_streak < ENVIRONMENTAL_FAILURE_LIMIT
    ):
        return True, "environmental_retry"
    return False, ""


def is_environmental_failure(reason: str) -> bool:
    """環境起因の失敗 (``server_unhealthy`` / ``http_error:*`` / ``low_memory``) か。

    PC の性質ではないので、すぐには無効として確定しない。``low_memory`` は保存も数えもせず
    次の起動でまた試す。``server_unhealthy`` / ``http_error:*`` は :data:`ENVIRONMENTAL_FAILURE_LIMIT`
    回続いたら無効を保存する (:func:`next_environmental_streak`)。
    """
    return reason in ("server_unhealthy", LOW_MEMORY) or reason.startswith("http_error:")


def next_environmental_streak(saved: RerankSelftestResult | None, fingerprint: str) -> int:
    """今回の環境起因の失敗を数えた連続回数 (同じ PC で環境起因の失敗を数えている記録があれば +1)。"""
    prior = 0
    if (
        saved is not None and saved.fingerprint == fingerprint and not saved.enabled
        and is_environmental_failure(saved.reason) and saved.reason != LOW_MEMORY
    ):
        prior = saved.environmental_streak
    return max(0, prior) + 1


def warm_budget_ms(deadline_ms: int, min_candidates: int, n_docs: int = 0) -> float:
    """暖機 1 回の打ち切り時間 (ms)。超えたら ``min_candidates`` 件すら締切に収まらない。"""
    n = n_docs or len(SELFTEST_DOCUMENTS)
    return deadline_ms * n / max(1, min_candidates)


def effective_candidates(
    saved: RerankSelftestResult,
    *,
    deadline_ms: int,
    max_candidates: int,
    min_candidates: int,
) -> tuple[bool, int, str]:
    """保存済みの ``ms_per_doc`` と **現在の** 締切・候補数から (有効か, 候補数, 理由) を引き直す。

    再テストはしない。退化などスコアが原因の無効はそのまま。``too_slow`` で保存された
    結果も、設定を緩めれば有効になる。
    """
    speed_only = saved.enabled or saved.reason == "too_slow"
    if not speed_only or saved.ms_per_doc is None:
        return saved.enabled, saved.candidates, saved.reason
    candidates, reason = decide_candidates(
        saved.ms_per_doc, deadline_ms=deadline_ms,
        max_candidates=max_candidates, min_candidates=min_candidates,
    )
    return not reason, candidates, reason


def pc_mismatch(saved: PcInfo, current: PcInfo) -> bool:
    """subprocess 無しで取れる部分 (ホスト名・CPU・コア数・メモリ) が食い違うか。

    backend は ``--list-devices`` を呼ばないので GPU 名は比べない。ホスト名は大小無視。
    """
    return (
        saved.hostname.strip().lower() != current.hostname.strip().lower()
        or saved.cpu.strip() != current.cpu.strip()
        or saved.logical_cores != current.logical_cores
        or saved.memory_gb != current.memory_gb
    )


# ── HTTP (薄い層) ─────────────────────────────────────────


def measure_rerank_server(
    base_url: str,
    *,
    post: Callable[..., httpx.Response] = httpx.post,
    clock: Callable[[], float] = time.perf_counter,
    warm_runs: int = WARM_RUNS,
    measure_runs: int = MEASURE_RUNS,
    timeout: float = REQUEST_TIMEOUT_SEC,
    warm_budget: float | None = None,
    warm_retries: int = WARM_RETRIES,
    sleep: Callable[[float], None] = time.sleep,
    warm_retry_wait: float = WARM_RETRY_WAIT_SEC,
) -> Measurement:
    """起動済みの rerank サーバへ既知の組を ``warm_runs + measure_runs`` 回送る (各 1 回、再試行しない)。

    スコアは最後の計測の値。失敗した時点で ``error`` を入れて返す。暖機 1 回目が
    ``warm_budget`` (ms、:func:`warm_budget_ms`) を超えたら暖機を ``warm_retries`` 回だけ
    待機してからやり直し (GPU の解放待ち等の一時的な遅さで ``too_slow`` を保存しない)、やり直しも超えたら計測を
    省いて ``aborted_slow`` で返す (遅い CPU で自己テストの最悪時間を膨らませない)。
    """
    url = f"{base_url}{RERANK_PATH}"
    payload = build_rerank_payload(SELFTEST_QUERY, SELFTEST_DOCUMENTS)
    result = Measurement()
    retries_left = warm_retries
    run = 0
    while run < warm_runs + measure_runs:
        started = clock()
        try:
            resp = post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            scores, prompt_tokens = parse_rerank_response(resp.json(), len(SELFTEST_DOCUMENTS))
        except httpx.HTTPStatusError as e:
            result.error = f"http_{e.response.status_code}"
            return result
        except httpx.HTTPError as e:
            result.error = f"http_error:{type(e).__name__}"
            return result
        except ValueError:
            result.error = "bad_response"
            return result
        elapsed_ms = (clock() - started) * 1000.0
        result.scores = scores
        result.prompt_tokens = prompt_tokens
        if run >= warm_runs:
            result.timings_ms.append(elapsed_ms)
        elif run == 0 and warm_budget is not None and elapsed_ms > warm_budget:
            if retries_left > 0:
                retries_left -= 1
                sleep(warm_retry_wait)
                continue
            result.timings_ms = [elapsed_ms]
            result.aborted_slow = True
            return result
        run += 1
    return result



# ── backend 側: 保存結果から状態を決める ─────────────────────


@dataclass(frozen=True)
class RerankStatus:
    """``/api/status`` に出す rerank の状態 (backend は結果ファイルを読むだけ)。"""

    mode: str = "off"
    enabled: bool = False
    placement: str = ""
    ms_per_doc: float | None = None
    candidates: int = 0
    #: 無効の理由 (``off`` / ``no_model`` (``model_paths.rerank_model`` 未設定) /
    #: ``model_missing`` (設定したファイルが無い) / ``not_tested`` / ``stale_fingerprint`` /
    #: 自己テストの理由 / ``server_unreachable``)。``no_model`` / ``model_missing`` は
    #: 既定 on でモデルを置いていない環境の正常な状態 (失敗ではない)。
    reason: str = ""
    tested_at: str | None = None
    #: 自己テストの後に ``model_paths.rerank_model`` が変わった (再テストはしない、警告だけ)。
    model_changed_since_selftest: bool = False


def rerank_model_unavailable_reason(model_rel: object, model_exists: bool) -> str:
    """リランカーのモデルが使えない理由 (純関数)。使えれば空文字。

    ``model_rel`` は ``model_paths.rerank_model`` の生の値、``model_exists`` はそれを
    解決したパスにファイルがあるか (``model_rel`` が空なら見ない)。
    未設定 (``None`` / 空) → ``no_model``、ファイルが無い → ``model_missing``。
    """
    if not model_rel:
        return "no_model"
    if not model_exists:
        return "model_missing"
    return ""


def resolve_rerank_status(
    mode: str,
    saved: RerankSelftestResult | None,
    *,
    deadline_ms: int = 1000,
    max_candidates: int = 20,
    min_candidates: int = 3,
    current_pc: PcInfo | None = None,
    current_model_key: str | None = None,
) -> RerankStatus:
    """設定と保存結果から状態を決める (純関数、health は見ない)。

    - 結果が無い / 読めない → **無効** (``not_tested``)。未テストの候補数では動かさない。
    - ``current_pc`` と保存時の PC が食い違う → 無効 (``stale_fingerprint``)。
    - 候補数は保存済みの ``ms_per_doc`` と現在の締切・候補数の設定で引き直す。
    - モデルが変わっていても再テストはしない (``model_changed_since_selftest`` で知らせる)。
    """
    if mode == "off":
        return RerankStatus(mode="off", reason="off")
    if saved is None:
        return RerankStatus(mode=mode, reason="not_tested")
    model_changed = bool(
        current_model_key and saved.model_key and current_model_key != saved.model_key,
    )
    common = {
        "mode": mode, "placement": saved.placement, "ms_per_doc": saved.ms_per_doc,
        "tested_at": saved.tested_at or None, "model_changed_since_selftest": model_changed,
    }
    if current_pc is not None and pc_mismatch(saved.pc, current_pc):
        return RerankStatus(enabled=False, candidates=0, reason="stale_fingerprint", **common)
    enabled, candidates, reason = effective_candidates(
        saved, deadline_ms=deadline_ms,
        max_candidates=max_candidates, min_candidates=min_candidates,
    )
    return RerankStatus(enabled=enabled, candidates=candidates, reason=reason, **common)

__all__ = [
    "CANDIDATE_HEADROOM",
    "MIN_MARGIN",
    "RERANK_SELFTEST_FORMAT",
    "SELFTEST_DOCUMENTS",
    "SELFTEST_QUERY",
    "Measurement",
    "PcInfo",
    "RerankSelftestFile",
    "RerankSelftestResult",
    "RerankStatus",
    "ScoreVerdict",
    "SelftestVerdict",
    "collect_pc_info",
    "decide_candidates",
    "effective_candidates",
    "ENVIRONMENTAL_FAILURE_LIMIT",
    "LOW_MEMORY",
    "available_physical_memory_mb",
    "is_environmental_failure",
    "next_environmental_streak",
    "evaluate_measurement",
    "judge_scores",
    "load_selftest_result",
    "measure_rerank_server",
    "ms_per_doc",
    "pc_mismatch",
    "rerank_model_unavailable_reason",
    "resolve_rerank_status",
    "save_selftest_result",
    "selftest_needed",
    "warm_budget_ms",
]
