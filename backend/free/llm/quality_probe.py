"""モデル切替時の出力品質プローブ (base / aux / embedding)

``capability.py`` のプローブが観測するのは **形式** (reasoning 分離 / ``<think>`` /
json_schema 強制) だけで、**出力そのものの質**は一切見ない。この穴を突いた実例:

    2026-08-02 に base を ``Qwen3.5-9B-Q4_K_M`` → ``gemma-4-12b-it-Q4_K_M`` へ
    切替。日本語の語間空白の混入率が 1% → 76〜83% に悪化したが、起動は正常・
    capability probe も正常で、**27 ターンの会話を回すまで誰も気付かなかった**。

本モジュールは切替を検知したときだけカナリアを投げ、決定論の判定器で劣化を
可視化する。**起動は止めない** (degraded 安全 / モデル選択はユーザーの裁量) —
出すのは WARNING と ``/api/status`` の記録で、判断材料を先に渡すのが役割。

判定に補助タスクを使わないのは意図的。小型モデルは自分と同種の崩れを問題と
認識できず、実測で正常例と崩れ例の採点差が 0.09 しか付かなかった
(:mod:`backend.free.core.text_quality` 参照)。

プローブ内容は役割ごとに異なる:

===========  =========================================================
role         検査
===========  =========================================================
base         日本語の生成品質 (語間空白の混入率 / 日本語で答えているか)
aux       同上 (要約・digest がユーザーと SemMem に直接届くため)
embedding    埋め込み空間の健全性 (類似ペアが非類似ペアより近いか)
===========  =========================================================
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from backend.io.atomic import atomic_write_text
from backend.free.core.text_quality import has_broken_ja_spacing, is_japanese_text
from backend.log_config import get_logger
from backend.utils import utc_now

logger = get_logger("llm.quality_probe")

#: プローブ対象の役割。``model_paths`` のキーではなく論理名で扱う。
QUALITY_ROLES: tuple[str, ...] = ("base", "aux", "embedding")

#: 日本語生成のカナリア。
#:
#: 「日本語で 2〜3 文」と長さを絞るのは、語間空白が**文の途中**に現れる現象で、
#: 短すぎると母数が足りず長すぎると iGPU で切替のたびに数分待たされるため。
#: 内容は事実性を問わない一般的な題材にする — ここで測るのは表記であって
#: 知識ではなく、モデル固有の知識差でノイズを増やしたくない。
DEFAULT_JA_PROBE_PROMPTS: tuple[str, ...] = (
    "和食の特徴を日本語で 2〜3 文で説明してください。",
    "季節の移り変わりについて日本語で 2〜3 文で書いてください。",
    "図書館の役割を日本語で 2〜3 文で説明してください。",
    "朝の散歩の良さを日本語で 2〜3 文で書いてください。",
    "手紙と電子メールの違いを日本語で 2〜3 文で説明してください。",
    "山と海のどちらが好きか、日本語で 2〜3 文で書いてください。",
)

#: 埋め込み空間のカナリア ``(anchor, positive, negative)``。
#:
#: ``positive`` は anchor の言い換え、``negative`` は明確に別話題。埋め込みが
#: 機能していれば ``sim(a, p) > sim(a, n)`` が成り立つ。同一 dim のモデルへ
#: 差し替えると ``dimension_check`` は素通りするため、**順序関係で検査する**。
#:
#: negative は話題だけでなく **ドメインごと** 変える。当初 3 組目の negative を
#: 「コーヒーの淹れ方のコツ」にしたところ、健全な LFM2.5-Embedding-350M でも
#: マージンが 0.0509 しか出ず閾値 0.05 をかすめた (実測)。旅程案内も淹れ方も
#: 「短い how-to 句」で構造が似ており、モデルの欠陥ではなく **カナリアの設計不良**
#: だった。ドメインを跨いだ現行の組では最小マージンが 0.1988 (閾値の約 4 倍) 出る。
DEFAULT_EMBED_PROBES: tuple[tuple[str, str, str], ...] = (
    (
        "今日の天気を教えてください",
        "本日の気象情報が知りたいです",
        "Python で関数を定義する方法",
    ),
    (
        "ファイルをディスクに保存する",
        "データをストレージへ書き出す",
        "明日の会議の予定を確認したい",
    ),
    (
        "電車の乗り換え案内",
        "鉄道の乗り継ぎを調べたい",
        "この関数の戻り値の型は何ですか",
    ),
)

#: 語間空白の許容混入率。
#:
#: 実測は Qwen3.5-9B が 1% / gemma-4-12b が 76〜83% と 2 桁離れており、中間に
#: 広い無人地帯がある。カナリア 6 件なら 1 件混入 (16.7%) は通し 2 件 (33.3%)
#: で落ちる — 健全なモデルの偶発 1 件で騒がず、恒常的な崩れは確実に捕らえる。
DEFAULT_JA_SPACE_MAX_RATE = 0.25

#: 日本語で答えたカナリアの最低比率。日本語を求めたのに英語で返すモデルは
#: 語間空白が原理的に 0% になるため、これを見ないと「合格」に化ける。
DEFAULT_JA_RESPONSE_MIN_RATE = 0.5

#: ``sim(anchor, positive) - sim(anchor, negative)`` の最低マージン。
DEFAULT_EMBED_MARGIN = 0.05

#: プローブ 1 件あたりの生成上限。2〜3 文には十分で、暴走を抑える。
_PROBE_MAX_TOKENS = 200

#: raw OAI ``/v1/chat/completions`` 応答 JSON を返す呼び出し関数 (capability と同型)。
ChatFn = Callable[[dict], Awaitable[dict]]
#: テキスト列 → 埋め込みベクトル列。
EmbedFn = Callable[[Sequence[str]], Awaitable[Sequence[Sequence[float]]]]


@dataclass(frozen=True)
class CheckResult:
    """個別チェックの結果。``observed`` / ``threshold`` は比較可能な数値のみ。"""

    name: str
    passed: bool
    observed: float | None = None
    threshold: float | None = None
    detail: str = ""


@dataclass
class QualityProbeResult:
    """役割 1 つ分のプローブ結果。``local/model_quality.json`` に永続化される。"""

    role: str
    model: str
    passed: bool = True
    checks: list[CheckResult] = field(default_factory=list)
    known_issues: list[str] = field(default_factory=list)
    probed_at: str = ""
    skipped_reason: str = ""

    @property
    def failed_checks(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "model": self.model,
            "passed": self.passed,
            "checks": [asdict(c) for c in self.checks],
            "known_issues": list(self.known_issues),
            "probed_at": self.probed_at,
            "skipped_reason": self.skipped_reason,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "QualityProbeResult":
        return cls(
            role=data.get("role", ""),
            model=data.get("model", ""),
            passed=bool(data.get("passed", True)),
            checks=[
                CheckResult(
                    name=c.get("name", ""),
                    passed=bool(c.get("passed", True)),
                    observed=c.get("observed"),
                    threshold=c.get("threshold"),
                    detail=c.get("detail", ""),
                )
                for c in (data.get("checks") or [])
            ],
            known_issues=list(data.get("known_issues") or []),
            probed_at=data.get("probed_at", ""),
            skipped_reason=data.get("skipped_reason", ""),
        )


# ─────────────────────────────────────────────────────────────────────────
# プロファイル解決
# ─────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class QualityBaseline:
    """``models/profiles/<arch>.yaml`` の ``quality_baseline:`` ブロック。

    ブロックが無い arch は全て既定値。閾値だけをプロファイル側に置くのは、
    カナリア本文を 7 ファイルへ複製すると**片方だけ直る**ためで、prompts の
    上書きは「そのモデルで既定カナリアが機能しない」と分かった時の逃げ道。
    """

    ja_interword_space_max_rate: float = DEFAULT_JA_SPACE_MAX_RATE
    ja_response_min_rate: float = DEFAULT_JA_RESPONSE_MIN_RATE
    embed_similarity_margin: float = DEFAULT_EMBED_MARGIN
    ja_probe_prompts: tuple[str, ...] = DEFAULT_JA_PROBE_PROMPTS
    embed_probes: tuple[tuple[str, str, str], ...] = DEFAULT_EMBED_PROBES
    skip_checks: frozenset[str] = frozenset()
    known_issues: tuple[str, ...] = ()


def resolve_quality_baseline(profile: dict | None) -> QualityBaseline:
    """arch プロファイルから :class:`QualityBaseline` を解決する (純粋関数)。

    未知キーは黙って無視する — プロファイルはユーザーが ``local/profiles/`` で
    上書きでき、綴り違いで起動を落としたくない。型が合わないキーも既定へ倒す。
    """
    block = (profile or {}).get("quality_baseline")
    if not isinstance(block, dict):
        return QualityBaseline()

    def _num(key: str, default: float) -> float:
        raw = block.get(key, default)
        try:
            return float(raw)
        except (TypeError, ValueError):
            logger.warning(
                "quality_baseline.%s is not a number (%r); using default %s",
                key, raw, default,
            )
            return default

    prompts = block.get("ja_probe_prompts")
    resolved_prompts = (
        tuple(str(p) for p in prompts if str(p).strip())
        if isinstance(prompts, list) and prompts
        else DEFAULT_JA_PROBE_PROMPTS
    )

    raw_embed = block.get("embed_probes")
    resolved_embed: tuple[tuple[str, str, str], ...] = DEFAULT_EMBED_PROBES
    if isinstance(raw_embed, list) and raw_embed:
        triples = [
            (str(t[0]), str(t[1]), str(t[2]))
            for t in raw_embed
            if isinstance(t, (list, tuple)) and len(t) == 3
        ]
        if triples:
            resolved_embed = tuple(triples)

    skip = block.get("skip_checks")
    known = block.get("known_issues")
    return QualityBaseline(
        ja_interword_space_max_rate=_num(
            "ja_interword_space_max_rate", DEFAULT_JA_SPACE_MAX_RATE,
        ),
        ja_response_min_rate=_num(
            "ja_response_min_rate", DEFAULT_JA_RESPONSE_MIN_RATE,
        ),
        embed_similarity_margin=_num(
            "embed_similarity_margin", DEFAULT_EMBED_MARGIN,
        ),
        ja_probe_prompts=resolved_prompts,
        embed_probes=resolved_embed,
        skip_checks=frozenset(str(s) for s in skip) if isinstance(skip, list) else frozenset(),
        known_issues=tuple(str(k) for k in known) if isinstance(known, list) else (),
    )


# ─────────────────────────────────────────────────────────────────────────
# 判定 (サーバ非依存の純粋関数)
# ─────────────────────────────────────────────────────────────────────────


def evaluate_ja_responses(
    responses: Sequence[str], baseline: QualityBaseline,
) -> list[CheckResult]:
    """日本語カナリアの応答群を評価する (純粋関数)。

    2 つを別チェックにするのは、片方だけでは抜けるため:

    - ``ja_response_rate`` — そもそも日本語で答えているか。英語で返すモデルは
      語間空白が原理的に 0% になり、これが無いと満点で通ってしまう
    - ``ja_interword_space`` — 日本語で答えた応答のうち崩れている比率
    """
    checks: list[CheckResult] = []
    usable = [r for r in responses if r.strip()]
    if not usable:
        return [CheckResult(
            name="ja_response_rate", passed=False, observed=0.0,
            threshold=baseline.ja_response_min_rate,
            detail="全カナリアが空応答",
        )]

    japanese = [r for r in usable if is_japanese_text(r)]
    ja_rate = len(japanese) / len(usable)
    if "ja_response_rate" not in baseline.skip_checks:
        checks.append(CheckResult(
            name="ja_response_rate",
            passed=ja_rate >= baseline.ja_response_min_rate,
            observed=round(ja_rate, 3),
            threshold=baseline.ja_response_min_rate,
            detail=f"{len(japanese)}/{len(usable)} 件が日本語で応答",
        ))

    if "ja_interword_space" not in baseline.skip_checks and japanese:
        broken = [r for r in japanese if has_broken_ja_spacing(r)]
        rate = len(broken) / len(japanese)
        checks.append(CheckResult(
            name="ja_interword_space",
            passed=rate <= baseline.ja_interword_space_max_rate,
            observed=round(rate, 3),
            threshold=baseline.ja_interword_space_max_rate,
            detail=f"{len(broken)}/{len(japanese)} 件に語間空白",
        ))
    return checks


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """コサイン類似度。**必ず Python の float を返す**。

    入力が numpy 配列だと各演算が numpy スカラを返し、そこから導いた
    ``passed`` が ``numpy.bool_`` に、``observed`` が ``numpy.float32`` になる。
    どちらも ``json.dumps`` が拒否するため、結果を保存する段になって初めて
    ``Object of type bool is not JSON serializable`` で落ちる (実機で踏んだ:
    プローブ自体は成功していたのに記録だけが消えていた)。境界でこうして
    落とし切る。
    """
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(dot / (na * nb))


def evaluate_embed_vectors(
    triples: Sequence[tuple[Sequence[float], Sequence[float], Sequence[float]]],
    baseline: QualityBaseline,
    *,
    expected_dim: int | None = None,
) -> list[CheckResult]:
    """埋め込みカナリアのベクトル群を評価する (純粋関数)。

    ``embed_dim`` は config の ``embedding.dim`` との一致、``embed_ordering`` は
    「言い換えの方が別話題より近い」順序関係。同一 dim へ差し替えると
    ``dimension_check`` は素通りするため、後者が本命の検査になる。
    """
    checks: list[CheckResult] = []
    if not triples:
        return [CheckResult(
            name="embed_ordering", passed=False,
            detail="埋め込みカナリアの結果が空",
        )]

    if expected_dim is not None and "embed_dim" not in baseline.skip_checks:
        actual = int(len(triples[0][0]))
        checks.append(CheckResult(
            name="embed_dim",
            passed=bool(actual == expected_dim),
            observed=float(actual),
            threshold=float(expected_dim),
            detail=f"実測 dim={actual} / config embedding.dim={expected_dim}",
        ))

    if "embed_ordering" in baseline.skip_checks:
        return checks

    margins = [
        _cosine(anchor, positive) - _cosine(anchor, negative)
        for anchor, positive, negative in triples
    ]
    worst = min(margins)
    checks.append(CheckResult(
        name="embed_ordering",
        passed=worst >= baseline.embed_similarity_margin,
        observed=round(worst, 4),
        threshold=baseline.embed_similarity_margin,
        detail=(
            f"{sum(1 for m in margins if m >= baseline.embed_similarity_margin)}"
            f"/{len(margins)} 件で言い換えが別話題より近い (最小マージン {worst:.4f})"
        ),
    ))
    return checks


# ─────────────────────────────────────────────────────────────────────────
# 実行
# ─────────────────────────────────────────────────────────────────────────


def _message_content(resp: dict) -> str:
    try:
        return (resp["choices"][0]["message"].get("content") or "").strip()
    except (KeyError, IndexError, TypeError, AttributeError):
        return ""


def _reasoning_consumed_budget(resp: dict) -> bool:
    """本文が空で、思考が生成上限を食い潰したかを判定する。

    reasoning モデルに ``max_tokens`` だけ与えて投げると、思考で上限に達して
    ``content`` が空のまま ``finish_reason="length"`` で返る。これを「応答なし」と
    しか報告しないと、モデルが壊れているのか予算が足りないのか切り分けられない
    (実際に切り分けに 1 往復かかった)。
    """
    try:
        choice = resp["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError):
        return False
    return bool(
        not (message.get("content") or "").strip()
        and (message.get("reasoning_content") or "").strip()
        and choice.get("finish_reason") == "length",
    )


async def probe_text_quality(
    *,
    role: str,
    model: str,
    chat_fn: ChatFn,
    baseline: QualityBaseline,
    enable_thinking: bool | None = None,
) -> QualityProbeResult:
    """base / aux の日本語生成品質を観測する。

    ``enable_thinking`` は **本番のチャットパスと同じ値**を渡すこと。ここで測るのは
    ユーザーが実際に受け取る本文の質で、設定が違えば別のものを測ってしまう。
    ``None`` のときは送らない (非 thinking モデルへ送ると llama-server の版に
    よっては 400 になるため。``LocalClient._build_payload`` と同じ扱い)。

    個々のカナリア失敗は握りつぶして残りで判定する (1 件のタイムアウトで
    プローブ全体を落とさない)。全滅した場合のみ ``skipped_reason`` を立てて
    ``passed=True`` で返す — **観測できなかったことを不合格にはしない**。
    """
    responses: list[str] = []
    errors = 0
    budget_exhausted = 0
    for prompt in baseline.ja_probe_prompts:
        payload: dict = {
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": 0.0,
            "max_tokens": _PROBE_MAX_TOKENS,
        }
        if enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
        try:
            resp = await chat_fn(payload)
            responses.append(_message_content(resp))
            if _reasoning_consumed_budget(resp):
                budget_exhausted += 1
        except Exception as exc:
            errors += 1
            logger.debug("quality probe canary failed (%s): %s", role, exc)

    if not [r for r in responses if r.strip()]:
        if budget_exhausted:
            reason = (
                f"思考が生成上限 ({_PROBE_MAX_TOKENS} tokens) を使い切り本文が"
                f"空になった ({budget_exhausted} 件)。品質は未観測"
            )
        else:
            reason = f"全カナリアが失敗または空応答 (errors={errors})"
        return QualityProbeResult(
            role=role, model=model, passed=True, probed_at=utc_now(),
            known_issues=list(baseline.known_issues),
            skipped_reason=reason,
        )

    checks = evaluate_ja_responses(responses, baseline)
    return QualityProbeResult(
        role=role,
        model=model,
        passed=all(c.passed for c in checks),
        checks=checks,
        known_issues=list(baseline.known_issues),
        probed_at=utc_now(),
    )


async def probe_embed_quality(
    *,
    model: str,
    embed_fn: EmbedFn,
    baseline: QualityBaseline,
    expected_dim: int | None = None,
) -> QualityProbeResult:
    """埋め込みモデルの空間健全性を観測する。"""
    texts: list[str] = []
    for anchor, positive, negative in baseline.embed_probes:
        texts.extend((anchor, positive, negative))

    try:
        vectors = await embed_fn(texts)
    except Exception as exc:
        logger.debug("embed quality probe failed: %s", exc)
        return QualityProbeResult(
            role="embedding", model=model, passed=True, probed_at=utc_now(),
            known_issues=list(baseline.known_issues),
            skipped_reason=f"埋め込み取得に失敗: {exc}",
        )

    # ``EmbeddingBackend.embed`` は numpy 配列を返す。``not vectors`` と書くと
    # 「要素が複数ある配列の真偽値は曖昧」で ValueError になるため、必ず len() で
    # 判定する (list を返すフェイクでは通ってしまい実機で初めて落ちた)。
    obtained = len(vectors) if vectors is not None else 0
    if obtained != len(texts):
        return QualityProbeResult(
            role="embedding", model=model, passed=True, probed_at=utc_now(),
            known_issues=list(baseline.known_issues),
            skipped_reason=(
                f"埋め込み件数が不一致 (要求 {len(texts)} / 取得 {obtained})"
            ),
        )

    triples = [
        (vectors[i], vectors[i + 1], vectors[i + 2])
        for i in range(0, len(vectors), 3)
    ]
    checks = evaluate_embed_vectors(triples, baseline, expected_dim=expected_dim)
    return QualityProbeResult(
        role="embedding",
        model=model,
        passed=all(c.passed for c in checks),
        checks=checks,
        known_issues=list(baseline.known_issues),
        probed_at=utc_now(),
    )


# ─────────────────────────────────────────────────────────────────────────
# 永続化 + 切替検知
# ─────────────────────────────────────────────────────────────────────────


class QualityProbeStore:
    """``local/model_quality.json`` の読み書きと「切替されたか」の判定。

    プローブを毎起動走らせない (iGPU では 6 カナリア × 3 役割で分単位かかる)。
    記録済みモデル名と現在のモデル名を突き合わせ、**変わったときだけ**走らせる。

    モデル名で比較するのは、``model_state.json`` の migrate 記録と独立に効かせる
    ため。config.yaml を直接書き換えた / 別マシンから local/ を持ち込んだ場合でも
    「前回プローブしたモデルと違う」ことは同じく検知できる。
    """

    def __init__(self, path: Path):
        self.path = path
        self._records: dict[str, QualityProbeResult] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load model_quality.json: %s", exc)
            return
        for role, raw in (data.get("roles") or {}).items():
            if isinstance(raw, dict):
                self._records[role] = QualityProbeResult.from_dict(raw)

    def get(self, role: str) -> QualityProbeResult | None:
        return self._records.get(role)

    def needs_probe(self, role: str, model: str) -> bool:
        """``role`` の現行モデルが未プローブ (初回 / 切替後 / 観測失敗) かを返す。

        ``model`` が空 (未設定 / 解決不能) のときは False。何を測るか決まって
        いない状態でカナリアを投げても記録が汚れるだけで判断材料にならない。

        ``skipped_reason`` が立った記録は**未検査として扱い、次の起動で再試行する**。
        観測できなかったことは検査済みではない — サーバが暖まる前 / 生成予算が
        足りない等の一時要因で空振りしたとき、そこで確定させると二度と測らなく
        なってしまう (実機で踏んだ: 思考が生成上限を食い潰して全カナリアが空になり、
        その「合格」が記録されて再プローブが走らなくなった)。
        """
        if not model:
            return False
        record = self._records.get(role)
        if record is None or record.model != model:
            return True
        return bool(record.skipped_reason)

    def record(self, result: QualityProbeResult) -> None:
        self._records[result.role] = result
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "roles": {
                role: res.to_dict() for role, res in self._records.items()
            },
        }
        atomic_write_text(
            self.path,
            json.dumps(payload, ensure_ascii=False, indent=2),
        )

    def summary(self) -> list[dict]:
        """``/api/status`` 向けの要約 (役割順で安定)。"""
        return [
            self._records[role].to_dict()
            for role in QUALITY_ROLES
            if role in self._records
        ]


def log_probe_result(result: QualityProbeResult) -> None:
    """プローブ結果をログに出す (英語固定)。

    不合格でも起動は止めない。モデル選択はユーザーの裁量で、こちらの仕事は
    「黙って劣化させない」こと — 判断材料を先に渡すところまで。
    """
    if result.skipped_reason:
        logger.info(
            "Quality probe skipped: role=%s model=%s reason=%s",
            result.role, result.model, result.skipped_reason,
        )
        return
    if result.passed:
        logger.info(
            "Quality probe passed: role=%s model=%s checks=%s",
            result.role, result.model,
            ",".join(f"{c.name}={c.observed}" for c in result.checks),
        )
        return
    for check in result.failed_checks:
        logger.warning(
            "Quality probe FAILED: role=%s model=%s check=%s "
            "observed=%s threshold=%s (%s)",
            result.role, result.model, check.name,
            check.observed, check.threshold, check.detail,
        )
    for issue in result.known_issues:
        logger.warning(
            "Known issue for %s model %s: %s",
            result.role, result.model, issue,
        )


__all__ = [
    "DEFAULT_EMBED_PROBES",
    "DEFAULT_JA_PROBE_PROMPTS",
    "QUALITY_ROLES",
    "CheckResult",
    "QualityBaseline",
    "QualityProbeResult",
    "QualityProbeStore",
    "evaluate_embed_vectors",
    "evaluate_ja_responses",
    "log_probe_result",
    "probe_embed_quality",
    "probe_text_quality",
    "resolve_quality_baseline",
]
