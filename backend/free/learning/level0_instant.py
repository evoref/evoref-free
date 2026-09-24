"""Level 0 即時学習: 経験バッファ"""

import os
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, get_args

from backend.free.core.correction_verdict import VERDICT_CODES, VerdictCode
from backend.free.learning.fitness import DEFECT_WEIGHTS, signal_is_defect
from backend.io import jsoncodec
from backend.io.codec import CodecError, codec_for, persisted
from backend.io.format_registry import FormatSpec, register_format
from backend.io.id_registry import new_id
from backend.io.readonly import DataReadonlyError
from backend.io.writer_thread import ChatWriter, default_writer
from backend.log_config import get_logger
from backend.utils import utc_now

logger = get_logger("learning.level0")

MAX_ENTRIES = 1000

#: バッファ表示・集計用の応答要約の最大長 (従来の response[:200])。
RESPONSE_SUMMARY_CAP = 200
#: few-shot 採用時に保持する全文応答の上限。response_summary (200字) では
#: 文の途中で切れた応答が few-shot 例として注入されるため、採用候補向けに
#: より長い全文を文境界で切り詰めて保持する。experience.json (experience.jsonl) の肥大を避けるため
#: 青天井にはしない (cap × MAX_ENTRIES が永続化サイズの上限)。
RESPONSE_FULL_CAP = 4000

#: ターン成否の語彙 (``FeedbackSignals.turn_outcome``)。台帳では ``open`` の列挙。
TurnOutcome = Literal["success", "partial", "failed"]
TURN_OUTCOMES: frozenset[str] = frozenset(get_args(TurnOutcome))


def _has_defect(signals: "FeedbackSignals") -> bool:
    """共有の重み表 (:data:`DEFECT_WEIGHTS`) のどれかが立っているか。

    ``vars()`` で十分 (``FeedbackSignals`` はスカラだけのフラットな dataclass で、
    ``asdict`` の再帰コピーは要らない)。
    """
    raw = vars(signals)
    return any(signal_is_defect(raw, key) for key in DEFECT_WEIGHTS)


def used_corpus_evidence(experience: dict) -> bool:
    """そのターンに corpus 由来の材料を **注入したか** (c_16 §5.5)。

    ``gen_config.evidence_ids`` は ``"<store>:<evidence_id>"`` 形式で、
    ``corpus:`` が 1 件でもあれば ``[参考情報]`` 枠に文書チャンクが載っている。
    旧 ``signals.rag_used`` は「検索が何か返したか」でしかなく、フロア / gate /
    top_k で全部落ちたターンも 1 と数えていた。読み手は Level 1 の
    ``rag_usage_rate`` と few-shot の根拠ゲート。
    """
    gen_config = experience.get("gen_config")
    if not isinstance(gen_config, dict):
        return False
    return any(
        isinstance(eid, str) and eid.startswith("corpus:")
        for eid in (gen_config.get("evidence_ids") or ())
    )


def truncate_at_boundary(text: str, cap: int) -> str:
    """``cap`` 字以内で文境界 (。．.!?改行) を優先して切り詰める。

    cap 未満は無加工。末尾付近 (cap*0.6 以降) に文境界があればそこで切り、
    無ければハード cap。few-shot 応答が文の途中でぶつ切りになるのを避ける。
    """
    if len(text) <= cap:
        return text
    head = text[:cap]
    floor = int(cap * 0.6)
    best = -1
    for ch in ("。", "．", ".", "!", "?", "\n"):
        idx = head.rfind(ch)
        if idx >= floor and idx > best:
            best = idx
    return head[: best + 1] if best >= floor else head


@persisted()
@dataclass
class FeedbackSignals:
    """暗黙的フィードバックシグナル"""
    conversation_ended: bool = False
    # ターン成否の SSOT: "success" | "partial" | "failed"。
    # response の [failed] マーカー / step_credits 全 0 / ルーティング
    # false_positive から FeedbackCollector が決定論導出する。
    # tool_routing_success / long_form_success と矛盾する場合は failed 側に
    # 倒し、偽成功が Level 1 の正例学習へ伝播しないようにする。
    turn_outcome: TurnOutcome = "success"
    #: ``turn_outcome`` の導出理由 (``FeedbackCollector._derive_turn_outcome_with_reason``
    #: の文字列)。few-shot の curate が「手本に帰属できる失敗か」を理由で
    #: 選別する (f_04 §3.2.2)。旧レコードは None。
    turn_outcome_reason: str | None = None
    rephrased_query: bool = False
    rag_used: bool = False
    rag_top1_score: float | None = None
    agent_loops: int = 0
    user_correction: str | None = None
    """**検証済み** の訂正発話本文 (学習側の唯一の入口)。

    記録時には立たない。字句検出は ``correction_candidate`` に入り、
    ``learning.correction_verifier`` が「直前のアシスタント応答の誤りを
    指しているか」を補助タスクで判定して初めてここへ昇格する
    (2026-09-08 監査 F-03: 100 ターンで立った 2 件がどちらも偽陽性で、
    その 2 件だけが Level 1 採用ゲートの評価ケースになった)。
    """
    # "hardcoded" | "prev_failed" | "same_target" | None。旧 "learned"
    # (学習パターン照合) は 2026-07-21 廃止 — 過去データには残存しうる
    correction_detected_by: str | None = None
    correction_candidate: str | None = None
    """字句検出が拾った訂正 **候補** の発話本文 (recall 側)。

    除外規則を通した後の候補で、精度の責任は負わない。チャット応答パスの
    軽い用途 (遡及 false_negative マーク / 数値の保留判定 / few-shot の
    除外) はこちらを見る。学習の目的関数は ``user_correction`` だけを見る。
    """
    correction_verified_at: str | None = None
    """``correction_candidate`` を検証した時刻 (ISO 8601 UTC)。冪等性の鍵。"""
    correction_verdict: VerdictCode | None = None
    """検証結果 (語彙は :data:`~backend.free.core.correction_verdict.VerdictCode`)。``assistant`` / ``self`` / ``third_party`` / ``premise_change``
    / ``none`` (LLM の判定) と、コード側で付ける ``no_context`` (直前応答を
    解決できない) / ``invalid_span`` (逐語検証に落ちた) / ``no_verdict``
    (補助タスクが空を返した)。``assistant`` のみが昇格する。"""
    correction_wrong_claim: str | None = None
    """直前応答のうち誤っていた逐語 span (検証済み)。"""
    correction_correct_value: str | None = None
    """訂正発話が示した正しい値の逐語 span (検証済み)。

    eval_core / 訂正ペアの期待語はここを優先する。字句の否定境界
    (``ではなく`` / ``じゃなく``) だけに頼ると **誤り側** が期待語に載る
    (2026-09-07 監査 F-01)。"""
    # この訂正が指す **誤っていたターン** の ``ExperienceEntry.id``。
    #
    # 訂正ペア (「元の問い → 訂正後の正しい回答」) を組むのに要る対応関係で、
    # **記録時にしか確定できない** (セッションが分かっていて、直近ターンの
    # 応答本文が手元にある)。以前はこれを残さず、後段の
    # ``learning.corrected_pairs`` がバッファの直前エントリを元の問いとみなして
    # 再導出していた。バッファは全セッション横断の 1 本なので、別会話の訂正が
    # 隣り合うと問いと訂正が食い違う (2026-09-06 ライブ監査 F-01: 訂正 7 件が
    # few-shot にも eval_core にも 1 件も入らなかった)。
    #
    # 同定は ``core.correction_target.resolve_correction_target`` (訂正文が
    # 引用する値・識別子と応答本文の重なり) で行う。旧データには無いので
    # ``None`` を許容し、消費側はセッション単位のフォールバックを持つ。
    corrected_entry_id: str | None = None
    # アシスタント自身が応答冒頭で前ターンの誤りを撤回したか
    # (「失礼いたしました」「訂正します」等)。ユーザーの字句に依らない
    # 高確度シグナルで、**誤っていたのは 1 つ前のターン**。検出時は
    # FeedbackCollector が直前エントリの turn_outcome を failed へ落とす。
    #
    # user_correction とは別枠にする — あちらは「ユーザーの訂正発話そのもの」
    # を保持し critique_synthesizer が本文を引用するため、ユーザーが訂正して
    # いないターンのクエリを入れると引用が破綻する (2026-08-05 ライブ監査で
    # 訂正検出 0/40。アシスタントが「失礼いたしました」と撤回したターンすら
    # 検出されていなかった)。
    assistant_self_retraction: bool = False
    perplexity: float | None = None
    # ── コストシグナル (2026-08-18 配線) ──
    # 「取得件数・予算を増やすほど品質指標が上がる」単調パラメータは、代償が
    # fitness に現れない限り最適化器が必ず制約の端まで膨らませる。実際に
    # search.top_k は上限を 50→10 へ切る対症療法が入り、long_form の
    # unit_target_tokens は下限を 128→512 に上げている。恒久的な対処には
    # コストを観測項として持つ必要があるため、既に一次情報が取れている 3 つを
    # 記録する。
    #
    # completion_tokens は生成トークン数 (呼出側が既に保持している厳密値)。
    # prompt_tokens / cached_prompt_tokens は llama-server の usage 由来で、
    # 再プリフィル量 = prompt_tokens - cached_prompt_tokens。取得不能な構成では
    # いずれも None (0 ではない = 「計測できなかった」と「消費ゼロ」を区別する)。
    completion_tokens: int | None = None
    prompt_tokens: int | None = None
    cached_prompt_tokens: int | None = None
    # ── 生成の結末 (2026-09-02 配線) ──
    # truncated: llama-server が finish_reason=length を返した = 応答が
    # max_tokens 到達で文の途中で切れている。切れた応答は few-shot の手本に
    # 採らない (FewShotPool.add_from_experiences)。
    truncated: bool = False
    # generation_failed: ユーザーへ 1 文字も届かなかった / error フレームで
    # 終わったターン。response は空で記録され turn_outcome は failed。
    # 以前はこのターンが記録されず、失敗が選択圧に一件も入らなかった。
    generation_failed: bool = False
    # 長文生成シグナル
    long_form_used: bool = False
    long_form_content_type: str | None = None    # "code" | "text"
    long_form_strategy: str | None = None        # "cogwriter" | "recurrent"
    long_form_units_total: int = 0
    long_form_units_completed: int = 0
    long_form_validation_errors: int = 0
    long_form_budget_used_pct: float | None = None
    # ツールルーティングシグナル
    tool_routing_success: bool = False
    tool_routing_false_positive: bool = False
    tool_routing_false_negative: bool = False
    #: 応答がツール実行結果 (calculate / run_command / search_history …) を
    #: 根拠にしたか。ツール由来の値は **その問いの数値** であって文体の手本では
    #: なく、手本に載ると同じ形の問いにツールを撃たず手本の値を復唱する。
    #: ツール種別に依らず「プロンプトにツール実行結果が注入されていた」で
    #: 判定する (2026-09-10 ライブ監査 (f) F-09: calculate 由来の「45 km」が
    #: few-shot に採用された。tool_routing_* は run_command しか見ていない)。
    tool_grounded: bool = False
    #: 文書を注入した turn で「参考情報には記載が無い」型の抑止応答をしたか
    #: (f_04 §3.2、2026-09-12 (b))。``None`` = 文書を注入していない (判定外)、
    #: ``False`` = 注入して答えた、``True`` = 注入したのに差し控えた =
    #: 検索の取りこぼしの観測。プロンプト進化の圧には使わない。
    rag_abstained: bool | None = None
    #: 注入した turn で応答が ``[参考情報]`` を明示的に引いたか (2026-09-14)。
    #: ``None`` = 検索が何も注入していない。abstained と対で「注入が答えに
    #: 使われたか」を測る材料 (09-12: 引用 19/94、差し控え 13/74)。
    rag_cited: bool | None = None
    #: ユーザーの明示評価 (2026-09-14、f_04 §3.2.3)。``None`` = 評価なし、
    #: ``True`` = 👎 (本人が失敗と言った唯一の信号。採用ゲートのケースに最優先で
    #: 使い、手本にはしない)、``False`` = 👍。``user_note`` は任意の一言で、
    #: 訂正文と同じく judge のヒントになる。
    user_negative: bool | None = None
    user_note: str = ""
    #: 根拠台帳 (f_04 §2.2、2026-09-10 (h))。``tool_uses`` は tool_ledger 由来の
    #: 実行順 ``[{"tool", "success", "reason"}]``。3 つの疑義は ``None`` = ツール
    #: 判定を通っていない (reactive 経路等)、``[]`` / ``False`` = 判定してクリーン。
    #: 「未判定」と「クリーン」を混ぜない (c_05 §0.5)。
    tool_uses: list[dict[str, Any]] = field(default_factory=list)
    unexplained_numbers: list[str] | None = None
    expression_issues: list[str] | None = None
    unexplained_date_math: bool | None = None
    # 長文ルーティングシグナル (router._detect_long_form の学習用)
    # success: 長文分類が成功し generation 完了 → 該当キーワードを強化 + 学習
    # false_positive: long_form 分類されたが短文応答で十分だった → 該当キーワードを減衰
    # false_negative: deliberative 分類されたがユーザが「長文で」等で再要求 → 新キーワード学習
    long_form_success: bool = False
    long_form_false_positive: bool = False
    long_form_false_negative: bool = False
    # MDP ステップクレジット
    step_credits: list[dict[str, Any]] = field(default_factory=list)
    #: この版が知らないシグナル (書き戻しで元の位置へ戻す、c_05 §0.5.2)。
    _extra: dict[str, Any] | None = None


@persisted()
@dataclass
class GenerationConfigRef:
    """その応答を生んだ構成の参照 (c_05 §0.6 ID 連鎖)。

    Level 1 / Level 2 は経験の fitness を候補 (プロンプト版 / few-shot /
    ポリシー / LoRA) へ帰属させるが、以前は **何が有効だったかを記録して
    いなかった** ため、帰属は時刻からの推測でしかなかった (2026-09-05 監査)。
    """

    prompt_version: int | None = None
    """このターンで使ったシステムプロンプトの版 (``PromptMeta.version``)。"""

    fewshot_ids: list[str] = field(default_factory=list)
    """注入した few-shot 例の ID (``FewShotExample.id``)。"""

    policy_generation: int | None = None
    """ポリシーパラメータの世代 (``PolicyParamEvolver.generation``。パーティション内で
    params を動かすたびに 1 進む通番)。"""

    adapters: list[dict[str, Any]] = field(default_factory=list)
    """有効だったアダプタ ``[{"kind": "lora" | "cvector", "version": int}]``。

    記録のみ (帰属の読み手はまだ無い)。llama-server が実際に載せたアダプタは起動時に
    決まり、backend はその版を追跡していないので、今は常に空で記録する。"""

    locale: str = ""
    """UI ロケール。プロンプト本文と few-shot の言語を決める軸。"""

    corpus_gated: bool | None = None
    """疑似クエリの関連性ゲート (f_01 §6.6) で corpus を引かなかった turn なら
    ``True``。検索を通っていない turn (軽量パス等) は ``None`` (0 で埋めない)。
    Level 1 の「注入ゼロ」を無関係 / 較正 / 未充足で区別する材料。"""

    pseudo_derived_count: int = 0
    """採用した corpus チャンクのうち疑似クエリ索引経由で拾った件数。"""

    sampling: dict[str, Any] = field(default_factory=dict)
    """temperature / top_p / max_tokens 等、生成時に効いたパラメータ (自由形、opaque)。"""

    evidence_ids: list[str] = field(default_factory=list)
    """このターンで **実際に注入した** Evidence の id (c_16 §5.5)。

    形式は ``"<store>:<evidence_id>"`` で、``store`` は ``episodic`` /
    ``semantic`` / ``corpus``。``[参考情報]`` 枠 (統合検索) と
    ``[関連する記憶]`` 枠 (``MemoryInjector``) の両方を含む。

    廃止した ``ExperienceEntry.cartridge_ids`` は「そのとき **ロードされて
    いた** カートリッジ一覧」で、実際に見せた材料とは無関係だった。Level 1 の
    ``rag_usage_rate`` はここに ``corpus:`` があるかで数える。
    """

    template: str = ""
    """このターンに **実際に適用した** 文書テンプレートの来歴鍵
    (``"<package_id>@<version>:<entry_id>"``、c_16 §4.5.2 / c_05 §0.6)。

    体裁の継承 (``write_file`` が ``WriteResult.metadata`` に載せた場合) /
    構成テンプレートによる計画の seed (``plan_seeded``) / 帳票の穴埋め
    (書込み成功時) のいずれかで適用されたターンだけ埋まる。選ばれただけで
    未適用のターンは空文字のまま (``None`` で埋めない既定と同じ扱い)。
    """

    _extra: dict[str, Any] | None = None


@persisted()
@dataclass
class ExperienceEntry:
    """経験バッファの1エントリ"""

    id: str = ""
    """このエントリの ID (``exp_<hex12>``)。**同じターンの二重記録を検出できる
    唯一の手段**。以前は秒精度の timestamp しか無く、同一秒の 2 ターンが
    区別できず、再送・再取り込みの重複も検出できなかった。"""

    session_id: str = ""
    """発生した会話のセッション ID。"""

    turn_id: str = ""
    """発生したターンの ID (``WorkingMemory.add_turn`` が発行)。"""

    trace_id: str = ""
    """リクエストの trace_id。JSONL 側のログと突き合わせる鍵。"""

    timestamp: str = ""
    mode: str = "chat"
    query: str = ""
    response_summary: str = ""
    # few-shot 採用例用の全文応答 (文境界で RESPONSE_FULL_CAP に切り詰め)。
    # 表示・集計・cvector は response_summary を使い、本フィールドは
    # fewshot_pool.add_from_experiences が切れていない応答例を採るためだけに使う。
    response_full: str = ""
    base_model: str = ""
    """応答を生んだモデルの表示名 (GGUF のファイル名)。照合には使わない。"""
    model_key: str | None = None
    """応答を生んだモデルの ``model_key`` (c_05 §0.5.7。create は create_model の key)。
    Level 2 の経験の絞り込みはこれで行う。"""
    embedding_model: str = ""
    lang: str = ""
    """応答本文の言語 (``ja`` / ``en`` / 未判定は空)。決定論判定で埋める。"""

    gen_config: GenerationConfigRef = field(default_factory=GenerationConfigRef)
    """この応答を生んだ構成 (fitness の帰属先)。"""

    signals: FeedbackSignals = field(default_factory=FeedbackSignals)

    #: この版が知らないキー (各階層の未知キーはその階層の ``_extra``)。
    _extra: dict[str, Any] | None = None

    @staticmethod
    def new_id() -> str:
        """``exp_<hex16>`` 形式の ID を発行する (ID 台帳、c_05 §0.5.5)。"""
        return new_id("exp_")


EXPERIENCE_FORMAT = register_format(FormatSpec(
    format_id="learning.experience",
    version=1,
    klass="sot",
    writers=frozenset({"free"}),
    path_key="store/learning/<mk>/experience.jsonl",
    retention="max_entries (1000), trimmed by compaction",
    export=True,
    encodings=("jsonl",),
    enums={"turn_outcome": "open", "correction_verdict": "open"},
    records=(ExperienceEntry,),
))

#: 行の版 (c_05 §0.5.1 の ``_v``)。形式の版と同じ。
ROW_VERSION = EXPERIENCE_FORMAT.version
#: patch 行の ``op``。これ以外 (``op`` 無し) は記録の行。
PATCH_OP = "patch"
#: 階層ごとに畳む入れ子 (patch の ``fields`` はこの 2 つを 1 段だけ merge する)。
_NESTED = ("gen_config", "signals")
_ENTRY_CODEC = codec_for(ExperienceEntry)
_TOP_FIELDS = tuple(f.name for f in _ENTRY_CODEC.fields if f.name not in _NESTED)
_GEN_CONFIG_FIELDS = tuple(f.name for f in codec_for(GenerationConfigRef).fields)
_SIGNAL_FIELDS = tuple(f.name for f in codec_for(FeedbackSignals).fields)
#: ``record`` のたびに差分を見る直近の件数 (前ターンへの遡及の印・自己撤回・
#: 保留した訂正の確定は、どれも直近のエントリに付く)。それより古いエントリの
#: 変更は :meth:`ExperienceBuffer.touch` か、全件を見る :meth:`ExperienceBuffer.flush`。
_RECENT_WINDOW = 8


def _copied(value: Any) -> Any:
    """list / dict の値は浅く複写する (差分の基準がメモリ上の値と同じ実体を持たない)。"""
    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        return dict(value)
    return value


def _level_dict(obj: Any, names: tuple[str, ...]) -> dict:
    """1 階層を dict にする (既知フィールド + ``_extra`` をその階層のトップへ)。

    ``asdict`` の深い複写はしない (c_05 §0.5.2)。シグナルと構成参照の中身は
    スカラーか、スカラー / dict の list だけ。
    """
    out = {name: _copied(getattr(obj, name)) for name in names}
    extra = obj._extra
    if extra:
        for key, value in extra.items():
            if key not in out:
                out[key] = value
    return out


def entry_to_dict(entry: ExperienceEntry) -> dict:
    """1 エントリの永続形 (``gen_config`` / ``signals`` は入れ子の dict。未知キーは各階層のトップ)。"""
    out = {name: getattr(entry, name) for name in _TOP_FIELDS}
    out["gen_config"] = _level_dict(entry.gen_config, _GEN_CONFIG_FIELDS)
    out["signals"] = _level_dict(entry.signals, _SIGNAL_FIELDS)
    extra = entry._extra
    if extra:
        for key, value in extra.items():
            if key not in out:
                out[key] = value
    return out


def entry_from_dict(data: dict) -> ExperienceEntry:
    """永続形 (記録の dict) を ``ExperienceEntry`` へ読む。読めなければ :class:`CodecError`。"""
    return _ENTRY_CODEC.decode(data)


def _level_unchanged(obj: Any, names: tuple[str, ...], persisted: Any, *, nested: int = 0) -> bool:
    """1 階層が最後に出した永続形と同じか (複写しない比較。``nested`` は別に比べる入れ子の数)。"""
    if not isinstance(persisted, dict):
        return False
    extra = obj._extra or {}
    if len(persisted) != len(names) + nested + sum(1 for k in extra if k not in names):
        return False
    return all(getattr(obj, n) == persisted.get(n, _MISSING) for n in names) and all(
        persisted.get(k, _MISSING) == v for k, v in extra.items() if k not in names
    )


def _unchanged(entry: ExperienceEntry, persisted: dict) -> bool:
    """``entry`` が最後に出した永続形 ``persisted`` と同じか (複写しない比較)。"""
    return (
        _level_unchanged(entry, _TOP_FIELDS, persisted, nested=len(_NESTED))
        and _level_unchanged(entry.gen_config, _GEN_CONFIG_FIELDS, persisted.get("gen_config"))
        and _level_unchanged(entry.signals, _SIGNAL_FIELDS, persisted.get("signals"))
    )


def _diff(old: dict, new: dict) -> dict:
    """``old`` → ``new`` の変更だけを patch の ``fields`` にする。"""
    fields_: dict = {}
    for key, value in new.items():
        if key in _NESTED:
            sub = {k: v for k, v in value.items() if old.get(key, {}).get(k, _MISSING) != v}
            if sub:
                fields_[key] = sub
        elif old.get(key, _MISSING) != value:
            fields_[key] = value
    return fields_


_MISSING = object()


def _row_line(record: dict) -> str:
    return jsoncodec.dumps({"_v": ROW_VERSION, **record})


def rows_text(records: Iterable[dict]) -> str:
    """記録の dict の列を JSONL の本文にする (コンパクション・別パスへの保存)。"""
    return "".join(_row_line(r) + "\n" for r in records)


def _patch_line(entry_id: str, fields_: dict) -> str:
    return jsoncodec.dumps({"_v": ROW_VERSION, "op": PATCH_OP, "id": entry_id, "fields": fields_})


def _apply_patch(record: dict, fields_: dict) -> None:
    for key, value in fields_.items():
        if key in _NESTED and isinstance(value, dict):
            nested = record.get(key)
            if not isinstance(nested, dict):
                nested = record[key] = {}
            nested.update(value)
        else:
            record[key] = value


def unknown_enums(record: dict) -> tuple[str, ...]:
    """記録の dict が持つ未知の列挙値の名前 (空なら学習に使える、c_05 §0.5.3)。

    未知の値の行は **原形のまま保持して使わない** (few-shot・fitness・Level 1 / 2 の
    選択・訂正ペア・件数上限の対象外)。``turn_outcome`` の null も未知として扱う —
    既定値 (success) へ寄せると、読めない成否が成功として数えられる。
    """
    signals = record.get("signals")
    if not isinstance(signals, dict):
        return ()
    out: list[str] = []
    outcome = signals.get("turn_outcome", "success")
    if not (isinstance(outcome, str) and outcome in TURN_OUTCOMES):
        out.append("turn_outcome")
    verdict = signals.get("correction_verdict")
    if verdict is not None and not (isinstance(verdict, str) and verdict in VERDICT_CODES):
        out.append("correction_verdict")
    return tuple(out)


def fold_experience_file(path: Path | str) -> tuple[list[dict], dict[str, int]]:
    """JSONL を id ごとに畳んで記録の dict を並べて返す (記録の行の順)。

    patch 行はその id の記録へ (``gen_config`` / ``signals`` は 1 段だけ) merge する。
    途中で切れた行・NUL・壊れた JSON・知らない版の行は飛ばして数える
    (c_05 §0.5.2 / §0.5.8)。宛先の無い patch も数える。

    Returns:
        ``(records, stats)``。``stats`` は ``lines`` / ``bad`` / ``newer`` / ``orphan``。
    """
    records: dict[str, dict] = {}
    anonymous = 0
    stats = {"lines": 0, "bad": 0, "newer": 0, "orphan": 0}
    try:
        raw = Path(path).read_bytes()
    except FileNotFoundError:
        return [], stats
    for line in raw.split(b"\n"):
        if not line.strip():
            continue
        stats["lines"] += 1
        if b"\x00" in line:
            stats["bad"] += 1
            continue
        try:
            obj = jsoncodec.loads(line)
        except (ValueError, UnicodeDecodeError):
            stats["bad"] += 1
            continue
        if not isinstance(obj, dict):
            stats["bad"] += 1
            continue
        version = obj.pop("_v", 1)
        if not isinstance(version, int) or version > ROW_VERSION:
            stats["newer"] += 1
            continue
        if obj.get("op") == PATCH_OP:
            target = records.get(obj.get("id") or "")
            fields_ = obj.get("fields")
            if target is None or not isinstance(fields_, dict):
                stats["orphan"] += 1
                continue
            _apply_patch(target, fields_)
            continue
        entry_id = obj.get("id") or ""
        if not entry_id:
            # id 以前の行は patch の宛先にならない (上書きもしない)。
            anonymous += 1
            entry_id = f"\x00anon{anonymous}"
        # 同じ id の記録がもう 1 度来たら後勝ち (位置は最初のまま)。
        records[entry_id] = obj
    return list(records.values()), stats


def compacted_body(path: Path, max_entries: int) -> tuple[str, int]:
    """``path`` を畳み、直近 ``max_entries`` 件の記録だけの本文と行数を返す。

    書き手スレッドで呼ぶ (この操作より前の追記は全て書かれている) ので、メモリ上の
    バッファを直列化し直さない。未知の列挙値の行は件数に数えず、全て原形のまま残す
    (c_05 §0.5.3)。
    """
    records, stats = fold_experience_file(path)
    if stats["newer"]:
        # 新しい版の行を落として書き戻さない (c_05 §0.4.5)。
        raise RuntimeError(f"{path} has {stats['newer']} row(s) of a newer version; not compacting")
    known = [r for r in records if not unknown_enums(r)]
    trimmed = {id(r) for r in known[:-max_entries]} if max_entries > 0 else set()
    kept = [r for r in records if id(r) not in trimmed]
    return rows_text(kept), len(kept)


class ExperienceBuffer:
    """経験バッファ: 毎応答時にエントリを記録し、JSONL へ追記する (c_05 §5.3)。

    - ``record()`` は記録の行を 1 行、**書き手スレッド** (c_05 §0.5.9) へ出すだけ。
      G0 は毎ターン ``done`` の前にバッファ全体 (最大 1000 件・約 7〜9MB) を
      置き換えており、上限時に 1 ターン約 60ms ループを止めていた。
    - 後からの変更 (前ターンへの遡及の印・``mark_user_feedback``・
      ``conversation_ended``・訂正の昇格・sleep-time の書き戻し) は全て
      ``{"op": "patch", "id": ..., "fields": {...}}`` の行。最後に書いた内容を
      メモリに持ち (``_shadow``)、変わったフィールドだけを書く。G0 は書き手が
      3 つあり、互いに全置換していた。
    - 読むときは id ごとに畳む (:func:`fold_experience_file`)。1000 件への刈り込みと
      全体の書き直しは **コンパクション** だけ (行数が上限の 2 倍超 / sleep-time の
      :meth:`save` / 起動時の :meth:`load`)。

    **耐久性はこのストア自身の責務** (2026-09-06 監査 F-04: 保存を sleep-time に
    預けていた間、ログ上は記録済みでも実体が無い窓が常時開いていた)。追記は
    ターンの終わりに fsync される (書き手スレッドの契約)。
    """

    FORMAT = EXPERIENCE_FORMAT

    def __init__(
        self, max_entries: int = MAX_ENTRIES, *, autosave: bool = True,
        writer: ChatWriter | None = None,
    ):
        self.max_entries = max_entries
        self.entries: list[ExperienceEntry] = []
        # 直近に load / save したパーティションのファイル (rebind 時の退避先)。
        self.bound_path: Path | None = None
        #: ``record`` / ``flush`` で :attr:`bound_path` へ自動保存するか。
        #: 単体テストが一時ディレクトリを汚さないよう無効化できる。
        self.autosave = autosave
        self._writer = writer
        #: id → 最後にディスクへ出した永続形 (差分の基準)。
        self._shadow: dict[str, dict] = {}
        #: id を持たないエントリで記録の行を出したもの (patch できない)。
        self._anonymous_written: set[int] = set()
        #: 明示的に変更を知らされたエントリ (:meth:`touch`)。
        self._touched: dict[int, ExperienceEntry] = {}
        #: :attr:`bound_path` の物理行数 (コンパクションの判定)。
        self._lines = 0
        #: 新しい版の行を見つけたら書かない (c_05 §0.4.5)。
        self._newer_on_disk = False
        #: 未知の列挙値を持つ記録 (原形の dict)。学習には使わず、書き直しで保つ。
        self._ignored: list[dict] = []
        # 差分の計算と enqueue を 1 つにする (sleep-time の executor とループの
        # 両方から来る。先に差分を取った方が後から enqueue すると古い値が勝つ)。
        self._persist_lock = threading.RLock()

    @property
    def writer(self) -> ChatWriter:
        return self._writer if self._writer is not None else default_writer()

    # ── 束縛・読み込み・保存 ──

    def bind(self, path: str | Path) -> None:
        """保存先だけを設定する (読み込みはしない)。

        ファイルが未作成の初回起動でも :meth:`flush` が働くようにするための
        入口。``load`` は「ファイルがあるときだけ」呼ばれるため、これが無いと
        **初回セッションのあいだ自動保存が無効** になり、F-04 が新規環境で
        そのまま再現する。
        """
        with self._persist_lock:
            if self.bound_path is None or not self._same(path, self.bound_path):
                self._reset_tracking()
            self.bound_path = Path(path)

    def load(self, path: str | Path) -> None:
        """``path`` を畳んで読み込み、以後の保存先にする。

        読み込んだエントリは ``conversation_ended`` を確定させ
        (:meth:`_mark_loaded_conversations_ended`)、畳めた行が記録数より多ければ
        (patch 行・刈り込み待ち・印の付け直し) 起動時のコンパクションとして書き直す。
        """
        target = Path(path)
        records, stats = fold_experience_file(target)
        parsed: list[ExperienceEntry] = []
        ignored: list[dict] = []
        skipped = 0
        for d in records:
            if unknown_enums(d):
                ignored.append(d)
                continue
            try:
                parsed.append(entry_from_dict(d))
            except CodecError as exc:
                skipped += 1
                logger.warning(
                    "Skipping malformed experience entry on load: %s (%r)", exc, str(d)[:120],
                )
        broken = stats["bad"] + stats["orphan"] + skipped
        if broken:
            logger.warning(
                "Skipped %d unreadable experience line(s) in %s (bad=%d, orphan patch=%d, "
                "malformed=%d; kept %d)",
                broken, target, stats["bad"], stats["orphan"], skipped, len(parsed),
            )
        if stats["newer"]:
            logger.error(
                "%s has %d experience row(s) of a newer version; not writing to it",
                target, stats["newer"],
            )
        if ignored:
            logger.warning(
                "Kept %d experience row(s) with unknown enum values in %s unchanged; "
                "they are not used for learning",
                len(ignored), target,
            )
        if len(parsed) > self.max_entries:
            parsed = parsed[-self.max_entries:]
        with self._persist_lock:
            self.entries = parsed
            self.bound_path = target
            self._reset_tracking()
            self._newer_on_disk = bool(stats["newer"])
            self._ignored = ignored
            marked = self._mark_loaded_conversations_ended()
            self._shadow = {e.id: entry_to_dict(e) for e in self.entries if e.id}
            self._anonymous_written = {id(e) for e in self.entries if not e.id}
            self._lines = stats["lines"]
            if stats["lines"] and (marked or stats["lines"] > self._record_count):
                self._rewrite(target)
        if stats["lines"] or parsed:
            logger.info("Loaded %d experience entries from %s", len(self.entries), target)

    def save(self, path: str | Path) -> None:
        """保存する。

        ``path`` が束縛先なら、変わったエントリを patch 行で出してから
        (sleep-time / shutdown の書き戻し)、行が記録数より多ければコンパクションを
        書き手スレッドへ出す。別のパス (rebind の退避・移行先) なら今のエントリで
        そのファイルを書き直す。
        """
        target = Path(path)
        if self.bound_path is not None and self._same(target, self.bound_path):
            self.flush()
            if self._lines > self._record_count:
                self._compact()
            return
        if self._newer_on_disk:
            return
        try:
            self.writer.replace(
                target, rows_text([*self._ignored, *(entry_to_dict(e) for e in self.entries)]),
                format_id=EXPERIENCE_FORMAT.format_id, fsync=True,
            )
        except DataReadonlyError:
            logger.debug("Experience buffer not saved to %s: data root is read-only", target)
            return
        logger.info("Saved %d experience entries to %s", len(self.entries), target)

    def rebind(self, path: str | Path, *, previous: str | Path | None = None) -> None:
        """base モデル切替で経験バッファを新パーティションへ向け直す。

        現在のエントリを **先に** 旧パーティション (``previous``、省略時は
        :attr:`bound_path`) へ保存してから空にし、``path`` を読み込む
        (ファイル未存在なら空のまま)。以後の save は ``path`` が既定になる。
        """
        prev = Path(previous) if previous is not None else self.bound_path
        if prev is not None and self.entries:
            self.save(prev)
        self.entries = []
        self.load(path)
        logger.info(
            "Experience buffer rebound: %s -> %s (%d entries)",
            prev, self.bound_path, len(self.entries),
        )

    # ── 記録・変更 ──

    def record(self, entry: ExperienceEntry) -> None:
        """エントリを追加 (同じターンの二重記録は無視する)"""
        if not entry.timestamp:
            entry.timestamp = utc_now()
        if not entry.id:
            entry.id = ExperienceEntry.new_id()

        # 同一 turn_id が既にあるなら再送・再取り込み。timestamp は秒精度で
        # 同一秒の別ターンと区別できないため、ID で判定する。
        if entry.turn_id and any(
            e.turn_id == entry.turn_id for e in reversed(self.entries[-32:])
        ):
            logger.debug(
                "Skipping duplicate experience for turn %s", entry.turn_id,
            )
            return

        self.entries.append(entry)

        # ローテーション (ファイルからの刈り込みはコンパクションで)
        if len(self.entries) > self.max_entries:
            overflow = len(self.entries) - self.max_entries
            for old in self.entries[:overflow]:
                self._shadow.pop(old.id, None)
            self.entries = self.entries[overflow:]
            logger.info("Rotated %d old entries", overflow)

        self._persist(self.entries[-_RECENT_WINDOW:])

    def touch(self, *entries: ExperienceEntry) -> None:
        """エントリを書き換えたことを知らせる (次の保存で patch 行を出す)。"""
        for entry in entries:
            self._touched[id(entry)] = entry

    def flush(self, *, touched_only: bool = False) -> None:
        """変わったエントリを patch 行で出す (未バインド / 無効時は no-op)。

        既定は全件の差分を取る (呼出元が印を付けたエントリを知らせなくても落とさない。
        sleep-time の書き戻し・Level 1 の訂正の昇格)。``touched_only=True`` は
        :meth:`touch` で知らされたエントリだけを見る (ループ上の経路)。
        """
        self._persist([] if touched_only else self.entries)

    def _persist(self, candidates: list[ExperienceEntry]) -> None:
        if not self.autosave or self.bound_path is None or self._newer_on_disk:
            return
        with self._persist_lock:
            path = self.bound_path
            if path is None:
                return
            touched = list(self._touched.values())
            self._touched.clear()
            lines: list[str] = []
            seen: set[int] = set()
            for entry in (*touched, *candidates):
                if id(entry) in seen:
                    continue
                seen.add(id(entry))
                line = self._line_for(entry)
                if line is not None:
                    lines.append(line)
            if not lines:
                return
            try:
                self.writer.append(path, lines, format_id=EXPERIENCE_FORMAT.format_id)
            except DataReadonlyError:
                # readonly は起動側で学習を止めている。ここは最後の砦で、ターンごとに
                # WARNING を出さない (G1 設計 §16.2 #14)。
                logger.debug("Experience not saved: data root is read-only")
                return
            self._lines += len(lines)
            overfull = self._lines > 2 * self.max_entries
        if overfull:
            self._compact()

    def _line_for(self, entry: ExperienceEntry) -> str | None:
        """未出力なら記録の行、出力済みで変わっていれば patch 行、無変更なら ``None``。"""
        previous = self._shadow.get(entry.id) if entry.id else None
        if previous is not None and _unchanged(entry, previous):
            # 複写を作らずに比べる (全件の差分を取る flush で 1000 件を複写しない)。
            return None
        current = entry_to_dict(entry)
        if not entry.id:
            if id(entry) in self._anonymous_written:
                return None
            self._anonymous_written.add(id(entry))
            return _row_line(current)
        previous = self._shadow.get(entry.id)
        self._shadow[entry.id] = current
        if previous is None:
            return _row_line(current)
        changed = _diff(previous, current)
        if not changed:
            return None
        return _patch_line(entry.id, changed)

    def _compact(self) -> None:
        """書き手スレッドでファイルを畳んで書き直す (完了を待たない)。"""
        path = self.bound_path
        if path is None or not self.autosave or self._newer_on_disk:
            return
        with self._persist_lock:
            lines_at_enqueue = self._lines

        def build() -> str:
            body, kept = compacted_body(path, self.max_entries)
            with self._persist_lock:
                if self.bound_path is not None and self._same(path, self.bound_path):
                    # enqueue 後に追記された行はコンパクションの後ろに積まれる。
                    self._lines = kept + max(0, self._lines - lines_at_enqueue)
            logger.info("Compacted experience log %s to %d entries", path, kept)
            return body

        try:
            self.writer.replace(path, build, format_id=EXPERIENCE_FORMAT.format_id, fsync=True)
        except DataReadonlyError:
            logger.debug("Experience compaction skipped: data root is read-only")

    def _rewrite(self, path: Path) -> None:
        """今のエントリでファイルを書き直す (起動時のコンパクション)。

        未知の列挙値の記録は先頭に原形のまま置く (使わないので位置に意味は無い)。
        """
        if not self.autosave or self._newer_on_disk:
            return
        rows = [*self._ignored, *(self._shadow.get(e.id) or entry_to_dict(e) for e in self.entries)]
        try:
            self.writer.replace(
                path, rows_text(rows), format_id=EXPERIENCE_FORMAT.format_id, fsync=True,
            )
        except DataReadonlyError:
            logger.debug("Experience rewrite skipped: data root is read-only")
            return
        self._lines = len(rows)

    @property
    def _record_count(self) -> int:
        """ファイルに残るべき記録の数 (使う記録 + 未知の列挙値の記録)。"""
        return len(self.entries) + len(self._ignored)

    @property
    def ignored_count(self) -> int:
        """未知の列挙値のため学習に使わず保持している記録の数 (c_05 §0.5.3)。"""
        return len(self._ignored)

    def _reset_tracking(self) -> None:
        self._shadow = {}
        self._anonymous_written = set()
        self._touched = {}
        self._lines = 0
        self._newer_on_disk = False
        self._ignored = []

    @staticmethod
    def _same(a: str | Path, b: str | Path) -> bool:
        return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))

    def get_recent(self, n: int = 10) -> list[ExperienceEntry]:
        """直近 n 件取得"""
        return self.entries[-n:]

    def mark_user_feedback(
        self, session_id: str, *, negative: bool | None, note: str = "",
        query: str | None = None,
    ) -> ExperienceEntry | None:
        """セッションの最新 (または ``query`` が一致する最新) の経験に明示評価を刻む。

        ``negative=None`` は評価の取り消し。見つからなければ ``None``
        (private ターン等、経験が無い応答)。変更は patch 行で出す。
        """
        wanted = " ".join((query or "").split())
        for entry in reversed(self.entries):
            if entry.session_id != session_id:
                continue
            if wanted and " ".join(entry.query.split()) != wanted:
                continue
            entry.signals.user_negative = negative
            entry.signals.user_note = (note or "").strip()[:400]
            self._persist([entry])
            return entry
        return None

    def get_failures(self, mode: str | None = None) -> list[ExperienceEntry]:
        """失敗エントリ抽出 (Level 2 の失敗プール / 訂正ペアの母集団)。

        失敗の語彙は :data:`backend.free.learning.fitness.DEFECT_WEIGHTS` を
        SSOT にする (2026-09-21)。旧実装はここに 4 条件を手で並べており、
        Level 1 の重み表と **食い違っていた**:

        - ``rephrased_query`` を数えていた (重み表からは外した字句判定。実データ
          161 件で発火 0 件)。
        - ``long_form`` の検証落ちを数えていなかった。create の失敗の大半は
          これで、実測 19 件中 15 件。そのため Level 2 の create プールは
          **2 件** しか集まらず (``Bootstrap skipped: not enough failures
          (2 < 20)``)、閾値をいくら下げても発火しなかった。

        共有表へ寄せると、検証器が立てた失敗 (算術の破綻 / ツール結果の不使用 /
        指示違反 / 出力の崩れ) と長文の検証落ちが同じ 1 本の定義で入る。

        Args:
            mode: 指定時はそのモード ("chat"/"create") のエントリのみに絞る。
                None (省略、既定) の場合は全モード横断 (後方互換)。
        """
        result = [
            e for e in self.entries
            if _has_defect(e.signals)
            # max_tokens で切れた応答は「モデルの失敗」ではなく設定由来の
            # 打ち切りで、正しい答えも持たない。2026-09-07 ライブ監査では
            # 失敗プール 6 件中 5 件がこれで、Level 2 の目的関数がほぼ打ち切りで
            # 埋まった。few-shot (add_from_experiences) と訂正ペアも同じ理由で
            # 除外している。
            and not e.signals.truncated
        ]
        if mode is not None:
            result = [e for e in result if e.mode == mode]
        return result

    @property
    def count(self) -> int:
        return len(self.entries)

    @property
    def source_memory_ids(self) -> list[str]:
        """FadeMem ガード用: 空リスト（将来拡張）"""
        return []

    @property
    def pending_memory_ids(self) -> list[str]:
        """FadeMem ガード用: 空リスト（将来拡張）"""
        return []

    def as_dicts(self) -> list[dict]:
        """全エントリを dict 化して返す (時系列順)。学習側の純粋関数の入力用。"""
        return [entry_to_dict(e) for e in self.entries]

    def _mark_loaded_conversations_ended(self) -> int:
        """読み込んだエントリの ``conversation_ended`` を確定させる。

        ``FeedbackCollector.mark_conversation_ended`` は ``_session_entries``
        (**メモリ上のオブジェクト参照リスト**) を辿って印を付けるため、印を付ける前に
        プロセスが落ちると紐付けごと消え、そのエントリは**二度と** ended にならない
        (実測: 69 件中 42 件が未マークのまま死蔵。fitness が 0.528 と 0.806 で
        二分され、学習の選択圧から丸ごと外れていた)。

        永続ファイルにあるエントリは定義上すべて**このプロセスの起動より前**に
        書かれたもので、それを書いたプロセスはもう存在しない。したがって当該会話は
        既に終了している。読み込み時点で確定させるのが正しく、ここが唯一の
        再起動耐性のある地点になる。

        Returns:
            新たに ended を立てた件数。
        """
        marked = 0
        for entry in self.entries:
            if not entry.signals.conversation_ended:
                entry.signals.conversation_ended = True
                marked += 1
        if marked:
            logger.info(
                "Marked %d loaded entries as conversation_ended "
                "(their writing process is gone, so those conversations are over)",
                marked,
            )
        return marked
