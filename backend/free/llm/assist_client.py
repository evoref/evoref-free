"""アシストモデル専用クライアント（Free版: ローカルバックエンドのみ）

メモリ・RAG処理専用の軽量クライアント。
メインモデル（:8080）とは別インスタンス（:8081）で動作し、
応答レイテンシへの影響なしにバックグラウンド処理を実行する。
"""

import asyncio
import contextlib
import json
import time
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from backend.free.llm._base_client import (
    BaseHTTPClient,
    MAX_ATTEMPTS,
    RETRY_WAIT_INITIAL,
    RETRY_WAIT_MAX,
    RETRYABLE_STATUS_CODES,
    _RETRYABLE_EXCEPTIONS,
    async_retry_http_call,
    make_retry_logger,
)
from backend.free.llm.json_extract import extract_json_object
from backend.free.llm.json_schemas import (
    make_response_format,
    resolve_response_format_for_purpose,
)
from backend.free.llm.utils import extract_content
from backend.log_config import get_logger

logger = get_logger("llm.assist_client")

# セマフォ取得タイムアウト（秒）
# リトライ合計待機(1+2+4=7s) + リクエスト(10s) × リトライ数を考慮
_SEMAPHORE_ACQUIRE_TIMEOUT = 30.0


Priority = Literal["realtime", "background", "learning"]


# purpose 文字列 → priority の割付
# realtime: チャット応答パスで同期発火するもの (ユーザ体感レイテンシ直結)
# background: Sleep-time worker / 自律ループ / 長文生成バックグラウンド
# learning: Level 1/2 学習サイクル (CritiqueSynthesizer / PolicyEvolver 等)
#
# 未知の purpose は "background" にフォールバックする (realtime への
# 誤昇格を避け、ユーザ応答パスを保護する安全側デフォルト)。
_PURPOSE_PRIORITY_MAP: dict[str, Priority] = {
    # realtime — チャット応答パス
    "retrieval_quality_judge": "realtime",
    "retrieval_necessity_judge": "realtime",
    # 取得直後 content gate の marginal band 関連性判定。チャット応答パス
    # (coding mode) で同期発火するため realtime。
    "retrieval_chunk_gate": "realtime",
    "tool_judgment": "realtime",
    # executable query (環境依存事実) のコマンド合成。チャット応答パスで
    # tool_call_judge から同期発火するため realtime。
    "executable_command_synth": "realtime",
    # meta-cognitive 計画は coding mode のチャット応答パスで
    # 同期発火する (タスク分解後に他のエージェントが実行される) ため
    # realtime に分類する。
    "meta_cognitive_plan": "realtime",
    # staged コーディングのタスクグラフ合成。coding mode のチャット応答パス
    # 冒頭で同期発火し、ユーザは plan 確定を待つ (meta_cognitive_plan と同根拠)。
    "coding_task_graph": "realtime",
    # エディタタブ名導出。coding mode のチャット応答パス (long_form /
    # meta_cognitive のエディタ出力終端) で同期発火し、ユーザはタブ名表示を
    # 待つ。max_tokens 数十・budget 0 の極短呼出なので realtime に乗せる。
    "editor_filename": "realtime",
    # ツール結果からユーザ質問の答えに必要な要点を抽出する digest。Deliberative の
    # チャット応答パスで base 生成の直前に同期発火するため realtime。
    "tool_result_digest": "realtime",
    # SemMem pending 競合のチャット回答判定。pending 存在ターンのチャット
    # 応答パスで同期発火するため realtime。sleep-time のノートマージ
    # (conflict_resolution、background) とは別 purpose。
    "conflict_chat_judge": "realtime",
    # background — Sleep-time / 自律ループ / 長文生成
    # conflict_resolution は SleepTimeWorker._step6_resolve_conflicts (run_full)
    # からのみ発火する sleep-time 専用処理。チャット応答パスから同期発火しない
    # ため background。realtime セマフォ (concurrency=1) を Full サイクル中の
    # ノートマージで専有し、チャット判定 (tool_judgment 等) をブロックしないよう
    # background スロットに乗せる。
    "conflict_resolution": "background",
    "contextual_prefix": "background",
    "long_form_planning": "background",
    "long_form_code_review": "background",
    "long_form_text_review": "background",
    # コード生成の事前準備: 共有設計仕様 (contract) 合成とフローチャート生成。
    # コード生成完了前のプラン段階で発火する重い JSON 合成のため long_form_*
    # と同列の background スロット。
    "code_spec_synthesis": "background",
    "flowchart_synthesis": "background",
    # staged コーディングの spec 工程: 設計仕様 (spec.md) のテキスト生成。
    # 自律ループ内 (inline 駆動) で発火する重いテキスト生成のため background。
    "coding_spec_doc": "background",
    # コードリペア (長文生成末尾の検証ゲート付き修正)。生成完了後に発火する
    # 重い修正パスで long_form_* と同列。background スロット。
    "code_repair": "background",
    "ralph_loop": "background",
    "note_evolution": "background",
    "summarize": "background",
    # cartridge_creator の eval.json QA ペア生成。Pro API
    # `/api/pro/cartridges/create` のレスポンスとして同期発火するが、
    # 実体は数千 chars の document に対する 5-10 QA バッチ生成で、
    # ユーザはローディング UI 越しに待つ。realtime に乗せるとチャット応答
    # パスの semaphore を専有してしまうため background に分類する。
    "cartridge_eval_generation": "background",
    # URL リコール (Phase 1) の自己採点。sleep-time 内で発火する
    # ため background スロットを使う。
    "url_relevance_score": "background",
    # learning — 学習サイクル
    "critique_synthesis": "learning",
    "policy_evolution": "learning",
    # Level 1 プロンプト進化の mutation 生成。低速 base 占有を避けるため
    # assist へ寄せる (learning スロットで直列実行)。
    "prompt_evolution": "learning",
    "spsa": "learning",
    "memory_scoring": "learning",
}

_DEFAULT_PRIORITY: Priority = "background"


# purpose 文字列 → タイムアウト秒の既定マップ
# config.yaml の ``assist_model.timeouts`` に該当キーが無い場合でも、
# ここで定義したデフォルト値が自動適用される。config 値は本マップより
# 優先される (明示指定 → config.timeouts → DEFAULTS → assist_model.timeout)。
#
# 既定値の根拠:
#   - long_form_* (90s)       : CogWriter plan/review は 4-8k tok 入力を処理
#   - contextual_prefix (60s) : Sleep-time Step 5.8 の長文 document prefix 生成
#   - critique (45s)          : CritiqueSynthesizer の失敗クラスタ解析
#   - policy_evolution (45s)  : PolicyParamEvolver の候補 JSON 生成
#   - note_evolution (20s)    : A-MEM note 進化の軽量要約
#   - conflict_resolution (15s): 短文マージ判定
#   - retrieval_quality_judge: 既定 ``timeout`` をそのまま使う (短時間で済む)
#   - tool_judgment (15s): チャット応答パスで同期発火。アシスト接続時は
#     モード非依存で常時有効化されるため呼出頻度が上がる。base との GPU 競合に
#     加え、realtime 並列化 (concurrency.realtime>=2) 時は assist 同士の
#     バッチ競合で個々の応答が 1.3-1.6 倍に延びるため 15s に緩める
#     (失敗時はルールベースに fallback)。
PURPOSE_TIMEOUT_DEFAULTS: dict[str, float] = {
    "contextual_prefix": 60.0,
    "long_form_planning": 90.0,
    "long_form_code_review": 90.0,
    "long_form_text_review": 90.0,
    # 設計仕様合成は modules/data_models/interfaces 等を含む JSON を返す。
    # max_tokens は 1536 に抑制済 (strategy_cogwriter) だが、iGPU + 低速 assist で
    # cold prefill 時に 90s を超えると ReadTimeout+retry で ~90s 余分に待つため、
    # 1 attempt で完了させるマージンとして 120s を採る。フローチャートは mermaid
    # 文字列 1 本のため 45s。
    "code_spec_synthesis": 120.0,
    "flowchart_synthesis": 45.0,
    # コードリペアは assembled 全体を再出力するため long_form_* と同等に確保。
    "code_repair": 90.0,
    "conflict_resolution": 15.0,
    # pending 競合のチャット回答判定。チャット応答パスで同期発火し、
    # base との GPU 競合 + realtime 並列化時の assist 相互競合があるため
    # tool_judgment と同根拠で 15s (失敗時は no-op で pending 維持)。
    "conflict_chat_judge": 15.0,
    "critique_synthesis": 45.0,
    "policy_evolution": 45.0,
    # プロンプト変異 (~1024 tok のテキスト生成)。assist は base の約5倍速で
    # 1024 tok ≈ 35-40s のため 45s で安全マージン。
    "prompt_evolution": 45.0,
    "note_evolution": 20.0,
    # 検索必要性判定 (1 bit JSON)。チャット応答パスの先頭で同期発火する
    # ため、ユーザ体感を阻害しないよう短く打ち切る。
    "retrieval_necessity_judge": 5.0,
    # 取得直後 content gate の marginal band 関連性判定 (1 回の batched JSON)。
    # チャット応答パスで同期発火するため短く打ち切る。
    "retrieval_chunk_gate": 5.0,
    # executable query 判定 + コマンド合成。チャット応答パスで
    # tool_call_judge から発火する。is_executable: bool + command: str の
    # 短い JSON を返すだけだが、base モデルとの GPU 競合 + realtime 並列化時の
    # 投機実行 (tool_judgment との並走) で 8s では空振りしやすいため 15s に
    # 緩める。失敗時は regex フォールバックが正しさを担保。
    "executable_command_synth": 15.0,
    # ツール呼出判定 ({"tool": ..., "args": {...}} の小さな JSON)。チャット
    # 応答パスで Deliberative / MetaCognitive から同期発火し、アシスト接続は
    # 常時有効。base モデルとの GPU 競合 + realtime 並列化時の assist 相互競合で
    # 8s では空振りしやすいため 15s に緩める。失敗時はルールベース判定に
    # フォールバックする。
    "tool_judgment": 15.0,
    # meta-cognitive 計画 (タスク分解) は coding mode の
    # 応答パスで発火するため、長すぎるとユーザ体感を阻害する。30s で打ち切り。
    "meta_cognitive_plan": 30.0,
    # staged コーディングのタスクグラフ合成 (summary + modules[] の JSON)。
    # meta_cognitive_plan より構造が大きいため 45s。失敗時は longform へ fallback。
    "coding_task_graph": 45.0,
    # staged コーディングの spec.md テキスト生成。long_form_* と同等に確保。
    "coding_spec_doc": 90.0,
    # エディタタブ名導出 ({"file_name": "..."} の極短 JSON)。応答パスで
    # 同期発火するため短く打ち切る。失敗時は言語別 fallback stem に倒す。
    "editor_filename": 8.0,
    # Recurrent 戦略の summary 再帰更新。1 セクション末尾を
    # 短文要約するだけのため 30s で十分。
    "summarize": 30.0,
    # ツール結果 (最大 4096 chars) からの query 連動抽出。チャット応答パスで
    # 同期発火するが、コスト無視・品質優先方針で余裕を持たせ 20s。
    "tool_result_digest": 20.0,
    # cartridge eval.json 生成。最大 4000 chars の document を
    # 連結したプロンプトから最大 10 QA を生成する。アシストモデルが
    # thinking 抑制下で QA を逐次出力するのに 60s 程度を想定。長文プラン
    # (90s) より軽いため 60s で打ち切る。
    "cartridge_eval_generation": 60.0,
    # URL リコール 自己採点。質問文 + 応答 + URL 本文プレビュー (1500 chars)
    # を入力に二値判定 + 短文 reason を返すだけのため、長くは要らない。
    "url_relevance_score": 15.0,
    # ラルフループの action 列生成。バックグラウンド周回 (priority=background、
    # LoopDriver 側に max_wall=1800s の外枠がある) のためユーザ体感を阻害せず、
    # task → <actions> JSON 配列の生成にローカルモデルで時間がかかることがある。
    # グローバル既定だと ReadTimeout で 1 周回を空振りさせるため長めに確保。
    "ralph_loop": 120.0,
}


# purpose 文字列 → reasoning_budget の既定マップ
# override 値。``-1`` 無制限 / ``0`` 即終了 / ``N>0`` token 上限。
# config.yaml の ``assist_model.reasoning_budgets`` 未指定時に適用される。
#
# サーバ側 ``-rea off`` で起動している場合 (推奨設定) はそもそも thinking
# が disable されるため本値は no-op だが、``-rea auto`` / ``on`` で運用する
# 場合に purpose ごとに budget を絞る安全側デフォルトとして機能する。
#
# 既定値の根拠:
#   - long_form_planning / long_form_code_review (2048 tok):
#       CogWriter のプラン/レビューは大きめの推論余地が有効
#   - contextual_prefix (256 tok):
#       Sleep-time の document prefix は短い要約で十分
#   - critique_synthesis (512 tok):
#       失敗クラスタ解析。中程度の推論で打ち切る
#   - policy_evolution (1024 tok):
#       政策候補 JSON 生成。設計探索のため広めに確保
#   - conflict_resolution (0):
#       マージ判定は thinking 不要 → 即終了
PURPOSE_REASONING_BUDGET_DEFAULTS: dict[str, int] = {
    "long_form_planning": 2048,
    "long_form_code_review": 2048,
    "long_form_text_review": 2048,
    # 設計仕様合成は API/データモデル/依存の設計探索のため広めに確保。
    # フローチャートは構造抽出が主で中程度。
    "code_spec_synthesis": 2048,
    "flowchart_synthesis": 512,
    # コードリペアは検出エラーを明示して渡す機械的修正。thinking 不要で即終了。
    "code_repair": 0,
    "contextual_prefix": 256,
    "critique_synthesis": 512,
    "conflict_resolution": 0,
    # チャット回答の機械的分類。thinking 不要 → 即終了
    "conflict_chat_judge": 0,
    "policy_evolution": 1024,
    # プロンプト変異は機械的なテキスト書き換え。<think> は _strip_think_tags で
    # 除去・汚染判定の対象であり thinking は有害なため即終了。
    "prompt_evolution": 0,
    # 検索必要性判定は機械的な 1 bit 分類で thinking 不要
    "retrieval_necessity_judge": 0,
    # content gate の関連性判定も機械的な index 選別で thinking 不要
    "retrieval_chunk_gate": 0,
    # executable query 判定 + コマンド合成は機械的な抜き出し + 短文生成で
    # thinking 不要。response_format (ExecutableCommandSynth) で構造を固定する。
    "executable_command_synth": 0,
    # ツール呼出判定は機械的な分類 + 引数抽出で thinking 不要。
    # response_format (ToolJudgmentResult) で構造を固定する。
    "tool_judgment": 0,
    # meta-cognitive 計画は機械的なタスク分解で thinking 不要
    "meta_cognitive_plan": 0,
    # タスクグラフ合成も機械的な分割で thinking 不要 (response_format で構造固定)。
    "coding_task_graph": 0,
    # spec.md 生成は設計の言語化で中程度の推論余地が有効。
    "coding_spec_doc": 1024,
    # エディタタブ名導出は機械的な命名で thinking 不要。
    # response_format (EditorFilenameResult) で構造を固定する。
    "editor_filename": 0,
    # 既存要約 + 新セクションの再要約は機械的処理で thinking 不要
    "summarize": 0,
    # ツール結果からの要点抽出は機械的処理で thinking 不要 (assist の思考暴走回避)
    "tool_result_digest": 0,
    # cartridge eval.json QA ペア生成は機械的な抜き出しで
    # thinking 不要。response_format (CartridgeEvalQAList) で構造を固定する。
    "cartridge_eval_generation": 0,
    # URL リコール 自己採点は二値判定 + 短文 reason のため thinking 不要。
    "url_relevance_score": 0,
}


def resolve_priority(purpose: str) -> Priority:
    """purpose 文字列から priority を解決する

    マップにない purpose は安全側として ``background`` を返す。
    realtime への誤昇格でユーザ応答パスを阻害しないため。
    """
    return _PURPOSE_PRIORITY_MAP.get(purpose, _DEFAULT_PRIORITY)


# ── purpose 別 timeout の反応的自己較正 ───────────────────────────────
# モデル切替後、``PURPOSE_TIMEOUT_DEFAULTS`` が新モデル (特に低速 iGPU 上の
# アシスト) のレイテンシに合わず ReadTimeout を頻発させる問題への対処。
# ``generate()`` で timeout を観測する度に当該 purpose の timeout 天井を
# ``_CALIB_BUMP_FACTOR`` 倍ずつ引き上げ、base (DEFAULTS) の
# ``_CALIB_MAX_SCALE`` 倍を上限とする。引き上げ値はアシストモデル名でキー化して
# ``local/assist_calibration.json`` に永続化し、次回起動時に同一モデルなら
# 再利用する (別モデルに戻した場合は適用しない)。config の
# ``assist_model.timeouts[purpose]`` が明示された purpose はユーザー権威として
# 較正対象外 (clobber しない)。
#
# per-attempt timeout を引き上げるため、background / learning では総ウォール
# クロック ≈ ``per_attempt * MAX_ATTEMPTS + backoff`` になる点を踏まえ
# ``_CALIB_MAX_SCALE`` は控えめ (3.0 = 3 attempt で実質 ~9x 上限) に留める。
_CALIB_BUMP_FACTOR = 1.5
_CALIB_MAX_SCALE = 3.0


class _PurposeStat:
    """purpose 単位のレイテンシ観測 (反応的較正用)。

    成功レイテンシの EWMA と連続 timeout 回数を保持する。EWMA は較正の引き金
    には使わず観測値 (WARNING ログ / 将来の /api/status 露出) として用いる。
    較正のトリガは timeout イベントそのもの。
    """

    __slots__ = ("ewma", "consecutive_timeouts", "samples")

    def __init__(self) -> None:
        self.ewma: float | None = None
        self.consecutive_timeouts: int = 0
        self.samples: int = 0

    def observe_success(self, elapsed: float, *, alpha: float = 0.3) -> None:
        """成功レイテンシを EWMA に取り込む。"""
        self.samples += 1
        if self.ewma is None:
            self.ewma = elapsed
        else:
            self.ewma = alpha * elapsed + (1.0 - alpha) * self.ewma


def _resolve_assist_model_filename(config: dict, local_cfg: dict) -> str:
    """アシストモデルの GGUF ファイル名 (basename) を解決する。

    較正値を model-scoped に保存 / ロードするためのキー。
    優先: ``local.model`` / ``local.model_path`` → ``model_paths.assist_model``。
    解決できない場合は空文字列 (較正の永続化を無効化)。
    """
    raw = local_cfg.get("model") or local_cfg.get("model_path") or ""
    if not raw:
        raw = (config.get("model_paths") or {}).get("assist_model", "")
    if not raw:
        return ""
    return Path(str(raw)).name


def _finish_reason_of(result: dict) -> str:
    """llama-server レスポンスから ``choices[0].finish_reason`` を取り出す。

    取得できない場合は空文字列。``"length"`` は max_tokens 切断を示す。
    """
    if not isinstance(result, dict):
        return ""
    choices = result.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        fr = choices[0].get("finish_reason")
        if isinstance(fr, str):
            return fr
    return ""


def _extract_cache_metrics(result: dict) -> dict | None:
    """llama-server レスポンスから KV キャッシュ命中率を抽出する

    llama.cpp のレスポンスには ``timings`` や ``usage`` に prompt の
    トークン数と再利用されたキャッシュトークン数が含まれる場合がある。
    具体的なフィールド名はビルドにより ``cache_n`` / ``prompt_cached_n`` /
    ``cached_tokens`` などとばらつくため、既知のキーを順に探索する。
    該当情報が無ければ ``None`` を返す。

    Returns:
        ``{"prompt_n": int, "cache_n": int, "hit_ratio": float}`` または
        ``None`` (timings が欠落している場合)。
    """
    if not isinstance(result, dict):
        return None

    timings = result.get("timings")
    if not isinstance(timings, dict):
        timings = {}

    prompt_n: int | None = None
    cache_n: int | None = None

    # prompt_n 候補: timings.prompt_n / usage.prompt_tokens
    for key in ("prompt_n", "n_prompt_tokens_processed"):
        val = timings.get(key)
        if isinstance(val, (int, float)):
            prompt_n = int(val)
            break
    if prompt_n is None:
        usage = result.get("usage")
        if isinstance(usage, dict):
            val = usage.get("prompt_tokens")
            if isinstance(val, (int, float)):
                prompt_n = int(val)

    # cache_n 候補: timings.cache_n / prompt_cached_n / cached_tokens /
    # usage.prompt_tokens_details.cached_tokens
    for key in (
        "cache_n", "prompt_cached_n", "cached_tokens", "n_cached",
    ):
        val = timings.get(key)
        if isinstance(val, (int, float)):
            cache_n = int(val)
            break
    if cache_n is None:
        usage = result.get("usage")
        if isinstance(usage, dict):
            details = usage.get("prompt_tokens_details")
            if isinstance(details, dict):
                val = details.get("cached_tokens")
                if isinstance(val, (int, float)):
                    cache_n = int(val)

    if prompt_n is None and cache_n is None:
        return None

    metrics: dict = {}
    if prompt_n is not None:
        metrics["prompt_n"] = prompt_n
    if cache_n is not None:
        metrics["cache_n"] = cache_n
    if prompt_n is not None and prompt_n > 0 and cache_n is not None:
        metrics["hit_ratio"] = round(min(1.0, cache_n / prompt_n), 4)
    return metrics


class _EmptyResponseError(RuntimeError):
    """assist モデルから空 content の応答が返ってきたことを示す内部例外

    `_request_with_retry` 内で raise し、リトライループ側でキャッチして
    指数バックオフ付き再試行を行うために使用する。リトライ枯渇後は
    呼び出し側に伝播せず、最終レスポンス (空 content の dict) を返す。
    """

    def __init__(self, data: dict) -> None:
        super().__init__("assist model returned empty content")
        self.data = data


def _is_retryable_status(exc: BaseException) -> bool:
    """``HTTPStatusError`` のうち retryable status のみ True を返す述語"""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS_CODES
    return False


def _assist_before_sleep_callback(
    *,
    purpose: str,
    retry_logger,
):
    """assist リトライ用の ``before_sleep`` callback を生成する。

    空応答 / HTTP / Timeout を統一フォーマットで WARNING ログに残し、
    ``DebugLogger.log_retry_attempt`` にも転送する。``retry_logger`` が
    ``None`` の場合は WARNING のみで構造化ログは送出しない。
    """

    def _callback(state) -> None:
        if state.outcome is None or state.next_action is None:
            return
        exc = state.outcome.exception()
        if exc is None:
            return
        wait_sec = state.next_action.sleep
        attempt = state.attempt_number
        if isinstance(exc, _EmptyResponseError):
            logger.warning(
                "Assist model returned empty content (purpose=%s), "
                "retry %d/%d, waiting %.2fs",
                purpose or "<unspecified>",
                attempt, MAX_ATTEMPTS - 1, wait_sec,
            )
        elif isinstance(exc, httpx.HTTPStatusError):
            logger.warning(
                "Assist model request failed (status=%d, purpose=%s), "
                "retry %d/%d, waiting %.2fs",
                exc.response.status_code, purpose or "<unspecified>",
                attempt, MAX_ATTEMPTS - 1, wait_sec,
            )
        else:
            logger.warning(
                "Assist model request failed (%s, purpose=%s), "
                "retry %d/%d, waiting %.2fs",
                type(exc).__name__, purpose or "<unspecified>",
                attempt, MAX_ATTEMPTS - 1, wait_sec,
            )
        if retry_logger is not None:
            try:
                retry_logger(attempt, exc, wait_sec)
            except Exception:  # pragma: no cover
                logger.debug("retry_logger raised; ignored", exc_info=True)

    return _callback


class AssistModelClient(BaseHTTPClient):
    """アシストモデル（ローカル llama-server :8081）クライアント

    設計書 §2.3.2 に基づき、メモリ・RAG処理専用の別インスタンスと通信する。
    ストリーミング非対応（バックグラウンド処理のため不要）。
    セマフォで同時呼び出し数を制限し、アシストモデルの過負荷を防止する。
    """

    # 呼出側が base (LocalClient) と assist を区別するための capability マーカー。
    # base クライアントは本属性を持たないため getattr(..., False) で判別できる。
    is_assist_client: bool = True

    def __init__(self, config: dict, debug_logger=None):
        """アシストモデルクライアントを初期化

        Args:
            config: config.yaml 全体の dict。assist_model セクションを参照する。
            debug_logger: DebugLogger インスタンス（任意）
        """
        self._debug_logger = debug_logger
        assist_cfg = config.get("assist_model", {})
        local_cfg = assist_cfg.get("local", {})

        host = local_cfg.get("host", "127.0.0.1")
        port = local_cfg.get("port", 8081)
        self.url = f"http://{host}:{port}"

        # モデル能力スナップショット (capability probe が背景で確定する; docs/c_15)。
        # 型: CapabilitySnapshot | None。未プローブ / プローブ無効時は None。
        self.capabilities = None
        self._capability_probe_task = None

        self.timeout: float = float(assist_cfg.get("timeout", 30))
        super().__init__(timeout=self.timeout)

        # response_format。``true`` の場合、purpose 別に解決した
        # OAI 互換 ``response_format: {"type": "json_schema", ...}`` を
        # ``/v1/chat/completions`` ペイロードに付与し、llama.cpp の制約サン
        # プリングで JSON 構文エラーを原理的にゼロ化する。古い llama-server
        # build やデバッグ用途で無効化したい場合は ``false`` を指定する。
        self._response_format_enabled: bool = bool(
            assist_cfg.get("response_format_enabled", True)
        )

        # 用途別セマフォ。realtime / background / learning を
        # 独立スロット化し、Sleep-time と Level 2 が並行してもチャット応答
        # パスの realtime リクエストがキュー待ちしないようにする。
        concurrency_cfg = assist_cfg.get("concurrency") or {}
        self._concurrency: dict[Priority, int] = {
            "realtime": int(concurrency_cfg.get("realtime", 1)),
            "background": int(concurrency_cfg.get("background", 1)),
            "learning": int(concurrency_cfg.get("learning", 1)),
        }
        self._semaphores: dict[Priority, asyncio.Semaphore] = {
            p: asyncio.Semaphore(n) for p, n in self._concurrency.items()
        }
        # purpose 別タイムアウト。長文生成プラン/レビュー
        # critique 等の重いタスクは既定の `timeout` では枯渇するため、
        # purpose 文字列をキーに override する。
        raw_timeouts = assist_cfg.get("timeouts") or {}
        self._purpose_timeouts: dict[str, float] = {
            str(k): float(v) for k, v in raw_timeouts.items()
        }

        # purpose 別 reasoning_budget。llama.cpp 上流 PR
        # (b8870+) の per-request override。``-1`` 無制限 / ``0``
        # 即終了 / ``N>0`` token 上限。config 未指定 purpose は
        # ``PURPOSE_REASONING_BUDGET_DEFAULTS`` を適用する。
        raw_budgets = assist_cfg.get("reasoning_budgets") or {}
        self._purpose_reasoning_budgets: dict[str, int] = {
            str(k): int(v) for k, v in raw_budgets.items()
        }

        # purpose 別 timeout の反応的自己較正 (model-keyed)。アシストモデル名で
        # キー化した較正値を起動時にロードし、``generate()`` で timeout を観測する
        # 度に天井を引き上げて ``local/assist_calibration.json`` に永続化する。
        # config の ``assist_model.timeouts[purpose]`` が明示された purpose は
        # ユーザー権威として較正対象外。
        self._latency_stats: dict[str, _PurposeStat] = {}
        self._assist_model_filename = _resolve_assist_model_filename(
            config, local_cfg,
        )
        self._calibrated_timeouts: dict[str, float] = {}
        self._calibration_path: Path | None = None
        try:
            from backend.config import get_path_resolver
            self._calibration_path = get_path_resolver().resolve_local(
                "assist_calibration_file",
            )
        except Exception:  # noqa: BLE001 — config 未ロード (単体テスト等) では
            # 永続化を無効化し、in-memory 較正のみ動作させる。
            self._calibration_path = None
        if self._calibration_path is not None and self._assist_model_filename:
            from backend.free.llm.assist_calibration_store import (
                AssistCalibrationStore,
            )
            loaded = AssistCalibrationStore.load_timeouts(
                self._calibration_path, self._assist_model_filename,
            )
            # config 明示 purpose は権威として上書きしない。
            self._calibrated_timeouts = {
                p: v for p, v in loaded.items()
                if p not in self._purpose_timeouts
            }
            if self._calibrated_timeouts:
                logger.info(
                    "Loaded assist timeout calibration for model=%s "
                    "(%d purposes)",
                    self._assist_model_filename, len(self._calibrated_timeouts),
                )

        # 全 assist リクエストに per-request で注入する chat_template_kwargs。
        # 非 thinking モデル運用時は空 dict を指定して注入そのものを停止する。
        # Qwen3 系既定: ``{"enable_thinking": False}``。
        raw_kwargs = assist_cfg.get(
            "chat_template_kwargs", {"enable_thinking": False},
        )
        self._chat_template_kwargs: dict = (
            dict(raw_kwargs) if isinstance(raw_kwargs, dict) else {}
        )

        # 能力ゲート: assist モデルの reasoning mode を解決し、enable_thinking /
        # reasoning_budget の送信可否を決める。
        #   none   = reasoning 系を一切送らない (非 thinking モデルでの HTTP 400 回避)
        #   always = enable_thinking は送らない (OFF 不可) が reasoning_budget は送る
        #            (budget=0 等で常時思考を抑制できるため)
        #   toggle = 従来どおり enable_thinking + reasoning_budget
        from backend.config import resolve_enable_thinking, resolve_reasoning_mode

        self._reasoning_mode = resolve_reasoning_mode(config, "assist")
        # reasoning_budget は mode="none" のときだけ送らない (always/toggle は honor しうる)
        self._reasoning_capability_off = self._reasoning_mode == "none"
        resolved_enable_thinking = resolve_enable_thinking(
            config, "assist",
            explicit=self._chat_template_kwargs.get("enable_thinking"),
        )
        if resolved_enable_thinking is None:
            self._chat_template_kwargs.pop("enable_thinking", None)
        else:
            self._chat_template_kwargs["enable_thinking"] = resolved_enable_thinking

        # モデルサイズ推定 (LLM 呼び出しインターバル計算用)
        # 優先順位:
        #   1. assist_model.local.params_b (config 明示指定)
        #   2. assist_model.local.model → model_paths.assist_model から regex 推定
        #   3. (起動後) update_params_from_server で /props の model_alias から再推定
        explicit_params_b = local_cfg.get("params_b")
        if explicit_params_b is not None and explicit_params_b > 0:
            self._params_b = float(explicit_params_b)
            self._params_b_explicit = True
        else:
            model_path = local_cfg.get("model", local_cfg.get("model_path", ""))
            if not model_path:
                model_path = config.get("model_paths", {}).get("assist_model", "")
            from backend.free.llm.model_metadata import estimate_params_b
            self._params_b = estimate_params_b(str(model_path))
            self._params_b_explicit = False

        logger.info(
            "AssistModelClient initialized: url=%s, timeout=%.1fs, "
            "concurrency(realtime=%d, background=%d, learning=%d), "
            "estimated_params=%.1fB, response_format_enabled=%s",
            self.url, self.timeout,
            self._concurrency["realtime"],
            self._concurrency["background"],
            self._concurrency["learning"],
            self._params_b,
            self._response_format_enabled,
        )

    @property
    def params_b(self) -> float:
        """アシストモデルのパラメータ数推定値（B 単位）"""
        return self._params_b

    @property
    def concurrency(self) -> dict[str, int]:
        """用途別セマフォスロット数

        ``{"realtime": int, "background": int, "learning": int}`` を返す。
        外部 (API レスポンス / ログ) から参照するための公開プロパティ。
        """
        return dict(self._concurrency)

    @property
    def realtime_concurrency(self) -> int:
        """realtime セマフォスロット数 (チャット応答パスの並列度)

        チャット応答パス (chat.py / tool_call_judge.py) が assist 判定の
        並列化・投機実行を有効化するかの判定に使う。``>=2`` で並列化 ON、
        ``1`` (既定) で現行の直列フローを維持する。assist llama-server 側の
        ``slots`` とセットで設定する前提 (片方だけではキュー待ちで効果ゼロ)。
        """
        return self._concurrency["realtime"]

    async def update_params_from_server(self) -> None:
        """llama-server /props から実際のモデル名を取得してパラメータ数を更新

        transient I/O 失敗 (ConnectError / ReadTimeout / 5xx 等) は
        ``async_retry_http_call`` でリトライする。最終的に失敗した場合は
        既定値の ``params_b`` 推定で稼働継続する (degraded mode)。

        最終失敗時のログレベルを WARNING に引き上げる
        ``params_b`` は sleep-time の LLM 呼び出しインターバル計算
        (`backend/free/memory/sleep_update.py`) と
        `model_params_b` API レスポンス (`assist_model_api.py`) で参照されるため、
        silent fallback だと sleep-time のインターバルが想定と乖離 + UI 表示が
        実モデルと不整合のまま運用されてしまう。
        ``debug.enabled=false`` の標準運用でも `local/logs/backend.log` に
        記録されるよう WARNING にする。
        """
        # config で明示指定されている場合は /props 推定で上書きしない
        if self._params_b_explicit:
            logger.debug(
                "Skipping /props params_b update (explicit override = %.1fB)",
                self._params_b,
            )
            return

        client = self._get_http_client()
        retry_logger = make_retry_logger(self._debug_logger, backend="assist")

        async def _fetch_props() -> dict:
            resp = await client.get(f"{self.url}/props", timeout=5.0)
            resp.raise_for_status()
            return resp.json()

        try:
            props = await async_retry_http_call(
                _fetch_props,
                request_label="Assist /props",
                retry_logger=retry_logger,
            )
        except Exception as e:
            logger.warning(
                "Failed to update params_b from /props after retries "
                "(continuing with model_path-derived estimate %.1fB): %s",
                self._params_b, e,
            )
            return
        model_id = props.get("model_alias", "") or props.get("model_path", "")
        if model_id:
            from backend.free.llm.model_metadata import estimate_params_b
            new_params = estimate_params_b(model_id)
            if new_params != self._params_b:
                logger.info(
                    "Updated params estimate from /props: %.1fB → %.1fB (model=%s)",
                    self._params_b, new_params, model_id,
                )
                self._params_b = new_params

    @property
    def background_slot(self) -> int:
        """バックグラウンドスロット（自動割当）"""
        return -1

    async def generate(
        self,
        messages: list[dict],
        *,
        stream: bool = False,
        temperature: float = 0.7,
        max_tokens: int | None = 256,
        id_slot: int | None = None,
        timeout: float | None = None,
        purpose: str = "",
        cache_prompt: bool = False,
        response_format: dict | None = None,
        response_schema: type[BaseModel] | None = None,
    ) -> dict:
        """アシストモデルで推論（非ストリーミング専用）

        LocalClient.generate(stream=False) と同じ戻り値形式を返す。
        conflict_resolver / note_evolver が同一インターフェースで動作する。

        Args:
            messages: OpenAI 互換の messages 配列
            stream: 無視（常に非ストリーミング）
            temperature: 生成温度
            max_tokens: 最大トークン数
            id_slot: 無視（アシストモデルは単一スロット）
            timeout: リクエストタイムアウト秒数。省略時は self.timeout を使用。
                     バックグラウンド処理で早期に諦めたい場合に指定する。
            purpose: デバッグログに記録する呼び出し目的（例: "retrieval_quality_judge"）
            cache_prompt: llama-server の KV キャッシュ再利用を要求する。
                     同一 slot 内で前回リクエストと共通する prefix の prefill を
                     スキップする。contextual_prefix のように「同一ドキュメントを
                     先頭に含む複数リクエスト」を連続発行する用途で有効化する。
            response_format: OAI 互換 ``response_format`` dict を直接指定する
。``{"type": "json_schema", "json_schema": {...}}``
                形式。``response_schema`` と purpose 別自動解決より優先される。
            response_schema: Pydantic v2 BaseModel サブクラスを指定すると、
                ``make_response_format()`` で OAI 互換 dict に変換して
                ``response_format`` として送信する。``content_type``
                で schema が分岐する purpose (``long_form_planning``) で利用する。

        Returns:
            OpenAI 互換のレスポンス dict
            {"choices": [{"message": {"content": "..."}}]}
        """
        payload: dict = {
            "messages": messages,
            "stream": False,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if cache_prompt:
            payload["cache_prompt"] = True

        # response_format の解決。優先順位:
        #   1. 引数 ``response_format`` (明示 dict)
        #   2. 引数 ``response_schema`` (Pydantic クラス → dict 変換)
        #   3. ``PURPOSE_SCHEMAS`` による purpose 文字列からの自動解決
        # フィーチャフラグ ``response_format_enabled`` が False の場合は何も
        # 付与せず、json_extract.py のフォールバック経路に委ねる。
        resolved_response_format: dict | None = None
        if self._response_format_enabled:
            if response_format is not None:
                resolved_response_format = response_format
            elif response_schema is not None:
                resolved_response_format = make_response_format(
                    response_schema, name=purpose or response_schema.__name__,
                )
            elif purpose:
                resolved_response_format = resolve_response_format_for_purpose(
                    purpose,
                )
        if resolved_response_format is not None:
            payload["response_format"] = resolved_response_format

        # config 駆動の chat_template_kwargs に purpose 別 reasoning_budget を
        # マージして per-request 注入する。
        #
        # 既定 (Qwen3 / Gemma-4 系): ``{"enable_thinking": False}`` で
        # reasoning_content にトークンを消費し content が空になる問題を防止する。
        # ``assist_model.local.skip_chat_parsing: true`` で起動した場合は
        # サーバ側構造解析が OFF となり ``<think>...</think>`` が ``message.content``
        # に混入するが、Python 側 ``_ReasoningFilter`` が post-strip で除去する。
        # ``enable_thinking`` / ``reasoning_budget`` を送る理由は、jinja template
        # が受理した場合のフォールバック保険として機能させるため。
        #
        # 非 thinking モデル運用時は ``assist_model.chat_template_kwargs: {}`` を
        # 指定し、かつ purpose 別 ``reasoning_budgets`` も未設定にすれば、本処理は
        # ``chat_template_kwargs`` キー自体をペイロードから外す。
        chat_template_kwargs = dict(self._chat_template_kwargs)
        # thinking 非対応 arch には reasoning_budget も送らない (能力ゲート)。
        if not self._reasoning_capability_off:
            budget = self._resolve_reasoning_budget(purpose)
            if budget is not None:
                chat_template_kwargs["reasoning_budget"] = budget
        if chat_template_kwargs:
            payload["chat_template_kwargs"] = chat_template_kwargs

        logger.debug(
            "Generate: messages=%d, temperature=%.2f, max_tokens=%s",
            len(messages), temperature, max_tokens,
        )

        effective_timeout = self._resolve_timeout(timeout, purpose)
        priority = resolve_priority(purpose)

        t0 = time.monotonic()
        try:
            result = await self._request_with_retry(
                payload, timeout=effective_timeout, priority=priority,
                purpose=purpose,
            )
        except (httpx.TimeoutException, TimeoutError):
            # timeout を purpose 別較正シグナルとして記録し、天井を反応的に
            # 引き上げてから例外を再送出する (caller の degraded fallback は不変)。
            self._record_assist_latency(
                purpose, time.monotonic() - t0,
                timed_out=True, budget=effective_timeout,
            )
            raise
        elapsed = time.monotonic() - t0
        self._record_assist_latency(
            purpose, elapsed, timed_out=False, budget=effective_timeout,
        )

        # DebugLogger にアシストモデルリクエストを記録
        dl = self._debug_logger
        if dl:
            content = extract_content(result)
            cache_metrics = _extract_cache_metrics(result) if cache_prompt else None
            choices = result.get("choices") or []
            finish_reason = ""
            if choices:
                fr = choices[0].get("finish_reason")
                if isinstance(fr, str):
                    finish_reason = fr
            dl.log_assist_request(
                messages_count=len(messages),
                response_preview=content,
                elapsed_sec=elapsed,
                purpose=purpose,
                priority=priority,
                resolved_timeout=effective_timeout,
                cache_metrics=cache_metrics,
                response_format_used=resolved_response_format is not None,
                finish_reason=finish_reason,
                response_length=len(content),
            )

        return result

    async def generate_json(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.3,
        purpose: str = "",
        timeout: float | None = None,
        list_key: str | None = None,
        response_format: dict | None = None,
        response_schema: type[BaseModel] | None = None,
        telemetry: dict | None = None,
    ) -> dict:
        """アシストモデルで JSON 出力を生成・パース

        Args:
            prompt: ユーザープロンプト（JSON 出力を指示する内容）
            max_tokens: 最大トークン数
            temperature: 生成温度（JSON 生成は低め推奨）
            timeout: リクエストタイムアウト秒数。省略時は purpose 別タイムアウト
                (`assist_model.timeouts`) → `assist_model.timeout` の順で解決。
            list_key: 裸の JSON 配列が返ってきた場合に dict 化する際のキー
                (例: cogwriter の "units")。``response_format`` 制約が効いて
                いる場合は通常裸配列は返らないが、フォールバック経路で参照される。
            response_format: OAI 互換 ``response_format`` dict を直接指定する
。``response_schema`` より優先
            response_schema: Pydantic v2 BaseModel サブクラスを指定すると、
                purpose 別自動解決より優先して制約サンプリングに使われる
。``content_type`` で schema が分岐する purpose
                (``long_form_planning``) で利用する。
            telemetry: 省略可の out-param dict。指定すると ``finish_reason`` /
                ``truncated`` (max_tokens 切断か) / ``repair_used`` を書き戻す。
                plan/spec 合成側が「JSON が途中切断されモジュール/ユニットが
                欠落した可能性」を検知するために使う。

        Returns:
            パースされた JSON dict。パース失敗時は空 dict。
        """
        messages = [{"role": "user", "content": prompt}]
        result = await self.generate(
            messages, max_tokens=max_tokens, temperature=temperature,
            purpose=purpose, timeout=timeout,
            response_format=response_format,
            response_schema=response_schema,
        )
        content = extract_content(result)

        # finish_reason="length" (max_tokens 切断) を検知して可視化する。
        # 切断された JSON は下の json-repair で部分復元され「成功」に見えるが、
        # plan/spec ではモジュール/ユニットが黙って欠落する原因になるため、
        # WARNING + out-param telemetry で呼出側が気付けるようにする。
        finish_reason = _finish_reason_of(result)
        truncated = finish_reason == "length"
        if telemetry is not None:
            telemetry["finish_reason"] = finish_reason
            telemetry["truncated"] = truncated
            # WARNING は telemetry を渡した「切断を気にする」呼出 (plan/spec/
            # task-graph 合成) のみに限定する。long_form_*_review 等は設計上
            # 毎回 length 切断され json-repair で復元する purpose なので、
            # 無条件 WARNING だと backend.log を汚す。
            if truncated:
                logger.warning(
                    "assist JSON truncated (purpose=%s): output hit max_tokens=%s; "
                    "result may be incomplete (partial JSON will be repaired)",
                    purpose or "<unspecified>", max_tokens,
                )

        # JSON 部分を抽出してパース
        # ``response_format`` 制約サンプリングが効く llama-server build では
        # ``content`` がそのまま valid JSON になるため戦略 1 で即パースされる。
        # 古い build / フラグ無効時 / max_tokens 切断時の保険として戦略 2-3
        # と json-repair を残置。repair 使用の有無は DebugLogger に
        # ``op="json_repair"`` を別エントリとして書き出す。
        repair_telemetry: dict = {}
        parsed = extract_json_object(
            content, list_key=list_key, telemetry=repair_telemetry,
        )
        if telemetry is not None:
            telemetry["repair_used"] = bool(repair_telemetry.get("repair_used"))
        if repair_telemetry.get("repair_used") and self._debug_logger is not None:
            self._debug_logger.log_assist_json_repair(
                purpose=purpose,
                list_key=list_key,
                raw_preview=content,
                repaired_preview=str(parsed),
            )
        if parsed is None:
            logger.warning(
                "Failed to extract JSON from assist model response "
                "(purpose=%s): %s",
                purpose or "<unspecified>", content[:200],
            )
            return {}
        return parsed

    async def health_check(self) -> bool:
        """アシストモデル llama-server のヘルスチェック"""
        try:
            client = self._get_http_client()
            resp = await client.get(f"{self.url}/health", timeout=5.0)
            healthy = resp.status_code == 200
            logger.debug("Health check: status=%d, healthy=%s", resp.status_code, healthy)
            return healthy
        except (httpx.ConnectError, httpx.TimeoutException):
            logger.debug("Health check: connection failed")
            return False

    async def _request_with_retry(
        self, payload: dict, *,
        timeout: float | None = None,
        priority: Priority = "background",
        purpose: str = "",
    ) -> dict:
        """tenacity ベースのリトライ付き HTTP リクエスト

        ``AsyncRetrying`` を ``async for attempt`` ループとして使い、
        各 attempt の前後でセマフォを acquire/release する。これにより:

        - リトライ間でセマフォを保持しないため、バックオフ待機中に他の
          リクエストがスロットを使える
        - tenacity が ``_EmptyResponseError`` / ``HTTPStatusError`` (retryable
          status のみ) / transient I/O 例外 (``ConnectError`` /
          ``ReadTimeout`` 等) を統一的にリトライ判定する
        - ``before_sleep`` callback で WARNING ログ + DebugLogger
          ``log_retry_attempt`` への構造化ログを記録する

        セマフォ取得自体にもタイムアウトを設け、過負荷時の無期限ブロックを防止する。

        Args:
            payload: リクエストペイロード
            timeout: リクエストタイムアウト秒数。省略時は self.timeout を使用。
            priority: 用途別セマフォ選択キー
                realtime / background / learning のいずれか。
            purpose: ``DebugLogger.log_retry_attempt`` に転送する purpose。
        """
        client = self._get_http_client()
        effective_timeout = timeout if timeout is not None else self.timeout
        semaphore = self._semaphores[priority]
        slot_count = self._concurrency[priority]
        last_empty_data: dict | None = None

        retry_predicate = (
            retry_if_exception_type(_EmptyResponseError)
            | retry_if_exception_type(_RETRYABLE_EXCEPTIONS)
            | retry_if_exception(_is_retryable_status)
        )

        retry_logger = make_retry_logger(
            self._debug_logger, backend="assist", purpose=purpose,
        )

        # realtime は purpose timeout を「総ウォールクロック予算」として強制する。
        # per-attempt timeout (client.post) は維持しつつ、リトライ + バックオフ込みの
        # 総時間が effective_timeout を超えたら打ち切り、チャット応答パスの体感
        # レイテンシを保証する (旧実装は per-attempt timeout × リトライで 8s 予算が
        # 14-24s に膨張していた)。background / learning は信頼性優先で従来挙動を維持。
        deadline = (
            asyncio.timeout(effective_timeout)
            if priority == "realtime"
            else contextlib.nullcontext()
        )
        try:
            async with deadline:
                async for attempt in AsyncRetrying(
                    retry=retry_predicate,
                    stop=stop_after_attempt(MAX_ATTEMPTS),
                    wait=wait_exponential_jitter(
                        initial=RETRY_WAIT_INITIAL, max=RETRY_WAIT_MAX,
                    ),
                    before_sleep=_assist_before_sleep_callback(
                        purpose=purpose, retry_logger=retry_logger,
                    ),
                    reraise=True,
                ):
                    with attempt:
                        # セマフォ取得 (リトライごとに再取得し、待機中は解放)
                        try:
                            await asyncio.wait_for(
                                semaphore.acquire(),
                                timeout=_SEMAPHORE_ACQUIRE_TIMEOUT,
                            )
                        except asyncio.TimeoutError:
                            logger.warning(
                                "Semaphore acquire timed out after %.0fs "
                                "(priority=%s, slots=%d)",
                                _SEMAPHORE_ACQUIRE_TIMEOUT, priority, slot_count,
                            )
                            # セマフォ獲得失敗自体はリトライ対象外として
                            # 即座に伝播させる (overload 状態の自己回復は別系統)
                            raise TimeoutError(
                                f"Assist model overloaded: semaphore timeout "
                                f"after {_SEMAPHORE_ACQUIRE_TIMEOUT}s "
                                f"(priority={priority})"
                            )

                        try:
                            resp = await client.post(
                                f"{self.url}/v1/chat/completions",
                                json=payload,
                                timeout=effective_timeout,
                            )
                            resp.raise_for_status()
                            data = resp.json()
                            content = extract_content(data)
                            if len(content) == 0:
                                last_empty_data = data
                                self._log_empty_response(
                                    data, attempt.retry_state.attempt_number - 1,
                                )
                                raise _EmptyResponseError(data)
                            logger.debug(
                                "Generate complete: response_length=%d chars, attempt=%d",
                                len(content),
                                attempt.retry_state.attempt_number,
                            )
                            return data
                        finally:
                            semaphore.release()
        except _EmptyResponseError:
            # 空応答で枯渇した場合は最後のレスポンスをそのまま返す
            # (呼び出し側の既存の空応答ハンドリング経路を壊さないため)
            if last_empty_data is not None:
                if self._debug_logger is not None:
                    self._debug_logger.log_decision(
                        decision_point="assist_health_fallback",
                        chosen="degraded_local",
                        candidates=["external_assist", "local_assist", "degraded_local"],
                        reason="empty_response_retries_exhausted",
                        context={"purpose": purpose},
                        scope="request",
                    )
                return last_empty_data
            raise
        except TimeoutError:
            # realtime 総予算超過 or セマフォ overload。caller は degraded fallback へ倒す。
            logger.warning(
                "Assist request aborted by timeout "
                "(purpose=%s, priority=%s, budget=%.1fs)",
                purpose, priority, effective_timeout,
            )
            raise

        # AsyncRetrying は reraise=True で必ず例外を上げるか値を return するため
        # ここに到達することはない (型チェッカ向けフォールバック)。
        raise RuntimeError("unreachable")  # pragma: no cover

    def _log_empty_response(self, data: dict, attempt: int) -> None:
        """空 content レスポンスの診断情報を WARNING ログに記録する

        llama-server が HTTP 200 を返しつつ content が空になる原因を
        切り分けるため、finish_reason / usage / raw choices preview を出力する。
        """
        choices = data.get("choices") or []
        first = choices[0] if choices else {}
        finish_reason = first.get("finish_reason")
        usage = data.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        # raw body は機微情報を含む可能性があるため 300 文字に制限
        try:
            raw_preview = json.dumps(data, ensure_ascii=False)[:300]
        except (TypeError, ValueError):
            raw_preview = str(data)[:300]
        logger.warning(
            "Generate returned empty response (attempt=%d, finish_reason=%s, "
            "prompt_tokens=%s, completion_tokens=%s, raw_preview=%s)",
            attempt + 1, finish_reason, prompt_tokens, completion_tokens,
            raw_preview,
        )

    def _resolve_reasoning_budget(self, purpose: str) -> int | None:
        """purpose 別 reasoning_budget を解決する

        優先順位:
            1. ``assist_model.reasoning_budgets[purpose]`` (config 上書き)
            2. ``PURPOSE_REASONING_BUDGET_DEFAULTS[purpose]`` (コード既定)
            3. ``None`` (どこにも設定が無い場合は per-request override
               を送信せず、サーバ側起動時 default に従う)


            -1: 無制限 (per-request 上限なし、起動時 default に追従)
             0: 即終了 (思考トリガー直後に thinking 終了 grammar)
            N>0: N トークンの token cap

        purpose が空文字列の場合は ``None`` を返す。
        """
        if not purpose:
            return None
        if purpose in self._purpose_reasoning_budgets:
            return self._purpose_reasoning_budgets[purpose]
        if purpose in PURPOSE_REASONING_BUDGET_DEFAULTS:
            return PURPOSE_REASONING_BUDGET_DEFAULTS[purpose]
        return None

    def _record_assist_latency(
        self, purpose: str, elapsed: float, *,
        timed_out: bool, budget: float,
    ) -> None:
        """``generate()`` のレイテンシ観測を purpose 別の反応的較正に反映する。

        observability が応答パスを壊さないよう、本メソッドは例外を外へ伝播しない。
        config で timeout を明示している purpose (ユーザー権威) は較正対象外。
        """
        if not purpose or purpose in self._purpose_timeouts:
            return
        try:
            stat = self._latency_stats.get(purpose)
            if stat is None:
                stat = _PurposeStat()
                self._latency_stats[purpose] = stat
            if timed_out:
                stat.consecutive_timeouts += 1
                self._bump_calibrated_timeout(purpose, budget)
            else:
                stat.consecutive_timeouts = 0
                stat.observe_success(elapsed)
        except Exception:  # noqa: BLE001 — 観測は応答パスを壊さない
            logger.debug("assist latency recording failed", exc_info=True)

    def _bump_calibrated_timeout(self, purpose: str, budget: float) -> None:
        """timeout を観測した purpose の較正天井を反応的に引き上げる。

        base = ``PURPOSE_TIMEOUT_DEFAULTS[purpose]`` (無ければ ``self.timeout``)
        を基準に ``_CALIB_BUMP_FACTOR`` 倍ずつ、``base * _CALIB_MAX_SCALE`` を
        上限に上げる。値が変化した場合のみ WARNING ログ + 永続化する。
        """
        base = PURPOSE_TIMEOUT_DEFAULTS.get(purpose, self.timeout)
        ceiling = base * _CALIB_MAX_SCALE
        current = self._calibrated_timeouts.get(purpose, base)
        new = min(current * _CALIB_BUMP_FACTOR, ceiling)
        if new <= current + 1e-6:
            # 既に上限。これ以上引き上げない (無限増殖防止)。
            return
        self._calibrated_timeouts[purpose] = new
        logger.warning(
            "Assist purpose '%s' timed out (budget=%.1fs); raising calibrated "
            "timeout %.1fs -> %.1fs (cap=%.1fs, model=%s)",
            purpose, budget, current, new, ceiling,
            self._assist_model_filename or "unknown",
        )
        self._persist_calibration()

    def _persist_calibration(self) -> None:
        """較正値をアシストモデル名でキー化して ``local/`` に永続化する。"""
        if self._calibration_path is None or not self._assist_model_filename:
            return
        try:
            from backend.free.llm.assist_calibration_store import (
                AssistCalibrationStore,
            )
            AssistCalibrationStore.save_timeouts(
                self._calibration_path,
                self._assist_model_filename,
                self._calibrated_timeouts,
            )
        except OSError as e:
            logger.warning("Failed to persist assist calibration: %s", e)

    def _resolve_timeout(
        self, explicit: float | None, purpose: str,
    ) -> float:
        """リクエストタイムアウトを解決する

        優先順位:
            1. 明示指定 (``timeout`` 引数)
            2. ``assist_model.timeouts[purpose]`` (config 上書き)
            3. 反応的較正値 (``_calibrated_timeouts[purpose]``、ReadTimeout 観測で
               引き上げ・model-keyed 永続化)
            4. ``PURPOSE_TIMEOUT_DEFAULTS[purpose]``
            5. ``assist_model.timeout`` (グローバル既定)

        config で purpose 別 override を明示していなくても、重いタスク
        (long_form_planning / contextual_prefix / critique_synthesis 等)
        が CLAUDE.md の既定仕様通り延長されることを保証する。config 明示
        purpose は較正値より優先され、ユーザー設定を上書きしない。
        """
        if explicit is not None:
            return float(explicit)
        if purpose:
            if purpose in self._purpose_timeouts:
                return self._purpose_timeouts[purpose]
            if purpose in self._calibrated_timeouts:
                return self._calibrated_timeouts[purpose]
            if purpose in PURPOSE_TIMEOUT_DEFAULTS:
                return PURPOSE_TIMEOUT_DEFAULTS[purpose]
        return self.timeout
