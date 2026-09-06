"""補助タスク用クライアント — ベースモデルの専有スロットで構造化生成を回す。

チャット応答の生成そのもの以外の LLM 呼出 (sleep-time のノート進化 / 競合解決 /
要約、学習サイクルの critique・プロンプト変異、長文生成のプラン・レビュー、
create モードの計画・設計仕様合成) を一手に引き受ける。かつては専用の
アシストモデル (:8081 の 2 つ目の llama-server) が担っていたが、以下の理由で
**ベースモデル 1 本に集約** した (2026-08-14)。

- 小型アシストでは構造化判定の実効品質が出なかった。Level 1 の差分編集変異は
  旧アシスト (gemma-4-E4B) では通らず、ベースへ寄せて実機 8/8 成功になった。
  ``json_schema grammar not enforced`` の警告自体は build / chat template 側の
  性質でベースでも出うるため、パースのフォールバックは常に効かせておく。
- 旧アシストの VRAM / 帯域をベースへ明け渡せる。2 モデル同時常駐をやめた分、
  ベースのコンテキストと GPU レイヤに余裕が生まれる。
- degraded 経路 (``aux_client=None``) と residency (on_demand 起動待ち) の
  分岐が消え、呼出側が「呼べば返る」前提で書けるようになる。

不変則との関係 (CLAUDE.md §6 #1): 禁じられているのは「ユーザー応答の生成中に
補助処理を同じスロットへ割り込ませる」こと。本クライアントはチャットスロットを
使わず、purpose 別のスロット方針 (``CHAT_PATH_PURPOSES`` はチャット応答パスで
同期発火するので ``classifier_slot``、それ以外は ``background_slot``) で
チャットと KV を分離する。同一スロットへの同時要求は per-slot ロックで直列化し、
purpose タイムアウトは **ロック取得後 (dispatch) から** 数える。さらに
json_schema が解決できる purpose は文法制約経路 (``generate_constrained``) を
通すため、チャット応答パスから同期発火する補助判定も出力長が読める。
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from backend.aux_telemetry import record_aux_failure
from backend.exceptions import LLMTimeoutError
from backend.free.llm.generation_gate import (
    activity_token,
    wait_for_idle,
    was_contended_since,
)
from backend.free.llm.json_extract import extract_json_object
from backend.free.llm.json_schemas import (
    make_response_format,
    resolve_response_format_for_purpose,
)
from backend.log_config import get_logger

if TYPE_CHECKING:
    from pydantic import BaseModel

    from backend.debug_logger import DebugLogger
    from backend.free.llm.local_client import LocalClient

logger = get_logger("llm.aux_client")


# purpose 文字列 → タイムアウト秒の既定マップ。
#
# 明示指定 (呼出側の ``timeout=``) > 較正値 > 本マップ > ``_DEFAULT_TIMEOUT``。
#
# 既定値は旧アシストモデル時代の実測をベースに、ベースモデル (旧アシストより
# 2〜4 倍大きい) での decode 差を見込んで引き上げてある。実レイテンシとの
# ずれは ``_bump_calibrated_timeout`` が反応的に埋める。
PURPOSE_TIMEOUT_DEFAULTS: dict[str, float] = {
    # ── sleep-time / メモリ ──────────────────────────────────────────
    "conflict_resolution": 45.0,
    "note_evolution": 60.0,
    "summarize": 90.0,
    "contextual_prefix": 120.0,
    "url_relevance_score": 45.0,
    "assertion_naming": 45.0,
    # ── 学習サイクル ─────────────────────────────────────────────────
    "critique_synthesis": 120.0,
    "fewshot_quality_score": 45.0,
    "prompt_candidate_judge": 60.0,
    # ── 長文生成 ─────────────────────────────────────────────────────
    "long_form_planning": 180.0,
    "long_form_code_review": 180.0,
    "long_form_text_review": 180.0,
    "code_spec_synthesis": 240.0,
    "flowchart_synthesis": 120.0,
    "code_repair": 180.0,
    # ── create モード / 自律ループ ────────────────────────────────────
    # 取得直後 content gate の marginal band 関連性判定。チャット応答パスで
    # 同期発火するため短く打ち切る (timeout 時は prune せず全件通す)。
    "retrieval_chunk_gate": 15.0,
    # 計画 (タスク分解) は create モードの応答パスで発火する。長すぎると
    # ユーザ体感を阻害するため、失敗時は単一タスクへ倒して先へ進む。
    "meta_cognitive_plan": 90.0,
    "create_task_graph": 120.0,
    # spec.md の生成 / 節深化。実効値は executor が config の
    # ``create.staged.spec_timeout_sec`` を明示指定するためそちらが優先
    # (本値は整合目的)。
    "create_spec_doc": 600.0,
    "create_spec_deepen": 600.0,
    "spec_revision_judge": 180.0,
    "flow_spec_synthesis": 240.0,
    "flow_spec_part_synthesis": 120.0,
    "ralph_loop": 240.0,
    # ── LLM 委譲ツール (chat の summarize / translate / draft_document) ──
    # 自由文生成。ツール側の実行上限 (chat_constants.LLM_TOOL_EXECUTION_TIMEOUT_SEC
    # = 180 秒) と揃える。
    "tool_summarize": 180.0,
    "tool_translate": 180.0,
    "tool_draft_document": 180.0,
    # ── Pro ──────────────────────────────────────────────────────────
    "cartridge_eval_generation": 120.0,
}

_DEFAULT_TIMEOUT = 60.0


# 反応的タイムアウト較正の対象外 purpose。
#
# 判定基準は **タイムアウト時に答えを必要としない安全側の既定へ倒れるか**。
# そうした purpose では、予算を上げても得られるのは「より高い失敗コスト」だけで、
# 成功率の改善が結果に反映されない (2026-08-01 プロファイリングで、検索ゲートの
# 較正が失敗 1 回のコストを 3 倍にしていた)。
PURPOSE_TIMEOUT_CALIBRATION_EXEMPT: frozenset[str] = frozenset({
    "retrieval_chunk_gate",  # timeout → prune せず全件通す
})

# 較正の引き上げ幅と上限 / 下限倍率。生成 POST は ``ReadTimeout`` をリトライ
# しないので per-attempt timeout ≒ 総ウォールクロック。天井は既定の 3 倍、
# 床は既定の 0.5 倍 (実測が速いモデルではタイムアウトを **縮める** 方向にも動く)。
_CALIB_BUMP_FACTOR = 1.5
_CALIB_MAX_SCALE = 3.0
_CALIB_MIN_SCALE = 0.5
#: 成功時の所要秒 p95 に掛ける余裕率。
_CALIB_P95_HEADROOM = 1.3
#: p95 を取る直近サンプル数と、較正を始める最小サンプル数。
_CALIB_SAMPLE_WINDOW = 20
_CALIB_MIN_SAMPLES = 3

#: 背景 purpose のタイムアウト下限を **モデルの decode 速度に追随** させる。
#:
#: ``PURPOSE_TIMEOUT_DEFAULTS`` は絶対秒の定数で、どのモデルで測ったのかを
#: 表現できない。27B では 45 秒の purpose が実測 48.5 秒に届かず落ちた
#: (2026-09-06 監査 F-06: ``fewshot_quality_score`` / ``assertion_naming``)。
#: 較正は **失敗か成功を観測してから** しか効かないので、低頻度の purpose は
#: 毎回 1 回目を捨てることになる。同じ轍は c_15 のプローブでも踏んでおり、
#: そちらは ``capability.probe_timeout_sec`` (base + tokens / decode_tps) で
#: 解いた。ここでも同じ式を使う。
#:
#: **チャット応答パスの purpose には掛けない**。あちらの短い予算は
#: 「ユーザーを待たせない」ための意図的な打ち切りで、伸ばすと目的が壊れる。
#: 背景 purpose は伸ばしても失うのはアイドル時間だけで、落ちると学習データを
#: 丸ごと失う — 非対称なので下限を厚く取る側が正しい。
_BACKGROUND_TIMEOUT_BASE_SEC = 30.0
#: 下限の見積りに使う生成トークン数。実際の ``max_tokens`` は呼出ごとに違うが、
#: 下限は「この purpose なら最低これだけは待つ」という床なので固定で足りる。
_BACKGROUND_TIMEOUT_TOKENS = 512

#: **チャット応答パスで同期発火する** purpose。背景スロットは sleep-time /
#: 学習と共有で、そちらが走っていると per-slot ロック待ちでユーザー応答が
#: 遅れる。``llama.slots >= 3`` なら分類器スロット (``classifier_slot``、
#: ツール分類器と同じ「チャットの合間にしか使われない」スロット) へ寄せる。
#: 3 スロット未満では ``classifier_slot`` 自体が背景へ倒れる。
#: 判定基準は docs/c_14 §7 の「チャット応答パスで発火」の注記。
CHAT_PATH_PURPOSES: frozenset[str] = frozenset({
    "retrieval_chunk_gate",   # rag/chunk_content_gate (取得直後の関連性判定)
    "meta_cognitive_plan",    # agent/meta_cognitive (create 応答パスの計画)
    "tool_summarize",         # agent/tools/builtin (deliberative のツール実行)
    "tool_translate",
    "tool_draft_document",
})

#: **チャットがアイドルになるまで dispatch を待つ** purpose (sleep-time / 学習)。
#: スロットを分けても GPU 演算は分かれないため、これらが走っている間ユーザー
#: 応答の decode は実測で 2 倍以上遅くなる (2026-09-03 監査: 200-218 →
#: 416-445 ms/tok)。CLAUDE.md §6 #1 の「**アイドル窓の** sleep-time / 学習」を
#: 実際に強制する集合。
#:
#: **ここに入れてよいのは「ユーザーが待っていない」purpose だけ**。ユーザー起点の
#: 前景処理 (long_form_* / create_* / code_spec_* / ralph_loop / tool_*) を入れると
#: 自分のターンの完了を待つことになり、上限まで無駄に待ってから走る。
#: 同じ purpose でも呼出側によって前景/背景が分かれる場合 (``summarize`` は
#: sleep-time と長文生成の両方から呼ばれる) は、前景側が
#: ``deferrable=False`` を明示する。
DEFERRABLE_AUX_PURPOSES: frozenset[str] = frozenset({
    # sleep-time / メモリ
    "conflict_resolution",
    "note_evolution",
    "summarize",
    "contextual_prefix",
    "url_relevance_score",
    "assertion_naming",
    # 学習サイクル
    "critique_synthesis",
    "fewshot_quality_score",
    "prompt_candidate_judge",
})

#: チャットのアイドル窓を待つ上限。超えたら競合覚悟で走らせる。
#: 無期限に待たせると、会話が続く限り記憶の統合と学習が永久に走らない。
_CHAT_YIELD_MAX_WAIT_SEC = 120.0


class AuxClient:
    """ベースモデル上で補助タスクを実行するクライアント。

    呼出側から見える約束:

    - ``id_slot`` / ``cache_prompt`` は面の互換のため受けるが無視する。スロットは
      purpose 別方針 (:data:`CHAT_PATH_PURPOSES` → ``classifier_slot``、それ以外
      → ``background_slot``) で決め、チャットスロットには決して触らない。
      ``cache_prompt`` は ``LocalClient`` 側で常時 ON
    - 同一スロットへの同時要求は per-slot ロックで直列化する (llama-server 側で
      同じ ``id_slot`` を取り合うと KV を相互に追い出す)。ロック待ちは
      ``queue_wait_sec`` として別計上し、purpose タイムアウトは dispatch から数える
    - ``purpose`` から json_schema・タイムアウトを解決する
    - json_schema が解決できる purpose は文法制約経路を通る
    - 失敗は握り潰さず、``generate_json`` は空 dict を返して呼出側の
      フォールバックに委ねる。``finish_reason=length`` (JSON が途中で切れた) も
      修復せず空 dict (``telemetry["json_truncated"]=True``)
    """

    def __init__(
        self,
        local: LocalClient,
        *,
        config: dict | None = None,
        debug_logger: DebugLogger | None = None,
    ):
        self.local = local
        self._debug_logger = debug_logger
        self._model_filename = ""
        self._calibration_path = "local/aux_calibration.json"
        self._calibrated: dict[str, float] = {}
        #: purpose → 直近の成功所要秒 (p95 較正の母集団)。
        self._samples: dict[str, deque[float]] = {}
        #: スロット ID → ロック (同一スロットへの同時要求を直列化)。
        self._slot_locks: dict[int, asyncio.Lock] = {}
        self._load_calibration(config or {})

    def _load_calibration(self, config: dict) -> None:
        """config からモデル名 / 較正ファイルを解決し、較正値とサンプルを読み込む。"""
        #: 較正値の永続化キー (ベースモデルの GGUF ファイル名)。空なら永続化しない。
        self._model_filename = _resolve_base_model_filename(config)
        self._calibration_path = str(
            (config.get("local_paths") or {}).get(
                "aux_calibration_file", "local/aux_calibration.json",
            ),
        )
        self._calibrated = {}
        self._samples = {}
        if not self._model_filename:
            return
        from backend.free.llm.aux_calibration_store import AuxCalibrationStore

        self._calibrated = AuxCalibrationStore.load_timeouts(
            self._calibration_path, self._model_filename,
        )
        for purpose, values in AuxCalibrationStore.load_samples(
            self._calibration_path, self._model_filename,
        ).items():
            self._samples[purpose] = deque(values, maxlen=_CALIB_SAMPLE_WINDOW)
        if self._calibrated:
            logger.info(
                "Loaded aux timeout calibration for model=%s (%d purposes)",
                self._model_filename, len(self._calibrated),
            )

    def rebind(self, local: LocalClient, config: dict | None = None) -> None:
        """ベース差し替え (モード切替 / モデル移行) に追随する。

        ``local`` を差し替えるだけでは、較正値が **旧モデルのファイル名** に
        ぶら下がったままになる (27B の較正値を 4B に適用する / 逆)。config を
        渡せばモデル名を再解決して較正を読み直す。スロットロックは新クライアント
        のスロット配置に合わせて捨てる。
        """
        self.local = local
        self._slot_locks = {}
        if config is not None:
            self._load_calibration(config)

    @property
    def metadata(self):
        """モデルメタデータ (``run_full`` が ``params_b`` を読む)。"""
        return self.local.metadata

    @property
    def context_size(self) -> int:
        """ベースモデルの有効 context_size (サイズガードが参照する)。"""
        return int(getattr(self.local, "context_size", 8192) or 8192)

    def _background_timeout_floor(self) -> float:
        """背景 purpose のタイムアウト下限 (モデルの decode 速度由来)。

        取得できない構成 (メタデータ未解決) では 0.0 を返し、床を掛けない。
        """
        from backend.free.llm.capability import probe_timeout_sec

        try:
            params_b = float(self.metadata.params_b)
        except Exception:
            return 0.0
        if params_b <= 0:
            return 0.0
        return probe_timeout_sec(params_b, _BACKGROUND_TIMEOUT_TOKENS)

    def resolve_effective_timeout(self, purpose: str) -> float:
        """purpose に適用されるタイムアウト秒を返す (較正値込み)。

        較正値が無い背景 purpose には **モデルサイズ由来の下限** を掛ける。
        既定の絶対秒はどのモデルで測ったのかを表現できず、大型モデルでは
        1 回目が必ず落ちる (``_BACKGROUND_TIMEOUT_BASE_SEC`` の説明を参照)。
        """
        calibrated = self._calibrated.get(purpose)
        if calibrated is not None:
            return calibrated
        base = PURPOSE_TIMEOUT_DEFAULTS.get(purpose, _DEFAULT_TIMEOUT)
        if purpose in CHAT_PATH_PURPOSES:
            return base
        return max(base, self._background_timeout_floor())

    def _resolve_response_format(
        self,
        purpose: str,
        response_format: dict | None,
        response_schema: type[BaseModel] | None,
    ) -> dict[str, Any] | None:
        """明示指定 > Pydantic スキーマ > purpose 別自動解決 の順で解決する。"""
        if response_format is not None:
            return response_format
        if response_schema is not None:
            return make_response_format(response_schema)
        if purpose:
            return resolve_response_format_for_purpose(purpose)
        return None

    def _slot_for(self, purpose: str) -> int:
        """purpose に応じたスロット ID (チャットスロットは決して返さない)。"""
        background = int(self.local.background_slot)
        if purpose in CHAT_PATH_PURPOSES:
            classifier = getattr(self.local, "classifier_slot", background)
            if isinstance(classifier, int) and not isinstance(classifier, bool):
                return classifier
        return background

    @staticmethod
    def _is_deferrable(purpose: str, override: bool | None) -> bool:
        """この呼び出しがチャットのアイドル窓を待つべきか。

        呼出側の明示 (``deferrable=``) が最優先。省略時のみ purpose の既定を見る。
        """
        if override is not None:
            return override
        return purpose in DEFERRABLE_AUX_PURPOSES

    def _lock_for(self, slot: int) -> asyncio.Lock:
        lock = self._slot_locks.get(slot)
        if lock is None:
            lock = self._slot_locks[slot] = asyncio.Lock()
        return lock

    def _persist_calibration(self) -> None:
        if not self._model_filename:
            return
        try:
            from backend.free.llm.aux_calibration_store import AuxCalibrationStore

            AuxCalibrationStore.save_timeouts(
                self._calibration_path, self._model_filename, self._calibrated,
                samples={k: list(v) for k, v in self._samples.items()},
            )
        except OSError as e:
            logger.warning("Failed to persist aux calibration: %s", e)

    def _bump_calibrated_timeout(self, purpose: str) -> None:
        """タイムアウト観測を受けて当該 purpose の予算を引き上げる。

        ベースは構成 (量子化 / GPU レイヤ / コンテキスト長) でレイテンシが数倍
        変わるため、コード既定値が実機に合わない構成が必ず出る。天井は既定値の
        ``_CALIB_MAX_SCALE`` 倍まで。タイムアウトはロック取得後 (dispatch) から
        数えているので、スロット待ちで消えた時間は原因に含まれない。
        """
        if not purpose or purpose in PURPOSE_TIMEOUT_CALIBRATION_EXEMPT:
            return
        base = PURPOSE_TIMEOUT_DEFAULTS.get(purpose, _DEFAULT_TIMEOUT)
        current = self._calibrated.get(purpose, base)
        bumped = min(current * _CALIB_BUMP_FACTOR, base * _CALIB_MAX_SCALE)
        if bumped <= current:
            return
        self._calibrated[purpose] = bumped
        logger.warning(
            "Aux timeout calibrated up for purpose=%s: %.1fs -> %.1fs (base %.1fs)",
            purpose, current, bumped, base,
        )
        self._persist_calibration()

    def _record_success(self, purpose: str, elapsed: float) -> None:
        """成功所要秒を記録し、直近 p95 から較正値を組み直す。

        ``calibrated = clamp(p95 × 1.3, base × 0.5, base × 3)``。旧実装はタイムアウト
        時に引き上げるだけで、一度膨れた予算は二度と戻らなかった (モデルを軽い
        ものへ替えても、失敗 1 回のコストが 3 倍のまま)。直近窓の p95 なら
        実測が速くなれば自然に base 側へ戻る。
        """
        if not purpose or purpose in PURPOSE_TIMEOUT_CALIBRATION_EXEMPT:
            return
        samples = self._samples.setdefault(purpose, deque(maxlen=_CALIB_SAMPLE_WINDOW))
        samples.append(float(elapsed))
        if len(samples) < _CALIB_MIN_SAMPLES:
            return
        ordered = sorted(samples)
        p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
        base = PURPOSE_TIMEOUT_DEFAULTS.get(purpose, _DEFAULT_TIMEOUT)
        target = min(max(p95 * _CALIB_P95_HEADROOM, base * _CALIB_MIN_SCALE), base * _CALIB_MAX_SCALE)
        target = round(target, 1)
        current = self._calibrated.get(purpose, base)
        if abs(target - current) < 0.05 * base:
            return
        if target == base:
            self._calibrated.pop(purpose, None)
        else:
            self._calibrated[purpose] = target
        logger.info(
            "Aux timeout recalibrated for purpose=%s: %.1fs -> %.1fs "
            "(p95 %.1fs over %d samples, base %.1fs)",
            purpose, current, target, p95, len(samples), base,
        )
        self._persist_calibration()

    async def generate(
        self,
        messages: list[dict],
        *,
        stream: bool = False,  # noqa: ARG002 - 常に非ストリーミング (面の互換のため受ける)
        temperature: float | None = None,
        max_tokens: int | None = 256,
        id_slot: int | None = None,  # noqa: ARG002 - スロットは purpose 別方針で決める
        timeout: float | None = None,
        purpose: str = "",
        cache_prompt: bool = False,  # noqa: ARG002 - LocalClient 側で常時 ON
        response_format: dict | None = None,
        response_schema: type[BaseModel] | None = None,
        deferrable: bool | None = None,
    ) -> dict:
        """ベースモデルで非ストリーミング生成し、OAI 互換 dict を返す。

        json_schema が解決できる purpose は文法制約経路
        (``generate_constrained``) を通す。解決できない自由文の purpose
        (``summarize`` / ``create_spec_doc`` 等) は通常生成へ落とす。
        戻り値の ``choices[0].finish_reason`` は経路を問わず埋める
        (``"length"`` = 切断)。

        Raises:
            TimeoutError: 予算超過。下位が投げる ``httpx.TimeoutException`` /
                ``LLMTimeoutError`` は本例外へ正規化する — 呼出側は経路
                (制約あり / なし) を意識せず 1 つの degraded 分岐で受けられる。
                予算はスロットのロック取得後 (dispatch) から数える。
        """
        resolved = self._resolve_response_format(
            purpose, response_format, response_schema,
        )
        slot = self._slot_for(purpose)
        effective_timeout = (
            timeout if timeout is not None else self.resolve_effective_timeout(purpose)
        )
        queued_at = time.monotonic()
        if self._is_deferrable(purpose, deferrable):
            # ロックの **外** で待つ。ロックを持ったまま待つと、同じスロットの
            # 他の背景タスクまで道連れに直列化される。
            await wait_for_idle(_CHAT_YIELD_MAX_WAIT_SEC, purpose=purpose)
        gate_token = activity_token()
        async with self._lock_for(slot):
            started = time.monotonic()
            queue_wait = started - queued_at
            if queue_wait >= 1.0:
                logger.info(
                    "Aux request waited %.1fs for slot %d (purpose=%s)",
                    queue_wait, slot, purpose or "<unspecified>",
                )
            meta: dict = {}
            try:
                if resolved is not None:
                    content = await self.local.generate_constrained(
                        messages,
                        response_format=resolved,
                        temperature=0.1 if temperature is None else temperature,
                        max_tokens=max_tokens if max_tokens is not None else 256,
                        id_slot=slot,
                        timeout=effective_timeout,
                        result_meta=meta,
                    )
                    result = {"choices": [{
                        "message": {"content": content or ""},
                        "finish_reason": meta.get("finish_reason"),
                    }]}
                else:
                    result = await self.local.generate(
                        messages=messages,
                        stream=False,
                        temperature=0.3 if temperature is None else temperature,
                        max_tokens=max_tokens,
                        id_slot=slot,
                        request_timeout=effective_timeout,
                    )
                    if not isinstance(result, dict):
                        # stream=False なので dict のはずだが、差し替え実装の事故は握り潰さない
                        logger.warning(
                            "Base generate returned %s for purpose=%s; treating as empty",
                            type(result).__name__, purpose or "<unspecified>",
                        )
                        result = {"choices": [{"message": {"content": ""}}]}
            except (httpx.TimeoutException, LLMTimeoutError, TimeoutError) as e:
                # 競合由来の遅さを較正へ食わせない。チャットと重なると decode は
                # 実測で 2 倍以上遅くなるので、その所要時間を「この purpose に
                # 必要な予算」として学習すると、混雑が去っても膨れたままになる
                # (2026-09-03 監査: summarize が 54.6→81.9→122.9→59.0→88.5s と振動)。
                contended = was_contended_since(gate_token)
                if contended:
                    logger.info(
                        "Aux timeout for purpose=%s overlapped chat generation; "
                        "not calibrating (contention, not a budget shortfall)",
                        purpose or "<unspecified>",
                    )
                else:
                    self._bump_calibrated_timeout(purpose)
                # 呼出側はこの例外を握って縮退するので、ここで記録しないと
                # 「サイクルは成功、中身は全滅」が結末に残らない。
                record_aux_failure(
                    purpose, "timeout_contended" if contended else "timeout",
                )
                self._log_request(
                    messages, {}, time.monotonic() - started,
                    purpose=purpose, effective_timeout=effective_timeout,
                    constrained=resolved is not None, finish_reason="timeout",
                    queue_wait=queue_wait, slot=slot,
                )
                raise TimeoutError(
                    f"Aux generation timed out after {effective_timeout:.1f}s "
                    f"(purpose={purpose or '<unspecified>'})",
                ) from e

        elapsed = time.monotonic() - started
        # 成功側も同じ理由で競合サンプルを弾く (p95 が競合時の遅さで押し上がる)。
        if not was_contended_since(gate_token):
            self._record_success(purpose, elapsed)
        self._log_request(
            messages, result, elapsed,
            purpose=purpose, effective_timeout=effective_timeout,
            constrained=resolved is not None,
            queue_wait=queue_wait, slot=slot,
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
        deferrable: bool | None = None,
    ) -> dict:
        """JSON 出力を生成してパースする。パース不能時は空 dict を返す。

        ``finish_reason=length`` (max_tokens 到達で JSON が途中で切れた) は
        **修復しない**。``json_repair`` は ``[0, 1,`` を ``[0, 1]`` に閉じるので、
        欠けた要素が「無かった」ことになって静かに誤った判定が通る (content
        gate なら関連チャンクが落ちる)。空 dict で呼出側のフォールバックへ倒し、
        ``telemetry["json_truncated"] = True`` で観測できるようにする。
        """
        result = await self.generate(
            [{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            purpose=purpose,
            response_format=response_format,
            response_schema=response_schema,
            deferrable=deferrable,
        )
        content = _content_of(result)
        if not content.strip():
            return {}
        if _finish_reason_of(result) == "length":
            if telemetry is not None:
                telemetry["json_truncated"] = True
            logger.warning(
                "Aux JSON truncated at max_tokens=%d (purpose=%s, %d chars); "
                "returning empty result instead of repairing",
                max_tokens, purpose or "<unspecified>", len(content),
            )
            return {}

        repair_telemetry: dict = telemetry if telemetry is not None else {}
        parsed = extract_json_object(
            content, list_key=list_key, telemetry=repair_telemetry,
        )
        if parsed is None:
            logger.warning(
                "Aux JSON parse failed (purpose=%s, %d chars)",
                purpose or "<unspecified>", len(content),
            )
            return {}
        if repair_telemetry.get("repair_used") and self._debug_logger is not None:
            self._debug_logger.log_aux_json_repair(
                purpose=purpose,
                list_key=list_key,
                raw_preview=content,
                repaired_preview=str(parsed),
            )
        return parsed

    async def health_check(self) -> bool:
        """ベース llama-server のヘルスチェック。"""
        return await self.local.health_check()

    def _log_request(
        self,
        messages: list[dict],
        result: dict,
        elapsed: float,
        *,
        purpose: str,
        effective_timeout: float,
        constrained: bool,
        finish_reason: str = "",
        queue_wait: float = 0.0,
        slot: int | None = None,
    ) -> None:
        """``requests`` JSONL へ補助呼出を記録する (DebugLogger 未注入時は no-op)。

        ``queue_wait_sec`` / ``slot`` は ``log_aux_request`` がその kwarg を受ける
        場合だけ渡す (debug_logger.py は本経路の管轄外なので、面が追随するまで
        は落とす)。
        """
        if self._debug_logger is None:
            return
        content = _content_of(result)
        kwargs: dict[str, Any] = {
            "messages_count": len(messages),
            "response_preview": content,
            "elapsed_sec": elapsed,
            "purpose": purpose,
            "resolved_timeout": effective_timeout,
            "response_format_used": constrained,
            "finish_reason": finish_reason or _finish_reason_of(result),
            "response_length": len(content),
        }
        accepted = _accepted_kwargs(self._debug_logger.log_aux_request)
        if accepted is None or "queue_wait_sec" in accepted:
            kwargs["queue_wait_sec"] = round(queue_wait, 3)
        if slot is not None and (accepted is None or "slot" in accepted):
            kwargs["slot"] = slot
        self._debug_logger.log_aux_request(**kwargs)


def _accepted_kwargs(fn: Any) -> set[str] | None:
    """``fn`` が受け取るキーワード名の集合 (``**kwargs`` を受けるなら ``None``)。"""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return None
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return None
    return set(params)


def _content_of(result: dict) -> str:
    """OAI 互換 dict から ``choices[0].message.content`` を取り出す。"""
    if not isinstance(result, dict):
        return ""
    choices = result.get("choices") or [{}]
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") or {} if isinstance(choices[0], dict) else {}
    raw = message.get("content")
    return raw if isinstance(raw, str) else ""


def _finish_reason_of(result: dict) -> str:
    """``choices[0].finish_reason`` を取り出す。取得できなければ空文字列。"""
    if not isinstance(result, dict):
        return ""
    choices = result.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        fr = choices[0].get("finish_reason")
        if isinstance(fr, str):
            return fr
    return ""


def _resolve_base_model_filename(config: dict) -> str:
    """ベースモデルの GGUF ファイル名 (basename) を解決する。

    較正値を model-scoped に保存 / ロードするためのキー。解決できない場合は
    空文字列 (較正の永続化を無効化)。
    """
    raw = (config.get("model_paths") or {}).get("base_model", "")
    return Path(str(raw)).name if raw else ""


__all__ = [
    "AuxClient",
    "CHAT_PATH_PURPOSES",
    "PURPOSE_TIMEOUT_CALIBRATION_EXEMPT",
    "PURPOSE_TIMEOUT_DEFAULTS",
]
