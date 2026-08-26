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
補助処理を同じスロットへ割り込ませる」こと。本クライアントは常に
``LocalClient.background_slot`` を使い、チャットと KV を分離する。さらに
json_schema が解決できる purpose は文法制約経路 (``generate_constrained``) を
通すため、チャット応答パスから同期発火する補助判定も出力長が読める。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from backend.exceptions import LLMTimeoutError
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
    "policy_evolution": 120.0,
    "fewshot_quality_score": 45.0,
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

# 較正の引き上げ幅と上限倍率。per-attempt timeout を引き上げるため、
# 総ウォールクロックは ``per_attempt * リトライ数`` になる点を踏まえ控えめに保つ。
_CALIB_BUMP_FACTOR = 1.5
_CALIB_MAX_SCALE = 3.0


class AuxClient:
    """ベースモデル上で補助タスクを実行するクライアント。

    呼出側から見える約束:

    - ``id_slot`` / ``cache_prompt`` は面の互換のため受けるが無視する。スロットは
      常に ``LocalClient.background_slot`` (チャットと KV を分離)、``cache_prompt``
      は ``LocalClient`` 側で常時 ON
    - ``purpose`` から json_schema・タイムアウトを解決する
    - json_schema が解決できる purpose は文法制約経路を通る
    - 失敗は握り潰さず、``generate_json`` は空 dict を返して呼出側の
      フォールバックに委ねる
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
        #: 較正値の永続化キー (ベースモデルの GGUF ファイル名)。空なら永続化しない。
        self._model_filename = _resolve_base_model_filename(config or {})
        self._calibration_path = str(
            (config or {}).get("local_paths", {}).get(
                "aux_calibration_file", "local/aux_calibration.json",
            ),
        )
        self._calibrated: dict[str, float] = {}
        if self._model_filename:
            from backend.free.llm.aux_calibration_store import AuxCalibrationStore

            self._calibrated = AuxCalibrationStore.load_timeouts(
                self._calibration_path, self._model_filename,
            )
            if self._calibrated:
                logger.info(
                    "Loaded aux timeout calibration for model=%s (%d purposes)",
                    self._model_filename, len(self._calibrated),
                )

    @property
    def metadata(self):
        """モデルメタデータ (``run_full`` が ``params_b`` を読む)。"""
        return self.local.metadata

    @property
    def context_size(self) -> int:
        """ベースモデルの有効 context_size (サイズガードが参照する)。"""
        return int(getattr(self.local, "context_size", 8192) or 8192)

    def resolve_effective_timeout(self, purpose: str) -> float:
        """purpose に適用されるタイムアウト秒を返す (較正値込み)。"""
        calibrated = self._calibrated.get(purpose)
        if calibrated is not None:
            return calibrated
        return PURPOSE_TIMEOUT_DEFAULTS.get(purpose, _DEFAULT_TIMEOUT)

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

    def _bump_calibrated_timeout(self, purpose: str) -> None:
        """タイムアウト観測を受けて当該 purpose の予算を引き上げる。

        ベースは構成 (量子化 / GPU レイヤ / コンテキスト長) でレイテンシが数倍
        変わるため、コード既定値が実機に合わない構成が必ず出る。天井は既定値の
        ``_CALIB_MAX_SCALE`` 倍まで。
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
        if not self._model_filename:
            return
        try:
            from backend.free.llm.aux_calibration_store import AuxCalibrationStore

            AuxCalibrationStore.save_timeouts(
                self._calibration_path, self._model_filename, self._calibrated,
            )
        except OSError as e:
            logger.warning("Failed to persist aux calibration: %s", e)

    async def generate(
        self,
        messages: list[dict],
        *,
        stream: bool = False,  # noqa: ARG002 - 常に非ストリーミング (面の互換のため受ける)
        temperature: float | None = None,
        max_tokens: int | None = 256,
        id_slot: int | None = None,  # noqa: ARG002 - 常に background_slot
        timeout: float | None = None,
        purpose: str = "",
        cache_prompt: bool = False,  # noqa: ARG002 - LocalClient 側で常時 ON
        response_format: dict | None = None,
        response_schema: type[BaseModel] | None = None,
    ) -> dict:
        """ベースモデルで非ストリーミング生成し、OAI 互換 dict を返す。

        json_schema が解決できる purpose は文法制約経路
        (``generate_constrained``) を通す。解決できない自由文の purpose
        (``summarize`` / ``create_spec_doc`` 等) は通常生成へ落とす。

        Raises:
            TimeoutError: 予算超過。下位が投げる ``httpx.TimeoutException`` /
                ``LLMTimeoutError`` は本例外へ正規化する — 呼出側は経路
                (制約あり / なし) を意識せず 1 つの degraded 分岐で受けられる。
        """
        resolved = self._resolve_response_format(
            purpose, response_format, response_schema,
        )
        slot = self.local.background_slot
        effective_timeout = (
            timeout if timeout is not None else self.resolve_effective_timeout(purpose)
        )
        started = time.monotonic()

        try:
            if resolved is not None:
                content = await self.local.generate_constrained(
                    messages,
                    response_format=resolved,
                    temperature=0.1 if temperature is None else temperature,
                    max_tokens=max_tokens if max_tokens is not None else 256,
                    id_slot=slot,
                    timeout=effective_timeout,
                )
                result = {"choices": [{"message": {"content": content or ""}}]}
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
            self._bump_calibrated_timeout(purpose)
            self._log_request(
                messages, {}, time.monotonic() - started,
                purpose=purpose, effective_timeout=effective_timeout,
                constrained=resolved is not None, finish_reason="timeout",
            )
            raise TimeoutError(
                f"Aux generation timed out after {effective_timeout:.1f}s "
                f"(purpose={purpose or '<unspecified>'})",
            ) from e

        self._log_request(
            messages, result, time.monotonic() - started,
            purpose=purpose, effective_timeout=effective_timeout,
            constrained=resolved is not None,
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
        """JSON 出力を生成してパースする。パース不能時は空 dict を返す。"""
        result = await self.generate(
            [{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            purpose=purpose,
            response_format=response_format,
            response_schema=response_schema,
        )
        content = _content_of(result)
        if not content.strip():
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
    ) -> None:
        """``requests`` JSONL へ補助呼出を記録する (DebugLogger 未注入時は no-op)。"""
        if self._debug_logger is None:
            return
        content = _content_of(result)
        self._debug_logger.log_aux_request(
            messages_count=len(messages),
            response_preview=content,
            elapsed_sec=elapsed,
            purpose=purpose,
            resolved_timeout=effective_timeout,
            response_format_used=constrained,
            finish_reason=finish_reason or _finish_reason_of(result),
            response_length=len(content),
        )


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
    "PURPOSE_TIMEOUT_CALIBRATION_EXEMPT",
    "PURPOSE_TIMEOUT_DEFAULTS",
]
