"""llama-server との通信クライアント"""

from collections.abc import AsyncIterator
from dataclasses import dataclass
import json

import httpx

from backend.exceptions import LLMConnectionError, LLMTimeoutError
from backend.free.llm._base_client import (
    BaseHTTPClient,
    HealthLogGate,
    MAX_ATTEMPTS,
    RETRYABLE_STATUS_CODES,
    async_retry_http_call,
    make_retry_logger,
)
from backend.free.llm.model_metadata import ModelMetadata
from backend.free.llm.utils import extract_content
from backend.log_config import get_logger
from backend.utils import estimate_tokens

logger = get_logger("llm.local_client")

# スロット定数
SLOT_CHAT = 0
SLOT_BACKGROUND = 1

__all__ = [
    "LocalClient",
    "SLOT_CHAT",
    "SLOT_BACKGROUND",
    "MAX_ATTEMPTS",
    "RETRYABLE_STATUS_CODES",
]

# ストリーミング設定
# 最初のデータまでの最大待機時間 (秒)。``LlamaConfig.stream_first_token_timeout_sec``
# のデフォルトと合わせる。冷えた KV キャッシュ + 長プロンプト + iGPU 環境を許容。
STREAM_FIRST_TOKEN_TIMEOUT = 60.0
STREAM_CONTENT_TIMEOUT_BASE = 30.0  # reasoning 開始から最初の content トークンまでの初期タイムアウト（秒）
STREAM_CONTENT_TIMEOUT_EXTEND = 10.0  # reasoning チャンク受信ごとの延長（秒）
STREAM_CONTENT_TIMEOUT_CAP = 120.0  # 最大タイムアウト上限（秒）


class _ReasoningFilter:
    """ストリーミングトークンから思考 (reasoning) 領域を除去するフィルタ。

    複数モデル系統の「思考テキスト/非 final 領域」が content
    フィールドに混入しても UI に漏らさない defense-in-depth として、
    次の 2 系統を単一ステートマシンで同時に処理する。

    1. Qwen3 / DeepSeek-R1 系: ``<think>...</think>`` ブロック
    2. Harmony (gpt-oss / llm-jp-4-thinking 等): チャンネル構造
       ``<|start|>assistant<|channel|>{analysis,commentary,final}<|message|>...<|end|>``
       - ``final`` チャンネルのみ通過
       - ``analysis`` / ``commentary`` / 未知のチャンネルは破棄
       - ``<|return|>`` はレスポンス終端マーカとして単に消費する

    llama-server の ``--reasoning-format`` で thinking が
    ``delta.reasoning_content`` に分離されているケースではこのフィルタは
    no-op となる。分離が無い / 設定漏れのケースでも UI に思考が漏れないよう、
    content ストリームに対しても必ずこのフィルタを通す。

    ストリーミングでトークン境界を跨ぐ部分マッチもバッファで安全に扱う。
    """

    # 状態
    _NORMAL = 0
    _IN_THINK = 1
    _HARMONY_HEADER = 2  # <|channel|> / <|start|> 後、<|message|> を待機
    _HARMONY_EMIT = 3    # final チャンネル本文を出力中
    _HARMONY_SUPPRESS = 4  # 非 final チャンネル本文を破棄中

    _THINK_OPEN = "<think>"
    _THINK_CLOSE = "</think>"
    _H_CHANNEL = "<|channel|>"
    _H_MESSAGE = "<|message|>"
    _H_END = "<|end|>"
    _H_START = "<|start|>"
    _H_RETURN = "<|return|>"

    # 部分マッチ判定に使う全トークン
    _ALL_TOKENS: tuple[str, ...] = (
        _THINK_OPEN, _THINK_CLOSE,
        _H_CHANNEL, _H_MESSAGE, _H_END, _H_START, _H_RETURN,
    )

    def __init__(self) -> None:
        self._state = self._NORMAL
        self._buffer = ""
        # HARMONY_HEADER に遷移した経路: True なら <|channel|> 直後
        # (header がチャンネル名そのもの)、False なら <|start|> 直後
        # (header は役割名 + 任意の <|channel|>name)
        self._harmony_via_channel = False

    @staticmethod
    def _safe_emit_len(buf: str, tokens: tuple[str, ...]) -> int:
        """``buf`` のうち、どのトークン先頭とも重ならない「確定して吐き出して良い」長さ。"""
        max_partial = 0
        for tok in tokens:
            for i in range(1, len(tok)):
                if buf.endswith(tok[:i]) and i > max_partial:
                    max_partial = i
        return len(buf) - max_partial

    @staticmethod
    def _find_earliest(buf: str, tokens: tuple[str, ...]) -> tuple[int, str]:
        """``tokens`` のうち ``buf`` 内で最も早く現れるものを返す (見つからなければ ``(-1, "")``)。"""
        earliest_idx = -1
        earliest_tok = ""
        for tok in tokens:
            idx = buf.find(tok)
            if idx >= 0 and (earliest_idx == -1 or idx < earliest_idx):
                earliest_idx = idx
                earliest_tok = tok
        return earliest_idx, earliest_tok

    def feed(self, token: str) -> str:
        """トークンを受け取り、思考/非 final 領域を除去したテキストを返す。"""
        self._buffer += token
        result: list[str] = []
        while self._step(result):
            pass
        return "".join(result)

    def _step(self, result: list[str]) -> bool:
        if not self._buffer:
            return False
        match self._state:
            case self._NORMAL:
                return self._step_normal(result)
            case self._IN_THINK:
                return self._step_in_think()
            case self._HARMONY_HEADER:
                return self._step_harmony_header()
            case self._HARMONY_EMIT:
                return self._step_harmony_emit(result)
            case self._HARMONY_SUPPRESS:
                return self._step_harmony_suppress()
        return False

    def _step_normal(self, result: list[str]) -> bool:
        # NORMAL では <think> / Harmony 制御トークンを検出
        idx, tok = self._find_earliest(
            self._buffer,
            (self._THINK_OPEN, self._H_CHANNEL, self._H_START,
             self._H_END, self._H_RETURN),
        )
        if idx >= 0:
            if idx > 0:
                result.append(self._buffer[:idx])
            self._buffer = self._buffer[idx + len(tok):]
            if tok == self._THINK_OPEN:
                self._state = self._IN_THINK
            elif tok == self._H_CHANNEL:
                self._state = self._HARMONY_HEADER
                self._harmony_via_channel = True
            elif tok == self._H_START:
                self._state = self._HARMONY_HEADER
                self._harmony_via_channel = False
            # <|end|> / <|return|> は NORMAL 中では単に境界として消費
            return True
        safe = self._safe_emit_len(self._buffer, self._ALL_TOKENS)
        if safe > 0:
            result.append(self._buffer[:safe])
            self._buffer = self._buffer[safe:]
            return True
        return False

    def _step_in_think(self) -> bool:
        idx = self._buffer.find(self._THINK_CLOSE)
        if idx >= 0:
            self._buffer = self._buffer[idx + len(self._THINK_CLOSE):]
            self._state = self._NORMAL
            return True
        safe = self._safe_emit_len(self._buffer, (self._THINK_CLOSE,))
        if safe > 0:
            self._buffer = self._buffer[safe:]
            return True
        return False

    def _step_harmony_header(self) -> bool:
        # <|message|> を待機。到達したらヘッダ文字列からチャンネル名を抽出
        idx = self._buffer.find(self._H_MESSAGE)
        if idx < 0:
            return False
        header = self._buffer[:idx]
        self._buffer = self._buffer[idx + len(self._H_MESSAGE):]
        if self._harmony_via_channel:
            # <|channel|> 直後: header はチャンネル名そのもの
            channel = header.strip().lower()
        elif self._H_CHANNEL in header:
            # <|start|> 経由で "role<|channel|>name" 形式
            channel = header.split(self._H_CHANNEL, 1)[1].strip().lower()
        else:
            # <|start|> 経由で channel マーカ無し → final 扱い (寛容フォールバック)
            channel = "final"
        if channel and channel != "final":
            self._state = self._HARMONY_SUPPRESS
        else:
            self._state = self._HARMONY_EMIT
        return True

    def _step_harmony_emit(self, result: list[str]) -> bool:
        idx, tok = self._find_earliest(
            self._buffer, (self._H_END, self._H_RETURN, self._H_START),
        )
        if idx >= 0:
            if idx > 0:
                result.append(self._buffer[:idx])
            self._buffer = self._buffer[idx + len(tok):]
            if tok == self._H_START:
                self._state = self._HARMONY_HEADER
                self._harmony_via_channel = False
            else:
                self._state = self._NORMAL
            return True
        safe = self._safe_emit_len(
            self._buffer, (self._H_END, self._H_RETURN, self._H_START),
        )
        if safe > 0:
            result.append(self._buffer[:safe])
            self._buffer = self._buffer[safe:]
            return True
        return False

    def _step_harmony_suppress(self) -> bool:
        idx, tok = self._find_earliest(
            self._buffer, (self._H_END, self._H_RETURN, self._H_START),
        )
        if idx >= 0:
            self._buffer = self._buffer[idx + len(tok):]
            if tok == self._H_START:
                self._state = self._HARMONY_HEADER
                self._harmony_via_channel = False
            else:
                self._state = self._NORMAL
            return True
        safe = self._safe_emit_len(
            self._buffer, (self._H_END, self._H_RETURN, self._H_START),
        )
        if safe > 0:
            self._buffer = self._buffer[safe:]
            return True
        return False

    def flush(self) -> str:
        """ストリーム終端処理。未終了の思考領域は破棄、emit 中の残りは出力。"""
        if self._state in (self._IN_THINK, self._HARMONY_HEADER, self._HARMONY_SUPPRESS):
            self._buffer = ""
            self._state = self._NORMAL
            return ""
        out = self._buffer
        self._buffer = ""
        self._state = self._NORMAL
        return out

    @property
    def in_think(self) -> bool:
        """現在 ``<think>`` 領域 (または Harmony 非 final チャンネル) 内か。

        watchdog (docs/c_15 B3) が「未閉じ思考が続いている」判定に使う。
        """
        return self._state in (self._IN_THINK, self._HARMONY_SUPPRESS)


@dataclass
class _ReasoningTimeoutTracker:
    """reasoning-only ストリームのタイムアウト追跡 (`_generate_stream` 用)。

    Qwen3 等の reasoning モデルでは `delta.reasoning_content` だけが流れて
    `delta.content` が出ないまま停止することがある。チャンクごとにタイムアウトを
    段階的に延長 (`STREAM_CONTENT_TIMEOUT_EXTEND`) し、上限 (`STREAM_CONTENT_TIMEOUT_CAP`)
    を超えたら打ち切って再試行を促す。
    """

    start: float | None = None
    timeout: float = STREAM_CONTENT_TIMEOUT_BASE
    count: int = 0

    def observe(self, now: float, token_count: int) -> bool:
        """reasoning チャンクを 1 件観測。打ち切るべきなら ``True`` を返す。

        - 初回観測時は `start` を記録するだけで打ち切らない
        - 2 回目以降はタイムアウトを段階的に延長
        - すでに content トークンが出ていれば (`token_count > 0`) 打ち切らない
        """
        self.count += 1
        if self.start is None:
            self.start = now
            return False
        self.timeout = min(
            self.timeout + STREAM_CONTENT_TIMEOUT_EXTEND,
            STREAM_CONTENT_TIMEOUT_CAP,
        )
        if token_count > 0:
            return False
        return (now - self.start) > self.timeout

    def elapsed(self, now: float) -> float:
        """`start` からの経過秒。未開始なら 0。"""
        if self.start is None:
            return 0.0
        return now - self.start


class LocalClient(BaseHTTPClient):
    """llama-server /v1/chat/completions クライアント

    KVキャッシュ最適化:
    - cache_prompt: 共通プレフィックスの再計算をスキップ
    - id_slot: チャットとバックグラウンドでスロットを分離し、KVキャッシュの相互汚染を防止
    - httpx.AsyncClient の再利用: TCP 接続のオーバーヘッドを削減
    """

    def __init__(
        self,
        llama_url: str,
        metadata: ModelMetadata,
        *,
        cache_prompt: bool = True,
        slots: int = 1,
        enable_thinking: bool | None = None,
        stream_first_token_timeout: float = STREAM_FIRST_TOKEN_TIMEOUT,
        debug_logger=None,
        client_think_budget: int = 0,
        on_runaway: str = "fallback",
        context_size: int | None = None,
    ):
        super().__init__(timeout=120.0)
        # health check の DEBUG を状態変化時のみに絞る (ポーリングでログが埋まるのを防ぐ)
        self._health_log_gate = HealthLogGate()

        self.url = llama_url
        self.metadata = metadata
        self._cache_prompt = cache_prompt
        self._slots = slots
        self._enable_thinking = enable_thinking
        self._stream_first_token_timeout = stream_first_token_timeout
        self._debug_logger = debug_logger
        # 送信前コンテキスト超過ガード用の n_ctx (config llama.context_size)。
        # None なら無効 (テスト経路 / 旧呼出互換)。
        self._context_size = context_size
        # 暴走 reasoning watchdog (docs/c_15 B3)。profile.reasoning から解決。
        # client_think_budget>0 のとき、未閉じ <think> がこの chunk 数を超えたら
        # ストリームを中断する (thinking=0 モデルの runaway 上限化)。サーバ側で
        # reasoning が分離されるモデルは content に <think> が出ないため発火しない。
        self._client_think_budget = max(0, int(client_think_budget or 0))
        self._on_runaway = on_runaway or "fallback"
        # モデル能力スナップショット (capability probe が背景で確定する; docs/c_15)。
        # 型: CapabilitySnapshot | None。未プローブ / プローブ無効時は None (prior 動作)。
        self.capabilities = None
        self._capability_probe_task = None
        # モデル能力スナップショット (capability probe が背景で確定する; docs/c_15)。
        # 型: CapabilitySnapshot | None。未プローブ / プローブ無効時は None (prior 動作)。
        self.capabilities = None
        self._capability_probe_task = None

    @property
    def chat_slot(self) -> int:
        """チャット用スロット ID

        常に -1（自動割当）を返す。
        以前は slots>=2 で SLOT_CHAT(0) を固定していたが、
        Qwen3 等の reasoning モデルで stale KV キャッシュが思考ループを引き起こす
        問題があるため、llama-server 側の自動割当に委譲する。
        """
        return -1

    @property
    def background_slot(self) -> int:
        """バックグラウンド用スロット ID（2スロット以上で 1、それ以外は -1=自動割当）"""
        return SLOT_BACKGROUND if self._slots >= 2 else -1

    def _apply_system_fallback(self, messages: list[dict]) -> list[dict]:
        """systemロール非対応モデル: systemをuserの先頭に結合"""
        if self.metadata.has_system_role:
            logger.debug("System role supported, passing messages as-is")
            return messages
        logger.debug("System role not supported, merging system messages into user")

        sys_msgs = [m for m in messages if m["role"] == "system"]
        rest = [m for m in messages if m["role"] != "system"]

        if not sys_msgs:
            return rest

        sys_text = "\n".join(m["content"] for m in sys_msgs)
        if rest and rest[0]["role"] == "user":
            rest[0] = {
                "role": "user",
                "content": sys_text + "\n\n" + rest[0]["content"],
            }
        elif rest:
            rest.insert(0, {"role": "user", "content": sys_text})
        else:
            rest = [{"role": "user", "content": sys_text}]

        return rest

    # 送信前ガードの安全マージン (チャットテンプレート展開 / 推定誤差の吸収分)
    _CTX_GUARD_MARGIN = 192
    # 1 メッセージあたりのテンプレートオーバーヘッド見積り (role トークン等)
    _CTX_GUARD_PER_MSG = 4
    # 単一メッセージ切り詰めの下限文字数 (これ以下には削らない)
    _CTX_GUARD_MIN_CONTENT_CHARS = 2000
    # max_tokens 込みで budget が負/ゼロになっても、プロンプトを丸ごと消し去る
    # ような病的なトリムに倒れないための下限。
    _CTX_GUARD_MIN_BUDGET_TOKENS = 256

    def _estimate_prompt_tokens(self, messages: list[dict]) -> int:
        """messages 全体のプロンプトトークン数を高速見積りする。"""
        return sum(
            estimate_tokens(str(m.get("content") or "")) + self._CTX_GUARD_PER_MSG
            for m in messages
        )

    def _enforce_context_budget(
        self, messages: list[dict], max_tokens: int | None = None,
    ) -> list[dict]:
        """送信前にプロンプトが n_ctx を超えないよう再トリムする。

        build_messages は静的予約でトリムするが、deliberative / meta_cognitive
        経路はその後にツール結果・既存ファイル内容等を追加注入するため、
        送信時点で n_ctx を超過して llama-server が HTTP 400 を返すことがある
        (2026-07-15: 5584 > 4096 でストリーム破壊、4204 > 4096 でフォールバック)。
        最終防衛として (1) 古い非 system メッセージの除去 → (2) 最大メッセージ
        の中間切除、の順で予算内へ収める。``context_size`` 未設定なら no-op。

        ``max_tokens`` (同一リクエストで要求する completion 上限) が渡された場合は
        あらかじめ budget から差し引く。プロンプト単体は budget 内でも、直後に
        max_tokens 分の completion を同一リクエストで要求すれば合計で n_ctx を
        超え、llama-server が ``exceed_context_size_error`` (HTTP 400) を返して
        生成そのものが失敗する (2026-07-22: singly_linked_list.py / bank_account.py
        で確認)。max_tokens が budget を独力で食い潰す場合は
        ``_CTX_GUARD_MIN_BUDGET_TOKENS`` を下限にクランプする。
        """
        if not self._context_size or not messages:
            return messages
        budget = self._context_size - self._CTX_GUARD_MARGIN - (max_tokens or 0)
        budget = max(budget, self._CTX_GUARD_MIN_BUDGET_TOKENS)
        estimate = self._estimate_prompt_tokens(messages)
        if estimate <= budget:
            return messages

        logger.warning(
            "Prompt exceeds context budget (est=%d > budget=%d, n_ctx=%d, "
            "reserved max_tokens=%s); re-trimming before send",
            estimate, budget, self._context_size, max_tokens,
        )
        # (1) 先頭の system 群と末尾メッセージ (現在ターン) を保持し、
        #     間の古いメッセージから順に落とす
        trimmed = list(messages)
        while len(trimmed) > 2 and self._estimate_prompt_tokens(trimmed) > budget:
            for i, m in enumerate(trimmed[:-1]):
                if m.get("role") != "system":
                    dropped = trimmed.pop(i)
                    logger.info(
                        "Context guard dropped %s message (%d chars)",
                        dropped.get("role"), len(str(dropped.get("content") or "")),
                    )
                    break
            else:
                break

        # (2) それでも超過する場合は最大メッセージの中間を切除する
        #     (system プロンプト先頭 / クエリ末尾のような重要部を残す)
        for _ in range(20):
            if self._estimate_prompt_tokens(trimmed) <= budget:
                break
            idx = max(
                range(len(trimmed)),
                key=lambda i: len(str(trimmed[i].get("content") or "")),
            )
            content = str(trimmed[idx].get("content") or "")
            if len(content) <= self._CTX_GUARD_MIN_CONTENT_CHARS:
                break
            keep_head = int(len(content) * 0.5)
            keep_tail = int(len(content) * 0.25)
            new_content = (
                content[:keep_head]
                + "\n…(コンテキスト予算超過のため中略)…\n"
                + content[-keep_tail:]
            )
            trimmed[idx] = {**trimmed[idx], "content": new_content}
            logger.info(
                "Context guard truncated %s message: %d -> %d chars",
                trimmed[idx].get("role"), len(content), len(new_content),
            )

        final = self._estimate_prompt_tokens(trimmed)
        if final > budget:
            logger.warning(
                "Context guard could not fully fit prompt (est=%d > budget=%d); "
                "sending as-is", final, budget,
            )
        return trimmed

    def _build_payload(
        self,
        messages: list[dict],
        *,
        stream: bool = True,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        repetition_penalty: float | None = None,
        id_slot: int | None = None,
        **extra,
    ) -> dict:
        """共通ペイロード構築"""
        msgs = self._apply_system_fallback(messages)
        msgs = self._enforce_context_budget(msgs, max_tokens=max_tokens)

        payload: dict = {
            "messages": msgs,
            "stream": stream,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if top_p is not None:
            payload["top_p"] = top_p
        if top_k is not None and top_k > 0:
            payload["top_k"] = top_k
        if presence_penalty is not None:
            payload["presence_penalty"] = presence_penalty
        if frequency_penalty is not None:
            payload["frequency_penalty"] = frequency_penalty
        if repetition_penalty is not None:
            payload["repetition_penalty"] = repetition_penalty

        # KVキャッシュ最適化
        if self._cache_prompt:
            payload["cache_prompt"] = True
        if id_slot is not None and id_slot >= 0:
            payload["id_slot"] = id_slot

        # Qwen3 等の reasoning モデル: thinking モード制御
        # enable_thinking=false でコンテキスト枯渇による思考ループを防止
        # 非 thinking モデルでは送信しない（llama-server バージョンによっては 400 になるため）
        if self._enable_thinking is not None:
            payload["chat_template_kwargs"] = {
                "enable_thinking": self._enable_thinking,
            }

        payload.update(extra)

        logger.debug(
            "Payload built: stream=%s, temperature=%.2f, max_tokens=%s, "
            "top_p=%s, top_k=%s, presence_penalty=%s, frequency_penalty=%s, "
            "repetition_penalty=%s, "
            "id_slot=%s, cache_prompt=%s, messages=%d, extra_keys=%s",
            stream, temperature, max_tokens,
            top_p, top_k, presence_penalty, frequency_penalty, repetition_penalty,
            id_slot, self._cache_prompt, len(msgs),
            list(extra.keys()) or "none",
        )
        return payload

    async def generate(
        self,
        messages: list[dict],
        stream: bool = True,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        repetition_penalty: float | None = None,
        id_slot: int | None = None,
        request_timeout: float | None = None,
    ) -> dict | AsyncIterator[str]:
        """llama-server に推論リクエストを送信

        Args:
            top_p: Top-P サンプリング（None で送信しない）
            top_k: Top-K サンプリング（None または 0 で送信しない）
            presence_penalty: 存在ペナルティ（None で送信しない）
            id_slot: KVキャッシュスロット指定。
                     chat_slot / background_slot プロパティを使用推奨。
                     None または -1 で自動割当。
            request_timeout: 非ストリーミング呼び出し (``stream=False``) 専用の
                     per-request タイムアウト上書き (秒)。既定 (None) は
                     ``self._http_timeout`` (120s)。大きな max_tokens を同期
                     生成する呼出側 (例: staged クリエイトの単発ファイル生成)
                     が、iGPU 等の低速環境で総生成時間が既定を超える場合に使う。
                     ストリーミング (``stream=True``) には影響しない。
        """
        payload = self._build_payload(
            messages,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repetition_penalty=repetition_penalty,
            id_slot=id_slot,
        )

        if stream:
            return self._generate_stream(payload)
        else:
            return await self._generate_sync(payload, request_timeout=request_timeout)

    async def count_tokens(self, text: str) -> int | None:
        """llama-server /tokenize でテキストの実トークン数を取得。

        transient I/O 失敗 (ConnectError / ReadTimeout / 5xx 等) は
        ``async_retry_http_call`` でリトライする。永続的な失敗 (4xx / その他例外)
        は char-based フォールバックに委ねるため ``None`` を返す。
        """
        if not text:
            return 0
        client = self._get_http_client()
        retry_logger = make_retry_logger(self._debug_logger, backend="base")

        async def _do_post() -> dict:
            resp = await client.post(
                f"{self.url}/tokenize",
                json={"content": text},
            )
            resp.raise_for_status()
            return resp.json()

        try:
            data = await async_retry_http_call(
                _do_post,
                request_label="LocalClient /tokenize",
                retry_logger=retry_logger,
            )
        except httpx.HTTPStatusError as e:
            logger.debug(
                "count_tokens: /tokenize returned HTTP %d", e.response.status_code,
            )
            return None
        except (httpx.HTTPError, ValueError) as e:
            logger.debug("count_tokens failed: %s", e)
            return None
        tokens = data.get("tokens")
        if isinstance(tokens, list):
            return len(tokens)
        return None

    async def _generate_sync(
        self, payload: dict, *, request_timeout: float | None = None,
    ) -> dict:
        """非ストリーミング推論（リトライ付き）

        ``request_timeout`` 指定時はこの呼び出しに限り per-request タイムアウトを
        上書きする (``async_retry_http_call`` は ``MAX_ATTEMPTS`` 回まで per-attempt
        でこのタイムアウトを適用するため、大きい値を渡すと worst case は
        ``request_timeout × MAX_ATTEMPTS`` の壁時計時間になり得る点に注意)。
        """
        client = self._get_http_client()
        logger.debug("Sync generate: POST %s/v1/chat/completions", self.url)

        async def _do_post() -> dict:
            post_kwargs: dict = {"json": payload}
            if request_timeout is not None:
                post_kwargs["timeout"] = request_timeout
            resp = await client.post(
                f"{self.url}/v1/chat/completions",
                **post_kwargs,
            )
            if resp.status_code >= 400:
                # raise_for_status は本文を載せないため、先に本文を抽出してログ + 例外に添付する
                body = resp.text[:500] if resp.text else "(empty response body)"
                logger.error(
                    "llama-server returned HTTP %d (sync): %s",
                    resp.status_code, body,
                )
                raise httpx.HTTPStatusError(
                    f"llama-server HTTP {resp.status_code}: {body}",
                    request=resp.request,
                    response=resp,
                )
            data = resp.json()
            content = extract_content(data)
            logger.debug(
                "Sync generate complete: response_length=%d chars",
                len(content),
            )
            return data

        try:
            return await async_retry_http_call(
                _do_post,
                request_label="LLM request",
            )
        except httpx.ConnectError as e:
            raise LLMConnectionError(
                f"llama-server unreachable: {e}", host=self.url,
            ) from e
        except httpx.TimeoutException as e:
            raise LLMTimeoutError(
                "llama-server timeout after retries", host=self.url,
            ) from e

    @staticmethod
    async def _check_stream_response(resp: httpx.Response) -> None:
        """ステータスエラー時にボディを読んで RuntimeError を投げる + content-type 警告。

        ストリーミング応答では `raise_for_status()` だとボディ未読のためエラー詳細が
        取得できない。手動でステータスを確認し、エラー時は `aread()` でボディを取得。
        """
        if resp.status_code >= 400:
            await resp.aread()
            body = resp.text[:500] if resp.text else "(empty response body)"
            logger.error(
                "llama-server returned HTTP %d: %s", resp.status_code, body,
            )
            raise RuntimeError(
                f"llama-server HTTP {resp.status_code}: {body}"
            )
        content_type = resp.headers.get("content-type", "")
        logger.debug(
            "Stream response: status=%d, content-type=%s",
            resp.status_code, content_type,
        )
        if "text/event-stream" not in content_type:
            logger.warning(
                "Unexpected content-type from llama-server: %s "
                "(expected text/event-stream). Response may not stream.",
                content_type,
            )

    @staticmethod
    def _parse_sse_delta(data: str) -> tuple[str, str, dict | None]:
        """SSE chunk JSON から `(content, reasoning, raw_chunk)` を抽出。

        パース失敗 / フィールド欠落時は `("", "", None)` を返し、呼び出し側で
        ログ警告を出す。`raw_chunk` は `finish_reason` 取得のため返す。
        """
        try:
            chunk = json.loads(data)
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content") or ""
            reasoning = delta.get("reasoning_content") or ""
            return content, reasoning, chunk
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.warning(
                "Failed to parse SSE chunk: error=%s, data=%s",
                e, data[:200],
            )
            return "", "", None

    async def _generate_stream(self, payload: dict) -> AsyncIterator[str]:
        """ストリーミング推論（async generator）

        接続プールの stale 接続問題を回避するため、毎回新規クライアントを使用。
        初回トークンまでのタイムアウト（STREAM_FIRST_TOKEN_TIMEOUT）を設け、
        ハング状態を早期検出する。

        Qwen3 等の reasoning モデルは delta.reasoning_content で思考トークンを送信し、
        その後 delta.content で回答トークンを送信する。両方を処理する。
        """
        import time as _time

        logger.debug("Stream generate: POST %s/v1/chat/completions", self.url)
        line_count = 0
        data_line_count = 0
        token_count = 0
        reasoning_timer = _ReasoningTimeoutTracker()
        reasoning_filter = _ReasoningFilter()
        first_data_received = False
        think_chunk_count = 0  # watchdog: 未閉じ <think> 内の連続 chunk 数 (docs/c_15 B3)
        try:
            # ストリーミング専用の新規クライアント（接続プール共有による stale 接続を回避）
            async with httpx.AsyncClient(timeout=120.0) as stream_client:
                async with stream_client.stream(
                    "POST",
                    f"{self.url}/v1/chat/completions",
                    json=payload,
                    timeout=httpx.Timeout(
                        connect=10.0,
                        read=self._stream_first_token_timeout,
                        write=10.0,
                        pool=5.0,
                    ),
                ) as resp:
                    await self._check_stream_response(resp)
                    async for line in resp.aiter_lines():
                        line_count += 1
                        if line_count <= 3 or (line.startswith("data: ") and data_line_count < 3):
                            logger.debug("SSE raw line #%d: %s", line_count, line[:300])
                        if not line.startswith("data: "):
                            continue
                        data_line_count += 1
                        data = line[6:]
                        if data.strip() == "[DONE]":
                            logger.debug(
                                "SSE [DONE]: lines=%d, data=%d, tokens=%d, reasoning=%d",
                                line_count, data_line_count, token_count, reasoning_timer.count,
                            )
                            tail = reasoning_filter.flush()
                            if tail:
                                yield tail
                            break

                        content, reasoning, chunk = self._parse_sse_delta(data)
                        if chunk is None:
                            continue

                        if content:
                            first_data_received = True
                            filtered = reasoning_filter.feed(content)
                            # 暴走 reasoning watchdog (docs/c_15 B3): 未閉じ <think> が
                            # client_think_budget chunk を超えたらストリームを中断する
                            # (thinking=0 モデルの runaway 上限化)。サーバ側で reasoning が
                            # 分離されるモデルは content に <think> が出ないため発火しない。
                            if reasoning_filter.in_think:
                                think_chunk_count += 1
                                if (
                                    self._client_think_budget
                                    and think_chunk_count > self._client_think_budget
                                ):
                                    logger.warning(
                                        "Reasoning watchdog: unclosed <think> exceeded %d "
                                        "chunks, aborting stream (on_runaway=%s)",
                                        self._client_think_budget, self._on_runaway,
                                    )
                                    # on_runaway はログ表示のみ。reask(二段生成)/truncate の
                                    # variant 別挙動は未配線で、現状いずれも stream 中断
                                    # (docs/c_15 §2.7)。
                                    break
                            else:
                                think_chunk_count = 0
                            if filtered:
                                token_count += 1
                                yield filtered
                            continue

                        if reasoning:
                            first_data_received = True
                            if reasoning_timer.observe(_time.monotonic(), token_count):
                                logger.warning(
                                    "Reasoning-only timeout: %.0fs without content tokens "
                                    "(reasoning=%d chunks, timeout=%.0fs). "
                                    "Aborting stream for retry.",
                                    reasoning_timer.elapsed(_time.monotonic()),
                                    reasoning_timer.count, reasoning_timer.timeout,
                                )
                                break
                            continue

                        # KV キャッシュヒット情報を usage チャンクから捕捉
                        if "usage" in chunk and self._debug_logger is not None:
                            usage = chunk["usage"]
                            cached = usage.get("prompt_tokens_details", {}).get("cached_tokens")
                            if cached is not None:
                                self._debug_logger.log_kv_cache(
                                    tokens_prompt=usage.get("prompt_tokens", 0),
                                    tokens_cached=cached,
                                )

                        if data_line_count <= 3:
                            finish = chunk["choices"][0].get("finish_reason")
                            logger.debug(
                                "SSE non-content chunk #%d: delta=%s, finish_reason=%s",
                                data_line_count,
                                chunk["choices"][0].get("delta", {}), finish,
                            )
                    else:
                        logger.warning(
                            "SSE stream ended without [DONE]: lines=%d, data=%d, tokens=%d, reasoning=%d",
                            line_count, data_line_count, token_count, reasoning_timer.count,
                        )
                        tail = reasoning_filter.flush()
                        if tail:
                            yield tail
        except httpx.ConnectError as e:
            raise LLMConnectionError(
                f"llama-server unreachable: {e}", host=self.url,
            ) from e
        except httpx.TimeoutException as e:
            if not first_data_received:
                logger.warning(
                    "First token timeout after %.0fs: lines=%d, data_lines=%d "
                    "(llama-server may have a stuck slot or resource contention; "
                    "increase llama.stream_first_token_timeout_sec if cold prefill "
                    "regularly exceeds this window)",
                    self._stream_first_token_timeout, line_count, data_line_count,
                )
            raise LLMTimeoutError(
                f"llama-server streaming timeout: {e}", host=self.url,
            ) from e

    async def generate_with_logprobs(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 256,
        id_slot: int | None = None,
    ) -> dict:
        """logprobs 付き非ストリーミング推論

        主推論パス (`_generate_sync`) と同様に
        ``async_retry_http_call`` で transient I/O 失敗をリトライする。

        Returns:
            {"content": str, "logprobs": list[float]}
        """
        payload = self._build_payload(
            messages,
            stream=False,
            temperature=temperature,
            max_tokens=max_tokens,
            id_slot=id_slot,
            logprobs=True,
            top_logprobs=1,
        )

        client = self._get_http_client()
        retry_logger = make_retry_logger(self._debug_logger, backend="base")

        async def _do_post() -> dict:
            resp = await client.post(
                f"{self.url}/v1/chat/completions",
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()

        data = await async_retry_http_call(
            _do_post,
            request_label="LocalClient /v1/chat/completions (logprobs)",
            retry_logger=retry_logger,
        )

        choice = data["choices"][0]
        content = choice["message"].get("content", "")

        logprobs: list[float] = []
        lp_data = choice.get("logprobs")
        if lp_data and "content" in lp_data:
            logprobs = [tok["logprob"] for tok in lp_data["content"]]

        return {"content": content, "logprobs": logprobs}

    async def generate_tool_call(
        self,
        messages: list[dict],
        *,
        tools: list[dict],
        temperature: float = 0.2,
        max_tokens: int = 256,
        id_slot: int | None = None,
        timeout: float | None = None,
    ) -> dict:
        """OAI ``tools`` を渡してツール選択だけをさせる非ストリーミング推論。

        ``assist_model.residency: on_demand`` のチャット中はアシストの
        ``tool_judgment`` が撃てないため、ベースモデル自身のネイティブ tool
        calling で代替する (docs/c_14 §1.3)。本文生成はしないので
        ``max_tokens`` は tool_call を吐ける程度で足りる。

        ``tools`` を受け付けない llama-server build / chat template では 4xx が
        返る。リトライしても回復しないため呼出側で 1 度だけ判定し、以後は
        ネイティブ経路を無効化すること。

        Returns:
            ``choices[0].message`` の dict。``tool_calls`` が無ければツール不要と
            モデルが判断したことを意味する。
        """
        payload = self._build_payload(
            messages,
            stream=False,
            temperature=temperature,
            max_tokens=max_tokens,
            id_slot=id_slot,
            tools=tools,
        )

        client = self._get_http_client()
        retry_logger = make_retry_logger(self._debug_logger, backend="base")

        async def _do_post() -> dict:
            resp = await client.post(
                f"{self.url}/v1/chat/completions",
                json=payload,
                timeout=timeout if timeout is not None else self._http_timeout,
            )
            resp.raise_for_status()
            return resp.json()

        data = await async_retry_http_call(
            _do_post,
            request_label="LocalClient /v1/chat/completions (tools)",
            retry_logger=retry_logger,
        )
        choices = data.get("choices") or [{}]
        message = choices[0].get("message") or {}

        # 予算を使い切って tool_call が無い応答は 100% 無駄撃ち (実測 28 秒)。
        # 判定させたつもりがモデルが本文を書き始めた場合に起きるので、次の監査で
        # 追えるよう記録する。戻り値の契約 (message dict) は変えない。
        if choices[0].get("finish_reason") == "length" and not message.get("tool_calls"):
            logger.warning(
                "Native tool calling exhausted max_tokens=%d without a tool call "
                "(model wrote prose instead of judging); falling through to no_tool",
                max_tokens,
            )
        return message

    async def health_check(self) -> bool:
        """llama-server のヘルスチェック"""
        try:
            client = self._get_http_client()
            resp = await client.get(f"{self.url}/health", timeout=5.0)
            healthy = resp.status_code == 200
            if self._health_log_gate.should_log(resp.status_code, healthy):
                logger.debug(
                    "Health check: status=%d, healthy=%s (suppressed %d identical)",
                    resp.status_code, healthy,
                    self._health_log_gate.take_suppressed(),
                )
            return healthy
        except (httpx.ConnectError, httpx.TimeoutException):
            logger.debug("Health check: connection failed")
            return False
