"""llama-server との通信クライアント"""

from collections.abc import AsyncIterator
from dataclasses import dataclass
import asyncio
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


def truncation_notice() -> str:
    """max_tokens 到達でストリームが途中終了したことをユーザーへ開示する注記。

    i18n_helper は横断基盤なので pillar から参照してよい。``msg`` は未解決キーを
    **キー文字列のまま返す** 仕様なので、それを検出して素の英語へ縮退させる
    (i18n 未初期化のパスでも注記そのものは消さない — **切断を黙って完結扱いに
    しない**ことが本注記の目的)。

    この文字列は **表示専用** で、応答本文へは決して連結しない
    (:class:`StreamOutcome` の docstring 参照)。SSE を持たない表示経路
    (CLI) だけが呼ぶ。
    """
    key = "warning.llm.response_truncated"
    fallback = "\n\n> ⚠ Output token limit reached; this response is cut off."
    try:
        from backend.i18n_helper import msg
        text = msg(key)
    except Exception:  # pragma: no cover - i18n 未初期化時の保険
        return fallback
    return fallback if text == key else text


@dataclass
class StreamOutcome:
    """ストリーム終端で確定するメタ情報。**本文には決して混ぜない**。

    かつて ``finish_reason == "length"`` の開示は :func:`truncation_notice` を
    content ストリームへ ``yield`` することで行っていた。注記はそのまま
    ``full_response`` に積まれ、履歴 / WM / STM / experience / few-shot まで
    **モデル自身の出力として保存** されていた。実インシデント
    (2026-08-25 セッション ``20260825_045637``): 512 トークンで切れた小説に
    注記が付いて履歴へ入り、次ターンの「続けて」でモデルが末尾ブロックごと
    **注記を逐語コピー** した (backend.log にその 2 ターンの
    ``Stream hit max_tokens`` 警告は無く、注記はモデルが書いたもの)。

    非ストリーム経路 (``_generate_sync``) は同じ理由で最初から本文へ足さず
    ログだけ残す方針だった。ストリーム経路をその方針に揃える。
    """

    #: llama-server が返した最後の ``finish_reason`` (未観測なら ``None``)。
    finish_reason: str | None = None
    #: フィルタ前の生トークン数 (デバッグ / 開示用)。
    tokens_generated: int = 0
    #: このリクエストで指定した ``max_tokens`` (未指定なら ``None``)。
    max_tokens: int | None = None

    @property
    def truncated(self) -> bool:
        """max_tokens 到達で文の途中で切れたか。"""
        return self.finish_reason == "length"


class TokenStream:
    """``AsyncIterator[str]`` + 終端メタ (:attr:`outcome`)。

    既存の消費側は ``async for token in stream`` のまま **str だけ** を受け取る。
    切断を扱う消費側だけがループ後に ``stream.outcome.truncated`` を見る。
    センチネル値を本文ストリームへ混ぜる設計は採らない — 見落とした消費側が
    ``chunks.append(token)`` / ``text += token`` で壊れる (``meta_cognitive`` /
    ``strategy_cogwriter`` / ``task_exec`` が該当)。
    """

    __slots__ = ("_agen", "outcome")

    def __init__(self, agen: AsyncIterator[str], outcome: StreamOutcome) -> None:
        self._agen = agen
        self.outcome = outcome

    def __aiter__(self) -> "TokenStream":
        return self

    async def __anext__(self) -> str:
        return await self._agen.__anext__()

    async def aclose(self) -> None:
        """基底 async generator を閉じる (呼出側の早期 break 用)。"""
        aclose = getattr(self._agen, "aclose", None)
        if aclose is not None:
            await aclose()


# スロット定数
SLOT_CHAT = 0
SLOT_BACKGROUND = 1
#: ツール分類器 (層 5.9) 専用スロット。``llama.slots >= 3`` のときだけ使う。
SLOT_CLASSIFIER = 2

__all__ = [
    "LocalClient",
    "StreamOutcome",
    "TokenStream",
    "truncation_notice",
    "SLOT_CHAT",
    "SLOT_BACKGROUND",
    "SLOT_CLASSIFIER",
    "MAX_ATTEMPTS",
    "RETRYABLE_STATUS_CODES",
]

# ストリーミング設定
# 最初のデータまでの最大待機時間 (秒)。``LlamaConfig.stream_first_token_timeout_sec``
# のデフォルトと合わせる。冷えた KV キャッシュ + 長プロンプト + iGPU 環境を許容。
STREAM_FIRST_TOKEN_TIMEOUT = 60.0
#: 最初のトークンまでの締め切りに **プロンプト長ぶんを足す** ときの prefill 速度
#: (tok/s)。締め切りは ``stream_first_token_timeout + prompt_tokens / この値``。
#:
#: 固定 60 秒だと、プロンプトが伸びた分だけ prefill が延びて **健全な要求まで
#: 打ち切る**。実インシデント (2026-08-31 ライブ監査 T07#3): 長文生成の直後、
#: 履歴 67 ターン (約 4,500 トークン) の要約依頼が
#: ``First token timeout after 60s: lines=2, data_lines=0`` で落ち、ユーザーには
#: ``Error: llama-server streaming timeout:`` だけが表示された (ターンは失われた)。
#: llama-server は詰まっていない — prefill が 60 秒に収まらなかっただけ。
#:
#: 同じ環境の実測は 70〜80 tok/s。壁時計の上限そのものは残さないと 51.9 分の
#: ハング (2026-08-31 t06#10) を取り逃すので、**比例項を足すだけ** にする。
STREAM_PREFILL_TOKENS_PER_SEC = 40.0
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
        #: 直近ストリームの llama-server ``timings`` (prompt_n / cache_n 等)。
        #: 接頭辞 KV キャッシュの効きを測れる唯一の一次情報で、これが無いと
        #: llama-base.stderr.log の行とチャットターンを突き合わせるしかない。
        self._last_timings: dict | None = None

    @property
    def chat_slot(self) -> int:
        """チャット用スロット ID（2スロット以上で 0、それ以外は -1=自動割当）

        ``background_slot`` (=1) と物理的に分離し、aux / sleep-time の呼び出しが
        会話の KV を踏まないようにする (CLAUDE.md §6 #1「チャットと KV を分離」)。

        一時期 **常に -1** を返していた。Qwen3 系で stale KV キャッシュが思考
        ループを誘発したための回避で、自動割当を llama-server に委ねていた。
        この前提は既に無い — thinking は ``llama.enable_thinking`` (既定 false、
        ``resolve_enable_thinking`` が解決) で切っており、2026-08-16 のライブ監査
        でも生成トークン数と本文トークン数が一致していた (eval 767 / 本文 762 =
        reasoning 出力ゼロ)。一方で自動割当の実害は大きく、同監査では会話が
        slot 0 と slot 1 の間を渡り歩き、aux が slot 0 を取った直後のターンで
        cache 18.4% / prefill 125 秒の全損が出ていた。
        """
        return SLOT_CHAT if self._slots >= 2 else -1

    @property
    def background_slot(self) -> int:
        """バックグラウンド用スロット ID（2スロット以上で 1、それ以外は -1=自動割当）"""
        return SLOT_BACKGROUND if self._slots >= 2 else -1

    @property
    def classifier_slot(self) -> int:
        """ツール分類器 (層 5.9) 用スロット ID。3 スロット未満なら背景スロット。

        **なぜ専用スロットが要るか**: 分類器のプロンプトは
        「ツールメニュー (385 トークン、毎回同一) + 直近会話 + クエリ」で、
        メニュー部分は接頭辞キャッシュに完全に乗る形をしている。ところが
        背景スロットを sleep-time / aux と共有しているため、ターンとターンの
        あいだに走る補助タスクが毎回そのプレフィクスを追い出す。結果、
        **分類器は毎回 cache_n=0 でフルプリフィル**していた。

        実測 (2026-08-25、Qwen3.8-27B / iGPU、prefill 約 16 tok/s):

            追い出される側 (背景スロット)   422 tok / cache 0   … 37.7 秒
            追い出されない側 (専用スロット) 同一クエリ再送     … 10.2 秒
            同上・**別のクエリ**  28 tok / cache 390          … 11.3 秒

        別のクエリでも 390/418 トークンが再利用でき、**1 回あたり約 26 秒**
        (3.3 倍) 縮む。層 5.9 はチャット遅延の最大成分なので効果が直接出る。

        3 スロット未満の構成では従来どおり背景スロットへ倒す (退行しない)。
        """
        if self._slots >= 3:
            return SLOT_CLASSIFIER
        return self.background_slot

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

    def _first_token_budget(self, messages: list[dict]) -> float:
        """最初のトークンを待つ秒数 (プロンプト長に比例させた締め切り)。

        ``stream_first_token_timeout`` を **下限** とし、prefill に要する分
        (``prompt_tokens / STREAM_PREFILL_TOKENS_PER_SEC``) を足す。壁時計の
        上限そのものは残るので、スロット詰まりの長時間ハングは従来どおり
        捕まる (``STREAM_PREFILL_TOKENS_PER_SEC`` の説明を参照)。
        """
        base = self._stream_first_token_timeout
        if not messages:
            return base
        try:
            tokens = self._estimate_prompt_tokens(messages)
        except Exception:
            return base
        return base + tokens / STREAM_PREFILL_TOKENS_PER_SEC

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
        # 最終チャンクに usage を載せてもらう (OAI 標準)。
        # ``usage.prompt_tokens_details.cached_tokens`` が接頭辞 KV キャッシュの
        # 再利用量で、``_generate_stream`` の usage ハンドラと
        # ``DebugLogger.log_kv_cache`` は元からこれを待っていたが、この 1 行が
        # 無いため llama-server が usage チャンクを送らず、**キャッシュ計測が
        # 丸ごと死んでいた** (``GET /api/status`` の cache_hit_rate が常に 0.0)。
        if stream:
            payload["stream_options"] = {"include_usage": True}
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

        # chat_template_kwargs は extra ではなく payload へ直接入るため、
        # extra_keys だけを出すと enable_thinking を送ったのかが分からない
        # (2026-08-16 の監査では eval トークン数から逆算する羽目になった)。
        logger.debug(
            "Payload built: stream=%s, temperature=%.2f, max_tokens=%s, "
            "top_p=%s, top_k=%s, presence_penalty=%s, frequency_penalty=%s, "
            "repetition_penalty=%s, "
            "id_slot=%s, cache_prompt=%s, chat_template_kwargs=%s, "
            "messages=%d, extra_keys=%s",
            stream, temperature, max_tokens,
            top_p, top_k, presence_penalty, frequency_penalty, repetition_penalty,
            id_slot, self._cache_prompt,
            payload.get("chat_template_kwargs") or "none", len(msgs),
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
    ) -> dict | TokenStream:
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
            # TokenStream は str のイテレータのまま (既存の消費側は無改変)。
            # 切断を扱う消費側だけがループ後に ``.outcome`` を見る。
            outcome = StreamOutcome()
            return TokenStream(self._generate_stream(payload, outcome), outcome)
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
            if (data.get("choices") or [{}])[0].get("finish_reason") == "length":
                # 非ストリーミング経路は補助タスク (JSON 応答) にも使われるため
                # **本文へ注記を足さない** (パースを壊す)。診断できるようログだけ
                # 残す。ユーザーへの開示はチャット応答が通るストリーム経路で行う。
                logger.warning(
                    "Sync generate hit max_tokens=%s (finish_reason=length); "
                    "output is truncated (%d chars)",
                    payload.get("max_tokens"), len(content),
                )
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
            # ``stream_options.include_usage`` を付けると llama-server は最終チャンクを
            # **choices 空配列 + usage** で送る (OAI 仕様)。``choices[0]`` を無条件に
            # 引くと IndexError になり、chunk ごと捨てられて usage ハンドラまで
            # 届かない。**KV キャッシュ計測が丸ごと死んでいた原因がこれ**。
            choices = chunk.get("choices") or []
            if not choices:
                return "", "", chunk
            delta = choices[0].get("delta", {}) or {}
            content = delta.get("content") or ""
            reasoning = delta.get("reasoning_content") or ""
            return content, reasoning, chunk
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.warning(
                "Failed to parse SSE chunk: error=%s, data=%s",
                e, data[:200],
            )
            return "", "", None

    async def _generate_stream(
        self, payload: dict, outcome: StreamOutcome,
    ) -> AsyncIterator[str]:
        """ストリーミング推論（async generator）

        ``outcome`` は終端メタの書き戻し先。**本文へは何も足さない** ため、
        切断の開示は呼出側が ``outcome.truncated`` を見て行う
        (:class:`StreamOutcome` 参照)。

        接続プールの stale 接続問題を回避するため、毎回新規クライアントを使用。
        初回トークンまでのタイムアウト（STREAM_FIRST_TOKEN_TIMEOUT）を設け、
        ハング状態を早期検出する。

        Qwen3 等の reasoning モデルは delta.reasoning_content で思考トークンを送信し、
        その後 delta.content で回答トークンを送信する。両方を処理する。
        """
        import time as _time

        # 直近 timings を捨ててから開始する。``_last_timings`` は「このクライアント
        # が最後に観測した値」でしかないため、今回のリクエストが usage を伴わずに
        # 終わると **前回のターンの値がそのまま読まれる**。実測 (2026-08-18): 連続
        # 2 ターンが同一の prompt=2337 / cached=3 で記録され、2 ターン目は生成を
        # 通っていなかった。未計測は None のまま残すのが正しい (消費側は「未計測」
        # と「消費ゼロ」を区別する)。
        self._last_timings = None

        logger.debug("Stream generate: POST %s/v1/chat/completions", self.url)
        line_count = 0
        data_line_count = 0
        token_count = 0
        reasoning_timer = _ReasoningTimeoutTracker()
        reasoning_filter = _ReasoningFilter()
        first_data_received = False
        think_chunk_count = 0  # watchdog: 未閉じ <think> 内の連続 chunk 数 (docs/c_15 B3)
        #: 最後に観測した ``finish_reason``。``"length"`` なら max_tokens 到達で
        #: 応答が文の途中で切れている。ストリーム経路はこれを見ておらず
        #: (先頭 3 チャンクの DEBUG ログのみ)、**切断がユーザーにもログにも
        #: 一切出ていなかった**。2026-08-16 ライブ監査ターン 14: 「README.md は
        #: 存在しますか？」に対し全文復唱を始め、ちょうど 1,024 トークン
        #: (llama.max_tokens の既定値) で「| llama-server (base / embed) | 8080 / 8」
        #: と表の途中で停止。197 秒かけた回答が未完のまま、完結した回答として
        #: 提示されていた。
        last_finish_reason: str | None = None
        outcome.max_tokens = payload.get("max_tokens")
        # 最初のトークンまでの締め切りは **プロンプト長に比例させる**
        # (``STREAM_PREFILL_TOKENS_PER_SEC`` の説明を参照)。固定値だと、
        # 履歴が伸びた分だけ延びる prefill を「ハング」と誤判定して健全な
        # 要求を打ち切る。
        first_token_budget = self._first_token_budget(payload.get("messages") or [])
        started_at = _time.monotonic()
        try:
            # ストリーミング専用の新規クライアント（接続プール共有による stale 接続を回避）
            async with httpx.AsyncClient(timeout=120.0) as stream_client:
                async with stream_client.stream(
                    "POST",
                    f"{self.url}/v1/chat/completions",
                    json=payload,
                    timeout=httpx.Timeout(
                        connect=10.0,
                        read=first_token_budget,
                        write=10.0,
                        pool=5.0,
                    ),
                ) as resp:
                    await self._check_stream_response(resp)
                    # ``httpx`` の ``read`` タイムアウトは **バイト間の空き** に
                    # 掛かるもので、最初のトークンまでの壁時計時間は縛れない。
                    # llama-server がスロット待ちで接続を開いたまま少しずつ
                    # 何かを流すと、read タイマは毎回リセットされて発火しない。
                    #
                    # 実インシデント (2026-08-31 ライブ監査 t06#10):
                    # ``read=60s`` を設定していたのに、ストリーム開始
                    # 04:58:45 → タイムアウト 05:50:36 と **51.9 分** 待たされ、
                    # ユーザーには空応答が返った (``lines=0, data_lines=0`` の
                    # まま = 1 行も届いていない)。ログは "after 60s" と出るので
                    # 実際の待ち時間が分からず、二重に紛らわしい。
                    #
                    # 最初のデータが来るまでは壁時計で締め切り、来たら解除して
                    # 以降は従来どおり read タイムアウトに委ねる。
                    async with asyncio.timeout(
                        first_token_budget,
                    ) as first_token_deadline:
                        async for line in resp.aiter_lines():
                            if first_data_received:
                                first_token_deadline.reschedule(None)
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
                            reason = (chunk.get("choices") or [{}])[0].get("finish_reason")
                            if reason:
                                last_finish_reason = reason
                                outcome.finish_reason = reason
                            # llama.cpp が timings を載せる構成なら、より正確なそちらを優先。
                            # (既定では付かないので通常は下の usage 経路が使われる)
                            timings = chunk.get("timings")
                            if isinstance(timings, dict) and "cache_n" in timings:
                                self._last_timings = timings

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
                            if "usage" in chunk:
                                usage = chunk["usage"] or {}
                                details = usage.get("prompt_tokens_details") or {}
                                cached = details.get("cached_tokens")
                                if cached is not None:
                                    prompt_tokens = usage.get("prompt_tokens", 0)
                                    # 再評価分 = プロンプト全体 - 再利用分。
                                    # requests.jsonl の timing へ畳んで、ターン単位で
                                    # 接頭辞キャッシュの効きを追えるようにする。
                                    self._last_timings = {
                                        "prompt_n": max(0, prompt_tokens - cached),
                                        "cache_n": cached,
                                    }
                                    if self._debug_logger is not None:
                                        self._debug_logger.log_kv_cache(
                                            tokens_prompt=prompt_tokens,
                                            tokens_cached=cached,
                                        )

                            if data_line_count <= 3:
                                # usage チャンクは choices 空配列なので添字を引かない
                                first = (chunk.get("choices") or [{}])[0]
                                finish = first.get("finish_reason")
                                logger.debug(
                                    "SSE non-content chunk #%d: delta=%s, finish_reason=%s",
                                    data_line_count,
                                    first.get("delta", {}), finish,
                                )
                        else:
                            logger.warning(
                                "SSE stream ended without [DONE]: lines=%d, data=%d, tokens=%d, reasoning=%d",
                                line_count, data_line_count, token_count, reasoning_timer.count,
                            )
                            tail = reasoning_filter.flush()
                            if tail:
                                yield tail

                    outcome.tokens_generated = token_count
                    if last_finish_reason == "length":
                        # 切断を黙って完結扱いにしない。ログ (英語固定) はここで、
                        # ユーザーへの開示は呼出側が ``outcome.truncated`` を見て
                        # **本文の外** (SSE フレーム / CLI の別行) で行う。
                        # 本文へ注記を連結すると履歴へ保存され、次ターンで
                        # モデルが復唱する (StreamOutcome の docstring 参照)。
                        logger.warning(
                            "Stream hit max_tokens=%s (finish_reason=length); "
                            "the response is cut mid-sentence after %d tokens",
                            payload.get("max_tokens"), token_count,
                        )
        except httpx.ConnectError as e:
            raise LLMConnectionError(
                f"llama-server unreachable: {e}", host=self.url,
            ) from e
        except (httpx.TimeoutException, TimeoutError) as e:
            # ``TimeoutError`` は上の ``asyncio.timeout`` (最初のトークンまでの
            # 壁時計の締め切り) 由来。httpx の read タイムアウトと同じ結末へ
            # 落とす — 呼出側から見ればどちらも「llama-server が応答しない」。
            waited = _time.monotonic() - started_at
            if not first_data_received:
                logger.warning(
                    "First token timeout after %.1fs (budget %.0fs): "
                    "lines=%d, data_lines=%d "
                    "(llama-server may have a stuck slot or resource contention; "
                    "increase llama.stream_first_token_timeout_sec if cold prefill "
                    "regularly exceeds this window)",
                    waited, first_token_budget, line_count, data_line_count,
                )
            # ``asyncio.TimeoutError`` は ``str(e)`` が空。そのまま埋め込むと
            # ユーザーには ``Error: llama-server streaming timeout:`` という
            # 中身の無い行だけが表示される (2026-08-31 ライブ監査 T07#3 実測)。
            # 何秒待って諦めたのかを必ず書く。
            detail = str(e) or (
                f"no data for {waited:.0f}s (budget {first_token_budget:.0f}s)"
                if not first_data_received
                else f"stream stalled after {waited:.0f}s"
            )
            raise LLMTimeoutError(
                f"llama-server streaming timeout: {detail}", host=self.url,
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

    async def generate_constrained(
        self,
        messages: list[dict],
        *,
        response_format: dict,
        temperature: float = 0.1,
        max_tokens: int = 64,
        id_slot: int | None = None,
        timeout: float | None = None,
    ) -> str | None:
        """``response_format`` (json_schema) で文法制約した非ストリーミング生成。

        OAI ``tools`` を使わない理由は **強制力**。実測
        (2026-08-12, Qwen3.5-27B / gemma-4-12b) では ``tools`` は 200 で受理
        されてもモデルが tool_call を出さずに本文を書き始め、``max_tokens`` を
        使い切って 15.6〜60.2 秒を捨てることがある (``tool_choice="required"``
        でも 6 件中 3 件で無視された)。一方 ``response_format`` の json_schema は
        llama-server 側の GBNF 制約なので、**出力が必ずスキーマに従い、
        トークン数の上限が読める**。判定系はこちらを使う。

        Returns:
            ``choices[0].message.content`` の文字列。取得できなければ ``None``。
            スキーマを強制しない build / chat template では非 JSON が返り得るので、
            呼出側は必ずパース失敗を許容すること (capability probe が
            ``json_schema grammar not enforced`` を警告する構成が実在する)。
        """
        payload = self._build_payload(
            messages,
            stream=False,
            temperature=temperature,
            max_tokens=max_tokens,
            id_slot=id_slot,
            response_format=response_format,
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
            request_label="LocalClient /v1/chat/completions (json_schema)",
            retry_logger=retry_logger,
        )
        choices = data.get("choices") or [{}]
        message = choices[0].get("message") or {}
        content = message.get("content")
        if choices[0].get("finish_reason") == "length":
            logger.warning(
                "Constrained generation hit max_tokens=%d; output is likely "
                "truncated mid-JSON", max_tokens,
            )
        return content if isinstance(content, str) and content.strip() else None

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
