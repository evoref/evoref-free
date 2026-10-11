"""日本語チャット応答への簡体字・ハングル混入を、生成時に禁止する (``logit_bias``)

ベースモデル (Qwen3.6-35B) は日本語の応答に簡体字の語 (「岚山」「杂音」) やハングルを
混ぜる (実機 3〜4%、長い応答ほど多い)。プロンプトでは直らない。**日本語に現れない
字を含むトークンだけ** を ``logit_bias`` で禁止すると、実機 (2026-10-11、282 生成) で
混入 6/141 → 0/141、速度 −0.8%、MTP 受理率・応答の中身は不変だった。

禁止する ID は、載っているモデルの GGUF 語彙から 1 度だけ作る (モデルごとに語彙が違う)。
文字列キーは字形がずれるので使わず、数値 ID だけを送る。部分 UTF-8 のバイトトークンは
日本語の漢字と共有するので禁止しない。

対象は **日本語ロケールのチャットモード** のユーザー応答パスだけ。問いが中国語・韓国語を
名指ししている / 当該の文字を含むときは送らない (``text_quality.scripts_allowed_by_query``、
混入の判定と同じ除外)。補助タスクと create は対象外。語彙が準備できるまで (起動直後の
数秒) は送らず、従来どおり動く。
"""

from __future__ import annotations

import re
import struct
import threading
import time
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any

from backend.free.core.script_ranges import HANGUL, simplified_only_hanzi
from backend.free.core.text_quality import scripts_allowed_by_query
from backend.log_config import get_logger

logger = get_logger("llm.leak_bias")

#: 禁止の強さ。-100 で実質ゼロ確率 (実機で 31,194 件を指定して問題なし)。
BAN_BIAS = -100

_GGUF_MAGIC = b"GGUF"
#: GGUF の値型 (固定長のものの大きさ)。8 = 文字列, 9 = 配列。
_FIXED_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_HANGUL_RE = re.compile(f"[{HANGUL}]")


def _read_str(f) -> bytes:
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n)


def _skip_value(f, vtype: int) -> None:
    if vtype in _FIXED_SIZES:
        f.seek(_FIXED_SIZES[vtype], 1)
    elif vtype == 8:
        (n,) = struct.unpack("<Q", f.read(8))
        f.seek(n, 1)
    elif vtype == 9:
        (etype,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        if etype in _FIXED_SIZES:
            f.seek(_FIXED_SIZES[etype] * n, 1)
        else:
            for _ in range(n):
                _skip_value(f, etype)
    else:
        raise ValueError(f"unsupported GGUF value type {vtype}")


def read_gguf_vocab(path: str | Path) -> tuple[str, list[bytes]]:
    """GGUF から ``(tokenizer.ggml.model, トークン文字列 (バイト列) の並び)`` を読む。

    メタデータだけを読み、テンソル本体には触れない。トークンが無ければ空の並び。
    """
    model = ""
    tokens: list[bytes] = []
    with open(path, "rb") as f:
        if f.read(4) != _GGUF_MAGIC:
            raise ValueError("not a GGUF file")
        (_version,) = struct.unpack("<I", f.read(4))
        _n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
        for _ in range(n_kv):
            key = _read_str(f).decode("utf-8", "replace")
            (vtype,) = struct.unpack("<I", f.read(4))
            if key == "tokenizer.ggml.model" and vtype == 8:
                model = _read_str(f).decode("utf-8", "replace")
            elif key == "tokenizer.ggml.tokens" and vtype == 9:
                (etype,) = struct.unpack("<I", f.read(4))
                (n,) = struct.unpack("<Q", f.read(8))
                if etype != 8:
                    raise ValueError("tokenizer.ggml.tokens is not a string array")
                tokens = [_read_str(f) for _ in range(n)]
            else:
                _skip_value(f, vtype)
    return model, tokens


@cache
def _byte_decoder() -> dict[str, int]:
    """GPT-2 の byte-level BPE が使う「バイト ↔ 表示用の文字」の逆引き。"""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs, strict=True)}


def build_ban_ids(path: str | Path) -> list[int]:
    """``path`` の語彙から、簡体字専用字またはハングルを含むトークンの ID を作る。

    byte-level BPE (``gpt2``) の語彙だけを扱う。それ以外の語彙 (SentencePiece 等) は
    表記が違うので禁止しない (空)。完全な UTF-8 に戻らないトークン (文字の途中の
    バイト) は日本語の漢字と共有するので含めない。
    """
    model, tokens = read_gguf_vocab(path)
    if model != "gpt2":
        return []
    decoder = _byte_decoder()
    simplified = set(simplified_only_hanzi())
    ids: list[int] = []
    for token_id, raw in enumerate(tokens):
        try:
            data = bytes(decoder[c] for c in raw.decode("utf-8"))
        except (KeyError, UnicodeDecodeError):
            continue
        text = data.decode("utf-8", "replace")
        if "�" in text:
            continue
        if _HANGUL_RE.search(text) or any(c in simplified for c in text):
            ids.append(token_id)
    return ids


def _user_texts(messages: list[dict[str, Any]]) -> list[str]:
    texts: list[str] = []
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            texts.extend(
                str(part.get("text") or "") for part in content if isinstance(part, dict)
            )
    return texts


class LeakBiasGuard:
    """載っているモデルの語彙から禁止 ID を作り、該当する要求へ ``logit_bias`` を返す。

    ``bias_for`` は同期・非ブロッキング (イベントループから呼ばれる)。語彙の読み込みは
    背景スレッドで 1 度だけ行い、終わるまでは ``None`` (= 従来どおり送らない)。
    """

    def __init__(
        self,
        *,
        enabled: bool,
        locale: str,
        mode_getter: Callable[[], str],
        served_model_getter: Callable[[], str],
        loader: Callable[[str], list[int]] = build_ban_ids,
    ) -> None:
        self._enabled = enabled
        self._locale = locale
        self._mode_getter = mode_getter
        self._served_model_getter = served_model_getter
        self._loader = loader
        self._lock = threading.Lock()
        #: モデルのパス → 禁止の指定 (読み込み中は ``None``、失敗・語彙が対象外は空)
        self._bias: dict[str, list[list[int]] | None] = {}

    def bias_for(self, messages: list[dict[str, Any]]) -> list[list[int]] | None:
        """この要求に送る ``logit_bias`` (送らないなら ``None``)。"""
        if not self._enabled or self._locale != "ja":
            return None
        if self._mode_getter() != "chat":
            return None
        path = self._served_model_getter()
        if not path:
            return None
        if any(scripts_allowed_by_query(text) for text in _user_texts(messages)):
            return None
        with self._lock:
            if path not in self._bias:
                self._bias[path] = None
                threading.Thread(
                    target=self._load, args=(path,), name="leak-bias-vocab", daemon=True,
                ).start()
                return None
            return self._bias[path] or None

    def _load(self, path: str) -> None:
        started = time.monotonic()
        try:
            ids = self._loader(path)
        except Exception as exc:  # noqa: BLE001 - 禁止の準備に失敗しても生成は止めない
            logger.warning("leak bias unavailable for %s: %s", Path(path).name, exc)
            ids = []
        bias = [[token_id, BAN_BIAS] for token_id in ids]
        with self._lock:
            self._bias[path] = bias
        logger.info(
            "leak bias ready for %s: %d token(s) in %.1fs",
            Path(path).name, len(bias), time.monotonic() - started,
        )


__all__ = [
    "BAN_BIAS",
    "LeakBiasGuard",
    "build_ban_ids",
    "read_gguf_vocab",
]
