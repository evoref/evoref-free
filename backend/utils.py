"""共通ユーティリティ関数"""

import re
import time
from datetime import datetime, timezone


def estimate_tokens(text: str) -> int:
    """高速トークン数推定（CJK: 1文字≒1トークン、ASCII: 4文字≒1トークン）"""
    if not text:
        return 0
    cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff')
    ascii_ = len(text) - cjk
    return cjk + ascii_ // 4


_SCOPE_PATTERN = re.compile(r'^(def |class )', re.MULTILINE)


def split_by_scope(text: str, budget: int) -> list[str]:
    """Pythonコードをdef/class境界でトークン予算内チャンクに分割する。

    設計書 f_02_memory_system.md §3.3 準拠。
    def/class境界が見つからない場合はテキスト全体を1チャンクとして返す。
    単一ブロックが予算を超える場合はそのまま1チャンクとして返す
    （関数途中で分割すると意味が壊れるため）。
    """
    if not text:
        return []

    positions = [m.start() for m in _SCOPE_PATTERN.finditer(text)]

    if not positions:
        # def/class境界なし: テキスト全体を1チャンクとして返す
        return [text]

    # モジュールレベルコード（import文等）を含めるため、
    # 先頭が0でなければ0を挿入
    if positions[0] > 0:
        positions.insert(0, 0)

    chunks: list[str] = []
    current = ''
    for i, pos in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(text)
        block = text[pos:end]
        if estimate_tokens(current + block) > budget:
            if current:
                chunks.append(current)
            current = block
        else:
            current += block
    if current:
        chunks.append(current)

    return chunks


def utc_now() -> str:
    """UTC タイムスタンプ (ISO 8601, 末尾 ``Z``)

    永続化・ログ・ファイル名で共通利用する文字列表現。内部処理で
    計算用の tz-aware :class:`datetime` が必要な場合は :func:`utc_now_dt`
    を使用すること
    """
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def utc_now_dt() -> datetime:
    """tz-aware な現在時刻 (UTC, :class:`datetime`)。

    サーバ実行 tz に依存しないよう、内部計算は全てこの
    ヘルパ経由で取得する。:func:`datetime.now` / :func:`datetime.utcnow`
    の直接呼び出しは ruff ``DTZ`` ルールで禁止する。
    """
    return datetime.now(timezone.utc)


def utc_compact_stamp() -> str:
    """ファイル名向けコンパクト UTC タイムスタンプ (``YYYYmmddTHHMMSSZ``)。"""
    return utc_now_dt().strftime("%Y%m%dT%H%M%SZ")


def compress_turn(
    turn: dict,
    *,
    max_chars: int = 200,
    style: str = "truncate",
) -> dict:
    """ターンのコンテンツを圧縮する。

    Args:
        turn: {"role": str, "content": str, ...} 形式のターン辞書
        max_chars: 圧縮上限文字数
        style:
            "truncate" — 先頭 max_chars 文字 + "..."（max_chars 以下はそのまま返す）
            "summary"  — "[要約] 先頭 max_chars 文字…（N文字）"（常に圧縮マーク付き）
    """
    content = turn.get("content", "")

    if style == "summary":
        summary = content[:max_chars].replace("\n", " ")
        if len(content) > max_chars:
            summary += f"…（{len(content)}文字）"
        return {**turn, "content": f"[要約] {summary}", "compressed": True}

    # style == "truncate"
    if len(content) <= max_chars:
        return turn
    return {**turn, "content": content[:max_chars] + "...", "compressed": True}
