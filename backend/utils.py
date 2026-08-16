"""共通ユーティリティ関数"""

import re
import time
from datetime import datetime, timezone


#: :func:`estimate_tokens` の **CJK 項だけ** に掛ける補正係数。既定 1.0 = 無補正。
#: 起動時に実トークナイザで較正した値を :func:`set_cjk_token_scale` が入れる。
#:
#: ASCII 項 (4 文字 ≒ 1 トークン) には掛けない。2026-08-16 実測の誤差は文種で
#: **向きが逆** だった: 日本語散文 1.55 倍の過大評価 / 英語散文 1.34 倍の過大評価 /
#: **Python コード 0.85 倍の過小評価**。全体に係数を掛けるとコードが 0.7 倍まで
#: 過小評価され、create 経路でコンテキスト超過 (400) を招く。系統的に外している
#: CJK 項だけを直し、ASCII 項は両方向に振れる = 平均では安全なので触らない。
_CJK_TOKEN_SCALE: float = 1.0

#: 較正係数の許容域。**上限は 1.0** — 素の推定より増やさない (予算超過側へ倒さない)。
#: 下限は、誤較正で予算が倍以上に膨らんで llama-server が 400 を返す事態のガード。
_TOKEN_SCALE_MIN = 0.5
_TOKEN_SCALE_MAX = 1.0

#: 較正値へ掛ける安全マージン。実測比ぴったりだと推定が実トークン数を下回りうる
#: (文章により 1.20〜1.93 倍とばらつく) ため、多めに見積もる側へ寄せる。
_TOKEN_SCALE_SAFETY = 1.10


def set_cjk_token_scale(measured_estimate: int, measured_actual: int) -> float:
    """**日本語サンプル** の実測比から CJK 項の補正係数を決める。

    素の推定は「CJK 1 文字 ≒ 1 トークン」だが、Qwen3 系の日本語では実測 1.5〜1.9 倍の
    **過大評価** になる (2026-08-16 実測: system プロンプト 推定 1973 / 実測 1300 =
    1.52 倍、履歴メッセージ 40 件の中央値 1.64 倍、日本語散文 1.55 倍)。予算は推定
    トークン建てなので、過大評価のぶんだけコンテキストを使い残したまま履歴を切り、
    RAG チャンクを落としていた (実測で約 2500 トークン相当)。

    **サンプルは日本語のみにすること。** ASCII を混ぜると比が薄まるうえ、
    コードは逆に過小評価側なので係数の意味が壊れる (:data:`_CJK_TOKEN_SCALE` 参照)。

    係数は **減らす方向のみ** に制限し、安全マージンを掛けて実トークン数を少し
    上回る側へ寄せる。較正不能 (0 除算・異常値) なら 1.0 のまま = 従来動作。

    Args:
        measured_estimate: 日本語サンプルの :func:`estimate_tokens` 素の推定値。
        measured_actual: 同じサンプルの実トークン数 (``/tokenize``)。

    Returns:
        適用した係数。
    """
    global _CJK_TOKEN_SCALE
    if measured_estimate <= 0 or measured_actual <= 0:
        return _CJK_TOKEN_SCALE
    ratio = (measured_actual / measured_estimate) * _TOKEN_SCALE_SAFETY
    _CJK_TOKEN_SCALE = max(_TOKEN_SCALE_MIN, min(_TOKEN_SCALE_MAX, ratio))
    return _CJK_TOKEN_SCALE


def get_cjk_token_scale() -> float:
    """現在の CJK 補正係数を返す (観測・テスト用)。"""
    return _CJK_TOKEN_SCALE


def estimate_tokens(text: str) -> int:
    """高速トークン数推定（CJK: 1文字≒1トークン、ASCII: 4文字≒1トークン）

    CJK 項のみ実トークナイザ較正済みの係数を掛ける
    (:func:`set_cjk_token_scale`)。未較正なら係数 1.0 = 従来どおり。
    """
    if not text:
        return 0
    cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff')
    ascii_ = len(text) - cjk
    raw = int(cjk * _CJK_TOKEN_SCALE) + ascii_ // 4
    return max(1, raw) if (cjk or ascii_ >= 4) else raw


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
