"""ストリーミング出力フィルタ: LLM の内部思考ブロック・装飾ラベルを除去する

ローカル LLM がシステムプロンプトの指示を無視して [内部思考] 等の
構造化ラベルを出力するケースに対応する。

ブロック検出はラベルパターンベースだが、汎用的なパターンを使用し
特定モデルに依存しない。

全フィルタは StreamFilter プロトコル（stream_pipeline.py）に準拠し、
StreamPipeline でチェーン適用できる。
"""

import re

from backend.log_config import get_logger

logger = get_logger("core.stream_filter")

# ---------------------------------------------------------------------------
# コンパイル済み正規表現パターン（モジュールレベルで1回だけコンパイル）
# ---------------------------------------------------------------------------

# 思考ブロック開始ラベル（行頭の [...] パターン）
# [内部思考], [分析], [アクション], [処理実行], [推論], [思考] 等
_THINKING_LABELS = re.compile(
    r"^\[(?:内部思考|思考|分析|推論|アクション|処理実行|状況判断|検証|事前検証"
    r"|要求解析|現状確認|エラー解析|Thinking|Analysis|Action|Reasoning)\]",
)

# 応答ブロック開始ラベル（思考ブロックの終了を示す）
_RESPONSE_LABELS = re.compile(
    r"^\[(?:応答|回答|結果報告|結論|Response|Answer|Result)\]",
)

# 装飾付き応答ラベル（行頭から除去する）
# 例: **回答:  **応答:**  **## 回答  回答:  ## 回答
_DECORATED_RESPONSE_LABEL = re.compile(
    r"^\*{0,2}\s*(?:#{1,3}\s+)?"
    r"(?:応答|回答|結果|結論|Response|Answer|Result)"
    r"[：:]?\s*\*{0,2}\s*",
)

# リスト項目の先頭文字（思考ブロック暗黙終了の判定で使用）
_LIST_PREFIXES = ("-", "*", "•")


class StreamThinkingFilter:
    """ストリーミングトークンから思考ブロックをフィルタリングする

    token 単位で呼ばれ、思考ブロック内のトークンを抑制する。
    行単位でバッファリングし、ラベルを検出する。

    StreamFilter プロトコルに準拠: process() / flush()

    Usage:
        f = StreamThinkingFilter()
        for token in stream:
            filtered = f.process(token)
            if filtered:
                yield filtered
        remaining = f.flush()
        if remaining:
            yield remaining
    """

    def __init__(self) -> None:
        self._in_thinking = False
        self._line_buffer = ""
        self._suppressed_chars = 0
        self._blank_line_count = 0  # 思考ブロック内の連続空行数

    def process(self, token: str) -> str:
        """トークンをフィルタリングする

        Args:
            token: 入力トークン

        Returns:
            出力すべきテキスト。思考ブロック内なら空文字列。
        """
        if not token:
            return ""

        result_parts: list[str] = []

        for char in token:
            self._line_buffer += char

            if char != "\n":
                continue

            # 行が完成 → ラベル判定
            line = self._line_buffer
            self._line_buffer = ""
            stripped = line.strip()

            if _THINKING_LABELS.match(stripped):
                self._in_thinking = True
                self._blank_line_count = 0
                self._suppressed_chars += len(line)
                continue

            if _RESPONSE_LABELS.match(stripped):
                self._in_thinking = False
                self._blank_line_count = 0
                self._suppressed_chars += len(line)
                continue

            if self._in_thinking:
                if not stripped:
                    self._blank_line_count += 1
                elif self._check_implicit_exit(line, stripped):
                    self._in_thinking = False
                    self._blank_line_count = 0
                    result_parts.append(line)
                    continue
                else:
                    self._blank_line_count = 0

                self._suppressed_chars += len(line)
                continue

            result_parts.append(line)

        # 改行なしのバッファは保留（次の token で行が完成するまで待つ）
        # ただし思考ブロック外かつラベル候補でない場合は出力する（応答性のため）
        if self._line_buffer and not self._in_thinking:
            # [ で始まる行はラベル候補なので改行まで保留
            if not self._line_buffer.lstrip().startswith("["):
                result_parts.append(self._line_buffer)
                self._line_buffer = ""

        return "".join(result_parts)

    def flush(self) -> str:
        """残りのバッファをフラッシュする"""
        if self._line_buffer and not self._in_thinking:
            result = self._line_buffer
            self._line_buffer = ""
            return result
        if self._suppressed_chars > 0:
            logger.info(
                "StreamThinkingFilter: suppressed %d chars of thinking blocks",
                self._suppressed_chars,
            )
        self._line_buffer = ""
        return ""

    def _check_implicit_exit(self, line: str, stripped: str) -> bool:
        """空行の後のインデントなし通常テキストかどうかを判定する

        思考ブロックの暗黙的な終了条件:
        空行が1つ以上あった後に、インデントなし・リスト項目でない・
        数字始まりでない通常テキストが来たら思考ブロック終了。
        """
        return (
            self._blank_line_count > 0
            and not line[0].isspace()
            and not stripped.startswith(_LIST_PREFIXES)
            and not stripped[0].isdigit()
        )


class RepetitionGuardFilter:
    """同一行の暴走反復を検知して以降の出力を打ち切る。

    サンプリング側のペナルティ (``frequency_penalty``) は反復の確率を下げるだけで
    ゼロにはできず、実測 2026-07-25 のライブ検証では ``frequency_penalty=0.3``
    を適用した状態でも、ユーザーの質問文をそのまま 20 回複製し続ける応答が
    発生した (回答全体がユーザー自身の質問の羅列に置き換わり、実行済みの
    ツール結果も反映されない)。確率的な緩和とは独立した決定論的な安全網として、
    同一行が連続で規定回数現れた時点で以降を捨てる。

    誤爆を避けるため、判定は以下を満たす行に限る:

    - 空白除去後 ``_MIN_LINE_CHARS`` 文字以上 (箇条書きの短い項目や
      区切り線、表の罫線を巻き込まない)
    - 直前に出力した行と完全一致 (間に別の行が挟まればカウントはリセット)

    StreamFilter プロトコルに準拠: process() / flush()
    """

    _MIN_LINE_CHARS = 12
    _MAX_CONSECUTIVE = 3

    def __init__(self) -> None:
        self._buffer = ""
        self._last_line = ""
        self._repeat_count = 1
        self._tripped = False

    @property
    def tripped(self) -> bool:
        """反復を検知して打ち切ったかどうか"""
        return self._tripped

    def _line_is_repeat(self, line: str) -> bool:
        """行を消費し、打ち切るべきなら True を返す。"""
        stripped = line.strip()
        if len(stripped) < self._MIN_LINE_CHARS:
            # 短い行は正当な繰り返し (箇条書き記号・空行) がありうるため
            # カウント対象にせず、連鎖も切らない
            return False
        if stripped == self._last_line:
            self._repeat_count += 1
        else:
            self._last_line = stripped
            self._repeat_count = 1
        return self._repeat_count >= self._MAX_CONSECUTIVE

    def process(self, text: str) -> str:
        """テキストを供給し、打ち切り後は空文字列を返す。"""
        if self._tripped:
            return ""
        if not text:
            return ""
        self._buffer += text
        out: list[str] = []
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if self._line_is_repeat(line):
                self._tripped = True
                self._buffer = ""
                logger.warning(
                    "RepetitionGuardFilter: truncated output after %d identical "
                    "lines (line=%.40s)",
                    self._repeat_count, self._last_line,
                )
                return "".join(out)
            out.append(line + "\n")
        return "".join(out)

    def flush(self) -> str:
        """残りバッファを排出する (打ち切り済みなら空)。"""
        if self._tripped:
            return ""
        remaining, self._buffer = self._buffer, ""
        return remaining


class HeadBufferFilter:
    """先頭バッファ: 最初の改行までバッファリングし、装飾ラベルを除去

    LLM が出力しがちな「**回答:** 」「## 応答」等の装飾ラベルを
    先頭行から除去する。

    StreamFilter プロトコルに準拠: process() / flush()
    """

    # 装飾ラベル (_DECORATED_RESPONSE_LABEL) は最長でも "**### Response:** "
    # 程度 (~20 文字)。これを超えてもなお改行が来ない場合は
    # 「改行を含まない長い単文応答」なので、バッファを強制的に吐き出して
    # トークン逐次表示を復活させる (メトリクス tok/s も改善)。
    _MAX_BUFFER_CHARS = 64

    def __init__(self) -> None:
        self._buffer = ""
        self._flushed = False

    @property
    def flushed(self) -> bool:
        """先頭行のフラッシュが完了したかどうか"""
        return self._flushed

    def process(self, text: str) -> str:
        """テキストを供給し、出力可能なテキストを返す

        Args:
            text: 入力テキスト

        Returns:
            出力すべきテキスト。まだバッファリング中なら空文字列。
        """
        if not text:
            return ""
        if self._flushed:
            return text
        self._buffer += text
        if "\n" in self._buffer:
            # 改行始まりエッジケース: バッファが空白のみの場合は
            # 装飾ラベルが後続トークンに含まれる可能性があるため、
            # フラッシュを保留してバッファリングを継続する
            if not self._buffer.strip():
                return ""
            self._buffer = self._remove_label(self._buffer)
            self._flushed = True
            return self._buffer if self._buffer else ""
        # 改行なしでバッファが上限を超えた場合は強制フラッシュ
        # (装飾ラベルは先頭 ~20 文字で完結するため、64 文字溜まっても
        # ラベルが現れないなら本文が続くだけと判断してよい)
        if len(self._buffer) >= self._MAX_BUFFER_CHARS:
            self._buffer = self._remove_label(self._buffer)
            self._flushed = True
            return self._buffer if self._buffer else ""
        return ""

    def flush(self) -> str:
        """バッファを強制フラッシュ（ストリーム終了時に呼ぶ）

        Returns:
            残っていたテキスト。空なら空文字列。
        """
        if not self._flushed and self._buffer:
            self._buffer = self._remove_label(self._buffer)
            self._flushed = True
            return self._buffer if self._buffer else ""
        return ""

    @staticmethod
    def _remove_label(text: str) -> str:
        """先頭の空白を除去してから装飾ラベルを除去し、元の先頭空白を復元する

        _DECORATED_RESPONSE_LABEL の ^ アンカーは文字列先頭にマッチするため、
        先頭に改行がある場合はラベルを検出できない。先頭空白を一時的に除去
        してからラベル検出を行い、除去された先頭空白を復元する
        """
        stripped = text.lstrip()
        if not stripped:
            return text
        leading = text[:len(text) - len(stripped)]
        cleaned = _DECORATED_RESPONSE_LABEL.sub("", stripped)
        return leading + cleaned


def strip_thinking_blocks(text: str) -> str:
    """完成したテキストから思考ブロックを一括除去する（非ストリーミング用）"""
    lines = text.split("\n")
    result: list[str] = []
    in_thinking = False
    blank_count = 0

    for line in lines:
        stripped = line.strip()

        if _THINKING_LABELS.match(stripped):
            in_thinking = True
            blank_count = 0
            continue
        if _RESPONSE_LABELS.match(stripped):
            in_thinking = False
            blank_count = 0
            continue
        if in_thinking:
            if not stripped:
                blank_count += 1
            elif (
                blank_count > 0
                and not line[0].isspace()
                and not stripped.startswith(_LIST_PREFIXES)
                and not stripped[0].isdigit()
            ):
                in_thinking = False
                blank_count = 0
                result.append(line)
                continue
            else:
                blank_count = 0
            continue
        result.append(line)

    filtered = "\n".join(result).strip()
    if len(filtered) < len(text.strip()):
        logger.info(
            "strip_thinking_blocks: removed %d chars",
            len(text.strip()) - len(filtered),
        )
    return filtered
