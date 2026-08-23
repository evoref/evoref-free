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

    加えて「ブロック単位の巡回反復」も検知する。連続一致だけでは、番号や
    見出しが挟まって完全連続にならない反復を取りこぼす (実測 2026-07-27:
    引っ越し手続きの箇条書きで項目 5〜8 がそのまま 9〜12 として再出力された。
    見出し行は採番だけが違い、説明行は完全一致していた)。行頭の採番・箇条書き
    記号を正規化したうえで、既出行の再出現が ``_MAX_DUPLICATE_LINES`` 回
    たまった時点で打ち切る。コードフェンス内は正当な重複行 (``return None``
    等) が普通に現れるためカウントしない。

    StreamFilter プロトコルに準拠: process() / flush()
    """

    _MIN_LINE_CHARS = 12
    _MAX_CONSECUTIVE = 3
    _MAX_DUPLICATE_LINES = 4
    _LIST_MARKER_RE = re.compile(r"^(?:\d{1,3}[.)]|[-*+・>#]+)\s*")

    #: 1 行の中で同一トークンが連続してよい回数の上限。判定が行単位なので
    #: 「ador ador ador ador …」のように **1 行に収まる暴走** は素通りしていた
    #: (実インシデント 2026-08-10 ライブ監査: 日本三景の 2 項目目が "ador" の
    #: 連呼になり、そのまま画面に出た)。長文生成側には同種のガード
    #: (generation.validators.collapse_runaway_repetition) があるが、チャットの
    #: ストリームには繋がっていない。正当な散文で同じ短語が 6 回続くことはない。
    _MAX_TOKEN_RUN = 6

    #: 改行を待たずに先出しを始める、未確定行の文字数。
    #:
    #: 判定は行単位なので素朴には改行が来るまで何も出せないが、それだと
    #: **改行を含まない長い段落は最後まで 1 文字も表示されない**。実測
    #: (2026-08-08 ライブ監査、gemma-4-12b/iGPU): 300 字の説明と会議メモの要約で、
    #: llama-server が最初のトークンを出してからクライアントに最初の SSE が届く
    #: までが 23.8 秒 / 22.2 秒だった (他 31 ターンの中央値は 3.3 秒)。
    #: バックエンドは生成できているのに画面が空のままになる。
    #:
    #: 一方この閾値より短い未確定行は従来どおり改行まで待つ。反復ガードが守る
    #: 対象 (箇条書き・見出し・質問文の複製) はいずれも短い行で、そこでの挙動を
    #: 変えないため。閾値を超えた行は既に「反復ではない散文」とみなせる長さで、
    #: 先出ししても打ち切り判定そのものは行完成時に従来どおり効く。
    _EAGER_FLUSH_CHARS = 120

    def __init__(self, query: str | None = None) -> None:
        self._buffer = ""
        #: ``_buffer`` の先頭から何文字を既に先出ししたか (行完成時にリセット)。
        self._emitted_in_line = 0
        self._last_line = ""
        self._repeat_count = 1
        self._tripped = False
        self._seen_lines: set[str] = set()
        self._duplicate_count = 0
        self._in_code_fence = False
        # ユーザーの質問文そのものの反復は、長さに関わらず正当な繰り返しでは
        # ない。下限 (_MIN_LINE_CHARS) は箇条書きや区切り線を巻き込まないため
        # のものだが、短い質問文が素通りする穴になっていた (実インシデント
        # 2026-08-04 ライブ監査: 10 文字の「今日は何曜日ですか。」が 5 回
        # 繰り返され、答えが 1 文字も出なかった)。
        self._query_line = (query or "").strip()

    @classmethod
    def _normalize(cls, line: str) -> str:
        """行頭の採番・箇条書き記号と強調記号を落として比較用に正規化する。"""
        stripped = cls._LIST_MARKER_RE.sub("", line.strip())
        return stripped.replace("*", "").replace("　", " ").strip()

    @classmethod
    def _has_token_runaway(cls, line: str) -> bool:
        """1 行の中で同一トークンが連続しすぎているか (純粋関数)。"""
        prev = ""
        run = 1
        for token in line.split():
            if token == prev:
                run += 1
                if run >= cls._MAX_TOKEN_RUN:
                    return True
            else:
                prev, run = token, 1
        return False

    @property
    def tripped(self) -> bool:
        """反復を検知して打ち切ったかどうか"""
        return self._tripped

    def _line_is_repeat(self, line: str) -> bool:
        """行を消費し、打ち切るべきなら True を返す。"""
        stripped = line.strip()
        if stripped.startswith("```"):
            self._in_code_fence = not self._in_code_fence
            return False
        is_query_line = bool(self._query_line) and stripped == self._query_line
        # 行内の暴走は長さに関わらず打ち切る (コード中は 0 や | の連続が正当)。
        if not self._in_code_fence and self._has_token_runaway(stripped):
            logger.warning(
                "RepetitionGuardFilter: truncated output after an in-line token "
                "runaway (line=%.40s)", stripped,
            )
            return True
        if len(stripped) < self._MIN_LINE_CHARS and not is_query_line:
            # 短い行は正当な繰り返し (箇条書き記号・空行) がありうるため
            # カウント対象にせず、連鎖も切らない
            return False
        if stripped == self._last_line:
            self._repeat_count += 1
        else:
            self._last_line = stripped
            self._repeat_count = 1
        if self._repeat_count >= self._MAX_CONSECUTIVE:
            return True
        if self._in_code_fence:
            return False
        normalized = self._normalize(stripped)
        if len(normalized) < self._MIN_LINE_CHARS and not is_query_line:
            return False
        if normalized in self._seen_lines:
            self._duplicate_count += 1
            if self._duplicate_count >= self._MAX_DUPLICATE_LINES:
                logger.warning(
                    "RepetitionGuardFilter: truncated output after %d duplicated "
                    "lines (line=%.40s)",
                    self._duplicate_count, normalized,
                )
                return True
        else:
            self._seen_lines.add(normalized)
        return False

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
                self._emitted_in_line = 0
                logger.warning(
                    "RepetitionGuardFilter: truncated output after %d identical "
                    "lines (line=%.40s)",
                    self._repeat_count, self._last_line,
                )
                return "".join(out)
            # 先出し済みの分は二重に出さない。
            out.append(line[self._emitted_in_line:] + "\n")
            self._emitted_in_line = 0
        # 長い未確定行は改行を待たずに先出しする (_EAGER_FLUSH_CHARS 参照)。
        if len(self._buffer) >= self._EAGER_FLUSH_CHARS:
            out.append(self._buffer[self._emitted_in_line:])
            self._emitted_in_line = len(self._buffer)
        return "".join(out)

    def flush(self) -> str:
        """残りバッファを排出する (打ち切り済みなら空)。"""
        if self._tripped:
            return ""
        remaining = self._buffer[self._emitted_in_line:]
        self._buffer = ""
        self._emitted_in_line = 0
        return remaining


class QueryEchoFilter:
    """先頭に現れたユーザー質問の逐語コピーを落とす。

    質問をそのまま復唱してから答える崩れが実在する。``RepetitionGuardFilter``
    は同一行の反復を打ち切るが、1 回だけの復唱は反復にならず素通りする。また
    反復の判定には ``_MIN_LINE_CHARS`` の下限があり、短い質問文
    (「今日は何曜日ですか。」= 10 文字) は複数回繰り返されても数えられない
    (実インシデント 2026-08-04 ライブ監査)。

    本文の冒頭に限って落とす。途中に現れる引用 (「『〜』というご質問ですね」)
    は正当なので触らない。``query`` が空か短すぎる場合は素通しする。

    StreamFilter プロトコルに準拠: process() / flush()
    """

    #: 復唱の判定に載せる質問の下限。これ未満は「はい」等の相槌で、応答が
    #: 偶然同じ語で始まっただけの誤爆になりうる。
    _MIN_QUERY_CHARS = 8

    #: 復唱の前に許す前置きの長さ。呼びかけ (「小川さん、」) を挟んでから復唱
    #: する形が実在する (実測 2026-08-04:「私の名前を覚えていますか。」に対し
    #: 「小川さん、私の名前を覚えていますか。」)。短い前置きに限るのは、問いを
    #: 引用してから答える正常な応答を巻き込まないため。
    _MAX_LEAD_CHARS = 16

    def __init__(self, query: str | None = None) -> None:
        self._query = (query or "").strip()
        self._active = len(self._query) >= self._MIN_QUERY_CHARS
        self._buffer = ""
        self._flushed = not self._active
        self._lead_resolved = not self._active

    def _strip_leading_echoes(self, text: str) -> str:
        out = text.lstrip()
        while out.startswith(self._query):
            out = out[len(self._query):].lstrip()
        return out

    def _resolve_lead(self) -> bool:
        """前置きの有無を確定する。まだ判断材料が足りなければ False。

        呼びかけを挟んだ復唱 (「小川さん、<質問>」) と、引用してから答える形
        (「はい。『<質問>』というご質問ですね」) は括弧の有無で分かれる。
        引用符に囲まれていれば言及なので触らない。
        """
        head = self._buffer.lstrip()
        if head.startswith(self._query):
            self._lead_resolved = True
            return True
        idx = head.find(self._query)
        if 0 < idx <= self._MAX_LEAD_CHARS:
            if head[idx - 1] in "「『\"'“‘":
                # 引用符付き = 質問への言及。復唱ではない。
                self._lead_resolved = True
                return True
            self._buffer = head[idx:]
            self._lead_resolved = True
            return True
        # 前置き + 質問文が全部届くまでは判断できない。
        if len(head) < len(self._query) + self._MAX_LEAD_CHARS:
            return False
        self._lead_resolved = True
        return True

    def process(self, text: str) -> str:
        if not text or self._flushed:
            return text
        self._buffer += text
        if not self._lead_resolved and not self._resolve_lead():
            return ""
        stripped = self._strip_leading_echoes(self._buffer)
        if not stripped:
            # 復唱を消費し切った状態。続きが更なる復唱か本文かはまだ
            # 分からないので、判定を続ける。
            self._buffer = ""
            return ""
        if self._query.startswith(stripped):
            # 質問文の前方一致 = 復唱の途中かもしれない。全部届くまで待つ。
            self._buffer = stripped
            return ""
        self._flushed = True
        self._buffer = ""
        return stripped

    def flush(self) -> str:
        if self._flushed:
            return ""
        self._flushed = True
        if not self._lead_resolved:
            self._resolve_lead()
        head = self._buffer.lstrip()
        self._buffer = ""
        return self._strip_leading_echoes(head)


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


#: 内部の根拠枠 ([参考情報 N] / [関連する記憶] / ツール実行結果) への言い及び。
#: 括弧書きの出典表記 (「（参考情報1に基づく）」) と、文中の言及の両方を拾う。
#: 枠名は ``core.inference`` の ``_RAG_HEADER`` / ``_SEMMEM_BLOCK_LABEL`` と
#: ``agent.deliberative`` のツール結果ヘッダに対応する。
_INTERNAL_FRAME_WORDS = r"参考情報|関連する記憶|ツール実行結果|参照情報"

#: 括弧でくくられた出典表記。丸ごと落としても文は成立する。
_INTERNAL_FRAME_PAREN_RE = re.compile(
    r"[（(]\s*(?:" + _INTERNAL_FRAME_WORDS + r")\s*\d*\s*"
    r"(?:に基づく|より|参照|から|を参照|に記載)?\s*[）)]",
)

#: 括弧なしの読点区切りの出典表記 (「参考情報1によると、」)。読点まで落とす。
_INTERNAL_FRAME_CLAUSE_RE = re.compile(
    r"(?:" + _INTERNAL_FRAME_WORDS + r")\s*\d*\s*"
    r"(?:に基づくと|によると|に基づけば|に記載のとおり|を踏まえると)\s*[、,]?\s*",
)


def strip_internal_frame_mentions(text: str) -> str:
    """内部の根拠枠への言及を落とす (純粋関数)。

    システムプロンプトは「[関連する記憶]・[参考情報]・ツール実行結果が
    『有ったか / 無かったか』自体を話題にしない」と規定し、動的ブロックの区切り文でも
    同じことを言っているが、実機では両方とも破られる。

    実インシデント (2026-08-16 ライブ監査 ターン25): 「競合が20%値下げしてきた。
    追随すべきか」への応答が「自社の差別化要因（参考情報1に基づく）」と、
    ユーザーには見えない内部枠を名指しした。指示文での抑止は 2 層とも効かなかったので
    決定論で落とす。
    """
    out = _INTERNAL_FRAME_PAREN_RE.sub("", text)
    out = _INTERNAL_FRAME_CLAUSE_RE.sub("", out)
    return out


#: 枠名の全接頭辞。ストリーミング中に「まだ枠名になりうるか」を 1 文字単位で
#: 判定するために使う (「関数」の「関」で一度バッファしても、次の 1 文字で
#: 見切れるようにする)。
_FRAME_WORD_PREFIXES = frozenset(
    w[:i]
    for w in ("参考情報", "関連する記憶", "ツール実行結果", "参照情報")
    for i in range(1, len(w) + 1)
)


class InternalFrameMentionFilter:
    """ストリーミング中に内部の根拠枠への言及を落とす。

    表記は括弧や読点で閉じるまで判定できないため、開き括弧または枠名になりうる
    文字が見えた時点でバッファし、閉じるか「もう枠名にならない」と分かった時点で
    吐き出す。「関数」「参加」のような通常語は 1〜2 文字の遅延で見切れるので、
    体感できるストリーミングの引っかかりは生じない。

    StreamFilter プロトコル準拠: process() / flush()
    """

    #: 表記が閉じないままこの文字数を超えたら、通常の本文とみなして吐き出す。
    #: 「（参考情報1に基づく）」で 13 文字なので 48 あれば十分な余裕がある。
    _MAX_BUFFER_CHARS = 48

    _OPEN_PARENS = ("（", "(")

    def __init__(self) -> None:
        self._buffer = ""

    @classmethod
    def _could_still_match(cls, buf: str) -> bool:
        """バッファがまだ枠名への言及になりうるか。"""
        body = buf.lstrip("（(")
        if not body:
            return True
        # 枠名の途中 (「参考情」) か、枠名を言い切った後 (「参考情報1に基づ」)。
        for w in ("参考情報", "関連する記憶", "ツール実行結果", "参照情報"):
            if body.startswith(w):
                return True
        return body in _FRAME_WORD_PREFIXES

    def _trigger_index(self, text: str) -> int | None:
        for i, ch in enumerate(text):
            if ch in self._OPEN_PARENS or ch in _FRAME_WORD_PREFIXES:
                return i
        return None

    def process(self, text: str) -> str:
        if not text:
            return ""
        if not self._buffer:
            idx = self._trigger_index(text)
            if idx is None:
                return text
            head, self._buffer = text[:idx], text[idx:]
        else:
            head = ""
            self._buffer += text
        cleaned = strip_internal_frame_mentions(self._buffer)
        if cleaned != self._buffer:
            # 表記が閉じた。除去後を出して、続きは素通しに戻す。
            logger.info(
                "InternalFrameMentionFilter: removed a mention of the internal "
                "evidence frame from the visible answer",
            )
            self._buffer = ""
            return head + cleaned
        if (
            not self._could_still_match(self._buffer)
            or len(self._buffer) >= self._MAX_BUFFER_CHARS
        ):
            out, self._buffer = self._buffer, ""
            return head + out
        return head

    def flush(self) -> str:
        if not self._buffer:
            return ""
        out = strip_internal_frame_mentions(self._buffer)
        self._buffer = ""
        return out


class LengthDisclosureFilter:
    """明示された文字数指定を破ったことを、応答の末尾で開示する。

    ``violates_length_constraint`` は学習シグナル (``FeedbackCollector``) と
    few-shot の除外にしか使われておらず、**応答時は無検証**だった。指定は本文に
    あり長さは数えるだけなので、これは推定ではなく確定した矛盾である。黙って
    出すと「制約違反の隠蔽」になり、後続ターンの自己申告 (「いま書いた説明は
    何文字でしたか？」) とも食い違う。

    実測 (2026-08-23 ライブ監査 194 ターン): 違反 2 件 (20 文字ちょうどの指定に
    28 文字 / 30 文字ちょうどの指定に 38 文字)。再生成はストリーミングの
    作り直しが必要で、この発火率には見合わない。

    **フィルタとして実装する理由**: 判定を層 (deliberative / reactive / …) の
    終端処理に置くと、そこを通らない層で黙って外れる。実インシデント
    (2026-08-23 検証): 終端処理へ入れた直後の再測定で「20文字ちょうどで自己
    紹介してください。」が ``short_query`` → **reactive** に分類され、注記が
    一度も出なかった。パイプラインは全 LLM ストリーミング経路が組むので、
    ここに置けば層に依存しない。
    """

    def __init__(self, query: str = "") -> None:
        self._query = query or ""
        self._seen: list[str] = []

    def process(self, token: str) -> str:
        if token:
            self._seen.append(token)
        return token

    def flush(self) -> str:
        from backend.free.core.text_quality import violates_length_constraint

        if not self._query:
            return ""
        response = "".join(self._seen)
        self._seen.clear()
        if not response.strip():
            return ""
        reason = violates_length_constraint(self._query, response)
        if reason is None:
            return ""
        logger.info("Length constraint violated (%s)", reason)
        return f"\n\n(注: 指定された文字数に対し、上の回答は {len(response.strip())} 文字です)"
