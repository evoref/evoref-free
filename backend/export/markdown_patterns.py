"""エクスポート writer 共通の Markdown インライン解析。

全 writer (docx / pptx / odf / html / latex / plaintext) が ``iter_inline_segments`` で
本文を ``(断片, 種別)`` に分け、種別を各形式の run / span / 要素へ写す。強調は
CommonMark のデリミタ連の処理を最小限に実装した解析器で、入れ子 (``**太字*斜*体**`` /
``***bi***``) を ``bold_italic`` として返す (2026-10-02、f_11 §3.2)。

``backend/export/`` は Free/Pro 双方から参照可能な中立レイヤー。

強調の境界規則 (2026-09-16、docs/f_11_file_export.md §3.2):

``_`` / ``__`` は **語中では強調にしない** (CommonMark 準拠)。以前は
``_(.+?)_`` だったため ``snake_case_name`` の中間アンダースコアを区切りと
誤認し、区切り文字が落ちて ``snakecasename`` になっていた。Windows パス
(``E:\\tmp\\office_test\\with_image.docx``) も同じ経路で
``E:\\tmp\\officetest\\withimage.docx`` に壊れる。識別子とパスが黙って
書き換わるので実害が大きい。

``*`` は CommonMark では語中でも強調になるが、``=A1*B1*C1`` / ``2*3 と 4*5`` の
``*`` は演算子で、斜体にすると記号が落ちる。そこで区切りの外側が ASCII 英数字に
隣接する ``*`` は強調にしない (日本語に隣接する ``日本*語*`` は強調のまま)。
``2 * 3 * 4`` は CommonMark の flanking 規則で強調にならない。
"""

from __future__ import annotations

import re
import unicodedata

#: インラインコードの本体パターン。
_INLINE_CODE_SRC = r"`(.+?)`"

#: インラインコード `` `...` `` (group1 が本文)
RE_INLINE_CODE = re.compile(_INLINE_CODE_SRC)

#: リンク ``[text](url)`` (group1=text, group2=url)。
#: 本文は角括弧と改行を、URL は空白・``)``・角括弧を含まない (どちらも 1 文字以上)。
#: 以前の ``\[(.+?)\]\((.+?)\)`` は ``[a](`` の繰り返しでバックトラックが三次に
#: なり (2000 回で 54 秒)、``[]() 後 [x]()`` から壊れたリンクを作っていた。
#: 括弧を含む URL (``…/Foo_(bar)``) はリンクにならず文字のまま残る。
#: 直前が ``!`` のもの (画像の記法 ``![alt](src)``) はリンクにせず文字のまま残す。
RE_LINK = re.compile(r"(?<!!)\[([^\[\]\n]+)\]\(([^()\s\[\]]+)\)")

#: ``iter_inline_segments`` の種別のうち太字 / 斜体 / コードを含むもの (writer が
#: run の属性に写す)。種別は ``bold`` / ``italic`` / ``code`` をこの順に ``_`` で
#: つないだ名前 (``bold_italic`` / ``bold_code`` など)、どれも無ければ ``plain``。
_FLAG_NAMES = ("bold", "italic", "code")
_ALL_KINDS = [
    "_".join(n for n, on in zip(_FLAG_NAMES, (b, i, c)) if on) or "plain"
    for b in (0, 1) for i in (0, 1) for c in (0, 1)
]
BOLD_KINDS = frozenset(k for k in _ALL_KINDS if "bold" in k.split("_"))
ITALIC_KINDS = frozenset(k for k in _ALL_KINDS if "italic" in k.split("_"))
CODE_KINDS = frozenset(k for k in _ALL_KINDS if "code" in k.split("_"))

#: リンクにしてよい URL のスキーム。スキームの無い相対パスとアンカー (``#x``) も可。
_ALLOWED_SCHEMES = frozenset({"http", "https", "mailto"})
_RE_SCHEME = re.compile(r"([A-Za-z][A-Za-z0-9+.\-]*):")


def safe_link_url(url: str) -> str | None:
    """リンクにしてよい URL を返す。だめなら ``None`` (呼び出し側は原文のまま残す)。

    制御文字は取り除く (pptx は制御文字を含む URL で ``ValueError``)。
    ``javascript:`` / ``file:`` / ドライブレター (``C:``) などの許可外のスキームと、
    ``//host`` / ``\\\\server`` (UNC)、``{`` ``}`` ``\\`` を含むものはリンクにしない。
    """
    url = "".join(ch for ch in url if unicodedata.category(ch) != "Cc")
    # { } \ は LaTeX の \href{…} に命令を差し込める (``http://x}\input{…}``)。
    # UNC (``\\server``) とドライブレターのパスもここで落ちる。
    if not url or url.startswith("//") or any(ch in url for ch in "{}\\"):
        return None
    m = _RE_SCHEME.match(url)
    if m and m.group(1).lower() not in _ALLOWED_SCHEMES:
        return None
    return url


_RE_DELIM_RUN = re.compile(r"\*+|_+")

#: コードとリンクの位置を強調の解析から隠す伏せ字の候補 (私用領域)。入力に
#: 含まれない 2 文字を呼び出しごとに選ぶ (入力の私用領域の文字を消さない)。
#: 伏せ字は非 ASCII の文字として flanking を判定される (仮名・漢字と同じ扱い)。
_MARK_RANGE = range(0xE000, 0xF900)


def _pick_marks(text: str) -> tuple[str, str]:
    used = set(text)
    found = []
    for code in _MARK_RANGE:
        ch = chr(code)
        if ch not in used:
            found.append(ch)
            if len(found) == 2:
                return found[0], found[1]
    raise ValueError("no unused private-use character for inline masking")


def _is_punct(ch: str) -> bool:
    return unicodedata.category(ch)[0] in "PS"


def _flanking(prev: str, nxt: str) -> tuple[bool, bool]:
    """デリミタ連の ``(left-flanking, right-flanking)`` (CommonMark 6.2)。

    行頭 / 行末は空白とみなす。外側の判定では非 ASCII の文字 (仮名・漢字など) を
    空白と同じに扱う — 素の CommonMark だと ``**「強調」**です`` の閉じが
    right-flanking にならず強調が外れる (CJK 向けの緩和)。
    """
    prev_ws = not prev or prev.isspace()
    next_ws = not nxt or nxt.isspace()
    prev_punct = bool(prev) and _is_punct(prev)
    next_punct = bool(nxt) and _is_punct(nxt)
    prev_outer = prev_ws or prev_punct or not prev.isascii()
    next_outer = next_ws or next_punct or not nxt.isascii()
    left = not next_ws and (not next_punct or prev_outer)
    right = not prev_ws and (not prev_punct or next_outer)
    return left, right


def _can_open_close(char: str, prev: str, nxt: str) -> tuple[bool, bool]:
    """デリミタ連が開き / 閉じになれるか。

    ``*`` は外側が ASCII 英数字に接していたら開き / 閉じにならない (``=A1*B1*C1`` /
    ``2**3`` は演算子)。``_`` は外側が単語文字 (``\\w``) か ``\\`` に接していたら
    ならない (``snake_case`` / Windows パス)。
    """
    left, right = _flanking(prev, nxt)
    if char == "*":
        prev_blocked = prev.isascii() and prev.isalnum()
        next_blocked = nxt.isascii() and nxt.isalnum()
    else:
        prev_blocked = bool(prev) and (prev.isalnum() or prev in "_\\")
        next_blocked = bool(nxt) and (nxt.isalnum() or nxt in "_\\")
    return left and not prev_blocked, right and not next_blocked


class _Delim:
    """デリミタ連 (CommonMark の delimiter stack の 1 要素、双方向リスト)。"""

    __slots__ = ("idx", "char", "count", "orig", "can_open", "can_close", "opens", "closes",
                 "prev", "next")

    def __init__(
        self, idx: int, char: str, length: int, can_open: bool, can_close: bool,
    ) -> None:
        #: 行の中の通し番号。``openers_bottom`` は番号で覚える (ノードの参照だと、
        #: そのノードが後で外されたとき一致せず行頭まで遡って二次になる)。
        self.idx = idx
        self.char = char
        self.count = length
        self.orig = length
        self.can_open = can_open
        self.can_close = can_close
        self.opens: list[str] = []
        self.closes: list[str] = []
        self.prev: _Delim | None = None
        self.next: _Delim | None = None


def _unlink(node: _Delim) -> None:
    if node.prev is not None:
        node.prev.next = node.next
    if node.next is not None:
        node.next.prev = node.prev
    node.prev = node.next = None


def _parse_emphasis(line: str) -> list[tuple[str, int, int]]:
    """1 行 (コードとリンクは伏せ字) の強調を解析し ``(断片, 太字の深さ, 斜体の深さ)`` の列にする。

    CommonMark の process emphasis (付録 A) の最小実装。入れ子 (``**a *b* c**`` /
    ``***bi***``) と 3 の倍数の規則を扱う。デリミタは双方向リストに持ち、対応の
    取れなかった検索の下限を ``openers_bottom`` に覚えるので、行の長さにほぼ線形。
    対応の取れない記号は文字のまま残る。
    """
    tokens: list[str | _Delim] = []
    first: _Delim | None = None
    last_delim: _Delim | None = None
    last = 0
    for m in _RE_DELIM_RUN.finditer(line):
        if m.start() > last:
            tokens.append(line[last:m.start()])
        prev = line[m.start() - 1] if m.start() > 0 else ""
        nxt = line[m.end()] if m.end() < len(line) else ""
        run = m.group(0)
        idx = last_delim.idx + 1 if last_delim is not None else 0
        node = _Delim(idx, run[0], len(run), *_can_open_close(run[0], prev, nxt))
        tokens.append(node)
        if last_delim is None:
            first = node
        else:
            last_delim.next = node
            node.prev = last_delim
        last_delim = node
        last = m.end()
    if last < len(line):
        tokens.append(line[last:])

    # (閉じの文字, 閉じが開きにもなれるか, 長さ % 3) → この番号以下は探さない
    openers_bottom: dict[tuple[str, bool, int], int] = {}
    closer = first
    while closer is not None:
        if not closer.can_close:
            closer = closer.next
            continue
        key = (closer.char, closer.can_open, closer.orig % 3)
        bottom = openers_bottom.get(key, -1)
        opener = closer.prev
        while opener is not None and opener.idx > bottom:
            if opener.char == closer.char and opener.can_open:
                odd = (opener.can_close or closer.can_open) and (
                    (opener.orig + closer.orig) % 3 == 0
                    and not (opener.orig % 3 == 0 and closer.orig % 3 == 0)
                )
                if not odd:
                    break
            opener = opener.prev
        if opener is None or opener.idx <= bottom:
            openers_bottom[key] = closer.prev.idx if closer.prev is not None else -1
            nxt = closer.next
            if not closer.can_open:
                _unlink(closer)
            closer = nxt
            continue
        use = 2 if opener.count >= 2 and closer.count >= 2 else 1
        kind = "bold" if use == 2 else "italic"
        opener.count -= use
        closer.count -= use
        opener.opens.append(kind)
        closer.closes.append(kind)
        # 間に挟まったデリミタは対応を失い、文字のまま残る
        between = opener.next
        while between is not None and between is not closer:
            following = between.next
            _unlink(between)
            between = following
        if opener.count == 0:
            _unlink(opener)
        if closer.count == 0:
            nxt = closer.next
            _unlink(closer)
            closer = nxt

    out: list[tuple[str, int, int]] = []
    depth = {"bold": 0, "italic": 0}

    def emit(piece: str) -> None:
        if piece:
            out.append((piece, depth["bold"], depth["italic"]))

    for tok in tokens:
        if isinstance(tok, str):
            emit(tok)
            continue
        # 閉じは連の左 (内側) から、開きは連の右 (内側) から消費される。
        # 開きは後で対応したものほど外側なので逆順に積む。
        for kind in tok.closes:
            depth[kind] -= 1
        emit(tok.char * tok.count)
        for kind in reversed(tok.opens):
            depth[kind] += 1
    return out


_KIND_OF = {
    (b, i, c): "_".join(n for n, on in zip(_FLAG_NAMES, (b, i, c)) if on) or "plain"
    for b in (False, True) for i in (False, True) for c in (False, True)
}


def _kind(bold: int, italic: int, code: bool = False) -> str:
    return _KIND_OF[bool(bold), bool(italic), bool(code)]


def _inline_items(text: str) -> list[tuple[str, str, int, str | None]]:
    """``text`` を ``(断片, 種別, リンク番号, URL)`` の列にする (リンク外は番号 -1)。

    1. コード (`` `...` ``) を伏せ字にする — 中の ``*`` / ``_`` / ``[..](..)`` は解析しない。
    2. リンク ``[text](url)`` を伏せ字にする — URL は強調の解析にかけず、リンクは
       周りの強調から見て 1 文字 (``**詳細は [公式](u) 参照**`` の太字がリンクを
       またぐ)。URL が ``safe_link_url`` を通らなければリンクにせず原文のまま残す。
    3. 行ごとに強調を解析し、伏せ字を元に戻す。リンクの本文は別に強調を解析し、
       外側の強調と重ねる。コードは周りの強調を保つ (``bold_code`` など)。
    """
    code_mark, link_mark = _pick_marks(text)
    marks_re = re.compile(f"([{code_mark}{link_mark}])")
    codes: list[str] = []

    def stash_code(m: re.Match) -> str:
        codes.append(m.group(1))
        return code_mark

    masked = RE_INLINE_CODE.sub(stash_code, text)
    links: list[tuple[str, str]] = []

    def stash_link(m: re.Match) -> str:
        url = safe_link_url(m.group(2)) if code_mark not in m.group(2) else None
        if url is None:
            return m.group(0)
        links.append((m.group(1), url))
        return link_mark

    masked = RE_LINK.sub(stash_link, masked)
    code_iter = iter(codes)
    link_iter = iter(enumerate(links))
    items: list[tuple[str, str, int, str | None]] = []

    def expand(line: str, outer_bold: int, outer_italic: int, link_no: int, url) -> None:
        for piece, bold, italic in _parse_emphasis(line):
            bold, italic = bold or outer_bold, italic or outer_italic
            for part in marks_re.split(piece):
                if part == code_mark:
                    items.append((next(code_iter), _kind(bold, italic, True), link_no, url))
                elif part == link_mark:
                    no, (label, link_url) = next(link_iter)
                    expand(label, bold, italic, no, link_url)
                elif part:
                    items.append((part, _kind(bold, italic), link_no, url))

    for li, line in enumerate(masked.split("\n")):
        if li:
            items.append(("\n", "plain", -1, None))
        expand(line, 0, 0, -1, None)
    return items


def _merged(pairs) -> list[tuple[str, str]]:
    """同じ種別の隣り合う断片を 1 つにまとめる (連結は最後に 1 回)。"""
    out: list[list] = []
    for piece, kind in pairs:
        if out and out[-1][1] == kind:
            out[-1][0].append(piece)
        else:
            out.append([[piece], kind])
    return [("".join(parts), kind) for parts, kind in out]


def iter_inline_segments(text: str):
    """``text`` を ``(断片, 種別)`` の列に分ける (リンクは本文だけ残す)。

    種別は ``plain`` か、``bold`` / ``italic`` / ``code`` をこの順に ``_`` でつないだ名前
    (``bold_italic`` / ``bold_code`` など)。何を含むかは ``BOLD_KINDS`` / ``ITALIC_KINDS`` /
    ``CODE_KINDS`` で判定する。記号は本文から外れ、対応の取れない記号は文字のまま
    残る。リンクを描く writer は ``iter_inline_groups`` を使う。同じ種別の隣り合う
    断片は 1 つにまとめる。
    """
    yield from _merged((piece, kind) for piece, kind, _no, _url in _inline_items(text))


def iter_inline_groups(text: str):
    """``text`` を ``(URL, [(断片, 種別), ...])`` の列に分ける。リンクの外は URL が ``None``。

    リンク 1 つが 1 グループ (隣り合う別のリンクは別のグループ)。run / span を組む
    writer はグループごとにリンク要素を作り、中の断片を種別どおりに描く。
    """
    group: list[tuple[str, str]] = []
    current: tuple[int, str | None] | None = None
    for piece, kind, no, url in _inline_items(text):
        if current is not None and current[0] != no:
            yield current[1], _merged(group)
            group = []
        current = (no, url)
        group.append((piece, kind))
    if current is not None:
        yield current[1], _merged(group)


def strip_inline(text: str) -> str:
    """inline Markdown の記号を外したプレーンテキスト (題など、書式を持てない箇所用)。"""
    return "".join(piece for piece, _kind in iter_inline_segments(text))
