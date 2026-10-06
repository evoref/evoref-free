"""HTML → 構造付きテキスト (見出し ``#`` / GFM 表 / 画像の alt / 文書題)

``fetch_url`` (``backend/free/agent/tools/html_text.py``) と corpus の HTML 抽出器
(``extractors/html.py``) が共有する唯一の実装 (不変則 #14(a))。ナビゲーションや
リンク密度の高いブロックを落とす強い除去は ``fetch_url`` だけ
(``html_to_text(strip_boilerplate=True)``)。利用者自身のファイルは script / style /
noscript だけを除き、構造化だけを掛ける。

- 見出し ``h1``〜``h6`` は ``#`` 行にする (分割器が見出しで閉じられる、f_01 §3.1.3)。
- 表は GFM にし、``<caption>`` を表の直前の行に置く (f_01 §3.1)。セルにブロック要素を
  含む表・1 セルだけの表はレイアウト用とみなし、GFM にしない。
- 画像の ``alt`` は ``[image] <alt>`` の行にする。``figcaption`` は本文のまま残る。
- 文書題 (``og:title`` → ``<title>`` のサイト名を除いた部分) は、本文に h1 が無く本文の
  見出しと重ならなければ先頭の ``#`` 行にする。
"""

from __future__ import annotations

import re

from backend.free.extraction.extractors._document_parts import (
    GFM_TABLE_SEP_LINE_RE,
    IMAGE_MARK,
    gfm_table_lines,
)
from backend.log_config import get_logger

logger = get_logger("extraction.html_structure")


# 除去する HTML タグ（ノイズ源）。
# 注: 追加分は void 要素 (input/source/area 等) を避ける。stdlib フォールバック
# (strip_html_fallback) は終了タグで skip 深度を戻すため、void 要素を入れると
# 深度が戻らず以降が全て欠落する。bs4 経路 (本番) は void を正しく扱う。
# ``picture`` は入れない — 中身は source / img (文字を持たない) で、外すと img の alt まで消える。
_HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")

#: 利用者自身のファイル (corpus の HTML 抽出) で除去するタグ。文字が本文でないものだけ。
#: ナビ・ヘッダ・フッタ・class 名による除去は ``fetch_url`` だけ (``strip_boilerplate``)
#: — 自分のファイルでは ``<header>`` の中の h1 や ``class="share-price"`` の節も本文。
SAFE_STRIP_TAGS = ["script", "style", "noscript"]

STRIP_TAGS = [
    "script", "style", "nav", "footer", "header", "aside",
    "noscript", "iframe", "form", "svg", "meta", "link",
    "button", "select", "textarea", "label", "template", "dialog",
    "video", "audio", "canvas",
]


def strip_html_fallback(html: str) -> str:
    """BeautifulSoup なしで HTML タグを除去するフォールバック

    stdlib の html.parser を使い、タグ構造を正確にパースする。
    """
    from html.parser import HTMLParser

    class _TextExtractor(HTMLParser):
        def __init__(self):
            super().__init__()
            self._result: list[str] = []
            self._skip_depth = 0  # スキップ中のタグのネスト深度

        def handle_starttag(self, tag: str, attrs):  # noqa: ARG002
            if tag.lower() in STRIP_TAGS:
                self._skip_depth += 1

        def handle_endtag(self, tag: str):
            if tag.lower() in STRIP_TAGS and self._skip_depth > 0:
                self._skip_depth -= 1

        def handle_data(self, data: str):
            if self._skip_depth == 0:
                stripped = data.strip()
                if stripped:
                    self._result.append(stripped)

        def get_text(self) -> str:
            return "\n".join(self._result)

    extractor = _TextExtractor()
    try:
        extractor.feed(html)
    except Exception:
        # パース失敗時は最低限の正規表現フォールバック
        text = re.sub(r"<[^>]+>", "", html)
        text = re.sub(r"\n\s*\n", "\n", text)
        return text.strip()
    return extractor.get_text()


def contains_markdown_table(text: str) -> bool:
    """テキストに GFM テーブル (区切り行付き) が含まれるか判定する。"""
    return bool(GFM_TABLE_SEP_LINE_RE.search(text))


# ── 本文抽出ヒューリスティック ──────────────────────────────
# class/id/role/aria-label がボイラープレート (nav/menu/footer/breadcrumb 等) を
# 示す要素を除去するための境界アンカー正規表現。短語 (ad/ads) の誤爆 (address 等)
# を避けるため前後を区切り文字/端でアンカーする。
BOILERPLATE_ATTR_RE = re.compile(
    r"(?:^|[\s_-])(?:"
    r"nav|navbar|navigation|globalnav|gnav|subnav|menu|"
    r"footer|contentinfo|header|masthead|breadcrumb|breadcrumbs|"
    r"sidebar|widget|banner|advert|advertisement|ads?|adsbygoogle|"
    r"promo|cookie|consent|gdpr|social|share|sns|related|recommend|"
    r"pager|pagination|toc|skiplink|utility|copyright|legal|disclaimer"
    r")(?:$|[\s_-])",
    re.IGNORECASE,
)
# リンク密度判定の対象ブロックタグと閾値 (ナビ/メニュー/リンク一覧の駆除)。
LINK_DENSE_BLOCK_TAGS = ("ul", "ol", "div", "section")
LINK_DENSITY_THRESHOLD = 0.6
LINK_DENSITY_MIN_LINKS = 4
LINK_DENSITY_MIN_TEXT = 40
# 本文コンテナ候補 (存在すればここへスコープを絞る)。
MAIN_CONTENT_SELECTORS = ("main", "article", "[role=main]", "#main", "#content", "#main-content")
# ヒューリスティックが naive のこの比率未満しか残さない場合は過剰除去とみなし
# naive へ退避する (本文があるのに空を返さないためのセーフティネット)。
EXTRACTION_MIN_RETAIN_RATIO = 0.10



def _cell_text(el) -> str:
    """要素の文字を 1 行にする (セル・見出し・caption 用。改行は表や見出しを壊す)。"""
    return " ".join(el.get_text(" ", strip=True).split())


#: セルに入っていたら表を「レイアウト用」とみなす要素。GFM の 1 セルに入れ子の表・
#: 見出し・節を畳むと、ページ全体が 1 セルの 1 行に潰れる。セル内のリスト・``div``・
#: 複数の段落 (Wikipedia のセル内リスト、Confluence / Word の書き出し) はデータの表に
#: よくあるので数えず、セルの文字に空白区切りで畳む (``_cell_text``)。
_LAYOUT_BLOCK_TAGS = ("table", "section", "article", *_HEADING_TAGS)


def _is_layout_table(table) -> bool:
    """レイアウト用の表か (セルに入れ子の表・見出し・節を含む / 1 セルだけの表)。

    レイアウト用の表は GFM にせず、中身を本文としてそのまま構造化する。
    """
    cells = table.find_all(["th", "td"])
    if len(cells) <= 1:
        return True
    return any(cell.find(_LAYOUT_BLOCK_TAGS) is not None for cell in cells)


def flatten_tables(root) -> None:
    """<table> を GitHub-flavored Markdown 表へ置換する。

    ヘッダ行 + 区切り行 + 各データ行を前後パイプ付きで出力する (``gfm_table_lines``)。
    これにより fetch_url 結果中の表を ``ContentConverter.from_markdown`` が table
    ブロックとして解釈でき、取得 → xlsx 出力の経路が成立する。``<caption>`` は
    表の直前の行に置く (空行を挟まない — 分割器が表の題として読む、f_01 §3.1)。
    """
    for table in root.find_all("table"):
        try:
            if table.parent is None or _is_layout_table(table):
                continue
            caption_el = table.find("caption")
            caption = _cell_text(caption_el) if caption_el is not None else ""
            rows = [
                [_cell_text(c) for c in tr.find_all(["th", "td"])]
                for tr in table.find_all("tr")
            ]
            md_lines = gfm_table_lines(rows)
            if not md_lines:
                continue
            if caption:
                md_lines.insert(0, caption)
            table.replace_with("\n\n" + "\n".join(md_lines) + "\n\n")
        except Exception:
            continue


def mark_headings(root) -> None:
    """``h1``〜``h6`` を ``#`` 行に置き換える。文字の無い見出しは捨てる。"""
    for tag in root.find_all(_HEADING_TAGS):
        try:
            if tag.parent is None:
                continue
            text = _cell_text(tag)
            if not text:
                tag.decompose()
                continue
            level = int(tag.name[1])
            tag.replace_with(f"\n\n{'#' * level} {text}\n\n")
        except Exception:
            continue


def inline_image_alts(root) -> None:
    """``<img alt>`` を ``[image] <alt>`` の行にする。alt の無い画像 (装飾) は捨てる。"""
    for img in root.find_all("img"):
        try:
            if img.parent is None:
                continue
            alt = " ".join(str(img.get("alt") or "").split())
            if alt:
                img.replace_with(f"\n{IMAGE_MARK} {alt}\n")
            else:
                img.decompose()
        except Exception:
            continue


#: ``<title>`` の「題 | サイト名」の区切り (空白で囲んだ ``|`` と全角 ``｜``)。``-`` / ``–``
#: は題の中にもよく出る (「Python 3.12 - What's New」) ので区切りにしない。
_TITLE_SITE_SEP_RE = re.compile(r"\s+\|\s+|\s*｜\s*")


def document_title(soup) -> str:
    """文書題。``og:title`` を優先し、無ければ ``<title>`` の末尾の区切り 1 つより前 (サイト名を落とす)。"""
    og = soup.find("meta", attrs={"property": "og:title"})
    if og is not None:
        content = " ".join(str(og.get("content") or "").split())
        if content:
            return content
    title = soup.find("title")
    if title is None:
        return ""
    text = _cell_text(title)
    separators = list(_TITLE_SITE_SEP_RE.finditer(text))
    head = text[:separators[-1].start()].strip() if separators else text
    return head or text


def _structure(root) -> None:
    """表 → 見出し → 画像の順に構造を文字へ写す (表の中の見出しはセルの文字になる)。"""
    flatten_tables(root)
    mark_headings(root)
    inline_image_alts(root)


def collapse_blank_lines(text: str) -> str:
    """各行の前後空白を落とし、連続する空行を 1 行に畳む。"""
    out: list[str] = []
    prev_empty = True
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            if not prev_empty:
                out.append("")
            prev_empty = True
            continue
        out.append(line)
        prev_empty = False
    return "\n".join(out).strip()


def _render(root) -> str:
    return collapse_blank_lines(root.get_text(separator="\n"))


def extract_naive(html: str) -> str:
    """素朴抽出 (タグ名 strip → 構造化 → テキスト)。比較・退避用。"""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(STRIP_TAGS):
            tag.decompose()
        root = soup.body or soup
        _structure(root)
        return _render(root)
    except ImportError:
        logger.warning("bs4 not available, falling back to stdlib HTML parser")
        return strip_html_fallback(html)


def has_boilerplate_attr(tag) -> bool:
    """tag の class/id/role/aria-label がボイラープレートを示すか。"""
    name = getattr(tag, "name", None)
    if name in (None, "html", "body", "[document]"):
        return False
    parts: list[str] = []
    cls = tag.get("class")
    if cls:
        parts.append(" ".join(cls) if isinstance(cls, list) else str(cls))
    for attr in ("id", "role", "aria-label"):
        val = tag.get(attr)
        if val:
            parts.append(str(val))
    if not parts:
        return False
    return bool(BOILERPLATE_ATTR_RE.search(" ".join(parts)))


def select_main_root(soup):
    """本文コンテナ (main/article 等) があればそれを、無ければ body を返す。"""
    for sel in MAIN_CONTENT_SELECTORS:
        try:
            el = soup.select_one(sel)
        except Exception:
            el = None
        if el is not None and len(el.get_text(strip=True)) >= 200:
            return el
    return soup.body or soup


def prune_link_dense_blocks(root) -> None:
    """リンク密度の高いブロック (ナビ/メニュー/リンク一覧) を除去する。"""
    for el in root.find_all(LINK_DENSE_BLOCK_TAGS):
        try:
            if el.parent is None:  # 祖先 decompose 済みで既に分離
                continue
            text = el.get_text(strip=True)
            if len(text) < LINK_DENSITY_MIN_TEXT:
                continue
            links = el.find_all("a")
            if len(links) < LINK_DENSITY_MIN_LINKS:
                continue
            link_len = sum(len(a.get_text(strip=True)) for a in links)
            if link_len / max(len(text), 1) >= LINK_DENSITY_THRESHOLD:
                el.decompose()
        except Exception:
            continue


def extract_main_content(html: str) -> str:
    """ヒューリスティックで本文を抽出する (bs4 必須・各段は例外を投げない設計)。"""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(STRIP_TAGS):
        tag.decompose()
    for el in soup.find_all(has_boilerplate_attr):
        try:
            el.decompose()
        except Exception:
            continue
    root = select_main_root(soup)
    prune_link_dense_blocks(root)
    _structure(root)
    return _render(root)


def _with_title(html: str, body: str) -> str:
    """文書題を先頭の ``#`` 行にする。

    本文に h1 があるか、本文の見出しと同じ文字なら足さない (「題 | サイト名」と h1 が
    二重になる)。見出しは HTML の ``h1``〜``h6`` から写した行だけを数える — ``<pre>`` の
    中の ``# コメント`` は見出しではない。
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return body
    try:
        soup = BeautifulSoup(html, "html.parser")
        title = document_title(soup)
        real = {(int(h.name[1]), _cell_text(h)) for h in soup.find_all(_HEADING_TAGS)}
    except Exception:
        return body
    if not title:
        return body
    lines = set(body.splitlines())
    shown = {(level, text) for level, text in real if f"{'#' * level} {text}" in lines}
    if any(level == 1 for level, _ in shown) or title in {text for _, text in shown}:
        return body
    return f"# {title}\n\n{body}" if body else f"# {title}"


def _extract_structured(html: str) -> str:
    """保守的な抽出: ``SAFE_STRIP_TAGS`` だけ除き、全体を構造化する (bs4 必須)。"""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(SAFE_STRIP_TAGS):
        tag.decompose()
    root = soup.body or soup
    _structure(root)
    return _render(root)


def html_to_text(html: str, *, strip_boilerplate: bool = False) -> str:
    """HTML を本文テキストへ変換する。

    既定 (``strip_boilerplate=False``、利用者自身のファイル) は script / style /
    noscript だけを除いて全体を構造化する。``strip_boilerplate=True`` (``fetch_url``)
    はヒューリスティック抽出 (extract_main_content) を試み、bs4 不在・例外・空・
    過剰除去 (naive 比 EXTRACTION_MIN_RETAIN_RATIO 未満) の場合は naive 抽出へ
    退避する。本文があるのに空を返さないことを保証する。
    """
    if not strip_boilerplate:
        try:
            import bs4  # noqa: F401
        except ImportError:
            return strip_html_fallback(html)
        return _with_title(html, _extract_structured(html))
    naive = extract_naive(html)
    try:
        import bs4  # noqa: F401
    except ImportError:
        return naive
    try:
        improved = extract_main_content(html)
    except Exception as e:
        logger.warning("HTML heuristic extraction failed (%r); using naive", e)
        return _with_title(html, naive)
    if not improved.strip():
        return _with_title(html, naive)
    if naive and len(improved) < len(naive) * EXTRACTION_MIN_RETAIN_RATIO:
        logger.info(
            "HTML heuristic extraction too aggressive (%d << %d chars); using naive",
            len(improved), len(naive),
        )
        return _with_title(html, naive)
    return _with_title(html, improved)
