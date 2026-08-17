"""HTML → プレーンテキスト抽出

``fetch_url`` が取得した HTML から本文だけを取り出す層。ナビゲーションや
リンク密度の高いブロックを落とし、表は Markdown へ畳む。取得そのものは
``web_fetch`` の責務。
"""

from __future__ import annotations

import re

from backend.log_config import get_logger

logger = get_logger("agent.tools.builtin")


# fetch_url で除去する HTML タグ（ノイズ源）。
# 注: 追加分は void 要素 (input/source/area 等) を避ける。stdlib フォールバック
# (_strip_html_fallback) は終了タグで skip 深度を戻すため、void 要素を入れると
# 深度が戻らず以降が全て欠落する。bs4 経路 (本番) は void を正しく扱う。
_STRIP_TAGS = [
    "script", "style", "nav", "footer", "header", "aside",
    "noscript", "iframe", "form", "svg", "meta", "link",
    "button", "select", "textarea", "label", "template", "dialog",
    "picture", "video", "audio", "canvas",
]


def _strip_html_fallback(html: str) -> str:
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
            if tag.lower() in _STRIP_TAGS:
                self._skip_depth += 1

        def handle_endtag(self, tag: str):
            if tag.lower() in _STRIP_TAGS and self._skip_depth > 0:
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
        import re as _re
        text = _re.sub(r"<[^>]+>", "", html)
        text = _re.sub(r"\n\s*\n", "\n", text)
        return text.strip()
    return extractor.get_text()
_MD_TABLE_SEP_LINE_RE = re.compile(r"^\s*\|[\s:\-|]+\|\s*$", re.MULTILINE)


def _contains_markdown_table(text: str) -> bool:
    """テキストに GFM テーブル (区切り行付き) が含まれるか判定する。"""
    return bool(_MD_TABLE_SEP_LINE_RE.search(text))


# ── fetch_url 本文抽出ヒューリスティック ──────────────────────────────
# class/id/role/aria-label がボイラープレート (nav/menu/footer/breadcrumb 等) を
# 示す要素を除去するための境界アンカー正規表現。短語 (ad/ads) の誤爆 (address 等)
# を避けるため前後を区切り文字/端でアンカーする。
_BOILERPLATE_ATTR_RE = re.compile(
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
_LINK_DENSE_BLOCK_TAGS = ("ul", "ol", "div", "section")
_LINK_DENSITY_THRESHOLD = 0.6
_LINK_DENSITY_MIN_LINKS = 4
_LINK_DENSITY_MIN_TEXT = 40
# 本文コンテナ候補 (存在すればここへスコープを絞る)。
_MAIN_CONTENT_SELECTORS = ("main", "article", "[role=main]", "#main", "#content", "#main-content")
# ヒューリスティックが naive のこの比率未満しか残さない場合は過剰除去とみなし
# naive へ退避する (本文があるのに空を返さないためのセーフティネット)。
_EXTRACTION_MIN_RETAIN_RATIO = 0.10


def _extract_naive(html: str) -> str:
    """現行どおりの素朴抽出 (タグ名 strip → get_text)。比較・退避用。"""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(_STRIP_TAGS):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)
    except ImportError:
        logger.warning("bs4 not available, falling back to stdlib HTML parser")
        return _strip_html_fallback(html)


def _has_boilerplate_attr(tag) -> bool:
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
    return bool(_BOILERPLATE_ATTR_RE.search(" ".join(parts)))


def _select_main_root(soup):
    """本文コンテナ (main/article 等) があればそれを、無ければ body を返す。"""
    for sel in _MAIN_CONTENT_SELECTORS:
        try:
            el = soup.select_one(sel)
        except Exception:
            el = None
        if el is not None and len(el.get_text(strip=True)) >= 200:
            return el
    return soup.body or soup


def _prune_link_dense_blocks(root) -> None:
    """リンク密度の高いブロック (ナビ/メニュー/リンク一覧) を除去する。"""
    for el in root.find_all(_LINK_DENSE_BLOCK_TAGS):
        try:
            if el.parent is None:  # 祖先 decompose 済みで既に分離
                continue
            text = el.get_text(strip=True)
            if len(text) < _LINK_DENSITY_MIN_TEXT:
                continue
            links = el.find_all("a")
            if len(links) < _LINK_DENSITY_MIN_LINKS:
                continue
            link_len = sum(len(a.get_text(strip=True)) for a in links)
            if link_len / max(len(text), 1) >= _LINK_DENSITY_THRESHOLD:
                el.decompose()
        except Exception:
            continue


def _flatten_tables(root) -> None:
    """<table> を GitHub-flavored Markdown 表へ置換する。

    ヘッダ行 + 区切り行 (``| --- |``) + 各データ行を前後パイプ付きで出力する。
    これにより fetch_url 結果中の表を ``ContentConverter.from_markdown`` が table
    ブロックとして解釈でき、取得 → xlsx 出力の経路が成立する。列数が不揃いな行は
    最大列数にパディングし、セル内の ``|`` はエスケープする。
    """
    for table in root.find_all("table"):
        try:
            if table.parent is None:
                continue
            rows: list[list[str]] = []
            for tr in table.find_all("tr"):
                cells = [
                    c.get_text(strip=True).replace("|", "\\|")
                    for c in tr.find_all(["th", "td"])
                ]
                if any(cells):
                    rows.append(cells)
            if not rows:
                continue
            ncol = max(len(r) for r in rows)
            md_lines: list[str] = []
            for idx, r in enumerate(rows):
                padded = r + [""] * (ncol - len(r))
                md_lines.append("| " + " | ".join(padded) + " |")
                if idx == 0:
                    md_lines.append("| " + " | ".join(["---"] * ncol) + " |")
            table.replace_with("\n" + "\n".join(md_lines) + "\n")
        except Exception:
            continue


def _collapse_blank_lines(text: str) -> str:
    """3 連以上の改行を 2 連に畳む。"""
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _extract_main_content(html: str) -> str:
    """ヒューリスティックで本文を抽出する (bs4 必須・各段は例外を投げない設計)。"""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(_STRIP_TAGS):
        tag.decompose()
    for el in soup.find_all(_has_boilerplate_attr):
        try:
            el.decompose()
        except Exception:
            continue
    root = _select_main_root(soup)
    _prune_link_dense_blocks(root)
    _flatten_tables(root)
    text = root.get_text(separator="\n", strip=True)
    return _collapse_blank_lines(text)


def _html_to_text(html: str) -> str:
    """HTML を本文テキストへ変換する。

    ヒューリスティック抽出 (_extract_main_content) を試み、bs4 不在・例外・空・
    過剰除去 (naive 比 _EXTRACTION_MIN_RETAIN_RATIO 未満) の場合は naive 抽出へ
    退避する。本文があるのに空を返さないことを保証する。
    """
    naive = _extract_naive(html)
    try:
        import bs4  # noqa: F401
    except ImportError:
        return naive
    try:
        improved = _extract_main_content(html)
    except Exception as e:
        logger.warning("fetch_url heuristic extraction failed (%r); using naive", e)
        return naive
    if not improved.strip():
        return naive
    if naive and len(improved) < len(naive) * _EXTRACTION_MIN_RETAIN_RATIO:
        logger.info(
            "fetch_url heuristic extraction too aggressive (%d << %d chars); using naive",
            len(improved), len(naive),
        )
        return naive
    return improved
