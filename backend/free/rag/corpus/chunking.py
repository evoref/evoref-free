"""パッケージ ``docs/`` → ``doc_chunk`` Evidence の決定論生成 (c_16 §3.5 / §4.3)

`docs/` が真実で、チャンクはその派生物。**同じ ``docs/`` からは何度やっても
同じ Evidence が出る**ようにしてあり、再インストール・別 PC での展開・
埋め込みモデル切替後の再構築で id が変わらない。

## evidence id はなぜ内容由来ハッシュか

c_05 §0.5.5 は「位置カウンタを鍵にしない」— ``len(...)`` 由来の連番は削除後に
再発行され、既存レコードを別内容で上書きする (VectorStore の chunk id で実際に
起きた、2026-09-05 監査)。ここで使うのは連番ではなく

``sha256(content_digest | chunker_version | doc_id | position)`` の先頭 12 hex

で、**同じ内容には同じ id、違う内容には違う id** が付く。``position`` は入って
いるが単独の鍵ではない — ``content_digest`` (docs 全体の本文 sha256) が変われば
全チャンクの id が変わるので、「削除後に同じ id が別内容で再発行される」形には
ならない。ランダム id にしないのは、prebuilt (作成側が事前に作ったチャンクと
埋め込み) を install 側が **id で突き合わせて流用する**ため — ランダムだと
毎回全件埋め込み直しになる。
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from backend.free.rag.chunker import SemanticChunker
from backend.free.rag.evidence.types import (
    Evidence,
    compute_claim_key,
    derive_confidence,
)
from backend.free.rag.text_extractor import (
    SUPPORTED_DOC_EXTENSIONS,
    extract_text,
    parse_csv_to_chunks,
)
from backend.log_config import get_logger
from backend.utils import utc_now

logger = get_logger("rag.corpus.chunking")

#: チャンク生成規則の版。**分割の結果が変わる変更で上げる** (chunk_size の
#: 既定変更 / 抽出器の差し替え / heading の付け方)。prebuilt はこの版が一致
#: したときだけ採用される (c_16 §4.3)。
CHUNKER_VERSION = 3

#: ``provenance[].extractor``。
EXTRACTOR_NAME = "corpus_chunker"

#: 文書由来 (``origin=document``) の裏取り件数。パッケージ 1 本しか出所が
#: 無いので常に 1 (c_16 §3.3 の corroboration_bonus = 0.8)。
_CORROBORATION = 1

#: markdown 見出し行 (原文の行構造に対して当てる)。
_HEADING_RE = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$", re.MULTILINE)

#: 見出しとして持ち回る最大長 (これを超える行は見出しではなく本文とみなす)。
_MAX_HEADING_CHARS = 120


def chunk_evidence_id(
    content_digest: str, chunker_version: int, doc_id: str, position: int,
) -> str:
    """``(content_digest, chunker_version, doc_id, position)`` から id を導く。

    再インストールで同じ id になることが prebuilt 流用の前提 (モジュール
    docstring 参照)。
    """
    payload = f"{content_digest}\x00{chunker_version}\x00{doc_id}\x00{position}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"ev_{digest[:12]}"


def document_headings(raw_text: str) -> list[str]:
    """原文から markdown 見出しを出現順に集める。

    **チャンク本文ではなく原文に当てる**のがポイント。
    :class:`SemanticChunker` は文単位に分割して空白で連結するため、チャンクの
    中では改行が消えており、``^#{1,6} …$`` を当てると見出し行と後続の本文が
    1 行に見えて「本文まるごとが見出し」になる (2026-09-07 の実測で確認)。
    """
    headings: list[str] = []
    for match in _HEADING_RE.finditer(raw_text):
        title = match.group(1).strip()
        if title and len(title) <= _MAX_HEADING_CHARS:
            headings.append(title)
    return headings


def _heading_of(text: str, headings: list[str], carried: str) -> str:
    """チャンクが属する節の見出しを決める。

    原文から拾った見出しのうち、そのチャンクに **最後に現れるもの** を採る。
    どれも現れなければ直前のチャンクの見出しを引き継ぐ — 見出しの下に
    ぶら下がる本文チャンクが「どの節の話か」を失わないようにするため。
    文書を頭から順に走るだけなので決定論。
    """
    found = carried
    for title in headings:
        if title in text:
            found = title
    return found


def _make_chunker(rag_config: Any) -> SemanticChunker:
    """``rag`` 設定 (dict / オブジェクト) から分割器を作る。"""

    def value(key: str, default: Any) -> Any:
        if rag_config is None:
            return default
        got = (
            rag_config.get(key) if isinstance(rag_config, dict)
            else getattr(rag_config, key, None)
        )
        return default if got is None else got

    return SemanticChunker(
        chunk_size=int(value("chunk_size", 512)),
        chunk_overlap=int(value("chunk_overlap", 128)),
        min_chunk=int(value("semantic_min_chunk", 64)),
        max_chunk=int(value("semantic_max_chunk", 512)),
        strategy=str(value("chunking_strategy", "semantic")),
    )


def list_documents(docs_dir: Path | str) -> list[tuple[str, Path]]:
    """``docs/`` の対象ファイルを ``(doc_id, パス)`` で列挙する (doc_id 昇順)。

    ``doc_id`` は ``docs/`` からの相対 posix パス。サブディレクトリを潰さない
    (潰すと別ディレクトリの同名ファイルが同じ doc_id になる)。
    """
    root = Path(docs_dir)
    if not root.is_dir():
        return []
    found: list[tuple[str, Path]] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_DOC_EXTENSIONS:
            continue
        found.append((path.relative_to(root).as_posix(), path))
    found.sort(key=lambda item: item[0])
    return found


def extract_chunks(
    doc_path: Path, chunker: SemanticChunker,
) -> tuple[list[str], str]:
    """1 ファイルから ``(チャンク本文, 原文)`` を作る。

    CSV は行ごとに 1 チャンク (ヘッダー付与)、それ以外は抽出 →
    :class:`SemanticChunker`。読めない / 空のファイルは ``([], "")``。

    原文も返すのは、見出しを **原文の行構造** から拾うため
    (:func:`document_headings` の docstring 参照)。
    """
    if doc_path.suffix.lower() == ".csv":
        return parse_csv_to_chunks(doc_path), ""
    text = extract_text(doc_path)
    if not text.strip():
        return [], ""
    return chunker.chunk(text), text


def chunk_documents(
    docs_dir: Path | str,
    *,
    package_id: str,
    package_version: str,
    content_digest: str,
    language: str = "",
    rag_config: Any = None,
    now: str | None = None,
    on_document: Callable[[int, int, str], None] | None = None,
) -> list[Evidence]:
    """``docs/`` 全体を ``doc_chunk`` Evidence の列にする (c_16 §3.5)。

    Args:
        docs_dir: パッケージの ``docs/``。
        package_id / package_version: ``attrs`` と ``source_id`` に刻む。
        content_digest: :func:`~backend.free.rag.corpus.package.compute_content_digest`
            の値。evidence id の素になるので **必ず実際の docs から計算した値**
            を渡す (持ち回った古い値だと別内容に同じ id が付く)。
        language: ``Evidence.lang`` に入れる (パッケージの宣言言語)。
        rag_config: ``chunk_size`` 等を読む ``rag`` セクション。
        now: 観測時刻。``None`` で現在時刻。1 回のインストールで全チャンクに
            同じ値を使う (行ごとに時刻を取ると鮮度が揃わない)。
        on_document: ``(index, total, doc_id)`` を受ける進捗コールバック。

    Returns:
        doc_id 昇順 → position 昇順の :class:`Evidence` 列。
    """
    stamp = now or utc_now()
    chunker = _make_chunker(rag_config)
    documents = list_documents(docs_dir)
    total = len(documents)
    confidence = derive_confidence("document", _CORROBORATION)

    out: list[Evidence] = []
    for index, (doc_id, doc_path) in enumerate(documents, start=1):
        if on_document is not None:
            on_document(index, total, doc_id)
        try:
            texts, raw_text = extract_chunks(doc_path, chunker)
        except (OSError, ValueError, ImportError) as e:
            # 1 ファイルの抽出失敗でパッケージ全体を落とさない (c_05 §0.5.2)。
            logger.warning("skipping document %s: %s", doc_id, e)
            continue

        headings = document_headings(raw_text)
        heading = ""
        for position, text in enumerate(texts):
            body = text.strip()
            if not body:
                continue
            heading = _heading_of(body, headings, heading)
            out.append(
                Evidence(
                    id=chunk_evidence_id(
                        content_digest, CHUNKER_VERSION, doc_id, position,
                    ),
                    kind="doc_chunk",
                    store="corpus",
                    text=body,
                    lang=language or None,
                    origin="document",
                    provenance=[
                        {
                            "source_id": f"doc:{package_id}/{doc_id}",
                            "extractor": EXTRACTOR_NAME,
                            "extractor_version": CHUNKER_VERSION,
                            "captured_at": stamp,
                        },
                    ],
                    observed_at=stamp,
                    confidence=confidence,
                    claim_key=compute_claim_key(body),
                    created_at=stamp,
                    attrs={
                        "package_id": package_id,
                        "package_version": package_version,
                        "doc_id": doc_id,
                        "position": position,
                        "heading": heading,
                    },
                ),
            )

    logger.info(
        "Chunked package %s v%s: %d doc(s) -> %d chunk(s) (chunker v%d)",
        package_id, package_version, total, len(out), CHUNKER_VERSION,
    )
    return out


__all__ = [
    "CHUNKER_VERSION",
    "EXTRACTOR_NAME",
    "chunk_documents",
    "chunk_evidence_id",
    "document_headings",
    "extract_chunks",
    "list_documents",
]
