"""パッケージ ``docs/`` → ``doc_chunk`` Evidence の決定論生成 (c_16 §3.5 / §4.3)

`docs/` が真実で、チャンクはその派生物。**同じ ``docs/`` からは何度やっても
同じ Evidence が出る**ようにしてあり、再インストール・別 PC での展開・
埋め込みモデル切替後の再構築で id が変わらない。

## evidence id はなぜ内容由来ハッシュか

c_05 §0.5.5 は「位置カウンタを鍵にしない」— ``len(...)`` 由来の連番は削除後に
再発行され、既存レコードを別内容で上書きする (VectorStore の chunk id で実際に
起きた、2026-09-05 監査)。ここで使うのは連番ではなく

``"ev_" + sha256("doc_chunk", CHUNKER_VERSION, doc_id, sha256(正規化した本文),
同一本文の出現番号)`` の先頭 16 hex (:func:`chunk_evidence_id`)

で、**同じ本文には同じ id、違う本文には違う id** が付く。位置もパッケージ全体の
ダイジェストも入れないので、**版を跨いで本文が同じチャンクは同じ id** を保つ
(疑似クエリの ``target_id`` や学習の参照が版上げで切れない)。同じ文書に同じ本文が
2 回出るときだけ出現番号で分ける。パッケージ内で id が衝突したら組み立てを拒否する
(:class:`ChunkIdCollisionError`)。ランダム id にしないのは、prebuilt (作成側が事前に
作ったチャンクと埋め込み) を install 側が **id で突き合わせて流用する**ため —
ランダムだと毎回全件埋め込み直しになる。
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
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
from backend.io.id_registry import derived_id
from backend.log_config import get_logger
from backend.utils import utc_now

logger = get_logger("rag.corpus.chunking")

#: チャンク生成規則の版。**分割の結果が変わる変更で上げる** (chunk_size の
#: 既定変更 / 抽出器の差し替え / heading の付け方)。prebuilt はこの版が一致
#: したときだけ採用される (c_16 §4.3)。
CHUNKER_VERSION = 4

#: ``provenance[].extractor``。
EXTRACTOR_NAME = "corpus_chunker"

#: 文書由来 (``origin=document``) の裏取り件数。パッケージ 1 本しか出所が
#: 無いので常に 1 (c_16 §3.3 の corroboration_bonus = 0.8)。
_CORROBORATION = 1

#: markdown 見出し行 (原文の行構造に対して当てる)。
_HEADING_RE = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$", re.MULTILINE)

#: 見出しとして持ち回る最大長 (これを超える行は見出しではなく本文とみなす)。
_MAX_HEADING_CHARS = 120


class ChunkIdCollisionError(ValueError):
    """パッケージ内で 2 つのチャンクが同じ id になった (組み立てを拒否する)。"""


def normalize_chunk_text(text: str) -> str:
    """id の素にする本文の正規化: NFC + 改行を LF に + 前後の空白を除く。"""
    unified = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    return unified.strip()


def chunk_evidence_id(
    chunker_version: int, doc_id: str, text: str, occurrence: int = 0,
) -> str:
    """``("doc_chunk", chunker_version, doc_id, sha256(本文), 出現番号)`` から id を導く。

    ``occurrence`` は同じ文書の中で同じ (正規化後の) 本文が何回目に現れたか (0 始まり)。
    再インストール・版上げで本文が同じなら同じ id になる (モジュール docstring、
    c_05 §0.5.5)。golden vector は ``test_chunking.py`` で固定する。
    """
    text_digest = hashlib.sha256(normalize_chunk_text(text).encode("utf-8")).hexdigest()
    payload = "\x00".join(
        ("doc_chunk", str(int(chunker_version)), doc_id, text_digest, str(int(occurrence))),
    )
    return derived_id("ev_", hashlib.sha256(payload.encode("utf-8")).hexdigest())


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
    language: str = "",
    rag_config: Any = None,
    now: str | None = None,
    on_document: Callable[[int, int, str], None] | None = None,
) -> list[Evidence]:
    """``docs/`` 全体を ``doc_chunk`` Evidence の列にする (c_16 §3.5)。

    Args:
        docs_dir: パッケージの ``docs/``。
        package_id / package_version: ``attrs`` と ``source_id`` に刻む。
        language: ``Evidence.lang`` に入れる (パッケージの宣言言語)。
        rag_config: ``chunk_size`` 等を読む ``rag`` セクション。
        now: 観測時刻。``None`` で現在時刻。1 回のインストールで全チャンクに
            同じ値を使う (行ごとに時刻を取ると鮮度が揃わない)。
        on_document: ``(index, total, doc_id)`` を受ける進捗コールバック。

    Returns:
        doc_id 昇順 → position 昇順の :class:`Evidence` 列。

    Raises:
        ChunkIdCollisionError: パッケージ内で 2 つのチャンクの id が衝突した。
    """
    stamp = now or utc_now()
    chunker = _make_chunker(rag_config)
    documents = list_documents(docs_dir)
    total = len(documents)
    confidence = derive_confidence("document", _CORROBORATION)

    out: list[Evidence] = []
    seen_ids: dict[str, tuple[str, int]] = {}
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
        occurrences: dict[str, int] = {}
        for position, text in enumerate(texts):
            body = text.strip()
            if not body:
                continue
            heading = _heading_of(body, headings, heading)
            normalized = normalize_chunk_text(body)
            occurrence = occurrences.get(normalized, 0)
            occurrences[normalized] = occurrence + 1
            record_id = chunk_evidence_id(CHUNKER_VERSION, doc_id, body, occurrence)
            clash = seen_ids.get(record_id)
            if clash is not None:
                raise ChunkIdCollisionError(
                    f"chunk id {record_id} collides in package {package_id}: "
                    f"{clash[0]}#{clash[1]} and {doc_id}#{position}",
                )
            seen_ids[record_id] = (doc_id, position)
            out.append(
                Evidence(
                    id=record_id,
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
    "ChunkIdCollisionError",
    "chunk_documents",
    "chunk_evidence_id",
    "document_headings",
    "extract_chunks",
    "list_documents",
    "normalize_chunk_text",
]
