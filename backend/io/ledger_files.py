"""形式台帳の ``path_key`` と実ファイルの対応 (c_05 §0.7)。

``store/`` の下のファイルがどの形式 (:class:`FormatSpec`) に属するかを決める唯一の
実装。reset (``keep_on_reset``)・export (``export``)・doctor・台帳の完全性テスト
(「台帳外のファイルを書かない」、c_05 §0.7.2) が使う。

``path_key`` の文法 (世代フォルダ ``<data_root>/g<N>/`` からの相対 posix パス、c_05 §0.2):

- ``<name>`` — 置換子。1 つのパス要素の中で 1 文字以上に当たる (``/`` は跨がない)。
  要素の一部でもよい (``aux_<task>.md`` / ``pm-<hash>``)。
- ``**`` — 要素全体に書き、0 個以上の要素に当たる (``path_key`` に高々 1 つ)。
  末尾の ``/**`` は「このディレクトリの下の全ファイル」(1 要素以上) で、中身を
  個別に宣言しない木 (corpus パッケージの版・埋め込みの版・作業ディレクトリ) に使う。
  木は ``encodings=("dir",)`` と組にする (:class:`FormatSpec` が検査する)。
- 末尾要素の拡張子が形式の ``encodings`` のどれかなら、拡張子は ``encodings`` の
  どれでもよい (``history.summary_embeddings`` の ``<model>.npy`` は
  ``<model>.ids.json`` にも当たる)。それ以外の拡張子 (``.gguf`` 等) は字句どおり。

複数の形式に当たるときは **最も具体的な宣言** を採る。左の要素から比べ、最初に
違う要素で「字句だけ > 字句と置換子 > 置換子だけ > ``**``」の順に強い方、
要素が尽きた方は弱い。並んだら字句の文字数が多い方、それでも並べば
``format_id`` の辞書順で先の方。これで木 (``.../<version>/**``) の中の個別の宣言
(``.../<version>/docs/**`` や ``.../package.json``) が木に勝つ。

退避名 (c_05 §0.5.8 の ``<name>.<kind>-<utcstamp>``、G0 の接頭辞形 ``.trash-<name>``、
``AtomicWriter`` の取り残し ``<name>.<rand>.tmp``) を含むパスは、他の宣言に
当たっても :data:`RESIDUE_FORMAT` (system) とする — 退避した版の ``records.jsonl``
を SoT として export しないため。

Free (Pro の形式が台帳に無い) は ``store/pro/`` の中を見ない (c_05 §0.8)。
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable, Iterator
from functools import lru_cache
from pathlib import Path

from backend.io.format_registry import (
    FORMATS,
    GLOBSTAR,
    PLACEHOLDER_RE,
    PRO_STORE_PREFIX,
    FormatRegistry,
    FormatSpec,
    register_format,
)

#: 台帳の範囲 (世代フォルダ直下の ``store/``)。
STORE_DIR = "store"

#: 退避名・取り残しの一時ファイル (パス要素 1 つに当てる)。
_RESIDUE_SEGMENT_RE = re.compile(
    r"\.trash-.+"  # 接頭辞形 (corpus の版 GC・create の run GC・ProjectMap)
    r"|.+\.(?:corrupt|trash|reset|pre-\d+)-\d{8}T\d{6}Z(?:-\d+)?"  # quarantine()
    r"|.+\.tmp",  # AtomicWriter の取り残し (起動ゲートが掃く)
)

RESIDUE_FORMAT = register_format(FormatSpec(
    format_id="store.residue",
    version=1,
    klass="system",
    writers=frozenset({"free", "pro"}),
    path_key="store/**/<name>.<kind>-<stamp>",
    retention="tmp files are swept by the startup gate; trash by the next prune; corrupt copies are kept",
    encodings=("bin",),
))

Classifier = Callable[[str], FormatSpec | None]


@lru_cache(maxsize=512)
def path_pattern(path_key: str, encodings: tuple[str, ...]) -> re.Pattern[str]:
    """``path_key`` を相対 posix パスの正規表現 (``fullmatch`` で使う) にする。"""
    segments = path_key.split("/")
    parts: list[str] = []
    for i, segment in enumerate(segments):
        last = i == len(segments) - 1
        if segment == GLOBSTAR:
            # 末尾は 1 要素以上 (木の中身)、途中は 0 要素以上
            parts.append(r"[^/]+(?:/[^/]+)*" if last else r"(?:[^/]+/)*")
            continue
        body = _segment_regex(segment, encodings if last else ())
        parts.append(body if last else body + "/")
    return re.compile("".join(parts))


def _segment_regex(segment: str, encodings: tuple[str, ...]) -> str:
    stem, dot, ext = segment.rpartition(".")
    if dot and ext in encodings:
        return _literal_with_placeholders(stem) + r"\.(?:" + "|".join(map(re.escape, encodings)) + ")"
    return _literal_with_placeholders(segment)


def _literal_with_placeholders(text: str) -> str:
    return "[^/]+".join(re.escape(part) for part in PLACEHOLDER_RE.split(text))


def specificity(path_key: str) -> tuple[tuple[int, ...], int]:
    """宣言の具体性 (大きいほど具体的)。要素ごとの強さの列と字句の文字数。"""
    ranks: list[int] = []
    literal_chars = 0
    for segment in path_key.split("/"):
        if segment == GLOBSTAR:
            ranks.append(0)
            continue
        literal = PLACEHOLDER_RE.sub("", segment)
        literal_chars += len(literal)
        if literal == segment:
            ranks.append(3)
        else:
            ranks.append(2 if literal else 1)
    return tuple(ranks), literal_chars


def is_residue(rel: str) -> bool:
    """退避名・取り残しの一時ファイルを含むパスか。"""
    return any(_RESIDUE_SEGMENT_RE.fullmatch(part) for part in rel.split("/"))


def classifier(registry: FormatRegistry = FORMATS) -> Classifier:
    """相対 posix パス → 形式 (無ければ ``None``) の関数を作る (宣言を 1 回だけ並べる)。"""
    specs = sorted(
        (s for s in registry.all() if s is not RESIDUE_FORMAT),
        key=lambda s: s.format_id,
    )
    specs.sort(key=lambda s: specificity(s.path_key), reverse=True)  # 安定: 同順位は id 順
    compiled = [(path_pattern(s.path_key, s.encodings), s) for s in specs]

    def classify(rel: str) -> FormatSpec | None:
        if is_residue(rel):
            return RESIDUE_FORMAT
        for pattern, spec in compiled:
            if pattern.fullmatch(rel):
                return spec
        return None

    return classify


def match(rel: str, registry: FormatRegistry = FORMATS) -> FormatSpec | None:
    """世代フォルダからの相対 posix パス ``rel`` の形式。"""
    return classifier(registry)(rel)


def _pro_declared(specs: Iterable[FormatSpec]) -> bool:
    return any(s.path_key.startswith(PRO_STORE_PREFIX) for s in specs)


def walk(generation_dir: Path, registry: FormatRegistry = FORMATS) -> Iterator[tuple[Path, FormatSpec | None]]:
    """``store/`` の下の全ファイルを (パス, 形式) で返す (パス順、台帳外は ``None``)。

    ``generation_dir`` は世代フォルダ (``backend.data_root.generation_root``)。
    """
    root = Path(generation_dir)
    store = root / STORE_DIR
    if not store.is_dir():
        return
    classify = classifier(registry)
    skip_pro = not _pro_declared(registry.all())
    pro_dir = os.path.normcase(str(root / PRO_STORE_PREFIX.rstrip("/")))
    for current, dirnames, filenames in os.walk(store):
        if skip_pro:
            dirnames[:] = [d for d in dirnames if os.path.normcase(os.path.join(current, d)) != pro_dir]
        dirnames.sort()
        for name in sorted(filenames):
            path = Path(current) / name
            yield path, classify(path.relative_to(root).as_posix())


def iter_files(generation_dir: Path, registry: FormatRegistry = FORMATS) -> Iterator[tuple[Path, FormatSpec]]:
    """``store/`` の下で台帳のどれかの形式に属するファイルと、その形式。"""
    for path, spec in walk(generation_dir, registry):
        if spec is not None:
            yield path, spec


def unledgered(generation_dir: Path, registry: FormatRegistry = FORMATS) -> list[Path]:
    """``store/`` の下でどの形式にも属さないファイル (台帳の穴)。"""
    return [path for path, spec in walk(generation_dir, registry) if spec is None]


__all__ = [
    "RESIDUE_FORMAT",
    "STORE_DIR",
    "Classifier",
    "classifier",
    "is_residue",
    "iter_files",
    "match",
    "path_pattern",
    "specificity",
    "unledgered",
    "walk",
]
