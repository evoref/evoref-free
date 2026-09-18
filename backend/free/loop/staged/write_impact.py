"""書込み後の変更分類 (f_10 §8.1、Phase 2.5 → 3a-2 で meta 側の写像に統合)。

制作ステージ (:mod:`backend.free.agent.meta_cognitive`::``_execute_production_task``)
が ``output_target=="file"`` の書込みを終えた直後に呼ぶ。ProjectMap の root 配下へ
書いたパスだけを対象に、書いた後のファイルの fingerprint と active 版の fingerprint
を突き合わせて ``skip | partial | architecture | full`` を分類する。map 自体は
更新しない (書き手は sleep-time Step 5.87 の 1 本のまま、ここは観測専用)。

以前は ``backend/free/api/chat/chat_stream_staged.py`` (legacy ``_finalize_staged_stream``
の file 分岐) に置かれていたが、3a-2 でディスパッチを meta 経路の 1 本にしたのに伴い、
呼び手 (composition 層が組む classifier 経由で meta の ``_execute_production_task``)
に合わせてここへ移した。
"""

from __future__ import annotations

from pathlib import Path

from backend.free.rag.projectmap.classify import classify_update
from backend.free.rag.projectmap.fingerprint import compute_fingerprint, load_fingerprints

__all__ = ["staged_write_impact_payload"]


def _projectmap_subreaders(reader: object) -> list[object]:
    """``reader`` を ``.root``/``.directory`` を持つ単独 reader のリストへ正規化する。

    ``state.project_map_reader_getter()`` は単独 ``ProjectMapReader`` または
    複数 root を束ねる ``MultiProjectMapReader`` (``.readers`` 属性) のどちらか
    を返す (production_brief.py の Code map 節と同じ両対応)。
    """
    sub = getattr(reader, "readers", None)
    if sub is not None:
        return list(sub)
    return [reader] if reader is not None else []


def staged_write_impact_payload(
    reader: object, written_paths: list[str],
) -> list[dict]:
    """書込み後の変更分類 (f_10 §8.1)。

    ``reader`` の root 配下に書かれたパスだけを対象に、書いた後のファイルの
    fingerprint (:func:`compute_fingerprint`) と active 版の fingerprint
    (:func:`load_fingerprints`) の該当サブセットを ``classify_update`` に掛ける。
    map 自体は更新しない (書き手は sleep-time Step 5.87 の 1 本のまま、ここは
    観測専用)。reader が ``None`` / 対象パスが root 配下に無い場合は空リスト。
    """
    if reader is None or not written_paths:
        return []

    out: list[dict] = []
    for sub_reader in _projectmap_subreaders(reader):
        root = getattr(sub_reader, "root", None)
        directory = getattr(sub_reader, "directory", None)
        if root is None or directory is None:
            continue
        rel_to_abs: dict[str, Path] = {}
        for raw in written_paths:
            abs_path = Path(raw)
            try:
                rel = abs_path.relative_to(root)
            except ValueError:
                continue
            rel_to_abs[rel.as_posix()] = abs_path
        if not rel_to_abs:
            continue
        new_fp: dict[str, str] = {}
        for rel, abs_path in rel_to_abs.items():
            try:
                new_fp[rel] = compute_fingerprint(abs_path)
            except OSError:
                continue
        if not new_fp:
            continue
        old_fp = load_fingerprints(directory)
        old_subset = {p: h for p, h in old_fp.items() if p in new_fp}
        kind = classify_update(old_fingerprints=old_subset, new_fingerprints=new_fp)
        changed_files = sorted(p for p in new_fp if old_subset.get(p) != new_fp[p])
        out.append({
            "update_kind": kind,
            "changed_files": changed_files,
            "paths": sorted(new_fp),
        })
    return out
