"""``evorefmem_cli reembed-facts`` 実装

埋め込みモデルを差し替えた後、SemMem semantic fact の embedding ベクトルを
**新モデルで再生成**して active embedding store に書き戻す運用コマンド。

背景: ``migrate-embedding`` は manifest を書き換えるだけ (再埋め込みしない)、
``evoref reindex`` は RAG VectorStore / STM のみが対象で、semantic fact の
embedding は producer (url_curator / executable_command_curator) が生成時に
``fact.object`` を ``is_query=False`` で embed したきり再生成されない。
埋め込みモデルを同次元の別モデルへ swap すると ``dimension_check`` が差替を
検知できず (dim 一致)、旧モデル空間の fact ベクトルと新モデル空間のクエリが
非互換になり ``search_by_embedding`` (URL/コマンドリコール) が空振りする。

本コマンドは:

1. manifest の active model_id を ``--model-id`` / ``--dim`` へ swap し
   (``register_new_model`` + ``swap_active_model_id``)、
2. 全 scope の embedded fact を ``fact.object`` で再 embed して
   ``update_fact(embedding=...)`` で新 active store に書き戻し、
3. manifest ``last_migrated_at`` を更新する。

破壊的なため **デフォルト dry-run** (``--apply`` 必須)。``--apply`` 時は
``migration_archive/cli_<utc_ts>/reembed_facts/semantic/`` に semantic/ 全体を
退避してから書き換える。

注意: 埋め込み計算は呼び出し側が注入する ``embed_fn`` に委譲する
(CLI 本体は live llama-embed サーバへ ``doc_template`` 適用で問い合わせる、
単体テストはダミーを注入する)。``embed_fn`` は **embedding が None でない
live fact の ``fact.object`` テキスト** を受け取り、doc 側 (正規化済み) の
ベクトルを返す。
"""

from __future__ import annotations

import inspect
import shutil
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from backend.free.memory.semantic.cli._paths import (
    cli_backup_root,
    enumerate_scopes,
)
from backend.free.memory.semantic.embedding_store import (
    register_new_model,
    reset_model_store,
    swap_active_model_id,
)
from backend.free.memory.semantic.manifest import (
    load_manifest,
    update_manifest,
)
from backend.free.memory.semantic.stale_guard import (
    clear_semmem_reembed_required,
)
from backend.free.memory.semantic.store import SemanticFactStore
from backend.free.memory.sleep._curator_common import embed_kwargs_for_subject
from backend.log_config import get_logger

logger = get_logger("memory.semantic.cli.reembed_facts")

# embed_fn: 生の object テキスト列 -> ベクトル列 (正規化済み)。
# 内部索引ファクト (executable command 等) は query 側で埋め込む必要がある
# (``_curator_common.INDEX_EMBED_IS_QUERY``)。embed_fn が ``is_query`` /
# ``mode`` キーワードを受け取れる場合はそれを渡し、受け取れない (doc 側専用の)
# embed_fn には渡さず WARNING を出して doc 側で埋める。
EmbedFn = Callable[..., Sequence[Sequence[float]]]


class ReembedTarget(NamedTuple):
    """再埋め込み対象 1 件。

    ``embed_kwargs`` は ``embedder.embed(texts, **embed_kwargs)`` へ渡す側の
    指定 (``None`` = document 側の既定)。``targets[i][3]`` で本文を取る既存
    呼出側 (``POST /api/model/reembed-facts``) との互換のため tuple のまま。
    """

    scope_name: str
    scope_root: Path
    fact_id: str
    text: str
    embed_kwargs: dict[str, Any] | None = None


def group_targets_by_embed_kwargs(
    targets: Sequence[ReembedTarget],
) -> list[tuple[dict[str, Any] | None, list[int]]]:
    """``embed_kwargs`` が同じ target の添字をまとめる (順序は初出順)。"""
    groups: dict[tuple[tuple[str, Any], ...] | None, list[int]] = {}
    kwargs_of: dict[tuple[tuple[str, Any], ...] | None, dict[str, Any] | None] = {}
    for i, t in enumerate(targets):
        kw = t[4] if len(t) > 4 else None
        key = None if not kw else tuple(sorted(kw.items()))
        groups.setdefault(key, []).append(i)
        kwargs_of.setdefault(key, kw or None)
    return [(kwargs_of[k], idxs) for k, idxs in groups.items()]


def _embed_fn_accepts_side(embed_fn: Callable[..., Any]) -> bool:
    """``embed_fn`` が ``is_query`` キーワード (または ``**kwargs``) を受けるか。"""
    try:
        params = inspect.signature(embed_fn).parameters
    except (TypeError, ValueError):
        return False
    if "is_query" in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def embed_targets(
    embed_fn: EmbedFn, targets: Sequence[ReembedTarget],
) -> list[Sequence[float]]:
    """``targets`` を側ごとにまとめて embed し、元の順序でベクトル列を返す (sync)。"""
    out: list[Sequence[float] | None] = [None] * len(targets)
    accepts_side = _embed_fn_accepts_side(embed_fn)
    for kwargs, idxs in group_targets_by_embed_kwargs(targets):
        texts = [targets[i][3] for i in idxs]
        if kwargs and not accepts_side:
            logger.warning(
                "reembed-facts: embed_fn does not accept is_query/mode; %d "
                "index fact(s) will be embedded on the document side "
                "(recall similarity will be degraded)", len(idxs),
            )
        vecs = embed_fn(texts, **kwargs) if (kwargs and accepts_side) else embed_fn(texts)
        if len(vecs) != len(idxs):
            raise ValueError(
                f"embed_fn returned {len(vecs)} vectors for {len(idxs)} texts",
            )
        for i, v in zip(idxs, vecs):
            out[i] = v
    return [v if v is not None else [] for v in out]


async def embed_targets_async(
    embedder: Any, targets: Sequence[ReembedTarget],
) -> list[Sequence[float]]:
    """``embedder.embed(texts, is_query=..., mode=...)`` で側ごとに埋め込む (async)。

    ``POST /api/model/reembed-facts`` のような ``EmbeddingBackend`` を持つ呼出側
    向け。全 target を doc 側で ``embed(objs, is_query=False)`` すると索引
    ファクトの側が反転するので、こちらを使うこと。
    """
    out: list[Sequence[float] | None] = [None] * len(targets)
    for kwargs, idxs in group_targets_by_embed_kwargs(targets):
        texts = [targets[i][3] for i in idxs]
        vecs = await embedder.embed(texts, **(kwargs or {"is_query": False}))
        if vecs is None or len(vecs) != len(idxs):
            raise ValueError(
                f"embedder returned {0 if vecs is None else len(vecs)} vectors "
                f"for {len(idxs)} texts",
            )
        for i, v in zip(idxs, vecs):
            out[i] = v
    return [v if v is not None else [] for v in out]


@dataclass
class ReembedFactsReport:
    memory_dir: str
    new_model_id: str
    new_dim: int
    normalized: bool
    applied: bool
    current_model_id: str | None = None
    current_dim: int | None = None
    # scope 名 -> 再 embed 対象 fact 数
    per_scope_counts: dict[str, int] = field(default_factory=dict)
    total_targets: int = 0
    reembedded: int = 0
    backup_path: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        import json
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


def collect_reembed_targets(
    memory_dir: Path, manifest,
) -> list[ReembedTarget]:
    """全 scope の「embedding を持つ live fact」を列挙する.

    Returns:
        :class:`ReembedTarget` (``scope_name, scope_root, fact_id, embed_text,
        embed_kwargs``) のリスト。``embedding`` が None / superseded / 本文が
        空の fact は除外。本文は ``fact.text`` (= ``statement or object``)。
        ``embed_kwargs`` は内部索引ファクトのみ非 None
        (:func:`~backend.free.memory.sleep._curator_common.embed_kwargs_for_subject`)。
    """
    targets: list[ReembedTarget] = []
    for scope in enumerate_scopes(memory_dir):
        store = SemanticFactStore(scope.root_dir, manifest=manifest)
        for fact in store.all_facts(include_superseded=False):
            if fact.embedding is None:
                continue
            # Step 8.8 (sleep/fact_embedding) と同じ本文を使う。片方が
            # ``object``、片方が ``text`` だと、モデル切替後のストアで
            # **同じファクトが別の本文で埋め込まれた状態** (split-brain) が残り、
            # 検索の当たり方が再埋め込みの有無で変わる (2026-08-26 に是正)。
            text = (fact.text or "").strip()
            if not text:
                continue
            targets.append(ReembedTarget(
                scope.name, scope.root_dir, fact.id, text,
                embed_kwargs_for_subject(fact.subject),
            ))
    return targets


def apply_reembed_swap(
    memory_dir: Path,
    migration_archive_dir: Path,
    targets: Sequence[ReembedTarget],
    vectors: Sequence[Sequence[float]],
    *,
    new_model_id: str,
    new_dim: int,
    normalized: bool = True,
    now: float | None = None,
) -> tuple[int, str | None]:
    """embed 済みベクトル列で新 model_id ストアへ swap + 書き戻し + manifest 更新.

    ``run_reembed_facts`` (CLI, sync embed) と ``POST /api/model/reembed-facts``
    (endpoint, async embed) が共有する cross-model swap のコア。呼出側は
    ``collect_reembed_targets`` で得た ``targets`` の object テキストを embed 済みの
    ``vectors`` (``targets`` と同順・同数) を渡す。実行内容:

    1. ``semantic/`` 全体を ``migration_archive`` へ退避、
    2. 全 scope に新 model_id 用の空ストアを作成し active model を atomic swap、
    3. ``targets`` の各 fact を ``update_fact(embedding=...)`` で新ストアへ書戻し、
    4. manifest の ``last_migrated_at`` 更新 + SemMem stale マーカー解除。

    ``targets`` が空でも manifest swap + マーカー解除は行う (fact ゼロの環境でも
    モデルラベルを揃えて stale 検知の 409 を解消するため)。

    Returns:
        ``(書き戻した fact 数, backup_path)``。

    Raises:
        ValueError: ``new_model_id`` 空 / ``new_dim < 1`` / ベクトル数・次元の不整合。
    """
    if not new_model_id:
        raise ValueError("new_model_id must be non-empty")
    if new_dim < 1:
        raise ValueError("new_dim must be >= 1")
    if len(vectors) != len(targets):
        raise ValueError(
            f"vectors count {len(vectors)} != targets count {len(targets)}",
        )

    # ベクトルを swap 前に全件検証する (不整合時に manifest を壊さないため)。
    vecs: list[np.ndarray] = []
    for i, v in enumerate(vectors):
        arr = np.asarray(v, dtype=np.float32).reshape(-1)
        if arr.shape[0] != new_dim:
            raise ValueError(
                f"vector #{i} has dim {arr.shape[0]}, expected {new_dim}",
            )
        vecs.append(arr)

    # 1. semantic/ 全体をバックアップ。
    backup_root = cli_backup_root(
        migration_archive_dir, "reembed_facts", now=now,
    )
    semantic_dir = memory_dir / "semantic"
    backup_dst = backup_root / "semantic"
    backup_path: str | None = None
    if semantic_dir.exists():
        shutil.copytree(semantic_dir, backup_dst, dirs_exist_ok=True)
        backup_path = str(backup_dst)

    # 2. 新 model_id 用のストアを全 scope で **空にリセット** してから作成し、
    #    active model を swap。reset により、既存 dir の残存ベクトル (異なる dim /
    #    削除済み fact の orphan 行) が原因で swap 後に書込み dim-mismatch → half-swap
    #    で復旧不能になる事故を防ぐ (authoritative swap)。backup は step 1 で取得済み。
    for scope in enumerate_scopes(memory_dir):
        reset_model_store(scope.root_dir, new_model_id)
        register_new_model(scope.root_dir, new_model_id)
    swap_active_model_id(
        memory_dir, new_model_id, new_dim=new_dim, normalized=normalized,
    )

    # 3. 新 manifest で各 scope の store を開き直し、update_fact で書き戻す。
    new_manifest = load_manifest(memory_dir)
    stores: dict[Path, SemanticFactStore] = {}
    reembedded = 0
    for target, vec in zip(targets, vecs):
        scope_root, fact_id = target[1], target[2]
        store = stores.get(scope_root)
        if store is None:
            store = SemanticFactStore(scope_root, manifest=new_manifest)
            stores[scope_root] = store
        store.update_fact(fact_id, embedding=vec)
        reembedded += 1

    # 4. manifest の last_migrated_at を更新 + stale マーカー解除。
    stamp = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ",
        time.gmtime(time.time() if now is None else now),
    )
    update_manifest(memory_dir, last_migrated_at=stamp)
    clear_semmem_reembed_required(memory_dir=memory_dir)
    return reembedded, backup_path


def run_reembed_facts(
    memory_dir: Path,
    migration_archive_dir: Path,
    *,
    new_model_id: str,
    new_dim: int,
    embed_fn: EmbedFn | None = None,
    normalized: bool = True,
    apply: bool = False,
    now: float | None = None,
) -> ReembedFactsReport:
    """semantic fact の embedding を新モデルで再生成して書き戻す。"""
    if not new_model_id:
        raise ValueError("new_model_id must be non-empty")
    if new_dim < 1:
        raise ValueError("new_dim must be >= 1")

    manifest = load_manifest(memory_dir)
    rep = ReembedFactsReport(
        memory_dir=str(memory_dir),
        new_model_id=new_model_id,
        new_dim=new_dim,
        normalized=normalized,
        applied=apply,
        current_model_id=manifest.embedding.model_id if manifest else None,
        current_dim=manifest.embedding.dim if manifest else None,
    )
    if manifest is None:
        rep.error = (
            "manifest.json not found; run `evorefmem_cli init` first"
        )
        return rep

    targets = collect_reembed_targets(memory_dir, manifest)
    for target in targets:
        scope_name = target[0]
        rep.per_scope_counts[scope_name] = (
            rep.per_scope_counts.get(scope_name, 0) + 1
        )
    rep.total_targets = len(targets)

    if not apply:
        return rep

    if embed_fn is None:
        rep.error = "embed_fn is required for --apply"
        return rep
    if not targets:
        rep.error = "no embedded facts to reembed (nothing to do)"
        return rep

    # ベクトルを swap 前に全件計算する (embed 失敗時に manifest を壊さないため)。
    # 内部索引ファクトは query 側で埋める (embed_fn が側を受け取れる場合)。
    vectors = embed_targets(embed_fn, targets)
    try:
        reembedded, backup_path = apply_reembed_swap(
            memory_dir, migration_archive_dir, targets, vectors,
            new_model_id=new_model_id, new_dim=new_dim,
            normalized=normalized, now=now,
        )
    except ValueError as exc:
        rep.error = str(exc)
        return rep
    rep.reembedded = reembedded
    rep.backup_path = backup_path
    return rep


def format_report_text(report: ReembedFactsReport) -> str:
    lines: list[str] = []
    lines.append(f"memory_dir       : {report.memory_dir}")
    mode = "apply" if report.applied else "dry-run"
    lines.append(f"mode             : {mode}")
    lines.append(
        f"current          : model_id={report.current_model_id} "
        f"dim={report.current_dim}",
    )
    lines.append(
        f"target           : model_id={report.new_model_id} "
        f"dim={report.new_dim} normalized={report.normalized}",
    )
    if report.error:
        lines.append("")
        lines.append(f"ERROR: {report.error}")
        return "\n".join(lines)
    lines.append(f"embedded facts   : {report.total_targets}")
    for scope_name in sorted(report.per_scope_counts):
        lines.append(
            f"  - {scope_name:24} {report.per_scope_counts[scope_name]}",
        )
    if report.applied:
        lines.append(f"reembedded       : {report.reembedded}")
        if report.backup_path:
            lines.append(f"backup           : {report.backup_path}")
        lines.append("")
        lines.append(
            "NOTE: restart `evoref serve` so the backend reloads the new "
            "manifest + vectors (a running backend still holds the old "
            "embedding store in memory).",
        )
    else:
        lines.append("")
        lines.append(
            "(dry-run; rerun with --apply to reembed + swap manifest)",
        )
    return "\n".join(lines)


__all__ = [
    "EmbedFn",
    "ReembedFactsReport",
    "ReembedTarget",
    "apply_reembed_swap",
    "collect_reembed_targets",
    "embed_targets",
    "embed_targets_async",
    "format_report_text",
    "group_targets_by_embed_kwargs",
    "run_reembed_facts",
]
