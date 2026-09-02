"""model-agnostic な埋め込みストア

埋め込みは ``<scope_dir>/embeddings/<model_id>/vectors.npy`` +
``<scope_dir>/embeddings/<model_id>/row_to_id.json`` の **model_id 別
ディレクトリ** に分離して保持する。これにより以下が可能になる:

- BGE-M3 (1024 次元) と Qwen3-Embedding (4096 次元) のように次元違いの
  モデルを並存させ、manifest の ``embedding.model_id`` 書換だけで atomic に
  アクティブ切替できる
- 旧モデル配下は retention として残したまま新モデル配下の再埋め込みを
  バックグラウンドで走らせられる
- ``SemanticFactStore`` 本体から「次元がモデル依存」という暗黙結合を取り除く

## 置き所

``backend.free.memory.semantic`` pillar 内部の補助モジュール。``SemanticFactStore``
からのみ参照される。他 pillar は Fact View 経由でしか Store にアクセス
できないため、``EmbeddingStore`` が pillar 境界を超えることは無い。
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Final

import numpy as np

from backend.io import AtomicWriter
from backend.log_config import get_logger

logger = get_logger("memory.semantic.embedding_store")


# ──────────────────────────────────────────────────────────────────────────
# 定数
# ──────────────────────────────────────────────────────────────────────────

EMBEDDINGS_DIRNAME: Final[str] = "embeddings"
"""scope ディレクトリ直下の埋め込みルート名 (例: ``global/embeddings/``)。"""

VECTORS_FILENAME: Final[str] = "vectors.npy"
"""1 モデル分の埋め込み配列ファイル名。"""

ROW_TO_ID_FILENAME: Final[str] = "row_to_id.json"
"""vectors.npy の各行と fact_id の対応表。"""

DEFAULT_MODEL_ID: Final[str] = "default"
"""manifest が無い fixture 向けのフォールバック model_id。

本番経路では必ず manifest から model_id が解決されるため、このフォールバックは
テストコードの呼び出し互換を維持するためだけに使う (pytest の
``SemanticFactStore(tmp_path / "global")`` 等)。
"""


# ──────────────────────────────────────────────────────────────────────────
# パス
# ──────────────────────────────────────────────────────────────────────────


def embeddings_root(scope_dir: Path) -> Path:
    """``<scope_dir>/embeddings/`` の絶対パス。"""
    return Path(scope_dir) / EMBEDDINGS_DIRNAME


def model_dir(scope_dir: Path, model_id: str) -> Path:
    """``<scope_dir>/embeddings/<model_id>/`` の絶対パス。"""
    if not model_id:
        raise ValueError("model_id must be non-empty")
    return embeddings_root(scope_dir) / model_id


def vectors_path(scope_dir: Path, model_id: str) -> Path:
    return model_dir(scope_dir, model_id) / VECTORS_FILENAME


def row_to_id_path(scope_dir: Path, model_id: str) -> Path:
    return model_dir(scope_dir, model_id) / ROW_TO_ID_FILENAME


# ──────────────────────────────────────────────────────────────────────────
# EmbeddingStore 本体
# ──────────────────────────────────────────────────────────────────────────


class EmbeddingStore:
    """1 scope × 1 model_id 分の埋め込みを管理するストア.

    ``<scope_dir>/embeddings/<model_id>/vectors.npy`` と ``row_to_id.json``
    のペアを読み書きする。インスタンスは「**アクティブ 1 モデル**」の
    抽象で、モデル切替は :func:`swap_active_model_id` 経由で manifest
    を書換えた後、新しい EmbeddingStore を作り直す想定。

    ``SemanticFactStore`` から保持される lifetime は同一。embedding 未設定
    の fact が大半のストアでは ``vectors.npy`` / ``row_to_id.json`` を
    作成しない (空ファイルを残さない)。

    Parameters
    ----------
    scope_dir:
        ``semantic/global/`` または ``semantic/projects/<id>/`` ディレクトリ。
    model_id:
        manifest.embedding.model_id。ファイルパスの subdir 名になる。
        fixture 等で model_id が不明な場合は :data:`DEFAULT_MODEL_ID` を使う。
    """

    def __init__(self, scope_dir: Path, model_id: str) -> None:
        if not model_id:
            raise ValueError("model_id must be non-empty")
        self._scope_dir = Path(scope_dir)
        self._model_id = model_id
        self._vectors: np.ndarray | None = None  # shape (N, dim)
        self._row_to_id: list[str] = []
        self._id_to_row: dict[str, int] = {}
        #: ``flush=False`` の変更が未保存で残っているか (:meth:`flush` 用)。
        self._dirty: bool = False
        #: 行ノルムのキャッシュ (``None`` = 正規化済みで割り算不要)。
        self._norms_cache: np.ndarray | None = None
        #: ``_norms_cache`` を計算したときの行数 (行列が変わったら破棄する)。
        self._norms_rows: int = -1
        self._load()

    # ── クラスメソッド ────────────────────────────────────────────────

    @classmethod
    def active(
        cls,
        scope_dir: Path,
        *,
        manifest_model_id: str | None = None,
    ) -> EmbeddingStore:
        """アクティブなモデルのストアを返す.

        ``manifest_model_id`` を渡せばそれをそのまま使う。渡されなかった場合は
        ``<scope_dir>/embeddings/`` 配下を走査し、以下の優先順で model_id を
        決定する:

        1. 配下にちょうど 1 つ subdir があればそれを採用
        2. 何も無ければ :data:`DEFAULT_MODEL_ID`
        3. 複数 subdir がある場合は ``RuntimeError`` (manifest を渡してほしい)
        """
        scope_dir = Path(scope_dir)
        if manifest_model_id:
            return cls(scope_dir, manifest_model_id)
        root = embeddings_root(scope_dir)
        if not root.exists():
            return cls(scope_dir, DEFAULT_MODEL_ID)
        subdirs = sorted(p.name for p in root.iterdir() if p.is_dir())
        if len(subdirs) == 1:
            return cls(scope_dir, subdirs[0])
        if not subdirs:
            return cls(scope_dir, DEFAULT_MODEL_ID)
        raise RuntimeError(
            f"ambiguous active embedding model under {root}: found "
            f"{subdirs!r}; pass manifest_model_id explicitly",
        )

    # ── プロパティ ────────────────────────────────────────────────────

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def scope_dir(self) -> Path:
        return self._scope_dir

    @property
    def count(self) -> int:
        return 0 if self._vectors is None else int(self._vectors.shape[0])

    @property
    def dim(self) -> int | None:
        if self._vectors is None or self._vectors.shape[0] == 0:
            return None
        return int(self._vectors.shape[1])

    @property
    def is_empty(self) -> bool:
        return self.count == 0

    @property
    def vectors(self) -> np.ndarray | None:
        """生の埋め込み配列 (読み取り専用想定)。空なら None。"""
        return self._vectors

    @property
    def row_to_id(self) -> list[str]:
        """行番号 -> fact_id マッピング (浅いコピー)。"""
        return list(self._row_to_id)

    # ── パス ──────────────────────────────────────────────────────────

    def _vectors_path(self) -> Path:
        return vectors_path(self._scope_dir, self._model_id)

    def _row_to_id_path(self) -> Path:
        return row_to_id_path(self._scope_dir, self._model_id)

    # ── ロード / 保存 ─────────────────────────────────────────────────

    def _load(self) -> None:
        vp = self._vectors_path()
        ip = self._row_to_id_path()
        if not (vp.exists() and ip.exists()):
            return
        try:
            arr = np.load(vp)
            idx = json.loads(ip.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            logger.warning(
                "Failed to load embeddings from %s: %s. Resetting store.",
                vp, exc,
            )
            return
        row_to_id = idx.get("row_to_id", [])
        if not isinstance(row_to_id, list) or len(row_to_id) != arr.shape[0]:
            logger.warning(
                "Embeddings index/array mismatch at %s (%s rows vs %s ids). "
                "Resetting.",
                vp, arr.shape[0], len(row_to_id),
            )
            return
        self._vectors = arr.astype(np.float32, copy=False)
        self._row_to_id = list(row_to_id)
        self._id_to_row = {fid: i for i, fid in enumerate(self._row_to_id)}
        self._norms_rows = -1

    def save(self) -> None:
        """現在の state を ``vectors.npy`` + ``row_to_id.json`` に書き出す.

        空の場合は既存ファイルを削除し、空ファイルを残さない。

        **2 ファイルとも :class:`AtomicWriter` 経由で書く。** EvorefMem の
        永続化はすべて atomic 書込を通る (manifest / local_state_store /
        short_term_store / ``SemanticFactStore._rewrite_facts_log``) 中で、
        ここだけが ``np.save`` + ``write_text`` の直接書き込みだった。この 2 手
        の間で落ちると行数が食い違い、次回起動の :meth:`_load` が
        ``len(row_to_id) != arr.shape[0]`` を検出して **WARNING 1 行を出して
        ストアを空にする**。その後の最初の :meth:`upsert` が
        ``self._vectors is None`` 分岐へ入り、(1, dim) の配列で
        ``vectors.npy`` を上書きしてそのスコープの全ベクトルを消す。

        順序は **vectors → row_to_id** に固定する。``facts.jsonl`` から
        ベクトルを外した (:func:`~backend.free.memory.types.serialize_fact_jsonl`)
        今、npy が唯一の正なので、破断時に復旧できない状態を作らない。
        """
        vp = self._vectors_path()
        ip = self._row_to_id_path()
        if self._vectors is None or self._vectors.shape[0] == 0:
            for path in (vp, ip):
                if path.exists():
                    path.unlink()
            self._dirty = False
            return
        vp.parent.mkdir(parents=True, exist_ok=True)
        with AtomicWriter(vp, mode="wb") as fh:
            np.save(fh, self._vectors)
        with AtomicWriter(ip) as fh:
            fh.write(
                json.dumps(
                    {"row_to_id": list(self._row_to_id)}, ensure_ascii=False,
                ),
            )
        self._dirty = False

    def flush(self) -> None:
        """``flush=False`` で溜めた変更があれば書き出す (無ければ何もしない)。"""
        if self._dirty:
            self.save()

    # ── CRUD ──────────────────────────────────────────────────────────

    def upsert(self, fact_id: str, vec: np.ndarray, *, flush: bool = True) -> None:
        """埋め込みを追加または上書きする.

        - 初回追加時は (1, dim) の配列を作る
        - 既存行があれば in-place 更新
        - 新規行は末尾 append
        - 次元が既存配列と一致しない場合は ``ValueError``

        Args:
            flush: ``True`` (既定) なら即座に :meth:`save` する。バッチで
                書く呼出側 (Step 8.8 の埋め込み遡及生成は 1 サイクル最大 200
                件、curator 群も対を束ねて書く) は ``False`` を渡し、末尾で
                :meth:`flush` を 1 回呼ぶ。1 件ごとに保存すると **全配列を
                毎回ディスクへ書き出す** ため、200 件で 200 回の全書き出しに
                なる。
        """
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        if self._vectors is None or self._vectors.shape[0] == 0:
            self._vectors = v.reshape(1, -1).copy()
            self._row_to_id = [fact_id]
            self._id_to_row = {fact_id: 0}
            self._dirty = True
            self._norms_rows = -1
            if flush:
                self.save()
            return
        if v.shape[0] != self._vectors.shape[1]:
            raise ValueError(
                f"embedding dim mismatch: existing={self._vectors.shape[1]} "
                f"new={v.shape[0]}",
            )
        existing_row = self._id_to_row.get(fact_id)
        if existing_row is not None:
            self._vectors[existing_row] = v
        else:
            self._vectors = np.vstack([self._vectors, v[np.newaxis, :]])
            row = self._vectors.shape[0] - 1
            self._row_to_id.append(fact_id)
            self._id_to_row[fact_id] = row
        self._dirty = True
        self._norms_rows = -1
        if flush:
            self.save()

    def delete(self, fact_id: str, *, flush: bool = True) -> bool:
        """指定 fact の埋め込みを取り除く.

        Args:
            flush: ``False`` なら保存を遅延する (:meth:`delete_many` /
                バッチ削除用)。

        Returns:
            実際に削除した場合 True、元々存在しなかった場合 False。
        """
        if self._vectors is None or fact_id not in self._id_to_row:
            return False
        row = self._id_to_row.pop(fact_id)
        self._vectors = np.delete(self._vectors, row, axis=0)
        del self._row_to_id[row]
        self._id_to_row = {fid: i for i, fid in enumerate(self._row_to_id)}
        self._dirty = True
        self._norms_rows = -1
        if flush:
            self.save()
        return True

    def delete_many(self, fact_ids: "Iterable[str]") -> int:
        """複数 fact の埋め込みをまとめて取り除く.

        1 件ずつ :meth:`delete` を回すと、``np.delete`` の配列コピー・
        ``_id_to_row`` の全再構築・``vectors.npy`` の全書き出しが件数分
        走る (M 件で O(N*M) + M 回の全書き出し)。行マスクを 1 度作って
        1 回で落とす。

        Returns:
            実際に削除した件数。
        """
        if self._vectors is None or self._vectors.shape[0] == 0:
            return 0
        targets = {
            fid for fid in fact_ids if fid in self._id_to_row
        }
        if not targets:
            return 0
        keep = [
            i for i, fid in enumerate(self._row_to_id) if fid not in targets
        ]
        self._vectors = self._vectors[keep]
        self._row_to_id = [self._row_to_id[i] for i in keep]
        self._id_to_row = {fid: i for i, fid in enumerate(self._row_to_id)}
        self._dirty = True
        self._norms_rows = -1
        self.save()
        return len(targets)

    def get(self, fact_id: str) -> np.ndarray | None:
        """fact_id の埋め込みベクトル (コピー) を返す。無ければ None。"""
        if self._vectors is None:
            return None
        row = self._id_to_row.get(fact_id)
        if row is None:
            return None
        return self._vectors[row].copy()

    # ── 検索 ──────────────────────────────────────────────────────────

    def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
        *,
        fact_ids: Iterable[str] | None = None,
    ) -> list[tuple[str, float]]:
        """cosine similarity top-k を返す (fact_id, score の降順).

        空ストアの場合は空リスト。query 次元不一致は ``ValueError``。

        ``top_k`` が全件より小さいときは ``np.argpartition`` で上位だけを
        取り出す (``np.argsort`` は常に全件 O(N log N) を払う)。行のノルムは
        :meth:`_row_norms` がキャッシュし、正規化済みストアでは計算自体を
        省く。

        Args:
            fact_ids: 与えると **その行だけ** を候補にして順位付けする
                (subject 接頭辞で絞った索引リコール用)。ストアに無い id は
                黙って落とす。``None`` なら全行。
        """
        if self._vectors is None or self._vectors.shape[0] == 0:
            return []
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        if q.shape[0] != self._vectors.shape[1]:
            raise ValueError(
                f"query dim mismatch: expected={self._vectors.shape[1]} "
                f"got={q.shape[0]}",
            )
        qn = float(np.linalg.norm(q))
        if qn == 0.0:
            return []
        rows: np.ndarray | None = None
        if fact_ids is not None:
            picked = sorted(
                {r for fid in fact_ids if (r := self._id_to_row.get(fid)) is not None},
            )
            if not picked:
                return []
            rows = np.asarray(picked, dtype=np.int64)
        vectors = self._vectors if rows is None else self._vectors[rows]
        sims = vectors @ q
        norms = self._row_norms()
        if norms is None:
            sims = sims / qn
        else:
            denom = (norms if rows is None else norms[rows]) * qn
            denom[denom == 0.0] = 1e-9
            sims = sims / denom
        n = int(sims.shape[0])
        k = max(1, min(int(top_k), n))
        if k < n:
            # 上位 k 件だけを選んでから、その中だけを降順に並べる。
            head = np.argpartition(-sims, k - 1)[:k]
            order = head[np.argsort(-sims[head])]
        else:
            order = np.argsort(-sims)
        return [
            (
                self._row_to_id[int(row) if rows is None else int(rows[int(row)])],
                float(sims[int(row)]),
            )
            for row in order
        ]

    def score_all(self, query: np.ndarray) -> dict[str, float]:
        """全 fact に対する cosine similarity を ``{fact_id: score}`` で返す.

        並べ替えをしない点が :meth:`search` と違う。注入の関連度ゲートは
        「閾値を超えたか」しか見ないので順位付けが要らず、**行列がストア側に
        常駐している** ぶん候補ごとの numpy 演算をまるごと省ける。

        実測 (2026-09-01、1024 次元): 候補ごとに ``np.asarray`` →
        ``np.linalg.norm`` → 除算 を回すと N=10000 で 52.5ms かかるのに対し、
        常駐行列との 1 回の積は 1.5ms (**35 倍**)。候補リストからその都度
        行列を組み直す方式は ``np.stack`` のコピーが乗るぶん逆に遅くなるので
        (実測で 0.5〜0.8 倍)、**行列を持っている側で計算する** ことが要点。

        次元不一致 / 空ストア / ゼロクエリは空 dict を返す (呼出側は従来の
        候補ごとの判定へ落ちる)。
        """
        if self._vectors is None or self._vectors.shape[0] == 0:
            return {}
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        if q.shape[0] != self._vectors.shape[1]:
            return {}
        qn = float(np.linalg.norm(q))
        if not qn or not np.isfinite(qn):
            return {}
        sims = self._vectors @ q
        norms = self._row_norms()
        if norms is None:
            sims = sims / qn
        else:
            denom = norms * qn
            denom[denom == 0.0] = 1e-9
            sims = sims / denom
        return {
            fid: float(sims[i]) for i, fid in enumerate(self._row_to_id)
        }

    def _row_norms(self) -> np.ndarray | None:
        """行ごとの L2 ノルム。正規化済みと分かっていれば ``None``。

        ``manifest.embedding.normalized`` は「L2 正規化済み」を宣言する
        フィールドだが、**消費者が 1 つも無かった** — manifest / CLI 内で
        書かれ比較されるだけで、検索側は誰も読んでいなかった (2026-09-01 監査
        F8)。実ストアのベクトルは実際に ``norm min=1.0000 max=1.0000`` で、
        それでも検索のたびに全行のノルムを計算し直していた。

        宣言を鵜呑みにはしない: 実データを 1 度だけサンプル検査し、本当に
        正規化されているときだけ計算を省く。結果は行列が入れ替わるまで
        キャッシュする。
        """
        if self._vectors is None:
            return None
        rows = int(self._vectors.shape[0])
        # ``_norms_rows`` が有効性の印。``_norms_cache is None`` は「正規化済み
        # なので割り算不要」を意味するので、None を「未計算」と混同しない。
        if self._norms_rows == rows:
            return self._norms_cache
        norms = np.linalg.norm(self._vectors, axis=1)
        # 全行が単位ベクトルなら以後の割り算ごと省ける。
        self._norms_cache = (
            None if bool(np.allclose(norms, 1.0, atol=1e-4)) else norms
        )
        self._norms_rows = rows
        return self._norms_cache

    # ── 観測 ──────────────────────────────────────────────────────────

    def __contains__(self, fact_id: object) -> bool:
        return isinstance(fact_id, str) and fact_id in self._id_to_row

    def __len__(self) -> int:
        return self.count


# ──────────────────────────────────────────────────────────────────────────
# マルチモデル運用ヘルパ
# ──────────────────────────────────────────────────────────────────────────


def list_stored_models(scope_dir: Path) -> list[str]:
    """``<scope_dir>/embeddings/`` 配下に存在する model_id を列挙する."""
    root = embeddings_root(Path(scope_dir))
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def reset_model_store(scope_dir: Path, model_id: str) -> None:
    """指定 model_id の埋め込みストア (vectors.npy + row_to_id.json) を削除して空にする.

    full re-embed (:func:`apply_reembed_swap`) で新 model_id dir が既存の stale
    データを持つ場合に、書き込み前にリセットして「そのモデルのベクトルは今回の
    対象 fact のみ」を保証する (authoritative swap)。これにより:

    - 既存 dir の dim が新 dim と異なる場合の書込み時 dim-mismatch (manifest swap
      後に発生すると half-swap で復旧不能になる) を防ぐ、
    - 削除済み fact の orphan 行が残って誤ヒットするのを防ぐ。

    ``register_new_model`` は idempotent で既存 dir をそのまま返すため、それだけ
    では上記を保証できない。呼出側は本関数の後に ``register_new_model`` を呼ぶ。
    ディレクトリ自体は残す (ファイルのみ削除)。
    """
    scope_dir = Path(scope_dir)
    vectors_path(scope_dir, model_id).unlink(missing_ok=True)
    row_to_id_path(scope_dir, model_id).unlink(missing_ok=True)


def register_new_model(
    scope_dir: Path,
    model_id: str,
) -> EmbeddingStore:
    """新しい model_id 用の空ストアディレクトリを作成して返す.

    並行埋め込み計算のために呼ぶ想定。既にディレクトリがあっても例外は
    出さず、既存のストアを返す (idempotent)。
    """
    scope_dir = Path(scope_dir)
    model_dir(scope_dir, model_id).mkdir(parents=True, exist_ok=True)
    logger.info(
        "registered new embedding model dir: %s",
        model_dir(scope_dir, model_id),
    )
    return EmbeddingStore(scope_dir, model_id)


def swap_active_model_id(
    memory_dir: Path,
    new_model_id: str,
    *,
    new_dim: int,
    normalized: bool = True,
) -> None:
    """manifest.embedding を atomic 書換えてアクティブモデルを切替える.

    呼び出し側は事前に ``register_new_model`` で新モデル用の埋め込みを
    完成させている必要がある。本関数は manifest のみ更新し、旧モデルの
    ディレクトリ削除は行わない (retention 期間中に retention GC が担う)。

    Args:
        memory_dir: ``<local>/memory/`` ルート。
        new_model_id: アクティブ切替先の model_id。
        new_dim: 新モデルの埋め込み次元数。
        normalized: L2 正規化済み埋め込みか。manifest.embedding.normalized
            と同じ意味。

    Raises:
        FileNotFoundError: manifest.json が未作成の場合。
    """
    from backend.free.memory.semantic.manifest import (
        EmbeddingManifest,
        update_manifest,
    )

    if not new_model_id:
        raise ValueError("new_model_id must be non-empty")
    if new_dim < 1:
        raise ValueError("new_dim must be >= 1")

    update_manifest(
        memory_dir,
        embedding=EmbeddingManifest(
            model_id=new_model_id, dim=new_dim, normalized=normalized,
        ),
    )
    logger.info(
        "swap_active_model_id: memory_dir=%s model_id=%s dim=%d",
        memory_dir, new_model_id, new_dim,
    )


__all__ = [
    "DEFAULT_MODEL_ID",
    "EMBEDDINGS_DIRNAME",
    "EmbeddingStore",
    "ROW_TO_ID_FILENAME",
    "VECTORS_FILENAME",
    "embeddings_root",
    "list_stored_models",
    "model_dir",
    "register_new_model",
    "reset_model_store",
    "row_to_id_path",
    "swap_active_model_id",
    "vectors_path",
]
