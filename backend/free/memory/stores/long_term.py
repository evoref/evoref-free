"""Layer 3: 長期記憶（RAG ベクトル DB + BM25 ハイブリッド + タグ検索）"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from backend.log_config import get_logger
from backend.free.core.text_quality import carries_no_assertion
from backend.free.rag.vector_store import VectorStore

if TYPE_CHECKING:
    from backend.free.rag.bm25_retriever import BM25Retriever

logger = get_logger("memory.long_term")

#: RRF (Reciprocal Rank Fusion) の平滑化定数の既定値。論文推奨の 60。
#: ``rag.rrf_k`` で上書きできる。
DEFAULT_RRF_K: int = 60

#: 語彙チャネルの役割を「希少 literal の引き当て」に限定する理由。
#:
#: 日本語トークナイズは文字 bi-gram なので、**ほぼ全ドキュメントが非ゼロの
#: BM25 スコアを持つ**。素の上位 N をそのまま RRF に入れると「2 チャネル両方に
#: 現れた」というだけで雑音が上位を占める (実測、10 件のコーパス: 本命 22.1 に
#: 対し **クエリ語を 1 文字も含まない** 3 件が横並びで 12.0 = 本命の 54%。RRF 後の
#: 上位 2 件はコサイン 0.208 / -0.037 で、コサイン 1.0 のチャンクが 4 位に落ちた)。
#:
#: スコア比による足切りはこの分布では効かず、効く値に上げるとコーパスが大きく
#: なった途端にチャネルごと黙って死ぬ。静的な絶対閾値が埋め込み差し替えで
#: 到達不能になる事故を 2 度起こしているので、同じ形は採らない。
#:
#: 代わりに **「クエリ中の希少語 (df がコーパスの 1% 未満) を実際に含むか」**
#: で採否を決める。これは df 比なのでコーパス規模にも言語にも依存せず、かつ
#: 語彙チャネルの本来の役割 — 型番・パス・エラーコード・固有名詞のような、
#: 密ベクトルが原理的に苦手な literal の引き当て — と完全に一致する。
#: 一般語だけのクエリでは語彙チャネルは何も足さない (そこは密ベクトルが得意)。


class LongTermMemory:
    """Layer 3: RAG ベクトル DB + タグ検索（LLM ゼロ）"""

    def __init__(self, vector_store: VectorStore):
        self.vectors = vector_store
        self.note_meta: dict[str, dict] = {}  # {note_id: meta_dict}
        #: 語彙チャネル。``set_bm25_retriever`` で注入され、``search_hybrid``
        #: が使う。``None`` ならベクトル単独 (従来動作) に縮退する。
        self._bm25: "BM25Retriever | None" = None
        self._rrf_k: int = DEFAULT_RRF_K

    def set_bm25_retriever(
        self, bm25: "BM25Retriever | None", *, rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        """語彙検索チャネル (BM25) を注入する。

        チャット応答経路の記憶検索は長らく **密ベクトル単独** で、BM25 は
        ``HybridRetriever`` (ベンチマーク専用) の中にしか居なかった。固有名詞・
        パス・型番・エラーコードのような literal は密ベクトルが最も苦手とする
        一方、BM25 では必ず立つ。索引はどのみち起動時に構築済みなので、
        使わない理由が無い。
        """
        self._bm25 = bm25
        self._rrf_k = max(1, int(rrf_k))

    def chunk_source(self, chunk_id: str) -> str | None:
        """チャンクの発話者 (``user`` / ``assistant`` / ``rag`` / ``system``)。

        ``note_meta`` (プロセス内) を先に見て、無ければベクトルストアの
        ``metadata.json`` を見る。前者は再起動で消えるため、**永続化されている
        後者が本命**。どちらにも無ければ ``None`` (ドキュメント由来のチャンクと、
        speaker を記録する前に取り込まれた古いチャンク)。
        """
        meta = self.note_meta.get(chunk_id) or {}
        if meta.get("source"):
            return meta["source"]
        try:
            for entry in getattr(self.vectors, "metadata", []) or []:
                if entry.get("id") == chunk_id:
                    return entry.get("speaker")
        except Exception:
            return None
        return None

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 5,
        tag_filter: list[str] | None = None,
    ) -> list[tuple[str, float, str]]:
        """ベクトル検索 + オプショナルタグフィルタ

        Returns:
            list of (chunk_id, score, chunk_text)
        """
        candidates = self.vectors.search(query_vec, top_k=top_k * 2)
        logger.debug("LTM search: %d candidates from vector store (fetch_k=%d)", len(candidates), top_k * 2)

        # 問いだけのチャンクは答えを含まないので、検索で当たっても参考情報に
        # ならない。しかも **想起クエリでは必ず最上位に来る** — 過去に同じ質問を
        # していれば意味的に最も近いのはその質問文そのものだから。
        #
        # 実測 (2026-08-09): 「私の趣味は何でしたか？」の LTM 上位 1 位が
        # 0.541「私の趣味は何でしたか？」(過去の同一質問)、2 位が 0.449
        # 「私の趣味はゴルフでしたよね？」。答えを持つチャンクは 3 位以下へ
        # 押し下げられていた。実ストア 272 チャンク中 50 件 (18%) が該当する。
        #
        # 同じ判定は注入層 (MemoryInjector) がファクト / STM ノートに既に
        # 掛けているが、LTM 経路だけ素通りだった。書込側 (absorb_from_short_term)
        # でも弾くが、既存チャンクは reindex しないと消えないため読込時にも掛ける
        # (2026-08-07 の教訓: ゲートは採用時と読込時の両方に要る)。
        candidates = self._drop_question_only(candidates)

        candidates = self._apply_tag_filter(candidates, tag_filter)
        return candidates[:top_k]

    def _apply_tag_filter(
        self,
        candidates: list[tuple[str, float, str]],
        tag_filter: list[str] | None,
    ) -> list[tuple[str, float, str]]:
        """タグフィルタ (指定が無ければ素通し)。"""
        if not tag_filter:
            return candidates
        before = len(candidates)
        filtered = [
            c for c in candidates
            if any(t in self.note_meta.get(c[0], {}).get("tags", []) for t in tag_filter)
        ]
        logger.debug(
            "LTM search: tag filter %s reduced %d -> %d",
            tag_filter, before, len(filtered),
        )
        return filtered

    def _drop_question_only(
        self, candidates: list[tuple[str, float, str]],
    ) -> list[tuple[str, float, str]]:
        """問いだけのチャンクを落とす (``search`` 内コメントの判定を共有)。"""
        before = len(candidates)
        kept = [c for c in candidates if not carries_no_assertion(c[2])]
        if len(kept) != before:
            logger.debug(
                "LTM search: assertion gate dropped %d question-only chunks",
                before - len(kept),
            )
        return kept

    def _lexical_channel(
        self, query_text: str, fetch_k: int,
    ) -> tuple[list[tuple[str, float]], frozenset[str]]:
        """語彙チャネルのヒットと、その根拠になった希少語アンカー集合。

        BM25 の生の上位ではなく **クエリ中の希少語を実際に含むもの** だけを返す
        (理由は本モジュール冒頭の解説を参照)。
        """
        if self._bm25 is None:
            return [], frozenset()
        hits = self._bm25.search(query_text, top_k=fetch_k)
        if not hits:
            return [], frozenset()
        anchors = self._bm25.lexical_anchors(query_text, [cid for cid, _ in hits])
        kept = [(cid, sc) for cid, sc in hits if cid in anchors]
        if len(kept) != len(hits):
            logger.debug(
                "LTM hybrid: lexical channel kept %d/%d hits "
                "(dropped incidental n-gram matches)", len(kept), len(hits),
            )
        return kept, anchors

    def search_hybrid(
        self,
        query_vec: np.ndarray,
        query_text: str,
        top_k: int = 5,
        tag_filter: list[str] | None = None,
    ) -> tuple[list[tuple[str, float, str]], frozenset[str]]:
        """ベクトル + BM25 の RRF ハイブリッド検索。

        **スコアは常に素のコサイン**を返す。順位付けだけを RRF で決める。
        品質判定 (Step 5) / 関連度フロア (Step 6.5) / content gate (Step 4.5) は
        すべて cosine スケール前提で閾値が決まっているため、RRF スコア
        (≒0.016) をそのまま流すと閾値の意味が壊れる。STM 層が「順位付け用
        (combined) とゲート用 (素の cosine)」を分けているのと同じ考え方。

        密ベクトル単独の弱点は **閾値ではなく打ち切り**にある。LTM は
        ``top_k*2`` 件しか候補にせず ``top_k`` 件しか返さないため、コサインが
        十分高くても順位が 6 位以下の関連チャンクは一度も注入されない。
        BM25 チャネルを足すと、語彙的に強く一致するチャンクが候補集合へ
        入り直す。入った後は **自分の本当のコサイン**でゲートを通るので、
        既存の閾値較正はそのまま効く。

        Returns:
            ``(結果リスト, 語彙アンカーの chunk_id 集合)``。後者は「クエリ中の
            希少トークンを実際に含む」チャンクで、呼出側がコサインのフロアを
            免除する根拠に使う (:mod:`search_pipeline` Step 6.5)。
        """
        fetch_k = max(1, top_k) * 2
        vec_hits = self.vectors.search(query_vec, top_k=fetch_k)
        if self._bm25 is None or not (query_text or "").strip():
            candidates = self._apply_tag_filter(
                self._drop_question_only(vec_hits), tag_filter,
            )
            return candidates[:top_k], frozenset()

        try:
            lex_hits, anchor_ids = self._lexical_channel(query_text, fetch_k)
        except Exception as e:  # 索引不整合等は語彙チャネルを落として続行
            logger.warning("LTM BM25 search failed (vector only): %s", e)
            candidates = self._apply_tag_filter(
                self._drop_question_only(vec_hits), tag_filter,
            )
            return candidates[:top_k], frozenset()

        texts: dict[str, str] = {cid: text for cid, _, text in vec_hits}
        scores: dict[str, float] = {cid: sc for cid, sc, _ in vec_hits}

        # ベクトル側に居なかった語彙ヒットへ、**本当のコサイン**を与える。
        missing = [cid for cid, _ in lex_hits if cid not in scores]
        if missing:
            resolved = self.vectors.similarity_for(query_vec, missing)
            for cid in missing:
                if cid not in resolved:
                    continue
                scores[cid] = resolved[cid]
                texts[cid] = self.vectors.load_chunk(cid)

        vec_rank = {cid: i for i, (cid, _, _) in enumerate(vec_hits)}
        lex_rank = {cid: i for i, (cid, _) in enumerate(lex_hits) if cid in scores}

        k = self._rrf_k
        fused = sorted(
            scores.keys(),
            key=lambda cid: -(
                (1.0 / (k + vec_rank[cid] + 1) if cid in vec_rank else 0.0)
                + (1.0 / (k + lex_rank[cid] + 1) if cid in lex_rank else 0.0)
            ),
        )
        candidates = [(cid, scores[cid], texts.get(cid, "")) for cid in fused]
        candidates = self._apply_tag_filter(
            self._drop_question_only(candidates), tag_filter,
        )
        selected = candidates[:top_k]

        anchors = frozenset(
            cid for cid, _, _ in candidates if cid in anchor_ids
        )
        logger.debug(
            "LTM hybrid: vec=%d lex=%d fused=%d -> %d (lexical anchors=%d)",
            len(vec_hits), len(lex_hits), len(fused), len(selected), len(anchors),
        )
        return selected, anchors

    def absorb_from_short_term(self, note) -> str | None:
        """STM ノートを LTM に昇格（ベクトル DB に追加）"""
        if note.embedding is None:
            logger.warning("Note %s has no embedding, skipping LTM absorption", note.id)
            return None

        # 問いだけのノートは LTM へ入れない (``search`` の読込時ゲートと対)。
        # 入れても検索ノイズにしかならず、想起クエリでは最上位を占める。
        if carries_no_assertion(getattr(note, "content", "") or ""):
            logger.debug(
                "Note %s carries no assertion, skipping LTM absorption", note.id,
            )
            return None

        # 次元不一致ノートはスキップ
        stored_dim = self.vectors.stored_dim()
        note_dim = int(note.embedding.shape[0])
        if stored_dim is not None and stored_dim != note_dim:
            logger.warning(
                "Note %s embedding dim (%d) != store dim (%d), "
                "skipping LTM absorption. Run 'evoref reindex'.",
                note.id, note_dim, stored_dim,
            )
            return None

        vec = note.embedding.reshape(1, -1)
        # memory ノートは自己完結したチャンクで contextual prefix の価値が薄く、
        # source text も保存しないため has_context=True で登録し、sleep-time
        # Step 5.8 の「source text not found」空振りスキャンから除外する。
        chunk_ids = self.vectors.add_vectors(
            vec,
            [note.content],
            source=f"memory:{note.session_id}",
            category="memory",
            has_context=True,
            # 発話者は下の note_meta にも入れるが、あちらはプロセス内メモリのみで
            # 永続化されない。再起動後に読めるよう metadata.json 側にも残す。
            speaker=getattr(note, "source", None),
        )

        chunk_id = chunk_ids[0]
        # EvorefMem: trace_id を note_meta に保存し
        # ファクトと episodic LTM チャンクを ``trace_id`` で連結可能にする。
        self.note_meta[chunk_id] = {
            "note_id": note.id,
            "tags": note.tags,
            "keywords": note.keywords,
            "session_id": note.session_id,
            "trace_id": getattr(note, "trace_id", None),
            # 発話者 (user / assistant / rag / system)。STM ノートは全件持って
            # いるのに LTM へ移す時に落ちており、検索結果では「ユーザーが述べた
            # 事実」と「アシスタント自身が過去に答えた内容」が区別できなかった。
            #
            # 実測 (2026-08-09): STM 99 件の内訳は user 49 / assistant 50 で、
            # **半分が自分の出力**。「私の趣味は？」の LTM 上位には自分の過去の
            # 回答が並び、古い値 (0.444) が新しい値 (0.438) を上回っていた。
            # まず素性を残す — 検索側でどう扱うかは、これが観測できるように
            # なってから実測で決める。
            "source": getattr(note, "source", None),
        }
        logger.info("Absorbed note %s to LTM as chunk %s", note.id, chunk_id)
        return chunk_id
