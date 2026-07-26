"""Sleep-time update ワーカー（Level 0.5）

Trigger A (Light): LLM なし、応答後即座に実行（Steps 1-5）
Trigger B (Full):  LLM あり、アイドル後に実行（Steps 1-10）
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from backend.log_config import get_logger
from backend.utils import utc_now
from backend.free.memory.stores.short_term import ShortTermMemory
from backend.free.memory.stores.long_term import LongTermMemory
from backend.free.memory.pipeline.lightmem_scorer import FadeMemScorer, MemoryEviction
from backend.free.rag.bm25_retriever import BM25Retriever

if TYPE_CHECKING:
    from backend.free.agent.assist_prompt_manager import AssistPromptManager
    from backend.free.llm.assist_client import AssistModelClient
    from backend.free.memory.pipeline.rag_judge_assist_log import RagJudgeAssistLog
    from backend.free.memory.semantic.store import SemanticFactStore
    from backend.free.memory.notes.subject_canonicalizer import SubjectCanonicalizer
    from backend.free.rag.embedding_backend import EmbeddingBackend
    from backend.free.rag.vector_store import VectorStore
    from backend.free.rag.cartridge_manager import CartridgeManager

SemanticStoreProvider = Callable[[str], "SemanticFactStore"]
"""``scope`` 文字列を受けて ``SemanticFactStore`` を返すコールバック型。
通常は ``AppState.get_semantic_store`` をバインドして渡す。"""

SemanticStoreInvalidator = Callable[[str], None]
"""``scope`` 文字列を受けてキャッシュ済 ``SemanticFactStore`` を破棄する
コールバック型。Step 10 のアーカイブ後に AppState 側の
キャッシュをクリアするため。"""

logger = get_logger("memory.sleep_update")

# STM ノートを埋め込む際の本文上限 (文字数)。llama-server の embed インスタンスは
# n_ctx_slot が 2048〜4096 程度で運用されるため、超過すると
# `input (N tokens) is larger than the max context size` で **400** が返り、
# バッチ全体が失敗する (2026-07-25 の恒久デッドロックの起点)。
# 日本語は 1 文字 ≒ 1 トークン強なので、最小構成 (2048) でも収まる値に取る。
# ノートの先頭側に主題が来るため単純な前方切り出しで足りる。
_EMBED_MAX_CHARS = 1500
# 1 リクエストあたりのノート数。失敗時の巻き添えを小さくする。
_EMBED_BATCH_SIZE = 16
# 同一ノートの埋め込み連続失敗をこの回数まで許容し、超えたら以降スキップする
# (毎サイクル同じノートで失敗し続けて Step 1 が前進しなくなるのを防ぐ)。
_EMBED_MAX_FAILURES = 3


def _truncate_for_embedding(text: str) -> str:
    """埋め込み入力を ``_EMBED_MAX_CHARS`` で切り詰める。"""
    if len(text) <= _EMBED_MAX_CHARS:
        return text
    return text[:_EMBED_MAX_CHARS]


class SleepTimeWorker:
    """Sleep-time update の実行ワーカー"""

    def __init__(
        self,
        short_term: ShortTermMemory,
        long_term: LongTermMemory,
        embedder: EmbeddingBackend,
        scorer: FadeMemScorer,
        config: dict,
        experience_buf=None,
        debug_logger=None,
        learned_patterns=None,
        vector_store: VectorStore | None = None,
        cartridge_manager: CartridgeManager | None = None,
        policy=None,
        assist_prompt_manager: AssistPromptManager | None = None,
        semantic_store_provider: SemanticStoreProvider | None = None,
        current_project_id: str | None = None,
        agent_trace_dir: Path | None = None,
        subject_canonicalizer: SubjectCanonicalizer | None = None,
        semantic_store_invalidator: SemanticStoreInvalidator | None = None,
        assist_client: AssistModelClient | None = None,
        profile_id: str = "default",
        rag_judge_assist_log: "RagJudgeAssistLog | None" = None,
    ):
        self.short_term = short_term
        self.long_term = long_term
        self.embedder = embedder
        self.scorer = scorer
        self.config = config
        self.experience_buf = experience_buf
        self._policy = policy
        self._debug_logger = debug_logger
        self.learned_patterns = learned_patterns
        self.vector_store = vector_store
        self.cartridge_manager = cartridge_manager
        self._assist_prompt_manager = assist_prompt_manager
        self._bm25_retriever: BM25Retriever | None = None
        self._cancelled = False
        # ── Step 8 (Extractor) 用 ──
        self._semantic_store_provider = semantic_store_provider
        self._current_project_id = current_project_id
        self._agent_trace_dir = (
            Path(agent_trace_dir) if agent_trace_dir is not None else None
        )
        self._subject_canonicalizer = subject_canonicalizer
        # ── Step 10 で アーカイブ後にキャッシュ破棄するための callback ──
        self._semantic_store_invalidator = semantic_store_invalidator
        # ── Step 8.5 (URL curator) 用 ──
        self._assist_client = assist_client
        self._profile_id = profile_id
        # ── Step 8.7 (RAG judge curator) 用 ──
        self._rag_judge_assist_log = rag_judge_assist_log
        # MDPTraceExtractor はプロセス内で episode の二重抽出を防ぐため
        # ワーカー側で 1 インスタンスを保持する。
        self._mdp_trace_extractor = None
        # ── MDPIngester (agent_trace*.jsonl → episodic LTM) ──
        # log_dir は debug_logger の出力先を流用 (agent_trace_dir そのもの)。
        # state ファイルはメモリディレクトリ配下に置く想定だが、テスト容易性
        # のため lazy 初期化する。
        self._mdp_ingester = None

    def set_bm25_retriever(self, bm25: BM25Retriever) -> None:
        """BM25Retriever を設定（プレフィックス生成後の再構築に使用）"""
        self._bm25_retriever = bm25

    def cancel(self) -> None:
        """実行中の処理をキャンセル（現在のステップ完了後に停止）"""
        self._cancelled = True

    def _check_cancelled(self) -> bool:
        """キャンセルチェック"""
        if self._cancelled:
            logger.info("Sleep-time update cancelled by user input")
            return True
        return False

    async def run_light(self) -> dict:
        """Light 版: LLM なし、Steps 1-5

        Returns:
            実行結果サマリ dict
        """
        self._cancelled = False
        started_at = utc_now()
        t0 = time.monotonic()
        step_durations: dict[str, float] = {}
        result = {
            "embedded": 0,
            "tags_refined": 0,
            "scores_updated": 0,
            "evicted": 0,
        }

        logger.info("Sleep-time Light started (%d notes)", len(self.short_term.notes))

        # Step 1 が例外で抜けると Step 2-5.5 と _save_state() がまるごと飛び、
        # 経験バッファ / STM / LTM / learned_patterns が一切永続化されないまま
        # 次サイクルも同じ理由で落ち続ける (2026-07-25: 過大ノート 1 件の
        # embed 400 で 26 分間・17 回連続失敗し、未埋め込みノートが 2→84 件へ
        # 単調増加した。除去できるのは Step 4 eviction だが Step 1 で落ちるため
        # 永久に到達しないデッドロック)。
        # 各 step を独立させ、`_save_state()` は finally で必ず走らせる。
        try:
            # Step 1: 未埋め込みノートの埋め込み生成
            ts = time.monotonic()
            result["embedded"] = await self._step1_embed_notes()
            step_durations["step1_embedding"] = round(time.monotonic() - ts, 3)
            if self._check_cancelled():
                return result

            # Step 2: タグ補完（ルールベース）
            ts = time.monotonic()
            result["tags_refined"] = self._step2_refine_tags()
            step_durations["step2_tagging"] = round(time.monotonic() - ts, 3)
            if self._check_cancelled():
                return result

            # Step 3: LightMem スコア再計算
            ts = time.monotonic()
            result["scores_updated"] = self._step3_recalc_scores()
            step_durations["step3_scoring"] = round(time.monotonic() - ts, 3)
            if self._check_cancelled():
                return result

            # Step 4: FadeMem eviction 判定
            ts = time.monotonic()
            result["evicted"] = self._step4_eviction()
            step_durations["step4_eviction"] = round(time.monotonic() - ts, 3)
            if self._check_cancelled():
                return result

            # Step 5.5: 学習済みパターンの減衰と永続化
            ts = time.monotonic()
            result["patterns_decayed"] = self._step5_5_decay_patterns()
            step_durations["step5_5_patterns"] = round(time.monotonic() - ts, 3)
        finally:
            # 永続化 (途中で落ちても、それまでの進捗は必ず書き出す)
            self._save_state()

        elapsed = round(time.monotonic() - t0, 3)
        logger.info("Sleep-time Light completed in %.3fs: %s", elapsed, result)

        # DebugLogger に Level 0.5 Light を記録
        dl = self._debug_logger
        if dl:
            dl.log_learning_cycle(cycle_num=0, data={
                "level": "0.5-light",
                "started_at": started_at,
                "elapsed_sec": elapsed,
                "step_durations_sec": step_durations,
                "notes_count": len(self.short_term.notes),
                **result,
            })
            dl.log_outcome(
                kind="learning_cycle_l05",
                success=True,
                duration_ms=elapsed * 1000,
                quality_signals={
                    "level": "0.5-light",
                    "notes_count": len(self.short_term.notes),
                    **{k: v for k, v in result.items() if isinstance(v, (int, float, bool))},
                },
            )

        return result

    async def _step6_resolve_conflicts(
        self, llm_client, params_b: float, result: dict,
    ) -> None:
        """Step 6: ConflictResolver による短期記憶のコンフリクト解決。

        ShortTermMemory (FadeMem) の競合解決後に SemanticFactStore 上の
        コンフリクト解消 を続けて実行する
        """
        from backend.free.memory.pipeline.conflict_resolver import ConflictResolver
        resolver = ConflictResolver(
            self.config, params_b=params_b,
            policy=getattr(self, "_policy", None),
            debug_logger=self._debug_logger,
        )
        result["conflicts_resolved"] = await resolver.resolve_conflicts(
            self.short_term, llm_client,
        )
        # ── SemMem 競合解消 ──
        sem_summary = self._step6b_resolve_semmem_conflicts()
        if sem_summary is not None:
            result["semmem_conflicts"] = sem_summary

    def _step6b_resolve_semmem_conflicts(self) -> dict[str, int] | None:
        """Step 6 後段: SemanticFactStore 内の競合を検出・解消する。

        ``semantic_store_provider`` が未設定の場合は no-op (テスト等)。
        ``global`` ストアと、``current_project_id`` がセットされていれば
        その project ストアの 2 系統に対して順次実行する。
        """
        provider = self._semantic_store_provider
        if provider is None:
            return None
        from backend.free.memory.pipeline.semantic_conflict_resolver import (
            resolve_semmem_conflicts,
        )

        stores = []
        try:
            stores.append(provider("global"))
        except Exception as exc:
            logger.warning("Failed to open global semantic store: %s", exc)
        if self._current_project_id:
            try:
                stores.append(
                    provider(f"project:{self._current_project_id}"),
                )
            except Exception as exc:
                logger.warning(
                    "Failed to open project semantic store %s: %s",
                    self._current_project_id, exc,
                )
        if not stores:
            return None
        return resolve_semmem_conflicts(stores, self.config)

    async def _step7_evolve_notes(
        self, llm_client, params_b: float, result: dict,
    ) -> None:
        """Step 7: NoteEvolver による A-MEM ノート進化。

        ``context_description`` 生成の前に
        ``rebuild_links_and_clusters`` を呼び、リンクとクラスタ ID を更新する。
        これにより LLM 文脈生成は事前構築されたリンクを参照できる。
        """
        from backend.free.memory.notes.note_evolver import NoteEvolver
        evolver = NoteEvolver(
            self.config,
            params_b=params_b,
            assist_prompt_manager=self._assist_prompt_manager,
            debug_logger=self._debug_logger,
        )
        # リンク張り直し + クラスタリング (LLM 不要)
        link_stats = evolver.rebuild_links_and_clusters(self.short_term)
        result["notes_links_rebuilt"] = link_stats.get("links", 0)
        result["notes_clusters"] = link_stats.get("clusters", 0)
        result["notes_evolved"] = await evolver.evolve_notes(
            self.short_term, self.long_term, llm_client,
        )

    async def _step6_5_reembed_after_conflicts(
        self, result: dict,
    ) -> None:
        """Step 6.5: コンフリクト解決で embedding=None になったノートを再埋め込み。"""
        re_embedded = await self._step1_embed_notes()
        if re_embedded:
            result["re_embedded"] = re_embedded
            logger.info(
                "Re-embedded %d conflict-resolved notes", re_embedded,
            )

    def _log_full_completion(
        self,
        result: dict,
        started_at: str,
        elapsed: float,
        step_durations: dict[str, float],
    ) -> None:
        """Full サイクル完了の info ログ + DebugLogger 記録。"""
        logger.info("Sleep-time Full completed in %.3fs: %s", elapsed, result)
        dl = self._debug_logger
        if dl:
            dl.log_learning_cycle(cycle_num=0, data={
                "level": "0.5-full",
                "started_at": started_at,
                "elapsed_sec": elapsed,
                "step_durations_sec": step_durations,
                "notes_count": len(self.short_term.notes),
                **result,
            })
            dl.log_outcome(
                kind="learning_cycle_l05_full",
                success=True,
                duration_ms=elapsed * 1000,
                quality_signals={
                    "level": "0.5-full",
                    "notes_count": len(self.short_term.notes),
                    **{k: v for k, v in result.items() if isinstance(v, (int, float, bool))},
                },
            )

    async def run_full(self, llm_client=None) -> dict:
        """Full 版: LLM あり、Steps 1-10

        Args:
            llm_client: LLM クライアント（Steps 6-10 で使用）。
                        AssistModelClient（推奨）または LocalClient。
                        設計書 §5.5.2 に基づき、アシストモデルの使用を推奨。

        Returns:
            実行結果サマリ dict
        """
        started_at = utc_now()
        t0 = time.monotonic()
        step_durations: dict[str, float] = {}

        # まず Light 版を実行
        ts = time.monotonic()
        result = await self.run_light()
        step_durations["light_total"] = round(time.monotonic() - ts, 3)
        if self._check_cancelled():
            return result

        # Steps 5.8-10: LLM あり版
        if llm_client is None:
            logger.warning(
                "No LLM client for Full sleep-time update, skipping steps 5.8-10",
            )
            return result

        # Step 5.8: Contextual Retrieval プレフィックス生成
        ts = time.monotonic()
        result["contextual_prefixes"] = await self._step5_8_contextual_prefixes(llm_client)
        step_durations["step5_8_contextual"] = round(time.monotonic() - ts, 3)
        if self._check_cancelled():
            return result

        params_b = getattr(getattr(llm_client, "metadata", None), "params_b", 7.0)

        # Step 6: コンフリクト解決
        ts = time.monotonic()
        await self._step6_resolve_conflicts(llm_client, params_b, result)
        step_durations["step6_conflict"] = round(time.monotonic() - ts, 3)
        if self._check_cancelled():
            return result

        # Step 6.5: コンフリクト解決で消えた embedding を再生成
        ts = time.monotonic()
        await self._step6_5_reembed_after_conflicts(result)
        step_durations["step6_5_reembed"] = round(time.monotonic() - ts, 3)
        if self._check_cancelled():
            return result

        # Step 7: A-MEM ノート進化
        ts = time.monotonic()
        await self._step7_evolve_notes(llm_client, params_b, result)
        step_durations["step7_evolution"] = round(time.monotonic() - ts, 3)
        if self._check_cancelled():
            return result

        # Step 7.5: MDP トレース → episodic LTM 投入
        # B-1 の Step 8 (extractor) よりも前に行うことで、当該 trace_id の
        # MemoryNote が STM/LTM に存在する状態でファクト抽出が走り、
        # ``trace_id`` がエピソード記憶 / 意味記憶の双方に伝播する。
        ts = time.monotonic()
        result["mdp_traces_ingested"] = await self._step7_5_ingest_mdp_traces()
        step_durations["step7_5_mdp_ingest"] = round(time.monotonic() - ts, 3)
        if self._check_cancelled():
            return result

        # Step 8: SemanticFact Extractor
        # 既存 _step8_9_summarize_sessions は Step 9 に再配置される想定。
        # メソッド名を変えずに前段に Step 8 を挿入する形で共存させる。
        ts = time.monotonic()
        result["facts_extracted"] = self._step8_extract_facts()
        step_durations["step8_extract_facts"] = round(time.monotonic() - ts, 3)
        if self._check_cancelled():
            return result

        # Step 8.5: URL リコール用 world_fact のキュレーション
        # CLAUDE.md §6 #2 に従い、SemMem 書込はここに閉じる。
        # アシスト未接続 (degraded) の場合は no-op で通過する。
        ts = time.monotonic()
        result["url_facts_curated"] = await self._step8_5_curate_urls(
            llm_client,
        )
        step_durations["step8_5_url_curator"] = round(time.monotonic() - ts, 3)
        if self._check_cancelled():
            return result

        # Step 8.6: executable command リコール用 world_fact のキュレーション
        # run_command 成功ターン (MemoryNote.tool_command) を world_fact 化する。
        # アシスト採点不要なので degraded でも動作する。SemMem 書込はここに閉じる。
        ts = time.monotonic()
        result["command_facts_curated"] = await self._step8_6_curate_commands()
        step_durations["step8_6_command_curator"] = round(time.monotonic() - ts, 3)
        if self._check_cancelled():
            return result

        # Step 8.7: RAG necessity/quality リコール用 world_fact のキュレーション
        # チャット応答パスで RagJudgeAssistLog に蓄積された判定を自己採点して
        # world_fact 化する。CLAUDE.md §6 #2 に従い、SemMem 書込はここに閉じる。
        # アシスト未接続 (degraded) の場合は no-op で通過する。
        ts = time.monotonic()
        result["rag_judge_facts_curated"] = await self._step8_7_curate_rag_judge_facts(
            llm_client,
        )
        step_durations["step8_7_rag_judge_curator"] = round(time.monotonic() - ts, 3)
        if self._check_cancelled():
            return result

        # Step 13: failure_pattern 統合
        # Step 8 の直後に呼ぶことで、当該イテレーションで新たに抽出された
        # failure_pattern と、既に loop.write_failure_note で即時書き込みされた
        # 失敗レコードを同一 signature でマージする。
        ts = time.monotonic()
        result["failure_patterns_consolidated"] = (
            self._step13_consolidate_failure_patterns()
        )
        step_durations["step13_failure_consolidation"] = round(
            time.monotonic() - ts, 3,
        )
        if self._check_cancelled():
            return result

        # Step 8-9: 未要約セッションの要約生成 + 埋め込み
        ts = time.monotonic()
        result["summaries_generated"] = await self._step8_9_summarize_sessions(llm_client)
        step_durations["step8_9_summarize"] = round(time.monotonic() - ts, 3)
        if self._check_cancelled():
            return result

        # Step 9: 履歴要約を SemMem に decision/commitment として昇格
        ts = time.monotonic()
        result["semmem_promoted"] = self._step9_promote_summaries_to_semmem()
        step_durations["step9_promote"] = round(time.monotonic() - ts, 3)
        if self._check_cancelled():
            return result

        # Step 9 GC: semmem_limits 超過時に lowest_score 戦略で削除
        ts = time.monotonic()
        result["semmem_gc_deleted"] = self._step9_run_semmem_gc()
        step_durations["step9_gc"] = round(time.monotonic() - ts, 3)
        if self._check_cancelled():
            return result

        # Step 10: 180 日無アクセスのプロジェクトをアーカイブ
        ts = time.monotonic()
        result["archived_projects"] = self._step10_archive_inactive_projects()
        step_durations["step10_archive"] = round(time.monotonic() - ts, 3)
        if self._check_cancelled():
            return result

        # Step 10b (補助): 履歴ファイル本体の圧縮処理
        ts = time.monotonic()
        result["compressed"] = self._step10_compact_sessions()
        step_durations["step10b_history_compact"] = round(time.monotonic() - ts, 3)

        # 永続化 + 完了ログ
        self._save_state()
        elapsed = round(time.monotonic() - t0, 3)
        self._log_full_completion(result, started_at, elapsed, step_durations)
        return result

    async def _step1_embed_notes(self) -> int:
        """Step 1: 未埋め込みノートの埋め込み生成

        1 件でも embed サーバの context を超えるノートがあるとバッチ全体が
        400 で落ち、Step 2 以降と永続化がすべて飛ぶ (2026-07-25 のデッドロック)。
        対策は 3 段:

        1. ``_EMBED_MAX_CHARS`` でノート本文を切り詰めてから投げる (根本対策)
        2. バッチ失敗時はノート単位へフォールバックし、健全なノートを救う
        3. 連続失敗したノートは ``embed_failures`` を加算し、上限に達したら
           以降スキップする (同じノートで永久に再試行しない)
        """
        unembedded = [
            note for note in self.short_term.notes.values()
            if note.embedding is None
            and note.embed_failures < _EMBED_MAX_FAILURES
        ]
        if not unembedded:
            logger.debug("Step 1: no unembedded notes, skipping")
            return 0

        logger.info("Step 1: embedding %d notes...", len(unembedded))
        embedded = 0
        for start in range(0, len(unembedded), _EMBED_BATCH_SIZE):
            batch = unembedded[start:start + _EMBED_BATCH_SIZE]
            texts = [_truncate_for_embedding(n.content) for n in batch]
            try:
                embeddings = await self.embedder.embed(texts, is_query=False)
            except Exception as exc:
                logger.warning(
                    "Step 1: batch embed failed (%d notes), "
                    "falling back to per-note: %s", len(batch), exc,
                )
                embedded += await self._embed_notes_individually(batch)
                continue
            for note, emb in zip(batch, embeddings):
                note.embedding = emb.astype(np.float32)
                note.embed_failures = 0
                embedded += 1

        self.short_term._cache_dirty = True
        logger.info("Step 1: embedded %d notes", embedded)
        return embedded

    async def _embed_notes_individually(self, notes: list) -> int:
        """バッチ失敗時のフォールバック。健全なノートだけ個別に埋め込む。"""
        embedded = 0
        for note in notes:
            try:
                emb = await self.embedder.embed(
                    [_truncate_for_embedding(note.content)], is_query=False,
                )
            except Exception as exc:
                note.embed_failures += 1
                logger.warning(
                    "Step 1: note embed failed (%d/%d, len=%d): %s",
                    note.embed_failures, _EMBED_MAX_FAILURES,
                    len(note.content), exc,
                )
                continue
            if emb is None or len(emb) == 0:
                note.embed_failures += 1
                continue
            note.embedding = emb[0].astype(np.float32)
            note.embed_failures = 0
            embedded += 1
        return embedded

    def _step2_refine_tags(self) -> int:
        """Step 2: タグ補完（ルールベース）

        埋め込みが生成されたノートについて、キーワードとタグを再抽出。
        """
        from backend.free.memory.notes.note_builder import NoteBuilder

        refined = 0
        for note in self.short_term.notes.values():
            if not note.tags:
                note.tags = NoteBuilder.auto_tag(note.content)
                if note.tags:
                    refined += 1
            if not note.keywords:
                note.keywords = NoteBuilder.extract_keywords(note.content)
                if note.keywords:
                    refined += 1

        if refined > 0:
            logger.info("Refined tags/keywords for %d notes", refined)
        return refined

    def _step3_recalc_scores(self) -> int:
        """Step 3: LightMem スコア再計算"""
        updated = 0
        for note in self.short_term.notes.values():
            new_score = self.scorer.compute(note)
            if abs(new_score - note.lightmem_score) > 0.01:
                note.lightmem_score = new_score
                updated += 1

        if updated > 0:
            self.short_term._cache_dirty = True
            logger.info("Updated scores for %d notes", updated)
        return updated

    def _step4_eviction(self) -> int:
        """Step 4: FadeMem eviction 判定"""
        logger.info("Step 4: running eviction check on %d notes", len(self.short_term.notes))
        eviction = MemoryEviction(policy=getattr(self, "_policy", None))
        exp_dict = None
        if self.experience_buf is not None:
            exp_dict = {
                "source_memory_ids": self.experience_buf.source_memory_ids,
                "pending_memory_ids": self.experience_buf.pending_memory_ids,
            }
        evicted = eviction.evict(
            self.short_term,
            self.long_term,
            exp_dict,
            self.scorer,
            self.config,
        )
        if evicted:
            logger.info("Step 4: evicted %d notes", evicted)
        else:
            logger.debug("Step 4: no notes evicted")
        return evicted


    def _step5_5_decay_patterns(self) -> int:
        """Step 5.5: 学習済みパターンの重み減衰と永続化

        sleep-time Light で毎回実行。LLM 不要。
        重みが閾値未満に低下したパターンは自動削除される。
        """
        if self.learned_patterns is None:
            return 0

        removed = self.learned_patterns.decay_all()

        # 永続化
        try:
            from backend.config import get_path_resolver
            resolver = get_path_resolver()
            patterns_file = resolver.resolve_local("learned_patterns_file")
            self.learned_patterns.save(patterns_file)
        except Exception as e:
            logger.warning("Failed to save learned patterns: %s", e)

        if removed:
            logger.info("Step 5.5: decayed %d patterns", removed)
        return removed

    async def _step5_8_contextual_prefixes(self, llm_client) -> int:
        """Step 5.8: Contextual Retrieval プレフィックス生成。

        実ロジックは
        :mod:`backend.free.memory.sleep.contextual` に分離された。
        本メソッドはメイン VectorStore / カートリッジ / BM25 を
        引数に詰め替える薄いラッパ。
        """
        from backend.free.memory.sleep.contextual import (
            generate_contextual_prefixes,
        )

        return await generate_contextual_prefixes(
            llm_client,
            config=self.config,
            embedder=self.embedder,
            vector_store=self.vector_store,
            cartridge_manager=self.cartridge_manager,
            bm25_retriever=self._bm25_retriever,
            is_cancelled=self._check_cancelled,
        )

    async def _step8_9_summarize_sessions(self, llm_client) -> int:
        """Step 8-9: 未要約セッションの要約生成 + 埋め込みベクトル生成。

        実ロジックは :mod:`backend.free.memory.sleep.summarize`
        に分離された。本メソッドは cancel 判定を渡す薄いラッパ。
        """
        from backend.free.memory.sleep.summarize import (
            summarize_unsummarized_sessions,
        )

        history_cfg = self.config.get("history") or {}
        return await summarize_unsummarized_sessions(
            llm_client,
            self.embedder,
            batch_size=int(history_cfg.get("summary_batch_size", 5)),
            is_cancelled=self._check_cancelled,
        )

    # ── Step 7.5 (MDP トレース → episodic LTM) ─────────

    async def _step7_5_ingest_mdp_traces(self) -> int:
        """Step 7.5: ``agent_trace*.jsonl`` を episodic LTM に取り込む。

        実ロジックは :mod:`backend.free.memory.sleep.mdp_ingest`
        に分離された。本メソッドは state を詰め替えて委譲する薄いラッパ。
        """
        from backend.free.memory.sleep.mdp_ingest import ingest_mdp_traces

        ingested, self._mdp_ingester = await ingest_mdp_traces(
            self.short_term,
            self.long_term,
            self.embedder,
            config=self.config,
            agent_trace_dir=self._agent_trace_dir,
            current_project_id=self._current_project_id,
            cached_ingester=self._mdp_ingester,
        )
        return ingested

    # ── Step 8 (Chat/Coding/MDP Extractor) ─────────────

    def _step8_extract_facts(self) -> int:
        """Step 8: SemanticFact 抽出

        実ロジックは :mod:`backend.free.memory.sleep.extraction`
        に分離された。本メソッドは state (短期記憶 / 設定 / provider / MDP
        キャッシュ) を引数に詰め替えて委譲する薄いラッパ。
        """
        from backend.free.memory.sleep.extraction import extract_semantic_facts

        notes = list(self.short_term.notes.values())
        total, self._mdp_trace_extractor = extract_semantic_facts(
            notes,
            config=self.config,
            store_provider=self._semantic_store_provider,
            current_project_id=self._current_project_id,
            agent_trace_dir=self._agent_trace_dir,
            subject_canonicalizer=self._subject_canonicalizer,
            mdp_trace_extractor=self._mdp_trace_extractor,
        )
        return total

    # ── Step 8.5 (URL curator) ─────────────────────────

    async def _step8_5_curate_urls(self, llm_client=None) -> int:
        """Step 8.5: URL リコール用の ``world_fact`` を sleep-time で書く。

        実ロジックは :mod:`backend.free.memory.sleep.url_curator`
        に分離されている。本メソッドは state を詰め替える薄いラッパ。

        ``self._assist_client`` がコンストラクタ時点で None だった場合
        (起動時 health check 失敗で degraded staged) でも、``run_full`` 経由で
        渡された ``llm_client`` が AssistModelClient なら lazy_connect 後の
        最新参照とみなしてそちらを使う。
        """
        from backend.free.llm.assist_client import AssistModelClient
        from backend.free.memory.sleep.url_curator import curate_url_facts

        assist_client = self._assist_client
        if assist_client is None and isinstance(llm_client, AssistModelClient):
            assist_client = llm_client
            # 以後の sleep-time でも使えるよう保持
            self._assist_client = assist_client

        notes = list(self.short_term.notes.values())
        return await curate_url_facts(
            notes,
            config=self.config,
            store_provider=self._semantic_store_provider,
            assist_client=assist_client,
            embedder=self.embedder,
            profile_id=self._profile_id,
            debug_logger=self._debug_logger,
        )

    async def _step8_6_curate_commands(self) -> int:
        """Step 8.6: executable command リコール用の ``world_fact`` を書く。

        実ロジックは
        :mod:`backend.free.memory.sleep.executable_command_curator` に分離。
        本メソッドは state を詰め替える薄いラッパ。url_curator と違い
        アシスト採点をしない (``MemoryNote.tool_command_success`` を使う) ため
        ``assist_client`` は渡さない。
        """
        from backend.free.memory.sleep.executable_command_curator import (
            curate_executable_command_facts,
        )

        notes = list(self.short_term.notes.values())
        return await curate_executable_command_facts(
            notes,
            config=self.config,
            store_provider=self._semantic_store_provider,
            embedder=self.embedder,
            profile_id=self._profile_id,
            debug_logger=self._debug_logger,
        )

    async def _step8_7_curate_rag_judge_facts(self, llm_client=None) -> int:
        """Step 8.7: RAG necessity/quality リコール用の ``world_fact`` を書く。

        実ロジックは :mod:`backend.free.memory.sleep.rag_judge_curator`
        に分離されている。本メソッドは state を詰め替える薄いラッパ
        (``_step8_5_curate_urls`` と同型)。
        """
        from backend.free.llm.assist_client import AssistModelClient
        from backend.free.memory.sleep.rag_judge_curator import curate_rag_judge_facts

        assist_client = self._assist_client
        if assist_client is None and isinstance(llm_client, AssistModelClient):
            assist_client = llm_client
            self._assist_client = assist_client

        if self._rag_judge_assist_log is None:
            return 0
        events = self._rag_judge_assist_log.drain()
        if not events:
            return 0

        notes = list(self.short_term.notes.values())
        return await curate_rag_judge_facts(
            events,
            notes,
            config=self.config,
            store_provider=self._semantic_store_provider,
            assist_client=assist_client,
            embedder=self.embedder,
            profile_id=self._profile_id,
            debug_logger=self._debug_logger,
        )

    # ── Step 13 (failure_pattern 統合) ─────────────────

    def _step13_consolidate_failure_patterns(self) -> dict[str, int]:
        """Step 13: 同一 ``failure_signature`` の failure_pattern を統合する。

        実ロジックは
        :mod:`backend.free.memory.sleep.failure_consolidator` に分離された。
        本メソッドは state を詰め替えて委譲する薄いラッパ。

        Returns:
            ``ConsolidationSummary.as_dict()`` 互換の dict。no-op 時は
            空 dict。
        """
        from backend.free.memory.sleep.failure_consolidator import (
            consolidate_failure_patterns_for_project,
        )

        return consolidate_failure_patterns_for_project(
            self._semantic_store_provider,
            config=self.config,
            current_project_id=self._current_project_id,
        )

    def _step10_compact_sessions(self) -> int:
        """Step 10b (補助): 保持ポリシーに基づく履歴圧縮処理。

        で正式 Step 10 はプロジェクトアーカイブに置き換わったが
        履歴の物理ストレージ削減のため本処理は補助ステップとして残す。
        """
        from backend.free.history.history_manager import get_history_manager

        try:
            mgr = get_history_manager()
            result = mgr.compact_sessions()
            total = result.get("compressed", 0) + result.get("summarized", 0) + result.get("deleted", 0)
            return total
        except Exception as e:
            logger.warning("Failed to compact sessions in step 10: %s", e)
            return 0

    # ── Step 9 (promotion + GC) + Step 10 (project archive) ──

    def _step9_promote_summaries_to_semmem(self) -> int:
        """Step 9: 古い history 要約を SemMem に decision/commitment として昇格

        実ロジックは
        :mod:`backend.free.memory.sleep.promotion` に分離された (D7 決定)。
        本メソッドは HistoryManager 初期化と provider/cancel を引数に詰め替える
        薄いラッパ。
        """
        from backend.free.memory.sleep.promotion import promote_history_to_semmem

        provider = self._semantic_store_provider
        if provider is None:
            logger.debug("Step 9 promotion: no semantic store provider, skipping")
            return 0
        try:
            from backend.free.history.history_manager import get_history_manager
            mgr = get_history_manager()
        except Exception as exc:
            logger.warning("Step 9 promotion: failed to init HistoryManager: %s", exc)
            return 0

        return promote_history_to_semmem(
            mgr,
            provider,
            current_project_id=self._current_project_id,
            is_cancelled=self._check_cancelled,
        )

    def _step9_run_semmem_gc(self) -> dict[str, int]:
        """Step 9 (GC): semmem_limits 超過時に lowest_score 戦略で削除する

        実ロジックは :mod:`backend.free.memory.sleep.gc`
        に分離された。本メソッドは state を詰め替える薄いラッパ。

        Returns:
            ``{scope:type: deleted_count}`` の dict。何も削除されなかった
            組合せは省略。
        """
        from backend.free.memory.sleep.gc import run_semmem_gc

        if self._semantic_store_provider is None:
            return {}
        return run_semmem_gc(
            self._semantic_store_provider,
            config=self.config,
            current_project_id=self._current_project_id,
        )

    def _step10_archive_inactive_projects(self) -> list[str]:
        """Step 10: 180 日無アクセスのプロジェクトを semantic/archive/ に移動

        実ロジックは :mod:`backend.free.memory.sleep.archive`
        に分離された。本メソッドは invalidator コールバックを詰め替える薄いラッパ。

        Returns:
            アーカイブ対象となったプロジェクト ID のリスト (state フラグ更新済)。
        """
        from backend.free.memory.sleep.archive import archive_inactive_projects

        return archive_inactive_projects(
            config=self.config,
            store_invalidator=self._semantic_store_invalidator,
        )

    def _save_state(self) -> None:
        """メモリ状態を永続化"""
        from backend.config import get_path_resolver

        try:
            resolver = get_path_resolver()
            memory_dir = resolver.resolve_local("memory_dir")
            self.short_term.save(memory_dir / "short_term_notes.json")
            logger.info("State saved to disk")
        except Exception as e:
            logger.warning("Failed to save state: %s", e)

        # 経験バッファを永続化 (起動時ロードと同じ resolve_learning でパーティション先に揃える)
        if self.experience_buf is not None:
            try:
                resolver = get_path_resolver()
                exp_file = resolver.resolve_learning("experience_file")
                self.experience_buf.save(exp_file)
            except Exception as e:
                logger.warning("Failed to save experience buffer: %s", e)

        # LTM ベクトルインデックスを永続化 (absorb_from_short_term は in-memory
        # 追加のみで save しないため、ここでまとめてディスクへ書き戻す)
        if self.vector_store is not None and self.vector_store.count > 0:
            try:
                self.vector_store.save()
                logger.info(
                    "Vector store (LTM) saved: %d vectors", self.vector_store.count,
                )
            except Exception as e:
                logger.warning("Vector store save on sleep failed: %s", e)
