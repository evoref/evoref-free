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
    from backend.free.agent.aux_prompt_manager import AuxPromptManager
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
        aux_prompt_manager: AuxPromptManager | None = None,
        semantic_store_provider: SemanticStoreProvider | None = None,
        current_project_id: str | None = None,
        agent_trace_dir: Path | None = None,
        subject_canonicalizer: SubjectCanonicalizer | None = None,
        semantic_store_invalidator: SemanticStoreInvalidator | None = None,
        profile_id: str = "default",
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
        self._aux_prompt_manager = aux_prompt_manager
        self._bm25_retriever: BM25Retriever | None = None
        self._fewshot_pool = None
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
        self._profile_id = profile_id
        # MDPTraceExtractor はプロセス内で episode の二重抽出を防ぐため
        # ワーカー側で 1 インスタンスを保持する。
        self._mdp_trace_extractor = None
        #: 最後に Step 8 (ファクト抽出) が走った時刻。Step 4 の eviction が
        #: 「まだ抽出器が見ていないノート」を落とさないための基準
        #: (:meth:`MemoryEviction.evict` の ``unextracted_cutoff``)。
        #: 0.0 は「まだ一度も走っていない」= 全ノートを保護する側に倒す。
        self._last_extraction_at: float = 0.0
        # ── MDPIngester (agent_trace*.jsonl → episodic LTM) ──
        # log_dir は debug_logger の出力先を流用 (agent_trace_dir そのもの)。
        # state ファイルはメモリディレクトリ配下に置く想定だが、テスト容易性
        # のため lazy 初期化する。
        self._mdp_ingester = None
        #: 「今チャット生成が走っているか」の判定 (scheduler が注入)。
        self._chat_in_flight_probe = None

    def set_bm25_retriever(self, bm25: BM25Retriever) -> None:
        """BM25Retriever を設定（プレフィックス生成後の再構築に使用）"""
        self._bm25_retriever = bm25

    def set_fewshot_pool(self, pool) -> None:
        """FewShotPool を設定 (手本の埋め込み backfill に使用)。"""
        self._fewshot_pool = pool

    def set_chat_in_flight(self, probe: "Callable[[], bool] | None") -> None:
        """「今チャット生成が走っているか」を返す判定を注入する。

        LLM を逐次に何度も叩くステップ (Step 7 のノート進化) が、ユーザーの
        ターンを待たせないための協調 yield に使う。``SleepTimeScheduler`` が
        ``set_worker`` 時に自分の ``_chat_in_flight`` を渡す。
        """
        self._chat_in_flight_probe = probe

    def _chat_in_flight(self) -> bool:
        """チャット生成が実行中か (未注入なら常に False = 従来どおり止まらない)。"""
        probe = getattr(self, "_chat_in_flight_probe", None)
        if probe is None:
            return False
        try:
            return bool(probe())
        except Exception:
            return False

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
            aux_prompt_manager=self._aux_prompt_manager,
            debug_logger=self._debug_logger,
        )
        # リンク張り直し + クラスタリング (LLM 不要)
        link_stats = evolver.rebuild_links_and_clusters(self.short_term)
        result["notes_links_rebuilt"] = link_stats.get("links", 0)
        result["notes_clusters"] = link_stats.get("clusters", 0)
        result["notes_evolved"] = await evolver.evolve_notes(
            self.short_term, self.long_term, llm_client,
            should_pause=self._chat_in_flight,
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
                        AuxClient（推奨）または LocalClient。
                        設計書 §5.5.2 に基づき、補助タスクの使用を推奨。

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

        # Step 8.4: Step 8 が型付けできなかった言明のキュレーション。
        # 日本語の断定は大半が「です」で終わり world_fact トリガに掛からず、
        # subject 側も ASCII 英字必須で日本語キーワードを弾くため、Step 8 では
        # 構造的に届かない (2026-08-19 ライブ監査)。命名だけ補助タスクへ出す。
        ts = time.monotonic()
        result["assertion_facts_curated"] = await self._step8_4_curate_assertions(
            llm_client,
        )
        step_durations["step8_4_assertion_curator"] = round(time.monotonic() - ts, 3)
        if self._check_cancelled():
            return result

        # Step 8.5: URL リコール用 world_fact のキュレーション
        # CLAUDE.md §6 #2 に従い、SemMem 書込はここに閉じる。
        # 補助タスク未接続 (degraded) の場合は no-op で通過する。
        ts = time.monotonic()
        result["url_facts_curated"] = await self._step8_5_curate_urls(
            llm_client,
        )
        step_durations["step8_5_url_curator"] = round(time.monotonic() - ts, 3)
        if self._check_cancelled():
            return result

        # Step 8.6: executable command リコール用 world_fact のキュレーション
        # run_command 成功ターン (MemoryNote.tool_command) を world_fact 化する。
        # 補助タスク採点不要なので degraded でも動作する。SemMem 書込はここに閉じる。
        ts = time.monotonic()
        result["command_facts_curated"] = await self._step8_6_curate_commands()
        step_durations["step8_6_command_curator"] = round(time.monotonic() - ts, 3)
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

        # Step 8.8: 埋め込みが無いファクトの遡及生成
        # Step 8 (extractor) / 8.4-8.6 (curator) / Step 9 (history 昇格) は
        # いずれも同期関数で embedder を持てないため、生成した直後にここで埋める。
        #
        # **Step 9 promotion より後**に置くこと。以前は 8.6 の直後にあり、
        # Step 9 が作った decision / commitment は embedding=None のまま
        # 次サイクルまで残っていた。その間 MemoryInjector._is_relevant の
        # 「埋め込みが無い候補は判定不能として通す」に該当し、関連度ゲートを
        # 素通りする (セッション要約は is_internal_index_subject が落とすが、
        # commitment とプロジェクト系 decision は落ちない)。
        ts = time.monotonic()
        result["fact_embeddings_backfilled"] = (
            await self._step8_8_backfill_fact_embeddings()
        )
        step_durations["step8_8_fact_embedding"] = round(time.monotonic() - ts, 3)
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

        # Step 10b-2: 手本 (few-shot) の埋め込みを遡って生成する。
        # 埋め込みが載るまでその手本は密ベクトル選択の候補にならない
        # (STM ノートが embed 工程を通るまで注入対象にならないのと同じ契約)。
        ts = time.monotonic()
        result["fewshot_embeddings_backfilled"] = (
            await self._step10b2_backfill_fewshot_embeddings()
        )
        step_durations["step10b2_fewshot_embedding"] = round(time.monotonic() - ts, 3)

        # Step 10c: 語彙索引 (BM25) をベクトルストアと同期させる。
        # チャット応答経路の LTM がこの索引を引くようになったため、昇格した
        # ばかりのチャンクが語彙検索から漏れたままになるのを防ぐ。
        # 以前の再構築点は contextual prefix 生成 (Step 5.8) の中だけで、
        # ``contextual_prefix.enabled=false`` の構成では **一度も更新されなかった**。
        ts = time.monotonic()
        result["lexical_index_rebuilt"] = self._step10c_refresh_lexical_index()
        step_durations["step10c_lexical_index"] = round(time.monotonic() - ts, 3)

        # Step 10d: 起動時に条件を満たさず見送られた閾値較正を拾い直す。
        # 較正は起動時 1 回きりだったため、ノートが少ない状態で起動すると
        # プロセスの生涯にわたり config の静的閾値 (別モデル前提で到達不能)
        # が使われ続けていた。ここまでで Step 1-8 がノートを増やしているので、
        # 同じサイクル内で増えた分をそのまま使える。
        ts = time.monotonic()
        result["threshold_calibrated"] = await self._step10d_retry_calibration()
        step_durations["step10d_threshold_calibration"] = round(
            time.monotonic() - ts, 3,
        )

        # 永続化 + 完了ログ
        self._save_state()
        elapsed = round(time.monotonic() - t0, 3)
        self._log_full_completion(result, started_at, elapsed, step_durations)
        return result

    async def _step10d_retry_calibration(self) -> bool:
        """未確定の記憶検索閾値較正を 1 回だけ再試行する (失敗は握る)。"""
        try:
            from backend.free.rag.memory_threshold_calibration import (
                retry_pending_calibration,
            )
            return await retry_pending_calibration()
        except Exception as e:
            logger.warning("Step 10d: threshold recalibration failed: %s", e)
            return False

    async def _step10b2_backfill_fewshot_embeddings(self) -> int:
        """手本プールの未埋め込みエントリを遡って埋め込む。

        起動時にも背景タスクで一度張るが (``_pillar_wirer``)、以降に採用された
        手本はここで拾う。埋め込みは永続化しない設計なので、件数は高々
        「前回サイクル以降の新規」に収まる。

        Returns:
            埋め込みを新たに付与した手本の数。
        """
        pool = self._fewshot_pool
        if pool is None or self.embedder is None:
            return 0
        backfill = getattr(pool, "backfill_embeddings", None)
        if backfill is None:
            return 0
        try:
            return await backfill(self.embedder)
        except Exception as e:
            logger.warning("Step 10b-2: fewshot embedding backfill failed: %s", e)
            return 0

    def _step10c_refresh_lexical_index(self) -> int:
        """BM25 索引をベクトルストアの現在のチャンク集合へ張り直す。

        件数が一致していれば no-op。再構築は O(N) のトークナイズなので、
        チャット応答をブロックしない sleep-time にだけ置く。

        Returns:
            張り直した場合はチャンク数、no-op なら 0。
        """
        bm25 = self._bm25_retriever
        vs = self.vector_store
        if bm25 is None or vs is None:
            return 0
        try:
            if bm25.count == vs.count:
                return 0
            from backend.free.rag.bm25_retriever import (
                build_index_from_vector_store,
            )

            n = build_index_from_vector_store(bm25, vs)
            logger.info(
                "Step 10c: lexical index rebuilt (%d -> %d chunks)",
                bm25.count if n == 0 else n, vs.count,
            )
            return n
        except Exception as e:
            logger.warning("Step 10c: lexical index rebuild failed: %s", e)
            return 0

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
                # 生成時と同じ source を渡す。渡さないと assistant ノートに
                # ``fact`` が付き直し、生成側の抑止 (ASSISTANT_EXCLUDED_TAGS) が
                # sleep-time で無効化されてしまう。
                note.tags = NoteBuilder.auto_tag(
                    note.content, getattr(note, "source", "user"),
                )
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

    def _extract_before_overflow(self) -> int:
        """未抽出ノートを **捨てる前に** Step 8 を前倒しする (背圧)。

        なぜ要るか: SemMem の唯一の入力は STM のノートで、抽出 (Step 8) は
        idle Full にしか載っていない (``memory.facts.trigger =
        idle_full_only``)。会話が続く限り Full は来ないので STM は伸び続け、
        ``UNEXTRACTED_PROTECTION_CEILING`` に当たった時点で保護が **丸ごと**
        外れ、下位 20% が古い順に降格される。落ちるのは「まだ抽出器が見て
        いない最も古いノート」— つまり **SemMem の入力そのもの**。

        実インシデント (2026-08-28 ライブ監査、20 テーマ 200 ターン):

        - Full は会話中ずっと保留された (``deferred=57.3 min``)
        - STM が 200 件に達するたび 40 件が降格され、これが 6 回起きた
          (08:41 / 08:46 / 08:51 / 08:59 / 09:05 / 09:12、計 237 件)
        - 降格された中にテーマ 1 の「私の名前は佐倉レンです。」があり、
          ``mem.personal.name`` のファクトは **最後まで 1 件も作られなかった**
          (居住地 / 職業 / 猫 / 飲み物 / 誕生日 / 会員番号 は全て作られている)
        - 結果「あなたの名前は確認できていません」と「佐倉レンさんですね」が
          問いの言い回し次第で入れ替わる (LTM / 履歴検索へのフォールバックは
          語順に依存するため)

        天井の意味を「抽出を諦める」から「抽出が遅れている — 今すぐ走らせる」
        へ反転させる。Step 8 は LLM 非依存の同期処理で、実測 590 秒の Full の
        中でも計上されない (< 0.5 秒) ため、この前倒しはほぼ無償。抽出済みの
        ノートは ``extracted_fact_ids`` で弾かれるので二重抽出も起きない。

        書き込みは sleep-time の内側 (SleepTimeWorker) に閉じたままなので、
        「SemMem はチャット応答パスから読むだけ」の不変則は保たれる。

        Returns:
            前倒しで抽出したファクト数 (前倒し不要なら ``0``)。
        """
        mem_cfg = (self.config.get("memory") or {})
        if not ((mem_cfg.get("facts") or {}).get("enable_extraction", True)):
            return 0
        max_notes = int(mem_cfg.get("short_term_max_notes", 100))
        ceiling = max_notes * MemoryEviction.UNEXTRACTED_PROTECTION_CEILING
        if len(self.short_term.notes) < ceiling:
            return 0
        cutoff = self._last_extraction_at
        if not any(
            float(getattr(n, "created_at", 0.0) or 0.0) > cutoff
            for n in self.short_term.notes.values()
        ):
            return 0
        extracted = self._step8_extract_facts()
        logger.info(
            "Step 4: extraction pulled forward before eviction "
            "(%d notes at ceiling %d, %d facts extracted)",
            len(self.short_term.notes), int(ceiling), extracted,
        )
        return extracted

    def _step4_eviction(self) -> int:
        """Step 4: FadeMem eviction 判定"""
        logger.info("Step 4: running eviction check on %d notes", len(self.short_term.notes))
        # 捨てる前に抽出する (:meth:`_extract_before_overflow`)。
        self._extract_before_overflow()
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
            unextracted_cutoff=self._extraction_cutoff(),
        )
        if evicted:
            logger.info("Step 4: evicted %d notes", evicted)
        else:
            logger.debug("Step 4: no notes evicted")
        return evicted


    def _extraction_cutoff(self) -> float | None:
        """eviction へ渡す「Step 8 が最後に走った時刻」。

        抽出が無効な構成では ``None`` を返して保護を掛けない — 走らない工程の
        入力を守り続けると STM が伸びるだけになる。
        """
        facts_cfg = (self.config.get("memory") or {}).get("facts") or {}
        if not facts_cfg.get("enable_extraction", True):
            return None
        return self._last_extraction_at

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

    # ── Step 8 (Chat/Create/MDP Extractor) ─────────────

    def _step8_extract_facts(self) -> int:
        """Step 8: SemanticFact 抽出

        実ロジックは :mod:`backend.free.memory.sleep.extraction`
        に分離された。本メソッドは state (短期記憶 / 設定 / provider / MDP
        キャッシュ) を引数に詰め替えて委譲する薄いラッパ。
        """
        from backend.free.memory.sleep.extraction import extract_semantic_facts

        notes = list(self.short_term.notes.values())
        # 抽出を「走らせた」時刻。次サイクル以降の eviction は、これより後に
        # 作られたノートだけを保護する (未消費の入力を落とさないため)。
        self._last_extraction_at = time.time()
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

    def _curatable_notes(self) -> list:
        """キュレーター (Step 8.4 / 8.5 / 8.6) へ渡す STM ノート。

        private セッション由来を落とす。キュレーター側も入口で
        :func:`~backend.free.memory.sleep._curator_common.public_notes` を
        通すので二重だが、「渡す前に落とす」を呼出側にも置くことで、新しい
        キュレーターを足したときにガードを書き忘れても漏れない側へ倒す
        (Step 8.4-8.6 は Step 8 抽出器の private ガードを継がずに足された)。
        """
        from backend.free.memory.sleep._curator_common import public_notes

        return public_notes(list(self.short_term.notes.values()))

    # ── Step 8.4 (assertion curator) ───────────────────

    async def _step8_4_curate_assertions(self, llm_client=None) -> int:
        """Step 8.4: 型付けできなかった言明を ``world_fact`` として書く。

        実ロジックは :mod:`backend.free.memory.sleep.assertion_curator`
        に分離されている。本メソッドは state を詰め替える薄いラッパ。
        """
        from backend.free.memory.sleep.assertion_curator import (
            curate_assertion_facts,
        )

        notes = self._curatable_notes()
        return await curate_assertion_facts(
            notes,
            store_provider=self._semantic_store_provider,
            aux_client=llm_client,
            embedder=self.embedder,
            profile_id=self._profile_id,
        )

    # ── Step 8.5 (URL curator) ─────────────────────────

    async def _step8_5_curate_urls(self, llm_client=None) -> int:
        """Step 8.5: URL リコール用の ``world_fact`` を sleep-time で書く。

        実ロジックは :mod:`backend.free.memory.sleep.url_curator`
        に分離されている。本メソッドは state を詰め替える薄いラッパ。

        採点は ``run_full`` 経由で渡された sleep-time クライアント (ベース
        モデルの :class:`AuxClient`) で行う。
        """
        from backend.free.memory.sleep.url_curator import curate_url_facts

        notes = self._curatable_notes()
        return await curate_url_facts(
            notes,
            config=self.config,
            store_provider=self._semantic_store_provider,
            scorer_client=llm_client,
            embedder=self.embedder,
            profile_id=self._profile_id,
            debug_logger=self._debug_logger,
        )

    async def _step8_6_curate_commands(self) -> int:
        """Step 8.6: executable command リコール用の ``world_fact`` を書く。

        実ロジックは
        :mod:`backend.free.memory.sleep.executable_command_curator` に分離。
        本メソッドは state を詰め替える薄いラッパ。url_curator と違い
        補助タスク採点をしない (``MemoryNote.tool_command_success`` を使う) ため
        ``aux_client`` は渡さない。
        """
        from backend.free.memory.sleep.executable_command_curator import (
            curate_executable_command_facts,
        )

        notes = self._curatable_notes()
        return await curate_executable_command_facts(
            notes,
            config=self.config,
            store_provider=self._semantic_store_provider,
            embedder=self.embedder,
            profile_id=self._profile_id,
            debug_logger=self._debug_logger,
        )

    # ── Step 13 (failure_pattern 統合) ─────────────────

    async def _step8_8_backfill_fact_embeddings(self) -> int:
        """Step 8.8: 埋め込みが無い SemanticFact へ遡及的に埋め込みを生成する。

        実ロジックは :mod:`backend.free.memory.sleep.fact_embedding` に分離。
        本メソッドは provider / embedder / cancel を詰め替える薄いラッパ。
        """
        from backend.free.memory.sleep.fact_embedding import (
            backfill_fact_embeddings,
        )

        if self._semantic_store_provider is None:
            return 0
        return await backfill_fact_embeddings(
            self._semantic_store_provider,
            self.embedder,
            current_project_id=self._current_project_id,
            is_cancelled=self._check_cancelled,
        )

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
