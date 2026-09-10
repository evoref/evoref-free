"""Sleep-time update ワーカー（Level 0.5）

Trigger A (Light): LLM なし、応答後即座に実行（Steps 1-5）
Trigger B (Full):  LLM あり、アイドル後に実行（Steps 1-10）
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from backend.log_config import get_logger
from backend.trace_context import run_in_executor_with_context
from backend.utils import utc_now
from backend.free.memory.episodic.ingest import ingest_new_turns
from backend.free.memory.episodic.store import EpisodicStore
from backend.free.memory.episodic.turn_source import (
    DEFAULT_SESSION_LIMIT,
    HistoryTurnSource,
    TurnSource,
)

if TYPE_CHECKING:
    from backend.free.agent.aux_prompt_manager import AuxPromptManager
    from backend.free.memory.episodic.workspace import EpisodicWorkspace
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

#: ``_save_state`` 用の 1 スレッド executor。保存を直列化して順序を保つ
#: (:meth:`SleepTimeWorker._save_state_async`)。
_SAVE_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="episodic-save")

#: Light サイクルで snapshot を作る事象数の下限。
#:
#: snapshot 生成は畳み込み + 索引再構築 + 増分埋め込みで、Light は応答のたびに
#: 走る。1 ターン (= 2 ノート) ごとに版を積むと、会話中ずっと索引を作り直す
#: ことになる。**新しいノートが検索に出るのは次の snapshot から** だが、直近の
#: 会話はワーキングメモリの窓がそのままプロンプトに載るので、そこは失われない
#: (c_16 §2.1 の割り切りと同じ)。Full は溜まった事象があれば必ず版を作る。
_SNAPSHOT_MIN_EVENTS_LIGHT = 16

# ノートを埋め込む際の本文上限 (文字数)。llama-server の embed インスタンスは
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
        episodic: EpisodicStore,
        embedder: EmbeddingBackend,
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
        turn_source: TurnSource | None = None,
        triggers_dir: Path | str | None = None,
        private_trace_ids_provider: Callable[[], set[str]] | None = None,
        learning_disabled: bool = False,
    ):
        self.episodic = episodic
        self.embedder = embedder
        self.config = config
        self.experience_buf = experience_buf
        self._policy = policy
        self._debug_logger = debug_logger
        self.learned_patterns = learned_patterns
        self.vector_store = vector_store
        self.cartridge_manager = cartridge_manager
        self._aux_prompt_manager = aux_prompt_manager
        self._fewshot_pool = None
        self._cancelled = False
        #: サイクル (Light / Full) の排他。3 ストアの書き手は sleep-time だけ
        #: という不変則 (c_16 §2.1) を **慣習ではなくロックで** 守る。Full は
        #: 待って直列化し、Light は取れなければ飛ばす (:meth:`run_light`)。
        self._cycle_lock = asyncio.Lock()
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
        #: 最後に Step 8 (ファクト抽出) が走った時刻 (観測用)。tier 遷移と
        #: 保持方針はストア側 (c_16 §4.1 / §5.4) が持つので、eviction の保護
        #: 基準としては使わなくなった。
        self._last_extraction_at: float = 0.0
        # ── MDPIngester (agent_trace*.jsonl → エピソード記憶) ──
        # log_dir は AgentTraceStore の常設ディレクトリ (local_paths.agent_trace_dir)。
        # state ファイルはメモリディレクトリ配下に置く想定だが、テスト容易性
        # のため lazy 初期化する。
        self._mdp_ingester = None
        #: 「今チャット生成が走っているか」の判定 (scheduler が注入)。
        self._chat_in_flight_probe = None
        #: ノート化していないターンの供給元 (既定は会話履歴)。
        self._turn_source: TurnSource = turn_source or HistoryTurnSource()
        #: pin トリガ辞書の user override ディレクトリ。
        self._triggers_dir = triggers_dir
        #: private セッションの trace_id (MDP 昇格の除外に使う)。ターン自体は
        #: 履歴にもストアにも残らないので、窓を持つ側から貰う。
        self._private_trace_ids_provider = private_trace_ids_provider
        #: 現在のサイクルで開いている作業領域 (Full の間だけ生きる)。
        self._workspace: "EpisodicWorkspace | None" = None
        #: ``--no-learning``。**学習に属するステップだけ** no-op にする。
        #: ノート化 / 抽出 / 保持方針 / snapshot は記憶側の仕事なので走らせる
        #: (c_16 §2.1: 3 ストアの書き手は sleep-time だけ — ここを止めると
        #: 記憶が一切書かれなくなる)。
        self.learning_disabled = bool(learning_disabled)

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
        """Light 版のサイクルロックを取って :meth:`_run_light_locked` を回す。

        既に別のサイクル (Light / Full) が走っている場合は **待たずに飛ばす**。
        Light は応答のたびに走る軽い版で、飛ばしても次の応答で同じ入力から
        やり直せる。待たせると in-flight の Full の裏で Light が積み上がる。

        Returns:
            実行結果サマリ dict。飛ばした場合は ``skipped="cycle_in_progress"``。
        """
        if self._cycle_lock.locked():
            logger.info(
                "Sleep-time Light skipped: another sleep-time cycle is in progress",
            )
            return {
                "notes_created": 0, "touched": 0, "snapshot": "",
                "knowledge_claims": 0, "skipped": "cycle_in_progress",
            }
        async with self._cycle_lock:
            self._cancelled = False
            return await self._run_light_locked()

    async def _run_light_locked(self) -> dict:
        """Light 版: LLM なし。ノート生成 → touch flush → (必要なら) snapshot。

        **サイクルロック保持中に呼ぶこと** (:meth:`run_light` /
        :meth:`run_full` が入口)。``_cancelled`` のリセットも入口側が持つ。

        旧 Light の Step 1-4 (埋め込み / タグ補完 / LightMem スコア再計算 /
        eviction) は無くなった:

        - 埋め込みは snapshot 生成時に **増分で** 作られる (c_16 §6.1)
        - タグ / キーワードはノート生成時に確定する (``NoteBuilder``)
        - LightMem スコアは廃止 (順位式は c_16 §7.2 の 1 本)
        - eviction は tier 遷移と保持方針 (:meth:`_step_e_lifecycle`) が担う。
          ストア間のコピーは無くなったので、ここで「落とす」ものは無い

        Returns:
            実行結果サマリ dict
        """
        started_at = utc_now()
        t0 = time.monotonic()
        step_durations: dict[str, float] = {}
        result: dict = {
            "notes_created": 0, "touched": 0, "snapshot": "", "knowledge_claims": 0,
        }

        logger.info(
            "Sleep-time Light started (%d episodic record(s))", len(self.episodic),
        )
        try:
            ts = time.monotonic()
            result["notes_created"] = self._step_e1_build_notes()
            step_durations["step_e1_note_build"] = round(time.monotonic() - ts, 3)
            if self._check_cancelled():
                return result

            ts = time.monotonic()
            result["touched"] = self.episodic.flush_touch()
            step_durations["step_e4_touch_flush"] = round(time.monotonic() - ts, 3)

            ts = time.monotonic()
            result["patterns_decayed"] = self._step5_5_decay_patterns()
            step_durations["step5_5_patterns"] = round(time.monotonic() - ts, 3)

            # 手動予約 (POST /api/pro/knowledge/fetch) が立っていれば取得器を
            # 回す。定期取得は Full のアイドル窓のまま — 予約された 1 回だけ
            # Light でも拾い、API が SemMem を直接書かなくて済むようにする。
            ts = time.monotonic()
            result["knowledge_claims"] = await self._step8_7_fetch_knowledge(
                only_if_requested=True,
            )
            step_durations["step8_7_knowledge_fetch"] = round(time.monotonic() - ts, 3)
        finally:
            ts = time.monotonic()
            result["snapshot"] = await self._maybe_snapshot(
                min_events=_SNAPSHOT_MIN_EVENTS_LIGHT,
            )
            step_durations["step_e5_snapshot"] = round(time.monotonic() - ts, 3)
            await self._save_state_async()

        elapsed = round(time.monotonic() - t0, 3)
        logger.info("Sleep-time Light completed in %.3fs: %s", elapsed, result)

        dl = self._debug_logger
        if dl:
            dl.log_learning_cycle(cycle_num=0, data={
                "level": "0.5-light",
                "started_at": started_at,
                "elapsed_sec": elapsed,
                "step_durations_sec": step_durations,
                "notes_count": len(self.episodic),
                **result,
            })
            dl.log_outcome(
                kind="learning_cycle_l05",
                success=True,
                duration_ms=elapsed * 1000,
                quality_signals={
                    "level": "0.5-light",
                    "notes_count": len(self.episodic),
                    **{
                        k: v for k, v in result.items()
                        if isinstance(v, (int, float, bool))
                    },
                },
            )

        return result

    # ── エピソード記憶のライフサイクル (c_16 §4.1) ──────────

    def _step_e1_build_notes(self) -> int:
        """会話履歴のうち、まだノートにしていないターンをノート化する。

        応答パスは ``WorkingMemory`` に積むだけになったので (c_16 §2.1: 書き手
        は sleep-time だけ)、ノートの入力はターン ID の発行元である会話履歴。
        どこまでノート化したかは ``episodic/progress.json`` が持つ。
        """
        try:
            created = ingest_new_turns(
                self.episodic,
                self._turn_source,
                session_limit=DEFAULT_SESSION_LIMIT,
                triggers_dir=self._triggers_dir,
                config=self.config,
            )
        except Exception as e:  # noqa: BLE001 — 1 セッションの失敗で止めない
            logger.warning("Note build from conversation turns failed: %s", e)
            return 0
        if created:
            logger.info("Sleep-time: built %d episodic note(s)", created)
        return created

    def _retention_value(self, key: str, default: float) -> float:
        """``memory.evidence.retention.<key>`` (無ければ manifest の宣言値)。"""
        cfg = ((self.config.get("memory") or {}).get("evidence") or {})
        raw = (cfg.get("retention") or {}).get(key)
        if raw is None:
            raw = self.episodic.evidence.manifest.retention_value(key)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return default

    def _step_e2_promote_tiers(self) -> int:
        """``short_days`` を超えたノートを ``long`` へ (``patch`` のみ)。"""
        return self.episodic.promote_aged_notes(
            short_days=self._retention_value("short_days", 14.0),
        )

    def _step_e3_summarize_sessions(self) -> int:
        """要約が出来ているセッションのノートを要約 1 件へ畳む (c_16 §4.1)。

        要約本文は履歴側 (Step 8-9) が LLM で作ったもの。ここで作り直さない
        のは、同じ会話の要約を 2 か所で持つと必ず食い違うため。元ノートは
        ``superseded_by`` で要約を指し、消えはしない。
        """
        summarized = 0
        try:
            sessions = self._turn_source.recent_sessions(limit=DEFAULT_SESSION_LIMIT)
        except Exception as e:  # noqa: BLE001
            logger.warning("Episodic summary: failed to list sessions: %s", e)
            return 0
        by_session: dict[str, list[str]] = {}
        #: 既に書いた要約の本文 (同じ文面を二度書かない)。会話が伸びると
        #: 後から昇格したノートで再度この工程に入るが、履歴側の要約が
        #: 作り直されていなければ畳む意味が無い。
        existing: dict[str, set[str]] = {}
        for note in self.episodic.iter_notes(tier="long"):
            if note.summary_of:
                existing.setdefault(note.session_id, set()).add(note.content.strip())
                continue
            if note.superseded_by:
                continue
            by_session.setdefault(note.session_id, []).append(note.id)
        for session in sessions:
            summary = (session.summary or "").strip()
            note_ids = by_session.get(session.session_id) or []
            if not summary or len(note_ids) < 2:
                continue
            if summary in existing.get(session.session_id, ()):
                continue
            if self.episodic.write_summary_note(
                session_id=session.session_id,
                text=summary,
                note_ids=note_ids,
                mode=session.mode if session.mode in ("chat", "create") else "chat",
                lang=session.lang,
                project_id=session.project_id,
            ):
                summarized += 1
        return summarized

    def _step_e_lifecycle(self, result: dict) -> None:
        """tier 昇格 → 要約 → 保持方針 の 3 つをまとめて回す。"""
        result["notes_promoted"] = self._step_e2_promote_tiers()
        result["notes_summarized"] = self._step_e3_summarize_sessions()
        result["notes_retracted"] = self.episodic.enforce_retention(
            long_max_records=int(self._retention_value("long_max_records", 50000)),
        )

    async def _maybe_snapshot(self, *, min_events: int = 1) -> str:
        """未畳み込み事象が ``min_events`` 以上あれば版を作る。

        episodic と semantic の 2 ストアを同じ条件で畳む。版を作った時点で
        埋め込みとクラスタ索引・転置索引がまとめて更新される (c_16 §6.1)。

        Returns:
            作った episodic の版名。作らなければ空文字。
        """
        version = ""
        pending = int(self.episodic.evidence.manifest.events_since_snapshot)
        if pending < max(1, min_events):
            self.episodic.save_progress()
        else:
            try:
                version = await self.episodic.create_snapshot() or ""
            except Exception as e:  # noqa: BLE001 — 版が作れなくても事象は残る
                logger.warning("Episodic snapshot failed: %s", e)
        await self._maybe_snapshot_semantic(min_events=min_events)
        return version

    async def _maybe_snapshot_semantic(self, *, min_events: int = 1) -> str:
        """SemMem 側の保持方針を適用してから版を作る (c_16 §5.4 / §6.1)。"""
        store = self._semantic_store()
        if store is None:
            return ""
        try:
            store.enforce_retention()
        except Exception as e:  # noqa: BLE001 — 保持で落ちても版は作る
            logger.warning("Semantic retention failed: %s", e)
        try:
            store.flush_touch()
        except Exception as e:  # noqa: BLE001
            logger.warning("Semantic touch flush failed: %s", e)
        pending = int(store.evidence.manifest.events_since_snapshot)
        if pending < max(1, min_events):
            store.save_manifest()
            return ""
        try:
            return await store.create_snapshot() or ""
        except Exception as e:  # noqa: BLE001 — 版が作れなくても事象は残る
            logger.warning("Semantic snapshot failed: %s", e)
            return ""

    def _semantic_store(self):
        """SemMem 本体 (スコープ束縛ビューではない)。未配線なら ``None``。

        ``semantic_store_provider`` は 1 スコープに束縛したビューを返すので、
        版の生成・保持方針のようなストア全体の操作はここで本体を取り出す。
        """
        provider = self._semantic_store_provider
        if provider is None:
            return None
        try:
            return provider("global").store
        except Exception as e:  # noqa: BLE001 — 未初期化なら黙って諦める
            logger.debug("Semantic store unavailable for snapshot: %s", e)
            return None

    def _workspace_or_open(self) -> "EpisodicWorkspace":
        """現サイクルの作業領域 (無ければ開く)。"""
        if self._workspace is None:
            self._workspace = self.episodic.open_workspace()
        return self._workspace

    async def _step6_resolve_conflicts(
        self, llm_client, params_b: float, result: dict,
    ) -> None:
        """Step 6: ConflictResolver による短期ノートのコンフリクト解決。

        作業領域 (``short`` tier のノート) の競合解決後に SemanticFactStore
        上のコンフリクト解消を続けて実行する。
        """
        from backend.free.memory.pipeline.conflict_resolver import ConflictResolver
        resolver = ConflictResolver(
            self.config, params_b=params_b,
            policy=getattr(self, "_policy", None),
            debug_logger=self._debug_logger,
        )
        result["conflicts_resolved"] = await resolver.resolve_conflicts(
            self._workspace_or_open(), llm_client,
            should_pause=self._chat_in_flight,
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
        workspace = self._workspace_or_open()
        # リンク張り直し + クラスタリング (LLM 不要)
        link_stats = evolver.rebuild_links_and_clusters(workspace)
        result["notes_links_rebuilt"] = link_stats.get("links", 0)
        result["notes_clusters"] = link_stats.get("clusters", 0)
        result["notes_evolved"] = await evolver.evolve_notes(
            workspace, self.episodic, llm_client,
            should_pause=self._chat_in_flight,
        )

    async def _step6_5_reembed_after_conflicts(
        self, result: dict,
    ) -> None:
        """Step 6.5: コンフリクト解決で embedding=None になったノートを再埋め込み。"""
        re_embedded = await self._embed_workspace_notes()
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
                "notes_count": len(self.episodic),
                **result,
            })
            dl.log_outcome(
                kind="learning_cycle_l05_full",
                success=True,
                duration_ms=elapsed * 1000,
                quality_signals={
                    "level": "0.5-full",
                    "notes_count": len(self.episodic),
                    **{k: v for k, v in result.items() if isinstance(v, (int, float, bool))},
                },
            )

    async def run_full(self, llm_client=None) -> dict:
        """Full 版のサイクルロックを取って :meth:`_run_full_locked` を回す。

        Full は **待つ** (Light と違って飛ばせない — ファクト抽出 / 競合解決 /
        保持方針は Full にしか無い)。2 本目が 1 本目の作業領域
        (``_workspace``) とキャンセル要求を消す事故を構造的に止める
        (2026-09-08 ライブ監査: 手動 Full と Trigger B が 523 秒並走)。

        Args:
            llm_client: LLM クライアント（Steps 6-10 で使用）。
                        AuxClient（推奨）または LocalClient。
                        設計書 §5.5.2 に基づき、補助タスクの使用を推奨。

        Returns:
            実行結果サマリ dict
        """
        if self._cycle_lock.locked():
            logger.info(
                "Sleep-time Full is waiting for the in-progress cycle to finish",
            )
        async with self._cycle_lock:
            self._cancelled = False
            return await self._run_full_locked(llm_client)

    async def _run_full_locked(self, llm_client=None) -> dict:
        """Full 版: LLM あり、Steps 1-10 (サイクルロック保持中に呼ぶこと)。

        Args:
            llm_client: LLM クライアント（Steps 6-10 で使用）。

        Returns:
            実行結果サマリ dict
        """
        started_at = utc_now()
        t0 = time.monotonic()
        step_durations: dict[str, float] = {}
        # 前サイクルが途中で打ち切られていると作業領域が残る。ノートは
        # そのあいだに Light が増やしているので、必ず開き直す。
        self._workspace = None

        # まず Light 版を実行 (ロック保持者として本体を直接呼ぶ)
        ts = time.monotonic()
        result = await self._run_light_locked()
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

        # Step E0: 作業領域を開いて、ベクトルの無いノートを埋め込む。
        # Step 6 (競合検出) / Step 7 (ノート進化) はノート同士の類似度を要る。
        ts = time.monotonic()
        result["workspace_embedded"] = await self._embed_workspace_notes()
        step_durations["step_e0_workspace_embed"] = round(time.monotonic() - ts, 3)
        if self._check_cancelled():
            return result

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

        # Step 7.5: MDP トレース → エピソード記憶 (long tier) 投入。
        # Step 8 (extractor) よりも前に行うことで、当該 trace_id のノートが
        # ストアに存在する状態でファクト抽出が走り、``trace_id`` が
        # エピソード記憶 / 意味記憶の双方に伝播する。
        ts = time.monotonic()
        result["mdp_traces_ingested"] = await self._step7_5_ingest_mdp_traces()
        step_durations["step7_5_mdp_ingest"] = round(time.monotonic() - ts, 3)
        if self._check_cancelled():
            return result

        # Step 8.0: 訂正候補の検証。Step 8 が ``from_correction`` / 値アンカー /
        # 継承で **訂正の力** を使う前に、「本当に過去の発言の誤りを指して
        # いるか」を判定してノートへ刻む (2026-09-08 夜の監査 G-01 / G-04)。
        ts = time.monotonic()
        result["corrections_verified"] = await self._step8_0_verify_corrections(
            llm_client,
        )
        step_durations["step8_0_correction_verify"] = round(time.monotonic() - ts, 3)
        if self._check_cancelled():
            return result

        # Step 8: SemanticFact Extractor
        # 既存 _step8_9_summarize_sessions は Step 9 に再配置される想定。
        # メソッド名を変えずに前段に Step 8 を挿入する形で共存させる。
        ts = time.monotonic()
        result["facts_extracted"] = self._step8_extract_facts(
            verification_available=llm_client is not None,
        )
        step_durations["step8_extract_facts"] = round(time.monotonic() - ts, 3)
        if self._check_cancelled():
            return result

        # Step 8.3: Step 8 (regex) が「私は〜」発話から属性を取りこぼした /
        # 複数節を 1 属性に飲み込んだときだけ、補助タスクで逐語 span に分ける。
        # 語形 1 つの欠落で 0 件になる事故 (2026-08-31 / 09-04 / 09-08 F-01) の
        # 受け皿。値は発話の部分文字列であることをコード側で検証する。
        ts = time.monotonic()
        result["personal_facts_split"] = await self._step8_3_split_personal_facts(
            llm_client,
        )
        step_durations["step8_3_personal_fact_split"] = round(time.monotonic() - ts, 3)
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

        # Step 8.7: know.* 取得器 (Pro)。Free には origin=web の書き手が無い
        # ので、登録が無ければ 0 で通過する。
        ts = time.monotonic()
        result["knowledge_claims"] = await self._step8_7_fetch_knowledge()
        step_durations["step8_7_knowledge_fetch"] = round(time.monotonic() - ts, 3)
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

        # 旧 Step 8.8 (ファクト埋め込みの遡及生成) は無くなった。埋め込みは
        # SemMem の snapshot 生成時に **増分で** 作られる (c_16 §6.1)。
        # 「版に載るまで密ベクトル検索の対象にならない」契約は episodic と同じ。

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
        # (ノートが snapshot に載るまで注入対象にならないのと同じ契約)。
        ts = time.monotonic()
        result["fewshot_embeddings_backfilled"] = (
            await self._step10b2_backfill_fewshot_embeddings()
        )
        step_durations["step10b2_fewshot_embedding"] = round(time.monotonic() - ts, 3)

        # Step 10c (語彙索引の再構築) は廃止した (c_16 §6.2 / §8)。3 ストアとも
        # ``EvidenceStore`` が snapshot 生成のたびに転置索引を作り直すので、
        # 別立てで張り直す索引がもう無い。

        # Step E2-E5: tier 昇格 → 要約 → 保持方針 → touch flush → snapshot。
        # 作業領域の変更を先に事象へ落とす (畳み込みの入力に載せるため)。
        ts = time.monotonic()
        if self._workspace is not None:
            result["workspace_flushed"] = self._workspace.flush()
            self._workspace = None
        self._step_e_lifecycle(result)
        result["touched"] = self.episodic.flush_touch()
        step_durations["step_e_lifecycle"] = round(time.monotonic() - ts, 3)

        ts = time.monotonic()
        result["snapshot"] = await self._maybe_snapshot(min_events=1)
        step_durations["step_e5_snapshot"] = round(time.monotonic() - ts, 3)

        # Step 10d: 起動時に条件を満たさず見送られた閾値較正を拾い直す。
        # 較正は起動時 1 回きりだったため、ノートが少ない状態で起動すると
        # プロセスの生涯にわたり config の静的閾値 (別モデル前提で到達不能)
        # が使われ続けていた。**版を作った後**に置くこと — 較正はノートの
        # ベクトルを読むので、同じサイクルで増えた分を使うには索引が要る。
        ts = time.monotonic()
        result["threshold_calibrated"] = await self._step10d_retry_calibration()
        step_durations["step10d_threshold_calibration"] = round(
            time.monotonic() - ts, 3,
        )

        # 永続化 + 完了ログ
        await self._save_state_async()
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
        if self._skip_for_learning("Step 10b-2 (few-shot embedding backfill)"):
            return 0
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

    async def _embed_workspace_notes(self) -> int:
        """作業領域のノートのうち、ベクトルを持たないものを埋め込む。

        **永続化しない一時値** (:attr:`MemoryNote.embedding`)。競合検出
        (Step 6) とノート進化 (Step 7) がノート同士の類似度を要るためだけに
        載せる。ストアの索引は snapshot 生成時に増分で作られる (c_16 §6.1)。

        1 件でも embed サーバの context を超えるノートがあるとバッチ全体が
        400 で落ちるので、(a) 本文を切り詰め、(b) バッチ失敗時はノート単位へ
        フォールバックする (2026-07-25 のデッドロックの対策をそのまま踏襲)。
        """
        if self.embedder is None:
            return 0
        workspace = self._workspace_or_open()
        pending = workspace.unembedded()
        if not pending:
            logger.debug("Embedding: no unembedded notes in the workspace")
            return 0

        logger.info("Embedding %d workspace note(s)...", len(pending))
        embedded = 0
        for start_at in range(0, len(pending), _EMBED_BATCH_SIZE):
            batch = pending[start_at:start_at + _EMBED_BATCH_SIZE]
            texts = [_truncate_for_embedding(n.content) for n in batch]
            try:
                embeddings = await self.embedder.embed(texts, is_query=False)
            except Exception as exc:
                logger.warning(
                    "Batch embed failed (%d notes), falling back to per-note: %s",
                    len(batch), exc,
                )
                embedded += await self._embed_notes_individually(batch)
                continue
            for note, emb in zip(batch, embeddings):
                note.embedding = emb.astype(np.float32)
                embedded += 1
        logger.info("Embedded %d workspace note(s)", embedded)
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
                logger.warning(
                    "Note embed failed (len=%d): %s", len(note.content), exc,
                )
                continue
            if emb is None or len(emb) == 0:
                continue
            note.embedding = emb[0].astype(np.float32)
            embedded += 1
        return embedded

    def _skip_for_learning(self, step: str) -> bool:
        """``--no-learning`` で飛ばす学習ステップか (飛ばすなら DEBUG を 1 行)。"""
        if not self.learning_disabled:
            return False
        logger.debug("%s skipped (learning disabled)", step)
        return True

    def _step5_5_decay_patterns(self) -> int:
        """Step 5.5: 学習済みパターンの重み減衰と永続化

        ``--no-learning`` では走らない (学習済みパターンは EvorefLearn の
        資産で、記憶の書き戻しではない)。

        sleep-time Light で毎回呼ばれるが、減衰そのものは
        ``LearnedPatternStore.maybe_decay_all`` が壁時計で間引く
        (``PATTERN_DECAY_INTERVAL_SEC``)。Light は LLM 生成ごとに走るため、
        毎回減衰すると学習した語が数ターンで消えていた。永続化は追加 /
        重み変更 / 削除があった時 (``dirty``) だけ行う。LLM 不要。
        """
        if self._skip_for_learning("Step 5.5 (pattern decay)"):
            return 0
        if self.learned_patterns is None:
            return 0

        removed = self.learned_patterns.maybe_decay_all()

        if self.learned_patterns.dirty:
            try:
                from backend.config import get_path_resolver
                resolver = get_path_resolver()
                patterns_file = resolver.resolve_learning("learned_patterns_file")
                self.learned_patterns.save(patterns_file)
            except Exception as e:
                logger.warning("Failed to save learned patterns: %s", e)

        if removed:
            logger.info("Step 5.5: decayed %d patterns", removed)
        return removed or 0

    async def _step5_8_contextual_prefixes(self, llm_client) -> int:
        """Step 5.8: Contextual Retrieval プレフィックス生成。

        実ロジックは
        :mod:`backend.free.memory.sleep.contextual` に分離された。
        本メソッドはメイン VectorStore / カートリッジを
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
            is_cancelled=self._check_cancelled,
            should_pause=self._chat_in_flight,
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
            batch_size=int(history_cfg.get("summary_batch_size", 20)),
            is_cancelled=self._check_cancelled,
            should_pause=self._chat_in_flight,
        )

    # ── Step 7.5 (MDP トレース → episodic LTM) ─────────

    async def _step7_5_ingest_mdp_traces(self) -> int:
        """Step 7.5: ``agent_trace*.jsonl`` をエピソード記憶に取り込む。

        実ロジックは :mod:`backend.free.memory.sleep.mdp_ingest`
        に分離された。本メソッドは state を詰め替えて委譲する薄いラッパ。
        """
        from backend.free.memory.sleep.mdp_ingest import ingest_mdp_traces

        provider = self._private_trace_ids_provider
        private_trace_ids: set[str] = set()
        if provider is not None:
            try:
                private_trace_ids = set(provider() or ())
            except Exception as exc:  # noqa: BLE001
                logger.warning("Step 7.5: private trace id lookup failed: %s", exc)
        ingested, self._mdp_ingester = ingest_mdp_traces(
            self.episodic,
            config=self.config,
            agent_trace_dir=self._agent_trace_dir,
            current_project_id=self._current_project_id,
            cached_ingester=self._mdp_ingester,
            private_trace_ids=private_trace_ids,
        )
        return ingested

    # ── Step 8.0 (correction curator) ──────────────────

    async def _step8_0_verify_corrections(self, llm_client=None) -> int:
        """Step 8.0: 字句で立てた訂正候補を検証してノートへ帰属を刻む。

        実ロジックは :mod:`backend.free.memory.sleep.correction_curator`
        に分離されている。本メソッドは state を詰め替える薄いラッパ。
        """
        from backend.free.memory.sleep.correction_curator import curate_corrections

        return await curate_corrections(
            self._curatable_notes(), aux_client=llm_client,
        )

    # ── Step 8 (Chat/Create/MDP Extractor) ─────────────

    def _step8_extract_facts(self, *, verification_available: bool = False) -> int:
        """Step 8: SemanticFact 抽出

        実ロジックは :mod:`backend.free.memory.sleep.extraction`
        に分離された。本メソッドは state (短期記憶 / 設定 / provider / MDP
        キャッシュ) を引数に詰め替えて委譲する薄いラッパ。
        """
        from backend.free.memory.sleep.extraction import extract_semantic_facts

        notes = list(self._workspace_or_open().notes.values())
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
            verification_available=verification_available,
        )
        return total

    def _curatable_notes(self) -> list:
        """キュレーター (Step 8.4 / 8.5 / 8.6) へ渡す ``short`` ノート。

        private セッション由来を落とす。キュレーター側も入口で
        :func:`~backend.free.memory.sleep._curator_common.public_notes` を
        通すので二重だが、「渡す前に落とす」を呼出側にも置くことで、新しい
        キュレーターを足したときにガードを書き忘れても漏れない側へ倒す
        (Step 8.4-8.6 は Step 8 抽出器の private ガードを継がずに足された)。
        """
        from backend.free.memory.sleep._curator_common import public_notes

        return public_notes(list(self._workspace_or_open().notes.values()))

    # ── Step 8.3 (personal fact split) ─────────────────

    async def _step8_3_split_personal_facts(self, llm_client=None) -> int:
        """Step 8.3: regex が取りこぼした自己開示発話を属性ごとに分けて書く。

        実ロジックは :mod:`backend.free.memory.sleep.personal_fact_curator`
        に分離されている。本メソッドは state を詰め替える薄いラッパ。
        """
        from backend.free.memory.sleep.personal_fact_curator import (
            curate_personal_facts,
        )

        notes = self._curatable_notes()
        return await curate_personal_facts(
            notes,
            store_provider=self._semantic_store_provider,
            aux_client=llm_client,
            embedder=self.embedder,
            profile_id=self._profile_id,
            should_pause=self._chat_in_flight,
        )

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
            should_pause=self._chat_in_flight,
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
            should_pause=self._chat_in_flight,
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

    async def _step8_7_fetch_knowledge(self, *, only_if_requested: bool = False) -> int:
        """Step 8.7: ``know.*`` 取得器を回す (Pro 限定)。

        取得器は ``backend.pro.knowledge.KnowledgeFetcher`` で、Pro 起動時に
        ``register_pro_handler("knowledge_fetcher", …)`` で登録される。Free
        では未登録なので 0 を返して素通りする (Edition Gate の Handler 経路、
        e_02 §3.3)。

        ここに置くのは **版を作る前** だから — 取得した claim を同じサイクルの
        snapshot に載せる。取得器自身は ``create_snapshot`` を呼ばない
        (稼働中の索引を書き換えない、c_16 §2.1)。

        Args:
            only_if_requested: 手動予約 (``POST /api/pro/knowledge/fetch``)
                が立っているときだけ回す。Light サイクルはこれで呼ぶ —
                定期取得は Full のアイドル窓に閉じたまま、手で頼んだ 1 回
                だけ次の Light で拾えるようにする。

        Returns:
            書き込んだ claim 件数。取得器が無い / 失敗した場合は 0。
        """
        from backend.edition import get_pro_handler

        fetcher = get_pro_handler("knowledge_fetcher")
        if fetcher is None:
            return 0
        if only_if_requested and not getattr(fetcher, "fetch_requested", False):
            return 0
        try:
            return await fetcher.run_once()
        except Exception as e:  # noqa: BLE001 — 取得失敗でサイクルを落とさない
            logger.warning("Knowledge fetch failed: %s", e)
            return 0

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
        if self._skip_for_learning("Step 13 (failure pattern consolidation)"):
            return {}

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
        """Step 10: 180 日無アクセスのプロジェクトのファクトを退役させる

        実ロジックは :mod:`backend.free.memory.sleep.archive`
        に分離された。本メソッドはコールバックを詰め替える薄いラッパ。

        ``store_provider`` を渡すのは、アーカイブが **ディレクトリ移動から
        scope 単位の ``retract(reason="project_archived")`` へ変わった** ため
        (c_16 §8)。渡さないと退役が一切走らず、アーカイブ済みプロジェクトの
        ファクトが live のまま残る。

        Returns:
            アーカイブ対象となったプロジェクト ID のリスト (state フラグ更新済)。
        """
        from backend.free.memory.sleep.archive import archive_inactive_projects

        return archive_inactive_projects(
            config=self.config,
            store_invalidator=self._semantic_store_invalidator,
            store_provider=self._semantic_store_provider,
        )

    async def _save_state_async(self) -> None:
        """:meth:`_save_state` をワーカースレッドで走らせる。

        進捗ファイルと経験バッファの書き出しは同期 I/O で、Light はチャットの
        ストリーミング中にも走る。イベントループ上で書くとその間トークンが
        1 つも流れない。``run_in_executor_with_context``
        で trace_id を保ったままスレッドへ逃がす。専用の 1 スレッド executor
        なので保存同士は直列化され、順序が入れ替わらない。
        """
        loop = asyncio.get_running_loop()
        await run_in_executor_with_context(loop, _SAVE_EXECUTOR, self._save_state)

    def _save_state(self) -> None:
        """メモリ状態を永続化 (ノート化の進捗 + 経験バッファ)。"""
        from backend.config import get_path_resolver

        try:
            self.episodic.save_progress()
        except Exception as e:
            logger.warning("Failed to save episodic progress: %s", e)

        # 経験バッファを永続化 (起動時ロードと同じ resolve_learning でパーティション先に揃える)
        # ``--no-learning`` では経験の書き戻し自体を行わない。
        if self.experience_buf is not None and not self.learning_disabled:
            try:
                resolver = get_path_resolver()
                exp_file = resolver.resolve_learning("experience_file")
                self.experience_buf.save(exp_file)
            except Exception as e:
                logger.warning("Failed to save experience buffer: %s", e)

        # エピソード記憶のベクトル索引は snapshot 生成時に書かれるので、
        # ここで save するものは無い (c_16 §5.3)。
