"""A-MEM ノート進化: LLM による context_description 生成 + リンク張り直し / クラスタリング

context_description 生成に加え、STM ノート間の
リンク (`links`) と所属クラスタ (`cluster_id`) を sleep-time Step 7 で再構築する。

リンクは「embedding コサイン類似度がしきい値以上の上位 K ノート」を採用する
(LLM 不要、決定論的)。クラスタは union-find で連結成分を求め、各クラスタの
代表 ID (含まれるノート ID の最小値) を `cluster_id` として割り当てる。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from backend.log_config import get_logger
from backend.free.llm.model_metadata import DEFAULT_PARAMS_B
from backend.free.memory.stores.short_term import MemoryNote, ShortTermMemory
from backend.free.memory.stores.long_term import LongTermMemory

if TYPE_CHECKING:
    from backend.debug_logger import DebugLogger
    from backend.free.agent.aux_prompt_manager import AuxPromptManager

logger = get_logger("memory.note_evolver")

# ノート内容の最大文字数（コンテキストウィンドウ超過防止）
MAX_NOTE_CONTENT_LEN = 800
MAX_CONTEXT_CONTENT_LEN = 150

# 空レスポンス時のリトライ回数
# aux_client 側でも MAX_RETRIES (=3) 分リトライするため、ここでは
# 「KV キャッシュ汚染等で連続失敗が発生した場合に slot 回復を待つ」目的で
# 追加の外側リトライを行う。合計で最大 (1 + _EMPTY_RESPONSE_MAX_RETRIES) 回試行する。
_EMPTY_RESPONSE_MAX_RETRIES = 2

# aux_prompt_manager が利用不可な場合のフォールバックプロンプト
_FALLBACK_SYSTEM_PROMPT = (
    "メモリノートの暗黙的な意味・トピック・重要性を捉えた簡潔な文脈説明（1〜2文）を"
    "生成してください。説明文のみを出力してください。"
)


#: ``base_interval`` に対する自動スケールの上限倍率。
#: sleep-time をベースモデル (27B) で回すようになり、素の線形スケールでは
#: 呼出間に 3.86 秒 (= 1.0 × 27/7) の待機が入るようになった。1 サイクル 10〜15 件
#: なら 40〜58 秒がまるまる待機で消える。インターバルの役割 (チャットへ GPU を
#: 明け渡す) は :meth:`NoteEvolver.evolve_notes` の ``should_pause`` (1 件ごとの
#: 協調 yield) が担っており、かつ大きいモデルほど 1 呼出自体が長くなるので、
#: 待機まで比例させる必要はない。
#:
#: **専有スロットは当てにしない** — llama-server は全スロットを逐次実行するので
#: ``background_slot`` は KV を分離するがレイテンシは分離しない
#: (2026-08-27 ライブ監査で実測)。
_MAX_INTERVAL_SCALE = 2.0


def compute_llm_call_interval(base_interval: float, params_b: float) -> float:
    """モデルサイズに応じた LLM 呼び出しインターバルを算出

    Args:
        base_interval: 7B モデル基準の基本インターバル（秒）
        params_b: モデルのパラメータ数（B 単位）

    Returns:
        実効インターバル（秒）。最低 0.1 秒、上限 ``base_interval * 2``
    """
    if base_interval <= 0:
        return 0.0
    interval = base_interval * (params_b / DEFAULT_PARAMS_B)
    interval = min(interval, base_interval * _MAX_INTERVAL_SCALE)
    return max(interval, 0.1)


class NoteEvolver:
    """evolution_pending なノートに対して LLM で文脈説明を生成"""

    def __init__(
        self,
        config: dict,
        params_b: float = DEFAULT_PARAMS_B,
        aux_prompt_manager: AuxPromptManager | None = None,
        debug_logger: DebugLogger | None = None,
    ):
        mc = config.get("memory", {})
        self.enabled: bool = mc.get("note_evolution_enabled", True)
        self.batch_size: int = mc.get("note_evolution_batch", 10)
        self.context_k: int = mc.get("note_evolution_context_k", 3)
        # リンク張り直し + クラスタリング設定
        self.link_rebuild_enabled: bool = mc.get("note_link_rebuild_enabled", True)
        self.link_top_k: int = int(mc.get("note_link_top_k", 5))
        self.link_threshold: float = float(mc.get("note_link_threshold", 0.5))
        self.clustering_enabled: bool = mc.get("note_clustering_enabled", True)

        # LLM スキップ閾値 (confidence / max_per_cycle)
        ne_cfg = mc.get("note_evolver", {}) or {}
        self.confidence_threshold: float = float(
            ne_cfg.get("confidence_threshold", 0.7)
        )
        self.max_per_cycle: int = int(ne_cfg.get("max_per_cycle", 10))

        # インターバル: モデルサイズから自動計算 (``memory.llm_call_base_interval``)。
        # ``memory.llm_call_interval`` の直接指定は MemoryConfig (extra=forbid)
        # に無く到達不能だったので読まない。
        base = mc.get("llm_call_base_interval", 1.0)
        self.llm_call_interval: float = compute_llm_call_interval(base, params_b)

        # システムプロンプト: aux_prompt_manager → フォールバック
        self._system_prompt = _FALLBACK_SYSTEM_PROMPT
        if aux_prompt_manager is not None:
            try:
                self._system_prompt = aux_prompt_manager.get_aux_prompt(
                    "note_evolve"
                )
            except (ValueError, KeyError):
                logger.warning(
                    "Failed to get note_evolve prompt from aux_prompt_manager, "
                    "using fallback"
                )

        # memory.jsonl へ LLM 呼び出し/スキップ件数を記録する任意ロガー
        self._debug_logger: DebugLogger | None = debug_logger

        logger.info("LLM call interval: %.2fs (params_b=%.1f)", self.llm_call_interval, params_b)

    async def evolve_notes(
        self,
        short_term: ShortTermMemory,
        long_term: LongTermMemory,
        llm_client,
        should_pause: Callable[[], bool] | None = None,
    ) -> int:
        """evolution_pending なノートの context_description を LLM で生成

        Args:
            should_pause: ``True`` を返したらループを打ち切る協調 yield。
                残りのノートは ``evolution_pending`` のままなので次サイクルが拾う。

        Returns:
            進化させたノート数
        """
        if not self.enabled:
            return 0

        pending = [
            note for note in short_term.notes.values()
            if note.evolution_pending and note.embedding is not None
        ]
        if not pending:
            return 0

        # confidence が閾値以上のノートは LLM 進化をスキップし
        # rule-based evolution (pending フラグを落とすだけ) で済ませる。
        # confidence 属性を持たないノートは保守的に全件 LLM 判定対象にする。
        high_confidence: list[MemoryNote] = []
        llm_targets: list[MemoryNote] = []
        for note in pending:
            conf = getattr(note, "confidence", None)
            if conf is not None and float(conf) >= self.confidence_threshold:
                high_confidence.append(note)
            else:
                llm_targets.append(note)

        rule_based_evolved = 0
        for note in high_confidence:
            # ルールベース evolution: LLM 呼ばず pending フラグだけ落とす。
            # context_description は空のままだが、Sleep-time 次サイクルで
            # 再処理されないため線形肥大が抑えられる。
            note.evolution_pending = False
            rule_based_evolved += 1

        cycle_cap = min(self.batch_size, self.max_per_cycle)
        targets = llm_targets[:cycle_cap]
        skipped_over_cap = max(0, len(llm_targets) - cycle_cap)

        stats: dict = {
            "pending": len(pending),
            "high_confidence_skipped": len(high_confidence),
            "over_cap_skipped": skipped_over_cap,
            "confidence_threshold": self.confidence_threshold,
            "max_per_cycle": self.max_per_cycle,
            "batch_size": self.batch_size,
        }

        if not targets:
            # LLM 呼び出しなし — ヘルスチェック/サーキットブレーカーを経ずに
            # 集計だけ記録して早期リターンする
            stats["llm_calls"] = 0
            stats["llm_evolved"] = 0
            stats["rule_based_evolved"] = rule_based_evolved
            stats["health_skipped"] = False
            self._log_op_stats(stats)
            if self._debug_logger is not None and rule_based_evolved > 0:
                self._debug_logger.log_decision(
                    decision_point="note_evolution_path",
                    chosen="rule_based",
                    candidates=["llm", "rule_based", "health_check_skip"],
                    reason="all_high_confidence",
                    context={
                        "rule_based_evolved": rule_based_evolved,
                        "confidence_threshold": self.confidence_threshold,
                    },
                    scope="cycle",
                )
            return rule_based_evolved

        # 事前ヘルスチェック: 補助タスクが応答不能ならスキップ
        if hasattr(llm_client, "health_check"):
            healthy = await llm_client.health_check()
            if not healthy:
                logger.warning(
                    "Aux task unhealthy, skipping note evolution "
                    "(%d llm targets, %d rule-based evolved)",
                    len(targets), rule_based_evolved,
                )
                stats["llm_calls"] = 0
                stats["llm_evolved"] = 0
                stats["rule_based_evolved"] = rule_based_evolved
                stats["health_skipped"] = True
                self._log_op_stats(stats)
                return rule_based_evolved

        evolved = 0
        consecutive_failures = 0
        total_failures = 0
        total_attempts = 0

        paused = 0
        for note in targets:
            # 協調 yield: チャット生成が走っている間は 1 件ごとに手を止める。
            #
            # llama-server は**全スロットを逐次実行**する (launch_slot_ と
            # release が交互に出る) ため、専有スロット (``background_slot``) は
            # KV を分離するがレイテンシは分離しない。1 呼出 25〜30 秒 ×
            # ``max_per_cycle`` 件が丸ごとユーザーターンの待ち時間に乗る。
            #
            # 実測 (2026-08-27 ライブ監査、Qwen3.8-27B / iGPU):
            #   06:19:21 Full 開始 → 06:21:16〜06:25:28 に 10 件を逐次 LLM
            #   06:23:06 に届いたユーザーターンの応答は 06:26:55 (228,311 ms)
            #
            # ここを止めても Full サイクル自体は完走する (Step 8 のファクト
            # 抽出は別ステップ)。残りは ``evolution_pending`` のままなので
            # 次サイクルが続きから拾う。``on_user_input`` 側の
            # ``worker.cancel()`` は ``_full_forced_run`` のとき意図的に
            # 呼ばれない (2026-08-22 の修正: 止めると Full が一度も完走しない)
            # ので、**そこには頼れない**。
            if should_pause is not None and should_pause():
                paused = len(targets) - evolved - total_failures
                logger.info(
                    "Note evolution paused for the user turn: %d note(s) left "
                    "pending for the next cycle", paused,
                )
                break

            # サーキットブレーカー: 連続3回失敗で残りをスキップ
            if consecutive_failures >= 3:
                logger.warning(
                    "Circuit breaker (consecutive): skipping remaining notes "
                    "after %d consecutive failures",
                    consecutive_failures,
                )
                break

            # サーキットブレーカー: 総失敗率 50% 超過（最低4回試行後）で停止
            if total_attempts >= 4 and total_failures / total_attempts > 0.5:
                logger.warning(
                    "Circuit breaker (failure rate): stopping at %.0f%% failure rate "
                    "(%d/%d attempts)",
                    total_failures / total_attempts * 100,
                    total_failures, total_attempts,
                )
                break

            # 関連ノートを取得して文脈を構築
            context_notes = self._gather_context(note, short_term, long_term)
            description = await self._generate_description(
                note, context_notes, llm_client
            )
            total_attempts += 1
            if description:
                note.context_description = description
                note.evolution_pending = False
                evolved += 1
                consecutive_failures = 0
                logger.info("Evolved note %s", note.id)
            else:
                consecutive_failures += 1
                total_failures += 1

            # llama-server の負荷軽減のためインターバルを挿入
            if self.llm_call_interval > 0:
                await asyncio.sleep(self.llm_call_interval)

        stats["llm_calls"] = total_attempts
        stats["llm_evolved"] = evolved
        stats["llm_failures"] = total_failures
        stats["rule_based_evolved"] = rule_based_evolved
        stats["paused_for_user"] = paused
        stats["health_skipped"] = False
        self._log_op_stats(stats)
        return evolved + rule_based_evolved

    def _log_op_stats(self, stats: dict) -> None:
        """memory.jsonl に note_evolve の LLM 判定削減統計を記録"""
        dl = self._debug_logger
        if dl is None:
            return
        try:
            dl.log_memory_op("note_evolve", stats)
        except Exception as exc:
            logger.warning("log_memory_op(note_evolve) failed: %s", exc)

    def _gather_context(
        self,
        note,
        short_term: ShortTermMemory,
        long_term: LongTermMemory,
    ) -> list[str]:
        """ノートの文脈となる関連テキストを収集

        ``note.links`` が rebuild_links_and_clusters で
        既に張られている場合、それを優先利用する (LLM 呼び出しコストを削減し、
        クラスタ内の文脈を一貫して利用できる)。``links`` が空の場合のみ
        従来通りベクトル検索フォールバックを行う。
        """
        contexts: list[str] = []

        # 事前に張られたリンクを優先利用
        linked_ids = getattr(note, "links", None) or []
        for nid in linked_ids[: self.context_k]:
            related = short_term.notes.get(nid)
            if related is not None and related.id != note.id:
                contexts.append(related.content)
        if len(contexts) >= self.context_k:
            return contexts[: self.context_k]

        if note.embedding is None:
            return contexts

        # STM から関連ノートを取得 (フォールバック)
        stm_results = short_term.retrieve_top_k(note.embedding, k=self.context_k)
        for related_note, score in stm_results:
            if related_note.id != note.id and related_note.content not in contexts:
                contexts.append(related_note.content)

        # LTM からも取得
        ltm_results = long_term.search(note.embedding, top_k=self.context_k)
        for chunk_id, score, text in ltm_results:
            if text not in contexts:
                contexts.append(text)

        return contexts[:self.context_k]

    def rebuild_links_and_clusters(
        self, short_term: ShortTermMemory,
    ) -> dict[str, int]:
        """STM 全体に対してリンク張り直し + クラスタリングを行う (LLM 不使用)。

        - 各ノートの embedding コサイン類似度を全ペアで計算し、
          ``link_threshold`` 以上かつ自分自身を除いた上位 ``link_top_k`` 件を
          ``MemoryNote.links`` にセットする。
        - リンクで連結された連結成分を union-find で求め、各成分の最小ノート ID
          を ``cluster_id`` として全メンバに割り当てる。

        embedding を持たないノート、次元不一致ノートは「孤立」扱いとして
        ``links=[]`` / ``cluster_id=<self.id>`` を割り当てる (孤立クラスタ)。

        Returns:
            ``{"notes": int, "links": int, "clusters": int, "skipped": int}``
        """
        stats = {"notes": 0, "links": 0, "clusters": 0, "skipped": 0}
        if not self.link_rebuild_enabled:
            logger.debug("Link rebuild disabled, skipping")
            return stats

        notes = list(short_term.notes.values())
        stats["notes"] = len(notes)
        if not notes:
            return stats

        # embedding 次元を決定 (最頻出次元を採用 — 過渡期で混在するケースを許容)
        dims = [int(n.embedding.shape[0]) for n in notes if n.embedding is not None]
        if not dims:
            # 全ノートに embedding が無いので孤立化のみ
            for n in notes:
                n.links = []
                n.cluster_id = n.id
            stats["clusters"] = len(notes)
            stats["skipped"] = len(notes)
            return stats
        target_dim = max(set(dims), key=dims.count)

        # 有効ノート (target_dim と一致する embedding を持つ) と孤立ノートを分離
        valid: list[MemoryNote] = []
        valid_vecs: list[np.ndarray] = []
        for n in notes:
            if n.embedding is None or int(n.embedding.shape[0]) != target_dim:
                n.links = []
                n.cluster_id = n.id
                stats["skipped"] += 1
                continue
            valid.append(n)
            valid_vecs.append(n.embedding)

        if not valid:
            stats["clusters"] = stats["skipped"]
            return stats

        # コサイン類似度行列 (embedding は L2 正規化済を想定。
        # 万一未正規化でも動作するよう norm を補正する)
        matrix = np.stack(valid_vecs).astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normalized = matrix / norms
        sim = normalized @ normalized.T  # (N, N)
        # 自己類似度を除外
        np.fill_diagonal(sim, -np.inf)

        n_valid = len(valid)
        link_lists: list[list[str]] = [[] for _ in range(n_valid)]
        # 各行で上位 K 件をしきい値フィルタしつつ採用
        k = min(self.link_top_k, n_valid - 1) if n_valid > 1 else 0
        if k > 0:
            # argpartition で上位 K 候補を高速取得 → 厳密ソート
            top_idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
            for i in range(n_valid):
                cand = top_idx[i]
                # 類似度降順に並べ直し
                cand_sorted = cand[np.argsort(-sim[i, cand])]
                accepted: list[str] = []
                for j in cand_sorted:
                    score = float(sim[i, j])
                    if score < self.link_threshold:
                        break
                    accepted.append(valid[int(j)].id)
                link_lists[i] = accepted

        # union-find でクラスタリング
        parent = list(range(n_valid))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            # 小さい方を親にして決定論的に
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

        id_to_idx = {n.id: i for i, n in enumerate(valid)}
        if self.clustering_enabled:
            for i, links in enumerate(link_lists):
                for nid in links:
                    j = id_to_idx.get(nid)
                    if j is not None:
                        union(i, j)

        # クラスタ代表 ID を確定 (連結成分内の最小 note id)
        component_members: dict[int, list[int]] = {}
        for i in range(n_valid):
            root = find(i)
            component_members.setdefault(root, []).append(i)

        cluster_id_by_idx: dict[int, str] = {}
        for members in component_members.values():
            rep_id = min(valid[i].id for i in members)
            for i in members:
                cluster_id_by_idx[i] = rep_id

        # 結果書き戻し
        link_total = 0
        for i, n in enumerate(valid):
            n.links = link_lists[i]
            link_total += len(link_lists[i])
            if self.clustering_enabled:
                n.cluster_id = cluster_id_by_idx[i]
            else:
                # クラスタリング無効時は自分自身を cluster_id に
                n.cluster_id = n.id

        stats["links"] = link_total
        # クラスタ数 = 有効ノートの連結成分数 + 孤立ノート数
        stats["clusters"] = len(component_members) + stats["skipped"]
        # キャッシュ無効化 (lightmem_score は変えていないが links / cluster_id を反映)
        mark = getattr(short_term, "mark_dirty", None)
        if callable(mark):
            mark()
        logger.info(
            "Rebuilt links/clusters: notes=%d valid=%d links=%d clusters=%d skipped=%d",
            stats["notes"], n_valid, stats["links"], stats["clusters"], stats["skipped"],
        )
        return stats

    async def _generate_description(
        self,
        note,
        context_notes: list[str],
        llm_client,
    ) -> str | None:
        """LLM で context_description を生成

        空レスポンスの場合は1回リトライする。
        """
        # コンテンツ長を制限（コンテキストウィンドウ超過防止）
        note_content = note.content[:MAX_NOTE_CONTENT_LEN]

        context_block = ""
        if context_notes:
            formatted = "\n".join(
                f"- {c[:MAX_CONTEXT_CONTENT_LEN]}" for c in context_notes
            )
            context_block = f"\n\n関連メモリ:\n{formatted}"

        messages = [
            {
                "role": "system",
                "content": self._system_prompt,
            },
            {
                "role": "user",
                "content": f"メモリノート:\n{note_content}{context_block}",
            },
        ]

        for attempt in range(_EMPTY_RESPONSE_MAX_RETRIES + 1):
            try:
                result = await llm_client.generate(
                    messages, stream=False, max_tokens=256,
                    purpose="note_evolution",
                    id_slot=getattr(llm_client, 'background_slot', -1),
                )
                content = result["choices"][0]["message"]["content"]
                stripped = content.strip()
                if stripped:
                    return stripped
                # 空レスポンス: リトライ可能ならリトライ
                if attempt < _EMPTY_RESPONSE_MAX_RETRIES:
                    logger.warning(
                        "Empty response for note %s (attempt %d/%d), retrying",
                        note.id, attempt + 1, _EMPTY_RESPONSE_MAX_RETRIES + 1,
                    )
                else:
                    logger.warning(
                        "Empty response for note %s after %d attempts, giving up",
                        note.id, attempt + 1,
                    )
            except Exception as e:
                logger.warning("Failed to evolve note %s: %s", note.id, e)
                return None

        return None
