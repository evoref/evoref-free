"""デバッグロガー: LLM リクエスト・RAG 結果・メモリ状態・学習サイクルの記録

内部書き込みは ``backend.structlog_config.DebugLogSink`` に委譲
する。``trace_id`` 自動付与・redaction (API キー / Bearer / メール) ・日付
分割 + 世代ローテ + retention は sink 側の structlog 互換 processor チェーン
で集約管理する。

develop モードを 3 段階 (``debug`` / ``investigate`` / ``evolve``)
に再設計。``__init__`` は ``develop_level: DevelopLevel`` を受け取り、各
JSONL カテゴリの enabled / max_log_mb / log_retention_days を内部マップから
導出する。``config.yaml`` の ``debug:`` セクションは廃止 (起動時に拒否)。

公開 API (``log_request`` / ``log_assist_request`` / ``log_rag_result`` /
``log_embedding`` / ``log_rerank_result`` / ``log_assist_judge`` /
``log_memory_state`` / ``log_memory_op`` / ``log_learning_cycle`` /
``log_long_form_event`` / ``log_agent_trace_event`` / ``log_request_timing`` /
``log_retry_attempt`` / ``log_assist_json_repair`` / ``log_rerank_skipped``)
のシグネチャは変更しないため、注入点 (CLAUDE.md §9 / `.claude/rules/backend.md`
チェックリスト) の修正は不要。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from backend.log_config import DevelopLevel, get_logger
from backend.structlog_config import DebugLogSink
from backend.utils import utc_now as _now

logger = get_logger("debug_logger")


# develop モードレベル別のログ設定
#
# - ``categories``: 有効化する JSONL カテゴリ集合。
#     - debug:        requests のみ (即時 BUG 解析向け)
#     - investigate:  requests / rag / memory / long_form (人間レビュー向け)
#     - evolve:       既存 6 系統すべて (loop 自己学習向け、PR-C で
#                     decision / outcome 追加予定)
# - ``max_log_mb``:  ファイルあたりのローテ閾値 (MB)。
# - ``retention_days``: 古いログの自動削除日数 (0 で無効化)。
#
# ``off`` は通常起動 (DebugLogger 無効、JSONL 出力なし)。
_LEVEL_CONFIGS: dict[DevelopLevel, dict[str, Any]] = {
    "off": {
        "enabled": False,
        "categories": frozenset(),
        "max_log_mb": 100,
        "retention_days": 7,
    },
    "debug": {
        "enabled": True,
        "categories": frozenset({"requests"}),
        "max_log_mb": 100,
        "retention_days": 3,
    },
    "investigate": {
        "enabled": True,
        "categories": frozenset({"requests", "rag", "memory", "long_form"}),
        "max_log_mb": 200,
        "retention_days": 7,
    },
    "evolve": {
        "enabled": True,
        "categories": frozenset({
            "requests", "rag", "memory", "learning", "long_form", "agent_trace",
            # 埋込点は PR-C2 (10 decision sites) / PR-C3 (5 outcome sites) で追加予定。
            "decision", "outcome",
        }),
        "max_log_mb": 500,
        "retention_days": 30,
    },
}


# develop モード時の DebugLogger 出力先 (project_root 相対)。
# `--isolate-data` (Pro) 指定時は ``apply_data_isolation`` が
# ``local_paths.logs_dir`` を ``local/test_data/logs/`` 系へ差し替えるため、
# ``project_root`` 側で吸収される。
_DEFAULT_LOG_SUBDIR = ("local", "logs", "debug")


class DebugLogger:
    """デバッグ情報を JSONL に記録するロガー (structlog ベース)。

    ``develop_level`` で有効/無効と各カテゴリ enabled を制御
    内部書き込みは ``DebugLogSink`` に委譲し、本クラスは公開 API の整形のみ
    を担う。
    """

    def __init__(
        self,
        develop_level: DevelopLevel = "off",
        project_root: Path | None = None,
    ) -> None:
        if develop_level not in _LEVEL_CONFIGS:
            raise ValueError(
                f"unknown develop_level: {develop_level!r}; "
                f"expected one of {sorted(_LEVEL_CONFIGS)}",
            )
        self.develop_level: DevelopLevel = develop_level
        cfg = _LEVEL_CONFIGS[develop_level]
        categories: frozenset[str] = cfg["categories"]

        self.enabled: bool = bool(cfg["enabled"])
        self.log_requests: bool = "requests" in categories
        self.log_rag: bool = "rag" in categories
        self.log_memory: bool = "memory" in categories
        self.log_learning: bool = "learning" in categories
        self.log_long_form: bool = "long_form" in categories
        self.log_agent_trace: bool = "agent_trace" in categories
        # フラグ名は既存パターン (``log_request`` メソッド ↔ ``log_requests``
        # フラグ) と整合性を取り複数形に揃える。
        self.log_decisions: bool = "decision" in categories
        self.log_outcomes: bool = "outcome" in categories

        self.max_log_mb: int = int(cfg["max_log_mb"])
        # 世代数は固定 (3) — develop_level で可変にする要件はない
        self.max_log_generations: int = 3
        self.log_retention_days: int = int(cfg["retention_days"])

        if project_root is None:
            project_root = Path(__file__).parent.parent
        self.log_dir: Path = project_root.joinpath(*_DEFAULT_LOG_SUBDIR)

        if self.enabled:
            self._sink: DebugLogSink | None = DebugLogSink(
                log_dir=self.log_dir,
                max_log_mb=self.max_log_mb,
                max_log_generations=self.max_log_generations,
                log_retention_days=self.log_retention_days,
            )
        else:
            self._sink = None

    def _emit(self, category: str, payload: Mapping[str, Any]) -> None:
        """``category`` 別 JSONL に書き込む共通ヘルパ。"""
        sink = self._sink
        if sink is None:
            return
        sink.emit(category, payload)

    # ------------------------------------------------------------------
    # requests.jsonl
    # ------------------------------------------------------------------

    def log_request(
        self,
        n: int,
        messages: list[dict],
        response: str,
    ) -> None:
        """LLM リクエスト/レスポンスを記録"""
        if not self.enabled or not self.log_requests:
            return
        self._emit("requests", {
            "n": n,
            "timestamp": _now(),
            "messages_count": len(messages),
            "messages_preview": [
                {"role": m["role"], "content": m["content"][:100]}
                for m in messages[:3]
            ],
            "response_preview": response[:200],
        })

    def log_assist_request(
        self,
        messages_count: int,
        response_preview: str,
        elapsed_sec: float,
        *,
        purpose: str = "",
        priority: str = "",
        resolved_timeout: float | None = None,
        cache_metrics: dict | None = None,
        response_format_used: bool = False,
    ) -> None:
        """アシストモデルのリクエスト/レスポンスを記録

        Args:
            priority: 用途別セマフォ選択
                ``realtime`` / ``background`` / ``learning`` のいずれか。
                ``AssistModelClient`` 以外から呼び出される場合のみ空文字列。
            resolved_timeout: 実際に適用されたタイムアウト秒
                purpose 別 override (``assist_model.timeouts`` または
                ``PURPOSE_TIMEOUT_DEFAULTS``) と既定値のどちらが採用されたか
                を回帰追跡できるようにする。``None`` なら記録しない。
            cache_metrics: llama-server の KV キャッシュ命中率指標
。``{"prompt_n": int, "cache_n": int, "hit_ratio": float}``
                の dict を受け取り、``cache`` フィールドとして JSONL に埋め込む。
                ``cache_prompt=True`` を付けたリクエストで効果測定する用途。
                ``None`` なら記録しない。
            response_format_used: GBNF/json_schema 制約サンプリングを適用
                したか。``True`` なら purpose 別 Pydantic schema
                由来の ``response_format`` が ``/v1/chat/completions`` に付与
                されたことを意味する。フラグ無効化 / 古い build 経路と切り
                分けるための効果測定用フィールド。
        """
        if not self.enabled or not self.log_requests:
            return
        entry: dict = {
            "timestamp": _now(),
            "model": "assist",
            "purpose": purpose,
            "priority": priority,
            "messages_count": messages_count,
            "response_preview": response_preview[:200],
            "elapsed_sec": round(elapsed_sec, 3),
            "response_format_used": bool(response_format_used),
        }
        if resolved_timeout is not None:
            entry["resolved_timeout"] = round(float(resolved_timeout), 3)
        if cache_metrics:
            entry["cache"] = cache_metrics
        self._emit("requests", entry)

    def log_retry_attempt(
        self,
        *,
        backend: str,
        purpose: str,
        attempt: int,
        wait_sec: float,
        exception: str,
        status_code: int | None,
        trace_id: str = "",
    ) -> None:
        """tenacity リトライ発火を ``requests.jsonl`` に記録

        ``async_retry_http_call`` の ``before_sleep`` callback から呼ばれ、
        backend (``base`` / ``assist`` / ``embedding`` / ``reranker``) ごとの
        リトライ頻度・原因例外・status code をモニタする。

        Args:
            backend: リトライを発火させた HTTP クライアントの種別。
                ``"base"`` (LocalClient) / ``"assist"`` (AssistModelClient) /
                ``"embedding"`` (LlamaCppEmbedder) /
                ``"reranker"`` (LlamaCppReranker) のいずれか。
            purpose: assist リクエストの purpose 文字列。
                base / embedding / reranker では空文字列。
            attempt: 1-based attempt 番号 (リトライ発火直前の試行回数)。
            wait_sec: 次の試行までの待機秒数 (jitter 込みの実値)。
            exception: 直前の例外クラス名 (``ReadTimeout`` / ``ConnectError`` /
                ``HTTPStatusError`` / ``_EmptyResponseError`` 等)。
            status_code: ``HTTPStatusError`` の場合の HTTP ステータスコード。
                該当しない場合は ``None``。
            trace_id: 呼出元の trace_id (空文字列で省略)。``DebugLogSink``
                の ``_trace_id_processor`` が ``contextvars`` から自動付与
                するため通常は不要だが、リトライは ``contextvars`` の外
                から呼ばれる可能性があるため明示的に上書きできる。
        """
        if not self.enabled or not self.log_requests:
            return
        entry: dict = {
            "timestamp": _now(),
            "op": "retry",
            "backend": backend,
            "attempt": int(attempt),
            "wait_sec": round(float(wait_sec), 4),
            "exception": exception,
        }
        if purpose:
            entry["purpose"] = purpose
        if status_code is not None:
            entry["status_code"] = int(status_code)
        if trace_id:
            entry["trace_id"] = trace_id
        self._emit("requests", entry)

    def log_assist_json_repair(
        self,
        *,
        purpose: str,
        list_key: str | None,
        raw_preview: str,
        repaired_preview: str,
    ) -> None:
        """assist 応答が json-repair (戦略 4) で復旧されたことを記録

        ``response_format=json_schema`` を適用しても
        ``max_tokens`` / ``reasoning_budgets`` 到達による truncation、Pro
        外部 API (GBNF 非対応)、``--skip-chat-parsing`` の
        raw content 経路では構造的に不完全な JSON が返ることがある。
        ``json_extract.extract_json_object`` の戦略 4 が該当ケースを修復
        した場合、本メソッドで ``op="json_repair"`` エントリを
        ``requests.jsonl`` に追記し、GBNF 採用後の必要性評価と prompt /
        ``reasoning_budget`` 調整の機会を可視化する。

        ``log_assist_request`` とは別エントリだが、``DebugLogSink``
        が ``trace_id`` を自動付与するため、同一 trace_id で時系列に
        並べることで repair 対象のリクエストを特定できる。

        Args:
            purpose: 呼出元 purpose 文字列 (``retrieval_quality_judge`` 等)。
                未指定の汎用 ``generate_json`` 呼出では空文字列。
            list_key: ``extract_json_object`` の ``list_key`` 引数。
                裸配列応答を dict ラップする際のキー。``None`` の場合は
                JSONL に空文字列として記録する。
            raw_preview: 修復前の生 content (先頭 200 文字)。
            repaired_preview: 修復後の dict を文字列化した結果 (先頭 200 文字)。
        """
        if not self.enabled or not self.log_requests:
            return
        self._emit("requests", {
            "timestamp": _now(),
            "model": "assist",
            "op": "json_repair",
            "purpose": purpose,
            "list_key": list_key or "",
            "raw_preview": raw_preview[:200],
            "repaired_preview": repaired_preview[:200],
        })

    def log_request_timing(
        self,
        timing: dict[str, float],
        *,
        agent_layer: str = "",
        tokens_generated: int = 0,
    ) -> None:
        """リクエスト処理のステージ別タイミング内訳を記録"""
        if not self.enabled or not self.log_requests:
            return
        self._emit("requests", {
            "timestamp": _now(),
            "op": "timing",
            "agent_layer": agent_layer,
            "tokens_generated": tokens_generated,
            "timing": timing,
        })

    def log_kv_cache(self, *, tokens_prompt: int, tokens_cached: int) -> None:
        """KV キャッシュヒット状況を記録（llama-server usage.prompt_tokens_details から）"""
        if not self.enabled or not self.log_requests:
            return
        self._emit("requests", {
            "timestamp": _now(),
            "op": "kv_cache",
            "tokens_prompt": tokens_prompt,
            "tokens_cached": tokens_cached,
        })

    # ------------------------------------------------------------------
    # rag.jsonl
    # ------------------------------------------------------------------

    def log_rag_result(
        self,
        n: int,
        query: str,
        chunks: list[tuple[str, float, str]],
        scores: list[float] | None = None,
    ) -> None:
        """RAG 検索結果を記録"""
        if not self.enabled or not self.log_rag:
            return
        self._emit("rag", {
            "n": n,
            "timestamp": _now(),
            "query": query[:100],
            "num_chunks": len(chunks),
            "top_scores": [c[1] for c in chunks[:5]],
        })

    def log_embedding(
        self,
        batch_size: int,
        backend: str,
        elapsed_sec: float,
        *,
        is_query: bool = False,
        cache_hit: bool = False,
    ) -> None:
        """埋め込み生成の計測結果を記録"""
        if not self.enabled or not self.log_rag:
            return
        self._emit("rag", {
            "timestamp": _now(),
            "op": "embedding",
            "backend": backend,
            "batch_size": batch_size,
            "is_query": is_query,
            "cache_hit": cache_hit,
            "elapsed_sec": round(elapsed_sec, 4),
        })

    def log_rerank_result(
        self,
        query_preview: str,
        doc_count: int,
        top_scores: list[float],
        elapsed_sec: float,
    ) -> None:
        """リランカーの実行結果を記録"""
        if not self.enabled or not self.log_rag:
            return
        self._emit("rag", {
            "timestamp": _now(),
            "op": "rerank",
            "query_preview": query_preview[:100],
            "doc_count": doc_count,
            "top_scores": [round(s, 4) for s in top_scores[:5]],
            "elapsed_sec": round(elapsed_sec, 4),
        })

    def log_rerank_skipped(
        self,
        query_preview: str,
        candidates_count: int,
        reason: str,
        *,
        top_score: float = 0.0,
        gap: float = 0.0,
        source: str = "",
    ) -> None:
        """Reranker quality-aware skip を記録

        hybrid top score / gap / 候補数による skip 条件で reranker を
        短絡した際に ``rag.jsonl`` に ``op="rerank_skip"`` として書き出す。
        skip 率を後追いすることで、しきい値チューニングと品質回帰の
        モニタリングを可能にする。

        Args:
            query_preview: クエリ先頭 100 文字 (プライバシー保護のため切詰)。
            candidates_count: skip 判定時点の融合後候補数。
            reason: ``high_score`` / ``large_gap`` / ``few_candidates`` のいずれか。
            top_score: 融合後 top-1 スコア (未算出時は 0.0)。
            gap: top-1 と top-2 のスコア差 (候補 1 件以下なら 0.0)。
            source: 呼び出し経路識別子 (``hybrid_retriever`` / ``unified_search``)。
        """
        if not self.enabled or not self.log_rag:
            return
        entry: dict = {
            "timestamp": _now(),
            "op": "rerank_skip",
            "query_preview": query_preview[:100],
            "candidates_count": candidates_count,
            "reason": reason,
            "top_score": round(float(top_score), 4),
            "gap": round(float(gap), 4),
        }
        if source:
            entry["source"] = source
        self._emit("rag", entry)

    def log_assist_judge(
        self,
        *,
        query_preview: str,
        rule_based_quality: str,
        used: bool,
        final_quality: str,
        session_count: int,
        query_count: int,
        skipped_reason: str = "",
        elapsed_sec: float | None = None,
        source: str = "unified_search",
    ) -> None:
        """Self-RAG assist_judge の発火/スキップ結果を記録

        ``rag.jsonl`` に ``op="assist_judge"`` エントリとして書き出す。
        セッション上限 / クエリ上限 / only_when_quality で skip された
        ケースも ``used=False`` + ``skipped_reason=<理由>`` として記録し、
        skip 率の追跡とチューニングに使う。

        Args:
            query_preview: クエリ先頭 100 文字 (プライバシー保護のため切詰)。
            rule_based_quality: ルールベース判定の結果 ("high"/"medium"/"low")。
            used: 実際にアシスト LLM を呼んだかどうか。
            final_quality: 最終的に採用した品質ラベル。
                ``used=False`` なら ``rule_based_quality`` と同値。
            session_count: 判定前のセッション累計発火回数。
                ``used=True`` の場合、DebugLogger に記録される値は
                "発火前" の数値 (呼び出し側で再計算不要にするため)。
            query_count: 判定前のクエリ内発火回数。
            skipped_reason: skip 理由 (``session_cap`` / ``query_cap`` /
                ``disabled`` / ``quality_not_applicable``)。
                ``used=True`` なら空文字列。
            elapsed_sec: アシスト LLM 呼び出し所要時間 (秒)。``used=False``
                ならレイテンシは発生しないので ``None`` を渡す。
            source: 呼び出し経路識別子 (既定 ``unified_search``)。
        """
        if not self.enabled or not self.log_rag:
            return
        entry: dict = {
            "timestamp": _now(),
            "op": "assist_judge",
            "query_preview": query_preview[:100],
            "rule_based_quality": rule_based_quality,
            "assist_judge_used": bool(used),
            "final_quality": final_quality,
            "session_count": int(session_count),
            "query_count": int(query_count),
        }
        if skipped_reason:
            entry["assist_judge_skipped_reason"] = skipped_reason
        if elapsed_sec is not None:
            entry["elapsed_sec"] = round(float(elapsed_sec), 4)
        if source:
            entry["source"] = source
        self._emit("rag", entry)

    # ------------------------------------------------------------------
    # memory.jsonl
    # ------------------------------------------------------------------

    def log_memory_state(
        self,
        session_id: str,
        memory_dump: dict,
        *,
        semmem_stats: dict | None = None,
    ) -> None:
        """メモリ状態をダンプ

        Args:
            session_id: 呼び出しコンテキスト識別子 (例: ``"unified_search"``、
                ``"startup_bootstrap"``)。
            memory_dump: WM/STM/LTM スナップショット。
            semmem_stats: SemMem ストアの統計。指定された場合
                ``semmem`` フィールドにそのまま埋め込まれる。Tier 別件数や
                bootstrap 結果など可変な指標を扱うため辞書を素通しする。
        """
        if not self.enabled or not self.log_memory:
            return
        entry: dict = {
            "timestamp": _now(),
            "session_id": session_id,
            "working_turns": memory_dump.get("working_turns", 0),
            "stm_notes": memory_dump.get("stm_notes", 0),
            "ltm_vectors": memory_dump.get("ltm_vectors", 0),
        }
        if semmem_stats:
            entry["semmem"] = semmem_stats
        self._emit("memory", entry)

    def log_memory_op(self, op: str, stats: dict) -> None:
        """メモリサイクル内の個別 op 統計を記録

        sleep-time Step 6 (ConflictResolver) / Step 7 (NoteEvolver) などが
        ``op="conflict_resolve"`` / ``op="note_evolve"`` と LLM 呼び出し件数 /
        スキップ件数を ``memory.jsonl`` に書き出し、判定削減率を後追いできる
        ようにする。``log_memory_state`` が状態スナップショットを扱うのに対し
        こちらはサイクル単位の動作サマリを記録する。
        """
        if not self.enabled or not self.log_memory:
            return
        entry: dict = {
            "timestamp": _now(),
            "op": op,
            **stats,
        }
        self._emit("memory", entry)

    # ------------------------------------------------------------------
    # learning.jsonl
    # ------------------------------------------------------------------

    def log_learning_cycle(
        self,
        cycle_num: int,
        data: dict,
    ) -> None:
        """学習サイクルの情報を記録"""
        if not self.enabled or not self.log_learning:
            return
        self._emit("learning", {
            "timestamp": _now(),
            "cycle": cycle_num,
            **data,
        })

    # ------------------------------------------------------------------
    # long_form.jsonl
    # ------------------------------------------------------------------

    def log_long_form_event(self, data: dict) -> None:
        """長文生成のデバッグログ"""
        if not self.enabled or not self.log_long_form:
            return
        self._emit("long_form", {
            "timestamp": _now(),
            **data,
        })

    # ------------------------------------------------------------------
    # agent_trace.jsonl
    # ------------------------------------------------------------------

    def log_agent_trace_event(self, data: dict) -> None:
        """エージェント MDP トレースの記録"""
        if not self.enabled or not self.log_agent_trace:
            return
        self._emit("agent_trace", data)

    # ------------------------------------------------------------------
    # decision.jsonl
    # ------------------------------------------------------------------

    def log_decision(
        self,
        *,
        decision_point: str,
        chosen: str,
        candidates: list[str],
        reason: str = "",
        context: dict | None = None,
        scope: str = "request",
    ) -> None:
        """決定根拠と分岐を ``decision.jsonl`` に構造化記録する

        ``evolve`` レベル限定で書き込まれる loop driver 自己学習用因果ログ。
        ``trace_id`` で同一スコープの ``outcome.jsonl`` エントリと causal join
        できる (``schema_version=1`` 付与)。``debug`` / ``investigate``
        レベルでは ``self.log_decisions=False`` のため no-op。

        Args:
            decision_point: 分岐点の識別子 (例: ``"assist_health_fallback"``,
                ``"quality_gate_action"``, ``"loop_continue_or_abort"``)。
                埋込点ごとにユニークな snake_case 文字列を採用する。
            chosen: 採択された候補の文字列 (例: ``"degraded_local"`` /
                ``"retry"`` / ``"continue"``)。
            candidates: 候補集合 (例:
                ``["external_assist", "local_assist", "degraded_local"]``)。
                空 list を渡すと「分岐先が単一だったが記録すべき決定」を表す。
            reason: 採択理由を表す短い英語識別子
                (例: ``"external_health_check_failed_3x"``,
                ``"sim_below_threshold"``)。redaction を考慮し
                ユーザー入力をそのまま入れない。
            context: 任意のメタ情報 (purpose / elapsed_sec / counts 等)。
                ``_redaction_processor`` が dict を再帰的に走査して
                秘匿パターン (Bearer / API キー / メール / GitHub PAT 等)
                を ``[REDACTED]`` に置換する。
            scope: 因果対の所属スコープ (``"request"`` / ``"cycle"`` /
                ``"loop_iter"`` / ``"sleep"`` / ``"bg_task"``)。
                trace_id が無いコンテキストでも outcome と join できるよう
                明示する。
        """
        if not self.enabled or not self.log_decisions:
            return
        entry: dict = {
            "timestamp": _now(),
            "category": "decision",
            "decision_point": decision_point,
            "chosen": chosen,
            "candidates": list(candidates),
            "scope": scope,
        }
        if reason:
            entry["reason"] = reason
        if context:
            entry["context"] = context
        self._emit("decision", entry)

    # ------------------------------------------------------------------
    # outcome.jsonl
    # ------------------------------------------------------------------

    def log_outcome(
        self,
        *,
        kind: str,
        success: bool,
        duration_ms: float | None = None,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        quality_signals: dict | None = None,
        user_feedback: str | None = None,
    ) -> None:
        """因果結末を ``outcome.jsonl`` に構造化記録する

        ``evolve`` レベル限定で書き込まれる loop driver 自己学習用因果ログ。
        ``trace_id`` で対応する ``decision.jsonl`` エントリと causal join
        できる。``debug`` / ``investigate`` レベルでは ``self.log_outcomes=False``
        のため no-op。

        Args:
            kind: outcome の種別を表す snake_case 識別子。代表値:

                - ``"chat_response"`` (chat エンドポイントの SSE 終端)
                - ``"learning_cycle_l05"`` / ``"learning_cycle_l05_full"``
                  (sleep-time worker Light/Full 完了)
                - ``"learning_cycle_l1"`` / ``"learning_cycle_l2"``
                  (LearningScheduler / Level2Trainer 完了)
                - ``"loop_iter"`` (loop driver の 1 iteration 完了)
                - ``"sleep_worker"`` (sleep-time worker の最終 step 完了)
                - ``"bg_task"`` (asyncio.create_task 配下の任意 background)
            success: 結末が成功か失敗か。``True`` で正常完了、``False`` で
                例外 / quality_gate failure / cancel など。
            duration_ms: 経過時間 (ミリ秒)。``None`` ならフィールド省略。
            tokens_in: 入力トークン数。``None`` ならフィールド省略
                (chat 以外の outcome では基本的に ``None``)。
            tokens_out: 出力トークン数。``None`` ならフィールド省略。
            quality_signals: 品質指標 (例:
                ``{"json_repair_count": 0, "retry_count": 0, "rerank_skipped": false}``)。
                outcome 種別ごとに任意の信号を入れられる。
                ``_redaction_processor`` が dict を再帰走査するため、
                ユーザー入力を含む文字列を入れた場合も自動的にマスクされる。
            user_feedback: ユーザーからの明示的フィードバック
                (``"thumbs_up"`` / ``"thumbs_down"`` / ``"corrected"``)。
                通常は ``None``。
        """
        if not self.enabled or not self.log_outcomes:
            return
        entry: dict = {
            "timestamp": _now(),
            "category": "outcome",
            "kind": kind,
            "success": bool(success),
        }
        if duration_ms is not None:
            entry["duration_ms"] = round(float(duration_ms), 3)
        if tokens_in is not None:
            entry["tokens_in"] = int(tokens_in)
        if tokens_out is not None:
            entry["tokens_out"] = int(tokens_out)
        if quality_signals:
            entry["quality_signals"] = quality_signals
        if user_feedback is not None:
            entry["user_feedback"] = user_feedback
        self._emit("outcome", entry)
