"""Level 0 即時学習: 経験バッファ"""

import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from backend.free.learning.json_state_store import JsonPayload, JsonStateStore
from backend.log_config import get_logger

logger = get_logger("learning.level0")

MAX_ENTRIES = 1000


@dataclass
class FeedbackSignals:
    """暗黙的フィードバックシグナル"""
    conversation_ended: bool = False
    rephrased_query: bool = False
    rag_used: bool = False
    rag_source: str | None = None
    rag_top1_score: float | None = None
    agent_loops: int = 0
    user_correction: str | None = None
    correction_detected_by: str | None = None  # "hardcoded" | "learned" | None
    perplexity: float | None = None
    # 長文生成シグナル
    long_form_used: bool = False
    long_form_content_type: str | None = None    # "code" | "text"
    long_form_strategy: str | None = None        # "cogwriter" | "recurrent"
    long_form_units_total: int = 0
    long_form_units_completed: int = 0
    long_form_validation_errors: int = 0
    long_form_budget_used_pct: float | None = None
    # ツールルーティングシグナル
    tool_routing_success: bool = False
    tool_routing_false_positive: bool = False
    tool_routing_false_negative: bool = False
    # 長文ルーティングシグナル (router._detect_long_form の学習用)
    # success: 長文分類が成功し generation 完了 → 該当キーワードを強化 + 学習
    # false_positive: long_form 分類されたが短文応答で十分だった → 該当キーワードを減衰
    # false_negative: deliberative 分類されたがユーザが「長文で」等で再要求 → 新キーワード学習
    long_form_success: bool = False
    long_form_false_positive: bool = False
    long_form_false_negative: bool = False
    # MDP ステップクレジット
    step_credits: list[dict] = field(default_factory=list)


@dataclass
class ExperienceEntry:
    """経験バッファの1エントリ"""
    timestamp: str = ""
    mode: str = "chat"
    query: str = ""
    response_summary: str = ""
    base_model: str = ""
    embedding_model: str = ""
    cartridge_ids: list[str] = field(default_factory=list)
    signals: FeedbackSignals = field(default_factory=FeedbackSignals)


class ExperienceBuffer(JsonStateStore):
    """経験バッファ: 毎応答時にエントリを記録"""

    _state_logger = logger

    def __init__(self, max_entries: int = MAX_ENTRIES):
        self.max_entries = max_entries
        self.entries: list[ExperienceEntry] = []

    def record(self, entry: ExperienceEntry) -> None:
        """エントリを追加"""
        if not entry.timestamp:
            entry.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        self.entries.append(entry)

        # ローテーション
        if len(self.entries) > self.max_entries:
            overflow = len(self.entries) - self.max_entries
            self.entries = self.entries[overflow:]
            logger.info("Rotated %d old entries", overflow)

    def get_recent(self, n: int = 10) -> list[ExperienceEntry]:
        """直近 n 件取得"""
        return self.entries[-n:]

    def get_failures(self) -> list[ExperienceEntry]:
        """失敗エントリ抽出（rephrased_query=True or user_correction 非 None）"""
        return [
            e for e in self.entries
            if e.signals.rephrased_query or e.signals.user_correction is not None
        ]

    @property
    def count(self) -> int:
        return len(self.entries)

    @property
    def source_memory_ids(self) -> list[str]:
        """FadeMem ガード用: 空リスト（将来拡張）"""
        return []

    @property
    def pending_memory_ids(self) -> list[str]:
        """FadeMem ガード用: 空リスト（将来拡張）"""
        return []

    # ── 永続化 (JsonStateStore) ──

    def _to_payload(self) -> JsonPayload:
        return [
            {
                "timestamp": entry.timestamp,
                "mode": entry.mode,
                "query": entry.query,
                "response_summary": entry.response_summary,
                "base_model": entry.base_model,
                "embedding_model": entry.embedding_model,
                "cartridge_ids": entry.cartridge_ids,
                "signals": asdict(entry.signals),
            }
            for entry in self.entries
        ]

    def _from_payload(self, payload: JsonPayload) -> None:
        if not isinstance(payload, list):
            raise TypeError(
                f"experience.json must be a list, got {type(payload).__name__}"
            )
        self.entries.clear()
        for d in payload:
            signals_data = d.get("signals", {})
            entry = ExperienceEntry(
                timestamp=d.get("timestamp", ""),
                mode=d.get("mode", "chat"),
                query=d.get("query", ""),
                response_summary=d.get("response_summary", ""),
                base_model=d.get("base_model", ""),
                embedding_model=d.get("embedding_model", ""),
                cartridge_ids=d.get("cartridge_ids", []),
                signals=FeedbackSignals(
                    conversation_ended=signals_data.get("conversation_ended", False),
                    rephrased_query=signals_data.get("rephrased_query", False),
                    rag_used=signals_data.get("rag_used", False),
                    rag_source=signals_data.get("rag_source"),
                    rag_top1_score=signals_data.get("rag_top1_score"),
                    agent_loops=signals_data.get("agent_loops", 0),
                    user_correction=signals_data.get("user_correction"),
                    correction_detected_by=signals_data.get("correction_detected_by"),
                    perplexity=signals_data.get("perplexity"),
                    long_form_used=signals_data.get("long_form_used", False),
                    long_form_content_type=signals_data.get("long_form_content_type"),
                    long_form_strategy=signals_data.get("long_form_strategy"),
                    long_form_units_total=signals_data.get("long_form_units_total", 0),
                    long_form_units_completed=signals_data.get("long_form_units_completed", 0),
                    long_form_validation_errors=signals_data.get("long_form_validation_errors", 0),
                    long_form_budget_used_pct=signals_data.get("long_form_budget_used_pct"),
                    tool_routing_success=signals_data.get("tool_routing_success", False),
                    tool_routing_false_positive=signals_data.get("tool_routing_false_positive", False),
                    tool_routing_false_negative=signals_data.get("tool_routing_false_negative", False),
                    long_form_success=signals_data.get("long_form_success", False),
                    long_form_false_positive=signals_data.get("long_form_false_positive", False),
                    long_form_false_negative=signals_data.get("long_form_false_negative", False),
                    step_credits=signals_data.get("step_credits", []),
                ),
            )
            self.entries.append(entry)

    def _on_save_success(self, path: Path) -> None:
        logger.info("Saved %d experience entries to %s", len(self.entries), path)

    def _on_load_success(self, path: Path) -> None:
        logger.info("Loaded %d experience entries from %s", len(self.entries), path)
