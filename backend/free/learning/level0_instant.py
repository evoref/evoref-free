"""Level 0 即時学習: 経験バッファ"""

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from backend.free.learning.json_state_store import JsonPayload, JsonStateStore
from backend.log_config import get_logger
from backend.utils import utc_now

logger = get_logger("learning.level0")

MAX_ENTRIES = 1000

#: バッファ表示・集計用の応答要約の最大長 (従来の response[:200])。
RESPONSE_SUMMARY_CAP = 200
#: few-shot 採用時に保持する全文応答の上限。response_summary (200字) では
#: 文の途中で切れた応答が few-shot 例として注入されるため、採用候補向けに
#: より長い全文を文境界で切り詰めて保持する。experience.json 肥大を避けるため
#: 青天井にはしない (cap × MAX_ENTRIES が永続化サイズの上限)。
RESPONSE_FULL_CAP = 4000


def truncate_at_boundary(text: str, cap: int) -> str:
    """``cap`` 字以内で文境界 (。．.!?改行) を優先して切り詰める。

    cap 未満は無加工。末尾付近 (cap*0.6 以降) に文境界があればそこで切り、
    無ければハード cap。few-shot 応答が文の途中でぶつ切りになるのを避ける。
    """
    if len(text) <= cap:
        return text
    head = text[:cap]
    floor = int(cap * 0.6)
    best = -1
    for ch in ("。", "．", ".", "!", "?", "\n"):
        idx = head.rfind(ch)
        if idx >= floor and idx > best:
            best = idx
    return head[: best + 1] if best >= floor else head


@dataclass
class FeedbackSignals:
    """暗黙的フィードバックシグナル"""
    conversation_ended: bool = False
    # ターン成否の SSOT: "success" | "partial" | "failed"。
    # response の [failed] マーカー / step_credits 全 0 / ルーティング
    # false_positive から FeedbackCollector が決定論導出する。
    # tool_routing_success / long_form_success と矛盾する場合は failed 側に
    # 倒し、偽成功が Level 1 の正例学習へ伝播しないようにする。
    turn_outcome: str = "success"
    rephrased_query: bool = False
    rag_used: bool = False
    rag_source: str | None = None
    rag_top1_score: float | None = None
    agent_loops: int = 0
    user_correction: str | None = None
    # "hardcoded" | "prev_failed" | "same_target" | None。旧 "learned"
    # (学習パターン照合) は 2026-07-21 廃止 — 過去データには残存しうる
    correction_detected_by: str | None = None
    # アシスタント自身が応答冒頭で前ターンの誤りを撤回したか
    # (「失礼いたしました」「訂正します」等)。ユーザーの字句に依らない
    # 高確度シグナルで、**誤っていたのは 1 つ前のターン**。検出時は
    # FeedbackCollector が直前エントリの turn_outcome を failed へ落とす。
    #
    # user_correction とは別枠にする — あちらは「ユーザーの訂正発話そのもの」
    # を保持し critique_synthesizer が本文を引用するため、ユーザーが訂正して
    # いないターンのクエリを入れると引用が破綻する (2026-08-05 ライブ監査で
    # 訂正検出 0/40。アシスタントが「失礼いたしました」と撤回したターンすら
    # 検出されていなかった)。
    assistant_self_retraction: bool = False
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


#: ``_from_payload`` で JSON から復元するシグナルのキー集合。FeedbackSignals の
#: 全フィールドにデフォルト値がある前提 (欠損キーは dataclass 既定にフォールバック)。
_SIGNAL_FIELD_NAMES = frozenset(f.name for f in fields(FeedbackSignals))


@dataclass
class ExperienceEntry:
    """経験バッファの1エントリ"""
    timestamp: str = ""
    mode: str = "chat"
    query: str = ""
    response_summary: str = ""
    # few-shot 採用例用の全文応答 (文境界で RESPONSE_FULL_CAP に切り詰め)。
    # 表示・集計・cvector は response_summary を使い、本フィールドは
    # fewshot_pool.add_from_experiences が切れていない応答例を採るためだけに使う。
    response_full: str = ""
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
            entry.timestamp = utc_now()

        self.entries.append(entry)

        # ローテーション
        if len(self.entries) > self.max_entries:
            overflow = len(self.entries) - self.max_entries
            self.entries = self.entries[overflow:]
            logger.info("Rotated %d old entries", overflow)

    def get_recent(self, n: int = 10) -> list[ExperienceEntry]:
        """直近 n 件取得"""
        return self.entries[-n:]

    def get_failures(self, mode: str | None = None) -> list[ExperienceEntry]:
        """失敗エントリ抽出（rephrased_query=True or user_correction 非 None）

        Args:
            mode: 指定時はそのモード ("chat"/"create") のエントリのみに絞る。
                None (省略、既定) の場合は全モード横断 (後方互換)。
        """
        result = [
            e for e in self.entries
            if e.signals.rephrased_query or e.signals.user_correction is not None
        ]
        if mode is not None:
            result = [e for e in result if e.mode == mode]
        return result

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
                "response_full": entry.response_full,
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
                response_full=d.get("response_full", ""),
                base_model=d.get("base_model") or "",
                embedding_model=d.get("embedding_model", ""),
                cartridge_ids=d.get("cartridge_ids", []),
                # JSON に存在するキーのみ採用 (欠損キーは FeedbackSignals の
                # 既定値に委ねる)。新シグナル追加時もここの編集は不要。
                signals=FeedbackSignals(**{
                    k: signals_data[k]
                    for k in _SIGNAL_FIELD_NAMES
                    if k in signals_data
                }),
            )
            self.entries.append(entry)
        self._mark_loaded_conversations_ended()

    def _mark_loaded_conversations_ended(self) -> int:
        """読み込んだエントリの ``conversation_ended`` を確定させる。

        ``FeedbackCollector.mark_conversation_ended`` は ``_session_entries``
        (**メモリ上のオブジェクト参照リスト**) を辿って印を付けるため、印を付ける前に
        プロセスが落ちると紐付けごと消え、そのエントリは**二度と** ended にならない
        (実測: 69 件中 42 件が未マークのまま死蔵。fitness が 0.528 と 0.806 で
        二分され、学習の選択圧から丸ごと外れていた)。

        永続ファイルにあるエントリは定義上すべて**このプロセスの起動より前**に
        書かれたもので、それを書いたプロセスはもう存在しない。したがって当該会話は
        既に終了している。読み込み時点で確定させるのが正しく、ここが唯一の
        再起動耐性のある地点になる。

        Returns:
            新たに ended を立てた件数。
        """
        marked = 0
        for entry in self.entries:
            if not entry.signals.conversation_ended:
                entry.signals.conversation_ended = True
                marked += 1
        if marked:
            logger.info(
                "Marked %d loaded entries as conversation_ended "
                "(their writing process is gone, so those conversations are over)",
                marked,
            )
        return marked

    def _on_save_success(self, path: Path) -> None:
        logger.info("Saved %d experience entries to %s", len(self.entries), path)

    def _on_load_success(self, path: Path) -> None:
        logger.info("Loaded %d experience entries from %s", len(self.entries), path)
