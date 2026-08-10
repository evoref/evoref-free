"""Layer 2: 短期記憶（A-MEM ノート形式 + LightMem スコア）"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from backend.log_config import get_logger
from backend.free.memory.notes.note_builder import get_note_builder
from backend.free.memory.notes.pin_detector import (
    PinTriggers,
    detect_pin,
    get_pin_triggers_for,
)
from backend.free.memory.types import MemoryMode, NoteSource, TaskStatus

logger = get_logger("memory.short_term")

# pin 済みノート (pin_flag=True) への検索スコア加点。sim(0-1) と
# lightmem_score(0-1) の合成値 (最大1.0) に対する加点のため、1.0 を超えても
# 他ノートとの相対順位にのみ影響し実害はない。pin は従来 eviction 保護にのみ
# 使われ retrieve_top_k のランキングには反映されていなかったギャップの是正
# (docs/f_02_memory_system.md 参照)。
_PIN_RETRIEVAL_BOOST: float = 0.15


@dataclass
class MemoryNote:
    """A-MEM Zettelkasten 式ノート"""
    id: str
    content: str
    keywords: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    embedding: np.ndarray | None = None
    #: 埋め込み生成の連続失敗回数。上限に達したノートは Step 1 の対象外にする
    #: (1 件の失敗ノートが毎サイクル Step 1 を落とし、sleep-time 全体が
    #: 前進しなくなるのを防ぐ。詳細は sleep_update._step1_embed_notes)。
    embed_failures: int = 0
    lightmem_score: float = 0.5
    created_at: float = 0.0
    accessed_at: float = 0.0
    access_count: int = 0
    session_id: str = ""
    context_description: str = ""
    evolution_pending: bool = True
    conflict_candidate: bool = False
    conflict_partner_id: str | None = None

    # ── EvorefMem 拡張フィールド ──────────────────
    source: NoteSource = "user"
    """発生源 (user / assistant / system / rag)"""

    confidence: float = 1.0
    """信頼度。user 発話は 1.0、assistant 生成は 0.5 など Builder 側で決める"""

    pin_flag: bool = False
    """ユーザーが明示 / 自動検出で pin 指定したか"""

    pin_reason: str | None = None
    """pin が付いた理由 ("user_explicit" / "auto_detect:keep_in_mind" 等)"""

    extracted_fact_ids: list[str] = field(default_factory=list)
    """このノートから抽出された SemanticFact の ID 一覧"""

    private: bool = False
    """プライベートセッション由来か (semantic 昇格時にスキップ)"""

    mode: MemoryMode = "chat"
    """このノートが生成されたモード"""

    project_id: str | None = None
    """create モード時のプロジェクト ID"""

    is_tool_output: bool = False
    """ツール出力か (WM までのみ保持、STM 以降は除外)"""

    is_code_block: bool = False
    """コードブロック由来か (extraction 完全スキップ対象)"""

    extraction_skipped: bool = False
    """Extractor で抽出スキップされたか"""

    extraction_skip_reason: str | None = None
    """スキップ理由 ("code_block" / "tool_output" / "private" 等)"""

    # ── executable command 学習用 (sleep-time curator が参照) ──
    tool_command: str | None = None
    """このターンで実行された run_command のコマンド文字列 (それ以外は None)"""

    tool_command_name: str | None = None
    """コマンドを実行したツール名 (通常 "run_command"、それ以外は None)"""

    tool_command_success: bool | None = None
    #: 判定層 ("rule" / "assist" / "recall" ...)。executable_command_curator が
    #: "recall" 由来 (= 過去 fact の引き当てで発火) を学習対象から外すために使う。
    #: これが無いと「誤発火 → 成功記録 → fact 延命 → また誤発火」で自己強化する。
    tool_command_source: str | None = None
    """コマンド実行が成功したか (出力が "Error:" prefix でない)。command 無しは None"""

    tool_command_query: str | None = None
    """このコマンドを発火させたユーザークエリ (executable_command_curator の対応付け用)。

    STM は**選択的に吸収された部分集合**であり会話の完全な転写ではない。
    curator が「直前で最も近い user note」を走査して対応付けると、当該ターンの
    user note が吸収されていない場合に **別ターンのクエリ**と結び付く
    (2026-08-05 実測: 日付コマンドが「富士山の標高は何メートルですか。」の
    答えとして success_avg=1.0 で保存され、類似クエリで日付コマンドが発火する
    状態になっていた)。発火時点で確定しているクエリをそのまま持たせて、
    走査による推測をなくす。
    """

    # ── 統合追加フィールド ────────────────────────────────────────
    task_status: TaskStatus | None = None
    """task ファクト lifecycle 用ステータス (open/in_progress/done/failed)"""

    task_id: str | None = None
    """関連する task ファクト ID (自律ループ駆動用)"""

    depends_on: list[str] = field(default_factory=list)
    """依存する task / fact ID 一覧"""

    failure_signature: str | None = None
    """失敗パターンシグネチャ (failure_pattern 即時記録用)"""

    trace_id: str | None = None
    """MDP / リクエスト相関 ID (contextvar から自動付与される想定)"""

    # ── A-MEM ノートリンク + クラスタリング ─────────
    links: list[str] = field(default_factory=list)
    """関連ノート ID 一覧 (sleep-time Step 7 の rebuild_links_and_clusters で更新)。
    類似度しきい値以上の上位 K 件 (自分自身を除く)。"""

    cluster_id: str | None = None
    """所属クラスタ ID。``rebuild_links_and_clusters`` が union-find で算出する。
    None の場合は未クラスタリング (孤立ノートまたは未実行)。"""

    # ── sleep-time curator の処理済みマーカー ─────────
    url_curated_at: float | None = None
    """url_curator (Step 8.5) がこの user note を処理済みにした時刻。
    None は未処理。次サイクルで同一ペアを再採点・再記録しないための冪等マーカー。"""

    command_curated_at: float | None = None
    """executable_command_curator (Step 8.6) がこの assistant note を処理済みに
    した時刻。None は未処理。次サイクルで同一コマンドを再記録しないためのマーカー。"""

    # ── conflict 解決の失敗 quarantine マーカー ─────────
    conflict_fail_count: int = 0
    """このノートを含むペアが LLM マージに連続失敗した回数。閾値到達で
    ``conflict_cooldown_until`` を設定し、次サイクル以降の conflict 検出から
    一定時間除外する (同一ペアを毎サイクル再試行して circuit breaker を起こし
    続ける livelock の防止)。マージ成功で 0 にリセットされる。"""

    conflict_cooldown_until: float | None = None
    """conflict 検出を再開してよい時刻 (float epoch)。``None`` または現在時刻
    超過で検出対象に戻る。``url_curated_at`` 等と同じ float epoch マーカー。"""


class ShortTermMemory:
    """Layer 2: A-MEM ノート + LightMem スコア"""

    def __init__(
        self,
        config: dict,
        *,
        triggers_dir: str | Path | None = None,
    ) -> None:
        mem = config.get("memory", {})
        self.max_notes: int = mem.get("short_term_max_notes", 100)
        self.notes: dict[str, MemoryNote] = {}
        self._cache: list[MemoryNote] = []
        self._cache_dirty: bool = False

        # ── 自動 Pin 検出設定 ──
        pin_cfg = mem.get("pin", {}) or {}
        self._pin_auto_detect: bool = bool(pin_cfg.get("auto_detect", True))
        # Pin トリガ辞書は user override (``<triggers_dir>/pin_triggers.yaml``) →
        # package 同梱 default の 2 段階で解決する。``triggers_dir=None`` の
        # 場合は override を使わず default のみ (テスト / 最小 config 用途)。
        self._pin_triggers_dir: str | Path | None = triggers_dir
        # NOTE: config キー ``memory.pin.auto_detect_confirm`` は将来の
        # 「自動 pin 検出時にユーザー確認を挟む」フローのための予約 (未実装)。
        # 現状この値を消費するコードは無いため、ここでは読み込まない
        # (config.yaml.example には予約キーとして残す)。確認フローを実装する際に
        # 本クラス or pin_detector で参照する。

    def _load_pin_triggers(self) -> PinTriggers:
        """Pin トリガ辞書をプロセス内シングルトンから取得 (パスごと cache)。"""
        return get_pin_triggers_for(self._pin_triggers_dir)

    def absorb(self, turn: dict, session_id: str) -> MemoryNote | None:
        """ワーキングメモリからターンを吸収してノート化

        ``turn`` 辞書から以下のメタデータを読み取り、適切なモード別
        ``NoteBuilder`` を選択して ``MemoryNote`` を構築する:

        - ``role`` (``user`` / ``assistant``)
        - ``mode`` (``chat`` / ``create``) — 省略時は ``chat``
        - ``project_id`` — create モード時のプロジェクト ID
        - ``source`` (``user`` / ``assistant`` / ``system`` / ``rag``)
        - ``is_tool_output`` — ``True`` の場合 STM 以降は除外 (絶対に保存しない)

        Returns:
            生成された ``MemoryNote``。``is_tool_output=True`` の場合は STM
            以降は除外する仕様 のため ``None`` を返す
        """
        content = turn.get("content") or ""
        role = turn.get("role", "user")
        mode: MemoryMode = turn.get("mode", "chat")
        project_id: str | None = turn.get("project_id")
        source: NoteSource | None = turn.get("source")
        is_tool_output: bool = bool(turn.get("is_tool_output", False))
        # プライベートセッション。``private=True`` のターンは
        # ``MemoryNote.private=True`` で吸収され、Step 8 Extractor / LTM 昇格は
        # ``BaseExtractor.is_eligible`` および ``LightMemScorer.evict_low_score``
        # 側でスキップされる。
        private: bool = bool(turn.get("private", False))

        # ツール出力は WM までで止め、STM 以降には残さない
        if is_tool_output:
            logger.debug(
                "absorb: skip tool_output turn (mode=%s, session=%s, len=%d)",
                mode, session_id, len(content),
            )
            return None

        builder = get_note_builder(mode)
        data = builder.build(
            content,
            session_id,
            role=role,
            source=source,
            mode=mode,
            project_id=project_id,
            is_tool_output=is_tool_output,
        )

        # ── 自動 Pin 検出 ──
        # ユーザー発話 (role=user / source=user) のみ自動 pin 対象。
        # assistant 生成・rag・system は対象外
        # コードブロック / ツール出力もスキップ (extraction 完全スキップに準拠)。
        pin_flag = False
        pin_reason: str | None = None
        if (
            self._pin_auto_detect
            and not data["is_code_block"]
            and not data["is_tool_output"]
            and data["source"] == "user"
        ):
            triggers = self._load_pin_triggers()
            if not triggers.empty:
                detection = detect_pin(content, mode, triggers)
                if detection.should_pin:
                    pin_flag = True
                    pin_reason = detection.reason
                elif detection.negated:
                    # 否定マッチは pin しないが理由を残す (デバッグ可視化用)
                    pin_reason = detection.reason

        # プライベートターンは extraction を強制スキップ
        extraction_skipped = data["extraction_skipped"] or private
        extraction_skip_reason = data["extraction_skip_reason"]
        if private and not extraction_skip_reason:
            extraction_skip_reason = "private"

        note = MemoryNote(
            id=data["id"],
            content=data["content"],
            keywords=data["keywords"],
            tags=data["tags"],
            lightmem_score=data["lightmem_score"],
            created_at=data["created_at"],
            accessed_at=data["accessed_at"],
            access_count=data["access_count"],
            session_id=data["session_id"],
            confidence=data["confidence"],
            source=data["source"],
            mode=data["mode"],
            project_id=data["project_id"],
            is_tool_output=data["is_tool_output"],
            is_code_block=data["is_code_block"],
            extraction_skipped=extraction_skipped,
            extraction_skip_reason=extraction_skip_reason,
            pin_flag=pin_flag,
            pin_reason=pin_reason,
            private=private,
            tool_command=turn.get("tool_command"),
            tool_command_name=turn.get("tool_command_name"),
            tool_command_success=turn.get("tool_command_success"),
            tool_command_source=turn.get("tool_command_source"),
            tool_command_query=turn.get("tool_command_query"),
        )
        self.notes[note.id] = note
        self._cache_dirty = True
        logger.info(
            "Absorbed note %s (mode=%s, tags=%s, keywords=%s, code_block=%s, "
            "pin_flag=%s, pin_reason=%s)",
            note.id, note.mode,
            json.dumps(note.tags, ensure_ascii=False),
            json.dumps(note.keywords[:5], ensure_ascii=False),
            note.is_code_block,
            note.pin_flag,
            note.pin_reason,
        )
        return note

    def retrieve_top_k(self, query_vec: np.ndarray, k: int = 3) -> list[tuple[MemoryNote, float]]:
        """スコア加重検索: ベクトル類似度 × LightMem スコア"""
        if not self.notes:
            return []

        self._rebuild_cache_if_needed()

        scored: list[tuple[MemoryNote, float]] = []
        query_dim = int(query_vec.shape[0])
        skipped_dim = 0
        for note in self._cache:
            if note.embedding is None:
                continue
            # 次元不一致ノートはスキップ
            if int(note.embedding.shape[0]) != query_dim:
                skipped_dim += 1
                continue
            sim = float(np.dot(note.embedding, query_vec))
            combined = sim * 0.6 + note.lightmem_score * 0.4
            if note.pin_flag:
                combined += _PIN_RETRIEVAL_BOOST
            scored.append((note, combined))

        if skipped_dim > 0:
            logger.warning(
                "retrieve_top_k: skipped %d notes with mismatched embedding dim "
                "(query_dim=%d). Run 'evoref reindex' or wait for sleep-time "
                "re-embedding.",
                skipped_dim, query_dim,
            )

        scored.sort(key=lambda x: -x[1])

        # アクセス記録
        for note, _ in scored[:k]:
            note.access_count += 1
            note.accessed_at = time.time()

        if scored:
            logger.debug(
                "retrieve_top_k: %d/%d notes with embeddings, top scores=[%s]",
                len(scored), len(self.notes),
                ", ".join(f"{s:.3f}" for _, s in scored[:k]),
            )

        return scored[:k]

    def save(self, path: str | Path) -> None:
        """JSON 永続化"""
        from backend.free.memory.stores.short_term_store import ShortTermMemoryStore
        ShortTermMemoryStore.save(self.notes, path)

    def load(self, path: str | Path) -> None:
        """JSON からロード

        ファイル未存在時は何もしない (既存ノート状態を保持)。
        """
        from backend.free.memory.stores.short_term_store import ShortTermMemoryStore
        loaded = ShortTermMemoryStore.load(path)
        if loaded is None:
            return
        self.notes = loaded
        self._cache_dirty = True

    def _rebuild_cache_if_needed(self) -> None:
        """スコア降順キャッシュを再構築"""
        if self._cache_dirty:
            self._cache = sorted(
                self.notes.values(),
                key=lambda n: -n.lightmem_score,
            )
            self._cache_dirty = False


