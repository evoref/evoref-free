"""Layer 2: 短期記憶（A-MEM ノート形式 + LightMem スコア）"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from backend.log_config import get_logger
from backend.free.memory.volatile_values import is_volatile_measurement_report
from backend.free.memory.notes.note_builder import (
    get_note_builder,
    restates_attribute_value,
)
from backend.free.memory.notes.pin_detector import (
    PinTriggers,
    detect_pin,
    get_pin_triggers_for,
)
from backend.free.memory.types import MemoryMode, NoteSource
from backend.trace_context import get_trace_id

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
    #: 永続化対象 — 再起動でリセットされると恒久に失敗するノートを毎回
    #: 上限まで再試行し続ける。
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

    is_correction: bool = False
    """ユーザーが **自分の値を言い直した** ターンか。

    判定は :func:`backend.free.agent.feedback.restates_a_value` で、アシスタントの
    誤りの指摘 (``assistant``) とユーザー自身の申告訂正 (``self``) の両方を含む
    — 記憶側は「その属性の現在値が何か」を持つので、後者も正当な値更新。
    学習側の欠陥シグナル (``FeedbackCollector``) は従来どおり ``assistant`` のみ。

    チャット応答パス (``prepare_memory_context``) が ``WorkingMemory.add_turn``
    に渡し、sleep-time Step 8 が ``SemanticFact.from_correction`` へ引き継ぐ。
    抽出器はこの印が立っているノートに限り **直前の名前付き属性を継承** する。

    応答パスが立てなかった場合でも、``absorb`` が
    :func:`~backend.free.memory.notes.note_builder.restates_attribute_value`
    で拾い直す。``XではなくY`` 形は feedback 側では
    ``WEAK_CORRECTION_PATTERNS`` 扱い (直前ターンの失敗を伴うときだけ訂正) で、
    記憶層に必要な範囲と食い違うため (2026-08-25 の実測で
    「登壇日は11月3日ではなく11月10日に変更になりました。」が
    ``from_correction=False`` で書かれ、新旧 2 世代が live のまま残った)。
    """

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
    """コマンド実行が成功したか (出力が "Error:" prefix でない)。command 無しは None"""

    tool_command_source: str | None = None
    """判定層 ("rule" / "aux" / "recall" ...)。executable_command_curator が
    "recall" 由来 (= 過去 fact の引き当てで発火) を学習対象から外すために使う。
    これが無いと「誤発火 → 成功記録 → fact 延命 → また誤発火」で自己強化する。"""

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

    extraction_deferred: bool = False
    """Step 8 がセッション別上限 (``apply_session_caps``) で **今回は見送った**
    ノートか。上限超過で落ちた候補は ``extracted_fact_ids`` を持たないため、
    ``_last_extraction_at`` だけを基準にした eviction 保護から外れ、次の Step 8
    が来る前に LTM へ降格されうる。この印を立てて保護対象に含める。次サイクル
    で採用 (または候補にならなくなった) 時点で ``apply_session_caps`` が下ろす。"""

    trace_id: str | None = None
    """リクエスト相関 ID。``absorb`` が :func:`backend.trace_context.get_trace_id`
    (contextvar) から付与する。リクエスト外 (sleep-time の flush 等) では ``None``。"""

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

    assertion_curated_at: float | None = None
    """assertion_curator (Step 8.4) がこの user note を処理済みにした時刻。
    None は未処理。次サイクルで同じ言明を再命名・再記録しないためのマーカー。"""

    assertion_slug: str | None = None
    """assertion_curator (Step 8.4) がこのノートに割り当てた subject slug。
    訂正ノートは被訂正ノートからこれを継ぐ (別 slug になると SemMem の
    競合検出が対にできず supersede できない)。サイクルを跨いで継げるよう
    ノート側に持たせる。"""

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
        #: 前回の ``save`` 以降にノートが変わったか。``_cache_dirty`` は検索
        #: キャッシュの再構築で下りるため永続化の判定には使えない。
        self._persist_dirty: bool = False

        # ── 自動 Pin 検出設定 ──
        pin_cfg = mem.get("pin", {}) or {}
        self._pin_auto_detect: bool = bool(pin_cfg.get("auto_detect", True))
        # Pin トリガ辞書は user override (``<triggers_dir>/pin_triggers.yaml``) →
        # package 同梱 default の 2 段階で解決する。``triggers_dir=None`` の
        # 場合は override を使わず default のみ (テスト / 最小 config 用途)。
        self._pin_triggers_dir: str | Path | None = triggers_dir
        # NOTE: config キー ``memory.pin.auto_detect_confirm`` (「自動 pin 検出時に
        # ユーザー確認を挟む」フロー用に予約されていた) は値を読むコードが無いため
        # 撤去済み (``MemoryConfig.REMOVED_MEMORY_KEYS``)。確認フローを実装する際は
        # 新たにキーを追加した上で本クラス or pin_detector で参照する。

    def _load_pin_triggers(self) -> PinTriggers:
        """Pin トリガ辞書をプロセス内シングルトンから取得 (パスごと cache)。"""
        return get_pin_triggers_for(self._pin_triggers_dir)

    def mark_dirty(self) -> None:
        """ノート集合 / スコアが変わったことを記録する。

        検索キャッシュを無効化し、次回 ``save`` の対象にする。sleep-time の
        各 Step / eviction / conflict merge はノートを直接書き換えた後にこれを呼ぶ
        (``_cache_dirty`` への直接代入は行わない)。
        """
        self._cache_dirty = True
        self._persist_dirty = True

    @property
    def dirty(self) -> bool:
        """前回の ``save`` 以降に永続化すべき変更があるか。"""
        return self._persist_dirty

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
        is_correction: bool = bool(turn.get("is_correction", False))
        if not is_correction and (source or "user") == "user":
            # 応答パス側の判定 (``agent.feedback.restates_a_value``) が取り
            # こぼした値の言い直しを、**属性の裏取り付き**でここで拾い直す。
            #
            # ``XではなくY`` は日本語で最も普通の訂正形だが、feedback 側では
            # ``WEAK_CORRECTION_PATTERNS`` に置かれており、直前ターンの失敗か
            # 同一成果物の再指定を伴うときだけ訂正とみなす (学習層は「アシス
            # タントが誤ったか」を数えるので、それが正しい)。**記憶層は
            # 「その属性の現在値は何か」を持つ** ので必要な範囲が違う。
            #
            # 実インシデント (2026-08-25 ライブ監査の追調査): 「登壇日は11月3日
            # ではなく11月10日に変更になりました。」「犬の名前はコタロウではなく
            # ハナでした。」がどちらも ``from_correction=False`` で書かれ、
            # ``SemanticConflictResolver._decide`` の ``user_correction`` 即時
            # supersede が発火せず、新旧 2 世代が live のまま残った。
            #
            # 属性が解決できることを条件にするのは ``candidate_fact_tags`` が
            # 訂正形単独の証拠に課しているのと同じゲート — 「さっきの
            # 1234 × 5678 の答えは間違っています。正しくは 7006653 です。」の
            # ような、値の言い直しでない訂正を拾わないため。
            is_correction = restates_attribute_value(content, mode=mode)

        # ツール出力は WM までで止め、STM 以降には残さない
        if is_tool_output:
            logger.debug(
                "absorb: skip tool_output turn (mode=%s, session=%s, len=%d)",
                mode, session_id, len(content),
            )
            return None

        # ツール出力を **言い直した** アシスタント発話も同じ扱いにする。
        # ここを通すと ``is_tool_output`` の除外が 1 ホップで迂回され、揮発する
        # 計測値がエピソード記憶に焼き付く (詳細は
        # :func:`is_volatile_measurement_report` の docstring)。
        if role == "assistant" and is_volatile_measurement_report(content):
            logger.debug(
                "absorb: skip volatile measurement report (session=%s, len=%d)",
                session_id, len(content),
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
            is_correction=is_correction,
            trace_id=get_trace_id() or None,
        )
        self.notes[note.id] = note
        self.mark_dirty()
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

    def retrieve_top_k_detailed(
        self, query_vec: np.ndarray, k: int = 3, *, include_private: bool = False,
    ) -> list[tuple[MemoryNote, float, float]]:
        """``(note, combined, relevance)`` を返すスコア加重検索。

        ``combined`` は ``類似度 × 0.6 + LightMem × 0.4`` (+ pin 加点) で
        **順位付け専用**。``relevance`` は加工前の素のコサイン類似度で、検索品質
        ゲート (``QualityThresholds`` / ``low_quality_keep_floor`` /
        ``ChunkContentGate``) 用の値。これらのゲートは cosine スケールを前提に
        閾値が決められているため、LightMem を混ぜた ``combined`` を渡すと
        (a) スコアが圧縮されて閾値に到達せず、(b) LightMem が高いだけの無関連
        ノートが上位に来る (実測 2026-08-12: cos -0.05 のノートが 1 位)。
        したがって順位付けとゲートで別の値を使う。

        ``private=True`` のノートは既定で返さない (``include_private=False``)。
        プライベートセッションのターンは WM/STM に留めるだけの契約で、検索
        結果として別セッションのプロンプトへ出てはいけない。``MemoryInjector``
        は同じ除外を持っていたが ``[参考情報]`` 経路 (``_search_stm_layer``)
        には無かったため、ここで塞ぐ。
        """
        if not self.notes:
            return []

        self._rebuild_cache_if_needed()

        scored: list[tuple[MemoryNote, float, float]] = []
        query_dim = int(query_vec.shape[0])
        skipped_dim = 0
        for note in self._cache:
            if note.embedding is None:
                continue
            if note.private and not include_private:
                continue
            # 次元不一致ノートはスキップ
            if int(note.embedding.shape[0]) != query_dim:
                skipped_dim += 1
                continue
            sim = float(np.dot(note.embedding, query_vec))
            combined = sim * 0.6 + note.lightmem_score * 0.4
            if note.pin_flag:
                combined += _PIN_RETRIEVAL_BOOST
            scored.append((note, combined, sim))

        if skipped_dim > 0:
            logger.warning(
                "retrieve_top_k: skipped %d notes with mismatched embedding dim "
                "(query_dim=%d). Run 'evoref reindex' or wait for sleep-time "
                "re-embedding.",
                skipped_dim, query_dim,
            )

        scored.sort(key=lambda x: -x[1])

        # アクセス記録 (FadeMem の frequency 項に効くので永続化対象)
        for note, _, _ in scored[:k]:
            note.access_count += 1
            note.accessed_at = time.time()
        if scored:
            self._persist_dirty = True

        if scored:
            logger.debug(
                "retrieve_top_k: %d/%d notes with embeddings, "
                "top combined=[%s] relevance=[%s]",
                len(scored), len(self.notes),
                ", ".join(f"{c:.3f}" for _, c, _ in scored[:k]),
                ", ".join(f"{r:.3f}" for _, _, r in scored[:k]),
            )

        return scored[:k]

    def retrieve_top_k(
        self, query_vec: np.ndarray, k: int = 3, *, include_private: bool = False,
    ) -> list[tuple[MemoryNote, float]]:
        """スコア加重検索: ベクトル類似度 × LightMem スコア

        順位付け用の ``combined`` のみを返す従来 API。ゲート判定にも使う
        呼出側は :meth:`retrieve_top_k_detailed` を使うこと。
        """
        return [(note, combined) for note, combined, _ in
                self.retrieve_top_k_detailed(
                    query_vec, k, include_private=include_private,
                )]

    def save(self, path: str | Path, *, allow_empty: bool = False) -> None:
        """JSON 永続化

        既定では **空の STM で既存スナップショットを上書きしない**
        (理由は ``ShortTermMemoryStore.save`` の docstring)。書き出せたら
        :attr:`dirty` を下ろす (拒否されたときは次回に持ち越す)。
        """
        from backend.free.memory.stores.short_term_store import ShortTermMemoryStore
        if ShortTermMemoryStore.save(self.notes, path, allow_empty=allow_empty):
            self._persist_dirty = False

    def load(self, path: str | Path) -> None:
        """JSON からロード

        ファイル未存在時は何もしない (既存ノート状態を保持)。
        """
        from backend.free.memory.stores.short_term_store import ShortTermMemoryStore
        loaded = ShortTermMemoryStore.load(path)
        if loaded is None:
            return
        self.notes = loaded
        self.mark_dirty()

    def _rebuild_cache_if_needed(self) -> None:
        """スコア降順キャッシュを再構築"""
        if self._cache_dirty:
            self._cache = sorted(
                self.notes.values(),
                key=lambda n: -n.lightmem_score,
            )
            self._cache_dirty = False


