"""Level 0 即時学習: 経験バッファ"""

import uuid
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
    # この訂正が指す **誤っていたターン** の ``ExperienceEntry.id``。
    #
    # 訂正ペア (「元の問い → 訂正後の正しい回答」) を組むのに要る対応関係で、
    # **記録時にしか確定できない** (セッションが分かっていて、直近ターンの
    # 応答本文が手元にある)。以前はこれを残さず、後段の
    # ``learning.corrected_pairs`` がバッファの直前エントリを元の問いとみなして
    # 再導出していた。バッファは全セッション横断の 1 本なので、別会話の訂正が
    # 隣り合うと問いと訂正が食い違う (2026-09-06 ライブ監査 F-01: 訂正 7 件が
    # few-shot にも eval_core にも 1 件も入らなかった)。
    #
    # 同定は ``core.correction_target.resolve_correction_target`` (訂正文が
    # 引用する値・識別子と応答本文の重なり) で行う。旧データには無いので
    # ``None`` を許容し、消費側はセッション単位のフォールバックを持つ。
    corrected_entry_id: str | None = None
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
    # ── コストシグナル (2026-08-18 配線) ──
    # 「取得件数・予算を増やすほど品質指標が上がる」単調パラメータは、代償が
    # fitness に現れない限り最適化器が必ず制約の端まで膨らませる。実際に
    # search.top_k は上限を 50→10 へ切る対症療法が入り、long_form の
    # unit_target_tokens は下限を 128→512 に上げている。恒久的な対処には
    # コストを観測項として持つ必要があるため、既に一次情報が取れている 3 つを
    # 記録する。
    #
    # completion_tokens は生成トークン数 (呼出側が既に保持している厳密値)。
    # prompt_tokens / cached_prompt_tokens は llama-server の usage 由来で、
    # 再プリフィル量 = prompt_tokens - cached_prompt_tokens。取得不能な構成では
    # いずれも None (0 ではない = 「計測できなかった」と「消費ゼロ」を区別する)。
    completion_tokens: int | None = None
    prompt_tokens: int | None = None
    cached_prompt_tokens: int | None = None
    # ── 生成の結末 (2026-09-02 配線) ──
    # truncated: llama-server が finish_reason=length を返した = 応答が
    # max_tokens 到達で文の途中で切れている。切れた応答は few-shot の手本に
    # 採らない (FewShotPool.add_from_experiences)。
    truncated: bool = False
    # generation_failed: ユーザーへ 1 文字も届かなかった / error フレームで
    # 終わったターン。response は空で記録され turn_outcome は failed。
    # 以前はこのターンが記録されず、失敗が選択圧に一件も入らなかった。
    generation_failed: bool = False
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
class GenerationConfigRef:
    """その応答を生んだ構成の参照 (c_05 §0.6 ID 連鎖)。

    Level 1 / Level 2 は経験の fitness を候補 (プロンプト版 / few-shot /
    ポリシー / LoRA) へ帰属させるが、以前は **何が有効だったかを記録して
    いなかった** ため、帰属は時刻からの推測でしかなかった (2026-09-05 監査)。
    """

    prompt_version: int | None = None
    """このターンで使ったシステムプロンプトの版 (``PromptMeta.version``)。"""

    fewshot_ids: list[str] = field(default_factory=list)
    """注入した few-shot 例の ID (``FewShotExample.id``)。"""

    policy_generation: int | None = None
    """ポリシーパラメータの世代 (``PolicyParamEvolver`` の generation)。"""

    lora_version: int | None = None
    """有効だった base LoRA のスナップショット版 (未適用なら ``None``)。"""

    locale: str = ""
    """UI ロケール。プロンプト本文と few-shot の言語を決める軸。"""

    sampling: dict[str, float] = field(default_factory=dict)
    """temperature / top_p / max_tokens 等、生成時に効いたパラメータ。"""


@dataclass
class ExperienceEntry:
    """経験バッファの1エントリ"""

    id: str = ""
    """このエントリの ID (``exp_<hex12>``)。**同じターンの二重記録を検出できる
    唯一の手段**。以前は秒精度の timestamp しか無く、同一秒の 2 ターンが
    区別できず、再送・再取り込みの重複も検出できなかった。"""

    session_id: str = ""
    """発生した会話のセッション ID。"""

    turn_id: str = ""
    """発生したターンの ID (``WorkingMemory.add_turn`` が発行)。"""

    trace_id: str = ""
    """リクエストの trace_id。JSONL 側のログと突き合わせる鍵。"""

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
    lang: str = ""
    """応答本文の言語 (``ja`` / ``en`` / 未判定は空)。決定論判定で埋める。"""

    gen_config: GenerationConfigRef = field(default_factory=GenerationConfigRef)
    """この応答を生んだ構成 (fitness の帰属先)。"""

    signals: FeedbackSignals = field(default_factory=FeedbackSignals)

    @staticmethod
    def new_id() -> str:
        """``exp_<hex12>`` 形式の ID を発行する。"""
        return f"exp_{uuid.uuid4().hex[:12]}"


#: ``_entry_from_dict`` で復元する ``GenerationConfigRef`` のキー集合。
_GEN_CONFIG_FIELD_NAMES = frozenset(f.name for f in fields(GenerationConfigRef))


class ExperienceBuffer(JsonStateStore):
    """経験バッファ: 毎応答時にエントリを記録し、その場で永続化する。

    **耐久性はこのストア自身の責務**。以前は ``save()`` の呼出元が
    ``memory.sleep_update`` と ``core.model_migration`` しか無く、記録された
    エントリはアイドル窓の sleep-time サイクルが回るまでディスクに存在しな
    かった。``FeedbackCollector`` は記録のたびに "Recorded experience" を
    ログへ出すため、**ログ上は記録済みに見えて実体が無い** 窓が常時開いて
    いる (2026-09-06 監査 F-04: 50 ターン完走直後、メモリ 50 件に対しファイル
    49 件)。Level 2 の発火判定はこのバッファの失敗件数を見るので、再起動や
    クラッシュの時刻次第で学習データが目減りする。

    c_05 §0.5 は「各ストアは保持方針を宣言する」と定めており、保持を他の
    サブシステムのスケジュールに預ける形はその趣旨から外れる。
    """

    _state_logger = logger

    def __init__(self, max_entries: int = MAX_ENTRIES, *, autosave: bool = True):
        self.max_entries = max_entries
        self.entries: list[ExperienceEntry] = []
        # 直近に load / save したパーティションのファイル (rebind 時の退避先)。
        self.bound_path: Path | None = None
        #: ``record`` / ``flush`` で :attr:`bound_path` へ自動保存するか。
        #: 単体テストが一時ディレクトリを汚さないよう無効化できる。
        self.autosave = autosave

    def save(self, path: str | Path) -> None:
        super().save(path)
        self.bound_path = Path(path)

    def load(self, path: str | Path) -> None:
        super().load(path)
        self.bound_path = Path(path)

    def bind(self, path: str | Path) -> None:
        """保存先だけを設定する (読み込みはしない)。

        ファイルが未作成の初回起動でも :meth:`flush` が働くようにするための
        入口。``load`` は「ファイルがあるときだけ」呼ばれるため、これが無いと
        **初回セッションのあいだ自動保存が無効** になり、F-04 が新規環境で
        そのまま再現する。
        """
        self.bound_path = Path(path)

    def rebind(self, path: str | Path, *, previous: str | Path | None = None) -> None:
        """base モデル切替で経験バッファを新パーティションへ向け直す。

        現在のエントリを **先に** 旧パーティション (``previous``、省略時は
        :attr:`bound_path`) へ保存してから空にし、``path`` を読み込む
        (ファイル未存在なら空のまま)。以後の save は ``path`` が既定になる。
        """
        prev = Path(previous) if previous is not None else self.bound_path
        if prev is not None and self.entries:
            self.save(prev)
        self.entries = []
        self.load(path)
        self.bound_path = Path(path)
        logger.info(
            "Experience buffer rebound: %s -> %s (%d entries)",
            prev, self.bound_path, len(self.entries),
        )

    def record(self, entry: ExperienceEntry) -> None:
        """エントリを追加 (同じターンの二重記録は無視する)"""
        if not entry.timestamp:
            entry.timestamp = utc_now()
        if not entry.id:
            entry.id = ExperienceEntry.new_id()

        # 同一 turn_id が既にあるなら再送・再取り込み。timestamp は秒精度で
        # 同一秒の別ターンと区別できないため、ID で判定する。
        if entry.turn_id and any(
            e.turn_id == entry.turn_id for e in reversed(self.entries[-32:])
        ):
            logger.debug(
                "Skipping duplicate experience for turn %s", entry.turn_id,
            )
            return

        self.entries.append(entry)

        # ローテーション
        if len(self.entries) > self.max_entries:
            overflow = len(self.entries) - self.max_entries
            self.entries = self.entries[overflow:]
            logger.info("Rotated %d old entries", overflow)

        self.flush()

    def flush(self) -> None:
        """バインド済みパーティションへ書き出す (未バインド / 無効時は no-op)。

        ``record`` から毎ターン呼ぶ。書き込みは ``AtomicWriter`` 経由で、
        失敗しても ``JsonStateStore.save`` が WARNING を出して縮退するため
        チャット応答は止まらない。まとめ書き (デバウンス) はしない —
        「直近の 1 ターンだけ落ちる」窓を残すと、それが F-04 そのものになる。
        """
        if not self.autosave or self.bound_path is None:
            return
        super().save(self.bound_path)

    def get_recent(self, n: int = 10) -> list[ExperienceEntry]:
        """直近 n 件取得"""
        return self.entries[-n:]

    def get_failures(self, mode: str | None = None) -> list[ExperienceEntry]:
        """失敗エントリ抽出 (言い直し / ユーザー訂正 / 決定論の失敗判定)。

        ``turn_outcome == "failed"`` を含めないと、Level 2 の失敗プールは
        「ユーザーが言い直した or 訂正した」ターンだけになり、算術の破綻・
        ツール結果の不使用・自己矛盾のような **システムが自分で検出した失敗** が
        1 件も入らない (2026-09-05 ライブ監査 F-11: 100 ターンで 3 件)。

        Args:
            mode: 指定時はそのモード ("chat"/"create") のエントリのみに絞る。
                None (省略、既定) の場合は全モード横断 (後方互換)。
        """
        result = [
            e for e in self.entries
            if e.signals.rephrased_query
            or e.signals.user_correction is not None
            or e.signals.turn_outcome == "failed"
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

    def as_dicts(self) -> list[dict]:
        """全エントリを dict 化して返す (時系列順)。学習側の純粋関数の入力用。"""
        return list(self._to_payload())

    def _to_payload(self) -> JsonPayload:
        return [
            {
                "id": entry.id,
                "session_id": entry.session_id,
                "turn_id": entry.turn_id,
                "trace_id": entry.trace_id,
                "timestamp": entry.timestamp,
                "mode": entry.mode,
                "query": entry.query,
                "response_summary": entry.response_summary,
                "response_full": entry.response_full,
                "base_model": entry.base_model,
                "embedding_model": entry.embedding_model,
                "cartridge_ids": entry.cartridge_ids,
                "lang": entry.lang,
                "gen_config": asdict(entry.gen_config),
                "signals": asdict(entry.signals),
            }
            for entry in self.entries
        ]

    def _from_payload(self, payload: JsonPayload) -> None:
        if not isinstance(payload, list):
            raise TypeError(
                f"experience.json must be a list, got {type(payload).__name__}"
            )
        # 一時リストへ復元し、全件通ったあとで差し替える。以前は live の
        # entries を先に clear してから逐次 append していたため、壊れた要素
        # 1 件で途中の例外 → 半分だけ読んだ状態 → 次の save がそれをファイルへ
        # 書き戻して残りを失っていた (2026-09-02 監査 R-B1)。壊れた要素は
        # WARNING を出して飛ばす (黙って削らない)。
        parsed: list[ExperienceEntry] = []
        skipped = 0
        for d in payload:
            try:
                parsed.append(self._entry_from_dict(d))
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                skipped += 1
                logger.warning(
                    "Skipping malformed experience entry on load: %s (%r)",
                    exc, str(d)[:120],
                )
        if skipped:
            logger.warning(
                "Skipped %d malformed experience entr%s on load (kept %d)",
                skipped, "y" if skipped == 1 else "ies", len(parsed),
            )
        self.entries = parsed
        self._mark_loaded_conversations_ended()

    @staticmethod
    def _entry_from_dict(d: dict) -> ExperienceEntry:
        """JSON の 1 要素を ``ExperienceEntry`` へ復元する (壊れていれば例外)。"""
        signals_data = d.get("signals", {})
        if not isinstance(signals_data, dict):
            raise TypeError("signals must be a dict")
        gen_data = d.get("gen_config") or {}
        if not isinstance(gen_data, dict):
            gen_data = {}
        return ExperienceEntry(
            # 旧レコードは id を持たない。**ここで採番しない** — 読むたびに別 ID に
            # なると重複検出の役に立たないので、空のままにして「ID 以前のレコード」
            # と分かるようにする。
            id=d.get("id", ""),
            session_id=d.get("session_id", ""),
            turn_id=d.get("turn_id", ""),
            trace_id=d.get("trace_id", ""),
            timestamp=d.get("timestamp", ""),
            mode=d.get("mode", "chat"),
            query=d.get("query", ""),
            response_summary=d.get("response_summary", ""),
            response_full=d.get("response_full", ""),
            base_model=d.get("base_model") or "",
            embedding_model=d.get("embedding_model", ""),
            cartridge_ids=d.get("cartridge_ids", []),
            lang=d.get("lang", ""),
            gen_config=GenerationConfigRef(**{
                k: gen_data[k] for k in _GEN_CONFIG_FIELD_NAMES if k in gen_data
            }),
            # JSON に存在するキーのみ採用 (欠損キーは FeedbackSignals の
            # 既定値に委ねる)。新シグナル追加時もここの編集は不要。
            signals=FeedbackSignals(**{
                k: signals_data[k]
                for k in _SIGNAL_FIELD_NAMES
                if k in signals_data
            }),
        )

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
