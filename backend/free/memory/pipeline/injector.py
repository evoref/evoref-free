"""

EvorefMem 統合仕様 におけるモード別の意味記憶 / 短期記憶注入器
``SemanticFact`` と ``MemoryNote`` の集合を入力として、モード別の Tier 配分
(チャット 800 トークン / クリエイト 2000 トークン) に従って LLM プロンプト
へ注入する候補列を生成する。

設計仕様:

1. **モード別予算と Tier 比率**

   ============  =============  =================================
   モード        予算 (tokens)  Tier 比率 (T1, T2, T3, T4)
   ============  =============  =================================
   chat            800          (0.40, 0.35, 0.15, 0.10)
   create         2000          (0.40, 0.35, 0.15, 0.10)
   ============  =============  =================================

2. **Tier 1 ボーナス**: ``pinned=True`` の項目はスコアに ``+1.0`` を付与し、
   かつ Tier 1 へ強制配置する (タグ種別を問わない)。

3. **Create Tier 1 拡張**:

   - ``policy`` ファクトは ``confidence >= 0.7`` (active) のみ採用。
   - ``failure_pattern`` ファクトは ``failure_signature`` が呼び出し側
     から提供された signature 集合と一致したものだけ Tier 1 注入。

4. **Create Tier 2 拡張**: ``current_project_id`` と一致する ``task``
   ファクト (current task) を Tier 2 に注入する。

5. **予算オーバー時の挙動**: 各 Tier はその予算を厳密上限とする
   (``int(total_budget * ratio)``)。さらに最終的に総使用量が総予算を
   超えた場合は **Tier 4 から順に削除** する

本モジュールは外部 I/O を一切持たない純粋関数的な計画器であり、
チャット応答パスへの配線は別層で行う。本モジュールのスコープは
「ロジック実装 + ユニットテスト」までとする。
"""

from __future__ import annotations

import math
import time

import numpy as np
from dataclasses import dataclass, field
from typing import Iterable, Literal

from backend.free.core.text_quality import carries_no_assertion, is_payload_dump
from backend.free.core.session_mode import (
    is_chat_mode,
    is_create_mode,
    is_valid_session_mode,
)
from backend.free.memory.stores.short_term import MemoryNote
from backend.free.memory.types import MemoryMode, SemanticFact
from backend.log_config import get_logger
from backend.utils import estimate_tokens

logger = get_logger("memory.injector")

# ──────────────────────────────────────────────────────────────────────────
# 定数
# ──────────────────────────────────────────────────────────────────────────

DEFAULT_CHAT_BUDGET_TOKENS = 800
DEFAULT_CREATE_BUDGET_TOKENS = 2000
DEFAULT_TIER_RATIOS: tuple[float, float, float, float] = (0.40, 0.35, 0.15, 0.10)

#: pin ボーナス値
PINNED_BONUS = 1.0

#: ``from_correction`` ファクトのスコア加点。
#:
#: スロット継承 (``ChatExtractor``) が効けば訂正は対象と同じスロットに入り、
#: ``_collapse_to_current_values`` が最新だけを残すのでこの加点は要らない。
#: **効かなかった残り** — 対象が数ターン前で継承できなかった / 対象自体が
#: フォールバックスロットだった場合 — に、古い値と訂正後の値が別スロットで
#: 並ぶことがある。そのとき順位で訂正側を上に出すための保険。
#:
#: 実測 (2026-08-19 ライブ検証): 「私の好きな飲み物は何ですか？」に対し
#: 訂正済みの「緑茶」が sim 0.762 で最上位、訂正後の「ほうじ茶」が 0.487 で
#: 下位に並んでいた。関連度はクエリとの類似で決まるため、訂正の方が下に来る。
#:
#: pin (1.0) より小さくするのは、明示 pin の優先を崩さないため。

CORRECTION_BONUS = 0.5

#: policy ファクトを active と見なす最小 confidence
DEFAULT_POLICY_ACTIVATION_MIN_CONFIDENCE = 0.7

#: スコア計算で用いる recency 半減期 (日)
_RECENCY_HALF_LIFE_DAYS = 14.0

#: この日数以上前に記録されたファクトは、行に「N日前の記録」と書き添える。
#: 当日のもの (= 今の会話で書かれた可能性がある) には付けない。
_FACT_STALE_LABEL_DAYS = 1.0

#: 関連度ゲートの既定閾値 (クエリ埋め込みとのコサイン類似度)。
#:
#: 従来 inject() は query を受け取らず、recency / type / tier 比率の静的スコア
#: だけで候補を選んでいた。そのため話題に関係なく毎ターン同じ「ユーザーの
#: プロフィール」や「過去の質問文」が注入され、弱い base モデルがそれを
#: 「いま答えるべき対象」と誤解して復唱する事象が起きていた
#: (実測 2026-07-25: 「応答は日本語で書いてほしい」に対し、注入された
#:  過去の質問「私の名前と出身地、覚えていますか？」へ回答した)。
#:
#: 閾値は稼働 embed (LFM2.5-Embedding-350M) での実測から決めた:
#:   真陽性  GPU 想起→GPU ノート 0.3787 / コメント方針→方針ノート 0.4441
#:   ノイズ  中央値 0.12〜0.17
#: 0.40 では GPU ノート (0.3787) を落とすため 0.35 を採用する。
DEFAULT_RELEVANCE_MIN_SCORE = 0.35

#: pinned ファクトに課す関連度の **下限**。
#:
#: pin は「優先度」の指定であって「常に関連する」の宣言ではない。しかも pin 検出は
#: 「覚えておいてください」等の語で発火するため、``mem.personal.name`` /
#: ``birthday`` のようなファクトが**自動的に** pin され、以後すべてのターンで
#: 無条件に載り続けていた (ゲート完全迂回)。
#:
#: 実測 (2026-08-09、実ファクトストア 45 件 / LFM2.5-Embedding-350M):
#: 「1 キロメートルは何メートルですか。」に対し pin 済み 5 件が類似度
#: -0.006 / 0.020 / 0.040 / -0.013 / -0.013 で注入されていた
#: (``relevance_min_score`` を 0.80 まで上げても素通り)。
#:
#: 下限別の比較 — 記憶不要ターン / 想起ターンの注入量と想起の正否:
#:   0.00 (旧)  297 tok (約 2.2 秒) / 297 tok / 全 OK
#:   0.10 (既定)  81 tok (約 0.6 秒) / 297 tok / 全 OK   ← 想起を一切損なわない
#:   0.25         70 tok            / 192 tok / 全 OK
#:   0.35         70 tok            / 179 tok / 1 件 NG
#:
#: 0.10 は想起ターンの注入量・正答をそのままに、記憶不要ターンだけ 73% 削る。
#: 破綻するのは 0.35 なので 3.5 倍の余裕がある。
DEFAULT_PINNED_RELEVANCE_MIN_SCORE = 0.10

#: 較正が効いているときの pinned 下限 = ``relevance × この比率``。
#:
#: 静的既定 0.10 は **背景ノイズ帯の内側**にある。実測の背景分布
#: (LFM2.5-Embedding-350M / 2026-08-18 の較正キャッシュ) は p50 0.039 / p95 0.302
#: で、0.10 は無関係ペアの過半より上とはいえ p95 を大きく下回る。つまり pin が
#: 付いた瞬間、そのファクトはほぼ全ターンに載る。
#:
#: 較正値に対する比率にすれば埋め込みスケールに追随する。0.5 は現行 embedder で
#: 0.302 × 0.5 = 0.151 となり、実測表 (0.10 / 0.25 とも想起は全 OK、0.35 で 1 件 NG)
#: の安全域の中に収まる。pin は「優先度」であって「常に関連する」の宣言ではない
#: ので、通常ゲートより緩く・ノイズ帯より上、という位置付けを保つ。
PINNED_RELEVANCE_RATIO = 0.5


#: セッション要約ファクトの subject 接頭辞。会話のメタ記録であって、ユーザーに
#: ついての事実ではないため [関連する記憶] へ注入しない (``inject`` 内の判定参照)。
_SESSION_SUMMARY_SUBJECT_PREFIX = "mem.decision.history.session"

#: MDP エピソードトレース由来の decision ファクトの subject 接頭辞
#: (``MDPTraceExtractor`` が ``mem.decision.<episode_id>`` で書く。episode_id は
#: ``ep_`` 始まり)。セッション要約と同じく「アシスタントが何をしたか」の内部記録で、
#: ユーザーについての事実ではない。
#:
#: 実インシデント (2026-08-16 ライブ監査 ターン34): 「今日の会話のいちばん最初、
#: 私は何の話をした？」の [関連する記憶] に
#:   - (decision) mem.decision.ep_5666c421 resolved:
#:     outcome=success; result=No results found for: …; actions=search_history
#: が入っていた。**0 件で終わった検索が「成功した記録」として** ユーザー向けの
#: 根拠枠に並ぶ。outcome=success 自体はツールが壊れていない意味で正しい
#: (``agent.deliberative._trace_tool_episode`` の設計) が、それをそのまま
#: [関連する記憶] に出すと読み手には検索が当たったように見える。
#: 「前回何をしたか」は search_history ツールの担当。
#:
#: ``mem.decision.*`` 全体は落とさない — create モードの採否判断
#: (``mem.decision.<project_id>``) は正当なユーザーファクト。
_EPISODE_TRACE_SUBJECT_PREFIX = "mem.decision.ep_"

#: executable command リコール索引の subject 接頭辞
#: (``memory.sleep.executable_command_curator``)。``world_fact`` を流用して
#: いるが中身は「このクエリはこのコマンドで答えた」という **索引** で、
#: ``object`` には過去のユーザーの質問文がそのまま入る。読み手は
#: ``ToolCallJudge`` だけ (``agent.tool_call_judge`` が同じ接頭辞で引く)。
#:
#: [関連する記憶] に並べると、過去の質問文が「世界の事実」として提示される。
#: 実データ (2026-08-16 監査時点):
#:   (world_fact) mem.world.executable_command.chat.0e480f56 answers_query:
#:   今日は8月16日ですよね。今日から100日後は何月何日になりますか？
#: セッション要約 / エピソードトレースと同じ「内部索引」の類。
_EXECUTABLE_COMMAND_SUBJECT_PREFIX = "mem.world.executable_command."

#: 内部索引の subject 接頭辞。いずれも「アシスタント側の記録」であって
#: ユーザーについての事実ではないため、ユーザーに見える枠へ出さない。
#:
#: 消費側は 2 つ: ``[関連する記憶]`` (本モジュールの :meth:`MemoryInjector.inject`)
#: と ``[記憶の競合]`` (``conflict_review.collect_review_groups``)。片方だけに
#: 掛けると同じ内容が別の窓から出る — 実際 2026-08-19 時点の pending は全 2 件が
#: セッション要約で、競合セクション側から素通しになっていた。
INTERNAL_INDEX_SUBJECT_PREFIXES: tuple[str, ...] = (
    _SESSION_SUMMARY_SUBJECT_PREFIX,
    _EPISODE_TRACE_SUBJECT_PREFIX,
    _EXECUTABLE_COMMAND_SUBJECT_PREFIX,
)


def _normalize_for_dup(text: str) -> str:
    """重複判定用の正規化 (空白除去のみ)。曖昧一致はしない。"""
    return "".join((text or "").split())


# ──────────────────────────────────────────────────────────────────────────
# 結果データクラス
# ──────────────────────────────────────────────────────────────────────────

ItemSource = Literal["fact", "note"]


@dataclass(frozen=True)
class InjectedItem:
    """注入候補 1 件 (ファクト or ノート)"""

    tier: int
    source: ItemSource
    item_id: str
    text: str
    tokens: int
    score: float


@dataclass
class InjectionPlan:
    """:class:`MemoryInjector.inject` の結果。

    :attr:`items` は Tier 1 → 4 の順、各 Tier 内はスコア降順。
    :attr:`dropped` は予算オーバーや Tier 不適合で除外された候補。
    """

    mode: MemoryMode
    budget_tokens: int
    tier_budgets: list[int]
    items: list[InjectedItem] = field(default_factory=list)
    dropped: list[InjectedItem] = field(default_factory=list)
    used_tokens: int = 0

    def render(self) -> str:
        """注入対象を 1 つのテキストに連結する (改行区切り)。"""
        return "\n".join(it.text for it in self.items)

    def by_tier(self, tier: int) -> list[InjectedItem]:
        return [it for it in self.items if it.tier == tier]


# ──────────────────────────────────────────────────────────────────────────
# MemoryInjector
# ──────────────────────────────────────────────────────────────────────────


class MemoryInjector:
    """モード別 Tier 注入器。

    config 例 (``config.yaml`` 全体をそのまま渡す想定だが、未指定でも
    デフォルト値で動作する)::

        memory:
          injection:
            chat_budget_tokens: 800
            create_budget_tokens: 2000
            tier_ratios:
              chat: [0.40, 0.35, 0.15, 0.10]
              create: [0.40, 0.35, 0.15, 0.10]
        learning:
          policy:
            activation_min_confidence: 0.7

    ``policy_activation_min_confidence`` は
    ``learning.policy.activation_min_confidence`` を SSOT とし、
    ``PolicyInterpreter`` / ``LoopFactView`` 等と必ず同一閾値で動作する
    (旧 ``harness:`` セクションから移行済)。
    """

    def __init__(
        self,
        config: dict | None = None,
        *,
        now_provider=None,
    ) -> None:
        cfg_root = config or {}
        cfg = ((cfg_root.get("memory") or {}).get("injection") or {})
        learning_policy_cfg = (
            (cfg_root.get("learning") or {}).get("policy") or {}
        )
        self.chat_budget: int = int(
            cfg.get("chat_budget_tokens", DEFAULT_CHAT_BUDGET_TOKENS),
        )
        self.create_budget: int = int(
            cfg.get("create_budget_tokens", DEFAULT_CREATE_BUDGET_TOKENS),
        )
        ratios = cfg.get("tier_ratios") or {}
        self.chat_ratios = self._normalize_ratios(
            ratios.get("chat"), DEFAULT_TIER_RATIOS,
        )
        self.create_ratios = self._normalize_ratios(
            ratios.get("create"), DEFAULT_TIER_RATIOS,
        )
        self.policy_min_confidence: float = float(
            learning_policy_cfg.get(
                "activation_min_confidence",
                DEFAULT_POLICY_ACTIVATION_MIN_CONFIDENCE,
            ),
        )
        self.relevance_enabled: bool = bool(
            cfg.get("relevance_enabled", True),
        )
        (
            self.relevance_min_score,
            self.pinned_relevance_min_score,
        ) = self._resolve_relevance_thresholds(cfg)
        self._now_provider = now_provider or time.time

    @staticmethod
    def _resolve_relevance_thresholds(cfg: dict) -> tuple[float, float]:
        """``(relevance_min_score, pinned_relevance_min_score)`` を解決する。

        ``threshold_mode: auto`` (既定) で較正
        (:mod:`backend.free.rag.memory_threshold_calibration`) が効いていれば、
        RAG 側の ``relevance`` と **同じ棒**を使う。

        なぜ較正に載せるか: 静的な絶対閾値は埋め込みモデルを替えると到達不能に
        なり、**黙って全部落とす**。このプロジェクトは既に 2 度同じ壊れ方をして
        いる (``rag.relevance_threshold: 0.65`` が LFM2.5 で記憶採用 0 件、
        ``low_quality_keep_floor: 0.40`` が観測最大 0.381 を上回り通過 0 件)。
        RAG 側は ``threshold_mode: auto`` + 相対フロアで二重に守られたが、
        **注入側のこの 1 本だけが静的なまま残っていた**。

        相対フロア (top × 0.6) は採らない。RAG のそれは「このクエリで検索して
        取れた結果集合」の top を基準にするので意味を持つが、注入側の候補は
        **ストア全件**で、大半はクエリと無関係。無関係な集合の top に対する相対で
        緩めると、ノイズの中の最上位を「関連あり」に格上げしてしまう。到達不能を
        防ぐ役割は較正が果たし、較正が効かない構成では静的値のまま
        (:meth:`inject` が全件却下をログに出すので、沈黙はしない)。
        """
        static_relevance = float(
            cfg.get("relevance_min_score", DEFAULT_RELEVANCE_MIN_SCORE),
        )
        static_pinned = float(
            cfg.get(
                "pinned_relevance_min_score",
                DEFAULT_PINNED_RELEVANCE_MIN_SCORE,
            ),
        )
        if str(cfg.get("threshold_mode", "auto")) != "auto":
            return static_relevance, static_pinned

        from backend.free.rag.memory_threshold_calibration import (
            get_active_calibration,
        )

        calibration = get_active_calibration()
        if not calibration:
            return static_relevance, static_pinned
        relevance = float(
            calibration.get("relevance_threshold", static_relevance),
        )
        return relevance, relevance * PINNED_RELEVANCE_RATIO

    # ── public API ────────────────────────────────────────────────────

    def inject(
        self,
        *,
        mode: MemoryMode,
        facts: Iterable[SemanticFact] = (),
        stm_notes: Iterable[MemoryNote] = (),
        current_project_id: str | None = None,
        failure_signatures: Iterable[str] | None = None,
        query_embedding: "np.ndarray | None" = None,
    ) -> InjectionPlan:
        """注入計画を構築する。

        Args:
            mode: ``"chat"`` または ``"create"``。
            facts: 候補 ``SemanticFact`` の集合 (global / project 混在可)。
            stm_notes: 候補 ``MemoryNote`` の集合 (Tier 2 配置)。
            current_project_id: クリエイトモードで「現在プロジェクト」と
                見なすプロジェクト ID。``None`` の場合 project ファクトは
                すべて他プロジェクト扱いになる。
            failure_signatures: Tier 1 注入を許可する failure_pattern の
                ``failure_signature`` 集合 (signature 一致時のみ Tier 1)。
            query_embedding: 現在のユーザー発話の埋め込み。与えられた場合、
                埋め込みを持つ候補は類似度 ``relevance_min_score`` 未満なら
                注入しない (``pinned`` は明示指定なので常に通す)。``None``
                なら関連度ゲートは無効 (従来どおり静的スコアのみで選ぶ)。

        Returns:
            :class:`InjectionPlan` — 採用候補 / 削除候補 / 使用トークン。
        """
        if not is_valid_session_mode(mode):
            raise ValueError(f"unsupported mode: {mode}")

        sigs: set[str] = set(failure_signatures or ())
        budget = self.chat_budget if is_chat_mode(mode) else self.create_budget
        ratios = self.chat_ratios if is_chat_mode(mode) else self.create_ratios
        tier_budgets = self._tier_budgets(budget, ratios)
        query_vec = self._prepare_query_vec(query_embedding)
        filtered_out = 0
        # 関連度ゲートに **到達した** 候補と、そこで落ちた数。到達数と却下数が
        # 一致したら「棒が到達不能になっている」サイン (較正が効かない構成で
        # 埋め込みモデルを替えたとき、静的閾値が黙って全部落とす)。
        gate_reached = 0
        gate_rejected = 0

        # Tier ごとに分類
        buckets: dict[int, list[InjectedItem]] = {1: [], 2: [], 3: [], 4: []}

        facts, collapsed, stale_texts = self._collapse_to_current_values(facts)
        filtered_out += collapsed

        for fact in facts:
            if fact.superseded_by:
                continue
            # 問いだけのファクトは主張を含まないのに「(personal_fact)
            # mem.personal.birthday states: ...」と **断定形** で注入され、
            # モデルがそれを既知の事実として扱う。抽出側のゲート
            # (extractors/chat.py の _tag_evidence_is_question_only) は
            # 2026-08-06 に入ったが、それ以前に保存された行は残り続けるため、
            # 読込時にも同じ判定を掛ける。
            #
            # ノート側 (下の carries_no_assertion) と違い pin を例外にしない。
            # pin は**優先度**の指定であって主張の有無とは無関係で、問いだけの
            # 行を優先注入しても「既知の事実」の誤認を強めるだけだから。しかも
            # ピン検出は「覚えておいてください」等の語で発火するので、
            # **問い自体がその語を含むと自動で pin される** (実インシデント
            # 2026-08-07 ライブ監査: 「私の猫の名前と誕生日を覚えていますか。」
            # が pinned な personal_fact として全ターンの [関連する記憶] に
            # 載り続けていた — pin 例外を付けるとこの実例が素通りする)。
            if carries_no_assertion(fact.object or ""):
                filtered_out += 1
                continue
            # セッション要約は「会話で何が起きたか」のメタ記録であって、
            # ユーザーについての事実ではない。[関連する記憶] に並べると
            # **アシスタント自身の過去の回答が事実として提示される**。
            #
            # 実インシデント (2026-08-09): 「私の趣味は？」に対し
            #   (decision) ...: ユーザーが自分の趣味を尋ねたところ、
            #                   アシスタントは「自転車と写真」であると回答しました。
            # が 3 件並んでいた。趣味は既に「登山と写真」へ更新済みだったが、
            # 要約は古い誤答をそのまま事実として記録しており、**注入内で
            # 自転車 5 回 vs 登山 2 回**と数で上回っていた。しかも誤答するたびに
            # 新しい要約が生まれ、次のターンでさらに数が増える自己増幅になる
            # (実測: テスト実行のたびに 1 件ずつ増加。ある要約は誤答を
            #  「正しく提供しました」と記録していた)。
            #
            # 実ストアでは live 50 件中 33 件 (66%) がこの種別で、注入予算も
            # 大きく食っていた (想起クエリで 582 → 219 文字)。
            # 「前回何を話したか」は search_history ツールの担当。
            # MDP エピソードトレース / executable command 索引も同じ理由で落とす
            # (:data:`INTERNAL_INDEX_SUBJECT_PREFIXES` の各説明を参照)。
            if fact.subject.startswith(INTERNAL_INDEX_SUBJECT_PREFIXES):
                filtered_out += 1
                continue
            gate_reached += 1
            if not self._is_relevant(
                query_vec, getattr(fact, "embedding", None),
                pinned=bool(fact.pinned),
            ):
                filtered_out += 1
                gate_rejected += 1
                continue
            tier = self._classify_fact(
                fact, mode, current_project_id, sigs,
            )
            if tier is None:
                continue
            score = self._score_fact(fact)
            text = self._render_fact(fact)
            tokens = estimate_tokens(text)
            buckets[tier].append(
                InjectedItem(
                    tier=tier,
                    source="fact",
                    item_id=fact.id,
                    text=text,
                    tokens=tokens,
                    score=score,
                ),
            )

        for note in stm_notes:
            pinned = bool(getattr(note, "pin_flag", False))
            # ファクト側で却下した世代が、同じ本文のままノート経由で戻るのを塞ぐ
            # (``_is_stale_duplicate`` 参照)。pin も例外にしない — 判断済みの
            # 内容の再提示に優先度を与える理由が無い。
            if self._is_stale_duplicate(
                getattr(note, "content", "") or "", stale_texts,
            ):
                filtered_out += 1
                continue
            # 問いだけのノートは答えを含まないのに (過去の記録) として注入され、
            # モデルがそれを回答として復唱する (実インシデント 2026-08-04
            # ライブ監査:「今日は何曜日ですか。」が過去の記録として想起され、
            # 同文がそのまま出力された)。明示的に pin されたものは尊重する。
            if not pinned and carries_no_assertion(getattr(note, "content", "")):
                filtered_out += 1
                continue
            # 本文の過半がコードフェンスの中身 = 「いつでも取り直せるデータの
            # コピー」。記憶として再注入すると内容が古びるうえ、「ペイロードを
            # 貼るのが正解」という手本として働く (2026-08-16 動作検証: README を
            # 全文ダンプした回答がノート化され、(過去の記録) として再注入され、
            # 次のターンでまたダンプさせていた — 自己増幅ループ)。
            if not pinned and is_payload_dump(getattr(note, "content", "")):
                filtered_out += 1
                continue
            gate_reached += 1
            if not self._is_relevant(
                query_vec, getattr(note, "embedding", None),
                # MemoryNote 側の pin 属性は ``pin_flag`` (SemanticFact は ``pinned``)。
                pinned=pinned,
                require_embedding=True,
            ):
                filtered_out += 1
                gate_rejected += 1
                continue
            tier = self._classify_note(note, mode, current_project_id)
            if tier is None:
                continue
            score = self._score_note(note)
            text = self._render_note(note)
            tokens = estimate_tokens(text)
            buckets[tier].append(
                InjectedItem(
                    tier=tier,
                    source="note",
                    item_id=note.id,
                    text=text,
                    tokens=tokens,
                    score=score,
                ),
            )

        plan = InjectionPlan(
            mode=mode,
            budget_tokens=budget,
            tier_budgets=tier_budgets,
        )

        # Tier 1 → 4 の順にパック
        for tier in (1, 2, 3, 4):
            cap = tier_budgets[tier - 1]
            accepted, dropped = self._pack_tier(buckets[tier], cap)
            plan.items.extend(accepted)
            plan.dropped.extend(dropped)
            plan.used_tokens += sum(it.tokens for it in accepted)

        # 総予算オーバー時は Tier 4 から削除
        if plan.used_tokens > budget:
            self._spill_from_tier4(plan, budget)

        logger.debug(
            "MemoryInjector.inject: mode=%s budget=%d used=%d items=%d "
            "dropped=%d project=%s sigs=%d relevance=%s filtered=%d",
            mode, budget, plan.used_tokens, len(plan.items),
            len(plan.dropped), current_project_id, len(sigs),
            "on" if query_vec is not None else "off", filtered_out,
        )
        # 全件却下は「関連する記憶が無いターン」でも起きるが、それが**続く**なら
        # 棒が到達不能になっている。静的閾値のまま埋め込みモデルを替えたときの
        # 沈黙故障 (このプロジェクトで 2 度起きている) を観測可能にする。
        if query_vec is not None and gate_reached and gate_rejected == gate_reached:
            logger.info(
                "MemoryInjector: the relevance gate rejected all %d candidate(s) "
                "(min_score=%.3f, pinned_min=%.3f). If this persists across turns, "
                "the threshold is unreachable on this embedding scale - check the "
                "calibration cache (memory_threshold_calibration).",
                gate_reached, self.relevance_min_score,
                self.pinned_relevance_min_score,
            )
        return plan

    # ── 関連度ゲート ──────────────────────────────────────────────────

    def _prepare_query_vec(
        self, query_embedding: "np.ndarray | None",
    ) -> "np.ndarray | None":
        """関連度ゲート用に正規化済みクエリベクトルを返す。無効なら ``None``。"""
        if query_embedding is None or not self.relevance_enabled:
            return None
        v = np.asarray(query_embedding, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(v))
        if not norm or not np.isfinite(norm):
            return None
        return v / norm

    def _is_relevant(
        self,
        query_vec: "np.ndarray | None",
        embedding: "np.ndarray | None",
        *,
        pinned: bool,
        require_embedding: bool = False,
    ) -> bool:
        """関連度ゲートを通すか判定する。

        - ゲート無効 (``query_vec is None``) → 常に通す (従来挙動)
        - ``pinned`` → 通常のゲートは免除するが、``pinned_relevance_min_score``
          の**下限だけは課す** (下記)
        - 埋め込みを持たない候補 → ``require_embedding`` なら落とす、
          さもなくば判定不能として通す
        - それ以外 → コサイン類似度 >= ``relevance_min_score`` のみ通す

        pin に下限を課す理由 (実測 2026-08-09): pin は「優先度」の指定であって
        「常に関連する」の宣言ではない。しかも pin 検出は「覚えておいてください」
        等の語で発火するため、``mem.personal.name`` / ``birthday`` のような
        ファクトが**自動的に** pin され、以後すべてのターンで無条件に載り続ける。

        実測: 「1 キロメートルは何メートルですか。」に対し pin 済み 5 件が
        類似度 -0.006 / 0.020 / 0.040 / -0.013 / -0.013 で注入されていた
        (``relevance_min_score`` を 0.80 まで上げても素通り)。記憶が要らない
        ターンで毎回 297 token = 約 2.2 秒の prefill を払っていた。

        下限 0.10 での実測 (実ファクトストア 45 件):

        =========================== ============== ============== ========
        floor                       記憶不要ターン 想起ターン     想起
        =========================== ============== ============== ========
        0.00 (旧: 完全迂回)         297 tok (2.2s) 297 tok        全 OK
        **0.10 (既定)**             **81 tok**     **297 tok**    全 OK
        0.25                         70 tok        192 tok        全 OK
        0.35                         70 tok        179 tok        1 件 NG
        =========================== ============== ============== ========

        0.10 は想起ターンの注入量・正答を一切損なわずに、記憶不要ターンだけを
        73% 削る。破綻するのは 0.35 なので 3.5 倍の余裕がある。

        ``require_embedding`` は STM ノート用。ノートは生成時点では
        ``embedding=None`` で、sleep-time の embed 工程を通るまで値が入らない
        (note_builder / sleep_update 参照)。埋め込み無しを「判定不能なので通す」
        にすると、直近の別セッションのノートが**クエリと無関係でも新しい順に
        必ず注入される** (実インシデント 2026-07-27 ライブ検証: 新規セッション
        1 ターン目の「明日の予定を整理しておいてください」に対し、別セッション
        の歯科予約・会議進行の話が注入され、この会話に存在しない予定を捏造した)。
        ノート本文は素の会話テキストで一人称・時制をそのまま含むため誤読の害が
        大きい。関連性を確認できないノートは注入しない方が安全側になる。
        """
        if query_vec is None:
            return True
        if embedding is None:
            # pin されていても判定材料が無いのは従来どおり通す (下限は課せない)。
            return True if pinned else not require_embedding
        w = np.asarray(embedding, dtype=np.float32).ravel()
        if w.shape != query_vec.shape:
            return True
        norm = float(np.linalg.norm(w))
        if not norm or not np.isfinite(norm):
            return True
        score = float(query_vec @ (w / norm))
        if pinned:
            return score >= self.pinned_relevance_min_score
        return score >= self.relevance_min_score

    # ── tier 分類 ────────────────────────────────────────────────────

    def _classify_fact(
        self,
        fact: SemanticFact,
        mode: MemoryMode,
        current_project_id: str | None,
        failure_signatures: set[str],
    ) -> int | None:
        """ファクトの Tier を返す。配置対象外なら ``None``。"""
        if fact.private:
            # private ファクトは注入対象外
            return None
        if fact.pinned:
            # pinned は常に Tier 1 に強制配置
            return 1

        is_current_project = (
            current_project_id is not None
            and fact.scope == f"project:{current_project_id}"
        )
        is_other_project = fact.is_project_scoped() and not is_current_project
        t = fact.type

        if is_chat_mode(mode):
            if t in ("personal_fact", "preference", "emotion"):
                return 1
            if t in ("decision", "commitment"):
                return 2
            if t in ("belief", "opinion"):
                return 3
            if t in ("world_fact", "project"):
                return 4
            return None

        # create mode
        if t == "policy":
            if fact.confidence >= self.policy_min_confidence:
                return 1
            return None
        if t == "failure_pattern":
            if (
                fact.failure_signature is not None
                and fact.failure_signature in failure_signatures
            ):
                return 1
            return None
        if t == "project":
            return 1 if is_current_project else 4
        if t == "decision":
            if is_current_project:
                return 1
            if is_other_project:
                return 3
            # global decision
            return 1
        if t == "commitment":
            return 1
        if t == "task":
            return 2 if is_current_project else None
        if t == "create":
            return 2 if is_current_project else 4
        if t == "preference":
            return 3
        if t == "personal_fact":
            return 4
        if t == "world_fact":
            return 4
        if t == "model":
            return 4
        # progress_marker などは別層で扱う
        return None

    def _classify_note(
        self,
        note: MemoryNote,
        mode: MemoryMode,
        current_project_id: str | None,
    ) -> int | None:
        """STM ノートの Tier を返す。"""
        if note.private:
            return None
        if note.is_tool_output:
            # ツール出力は WM までで止める
            return None
        if note.pin_flag:
            return 1
        # モード不一致は対象外
        if note.mode != mode:
            return None
        if is_create_mode(mode) and current_project_id is not None:
            if note.project_id != current_project_id:
                return None
        return 2

    # ── スコアリング ─────────────────────────────────────────────────

    def _score_fact(self, fact: SemanticFact) -> float:
        """ファクトのスコア (高いほど優先)。

        ``confidence`` を基準とし、``access_count`` の対数増分と recency
        による減衰を加える。Tier 1 配置時に pinned ボーナスを足す。
        """
        base = float(fact.confidence)
        base += 0.2 * math.log1p(max(0, fact.access_count))
        base += self._recency_term(fact.accessed_at)
        if fact.pinned:
            base += PINNED_BONUS
        if getattr(fact, "from_correction", False):
            base += CORRECTION_BONUS
        return base

    def _score_note(self, note: MemoryNote) -> float:
        base = float(note.confidence)
        base += 0.2 * math.log1p(max(0, note.access_count))
        base += self._recency_term(note.accessed_at)
        if note.pin_flag:
            base += PINNED_BONUS
        return base

    def _recency_term(self, accessed_at: float) -> float:
        if accessed_at <= 0:
            return 0.0
        age_days = max(0.0, (self._now_provider() - accessed_at) / 86400.0)
        # 半減期に基づく指数減衰 (0..1)
        return math.exp(-age_days * math.log(2) / _RECENCY_HALF_LIFE_DAYS)

    # ── レンダリング ─────────────────────────────────────────────────

    def _render_fact(self, fact: SemanticFact) -> str:
        age = self._fact_age_days(fact)
        corrected = bool(getattr(fact, "from_correction", False))
        if age is None or age < _FACT_STALE_LABEL_DAYS:
            if corrected:
                # 同じ日に古い値と並ぶと、下の「N日前の記録」も付かないため
                # どちらが現在値かを示す手掛かりが行に無くなる。
                return (
                    f"- ({fact.type}) {fact.subject} {fact.predicate}:"
                    f" {fact.text} (訂正後の記録)"
                )
            return f"- ({fact.type}) {fact.subject} {fact.predicate}: {fact.text}"
        if corrected:
            return (
                f"- ({fact.type}) {fact.subject} {fact.predicate}: {fact.object}"
                f" (訂正後の記録・{int(age)}日前)"
            )
        # 何日前の記録かを行ごとに書く。ノート側 (_render_note の (過去の記録))
        # と同じ理由で、ブロック先頭の注意書きは数百トークン離れると効かない。
        #
        # 実インシデント (2026-08-08 ライブ監査 ターン34): 同一セッションの
        # 前半で「趣味は登山と写真」と伝えていたが、その発言は 25 ターンの
        # WorkingMemory 窓から押し出されており、3 日前の別セッションで記録した
        # 「趣味は自転車と写真」が検索で 1.000 の最上位に立った。**どちらが今の
        # 話かを示す情報が行に無かった**ため、古い方がそのまま回答になった。
        return (
            f"- ({fact.type}) {fact.subject} {fact.predicate}: {fact.object}"
            f" ({int(age)}日前の記録)"
        )

    def _fact_age_days(self, fact: SemanticFact) -> float | None:
        """ファクトが記録されてからの経過日数 (未記録なら ``None``)。"""
        created_at = float(getattr(fact, "created_at", 0.0) or 0.0)
        if created_at <= 0:
            return None
        return max(0.0, (self._now_provider() - created_at) / 86400.0)

    def _collapse_to_current_values(
        self, facts: Iterable[SemanticFact],
    ) -> tuple[list[SemanticFact], int, set[str]]:
        """1 スロット 1 値へ畳む (純粋関数的。入力は変更しない)。

        2 段階で落とす:

        1. **完全重複** — ``(subject, predicate, object)`` が同一の行。
        2. **古い世代** — 同じ ``(subject, predicate)`` に異なる値が並ぶ場合、
           ``created_at`` が最新のものだけ残す。

        ``subject`` は ``mem.personal.name`` / ``mem.world.url.<hash>`` のように
        名前空間化されており、``(subject, predicate)`` が**そのファクトのスロット
        識別子**である (競合検出 ``semantic_conflict_resolver._group_conflicts``
        も同じキーでグループ化する)。したがって同一スロットに live な値が複数
        並ぶのは「世代が溜まっている」状態であって、並列に主張してよい別々の
        事実ではない。

        実測 (2026-08-09、実ファクトストア): live 88 件 / 38 スロットに対し、
        **43 件 (49%) が完全重複**だった。同じ文が 2〜3 件ずつ複製されており、

        - 注入予算の約半分を複製が食う
        - **多数決が壊れる** — 「趣味は自転車と写真」(3 複製) が
          「趣味は登山と写真」(2 複製) を回数で上回り、実機で古い方が採用された
          (ライブ監査 ターン34)。意味ではなく複製数の差で負けていた

        異なる値を持つスロットは 4 件しかなく、その大半は「質問がファクト化
        されたもの」(``carries_no_assertion`` が読込時に落とす) だった。真に
        矛盾していたのは ``mem.personal.name`` のみで、最新 (08-08 12:58
        「登山と写真」) が実際に正しい値だった。

        なぜ注入側でやるか: 競合検出は同じ矛盾を既に検出済みだが、
        ``_decide`` が ``any(f.pinned)`` で ``pinned_present`` として自動解決を
        見送るため pending のまま滞留する。そして pin は「覚えておいてください」
        から**自動的に**付く。解決を待つ間も**矛盾を並べて注入してはいけない**
        ので、注入の入口で 1 値に畳む。ユーザーへの確認は競合レビュー側の
        担当で、こちらはそれと独立に成立する。

        Returns:
            ``(残したファクト, 落とした行数, 却下した世代の本文集合)``。
            3 つ目は STM ノート側の重複抑止に使う (``_is_stale_duplicate``)。
        """
        by_slot: dict[tuple[str, str], SemanticFact] = {}
        seen_exact: set[tuple[str, str, str]] = set()
        out: list[SemanticFact] = []
        stale_texts: set[str] = set()
        dropped = 0
        for fact in facts:
            if fact.superseded_by:
                out.append(fact)          # 既存ループ側で落とす (カウント二重計上を避ける)
                continue
            # **主張を含まない行はスロットの代表になれない。**
            #
            # 畳み込みは「最新を残す」ので、質問が最新だとそれがスロットを取り、
            # 後段の carries_no_assertion で落とされて **スロットごと消える**。
            # 実インシデント (2026-08-09): mem.personal.birthday に
            #   「私の誕生日は 3 月 14 日で、飼っている猫の名前はコトラです。」(答え)
            #   「私の猫の名前と誕生日を覚えていますか。」(質問・同日で後勝ち)
            # が並び、質問が代表になった結果「猫の名前は？」に対して答えが 1 件も
            # 注入されず「文脈が不足しています」と回答した (類似度 0.490 で
            # 関連度ゲートは通っていた)。判定順序だけの問題。
            if carries_no_assertion(fact.object or ""):
                out.append(fact)          # 既存ループ側の同じ判定に委ねる
                continue
            exact = (fact.subject, fact.predicate, fact.object)
            if exact in seen_exact:
                dropped += 1
                continue
            seen_exact.add(exact)
            slot = (fact.subject, fact.predicate)
            current = by_slot.get(slot)
            if current is None:
                by_slot[slot] = fact
                out.append(fact)
                continue
            dropped += 1
            if float(getattr(fact, "created_at", 0.0) or 0.0) > float(
                getattr(current, "created_at", 0.0) or 0.0,
            ):
                stale_texts.add(_normalize_for_dup(current.object))
                out[out.index(current)] = fact
                by_slot[slot] = fact
            else:
                stale_texts.add(_normalize_for_dup(fact.object))
        if dropped:
            logger.debug(
                "MemoryInjector: collapsed %d duplicate/stale fact rows", dropped,
            )
        return out, dropped, stale_texts

    @staticmethod
    def _is_stale_duplicate(content: str, stale_texts: set[str]) -> bool:
        """却下した世代と**同一本文**の STM ノートか。

        ファクトと STM ノートは同じ発話を別々のストアに持つ。世代をファクト側で
        畳んでも、同じ文がノート経由でそのまま戻ってくる (実測 2026-08-09:
        「趣味は自転車と写真」のファクトを落とした後もノートが同文を注入していた)。
        却下済みの本文と一致するノートは、判断済みの内容の再提示でしかない。

        完全一致 (空白正規化のみ) に限る。曖昧一致にすると、言い換えただけの
        別の発言まで巻き込む。
        """
        return bool(stale_texts) and _normalize_for_dup(content) in stale_texts

    def _render_note(self, note: MemoryNote) -> str:
        """STM ノートを 1 行にレンダリングする。

        ノート本文は過去セッションの生の会話テキストで、「先ほど〜と言いました
        が訂正します」のような一人称・時制表現をそのまま含む。``(note)`` という
        中立ラベルだと、ブロック先頭の注意書きから数百トークン離れた位置では
        効かず、モデルが「先ほど話した〜」と今回の会話の発言として誤って帰属
        する (実インシデント 2026-07-27 ライブ検証)。行ごとに過去の記録である
        ことを示す。
        """
        return f"- (過去の記録) {note.content}"

    # ── パッキング ───────────────────────────────────────────────────

    def _pack_tier(
        self,
        items: list[InjectedItem],
        cap: int,
    ) -> tuple[list[InjectedItem], list[InjectedItem]]:
        """1 Tier 分の greedy パッキング。

        スコア降順に詰め、cap 超過分は dropped に回す。
        """
        items.sort(key=lambda it: it.score, reverse=True)
        accepted: list[InjectedItem] = []
        dropped: list[InjectedItem] = []
        used = 0
        for it in items:
            if it.tokens <= 0:
                # 空テキスト等は無視
                continue
            if used + it.tokens <= cap:
                accepted.append(it)
                used += it.tokens
            else:
                dropped.append(it)
        return accepted, dropped

    def _spill_from_tier4(self, plan: InjectionPlan, budget: int) -> None:
        """総予算超過時に Tier 4 から低スコア順に削除する。"""
        # Tier 4 をスコア昇順に並べ、削除対象を決定
        tier4 = [it for it in plan.items if it.tier == 4]
        tier4.sort(key=lambda it: it.score)
        removed_ids: set[str] = set()
        for it in tier4:
            if plan.used_tokens <= budget:
                break
            removed_ids.add(it.item_id)
            plan.used_tokens -= it.tokens
            plan.dropped.append(it)
        if removed_ids:
            plan.items = [
                it for it in plan.items
                if not (it.tier == 4 and it.item_id in removed_ids)
            ]

    # ── 補助 ─────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_ratios(
        raw: list | tuple | None,
        default: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        if not raw or len(raw) != 4:
            return default
        try:
            r = tuple(float(x) for x in raw)
        except (TypeError, ValueError):
            return default
        return r  # type: ignore[return-value]

    @staticmethod
    def _tier_budgets(total: int, ratios: tuple[float, ...]) -> list[int]:
        """各 Tier の token cap を整数で返す。

        合計が total を超えないよう ``int()`` で切り捨てる。
        """
        return [max(0, int(total * r)) for r in ratios]
