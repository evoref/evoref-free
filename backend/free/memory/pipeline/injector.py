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

import re
import math
import time

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, Sequence

from backend.free.core.intent_vocab import is_plain_statement
from backend.free.core.text_quality import (
    carries_no_assertion,
    is_payload_dump,
    looks_like_task_log_residue,
    states_no_user_value,
)
from backend.free.core.session_mode import (
    is_chat_mode,
    is_create_mode,
    is_valid_session_mode,
)
from backend.free.memory.attribute_key import (
    NON_ATTRIBUTE_TAILS,
    attribute_key,
)
from backend.free.memory.notes.subject_ns import is_session_summary_subject
from backend.free.memory.stores.short_term import MemoryNote
from backend.free.memory.types import MemoryMode, SemanticFact
from backend.i18n_helper import prompt_locale
from backend.log_config import get_logger
from backend.utils import estimate_tokens

logger = get_logger("memory.injector")

#: 行末 / 行頭に添える固定文 (``i18n.prompt_locale`` 別)。``ja`` が従来出力。
#: ``note`` の英語ラベル ``(past record)`` は ``inference._normalize_for_frame_dedup``
#: が [参考情報] との重複判定で剥がすので、変えるときは両方を揃える。
_RENDER_LABELS: dict[str, dict[str, str]] = {
    "ja": {
        "corrected": " (訂正後の記録)",
        "corrected_aged": " (訂正後の記録・{age}日前)",
        "aged": " ({age}日前の記録)",
        "note": "- (過去の記録) {content}",
    },
    "en": {
        "corrected": " (corrected record)",
        "corrected_aged": " (corrected record, {age} days ago)",
        "aged": " (recorded {age} days ago)",
        "note": "- (past record) {content}",
    },
}


def _render_labels() -> dict[str, str]:
    """prompt_locale に応じたラベル辞書 (未知 locale は ja)。"""
    return _RENDER_LABELS.get(prompt_locale(), _RENDER_LABELS["ja"])

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


#: セッション要約ファクトの判定は :func:`is_session_summary_subject` (SSOT は
#: ``notes.subject_ns``)。会話のメタ記録であってユーザーについての事実ではない
#: ため [関連する記憶] へ注入しない (``inject`` 内の判定参照)。
#:
#: **接頭辞リテラルを持たない** — 要約の subject は
#: ``mem.<decision|commitment>.history.session.<id12>`` で型が本文から推定される
#: ため、片方をリテラルで書くと他方が素通りする。実際
#: ``"mem.decision.history.session"`` と書かれていた期間、
#: ``mem.commitment.history.session.*`` だけが注入され、訂正前の値を運び続けた
#: (:data:`~backend.free.memory.notes.subject_ns.SESSION_SUMMARY_SUBJECT_RE`)。

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

#: URL リコール索引の subject 接頭辞 (``memory.sleep.url_curator``)。
#: executable command 索引と同型で、``object`` は過去のユーザーの質問文
#: (``answers_topic``)。読み手は ``ToolCallJudge`` の URL リコールだけ。
#: 接頭辞リストから漏れていたため ``world_fact`` として [関連する記憶] へ
#: 出ていた (2026-09-02 監査 M21)。
_URL_INDEX_SUBJECT_PREFIX = "mem.world.url."

#: 内部索引の subject 接頭辞。いずれも「アシスタント側の記録」であって
#: ユーザーについての事実ではないため、ユーザーに見える枠へ出さない。
#:
#: 消費側は 2 つ: ``[関連する記憶]`` (本モジュールの :meth:`MemoryInjector.inject`)
#: と ``[記憶の競合]`` (``conflict_review.collect_review_groups``)。片方だけに
#: 掛けると同じ内容が別の窓から出る — 実際 2026-08-19 時点の pending は全 2 件が
#: セッション要約で、競合セクション側から素通しになっていた。
INTERNAL_INDEX_SUBJECT_PREFIXES: tuple[str, ...] = (
    _EPISODE_TRACE_SUBJECT_PREFIX,
    _EXECUTABLE_COMMAND_SUBJECT_PREFIX,
    _URL_INDEX_SUBJECT_PREFIX,
)


def is_internal_index_subject(subject: str) -> bool:
    """``subject`` が内部索引 (ユーザー向けに出さない記録) のものか。

    セッション要約は型が可変なので接頭辞ではなく形で判定する
    (:func:`~backend.free.memory.notes.subject_ns.is_session_summary_subject`)。
    エピソードトレースと executable command 索引は subject が固定接頭辞。

    **両方の消費側 (注入 / 競合レビュー) は必ずこの関数を通す** — 片方だけに
    掛けると同じ内容が別の窓から出る (:data:`INTERNAL_INDEX_SUBJECT_PREFIXES`
    のコメント参照)。
    """
    if not subject:
        return False
    return (
        is_session_summary_subject(subject)
        or subject.startswith(INTERNAL_INDEX_SUBJECT_PREFIXES)
    )


#: 属性スロットを持つユーザーファクトの型 (``fact_attributes.yaml`` の chat 節)。
#: ``extractors.chat._USER_SUBJECT_TAGS`` と同じ集合 — 片方だけ増やさないこと。
_USER_ATTRIBUTE_FACT_TYPES: tuple[str, ...] = (
    "personal_fact",
    "preference",
    "emotion",
    "opinion",
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
    #: ファクトの属性スロット名 (``location`` / ``schedule`` …)。ノートは ``None``。
    #: 採用された item だけを見て ``InjectionPlan.covered_attributes`` を作る。
    attribute: str | None = None


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

    #: 実際に注入されたファクトの属性スロット名 (``location`` / ``schedule`` …)。
    #: 「この属性の現在値はもうプロンプトに載っている」を呼出側が判定するための
    #: 出力。``dropped`` になった候補は含めない — 載っていないものを載ったと
    #: 数えると、履歴検索の抑止が「答えを持たないまま撃たない」に化ける。
    covered_attributes: set[str] = field(default_factory=set)

    def render(self) -> str:
        """注入対象を 1 つのテキストに連結する (改行区切り)。"""
        return "\n".join(it.text for it in self.items)

    def by_tier(self, tier: int) -> list[InjectedItem]:
        return [it for it in self.items if it.tier == tier]


# ──────────────────────────────────────────────────────────────────────────
# MemoryInjector
# ──────────────────────────────────────────────────────────────────────────


#: スロット同定に使えない汎用の末尾セグメント。
#: SSOT は :mod:`backend.free.memory.attribute_key` (後方互換の別名)。
_GENERIC_SUBJECT_TAIL: frozenset[str] = NON_ATTRIBUTE_TAILS


#: 「ほぼ同じ本文」とみなす bi-gram Jaccard の下限。``ChunkContentGate`` の
#: ``dedup_jaccard`` と同じ棒。意味的に近いだけの別内容を巻き込まないため
#: 高めに置く (相対でなく絶対だが、Jaccard は 0..1 の比なのでスケール依存が無い)。
NEAR_DUPLICATE_JACCARD: float = 0.85

#: 「尋ねられている属性」に一致するファクトへ足すスコア。
#:
#: 他の加点 (confidence 最大 1.0 + access_count の対数 + recency 最大 1.0 +
#: pinned/correction ボーナス) の合計を確実に上回る量にする。属性一致は
#: 埋め込みのスケールに依存しない決定論の根拠なので、確率的なスコアと
#: 競わせる意味が無い (競わせると実測で飲み物が趣味に勝つ)。
_ASKED_ATTRIBUTE_BONUS: float = 100.0

#: 語彙アンカーで通した候補に足すスコア。属性一致 (決定論) ほど強くはないが、
#: コサインだけで並ぶ候補より上に出す。
_ANCHOR_BONUS: float = 50.0

#: クエリから取り出す **内容語**。2 文字以上の漢字 / カタカナ / 英数字の連なり。
#: 1 文字を採らないのは助詞・接辞の断片が全文にマッチしてしまうため。
_QUERY_ANCHOR_RE = re.compile(r"[一-鿿]{2,}|[ァ-ヴー]{2,}|[A-Za-z0-9]{2,}")

#: 想起の **足場語**。どの想起クエリにも現れるので、これが一致しても
#: 「その話題だ」とは言えない。アンカーから除く。
_ANCHOR_SCAFFOLD: frozenset[str] = frozenset({
    "自分", "今回", "会話", "記録", "情報", "内容", "以前", "過去", "確認",
    "教えて", "何度", "全部", "一度", "本当", "具体", "詳細", "最初", "最後",
    "さっき", "先ほど", "いま", "現在",
})


def query_anchors(query_text: str) -> tuple[str, ...]:
    """クエリの内容語 (語彙アンカー) を返す (純粋関数)。

    埋め込みのスケールに依存しない **決定論の根拠**。属性辞書に無い話題
    (「蕎麦」「苦手」「約束」) はコサインでしか拾えず、実測ではその
    コサインが背景と重なって届かない (下記 :meth:`_is_relevant` の測定)。
    """
    if not query_text:
        return ()
    return tuple({
        w for w in _QUERY_ANCHOR_RE.findall(query_text)
        if w not in _ANCHOR_SCAFFOLD
    })


def _has_anchor(text: str, anchors: tuple[str, ...]) -> bool:
    """``text`` に語彙アンカーのいずれかがそのまま出現するか。"""
    return bool(anchors) and any(a in text for a in anchors)

#: 近似重複判定は O(n^2)。注入候補がこの件数を超えたら判定ごと見送る
#: (実運用の候補数は数十件で、超えるのは異常系)。
_NEAR_DUP_MAX_ITEMS: int = 200


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """2 つのトークン集合の Jaccard 係数 (純粋関数)。"""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if not inter:
        return 0.0
    return inter / len(a | b)


#: subject から属性名を取り出す純粋関数。SSOT は
#: :mod:`backend.free.memory.attribute_key` — 「同じスロットか」の判定を
#: 注入 / 競合 / 影の回収で分裂させないため (本モジュールの旧実装と同値)。
_attribute_key = attribute_key


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
        #: 次元不一致 WARNING の重複抑止 (_note_dim_mismatch)。
        self._dim_mismatch_warned: bool = False

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

    @staticmethod
    def _restated_slots(
        session_user_texts: Iterable[str], mode: MemoryMode,
    ) -> set[tuple[str, str]]:
        """今回の会話でユーザーが値を述べ直した ``(fact_type, attribute)`` 集合。

        属性の同定は ``fact_attributes.yaml`` の決定論辞書
        (:func:`resolve_fact_attribute`) をそのまま使う — 抽出側が subject を
        決めるのと同じ辞書なので、注入側だけ別基準になることがない。
        LLM 呼び出しは無い。
        """
        from backend.free.memory.notes.note_builder import (
            MAX_ASKED_ATTRIBUTES,
            resolve_fact_attribute_matches,
        )

        slots: set[tuple[str, str]] = set()
        for text in session_user_texts or ():
            if not text:
                continue
            # **問い・依頼は「値の述べ直し」ではない。**
            #
            # 属性辞書は「どの属性の話か」しか見ないので、
            # 「私が住んでいるのはどこでしたか。」も location に解決される。
            # その結果この判定が真になり、**想起の質問そのものが「もう述べた」
            # 扱いになって記憶が抑止される** — 想起したいときほど記憶が消える
            # という逆転が起きる。
            #
            # 実インシデント (2026-08-25 ライブ監査): 新規セッションで
            # 「私が住んでいるのはどこでしたか。」「私の好きな飲み物は？」
            # 「私の趣味は何でしたか。」等 6 問すべてに「確認できる情報を
            # 持ち合わせていません」と回答した。ストアには正しいファクトが
            # あり、属性も正しく解決できていた (asked_attrs=['location'])。
            # 落ちていたのはこの判定だけ。
            #
            # 判定は既存の決定論ヘルパを使う (実測で問い 3 件 / 言明 2 件を
            # 完全分離)。語彙を新しく列挙しない。
            #
            # **否定形の防御だけでは漏れ続ける。** 上の 2 つは「問い / 依頼の
            # 形」を列挙して落とす閉じた語彙で、外れた瞬間に想起クエリが
            # 「述べ直し」に化け、**答えを持つファクトが消える**。
            #
            # 実インシデント (2026-08-30 ライブ監査 T12#7): 「私の名前を、いま
            # 確実に知っていますか。根拠も。」は ``states_no_user_value`` /
            # ``carries_no_assertion`` の**どちらにも当たらず**
            # (末尾が「根拠も。」で問いの形をしていない)、``name`` スロットが
            # restated 扱いになった。``asked_attrs=['name']`` まで正しく出て
            # いたのに、属性免除に到達する前にファクトが落ちて
            # ``attr_exempt=0 items=0``。実機の回答は「いいえ、知りません。」。
            #
            # そこで **肯定の証拠を要求する** 側へ反転する。抑止は
            # 「ユーザーがこのセッションで値を言明した」ときだけ働けばよく、
            # 判定を外した場合の代償は非対称:
            #
            #   - 抑止し損ね → 古い値が新しい値と並ぶ (ラベルとプロンプト規範
            #     で「今回の会話が優先」と示されており、実害は小さい)
            #   - 誤って抑止 → **答えそのものが消える** (上の実インシデント)
            #
            # ``is_plain_statement`` は平叙文かどうかだけを見る既存の決定論
            # ヘルパ。実測 16 件 (言明 6 / 想起の問い 10) で完全分離した。
            if not is_plain_statement(text):
                continue
            if states_no_user_value(text) or carries_no_assertion(text):
                continue
            # **複数形の解決を使う。** 単数版は YAML 記載順で最初の 1 件を
            # 返して打ち切るため、1 発話で複数の属性を述べると **先頭以外の
            # スロットは抑止されず、古い値が並ぶ**。日本語の自己紹介は
            # 1 発話に複数属性を詰めるのが普通なので、単数版では常に漏れる。
            # 抽出側は同じ理由で 2026-08-25 に、``_asked_attributes`` は
            # 2026-08-30 に複数形へ移行済みで、抑止側だけ単数のまま残っていた。
            #
            # 実測 (2026-09-04): 「私は小川です。横浜に住んでいます。仕事は
            # バックエンドエンジニアです。」→ 単数版は ``location`` だけ、
            # 複数版は ``location`` + ``occupation``。単数版のままだと前
            # セッションの ``mem.personal.occupation`` が抑止されずに並ぶ。
            for fact_type in _USER_ATTRIBUTE_FACT_TYPES:
                for slug, _ in resolve_fact_attribute_matches(
                    text, fact_type, mode=mode, limit=MAX_ASKED_ATTRIBUTES,
                ):
                    slots.add((fact_type, slug))
        return slots

    @staticmethod
    def _slot_restated_in_session(
        fact: SemanticFact, restated: set[tuple[str, str]],
    ) -> bool:
        """``fact`` の属性スロットが今回の会話で述べ直されているか。"""
        if not restated:
            return False
        fact_type = str(getattr(fact, "type", "") or "")
        subject = str(getattr(fact, "subject", "") or "")
        if not fact_type or "." not in subject:
            return False
        attr = subject.rsplit(".", 1)[-1]
        return (fact_type, attr) in restated

    @staticmethod
    def _asked_attributes(query_text: str, mode: MemoryMode) -> set[str]:
        """発話が **どの属性を尋ねているか** を決定論辞書で解決する。

        書き込み側 (抽出器が subject を決める) と読み出し側で同じ辞書
        (``fact_attributes.yaml`` / :func:`resolve_fact_attribute`) を使う。
        「私の趣味は何でしたか。」→ ``{"hobby"}``、
        「私が住んでいるのはどこでしたか。」→ ``{"location"}``。

        **なぜ要るか**: 記憶は正規化された三人称の命題として保存される
        (「小川宏之は埼玉県川口市に住んでいる。」) が、想起クエリは一人称
        (「私が住んでいるのはどこでしたか。」)。実測 (2026-08-25 ライブ監査、
        LFM2.5-Embedding-350M):

            正規化後(三人称) の保存文 vs 一人称クエリ … cos +0.098
            一人称のままの文   vs 同じクエリ        … cos +0.330

        較正済みの関連度フロアは 0.29 なので、**正規化した瞬間にフロアの下へ
        落ちる**。しかも順位も壊れていて、「私の趣味は何でしたか。」に対し
        飲み物のファクト (+0.352) が趣味のファクト (+0.228) を上回っていた。
        実機では 6 問中 6 問が「確認できる情報を持ち合わせていません」。

        属性名の一致は **埋め込みのスケールに依存しない決定論の根拠** なので、
        ここだけはコサインの棒を免除できる (RAG 側の語彙アンカーと同じ立て付け)。
        """
        if not query_text:
            return set()
        from backend.free.memory.notes.note_builder import (
            MAX_ASKED_ATTRIBUTES,
            resolve_fact_attribute_matches,
        )

        attrs: set[str] = set()
        for fact_type in _USER_ATTRIBUTE_FACT_TYPES:
            try:
                # **複数形の解決を使う。** 単数版は YAML 記載順で最初の 1 件を
                # 返して打ち切るため、1 つの問いが 2 つ以上の属性を尋ねると
                # **先頭以外は免除されず、コサインのゲートに落ちる**。
                #
                # 実インシデント (2026-08-30 ライブ監査の検証 V4):
                # 「私の勤務地と居住地をもう一度。」→ ``work_location`` だけ、
                # 「私の名前、住所、職業、ペットをもう一度。」→ ``occupation``
                # だけが解決し、実機は「確認できていません」と答えた。抽出側は
                # 同じ理由で 2026-08-25 に複数形へ移行済みで、読み出し側だけ
                # 単数のまま残っていた。
                matches = resolve_fact_attribute_matches(
                    query_text, fact_type, mode=mode,
                    # 読み出しは書き込みの上限に縛られない
                    # (``MAX_ASKED_ATTRIBUTES`` の説明を参照)。
                    limit=MAX_ASKED_ATTRIBUTES,
                )
            except Exception:
                continue
            attrs.update(slug for slug, _ in matches)
        return attrs

    @staticmethod
    def _fact_attribute(fact: SemanticFact) -> str | None:
        """ファクトの属性名 (subject 末尾)。取り出せなければ ``None``。"""
        return _attribute_key(str(getattr(fact, "subject", "") or ""))

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
        session_user_texts: Iterable[str] = (),
        query_text: str = "",
        retired_note_ids: "set[str] | None" = None,
        fact_relevance_scores: "dict[str, float] | None" = None,
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
            session_user_texts: 今回の会話でユーザーが述べた本文の列。ここに
                同じ属性スロットの言明があるファクトは注入しない
                (:meth:`_restated_slots` を参照)。
            retired_note_ids: 値が supersede された発話ノートの ID 集合。
                SemMem 側で世代を閉じても STM ノートの原文は残るため、
                訂正前の発話が「現在値」として Tier 2 に載り続ける。
                実機検証 (2026-08-27) では occupation が supersede 済みでも
                「職業はデータベース管理者で、名古屋に住んでいます。」の
                ノートから訂正前の値が返っていた。履歴は消さず、
                **現在値として黙って提示する経路だけ**を止める。
            query_text: 現在のユーザー発話の本文。**どの属性を尋ねているか**を
                決定論辞書 (:func:`resolve_fact_attribute`) で解決し、一致する
                ファクトを関連度ゲートから免除する (:meth:`_asked_attributes`)。
            fact_relevance_scores: ``{fact_id: cosine}`` の事前計算済みスコア。
                ``SemanticFactStore.embedding_scores(query_vec)`` の出力を
                そのまま渡す。埋め込み行列を持っているストア側で 1 回の積を
                取るのが最も安いため (:meth:`_relevance_scores` の実測)。
                ``None`` なら従来どおり候補ごとに判定する。
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
        retired_dropped = 0
        retired = retired_note_ids or set()

        # Tier ごとに分類
        buckets: dict[int, list[InjectedItem]] = {1: [], 2: [], 3: [], 4: []}

        # ``states_no_user_value`` は正規表現の束で、以前は畳み込み 2 段 +
        # 注入ループの 3 箇所で同じファクトに対して計 3 回走っていた。
        # ファクト単位で 1 回に畳む (id(fact) 引き)。
        no_value: dict[int, bool] = {}

        # 属性またぎの畳み込み (``_collapse_by_attribute``) に参加させるファクト。
        # 型はユーザー属性 4 種に限る (sleep-time 側の
        # ``semantic_conflict_resolver._CROSS_SLOT_TYPES`` と同じ集合) — 別型の
        # ``mem.project.<id>.location`` や ``loop.task.<x>.location`` が末尾
        # セグメントの一致だけで ``mem.personal.location`` を畳み、しかも自分は
        # chat で注入されない (``_classify_fact`` が None) ため、スロットごと
        # 消えていた。**注入されない側が注入される側を抑えてはいけない** ので、
        # 畳み込みは「この mode で配置対象になる」ファクト同士に限る。
        def _collapsible(fact: SemanticFact) -> bool:
            if fact.type not in _USER_ATTRIBUTE_FACT_TYPES:
                return False
            if str(fact.subject or "").startswith(("loop.", "learn.")):
                return False
            return self._classify_fact(
                fact, mode, current_project_id, sigs,
            ) is not None

        facts, collapsed, stale_texts = self._collapse_to_current_values(
            facts, collapsible=_collapsible, no_value_cache=no_value,
        )
        filtered_out += collapsed
        # 関連度スコアは候補ごとの numpy 演算ではなく 1 回の行列積で求める
        # (:meth:`_relevance_scores`)。判定の意味は変わらない。
        fact_scores = self._relevance_scores(
            query_vec, facts, fact_relevance_scores,
        )
        restated = self._restated_slots(session_user_texts, mode)
        asked_attrs = self._asked_attributes(query_text, mode)
        attr_exempt = 0
        anchors = query_anchors(query_text)
        anchor_exempt = 0
        # **話題を持たない継続指示には記憶を注入しない。**
        #
        # 「表にしてください。」「もう一度お願いします。」のように内容語を
        # 1 つも持たない指示は、直前のターンを指す照応でしかない。ところが
        # 埋め込みは中身の無い文にも何らかのベクトルを与えるので、関連度ゲート
        # だけでは **無関係なファクトが通る**。しかも注入された行は会話履歴より
        # プロンプトの近くに置かれるため、モデルの話題ごと持っていかれる。
        #
        # 実インシデント (2026-09-04 ライブ監査 T05#1): pytest のテストケースを
        # 3 つ挙げた直後の「表にしてください。」に対し、注入は
        # ``anchors=- asked_attrs=- items=7`` (名古屋 / Rust / 好きな色は緑 …)
        # となり、実機はテストケースではなく **ユーザープロフィールの表** を
        # 出した。
        #
        # 判定は既存の決定論ヘルパだけで組む (語彙を新しく列挙しない):
        # 属性を 1 つも尋ねておらず (``asked_attrs``)、内容語アンカーも無く
        # (``anchors``)、かつ本文が値を述べていない (``states_no_user_value``)
        # ときに限る。「私の趣味は？」は ``asked_attrs`` が、「インデックスの話
        # を 3 行で。」は ``anchors`` が立つので影響を受けない。
        #
        # **``pinned`` も免除しない。** pin は「優先度」の宣言であって話題との
        # 関連の証拠ではない。しかも pin 検出は「覚えて」「重要」等の語で
        # 発火するので、**訂正文や依頼文がそのまま自動 pin される** (同ファイル
        # の ``carries_no_assertion`` / ``states_no_user_value`` が pin を
        # 例外にしないのと同じ理由)。実測 (2026-09-04 修正後の再現): pin を
        # 免除したところ ``topicless=228`` で 228 件落としてなお 5 件
        # (「好きな色は緑」「会社は名古屋」…) が残り、話題の無い指示に
        # 陳腐なプロフィールが並び続けた。次の内容のあるターンで戻る。
        topicless_directive = bool(
            query_text
            and not asked_attrs
            and not anchors
            and states_no_user_value(query_text)
        )
        topicless_dropped = 0

        for fact in facts:
            if fact.superseded_by:
                continue
            if topicless_directive:
                filtered_out += 1
                topicless_dropped += 1
                continue
            # 今回の会話で同じ属性スロットを述べ直しているファクトは注入しない。
            # ラベルには「今回の会話と食い違えば今回が優先」と書いてあるが、
            # 規範文では勝てない — 実インシデント (2026-08-23 ライブ監査
            # セット 2): 「私は今、埼玉県川口市に住んでいます。」と述べた後の
            # 「私が住んでいるのはどこでしたか？」に対し、前セッションの
            # ``mem.personal.location states: …武蔵野市に引っ越しました。`` が
            # 勝って「武蔵野市です」と答えた。数値ラベル版の同じ対処
            # (``core.inference._drop_superseded_context``) を属性スロットへ
            # 一般化したもの。
            if self._slot_restated_in_session(fact, restated):
                filtered_out += 1
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
            # 依頼形も落とす (``states_no_user_value``)。「私の好きな飲み物を
            # もう一度教えてください。」が ``mem.personal.beverage states: …``
            # として **断定形**で並び、本人の実際の言明と競合していた
            # (2026-08-19 実測: 実ストアの依頼形 21 件中 14 件が飲み物スロット)。
            # ノート側 (下) は ``carries_no_assertion`` のまま — 依頼は
            # 「後から引きたい記録」としては正当なので落とさない。
            if self._no_user_value(fact, no_value):
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
            if is_internal_index_subject(fact.subject):
                filtered_out += 1
                continue
            gate_reached += 1
            # 尋ねられている属性と subject の属性が一致するファクトは、
            # コサインの棒を免除する (:meth:`_asked_attributes` の実測を参照)。
            asked_this = bool(
                asked_attrs and self._fact_attribute(fact) in asked_attrs
            )
            # 語彙アンカー: クエリの内容語がファクト本文にそのまま出ていれば、
            # 属性辞書に無い話題でも「その話をしている」ことが決定論で言える。
            anchored = not asked_this and _has_anchor(
                str(getattr(fact, "object", "") or ""), anchors,
            )
            if asked_this:
                attr_exempt += 1
            elif anchored:
                anchor_exempt += 1
            elif not self._passes_gate(
                query_vec, fact, pinned=bool(fact.pinned), scores=fact_scores,
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
            if anchored:
                score += _ANCHOR_BONUS
            if asked_this:
                # 順位も直す。免除しただけでは Tier 内の並びで負ける
                # (実測: 「私の趣味は何でしたか。」で飲み物 +0.352 が
                #  趣味 +0.228 を上回っていた)。属性一致は決定論の根拠なので、
                #  スコアの下駄ではなく **確実に先頭へ** 出す量を足す。
                score += _ASKED_ATTRIBUTE_BONUS
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
                    attribute=self._fact_attribute(fact),
                ),
            )

        stm_notes = list(stm_notes)
        # STM ノートは埋め込み行列を持つストアが無いので事前計算できない。
        # 候補ごとの判定のまま (件数は ``memory.max_notes`` で抑えられている)。
        note_scores: dict[int, float] = {}
        for note in stm_notes:
            pinned = bool(getattr(note, "pin_flag", False))
            if topicless_directive:
                filtered_out += 1
                topicless_dropped += 1
                continue
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
            # 同文がそのまま出力された)。
            #
            # **pin を例外にしない。** ファクト側は 2026-08-07 に同じ結論へ
            # 到達済み — pin 検出は「重要」「覚えて」等の語で発火するため、
            # **問い自体がその語を含むと自動で pin され、この 2 つのガードを
            # まるごと迂回する**。ノート側だけ ``not pinned and`` が残っていた。
            #
            # 実測 (2026-08-30 ライブ監査、300 ターン): 注入された記憶行
            # 165 件のうち **106 件 (64%) がユーザー自身の過去の問い / 依頼**
            # で、上位 4 件はすべて pin 免除で通っていた:
            #
            #   38 回「訂正前の値を覚えていますか。…」        (pin 語「覚え」)
            #   29 回「その重要な点だけを補足として3行で…」  (pin 語「重要」)
            #   22 回「圧縮で落とした情報のうち、重要な…」    (pin 語「重要」)
            #   17 回「いま答えた内容に、私が訂正した古い値は…」
            #
            # いずれも ``states_no_user_value`` / ``carries_no_assertion`` が
            # 正しく真を返しており、落ちていたのは pin 免除だけ。本物のファクトは
            # 1〜6 回しか載らず、予算を問いに食われていた。
            # ノート自体は残るので履歴検索からは従来どおり引ける。
            if carries_no_assertion(getattr(note, "content", "")):
                filtered_out += 1
                continue
            # 依頼だけのノートも同じ理由で ``(過去の記録)`` に載せない。
            # ``carries_no_assertion`` は「問いだけか」しか見ないので、
            # 「〜を全部挙げてください」型の依頼が素通りする。
            #
            # 実インシデント (2026-08-23 ライブ監査セット 2 ターン 1): 挨拶
            # 「おはようございます。改めて、私は小川宏之です。」に 11 件が注入され、
            # うち「この会話で私が訂正した項目はいくつありますか？内容も挙げて
            # ください。」「ここまでで私について覚えたことを、箇条書きで全部挙げて
            # ください。」など **過去のユーザーの依頼文** が (過去の記録) として
            # 並んでいた。依頼は事実を含まないので根拠にならず、本物のファクトを
            # 予算から押し出す。ノート自体は残るので履歴検索からは引ける。
            # pin を例外にしない理由は上の ``carries_no_assertion`` と同じ。
            if states_no_user_value(getattr(note, "content", "")):
                filtered_out += 1
                continue
            # アシスタント自身の発話は ``(過去の記録)`` の根拠にしない。
            #
            # 派生物であって出典ではない。ユーザーの言明を復唱しただけのものは
            # **同じ事実の重複** になり (2026-08-23 実測: PC 環境の 2 文が
            # [関連する記憶] と [参考情報] に計 4 コピー)、過去の回答が誤って
            # いた場合は **作話が「過去の記録」に昇格** する (同監査セット 1 の
            # 「確信度は100%です。あなたが来月出張する都市は東京です。」が
            # セット 2 で過去の記録として注入された。正解は大阪)。
            # system プロンプトは既に「自分自身の過去の発言をそのまま繰り返さない」
            # と書いているが、規範文では勝てない。
            if not pinned and getattr(note, "source", "user") == "assistant":
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
            # 内部タスク進捗ノート (「- [done] … / Written N bytes to …」) は
            # ユーザー向け本文ではない。記録側の浄化漏れで STM に入ってしまうと
            # (過去の記録) として注入され、**実行していない操作の完了報告**を
            # モデルに手本として与える。実インシデント 2026-08-22 ライブ監査:
            # ``- [done] Confirm the file <path> has been deleted``
            # が注入され、削除ツールが無いターンで「削除しました。」と答えた
            # (ファイルは残存)。記録側 (meta_cognitive_recorded_text) を直しても
            # 既存ノートは寿命が尽きるまで残るため、読み出し側でも落とす。
            if looks_like_task_log_residue(getattr(note, "content", "")):
                filtered_out += 1
                continue
            # 値が supersede された発話は「現在値」として提示しない。
            # SemMem 側で世代を閉じても STM の原文は残るため、訂正前の発話が
            # Tier 2 に載り続ける (retired_note_ids の説明を参照)。
            # ``pinned`` も免除しない — ユーザーが pin したのは当時の値で、
            # 本人が後から訂正している。
            if retired and note.id in retired:
                filtered_out += 1
                retired_dropped += 1
                continue
            gate_reached += 1
            if not self._passes_gate(
                query_vec, note,
                # MemoryNote 側の pin 属性は ``pin_flag`` (SemanticFact は ``pinned``)。
                pinned=pinned,
                scores=note_scores,
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

        # 近似重複の抑止 (tier をまたいで掛ける)。
        #
        # 完全一致の重複はファクト側 (_collapse_to_current_values) と
        # ノート側 (_is_stale_duplicate) で既に落としているが、**言い回しが
        # 少し違うだけの同一内容**は素通りする。しかも 2 つのストアは同じ発話を
        # 別々に持つので、同じ事実がファクト行とノート行で並ぶ (実測 2026-08-23:
        # PC 環境の 2 文が [関連する記憶] と [参考情報] に計 4 コピー)。
        # 予算を食うだけでなく、**同じ主張が複数回現れることが重み付けとして
        # 働く** — 複製数の多い古い値が新しい値に勝つ壊れ方は既に一度起きている。
        #
        # 意味的に近いだけの別内容まで巻き込まないよう、判定は「ほぼ同じ本文」
        # に限る (bi-gram 集合の Jaccard、既定 0.85 = ChunkContentGate と同じ棒)。
        dedup_dropped = self._suppress_near_duplicates(buckets)

        plan = InjectionPlan(
            mode=mode,
            budget_tokens=budget,
            tier_budgets=tier_budgets,
        )
        # item_id (= fact.id) → 属性スロット。パック後に **採用されたものだけ**
        # を covered_attributes へ移す。
        attr_by_fact_id = {
            it.item_id: it.attribute
            for tier_items in buckets.values() for it in tier_items
            if it.source == "fact" and it.attribute
        }

        # Tier 1 → 4 の順にパック
        for tier in (1, 2, 3, 4):
            cap = tier_budgets[tier - 1]
            accepted, dropped = self._pack_tier(buckets[tier], cap)
            plan.items.extend(accepted)
            plan.dropped.extend(dropped)
            plan.used_tokens += sum(it.tokens for it in accepted)

        plan.covered_attributes = {
            attr for it in plan.items
            if (attr := attr_by_fact_id.get(it.item_id))
        }

        # 総予算オーバー時は Tier 4 から削除
        if plan.used_tokens > budget:
            self._spill_from_tier4(plan, budget)

        logger.debug(
            "MemoryInjector.inject: mode=%s budget=%d used=%d items=%d "
            "dropped=%d project=%s sigs=%d relevance=%s filtered=%d "
            "near_dup=%d asked_attrs=%s attr_exempt=%d anchors=%s "
            "anchor_exempt=%d retired=%d topicless=%d",
            mode, budget, plan.used_tokens, len(plan.items),
            len(plan.dropped), current_project_id, len(sigs),
            "on" if query_vec is not None else "off", filtered_out,
            dedup_dropped, sorted(asked_attrs) or "-", attr_exempt,
            sorted(anchors) or "-", anchor_exempt,
            retired_dropped, topicless_dropped,
        )
        if retired_dropped:
            # 訂正が効いているかを実機で追えるようにする (沈黙で落とさない)。
            logger.info(
                "MemoryInjector: dropped %d STM note(s) whose value was "
                "superseded by a later correction",
                retired_dropped,
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

    def relevant_fact_ids(
        self,
        facts: Iterable[SemanticFact],
        query_embedding: "np.ndarray | None",
        fact_relevance_scores: "dict[str, float] | None" = None,
        *,
        require_embedding: bool = False,
    ) -> set[str] | None:
        """関連度ゲートを通ったファクト id を返す。ゲート無効時は ``None``。

        ``require_embedding=True`` で、スコアも埋め込みも無いファクトを
        「判定不能なので通す」ではなく落とす (競合セクション用。予算の外で
        毎ターン連結される枠なので、判定できないものを通すと無関係な矛盾が
        全ターンに載る)。``pinned`` は従来どおり通す。

        Tier パッキングを経由しない注入経路 (``[記憶の競合]`` セクション) が、
        注入本体と **同じ棒** を使うための入口。``None`` は「ゲートを掛けられ
        なかった」を意味し、呼出側は従来どおり全件を通す。

        競合セクションは予算の外で毎ターン連結されるため、ゲートが無いと
        クエリと無関係な矛盾が全ターンのプロンプトに載り続ける (実測
        2026-08-19: 飲み物の競合 2 件が、飲み物と無関係な 28/29 ターンへ
        注入されていた)。
        """
        query_vec = self._prepare_query_vec(query_embedding)
        if query_vec is None:
            return None
        materialized = list(facts)
        scores = self._relevance_scores(
            query_vec, materialized, fact_relevance_scores,
        )
        return {
            f.id
            for f in materialized
            if self._passes_gate(
                query_vec, f,
                pinned=bool(getattr(f, "pinned", False)), scores=scores,
                require_embedding=require_embedding,
            )
        }

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

    def _relevance_scores(
        self,
        query_vec: "np.ndarray | None",
        items: "Sequence[Any]",
        precomputed: "dict[str, float] | None" = None,
    ) -> "dict[int, float]":
        """候補の関連度スコアを ``id(item)`` 引きの辞書にまとめる。

        速い経路は **ストア側で計算済みのスコアを受け取る** こと
        (``precomputed``、:meth:`SemanticFactStore.embedding_scores`)。
        埋め込み行列は ``EmbeddingStore`` に (N, dim) の連続配列として常駐して
        いるので、そこで 1 回の積を取るのが最も安い。

        実測 (2026-09-01、1024 次元): 候補ごとに ``np.asarray`` →
        ``np.linalg.norm`` → 除算 を回すと N=10000 で 52.5ms、常駐行列との
        積なら 1.5ms (**35 倍**)。**候補リストからその都度行列を組み直す方式は
        採らない** — ``np.stack`` のコピーが乗るぶん候補ごとのループより
        むしろ遅い (実測 0.48〜0.76 倍で全スケール敗北)。行列を持っていない
        側でベクトル化しても勝てない。

        ``precomputed`` に無い候補 (STM ノート / 別スコープ / 未 embed) は
        辞書に載せず、呼出側が :meth:`_is_relevant` の候補ごとの分岐へ落ちる。
        """
        if query_vec is None or not precomputed:
            return {}
        out: dict[int, float] = {}
        for item in items:
            fid = getattr(item, "id", None)
            if not fid:
                continue
            score = precomputed.get(fid)
            if score is not None:
                out[id(item)] = score
        return out

    def _passes_gate(
        self,
        query_vec: "np.ndarray | None",
        item: "Any",
        *,
        pinned: bool,
        scores: "dict[int, float]",
        require_embedding: bool = False,
    ) -> bool:
        """事前計算したスコアで関連度ゲートを判定する。

        スコアが引けている候補はその場で閾値比較し、引けなかった候補
        (埋め込み無し / 次元不一致 / ゼロベクトル) だけ :meth:`_is_relevant`
        の分岐へ落とす。判定の意味は :meth:`_is_relevant` と同一。
        """
        if query_vec is None:
            return True
        score = scores.get(id(item))
        if score is None:
            return self._is_relevant(
                query_vec, getattr(item, "embedding", None),
                pinned=pinned, require_embedding=require_embedding,
            )
        if pinned:
            return score >= self.pinned_relevance_min_score
        return score >= self.relevance_min_score

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
            # **次元違いは「判定不能」ではなく「壊れている」。**
            #
            # embed モデルを次元違いのものへ替えると、再 embed が済むまで既存
            # ファクトは旧空間のベクトルを持つ。ここを ``True`` (素通し) に
            # していたため、ゲートが最も要る状態で **ストア全件が毎ターン
            # 注入されていた** — 関連度ゲートを入れた元の事象そのもの
            # (実測 2026-09-01: 4096 次元のベクトルに対し 1024 次元のクエリで
            # ``passes gate: True``)。
            #
            # ``embedding is None`` (まだ生成されていない) を通すのは、Step 8.8
            # が次サイクルで必ず埋めるという前提があるから。次元違いには
            # その前提が無く、``reembed-facts`` を人が走らせるまで直らない。
            # 落とす側 = 記憶が出なくなるのは観測できるが、通す側 = 無関係な
            # 記憶が全部載るのは静かに品質だけ壊す。
            self._note_dim_mismatch(int(w.shape[0]), int(query_vec.shape[0]))
            return False
        norm = float(np.linalg.norm(w))
        if not norm or not np.isfinite(norm):
            return True
        score = float(query_vec @ (w / norm))
        if pinned:
            return score >= self.pinned_relevance_min_score
        return score >= self.relevance_min_score

    def _note_dim_mismatch(self, fact_dim: int, query_dim: int) -> None:
        """次元不一致を 1 インスタンスにつき 1 回だけ WARNING に出す。

        毎ファクト出すとログが埋まるので、``inject`` 1 回あたり 1 行に畳む。
        """
        if self._dim_mismatch_warned:
            return
        self._dim_mismatch_warned = True
        logger.warning(
            "Memory injection: fact embedding dim %d != query dim %d. "
            "These facts are excluded from injection until re-embedded. "
            "Run 'python scripts/evorefmem_cli.py reembed-facts --apply' "
            "(or POST /api/model/reembed-facts).",
            fact_dim, query_dim,
        )

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
        """1 ファクトを [関連する記憶] の 1 行へ整形する。

        本文は **必ず ``fact.text``** (= ``statement or object``) を使う。
        古い記録の 2 分岐だけが生の ``object`` を読んでおり、``statement`` が
        埋まると同じファクトが鮮度によって別の本文で出ていた
        (2026-08-26 に是正)。訂正ファクトの ``object`` は古い値を同居させた
        原文なので、この分岐だけ陳腐値を出し続ける形になっていた。
        """
        age = self._fact_age_days(fact)
        corrected = bool(getattr(fact, "from_correction", False))
        labels = _render_labels()
        if age is None or age < _FACT_STALE_LABEL_DAYS:
            if corrected:
                # 同じ日に古い値と並ぶと、下の「N日前の記録」も付かないため
                # どちらが現在値かを示す手掛かりが行に無くなる。
                return (
                    f"- ({fact.type}) {fact.subject} {fact.predicate}:"
                    f" {fact.text}{labels['corrected']}"
                )
            return f"- ({fact.type}) {fact.subject} {fact.predicate}: {fact.text}"
        if corrected:
            return (
                f"- ({fact.type}) {fact.subject} {fact.predicate}: {fact.text}"
                f"{labels['corrected_aged'].format(age=int(age))}"
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
            f"- ({fact.type}) {fact.subject} {fact.predicate}: {fact.text}"
            f"{labels['aged'].format(age=int(age))}"
        )

    def _fact_age_days(self, fact: SemanticFact) -> float | None:
        """ファクトが記録されてからの経過日数 (未記録なら ``None``)。"""
        created_at = float(getattr(fact, "created_at", 0.0) or 0.0)
        if created_at <= 0:
            return None
        return max(0.0, (self._now_provider() - created_at) / 86400.0)

    @staticmethod
    def _supersedes(candidate: SemanticFact, current: SemanticFact) -> bool:
        """同一スロットで ``candidate`` が ``current`` を置き換えるべきか。

        ``created_at`` の新しい方を残すのが基本だが、**同着を落としてはいけない**。
        1 回の sleep-time バッチで抽出されたファクトは秒未満の差しか持たず、
        実測では ``created_at`` が完全に一致する。訂正は同じセッションで起きる
        ため、まさにその同着に当たる。

        実インシデント (2026-08-22 ライブ監査 2 回目): ``mem.preference.beverage``
        ``prefers`` に

          - 「私はコーヒーより紅茶が好きです。」 (created_at 1787362137.6008642)
          - 「さっき「紅茶が好き」と言いましたが、やっぱりコーヒーの方が好きです。」
            (created_at 1787362137.6017072)

        が並び、厳密な ``>`` 比較では **先に見た訂正前の値が残った**。ファクトは
        会話順に append されるので、同着なら **後から来た方が新しい**。加えて
        ``from_correction`` が立っている行は訂正そのものなので、同着では常に
        優先する (同一バッチ内で順序が入れ替わっても壊れないようにする)。
        """
        cand_at = float(getattr(candidate, "created_at", 0.0) or 0.0)
        cur_at = float(getattr(current, "created_at", 0.0) or 0.0)
        if cand_at != cur_at:
            return cand_at > cur_at
        cand_corr = bool(getattr(candidate, "from_correction", False))
        cur_corr = bool(getattr(current, "from_correction", False))
        if cand_corr != cur_corr:
            return cand_corr
        # 同着かつ訂正フラグも同じ — 抽出順 (= 会話順) で後の方が新しい。
        return True

    @staticmethod
    def _no_user_value(fact: SemanticFact, cache: "dict[int, bool] | None") -> bool:
        """``states_no_user_value(fact.object)`` をファクト単位で 1 回だけ評価する。"""
        if cache is None:
            return states_no_user_value(fact.object or "")
        key = id(fact)
        hit = cache.get(key)
        if hit is None:
            hit = states_no_user_value(fact.object or "")
            cache[key] = hit
        return hit

    def _collapse_to_current_values(
        self,
        facts: Iterable[SemanticFact],
        *,
        collapsible: "Callable[[SemanticFact], bool] | None" = None,
        no_value_cache: "dict[int, bool] | None" = None,
    ) -> tuple[list[SemanticFact], int, set[str]]:
        """1 スロット 1 値へ畳む (純粋関数的。入力は変更しない)。

        ``collapsible`` は属性またぎの畳み込み (:meth:`_collapse_by_attribute`)
        に参加させるファクトの述語。``None`` なら全件 (旧挙動、テスト互換)。
        ``no_value_cache`` は ``states_no_user_value`` の評価結果を共有する
        辞書 (:meth:`_no_user_value`)。

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
        #: スロット -> ``out`` 内の位置。``out.index()`` は線形探索なので
        #: 畳み込み全体が O(n^2) になる (``semmem_limits`` の既定まで伸びると
        #: 効いてくる)。差し替え先を辞書で覚えておく。
        slot_pos: dict[tuple[str, str], int] = {}
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
            # 後段の states_no_user_value で落とされて **スロットごと消える**。
            # 実インシデント (2026-08-09): mem.personal.birthday に
            #   「私の誕生日は 3 月 14 日で、飼っている猫の名前はコトラです。」(答え)
            #   「私の猫の名前と誕生日を覚えていますか。」(質問・同日で後勝ち)
            # が並び、質問が代表になった結果「猫の名前は？」に対して答えが 1 件も
            # 注入されず「文脈が不足しています」と回答した (類似度 0.490 で
            # 関連度ゲートは通っていた)。判定順序だけの問題。
            if self._no_user_value(fact, no_value_cache):
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
                slot_pos[slot] = len(out)
                out.append(fact)
                continue
            dropped += 1
            if self._supersedes(fact, current):
                stale_texts.add(_normalize_for_dup(current.object))
                out[slot_pos[slot]] = fact
                by_slot[slot] = fact
            else:
                stale_texts.add(_normalize_for_dup(fact.object))
        # ここまでは ``(subject, predicate)`` 単位。**同じ属性が別スロットへ
        # 分散する** ケースはこれでは畳めない。実測 (2026-08-22 ライブ監査
        # 2 回目、実ファクトストア): 「好きな飲み物」が
        # ``mem.personal.beverage`` / ``mem.preference.beverage`` /
        # ``mem.personal.user`` / ``mem.preference.user`` の 4 スロットへ散り、
        # 緑茶 / ほうじ茶 / 紅茶 / コーヒーの歴代 4 世代がすべて live のまま
        # 注入されていた。その結果、同じ会話で「緑茶」に訂正した直後の
        # 「好きな飲み物は？」に **数日前の「ほうじ茶」**、まとめでは
        # 「コーヒー」と答えた。属性名 (subject の末尾セグメント) で
        # もう一段畳んで最新 1 値に寄せる。
        out, attr_dropped = self._collapse_by_attribute(
            out, stale_texts, collapsible=collapsible, no_value_cache=no_value_cache,
        )
        dropped += attr_dropped
        if dropped:
            logger.debug(
                "MemoryInjector: collapsed %d duplicate/stale fact rows", dropped,
            )
        return out, dropped, stale_texts

    def _collapse_by_attribute(
        self,
        facts: list[SemanticFact],
        stale_texts: set[str],
        *,
        collapsible: "Callable[[SemanticFact], bool] | None" = None,
        no_value_cache: "dict[int, bool] | None" = None,
    ) -> tuple[list[SemanticFact], int]:
        """属性名 (subject 末尾) をまたいだ世代を 1 値へ畳む。

        ``_attribute_key`` が ``None`` を返す subject (階層が浅い / 汎用語) は
        対象外。会話要約 (``mem.decision.history.session.<id>``) は末尾が
        セッション ID なので互いに衝突せず、そのまま残る。

        ``collapsible`` が偽を返すファクトは畳み込みに **参加しない** (勝者にも
        敗者にもならず、そのまま残る)。``mem.project.<id>.location`` のような
        別型 / 別名前空間のファクトが、末尾セグメントの一致だけで
        ``mem.personal.location`` を消していた (2026-09-02 監査 S-A2)。
        """
        by_attr: dict[str, SemanticFact] = {}
        attr_pos: dict[str, int] = {}
        out: list[SemanticFact] = []
        dropped = 0
        for fact in facts:
            if getattr(fact, "superseded_by", None) or self._no_user_value(
                fact, no_value_cache,
            ):
                out.append(fact)
                continue
            if collapsible is not None and not collapsible(fact):
                out.append(fact)
                continue
            key = _attribute_key(fact.subject)
            if key is None:
                out.append(fact)
                continue
            current = by_attr.get(key)
            if current is None:
                by_attr[key] = fact
                attr_pos[key] = len(out)
                out.append(fact)
                continue
            dropped += 1
            if self._supersedes(fact, current):
                stale_texts.add(_normalize_for_dup(current.object))
                out[attr_pos[key]] = fact
                by_attr[key] = fact
            else:
                stale_texts.add(_normalize_for_dup(fact.object))
        return out, dropped

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
        return _render_labels()["note"].format(content=note.content)

    # ── パッキング ───────────────────────────────────────────────────

    def _suppress_near_duplicates(
        self, buckets: dict[int, list["InjectedItem"]],
    ) -> int:
        """tier をまたいで近似重複を落とす (buckets を破壊的に更新)。

        走査は「上位 tier / 高スコア順」で、先に採った項目に似すぎている後続を
        落とす。判定は bi-gram 集合の Jaccard で、``NEAR_DUPLICATE_JACCARD``
        以上を「ほぼ同じ本文」とみなす。

        Returns:
            落とした項目数。
        """
        order = [
            (tier, item)
            for tier in (1, 2, 3, 4)
            for item in sorted(buckets.get(tier, []), key=lambda it: -it.score)
        ]
        if len(order) <= 1 or len(order) > _NEAR_DUP_MAX_ITEMS:
            return 0
        try:
            from backend.free.rag.bm25_retriever import tokenize_ja
        except Exception:  # 語彙トークナイザが無い構成では抑止しない
            return 0

        kept: dict[int, list["InjectedItem"]] = {1: [], 2: [], 3: [], 4: []}
        kept_tokens: list[frozenset[str]] = []
        dropped = 0
        for tier, item in order:
            tokens = frozenset(tokenize_ja(item.text))
            if not tokens:
                kept[tier].append(item)
                continue
            if any(
                _jaccard(tokens, other) >= NEAR_DUPLICATE_JACCARD
                for other in kept_tokens
            ):
                dropped += 1
                continue
            kept_tokens.append(tokens)
            kept[tier].append(item)
        if dropped:
            for tier in (1, 2, 3, 4):
                buckets[tier] = kept[tier]
            logger.debug(
                "MemoryInjector: suppressed %d near-duplicate item(s)", dropped,
            )
        return dropped

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
