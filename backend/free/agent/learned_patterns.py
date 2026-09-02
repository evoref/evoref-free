"""学習済みパターンストア: 会話から学習したパターン検出キーワードの管理

ツールルーティング (``tool_routing``) / 長文ルーティング (``long_form``) の
シグナルからキーワードを学習し、次回以降のルーティング判定に活用する。
LLM 呼び出しなし。旧 ``correction`` / ``rephrase`` カテゴリは 2026-07-21 に
廃止 (書き手・読み手とも存在せず、話題語学習による訂正誤検出の温床だった —
``_DISCONTINUED_CATEGORIES`` 参照)。
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

from backend.free.agent.learned_pattern_store import LearnedPatternRepository
from backend.free.agent.learned_patterns_types import LearnedPattern
from backend.free.document_nouns import DOCUMENT_NOUN_LEARNABLE_JA
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.free.core.policy_interpreter import PolicyInterpreter

logger = get_logger("agent.learned_patterns")

# デフォルト設定
DEFAULT_MAX_PATTERNS = 200
DEFAULT_INITIAL_WEIGHT = 0.5
DEFAULT_DECAY_RATE = 0.05
DEFAULT_BOOST_AMOUNT = 0.15
DEFAULT_MIN_WEIGHT = 0.1
DEFAULT_MATCH_THRESHOLD = 0.3

#: ``decay_all`` を sleep-time から適用する最小間隔 (秒)。
#:
#: Step 5.5 は Light ごと (= LLM 生成のたび) に走るため、ターン単位で減衰
#: すると初期重み 0.5 の語は 8 ターンで min_weight を割って消える
#: (2026-09-02 監査 R-C1: 「学習した語が翌日には無い」)。減衰は「使われない
#: まま時間が経った」の代理なので、壁時計で 6 時間に 1 回へ絞る。学習の
#: 設定キーには載せない (schema 変更を避ける) 定数。
PATTERN_DECAY_INTERVAL_SEC = 6 * 3600

# 意図キーワード抽出用パターン（動詞・指示語を重点抽出）
_INTENT_PATTERNS = [
    # 日本語動詞・指示語（「〜して」「〜しろ」「〜してください」等の語幹）
    re.compile(r"([\u4e00-\u9fff]{1,4})[しさせ]て"),
    re.compile(r"([\u4e00-\u9fff]{1,4})[しさせ]ろ"),
    re.compile(r"([\u4e00-\u9fff]{1,4})[しさせ]てください"),
    # カタカナ動作語
    re.compile(r"([\u30a0-\u30ff]{2,})して"),
    re.compile(r"([\u30a0-\u30ff]{2,})する"),
    # 英語動詞（先頭の動詞を抽出）
    re.compile(r"\b(add|remove|delete|update|fix|change|create|append|insert|replace|move|copy|rename|merge)\b", re.IGNORECASE),
]

# 汎用キーワード抽出（NoteBuilder 互換）
_KEYWORD_PATTERNS = [
    re.compile(r"[A-Za-z][A-Za-z0-9_.-]+"),
    re.compile(r"[\u4e00-\u9fff]{2,8}"),
    re.compile(r"[\u30a0-\u30ff]{2,}"),
]

# ストップワード（パターンとして学習しない一般的な語）
_STOPWORDS = frozenset({
    # 日本語
    "これ", "それ", "あれ", "ここ", "そこ", "この", "その",
    "ある", "いる", "なる", "する", "できる", "ない", "です",
    "ます", "した", "して", "から", "まで", "より", "ため",
    "こと", "もの", "ところ", "とき", "よう", "ほう",
    # 英語
    "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "this", "that",
    "with", "from", "for", "not", "but", "and", "or",
})

# long_form カテゴリ専用の学習除外語 (CJK)。出力先指定などで頻出するが、
# 文書(散文)の種別を示さない汎用ファイル操作語。ASCII トークンは
# ``_LONG_FORM_ASCII_FILEISH_RE`` 側で一律除外するためここには含めない。
# 2026-07-15 に「文書 0.90 / 明日 0.66 / 共有 0.66 / 形式 0.54 / 部署 0.54」等の
# 汎用語が自己強化され、朝礼メモ・12 行 CSV 依頼までユニット分割パイプラインへ
# 誘導した実績があるため、時間表現・組織語・汎用文書語も除外する。
_LONG_FORM_NONLEARNABLE_EXACT = frozenset({
    "出力", "出力先", "保存", "保存先", "ファイル",
    "書き出し", "書き込み", "配下", "エクスポート", "セーブ",
    # 文書「種別」を示さない汎用文書語
    "文書", "書類", "形式", "内容", "作成", "一覧", "雛形", "ひな形",
    # 時間表現 (依頼の期日であって文書種別ではない)
    "今日", "明日", "昨日", "今週", "来週", "今月", "来月", "本日",
    # 組織・共有語 (社内文書の話題語であって種別ではない)
    "共有", "連絡", "連絡事項", "部署", "担当", "社内", "全社",
})
# ASCII のみのトークン (パス片 / URL 片 / ホスト名 / 形式名 / 識別子) を表す。
# long_form の文書種別シグナルとしては信頼できないため学習対象から外す。
_LONG_FORM_ASCII_FILEISH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")

# 廃止カテゴリ (2026-07-21)。correction は「訂正が起きたときの話題語」を
# 「訂正の言い回し」として照合する意味論の破綻により偽陽性率 ~85% (経験 65 件
# の実測) に達し、learned correction 検出 (feedback 旧・層2) ごと廃止した。
# rephrase は書き込み専用で match() の参照箇所が存在しない dead カテゴリ
# だった。load 時に落とし、廃止前に学習された残存データ (「会話」w=0.95 等)
# を再起動時に自動浄化する。
_DISCONTINUED_CATEGORIES = frozenset({"correction", "rephrase"})

# tool_routing カテゴリ専用の学習除外語。ツールシグナル付きクエリに随伴して
# 抽出される (「main.py を実行して結果を説明して」→「実行」「説明」) が、
# それ自体は知識質問にも頻出する LLM ネイティブな言語タスク語であり、ツール
# 意図のシグナルにならない。実インシデント (2026-07-20 ライブ検証): 過去に
# 学習された「説明」(w=0.630) が知識質問「〜を説明して」にマッチし、create
# モードで引数なし run_command 誘導を誘発し得た。add 時 (extract_tool_routing_
# keywords) と load 時の両方で弾き、既学習の残存データも再起動時に自動浄化する。
_TOOL_ROUTING_NONLEARNABLE_EXACT = frozenset({
    "説明", "解説", "要約", "紹介", "定義", "比較", "翻訳",
})


__all__ = ["LearnedPattern", "LearnedPatternStore"]


class LearnedPatternStore:
    """学習済みパターン検出キーワードの永続化管理

    訂正・言い直し検出時に会話から抽出したキーワードを蓄積し、
    次回以降のフィードバック検出で活用する。
    """

    def __init__(
        self,
        config: dict | None = None,
        policy: PolicyInterpreter | None = None,
    ):
        self._policy = policy
        cfg = (config or {}).get("learning", {})
        self.max_patterns: int = cfg.get("pattern_max_patterns", DEFAULT_MAX_PATTERNS)
        self.initial_weight: float = cfg.get("pattern_initial_weight", DEFAULT_INITIAL_WEIGHT)
        self.decay_rate: float = cfg.get("pattern_decay_rate", DEFAULT_DECAY_RATE)
        self.boost_amount: float = cfg.get("pattern_boost_amount", DEFAULT_BOOST_AMOUNT)
        self.min_weight: float = cfg.get("pattern_min_weight", DEFAULT_MIN_WEIGHT)
        self.match_threshold: float = cfg.get("pattern_match_threshold", DEFAULT_MATCH_THRESHOLD)

        self._patterns: dict[str, LearnedPattern] = {}
        # 直近に ``decay_all`` を適用した壁時計。起動時刻で初期化し、
        # ``load`` でファイルに残した値があればそれで上書きする (6h 間隔が
        # 再起動で仕切り直されないようにする。欠損 = 旧形式は「今」扱い)。
        self._last_decay_at: float = time.time()
        # 重み / 件数が変わったか (Step 5.5 が「変更があった時だけ保存」する印)。
        # ``match`` のヒット統計は含めない — 毎ターン変わる値で毎ターン書くのを
        # 避ける (次の実変更時にまとめて永続化される)。
        self._dirty: bool = False

    @property
    def count(self) -> int:
        return len(self._patterns)

    @property
    def dirty(self) -> bool:
        """前回の ``save`` 以降にパターンの追加 / 重み変更 / 削除があったか。"""
        return self._dirty

    @property
    def patterns(self) -> dict[str, LearnedPattern]:
        return self._patterns

    def add_pattern(
        self,
        keyword: str,
        category: str = "correction",
        weight: float | None = None,
    ) -> LearnedPattern:
        """パターンを追加または既存のパターンを強化する

        Args:
            keyword: 学習するキーワード
            category: カテゴリ ("tool_routing" | "long_form")。既定値の
                "correction" は廃止カテゴリ (後方互換のため温存。プロダクション
                呼出は全て明示指定で、省略で add しても save → load 時に
                ``_DISCONTINUED_CATEGORIES`` フィルタで自動浄化される)
            weight: 初期重み（None でデフォルト値）

        Returns:
            追加/更新されたパターン
        """
        key = keyword.lower()
        if key in _STOPWORDS or len(key) < 2:
            return LearnedPattern(keyword=keyword)

        now = time.time()

        if key in self._patterns:
            pattern = self._patterns[key]
            pattern.source_count += 1
            pattern.last_seen = now
            # 重みをブースト（上限 1.0）
            pattern.weight = min(1.0, pattern.weight + self.boost_amount)
            logger.debug(
                "Boosted pattern '%s': weight=%.3f, sources=%d",
                key, pattern.weight, pattern.source_count,
            )
        else:
            pattern = LearnedPattern(
                keyword=keyword,
                category=category,
                weight=weight if weight is not None else self.initial_weight,
                source_count=1,
                first_seen=now,
                last_seen=now,
            )
            self._patterns[key] = pattern
            logger.info("Learned new pattern: '%s' (category=%s)", keyword, category)

        self._dirty = True
        # 上限超過時は低重みパターンを削除
        self._enforce_limit()
        return pattern

    def match(
        self,
        text: str,
        category: str | None = None,
        *,
        record_hit: bool = True,
    ) -> list[tuple[str, float]]:
        """テキストに対して学習済みパターンをマッチングする

        Args:
            text: マッチング対象テキスト
            category: フィルタするカテゴリ（None で全カテゴリ）
            record_hit: ``hit_count`` / ``last_hit`` を更新するか。実際の
                ルーティング判定 (tool judge / router) では True。評価・
                Level 1 の再判定など「観測するだけ」の呼び出しでは False を
                渡し、読むだけで統計が動かないようにする。

        Returns:
            マッチしたパターンのリスト [(keyword, weight), ...]
        """
        if not self._patterns:
            return []

        text_lower = text.lower()
        matches: list[tuple[str, float]] = []
        # 既にヒットした文字区間。同一スパンから重複カウントしない
        # (「技術的な議論」に対し「技術」と「技術的」が 2 語成立し、
        #  learned 語 2 件で long_form が発火していた。2026-07-25)。
        # 長いキーワードを優先するため長さ降順に評価する。
        claimed: list[tuple[int, int]] = []

        for key, pattern in sorted(
            self._patterns.items(), key=lambda kv: -len(kv[0]),
        ):
            if pattern.weight < self.match_threshold:
                continue
            if category is not None and pattern.category != category:
                continue
            start = text_lower.find(key)
            if start < 0:
                continue
            end = start + len(key)
            if any(s <= start and end <= e for s, e in claimed):
                continue
            claimed.append((start, end))
            matches.append((pattern.keyword, pattern.weight))
            if record_hit:
                pattern.hit_count += 1
                pattern.last_hit = time.time()

        return sorted(matches, key=lambda x: -x[1])

    def boost(self, keyword: str, amount: float | None = None) -> None:
        """指定パターンの重みをブーストする

        Args:
            keyword: ブースト対象のキーワード
            amount: ブースト量（None でデフォルトの boost_amount）
        """
        key = keyword.lower()
        if key not in self._patterns:
            return
        amt = amount if amount is not None else self.boost_amount
        pattern = self._patterns[key]
        pattern.weight = min(1.0, pattern.weight + amt)
        self._dirty = True
        logger.debug("Boosted pattern '%s': weight=%.3f (+%.3f)", key, pattern.weight, amt)

    def decay_one(self, keyword: str, amount: float | None = None) -> None:
        """指定パターンの重みを減衰させる

        min_weight を下回ったパターンは削除する。

        Args:
            keyword: 減衰対象のキーワード
            amount: 減衰量（None でデフォルトの decay_rate）
        """
        key = keyword.lower()
        if key not in self._patterns:
            return
        amt = amount if amount is not None else self.decay_rate
        pattern = self._patterns[key]
        pattern.weight -= amt
        self._dirty = True
        if pattern.weight < self.min_weight:
            del self._patterns[key]
            logger.debug("Removed pattern '%s' (below min_weight after decay)", key)
        else:
            logger.debug("Decayed pattern '%s': weight=%.3f (-%.3f)", key, pattern.weight, amt)

    def decay_all(self) -> int:
        """全パターンの重みを減衰させる（sleep-time で実行）

        Returns:
            削除されたパターン数
        """
        removed = 0
        to_remove: list[str] = []

        for key, pattern in self._patterns.items():
            pattern.weight -= self.decay_rate
            if pattern.weight < self.min_weight:
                to_remove.append(key)

        for key in to_remove:
            del self._patterns[key]
            removed += 1

        if self._patterns or removed:
            self._dirty = True
        if removed:
            logger.info("Decayed and removed %d patterns", removed)
        return removed

    def maybe_decay_all(self, now: float | None = None) -> int | None:
        """``PATTERN_DECAY_INTERVAL_SEC`` 以上経っていれば ``decay_all`` を適用する。

        sleep-time Light (Step 5.5) から毎回呼ばれる入口。間隔内なら何もせず
        ``None`` を返す (「減衰しなかった」と「減衰したが削除 0 件」を区別する)。

        Returns:
            適用時は削除されたパターン数、スキップ時は ``None``。
        """
        current = now if now is not None else time.time()
        if current - self._last_decay_at < PATTERN_DECAY_INTERVAL_SEC:
            return None
        self._last_decay_at = current
        return self.decay_all()

    def decrement_source_count(self, keyword: str) -> None:
        """パターンの source_count を減算する（カートリッジ unload 用）

        source_count が 0 になったパターンは削除する。
        """
        key = keyword.lower()
        if key not in self._patterns:
            return
        pattern = self._patterns[key]
        pattern.source_count -= 1
        self._dirty = True
        if pattern.source_count <= 0:
            del self._patterns[key]
            logger.debug("Removed pattern '%s' (source_count=0)", key)

    def extract_intent_keywords(self, text: str) -> list[str]:
        """テキストから意図キーワードを抽出する（LLM 不要）

        ユーザーの指示文から動作指示語を抽出し、パターン候補として返す。

        Args:
            text: ユーザーのクエリテキスト

        Returns:
            抽出されたキーワードリスト
        """
        keywords: list[str] = []

        # 意図パターンからの抽出（動詞・指示語重視）
        for pattern in _INTENT_PATTERNS:
            for m in pattern.finditer(text):
                kw = m.group(1) if m.lastindex else m.group(0)
                if kw.lower() not in _STOPWORDS and len(kw) >= 2:
                    keywords.append(kw)

        # 汎用キーワード抽出（NoteBuilder 互換、補完用）
        for pattern in _KEYWORD_PATTERNS:
            for m in pattern.finditer(text):
                kw = m.group(0)
                if kw.lower() not in _STOPWORDS and len(kw) >= 2:
                    keywords.append(kw)

        # 重複排除（大文字小文字無視）
        seen: set[str] = set()
        result: list[str] = []
        for kw in keywords:
            lower = kw.lower()
            if lower not in seen:
                seen.add(lower)
                result.append(kw)

        return result[:10]

    def extract_tool_routing_keywords(self, text: str) -> list[str]:
        """``category="tool_routing"`` として学習してよいキーワードを抽出する。

        ``extract_intent_keywords`` と異なり 3 条件で絞る:

        1. クエリ全体にツールシグナル (``tool_call_judge._query_has_tool_signal``)
           があること。雑談・知識質問からの誤学習を防ぐ (2026-07-18 に
           feedback 側へ入れたガードの一元化。従来 Level 1 バッチ側
           ``LearningScheduler._evolve_tool_routing_patterns`` にはこのガードが
           無く、feedback 側で弾いた経験からも再学習される穴があった)。
        2. 動作指示語 (``_INTENT_PATTERNS``) のみ。汎用キーワード
           (``_KEYWORD_PATTERNS``: 漢字連続・カタカナ・ASCII トークン) は
           クエリの話題名詞 (「天気」「歴史」「コーヒー」等) を拾ってしまい、
           ツール意図のシグナルにならないため使わない (実データ 2026-07-21:
           tool_routing 62 件中ほぼ全てが話題名詞の汚染だった)。
        3. LLM ネイティブな言語タスク語
           (``_TOOL_ROUTING_NONLEARNABLE_EXACT``) を除外する。

        Returns:
            学習に適したキーワードリスト (最大 10 件)。ツールシグナルの無い
            クエリでは空リスト。
        """
        # 同一パッケージ内の遅延 import (module ロード順の循環回避)
        from backend.free.agent.tool_call_judge import _query_has_tool_signal
        if not _query_has_tool_signal(text):
            return []
        seen: set[str] = set()
        result: list[str] = []
        for pattern in _INTENT_PATTERNS:
            for m in pattern.finditer(text):
                kw = m.group(1) if m.lastindex else m.group(0)
                lower = kw.lower()
                if lower in _STOPWORDS or len(kw) < 2:
                    continue
                if kw in _TOOL_ROUTING_NONLEARNABLE_EXACT:
                    continue
                if lower not in seen:
                    seen.add(lower)
                    result.append(kw)
        return result[:10]

    @staticmethod
    def is_long_form_learnable(keyword: str) -> bool:
        """``keyword`` を ``category="long_form"`` として学習してよいか判定する (pure)。

        long_form は文書(散文)生成の意図シグナルを学習するカテゴリ。
        **許容リスト方式** で、``DOCUMENT_NOUN_LEARNABLE_JA`` の文書種別名詞を
        含む語だけを学習可とする (仕様書 / レポート / 議事録 / ガイドライン 等)。

        2026-07-25 に除外リスト方式から反転した。除外リストは「新種の一般語が
        来るたび追記する」運用になり、2026-07-15 の対処後も「技術 / 方針 / 会話 /
        検索 / 議論 / 有益 / 簡潔 / 字以内 / 捏造 / 内線番号」等 52 語が学習され、
        謝辞 1 行が 7 ユニット 6,436 字の長文生成に化けた。未知語を既定で
        学習しない構造にしないと同型の再発が止まらない。

        割り切り: ASCII のみのトークン (python / Excel / wcup 等) は文書種別
        シグナルとして信頼できないため一律除外する (英語 long_form 語の学習は
        犠牲にする)。``tool_routing`` 等の他カテゴリには適用しない (ASCII 語が正当)。
        """
        kw = keyword.strip()
        if len(kw) < 2:
            return False
        if kw.lower() in _LONG_FORM_NONLEARNABLE_EXACT:
            return False
        if _LONG_FORM_ASCII_FILEISH_RE.match(kw):
            return False
        return any(noun in kw for noun in DOCUMENT_NOUN_LEARNABLE_JA)

    def _enforce_limit(self) -> None:
        """パターン数上限を超えた場合、低重みパターンを削除"""
        if len(self._patterns) <= self.max_patterns:
            return

        # 重みの低い順にソートして超過分を削除
        sorted_patterns = sorted(
            self._patterns.items(),
            key=lambda x: x[1].weight,
        )
        overflow = len(self._patterns) - self.max_patterns
        for key, _ in sorted_patterns[:overflow]:
            del self._patterns[key]
        self._dirty = True
        logger.info("Removed %d low-weight patterns (limit: %d)", overflow, self.max_patterns)

    def get_stats(self) -> dict:
        """統計情報を返す"""
        if not self._patterns:
            return {"count": 0, "categories": {}}

        categories: dict[str, int] = {}
        for p in self._patterns.values():
            categories[p.category] = categories.get(p.category, 0) + 1

        weights = [p.weight for p in self._patterns.values()]
        return {
            "count": len(self._patterns),
            "categories": categories,
            "weight_avg": sum(weights) / len(weights),
            "weight_min": min(weights),
            "weight_max": max(weights),
            "total_hits": sum(p.hit_count for p in self._patterns.values()),
        }

    def save(self, path: str | Path) -> None:
        """JSON ファイルに永続化する (infra 層に委譲)。常に書き、dirty を下ろす。

        ``_last_decay_at`` も同じファイルへ残し、減衰間隔を再起動越しに保つ。
        """
        LearnedPatternRepository.save(
            self._patterns, path, last_decay_at=self._last_decay_at,
        )
        self._dirty = False

    def load(self, path: str | Path) -> None:
        """JSON ファイルからロードする (infra 層に委譲)

        ドメインルール: ストップワード / 空 keyword / 廃止カテゴリ
        (``_DISCONTINUED_CATEGORIES``) のエントリはロード時に除外する。
        tool_routing カテゴリの言語タスク語
        (``_TOOL_ROUTING_NONLEARNABLE_EXACT``) も除外し、除外セット導入前に
        学習された残存データ (「説明」w=0.630 等) を再起動時に自動浄化する。
        """
        result = LearnedPatternRepository.load_with_meta(path)
        if result is None:
            return
        loaded, meta = result
        stored_decay = meta.get("last_decay_at")
        if isinstance(stored_decay, (int, float)) and stored_decay > 0:
            self._last_decay_at = float(stored_decay)

        self._patterns.clear()
        for key, pattern in loaded.items():
            if not key or key in _STOPWORDS:
                continue
            if pattern.category in _DISCONTINUED_CATEGORIES:
                logger.info(
                    "Dropped discontinued-category pattern on load: "
                    "'%s' (category=%s)", key, pattern.category,
                )
                continue
            if (
                pattern.category == "tool_routing"
                and key in _TOOL_ROUTING_NONLEARNABLE_EXACT
            ):
                logger.info(
                    "Dropped non-learnable tool_routing pattern on load: '%s'",
                    key,
                )
                continue
            self._patterns[key] = pattern
        self._dirty = False
        logger.info("Loaded %d learned patterns from %s", len(self._patterns), path)
