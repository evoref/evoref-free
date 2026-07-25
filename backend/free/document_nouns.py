"""長文/文書生成依頼を検出するための共有語彙。

``backend/free/agent/router.py`` の ``LONG_FORM_PATTERNS`` (EvorefLoop pillar)
と ``backend/free/generation/content_detector.py`` の ``TEXT_PATTERNS``
(EvorefGen pillar) が、それぞれ独立に「これは文書生成依頼か」を判定する
日本語名詞リストを保持しており、語彙が乖離していた (例: README/読本/計画書は
router のみ、議事録/テンプレート/送付状等は content_detector のみ)。

pillar 境界 (``backend/free/tests/test_pillar_boundary.py``) 上、gen pillar は
他 pillar を一切 import できず (``ALLOWED_CROSS_PILLAR_IMPORTS["gen"] ==
frozenset()``)、loop → gen 方向は許可されているが非対称になる。加えて
``backend.free.generation`` のどのサブモジュールを import しても
``generation/__init__.py`` 経由で長文生成エンジン一式 (orchestrator 等) が
連鎖 import されるため、頻繁に呼ばれる router.py のホットパスへ持ち込みたくない。
そのためどちらの pillar にも属さない ``backend/free/`` 直下 (``constants.py`` と
同格) に置く。

2つの tier に分ける:

- ``DOCUMENT_NOUNS_NEEDS_SUFFIX``: 動作動詞 (書/作成/生成 等) との共起を要求する
  名詞。router.py は常にこの前提で判定するため、下記 STANDALONE も含め全語彙に
  サフィックスを要求する (安全側)。
- ``DOCUMENT_NOUNS_STANDALONE``: content_detector.py が名詞単体でも TEXT と
  確定する語彙 (2026-07-15 に「受付フロー手順書」等が CODE 判定されファイルへ
  Python コードが書き込まれた退行への対策で追加された社内文書語彙を含む)。

新規に語彙拡張する場合は router.py 冒頭のコメントに従い
``docs/f_03_agent_engine.md`` §1.2 のテーブルも同時更新すること。

英語版 (``_EN`` サフィックス) は GUI 左下の言語設定 (``get_locale()``) が
``"en"`` の場合にのみ使われる (``backend/free/core/locale_patterns.py`` 経由)。
日本語版と英語版は locale で完全に排他利用されるため、必ずしも1対1の逐語訳
である必要はなく、各言語で自然な語彙を選ぶ。
"""

from __future__ import annotations

# 動作動詞 (書/作成/生成/まとめ/出力 等) との共起が必要な文書名詞。
# 「計画書」「README」等、content_detector.py 側には過去実績が無い語彙は
# 誤爆を避けるため標準単体マッチ (STANDALONE) へ安易に昇格させず、ここに留める。
DOCUMENT_NOUNS_NEEDS_SUFFIX: tuple[str, ...] = (
    "記事", "レポート", "文書", "論文", "マニュアル", "ドキュメント",
    "ブログ", "説明文", "論説", "手引き", "要件定義", "計画書",
    "README", "読本",
)

# 名詞単体でも文書生成依頼として確定してよい語彙
# (content_detector.py が 2026-07-15 のインシデント対応で確立した社内文書語彙を含む)。
DOCUMENT_NOUNS_STANDALONE: tuple[str, ...] = (
    "仕様書", "要件定義書?", "設計書", "設計仕様書?", "基本設計書",
    "詳細設計書", "要求仕様書", "デザインドキュメント",
    "手順書", "議事録", "テンプレート", "ひな形", "雛形",
    "案内文?", "通知文?", "お知らせ", "報告書", "提案書", "企画書",
    "送付状", "依頼文", "挨拶文", "文面", "チェックリスト", "アジェンダ",
    "ガイドライン", "規約", "規定", "スクリプト台本", "台本",
)

def _plain_variants(pattern: str) -> set[str]:
    """正規表現片から素の語彙候補を起こす (``案内文?`` → ``{案内文, 案内}``)。

    JA の名詞タプルは ``?`` による任意末尾しか正規表現要素を持たないため、
    それだけを展開する。ASCII を含む語は学習許可判定で一律除外されるので対象外。
    """
    if "?" not in pattern:
        return {pattern}
    out = {pattern.replace("?", "")}
    idx = pattern.find("?")
    while idx != -1:
        # ``?`` の直前 1 文字を落とした形も許容語として登録する。
        out.add((pattern[: idx - 1] + pattern[idx + 1 :]).replace("?", ""))
        idx = pattern.find("?", idx + 1)
    return {v for v in out if len(v) >= 2}


#: ``category="long_form"`` として **学習を許可する** 文書種別名詞の許容リスト。
#: 2026-07-25 に除外リスト方式から反転した。除外リスト (``_LONG_FORM_NONLEARNABLE_EXACT``)
#: は「新種の一般語が来るたびに追記する」運用になり、2026-07-15 に一度対処した後も
#: 2026-07-25 に「技術 / 方針 / 会話 / 検索 / 議論 / 有益 / 簡潔 / 字以内 / 捏造 /
#: 内線番号」等 52 語が学習され、謝辞 1 行が 6,436 字の長文生成に化けた。
#: 許容リストなら未知の一般語は既定で学習されない。
DOCUMENT_NOUN_LEARNABLE_JA: frozenset[str] = frozenset(
    v
    for noun in (*DOCUMENT_NOUNS_NEEDS_SUFFIX, *DOCUMENT_NOUNS_STANDALONE)
    for v in _plain_variants(noun)
    if not noun.isascii()
)


# DOCUMENT_NOUNS_NEEDS_SUFFIX の英語版。
DOCUMENT_NOUNS_NEEDS_SUFFIX_EN: tuple[str, ...] = (
    "article", "report", "document", "paper", "manual", "documentation",
    "blog(?:\\s*post)?", "essay", "editorial", "guide", "plan",
    "README", "handbook",
)

# DOCUMENT_NOUNS_STANDALONE の英語版。
DOCUMENT_NOUNS_STANDALONE_EN: tuple[str, ...] = (
    "specifications?", "requirements?\\s*doc(?:ument)?s?",
    "design\\s*doc(?:ument)?s?", "design\\s*specifications?",
    "detailed\\s*design\\s*doc(?:ument)?s?", "SOPs?",
    "standard\\s*operating\\s*procedures?", "procedure\\s*manuals?",
    "meeting\\s*minutes", "templates?", "checklists?", "agendas?",
    "guidelines?", "cover\\s*letters?", "proposals?", "memos?",
    "notices?", "announcements?", "runbooks?", "playbooks?",
    "style\\s*guides?",
)
