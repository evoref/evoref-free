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
