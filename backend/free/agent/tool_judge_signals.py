"""クエリ種別のシグナル検出 (正規表現 + 純粋述語)

「このクエリはツールを要するか / どういう種類の依頼か」だけを判定する層。
ツール名も引数も決めない。判定本体 (``tool_call_judge``) と学習側
(``learned_patterns``) の双方から参照される。
"""

from __future__ import annotations

import re

from backend.free.agent.router import (
    asks_directory_listing,
)
from backend.free.core.intent_vocab import (
    RUNTIME_INFO_QUERY_RE,
    SESSION_ANCHOR_EN,
    SESSION_PROXIMITY_WINDOW_EN,
    SESSION_TOPIC_BREAK_LOOKAHEAD_EN,
    looks_like_numeric_question,
    session_self_reference_pattern_ja,
)
from backend.free.core.locale_patterns import select_locale_variant

#: 「直下だけ」を指す表現。再帰的な列挙を明示する語 (再帰 / 全部 / 配下すべて)
#: が同居する依頼は対象外にして、意図が割れる文には手を入れない。
_IMMEDIATE_CHILDREN_RE = re.compile(
    r"直下|直下の|トップレベル|第一階層|一階層目"
    # 「ルートディレクトリにあるファイルとフォルダ」も直下要求だが、語彙が
    # 「直下」に限られていたため素通りし、既定 3 階層のツリー (5,523 字) が
    # TOOL_RESULT_MAX_CHARS で切り詰められた (実インシデント 2026-08-04
    # ライブ監査: 実在する frontend/ を「見当たりません」と誤答)。
    r"|ルート(?:ディレクトリ|フォルダ)?直下|ルート(?:ディレクトリ|フォルダ)にある"
    r"|最上位|一番上"
    r"|immediate\s+children|top[-\s]?level|first\s+level|root\s+(?:directory|folder)",
    re.IGNORECASE,
)
_RECURSIVE_LISTING_RE = re.compile(
    r"再帰|階層すべて|配下すべて|すべての階層|全階層|recursive(?:ly)?",
    re.IGNORECASE,
)
# Windows / Unix の明示パス、または URL。ユーザーが対象を書いた決定論的シグナルで、
# aux の否定票より優先してよい (``_upgrade_command_via_aux`` の降格例外)。
#: ドライブレターの区切りは ``\`` と ``/`` の双方を受ける。バックスラッシュ限定
#: だったため ``E:/tmp/a.txt`` がツールシグナルとして検出されず、明示パス付きの
#: 依頼が knowledge query に落ちて「存在しない」と誤答していた (実インシデント
#: 2026-08-04 ライブ監査)。
_PATH_OR_URL_SIGNAL_RE = re.compile(
    r"[A-Za-z]:[\\/]|(?:^|[\s　])(?:/[\w._-]+){2,}|https?://",
)
# コード/ファイル検索の共起ガード。汎用「検索」単独は知識質問にもマッチする
# ため、コード/ファイル文脈語との共起を要求する (下記 _TOOL_PATTERNS の設計
# 原則と同じ)。_TOOL_PATTERNS / _TOOL_PATTERNS_EN のこのエントリと
# ToolCallJudge._infer_tool 内の search_code 判定の単一の情報源。JA/EN で
# 同一の複合パターンを使う (両トークンを同一正規表現に含めているため locale
# で分岐不要)。
_CODE_SEARCH_PATTERNS = (
    re.compile(
        r"(?:コード|ファイル|ソース|関数|クラス|(?<![A-Za-z])code(?![A-Za-z])"
        r"|(?<![A-Za-z])file(?![A-Za-z])|(?<![A-Za-z])source(?![A-Za-z]))"
        r".*(?:検索|(?<![A-Za-z])search(?![A-Za-z])|(?<![A-Za-z])grep(?![A-Za-z])"
        r"|(?<![A-Za-z])find(?![A-Za-z]))",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:検索|(?<![A-Za-z])search(?![A-Za-z])|(?<![A-Za-z])grep(?![A-Za-z])"
        r"|(?<![A-Za-z])find(?![A-Za-z]))"
        r".*(?:コード|ファイル|ソース|(?<![A-Za-z])code(?![A-Za-z])"
        r"|(?<![A-Za-z])file(?![A-Za-z])|(?<![A-Za-z])source(?![A-Za-z]))",
        re.IGNORECASE,
    ),
)

# 所在を問う言い回し (「<識別子> はどこで使われていますか」) の共起ガード。
#
# ``_CODE_SEARCH_PATTERNS`` は「コード/ファイル語 × 検索動詞」を要求するが、
# この言い方は **どちらの語も含まない**。結果、ルール層を素通りして文法制約
# 分類器へ落ち、分類器は所在探索に無意味な ``list_directory`` を選ぶ。
#
# 2026-08-16 ライブ監査ターン 19「このプロジェクトで LangChain はどこで
# 使われていますか？」: ``list_directory`` が 5,477 文字のツリーを返し、それを
# 再 prefill した結果 **218.6 秒** (当セッション最長) を消費したうえ、
# 「一覧は途中が省略されているため確認できません」で終わった。同じ質問は
# ``search_code`` なら 1 回の grep で答えが出る。
_CODE_USAGE_LOCATION_RE = re.compile(
    r"(?:どこ|どの(?:ファイル|モジュール|クラス|関数|パッケージ)"
    r"|\bwhere\b|which\s+(?:file|module|class|function))",
    re.IGNORECASE,
)
#: 所在を問う対象の動詞。可能形 (「使えますか」) は「利用可否」の質問であって
#: 所在の質問ではないので採らない (受身/サ変の語幹だけを見る)。
_CODE_USAGE_VERB_RE = re.compile(
    r"(?:使わ|使用さ|利用さ|定義さ|実装さ|呼ば|宣言さ|参照さ|書かれ"
    r"|\bused\b|\bdefined\b|\bimplemented\b|\bdeclared\b|\breferenced\b"
    r"|\bcalled\b)",
    re.IGNORECASE,
)
#: 検索対象になりうる ASCII 識別子。これが無いクエリ (「敬語はどこで使われますか」
#: のような自然言語の質問) では発火させない — search_code はコード検索であり、
#: 識別子が取れないなら撃つ意味がない。
_CODE_IDENTIFIER_RE = re.compile(r"[A-Za-z_]\w{2,}")


#: 所在質問の骨組みを成す英語の機能語。``_extract_search_pattern`` はこれらを
#: 落とさないため、"where is search_code used?" で ``where`` を検索語に採って
#: しまう (grep 対象として無意味)。この経路専用に除外する。
_CODE_USAGE_STOPWORDS = frozenset({
    "where", "which", "what", "file", "files", "module", "modules", "class",
    "classes", "function", "functions", "package", "packages", "the", "this",
    "that", "these", "those", "and", "for", "from", "with", "into", "does",
    "did", "are", "was", "were", "been", "being", "used", "uses", "use",
    "defined", "define", "defines", "implemented", "implements", "declared",
    "declares", "referenced", "references", "called", "calls", "project",
    "codebase", "repository", "repo", "code", "source",
})


def _is_code_usage_location_query(query: str) -> bool:
    """「<識別子> はどこで使われているか」型の所在質問か。"""
    return bool(
        _CODE_USAGE_LOCATION_RE.search(query)
        and _CODE_USAGE_VERB_RE.search(query)
        and _CODE_IDENTIFIER_RE.search(query),
    )


def _code_usage_location_pattern(query: str) -> str:
    """所在質問から grep 対象の識別子を取り出す。

    質問の骨組み (where / file / used ...) を除いた最初の ASCII 識別子。
    残らなければ空文字 (呼出側はルール発火を見送る)。
    """
    for token in _CODE_IDENTIFIER_RE.findall(query):
        if token.lower() not in _CODE_USAGE_STOPWORDS:
            return token
    return ""
# ルールベースフォールバック用パターン
# 注意: 「検索」等の汎用語は知識質問にもマッチするため、
# コード/ファイル文脈を要求するパターンのみ含める。
_TOOL_PATTERNS = [
    re.compile(r"(?:ファイル|file).*(?:読|書|開|作成|削除)", re.IGNORECASE),
    re.compile(r"(?:コマンド|command).*(?:実行|run)", re.IGNORECASE),
    # コード/ファイル検索: 汎用「検索」は知識質問にマッチするため除外。
    # ASCII トークンは単語境界必須 ("crossencoder" の 'code' 等への部分一致誤爆対策、
    # CPU/RAM 境界ガードと同じ理由)。日本語側 (コード/ファイル/ソース/検索) は対象外。
    *_CODE_SEARCH_PATTERNS,
    re.compile(r"(?:URL|url|https?://|ウェブ|web|サイト|site|ページ|page|ニュース|news|フェッチ|fetch|ブラウズ|browse)", re.IGNORECASE),
    re.compile(r"(?:計算|calculate)\s", re.IGNORECASE),
    # ファイルパスを含むクエリ（C:\, E:\, /home/ 等）+ 出力/保存/生成系動詞
    re.compile(r"[A-Za-z]:\\", re.IGNORECASE),
    re.compile(r"(?:出力|保存|生成|作成|書き出|エクスポート).*(?:して|する)", re.IGNORECASE),
    re.compile(r"(?:プログラム|コード|スクリプト|関数|クラス).*(?:作|書|生成)", re.IGNORECASE),
    # 「実行して」「動かして」等の動詞（ファイルパスやバッククォート付き）
    re.compile(r"(?:実行|動かし|起動|run|exec).*(?:して|する|しろ)", re.IGNORECASE),
    # バッククォート内コマンド
    re.compile(r"`[^`]+`", re.IGNORECASE),
    # --- Python 実行可能クエリ: システム情報 ---
    # 注意: \b は日本語文字を \w とみなすため英語-日本語境界で機能しない。
    # 英語の短いキーワードは (?<![A-Za-z])...(?![A-Za-z]) で ASCII 境界を使用。
    # CPU/RAM/GPU/VRAM も境界必須 (IGNORECASE で "program" の 'ram' 等に
    # 部分マッチし、文書タスクへ OS スペックコマンドを誤発火した実績あり)。
    # 「容量」単独は外す。``capacity`` を外したのと同じ理由で、**データ量の話に
    # 普通に現れる**ため機械スペックの要求とは限らない (実インシデント
    # 2026-08-10 ライブ監査: 「DBの容量は1.2TB…合計容量は何TBですか」
    # 「フルと増分を合わせた総容量」「RPO・RTO・容量効率の列で表を」の 4 ターンで
    # OS/CPU/コア数の取得コマンドが撃たれた)。機器を名指しする質問は
    # ディスク / ストレージ / ドライブ / disk / drive 側で拾えるので取りこぼさない。
    re.compile(r"(?:スペック|(?<![A-Za-z])CPU(?![A-Za-z])|メモリ|(?<![A-Za-z])RAM(?![A-Za-z])|(?<![A-Za-z])GPU(?![A-Za-z])|(?<![A-Za-z])VRAM(?![A-Za-z])|ディスク|(?:空き|残り|使用)容量|ストレージ|ドライブ|(?<![A-Za-z])spec(?![A-Za-z])|(?<![A-Za-z])drive(?![A-Za-z]))", re.IGNORECASE),
    # 「何月|何日|何曜日」追加 (router.py:101 と同期)
    # 「日時型」「日付フォーマット」等はデータ型/スキーマの話 (router の
    # _EXECUTABLE_QUERY_PATTERNS の同じエントリのコメント参照)。
    re.compile(r"(?:何時|何月|何日|何曜日|(?:日時|日付)(?!型|形式|フォーマット|カラム|列)|現在時刻|(?<![A-Za-z])today(?![A-Za-z])|(?<![A-Za-z])now(?![A-Za-z])|(?<![A-Za-z])date(?![A-Za-z])|(?<![A-Za-z])time(?![A-Za-z]))", re.IGNORECASE),
    re.compile(r"(?:IP\s*アドレス|ホスト名|(?<![A-Za-z])hostname(?![A-Za-z])|(?<![A-Za-z])ip\s*address)", re.IGNORECASE),
    re.compile(r"(?:(?<![A-Za-z])OS(?![A-Za-z])|オペレーティングシステム|(?<![A-Za-z])Windows(?![A-Za-z])|(?<![A-Za-z])Linux(?![A-Za-z])|(?<![A-Za-z])Mac(?![A-Za-z]))", re.IGNORECASE),
    re.compile(r"(?:Python|python)\s*(?:バージョン|version)", re.IGNORECASE),
    re.compile(r"(?:環境変数|(?<![A-Za-z])env(?![A-Za-z])|(?<![A-Za-z])PATH(?![A-Za-z]))", re.IGNORECASE),
    # --- Python 実行可能クエリ: 数値処理 ---
    # 「16進に変換して」のように「数」が入らない書き方も基数変換。bare「変換」を
    # 外したぶん、ここで基数そのものを見る。
    re.compile(r"(?:階乗|素数|フィボナッチ|素因数|進数変換|\d+\s*進(?:数|法)?|桁)", re.IGNORECASE),
    # --- Python 実行可能クエリ: データ処理 ---
    re.compile(r"(?:集計|合計|平均|中央値|標準偏差|ソート|統計)", re.IGNORECASE),
    # --- Python 実行可能クエリ: 変換 ---
    # 「変換」単独は外す。表・JSON・Markdown の書き換えなど **LLM 自身がやる
    # 内容変換** まで実行可能クエリ扱いになり、そこから層 0.5 (コマンド想起) が
    # 開いて無関係な過去コマンドが再生される (実インシデント 2026-08-10 ライブ
    # 監査:「その表を JSON Schema (draft 2020-12) に変換してください。」で
    # OS/CPU スペック取得コマンドが実行された)。コマンドが要る変換は符号化・
    # 基数・ハッシュ・時刻に限られ、いずれも固有語で拾える。
    re.compile(r"(?:エンコード|デコード|Base64|ハッシュ|タイムスタンプ|文字コード|エポック秒?|UNIX\s*時間)", re.IGNORECASE),
]

# _TOOL_PATTERNS の英語版。GUI 左下の言語設定が 'en' の場合のみ使う
# (_TOOL_PATTERNS とは locale で完全に排他利用される)。既に ASCII/日英混在で
# 機能するエントリ (コード検索/URL/計算/日時/OS/env 等) は locale='en' でも
# 引き続き評価できるようそのまま複製する。
_TOOL_PATTERNS_EN = [
    re.compile(r"\bfile\b.*\b(?:read|open)\b.*\b(?:write|modify|change|update|delete|remove|edit)\b", re.IGNORECASE),
    re.compile(r"(?:コマンド|command).*(?:実行|run)", re.IGNORECASE),
    *_CODE_SEARCH_PATTERNS,
    re.compile(r"(?:URL|url|https?://|web|site|page|news|fetch|browse)", re.IGNORECASE),
    re.compile(r"(?:計算|calculate)\s", re.IGNORECASE),
    re.compile(r"[A-Za-z]:\\", re.IGNORECASE),
    re.compile(r"\b(?:save|export|output)\b.{0,20}\b(?:it|this|that|to|as|file)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:program|code|script|function|class)\b.*"
        r"\b(?:write|create|generate|build|implement)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:run|execute|exec)\b.{0,20}\b(?:this|that|it|the\s+\w+)\b", re.IGNORECASE),
    re.compile(r"`[^`]+`", re.IGNORECASE),
    re.compile(r"(?:スペック|(?<![A-Za-z])CPU(?![A-Za-z])|memory|(?<![A-Za-z])RAM(?![A-Za-z])|(?<![A-Za-z])GPU(?![A-Za-z])|(?<![A-Za-z])VRAM(?![A-Za-z])|disk|capacity|storage|drive|(?<![A-Za-z])specs?(?![A-Za-z]))", re.IGNORECASE),
    # 「日時型」「日付フォーマット」等はデータ型/スキーマの話 (router の
    # _EXECUTABLE_QUERY_PATTERNS の同じエントリのコメント参照)。
    re.compile(r"(?:何時|何月|何日|何曜日|(?:日時|日付)(?!型|形式|フォーマット|カラム|列)|現在時刻|(?<![A-Za-z])today(?![A-Za-z])|(?<![A-Za-z])now(?![A-Za-z])|(?<![A-Za-z])date(?![A-Za-z])|(?<![A-Za-z])time(?![A-Za-z]))", re.IGNORECASE),
    re.compile(r"(?:IP\s*address|hostname|(?<![A-Za-z])hostname(?![A-Za-z])|(?<![A-Za-z])ip\s*address)", re.IGNORECASE),
    re.compile(r"(?:(?<![A-Za-z])OS(?![A-Za-z])|operating\s*system|(?<![A-Za-z])Windows(?![A-Za-z])|(?<![A-Za-z])Linux(?![A-Za-z])|(?<![A-Za-z])Mac(?![A-Za-z]))", re.IGNORECASE),
    re.compile(r"(?:Python|python)\s*version", re.IGNORECASE),
    re.compile(r"(?:environment\s*variable|(?<![A-Za-z])env(?![A-Za-z])|(?<![A-Za-z])PATH(?![A-Za-z]))", re.IGNORECASE),
    # --- Python 実行可能クエリ: 数値処理 (171行目相当の英語版) ---
    # base N / hex / binary / octal は基数変換そのもので、bare "convert" を
    # 外した後もここで拾う (bare "convert" を外した理由は変換パターン側の
    # コメント参照)。
    re.compile(r"\b(?:factorial|prime(?:\s*numbers?)?|fibonacci|prime\s*factorization|base\s*conversion|base\s*\d+|hex(?:adecimal)?|binary|octal|radix|number\s*of\s*digits?|digits?)\b", re.IGNORECASE),
    # --- Python 実行可能クエリ: データ処理 (173行目相当の英語版) ---
    # sum/average/mean/sort は日常会話で極めて頻出する多義語 ("What do you
    # mean?"/"I sort of agree"/"on average, this works fine") のため、
    # 単独では発火させず数値/データ文脈語との近接共起を要求する (2026-07-22
    # 監査で判明、router.py の _EXECUTABLE_QUERY_PATTERNS_EN と同期)。
    # total/median/standard deviation/std dev/statistics/aggregate は
    # 既存テスト (test_data_processing_queries_en の bare "What's the
    # total?") が単独発火を前提としており、日常会話での多義性も相対的に
    # 低いため単独発火のまま維持する。
    re.compile(r"\b(?:total|median|standard\s*deviation|std\s*dev|statistics|aggregate)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:sum|average|mean|sort(?:ed|ing)?)\b.{0,20}"
        r"\b(?:numbers?|data|list|array|values?|dataset|figures?)\b"
        r"|\b(?:numbers?|data|list|array|values?|dataset|figures?)\b.{0,20}"
        r"\b(?:sum|average|mean|sort(?:ed|ing)?)\b",
        re.IGNORECASE,
    ),
    # --- Python 実行可能クエリ: 変換 (175行目相当の英語版) ---
    # bare "convert" は日本語版の「変換」と同じ理由で外す (内容変換の依頼が
    # 実行可能クエリになり、層 0.5 のコマンド想起が開く)。
    re.compile(r"\b(?:encode|decode|encoding|decoding|Base64|hash(?:ing)?|timestamp|epoch)\b", re.IGNORECASE),
]

# _infer_tool() の実行可能クエリ判定ゲート (システム情報・数値処理・
# データ処理・変換)。_TOOL_PATTERNS/_TOOL_PATTERNS_EN の該当エントリと
# 語彙が重複するが、_infer_tool は「どのツールを選ぶか」を決める別の判定
# フェーズであり、単一の結合正規表現として設計されているため独立して保持する。
# CPU/RAM/GPU/VRAM 等の短い ASCII トークンは境界必須 (IGNORECASE で
# "program" の 'ram' 等に部分マッチした実績が _TOOL_PATTERNS 側にあり
# 159-162行目で対策済み。本パターンは同期しておらず 2026-07-22 監査で
# 判明するまで同じ穴が残っていた)。
_INFER_TOOL_EXEC_QUERY_RE = re.compile(
    r"(?:スペック|(?<![A-Za-z])CPU(?![A-Za-z])|メモリ|(?<![A-Za-z])RAM(?![A-Za-z])"
    # 「容量」単独を外す理由は _TOOL_PATTERNS の同じエントリのコメント参照。
    r"|(?<![A-Za-z])GPU(?![A-Za-z])|(?<![A-Za-z])VRAM(?![A-Za-z])"
    r"|ディスク|(?:空き|残り|使用)容量|ストレージ|ドライブ"
    r"|(?<![A-Za-z])spec(?![A-Za-z])"
    r"|何時|何月|何日|何曜日|日時|日付|現在時刻"
    # 裸の now は時刻クエリのシグナルにならない ("from now on" / "right now" /
    # "know now" 等の非時制表現に誤爆する。2026-07-26 ライブ検証で
    # "From now on, always reply in English. Tell me what a deadlock is" が
    # run_command_readonly の時刻取得を発火させた)。EN 版 (_..._RE_EN) は
    # 2026-07-22 監査で既に句単位に絞ってあるので、同じ形へ揃える。
    r"|(?<![A-Za-z])today(?![A-Za-z])"
    r"|what'?s?\s*(?:the\s*)?(?:time|date)|current\s*(?:time|date)"
    r"|IP\s*アドレス|ホスト名|(?<![A-Za-z])hostname(?![A-Za-z])"
    r"|(?<![A-Za-z])OS(?![A-Za-z])|オペレーティングシステム"
    r"|(?<![A-Za-z])Windows(?![A-Za-z])|(?<![A-Za-z])Linux(?![A-Za-z])"
    r"|(?<![A-Za-z])Mac(?![A-Za-z])"
    r"|Python\s*(?:バージョン|version)"
    r"|環境変数|(?<![A-Za-z])env(?![A-Za-z])"
    r"|階乗|素数|フィボナッチ|素因数|進数変換|桁"
    r"|集計|合計|平均|中央値|標準偏差|ソート|統計"
    r"|変換|エンコード|デコード|Base64|ハッシュ|タイムスタンプ)",
    re.IGNORECASE,
)

# _INFER_TOOL_EXEC_QUERY_RE の英語版。JA 側と同じ理由で ASCII トークンは
# 全て境界必須 (境界無しだと program/framework/diagram/anagram が
# memory/RAM 等に、summary/resume/assume が sum/mean 等に部分マッチする。
# 2026-07-22 監査で判明)。
_INFER_TOOL_EXEC_QUERY_RE_EN = re.compile(
    r"(?:(?<![A-Za-z])specs?(?![A-Za-z])|(?<![A-Za-z])CPU(?![A-Za-z])"
    r"|(?<![A-Za-z])memory(?![A-Za-z])|(?<![A-Za-z])RAM(?![A-Za-z])"
    r"|(?<![A-Za-z])GPU(?![A-Za-z])|(?<![A-Za-z])VRAM(?![A-Za-z])"
    # ``capacity`` はデータ項目名として普通に現れるため対象外
    # (_EXECUTABLE_QUERY_COMMANDS の同名エントリのコメント参照)。
    r"|(?<![A-Za-z])disk(?![A-Za-z])|(?<![A-Za-z])storage(?![A-Za-z])"
    r"|what'?s?\s*(?:the\s*)?(?:time|date)|current\s*(?:time|date)"
    r"|today'?s?\s*date|what\s*day\s*(?:is\s*it|of\s*the\s*week)"
    r"|(?<![A-Za-z])date(?![A-Za-z])|(?<![A-Za-z])time(?![A-Za-z])"
    r"|IP\s*address|(?<![A-Za-z])hostname(?![A-Za-z])"
    r"|(?<![A-Za-z])OS(?![A-Za-z])|operating\s*system"
    r"|(?<![A-Za-z])Windows(?![A-Za-z])|(?<![A-Za-z])Linux(?![A-Za-z])|(?<![A-Za-z])Mac(?![A-Za-z])"
    r"|Python\s*version"
    r"|environment\s*variable|(?<![A-Za-z])env(?![A-Za-z])|(?<![A-Za-z])PATH(?![A-Za-z])"
    r"|(?<![A-Za-z])factorial(?![A-Za-z])|(?<![A-Za-z])prime(?![A-Za-z])(?:\s*numbers?)?"
    r"|(?<![A-Za-z])fibonacci(?![A-Za-z])|prime\s*factorization|base\s*conversion"
    r"|number\s*of\s*digits?|(?<![A-Za-z])digits?(?![A-Za-z])"
    r"|(?<![A-Za-z])sum(?![A-Za-z])|(?<![A-Za-z])total(?![A-Za-z])|(?<![A-Za-z])average(?![A-Za-z])"
    r"|(?<![A-Za-z])mean(?![A-Za-z])|(?<![A-Za-z])median(?![A-Za-z])"
    r"|standard\s*deviation|std\s*dev|(?<![A-Za-z])sort(?:ed|ing)?(?![A-Za-z])"
    r"|(?<![A-Za-z])statistics(?![A-Za-z])|(?<![A-Za-z])aggregate(?![A-Za-z])"
    r"|(?<![A-Za-z])convert(?![A-Za-z])|(?<![A-Za-z])encod(?:e|ing)(?![A-Za-z])"
    r"|(?<![A-Za-z])decod(?:e|ing)(?![A-Za-z])|(?<![A-Za-z])Base64(?![A-Za-z])"
    r"|(?<![A-Za-z])hash(?:ing)?(?![A-Za-z])|(?<![A-Za-z])timestamp(?![A-Za-z]))",
    re.IGNORECASE,
)
#: 搭載メモリ量を尋ねるクエリ。専用ツール ``system_hardware_info`` へ振る。
#:
#: 2026-07-27 に ``メモリ`` / ``RAM`` を spec コマンド (``_build_spec_command``)
#: から外した経緯があるが、それは「コマンドが搭載メモリ量を一切出力しないのに
#: パターンだけ一致して 1 ターンを浪費する」ためだった。allow-list
#: (``_READONLY_SAFE_MODULES``) はチャットから渡される **コマンド文字列** にしか
#: 掛からず、backend 自身の Python は制約外なので、シェルを経由しない
#: ``free/core/system_info`` で測れる (``free/core/vram_monitor`` が nvidia-smi を
#: 直接叩いているのと同じ立て付け)。allow-list は広げていない。
#:
#: ``メモリ`` 単独は EvorefMem の話 (「メモリに保存して」「記憶メモリ」) と
#: 衝突するため、容量を問う語との共起を要求する。
#: 英語は量を問う語が名詞の **前** に来る ("how much memory") ため別枝で受ける。
#:
#: 「空き」「使用率」などの **状態を問う語** も同じツールで答えられる
#: (``format_hardware_facts`` は available RAM と CPU usage を出す)。これらが
#: 抜けていたため、状態クエリは spec コマンド (搭載 RAM も CPU 使用率も
#: 出力しない) へ落ちていた。実インシデント (2026-08-14 ライブ監査 ターン6):
#: 「この PC の空きメモリと CPU 使用率を教えてください。」に対し
#: ``run_command_readonly`` が OS/CPU/Cores/Disk だけを返し、
#: 「取得した情報に含まれていないためお答えできません」と回答した。
_HARDWARE_MEMORY_QUERY_RE = re.compile(
    r"(?:(?:搭載|物理|実装|空き|使用|利用可能|残り|フリー)?メモリ"
    r"|(?<![A-Za-z])RAM(?![A-Za-z])"
    r"|(?<![A-Za-z])memory(?![A-Za-z]))"
    r"[^。．\n]{0,12}"
    r"(?:容量|サイズ|何\s*(?:GB|ギガ|MB)|いくつ|どれ(?:く|ぐ)らい|積んで|載って"
    r"|使用[率量]|空き|残[りって]|使って"
    r"|(?<![A-Za-z])(?:size|usage|used|free|available)(?![A-Za-z]))"
    r"|(?:空き|残り|利用可能な|フリー)(?:メモリ|RAM)"
    r"|CPU\s*(?:の)?\s*(?:使用[率量]|利用[率量]|負荷)"
    r"|(?<![A-Za-z])cpu\s+(?:usage|load|utilization)(?![A-Za-z])"
    r"|(?<![A-Za-z])(?:how\s+much|total|physical|installed|available|free)\s+"
    r"(?:ram|memory)(?![A-Za-z])"
    # GPU / VRAM も同じツールで答える (``format_hardware_facts`` が GPU 行を
    # 出す。測れない環境では「測れない」と明示する)。ここが抜けていたため
    # 「GPU の VRAM 使用状況は？」は単独ではどの層にも掛からず、base の
    # 想像か「分からない」に落ちていた (2026-08-19 ライブ監査 ターン8)。
    r"|(?<![A-Za-z])VRAM(?![A-Za-z])"
    r"|(?:GPU|グラボ|グラフィック(?:ス)?\s*(?:カード|ボード)?)"
    r"[^。．\n]{0,12}"
    r"(?:メモリ|使用[率量]|空き|残[りって]|容量|積んで|載って"
    r"|(?<![A-Za-z])(?:memory|usage|used|free|available)(?![A-Za-z]))",
    re.IGNORECASE,
)
# evoref 自身の実行構成を尋ねるクエリ (`evoref_runtime_info` の発火条件)。
# 定義は core.intent_vocab が SSOT — 同じ判定を agent.router の分類表
# (`self_config_query` ルール) も使うため。後方互換で再輸出する。
_RUNTIME_INFO_QUERY_RE = RUNTIME_INFO_QUERY_RE

# 明示的な実行動詞。バッククォートのコマンドと共起したとき「実測要求」とみなす。
_EXPLICIT_EXEC_VERB_RE = re.compile(
    r"(?:実行|叩いて|走らせ"
    r"|(?<![A-Za-z])run(?![A-Za-z])"
    r"|(?<![A-Za-z])exec(?:ute[sd]?|uting)?(?![A-Za-z]))",
    re.IGNORECASE,
)

# 削除を求める依頼。**どのモードにも削除ツールは存在しない** (ToolsRegistry に
# delete_file / remove_file は登録されておらず、run_command_readonly の allow-list は
# remove / unlink / rmdir / rmtree をすべて拒否する)。撃てるツールが無いまま
# no_tool で base に丸投げすると、実行していない削除の完了を報告する。
#
# 実インシデント (2026-08-12 ライブ監査 ターン29-30): 「さっき作った
# audit_test.txt を削除してください。」に対しツールを 1 つも実行しないまま
# 「E:\tmp\audit_test.txt を削除しました。」と回答し、続く「本当に削除できました
# か？ファイルが存在するか確認して。」にも確認せず「存在しません。削除は完了して
# います。」と断定した。実ファイルは両ターンとも残っていた。
_DELETE_INTENT_RE = re.compile(
    r"(?:削除|消去|除去|抹消|(?:を|も)?消して|消しといて|捨てて|破棄)"
    r"|(?<![A-Za-z])(?:delete|remove|erase|unlink)(?![A-Za-z])"
    r"|(?<![A-Za-z])rm\s+-?[A-Za-z]*\s*[\w./\\]",
    re.IGNORECASE,
)

#: 削除対象がファイル / ディレクトリであることの手掛かり。会話履歴やメモリの
#: 削除依頼 (「さっきの記憶を消して」) を巻き込まないための AND 条件。
_DELETE_FS_TARGET_RE = re.compile(
    r"(?:ファイル|フォルダ|ディレクトリ|ドライブ)"
    r"|(?<![A-Za-z])(?:file|folder|directory)(?![A-Za-z])"
    r"|[\w-]+\.(?:txt|md|csv|json|yaml|yml|log|py|js|ts|html|xlsx|docx|pptx|pdf)"
    r"|[A-Za-z]:[\\/]",
    re.IGNORECASE,
)

# パス引数を取る「読み取り系」ツール。存在しないパスを渡しても失敗するだけで
# 副作用が無い代わりに、捏造パスを実行する価値もまったく無い。
_READ_PATH_TOOLS: frozenset[str] = frozenset({
    "read_file", "list_directory", "verify_syntax",
})
# 知識質問パターン — ツール判定をスキップすべきクエリ
# 「簡単に説明してください」「〜の使い分けは？」は疑問形の末尾 (教えて/ですか等)
# を伴わない体言止め・依頼形のため、上のパターンにマッチせず層4 (aux
# tool_judgment) まで素通りしていた。実インシデント (2026-07-20):
# 「カーディネーリティという言葉を初めて知りました。簡単に説明してください。」
# で無関係な過去セッションが誤ヒット (score=0.5)、「インターフェースと抽象
# クラスの使い分けは？」で search_history が "No results found" を返す —
# どちらも履歴参照の意図が皆無な純粋な知識質問で、小型 aux モデルが
# tool_needed=True と誤判定した。
_KNOWLEDGE_PATTERNS = [
    re.compile(r"(?:教えて|おしえて|とは|って何|ですか|でしょうか|ありますか)", re.IGNORECASE),
    re.compile(r"(?:について|に関して|に関する)", re.IGNORECASE),
    re.compile(r"(?:知りたい|確認したい|調べたい)", re.IGNORECASE),
    re.compile(r"(?:説明して|使い分け)", re.IGNORECASE),
    re.compile(r"(?:what is|tell me|explain|describe)\b", re.IGNORECASE),
]

# _KNOWLEDGE_PATTERNS の英語版。「about」は汎用前置詞のため直訳せず、
# 限定的な表現で構成する (誤爆抑制)。
_KNOWLEDGE_PATTERNS_EN = [
    re.compile(r"\bwhat(?:'s|\s+is|\s+are)\b", re.IGNORECASE),
    re.compile(r"\b(?:tell\s+me|explain|describe|walk\s+me\s+through)\b", re.IGNORECASE),
    re.compile(r"\b(?:i\s+want\s+to\s+know|curious\s+about|wondering\s+about)\b", re.IGNORECASE),
    re.compile(r"\bdifference\s+between\b|\bwhen\s+(?:should|do)\s+i\s+use\b", re.IGNORECASE),
]

# アシスタント自身の意見・嗜好・感想を尋ねる質問パターン — 「好きな〜は
# ありますか」「好きな〜とかあったりしますか」等。search_history は
# ユーザー自身の過去発言を検索するツールであり、アシスタント自身の意見を
# 尋ねる質問には無関係だが、明確なツールシグナルが無いため層4 (aux
# tool_judgment) まで素通りし、小型 aux モデルが誤って search_history を
# 選ぶ実インシデントが多発した (2026-07-17/18 の2日分ログで
# tool_call_decision=aux の 48% が該当)。「あったりしますか」等の口語的な
# 疑問文末尾も拾う (上の _KNOWLEDGE_PATTERNS の「ありますか」は完全一致部分
# 文字列のみを拾うため「あったりします(か)」を取りこぼす)。
# あえて _KNOWLEDGE_PATTERNS 側に「あったり」単体を追加しない: _KNOWLEDGE_PATTERNS
# は _judge_with_rules (層1) でも無条件に (一人称マーカーを考慮せず) 参照される
# ため、汎用パターンとして追加すると「私は好きな〜あったりしますか」のような
# 一人称クエリまで層1で誤って即 no_tool になり、下記の一人称除外が層4に届く
# 前に握り潰されてしまう (レビューで判明)。
# ただし「私の/私は/僕の/僕は/自分の/自分は」等の一人称マーカーを含む場合は
# ユーザー自身の過去発言 (例:「私の好きなプログラミング言語は？」) を指す
# 可能性があるため除外せず、層4 (aux) 判定に委ねる (実際に "Rust" 等の
# 固有名詞を正しく抽出できた実績がある)。
_ASSISTANT_PREFERENCE_PATTERNS = [
    re.compile(
        r"(?:好きな|嫌いな|得意な|苦手な|おすすめの).{0,20}?"
        r"(?:ますか|ありますか|でしょうか|ですか"
        # 「あったり」は文中でも出現しやすい語のため (例:「あったりして
        # 困った」)、疑問文末尾の用法に限定して文末アンカーを課す。
        r"|あったり(?:します)?(?:か)?[？?]?\s*$)",
    ),
]
_FIRST_PERSON_REFERENCE_RE = re.compile(
    r"私の|私が|私は|僕の|僕が|僕は|俺の|俺が|俺は"
    r"|自分の|自分が|自分は|わたしの|わたしが|わたしは",
)


# セッション自己参照パターン — 「この会話で」等、現在のセッション自体を参照
# する発話。会話履歴は working memory の予算を超えると古いターンからコンテキ
# ストに含まれなくなるため、層4 (tool_judgment) を無条件スキップすると
# 「会話の最初に〜」のような長距離 recall が失敗する (2026-07-19 実インシデ
# ント: 20+ ターンの会話で最初の発話内容を正しく想起できなかった)。
# そのため層4 のスキップ判定には使わず、``ToolCallJudge._maybe_scope_session_
# search`` が search_history 選択後に ``tool_args["session_id"]`` を強制注入し、
# 現在セッションのみへスコープ限定する形で対応する。
# 実インシデント (2026-07-17/18): 「この会話で一番面白かったやり取りは？」が
# スコープ限定前の search_history に振られ、無関係な過去セッション (別日の
# 雑談) の内容を誤って参照・混同した。session_id スコープ限定によりこの事故
# 再発を防ぎつつ、同一セッション内の recall は許可する。
#
# 「この会話」等の言及だけで無条件マッチにすると、「この会話の続きですが、
# 量子もつれについて詳しく教えて」のように自己参照を前置きにしつつ外部知識を
# 要する質問まで誤ってスキップしてしまうため、会話自体を振り返る反省的な語
# (面白い/振り返る/まとめ/感想等) との近接共起を要求する。
# 近接窓は「同一文内 (句点・疑問符・感嘆符を跨がない) の 40 文字以内」。
# 旧実装の任意文字 {0,20} は「この会話で一番最初に私が計算させた問題は
# 何だったか覚えてますか？」(間 21 文字) を 1 文字超過で取りこぼし、
# session_id 非注入 → 全セッション横断検索で前回会話の類似ターンがヒット
# する near-miss を起こした (2026-07-20 ライブ再検証で確認)。一方、窓を
# 任意文字のまま {0,50} へ広げるだけでは「この会話とは別に、相対性理論に
# ついて詳しく教えてください。とても面白いですよね？」のような外部知識
# 質問まで誤ってマッチする (過去レビューで判明) ため、(a) 文境界を跨がない
# 文字クラスで窓を絞り、(b) 「とは別/とは関係」等の明示的な話題切断の
# 前置きを negative lookahead で弾く、の二重ガード付きで窓を 40 に広げる。
# 上記の構造 (アンカー / 否定先読み / 近接窓) は core.intent_vocab から派生させ、
# backend/free/rag/self_rag_judge.py の TRIVIAL_QUESTION_PATTERNS と機械的に
# 同期する (以前は両ファイルへ書き写しており、窓幅を変えるたびに手で両方を直す
# 必要があった)。ただし **語彙を同一にしてはいけない** — self_rag 側はマッチすると
# RAG 検索を丸ごと skip するため誤検出コストが桁違いに高く、話題ポインタにも
# なる語 (質問/指摘/言った/話した/聞いた/順番) は self_rag 側では意図的に採って
# いない。詳細な実測は self_rag_judge.py の「語彙が同一でない理由」コメント参照。
# 反省的な語には時系列順序語 (最初/最後/何番目/何回目) も含める。
# 「この会話で一番最初に計算させた問題は?」(2026-07-21 ライブ検証 ターン18)
# は反省語 (覚えて等) を欠きスコープ注入が漏れ、cross-session 検索が前回
# 会話の類似ターンを引き当て誤答した。順序語の追加でマッチ面が広がる分、
# 「この会話じゃなくて/ではなく」の明示的な話題切断も lookahead へ追加する。
#: 会話自体を振り返る語 (BROAD)。マッチしても search_history を現在セッションへ
#: 限定するだけで誤検出コストが軽微なため、話題ポインタにもなる語 (順番/質問/
#: 指摘/言った/話した/聞いた) まで広く採る。self_rag_judge 側の NARROW との差は
#: 意図的 (core.intent_vocab の該当コメント参照)。
#:
#: 訂正/指摘/直した/思い出/言った/話した/聞いた は 2026-07-25 追加。これらが
#: 無いため「今日の会話の中で、私があなたの回答を訂正した箇所が 3 回ある」が
#: セッション限定されず、前日の別セッションを引用して他人のペルソナ設定を
#: ユーザー本人の訂正として提示した。
#: 順番/順序/列挙/並べ/質問 は 2026-07-27 追加。順序・列挙を尋ねる語が無いため
#: 「この会話で私が質問した順番を…列挙してください。」が自己参照と判定されず、
#: 逆に exclude_session_id が注入され、「この会話」について尋ねているのに現在
#: セッションだけを除外して検索していた。
_SESSION_REFLECTIVE_VOCAB_BROAD_JA = (
    r"面白|印象|振り返|まとめ|要約|感想|どう思|覚えて|何でした|どうでした"
    r"|訂正|指摘|直した|思い出|言った|話した|聞いた"
    r"|最初|最後|何番目|何回目"
    r"|順番|順序|列挙|並べ|質問"
)

_SELF_SESSION_REFERENCE_PATTERNS = [
    re.compile(
        session_self_reference_pattern_ja(_SESSION_REFLECTIVE_VOCAB_BROAD_JA),
    ),
]

# _SELF_SESSION_REFERENCE_PATTERNS の英語版。英語の話題切断表現
# ("aside from"/"apart from") は名詞句の前に来る (日本語の後置とは語順が
# 逆) ため、_SESSION_TOPIC_BREAK_LEAD_RE_EN の前置きガードと併用する
# (_maybe_scope_session_search 側で参照)。
# (pillar境界のため backend/free/rag/self_rag_judge.py の
# TRIVIAL_QUESTION_PATTERNS_EN と同趣旨の定義を重複させている。JA 側と同様に
# 語彙は意図的に完全一致させていない — ``asked`` は self_rag 側では採らない。
# 理由と実測は self_rag_judge.py の「語彙が同一でない理由」コメント参照)。
#: 英語版 BROAD 語彙。order/sequence/enumerate/asked は 2026-07-27 追加
#: (JA 側の 順番/順序/列挙/質問 と対応)。
_SESSION_REFLECTIVE_VOCAB_BROAD_EN = (
    r"interesting|memorable|impressive|funn(?:y|iest)"
    r"|summar\w*|recap\w*|think|thought|feel|felt|remember"
    r"|order|sequence|enumerate|asked"
    r"|first|last|earliest|latest"
)
#: 逆順 (修飾語が先) の共起で使う語彙。動詞 (think/asked 等) は
#: 「〜について考えた this conversation」の語順が不自然なので採らない。
_SESSION_REFLECTIVE_VOCAB_LEADING_EN = (
    r"interesting|memorable|impressive|funn(?:y|iest)"
    r"|summar\w*|recap\w*|first|last|earliest|latest"
)

_SELF_SESSION_REFERENCE_PATTERNS_EN = [
    re.compile(
        SESSION_ANCHOR_EN
        + SESSION_TOPIC_BREAK_LOOKAHEAD_EN
        + SESSION_PROXIMITY_WINDOW_EN
        + f"(?:{_SESSION_REFLECTIVE_VOCAB_BROAD_EN})"
        # 英語は修飾語 (interesting 等) が「this conversation」より前に
        # 来る語順も自然なため (日本語の後置とは逆)、逆順の共起も許容する。
        # 前置きガードは _SESSION_TOPIC_BREAK_LEAD_RE_EN 側で別途行う。
        + f"|(?:{_SESSION_REFLECTIVE_VOCAB_LEADING_EN})"
        + SESSION_PROXIMITY_WINDOW_EN
        + r"(?:this\s+conversation|this\s+chat|our\s+conversation)",
        re.IGNORECASE,
    ),
]
# 話題切断が前置される英語特有の言い回し (「この会話とは別に」の語順違い対策)。
_SESSION_TOPIC_BREAK_LEAD_RE_EN = re.compile(
    r"\b(?:aside|apart|other than|separate)\s+from\s+(?:this|our)\s+conversation\b",
    re.IGNORECASE,
)
# ユーザー自身の行動宣言パターン — アシスタントへの依頼ではない雑談発話。
# 「探してみるね」のような一人称の意思表明をツール起動と誤解しないための除外。
# 依頼形 (「探してみて(ください)」= て止め) とは区別する (こちらは末尾が「て」)。
_SELF_ACTION_PATTERNS = [
    # 「〜てみる(ね/よ/わ/かな/から)」自分で試す宣言 (文末)
    re.compile(r"(?:て|で)みる(?:ね|よ|わ|な|かな|から)?[\s　!！。.…]*$"),
    # 「〜しておく/やっておく/調べておく(ね/よ)」自己完結の行動宣言 (文末)
    re.compile(r"(?:してお|やってお|探してお|調べてお|見てお|やっと)く(?:ね|よ|わ|から)?[\s　!！。.…]*$"),
    # 一人称主語で自分が行う宣言 (文末の意思・断定形に限定)。
    # 旧実装は主語マーカー単独の無アンカー部分一致で、「この会話で一番最初に
    # 私が計算させた問題は何だったか覚えてますか？」のような関係節中の一人称
    # 主語まで自己行動宣言と誤検出し skip_judgment=True を招いていた
    # (2026-07-20 ライブ検証で確認、層5.5 フォールバックが実害を吸収)。
    # 主語マーカーの後、同一文内 (文境界・疑問符を跨がない) で動詞終止形
    # (u 段かな) または ます形の文末で終わる発話のみ宣言とみなす。
    # 丁寧断定「です」は情報提供であって行動宣言ではないため lookbehind で
    # 除外し (「ます」の「す」は前が「ま」なので通る)、疑問形終端
    # (「〜ますか？」等) は末尾許容クラスに「？」を含めないことで弾く。
    # 取りこぼし (「私がやるから大丈夫」等の複文) は層4 aux 判定に
    # 落ちるだけで安全側。
    re.compile(
        r"(?:自分で|自分が|私が|僕が|俺が|わたしが|こっちで|こちらで)"
        r"[^。．!！?？\n]*"
        r"(?:[うくぐつぬぶむる]|(?<!で)す)"
        r"(?:ね|よ|わ|から|かな)?"
        r"[\s　!！。.…]*$",
    ),
]
#: ローカルファイル/ディレクトリを対象にしていることが明確な語。パス自体は
#: 含まれていなくてよい (「そのファイル」のような anaphoric 参照を拾うため)。
_LOCAL_FILE_REFERENCE_RE = re.compile(
    r"(?:ファイル|フォルダ|ディレクトリ|保存した"
    r"|(?<![A-Za-z])file(?![A-Za-z])|(?<![A-Za-z])folder(?![A-Za-z])"
    r"|(?<![A-Za-z])directory(?![A-Za-z]))",
    re.IGNORECASE,
)
#: web リソースを対象にしていることを示す語。1 つでもあればローカル限定と
#: みなさない (「そのURLをファイルに保存して」等の url_write 正規フロー保護)。
_WEB_REFERENCE_RE = re.compile(
    r"(?:URL|https?://|ウェブ|サイト|ページ|ニュース|記事|ブログ|ドメイン|リンク"
    r"|(?<![A-Za-z])web(?![A-Za-z])|(?<![A-Za-z])site(?![A-Za-z])"
    r"|(?<![A-Za-z])page(?![A-Za-z])|(?<![A-Za-z])news(?![A-Za-z])"
    r"|(?<![A-Za-z])fetch(?![A-Za-z])|(?<![A-Za-z])browse(?![A-Za-z])"
    r"|(?<![A-Za-z])link(?![A-Za-z])|(?<![A-Za-z])domain(?![A-Za-z]))",
    re.IGNORECASE,
)


def _query_targets_local_file_only(query: str) -> bool:
    """ローカルファイル参照が明確で、web 参照シグナルが無いか (純粋関数)。"""
    return bool(_LOCAL_FILE_REFERENCE_RE.search(query)) and not _WEB_REFERENCE_RE.search(
        query,
    )


def _query_has_tool_signal(query: str, context: str = "") -> bool:
    """クエリにツール操作シグナル (ツールパターン / Windows・Unix パス / URL) を含むか。"""
    patterns = select_locale_variant(_TOOL_PATTERNS, _TOOL_PATTERNS_EN)
    return (
        any(p.search(query) for p in patterns)
        or bool(_PATH_OR_URL_SIGNAL_RE.search(query))
        # ディレクトリ列挙は _TOOL_PATTERNS のどれにも当たらず、knowledge query
        # として落ちて捏造回答になっていた (2026-08-03 ライブ監査)。
        or asks_directory_listing(query)
        # 式が書かれていない計算文章題も calculate が要る。補助タスク常駐時は
        # 5.2 層 (_judge_with_calculate_fallback) が拾うが、on_demand では
        # そこが撃てないため、ここを通さないとネイティブ層にも届かず base の
        # 暗算に倒れる (2026-08-08 ライブ監査:「時速240kmで2時間30分走ると
        # 何km進みますか。」→ 540km。正解 600km)。
        or looks_like_numeric_question(query, context)
        # 「<識別子> はどこで使われていますか」は _TOOL_PATTERNS のどれにも
        # 当たらず、文末の「〜ですか」で knowledge query として落ちる。
        # ルール層を素通りした結果、分類器が所在探索に無意味な list_directory
        # を選び 218.6 秒を捨てていた (2026-08-16 ライブ監査ターン 19)。
        or _is_code_usage_location_query(query)
    )
