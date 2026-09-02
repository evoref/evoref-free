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
    CALCULATE_TERM,
    DATETIME_SIGNAL_TERMS_EN,
    EXECUTABLE_QUERY_PATTERNS_EN,
    EXECUTABLE_QUERY_PATTERNS_JA,
    EXECUTABLE_QUERY_RE_EN,
    EXECUTABLE_QUERY_RE_JA,
    RUNTIME_INFO_QUERY_RE,
    SESSION_ANCHOR_EN,
    SESSION_PROXIMITY_WINDOW_EN,
    SESSION_TOPIC_BREAK_LOOKAHEAD_EN,
    ascii_boundary_alternation,
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
#: web リソースを対象にしていることを示す語。**web 意図の唯一の定義** —
#: ツール要否シグナル (``_TOOL_PATTERNS`` / ``_TOOL_PATTERNS_EN``)、
#: ``_infer_tool`` の fetch_url 分岐、``_query_targets_local_file_only`` の
#: 3 箇所が同じ語彙を使う。1 つでもあればローカル限定とみなさない
#: (「そのURLをファイルに保存して」等の url_write 正規フロー保護)。
#: ASCII 語は境界必須 — 境界無しの ``site`` は "opposite" に、``page`` は
#: "homepage" に部分一致し、無関係な依頼が fetch_url へ振られていた。
_WEB_REFERENCE_RE = re.compile(
    r"(?:URL|https?://|ウェブ|サイト|ページ|ニュース|記事|ブログ|ドメイン|リンク"
    r"|(?<![A-Za-z])web(?![A-Za-z])|(?<![A-Za-z])site(?![A-Za-z])"
    r"|(?<![A-Za-z])page(?![A-Za-z])|(?<![A-Za-z])news(?![A-Za-z])"
    r"|(?<![A-Za-z])fetch(?![A-Za-z])|(?<![A-Za-z])browse(?![A-Za-z])"
    r"|(?<![A-Za-z])link(?![A-Za-z])|(?<![A-Za-z])domain(?![A-Za-z]))",
    re.IGNORECASE,
)

#: 「動くか / エラーが無いか」を確かめる依頼 (``_infer_tool`` の verify_syntax
#: 分岐)。ASCII 語は境界必須 — 境界無しの ``work`` は "network.py" に、``bug`` は
#: "debug.py" に部分一致し、単なる読取依頼が構文検証へ振られていた。
_VERIFY_SYNTAX_INTENT_RE = re.compile(
    r"(?:動作|動[くい]|実行でき|エラー|バグ|正常|正しく動)|"
    + ascii_boundary_alternation("work", "run correctly", "execute", "error", "bug"),
    re.IGNORECASE,
)

#: 「計算 / calculate」。語彙は core.intent_vocab が SSOT (``_infer_tool`` と共有)。
_CALCULATE_RE = re.compile(CALCULATE_TERM, re.IGNORECASE)

#: 裸の時制語 (today / now / date / time)。**ツール要否シグナル専用** の分岐 —
#: ``_infer_tool`` / コマンド合成は疑問構文限定の ``DATETIME_QUERY_RE`` を使う
#: (core.intent_vocab の分岐理由コメント参照)。
_DATETIME_SIGNAL_RE_EN = re.compile(DATETIME_SIGNAL_TERMS_EN, re.IGNORECASE)

# ルールベースフォールバック用パターン
# 注意: 「検索」等の汎用語は知識質問にもマッチするため、
# コード/ファイル文脈を要求するパターンのみ含める。
# 実行可能クエリ (システム情報 / 日時 / 数値処理 / データ処理 / 変換) の語彙は
# core.intent_vocab が SSOT (``EXECUTABLE_QUERY_PATTERNS_JA``)。router の
# ``_EXECUTABLE_QUERY_PATTERNS`` / ``_infer_tool`` のゲート / コマンド合成の
# ルール表と同じ部品から組む — 以前は 4 箇所に書き写され、「変換」の除外や
# 「日付型」のガードが片方にしか入っていなかった。
_TOOL_PATTERNS = [
    re.compile(r"(?:ファイル|file).*(?:読|書|開|作成|削除)", re.IGNORECASE),
    re.compile(r"(?:コマンド|command).*(?:実行|run)", re.IGNORECASE),
    # コード/ファイル検索: 汎用「検索」は知識質問にマッチするため除外。
    # ASCII トークンは単語境界必須 ("crossencoder" の 'code' 等への部分一致誤爆対策、
    # CPU/RAM 境界ガードと同じ理由)。日本語側 (コード/ファイル/ソース/検索) は対象外。
    *_CODE_SEARCH_PATTERNS,
    _WEB_REFERENCE_RE,
    _CALCULATE_RE,
    # ファイルパスを含むクエリ（C:\, E:\, /home/ 等）+ 出力/保存/生成系動詞
    re.compile(r"[A-Za-z]:\\", re.IGNORECASE),
    re.compile(r"(?:出力|保存|生成|作成|書き出|エクスポート).*(?:して|する)", re.IGNORECASE),
    re.compile(r"(?:プログラム|コード|スクリプト|関数|クラス).*(?:作|書|生成)", re.IGNORECASE),
    # 「実行して」「動かして」等の動詞（ファイルパスやバッククォート付き）
    re.compile(r"(?:実行|動かし|起動|run|exec).*(?:して|する|しろ)", re.IGNORECASE),
    # バッククォート内コマンド
    re.compile(r"`[^`]+`", re.IGNORECASE),
    # --- Python 実行可能クエリ (SSOT: core.intent_vocab) ---
    *EXECUTABLE_QUERY_PATTERNS_JA,
    _DATETIME_SIGNAL_RE_EN,
]

# _TOOL_PATTERNS の英語版。GUI 左下の言語設定が 'en' の場合のみ使う
# (_TOOL_PATTERNS とは locale で完全に排他利用される)。既に ASCII/日英混在で
# 機能するエントリ (コード検索/URL/計算/日時/OS/env 等) は locale='en' でも
# 引き続き評価できるようそのまま複製する。
_TOOL_PATTERNS_EN = [
    # 「ファイル + 読み書き動詞」。旧定義は read/open の **後に** write 系動詞を
    # 必須にしており、"read the file config.yaml" のような単一操作に一度も
    # 一致しなかった (JA 側の ``ファイル.*(?:読|書|…)`` と非対称だった)。英語は
    # 動詞が名詞の前に来る ("read the file") ので語順も問わない。
    re.compile(
        r"\bfile\b.*\b(?:read|open|write|modify|change|update|delete|remove|edit)\b"
        r"|\b(?:read|open|write|modify|change|update|delete|remove|edit)\b.*\bfile\b",
        re.IGNORECASE,
    ),
    re.compile(r"(?:コマンド|command).*(?:実行|run)", re.IGNORECASE),
    *_CODE_SEARCH_PATTERNS,
    _WEB_REFERENCE_RE,
    _CALCULATE_RE,
    re.compile(r"[A-Za-z]:\\", re.IGNORECASE),
    re.compile(r"\b(?:save|export|output)\b.{0,20}\b(?:it|this|that|to|as|file)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:program|code|script|function|class)\b.*"
        r"\b(?:write|create|generate|build|implement)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:run|execute|exec)\b.{0,20}\b(?:this|that|it|the\s+\w+)\b", re.IGNORECASE),
    re.compile(r"`[^`]+`", re.IGNORECASE),
    # --- Python 実行可能クエリ (SSOT: core.intent_vocab) ---
    *EXECUTABLE_QUERY_PATTERNS_EN,
    _DATETIME_SIGNAL_RE_EN,
]

# _infer_tool() の実行可能クエリ判定ゲート (システム情報・数値処理・
# データ処理・変換)。語彙は _TOOL_PATTERNS と同じ SSOT を 1 本の alternation に
# 畳んだもの。以前は独立に保持されており、bare「変換」/ 無ガードの「日付」/
# ``CPU バウンド`` の除外漏れが **ここだけ** に残っていた (2026-09-02 監査)。
_INFER_TOOL_EXEC_QUERY_RE = EXECUTABLE_QUERY_RE_JA

# _INFER_TOOL_EXEC_QUERY_RE の英語版 (SSOT 由来。ASCII 語は全て境界付き —
# 境界無しだと program/framework/diagram が RAM 等に、summary/resume/assume が
# sum/mean 等に部分マッチする。2026-07-22 監査で判明)。
_INFER_TOOL_EXEC_QUERY_RE_EN = EXECUTABLE_QUERY_RE_EN
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
#: 衝突するため、容量を問う語との共起を要求する。それでも「メモリに保存して
#: **使って**ください」「メモリに**残って**いますか」は状態語の共起で当たって
#: いた (2026-09-02 監査) ので、``メモリ(に|で|へ)`` の直後が記憶動詞
#: (保存 / 残 / 記録 / 覚え …) なら EvorefMem の話として除外する。ハードウェア側の
#: 「メモリに何GB積んで」は記憶動詞ではないので通る。
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
    r"(?![にでへ]\s*(?:保存|残|記録|入れ|覚え|書|蓄え|溜め|しま))"
    r"|(?<![A-Za-z])RAM(?![A-Za-z])"
    r"|(?<![A-Za-z])memory(?![A-Za-z]))"
    r"[^。．\n]{0,12}"
    # 総量 / 合計 / トータル / いくら は 2026-09-02 追加 (「搭載メモリの総量は？」
    # 「搭載メモリはいくら？」が語彙に無くどの層にも掛からなかった)。
    r"(?:容量|サイズ|総量|合計|トータル|いくら|何\s*(?:GB|ギガ|MB)|いくつ"
    r"|どれ(?:く|ぐ)らい|積んで|載って"
    r"|使用[率量]|空き|残[りって]|使って"
    r"|(?<![A-Za-z])(?:size|usage|used|free|available|total)(?![A-Za-z]))"
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
#: 「ドライブ」はドライブレターを伴う形 (``E ドライブ``) だけ採る — 裸の
#: 「ドライブ」は「ハードドライブの寿命について記憶を消して」のような
#: **記憶の削除依頼** にも現れる (2026-09-02 監査)。
_DELETE_FS_TARGET_RE = re.compile(
    r"(?:ファイル|フォルダ|ディレクトリ|(?<![A-Za-z])[A-Za-z]\s*ドライブ)"
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
#: ローカルファイル/ディレクトリを対象にしていることが明確な語。パス自体は
#: 含まれていなくてよい (「そのファイル」のような anaphoric 参照を拾うため)。
_LOCAL_FILE_REFERENCE_RE = re.compile(
    r"(?:ファイル|フォルダ|ディレクトリ|保存した"
    r"|(?<![A-Za-z])file(?![A-Za-z])|(?<![A-Za-z])folder(?![A-Za-z])"
    r"|(?<![A-Za-z])directory(?![A-Za-z]))",
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
        # 式が書かれていない計算文章題も calculate が要る。ここを通さないと
        # 分類器 (層 5.9) のゲートが閉じたまま base の暗算に倒れる
        # (2026-08-08 ライブ監査:「時速240kmで2時間30分走ると何km進みますか。」
        # → 540km。正解 600km)。
        or looks_like_numeric_question(query, context)
        # 「<識別子> はどこで使われていますか」は _TOOL_PATTERNS のどれにも
        # 当たらず、文末の「〜ですか」で knowledge query として落ちる。
        # ルール層を素通りした結果、分類器が所在探索に無意味な list_directory
        # を選び 218.6 秒を捨てていた (2026-08-16 ライブ監査ターン 19)。
        or _is_code_usage_location_query(query)
    )
