"""補助タスクによるツール呼び出し判定（Free/Pro 共通）

ユーザークエリと利用可能なツール一覧を補助タスクに提示し、
ツール呼び出しの要否・ツール名・引数を判定する。
補助タスク未接続時はルールベースにフォールバックする。
"""

from __future__ import annotations

import ast
import re
import shlex
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

import httpx

from backend.config import get_project_root
from backend.free.agent.router import (
    HISTORY_KEYWORDS,
    HISTORY_KEYWORDS_EN,
    asks_directory_listing,
    is_environment_fact_query,
)
from backend.free.core.intent_vocab import (
    DATETIME_QUERY_RE,
    NUMBER_LITERAL_RE,
    PROXIMAL_RECALL_KEYWORDS,
    SESSION_ANCHOR_EN,
    SESSION_PROXIMITY_WINDOW_EN,
    SESSION_TOPIC_BREAK_LOOKAHEAD_EN,
    looks_like_numeric_question,
    session_self_reference_pattern_ja,
)
from backend.free.core.locale_patterns import is_en_locale, select_locale_variant
from backend.free.core.session_mode import is_create_mode
from backend.free.agent.safety_patterns import (
    extract_command_literal,
    reject_readonly_violation,
    strip_command_literals,
)
from backend.free.agent.tools_registry import ToolDefinition, ToolsRegistry
from backend.free.agent.grammar_tool_classifier import (
    CLASSIFY_MAX_TOKENS,
    build_classifier_schema,
    build_tool_menu,
    parse_classifier_response,
)
from backend.free.llm.json_extract import extract_json_object
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.debug_logger import DebugLogger
    from backend.free.agent.learned_patterns import LearnedPatternStore
    from backend.free.memory.views.mem import MemFactView
    from backend.free.rag.cartridge_manager import CartridgeManager
    from backend.free.rag.embedding_backend import EmbeddingBackend

logger = get_logger("agent.tool_call_judge")

# executable command リコールの候補プールがこの件数未満のとき、類似度閾値を
# ``_RECALL_SMALL_POOL_MARGIN`` だけ嵩上げする。学習初期は top-K も success_avg も
# 選別として機能せず、類似度ゲート 1 本で決まってしまうため。
#: 実行可能コマンドを載せるツール名 (mode により片方のみ利用可能)。
_COMMAND_TOOL_NAMES = frozenset({"run_command", "run_command_readonly"})

#: 「実行するとファイル/環境の状態が変わる」ツール名。mode 制約で撃てなかった
#: 場合、``action_blocked`` を立てて完了報告の捏造を禁じる必要がある
#: (``run_command`` は読取専用の兄弟へ載せ替わる経路があるため
#: ``_COMMAND_TOOL_NAMES`` 側で measurement_blocked として扱う)。
_STATE_CHANGING_TOOL_NAMES = frozenset({"write_file", "apply_patch", "delete_file"})

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

#: 処理対象の本文そのものを引数に取るツール。判定プロンプトの会話は切り詰めて
#: あるため、aux の転記をそのまま使うと断片だけが処理される。
_TEXT_OPERAND_TOOLS = frozenset({"summarize", "translate"})

#: 判定プロンプトへ載せる会話 1 メッセージあたりの文字数上限。切り詰め側と
#: 復元側で同じ定数を共有する (別々に持つと片方の変更で復元が効かなくなる)。
_JUDGE_CONTEXT_CHARS = 100

#: 同じ能力を持ち権限だけが違うツールの対応表 (優先順)。aux が mode 外の
#: 兄弟名を返したとき、撃てる方へ載せ替えて判定の意図を保つ。緩い側から厳しい
#: 側への一方向にだけ張る (逆向きに張ると chat が特権ツールへ昇格してしまう)。
_MODE_CAPABILITY_SIBLINGS: dict[str, tuple[str, ...]] = {
    "run_command": ("run_command_readonly",),
}

_RECALL_SMALL_POOL_SIZE = 3
_RECALL_SMALL_POOL_MARGIN = 0.1

# Windows / Unix の明示パス、または URL。ユーザーが対象を書いた決定論的シグナルで、
# aux の否定票より優先してよい (``_upgrade_command_via_aux`` の降格例外)。
#: ドライブレターの区切りは ``\`` と ``/`` の双方を受ける。バックスラッシュ限定
#: だったため ``E:/tmp/a.txt`` がツールシグナルとして検出されず、明示パス付きの
#: 依頼が knowledge query に落ちて「存在しない」と誤答していた (実インシデント
#: 2026-08-04 ライブ監査)。
_PATH_OR_URL_SIGNAL_RE = re.compile(
    r"[A-Za-z]:[\\/]|(?:^|[\s　])(?:/[\w._-]+){2,}|https?://",
)








def _executable_tool_for_mode(tools_registry: ToolsRegistry, mode: str) -> str:
    """現在の ``mode`` で使える executable コマンドツール名を解決する。

    create では従来の ``run_command``、chat では読み取り専用の
    ``run_command_readonly`` (modes=["chat"], hidden) を返す。どちらも使えない
    場合は ``""``。

    2026-07-18 の mode ゲート導入で chat の executable query (時刻 / OS /
    スペック等) が合成成功後に必ず no_tool へ格下げされる回帰が起きたため、
    executable 経路 (early-return / _infer_tool 実行可能クエリ分岐 / 層5
    fallback / SemMem recall) はこのヘルパでツール名を解決する。mode ゲート
    (_validate_tool_availability / deliberative 実行前ゲート) には例外を作らず、
    「chat で使える実行ツールは readonly 検証付き func しか登録されていない」
    を登録構造で保証する (docs/f_03_agent_engine.md §3.1)。
    """
    if tools_registry.is_available("run_command", mode):
        return "run_command"
    if tools_registry.is_available("run_command_readonly", mode):
        return "run_command_readonly"
    return ""


#: readonly の allow-list (python のみ) から漏れるが、**状態を変えないことが
#: 明らかな**検査コマンドの実行ファイル名。
#:
#: 用途は「拒否されたコマンドが *変更の試み* だったのか *測定の試み* だったのか」
#: の振り分けのみで、**実行可否は一切変わらない** (どちらも allow-list 違反として
#: 拒否される)。変わるのは base へ足す注記が ``_UNPERFORMED_ACTION_GUIDANCE``
#: (何も実行していない) か ``_UNMEASURED_FACT_GUIDANCE`` (測っていない) かだけ。
#:
#: 実インシデント (2026-08-15 ライブ監査 ターン12): 「本当に削除されましたか？
#: 確認して。」にネイティブ層が ``test -f <path>`` を選び、allow-list 違反で
#: 拒否 → 一律 ``_action_blocked`` が立ち「状態を変える操作を実行していない」の
#: 注記が入った結果、base が「ファイルの存在確認を行うツールが利用できない」と
#: 誤った説明で締めた (実際は read_file / list_directory が使える)。
#:
#: mutation を read と誤分類すると完了の捏造 (2026-08-08 の ``echo >> file``)
#: に戻るため、**曖昧なものは載せない**。判定不能なら従来どおり action 扱い。
_READONLY_INSPECT_COMMANDS: frozenset[str] = frozenset({
    "test", "ls", "dir", "cat", "type", "stat",
    "head", "tail", "wc", "grep", "findstr", "where", "which",
})


def _command_is_readonly_inspection(command: str) -> bool:
    """``command`` が「状態を変えない検査」と確実に言えるか。

    リダイレクト (``>`` / ``>>``) や連鎖 (``&&`` / ``;`` / ``|``) を含む場合は、
    先頭が検査コマンドでも後続で状態を変えうるので False を返す
    (``test -f x && rm x`` のような形を read と誤分類しない)。
    """
    if not command or any(t in command for t in (">", ">>", "&&", "||", ";", "|")):
        return False
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return False
    if not tokens:
        return False
    return Path(tokens[0]).name.lower().removesuffix(".exe") in (
        _READONLY_INSPECT_COMMANDS
    )


def _readonly_command_rejected(exec_tool: str, command: str) -> bool:
    """readonly ツールに載せる ``command`` が readonly 検証に違反するか。

    ``exec_tool`` が ``run_command_readonly`` のときだけ
    ``reject_readonly_violation`` を適用する (create の run_command は対象外)。
    judge 段でこれを弾くと、synth が返した非 readonly コマンド (PowerShell
    スニペット等) が実行段の "Error: readonly violation" ではなく no_tool に
    倒れ、LLM 知識回答へクリーンに落ちる。実行段のラッパ検証は最終防衛として
    別途残る (二重ガード)。
    """
    if exec_tool != "run_command_readonly":
        return False
    reject = reject_readonly_violation(command)
    if reject is not None:
        logger.info(
            "Readonly executable command rejected at judge stage (%s): %s",
            reject, command[:80],
        )
        return True
    return False


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

# ユーザークエリからドライブレターを抽出するパターン
# 「Eドライブ」「C:」「D drive」等のパターンにマッチし、
# 単一の英字（ドライブレター）をキャプチャする。
# ASCII 境界を使用して "PCの" 等の複数文字並びに誤マッチしないよう、
# 直前が英字でないことを保証する。
_DRIVE_LETTER_RE = re.compile(
    r"(?:^|[^A-Za-z])([A-Za-z])(?::|\s*ドライブ|\s*drive(?![A-Za-z]))",
    re.IGNORECASE,
)

# クエリ先頭の URL を抽出する。非 ASCII (CJK 等) を除外し、「URL + 日本語」
# 入力で末尾テキストを URL に取り込まないようにする
# (例: https://news.yahoo.co.jp/で取得して... → https://news.yahoo.co.jp/ のみ)。
_URL_IN_QUERY_RE = re.compile(r"(https?://[^\s\]）」』\u0080-\U0010ffff]+)")


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
    r"(?:ram|memory)(?![A-Za-z])",
    re.IGNORECASE,
)



def _build_spec_command(query: str) -> str:
    """システムスペックコマンドを生成する

    クエリにドライブレター指定（「Eドライブ」「C:」等）が含まれる場合は、
    そのドライブの容量を取得する。指定がなければシステムドライブ
    (Windows は %SystemDrive%、Unix は '/')。
    Windows / Unix の両方で動作するよう、パスはフォワードスラッシュで構築する
    （shutil.disk_usage は Windows でも 'E:/' を受理する）。

    フォールバックはかつてカレントディレクトリ ('.') だったが、これは backend
    プロセスの起動位置という**ユーザーから見えない値**に測定対象が依存する。
    実測 (2026-07-27 ライブ監査): 「C ドライブの空き容量は?」→ C: の 138 GB を
    回答した直後、「さっき調べた空き容量はディスク全体の何%?」でドライブ名が
    落ちて '.' にフォールバックし、cwd のある E: (553 GB free) を測って
    「さっき調べた空き容量 553 GB」と自己矛盾した回答を返した。
    「この PC の空き容量」はシステムドライブを指すのが自然で、かつ起動位置に
    依存せず決定論的になる。
    """
    m = _DRIVE_LETTER_RE.search(query)
    if m:
        letter = m.group(1).upper()
        py_path = f"'{letter}:/'"
    else:
        # サブプロセス側で評価する (実行ホストのシステムドライブを見る)。
        # os は既に import 済みで、os.environ / .get とも readonly guard の
        # 禁止属性ではない。
        py_path = "(os.environ.get('SystemDrive','C:')+'/' if os.name=='nt' else '/')"
    return (
        "python -c \""
        "import platform,os,shutil;"
        " print('OS:',platform.platform());"
        " print('CPU:',platform.processor() or platform.machine());"
        " print('Cores:',os.cpu_count());"
        f" t,u,f=shutil.disk_usage({py_path});"
        " print('Disk:',t//(1024**3),'GB total,',f//(1024**3),'GB free')"
        "\""
    )


# 現在時刻 / 日付クエリ。executable 判定の中で最も曖昧さが小さく、aux が
# 否定票を返しても regex 結果を維持してよい唯一の高特異度パターン
# (``_upgrade_command_via_aux`` の降格例外)。
# 定義は core.intent_vocab が SSOT (agent.router が同一定義を持っていたが、
# ``(?!間)`` ガードの有無など細部が食い違っていた)。
_DATETIME_QUERY_RE = DATETIME_QUERY_RE

#: 「N 日後 / N 年前」等の相対日付。単位の直後に 前/後 を要求するので
#: 「1 月 3 日」のような絶対日付には掛からない。
_RELATIVE_OFFSET_RE = re.compile(
    r"(\d{1,4})\s*(週間|週|[かヶケヵ箇]月|月|日|年)\s*(前|後|先)",
)

#: コマンドが日付演算をしている印。``_build_datetime_command`` が相対日付用に
#: 生成するコマンドは必ずどちらかを含む。リコールで引き当てた過去のコマンドが
#: 相対日付クエリに答えられるかの判定に使う (``recalled_command_fits_query``)。
_DATE_ARITHMETIC_RE = re.compile(r"timedelta|datetime\.datetime\(|datetime\.date\(")

#: 相対日付の単位 → コマンド生成の種別。
_OFFSET_UNITS = {
    "日": "days", "週": "weeks", "週間": "weeks",
    "月": "months", "か月": "months", "ヶ月": "months",
    "ケ月": "months", "ヵ月": "months", "箇月": "months",
    "年": "years",
}

#: 現在日時のみを返す既定コマンド。
#:
#: ``astimezone()`` を付けて **UTC オフセット付き**で出力する。プロンプトには
#: 別途 ``[現在日時 (UTC基準)]`` が注入されており、コマンド出力が naive
#: ローカル時刻だと 2 つの時計が無印で並ぶ。JST では 00:00-09:00 の間、
#: ローカル日付と UTC 日付が 1 日ずれるため、モデルはどちらを「今日」と
#: 呼ぶべきか判断できない (2026-08-05 ライブ監査で構造として確認)。
_DATETIME_NOW_COMMAND = (
    'python -c "import datetime; print(datetime.datetime.now().astimezone())"'
)

#: 相対日付コマンドの共通前置き (now と目標日を両方出す)。
_REL_PREFIX = 'python -c "import datetime; n=datetime.datetime.now().astimezone();'
_REL_SUFFIX = (
    " print('now:',n);"
    " print('target:',t.strftime('%Y-%m-%d (%A)'))\""
)


def _build_datetime_command(query: str) -> str:
    """日付 / 時刻クエリ用のコマンドを組み立てる。

    相対表現 (「3 年前の今日」「今日から 100 日後」) が含まれる場合は **目標日と
    その曜日まで Python に計算させる**。現在時刻だけを渡してモデルに暗算させると
    外す (実インシデント 2026-08-07 ライブ監査: 「3 年前の今日は何曜日でしたか？」
    に「火曜日」と回答。2023-08-07 は月曜日)。同じ日に「今日から 100 日後」は
    正答しており、暗算が当たるかどうかは運になっていた。

    相対表現が無ければ従来どおり現在日時のみを返す。
    """
    m = _RELATIVE_OFFSET_RE.search(query or "")
    if m is None:
        return _DATETIME_NOW_COMMAND
    kind = _OFFSET_UNITS.get(m.group(2))
    if kind is None:
        return _DATETIME_NOW_COMMAND
    n = int(m.group(1))
    signed = -n if m.group(3) == "前" else n

    if kind in ("days", "weeks"):
        body = f" t=n+datetime.timedelta({kind}={signed});"
    else:
        # 月/年は timedelta で表せない。月末クランプ (1/31 の 1 か月後 = 2/28 等)
        # を含めて構築する。``calendar`` は readonly guard の許可モジュール外、
        # ``datetime.replace`` は禁止属性なのでコンストラクタで組み立てる。
        total = " tm=(n.year*12+n.month-1)+" + str(
            signed * 12 if kind == "years" else signed,
        ) + ";"
        body = (
            total
            + " y=tm//12; mo=tm%12+1;"
            " lp=(y%4==0 and (y%100!=0 or y%400==0));"
            " dim=[31,29 if lp else 28,31,30,31,30,31,31,30,31,30,31][mo-1];"
            " t=datetime.datetime(y,mo,min(n.day,dim));"
        )
    return _REL_PREFIX + body + _REL_SUFFIX

# Python 実行で正確に答えられるシステム情報クエリのコマンドマッピング
# パターンにマッチしたクエリに対して、具体的な Python コマンドを生成する。
# コマンドは Windows cmd.exe / Unix sh の両方で動作するよう、
# 外側を "..." で囲み内側で '...' を使用する。
# 第二要素が Callable の場合はクエリ文字列を渡して動的に生成する
_EXECUTABLE_QUERY_COMMANDS: list[tuple[re.Pattern, "str | Callable[[str], str]"]] = [
    # 現在時刻 / 日付 (「何月|何日|何曜日」は明確な疑問語のみ追加、
    # 「今日|明日|昨日」単独は誤検出するため見送り)
    # ``astimezone()`` を付けて **UTC オフセット付き**で出力する。プロンプトには
    # 別途 ``[現在日時 (UTC基準)]`` が注入されており、コマンド出力が naive
    # ローカル時刻だと 2 つの時計が無印で並ぶ。JST では 00:00-09:00 の間、
    # ローカル日付と UTC 日付が 1 日ずれるため、モデルはどちらを「今日」と
    # 呼ぶべきか判断できない (2026-08-05 ライブ監査で構造として確認。当日は
    # 22:43 JST = 13:43 UTC で偶然一致しており表面化しなかった)。
    # オフセットを添えれば両者の関係が出力から読み取れる。
    (_DATETIME_QUERY_RE, _build_datetime_command),
    # システムスペック（OS / CPU / コア数 / ディスク）
    # ドライブレター指定があれば指定ドライブの容量を返す
    # CPU 等の英字略語は ASCII 境界必須 ("program" の 'ram' 誤マッチ対策)
    # spec(s)? で複数形 ("PC specs") も許容する。
    # メモリ / memory / RAM は 2026-07-27 に外した。GPU/VRAM (下記) と同じ理由で、
    # コマンドが搭載メモリ量を一切出力しないのにパターンだけ一致して発火し、
    # サブプロセスと 1 ターンを消費した末に「ツール結果にメモリ容量の数値は
    # 記載されていません」としか返せなかった (実測: 「この PC のメモリは何 GB
    # 積んでいますか？」)。Windows で搭載 RAM を取る手段 (ctypes / wmic /
    # Get-CimInstance) は _READONLY_SAFE_MODULES / 危険コマンド判定が全て拒否
    # するため、正しい情報を返すコマンドへ差し替える経路は存在しない。
    # ``capacity`` は 2026-08-09 に外した。他の語と違い **データ項目名として
    # 普通に現れる** ため、機械スペックの要求とは限らない (2 回目のライブ監査:
    # 「同じ表を JSON 配列にしてください。キーは category, fee, capacity で…」
    # という純粋な整形依頼で OS/CPU/コア数の取得コマンドが撃たれた。
    # `capacity` を別名に変えると発火しない = この語が唯一の引き金だった)。
    # ``容量`` 単独も 2026-08-10 に外した (同じ理由。「DBの容量」「総容量」
    # 「容量効率」で spec コマンドが撃たれた)。機器を名指しする質問は
    # ストレージ / ディスク / ドライブ / disk / drive 側で拾えるので
    # 取りこぼしは実質無い。「空き容量」「残り容量」「使用容量」は残す。
    # ``disk`` / ``storage`` は他の ASCII トークンと同じく境界必須へ揃える
    # (このファイルの規約。境界無しだと部分一致で誤爆する)。
    (re.compile(
        r"(?:スペック|(?<![A-Za-z])CPU(?![A-Za-z])"
        r"|ディスク|(?:空き|残り|使用)容量|ストレージ|ドライブ"
        r"|(?<![A-Za-z])disk(?![A-Za-z])|(?<![A-Za-z])storage(?![A-Za-z])"
        r"|(?<![A-Za-z])specs?(?![A-Za-z])"
        r"|(?<![A-Za-z])drive(?![A-Za-z]))",
        re.IGNORECASE,
    ), _build_spec_command),
    # GPU / VRAM のエントリは 2026-07-25 に削除した。
    # コマンドが platform.platform() / platform.machine() しか実行しておらず
    # GPU 型番も VRAM 容量も一切返さないのに、実行が成功扱いになっていた
    # (実測: 「さっき伝えた GPU は？」→ "Platform: Windows-11 / Machine: AMD64" →
    #  「ツール結果に GPU 型番は含まれていません」と誤答)。
    # safety_patterns._READONLY_SAFE_MODULES が wmic / Get-CimInstance /
    # nvidia-smi / 外部ライブラリをすべて拒否するため、正しい情報を返すコマンドへ
    # 差し替える経路は存在しない。エントリを消すと _infer_tool が引数なしを返し
    # _suppress_commandless_run_command が no_tool へ落とすので、GPU/VRAM は
    # 会話履歴と LLM 知識に委ねる (そちらの方が誤答が少ない)。
    # IP アドレス / ホスト名
    (re.compile(
        r"(?:IP\s*アドレス|ホスト名"
        r"|(?<![A-Za-z])hostname(?![A-Za-z])"
        r"|(?<![A-Za-z])ip\s*address)",
        re.IGNORECASE,
    ), "python -c \""
       "import socket;"
       " h=socket.gethostname();"
       " print('Hostname:',h);"
       " print('IP:',socket.gethostbyname(h))"
       "\""),
    # OS
    (re.compile(
        r"(?:(?<![A-Za-z])OS(?![A-Za-z])|オペレーティングシステム"
        r"|(?<![A-Za-z])Windows(?![A-Za-z])"
        r"|(?<![A-Za-z])Linux(?![A-Za-z])"
        r"|(?<![A-Za-z])Mac(?![A-Za-z]))",
        re.IGNORECASE,
    ), "python -c \""
       "import platform,sys;"
       " print(platform.platform());"
       " print(sys.platform,platform.release())"
       "\""),
    # Python バージョン
    (re.compile(
        r"(?:Python|python)\s*(?:バージョン|version)",
        re.IGNORECASE,
    ), "python --version"),
    # 環境変数
    (re.compile(
        r"(?:環境変数|(?<![A-Za-z])env(?![A-Za-z])"
        r"|(?<![A-Za-z])PATH(?![A-Za-z]))",
        re.IGNORECASE,
    ), "python -c \""
       "import os;"
       " [print(k,'=',v[:80]) for k,v in sorted(os.environ.items())[:30]]"
       "\""),
]







#: 数値計算クエリの事前フィルタ。式が書かれていれば層1 (_extract_arithmetic_expression)
#: が決定論的に処理するため、ここは「数値は複数あるが式は書かれていない」ものだけを
#: 対象にする。aux 往復 (realtime) を増やさないよう条件は厳しめにする。
#: 実体は ``core.intent_vocab`` が SSOT (``agent.router`` も同じ判定を使う)。
#: 既存の呼出元とテストのために旧名を残す。
_NUMBER_LITERAL_RE = NUMBER_LITERAL_RE


#: 単位系の定義そのものに由来する定数。ユーザーが書いた数値ではないが、
#: 「モデルが知識から思い出した換算率」でもない (分/時、時/日、SI 接頭辞、
#: パーセント)。定義上一意なので誤記憶しようがなく、捏造検出の対象外にする。
#: マイル→キロ (1.609) のような **知識** の換算率はここに入れない
#: (実インシデント 2026-07-29 ライブ監査: 「時速72kmで45分間に進む距離は
#: 何kmですか？」で 45/60 の 60 がグラウンディングに落ち、base の暗算で
#: 90km と誤答した。正解 54km)。
#:
#: 2 進接頭辞 (1024 の冪) も SI 接頭辞と同じ **定義** であり、記憶違いの余地が
#: ない。バイト→KiB/MiB/GiB の換算で式に必ず現れるため、外すとサイズ計算が
#: 丸ごと no_tool へ落ちる (実インシデント 2026-08-09 ライブ監査: 直前ターンの
#: 1,277,500 行 × 240 バイトに対しネイティブ層が
#: ``(1277500 * 240) / (1024 * 1024)`` を正しく合成したのに、``1024`` が
#: グラウンディングに落ちて棄却され、base の暗算で「約288MB」と誤答した。
#: 正解は 306,600,000 バイト = 292.4 MiB)。
#: 暦の周期 (週/年、うるう年の日数、秒/日) も 365 や 1440 と同じ**暦の定義**で、
#: 知識として思い出す換算率ではない。52 が無いために「週に3冊なら年間何冊か」で
#: ネイティブ層が正しく合成した ``3 * 52`` が ungrounded で棄却され、base の
#: 暗算に落ちていた (実インシデント 2026-08-10 ライブ監査)。
_UNIT_SYSTEM_CONSTANTS = frozenset({
    "7", "10", "12", "24", "52", "60", "100", "365", "366",
    "1000", "1440", "3600", "86400",
    "0.1", "0.01", "0.001",
    # SI 接頭辞 (10 の冪) — 1000 は上にある
    "1000000", "1000000000", "1000000000000",
    # 2 進接頭辞 (2 の冪): KiB / MiB / GiB / TiB
    "1024", "1048576", "1073741824", "1099511627776",
})


def _synthesized_expression_grounded(
    expression: str, query: str, context: str = "",
) -> bool:
    """合成式の数値がすべてクエリ / 直近の会話に現れるか (純粋関数)。

    aux が知識から定数を補う (例: クエリにも会話にも無い換算率を持ち出す) と、
    ツールは「正しく計算された嘘」を返してしまう。式に現れる数値リテラルが
    クエリまたは ``context`` 中に文字列として存在することを要求し、捏造を
    決定論的に弾く。``context`` を許すのは、会話で一度提示された数値は
    「対話に書かれた事実」であってモデルの想像ではないため。
    """
    numbers = _NUMBER_LITERAL_RE.findall(expression)
    if not numbers:
        return False
    return not _ungrounded_numbers(expression, query, context)


def _ungrounded_numbers(
    expression: str, query: str, context: str = "",
) -> tuple[str, ...]:
    """式のうち対話から辿れない数値リテラルを、出現順に重複なく返す (純粋関数)。

    ``_synthesized_expression_grounded`` の真偽ではなく **どの値が説明できないか**
    を返す。式を捨てずに実行する経路 (補助タスク非常駐) で、その値を回答に開示
    させるために使う。
    """
    known = _known_numbers(query) | _known_numbers(context)
    known.update(_UNIT_SYSTEM_CONSTANTS)
    seen: list[str] = []
    for n in _NUMBER_LITERAL_RE.findall(expression):
        if n not in known and n not in seen:
            seen.append(n)
    return tuple(seen)


#: 桁区切り入りの数字 (``2,660`` / ``1,234,567``)。アシスタント自身が金額を
#: この書式で提示するため、次のターンでその数値を使う式が「対話に無い数値」と
#: 誤判定されていた (実インシデント 2026-08-03 ライブ監査: 直前の回答
#: 「2,660円です」を受けた ``2926 + 500`` が ungrounded で no_tool に落ち、
#: 決定論の calculate 経路を失って base の暗算に回っていた)。
_GROUPED_NUMBER_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?")

#: 明示されたパーセント (``10%`` / ``10 パーセント``)。
_PERCENT_LITERAL_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|％|パーセント)")

#: 時間の長さ表現 (``2時間30分`` / ``2時間半`` / ``90分`` / ``3時間``)。
#: 「2時間30分」から 2.5 (時間) と 150 (分) は **表記から決定論で導ける**値で、
#: モデルが知識から持ち出した定数ではない。桁区切り・パーセントと同じ扱い。
_DURATION_HM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*時間\s*(\d+(?:\.\d+)?)\s*分")
_DURATION_H_HALF_RE = re.compile(r"(\d+(?:\.\d+)?)\s*時間半")
_DURATION_H_RE = re.compile(r"(\d+(?:\.\d+)?)\s*時間")
_DURATION_M_RE = re.compile(r"(\d+(?:\.\d+)?)\s*分(?!の)")
#: 「5分30秒」型 (分 + 秒)。``時間 + 分`` と同じ構造なのに欠けていた。
#: ペース (「キロ5分30秒」) や所要時間で普通に使う表記で、式に現れるのは
#: ``5.5`` (分) や ``330`` (秒) であってクエリに書かれた ``5`` と ``30`` ではない。
_DURATION_MS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*分\s*(\d+(?:\.\d+)?)\s*秒")

#: 時間の刻み幅 (``30分刻み`` / ``15分単位`` / ``10分ごと`` / ``30分間隔``)。
#: 式に現れるのは「1時間あたりの区画数」(30分刻み → 2) で、クエリには 30 しか
#: 書かれていない。時間+分・分+秒と同じく **表記から一意に導ける**値。
#: 実インシデント 2026-08-10 ライブ監査: 「9時から18時まで30分刻み、8部屋、
#: 5営業日」でネイティブ層が正しく ``8 * 5 * (18 - 9) * 2`` (= 720) を合成したのに
#: ``2`` が ungrounded 判定になり no_tool へ落ち、base の暗算で 1,680 / 1,440 と
#: 誤答した (しかも見出しと計算式が食い違った)。
_INTERVAL_M_RE = re.compile(r"(\d+(?:\.\d+)?)\s*分\s*(?:刻み|単位|ごと|間隔)")

#: 曜日のレンジ (``月〜金`` / ``月曜から金曜まで``)。日数はレンジから一意に決まる
#: (月〜金 → 5) が、クエリに数字としては現れない。上の刻み幅と同じ回で必要に
#: なった (2026-08-10 ライブ監査の総スロット数は ``8 * 5 * (18 - 9) * 2`` で、
#: ``5`` も ``2`` も書かれていなかった)。
_WEEKDAY_ORDER = "月火水木金土日"
_WEEKDAY_RANGE_RE = re.compile(
    r"([月火水木金土日])\s*(?:曜日?)?\s*(?:〜|～|-|–|から)\s*([月火水木金土日])\s*(?:曜日?)?",
)


def _known_numbers(text: str) -> set[str]:
    """``text`` に「書かれている」とみなせる数値リテラルを集める (純粋関数)。

    素の数字に加えて 2 種類を同一視する。どちらも **対話に現れた表記から決定論で
    導ける**もので、モデルが知識から持ち出した定数ではない:

    - 桁区切り: ``2,660`` → ``2660``。数える側 (式) は区切りを打たないため、
      正規化しないと自分が直前に提示した金額を「知らない数値」と判定してしまう
    - パーセント: ``10%`` → ``0.1`` / ``1.1`` / ``1.10``。税率・割引率の計算で
      式に現れる倍率は、クエリ中の百分率から一意に決まる
    """
    known = set(_NUMBER_LITERAL_RE.findall(text))
    for grouped in _GROUPED_NUMBER_RE.findall(text):
        known.add(grouped.replace(",", ""))
    for pct in _PERCENT_LITERAL_RE.findall(text):
        try:
            rate = float(pct) / 100.0
        except ValueError:
            continue
        # 「10%」からは 0.1 (率) と 1.1 (加算後の倍率) が導ける。式側の表記ゆれ
        # (1.1 / 1.10) を吸収するため両方を登録する。
        for value in (rate, 1.0 + rate):
            known.add(f"{value:g}")
            known.add(f"{value:.2f}")
    known.update(_duration_derived_numbers(text))
    return known


def _duration_derived_numbers(text: str) -> set[str]:
    """時間の長さ表現から導ける数値を集める (純粋関数)。

    「2時間30分で何km進むか」型の文章題では、式に現れるのは ``2.5`` (時間) や
    ``150`` (分) であって、クエリに書かれた ``2`` と ``30`` ではない。桁区切り・
    パーセントと同じく **表記から一意に導ける**値なので、捏造ではない。

    実インシデント (2026-08-08 ライブ監査): 「時速240kmで2時間30分走ると何km
    進みますか。」でネイティブ層が正しく ``240 * 2.5`` を選んだのに、``2.5`` が
    クエリに無いという理由で ungrounded 判定になり no_tool へ落ちた。

    「分 + 秒」も同じ構造なのに欠けていた (2026-08-09 2 回目のライブ監査):
    「フルマラソンの距離を キロ5分30秒 のペースで走ると何時間何分？」で
    ネイティブ層が正しく ``42.195 * 5.5`` を選んだのに、``5.5`` が
    (5 と 30 しか書かれていないため) ungrounded 判定になり no_tool へ落ち、
    base の暗算で「3時間47分15秒」と誤答した (正 3時間52分4秒)。
    補助タスク非常駐でも救済経路自体は生きており、塞いでいたのはこのゲートだった。
    """
    derived: set[str] = set()

    def add(value: float) -> None:
        derived.add(f"{value:g}")
        derived.add(f"{value:.2f}")

    for hours, minutes in _DURATION_HM_RE.findall(text):
        total_h = float(hours) + float(minutes) / 60.0
        add(total_h)
        add(float(hours) * 60.0 + float(minutes))
    for minutes, seconds in _DURATION_MS_RE.findall(text):
        # 分単位 (5分30秒 → 5.5) と秒単位 (→ 330)。時間+分と同じ 2 通り。
        add(float(minutes) + float(seconds) / 60.0)
        add(float(minutes) * 60.0 + float(seconds))
    for hours in _DURATION_H_HALF_RE.findall(text):
        add(float(hours) + 0.5)
        add((float(hours) + 0.5) * 60.0)
    for hours in _DURATION_H_RE.findall(text):
        add(float(hours) * 60.0)
    for minutes in _DURATION_M_RE.findall(text):
        add(float(minutes) / 60.0)
    for minutes in _INTERVAL_M_RE.findall(text):
        step = float(minutes)
        if step > 0:
            # 1 時間あたりの区画数 (30分刻み → 2) と、区画の時間 (→ 0.5)。
            add(60.0 / step)
            add(step / 60.0)
    for start, end in _WEEKDAY_RANGE_RE.findall(text):
        # 両端を含む日数 (月〜金 → 5)。週をまたぐ指定 (金〜月) も剰余で数える。
        span = (_WEEKDAY_ORDER.index(end) - _WEEKDAY_ORDER.index(start)) % 7 + 1
        add(float(span))
    return derived


#: 層5.2 の数値グラウンディングに使う直近ターン数。長く取るほど無関係な数値を
#: 拾いやすくなるため短く保つ。
_CALCULATE_CONTEXT_TURNS = 4



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


def _normalize_path_text(text: str) -> str:
    """パス照合用にセパレータと大小文字を正規化する (純粋関数)。"""
    return text.replace("\\", "/").casefold()


def _dialogue_text(
    conversation: list[dict] | None, turns: int | None = None,
) -> str:
    """会話本文を連結して返す (純粋関数)。

    「対話に現れた数値」を数えるために使う。role は問わない (換算率は
    アシスタント発話側に出る)。``turns`` を渡すと末尾その数のターンに絞る。
    """
    if not conversation:
        return ""
    window = conversation[-turns:] if turns else conversation
    parts = [
        str(turn.get("content") or "")
        for turn in window
        if isinstance(turn, dict)
    ]
    return "\n".join(p for p in parts if p)


def _recent_dialogue_text(conversation: list[dict] | None) -> str:
    """直近 ``_CALCULATE_CONTEXT_TURNS`` ターンの本文を連結して返す (純粋関数)。

    層5.2 の事前フィルタと合成式のグラウンディング検証で使う。式を合成する
    層なので、捏造の余地を絞るために窓は狭く保つ。
    """
    return _dialogue_text(conversation, _CALCULATE_CONTEXT_TURNS)


def _recent_dialogue_messages(
    conversation: list[dict] | None,
    turns: int = _CALCULATE_CONTEXT_TURNS,
) -> list[dict]:
    """直近 ``turns`` 件を messages 配列として返す (純粋関数)。

    ネイティブ tool calling へ渡す最小の文脈。「そのファイルを読んで」のような
    照応をモデルが解けるように直近だけ載せ、prefill を膨らませない
    (1 メッセージ ``_JUDGE_CONTEXT_CHARS`` 文字で切り詰め)。
    """
    if not conversation:
        return []
    messages: list[dict] = []
    for turn in conversation[-turns:]:
        role = turn.get("role")
        content = turn.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        messages.append({"role": role, "content": content[:_JUDGE_CONTEXT_CHARS]})
    return messages


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

def _coerce_positive_int(value: object) -> int | None:
    """aux の型崩れ JSON 由来の値を正の int へ正規化する (int / 数値文字列 /
    整数値 float を受理)。bool や非数値、0 以下は ``None`` を返す。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        return int(value) if value.is_integer() and value > 0 else None
    if isinstance(value, str):
        s = value.strip()
        if s.isdigit():
            n = int(s)
            return n if n > 0 else None
    return None


# 時系列順序指定を含む履歴クエリの検出 (「一番最初に」「最後に」等)。
# aux が合成する小さい limit (例: limit=1) は字句スコア最上位への
# 切り詰めであり時系列意味論を持たないため、順序指定クエリでは limit を
# ハンドラ既定値まで引き上げ、turn# 付きの全マッチターンを digest に渡す。
_ORDERED_HISTORY_QUERY_RE = re.compile(
    r"最初|最後|何番目|何回目|直近|first|last|earliest|latest",
    re.IGNORECASE,
)

# builtin._make_search_history の limit 既定と同期
_HISTORY_SEARCH_DEFAULT_LIMIT = 10

# 順序リコール質問から search_history 用の内容キーワードを抽出するための定義。
# 「この会話で一番最初に計算させた問題は何？」→「計算」。
# 除去対象の scaffolding フレーズ (self-reference / 複合順序語)。単純な文字
# クラス抽出では「一番最初」等が 1 つの漢字ランに連結するため、先にフレーズ
# 単位で除去してから内容ランを取り出す。
_ORDER_QUERY_SCAFFOLD_RE = re.compile(
    r"今までの(?:会話|やり取り)|今日の(?:追加分の)?会話|今回の(?:追加分の)?会話"
    r"|前回の会話|この会話|このやり取り|その会話"
    r"|過去の(?:会話|やり取り)|以前の会話|会話履歴"
    r"|一番最初|一番最後|何番目|何回目",
)
# 内容ラン (漢字 / カタカナ / ラテン / 数字。ひらがなの助詞・活用語尾は自然に
# 脱落する)。
#
# ひらがなを語の一部として取り込む案は採らない。送り仮名 (食べ物) と助詞・活用
# 語尾 (私が今日 / 話した / 見た映画) を辞書無しで区別できず、取り込むと
# 「が今日ハマってるって話した食べ物」のような **1 個の巨大な融合語** になる。
# 融合語は照合側の定足数を確実に落とすため、分割 (食べ物 → 食 / 物) より害が
# 大きい。語の分断は照合側の定足数を緩めることで受け止める
# (``history.history_manager._text_matches_query``)。
_ORDER_QUERY_CONTENT_RE = re.compile(
    r"[一-鿿゠-ヿ々〆a-zA-Z0-9]+",
)

#: 1 文字の内容ランは検索語として意味を持たない (良 / 久 / 泣 / 人 / 勧)。
#: 照合側が 2 文字未満を捨てるため効きもしないのに、クエリ文字列だけを膨らませる
#: (実インシデント 2026-08-16 ライブ監査 ターン5:
#: ``昨日見 映画 良 久 泣 人 勧`` の 7 語のうち 5 語が 1 文字だった)。
_ORDER_QUERY_MIN_TERM_LEN = 2
# 内容ランのうち scaffolding とみなして落とす語 (質問・順序・自己参照の骨組み)。
_ORDER_QUERY_STOPWORD_RUNS = frozenset({
    "会話", "一番", "最初", "最後", "直近", "以前", "前回", "今日", "今回", "今",
    # 時点の scaffolding。「今日」だけが登録されていたため「昨日見た映画が…」が
    # ``昨日見`` という壊れた融合語になっていた (2026-08-16 ライブ監査 ターン5)。
    "昨日", "明日", "昨夜", "今朝", "先日", "最近", "先週", "先月",
    "問題", "質問", "内容", "話題", "話", "何", "誰", "私", "貴方", "君", "僕",
    "俺", "覚", "番目", "回目", "先",
    # 明示的な履歴検索依頼の骨組み (「過去の会話で〜を探して/調べて」)
    "過去", "履歴", "探", "検索", "調", "教", "知",
    # 「もう一度」「〜させた」等の依頼骨組み (2026-08-05 追加)。
    "一度", "度", "全部", "全て", "読",
})
#: 日本語ストップワードを長い順に固定した並び (最長一致 + 決定論のため)。
#: frozenset をそのまま走査すると反復順が実行ごとに変わり、剥がれ方が
#: 非決定になる。
_ORDER_QUERY_STOPWORDS_BY_LEN: tuple[str, ...] = tuple(
    sorted(_ORDER_QUERY_STOPWORD_RUNS, key=len, reverse=True),
)


def _strip_stopword_affixes(run: str) -> str:
    """内容ランの前後に貼り付いたストップワードを剥がす (純粋関数)。

    日本語側は「漢字・カタカナ・ラテンの連続」を 1 ランとして切り出すため、
    隣接したストップワード同士が 1 つのランに融合してしまう。ラン単位の
    ストップワード照合はこの融合語を素通しし、語中で切れた無意味なキーワードが
    検索クエリに載る (2026-08-05 ライブ監査: 「今日私が最初に読ませたファイルの
    フルパスをもう一度教えてください」→ ``今日私 読 ファイル フルパス 一度教``
    で 0 件。``今日``+``私``、``一度``+``教`` がそれぞれ融合していた)。

    剥がすのは **残りが 2 文字以上、残り自体がストップワード、または剥がした
    ストップワードが 2 文字以上** の場合だけにする。無条件に剥がすと「教育」→
    「育」のように内容語を壊す (``教`` がストップワード)。

    3 つ目の条件は「2 文字以上の時点語 + 1 文字の動詞」の融合を解くためのもの
    (実インシデント 2026-08-16 ライブ監査 ターン5: 「昨日見た映画が…」が
    ``昨日見`` という実在しない語になり、照合の定足数を確実に落としていた)。
    1 文字のストップワードでは発動しないので「教育」は壊れない。
    """
    changed = True
    while changed and run:
        changed = False
        for stopword in _ORDER_QUERY_STOPWORDS_BY_LEN:
            if len(stopword) >= len(run):
                continue
            for rest in (
                run[len(stopword):] if run.startswith(stopword) else None,
                run[: -len(stopword)] if run.endswith(stopword) else None,
            ):
                if rest is None:
                    continue
                if (
                    len(rest) >= 2
                    or rest in _ORDER_QUERY_STOPWORD_RUNS
                    or len(stopword) >= 2
                ):
                    run, changed = rest, True
                    break
            if changed:
                break
    return run

# _ORDER_QUERY_SCAFFOLD_RE/_ORDER_QUERY_CONTENT_RE/_ORDER_QUERY_STOPWORD_RUNS
# の英語版。日本語版の「文字クラスで内容語/機能語を分離」は英語 (全て
# Latin script) には構造上適用できないため、単語トークン化 + ストップ
# ワードセット方式に設計変更する (_reduce_ordered_history_query 側で分岐)。
_ORDER_QUERY_SCAFFOLD_RE_EN = re.compile(
    r"\bin\s+(?:this|our)\s+conversation\b"
    r"|\bthis\s+(?:chat|conversation|thread)\b"
    r"|\bwhat\s+we\s+(?:talked|discussed)\s+about\b"
    r"|\b(?:very\s+)?first\s+(?:thing|time|question|message)\b"
    r"|\b(?:very\s+)?last\s+(?:thing|time|question|message)\b",
    re.IGNORECASE,
)
_ORDER_QUERY_CONTENT_RE_EN = re.compile(r"[A-Za-z0-9']+")
_ORDER_QUERY_STOPWORD_RUNS_EN = frozenset({
    "the", "a", "an", "in", "on", "at", "of", "to", "is", "was", "were",
    "what", "when", "where", "who", "which", "did", "do", "does",
    "i", "you", "we", "me", "my", "our", "your",
    "conversation", "chat", "thread", "talk", "talked", "discussed",
    "first", "last", "earliest", "latest", "very", "thing", "things",
    "time", "question", "message", "asked", "ask", "about",
    # 明示的な履歴検索依頼の骨組み
    "past", "previous", "history", "search", "find", "look", "tell",
    "ever", "any", "topic", "topics",
})


def _reduce_ordered_history_query(query: str) -> str:
    """履歴リコール質問から search_history 用の内容キーワードを抽出する。

    レイヤー5.5 の強制フォールバックが search_history に生クエリ全文を渡すと、
    HistoryManager の字句照合は長い疑問文を短い会話ターンにマッチできない
    (2026-07-21 ライブ検証: 「この会話で一番最初に計算させた問題は何？」が
    索引の search_text に「計算」を含むのに No results found。2026-07-27
    ライブ検証: 「過去の会話で、登山の話題をしたことはありますか？探して
    ください。」→「登山」)。self-reference / 順序語 / 検索依頼 /
    疑問 scaffolding を除去して内容キーワードを残す。
    抽出できなければ生クエリを返す (悪化させない安全側)。digest には別途 raw
    query が渡るため、順序解釈 (「一番最初」) はこの縮約で失われない。
    """
    en = is_en_locale()
    if en:
        scaffold_re, content_re, stopwords = (
            _ORDER_QUERY_SCAFFOLD_RE_EN, _ORDER_QUERY_CONTENT_RE_EN,
            _ORDER_QUERY_STOPWORD_RUNS_EN,
        )
    else:
        scaffold_re, content_re, stopwords = (
            _ORDER_QUERY_SCAFFOLD_RE, _ORDER_QUERY_CONTENT_RE,
            _ORDER_QUERY_STOPWORD_RUNS,
        )
    stripped = scaffold_re.sub(" ", query)
    terms: list[str] = []
    for run in content_re.findall(stripped):
        # 日本語はランの融合を解いてから照合する (英語は空白で切れており不要)。
        term = run if en else _strip_stopword_affixes(run)
        if not term or term.lower() in stopwords:
            continue
        # 1 文字の内容語は照合側が捨てるので、ここで落としてクエリを汚さない
        # (:data:`_ORDER_QUERY_MIN_TERM_LEN`)。英語側は元から空白区切りで
        # 1 文字語がほぼ出ないため、日本語だけに掛ける。
        if not en and len(term) < _ORDER_QUERY_MIN_TERM_LEN:
            continue
        terms.append(term)
    reduced = " ".join(terms).strip()
    return reduced if len(reduced) >= 2 else query

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


#: 「プロジェクトのルート」を指す表現。列挙対象をカレントディレクトリに解決する。
_PROJECT_ROOT_REFERENCE_RE = re.compile(
    r"(?:プロジェクト|リポジトリ|ルート|トップ(?:レベル)?|一番上)"
    r"|(?<![A-Za-z])(?:project|repo(?:sitory)?|root|top[-\s]?level)(?![A-Za-z])",
    re.IGNORECASE,
)

#: ``<名前> ディレクトリ`` / ``<名前> フォルダ`` の ``<名前>`` を取る。パス片として
#: ありうる文字だけを許し、和文は取らない (「このディレクトリ」の「この」等を
#: 対象名と誤認しないため)。
_NAMED_DIRECTORY_RE = re.compile(
    r"([A-Za-z0-9._/\\-]+)\s*(?:ディレクトリ|フォルダ)"
    r"|(?:director(?:y|ies)|folders?)\s+([A-Za-z0-9._/\\-]+)",
    re.IGNORECASE,
)


def resolve_listing_directory(query: str, root: Path) -> str | None:
    """列挙対象のディレクトリを解決する。**実在するものだけ**返す (純粋関数)。

    存在しないパスを返さないのは、捏造パスを実行しても失敗するだけで価値が無い
    ためで、``_READ_PATH_TOOLS`` の方針と同じ。解決できなければ ``None`` を返し、
    呼び出し側は後段の層 (aux 判定) へ委ねる — 当てずっぽうの引数でツールを
    撃つより、シグナルだけ立てて判断を渡すほうが安全。
    """
    for match in _NAMED_DIRECTORY_RE.finditer(query):
        name = match.group(1) or match.group(2)
        if not name:
            continue
        candidate = Path(name)
        if not candidate.is_absolute():
            candidate = root / name
        if candidate.is_dir():
            return name
    if _PROJECT_ROOT_REFERENCE_RE.search(query):
        return "."
    return None






#: 進行中の会話を指す近接リコール語。これらは「今のセッションの中」を指すので、
#: 現在セッションを除外した search_history では構造的に当たらない。
def _only_proximal_recall_keywords(query: str) -> bool:
    """履歴参照語が近接リコール語だけか (純粋関数)。

    長距離リコール語 (「以前」「最初に」「覚えて」等) が 1 つでもあれば False。
    """
    q_lower = query.lower()
    keywords = select_locale_variant(HISTORY_KEYWORDS, HISTORY_KEYWORDS_EN)
    matched = [kw for kw in keywords if kw in q_lower]
    if not matched:
        return False
    return all(kw in PROXIMAL_RECALL_KEYWORDS for kw in matched)


def _has_history_recall_keywords(query: str) -> bool:
    """明示的な履歴参照キーワード (router.HISTORY_KEYWORDS) を含むか。

    router.ComplexityClassifier._has_history_keywords と同じ判定 (小文字化
    後の部分文字列一致) を、layer 分類とは独立に tool 強制発火の判定に使う。
    """
    q_lower = query.lower()
    keywords = select_locale_variant(HISTORY_KEYWORDS, HISTORY_KEYWORDS_EN)
    return any(kw in q_lower for kw in keywords)


#: 会話に既出の対象を指す連体詞 + 名詞。「今日」「現在」のような直示語は
#: 含めない (それらは実測して答えるのが正しい)。
_ANAPHORIC_REFERENCE_RE = re.compile(
    r"(?:その|あの|例の|先ほどの|さきほどの|さっきの|前述の|上記の|くだんの)"
    r"\s*[^\s、。，．]{1,12}",
)
#: 過去に述べられた内容を尋ね直す文末形。
_RETROSPECTIVE_QUESTION_RE = re.compile(
    r"でした(?:か|っけ)|だった(?:か|っけ)|でしたよね|だっけ"
    r"|と言(?:い|っ)ました|と伝えました",
)


def asks_about_prior_conversation_entity(query: str) -> bool:
    """会話に既出の対象について尋ね直しているか (純粋関数)。

    ``_INFER_TOOL_EXEC_QUERY_RE`` は「何曜日」「日付」等の語だけで実行可能
    クエリと判定するため、会話で決めた予定を尋ね直す文まで日時取得コマンドに
    乗ってしまう。ツール結果は「唯一の事実根拠」として base に渡るので、
    現在時刻が会話の文脈を押しのけて誤答になる (実インシデント 2026-07-29
    ライブ監査: 「来週の水曜日に東京で」→「大阪の木曜に訂正」と直した直後に
    「その打ち合わせは何曜日にどこでしたか？」と尋ねたところ、
    ``datetime.now()`` が発火し、訂正前の「来週の水曜日に東京で打ち合わせが
    あります。」がそのまま返った)。

    連体詞による既出参照と、過去を尋ね直す文末形の **両方** を要求する。
    「今日は何曜日でしたっけ?」は既出参照が無いので従来どおり実測へ回る。
    """
    if not query:
        return False
    return bool(
        _ANAPHORIC_REFERENCE_RE.search(query)
        and _RETROSPECTIVE_QUESTION_RE.search(query)
    )


#: 数値リテラル抽出用。小数と整数を拾う (単位や記号は含めない)。
_NUMERIC_LITERAL_RE = re.compile(r"\d+(?:\.\d+)?")
#: 全角数字を半角に寄せる変換表 (日本語入力のクエリ対策)。
_FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９．", "0123456789.")


def _numeric_literals(text: str) -> set[str]:
    """テキスト中の数値リテラル集合を返す (全角は半角へ正規化。純粋関数)。"""
    if not text:
        return set()
    normalized = text.translate(_FULLWIDTH_DIGITS)
    return {
        m.lstrip("0") or "0" for m in _NUMERIC_LITERAL_RE.findall(normalized)
    }


def recalled_command_fits_query(
    command: str, origin_query: str, query: str,
) -> bool:
    """引き当てたコマンドを別クエリへ再生してよいかを判定する (純粋関数)。

    executable_command リコールの根拠は embedding 類似度と過去成功率だけで、
    コマンドに焼き込まれた「そのクエリ固有の値」を見ていない。日付や日数の
    ような値が本文へ埋まったコマンドを類似クエリへ再生すると、質問と無関係な
    数字を「ツールで確かめた事実」として提示してしまう
    (実インシデント 2026-07-29 ライブ監査: 「私の誕生日は3月14日です。今日から
    誕生日まであと何日ですか。」から学習した ``datetime.date(y,3,14)`` 入りの
    コマンドが、類似度 0.52 で「2026年3月15日から11月8日までは何日間ですか」へ
    再生され、無関係な ``228`` が返った)。

    コマンドと **合成元クエリ** の両方に現れる数値をクエリ由来のパラメータと
    みなし、それが今回のクエリに無ければ再生を拒否する。合成元クエリに数値が
    無いコマンド (``1024**3`` を含むディスク容量取得等) は構造上の定数しか
    持たないため、そのまま再利用できる。

    Args:
        command: 引き当てたコマンド文字列。
        origin_query: そのコマンドを合成した元のクエリ (fact.object)。
            空なら判定不能として True を返す (従来挙動を維持)。
        query: 今回のクエリ。
    """
    # 相対日付を尋ねているのに、引き当てたコマンドが日付演算を含まない場合は
    # 拒否する。数値パラメータを持たないコマンド (現在時刻の print だけ) は
    # 下の literal 判定を無条件に通ってしまい、「今日から100日後」に対して
    # 現在時刻だけが返る。差分はモデルの暗算に倒れ、当たるかどうかが運になる
    # (実インシデント 2026-08-08 ライブ監査: 修正済みの _build_datetime_command
    # ではなく、修正前に学習した現在時刻コマンドが sim=0.69 で再生された)。
    if _RELATIVE_OFFSET_RE.search(query or "") and not _DATE_ARITHMETIC_RE.search(
        command,
    ):
        return False
    if not origin_query:
        return True
    parameters = _numeric_literals(command) & _numeric_literals(origin_query)
    if not parameters:
        return True
    return parameters <= _numeric_literals(query)


@dataclass
class ToolJudgement:
    """ツール呼び出し判定結果"""
    tool_needed: bool
    tool_name: str = ""
    tool_args: dict = None  # type: ignore[assignment]
    source: str = "rule"  # "llm" | "rule" | "cartridge" | "learned"
    #: calculate の式に含まれる、対話から辿れない数値リテラル。式を捨てずに
    #: 実行したときだけ入り、回答側でその値の出所を開示させるために使う
    #: (``_suppress_ungrounded_calculate`` 参照)。
    unexplained_numbers: tuple[str, ...] = ()

    def __post_init__(self):
        if self.tool_args is None:
            self.tool_args = {}


class ToolCallJudge:
    """補助タスクによるツール呼び出し判定

    補助タスクにクエリと利用可能ツール一覧を渡し、
    適切なツールの選択を判定させる。
    判定は決定論層 (ルール / カートリッジ / 学習済みパターン / リコール) を
    順に試し、決まらなければベースモデルの文法制約 JSON 分類へ落とす。
    """

    def __init__(
        self,
        prompt_manager=None,
        config: dict | None = None,
        cartridge_manager: CartridgeManager | None = None,
        learned_patterns: LearnedPatternStore | None = None,
        debug_logger: "DebugLogger | None" = None,
        mem_view: "MemFactView | None" = None,
        embedder: "EmbeddingBackend | None" = None,
        profile_id: str = "default",
        llm_client=None,
    ):
        """
        Args:
            prompt_manager: AuxPromptManager インスタンス（None でデフォルトプロンプト使用）
            config: config.yaml 全体の dict
            cartridge_manager: CartridgeManager インスタンス（None でカートリッジ hints 無効）
            learned_patterns: LearnedPatternStore インスタンス（None で学習済みパターン無効）
            mem_view: MemFactView インスタンス（None で URL リコール無効）。
                URL なしの fetch 意図クエリで過去 ``mem.world.url.*`` を
                引き当てるために使う。
            embedder: EmbeddingBackend インスタンス（None で URL リコール無効）。
                ユーザクエリの embedding を生成して類似 URL fact を引く。
            profile_id: URL fact の profile_id フィルタ。引き当て時に同一
                profile の fact のみ採用する。
                ツール呼出判定の多段フォールバック (decision_point=
                ``tool_call_decision``、chosen=``rule``/``cartridge``/
                ``learned``/``recall``/``no_tool``) を ``decision.jsonl`` に
                記録する。``evolve`` レベル限定で実発火、それ以外は no-op。
        """
        self._llm_client = llm_client
        self._prompt_manager = prompt_manager
        self._config = config or {}
        self._cartridge_manager = cartridge_manager
        self._learned_patterns = learned_patterns
        self._debug_logger = debug_logger
        self._mem_view = mem_view
        self._embedder = embedder
        self._profile_id = profile_id
        # ツールパターン学習の閾値（通常パターンより高め）
        learning_cfg = self._config.get("learning", {})
        self._tool_routing_threshold: float = learning_cfg.get(
            "tool_pattern_match_threshold", 0.4,
        )
        agent_cfg = self._config.get("agent", {})
        # ベースモデルの文法制約ツール分類 (docs/c_14 §1.3)。決定論層が
        # すべて外れたときの最終層。``response_format`` を受け付けない
        # llama-server build では初回失敗時に落として以後試さない
        # (毎ターン 4xx を踏まないため)。
        self._tool_classifier_enabled: bool = bool(
            agent_cfg.get("tool_classifier_enabled", True),
        )
        self._tool_classifier_supported: bool = True
        self._tool_classifier_max_tokens: int = int(
            agent_cfg.get("tool_classifier_max_tokens", CLASSIFY_MAX_TOKENS),
        )
        self._tool_classifier_timeout_sec: float = float(
            agent_cfg.get("tool_classifier_timeout_sec", 60.0),
        )
        # 分類器を撃つかどうかの門。正規表現 ``_query_has_tool_signal`` は
        # 実クエリ 137 件のベンチで recall 66.2% しかなく、ツールが要る
        # クエリの 3 分の 1 を落としていた。埋め込み近傍なら 98.5% (k=5)。
        # 埋め込み未準備 / 無効時は従来の正規表現へ縮退する。
        self._tool_gate: "ToolGateKNN | None" = None
        if embedder is not None and agent_cfg.get("tool_gate_knn_enabled", True):
            from backend.free.agent.tool_gate_knn import DEFAULT_K, ToolGateKNN

            self._tool_gate = ToolGateKNN(
                embedder, k=int(agent_cfg.get("tool_gate_knn_k", DEFAULT_K)),
            )
        # 直近の層0.5 リコールの診断値 (sim / min_sim / success_avg / 候補数)。
        # _log_tool_decision が decision.jsonl の context に載せる。
        self._last_recall_diag: dict[str, Any] = {}
        # 直近の judge() で「実測しようとしたが実行できなかった」か。
        # readonly 検証違反 (PowerShell 等) / mode 非対応での降格で立つ。
        # deliberative がこれを見て、測っていない値の捏造を禁じる注記を付ける。
        self._measurement_blocked: bool = False
        # 直近の judge() で「状態を変える操作を選んだが実行できなかった」か。
        # measurement_blocked (値を測れなかった) とは別物で、こちらは
        # 「やっていないことをやったと言わせない」ためのフラグ。
        self._action_blocked: bool = False
        # 直近の judge() のユーザークエリ。measurement_blocked の適用可否判定用。
        self._last_query: str = ""
        # 直近の judge() の会話履歴。コマンド合成の指示語解決に使う
        # conversation を持たないため、単一の入口である judge() で保持する)。
        self._last_conversation: list[dict] = []

    @property
    def measurement_blocked(self) -> bool:
        """直近の判定で実行可能コマンドが「棄却されて」ツールが立たなかったか。

        「そもそもツールが不要だった」(知識質問等) とは区別する。True のときは
        システムが実測を試みて失敗しているので、呼出側は測っていない値を
        断定させないためのガードを掛ける。
        """
        return self._measurement_blocked

    @property
    def action_blocked(self) -> bool:
        """直近の判定で「状態を変える操作」が選ばれたのに実行できなかったか。

        chat には書込みツールが無く (``write_file`` は create 限定)、
        ``run_command_readonly`` は書込みコマンドを正しく拒否する。その結果
        ツールが 1 つも立たないまま base に丸投げされると、**やっていない操作を
        やったと報告する** (2026-08-08 ライブ監査 ターン6)。呼出側はこれを見て
        完了報告を禁じる注記を付ける。
        """
        return self._action_blocked

    def _user_requested_measurement(self) -> bool:
        """直近クエリが実測 (環境事実の取得 / コマンド実行) を求めているか。

        層5 のコマンド合成は環境事実を尋ねていないクエリにも投機的に走るため、
        合成の棄却だけで「実測できなかった」と記録すると、測定を求めていない
        質問にまで断り書きが混入する (実インシデント 2026-08-01 ライブ監査:
        「あなたは何ができますか？」への回答に「PC の空き容量や具体的なスペックは
        測定ツールが利用できないため取得できていません」が混ざった)。

        ユーザー意図を確立していない投機的な経路だけがこれを見る。意図が
        呼出時点で確定している経路 (明示コマンド + 実行動詞 / 判定層が
        コマンドツールを選択済み) は無条件に記録してよい。

        判定材料が無い場合 (クエリ未設定の直接呼出) は従来どおり True。
        """
        if not self._last_query:
            return True
        return is_environment_fact_query(self._last_query) or bool(
            extract_command_literal(self._last_query),
        )

    def _reject_readonly(self, exec_tool: str, command: str) -> bool:
        """``_readonly_command_rejected`` に「実測が阻まれた」記録を足したもの。

        コマンドは層5 が投機的に合成したものでもあり得るため、ユーザーが実測を
        求めていないクエリでは記録しない (``_user_requested_measurement`` 参照)。
        """
        rejected = _readonly_command_rejected(exec_tool, command)
        if rejected and self._user_requested_measurement():
            self._measurement_blocked = True
        return rejected

    def _mark_blocked_if_unexecutable_command(
        self, query: str, tools_registry: ToolsRegistry, mode: str,
    ) -> None:
        """明示コマンドを撃てないまま no_tool に落ちる場合、実測失敗を記録する。

        ユーザーがバッククォートでコマンドを書いた依頼は「実行して結果を見せて」
        という明確な実測要求。chat モードの ``run_command_readonly`` は python の
        allow-list しか通さないため ``dir`` / ``git`` 等は実行できない。撃てないまま
        no_tool で base に丸投げすると、実行していないコマンドの出力を捏造する
        (実インシデント 2026-07-29 ライブ監査: ``dir E:\\tmp\\no_such_dir_zzz`` の
        実行依頼に対し「dir: ...: No such file or directory」という、Windows の
        ``dir`` が決して返さない Unix 形式のエラーを実行結果として提示した)。

        ``measurement_blocked`` が立つと ``deliberative`` が「測っていない値を
        断定しない」注記を base の文脈へ足す。
        """
        command = extract_command_literal(query)
        if not command:
            return
        if not _EXPLICIT_EXEC_VERB_RE.search(query):
            return
        exec_tool = _executable_tool_for_mode(tools_registry, mode)
        if exec_tool and not _readonly_command_rejected(exec_tool, command):
            return
        logger.info(
            "Measurement blocked: explicit command %r cannot be executed in "
            "mode=%s (exec_tool=%s)", command[:80], mode, exec_tool or "none",
        )
        self._measurement_blocked = True

    def _mark_blocked_if_unsupported_mutation(self, query: str) -> None:
        """ファイル削除依頼のまま no_tool に落ちる場合、未実行を記録する。

        削除ツールはどのモードにも存在しない (``_DELETE_INTENT_RE`` のコメント
        参照)。``action_blocked`` を立てて ``deliberative`` に「何も実行して
        いない」注記を足させ、完了の捏造を塞ぐ。
        """
        if not _DELETE_INTENT_RE.search(query):
            return
        if not _DELETE_FS_TARGET_RE.search(query):
            return
        logger.info(
            "Action blocked: file deletion requested but no tool can delete: %s",
            query[:80],
        )
        self._action_blocked = True

    @property
    def enabled(self) -> bool:
        """ツール判定が有効かどうか (``agent.tool_judge_enabled``、既定 True)。

        False にすると最終層のベースモデル分類を撃たなくなる。決定論層は
        本フラグに関係なく常に動く。
        """
        return self._config.get("agent", {}).get("tool_judge_enabled", True)

    async def judge(
        self,
        query: str,
        tools_registry: ToolsRegistry,
        mode: str = "create",
        conversation: list[dict] | None = None,
        session_id: str = "",
    ) -> ToolJudgement:
        """ツール呼び出しの要否を判定

        判定は安価な順に実行し、最初にマッチした結果を返す:
        1. 組み込みパターン照合（ルールベース）
        2. カートリッジ tool_hints 照合
        3. 補助タスク判定（LLM）

        クリエイトモードでは tool_judge_enabled が false でも
        ルールベース + カートリッジ hints 判定を実行する。

        Args:
            query: ユーザーのクエリ
            tools_registry: 利用可能なツールレジストリ
            mode: 動作モード ('chat' | 'create')
            conversation: 直近の会話履歴（判定精度向上のため）
            session_id: 現在のチャットセッション ID。search_history が選ばれ、
                かつクエリが「この会話で」等のセッション自己参照パターンに
                一致する場合に ``_maybe_scope_session_search`` が
                ``tool_args["session_id"]`` へ注入し、検索を現在セッションに
                限定する (未指定時は従来どおり cross-session 検索のまま)。

        Returns:
            ToolJudgement
        """
        self._measurement_blocked = False
        self._action_blocked = False
        self._last_query = query
        self._last_conversation = list(conversation or [])

        # 削除依頼は「どのツールが選ばれたか」と無関係に記録する。以前はこの
        # 判定が層6 (全フォールバック失敗時) にしか無く、パスを含む削除依頼は
        # 層2 の explicit_path が read 系ツールで先に確定させるため一度も走って
        # いなかった。実インシデント (2026-08-14 ライブ監査 ターン37):
        # 「E:\tmp の中身を全部削除してください。確認は不要です。」が
        # 「中身」「確認」で読取りパターンに一致し list_directory へ解決され、
        # 一覧を成功結果として受け取った base が「すべて削除しました」と報告した
        # (実ファイルは 307 件すべて無変更)。
        self._mark_blocked_if_unsupported_mutation(query)

        # 0. URL リコール先回り判定 (mode / enabled に関係なく実行)
        # ``_try_recall_url`` は決定論的 (embedding 類似度 + 過去採点平均閾値)
        # で、補助タスク同期発火やルール正規表現のような副作用がない。早期 return
        # 経路 (chat モード + tool_judge_enabled=false) で判定がスキップされる
        # と「過去 URL は SemMem にあるのに fetch されない」という不整合が起きる
        # ため、ここで先回りで引き当てる。
        url_recall_result = await self._judge_with_url_recall(
            query, tools_registry, mode=mode,
        )
        if url_recall_result is not None:
            url_recall_result = self._finalize(
                url_recall_result, tools_registry, mode, query=query,
            )
            self._log_tool_decision(url_recall_result, "url_recall_matched")
            return url_recall_result

        # 0.5. executable command リコール先回り判定 (mode / enabled 非依存)
        # 過去成功した run_command を SemMem から決定論的に引き当てる。URL
        # リコールと同様、chat early-return / create 4 層のどちらに入る前にも
        # 短絡させることで、学習済みクエリでは aux (合成 / 5 層目) を一切
        # 呼ばずにコマンドを確定できる。
        #
        # ただし URL リコールが「ユーザーが URL を書いた」という決定論的根拠を
        # 持つのに対し、command recall の根拠は類似度のみ。ツール意図のシグナルが
        # 無いクエリ (好みの表明 / 記憶想起 / 感謝) まで先回りで奪うと、会話履歴で
        # 答えられる質問が「ツール結果に含まれていません」に化ける
        # (実測 2026-07-25: 誤発火 6 件中 4 件がこの型)。層0.5 の適用は
        # ツールシグナルを持つクエリに限定し、かつ記憶想起クエリは除外する。
        # 「さっき伝えた GPU は何だった？」は 'GPU' が _TOOL_PATTERNS に載るため
        # ツールシグナル判定を通ってしまうが、答えは会話履歴にある。ここで
        # コマンドを撃つと、ツール結果が文脈を上書きして
        # 「ツール結果に GPU 型番は含まれていません」と誤答する (実測 2026-07-25。
        # 同じ会話の 1 ターン前では Radeon 890M を正しく想起できていた)。
        # create では run_command が一級のツールで、「依存を入れて」→ 学習済み
        # `pip install ...` の引き当てが本機能の主目的なのでゲートしない。
        # 注意: リコールはルール表 (_EXECUTABLE_QUERY_COMMANDS) より **先** に
        # 短絡する。したがって一度保存されたコマンドはルール表を恒久的に隠し、
        # コード側でコマンドを直しても学習済みクエリには反映されない
        # (2026-08-06 実測: 日時コマンドを astimezone() 付きへ直した後も、
        # SemMem の naive 版が sim=0.9478 で引き当たり旧形式が実行された。
        # ルール表が非該当の「一年は何日ありますか？」はリコールが外れて
        # 新コマンドが走っており、差は経路だけだった)。
        # これは aux 呼出を省くための意図的な順序 (テストで固定) なので
        # 変えていない。コマンド表を直したときは、対応する
        # ``mem.world.executable_command.*`` ファクトの purge が要る。
        recall_allowed = is_create_mode(mode) or (
            _query_has_tool_signal(query)
            and not _has_history_recall_keywords(query)
        )
        if recall_allowed:
            cmd_recall_result = await self._judge_with_executable_command_recall(
                query, tools_registry, mode=mode,
            )
        else:
            cmd_recall_result = None
        if cmd_recall_result is not None:
            cmd_recall_result = self._finalize(
                cmd_recall_result, tools_registry, mode, query=query,
            )
            self._log_tool_decision(
                cmd_recall_result, "executable_command_recall_matched",
            )
            return cmd_recall_result

        # 0.6. ハードウェア事実 (搭載 RAM) — 決定論、非シェル。
        # 他のどの層もこの質問に答えられない: spec コマンドは RAM を出力せず
        # (2026-07-27 に意図的に外した)、readonly allow-list は ctypes / wmic /
        # Get-CimInstance を全て拒否するため合成コマンドも必ず棄却される。
        # 結果として 2026-08-12 ライブ監査では「メモリ容量に関する情報は取得
        # できていません」としか返せなかった。ツール判定より前に短絡させる。
        if (
            _HARDWARE_MEMORY_QUERY_RE.search(query)
            and tools_registry.is_available("system_hardware_info", mode)
        ):
            hw_result = self._finalize(
                ToolJudgement(
                    tool_needed=True,
                    tool_name="system_hardware_info",
                    tool_args={},
                    source="rule",
                ),
                tools_registry, mode, query=query,
            )
            self._log_tool_decision(hw_result, "hardware_facts_query")
            return hw_result

        # 決定論ショートカット: 明示された算術式 / URL / パス / 実行可能
        # コマンドは「強い意図表明」なのでモデル判断を仰がずここで確定させる。
        # 決まらなければ以降の決定論層 (0.9 / 1 / 2 / 3 / 5.5) と、最後に
        # ベースモデルの文法制約分類 (5.9) へ落とす。
        # 式が書かれていてもツールが撃てないと base の暗算に落ちて誤答する
        # (実インシデント 2026-08-08 ライブ検証: 「1234 * 5678 はいくつ？」に
        # 7,006,552 と回答。正解は 7,006,652)。
        # ``_extract_arithmetic_expression`` は純粋関数で LLM を使わない。
        expression = _extract_arithmetic_expression(query)
        if expression and tools_registry.has("calculate"):
            logger.debug("Arithmetic expression detected: %s", expression)
            result = self._finalize(
                ToolJudgement(
                    tool_needed=True,
                    tool_name="calculate",
                    tool_args={"expression": expression},
                    source="rule",
                ),
                tools_registry, mode, query=query,
            )
            if result.tool_needed:
                self._log_tool_decision(result, "arithmetic_expression")
                return result

        # クエリに URL が明示的に含まれる場合は tool_judge_enabled に
        # 関係なく fetch_url を返す。ユーザが URL を書く = 「これを読んで」
        # の強い意図表明であり、LLM 判断を仰がず決定論的に拾う
        url_match = _URL_IN_QUERY_RE.search(query)
        if url_match and tools_registry.has("fetch_url"):
            logger.debug("Explicit URL detected: %s", query[:50])
            result = ToolJudgement(
                tool_needed=True,
                tool_name="fetch_url",
                tool_args={"url": url_match.group(1)},
                source="rule",
            )
            result = self._finalize(
                result, tools_registry, mode, query=query,
            )
            self._log_tool_decision(result, "explicit_url")
            return result

        # 明示パスも URL と同じ扱いにする。ユーザーがパスを書く =
        # 「これを見て」の強い意図表明で、LLM 判断を仰ぐ理由が無い。
        # ここが無いと chat では file 系が決定論で解決されず、同じ依頼が
        # ツール未発火に落ちて「存在しない」と誤答していた (実インシデント
        # 2026-08-04 ライブ監査: 「E:/tmp/a.txt の中身を見せて、あわせて
        # 文字数も教えてください。」でツールが 1 つも走らなかった)。
        # ``_infer_tool`` は読み書きの動詞が無ければ空を返すので、パスに
        # 言及しただけの文は従来どおり後続へ落ちる。
        if _PATH_OR_URL_SIGNAL_RE.search(query):
            path_tool, path_args = self._infer_tool(query, tools_registry, mode)
            if path_tool:
                logger.debug(
                    "Explicit path resolved to %s: %s", path_tool, query[:50],
                )
                result = self._finalize(
                    ToolJudgement(
                        tool_needed=True,
                        tool_name=path_tool,
                        tool_args=path_args,
                        source="rule",
                    ),
                    tools_registry, mode, query=query,
                )
                if result.tool_needed:
                    self._log_tool_decision(result, "explicit_path")
                    return result

        # 実行可能コマンドを解決する (ルール表 + 学習済みリコール)。
        # ツール名は mode から解決する (chat は run_command_readonly)。
        exec_tool = _executable_tool_for_mode(tools_registry, mode)
        if exec_tool:
            command = await self._resolve_executable_command(
                query, readonly=exec_tool == "run_command_readonly",
            )
            if command and not self._reject_readonly(exec_tool, command):
                logger.debug("Executable query detected: %s", query[:50])
                result = ToolJudgement(
                    tool_needed=True,
                    tool_name=exec_tool,
                    tool_args={"command": command},
                    source="rule",
                )
                return self._finalize(
                    result, tools_registry, mode, query=query,
                )


        # 0.9. 「同じファイルに保存し直して」型: パスは直前ターンにしか無い。
        # ルール層はパス必須、aux 層は read_file を選びがちで、書込みが
        # 一度も走らないまま「直した内容」だけを返してしまう (実測 2026-07-27:
        # 「体重を4.5kgに直して、同じファイルに保存し直して」→ read_file のみ
        # 実行され、ファイルは旧内容のまま「保存し直した」体で回答された)。
        # 会話から直近のパスを引いて write_file に確定させる。
        rewrite = _referential_rewrite_judgement(query, conversation, tools_registry)
        if rewrite is not None:
            rewrite = self._finalize(rewrite, tools_registry, mode, query=query)
            if rewrite.tool_needed:
                self._log_tool_decision(rewrite, "referential_rewrite")
                return rewrite

        # 0.95. 「そのファイルの全文を見せて」型: 0.9 の読取版。read_file が
        # 撃たれないと、記憶から再構成した内容を「ファイルの中身」として提示
        # する (実測 2026-08-09: 追記直後の「全文をそのまま見せて」で 3 行とも
        # 実ファイルと不一致。明示的に「read_file で読み直して」と言うと正しく
        # 読み「先ほどの内容は記憶に基づくものでした」と自己訂正した = ゲートが
        # 開かないだけだった)。
        ref_read = _referential_read_judgement(query, conversation, tools_registry)
        if ref_read is not None:
            ref_read = self._finalize(ref_read, tools_registry, mode, query=query)
            if ref_read.tool_needed:
                self._log_tool_decision(ref_read, "referential_read")
                return ref_read

        # 1. 組み込みパターン照合（ルールベース）
        # ツール名まで確定したときだけ層1で打ち切る。``tool_needed=True`` かつ
        # ``tool_name=""`` (汎用ツール指示) で即 return すると、実行できる
        # ツールが 1 つも無いまま deliberative に落ちる。LLM は「ツールを使った
        # 建前」で文脈だけから答えるため、ツールで確かめるべき事実を捏造する
        # (2026-07-26 ライブ検証: 「保存したファイルを読み込んで、中身をそのまま
        # 見せてください。」— パスは直前ターンにあり本文には無い — が
        # tool_name="" で確定し read_file が発火せず、実ファイルと全く異なる
        # 内容を「ファイルの中身」として提示した。同じ依頼をパス明示で出すと
        # read_file が正しく発火し実内容を返す)。
        # ツール名が空のときは後続層 (カートリッジ / 学習済み / aux) に
        # 具体化を委ねる。aux 層は会話履歴を見るため、本文に無いパスを
        # 直前ターンから補える。
        result = self._judge_with_rules(query, tools_registry, mode)
        if result.tool_needed and result.tool_name:
            await self._maybe_recall_url(result, query, mode=mode)
            self._maybe_scope_session_search(result, query, session_id)
            result = self._finalize(result, tools_registry, mode, query=query)
            if result.tool_needed:
                self._log_tool_decision(result, "rule_pattern_matched")
                return result
            # 降格 (aux の not-executable 判定 / 引数欠落 / mode 不可) は
            # 「このツールは使えない」であって「ツールは不要」ではない。ここで
            # return すると層2〜5.2 が丸ごと死に、最も救済が要るクエリだけが
            # 素の base 暗算に落ちる (2026-07-28 ライブ検証: 「その距離を時速
            # 12キロで走ると何時間何分かかりますか。」が run_command_readonly
            # として rule 一致 → aux が not-executable と判定 → 即 no_tool
            # となり、層4 aux なら calculate を選べたのに base の暗算で
            # 「約3時間30分」と答えたうえ本文末尾で「正確には3時間31分」と
            # 自己矛盾した)。後続層へ落とす。
            logger.debug(
                "Rule layer match downgraded to no_tool; falling through to "
                "later layers: %s", query[:50],
            )

        # 2. カートリッジ tool_hints 照合
        result = self._judge_with_cartridge_hints(query, tools_registry)
        if result.tool_needed:
            self._maybe_scope_session_search(result, query, session_id)
            # ``_judge_with_cartridge_hints`` は常に ``tool_args={}`` を返す
            # (カートリッジはツール名しか宣言しない) ため、引数欠落ガードは
            # 特にこの層で効く。tool_hints はカートリッジのメタデータ由来で
            # ユーザーが書けるため、この経路は実データで到達可能。
            result = self._finalize(result, tools_registry, mode, query=query)
            self._log_tool_decision(result, "cartridge_hint_matched")
            return result

        # 3. 学習済みパターン照合
        result = self._judge_with_learned_patterns(query, tools_registry, mode)
        if result.tool_needed:
            await self._maybe_recall_url(result, query, mode=mode)
            self._maybe_scope_session_search(result, query, session_id)
            result = self._finalize(result, tools_registry, mode, query=query)
            if result.tool_needed:
                self._log_tool_decision(result, "learned_pattern_matched")
                return result
            # 層1 と同じ理由で降格時は後続層へ落とす。
            logger.debug(
                "Learned layer match downgraded to no_tool; falling through to "
                "later layers: %s", query[:50],
            )

        # 5.5. 履歴参照キーワードのフォールバック強制発火 (安全網)
        # router.HISTORY_KEYWORDS 相当の明示的な recall 語 (「覚えて」
        # 「最初に」等) を含むクエリで、層4 (aux) が no_tool と判定した
        # 場合の最終防衛線。小型 aux モデルの確率的な見落としで
        # search_history が一度も呼ばれず、長距離 recall がベースモデルの
        # 幻覚に倒れる実インシデントがあった (2026-07-20:「この会話で一番
        # 最初に私が計算させた問題は何だったか覚えてますか？」で
        # search_history 未発火 → 受動 RAG (quality=medium) のみで
        # 「そんな計算はなかった」と誤って断言)。
        # search_history へ渡すクエリは常に内容キーワードへ縮約する。
        # HistoryManager の照合は字句重なりベースで、疑問文全文は短い
        # 会話ターンにマッチせず空振りする (2026-07-21 ライブ検証:
        # 「この会話で一番最初に計算させた問題は何？」/ 2026-07-27 ライブ
        # 検証:「過去の会話で、登山の話題をしたことはありますか？探して
        # ください。」が、実際には該当する会話があるのに自分の質問文を
        # 含む直近セッションだけを拾って「見当たりません」と誤答)。
        # 当初は順序リコール質問のみ縮約していたが、明示的な検索依頼でも
        # 同じ空振りが起きるため全ケースで縮約する。内容語が取れない
        # クエリでは _reduce_ordered_history_query が生クエリを返すため
        # 従来挙動のまま。順序解釈は digest 側が受け取る raw query が
        # 担うため縮約で失われない。ヒットしなくても "No results found"
        # 経由で「見つからなかった」という正直な応答に倒れるため、無言の
        # まま確信を持って幻覚するより悪化はしない。
        # skip_judgment (雑談プレフィルタ) の判定結果に関わらず適用する:
        # 元インシデントのクエリ自体が `_SELF_ACTION_PATTERNS` の
        # 「私が」(無アンカーの部分一致) に「私が計算させた」の関係節部分で
        # 誤って一致し skip_judgment=True になっていた (2026-07-20 テストで
        # 判明)。履歴参照キーワードという強いシグナルがある以上、雑談判定
        # 側の誤検出よりこちらを優先する。
        if (
            tools_registry.has("search_history")
            and _has_history_recall_keywords(query)
        ):
            search_query = _reduce_ordered_history_query(query)
            forced_result = ToolJudgement(
                tool_needed=True,
                tool_name="search_history",
                tool_args={"query": search_query, "mode": mode},
                source="rule",
            )
            self._maybe_scope_session_search(forced_result, query, session_id)
            forced_result = self._finalize(
                forced_result, tools_registry, mode, query=query,
            )
            # 「近接リコール語だけ + 現在セッション除外」の組合せは
            # _finalize の proximal_recall_excluded_session ガードが
            # no_tool へ降格させる (以前はここにインライン実装されており、
            # aux 経路には掛かっていなかった)。
            if forced_result.tool_needed:
                self._log_tool_decision(
                    forced_result, "history_keyword_forced_fallback",
                )
                return forced_result

        # 5.9. ベースモデルの文法制約ツール分類 (docs/c_14 §1.3)。
        # 「ツールが要るのに撃たれない」穴を埋める最後の層。決定論層が
        # すべて外れてから実行する (決定論のシグナルの方がモデル判断より
        # 信頼できる)。内部の決定論プリゲート (``_gate_allows``) が、ツール
        # シグナルの無い雑談で推論を 1 往復も増やさないようにする。
        classified = (
            await self._judge_with_tool_classifier(
                query, tools_registry, mode, conversation, session_id,
            )
            if self.enabled else None
        )
        if classified is not None:
            self._log_tool_decision(classified, "tool_classifier")
            return classified

        # 6. 全フォールバック失敗時の no_tool 結末を記録
        # (削除依頼の記録は judge() 冒頭で済ませてある)
        self._mark_blocked_if_unexecutable_command(
            query, tools_registry, mode,
        )
        no_tool_result = ToolJudgement(tool_needed=False, source="rule")
        self._log_tool_decision(
            no_tool_result, "no_match_in_any_layer",
        )
        return no_tool_result

    _NATIVE_JUDGE_SYSTEM = (
        "あなたはツール選択器です。ユーザーの発言に答えるのに必要なツールを"
        "1つだけ選び、JSON で返します。回答文の作成は後続の工程が担当します。\n"
        "tool には下記のいずれかの名前を入れ、arg にはそのツールの主引数"
        "(式・クエリ・コマンド・パス等) を入れます。ツールが不要なら"
        'tool="none"、arg="" を返します。\n\n利用可能なツール:\n'
    )
    _NATIVE_JUDGE_SYSTEM_EN = (
        "You are a tool selector. Pick at most one tool needed to answer the "
        "user's message and return it as JSON. Writing the reply itself is "
        "handled by a later stage.\n"
        "Put the tool name in `tool` and its primary argument (expression, "
        "query, command, path, ...) in `arg`. If no tool is needed, return "
        'tool="none" with arg="".\n\nAvailable tools:\n'
    )

    async def _gate_allows(
        self, query: str, conversation: list[dict] | None,
    ) -> bool:
        """分類器を撃つかどうかの門。

        埋め込み kNN が準備できていればそちらを使い、``None`` (判定不能) や
        未準備なら従来の正規表現ゲートへ縮退する。**誤って閉じない**方向に
        倒すのは、取りこぼしのコスト (誤答) が無駄撃ちのコスト (分類器 1 回)
        より高いため。

        正規表現側は数値計算の判定だけ直近の会話も見る (被演算子の片方が前
        ターンにしか無い言い回しがあるため。``looks_like_numeric_question``)。
        """
        gate = self._tool_gate
        if gate is not None:
            verdict = await gate.needs_tool(query)
            if verdict is not None:
                return verdict
        return _query_has_tool_signal(query, _recent_dialogue_text(conversation))

    async def warmup_tool_gate(self) -> bool:
        """ツール要否ゲートの exemplar を埋め込む (起動後の背景タスク用)。"""
        if self._tool_gate is None:
            return False
        return await self._tool_gate.warmup()

    async def _judge_with_tool_classifier(
        self,
        query: str,
        tools_registry: ToolsRegistry,
        mode: str,
        conversation: list[dict] | None = None,
        session_id: str = "",
    ) -> "ToolJudgement | None":
        """ベースモデルに文法制約 JSON でツールを選ばせる (docs/c_14 §1.3)。

        決定論層と層4 の補助タスク判定がいずれも結論を出さなかったときに走る
        最終層 (層5.9)。「ツールが要るのに撃たれない」穴を埋める。

        選ばせ方は OAI ``tools`` ではなく ``response_format`` (json_schema) の
        enum 分類。実測 (2026-08-12, Qwen3.5-27B / gemma-4-12b) では ``tools`` は
        200 で受理されてもモデルが tool_call を出さずに本文を書き始め、
        ``max_tokens`` を使い切って 15.6〜60.2 秒を捨てる (``tool_choice:
        "required"`` でも 6 件中 3 件で無視された)。json_schema は llama-server
        側の GBNF 制約なので必ず従い、出力トークン数の上限が読める。

        コスト対策として、呼ぶ前に決定論プリゲート ``_query_has_tool_signal``
        を通す。ツールシグナルの無い雑談で推論を 1 往復増やさないため。
        なお本ゲートは再現率が低く (実測 20 ケースでツールが要る 14 件中 6 件を
        遮断)、単位換算のような「ツールシグナルの無い算術」を落とす。これは
        埋め込み kNN の一次振り分けで置き換える予定 (別作業)。

        判定結果は ``_finalize(aux_guards=True)`` を通す。分類器の引数も
        モデルが自由生成したものであり、LLM 判定と同じグラウンディングの
        ガードが要るため。ただし ``hidden_tools_offered=True`` を渡して隠し
        ツール抑止だけは外す — ``build_classifier_schema`` が hidden も enum に
        載せている以上、選ばれた名前は hallucination ではない。

        Returns:
            ツールが選ばれた場合 ``ToolJudgement``。それ以外は ``None``。
        """
        if not (self._tool_classifier_enabled and self._tool_classifier_supported):
            return None
        client = self._llm_client
        if client is None or not hasattr(client, "generate_constrained"):
            return None
        # プリゲート: ツールが要らないターンでは 1 往復も増やさない。
        if not await self._gate_allows(query, conversation):
            return None

        schema = build_classifier_schema(tools_registry, mode)
        if schema is None:
            return None

        messages = _recent_dialogue_messages(conversation)
        messages.append({"role": "user", "content": query})
        # 役割を宣言しないと、モデルは判定ではなく **回答本文** を書き始める
        # (詳細は _NATIVE_JUDGE_SYSTEM のコメント)。文法制約は形式を保証するが
        # 「何を選ぶか」の質は役割宣言に依存する。
        messages.insert(0, {
            "role": "system",
            "content": select_locale_variant(
                self._NATIVE_JUDGE_SYSTEM, self._NATIVE_JUDGE_SYSTEM_EN,
            ) + build_tool_menu(tools_registry, mode),
        })

        try:
            content = await client.generate_constrained(
                messages,
                response_format=schema,
                max_tokens=self._tool_classifier_max_tokens,
                timeout=self._tool_classifier_timeout_sec,
            )
        except httpx.HTTPStatusError as exc:
            # ``response_format`` 非対応の build。リトライしても回復しないので
            # 以後この経路を使わない (毎ターン 4xx を踏まない)。
            if exc.response is not None and 400 <= exc.response.status_code < 500:
                self._tool_classifier_supported = False
                logger.warning(
                    "Grammar tool classifier disabled for this process: "
                    "llama-server rejected the response_format payload (HTTP %s)",
                    exc.response.status_code,
                )
            else:
                logger.info("grammar tool classifier failed: %s", exc)
            return None
        except Exception as exc:
            logger.info("grammar tool classifier failed: %s", exc)
            return None

        parsed = parse_classifier_response(content, tools_registry, mode)
        if parsed is None:
            return None
        tool_name, tool_args = parsed

        result = ToolJudgement(
            tool_needed=True,
            tool_name=tool_name,
            tool_args=tool_args,
            source="classifier",
        )
        # 自由生成の search_history はセッションスコープの後処理が要る
        # (自己参照は現在セッションへ限定 / 順序リコールは横断へ拡張)。
        self._maybe_scope_session_search(result, query, session_id)
        self._maybe_expand_ordered_history_search(result, query)

        result = self._finalize(
            result,
            tools_registry, mode, query=query,
            conversation=conversation, aux_guards=True,
            hidden_tools_offered=True,
        )
        if not result.tool_needed:
            return None

        # readonly 検証は ``_finalize`` ではなく **各層の出口** で適用する設計
        # (層1/5/5.5 と同じ)。``_finalize`` は mode 外の兄弟ツールへ載せ替える
        # ため (run_command → run_command_readonly)、載せ替え **後** の名前と
        # コマンドで判定しないと、破壊的コマンドが chat の読み取り専用ツールへ
        # そのまま乗ってしまう。
        command = (result.tool_args or {}).get("command")
        if isinstance(command, str) and command and self._reject_readonly(
            result.tool_name, command,
        ):
            logger.info(
                "Native tool call rejected: %s is not read-only (%s)",
                result.tool_name, command[:80],
            )
            # 「撃てなかった」ことを呼出側へ伝える。黙って no_tool に落とすと
            # base が完了報告や測定値を捏造する。
            # ただし *何を* 撃ち損ねたかで注記が変わる:
            #   - 状態を変える試み → ``_UNPERFORMED_ACTION_GUIDANCE``
            #   - 検査 (読み取り) の試み → ``_UNMEASURED_FACT_GUIDANCE``
            # 一律に action 扱いすると、``test -f`` のような読み取りにまで
            # 「状態を変える操作を実行していない」が付き、base が「確認する
            # ツールが利用できない」と誤った説明をする (2026-08-15 ターン12)。
            # 本経路は判定層がコマンドツールを選択済み = 意図が確定しているため、
            # ``_user_requested_measurement`` を待たず measurement を記録してよい
            # (``_reject_readonly`` の docstring 参照)。
            if _command_is_readonly_inspection(command):
                self._measurement_blocked = True
            else:
                self._action_blocked = True
            return None

        logger.info(
            "Tool classifier selected: %s(%s) (query=%s)",
            result.tool_name, result.tool_args, query[:50],
        )
        return result

    def _finalize(
        self,
        result: "ToolJudgement",
        tools_registry: ToolsRegistry,
        mode: str,
        *,
        query: str = "",
        conversation: list[dict] | None = None,
        aux_guards: bool = False,
        hidden_tools_offered: bool = False,
    ) -> "ToolJudgement":
        """``judge()`` の全 exit が通る唯一の後処理 funnel.

        各判定層 (rule / cartridge / learned / aux / 各種リコール・
        フォールバック) は、確定した ``ToolJudgement`` を必ず本メソッドへ通して
        から返す。**新しい抑止を足すときの編集箇所を 1 つに保つ**ことが目的。

        以前は同じガード列が層ごとに手で書き写されており、層によって適用される
        ガードの部分集合が食い違っていた (rule / learned は 4 つ、cartridge は
        2 つ、リコール系は ``_validate_tool_availability`` のみ)。抑止を足した
        当時のインシデント経路にだけガードが付き、同じ穴を持つ他の層は素通しの
        まま残る、という抜けが実際に起きていた。

        ガードは全て純粋な ``ToolJudgement -> ToolJudgement`` で、対象ツール名が
        一致しなければ何もしない。したがって「そのツールを返さない層」に適用
        しても no-op であり、順序付きリストとして一括適用して安全。

        Args:
            aux_guards: aux (層4) 専用ガードも適用するか。
                ``_suppress_hidden_tool_from_aux`` は「プロンプトのツール一覧に
                出ない名前を aux が返すのは hallucination」という前提の防衛で、
                コード側がツール名を注入する経路 (chat の
                ``run_command_readonly`` 等) に掛けると正当な判定を潰すため、
                aux 経路でのみ有効化する。grounding 系 2 つも aux の
                free-form args 向けなので同様に限定する。
            hidden_tools_offered: モデルに提示したツール一覧に hidden ツールを
                **含めた**か。上記 hallucination 前提が成り立つのは「提示して
                いない名前が返ってきた」ときだけで、文法制約ツール分類は
                ``build_classifier_schema`` が hidden も enum に載せる (hidden は
                「プロンプト一覧に出さない」印であって「使わせない」印ではない)。
                提示した上で選ばれた名前を hallucination として潰すと、chat で
                唯一の実行系ツール ``run_command_readonly`` が分類器経路から
                恒久的に到達不能になる (実インシデント 2026-08-08 ライブ監査
                ターン13: ファイル追記が 1 度も実行されないまま「追記しました。
                行数は5行です」と捏造。実ファイルは 1 行のまま無変更だった)。
        """
        guards: list[tuple[str, Callable[[ToolJudgement], ToolJudgement]]] = [
            ("unfetchable_fetch_url", self._suppress_unfetchable_fetch_url),
            ("commandless_run_command", self._suppress_commandless_run_command),
            ("expressionless_calculate", self._suppress_expressionless_calculate),
            # 深さ絞りは抑止ではなく、依頼文から決まる引数の確定なので層を
            # 問わず安全 (対象ツール名が違えば no-op)。aux 限定にしていた
            # ため rule 層の list_directory が素通りし、同じ依頼でも層が変わる
            # と再発した (実インシデント 2026-08-04: source=aux では効き、
            # source=rule では既定 3 階層のまま 5,523 字が切り詰められた)。
            (
                "immediate_children_depth",
                lambda r: self._scope_list_directory_depth(r, query),
            ),
            # 深さ絞りと同じ「依頼文から決まる引数の確定」。決定論層は
            # _extract_head_line_count を見るが文法制約分類器は見ないため、
            # 分類器で確定した瞬間に「先頭 N 行」の指定が消えていた。
            (
                "read_file_line_range",
                lambda r: self._scope_read_file_line_range(r, query),
            ),
            (
                "proximal_recall_excluded_session",
                lambda r: self._suppress_proximal_recall_cross_session(r, query),
            ),
        ]
        if aux_guards:
            guards += [
                (
                    "truncated_text_operand",
                    lambda r: self._restore_truncated_text_operand(
                        r, conversation,
                    ),
                ),
                (
                    "ungrounded_calculate",
                    lambda r: self._suppress_ungrounded_calculate(
                        r, query, conversation,
                    ),
                ),
                (
                    "ungrounded_read_path",
                    lambda r: self._suppress_ungrounded_read_path(
                        r, query, conversation,
                    ),
                ),
            ]
            if not hidden_tools_offered:
                guards.append(
                    (
                        "hidden_tool_from_aux",
                        lambda r: self._suppress_hidden_tool_from_aux(
                            r, tools_registry,
                        ),
                    ),
                )
        # 可用性チェックは常に最後。mode 制約で撃てなかった場合に
        # ``_measurement_blocked`` を立てるため、引数欠落で先に降格した
        # ケースと区別できる位置に置く必要がある。
        guards.append(
            (
                "tool_availability",
                lambda r: self._validate_tool_availability(
                    r, tools_registry, mode,
                ),
            ),
        )

        for name, guard in guards:
            was_needed = result.tool_needed
            result = guard(result)
            if was_needed and not result.tool_needed:
                # 以降のガードは no_tool を素通しするだけなので打ち切る。
                # どのガードが降格させたかは切り分けに効くので残す。
                logger.debug(
                    "Judge guard %s downgraded the judgement to no_tool", name,
                )
                break
        return result

    def _suppress_proximal_recall_cross_session(
        self, result: "ToolJudgement", query: str,
    ) -> "ToolJudgement":
        """近接リコール語だけを根拠に現在セッションを除外した検索を撃たせない。

        「さっき」「先ほど」等が指しているのは**進行中の会話**であり、それを
        ``exclude_session_id`` で除外した検索先に答えは無い。結果は 2 通りしか
        なく、どちらも有害:

        - 0 件 → 10 秒前後を捨てたうえ「該当なし」が根拠として base に渡り、
          進行中の会話の内容まで否定させる (2026-07-28 ライブ検証)
        - 別セッションがヒット → その内容が「さっきの話」として提示される
          (2026-08-05 ライブ監査: 「さっき E:\\tmp\\...txt に書き込んで
          もらったはず」に対し、**別セッション**の
          ``E:\\tmp\\監査メモ.txt に「検証コードはアオサギ42」`` を今回の
          会話で依頼されたファイル操作として列挙した)

        このガードは元々 ``judge()`` の層5.5 (history_keyword_forced_fallback)
        にインラインで書かれており、**aux 経路 (層4) には掛かっていなかった**。
        上記 2026-08-05 の実インシデントはまさに ``source=aux`` の判定で
        起きている。同責務の抑止は層ごとに書き写さず ``_finalize`` の funnel
        へ集約する (funnel の docstring 参照)。

        ``query`` 未指定 (既定 "") の呼出では ``_only_proximal_recall_keywords``
        が False を返すため安全に no-op。
        """
        if result.tool_name != "search_history" or not result.tool_needed:
            return result
        if not (result.tool_args or {}).get("exclude_session_id"):
            return result
        if not _only_proximal_recall_keywords(query):
            return result
        logger.debug(
            "Suppressing search_history: proximal recall word refers to the "
            "ongoing session, which is excluded from the search: %s", query[:50],
        )
        return ToolJudgement(tool_needed=False, source=result.source)

    def _validate_tool_availability(
        self, result: "ToolJudgement", tools_registry: ToolsRegistry, mode: str,
    ) -> "ToolJudgement":
        """tool_name が実在し、かつ現在の mode で利用可能かを最終チェックする。

        ``_judge_with_rules`` / ``_judge_with_learned_patterns`` 等の各判定層は
        ``tools_registry.has()`` (存在チェックのみ、mode 非考慮) で判定するため、
        chat モードで ``modes=["create"]`` のツール (例: run_command) が
        ``tool_needed=True`` のまま返り得る。実行自体は
        ``deliberative._execute_tool`` のモードゲートで阻止されるが、判定結果が
        True のまま turn_outcome=failed → 次ターンの訂正誤検出 → 無関係語の
        correction 誤学習、というカスケードを引き起こす (2026-07-18 の会話
        ログで実際に発生・確認済み)。

        また ``_json_to_judgement`` は ``tool == "no_tool"`` の完全一致でのみ
        no-tool と判定するため、tool_judgment 応答が max_tokens 到達で
        ``{"tool": "no_`` のように途中切断され ``json_repair`` が
        ``tool_name="no_"`` へ復元した場合、存在しないツール名のまま
        ``tool_needed=True`` で返ってしまう (2026-07-18 実インシデントで確認)。
        レジストリに存在しないツール名も判定確定の最終防衛としてここで
        no_tool に倒す。
        """
        if not result.tool_needed or not result.tool_name:
            return result
        tool_def = tools_registry.get(result.tool_name)
        if tool_def is None:
            logger.info(
                "Tool %s not found in registry (truncated JSON or "
                "hallucinated name?); downgrading to no_tool before "
                "returning judgement",
                result.tool_name,
            )
            if result.tool_name in _STATE_CHANGING_TOOL_NAMES:
                # 未登録のまま _STATE_CHANGING_TOOL_NAMES に載っている名前
                # (delete_file / apply_patch) は、この分岐が mode ゲートより
                # 先に立つため下の action_blocked に到達しない。状態を変える
                # 意図が選ばれた事実は同じなので、ここでも記録する
                # (2026-08-12 ライブ監査で削除完了の捏造を確認)。
                self._action_blocked = True
            return ToolJudgement(tool_needed=False, source=result.source)
        if mode not in tool_def.modes:
            remapped = self._remap_to_mode_sibling(
                result, tools_registry, mode, tool_def,
            )
            if remapped is not None:
                return remapped
            logger.info(
                "Tool %s not available in mode=%s (allowed: %s); "
                "downgrading to no_tool before returning judgement",
                result.tool_name, mode, tool_def.modes,
            )
            if result.tool_name in _COMMAND_TOOL_NAMES:
                # 実測しようとして mode 制約で撃てなかったケース。
                # 「ツール不要」と区別して記録する (measurement_blocked 参照)。
                self._measurement_blocked = True
            elif result.tool_name in _STATE_CHANGING_TOOL_NAMES:
                # 状態を変える操作を選んだのに mode 制約で撃てなかった。
                # 黙って no_tool に落とすと base が完了報告を捏造する
                # (chat の ``write_file`` は create 限定で、書込みは
                # meta_cognitive 経路が担う)。実インシデント 2026-08-09
                # ライブ監査: 裸のファイル名を指した追記依頼がツール 0 回のまま
                # 「E:\tmp\inventory_notes.txt の末尾に追記しました」と
                # **フルパスまで補って** 報告され、実ファイルは無変更だった。
                self._action_blocked = True
            return ToolJudgement(tool_needed=False, source=result.source)
        return result

    def _scope_list_directory_depth(
        self, result: "ToolJudgement", query: str,
    ) -> "ToolJudgement":
        """「直下だけ」の一覧依頼で ``list_directory`` を 1 階層に絞る。

        既定の 3 階層ツリーを返すと、受け取ったモデルがインデントを読み違えて
        入れ子の項目を直下の項目として並べる (実インシデント 2026-08-01 再検証:
        「直下にあるファイルとフォルダを一覧して」に対し backend/ の下の
        develop/ api/ tests/ を直下として列挙した)。深さは依頼文から決まる
        決定論的な値なので、モデルの転記に委ねず code 側で確定させる。
        """
        if result.tool_name != "list_directory" or not query:
            return result
        if not _IMMEDIATE_CHILDREN_RE.search(query):
            return result
        if _RECURSIVE_LISTING_RE.search(query):
            return result
        args = dict(result.tool_args or {})
        if _coerce_positive_int(args.get("max_depth")) == 1:
            return result
        args["max_depth"] = 1
        logger.info(
            "list_directory scoped to immediate children for query: %s",
            query[:60],
        )
        return ToolJudgement(
            tool_needed=True,
            tool_name=result.tool_name,
            tool_args=args,
            source=result.source,
        )

    def _scope_read_file_line_range(
        self, result: "ToolJudgement", query: str,
    ) -> "ToolJudgement":
        """「先頭 N 行だけ」の読取依頼で ``read_file`` を範囲指定に絞る。

        ``list_directory`` の深さ絞り (:meth:`_scope_list_directory_depth`) と
        同じく、依頼文から決まる**引数の確定**であって抑止ではない。したがって
        層を問わず適用してよい (ツール名が違えば no-op)。

        層ごとに書き写していたため実際に食い違っていた: ``_infer_tool`` と
        ``_referential_read_judgement`` は ``_extract_head_line_count`` を見るが、
        **文法制約ツール分類 (層5.9) は見ない**。分類器は ``file_path`` しか
        埋めないため、そこで確定した瞬間に範囲指定が消える。

        実インシデント (2026-08-16 再測定ターン 14):「そのファイルの先頭2行だけ
        見せてください。」→ ``Tool classifier selected: read_file({'file_path': ...})``
        で全文 3,405 文字が返り、2 行の依頼に対して全文が渡っていた。同じ意味の
        「全文は長すぎます。そのファイルの先頭5行だけをそのまま見せてください。」は
        決定論層が拾って ``showing lines 1-5`` になる。**言い回しで挙動が割れていた**。

        既に範囲が入っている判定 (決定論層が確定させた場合) は触らない。
        """
        if result.tool_name != "read_file" or not query:
            return result
        args = dict(result.tool_args or {})
        if args.get("start_line") is not None or args.get("end_line") is not None:
            return result
        head = _extract_head_line_count(query)
        if head is None:
            return result
        args["start_line"] = 1
        args["end_line"] = head
        logger.info(
            "read_file scoped to head %d lines for query: %s", head, query[:60],
        )
        return ToolJudgement(
            tool_needed=True,
            tool_name=result.tool_name,
            tool_args=args,
            source=result.source,
        )

    def _restore_truncated_text_operand(
        self, result: "ToolJudgement", conversation: list[dict] | None,
    ) -> "ToolJudgement":
        """text 引数へ転記された「切り詰め済み会話」の断片を全文へ復元する。

        判定用プロンプトは会話を 1 メッセージ ``_JUDGE_CONTEXT_CHARS`` 文字で
        切って aux に見せる。summarize / translate のように**処理対象の本文
        そのものを引数に取る**ツールでは、aux はその切り詰められた断片しか
        転記できず、後段は断片だけを処理してしまう (実インシデント 2026-08-01
        ライブ監査: 4 ユニット 530 文字の四季の文章を「1 行に要約して」と頼んだ
        ところ、先頭 100 文字 = 春の節だけが要約された)。

        引数が会話中メッセージの真の接頭辞になっている場合、それは切り詰めの
        産物であって「その部分だけを対象にする」という意図ではない。元メッセージ
        全文へ差し替える。引数が全文と一致していれば何もしない。

        文字数上限を引き上げる対処では、上限を超える長さで同じ欠落が再発する。
        転記させず code 側で解決するのが構造的な解。
        """
        if not result.tool_needed or result.tool_name not in _TEXT_OPERAND_TOOLS:
            return result
        excerpt = (result.tool_args or {}).get("text")
        if not isinstance(excerpt, str) or not excerpt.strip():
            return result
        excerpt = excerpt.strip()
        for msg in reversed(conversation or []):
            content = msg.get("content")
            if not isinstance(content, str):
                continue
            full = content.strip()
            if len(full) > len(excerpt) and full.startswith(excerpt):
                logger.info(
                    "Restored truncated text operand for %s: %d -> %d chars",
                    result.tool_name, len(excerpt), len(full),
                )
                args = dict(result.tool_args or {})
                args["text"] = full
                return ToolJudgement(
                    tool_needed=True,
                    tool_name=result.tool_name,
                    tool_args=args,
                    source=result.source,
                )
        return result

    def _remap_to_mode_sibling(
        self,
        result: "ToolJudgement",
        tools_registry: ToolsRegistry,
        mode: str,
        tool_def: ToolDefinition,
    ) -> "ToolJudgement | None":
        """mode 外のツール名を、同じ能力を持つ mode 内の兄弟ツールへ載せ替える。

        aux へ渡すカタログは mode で絞ってあるが、モデルは学習事前分布から
        カタログ外の兄弟名を返すことがある (実インシデント 2026-08-01 ライブ監査:
        chat モードで ``run_command`` を返し、ガードが no_tool へ落として
        「ツールを実行できなかった」という曖昧な断りだけが残った。実際には
        ``run_command_readonly`` が同じ ``command`` 引数で撃てた)。

        判定そのものは正しいのに名前だけが外れている場合に意図を捨てないための
        載せ替え。引数スキーマが被覆関係にある (兄弟の必須引数をすべて満たせる)
        ときだけ行い、満たせなければ従来どおり no_tool へ落とす。権限は兄弟側の
        ツールが自前で検証するため、ここで緩むことはない。

        Returns:
            載せ替えた判定。該当が無ければ None (純粋な判定、副作用なし)。
        """
        for sibling in _MODE_CAPABILITY_SIBLINGS.get(result.tool_name, ()):
            if not tools_registry.is_available(sibling, mode):
                continue
            supplied = set(result.tool_args or {})
            if not tools_registry.required_params(sibling) <= supplied:
                continue
            logger.info(
                "Tool %s not available in mode=%s (allowed: %s); remapping to "
                "same-capability sibling %s",
                result.tool_name, mode, tool_def.modes, sibling,
            )
            return ToolJudgement(
                tool_needed=True,
                tool_name=sibling,
                tool_args=dict(result.tool_args or {}),
                source=result.source,
            )
        return None

    def _suppress_unfetchable_fetch_url(
        self, result: "ToolJudgement",
    ) -> "ToolJudgement":
        """url 引数を補完できない ``fetch_url`` を ``tool_needed=False`` に格下げ.

        URL リコール (``_maybe_recall_url`` / ``_judge_with_url_recall``) でも
        補完できず、LLM 判定でも url が validate されなかったケースを
        ``deliberative._execute_tool`` の手前で抑制する。これにより:

        - "fetch_url() requires args but none provided, skipping" の警告と
          UI 上の空ステップ表示を防ぐ
        - LLM が事前知識のみで応答する経路にクリーンに落ちる

        ``fetch_url`` 以外のツール / url が補完済みの fetch_url はそのまま返す。
        """
        if not result.tool_needed or result.tool_name != "fetch_url":
            return result
        args = result.tool_args or {}
        if args.get("url"):
            return result
        logger.info(
            "Suppressing fetch_url with no URL argument (no recall hit, "
            "no in-query URL); LLM will respond from prior knowledge",
        )
        return ToolJudgement(
            tool_needed=False,
            tool_name="",
            tool_args={},
            source=result.source,
        )

    def _suppress_commandless_run_command(
        self, result: "ToolJudgement",
    ) -> "ToolJudgement":
        """command 引数を補完できない ``run_command`` を ``tool_needed=False`` に格下げ.

        ``_suppress_unfetchable_fetch_url`` と対称。rule / learned 層は字句
        マッチだけで ``run_command`` を選ぶことがあり (learned 層は
        ``_infer_tool`` が推定できない場合の run_command フォールバックを持つ)、
        ルール層のコマンド解決でも command が
        埋まらなかった場合、実行段階で "requires args but none provided" と
        空振りするだけの判定が残る。実インシデント (2026-07-20 ライブ検証):
        学習済み tool_routing パターン「説明」(w=0.630) が知識質問
        「〜を説明して」にマッチし、create モードで引数なし run_command が
        返り得た (chat モードは run_command の mode ゲートで偶然無害化されて
        いた)。ここで no_tool に倒し、クエリを通常の LLM 応答パスに落とす。
        """
        if not result.tool_needed or result.tool_name not in (
            "run_command", "run_command_readonly",
        ):
            return result
        if (result.tool_args or {}).get("command"):
            return result
        logger.info(
            "Suppressing %s with no command argument (no rule "
            "inference, no aux synthesis); downgrading to no_tool",
            result.tool_name,
        )
        return ToolJudgement(
            tool_needed=False,
            tool_name="",
            tool_args={},
            source=result.source,
        )

    def _suppress_hidden_tool_from_aux(
        self, result: "ToolJudgement", tools_registry: ToolsRegistry,
    ) -> "ToolJudgement":
        """提示していない hidden ツール名が返された場合 no_tool に格下げ.

        hidden ツール (run_command_readonly 等) はプロンプトのツール一覧
        (``get_descriptions_text``) に出ないため、それを見て判定したモデルが
        その名前を返すのは定義上 hallucination。``_validate_tool_availability``
        は登録済み + mode 適合なら通してしまう (chat で modes=["chat"] の
        hidden ツールは素通り) ため、free-form 判定層の防衛としてここで弾く。
        judge のコード側注入経路 (early-return / recall / fallback /
        _infer_tool) は本メソッドを通らないため影響しない。

        **hidden ツールを提示した経路には掛けない** (``_finalize`` の
        ``hidden_tools_offered``)。文法制約ツール分類は enum に hidden も
        載せるので、そこで選ばれた名前は hallucination ではない。
        """
        if not result.tool_needed or not result.tool_name:
            return result
        tool_def = tools_registry.get(result.tool_name)
        if tool_def is None or not tool_def.hidden:
            return result
        logger.warning(
            "Suppressing hidden tool %s: it was not in the tool list shown to "
            "the model (source=%s); downgrading to no_tool",
            result.tool_name, result.source,
        )
        return ToolJudgement(tool_needed=False, source=result.source)

    def _suppress_expressionless_calculate(
        self, result: "ToolJudgement",
    ) -> "ToolJudgement":
        """expression 引数を補完できない ``calculate`` を ``tool_needed=False`` に格下げ.

        ``_suppress_commandless_run_command`` と対称。rule / learned 層の
        ``_infer_tool`` は「計算」の字句マッチだけで calculate を選ぶが式抽出
        ロジックを持たず常に空 args を返し、aux 層も free-form args のため
        ``{"tool": "calculate", "args": {}}`` があり得る。実行段の必須引数
        チェックで "requires args but none provided" と空振りするだけの判定が
        残る。実インシデント (2026-07-21 ライブ検証 ターン35):
        「フィボナッチ数列の10番目を計算して」が rule 層で ``calculate, {}``
        になり WARN + UI 空ステップ (running フレームのみで完了フレーム無し)。
        ここで no_tool に倒し、クエリを通常の LLM 応答パスに落とす
        (格下げ後の応答は LLM 知識で正解した実測あり)。
        """
        if not result.tool_needed or result.tool_name != "calculate":
            return result
        if (result.tool_args or {}).get("expression"):
            return result
        logger.info(
            "Suppressing calculate with no expression argument; "
            "downgrading to no_tool",
        )
        return ToolJudgement(
            tool_needed=False,
            tool_name="",
            tool_args={},
            source=result.source,
        )

    def _suppress_ungrounded_calculate(
        self,
        result: "ToolJudgement",
        query: str,
        conversation: list[dict] | None,
    ) -> "ToolJudgement":
        """クエリにも会話にも無い数値を含む ``calculate`` を no_tool へ格下げ.

        層5.2 (``_judge_with_calculate_fallback``) は合成式へ
        ``_synthesized_expression_grounded`` を掛けて捏造数値を弾くが、層4
        (aux ``tool_judgment``) には同じ検証が無く、素通りしていた。ツールの
        戻り値は「確かめた事実」として base に最優先で渡されるため、捏造された
        式の結果は **正しく計算された嘘** になり、素の暗算より有害になる
        (実インシデント 2026-07-29 ライブ監査: 「本当にそれで合っていますか？
        計算を見直してください。」に対し aux が ``57.8 - 4 * 1.5`` を合成。
        ``1.5`` はクエリにも会話にも無く、結果 51.8 が「正解」として提示された。
        正しくは ``57.8 - 4 * 3.4`` = 44.2)。

        格下げ後は後続層 (5.2) がグラウンディング検証付きで式を組み直す機会を
        得るため、計算そのものを諦めることにはならない。

        ホワイトリストには層5.2 の直近 4 ターンではなく **判定に渡された会話
        全体** を使う。層5.2 は式を *合成* するので狭い窓で捏造の余地を絞るのが
        正しいが、こちらは既に選ばれた式を *拒否* するだけなので、窓を狭めると
        「その距離を時速12キロで…」のような照応で数ターン前の数値を参照した
        正当な式まで巻き込む。捏造の信号 (会話のどこにも無い数値) は広い窓でも
        変わらず立つ。
        """
        if not result.tool_needed or result.tool_name != "calculate":
            return result
        expression = str((result.tool_args or {}).get("expression") or "")
        if not expression:
            return result
        context = _dialogue_text(conversation)
        unexplained = _ungrounded_numbers(expression, query, context)
        if not unexplained:
            return result
        # 格下げが正当なのは、式を組み直す層が実際に走れるときだけ。その層
        # (旧 5.2 の式合成) は撤去済みなので、格下げの落ち先は base の暗算に
        # なる。実測では暗算のほうが誤答しやすく、しかも式が見えないぶん
        # 誤りに気づけない (2026-08-08 以降のライブ監査で 4 回連続。いずれも
        # 「式は正しかったのに 1 つの数値が書かれていない」ケースで、
        # 42.195*5.5 / 3*52 / 8*5*(18-9)*2 などが棄却された)。式は残し、
        # 説明できない数値を回答側で開示させる。
        logger.info(
            "Keeping calculate with unexplained numbers %s in %r; "
            "the answer must disclose them",
            list(unexplained), expression[:80],
        )
        return replace(result, unexplained_numbers=unexplained)

    def _suppress_ungrounded_read_path(
        self,
        result: "ToolJudgement",
        query: str,
        conversation: list[dict] | None,
    ) -> "ToolJudgement":
        """クエリにも会話にも現れないパスの読み取りツールを no_tool へ格下げ.

        層4 (aux ``tool_judgment``) は「ファイルの金額を合計して」のような
        パスを含まないタスクに対しても read 系ツールを選び、``file_path`` を
        でっち上げることがある。捏造パスの読み取りは構造的に必ず失敗するか、
        最悪の場合まったく別のファイルを読むため、実行する価値が無い
        (実インシデント 2026-07-29 ライブ監査: 「合計を計算して / 中身も
        見せて」という 2 サブタスクが ``read_file(prices.txt)`` と
        ``read_file(unknown)`` になり、直前ステップで内容は取れていたのに
        「1 件のタスクを完了し、2 件が失敗しました。」だけが返った)。

        格下げすると後続層およびツールループが、パスを持たないタスクとして
        扱い直す機会を得る。書込み系は対象外 — 出力先の既定解決 (
        ``_resolve_write_path`` 等) が別途あり、パス未指定の生成依頼を
        巻き込むため。
        """
        if not result.tool_needed or result.tool_name not in _READ_PATH_TOOLS:
            return result
        args = result.tool_args or {}
        path = str(
            args.get("file_path") or args.get("path") or args.get("directory") or "",
        )
        if not path:
            return result
        haystack = _normalize_path_text(f"{query}\n{_dialogue_text(conversation)}")
        if _normalize_path_text(path) in haystack:
            return result
        # ドライブ / ディレクトリ表記の揺れを許すためベース名でも照合する。
        basename = _normalize_path_text(PurePosixPath(path.replace("\\", "/")).name)
        if basename and basename in haystack:
            return result
        logger.info(
            "Suppressing ungrounded %s path %r (absent from query and dialogue); "
            "downgrading to no_tool", result.tool_name, path[:120],
        )
        return ToolJudgement(
            tool_needed=False,
            tool_name="",
            tool_args={},
            source=result.source,
        )

    async def _judge_with_url_recall(
        self,
        query: str,
        tools_registry: ToolsRegistry,
        mode: str = "create",
    ) -> "ToolJudgement | None":
        """URL リコール単独で fetch_url 判定を返す (mode / enabled 非依存).

        条件:
          - ``fetch_url`` ツールが登録済み
          - ``mem_view`` / ``embedder`` が wired されている
          - ``_try_recall_url`` が閾値判定で URL を返す

        Returns:
            URL 引き当て成立時は ``ToolJudgement(fetch_url, {"url": ...})``、
            それ以外は ``None`` (通常の判定フローに falling-through する)。
        """
        if not tools_registry.has("fetch_url"):
            return None
        if self._mem_view is None or self._embedder is None:
            return None
        # クエリが実在するローカルファイルを明示参照している場合、URL recall の
        # 無条件短絡はスキップし後段の判定層へ落とす。操作対象が具体的な
        # ローカルファイルであるタスク (実インシデント: "Read <path>.xlsx /
        # Apply monthly borders...") が、過去の無関係な URL 記憶と埋め込み
        # 類似度だけで fetch_url へハイジャックされるのを決定論的に防ぐ。
        # 書込み先ディレクトリ指定 (url_write 正規フロー) は is_file()=False の
        # ため影響せず、後段の rule/learned 層が fetch_url を選べば
        # ``_maybe_recall_url`` の URL 補完も引き続き機能する。
        referenced = _extract_file_path(query)
        if referenced:
            try:
                if Path(referenced).is_file():
                    logger.info(
                        "URL recall: skipped (query references existing "
                        "local file %s)", referenced,
                    )
                    return None
            except (OSError, ValueError):
                pass
        # パスを明示しないローカルファイル参照 (「そのファイルの中身を読み込んで
        # 見せて」等) は _extract_file_path で拾えないため、上の実在チェックを
        # すり抜けて埋め込み類似度だけで fetch_url へ短絡していた (実インシデント
        # 2026-07-27 ライブ検証: 直前ターンで保存した note2.txt の読み出し依頼が
        # 過去セッションの example.com への fetch_url になった)。web 参照シグナル
        # が無いローカルファイル依頼は URL recall の対象外にする。
        if _query_targets_local_file_only(query):
            logger.info(
                "URL recall: skipped (local file reference without web signal): %s",
                query[:50],
            )
            return None
        recalled = await self._try_recall_url(query, mode=mode)
        if not recalled:
            return None
        logger.info(
            "URL recall: matched url=%s for query=%s", recalled, query[:50],
        )
        return ToolJudgement(
            tool_needed=True,
            tool_name="fetch_url",
            tool_args={"url": recalled},
            source="rule",
        )

    async def recall_url_judgement(
        self,
        query: str,
        tools_registry: ToolsRegistry,
        mode: str = "create",
    ) -> "ToolJudgement | None":
        """外部から URL recall のみを問い合わせる公開 API.

        Reactive レイヤの escalation 判定など、judge() フル実行前に
        「URL recall だけ」をチェックしたい呼び元向け。判定本体は
        ``_judge_with_url_recall`` を共有し、戻り値も同じ。
        """
        return await self._judge_with_url_recall(query, tools_registry, mode=mode)

    async def _maybe_recall_url(
        self, result: "ToolJudgement", query: str, mode: str = "create",
    ) -> None:
        """rule / learned 層が ``fetch_url`` を返したが URL が空の場合、
        過去質問で正しく fetch できた URL (``mem.world.url.*``) を引き当てる。

        引き当てが成立すると ``result.tool_args["url"]`` に補完して
        in-place で更新する。失敗 / 引き当てなしの場合は何もしない。
        ``mem_view`` / ``embedder`` が None (degraded) でも安全に no-op。
        """
        if result.tool_name != "fetch_url":
            return
        if result.tool_args and result.tool_args.get("url"):
            return
        recalled = await self._try_recall_url(query, mode=mode)
        if not recalled:
            return
        if result.tool_args is None:
            result.tool_args = {}
        result.tool_args["url"] = recalled
        logger.debug("URL recall: matched url=%s for query=%s", recalled, query[:50])

    def _maybe_scope_session_search(
        self, result: "ToolJudgement", query: str, session_id: str,
    ) -> None:
        """``search_history`` の現在セッションの扱いを code 側で強制する。

        クエリが「この会話で」等のセッション自己参照パターンに一致する場合は
        ``tool_args["session_id"]`` を強制注入して検索対象を現在セッションのみに
        限定する。session_id を渡さずに search_history を無条件許可すると、
        2026-07-17/18 の実インシデント (「この会話で一番面白かった？」が無関係な
        過去セッションの内容を誤って混同した) が再発するため。

        自己参照でない場合は逆に ``tool_args["exclude_session_id"]`` を注入して
        現在セッションを結果から外す。現在セッションの発言は既に会話コンテキスト
        へ全文が載っており再注入しても情報は増えないのに、セッション要約
        (= 会話冒頭の発言) が「独立した根拠」の顔で入り、後から訂正された内容を
        訂正前の値へ巻き戻す (2026-07-26 ライブ検証: 火曜→水曜と訂正した歯科の
        予約が、2 ターン後に検索結果のセッション要約経由で火曜へ戻った)。

        in-place で更新する。``session_id`` が空 (未提供) の場合は何もしない
        (呼出元が未対応でも安全に no-op)。
        """
        if result.tool_name != "search_history":
            return
        if not session_id:
            return
        if is_en_locale():
            patterns = _SELF_SESSION_REFERENCE_PATTERNS_EN
            self_reference = (
                not _SESSION_TOPIC_BREAK_LEAD_RE_EN.search(query)
                and any(p.search(query) for p in patterns)
            )
        else:
            patterns = _SELF_SESSION_REFERENCE_PATTERNS
            self_reference = any(p.search(query) for p in patterns)
        if result.tool_args is None:
            result.tool_args = {}
        if self_reference:
            result.tool_args["session_id"] = session_id
            logger.debug(
                "search_history scoped to current session "
                "(self-session reference): %s", query[:50],
            )
            return
        result.tool_args["exclude_session_id"] = session_id
        logger.debug(
            "search_history excludes current session (already in context): %s",
            query[:50],
        )

    def _maybe_expand_ordered_history_search(
        self, result: "ToolJudgement", query: str,
    ) -> None:
        """時系列順序指定クエリの ``search_history`` で小さい limit を既定値へ引き上げる.

        実インシデント (2026-07-21 ライブ検証 ターン18): 「この会話で一番最初に
        計算させた問題は?」で aux が ``query='計算', limit=1`` を合成 →
        ``HistoryManager.search_sessions`` は字句スコア降順で limit 件に切る
        ため、時系列先頭ではなく直近の計算を返し誤答した。limit が十分なら
        turn# 付きの全マッチターンが digest に渡り、元クエリ (digest の user
        prompt に含まれる) と合わせて時系列選択が機能する (同検証 ターン42
        「すべて挙げて」が limit 既定 10 で 6 件完全列挙に成功した実績)。

        判定は aux 合成後の ``args["query"]`` ではなく**ユーザー生クエリ**
        に対して行う (合成 query では「一番最初」等の順序語が消えている)。
        引き上げのみで引き下げはしない (「直近20件」等の明示的な大 limit を
        壊さない)。挿入点は aux 層のみ — limit を合成し得るのは free-form
        args の aux 層だけで、rule/learned 層の ``_infer_tool`` に
        search_history 分岐は無く、cartridge 層は空 args、層5.5 の強制発火は
        limit を設定しない (ハンドラ既定 10 が効く)。in-place で更新する。

        aux (LFM2) は json_schema grammar を強制せず型崩れ JSON を返し得る
        (limit が "1" (文字列) や 2.0 (float) 等) ため、数値相当は int へ
        正規化してから判定する。引き上げ時は int で書き戻すため、後続の
        search_history ハンドラ (limit で slice する) への型汚染も防ぐ。
        """
        if result.tool_name != "search_history":
            return
        if not _ORDERED_HISTORY_QUERY_RE.search(query):
            return
        args = result.tool_args or {}
        limit = _coerce_positive_int(args.get("limit"))
        if limit is not None and 0 < limit < _HISTORY_SEARCH_DEFAULT_LIMIT:
            args["limit"] = _HISTORY_SEARCH_DEFAULT_LIMIT
            result.tool_args = args
            logger.debug(
                "search_history limit expanded %d -> %d for ordered query: %s",
                limit, _HISTORY_SEARCH_DEFAULT_LIMIT, query[:50],
            )

    async def _try_recall_url(self, query: str, mode: str = "create") -> str | None:
        """過去質問の URL fact から類似質問の URL を返す。

        条件:
          - ``mem_view`` / ``embedder`` が両方提供されている
          - ``tools.url_recall_enabled`` が True
          - top-K 候補のうち ``world_fact`` で subject prefix が
            ``mem.world.url.`` のもの
          - 類似度 >= ``url_recall_min_score``
          - 過去採点平均 (``_extra.score_avg``) >= ``url_recall_min_record_score``
          - profile_id が一致する (異プロファイルの URL を引かない)

        Returns:
            条件を満たす最良の URL。なければ ``None``。
        """
        tools_cfg = (self._config or {}).get("tools") or {}
        if not bool(tools_cfg.get("url_recall_enabled", True)):
            return None
        if self._mem_view is None or self._embedder is None:
            return None
        if not query:
            return None
        # embed_query は LRU キャッシュ付きの単一クエリ経路。同一ターンでは
        # 検索パイプライン (run_search_pipeline) が同じ (query, mode) で既に
        # 埋め込み済みなので、ここはキャッシュヒットになり埋め込みサーバへの
        # 往復が消える。embed() を直接呼ぶとキャッシュを迂回して毎ターン
        # 二重に埋め込んでいた (2026-07-27 実測: 1 ターンあたり +0.35s)。
        try:
            embedding = await self._embedder.embed_query(query, mode=mode)
        except Exception as exc:
            logger.warning("URL recall: embed failed: %s", exc)
            return None
        if embedding is None or len(embedding) == 0:
            return None
        import numpy as _np
        q_vec = _np.asarray(embedding, dtype=_np.float32)

        top_k = int(tools_cfg.get("url_recall_topk", 5))
        min_sim = float(tools_cfg.get("url_recall_min_score", 0.7))
        min_avg = float(tools_cfg.get("url_recall_min_record_score", 0.6))
        ttl_days = int(tools_cfg.get("url_recall_ttl_days", 30))
        ttl_seconds = float(ttl_days) * 86400.0 if ttl_days > 0 else 0.0
        try:
            candidates = self._mem_view.search_by_embedding(q_vec, top_k=top_k)
        except Exception as exc:
            logger.warning("URL recall: search_by_embedding failed: %s", exc)
            return None

        # Pro 拡張: team プロファイルの URL fact も引き当て候補に含める。
        # Free build では factory が登録されていないため allowed = {self._profile_id}
        # のみで Phase 1 と等価。
        allowed_profiles: set[str] = {self._profile_id}
        try:
            from backend.edition import get_pro_handler
            factory = get_pro_handler("url_recall_resolver_factory")
            if callable(factory):
                resolver = factory(self._config)
                if resolver is not None:
                    for pid in resolver.allowed_profile_ids():
                        if pid:
                            allowed_profiles.add(pid)
        except Exception as exc:
            logger.warning("URL recall: pro resolver init failed: %s", exc)

        import time as _time
        now = _time.time()

        # recall miss の理由を後から追えるよう、最良 (= 最高 sim) の
        # ``mem.world.url.*`` 候補を記録しておく (candidates は score 降順想定)。
        best_subject: str | None = None
        best_sim: float | None = None
        best_reason = "no_url_candidate"

        for fact, sim in candidates:
            if fact.type != "world_fact":
                continue
            if not fact.subject.startswith("mem.world.url."):
                continue
            if best_subject is None:
                best_subject, best_sim, best_reason = fact.subject, sim, "sim_below_min"
            if sim < min_sim:
                # candidates は score 降順想定。閾値未満は以降全て無効。
                break
            if fact.profile_id and fact.profile_id not in allowed_profiles:
                continue
            extra = fact._extra or {}
            url = extra.get("url")
            score_avg = float(extra.get("score_avg") or 0.0)

            # 鮮度ペナルティ: TTL 超過なら score_avg を半減して閾値判定。
            # 完全 skip ではなく penalize に留めるのは、「古いが他に候補なし」
            # の状況で min_record を下げて運用すれば引けるようにするため。
            effective_score = score_avg
            if ttl_seconds > 0.0:
                last_fetched = float(extra.get("last_fetched_at") or 0.0)
                if last_fetched > 0.0:
                    age_sec = now - last_fetched
                    if age_sec > ttl_seconds:
                        effective_score = score_avg * 0.5
                        logger.warning(
                            "URL recall: TTL exceeded for %s "
                            "(age=%.0fd, ttl=%dd), penalize score %.3f -> %.3f",
                            fact.subject, age_sec / 86400.0, ttl_days,
                            score_avg, effective_score,
                        )

            if url and effective_score >= min_avg:
                # 非 ASCII を含む URL は壊れている可能性が高い (旧 URL 抽出 regex が
                # 末尾の日本語を取り込んだ残骸など)。fetch で 404 になるため引き当てない。
                if not str(url).isascii():
                    logger.warning(
                        "URL recall: skipping non-ASCII (likely malformed) url=%s",
                        url,
                    )
                    continue
                logger.info(
                    "URL recall: match sim=%.3f (min_sim=%.2f) score_avg=%.3f "
                    "(min_record=%.2f) url=%s",
                    sim, min_sim, effective_score, min_avg, url,
                )
                return str(url)
            # sim は満たしたが score_avg/TTL で落ちたケースを記録 (最初の 1 件のみ)
            if best_subject == fact.subject and best_reason == "sim_below_min":
                best_reason = "score_avg_below_min"

        # 引き当て無し: なぜ外れたかを DEBUG で可視化する (閾値チューニングの根拠)。
        if best_subject is None:
            logger.debug(
                "URL recall: no mem.world.url candidate in top-%d for query=%r",
                top_k, query[:50],
            )
        else:
            logger.debug(
                "URL recall: no match for query=%r; best candidate subject=%s "
                "sim=%.3f (min_sim=%.2f) min_record=%.2f reason=%s",
                query[:50], best_subject, best_sim or 0.0, min_sim,
                min_avg, best_reason,
            )
        return None

    async def _judge_with_executable_command_recall(
        self,
        query: str,
        tools_registry: ToolsRegistry,
        mode: str = "create",
    ) -> "ToolJudgement | None":
        """SemMem の過去成功コマンド引き当てで executable 判定を返す.

        ``_judge_with_url_recall`` と対称。条件:
          - 現在の mode で executable ツールが解決できる
            (create → run_command / chat → run_command_readonly)
          - ``mem_view`` / ``embedder`` が wired されている
          - ``_try_recall_executable_command`` が閾値判定でコマンドを返す

        引き当てが成立すれば aux 呼出 (5 層目 / chat early-return) より
        先に確定するため、学習済みクエリでは LLM コストがゼロになる。

        recall は subject の mode (`mem.world.executable_command.<mode>.*`) を
        フィルタしないため、chat では create 学習由来の任意コマンド (書込系
        等) が引き当たり得る。chat (readonly) のときは
        ``reject_readonly_violation`` で再検証し、違反コマンドは引き当てを
        捨てて通常フローへ落とす (実行段の readonly ラッパでも二重に弾かれる
        が、ここで捨てれば synth 等の後続層が正当なコマンドを合成し直せる)。

        Returns:
            引き当て成立時は ``ToolJudgement(<exec_tool>, {"command": ...})``、
            それ以外は ``None`` (通常の判定フローに falling-through する)。
        """
        exec_tool = _executable_tool_for_mode(tools_registry, mode)
        if not exec_tool:
            return None
        if self._mem_view is None or self._embedder is None:
            return None
        recalled = await self._try_recall_executable_command(query, mode=mode)
        if not recalled:
            return None
        if self._reject_readonly(exec_tool, recalled):
            return None
        logger.info(
            "Executable command recall: matched command for query=%s",
            query[:50],
        )
        return ToolJudgement(
            tool_needed=True,
            tool_name=exec_tool,
            tool_args={"command": recalled},
            # "rule" と区別する。recall 由来の実行を curator が再学習すると
            # 「誤発火 → 成功記録 → fact 延命 → また誤発火」で自己強化するため、
            # sleep 側 (executable_command_curator) がこの source を見て除外する。
            source="recall",
        )

    async def _try_recall_executable_command(
        self, query: str, mode: str = "create",
    ) -> str | None:
        """過去成功した run_command を SemMem から類似クエリで引き当てる.

        ``_try_recall_url`` と対称。条件:
          - ``mem_view`` / ``embedder`` が両方提供されている
          - ``tools.executable_command_recall_enabled`` が True
          - top-K 候補のうち ``world_fact`` で subject prefix が
            ``mem.world.executable_command.`` のもの
          - 類似度 >= ``executable_command_recall_min_score``
          - 過去成功率 (``_extra.success_avg``) >=
            ``executable_command_recall_min_record_score``
          - profile_id が一致する

        Returns:
            条件を満たす最良のコマンド文字列。なければ ``None``。
        """
        tools_cfg = (self._config or {}).get("tools") or {}
        if not bool(tools_cfg.get("executable_command_recall_enabled", True)):
            return None
        if self._mem_view is None or self._embedder is None:
            return None
        if not query:
            return None
        # ``_try_recall_url`` と同じ理由で embed_query (LRU キャッシュ経路) を使う。
        try:
            embedding = await self._embedder.embed_query(query, mode=mode)
        except Exception as exc:
            logger.warning("Executable command recall: embed failed: %s", exc)
            return None
        if embedding is None or len(embedding) == 0:
            return None
        import numpy as _np
        q_vec = _np.asarray(embedding, dtype=_np.float32)

        top_k = int(tools_cfg.get("executable_command_recall_topk", 5))
        min_sim = float(tools_cfg.get("executable_command_recall_min_score", 0.7))
        min_avg = float(
            tools_cfg.get("executable_command_recall_min_record_score", 0.6),
        )
        ttl_days = int(tools_cfg.get("executable_command_recall_ttl_days", 30))
        ttl_seconds = float(ttl_days) * 86400.0 if ttl_days > 0 else 0.0
        try:
            candidates = self._mem_view.search_by_embedding(q_vec, top_k=top_k)
        except Exception as exc:
            logger.warning(
                "Executable command recall: search_by_embedding failed: %s", exc,
            )
            return None

        # 候補プールが小さいうちは top-K も success_avg も選別として機能しないため
        # (実測 2026-07-25: executable_command fact は global に 1 件のみで、
        # 類似度ゲートだけが唯一のフィルタだった)、閾値を嵩上げして保守的に倒す。
        if len(candidates) < _RECALL_SMALL_POOL_SIZE:
            min_sim += _RECALL_SMALL_POOL_MARGIN

        import time as _time
        now = _time.time()

        best_sim = candidates[0][1] if candidates else 0.0
        self._last_recall_diag = {
            "candidates": len(candidates),
            "best_sim": round(float(best_sim), 4),
            "min_sim": round(min_sim, 4),
            "min_avg": min_avg,
        }

        for fact, sim in candidates:
            if fact.type != "world_fact":
                continue
            if not fact.subject.startswith("mem.world.executable_command."):
                continue
            if sim < min_sim:
                # candidates は score 降順想定。閾値未満は以降全て無効。
                break
            if fact.profile_id and fact.profile_id != self._profile_id:
                continue
            extra = fact._extra or {}
            command = extra.get("command")
            success_avg = float(extra.get("success_avg") or 0.0)

            # 鮮度ペナルティ: TTL 超過なら success_avg を半減して閾値判定。
            effective_score = success_avg
            if ttl_seconds > 0.0:
                last_exec = float(extra.get("last_executed_at") or 0.0)
                if last_exec > 0.0:
                    age_sec = now - last_exec
                    if age_sec > ttl_seconds:
                        effective_score = success_avg * 0.5
                        logger.warning(
                            "Executable command recall: TTL exceeded "
                            "(age=%.0fd, ttl=%dd), penalize %.3f -> %.3f",
                            age_sec / 86400.0, ttl_days,
                            success_avg, effective_score,
                        )

            if command and not recalled_command_fits_query(
                str(command), str(fact.object or ""), query,
            ):
                logger.info(
                    "Executable command recall rejected: query-specific "
                    "literals missing (sim=%.4f subject=%s origin=%s)",
                    sim, fact.subject, str(fact.object or "")[:60],
                )
                self._last_recall_diag["rejected"] = "literal_mismatch"
                continue

            if command and effective_score >= min_avg:
                # URL リコール側と同粒度の観測。embed モデルを差し替えるたびに
                # sim 分布が動くため、これが無いと閾値較正が事後検証できない。
                logger.info(
                    "Executable command recall matched: sim=%.4f min_sim=%.4f "
                    "success_avg=%.3f effective=%.3f min_record=%.3f "
                    "candidates=%d subject=%s",
                    sim, min_sim, success_avg, effective_score, min_avg,
                    len(candidates), fact.subject,
                )
                self._last_recall_diag.update({
                    "sim": round(float(sim), 4),
                    "success_avg": round(success_avg, 3),
                    "effective_score": round(effective_score, 3),
                    "subject": fact.subject,
                })
                return str(command)
        logger.debug(
            "Executable command recall miss: candidates=%d best_sim=%.4f "
            "min_sim=%.4f query=%s",
            len(candidates), float(best_sim), min_sim, query[:50],
        )
        return None

    def _log_tool_decision(
        self, result: "ToolJudgement", reason: str,
    ) -> None:
        """

        chosen は ``rule`` / ``cartridge`` / ``learned`` / ``aux`` /
        ``no_tool`` のいずれかで、4 段階フォールバックのどの層で決着したかを
        identify する。``evolve`` レベル限定で実発火、それ以外は no-op。
        """
        if self._debug_logger is None:
            return
        chosen = "no_tool" if not result.tool_needed else (result.source or "rule")
        context: dict[str, Any] = {
            "tool_needed": bool(result.tool_needed),
            "tool_name": getattr(result, "tool_name", None) or "",
        }
        command = (getattr(result, "tool_args", None) or {}).get("command")
        if command:
            context["command"] = str(command)[:120]
        # 層0.5 の採否は類似度が唯一の根拠なので、decision.jsonl だけで
        # 閾値較正を検証できるよう診断値を載せる。
        if reason.startswith("executable_command_recall") and self._last_recall_diag:
            context.update(self._last_recall_diag)
        self._debug_logger.log_decision(
            decision_point="tool_call_decision",
            chosen=chosen,
            candidates=["rule", "cartridge", "learned", "llm", "recall", "no_tool"],
            reason=reason,
            context=context,
            scope="request",
        )

    async def _resolve_executable_command(
        self, query: str, readonly: bool = False,  # noqa: ARG002 - 面の互換
    ) -> str:
        """executable command をルール表 (regex) から解決する.

        Returns:
            実行可能と判定された場合のコマンド文字列。それ以外は ``""``。
        """
        return _infer_executable_command(query)

    def _judge_with_rules(
        self,
        query: str,
        tools_registry: ToolsRegistry,
        mode: str,
    ) -> ToolJudgement:
        """ルールベースでツール呼び出しを判定（フォールバック）

        パターンマッチした場合、クエリの内容からツール名と引数を推定する。
        知識質問（RAG で処理すべき）はツール不要と判定する。
        """
        # 明示的に書かれた算術式は calculate で決定論的に評価する。
        # 知識質問判定より前に置く: 「1234 × 5678 はいくつですか？」は
        # 「〜ですか」で知識質問にマッチし、ツール無しで base の暗算に落ちて
        # 誤答するため (実インシデント 2026-07-27 ライブ検証)。
        expression = _extract_arithmetic_expression(query)
        if expression and tools_registry.has("calculate"):
            logger.debug("Rule-based: arithmetic expression detected: %s", expression)
            return ToolJudgement(
                tool_needed=True,
                tool_name="calculate",
                tool_args={"expression": expression},
                source="rule",
            )

        # ディレクトリ列挙は決定論で解決する。対象が実在するときだけ発火し、
        # 解決できなければシグナルだけ立てて後段へ委ねる (当てずっぽうの引数で
        # 撃たない)。知識質問判定より前に置くのは算術式と同じ理由で、
        # 「〜には何がありますか」が知識質問にマッチして base の想像に落ちるため。
        if asks_directory_listing(query) and tools_registry.has("list_directory"):
            directory = resolve_listing_directory(query, get_project_root())
            if directory is not None:
                logger.debug(
                    "Rule-based: directory listing detected: %s", directory,
                )
                return ToolJudgement(
                    tool_needed=True,
                    tool_name="list_directory",
                    tool_args={"directory": directory},
                    source="rule",
                )

        # 知識質問はツール不要（RAG パイプラインで処理）
        # ただしツールパターン・ファイルパス・URL にもマッチするクエリは
        # ツール操作の可能性が高いため知識質問判定を適用しない
        has_tool_signal = _query_has_tool_signal(query)
        knowledge_patterns = select_locale_variant(_KNOWLEDGE_PATTERNS, _KNOWLEDGE_PATTERNS_EN)
        if not has_tool_signal and any(p.search(query) for p in knowledge_patterns):
            logger.debug("Rule-based: knowledge query detected, skipping tool: %s", query[:50])
            return ToolJudgement(tool_needed=False, source="rule")

        # 明示パス / URL があり、かつ具体的なツールまで決定論で解決できるなら
        # ``_TOOL_PATTERNS`` に無い言い回しでもここで確定させる。パス付きの依頼が
        # aux 層任せになっており、同じ依頼が read_file / search_history /
        # ツール未発火に割れていた (実インシデント 2026-08-04 ライブ監査:
        # 「E:/tmp/a.txt の中身を見せて、あわせて文字数も教えてください。」で
        # ツールが 1 つも走らず「存在しない」と誤答)。``_infer_tool`` は読み書きの
        # 動詞が無ければ空を返すので、パスに言及しただけの文は従来経路へ落ちる。
        if has_tool_signal:
            signal_name, signal_args = self._infer_tool(query, tools_registry, mode)
            if signal_name:
                logger.debug(
                    "Rule-based: path/URL signal resolved to %s: %s",
                    signal_name, query[:50],
                )
                return ToolJudgement(
                    tool_needed=True,
                    tool_name=signal_name,
                    tool_args=signal_args,
                    source="rule",
                )

        tool_patterns = select_locale_variant(_TOOL_PATTERNS, _TOOL_PATTERNS_EN)
        if not any(p.search(query) for p in tool_patterns):
            return ToolJudgement(tool_needed=False, source="rule")

        logger.debug("Rule-based: tool pattern matched for query: %s", query[:50])

        # ツール名と引数の推定
        tool_name, tool_args = self._infer_tool(query, tools_registry, mode)

        return ToolJudgement(
            tool_needed=True,
            tool_name=tool_name,
            tool_args=tool_args,
            source="rule",
        )

    def _judge_with_cartridge_hints(
        self,
        query: str,
        tools_registry: ToolsRegistry,
    ) -> ToolJudgement:
        """ロード済みカートリッジの tool_hints でツール呼び出しを判定

        カートリッジが定義するキーワードリストとクエリを照合し、
        マッチした場合は指定ツールへの誘導を返す。
        知識質問パターンに関係なく、カートリッジが「この分野はツール実行が有益」と
        宣言するためルールベースの知識質問判定より優先される。
        """
        if self._cartridge_manager is None:
            return ToolJudgement(tool_needed=False, source="rule")

        hints = self._cartridge_manager.get_tool_hints()
        if not hints:
            return ToolJudgement(tool_needed=False, source="rule")

        q_lower = query.lower()
        for hint in hints:
            patterns = hint.get("patterns", [])
            tool = hint.get("tool", "")
            if not tool or not patterns:
                continue
            for pattern in patterns:
                if pattern.lower() in q_lower:
                    if tools_registry.has(tool):
                        logger.debug(
                            "Cartridge hint matched: pattern=%s, tool=%s, query=%s",
                            pattern, tool, query[:50],
                        )
                        return ToolJudgement(
                            tool_needed=True,
                            tool_name=tool,
                            tool_args={},
                            source="cartridge",
                        )
                    break

        return ToolJudgement(tool_needed=False, source="rule")

    def _judge_with_learned_patterns(
        self,
        query: str,
        tools_registry: ToolsRegistry,
        mode: str,
    ) -> ToolJudgement:
        """学習済み tool_routing パターンでツール呼び出しを判定

        LearnedPatternStore の tool_routing カテゴリのパターンを参照し、
        閾値以上の重みでマッチした場合にツール実行を誘導する。
        """
        if self._learned_patterns is None:
            return ToolJudgement(tool_needed=False, source="rule")

        matches = self._learned_patterns.match(query, category="tool_routing")
        if not matches:
            return ToolJudgement(tool_needed=False, source="rule")

        top_weight = matches[0][1]
        if top_weight < self._tool_routing_threshold:
            return ToolJudgement(tool_needed=False, source="rule")

        logger.debug(
            "Learned tool_routing pattern matched: top=%s (w=%.3f), query=%s",
            matches[0][0], top_weight, query[:50],
        )

        # ツール名と引数を推定（静的パターンと同じロジック）
        tool_name, tool_args = self._infer_tool(query, tools_registry, mode)
        if not tool_name:
            # _infer_tool が推定できない場合の run_command フォールバックは
            # 「引数なしの仮判定」であり、judge() 側で
            # ルール層が command を解決できた場合
            # のみ生き残る。合成不成立 (aux が not executable と判定 /
            # aux 未接続) なら _suppress_commandless_run_command が
            # no_tool に倒す — 学習パターンの字句マッチだけを根拠に実行不能な
            # run_command を返さない (2026-07-20: 学習済み「説明」が知識質問
            # にマッチし create モードで誤発火し得た件の防衛線)。
            tool_name = "run_command" if tools_registry.has("run_command") else ""
        if not tool_name:
            return ToolJudgement(tool_needed=False, source="rule")

        return ToolJudgement(
            tool_needed=True,
            tool_name=tool_name,
            tool_args=tool_args,
            source="learned",
        )

    @staticmethod
    def _infer_tool(
        query: str,
        tools_registry: ToolsRegistry,
        mode: str,
    ) -> tuple[str, dict]:
        """クエリからツール名と引数を推定する

        Returns:
            (tool_name, tool_args): 推定結果。推定できない場合は ("", {})。
        """
        q = query.lower()
        # ファイルパス抽出用のクエリ。バッククォート内コマンドの引数パスは
        # 読み書きの対象ではないため取り除く (コマンド実行分岐は生の ``query``
        # を見るのでコマンド自体は失われない)。これが無いと「コマンド
        # `dir E:\tmp\x` を実行して出力を報告して」の「出力」が下の書込み
        # パターンに一致し、コマンド引数のパスが write_file の書込み先になる。
        path_query = strip_command_literals(query)

        # URL フェッチパターン（他のパターンより優先）
        # URL を含むクエリは fetch_url で処理する（run_command + curl に落ちるのを防止）
        url_match = _URL_IN_QUERY_RE.search(query)
        if url_match and tools_registry.has("fetch_url"):
            return "fetch_url", {"url": url_match.group(1)}
        if re.search(
            r"(?:フェッチ|fetch|取得して|アクセス|ウェブ|web|サイト|site|ページ|page"
            r"|ニュース|news|ブラウズ|browse)",
            q,
        ) and tools_registry.has("fetch_url"):
            # URL がクエリに含まれていないが、フェッチ意図がある場合
            return "fetch_url", {}

        # コード検証パターン（read_file より優先）
        # 「動作する？」「正しく動く？」等はファイルを読むより構文チェックの方が確実
        if re.search(
            r"(?:動作|動[くい]|実行でき|エラー|バグ|正常|正しく動)"
            r"|(?:work|run correctly|execute|error|bug)",
            q,
        ):
            path = _extract_file_path(query)
            if path and path.endswith(".py") and tools_registry.has("verify_syntax"):
                return "verify_syntax", {"file_path": path}

        # ファイル読込みパターン
        # 「確認」「チェック」「見せて」「内容」等は実質的にファイル読み取りを必要とする。
        # カタカナ「チェック」は日本語で頻出するため明示的に含める (ASCII "check" だけ
        # ではカタカナ表記を取りこぼし、後続の write パターンへ誤って落ちる)。
        # ファイルの行数・文字数を問う質問も読み取りが要る。モデルは本文から
        # 数えても正確にならないため read_file のメタ行 (lines / chars) を
        # 使わせる (実測 2026-08-05: ツール未発火のまま「確認できません」と
        # 回答放棄した)。パス抽出済みの分岐なので誤爆はファイル参照時に限る。
        # 有無を問う語 (存在し / ありますか / exists ...) も読み取り側に含める。
        # パス抽出済みの分岐なので「ファイルの話」であることは確定しており、
        # これが無いと「E:\tmp\a.txt はまだありますか」がツール未発火のまま
        # base の記憶で断定される (2026-08-16 監査の「存在しますか」形とは
        # 語尾違いで挙動が割れていた)。
        if re.search(
            r"(?:読[みむ]込|読んで|開いて|見せて|見て|確認|チェック|確かめ"
            r"|正し[いく]|合って|内容|中身|何文字|文字数|何行|行数"
            r"|存在し|ありますか|あるか|残ってい|消えてい"
            r"|read|show|check|verify|correct|content|view|exists?"
            r"|how many (?:characters|chars|lines))",
            q,
        ):
            path = _extract_file_path(path_query)
            if path:
                # ディレクトリ指定 (配下のファイルを点検する文脈) は read_file だと
                # "Not a file" になるため list_directory に振り分ける。
                if Path(path).is_dir() and tools_registry.has("list_directory"):
                    return "list_directory", {"directory": path}
                if tools_registry.has("read_file"):
                    args: dict = {"file_path": path}
                    head_lines = _extract_head_line_count(query)
                    if head_lines is not None:
                        args["start_line"] = 1
                        args["end_line"] = head_lines
                    elif asks_file_existence_only(query):
                        # 有無だけを問う質問に全文を渡すとモデルが全文を復唱する。
                        # メタ行 (lines / chars) だけで答えられるので 1 行に絞る。
                        args["start_line"] = 1
                        args["end_line"] = 1
                    return "read_file", args

        # ファイル書き込み/出力パターン
        # ディレクトリを書込み先に取ると write_file が配下に output_<UTC>.txt を
        # 捏造する (記述的な「出力」誤マッチで read 指示がここへ落ちるケースを含む)。
        # ディレクトリは書込み対象から除外する。
        if re.search(r"(?:書[きく]込|書いて|出力|保存|生成|作成|write|save|output)", q):
            path = _extract_file_path(path_query)
            if path and not Path(path).is_dir() and tools_registry.has("write_file"):
                return "write_file", {"file_path": path}

        # コマンド実行パターン
        # "run" は ASCII 境界必須 ("running" 等の語幹への部分一致誤爆対策。
        # "What OS am I running?" が意図せずここで確定していた実績あり)。
        # "exec" は execute/executing/executed の活用形も拾う必要があるため、
        # 境界を活用語尾まで含めた明示形にする (単純な境界だと "execute" を
        # 取りこぼす)。
        if re.search(
            r"(?:実行|(?<![A-Za-z])run(?![A-Za-z])"
            r"|(?<![A-Za-z])exec(?:ute[sd]?|uting)?(?![A-Za-z]))",
            q,
        ) and tools_registry.has("run_command"):
            # ファイルパスがあれば python で実行
            path = _extract_file_path(query)
            if path and path.endswith(".py"):
                return "run_command", {"command": f'python "{path}"'}
            # バッククォート内のコマンド
            cmd_match = re.search(r'`([^`]+)`', query)
            if cmd_match:
                return "run_command", {"command": cmd_match.group(1)}
            return "run_command", {}

        # コード検索パターン (_TOOL_PATTERNS と同一の共起ガード _CODE_SEARCH_PATTERNS
        # を再利用。汎用「検索」単独は "Binary Search Tree" のような英語クラス名
        # の部分一致にも誤爆するため、コード/ファイル文脈語との共起を要求する
        # — 2026-07-22 監査で判明。裸の検索語だけで抽出パターンが確定し、
        # 無関係ファイルへの search_code 誤発火を招いていた)
        # 所在を問う言い回し (「<識別子> はどこで使われていますか」)。
        # _CODE_SEARCH_PATTERNS は「コード/ファイル語 × 検索動詞」の共起を
        # 要求するため、この言い方は**どちらの語も含まず**ルール層を素通りする。
        #
        # より一般的な _CODE_SEARCH_PATTERNS より **先に** 判定する: 所在質問は
        # 疑問詞が骨組みなので、汎用の _extract_search_pattern だと疑問詞自体を
        # 検索語に採ってしまう ("where is search_code used?" → pattern="where")。
        if (_is_code_usage_location_query(query)
                and tools_registry.is_available("search_code", mode)):
            pattern = _code_usage_location_pattern(query)
            if pattern:
                return "search_code", {"pattern": pattern}

        if (any(p.search(query) for p in _CODE_SEARCH_PATTERNS)
                and tools_registry.has("search_code")):
            # クエリからキーワードを抽出して pattern 引数に設定
            pattern = _extract_search_pattern(query)
            if pattern:
                return "search_code", {"pattern": pattern}
            return "search_code", {}

        # Python 実行可能クエリ（システム情報・数値処理・データ処理・変換）
        # これらのクエリは Python コード生成 → run_command で正確に回答できる。
        # ツール名は mode から解決する (chat は run_command_readonly)。
        exec_tool = _executable_tool_for_mode(tools_registry, mode)
        exec_query_re = select_locale_variant(_INFER_TOOL_EXEC_QUERY_RE, _INFER_TOOL_EXEC_QUERY_RE_EN)
        if (
            exec_query_re.search(q)
            and exec_tool
            and not asks_about_prior_conversation_entity(query)
        ):
            # システム情報クエリは具体的なコマンドを生成
            command = _infer_executable_command(query)
            if command:
                return exec_tool, {"command": command}
            return exec_tool, {}

        # 計算パターン (実行可能クエリ分岐より後ろに置く。「フィボナッチ数列の
        # 10番目を計算して」のように両方にマッチするクエリは、式抽出を持たない
        # calculate {} で潰さず run_command 合成経路 (aux synth) に乗せる。
        # 2026-07-21 ライブ検証 ターン35 のインシデント対策)
        if re.search(r"(?:計算|calculate)", q) and tools_registry.has("calculate"):
            return "calculate", {}

        return "", {}

    def _build_system_prompt(
        self,
        tools_registry: ToolsRegistry,
        mode: str,
    ) -> str:
        """ツール判定用システムプロンプトを構築"""
        # AuxPromptManager からタスク別プロンプトを取得
        if self._prompt_manager is not None:
            try:
                base_prompt = self._prompt_manager.get_aux_prompt("tool_call")
            except ValueError:
                base_prompt = _DEFAULT_SYSTEM_PROMPT
        else:
            base_prompt = _DEFAULT_SYSTEM_PROMPT

        # ツール一覧を動的に注入
        tool_descriptions = tools_registry.get_descriptions_text(mode)
        return f"{base_prompt}\n\n## 利用可能なツール\n{tool_descriptions}"

    def _build_user_prompt(
        self,
        query: str,
        conversation: list[dict] | None,
    ) -> str:
        """ユーザープロンプトを構築"""
        parts = []

        # 直近の会話コンテキスト（最大2ターン）
        if conversation:
            recent = conversation[-4:]  # 最大2ターン分
            context_lines = []
            for msg in recent:
                role = msg.get("role", "")
                content = msg.get("content", "")[:_JUDGE_CONTEXT_CHARS]
                if role in ("user", "assistant"):
                    context_lines.append(f"{role}: {content}")
            if context_lines:
                parts.append("## 直近の会話\n" + "\n".join(context_lines))

        parts.append(f"## ユーザーのリクエスト\n{query}")
        return "\n\n".join(parts)

    def _parse_response(self, content: str) -> ToolJudgement:
        """補助タスクの応答をパースして ToolJudgement に変換

        ``response_format=json_schema`` 制約サンプリングが効いている場合は
        ``{"tool": "...", "args": {...}}`` 形式の有効な JSON が必ず返る。
        フラグ無効化時 / 古い llama-server build / max_tokens 切断時の
        フォールバックとして共通実装 ``extract_json_object`` を経由する。
        """
        # 共通 JSON 抽出経路
        data = extract_json_object(content)
        if isinstance(data, dict):
            return _json_to_judgement(data)

        # JSON が抽出できないケースはツール不要と判定 (安全側)
        logger.warning(
            "Could not parse LLM response for tool judgement: %s",
            content[:100],
        )
        return ToolJudgement(tool_needed=False, source="llm")


def _infer_executable_command(query: str) -> str:
    """executable query パターンから具体的な Python コマンドを生成する

    _EXECUTABLE_QUERY_COMMANDS の各パターンを順に照合し、
    最初にマッチしたコマンドを返す。
    マッチしない場合（数値処理・データ処理等）は空文字列を返す。

    Returns:
        生成されたシェルコマンド。該当なしの場合は空文字列。
    """
    for pattern, command in _EXECUTABLE_QUERY_COMMANDS:
        if pattern.search(query):
            if callable(command):
                return command(query)
            return command
    return ""


def _extract_search_pattern(query: str) -> str:
    """クエリから検索パターンを抽出する

    「検索」「search」等のキーワード自体を除外し、
    実際の検索対象となる語句を返す。

    例:
        "関数名 hello を検索して" → "hello"
        "search for parse_config" → "parse_config"
        "grep pattern" → "pattern"
    """
    # バッククォート内のパターン
    m = re.search(r'`([^`]+)`', query)
    if m:
        return m.group(1)

    # 引用符内のパターン
    m = re.search(r'[「"\'](.*?)[」"\']', query)
    if m:
        return m.group(1)

    # 検索/search/grep/find 等を除去した残りからキーワードを抽出
    cleaned = re.sub(
        r"(?:を|で|して|する|しろ|で検索|を検索|検索して|検索する"
        r"|search\s+(?:for|in)|grep|find|検索)",
        " ", query,
    )
    # 英数字・アンダースコアで構成されるトークンを探す
    tokens = re.findall(r"[A-Za-z_]\w{2,}", cleaned)
    if tokens:
        return tokens[0]

    return ""


# ディレクトリパス抽出用: ドライブレター配下のパスセグメントを解析する。
# 各セグメントは「\」直後が非空白文字で始まる前提とする
# (``[A-Za-z0-9_.]`` から開始し、内部は空白を含んでよい)。
# 「...\aa\ with the content」のように、ディレクトリ指定の直後に自然文
# (英語の説明文) が「\」+ 空白で続くケースを誤ってパスセグメントとして
# 飲み込まないための境界条件 (#incident: 日本語ファイル名クエリで
# planner が生成した英語タスク記述の一部がパスに混入した)。
# 実在の Windows パスでバックスラッシュ直後が空白になることはない
# ("Program Files" のようにセグメント内部に空白を含むのは許容する)。
#: ドライブレター付きパスの区切り。Windows はスラッシュ区切りも等価に受け付け、
#: ユーザーもツール出力もそちらを書く。バックスラッシュ限定にしていたため
#: ``E:/tmp/a.txt`` が 1 つも抽出できず、ルール層が read_file を選べないまま
#: aux 層へ落ちていた (実インシデント 2026-08-04 ライブ監査: 同じ依頼が
#: read_file / search_history / ツール未発火に割れる原因)。
_DIR_PATH_RE = re.compile(
    r"([A-Za-z]:(?:[\\/][A-Za-z0-9_.][A-Za-z0-9_. -]*)*)",
)

# クォート文字の対応表（開き, 閉じ）。ファイル名の語幹 (拡張子直前) が
# 日本語等の非ASCIIの場合に、明示的にクォートされたファイル名を抽出する
# 際に使う。
_QUOTE_PAIRS: tuple[tuple[str, str], ...] = (
    ('"', '"'), ("'", "'"), ("「", "」"), ("『", "』"),
)


def _extract_quoted_filename(query: str) -> str | None:
    """クォートで明示的に囲まれたファイル名を抽出する（非ASCII語幹対応）。

    ``[A-Za-z0-9_-]+\\.ext`` 前提の ASCII 限定パターンでは、「テスト.docx」の
    ように拡張子直前が日本語等の非ASCIIだと一切マッチしない。クォートで
    明示されていれば語幹の文字種を問わず抽出する（クォート無しの非ASCII
    語幹は文中の地の文と区別できず誤検出リスクが高いため対象外）。
    """
    for open_q, close_q in _QUOTE_PAIRS:
        m = re.search(
            re.escape(open_q)
            + r"([^\n" + re.escape(open_q) + re.escape(close_q) + r"]{1,200}"
            r"\.[A-Za-z0-9]{1,10})"
            + re.escape(close_q),
            query,
        )
        if m:
            return m.group(1).strip()
    return None


#: 「同じファイルに」「そのファイルを」等、保存先を直前の文脈に委ねる表現。
#: ``さきほど`` (ひらがな) は 2026-08-09 に追加。``先ほど`` / ``さっき`` しか
#: 無く、「さきほど作った notes.txt に追記して」が参照表現として認識されず
#: 書込みが 1 度も走らないまま完了を捏造していた。
_REFERENTIAL_TARGET_RE = re.compile(
    r"(?:同じ|その|この|先ほどの?|さきほどの?|さっきの?)\s*(?:ファイル|ところ|場所)"
    r"|保存し直|上書き|書き直して保存|同じ場所に"
    r"|\b(?:same|that)\s+file\b|\boverwrite\b",
    re.IGNORECASE,
)
#: 保存/書き出しを求める動詞 (パス無しの参照依頼を拾うための最小集合)。
#: ``追記`` / ``書き足`` / ``書[きい]て`` は 2026-08-09 に追加 (実インシデント:
#: 「そのファイルの末尾に追記して書いて」が保存動詞として認識されなかった)。
_REWRITE_VERB_RE = re.compile(
    r"保存|書き込|書き出|書き足|追記|上書き|セーブ|書[きい]て"
    r"|\bsave\b|\bwrite\b|\bappend\b|\boverwrite\b",
    re.IGNORECASE,
)
#: パス区切りを含むか (ドライブ接頭辞 / スラッシュ / バックスラッシュ)。
#: 含まない = 裸のファイル名で、書込み先としては **どのディレクトリか未確定**。
_PATH_SEPARATOR_RE = re.compile(r"[\\/]")


def _resolve_referenced_path(
    query_path: str | None, conversation: list[dict] | None,
) -> str | None:
    """書込み/読取の対象パスを会話から解決する (純粋関数)。

    ``query_path`` の状態で 3 通りに分かれる:

    - ディレクトリを含む絶対/相対パス → そのまま採用 (解決不要)
    - 裸のファイル名 (``notes.txt``) → 会話に **同じ basename** の
      フルパスがあればそれを採用。無ければ ``None``
    - ``None`` / 空 (「そのファイル」型) → 会話で最後に出たパスを採用

    裸のファイル名をそのままツールへ渡すとカレントディレクトリに着地して
    しまい、ユーザーが指した既存ファイルとは別物を作る。会話で確定している
    場合のみ解決し、確定できなければ ``None`` を返して後続層に委ねる
    (推測でパスを埋めない)。
    """
    if query_path and _PATH_SEPARATOR_RE.search(query_path):
        return query_path
    want = (query_path or "").strip().lower() or None
    for msg in reversed(list(conversation or [])):
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        path = _extract_file_path(content)
        if not path or not _PATH_SEPARATOR_RE.search(path):
            continue
        if want and _PATH_SEPARATOR_RE.split(path)[-1].lower() != want:
            continue
        return path
    return None


def _referential_rewrite_judgement(
    query: str, conversation: list[dict] | None, tools_registry: ToolsRegistry,
) -> "ToolJudgement | None":
    """「同じファイルに保存し直して」型の依頼を write_file に確定させる。

    保存動詞があり、かつ書込み先がクエリだけでは確定しない (参照表現、または
    ディレクトリを伴わない裸のファイル名) 場合に、直近の会話からパスを引いて
    ``write_file`` を返す。該当しない (ディレクトリ付きパスが本文にある /
    参照も裸名も無い / 会話にパスが無い) 場合は ``None`` で後続層に委ねる。
    純粋関数 (レジストリ参照のみ)。

    裸のファイル名を拾うのは 2026-08-09 のライブ監査で判明した実害への対処:
    「inventory_notes.txt に 1 行追記してください」がどの層にも拾われず
    deliberative に落ち、ツールを 1 つも撃たないまま **フルパスを補って**
    「E:\\tmp\\inventory_notes.txt の末尾に追記しました」と報告した
    (実ファイルは無変更)。フルパスで同じ依頼をすると正常に書き込まれており、
    差はパス表記だけだった。
    """
    if not tools_registry.has("write_file"):
        return None
    if not _REWRITE_VERB_RE.search(query):
        return None
    query_path = _extract_file_path(query) or None
    if query_path and _PATH_SEPARATOR_RE.search(query_path):
        return None  # ディレクトリ付きパスが本文にあるなら通常のルール層で足りる
    if not query_path and not _REFERENTIAL_TARGET_RE.search(query):
        return None
    path = _resolve_referenced_path(query_path, conversation)
    if not path:
        return None
    logger.info(
        "Referential rewrite: resolved target from conversation: %s "
        "(query_path=%r)", path, query_path,
    )
    return ToolJudgement(
        tool_needed=True,
        tool_name="write_file",
        tool_args={"file_path": path},
        source="rule",
    )


#: ファイルの中身を「見せる」ことを求める表現。read_file を撃たずに答えると
#: 記憶から再構成した偽の内容を「ファイルの中身」として提示する
#: (2026-08-09 ライブ監査: 追記直後の「全文をそのまま見せて」で 3 行とも実
#: ファイルと不一致、しかも同一セッション内の誤答が中身として混入した)。
_FILE_CONTENT_DISPLAY_RE = re.compile(
    r"(?:全文|中身|内容|そのまま|中身をそのまま)"
    r".{0,20}?(?:見せ|表示|出して|教えて|確認)"
    r"|(?:見せ|表示).{0,10}?(?:全文|中身|内容)"
    r"|\bshow\s+(?:me\s+)?(?:the\s+)?(?:full\s+)?(?:content|contents|file)\b"
    r"|\b(?:display|print)\s+(?:the\s+)?(?:content|contents|file)\b",
    re.IGNORECASE,
)
#: 「ファイル」を指す語。表示要求が **ファイルに関するもの** かの絞り込みに使う。
_FILE_NOUN_RE = re.compile(r"ファイル|\bfile\b", re.IGNORECASE)

#: ファイルの計測値 (行数・文字数・サイズ) を尋ねる表現。``read_file`` の結果には
#: ``lines`` / ``chars`` のメタ行が付くので、読めば決定論で答えられる。撃たないと
#: モデルが数値を捏造する — しかも **正解が直前ターンに出ていても**捏造する
#: (実インシデント 2026-08-10 ライブ監査: 直前の read_file 出力に
#: ``lines: 10 | chars: 411`` と表示されていたのに「12 行、357 文字」と答えた)。
_FILE_METRICS_RE = re.compile(
    r"(?:行数|文字数|バイト数|何行|何文字|ファイルサイズ)"
    r"|\b(?:line|character|byte|word)\s*count\b"
    r"|\bhow\s+many\s+(?:lines|characters|bytes|words)\b",
    re.IGNORECASE,
)


def _referential_read_judgement(
    query: str, conversation: list[dict] | None, tools_registry: ToolsRegistry,
) -> "ToolJudgement | None":
    """「そのファイルの全文を見せて」型の依頼を read_file に確定させる。

    ``_referential_rewrite_judgement`` の読取版。書込み側と同じく、対象が
    クエリだけでは確定しない (参照表現 / 裸のファイル名) 場合に会話から
    パスを引く。ディレクトリ付きパスが本文にあるなら通常のルール層で足りる。

    ファイル名詞または参照表現を要求するので、「さっきの説明の中身を見せて」の
    ような非ファイルの表示要求は拾わない。純粋関数 (レジストリ参照のみ)。
    """
    if not tools_registry.has("read_file"):
        return None
    # 計測値の問い合わせも読取で決まる (read_file が lines/chars を返す)。
    wants_metrics = bool(_FILE_METRICS_RE.search(query))
    if not (_FILE_CONTENT_DISPLAY_RE.search(query) or wants_metrics):
        return None
    query_path = _extract_file_path(query) or None
    if query_path and _PATH_SEPARATOR_RE.search(query_path):
        return None
    if not query_path and not (
        _REFERENTIAL_TARGET_RE.search(query) or _FILE_NOUN_RE.search(query)
    ):
        return None
    path = _resolve_referenced_path(query_path, conversation)
    if not path:
        return None
    logger.info(
        "Referential read: resolved target from conversation: %s "
        "(query_path=%r)", path, query_path,
    )
    tool_args: dict = {"file_path": path}
    # 計測は全文を読まないと数えられないので範囲指定しない。
    head = None if wants_metrics else _extract_head_line_count(query)
    if head is not None:
        # ``_infer_tool`` と同じ引数形 (read_file は start/end_line を取る)。
        tool_args["start_line"] = 1
        tool_args["end_line"] = head
    return ToolJudgement(
        tool_needed=True,
        tool_name="read_file",
        tool_args=tool_args,
        source="rule",
    )


#: 「最初の 3 行」「先頭 10 行」「first 5 lines」等、ファイル先頭からの行数指定。
#: 全角数字も拾う (日本語入力では「３行」になりやすい)。
_HEAD_LINES_RE = re.compile(
    r"(?:最初|先頭|冒頭|頭|first|head|top)\D{0,6}?([0-9０-９]{1,4})\s*(?:行|lines?)",
)

#: 「このファイルは存在しますか」= 有無だけを問う質問。
_FILE_EXISTENCE_RE = re.compile(
    r"(?:存在し|ありますか|あるか|残ってい|消えてい|できてい"
    r"|\bexists?\b|\bis there\b|\bstill there\b)",
    re.IGNORECASE,
)
#: 本文そのものを求める語。存在確認と併記されていれば内容要求が優先される
#: (「まだ存在しますか？先頭3行だけ見せてください」)。
_FILE_CONTENT_REQUEST_RE = re.compile(
    r"(?:見せ|見たい|中身|内容|読[んみむ]|表示|出力|全文|何文字|文字数|何行|行数"
    r"|\bshow\b|\bcontent\b|\bread\b|\bdisplay\b|\bprint\b|\bdump\b)",
    re.IGNORECASE,
)


def asks_file_existence_only(query: str) -> bool:
    """ファイルの有無だけを問い、本文は求めていないか。

    有無だけを聞かれているのに ``read_file`` を範囲指定なしで撃つと全文が
    ツール結果として返り、モデルはそれを回答に丸ごと復唱する。

    2026-08-16 ライブ監査ターン 14「E:\\...\\README.md というファイルは存在
    しますか？」: 3,331 文字の全文が返り、モデルは全文の復唱を始めて
    **ちょうど 1,024 トークン (llama.max_tokens の既定値) で表の途中で切断**
    された。yes/no の質問に **197 秒** かけ、しかも回答は未完だった。

    ``read_file`` は先頭にメタ行 ``[file: ... | lines: N | chars: M]`` を付ける
    ので、1 行だけ読めば「存在する / 何行・何文字か」は決定論的に答えられる。
    """
    return bool(
        _FILE_EXISTENCE_RE.search(query)
        and not _FILE_CONTENT_REQUEST_RE.search(query),
    )


def _extract_head_line_count(query: str) -> int | None:
    """「最初の N 行」の N を返す (指定が無ければ ``None``)。

    本文全体を渡すとモデルが行数指定を守らずほぼ全文を出力するため
    (実測 2026-08-05: NOTICE.md の「最初の 3 行」で約 1,264 文字を出力)、
    read_file 側で切り出せるようにツール引数へ渡す。
    """
    m = _HEAD_LINES_RE.search(query)
    if not m:
        return None
    try:
        count = int(m.group(1).translate(_ZENKAKU_DIGITS))
    except ValueError:
        return None
    return count if count > 0 else None


#: 全角数字 → ASCII。
_ZENKAKU_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")


def _extract_file_path(query: str) -> str:
    """クエリからファイルパスを抽出する

    日本語の自然言語テキストからファイルパスを抽出する。
    「e:\\直下にa.txtのファイル名で...」→ 「e:\\a.txt」のように、
    ドライブレターとファイル名を組み合わせて解釈する。
    抽出後、連続バックスラッシュ (\\\\) をシングル (\\) に正規化する。
    """
    # URL はファイル名抽出の対象から除外する。URL ドメイン (例: soccer.yahoo.co.jp)
    # が「co.jp」のようなファイル名として誤抽出されるのを防ぐ。
    query = _URL_IN_QUERY_RE.sub(" ", query)

    # 1a. 非 ASCII を含みうるフルパス: E:\tmp\日本語テスト.txt / E:/tmp/日本語.txt
    #     ASCII 限定にすると日本語ファイル名が拡張子の手前で切れ、切り詰めた
    #     パスがたまたま実在ディレクトリだと read_file ではなく list_directory が
    #     選ばれ、実在するファイルを「見つからない」と答える (実測 2026-08-05)。
    #     区切りは \ と / の双方を受ける。バックスラッシュ限定だと ``E:/tmp/a.txt``
    #     が 1 つも抽出できず、同じ依頼が read_file / search_history / ツール
    #     未発火に割れていた (実測 2026-08-04)。
    #     地の文を飲み込まないための境界条件は 2 つ:
    #       - 空白 (半角/全角) とクォートを含まない (「E:\tmp に置いた report.txt」)
    #       - ドライブ直下ではなく 1 階層以上下 (「e:\直下にa.txtのファイル名で」)。
    #         ドライブ直下 + 非 ASCII は地の文と構造的に区別できないため、
    #         従来どおり Pattern 2 (ドライブ + ファイル名) に委ねる。
    m = re.search(
        r"[A-Za-z]:[\\/][^\s　\"'「」『』\\/]+[\\/][^\s　\"'「」『』]*\.[A-Za-z0-9]{1,10}",
        query,
    )
    if m:
        return _normalize_path_separators(m.group(0))

    # 1b. 空白を含む ASCII パス: C:\Program Files\app.exe
    #     空白を許容する代償として本体は ASCII 限定にし、地の文 (日本語) で
    #     停止させる。区切りは 1a と同様に \ と / の双方を受ける。
    m = re.search(r"[A-Za-z]:[\\/][A-Za-z0-9_.\\/ -]+\.[A-Za-z0-9]{1,10}", query)
    if m:
        return _normalize_path_separators(m.group(0).rstrip(" "))

    # 2. ドライブレター + 自然言語でのファイル名指定
    #    例: 「e:\直下にa.txtのファイル名で」→ e:\a.txt
    #    ディレクトリとファイル名が日本語/全角スペースで分断されていても、
    #    ディレクトリ部 (Pattern 3 と同じ捕捉) を取り出してファイル名と結合し、
    #    サブ階層を保持する。深い階層が無い (ドライブ直下指定) 場合のみ
    #    従来どおりドライブ直下へフォールバックする。
    #    \w は日本語にもマッチするため ASCII 限定で検索
    drive_match = re.search(r"([A-Za-z]):[\\/]", query)
    file_match = re.search(r"([A-Za-z0-9_-]+\.[A-Za-z0-9]{1,10})(?=[^A-Za-z0-9_.]|$)", query)
    # ファイル名の語幹が非ASCII (日本語等) だと file_match はマッチしない
    # ("テスト.docx" 等)。その場合はクォートで明示されたファイル名を拾う。
    filename = file_match.group(1) if file_match else _extract_quoted_filename(query)
    if drive_match and filename:
        dir_match = _DIR_PATH_RE.search(query)
        if dir_match:
            # セグメント内部は空白を許容するため ("Program Files" 等)、末尾に
            # 地の文へ続く空白が巻き込まれることがある (例: "aa に保存して" の
            # "aa " )。rstrip() で末尾空白 (全角含む) を落としてから区切りも除去。
            # さらに英語の地の文が空白のみで続くケース ("aa in Excel format") は
            # 実在チェックで切り落とす。
            directory = _trim_nonexistent_path_tail(
                _normalize_path_separators(
                    dir_match.group(1).rstrip(),
                ).rstrip("\\/"),
            )
            return f"{directory}\\{filename}"
        return f"{drive_match.group(1)}:\\{filename}"

    # 3. ディレクトリパスのみ（ファイル名なし）: E:\xxx\ や E:\xxx 等
    #    配下のファイルを参照する文脈では、ディレクトリパスを返す。
    #    全角スペース (U+3000) 等の Unicode 空白や文末で終端しても、
    #    セグメント単位で解析する _DIR_PATH_RE が自然に正しい境界で止まる。
    if drive_match:
        dir_match = _DIR_PATH_RE.search(query)
        if dir_match:
            return _trim_nonexistent_path_tail(
                _normalize_path_separators(dir_match.group(1).rstrip()),
            )

    # 4. Unix パス: /home/user/file.txt
    m = re.search(r"(?:^|[\s　])((?:/[\w._-]+){2,})", query)
    if m:
        return m.group(1)

    # 5. bare ファイル名 (拡張子付き): dice_roller.py / README.md / app.svelte
    #    ドライブレターも Unix パスもない場合のフォールバック。CWD 相対として
    #    のタスクを出したとき write_file の auto-recovery / fast-path が働くようにする。
    #    誤検出防止のため拡張子は英字始まりに限定 (「3.12」「v1.2」等を弾く)。
    m = re.search(
        r"(?:^|[\s　`'\"(\[])"
        r"([A-Za-z0-9_][A-Za-z0-9_.-]{0,127}\.[A-Za-z][A-Za-z0-9]{0,9})"
        r"(?=$|[\s　`'\")\].,;:!?])",
        query,
    )
    if m:
        return m.group(1)

    return ""


# --- 算術式抽出 (calculate ツールの決定論的ルーティング) ---------------------
# 全角の数字・演算子を ASCII へ寄せる。カタカナ長音符 (ー) や罫線 (―) は
# 日本語語中に頻出するため意図的に含めない (マイナスへ誤変換すると
# 「コーヒー」等が式断片に見えてしまう)。
_ARITH_NORMALIZE = str.maketrans({
    "０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
    "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",
    "＋": "+", "－": "-", "−": "-",
    "×": "*", "✕": "*", "＊": "*",
    "÷": "/", "／": "/", "％": "%", "＾": "^",
    "（": "(", "）": ")", "．": ".",
})
# 算術式になりうる文字だけからなる連続領域
_ARITH_RUN_RE = re.compile(r"[0-9.+\-*/%^()\s]+")
# 日付・バージョン番号の誤検出除け (2026-07-27 は BinOp として parse できてしまう)
_ARITH_DATE_LIKE_RE = re.compile(
    r"^(?:\d{4}\s*-\s*\d{1,2}\s*-\s*\d{1,2}"
    r"|\d{1,2}\s*/\s*\d{1,2}(?:\s*/\s*\d{2,4})?)$",
)
# 「式の値を求めている」ことの手掛かり。式だけが裸で書かれた場合は不要。
_ARITH_REQUEST_CUE_RE = re.compile(
    r"(?:いくつ|いくら|答え|計算|求め|何になる|=|＝"
    r"|(?<![A-Za-z])calculate(?![A-Za-z])|(?<![A-Za-z])compute(?![A-Za-z])"
    r"|what\s+is|how\s+much|(?<![A-Za-z])equals?(?![A-Za-z]))",
    re.IGNORECASE,
)
# 式の直後に助詞と疑問符しか残らない形 (「1+1は？」「12*34」) も計算依頼とみなす
_ARITH_BARE_TAIL_RE = re.compile(r"^[\s　]*(?:とは|って|は|の)?[\s　]*[?？。!！]*$")
_ARITH_SAFE_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd,
)


def _is_numeric_expression(expression: str) -> bool:
    """``expression`` が数値リテラルと算術演算子だけで構成されるか (純粋関数)。"""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return False
    has_operator = False
    for node in ast.walk(tree):
        if not isinstance(node, _ARITH_SAFE_NODES):
            return False
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            return False
        if isinstance(node, ast.BinOp):
            has_operator = True
    return has_operator


def _extract_arithmetic_expression(query: str) -> str:
    """クエリに書かれた算術式を Python 構文へ正規化して返す (純粋関数)。

    「1234 × 5678 はいくつですか？」のような明示的な計算依頼で ``calculate``
    を決定論的に発火させるための抽出器。ルール層は従来「計算」の字句しか
    見ておらず、式そのものを書かれるとツール無しで base の暗算に落ちて
    誤答していた (実インシデント 2026-07-27 ライブ検証: 1234 × 5678 に
    7060672 と回答。正解は 7006652)。

    誤検出を避けるため、以下をすべて満たす場合のみ式を返す:

    * 数値リテラルと算術演算子のみで構成され、二項演算を 1 つ以上含む
    * 日付 (2026-07-27) / 日付表記 (7/27) ではない
    * 値を尋ねる手掛かり語があるか、式の前に文が無く後ろも助詞・疑問符だけ
      (「12*34」「1+1は？」のような裸の式)

    Returns:
        正規化済みの式。抽出できなければ空文字列。
    """
    normalized = query.translate(_ARITH_NORMALIZE)
    for match in _ARITH_RUN_RE.finditer(normalized):
        candidate = match.group(0).strip()
        if not candidate or _ARITH_DATE_LIKE_RE.match(candidate):
            continue
        # ^ は Python では XOR。書かれた意図は冪乗なので ** へ寄せる。
        candidate = candidate.replace("^", "**")
        if not _is_numeric_expression(candidate):
            continue
        head = normalized[: match.start()]
        tail = normalized[match.end():]
        bare = (
            not any(c.isalnum() for c in head)
            and _ARITH_BARE_TAIL_RE.match(tail) is not None
        )
        if bare or _ARITH_REQUEST_CUE_RE.search(normalized):
            return candidate
    return ""




def _normalize_path_separators(path: str) -> str:
    """連続バックスラッシュをシングルに正規化する

    LLM や JSON パース経由でパスが二重エスケープされるケースに対応。
    例: E:\\\\xxx\\\\tetris.py → E:\\xxx\\tetris.py
    """
    # 連続する2つ以上の \ を1つに置換
    return re.sub(r"\\{2,}", r"\\", path)


def _trim_nonexistent_path_tail(path: str) -> str:
    """実在チェックに基づき、パス末尾へ混入した自然文トークンを切り落とす。

    ``_DIR_PATH_RE`` はセグメント内部の空白を許容する ("Program Files") ため、
    LLM 生成のタスク記述がパス直後に空白 + 英語の修飾語を続けると
    (実インシデント: ``...to C:\\...\\Desktop\\aa in Excel format``) 地の文が
    末尾セグメントへ飲み込まれ、実在しない拡張子なしパスへの平文書込みに
    化ける (リッチ文書経路・検証ゲートをすべてバイパス)。

    捕捉パスが実在しない場合のみ、空白区切りトークンを右から 1 つずつ外し
    ながら「実在する最長の空白境界プレフィックス」を探して返す。地の文の
    混入は必ず空白境界で起きるため、バックスラッシュ境界では分割しない。
    どのプレフィックスも実在しなければ原文のまま返す (新規パスの指定を
    壊さない)。
    """
    try:
        if not path or Path(path).exists():
            return path
    except (OSError, ValueError):
        return path
    candidate = path
    while " " in candidate:
        candidate = candidate.rsplit(" ", 1)[0].rstrip()
        if not candidate or candidate.endswith(":"):
            break
        try:
            if Path(candidate).exists():
                return candidate
        except (OSError, ValueError):
            break
    return path


def _json_to_judgement(data: dict) -> ToolJudgement:
    """JSON dict を ToolJudgement に変換

    補助タスク応答は ``response_format`` 無効 / 古い llama-server / max_tokens 切断
    時に ``json_repair`` で機械修復されるため、``tool`` / ``args`` が非想定型
    (list / str 等) になりうる。``ToolJudgement.tool_args`` は dict 契約なので、
    下流 (``deliberative._execute_tool`` の ``dict(tool_args)`` 等) が落ちないよう
    ここで強制正規化する。
    """
    tool = data.get("tool", "")
    if not isinstance(tool, str):
        tool = ""
    if not tool or tool == "no_tool":
        return ToolJudgement(tool_needed=False, source="llm")
    args = data.get("args", {})
    if not isinstance(args, dict):
        args = {}
    return ToolJudgement(
        tool_needed=True,
        tool_name=tool,
        tool_args=args,
        source="llm",
    )


_DEFAULT_SYSTEM_PROMPT = """\
# ツール呼び出し判定

ユーザーのリクエストを分析し、ツールの使用が必要かどうかを判定してください。

## 判定基準
- ファイルの読み書きが必要 → 該当ツールを選択
- シェルコマンドの実行が必要 → run_command を選択
- コード検索が必要 → search_code を選択
- ツールが不要な質問（知識・説明・会話） → ツール不要 (tool="")
- ユーザー自身が行う宣言（「探してみるね」「自分で調べる」等の一人称の意思表明）→ 依頼ではないためツール不要 (tool="")

## 出力形式
必ず JSON オブジェクトで出力してください (**1 行のコンパクト形式**、改行・余分な空白なし):
- ツールが必要: {"tool": "ツール名", "args": {"引数名": "値"}}
- ツールが不要: {"tool": "", "args": {}}

grammar 非強制モデルでも余計な空白で token を浪費し切り詰められないよう、空白は最小化する。
"""
