"""アシストモデルによるツール呼び出し判定（Free/Pro 共通）

ユーザークエリと利用可能なツール一覧をアシストモデルに提示し、
ツール呼び出しの要否・ツール名・引数を判定する。
アシストモデル未接続時はルールベースにフォールバックする。
"""

from __future__ import annotations

import asyncio
import re
import shlex
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from backend.free.agent.reactive import GREETING_RESPONSES
from backend.free.agent.tools_registry import ToolsRegistry
from backend.free.llm.json_extract import extract_json_object
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.debug_logger import DebugLogger
    from backend.free.agent.learned_patterns import LearnedPatternStore
    from backend.free.memory.views.mem import MemFactView
    from backend.free.rag.cartridge_manager import CartridgeManager
    from backend.free.rag.embedding_backend import EmbeddingBackend

logger = get_logger("agent.tool_call_judge")

# executable_command_synth が合成したコマンドの事後検証用。
# synth プロンプトは「環境依存事実のみ・副作用/ネットワーク送信なし」を要求するが、
# アシストがこれを破ってネットワークコマンドや構文エラーの python -c を返すため、
# コード側で enforce する (壊れたコマンドの実行・ユーザ露出・誤 success 学習の防止)。
_NETWORK_EGRESS_MARKERS = (
    "http://", "https://",
    "import requests", "requests.",
    "import urllib", "urllib.request",
    "import httpx", "httpx.",
    "import socket", "socket.socket",
    "curl ", "wget ",
)


def _extract_python_c_payload(command: str) -> str | None:
    """``python -c <code>`` 形式ならその ``<code>`` を返す。それ以外は ``None``。"""
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return None
    if not tokens:
        return None
    base = Path(tokens[0]).name.lower()
    if not (base.startswith("python") or base in {"py", "py.exe"}):
        return None
    for i, tok in enumerate(tokens[:-1]):
        if tok == "-c":
            return tokens[i + 1]
    return None


def _reject_synthesized_command(command: str) -> str | None:
    """合成 executable command が synth プロンプトのスコープを逸脱していれば理由を返す。

    1. ネットワーク送信 (requests / urllib / httpx / socket / http(s):// /
       curl / wget) を含む → synth プロンプトの「ネットワーク送信なし」違反。
    2. ``python -c`` ペイロードが構文エラー → compile() で検出。

    問題なければ ``None``。呼出側は ``None`` 以外なら is_executable=False に降格する。
    """
    lowered = command.lower()
    for marker in _NETWORK_EGRESS_MARKERS:
        if marker in lowered:
            return f"network egress not allowed (matched {marker!r})"
    payload = _extract_python_c_payload(command)
    if payload is not None:
        try:
            compile(payload, "<synthesized-command>", "exec")
        except SyntaxError as e:
            return f"python -c payload has SyntaxError: {e.msg}"
    return None


# ルールベースフォールバック用パターン
# 注意: 「検索」等の汎用語は知識質問にもマッチするため、
# コード/ファイル文脈を要求するパターンのみ含める。
_TOOL_PATTERNS = [
    re.compile(r"(?:ファイル|file).*(?:読|書|開|作成|削除)", re.IGNORECASE),
    re.compile(r"(?:コマンド|command).*(?:実行|run)", re.IGNORECASE),
    # コード/ファイル検索: 汎用「検索」は知識質問にマッチするため除外
    re.compile(r"(?:コード|ファイル|ソース|関数|クラス|code|file|source).*(?:検索|search|grep|find)", re.IGNORECASE),
    re.compile(r"(?:検索|search|grep|find).*(?:コード|ファイル|ソース|code|file|source)", re.IGNORECASE),
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
    re.compile(r"(?:スペック|CPU|メモリ|RAM|GPU|VRAM|ディスク|容量|ストレージ|ドライブ|(?<![A-Za-z])spec(?![A-Za-z])|(?<![A-Za-z])drive(?![A-Za-z]))", re.IGNORECASE),
    # 「何月|何日|何曜日」追加 (router.py:101 と同期)
    re.compile(r"(?:何時|何月|何日|何曜日|日時|日付|現在時刻|(?<![A-Za-z])today(?![A-Za-z])|(?<![A-Za-z])now(?![A-Za-z])|(?<![A-Za-z])date(?![A-Za-z])|(?<![A-Za-z])time(?![A-Za-z]))", re.IGNORECASE),
    re.compile(r"(?:IP\s*アドレス|ホスト名|(?<![A-Za-z])hostname(?![A-Za-z])|(?<![A-Za-z])ip\s*address)", re.IGNORECASE),
    re.compile(r"(?:(?<![A-Za-z])OS(?![A-Za-z])|オペレーティングシステム|(?<![A-Za-z])Windows(?![A-Za-z])|(?<![A-Za-z])Linux(?![A-Za-z])|(?<![A-Za-z])Mac(?![A-Za-z]))", re.IGNORECASE),
    re.compile(r"(?:Python|python)\s*(?:バージョン|version)", re.IGNORECASE),
    re.compile(r"(?:環境変数|(?<![A-Za-z])env(?![A-Za-z])|(?<![A-Za-z])PATH(?![A-Za-z]))", re.IGNORECASE),
    # --- Python 実行可能クエリ: 数値処理 ---
    re.compile(r"(?:階乗|素数|フィボナッチ|素因数|進数変換|桁)", re.IGNORECASE),
    # --- Python 実行可能クエリ: データ処理 ---
    re.compile(r"(?:集計|合計|平均|中央値|標準偏差|ソート|統計)", re.IGNORECASE),
    # --- Python 実行可能クエリ: 変換 ---
    re.compile(r"(?:変換|エンコード|デコード|Base64|ハッシュ|タイムスタンプ)", re.IGNORECASE),
]

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


def _build_spec_command(query: str) -> str:
    """システムスペックコマンドを生成する

    クエリにドライブレター指定（「Eドライブ」「C:」等）が含まれる場合は、
    そのドライブの容量を取得する。指定がなければカレントディレクトリ('.')。
    Windows / Unix の両方で動作するよう、パスはフォワードスラッシュで構築する
    （shutil.disk_usage は Windows でも 'E:/' を受理する）。
    """
    m = _DRIVE_LETTER_RE.search(query)
    if m:
        letter = m.group(1).upper()
        py_path = f"'{letter}:/'"
    else:
        py_path = "'.'"
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


# Python 実行で正確に答えられるシステム情報クエリのコマンドマッピング
# パターンにマッチしたクエリに対して、具体的な Python コマンドを生成する。
# コマンドは Windows cmd.exe / Unix sh の両方で動作するよう、
# 外側を "..." で囲み内側で '...' を使用する。
# 第二要素が Callable の場合はクエリ文字列を渡して動的に生成する
_EXECUTABLE_QUERY_COMMANDS: list[tuple[re.Pattern, "str | Callable[[str], str]"]] = [
    # 現在時刻 / 日付 (「何月|何日|何曜日」は明確な疑問語のみ追加、
    # 「今日|明日|昨日」単独は誤検出するため見送り)
    (re.compile(
        r"(?:何時|何月|何日|何曜日|日時|日付|現在時刻"
        r"|(?<![A-Za-z])today(?![A-Za-z])|(?<![A-Za-z])now(?![A-Za-z])"
        r"|(?<![A-Za-z])date(?![A-Za-z])|(?<![A-Za-z])time(?![A-Za-z]))",
        re.IGNORECASE,
    ), 'python -c "import datetime; print(datetime.datetime.now())"'),
    # システムスペック（CPU / メモリ / ディスク）
    # ドライブレター指定があれば指定ドライブの容量を返す
    (re.compile(
        r"(?:スペック|CPU|メモリ|RAM|ディスク|容量|ストレージ|ドライブ"
        r"|(?<![A-Za-z])spec(?![A-Za-z])"
        r"|(?<![A-Za-z])drive(?![A-Za-z]))",
        re.IGNORECASE,
    ), _build_spec_command),
    # GPU / VRAM
    (re.compile(r"(?:GPU|VRAM)", re.IGNORECASE),
     "python -c \""
     "import platform;"
     " print('Platform:',platform.platform());"
     " print('Machine:',platform.machine())"
     "\""),
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
       "import platform;"
       " print(platform.platform());"
       " print(platform.system(),platform.release())"
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


# アシストモデルへの executable command 合成プロンプト
# purpose=executable_command_synth で発火する。response_format
# (ExecutableCommandSynth) で {is_executable, command, rationale} の strict
# JSON を強制するため、ここでは出力フォーマットの厳密さよりも判定基準と
# コマンド生成のガイドラインを明示する。
_EXECUTABLE_COMMAND_SYNTH_SYSTEM_PROMPT = """\
# 実行可能クエリ判定 + コマンド合成

ユーザのクエリが「Python / シェルから取得できる環境依存の事実」かを判定し、
取得用の単一行コマンドを生成してください。

## is_executable=true の例
- 「今日は何月何日?」「現在時刻は?」 → 日時取得
- 「Python のバージョン」「Chrome のバージョン」 → ソフトウェアバージョン取得
- 「ディスク使用量」「CPU 情報」「メモリ使用量」 → システムリソース取得
- 「IP アドレス」「ホスト名」「OS の情報」 → ネットワーク / OS 情報

## is_executable=false の例
- 「Python のリスト内包表記とは?」 (知識質問 — コードや LLM 知識で答えるべき)
- 「次の会議の予定は?」 (個人スケジュール — コマンドで取れない)
- 「おすすめの設計パターン」 (意見・知識)
- 「ファイルを削除して」「リネームして」 (副作用を伴う操作 — tool_judgment 側で扱う)

## コマンド生成ルール (is_executable=true のとき)
- 単一行、副作用なし (書き込み・削除・ネットワーク送信・対話プロンプトなし)
- 30 秒以内に終了する
- 可搬性を優先して `python -c "..."` 形式を基本とする
- Windows 固有情報は PowerShell スニペットでも可
  (例: `powershell -Command "(Get-Item 'C:\\\\Program Files\\\\Google\\\\Chrome\\\\Application\\\\chrome.exe').VersionInfo.ProductVersion"`)
- shell の文法に従い、クオートを適切に処理する
- is_executable=false のとき command は ""

## rationale
1 行で判定理由を述べる (UI 非表示、ログ用)。

## 出力形式
JSON は **1 行のコンパクト形式** (改行・余分な空白なし) で出力する。
grammar 非強制モデルでも余計な空白で token を浪費して切り詰められないようにする。
"""


# 知識質問パターン — ツール判定をスキップすべきクエリ
_KNOWLEDGE_PATTERNS = [
    re.compile(r"(?:教えて|おしえて|とは|って何|ですか|でしょうか|ありますか)", re.IGNORECASE),
    re.compile(r"(?:について|に関して|に関する)", re.IGNORECASE),
    re.compile(r"(?:知りたい|確認したい|調べたい)", re.IGNORECASE),
    re.compile(r"(?:what is|tell me|explain|describe)\b", re.IGNORECASE),
]

# ユーザー自身の行動宣言パターン — アシスタントへの依頼ではない雑談発話。
# 「探してみるね」のような一人称の意思表明をツール起動と誤解しないための除外。
# 依頼形 (「探してみて(ください)」= て止め) とは区別する (こちらは末尾が「て」)。
_SELF_ACTION_PATTERNS = [
    # 「〜てみる(ね/よ/わ/かな/から)」自分で試す宣言 (文末)
    re.compile(r"(?:て|で)みる(?:ね|よ|わ|な|かな|から)?[\s　!！。.…]*$"),
    # 「〜しておく/やっておく/調べておく(ね/よ)」自己完結の行動宣言 (文末)
    re.compile(r"(?:してお|やってお|探してお|調べてお|見てお|やっと)く(?:ね|よ|わ|から)?[\s　!！。.…]*$"),
    # 一人称主語で自分が行う宣言
    re.compile(r"(?:自分で|自分が|私が|僕が|俺が|わたしが|こっちで|こちらで)"),
]


def _query_has_tool_signal(query: str) -> bool:
    """クエリにツール操作シグナル (ツールパターン / Windows・Unix パス / URL) を含むか。"""
    return (
        any(p.search(query) for p in _TOOL_PATTERNS)
        or bool(re.search(r"[A-Za-z]:\\|(?:^|[\s　])(?:/[\w._-]+){2,}|https?://", query))
    )


def _is_conversational_query_without_tool_signal(query: str) -> bool:
    """ツールシグナルが無く、知識質問 or 自己行動宣言の雑談発話か (assist 判定スキップ条件)。

    True の場合、層4 (tool_judgment) / 層5 (executable synth) の realtime
    assist 呼出を省いて no-tool 即決してよい。対象は:

    - 知識質問 (「とは」「教えて」等) — RAG / LLM 知識で答える
    - ユーザー自身の行動宣言 (「探してみるね」「自分で調べる」等) — 依頼ではない

    時刻・スペック等の実行可能事実クエリやファイルパス・URL を含むクエリは
    ``_TOOL_PATTERNS`` / パスシグナルで弾かれるため、誤って no-tool に倒さない。
    依頼形 (「探してみて」) は ``_SELF_ACTION_PATTERNS`` が文末「てみる」に限定して
    いるためマッチしない。
    """
    if _query_has_tool_signal(query):
        return False
    return (
        any(p.search(query) for p in _KNOWLEDGE_PATTERNS)
        or any(p.search(query) for p in _SELF_ACTION_PATTERNS)
    )


@dataclass
class ToolJudgement:
    """ツール呼び出し判定結果"""
    tool_needed: bool
    tool_name: str = ""
    tool_args: dict = None  # type: ignore[assignment]
    source: str = "rule"  # "assist" | "rule" | "cartridge" | "learned"

    def __post_init__(self):
        if self.tool_args is None:
            self.tool_args = {}


class ToolCallJudge:
    """アシストモデルによるツール呼び出し判定

    アシストモデルにクエリと利用可能ツール一覧を渡し、
    適切なツールの選択を判定させる。
    アシストモデルが利用不可の場合はルールベースにフォールバック。
    """

    def __init__(
        self,
        assist_client=None,
        prompt_manager=None,
        config: dict | None = None,
        cartridge_manager: CartridgeManager | None = None,
        learned_patterns: LearnedPatternStore | None = None,
        debug_logger: "DebugLogger | None" = None,
        mem_view: "MemFactView | None" = None,
        embedder: "EmbeddingBackend | None" = None,
        profile_id: str = "default",
    ):
        """
        Args:
            assist_client: AssistModelClient インスタンス（None でルールベースのみ）
            prompt_manager: AssistPromptManager インスタンス（None でデフォルトプロンプト使用）
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
                ツール呼出判定の 4 段階フォールバック (decision_point=
                ``tool_call_decision``、chosen=``rule``/``cartridge``/
                ``learned``/``assist``/``no_tool``) を ``decision.jsonl`` に
                記録する。``evolve`` レベル限定で実発火、それ以外は no-op。
        """
        self._assist_client = assist_client
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
        # executable_command_synth (アシスト経由) の結果キャッシュ。
        # query -> (timestamp, (is_executable, command))。LRU + TTL。
        # 同一クエリの重複 assist 呼出を抑制し、チャット応答パスの
        # レイテンシ増を最小化する。
        agent_cfg = self._config.get("agent", {})
        self._command_cache: OrderedDict[
            str, tuple[float, tuple[bool, str]]
        ] = OrderedDict()
        self._command_cache_max: int = int(
            agent_cfg.get("executable_command_cache_size", 64),
        )
        self._command_cache_ttl: float = float(
            agent_cfg.get("executable_command_cache_ttl_sec", 300.0),
        )
        # 5 層目 (executable command fallback) のゲートと pre-filter 閾値。
        # 4 層判定が全て no-tool を返した coding mode / tool_judge_enabled
        # パスで、assist (executable_command_synth) を試行するかどうか。
        self._executable_fallback_enabled: bool = bool(
            agent_cfg.get("executable_command_fallback_enabled", True),
        )
        self._executable_fallback_min_chars: int = int(
            agent_cfg.get("executable_command_fallback_min_chars", 5),
        )

    @property
    def enabled(self) -> bool:
        """ツール判定が有効かどうか。

        アシストモデル接続時はモード非依存で **既定有効**。
        ``agent.tool_judge_enabled=false`` で明示的にオプトアウトできる。
        アシスト未接続 (degraded) 時は常に無効で、決定論ショートカット
        (明示 URL / 実行可能事実コマンド) のみに縮退する。
        """
        if self._assist_client is None:
            return False
        return self._config.get("agent", {}).get("tool_judge_enabled", True)

    def _speculation_enabled(self) -> bool:
        """層4 (tool_judgment) と層5 (executable_command_synth) の投機並走可否.

        assist の realtime セマフォが 2 以上 (= チャット応答パスの並列化が
        有効) のときだけ True。セマフォ=1 のまま投機すると層5 synth が先に
        スロットを取って層4 judgment を遅らせる優先度逆転が起きるため、
        並列度が確保された構成でのみ投機する。
        """
        return getattr(self._assist_client, "realtime_concurrency", 1) >= 2

    async def judge(
        self,
        query: str,
        tools_registry: ToolsRegistry,
        mode: str = "coding",
        conversation: list[dict] | None = None,
    ) -> ToolJudgement:
        """ツール呼び出しの要否を判定

        判定は安価な順に実行し、最初にマッチした結果を返す:
        1. 組み込みパターン照合（ルールベース）
        2. カートリッジ tool_hints 照合
        3. アシストモデル判定（LLM）

        コーディングモードでは tool_judge_enabled が false でも
        ルールベース + カートリッジ hints 判定を実行する。

        Args:
            query: ユーザーのクエリ
            tools_registry: 利用可能なツールレジストリ
            mode: 動作モード ('chat' | 'coding')
            conversation: 直近の会話履歴（判定精度向上のため）

        Returns:
            ToolJudgement
        """
        # 0. URL リコール先回り判定 (mode / enabled に関係なく実行)
        # ``_try_recall_url`` は決定論的 (embedding 類似度 + 過去採点平均閾値)
        # で、アシスト同期発火やルール正規表現のような副作用がない。早期 return
        # 経路 (chat モード + tool_judge_enabled=false) で判定がスキップされる
        # と「過去 URL は SemMem にあるのに fetch されない」という不整合が起きる
        # ため、ここで先回りで引き当てる。
        url_recall_result = await self._judge_with_url_recall(
            query, tools_registry, mode=mode,
        )
        if url_recall_result is not None:
            self._log_tool_decision(url_recall_result, "url_recall_matched")
            return url_recall_result

        # 0.5. executable command リコール先回り判定 (mode / enabled 非依存)
        # 過去成功した run_command を SemMem から決定論的に引き当てる。URL
        # リコールと同様、chat early-return / coding 4 層のどちらに入る前にも
        # 短絡させることで、学習済みクエリでは assist (合成 / 5 層目) を一切
        # 呼ばずにコマンドを確定できる。
        cmd_recall_result = await self._judge_with_executable_command_recall(
            query, tools_registry, mode=mode,
        )
        if cmd_recall_result is not None:
            self._log_tool_decision(
                cmd_recall_result, "executable_command_recall_matched",
            )
            return cmd_recall_result

        # config で明示的に無効化 → chat モードでは判定スキップ
        # coding モードではルールベース + カ���トリッジ hints にフ��ールバック
        # ただし chat モードでも executable query（時刻・スペック等）は
        # 安全なコマンド生成のみ行うためルールベース判定を許可する
        if not self.enabled and mode != "coding":
            # クエリに URL が明示的に含まれる場合は tool_judge_enabled に
            # 関係なく fetch_url を返す。ユーザが URL を書く = 「これを読んで」
            # の強い意図表明であり、LLM 判断を仰がず決定論的に拾う
            url_match = _URL_IN_QUERY_RE.search(query)
            if url_match and tools_registry.has("fetch_url"):
                logger.debug(
                    "Chat mode explicit URL detected: %s", query[:50],
                )
                result = ToolJudgement(
                    tool_needed=True,
                    tool_name="fetch_url",
                    tool_args={"url": url_match.group(1)},
                    source="rule",
                )
                self._log_tool_decision(result, "chat_mode_explicit_url")
                return result

            # assist 優先で実行可能コマンドを解決する。assist 未接続 / 失敗
            # 時はモジュールレベル sync 関数 (regex テーブル) にフォールバック。
            command = await self._resolve_executable_command(query)
            if command and tools_registry.has("run_command"):
                logger.debug(
                    "Chat mode executable query detected: %s", query[:50],
                )
                # assist が解決したか regex フォールバックかは
                # _command_cache_lookup の有無で判別可能だが、source は
                # ログ用途のため簡略化して "rule" のまま返す
                # (キャッシュヒット = assist 由来でも "rule" でログされる)。
                return ToolJudgement(
                    tool_needed=True,
                    tool_name="run_command",
                    tool_args={"command": command},
                    source="rule",
                )
            return ToolJudgement(tool_needed=False, source="rule")

        # 1. 組み込みパターン照合（ルールベース）
        result = self._judge_with_rules(query, tools_registry, mode)
        if result.tool_needed:
            # run_command の場合は assist でコマンド合成を試行 (assist=None
            # / 失敗時は regex 由来の既存コマンドを維持)
            result = await self._maybe_upgrade_command_via_assist(result, query)
            await self._maybe_recall_url(result, query, mode=mode)
            result = self._suppress_unfetchable_fetch_url(result)
            self._log_tool_decision(result, "rule_pattern_matched")
            return result

        # 2. カートリッジ tool_hints 照合
        result = self._judge_with_cartridge_hints(query, tools_registry)
        if result.tool_needed:
            result = self._suppress_unfetchable_fetch_url(result)
            self._log_tool_decision(result, "cartridge_hint_matched")
            return result

        # 3. 学習済みパターン照合
        result = self._judge_with_learned_patterns(query, tools_registry, mode)
        if result.tool_needed:
            result = await self._maybe_upgrade_command_via_assist(result, query)
            await self._maybe_recall_url(result, query, mode=mode)
            result = self._suppress_unfetchable_fetch_url(result)
            self._log_tool_decision(result, "learned_pattern_matched")
            return result

        # 雑談プレフィルタ: ツールシグナルが無く、知識質問 or ユーザー自身の
        # 行動宣言 (「探してみるね」等) のクエリは層4 (tool_judgment) をスキップ
        # する (詳細は下の try 内コメント)。
        skip_judgment = _is_conversational_query_without_tool_signal(query)
        if skip_judgment:
            logger.debug(
                "Skipping tool_judgment for conversational query w/o tool signal: %s",
                query[:50],
            )
        run_layer4 = (
            self.enabled and self._assist_client is not None and not skip_judgment
        )

        # 投機実行: 層4 を走らせる場合かつ realtime 並列度が確保されている
        # とき、層5 (executable_command_synth) を先回りで起動して層4 と
        # 並走させる。層4 が tool を返したら finally で破棄、no-tool なら
        # await して回収する。層4 を走らせない (skip_judgment / 無効) 場合は
        # 層5 が最初の assist 呼出なので投機の意味がなく、直列実行する。
        spec_task: asyncio.Task | None = None
        if run_layer4 and self._speculation_enabled():
            spec_task = asyncio.create_task(
                self._judge_with_executable_fallback(query, tools_registry),
            )
        try:
            # 4. アシストモデル判定（有効時のみ）
            # tool を選んだ場合のみ即 return する。no-tool の場合は 5 層目
            # (executable command fallback) に fall-through させる。
            #
            # 雑談プレフィルタ: ツールシグナルが無く、知識質問 or ユーザー自身の
            # 行動宣言 (「探してみるね」等) のクエリは、広範な tool_judgment 呼出を
            # スキップする。アシスト接続時の常時有効化で純粋な雑談のたびに
            # tool_judgment を呼ぶ無駄打ちと誤発火 (例: 「探してみるね」→ search_history)
            # を抑える狙い。ただし 5 層目 (executable_command_synth) はスキップしない
            # — 「Chrome のバージョン教えて」のように雑談風だが実行可能な事実クエリを
            # 拾うため (旧 chat early-return の synth 呼出と同等のベースラインを維持)。
            if run_layer4:
                try:
                    assist_result = await self._judge_with_assist(
                        query, tools_registry, mode, conversation,
                    )
                    assist_result = self._suppress_unfetchable_fetch_url(
                        assist_result,
                    )
                    if assist_result.tool_needed:
                        self._log_tool_decision(assist_result, "assist_judgement")
                        return assist_result  # finally が spec_task を cancel
                except Exception as e:
                    logger.warning(
                        "Assist model tool judgement failed: %r", e,
                    )

            # 5. executable command フォールバック (環境依存事実クエリの救済)
            # 4 層全てが no-tool を返した場合、assist (executable_command_synth)
            # で「これは run_command で取れる事実か?」を判定する。Free chat mode
            # は judge() 冒頭の early-return で既に処理済みのため、ここに到達する
            # のは coding mode / tool_judge_enabled=True パス。
            # 投機タスクがあれば await で回収、なければ直列実行する。
            if spec_task is not None:
                task, spec_task = spec_task, None  # 消費済 (finally で cancel しない)
                try:
                    fallback_result = await task
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.warning(
                        "Speculative executable_command_synth failed: %r", e,
                    )
                    fallback_result = None
            else:
                fallback_result = await self._judge_with_executable_fallback(
                    query, tools_registry,
                )
            if fallback_result is not None:
                self._log_tool_decision(
                    fallback_result, "executable_command_fallback",
                )
                return fallback_result

            # 6. 全フォールバック失敗時の no_tool 結末を記録
            no_tool_result = ToolJudgement(tool_needed=False, source="rule")
            self._log_tool_decision(
                no_tool_result, "no_match_in_any_layer",
            )
            return no_tool_result
        finally:
            # 層4 が tool を返した / 例外で抜けた場合、投機タスクが未消費なら
            # 破棄する。キャンセルされた synth は ``_synthesize_command_via_assist``
            # の ``except Exception`` が CancelledError を捕まえないため LRU
            # キャッシュを汚さない (= 失敗時キャッシュなしの現行挙動と同じ)。
            # なお非ストリーミング assist 呼出はサーバ側で切断検知できず、
            # 破棄した synth が最大 max_tokens 分 slot を占有しうるが、
            # slots>=2 構成が前提のため他スロットは空く。
            if spec_task is not None and not spec_task.done():
                spec_task.cancel()

    async def _judge_with_executable_fallback(
        self, query: str, tools_registry: ToolsRegistry,
    ) -> "ToolJudgement | None":
        """4 層判定後の executable command フォールバック (5 層目).

        環境依存事実クエリ (「Chrome のバージョン」「インストール済み pip
        パッケージ」等) を assist (executable_command_synth) で判定し、
        実行可能なら ``run_command`` 判定を返す。

        Pre-filter で挨拶 / 短文 / 機能 OFF / degraded mode を弾いてから
        assist を呼ぶ。assist 呼出は ``_synthesize_command_via_assist`` を
        再利用するため、Free chat mode の early-return と LRU キャッシュを
        共有する (同一 query は重複呼出ゼロ)。

        Returns:
            実行可能と判定された場合 ``ToolJudgement(run_command, ...)``。
            機能 OFF / pre-filter 弾き / assist が否定 / assist 失敗の場合は
            ``None`` (呼出側は no-tool として処理する)。
        """
        if not self._executable_fallback_enabled:
            return None
        if not tools_registry.has("run_command"):
            return None
        if self._assist_client is None:
            return None
        stripped = query.strip()
        if len(stripped) < self._executable_fallback_min_chars:
            return None
        # 挨拶は assist を呼ばずにスキップ (reactive 層の定型応答対象)
        if any(p.match(stripped) for p, _ in GREETING_RESPONSES):
            return None

        synth = await self._synthesize_command_via_assist(query)
        if synth is None:
            return None
        is_exec, command = synth
        if not is_exec or not command:
            return None
        logger.info(
            "Executable command synthesized via 5th layer fallback: query=%s",
            query[:50],
        )
        return ToolJudgement(
            tool_needed=True,
            tool_name="run_command",
            tool_args={"command": command},
            source="assist",
        )

    def _suppress_unfetchable_fetch_url(
        self, result: "ToolJudgement",
    ) -> "ToolJudgement":
        """url 引数を補完できない ``fetch_url`` を ``tool_needed=False`` に格下げ.

        URL リコール (``_maybe_recall_url`` / ``_judge_with_url_recall``) でも
        補完できず、アシスト判定でも url が validate されなかったケースを
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

    async def _judge_with_url_recall(
        self,
        query: str,
        tools_registry: ToolsRegistry,
        mode: str = "coding",
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
        mode: str = "coding",
    ) -> "ToolJudgement | None":
        """外部から URL recall のみを問い合わせる公開 API.

        Reactive レイヤの escalation 判定など、judge() フル実行前に
        「URL recall だけ」をチェックしたい呼び元向け。判定本体は
        ``_judge_with_url_recall`` を共有し、戻り値も同じ。
        """
        return await self._judge_with_url_recall(query, tools_registry, mode=mode)

    async def _maybe_recall_url(
        self, result: "ToolJudgement", query: str, mode: str = "coding",
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

    async def _try_recall_url(self, query: str, mode: str = "coding") -> str | None:
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
        try:
            embeddings = await self._embedder.embed([query], is_query=True, mode=mode)
        except Exception as exc:
            logger.warning("URL recall: embed failed: %s", exc)
            return None
        if embeddings is None or len(embeddings) == 0:
            return None
        import numpy as _np
        q_vec = _np.asarray(embeddings[0], dtype=_np.float32)

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
        mode: str = "coding",
    ) -> "ToolJudgement | None":
        """SemMem の過去成功コマンド引き当てで run_command 判定を返す.

        ``_judge_with_url_recall`` と対称。条件:
          - ``run_command`` ツールが登録済み
          - ``mem_view`` / ``embedder`` が wired されている
          - ``_try_recall_executable_command`` が閾値判定でコマンドを返す

        引き当てが成立すれば assist 呼出 (5 層目 / chat early-return) より
        先に確定するため、学習済みクエリでは LLM コストがゼロになる。

        Returns:
            引き当て成立時は ``ToolJudgement(run_command, {"command": ...})``、
            それ以外は ``None`` (通常の判定フローに falling-through する)。
        """
        if not tools_registry.has("run_command"):
            return None
        if self._mem_view is None or self._embedder is None:
            return None
        recalled = await self._try_recall_executable_command(query, mode=mode)
        if not recalled:
            return None
        logger.info(
            "Executable command recall: matched command for query=%s",
            query[:50],
        )
        return ToolJudgement(
            tool_needed=True,
            tool_name="run_command",
            tool_args={"command": recalled},
            source="rule",
        )

    async def _try_recall_executable_command(
        self, query: str, mode: str = "coding",
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
        try:
            embeddings = await self._embedder.embed([query], is_query=True, mode=mode)
        except Exception as exc:
            logger.warning("Executable command recall: embed failed: %s", exc)
            return None
        if embeddings is None or len(embeddings) == 0:
            return None
        import numpy as _np
        q_vec = _np.asarray(embeddings[0], dtype=_np.float32)

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

        import time as _time
        now = _time.time()

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

            if command and effective_score >= min_avg:
                return str(command)
        return None

    def _log_tool_decision(
        self, result: "ToolJudgement", reason: str,
    ) -> None:
        """

        chosen は ``rule`` / ``cartridge`` / ``learned`` / ``assist`` /
        ``no_tool`` のいずれかで、4 段階フォールバックのどの層で決着したかを
        identify する。``evolve`` レベル限定で実発火、それ以外は no-op。
        """
        if self._debug_logger is None:
            return
        chosen = "no_tool" if not result.tool_needed else (result.source or "rule")
        self._debug_logger.log_decision(
            decision_point="tool_call_decision",
            chosen=chosen,
            candidates=["rule", "cartridge", "learned", "assist", "no_tool"],
            reason=reason,
            context={
                "tool_needed": bool(result.tool_needed),
                "tool_name": getattr(result, "tool_name", None) or "",
            },
            scope="request",
        )

    async def _judge_with_assist(
        self,
        query: str,
        tools_registry: ToolsRegistry,
        mode: str,
        conversation: list[dict] | None,
    ) -> ToolJudgement:
        """アシストモデルでツール呼び出しを判定"""
        # プロンプト構築
        system_prompt = self._build_system_prompt(tools_registry, mode)
        user_prompt = self._build_user_prompt(query, conversation)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        result = await self._assist_client.generate(
            messages,
            # 256: E4B 等 json_schema grammar 非強制モデルが空白パディング JSON
            # を返すと 128 token では閉じ括弧前に finish_reason=length で切れる。
            # executable_command_synth (256) / conflict_chat_judge (192) と整合。
            max_tokens=256,
            temperature=0.1,
            purpose="tool_judgment",
        )
        content = (
            result.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )

        logger.debug("Assist model tool judgement response: %s", content[:200])

        judgement = self._parse_response(content)
        # アシストが fetch_url を選び url 引数を返した場合の hallucination 検証.
        # URL hallucination (例: クエリ「大阪の天気」に対して /jp/osaka/ という
        # 実在しない path を生成) を防ぐため、url がクエリ内に逐語的に存在する
        # ものでなければ破棄する。URL リコール経路 (mem_view 経由) は別ルート
        # で動くため、ここで url を空にしても問題ない。
        if judgement.tool_needed and judgement.tool_name == "fetch_url":
            args = judgement.tool_args or {}
            url = args.get("url")
            if url and not _url_appears_in_query(url, query):
                logger.warning(
                    "Assist returned URL not found in query (likely hallucination), "
                    "dropping url arg: url=%s", url,
                )
                judgement.tool_args = {
                    k: v for k, v in args.items() if k != "url"
                }
        return judgement

    def _command_cache_lookup(self, query: str) -> tuple[bool, str] | None:
        """executable_command_synth キャッシュから取得する.

        TTL を超過していたら削除して None を返す。LRU 末尾に move する。
        """
        key = query.strip().lower()
        if key not in self._command_cache:
            return None
        ts, value = self._command_cache[key]
        if time.time() - ts > self._command_cache_ttl:
            del self._command_cache[key]
            return None
        self._command_cache.move_to_end(key)
        return value

    def _command_cache_store(
        self, query: str, value: tuple[bool, str],
    ) -> None:
        """executable_command_synth キャッシュに格納する (LRU)."""
        key = query.strip().lower()
        self._command_cache[key] = (time.time(), value)
        self._command_cache.move_to_end(key)
        while len(self._command_cache) > self._command_cache_max:
            self._command_cache.popitem(last=False)

    async def _synthesize_command_via_assist(
        self, query: str,
    ) -> tuple[bool, str] | None:
        """アシストモデル (purpose=executable_command_synth) で
        ``(is_executable, command)`` を取得する.

        Returns:
            成功時 ``(is_executable, command)``。``is_executable=False`` の
            場合 ``command`` は ``""``。
            アシスト未接続 / 通信失敗 / JSON パース失敗時は ``None``
            (呼出側はハードコード regex (`_infer_executable_command`) に
            フォールバックする)。
        """
        if self._assist_client is None:
            return None
        cached = self._command_cache_lookup(query)
        if cached is not None:
            return cached

        import platform
        platform_info = f"{platform.system()} {platform.release()}"
        user_prompt = (
            f"## プラットフォーム\n{platform_info}\n\n"
            f"## クエリ\n{query}"
        )
        try:
            result = await self._assist_client.generate(
                [
                    {
                        "role": "system",
                        "content": _EXECUTABLE_COMMAND_SYNTH_SYSTEM_PROMPT,
                    },
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=256,
                temperature=0.1,
                purpose="executable_command_synth",
            )
        except Exception as e:
            logger.warning(
                "executable_command_synth assist call failed: %r", e,
            )
            return None
        content = (
            result.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        data = extract_json_object(content)
        if not isinstance(data, dict):
            logger.warning(
                "executable_command_synth assist returned non-JSON: %s",
                content[:120],
            )
            return None
        is_executable = bool(data.get("is_executable", False))
        command = str(data.get("command", "")).strip()
        # is_executable=True と主張するが command が空のケースは
        # 「コマンド合成失敗」とみなして False に降格する。
        if not is_executable or not command:
            value = (False, "")
        else:
            # synth プロンプトのスコープ (ネットワーク送信なし・構文妥当) を
            # コード側で enforce。逸脱コマンドは実行させず False に降格する。
            reject = _reject_synthesized_command(command)
            if reject is not None:
                logger.warning(
                    "executable_command_synth rejected command (%s): %s",
                    reject, command[:120],
                )
                value = (False, "")
            else:
                value = (True, command)
        self._command_cache_store(query, value)
        return value

    async def _resolve_executable_command(self, query: str) -> str:
        """assist を優先して executable command を解決する.

        Returns:
            実行可能と判定された場合のコマンド文字列。それ以外は ``""``。

        優先順:
            1. アシストモデル (purpose=executable_command_synth):
               - ``(True, cmd)`` → ``cmd`` を返す
               - ``(False, "")`` → ``""`` を返す (assist が明確に NO と判定)
            2. assist 未接続 / 失敗時はモジュールレベル sync 関数
               ``_infer_executable_command`` (regex テーブル) にフォールバック。
        """
        synth = await self._synthesize_command_via_assist(query)
        if synth is not None:
            is_exec, command = synth
            return command if is_exec else ""
        return _infer_executable_command(query)

    async def _maybe_upgrade_command_via_assist(
        self, result: "ToolJudgement", query: str,
    ) -> "ToolJudgement":
        """rule / learned 層が ``run_command`` を返した場合に、
        コマンド引数を assist で生成した値に置換する.

        判定ロジック:
            - ``run_command`` 以外のツール → 何もしない
            - assist が ``(True, cmd)`` を返す → コマンドを置換し
              ``source="assist"`` に更新
            - assist が ``(False, "")`` を返す → 既存 regex 結果を維持
              (regex が specific パターンにマッチした confidence を尊重)
            - assist 失敗 / 未接続 → 既存結果をそのまま返す
        """
        if not result.tool_needed or result.tool_name != "run_command":
            return result
        synth = await self._synthesize_command_via_assist(query)
        if synth is None:
            return result
        is_exec, command = synth
        if is_exec and command:
            new_args = dict(result.tool_args or {})
            new_args["command"] = command
            return ToolJudgement(
                tool_needed=True,
                tool_name="run_command",
                tool_args=new_args,
                source="assist",
            )
        return result

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
        # 知識質問はツール不要（RAG パイプラインで処理）
        # ただしツールパターン・ファイルパス・URL にもマッチするクエリは
        # ツール操作の可能性が高いため知識質問判定を適用しない
        has_tool_signal = _query_has_tool_signal(query)
        if not has_tool_signal and any(p.search(query) for p in _KNOWLEDGE_PATTERNS):
            logger.debug("Rule-based: knowledge query detected, skipping tool: %s", query[:50])
            return ToolJudgement(tool_needed=False, source="rule")

        if not any(p.search(query) for p in _TOOL_PATTERNS):
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
        if re.search(
            r"(?:読[みむ]込|読んで|開いて|見せて|見て|確認|チェック|確かめ"
            r"|正し[いく]|合って|内容|中身"
            r"|read|show|check|verify|correct|content|view)",
            q,
        ):
            path = _extract_file_path(query)
            if path:
                # ディレクトリ指定 (配下のファイルを点検する文脈) は read_file だと
                # "Not a file" になるため list_directory に振り分ける。
                if Path(path).is_dir() and tools_registry.has("list_directory"):
                    return "list_directory", {"directory": path}
                if tools_registry.has("read_file"):
                    return "read_file", {"file_path": path}

        # ファイル書き込み/出力パターン
        # ディレクトリを書込み先に取ると write_file が配下に output_<UTC>.txt を
        # 捏造する (記述的な「出力」誤マッチで read 指示がここへ落ちるケースを含む)。
        # ディレクトリは書込み対象から除外する。
        if re.search(r"(?:書[きく]込|書いて|出力|保存|生成|作成|write|save|output)", q):
            path = _extract_file_path(query)
            if path and not Path(path).is_dir() and tools_registry.has("write_file"):
                return "write_file", {"file_path": path}

        # コマンド実行パターン
        if re.search(r"(?:実行|run|exec)", q) and tools_registry.has("run_command"):
            # ファイルパスがあれば python で実行
            path = _extract_file_path(query)
            if path and path.endswith(".py"):
                return "run_command", {"command": f'python "{path}"'}
            # バッククォート内のコマンド
            cmd_match = re.search(r'`([^`]+)`', query)
            if cmd_match:
                return "run_command", {"command": cmd_match.group(1)}
            return "run_command", {}

        # コード検索パターン
        if re.search(r"(?:検索|search|grep|find)", q) and tools_registry.has("search_code"):
            # クエリからキーワードを抽出して pattern 引数に設定
            pattern = _extract_search_pattern(query)
            if pattern:
                return "search_code", {"pattern": pattern}
            return "search_code", {}

        # 計算パターン
        if re.search(r"(?:計算|calculate)", q) and tools_registry.has("calculate"):
            return "calculate", {}

        # Python 実行可能クエリ（システム情報・数値処理・データ処理・変換）
        # これらのクエリは Python コード生成 → run_command で正確に回答できる
        if re.search(
            r"(?:スペック|CPU|メモリ|RAM|GPU|VRAM|ディスク|容量|ストレージ"
            r"|(?<![A-Za-z])spec(?![A-Za-z])"
            r"|何時|何月|何日|何曜日|日時|日付|現在時刻"
            r"|(?<![A-Za-z])today(?![A-Za-z])|(?<![A-Za-z])now(?![A-Za-z])"
            r"|IP\s*アドレス|ホスト名|(?<![A-Za-z])hostname(?![A-Za-z])"
            r"|(?<![A-Za-z])OS(?![A-Za-z])|オペレーティングシステム"
            r"|(?<![A-Za-z])Windows(?![A-Za-z])|(?<![A-Za-z])Linux(?![A-Za-z])"
            r"|(?<![A-Za-z])Mac(?![A-Za-z])"
            r"|Python\s*(?:バージョン|version)"
            r"|環境変数|(?<![A-Za-z])env(?![A-Za-z])"
            r"|階乗|素数|フィボナッチ|素因数|進数変換|桁"
            r"|集計|合計|平均|中央値|標準偏差|ソート|統計"
            r"|変換|エンコード|デコード|Base64|ハッシュ|タイムスタンプ)",
            q,
            re.IGNORECASE,
        ) and tools_registry.has("run_command"):
            # システム情報クエリは具体的なコマンドを生成
            command = _infer_executable_command(query)
            if command:
                return "run_command", {"command": command}
            return "run_command", {}

        return "", {}

    def _build_system_prompt(
        self,
        tools_registry: ToolsRegistry,
        mode: str,
    ) -> str:
        """ツール判定用システムプロンプトを構築"""
        # AssistPromptManager からタスク別プロンプトを取得
        if self._prompt_manager is not None:
            try:
                base_prompt = self._prompt_manager.get_assist_prompt("tool_call")
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
                content = msg.get("content", "")[:100]
                if role in ("user", "assistant"):
                    context_lines.append(f"{role}: {content}")
            if context_lines:
                parts.append("## 直近の会話\n" + "\n".join(context_lines))

        parts.append(f"## ユーザーのリクエスト\n{query}")
        return "\n\n".join(parts)

    def _parse_response(self, content: str) -> ToolJudgement:
        """アシストモデルの応答をパースして ToolJudgement に変換

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
            "Could not parse assist model response for tool judgement: %s",
            content[:100],
        )
        return ToolJudgement(tool_needed=False, source="assist")


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
_DIR_PATH_RE = re.compile(
    r"([A-Za-z]:(?:\\[A-Za-z0-9_.][A-Za-z0-9_. -]*)*)",
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

    # 1. 明示的なフルパス: C:\Users\file.txt（ASCII 文字のみのパス部分）
    m = re.search(r"[A-Za-z]:\\[A-Za-z0-9_.\\/ -]+\.[A-Za-z0-9]{1,10}", query)
    if m:
        return _normalize_path_separators(m.group(0).rstrip(" "))

    # 2. ドライブレター + 自然言語でのファイル名指定
    #    例: 「e:\直下にa.txtのファイル名で」→ e:\a.txt
    #    ディレクトリとファイル名が日本語/全角スペースで分断されていても、
    #    ディレクトリ部 (Pattern 3 と同じ捕捉) を取り出してファイル名と結合し、
    #    サブ階層を保持する。深い階層が無い (ドライブ直下指定) 場合のみ
    #    従来どおりドライブ直下へフォールバックする。
    #    \w は日本語にもマッチするため ASCII 限定で検索
    drive_match = re.search(r"([A-Za-z]):\\", query)
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
            directory = _normalize_path_separators(
                dir_match.group(1).rstrip(),
            ).rstrip("\\/")
            return f"{directory}\\{filename}"
        return f"{drive_match.group(1)}:\\{filename}"

    # 3. ディレクトリパスのみ（ファイル名なし）: E:\xxx\ や E:\xxx 等
    #    配下のファイルを参照する文脈では、ディレクトリパスを返す。
    #    全角スペース (U+3000) 等の Unicode 空白や文末で終端しても、
    #    セグメント単位で解析する _DIR_PATH_RE が自然に正しい境界で止まる。
    if drive_match:
        dir_match = _DIR_PATH_RE.search(query)
        if dir_match:
            return _normalize_path_separators(dir_match.group(1).rstrip())

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


def _url_appears_in_query(url: str, query: str) -> bool:
    """``url`` が ``query`` 内にそのまま含まれているか判定する.

    アシスト LLM の URL hallucination を検出する用途。完全一致 (substring)
    のみを許可し、ホスト名だけマッチするようなゆるい検出は意図的に避ける。

    Returns:
        url がクエリの substring として現れていれば ``True``、それ以外は
        ``False``。``url`` が空 / None なら ``False``。
    """
    if not url or not query:
        return False
    return url in query


def _normalize_path_separators(path: str) -> str:
    """連続バックスラッシュをシングルに正規化する

    LLM や JSON パース経由でパスが二重エスケープされるケースに対応。
    例: E:\\\\xxx\\\\tetris.py → E:\\xxx\\tetris.py
    """
    # 連続する2つ以上の \ を1つに置換
    return re.sub(r"\\{2,}", r"\\", path)


def _json_to_judgement(data: dict) -> ToolJudgement:
    """JSON dict を ToolJudgement に変換

    アシスト応答は ``response_format`` 無効 / 古い llama-server / max_tokens 切断
    時に ``json_repair`` で機械修復されるため、``tool`` / ``args`` が非想定型
    (list / str 等) になりうる。``ToolJudgement.tool_args`` は dict 契約なので、
    下流 (``deliberative._execute_tool`` の ``dict(tool_args)`` 等) が落ちないよう
    ここで強制正規化する。
    """
    tool = data.get("tool", "")
    if not isinstance(tool, str):
        tool = ""
    if not tool or tool == "no_tool":
        return ToolJudgement(tool_needed=False, source="assist")
    args = data.get("args", {})
    if not isinstance(args, dict):
        args = {}
    return ToolJudgement(
        tool_needed=True,
        tool_name=tool,
        tool_args=args,
        source="assist",
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
