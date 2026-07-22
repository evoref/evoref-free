"""アシストモデルによるツール呼び出し判定（Free/Pro 共通）

ユーザークエリと利用可能なツール一覧をアシストモデルに提示し、
ツール呼び出しの要否・ツール名・引数を判定する。
アシストモデル未接続時はルールベースにフォールバックする。
"""

from __future__ import annotations

import asyncio
import re
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from backend.free.agent.reactive import GREETING_RESPONSES, GREETING_RESPONSES_EN
from backend.free.agent.router import HISTORY_KEYWORDS, HISTORY_KEYWORDS_EN
from backend.free.core.locale_patterns import is_en_locale, select_locale_variant
from backend.free.core.session_mode import is_coding_mode
from backend.free.agent.safety_patterns import (
    extract_python_c_payload,
    reject_readonly_violation,
)
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


# 実装本体は safety_patterns へ移動 (readonly ガードと共有)。既存テスト /
# 呼出側の互換のためエイリアスを残す。
_extract_python_c_payload = extract_python_c_payload


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


def _executable_tool_for_mode(tools_registry: ToolsRegistry, mode: str) -> str:
    """現在の ``mode`` で使える executable コマンドツール名を解決する。

    coding では従来の ``run_command``、chat では読み取り専用の
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


def _readonly_command_rejected(exec_tool: str, command: str) -> bool:
    """readonly ツールに載せる ``command`` が readonly 検証に違反するか。

    ``exec_tool`` が ``run_command_readonly`` のときだけ
    ``reject_readonly_violation`` を適用する (coding の run_command は対象外)。
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
    re.compile(r"(?:スペック|(?<![A-Za-z])CPU(?![A-Za-z])|メモリ|(?<![A-Za-z])RAM(?![A-Za-z])|(?<![A-Za-z])GPU(?![A-Za-z])|(?<![A-Za-z])VRAM(?![A-Za-z])|ディスク|容量|ストレージ|ドライブ|(?<![A-Za-z])spec(?![A-Za-z])|(?<![A-Za-z])drive(?![A-Za-z]))", re.IGNORECASE),
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
    re.compile(r"(?:何時|何月|何日|何曜日|日時|日付|現在時刻|(?<![A-Za-z])today(?![A-Za-z])|(?<![A-Za-z])now(?![A-Za-z])|(?<![A-Za-z])date(?![A-Za-z])|(?<![A-Za-z])time(?![A-Za-z]))", re.IGNORECASE),
    re.compile(r"(?:IP\s*address|hostname|(?<![A-Za-z])hostname(?![A-Za-z])|(?<![A-Za-z])ip\s*address)", re.IGNORECASE),
    re.compile(r"(?:(?<![A-Za-z])OS(?![A-Za-z])|operating\s*system|(?<![A-Za-z])Windows(?![A-Za-z])|(?<![A-Za-z])Linux(?![A-Za-z])|(?<![A-Za-z])Mac(?![A-Za-z]))", re.IGNORECASE),
    re.compile(r"(?:Python|python)\s*version", re.IGNORECASE),
    re.compile(r"(?:environment\s*variable|(?<![A-Za-z])env(?![A-Za-z])|(?<![A-Za-z])PATH(?![A-Za-z]))", re.IGNORECASE),
    # --- Python 実行可能クエリ: 数値処理 (171行目相当の英語版) ---
    re.compile(r"\b(?:factorial|prime(?:\s*numbers?)?|fibonacci|prime\s*factorization|base\s*conversion|number\s*of\s*digits?|digits?)\b", re.IGNORECASE),
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
    re.compile(r"\b(?:convert|encode|decode|encoding|decoding|Base64|hash(?:ing)?|timestamp)\b", re.IGNORECASE),
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
    r"|(?<![A-Za-z])GPU(?![A-Za-z])|(?<![A-Za-z])VRAM(?![A-Za-z])|ディスク|容量|ストレージ"
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
    r"|(?<![A-Za-z])disk(?![A-Za-z])|(?<![A-Za-z])capacity(?![A-Za-z])|(?<![A-Za-z])storage(?![A-Za-z])"
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
    # CPU/RAM 等の英字略語は ASCII 境界必須 ("program" の 'ram' 誤マッチ対策)
    # spec(s)? で複数形 ("PC specs") も許容する。
    (re.compile(
        r"(?:スペック|(?<![A-Za-z])CPU(?![A-Za-z])|メモリ|memory"
        r"|(?<![A-Za-z])RAM(?![A-Za-z])|ディスク|容量|ストレージ|ドライブ"
        r"|disk|capacity|storage"
        r"|(?<![A-Za-z])specs?(?![A-Za-z])"
        r"|(?<![A-Za-z])drive(?![A-Za-z]))",
        re.IGNORECASE,
    ), _build_spec_command),
    # GPU / VRAM
    (re.compile(
        r"(?:(?<![A-Za-z])GPU(?![A-Za-z])|(?<![A-Za-z])VRAM(?![A-Za-z]))",
        re.IGNORECASE,
    ),
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
# 「簡単に説明してください」「〜の使い分けは？」は疑問形の末尾 (教えて/ですか等)
# を伴わない体言止め・依頼形のため、上のパターンにマッチせず層4 (assist
# tool_judgment) まで素通りしていた。実インシデント (2026-07-20):
# 「カーディネーリティという言葉を初めて知りました。簡単に説明してください。」
# で無関係な過去セッションが誤ヒット (score=0.5)、「インターフェースと抽象
# クラスの使い分けは？」で search_history が "No results found" を返す —
# どちらも履歴参照の意図が皆無な純粋な知識質問で、小型 assist モデルが
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
# 尋ねる質問には無関係だが、明確なツールシグナルが無いため層4 (assist
# tool_judgment) まで素通りし、小型 assist モデルが誤って search_history を
# 選ぶ実インシデントが多発した (2026-07-17/18 の2日分ログで
# tool_call_decision=assist の 48% が該当)。「あったりしますか」等の口語的な
# 疑問文末尾も拾う (上の _KNOWLEDGE_PATTERNS の「ありますか」は完全一致部分
# 文字列のみを拾うため「あったりします(か)」を取りこぼす)。
# あえて _KNOWLEDGE_PATTERNS 側に「あったり」単体を追加しない: _KNOWLEDGE_PATTERNS
# は _judge_with_rules (層1) でも無条件に (一人称マーカーを考慮せず) 参照される
# ため、汎用パターンとして追加すると「私は好きな〜あったりしますか」のような
# 一人称クエリまで層1で誤って即 no_tool になり、下記の一人称除外が層4に届く
# 前に握り潰されてしまう (レビューで判明)。
# ただし「私の/私は/僕の/僕は/自分の/自分は」等の一人称マーカーを含む場合は
# ユーザー自身の過去発言 (例:「私の好きなプログラミング言語は？」) を指す
# 可能性があるため除外せず、層4 (assist) 判定に委ねる (実際に "Rust" 等の
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

# _ASSISTANT_PREFERENCE_PATTERNS / _FIRST_PERSON_REFERENCE_RE の英語版。
# 英語の疑問文は助動詞前置 (do/does/what's) で疑問文を示すため、日本語版の
# ような文末助詞アンカーではなく、助動詞前置構造をアンカーにする。
_ASSISTANT_PREFERENCE_PATTERNS_EN = [
    re.compile(
        r"\b(?:do|does)\s+you\s+(?:have\s+(?:a|any)\s+favou?rite|like|enjoy)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bwhat(?:'s|\s+is)\s+your\s+favou?rite\b", re.IGNORECASE),
    re.compile(r"\bare\s+you\s+(?:good|great|bad)\s+at\b", re.IGNORECASE),
    re.compile(r"\b(?:got|have)\s+a\s+favou?rite\b[^.!\n]{0,20}[?]?\s*$", re.IGNORECASE),
]
_FIRST_PERSON_REFERENCE_RE_EN = re.compile(
    r"\bmy\b|\bi(?:'m| am| was| like| liked| love| loved| prefer"
    r"| preferred| said| told you| mentioned| have)\b",
    re.IGNORECASE,
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
# (pillar境界のため backend/free/rag/self_rag_judge.py の
# TRIVIAL_QUESTION_PATTERNS と同義の定義を重複させている。両ファイルを
# 変更する際は同期させること)。
# 反省的な語には時系列順序語 (最初/最後/何番目/何回目) も含める。
# 「この会話で一番最初に計算させた問題は?」(2026-07-21 ライブ検証 ターン18)
# は反省語 (覚えて等) を欠きスコープ注入が漏れ、cross-session 検索が前回
# 会話の類似ターンを引き当て誤答した。順序語の追加でマッチ面が広がる分、
# 「この会話じゃなくて/ではなく」の明示的な話題切断も lookahead へ追加する。
_SELF_SESSION_REFERENCE_PATTERNS = [
    re.compile(
        r"(?:この会話|このやり取り|今までの(?:会話|やり取り)"
        r"|今日の(?:追加分の)?会話|今回の(?:追加分の)?会話)"
        r"(?!とは別|とは関係|は関係な|じゃなく|ではなく)"
        r"[^。．!！?？\n]{0,40}?"
        r"(?:面白|印象|振り返|まとめ|要約|感想|どう思|覚えて|何でした|どうでした"
        r"|最初|最後|何番目|何回目)",
    ),
]

# _SELF_SESSION_REFERENCE_PATTERNS の英語版。英語の話題切断表現
# ("aside from"/"apart from") は名詞句の前に来る (日本語の後置とは語順が
# 逆) ため、_SESSION_TOPIC_BREAK_LEAD_RE_EN の前置きガードと併用する
# (_maybe_scope_session_search 側で参照)。
# (pillar境界のため backend/free/rag/self_rag_judge.py の
# TRIVIAL_QUESTION_PATTERNS_EN と同義の定義を重複させている。両ファイルを
# 変更する際は同期させること)。
_SELF_SESSION_REFERENCE_PATTERNS_EN = [
    re.compile(
        r"(?:this\s+conversation|this\s+chat|our\s+conversation"
        r"|what\s+we\s+(?:talked|discussed|were\s+talking)\s+about"
        r"|earlier\s+in\s+this\s+(?:conversation|chat)"
        r"|so\s+far\s+in\s+this\s+conversation)"
        r"(?!\s*(?:is|was|has)?\s*(?:not\s+related|unrelated|nothing\s+to\s+do))"
        r"[^.!?\n]{0,40}?"
        r"(?:interesting|memorable|impressive|funn(?:y|iest)"
        r"|summar\w*|recap\w*|think|thought|feel|felt|remember"
        r"|first|last|earliest|latest)"
        # 英語は修飾語 (interesting 等) が「this conversation」より前に
        # 来る語順も自然なため (日本語の後置とは逆)、逆順の共起も許容する。
        # 前置きガードは _SESSION_TOPIC_BREAK_LEAD_RE_EN 側で別途行う。
        r"|(?:interesting|memorable|impressive|funn(?:y|iest)"
        r"|summar\w*|recap\w*|first|last|earliest|latest)"
        r"[^.!?\n]{0,40}?"
        r"(?:this\s+conversation|this\s+chat|our\s+conversation)",
        re.IGNORECASE,
    ),
]
# 話題切断が前置される英語特有の言い回し (「この会話とは別に」の語順違い対策)。
_SESSION_TOPIC_BREAK_LEAD_RE_EN = re.compile(
    r"\b(?:aside|apart|other than|separate)\s+from\s+(?:this|our)\s+conversation\b",
    re.IGNORECASE,
)

def _coerce_positive_int(value: object) -> int | None:
    """assist の型崩れ JSON 由来の値を正の int へ正規化する (int / 数値文字列 /
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
# assist が合成する小さい limit (例: limit=1) は字句スコア最上位への
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
    r"|一番最初|一番最後|何番目|何回目",
)
# 内容ラン (漢字 / カタカナ / ラテン / 数字。ひらがなの助詞・活用語尾は自然に
# 脱落する)。
_ORDER_QUERY_CONTENT_RE = re.compile(
    r"[一-鿿゠-ヿ々〆a-zA-Z0-9]+",
)
# 内容ランのうち scaffolding とみなして落とす語 (質問・順序・自己参照の骨組み)。
_ORDER_QUERY_STOPWORD_RUNS = frozenset({
    "会話", "一番", "最初", "最後", "直近", "以前", "前回", "今日", "今回", "今",
    "問題", "質問", "内容", "話題", "話", "何", "誰", "私", "貴方", "君", "僕",
    "俺", "覚", "番目", "回目", "先",
})

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
})


def _reduce_ordered_history_query(query: str) -> str:
    """順序リコール質問から search_history 用の内容キーワードを抽出する。

    レイヤー5.5 の強制フォールバックは search_history に生クエリ全文を渡すが、
    HistoryManager の字句照合は長い疑問文を短い会話ターンにマッチできない
    (2026-07-21 ライブ検証: 「この会話で一番最初に計算させた問題は何？」が
    索引の search_text に「計算」を含むのに No results found)。self-reference /
    順序語 / 疑問 scaffolding を除去して内容キーワード (例: 計算) を残す。
    抽出できなければ生クエリを返す (悪化させない安全側)。digest には別途 raw
    query が渡るため、順序解釈 (「一番最初」) はこの縮約で失われない。
    """
    if is_en_locale():
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
    terms = [
        run
        for run in content_re.findall(stripped)
        if run.lower() not in stopwords
    ]
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
    # 取りこぼし (「私がやるから大丈夫」等の複文) は層4 assist 判定に
    # 落ちるだけで安全側。
    re.compile(
        r"(?:自分で|自分が|私が|僕が|俺が|わたしが|こっちで|こちらで)"
        r"[^。．!！?？\n]*"
        r"(?:[うくぐつぬぶむる]|(?<!で)す)"
        r"(?:ね|よ|わ|から|かな)?"
        r"[\s　!！。.…]*$",
    ),
]

# _SELF_ACTION_PATTERNS の英語版。「I'll try it myself」等、文末アンカー方式
# は英語でも概ね踏襲できる (I'll/let me + 動詞 が文末付近に来る自己完結宣言)。
_SELF_ACTION_PATTERNS_EN = [
    re.compile(
        r"\b(?:i'?ll|i\s+will|let\s+me)\s+(?:try|give\s+it\s+a\s+(?:go|try|shot))"
        r"(?:\s+(?:it|this|that|myself))?\s*[.!]*\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bi'?ll\s+(?:handle|take\s+care\s+of|figure\s+out|sort\s+out"
        r"|deal\s+with|work\s+(?:it|this)\s+out)\b[^.!\n]*[.!]*\s*$",
        re.IGNORECASE,
    ),
    re.compile(r"\bi'?ll\s+.{0,25}\bmyself\b[.!]*\s*$", re.IGNORECASE),
]


def _query_has_tool_signal(query: str) -> bool:
    """クエリにツール操作シグナル (ツールパターン / Windows・Unix パス / URL) を含むか。"""
    patterns = select_locale_variant(_TOOL_PATTERNS, _TOOL_PATTERNS_EN)
    return (
        any(p.search(query) for p in patterns)
        or bool(re.search(r"[A-Za-z]:\\|(?:^|[\s　])(?:/[\w._-]+){2,}|https?://", query))
    )


def _is_conversational_query_without_tool_signal(query: str) -> bool:
    """ツールシグナルが無く、知識質問 or 自己行動宣言の雑談発話か (assist 判定スキップ条件)。

    True の場合、層4 (tool_judgment) / 層5 (executable synth) の realtime
    assist 呼出を省いて no-tool 即決してよい。対象は:

    - 知識質問 (「とは」「教えて」等) — RAG / LLM 知識で答える
    - ユーザー自身の行動宣言 (「探してみるね」「自分で調べる」等) — 依頼ではない
    - アシスタント自身の意見・嗜好を尋ねる質問 (「好きな〜はありますか」等、
      一人称マーカーを含まないもの) — search_history 等の過去発言検索は無関係

    セッション自己参照 (「この会話で」等、``_SELF_SESSION_REFERENCE_PATTERNS``)
    はここでは判定しない。長距離 recall のため層4 判定は通常どおり実行させ、
    search_history が選ばれた場合は ``_maybe_scope_session_search`` が事後的に
    現在セッションへスコープ限定する (詳細は同定数の定義コメント参照)。

    時刻・スペック等の実行可能事実クエリやファイルパス・URL を含むクエリは
    ``_TOOL_PATTERNS`` / パスシグナルで弾かれるため、誤って no-tool に倒さない。
    依頼形 (「探してみて」) は ``_SELF_ACTION_PATTERNS`` が文末「てみる」に限定して
    いるためマッチしない。
    """
    if _query_has_tool_signal(query):
        return False
    preference_patterns = select_locale_variant(
        _ASSISTANT_PREFERENCE_PATTERNS, _ASSISTANT_PREFERENCE_PATTERNS_EN,
    )
    first_person_re = select_locale_variant(
        _FIRST_PERSON_REFERENCE_RE, _FIRST_PERSON_REFERENCE_RE_EN,
    )
    if (
        any(p.search(query) for p in preference_patterns)
        and not first_person_re.search(query)
    ):
        return True
    knowledge_patterns = select_locale_variant(_KNOWLEDGE_PATTERNS, _KNOWLEDGE_PATTERNS_EN)
    self_action_patterns = select_locale_variant(_SELF_ACTION_PATTERNS, _SELF_ACTION_PATTERNS_EN)
    return (
        any(p.search(query) for p in knowledge_patterns)
        or any(p.search(query) for p in self_action_patterns)
    )


def _has_history_recall_keywords(query: str) -> bool:
    """明示的な履歴参照キーワード (router.HISTORY_KEYWORDS) を含むか。

    router.ComplexityClassifier._has_history_keywords と同じ判定 (小文字化
    後の部分文字列一致) を、layer 分類とは独立に tool 強制発火の判定に使う。
    """
    q_lower = query.lower()
    keywords = select_locale_variant(HISTORY_KEYWORDS, HISTORY_KEYWORDS_EN)
    return any(kw in q_lower for kw in keywords)


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
        session_id: str = "",
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
            session_id: 現在のチャットセッション ID。search_history が選ばれ、
                かつクエリが「この会話で」等のセッション自己参照パターンに
                一致する場合に ``_maybe_scope_session_search`` が
                ``tool_args["session_id"]`` へ注入し、検索を現在セッションに
                限定する (未指定時は従来どおり cross-session 検索のまま)。

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
            url_recall_result = self._validate_tool_availability(
                url_recall_result, tools_registry, mode,
            )
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
            cmd_recall_result = self._validate_tool_availability(
                cmd_recall_result, tools_registry, mode,
            )
            self._log_tool_decision(
                cmd_recall_result, "executable_command_recall_matched",
            )
            return cmd_recall_result

        # config で明示的に無効化 → chat モードでは判定スキップ
        # coding モードではルールベース + カ���トリッジ hints にフ��ールバック
        # ただし chat モードでも executable query（時刻・スペック等）は
        # 安全なコマンド生成のみ行うためルールベース判定を許可する
        if not self.enabled and not is_coding_mode(mode):
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
                result = self._validate_tool_availability(
                    result, tools_registry, mode,
                )
                self._log_tool_decision(result, "chat_mode_explicit_url")
                return result

            # assist 優先で実行可能コマンドを解決する。assist 未接続 / 失敗
            # 時はモジュールレベル sync 関数 (regex テーブル) にフォールバック。
            # ツール名は mode から解決する (chat は run_command_readonly)。
            # 解決不能な構成では assist 往復を省略して即 no_tool。
            exec_tool = _executable_tool_for_mode(tools_registry, mode)
            if exec_tool:
                command = await self._resolve_executable_command(query)
                if command and not _readonly_command_rejected(exec_tool, command):
                    logger.debug(
                        "Chat mode executable query detected: %s", query[:50],
                    )
                    # assist が解決したか regex フォールバックかは
                    # _command_cache_lookup の有無で判別可能だが、source は
                    # ログ用途のため簡略化して "rule" のまま返す
                    # (キャッシュヒット = assist 由来でも "rule" でログされる)。
                    result = ToolJudgement(
                        tool_needed=True,
                        tool_name=exec_tool,
                        tool_args={"command": command},
                        source="rule",
                    )
                    return self._validate_tool_availability(
                        result, tools_registry, mode,
                    )
            return ToolJudgement(tool_needed=False, source="rule")

        # 1. 組み込みパターン照合（ルールベース）
        result = self._judge_with_rules(query, tools_registry, mode)
        if result.tool_needed:
            # run_command の場合は assist でコマンド合成を試行 (assist=None
            # / 失敗時は regex 由来の既存コマンドを維持)
            result = await self._maybe_upgrade_command_via_assist(result, query)
            await self._maybe_recall_url(result, query, mode=mode)
            self._maybe_scope_session_search(result, query, session_id)
            result = self._suppress_unfetchable_fetch_url(result)
            result = self._suppress_commandless_run_command(result)
            result = self._suppress_expressionless_calculate(result)
            result = self._validate_tool_availability(result, tools_registry, mode)
            self._log_tool_decision(result, "rule_pattern_matched")
            return result

        # 2. カートリッジ tool_hints 照合
        result = self._judge_with_cartridge_hints(query, tools_registry)
        if result.tool_needed:
            self._maybe_scope_session_search(result, query, session_id)
            result = self._suppress_unfetchable_fetch_url(result)
            result = self._validate_tool_availability(result, tools_registry, mode)
            self._log_tool_decision(result, "cartridge_hint_matched")
            return result

        # 3. 学習済みパターン照合
        result = self._judge_with_learned_patterns(query, tools_registry, mode)
        if result.tool_needed:
            result = await self._maybe_upgrade_command_via_assist(result, query)
            await self._maybe_recall_url(result, query, mode=mode)
            self._maybe_scope_session_search(result, query, session_id)
            result = self._suppress_unfetchable_fetch_url(result)
            result = self._suppress_commandless_run_command(result)
            result = self._suppress_expressionless_calculate(result)
            result = self._validate_tool_availability(result, tools_registry, mode)
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
                self._judge_with_executable_fallback(query, tools_registry, mode),
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
                    self._maybe_scope_session_search(assist_result, query, session_id)
                    self._maybe_expand_ordered_history_search(assist_result, query)
                    assist_result = self._suppress_unfetchable_fetch_url(
                        assist_result,
                    )
                    assist_result = self._suppress_commandless_run_command(
                        assist_result,
                    )
                    assist_result = self._suppress_expressionless_calculate(
                        assist_result,
                    )
                    assist_result = self._suppress_hidden_tool_from_assist(
                        assist_result, tools_registry,
                    )
                    assist_result = self._validate_tool_availability(
                        assist_result, tools_registry, mode,
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
                    query, tools_registry, mode,
                )
            if fallback_result is not None:
                self._log_tool_decision(
                    fallback_result, "executable_command_fallback",
                )
                return fallback_result

            # 5.5. 履歴参照キーワードのフォールバック強制発火 (安全網)
            # router.HISTORY_KEYWORDS 相当の明示的な recall 語 (「覚えて」
            # 「最初に」等) を含むクエリで、層4 (assist) が no_tool と判定した
            # 場合の最終防衛線。小型 assist モデルの確率的な見落としで
            # search_history が一度も呼ばれず、長距離 recall がベースモデルの
            # 幻覚に倒れる実インシデントがあった (2026-07-20:「この会話で一番
            # 最初に私が計算させた問題は何だったか覚えてますか？」で
            # search_history 未発火 → 受動 RAG (quality=medium) のみで
            # 「そんな計算はなかった」と誤って断言)。
            # query は原則ユーザーの生クエリをそのまま渡す (search_history の
            # 照合は字句重なりベースのため、既に会話内で言及済みの語彙を含む
            # 再質問では十分ヒットしうる)。ただし順序リコール質問
            # (「一番最初に〜した〜は？」等) は生クエリ全文が長すぎて短い
            # 会話ターンに字句マッチせず空振りする (2026-07-21 ライブ検証で
            # 確認) ため、_reduce_ordered_history_query で内容キーワードへ
            # 縮約する (「この会話で一番最初に計算させた問題は何？」→「計算」)。
            # 順序解釈は digest 側が受け取る raw query が担うため縮約で失われ
            # ない。ヒットしなくても "No results found" 経由で「見つからな
            # かった」という正直な応答に倒れるため、無言のまま確信を持って
            # 幻覚するより悪化はしない。
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
                search_query = query
                if _ORDERED_HISTORY_QUERY_RE.search(query):
                    search_query = _reduce_ordered_history_query(query)
                forced_result = ToolJudgement(
                    tool_needed=True,
                    tool_name="search_history",
                    tool_args={"query": search_query, "mode": mode},
                    source="rule",
                )
                self._maybe_scope_session_search(forced_result, query, session_id)
                forced_result = self._validate_tool_availability(
                    forced_result, tools_registry, mode,
                )
                if forced_result.tool_needed:
                    self._log_tool_decision(
                        forced_result, "history_keyword_forced_fallback",
                    )
                    return forced_result

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
        self, query: str, tools_registry: ToolsRegistry, mode: str = "coding",
    ) -> "ToolJudgement | None":
        """4 層判定後の executable command フォールバック (5 層目).

        環境依存事実クエリ (「Chrome のバージョン」「インストール済み pip
        パッケージ」等) を assist (executable_command_synth) で判定し、
        実行可能なら ``run_command`` 判定を返す。

        Pre-filter で挨拶 / 短文 / 機能 OFF / degraded mode を弾いてから
        assist を呼ぶ。assist 呼出は ``_synthesize_command_via_assist`` を
        再利用するため、Free chat mode の early-return と LRU キャッシュを
        共有する (同一 query は重複呼出ゼロ)。

        ツール名は ``_executable_tool_for_mode`` で mode から解決する
        (coding → run_command / chat → run_command_readonly)。どちらも使えない
        構成では assist 呼出自体を省略する (判定確定前にここで弾かないと環境
        依存事実クエリのたびに無駄な assist 往復 (約2秒) が発生し、かつ判定
        結果が tool_needed=True のまま返って turn_outcome=failed → 誤学習
        カスケードを招く。2026-07-18 の会話ログで実際に発生・確認済み)。

        Returns:
            実行可能と判定された場合 ``ToolJudgement(<exec_tool>, ...)``。
            機能 OFF / pre-filter 弾き / assist が否定 / assist 失敗 / 現在
            mode で利用不可の場合は ``None`` (呼出側は no-tool として処理する)。
        """
        if not self._executable_fallback_enabled:
            return None
        exec_tool = _executable_tool_for_mode(tools_registry, mode)
        if not exec_tool:
            return None
        if self._assist_client is None:
            return None
        stripped = query.strip()
        if len(stripped) < self._executable_fallback_min_chars:
            return None
        # 挨拶は assist を呼ばずにスキップ (reactive 層の定型応答対象)
        greetings = select_locale_variant(GREETING_RESPONSES, GREETING_RESPONSES_EN)
        if any(p.match(stripped) for p, _ in greetings):
            return None

        synth = await self._synthesize_command_via_assist(query)
        if synth is None:
            return None
        is_exec, command = synth
        if not is_exec or not command:
            return None
        if _readonly_command_rejected(exec_tool, command):
            return None
        logger.info(
            "Executable command synthesized via 5th layer fallback: query=%s",
            query[:50],
        )
        return ToolJudgement(
            tool_needed=True,
            tool_name=exec_tool,
            tool_args={"command": command},
            source="assist",
        )

    def _validate_tool_availability(
        self, result: "ToolJudgement", tools_registry: ToolsRegistry, mode: str,
    ) -> "ToolJudgement":
        """tool_name が実在し、かつ現在の mode で利用可能かを最終チェックする。

        ``_judge_with_rules`` / ``_judge_with_learned_patterns`` 等の各判定層は
        ``tools_registry.has()`` (存在チェックのみ、mode 非考慮) で判定するため、
        chat モードで ``modes=["coding"]`` のツール (例: run_command) が
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
            return ToolJudgement(tool_needed=False, source=result.source)
        if mode not in tool_def.modes:
            logger.info(
                "Tool %s not available in mode=%s (allowed: %s); "
                "downgrading to no_tool before returning judgement",
                result.tool_name, mode, tool_def.modes,
            )
            return ToolJudgement(tool_needed=False, source=result.source)
        return result

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

    def _suppress_commandless_run_command(
        self, result: "ToolJudgement",
    ) -> "ToolJudgement":
        """command 引数を補完できない ``run_command`` を ``tool_needed=False`` に格下げ.

        ``_suppress_unfetchable_fetch_url`` と対称。rule / learned 層は字句
        マッチだけで ``run_command`` を選ぶことがあり (learned 層は
        ``_infer_tool`` が推定できない場合の run_command フォールバックを持つ)、
        ``_maybe_upgrade_command_via_assist`` の assist 合成でも command が
        埋まらなかった場合、実行段階で "requires args but none provided" と
        空振りするだけの判定が残る。実インシデント (2026-07-20 ライブ検証):
        学習済み tool_routing パターン「説明」(w=0.630) が知識質問
        「〜を説明して」にマッチし、coding モードで引数なし run_command が
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
            "inference, no assist synthesis); downgrading to no_tool",
            result.tool_name,
        )
        return ToolJudgement(
            tool_needed=False,
            tool_name="",
            tool_args={},
            source=result.source,
        )

    def _suppress_hidden_tool_from_assist(
        self, result: "ToolJudgement", tools_registry: ToolsRegistry,
    ) -> "ToolJudgement":
        """assist (tool_judgment) が hidden ツール名を返した場合 no_tool に格下げ.

        hidden ツール (run_command_readonly 等) はプロンプトのツール一覧
        (``get_descriptions_text``) に出ないため、assist がその名前を返すのは
        定義上 hallucination。``_validate_tool_availability`` は登録済み +
        mode 適合なら通してしまう (chat で modes=["chat"] の hidden ツールは
        素通り) ため、assist 層専用の防衛としてここで弾く。judge のコード側
        注入経路 (early-return / recall / fallback / _infer_tool) は本メソッド
        を通らないため影響しない。
        """
        if not result.tool_needed or not result.tool_name:
            return result
        tool_def = tools_registry.get(result.tool_name)
        if tool_def is None or not tool_def.hidden:
            return result
        logger.warning(
            "Suppressing hidden tool %s returned by assist judgement "
            "(not in prompt tool list); downgrading to no_tool",
            result.tool_name,
        )
        return ToolJudgement(tool_needed=False, source=result.source)

    def _suppress_expressionless_calculate(
        self, result: "ToolJudgement",
    ) -> "ToolJudgement":
        """expression 引数を補完できない ``calculate`` を ``tool_needed=False`` に格下げ.

        ``_suppress_commandless_run_command`` と対称。rule / learned 層の
        ``_infer_tool`` は「計算」の字句マッチだけで calculate を選ぶが式抽出
        ロジックを持たず常に空 args を返し、assist 層も free-form args のため
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

    def _maybe_scope_session_search(
        self, result: "ToolJudgement", query: str, session_id: str,
    ) -> None:
        """``search_history`` が選ばれ、クエリが「この会話で」等のセッション
        自己参照パターンに一致する場合、``tool_args["session_id"]`` を強制注入
        して検索対象を現在セッションのみに限定する。

        session_id を渡さずに search_history を無条件許可すると、2026-07-17/18
        の実インシデント (「この会話で一番面白かった？」が無関係な過去セッション
        の内容を誤って混同した) が再発するため、自己参照パターンに一致する場合
        のみ強制的にスコープを絞る。in-place で更新する。``session_id`` が空
        (未提供) の場合は何もしない (呼出元が未対応でも安全に no-op)。
        """
        if result.tool_name != "search_history":
            return
        if not session_id:
            return
        if is_en_locale():
            patterns = _SELF_SESSION_REFERENCE_PATTERNS_EN
            if _SESSION_TOPIC_BREAK_LEAD_RE_EN.search(query):
                return
        else:
            patterns = _SELF_SESSION_REFERENCE_PATTERNS
        if not any(p.search(query) for p in patterns):
            return
        if result.tool_args is None:
            result.tool_args = {}
        result.tool_args["session_id"] = session_id
        logger.debug(
            "search_history scoped to current session (self-session reference): %s",
            query[:50],
        )

    def _maybe_expand_ordered_history_search(
        self, result: "ToolJudgement", query: str,
    ) -> None:
        """時系列順序指定クエリの ``search_history`` で小さい limit を既定値へ引き上げる.

        実インシデント (2026-07-21 ライブ検証 ターン18): 「この会話で一番最初に
        計算させた問題は?」で assist が ``query='計算', limit=1`` を合成 →
        ``HistoryManager.search_sessions`` は字句スコア降順で limit 件に切る
        ため、時系列先頭ではなく直近の計算を返し誤答した。limit が十分なら
        turn# 付きの全マッチターンが digest に渡り、元クエリ (digest の user
        prompt に含まれる) と合わせて時系列選択が機能する (同検証 ターン42
        「すべて挙げて」が limit 既定 10 で 6 件完全列挙に成功した実績)。

        判定は assist 合成後の ``args["query"]`` ではなく**ユーザー生クエリ**
        に対して行う (合成 query では「一番最初」等の順序語が消えている)。
        引き上げのみで引き下げはしない (「直近20件」等の明示的な大 limit を
        壊さない)。挿入点は assist 層のみ — limit を合成し得るのは free-form
        args の assist 層だけで、rule/learned 層の ``_infer_tool`` に
        search_history 分岐は無く、cartridge 層は空 args、層5.5 の強制発火は
        limit を設定しない (ハンドラ既定 10 が効く)。in-place で更新する。

        assist (LFM2) は json_schema grammar を強制せず型崩れ JSON を返し得る
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
        """SemMem の過去成功コマンド引き当てで executable 判定を返す.

        ``_judge_with_url_recall`` と対称。条件:
          - 現在の mode で executable ツールが解決できる
            (coding → run_command / chat → run_command_readonly)
          - ``mem_view`` / ``embedder`` が wired されている
          - ``_try_recall_executable_command`` が閾値判定でコマンドを返す

        引き当てが成立すれば assist 呼出 (5 層目 / chat early-return) より
        先に確定するため、学習済みクエリでは LLM コストがゼロになる。

        recall は subject の mode (`mem.world.executable_command.<mode>.*`) を
        フィルタしないため、chat では coding 学習由来の任意コマンド (書込系
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
        if _readonly_command_rejected(exec_tool, recalled):
            return None
        logger.info(
            "Executable command recall: matched command for query=%s",
            query[:50],
        )
        return ToolJudgement(
            tool_needed=True,
            tool_name=exec_tool,
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
            # 768: reasoning_budget=0 (tool_judgment既定) は gemma4 系アシストで
            # 稀に (~6日で15件) 実効せず、reasoning_content が本文出力前に
            # max_tokens を消費し尽くして空応答 → リトライ → タイムアウトに至る
            # 実インシデントが確認された (2026-07-18)。256 では観測された
            # reasoning 長 (700-1245 token) を到底吸収できないため、この
            # anomaly が起きても本文 JSON まで到達できる余裕を持たせる。
            # 通常時 (thinking 抑制が効くケース) は数十 token で完結するため
            # 上限を上げても実害はない。
            max_tokens=768,
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

        # 書込み成果物タスク (write 動詞 + 明示ファイルパス) に環境コマンド合成は
        # 不要かつ自明に not-executable。assist 呼出 (15〜20 秒/回) を省き、
        # 決定論で (False, "") を返す (2026-07-15: 5 回全て is_executable=false の
        # 純オーバーヘッド 74.5 秒)。
        from backend.free.agent.meta_cognitive_tasks import task_expects_write
        if task_expects_write(query) and _extract_file_path(query):
            logger.debug(
                "executable_command_synth skipped (write deliverable task): %s",
                query[:60],
            )
            value = (False, "")
            self._command_cache_store(query, value)
            return value

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
                # reasoning 型 assist は thinking が completion を食い潰し
                # 256 では 2/5 が finish=length 切断になる (2026-07-15 実測)
                max_tokens=384,
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
            - assist が ``(False, "")`` を返し、かつクエリが書込み期待
              (``task_expects_write``) → ``no_tool`` に降格
              (書込みタスクへの環境コマンド発火は regex 部分マッチ由来の
              誤判定と断定できる。2026-07-15 の "program" 内 'ram' マッチで
              OS スペック収集を実行し write 不発で失敗した実績への防衛)
            - assist が ``(False, "")`` で書込み期待なし → 既存 regex 結果を
              維持 (specific パターンにマッチした confidence を尊重)
            - assist 失敗 / 未接続 → 既存結果をそのまま返す
        """
        from backend.free.agent.meta_cognitive_tasks import task_expects_write

        if not result.tool_needed or result.tool_name not in (
            "run_command", "run_command_readonly",
        ):
            return result
        synth = await self._synthesize_command_via_assist(query)
        if synth is None:
            return result
        is_exec, command = synth
        if is_exec and command:
            # readonly ツールに synth コマンドを載せる場合、readonly 検証に
            # 通らないコマンド (PowerShell スニペット等) では既存の安全な
            # コマンド (regex テーブル由来) を維持し、アップグレードしない。
            if _readonly_command_rejected(result.tool_name, command):
                return result
            new_args = dict(result.tool_args or {})
            new_args["command"] = command
            return ToolJudgement(
                tool_needed=True,
                tool_name=result.tool_name,
                tool_args=new_args,
                source="assist",
            )
        if task_expects_write(query):
            logger.info(
                "Demoting %s to no_tool: assist judged not executable "
                "and the task expects a write deliverable: %s",
                result.tool_name, query[:80],
            )
            return ToolJudgement(tool_needed=False, source="assist")
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
        knowledge_patterns = select_locale_variant(_KNOWLEDGE_PATTERNS, _KNOWLEDGE_PATTERNS_EN)
        if not has_tool_signal and any(p.search(query) for p in knowledge_patterns):
            logger.debug("Rule-based: knowledge query detected, skipping tool: %s", query[:50])
            return ToolJudgement(tool_needed=False, source="rule")

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
            # _maybe_upgrade_command_via_assist が command を合成できた場合
            # のみ生き残る。合成不成立 (assist が not executable と判定 /
            # assist 未接続) なら _suppress_commandless_run_command が
            # no_tool に倒す — 学習パターンの字句マッチだけを根拠に実行不能な
            # run_command を返さない (2026-07-20: 学習済み「説明」が知識質問
            # にマッチし coding モードで誤発火し得た件の防衛線)。
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
        if exec_query_re.search(q) and exec_tool:
            # システム情報クエリは具体的なコマンドを生成
            command = _infer_executable_command(query)
            if command:
                return exec_tool, {"command": command}
            return exec_tool, {}

        # 計算パターン (実行可能クエリ分岐より後ろに置く。「フィボナッチ数列の
        # 10番目を計算して」のように両方にマッチするクエリは、式抽出を持たない
        # calculate {} で潰さず run_command 合成経路 (assist synth) に乗せる。
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
