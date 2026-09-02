"""LLM 応答から JSON を頑健に抽出するユーティリティ

小規模 LLM はコードフェンス・前置き自然文・末尾切断などを混入させるため、
複数戦略でフォールバックしながら JSON を抽出する。judge.py / aux_client /
cogwriter など JSON 応答をパースする箇所で共通利用する

戦略 4: ``response_format=json_schema`` を採用しても、Pro 外部 API (GBNF 非対応)、
``--skip-chat-parsing`` の raw content 経路、古い llama-server build などでは
依然として構造的不正が発生し得る。最終フォールバックとして ``json_repair`` で
機械的に閉じ括弧補完を行い、必要性評価のため telemetry out-param で repair
使用を呼出側に伝える。

``max_tokens`` 到達 (``finish_reason=length``) の切断は **ここで修復しない**
方針 — ``[0, 1,`` を ``[0, 1]`` に閉じると欠けた要素が「無かった」ことになる。
``AuxClient.generate_json`` が finish_reason を見て空応答へ倒す。
"""

from __future__ import annotations

import json
import re

from backend.log_config import get_logger

logger = get_logger("llm.json_extract")

#: ドライブ文字付きの Windows パス (``E:\tmp\x.py``)。LLM は JSON 文字列内で
#: バックスラッシュをエスケープしないことが多く、``json.loads`` が ``\t`` / ``\f``
#: を制御文字に化かして ``E:mpizz`` のようなパスになる (2026-09-02 実機)。
_WINDOWS_PATH_RE = re.compile(r'(?<![A-Za-z0-9_])([A-Za-z]:)((?:\\+[^\\"\s]+)+)')
_SINGLE_BACKSLASH_RE = re.compile(r"(?<!\\)\\(?!\\)")


def escape_windows_path_backslashes(text: str) -> str:
    """JSON 文字列内のドライブ付きパスで、単独のバックスラッシュを ``\\\\`` に直す。

    既にエスケープ済み (``\\\\``) の箇所は触らない。パス以外の ``\\n`` (コード本文の
    改行エスケープ) はドライブ文字に続かないので対象外。
    """
    if "\\" not in text:
        return text

    def _fix(m: re.Match) -> str:
        return m.group(1) + _SINGLE_BACKSLASH_RE.sub(r"\\\\", m.group(2))

    return _WINDOWS_PATH_RE.sub(_fix, text)


def strip_code_fences(content: str) -> str:
    """Markdown コードフェンス (```json ... ``` / ``` ... ```) を除去する。

    終了フェンスが欠落している (途中切断や冒頭だけ ```json を出すケース)
    場合も許容する。
    """
    text = content.strip()
    m = re.match(r"^```(?:json|JSON)?\s*\n?", text)
    if m:
        text = text[m.end():]
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def extract_balanced(text: str, open_c: str, close_c: str) -> str | None:
    """最初の `open_c` から対応する閉じ文字までを balanced 抽出する。

    文字列リテラル内の括弧は無視する。balanced でない場合は末尾 `close_c`
    までの range をフォールバックとして返す (部分一致許容)。
    """
    start = text.find(open_c)
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_c:
            depth += 1
        elif ch == close_c:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    end = text.rfind(close_c)
    if end > start:
        return text[start:end + 1]
    return None


def extract_json_object(
    content: str,
    *,
    list_key: str | None = None,
    telemetry: dict | None = None,
) -> dict | None:
    """LLM 応答から JSON オブジェクトを抽出する。

    戦略:
        1. コードフェンス除去 → 直接 JSON パース
        2. balanced ``{...}`` 抽出
        3. balanced ``[...]`` 抽出 (裸配列応答 → ``list_key`` でラップ)
        4. ``json_repair.repair_json`` フォールバック
            ``max_tokens`` 切断・GBNF 非対応経路 (Pro 外部 API /
            ``--skip-chat-parsing``) 由来の構造的不正を機械修復する。

    Args:
        content: LLM 応答テキスト
        list_key: 裸の JSON 配列応答が来た場合にどのキーでラップするか
            (例: "claims" / "verdicts" / "units")。
            None の場合は配列単独応答を採用しない。
        telemetry: 抽出戦略の使用状況を記録する out-param dict。
            指定された場合、戦略 4 (json-repair) で成功した時に
            ``telemetry["repair_used"] = True`` がセットされる。
            ``response_format=json_schema`` 採用後の
            必要性評価のため、呼出側で記録経路に渡す想定。

    Returns:
        抽出された dict。失敗時は None。
    """
    if not content:
        return None

    normalized = escape_windows_path_backslashes(strip_code_fences(content))

    # 戦略 1: 正規化後の文字列をそのまま JSON として解釈
    try:
        parsed = json.loads(normalized)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list) and list_key:
            return {list_key: parsed}
    except (json.JSONDecodeError, TypeError):
        pass

    # 戦略 2: balanced `{...}` を抽出
    obj_str = extract_balanced(normalized, "{", "}")
    if obj_str:
        try:
            result = json.loads(obj_str)
            if isinstance(result, dict):
                logger.debug("JSON parsed via balanced object")
                return result
        except json.JSONDecodeError:
            pass

    # 戦略 3: balanced `[...]` を抽出 (裸配列応答)
    # 先頭非空白文字が `[` の場合のみ発火。先頭が `{` の場合は object 応答が
    # 意図されており、truncation 等で戦略 2 が失敗していても strategy 4
    # (json-repair) に任せる方が安全。内部のネスト配列 (`constraints` 等の
    # 文字列配列) を units として誤抽出するのを防ぐ。
    if list_key and normalized.lstrip().startswith("["):
        arr_str = extract_balanced(normalized, "[", "]")
        if arr_str:
            try:
                result = json.loads(arr_str)
                if isinstance(result, list):
                    logger.debug(
                        "JSON parsed via bare array -> %s", list_key,
                    )
                    return {list_key: result}
            except json.JSONDecodeError:
                pass

    # 戦略 4: json-repair による機械修復
    # truncation (``{"items":[{"a":1},``) や閉じクォート欠落など、戦略 1-3
    # では救えない構造的不正を補完する。``return_objects=True`` で Python
    # オブジェクトを直接受け取り、文字列が返ってきた場合 (= 完全失敗) は
    # None にフォールバックする。``json_repair`` 未インストール環境では
    # 戦略 4 をスキップする (defensive: requirements.txt 不整合時の保険)。
    repaired = _repair_json(normalized)
    if isinstance(repaired, dict):
        if telemetry is not None:
            telemetry["repair_used"] = True
        logger.debug("JSON parsed via json_repair fallback")
        return repaired
    if isinstance(repaired, list) and list_key:
        if telemetry is not None:
            telemetry["repair_used"] = True
        logger.debug(
            "JSON parsed via json_repair fallback (bare array -> %s)",
            list_key,
        )
        return {list_key: repaired}

    return None


def _repair_json(text: str):
    """``json_repair.repair_json`` 呼出を例外フリーで包む。

    依存ライブラリの import 失敗 / 想定外例外は warning ログに記録した
    上で ``None`` を返し、呼出側で「戦略 4 不発」として扱えるようにする。
    """
    try:
        from json_repair import repair_json
    except ImportError:
        logger.debug("json_repair not installed; skipping repair fallback")
        return None
    try:
        return repair_json(text, return_objects=True)
    except (ValueError, TypeError) as e:
        logger.debug("json_repair failed: %s", e)
        return None
