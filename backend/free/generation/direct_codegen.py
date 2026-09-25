"""staged クリエイトパイプライン専用の単発ファイル生成。

``LongFormOrchestrator`` (CogWriterStrategy) はモジュール内部で instruction から
独自に plan JSON / CodeSpec を **再合成**し、CodeUnit (関数/クラス粒度) に分割して
個別生成・連結する。この再合成は補助タスク LLM を介した lossy な圧縮であり、
staged executor が組み立てた instruction (spec.md 全文 + flowchart + 契約ブロック)
の大半は base モデルのユニット生成プロンプトに一切届かない
(``build_code_unit_messages`` が見るのは再合成後の CodeSpec render / plan 由来の
unit.spec のみ)。さらに CodeUnit 分割・再連結は import 重複や機能重複などの結合
不整合を生む。

staged は ``synthesize_create_task_graph`` が既にプログラムをファイル単位へ決定的に
分解済みであり、1 code タスク = 1 ファイルの単発生成で足りる。本モジュールは
その再計画・再合成・分割連結を経由せず、base モデルへの単発呼び出しのみで完結する
軽量パスを提供する。instruction は無劣化のままプロンプトへ渡る。
"""

from __future__ import annotations

import ast
import logging
import re

from backend.free.core.prompt_blocks import split_shared_context
from backend.free.generation.validators import is_degenerate_repetition, remove_code_fences
from backend.free.llm.utils import extract_content

logger = logging.getLogger("backend.free.generation.direct_codegen")

_SYSTEM_PROMPT = (
    "You are an expert programmer. Follow the design specification and "
    "instructions in the user message exactly and completely. Output ONLY the "
    "source code for the requested file. The program must work on a first run "
    "in an empty directory: create parent directories before writing a file, "
    "and initialize missing data files instead of failing."
)

# 切断時の再生成で許す max_tokens 上限。
_MAX_TOKENS_CEILING = 16384

#: 反復ループで切れた出力を作り直すときの温度の下限。
_LOOP_RETRY_TEMPERATURE = 0.6

#: 反復の判定から外す拡張子 (データ / 文書は正当な反復構造を持つ)。
_REPETITIVE_BY_NATURE_SUFFIXES = (".json", ".csv", ".tsv", ".yaml", ".yml", ".md", ".txt", ".svg")


def _looping(code: str, file_path: str) -> bool:
    """切れた出力が反復ループか (データ / 文書ファイルは判定しない)。"""
    if file_path.lower().endswith(_REPETITIVE_BY_NATURE_SUFFIXES):
        return False
    return is_degenerate_repetition(code)

# 非ストリーミング呼び出しの per-request タイムアウト算出パラメータ。
# LocalClient の既定タイムアウト (120s) は decode 速度の速い環境向けで、iGPU 等
def _finish_reason(resp: dict) -> str:
    """base client generate() 応答の finish_reason ('length' は max_tokens 切断)。"""
    try:
        fr = resp["choices"][0].get("finish_reason")
        return fr if isinstance(fr, str) else ""
    except (KeyError, IndexError, TypeError, AttributeError):
        return ""


def _reasoning_content(resp: dict) -> str:
    """base client generate() 応答の reasoning_content ('<think>' 分離出力)。"""
    try:
        rc = resp["choices"][0]["message"].get("reasoning_content")
        return rc if isinstance(rc, str) else ""
    except (KeyError, IndexError, TypeError, AttributeError):
        return ""


_CODE_FENCE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def _parses_to_nonempty_module(code: str) -> bool:
    try:
        return bool(ast.parse(code).body)
    except (SyntaxError, ValueError):
        return False


def _salvage_code_from_reasoning(reasoning_content: str) -> str:
    """打ち切られた reasoning から、最後に書かれた完成コードブロックを救出する。

    coder モデル (Qwopus3.5 等) は推論モデルであり、``<think>`` 内で一度コード
    を書き下してから自己ツッコミ・再検討を続けることがある (2026-07-23 live:
    gcd_calculator/fibonacci_generator の生成で確認 — max_tokens が reasoning
    だけで尽き、可視の ``content`` が空のまま finish_reason='length' で切断
    された。reasoning_content 側には既に完成した正しい関数が書かれていた)。
    reasoning 内の fenced code block を末尾から走査し、構文的に完成している
    (``ast.parse`` が通り body が空でない) 最初の候補を採用する (best-effort、
    無ければ空文字)。
    """
    for block in reversed(_CODE_FENCE_BLOCK_RE.findall(reasoning_content)):
        candidate = block.strip()
        if candidate and _parses_to_nonempty_module(candidate):
            return candidate
    return ""


def _is_test_file(file_path: str) -> bool:
    name = file_path.replace("\\", "/").rsplit("/", 1)[-1]
    return name.startswith("test_") and name.endswith(".py")


def _complete_test_prefix(code: str) -> str:
    """切断されたテストコードから、構文的に完結した先頭部分を返す (無ければ空)。

    トップレベル文の境界 (インデント無しの行) を末尾から遡り、``ast.parse`` が
    通って ``def test_`` を 1 つ以上含む最長の接頭辞を採る。
    """
    lines = code.splitlines()
    for end in range(len(lines) - 1, 0, -1):
        line = lines[end]
        if not line or line[0].isspace() or line.startswith(("#", ")", "]", "}")):
            continue
        prefix = "\n".join(lines[:end]).rstrip() + "\n"
        try:
            tree = ast.parse(prefix)
        except (SyntaxError, ValueError):
            continue
        if any(
            isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test_")
            or isinstance(n, ast.ClassDef) and n.name.startswith("Test")
            for n in tree.body
        ):
            return prefix
    return ""


def _extract_code(resp: dict, file_path: str) -> tuple[str, bool]:
    """応答からコードを取り出す。戻り値は ``(code, from_salvage)``。

    可視 ``content`` が空なら reasoning から救出する。``from_salvage=True`` は
    ``ast.parse`` で完成を検証済みという意味で、``finish_reason == "length"``
    でも不完全な断片ではないと信頼してよい (直接 content から取った切断済み
    テキストとは区別する — 後者は語尾が切れた不完全コードの可能性がある)。
    """
    code = remove_code_fences(extract_content(resp).strip())
    if code:
        return code, False
    salvaged = _salvage_code_from_reasoning(_reasoning_content(resp))
    if salvaged:
        logger.info(
            "direct codegen recovered code from truncated reasoning_content "
            "for %s (visible content was empty)", file_path,
        )
    return salvaged, bool(salvaged)


async def generate_single_file(
    client,
    instruction: str,
    file_path: str,
    *,
    max_tokens: int = 4096,
    temperature: float = 0.3,
    request_timeout: float | None = None,
    id_slot: int | None = None,
) -> dict[str, str]:
    """instruction から単一ファイルのコードを base モデルへの 1 回の呼び出しで生成する。

    Args:
        client: base LLM client (``LocalClient`` 互換、``generate()``/``longform_slot``
            を持つ)。
        instruction: 呼出側 (staged executor) が組み立てた完全な生成指示
            (spec.md 全文 + flowchart + 契約ブロックを含む)。加工・再合成せず
            渡す。``SHARED_CONTEXT_BOUNDARY`` があればその前を system に、後を
            user に置く (無ければ全文を user へ)。
        file_path: 生成対象のファイル論理パス。戻り値の辞書キーに使う
            (呼出側は常にこのキーで結果を取得できる)。
        max_tokens: 初回生成の最大トークン。
        temperature: 生成温度。
        id_slot: 使うスロット。既定 (``None``) は ``client.longform_slot``。staged v2 が
            独立したモジュールを別スロットで同時生成するときに指定する (f_10 §11)。
            チャットスロットは渡さない (不変則 #1)。
        request_timeout: 呼出予算 (f_10 §3)。``LocalClient.generate`` の
            ``request_timeout`` へそのまま渡す。``None`` (既定) は
            ``sync_request_timeout`` (実質無制限) に委ねる。切断時の再生成
            呼出にも同じ値を使う (再計算しない)。

    Returns:
        ``{file_path: code}``。生成失敗 / 空応答時は空 dict。再生成 (retry) 後も
        ``finish_reason == "length"`` (再切断) の場合も空 dict を返す (不完全
        コードは成果物として返さない)。

    応答が ``finish_reason == "length"`` (max_tokens 切断) の場合のみ、予算を倍に
    広げて 1 回だけ再生成する (``staged.executor._generate_spec_doc`` と同じ方針)。
    """
    # 境界の前 (run 内で共有する brief + spec) は system へ載せ、同じスロットの
    # 連続呼出で接頭辞 KV を再利用させる。system は文脈ガードでも捨てられない。
    slot = client.longform_slot if id_slot is None else id_slot
    shared, task_instruction = split_shared_context(instruction)
    messages = [
        {"role": "system",
         "content": f"{_SYSTEM_PROMPT}\n\n{shared}" if shared else _SYSTEM_PROMPT},
        {"role": "user", "content": task_instruction},
    ]
    try:
        resp = await client.generate(
            messages, stream=False, max_tokens=max_tokens, temperature=temperature,
            # chat_slot だとチャット接頭辞 KV を破壊する退行 (f_08 §2.2 実測、
            # f_10 §0)。staged の codegen は long_form 専有スロットを使う。
            id_slot=slot,
            request_timeout=request_timeout,
        )
    except Exception as exc:
        logger.warning("direct codegen failed for %s: %s", file_path, exc)
        return {}

    code, code_from_salvage = _extract_code(resp, file_path)

    # 切断されたテストファイルは、完結した test 関数までを採る (倍額再生成しない)。
    # テストは列挙が止まらず 4096 → 8192 とも上限まで伸び、約 20 分を使って
    # 破棄されていた (2026-09-19 ライブ監査で 4/4 テーマ)。
    if _finish_reason(resp) == "length" and not code_from_salvage and _is_test_file(file_path):
        complete = _complete_test_prefix(code)
        if complete:
            logger.warning(
                "direct codegen truncated at max_tokens=%d for %s; kept the "
                "complete test functions (%d chars) instead of regenerating",
                max_tokens, file_path, len(complete),
            )
            return {file_path: complete}

    # reasoning から検証済みで救出できた場合は、切断されていても完成コードと
    # して信頼できるため、費用のかかる倍額再生成をスキップする (2026-07-23
    # live: gcd_calculator/fibonacci_generator が reasoning だけで max_tokens
    # を使い切り、直接 content は空だが reasoning 内には既に完成した正しい
    # 関数が書かれていた。再生成を重ねても同じパターンで再度失敗し、
    # 最終的に「コードが生成されませんでした」まで至っていた)。
    if _finish_reason(resp) == "length" and not code_from_salvage:
        retry_tokens = min(max_tokens * 2, _MAX_TOKENS_CEILING)
        retry_temperature = temperature
        if _looping(code, file_path):
            # 反復ループで上限に達した出力。倍の上限で作り直すとループが倍の時間
            # 続きうるので、同じ上限・高めの温度で 1 回だけ作り直す (2026-09-22
            # 実機 K04: storage.js が 4096 トークンぶん 7.9 分ループした。同じ
            # プロンプトでもループは再現しなかったので、作り直し自体は有効)。
            retry_tokens = max_tokens
            retry_temperature = max(temperature, _LOOP_RETRY_TEMPERATURE)
            logger.warning(
                "direct codegen hit max_tokens=%d in a repetition loop for %s; "
                "regenerating at the same budget with temperature %.1f",
                max_tokens, file_path, retry_temperature,
            )
        elif retry_tokens > max_tokens:
            logger.warning(
                "direct codegen truncated at max_tokens=%d for %s; "
                "regenerating at %d", max_tokens, file_path, retry_tokens,
            )
        if retry_tokens > max_tokens or retry_temperature != temperature:
            try:
                resp2 = await client.generate(
                    messages, stream=False, max_tokens=retry_tokens,
                    temperature=retry_temperature, id_slot=slot,
                    request_timeout=request_timeout,
                )
                retry_code, retry_from_salvage = _extract_code(resp2, file_path)
                if _finish_reason(resp2) == "length" and not retry_from_salvage:
                    # 再生成も切断され、reasoning からの検証済み救出も得られな
                    # かった = 完成コードではなく壊れた断片 (直接 content が
                    # 非空でも未検証の切断済みテキストに過ぎない)。長さで
                    # 「マシ」に見えても SyntaxError を伴う不完全ファイルを完了
                    # 扱いで返さない (2026-07-22: test_email_validator.py が
                    # f-string 途中で切れたまま「成果物は配信します」扱いで出荷
                    # された実害の再発防止)。空扱いにして呼出側 (staged executor
                    # の empty-code フォールバック) に委ねる。
                    logger.warning(
                        "direct codegen still truncated after retry at "
                        "max_tokens=%d for %s; discarding incomplete code",
                        retry_tokens, file_path,
                    )
                    return {}
                # 作り直しが完結したら長さに関係なくそれを採る。「長い方」を
                # 採っていたため、切れたループ出力 (9942 字) が正常終了した
                # 作り直し (5947 字) に勝ち、構文エラーで捨てられた (2026-09-22
                # 実機 K04)。
                if retry_code.strip():
                    code = retry_code
            except Exception as exc:
                logger.warning(
                    "direct codegen regeneration failed for %s: %s", file_path, exc,
                )

    if not code.strip():
        return {}
    return {file_path: code}
