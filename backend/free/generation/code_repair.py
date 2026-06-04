"""生成コードの検証ゲート付きリペア

長文コード生成の最終段で assembled コードを検証し、エラーが残る場合に
アシストモデル (低温) で修正 → 再検証を最大 N ラウンド繰り返し、検出エラーが
最小の版を採用する。

- Python: ``validate_python`` (AST 構文 + 未定義名) で厳密検証し、各ラウンドで
  検出エラーが減った版のみ採用する (悪化させない)。
- 他言語 (TS/Svelte/Rust 等): AST 検証器が無いため、アシストによる軽い構文
  自己点検を 1 回だけ行う。長さが極端に変動した応答 (prose 混入 / truncation)
  は破棄して原文を維持する。

アシスト不在 (degraded) / 機能無効 / 例外時はいずれも原文をそのまま返す
(出力経路を止めず、品質を悪化させない)。
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from backend.free.generation.validators import (
    ValidationError,
    remove_code_fences,
    validate_python,
)
from backend.log_config import get_logger
from backend.utils import estimate_tokens

if TYPE_CHECKING:
    from backend.debug_logger import DebugLogger

logger = get_logger("generation.code_repair")

# 修正後コード出力に許す最大トークン (元コード規模 + 余裕、上限でガード)。
_MAX_OUTPUT_TOKENS = 6144

# 拡張子 → 言語ラベル (assist 自己点検プロンプト + Python 判定)。
_EXT_LANG: dict[str, str] = {
    "py": "python", "pyi": "python",
    "ts": "TypeScript", "tsx": "TypeScript",
    "js": "JavaScript", "jsx": "JavaScript", "mjs": "JavaScript",
    "svelte": "Svelte", "vue": "Vue",
    "rs": "Rust", "go": "Go", "java": "Java", "kt": "Kotlin",
    "c": "C", "h": "C", "cpp": "C++", "cc": "C++", "hpp": "C++",
    "rb": "Ruby", "php": "PHP", "cs": "C#", "swift": "Swift",
    "sh": "Bash", "bash": "Bash", "sql": "SQL",
    "css": "CSS", "scss": "SCSS", "html": "HTML", "yaml": "YAML", "yml": "YAML",
}

_PYTHON_REPAIR_PROMPT = """\
以下の Python コードに静的検証で検出されたエラーがあります。
エラーを修正した「完全なコード全体」を出力してください。

制約:
- 出力はコードのみ。説明・マークダウン・コードフェンスは付けないでください。
- ロジックと構造は元のコードを尊重し、エラー修正に必要な最小限の変更に留めること。
- 未定義名のエラーは、必要な import の追加か、呼び出し箇所の修正で解消すること。

# 検出されたエラー
{errors}

# コード
{code}

修正後のコード全体:"""

_GENERIC_SELFCHECK_PROMPT = """\
以下の {language} コードに明らかな構文エラーがあれば修正した「完全なコード全体」を、
無ければ元のコードをそのまま出力してください。

制約:
- 出力はコードのみ。説明・マークダウン・コードフェンスは付けないでください。
- ロジックと構造は変更せず、構文の修復のみ行うこと。

# コード
{code}

出力 (コードのみ):"""


def infer_language(file_paths: list[str]) -> str:
    """CodeUnit.file_path 群から代表言語ラベルを推定する (既定 ``python``)。"""
    counts: dict[str, int] = {}
    for p in file_paths:
        if not p:
            continue
        ext = PurePosixPath(p).suffix.lstrip(".").lower()
        if ext:
            counts[ext] = counts.get(ext, 0) + 1
    if not counts:
        return "python"
    top_ext = max(counts, key=lambda k: counts[k])
    return _EXT_LANG.get(top_ext, "python")


def _py_error_count(
    code: str, *, syntax_only: bool = False,
) -> tuple[int, list[ValidationError]]:
    errors = validate_python(code)
    if syntax_only:
        # 複数ファイル分割時、他ファイル定義シンボルの参照が undefined と誤検知
        # されるため、ファイル単位の修正は構文エラーのみを対象にする。
        n = sum(
            1 for e in errors
            if e.severity == "error" and e.error_type == "syntax"
        )
    else:
        n = sum(1 for e in errors if e.severity == "error")
    return n, errors


def _extract_content(resp: object) -> str:
    """AssistModelClient.generate の OAI 互換 dict から content を取り出す。"""
    if isinstance(resp, dict):
        choices = resp.get("choices") or []
        if choices and isinstance(choices[0], dict):
            return (choices[0].get("message") or {}).get("content", "") or ""
    return ""


class CodeRepairer:
    """検証ゲート付きのコードリペア。"""

    def __init__(
        self, assist_client, config: dict | None = None, *,
        debug_logger: DebugLogger | None = None,
    ):
        # 監査 (_is_assist_receiver) がレシーバ名で検出できるよう ``_assist_client``。
        self._assist_client = assist_client
        self._lf = (config or {}).get("long_form", {})
        self._debug_logger = debug_logger

    async def repair(
        self, assembled: str, *, language: str = "python", syntax_only: bool = False,
    ) -> str:
        """assembled を検証→修正→再検証し、エラー最小版を返す。例外時は原文。

        ``syntax_only=True`` (複数ファイル分割時) は Python の未定義名チェックを
        スキップし構文エラーのみ修正する (他ファイル定義シンボルの参照を
        undefined と誤検知して不要な placeholder を生成するのを防ぐ)。
        """
        if not self._lf.get("repair_enabled", True):
            return assembled
        if self._assist_client is None or not assembled.strip():
            return assembled
        max_rounds = int(self._lf.get("max_repair_rounds", 2))
        if max_rounds <= 0:
            return assembled
        try:
            if language == "python":
                return await self._repair_python(assembled, max_rounds, syntax_only)
            return await self._selfcheck_generic(assembled, language)
        except Exception as e:  # noqa: BLE001 — リペアは出力経路を止めない
            logger.warning("code repair failed, returning original: %s", e)
            return assembled

    async def _repair_python(
        self, code: str, max_rounds: int, syntax_only: bool = False,
    ) -> str:
        best = code
        best_n, errors = _py_error_count(best, syntax_only=syntax_only)
        if best_n == 0:
            return best  # 対象エラー無し → 何もしない
        attempts = 0
        for _ in range(max_rounds):
            err_lines = [str(e) for e in errors if e.severity == "error"][:10]
            candidate = await self._ask(
                _PYTHON_REPAIR_PROMPT.format(
                    errors="\n".join(err_lines), code=best,
                ),
                code,
            )
            attempts += 1
            if not candidate:
                break
            n, errs = _py_error_count(candidate, syntax_only=syntax_only)
            if n < best_n:
                best, best_n, errors = candidate, n, errs
            if best_n == 0:
                break
        self._log(language="python", attempts=attempts, final_error_count=best_n)
        return best

    async def _selfcheck_generic(self, code: str, language: str) -> str:
        candidate = await self._ask(
            _GENERIC_SELFCHECK_PROMPT.format(language=language, code=code), code,
        )
        # 検証器が無いため保守的に: 長さが大きく乖離した応答 (prose 混入 /
        # truncation) は採用せず原文を維持する。
        accepted = bool(candidate) and (
            0.5 * len(code) <= len(candidate) <= 1.8 * len(code)
        )
        self._log(
            language=language, attempts=1,
            final_error_count=-1, accepted=accepted,
        )
        return candidate if accepted else code

    async def _ask(self, prompt: str, code: str) -> str:
        max_tokens = min(_MAX_OUTPUT_TOKENS, max(512, int(estimate_tokens(code) * 1.4)))
        resp = await self._assist_client.generate(
            [{"role": "user", "content": prompt}],
            purpose="code_repair",
            temperature=0.2,
            max_tokens=max_tokens,
        )
        return remove_code_fences(_extract_content(resp)).strip()

    def _log(self, **fields) -> None:
        if self._debug_logger is not None:
            self._debug_logger.log_long_form_event({"phase": "code_repair", **fields})
