"""生成コードの検証ゲート付きリペア

長文コード生成の最終段で assembled コードを検証し、エラーが残る場合に
補助タスク (低温) で修正 → 再検証を最大 N ラウンド繰り返し、検出エラーが
最小の版を採用する。

- Python: ``validate_python`` (AST 構文 + 未定義名) で厳密検証し、各ラウンドで
  検出エラーが減った版のみ採用する (悪化させない)。
- 他言語 (TS/Svelte/Rust 等): AST 検証器が無いため、補助タスクによる軽い構文
  自己点検を 1 回だけ行う。長さが極端に変動した応答 (prose 混入 / truncation)
  は破棄して原文を維持する。

補助タスク不在 (degraded) / 機能無効 / 例外時はいずれも原文をそのまま返す
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

# aux context_size に対する安全マージン (chat template / 特殊トークン分)。
_CONTEXT_SAFETY_MARGIN = 256

# 拡張子 → 言語ラベル (aux 自己点検プロンプト + Python 判定)。
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
- 元のコードが実装している機能を削除しないこと。エラーを消すために該当箇所を丸ごと
  削除する・スタブ化するのは不可。

{spec_block}# 検出されたエラー
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


# 複数ファイル分割時に「ファイル単位で安全に修正できる」 intra-file エラー種別。
# undefined は他ファイル定義シンボル参照 (cross-file) の誤検知を含むため除外し、
# wire_imports の再配線に委ねる。dataclass-call / syntax は同一ファイル内で完結する
# ため複数ファイル時も修正対象に含める。
_INTRA_FILE_ERROR_TYPES: frozenset[str] = frozenset({"syntax", "dataclass-call"})


def _py_error_count(
    code: str, *, intra_file_only: bool = False,
) -> tuple[int, list[ValidationError]]:
    errors = validate_python(code)
    if intra_file_only:
        n = sum(
            1 for e in errors
            if e.severity == "error" and e.error_type in _INTRA_FILE_ERROR_TYPES
        )
    else:
        n = sum(1 for e in errors if e.severity == "error")
    return n, errors


def _extract_content(resp: object) -> str:
    """AuxClient.generate の OAI 互換 dict から content を取り出す。"""
    if isinstance(resp, dict):
        choices = resp.get("choices") or []
        if choices and isinstance(choices[0], dict):
            return (choices[0].get("message") or {}).get("content", "") or ""
    return ""


class CodeRepairer:
    """検証ゲート付きのコードリペア。"""

    def __init__(
        self, aux_client, config: dict | None = None, *,
        debug_logger: DebugLogger | None = None,
    ):
        # 監査 (_is_aux_receiver) がレシーバ名で検出できるよう ``_aux_client``。
        self._aux_client = aux_client
        self._lf = (config or {}).get("long_form", {})
        self._debug_logger = debug_logger

    async def repair(
        self, assembled: str, *, language: str = "python",
        intra_file_only: bool = False, spec: str = "",
    ) -> str:
        """assembled を検証→修正→再検証し、エラー最小版を返す。例外時は原文。

        ``intra_file_only=True`` (複数ファイル分割時) は cross-file 参照の誤検知を
        含む undefined を修正対象から外し、同一ファイル内で完結するエラー
        (構文 / dataclass 引数不整合) のみ修正する。undefined の解決は後段の
        ``wire_imports`` に委ねる (他ファイル定義シンボルへの placeholder 乱造を防ぐ)。
        ``spec`` (呼出側の共有設計仕様レンダリング、任意) を渡すと Python リペア
        プロンプトに同梱し、エラー消去だけを目標にした契約逸脱の修正を抑止する。
        本メソッドはローカルな構文/未定義名エラーの機械的修正が対象で、モジュール間
        制御フローの逸脱検出は対象外のため flowchart は受け取らない
        (``revise_unit`` との非対称は意図的、2ec746b 参照)。
        """
        if not self._lf.get("repair_enabled", True):
            return assembled
        if self._aux_client is None or not assembled.strip():
            return assembled
        max_rounds = int(self._lf.get("max_repair_rounds", 2))
        if max_rounds <= 0:
            return assembled
        try:
            if language == "python":
                return await self._repair_python(
                    assembled, max_rounds, intra_file_only, spec,
                )
            return await self._selfcheck_generic(assembled, language)
        except Exception as e:
            logger.warning("code repair failed, returning original: %s", e)
            return assembled

    async def _repair_python(
        self, code: str, max_rounds: int, intra_file_only: bool = False,
        spec: str = "",
    ) -> str:
        best = code
        best_n, errors = _py_error_count(best, intra_file_only=intra_file_only)
        if best_n == 0:
            return best  # 対象エラー無し → 何もしない
        spec_block = f"# 設計仕様 (契約 — 遵守すること)\n{spec}\n\n" if spec else ""
        attempts = 0
        for _ in range(max_rounds):
            # intra_file_only 時はプロンプトにも対象種別のエラーのみ提示し、cross-file
            # undefined を見せて placeholder を捏造させない。
            err_lines = [
                str(e) for e in errors
                if e.severity == "error"
                and (not intra_file_only or e.error_type in _INTRA_FILE_ERROR_TYPES)
            ][:10]
            candidate = await self._ask(
                _PYTHON_REPAIR_PROMPT.format(
                    spec_block=spec_block, errors="\n".join(err_lines), code=best,
                ),
                code,
            )
            attempts += 1
            if not candidate:
                break
            # エラー数が減っても、元コードの半分未満に縮んだ候補は機能削除による
            # 見せかけの改善の疑いが強いため採用しない (_selfcheck_generic の
            # 0.5〜1.8 倍ガードと同じ考え方)。
            if len(candidate) < 0.5 * len(best):
                logger.warning(
                    "code repair candidate shrank drastically (%d -> %d chars); "
                    "rejecting despite error count change",
                    len(best), len(candidate),
                )
                continue
            n, errs = _py_error_count(candidate, intra_file_only=intra_file_only)
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
        context_size = getattr(self._aux_client, "context_size", 8192)
        prompt_tokens = estimate_tokens(prompt)
        if prompt_tokens + max_tokens > context_size - _CONTEXT_SAFETY_MARGIN:
            logger.warning(
                "code repair skipped: prompt=%d tok + output budget=%d tok "
                "exceeds aux context_size=%d; returning original",
                prompt_tokens, max_tokens, context_size,
            )
            return ""
        resp = await self._aux_client.generate(
            [{"role": "user", "content": prompt}],
            purpose="code_repair",
            temperature=0.2,
            max_tokens=max_tokens,
        )
        return remove_code_fences(_extract_content(resp)).strip()

    def _log(self, **fields) -> None:
        if self._debug_logger is not None:
            self._debug_logger.log_long_form_event({"phase": "code_repair", **fields})
