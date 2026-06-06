"""コード検証（AST parse, import 整合性）

設計書 f_09_long_form_generation.md §7 準拠。
コード生成完了後にルールベースの静的検証を実施する。
テキスト生成には適用しない。
"""

from __future__ import annotations

import ast
import builtins
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger("backend.free.generation.validators")

PYTHON_BUILTINS: frozenset[str] = frozenset(dir(builtins))

# バッククォート 3 個以上の markdown コードフェンス区切り。言語タグ (python 等) は任意。
# CODE ユニットは区切り無しに連結されるため、閉じフェンス ``` と次ユニットの開きフェンス
# ```python が結合して ``````python (6 個) になったり、コード行末に ```python が結合する
# ことがある。`{3,}` で連数を問わず捕捉する。
# - 行全体がフェンスのみ → 行ごと削除
# - コード行末に結合したフェンス → 末尾フェンスのみ除去 (本文と改行は残す)
_FENCE_ONLY_LINE_RE = re.compile(r"^[ \t]*`{3,}[A-Za-z0-9_+\-.]*[ \t]*$")
_TRAILING_FENCE_RE = re.compile(r"[ \t]*`{3,}[A-Za-z0-9_+\-.]*[ \t]*$")


def remove_code_fences(code: str) -> str:
    """組み立て済みコードから markdown コードフェンス (```python / ``` / ``````python 等) を除去する。

    行全体がフェンスのみの行は丸ごと削除し、コード行末に結合したフェンス
    (例 ``ALIVE = 1```python``) は末尾フェンスのみ除去する。文字列リテラル内の
    ``` (例 ``x = "```"``) は行末に来ないため保持される。

    ``backend.free.llm.json_extract.strip_code_fences`` (単一ブロックの外側のみ剥がす
    JSON 用) とは別物。
    """
    out: list[str] = []
    for line in code.splitlines():
        if _FENCE_ONLY_LINE_RE.match(line):
            continue
        out.append(_TRAILING_FENCE_RE.sub("", line))
    return "\n".join(out)


# 退化反復 (LLM のトークン生成ループ) の検知しきい値。正常なコード/テキストでは
# 発火しない保守的な値にする。
_RUNAWAY_TOKEN_RUN = 50   # 1 行内で同一の短トークンが連続反復する回数
_RUNAWAY_LINE_REPEAT = 6  # 同一行が連続反復する回数
# <=4 字の短トークンが空白区切りで _RUNAWAY_TOKEN_RUN 回以上連続するパターン
# (例: "判定 1 1 1 ... 1" の数千回反復)。
_RUNAWAY_TOKEN_RE = re.compile(
    r"(\S{1,4})(?:[ \t]+\1){%d,}" % (_RUNAWAY_TOKEN_RUN - 1),
)


def collapse_runaway_repetition(text: str) -> str:
    """LLM の退化出力 (反復暴走) を切除する。

    ローカル LLM がトークン生成ループにはまった出力を 2 段階で抑制する:

    1. **行内トークン暴走**: 1 行内で短いトークン (<=4 字) が
       ``_RUNAWAY_TOKEN_RUN`` 回以上連続反復した場合、3 回 + ``…`` に切り詰める
       (例: ``判定 1 1 1 ... 1`` の数千回反復)。行単位の :func:`truncate_repetition`
       (EvorefLoop 側) では捕捉できない退化に対応する。
    2. **同一行反復**: 同一行 (strip 後) が ``_RUNAWAY_LINE_REPEAT`` 回以上連続した
       場合、最初の数回だけ残して以降を切除する。

    しきい値は正常なコードでは発火しない保守的な値。検証で残るエラーは別途
    ``GENERATION_ISSUES.md`` として提示される。
    """
    if not text:
        return text

    def _trim_run(m: re.Match) -> str:
        tok = m.group(1)
        return f"{tok} {tok} {tok} …"

    collapsed, token_runs = _RUNAWAY_TOKEN_RE.subn(_trim_run, text)

    lines = collapsed.split("\n")
    result: list[str] = []
    prev: str | None = None
    run = 0
    dropped_lines = 0
    for line in lines:
        stripped = line.strip()
        if stripped and stripped == prev:
            run += 1
            if run >= _RUNAWAY_LINE_REPEAT:
                dropped_lines += 1
                continue
        else:
            run = 1 if stripped else 0
            prev = stripped if stripped else prev
        result.append(line)
    out = "\n".join(result)
    if token_runs or dropped_lines:
        logger.warning(
            "Runaway repetition collapsed in generated unit "
            "(intra-line token-runs=%d, dropped-lines=%d, %d->%d chars)",
            token_runs, dropped_lines, len(text), len(out),
        )
    return out


@dataclass
class ValidationError:
    """検証エラー"""

    error_type: str
    message: str
    severity: str = "error"

    def __str__(self) -> str:
        return f"[{self.severity}:{self.error_type}] {self.message}"


def _extract_used_names(tree: ast.Module) -> set[str]:
    """AST から使用されている名前を抽出"""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            names.add(node.id)
    return names


def _target_names(target: ast.expr) -> set[str]:
    """代入 / ループ対象から束縛される名前を再帰的に抽出する。

    Tuple / List のアンパック (``a, b = ...`` / ``for a, b in ...``)、Starred
    (``a, *rest = ...``)、ネストした分解に対応する。Attribute / Subscript
    (``obj.x`` / ``d[k]``) は新規束縛ではないため無視する。
    """
    names: set[str] = set()
    if isinstance(target, ast.Name):
        names.add(target.id)
    elif isinstance(target, ast.Starred):
        names |= _target_names(target.value)
    elif isinstance(target, ast.Tuple | ast.List):
        for elt in target.elts:
            names |= _target_names(elt)
    return names


def _extract_defined_names(tree: ast.Module) -> set[str]:
    """AST から定義されている名前を抽出"""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                names |= _target_names(target)
        elif isinstance(node, ast.AnnAssign):
            names |= _target_names(node.target)
        elif isinstance(node, ast.NamedExpr):  # walrus (n := ...)
            names |= _target_names(node.target)
        elif isinstance(node, ast.For | ast.AsyncFor):
            names |= _target_names(node.target)
        elif isinstance(node, ast.With | ast.AsyncWith):
            for item in node.items:
                if item.optional_vars is not None:
                    names |= _target_names(item.optional_vars)
        elif isinstance(node, ast.comprehension):
            names |= _target_names(node.target)
        elif isinstance(node, ast.ExceptHandler):
            if node.name:  # except ... as e
                names.add(node.name)
    # 関数引数
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
                names.add(arg.arg)
            if node.args.vararg:
                names.add(node.args.vararg.arg)
            if node.args.kwarg:
                names.add(node.args.kwarg.arg)
    return names


def _extract_imported_names(tree: ast.Module) -> set[str]:
    """AST から import された名前を抽出"""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname if alias.asname else alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    continue
                names.add(alias.asname if alias.asname else alias.name)
    return names


def _annotation_root(expr: ast.expr) -> str | None:
    """型注釈の根の名前を返す (``ClassVar[int]`` → ``ClassVar`` 等)。"""
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        return expr.attr
    return None


def _is_dataclass_decorator(dec: ast.expr) -> bool:
    """``@dataclass`` / ``@dataclass(...)`` / ``@dataclasses.dataclass`` を判定する。"""
    target = dec.func if isinstance(dec, ast.Call) else dec
    return _annotation_root(target) == "dataclass"


def _collect_simple_dataclasses(
    tree: ast.Module,
) -> dict[str, tuple[set[str], set[str]]]:
    """同一モジュール内の「単純な」 dataclass を ``名前 → (全フィールド, 必須フィールド)``
    で収集する。

    継承ありクラス / ``ClassVar`` は init 引数の対応が複雑なため対象外 (誤検知回避)。
    デフォルト有りフィールド (``x: int = 0`` / ``= field(...)``) は任意扱い。
    """
    result: dict[str, tuple[set[str], set[str]]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if not any(_is_dataclass_decorator(d) for d in node.decorator_list):
            continue
        if node.bases:  # 継承は親フィールドを解決できないため対象外
            continue
        all_fields: set[str] = set()
        required: set[str] = set()
        for stmt in node.body:
            if not isinstance(stmt, ast.AnnAssign) or not isinstance(
                stmt.target, ast.Name,
            ):
                continue
            ann = stmt.annotation
            if (
                isinstance(ann, ast.Subscript)
                and _annotation_root(ann.value) == "ClassVar"
            ):
                continue
            all_fields.add(stmt.target.id)
            if stmt.value is None:  # デフォルト無し = 必須
                required.add(stmt.target.id)
        result[node.name] = (all_fields, required)
    return result


def _check_dataclass_calls(tree: ast.Module) -> list[ValidationError]:
    """同一モジュール内 dataclass のキーワード呼び出しを引数整合検証する。

    ``Block(shape=.., type=..)`` のような (a) 未知キーワード / (b) 必須欠落を検出する
    (AST の未定義名検出では捕捉できない実行時 ``TypeError`` の前倒し)。位置引数 /
    ``**kwargs`` を含む呼び出しは対応が曖昧なためスキップ (誤検知回避)。
    """
    dataclasses_map = _collect_simple_dataclasses(tree)
    if not dataclasses_map:
        return []
    errors: list[ValidationError] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        spec = dataclasses_map.get(node.func.id)
        if spec is None:
            continue
        if node.args or any(kw.arg is None for kw in node.keywords):
            continue  # 位置引数 / **kwargs ありはスキップ
        all_fields, required = spec
        provided = {kw.arg for kw in node.keywords if kw.arg}
        cls = node.func.id
        unknown = provided - all_fields
        missing = required - provided
        if unknown:
            errors.append(ValidationError(
                "dataclass-call",
                f"line {node.lineno}: {cls}() got unexpected keyword "
                f"argument(s): {sorted(unknown)}",
            ))
        if missing:
            errors.append(ValidationError(
                "dataclass-call",
                f"line {node.lineno}: {cls}() missing required "
                f"argument(s): {sorted(missing)}",
            ))
    return errors


def validate_python(assembled_code: str) -> list[ValidationError]:
    """Python コードの静的検証（LLM不要）"""
    errors: list[ValidationError] = []

    # 1. 構文検証
    try:
        tree = ast.parse(assembled_code)
    except SyntaxError as e:
        errors.append(
            ValidationError("syntax", f"line {e.lineno}: {e.msg}")
        )
        return errors

    # 2. 未定義名の検出
    used = _extract_used_names(tree)
    defined = _extract_defined_names(tree)
    imported = _extract_imported_names(tree)
    undefined = used - defined - imported - PYTHON_BUILTINS
    if undefined:
        errors.append(
            ValidationError("undefined", f"Undefined names: {sorted(undefined)}")
        )

    # 3. 未使用 import の検出（警告レベル）
    unused_imports = imported - used
    if unused_imports:
        errors.append(
            ValidationError(
                "unused_import",
                f"Unused imports: {sorted(unused_imports)}",
                severity="warning",
            )
        )

    # 4. 同一モジュール dataclass のキーワード呼び出し整合 (AST 超えの意味検証)
    errors.extend(_check_dataclass_calls(tree))

    return errors
