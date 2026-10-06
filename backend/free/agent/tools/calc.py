"""``calculate`` ツール — AST 許可リストによる安全な算術式評価"""

from __future__ import annotations

import ast
import math

from backend.free.agent.tools.calc_ops import CALC_OPS, STRING_RESULT_OPS


# 安全な計算用に許可するノード
_SAFE_NODES = {
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.FloorDiv,
    ast.USub, ast.UAdd,
    # Call / Name は許可リスト照合を通ったものだけを受け入れる (_validate_call
    # / _SAFE_NAMES 参照)。Attribute は許可しないため obj.attr 経由での脱出は不可。
    ast.Call, ast.Name, ast.Load,
}

# 数値だけを受け取り数値を返す純粋関数と数学定数の許可リスト。
#
# Call を一律拒否していたため、LLM が log2 / sqrt を含む式を書くたびに
# 「Unsafe expression」で失敗し、手計算にフォールバックして誤答していた
# (実測 2026-07-26: ISO 3200→6400 の段数を聞かれ calculate が 2 連続で失敗し、
#  「約 1.3 段 (1 段弱)」という誤りかつ自己矛盾した回答になった。正しくは 1 段)。
#
# ``__builtins__`` は空のままで、ここに載せた callable / 定数以外は名前解決
# できない。Attribute ノードを許可しないため ``x.__class__`` 等の経由もできない。
# ``**`` が既に許可されている以上、pow / factorial による巨大値生成のリスクは
# 現状から増えない。
#: ``sum`` / ``len`` は **リスト・タプルのリテラル 1 個だけを引数に取る形** で呼べる
#: (:data:`_SEQUENCE_FUNCS`)。以前は列を作れず構造的に到達不能だったので載せて
#: いなかった (2026-08-27 ライブ監査: ``sum([1, 2, 3])`` が disallowed node: List)。
#: 9B の分類器は会話の数を合計するとき ``sum(...)`` を組み、未知の名前で落ちて
#: 「計算ツールが利用できない」と答えていた (2026-10-03 再実行 D08#4)。
#: 列を **返す** 呼び方を作らないのが要点 — ``max([0], [1])`` / ``sum([], [0])`` は
#: 列を返し、その戻り値は ``* 10**9`` の被演算子にできる (独立レビュー 2026-10-03:
#: ``len(max((0,), (1,)) * 10**7)`` が 0.03 秒で 1000 万要素)。引数が列リテラル 1 個
#: なら、入れ子の列は拒むので戻り値は要素 (数か文字列) か数になる。文字列・列の
#: 繰り返しと連結は結果の長さで別に打ち切る (:data:`_MAX_SEQUENCE_RESULT_LEN`)。
#: ``range`` は ``sum`` / ``min`` / ``max`` / ``len`` の唯一の引数にだけ許し、長さを
#: :data:`_MAX_RANGE_LEN` で打ち切る (:data:`_RANGE_FUNC`、2026-10-05 ライブ監査:
#: 「1から100までの奇数の和」を ``sum(i for i in range(1, 101) if i % 2 != 0)`` と
#: 組み、生成式で落ちて答えを諦めた。``sum(range(1, 101, 2))`` なら書ける)。
#: ``str`` は数の桁数 (``len(str(factorial(37)))``) のため。整数の文字列化は
#: ``sys.int_max_str_digits`` (既定 4300 桁) が上限になり、結果の文字列は繰り返し・
#: 連結の長さの上限 (:data:`_MAX_SEQUENCE_RESULT_LEN`) にも掛かる。
_SAFE_NAMES: dict[str, object] = {
    "abs": abs, "round": round, "min": min, "max": max,
    "sum": sum, "len": len, "range": range, "str": str,
    "pow": pow,
    "sqrt": math.sqrt, "exp": math.exp,
    "log": math.log, "log2": math.log2, "log10": math.log10,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "floor": math.floor, "ceil": math.ceil,
    "factorial": math.factorial, "gcd": math.gcd, "hypot": math.hypot,
    "degrees": math.degrees, "radians": math.radians,
    "pi": math.pi, "e": math.e, "tau": math.tau,
    # 整数の基数変換は決定論で閉じる。無いと「2 の 20 乗を 16 進数で」が
    # ``hex(2**20)`` → Unsafe expression (unknown name: hex) で失敗し、暗算
    # フォールバックに落ちる (2026-09-10 ライブ監査 (i) I-07)。``int`` は
    # ``int("ff", 16)`` の逆変換用 (文字列リテラルは Constant として許可済み)。
    "hex": hex, "bin": bin, "oct": oct, "int": int,
    # 時間長・文字数・日付の演算 (calc_ops)。小型モデルが暗算で誤る数え上げ・繰り上がり・
    # 前後比較を、新しいツールを足さずにこの評価器へ寄せる (docs/f_03 §3.2)。
    **CALC_OPS,
}

#: リスト・タプルのリテラルを受け取れる関数。列のリテラルは、ここに載る関数の
#: **唯一の** 引数であるときだけ許す (2 個以上の引数の列・入れ子の列・演算の被演算子・
#: 他の関数の引数は拒む)。
_SEQUENCE_FUNCS = frozenset({"sum", "min", "max", "len"})
_SEQUENCE_NODES = (ast.List, ast.Tuple)
#: 列を生む関数。列のリテラルと同じく :data:`_SEQUENCE_FUNCS` の唯一の引数のときだけ
#: 呼べる (``range(10)`` 単独・``range(3) * 2``・``str(range(3))`` は拒む)。
_RANGE_FUNC = "range"
#: ``range`` の長さの上限。``sum(range(10**12))`` を素通しすると eval がスレッドを
#: 長時間占有する (calculate にタイムアウトは無い)。
_MAX_RANGE_LEN = 100_000
#: 式の前に置かれても落とす文 (``import math; math.factorial(37)``)。モデルは
#: Python の書き癖で ``import math`` を前置し、``math.`` を付けて呼ぶ (2026-10-05
#: ライブ監査: 「37の階乗の桁数」が構文エラーで落ち、会話の全ての数を尋ね返した)。
_MATH_MODULE = "math"

# 非許可ノードごとの自己修正ヒント。LLM が同一ターン内でエラーを見て
# 書き直せるよう、そのノードが生じがちな典型的な誤記法を指す (実インシデント:
# 「πr²」を "π*5^2" と書いて BitXor に、「GCD」を "gcd(360,504)" と書いて
# Call になり、いずれもエラー後は手計算にフォールバックしていた)。
_DISALLOWED_NODE_HINTS: dict[str, str] = {
    "BitXor": "use ** for exponentiation, not ^ (^ is bitwise XOR here)",
    # 一覧は _SAFE_NAMES から生成する (直書きすると許可リストとドリフトする)
    "Call": "only these names are available: " + " ".join(sorted(_SAFE_NAMES)),
    "Name": "only these names are available: " + " ".join(sorted(_SAFE_NAMES)),
    "List": "a list literal is allowed only as the single argument of "
            + "/".join(sorted(_SEQUENCE_FUNCS)) + ", e.g. sum([12, 25, 8])",
    "Tuple": "a tuple literal is allowed only as the single argument of "
             + "/".join(sorted(_SEQUENCE_FUNCS)),
    # 生成式・内包表記は許可しない (検証面が広い。セキュリティレビュー待ち)。
    # 等差の列は range の刻みで書ける (2026-10-05 ライブ監査: 奇数の和)。
    "GeneratorExp": "generator expressions are not supported; use range with a "
                    "step instead, e.g. sum(range(1, 101, 2)) for the odd numbers "
                    "1..99, or list the values: sum([1, 3, 5])",
    "ListComp": "list comprehensions are not supported; use range with a step "
                "instead, e.g. sum(range(1, 101, 2)), or list the values: "
                "sum([1, 3, 5])",
    "Attribute": "attribute access is not supported; call the function directly "
                 "without a module prefix, e.g. factorial(37) instead of "
                 "math.factorial(37)",
    "Import": "write a single expression without import statements; the "
              "functions are available directly, e.g. len(str(factorial(37)))",
    "ImportFrom": "write a single expression without import statements; the "
                  "functions are available directly, e.g. len(str(factorial(37)))",
}
#: 整数の文字列化の上限 (``sys.int_max_str_digits``) を超えたときの理由。
_TOO_MANY_DIGITS = (
    "Error: the number has too many digits to convert to a string -- for the digit "
    "count of a large integer use floor(log10(x)) + 1, e.g. "
    "floor(log10(factorial(3000))) + 1"
)
#: 式でない文 (代入・import・関数定義など) に付けるヒント。
_STATEMENT_HINT = (
    "write exactly one expression (no statements such as import or assignment), "
    "e.g. len(str(factorial(37)))"
)


#: 評価コストの上限。``**`` / ``pow`` / ``factorial`` は許可リストに載って
#: いるので、``10**10**10`` や ``factorial(10**6)`` のような式を素通しすると
#: eval がイベントループのスレッドプールを長時間占有する (chat 応答パスの
#: 同期ツールは ``to_thread`` で走る)。指数・引数の大きさと結果のビット数で
#: 事前に打ち切る。
_MAX_POW_EXPONENT = 10_000
_MAX_FACTORIAL_ARG = 5_000
_MAX_INT_POW_RESULT_BITS = 1_000_000
#: 文字列・バイト列・列の繰り返し (``*``) と連結 (``+``) の結果の長さの上限。
#: ``'a' * 10**9`` は数値の検査をすべて通り、1 GB の文字列を作っていた (calculate に
#: タイムアウトは無い。独立レビュー 2026-10-03)。
_MAX_SEQUENCE_RESULT_LEN = 100_000
_SIZED_TYPES = (str, bytes, list, tuple)


def _eval_subtree(node: ast.expr) -> object:
    """許可リスト検証済みの部分木を評価する (コスト検査用)。"""
    expr = ast.fix_missing_locations(ast.Expression(body=node))
    return eval(  # noqa: S307 - 呼出側で AST 許可リスト検証済み、builtins も無効
        compile(expr, "<calc>", "eval"), {"__builtins__": {}}, dict(_SAFE_NAMES),
    )


def _reject_expensive(node: ast.AST) -> str | None:
    """指数・階乗の大きさを **子から順に** 検査し、超過なら理由を返す。

    子を先に検査するので、ある Pow の指数部を評価する時点でその中の Pow /
    factorial は全て上限内と分かっている (評価自体が高コストにならない)。
    """
    for child in ast.iter_child_nodes(node):
        reason = _reject_expensive(child)
        if reason:
            return reason
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Add)):
        return _reject_long_sequence(node)
    exponent: object = None
    base: object = None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        exponent = _eval_subtree(node.right)
        base = _eval_subtree(node.left)
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == "pow" and len(node.args) >= 2:
            exponent = _eval_subtree(node.args[1])
            base = _eval_subtree(node.args[0])
        elif node.func.id == _RANGE_FUNC:
            return _reject_long_range(node)
        elif node.func.id == "factorial" and node.args:
            n = _eval_subtree(node.args[0])
            if isinstance(n, (int, float)) and n > _MAX_FACTORIAL_ARG:
                return (
                    f"Error: factorial argument too large ({n} > "
                    f"{_MAX_FACTORIAL_ARG})"
                )
            return None
    if exponent is None:
        return None
    if isinstance(exponent, (int, float)) and abs(exponent) > _MAX_POW_EXPONENT:
        return f"Error: exponent too large (|{exponent}| > {_MAX_POW_EXPONENT})"
    if (
        isinstance(base, int) and isinstance(exponent, int) and exponent > 0
        and abs(base).bit_length() * exponent > _MAX_INT_POW_RESULT_BITS
    ):
        return "Error: result too large to compute"
    return None


def _reject_long_range(node: ast.Call) -> str | None:
    """``range`` の刻みが 0 か長さが上限を超えれば理由を返す (引数は検査済み)。

    引数の評価が失敗する式 (``range(1.5)`` など) はここでは判定せず、実行時の
    エラーに任せる。``len(range(...))`` は要素を作らないので長さの計算は安い。
    """
    try:
        args = [_eval_subtree(arg) for arg in node.args]
    except Exception:  # noqa: BLE001 - 実行時に同じ例外が理由の文字列になる
        return None
    if len(args) == 3 and args[2] == 0:
        return "Error: range step must not be zero"
    try:
        length = len(range(*args))  # type: ignore[arg-type]
    except OverflowError:
        return "Error: result too large to compute"
    except Exception:  # noqa: BLE001 - 実行時に同じ例外が理由の文字列になる
        return None
    if length > _MAX_RANGE_LEN:
        return f"Error: range too long ({length} > {_MAX_RANGE_LEN} elements)"
    return None


#: 文字列を返しうる許可関数 (列のリテラルは別に見る)。
_STRING_FUNCS = frozenset({"str", "hex", "bin", "oct"}) | STRING_RESULT_OPS


def _may_be_sized(node: ast.AST) -> bool:
    """部分木の値が文字列・バイト列・列になりうるか (構文だけで判定、純粋関数)。

    文字列の定数・列のリテラル・文字列を返す関数を含まない部分木は数にしかならない。
    """
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, (str, bytes)):
            return True
        if isinstance(sub, _SEQUENCE_NODES):
            return True
        if (
            isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
            and sub.func.id in _STRING_FUNCS
        ):
            return True
    return False


def _reject_long_sequence(node: ast.BinOp) -> str | None:
    """文字列・列の繰り返し / 連結の結果が長すぎれば理由を返す (子は検査済み)。

    被演算子の評価が失敗する式 (0 除算など) はここでは判定せず、実行時のエラーに任せる。
    文字列・列になりえない被演算子 (数だけの部分木) は評価しない — ``+`` で連ねた式は
    親ごとに部分木を評価し直すので、``sum(range(100000)) + …`` の連なりで 2 乗の時間が
    掛かった (独立レビュー 2026-10-05: 100 個で 10.8 秒)。
    """
    if isinstance(node.op, ast.Mult):
        if not (_may_be_sized(node.left) or _may_be_sized(node.right)):
            return None
    elif not (_may_be_sized(node.left) and _may_be_sized(node.right)):
        return None
    try:
        left = _eval_subtree(node.left)
        right = _eval_subtree(node.right)
    except Exception:  # noqa: BLE001 - 実行時に同じ例外が理由の文字列になる
        return None
    if isinstance(node.op, ast.Mult):
        for seq, count in ((left, right), (right, left)):
            if (
                isinstance(seq, _SIZED_TYPES) and isinstance(count, int)
                and len(seq) * count > _MAX_SEQUENCE_RESULT_LEN
            ):
                return "Error: result too large to compute"
        return None
    if (
        isinstance(left, _SIZED_TYPES) and isinstance(right, _SIZED_TYPES)
        and len(left) + len(right) > _MAX_SEQUENCE_RESULT_LEN
    ):
        return "Error: result too large to compute"
    return None


def _format_calc_result(result: object) -> str:
    """計算結果を人間が読める形に整形する (純粋関数)。

    二進浮動小数の丸め誤差がそのまま UI とモデル文脈へ流れていた
    (実測 2026-07-25: ``38 - (11 + 7 + 2.4 + 1.8 + 2.2)`` が
    ``13.600000000000001``、割合の合算が ``64.21052631578947``)。有効数字 12 桁で
    丸めたうえで末尾ゼロを畳み、整数値の float は整数表記にする。12 桁は
    float64 の有効桁 (約 15〜17 桁) より十分内側で、誤差だけを落とし
    意味のある桁は保つ。int は Python の任意精度をそのまま活かすため素通し。
    """
    if isinstance(result, bool) or not isinstance(result, float):
        return str(result)
    if result != result or result in (float("inf"), float("-inf")):
        return str(result)
    rounded = float(f"{result:.12g}")
    if rounded.is_integer():
        return str(int(rounded))
    return f"{rounded:.12g}"


class _MathPrefixRemover(ast.NodeTransformer):
    """``math.<名前>`` を、名前が許可リストにあるときだけ素の ``<名前>`` に直す。

    それ以外の属性参照 (``math.inf`` / ``(1).__class__``) はそのまま残り、許可リスト
    の検査で ``Attribute`` として拒まれる。
    """

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:  # noqa: N802 - ast の命名
        self.generic_visit(node)
        if (
            isinstance(node.value, ast.Name) and node.value.id == _MATH_MODULE
            and node.attr in _SAFE_NAMES and isinstance(node.ctx, ast.Load)
        ):
            return ast.copy_location(ast.Name(id=node.attr, ctx=ast.Load()), node)
        return node


def _is_math_import(stmt: ast.stmt) -> bool:
    """``import math`` / ``from math import <許可名>`` か (別名・``*`` は数えない)。"""
    if isinstance(stmt, ast.Import):
        return all(a.name == _MATH_MODULE and a.asname is None for a in stmt.names)
    if isinstance(stmt, ast.ImportFrom):
        return (
            stmt.module == _MATH_MODULE and stmt.level == 0
            and all(a.name in _SAFE_NAMES and a.asname is None for a in stmt.names)
        )
    return False


class _StatementError(SyntaxError):
    """式でない文を含む入力 (先頭の ``import math`` を除く)。"""

    def __init__(self, node_name: str) -> None:
        super().__init__(node_name)
        self.node_name = node_name


#: 式の長さの上限。検査は部分木の評価を伴うので、長い式ほど検査そのものが重くなる
#: (独立レビュー 2026-10-05)。会話の数を並べた式でも数百字に収まる。
_MAX_EXPRESSION_CHARS = 400


def _parse_expression(expression: str) -> ast.Expression:
    """式を構文木にする。先頭の ``import math`` を落とし ``math.`` 接頭辞を外す。

    構造の正規化だけで、許可リストは広げない (``math.<許可名>`` は ``<許可名>``
    と同じ関数・定数)。式 1 つとして読めなければ、文として読んで「先頭の
    ``import math`` / ``from math import …`` + 最後の式 1 つ」の形なら最後の式を
    採る。それ以外の文を含む入力は :class:`_StatementError`、文としても読めなければ
    元の ``SyntaxError`` を送出する。:data:`_MAX_EXPRESSION_CHARS` を超える式も
    ``SyntaxError`` (文法の欠け) として拒む。
    """
    if len(expression) > _MAX_EXPRESSION_CHARS:
        raise SyntaxError(
            f"expression too long ({len(expression)} > {_MAX_EXPRESSION_CHARS} "
            "characters) -- write a shorter expression, e.g. sum(range(1, 101, 2)) "
            "instead of listing every term",
        )
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as eval_error:
        try:
            module = ast.parse(expression, mode="exec")
        except SyntaxError:
            raise eval_error from None
        body = module.body
        last = body[-1] if body else None
        if not isinstance(last, ast.Expr):
            raise _StatementError(type(last).__name__ if last else "Module") from None
        for stmt in body[:-1]:
            if not _is_math_import(stmt):
                raise _StatementError(type(stmt).__name__) from None
        tree = ast.Expression(body=last.value)
    tree = _MathPrefixRemover().visit(tree)
    return ast.fix_missing_locations(tree)


def _statement_message(node_name: str) -> str:
    """式でない文を含む入力の理由 (ヒント付き)。"""
    hint = _DISALLOWED_NODE_HINTS.get(node_name, _STATEMENT_HINT)
    return f"Error: Unsafe expression (disallowed node: {node_name}) -- {hint}"


def _check_allowlist(tree: ast.AST) -> str | None:
    """許可リスト (ノード・名前・呼び出し) を検査し、違反なら理由を返す (コストは見ない)。"""
    # ast.walk は親を子より先に返すので、列を受け取る呼び出しを見た時点でその
    # 直接の引数の列 (と range の呼び出し) を許可に積めば、それ自身の検査に間に合う。
    allowed_sequences: set[int] = set()
    allowed_ranges: set[int] = set()
    for node in ast.walk(tree):
        node_name = type(node).__name__
        if isinstance(node, _SEQUENCE_NODES) and id(node) in allowed_sequences:
            continue
        if type(node) not in _SAFE_NODES:
            msg = f"Error: Unsafe expression (disallowed node: {node_name})"
            hint = _DISALLOWED_NODE_HINTS.get(node_name)
            if hint:
                msg += f" -- {hint}"
            return msg
        if isinstance(node, ast.Name) and node.id not in _SAFE_NAMES:
            return (
                f"Error: Unsafe expression (unknown name: {node.id})"
                f" -- {_DISALLOWED_NODE_HINTS['Name']}"
            )
        if (
            isinstance(node, ast.Name) and node.id == _RANGE_FUNC
            and id(node) not in allowed_ranges
        ):
            return (
                "Error: Unsafe expression (range is allowed only as the single "
                "argument of " + "/".join(sorted(_SEQUENCE_FUNCS))
                + ", e.g. sum(range(1, 101, 2)))"
            )
        if isinstance(node, ast.Call):
            # 呼び出し先は許可リストの素の名前のみ (Attribute は _SAFE_NODES に
            # 無いのでここへ来る前に弾かれる)。キーワード引数と *args/**kwargs は
            # 許可しない — 数学関数の用途では不要で、検証面を最小に保つ。
            if not isinstance(node.func, ast.Name):
                return (
                    "Error: Unsafe expression (call target must be a plain "
                    f"function name) -- {_DISALLOWED_NODE_HINTS['Call']}"
                )
            if node.keywords:
                return (
                    "Error: Unsafe expression (keyword arguments are not "
                    "supported)"
                )
            if node.func.id == _RANGE_FUNC and not 1 <= len(node.args) <= 3:
                return "Error: Unsafe expression (range takes 1 to 3 arguments)"
            # 列リテラルと range は唯一の引数のときだけ。2 個以上の引数の列
            # (``max([0], [1])``) は列を返し、その戻り値を繰り返しの被演算子にできる。
            if node.func.id in _SEQUENCE_FUNCS and len(node.args) == 1:
                arg = node.args[0]
                if isinstance(arg, _SEQUENCE_NODES):
                    allowed_sequences.add(id(arg))
                elif (
                    isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name)
                    and arg.func.id == _RANGE_FUNC
                ):
                    allowed_ranges.add(id(arg.func))
    return None


def _check_tree(tree: ast.AST) -> str | None:
    """許可リスト (ノード・名前・呼び出し) とコスト上限を検査し、違反なら理由を返す。"""
    return _check_allowlist(tree) or _reject_expensive(tree)


def validate_expression(expression: str) -> str | None:
    """``calculate`` が受け付ける式か (評価はしない)。受け付けないなら理由を返す。

    式合成 (ツール判定の層 5.95) が合成直後に使う。合成器は SQL を「式」として
    返すことがあり、実行時に初めて ``invalid syntax`` になっていた (2026-09-27
    ライブ監査 C02#2、docs/f_03 §3.1)。規則は ``calculate`` と同じもの 1 つ。
    """
    try:
        tree = _parse_expression(expression)
        return _check_tree(tree)
    except _StatementError as e:
        return _statement_message(e.node_name)
    except Exception as e:  # noqa: BLE001 - calculate と同じく理由の文字列で返す
        return f"Error: {e}"


def _unknown_names_in(tree: ast.AST) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """構文木の許可リスト外の名前を ``(変数として使った名前, 関数として呼んだ名前)`` で返す。

    出現順・重複なし。:func:`expression_gap` が「会話の語を変数名にした式」と
    「計算機に無い関数を呼ぶ式」(``fibonacci(10)``) を分けるのに使う。
    """
    called = {
        id(node.func) for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    variables: list[str] = []
    functions: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in _SAFE_NAMES:
            bucket = functions if id(node) in called else variables
            if node.id not in bucket:
                bucket.append(node.id)
    return tuple(variables), tuple(functions)


class _VariableFiller(ast.NodeTransformer):
    """指定した変数名を数のリテラル 1 に置き換える (欠けた被演算子の仮置き)。"""

    def __init__(self, names: frozenset[str]) -> None:
        self._names = names

    def visit_Name(self, node: ast.Name) -> ast.AST:  # noqa: N802 - ast の命名
        if node.id in self._names:
            return ast.copy_location(ast.Constant(value=1), node)
        return node


def expression_gap(expression: str) -> tuple[str, str] | None:
    """式が calculate の式にならない理由を **種類付き** で返す (純粋関数、docs/f_03 §3.1)。

    返すのは ``(種類, validate_expression の理由)``。種類は 3 つ:

    - ``"operand"`` — 欠けているのは **数** だけ。許可リスト外の変数名 (会話の語を
      変数名にした ``sum(employees_overtime_hours) * 0.9``) を数で仮置きすれば
      許可リストを通る。利用者に被演算子を確かめる意味がある。
    - ``"grammar"`` — 計算機の **文法** に合わない。構文エラー・式でない文
      (``import`` の後に式以外が続く等)・許可外のノード (生成式・属性参照)・キーワード
      引数・長すぎる式。数を尋ねても直らない (2026-10-05 ライブ監査: 構文エラーの式で
      会話の全ての数を尋ね返していた)。
    - ``"function"`` — 計算機に無い関数の呼び出し (``fibonacci(10)`` / ``comb(10, 3)``)。
      数は揃っていて、足りないのは計算機の機能。組み直せなければ知識で答えてよい
      (独立レビュー 2026-10-05)。

    受け付ける式、およびコスト上限だけで落ちる式 (``10**10**10``) は ``None``。
    """
    reason = validate_expression(expression)
    if reason is None:
        return None
    try:
        tree = _parse_expression(expression)
    except Exception:  # noqa: BLE001 - 構文エラー・式でない文はすべて文法の欠け
        return ("grammar", reason)
    variables, functions = _unknown_names_in(tree)
    if functions:
        return ("function", reason)
    if variables:
        filled = _VariableFiller(frozenset(variables)).visit(tree)
        if _check_allowlist(filled) is None:
            return ("operand", reason)
        return ("grammar", reason)
    if _check_allowlist(tree) is not None:
        return ("grammar", reason)
    return None


def calculate(expression: str) -> str:
    """数式を安全に計算する"""
    try:
        tree = _parse_expression(expression)
        violation = _check_tree(tree)
        if violation:
            return violation
        result = eval(  # noqa: S307 - AST を許可リストで検証済み、builtins も無効
            compile(tree, "<calc>", "eval"),
            {"__builtins__": {}}, dict(_SAFE_NAMES),
        )
        return _format_calc_result(result)
    except _StatementError as e:
        return _statement_message(e.node_name)
    except ValueError as e:
        if "int_max_str_digits" in str(e) or "Exceeds the limit" in str(e):
            # Python の文言は ``sys.set_int_max_str_digits()`` を勧めるが、計算機では
            # 使えない。桁数なら対数で数えられる (独立レビュー 2026-10-05)。
            return _TOO_MANY_DIGITS
        return f"Error: {e}"
    except Exception as e:
        return f"Error: {e}"
