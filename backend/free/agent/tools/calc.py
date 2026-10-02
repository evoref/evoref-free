"""``calculate`` ツール — AST 許可リストによる安全な算術式評価"""

from __future__ import annotations

import ast
import math


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
#: ``range`` は上限なしだと ``sum(range(10**12))`` を許すので引き続き載せない。
_SAFE_NAMES: dict[str, object] = {
    "abs": abs, "round": round, "min": min, "max": max,
    "sum": sum, "len": len,
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
}

#: リスト・タプルのリテラルを受け取れる関数。列のリテラルは、ここに載る関数の
#: **唯一の** 引数であるときだけ許す (2 個以上の引数の列・入れ子の列・演算の被演算子・
#: 他の関数の引数は拒む)。
_SEQUENCE_FUNCS = frozenset({"sum", "min", "max", "len"})
_SEQUENCE_NODES = (ast.List, ast.Tuple)

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
}


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


def _reject_long_sequence(node: ast.BinOp) -> str | None:
    """文字列・列の繰り返し / 連結の結果が長すぎれば理由を返す (子は検査済み)。

    被演算子の評価が失敗する式 (0 除算など) はここでは判定せず、実行時のエラーに任せる。
    """
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


def _check_tree(tree: ast.AST) -> str | None:
    """許可リスト (ノード・名前・呼び出し) とコスト上限を検査し、違反なら理由を返す。"""
    # ast.walk は親を子より先に返すので、列を受け取る呼び出しを見た時点でその
    # 直接の引数の列を許可に積めば、列そのものの検査に間に合う。
    allowed_sequences: set[int] = set()
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
            # 列リテラルは唯一の引数のときだけ。2 個以上の引数の列 (``max([0], [1])``)
            # は列を返し、その戻り値を繰り返しの被演算子にできる。
            if (
                node.func.id in _SEQUENCE_FUNCS and len(node.args) == 1
                and isinstance(node.args[0], _SEQUENCE_NODES)
            ):
                allowed_sequences.add(id(node.args[0]))
    return _reject_expensive(tree)


def validate_expression(expression: str) -> str | None:
    """``calculate`` が受け付ける式か (評価はしない)。受け付けないなら理由を返す。

    式合成 (ツール判定の層 5.95) が合成直後に使う。合成器は SQL を「式」として
    返すことがあり、実行時に初めて ``invalid syntax`` になっていた (2026-09-27
    ライブ監査 C02#2、docs/f_03 §3.1)。規則は ``calculate`` と同じもの 1 つ。
    """
    try:
        tree = ast.parse(expression, mode="eval")
        return _check_tree(tree)
    except Exception as e:  # noqa: BLE001 - calculate と同じく理由の文字列で返す
        return f"Error: {e}"


def unknown_names(expression: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """式の許可リスト外の名前を ``(変数として使った名前, 関数として呼んだ名前)`` で返す。

    出現順・重複なし。構文エラーなら ``SyntaxError`` を送出する。ツール判定が
    「会話の語を変数名にした式」(組み直せる) と「計算機に無い関数を呼ぶ式」
    (``fibonacci(10)`` / ``timedelta(days=100)``、組み直しても数を尋ねる理由が無い) を
    分けるのに使う (docs/f_03 §3.1)。
    """
    tree = ast.parse(expression, mode="eval")
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


def calculate(expression: str) -> str:
    """数式を安全に計算する"""
    try:
        tree = ast.parse(expression, mode="eval")
        violation = _check_tree(tree)
        if violation:
            return violation
        result = eval(  # noqa: S307 - AST を許可リストで検証済み、builtins も無効
            compile(tree, "<calc>", "eval"),
            {"__builtins__": {}}, dict(_SAFE_NAMES),
        )
        return _format_calc_result(result)
    except Exception as e:
        return f"Error: {e}"
