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
#: ``sum`` は **載せない**。引数に渡せる列 (``range`` / リストリテラル) が
#: どちらも作れないので構造的に到達不能で、それでいて名前だけは許可リストと
#: エラーメッセージ (``_DISALLOWED_NODE_HINTS``) に出るため、モデルを必ず
#: 失敗する式へ誘導していた。実測 (2026-08-27 ライブ監査):
#:
#:     sum(range(1,101))  -> Error: Unsafe expression (unknown name: range)
#:     sum([1, 2, 3])     -> Error: Unsafe expression (disallowed node: List)
#:
#: 「1から100までの整数の和」で分類器が ``sum(range(1,101))`` を組み立てて
#: 失敗し (agent_trace の reward=0.0)、暗算フォールバックでたまたま正答した
#: ためユーザーからは見えなかった。``range`` を足す案は上限なしだと
#: ``sum(range(10**12))`` を許すことになり、AST の許可面も広がるので採らない
#: (数個の加算は ``1+2+3`` で足りる)。
_SAFE_NAMES: dict[str, object] = {
    "abs": abs, "round": round, "min": min, "max": max,
    "pow": pow,
    "sqrt": math.sqrt, "exp": math.exp,
    "log": math.log, "log2": math.log2, "log10": math.log10,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "floor": math.floor, "ceil": math.ceil,
    "factorial": math.factorial, "gcd": math.gcd, "hypot": math.hypot,
    "degrees": math.degrees, "radians": math.radians,
    "pi": math.pi, "e": math.e, "tau": math.tau,
}

# 非許可ノードごとの自己修正ヒント。LLM が同一ターン内でエラーを見て
# 書き直せるよう、そのノードが生じがちな典型的な誤記法を指す (実インシデント:
# 「πr²」を "π*5^2" と書いて BitXor に、「GCD」を "gcd(360,504)" と書いて
# Call になり、いずれもエラー後は手計算にフォールバックしていた)。
_DISALLOWED_NODE_HINTS: dict[str, str] = {
    "BitXor": "use ** for exponentiation, not ^ (^ is bitwise XOR here)",
    # 一覧は _SAFE_NAMES から生成する (直書きすると許可リストとドリフトする)
    "Call": "only these names are available: " + " ".join(sorted(_SAFE_NAMES)),
    "Name": "only these names are available: " + " ".join(sorted(_SAFE_NAMES)),
}


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


def calculate(expression: str) -> str:
    """数式を安全に計算する"""
    try:
        tree = ast.parse(expression, mode="eval")
        for node in ast.walk(tree):
            node_name = type(node).__name__
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
        result = eval(  # noqa: S307 - AST を許可リストで検証済み、builtins も無効
            compile(tree, "<calc>", "eval"),
            {"__builtins__": {}}, dict(_SAFE_NAMES),
        )
        return _format_calc_result(result)
    except Exception as e:
        return f"Error: {e}"
