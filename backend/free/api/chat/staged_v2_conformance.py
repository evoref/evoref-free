"""staged v2 の骨組みのシグネチャを宣言契約にする (f_10 §11.1-3、観測だけ)。

v1 の spec 準拠ゲート (``generation/spec_conformance.check_spec_conformance``) は spec 節から抽出した宣言契約
(``loop/staged/spec_contract.py`` の plain dict) を入力にする。v2 には spec 節が無いので、骨組みのモジュールの
``components`` の signature から同じ形の dict を組む。api 層に置くのは ``generation/`` が ``loop/`` を import
しないため (loop → api の import は許される)。

読む形 (Python のモジュールだけ。読めない signature は飛ばす — parse-or-skip):

- ``class C`` / ``class C(Base)`` → クラス ``C`` (以降の self メソッドの帰属先)
- ``def C.m(self, x)`` / ``C.m(self, x)`` → クラス ``C`` のメソッド ``m``
- ``C(x)`` (骨組みが先に ``class C`` を挙げたとき) → ``C.__init__``
- ``def f(x)`` / ``f(x)`` → 関数 ``f``。第 1 引数が self / cls で、直前にクラスがあればそのメソッド (無ければ飛ばす)
- ``def f(...`` のように引数が読めない def → 存在だけ (``name_only``)

各 dict には照合先のモジュール (``path``) を足す (``check_spec_conformance`` は知らないキーを読まない)。
"""

from __future__ import annotations

import ast

from backend.free.loop.staged.spec_contract import sig_dict


def _parse_def(name: str, rest: str) -> dict | None:
    """``def {name}{rest}: ...`` を読んで SIG を返す (読めなければ None)。"""
    try:
        tree = ast.parse(f"def {name}{rest.rstrip().rstrip(':')}: ...")
    except SyntaxError:
        return None
    node = tree.body[0] if tree.body else None
    return sig_dict(node) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else None


def skeleton_declared_contract(modules: list[dict]) -> list[dict]:
    """骨組みのモジュールの ``components`` から宣言契約 (``spec_contract`` と同じ形 + ``path``) を組む。"""
    out: list[dict] = []
    for m in modules:
        path = str(m.get("path") or "")
        if not path.endswith(".py"):
            continue
        classes: dict[str, dict] = {}
        current: str | None = None

        def _class(name: str) -> dict:
            if name not in classes:
                classes[name] = {"kind": "class", "name": name, "methods": {}, "path": path}
                out.append(classes[name])
            return classes[name]

        for c in m.get("components") or []:
            text = str(c.get("signature") or "").strip().strip("`").strip()
            if text.startswith("class "):
                head = text[len("class "):].split("(", 1)[0].split(":", 1)[0].strip()
                if head.isidentifier():
                    _class(head)
                    current = head
                continue
            body = text.removeprefix("async ").strip()
            has_def = body.startswith("def ")
            body = body.removeprefix("def ").strip()
            head, paren, rest = body.partition("(")
            head = head.strip()
            owner, _, name = head.rpartition(".")
            if not name.isidentifier() or (owner and not owner.isidentifier()):
                continue
            sig = _parse_def(name, paren + rest) if paren else None
            if sig is None:
                if has_def and not owner:
                    out.append({"kind": "name_only", "name": name, "path": path})
                continue
            if owner:
                _class(owner)["methods"].setdefault(name, sig)
            elif not has_def and name in classes:
                # ``C(x)`` はコンストラクタ (``spec_contract`` の裸シグネチャと同じ読み)
                classes[name]["methods"].setdefault("__init__", sig)
            elif sig.get("self_like"):
                if current is not None:
                    classes[current]["methods"].setdefault(name, sig)
            else:
                out.append({"kind": "function", "name": name, "sig": sig, "path": path})
    return out


__all__ = ["skeleton_declared_contract"]
