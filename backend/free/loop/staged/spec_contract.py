"""spec モジュール節から「宣言 API 契約」を決定論抽出する (parse-or-skip)。

準拠ゲート (gen 側 ``spec_conformance.check_spec_conformance``) の入力となる
宣言契約を、spec の ``### Component:`` 節から抽出する。原則は **parse-or-skip**:
``ast`` でパースできた宣言だけを契約とし、崩れた Signature は presence-only へ
段階降格するか沈黙スキップする (fuzzy 補完はしない — 誤検知は破壊的リペアを
招くため、パース失敗は常に「検出力低下」の側へ倒す。f_10 §5)。

執行アンカーは「パース成功した Signature の識別子」のみで、``### Component:``
見出し名そのものは照合に使わない (prose 名「helper functions」/ グループ
component / エントリ直書き実装の誤検知源を構造的に排除)。見出し名は
「クラス名の推定」(識別子 + self メソッド群が根拠) にのみ使う。

契約は plain dict の列で表現し、gen 側へ型 import なしで渡す::

    {"kind": "class",     "name": str, "methods": {name: SIG | None}}
    {"kind": "function",  "name": str, "sig": SIG | None}
    {"kind": "name_only", "name": str}
    SIG = {"min_pos", "max_pos",            # self/cls を除いた解釈
           "min_pos_alt", "max_pos_alt",    # 除かない解釈 (両解釈照合用)
           "has_varargs", "has_kwargs", "self_like"}
"""

from __future__ import annotations

import ast
import re

from backend.free.loop.staged.spec_parts import iter_component_sections

# Signature バレットのラベル行。_spec_language_constraint がバレットラベルを
# 英語構文のまま固定するため英語のみ対応 (逸脱 spec は fallback 走査へ)。
_SIG_LABEL_RE = re.compile(r"(?im)^\s*[-*]\s*Signatures?\s*[:：]")
# Signature 領域の終端 = 次の「任意の」バレットラベル or 見出し。既知ラベルの
# 閉集合にすると LLM 逸脱ラベル (Raises/Returns 等) の内容が領域に紛れ込み、
# その中の呼出表記が偽契約になる (レビュー確定指摘)。ラベルは英字始まりを
# 要求するため、backtick 始まりのサブバレット (`- `rotate(self)``) は終端に
# ならない。
_REGION_END_RE = re.compile(
    r"(?im)^\s*[-*]\s*[A-Za-z][A-Za-z0-9_ /]{0,40}\s*[:：]"
    r"|^#{2,3}\s",
)
_FENCE_RE = re.compile(r"```[a-zA-Z]*[ \t]*\n(.*?)```", re.DOTALL)
_BACKTICK_RE = re.compile(r"`([^`\n]+)`")
# def / class アンカーは候補テキストの「先頭」でのみ認める。prose 中の
# 『the Board class described above』等の言及をクラス契約に化けさせない
# (レビュー確定指摘)。
_DEF_ANCHOR_RE = re.compile(r"^(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(")
_CLASS_ANCHOR_RE = re.compile(r"^class\s+([A-Za-z_]\w*)")
# 裸シグネチャ (`__init__(self, x: int)` / `main()`)。Signature 領域の
# backtick/フェンス候補限定で許容し、名前と '(' の間の空白は認めない
# (`Coordinates (row, col)` のような prose を宣言にしない)。
_BARE_SIG_RE = re.compile(r"^([A-Za-z_]\w*)\(")

# 裸シグネチャとして契約化しない名前: 組込み (prose 中の `input()` 等の呼出
# 言及) と例外クラス表記 (`KeyError(key)` — raise 用法の言及)。明示的な
# def/class アンカー付き宣言はこのフィルタの対象外 (意図的なシャドーイング
# 宣言は通す)。
_BARE_NAME_DENY = frozenset(dir(__import__("builtins")))
_EXC_SUFFIX_RE = re.compile(r"(?:Error|Exception|Warning)$")


def _slice_paren_group(text: str, open_idx: int) -> str | None:
    """``text[open_idx] == '('`` から括弧バランスで引数リスト終端までを返す。

    領域内で閉じなければ ``None`` (日本語 prose 後続などの崩れ候補を破棄)。
    """
    depth = 0
    for i in range(open_idx, len(text)):
        c = text[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                return text[open_idx:i + 1]
    return None


def _sig_dict(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> dict:
    """def ノードから両解釈 (self 除外/込み) の位置引数レンジを構築する。"""
    a = fn.args
    all_pos = list(a.posonlyargs) + list(a.args)
    total = len(all_pos)
    min_pos = max(0, total - len(a.defaults))
    self_like = bool(all_pos) and all_pos[0].arg in ("self", "cls")
    strip = 1 if self_like else 0
    return {
        "min_pos": max(0, min_pos - strip),
        "max_pos": max(0, total - strip),
        "min_pos_alt": min_pos,
        "max_pos_alt": total,
        "has_varargs": a.vararg is not None,
        "has_kwargs": a.kwarg is not None,
        "self_like": self_like,
    }


def _parse_def_text(name: str, args_group: str) -> dict | None:
    """``def {name}{args_group}: ...`` を ast でパースし SIG を返す (失敗 None)。

    戻り値注釈は arity に寄与しないため取り込まない (崩れ源を減らす)。
    """
    try:
        tree = ast.parse(f"def {name}{args_group}: ...")
    except SyntaxError:
        return None
    node = tree.body[0]
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    return _sig_dict(node)


def _parse_candidate(
    candidate: str, *, allow_bare: bool,
) -> tuple[str, str, dict | None] | None:
    """1 候補テキストを (kind, name, sig) に解釈する (不成立 None)。

    kind: ``"class" | "def" | "bare" | "name_only"``。``name_only`` は
    def アンカーはあるが引数リストがパース不能だった場合の presence 専用降格。
    アンカーは候補先頭のみ・裸シグネチャは組込み/例外名を除外 (prose 中の
    呼出言及を宣言に化けさせない)。
    """
    text = candidate.strip().lstrip("-*").strip().strip("`").strip()
    if not text:
        return None
    cm = _CLASS_ANCHOR_RE.match(text)
    if cm is not None:
        return ("class", cm.group(1), None)
    dm = _DEF_ANCHOR_RE.match(text)
    if dm is not None:
        name = dm.group(1)
        open_idx = text.index("(", dm.start())
        args_group = _slice_paren_group(text, open_idx)
        if args_group is not None:
            sig = _parse_def_text(name, args_group)
            if sig is not None:
                return ("def", name, sig)
        # 引数リスト崩れ → presence-only へ 1 段だけ降格 (これ以上は救済しない)
        return ("name_only", name, None)
    if allow_bare:
        bm = _BARE_SIG_RE.match(text)
        if bm is not None:
            name = bm.group(1)
            if name in _BARE_NAME_DENY or _EXC_SUFFIX_RE.search(name):
                return None
            args_group = _slice_paren_group(text, text.index("(", bm.start()))
            if args_group is not None:
                sig = _parse_def_text(name, args_group)
                if sig is not None:
                    return ("bare", name, sig)
        return None
    return None


def _bullet_region(body: str, label_re: re.Pattern[str]) -> str | None:
    """component 本文から指定バレットの領域を切り出す (無ければ None)。

    終端は次の「任意の」バレットラベル or 見出し (:data:`_REGION_END_RE` —
    既知ラベルの閉集合にしない理由は同定数のコメント参照)。
    """
    m = label_re.search(body)
    if m is None:
        return None
    rest = body[m.end():]
    end = _REGION_END_RE.search(rest)
    return rest[: end.start()] if end else rest


def _signature_region(body: str) -> str | None:
    """component 本文から Signature バレットの領域を切り出す (無ければ None)。"""
    return _bullet_region(body, _SIG_LABEL_RE)


# Behavior バレットの番号ステップ抽出 (決定論フォールバックフローの入力)。
_BEHAVIOR_LABEL_RE = re.compile(r"(?im)^\s*[-*]\s*Behaviou?rs?\s*[:：]")
_NUMBERED_LINE_RE = re.compile(r"(?m)^\s*\d+[.)]\s+(\S.*)$")


def extract_behavior_steps(section: str) -> list[str]:
    """モジュール節から Behavior バレットの番号ステップを component 順に抽出する。

    決定論フォールバックフロー (``flow_render.fallback_flow``) の入力。LLM の
    FlowSpec 合成が 2 回とも検証不能だった場合の縮退品質を、既に生成済みの
    spec 本文 (Behavior 番号ステップ) から追加 LLM 呼出なしで底上げする
    (2026-07-07 live: 縮退フローが「モジュール概要を 1 行に圧縮した 3 step」
    まで粗くなっていた)。継続行(折返し)は拾わず単一行のみ対象 (parse-or-skip
    — 拾えなくても粗いだけで誤りにはならない)。
    """
    out: list[str] = []
    for _heading, body in iter_component_sections(section):
        region = _bullet_region(body, _BEHAVIOR_LABEL_RE)
        if region is None:
            continue
        out.extend(m.group(1).strip() for m in _NUMBERED_LINE_RE.finditer(region))
    return out


def _candidates(
    text: str, *, fenced_and_backtick_only: bool,
) -> list[tuple[str, str]]:
    """(候補テキスト, 出所) 列を出現順で返す。出所: fence | backtick | raw。

    裸シグネチャの解釈はコード扱いの出所 (fence/backtick) に限定するため、
    出所タグを付けて返す。順序はクラス→メソッドの帰属 (直前のクラス候補に
    属する) に使うため、フェンス/バッククォート/生行を原文の出現位置順で
    混在させる。
    """
    events: list[tuple[int, str, str]] = []
    fence_spans: list[tuple[int, int]] = []
    for fm in _FENCE_RE.finditer(text):
        fence_spans.append((fm.start(), fm.end()))
        offset = fm.start(1)
        pos = 0
        for ln in fm.group(1).splitlines(keepends=True):
            if ln.strip():
                events.append((offset + pos, ln.rstrip("\n"), "fence"))
            pos += len(ln)

    def _in_fence(i: int) -> bool:
        return any(s <= i < e for s, e in fence_spans)

    for m in _BACKTICK_RE.finditer(text):
        if not _in_fence(m.start()):
            events.append((m.start(), m.group(1), "backtick"))
    if not fenced_and_backtick_only:
        no_tick = _BACKTICK_RE.sub(lambda m: " " * len(m.group(0)), text)
        pos = 0
        for ln in no_tick.splitlines(keepends=True):
            if ln.strip() and not _in_fence(pos):
                events.append((pos, ln.rstrip("\n"), "raw"))
            pos += len(ln)
    events.sort(key=lambda e: e[0])
    return [(t, src) for _, t, src in events]


def parse_declared_contract(section: str) -> list[dict]:
    """モジュール節から宣言契約 (plain dict 列) を抽出する。

    - Signature バレット領域があればそこを主走査 (裸シグネチャは
      backtick/フェンス出所の候補のみ許容 — 生行 prose の呼出表記を宣言に
      しない)。無ければ component 本文全体からフェンス/backtick 内の明示的な
      def/class アンカーのみを保守的に走査する。
    - 候補は**出現順**に解釈し、def/bare は直前の class 候補へ帰属させる
      (1 component に小さい関連クラスが複数宣言される形式に対応)。
    - 見出しのクラス名推定は「識別子 + 先頭大文字 + self メソッド証拠」の
      三重根拠がある場合のみ (`utils` 等の関数グループ見出しを誤ってクラス化
      しない)。
    - 本文に ``__main__`` を含む component はエントリ扱いでスキップ
      (``if __name__`` 直書き実装が正当なため)。
    - 何もパースできなかった component は契約ゼロ = 検査対象外。
    """
    decls: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def _add(d: dict) -> None:
        key = (d["kind"], d["name"])
        if key not in seen:
            seen.add(key)
            decls.append(d)

    for heading, comp_body in iter_component_sections(section):
        if "__main__" in comp_body:
            continue
        region = _signature_region(comp_body)
        if region is not None:
            candidates = _candidates(region, fenced_and_backtick_only=False)
            bare_sources = frozenset({"fence", "backtick"})
        else:
            candidates = _candidates(comp_body, fenced_and_backtick_only=True)
            bare_sources = frozenset()

        classes: dict[str, dict[str, dict | None]] = {}
        class_order: list[str] = []
        pool: list[tuple[str, str, dict]] = []  # class 出現前の (kind, name, sig)
        name_onlys: list[str] = []
        current: str | None = None
        for cand, source in candidates:
            r = _parse_candidate(cand, allow_bare=source in bare_sources)
            if r is None:
                continue
            kind, name, sig = r
            if kind == "class":
                if name not in classes:
                    classes[name] = {}
                    class_order.append(name)
                current = name
            elif kind == "name_only":
                name_onlys.append(name)
            elif current is not None:
                # 直前のクラスへ帰属。裸候補は「コンストラクタ (クラス同名)」
                # か「self メソッド」の確証がある時のみ (それ以外の裸呼出
                # 表記は用法言及の可能性が高いためスキップ)。
                if kind == "bare" and name != current and not sig.get("self_like"):
                    continue
                key = "__init__" if (kind == "bare" and name == current) else name
                classes[current].setdefault(key, sig)
            else:
                pool.append((kind, name, sig))

        if not class_order and pool:
            # 明示 class 行は無いが self/cls メソッド群がある → 見出しを
            # クラス名と推定 (識別子 + 先頭大文字 + メソッド証拠の三重根拠)。
            if (
                heading.isidentifier()
                and heading[:1].isupper()
                and any(s.get("self_like") for _k, _n, s in pool)
            ):
                classes[heading] = {}
                class_order.append(heading)
                for kind, name, sig in pool:
                    if (
                        kind == "bare" and name != heading
                        and not sig.get("self_like")
                    ):
                        continue
                    key = (
                        "__init__" if (kind == "bare" and name == heading)
                        else name
                    )
                    classes[heading].setdefault(key, sig)
                pool = []

        for cname in class_order:
            _add({"kind": "class", "name": cname, "methods": classes[cname]})
        for _kind, name, sig in pool:
            if sig.get("self_like"):
                # クラス名を推定できない self メソッドは帰属先不明のため
                # スキップ (parse-or-skip)。
                continue
            _add({"kind": "function", "name": name, "sig": sig})
        for n in name_onlys:
            _add({"kind": "name_only", "name": n})
    return decls


def declared_definition_names(contract: list[dict]) -> list[str]:
    """契約中の def/class 名 (メソッド名を除く) を返す (per-part 存在チェック用)。"""
    return [d["name"] for d in contract if d.get("name")]


_SIG_COMPARE_KEYS = (
    "min_pos", "max_pos", "min_pos_alt", "max_pos_alt",
    "has_varargs", "has_kwargs",
)


def signatures_differ(a: object, b: object) -> bool:
    """2 つの SIG dict の位置引数契約が異なるか (片方不明なら False = 同一扱い)。

    深化ガードの契約保存判定用。不明 (presence-only) はドリフト検出不能な
    ため保守的に同一へ倒す。
    """
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    return any(a.get(k) != b.get(k) for k in _SIG_COMPARE_KEYS)


def contract_drift_reason(original: str, deepened: str) -> str:
    """深化前後の宣言契約を比較し、契約が変質していれば理由を返す ("" = 保存)。

    substring ベースの Signature スニペット保存だけでは「原文を残したまま
    別シグネチャを追記して契約を乗っ取る」経路 (フェンス追加・setdefault の
    先勝ち) を検出できないため、**契約そのもの** (parse_declared_contract の
    出力) の保存 (原契約 ⊆ 深化後契約、シグネチャ同一) を要求する。
    詳細の追加 (新メソッド・新宣言) は許容する。
    """
    orig = parse_declared_contract(original)
    if not orig:
        return ""
    new_by_key = {
        (d["kind"], d["name"]): d for d in parse_declared_contract(deepened)
    }
    for d in orig:
        nd = new_by_key.get((d["kind"], d["name"]))
        if nd is None:
            return f"lost {d['kind']} {d['name']}"
        if d["kind"] == "function" and signatures_differ(
            d.get("sig"), nd.get("sig"),
        ):
            return f"sig changed: {d['name']}"
        if d["kind"] == "class":
            new_methods = nd.get("methods") or {}
            for mname, msig in (d.get("methods") or {}).items():
                if mname not in new_methods:
                    return f"lost method {d['name']}.{mname}"
                if signatures_differ(msig, new_methods[mname]):
                    return f"sig changed: {d['name']}.{mname}"
    return ""


def signature_snippets(section: str) -> list[str]:
    """Signature 領域内の backtick/フェンス内容を空白正規化して返す。

    深化ガード (宣言シグネチャの無断変更検出) 用: 深化後の節の正規化全文に
    元スニペットが substring として全て残っていることを要求する。
    Signature 領域外の backtick (prose 内の型名等) は対象にしない
    (深化は prose の書き直しを許すため)。
    """
    out: list[str] = []
    for _heading, comp_body in iter_component_sections(section):
        region = _signature_region(comp_body)
        if region is None:
            continue
        for fm in _FENCE_RE.finditer(region):
            for ln in fm.group(1).splitlines():
                s = " ".join(ln.split())
                if s:
                    out.append(s)
        rest = _FENCE_RE.sub(" ", region)
        for m in _BACKTICK_RE.finditer(rest):
            s = " ".join(m.group(1).split())
            if s:
                out.append(s)
    return out


def normalized_text(text: str) -> str:
    """空白 collapse の正規化 (signature_snippets の照合先を作る)。"""
    return " ".join(text.split())
