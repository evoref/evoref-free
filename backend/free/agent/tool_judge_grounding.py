"""合成式の数値グラウンディング (純粋関数)

「式に現れる数値が対話から辿れるか」だけを扱う。ツールの戻り値は確かめた事実
として base に最優先で渡るため、捏造された定数が混ざった式は *正しく計算された
嘘* になる。その検出をここに集約する。
"""

from __future__ import annotations

import ast
import math
import re

from backend.free.core.intent_vocab import NUMBER_LITERAL_RE
from backend.free.core.numerals import kanji_number_value
from backend.free.core.response_arithmetic import JaNumber, iter_ja_numbers

#: 数値計算クエリの事前フィルタ。式が書かれていれば層1 (_extract_arithmetic_expression)
#: が決定論的に処理するため、ここは「数値は複数あるが式は書かれていない」ものだけを
#: 対象にする。aux 往復 (realtime) を増やさないよう条件は厳しめにする。
#: 実体は ``core.intent_vocab`` が SSOT (``agent.router`` も同じ判定を使う)。
#: 既存の呼出元とテストのために旧名を残す。
_NUMBER_LITERAL_RE = NUMBER_LITERAL_RE


#: 単位系の定義そのものに由来する定数。ユーザーが書いた数値ではないが、
#: 「モデルが知識から思い出した換算率」でもない (分/時、時/日、SI 接頭辞、
#: パーセント)。定義上一意なので誤記憶しようがなく、捏造検出の対象外にする。
#: マイル→キロ (1.609) のような **知識** の換算率はここに入れない
#: (実インシデント 2026-07-29 ライブ監査: 「時速72kmで45分間に進む距離は
#: 何kmですか？」で 45/60 の 60 がグラウンディングに落ち、base の暗算で
#: 90km と誤答した。正解 54km)。
#:
#: 2 進接頭辞 (1024 の冪) も SI 接頭辞と同じ **定義** であり、記憶違いの余地が
#: ない。バイト→KiB/MiB/GiB の換算で式に必ず現れるため、外すとサイズ計算が
#: 丸ごと no_tool へ落ちる (実インシデント 2026-08-09 ライブ監査: 直前ターンの
#: 1,277,500 行 × 240 バイトに対しネイティブ層が
#: ``(1277500 * 240) / (1024 * 1024)`` を正しく合成したのに、``1024`` が
#: グラウンディングに落ちて棄却され、base の暗算で「約288MB」と誤答した。
#: 正解は 306,600,000 バイト = 292.4 MiB)。
#: 暦の周期 (週/年、うるう年の日数、秒/日) も 365 や 1440 と同じ**暦の定義**で、
#: 知識として思い出す換算率ではない。52 が無いために「週に3冊なら年間何冊か」で
#: ネイティブ層が正しく合成した ``3 * 52`` が ungrounded で棄却され、base の
#: 暗算に落ちていた (実インシデント 2026-08-10 ライブ監査)。
#: ``1`` は乗法の単位元で、``(1 + r)`` / ``1 - x`` のように式の構造そのものが
#: 要求する値。知識由来の定数ではない。無いと「1 が対話に書かれているか」が
#: クエリにたまたま ``1,000`` があるかで揺れる (実インシデント 2026-09-05
#: ライブ監査 T03/1: 年金現価式の ``(1 - (1 + r)**-n)`` の 1 が ungrounded に
#: 数えられ、根拠を示せという注記がプロンプトに載った)。
_UNIT_SYSTEM_CONSTANTS = frozenset({
    "1", "7", "10", "12", "24", "52", "60", "100", "365", "366",
    "1000", "1440", "3600", "86400",
    "0.1", "0.01", "0.001",
})

#: 1e6 以上の SI 接頭辞と 2 進接頭辞 (KiB / MiB / GiB / TiB)。定義なので記憶違いの
#: 余地は無いが、**どこに置かれたか** を見ずに説明済みにすると、会話に無い金額の
#: 捏造 (「100 万」) まで定義として通る (実インシデント 2026-09-27 ライブ監査
#: C08#3: 「広告費を20%削減した場合の合計」に ``1000000 * 0.8`` — 会話の広告費は
#: 120 万 — が組まれ、1000000 が SI 接頭辞として無条件に説明済みだったため
#: 561.4 万円 (正 683 万円) が「厳密な計算結果」になった)。換算の位置
#: (:func:`_conversion_position_constants`) か単位の定義 (:func:`_unit_defined_constants`)
#: のときだけ説明済みにする (docs/f_03 §3.4.1)。
_LARGE_UNIT_CONSTANTS = frozenset({
    "1000000", "1000000000", "1000000000000",
    "1024", "1048576", "1073741824", "1099511627776",
})

#: 単位の記号がその換算率を **定義として** 名指しするもの。``ppm`` = 10^6 は
#: 「0.02%は何ppm?」の ``0.0002 * 1000000`` のように、相手が百分率由来でも換算。
#: バイトの接頭辞は SI (10^3n) と 2 進 (1024^k, k≤n) の両定義を受ける。
_PARTS_PER_UNITS: dict[str, tuple[str, ...]] = {
    "ppm": ("1000000",), "ppb": ("1000000000",), "ppt": ("1000000000000",),
}
_BYTE_PREFIX_POWER: dict[str, int] = {
    "K": 1, "M": 2, "G": 3, "T": 4, "キロ": 1, "メガ": 2, "ギガ": 3, "テラ": 4,
}
#: 単位記号 (``ppm`` / ``KB`` / ``gb`` / ``MiB``) と日本語の接頭辞 (「3ギガバイト」
#: 「1ギガは何メガ」)。日本語は長さ・重さ・電力などの単位が続く形を除く。
_UNIT_SYMBOL_RE = re.compile(
    r"(?<![A-Za-z])(?:(ppm|ppb|ppt)|([KMGT])i?B)(?![A-Za-z])"
    r"|(キロ|メガ|ギガ|テラ)(?!メートル|グラム|ワット|カロリー|ヘルツ|トン)",
    re.IGNORECASE,
)


def _unit_defined_constants(text: str) -> set[str]:
    """対話が書いた単位記号が **定義として** 名指しする換算率 (純粋関数)。"""
    found: set[str] = set()
    for parts_per, prefix, ja_prefix in _UNIT_SYMBOL_RE.findall(text or ""):
        if parts_per:
            found.update(_PARTS_PER_UNITS[parts_per.lower()])
            continue
        power = _BYTE_PREFIX_POWER[ja_prefix or prefix.upper()]
        found.add(str(1000 ** power))
        found.update(str(1024 ** k) for k in range(1, power + 1))
    return found & _LARGE_UNIT_CONSTANTS


def _product_leaves(node: ast.AST) -> list[ast.AST]:
    """乗除の連鎖を括弧の中まで平坦化した項 (``(a*b)/(c*d)`` → a, b, c, d)。"""
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Div)):
        return _product_leaves(node.left) + _product_leaves(node.right)
    return [node]


def _numeric_leaf_text(expression: str, node: ast.AST) -> str | None:
    """項が数値リテラル (符号付きを含む) なら、その綴り (符号抜き) を返す。

    底が数値リテラルの冪 (``1024 ** 3``) は底の綴りを返す — 「3ギガバイトは何
    バイト」の ``3 * 1024 ** 3`` で 1024 は相手 3 と並ぶ換算定数 (独立レビュー)。
    """
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        node = node.operand
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        node = node.left
    if not (isinstance(node, ast.Constant) and isinstance(node.value, (int, float))):
        return None
    if isinstance(node.value, bool):
        return None
    return ast.get_source_segment(expression, node)


def _conversion_position_constants(expression: str, partners: set[str]) -> set[str]:
    """**換算の位置** にある大きな換算定数を返す (純粋関数)。

    換算の位置 = 乗除の連鎖 (括弧の中まで平坦化した積の項) の相手に、
    ``partners`` (百分率由来でない会話の数) の数値リテラルがある。
    ``(1277500 * 240) / (1024 * 1024)`` の 1024 は相手が 1277500 / 240 なので
    換算、``1000000 * 0.8`` の 1000000 は相手が 20% 由来の 0.8 だけなので換算では
    ない。相手は数値リテラルの項だけを数える (冪・和の項の中は見ない)。
    構文として読めない式は換算の位置を持たない。
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return set()
    explained: set[str] = set()
    roots = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Mult, ast.Div))
    ]
    for root in roots:
        texts = [_numeric_leaf_text(expression, leaf) for leaf in _product_leaves(root)]
        for i, text in enumerate(texts):
            if text not in _LARGE_UNIT_CONSTANTS:
                continue
            if any(
                other is not None and other in partners
                for j, other in enumerate(texts) if j != i
            ):
                explained.add(text)
    return explained


def _ungrounded_numbers(
    expression: str, query: str, context: str = "",
) -> tuple[str, ...]:
    """式のうち対話から辿れない数値リテラルを、出現順に重複なく返す (純粋関数)。

    モデルが知識から定数を補う (例: クエリにも会話にも無い換算率を持ち出す) と、
    ツールは「正しく計算された嘘」を返してしまう。式に現れる数値リテラルが
    クエリまたは ``context`` 中に文字列として存在するかを見て、説明できない値を
    返す。``context`` を許すのは、会話で一度提示された数値は「対話に書かれた
    事実」であってモデルの想像ではないため。空タプルなら式は接地している。
    真偽ではなく **どの値が説明できないか** を返すのは、式を捨てずに実行する
    経路でその値を回答に開示させるため (``_suppress_ungrounded_calculate``)。

    大きな換算定数 (:data:`_LARGE_UNIT_CONSTANTS`) は、対話に書かれているか、
    換算の位置にあるか、対話の単位記号が定義として名指しするときだけ説明済み
    (docs/f_03 §3.4.1)。
    """
    known = _known_numbers(query) | _known_numbers(context)
    known.update(_UNIT_SYSTEM_CONSTANTS)
    text = f"{query or ''}\n{context or ''}"
    known.update(_unit_defined_constants(text))
    # 相手に数えるのは百分率由来でない会話の数。「20%」の 20 そのものも除く
    # (``1000000*20/100`` を換算と読んでいた、独立レビュー)。
    partners = _known_numbers(
        _PERCENT_LITERAL_RE.sub(" ", query or ""), include_percent=False,
    ) | _known_numbers(_PERCENT_LITERAL_RE.sub(" ", context or ""), include_percent=False)
    known.update(_conversion_position_constants(expression, partners))
    seen: list[str] = []
    # 冪の指数 (``x ** 2``) は式の構造が要求する値で、対話の数値ではない
    # (BMI = 体重 / 身長 ** 2。2026-09-21 ライブ監査の再検証で棄却されていた)。
    for n in _NUMBER_LITERAL_RE.findall(_POWER_EXPONENT_RE.sub("", expression)):
        if n not in known and n not in seen:
            seen.append(n)
    return tuple(seen)


def conversation_operands(
    expression: str, query: str, dialogue: str,
) -> tuple[str, ...]:
    """式の数値のうち、クエリに無く会話 (``dialogue``) に書かれたものを返す (純粋関数)。

    被演算子を会話から取った式か (「最初の計算をやり直して」) を見る。単位系の
    定数 (``1000000`` 等) は ``100万円`` とも読めるので除外しない — 会話に
    書かれていれば会話の値として数える (2026-09-26 監査 C03#3)。
    """
    in_query = _known_numbers(query)
    in_dialogue = _known_numbers(dialogue)
    seen: list[str] = []
    for n in _NUMBER_LITERAL_RE.findall(_POWER_EXPONENT_RE.sub("", expression)):
        if n in in_dialogue and n not in in_query and n not in seen:
            seen.append(n)
    return tuple(seen)


#: 桁区切り入りの数字 (``2,660`` / ``1,234,567``)。アシスタント自身が金額を
#: この書式で提示するため、次のターンでその数値を使う式が「対話に無い数値」と
#: 誤判定されていた (実インシデント 2026-08-03 ライブ監査: 直前の回答
#: 「2,660円です」を受けた ``2926 + 500`` が ungrounded で no_tool に落ち、
#: 決定論の calculate 経路を失って base の暗算に回っていた)。
_GROUPED_NUMBER_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?")

#: 明示されたパーセント (``10%`` / ``10 パーセント``)。
_PERCENT_LITERAL_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|％|パーセント)")

#: 時間の長さ表現 (``2時間30分`` / ``2時間半`` / ``90分`` / ``3時間``)。
#: 「2時間30分」から 2.5 (時間) と 150 (分) は **表記から決定論で導ける**値で、
#: モデルが知識から持ち出した定数ではない。桁区切り・パーセントと同じ扱い。
_DURATION_HM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*時間\s*(\d+(?:\.\d+)?)\s*分")
_DURATION_H_HALF_RE = re.compile(r"(\d+(?:\.\d+)?)\s*時間半")
_DURATION_H_RE = re.compile(r"(\d+(?:\.\d+)?)\s*時間")
_DURATION_M_RE = re.compile(r"(\d+(?:\.\d+)?)\s*分(?!の)")
#: 「5分30秒」型 (分 + 秒)。``時間 + 分`` と同じ構造なのに欠けていた。
#: ペース (「キロ5分30秒」) や所要時間で普通に使う表記で、式に現れるのは
#: ``5.5`` (分) や ``330`` (秒) であってクエリに書かれた ``5`` と ``30`` ではない。
_DURATION_MS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*分\s*(\d+(?:\.\d+)?)\s*秒")

#: 時間の刻み幅 (``30分刻み`` / ``15分単位`` / ``10分ごと`` / ``30分間隔``)。
#: 式に現れるのは「1時間あたりの区画数」(30分刻み → 2) で、クエリには 30 しか
#: 書かれていない。時間+分・分+秒と同じく **表記から一意に導ける**値。
#: 実インシデント 2026-08-10 ライブ監査: 「9時から18時まで30分刻み、8部屋、
#: 5営業日」でネイティブ層が正しく ``8 * 5 * (18 - 9) * 2`` (= 720) を合成したのに
#: ``2`` が ungrounded 判定になり no_tool へ落ち、base の暗算で 1,680 / 1,440 と
#: 誤答した (しかも見出しと計算式が食い違った)。
_INTERVAL_M_RE = re.compile(r"(\d+(?:\.\d+)?)\s*分\s*(?:刻み|単位|ごと|間隔)")

#: 曜日のレンジ (``月〜金`` / ``月曜から金曜まで``)。日数はレンジから一意に決まる
#: (月〜金 → 5) が、クエリに数字としては現れない。上の刻み幅と同じ回で必要に
#: なった (2026-08-10 ライブ監査の総スロット数は ``8 * 5 * (18 - 9) * 2`` で、
#: ``5`` も ``2`` も書かれていなかった)。
#: 日本語の万進表記 (``2,850万`` / ``3千万`` / ``1億2000万`` / ``2.5億``)。式に
#: 現れるのは展開した整数 (``28500000``) で、クエリには ``2,850`` と ``万`` しか
#: 書かれていない。桁区切りと同じく **表記から一意に導ける**値。
#: 実インシデント 2026-09-05 ライブ監査 T03/1: 「残債が2,850万円」に対し
#: ネイティブ層が正しく ``28500000`` を使ったのに ungrounded と判定され、
#: 「この値の根拠を示せ」という無意味な注記がプロンプトに載った。
#: 読みは ``response_arithmetic.iter_ja_numbers`` (:func:`_myriad_derived_numbers` /
#: :func:`_myriad_pairs`)。

_WEEKDAY_ORDER = "月火水木金土日"
#: SI 接頭辞付きの長さ / 質量 (``172cm`` / ``500g`` / ``3km`` は対象外 — km は
#: 基本単位への換算が乗算になり、式側でも 1000 を使うので定数で足りる)。
_METRIC_PREFIX_DIVISORS: dict[str, float] = {"cm": 100.0, "mm": 1000.0, "g": 1000.0}
_METRIC_PREFIXED_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(cm|mm|g)(?![a-zA-Z])", re.IGNORECASE)

#: ``** 2`` のような冪の指数 (整数リテラル)。符号付き (``**-10``) と括弧で囲んだ
#: 1 つの数 (``**(-10)``) も剥がす — ``\*\*\s*\d+`` だけだった頃は ``**-10`` の 10 が
#: 残り、定数表に 10 があったから通っていただけだった (2026-09-27)。式の指数
#: (``**(-23*12)``) の中の数は剥がさない — 年数 23 は対話から辿れるべき被演算子。
_POWER_EXPONENT_RE = re.compile(r"\*\*\s*(?:[-+]?\s*\d+|\(\s*[-+]?\s*\d+\s*\))")

_WEEKDAY_RANGE_RE = re.compile(
    r"([月火水木金土日])\s*(?:曜日?)?\s*(?:〜|～|-|–|から)\s*([月火水木金土日])\s*(?:曜日?)?",
)


def percent_derived_values(text: str) -> list[float]:
    """``text`` の百分率・割合の語から導ける値 (率 / 加算後の倍率 / 割引後の倍率、純粋関数)。

    「10%」からは 0.1 (率) と 1.1 (加算後の倍率) と 0.9 (割引後の倍率) が
    導ける。「15% 引き」の式は ``× 0.85`` になるのが普通で、0.85 を
    「対話に無い数値」と記録していた (2026-09-10 ライブ監査 (h) H-07)。

    **百分率由来の値の分類はこの 1 関数** (docs/f_03 §3.4.1)。既知の数
    (:func:`_known_numbers`)、換算定数の相手 (:func:`_ungrounded_numbers`)、
    桁の取り違え (:func:`percent_scale_slips`)、組み直しの新値判定
    (``tool_call_judge``) が共有する — 百分率の展開を複数書くと片方だけ直る。
    """
    values: list[float] = []
    for pct in _PERCENT_LITERAL_RE.findall(text or ""):
        try:
            rate = float(pct) / 100.0
        except ValueError:
            continue
        values.extend((rate, 1.0 + rate, 1.0 - rate))
    # 割合の語 (「3割」「3割引」「半分」「半額」) も同じ種類の値。無いと
    # ``100000000*0.03*0.3`` (売上 1 億の 3% の 3 割) の 0.3 や ``5000*0.95*0.5``
    # (半額) の 0.5 が説明できない数になり、桁の取り違えと誤って読まれた
    # (独立レビュー 2026-09-27)。
    for token in _WARI_RE.findall(text or ""):
        tenths = _wari_tenths(token)
        if tenths is not None:
            rate = tenths / 10.0
            values.extend((rate, 1.0 + rate, 1.0 - rate))
    if _HALF_RE.search(text or ""):
        values.append(0.5)
    return values


#: 「3割」「3割引」「三割」 (割合の語)。
_WARI_RE = re.compile(r"(\d+(?:\.\d+)?|[一二三四五六七八九十])\s*割")
#: 「半分」「半額」「半値」「半減」。
_HALF_RE = re.compile(r"半(?:分|額|値|減)")


def _wari_tenths(token: str) -> float | None:
    """「3」「三」「十」を割の数に (純粋関数)。"""
    value = kanji_number_value(token)
    if value is not None:
        return float(value)
    try:
        return float(token)
    except ValueError:
        return None


def percent_derived_numbers(text: str) -> set[str]:
    """:func:`percent_derived_values` を式の綴りで (``0.1`` / ``1.10`` の両表記)。"""
    found: set[str] = set()
    for value in percent_derived_values(text):
        # 式側の表記ゆれ (1.1 / 1.10) を吸収するため両方を登録する。
        found.add(f"{value:g}")
        found.add(f"{value:.2f}")
    return found


def _known_numbers(text: str, *, include_percent: bool = True) -> set[str]:
    """``text`` に「書かれている」とみなせる数値リテラルを集める (純粋関数)。

    素の数字に加えて次を同一視する。いずれも **対話に現れた表記から決定論で
    導ける**もので、モデルが知識から持ち出した定数ではない:

    - 桁区切り: ``2,660`` → ``2660``。数える側 (式) は区切りを打たないため、
      正規化しないと自分が直前に提示した金額を「知らない数値」と判定してしまう
    - パーセント: ``10%`` → ``0.1`` / ``1.1`` / ``1.10`` (:func:`percent_derived_numbers`)。
      税率・割引率の計算で式に現れる倍率は、クエリ中の百分率から一意に決まる。
      ``include_percent=False`` は百分率由来の値を除いた「会話の数」
      (換算定数の相手の判定、docs/f_03 §3.4.1)
    - 時間表現 / 万進表記 / 漢数字の量 / SI 接頭辞付きの長さ・質量
    """
    known = set(_NUMBER_LITERAL_RE.findall(text))
    for grouped in _GROUPED_NUMBER_RE.findall(text):
        known.add(grouped.replace(",", ""))
    if include_percent:
        known.update(percent_derived_numbers(text))
    known.update(_duration_derived_numbers(text))
    known.update(_myriad_derived_numbers(text))
    known.update(_kanji_quantity_numbers(text))
    known.update(_metric_prefix_derived_numbers(text))
    return known


#: 漢数字の量 (``百万`` / ``三千万`` / ``一億二千万`` / ``二十``)。万進表記
#: (``iter_ja_numbers``) は係数に算用数字を要求するので、「百万円を年利3%で」の
#: 1000000 を拾えなかった (2026-09-27、F1 の反例)。先頭は数字か 十百千 に限る
#: (単独の「万」「億」は量ではない — 「万が一」)。
_KANJI_QUANTITY_RE = re.compile(r"[〇一二三四五六七八九十百千][〇一二三四五六七八九十百千万億兆]*")
def _kanji_quantity_numbers(text: str) -> set[str]:
    """漢数字の量を算用数字の綴りで集める (``百万円`` → ``1000000``、純粋関数)。"""
    found: set[str] = set()
    for token in _KANJI_QUANTITY_RE.findall(text or ""):
        value = kanji_number_value(token)
        if value is not None:
            found.add(str(value))
    return found


def _metric_prefix_derived_numbers(text: str) -> set[str]:
    """SI 接頭辞付きの量を基本単位へ直した値を集める (純粋関数)。

    ``172cm`` → ``1.72`` (m)。BMI の式は身長を m で使うので、式に現れるのは
    クエリに書かれた 172 ではない (2026-09-21 ライブ監査の再検証で
    ``82 / (1.72 ** 2)`` が ungrounded として棄却された)。センチ = 1/100 は
    接頭辞の **定義** で、``_UNIT_SYSTEM_CONSTANTS`` の ``0.01`` と同じ扱い。
    """
    derived: set[str] = set()
    for num, unit in _METRIC_PREFIXED_RE.findall(text):
        value = float(num) / _METRIC_PREFIX_DIVISORS[unit.lower()]
        derived.add(f"{value:g}")
        derived.add(f"{value:.2f}")
    return derived


def _myriad_derived_numbers(text: str) -> set[str]:
    """万進表記から導ける数値を集める (純粋関数)。

    ``2,850万`` → ``28500000`` (展開値) と ``2850`` (万単位の係数)。モデルは
    万単位のまま式を組むこともある (``2850 * 0.0135``) ので両方を登録する。
    ``1億2000万`` / ``9 万 1,855.33`` のような連結は 1 つの数 (``120000000`` /
    ``91855.33``)。展開値の読みは応答の数の読みと同じ
    ``response_arithmetic.iter_ja_numbers`` の 1 本 (不変則 #14(a)) — 別に持っていた
    読みは万の後ろの端数を落とし、直前の回答「9 万 1,855.33 円」を使った式
    ``30000000 + 91855.33 * 35 * 12`` を「会話に無い数」と判定した
    (2026-09-28 再監査 C03#2)。
    """
    derived: set[str] = set()

    def add(value: float) -> None:
        if value == int(value):
            derived.add(str(int(value)))
        else:
            derived.add(f"{value:.10f}".rstrip("0").rstrip("."))

    for num in iter_ja_numbers(text):
        if not num.has_unit:
            continue
        add(num.value)
        for coefficient, _multiplier in _myriad_terms(num):
            add(coefficient)
    return derived


def _myriad_terms(num: JaNumber) -> list[tuple[float, float]]:
    """万進の項 (係数, 倍率)。倍率が千以上の項だけ (「3百」の 3 は係数に数えない)。

    係数の読みも ``iter_ja_numbers`` の 1 本 (以前は別の正規表現で読み、全角の桁区切り
    や「2 万 6 千」の空白で読みが割れていた、独立レビュー L3)。
    """
    return [(c, m) for c, m in num.unit_terms if m >= 1000]


def _duration_derived_numbers(text: str) -> set[str]:
    """時間の長さ表現から導ける数値を集める (純粋関数)。

    「2時間30分で何km進むか」型の文章題では、式に現れるのは ``2.5`` (時間) や
    ``150`` (分) であって、クエリに書かれた ``2`` と ``30`` ではない。桁区切り・
    パーセントと同じく **表記から一意に導ける**値なので、捏造ではない。

    実インシデント (2026-08-08 ライブ監査): 「時速240kmで2時間30分走ると何km
    進みますか。」でネイティブ層が正しく ``240 * 2.5`` を選んだのに、``2.5`` が
    クエリに無いという理由で ungrounded 判定になり no_tool へ落ちた。

    「分 + 秒」も同じ構造なのに欠けていた (2026-08-09 2 回目のライブ監査):
    「フルマラソンの距離を キロ5分30秒 のペースで走ると何時間何分？」で
    ネイティブ層が正しく ``42.195 * 5.5`` を選んだのに、``5.5`` が
    (5 と 30 しか書かれていないため) ungrounded 判定になり no_tool へ落ち、
    base の暗算で「3時間47分15秒」と誤答した (正 3時間52分4秒)。
    補助タスク非常駐でも救済経路自体は生きており、塞いでいたのはこのゲートだった。
    """
    derived: set[str] = set()

    def add(value: float) -> None:
        derived.add(f"{value:g}")
        derived.add(f"{value:.2f}")

    for hours, minutes in _DURATION_HM_RE.findall(text):
        total_h = float(hours) + float(minutes) / 60.0
        add(total_h)
        add(float(hours) * 60.0 + float(minutes))
    for minutes, seconds in _DURATION_MS_RE.findall(text):
        # 分単位 (5分30秒 → 5.5) と秒単位 (→ 330)。時間+分と同じ 2 通り。
        add(float(minutes) + float(seconds) / 60.0)
        add(float(minutes) * 60.0 + float(seconds))
    for hours in _DURATION_H_HALF_RE.findall(text):
        add(float(hours) + 0.5)
        add((float(hours) + 0.5) * 60.0)
    for hours in _DURATION_H_RE.findall(text):
        add(float(hours) * 60.0)
    for minutes in _DURATION_M_RE.findall(text):
        add(float(minutes) / 60.0)
    for minutes in _INTERVAL_M_RE.findall(text):
        step = float(minutes)
        if step > 0:
            # 1 時間あたりの区画数 (30分刻み → 2) と、区画の時間 (→ 0.5)。
            add(60.0 / step)
            add(step / 60.0)
    for start, end in _WEEKDAY_RANGE_RE.findall(text):
        # 両端を含む日数 (月〜金 → 5)。週をまたぐ指定 (金〜月) も剰余で数える。
        span = (_WEEKDAY_ORDER.index(end) - _WEEKDAY_ORDER.index(start)) % 7 + 1
        add(float(span))
    return derived
# ── 式の妥当性 (接地とは別の軸) ──────────────────────────────────────
#
# 接地判定は「式の数値が対話に書かれているか」しか見ない。数値が全て正しく
# 接地していても **式の組み方** が誤ることがあり、ツールは「正しく計算された嘘」
# を返す (実インシデント 2026-09-05 ライブ監査 T03/1: 年 1.35% のローンで
# ``28500000 * 0.0135 * 12 / (1 - (1 + 0.0135/12)**(-23*12))`` と **年利を
# 12 倍** した式が合成され、月返済額の 144 倍 = 17,305,634 が「厳密な計算結果」
# として提示された)。意味解析はできないが、**対話の表記と式の構造の食い違い**
# は決定論で拾える。接地の注記と同じく、式は捨てずに疑いを名指しして検算を
# 求める (``_suppress_ungrounded_calculate`` の方針)。

#: 「年 1.35%」「年利 1.35 %」「年率1.35%」 — 年あたりの率。
_ANNUAL_RATE_RE = re.compile(r"年(?:利|率|利率)?\s*(\d+(?:\.\d+)?)\s*(?:%|％|パーセント)")
#: 「月 0.5%」「月利 0.5%」 — 月あたりの率。
_MONTHLY_RATE_RE = re.compile(r"月(?:利|率|利率)?\s*(\d+(?:\.\d+)?)\s*(?:%|％|パーセント)")


def _rate_forms(pct: str) -> list[str]:
    """百分率 ``1.35`` が式に現れうる表記 (``0.0135`` / ``1.35/100`` / ``1.35 / 100``)。"""
    try:
        rate = float(pct) / 100.0
    except ValueError:
        return []
    forms = {f"{rate:g}", f"{rate:.4f}".rstrip("0").rstrip("."), f"{pct}/100", f"{pct} / 100"}
    return [re.escape(f) for f in forms if f]


def expression_sanity_issues(
    expression: str, query: str, context: str = "", *, user_text: str | None = None,
) -> tuple[str, ...]:
    """式の組み方が対話の表記と食い違う疑いを返す (純粋関数)。

    返すのは **疑いの説明文** (回答側の注記にそのまま使う)。空タプルなら疑い無し。
    ``user_text`` は会話のうち user の発言だけの本文で、5 の期間の候補はここからだけ
    取る (``None`` なら ``context`` 全体)。検出するのは構造だけで決まる 5 種:

    1. 年率を 12 倍 / 月率を 12 で割る (期間単位の取り違え)
    2. 百分率を 2 回割る (``1.35/100/100`` / ``0.0135/100``)
    3. 万円の係数と円の展開値の混在 (``2850 * 0.0135 + 627000``)
    4. 月数 (年数×12) の冪に年率をそのまま使う (``1.012 ** 420``、
       :func:`_compounding_period_issues`)
    5. 月の複利の冪の指数が会話の期間と合わない (「35年」に ``** -360``、
       :func:`period_exponent_mismatches`)
    """
    expr = expression or ""
    text = f"{query or ''}\n{context or ''}"
    issues: list[str] = []

    for pct in _ANNUAL_RATE_RE.findall(text):
        for form in _rate_forms(pct):
            if re.search(rf"(?:{form})\s*\*\s*12\b|\b12\s*\*\s*(?:{form})", expr):
                issues.append(
                    f"年率 {pct}% を 12 倍している (年率を月率にするのは ÷12、"
                    "年額はそのまま年率を掛ける)"
                )
                break
    for pct in _MONTHLY_RATE_RE.findall(text):
        for form in [*_rate_forms(pct), re.escape(pct)]:
            if re.search(rf"(?:{form})\s*/\s*12\b", expr):
                issues.append(f"月率 {pct}% を 12 で割っている (月率はそのまま月に掛ける)")
                break

    for pct in _PERCENT_LITERAL_RE.findall(text):
        # 小数形 (0.0135) の直後に /100、または pct/100/100 — 百分率を 2 回割っている
        decimal_forms = [f for f in _rate_forms(pct) if "/" not in f]
        double = any(
            re.search(rf"(?<![\d.])(?:{form})\s*/\s*100\b", expr) for form in decimal_forms
        ) or re.search(rf"(?<![\d.]){re.escape(pct)}\s*/\s*100\s*/\s*100\b", expr)
        if double:
            try:
                issues.append(
                    f"{pct}% を百分率として 2 回割っている ({pct}% は {float(pct) / 100:g})"
                )
            except ValueError:
                pass

    issues.extend(_compounding_period_issues(expr, text))
    for exponent, periods in _period_exponent_mismatch_pairs(
        expr, text, _period_text(query, context, user_text),
    ):
        listed = "、".join(str(p) for p in periods)
        issues.append(
            f"月の複利の冪の指数 {exponent} が会話の期間の月数 ({listed}) と一致しない "
            "(年数 × 12 か、会話に書かれた回数を使う)"
        )
    issues.extend(_annuity_without_monthly_rate(expr, text))
    for exponent in _annual_period_mismatches(expr, text, _period_text(query, context, user_text)):
        issues.append(
            f"年率の複利の冪の指数 {exponent} が会話の期間 (年数) と一致しない "
            "(月の複利なら底は 1 + 年率/12 で指数は年数 × 12、年の複利なら指数は年数)"
        )

    pairs = _myriad_pairs(text)
    if len(pairs) >= 1:
        used_coefficient = any(
            re.search(rf"(?<![\d.]){re.escape(c)}(?![\d.])", expr) for c, _ in pairs
        )
        # 展開値そのもの、または万単位では現れない大きな円の値 (10 万以上)
        used_expanded = any(
            re.search(rf"(?<![\d.]){re.escape(e)}(?![\d.])", expr) for _, e in pairs
        ) or any(
            float(n) >= 100000 and n not in {c for c, _ in pairs}
            for n in _NUMERIC_LITERAL_RE.findall(expr)
        )
        if used_coefficient and used_expanded:
            issues.append("万円の係数 (万単位) と円の展開値が同じ式に混在している (単位を揃える)")
    return tuple(issues)


#: 期間の年数 (「35年」「20 年間」)。「年利」「年率」「年齢」の年は数えない。
_YEARS_RE = re.compile(r"(\d+)\s*年(?!利|率|齢|生|代|目|前|後)")
#: 「月々 1%」「毎月 0.5% ずつ」— 月あたりの率 (``_MONTHLY_RATE_RE`` の「月利」表記以外)。
_PER_MONTH_RATE_RE = re.compile(
    r"(?:月々|毎月|月あたり|ひと月)[^。\n\d]{0,8}?(\d+(?:\.\d+)?)\s*(?:%|％|パーセント)",
)
#: 定数部分木の評価で許す指数の上限 (``calc`` の上限と同じ桁)。
_MAX_CONSTANT_EXPONENT = 10_000


def _constant_value(node: ast.AST) -> float | None:
    """数値リテラルと四則・冪だけの部分木を評価する (純粋関数)。それ以外は ``None``。"""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return None if isinstance(node.value, bool) else float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        inner = _constant_value(node.operand)
        if inner is None:
            return None
        return -inner if isinstance(node.op, ast.USub) else inner
    if not isinstance(node, ast.BinOp):
        return None
    left = _constant_value(node.left)
    right = _constant_value(node.right)
    if left is None or right is None:
        return None
    try:
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow) and abs(right) <= _MAX_CONSTANT_EXPONENT:
            return float(left ** right)
    except (ZeroDivisionError, OverflowError, ValueError, TypeError):
        return None
    return None


def _power_operands(tree: ast.AST) -> list[tuple[ast.AST, ast.AST]]:
    """式の全ての冪 (``a ** b`` / ``pow(a, b)``) の (底, 指数)。"""
    found: list[tuple[ast.AST, ast.AST]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            found.append((node.left, node.right))
        elif (
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "pow" and len(node.args) == 2
        ):
            found.append((node.args[0], node.args[1]))
    return found


def _annual_percents(text: str) -> list[str]:
    """年率とみなす百分率 (出現順・重複なし、純粋関数)。

    月率と明示された百分率 (``月利 0.5%``、「月々 1%」「毎月 0.5%」) を除くすべて。
    「金利1.5%」は年率で書かれるのが普通で、「年」の接頭は付かない。
    """
    # 「月々 1%」「毎月 0.5%」も月率 (独立レビュー: ``1.01 ** 60`` に誤った案内)
    monthly = set(_MONTHLY_RATE_RE.findall(text or "")) | set(
        _PER_MONTH_RATE_RE.findall(text or ""),
    )
    annual: list[str] = []
    for pct in _PERCENT_LITERAL_RE.findall(text or ""):
        if pct not in monthly and pct not in annual:
            annual.append(pct)
    return annual


#: 期間の年数の候補 (「35年」「10年後」「3年目」「20年間」)。:data:`_YEARS_RE` と違い
#: 「年後」「年目」も数える — ここは指数の照合の **許す側** なので、候補を広げるほど
#: 疑いは減る (「10年後の残債」の ``** 120`` を疑わない)。
_PERIOD_YEARS_RE = re.compile(r"(\d+)\s*年(?!利|率|齢)")
#: 月数の明示 (「24か月」「420 ヶ月」「300回払い」)。
_PERIOD_MONTHS_RE = re.compile(r"(\d+)\s*(?:[かヶケヵカ]月|回払い)")


def _period_months(text: str) -> set[int]:
    """会話に書かれた期間の月数 (年数 × 12 と明示の月数、純粋関数)。"""
    normalized = (text or "").translate(_FULLWIDTH_DIGITS)
    months = {int(y) * 12 for y in _PERIOD_YEARS_RE.findall(normalized) if int(y) > 0}
    months.update(int(m) for m in _PERIOD_MONTHS_RE.findall(normalized) if int(m) > 0)
    return months


def _period_text(query: str, context: str, user_text: str | None) -> str:
    """期間の候補を取る本文 (純粋関数)。``user_text`` があればクエリ + user の発言だけ。

    アシスタントの前の回答が誤って「360回払い」と書くと、それを根拠に ``** -360`` を
    許してしまう (2026-09-28 独立レビュー L1)。
    """
    return f"{query or ''}\n{context if user_text is None else user_text or ''}"


def _period_exponent_mismatch_pairs(
    expression: str, text: str, period_text: str | None = None,
) -> list[tuple[str, tuple[int, ...]]]:
    """月の複利の冪で、指数が会話の期間と合わないもの (純粋関数)。

    返すのは ``(指数の綴り (符号なし), 会話の期間の月数)`` の列。底が 1 + 年率/12
    (元利均等・積立の形) の冪だけを見る。期間は ``period_text`` (無ければ ``text``)
    から取る。12 (1 年ぶんの月の複利、実効年率) と **期間どうしの差** (残りの期間・
    据置のあとの期間、「35年…5年後に繰上げ返済」の 360) は許す (独立レビュー M1)。
    会話に期間も年率も無ければ空。
    """
    months, powers = _mismatched_period_powers(expression, text, period_text)
    periods = tuple(sorted(months))
    found: list[tuple[str, tuple[int, ...]]] = []
    for _node, _sign, n in powers:
        if not any(str(n) == seen for seen, _ in found):
            found.append((str(n), periods))
    return found


def _mismatched_period_powers(
    expression: str, text: str, period_text: str | None = None,
) -> tuple[set[int], list[tuple[ast.AST, int, int]]]:
    """会話の期間の月数と、指数が期間と合わない月の複利の冪 (純粋関数)。

    返すのは ``(期間の月数, [(指数の節, 指数の符号 (±1), 指数の絶対値)])``。判定は
    :func:`_period_exponent_mismatch_pairs` の docstring のとおり (同じ冪を 2 回
    書いた式は節ごとに返す)。
    """
    months = _period_months(text if period_text is None else period_text)
    if not months:
        return set(), []
    bases = [1.0 + float(pct) / 100.0 / 12.0 for pct in _annual_percents(text)]
    if not bases:
        return months, []
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return months, []
    allowed = months | {12} | {a - b for a in months for b in months if a > b}
    found: list[tuple[ast.AST, int, int]] = []
    for base_node, exponent_node in _power_operands(tree):
        base = _constant_value(base_node)
        if base is None or not any(
            math.isclose(base, b, rel_tol=0.0, abs_tol=1e-12) for b in bases
        ):
            continue
        exponent = _constant_value(exponent_node)
        if exponent is None or not float(abs(exponent)).is_integer():
            continue
        n = int(abs(exponent))
        if n in allowed:
            continue
        found.append((exponent_node, -1 if exponent < 0 else 1, n))
    return months, found


def _annuity_without_monthly_rate(expression: str, text: str) -> list[str]:
    """元利均等の係数を使うのに月利を掛けていない疑い (純粋関数)。

    2026-09-29 実機確認 (R27_loan_correction #3): 理由付きの作り直しで底と回数は直ったが
    ``30000000 * (1 + 0.012/12) ** 420 / ((1 + 0.012/12) ** 420 - 1) / 12`` (72,926 円、
    正しくは 87,510 円) が通った。毎月の返済額は 元本 × 月利 × g^n / (g^n - 1)
    (g = 1 + 年率/12) で、月利を ÷12 にしている。月の複利の冪が 2 回以上現れるか、
    ``1 - g ** -n`` の形 (元利均等の係数) なのに、式のどこにも月利 (年率/12) も年率も
    冪の底の外で現れなければ返す。一括の複利 (``P * g ** n``) は係数ではないので見ない。
    """
    rates = [(pct, float(pct) / 100.0 / 12.0) for pct in _annual_percents(text)]
    if not rates:
        return []
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return []
    monthly_powers = []
    used: list[tuple[str, float]] = []
    base_ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            base = _constant_value(node.left)
            matched = [
                (pct, r) for pct, r in rates
                if base is not None and math.isclose(base, 1.0 + r, abs_tol=1e-12)
            ]
            if matched:
                monthly_powers.append(node)
                used.extend(m for m in matched if m not in used)
                base_ids.update(id(n) for n in ast.walk(node.left))
    if not monthly_powers:
        return []
    negative = any(
        (_constant_value(p.right) or 0) < 0 for p in monthly_powers
    )
    if len(monthly_powers) < 2 and not negative:
        return []
    for node in ast.walk(tree):
        if id(node) in base_ids:
            continue
        value = _constant_value(node)
        # 月利そのもの、または年率 (``P * 0.0135 / 12 / (...)`` は左結合で
        # 0.0135/12 が部分木にならない) が底の外にあれば掛けている
        if value is not None and any(
            math.isclose(value, r, abs_tol=1e-12) or math.isclose(value, r * 12.0, abs_tol=1e-12)
            for _p, r in rates
        ):
            return []
    # 名指すのは底に使われた率 (訂正前の率が会話の先に出ていても取り違えない)
    pct, rate = used[0]
    return [
        f"元利均等の係数 g^n / (g^n - 1) (g = 1 + 年率/12) を使っているのに月利 {pct}%/12 = "
        f"{rate:.6g} を掛けていない (毎月の返済額 = 元本 × 月利 × 係数)"
    ]


def _annual_period_mismatches(
    expression: str, text: str, period_text: str | None = None,
) -> tuple[str, ...]:
    """年の複利の冪 (底が 1 + 年率) で、指数が会話の年数と合わないもの (純粋関数)。

    2026-09-29 回帰確認 (R27_loan_correction #3): 35 年・1.2% の訂正の組み直しで
    ``30000000 * (1.012 ** 360) / ((1.012 ** 360) - 1) / 12`` が通った。4 つ目の検査
    (:func:`_compounding_period_issues`) は指数が会話の月数のときだけ、5 つ目
    (:func:`_mismatched_period_powers`) は底が 1 + 年率/12 のときだけ見るので、底も
    回数も違う式はどちらにも掛からなかった。年の複利の指数は会話の年数か年数どうしの
    差 (残りの期間) のはずなので、そのどれとも合わなければ返す。指数が会話の月数なら
    4 つ目が「年率のまま」と言うので、ここでは返さない (二重に出さない)。会話に年数が
    無ければ空。回数の置き換え (:func:`repair_period_exponents`) には使わない — 底が
    年率のままなので、回数だけ直しても誤りのまま。
    """
    months = _period_months(text if period_text is None else period_text)
    years = {m // 12 for m in months if m % 12 == 0}
    if not years:
        return ()
    bases = [1.0 + float(pct) / 100.0 for pct in _annual_percents(text)]
    if not bases:
        return ()
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return ()
    allowed = years | months | {a - b for a in years for b in years if a > b}
    found: list[str] = []
    for base_node, exponent_node in _power_operands(tree):
        base = _constant_value(base_node)
        if base is None or not any(
            math.isclose(base, b, rel_tol=0.0, abs_tol=1e-12) for b in bases
        ):
            continue
        exponent = _constant_value(exponent_node)
        if exponent is None or not float(abs(exponent)).is_integer():
            continue
        n = int(abs(exponent))
        if n not in allowed and str(n) not in found:
            found.append(str(n))
    return tuple(found)


def repair_period_exponents(
    expression: str, query: str, context: str = "", *, user_text: str | None = None,
) -> tuple[str, tuple[str, ...], int] | None:
    """期間と合わない月の複利の冪の指数を、会話の期間の月数に置き換える (純粋関数)。

    返すのは ``(置き換えた式, 置き換えた指数 (符号なし), 会話の期間の月数)``。
    置き換えは式の構造 (AST の指数の節の位置) で行い、符号は保つ
    (``** -360`` → ``** -420``)。式の他の部分は 1 文字も変えない。期間の候補
    (:func:`_period_months`、:func:`period_exponent_mismatches` と同じ 1 関数) が
    **1 つに決まらない**とき、合わない冪が無いとき、式が 1 行でないときは ``None``。
    次の 2 つも ``None`` (2026-09-28 独立レビュー):

    - 合わない指数が **ユーザーの発言に数として書かれている** — 指数そのものが接地
      している。「返済回数420回…10年固定」は「420回」が期間の候補に読めず、無関係な
      「10年」だけが候補になって正しい ``** -420`` を ``** -120`` に書き換えていた。
    - 指数 × 12 が期間の月数 — 年数を指数にした構造の誤り (``(1 + r/12) ** -35 * 12``)。
      回数だけ直すと ``** -420 * 12`` の負の値を実行する。

    置き換えた式が接地・式の妥当性・値の符号を通るかは呼出側が確かめる (docs/f_03 §3.1、
    2026-09-28 実機確認 R10)。
    """
    expr = expression or ""
    period_text = _period_text(query, context, user_text)
    months, powers = _mismatched_period_powers(
        expr, f"{query or ''}\n{context or ''}", period_text,
    )
    if len(months) != 1 or not powers:
        return None
    user_numbers = _numeric_literals(period_text)
    if any(str(n) in user_numbers or n * 12 in months for _node, _sign, n in powers):
        return None
    (term,) = months
    raw = expr.encode("utf-8")
    spans: list[tuple[int, int, str]] = []
    for node, sign, _n in powers:
        if node.lineno != 1 or node.end_lineno != 1 or node.end_col_offset is None:
            return None
        spans.append((node.col_offset, node.end_col_offset, f"{'-' if sign < 0 else ''}{term}"))
    # 後ろから置き換える (前の節のバイト位置をずらさない)。入れ子の冪で節が重なれば諦める。
    spans.sort(reverse=True)
    if any(earlier[1] > later[0] for later, earlier in zip(spans, spans[1:], strict=False)):
        return None
    for start, end, replacement in spans:
        raw = raw[:start] + replacement.encode("utf-8") + raw[end:]
    replaced = tuple(dict.fromkeys(str(n) for _node, _sign, n in powers))
    return raw.decode("utf-8"), replaced, term


def period_exponent_mismatches(
    expression: str, query: str, context: str = "", *, user_text: str | None = None,
) -> tuple[str, ...]:
    """月の複利の冪の指数のうち、会話の期間と合わないもの (純粋関数)。

    実インシデント (2026-09-28 再監査 C03#3): 「35年」の住宅ローンの訂正の再計算に
    ``10000000 * (0.012 / 12) / (1 - (1 + 0.012 / 12) ** -360)`` が組まれた。冪の
    指数は接地 (:func:`_ungrounded_numbers`) の前に剥がすので 360 は見えず、
    構造検査 (:func:`_compounding_period_issues`) も率が ÷12 されているので不発
    だった。会話に「N年」(→ N×12) か月数の明示 (「24か月」「300回払い」) があり、
    底が 1 + 年率/12 の冪の指数 (符号を除く) がそのどれとも合わなければ返す
    (docs/f_03 §3.4.1 の 5 つ目)。期間の候補は ``user_text`` (user の発言だけ) から
    取る (``None`` なら ``context`` 全体)。
    """
    text = f"{query or ''}\n{context or ''}"
    period_text = _period_text(query, context, user_text)
    monthly = tuple(
        exponent for exponent, _ in _period_exponent_mismatch_pairs(
            expression or "", text, period_text,
        )
    )
    annual = _annual_period_mismatches(expression or "", text, period_text)
    return monthly + tuple(n for n in annual if n not in monthly)


def _compounding_period_issues(expression: str, text: str) -> list[str]:
    """月数 (年数×12) の冪に **年率のまま** の率を使っている疑い (純粋関数)。

    実インシデント (2026-09-27 ライブ監査 C03#3): 35 年・1.2% の元利均等返済で
    ``30000000 * (1.012 ** 420) / ((1.012 ** 420) - 1) / 12`` が組まれ、月返済額
    2,516,789 円 (正 87,510 円) が「厳密な計算結果」になった。月の複利は
    1 + 年率/12 で、420 乗するなら底は 1.001。月率と明示された百分率 (``月利``)
    だけを除き、他の百分率は年率とみなす (「金利1.5%」は年率で書かれる)。
    """
    months = {int(y) * 12 for y in _YEARS_RE.findall(text or "") if int(y) > 0}
    if not months:
        return []
    annual = _annual_percents(text)
    if not annual:
        return []
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return []
    issues: list[str] = []
    for base_node, exponent_node in _power_operands(tree):
        exponent = _constant_value(exponent_node)
        if exponent is None or not float(abs(exponent)).is_integer():
            continue
        n = int(abs(exponent))
        if n not in months:
            continue
        base = _constant_value(base_node)
        if base is None:
            continue
        for pct in annual:
            rate = float(pct) / 100.0
            if math.isclose(base, 1.0 + rate, rel_tol=0.0, abs_tol=1e-12):
                issue = (
                    f"月数 {n} (= {n // 12} 年 × 12) の冪に年率 {pct}% をそのまま使っている "
                    f"(月の複利の底は 1 + {pct}%/12 = {1.0 + rate / 12:.6g})"
                )
                if issue not in issues:
                    issues.append(issue)
    return issues


#: 桁の取り違えとみなす倍率 (n×10 / n÷10 / n×100 / n÷100)。
_SCALE_SLIP_FACTORS = (10.0, 0.1, 100.0, 0.01)
#: 期間の単位 (倍率の 10 / 100 が期間の数のときは率 × 期間の正しい式でありうる)。
_PERIOD_UNIT_PATTERN = r"(?:年|回|[かヶケヵカ]月|期|週)"


def percent_scale_slips(
    expression: str, query: str, context: str = "",
) -> tuple[str, ...]:
    """百分率の **桁の取り違え** とみなせる説明できない数を返す (純粋関数)。

    説明できない数 n (:func:`_ungrounded_numbers`) のうち、n×10 / n÷10 /
    n×100 / n÷100 が百分率由来の値 (:func:`percent_derived_values`) と一致する
    もの。実インシデント (2026-09-27 ライブ監査 C03#3): 1.2% を ``0.12`` とした
    ``30000000 * (0.12/12) / (1 - (1 + 0.12/12)**(-35*12))`` が「会話に無い値を
    仮定した試算」として開示付きで実行され、304,665 円 (正 87,510 円) が答えに
    なった。仮定ではなく読み違いなので、開示ではなく不合格にする (docs/f_03 §3.1)。

    倍率の 10 / 100 そのものが **期間の数** (「10年」「10回」「10か月」) か式の中の
    数なら取り違えとみなさない — 単利の ``1000000 * (1 + 0.12)`` (1.2% × 10 年) は
    正しい式である。会話のどこかの 10 (「頭金は10万円」「返済開始は10月」) では
    例外にしない (独立レビュー: C03 の 0.12 が見逃された)。
    """
    unexplained = _ungrounded_numbers(expression, query, context)
    if not unexplained:
        return ()
    text = f"{query or ''}\n{context or ''}"
    derived = percent_derived_values(text)
    if not derived:
        return ()
    expression_numbers = set(_NUMBER_LITERAL_RE.findall(expression or ""))
    slips: list[str] = []
    for literal in unexplained:
        try:
            n = float(literal)
        except ValueError:
            continue
        for factor in _SCALE_SLIP_FACTORS:
            scale = f"{(factor if factor > 1 else 1.0 / factor):g}"
            if scale in expression_numbers or re.search(
                rf"(?<![\d.]){scale}\s*{_PERIOD_UNIT_PATTERN}", text,
            ):
                continue
            if any(
                math.isclose(n * factor, value, rel_tol=1e-9, abs_tol=1e-12)
                for value in derived
            ):
                slips.append(literal)
                break
    return tuple(slips)


#: 価格に含まれた百分率 (「消費税10%込み」「10%税込」)。式の被演算子にならない。
_INCLUSIVE_PERCENT_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:%|％|パーセント)\s*税?込")


def _percent_usage_values(query: str) -> list[float]:
    """クエリの百分率・割合を式が「使った」とみなす値 (純粋関数)。

    率 r ごとに r / 1 ± r / 百分率 p / 100 ± p (``90/100`` の 90)。割合の語は
    さらに n / 10 / 10 ± n (``7/10``、``3/10``)、半分・半額は 0.5 / 2 / 50。
    内税の百分率 (:data:`_INCLUSIVE_PERCENT_RE`) は数えない。
    """
    text = _INCLUSIVE_PERCENT_RE.sub(" ", query or "")
    values: list[float] = []

    def add_rate(rate: float) -> None:
        pct = rate * 100.0
        values.extend((rate, 1.0 + rate, 1.0 - rate, pct, 100.0 - pct, 100.0 + pct))

    for pct in _PERCENT_LITERAL_RE.findall(text):
        add_rate(float(pct) / 100.0)
    for token in _WARI_RE.findall(text):
        tenths = _wari_tenths(token)
        if tenths is None:
            continue
        add_rate(tenths / 10.0)
        values.extend((tenths, 10.0, 10.0 - tenths, 10.0 + tenths))
    if _HALF_RE.search(text):
        values.extend((0.5, 2.0, 50.0))
    return values


def query_percents_unused(expression: str, query: str) -> bool:
    """クエリに百分率・割合があり、式がそのどれも使っていないか (純粋関数)。

    使ったとみなす値は :func:`_percent_usage_values` (率・倍率・百分率・100 - p・
    割合の分子と分母・半分の 2)。値で比べる (``0.90`` / ``.9`` の表記ゆれを吸収する)。
    実インシデント (2026-10-02 ライブ監査 D08#4): 「全員の残業を10%減らしたら合計は？」
    に分類器が窓内の ``25 + 31`` を返し、10% が式に無かった。
    """
    candidates = _percent_usage_values(query)
    if not candidates:
        return False
    for literal in _NUMBER_LITERAL_RE.findall(expression or ""):
        try:
            value = float(literal)
        except ValueError:
            continue
        if any(math.isclose(value, c, rel_tol=1e-9, abs_tol=1e-12) for c in candidates):
            return False
    return True


def _myriad_pairs(text: str) -> list[tuple[str, str]]:
    """万進表記の (係数, 展開値) 対。``2,850万`` → ``("2850", "28500000")``。"""
    out: list[tuple[str, str]] = []
    for num in iter_ja_numbers(text):
        for coefficient, multiplier in _myriad_terms(num):
            expanded = coefficient * multiplier
            if expanded == int(expanded) and coefficient == int(coefficient):
                out.append((str(int(coefficient)), str(int(expanded))))
    return out


#: 数値リテラル抽出用。小数と整数を拾う (単位や記号は含めない)。
_NUMERIC_LITERAL_RE = re.compile(r"\d+(?:\.\d+)?")
#: 全角数字を半角に寄せる変換表 (日本語入力のクエリ対策)。
_FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９．", "0123456789.")


def _numeric_literals(text: str) -> set[str]:
    """テキスト中の数値リテラル集合を返す (全角は半角へ正規化。純粋関数)。"""
    if not text:
        return set()
    normalized = text.translate(_FULLWIDTH_DIGITS)
    return {
        m.lstrip("0") or "0" for m in _NUMERIC_LITERAL_RE.findall(normalized)
    }
