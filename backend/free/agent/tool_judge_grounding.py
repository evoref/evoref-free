"""合成式の数値グラウンディング (純粋関数)

「式に現れる数値が対話から辿れるか」だけを扱う。ツールの戻り値は確かめた事実
として base に最優先で渡るため、捏造された定数が混ざった式は *正しく計算された
嘘* になる。その検出をここに集約する。
"""

from __future__ import annotations

import re

from backend.free.core.intent_vocab import NUMBER_LITERAL_RE

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
_UNIT_SYSTEM_CONSTANTS = frozenset({
    "7", "10", "12", "24", "52", "60", "100", "365", "366",
    "1000", "1440", "3600", "86400",
    "0.1", "0.01", "0.001",
    # SI 接頭辞 (10 の冪) — 1000 は上にある
    "1000000", "1000000000", "1000000000000",
    # 2 進接頭辞 (2 の冪): KiB / MiB / GiB / TiB
    "1024", "1048576", "1073741824", "1099511627776",
})


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
    """
    known = _known_numbers(query) | _known_numbers(context)
    known.update(_UNIT_SYSTEM_CONSTANTS)
    seen: list[str] = []
    for n in _NUMBER_LITERAL_RE.findall(expression):
        if n not in known and n not in seen:
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
_WEEKDAY_ORDER = "月火水木金土日"
_WEEKDAY_RANGE_RE = re.compile(
    r"([月火水木金土日])\s*(?:曜日?)?\s*(?:〜|～|-|–|から)\s*([月火水木金土日])\s*(?:曜日?)?",
)


def _known_numbers(text: str) -> set[str]:
    """``text`` に「書かれている」とみなせる数値リテラルを集める (純粋関数)。

    素の数字に加えて 2 種類を同一視する。どちらも **対話に現れた表記から決定論で
    導ける**もので、モデルが知識から持ち出した定数ではない:

    - 桁区切り: ``2,660`` → ``2660``。数える側 (式) は区切りを打たないため、
      正規化しないと自分が直前に提示した金額を「知らない数値」と判定してしまう
    - パーセント: ``10%`` → ``0.1`` / ``1.1`` / ``1.10``。税率・割引率の計算で
      式に現れる倍率は、クエリ中の百分率から一意に決まる
    """
    known = set(_NUMBER_LITERAL_RE.findall(text))
    for grouped in _GROUPED_NUMBER_RE.findall(text):
        known.add(grouped.replace(",", ""))
    for pct in _PERCENT_LITERAL_RE.findall(text):
        try:
            rate = float(pct) / 100.0
        except ValueError:
            continue
        # 「10%」からは 0.1 (率) と 1.1 (加算後の倍率) が導ける。式側の表記ゆれ
        # (1.1 / 1.10) を吸収するため両方を登録する。
        for value in (rate, 1.0 + rate):
            known.add(f"{value:g}")
            known.add(f"{value:.2f}")
    known.update(_duration_derived_numbers(text))
    return known


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
