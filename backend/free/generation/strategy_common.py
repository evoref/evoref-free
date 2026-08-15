"""Long-form 戦略の共通ユーティリティ

`strategy_recurrent.py` と `strategy_cogwriter.py` で AST 完全一致していた
3 関数とプロンプト定数を集約する

責務:
- 計画 JSON → :class:`GenerationPlan` 変換 (:func:`parse_plan`)
- JSON 解析失敗時の単一ユニットフォールバック (:func:`fallback_plan`)
- コードユニット生成用メッセージ構築 (:func:`build_code_unit_messages`)
- 共通プロンプトテンプレート定数

テキストユニットメッセージ構築 (:func:`build_text_unit_messages`) は
``include_long_term_summary`` フラグで recurrent 側の要約スロット
(``long_term_summary``) 差を吸収し、両戦略で共通化する。
"""

from __future__ import annotations

import logging
import math
import re
from graphlib import TopologicalSorter
from typing import TYPE_CHECKING

from backend.free.generation.models import (
    BREVITY_CHARS_PER_UNIT,
    CodeUnit,
    ContentType,
    GenerationPlan,
    LongFormMode,
    SectionPlan,
    chars_to_tokens,
    detect_brevity_cap,
    detect_line_limit_chars,
    extract_target_chars,
)
from backend.free.generation.spec_renderer import render_spec_for_prompt
from backend.free.llm.json_schemas import CodePlan, TextPlan
from backend.i18n_helper import prose_language_name

logger = logging.getLogger("backend.free.generation.strategy_common")

if TYPE_CHECKING:
    from backend.free.generation.rolling_context import RollingContext
    from backend.free.llm.aux_client import AuxClient
    from backend.free.llm.json_schemas import CodeSpec


# ── 共通プロンプトテンプレート ──

CODE_UNIT_SYSTEM = """\
あなたはPythonプログラマーです。以下の計画に従い、指定ユニットのコードを生成してください。
他ユニット (別ファイル・別関数/クラス) に属する関数・クラスを再定義しないこと。必要なら import して利用する。
{global_context}"""

CODE_UNIT_USER = """\
# 実装計画
ファイル: {file_path}
全ユニット: {unit_names}

# 他ユニットの公開定義 (これらは別途生成される。再定義せず、必要なら import して使う)
{sibling_interfaces}

# 生成済みコード構造
{skeleton}

# 直前の生成コード
{short_term}

# 現在のユニット
種別: {kind}
名前: {name}
仕様: {spec}
依存: {depends_on}

# 参考コード
{rag_context}

コードのみ出力してください:"""

TEXT_UNIT_SYSTEM = """\
以下の計画に従い、指定セクションの本文を生成してください。
- 見出し行（# や ## など）は出力しないでください。本文のみを出力してください。
- 文章は自然な段落で区切り、1文ごとに改行を入れないでください。
- 本文の内容そのものだけを出力してください。\
執筆意図・方針・プロセスの説明などメタ的な記述は一切含めないでください。
- 特に指定が無い限り、本文は{output_language}で書いてください\
（見出し・要点が別言語ならその言語に合わせる）。
- このセクションの目標文字数は約{unit_target_chars}文字です。必ずこの文字数に近い量を生成してください。\
短すぎる出力は不可です。
{global_context}"""

TEXT_UNIT_CONTINUATION_SYSTEM = """\
既存テキストを踏まえ、ユーザー指示に沿った内容を生成してください。以下のルールを厳守してください。
- 既存テキストの文体（語り口、文末表現、語彙、構造）を正確に維持してください。
- 生成する内容そのものだけを出力してください。\
メタ的な記述（「確認したところ」「〜する予定」「〜について記述する」等）は絶対に含めないでください。
- 見出し行（# や ## など）は出力しないでください。
- 文章は自然な段落で区切り、1文ごとに改行を入れないでください。
- 「直前テキスト末尾」から自然に繋がるように書いてください。
- このセクションの目標文字数は約{unit_target_chars}文字です。必ずこの文字数に近い量を生成してください。\
短すぎる出力は不可です。
{global_context}"""


# ── 計画パース ──

# 1 ユニットあたりの estimated_tokens 上限。LLM (json_schema grammar を強制
# しない補助タスクもある) が桁違いの値を返すと、orchestrator の
# _split_oversized_text_units が n_splits = ceil(estimated_tokens /
# unit_target_tokens) だけ同期ループしてイベントループを長時間ブロックする
# (実運用で発生: /api/status ポーリングまで停止する完全ハング)。実際の
# 妥当な最大単一セクション規模を大きく超えた値なので安全に切り詰める。
_ESTIMATED_TOKENS_MAX = 20_000

#: 分割された続きユニット (``SectionPlan.sub_index > 0``) の system プロンプトへ
#: 追記する継続指示。以前は key_points 側に入れていたため「本文に含めるべき要点」
#: として扱われ、冒頭宣言文の再掲を促していた (2026-07-25)。
_CONTINUATION_SYSTEM_NOTE = (
    "これは直前セクションの続きです。見出し・件名・宛名・挨拶などの"
    "冒頭部分は既に書かれているので繰り返さず、直前の文章の続きだけを書いてください。"
    "既に書いた文の再掲も禁止です。"
)


#: 「〜を描写する」のような執筆指示形の語尾。planner プロンプトはメタ的な
#: key_points を禁じているが (strategy_cogwriter の計画プロンプト参照)、
#: 強制はしておらず parse は受け取った文字列をそのまま保持していた。指示形の
#: まま unit プロンプトの「含めるべき要点」に載ると、弱い base モデルがそれを
#: 本文冒頭にそのまま書き写す (実インシデント 2026-07-27 ライブ検証:
#: 「春について 200 字で」の応答が「桜の花が満開になり…様子を描写する。
#: …情景を記述する。…特徴を伝える。」という指示文の羅列で始まった)。
#: parse 時に語尾を落として体言止めへ寄せ、写されても本文として読める形にする。
_KEY_POINT_INSTRUCTION_TAIL_RE = re.compile(
    r"(?:[をにへ](?:ついて|関して)?)?\s*"
    r"(?:描写|記述|説明|紹介|言及|提示|強調|表現|詳述|概説|解説|叙述)"
    r"(?:する|します|していく|していきます)?[。．.]?\s*$"
    r"|(?:[をにへ](?:ついて|関して)?)?\s*"
    r"(?:書く|書きます|述べる|述べます|伝える|伝えます"
    r"|まとめる|まとめます|触れる|触れます)[。．.]?\s*$",
)


def _declarative_key_points(raw: object) -> list[str]:
    """key_points から執筆指示の語尾を落として体言止めに揃える (純粋関数)。

    文字列以外の要素・空要素は捨てる。語尾を落として空になる項目
    (「説明する」だけ等) は情報が無いので元の文字列を残す。
    """
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if not text:
            continue
        stripped = _KEY_POINT_INSTRUCTION_TAIL_RE.sub("", text).strip(" 、,。．.")
        out.append(stripped if stripped else text)
    return out


def _to_int(value: object, default: int) -> int:
    """JSON 由来の数値フィールドを安全に int 化する。

    ``json_extract`` の戦略 4 (``json_repair`` フォールバック) は型強制を
    行わないため、LLM が ``"500"`` のように文字列で返した数値がそのまま
    dict に残ることがある。下流の算術 (sum / 比較 / 乗算) が ``int + str``
    で落ちないよう parse 時点で int に揃える。
    """
    if isinstance(value, bool):
        return default
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _to_estimated_tokens(value: object) -> int:
    """``estimated_tokens`` を安全な int に変換し、異常な巨大値を切り詰める。"""
    return min(_to_int(value, 500), _ESTIMATED_TOKENS_MAX)


def parse_plan(
    data: dict,
    content_type: ContentType,
    instruction: str = "",
    code_spec: CodeSpec | None = None,
) -> GenerationPlan:
    """JSON 辞書から :class:`GenerationPlan` を構築する。

    ユーザー指示の文字数指定 (``extract_target_chars``) が LLM が返した
    ``target_length`` より優先される。``target_length`` に対して
    ``estimated_tokens`` の合計が不足している場合は比例スケーリングで補正する。
    """
    units_raw = data.get("units", [])
    units: list[CodeUnit | SectionPlan] = []
    for u in units_raw:
        # Defense in depth: 上流 (json_extract) が壊れた応答から非 dict 要素を
        # units として渡してきた場合に AttributeError で全体クラッシュさせず、
        # 警告して当該要素をスキップする。
        if not isinstance(u, dict):
            logger.warning(
                "parse_plan: skipping non-dict unit element (type=%s, repr=%.80s)",
                type(u).__name__, repr(u),
            )
            continue
        if content_type == ContentType.CODE:
            units.append(CodeUnit(
                kind=u.get("kind", "function"),
                name=u.get("name", "unknown"),
                file_path=u.get("file_path", ""),
                spec=u.get("spec", ""),
                depends_on=u.get("depends_on", []),
                estimated_tokens=_to_estimated_tokens(u.get("estimated_tokens")),
            ))
        else:
            units.append(SectionPlan(
                heading=u.get("heading", ""),
                key_points=_declarative_key_points(u.get("key_points", [])),
                estimated_tokens=_to_estimated_tokens(u.get("estimated_tokens")),
                file_name=u.get("file_name") or None,  # SPLIT モード時のみ非 None
            ))

    # ユーザー指示の文字数指定を優先（LLM の計画値より信頼できる）
    user_target = extract_target_chars(instruction, default=0)
    plan_target = _to_int(data.get("target_length"), 0)
    target_length = user_target if user_target > 0 else plan_target

    # 数値指定が無く「簡潔に」「冗長にならない」「箇条書きで」等を要求された場合、
    # planner の target_length を上限で抑える。これが無いと LLM が要求と無関係に
    # 大きな target を選び、ユニット分割まで含めて冗長化する
    # (実測 2026-07-25: 「丁寧だが冗長にならない書き方で」→ 6 ユニット 7,192 字)。
    if user_target <= 0:
        brevity_cap = detect_brevity_cap(instruction)
        # 「3 行で」等の行数指定は文字数換算して同じ上限機構に載せる
        # (コードの「50 行くらい」は行数=規模の指定なので TEXT のみ)。
        if content_type != ContentType.CODE:
            line_cap = detect_line_limit_chars(instruction)
            if line_cap > 0:
                brevity_cap = (
                    line_cap if brevity_cap <= 0 else min(brevity_cap, line_cap)
                )
        if brevity_cap > 0 and (target_length <= 0 or target_length > brevity_cap):
            logger.info(
                "Brevity signal detected: capping target_length %d -> %d",
                target_length, brevity_cap,
            )
            target_length = brevity_cap
            # ユニットは最低 200 トークン (≒330 字) 生成されるため、目標に対して
            # ユニット数が多いとそれ自体が冗長化の原因になる。SPLIT モード
            # (file_name 付き = 1 ユニット 1 ファイル) は件数が意図的なので対象外。
            if units and not any(
                getattr(u, "file_name", None) for u in units
            ):
                unit_cap = max(
                    1, math.ceil(brevity_cap / BREVITY_CHARS_PER_UNIT),
                )
                if len(units) > unit_cap:
                    logger.info(
                        "Brevity signal: truncating %d units -> %d",
                        len(units), unit_cap,
                    )
                    units = units[:unit_cap]

    # target_length に基づき estimated_tokens を補正 (不足分の引き上げだけでなく
    # 超過分の引き下げも行う)。LLM が estimated_tokens を target_length と無関係に
    # 桁違いの値で返すことがあり (実運用で発生: 各ユニット 20000 超、目標
    # 2000文字=1200トークン超過に対し補正なしで 10000 文字超を出力していた)、
    # 上げ方向のみの補正だと超過ケースを一切是正できなかった。
    if target_length > 0 and units:
        target_tokens = chars_to_tokens(target_length)
        total_estimated = sum(u.estimated_tokens for u in units) or 1
        scale = target_tokens / total_estimated
        for u in units:
            u.estimated_tokens = max(int(u.estimated_tokens * scale), 200)

    return GenerationPlan(
        content_type=content_type,
        title=data.get("title", ""),
        target_length=target_length,
        global_context=data.get("global_context", ""),
        constraints=data.get("constraints", []),
        units=units,
        code_spec=code_spec,
        needs_clarification=bool(data.get("needs_clarification", False)),
        clarification_question=data.get("clarification_question", ""),
    )


def fallback_plan(
    instruction: str,
    content_type: ContentType,
    code_spec: CodeSpec | None = None,
) -> GenerationPlan:
    """JSON 解析失敗時の単一ユニット フォールバック計画を返す。"""
    target_length = extract_target_chars(instruction, default=1000)
    fallback_tokens = max(chars_to_tokens(target_length), 1000)

    if content_type == ContentType.CODE:
        unit: CodeUnit | SectionPlan = CodeUnit(
            kind="function",
            name="main",
            file_path="output.py",
            spec=instruction,
            depends_on=[],
            estimated_tokens=fallback_tokens,
        )
    else:
        unit = SectionPlan(
            heading="本文",
            key_points=[instruction],
            estimated_tokens=fallback_tokens,
        )
    return GenerationPlan(
        content_type=content_type,
        title="",
        target_length=target_length,
        global_context="",
        constraints=[],
        units=[unit],
        code_spec=code_spec,
    )


# ── プロンプト構築 ──

# 兄弟インタフェースブロックの最大文字数 (プロンプト肥大防止)。
_SIBLING_INTERFACES_MAX_CHARS = 1500


def _render_sibling_interfaces(plan: GenerationPlan, current: CodeUnit) -> str:
    """現在のユニットを除く全 :class:`CodeUnit` の公開定義を file 別に整形する。

    各ユニットの ``name`` / ``kind`` / 仕様冒頭 1 行を file_path でグルーピングして
    列挙し、「これらは別途生成されるので再定義せず import せよ」という契約を与える。
    肥大防止のため上限文字数で打ち切る。兄弟が無ければ ``(なし)`` を返す。
    """
    others = [
        u for u in plan.units
        if isinstance(u, CodeUnit) and u is not current
    ]
    if not others:
        return "(なし)"
    by_file: dict[str, list[str]] = {}
    for u in others:
        head = ""
        if u.spec:
            first = u.spec.strip().splitlines()
            head = (": " + first[0][:80]) if first else ""
        by_file.setdefault(u.file_path or "(同一ファイル)", []).append(
            f"  - {u.name} ({u.kind}){head}"
        )
    block = "\n".join(
        f"{fp}:\n" + "\n".join(lines) for fp, lines in by_file.items()
    )
    if len(block) > _SIBLING_INTERFACES_MAX_CHARS:
        block = block[:_SIBLING_INTERFACES_MAX_CHARS] + "\n…"
    return block


def build_code_unit_messages(
    unit: CodeUnit,
    rolling: RollingContext,
) -> list[dict]:
    """コードユニット生成用のメッセージ列を構築する。

    両戦略 (Recurrent / CogWriter) で完全に共通の純粋関数。`self` には依存しない。
    """
    budget = rolling.budget
    plan = rolling.plan

    unit_names = ", ".join(
        u.name for u in plan.units if isinstance(u, CodeUnit)
    )

    # 兄弟ユニットの公開シグネチャを明示注入し、API 不整合・同名再定義を抑制する
    # (spec 契約に加えた defense-in-depth)。現在のユニット自身は除く。
    sibling_interfaces = _render_sibling_interfaces(plan, unit)

    skeleton_text = ""
    if rolling.skeleton:
        skeleton_text = budget.fit_content(
            "skeleton_or_summary",
            rolling.skeleton.to_prompt(budget.skeleton_or_summary),
        )

    short_term = budget.fit_content("short_term", rolling.short_term)

    # 共有設計仕様 (契約) を全ユニットに注入する。これにより小ブロックが
    # 同一のモジュール名・データモデルのフィールド/型・公開シグネチャ・
    # エントリポイント・プロトコルに準拠し、ファイル横断の不整合を防ぐ。
    system_text = CODE_UNIT_SYSTEM.format(global_context=plan.global_context)
    spec_text = render_spec_for_prompt(plan.code_spec)
    if spec_text:
        system_text = (
            f"{system_text}\n\n# 設計仕様 (契約 — 厳密準拠せよ)\n{spec_text}\n\n"
            "上記のモジュール名・データモデルのフィールド名/型・公開シグネチャ・"
            "エントリポイント・プロトコルを変更しない。存在しないモジュールから"
            "import しない。"
        )
    # モジュール構成図 (Phase 2、config code_flowchart_enabled=True 時のみ合成)。
    # 従来は SPEC.md への添付のみでコード生成には一切渡っていなかった。
    if plan.code_flowchart:
        system_text = (
            f"{system_text}\n\n# モジュール構成図 (Mermaid)\n"
            f"```mermaid\n{plan.code_flowchart}\n```"
        )
    system = budget.fit_content("system_prompt", system_text)
    user = CODE_UNIT_USER.format(
        file_path=unit.file_path,
        unit_names=unit_names,
        sibling_interfaces=sibling_interfaces,
        skeleton=skeleton_text or "(なし)",
        short_term=short_term or "(なし)",
        kind=unit.kind,
        name=unit.name,
        spec=budget.fit_content("unit_spec", unit.spec),
        depends_on=", ".join(unit.depends_on) or "(なし)",
        rag_context=budget.fit_content("rag_chunks", rolling.unit_rag) or "(なし)",
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ── 既存テキスト抜粋 (continuation / expand) ──

def excerpt_continuation_content(existing: str) -> str:
    """継続(追記)モード用に既存テキストを抜粋する (冒頭 500 + 末尾 800 char)。

    1500 char 以下はそのまま返す。LLM コンテキストと計画精度のバランスを
    考慮した固定値。
    """
    if len(existing) > 1500:
        return existing[:500] + "\n\n[...中略...]\n\n" + existing[-800:]
    return existing


def excerpt_for_expand(existing_content: str) -> str:
    """EXPAND/SPLIT モード用に既存テキストを抜粋する (冒頭 2000 + 末尾 2000 char)。

    継続モード (500+800=1300) より広く取り、planner が機能境界を識別できる
    解像度を確保する。plan ``max_tokens=1024`` 制約とのバランスで 4000 char 上限。
    """
    if len(existing_content) <= 4000:
        return existing_content
    return (
        existing_content[:2000]
        + "\n\n[...中略...]\n\n"
        + existing_content[-2000:]
    )


# ── 計画の最大ユニット数 / 生成順序 ──

def resolve_max_units(config: dict, long_form_mode: LongFormMode) -> int:
    """長文生成の最大ユニット数。EXPAND/SPLIT では機能ごと節分割のため下限 8 を保証。"""
    max_units = config.get("max_units", 20)
    if long_form_mode in (LongFormMode.EXPAND, LongFormMode.SPLIT):
        max_units = max(max_units, 8)
    return max_units


def resolve_generation_order(units: list[CodeUnit]) -> list[CodeUnit]:
    """依存関係に基づくトポロジカルソート (Recurrent / CogWriter 共通)。"""
    graph = {u.name: set(u.depends_on) for u in units}
    sorter = TopologicalSorter(graph)
    try:
        order = list(sorter.static_order())
    except Exception:
        logger.warning("Topological sort failed, using original order")
        return units
    unit_map = {u.name: u for u in units}
    return [unit_map[name] for name in order if name in unit_map]


# エントリポイント相当とみなすユニット名 (これらが entry file に既にあれば二重生成しない)。
_ENTRY_UNIT_NAMES: frozenset[str] = frozenset({
    "main", "__main__", "__entry__", "run", "app", "cli", "entry", "entrypoint",
})


def _entry_leaf(entry_module: str) -> str:
    """エントリモジュール表記から末尾モジュール名 (stem) を取り出す。

    ``main.py`` / ``tetris`` / ``game_of_life.main`` のいずれも末尾セグメントを返す
    (``game_of_life.main`` → ``main``)。
    """
    ep = entry_module[:-3] if entry_module.endswith(".py") else entry_module
    return ep.replace("\\", "/").replace("/", ".").rsplit(".", 1)[-1]


def _path_stem(path: str) -> str:
    """ファイルパスから拡張子・ディレクトリを除いた stem を返す。"""
    stem = (path or "").replace("\\", "/").rsplit("/", 1)[-1]
    return stem[:-3] if stem.endswith(".py") else stem


def _resolve_entry_file(
    entry_module: str, code_units: list[CodeUnit], spec: "CodeSpec",
) -> str:
    """合成するエントリユニットを書き込む file_path を決める。

    既存ユニット → ``spec.modules`` の順で entry stem に一致する path を優先し、
    単一ファイル構成ならそのファイル、無ければ ``<leaf>.py`` を生成する。
    """
    leaf = _entry_leaf(entry_module)
    for u in code_units:
        if _path_stem(u.file_path) == leaf:
            return u.file_path
    for m in spec.modules:
        if _path_stem(m.path) == leaf:
            return m.path
    paths = {u.file_path for u in code_units if u.file_path}
    if len(paths) == 1:
        return next(iter(paths))
    return f"{leaf}.py"


def _ensure_entry_unit(plan: GenerationPlan) -> None:
    """``spec.entry_point`` があるのに対応ユニットが計画に無い場合、合成して追加する。

    planner プロンプトは「entry_point を実装する unit を必ず含める」と指示するが、
    weak local model はこれを無視して ``main()`` / ``if __name__ == '__main__'`` ガードを
    生成しないことがある (起動不能なプログラム = 検証で「__main__ ガードが無い」/
    「エントリポイントが存在しない」エラーになる)。本処理はその指示を計画段階で
    決定的に enforce する。合成ユニットは全ユニットに依存させ最後に生成する
    (rolling context で他ユニットの skeleton を参照できる)。
    """
    spec = plan.code_spec
    if spec is None or not spec.entry_point or not spec.entry_point.module:
        return
    code_units = [u for u in plan.units if isinstance(u, CodeUnit)]
    if not code_units:
        return
    entry_file = _resolve_entry_file(spec.entry_point.module, code_units, spec)
    # 既にエントリ相当の unit が同一ファイルにあれば二重生成しない。
    for u in code_units:
        if u.file_path == entry_file and (
            u.name.lower() in _ENTRY_UNIT_NAMES or "__main__" in u.spec
        ):
            return
    invocation = (spec.entry_point.invocation or "").strip()
    unit_spec = (
        "プログラムのエントリポイントを実装する。これまでのユニットを統合して動作させる "
        'main() 関数と、`if __name__ == "__main__":` ガード (その中で main を起動する) を'
        "必ず含めること。"
    )
    if invocation:
        unit_spec += f" 起動方法の契約: {invocation}"
    plan.units.append(CodeUnit(
        kind="function",
        name="__entry__",
        file_path=entry_file,
        spec=unit_spec,
        depends_on=[u.name for u in code_units],
        estimated_tokens=500,
    ))
    logger.info(
        "Injected synthetic entry-point unit for module '%s' (file=%s)",
        spec.entry_point.module, entry_file,
    )


def finalize_plan_units(
    plan: GenerationPlan,
    max_units: int,
    content_type: ContentType,
) -> None:
    """計画ユニットを ``max_units`` に切り詰め、コードなら依存順ソートする (in-place)。

    両戦略の create_plan 末尾で共通の後処理。コードはエントリポイント契約がある場合、
    切り詰め後に合成エントリユニットを保証する (truncation で落とさない)。
    """
    if len(plan.units) > max_units:
        logger.warning(
            "Plan has %d units, truncating to %d", len(plan.units), max_units,
        )
        plan.units = plan.units[:max_units]
    if content_type == ContentType.CODE:
        _ensure_entry_unit(plan)
        code_units = [u for u in plan.units if isinstance(u, CodeUnit)]
        plan.units = resolve_generation_order(code_units)


# ── 計画 JSON 生成 ──

# プラン JSON の出力トークン上限。1024 では多モジュール計画が頻繁に切断され
# (実測: 1 日 9 回 plan_truncated) 末尾ユニットが黙って欠落していた。1536 は
# code_spec で実績のある値で long_form_planning の timeout 内に収まる。切断時のみ
# 下記で再プランする。
_PLAN_MAX_TOKENS = 1536
# 切断時の 1 回限り再プラン。出力を増やし、iGPU の decode 長増を見越して timeout も
# 明示的に延長する (purpose 既定 90s のままだと ReadTimeout を誘発するため)。
_PLAN_RETRY_MAX_TOKENS = 3072
_PLAN_RETRY_TIMEOUT_SEC = 150.0


async def generate_plan_json(
    aux_client: AuxClient | None,
    prompt: str,
    content_type: ContentType,
    *,
    telemetry: dict | None = None,
) -> dict:
    """補助タスクで計画 JSON を生成する。

    ``content_type`` に応じた schema (CodePlan / TextPlan) を明示選択し、
    ``aux_client is None`` (degraded) / 例外時は空 dict を返して呼出側の
    単一ユニットフォールバックに委ねる。

    出力が ``max_tokens`` で切断された場合 (``telemetry['truncated']``)、ユニット
    欠落を防ぐため **より大きい出力 + 延長 timeout で 1 回だけ再プラン**する。再プラン
    が非切断、または取得ユニットが増えた場合のみ採用する。

    ``telemetry`` を渡すと最終結果の ``truncated`` / ``replanned`` 等が書き戻される。
    """
    if aux_client is None:
        logger.info(
            "create_plan: aux client is not wired; "
            "falling back to single-unit plan",
        )
        return {}
    plan_schema = CodePlan if content_type == ContentType.CODE else TextPlan

    async def _gen(max_tokens: int, timeout: float | None, tel: dict) -> dict:
        return await aux_client.generate_json(
            prompt,
            max_tokens=max_tokens,
            temperature=0.3,
            purpose="long_form_planning",
            list_key="units",
            response_schema=plan_schema,
            timeout=timeout,
            telemetry=tel,
        )

    try:
        tel1: dict = {}
        data = await _gen(_PLAN_MAX_TOKENS, None, tel1)
    except Exception as e:
        logger.warning("Plan generation failed: %s", e)
        return {}

    result, result_tel = data, tel1
    if tel1.get("truncated"):
        logger.warning(
            "Plan JSON truncated at max_tokens=%d; replanning at %d (timeout=%.0fs)",
            _PLAN_MAX_TOKENS, _PLAN_RETRY_MAX_TOKENS, _PLAN_RETRY_TIMEOUT_SEC,
        )
        try:
            tel2: dict = {}
            data2 = await _gen(
                _PLAN_RETRY_MAX_TOKENS, _PLAN_RETRY_TIMEOUT_SEC, tel2,
            )
            n1 = len((data or {}).get("units", []) or [])
            n2 = len((data2 or {}).get("units", []) or [])
            if data2 and (not tel2.get("truncated") or n2 > n1):
                tel2["replanned"] = True
                result, result_tel = data2, tel2
        except Exception as e:
            logger.warning("Plan replan failed: %s", e)

    if telemetry is not None:
        telemetry.clear()
        telemetry.update(result_tel)
    return result


# ── テキストユニットメッセージ構築 ──

def build_text_unit_messages(
    unit: SectionPlan,
    rolling: RollingContext,
    text_unit_user_template: str,
    *,
    include_long_term_summary: bool = False,
) -> list[dict]:
    """テキストユニット生成用のメッセージ列を構築する (両戦略共通スケルトン)。

    ``text_unit_user_template`` は各戦略が保持する ``_TEXT_UNIT_USER`` 定数を渡す
    (Recurrent 側は ``{long_term_summary}`` プレースホルダを含む)。
    ``include_long_term_summary=True`` のとき要約スロットを埋める (Recurrent)。
    """
    budget = rolling.budget
    plan = rolling.plan

    # 分割で生まれた続きユニット (sub_index > 0) は親と同じ heading を持つため
    # 重複除去する。順序は保つ。
    section_headings = ", ".join(
        dict.fromkeys(
            u.heading
            for u in plan.units
            if isinstance(u, SectionPlan) and not u.sub_index
        )
    )
    short_term = budget.fit_content("short_term", rolling.short_term)

    total_estimated = sum(
        u.estimated_tokens for u in plan.units if isinstance(u, SectionPlan)
    ) or 1
    unit_ratio = unit.estimated_tokens / total_estimated
    unit_target_chars = max(int(plan.target_length * unit_ratio), 200)

    system_template = (
        TEXT_UNIT_CONTINUATION_SYSTEM
        if rolling.has_existing_context
        else TEXT_UNIT_SYSTEM
    )
    system_text = system_template.format(
        global_context=plan.global_context,
        unit_target_chars=unit_target_chars,
        output_language=prose_language_name(),
    )
    # 分割で生まれた続きユニットには再掲禁止を **system 側** で与える。
    # key_points (=「本文に含めるべき要点」) に混ぜると制約が本文の一部として
    # 提示され、見出し・冒頭の宣言文ごと再生成される (2026-07-25 実測: 同一文が
    # 34 回反復し「件名：」が 4 回出るメールになった)。
    if getattr(unit, "sub_index", 0):
        system_text += "\n" + _CONTINUATION_SYSTEM_NOTE
    system = budget.fit_content(
        # output_language は新規生成テンプレートのみが持つ (継続テンプレートは
        # 既存テキストの言語追従が正のため指示しない。余剰 kwarg は無害)
        "system_prompt", system_text,
    )

    fmt_kwargs = {
        "title": plan.title,
        "section_headings": section_headings,
        "short_term": short_term or "(なし)",
        "heading": unit.heading,
        "key_points": ", ".join(unit.key_points),
        "rag_context": budget.fit_content("rag_chunks", rolling.unit_rag) or "(なし)",
    }
    if include_long_term_summary:
        fmt_kwargs["long_term_summary"] = budget.fit_content(
            "skeleton_or_summary",
            rolling.long_term_summary,
        ) or "(なし)"

    user = text_unit_user_template.format(**fmt_kwargs)
    # 追記モード: short_term ラベルを差し替え
    if rolling.has_existing_context:
        user = user.replace(
            "# 直前セクション末尾",
            "# 直前テキスト末尾（この直後に自然に続く文章を書いてください）",
        )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
