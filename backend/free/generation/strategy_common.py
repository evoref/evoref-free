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
from backend.free.generation.outline_seed import ParsedOutline
from backend.free.generation.spec_renderer import render_spec_for_prompt
from backend.free.generation.text_skeleton import TextSkeleton
from backend.free.llm.json_schemas import CodePlan, TextPlan, TextPlanSeeded
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

#: 利用者が示していない実務上の固有値を本文・計画で補わせない規則 (本文と計画で共有)。
#:
#: 「生成すべき具体的な内容を書く」「短すぎる出力は不可」という圧だけがあると、
#: 案内文のような実務文書でモデルは日時・会場・接続先を **それらしく埋める**。
#: 2026-09-17 監査: 「新人向けセキュリティ研修の案内文」に「来月10日（水）」
#: (実在しない曜日)・「本社5階の第3会議室」・Zoom が書かれた。チャット経路の
#: system には同じ禁止があるが、長文のユニット生成は別の system を使うため
#: 届いていなかった。物語などの創作の中の出来事は対象外。
UNSPECIFIED_FACTS_RULE = (
    "ユーザー指示・参考情報に無い実務上の固有値（日時・曜日・場所・連絡先・URL・"
    "金額・担当者名など）は創作しないでください。案内・通知・招待・メール等の"
    "実務文書でそれらが必要な箇所は「〇月〇日（〇）」「【会場】」のような空欄で"
    "示してください。物語などの創作文書の中の出来事はこの限りではありません。"
)

#: unit 依存 (unit_target_chars / 継続指示) を含まない system テンプレート。
#: system は run 中 byte 固定でなければならない (接頭辞 KV の再利用、f_08 §2.2 /
#: §8 禁則 10) — unit ごとに変わる文言は :func:`build_text_unit_messages` が
#: user 側 (:data:`_UNIT_LENGTH_INSTRUCTION_HEADER` 経由) に置く。
TEXT_UNIT_SYSTEM = """\
以下の計画に従い、指定セクションの本文を生成してください。
- 見出し行（# や ## など）は出力しないでください。本文のみを出力してください。
- 文章は自然な段落で区切り、1文ごとに改行を入れないでください。
- 本文の内容そのものだけを出力してください。\
執筆意図・方針・プロセスの説明などメタ的な記述は一切含めないでください。
- 特に指定が無い限り、本文は{output_language}で書いてください\
（見出し・要点が別言語ならその言語に合わせる）。
- """ + UNSPECIFIED_FACTS_RULE + """
{global_context}"""

TEXT_UNIT_CONTINUATION_SYSTEM = """\
既存テキストを踏まえ、ユーザー指示に沿った内容を生成してください。以下のルールを厳守してください。
- 既存テキストの文体（語り口、文末表現、語彙、構造）を正確に維持してください。
- 生成する内容そのものだけを出力してください。\
メタ的な記述（「確認したところ」「〜する予定」「〜について記述する」等）は絶対に含めないでください。
- 見出し行（# や ## など）は出力しないでください。
- 文章は自然な段落で区切り、1文ごとに改行を入れないでください。
- 「直前テキスト末尾」から自然に繋がるように書いてください。
- """ + UNSPECIFIED_FACTS_RULE + """
{global_context}"""


#: unit の system に入れる依頼の原文の上限 (文字)。添付の本文などが混ざった長い
#: 依頼で system を膨らませない。
_PLAN_REQUEST_MAX_CHARS = 600


def plan_context_block(plan: GenerationPlan) -> str:
    """unit の system に入れる計画の文脈 (``global_context`` + ``constraints`` + 依頼の原文)。

    ``constraints`` は計画が「守るべき条件」を退避する欄 (計画プロンプトが、依頼に
    含まれる形式・件数・対象読者などを constraints に retain せよと指示している)
    なのに、2026-09-20 まで **本文生成のどこにも渡っていなかった** — 「3 点、
    箇条書きで」の依頼に箇条書きが 15 個出た (実機)。

    ただし constraints に何が写るかは計画 (LLM) 次第で、同じ依頼でも件数が写る回と
    写らない回がある (constraints を渡した後も 5 個 → 13 個と振れた)。条件を語彙で
    拾う規則を足すのではなく、**依頼の原文をそのまま見せる**。どれも計画単位で
    不変なので system に置いても run 中の byte は変わらない (f_08 §2.2)。
    """
    parts = [plan.global_context or ""]
    constraints = [c.strip() for c in (plan.constraints or []) if c and c.strip()]
    if constraints:
        lines = "\n".join(f"- {c}" for c in constraints)
        parts.append(f"# 制約 (必ず守る)\n{lines}")
    request = (getattr(plan, "instruction", "") or "").strip()
    if request:
        parts.append(
            "# 依頼の原文 (件数・形式・対象読者などの条件を守るために参照する。"
            "保存先やファイル形式の指示は配信の話なので本文には書かない)\n"
            + request[:_PLAN_REQUEST_MAX_CHARS]
        )
    return "\n\n".join(p for p in parts if p).strip()


# ── 計画パース ──

# 1 ユニットあたりの estimated_tokens 上限。LLM (json_schema grammar を強制
# しない補助タスクもある) が桁違いの値を返すと、orchestrator の
# _split_oversized_text_units が n_splits = ceil(estimated_tokens /
# unit_target_tokens) だけ同期ループしてイベントループを長時間ブロックする
# (実運用で発生: /api/status ポーリングまで停止する完全ハング)。実際の
# 妥当な最大単一セクション規模を大きく超えた値なので安全に切り詰める。
_ESTIMATED_TOKENS_MAX = 20_000

#: 分割された続きユニット (``SectionPlan.sub_index > 0``) の user プロンプトへ
#: 追記する継続指示。以前は key_points 側に入れていたため「本文に含めるべき要点」
#: として扱われ、冒頭宣言文の再掲を促していた (2026-07-25)。system は run 中
#: byte 固定でなければならないため、以前 system に足していたこの注記も
#: unit_target_chars と同じ理由で user 側へ移した (2026-09-18、f_08 §2.2)。
_CONTINUATION_SYSTEM_NOTE = (
    "これは直前セクションの続きです。見出し・件名・宛名・挨拶などの"
    "冒頭部分は既に書かれているので繰り返さず、直前の文章の続きだけを書いてください。"
    "既に書いた文の再掲も禁止です。"
)

#: 両テンプレート (``_TEXT_UNIT_USER`` / recurrent 側の同名定数) が末尾に持つ
#: 共通の最終指示。unit 依存の分量・継続指示をこの直前へ挿入する目印にする。
_FINAL_INSTRUCTION_MARKER = (
    "本文のみを出力してください（見出し行・メタ解説・執筆意図の説明は不要）:"
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


# ── 構成テンプレートによる計画の seed (f_08 §3.1.1) ──

_TEXT_PLAN_SEED_PROMPT = """\
以下の構成 (見出しと順序は固定、変更できません) に沿って文書を書きます。
文書全体の背景 (global_context) と目標文字数 (target_length)、各セクションへの
追加要点 (unit_notes、無ければ空配列) だけを JSON で返してください。

# ユーザー指示
{instruction}

# 構成 (見出しは変更不可、0 始まりの番号は unit_notes.index に対応)
{outline}
"""

#: seed 計画の目標文字数の既定値 (unit 1 個あたり)。ユーザー指定も補助タスクの
#: 提案も無いときの最終フォールバック。
_SEEDED_DEFAULT_CHARS_PER_UNIT = 500


async def generate_seeded_plan_json(
    aux_client: AuxClient | None,
    instruction: str,
    outline: ParsedOutline,
    *,
    telemetry: dict | None = None,
) -> dict:
    """縮小 schema (``TextPlanSeeded``) で補助タスクを呼ぶ (f_08 §3.1.1)。

    ``aux_client is None`` (degraded) / 例外時は空 dict を返し、呼出側
    (:func:`build_seeded_plan`) は outline だけで計画を組む。
    """
    if aux_client is None:
        return {}
    numbered = "\n".join(
        f"{i}. {u.heading}" + (f" — {', '.join(u.key_points)}" if u.key_points else "")
        for i, u in enumerate(outline.units)
    )
    prompt = _TEXT_PLAN_SEED_PROMPT.format(instruction=instruction, outline=numbered)
    try:
        return await aux_client.generate_json(
            prompt,
            max_tokens=768,
            temperature=0.3,
            purpose="long_form_planning",
            response_schema=TextPlanSeeded,
            telemetry=telemetry,
        )
    except Exception as e:
        logger.warning("Seeded plan generation failed: %s", e)
        return {}


def build_seeded_plan(
    outline: ParsedOutline,
    instruction: str,
    aux_data: dict,
) -> GenerationPlan:
    """outline + 補助タスク結果 (縮小 schema) から計画を組む (純粋関数)。

    ``units`` の heading と順序は outline がそのまま決める。補助タスクの
    返り値は heading を持たないため、見出しの追加・削除・並べ替えは
    起こり得ない。``aux_data`` が空 (degraded / 失敗) なら outline と
    ユーザー指示だけで組む。
    """
    aux_ok = bool(aux_data)

    user_target = extract_target_chars(instruction, default=0)
    brevity_cap = 0
    if user_target <= 0:
        brevity_cap = detect_brevity_cap(instruction)
        line_cap = detect_line_limit_chars(instruction)
        if line_cap > 0:
            brevity_cap = line_cap if brevity_cap <= 0 else min(brevity_cap, line_cap)

    aux_target = _to_int(aux_data.get("target_length"), 0) if aux_ok else 0
    if user_target > 0:
        target_length = user_target
    elif brevity_cap > 0:
        target_length = brevity_cap
    elif aux_target > 0:
        target_length = aux_target
    else:
        target_length = max(
            len(outline.units) * _SEEDED_DEFAULT_CHARS_PER_UNIT,
            _SEEDED_DEFAULT_CHARS_PER_UNIT,
        )

    global_context = str(aux_data.get("global_context") or "") if aux_ok else ""
    if not global_context:
        global_context = instruction

    notes_by_index: dict[int, list[str]] = {}
    if aux_ok:
        for raw_note in aux_data.get("unit_notes") or []:
            if not isinstance(raw_note, dict):
                continue
            idx = _to_int(raw_note.get("index"), -1)
            if idx < 0 or idx >= len(outline.units):
                logger.warning(
                    "Seeded plan: unit_notes index %r out of range (units=%d); "
                    "dropped", raw_note.get("index"), len(outline.units),
                )
                continue
            extra = _declarative_key_points(raw_note.get("extra_key_points"))
            if extra:
                notes_by_index.setdefault(idx, []).extend(extra)

    total_units = len(outline.units) or 1
    per_unit_tokens = max(int(chars_to_tokens(target_length) / total_units), 200)

    units: list[SectionPlan] = [
        SectionPlan(
            heading=ou.heading,
            key_points=list(ou.key_points) + notes_by_index.get(idx, []),
            estimated_tokens=per_unit_tokens,
            verbatim=ou.verbatim,
        )
        for idx, ou in enumerate(outline.units)
    ]

    return GenerationPlan(
        content_type=ContentType.TEXT,
        title=outline.title,
        target_length=target_length,
        global_context=global_context,
        constraints=[],
        units=units,
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
    # CODE の system (CODE_UNIT_SYSTEM + spec 契約 + 構成図) は予算外
    # (f_08 §3.3.1、2026-09-18)。system は run 中 byte 不変で接頭辞 KV に
    # 乗るため窓を圧迫しない。以前は system_prompt スロットで head 切りし
    # 「BINDING CONTRACT」宣言を欠落させていた。
    system = system_text
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
    # ProductionBrief (f_08 §2.2): user の先頭に置く (system は brief を含まず
    # run 中 byte 固定のまま)。
    if rolling.brief:
        user = f"{rolling.brief}\n\n{user}"

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
    """EXPAND/SPLIT モード用に既存テキストを抜粋する (4000 char 上限、見出し単位で均等)。

    継続モード (500+800=1300) より広く取り、planner が機能境界を識別できる
    解像度を確保する。plan ``max_tokens=1024`` 制約とのバランスで 4000 char 上限。
    以前は冒頭 2000 + 末尾 2000 字で、中盤の節 (データモデル / 入力検証) が
    ちょうど落ちた (2026-09-21 ライブ監査 K01、f_08 §2.2)。
    """
    return condense_design_doc(existing_content, 4000)


_MARKDOWN_HEADING_RE = re.compile(r"^#{1,6}\s", re.MULTILINE)
_CONDENSED_MARK = "\n…(中略)…"


def condense_design_doc(text: str, max_chars: int) -> str:
    """設計書を ``max_chars`` 以内へ **見出し単位で均等に** 縮める (純粋関数)。

    見出し行はすべて残し、本文の予算を節の数で割る。短い節が使い切らなかった
    分は長い節へ回す (水位の均等化)。冒頭 + 末尾だけを残す切り方だと中盤の
    節が丸ごと消える (f_10 §2 / f_08 §2.2)。見出しが無い文書は先頭から切る。
    """
    text = text or ""
    if len(text) <= max_chars:
        return text
    starts = [m.start() for m in _MARKDOWN_HEADING_RE.finditer(text)]
    if not starts:
        return text[:max_chars]
    if starts[0] != 0:
        starts.insert(0, 0)
    chunks = [text[a:b] for a, b in zip(starts, [*starts[1:], len(text)])]
    heads: list[str] = []
    bodies: list[str] = []
    for chunk in chunks:
        if _MARKDOWN_HEADING_RE.match(chunk):
            head, _, body = chunk.partition("\n")
            heads.append(head + "\n")
            bodies.append(body)
        else:
            heads.append("")
            bodies.append(chunk)
    budget = max_chars - sum(len(h) for h in heads) - len(chunks) * len(_CONDENSED_MARK)
    if budget <= 0:
        return "".join(heads)[:max_chars]
    alloc = [0] * len(bodies)
    remaining = len(bodies)
    for i in sorted(range(len(bodies)), key=lambda k: len(bodies[k])):
        share = budget // remaining
        alloc[i] = min(len(bodies[i]), share)
        budget -= alloc[i]
        remaining -= 1
    out = []
    for head, body, n in zip(heads, bodies, alloc):
        kept = body if n >= len(body) else body[:n].rstrip() + _CONDENSED_MARK + "\n\n"
        out.append(head + kept)
    return "".join(out)


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
    # unit_target_chars / 継続指示は unit ごとに値が変わるため system には含め
    # ない (system は run 中 byte 固定 — f_08 §2.2 / §8 禁則 10)。user 側
    # (_FINAL_INSTRUCTION_MARKER の直前) へ置く。
    system_text = system_template.format(
        global_context=plan_context_block(plan),
        output_language=prose_language_name(),
    )
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

    # 分量指示 (unit ごとに変わる) + 分割続きユニットの再掲禁止。以前は
    # system 側に足していたため unit ごとに system の bytes が変わり、接頭辞
    # KV キャッシュが unit 毎に 0 になっていた (2026-09-18 監査、f_08 §2.2)。
    unit_instructions = (
        f"このセクションの目標文字数は約{unit_target_chars}文字です。"
        "必ずこの文字数に近い量を生成してください。短すぎる出力は不可です。"
    )
    if (getattr(plan, "instruction", "") or "").strip():
        # 分量の指示は user の最後に置く最も強い指示なので、依頼が「3 点」と言って
        # いても、モデルは文字数に届かせるために項目を増やす (実機 2026-09-21:
        # 同じ依頼で 3 個の回と 13 個の回に割れた)。優先順位と、両立のさせ方を明示する。
        unit_instructions += (
            "\nただし、依頼の原文に件数や形式の指定 (「3 点」「箇条書きで」「表で」等) が"
            "あれば、文字数の目安よりもそちらを優先してください。項目を増やして文字数を"
            "稼がず、足りない分は各項目の説明を厚くしてください。"
        )
    if getattr(unit, "sub_index", 0):
        unit_instructions += "\n" + _CONTINUATION_SYSTEM_NOTE
    # 構成テンプレートの引用ブロック由来の固定文 (f_08 §3.1.1)。ここでは
    # 「含めてほしい」と指示するだけで、逐語一致の担保は生成後の決定論検査
    # (orchestrator.ensure_verbatim_sentence) が行う (再生成しない)。
    if getattr(unit, "verbatim", None):
        unit_instructions += (
            "\n次の一文をそのまま逐語で（一字一句変えずに）本文に含めてください:\n"
            f"「{unit.verbatim}」"
        )
    if _FINAL_INSTRUCTION_MARKER in user:
        # 目印はテンプレート末尾の 1 つだけが本物。参考情報の本文に同じ文が
        # 紛れていても、そこへは差し込まない。
        head, marker, tail = user.rpartition(_FINAL_INSTRUCTION_MARKER)
        user = f"{head}# 分量・継続の指示\n{unit_instructions}\n\n{marker}{tail}"
    elif user:
        user = f"{user}\n\n# 分量・継続の指示\n{unit_instructions}"
    else:
        user = unit_instructions

    # ProductionBrief (brief) + TextSkeleton (状態) を user の先頭へ置く
    # (f_08 §2.2 の配置順: brief → 状態 → 窓 → 参考資料 → 指示)。
    prefix_parts: list[str] = []
    if rolling.brief:
        prefix_parts.append(rolling.brief)
    skeleton = rolling.text_skeleton
    if isinstance(skeleton, TextSkeleton):
        skeleton_text = budget.fit_content(
            "skeleton_or_summary", skeleton.to_prompt(budget.skeleton_or_summary),
        )
        if skeleton_text:
            prefix_parts.append(f"# 文書の状態\n{skeleton_text}")
    if prefix_parts:
        user = "\n\n".join([*prefix_parts, user])

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
