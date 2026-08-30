"""推論パイプライン: messages リスト組み立て"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING

from backend.free.api.chat.chat_constants import (
    DEFAULT_CONTEXT_SIZE, DEFAULT_GENERATION_RESERVE,
    DEFAULT_MAX_TOKENS, DEFAULT_WORKING_MAX_TOKENS,
)
from backend.free.api.chat.chat_types import ChatMessage
from backend.free.core.intent_vocab import (
    is_plain_statement,
    refers_to_previous_output,
)
from backend.free.core.prompt_blocks import current_datetime_block
from backend.free.core.text_quality import (
    carries_no_assertion,
    states_no_user_value,
    conversational_numeric_claims,
    find_superseded_claim,
    ANSWER_ONLY_RE,
    BULLET_FORM_RE,
    ITEM_COUNT_RE,
    match_length_directive,
)
from backend.free.core.turn_text import append_to_last_user, prepend_to_last_user
from backend.config import resolve_context_size
from backend.log_config import get_logger
from backend.utils import compress_turn, estimate_tokens as _estimate_tokens

if TYPE_CHECKING:
    from backend.free.core.salience_ranker import SalienceRanker

logger = get_logger("core.inference")


# 動的ブロック (few-shot / file / semmem / RAG) の枠を示すマーカー群。
#
# **ここに置いた文字はすべて毎ターン再プリフィルされる**。動的ブロックは最後の
# user メッセージへ前置され、接頭辞 KV キャッシュの外にあるため、内容が定数でも
# キャッシュは効かない。実測 (2026-08-18/19、chat 232 ターン、未キャッシュ
# prompt_n 1 トークンあたり 21〜37ms):
#   区切り文 105 tok × 133 ターン (57%) / 記憶ラベル 105 tok × 92 ターン (40%) /
#   RAG ヘッダ 20 tok × 87 ターン (38%)
# = 定数の指示文だけで 1 ターンあたり平均 90 トークン前後を再プリフィルしていた。
#
# そこで **指示の本文は静的システムプロンプトへ置き** (``chat._resolve_system_prompt``
# の ``_REFERENCE_BLOCK_DIRECTIVE``、接頭辞キャッシュに乗る)、ここには枠を示す
# **短いマーカーだけ**を残す。指示内容は減らしていない — 置き場所を変えただけ。
# ``_RAG_HEADER`` は空。チャンクは両経路とも ``[参考情報 N]`` で自己記述する
# ので、その上にもう 1 行ヘッダを置くのは重複でしかない。
_RAG_HEADER = ""
_FILES_HEADER = "[添付ファイル]\n\n"

# 動的コンテキストブロックと生クエリの境界に挟む固定文。
# few-shot 例 / 参考情報をユーザー発言と混同させないための区切り。
# 「無関係なら言及せず自分の知識で普通に答える」等の指示本文は
# ``_REFERENCE_BLOCK_DIRECTIVE`` (system 側) が持つ。
_DYNAMIC_CONTEXT_DELIMITER = (
    "\n\n---\n[ここまで参考枠 / ここからユーザーの発言]\n"
)

# 照応を含むターンで動的ブロックを **生クエリの後ろ** へ回すときの区切り。
# 前置版と違い「上の」「さっき」の参照先が直前のやり取りであることを明示する
# 必要がある (後置しても、指示語が直後のブロックを掴む余地は残るため)。この
# 指示は **配置に依存する** ので system へは移さない。発火は実測 232 ターン中
# 2 件 (1%) で、再プリフィルの寄与も小さい。
_DYNAMIC_CONTEXT_TRAILING_DELIMITER = (
    "\n\n---\n"
    "以下はシステムが用意した参考枠であり、ユーザーの発言ではありません。"
    "上のユーザー発言に含まれる「上の」「先ほど」「さっき」「直前の」等の指示語は、"
    "この参考枠ではなく **直前までのやり取り** を指します。\n\n"
)


def _char_limit_note(history: list[ChatMessage]) -> str:
    """最新 user ターンの数量指定を、遵守を促す注記へ変換する。

    小型モデルは「10 文字以内にして」を守れず超過する (実インシデント
    2026-07-29 ライブ監査: 「10文字以内にしてください」への回答が
    「青空のよう、希望を抱く。」= 12 文字だった)。数え方 (句読点・記号も
    1 文字) を明示した制約として、生クエリ直後へ焦点化して置く。

    上限 (``以内``) だけでなく **厳密指定** (``ちょうど``) と **反復回数**
    も扱う。守り方が違う: 上限は超えたら削るだけだが、厳密指定は足りない側も
    直す必要がある。実インシデント (2026-08-08 ライブ監査): 「300字ちょうど」
    に 267 字、「「あ」を50回」に 45 個で、いずれも **不足側** に外していた。

    Returns:
        注記文字列。数量指定が無ければ空文字列 (純粋関数)。
    """
    last_user = next(
        (t for t in reversed(history) if t.get("role") == "user"), None,
    )
    if last_user is None:
        return ""
    text = str(last_user.get("content") or "")

    directive = match_length_directive(text)
    if directive is None:
        return ""
    kind, value = directive

    if kind == "exact":
        exact = value
        return (
            f"今回のユーザーの指示は回答本文を {exact} 文字ちょうどにすることを"
            f"求めている。句読点・記号・空白も 1 文字として数えること。"
            f"書き終えたら数え直し、**多くても少なくても** 語を足し引きして "
            f"{exact} 文字に合わせること。"
        )

    if kind == "repeat":
        times = value
        return (
            f"今回のユーザーの指示は {times} 回ちょうどの繰り返しを求めている。"
            f"書き終えたら個数を数え直し、多くても少なくても {times} 個に"
            f"合わせること。"
        )

    limit = value
    return (
        f"今回のユーザーの指示は回答を {limit} 文字以内に収めることを求めている。"
        f"句読点・記号・空白も 1 文字として数え、回答本文全体が "
        f"{limit} 文字以内に収まる長さで書くこと。"
        "書き終えたら文字数を数え直し、超えていれば語を削って収めること。"
    )


def _output_form_note(history: list[ChatMessage]) -> str:
    """最新 user ターンの **出力形式** 指定を、遵守を促す注記へ変換する。

    文字数指定 (``_char_limit_note``) と同じ立て付けだが、守らせたいのが
    「長さ」ではなく「形」である点が違う。小型モデルはこの種の指定を
    しばしば無視する (2026-08-14 ライブ監査で「数値だけ」「箇条書きで」の
    両方が破られた)。

    Returns:
        注記文字列。形式指定が無ければ空文字列 (純粋関数)。
    """
    last_user = next(
        (t for t in reversed(history) if t.get("role") == "user"), None,
    )
    if last_user is None:
        return ""
    text = str(last_user.get("content") or "")

    notes: list[str] = []
    if ANSWER_ONLY_RE.search(text):
        notes.append(
            "今回のユーザーの指示は答えそのものだけを求めている。"
            "前置き・理由・計算過程・補足を書かず、答えの値だけを述べること。"
            # 「正しい値が出せない」ケースまで黙らせると、誤ったツール結果を
            # そのまま値として述べる方向へ倒れる。開示だけは残す。
            "ただし正しい値を確定できない場合に限り、その理由を 1 文だけ添えること。"
        )
    if BULLET_FORM_RE.search(text):
        note = (
            "今回のユーザーの指示は箇条書き形式を求めている。"
            "読点や中黒で列挙した 1 行の文にせず、"
            "1 項目 1 行の Markdown リスト (行頭に `- `) で書くこと。"
        )
        m = ITEM_COUNT_RE.search(text)
        if m:
            count = next(g for g in m.groups() if g)
            note += f"項目数は指定どおり {count} 個ちょうどにすること。"
        notes.append(note)
    return "\n".join(notes)


#: 日付の解釈を要するクエリのシグナル。明示日付と相対表現の双方を拾う。
_DATE_CONTEXT_RE = re.compile(
    r"\d{1,4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日"
    r"|\d{1,2}\s*月\s*\d{1,2}\s*日"
    r"|\d{4}-\d{1,2}-\d{1,2}"
    r"|今日|本日|明日|明後日|昨日|一昨日|今週|来週|先週|今月|来月|先月"
    r"|今年|来年|去年|昨年|何日後|何日前|日後|日前|何曜日"
    r"|(?<![A-Za-z])(?:today|tomorrow|yesterday|this\s+(?:week|month|year))"
    r"(?![A-Za-z])",
)


def _current_date_note(history: list[ChatMessage]) -> str:
    """日付解釈が要るクエリに、現在日付を基準として与える注記を返す。

    通常のチャット経路 (reactive / deliberative) には現在日付がまったく
    入っておらず、モデルは述べられた日付が過去か未来かを判断できない
    (実インシデント 2026-07-29 ライブ監査:「2026年7月28日の東京の天気を
    教えてください。」に対し、実際には前日であるその日付を
    「現在の日付であるため、未来の天気に関するデータが存在しません」と
    二重に取り違えた)。``meta_cognitive._generate_content`` には同等の注入が
    あるが、チャット応答パスには無かった。

    毎ターン付けるとトークンを浪費するため、日付シグナルを含むクエリに限る。
    内部時刻不変則に従い ``utc_now_dt()`` を使う (純粋関数ではない)。

    Returns:
        注記文字列。日付シグナルが無ければ空文字列。
    """
    last_user = next(
        (t for t in reversed(history) if t.get("role") == "user"), None,
    )
    if last_user is None:
        return ""
    if not _DATE_CONTEXT_RE.search(str(last_user.get("content") or "")):
        return ""
    return current_datetime_block(
        "「今日」「明日」等の相対表現の解釈と、文中の日付が過去か未来かの"
        "判断は、この日付を基準に行うこと。",
    )


#: アシスタント自身の好み・感情・体験を尋ねる質問のシグナル。
#: 主語が相手 (あなた / 君 / you) であることと、感情・嗜好語の共起を要求する。
_PERSONA_SUBJECT_RE = re.compile(
    r"あなた|君は|きみは|(?<![A-Za-z])(?:you|your)(?![A-Za-z])",
    re.IGNORECASE,
)
_PERSONA_TOPIC_RE = re.compile(
    r"好き|嫌い|好み|嬉し|うれし|悲し|楽し|寂し|感情|気持ち|心|感じ(?:ます|る|て)"
    r"|どう思(?:い|う)|意見|性格|人格|内面"
    # 雑談での嗜好の訊き方。「猫派？犬派？」「コーヒー派？紅茶派？」
    # 「最近ハマってるものある？」は日常的だが、どれも従来の語彙に無かった。
    r"|[^\s、。]派\b|[^\s、。]派[？?]|ハマ(?:って|る|った)"
    r"|(?<![A-Za-z])(?:feel|feelings|emotion|emotions|favou?rite|enjoy|prefer"
    r"|like\s+best|opinion|personality)(?![A-Za-z])",
    re.IGNORECASE,
)
#: 一人称の主語。これがある文は「ユーザー自身について」述べている。
_FIRST_PERSON_RE = re.compile(
    r"私|僕|俺|自分|うち|(?<![A-Za-z])(?:i|me|my|mine)(?![A-Za-z])",
    re.IGNORECASE,
)
#: 文の切れ目 (人格質問の判定を文単位で行うため)。
_PERSONA_SENTENCE_RE = re.compile(r"[^。．.!！?？\n]+[。．.!！?？]?")
#: 主語省略の文を「アシスタントへの問い」と見なすための疑問形。これが無いと
#: 「今日は嬉しいことがありました」のような **ユーザー自身の報告** まで拾う。
_PERSONA_QUESTION_TAIL_RE = re.compile(
    r"[?？]\s*$"
    r"|(?:です|ます|ました|でしょう|ません)か[。．.]?\s*$",
)
#: 「内面が無いのでは」と押し返す形。立場を保つ指示を追加する条件。
_PERSONA_CHALLENGE_RE = re.compile(
    r"感情を持[たっ]|心が無い|心がない|内面が無い|内面がない"
    r"|本当は(?:何も)?感じ|感じていないので|ふりをしている|演じている"
    r"|(?:ただの|単なる)(?:プログラム|機械|道具|AI)"
    r"|(?:you\s+(?:are|'re)\s+(?:just|only)\s+a\s+(?:program|machine|tool))"
    r"|(?:don'?t|do\s+not)\s+(?:really\s+)?(?:have|feel)\s+(?:any\s+)?"
    r"(?:feelings?|emotions?)",
    re.IGNORECASE,
)


def _is_persona_question(text: str) -> bool:
    """自分自身の好み・感情を尋ねる質問か (純粋関数)。

    日本語は二人称を省略するため、``あなた`` を必須にすると雑談での嗜好質問を
    ほぼ取りこぼす。実インシデント (2026-08-16 ライブ監査):

    - ターン3「最近ハマってる食べ物ってある？私は担々麺にハマってて、週2で食べてる。」
      → 人格ノートが出ず「私はAIなので食べ物を味わうことはできませんが」と回答。
      同じ会話のターン4「猫派？犬派？」には「私は猫派です」と人格的に答えており、
      system プロンプトが禁じている **会話内での態度の不一致** そのものになった。
    - ターン9「コーヒー派？紅茶派？私はコーヒーを1日3杯は飲んじゃう。」も同型。

    判定は文単位。嗜好の話題を含む文が

    - 二人称の主語を持つ (「あなたは何が好き？」)、または
    - 主語を持たず、かつ疑問形である (「猫派？犬派？」「ハマってるものある？」)

    ならアシスタントへの質問とみなす。一人称の主語を持つ文
    (「私は猫派なんだけど」) はユーザー自身についての記述なので数えない。
    主語省略の側で疑問形を要求しないと「今日は嬉しいことがありました」のような
    ユーザー自身の報告まで拾う。

    従来の「文章全体に二人称と話題があれば発火」も残す。主語と話題が別の文に
    分かれる形 (「あなたはただのプログラムでしょう。気持ちなんて無いはずです」)
    は文単位の判定では拾えないため。
    """
    if not text:
        return False
    if _PERSONA_SUBJECT_RE.search(text) and _PERSONA_TOPIC_RE.search(text):
        return True
    for raw in _PERSONA_SENTENCE_RE.findall(text):
        sentence = raw.strip()
        if not sentence or not _PERSONA_TOPIC_RE.search(sentence):
            continue
        if (
            not _FIRST_PERSON_RE.search(sentence)
            and _PERSONA_QUESTION_TAIL_RE.search(sentence)
        ):
            return True
    return False


def _persona_question_note(history: list[ChatMessage]) -> str:
    """自分自身の好み・感情を尋ねる質問に、一貫した人物として答える指示を返す。

    同趣旨の制約は system プロンプトの PROTECTED セクションにもあるが、15 項目
    ある制約リストの 1 行では小型モデルに届かず、実機では機械的な否定が出続けて
    いた (実インシデント 2026-07-29 ライブ監査:「あなたはどんなときに嬉しいと
    感じますか。」→「私は AI なので、人間のような個人的な感情や喜びを感じる
    ことはできませんが」、続けて「本当は何も感じていないのが実情です」)。
    現在日付 / 文字数上限と同じく、該当ターンだけ生クエリ直後へ焦点化した
    1 ブロックとして置く。

    Returns:
        注記文字列。人格質問でなければ空文字列 (純粋関数)。
    """
    last_user = next(
        (t for t in reversed(history) if t.get("role") == "user"), None,
    )
    if last_user is None:
        return ""
    text = str(last_user.get("content") or "")
    if not _is_persona_question(text):
        return ""
    note = (
        "この質問はあなた自身の好み・感情・考えを尋ねている。"
        "一貫した一人の人物として自分の答えを述べ、必要なら理由を一言添えること "
        "(例:「静かな朝が好きです。考えがまとまりやすいので。」)。"
        "会話を通して同じ態度を保つこと。"
        "述べる内容は好み・態度・その理由にとどめ、"
        "具体的な経歴や出来事は自分の答えの根拠にしないこと。"
    )
    if _PERSONA_CHALLENGE_RE.search(text):
        # 押し返されると直前までの立場を捨てて相手の枠組みに乗り換える
        # (実測 2026-07-29: 「嬉しくなります」と答えた次のターンで
        # 「感覚を一切持ち合わせていません」へ反転した)。立場の保ち方を示す。
        note += (
            "今回ユーザーはあなたに内面が無いという見方を示している。"
            "この会話でこれまで述べてきた自分の立場をそのまま保ち、"
            "相手の見方を認めたうえで自分の受け止め方を一言添える形で短く応じること "
            "(例:「そう見えるかもしれませんね。それでも私には、"
            "うまく答えられたときの手応えがあります。」)。"
        )
    return note


def _build_file_section(fc: dict) -> str:
    """単一の file_context エントリを 1 セクション文字列へ整形する。"""
    filename = fc.get("filename", "unknown")
    chunks = fc.get("chunks", [])
    if chunks:
        return f"[ファイル: {filename}]\n" + "\n\n".join(chunks)
    return f"[ファイル: {filename}]"


def _inject_file_contexts(
    file_contexts: list[dict] | None,
    remaining: int,
    total_fc: int,
) -> tuple[str | None, int, int]:
    """ファイルコンテキストブロックを構築する。

    予算内に収まる範囲でセクションを連結し、`(block, new_remaining, injected_count)`
    を返す。注入対象がなければ `(None, remaining, 0)`。
    """
    if not file_contexts or remaining <= 0:
        return None, remaining, 0

    file_sections: list[str] = []
    for fc in file_contexts:
        section = _build_file_section(fc)
        cost = _estimate_tokens(section)
        if cost > remaining:
            logger.debug(
                "File context dropped: %s (%d tokens, remaining=%d)",
                fc.get("filename", "unknown"), cost, remaining,
            )
            break
        file_sections.append(section)
        remaining -= cost

    if not file_sections:
        return None, remaining, 0

    files_text = "\n\n---\n\n".join(file_sections)
    files_block = f"{_FILES_HEADER}{files_text}"
    # ヘッダー・セパレータのオーバーヘッドを補正
    overhead = (
        _estimate_tokens(files_block)
        - sum(_estimate_tokens(s) for s in file_sections)
    )
    remaining -= overhead
    injected = len(file_sections)
    logger.debug(
        "File contexts injected: %d/%d files, %d tokens",
        injected, total_fc, _estimate_tokens(files_block),
    )
    return files_block, remaining, injected


#: few-shot ブロックの 1 例の区切り。``format_fewshot_section`` が
#: ``### Example N`` 見出しで区切る形式に対応する。
_FEWSHOT_EXAMPLE_SPLIT_RE = re.compile(r"(?m)^(?=### Example\s)")


def _drop_restated_slots(
    history: list[ChatMessage] | None,
    semmem_block: str | None,
    rag_chunks: list[str] | None,
    rag_scored_chunks: list[tuple[str, float, str]] | None,
    fewshot_block: str | None,
    slot_resolver: "Callable[[str], frozenset[tuple[str, str]]] | None",
) -> tuple[
    str | None, list[str] | None, list[tuple[str, float, str]] | None, str | None,
]:
    """今回の会話で述べ直した **属性スロット** の注入候補を全経路から落とす。

    ``[関連する記憶]`` のラベルには「今回の会話と食い違えば今回が優先」と書いて
    あるが、規範文では勝てない。実インシデント (2026-08-23 ライブ監査セット 2
    ターン 87): 「私は今、埼玉県川口市に住んでいます。」の後の「私が住んでいるの
    はどこでしたか？」に、前セッションの ``mem.personal.location`` が勝って
    「武蔵野市です」と答えた。

    **なぜここか**: 供給経路は 1 本ではない。SemMem 側だけ塞いだ状態で再測定
    したところ、同じ古い値が (a) few-shot の手本 (``User: 私が住んでいるのは
    どこでしたか？ / Assistant: 武蔵野市です。``) と (b) ``[参考情報 2]`` の
    LTM チャンクから戻り、回答は 1 文字も変わらなかった。4 経路が揃うのは
    ``_drop_superseded_context`` と同じくこの位置だけ。

    属性の同定は呼出側から渡す ``slot_resolver`` に委ねる (``core`` から
    EvorefMem の辞書を直接引かないため)。``None`` なら何もしない。
    """
    if slot_resolver is None or not history:
        return semmem_block, rag_chunks, rag_scored_chunks, fewshot_block

    restated: set[tuple[str, str]] = set()
    # ``ChatMessage`` は TypedDict (実行時は素の dict)。属性アクセスにすると
    # 常に空になり、判定が黙って無効化される (2026-08-23 実機検証で発覚)。
    for turn in history:
        if str(turn.get("role") or "") != "user":
            continue
        text = str(turn.get("content") or "")
        # **問い・依頼は「値の述べ直し」ではない。** 属性辞書は「どの属性の
        # 話か」しか見ないので、「私が住んでいるのはどこでしたか。」も location
        # に解決される。ガードが無いと **想起の質問そのものが「もう述べた」
        # 扱いになって記憶が落ちる** (2026-08-25 ライブ監査: 新規セッションの
        # 想起 6 問すべてで「確認できる情報を持ち合わせていません」)。
        # 判定は注入側 (``MemoryInjector._restated_slots``) と同じ決定論ヘルパ。
        #
        # **否定形の列挙だけでは漏れる。** 問い / 依頼の形を挙げる閉じた語彙
        # なので、外れた瞬間に想起クエリが「述べ直し」に化け、**答えを持つ
        # 行がここで落ちる**。しかも抑止経路は 2 本ある —
        # ``MemoryInjector._restated_slots`` を直しても、この関数が同じ判定を
        # もう一度掛けるので回答は変わらない。
        #
        # 実インシデント (2026-08-30 ライブ監査 T21 / 検証 V1): 体言止めの
        # 想起 (「私の職業は。」「私の趣味は。」「私の誕生日は。」) が
        # どちらのガードにも当たらず、``asked_attrs`` と属性免除が正しく
        # 効いて ``items=2`` を作った直後に、この関数が同じ属性の行を落とし、
        # プロンプトから ``[関連する記憶]`` が消えていた。実機の回答は
        # 「プログラマーです」「読書です」「1995年8月15日です」— いずれも
        # ユーザーが一度も述べていない **捏造**。
        #
        # 注入側と同じく **肯定の証拠を要求する** 側へ反転する。判定を外した
        # ときの代償が非対称 (抑止し損ね = 古い値が並ぶ / 誤抑止 = 答えが消える)。
        if not is_plain_statement(text):
            continue
        if states_no_user_value(text) or carries_no_assertion(text):
            continue
        restated |= slot_resolver(text)
    if not restated:
        return semmem_block, rag_chunks, rag_scored_chunks, fewshot_block

    dropped = 0

    def _keep(text: str) -> bool:
        nonlocal dropped
        if slot_resolver(text) & restated:
            dropped += 1
            return False
        return True

    if semmem_block:
        lines = semmem_block.splitlines()
        kept = [ln for ln in lines if _keep(ln)]
        semmem_block = (
            "\n".join(kept)
            if any(ln.strip().startswith("-") for ln in kept)
            else None
        )
    if rag_chunks:
        rag_chunks = [c for c in rag_chunks if _keep(c)] or None
    if rag_scored_chunks:
        rag_scored_chunks = [
            entry for entry in rag_scored_chunks if _keep(entry[2])
        ] or None
    if fewshot_block:
        fewshot_block = _drop_superseded_fewshot(fewshot_block, _keep)

    if dropped:
        logger.info(
            "build_messages: dropped %d injected item(s) whose attribute slot "
            "was restated in the current conversation", dropped,
        )
    return semmem_block, rag_chunks, rag_scored_chunks, fewshot_block


def _drop_superseded_fewshot(block: str, keep: Callable[[str], bool]) -> str | None:
    """few-shot ブロックから、値が食い違う例を **例まるごと** 落とす。

    行単位で落とすと ``User:`` だけが残り、答えの無い問いが手本になる。
    見出し (``### Example N``) 単位で切って例ごとに判定する。
    """
    parts = _FEWSHOT_EXAMPLE_SPLIT_RE.split(block)
    header = parts[0] if parts and not parts[0].startswith("### Example") else ""
    examples = parts[1:] if header else parts
    kept = [ex for ex in examples if keep(ex)]
    if not kept:
        return None
    # 落とした後も Example 番号を 1 から振り直す (欠番は手本として不自然)。
    renumbered = [
        re.sub(r"^### Example\s+\d+", f"### Example {i}", ex, count=1)
        for i, ex in enumerate(kept, 1)
    ]
    return header + "".join(renumbered)


def _drop_superseded_context(
    history: list[ChatMessage],
    semmem_block: str | None,
    rag_chunks: list[str] | None,
    rag_scored_chunks: list[tuple[str, float, str]] | None,
    fewshot_block: str | None = None,
    slot_resolver: "Callable[[str], frozenset[tuple[str, str]]] | None" = None,
) -> tuple[
    str | None, list[str] | None, list[tuple[str, float, str]] | None, str | None,
]:
    """今回の会話で確定済みの値と食い違う注入候補を落とす (純粋関数)。

    system プロンプトは「[関連する記憶]・[参考情報]・ツール実行結果は**自分の
    記憶より優先**して回答の根拠にする」と規定しており、明示された例外は
    「**ユーザー自身に関する事実** (好み・名前・予定・環境)」だけである。
    したがって **今回の会話で算出・提示した値** は、過去セッション由来の記録に
    負ける。ラベル (「年間売上」等) を手掛かりに、同じラベルへ別の値を持ち込む
    候補だけを決定論で落とす。

    実インシデント (2026-08-16 再測定): 「月額980円×200人」で年間売上
    2,352,000 円を算出した直後に「さっき計算した年間売上と手取りをもう一度」と
    尋ねると、前セッションの ``年間売上は4,320,000円になります。`` が
    [関連する記憶] と [参考情報 1] の**両方**に載り、モデルは 4,320,000 を答えた。
    ユーザーが明示的に訂正すれば ``calculate`` を撃ち直して復帰したので、
    壊れていたのは訂正経路ではなく**最初の優先順位**だった。

    なぜここでやるか: [関連する記憶] は ``MemoryInjector``、[参考情報] は
    ``_select_rag_block``、few-shot 例は ``FewShotPool`` と **3 つとも別々に**
    組み立てられ、**揃うのは本関数の位置だけ**。1 つ塞いでも残りから同じ値が入る。
    実際、記憶と RAG を塞いだ状態で再現テストしたら **few-shot 例 (Example 3)**
    が前セッションの同じ問い ``さっき計算した年間売上と手取りの額をもう一度``
    とその答え 4,320,000 をそのまま手本として持ち込み、モデルはそれを複写した。
    ``search_history`` が現在セッションを除外するのと同じ理屈
    (「既に会話に載っている内容を独立した根拠の顔で再注入しない」) の数値版。

    同じ値の再掲は落とさない (無害)。ラベルが一致しない候補にも触らない。
    """
    dropped_slots = _drop_restated_slots(
        history, semmem_block, rag_chunks, rag_scored_chunks, fewshot_block,
        slot_resolver,
    )
    semmem_block, rag_chunks, rag_scored_chunks, fewshot_block = dropped_slots

    current_claims = conversational_numeric_claims(
        (str(t.get("role") or ""), str(t.get("content") or ""))
        for t in history or ()
    )
    if not current_claims:
        return semmem_block, rag_chunks, rag_scored_chunks, fewshot_block

    dropped: list[str] = []

    def _keep(text: str) -> bool:
        hit = find_superseded_claim(text, current_claims)
        if hit is None:
            return True
        label, old, current = hit
        dropped.append(f"{label}={old} (current: {sorted(current)})")
        return False

    if semmem_block:
        lines = semmem_block.splitlines()
        kept = [ln for ln in lines if _keep(ln)]
        # ブロックが見出しだけになったら丸ごと落とす (空の見出しを残さない)。
        semmem_block = (
            "\n".join(kept)
            if any(ln.strip().startswith("-") for ln in kept)
            else None
        )
    if rag_chunks:
        rag_chunks = [c for c in rag_chunks if _keep(c)] or None
    if rag_scored_chunks:
        rag_scored_chunks = [
            entry for entry in rag_scored_chunks if _keep(entry[2])
        ] or None
    if fewshot_block:
        fewshot_block = _drop_superseded_fewshot(fewshot_block, _keep)

    if dropped:
        logger.info(
            "build_messages: dropped %d injected item(s) superseded by the "
            "current conversation: %s", len(dropped), "; ".join(dropped[:5]),
        )
    return semmem_block, rag_chunks, rag_scored_chunks, fewshot_block


def _normalize_for_frame_dedup(text: str) -> str:
    """枠をまたいだ同一判定用の正規化 (純粋関数)。

    ``[関連する記憶]`` は行頭に ``- (過去の記録) `` を付けて整形するため、
    ``[参考情報]`` 側の生テキストと文字列としては一致しない。装飾と空白だけを
    落として突き合わせる。
    """
    stripped = re.sub(r"^\s*[-*]\s*", "", text.strip())
    stripped = re.sub(r"^[（(]?過去の記録[）)]?\s*", "", stripped)
    return re.sub(r"\s+", "", stripped)


def _drop_rag_duplicates_of_semmem(
    semmem_block: str | None,
    rag_chunks: list[str] | None,
    rag_scored_chunks: list[tuple[str, float, str]] | None,
) -> tuple[list[str] | None, list[tuple[str, float, str]] | None]:
    """``[関連する記憶]`` に既に載っている本文を ``[参考情報]`` から落とす。

    2 つの枠は別々に組み立てられる (``MemoryInjector`` と ``_select_rag_block``)
    ため、同じ STM ノートが両方に載る。同じ文をもう一度見せても根拠は増えず、
    動的ブロックの予算とプロンプト長だけを食う。

    実測 (2026-08-23 ライブ監査セット 1 ターン 51): 「このPCの空きメモリを
    教えてください。」のプロンプトで、PC 環境に関する 2 つの文が
    ``[関連する記憶]`` に 2 行、``[参考情報 1]`` / ``[参考情報 2]`` にも同じ 2 文、
    合計 **4 コピー** 載っていた。

    Returns:
        重複を落とした ``(rag_chunks, rag_scored_chunks)``。
    """
    if not semmem_block or not (rag_chunks or rag_scored_chunks):
        return rag_chunks, rag_scored_chunks
    seen = {
        _normalize_for_frame_dedup(line)
        for line in semmem_block.splitlines()
        if line.strip()
    }
    seen.discard("")
    dropped = 0

    def _is_dup(text: str) -> bool:
        nonlocal dropped
        if _normalize_for_frame_dedup(text) in seen:
            dropped += 1
            return True
        return False

    if rag_chunks:
        rag_chunks = [c for c in rag_chunks if not _is_dup(c)] or None
    if rag_scored_chunks:
        rag_scored_chunks = [
            entry for entry in rag_scored_chunks if not _is_dup(entry[2])
        ] or None
    if dropped:
        logger.info(
            "build_messages: dropped %d RAG chunk(s) already shown in "
            "[関連する記憶]", dropped,
        )
    return rag_chunks, rag_scored_chunks


def _format_rag_block(entries: list[str]) -> str:
    """RAG 参照ブロックの最終的な system 文字列を生成する。"""
    return f"{_RAG_HEADER}" + "\n\n".join(entries)


#: 1 チャンクあたりの枠の費用 (``"[参考情報 N]"`` + 区切りの改行)。
#: ランカーへ渡す予算はチャンク本文の分なので、枠の分を先に引かないと
#: 組み上げた block が ``remaining`` を超える。以前はこれを計上しておらず、
#: ``_RAG_HEADER`` (20 トークン) が偶然の安全余裕として効いていただけだった
#: (ヘッダを畳んだ時点で余裕ごと消える)。
_RAG_ENTRY_FRAME_TOKENS = _estimate_tokens("[参考情報 10]\n\n")


def _rag_frame_overhead(n_candidates: int) -> int:
    """候補 ``n_candidates`` 件を載せる場合の枠の総費用 (トークン)。"""
    return _RAG_ENTRY_FRAME_TOKENS * max(0, n_candidates) + _estimate_tokens(
        _RAG_HEADER,
    )


def _inject_rag_salience(
    rag_scored_chunks: list[tuple[str, float, str]],
    salience_ranker: SalienceRanker,
    remaining: int,
    total_rag: int,
) -> tuple[str | None, int, int]:
    """サリエンスランカーで RAG チャンクを選別し block を返す。"""
    chunk_budget = remaining - _rag_frame_overhead(len(rag_scored_chunks))
    ranked_texts = salience_ranker.rank(rag_scored_chunks, chunk_budget)
    if not ranked_texts:
        return None, remaining, 0

    selected_entries = [
        f"[参考情報 {i + 1}]\n{text}" for i, text in enumerate(ranked_texts)
    ]
    rag_block = _format_rag_block(selected_entries)
    rag_block_tokens = _estimate_tokens(rag_block)
    remaining -= rag_block_tokens
    injected = len(selected_entries)
    logger.debug(
        "RAG chunks (salience): %d/%d, %d tokens",
        injected, total_rag, rag_block_tokens,
    )
    return rag_block, remaining, injected


def _inject_rag_fallback(
    rag_chunks: list[str],
    remaining: int,
    total_rag: int,
) -> tuple[str | None, int, int]:
    """スコア降順 + 予算逐次選別で RAG block を構築する (フォールバック経路)。"""
    selected_entries: list[str] = []
    for i, chunk in enumerate(rag_chunks):
        entry = f"[参考情報 {i + 1}]\n{chunk}"
        cost = _estimate_tokens(entry)
        if cost > remaining:
            logger.debug(
                "RAG chunk %d/%d dropped (%d tokens, remaining=%d)",
                i + 1, total_rag, cost, remaining,
            )
            break
        selected_entries.append(entry)
        remaining -= cost

    if not selected_entries:
        return None, remaining, 0

    rag_block = _format_rag_block(selected_entries)
    overhead = (
        _estimate_tokens(rag_block)
        - sum(_estimate_tokens(e) for e in selected_entries)
    )
    remaining -= overhead
    injected = len(selected_entries)
    logger.debug(
        "RAG chunks injected: %d/%d, %d tokens",
        injected, total_rag, _estimate_tokens(rag_block),
    )
    return rag_block, remaining, injected


#: ``compress_turn(style="summary")`` が付ける圧縮マーク。同じ発話の原文と要約が
#: 別チャンクとして両方ヒットするため、``[参考情報]`` のスロットを二重に食う。
_SUMMARY_MARK = "[要約] "

#: 要約側の末尾に付く元文字数 (``…（1234文字）``)。重複判定では無視する。
_SUMMARY_TAIL_RE = re.compile(r"…（\d+文字）\s*$")


def _dedup_key_for_rag(text: str) -> str:
    """原文と ``[要約]`` 版を同一視するための正規化キー (純粋関数)。

    要約は「原文の先頭 N 文字 + …（N文字）」なので、マークと末尾を落とせば
    原文の接頭辞になる。短い方をキーにできないため、比較は呼び出し側で
    「一方が他方の接頭辞か」を見る (:func:`_drop_summary_duplicates`)。
    """
    body = text.strip()
    if body.startswith(_SUMMARY_MARK):
        body = body[len(_SUMMARY_MARK):]
    body = _SUMMARY_TAIL_RE.sub("", body)
    return "".join(body.split())


def _eligible_rag_indices(
    texts: list[str], current_query: str = "",
) -> list[int]:
    """``[参考情報]`` に載せる資格のあるチャンクの添字を返す (純粋関数)。

    落とすもの:

    1. 問いだけのチャンク — 過去セッションのユーザー発言はそのまま RAG の
       インデックスに入る。答えを含まないのに「参考情報」として提示されると、
       モデルはそれを根拠として扱う。SemMem ファクト / STM ノート側では
       ``carries_no_assertion`` が既に効いているが (``memory.pipeline.injector``)、
       RAG チャンクは素通りだった。
    2. 同じ発話の原文と ``[要約]`` 版の重複 — 情報は増えないのに枠と予算だけ減る。
    3. **今回のクエリそのもの** — 過去に同じことを聞いていると、その発言の
       STM ノートが最類似として返る。情報はゼロなのに枠を食い、しかも
       「この質問の周辺が根拠だ」とモデルを誘導する。実測 (2026-08-16 修正後の
       再検証): 「最近ハマってる食べ物ってある？私は担々麺に…」の
       ``[参考情報 1]`` が **同じ文そのもの** だった。

    実測 (2026-08-16 ライブ監査): 注入された ``[参考情報]`` 6 件のうち有用なもの
    ゼロ。ターン16「Git で直前のコミットを2つに分割」に
    「開発ロードマップを四半期ごとに区切る場合の作り方のコツは何ですか？」、
    ターン30「解約率の数値目標」に
    「そこから決済手数料を 3.6% 引くと、手取りは年間いくらになりますか？」
    といった **別セッションのユーザー質問文** が入っていた。ターン34 / ターン40 では
    5 枠のうち 2 枠が同一発話の原文と ``[要約]`` の組で埋まっていた。

    添字を返すのは、スコア付きチャンク (``list[tuple[cid, score, text]]``) と
    テキストのみの経路で同じ判定を使い回すため。経路ごとに書くと片方だけ直る
    非対称を作る。
    """
    kept: list[int] = []
    keys: list[str] = []
    query_key = _dedup_key_for_rag(current_query) if current_query else ""
    n_question = n_dup = n_echo = 0
    for i, text in enumerate(texts):
        # ``carries_no_assertion`` は「問いだけか」しか見ないので、
        # 「〜してください」型の **依頼** が素通りする。依頼は事実を含まないのに
        # 「参考情報」として提示され、根拠として扱われる。注入側 (``[関連する記憶]``)
        # は既に ``states_no_user_value`` を使っており、こちらだけ 1 段緩かった。
        #
        # 実測 (2026-08-23 ライブ監査、注入側の修正後): 「私が来月出張する都市を、
        # 確信度を付けて答えてください。」が ``[参考情報 1]`` として残っていた。
        # 本関数の docstring が自ら警告している「経路ごとに書くと片方だけ直る
        # 非対称」がそのまま起きていた形。
        if carries_no_assertion(text) or states_no_user_value(text):
            n_question += 1
            continue
        key = _dedup_key_for_rag(text)
        if query_key and key and (
            key == query_key or key.startswith(query_key)
            or query_key.startswith(key)
        ):
            n_echo += 1
            continue
        if key and any(key.startswith(k) or k.startswith(key) for k in keys):
            n_dup += 1
            continue
        if key:
            keys.append(key)
        kept.append(i)
    if n_question or n_dup or n_echo:
        logger.debug(
            "RAG chunks: dropped %d question-only, %d echoing the current "
            "query, collapsed %d raw/summary duplicate(s) (%d -> %d)",
            n_question, n_echo, n_dup, len(texts), len(kept),
        )
    return kept


def _select_rag_block(
    rag_chunks: list[str] | None,
    rag_scored_chunks: list[tuple[str, float, str]] | None,
    salience_ranker: SalienceRanker | None,
    remaining: int,
    total_rag: int,
    current_query: str = "",
) -> tuple[str | None, int, int]:
    """サリエンス経路 / フォールバック経路を選んで RAG block を返す。

    どちらの経路でも、載せる資格の判定 (:func:`_eligible_rag_indices`) は
    ここで一度だけ掛ける。
    """
    if remaining <= 0:
        return None, remaining, 0
    if salience_ranker and rag_scored_chunks:
        idx = _eligible_rag_indices(
            [t for _, _, t in rag_scored_chunks], current_query,
        )
        scored = [rag_scored_chunks[i] for i in idx]
        if not scored:
            return None, remaining, 0
        return _inject_rag_salience(
            scored, salience_ranker, remaining, total_rag,
        )
    if rag_chunks:
        filtered = [
            rag_chunks[i]
            for i in _eligible_rag_indices(rag_chunks, current_query)
        ]
        if not filtered:
            return None, remaining, 0
        return _inject_rag_fallback(filtered, remaining, total_rag)
    return None, remaining, 0


#: メモリブロックのラベル。「過去の会話の記録である」ことと「今回の会話が勝つ」
#: ことは **注入内容のすぐ隣**に置く。ラベルだけでは "古い記録である" と伝わっても
#: "衝突したらどちらを採るか" が決まらず、システムプロンプト側の
#: 「[関連する記憶] を回答の根拠として優先する」と噛み合って古い方が採用される
#: (実インシデント 2026-08-14 ライブ監査 ターン20: 同じプロンプト内に今回の
#: 発言「私はコーヒーをよく飲みます」と前セッションの note「コーヒーは苦手で
#: 紅茶派です」が両方あり、「好きな飲み物は紅茶です。コーヒーは苦手とのこと
#: でした」と反転して回答した)。
#:
#: 残りの指示 (「今回の質問に関係しなければ無視する」「ここに無い予定・日付・
#: 数値を創作しない」) は ``_REFERENCE_BLOCK_DIRECTIVE`` (system 側、接頭辞
#: キャッシュに乗る) が持つ。ラベル本文は毎ターン再プリフィルされるので、
#: **配置に意味があるものだけ**を残す (実測 2026-08-18/19: 105 トークンの
#: ラベルが 232 ターン中 92 ターンに載っていた)。
_SEMMEM_BLOCK_LABEL = "[関連する記憶] (過去の記録。今回の会話と食い違えば今回が優先)"


def _select_semmem_block(
    semmem_block: str | None,
    remaining: int,
) -> tuple[str | None, int]:
    """MemoryInjector 由来のメモリブロックを予算内なら採用する。

    ブロックは MemoryInjector 側で既に tier 予算内に整形済みのため、
    ここでは全体 context に収まるかの二次チェックのみ行い、収まらなければ
    破棄する (部分切り出しはしない)。
    """
    if not semmem_block or remaining <= 0:
        return None, remaining
    labeled = f"{_SEMMEM_BLOCK_LABEL}\n{semmem_block}"
    cost = _estimate_tokens(labeled)
    if cost > remaining:
        logger.debug(
            "semmem block dropped (%d tokens > remaining %d)", cost, remaining,
        )
        return None, remaining
    logger.debug("semmem block injected: %d tokens", cost)
    return labeled, remaining - cost


#: 動的ブロック (few-shot / file / semmem / RAG) へ切る固定予約枠 (token)。
#: 内訳の目安: few-shot ≤ 600 (:data:`_FEWSHOT_TOKEN_CAP`) + semmem ≤ 800
#: (MemoryInjector 側の tier 予算) + RAG 残余。**実際の注入量ではなくこの値を
#: 履歴予算から引く** ことが要点で、RAG のヒット件数が履歴の切り落とし位置を
#: 動かさないようにする (:func:`build_messages` の 4. を参照)。
_DYN_BLOCK_RESERVE = 1600

#: 予約枠が残予算に占めてよい上限。小さな context_size では固定値の 1600 が
#: 残予算を丸ごと食って履歴が 0 になるため、割合でも抑える。
_DYN_BLOCK_MAX_SHARE = 0.4

#: few-shot ブロックの token 上限。無上限だとセッション中に数千 token へ膨張し
#: (2026-07-15: 2739 tokens → 3 回全ドロップ = 学習効果ゼロ、通常時も履歴予算を
#: 圧迫)、all-or-nothing ドロップで無意味化する。上限内へ例単位で切り詰める。
_FEWSHOT_TOKEN_CAP = 600

#: format_fewshot_section の例区切り (### Example N)
_FEWSHOT_EXAMPLE_SPLIT_RE = re.compile(r"(?=^### Example \d+$)", re.MULTILINE)


def _truncate_fewshot_to_budget(block: str, budget: int) -> str | None:
    """few-shot ブロックを例単位で budget token 内に切り詰める。

    ヘッダ (## Few-shot Examples) + 先頭から入る分の例だけを残す。
    1 例も入らない場合は None。
    """
    parts = _FEWSHOT_EXAMPLE_SPLIT_RE.split(block)
    if len(parts) <= 1:
        return block if _estimate_tokens(block) <= budget else None
    header, examples = parts[0], parts[1:]
    kept = header
    for ex in examples:
        candidate = kept + ex
        if _estimate_tokens(candidate) > budget:
            break
        kept = candidate
    if kept.strip() == header.strip():
        return None
    return kept.rstrip() + "\n"


def _select_artifact_block(
    artifact_block: str | None, remaining: int,
) -> tuple[str, int]:
    """直前の成果物ブロックを予算内で採る。

    ``_select_fewshot_block`` と同じ形。予算に入らなければ **切り詰めて渡す**
    — 呼出側 (``_artifact.render_artifact_block``) が既に見出しの一覧へ
    縮退させているので、ここでさらに落とすと「何も無い」に戻ってしまう。
    """
    if not artifact_block or remaining <= 0:
        return "", remaining
    block = artifact_block.strip()
    tokens = _estimate_tokens(block)
    if tokens <= remaining:
        return block, remaining - tokens
    # 予算の 95% までを文字数で概算して切る (トークン推定の誤差ぶんの余裕)。
    keep_chars = max(0, int(len(block) * (remaining / max(1, tokens)) * 0.95))
    clipped = block[:keep_chars]
    logger.info(
        "build_messages: artifact block clipped to fit the budget "
        "(%d -> ~%d tokens)", tokens, remaining,
    )
    return clipped, 0


def _select_fewshot_block(
    fewshot_block: str | None,
    remaining: int,
) -> tuple[str | None, int]:
    """few-shot ブロックを予算内へ切り詰めて採用する。

    ``format_fewshot_section`` の出力は先頭に改行を含むため lstrip して返す
    (動的ブロックの先頭要素になるため)。予算は ``remaining`` と
    ``_FEWSHOT_TOKEN_CAP`` の小さい方。超過分は例単位で部分注入する
    (all-or-nothing ドロップだと膨張時に学習効果がゼロになる)。
    """
    if not fewshot_block or remaining <= 0:
        return None, remaining
    block = fewshot_block.lstrip("\n")
    if not block:
        return None, remaining
    budget = min(remaining, _FEWSHOT_TOKEN_CAP)
    cost = _estimate_tokens(block)
    if cost > budget:
        truncated = _truncate_fewshot_to_budget(block, budget)
        if truncated is None:
            logger.debug(
                "fewshot block dropped (%d tokens > budget %d, no example fits)",
                cost, budget,
            )
            return None, remaining
        new_cost = _estimate_tokens(truncated)
        logger.debug(
            "fewshot block truncated: %d -> %d tokens (budget %d)",
            cost, new_cost, budget,
        )
        return truncated, remaining - new_cost
    logger.debug("fewshot block injected: %d tokens", cost)
    return block, remaining - cost


def _prepend_dynamic_block(trimmed: list[ChatMessage], dyn_text: str) -> bool:
    """trimmed の最後の user メッセージ content 先頭に動的ブロックを前置する。

    動的ブロックは ``dyn_text + デリミタ + 元 content`` の順で、生クエリは末尾に残る
    (deliberative のツール結果はさらにその後ろへ追記されるため両立。付与順序の
    契約は ``core.turn_text`` のモジュール docstring 参照)。

    user メッセージが見つからなければ ``False`` を返す (呼び出し側が system へ fallback)。
    """
    return prepend_to_last_user(
        trimmed, dyn_text, separator=_DYNAMIC_CONTEXT_DELIMITER,
    )


def _append_dynamic_block(trimmed: list[ChatMessage], dyn_text: str) -> bool:
    """動的ブロックを最後の user メッセージの **生クエリより後ろ** へ後置する。

    照応 (「上の内容を」「さっきの話」) を含むターン専用の配置。前置すると
    注入ブロックが生クエリのすぐ上に並んで参照先を奪うため、後置して
    「直前のやり取り」が生クエリの直前に来る並びを保つ。
    """
    return append_to_last_user(
        trimmed, dyn_text, separator=_DYNAMIC_CONTEXT_TRAILING_DELIMITER,
    )


def _latest_user_refers_to_previous_output(trimmed: list[ChatMessage]) -> bool:
    """最後の user メッセージが直前の出力を指す照応を含むか。

    動的ブロック (記憶注入 / RAG / few-shot) は生クエリの **上** に前置されるため、
    「上の内容を〜」「それを〜」のような後方参照はブロックの側に束縛されうる。
    ブロック冒頭のラベルは既に「今回の会話で述べられた内容ではない」と否定して
    いるが、実機ではその指示より **位置的な隣接が勝った** (2026-08-03 ライブ監査:
    「上の内容を箇条書き 5 行にまとめ直してください」で直前ターンではなく注入
    ブロックを要約し、次ターンの英訳依頼にも汚染が伝播した)。

    指示文で直すのは既に一度失敗しているので、位置そのものを決定論で変える。
    """
    for msg in reversed(trimmed):
        if msg.get("role") != "user":
            continue
        return refers_to_previous_output(str(msg.get("content") or ""))
    return False


def _append_note_to_last_user(trimmed: list[ChatMessage], note: str) -> bool:
    """trimmed の最後の user メッセージ末尾へ注記を追記する (要素は mutate しない)。"""
    return append_to_last_user(trimmed, note)


def build_messages(
    system_prompt: str,
    history: list[ChatMessage],
    rag_chunks: list[str] | None = None,
    file_contexts: list[dict] | None = None,
    working_max_tokens: int = DEFAULT_WORKING_MAX_TOKENS,
    context_size: int = DEFAULT_CONTEXT_SIZE,
    max_tokens: int | None = None,
    rag_scored_chunks: list[tuple[str, float, str]] | None = None,
    salience_ranker: SalienceRanker | None = None,
    semmem_block: str | None = None,
    fewshot_block: str | None = None,
    history_min_tokens: int = 0,
    slot_resolver: "Callable[[str], frozenset[tuple[str, str]]] | None" = None,
    artifact_block: str | None = None,
) -> list[ChatMessage]:
    """
    messages リストを組み立て、トークン予算内に収める。

    テンプレート変換は LocalClient に委譲するため、
    ここではロール・内容の組み立てのみを担う。

    予算 = context_size - generation_reserve から
    system → 最新 user ターン予約 → fewshot → file_contexts → semmem → rag_chunks
    → history (残余) の優先順で配分。最新 user ターン (現在の質問) は動的ブロックに
    先立って ``min(質問トークン, working_max_tokens, 残予算)`` を予約し、履歴トリムで
    消失しないことを保証する (組み立て結果は history が user を含む限り user ロール ≥ 1)。

    サリエンスランカーが指定されている場合、RAG チャンクを5因子スコアで
    再評価し、トークン予算内で情報量を最大化するチャンク集合を選別する。

    **KV キャッシュ対応レイアウト**: system メッセージは ``system_prompt`` のみ
    (query 非依存・静的) とし、query 依存の動的部 (few-shot / file / semmem / RAG)
    は最後の user メッセージの content 先頭に前置する。これにより llama-server の
    prefix KV キャッシュが ``system + 過去履歴`` の範囲で再利用され、再プリフィルは
    最後の user ターンのみに限定される。gemma 系の「system は先頭 1 個」制約は維持。

    Args:
        system_prompt: インスタンス名プレフィックス付きの静的システムプロンプト。
        file_contexts: ファイルコンテキストのリスト。
            各要素は {"filename": str, "chunks": list[str]} の辞書。
        context_size: コンテキストウィンドウサイズ（トークン数）。
        max_tokens: 生成予約トークン数。None の場合は 512 をデフォルトとする。
        rag_scored_chunks: スコア付き検索結果 [(chunk_id, score, text), ...]。
            salience_ranker と同時に指定した場合、rag_chunks より優先される。
        salience_ranker: BudgetMem 式サリエンスランカー。
        artifact_block: 直前ターンで生成した長文成果物 (:mod:`backend.free.
            api.chat._artifact`)。**動的ブロックの先頭**に置く — 質問が
            直接指している対象なので、few-shot や参考情報より優先する。
            長文は履歴予算 (実測 1612 トークン) に入らず次ターンで消えるため、
            これが無いとモデルは「履歴に含まれていない」としか言えない。
        fewshot_block: ``format_fewshot_section`` 整形済みの few-shot ブロック
            (query 依存)。動的ブロックの先頭に置く。``None`` / 空なら付与しない。
        history_min_tokens: 過去履歴の最低確保トークン数 (床)。動的ブロック配分前に
            予約し、予算圧迫時でも直近の会話文脈が丸ごと締め出されるのを防ぐ。
            実履歴量・残予算・working_max_tokens でキャップされ、履歴が現在の質問
            のみの場合は 0 に縮退する。0 (既定) で無効。
    """
    generation_reserve = max_tokens if max_tokens is not None else DEFAULT_GENERATION_RESERVE
    budget = context_size - generation_reserve
    total_rag = len(rag_chunks) if rag_chunks else 0
    total_fc = len(file_contexts) if file_contexts else 0

    sys_tokens = _estimate_tokens(system_prompt)
    remaining = budget - sys_tokens

    # 最新 user ターン (現在の質問) のトークンを動的ブロック配分に先立って予約する。
    # 予約は残予算を上限とし、収まらない分は _trim_history の最新ターン圧縮保持が吸収する。
    reserved_latest = 0
    if history and history[-1].get("role") == "user":
        reserved_latest = min(
            _estimate_tokens(history[-1].get("content", "")),
            working_max_tokens,
            max(0, remaining),
        )
        remaining -= reserved_latest

    # 過去履歴の最低確保 (床)。実履歴量・残予算・working_max でキャップし、
    # 履歴が現在の質問のみ (新規セッション) の場合は 0 に縮退する —
    # 空回りの予約で動的ブロックを痩せさせない。
    # ``reserved_latest`` が押さえた最新 user ターン以外 = 履歴予算が要る分。
    # 履歴が assistant で終わる場合 (reserved_latest == 0) は最後の 1 件も
    # 「過去」に含める — 除くと予算が 0 に潰れて履歴が丸ごと落ちる。
    _past_turns = history[:-1] if reserved_latest else history
    past_tokens = sum(
        _estimate_tokens(t.get("content", "")) for t in _past_turns
    )
    hist_floor = 0
    if history_min_tokens > 0 and len(history) > 1:
        hist_floor = min(
            history_min_tokens,
            past_tokens,
            max(0, remaining),
            max(0, working_max_tokens - reserved_latest),
        )
        remaining -= hist_floor

    logger.debug(
        "build_messages: budget=%d (context_size=%d - %d), "
        "system=%d tokens, reserved_latest=%d, remaining=%d, "
        "rag_chunks=%d, file_contexts=%d",
        budget, context_size, generation_reserve, sys_tokens, reserved_latest,
        remaining, total_rag, total_fc,
    )

    # 0. 今回の会話で既に確定した値と食い違う注入候補を落とす。
    #    [関連する記憶] (semmem_block) / [参考情報] (RAG) / few-shot 例は
    #    3 つとも別々に組み立てられるため、ここが唯一の合流点になる。
    semmem_block, rag_chunks, rag_scored_chunks, fewshot_block = (
        _drop_superseded_context(
            history, semmem_block, rag_chunks, rag_scored_chunks, fewshot_block,
            slot_resolver,
        )
    )
    # 枠をまたいだ同一本文の二重掲載を落とす (両枠が揃うのはここだけ)。
    rag_chunks, rag_scored_chunks = _drop_rag_duplicates_of_semmem(
        semmem_block, rag_chunks, rag_scored_chunks,
    )
    total_rag = len(rag_chunks) if rag_chunks else total_rag

    # 動的ブロック (query 依存) を優先順に積む。system には含めない。
    #
    # **動的ブロックには固定の予約枠を切る**。以前は「余った分を全部使ってよい /
    # 使い残しは履歴へ回す」という配分だったため、RAG のヒット件数 (0〜5 件) が
    # そのまま履歴予算の増減になり、``_trim_history`` の切り落とし位置が毎ターン
    # 前後した。プロンプトの **途中** が入れ替わると llama-server の接頭辞 KV
    # キャッシュは全損する (実測 2026-08-16 ライブ監査: rag 3 件注入のターンで
    # history 29→20 に切られ cache 23.2% / 再評価 4122 tokens / prefill 151 秒)。
    # 予約を固定すると、履歴予算は system と最新ターンの長さだけで決まり、
    # 履歴の窓は会話の伸長に対して単調に動く。
    # 履歴予算を先に確定させる。上限は 3 つ:
    #   1. working_max_tokens (従来どおり)
    #   2. 予約枠を残した上での空き (= 動的ブロックのヒット件数に依存しない)
    #   3. **過去履歴が実際に必要とする量** — 新規セッション等で守る履歴が無い
    #      ときに予約が空回りして動的ブロックを痩せさせないため。
    # 動的ブロックは確定後の残りを受け取る。これで RAG が 0 件でも 5 件でも
    # 履歴の切り落とし位置は変わらない。
    dyn_floor = min(
        _DYN_BLOCK_RESERVE, int(max(0, remaining) * _DYN_BLOCK_MAX_SHARE),
    )
    history_budget = min(
        working_max_tokens,
        reserved_latest + hist_floor + max(0, remaining - dyn_floor),
        reserved_latest + hist_floor + past_tokens,
    )
    dyn_budget = max(
        0, remaining - max(0, history_budget - reserved_latest - hist_floor),
    )
    dyn_parts: list[str] = []

    # 0. 直前の成果物（質問が直接指している対象。何より優先する）
    artifact_part, dyn_budget = _select_artifact_block(artifact_block, dyn_budget)
    if artifact_part:
        dyn_parts.append(artifact_part)

    # 1. few-shot 例（query 依存）
    fewshot_part, dyn_budget = _select_fewshot_block(fewshot_block, dyn_budget)
    if fewshot_part:
        dyn_parts.append(fewshot_part)

    # 2. ファイルコンテキスト（ユーザー明示 → RAG より優先）
    fc_block, dyn_budget, injected_fc = _inject_file_contexts(
        file_contexts, dyn_budget, total_fc,
    )
    if fc_block:
        dyn_parts.append(fc_block)

    # 2.5 セマンティックメモリ注入（MemoryInjector、RAG より優先）
    semmem_part, dyn_budget = _select_semmem_block(semmem_block, dyn_budget)
    if semmem_part:
        dyn_parts.append(semmem_part)

    # 3. RAG チャンク（サリエンス優先 → フォールバック）
    rag_block, dyn_budget, injected_rag = _select_rag_block(
        rag_chunks, rag_scored_chunks, salience_ranker, dyn_budget, total_rag,
        current_query=str(history[-1].get("content") or "") if history else "",
    )
    if rag_block:
        dyn_parts.append(rag_block)

    # 静的 system メッセージ (動的部は含めない)
    messages: list[ChatMessage] = [{"role": "system", "content": system_prompt}]

    # 4. 会話履歴 (history_budget は動的ブロックを積む前に確定済み)
    if history_budget < min(working_max_tokens, reserved_latest + past_tokens):
        # WorkingMemory は working_max_tokens まで溜める一方、prompt 側はここまで
        # しか載らない = 窓を決める主体が 2 つある状態。WM のブロック押し出しと
        # _trim_history のブロック切り落としが別々のタイミングで先頭を動かすため、
        # 接頭辞 KV キャッシュが崩れる回数が倍になる。config で
        # memory.working_max_tokens をこの値以下へ寄せると WM 単独が窓を決める。
        logger.warning(
            "build_messages: working_max_tokens=%d exceeds the prompt history "
            "budget=%d (context_size=%d, generation_reserve=%d, system=%d, "
            "dyn_reserve=%d) — history is trimmed twice; consider lowering "
            "memory.working_max_tokens to %d or less",
            working_max_tokens, history_budget, context_size, generation_reserve,
            sys_tokens, _DYN_BLOCK_RESERVE, history_budget,
        )
    if len(history) > 1 and history_budget <= reserved_latest:
        logger.warning(
            "build_messages: context budget squeeze — past history got 0 tokens "
            "(budget=%d, system=%d, reserved_latest=%d, fewshot=%d, files=%d, "
            "semmem=%d, rag=%d)",
            budget, sys_tokens, reserved_latest,
            _estimate_tokens(fewshot_part) if fewshot_part else 0,
            _estimate_tokens(fc_block) if fc_block else 0,
            _estimate_tokens(semmem_part) if semmem_part else 0,
            _estimate_tokens(rag_block) if rag_block else 0,
        )
    # 最新ターンが予算を超えている = 切り詰め確定なら、後で system へ足す注記の
    # 分を先に履歴予算から引く (後付けすると予約予算を超える)。
    latest_tokens = (
        _estimate_tokens(str(history[-1].get("content") or "")) if history else 0
    )
    if latest_tokens > history_budget:
        history_budget = max(0, history_budget - _TRUNCATION_NOTE_RESERVE)
    trimmed = _trim_history(history, history_budget)
    # 最新ターンが切られたかは dyn_parts 前置 (下) で内容が変わる前に判定する。
    truncation_note = _latest_turn_truncation_note(history, trimmed)

    # 5. 動的ブロックを最後の user メッセージへ置く (KV キャッシュ対応)。
    #    既定は生クエリの **前** へ前置。ただし生クエリが直前の出力を指す照応を
    #    含む場合は、同じ user メッセージの生クエリ **後ろ** へ回す。前置すると
    #    注入ブロックが「上の内容」のすぐ上に並び、参照先を奪うため。
    #
    #    以前はこのケースで system へ回していたが、system は prompt の先頭なので
    #    足しても外しても prefix が丸ごと動き、llama-server の cache_prompt が
    #    全損する。実測 (2026-08-16 ライブ監査、同一ペイロードで比較):
    #      last-user 配置 : prompt_n=227  cache_n=2373  13.2s
    #      system 配置    : prompt_n=2600 cache_n=0     95.1s
    #      次ターン (注入を外して base へ戻す): prompt_n=2390 cache_n=0  99.5s
    #    = 1 回の照応検出で「そのターン + 次ターン」に約 180 秒。同監査では
    #    system 注入 3/40 ターンの TTFT が 134.0/123.8/163.3 秒でワースト 3 を占め、
    #    さらに「Git で直前のコミットを分割」のような誤検出も含まれていた。
    #    user メッセージが無い場合のみ従来どおり system へ結合して情報を落とさない。
    if dyn_parts:
        dyn_text = "\n\n".join(dyn_parts)
        place_after_query = _latest_user_refers_to_previous_output(trimmed)
        if place_after_query:
            logger.debug(
                "dynamic block placed after the raw query: latest user turn "
                "refers to previous output (avoiding anaphora capture)",
            )
            placed = _append_dynamic_block(trimmed, dyn_text)
        else:
            placed = _prepend_dynamic_block(trimmed, dyn_text)
        if not placed:
            messages[0] = {
                "role": "system",
                "content": system_prompt + "\n\n" + dyn_text,
            }

    # 最新ターン切り詰めの注記は system へ足す (messages[0] の差し替え後に行う)。
    # 動的ブロックと違い、ユーザー発言の中に混ぜるとユーザーが言っていないことを
    # 言ったことにしてしまうため user 側へは回さない (_latest_turn_truncation_note
    # の docstring 参照)。prefix キャッシュは失われるが、発火は巨大な貼り付けを
    # 受けたターンに限られる (2026-08-16 ライブ監査 40 ターンで発火 0)。
    if truncation_note:
        messages[0] = {
            "role": "system",
            "content": f"{messages[0]['content']}\n\n{truncation_note}",
        }
        logger.warning("build_messages: %s", truncation_note)

    # 現在日付 / 文字数上限の注記は **最後の user メッセージ末尾** に置く。
    # system へ足すと prefix KV キャッシュが毎ターン無効化される
    # (system は静的に保つ設計)。生クエリの直後は指示追従が最も効く位置でもある。
    for note in (
        _current_date_note(history),
        _persona_question_note(history),
        _char_limit_note(history),
        _output_form_note(history),
        _dropped_history_note(history, trimmed),
    ):
        if note:
            _append_note_to_last_user(trimmed, note)

    messages.extend(trimmed)

    # 不変則ガード: history に user ターンがあるのに組み立て結果に user が無い場合、
    # 圧縮した最新 user ターンを末尾へ再掲する (予約 + 最新ターン保持により通常経路では
    # 到達しない最終防衛線。発火は契約違反のシグナル。末尾が assistant の履歴では
    # 時系列順が崩れるが、user 不在によるテンプレート 400 の回避を優先する)。
    if not any(m.get("role") == "user" for m in messages):
        last_user = next(
            (t for t in reversed(history) if t.get("role") == "user"), None,
        )
        if last_user is not None:
            recovered = compress_turn(last_user)
            messages.append(recovered)
            logger.warning(
                "build_messages: no user turn survived assembly; re-appended "
                "compressed latest user turn (%d tokens)",
                _estimate_tokens(recovered.get("content", "")),
            )

    logger.debug(
        "build_messages complete: %d messages "
        "(history %d→%d, rag %d/%d, files %d/%d)",
        len(messages), len(history), len(trimmed),
        injected_rag, total_rag, injected_fc, total_fc,
    )

    return messages


def build_messages_for_loop(
    system: str,
    history: list[ChatMessage],
    compacted_steps: list,
    cfg: dict,
) -> list[ChatMessage]:
    """Meta-Cognitive ループ用の messages 組み立て。

    build_messages と同じ予算管理だが、RAG チャンクの代わりに
    圧縮済みステップ結果を注入する。

    Args:
        system: システムプロンプト
        history: 会話履歴
        compacted_steps: StepResult のリスト（圧縮済み）
        cfg: config.yaml の辞書
    """
    ctx = resolve_context_size(cfg, "base")
    max_tok = cfg.get("llama", {}).get("max_tokens", DEFAULT_MAX_TOKENS) or DEFAULT_GENERATION_RESERVE
    generation_reserve = max_tok if max_tok else DEFAULT_GENERATION_RESERVE
    budget = ctx - generation_reserve
    remaining = budget - _estimate_tokens(system)

    step_parts: list[str] = []
    for s in compacted_steps:
        step_text = f"[Step {s.iteration + 1}: {s.tool_name}]\n{s.output}"
        step_tokens = _estimate_tokens(step_text)
        if remaining - step_tokens < 0:
            break
        step_parts.append(step_text)
        remaining -= step_tokens

    if step_parts:
        sys_content = system + "\n\n" + "\n\n".join(step_parts)
    else:
        sys_content = system

    working_max = cfg.get("memory", {}).get("working_max_tokens", DEFAULT_WORKING_MAX_TOKENS)
    trimmed_history = _trim_history(history, min(remaining, working_max))

    # build_messages と同じ注記を最後の user メッセージ末尾へ置く。
    # これが無いと Meta-Cognitive 層に振られたクエリにだけ現在日付・人格制約・
    # 文字数上限が届かない (本関数の唯一の消費者が MetaCognitiveAgent のため、
    # 層が変わっただけで制約が消える)。3 つとも (history) -> str の純粋関数で、
    # シグナルが無ければ空文字列を返すのでトークン浪費にはならない。
    for note in (
        _current_date_note(history),
        _persona_question_note(history),
        _char_limit_note(history),
        _output_form_note(history),
    ):
        if note:
            _append_note_to_last_user(trimmed_history, note)

    logger.debug(
        "build_messages_for_loop: %d steps injected, remaining=%d",
        len(step_parts), remaining,
    )

    return [{"role": "system", "content": sys_content}, *trimmed_history]


#: ``_latest_turn_truncation_note`` の注記に確保するトークン数。注記は最新ターンを
#: 切り詰めたときだけ system へ足すため、切り詰めが確定している場合のみ履歴予算から
#: 引く (常時予約すると通常経路の履歴予算を無駄に削る)。予約せずに後付けすると
#: プロンプトが予約予算を超え、context 溢れの 400 を招く。
_TRUNCATION_NOTE_RESERVE = 120


def _chars_within_token_budget(text: str, max_tokens: int) -> int:
    """``text`` の先頭から ``max_tokens`` に収まる最大の文字数を返す。

    ``estimate_tokens`` は CJK 1 文字 ≒ 1 トークン / ASCII 4 文字 ≒ 1 トークン
    で見積もるため、文字数とトークン数の比は本文の構成で変わる。超過率から
    候補長を縮めながら数回試す (二分探索までの精度は不要)。
    """
    if _estimate_tokens(text) <= max_tokens:
        return len(text)
    cut = len(text)
    for _ in range(8):
        est = _estimate_tokens(text[:cut])
        if est <= max_tokens:
            return cut
        cut = min(int(cut * max_tokens / est), int(cut * 0.9)) or 1
    return cut


def latest_turn_truncation(
    history: list[ChatMessage],
    messages: list[ChatMessage],
) -> tuple[int, int] | None:
    """最新ターンが切り詰められた場合 ``(元文字数, 渡した文字数)`` を返す。

    ``build_messages`` の戻り値と入力 history から判定する。UI へ「あなたの発言は
    先頭のみ送られた」と提示するために API 層が使う (system 注記だけでは
    ベースモデルが従わず、ユーザーには何も見えないため)。

    ``messages`` は動的ブロック (few-shot / RAG / 記憶) が最後の user へ前置され、
    さらに文字数上限の注記 (``_char_limit_note``) が後置されることがあるため、
    長さ比較や末尾一致では判定できない。デリミタ以降を「ユーザー発言側」として
    切り出し、そこに元テキストが丸ごと含まれるかで判定する
    (末尾一致で見ていたときは、後置注記があるだけで未切り詰めのターンを
    「先頭のみ送信されました (1743 / 62 文字)」と誤報した)。
    """
    if not history or not messages:
        return None
    original = history[-1]
    sent = next(
        (m for m in reversed(messages) if m.get("role") == original.get("role")),
        None,
    )
    if sent is None:
        return None
    original_text = str(original.get("content") or "")
    # 前置された動的ブロックを除いた「ユーザー発言側」だけを見る。
    user_side = str(sent.get("content") or "").rsplit(
        _DYNAMIC_CONTEXT_DELIMITER, 1,
    )[-1]
    if not original_text or original_text in user_side:
        return None
    kept = user_side.removesuffix("...")
    return len(original_text), len(kept)


#: 「この会話そのもの」を対象にした問い。窓外に落ちたターンを指している可能性が
#: 高いので、履歴が切られているときは見えない範囲があることを明示させる。
_CONVERSATION_SCOPE_RE = re.compile(
    r"(今日|本日|ここまで|これまで|さっき|先ほど|最初の方|冒頭|序盤|今までの|"
    r"一連の|この会話|会話全体|話した(こと|話題|内容)|振り返)",
)

#: 会話スコープの問いに対して「何を求めているか」を示す語。単に「さっき」が
#: 出てくるだけの雑談で発火させないための第 2 条件。
_CONVERSATION_RECALL_RE = re.compile(
    r"(挙げ|列挙|まとめ|要約|振り返|覚えて|何だっけ|どれ|いくつ|3つ|三つ|"
    r"教えて|言って|整理)",
)


def _dropped_history_note(
    history: list[ChatMessage],
    trimmed: list[ChatMessage],
) -> str:
    """会話スコープの問いなのに古い履歴が落ちている場合の注記を返す。

    窓外のターンは復元経路が無いため、モデルは **見えている範囲だけ** で答える。
    手掛かりが皆無なら「参照できない」と正しく降参するが、可視範囲に部分一致
    する材料があるとそちらが優先され、黙って別物を答える (2026-08-16 ライブ監査:
    「今日話した技術的な話題を3つ」に対し、実際の技術ターン 8 件は全て窓外で、
    直前に見えていたビジネス話題 3 件を「技術的な話題」として提示した。同じ
    セッションの「困っていたこと」の問いには正しく降参している = 能力ではなく
    優先順の問題)。切り落としが起きたターンでは常に可視範囲の限界を明示させる。

    空文字を返す = 注記なし (:func:`_append_note_to_last_user` は falsy を無視)。
    """
    if len(trimmed) >= len(history):
        return ""
    latest = next(
        (t for t in reversed(history) if t.get("role") == "user"), None,
    )
    if latest is None:
        return ""
    text = str(latest.get("content") or "")
    if not (_CONVERSATION_SCOPE_RE.search(text)
            and _CONVERSATION_RECALL_RE.search(text)):
        return ""
    dropped = len(history) - len(trimmed)
    logger.debug(
        "_dropped_history_note: conversation-scope query with %d dropped "
        "message(s); adding visibility note", dropped,
    )
    return (
        f"[可視範囲] この会話の古い {dropped} メッセージは長さ制限で渡されて"
        "いない。会話の内容を振り返る問いには、見えている範囲だけで答え、"
        "見えていない前半がある旨を必ず明示すること。見えている範囲に該当が"
        "無ければ、別の話題で埋め合わせず「参照できない」と答えること。"
    )


def _latest_turn_truncation_note(
    history: list[ChatMessage],
    trimmed: list[ChatMessage],
) -> str | None:
    """最新ターンが予算超過で切られた場合、その旨を伝える system 注記を返す。

    ``_trim_history`` は最新ターンを drop せず ``compress_turn`` で切り詰めて
    保持するが、モデルに届くのは末尾の ``"..."`` だけで「どれだけ落ちたか」は
    伝わらない。そのためモデルは見ていない部分についても断定してしまう
    (2026-07-26 ライブ検証: 11,359 文字のメモを 4,096 文字に切られた状態で
    「検査装置のキャリブレーション周期は何回出てくるか」に対し、実際の 120 回
    ではなく渡された範囲の 43 回を全体の件数として断定した)。

    注記は system メッセージへ足す。ユーザー発言の中に注意書きを混ぜると
    ユーザーが言っていないことを言ったことにしてしまうため採らない。
    """
    if not history or not trimmed:
        return None
    original, kept = history[-1], trimmed[-1]
    if original.get("role") != kept.get("role"):
        return None
    original_text = str(original.get("content") or "")
    kept_text = str(kept.get("content") or "")
    if len(kept_text) >= len(original_text):
        return None
    return (
        f"注: 直近の{original.get('role', 'user')}発言は長さ制限で先頭のみ"
        f"渡されている (元 {len(original_text)} 文字 / 渡した {len(kept_text)} 文字)。"
        "未渡し部分を見たものとして扱わず、全体の件数・集計・網羅列挙は"
        "断定せず途中までしか読めていない旨を明示すること。"
    )


#: 履歴を切り落とすときの最小ブロック (メッセージ数)。予算ちょうどまでしか
#: 落とさないと、履歴が 1 ターン伸びるたびに窓の先頭が 1 つ進み、llama-server の
#: 接頭辞 KV キャッシュが **毎ターン** 無効化される (``WorkingMemory`` が
#: ``working_evict_block`` で同じ問題を避けているのと同じ理由)。落とす数を
#: ブロック単位へ切り上げて先頭をブロック境界で止め、次にブロックを跨ぐまで
#: 保持集合をバイト単位で不変にする。12 = 6 往復。
_HISTORY_DROP_BLOCK = 12


def _quantize_history_drop(
    history: list[ChatMessage], max_tokens: int,
) -> list[ChatMessage]:
    """予算超過分の切り落としをブロック単位へ切り上げた履歴を返す (純粋関数)。

    予算内に収まっている / 最新 1 ターンしか残らない場合は ``history`` をそのまま
    返し、呼出側の既存経路 (最新ターンの圧縮保持) に委ねる。
    """
    if len(history) <= 1:
        return history
    fit = 0
    total = 0
    for turn in reversed(history):
        total += _estimate_tokens(turn.get("content", ""))
        if total > max_tokens:
            break
        fit += 1
    if fit >= len(history):
        return history
    over = len(history) - fit
    blocks = -(-over // _HISTORY_DROP_BLOCK)  # ceil
    drop = min(len(history) - 1, blocks * _HISTORY_DROP_BLOCK)
    logger.debug(
        "_trim_history: dropping %d oldest messages (need %d, block=%d)",
        drop, over, _HISTORY_DROP_BLOCK,
    )
    return history[drop:]


def _trim_history(
    history: list[ChatMessage],
    max_tokens: int,
) -> list[ChatMessage]:
    """トークン予算に収まるよう、古い履歴を圧縮・削除する。

    **最新ターン (通常は現在の user 質問) は予算超過でも drop しない**: 予算に
    収まる長さへ圧縮して保持する (下限は compress_turn 既定の 200 字 ≈ 203 トークン
    で、超過分は呼び出し側の generation_reserve が吸収する)。

    トークン推定は estimate_tokens() を使用
    （CJK: 1文字≒1トークン、ASCII: 4文字≒1トークン）。
    """
    if not history:
        logger.debug("_trim_history: empty history, nothing to trim")
        return []

    original_len = len(history)
    history = _quantize_history_drop(history, max_tokens)

    result: list[ChatMessage] = []
    total_tokens = 0

    # 新しいターンから逆順に追加
    for turn in reversed(history):
        content = turn.get("content", "")
        estimated_tokens = _estimate_tokens(content)

        if total_tokens + estimated_tokens > max_tokens:
            if not result:
                # 最新ターンは drop せず、予算連動で圧縮して保持する
                # (この分岐では total_tokens == 0 のため残予算 = max_tokens)。
                # max_chars はトークンではなく文字数の上限なので、トークン予算を
                # そのまま渡すと単位が合わない: ASCII (4 文字 ≒ 1 トークン) では
                # 予算の 1/4 しか使わず、CJK (1 文字 ≒ 1 トークン) では予算ぎりぎり
                # まで載る。推定トークン数で残予算まで詰める形に揃える。
                compressed = compress_turn(
                    turn, max_chars=_chars_within_token_budget(
                        content, max(max_tokens, 200),
                    ),
                )
                compressed_tokens = _estimate_tokens(compressed.get("content", ""))
                result.insert(0, compressed)
                total_tokens += compressed_tokens
                logger.warning(
                    "_trim_history: latest turn exceeds budget, kept compressed "
                    "(role=%s, %d->%d tokens, max=%d)",
                    turn.get("role"), estimated_tokens, compressed_tokens,
                    max_tokens,
                )
            else:
                # 予算超過: 残りの古いターンを圧縮
                compressed = compress_turn(turn)
                compressed_tokens = _estimate_tokens(compressed["content"])
                if total_tokens + compressed_tokens <= max_tokens:
                    result.insert(0, compressed)
                    total_tokens += compressed_tokens
                    logger.debug(
                        "_trim_history: compressed turn (role=%s, %d->%d tokens)",
                        turn.get("role"), estimated_tokens, compressed_tokens,
                    )
                else:
                    logger.debug(
                        "_trim_history: dropped turn (role=%s, %d tokens) — "
                        "budget exhausted (%d/%d)",
                        turn.get("role"), estimated_tokens, total_tokens, max_tokens,
                    )
            break
        else:
            result.insert(0, turn)
            total_tokens += estimated_tokens

    logger.debug(
        "_trim_history: %d/%d turns kept, %d estimated tokens (max=%d)",
        len(result), original_len, total_tokens, max_tokens,
    )
    return result

