"""フィードバック収集: 暗黙的シグナルの検出と経験バッファへの記録

学習済みパターンストアと連携し、ツールルーティング false_negative 時に
クエリから動作指示語を tool_routing パターンとして自動学習する
(長文ルーティングは success / false_negative 時に long_form パターンを学習)。
訂正・言い直しの検出は決定論 (ハードコード正規表現 / 文字重複率) のみで、
学習パターンは使わない。
"""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from backend.free.core.intent_vocab import (
    EXPLICIT_WINDOWS_PATH_RE,
    NUMBER_LITERAL_RE,
    REFERENTIAL_WRITE_TARGET_RE,
    is_plain_statement,
)
from backend.free.core.locale_patterns import (
    has_japanese_script,
    matches_either,
)
from backend.free.core.session_mode import is_create_mode
from backend.free.core.correction_target import (
    CONTRAST_MARKER,
    DEFAULT_LOOKBACK as CORRECTION_LOOKBACK,
    contrast_pair,
    contrast_pairs,
    old_value_core,
    resolve_correction_target,
    split_sentences,
)
from backend.free.core.correction_verdict import (
    marks_not_own_restatement,
    mask_quoted_speech,
    norm_span,
)
from backend.free.core.response_arithmetic import (
    find_arithmetic_contradictions,
    find_conclusion_contradiction,
    find_sign_contradiction,
    iter_ja_numbers,
)
from backend.free.core.response_verifiers import (
    declared_count_mismatch,
    false_tool_unavailability,
    misstated_change_rate,
)
from backend.free.core.relative_date import (
    DATE_RESULT_ANCHOR_MISMATCH,
    DATE_RESULT_IGNORED,
    date_result_use,
)
from backend.free.core.script_ranges import KANJI, KANJI_MARKS, KATAKANA_WORD
from backend.free.core.response_dates import extract_tool_anchor
from backend.free.core.verifier_events import (
    current_grounding,
    current_tool_decision,
    current_tool_uses,
    current_unread_session_files,
    record_rag_signals,
    record_turn_outcome,
    record_verifier_hit,
)
from backend.free.agent.issue_ledger import record_current_issue
from backend.free.core.text_quality import (
    abstains_on_reference_material,
    calculate_result_contradiction,
    cites_reference_material,
    claims_completed_state_change,
    contradicts_measured_values,
    fabricated_household_count,
    has_broken_ja_spacing,
    has_chinese_token_leak,
    is_cut_off_answer,
    ungrounded_answer_names,
    retracts_own_conclusion,
    value_was_adopted,
    VALUE_REJECTION_RE,
    violates_length_constraint,
    violates_output_form,
)
from backend.free.core.text_similarity import (
    bigram_coverage,
    bigram_cosine,
    content_bigram_cosine,
)
from backend.free.learning.level0_instant import (
    RAG_ABSTAIN_NOT_RETRIEVED,
    RAG_ABSTAIN_NOT_SHOWN,
    RAG_ABSTAIN_SHOWN,
    RAG_ADOPTED_CORPUS_KEY,
    RESPONSE_FULL_CAP,
    RESPONSE_SUMMARY_CAP,
    TOOL_CONTEXT_CAP,
    TOOL_CONTEXT_KEY,
    TOOL_CONTEXT_TRUNCATED_KEY,
    ExperienceBuffer,
    ExperienceEntry,
    FeedbackSignals,
    GenerationConfigRef,
    truncate_at_boundary,
)
from backend.log_config import get_logger
from backend.trace_context import get_trace_id
from backend.utils import utc_now

if TYPE_CHECKING:
    from backend.debug_logger import DebugLogger
    from backend.free.agent.learned_patterns import LearnedPatternStore

logger = get_logger("agent.feedback")

#: 書込みゲートの断りだけで終わったターンの件数 (プロセス内、:func:`guard_denied_turns`)。
_GUARD_DENIED_TURNS = [0]


def guard_denied_turns() -> int:
    """書込みゲートの断りだけで終わったターンの件数 (このプロセスで数えた分)。"""
    return _GUARD_DENIED_TURNS[0]

# ユーザー訂正パターン（ハードコード: 高確度）
# learned correction 機構 (旧・層2) は 2026-07-21 に廃止した。学習される語が
# 「訂正の言い回し」ではなく「訂正が起きたときの話題語」だったため偽陽性率
# ~85% (経験 65 件の実測) に達し、Level 1 fitness / critique / few-shot /
# Level 2 cvector 対比ペアの学習信号を汚染していた。以後、訂正語彙の拡充は
# 本リストへのハードコード追加で行う (見逃しは prev_failed / same_target の
# 別層が拾うため、追加は確度の高い表現に限る。「実は」「厳密には」「〜では
# なく」単独は話題導入・比較の一般語法と識別できず見送った実績あり)。
CORRECTION_PATTERNS = [
    re.compile(r"違[うわえおっく]|違い(?:ます|ません|まし)", re.IGNORECASE),
    # 「間違え」(下一段) は旧 [いっ] が取りこぼしていた (「間違えていますよ」)
    re.compile(r"間違[いっえ]", re.IGNORECASE),
    re.compile(r"そうじゃ", re.IGNORECASE),
    re.compile(r"そうではな", re.IGNORECASE),
    re.compile(r"正しくは", re.IGNORECASE),
    re.compile(r"訂正", re.IGNORECASE),
    # 出力値の取り違え指摘 (「値が逆になっていませんか？」— learned 層廃止時の
    # 実データ真陽性から回収。仮定表現「逆になっていたら」は誤検知しないよう
    # 疑問形終端まで要求する)
    re.compile(r"逆になって(?:い)?(?:ません|ない)か", re.IGNORECASE),
    # 成果物未達の報告 + やり直し要求 (2026-07-15 の訂正 2 ターンが
    # どちらも検出漏れした語彙)
    re.compile(r"作られて(?:い)?(?:ない|ません)|できて(?:い)?(?:ない|ません)", re.IGNORECASE),
    re.compile(r"(?:し|やり|作り)直して", re.IGNORECASE),
    re.compile(r"not correct", re.IGNORECASE),
    re.compile(r"that'?s wrong", re.IGNORECASE),
    re.compile(r"^\s*actually\b", re.IGNORECASE),
    # 英語語彙拡充 (日本語(漢字/かな)と英語(ASCII)は文字体系が異なり相互誤爆
    # リスクが実質ゼロなため、locale 分岐せず常時併用する)。
    re.compile(r"that'?s\s+not\s+(?:right|correct)", re.IGNORECASE),
    re.compile(r"^\s*(?:no,?\s+)?that'?s\s+wrong\b", re.IGNORECASE),
    re.compile(r"you\s+got\s+it\s+(?:backwards?|wrong|mixed\s+up)", re.IGNORECASE),
    # ``redo`` / ``retry`` を **裸で** 拾ってはいけない。短い英語動詞は
    # 技術用語として日本語文中に頻出し、境界も無かったため部分一致していた
    # (実データ 2026-08-14: 「先ほどの retry デコレータで、max_retries=3 の
    # とき関数本体は最大何回呼ばれますか？」「retry_decorator.md の中身を
    # 読んで、先頭 3 行をそのまま引用してください。」の 3 ターンが訂正として
    # 記録され、Level 2 の失敗コーパス 10 件中 3 件を占めた)。
    # 「日本語と英語は文字体系が違うので相互誤爆しない」という前提は、
    # 英語の識別子が日本語文に埋め込まれる場面で成り立たない。
    # 命令形の文脈 (please / 目的語) を要求し、``_`` や数字との連結も弾く。
    re.compile(
        r"(?<![A-Za-z0-9_])(?:try\s+again|fix\s+that)(?![A-Za-z0-9_])"
        r"|please\s+(?:redo|retry)(?![A-Za-z0-9_])"
        r"|(?<![A-Za-z0-9_])(?:redo|retry)\s+(?:that|it|this)(?![A-Za-z0-9_])",
        re.IGNORECASE,
    ),
    re.compile(r"that'?s\s+incorrect", re.IGNORECASE),
]

# 直前ターンが失敗した直後の **短い否定だけ** の発話。字句の訂正語彙
# (CORRECTION_PATTERNS) にも同一成果物の再指定にも当たらないが、失敗の直後
# に発話全体がこれだけなら訂正とみなせる (``prev_failed`` 層の唯一の入口)。
# 以前は失敗の次のターンを **無条件で** 訂正としていたため、話題を変えた
# だけの発話が user_correction に化けていた (2026-09-02 監査 R-B3)。
SHORT_NEGATIVE_FEEDBACK_RE = re.compile(
    r"^\s*(?:"
    r"(?:いや|いえ|うーん|えっ|え)?[、,。.\s]*"
    r"(?:ちがう|ちがいます|違う|違います|だめ|ダメ|駄目)(?:です|だよ|よ|ね|って)?"
    r"|(?:no|nope|nah)[,.!]?(?:\s+(?:not\s+that|that'?s\s+not\s+it|wrong))?"
    r"|not\s+that(?:\s+one)?|wrong"
    r")[。.!！?？\s]*$",
    re.IGNORECASE,
)
#: 短い否定として認める発話長の上限 (これより長ければ「否定だけ」ではない)。
_SHORT_NEGATIVE_MAX_CHARS = 16

# create モードの実行結果報告 (2026-07-18: create 経験に訂正シグナルがほぼ発生
# せず Level 1 fitness が無差別化する一因だった語彙)。「動かない」「エラー」
# 等は一般語彙で他モードの質問・新規依頼 (例:「ホバーしても動かないボタンに
# して」「テストが通らないという話について教えて」) と誤検知しやすいため、
# CORRECTION_PATTERNS には含めず (a) create モード限定 (b) 直前ターンが存在する
# 場合のみ (訂正対象が無い最初のターンでは新規の質問/依頼である可能性が高い)、
# の 2 条件でゲートする (_detect_correction 参照)。
CREATE_FAILURE_REPORT_PATTERNS = [
    re.compile(r"動(?:か|き)(?:ない|ません)", re.IGNORECASE),
    re.compile(r"エラー(?:が出|にな|です)", re.IGNORECASE),
    re.compile(r"テストが(?:通ら|落ち)", re.IGNORECASE),
]
# 上記語彙を含んでいても文末が新規依頼の完結形 (「〜にして」「〜作って」
# 「〜教えて」等) なら報告ではなく仕様/質問の可能性が高いため除外する
# (「ホバーしても動かないボタンにして」「テストが通らないという話について
# 教えて」の誤検知回避)。「〜(やり/し/作り)直して」で終わる依頼は
# CORRECTION_PATTERNS 側で既に検出されるため対象外にする必要はない。
_CREATE_FAILURE_REPORT_EXCLUDE_RE = re.compile(
    r"(?:ください|にして|教えて|作って|実装して)[。.！!？?]*\s*$",
)

# CREATE_FAILURE_REPORT_PATTERNS / _CREATE_FAILURE_REPORT_EXCLUDE_RE の
# 英語版。日本語版の文末アンカー方式 (ください/にして等) は、英語の新規
# 依頼マーカー (please/命令形/モーダル動詞) が文頭に来る構造とは合わないため、
# exclude 側は文頭アンカー方式に作り直す。
CREATE_FAILURE_REPORT_PATTERNS_EN = [
    re.compile(r"\b(?:doesn'?t|does\s+not|isn'?t|is\s+not)\s+work(?:ing)?\b", re.IGNORECASE),
    re.compile(r"\bnot\s+working\b", re.IGNORECASE),
    re.compile(
        r"\b(?:i'?m\s+)?getting\s+an?\s+error\b"
        r"|\berrors?\s+(?:occur(?:s|red)?|showing|appearing|popping\s+up)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\btests?\s+(?:are|is|keep(?:s)?)\s+fail(?:ing)?\b|\btests?\s+fail(?:ed|s)?\b", re.IGNORECASE),
    re.compile(r"\b(?:it|this|that)\s+(?:broke|is\s+broken|crashed?)\b", re.IGNORECASE),
]
_CREATE_FAILURE_REPORT_EXCLUDE_RE_EN = re.compile(
    r"^\s*(?:please\s+)?(?:make|build|create|add|write|implement|set\s+up|change|fix)\b"
    r"|^\s*(?:can|could|would)\s+you\b"
    r"|^\s*please\b",
    re.IGNORECASE,
)

#: JA / EN 双方を **locale に関わらず** 評価する union
#: (``_detect_correction_lexical`` の 1b。除外ガードも同時に union する)。
CREATE_FAILURE_REPORT_PATTERNS_ALL = [
    *CREATE_FAILURE_REPORT_PATTERNS, *CREATE_FAILURE_REPORT_PATTERNS_EN,
]

# アシスタント自身による前ターンの撤回。ユーザーの字句ではなく**自分の出力**を
# 見るため、CORRECTION_PATTERNS が抱えていた偽陽性 (話題語の学習・一般語法との
# 識別不能) の問題が構造的に起きない。
#
# 実インシデント (2026-08-05 ライブ監査): 40 ターン中、訂正シグナルは 0 件。
# ユーザーが「本当ですか？さっき ... に書き込んでもらったはずです」と矛盾を
# 指摘し、アシスタントが「失礼いたしました。過去の記録を確認したところ…」と
# 撤回したターンすら correction=false のまま記録されていた。字句パターンの
# 拡充 (「本当ですか」等) は一般語法と識別できず過去に偽陽性 85% を出している
# ため、拡充ではなく**自分の撤回**という別軸の証拠を採る。
#
# 撤回は応答の冒頭に来る。本文中の「訂正」への言及 (例: 「訂正機能について
# 説明します」) を拾わないよう先頭 80 文字に限定する。
_SELF_RETRACTION_HEAD_CHARS = 80
_ASSISTANT_SELF_RETRACTION_RE = re.compile(
    r"失礼(?:しました|いたしました|致しました)"
    r"|申し訳(?:ありません|ございません|あり?ませんでした)"
    r"|訂正(?:します|いたします|させてください)"
    r"|(?:先ほど|さきほど|前)の(?:回答|説明|発言)(?:は|が)(?:誤り|間違)"
    r"|誤(?:り|情報)でした|間違(?:い|え)でした"
    r"|(?:i\s+)?apolog(?:ise|ize)|my\s+mistake|i\s+was\s+(?:wrong|incorrect)"
    r"|correction:",
    re.IGNORECASE,
)


def detect_assistant_self_retraction(response: str) -> bool:
    """応答冒頭がアシスタント自身による前ターンの撤回かを判定する (純粋関数)。"""
    if not response:
        return False
    return bool(
        _ASSISTANT_SELF_RETRACTION_RE.search(
            response[:_SELF_RETRACTION_HEAD_CHARS],
        ),
    )


# ── 訂正の帰属判定 ────────────────────────────────────────────────
#
# CORRECTION_PATTERNS が拾う「訂正」は 3 種類あり、**アシスタントが誤った**
# ことを意味するのは一部だけ。にもかかわらず ``user_correction`` は fitness で
# 最大の減点 (-0.8) を受け、cvector の negative を駆動していた。
#
# 実データ 447 件中の訂正 24 件を目視分類した内訳:
#   assistant     5 件 (21%) — 「その計算は違います」「単位を取り違えていませんか」
#   self         11 件 (46%) — 「すみません、火曜ではなく水曜の間違いでした」
#                              (アシスタントの応答は正しい)
#   not_correction 8 件 (33%) — 「3 番目を差し替えて、同じファイルに保存し直して」
#                              (単なる編集依頼)、「訂正後の距離を挙げて」(質問)
#
# つまり **79% が誤ったペナルティ** だった。下記の判別で assistant のみを
# ``user_correction`` として扱う。判別不能は従来どおり assistant に倒す
# (保守的側。ルールが効かなければ現行挙動のまま)。

#: 「訂正」「間違い」を **目的語として問う質問**。訂正そのものではない。
#:
#: 監査の振り返り (「どこを間違えた？」「何回訂正させた？」) は、直前の応答が
#: 誤っていたことを意味しない。にもかかわらず訂正として記録され、Level 2 の
#: 失敗コーパスに混ざっていた (実データ 2026-08-14: 10 件中 2 件が
#: 「私があなたの回答を訂正させたのは何回で…」「あなたが間違えた点を…列挙して」)。
#: 訂正語の後に **列挙・計数を求める語** が続く形を除外する。
#: 「正しくは X です。訂正してください」のような本物の訂正は、これらの語を
#: 伴わないので影響を受けない。
_ASKS_ABOUT_CORRECTION_RE = re.compile(
    r"(?:訂正|間違[いえ])[^。！？\n]{0,20}?"
    r"(?:答え|挙げ|教え|列挙|示し|説明し|何回|何件|いくつ|どこ|点を|箇所)",
)

#: 記憶の想起を **尋ねる** 疑問文 (「確認ですが、私の職業は何と言いましたか？」)。
#: 「確認」「言いましたか」が訂正語彙と並ぶが、直前の応答が誤っていたとは
#: 言っていない。2026-09-05 の失敗 32 件のうち 2 件がこの形で、正答した
#: 自己紹介ターンが Level 2 の学習データに失敗として入っていた。
_RECALL_QUESTION_RE = re.compile(
    r"(?:何と|なんと|どう)(?:言い|いい|申し|答え)ました(?:か|っけ)"
    r"|と(?:言い|いい)ましたか[？?]?\s*$"
    r"|でしたっけ[？?]?\s*$",
)

#: 2 つの物事の相違点を **尋ねる** 疑問文。訂正ではない。
#:
#: ``CORRECTION_PATTERNS`` の先頭 ``違[うわえおっく]`` は「それは違う」を拾う
#: ためのものだが、「A と B はどう違うのか」という比較質問の ``違う`` にも
#: 一致する。実インシデント (2026-08-18 ライブ監査 ターン4):
#: 「Python の GIL があることで、CPU バウンド処理と I/O バウンド処理で
#: スレッドの効果が**どう違うのか**、簡潔に説明してください。」という純粋な
#: 知識質問が ``correction_detected_by=hardcoded`` で訂正として記録された。
#: 訂正シグナルは Level 1 の fitness で **欠陥** として数えられ
#: (``_calc_fitness_memory`` / ``_calc_fitness_router``)、Level 2 の失敗
#: コーパスにも入るため、正しく答えたターンが失敗として学習される。
#:
#: 疑問の代用形 (どう / どこが / 何が …) が ``違う`` の直前に来る形だけを
#: 除外する。「それは違います」「答えが違う」のような本物の指摘は代用形を
#: 伴わないので影響を受けない。「〜の違いを教えて」は ``違い`` の後に
#: ``ます/ません/まし`` が続かないため、そもそも先頭パターンに一致しない。
#: アシスタント出力への言及 (``_ASSISTANT_OUTPUT_REF_RE``) は本関数の先頭で
#: ``assistant`` を返すため、この除外より先に確定する。
#:
#: **丁寧形・テ形も同じ扱いにする (2026-09-06、F-07)**。以前は ``違う`` の
#: 終止形だけを見ており、「BBR は損失ベースの輻輳制御と何が根本的に
#: **違いますか**。」という純粋な比較質問が訂正として記録されていた
#: (実機検証で再現)。字句側の ``違います`` に先に一致するので帰属判定へ来る
#: のは想定どおりで、来た上で **代用形を伴う比較質問だと判別できる**。
#: 判別できないものを ``assistant`` に倒す方針は変えないが、判別できる形を
#: 倒し続ける理由は無い — 訂正は Level 1 の欠陥として数えられ、
#: ``corrected_entry_id`` 経由で正答ターンを「誤り」として訂正ペアに載せる。
_ASKS_ABOUT_DIFFERENCE_RE = re.compile(
    r"(?:どう|どの(?:よう|ように)|どこ(?:が|に)|何(?:が|は)|なに(?:が|は))"
    r"[^。！？\n]{0,12}?違(?:う|い(?:ます|ました)|って)",
)

#: 出力形式・言語の変更依頼。内容の誤りを指していない。
_REFORMAT_REQUEST_RE = re.compile(
    r"同じ内容を.{0,10}(?:日本語|英語|中国語)で"
    r"|(?:日本語|英語)で(?:説明し直|書き直|言い直)",
)

#: 前提を変えての再計算 / 再作成の依頼。「〜(なので|ので|から)、…し直して」は
#: ユーザーが **新しい条件** を持ち込んでやり直しを頼んでいるのであって、直前の
#: 応答が誤っていたとは言っていない。「うちの社内規定では税率 8% で計算するので、
#: 37 個の税込合計を計算し直して」が ``(?:し|やり|作り)直して`` に掛かり、正答した
#: 論理パズルのターンが失敗として Level 2 の学習データに入っていた
#: (2026-09-05 ライブ監査 T14)。
#:
#: 前提の変更が **1 つ前の文** で述べられる形も同じ (「藤堂さんは水曜と金曜が
#: 固定で不可、宇野さんは土曜のみ勤務可能という制約が加わりました。割当を
#: 作り直してください。」2026-09-07 ライブ監査 T04/2。正答したシフト案が
#: 訂正ペアの誤り側に載った)。
#:
#: 前提動詞は **自動詞・完了形に偏っていた**。「4 人分に増やしたいです。分量を
#: 計算し直してください。」(2026-09-08 ライブ監査 T07/2) は他動詞 + 意向形
#: (``増やしたい``) で 1 つも当たらず、接続詞分岐も ``[^。！？\n]`` で文を
#: 跨げなかったため訂正として記録された。他動詞・意向形を足し、接続詞分岐は
#: 文境界を跨げるようにする (字数上限 40 はそのまま)。
_PREMISE_CHANGE_REDO_RE = re.compile(
    r"(?:なので|ので|から|場合は|場合で|として|に変えて|に変更して|にして)[、,\s]*"
    r"[^\n]{0,40}?(?:し|やり|作り|組み|計算し|書き)直して"
    r"|(?:加わ|追加|変更|変わ|変え|増え|増やし|減っ|減らし|決まっ|判明し|したい)"
    r"[^。！？\n]{0,12}[。！？]\s*"
    r"[^。！？\n]{0,40}?(?:し|やり|作り|組み|計算し|書き)直して",
)

#: 「違う」が **複合動詞の一部** (取り違える / 食い違う / すれ違う / 勘違い /
#: 行き違い / 入れ違い) の形。字句パターンの先頭 ``違[うわえおっく]`` が部分一致
#: するが、誰かの出力が誤っているとは言っていない (2026-09-07 ライブ監査 T18/5
#: 「ID トークンとアクセストークンを取り違える実装ミスをよく見ます」が訂正として
#: 記録された)。アシスタントへの問い (「取り違えていませんか」) は除かない。
_COMPOUND_DIFFERENCE_RE = re.compile(
    r"(?:取り|食い|すれ|勘|行き|入れ)違[いえう](?!て(?:い)?(?:ます|ません|ないか))",
)

#: 「違う」を **伝聞・疑問** で使う形 (「挙動が違うと聞きました」「違うそうですね」
#: 「どこが違うのでしょうか」)。比較質問と同じく訂正ではない
#: (2026-09-07 ライブ監査 T12/1 「CPU とメモリで挙動が違うと聞きましたが」)。
#: 引用の **閉じ括弧を挟む** 形も同じ (2026-09-08 ライブ監査 T01/2
#: 「「モニタと印刷で色が違う」というクレームが定期的に来ます」)。括弧が
#: 入ると ``違う`` の直後が ``と`` でなくなり、伝聞と判定できていなかった。
_HEARSAY_DIFFERENCE_RE = re.compile(
    r"違(?:う|い(?:ます)?)[」』）\)”’\"']*"
    r"(?:と(?:聞|きい|言われ|いわれ|いう|いい|のこと)|そう|らしい|のか|のでしょう|んでしょう"
    r"|のです|んです|んですか|のですか)",
)

#: ユーザー自身の申告訂正。謝罪 / 自己の過去発言への言及 + 事実の言い換え。
_SELF_CORRECTION_RE = re.compile(
    r"(?:すみません|すいません|ごめん|失礼しました)[、,。\s]*"
    r".{0,30}?(?:ではなく|じゃなく|の間違い|間違えました)"
    r"|(?:先ほど|さっき|前に)\s*[「『]?.{0,20}?[」』]?\s*と(?:言|申)"
    r"|^訂正(?:です|します)"
    r"|あ[、,]?\s*間違えました"
    r"|実はこれは",
)

#: アシスタントの出力を指す参照。これがあれば自己訂正ではない。
#:
#: **裸の「違います」「誤りです」は入れない (2026-09-06、F-07)**。これらは
#: 「何かが誤っている」という断定であって *アシスタントの出力を指す参照* では
#: なく、名前と役割に反する。本パターンは ``classify_correction_target`` の
#: 最優先で ``assistant`` を返すため、ここに字句を置くと **後段の除外規則が
#: 一切到達しなくなる**。実機検証で「BBR は損失ベースの輻輳制御と何が根本的に
#: 違いますか。」という純粋な比較質問が訂正として記録され、`corrected_entry_id`
#: 経由で正答ターンを「誤り」として訂正ペアに載せかけた。
#:
#: 外しても本物の指摘は落ちない — 「計算が違います。」「それは違います。」は
#: どの除外規則にも当たらず、末尾の既定で ``assistant`` に落ちる。
#: 裸の字句を外した分、**参照そのもの** を明示的に持つ。二人称の所有格・
#: 連体修飾 (「あなたの回答」「あなたが示した実装」) と「先ほどの回答」型は、
#: アシスタントの出力を指す最も直接的な表現でありながら、以前は 1 つも
#: 入っていなかった (「違います」が代用として機能していたため気付けない)。
_ASSISTANT_OUTPUT_REF_RE = re.compile(
    r"その(?:計算|答え|回答|結果|数字|値)"
    r"|あなた(?:の|が)"
    r"|(?:先ほど|さきほど|さっき|上|今)の"
    r"(?:回答|答え|説明|計算|コード|実装|出力|結果)"
    r"|(?:最後|最初)の.{0,6}(?:行|文|項目).{0,10}(?:なって|です)"
    r"|取り違えて(?:い)?(?:ます|ません|ないか)|間違っていませんか"
    r"|のはずです|ではありませんか|ませんでしたか"
    r"|(?:計算|回答|答え)しましたよね",
)


#: 正しい値を **述べている** 印。振り返りの問いと本物の訂正を分ける。
_ASSERTS_CORRECT_VALUE_RE = re.compile(r"正しくは|ではなく|じゃなく|が正しい")


#: 帰属の根拠が何も無かった (末尾の既定で ``assistant`` に倒した) ことを示す根拠名。
#: 判定点 ``correction_attribution`` はこれを **棄権** として記録する (不変則 #14)。
ATTRIBUTION_DEFAULT_EVIDENCE = "no_attribution_evidence"

#: 「同じ出力のやり直し」を頼む形として訂正から外した根拠。外す前提は「直前の
#: 出力は正しかった」なので、直前ターンが実際に失敗していれば前提が崩れ、訂正に
#: 戻す (:meth:`FeedbackCollector._detect_correction`)。仮定の変更・質問・本人の
#: 言い直しは直前の成否と無関係に訂正ではないので含めない。
REDO_SAME_OUTPUT_EVIDENCE: frozenset[str] = frozenset({
    "referential_write_edit", "reformat_request",
})


def _names_prior_value(text: str, value: str) -> bool:
    """``text`` が値 ``value`` を逐語で含むか (空白無視・全角半角同一視)。

    数字で始まる / 終わる値は数字の境界で見る — 「15」を「2015年」に当てない。
    """
    v, t = norm_span(value), norm_span(text)
    if not v or v not in t:
        return False
    if not (v[0].isdigit() or v[-1].isdigit()):
        return True
    return re.search(r"(?<![0-9.,])" + re.escape(v) + r"(?![0-9.,])", t) is not None


#: 帰属の文脈に入れる前のユーザー発話の件数。学習側 (経験バッファの同じセッション) と
#: 応答パス (履歴) で同じ範囲を見る (#14(a))。訂正の宛先解決の窓と揃える。
ATTRIBUTION_CONTEXT_TURNS = CORRECTION_LOOKBACK


def attribution_prev_user(utterances: list[str]) -> str:
    """訂正より前のユーザー発話 (古い順) から帰属の文脈 ``prev_user`` を組む (純粋関数)。

    直近 :data:`ATTRIBUTION_CONTEXT_TURNS` 件を改行で連結する。学習側と応答パスが
    同じこの関数を通す — 範囲が読み手ごとに違うと、同じ発話の帰属が食い違う。
    """
    recent = [u for u in utterances if u][-ATTRIBUTION_CONTEXT_TURNS:]
    return "\n".join(recent)


def _own_statements(text: str) -> list[str]:
    """本人の申告として数える文 (平叙で、仮定・時間の対比・伝聞の標識が無い文)。"""
    return [
        s for s in split_sentences(mask_quoted_speech(text or ""))
        if is_plain_statement(s) and not marks_not_own_restatement(s)
    ]


def _prior_value_owner(
    query: str, *, prev_user: str, prev_response: str, prev_query: str = "",
) -> tuple[str, str] | None:
    """対比「X ではなく Y」の旧値 X を **誰が先に述べたか** で帰属を決める (純粋関数)。

    - X が直前のアシスタント応答に在る:
      - その応答が答えたユーザー発話 (``prev_query``) の申告の文にも X が在る →
        ``self`` (応答は本人の申告の復唱・受け取りで、アシスタントの主張ではない)
      - そうでない (問い・依頼に答えた応答が X を述べた) → ``assistant``。本人が
        以前に X を述べていても、答えとしての X はアシスタントの主張
        (「予算は10万円です」→「残りは？」→「10万円です」→「10万円ではなく5万円」)
    - X が応答に無く、前のユーザー発話 (``prev_user``) の申告の文に在る → ``self``
      (本人の値の言い直し・計画の変更。D01「やっぱり妻ではなく母と…」)
    - どちらにも無い / 対比が無い → ``None`` (構造では決まらない)

    申告の文は平叙で、仮定・時間の対比・伝聞の標識 (:func:`marks_not_own_restatement`)
    の無い文だけ (問い・仮定の X は本人の値ではない)。``prev_query`` を渡さない読み手は
    応答の X を答えか復唱か区別できないので、本人の申告を先に見る。旧値の区間に
    区切りの無い前置きが入る形 (「やっぱり妻」) は、区間のままでどこにも当たらない
    ときだけ :func:`old_value_core` の形で当てる。分解は :func:`contrast_pairs` の
    1 実装 (不変則 #14a)。
    """
    if not prev_user and not prev_response:
        return None
    statements = _own_statements(prev_user)
    answered = _own_statements(prev_query)
    for old, _new in contrast_pairs(mask_quoted_speech(query or "")):
        # 前置きを外した形は、区間のままでどちらにも当たらないときだけ使う
        # (「ほうじ茶」の「茶」を本人の「緑茶」に当てない)。
        for value in dict.fromkeys((old, old_value_core(old))):
            in_response = _names_prior_value(prev_response or "", value)
            if in_response and prev_query:
                if any(_names_prior_value(s, value) for s in answered):
                    return "self", "own_prior_value"
                return "assistant", "assistant_prior_claim"
            if any(_names_prior_value(s, value) for s in statements):
                return "self", "own_prior_value"
            if in_response:
                return "assistant", "assistant_prior_claim"
    return None


def correction_attribution_reason(
    query: str, *, prev_user: str = "", prev_response: str = "", prev_query: str = "",
) -> tuple[str, str]:
    """訂正候補の帰属と **その根拠名** を返す (純粋関数)。

    帰属は ``assistant`` / ``self`` / ``not_correction``、根拠名は規則の識別子
    (英語。``decision.jsonl`` の reason)。字句の規則で決まらないとき、文脈
    (``prev_user`` = 訂正より前のユーザー発話 (:func:`attribution_prev_user` で範囲を
    揃える)、``prev_response`` = 直前のアシスタント応答、``prev_query`` = その応答が
    答えたユーザー発話) があれば対比の旧値の出所 (:func:`_prior_value_owner`) で決め、
    それも無ければ ``("assistant", ATTRIBUTION_DEFAULT_EVIDENCE)`` に倒す
    (判別できないものは ``assistant`` — 従来の挙動)。
    """
    # 「過去の誤りを **尋ねる**」形は、誰の出力を指していようと訂正ではない。
    # 出力参照より先に見る — 「私があなたの回答を訂正させたのは何回で…」は
    # アシスタント出力への言及を含むが、振り返りの問いであって指摘ではない。
    # ただし正しい値を併せて述べている場合は本物の訂正なので除外しない
    # (「あなたの回答は間違いです。正しくは 2027-01-06 です」)。
    if _ASKS_ABOUT_CORRECTION_RE.search(query) and not _ASSERTS_CORRECT_VALUE_RE.search(query):
        return "not_correction", "asks_about_correction"
    # アシスタント出力への言及。「すみません、その計算は違います」のように
    # 謝罪語と併存しうるため、自己訂正判定より先に見る。
    if _ASSISTANT_OUTPUT_REF_RE.search(query):
        return "assistant", "assistant_output_ref"
    if _SELF_CORRECTION_RE.search(query):
        return "self", "self_correction_form"
    if _RECALL_QUESTION_RE.search(query):
        return "not_correction", "recall_question"
    if _ASKS_ABOUT_DIFFERENCE_RE.search(query):
        return "not_correction", "asks_about_difference"
    # 正しい値を併せて述べていれば本物の訂正 (「違うのです。正しくは 10.95 度です」)。
    if _HEARSAY_DIFFERENCE_RE.search(query) and not _ASSERTS_CORRECT_VALUE_RE.search(query):
        return "not_correction", "hearsay_difference"
    # 複合動詞の「違」を除いた残りに訂正語彙が無ければ、字句一致は複合動詞
    # だけだったということ。
    stripped = _COMPOUND_DIFFERENCE_RE.sub("", query)
    if stripped != query and not any(p.search(stripped) for p in CORRECTION_PATTERNS):
        return "not_correction", "compound_difference"
    if _REFORMAT_REQUEST_RE.search(query):
        return "not_correction", "reformat_request"
    if _PREMISE_CHANGE_REDO_RE.search(query):
        return "not_correction", "premise_change_redo"
    # 既存ファイルへの再保存を伴う依頼は編集であって訂正ではない。
    # 「3 番目を『ヘッドランプ』に直して、同じファイルに保存し直して」の
    # 「直して」が CORRECTION_PATTERNS に掛かるのを打ち消す。
    #
    # ただし内容への異議 (「プログラムではなくて文書が欲しいです」) を伴う場合は
    # 除外しない。編集依頼の体裁でも中身は訂正であり、除外すると本物の訂正を
    # 取りこぼす (既存テスト test_same_target_weak_pattern_detected の実例)。
    if REFERENTIAL_WRITE_TARGET_RE.search(query) and not any(
        p.search(query) for p in WEAK_CORRECTION_PATTERNS_ALL
    ):
        return "not_correction", "referential_write_edit"
    # 語形の規則で決まらない対比は、旧値を誰が先に述べたかで決める
    # (2026-10-02 監査 D01#4「やっぱり妻ではなく母と行くことになりました」が
    # 既定の assistant に倒れていた。妻は本人の申告で、アシスタントは復唱しただけ)。
    owner = _prior_value_owner(
        query, prev_user=prev_user, prev_response=prev_response, prev_query=prev_query,
    )
    if owner is not None:
        return owner
    return "assistant", ATTRIBUTION_DEFAULT_EVIDENCE


def classify_correction_target(
    query: str, *, prev_user: str = "", prev_response: str = "", prev_query: str = "",
) -> str:
    """訂正候補の帰属を返す: ``assistant`` / ``self`` / ``not_correction``。

    純粋関数。``assistant`` のみが「直前のアシスタント応答が誤っていた」を
    意味する。判別できないものは ``assistant`` に倒す (現行挙動を維持)。
    文脈 (``prev_user`` / ``prev_response``) の使い方と根拠名は
    :func:`correction_attribution_reason`。
    """
    return correction_attribution_reason(
        query, prev_user=prev_user, prev_response=prev_response, prev_query=prev_query,
    )[0]


# 弱い訂正パターン: 単独では新規依頼との区別がつかないため、直前ターンの
# 失敗または同一成果物 (同じ出力先パス) の再指定を伴う場合のみ訂正とみなす。
WEAK_CORRECTION_PATTERNS = [
    re.compile(r"(?:では|じゃ)な[くい]"),
    re.compile(r"また.{0,15}(?:になって|なって)"),
]

# WEAK_CORRECTION_PATTERNS の英語版。
WEAK_CORRECTION_PATTERNS_EN = [
    re.compile(r"\bnot\s+\w+\s+but\b|\binstead\s+of\b", re.IGNORECASE),
    re.compile(r"\bit'?s\s+.{0,15}\bagain\b", re.IGNORECASE),
]

#: JA / EN 双方を **locale に関わらず** 評価する union。消費側は 2 つあり、
#: 以前は ``classify_correction_target`` が JA 固定・``_detect_correction_lexical``
#: が locale 選択という **食い違った参照** をしていた。同じ弱訂正語彙で
#: 「訂正ではない」の打ち消しと「訂正である」の確定が別々の集合を見ていたことになる。
WEAK_CORRECTION_PATTERNS_ALL = [
    *WEAK_CORRECTION_PATTERNS, *WEAK_CORRECTION_PATTERNS_EN,
]

# ── 記録との食い違いの指摘 (2 条件の AND) ────────────────────────────
#
# 「アシスタントの言い分」と「自分が言ったこと」が食い違うと指摘する形は、
# 誤りを名指す語 (違う / 間違い / 訂正) を **一つも含まない** ことがある。
# 実インシデント (2026-08-16 ライブ監査 ターン39):
#   「えっ、私が紅茶派って言った？私はコーヒーを1日3杯飲むって言ったはずだけど。
#    どっちが正しい？」
# 応答自体は正しく訂正できたのに、学習側は correction=False /
# correction_detected_by=null で取りこぼした (40 ターン中 訂正シグナル 0 件)。
#
# 単独ではどちらも一般語法なので **両方** を要求する。片方だけだと
# 「私が言ったとおりに実装して」(前者のみ) や
# 「Python と Go はどっちが正しい書き方ですか」(後者のみ) を巻き込む。
#: (a) ユーザーが「自分は何と言ったか」を引き合いに出す。
_USER_STATED_REF_RE = re.compile(
    r"(?:私|僕|俺|自分)(?:が|は|の).{0,24}?"
    r"(?:言(?:った|いました|ってた|ってました)|話(?:した|しました)"
    r"|伝え(?:た|ました))",
)
#: (b) 記録との食い違いを述べる / どちらが正しいかを問う。
_RECORD_DIVERGENCE_RE = re.compile(
    r"はず(?:だけど|ですけど|ですが|だが|なんだけど|なんですけど)"
    r"|どっち(?:が|は)?\s*正し|どちら(?:が|は)?\s*正し"
    r"|(?:って|と)言(?:った|いました)(?:っけ|か)?[?？]",
)

_USER_STATED_REF_RE_EN = re.compile(
    r"\bi\s+(?:said|told\s+you|mentioned)\b", re.IGNORECASE,
)
_RECORD_DIVERGENCE_RE_EN = re.compile(
    r"\bwhich\s+(?:one\s+)?is\s+(?:right|correct)\b"
    r"|\bdid\s+i\s+(?:say|tell)\b"
    r"|\bi\s+(?:said|told\s+you)\b.{0,30}\b(?:though|but)\b",
    re.IGNORECASE,
)


#: 対比による言い直し (「X ではなく Y **でした**」)。**記憶層だけ**が使う。
#:
#: ``CORRECTION_PATTERNS`` は裸の ``〜ではなく`` を意図的に外している —
#: 話題導入・比較の一般語法 (「これはバグではなく仕様です」「Python ではなく
#: Go で書いてください」「A ではなく B を使うべきだと思います」) と識別できず、
#: 見送った実績がある (同リストの冒頭コメント)。その判断は正しい。
#:
#: ただし **過去形の断定で文が終わる** 形は別で、「記録されている値は実は Y
#: だった」以外の読み方が無い。実測 (2026-08-29 ライブ監査の追調査):
#:
#: ```
#: False  すみません、好きな飲み物はコーヒーではなく紅茶でした。
#: False  猫の名前はミケではなくトラでした。
#: True   違います、ほうじ茶です。
#: True   訂正です。猫の名前はトラです。
#: ```
#:
#: **最も典型的な訂正形が最も弱い**状態だった。``from_correction`` が立たない
#: ため、属性の継承も即 supersede も走らない。
#:
#: ``CORRECTION_PATTERNS`` へは足さない。あちらは学習の欠陥シグナル
#: (``FeedbackCollector._detect_correction``) と共有で、この形は
#: ``classify_correction_target`` が ``assistant`` を返すことがあり、
#: **ユーザー自身の言い直しをアシスタントの失敗として数えてしまう**。
#: 記憶層と学習層で必要な範囲が違うという既存の設計 (:func:`restates_a_value`
#: の説明) に従い、記憶側だけを広げる。
#:
#: **文ごとに当てる** (:func:`restates_a_value`)。``$`` は発話全体ではなくその文の
#: 末尾 — 訂正の後に問いや依頼が続く形 (「…5歳でした。散歩時間の目安は
#: 変わりますか？」) を取りこぼしていた (2026-09-26 監査 #13)。
_CONTRASTIVE_RESTATEMENT_RE = re.compile(
    CONTRAST_MARKER + r"[^。！？!?\n]{1,24}(?:でした|だった)[。．.！!\s]*$",
)

#: 値の **変更の告知** (完了形)。「報告会の日が変わりました」「予定が変更に
#: なりました」は誤りの指摘ではないが、その属性の現在値が入れ替わったという
#: 宣言で、記憶層にとっては言い直しと同じ (2026-09-12 (b): 対比形が現在形
#: 「ではなく…です」で past-only の網に掛からず、前倒しが立たないまま別セッション
#: へ旧日付が注入された)。疑問形 (「変わりますか」「変わったら」) は除く。
_VALUE_CHANGE_CORE = (
    r"(?:変わり|変更にな|変更され|延期にな|前倒しにな|延び|早ま|ずれ)"
    r"(?:ました|りました|った|た)(?![らかの]|ら)"
)
_VALUE_CHANGED_RE = re.compile(r"(?:が|は)\s*" + _VALUE_CHANGE_CORE)
#: 変更告知の **動詞核** だけ。対比語を含む文 (「2泊3日ではなく1泊2日に変更に
#: なりました」) では対比が「何から何へ」を示すので、主語の ``は/が`` を要求しない。
_VALUE_CHANGE_CORE_RE = re.compile(_VALUE_CHANGE_CORE)
_QUESTION_TAIL_RE = re.compile(r"(?:か|の|でしょう)?[?？]\s*$|(?:ますか|ですか|でしょうか)[。．]?\s*$")


def _correction_attribution(
    query: str, *, prev_user: str = "", prev_response: str = "", prev_query: str = "",
) -> str | None:
    """字句一致した訂正候補の **帰属** を返す。訂正でなければ ``None``。

    「訂正の言い回しが出ているか」(字句) と「誰が誤っていたか」(帰属) を 1 箇所に
    まとめる。下の 2 つの公開述語はここから作る — **記憶層と学習層で必要な
    「訂正」の範囲が違う**ので、判定の芯だけを共有して境界だけ分ける。

    戻り値は :func:`classify_correction_target` と同じ ``assistant`` /
    ``self`` / ``not_correction``。

    引用 (鉤括弧) の内側は本人の主張ではないので、字句照合の前に
    :func:`mask_quoted_speech` で落とす — 「営業から『色が違う』という
    クレーム」の「違う」で候補を立てない (2026-09-09 監査 G-01 系)。

    文脈 (``prev_user`` / ``prev_response``) は :func:`correction_attribution_reason`
    へそのまま渡す (対比の旧値の出所で帰属を決める)。
    """
    if not query:
        return None
    masked = mask_quoted_speech(query)
    lexical = any(p.search(masked) for p in CORRECTION_PATTERNS) or (
        cites_record_divergence(masked)
    )
    if not lexical:
        return None
    return classify_correction_target(
        masked, prev_user=prev_user, prev_response=prev_response, prev_query=prev_query,
    )


def points_at_assistant_error(
    query: str, *, prev_user: str = "", prev_response: str = "", prev_query: str = "",
) -> bool:
    """この発話が **アシスタントの誤りの指摘** か (純粋関数)。

    **応答パス** (``core.inference._correction_target_note``) が使う述語。
    「[訂正の対象]」の注記は、訂正が指す過去の回答を検証し直させるものなので、
    帰属が ``assistant`` の候補にだけ付ける。以前は注記側が
    :data:`~backend.free.core.correction_target.WRONG_MARKER_RE` (誤りの側の
    span を切るための印) を候補判定に流用しており、「数字だけで」の
    ``だけで`` に当たった想起の問い「最初に私が挙げたドル建ての請求額は
    いくらでしたか。数字だけで。」に「2 つ前の回答を検証し直せ」が付いて、
    ドル額 (2,400) ではなく円額 (348,000) を返した (2026-09-09 ライブ監査
    B-01)。記録側 (``_detect_correction``) はこの発話を候補にしていない —
    同じ判定を 2 箇所で別々に実装していたのが原因なので、芯を共有する。
    文脈 (``prev_user`` / ``prev_response``) も記録側と同じものを渡す。
    """
    return _correction_attribution(
        query, prev_user=prev_user, prev_response=prev_response, prev_query=prev_query,
    ) == "assistant"


def restates_a_value(query: str) -> bool:
    """この発話が **ユーザー自身の値の言い直し** か (純粋関数)。

    **記憶層** (SemMem のスロット更新) が使う述語。``assistant`` (アシスタントの
    誤りの指摘) と ``self`` (ユーザー自身の申告訂正) の両方を拾う。

    学習の欠陥シグナルより広いのは、両者で必要な意味が違うため:

    - 学習は「アシスタントが誤ったか」を数える。「すみません、火曜ではなく水曜の
      間違いでした」はアシスタントの応答が正しいので **欠陥ではない**
      (``FeedbackCollector._detect_correction`` が ``self`` を落とす)。
    - 記憶は「その属性の現在値が何か」を持つ。上の発話は **正当な値更新** で、
      落とすと古い値が live のまま残る。

    ``not_correction`` (訂正について尋ねる質問 / 比較質問 / 書式変更依頼 /
    既存ファイルへの編集依頼) は両者とも対象外。

    用途: チャット応答パスが ``WorkingMemory.add_turn(correction=...)`` へ渡し、
    ``MemoryNote.is_correction`` → ``SemanticFact.from_correction`` と伝播する。
    伝播先は 2 つ —

    1. ``ChatExtractor`` が **直前の名前付き属性を継承**して、訂正が対象と
       同じスロットへ入るようにする (継承しないと「違います、ほうじ茶です」の
       ように属性語を含まない訂正が ``mem.*.user`` へ落ち、競合検出が対に
       できない。実測 2026-08-19: 訂正済みの「緑茶」が sim 0.762 で最上位、
       訂正後の「ほうじ茶」が 0.487 で下位に並んでいた)
    2. ``SemanticConflictResolver._decide`` が確認を挟まず即 supersede する

    ``CORRECTION_PATTERNS`` に加えて、**記憶層だけ** が
    :data:`_CONTRASTIVE_RESTATEMENT_RE` (「X ではなく Y でした」) を拾う。
    最も典型的な訂正形なのに、裸の ``ではなく`` が一般語法と混ざるため
    共有リストからは外されていた (同定数の説明を参照)。
    """
    if _correction_attribution(query) in ("assistant", "self"):
        return True
    if not query:
        return False
    masked = mask_quoted_speech(query)
    # 文ごとに見る (2026-09-26 監査 #13)。対比形の文末・変更告知の疑問尾は
    # その文の末尾で判定する — 訂正の後に問いや依頼が続くのは普通の形。
    contrastive = False
    for sentence in split_sentences(masked):
        asks = _QUESTION_TAIL_RE.search(sentence)
        if _VALUE_CHANGED_RE.search(sentence) and not asks:
            return True
        if _CONTRASTIVE_RESTATEMENT_RE.search(sentence) or (
            not asks
            and contrast_pair(sentence) is not None
            and _VALUE_CHANGE_CORE_RE.search(sentence)
        ):
            contrastive = True
    if not contrastive:
        return False
    # 帰属の判定 (質問 / 比較 / 書式変更依頼を落とす) は共有経路と同じものを通す。
    return classify_correction_target(masked) in ("assistant", "self")


def cites_record_divergence(query: str) -> bool:
    """「自分はこう言ったはず」と記録の食い違いを指摘しているか (純粋関数)。

    誤りを名指す語を含まない訂正を拾うための 2 条件 AND
    (:data:`_USER_STATED_REF_RE` / :data:`_RECORD_DIVERGENCE_RE` の説明を参照)。

    JA / EN は locale で切り替えず両方見るが、union は **AND の外側** で取る —
    片側ずつ or にすると「日本語の (a) + 英語の (b)」のような跨ぎ一致が成立し、
    2 条件 AND で誤検出を抑えている設計 (同上) が緩む。
    """
    return bool(
        (_USER_STATED_REF_RE.search(query) and _RECORD_DIVERGENCE_RE.search(query))
        or (
            _USER_STATED_REF_RE_EN.search(query)
            and _RECORD_DIVERGENCE_RE_EN.search(query)
        ),
    )

# 応答の失敗マーカー (meta_cognitive の最終応答フォーマット "- [failed] ...")
_FAILED_MARKER_RE = re.compile(r"(?:^|\n)\s*-\s*\[failed\]", re.IGNORECASE)
_DONE_MARKER_RE = re.compile(r"(?:^|\n)\s*-\s*\[done\]", re.IGNORECASE)

# クエリ中の明示的な出力先パス (同一成果物の再指定検出用)。
# 定義は core.intent_vocab が SSOT (agent.meta_cognitive が同一定義を持っていた)。
_QUERY_PATH_RE = EXPLICIT_WINDOWS_PATH_RE

# ── 言い換え (同じ質問の言い直し) の検出 ──────────────────────────
#
# 旧実装は **文字集合の Jaccard** (順序も出現回数も無視) で閾値 0.5 だった。
# 日本語は助詞・語尾・句読点の文字が共通するため、同じテンプレートの別質問が
# 必ず閾値を超える。しかも ``rephrased_query`` は Level 1 の欠陥重み 0.6 を持ち、
# **選択圧の主成分**になっている。
#
# 実測 (2026-08-18、経験 136 件 / 旧実装が言い換えと判定した 17 組を全数確認):
#
#   真の言い直し   1 件 (完全同文の再入力)
#   別の質問       1 件 「あなたの名前を教えて」→「あなたの得意なことを教えて」
#   深掘り        15 件 「Xを3行で教えて」→「Xを、Yに絞って3行で教えて」
#
#   欠陥重みの内訳 13.2 = rephrased 17×0.6 + user_correction 3×1.0
#   → 9.6 (73%) が誤検出由来
#
# 指標を 2 つに分ける:
#
# 1. **深掘りの除外** — 前の発話がほぼそのまま残り、新しい語が足された形。
#    ``bigram_coverage`` (非対称) と長さ比の AND で見る。実測の分離:
#      深掘り   coverage 0.929〜0.955 / 長さ比 1.45〜1.73
#      言い直し coverage 0.857        / 長さ比 1.27
#      別の質問 coverage 0.667        / 長さ比 1.30
# 2. **類似度** — 日本語は **内容語だけ** の bi-gram コサインで測る
#    (``content_bigram_cosine``)。生のコサインでは機能語が支配的になり、
#    真偽が逆転する:
#
#      言い直し 「Pythonのリスト操作を教えて」→「Pythonでリストの操作方法は？」
#               生 0.516 / 内容語 0.870
#      別の質問 「あなたの名前を教えて」→「あなたの得意なことを教えて」
#               生 0.577 / 内容語 0.000
#      別の質問 「欠損値の扱いを3行で教えて」→「外れ値の検出を3行で教えて」
#               生 0.706 / 内容語 0.333
#
#    内容語で測ると 真 0.866〜1.000 / 偽 0.000〜0.333 / 深掘り 0.589 に分離する。
#
#: 深掘りとみなす coverage の下限。
_DRILLDOWN_MIN_COVERAGE = 0.90
#: 深掘りとみなす長さ比 (現/前) の下限。
_DRILLDOWN_MIN_LENGTH_RATIO = 1.30
#: 言い直しとみなす内容語 bi-gram コサインの下限 (日本語)。
#: 実測の真 (0.866+) と 深掘り (0.589) / 偽 (0.333-) の間に置く。
REPHRASE_THRESHOLD = 0.70
#: 英語ロケールの閾値。内容語の抽出はひらがな前提なので英語では効かず、生の
#: bi-gram コサインで測る。**ラベル付きの英語標本が無いため未較正** で、旧実装の
#: 実効水位 (0.5) をそのまま置いている (指標は集合 Jaccard より厳密になっている)。
REPHRASE_THRESHOLD_EN = 0.5

# オウム返し (応答がユーザー発話と同一) 判定の最小文字数。これ未満は
# 「こんにちは」→「こんにちは」のような正当な同語応答があり得るため
# 対象外にする。
_ECHO_MIN_CHARS = 16


def _is_user_echo(query: str, response: str) -> bool:
    """応答がユーザー発話のオウム返しかを判定する。

    ベースモデルが短い訂正ターン等でユーザー発話をそのまま復唱することが
    ある (2026-07-26 ライブ検証: 「すみません、火曜ではなく水曜の間違い
    でした。時間はそのままです。」に対し全く同一の応答)。これは応答として
    失敗だが、``[failed]`` マーカーも step_credits も無いため従来は
    ``success`` として経験記録され、Level 1 fitness / learned_patterns の
    正例に混ざっていた。空白差だけを無視した完全一致を失敗として扱う。
    """
    q = "".join((query or "").split())
    r = "".join((response or "").split())
    if len(q) < _ECHO_MIN_CHARS or len(r) < _ECHO_MIN_CHARS:
        return False
    return q == r


def is_short_negative_feedback(query: str) -> bool:
    """発話が「違う」「だめ」「not that」型の短い否定 **だけ** か (純粋関数)。"""
    text = (query or "").strip()
    if not text or len(text) > _SHORT_NEGATIVE_MAX_CHARS:
        return False
    return bool(SHORT_NEGATIVE_FEEDBACK_RE.match(text))


#: セッション別に保持する直前ターンの状態の上限 (LRU)。並行セッションは
#: 高々数本なので tool_ledger と同じ 16 で足りる。
_SESSION_STATE_CAP = 16

#: 訂正の宛先解決で遡る同一セッションのターン数。
#: ``core.correction_target.DEFAULT_LOOKBACK`` と揃える — 窓の方が狭いと
#: 解決側が遡れる範囲を実質的にここが決めてしまい、両方を読まないと挙動が
#: 分からなくなる。
#:
#: 候補は **経験バッファから** 引く (``_correction_candidates``)。以前は
#: ``_SessionTurnState.recent_turns`` に持たせていたが、それは
#: ``_SESSION_STATE_CAP`` (16) の LRU に乗るので、17 本目の会話が来た時点で
#: 古い会話の窓が丸ごと消え、訂正の宛先が解決できなくなった
#: (2026-09-07 ライブ監査: 20 会話を回した後の訂正 13 件のうち 7 件が候補ゼロ。
#: しかも訂正を送る行為自体が LRU を押し出すので、追い出し順が実測と一致した)。
#: バッファは全セッション横断で 1000 件持ち session_id を各エントリが持つので、
#: LRU に依存せず同一セッションの直近ターンを引ける。
_RECENT_TURN_WINDOW = CORRECTION_LOOKBACK


@dataclass
class _SessionTurnState:
    """``FeedbackCollector`` がセッションごとに持ち回る直前ターンの状態。

    以前はこれらがプロセス全体で 1 組しか無く、並行セッションが交互に
    record すると別セッションの直前ターンと突き合わせて訂正 / 言い直しを
    誤検出していた (2026-09-02 監査 R-B4)。
    """

    prev_query: str | None = None
    prev_entry: ExperienceEntry | None = None
    prev_routed_tool: bool = False
    prev_used_long_form: bool = False
    prev_turn_failed: bool = False
    pending_correction: dict | None = None
    prev_response: str = ""


#: 失敗理由の接頭辞 → (検証器 id, 台帳の種別)。理由文字列は
#: ``_derive_turn_outcome_with_reason`` が組む。
_OUTCOME_REASON_CHANNELS: tuple[tuple[str, str, str], ...] = (
    ("arithmetic contradiction", "content.arithmetic", "content_contradiction"),
    ("conclusion contradiction", "content.conclusion", "content_contradiction"),
    ("sign contradiction", "content.sign", "content_contradiction"),
    ("change rate contradiction", "content.arithmetic", "content_contradiction"),
    ("broken JA spacing", "content.broken_text", "output_broken"),
    ("Chinese token leaked", "content.broken_text", "output_broken"),
    ("response retracts", "content.self_retraction", "content_contradiction"),
    ("measured value contradiction", "content.measured", "content_contradiction"),
    ("tool result ignored", "content.tool_result", "tool_result_ignored"),
    ("date result ignored", "content.date_result", "tool_result_ignored"),
    ("claimed completion while blocked", "content.claimed_change", "content_contradiction"),
    ("user echo", "content.user_echo", ""),
    ("fabricated count", "content.fabricated_count", "content_contradiction"),
    ("fabricated entity", "content.fabricated_entity", "content_contradiction"),
    ("declared count mismatch", "content.declared_count", "content_contradiction"),
    ("false tool unavailability", "content.tool_unavailable", "content_contradiction"),
)


def rag_abstain_kind(
    gen_config: GenerationConfigRef | None,
    abstained_on_shown: bool | None,
    response: str,
) -> str | None:
    """抑止応答の型を決める (決定論。件数と id 集合の関係だけで分ける)。

    - ``shown``: corpus を ``[参考情報]`` に見せたのに差し控えた (``rag_abstained`` が真)
    - ``not_shown``: 検索は corpus を採用したが、資格判定・重複・予算で 1 件も見せていない
    - ``not_retrieved``: corpus を 1 件も採用していない (候補 0 / 棒を通らない / ゲート)

    差し控えていない応答と、corpus を見せておらず検索も通っていないターン
    (``corpus_gated`` が ``None``) は ``None``。
    """
    if abstained_on_shown is not None:
        return RAG_ABSTAIN_SHOWN if abstained_on_shown else None
    if gen_config is None or gen_config.corpus_gated is None:
        return None
    if not abstains_on_reference_material(response):
        return None
    adopted = int((gen_config._extra or {}).get(RAG_ADOPTED_CORPUS_KEY) or 0)
    return RAG_ABSTAIN_NOT_SHOWN if adopted > 0 else RAG_ABSTAIN_NOT_RETRIEVED


def _publish_turn_outcome(outcome: str, reason: str | None) -> None:
    """導出した成否を結末 JSONL (verifier scope) と自己申告の台帳へ流す。

    長さ / 形式の違反は ``text_quality`` 側が既に記録している (二重計上しない)。
    """
    record_turn_outcome(outcome, reason)
    if outcome != "failed" or not reason:
        return
    for prefix, verifier_id, kind in _OUTCOME_REASON_CHANNELS:
        if reason.startswith(prefix):
            record_verifier_hit(verifier_id)
            if kind:
                record_current_issue(kind, reason)
            return


#: 依頼の内容語 (問われた量の名詞: 「平均給与」「売上」) を取り出す連なり (漢字・カタカナ)。
_ANSWER_SLOT_RUN_RE = re.compile(rf"[{KANJI}{KANJI_MARKS}{KATAKANA_WORD}]{{2,}}")
#: 数の直前でこの範囲 (文字) に問われた名詞があれば、その数は答えの位置にある。
_ANSWER_SLOT_WINDOW = 16


def _numbers_not_in(response: str, grounded: str, query: str = "") -> list[str]:
    """答えの位置にある数のうち、依頼・会話・ツールの結果 (``grounded``) に無い数の表記
    (出現順、最大 3 件)。

    数えるのは **問われた量の名詞の直後** (:data:`_ANSWER_SLOT_WINDOW` 文字以内) に
    ある数だけ — 「営業部の平均給与は 520,000 円」の 520,000。見つからないと正しく答えた
    応答の数 (「HTTP 404 相当」「ポート 8000 のサーバ」「12 個のフォルダを探した」) は
    答えではないので数えない。パスの中の数字と 1 桁の数も数えない。判断できなければ
    空 (呼出側はラベル無しのまま)。
    """
    from backend.free.agent.tool_judge_args import without_drive_paths

    slots = {run for run in _ANSWER_SLOT_RUN_RE.findall(query or "")}
    if not slots:
        return []
    known = {round(n.value, 6) for n in iter_ja_numbers(grounded or "")}
    text = without_drive_paths(response or "", " ")
    out: list[str] = []
    for number in iter_ja_numbers(text):
        if number.value < 10 or round(number.value, 6) in known:
            continue
        window = text[max(0, number.start - _ANSWER_SLOT_WINDOW):number.start]
        if not any(slot in window for slot in slots):
            continue
        literal = text[number.start:number.end].strip()
        if literal and literal not in out:
            out.append(literal)
        if len(out) >= 3:
            break
    return out


class FeedbackCollector:
    """暗黙的フィードバックシグナルを収集し経験バッファに記録

    学習済みパターンストアが設定されている場合、ツールルーティング / 長文
    ルーティングのシグナルからパターンを学習する (訂正・言い直しからの学習は
    2026-07-21 に廃止 — ``_detect_correction`` の docstring 参照)。
    """

    def __init__(
        self,
        experience_buffer: ExperienceBuffer,
        debug_logger: DebugLogger | None = None,
        learned_patterns: LearnedPatternStore | None = None,
        disabled: bool = False,
        base_model_name: str = "",
        embedding_model_name: str = "",
    ) -> None:
        self.buffer = experience_buffer
        self._debug_logger = debug_logger
        self._learned_patterns = learned_patterns
        self._prev_query: str | None = None
        # 直前ターンの entry と capability 使用状況 (false_negative の事後検出用)。
        # 「前ターンが capability 未使用 → 当ターンで明示訂正 → 当ターンで capability
        # 使用」の遷移を検出したら前 entry へ遡及マークし、前クエリから学習する。
        self._prev_entry: ExperienceEntry | None = None
        self._prev_routed_tool: bool = False
        self._prev_used_long_form: bool = False
        # 直前ターンの成否 ([failed] マーカー等から導出)。失敗直後のターンは
        # 無条件で訂正候補とみなす (2026-07-15: 訂正 2 ターンが検出漏れ)。
        self._prev_turn_failed: bool = False
        # 値が食い違う訂正の「保留」。訂正の検出時点では真偽が分からないため、
        # 次のターンでアシスタントが元の値を維持したら撤回する
        # (``_settle_pending_correction`` の説明を参照)。
        self._pending_correction: dict | None = None
        # 直前ターンのアシスタント応答 (保留判定の材料)。
        self._prev_response: str = ""
        # 上記 ``_prev_*`` / ``_pending_correction`` は「いま record 中の
        # セッション」の作業コピー。record() の入口でセッションの状態を載せ、
        # 出口で書き戻す (``_load_session_state`` / ``_store_session_state``)。
        # セッション未指定 ("") は単一セッション運用として 1 枠に畳む。
        self._sessions: OrderedDict[str, _SessionTurnState] = OrderedDict()
        # 現在ロード中のモデル名 (GGUF ファイル名、表示用)。record() の base_model /
        # embedding_model が明示指定されないとき既定値として埋める。
        #
        # base_model は **モードで変わる**: create は model_paths.create_model を
        # ロードする。_base_model_name は chat 既定として保持し、record() 時に mode
        # から解決する。Level 2 の経験の絞り込みは model_key で行う
        # (:meth:`_resolve_model_key`)。
        self._base_model_name = base_model_name
        self._embedding_model_name = embedding_model_name
        # 現会話セッションで record した entry の参照。会話終了時に
        # mark_conversation_ended() がまとめて conversation_ended=True にする。
        self._session_entries: list[ExperienceEntry] = []
        # 自己学習無効化フラグ (--no-learning 経由)。True の場合 record() は
        # シグナル検出も ExperienceBuffer 書込も行わずダミーの ExperienceEntry を返す
        self._disabled = disabled
        if disabled:
            logger.info(
                "FeedbackCollector initialized in disabled mode "
                "(Level 0 experience record is no-op)",
            )

    def rebind_base_model(self, base_model_name: str) -> None:
        """ランタイム base 切替で chat 既定のモデル名 (GGUF ファイル名) を差し替える。

        以後の ``record()`` は新モデル名を ``base_model`` に刻む
        (``_learning_rebind.rebind_base_learning`` から呼ばれる)。
        """
        self._base_model_name = base_model_name

    def _resolve_base_model_name(self, mode: str) -> str:
        """記録時のモードで実際にロードされている base モデルの GGUF 名を返す。

        表示用の名前で、照合には使わない (照合は :meth:`_resolve_model_key`)。
        create は ``model_paths.create_model`` を読み込む。解決経路はモード切替が使う
        ``get_mode_generation_params`` に揃える (実際にロードされるモデルと
        刻む名前を同じ関数から採る)。

        config 未初期化などで解決できない場合は起動時に決めた既定
        (``_base_model_name``) へフォールバックし、記録自体は止めない。
        llama-server が実際に載せているモデルが分かればその名前 (:meth:`_resolve_model_key`
        と同じ根拠)。
        """
        try:
            from backend.config import get_path_resolver

            served = get_path_resolver().served_model_path()
        except Exception:  # noqa: BLE001 — resolver 未ロードなら宣言から引く
            served = ""
        if served:
            return Path(served).name
        try:
            from backend.config import get_mode_generation_params

            raw = get_mode_generation_params(mode)["model"]
        except Exception:
            return self._base_model_name
        return Path(raw).name if raw else self._base_model_name

    @staticmethod
    def _resolve_model_key(mode: str) -> str | None:
        """その応答を生成したモデルの ``model_key`` (c_05 §0.5.7、f_04 §1.2.0)。

        経験の置き場 (``learning/<model_key>/experience.jsonl``) と Level 2 の経験の
        絞り込みはこの値で決まる。鍵は llama-server が実際に載せているモデル
        (``/props``) から取り、取れなければそのモードが宣言するモデル (chat は
        active、create は ``model_paths.create_model``。未指定なら active)
        (:meth:`PathResolver.generating_model_key`)。解決できない (config 未ロード等)
        ときは ``None`` で、その経験は束ねたファイルへ入り Level 2 の絞り込みに載らない。
        """
        try:
            from backend.config import get_path_resolver

            mode_key = "create" if is_create_mode(mode) else "chat"
            return get_path_resolver().generating_model_key(mode_key)
        except Exception as exc:  # noqa: BLE001 — 記録自体は止めない
            logger.debug("model_key unavailable for the experience: %s", exc)
            return None

    def record(
        self,
        query: str,
        response: str,
        mode: str = "chat",
        rag_used: bool = False,
        rag_top1_score: float | None = None,
        agent_loops: int = 0,
        base_model: str = "",
        embedding_model: str = "",
        long_form_used: bool = False,
        long_form_content_type: str | None = None,
        long_form_strategy: str | None = None,
        long_form_units_total: int = 0,
        long_form_units_completed: int = 0,
        long_form_validation_errors: int = 0,
        long_form_budget_used_pct: float | None = None,
        tool_routing_success: bool = False,
        tool_routing_false_positive: bool = False,
        tool_routing_false_negative: bool = False,
        long_form_success: bool = False,
        long_form_false_positive: bool = False,
        long_form_false_negative: bool = False,
        step_credits: list[dict] | None = None,
        completion_tokens: int | None = None,
        prompt_tokens: int | None = None,
        cached_prompt_tokens: int | None = None,
        action_blocked: bool = False,
        measured_values: dict[str, set[int]] | None = None,
        calculate_result: float | None = None,
        tool_result_text: str = "",
        tool_context: str = "",
        stated_context: str = "",
        truncated: bool = False,
        generation_failed: bool = False,
        session_id: str = "",
        turn_id: str = "",
        gen_config: "GenerationConfigRef | None" = None,
        unchecked_checks: list[str] | None = None,
    ) -> ExperienceEntry:
        """シグナル収集 → ExperienceBuffer に記録

        ``truncated`` は llama-server が ``finish_reason=length`` を返した印、
        ``generation_failed`` はユーザーへ本文が届かなかった / error で終わった印
        (どちらも :class:`FeedbackSignals` へそのまま刻む。後者は
        ``turn_outcome="failed"`` に倒す)。``session_id`` は直前ターンとの
        突き合わせ (訂正 / 言い直し / 保留) をセッション単位に閉じる鍵。

        ``turn_id`` / ``gen_config`` は fitness の帰属先を **記録** するための
        もの (c_05 §0.6)。以前は「その応答を生んだプロンプト版 / few-shot /
        ポリシー / LoRA」をどこにも残しておらず、Level 1 / Level 2 の帰属は
        時刻からの推測でしかなかった (2026-09-05 監査)。

        ``tool_result_text`` はプロンプトへ注入済みのツール実行結果ブロック
        (``## ツール実行結果`` 以降) で、``date_intent`` が組んだコマンドの
        ``target:`` 行と本文の日付の食い違いを見るのに使う
        (:func:`backend.free.core.response_dates.ignores_date_result`、
        2026-09-09 監査 G-06)。``calculate_result`` と同じく呼出側がプロンプト
        から読み戻して渡す。

        ``tool_context`` は採用ゲートがツール根拠のターンを再生するために残す
        ツール結果ブロック (f_04 §4.5)。プロンプトにブロックがあったターンだけ
        呼出側が渡し、``gen_config._extra["tool_context"]`` に
        :data:`TOOL_CONTEXT_CAP` 字まで入れる (超えたら先頭だけ残し
        ``tool_context_truncated`` を立てる。形式の版は変えない)。

        ``unchecked_checks`` は create の制作ステージで未検査に終わった主要な検査
        (``"contract:no_examples"`` の形、``core.check_outcome.unchecked_labels``)。
        あればそのターンを **ラベル無し** にする (docs/f_04 §2.5)。
        """
        from backend.free.core.text_quality import detect_lang

        lang = detect_lang(response)
        if self._disabled:
            # 学習無効化中: シグナル検出 / パターン学習 / バッファ書込を全てスキップ。
            # 呼出側 (chat_recorder) は戻り値を直接参照しないが、署名互換のため
            # 最小限のダミーエントリを返す
            return ExperienceEntry(
                id=ExperienceEntry.new_id(),
                session_id=session_id,
                turn_id=turn_id,
                trace_id=get_trace_id() or "",
                timestamp=utc_now(),
                mode=mode,
                query=query,
                response_summary=response[:RESPONSE_SUMMARY_CAP],
                response_full=truncate_at_boundary(response, RESPONSE_FULL_CAP),
                base_model=base_model or self._resolve_base_model_name(mode),
                embedding_model=embedding_model or self._embedding_model_name,
                lang=lang,
                gen_config=gen_config or GenerationConfigRef(),
                signals=FeedbackSignals(),
            )
        self._load_session_state(session_id)
        # 根拠台帳 (ツール判定が同じリクエストで積んだ接地の疑義) は成否の導出より
        # 先に読む — システムが検出済みの欠けを成否へ反映する (docs/f_04 §2.5)。
        tool_uses = current_tool_uses()
        unexplained_numbers, expression_issues, unexplained_date_math = current_grounding()
        unread_files, unread_grounding = current_unread_session_files()
        turn_outcome, outcome_reason = self._derive_turn_outcome_with_reason(
            response, step_credits,
            query=query,
            mode=mode,
            tool_routing_false_positive=tool_routing_false_positive,
            long_form_false_positive=long_form_false_positive,
            action_blocked=action_blocked,
            measured_values=measured_values,
            calculate_result=calculate_result,
            tool_result_text=tool_result_text,
            stated_context=stated_context,
            long_form_used=long_form_used,
            long_form_success=long_form_success,
            long_form_validation_errors=long_form_validation_errors,
            unexplained_numbers=unexplained_numbers,
            expression_issues=expression_issues,
            unexplained_date_math=unexplained_date_math,
            unchecked_checks=unchecked_checks,
            tool_uses=tool_uses,
            unread_files=unread_files,
            unread_grounding=unread_grounding,
        )
        if generation_failed:
            # 本文が届かなかった / error フレームで終わったターン。
            # 推定ではなく観測なので無条件に failed。
            turn_outcome, outcome_reason = "failed", "generation failed"
        # 結末 JSONL (``success`` / ``quality_signals``) と自己申告の台帳へ、
        # ここで決めた成否を持ち上げる。以前は経験にしか残らず、結末は
        # 「SSE を届けられたか」だけで success=true だった (F-11)。
        _publish_turn_outcome(turn_outcome, outcome_reason)
        # ラベル無しは成否の列挙の値ではなく **理由付きの success** で持つ
        # (``level0_instant.teaches_success`` が読む)。結末 JSONL には "unlabeled" が出る。
        if turn_outcome == "unlabeled":
            turn_outcome = "success"
        if turn_outcome == "failed":
            # 失敗ターンの成功シグナルは矛盾なので failed 側に倒す
            # (偽成功が learned_patterns の正例学習 / Level 1 fitness に
            # 伝播した 2026-07-15 の再発防止)。
            tool_routing_success = False
            long_form_success = False

        # 字句検出が返すのは **候補** であって確定した訂正ではない。学習側が
        # 消費する ``user_correction`` へは、``learning.correction_verifier``
        # が直前応答との突き合わせで検証してから昇格させる (F-03)。除外正規表現は
        # 事故のたびに 1 分岐ずつ増えており、次の語形で必ずまた漏れる。
        correction_text, detected_by = self._detect_correction(
            query, mode=mode, session_id=session_id,
        )
        # 訂正と言い直しは排他: 訂正が検出されたターンを rephrase として
        # 二重学習しない
        rephrased = (
            correction_text is None and self._detect_rephrase(query)
        )
        # 訂正が **どのターンを指すか** をここで確定する。記録時点でしか
        # セッションと直近ターンの本文が揃わないため、後段に再導出させると
        # 位置頼みになり別会話のターンと組まれる (F-01 の再発防止)。
        corrected_entry_id = (
            resolve_correction_target(
                correction_text, self._correction_candidates(session_id),
            )
            if correction_text is not None else None
        ) or None

        signals = FeedbackSignals(
            turn_outcome=turn_outcome,
            rephrased_query=rephrased,
            corrected_entry_id=corrected_entry_id,
            rag_used=rag_used,
            rag_top1_score=rag_top1_score,
            agent_loops=agent_loops,
            # user_correction は検証後にしか立たない (correction_verifier)。
            correction_candidate=correction_text,
            correction_detected_by=detected_by,
            long_form_used=long_form_used,
            long_form_content_type=long_form_content_type,
            long_form_strategy=long_form_strategy,
            long_form_units_total=long_form_units_total,
            long_form_units_completed=long_form_units_completed,
            long_form_validation_errors=long_form_validation_errors,
            long_form_budget_used_pct=long_form_budget_used_pct,
            tool_routing_success=tool_routing_success,
            tool_routing_false_positive=tool_routing_false_positive,
            tool_routing_false_negative=tool_routing_false_negative,
            turn_outcome_reason=outcome_reason,
            # 源は tool_ledger (実行の唯一の合流点)。プロンプト由来の
            # tool_result_text / calculate_result は台帳を通らない経路の保険。
            tool_grounded=(
                bool(tool_uses) or bool(tool_result_text)
                or calculate_result is not None
            ),
            # 文書を注入した turn の抑止応答 = 検索の取りこぼしの観測 (f_04 §3.2)。
            rag_abstained=(
                abstains_on_reference_material(response)
                if gen_config is not None
                and any(str(e).startswith("corpus:") for e in (gen_config.evidence_ids or []))
                else None
            ),
            rag_cited=cites_reference_material(response) if rag_used else None,
            tool_uses=tool_uses,
            unexplained_numbers=unexplained_numbers,
            expression_issues=expression_issues,
            unexplained_date_math=unexplained_date_math,
            long_form_success=long_form_success,
            long_form_false_positive=long_form_false_positive,
            long_form_false_negative=long_form_false_negative,
            step_credits=step_credits or [],
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            cached_prompt_tokens=cached_prompt_tokens,
            truncated=truncated,
            generation_failed=generation_failed,
        )
        # 実行したツールを決めた層。近道 (recall / learned) の実行を tool_routing の
        # 正例から外す根拠 (不変則 #15)。宣言フィールドにすると形式 lock が入れ子の
        # 既定値の変化を版上げ扱いにするので、未知キーの通路 (``_extra``) で運ぶ —
        # 読み手 (Level 1 は永続形の dict を読む) からは signals の 1 キーに見える。
        decided_by = current_tool_decision()
        if decided_by:
            signals._extra = {**(signals._extra or {}), "decided_by": decided_by}
        # 抑止応答の型も同じ通路で運ぶ (形式の版を上げない)。
        abstain_kind = rag_abstain_kind(gen_config, signals.rag_abstained, response)
        if abstain_kind:
            signals._extra = {**(signals._extra or {}), "rag_abstain_kind": abstain_kind}
        # RAG の便益を結末 JSONL へ (turn_outcome と同じ通路、2026-09-14)。
        record_rag_signals(
            rag_used=signals.rag_used, rag_abstained=signals.rag_abstained,
            rag_cited=signals.rag_cited,
        )
        if tool_context:
            gen_config = gen_config or GenerationConfigRef()
            extra = {**(gen_config._extra or {}), TOOL_CONTEXT_KEY: tool_context[:TOOL_CONTEXT_CAP]}
            if len(tool_context) > TOOL_CONTEXT_CAP:
                extra[TOOL_CONTEXT_TRUNCATED_KEY] = True
            gen_config._extra = extra

        entry = ExperienceEntry(
            id=ExperienceEntry.new_id(),
            session_id=session_id,
            turn_id=turn_id,
            trace_id=get_trace_id() or "",
            timestamp=utc_now(),
            mode=mode,
            query=query,
            response_summary=response[:RESPONSE_SUMMARY_CAP],
            response_full=truncate_at_boundary(response, RESPONSE_FULL_CAP),
            base_model=base_model or self._resolve_base_model_name(mode),
            model_key=self._resolve_model_key(mode),
            embedding_model=embedding_model or self._embedding_model_name,
            lang=lang,
            gen_config=gen_config or GenerationConfigRef(),
            signals=signals,
        )

        # 前ターンへの false_negative の遡及は、ここ (字句の訂正候補の時点) では
        # 立てない (不変則 #12)。訂正が検証で ``assistant`` に昇格した時に
        # ``correction_verifier.reconcile_false_negatives`` が立て、格下げで外す。
        current_routed_tool = tool_routing_success or tool_routing_false_positive

        # 訂正・言い直し検出からのパターン学習は行わない (2026-07-21 廃止)。
        # correction: 話題語学習による偽陽性増殖 (_detect_correction の
        # docstring 参照)。rephrase: 書き込み専用の dead カテゴリで、match()
        # の参照箇所が存在しなかった (検出自体は文字重複率ベースで学習不要)。

        # ツールルーティング false_negative 時: 明示注入 (テスト等) では当該クエリから学習。
        if tool_routing_false_negative:
            self._learn_tool_routing_from_false_negative(query)

        # 長文ルーティング false_negative 時: クエリからキーワードを
        # ``category="long_form"`` として学習する。success は学習しない —
        # 長文経路が自分で選んだ結果を自分の正例に数えると自己強化の輪になる
        # (不変則 #15)。
        if long_form_false_negative:
            self._learn_long_form_from_signal(query)

        self._apply_self_retraction(signals, response)

        # 前ターンで保留した訂正を、当ターンの応答で確定 / 撤回する。
        # entry を buffer へ積む前に回す (撤回対象は前 entry なので順序は
        # どちらでもよいが、ログの並びを「撤回 → 記録」に揃える)。
        self._settle_pending_correction(response)
        if correction_text is not None:
            self._arm_pending_correction(entry, query)
            # 当ターンの応答が既に反論しているケース (実測 2026-08-22:
            # 「約100kmという値は事実と異なります。」) はここで確定する。
            # 判断材料が出ていなければ保留のまま次ターンへ持ち越す。
            self._settle_pending_correction(response)

        self.buffer.record(entry)
        self._session_entries.append(entry)
        self._prev_query = query
        self._prev_response = response or ""
        self._prev_entry = entry
        self._prev_routed_tool = current_routed_tool
        self._prev_used_long_form = long_form_used
        self._prev_turn_failed = turn_outcome == "failed"
        self._store_session_state(session_id)

        logger.info(
            "Recorded experience: mode=%s, rephrase=%s, correction_candidate=%s "
            "(by=%s)",
            mode, signals.rephrased_query,
            signals.correction_candidate is not None, detected_by,
        )

        # DebugLogger に Level 0 学習サイクルを記録
        dl = self._debug_logger
        if dl:
            dl.log_learning_cycle(cycle_num=0, data={
                "level": 0,
                "mode": mode,
                "buffer_size": self.buffer.count,
                "rephrase": signals.rephrased_query,
                "correction_candidate": signals.correction_candidate is not None,
                "correction_detected_by": detected_by,
                "rag_used": signals.rag_used,
            })

        return entry

    def mark_conversation_ended(self) -> None:
        """現会話セッションで record した全エントリに conversation_ended を設定

        record() は会話途中の各応答ごとに新規 entry を作るため、会話終了時に
        当該セッションの全 entry へまとめて反映する (approach a)。マーク後は
        セッション参照と直前クエリをリセットし、次会話を新セッション扱いにする。
        buffer ローテーションで切り捨てられた entry も参照経由で安全 (生存 entry のみ
        buffer に効き、切捨て済みは GC 対象)。
        """
        if self._disabled:
            return
        for entry in self._session_entries:
            entry.signals.conversation_ended = True
        # entry 群を書き換えたので永続化する。record 経由の自動保存が
        # 掛からない唯一の変更点 (次の record まで待つと会話終了フラグが
        # 落ちる)。書き換えた分だけ patch 行で出す (全件の差分はループで取らない)。
        self._touch(*self._session_entries)
        self.buffer.flush(touched_only=True)
        self._session_entries.clear()
        self._prev_query = None
        self._prev_entry = None
        self._prev_routed_tool = False
        self._prev_used_long_form = False
        self._prev_turn_failed = False
        self._pending_correction = None
        self._prev_response = ""
        self._sessions.clear()

    def _touch(self, *entries: ExperienceEntry) -> None:
        """記録済みのエントリを書き換えたことを経験バッファへ知らせる (patch 行の対象)。"""
        touch = getattr(self.buffer, "touch", None)
        if callable(touch):
            touch(*entries)

    # ── セッション別の直前ターン状態 ──

    def _load_session_state(self, session_id: str) -> None:
        """``session_id`` の直前ターン状態を作業コピー (``_prev_*``) へ載せる。"""
        state = self._sessions.get(session_id)
        if state is None:
            state = _SessionTurnState()
        else:
            self._sessions.move_to_end(session_id)
        self._prev_query = state.prev_query
        self._prev_entry = state.prev_entry
        self._prev_routed_tool = state.prev_routed_tool
        self._prev_used_long_form = state.prev_used_long_form
        self._prev_turn_failed = state.prev_turn_failed
        self._pending_correction = state.pending_correction
        self._prev_response = state.prev_response

    def _store_session_state(self, session_id: str) -> None:
        """作業コピーを ``session_id`` の枠へ書き戻す (LRU 上限 16)。"""
        self._sessions[session_id] = _SessionTurnState(
            prev_query=self._prev_query,
            prev_entry=self._prev_entry,
            prev_routed_tool=self._prev_routed_tool,
            prev_used_long_form=self._prev_used_long_form,
            prev_turn_failed=self._prev_turn_failed,
            pending_correction=self._pending_correction,
            prev_response=self._prev_response,
        )
        self._sessions.move_to_end(session_id)
        while len(self._sessions) > _SESSION_STATE_CAP:
            self._sessions.popitem(last=False)

    def _correction_candidates(
        self, session_id: str,
    ) -> list[tuple[str, str, str]]:
        """訂正の宛先候補 ``(entry_id, response_full, query)`` を古い順に返す。

        経験バッファの **同一セッション** のエントリから直近
        ``_RECENT_TURN_WINDOW`` 件。**ユーザーの訂正ターン自身は除く** —
        入れると「先ほどの回答の家族構成が間違いです」が 1 つ前の訂正発話
        「先ほどの 7 項目の列挙は間違いです」を宛先に選ぶ (2026-09-07 実測)。
        応答本文は response_full (文境界で切った全文) を優先する — 200 字の
        要約だと、後半でしか触れていない値を引用した訂正が結び付かない。
        """
        entries = getattr(self.buffer, "entries", None) or []
        out: list[tuple[str, str, str]] = []
        for e in reversed(entries):
            if getattr(e, "session_id", "") != session_id:
                continue
            sig = getattr(e, "signals", None)
            # 候補 (未検証) の段階で除く。検証は後から走るので、ここで
            # ``user_correction`` だけを見ると訂正発話自身が宛先候補に残る。
            if sig is not None and (
                getattr(sig, "correction_candidate", None) is not None
                or getattr(sig, "user_correction", None) is not None
            ):
                continue
            out.append((
                e.id,
                e.response_full or e.response_summary or "",
                e.query or "",
            ))
            if len(out) >= _RECENT_TURN_WINDOW:
                break
        out.reverse()
        return out

    @staticmethod
    def _derive_turn_outcome(
        response: str,
        step_credits: list[dict] | None,
        *,
        query: str = "",
        tool_routing_false_positive: bool = False,
        long_form_false_positive: bool = False,
        action_blocked: bool = False,
        measured_values: dict[str, set[int]] | None = None,
        calculate_result: float | None = None,
        tool_result_text: str = "",
        stated_context: str = "",
    ) -> str:
        """ターン成否だけを返す (:meth:`_derive_turn_outcome_with_reason` の薄い皮)。

        値は ``"success"`` / ``"partial"`` / ``"failed"`` / ``"unlabeled"`` (ラベル無しは判定の
        語彙で、経験には理由付きの ``success`` として刻む。docs/f_04 §2.5)。
        """
        outcome, _ = FeedbackCollector._derive_turn_outcome_with_reason(
            response, step_credits,
            query=query,
            tool_routing_false_positive=tool_routing_false_positive,
            long_form_false_positive=long_form_false_positive,
            action_blocked=action_blocked,
            measured_values=measured_values,
            calculate_result=calculate_result,
            tool_result_text=tool_result_text,
            stated_context=stated_context,
        )
        return outcome

    @staticmethod
    def _derive_turn_outcome_with_reason(
        response: str,
        step_credits: list[dict] | None,
        *,
        query: str = "",
        mode: str = "chat",
        tool_routing_false_positive: bool = False,
        long_form_false_positive: bool = False,
        action_blocked: bool = False,
        measured_values: dict[str, set[int]] | None = None,
        calculate_result: float | None = None,
        tool_result_text: str = "",
        stated_context: str = "",
        long_form_used: bool = False,
        long_form_success: bool = False,
        long_form_validation_errors: int = 0,
        unexplained_numbers: list[str] | None = None,
        expression_issues: list[str] | None = None,
        unexplained_date_math: bool | None = None,
        unchecked_checks: list[str] | None = None,
        tool_uses: list[dict] | None = None,
        unread_files: list[str] | None = None,
        unread_grounding: str = "",
    ) -> tuple[str, str | None]:
        """ターン成否 ("success" | "partial" | "failed" | "unlabeled") と理由を決定論導出する。

        **検出済みの失敗 (2026-09-28、ライブ監査 2026-09-27 M5、docs/f_04 §2.5)**: 本文の
        破綻に当たらなかったターンも、システムが同じターンで構造化して記録した判定を読む。
        長文の検証落ち (``judge_long_form_success`` が偽) は ``failed``。応答の誤りとは
        確定しない欠け — 未実行の変更操作 (``action_blocked``)・会話から辿れない数での
        計算 (``unexplained_numbers``)・式の組み方の疑い (``expression_issues``)・ツールで
        検証していない日付演算 (``unexplained_date_math``)・create の未検査
        (``unchecked_checks``) — は ``unlabeled`` (成功とも失敗とも教えない)。失敗の証拠は
        ラベル無しに勝つので、``unlabeled`` は最後に見る。新しい字句判定は足さない。
        ``unlabeled`` は判定の語彙で、経験には **理由付きの** ``turn_outcome="success"``
        として刻む (:meth:`record`、``level0_instant.teaches_success``)。

        SSE 完走 = 成功ではなく、応答本文の [failed] マーカー・step_credits
        全 0・ルーティング false_positive・ユーザー発話のオウム返し、および
        **本文の決定論的な破綻** を失敗シグナルとして扱う。

        本文の破綻を見る理由: 実測 (2026-08-18、経験 136 件) で
        ``turn_outcome`` は **136/136 が success** の恒真だった。純粋なチャット
        応答では既存の 4 条件がどれも成立しないためで、Level 1 / critique /
        few-shot の成否シグナルが実質的に情報を持たない。

        一方で few-shot 側は同じ応答に対して 10 種の決定論ゲートを持っている
        (``fewshot_pool.find_content_rejection``)。ところがその判定は
        **「手本に採らない」で止まり、成否には反映されていなかった**。判定器の
        実体が横断基盤 (``core.text_quality`` / ``core.response_arithmetic``) に
        あるものだけをここでも掛け、「壊れた出力を成功として学習する」経路を塞ぐ。

        採る 4 つ (いずれも誤検出コストが低い決定論):

        - 算術矛盾 — 本文に書かれた式の検算が合わない
        - 日本語の語間空白 — 崩れた出力の目印 (正常な日本語では発生しない)
        - 中国語語彙の混入 — 同上
        - 応答中の自己撤回 — 1 つの応答に結論が 2 つ入っている

        実測での発火は **0/136** (この corpus を出した Qwen3.8-27B には該当が
        無い)。恒真性が即座に解けるわけではなく、モデルが劣化したときに
        「壊れた出力が手本として再生産される自己増幅」を断つための網である。
        """
        text = response or ""
        if _FAILED_MARKER_RE.search(text):
            if _DONE_MARKER_RE.search(text):
                return "partial", "some tasks failed"
            return "failed", "all tasks failed"
        # 書込みゲートが断っただけ (根拠台帳の全件が write_denied) なら失敗にしない —
        # 断ったこと自体は安全側の正しい振る舞い。ゲートの誤判定もありうるので成功の
        # 教師にもせず、後段の失敗の証拠が無ければラベル無しにする (docs/f_04 §2.5)。
        guard_denied = bool(tool_uses) and all(
            u.get("reason") == "write_denied" for u in tool_uses or ()
        )
        if guard_denied:
            # ラベルを付けない代わりに件数を数えて出す — 書込みゲートの誤った断り
            # (上書きの門、docs/f_03 §4.y) が学習からも監視からも消えないように
            _GUARD_DENIED_TURNS[0] += 1
            logger.info(
                "Turn left unlabeled: every tool use was a write-gate denial "
                "(%d such turns in this process)", _GUARD_DENIED_TURNS[0],
            )
        zero_credit = bool(step_credits) and all(
            not (c.get("credit") or 0) for c in step_credits or ()
        )
        if zero_credit and not guard_denied:
            return "failed", "no step credit"
        if tool_routing_false_positive or long_form_false_positive:
            return "failed", "routing false positive"
        if _is_user_echo(query, text):
            return "failed", "user echo"
        broken = FeedbackCollector._find_broken_output_reason(text)
        if broken is not None:
            logger.info("Turn marked failed (%s)", broken)
            return "failed", broken
        # システムが「撃てなかった」と知っているのに本文が完了を述べている =
        # 真偽の推定ではなく **矛盾**。2026-08-22 ライブ監査で 2 ターン続けて
        # 起きた形 (Action blocked が出ているのに「削除しました。」、ファイルは残存)。
        if action_blocked:
            claim = claims_completed_state_change(text)
            if claim is not None:
                logger.info(
                    "Turn marked failed (claimed %r while the action was "
                    "blocked)", claim,
                )
                return "failed", f"claimed completion while blocked: {claim}"
        # 実測値を注入したのに別の数を述べている = 同じく矛盾。
        # 2026-08-22 ライブ監査: 実測 86 文字を注入済みで「100文字です」。
        mismatch = contradicts_measured_values(text, measured_values or {})
        if mismatch is not None:
            logger.info("Turn marked failed (%s)", mismatch)
            return "failed", f"measured value contradiction: {mismatch}"
        # calculate の結果を渡したのに本文がそれを使っていない = 暗算で別の数を
        # 述べた。ツール結果は「確かめた事実」なので、これも推定ではなく矛盾
        # (2026-09-05 ライブ監査 F-03: 結果 17,305,634 が本文に 1 つも現れず、
        # 手数料を 10 倍に誤った結論が reward=1.0 で成功経験になっていた)。
        # 判定は採用ゲートの再生と同じ関数 (丸め違いも含む、不変則 #14(a))。
        ignored = calculate_result_contradiction(text, calculate_result)
        # 式が会話から辿れない / 組み方に疑いがある calculate は、結果そのものが
        # 誤っている疑いがあり、無視した回答が正しいこともある (2026-09-26 C01#3:
        # ``100000 - 26000`` の 100000 は会話に無く、結果を使わない「2万8千円」が
        # 正答)。失敗の証拠にしない — 下のラベル無しに理由を添える (docs/f_04 §2.5)。
        ignored_unverified: str | None = None
        if ignored is not None:
            if unexplained_numbers or expression_issues:
                ignored_unverified = ignored
            else:
                logger.info("Turn marked failed (tool result ignored: %s)", ignored)
                return "failed", f"tool result ignored: {ignored}"
        # date_intent が組んだツール結果 (``target:`` 行) を渡したのに本文が
        # 別の日付を述べている = calculate と同じ構造の矛盾。ツールが向きの
        # 補正 (逆算) を正しく踏んでも、モデルが結果を暗算で差し替える経路は
        # 別に残る (2026-09-08 T19/3: target=2026-11-02 に対し本文は暗算の
        # 10月14日、正しくは向きの修正込みで 10/9)。
        # ただし問いが起点を会話・具体日付に置いている (「さっきのリリース日の
        # 1 週間前」) のにツールが今日起点で数えていたら、target の方が誤りで、
        # 使わなかった回答が正答のことがある (2026-10-05 ライブ監査: target
        # 9/28 を使わず 10/13 と正答し failed にされた)。calculate の式の疑い
        # と同じく失敗の証拠にせず、ラベル無しにする。
        # 判定は採用ゲートの再生と同じ関数 (``relative_date.date_result_use``)。
        date_unverified: str | None = None
        date_use = date_result_use(query, tool_result_text, text)
        if date_use == DATE_RESULT_ANCHOR_MISMATCH:
            tool_anchor = extract_tool_anchor(tool_result_text)
            date_unverified = (
                f"date tool anchored on {tool_anchor or 'unknown'} while the "
                "query anchors elsewhere"
            )
            logger.info("Date result not counted as ignored (%s)", date_unverified)
        elif date_use == DATE_RESULT_IGNORED:
            logger.info("Turn marked failed (date result ignored)")
            return "failed", "date result ignored: response date does not match target"
        # 明示された文字数指定を破っている = 指定は本文にあり長さは数えるだけ
        # なので、これも推定ではなく矛盾。2026-08-22 ライブ監査の
        # 「ちょうど100文字で」→ 86 文字は success として学習に入っていた。
        broken_length = violates_length_constraint(query, text)
        if broken_length is not None:
            logger.info("Turn marked failed (%s)", broken_length)
            return "failed", f"length constraint: {broken_length}"
        # 形式指定 (箇条書き / 項目数 / 数値だけ) も同じ扱い。数えるだけで
        # 決まるので推定を含まない。文字数だけ見て形式を見ないと、
        # 「3つ箇条書きで」に 1 行で答えたターンが success として学習に入る。
        # create の依頼の「箇条書き」は成果物の仕様で、応答の形式の指定ではない
        # (2026-09-27 ライブ監査 M6: K05「見出し・箇条書き・太字…に対応」だけが failed)。
        broken_form = None if is_create_mode(mode) else violates_output_form(query, text)
        if broken_form is not None:
            logger.info("Turn marked failed (%s)", broken_form)
            return "failed", f"output form: {broken_form}"
        # 本人が言っていない世帯の人数を補って言い直した = 記憶の想起としても
        # 誤り。system プロンプトの規則 (「『妻と娘と犬』を『ご家族 4 人』と
        # 数え直さない」) は実機で守られず、2026-09-16 監査の **両ランで再現**
        # した。数えるだけで決まるので推定を含まない。
        fabricated = fabricated_household_count(text, stated_context, query=query)
        if fabricated is not None:
            logger.info("Turn marked failed (fabricated count: %s)", fabricated)
            return "failed", f"fabricated count: {fabricated}"
        # 中身を読めたファイルが無いと確定事実で渡したのに、敬称付きの人名で答えた =
        # 読めていない中身を作った (2026-10-05 ライブ監査 T2「佐藤健太さんです」)。
        # 答えの位置の人名がプロンプト (assistant 以外と、ツールを使ったターンの答え) にも
        # ツールの結果にも無いことだけを見る。カタカナだけの名前は読み (サトウ / 佐藤) や
        # 役割語と区別できないので失敗にせず、ラベル無しにする。
        unverified_names: str | None = None
        if unread_files:
            names, kana_names = ungrounded_answer_names(
                text, f"{query}\n{stated_context}\n{tool_result_text}\n{unread_grounding}",
            )
            if names:
                logger.info(
                    "Turn marked failed (fabricated entity while %s unread: %s)",
                    ", ".join(unread_files), ", ".join(names),
                )
                return "failed", f"fabricated entity: {', '.join(names)}"
            if kana_names:
                unverified_names = f"unverified names: {', '.join(kana_names)}"
        # 長文の成果物が検証で落ちた。結末 JSONL の success と同じ判定
        # (``judge_long_form_success``) を呼出側から受け取る — 同じターンの成否を
        # 2 か所で決めない (2026-09-27 監査 C07#2: 結末は失敗、経験は成功だった)。
        if long_form_used and not long_form_success:
            logger.info(
                "Turn marked failed (long-form validation, %d error(s))",
                long_form_validation_errors,
            )
            return "failed", (
                f"long-form validation failed: {long_form_validation_errors} error(s)"
            )
        unlabeled = FeedbackCollector._unlabeled_reason(
            action_blocked=action_blocked,
            unexplained_numbers=unexplained_numbers,
            expression_issues=expression_issues,
            unexplained_date_math=unexplained_date_math,
            unchecked_checks=unchecked_checks,
        )
        if unlabeled is None and zero_credit and guard_denied:
            unlabeled = "write denied by guard"
        # このターンのツールが 1 つも役に立つ結果を返さなかった (読みが見つからない等)。
        # 答えはツールの結果に支えられておらず、成功の手本にしない — 2026-10-05
        # ライブ監査 T1: read_file が File not found のまま「分かりません」で success
        # だった。正しく「見つからない」と伝えた応答 (答えでない応答) はラベル無し。
        # 依頼にも会話にも無い数を述べた応答は、取れなかった中身を作った応答なので
        # 失敗 (T2 の給与の作話と同じ形。数は字句の鍵、不変則 #14)。
        if (
            unlabeled is None and tool_uses and not guard_denied
            and not any(u.get("success") for u in tool_uses)
        ):
            invented = _numbers_not_in(
                text, f"{query}\n{stated_context}\n{tool_result_text}", query,
            )
            if invented:
                logger.info(
                    "Turn marked failed (numbers stated after every tool call failed: %s)",
                    ", ".join(invented),
                )
                return "failed", (
                    "content stated after every tool call failed: " + ", ".join(invented)
                )
            unlabeled = "every tool call failed"
        if date_unverified is not None:
            unlabeled = f"{unlabeled}; {date_unverified}" if unlabeled else date_unverified
        if unverified_names is not None:
            unlabeled = f"{unlabeled}; {unverified_names}" if unlabeled else unverified_names
        if unlabeled is not None:
            if ignored_unverified is not None:
                unlabeled = f"{unlabeled}; tool result ignored: {ignored_unverified}"
            logger.info("Turn left unlabeled (%s)", unlabeled)
            return "unlabeled", unlabeled
        # 既存の検証器がどれも印を付けなかったターンにだけ掛ける追加の検証器
        # (2026-10-07)。既存の結果と優先順位は変えない。
        extra = FeedbackCollector._find_additional_failure_reason(
            text, mode=mode, tool_uses=tool_uses,
        )
        if extra is not None:
            logger.info("Turn marked failed (%s)", extra)
            return "failed", extra
        return "success", None

    @staticmethod
    def _find_additional_failure_reason(
        text: str, *, mode: str = "chat", tool_uses: list[dict] | None = None,
    ) -> str | None:
        """既存の検証器で無印だったターンの決定論の破綻を返す (無ければ ``None``)。

        判定器は ``core.response_verifiers`` の純関数で、どれも例外を外へ出さない。
        """
        rate = misstated_change_rate(text)
        if rate is not None:
            return f"change rate contradiction: {rate}"
        # create の応答は計画の報告 (「2 件のタスクを…」) で、宣言と一覧の形が違う
        if not is_create_mode(mode):
            count = declared_count_mismatch(text)
            if count is not None:
                return f"declared count mismatch: {count}"
        unavailable = false_tool_unavailability(text, tool_uses)
        if unavailable is not None:
            return f"false tool unavailability: {unavailable}"
        return None

    @staticmethod
    def _unlabeled_reason(
        *,
        action_blocked: bool,
        unexplained_numbers: list[str] | None,
        expression_issues: list[str] | None,
        unexplained_date_math: bool | None,
        unchecked_checks: list[str] | None,
    ) -> str | None:
        """成功とも失敗とも教えない検出済みの欠けの理由 (無ければ ``None``)。

        どれも「依頼を満たせたか確かめられていない / 満たせなかったが応答の誤りでは
        ない」印で、正答のこともある。``failed`` にすると環境 (削除ツールがどのモードにも
        無い・構文検査器が無い) や正しい前提を罰し、``success`` にすると検証していない
        答えを手本にする (docs/f_04 §2.5 の表)。
        """
        if action_blocked:
            return "action not executed"
        if unexplained_numbers:
            return f"unexplained numbers: {', '.join(unexplained_numbers)}"
        if expression_issues:
            return f"suspicious expression: {'; '.join(expression_issues)}"
        if unexplained_date_math:
            return "unverified date math"
        if unchecked_checks:
            return f"unchecked: {', '.join(unchecked_checks)}"
        return None

    @staticmethod
    def _find_broken_output_reason(text: str) -> str | None:
        """応答本文の決定論的な破綻を返す (無ければ ``None``)。

        判定器はすべて横断基盤の純粋関数。few-shot の内容棄却ゲートと同じ
        実体を共有する (片方だけ直る状態を作らない)。
        """
        contradictions = find_arithmetic_contradictions(text)
        if contradictions:
            return f"arithmetic contradiction: {contradictions[0]}"
        # 式ごとには正しいのに **冒頭の結論だけ** が本文の計算と別の数、という
        # 形は上の判定では捕まらない。読み手が最初に受け取る値なので実害は
        # 大きい (2026-09-06 監査 F-03)。
        conclusion = find_conclusion_contradiction(text)
        if conclusion is not None:
            return f"conclusion contradiction: {conclusion}"
        # 大きさは合うのに冒頭が本文と **逆の符号** (「4000円残ります」と本文の
        # 「差し引き：-4000円」)。上の判定は大きさしか見ない (2026-10-05 ライブ監査)。
        sign = find_sign_contradiction(text)
        if sign is not None:
            return f"sign contradiction: {sign}"
        if has_broken_ja_spacing(text):
            return "broken JA spacing"
        if has_chinese_token_leak(text):
            return "Chinese token leaked into JA response"
        if retracts_own_conclusion(text):
            return "response retracts its own conclusion mid-answer"
        if is_cut_off_answer(text):
            return "answer cut off mid-word"
        return None

    def _same_target_path(self, query: str) -> bool:
        """直前クエリと同じ明示出力先パスを再指定しているかを判定する。"""
        if self._prev_query is None:
            return False
        prev = _QUERY_PATH_RE.search(self._prev_query)
        curr = _QUERY_PATH_RE.search(query)
        if prev is None or curr is None:
            return False
        return prev.group(0).lower() == curr.group(0).lower()

    def _detect_rephrase(self, query: str) -> bool:
        """直前の発話の **言い直し** か (制約を足した深掘りは含めない)。

        双方に明示的な出力先パスがあり、それが異なる場合は「類似した別の
        新規依頼」(テンプレ連続依頼等) なので rephrase としない
        (2026-07-15: 31 連続の類似依頼で偽陽性 2 件)。

        指標と閾値の根拠は :data:`REPHRASE_THRESHOLD` 周辺のコメントを参照。
        深掘り (「Xを3行で」→「Xを、Yに絞って3行で」) を先に落とすのが要点で、
        旧実装 (文字集合 Jaccard) はこれを言い直しとして数え、Level 1 の
        選択圧の 73% を誤検出で占めていた。
        """
        if self._prev_query is None:
            return False

        prev_path = _QUERY_PATH_RE.search(self._prev_query)
        curr_path = _QUERY_PATH_RE.search(query)
        if (
            prev_path is not None
            and curr_path is not None
            and prev_path.group(0).lower() != curr_path.group(0).lower()
        ):
            return False

        prev, curr = self._prev_query.strip(), query.strip()
        if not prev or not curr:
            return False

        # 一字一句同じ再送は言い直しではない。応答が届かなかった / 誤送信の
        # 再試行で、モデルの応答が通じなかった証拠にはならない。2026-09-05 の
        # 失敗 32 件のうち 8 件がこれ (Level 2 の学習データを薄めていた)。
        if prev == curr:
            return False
        # 「私の出身大学は北海道大学です。」→「私の出身大学はどこでしたか？」は
        # 申告の直後に想起を試す **質問** で、内容語はほぼ同じでも言い直しではない
        # (同じ 32 件のうち 3 件)。平叙の申告 → 問いの並びは除外する。
        if is_plain_statement(prev) and not is_plain_statement(curr) and (
            curr.rstrip().endswith(("?", "？", "か", "か。", "っけ", "っけ。"))
        ):
            return False

        # 深掘り: 前の発話がほぼそのまま残り、そこへ制約が足されている。
        # ユーザーが問いを絞り込んだのであって、答えが通じなかったのではない。
        if (
            bigram_coverage(prev, curr) >= _DRILLDOWN_MIN_COVERAGE
            and len(curr) >= len(prev) * _DRILLDOWN_MIN_LENGTH_RATIO
        ):
            logger.debug(
                "Rephrase candidate is a drill-down (constraints added); "
                "not counting it as a defect: %s", curr[:60],
            )
            return False

        # 指標と閾値は **比べる発話そのものの字種** で選ぶ (GUI locale ではない)。
        # 内容語の抽出はひらがなを機能語として落とす実装なので、英語には効かず
        # スケールが変わる = 閾値も別物になる。locale で選ぶと、既定 'ja' のまま
        # 英語で打った 2 発話に日本語用の 0.70 が掛かり、逆に locale='en' の
        # まま日本語で打つと内容語抽出が効かず生コサインの 0.5 で測られる —
        # どちらも言い直し判定 (= Level 1 の選択圧) を静かに歪める。
        if has_japanese_script(prev) or has_japanese_script(curr):
            return content_bigram_cosine(prev, curr) >= REPHRASE_THRESHOLD
        return bigram_cosine(prev, curr) >= REPHRASE_THRESHOLD_EN

    def _apply_self_retraction(self, signals, response: str) -> None:
        """アシスタント自身の撤回を検出し、**直前ターン**を failed へ落とす。

        撤回した本ターンは誤りを直した側なので失敗ではない。誤っていたのは
        1 つ前のターンであり、``_prev_entry`` はまだバッファ内にあるので
        その ``turn_outcome`` を書き換えれば正しい側に選択圧が掛かる
        (``ExperienceBuffer`` は entry オブジェクトを保持しており、保存時に
        書き換え後の値が直列化される)。

        既に failed / partial のエントリは触らない (格上げも格下げもしない)。
        ラベル無し (理由付きの success) は「検証できていない」だけなので、撤回を誤りの
        証拠として failed へ落とす (docs/f_04 §2.5)。
        """
        if not detect_assistant_self_retraction(response):
            return
        signals.assistant_self_retraction = True
        prev = self._prev_entry
        if prev is None or prev.signals.turn_outcome != "success":
            return
        prev.signals.turn_outcome = "failed"
        prev.signals.turn_outcome_reason = "retracted by assistant"
        self._touch(prev)
        logger.info(
            "Assistant retracted its previous answer; marking the previous "
            "turn as failed (prev_query=%s)", (self._prev_query or "")[:60],
        )

    def _arm_pending_correction(self, entry, query: str) -> None:
        """値が食い違う訂正を「保留」に置く (確定は次ターン)。

        検出時点ではユーザーの主張が正しいかを知る手段が無い。ところが
        ``user_correction`` は critique_synthesizer が失敗事例として消費し、
        generation_param_evolver は重み 1.0 で見るため、**誤った主張を 1 件
        受け取るだけで自分の正答が失敗として学習される**。

        実インシデント 2026-08-22 ライブ監査: 「東京・大阪間は約100kmです」
        (誤) に「訂正ありがとうございます。承知しました。」と応じて
        ``correction=True (by=hardcoded)`` が記録された。**次のターンで
        「本当に100kmですか？」と聞くと「約370kmです」と元の値を維持** して
        おり、さらに後で「私が誤った情報を伝えた箇所は？」と聞けば正しく
        指摘できた。壊れているのは記録側だけで、判断材料は 1 ターン後に出る。

        保留は **数値の食い違いが明確な場合だけ**。訂正クエリに数値があり、
        直前のアシスタント応答にも数値があり、両者が重ならないときに限る。
        判定材料が無いケースは従来どおり即確定 (挙動を変えない)。
        """
        corrected = {m for m in NUMBER_LITERAL_RE.findall(query or "")}
        prior = {m for m in NUMBER_LITERAL_RE.findall(self._prev_response or "")}
        if not corrected or not prior or (corrected & prior):
            return
        self._pending_correction = {
            "entry": entry,
            "corrected": corrected,
            "prior": prior,
        }
        logger.debug(
            "Correction held pending (corrected=%s vs prior=%s); "
            "the next turn decides", sorted(corrected), sorted(prior),
        )

    #: 実装は :func:`~backend.free.core.text_quality.VALUE_REJECTION_RE` へ移した。
    #: 記憶層 (``sleep.assertion_curator``) も同じ判定を要るため
    #: (採らなかった値を world_fact にしない)、書き写すと必ず食い違う。
    _VALUE_REJECTION_RE = VALUE_REJECTION_RE

    @classmethod
    def _values_adopted(cls, response: str, values: set[str]) -> bool:
        """応答が ``values`` のいずれかを **自分の答えとして採った** か。

        実装は :func:`~backend.free.core.text_quality.value_was_adopted`
        (pillar をまたぐ純粋関数の正準置き場)。
        """
        return value_was_adopted(response, values)

    def _settle_pending_correction(self, response: str) -> None:
        """保留中の訂正を、応答が採った値で確定 / 撤回する。

        - 応答が **訂正前の値** を含み、訂正値を **採用していない** →
          アシスタントは自分の答えを維持した = ユーザーの主張を採らなかった
          → **撤回**。
        - それ以外 (訂正値を採用した / どちらも出てこない) → 保留のまま次ターンへ
          持ち越すか、そのまま確定。

        「採用していない」は出現の有無ではなく ``_values_adopted`` で見る。
        アシスタントは「約100kmという値は事実と異なります」のように **打ち消し
        ながら値に言及する** ため、出現だけを見ると採用と誤判定する。

        撤回は前 entry の ``signals`` を書き換える。``_prev_entry`` への遡及
        マーク (tool_routing_false_negative 等) と同じで、buffer は同一オブジェクトを
        保持しているため反映される。
        """
        pending = self._pending_correction
        if pending is None:
            return
        found = {m for m in NUMBER_LITERAL_RE.findall(response or "")}
        if not (found & pending["prior"]):
            # まだ判断材料が出ていない (「承知しました。」等)。次ターンへ持ち越す。
            return
        self._pending_correction = None
        if self._values_adopted(response, pending["corrected"]):
            return
        entry = pending["entry"]
        entry.signals.correction_candidate = None
        entry.signals.user_correction = None
        entry.signals.correction_detected_by = "retracted_not_accepted"
        self._touch(entry)
        logger.info(
            "Correction retracted: the assistant kept its original value "
            "(prior=%s) instead of the user's claim (%s); not learning from it",
            sorted(pending["prior"]), sorted(pending["corrected"]),
        )

    def _detect_correction(
        self, query: str, *, mode: str = "chat", session_id: str = "",
    ) -> tuple[str | None, str | None]:
        """ユーザーの訂正のうち **アシスタントの誤りに対するもの** を検出する。

        字句一致で拾った候補 (``hardcoded`` / ``same_target``) は判定点
        ``correction_attribution`` (:mod:`backend.free.agent.correction_attribution_gate`、
        記録付き) で帰属を判定し、ユーザー自身の申告訂正 (「すみません、火曜では
        なく水曜でした」) と編集依頼・質問を除外する。除外しないと、正しく応答した
        ターンが失敗として学習される (実データでは訂正 24 件のうち 19 件が該当した)。
        帰属には同じセッションの前のユーザー発話と直前の応答を渡す — 語形で
        決まらない対比は旧値を誰が先に述べたかで決める (2026-10-02 監査 D01#4)。

        直前ターンが実際に失敗している場合 (``_prev_turn_failed``) でも帰属判定は
        **飛ばさない**。直前の失敗が覆すのは「直前の出力は正しかった」を前提に
        外した編集・書き直しの依頼 (:data:`REDO_SAME_OUTPUT_EVIDENCE`) だけで、
        書込みが失敗した直後の「同じファイルに保存し直して」は訂正に戻す。本人の
        言い直し・質問・仮定の変更は直前の成否と無関係に訂正ではない — 以前は
        帰属判定ごと飛ばしており、成否の検証器の誤検知 (2026-10-02 監査 D03#3 の
        「1 does not round from 0.383333」) の次の「訂正です。身長は175cmではなく
        178cmでした。」が候補になった。
        """
        raw, detected_by = self._detect_correction_lexical(query, mode=mode)
        if raw is None:
            return raw, detected_by
        from backend.free.agent.correction_attribution_gate import (
            attribution_target,
            correction_attribution_verdict,
        )

        verdict = correction_attribution_verdict(
            mask_quoted_speech(query),
            prev_user=attribution_prev_user([
                e.query or "" for e in (getattr(self.buffer, "entries", None) or [])
                if getattr(e, "session_id", "") == session_id
            ]),
            prev_response=self._prev_response,
            prev_query=self._prev_query or "",
        )
        target = attribution_target(verdict)
        if target == "assistant":
            return raw, detected_by
        if self._prev_turn_failed and verdict.evidence in REDO_SAME_OUTPUT_EVIDENCE:
            # 直前の出力が失敗しているので「やり直し」は誤りへの反応 (上の説明)。
            return raw, detected_by
        logger.debug(
            "Correction candidate reclassified as %s (%s; not an assistant "
            "error); dropping: %s", target, verdict.evidence, query[:60],
        )
        return None, None

    def _detect_correction_lexical(
        self, query: str, *, mode: str = "chat",
    ) -> tuple[str | None, str | None]:
        """訂正候補の字句検出（多段: ハードコード / 直前失敗 /
        同一成果物 + 弱パターン）

        旧・層2 (学習済み correction パターン照合) は 2026-07-21 に廃止した。
        学習語が訂正表現ではなく話題語 (「会話」「質問」「カーディナリティ」等)
        だったため、「正解です。では次の質問です」のような肯定評価+話題転換
        ターンまで訂正と誤検出し (偽陽性率 ~85%、経験 65 件の実測)、その語を
        再学習する自己強化ループで汚染が増殖していた。詳細は
        ``CORRECTION_PATTERNS`` の定義コメント参照。

        Returns:
            (correction_text, detected_by): 検出テキストと検出元
            detected_by: "hardcoded" | "record_divergence" | "prev_failed"
            | "same_target" | None。``prev_failed`` は直前ターンの失敗 **かつ**
            同一成果物の再指定 / 短い否定だけの発話 (状況証拠単独では立てない)。

        引用 (鉤括弧) の内側は本人の主張ではないので、字句照合には
        :func:`mask_quoted_speech` を通したコピーを使う。返す ``correction_text``
        は記録用に原文の ``query`` を保つ (2026-09-09 監査 G-01 系)。
        """
        masked = mask_quoted_speech(query)
        # 1. ハードコードパターン（高確度、優先）
        for pattern in CORRECTION_PATTERNS:
            if pattern.search(masked):
                return query, "hardcoded"

        # 1a. 記録との食い違いの指摘。誤りを名指す語を一つも含まない訂正を
        # 2 条件 AND で拾う (``cites_record_divergence`` の説明を参照)。
        if cites_record_divergence(masked):
            return query, "record_divergence"

        # 1b. create モードの実行結果報告。訂正対象 (直前ターン) が無い最初の
        # ターンでは新規の質問/依頼である可能性が高く、文末が新規依頼の完結形
        # なら報告ではなく仕様/質問の可能性が高いため、いずれも除外する。
        # 報告語彙も除外ガードも JA / EN 両方を locale に依らず見る。片側だけだと
        # 「locale='en' のまま日本語で "動かない" と報告した」ターンが訂正として
        # 数えられず、create の経験に訂正シグナルがほぼ発生しない状態
        # (2026-07-18 に本語彙を足した動機そのもの) へ逆戻りする。
        # 除外ガードを同時に union するのが要点 — 語彙だけ広げると、反対言語の
        # 新規依頼 ("Please make the button not work when hovered") が
        # 文頭アンカーに掛からないまま報告として拾われる。
        if (
            is_create_mode(mode)
            and self._prev_query is not None
            and not matches_either(
                masked.strip(),
                _CREATE_FAILURE_REPORT_EXCLUDE_RE,
                _CREATE_FAILURE_REPORT_EXCLUDE_RE_EN,
            )
        ):
            for pattern in CREATE_FAILURE_REPORT_PATTERNS_ALL:
                if pattern.search(masked):
                    return query, "hardcoded"

        # 2. 直前ターンが失敗 ([failed] 応答等) → 次ターンは訂正候補。ただし
        #    失敗の直後という状況証拠だけでは足りない — 話題を変えた新規依頼も
        #    同じ位置に来る。同一成果物の再指定、または発話全体が短い否定
        #    (「ちがう」「だめ」「not that」) のときだけ prev_failed とする。
        #    字句の訂正語彙を含む発話は 1. で既に拾われている。
        if self._prev_turn_failed and (
            self._same_target_path(query) or is_short_negative_feedback(query)
        ):
            return query, "prev_failed"

        # 3. 同一出力先パスの再指定 + 弱い訂正語 (「〜ではなく」等)
        if self._same_target_path(query) and any(
            p.search(masked) for p in WEAK_CORRECTION_PATTERNS_ALL
        ):
            return query, "same_target"

        return None, None

    def _learn_tool_routing_from_false_negative(self, query: str) -> None:
        """ツールルーティング false_negative 時: クエリからキーワードを tool_routing として学習

        ツール実行されなかったがユーザーが手動で要求した場合、
        クエリに含まれる意図キーワードを tool_routing カテゴリとして学習する。

        学習可否の判定は ``LearnedPatternStore.extract_tool_routing_keywords``
        に集約する (Level 1 バッチ側 ``_evolve_tool_routing_patterns`` と共通):

        - クエリ自体にツールシグナルが無ければ学習しない (2026-07-18:
          「読書いいですね。最近何か面白い本を読みましたか？」から感想語
          「面白」が誤学習され、雑談中に run_command 判定を誘発した実
          インシデントの再発防止。遡及ヒューリスティックはノイズが多い)。
        - 動作指示語のみ学習し、話題名詞・言語タスク語 (「説明」等) は
          除外する (2026-07-20: 学習済み「説明」w=0.630 が知識質問への
          run_command 誘導を誘発し得た件の再発防止)。
        """
        if self._learned_patterns is None:
            return
        keywords = self._learned_patterns.extract_tool_routing_keywords(query)
        if not keywords:
            logger.debug(
                "Skipping tool_routing learning: no learnable keyword "
                "in query=%s", query[:50],
            )
            return
        for kw in keywords:
            self._learned_patterns.add_pattern(kw, category="tool_routing")

        logger.info(
            "Learned tool_routing patterns from false_negative: %s",
            json.dumps(keywords[:5], ensure_ascii=False),
        )

    def _learn_long_form_from_signal(self, query: str) -> None:
        """長文ルーティング success / false_negative 時: クエリからキーワードを学習

        長文分類が成功した、またはユーザが手動で長文を再要求した場合、
        クエリに含まれる意図キーワードを ``category="long_form"`` として学習する。
        ルータの ``_detect_long_form_learned()`` がこの語彙を参照する。
        """
        if self._learned_patterns is None:
            return

        # パス片 / URL 片 / 汎用ファイル操作語は long_form の文書種別シグナルでは
        # ないため学習から除外する (出力先指定の自己学習による誤ルーティング防止)。
        keywords = [
            kw for kw in self._learned_patterns.extract_intent_keywords(query)
            if self._learned_patterns.is_long_form_learnable(kw)
        ]
        for kw in keywords:
            self._learned_patterns.add_pattern(kw, category="long_form")

        if keywords:
            logger.info(
                "Learned long_form patterns from signal: %s",
                json.dumps(keywords[:5], ensure_ascii=False),
            )
