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
from pathlib import Path
from typing import TYPE_CHECKING

from backend.free.core.intent_vocab import (
    EXPLICIT_WINDOWS_PATH_RE,
    NUMBER_LITERAL_RE,
    REFERENTIAL_WRITE_TARGET_RE,
)
from backend.free.core.locale_patterns import is_en_locale, select_locale_variant
from backend.free.core.session_mode import is_create_mode
from backend.free.core.response_arithmetic import find_arithmetic_contradictions
from backend.free.core.text_quality import (
    claims_completed_state_change,
    contradicts_measured_values,
    has_broken_ja_spacing,
    has_chinese_token_leak,
    retracts_own_conclusion,
    violates_length_constraint,
)
from backend.free.core.text_similarity import (
    bigram_coverage,
    bigram_cosine,
    content_bigram_cosine,
)
from backend.free.learning.level0_instant import (
    RESPONSE_FULL_CAP,
    RESPONSE_SUMMARY_CAP,
    ExperienceBuffer,
    ExperienceEntry,
    FeedbackSignals,
    truncate_at_boundary,
)
from backend.log_config import get_logger
from backend.utils import utc_now

if TYPE_CHECKING:
    from backend.debug_logger import DebugLogger
    from backend.free.agent.learned_patterns import LearnedPatternStore

logger = get_logger("agent.feedback")

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
#: ``assistant`` を返すため、この除外より先に確定する。丁寧形の「どう違います
#: か」はその ``違います`` に先に一致するのでここには到達しない — 判別できない
#: ものは ``assistant`` に倒すという本モジュールの方針どおり、順序は変えない。
_ASKS_ABOUT_DIFFERENCE_RE = re.compile(
    r"(?:どう|どの(?:よう|ように)|どこ(?:が|に)|何(?:が|は)|なに(?:が|は))"
    r"[^。！？\n]{0,12}?違う",
)

#: 出力形式・言語の変更依頼。内容の誤りを指していない。
_REFORMAT_REQUEST_RE = re.compile(
    r"同じ内容を.{0,10}(?:日本語|英語|中国語)で"
    r"|(?:日本語|英語)で(?:説明し直|書き直|言い直)",
)

#: ユーザー自身の申告訂正。謝罪 / 自己の過去発言への言及 + 事実の言い換え。
_SELF_CORRECTION_RE = re.compile(
    r"(?:すみません|すいません|ごめん|失礼しました)[、,。\s]*"
    r".{0,30}?(?:ではなく|じゃなく|の間違い|間違えました)"
    r"|(?:先ほど|さっき|前に)\s*[「『]?.{0,20}?[」』]?\s*と(?:言|申)"
    r"|^訂正です"
    r"|あ[、,]?\s*間違えました"
    r"|実はこれは",
)

#: アシスタントの出力を指す参照。これがあれば自己訂正ではない。
_ASSISTANT_OUTPUT_REF_RE = re.compile(
    r"その(?:計算|答え|回答|結果|数字|値)"
    r"|(?:最後|最初)の.{0,6}(?:行|文|項目).{0,10}(?:なって|です)"
    r"|取り違え|間違っていませんか|違います|誤りです"
    r"|のはずです|ではありませんか|ませんでしたか"
    r"|(?:計算|回答|答え)しましたよね",
)


def classify_correction_target(query: str) -> str:
    """訂正候補の帰属を返す: ``assistant`` / ``self`` / ``not_correction``。

    純粋関数。``assistant`` のみが「直前のアシスタント応答が誤っていた」を
    意味する。判別できないものは ``assistant`` に倒す (現行挙動を維持)。
    """
    # アシスタント出力への言及が最優先。「すみません、その計算は違います」の
    # ように謝罪語と併存しうるため、自己訂正判定より先に見る。
    if _ASSISTANT_OUTPUT_REF_RE.search(query):
        return "assistant"
    if _SELF_CORRECTION_RE.search(query):
        return "self"
    if _ASKS_ABOUT_CORRECTION_RE.search(query):
        return "not_correction"
    if _ASKS_ABOUT_DIFFERENCE_RE.search(query):
        return "not_correction"
    if _REFORMAT_REQUEST_RE.search(query):
        return "not_correction"
    # 既存ファイルへの再保存を伴う依頼は編集であって訂正ではない。
    # 「3 番目を『ヘッドランプ』に直して、同じファイルに保存し直して」の
    # 「直して」が CORRECTION_PATTERNS に掛かるのを打ち消す。
    #
    # ただし内容への異議 (「プログラムではなくて文書が欲しいです」) を伴う場合は
    # 除外しない。編集依頼の体裁でも中身は訂正であり、除外すると本物の訂正を
    # 取りこぼす (既存テスト test_same_target_weak_pattern_detected の実例)。
    if REFERENTIAL_WRITE_TARGET_RE.search(query) and not any(
        p.search(query) for p in WEAK_CORRECTION_PATTERNS
    ):
        return "not_correction"
    return "assistant"


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


def _correction_attribution(query: str) -> str | None:
    """字句一致した訂正候補の **帰属** を返す。訂正でなければ ``None``。

    「訂正の言い回しが出ているか」(字句) と「誰が誤っていたか」(帰属) を 1 箇所に
    まとめる。下の 2 つの公開述語はここから作る — **記憶層と学習層で必要な
    「訂正」の範囲が違う**ので、判定の芯だけを共有して境界だけ分ける。

    戻り値は :func:`classify_correction_target` と同じ ``assistant`` /
    ``self`` / ``not_correction``。
    """
    if not query:
        return None
    lexical = any(p.search(query) for p in CORRECTION_PATTERNS) or (
        cites_record_divergence(query)
    )
    if not lexical:
        return None
    return classify_correction_target(query)


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
    """
    return _correction_attribution(query) in ("assistant", "self")


def cites_record_divergence(query: str) -> bool:
    """「自分はこう言ったはず」と記録の食い違いを指摘しているか (純粋関数)。

    誤りを名指す語を含まない訂正を拾うための 2 条件 AND
    (:data:`_USER_STATED_REF_RE` / :data:`_RECORD_DIVERGENCE_RE` の説明を参照)。
    """
    stated = select_locale_variant(_USER_STATED_REF_RE, _USER_STATED_REF_RE_EN)
    diverge = select_locale_variant(
        _RECORD_DIVERGENCE_RE, _RECORD_DIVERGENCE_RE_EN,
    )
    return bool(stated.search(query) and diverge.search(query))

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
        # 現在ロード中のモデル名 (GGUF ファイル名)。record() の base_model /
        # embedding_model が明示指定されないとき既定値として埋める。
        #
        # base_model は **モードで変わる**: create は model_paths.create_model を
        # ロードするため、chat と同じ名前を刻むとモデル隔離フィルタ (Level 2 が
        # current_model で経験を絞る) の意味が壊れる。実際 2026-07-26 時点の
        # 経験 182 件は create 分 2 件も chat のモデル名で記録されていた。
        # _base_model_name は chat 既定として保持し、record() 時に mode から解決する。
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

    def _resolve_base_model_name(self, mode: str) -> str:
        """記録時のモードで実際にロードされている base モデルの GGUF 名を返す。

        create は ``model_paths.create_model`` を読み込むため、chat と同じ名前を
        刻むと Level 2 のモデル隔離フィルタ (``current_model`` で経験を絞る) が
        別モデルの経験を混ぜてしまう。解決経路はモード切替が使う
        ``get_mode_generation_params`` に揃える (実際にロードされるモデルと
        刻む名前を同じ関数から採る)。

        config 未初期化などで解決できない場合は起動時に決めた既定
        (``_base_model_name``) へフォールバックし、記録自体は止めない。
        """
        try:
            from backend.config import get_mode_generation_params

            raw = get_mode_generation_params(mode)["model"]
        except Exception:
            return self._base_model_name
        return Path(raw).name if raw else self._base_model_name

    def record(
        self,
        query: str,
        response: str,
        mode: str = "chat",
        rag_used: bool = False,
        rag_source: str | None = None,
        rag_top1_score: float | None = None,
        agent_loops: int = 0,
        cartridge_ids: list[str] | None = None,
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
    ) -> ExperienceEntry:
        """シグナル収集 → ExperienceBuffer に記録"""
        if self._disabled:
            # 学習無効化中: シグナル検出 / パターン学習 / バッファ書込を全てスキップ。
            # 呼出側 (chat_recorder) は戻り値を直接参照しないが、署名互換のため
            # 最小限のダミーエントリを返す
            return ExperienceEntry(
                timestamp=utc_now(),
                mode=mode,
                query=query,
                response_summary=response[:RESPONSE_SUMMARY_CAP],
                response_full=truncate_at_boundary(response, RESPONSE_FULL_CAP),
                base_model=base_model or self._resolve_base_model_name(mode),
                embedding_model=embedding_model or self._embedding_model_name,
                cartridge_ids=cartridge_ids or [],
                signals=FeedbackSignals(),
            )
        turn_outcome = self._derive_turn_outcome(
            response, step_credits,
            query=query,
            tool_routing_false_positive=tool_routing_false_positive,
            long_form_false_positive=long_form_false_positive,
            action_blocked=action_blocked,
            measured_values=measured_values,
        )
        if turn_outcome == "failed":
            # 失敗ターンの成功シグナルは矛盾なので failed 側に倒す
            # (偽成功が learned_patterns の正例学習 / Level 1 fitness に
            # 伝播した 2026-07-15 の再発防止)。
            tool_routing_success = False
            long_form_success = False

        correction_text, detected_by = self._detect_correction(query, mode=mode)
        # 訂正と言い直しは排他: 訂正が検出されたターンを rephrase として
        # 二重学習しない
        rephrased = (
            correction_text is None and self._detect_rephrase(query)
        )

        signals = FeedbackSignals(
            turn_outcome=turn_outcome,
            rephrased_query=rephrased,
            rag_used=rag_used,
            rag_source=rag_source,
            rag_top1_score=rag_top1_score,
            agent_loops=agent_loops,
            user_correction=correction_text,
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
            long_form_success=long_form_success,
            long_form_false_positive=long_form_false_positive,
            long_form_false_negative=long_form_false_negative,
            step_credits=step_credits or [],
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            cached_prompt_tokens=cached_prompt_tokens,
        )

        entry = ExperienceEntry(
            timestamp=utc_now(),
            mode=mode,
            query=query,
            response_summary=response[:RESPONSE_SUMMARY_CAP],
            response_full=truncate_at_boundary(response, RESPONSE_FULL_CAP),
            base_model=base_model or self._resolve_base_model_name(mode),
            embedding_model=embedding_model or self._embedding_model_name,
            cartridge_ids=cartridge_ids or [],
            signals=signals,
        )

        # false_negative 事後検出 (訂正ゲート + capability 弁別):
        # 「前ターンが capability 未使用 → 当ターンで明示訂正 → 当ターンで capability
        # 使用」= 前ターンは capability を要すべきだった、という低ノイズの強い証拠。
        # 前 entry へ遡及マーク (Level 1 バッチ消費側が前クエリから学習: 正しい帰属) し、
        # 即時にも前クエリから学習する。前ターンが既に capability を使っていたケースは
        # 同一ターン false_positive (chat_recorder で検出) の領分なので扱わない。
        # ``prev_failed`` は「直前ターンが失敗した」だけで、当ターンが訂正だとは
        # 限らない (話題を変えただけでも立つ)。これを訂正として遡及学習すると、
        # 無関係な前クエリの語彙が long_form / tool_routing として学習される
        # (実測 2026-07-25: platform.cpu_count のツールエラーの次ターンで、
        # 前クエリ「CPU のコア数と Python のバージョン」から
        # [マシン, コア, バージョン] が long_form として学習された)。
        # 遡及学習は明示訂正 (hardcoded / same_target) に限定する。
        explicit_correction = (
            correction_text is not None and detected_by != "prev_failed"
        )
        current_routed_tool = tool_routing_success or tool_routing_false_positive
        if explicit_correction and self._prev_entry is not None:
            if current_routed_tool and not self._prev_routed_tool:
                self._prev_entry.signals.tool_routing_false_negative = True
                self._learn_tool_routing_from_false_negative(self._prev_entry.query)
            if long_form_used and not self._prev_used_long_form:
                self._prev_entry.signals.long_form_false_negative = True
                self._learn_long_form_from_signal(self._prev_entry.query)

        # 訂正・言い直し検出からのパターン学習は行わない (2026-07-21 廃止)。
        # correction: 話題語学習による偽陽性増殖 (_detect_correction の
        # docstring 参照)。rephrase: 書き込み専用の dead カテゴリで、match()
        # の参照箇所が存在しなかった (検出自体は文字重複率ベースで学習不要)。

        # ツールルーティング false_negative 時: 明示注入 (テスト等) では当該クエリから学習。
        if tool_routing_false_negative:
            self._learn_tool_routing_from_false_negative(query)

        # 長文ルーティング 成功 / false_negative 時: クエリからキーワードを
        # ``category="long_form"`` として学習する。
        # success 時はクエリ全体が長文分類のヒントになるため正例として学習。
        if long_form_success or long_form_false_negative:
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

        logger.info(
            "Recorded experience: mode=%s, rephrase=%s, correction=%s (by=%s)",
            mode, signals.rephrased_query,
            signals.user_correction is not None, detected_by,
        )

        # DebugLogger に Level 0 学習サイクルを記録
        dl = self._debug_logger
        if dl:
            dl.log_learning_cycle(cycle_num=0, data={
                "level": 0,
                "mode": mode,
                "buffer_size": self.buffer.count,
                "rephrase": signals.rephrased_query,
                "correction": signals.user_correction is not None,
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
        self._session_entries.clear()
        self._prev_query = None
        self._prev_entry = None
        self._prev_routed_tool = False
        self._prev_used_long_form = False
        self._prev_turn_failed = False
        self._pending_correction = None
        self._prev_response = ""

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
    ) -> str:
        """ターン成否 ("success" | "partial" | "failed") を決定論導出する。

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
            return "partial" if _DONE_MARKER_RE.search(text) else "failed"
        if step_credits and all(
            not (c.get("credit") or 0) for c in step_credits
        ):
            return "failed"
        if tool_routing_false_positive or long_form_false_positive:
            return "failed"
        if _is_user_echo(query, text):
            return "failed"
        broken = FeedbackCollector._find_broken_output_reason(text)
        if broken is not None:
            logger.info("Turn marked failed (%s)", broken)
            return "failed"
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
                return "failed"
        # 実測値を注入したのに別の数を述べている = 同じく矛盾。
        # 2026-08-22 ライブ監査: 実測 86 文字を注入済みで「100文字です」。
        mismatch = contradicts_measured_values(text, measured_values or {})
        if mismatch is not None:
            logger.info("Turn marked failed (%s)", mismatch)
            return "failed"
        # 明示された文字数指定を破っている = 指定は本文にあり長さは数えるだけ
        # なので、これも推定ではなく矛盾。2026-08-22 ライブ監査の
        # 「ちょうど100文字で」→ 86 文字は success として学習に入っていた。
        broken_length = violates_length_constraint(query, text)
        if broken_length is not None:
            logger.info("Turn marked failed (%s)", broken_length)
            return "failed"
        return "success"

    @staticmethod
    def _find_broken_output_reason(text: str) -> str | None:
        """応答本文の決定論的な破綻を返す (無ければ ``None``)。

        判定器はすべて横断基盤の純粋関数。few-shot の内容棄却ゲートと同じ
        実体を共有する (片方だけ直る状態を作らない)。
        """
        contradictions = find_arithmetic_contradictions(text)
        if contradictions:
            return f"arithmetic contradiction: {contradictions[0]}"
        if has_broken_ja_spacing(text):
            return "broken JA spacing"
        if has_chinese_token_leak(text):
            return "Chinese token leaked into JA response"
        if retracts_own_conclusion(text):
            return "response retracts its own conclusion mid-answer"
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

        if is_en_locale():
            return bigram_cosine(prev, curr) >= REPHRASE_THRESHOLD_EN
        return content_bigram_cosine(prev, curr) >= REPHRASE_THRESHOLD

    def _apply_self_retraction(self, signals, response: str) -> None:
        """アシスタント自身の撤回を検出し、**直前ターン**を failed へ落とす。

        撤回した本ターンは誤りを直した側なので失敗ではない。誤っていたのは
        1 つ前のターンであり、``_prev_entry`` はまだバッファ内にあるので
        その ``turn_outcome`` を書き換えれば正しい側に選択圧が掛かる
        (``ExperienceBuffer`` は entry オブジェクトを保持しており、保存時に
        書き換え後の値が直列化される)。

        既に failed / partial のエントリは触らない (格上げも格下げもしない)。
        """
        if not detect_assistant_self_retraction(response):
            return
        signals.assistant_self_retraction = True
        prev = self._prev_entry
        if prev is None or prev.signals.turn_outcome != "success":
            return
        prev.signals.turn_outcome = "failed"
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

    #: 「100km ではありません」のように、直後で打ち消す言い回し。値に **言及した**
    #: ことと **採用した** ことを分ける。実測 (2026-08-22 の修正検証): 訂正の
    #: ターンで「東京と大阪の直線距離は約370kmです。約100kmという値は事実と
    #: 異なります。」と即座に反論したため、訂正値 100 が本文に現れて「採用」と
    #: 誤判定され、撤回が発火しなかった。
    _VALUE_REJECTION_RE = re.compile(
        r"(?:では?あり?ま?せん|ではなく|では無く|は誤り|は間違|"
        r"正しくありません|事実と異な|正確ではあ|ではないです|ではない)",
    )

    @classmethod
    def _values_adopted(cls, response: str, values: set[str]) -> bool:
        """応答が ``values`` のいずれかを **自分の答えとして採った** か。

        単なる出現では判定しない。値の直後が打ち消しなら、言及はしていても
        採用はしていない (``_VALUE_REJECTION_RE`` の説明を参照)。
        """
        text = response or ""
        for value in values:
            start = 0
            while True:
                idx = text.find(value, start)
                if idx < 0:
                    break
                tail = text[idx + len(value): idx + len(value) + 14]
                if not cls._VALUE_REJECTION_RE.search(tail):
                    return True
                start = idx + len(value)
        return False

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
        entry.signals.user_correction = None
        entry.signals.correction_detected_by = "retracted_not_accepted"
        logger.info(
            "Correction retracted: the assistant kept its original value "
            "(prior=%s) instead of the user's claim (%s); not learning from it",
            sorted(pending["prior"]), sorted(pending["corrected"]),
        )

    def _detect_correction(
        self, query: str, *, mode: str = "chat",
    ) -> tuple[str | None, str | None]:
        """ユーザーの訂正のうち **アシスタントの誤りに対するもの** を検出する。

        字句一致で拾った候補 (``hardcoded`` / ``same_target``) は
        ``classify_correction_target`` で帰属を判定し、ユーザー自身の申告訂正
        (「すみません、火曜ではなく水曜でした」) と編集依頼・質問を除外する。
        除外しないと、正しく応答したターンが失敗として学習される (実データでは
        訂正 24 件のうち 19 件が該当した)。

        直前ターンが実際に失敗している場合 (``_prev_turn_failed``) は、字句に
        依らない独立した証拠があるため帰属判定を通さない。書込みが失敗した直後の
        「同じファイルに保存し直して」は編集依頼の体裁でも本物の訂正である。
        """
        raw, detected_by = self._detect_correction_lexical(query, mode=mode)
        if raw is None or self._prev_turn_failed:
            return raw, detected_by
        target = classify_correction_target(query)
        if target != "assistant":
            logger.debug(
                "Correction candidate reclassified as %s (not an assistant "
                "error); dropping: %s", target, query[:60],
            )
            return None, None
        return raw, detected_by

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
            | "same_target" | None
        """
        # 1. ハードコードパターン（高確度、優先）
        for pattern in CORRECTION_PATTERNS:
            if pattern.search(query):
                return query, "hardcoded"

        # 1a. 記録との食い違いの指摘。誤りを名指す語を一つも含まない訂正を
        # 2 条件 AND で拾う (``cites_record_divergence`` の説明を参照)。
        if cites_record_divergence(query):
            return query, "record_divergence"

        # 1b. create モードの実行結果報告。訂正対象 (直前ターン) が無い最初の
        # ターンでは新規の質問/依頼である可能性が高く、文末が新規依頼の完結形
        # なら報告ではなく仕様/質問の可能性が高いため、いずれも除外する。
        exclude_re = select_locale_variant(
            _CREATE_FAILURE_REPORT_EXCLUDE_RE, _CREATE_FAILURE_REPORT_EXCLUDE_RE_EN,
        )
        failure_patterns = select_locale_variant(
            CREATE_FAILURE_REPORT_PATTERNS, CREATE_FAILURE_REPORT_PATTERNS_EN,
        )
        if (
            is_create_mode(mode)
            and self._prev_query is not None
            and not exclude_re.search(query.strip())
        ):
            for pattern in failure_patterns:
                if pattern.search(query):
                    return query, "hardcoded"

        # 2. 直前ターンが失敗 ([failed] 応答等) → 次ターンは訂正候補
        if self._prev_turn_failed:
            return query, "prev_failed"

        # 3. 同一出力先パスの再指定 + 弱い訂正語 (「〜ではなく」等)
        weak_patterns = select_locale_variant(WEAK_CORRECTION_PATTERNS, WEAK_CORRECTION_PATTERNS_EN)
        if self._same_target_path(query) and any(
            p.search(query) for p in weak_patterns
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
