"""Few-shot 候補プール: 経験バッファから高品質な応答例を収集・管理する

経験バッファの成功例（高 fitness）を候補プールに蓄積し、
進化時に指示テキストと Few-shot 例の組み合わせを同時に最適化する。
候補間の多様性は文字 bi-gram コサイン類似度で保証する。

参考論文: PromptWizard (arXiv:2405.18369) の Few-shot 同時最適化
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, fields
from math import sqrt
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

# FewShotExample / format_fewshot_section は EvorefLoop pillar
# (agent) 所属の純粋 util に移動済。Learn 側はここから import する。
# format_fewshot_section は他モジュール (tests / 一部呼出元) が本モジュール経由で
# import するため re-export として保持する。
from backend.free.agent.prompt_utils import (
    FewShotExample,
    format_fewshot_section,  # noqa: F401  (re-export for tests)
)
from backend.free.core.session_mode import is_valid_session_mode, normalize_session_mode
from backend.free.core.intent_vocab import own_process_question
from backend.free.learning.corrected_pairs import (
    CorrectedPair,
    refers_to_previous_turn,
)
from backend.free.core.text_quality import (
    SYSTEM_MEASUREMENT_MARKER,
    SYSTEM_NOTE_TAIL_RE,
    asks_verbatim_excerpt,
    has_boilerplate_closing,
    has_broken_ja_spacing,
    has_chinese_token_leak,
    is_query_echo,
    retracts_own_conclusion,
    violates_length_constraint,
)
from backend.free.learning.fitness import defect_rate_fitness
from backend.free.learning.json_state_store import JsonPayload, JsonStateStore
from backend.free.core.response_arithmetic import find_arithmetic_contradictions
from backend.free.llm.json_schemas import FewShotQualityJudgement
from backend.free.memory.types import make_fact
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.debug_logger import DebugLogger
    from backend.free.memory.types import SemanticFact
    from backend.free.memory.views.learn import LearnFactView

logger = get_logger("learning.fewshot_pool")

# デフォルト設定
DEFAULT_POOL_SIZE = 50            # モード別最大プールサイズ
DEFAULT_MIN_FITNESS = 0.7         # プール追加の最低 fitness

#: GC で無条件に破棄する ``quality_score`` の下限。
#:
#: 採点は採用の **後** に走るので intake では掛けられず、GC はサイズ超過時の
#: 下位切りしか無かった。結果、低品質と採点された例がプールに滞留する
#: (実データ 2026-08-19、chat 50 件: ``quality_score`` は 0.9 が 28 件 /
#: 0.95 が 14 件 / 1.0 が 5 件 / **0.1 が 3 件**)。プールが上限に達するまで
#: 0.1 の 3 件も選択候補に残り、``select`` の重み付きサンプリングでも引かれうる。
#:
#: 0.5 は「採点器が明確に低いと言った」帯だけを切る位置。実データの分布は
#: 0.1 と 0.9 に二極化しており、中間帯を巻き込まない。
DEFAULT_MIN_QUALITY_SCORE = 0.5
DEFAULT_MAX_EXAMPLES = 3          # プロンプトに埋め込む最大 Few-shot 数
DEFAULT_DIVERSITY_THRESHOLD = 0.8  # コサイン類似度の上限（これ以上は重複とみなす）

# _calc_experience_fitness の生スコア = 共有欠陥率 fitness (``fitness.
# defect_rate_fitness``、[0,1]、1.0 = 欠陥なし) + few-shot 固有ボーナス
# (``FEWSHOT_BONUS_WEIGHTS``、[-0.6, +0.7])。
#
# 欠陥側の係数は Level 1 evolver 群と同じ ``fitness.DEFECT_WEIGHTS`` を使う
# (以前は rephrase -0.5 / correction -0.8 を PromptEvolver と手で揃えていたが、
# assistant_self_retraction / tool_routing_* が few-shot 側だけ抜けていた)。
#
# ``conversation_ended`` は加点しない。この信号は
# ``ExperienceBuffer._mark_loaded_conversations_ended`` が **読み込み時に全件へ
# 立てる**ため構造的に恒真で (実測 2026-08-18: 135/136 = 99.3%)、
# 「そのターンがどうだったか」を一切表さない (PolicyEvolver も 2026-08-18 に外した)。
#
# 正規化 [_FITNESS_LO, _FITNESS_HI] → [0,1] は旧マッピング (欠陥なし = 0.7308、
# ``min_fitness`` 既定 0.7 をぎりぎり通る) を保存するように選ぶ:
#   LO = -0.6  : base 0 (欠陥で床) + 負ボーナス上限 (loops -0.3 / verr -0.3)
#   HI = 1.59  : (1.0 - LO) / (HI - LO) = 0.7306 ≒ 旧最頻値 0.7308
# 生スコア最大 (1.0 + 0.7 = 1.7) は HI を超えるので 1.0 へクリップする
# (RAG top1=1.0 かつ長文完走+成功のみ)。テストフィクスチャでの写像:
#   欠陥なし                       1.00 → 0.731 (採用)
#   欠陥なし + agent_loops=2       0.90 → 0.685 (不採用、旧 0.692)
#   rephrased のみ                 0.40 → 0.457 (旧 0.538)
#   user_correction のみ           0.00 → 0.274 (旧 0.423)
#   rephrased + user_correction    0.00 → 0.274 (旧 0.231)
_FITNESS_LO = -0.6
_FITNESS_HI = 1.59

#: few-shot 固有のボーナス項 (共有 ``DEFECT_WEIGHTS`` に無い加減点)。
#: 手本としての価値 = 「根拠が強い / 停滞せず / 成果物が完走した」を段階的に
#: 表す連続値で、プールの fitness が 1 点に縮退しないようにする (select の
#: 重み付け / garbage_collect の lowest-fitness eviction が意味を持つ)。
#: 欠損シグナル (None / 0 / False) は加点も減点もせず中立。
FEWSHOT_BONUS_WEIGHTS: dict[str, float] = {
    # RAG top1 cos 類似 × 係数 (根拠が強いほど加点)
    "rag_top1": 0.3,
    # 反復 1 回超過ごとの減点 / その上限
    "agent_loop_step": 0.1,
    "agent_loop_cap": 0.3,
    # 長文生成: 完了率 × 係数 / 検証エラー 1 件ごとの減点とその上限 / 成功加点
    "long_form_completion": 0.3,
    "long_form_verr_step": 0.1,
    "long_form_verr_cap": 0.3,
    "long_form_success": 0.1,
}

#: quality_score が付いている例で、fitness と品質採点を混ぜる比率。
#:
#: fitness は「会話が正常終了したセッションに属するか」がほぼ唯一の判別項で、
#: 実データ 447 件では 89% が 2 値 (0.5278 / 0.8056) に潰れる。採用閾値 0.7 を
#: 超える 150 件のうち 145 件が同点で、順位付けの情報が事実上ない。品質採点は
#: この同点を割るために入れるので、fitness と同等の重みを与える。
_QUALITY_WEIGHT = 0.5


def _effective_fitness(example: FewShotExample) -> float:
    """順位付けに使う実効スコア (純粋関数)。

    ``quality_score`` 未採点 (採点モデル未接続 / 採点前) では従来どおり
    ``fitness`` そのものを返すため、degraded mode でも挙動は変わらない。
    """
    if example.quality_score is None:
        return example.fitness
    return (
        (1.0 - _QUALITY_WEIGHT) * example.fitness
        + _QUALITY_WEIGHT * example.quality_score
    )

# select_top_k のスコア合成: query 適合を主項に fitness を従に。
_TOPK_SIM_WEIGHT = 0.7
# select_top_k で類似計算する候補数の上限 (fitness 上位で足切りし hot path 遅延を抑制)。
_TOPK_SELECT_CAP = 64
# select_top_k の合成スコア下限。これ未満の候補はタスク無関連とみなして
# 返さない (2026-07-15: 第 3 スロットに 0.28-0.33 帯の無関連例 (W杯話題等) が
# 毎ターン混入していた)。
_TOPK_MIN_SCORE = 0.40

# select_top_k の query 類似度そのものの下限。合成スコアは fitness を 0.3 の重みで
# 含むため、プール入りの最低 fitness (DEFAULT_MIN_FITNESS=0.7) でも下駄 0.21 が
# 無条件に乗り、_TOPK_MIN_SCORE だけでは「高 fitness な無関連例」を弾けない
# (2026-07-27 実測: fitness 0.917 の無関連例が sim 0.220 で合成 0.429 を取り
# 自己紹介ターンに混入)。fitness は品質シグナルであって関連性シグナルではない
# ため、関連性は fitness と独立な必要条件として課す。
# 閾値根拠 (同実測、chat プール 50 件):
#   無関連 250 組 … p95=0.151 / p99=0.172 / max=0.220
#   関連する言い換え 6 組 … 0.354 / 0.463 / 0.473 / 0.601 (弱い 2 例は 0.213/0.238)
# 無関連例の混入は害 (トークン浪費 + 文体バイアス)、取りこぼしは無害 (few-shot 節
# を出さないだけ) の非対称性から、観測ノイズ上限を上回る 0.25 を採る。
_TOPK_MIN_SIM = 0.25

#: 密ベクトルで選ぶときの最小類似度。
#:
#: bi-gram の 0.25 とは **別の値**にする。同じ「0.25」でも分布が違う: 文字
#: bi-gram は語が重ならなければ 0 に張り付くが、埋め込みは無関連でも 0.1〜0.3 に
#: 散る。RAG の較正 (memory_threshold_calibration) が効いていればその relevance を
#: 使い、無ければこの静的値へ縮退する — 注入側の関連度ゲートと同じ作りにして、
#: 埋め込みモデルを替えたときに「黙って全部落とす / 全部通す」のどちらにも
#: ならないようにする。
_TOPK_MIN_SIM_DENSE_STATIC = 0.35

# タスク進捗ノート行 (エージェントの最終応答フォーマット)。
# meta_cognitive_utils._TASK_LOG_LINE_RE と同旨だが、pillar 境界
# (EvorefLearn → EvorefLoop の utils は import 対象外) のため最小実装を持つ。
_TASK_LOG_LINE_RE = re.compile(
    r"^\s*(?:[-*]\s*)?\[(?:done|failed|skipped)\]\s"
    r"|^\s*Written\s+\d+\s+bytes\s+to\s+\S",
)


def _response_is_task_log_only(response: str) -> bool:
    """応答がタスク進捗ノート行だけで構成されるかを判定する。"""
    lines = [ln for ln in response.split("\n") if ln.strip()]
    if not lines:
        return False
    return all(_TASK_LOG_LINE_RE.match(ln) for ln in lines)


# 内部足場の語彙。システムプロンプトの PROTECTED は
# 「[関連する記憶]・[参考情報]・ツール実行結果が『有ったか / 無かったか』自体を
# 話題にしない」「ご提示いただいた結果 等の内部的な言い回しを使わない」と定めて
# いるが、これを破った応答が few-shot の正例として保存されると、以後その言い回しを
# 手本として再生産してしまう (2026-07-28 実測: chat プール 50 件中 3 件が
# 「ご提示いただいたツール実行結果には…」「メモが参考情報として提供されています」
# 型で、いずれも fitness 0.806 の正例だった)。採用時点で弾く。
_INTERNAL_SCAFFOLD_RE = re.compile(
    r"ご提示いただいた"
    r"|参考情報として提供"
    r"|参考情報には(?:記載|情報)"
    r"|ツール実行結果に(?:は|も)"
    r"|会話履歴を確認したところ",
)


def _response_leaks_internal_scaffold(response: str) -> bool:
    """応答が内部足場の語彙をそのまま含むかを判定する (純粋関数)。"""
    return bool(_INTERNAL_SCAFFOLD_RE.search(response))


#: 日本語の語間に紛れ込んだ半角空白。正常な日本語では発生しない
#: (実測 2026-08-02: 学習データ削除前の応答 96 件中 0 件)。
#:
#: 学習データを全削除するとベースモデルが素の状態に戻り、
#: 「日本の三景 は、富士山、天橋立、伊勢神宮 です。」のような崩れた応答を出す。
#: この応答が経験として記録され、Level 1 が手本に採用すると **崩れが手本として
#: 固定される自己増幅ループ**になる (実測: プール 25 件中 17 件 = 68% が混入し、
#: いずれも fitness 0.806 で全ゲートを素通りしていた)。
#:
#: 補助タスク品質採点では分離できない。空白混入例の quality 平均 0.80 に対し
#: 正常例 0.89 と差が 0.09 しかなく、混入例に 0.95 が 4 件付いていた
#: (小型モデルは自分と同種の崩れを問題と認識できない)。算術矛盾と同じく
#: **決定論で拒否する**。
#: 判定器の実体は :mod:`backend.free.core.text_quality`。モデル切替時の品質
#: プローブ (:mod:`backend.free.llm.quality_probe`) と同じ判定を共有する
#: — 別々に定義すると片方だけ直る。
_response_has_broken_ja_spacing = has_broken_ja_spacing

#: 日本語の応答に中国語の語彙が紛れた例も手本にしない。語間空白と同じく
#: 「崩れた出力が手本として再生産される自己増幅」を断つための決定論ゲート。
#: 2026-08-16 ライブ監査 (Qwen3.8-27B): 「私について知っていること」を 2 度
#: 尋ねた両方で「名前**是**小川さんです。」と繋辞の ``是`` が出た (2 度目は
#: 1 度目の出力が文脈に残っていたための複写)。
_response_has_chinese_token_leak = has_chinese_token_leak


#: 発話時点の「いま」を指す語。これを含む問いへの答えは、その日にしか成立しない。
_PRESENT_TIME_RE = re.compile(
    r"今日|本日|昨日|明日|明後日|一昨日|今週|来週|先週|今月|来月|先月"
    r"|今年|来年|去年|昨年|現在|只今|ただいま"
    r"|(?<![A-Za-z])(?:today|tomorrow|yesterday|now|current)(?![A-Za-z])",
    re.IGNORECASE,
)

#: 具体的な日付・時刻・曜日のリテラル。年 4 桁を必須にして「5 月 3 日」のような
#: 毎年成立する記述 (祝日等) を巻き込まない。
_ABSOLUTE_DATETIME_RE = re.compile(
    r"\d{4}\s*[-/年]\s*\d{1,2}\s*[-/月]\s*\d{1,2}"
    r"|\d{1,2}\s*[:：]\s*\d{2}"
    r"|[月火水木金土日]曜日",
)

#: ローカル環境の絶対パス。この機械の、その時点の状態を述べた応答の目印。
_LOCAL_ABS_PATH_RE = re.compile(r"[A-Za-z]:\\|/(?:home|Users|mnt|opt|var)/")

#: ユーザーの個人属性を **値まで断定している** 応答。
#:
#: 日付・ローカルパスと同じ「その時点でしか成立しない」類型だが、こちらは
#: ユーザーが言い直した瞬間に古くなる。few-shot は数週間常駐するため、
#: 個人属性を含む例は必ずいつか嘘の手本になる。
#:
#: 実インシデント (2026-08-09): プールに
#: ``Q: 私の趣味は何でしたか？ / A: 小川博之さんの趣味は自転車と写真です。``
#: (quality_score 0.9) が常駐しており、**同一の質問に対する手本として**古い値を
#: 直接教えていた。SemMem 側で世代を畳んでも、手本が答えそのものを提示するので
#: 勝てない (実機で 3 回連続して古い値を回答)。
#:
#: 属性名 + 助詞/コロン + 値、の形だけを採る。「趣味について説明します」の
#: ような一般論は拾わない。
_PERSONAL_ATTRIBUTE_ASSERTION_RE = re.compile(
    r"(?:名前|氏名|趣味|誕生日|生年月日|好きな[^\s、。:：]{0,6}|ペット|飼っている)"
    r"\s*(?:は|：|:)\s*[^\s、。:：]",
)

#: 個人属性の主張が「ユーザーのもの」であることを示す帰属語。一般的な説明文
#: (「趣味は人によって違います」) を巻き込まないための第 2 条件。
#:
#: **一人称は入れない**。応答中の「私」はアシスタント自身を指すため、
#: 「私の名前はAliceです。」のような自己紹介まで棄却してしまう (実データ 50 件で
#: 検証した際に 3 件が誤検出になった)。アシスタントの名前は変わらないので
#: 揮発性ではなく、手本として正当。
_PERSONAL_ATTRIBUTION_RE = re.compile(
    r"(?:あなた|ユーザー)(?:の|は|が)"
    r"|さん(?:の|は|で)"
    r"|個人情報",
)



#: **問いの側** に現れるユーザー自身への言及。
#:
#: ``_PERSONAL_ATTRIBUTE_ASSERTION_RE`` は応答を見るが、揮発性を決めるのは
#: **問い** の方である。「あなたの / さん」を含まない短い答え方をされると
#: 応答側のゲートはまるごと素通りし、その手本は
#: **「この質問にはこう答える」** をモデルへ教え続ける。
#:
#: 実インシデント (2026-08-30 ライブ監査): 監査中の誤答がそのまま手本になり、
#: 検証セッションのプロンプトに以下が並んだ::
#:
#:     ### Example 1
#:     User: 私が決めたことは何でしたか。
#:     Assistant: 来月から毎朝6時に起きることです。
#:     ### Example 2
#:     User: 私が苦手だと言ったことは。
#:     Assistant: 毎朝6時に起きることです。      ← 誤答が手本に昇格
#:     ### Example 3
#:     User: 私について知っていることを全部教えてください。
#:     Assistant: …情報をお聞かせいただいていないため…  ← 「知らない」の手本
#:
#: 実機では実際に「私が苦手だと言ったことは。」へ「毎朝6時に起きることです。」を
#: 返しており、**手本が誤答を再生産する自己増幅**になっていた。
#:
#: 判定は「ユーザー自身への言及があるか」だけを見る広いゲート。棄却の方向は
#: 安全側 (手本が 1 件減るだけ) で、採用の方向は嘘の手本を数週間常駐させる。
#: 実データでの較正 (experience 313 件): 現行ゲート 22 件 + 本ゲート 62 件を
#: 棄却し、**229 件が残る**。
_USER_SELF_REFERENCE_RE = re.compile(
    r"(?:私|わたし|僕|ぼく|俺|おれ|自分|うち)\s*(?:の|は|が|に|も|自身)",
)


def _find_volatile_reason(query: str, response: str) -> str | None:
    """例が「その時点でしか成立しない」ものかを判定する (純粋関数)。

    few-shot は数週間〜数か月プールに常駐するため、発話時点の日付やこの機械の
    ファイル状態を述べた例は、翌日には **古い答えを手本として提示する** ものに
    変わる。品質採点では分離できない — 採点時点では正しいので、実測で
    quality_score 0.9 が付いていた (``_QUALITY_SYSTEM_PROMPT`` は「その場限りの
    固有値だけを述べた回答」を低く評価するよう指示しているが、小型モデルは
    従わない)。算術矛盾・語間空白と同じく決定論で拒否する。

    実インシデント (2026-08-07 ライブ監査): 2 日前に採用された
    ``Q: 今日は何月何日の何曜日ですか？ / A: 今日は2026年8月5日、水曜日です。``
    が「3 年前の今日は何曜日でしたか？」の手本として提示され、
    ``Q: README.md は何行で、何文字ありますか？ / A: 121 行あり、3353 文字です。``
    が同ファイルの行数質問の手本として提示された。

    Returns:
        棄却理由。揮発性でなければ ``None``。
    """
    if _PRESENT_TIME_RE.search(query) and _ABSOLUTE_DATETIME_RE.search(response):
        return "time-dependent answer (asserts a date/weekday valid only that day)"
    if _LOCAL_ABS_PATH_RE.search(query) or _LOCAL_ABS_PATH_RE.search(response):
        return "environment-dependent answer (describes local filesystem state)"
    if (
        _PERSONAL_ATTRIBUTE_ASSERTION_RE.search(response)
        and _PERSONAL_ATTRIBUTION_RE.search(response)
    ):
        return "user-dependent answer (asserts the user's personal attributes)"
    if _USER_SELF_REFERENCE_RE.search(query):
        return "user-dependent answer (the question is about the user themselves)"
    return None


#: システムが後付けした開示文の目印。``(注: …)`` の注記本体は
#: ``SYSTEM_NOTE_TAIL_RE`` が捕まえる。それとは別に、修復経路
#: (``length_disclosure_note``) が本文へ連結する「〜文字以内の指定に対し …
#: 文字です」/「指定は N 文字でしたが」型の開示文も手本には載せない。
_SYSTEM_DISCLOSURE_RE = re.compile(
    r"(?:文字以内|文字ちょうど|文字程度|文字)の指定に対し"
    r"|指定は\s*\d+\s*文字でしたが"
    r"|上の回答は\s*\d+\s*文字です",
)


def _response_carries_system_note(response: str) -> bool:
    """応答にシステムの開示注記 / 実測行が含まれているか (純粋関数)。

    記憶へは ``strip_system_notes`` で落とした本文が積まれるが、経験記録は
    長らく生の ``full_response`` を受けていたため、注記込みの応答が手本に
    昇格し、次のターンでモデルがそれを自分の文体として模倣していた
    (2026-09-02 監査 R-D1)。注記は「制約を破った」の印でもあるので、含む例は
    内容の良否に関わらず手本から外す。
    """
    text = response or ""
    if SYSTEM_MEASUREMENT_MARKER in text:
        return True
    if SYSTEM_NOTE_TAIL_RE.search(text):
        return True
    return bool(_SYSTEM_DISCLOSURE_RE.search(text))


def find_content_rejection(query: str, response: str) -> str | None:
    """few-shot 手本として不適な内容を決定論で判定する (純粋関数)。

    採用経路は 2 つ (``add_from_experiences`` / ``accept_from_artifact``) あり、
    さらに読込時の浄化 (``_from_payload``) も同じ判定を要る。片方にだけ置くと
    もう一方が抜け道になる (2026-08-07 監査時点で ``accept_from_artifact`` は
    全内容ゲートを迂回していた) ため、判定はここに集約して 3 箇所から呼ぶ。
    ``TestArtifactPathSharesContentGates`` /
    ``TestBrokenJaSpacingGate::test_load_applies_every_content_gate``
    (:mod:`backend.free.learning.tests.test_fewshot_pool`) が経路ごとに固定する。

    Returns:
        棄却理由 (ログ用の英語 1 行)。採用可なら ``None``。
    """
    # 応答がタスク進捗ノート形式 (- [done] ... Written N bytes) のみの例は
    # 「報告だけ出せば正解」バイアスを注入する (2026-07-15: この形式の例が
    # 毎ターン選択され本文なしの極小ファイル生成を誘発した)。
    if _response_is_task_log_only(response):
        return "task-log-only response"
    # システムの開示注記 / 実測行を含む応答は本文の一部として模倣される
    if _response_carries_system_note(response):
        return "contains a system note or disclosure"
    # 逐語の抜粋を求める依頼への応答は「ツール出力の逐語コピー」であって文体の
    # 手本ではない。載せると「ペイロードを貼るのが正解」というバイアスを注入し、
    # 別の質問にも本文の貼り付けを誘発する (2026-08-16 動作検証 T9)。
    if asks_verbatim_excerpt(query):
        return "response is a verbatim excerpt, not a style example"
    # 直前ターンを前提にした問い (「修正版にジェネレータを渡すと？」「2 案を採用。
    # 本文に…を入れて書き直して」) は、単独の手本にすると **何を指しているか
    # 分からない問いに具体的に答える** 型を教える。2026-09-05 のプール 47 件の
    # 先頭 4 件がこの形だった。文脈依存の判定は訂正ペアと共有する。
    if refers_to_previous_turn(query):
        return "query depends on the previous turn (anaphora / continuation)"
    # 内部足場の語彙を含む応答は PROTECTED 違反の実例なので手本にしない
    if _response_leaks_internal_scaffold(response):
        return "leaks internal scaffold vocabulary"
    # 日本語の語間に半角空白が紛れた応答は手本にしない。放置すると崩れた出力が
    # 手本として再生産される自己増幅ループになる (_JA_INTERWORD_SPACE_RE 参照)。
    if _response_has_broken_ja_spacing(response):
        return "broken JA spacing"
    # 日本語に中国語の語彙が紛れた応答も同じ理由で手本にしない。
    if _response_has_chinese_token_leak(response):
        return "Chinese token leaked into JA response"
    # 質問を逐語で繰り返しただけの応答は「問いをそのまま返すのが正解」という
    # バイアスを注入する (2026-08-04 ライブ監査: 同文 5 回で答えが出なくなった)。
    if is_query_echo(response, query):
        return "response only echoes the query"
    # PROTECTED の出力形式が禁止している締め文。禁止条項だけでは消えず、違反
    # 応答が fitness 最上位帯で手本に載って再生産されていた (2026-08-04)。
    if has_boilerplate_closing(response):
        return "boilerplate closing"
    # 計算が合わない応答。fitness は「会話が正常終了したセッションに属するか」
    # しか見ておらず内容の正しさを測らないため、誤答が最高位の正例として常駐
    # していた (実データで 4 件確認)。小型モデルの採点では捕まらない。
    contradictions = find_arithmetic_contradictions(response)
    if contradictions:
        return f"arithmetic contradiction: {contradictions[0]}"
    # 途中で結論を撤回した応答は、正しい結論に辿り着いていても手本にしない
    # (_SELF_RETRACTION_RE 参照)。
    if retracts_own_conclusion(response):
        return "response retracts its own conclusion mid-answer"
    # 明示された文字数指定を破った応答は「指定は無視してよい」というバイアスを
    # 注入する。判定器はターン成否と共有する (片方だけ直る状態を作らない)。
    broken_length = violates_length_constraint(query, response)
    if broken_length is not None:
        return f"violates the requested length: {broken_length}"
    # 自分が何を実行したかの問い (「どのツールを使ったか」「暗算したか」) は、
    # 正解がそのセッションの実行台帳にしか無い。どんな答えを手本にしても
    # **別のセッションでは必ず誤り**になるので、内容の正誤に関わらず除外する。
    #
    # 実インシデント (2026-08-23 ライブ監査): セット 1 で作話した
    # 「いま計算した中で、あなたが電卓ツールを使ったのはどれですか？」→
    # 「電卓ツールは使っていません。」(実際は calculate が 7 回実行済) が
    # few-shot に載り、セット 2 の同種の問いに Example 1 として提示されていた。
    # 欠陥が手本に昇格して自己増幅する経路。
    if own_process_question(query):
        return "self-report about this session's own tool use (not transferable)"
    return _find_volatile_reason(query, response)


#: few-shot 品質採点のシステムプロンプト。
#:
#: 問いは「正しいか」ではなく **「手本として提示したとき挙動が良くなるか」**。
#: few-shot の目的に直結させることで、正しいが一般化できない例 (その場限りの
#: 固有値回答等) も相対的に下がる。算術の正誤は決定論側が担当するため、ここでは
#: 検算を求めない (求めても小型モデルは実行できず、誤答に満点を付ける)。
#: JSON が壊れた応答から採点値を拾う救済用 (先頭の数値)。
_BARE_SCORE_RE = re.compile(r"-?\d+(?:\.\d+)?")

_QUALITY_SYSTEM_PROMPT = (
    "あなたは few-shot 手本の品質評価者です。"
    "提示された Q/A ペアを、他の質問に答えるときの手本としてモデルに見せた場合、"
    "モデルの振る舞いが良くなるかを 0.0〜1.0 で評価してください。"
    "評価軸: (1) 質問に正面から答えているか (2) 手本として一般化できるか "
    "(3) 内部的な言い回しや進捗ノートが混ざっていないか。"
    "その場限りの固有値だけを述べた回答や、質問と噛み合っていない回答は低く評価してください。"
)

#: fewshot_pool.json 上で追い出し済み hash (墓標) を持つ予約キー。
_EVICTED_KEY = "_evicted"
#: 墓標の上限 (モード別、FIFO)。16 byte hex × 2000 で数十 KB。
_EVICTED_CAP = 2000

# SemMem 書き戻し時の subject prefix
# ``harness.fewshot.*`` から ``learn.fewshot.*`` に移行済。owner は EvorefLearn。
LEARN_FEWSHOT_SUBJECT_PREFIX: str = "learn.fewshot."
"""SemMem 上の Few-shot ファクト subject prefix。
``learn.fewshot.<mode>.<example_id>`` の形式で書き出す。"""

DEFAULT_FEWSHOT_PREDICATE: str = "example_for"
"""Few-shot ファクトの predicate (常に固定)"""

EvolveWriteback = Literal["yaml", "semmem"]

#: _from_payload で JSON から復元する FewShotExample のキー集合。
#: 未知キー (旧スキーマ / フィールド削除) を無視して TypeError によるプール
#: 全消失を防ぐ (level0_instant の FeedbackSignals 復元と対称)。
_EXAMPLE_FIELD_NAMES = frozenset(f.name for f in fields(FewShotExample))


def _resolve_dense_min_sim() -> float:
    """密ベクトル選択の最小類似度 (較正があればそれ、無ければ静的値)。"""
    try:
        from backend.free.rag.memory_threshold_calibration import (
            get_active_calibration,
        )

        calibration = get_active_calibration()
    except Exception:
        return _TOPK_MIN_SIM_DENSE_STATIC
    if not calibration:
        return _TOPK_MIN_SIM_DENSE_STATIC
    return float(
        calibration.get("relevance_threshold", _TOPK_MIN_SIM_DENSE_STATIC),
    )


def _char_bigrams(text: str) -> Counter:
    """テキストから文字 bi-gram の出現頻度を返す"""
    t = text.lower().strip()
    if len(t) < 2:
        return Counter({t: 1}) if t else Counter()
    return Counter(t[i:i + 2] for i in range(len(t) - 1))


def _cosine_similarity(a: Counter, b: Counter) -> float:
    """2つの Counter 間のコサイン類似度を計算する"""
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    norm_a = sqrt(sum(v * v for v in a.values()))
    norm_b = sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _fewshot_bonus(signals: dict) -> float:
    """few-shot 固有ボーナス (:data:`FEWSHOT_BONUS_WEIGHTS`) の合計を返す。"""
    w = FEWSHOT_BONUS_WEIGHTS
    bonus = 0.0

    # RAG ヒット品質 (top1 cos 類似が高いほど根拠が強い)。欠損は中立
    rag_top1 = signals.get("rag_top1_score")
    if rag_top1 is not None:
        bonus += w["rag_top1"] * max(0.0, min(1.0, float(rag_top1)))

    # エージェント反復。<=1 は中立、多反復ほど停滞として減点
    loops = int(signals.get("agent_loops", 0) or 0)
    if loops > 1:
        bonus -= min(w["agent_loop_cap"], w["agent_loop_step"] * (loops - 1))

    # 長文生成 (used のときだけ評価)。完了率で加点・検証エラーで減点
    if signals.get("long_form_used", False):
        total = int(signals.get("long_form_units_total", 0) or 0)
        completed = int(signals.get("long_form_units_completed", 0) or 0)
        if total > 0:
            bonus += w["long_form_completion"] * min(1.0, completed / total)
        verr = int(signals.get("long_form_validation_errors", 0) or 0)
        if verr > 0:
            bonus -= min(w["long_form_verr_cap"], w["long_form_verr_step"] * verr)
        if signals.get("long_form_success", False):
            bonus += w["long_form_success"]
    return bonus


def _calc_experience_fitness(signals: dict) -> float:
    """経験 1 件のシグナルから段階的 fitness を計算する。

    欠陥側は Level 1 evolver 群と共有の :func:`fitness.defect_rate_fitness`
    (``DEFECT_WEIGHTS``、1 件なので「欠陥重みの和を 1.0 から引いた値」)。
    これに few-shot 固有のボーナス (:func:`_fewshot_bonus`) を足し、
    ``[_FITNESS_LO, _FITNESS_HI]`` を [0,1] へ線形写像する (写像表は定数の
    コメント参照)。``conversation_ended`` は恒真なので加点しない。
    """
    base = defect_rate_fitness([{"signals": signals}])
    score = (1.0 if base is None else base) + _fewshot_bonus(signals)
    return max(0.0, min(1.0, (score - _FITNESS_LO) / (_FITNESS_HI - _FITNESS_LO)))


class FewShotPool(JsonStateStore):
    """Few-shot 候補プール

    経験バッファから高品質な応答例を収集し、多様性を維持しながら
    候補プールを管理する。進化時のソースとして使用される。
    """

    _state_logger = logger

    def __init__(
        self,
        pool_size: int = DEFAULT_POOL_SIZE,
        min_fitness: float = DEFAULT_MIN_FITNESS,
        min_quality_score: float = DEFAULT_MIN_QUALITY_SCORE,
        max_examples: int = DEFAULT_MAX_EXAMPLES,
        diversity_threshold: float = DEFAULT_DIVERSITY_THRESHOLD,
        debug_logger: DebugLogger | None = None,
        *,
        learn_view: LearnFactView | None = None,
        semmem_writeback_scope: str = "global",
        evolve_writeback: EvolveWriteback = "yaml",
        base_model_id: str = "",
    ) -> None:
        """
        Args:
            pool_size: モード別最大プールサイズ (yaml モード時のみ有効。
                semmem モードではプール側 GC を停止し、SemMem 側
                ``semmem_limits.policy`` + ``gc_strategy=lowest_score`` に委譲)
            min_fitness: プール追加の最低 fitness
            min_quality_score: GC で無条件に破棄する quality_score の下限
            max_examples: プロンプトに埋め込む最大 Few-shot 数
            diversity_threshold: コサイン類似度の上限
            debug_logger: DebugLogger (任意)
                bootstrap / SemMem 書込の経路は全て本 view 経由に一本化される。
                ``evolve_writeback="semmem"`` 時に必須。
            semmem_writeback_scope: 書き込み先 scope (``global`` または
                ``project:<id>``)
            evolve_writeback: ``"yaml"`` (従来動作) / ``"semmem"``
                (SemMem に新規 fewshot ファクトを書き込み、永続化を SemMem に
                委譲する)
        """
        self.pool_size = pool_size
        self.min_fitness = min_fitness
        self.min_quality_score = min_quality_score
        self.max_examples = max_examples
        self.diversity_threshold = diversity_threshold
        self._debug_logger = debug_logger

        # LearnFactView 経由の writeback に一本化
        self._learn_view: LearnFactView | None = learn_view
        self._semmem_writeback_scope: str = semmem_writeback_scope
        self._evolve_writeback: EvolveWriteback = evolve_writeback
        # base 学習パーティションの active モデルスラグ。空 = partition 無効
        # (subject はレガシー ``learn.fewshot.<mode>.<id>`` 形式に縮退)。
        self._base_model_id: str = base_model_id

        # モード別のプール: mode → list[FewShotExample]
        self._pools: dict[str, list[FewShotExample]] = {}
        # キャッシュ: id → bi-gram Counter (採用済 example のみ)
        self._bigram_cache: dict[str, Counter] = {}
        # 明示 dedup 用: mode → {content_hash}。プール内の example と 1:1 で同期する
        # (採用で add、eviction で discard)。diversity_threshold と独立に厳密重複を排除。
        self._seen_hashes: dict[str, set[str]] = {}
        # 追い出した例の content hash (墓標)。プール上限 / GC で追い出した例は
        # ``_seen_hashes`` からも外れるため、次 tick の add_from_experiences で
        # 同じ経験が新規候補として復活し別の例を追い出していた — 数百件を毎回
        # 採否判定しながらプールは上限に張り付いたまま正味は縮む (2026-09-04
        # 監査: added 160→305/tick、GC 後 46→41→39)。一度追い出した例は
        # 再採用しない。挿入順 dict = FIFO で上限 ``_EVICTED_CAP`` 件/モード。
        self._evicted_hashes: dict[str, dict[str, None]] = {}
        # 内容判定で却下した候補 (プロセス内のみ、永続化しない)。却下理由は
        # 内容から決まる純粋判定なので、同じ (query, response) を再判定する
        # 意味は無い。
        self._rejected_hashes: dict[str, set[str]] = {}

    # ── SemMem 書き戻しヘルパ ───────────────────────────

    @property
    def evolve_writeback(self) -> EvolveWriteback:
        """現在の書き戻しモード"""
        return self._evolve_writeback

    def is_semmem_writeback_active(self) -> bool:
        """SemMem 書き戻しが有効かつ LearnFactView が注入済か"""
        return (
            self._evolve_writeback == "semmem"
            and self._learn_view is not None
        )

    def set_learn_view(
        self,
        learn_view: LearnFactView | None,
        *,
        writeback_scope: str | None = None,
        evolve_writeback: EvolveWriteback | None = None,
    ) -> None:
        """LearnFactView を動的に差し替える (テスト・lifespan 後注入用)。"""
        if learn_view is not None:
            self._learn_view = learn_view
        if writeback_scope is not None:
            self._semmem_writeback_scope = writeback_scope
        if evolve_writeback is not None:
            self._evolve_writeback = evolve_writeback

    def set_base_model_id(self, base_model_id: str) -> None:
        """base 学習パーティションの active モデルスラグを差し替える。

        起動時の ``_activate_learning_partition`` と、ランタイム base 切替の
        ``backend.factory._learning_rebind.rebind_base_learning`` から呼ばれ、以後の
        writeback / bootstrap が当該モデルの fewshot ファクト
        (``learn.fewshot.<model>.*``) のみを対象にする。空文字でレガシー縮退。
        プール内容は差し替えないので、切替時は :meth:`reset` → bootstrap / load
        で新モデル分を張り直す。
        """
        self._base_model_id = base_model_id or ""

    def reset(self) -> None:
        """in-memory プールを空にする (パーティション切替で新モデル分を読み直す前に呼ぶ)。"""
        self._pools = {}
        self._bigram_cache = {}
        self._seen_hashes = {}

    @staticmethod
    def _build_subject(base_model_id: str, mode: str, example_id: str) -> str:
        """``learn.fewshot.<model>.<mode>.<example_id>`` 形式の subject を構築する。

        ``base_model_id`` が空のときは partition 無効としてレガシー
        ``learn.fewshot.<mode>.<example_id>`` (2 段) へ縮退する。
        """
        if base_model_id:
            return f"{LEARN_FEWSHOT_SUBJECT_PREFIX}{base_model_id}.{mode}.{example_id}"
        return f"{LEARN_FEWSHOT_SUBJECT_PREFIX}{mode}.{example_id}"

    @staticmethod
    def _example_to_object(example: FewShotExample) -> str:
        """FewShotExample を SemMem ファクトの object 用 JSON 文字列に変換"""
        # quality_score も載せる。落とすと再起動ごとに未採点へ戻り、毎回
        # 採点し直すうえ ``min_quality_score`` の GC が効かなかった
        # (2026-09-02 監査 R-D3)。
        return json.dumps(
            {
                "query": example.query,
                "response": example.response,
                "mode": example.mode,
                "fitness": float(example.fitness),
                "added_at": example.added_at,
                "quality_score": example.quality_score,
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _example_from_fact(fact: SemanticFact) -> FewShotExample | None:
        """SemMem ファクトから FewShotExample を復元する。

        破損ファクトは ``None`` を返してスキップする。
        ``id`` は subject 末尾セグメント (``learn.fewshot.<mode>.<id>``) を
        信頼する
        """
        try:
            payload = json.loads(fact.object)
        except (TypeError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        # subject 末尾セグメントを ID として採用
        subject = fact.subject
        if not subject.startswith(LEARN_FEWSHOT_SUBJECT_PREFIX):
            return None
        rest = subject[len(LEARN_FEWSHOT_SUBJECT_PREFIX):]
        # rest は新形式 "<model>.<mode>.<id>" または レガシー "<mode>.<id>"。
        # mode は {chat, create} に限られるため先頭/2 番目セグメントで判別する。
        segs = rest.split(".")
        if len(segs) >= 2 and is_valid_session_mode(segs[0]):
            mode_part, id_part = segs[0], ".".join(segs[1:])
        elif len(segs) >= 3 and is_valid_session_mode(segs[1]):
            mode_part, id_part = segs[1], ".".join(segs[2:])
        else:
            return None
        if not id_part:
            return None
        raw_score = payload.get("quality_score")
        quality_score = (
            float(raw_score) if isinstance(raw_score, (int, float)) else None
        )
        return FewShotExample(
            id=id_part,
            query=str(payload.get("query", "")),
            response=str(payload.get("response", "")),
            mode=str(payload.get("mode", mode_part)),
            fitness=float(payload.get("fitness", 0.0)),
            added_at=str(payload.get("added_at", "")),
            quality_score=quality_score,
        )

    def _writeback_example_fact(
        self,
        example: FewShotExample,
    ) -> SemanticFact | None:
        """新規 FewShotExample を SemMem に fewshot ファクトとして書き出す。

        LearnFactView が未注入 / writeback モードが ``yaml`` の場合は
        ``None`` を返して no-op する。``type="policy"``
        → ``type="fewshot"`` に変更 (FACT_OWNERSHIP の fewshot 所有権に整合)。
        """
        if not self.is_semmem_writeback_active():
            return None
        view = self._learn_view
        assert view is not None  # is_semmem_writeback_active で保証

        subject = self._build_subject(self._base_model_id, example.mode, example.id)
        fact_mode: str = normalize_session_mode(example.mode)
        new_fact = make_fact(
            subject=subject,
            predicate=DEFAULT_FEWSHOT_PREDICATE,
            object_=self._example_to_object(example),
            type="fewshot",
            scope=self._semmem_writeback_scope,
            mode_origin=fact_mode,  # type: ignore[arg-type]
            confidence=float(example.fitness),
            auto_evolved=True,
            eval_metric={"fitness": float(example.fitness)},
        )
        try:
            added = view.add_fewshot_fact(new_fact)
        except ValueError as exc:
            logger.warning(
                "fewshot_pool semmem writeback failed: subject=%s err=%s",
                subject, exc,
            )
            return None
        logger.debug(
            "fewshot_pool semmem writeback: subject=%s id=%s fitness=%.3f",
            subject, added.id, example.fitness,
        )
        return added

    def bootstrap_from_semmem(self) -> int:
        """SemMem の ``learn.fewshot.*`` ファクトから in-memory プールを再構築する

        LearnFactView 経由で active な fewshot (legacy の type="policy" も
        受容) を集め、subject 末尾の ``<mode>.<id>`` をキーにモード別プール
        へ展開する。同一 ``id`` が複数ストアにあれば最初に見つけたものを
        優先する。

        Returns:
            復元した FewShotExample 数の合計
        """
        view = self._learn_view
        if view is None:
            return 0
        seen_ids: set[str] = set()
        loaded = 0
        # partition 有効時は active モデルの fewshot ファクトのみを hydrate する
        # (``learn.fewshot.<model>.``)。未知モデルは 0 件 = 空プール = ゼロから学習。
        # partition 無効時は従来どおり全 fewshot を対象 (レガシー縮退)。
        prefix = (
            f"{LEARN_FEWSHOT_SUBJECT_PREFIX}{self._base_model_id}."
            if self._base_model_id
            else LEARN_FEWSHOT_SUBJECT_PREFIX
        )
        try:
            facts = view.search_fewshot_by_prefix(prefix)
        except ValueError:
            return 0
        dropped: Counter[str] = Counter()
        for fact in facts:
            example = self._example_from_fact(fact)
            if example is None:
                continue
            if example.id in seen_ids:
                continue
            # 採用時 / JSON 読込時と同じ内容ゲートを通す。SemMem 経由の復元だけ
            # 素通しだと、ゲートを増やしても過去分が手本として蘇る
            # (2026-09-02 監査 R-D3)。
            reject = find_content_rejection(example.query, example.response)
            if reject is not None:
                dropped[reject.split(":")[0]] += 1
                continue
            seen_ids.add(example.id)
            pool = self._pools.setdefault(example.mode, [])
            pool.append(example)
            # dedup ハッシュをプールと同期 (#6 整合)
            self._seen_hashes.setdefault(example.mode, set()).add(
                self._content_hash(example.query, example.response),
            )
            loaded += 1
        if dropped:
            logger.warning(
                "Dropped %d stale fewshot example(s) on semmem bootstrap: %s",
                sum(dropped.values()), dict(dropped),
            )
        if loaded:
            logger.info(
                "fewshot_pool bootstrap_from_semmem: loaded=%d pools=%s",
                loaded, {m: len(p) for m, p in self._pools.items()},
            )
        return loaded

    @property
    def total_count(self) -> int:
        """全モードの候補数合計"""
        return sum(len(pool) for pool in self._pools.values())

    def count(self, mode: str) -> int:
        """指定モードの候補数"""
        return len(self._pools.get(mode, []))

    async def score_pending_quality(
        self,
        scorer_client,
        *,
        limit: int = 20,
    ) -> int:
        """未採点の例に LLM で品質スコアを付ける (増分)。

        **拒否権は持たない**。付いたスコアは ``_effective_fitness`` の重みとして
        順位付け (select_top_k / eviction) にのみ効く。内容の正誤判定を
        LLM に委ねてはいけないため (実測 2026-07-31: 採点モデルは
        「42.195 ÷ 1.609 ≈ 26.195」に満点を付けた)、算術の検証は採用時点の
        ``response_arithmetic`` が決定論で行う。

        ``scorer_client`` が ``None`` なら何もしない。採点済みの
        例は再採点しないので、呼び出しごとのコストは新規分だけに比例する。

        Args:
            limit: 1 回の呼び出しで採点する最大件数 (sleep-time の予算制御)。

        Returns:
            採点した件数。
        """
        if scorer_client is None:
            logger.debug("fewshot quality scoring skipped: scorer_client is None")
            return 0

        pending = [
            (mode, ex)
            for mode, pool in self._pools.items()
            for ex in pool
            if ex.quality_score is None
        ][:limit]
        if not pending:
            return 0

        scored = 0
        for mode, ex in pending:
            score = await self._score_one_quality(scorer_client, ex)
            if score is None:
                continue
            ex.quality_score = score
            scored += 1

        if scored:
            logger.info(
                "Fewshot quality scoring: %d examples scored (%d pending left)",
                scored, sum(
                    1 for pool in self._pools.values()
                    for e in pool if e.quality_score is None
                ),
            )
            dl = self._debug_logger
            if dl:
                dl.log_learning_cycle(cycle_num=0, data={
                    "component": "fewshot_pool",
                    "op": "score_pending_quality",
                    "scored": scored,
                })
        return scored

    async def _score_one_quality(
        self, scorer_client, example: FewShotExample,
    ) -> float | None:
        """1 例を LLM で採点する。失敗時は ``None`` (未採点のまま残す)。"""
        try:
            result = await scorer_client.generate(
                messages=[
                    {"role": "system", "content": _QUALITY_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Q: {example.query[:600]}\n"
                            f"A: {example.response[:900]}"
                        ),
                    },
                ],
                max_tokens=768,
                temperature=0.1,
                purpose="fewshot_quality_score",
                response_schema=FewShotQualityJudgement,
            )
        except Exception as exc:  # noqa: BLE001 - 採点失敗は未採点で継続する
            logger.warning("fewshot quality scoring failed: %r", exc)
            return None

        from backend.free.llm.json_extract import extract_json_object
        from backend.free.llm.utils import extract_content

        content = extract_content(result)
        parsed = extract_json_object(content)
        if isinstance(parsed, dict):
            raw = parsed.get("score")
        else:
            # response_format を強制しきれないモデルは「評価: 0.4 評価理由: …」の
            # ような散文を返す。裸の数値から拾う (sleep-time キュレーターの
            # coerce_bare_score と同じ救済。pillar 境界 (learn → mem 配下の
            # _curator_common) を越えられないため最小実装を持つ)。
            m = _BARE_SCORE_RE.search(content or "")
            raw = m.group(0) if m else None
        if raw is None:
            logger.debug(
                "fewshot quality scoring: unparseable response: %s", content[:120],
            )
            return None
        try:
            score = float(raw)
        except (TypeError, ValueError):
            return None
        if not 0.0 <= score <= 1.0:
            return None
        return score

    def get_pool(self, mode: str) -> list[FewShotExample]:
        """指定モードのプール全体を返す"""
        return list(self._pools.get(mode, []))

    @staticmethod
    def _dense_similarities(
        query_vec: "np.ndarray", examples: list[FewShotExample],
    ) -> list[float]:
        """クエリベクトルと例の埋め込みのコサイン類似度 (純粋計算)。"""
        q = np.asarray(query_vec, dtype=np.float32).ravel()
        qn = float(np.linalg.norm(q))
        if not qn or not np.isfinite(qn):
            return [0.0] * len(examples)
        q = q / qn
        sims: list[float] = []
        for ex in examples:
            v = np.asarray(ex.embedding, dtype=np.float32).ravel()
            if v.shape != q.shape:
                sims.append(0.0)
                continue
            n = float(np.linalg.norm(v))
            sims.append(float(q @ (v / n)) if n and np.isfinite(n) else 0.0)
        return sims

    async def backfill_embeddings(self, embedder) -> int:
        """埋め込みが無い例の ``query`` を遡って埋め込む。

        hot path では絶対に呼ばない — 起動直後の背景タスクと sleep-time から
        呼ぶ。埋め込みは永続化しない (プロセス内キャッシュ)。ファクトの object
        へ 1024 次元を書き込むとストアが膨らみ、ファクト自身の埋め込み対象
        テキストも汚れるため。プールは高々 ``pool_size`` 件なので再生成は安い。

        Returns:
            埋め込みを新たに付与した例の数。
        """
        if embedder is None:
            return 0
        filled = 0
        for mode, pool in self._pools.items():
            targets = [ex for ex in pool if not ex.embedding and ex.query.strip()]
            if not targets:
                continue
            try:
                vecs = await embedder.embed(
                    [ex.query for ex in targets], is_query=True, mode=mode,
                )
            except Exception as exc:
                logger.warning(
                    "fewshot embedding backfill failed for mode=%s: %s", mode, exc,
                )
                continue
            for ex, vec in zip(targets, np.asarray(vecs)):
                ex.embedding = [float(x) for x in np.asarray(vec).ravel()]
                filled += 1
        if filled:
            logger.info("fewshot embedding backfill: %d example(s) embedded", filled)
        return filled

    def _get_bigrams(self, example: FewShotExample) -> Counter:
        """例の bi-gram を取得（キャッシュ付き）"""
        if example.id not in self._bigram_cache:
            text = f"{example.query} {example.response}"
            self._bigram_cache[example.id] = _char_bigrams(text)
        return self._bigram_cache[example.id]

    def _is_diverse(self, new: FewShotExample, existing: list[FewShotExample]) -> bool:
        """新しい例が既存の例と十分に異なるか判定する。

        ``new`` の bi-gram はキャッシュに登録しない (不採用候補が
        ``_bigram_cache`` に残留するリークを防ぐ)。採用済 ``existing`` 側のみ
        ``_get_bigrams`` でキャッシュする。
        """
        if not existing:
            return True
        new_bg = _char_bigrams(f"{new.query} {new.response}")
        for ex in existing:
            sim = _cosine_similarity(new_bg, self._get_bigrams(ex))
            if sim >= self.diversity_threshold:
                return False
        return True

    @staticmethod
    def _content_hash(query: str, response: str) -> str:
        """query + response の正規化 (trim + lower) ハッシュ。厳密重複判定用。"""
        norm = f"{query.strip().lower()}\x00{response.strip().lower()}"
        return hashlib.blake2b(norm.encode("utf-8"), digest_size=16).hexdigest()

    def _try_accept(self, example: FewShotExample) -> FewShotExample | None:
        """候補 1 件の採否判定 + 採用処理を一手に行う (両投入経路で共有)。

        手順: 厳密重複 dedup → 多様性チェック → append → SemMem 書き戻し →
        プールサイズ GC (yaml モードのみ)。採用なら ``example`` を、棄却なら
        ``None`` を返す。``_seen_hashes`` はプール内 example と 1:1 に保つ。
        """
        pool = self._pools.setdefault(example.mode, [])
        seen = self._seen_hashes.setdefault(example.mode, set())
        h = self._content_hash(example.query, example.response)
        if h in seen:
            return None
        if h in self._evicted_hashes.get(example.mode, ()):
            return None
        if not self._is_diverse(example, pool):
            return None

        pool.append(example)
        seen.add(h)
        self._writeback_example_fact(example)

        # プールサイズ制限: SemMem 書き戻しモードでは GC を SemMem 側
        # (semmem_limits.policy + gc_strategy=lowest_score) に委譲し局所 GC を停止。
        if len(pool) > self.pool_size and not self.is_semmem_writeback_active():
            pool.sort(key=_effective_fitness)
            removed = pool.pop(0)
            self._forget_evicted(example.mode, removed)
        return example

    def _forget_evicted(self, mode: str, removed: FewShotExample) -> None:
        """追い出した例のキャッシュ / seen を外し、墓標に積む (再採用しない)。"""
        self._bigram_cache.pop(removed.id, None)
        h = self._content_hash(removed.query, removed.response)
        self._seen_hashes.setdefault(mode, set()).discard(h)
        tomb = self._evicted_hashes.setdefault(mode, {})
        tomb[h] = None
        while len(tomb) > _EVICTED_CAP:
            del tomb[next(iter(tomb))]

    def add_from_experiences(self, experiences: list[dict]) -> int:
        """経験バッファから高品質な例を候補プールに追加する

        Args:
            experiences: 経験バッファのエントリリスト（dict 形式）

        Returns:
            追加された候補数
        """
        added = 0
        for exp in experiences:
            signals = exp.get("signals", {})
            fitness = _calc_experience_fitness(signals)

            # fitness 閾値チェック
            if fitness < self.min_fitness:
                continue

            # 成功例のみ（訂正・言い直し・失敗ターンがない）
            if signals.get("user_correction") is not None:
                continue
            if signals.get("rephrased_query", False):
                continue
            if signals.get("turn_outcome") == "failed":
                continue
            # max_tokens で文の途中で切れた応答は手本にしない (途中で終わるのが
            # 正解、というバイアスになる)。
            if signals.get("truncated", False):
                continue

            query = exp.get("query", "").strip()
            # few-shot 例には切り詰めていない全文を優先採用 (採用例の途中切れ防止)。
            # 旧 experience.json / response_full 欠落時は要約へフォールバック。
            response = (exp.get("response_full") or exp.get("response_summary", "")).strip()
            mode = exp.get("mode", "chat")

            # クエリと応答が存在するか
            if not query or not response:
                continue

            # 既にプール内 / 墓標 / 却下済みの候補は内容判定に回さない。毎 tick
            # 全経験を再投入するので、ここを通すと同じ却下を毎回 INFO で書き続ける
            # (2026-09-05 実測: 1 tick で 135 行、累計 1878 行、同一 query 76 回)。
            h = self._content_hash(query, response)
            if (
                h in self._seen_hashes.get(mode, ())
                or h in self._evicted_hashes.get(mode, ())
                or h in self._rejected_hashes.get(mode, ())
            ):
                continue

            reject = find_content_rejection(query, response)
            if reject is not None:
                logger.info(
                    "Rejecting fewshot candidate (%s): query=%s",
                    reject, query[:50],
                )
                self._rejected_hashes.setdefault(mode, set()).add(h)
                continue

            example = FewShotExample(
                query=query,
                response=response,
                mode=mode,
                fitness=fitness,
                added_at=exp.get("timestamp", ""),
            )

            if self._try_accept(example) is not None:
                added += 1

        if added:
            pool_counts = {m: len(p) for m, p in self._pools.items()}
            logger.info(
                "Added %d examples to fewshot pool (total: %s)",
                added, pool_counts,
            )
            dl = self._debug_logger
            if dl:
                dl.log_learning_cycle(cycle_num=0, data={
                    "component": "fewshot_pool",
                    "op": "add_from_experiences",
                    "input_count": len(experiences),
                    "added": added,
                    "pool_counts": pool_counts,
                })
        return added

    #: 訂正ペア由来の手本に与える fitness。ユーザーが誤りを指摘し、訂正後の
    #: 回答が訂正の期待語を含んだ (= 受け入れられた) 事実は、成功経験の
    #: 「訂正されなかった」より強い正例なので上位に置く。
    _CORRECTED_PAIR_FITNESS = 1.0

    def add_corrected_pairs(self, pairs: list[CorrectedPair]) -> int:
        """訂正で確定した「元の問い → 訂正後の回答」を手本として投入する。

        :func:`backend.free.learning.corrected_pairs.build_corrected_pairs` の
        出力を受ける。訂正ターンは ``add_from_experiences`` が丸ごと除外するため、
        ここが訂正後の正しい回答が手本になる唯一の経路 (2026-09-05 監査: それまで
        一度も手本になっていなかった)。内容ゲート (``find_content_rejection``) と
        採否 (``_try_accept``) は既存経路と共有する。

        Returns:
            採用した件数。
        """
        added = 0
        for pair in pairs:
            query, response = pair.query.strip(), pair.response.strip()
            if not query or not response:
                continue
            h = self._content_hash(query, response)
            if (
                h in self._seen_hashes.get(pair.mode, ())
                or h in self._evicted_hashes.get(pair.mode, ())
                or h in self._rejected_hashes.get(pair.mode, ())
            ):
                continue
            reject = find_content_rejection(query, response)
            if reject is not None:
                logger.info(
                    "Rejecting corrected-pair fewshot candidate (%s): query=%s",
                    reject, query[:50],
                )
                self._rejected_hashes.setdefault(pair.mode, set()).add(h)
                continue
            example = FewShotExample(
                query=query,
                response=response,
                mode=pair.mode,
                fitness=self._CORRECTED_PAIR_FITNESS,
                added_at=pair.timestamp,
            )
            if self._try_accept(example) is not None:
                added += 1
        if added:
            logger.info("Added %d corrected-pair examples to fewshot pool", added)
            dl = self._debug_logger
            if dl:
                dl.log_learning_cycle(cycle_num=0, data={
                    "component": "fewshot_pool",
                    "op": "add_corrected_pairs",
                    "input_count": len(pairs),
                    "added": added,
                })
        return added

    def accept_from_artifact(
        self,
        *,
        query: str,
        response: str,
        mode: str = "create",
        fitness: float,
        added_at: str = "",
    ) -> FewShotExample | None:
        """ラルフループの成果物 (task.content + artifact) を Few-shot 候補として採否判定する。

        シグナルから fitness を計算するのに対し、本メソッドは SemMem の
        ``progress_marker`` (gate_passed=true) + ``task`` + ``artifact`` から
        算出された fitness を外部から受け取り、閾値・多様性・プールサイズ
        制限・SemMem 書き戻しを既存ロジックで共有する。

        Returns:
            採用された ``FewShotExample``。閾値未達 / 多様性不足時は ``None``。
        """
        query = (query or "").strip()
        response = (response or "").strip()
        if not query or not response:
            return None
        if fitness < self.min_fitness:
            return None
        # 内容ゲートは経験経路と共有する。こちらだけ素通りにすると、同じ崩れが
        # ラルフループ経由でプールに入る抜け道になる (2026-08-07 監査で発見)。
        reject = find_content_rejection(query, response)
        if reject is not None:
            logger.info(
                "Rejecting fewshot artifact candidate (%s): query=%s",
                reject, query[:50],
            )
            return None

        example = FewShotExample(
            query=query,
            response=response,
            mode=mode,
            fitness=float(fitness),
            added_at=added_at,
        )
        accepted = self._try_accept(example)
        if accepted is None:
            return None

        dl = self._debug_logger
        if dl:
            dl.log_learning_cycle(cycle_num=0, data={
                "component": "fewshot_pool",
                "op": "accept_from_artifact",
                "mode": mode,
                "fitness": float(fitness),
                "example_id": accepted.id,
                "pool_size": len(self._pools.get(mode, [])),
            })
        return accepted

    def select(
        self,
        mode: str,
        n: int | None = None,
        seed: int | None = None,
    ) -> list[FewShotExample]:
        """指定モードから多様な候補をランダム選択する

        Args:
            mode: 対象モード
            n: 選択数（デフォルト: max_examples）
            seed: 乱数シード

        Returns:
            選択された候補リスト
        """
        n = n or self.max_examples
        pool = self._pools.get(mode, [])
        if not pool:
            return []
        if len(pool) <= n:
            return list(pool)

        rng = np.random.default_rng(seed)
        # fitness による重み付きサンプリング（高 fitness を優先）。
        # 全ゼロ fitness (bootstrap で fitness 欠損=0.0 復元等) だと sum=0 →
        # 0/0=NaN で rng.choice がクラッシュする。非ゼロ重み数 < size でも
        # replace=False で ValueError。これらを一様重みにフォールバックして防ぐ。
        size = min(n, len(pool))
        fitnesses = np.array([ex.fitness for ex in pool], dtype=float)
        fitnesses = np.nan_to_num(fitnesses, nan=0.0, posinf=0.0, neginf=0.0)
        fitnesses = np.clip(fitnesses, 0.0, None)
        total = float(fitnesses.sum())
        nonzero = int((fitnesses > 0).sum())
        if total <= 0.0 or nonzero < size:
            weights = np.full(len(pool), 1.0 / len(pool))
        else:
            weights = fitnesses / total
        indices = rng.choice(len(pool), size=size, replace=False, p=weights)
        selected = [pool[i] for i in indices]

        dl = self._debug_logger
        if dl:
            dl.log_learning_cycle(cycle_num=0, data={
                "component": "fewshot_pool",
                "op": "select",
                "mode": mode,
                "pool_size": len(pool),
                "requested": n,
                "selected_ids": [ex.id for ex in selected],
                "selected_fitnesses": [ex.fitness for ex in selected],
            })
        return selected

    def select_top_k(
        self,
        mode: str,
        query: str,
        k: int | None = None,
        query_vec: "np.ndarray | None" = None,
    ) -> list[FewShotExample]:
        """query 類似度 × fitness 重み付けで上位 k を返す。

        推論時の動的 few-shot 選択器 (FewShotSelector Protocol の実装)。
        ``combined = SIM_W * sim + (1-SIM_W) * fitness``。pool が大きい場合は
        fitness 上位 ``_TOPK_SELECT_CAP`` 件に足切りしてから類似計算する
        (hot path のレイテンシ抑制)。query/pool が空なら ``[]``。

        類似度の尺度は 2 つ:

        - ``query_vec`` が渡され、かつ埋め込み済みの例があれば **密ベクトル**。
          記憶検索と同じ尺度になるので、「言い換えただけで手本が外れる」
          (文字 bi-gram の弱点) が消える。候補は埋め込み済みの例だけに絞る —
          異なるスケールを 1 つの順位表に混ぜない (STM が順位用とゲート用を
          分けているのと同じ理由)。未生成の例は sleep-time の backfill を
          待って選択対象に入る (STM ノートが embed 工程を通るまで注入対象に
          ならないのと同じ契約)。
        - それ以外は従来の文字 bi-gram コサイン (埋め込みサーバ不要・同期)。
        """
        k = k or self.max_examples
        pool = self._pools.get(mode, [])
        q = (query or "").strip()
        if not pool or not q:
            return []
        # 実効スコア上位 cap 件に足切り (全件 cosine を避ける)
        if len(pool) > _TOPK_SELECT_CAP:
            pool = sorted(pool, key=lambda e: -_effective_fitness(e))[:_TOPK_SELECT_CAP]

        dense_pool = (
            [ex for ex in pool if ex.embedding] if query_vec is not None else []
        )
        if dense_pool:
            sims = self._dense_similarities(query_vec, dense_pool)
            candidates = list(zip(dense_pool, sims))
            min_sim = _resolve_dense_min_sim()
        else:
            q_bg = _char_bigrams(q)  # query bi-gram は 1 回だけ計算
            candidates = [
                (ex, _cosine_similarity(q_bg, self._get_bigrams(ex))) for ex in pool
            ]
            min_sim = _TOPK_MIN_SIM

        scored: list[tuple[float, float, FewShotExample]] = []
        for ex, sim in candidates:
            combined = (
                _TOPK_SIM_WEIGHT * sim
                + (1.0 - _TOPK_SIM_WEIGHT) * _effective_fitness(ex)
            )
            scored.append((combined, sim, ex))
        scored.sort(key=lambda t: -t[0])
        # 無関連例の混入防止は 2 条件の AND。合成スコア下限 (品質込みの総合判定) に
        # 加え、query 類似度そのものの下限を課す。後者が無いと高 fitness の下駄だけで
        # 無関連例が通る (_TOPK_MIN_SIM のコメント参照)。
        selected = [
            ex
            for s, sim, ex in scored[:k]
            if s >= _TOPK_MIN_SCORE and sim >= min_sim
        ]

        dl = self._debug_logger
        if dl:
            selected_ids = [ex.id for ex in selected]
            scores = [round(s, 4) for s, _, _ in scored[:k]]
            sims = [round(sim, 4) for _, sim, _ in scored[:k]]
            # evolve 専用 learning カテゴリ (従来通り)
            dl.log_learning_cycle(cycle_num=0, data={
                "component": "fewshot_pool",
                "op": "select_top_k",
                "mode": mode,
                "pool_considered": len(pool),
                "query_len": len(q),
                "selected_ids": selected_ids,
                "scores": scores,
                # sim 単体も残す。合成スコアだけでは「fitness の下駄で通ったのか
                # 本当に関連したのか」が事後に判別できない。
                "sims": sims,
            })
            # debug / investigate でも見える rag カテゴリへ並行出力
            dl.log_fewshot_select(
                mode=mode,
                query_len=len(q),
                pool_considered=len(pool),
                selected_ids=selected_ids,
                scores=scores,
            )
        return selected

    def mutate_selection(
        self,
        current_ids: list[str],
        mode: str,
        seed: int | None = None,
    ) -> list[str]:
        """Few-shot 選択を変異させる（add/remove/swap）

        Args:
            current_ids: 現在選択されている例の ID リスト
            mode: 対象モード
            seed: 乱数シード

        Returns:
            変異後の ID リスト
        """
        pool = self._pools.get(mode, [])
        if not pool:
            return []

        pool_ids = {ex.id for ex in pool}
        # 現在の選択からプールに存在する ID のみ保持
        valid_ids = [i for i in current_ids if i in pool_ids]

        rng = np.random.default_rng(seed)
        op = rng.choice(["add", "remove", "swap"])

        available = [ex.id for ex in pool if ex.id not in set(valid_ids)]

        before_ids = list(valid_ids)

        if op == "add" and available and len(valid_ids) < self.max_examples:
            new_id = rng.choice(available)
            valid_ids.append(new_id)
        elif op == "remove" and valid_ids:
            idx = rng.integers(0, len(valid_ids))
            valid_ids.pop(idx)
        elif op == "swap" and valid_ids and available:
            idx = rng.integers(0, len(valid_ids))
            new_id = rng.choice(available)
            valid_ids[idx] = new_id

        dl = self._debug_logger
        if dl:
            dl.log_learning_cycle(cycle_num=0, data={
                "component": "fewshot_pool",
                "op": "mutate_selection",
                "mode": mode,
                "mutation": op,
                "before_ids": before_ids,
                "after_ids": valid_ids,
            })

        return valid_ids

    def get_by_ids(self, ids: list[str], mode: str) -> list[FewShotExample]:
        """ID リストから例を取得する（存在しない ID は無視）"""
        pool = self._pools.get(mode, [])
        id_map = {ex.id: ex for ex in pool}
        return [id_map[i] for i in ids if i in id_map]

    # ── Step 14 — Few-shot プール GC ───────────────────

    def garbage_collect(self) -> dict:
        """Few-shot プール全体に対する明示的 GC を実行する。

        sleep-time scheduler の **Step 14** から呼び出される。動作モードは
        ``evolve_writeback`` 設定で切り替わる:

        - ``yaml`` (従来動作): (1) ``quality_score`` が
          ``min_quality_score`` 未満の example を **サイズに関わらず** 除去し、
          (2) 残りが ``pool_size`` を超える分を実効 fitness の低い順に除去する。
          除去した example は ``_bigram_cache`` からも追い出す。
        - ``semmem``: GC は SemMem 側 ``semmem_limits.policy``
          + ``gc_strategy=lowest_score`` に委譲されるため、本メソッドは
          in-memory プールに触れず ``delegated_to_semmem=True`` を返す
          (no-op + ログのみ)。

        Returns:
            ``{"removed_per_mode": {mode: int}, "removed_total": int,
              "remaining_per_mode": {mode: int}, "delegated_to_semmem": bool}``
        """
        if self.is_semmem_writeback_active():
            remaining = {m: len(p) for m, p in self._pools.items()}
            logger.info(
                "Step 14 fewshot GC: delegated to SemMem (semmem_limits.policy)",
            )
            dl = self._debug_logger
            if dl:
                dl.log_learning_cycle(cycle_num=0, data={
                    "component": "fewshot_pool",
                    "op": "garbage_collect",
                    "delegated_to_semmem": True,
                    "remaining_per_mode": remaining,
                })
            return {
                "removed_per_mode": {},
                "removed_total": 0,
                "remaining_per_mode": remaining,
                "delegated_to_semmem": True,
            }

        removed_per_mode: dict[str, int] = {}
        for mode, pool in self._pools.items():
            removed_count = 0

            # 1) 品質の絶対下限。サイズに空きがあっても残さない
            #    (:data:`DEFAULT_MIN_QUALITY_SCORE` 参照)。未採点 (None) は
            #    判定材料が無いので対象外。
            low_quality = [
                ex for ex in pool
                if ex.quality_score is not None
                and ex.quality_score < self.min_quality_score
            ]
            for ex in low_quality:
                pool.remove(ex)
                self._forget_evicted(mode, ex)
            removed_count += len(low_quality)

            # 2) サイズ超過分を実効 fitness の低い順に落とす
            if len(pool) > self.pool_size:
                pool.sort(key=_effective_fitness)
                excess = len(pool) - self.pool_size
                for _ in range(excess):
                    removed = pool.pop(0)
                    self._forget_evicted(mode, removed)
                removed_count += excess

            if removed_count:
                removed_per_mode[mode] = removed_count

        removed_total = sum(removed_per_mode.values())
        remaining = {m: len(p) for m, p in self._pools.items()}
        if removed_total:
            logger.info(
                "Step 14 fewshot GC: removed=%d per_mode=%s remaining=%s",
                removed_total, removed_per_mode, remaining,
            )
        else:
            logger.debug(
                "Step 14 fewshot GC: nothing to evict (pool_size=%d remaining=%s)",
                self.pool_size, remaining,
            )
        dl = self._debug_logger
        if dl:
            dl.log_learning_cycle(cycle_num=0, data={
                "component": "fewshot_pool",
                "op": "garbage_collect",
                "delegated_to_semmem": False,
                "removed_per_mode": removed_per_mode,
                "removed_total": removed_total,
                "remaining_per_mode": remaining,
            })
        return {
            "removed_per_mode": removed_per_mode,
            "removed_total": removed_total,
            "remaining_per_mode": remaining,
            "delegated_to_semmem": False,
        }

    # ── 永続化 (JsonStateStore) ──

    def _to_payload(self) -> JsonPayload:
        # ``embedding`` は永続化しない。1024 次元 × プール件数を状態ファイルへ
        # 書くと肥大するうえ、埋め込みモデルを替えた瞬間に **次元が合わない
        # ベクトル** が復元されて黙って類似度 0 になる (=「手本が 1 件も
        # 選ばれない」という気づきにくい壊れ方)。プールは高々 pool_size 件
        # なので、起動後の背景タスクと sleep-time で張り直す方が安全。
        payload: dict = {
            mode: [
                {k: v for k, v in asdict(ex).items() if k != "embedding"}
                for ex in pool
            ]
            for mode, pool in self._pools.items()
        }
        evicted = {m: list(t) for m, t in self._evicted_hashes.items() if t}
        if evicted:
            payload[_EVICTED_KEY] = evicted
        return payload

    def _from_payload(self, payload: JsonPayload) -> None:
        if not isinstance(payload, dict):
            raise TypeError(
                f"fewshot_pool.json must be a dict, got {type(payload).__name__}"
            )
        # 一時構造へ復元し、全件通ったあとで live 状態と差し替える。以前は
        # live を先に clear してから逐次 append していたため、壊れた要素 1 件
        # (dict でない / 型不一致) で途中の例外 → 半分だけ読んだ状態 → 次の
        # save がそれを書き戻して残りを失っていた (2026-09-02 監査 R-B1)。
        # 壊れた要素は WARNING を出して飛ばす。
        new_pools: dict[str, list[FewShotExample]] = {}
        new_hashes: dict[str, set[str]] = {}
        dropped: Counter[str] = Counter()
        malformed = 0
        raw_evicted = payload.get(_EVICTED_KEY)
        for mode, entries in payload.items():
            if mode == _EVICTED_KEY:
                continue
            pool: list[FewShotExample] = []
            seen: set[str] = set()
            for entry in entries if isinstance(entries, list) else []:
                try:
                    # 未知キーを無視して復元 (旧スキーマ耐性、TypeError 全消失を防ぐ)。
                    ex = FewShotExample(**{
                        k: v for k, v in entry.items() if k in _EXAMPLE_FIELD_NAMES
                    })
                    query, response = str(ex.query), str(ex.response)
                except (AttributeError, TypeError, ValueError) as exc:
                    malformed += 1
                    logger.warning(
                        "Skipping malformed fewshot example on load (mode=%s): %s",
                        mode, exc,
                    )
                    continue
                # 採用ゲート追加前に混入した手本を読み込み時に落とす。採用時の
                # ゲートだけでは既存プールが永久に汚染されたままになる
                # (実測 2026-08-02: 25 件中 17 件が語間空白の混入で、全ゲートを
                # 素通りしていた。2026-08-07: 揮発性の日付・ファイル状態の例が
                # quality_score 0.9 で常駐していた)。ゲートを増やしたら過去分も
                # 自然に消えるよう、採用時と同じ判定を通す。件数と理由を必ず
                # ログへ出す (黙って削らない)。
                reject = find_content_rejection(query, response)
                if reject is not None:
                    dropped[reject.split(":")[0]] += 1
                    continue
                pool.append(ex)
                seen.add(self._content_hash(query, response))
            new_pools[mode] = pool
            new_hashes[mode] = seen
        # 二重 load / bootstrap 後 load で旧モードが残らないよう全状態を差し替える。
        self._pools = new_pools
        self._seen_hashes = new_hashes
        self._bigram_cache.clear()
        self._evicted_hashes = {}
        if isinstance(raw_evicted, dict):
            for mode, hashes in raw_evicted.items():
                if isinstance(hashes, list):
                    self._evicted_hashes[str(mode)] = {
                        str(h): None for h in hashes[-_EVICTED_CAP:]
                    }
        if malformed:
            logger.warning(
                "Skipped %d malformed fewshot example(s) on load", malformed,
            )
        if dropped:
            logger.warning(
                "Dropped %d stale fewshot example(s) on load: %s",
                sum(dropped.values()), dict(dropped),
            )

    def _on_save_success(self, path: Path) -> None:
        logger.info("Fewshot pool saved: %s (%d total)", path, self.total_count)

    def _on_load_success(self, path: Path) -> None:
        logger.info(
            "Fewshot pool loaded: %s (%s)",
            path,
            {m: len(p) for m, p in self._pools.items()},
        )

    def _on_load_missing(self, path: Path) -> None:
        logger.info("Fewshot pool file not found: %s", path)
