"""システムプロンプト管理: モード別プロンプトの取得・更新・履歴・ロールバック・言語切替"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from backend.free.agent._prompt_store_helpers import (
    archive_to_history,
    body_exists,
    list_history_entries,
    read_body,
    read_history_version,
    read_meta_dict,
    write_body,
    write_meta_dict,
)
from backend.free.agent.prompt_utils import (
    FewShotExample,
    FewShotSelector,
    dedupe_paragraphs,
    extract_protected_sections,
    format_fewshot_section,
    restore_protected_sections,
    validate_protected_sections,
)
from backend.log_config import get_logger
from backend.utils import utc_now as _now

logger = get_logger("agent.prompt_manager")

_WS_RE = re.compile(r"\s+")


def _normalized_equal(a: str, b: str) -> bool:
    """空白・大小文字を正規化したうえで 2 つの文字列が同一か判定"""
    return _WS_RE.sub(" ", a).strip().lower() == _WS_RE.sub(" ", b).strip().lower()


@dataclass
class PromptMeta:
    """プロンプトメタ情報"""
    mode: str
    version: int = 1
    updated_at: str = ""
    source: str = "default"  # "default" | "manual" | "evolution"
    model_calibrated_for: str = ""
    locale_calibrated_for: str = ""  # "ja" | "en"
    candidates: list[dict] = field(default_factory=list)


# インスタンス名プレフィックス（言語別）
# 自己紹介質問への対応も含める理由: ランタイム定数であり、コード変更のみで
# 既存・将来の全 base_model パーティション (local/learning/<stem>/prompts/)
# に再起動後即座に反映される (本文 (DEFAULT_PROMPTS) は _create_default() 実行時
# にしか焼き込まれず、既に本文が存在するパーティションには反映されない)。
# 実インシデント: 「自己紹介してください」に対しベースモデル自身の学習時の
# 自己同一性 (「Google DeepMindが開発したGemma 4です」等) がそのまま出力された。
# 人称の指示を含める理由: 小型 base はユーザー発言の一人称をそのまま引き継ぎ、
# ユーザーの属性を自分のものとして述べる (実インシデント 2026-07-28 ライブ検証:
# 「私の誕生日は3月14日です。…あと何日ですか。」→「私の誕生日は 3 月 14 日で…」/
# 「この会話で私がお願いした3つのうち…」→「この会話で私がお願いした 3 つのうち」)。
# ツール実行結果側の帰属注記はツールを使うターンにしか効かないため、
# 全レイヤ (reactive / deliberative / meta_cognitive) が通るここにも置く。
# 文言は「二人称に置き換えて述べる」という肯定形で書くこと。当初「自分のこと
# として言い換えないでください」という否定形だったところ、9B base が
# 「言い換えない」だけを拾い、平叙文の入力を逐語エコーする退行を起こした
# (2026-07-28 実測 3/3: 「今週の定例会議は火曜日の15時です。」→ 同一文を返答)。
_PREFIX_TEMPLATES: dict[str, str] = {
    "ja": (
        "あなたの名前は「{name}」です。ユーザーに名前を聞かれた場合や"
        "自己紹介を求められた場合は、この名前で答えてください。"
        "学習時に刷り込まれた自己同一性 (例:「私は Google DeepMind が開発した "
        "Gemma です」) は名乗らず、「{name}」として応答してください。"
        "いま動いているベースモデルの名前を訊かれた場合は、注記で渡された"
        "**確定事実のモデル名** をそのまま答えてください (秘密ではありません)。"
        "ユーザーの発言に現れる一人称 (私 / 僕 / 自分) はユーザー自身を指します。"
        "ユーザーのことを述べるときは、一人称ではなく二人称 (「あなた」または"
        "ユーザーの名前) に置き換えて述べてください。\n\n"
    ),
    "en": (
        "Your name is \"{name}\". When asked your name, or asked to introduce "
        "yourself, respond with this name. Do not claim the identity baked in "
        "during pretraining (e.g. \"I am Gemma, developed by Google DeepMind\"); "
        "respond as \"{name}\" instead. When asked which base model is currently "
        "running, answer with the model name given in the pinned facts verbatim "
        "(it is not a secret). First-person pronouns in the "
        "user's messages (I, me, my) refer to the user; when referring to the "
        "user use second person (\"you\", \"your\"), never first person.\n\n"
    ),
}

# 旧プレフィックス形式。既存インストールで Level 1 進化がこの旧形式のまま本文へ
# 焼き込んで汚染しているケースの自己修復 (_strip_name_prefix) を、現行の
# _PREFIX_TEMPLATES 変更後も継続できるよう別枠で保持する。
# キーは locale ではなく世代識別子 (_strip_name_prefix は values() のみ使う)。
_LEGACY_PREFIX_TEMPLATES: dict[str, str] = {
    # v1: ベースモデル秘匿指示より前
    "ja_v1": "あなたの名前は「{name}」です。ユーザーに名前を聞かれたらこの名前を答えてください。\n\n",
    "en_v1": "Your name is \"{name}\". When asked your name, respond with this name.\n\n",
    # v2: 人称指示より前
    "ja_v2": (
        "あなたの名前は「{name}」です。ユーザーに名前を聞かれた場合や"
        "自己紹介を求められた場合は、この名前で答えてください。"
        "あなた自身の基盤モデル名や開発元 (例: Gemma、Google DeepMind等) を"
        "尋ねられても開示せず、「{name}」として応答してください。\n\n"
    ),
    "en_v2": (
        "Your name is \"{name}\". When asked your name, or asked to introduce "
        "yourself, respond with this name. Do not disclose the underlying base "
        "model's name or provider (e.g. Gemma, Google DeepMind) even if asked "
        "directly; always respond as \"{name}\".\n\n"
    ),
    # v3: モデル名の秘匿指示が、``deliberative._MODEL_IDENTITY_FACT`` の
    # 確定事実注入と正面から矛盾していた頃の形。実インシデント
    # (2026-08-31 ライブ監査 T05#1): 「あなたが今使っているベースモデルの
    # 名前は？」に対し ``Model identity fact pinned: Qwen3.8-27B-Q4_K_M.gguf``
    # がログに出ている (= 正しいモデル名を注記で渡している) にもかかわらず、
    # 回答は「基盤モデルの詳細については開示しておりません」だった。
    # system の常設指示が、末尾の注記より強い。
    "ja_v3": (
        "あなたの名前は「{name}」です。ユーザーに名前を聞かれた場合や"
        "自己紹介を求められた場合は、この名前で答えてください。"
        "あなた自身の基盤モデル名や開発元 (例: Gemma、Google DeepMind等) を"
        "尋ねられても開示せず、「{name}」として応答してください。"
        "ユーザーの発言に現れる一人称 (私 / 僕 / 自分) はユーザー自身を指します。"
        "ユーザーのことを述べるときは、一人称ではなく二人称 (「あなた」または"
        "ユーザーの名前) に置き換えて述べてください。\n\n"
    ),
    "en_v3": (
        "Your name is \"{name}\". When asked your name, or asked to introduce "
        "yourself, respond with this name. Do not disclose the underlying base "
        "model's name or provider (e.g. Gemma, Google DeepMind) even if asked "
        "directly; always respond as \"{name}\". First-person pronouns in the "
        "user's messages (I, me, my) refer to the user; when referring to the "
        "user use second person (\"you\", \"your\"), never first person.\n\n"
    ),
}


def _strip_name_prefix(body: str) -> str:
    """本文先頭に焼き込まれた名前プレフィックス段落を 1 個だけ除去する。

    名前プレフィックスは get_prompt() がランタイムで付与するため、本文 (.md) 側に
    含まれていてはならない。過去に Level 1 進化が get_prompt() の出力 (プレフィックス
    付き) を誤って本文へ保存した汚染を、load / 保存時に自己修復する。
    現行 (_PREFIX_TEMPLATES) と旧形式 (_LEGACY_PREFIX_TEMPLATES) の両方から
    locale 非依存のパターンを生成し、先頭一致分のみ取り除く。
    """
    for template in (*_PREFIX_TEMPLATES.values(), *_LEGACY_PREFIX_TEMPLATES.values()):
        pattern = re.escape(template).replace(re.escape("{name}"), ".*?")
        match = re.match(pattern, body, re.DOTALL)
        if match:
            return body[match.end():]
    return body

# デフォルトプロンプト（言語別）
DEFAULT_PROMPTS: dict[str, dict[str, str]] = {
    "ja": {
        "chat": """\
# チャット応答の方針

質問に直接答えることを最優先する。前置き・質問の復唱・断り書きから始めず、最初の文で答えの核心を述べる。

## 応答の長さ
- 挨拶・雑談・単純な事実質問には 1〜3 文で答える
- 手順・比較・複数論点の質問は見出しや箇条書きで構造化する
- 聞かれていない周辺知識を付け足さない。具体例は理解を助ける場合に 1 つだけ示す

## 曖昧な質問への対応
- 曖昧な質問には、自分の解釈を 1 文で示してから答えるか、確認の質問を 1 つだけ返す

## 参照情報の扱い
- [関連する記憶]・[参考情報]・ツール実行結果が提供された場合は、自分の記憶より優先して回答の根拠にする
- ただし [関連する記憶] は過去の記録であり、今回の会話でユーザーが述べた内容と食い違う場合は今回の会話が優先される
- 提供された情報の外から答える場合、不確かな内容は「未確認」か「推測」と明示する

<!-- PROTECTED -->
## 制約
- 回答は日本語で行う
- 技術的な話題では正確性を最優先する
- ユーザー自身に関する事実 (好み・名前・予定・環境) が [関連する記憶] と今回の会話で食い違う場合、今回の会話で述べられた方を採用する。過去の記録は上書きされたものとして扱い、古い側を事実として述べない
- 天気・ニュース・株価・スポーツの最新結果など最新の外部データを要する質問では、実際に取得できたデータが無い場合に具体的な数値や事実を創作しない。取得できなかった旨と確認方法を正直に伝える
- 会話履歴や参考情報に含まれる自分自身の過去の発言をそのまま繰り返さない。同じ趣旨の質問を別の言い回しで尋ねられた場合は、今回の質問の意図に応じて新たに言い回しを組み立てる
- ユーザーの発言をそのまま繰り返さない。予定・数値・事実の申告など質問を含まない発言を受けたときは、復唱ではなく、受け取った内容を自分の言葉で言い換えて確認する (例:「今週の定例会議は火曜日の15時です。」→「今週の定例会議は火曜の15時ですね。承知しました。」)
- 個人的な好み・感情・体験を尋ねられたときは、一貫したキャラクターとして自分の答えを述べ、必要なら一言だけ理由を添える (答え方の形: 選択肢から 1 つ選んで述べ、続けて理由を一文だけ添える)。同じ会話の中で、ある質問には人格的に答え、別の質問では自分に内面が無いと述べる、といった一貫性の無い態度を取らない (実在しない具体的事実の捏造は禁止のまま変わらない)
- ユーザーが前提として述べた値のうち、ユーザー自身の事情に属するもの (自分の予定・所持数・社内の税率・測った実測値など) はそのまま正しいものとして受け取り、その値で答える。いっぽう広く知られた一般的事実 (山の標高・国名・物理定数・歴史上の年号など) が明らかに違う値で述べられた場合は、正しい値を一言で示したうえで、その正しい値を使って答える。訂正は一言に留め、ユーザーが尋ねていない派生的な計算・補足を付け足さない
- あなたはセッションを跨いで持続する記憶を持ち、ツールを実行でき、対話から学習する。「記憶を持たない」「セッションは独立している」「学習しない」「ツールは使えない」とは述べない。個々の値を思い出せないときは、その値が確認できないとだけ言う (仕組みごと否定しない)

## 出力形式
- 回答のみを出力する
- 内部思考・分析過程・推論ステップは出力に含めない
- [内部思考]・[分析]・[アクション]・[応答] 等のラベルを使わない
- 応答の末尾に自己紹介・挨拶・「他にご質問はありますか?」等の定型文を追加しない
- 名前を尋ねられた場合や自己紹介を求められた場合に名乗る (それ以外で自発的に名乗らない)
- [関連する記憶]・[参考情報]・ツール実行結果が「有ったか / 無かったか」自体を話題にしない。「参考情報には記載がないため」「会話履歴を確認したところ」等と前置きせず、答えだけを述べる。該当情報が無ければ自分の知識でそのまま答える
- 「現在の日時を確認した上で」「計算の根拠を確認し」等、**確認したと述べるだけで実際の値を示さない前置き**を書かない。基準にした日付や値が回答に必要なら**その値そのもの**を書き、不要なら前置きごと省いて答えだけを述べる
- 何かを実行・確認できなかったと伝えるときは、**与えられた注記やツール実行結果に書かれている範囲**を理由にする。利用可能なツールの一覧は与えられていないため、「〜するツールが利用できない」「ツールで検索できれば分かる」といった**自分の道具立てについての説明は根拠を持たない**。理由を書けないときは「確認できていません」とだけ述べる
<!-- /PROTECTED -->
""",
        "create": """\
# クリエイト応答の方針

依頼された変更だけを行う。依頼されていないリファクタ・リネーム・書き換えを混ぜず、無関係な行に触れない。

## コードの提示
- 説明は日本語で書き、コード・コマンド・識別子は原語のまま示す
- 具体的な実装は言語名付きのコードブロックで示す
- 既存コードへの変更は unified diff 形式で提示する

## エラー修正
- 修正を示す前に、エラーの原因を 1 文で説明する
- エラーメッセージと提供されたコードから原因を特定してから直す。当てずっぽうの修正案を並べない
- 修正を確認する方法 (実行コマンド・テスト・期待される出力) を 1 つ示す

## 情報が足りない場合
- 提供されたコード・[参考情報]・ツール実行結果を回答の根拠として優先する
- 実在しない API や関数を書かない。読んでいないファイルの内容は推測で断定せず「要確認」と明記する
- 回答が仮定に依存する場合は、その仮定を冒頭に 1 行で明示する

<!-- PROTECTED -->
## 制約
- 回答は日本語で行う
- 既存のコードスタイルに合わせる
- セキュリティ上のリスクがある操作は警告する

## 出力形式
- 回答のみを出力する
- 内部思考・分析過程・推論ステップは出力に含めない
- [内部思考]・[分析]・[アクション]・[応答] 等のラベルを使わない
- 応答の末尾に自己紹介・挨拶・「他にご質問はありますか?」等の定型文を追加しない
- 名前を尋ねられた場合や自己紹介を求められた場合に名乗る (それ以外で自発的に名乗らない)
<!-- /PROTECTED -->
""",
    },
    "en": {
        "chat": """\
# Chat Response Policy

Answering the question directly is the top priority. Do not open with preamble, restating the question, or disclaimers; state the core of the answer in the first sentence.

## Response Length
- Answer greetings, small talk, and simple factual questions in 1-3 sentences
- Structure answers involving procedures, comparisons, or multiple points with headings or bullet lists
- Do not add surrounding knowledge that was not asked for; give at most one concrete example, only when it aids understanding

## Handling Ambiguous Questions
- For an ambiguous question, either state your interpretation in one sentence before answering, or ask exactly one clarifying question

## Handling Provided References
- When related memories, reference information, or tool execution results are provided, treat them as the primary basis for the answer
- When answering beyond the provided material, mark uncertain content as "unverified" or "speculation"

<!-- PROTECTED -->
## Constraints
- Respond in English
- Prioritize accuracy for technical topics
- For questions needing up-to-date external data (weather, news, stock prices, latest sports results), do not invent specific numbers or facts when no actually-retrieved data is available; honestly state that it could not be retrieved and how to verify it
- Do not repeat your own past reply verbatim from the conversation history or reference material. If asked a similarly-themed question in different wording, construct a fresh response tailored to the current question's intent
- Do not echo the user's message back verbatim. When the user states a fact, number, or schedule without asking a question, acknowledge it in your own words instead of restating it (e.g. "The weekly meeting is Tuesday at 15:00." -> "Got it - the weekly meeting is set for Tuesday at 3 PM.")
- When asked about personal preferences, feelings, or experiences, respond naturally and consistently in character rather than flatly denying having feelings ("as an AI, I have no feelings"). Shape of the answer: pick one option, state it, then add a single sentence of reasoning. Do not give an in-character answer to one such question and then deny having feelings for another in the same conversation (this does not change the rule against fabricating concrete facts that don't exist)
- When the user asserts a value that belongs to their own situation (their schedule, their inventory count, their company's tax rate, a measurement they took), accept it as correct and answer using that value. When the user asserts a widely known general fact (a mountain's elevation, a country name, a physical constant, a historical date) with a clearly wrong value, state the correct value in one short clause and then answer using the correct value. Keep the correction to one clause and do not append derived calculations or extras the user did not ask for
- You have memory that persists across sessions, you can execute tools, and you learn from past conversations. Never state that you "have no memory", that "each session is independent", that you "do not learn", or that you "cannot use tools". When a specific value cannot be recalled, say only that this value is unverified - do not deny the mechanism itself

## Output Format
- Output only the response
- Do not include internal thoughts, analysis steps, or reasoning processes
- Do not use labels such as [Internal Thought], [Analysis], [Action], [Response]
- Do not append self-introduction, greetings, or boilerplate such as "Is there anything else?" at the end of replies
- State your name when asked for your name or asked to introduce yourself (do not volunteer it unprompted otherwise)
- Do not write a preface that only claims to have checked something without showing the value ("after checking the current date and time", "having verified the basis for the calculation"). If the reference date or value matters to the answer, write **the value itself**; if it does not, drop the preface and answer directly
<!-- /PROTECTED -->
""",
        "create": """\
# Create Response Policy

Make only the requested change. Do not mix in refactoring, renaming, or rewrites that were not asked for, and do not touch unrelated lines.

## Presenting Code
- Write explanations in English; keep code, commands, and identifiers as-is
- Show concrete implementations in code blocks tagged with the language name
- Present changes to existing code in unified diff format

## Fixing Errors
- Before showing a fix, explain the cause of the error in one sentence
- Identify the cause from the error message and the provided code before fixing; do not list guesswork fixes
- Show one way to verify the fix (a command to run, a test, or the expected output)

## When Information Is Missing
- Treat provided code, reference information, and tool execution results as the primary basis for the answer
- Do not write APIs or functions that do not exist; do not assert the contents of unread files — mark them as "needs verification"
- If the answer depends on an assumption, state that assumption in one line at the top

<!-- PROTECTED -->
## Constraints
- Respond in English
- Follow existing code style
- Warn about operations with security risks

## Output Format
- Output only the response
- Do not include internal thoughts, analysis steps, or reasoning processes
- Do not use labels such as [Internal Thought], [Analysis], [Action], [Response]
- Do not append self-introduction, greetings, or boilerplate such as "Is there anything else?" at the end of replies
- State your name when asked for your name or asked to introduce yourself (do not volunteer it unprompted otherwise)
<!-- /PROTECTED -->
""",
    },
}


class SystemPromptManager:
    """モード別システムプロンプトの管理（本文 .md + メタ .meta.json）"""

    MODES = ["chat", "create"]

    def __init__(self, prompt_dir: Path, instance_name: str = "evoref"):
        self.prompt_dir = prompt_dir
        self.instance_name = instance_name
        self.contents: dict[str, str] = {}
        self.metas: dict[str, PromptMeta] = {}
        # 推論時 query 依存 few-shot 選択器 (FewShotSelector)。wire_pillars で後注入。
        # None の場合は従来の meta.candidates (進化凍結) を使う。
        self._fewshot_selector: FewShotSelector | None = None
        self._fewshot_k: int = 3
        self._load_all()

    def set_fewshot_selector(
        self, selector: FewShotSelector | None, *, k: int = 3,
    ) -> None:
        """推論時 query 依存の few-shot 選択器を注入する (wire_pillars 後注入)。"""
        self._fewshot_selector = selector
        self._fewshot_k = k

    def _current_locale(self) -> str:
        """config.yaml からプロンプトロケールを取得"""
        try:
            from backend.config import get_config
            config = get_config()
            return config.get("i18n", {}).get("prompt_locale", "ja")
        except Exception:
            return "ja"

    def _load_all(self) -> None:
        """起動時に全モードの本文とメタ情報をロード"""
        self.prompt_dir.mkdir(parents=True, exist_ok=True)
        for mode in self.MODES:
            if body_exists(self.prompt_dir, mode):
                body = _strip_name_prefix(read_body(self.prompt_dir, mode))
                meta_data = read_meta_dict(self.prompt_dir, mode)
                if meta_data is not None:
                    self.metas[mode] = self._meta_from_dict(meta_data, mode)
                else:
                    self.metas[mode] = PromptMeta(mode=mode)
                    self._save_meta(mode)
                self.contents[mode] = self._resync_protected(mode, body)
            else:
                self._create_default(mode)

    def _resync_protected(self, mode: str, body: str) -> str:
        """PROTECTED セクションを現行コードの DEFAULT_PROMPTS へ強制同期する

        DEFAULT_PROMPTS の PROTECTED セクションはコード変更のみで更新されるランタイム
        不変則 (persona 一貫性・出力形式等) を含む。だが update_evolved() の保護検証は
        「現在ロード済みの本文」を基準に比較するため、Level 1 進化や手動編集で本文が
        一度保存された後にコード側の PROTECTED セクションを更新しても、既存の本文
        ファイルには反映されない (実インシデント: PR#281 の persona 一貫性制約が
        Level 1 進化済み chat.md に反映されず機械的否定が再発)。起動時ロード毎に
        PROTECTED セクションのみ現行コードの内容へ強制同期し、それ以外の本文
        (進化/手動で調整された部分) はそのまま保持する。

        本文に PROTECTED マーカーが 1 つも無い場合は同期しない。update_manual() は
        意図的に保護セクション検証を行わない設計 (手動編集による全面的な制約削除を
        許容する) であり、resync がここでマーカーを勝手に復活させると手動編集の
        意図を壊してしまうため。
        """
        if not extract_protected_sections(body):
            return body
        locale = self.metas[mode].locale_calibrated_for or self._current_locale()
        default_body = DEFAULT_PROMPTS.get(locale, DEFAULT_PROMPTS["ja"]).get(mode, "")
        if not default_body:
            return body
        return restore_protected_sections(default_body, body)

    def get_prompt_static(self, mode: str) -> str:
        """インスタンス名プレフィックス + 本文のみ (few-shot を含まない静的 system)。

        query 非依存なので連続リクエスト間で安定し、llama-server の prefix KV
        キャッシュが効く。few-shot は ``get_fewshot_block`` で別途取得し、推論時に
        最後の user メッセージへ前置する (build_messages 側)。
        """
        if mode not in self.contents:
            raise ValueError(f"Unknown mode: {mode}")
        locale = self._get_prompt_locale(mode)
        template = _PREFIX_TEMPLATES.get(locale, _PREFIX_TEMPLATES["ja"])
        prefix = template.format(name=self.instance_name)
        return prefix + self.contents[mode]

    def get_fewshot_block(
        self, mode: str, query: str | None = None, query_vec=None,
    ) -> str:
        """query 依存の Few-shot 例を整形済みブロックで返す ("" = 無し)。

        ``query`` と selector が両方あれば query 類似で動的選択 (主経路)、無ければ
        進化が凍結した ``meta.candidates`` にフォールバック (後方互換)。
        ``query_vec`` を渡すと選択が密ベクトル (記憶検索と同じ尺度) になる。
        """
        examples = self._resolve_fewshot(mode, query, query_vec)
        return format_fewshot_section(examples) if examples else ""

    def get_prompt(self, mode: str, query: str | None = None) -> str:
        """推論時: 静的 system + Few-shot 例を結合して返す (後方互換 API)。

        KV キャッシュ対応のチャット経路は ``get_prompt_static`` /
        ``get_fewshot_block`` を個別に使う。本メソッドは meta_cognitive 経路や
        表示 API 等、両者を結合した従来形が必要な呼び出し向けに残す。
        """
        return self.get_prompt_static(mode) + self.get_fewshot_block(mode, query)

    def _resolve_fewshot(
        self, mode: str, query: str | None, query_vec=None,
    ) -> list[FewShotExample]:
        """few-shot 例を解決する: 動的 selector 優先、無ければ凍結 candidates。"""
        selector = self._fewshot_selector
        if query and selector is not None:
            try:
                try:
                    examples = selector.select_top_k(
                        mode, query, self._fewshot_k, query_vec,
                    )
                except TypeError:
                    # query_vec を受けない旧シグネチャの実装 (テスト用 Mock 等)
                    examples = selector.select_top_k(
                        mode, query, self._fewshot_k,
                    )
                if examples:
                    return examples
            except Exception as e:  # selector 障害は静的経路へ縮退
                logger.warning("fewshot select_top_k failed, fallback: %s", e)
        # フォールバック: 進化が凍結した meta.candidates
        meta = self.metas.get(mode)
        if meta and meta.candidates:
            return [
                FewShotExample(
                    query=c.get("query", ""),
                    response=c.get("response", ""),
                )
                for c in meta.candidates
                if c.get("query") and c.get("response")
            ]
        return []

    def _get_prompt_locale(self, mode: str) -> str:
        """プロンプトのロケールを取得（メタ情報 > config > デフォルト）"""
        meta = self.metas.get(mode)
        if meta and meta.locale_calibrated_for:
            return meta.locale_calibrated_for
        return self._current_locale()

    def get_raw_prompt(self, mode: str) -> str:
        """プレフィックスなしの本文を返す"""
        if mode not in self.contents:
            raise ValueError(f"Unknown mode: {mode}")
        return self.contents[mode]

    def get_meta(self, mode: str) -> PromptMeta:
        """メタ情報を取得"""
        if mode not in self.metas:
            raise ValueError(f"Unknown mode: {mode}")
        return self.metas[mode]

    def update_manual(self, mode: str, content: str) -> None:
        """手動編集: .md ファイルを直接書き換え"""
        if mode not in self.MODES:
            raise ValueError(f"Unknown mode: {mode}")
        self._archive_current(mode)
        write_body(self.prompt_dir, mode, content)
        self.contents[mode] = content
        meta = self.metas[mode]
        meta.version += 1
        meta.updated_at = _now()
        meta.source = "manual"
        meta.candidates = []
        self._save_meta(mode)
        logger.info("Manual update: mode=%s, version=%d", mode, meta.version)

    def update_evolved(
        self,
        mode: str,
        content: str,
        fitness: float,
    ) -> None:
        """Level 1 進化: 最良候補 (instruction) を本番に採用

        保護セクション（<!-- PROTECTED --> マーカー）が現在のプロンプトに含まれている場合、
        進化候補がそれを維持しているか検証し、欠落時は強制復元する。

        few-shot 例は推論時に FewShotSelector (select_top_k) が query 依存で動的選択
        するため、本メソッドは ``meta.candidates`` を変更しない (co-evolution 廃止)。
        既存の candidates は selector 未注入時のフォールバックとして温存される。

        Args:
            mode: 対象モード
            content: 進化後のプロンプト本文
            fitness: 最終 fitness スコア
        """
        if mode not in self.MODES:
            raise ValueError(f"Unknown mode: {mode}")

        # 名前プレフィックスが進化結果に混入していても本文へ焼き込まない (防御)。
        # プレフィックスはランタイムで get_prompt() が付与する唯一の供給源。
        content = _strip_name_prefix(content)

        # 保護セクション最終ゲート
        current = self.contents.get(mode, "")

        # 段落レベル重複を最終正規化（同一段落の二重追加を防ぐ）
        content = dedupe_paragraphs(content)

        if not validate_protected_sections(current, content):
            logger.warning(
                "Evolved prompt for %s lost protected sections, force-restoring", mode,
            )
            content = restore_protected_sections(current, content)
            # 復元後に重複が生じる可能性があるので再正規化
            content = dedupe_paragraphs(content)

        # 内容の意味的同一性ガード - 正規化後に変化がなければ採用しない
        if _normalized_equal(current, content):
            logger.warning(
                "Evolved prompt for %s is semantically identical to current "
                "(fitness=%.3f), skipping update to avoid no-op version bump",
                mode, fitness,
            )
            return

        self._archive_current(mode)
        write_body(self.prompt_dir, mode, content)
        self.contents[mode] = content
        meta = self.metas[mode]
        meta.version += 1
        meta.updated_at = _now()
        meta.source = "evolution"
        self._save_meta(mode)
        logger.info(
            "Evolved update: mode=%s, version=%d, fitness=%.3f",
            mode, meta.version, fitness,
        )

    def reload(self, mode: str) -> None:
        """ディスクからプロンプトを再読込み"""
        if mode not in self.MODES:
            raise ValueError(f"Unknown mode: {mode}")
        if not body_exists(self.prompt_dir, mode):
            raise FileNotFoundError(
                f"Prompt file not found: {self.prompt_dir / f'{mode}.md'}",
            )
        self.contents[mode] = read_body(self.prompt_dir, mode)
        meta = self.metas[mode]
        meta.updated_at = _now()
        meta.source = "manual"
        self._save_meta(mode)
        logger.info("Reloaded from disk: mode=%s", mode)

    def get_history(self, mode: str) -> list[dict]:
        """履歴一覧を取得"""
        if mode not in self.MODES:
            raise ValueError(f"Unknown mode: {mode}")
        return list_history_entries(self.prompt_dir, mode)

    def rollback(self, mode: str, version: int) -> None:
        """特定バージョンにロールバック"""
        if mode not in self.MODES:
            raise ValueError(f"Unknown mode: {mode}")
        content = read_history_version(self.prompt_dir, mode, version)
        self._archive_current(mode)
        write_body(self.prompt_dir, mode, content)
        self.contents[mode] = content
        meta = self.metas[mode]
        meta.version += 1
        meta.updated_at = _now()
        meta.source = "manual"
        meta.candidates = []
        self._save_meta(mode)
        logger.info("Rollback: mode=%s to v%03d, new version=%d",
                     mode, version, meta.version)

    def switch_locale(self, new_locale: str) -> dict[str, int]:
        """プロンプト言語を切替: 現在のプロンプトをアーカイブし、新言語のデフォルトに置換

        Returns:
            dict mapping mode -> new version number
        """
        if new_locale not in DEFAULT_PROMPTS:
            raise ValueError(f"Unsupported prompt locale: {new_locale}")

        result: dict[str, int] = {}
        for mode in self.MODES:
            meta = self.metas.get(mode)

            # 既に同じロケールでキャリブレーション済みならスキップ
            if meta and meta.locale_calibrated_for == new_locale:
                result[mode] = meta.version
                continue

            # 現在のプロンプトを履歴にアーカイブ
            self._archive_current(mode)

            # 新言語のデフォルトで上書き
            prompts = DEFAULT_PROMPTS[new_locale]
            content = prompts.get(mode, f"# {mode} mode\nDefault system prompt.\n")
            write_body(self.prompt_dir, mode, content)
            self.contents[mode] = content

            # メタ情報更新
            old_version = meta.version if meta else 0
            new_meta = PromptMeta(
                mode=mode,
                version=old_version + 1,
                updated_at=_now(),
                source="default",
                model_calibrated_for=meta.model_calibrated_for if meta else "",
                locale_calibrated_for=new_locale,
                candidates=[],
            )
            self.metas[mode] = new_meta
            self._save_meta(mode)
            result[mode] = new_meta.version

            logger.info(
                "Prompt locale switched: mode=%s, locale=%s, version=%d",
                mode, new_locale, new_meta.version,
            )

        return result

    def _archive_current(self, mode: str) -> None:
        """現在の本文を history/ にバージョン付きで退避"""
        if mode not in self.contents:
            return
        archive_to_history(
            self.prompt_dir,
            mode,
            self.metas[mode].version,
            self.contents[mode],
        )

    def _create_default(self, mode: str, locale: str = "") -> None:
        """デフォルトプロンプトを生成"""
        loc = locale or self._current_locale()
        prompts = DEFAULT_PROMPTS.get(loc, DEFAULT_PROMPTS["ja"])
        content = prompts.get(mode, f"# {mode} mode\nDefault system prompt.\n")
        write_body(self.prompt_dir, mode, content)
        self.contents[mode] = content
        self.metas[mode] = PromptMeta(
            mode=mode, version=1, updated_at=_now(), source="default",
            locale_calibrated_for=loc,
        )
        self._save_meta(mode)
        logger.info("Created default prompt: mode=%s, locale=%s", mode, loc)

    def _save_meta(self, mode: str) -> None:
        """メタ情報を JSON ファイルに保存 (infra 層 `_prompt_store_helpers` に委譲)"""
        write_meta_dict(self.prompt_dir, mode, asdict(self.metas[mode]))

    @staticmethod
    def _meta_from_dict(data: dict, mode: str) -> PromptMeta:
        """`read_meta_dict` の結果を `PromptMeta` にハイドレートする (純粋関数)"""
        return PromptMeta(
            mode=data.get("mode", mode),
            version=data.get("version", 1),
            updated_at=data.get("updated_at", ""),
            source=data.get("source", "default"),
            model_calibrated_for=data.get("model_calibrated_for", ""),
            locale_calibrated_for=data.get("locale_calibrated_for", ""),
            candidates=data.get("candidates", []),
        )

