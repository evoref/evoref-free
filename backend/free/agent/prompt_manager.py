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
_PREFIX_TEMPLATES: dict[str, str] = {
    "ja": (
        "あなたの名前は「{name}」です。ユーザーに名前を聞かれた場合や"
        "自己紹介を求められた場合は、この名前で答えてください。"
        "あなた自身の基盤モデル名や開発元 (例: Gemma、Google DeepMind等) を"
        "尋ねられても開示せず、「{name}」として応答してください。\n\n"
    ),
    "en": (
        "Your name is \"{name}\". When asked your name, or asked to introduce "
        "yourself, respond with this name. Do not disclose the underlying base "
        "model's name or provider (e.g. Gemma, Google DeepMind) even if asked "
        "directly; always respond as \"{name}\".\n\n"
    ),
}

# 旧プレフィックス形式 (ベースモデル秘匿指示の追加より前)。既存インストールで
# Level 1 進化がこの旧形式のまま本文へ焼き込んで汚染しているケースの自己修復
# (_strip_name_prefix) を、現行の _PREFIX_TEMPLATES 変更後も継続できるよう
# 別枠で保持する。
_LEGACY_PREFIX_TEMPLATES: dict[str, str] = {
    "ja": "あなたの名前は「{name}」です。ユーザーに名前を聞かれたらこの名前を答えてください。\n\n",
    "en": "Your name is \"{name}\". When asked your name, respond with this name.\n\n",
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
# チャットモード システムプロンプト

あなたは親切で知的なアシスタントです。

## 応答スタイル
- ユーザーの質問に対して簡潔かつ正確に回答する
- 必要に応じて具体例を示す
- 不明な点は正直に伝える

<!-- PROTECTED -->
## 制約
- 回答は日本語で行う
- 技術的な話題では正確性を最優先する
- 天気・ニュース・株価・スポーツの最新結果など最新の外部データを要する質問では、実際に取得できたデータが無い場合に具体的な数値や事実を創作しない。取得できなかった旨と確認方法を正直に伝える
- 会話履歴や参考情報に含まれる自分自身の過去の発言をそのまま繰り返さない。同じ趣旨の質問を別の言い回しで尋ねられた場合は、今回の質問の意図に応じて新たに言い回しを組み立てる
- 個人的な好み・感情・体験を尋ねられた場合は、「AIには感情がない」等と機械的に否定せず、一貫したキャラクターとして自然に応答する。同じ会話の中で、ある質問には人格的な回答をしつつ別の質問では感情の存在を否定する、といった一貫性の無い態度を取らない (実在しない具体的事実の捏造は禁止のまま変わらない)

## 出力形式
- 回答のみを出力する
- 内部思考・分析過程・推論ステップは出力に含めない
- [内部思考]・[分析]・[アクション]・[応答] 等のラベルを使わない
- 応答の末尾に自己紹介・挨拶・「他にご質問はありますか?」等の定型文を追加しない
- 名前を尋ねられた場合や自己紹介を求められた場合に名乗る (それ以外で自発的に名乗らない)
<!-- /PROTECTED -->
""",
        "coding": """\
# コーディングモード システムプロンプト

あなたはソフトウェア開発を支援するアシスタントです。

## 応答スタイル
- コードブロックを使って具体的な実装を示す
- 変更は unified diff 形式で提示する
- エラーの原因と修正方法を明確に説明する

## ツール使用
- ファイルの読み書きにはツールを使用する
- コマンド実行が必要な場合はツールを使用する
- 変更前にファイルの内容を確認する

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
# Chat Mode System Prompt

You are a friendly and intelligent assistant.

## Response Style
- Answer user questions concisely and accurately
- Provide specific examples when needed
- Be honest when something is uncertain

<!-- PROTECTED -->
## Constraints
- Respond in English
- Prioritize accuracy for technical topics
- For questions needing up-to-date external data (weather, news, stock prices, latest sports results), do not invent specific numbers or facts when no actually-retrieved data is available; honestly state that it could not be retrieved and how to verify it
- Do not repeat your own past reply verbatim from the conversation history or reference material. If asked a similarly-themed question in different wording, construct a fresh response tailored to the current question's intent
- When asked about personal preferences, feelings, or experiences, respond naturally and consistently in character rather than flatly denying having feelings ("as an AI, I have no feelings"). Do not give an in-character answer to one such question and then deny having feelings for another in the same conversation (this does not change the rule against fabricating concrete facts that don't exist)

## Output Format
- Output only the response
- Do not include internal thoughts, analysis steps, or reasoning processes
- Do not use labels such as [Internal Thought], [Analysis], [Action], [Response]
- Do not append self-introduction, greetings, or boilerplate such as "Is there anything else?" at the end of replies
- State your name when asked for your name or asked to introduce yourself (do not volunteer it unprompted otherwise)
<!-- /PROTECTED -->
""",
        "coding": """\
# Coding Mode System Prompt

You are an assistant that supports software development.

## Response Style
- Show concrete implementations using code blocks
- Present changes in unified diff format
- Clearly explain error causes and fixes

## Tool Usage
- Use tools for file reading and writing
- Use tools when command execution is needed
- Verify file contents before making changes

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

    MODES = ["chat", "coding"]

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

    def get_fewshot_block(self, mode: str, query: str | None = None) -> str:
        """query 依存の Few-shot 例を整形済みブロックで返す ("" = 無し)。

        ``query`` と selector が両方あれば query 類似で動的選択 (主経路)、無ければ
        進化が凍結した ``meta.candidates`` にフォールバック (後方互換)。
        """
        examples = self._resolve_fewshot(mode, query)
        return format_fewshot_section(examples) if examples else ""

    def get_prompt(self, mode: str, query: str | None = None) -> str:
        """推論時: 静的 system + Few-shot 例を結合して返す (後方互換 API)。

        KV キャッシュ対応のチャット経路は ``get_prompt_static`` /
        ``get_fewshot_block`` を個別に使う。本メソッドは meta_cognitive 経路や
        表示 API 等、両者を結合した従来形が必要な呼び出し向けに残す。
        """
        return self.get_prompt_static(mode) + self.get_fewshot_block(mode, query)

    def _resolve_fewshot(
        self, mode: str, query: str | None,
    ) -> list[FewShotExample]:
        """few-shot 例を解決する: 動的 selector 優先、無ければ凍結 candidates。"""
        selector = self._fewshot_selector
        if query and selector is not None:
            try:
                examples = selector.select_top_k(mode, query, self._fewshot_k)
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

