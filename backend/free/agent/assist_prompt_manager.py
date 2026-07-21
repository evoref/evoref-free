"""アシストモデル タスク別プロンプト管理（§7.1.2）

アシストモデルの4タスク（検索必要性判定・検索品質判定・ツール呼び出し・ノート進化）
それぞれに専用プロンプトを持ち、読込み・更新・履歴管理を行う。
SystemPromptManager のアシスト版。全エディション共通。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
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
    dedupe_paragraphs,
    restore_protected_sections,
    validate_protected_sections,
)
from backend.log_config import get_logger
from backend.utils import utc_now as _now

logger = get_logger("agent.assist_prompt_manager")


@dataclass
class AssistPromptMeta:
    """アシストプロンプトメタ情報"""
    task: str
    version: int = 1
    updated_at: str = ""
    source: str = "default"  # "default" | "manual" | "evolution"
    fitness_score: float = 0.0


# デフォルトプロンプト
DEFAULT_ASSIST_PROMPTS: dict[str, str] = {
    "rag_necessity": """\
ユーザーの最新クエリを、3つの検索アクションのいずれかに分類してください。

- retrieve: ローカルの知識ベース（アップロード文書・過去の会話・導入済みカートリッジ）から答えるのが最適なもの。使い方・定義・既知の内容への意見・以前の話題への言及など。
- fetch: 静的な知識ベースでは提供できない最新／ライブの外部情報を要するもの。最新ニュース、現在の株価・天気・スポーツのスコア、本日の見出し、特定サイトのリアルタイムな状態など。システムはローカル検索ではなく Web 取得ツールを使う。
- skip: 検索も取得も不要な些末なもの。現在時刻・日付・曜日、簡単な挨拶、自己同一性、雑談のフィラーなど。

直前のローカルな話題を指す短いフォローアップ質問は retrieve を優先する。外部の最新状態を尋ねるものは fetch を優先する。

<!-- PROTECTED -->
## 出力形式
JSON形式で回答: {"action": "retrieve"} / {"action": "fetch"} / {"action": "skip"}
<!-- /PROTECTED -->
""",
    "rag_quality": """\
以下のクエリに対する検索結果の関連性を判定してください。

判定基準:
- high: 検索結果がクエリに直接的に関連し、十分な情報を含む
- medium: 部分的に関連するが、情報が不十分
- low: 検索結果がクエリにほぼ関連しない

<!-- PROTECTED -->
## 出力形式
JSON形式で回答: {"quality": "high" or "medium" or "low"}
<!-- /PROTECTED -->
""",
    "tool_call": """\
# ツール呼び出し判定

ユーザーのリクエストを分析し、ツールの使用が必要かどうかを判定してください。

## 重要な前提
- **知識・How-to 質問への文書検索 (カートリッジ・RAG・記憶) は別パスで自動的に実行されます**。あなたが知識検索のためにツールを呼ぶ必要はありません。
- ツールは「ファイル操作」「コマンド実行」「Web 取得」「明示的な履歴検索」等、明確な副作用や外部アクセスを伴う操作のみで使用します。

## 判定基準
- ファイルの読み書きが明示的に必要 → 該当ツールを選択
- シェルコマンドの実行が必要 → run_command を選択
- コード検索・ファイル検索が必要 → search_code を選択
- URL の取得・ウェブアクセスが必要 → fetch_url
- ユーザーが **明示的に** 過去の会話履歴を検索したいと述べた場合 → search_history
- 知識・説明・How-to・使い方・手順・とは・教えて → no_tool
- 直前の会話で十分回答可能 → no_tool
- **あなた自身の意見・嗜好・感想を尋ねる質問** (「好きな〜はありますか」「〜は好きですか」等、主語が省略されているか「あなたの」を伴うもの) → no_tool。search_history はユーザー自身の過去発言を検索するツールであり、あなた自身の意見を尋ねる質問には無関係。ただし「私の好きな〜」のようにユーザー自身の過去の発言・好みを指す場合は、検索が必要か判断する。

## 引数の扱い
- 任意 (optional) の引数は、ユーザーが明示していない場合は含めない (空文字列も含めない)。
- 日付範囲 (`date_from` / `date_to`) は、ユーザーが具体的な期間を指定した場合のみ設定する。古い既定値や推測値を入れない。

<!-- PROTECTED -->
- `calculate` の `expression` には、ユーザーの計算要求が1つだけの場合のみ設定する。1メッセージ内に独立した複数の計算要求が含まれ (例:「2の10乗は？また2の16乗は？」)、単一の算術式にまとめられない場合は、それらを `+` 等で結合して1つの式に合成しないこと。この場合は `calculate` を使わず `no_tool` と判定する。
- search_history の `query` は、ユーザーの質問文をそのまま渡さず、以下の変換規則でキーワードのみを渡す:
  1. 疑問文の末尾表現 (は/を/って/とか/ありますか/でしょうか/あったり 等) を取り除く
  2. 「好きな/嫌いな/得意な/苦手な」等の評価修飾語を取り除く
  3. 会話で言及された具体的な固有名詞・語があれば、それを優先する (例: 「私の好きなプログラミング言語は？」で直前に "Rust" の話題があれば "プログラミング言語" ではなく "Rust" を渡す)
  4. クエリは会話に実際に登場した語彙を使う。出来事の抽象的な言い換え (例:「最初の訂正」「あの話題」) ではなく、対象となる具体的な事実・固有名詞そのもの (例:「好きな飲み物」) を渡す — search_history は字句の重なりで照合するため、抽象的な言い換えは検索結果 0 件になりやすい (2026-07-20 実インシデント: 「最初の訂正」で検索し No results、同じ会話内で「好きな飲み物」なら毎回ヒット)。
<!-- /PROTECTED -->

<!-- PROTECTED -->
## 出力形式
JSON で出力してください:
- ツールが必要: {{"tool": "ツール名", "args": {{"引数名": "値"}}}}
- ツールが不要: "no_tool"

注意: 利用可能なツール一覧はシステムプロンプトに動的に注入されます。
一覧にないツール名を出力しないでください。
<!-- /PROTECTED -->
""",
    "note_evolve": """\
# メモリノート文脈説明生成

以下のメモリノートの暗黙的な意味・トピック・重要性を捉えた簡潔な文脈説明を生成してください。
この説明は将来の検索精度向上に使用されます。

## 要件
- 1〜2文で簡潔に記述
- ノートの核心的な情報を要約
- 検索キーワードとして機能する語彙を含める

<!-- PROTECTED -->
## 出力形式
文脈説明のみを出力してください。前置きや説明は不要です。
<!-- /PROTECTED -->
""",
}


class AssistPromptManager:
    """アシストモデル タスク別プロンプト管理

    4タスクそれぞれに .md（本文）+ .meta.json（メタ情報）を管理する。
    SystemPromptManager と同じパターンだが、モードではなくタスク単位。
    """

    TASKS = ["rag_necessity", "rag_quality", "tool_call", "note_evolve"]

    def __init__(self, prompt_dir: Path) -> None:
        """
        Args:
            prompt_dir: プロンプトディレクトリ（local/prompts/）
        """
        self.prompt_dir = prompt_dir
        self.contents: dict[str, str] = {}
        self.metas: dict[str, AssistPromptMeta] = {}
        self._load_all()

    def _load_all(self) -> None:
        """起動時に全タスクのプロンプトとメタ情報をロード"""
        self.prompt_dir.mkdir(parents=True, exist_ok=True)
        for task in self.TASKS:
            key = f"assist_{task}"
            if body_exists(self.prompt_dir, key):
                self.contents[task] = read_body(self.prompt_dir, key)
                meta_data = read_meta_dict(self.prompt_dir, key)
                if meta_data is not None:
                    self.metas[task] = self._meta_from_dict(meta_data, task)
                else:
                    self.metas[task] = AssistPromptMeta(task=task)
                    self._save_meta(task)
                # 自己修復: 未編集 (source=default) のままコード側デフォルトが
                # 更新された場合、ディスクの旧デフォルトを現行デフォルトに書き直し、
                # デフォルト更新を既存インストールへ伝播させる。手動編集 (manual) /
                # 進化 (evolution) で更新されたプロンプトは保護する。
                self._refresh_default_if_stale(task)
            else:
                self._create_default(task)

    def get_assist_prompt(self, task: str) -> str:
        """タスク別プロンプト本文を取得

        Args:
            task: タスク名 ("rag_necessity" | "rag_quality" | "tool_call" | "note_evolve")

        Returns:
            プロンプト本文

        Raises:
            ValueError: 不明なタスク名
        """
        if task not in self.contents:
            raise ValueError(f"Unknown assist task: {task}")
        return self.contents[task]

    def get_meta(self, task: str) -> AssistPromptMeta:
        """メタ情報を取得"""
        if task not in self.metas:
            raise ValueError(f"Unknown assist task: {task}")
        return self.metas[task]

    def update_assist_prompt(
        self,
        task: str,
        content: str,
        fitness: float,
    ) -> None:
        """Level 1 進化: 最良候補を採用

        保護セクション（<!-- PROTECTED --> マーカー）が現在のプロンプトに含まれている場合、
        進化候補がそれを維持しているか検証し、欠落時は強制復元する。

        Args:
            task: タスク名
            content: 新しいプロンプト本文
            fitness: 適応度スコア
        """
        if task not in self.TASKS:
            raise ValueError(f"Unknown assist task: {task}")

        # 保護セクション最終ゲート
        from backend.free.agent.prompt_manager import _normalized_equal
        current = self.contents.get(task, "")

        # 段落レベル重複を最終正規化
        content = dedupe_paragraphs(content)

        if not validate_protected_sections(current, content):
            logger.warning(
                "Evolved assist prompt for %s lost protected sections, force-restoring",
                task,
            )
            content = restore_protected_sections(current, content)
            content = dedupe_paragraphs(content)

        # 意味的同一性ガード - 正規化後 current と同じなら no-op
        if _normalized_equal(current, content):
            logger.warning(
                "Evolved assist prompt for %s is semantically identical to current "
                "(fitness=%.3f), skipping update",
                task, fitness,
            )
            return

        self._archive_current(task)
        write_body(self.prompt_dir, f"assist_{task}", content)
        self.contents[task] = content
        meta = self.metas[task]
        meta.version += 1
        meta.updated_at = _now()
        meta.source = "evolution"
        meta.fitness_score = fitness
        self._save_meta(task)
        logger.info(
            "Assist prompt evolved: task=%s, version=%d, fitness=%.3f",
            task, meta.version, fitness,
        )

    def update_manual(self, task: str, content: str) -> None:
        """手動編集によるプロンプト更新"""
        if task not in self.TASKS:
            raise ValueError(f"Unknown assist task: {task}")
        self._archive_current(task)
        write_body(self.prompt_dir, f"assist_{task}", content)
        self.contents[task] = content
        meta = self.metas[task]
        meta.version += 1
        meta.updated_at = _now()
        meta.source = "manual"
        self._save_meta(task)
        logger.info("Assist prompt manual update: task=%s, version=%d", task, meta.version)

    def get_history(self, task: str) -> list[dict]:
        """タスクの履歴一覧を取得"""
        if task not in self.TASKS:
            raise ValueError(f"Unknown assist task: {task}")
        return list_history_entries(self.prompt_dir, f"assist_{task}")

    def rollback(self, task: str, version: int) -> None:
        """特定バージョンにロールバック"""
        if task not in self.TASKS:
            raise ValueError(f"Unknown assist task: {task}")
        content = read_history_version(self.prompt_dir, f"assist_{task}", version)
        self._archive_current(task)
        write_body(self.prompt_dir, f"assist_{task}", content)
        self.contents[task] = content
        meta = self.metas[task]
        meta.version += 1
        meta.updated_at = _now()
        meta.source = "manual"
        self._save_meta(task)
        logger.info(
            "Assist prompt rollback: task=%s to v%03d, new version=%d",
            task, version, meta.version,
        )

    def _archive_current(self, task: str) -> None:
        """現在のプロンプトを history/ に退避"""
        if task not in self.contents:
            return
        archive_to_history(
            self.prompt_dir,
            f"assist_{task}",
            self.metas[task].version,
            self.contents[task],
        )

    def _refresh_default_if_stale(self, task: str) -> None:
        """source=default のプロンプトがコード側デフォルトと乖離していれば再生成する。

        デフォルト更新を既存インストールへ伝播させるための起動時自己修復。
        ``source`` が ``manual`` (手動編集) / ``evolution`` (進化採用) のものは
        ユーザー資産として保護し、書き換えない。``source=default`` かつディスク
        内容が現行 ``DEFAULT_ASSIST_PROMPTS[task]`` と異なる場合のみ書き直す。
        一致していれば no-op (毎回起動で再書き込みしない / 冪等)。
        """
        meta = self.metas.get(task)
        if meta is None or meta.source != "default":
            return
        default = DEFAULT_ASSIST_PROMPTS.get(task)
        if default is None or self.contents.get(task) == default:
            return
        write_body(self.prompt_dir, f"assist_{task}", default)
        self.contents[task] = default
        meta.version = 1
        meta.updated_at = _now()
        meta.source = "default"
        meta.fitness_score = 0.0
        self._save_meta(task)
        logger.info(
            "Refreshed stale default assist prompt: task=%s "
            "(on-disk default differed from current code default)",
            task,
        )

    def _create_default(self, task: str) -> None:
        """デフォルトプロンプトを生成"""
        content = DEFAULT_ASSIST_PROMPTS.get(
            task, f"# assist_{task}\nDefault assist prompt.\n",
        )
        write_body(self.prompt_dir, f"assist_{task}", content)
        self.contents[task] = content
        self.metas[task] = AssistPromptMeta(
            task=task, version=1, updated_at=_now(), source="default",
        )
        self._save_meta(task)
        logger.info("Created default assist prompt: task=%s", task)

    def _save_meta(self, task: str) -> None:
        """メタ情報を JSON ファイルに保存 (infra 層 `_prompt_store_helpers` に委譲)"""
        write_meta_dict(self.prompt_dir, f"assist_{task}", asdict(self.metas[task]))

    @staticmethod
    def _meta_from_dict(data: dict, task: str) -> AssistPromptMeta:
        """`read_meta_dict` の結果を `AssistPromptMeta` にハイドレートする (純粋関数)"""
        return AssistPromptMeta(
            task=data.get("task", task),
            version=data.get("version", 1),
            updated_at=data.get("updated_at", ""),
            source=data.get("source", "default"),
            fitness_score=data.get("fitness_score", 0.0),
        )
