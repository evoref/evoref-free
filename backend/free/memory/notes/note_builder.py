"""NoteBuilder: A-MEM ノート構築（LLM 不要）

設計書 §5.2 / EvorefMem 統合仕様に基づく、キーワード抽出・
タグ付け・候補ファクト分類を一体的に行うクラス群。すべての処理は 0.1ms 以内で
完了し、LLM 呼び出しは行わない。

クラス階層:
- :class:`NoteBuilder` — 共通基底。``extract_keywords`` / ``auto_tag`` /
  ``initial_score`` の汎用ヘルパを提供する。``build()`` は EvorefMem の
  ``MemoryNote`` 拡張フィールド (mode / project_id / source / is_tool_output /
  is_code_block / extraction_skipped) を含む dict を返す。
- :class:`ChatNoteBuilder` — チャットモード用レンズ。``personal_fact`` /
  ``world_fact`` / ``preference`` / ``emotion`` / ``opinion`` の候補抽出を行う。
- :class:`CodingNoteBuilder` — コーディングモード用レンズ。``project`` /
  ``decision`` / ``commitment`` / ``task`` / ``coding`` の候補抽出を行う。
  コードブロックは完全スキップ、ツール出力は extraction_skipped を立てる。

候補ファクトタグは sleep-time Step 8 の Extractor がこのノートを
拾い上げる際のヒントとして使用される。本フェーズでは抽出までは行わない。
"""

from __future__ import annotations

import re
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from backend.free.memory.types import MemoryMode, NoteSource
from backend.log_config import get_logger

logger = get_logger("memory.note_builder")


# reasoning モデル (LFM2 / Qwen3 等) が応答に残す <think>...</think> を除去する。
# メモリ抽出が思考を STM ノートに焼き込むと、後続チャットで意味検索により再注入され
# 話題汚染 (例: ニュース質問に過去の天気ノートで返答) を招く。
# cf. backend.free.optimizer.prompt_evolver の同名処理 (pillar 境界のため非共有)。
_THINK_TAG_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think\b[^>]*>", re.IGNORECASE)


def _strip_think_tags(text: str) -> str:
    """``<think>...</think>`` (未閉鎖含む) を除去する。

    閉じたブロックは削除し、未閉鎖 ``<think>`` (暴走/打ち切り) 以降はすべて思考と
    みなして破棄する。本文に思考が無ければ原文をそのまま返す。
    """
    if "<think" not in text.lower():
        return text
    text = _THINK_TAG_RE.sub("", text)
    m = _THINK_OPEN_RE.search(text)
    if m:
        text = text[: m.start()]
    return text.strip()


# ──────────────────────────────────────────────────────────────────────────
# 候補ファクト判定用トリガ辞書のロード
# ──────────────────────────────────────────────────────────────────────────


#: ``<mode>`` → ``<fact_type>`` → 部分一致トリガ語 tuple
FactTriggerMap = dict[str, dict[str, tuple[str, ...]]]

#: 同梱 default で期待される fact_type 集合 (mode 別)
_EXPECTED_TAGS: dict[MemoryMode, tuple[str, ...]] = {
    "chat": ("personal_fact", "world_fact", "preference", "emotion", "opinion"),
    "coding": ("project", "decision", "commitment", "task", "coding"),
}

_TRIGGERS_LOCK = threading.Lock()
_TRIGGERS_CACHE: dict[str, FactTriggerMap] = {}

#: プロセス全体で共有される user override 配置先 (通常 ``local/triggers/``)。
#: app_factory が :func:`set_default_triggers_dir` で起動時にセットする。
#: これにより、明示的な ``triggers_dir`` を渡さずに生成された Builder /
#: Extractor (``sleep/extraction.py`` 内の ``ChatExtractor()`` 等) も
#: user override を参照できるようになる。
_DEFAULT_TRIGGERS_DIR: str | Path | None = None


def set_default_triggers_dir(triggers_dir: str | Path | None) -> None:
    """モジュールレベルの default triggers_dir を設定し、singleton builder を
    刷新する。``None`` を渡すと無効化 (= 常に package 同梱 default 使用)。

    app_factory 起動時に ``PathResolver.resolve_local("triggers_dir")`` の
    値を渡す想定。テスト / 再設定のため何度呼んでも安全。
    """
    global _DEFAULT_TRIGGERS_DIR, _CHAT_BUILDER, _CODING_BUILDER
    _DEFAULT_TRIGGERS_DIR = triggers_dir
    # Singleton を再構築してキャッシュを無効化 (新しい default を読ませる)。
    _CHAT_BUILDER = ChatNoteBuilder()
    _CODING_BUILDER = CodingNoteBuilder()


def get_default_triggers_dir() -> str | Path | None:
    """現在の default triggers_dir を返す (テスト / introspection 用)。"""
    return _DEFAULT_TRIGGERS_DIR


def _normalize_trigger(text: str) -> str:
    """trigger 語を保存用に正規化 (NFKC + lowercase、空白は保持)。

    ``candidate_fact_tags`` 側は入力テキストを ``str.lower()`` でのみ比較
    するため、全角英数字をそのまま保存すると一致しない。YAML 取込時に
    NFKC + lowercase を掛けることで pin_triggers.yaml と同じ ja/en フラット
    混在運用を許す。
    """
    if not text:
        return ""
    return unicodedata.normalize("NFKC", text).lower()


def _coerce_triggers(items: Any) -> tuple[str, ...]:
    if not isinstance(items, list):
        return ()
    out: list[str] = []
    for it in items:
        if not isinstance(it, str):
            continue
        n = _normalize_trigger(it)
        if n:
            out.append(n)
    return tuple(out)


def load_fact_triggers(path: str | Path) -> FactTriggerMap:
    """``fact_triggers.yaml`` をロードして ``{mode: {fact_type: triggers}}``
    を返す。ファイル欠落 / パース失敗時は空辞書を返し、呼び出し側は
    「候補判定無効」相当として扱える (例外を投げない)。
    """
    p = Path(path)
    result: FactTriggerMap = {"chat": {}, "coding": {}}
    if not p.exists():
        logger.warning("fact_triggers file not found: %s — candidates disabled", p)
        return result
    try:
        raw: Any = yaml.safe_load(p.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("fact_triggers load failed (%s): %s", p, exc)
        return result
    if not isinstance(raw, dict):
        logger.warning("fact_triggers root is not mapping: %s", p)
        return result

    for mode_key in ("chat", "coding"):
        section = raw.get(mode_key) or {}
        if not isinstance(section, dict):
            continue
        mapping: dict[str, tuple[str, ...]] = {}
        for tag, triggers in section.items():
            if not isinstance(tag, str):
                continue
            words = _coerce_triggers(triggers)
            if words:
                mapping[tag] = words
        result[mode_key] = mapping

    logger.info(
        "fact_triggers loaded: chat=%d tags / coding=%d tags (from %s)",
        len(result["chat"]),
        len(result["coding"]),
        p,
    )
    return result


def get_fact_triggers(path: str | Path) -> FactTriggerMap:
    """パスをキーとした ``FactTriggerMap`` のキャッシュ取得 (プロセス内シングルトン)。"""
    key = str(Path(path).resolve())
    with _TRIGGERS_LOCK:
        cached = _TRIGGERS_CACHE.get(key)
        if cached is not None:
            return cached
        triggers = load_fact_triggers(path)
        _TRIGGERS_CACHE[key] = triggers
        return triggers


def reset_fact_triggers_cache() -> None:
    """テスト用: キャッシュ全消去。"""
    with _TRIGGERS_LOCK:
        _TRIGGERS_CACHE.clear()


def resolve_fact_triggers_path(triggers_dir: str | Path | None = None) -> Path:
    """``fact_triggers.yaml`` の解決パスを返す (user override → default)。"""
    from backend.free.memory._defaults import resolve_trigger_file

    return resolve_trigger_file("fact_triggers.yaml", triggers_dir=triggers_dir)


def _default_fact_triggers() -> FactTriggerMap:
    """package 同梱 default を取得 (初期化時 / テスト既定値として使用)。"""
    return get_fact_triggers(resolve_fact_triggers_path(None))


# ──────────────────────────────────────────────────────────────────────────
# 共通基底
# ──────────────────────────────────────────────────────────────────────────


class NoteBuilder:
    """A-MEM ノート構築の共通基底 — LLM ゼロ。"""

    #: このビルダのデフォルトモード。サブクラスで上書きする。
    mode: MemoryMode = "chat"

    # 日本語キーワード抽出: 形態素解析不要の軽量版
    KEYWORD_PATTERNS = [
        re.compile(r"[A-Za-z][A-Za-z0-9_.-]+"),   # 英数字トークン
        re.compile(r"[\u4e00-\u9fff]{2,8}"),        # 漢字2-8文字
        re.compile(r"[\u30a0-\u30ff]{2,}"),          # カタカナ2文字以上
    ]

    # 自動タグ: キーワードからルールベースで付与（汎用）
    TAG_RULES: dict[str, list[str]] = {
        "coding": ["python", "code", "バグ", "実装", "関数", "class", "def", "import"],
        "model": ["gguf", "llama", "qwen", "lora", "モデル", "推論"],
        "preference": ["好き", "嫌い", "いつも", "よく使う", "お気に入り"],
        "fact": ["です", "である", "とは", "定義"],
        "task": ["やって", "して", "作って", "教えて", "確認"],
    }

    # ─── markdown フェンス検知 ──
    _CODE_FENCE_RE = re.compile(r"```")

    # ─── 公開 API ───────────────────────────────────────────────────────

    def build(
        self,
        content: str,
        session_id: str,
        *,
        role: str = "user",
        source: NoteSource | None = None,
        mode: MemoryMode | None = None,
        project_id: str | None = None,
        is_tool_output: bool = False,
    ) -> dict[str, Any]:
        """ノート構築 dict を返す — 0.1ms 以内・LLM 呼び出し無し。

        Args:
            content: ノート内容テキスト
            session_id: セッション ID
            role: 発言者ロール (``user`` / ``assistant``)
            source: 発生源 (``NoteSource``)。``None`` の場合は ``role`` から推測
            mode: モード。``None`` の場合はビルダの ``self.mode`` を使う
            project_id: コーディングモード時のプロジェクト ID
            is_tool_output: ツール出力か。``True`` の場合 STM 以降は除外される

        Returns:
            ``MemoryNote`` 構築用の dict
        """
        # reasoning モデルの思考漏れ (<think>...</think>) をノート化しない。
        # 残すと STM に焼き込まれ後続チャットへ再注入され話題汚染を招く。
        content = _strip_think_tags(content)

        effective_source: NoteSource = source if source is not None else (
            "assistant" if role == "assistant" else "user"
        )
        effective_mode: MemoryMode = mode if mode is not None else self.mode

        is_code_block = self._detect_code_block(content)
        extraction_skipped = False
        skip_reason: str | None = None

        if is_code_block:
            extraction_skipped = True
            skip_reason = "code_block"
        elif is_tool_output:
            extraction_skipped = True
            skip_reason = "tool_output"

        # コードブロックなら keywords/tags も付与しない (extraction を完全に止める)
        if is_code_block:
            keywords: list[str] = []
            generic_tags: list[str] = []
            candidate_tags: list[str] = []
        else:
            keywords = self.extract_keywords(content)
            generic_tags = self.auto_tag(content)
            # ツール出力は候補抽出も行わない (sleep-time Extractor に流さない)
            candidate_tags = (
                [] if is_tool_output else self.candidate_fact_tags(content)
            )

        merged_tags = sorted(set(generic_tags + candidate_tags))

        now = time.time()
        return {
            "id": uuid4().hex[:12],
            "content": content,
            "keywords": keywords,
            "tags": merged_tags,
            "embedding": None,
            "lightmem_score": self.initial_score(content, role),
            "confidence": self.source_confidence(effective_source),
            "created_at": now,
            "accessed_at": now,
            "access_count": 0,
            "session_id": session_id,
            "context_description": "",
            "evolution_pending": True,
            # ── 拡張フィールド ──
            "source": effective_source,
            "mode": effective_mode,
            "project_id": project_id,
            "is_tool_output": is_tool_output,
            "is_code_block": is_code_block,
            "extraction_skipped": extraction_skipped,
            "extraction_skip_reason": skip_reason,
            "candidate_fact_tags": candidate_tags,
        }

    # ─── ヘルパ ─────────────────────────────────────────────────────────

    @classmethod
    def extract_keywords(cls, content: str) -> list[str]:
        """正規表現ベースのキーワード抽出（LLM 不要）"""
        keywords: list[str] = []
        for pattern in cls.KEYWORD_PATTERNS:
            keywords.extend(pattern.findall(content))
        seen: set[str] = set()
        result: list[str] = []
        for kw in keywords:
            lower = kw.lower()
            if lower not in seen and len(kw) >= 2:
                seen.add(lower)
                result.append(kw)
        return result[:10]

    @classmethod
    def auto_tag(cls, content: str) -> list[str]:
        """ルールベースの自動タグ付け (汎用)"""
        content_lower = content.lower()
        tags: list[str] = []
        for tag, trigger_words in cls.TAG_RULES.items():
            if any(w in content_lower for w in trigger_words):
                tags.append(tag)
        return tags

    @staticmethod
    def initial_score(content: str, role: str = "user") -> float:
        """初期 LightMem スコア（ルールベース）"""
        score = 0.5
        if len(content) > 200:
            score += 0.1
        if role == "user":
            score += 0.1
        return min(1.0, score)

    # 発生源別の初期 confidence。NoteEvolver はこの値が
    # ``memory.note_evolver.confidence_threshold`` (既定 0.7) 未満のノートのみ
    # LLM 進化 (context_description 生成) の対象にする。ユーザー発話は権威性が
    # 高く LLM 進化不要なので閾値以上、assistant / rag / system 由来のノートは
    # context 補強の価値があるので閾値未満に置く。
    _SOURCE_CONFIDENCE: dict[str, float] = {
        "user": 1.0,
        "assistant": 0.5,
        "rag": 0.6,
        "system": 0.6,
    }

    @classmethod
    def source_confidence(cls, source: str) -> float:
        """発生源 (``NoteSource``) 別の初期 confidence を返す。"""
        return cls._SOURCE_CONFIDENCE.get(source, 1.0)

    @classmethod
    def _detect_code_block(cls, content: str) -> bool:
        """markdown フェンス (``` ... ```) を含むかを判定。

        コーディングモードで生成された markdown 応答にフェンスが含まれると、
        コード片はファクト抽出対象から外したいので、ここで早期検出する。
        部分一致でも True を返す (フェンス開閉が片方のみでも保守的に skip)。
        """
        if not content:
            return False
        return bool(cls._CODE_FENCE_RE.search(content))

    # サブクラスでオーバーライドする
    def candidate_fact_tags(self, content: str) -> list[str]:
        """モード別の候補ファクトタイプを返す。

        基底クラスは何も返さない。``ChatNoteBuilder`` /
        ``CodingNoteBuilder`` でオーバーライドする。
        """
        return []


# ──────────────────────────────────────────────────────────────────────────
# ChatNoteBuilder
# ──────────────────────────────────────────────────────────────────────────


class _ModeAwareNoteBuilder(NoteBuilder):
    """mode 固有のトリガ辞書を YAML からロードする共通基底。

    shipped default は :mod:`backend.free.memory._defaults` 配下の
    ``fact_triggers.yaml`` に、user override は ``<triggers_dir>/
    fact_triggers.yaml`` (通常 ``local/triggers/``) に置く。
    ``triggers_dir=None`` の場合は default のみ参照する。
    """

    mode: MemoryMode

    def __init__(self, triggers_dir: str | Path | None = None) -> None:
        # 明示的に ``triggers_dir`` を渡された場合はそれを優先、
        # 省略時はモジュールレベル default (:data:`_DEFAULT_TRIGGERS_DIR`) を
        # 動的に参照する。後者は ``set_default_triggers_dir`` による後追い
        # 変更を反映するため property 内で遅延解決する。
        self._explicit_triggers_dir = triggers_dir
        self._fact_triggers_cached: dict[str, tuple[str, ...]] | None = None

    @property
    def _effective_triggers_dir(self) -> str | Path | None:
        if self._explicit_triggers_dir is not None:
            return self._explicit_triggers_dir
        return _DEFAULT_TRIGGERS_DIR

    @property
    def fact_triggers(self) -> dict[str, tuple[str, ...]]:
        """mode 対応セクションの ``{fact_type: triggers}`` マップ。

        初回呼出しで YAML からロードしてキャッシュする。プロセス内で
        同じ解決パスを複数インスタンスが共有する (get_fact_triggers 側で
        再利用される)。
        """
        if self._fact_triggers_cached is None:
            path = resolve_fact_triggers_path(self._effective_triggers_dir)
            table = get_fact_triggers(path)
            self._fact_triggers_cached = table.get(self.mode, {})
        return self._fact_triggers_cached

    def candidate_fact_tags(self, content: str) -> list[str]:
        text = content.lower()
        results: list[str] = []
        for tag, triggers in self.fact_triggers.items():
            if any(t in text for t in triggers):
                results.append(tag)
        return results


class ChatNoteBuilder(_ModeAwareNoteBuilder):
    """チャットモード用 NoteBuilder。

    会話重視のレンズ — ユーザーの自己開示・好み・感情・意見を捕捉する。
    実際のファクト抽出は sleep-time Step 8 で行われ、本クラスは
    候補タグを ``MemoryNote.tags`` に積んでヒントを残すだけ。

    候補ファクトタイプ: ``personal_fact`` / ``world_fact`` / ``preference`` /
    ``emotion`` / ``opinion``。実際のトリガ語は
    ``backend/free/memory/_defaults/triggers/fact_triggers.yaml`` を参照。
    """

    mode: MemoryMode = "chat"


# ──────────────────────────────────────────────────────────────────────────
# CodingNoteBuilder
# ──────────────────────────────────────────────────────────────────────────


class CodingNoteBuilder(_ModeAwareNoteBuilder):
    """コーディングモード用 NoteBuilder。

    目的達成重視のレンズ — プロジェクトルール・判断・約束・タスク・コード関連
    知識を捕捉する。コードブロック (``` フェンス) は完全スキップ、ツール出力は
    extraction_skipped を立てる (基底クラスの ``build()`` 内で処理)。

    候補ファクトタイプ: ``project`` / ``decision`` / ``commitment`` /
    ``task`` / ``coding``。実際のトリガ語は
    ``backend/free/memory/_defaults/triggers/fact_triggers.yaml`` を参照。
    """

    mode: MemoryMode = "coding"


# ──────────────────────────────────────────────────────────────────────────
# Builder ファクトリ
# ──────────────────────────────────────────────────────────────────────────


_CHAT_BUILDER = ChatNoteBuilder()
_CODING_BUILDER = CodingNoteBuilder()


def get_note_builder(mode: MemoryMode) -> NoteBuilder:
    """モード別 NoteBuilder のシングルトンを返す。

    builder インスタンスはステートレスなので使い回して問題ない。
    ``fact_triggers.yaml`` の user override を使いたい場合は、
    ``ChatNoteBuilder(triggers_dir=...)`` / ``CodingNoteBuilder(triggers_dir=...)``
    を直接構築すること (通常は extractor / STM 側から注入される)。
    """
    if mode == "coding":
        return _CODING_BUILDER
    return _CHAT_BUILDER
