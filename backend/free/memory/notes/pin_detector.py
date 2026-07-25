"""

EvorefMem 統合仕様 の自動 Pin 検出を実装する

役割:
    ユーザー発話 (主に user role の MemoryNote 入力テキスト) に対し、
    「覚えて」「重要」「忘れないで」といった肯定形トリガ語と、
    「覚えなくていい」「忘れて」のような否定形トリガ語を判定し、
    ``pin_flag`` / ``pin_reason`` を導出する。

設計原則:
    - LLM ゼロ。文字列マッチのみで完結する (0.1ms 以内)
    - 純粋関数。I/O は ``load_pin_triggers`` のみで、検出本体は副作用なし
    - i18n: トリガ辞書を YAML に外出し (``pin_triggers.yaml``)。
      ja/en を同一ファイルにフラットに記述するため、文字列マッチで両言語拾える
    - 否定優先 (negative-first): negative に該当した瞬間 pin 発火を抑制する
    - mode 別: チャット / コーディングで固有トリガを持ち、common は両モード適用

辞書ファイルの解決:
    - **shipped default**: ``backend/free/memory/_defaults/triggers/pin_triggers.yaml``
      (パッケージ同梱 / git tracked)
    - **user override** (任意): ``local/triggers/pin_triggers.yaml``
      (``local/`` は git-ignored のためリポジトリ差分に現れない)
    - 解決は :func:`backend.free.memory._defaults.resolve_trigger_file`
      が override を優先し、無ければ default にフォールバックする。
      起動時コピー等のマイグレーションは行わない。

呼び出し側:
    ``ShortTermMemory.absorb()`` から本検出器を呼び、生成した ``MemoryNote``
    の ``pin_flag`` / ``pin_reason`` を設定する。SemanticFact への昇格は
    sleep-time Step 8 Extractor 以降で扱う

設定:
    ``config.yaml`` の ``memory.pin`` セクションで以下を制御する::

        memory:
          pin:
            auto_detect: true            # 自動検出の有効化
            auto_detect_confirm: false   # 検出時に確認を必須化するか
            unlimited: true              # pin 数上限無し

    トリガ辞書のパスは config では指定しない (convention-based 解決)。
"""

from __future__ import annotations

import re

import threading
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from backend.free.memory.types import MemoryMode
from backend.log_config import get_logger

logger = get_logger("memory.pin_detector")


# ──────────────────────────────────────────────────────────────────────────
# データクラス
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PinDetection:
    """Pin 検出結果。

    Attributes:
        should_pin: 最終的に pin するか
        matched: マッチしたトリガ語 (肯定/否定どちらも)
        reason: ``pin_reason`` に格納する文字列
            (例: ``auto_detect:覚えて`` / ``auto_detect_negated:忘れて``)
        negated: 否定形にマッチしたか (``should_pin=False`` の理由判別用)
    """

    should_pin: bool
    matched: str | None = None
    reason: str | None = None
    negated: bool = False


@dataclass
class PinTriggers:
    """ロード済みのトリガ語辞書。

    `mode_positive["chat"]` / `mode_positive["coding"]` は mode 固有 +
    common をマージ済みのリストを保持する。`negative` は両モード共通。
    全要素は NFKC + lowercase 正規化済み。
    """

    mode_positive: dict[MemoryMode, list[str]] = field(default_factory=dict)
    negative: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not any(self.mode_positive.values()) and not self.negative


# ──────────────────────────────────────────────────────────────────────────
# 正規化
# ──────────────────────────────────────────────────────────────────────────


def _normalize(text: str) -> str:
    """NFKC + 小文字化 + 連続空白圧縮 (検出用の軽量正規化)。"""
    if not text:
        return ""
    norm = unicodedata.normalize("NFKC", text).lower()
    # 連続空白を 1 つにまとめる (改行/タブも単一スペース化)
    return " ".join(norm.split())


# ──────────────────────────────────────────────────────────────────────────
# YAML ロード
# ──────────────────────────────────────────────────────────────────────────


def load_pin_triggers(path: str | Path) -> PinTriggers:
    """YAML からトリガ辞書をロードする。

    ファイルが存在しない / パース失敗時は空の ``PinTriggers`` を返し、
    呼び出し側で「検出無効」相当として扱える (例外を投げない)。

    YAML スキーマ::

        positive:
          common: [str, ...]
          chat:   [str, ...]
          coding: [str, ...]
        negative:
          common: [str, ...]
    """
    p = Path(path)
    if not p.exists():
        logger.warning("pin_triggers file not found: %s — auto-detect disabled", p)
        return PinTriggers()
    try:
        raw: Any = yaml.safe_load(p.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("pin_triggers load failed (%s): %s", p, exc)
        return PinTriggers()
    if not isinstance(raw, dict):
        logger.warning("pin_triggers root is not mapping: %s", p)
        return PinTriggers()

    pos = raw.get("positive") or {}
    neg = raw.get("negative") or {}

    def _flatten(items: Any) -> list[str]:
        if not isinstance(items, list):
            return []
        out: list[str] = []
        for it in items:
            if not isinstance(it, str):
                continue
            n = _normalize(it)
            if n:
                out.append(n)
        return out

    common = _flatten(pos.get("common"))
    chat_only = _flatten(pos.get("chat"))
    coding_only = _flatten(pos.get("coding"))
    negative = _flatten(neg.get("common"))

    triggers = PinTriggers(
        mode_positive={
            "chat": _dedup(common + chat_only),
            "coding": _dedup(common + coding_only),
        },
        negative=_dedup(negative),
    )
    logger.info(
        "pin_triggers loaded: chat=%d, coding=%d, negative=%d",
        len(triggers.mode_positive["chat"]),
        len(triggers.mode_positive["coding"]),
        len(triggers.negative),
    )
    return triggers


def _dedup(items: list[str]) -> list[str]:
    """順序保持で重複排除。長いトリガを先頭側に寄せる (より具体的な
    マッチを ``matched`` に拾わせるため)。
    """
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    # 長い順 (タイ時は元の順序維持) で並べ替え
    out.sort(key=lambda s: -len(s))
    return out


# ──────────────────────────────────────────────────────────────────────────
# 検出本体
# ──────────────────────────────────────────────────────────────────────────


#: 疑問形の文末 (「覚えていますか？」等)。
_PIN_QUESTION_ENDING_RE = re.compile(r"[?？]\s*$")

#: 文区切り (。！？!?) の直後で分割する。
_PIN_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?])\s*")


def _trigger_evidence_is_question_only(content: str, word: str) -> bool:
    """トリガ語を含む文がすべて疑問形かを判定する。

    「覚えておいて」は保存指示だが「覚えていますか？」は **想起の依頼** であり、
    保存すべき事実ではない。両方とも「覚えて」を含むため、トリガ語の部分一致
    だけでは区別できない。

    これを pin してしまうと、その質問文が Tier 1 の pinned ノートとして
    毎ターン注入され続け、MemoryInjector の関連度ゲート (pinned は
    ユーザーの明示指定として常に通す) も素通りする。実測 2026-07-25:
    「ちょっと確認ですが、私の名前と出身地、覚えていますか？」が pin され、
    無関係なターンで base モデルがこれを「いま答えるべき質問」と解釈していた。

    ChatExtractor の ``_tag_evidence_is_question_only`` と同旨 (EvorefMem 内の
    別モジュールなので最小実装を持つ)。
    """
    sentences = [s for s in _PIN_SENTENCE_SPLIT_RE.split(content) if s.strip()]
    relevant = [s for s in sentences if word in s.lower()]
    if not relevant:
        return False
    return all(_PIN_QUESTION_ENDING_RE.search(s.strip()) for s in relevant)


def detect_pin(
    content: str,
    mode: MemoryMode,
    triggers: PinTriggers,
) -> PinDetection:
    """テキストから自動 Pin 判定を行う。

    評価順:
        1. content を ``_normalize``
        2. ``triggers.negative`` に該当 → ``should_pin=False``,
           ``negated=True``, ``reason="auto_detect_negated:<word>"``
        3. ``triggers.mode_positive[mode]`` に該当 → ``should_pin=True``,
           ``reason="auto_detect:<word>"``
        4. いずれにもマッチしなければ ``should_pin=False``

    ``triggers`` が空の場合は常に ``should_pin=False`` を返す。

    Args:
        content: 判定対象テキスト (通常はユーザー発話)
        mode: ``chat`` / ``coding``
        triggers: ``load_pin_triggers`` で得た辞書

    Returns:
        ``PinDetection``
    """
    if not content or triggers.empty:
        return PinDetection(should_pin=False)

    norm = _normalize(content)
    if not norm:
        return PinDetection(should_pin=False)

    # 1) 否定優先
    for word in triggers.negative:
        if word and word in norm:
            return PinDetection(
                should_pin=False,
                matched=word,
                reason=f"auto_detect_negated:{word}",
                negated=True,
            )

    # 2) 肯定マッチ (mode 別 + common merged)
    positives = triggers.mode_positive.get(mode, [])
    for word in positives:
        if word and word in norm:
            # 「覚えていますか？」= 想起の依頼であって保存指示ではない。
            # pin すると Tier 1 で毎ターン注入され、関連度ゲートも素通りする。
            if _trigger_evidence_is_question_only(norm, word):
                return PinDetection(
                    should_pin=False,
                    matched=word,
                    reason=f"auto_detect_question:{word}",
                    negated=False,
                )
            return PinDetection(
                should_pin=True,
                matched=word,
                reason=f"auto_detect:{word}",
                negated=False,
            )

    return PinDetection(should_pin=False)


# ──────────────────────────────────────────────────────────────────────────
# シングルトン (config 経由のグローバルアクセス)
# ──────────────────────────────────────────────────────────────────────────


_TRIGGERS_LOCK = threading.Lock()
_TRIGGERS_CACHE: dict[str, PinTriggers] = {}


def get_pin_triggers(path: str | Path) -> PinTriggers:
    """パスをキーとした ``PinTriggers`` のキャッシュ取得。

    プロセス内シングルトン (パスごとに 1 インスタンス)。テストでは
    ``reset_pin_triggers_cache()`` でキャッシュをクリアできる。
    """
    key = str(Path(path).resolve())
    with _TRIGGERS_LOCK:
        cached = _TRIGGERS_CACHE.get(key)
        if cached is not None:
            return cached
        triggers = load_pin_triggers(path)
        _TRIGGERS_CACHE[key] = triggers
        return triggers


def reset_pin_triggers_cache() -> None:
    """テスト用: キャッシュ全消去。"""
    with _TRIGGERS_LOCK:
        _TRIGGERS_CACHE.clear()


def resolve_pin_triggers_path(triggers_dir: str | Path | None = None) -> Path:
    """``pin_triggers.yaml`` の解決パスを返す。

    ``triggers_dir/pin_triggers.yaml`` (user override) が存在すればそれ、
    なければ package 同梱 default を返す。詳細は
    :func:`backend.free.memory._defaults.resolve_trigger_file` 参照。
    """
    from backend.free.memory._defaults import resolve_trigger_file

    return resolve_trigger_file("pin_triggers.yaml", triggers_dir=triggers_dir)


def get_pin_triggers_for(triggers_dir: str | Path | None = None) -> PinTriggers:
    """user override (``<triggers_dir>/pin_triggers.yaml``) または package
    同梱 default から ``PinTriggers`` を得る。

    シングルトンキャッシュを共有するため、同じ解決パスに対する複数回呼出
    では同一インスタンスが返る。
    """
    return get_pin_triggers(resolve_pin_triggers_path(triggers_dir))
