"""規則台帳 (rules ledger) — 静的 system プロンプトの SSOT (f_03 §7.1.1)。

本文 `.md` は **レンダ結果**であり編集対象ではない。真実は `<mode>.rules.json`
で、静的 system はここから決定論のレンダラが毎回同じ順序で組み立てる。
LLM はレンダに関与しない。

背景 (2026-09-03 ライブ監査): chat の静的 system は 24 規則、うち 17 が
PROTECTED。Instruction Stacking Collapse (arXiv 2608.02639) は 1→20 規則で
遵守率 96%→60/43/20% と報告し、IFScale (2507.11538) は先頭の規則ほど守られる
(primacy) と報告する。ACE (2510.04618) は LLM の全文書き直しが要点を落として
崩壊すると報告する。したがって「短くする」のではなく、規則を項目として持ち、
順序・束ね・衝突・削除を **計数と決定論** で扱う。

台帳の文法は既存の Markdown 本文と可逆:

    # タイトル                      → kind="title"
    (見出し前の段落)                → kind="intro"
    ## 見出し                       → category (以降の箇条が属する)
    - 箇条                          → kind="bullet"
    <!-- PROTECTED --> … <!-- /PROTECTED -->  → 区間内の規則は protected=True

id は「カテゴリ slug + 本文ハッシュ」で決定論に採番し、既存台帳と本文が一致する
規則は id を引き継ぐ (計数 (§3.5.1) を失わないため)。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from backend.free.agent.prompt_utils import PROTECTED_CLOSE, PROTECTED_OPEN
from backend.io import atomic_write_text
from backend.log_config import get_logger
from backend.utils import estimate_tokens

logger = get_logger("agent.prompt_ledger")

LEDGER_SCHEMA_VERSION = 1

#: レンダで **衝突の優先規則** を足すときの定型文 (locale 別)。
_OVERRIDE_NOTES: dict[str, str] = {
    "ja": "（他の項目と矛盾する場合はこの項を優先する）",
    "en": "(When this conflicts with another item, this item takes precedence)",
}

_WS_RE = re.compile(r"\s+")


@dataclass
class Rule:
    """台帳の 1 項目。計数 (helpful / harmful / last_fired) はレンダに出ない。"""

    id: str
    category: str
    text: str
    kind: str = "bullet"  # "title" | "intro" | "bullet"
    priority: int = 0
    protected: bool = False
    verifier: str | None = None
    overrides: list[str] = field(default_factory=list)
    source_incident: str = ""
    helpful: int = 0
    harmful: int = 0
    last_fired: str = ""


@dataclass
class Ledger:
    """1 モードぶんの台帳。"""

    mode: str
    locale: str
    rules: list[Rule] = field(default_factory=list)
    schema_version: int = LEDGER_SCHEMA_VERSION

    def by_id(self, rule_id: str) -> Rule | None:
        for rule in self.rules:
            if rule.id == rule_id:
                return rule
        return None

    def content_hash(self) -> str:
        """レンダに影響する内容だけのハッシュ (計数は含めない)。"""
        payload = [
            (r.kind, r.category, r.text, r.protected, r.priority, tuple(r.overrides))
            for r in self.rules
        ]
        return hashlib.sha1(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()[:12]


# ──────────────────────────────────────────────────────────────────────────
# 解析 (Markdown → 台帳)
# ──────────────────────────────────────────────────────────────────────────


def normalize_text(text: str) -> str:
    """id の引き継ぎ判定に使う正規化 (空白の揺れを無視)。"""
    return _WS_RE.sub(" ", text or "").strip()


def _slug(text: str) -> str:
    base = re.sub(r"[^0-9A-Za-z぀-ヿ一-鿿]+", "_", text or "").strip("_")
    return base[:24] or "x"


def mint_id(category: str, text: str, kind: str = "bullet") -> str:
    """決定論の id。同じ (カテゴリ, 本文) は常に同じ id になる。"""
    digest = hashlib.sha1(normalize_text(text).encode("utf-8")).hexdigest()[:8]
    return f"{kind}.{_slug(category)}.{digest}"


def parse_markdown(
    text: str,
    *,
    mode: str,
    locale: str,
    existing: Ledger | None = None,
) -> Ledger:
    """Markdown 本文を台帳へ分解する (純粋関数)。

    ``existing`` を渡すと、本文が一致する規則の id・計数・verifier 等を引き継ぐ。
    一致しない行は新規 id、既存にあって本文に無い規則は落ちる
    (= 進化 / 手編集で削られた)。
    """
    carry: dict[str, Rule] = {}
    if existing is not None:
        for rule in existing.rules:
            carry.setdefault((rule.kind, normalize_text(rule.text)), rule)

    rules: list[Rule] = []
    category = ""
    protected = False
    intro_lines: list[str] = []
    seen_heading = False

    def flush_intro() -> None:
        if intro_lines:
            body = "\n".join(intro_lines).strip()
            if body:
                rules.append(_make_rule("intro", category or "intro", body, protected, carry, base_priority=1000))
            intro_lines.clear()

    for raw in (text or "").splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if stripped == PROTECTED_OPEN:
            flush_intro()
            protected = True
            continue
        if stripped == PROTECTED_CLOSE:
            flush_intro()
            protected = False
            continue
        if stripped.startswith("# ") and not seen_heading and not rules:
            rules.append(_make_rule("title", "title", stripped[2:].strip(), False, carry, base_priority=2000))
            continue
        if stripped.startswith("## "):
            flush_intro()
            seen_heading = True
            category = stripped[3:].strip()
            continue
        if stripped.startswith("- "):
            flush_intro()
            rules.append(_make_rule("bullet", category, stripped[2:].strip(), protected, carry))
            continue
        if not stripped:
            if intro_lines:
                flush_intro()
            continue
        # 見出しの外の平文 (冒頭の方針文など)
        intro_lines.append(stripped)
    flush_intro()
    return Ledger(mode=mode, locale=locale, rules=rules)


def _make_rule(
    kind: str, category: str, text: str, protected: bool,
    carry: dict, *, base_priority: int = 0,
) -> Rule:
    prior = carry.get((kind, normalize_text(text)))
    if prior is not None:
        return Rule(
            id=prior.id, category=category, text=text, kind=kind,
            priority=prior.priority if prior.priority else base_priority,
            protected=protected, verifier=prior.verifier,
            overrides=list(prior.overrides), source_incident=prior.source_incident,
            helpful=prior.helpful, harmful=prior.harmful, last_fired=prior.last_fired,
        )
    return Rule(
        id=mint_id(category, text, kind), category=category, text=text,
        kind=kind, priority=base_priority, protected=protected,
    )


# ──────────────────────────────────────────────────────────────────────────
# レンダ (台帳 → Markdown)
# ──────────────────────────────────────────────────────────────────────────


def _category_order(rules: list[Rule]) -> list[str]:
    """出現順を保ったカテゴリ列 (protected 群を先頭へ寄せる = primacy)。"""
    order: list[str] = []
    for rule in rules:
        if rule.kind != "bullet":
            continue
        if rule.category not in order:
            order.append(rule.category)
    protected_cats = [c for c in order if any(r.category == c and r.protected for r in rules)]
    others = [c for c in order if c not in protected_cats]
    return protected_cats + others


def _render_category(rules: list[Rule], category: str, locale: str) -> str:
    bullets = [r for r in rules if r.kind == "bullet" and r.category == category]
    # priority 降順、同点は台帳順 (安定ソート)
    bullets.sort(key=lambda r: -r.priority)
    lines = [f"## {category}"] if category else []
    note = _OVERRIDE_NOTES.get(locale, _OVERRIDE_NOTES["ja"])
    for rule in bullets:
        text = rule.text
        if rule.overrides:
            text = f"{text}{note}"
        lines.append(f"- {text}")
    return "\n".join(lines)


def render_markdown(
    ledger: Ledger,
    *,
    max_tokens: int | None = None,
    hoist_shared: set[str] | None = None,
) -> tuple[str, list[str]]:
    """台帳を Markdown 本文へ描く (決定論)。

    順序: タイトル → 冒頭段落 → **protected カテゴリ** (primacy) → その他カテゴリ。
    ``max_tokens`` を超えるときは priority の低い非 protected 箇条から落とし、
    落とした id の列を第 2 戻り値で返す (protected だけで超える場合は
    ``ValueError`` — 構成エラー)。

    ``hoist_shared`` に id 集合を渡すと、その箇条を **タイトルより前** に
    描く (モード間で共通の接頭辞を最大化する layout。``get_prompt_static`` 用。
    人が読む raw 本文ではタイトルを先頭に保つので渡さない)。
    """
    rules = [r for r in ledger.rules]
    dropped: list[str] = []

    def _build(active: list[Rule]) -> str:
        parts: list[str] = []
        hoisted: list[Rule] = []
        body_rules = active
        if hoist_shared:
            hoisted = [r for r in active if r.id in hoist_shared and r.kind == "bullet"]
            body_rules = [r for r in active if r not in hoisted]
            if hoisted:
                # 共通接頭辞: protected ならマーカーで囲む (resync / 検証と整合)
                cats: list[str] = []
                for r in hoisted:
                    if r.category not in cats:
                        cats.append(r.category)
                block = "\n\n".join(_render_category(hoisted, c, ledger.locale) for c in cats)
                if all(r.protected for r in hoisted):
                    block = f"{PROTECTED_OPEN}\n{block}\n{PROTECTED_CLOSE}"
                parts.append(block)
        title = next((r for r in body_rules if r.kind == "title"), None)
        if title is not None:
            parts.append(f"# {title.text}")
        intros = [r for r in body_rules if r.kind == "intro"]
        for r in intros:
            parts.append(r.text)
        cats = _category_order(body_rules)
        protected_cats = [c for c in cats if any(r.category == c and r.protected for r in body_rules)]
        other_cats = [c for c in cats if c not in protected_cats]
        if protected_cats:
            block = "\n\n".join(_render_category(body_rules, c, ledger.locale) for c in protected_cats)
            parts.append(f"{PROTECTED_OPEN}\n{block}\n{PROTECTED_CLOSE}")
        for c in other_cats:
            rendered = _render_category(body_rules, c, ledger.locale)
            if rendered.strip() and "\n- " in rendered + "\n":
                parts.append(rendered)
        # 末尾改行は足さない: 手編集 / テストが本文をそのまま突き合わせる。
        return "\n\n".join(p for p in parts if p)

    text = _build(rules)
    if max_tokens is None:
        return text, dropped
    while estimate_tokens(text) > max_tokens:
        candidates = [r for r in rules if r.kind == "bullet" and not r.protected]
        if not candidates:
            raise ValueError(
                f"protected rules alone exceed the system budget "
                f"({estimate_tokens(text)} > {max_tokens} tokens)",
            )
        victim = min(candidates, key=lambda r: (r.priority, -rules.index(r)))
        rules.remove(victim)
        dropped.append(victim.id)
        text = _build(rules)
    return text, dropped


def shared_bullet_ids(a: Ledger, b: Ledger) -> set[str]:
    """2 つの台帳で本文が一致する箇条の id (両方の id を含む)。"""
    texts_b = {normalize_text(r.text) for r in b.rules if r.kind == "bullet"}
    shared: set[str] = set()
    for r in a.rules:
        if r.kind == "bullet" and normalize_text(r.text) in texts_b:
            shared.add(r.id)
    texts_a = {normalize_text(r.text) for r in a.rules if r.kind == "bullet"}
    for r in b.rules:
        if r.kind == "bullet" and normalize_text(r.text) in texts_a:
            shared.add(r.id)
    return shared


# ──────────────────────────────────────────────────────────────────────────
# protected 同期 / 計数
# ──────────────────────────────────────────────────────────────────────────


#: DEFAULT_PROMPTS の規則 ↔ 検証器 id (f_03 §3.5.1)。キーは規則本文に含まれる
#: 識別しやすい部分文字列 (ja / en)。本文が変わればここも変える — 対応が
#: 切れると計数が付かず、その規則は「関与ゼロ」と誤って刈られる側に寄る。
DEFAULT_RULE_VERIFIERS: dict[str, str] = {
    "ユーザーの発言をそのまま繰り返さない": "query_echo",
    "Do not echo the user's message back verbatim": "query_echo",
    "内部思考・分析過程・推論ステップは出力に含めない": "thinking",
    "Do not include internal thoughts, analysis steps, or reasoning processes": "thinking",
    "等のラベルを使わない": "head_label",
    "Do not use labels such as": "head_label",
    "自体を話題にしない": "internal_frame",
    "自分自身の過去の発言をそのまま繰り返さない": "repetition",
    "Do not repeat your own past reply verbatim": "repetition",
    "今回の会話で述べられた方を採用する": "user_correction",
}


def apply_default_verifiers(ledger: Ledger) -> None:
    """本文の部分一致で ``verifier`` を埋める (既に付いている規則は触らない)。"""
    for rule in ledger.rules:
        if rule.kind != "bullet" or rule.verifier:
            continue
        for needle, verifier in DEFAULT_RULE_VERIFIERS.items():
            if needle in rule.text:
                rule.verifier = verifier
                break


def rule_ids_for_verifiers(ledger: Ledger, verifier_ids: set[str]) -> set[str]:
    """発火した検証器 id の集合を、その規則の id 集合へ写す。"""
    return {
        r.id for r in ledger.rules
        if r.kind == "bullet" and r.verifier and r.verifier in verifier_ids
    }


def sync_protected(ledger: Ledger, default: Ledger) -> Ledger:
    """protected 規則を現行コード (``default``) の集合へ強制同期する。

    旧 ``_resync_protected`` の規則単位版。非 protected 規則は触らない。
    default 側の protected 規則の計数は台帳側の同 id から引き継ぐ。
    """
    counts = {r.id: r for r in ledger.rules if r.protected}
    synced: list[Rule] = []
    for r in default.rules:
        if not r.protected:
            continue
        prior = counts.get(r.id)
        rule = Rule(**asdict(r))
        if prior is not None:
            rule.helpful, rule.harmful, rule.last_fired = prior.helpful, prior.harmful, prior.last_fired
            rule.verifier = rule.verifier or prior.verifier
        synced.append(rule)
    others = [r for r in ledger.rules if not r.protected]
    # 元の位置関係 (title / intro → 非 protected → protected) は render 側が
    # 決めるので、ここでは単に連結する。
    return Ledger(mode=ledger.mode, locale=ledger.locale, rules=others + synced)


def record_rule_outcome(ledger: Ledger, violated_ids: set[str], *, fired_at: str) -> None:
    """ターンの結末を計数へ反映する (f_03 §3.5.1)。

    ``violated_ids`` に入る規則は harmful += 1、それ以外の箇条は helpful += 1。
    """
    for rule in ledger.rules:
        if rule.kind != "bullet":
            continue
        if rule.id in violated_ids:
            rule.harmful += 1
            rule.last_fired = fired_at
        else:
            rule.helpful += 1


# ──────────────────────────────────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────────────────────────────────


def ledger_path(prompt_dir: Path, mode: str) -> Path:
    return prompt_dir / f"{mode}.rules.json"


def load_ledger(prompt_dir: Path, mode: str) -> Ledger | None:
    path = ledger_path(prompt_dir, mode)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        rules = [Rule(**{k: v for k, v in r.items() if k in Rule.__dataclass_fields__}) for r in data.get("rules", [])]
        return Ledger(
            mode=data.get("mode", mode), locale=data.get("locale", "ja"),
            rules=rules, schema_version=int(data.get("schema_version", 1)),
        )
    except (OSError, ValueError, TypeError) as e:
        logger.warning("Failed to load rules ledger %s: %s", path, e)
        return None


def save_ledger(prompt_dir: Path, ledger: Ledger) -> None:
    prompt_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "schema_version": ledger.schema_version,
        "mode": ledger.mode,
        "locale": ledger.locale,
        "rules_hash": ledger.content_hash(),
        "rules": [asdict(r) for r in ledger.rules],
    }
    atomic_write_text(
        ledger_path(prompt_dir, ledger.mode),
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8",
    )
