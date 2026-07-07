"""FlowSpec (フロー構造 JSON) の正規化・検証・決定論レンダリング。

staged spec 工程は purpose="flow_spec_synthesis" でフロー構造 (steps 列) を
1 回だけ合成し、本モジュールが同一 steps から

- flowchart.md の mermaid (:func:`render_mermaid`)
- spec.md の ``## Processing flow`` 節 (:func:`render_flow_section`)

の両方を決定論レンダリングする (f_10 §3)。両成果物が同一データの派生物に
なるため、spec と flowchart のフロー内容の相違は構造的に発生しない。

LLM・I/O は一切使わない (すべて純関数)。LLM 応答の防御的正規化
(:func:`steps_from_payload`)、決定論検証 (:func:`validate_flow`)、検証不能時の
線形フォールバック (:func:`fallback_flow`) もここに集約する。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import PurePosixPath

# フロー全体のステップ数上限 (これを超える応答は検証で棄却する)。
MAX_FLOW_STEPS = 30
# mermaid ノードラベルの上限 (過長ラベルは描画崩れの原因になるため切詰める)。
_LABEL_MAX_CHARS = 80
# mermaid エッジラベル (分岐 condition) の上限。
_CONDITION_MAX_CHARS = 40

_KINDS = frozenset({"start", "process", "decision", "error", "end"})


@dataclass(frozen=True)
class FlowEdge:
    """フローの 1 遷移。``condition`` は分岐ラベル (無条件遷移は空文字)。"""

    to: str
    condition: str = ""


@dataclass(frozen=True)
class FlowStep:
    """フローの 1 ステップ (正規化済み)。``module`` は正準 file_path or 空。"""

    id: str
    label: str
    kind: str = "process"  # start | process | decision | error | end
    module: str = ""
    edges: tuple[FlowEdge, ...] = ()


def _basename(path: str) -> str:
    """OS 非依存の basename (spec_parts と同規約: posix 前提 + backslash 許容)。"""
    return PurePosixPath(path.replace("\\", "/")).name


def steps_from_payload(
    raw: object, canonical_paths: list[str],
) -> list[FlowStep]:
    """``generate_json`` の応答を防御的に :class:`FlowStep` 列へ正規化する。

    grammar 非強制モデル (LFM2 系) の崩れ応答を想定し、非 dict 要素は
    黙ってスキップ、不明 kind は "process" へ落とす。``module`` は
    完全一致 → basename 一意一致の順で正準パスへ解決し、解決不能な値は
    原文のまま残す (:func:`validate_flow` が棄却判断する)。
    """
    if not isinstance(raw, dict):
        return []
    items = raw.get("steps")
    if not isinstance(items, list):
        return []

    by_base: dict[str, list[str]] = {}
    for p in canonical_paths:
        by_base.setdefault(_basename(p), []).append(p)

    steps: list[FlowStep] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        module = str(item.get("module", "") or "").strip()
        if module and module not in canonical_paths:
            candidates = by_base.get(_basename(module), [])
            if len(candidates) == 1:
                module = candidates[0]
        kind = str(item.get("kind", "") or "").strip().lower()
        if kind not in _KINDS:
            kind = "process"
        edges: list[FlowEdge] = []
        raw_next = item.get("next")
        if isinstance(raw_next, list):
            for e in raw_next:
                if not isinstance(e, dict):
                    continue
                to = str(e.get("to", "") or "").strip()
                if not to:
                    continue
                edges.append(FlowEdge(
                    to=to, condition=str(e.get("condition", "") or "").strip(),
                ))
        steps.append(FlowStep(
            id=str(item.get("id", "") or "").strip(),
            label=str(item.get("label", "") or "").strip(),
            kind=kind,
            module=module,
            edges=tuple(edges),
        ))
    return steps


def validate_flow(
    steps: list[FlowStep], canonical_paths: list[str],
) -> list[str]:
    """フロー構造の決定論検証。違反メッセージ列を返す (空 = 合格)。

    メッセージは英語固定 (リトライプロンプトへの注入とログに共用する)。
    循環 (リトライループ等) は正当な制御フローとして許容し、終端の存在
    (出辺 0 のステップ) で停止性を担保する。
    """
    errors: list[str] = []
    if not steps:
        return ["steps is empty"]
    if len(steps) > MAX_FLOW_STEPS:
        errors.append(f"too many steps: {len(steps)} > {MAX_FLOW_STEPS}")

    ids = [s.id for s in steps]
    if any(not i for i in ids):
        errors.append("every step needs a non-empty id")
    dup = sorted({i for i in ids if i and ids.count(i) > 1})
    if dup:
        errors.append(f"duplicate step ids: {', '.join(dup)}")
    id_set = set(ids)

    for s in steps:
        for e in s.edges:
            if e.to not in id_set:
                errors.append(
                    f"step {s.id or '?'}: edge target '{e.to}' does not exist",
                )

    if canonical_paths:
        canonical = set(canonical_paths)
        for s in steps:
            if s.module and s.module not in canonical:
                errors.append(
                    f"step {s.id or '?'}: module '{s.module}' is not in the "
                    f"canonical file list",
                )
        covered = {s.module for s in steps if s.module}
        for path in canonical_paths:
            if path not in covered:
                errors.append(f"canonical module '{path}' owns no step")

    starts = [s for s in steps if s.kind == "start"]
    if len(starts) != 1:
        errors.append(f"exactly one start step required (found {len(starts)})")
    elif not starts[0].edges:
        errors.append("start step has no outgoing edge")
    if not any(not s.edges for s in steps):
        errors.append("at least one terminal step (no outgoing edges) required")

    for s in steps:
        if s.kind == "end" and s.edges:
            errors.append(f"end step {s.id or '?'} must not have outgoing edges")
        if s.kind == "decision" and len(s.edges) < 2:
            errors.append(
                f"decision step {s.id or '?'} needs at least 2 outgoing edges",
            )
        if len(s.edges) >= 2:
            conds = [e.condition for e in s.edges]
            if any(not c for c in conds):
                errors.append(
                    f"step {s.id or '?'}: every branch edge needs a non-empty "
                    f"condition",
                )
            if len(set(conds)) != len(conds):
                errors.append(
                    f"step {s.id or '?'}: branch conditions must be distinct",
                )

    # 到達性 (BFS)。前提が壊れている (start 不定・id 重複/空) 場合はスキップ
    # (上のエラーで既に棄却される)。
    seen = _reachable_ids(steps)
    if seen is not None:
        unreachable = [i for i in ids if i not in seen]
        if unreachable:
            errors.append(
                f"steps not reachable from start: {', '.join(unreachable)}",
            )
    return errors


def _reachable_ids(steps: list[FlowStep]) -> set[str] | None:
    """start から BFS で到達可能な id 集合を返す。

    前提 (start が厳密に 1 個・id 重複無し・id 非空) が崩れている場合は
    判定不能として ``None`` を返す (呼出側は他の検証エラーに委ねる)。
    :func:`validate_flow` と :func:`repair_unreachable_tail` で共有する。
    """
    starts = [s for s in steps if s.kind == "start"]
    ids = [s.id for s in steps]
    if len(starts) != 1 or len(set(ids)) != len(ids) or not all(ids):
        return None
    id_set = set(ids)
    adjacency = {s.id: [e.to for e in s.edges] for s in steps}
    seen = {starts[0].id}
    queue = [starts[0].id]
    while queue:
        for nxt in adjacency.get(queue.pop(), []):
            if nxt in id_set and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen


def repair_unreachable_tail(steps: list[FlowStep]) -> list[FlowStep]:
    """到達不能ステップの決定論サルベージ (repair-not-reject)。

    ``validate_flow`` の "steps not reachable from start" 違反を狙った最小限の
    決定論修復。リトライで解消できる「局所的な穴 (condition 空文字等)」とは
    異なり、大域的な配線ミス (末尾ブロックの本線からの孤立) はリトライでも
    解消しない実測がある (2026-07-07 live: 到達不能違反はリトライ後も残存)。
    ``steps`` のリスト順 (プロンプトが定義する実行順) を根拠に機械的に架橋する。

    対象パターンを狭く限定する: ``steps[i]`` が到達済み・出辺 0 本 (偽終端)・
    ``kind`` が "end"/"error" でない (＝真の終端ではない) かつ、直後の
    ``steps[i+1]`` が未到達の場合のみ、``steps[i]`` から ``steps[i+1]`` への
    無条件エッジを追加する。橋渡し後は新たに到達可能になったステップから
    BFS を延長し、後続の複数ギャップにも同一パスで対応する。パターンに合致
    しない (偽終端が見つからない等) 場合は一切変更しない — 誤配線で意味不明な
    図を作るくらいなら「直さない」側に倒す安全側デフォルト。

    呼出側は必ず修復後に :func:`validate_flow` を再実行し、合格した場合のみ
    採用すること (本関数はヒューリスティックであり正当性の証明ではない)。
    """
    seen = _reachable_ids(steps)
    if seen is None or len(seen) == len(steps):
        return steps

    id_set = {s.id for s in steps}
    adjacency = {s.id: [e.to for e in s.edges] for s in steps}

    def _extend(start_id: str) -> None:
        queue = [start_id]
        seen.add(start_id)
        while queue:
            for nxt in adjacency.get(queue.pop(), []):
                if nxt in id_set and nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)

    repaired: list[FlowStep] = []
    changed = False
    for i, step in enumerate(steps):
        if (
            step.id in seen
            and not step.edges
            and step.kind not in ("end", "error")
            and i + 1 < len(steps)
            and steps[i + 1].id not in seen
        ):
            step = replace(step, edges=(FlowEdge(to=steps[i + 1].id),))
            _extend(steps[i + 1].id)
            changed = True
        repaired.append(step)
    return repaired if changed else steps


def prefix_unit_steps(steps: list[FlowStep], prefix: str) -> list[FlowStep]:
    """1 ユニット分の steps の id/edge 参照へユニット固有の接頭辞を付与する。

    複数ユニットを :func:`assemble_flow_parts` で結合する前に、ユニット間の
    id 衝突を避けるために各ユニットへ適用する (``prefix`` は ``"U1_"`` 等)。
    ユニット外への参照 (存在しない to) はそのまま素通しし、結合後の
    :func:`validate_flow` が「edge target が存在しない」として検出する。
    """
    ids = {s.id for s in steps}
    return [
        replace(
            s,
            id=f"{prefix}{s.id}",
            edges=tuple(
                FlowEdge(
                    to=f"{prefix}{e.to}" if e.to in ids else e.to,
                    condition=e.condition,
                )
                for e in s.edges
            ),
        )
        for s in steps
    ]


def assemble_flow_parts(units: list[list[FlowStep]]) -> list[FlowStep] | None:
    """ユニット毎のローカル部分グラフ列を 1 つの FlowSpec へ決定論結合する。

    各ユニットの ``steps[0]`` がそのユニットの入口 (前フェーズからの遷移先)
    とみなす。各ユニットの「``kind`` が end/error でなく ``edges`` が空」な
    ステップ (次フェーズへの暗黙継続点) を次ユニットの入口へ無条件エッジで
    接続する (複数あれば全て接続 — 分岐両経路が同じ次フェーズへ合流するのは
    正当な構造)。先頭に合成 start、末尾ユニットの継続点に合成 end を付与する。
    id 衝突・空ユニットなど構造不能時は ``None`` (呼出側は決定論フォール
    バックへ委譲する)。呼出側は結合後、必ず :func:`validate_flow` を実行
    すること (本関数はグラフを構築するだけで正当性を保証しない)。
    """
    if not units or any(not u for u in units):
        return None
    all_ids = [s.id for unit in units for s in unit]
    if len(set(all_ids)) != len(all_ids):
        return None

    entries = [unit[0].id for unit in units]
    start_id, end_id = "_flow_start", "_flow_end"
    if start_id in all_ids or end_id in all_ids:
        return None

    result: list[FlowStep] = [
        FlowStep(id=start_id, label="Start", kind="start",
                 edges=(FlowEdge(to=entries[0]),)),
    ]
    for i, unit in enumerate(units):
        next_entry = entries[i + 1] if i + 1 < len(units) else end_id
        for step in unit:
            if not step.edges and step.kind not in ("end", "error"):
                step = replace(step, edges=(FlowEdge(to=next_entry),))
            result.append(step)
    result.append(FlowStep(id=end_id, label="End", kind="end"))
    return result


def renumber_steps(steps: list[FlowStep]) -> list[FlowStep]:
    """steps 順に S1..Sn へ再採番し、edges の参照も追随させる。

    mermaid ノード id と ``## Processing flow`` 節のステップ番号を 1:1 に
    固定するため、検証合格後に必ず通す (id 一意が前提)。
    """
    mapping = {s.id: f"S{i}" for i, s in enumerate(steps, start=1)}
    return [
        replace(
            s,
            id=mapping[s.id],
            edges=tuple(
                FlowEdge(to=mapping.get(e.to, e.to), condition=e.condition)
                for e in s.edges
            ),
        )
        for s in steps
    ]


def fallback_flow(module_entries: list[tuple[str, list[str]]]) -> list[FlowStep]:
    """正準モジュール一覧から線形フローを決定論構築する (LLM 不使用)。

    合成が 2 回とも検証不能だった場合・assist degraded 時の縮退経路。
    ``module_entries`` は ``(path, labels)`` — labels は spec の Behavior
    番号ステップ (取得できれば、そちらが優先) または purpose 先頭行 1 個の
    いずれか (呼出側が決定する。ラベル 0 個のモジュールは無視される)。
    Start → 各モジュールの labels 順に 1 process/step → End の直列で、
    :func:`validate_flow` を構築的に満たす。総 step 数の上限
    (:data:`MAX_FLOW_STEPS`) は必ず守る (呼出側は再検証しない前提のため
    ハード制約) — モジュール毎の予算を均等割り (端数は先頭側、最低 1) して
    モジュール順・ラベル順に詰め、モジュール網羅と読み順の連続性を両立する。
    モジュール数自体が上限を超える極端なケースのみ後方モジュールを打ち切る。
    """
    entries = [(path, labels) for path, labels in module_entries if labels]
    if not entries:
        return []
    body_cap = MAX_FLOW_STEPS - 2
    if len(entries) > body_cap:
        entries = entries[:body_cap]
    base, extra_n = divmod(body_cap, len(entries))
    flat: list[tuple[str, str]] = []
    for i, (path, labels) in enumerate(entries):
        quota = max(base + (1 if i < extra_n else 0), 1)
        flat.extend((path, label) for label in labels[:quota])
    if len(flat) > body_cap:
        flat = flat[:body_cap]
    steps = [FlowStep(
        id="S1", label="Start", kind="start", edges=(FlowEdge(to="S2"),),
    )]
    n = 2
    for path, label in flat:
        steps.append(FlowStep(
            id=f"S{n}", label=label, module=path,
            edges=(FlowEdge(to=f"S{n + 1}"),),
        ))
        n += 1
    steps.append(FlowStep(id=f"S{n}", label="End", kind="end"))
    return steps


def _sanitize_label(text: str, limit: int = _LABEL_MAX_CHARS) -> str:
    """mermaid ラベルの決定論サニタイズ (空白圧縮・引用/パイプ退避・切詰め)。"""
    collapsed = " ".join(text.split()) or "-"
    collapsed = collapsed.replace('"', "#quot;").replace("|", "/")
    if len(collapsed) > limit:
        collapsed = collapsed[: limit - 1] + "…"
    return collapsed


def _node_decl(step: FlowStep, label: str) -> str:
    """kind 別のノード宣言 (ラベルは常時二重引用)。"""
    match step.kind:
        case "start" | "end":
            return f'{step.id}(["{label}"])'
        case "decision":
            return f'{step.id}{{"{label}"}}'
        case "error":
            return f'{step.id}[/"{label}"/]'
        case _:
            return f'{step.id}["{label}"]'


def render_mermaid(steps: list[FlowStep], canonical_paths: list[str]) -> str:
    """steps から mermaid (``flowchart TD``) を決定論レンダリングする。

    生の mermaid 文字列を返す (フェンス無し。配信側が ```mermaid で包むため
    二重フェンスにしない)。正準モジュールごとに subgraph へ束ね、module 無し
    (start/end 等) のステップは直下に置く。error ステップには classDef で
    強調を付ける。
    """
    lines = ["flowchart TD"]
    by_module: dict[str, list[FlowStep]] = {}
    loose: list[FlowStep] = []
    canonical = set(canonical_paths)
    for s in steps:
        if s.module and s.module in canonical:
            by_module.setdefault(s.module, []).append(s)
        else:
            loose.append(s)

    for s in loose:
        # subgraph 外に出るステップは module をラベルへ畳み込む (正準外 module
        # は validate で棄却済みだが、canonical 空のときも情報を落とさない)。
        text = f"{s.module}: {s.label}" if s.module else s.label
        lines.append(f"    {_node_decl(s, _sanitize_label(text))}")
    for k, path in enumerate(canonical_paths, start=1):
        group = by_module.get(path)
        if not group:
            continue
        lines.append(f'    subgraph M{k}["{_sanitize_label(path)}"]')
        for s in group:
            lines.append(f"        {_node_decl(s, _sanitize_label(s.label))}")
        lines.append("    end")

    for s in steps:
        for e in s.edges:
            if e.condition:
                cond = _sanitize_label(e.condition, _CONDITION_MAX_CHARS)
                lines.append(f'    {s.id} -->|"{cond}"| {e.to}')
            else:
                lines.append(f"    {s.id} --> {e.to}")

    error_ids = [s.id for s in steps if s.kind == "error"]
    if error_ids:
        lines.append("    classDef flowError stroke:#cc3333,stroke-width:2px")
        lines.append(f"    class {','.join(error_ids)} flowError")
    return "\n".join(lines) + "\n"


def render_flow_section(steps: list[FlowStep]) -> str:
    """steps から spec.md の ``## Processing flow`` 節を決定論レンダリングする。

    行形式は ``<n>. [<module>] <label>  {<kind>}`` (process の kind タグは
    省略)。次ステップ (n+1) への無条件遷移は暗黙、それ以外の単一遷移は
    ``-> <m>``、分岐はインデントした ``- <condition> -> <m>`` 弾で表す。
    ステップ番号 n は mermaid ノード id ``S<n>`` と 1:1 対応する
    (:func:`renumber_steps` 済み前提)。
    """
    number = {s.id: i for i, s in enumerate(steps, start=1)}
    lines = ["## Processing flow", ""]
    for i, s in enumerate(steps, start=1):
        label = " ".join(s.label.split()) or "-"
        head = f"{i}. " + (f"[{s.module}] " if s.module else "") + label
        if s.kind != "process":
            head += f"  {{{s.kind}}}"
        edges = list(s.edges)
        if len(edges) == 1 and not edges[0].condition:
            target = number.get(edges[0].to)
            if target is None:
                head += f" -> {edges[0].to}"
            elif target != i + 1:
                head += f" -> {target}"
            lines.append(head)
            continue
        lines.append(head)
        for e in edges:
            cond = " ".join(e.condition.split()) or "-"
            lines.append(f"   - {cond} -> {number.get(e.to, e.to)}")
    return "\n".join(lines) + "\n"
