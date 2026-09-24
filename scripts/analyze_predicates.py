"""判定点の発火ビットマップ・グラフ — 重複 / 恒真 / 語形の穴を機械的に見つける。

このリポジトリで最も再発している 3 つの欠陥は、どれも **1 つの計算** から
落ちてくる:

1. **同じ判定の二重実装** — 「訂正か」を 5 系統、日本語文字の判定を 4 実装
   (うち 3 つは文字範囲が違う) が別々に持っている。片方だけ直すと不整合が残る。
2. **恒真 / 恒偽の述語** — 発火率 0% か 100% の判定は何も判定していない。
   過去に ``tool_grounded`` が全ターン真で few-shot が 0 件になり、Level 1 の
   恒真フェーズが空回りし、fitness が恒真で乱歩した。
3. **語形の穴** — 意味的に近い発話なのに発火が割れているペアが、次に事例へ
   足すべき入力。

やることは単純で、**述語 × 発話** のビットマップを作り、行どうしの Jaccard と
包含を見るだけ。LLM も埋め込みサーバも要らない (近傍は既存の文字 bi-gram
コサインで代用する)。

使い方::

    python scripts/analyze_predicates.py
    python scripts/analyze_predicates.py --texts E:\\tmp\\live_audit\\run2\\results.jsonl
    python scripts/analyze_predicates.py --jaccard 0.9 --json out.json
    python scripts/analyze_predicates.py --holes --hole-sim 0.75

発話プールの既定はデータ根の ``store/learning/*/experience.jsonl`` の ``query``。監査の
results.jsonl / 任意の JSONL (``query`` か ``text`` キー) / 1 行 1 発話のテキスト
ファイルも ``--texts`` で足せる。
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import json
import sys
import typing
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import numpy as np

# Windows の既定コンソール encoding (cp932) では和文の判定名が化ける。
if hasattr(sys.stdout, "reconfigure"):  # pragma: no cover - 実行環境依存
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.free.core.predicate import (  # noqa: E402
    FiringMatrix,
    registry,
)

#: 既定で走査するモジュール。**字句リソースの集積地**から選んである
#: (棚卸しの内訳表で上位のもの)。import に失敗したものは黙って飛ばす
#: (config 依存で単体 import できないモジュールがある)。
DEFAULT_MODULES: tuple[str, ...] = (
    "backend.free.core.intent_vocab",
    "backend.free.core.text_quality",
    "backend.free.core.date_math_cue",
    "backend.free.core.correction_target",
    "backend.free.core.correction_verdict",
    "backend.free.core.query_anchors",
    "backend.free.core.locale_patterns",
    "backend.free.core.response_dates",
    "backend.free.core.relative_date",
    "backend.free.agent.tool_judge_signals",
    "backend.free.agent.tool_judge_history",
    "backend.free.agent.tool_judge_dialogue",
    "backend.free.agent.tool_judge_grounding",
    "backend.free.agent.tool_judge_commands",
    "backend.free.agent.meta_cognitive_content_gate",
    "backend.free.agent.meta_cognitive_scaffold",
    "backend.free.agent.feedback",
    "backend.free.learning.corrected_pairs",
    "backend.free.memory.volatile_values",
    "backend.free.memory.attribute_key",
    "backend.free.memory.notes.note_builder",
    "backend.free.rag.self_rag_judge",
)

#: 単一引数の名前として受け入れるもの (``text`` 以外の呼び方をする述語がある)。
_TEXT_PARAM_NAMES = frozenset(
    {"text", "query", "s", "value", "utterance", "body", "response", "content"},
)

#: 真偽を返す述語として受け入れる戻り値注釈。
_BOOL_RETURNS = frozenset({"bool", "bool | None", "Optional[bool]"})
#: ``--include-optional`` で足す戻り値注釈 (span 抽出器は真偽で読む)。
_OPTIONAL_RETURNS = frozenset(
    {"str | None", "Optional[str]", "date | None", "int | None"},
)


def _return_annotation(fn: Callable) -> str:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return ""
    ann = sig.return_annotation
    if ann is inspect.Signature.empty:
        return ""
    if isinstance(ann, str):
        return ann
    if ann is bool:
        return "bool"
    return str(ann)


def _takes_single_text(fn: Callable) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    params = list(sig.parameters.values())
    if not params:
        return False
    required = [
        p for p in params
        if p.default is inspect.Parameter.empty
        and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    if len(required) != 1:
        return False
    first = required[0]
    if first.name not in _TEXT_PARAM_NAMES:
        ann = first.annotation
        if not (ann is str or (isinstance(ann, str) and ann.startswith("str"))):
            return False
    return True


def collect_predicates(
    modules: Sequence[str],
    *,
    include_optional: bool = False,
) -> list[tuple[str, Callable[[str], object]]]:
    """モジュールを走査して ``(text) -> bool`` の述語を集める。

    レジストリ (:data:`backend.free.core.predicate.registry`) に登録済みの判定点も
    併せて返す。登録の有無で扱いを変えないのは、**移行の途中でも解析が効く**
    ようにするため — 340 点を一度に登録し直す必要は無い。
    """
    found: list[tuple[str, Callable[[str], object]]] = []
    seen: set[int] = set()
    accepted = set(_BOOL_RETURNS) | (set(_OPTIONAL_RETURNS) if include_optional else set())

    for mod_name in modules:
        try:
            mod = importlib.import_module(mod_name)
        except Exception as e:
            print(f"[skip] {mod_name}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        short = mod_name.rsplit(".", 1)[-1]
        for attr, obj in vars(mod).items():
            if attr.startswith("_") or not inspect.isfunction(obj):
                continue
            if getattr(obj, "__module__", "") != mod_name:
                continue  # re-export は元のモジュールで数える
            if id(obj) in seen:
                continue
            if _return_annotation(obj) not in accepted:
                continue
            if not _takes_single_text(obj):
                continue
            seen.add(id(obj))
            found.append((f"{short}.{attr}", obj))

    for predicate in registry:
        name = getattr(predicate, "name", "")
        evaluate = getattr(predicate, "evaluate", None)
        if not name or evaluate is None:
            continue
        found.append((f"registry.{name}", lambda t, _e=evaluate: _e(t).fired))
    found.sort(key=lambda kv: kv[0])
    return found


def collect_exemplar_gates(embedder: Any) -> dict[str, Any]:
    """``ExemplarPredicate`` を持つゲート (非同期判定点) を集める。

    字句述語と違い、これらは埋め込みが要るので既定では走らない
    (``--embed`` で有効化)。発火ビットマップへ載ると、**字句の述語と同じ土俵で
    重複・包含・恒真が見える** — 例えば「事例ゲートが結局ある正規表現と同じ
    発話でしか発火していない」ことが検出できる。
    """
    from backend.free.agent.layer_shadow import LayerClassificationShadow
    from backend.free.agent.write_intent_gate import WriteIntentGate
    from backend.free.learning.context_bound_gate import ContextBoundGate
    from backend.free.memory.notes.attribute_gate import AttributeSlotGate
    from backend.free.rag.retrieval_skip_gate import RetrievalSkipGate

    # **ここに載せ忘れるとその判定点は解析から消える** (静かに。エラーにならない)。
    # 網羅性は backend/tests/test_predicate_analyzer_coverage.py が固定する。
    return {
        "gate.fact_attribute_slot": AttributeSlotGate(embedder),
        "gate.retrieval_skip": RetrievalSkipGate(embedder),
        "gate.layer_classification_shadow": LayerClassificationShadow(embedder),
        "gate.local_write_intent": WriteIntentGate(embedder),
        "gate.fewshot_context_bound": ContextBoundGate(embedder),
    }


async def _async_gate_rows(
    gates: dict[str, Any], texts: Sequence[str], embedder: Any,
) -> dict[str, np.ndarray]:
    """発話プールを 1 度だけ埋め込み、各ゲートの発火行を作る。

    ``ExemplarPredicate.aevaluate`` に ``query_vec`` を渡すので、ゲートの数だけ
    埋め込みし直すことはない (N 件の batch embed が 1 回だけ)。
    """
    vecs = await embedder.embed(list(texts), is_query=True, mode="chat")
    vecs = np.asarray(vecs, dtype=np.float32)
    rows: dict[str, np.ndarray] = {}
    for name, gate in gates.items():
        if not await gate.warmup():
            print(f"[skip] {name}: warmup 失敗", file=sys.stderr)
            continue
        predicate = gate._exemplar
        row = np.zeros(len(texts), dtype=np.int8)
        for j, text in enumerate(texts):
            verdict = await predicate.aevaluate(text, query_vec=vecs[j])
            row[j] = 1 if verdict.band == "fire" else 0
        rows[name] = row
    return rows


def _iter_json_records(path: Path) -> Iterable[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".jsonl":
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except (ValueError, TypeError):
                continue
            if isinstance(rec, dict):
                yield rec
        return
    try:
        doc = json.loads(text)
    except (ValueError, TypeError):
        return
    payload = doc.get("payload") if isinstance(doc, dict) else doc
    if isinstance(payload, list):
        for rec in payload:
            if isinstance(rec, dict):
                yield rec
        return
    # 会話履歴 (<data_root>/store/history/<month>/<session>.json) は封筒の payload に turns[] を持つ。
    session = payload if isinstance(payload, dict) else doc
    if isinstance(session, dict) and isinstance(session.get("turns"), list):
        for turn in session["turns"]:
            if isinstance(turn, dict) and turn.get("role") == "user":
                yield turn


#: 発話を取り出すキーの既定 (ユーザー側)。``--key`` で差し替える。
DEFAULT_TEXT_KEYS: tuple[str, ...] = (
    "query", "text", "utterance", "user_message", "message", "content",
)


def load_utterances(
    paths: Sequence[Path],
    *,
    limit: int = 0,
    keys: Sequence[str] = DEFAULT_TEXT_KEYS,
) -> list[str]:
    """発話プールを読む。重複は落とし、出現順を保つ。

    ``keys`` を ``("response", "response_full")`` にすると **応答側** のプールに
    なる。``text_quality.*`` のような出力検査の述語はユーザー発話に対しては
    恒偽に見えるので、その群を測るときはプールを切り替える。
    """
    out: list[str] = []
    seen: set[str] = set()
    for path in paths:
        if not path.exists():
            print(f"[skip] not found: {path}", file=sys.stderr)
            continue
        if path.suffix in (".json", ".jsonl"):
            for rec in _iter_json_records(path):
                for key in keys:
                    val = rec.get(key)
                    if isinstance(val, str) and val.strip():
                        norm = val.strip()
                        if norm not in seen:
                            seen.add(norm)
                            out.append(norm)
                        break
        else:
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                norm = line.strip()
                if norm and norm not in seen:
                    seen.add(norm)
                    out.append(norm)
        if limit and len(out) >= limit:
            break
    return out[:limit] if limit else out


def default_text_sources(data_root: str | None = None) -> list[Path]:
    """既定の発話プール: 経験バッファと会話履歴 (どちらもデータ根の実データ)。

    経験バッファ (``experience.jsonl``) は保持方針で数十件に絞られるため、
    ``<data_root>/store/history/<month>/*.json`` の user ターンも足す。監査の
    ``results.jsonl`` を混ぜたいときは ``--texts`` で明示する。データ根は
    ``--data-root`` → ``EVOREF_DATA_ROOT`` → ``<repo>/userdata``。
    """
    from backend.config import PathResolver
    from backend.data_root import resolve_data_root

    resolver = PathResolver({}, REPO_ROOT, data_root=resolve_data_root(data_root, root=REPO_ROOT))
    out: list[Path] = []
    learning = resolver.resolve_local("learning_dir")
    if learning.exists():
        out.extend(sorted(learning.glob("*/experience.jsonl")))
    history = resolver.resolve_local("history_dir")
    if history.exists():
        out.extend(sorted(history.glob("*/*.json")))
    return out


def build_matrix(
    predicates: Sequence[tuple[str, Callable[[str], object]]],
    texts: Sequence[str],
) -> tuple[FiringMatrix, dict[str, int]]:
    """述語 × 発話のビットマップを作る。例外は棄権として数える。"""
    names = [name for name, _ in predicates]
    bits = np.zeros((len(predicates), len(texts)), dtype=np.int8)
    errors: dict[str, int] = {}
    for i, (name, fn) in enumerate(predicates):
        n_err = 0
        for j, text in enumerate(texts):
            try:
                bits[i, j] = 1 if fn(text) else 0
            except Exception:
                n_err += 1
        if n_err:
            errors[name] = n_err
    return FiringMatrix(names=names, bits=bits, n_texts=len(texts)), errors


def _informative_rows(matrix: FiringMatrix, max_fire_ratio: float) -> np.ndarray:
    """判別力のある行だけを残すマスク。

    発火率が 0 か、``max_fire_ratio`` を超える述語は「何も判定していない」ので
    重複・包含のレポートから外す (恒真・恒偽として別に報告する)。混ぜると
    ``has_japanese_script`` のような全発火の述語が全組と包含関係を作り、
    本物の重複が埋もれる。
    """
    counts = matrix.bits.sum(axis=1)
    cap = max(1, int(matrix.n_texts * max_fire_ratio))
    return (counts > 0) & (counts <= cap)


def find_duplicate_clusters(
    matrix: FiringMatrix, *, threshold: float, max_fire_ratio: float = 0.95,
) -> list[list[str]]:
    """Jaccard が閾値以上の述語を連結成分でまとめる (= 二重実装の候補)。"""
    jac = matrix.jaccard()
    counts = matrix.bits.sum(axis=1)
    keep = _informative_rows(matrix, max_fire_ratio)
    counts = np.where(keep, counts, 0)
    n = len(matrix.names)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        if counts[i] == 0:
            continue
        for j in range(i + 1, n):
            if counts[j] == 0:
                continue
            if jac[i, j] >= threshold:
                parent[find(i)] = find(j)
    groups: dict[int, list[str]] = {}
    for i in range(n):
        if counts[i] == 0:
            continue
        groups.setdefault(find(i), []).append(matrix.names[i])
    return sorted(
        (sorted(g) for g in groups.values() if len(g) > 1),
        key=lambda g: (-len(g), g[0]),
    )


def find_containments(
    matrix: FiringMatrix,
    *,
    threshold: float,
    min_gap: int = 1,
    max_fire_ratio: float = 0.95,
) -> list[tuple[str, str, int, int]]:
    """``A ⊆ B`` で発火数に差がある組 (A が冗長かもしれない)。"""
    cont = matrix.containment()
    counts = matrix.bits.sum(axis=1)
    counts = np.where(_informative_rows(matrix, max_fire_ratio), counts, 0)
    out: list[tuple[str, str, int, int]] = []
    for i, name_i in enumerate(matrix.names):
        if counts[i] == 0:
            continue
        for j, name_j in enumerate(matrix.names):
            if i == j or counts[j] == 0:
                continue
            if cont[i, j] >= threshold and counts[j] - counts[i] >= min_gap:
                out.append((name_i, name_j, int(counts[i]), int(counts[j])))
    return sorted(out, key=lambda t: (-t[2], t[0]))


def find_degenerate(matrix: FiringMatrix) -> tuple[list[str], list[str]]:
    """恒偽 (発火 0) と恒真 (全発火) の述語。"""
    counts = matrix.bits.sum(axis=1)
    never = [n for n, c in zip(matrix.names, counts, strict=True) if c == 0]
    always = [
        n for n, c in zip(matrix.names, counts, strict=True)
        if matrix.n_texts and c == matrix.n_texts
    ]
    return never, always


def find_holes(
    matrix: FiringMatrix,
    texts: Sequence[str],
    *,
    sim_threshold: float,
    max_per_predicate: int = 3,
) -> list[tuple[str, str, str, float]]:
    """意味的に近いのに発火が割れている発話ペア (= 次に足すべき事例)。

    近傍は文字 bi-gram コサインで測る。埋め込みサーバを起こさずに済み、
    ``fewshot_pool`` が密ベクトル不在時に使っているのと同じ代用手段。
    """
    from backend.free.core.text_similarity import char_bigrams, counter_cosine

    grams = [char_bigrams(t) for t in texts]
    out: list[tuple[str, str, str, float]] = []
    for i, name in enumerate(matrix.names):
        row = matrix.bits[i]
        fired = np.flatnonzero(row == 1)
        missed = np.flatnonzero(row == 0)
        if fired.size == 0 or missed.size == 0:
            continue
        found = 0
        for a in fired:
            if found >= max_per_predicate:
                break
            best_sim, best_b = 0.0, -1
            for b in missed:
                sim = counter_cosine(grams[int(a)], grams[int(b)])
                if sim > best_sim:
                    best_sim, best_b = sim, int(b)
            if best_b >= 0 and best_sim >= sim_threshold:
                out.append((name, texts[int(a)], texts[best_b], round(best_sim, 3)))
                found += 1
    return sorted(out, key=lambda t: -t[3])


def _append_gate_rows(
    matrix: FiringMatrix, texts: Sequence[str], port: int, dim: int,
) -> FiringMatrix:
    """事例ゲートの行をビットマップへ足す (埋め込みサーバが要る)。"""
    from backend.free.rag.embedding_llamacpp import LlamaCppEmbedder

    embedder = LlamaCppEmbedder(port=port, dim_size=dim)

    async def run() -> dict[str, np.ndarray]:
        if not await embedder.health_check():
            print(
                f"[skip] 埋め込みサーバ (:{port}) が応答しません。"
                "--embed を外すか、サーバを起動してください。",
                file=sys.stderr,
            )
            return {}
        return await _async_gate_rows(collect_exemplar_gates(embedder), texts, embedder)

    rows = asyncio.run(run())
    if not rows:
        return matrix
    names = list(matrix.names) + list(rows)
    bits = np.vstack([matrix.bits] + [rows[n] for n in rows])
    return FiringMatrix(names=names, bits=bits, n_texts=matrix.n_texts)


def _fmt(text: str, width: int = 46) -> str:
    one = " ".join(text.split())
    return one if len(one) <= width else one[: width - 1] + "…"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--texts", action="append", default=[],
        help="発話プールに足すファイル (json / jsonl / 1 行 1 発話)。複数指定可",
    )
    ap.add_argument("--limit", type=int, default=0, help="発話の上限 (0 = 無制限)")
    ap.add_argument(
        "--key", action="append", default=[],
        help=(
            "発話を取り出す JSON キー (既定: query/text/message/content)。"
            "出力検査の述語を測るときは --key response を指定する"
        ),
    )
    ap.add_argument(
        "--modules", action="append", default=[],
        help="走査するモジュール (既定は DEFAULT_MODULES)",
    )
    ap.add_argument(
        "--include-optional", action="store_true",
        help="``str | None`` 等を返す span 抽出器も真偽として含める",
    )
    ap.add_argument("--jaccard", type=float, default=0.95, help="重複判定の閾値")
    ap.add_argument(
        "--max-fire-ratio", type=float, default=0.95,
        help="これを超える発火率の述語は重複・包含のレポートから外す",
    )
    ap.add_argument("--containment", type=float, default=0.99, help="包含判定の閾値")
    ap.add_argument(
        "--embed", action="store_true",
        help="事例ゲート (非同期判定点) も測る。埋め込みサーバ (:8082) が要る",
    )
    ap.add_argument("--port", type=int, default=8082, help="埋め込みサーバのポート")
    ap.add_argument("--dim", type=int, default=1024, help="埋め込み次元")
    ap.add_argument("--holes", action="store_true", help="語形の穴の候補も出す")
    ap.add_argument("--hole-sim", type=float, default=0.8, help="穴判定の類似度")
    ap.add_argument("--json", type=Path, default=None, help="結果を JSON で保存")
    ap.add_argument(
        "--data-root", default=None,
        help="既定の発話プールを読むデータ根 (既定: EVOREF_DATA_ROOT → <repo>/userdata)",
    )
    args = ap.parse_args(argv)

    sources = [Path(p) for p in args.texts] or default_text_sources(args.data_root)
    keys = tuple(args.key) if args.key else DEFAULT_TEXT_KEYS
    texts = load_utterances(sources, limit=args.limit, keys=keys)
    if not texts:
        print(
            "発話プールが空です。--texts で監査ログを指定するか、"
            "データ根の store/learning/*/experience.jsonl を用意してください。",
            file=sys.stderr,
        )
        return 2

    modules = args.modules or list(DEFAULT_MODULES)
    predicates = collect_predicates(modules, include_optional=args.include_optional)
    if not predicates:
        print("述語が 1 つも集まりませんでした。", file=sys.stderr)
        return 2

    matrix, errors = build_matrix(predicates, texts)
    if args.embed:
        matrix = _append_gate_rows(matrix, texts, args.port, args.dim)
    counts = matrix.bits.sum(axis=1)
    never, always = find_degenerate(matrix)
    clusters = find_duplicate_clusters(
        matrix, threshold=args.jaccard, max_fire_ratio=args.max_fire_ratio,
    )
    contained = find_containments(
        matrix, threshold=args.containment, max_fire_ratio=args.max_fire_ratio,
    )

    print(f"述語 {len(predicates)} × 発話 {len(texts)} (keys={'/'.join(keys)})")
    print(
        f"発火率: 中央値 {float(np.median(counts)) / len(texts):.3f} / "
        f"平均 {float(counts.mean()) / len(texts):.3f}",
    )
    print()

    print(
        "※ 恒真 / 恒偽には **入力ドメインが違う述語** が混ざる "
        "(応答本文やコマンド文字列を取るものは、ユーザー発話のプールでは "
        "常に同じ答えになる)。その群は --key response で測り直すこと。",
    )
    print()
    print(f"■ 恒偽 (1 度も発火しない): {len(never)}")
    for name in never[:30]:
        print(f"   {name}")
    if len(never) > 30:
        print(f"   … 他 {len(never) - 30} 件")
    print()

    near_tautology = [
        (n, int(c)) for n, c in zip(matrix.names, counts, strict=True)
        if 0 < c < matrix.n_texts and c > matrix.n_texts * args.max_fire_ratio
    ]
    print(f"■ 恒真 (全発話で発火): {len(always)}")
    for name in always:
        print(f"   {name}")
    for name, c in near_tautology:
        print(f"   {name} (準恒真 {c}/{matrix.n_texts})")
    print()

    print(f"■ 重複クラスタ (Jaccard >= {args.jaccard}): {len(clusters)}")
    for group in clusters:
        idx = matrix.names.index(group[0])
        print(f"   [{int(counts[idx])} 発火] {' == '.join(group)}")
    print()

    print(f"■ 包含 A ⊆ B (>= {args.containment}): {len(contained)}")
    for a, b, ca, cb in contained[:40]:
        print(f"   {a} ({ca}) ⊆ {b} ({cb})")
    if len(contained) > 40:
        print(f"   … 他 {len(contained) - 40} 件")
    print()

    holes: list[tuple[str, str, str, float]] = []
    if args.holes:
        holes = find_holes(matrix, texts, sim_threshold=args.hole_sim)
        print(f"■ 語形の穴の候補 (bi-gram cos >= {args.hole_sim}): {len(holes)}")
        for name, fired, missed, sim in holes[:40]:
            print(f"   {name} (cos {sim})")
            print(f"      発火: {_fmt(fired)}")
            print(f"      不発: {_fmt(missed)}")
        print()

    if errors:
        print(f"■ 例外を投げた述語: {len(errors)}")
        for name, n in sorted(errors.items(), key=lambda kv: -kv[1])[:20]:
            print(f"   {name}: {n} 件")
        print()

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "n_predicates": len(predicates),
                    "n_texts": len(texts),
                    "never_fires": never,
                    "always_fires": always,
                    "duplicate_clusters": clusters,
                    "containments": [
                        {"subset": a, "superset": b, "n_subset": ca, "n_superset": cb}
                        for a, b, ca, cb in contained
                    ],
                    "holes": [
                        {"predicate": n, "fired": f, "missed": m, "similarity": s}
                        for n, f, m, s in holes
                    ],
                    "errors": errors,
                    "firing_counts": {
                        name: int(c)
                        for name, c in zip(matrix.names, counts, strict=True)
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"JSON を書き出しました: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
