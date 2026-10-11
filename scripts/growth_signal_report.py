"""回帰台帳の収穫源ごとの件数と「再生できる件数」を出す (読むだけ)。

自己学習の再設計 (Level 2 = 成長、2026-10-10) で、回帰台帳が何件のケースを収穫できるかを、
実装の前に経験 (``experience.jsonl``) から見積もる。データ根へは何も書かない。数える源:

- a 検証済み訂正 (``user_correction`` + 訂正の正値 ``correction_correct_value``、宛先は
  ``resolve_corrected_turn`` で解いた誤っていたターン。未検証の候補は数えない = 不変則 #12)
- b 👎 (``user_negative``)
- c 決定論の検証器による ``turn_outcome=failed`` (チャネル別、``fitness.outcome_channel``)
- d 整合性検査の検出は対象外 (経験に残らず、収穫は sleep-time の ``conversation_consistency`` 側)

「再生できる」= ``learning.case_filters`` の除外述語 (``regeneration_mismatch_reason`` /
``replay_context``) を通ること。述語と理由の名前は sleep-time Step 13.6 の収穫
(``learning.growth``) と共有し、ここでは再実装しない。モデル由来のチャネル
(Chinese / Hangul token leaked = ``foreign_script_leak``) は規則で直せないので、
「規則で直せる見込み」からは除く。

episodic の取りこぼしラベル (f_04 §6.9 の段 3 の着手条件) = 検証済み訂正の正値 Y が、訂正が
指すターン (``resolve_corrected_turn``) より前に書かれた live なノートに逐語で在り、そのターンで
注入されなかった件。注入の記録は経験の ``gen_config.evidence_ids`` (``episodic:<id>``、c_16 §5.5)、
ノートは episodic ストアを読むだけで開く (プロセスを readonly にしてから読む)。private の
ノートはラベルにせず件数だけ数える。出すのは ID・正値の digest と字数・ノート内の位置だけで、
本文は出さない。注入の記録が無い世代の経験しか無ければ ``not_computable`` (理由つき) とし、
0 件とは言わない。着手条件の判定 (``recall_gap_gate``) は自然な利用日の数え方を
``learning.health_report`` と共有する。

使い方::

    python scripts/growth_signal_report.py                # 既定のデータ根
    python scripts/growth_signal_report.py --data-root userdata --json
    python scripts/growth_signal_report.py --since 2026-10-07
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.constants import SECONDS_PER_DAY  # noqa: E402
from backend.data_root import resolve_data_root  # noqa: E402
from backend.free.learning.corrected_pairs import resolve_corrected_turn  # noqa: E402
from backend.free.learning.fitness import outcome_channel  # noqa: E402
from backend.free.learning.growth.harvest import (  # noqa: E402
    REJECT_CORRECTION_TURN_FAILED as EXCL_CORRECTION_TURN_FAILED,
    REJECT_NO_CORRECT_VALUE as EXCL_NO_CORRECT_VALUE,
    REJECT_TARGET_UNRESOLVED as EXCL_TARGET_UNRESOLVED,
)
from backend.free.learning.growth.regression import (  # noqa: E402
    MODEL_ORIGIN_PREFIXES,
    REASON_NO_QUERY as EXCL_NO_QUERY,
    replay_exclusion as _replay_exclusion,
)
from backend.free.learning.health_report import (  # noqa: E402
    MIN_NATURAL_DAYS,
    NATURAL_USE_DAYS,
    load_records,
    natural_day_keys,
    stamp_records,
)
from backend.free.learning.level0_instant import teaches_via_verified_correction  # noqa: E402
from backend.utils import parse_utc, utc_now, utc_now_dt  # noqa: E402

#: ``outcome_channel`` が ``None`` を返す failed (別の欠陥キーが数えている長文の検証落ち)。
CHANNEL_COUNTED_ELSEWHERE = "counted_elsewhere"

#: 段 3 (episodic 想起の鍵) の着手に要る取りこぼしラベルの件数 (f_04 §6.9)。
RECALL_GAP_MIN_LABELS = 10

#: 注入の記録 (経験の ``gen_config.evidence_ids``)。これが無い経験は判定できない。
INJECTION_RECORD = "gen_config.evidence_ids"

# 取りこぼしラベルの候補 (検証済み訂正) の行き先。label 以外は件数だけ出す。
GAP_LABEL = "label"
GAP_RECALLED = "recalled"  # 正値を持つノートは全部そのターンで注入されていた
GAP_NO_NOTE = "no_live_note"  # 前に書かれた live なノートに正値が無い
GAP_PRIVATE_ONLY = "private_note_only"  # 正値を持つ未注入のノートが private だけ
GAP_TARGET_UNRESOLVED = "target_unresolved"
GAP_NO_INJECTION_RECORD = "injection_not_recorded"
GAP_NO_TIMESTAMP = "target_without_timestamp"


def _signals(rec: dict) -> dict:
    sig = rec.get("signals")
    return sig if isinstance(sig, dict) else {}


def _new_source(label: str) -> dict[str, Any]:
    return {"label": label, "total": 0, "replayable": 0, "excluded": Counter()}


def _tally(src: dict[str, Any], reason: str | None) -> None:
    src["total"] += 1
    if reason is None:
        src["replayable"] += 1
    else:
        src["excluded"][reason] += 1


def load_episodic_notes(data_root: Path) -> tuple[list[dict] | None, str | None]:
    """episodic ストアのノートを読むだけで取り出す。``(notes, None)`` か ``(None, 読めない理由)``。

    各ノートは ``{id, text, created_at, active, private}``。``active`` は private を含めた
    live 判定 (``is_active(..., include_private=True)``: 撤回・supersede・無視・secret を除く)。
    読む間だけプロセスを readonly にする (``EvidenceStore.load`` の回復経路が書かないように)。
    ストアのディレクトリが無ければ空のストア (ノート 0 件) とする。
    """
    from backend.config import PathResolver
    from backend.data_root import store_root
    from backend.free.memory.episodic.store import STORE_DIRNAME
    from backend.free.rag.evidence.store import EvidenceStore, is_active
    from backend.io.readonly import disable_readonly, enable_readonly, is_readonly

    store_dir = PathResolver.layout_path(data_root, "memory_dir") / STORE_DIRNAME
    if not store_dir.is_dir():
        return [], None
    own_readonly = not is_readonly()
    if own_readonly:
        enable_readonly(store_root(data_root), "growth_signal_report reads only")
    store = EvidenceStore(store_dir, STORE_DIRNAME)
    try:
        store.load()
        if store.readonly:
            return None, f"episodic store is readonly ({store.readonly_reason})"
        now_epoch = utc_now_dt().timestamp()
        notes = [
            {
                "id": record.id,
                "text": record.text or "",
                "created_at": parse_utc(record.created_at),
                "active": is_active(record, now_epoch, include_private=True),
                "private": bool(record.private),
            }
            for record in store.iter_records()
            if record.kind == "note"
        ]
    except Exception as e:  # noqa: BLE001 — 報告は読めない理由を出して続ける
        return None, f"episodic store could not be read ({type(e).__name__}: {e})"
    finally:
        store.close()
        if own_readonly:
            disable_readonly()
    return notes, None


def _value_digest(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def episodic_recall_gaps(
    records_by_partition: dict[str, list[dict]],
    notes: list[dict] | None,
    *,
    notes_unavailable: str | None = None,
    since=None,
) -> dict[str, Any]:
    """検証済み訂正ごとに、正値が未注入の live なノートに逐語で在るかを数える (純関数)。

    「その回」= 訂正が指す誤っていたターン。そのターンの時刻より前に書かれた (``created_at``)
    いま live なノートだけを見る (訂正の発話から起こしたノートは訂正より後に書かれるので
    入らない)。いま live でないノート (後で撤回・supersede されたもの) は見ないので、件数は
    下振れ側に倒れる。
    """
    outcomes: Counter[str] = Counter()
    labels: list[dict[str, Any]] = []
    for recs in records_by_partition.values():
        modes: dict[str, list[dict]] = {}
        for rec in recs:
            modes.setdefault(str(rec.get("mode") or ""), []).append(rec)
        for scoped in modes.values():
            for i, exp in enumerate(scoped):
                sig = _signals(exp)
                value = str(sig.get("correction_correct_value") or "").strip()
                if not sig.get("user_correction") or not value:
                    continue
                ts = parse_utc(exp.get("timestamp"))
                if since is not None and (ts is None or ts < since):
                    continue
                target = resolve_corrected_turn(scoped, i)
                if target is None:
                    outcomes[GAP_TARGET_UNRESOLVED] += 1
                    continue
                gen = target.get("gen_config")
                if not isinstance(gen, dict) or not isinstance(gen.get("evidence_ids"), list):
                    outcomes[GAP_NO_INJECTION_RECORD] += 1
                    continue
                target_ts = parse_utc(target.get("timestamp"))
                if target_ts is None:
                    outcomes[GAP_NO_TIMESTAMP] += 1
                    continue
                if notes is None:
                    continue
                injected = {
                    str(eid).split(":", 1)[1] for eid in gen["evidence_ids"]
                    if str(eid).startswith("episodic:")
                }
                hits: list[dict[str, Any]] = []
                private_hits = recalled_hits = 0
                for note in notes:
                    created = note.get("created_at")
                    if not note.get("active") or created is None or created >= target_ts:
                        continue
                    pos = str(note.get("text") or "").find(value)
                    if pos < 0:
                        continue
                    if note["id"] in injected:
                        recalled_hits += 1
                    elif note.get("private"):
                        private_hits += 1
                    else:
                        hits.append({"note_id": note["id"], "position": pos})
                if hits:
                    outcomes[GAP_LABEL] += 1
                    labels.append({
                        "experience_id": exp.get("id"),
                        "target_experience_id": target.get("id"),
                        "timestamp": exp.get("timestamp"),
                        "value_digest": _value_digest(value),
                        "value_chars": len(value),
                        "notes": hits,
                    })
                elif private_hits:
                    outcomes[GAP_PRIVATE_ONLY] += 1
                elif recalled_hits:
                    outcomes[GAP_RECALLED] += 1
                else:
                    outcomes[GAP_NO_NOTE] += 1

    out: dict[str, Any] = {"outcomes": dict(sorted(outcomes.items()))}
    if notes is None:
        return {
            **out, "status": "not_computable", "count": None, "labels": [],
            "reason": notes_unavailable or "episodic store was not read",
            "missing": ["episodic store"],
        }
    judged = sum(n for k, n in outcomes.items() if k not in (GAP_TARGET_UNRESOLVED, GAP_NO_TIMESTAMP))
    if judged and outcomes[GAP_NO_INJECTION_RECORD] == judged:
        return {
            **out, "status": "not_computable", "count": None, "labels": [],
            "reason": f"訂正が指すターンのどれにも注入の記録 ({INJECTION_RECORD}) が無い",
            "missing": [INJECTION_RECORD],
        }
    return {**out, "status": "computed", "count": len(labels), "labels": labels, "reason": None, "missing": []}


def recall_gap_gate(
    records_by_partition: dict[str, list[dict]], gaps: dict[str, Any], *, now=None,
) -> dict[str, Any]:
    """段 3 の着手条件 (自然な利用 30 日で取りこぼしラベル ≥ ``RECALL_GAP_MIN_LABELS`` 件) を判定する。

    「自然な利用 30 日」は ``learning.health_report`` の判定と同じ (経験の始まりから
    ``NATURAL_USE_DAYS`` 日以上、かつ直近 ``NATURAL_USE_DAYS`` 日の窓に自然な利用日が
    ``MIN_NATURAL_DAYS`` 日以上)。ラベルも自然な利用日の訂正だけを数える (監査・試験の
    バーストを除く、``natural_*`` と同じ扱い)。
    """
    now = now or utc_now_dt()
    stamped = stamp_records(records_by_partition)
    natural = natural_day_keys(stamped, now)
    history = (now - stamped[0][0]).total_seconds() / SECONDS_PER_DAY if stamped else 0.0
    if gaps.get("status") != "computed":
        return {
            "natural_days": len(natural), "labels": None, "met": False,
            "reason": f"取りこぼしラベルを算出できない: {gaps.get('reason')}",
        }
    natural_labels = sum(
        1 for label in gaps["labels"]
        if (ts := parse_utc(label.get("timestamp"))) is not None and ts.date().isoformat() in natural
    )
    short: list[str] = []
    if history < NATURAL_USE_DAYS:
        short.append(f"経験の始まりから {history:.1f} 日 (基準 {NATURAL_USE_DAYS} 日以上)")
    if len(natural) < MIN_NATURAL_DAYS:
        short.append(
            f"直近 {NATURAL_USE_DAYS} 日の自然な利用日 {len(natural)} 日 (基準 {MIN_NATURAL_DAYS} 日以上)"
        )
    if natural_labels < RECALL_GAP_MIN_LABELS:
        short.append(
            f"自然な利用日の取りこぼしラベル {natural_labels} 件 (基準 {RECALL_GAP_MIN_LABELS} 件以上、"
            f"全体では {gaps['count']} 件)"
        )
    return {
        "natural_days": len(natural),
        "labels": natural_labels,
        "met": not short,
        "reason": "着手条件を満たした" if not short else "未達: " + " / ".join(short),
    }


def analyze(
    records_by_partition: dict[str, list[dict]],
    *,
    since=None,
    notes: list[dict] | None = None,
    notes_unavailable: str | None = None,
    now=None,
) -> dict[str, Any]:
    """経験から収穫源ごとの件数を集計する (純関数)。``since`` より前の「源のターン」は数えない。

    ``notes`` は :func:`load_episodic_notes` の結果 (``None`` なら取りこぼしラベルは
    ``not_computable``、理由は ``notes_unavailable``)。
    """
    a = _new_source("verified_correction")
    b = _new_source("user_negative")
    c = _new_source("verifier_failure")
    by_channel: dict[str, dict[str, Any]] = {}
    model_origin: Counter[str] = Counter()
    failed_total = 0
    fixable_ids: set[str] = set()
    turns = 0

    def in_scope(rec: dict) -> bool:
        if since is None:
            return True
        ts = parse_utc(rec.get("timestamp"))
        return ts is not None and ts >= since

    for recs in records_by_partition.values():
        modes: dict[str, list[dict]] = {}
        for rec in recs:
            modes.setdefault(str(rec.get("mode") or ""), []).append(rec)
        for scoped in modes.values():
            for i, exp in enumerate(scoped):
                if not in_scope(exp):
                    continue
                turns += 1
                sig = _signals(exp)
                if sig.get("user_correction"):
                    if not teaches_via_verified_correction(sig):
                        _tally(a, EXCL_CORRECTION_TURN_FAILED)
                    elif not str(sig.get("correction_correct_value") or "").strip():
                        _tally(a, EXCL_NO_CORRECT_VALUE)
                    else:
                        target = resolve_corrected_turn(scoped, i)
                        _tally(a, EXCL_TARGET_UNRESOLVED if target is None else _replay_exclusion(target))
                if sig.get("user_negative"):
                    _tally(b, _replay_exclusion(exp))
                if sig.get("turn_outcome") == "failed":
                    failed_total += 1
                    reason_text = str(sig.get("turn_outcome_reason") or "")
                    channel = outcome_channel(sig) or CHANNEL_COUNTED_ELSEWHERE
                    excl = _replay_exclusion(exp)
                    _tally(c, excl)
                    _tally(by_channel.setdefault(channel, _new_source(channel)), excl)
                    origin = next((p for p in MODEL_ORIGIN_PREFIXES if reason_text.startswith(p)), None)
                    if origin:
                        model_origin[origin] += 1
                    elif excl is None:
                        fixable_ids.add(f"{id(scoped)}:{i}")

    def plain(src: dict[str, Any]) -> dict[str, Any]:
        return {**src, "excluded": dict(sorted(src["excluded"].items()))}

    fixable = len(fixable_ids)
    gaps = episodic_recall_gaps(
        records_by_partition, notes, notes_unavailable=notes_unavailable, since=since,
    )
    return {
        "generated_at": utc_now(),
        "turns_in_scope": turns,
        "sources": {
            "a_verified_correction": plain(a),
            "b_user_negative": plain(b),
            "c_verifier_failure": {
                **plain(c),
                "by_channel": {k: plain(v) for k, v in sorted(by_channel.items())},
            },
            "d_consistency": {
                "label": "consistency",
                "status": "out_of_scope",
                "note": "整合性検査の検出は経験に残らない (収穫は sleep-time 側の find_*)。この報告の対象外",
            },
        },
        "model_origin": {"total": sum(model_origin.values()), "by_reason": dict(sorted(model_origin.items()))},
        "failures": {
            "total": failed_total,
            "fixable_by_rule": fixable,
            "fixable_ratio": round(fixable / failed_total, 4) if failed_total else None,
        },
        "episodic_missed_labels": gaps,
        "recall_gap_gate": recall_gap_gate(records_by_partition, gaps, now=now),
    }


def render(report: dict[str, Any]) -> str:
    """人が読む形に整形する。"""
    src = report["sources"]
    lines = [f"== 回帰台帳の収穫源 (対象ターン {report['turns_in_scope']} 件) =="]
    for key in ("a_verified_correction", "b_user_negative", "c_verifier_failure"):
        s = src[key]
        lines.append(f"{key}: {s['total']} 件 (再生可能 {s['replayable']} 件)  除外: {s['excluded'] or '-'}")
    for ch, s in src["c_verifier_failure"]["by_channel"].items():
        lines.append(f"    {ch}: {s['total']} 件 (再生可能 {s['replayable']} 件)  除外: {s['excluded'] or '-'}")
    lines.append(f"d_consistency: 対象外 ({src['d_consistency']['note']})")
    mo = report["model_origin"]
    lines.append(f"モデル由来のチャネル: {mo['total']} 件 {mo['by_reason'] or ''}".rstrip())
    f = report["failures"]
    ratio = "-" if f["fixable_ratio"] is None else f"{f['fixable_ratio'] * 100:.1f}%"
    lines.append(
        f"失敗 {f['total']} 件のうち規則で直せる見込み (再生可能かつモデル由来でない): "
        f"{f['fixable_by_rule']} 件 ({ratio})"
    )
    em = report["episodic_missed_labels"]
    if em["status"] == "computed":
        lines.append(f"episodic の取りこぼしラベル: {em['count']} 件  内訳: {em['outcomes'] or '-'}")
    else:
        lines.append(f"episodic の取りこぼしラベル: 算出不能 ({em['reason']})")
    gate = report["recall_gap_gate"]
    lines.append(f"段 3 の着手条件 (met={gate['met']}): {gate['reason']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="回帰台帳の収穫源ごとの件数と再生可能件数")
    parser.add_argument("--data-root", default=None, help="データ根 (既定は解決規則どおり)")
    parser.add_argument("--since", default=None, help="この日付 (UTC、YYYY-MM-DD) 以降のターンだけを数える")
    parser.add_argument("--json", action="store_true", help="JSON で出す")
    args = parser.parse_args(argv)

    data_root = resolve_data_root(args.data_root)
    since = parse_utc(args.since + "T00:00:00Z") if args.since else None
    if args.since and since is None:
        parser.error(f"--since を解釈できない: {args.since!r}")
    notes, unavailable = load_episodic_notes(data_root)
    report = analyze(load_records(data_root), since=since, notes=notes, notes_unavailable=unavailable)
    sys.stdout.reconfigure(errors="replace")  # type: ignore[union-attr]
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(render(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
