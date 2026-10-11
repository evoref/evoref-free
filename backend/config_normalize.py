"""``evoref config normalize`` — 古い ``config.yaml`` を現行の形へ一度だけ直す (c_05 §7.6)。

設定ファイルはインストール根の ``config.yaml`` のまま (G1 設計 §17.1)。版の無い
G0 の config は:

1. ``config.yaml.g0-<utcstamp>`` へ退避する (0.0.98 へ戻すときに要る)
2. 旧 "coding" モード名時代のキーを現行名へ読み替える (:data:`G0_KEY_RENAMES`)
3. 撤去キーとパスを値に持つキーを落とす — ``local_paths`` は ``outputs_dir`` 以外
   (``local/`` を指す ``outputs_dir`` も)、G0 で撤去済みだったキー
   (:data:`G0_REMOVED_KEYS`)、G1 で撤去したキー (:data:`G1_RETIRED_KEYS`)、
   スキーマが未知と言うキー
4. 先頭に ``config_version`` (:data:`CONFIG_VERSION`) を付けてその場で書き直す (原子的・fsync)
5. 落としたキーの一覧を返す (呼出側が表示する)

``config_version`` が :data:`CONFIG_VERSION` より古い G1 の config (v1) は
``config.yaml.g1v<旧版>-<utcstamp>`` へ退避してから :data:`G1_RETIRED_KEYS` (Pro Level 2
の撤去キー f_04 §6.7 など) を落として版を上げる (v1 からも直接現行へ上がる)。スキーマが未知と言うキーは落とさない
(v1 で起動できていた config なので、残っていれば利用者の誤りとして一覧に出す)。
Pro の上書き ``config.pro.yaml`` (同じディレクトリ) にも同じ撤去キーと版を当てる。

G0 で撤去されたキーの一覧はここだけにあり、スキーマは未知のキーを通常どおり
拒否する。G1 で撤去したキーはスキーマが :func:`reject_retired_keys` で移行の
案内付きで拒否する (``learning`` は ``extra="allow"`` なので黙って通さないため)。

値が不正なキーは直さずに一覧へ出すだけ (利用者の値を黙って消さない)。
現行の版を持つ config には何もしない。書き手ロックの下で呼ぶこと
(serve の稼働中に書き換えない)。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from backend.log_config import get_logger

logger = get_logger("config")

CONFIG_VERSION = 3

#: Pro の上書き設定 (``config.yaml`` と同じディレクトリ)。
PRO_OVERLAY_NAME = "config.pro.yaml"

#: 旧 "coding" モード名時代のキー → 現行キー (どの階層でも読み替える)。
G0_KEY_RENAMES: dict[str, str] = {
    "coding": "create",
    "coding_task": "create_task",
    "coding_model": "create_model",
    "coding_workspace_dir": "create_workspace_dir",
    "coding_budget_tokens": "create_budget_tokens",
    "coding_code_signal": "create_code_signal",
}

#: G0 で撤去済みだったキー (ドット区切りのパスを tuple で)。G0 のスキーマは
#: これらを WARNING 付きで落とすか理由付きで拒否していた。
G0_REMOVED_KEYS: tuple[tuple[str, ...], ...] = (
    # アクティブなセッションはターンごとの追記ログになり、チェックポイントは撤去 (G1-4)
    ("history", "checkpoint_interval"),
    # 宣言だけで読み手が無かった memory のキー (2026-09-02 監査)
    ("memory", "pin", "unlimited"),
    ("memory", "pin", "auto_detect_confirm"),
    ("memory", "facts", "trigger"),
    ("memory", "subject_dictionary", "auto_expand"),
    ("memory", "subject_dictionary", "file"),
    ("memory", "conflict", "chat_review", "max_judge_per_session"),
    ("memory", "project", "archive_dir"),
    # c_16 (2026-09-07) で機能ごと消えた順位付け・減衰のキー (c_16 §8.2)
    ("memory", "fade_alpha"),
    ("memory", "fade_beta"),
    ("memory", "fade_gamma"),
    ("memory", "fade_threshold"),
    ("memory", "lightmem_decay_days"),
    ("memory", "half_life_days_by_tag"),
    # EvorefMem のスキーマ版マーカー (G1 は形式ごとの版と世代印で持つ)
    ("memory", "schema_version"),
    # 3a-2 (2026-09-19) で撤去した create のディスパッチ
    ("agent", "delegate_codegen_to_longform"),
    ("create", "dispatch"),
    # キャッシュの置き場はデータ根の cache/embeddings/ 固定 (c_05 §0.2)
    ("embedding", "cache_dir"),
    # LLM 版の検索必要性 / 品質判定と決定リコール (2026-08-26 撤去)
    ("rag", "self_rag", "quality_judge"),
    ("rag", "self_rag", "necessity_judge"),
    ("rag", "self_rag", "necessity_recall"),
    ("rag", "self_rag", "quality_recall"),
    # Contextual Retrieval (2026-09-12 撤去)
    ("rag", "contextual_prefix"),
    # リランカーと専用アシストモデル (2026-08-14 撤去)
    ("reranker",),
    ("model_paths", "reranker_model"),
    ("model_paths", "assist_model"),
    ("model_paths", "assist_create_model"),
    ("model_paths", "aux_model"),
    # develop モードは CLI フラグ --develop=<level> だけで決める
    ("debug",),
    # 学習データは常に model_key でパーティション化する (c_05 §0.5.12)
    ("learning", "partition_by_base_model"),
    # 機能ごと撤去した常駐ループと Web ターミナル
    ("loop",),
    ("pro", "terminal"),
)

#: G1 で撤去したキー。どれも ``<節>.<キー>`` の 2 段。
#: v1 → v2: 2026-10-10 の Pro Level 2 撤去 (f_04 §6.7)。
#: v2 → v3: 読み手の無い ``prompt.fewshot_core_max_tokens`` (f_04 §10.1)。
G1_RETIRED_KEYS: tuple[tuple[str, str], ...] = (
    *(("learning", key) for key in (
        "level2_adapter_partition",
        "level2_min_failures",
        "level2_min_failures_by_mode",
        "level2_spsa_iterations",
        "level2_sparse_params",
        "level2_schedule_hour",
        "level2_recheck_interval_sec",
        "level2_overdue_hours",
        "level2_stale_streak",
        "level2_stale_backoff_hours",
        "level2_min_improvement_ratio",
        "level2_min_improvement_sigma",
        "level2_early_stop_patience",
        "level2_eval_no_warmup",
        "level2_bootstrap_enabled",
        "level2_bootstrap_rank",
        "level2_bootstrap_init_sigma",
        "level2_bootstrap_min_failures",
        "level2_base_method",
        "base_realeval_spsa_iterations",
        "base_eval_scratch_port",
        "base_eval_loss_w1",
        "base_eval_loss_w2",
        "base_eval_max_tokens",
        "base_eval_max_cases",
        "base_eval_health_timeout",
        "cvector_method",
        "cvector_pca_batch",
        "cvector_pca_iter",
        "cvector_layer_range",
        "cvector_scale",
        "cvector_seed_pairs_file",
        "cvector_min_experiences",
        "optimizer",
        "cma_sigma0",
        "cma_popsize",
    )),
    ("llama", "lora_target"),
    ("prompt", "fewshot_core_max_tokens"),
)


def reject_retired_keys(section: str, data: dict[str, Any]) -> None:
    """``section`` 節に :data:`G1_RETIRED_KEYS` が残っていれば移行の案内付きで拒否する。"""
    found = [f"{sec}.{key}" for sec, key in G1_RETIRED_KEYS if sec == section and key in data]
    if found:
        raise ValueError(
            f"{', '.join(found)}: removed (retired key, docs/c_05 §7.6). "
            "Stop evoref and run `evoref config normalize` to back up config.yaml "
            "(and config.pro.yaml) and drop these keys.",
        )


@dataclass(slots=True)
class NormalizeReport:
    changed: bool
    backup: Path | None = None
    dropped: list[str] = field(default_factory=list)
    invalid: list[str] = field(default_factory=list)
    renamed: list[str] = field(default_factory=list)
    #: ``config.pro.yaml`` の退避先 (直したときだけ)。そこから落としたキーは
    #: ``dropped`` に ``config.pro.yaml:`` を前置して入る。
    pro_backup: Path | None = None


def _pop(raw: dict[str, Any], dotted: tuple[str, ...], dropped: list[str]) -> None:
    node: Any = raw
    for part in dotted[:-1]:
        node = node.get(part) if isinstance(node, dict) else None
        if not isinstance(node, dict):
            return
    if isinstance(node, dict) and dotted[-1] in node:
        node.pop(dotted[-1])
        dropped.append(".".join(dotted))


def _rename_keys(node: Any, path: str, renamed: list[str]) -> Any:
    if isinstance(node, dict):
        out: dict[Any, Any] = {}
        for key, value in node.items():
            new_key = G0_KEY_RENAMES.get(key, key) if isinstance(key, str) else key
            if new_key != key:
                renamed.append(f"{path}{key}")
            out[new_key] = _rename_keys(value, f"{path}{new_key}.", renamed)
        return out
    if isinstance(node, list):
        return [_rename_keys(v, path, renamed) for v in node]
    return node


def _drop_paths(raw: dict[str, Any], dropped: list[str]) -> None:
    local = raw.get("local_paths")
    if isinstance(local, dict):
        for key in list(local):
            value = str(local[key])
            if key != "outputs_dir" or value.replace("\\", "/").startswith("local/"):
                local.pop(key)
                dropped.append(f"local_paths.{key}")
        if not local:
            raw.pop("local_paths")


def _drop_retired(raw: dict[str, Any], dropped: list[str]) -> None:
    """:data:`G1_RETIRED_KEYS` を落とし、それで空になった節も消す。"""
    before = len(dropped)
    for dotted in G1_RETIRED_KEYS:
        _pop(raw, dotted, dropped)
    if len(dropped) == before:
        return
    for section in dict.fromkeys(sec for sec, _key in G1_RETIRED_KEYS):
        if raw.get(section) == {}:
            raw.pop(section)


def _validate_and_prune(
    raw: dict[str, Any], dropped: list[str], invalid: list[str], *, prune: bool = True,
) -> None:
    """検証し、``prune`` ならスキーマが未知と言うキーを落とす。残りの誤りは ``invalid`` へ。"""
    from pydantic import ValidationError

    from backend.schemas import validate_config

    for _ in range(20):
        try:
            validate_config(copy.deepcopy(raw))
            return
        except ValidationError as e:
            errors = e.errors()
        pruned = False
        for err in errors if prune else ():
            loc = tuple(str(p) for p in err.get("loc", ()))
            if err.get("type") == "extra_forbidden" and loc:
                before = len(dropped)
                _pop(raw, loc, dropped)
                pruned = pruned or len(dropped) > before
        if not pruned:
            invalid.extend(
                f"{'.'.join(str(p) for p in err.get('loc', ()))}: {err.get('msg')}" for err in errors
            )
            return


def _stored_version(raw: dict[str, Any]) -> int | None:
    """``config_version`` の値 (無ければ ``None``)。整数でなければ触らない (現行扱い)。"""
    version = raw.get("config_version")
    if version is None:
        return None
    if isinstance(version, int) and not isinstance(version, bool):
        return version
    return CONFIG_VERSION


def _load_mapping(path: Path) -> tuple[str, dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    raw = yaml.safe_load(text) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} is not a mapping")
    return text, raw


def needs_normalize(config_path: Path) -> bool:
    """``config_path`` か隣の ``config.pro.yaml`` が現行より古い版か (無いファイルは見ない)。"""
    for path in (config_path, config_path.with_name(PRO_OVERLAY_NAME)):
        if not path.is_file():
            continue
        _text, raw = _load_mapping(path)
        version = _stored_version(raw)
        if version is None or version < CONFIG_VERSION:
            return True
    return False


def _write_versioned(path: Path, text: str, raw: dict[str, Any], tag: str) -> Path:
    """元の ``text`` を ``<name>.<tag>-<stamp>`` へ退避し、版を付けた ``raw`` で書き直す。"""
    from backend.io.atomic import atomic_write_text
    from backend.utils import utc_compact_stamp

    backup = path.with_name(f"{path.name}.{tag}-{utc_compact_stamp()}")
    n = 0
    while backup.exists():
        n += 1
        backup = path.with_name(f"{path.name}.{tag}-{utc_compact_stamp()}-{n}")
    atomic_write_text(backup, text, fsync=True)
    body = {key: value for key, value in raw.items() if key != "config_version"}
    out = {"config_version": CONFIG_VERSION, **body}
    atomic_write_text(
        path,
        yaml.dump(out, default_flow_style=False, allow_unicode=True, sort_keys=False),
        fsync=True,
    )
    return backup


def _normalize_main(config_path: Path) -> NormalizeReport:
    text, raw = _load_mapping(config_path)
    version = _stored_version(raw)
    if version is not None and version >= CONFIG_VERSION:
        return NormalizeReport(changed=False)

    dropped: list[str] = []
    invalid: list[str] = []
    renamed: list[str] = []
    if version is None:
        raw = _rename_keys(raw, "", renamed)
        _drop_paths(raw, dropped)
        for dotted in G0_REMOVED_KEYS:
            _pop(raw, dotted, dropped)
        _drop_retired(raw, dropped)
        _validate_and_prune(raw, dropped, invalid)
        tag = "g0"
    else:
        _drop_retired(raw, dropped)
        _validate_and_prune(raw, dropped, invalid, prune=False)
        tag = f"g1v{version}"

    backup = _write_versioned(config_path, text, raw, tag)
    logger.info(
        "Normalized %s from config_version %s to %d (backup: %s, dropped: %s)",
        config_path, version, CONFIG_VERSION, backup, ", ".join(dropped) or "-",
    )
    return NormalizeReport(
        changed=True, backup=backup, dropped=dropped, invalid=invalid, renamed=renamed,
    )


def _normalize_pro_overlay(path: Path, dropped: list[str]) -> Path | None:
    """``config.pro.yaml`` の撤去キーを落として版を上げる (部分設定なので検証はしない)。"""
    if not path.is_file():
        return None
    text, raw = _load_mapping(path)
    version = _stored_version(raw)
    if version is not None and version >= CONFIG_VERSION:
        return None
    found: list[str] = []
    _drop_retired(raw, found)
    backup = _write_versioned(path, text, raw, "g0" if version is None else f"g1v{version}")
    dropped.extend(f"{PRO_OVERLAY_NAME}:{key}" for key in found)
    logger.info(
        "Normalized %s from config_version %s to %d (backup: %s, dropped: %s)",
        path, version, CONFIG_VERSION, backup, ", ".join(found) or "-",
    )
    return backup


def normalize_config(config_path: Path) -> NormalizeReport:
    """``config_path`` と隣の ``config.pro.yaml`` を現行の形へ直す (現行の版なら何もしない)。"""
    report = _normalize_main(config_path)
    pro_backup = _normalize_pro_overlay(config_path.with_name(PRO_OVERLAY_NAME), report.dropped)
    if pro_backup is not None:
        report.changed = True
        report.pro_backup = pro_backup
    return report


__all__ = [
    "CONFIG_VERSION",
    "G0_KEY_RENAMES",
    "G0_REMOVED_KEYS",
    "G1_RETIRED_KEYS",
    "PRO_OVERLAY_NAME",
    "NormalizeReport",
    "needs_normalize",
    "normalize_config",
    "reject_retired_keys",
]
