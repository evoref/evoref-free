"""``evoref config normalize`` — G0 の ``config.yaml`` を G1 の形へ一度だけ直す (c_05 §7.6)。

設定ファイルはインストール根の ``config.yaml`` のまま (G1 設計 §17.1)。版の無い
config は:

1. ``config.yaml.g0-<utcstamp>`` へ退避する (0.0.98 へ戻すときに要る)
2. 旧 "coding" モード名時代のキーを現行名へ読み替える (:data:`G0_KEY_RENAMES`)
3. 撤去キーとパスを値に持つキーを落とす — ``local_paths`` は ``outputs_dir`` 以外
   (``local/`` を指す ``outputs_dir`` も)、G0 で撤去済みだったキー
   (:data:`G0_REMOVED_KEYS`)、スキーマが未知と言うキー
4. 先頭に ``config_version: 1`` を付けてその場で書き直す (原子的・fsync)
5. 落としたキーの一覧を返す (呼出側が表示する)

G1 のスキーマは撤去キーの一覧を持たない (空から始める)。G0 で撤去された
キーの一覧はここだけにあり、スキーマは未知のキーを通常どおり拒否する。

値が不正なキーは直さずに一覧へ出すだけ (利用者の値を黙って消さない)。
``config_version`` を持つ config には何もしない。書き手ロックの下で呼ぶこと
(serve の稼働中に書き換えない)。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONFIG_VERSION = 1

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


@dataclass(slots=True)
class NormalizeReport:
    changed: bool
    backup: Path | None = None
    dropped: list[str] = field(default_factory=list)
    invalid: list[str] = field(default_factory=list)
    renamed: list[str] = field(default_factory=list)


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


def _validate_and_prune(raw: dict[str, Any], dropped: list[str], invalid: list[str]) -> None:
    from pydantic import ValidationError

    from backend.schemas import validate_config

    for _ in range(20):
        try:
            validate_config(copy.deepcopy(raw))
            return
        except ValidationError as e:
            errors = e.errors()
        pruned = False
        for err in errors:
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


def normalize_config(config_path: Path) -> NormalizeReport:
    """``config_path`` を G1 の形へ直す (版があれば何もしない)。"""
    from backend.io.atomic import atomic_write_text
    from backend.utils import utc_compact_stamp

    text = config_path.read_text(encoding="utf-8")
    raw = yaml.safe_load(text) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path} is not a mapping")
    if "config_version" in raw:
        return NormalizeReport(changed=False)

    dropped: list[str] = []
    invalid: list[str] = []
    renamed: list[str] = []
    raw = _rename_keys(raw, "", renamed)
    _drop_paths(raw, dropped)
    for dotted in G0_REMOVED_KEYS:
        _pop(raw, dotted, dropped)
    _validate_and_prune(raw, dropped, invalid)

    backup = config_path.with_name(f"{config_path.name}.g0-{utc_compact_stamp()}")
    n = 0
    while backup.exists():
        n += 1
        backup = config_path.with_name(f"{config_path.name}.g0-{utc_compact_stamp()}-{n}")
    atomic_write_text(backup, text, fsync=True)
    out = {"config_version": CONFIG_VERSION, **raw}
    atomic_write_text(
        config_path,
        yaml.dump(out, default_flow_style=False, allow_unicode=True, sort_keys=False),
        fsync=True,
    )
    return NormalizeReport(
        changed=True, backup=backup, dropped=dropped, invalid=invalid, renamed=renamed,
    )


__all__ = ["CONFIG_VERSION", "G0_KEY_RENAMES", "G0_REMOVED_KEYS", "NormalizeReport", "normalize_config"]
