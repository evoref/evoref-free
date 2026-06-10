"""パフォーマンスプリセット定義

設定「一般」ページから推論・RAG の資源系設定を 3 段階
(軽量 / バランス / 高性能) に一括調整するためのプリセット定義とヘルパ。

調整対象は VRAM/RAM 消費に直結する llama-server 起動引数
(slots / context_size / KV キャッシュ量子化 / cache-ram) のみ。
メモリ/学習の挙動には触れない。すべて config.yaml に既存の
キーで、スキーマ変更・新規キー追加は伴わない。

`balanced` は現行 config.yaml.example の出荷既定値と一致させた安全ベースライン。
"""

from __future__ import annotations

# プリセット ID (表示順を保持)
PRESET_IDS: list[str] = ["light", "balanced", "performance"]

# プリセット定義: id -> {section: ネスト dict}
CONFIG_PRESETS: dict[str, dict[str, dict]] = {
    "light": {
        "llama": {
            "slots": 1,
            "context_size": 4096,
            "cache_type_k": "q8_0",
            "cache_type_v": "q4_1",
            "cache_ram_mib": 0,
        },
        "assist_model": {"local": {"context_size": 4096}},
        "embedding": {"max_length": 4096, "context_size": 4096},
    },
    "balanced": {
        "llama": {
            "slots": 1,
            "context_size": 8192,
            "cache_type_k": "q8_0",
            "cache_type_v": "q8_0",
            "cache_ram_mib": 0,
        },
        "assist_model": {"local": {"context_size": 8192}},
        "embedding": {"max_length": 8192, "context_size": 8192},
    },
    "performance": {
        "llama": {
            "slots": 2,
            "context_size": 8192,
            "cache_type_k": "q8_0",
            "cache_type_v": "q8_0",
            "cache_ram_mib": 4096,
        },
        "assist_model": {"local": {"context_size": 8192}},
        "embedding": {"max_length": 8192, "context_size": 8192},
    },
}

# 起動引数を変更するセクション -> 再起動が必要な llama-server 名。
_RESTART_SERVER_BY_SECTION: dict[str, str] = {
    "llama": "base",
    "assist_model": "assist",
    "embedding": "embed",
}


def _flatten(prefix: str, src: dict, out: dict[str, object]) -> None:
    """ネスト dict を 'a.b.c' -> 値 のフラット辞書へ展開する"""
    for key, value in src.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            _flatten(path, value, out)
        else:
            out[path] = value


def _section_matches(current_section: dict, preset_section: dict) -> bool:
    """preset_section の全 leaf 値が current_section に一致するか"""
    flat: dict[str, object] = {}
    _flatten("", preset_section, flat)
    for path, expected in flat.items():
        cur: object = current_section
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return False
            cur = cur[part]
        if cur != expected:
            return False
    return True


def detect_current(cfg: dict) -> str | None:
    """現在の設定に完全一致するプリセット id を返す (なければ None)"""
    for preset_id in PRESET_IDS:
        preset = CONFIG_PRESETS[preset_id]
        if all(
            _section_matches(cfg.get(section, {}) or {}, preset_section)
            for section, preset_section in preset.items()
        ):
            return preset_id
    return None


def compute_changed(cfg: dict, preset_id: str) -> tuple[list[str], list[str]]:
    """プリセット適用で実際に変わるセクションと、要再起動の llama-server 名を返す

    Returns:
        (changed_sections, restart_servers)
    """
    preset = CONFIG_PRESETS[preset_id]
    changed = [
        section
        for section, preset_section in preset.items()
        if not _section_matches(cfg.get(section, {}) or {}, preset_section)
    ]
    restart = [
        _RESTART_SERVER_BY_SECTION[s]
        for s in changed
        if s in _RESTART_SERVER_BY_SECTION
    ]
    return changed, restart
