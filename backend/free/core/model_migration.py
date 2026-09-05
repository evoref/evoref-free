"""モデル移行: model_state.json 管理 + 移行処理

設計書 docs/22_base_model_migration.md に準拠。
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

from backend.io import atomic_write_text
from backend.log_config import get_logger
from backend.utils import utc_now as _now

logger = get_logger("core.model_migration")


# ────────────────────────────────────────────
# 補助モデル: コンポーネント定義
# ────────────────────────────────────────────

ModelComponent = Literal["embedding"]
ALL_COMPONENTS: tuple[str, ...] = ("embedding",)

# config.yaml の model_paths 配下のキー対応
COMPONENT_CONFIG_KEY: dict[str, str] = {
    "embedding": "embed_model",
}

# コンポーネント別 LoRA 設定キー: (adapter_key, versions_key, archive_root, default_adapter_path)。
# base 用 (local_paths.lora_adapter / lora_versions_dir / local/lora_archive/)
# とは別キー・別アーカイブ先を使う (base の flat な lora_archive/<stem>/ との
# ファイル名衝突を避けるため component 別サブディレクトリに分離する)。
# default_adapter_path は backend/schemas/paths.py::LocalPathsConfig の
# デフォルト値と一致させる。
_COMPONENT_LORA_KEYS: dict[str, tuple[str, str, str, str]] = {
    "embedding": (
        "embed_lora_adapter", "embed_lora_versions_dir", "lora_archive/embedding",
        "local/models/embed_adapter.gguf",
    ),
}

# config.yaml の model_paths 配下で model_state.json と同期されるキー。
# これらは migrate API (POST /api/model/migrate, /api/model/{component}/migrate)
# 経由でしか変更できない。config を直書きすると model_state.json と desync し、
# 起動時に mismatch を起こすため API 層で遮断する。create_model は model_state
# 非追跡 (未指定時は base_model にフォールバック) のため対象外。
MODEL_STATE_TRACKED_KEYS: frozenset[str] = frozenset(
    ("base_model", *COMPONENT_CONFIG_KEY.values()),
)


# ────────────────────────────────────────────
# ModelState: local/model_state.json 管理
# ────────────────────────────────────────────


@dataclass
class ModelCurrent:
    """現在のベースモデル情報"""
    filename: str = ""
    chat_template_name: str = ""
    has_system_role: bool = True
    activated_at: str = ""


@dataclass
class MigrationHistoryEntry:
    """移行履歴の 1 エントリ"""
    from_model: str = ""
    to_model: str = ""
    migrated_at: str = ""
    lora_archived_to: str = ""


@dataclass
class ComponentState:
    """embedding の current + history"""
    current: ModelCurrent = field(default_factory=ModelCurrent)
    history: list[MigrationHistoryEntry] = field(default_factory=list)


class ModelState:
    """local/model_state.json の読み書き管理"""

    def __init__(self, state_path: Path):
        self.path = state_path
        self._current = ModelCurrent()
        self._lora_compatible = True
        self._migration_history: list[MigrationHistoryEntry] = []
        self._components: dict[str, ComponentState] = {
            name: ComponentState() for name in ALL_COMPONENTS
        }
        self._load()

    # ── プロパティ ──

    @property
    def current_filename(self) -> str:
        return self._current.filename

    @property
    def current(self) -> ModelCurrent:
        return self._current

    @property
    def lora_compatible(self) -> bool:
        return self._lora_compatible

    @lora_compatible.setter
    def lora_compatible(self, value: bool) -> None:
        self._lora_compatible = value

    @property
    def migration_history(self) -> list[MigrationHistoryEntry]:
        return self._migration_history

    # ── コンポーネント ──

    def get_component(self, name: str) -> ComponentState:
        if name not in self._components:
            raise ValueError(f"Unknown component: {name}")
        return self._components[name]

    def get_component_current_filename(self, name: str) -> str:
        return self.get_component(name).current.filename

    def update_component_current(self, name: str, filename: str) -> None:
        comp = self.get_component(name)
        comp.current = ModelCurrent(filename=filename, activated_at=_now())

    def add_component_migration(
        self, name: str, from_model: str, to_model: str,
        lora_archived_to: str = "",
    ) -> None:
        comp = self.get_component(name)
        comp.history.append(MigrationHistoryEntry(
            from_model=from_model,
            to_model=to_model,
            migrated_at=_now(),
            lora_archived_to=lora_archived_to,
        ))

    def get_component_last_migration(
        self, name: str,
    ) -> MigrationHistoryEntry | None:
        comp = self.get_component(name)
        return comp.history[-1] if comp.history else None

    # ── 永続化 ──

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            current = data.get("current", {})
            self._current = ModelCurrent(
                filename=current.get("filename", ""),
                chat_template_name=current.get("chat_template_name", ""),
                has_system_role=current.get("has_system_role", True),
                activated_at=current.get("activated_at", ""),
            )
            self._lora_compatible = data.get("lora_compatible", True)
            for h in data.get("migration_history", []):
                self._migration_history.append(MigrationHistoryEntry(
                    from_model=h.get("from", ""),
                    to_model=h.get("to", ""),
                    migrated_at=h.get("migrated_at", ""),
                    lora_archived_to=h.get("lora_archived_to", ""),
                ))
            # コンポーネント
            comps = data.get("components", {}) or {}
            for name in ALL_COMPONENTS:
                raw = comps.get(name, {}) or {}
                cur = raw.get("current", {}) or {}
                comp = ComponentState(
                    current=ModelCurrent(
                        filename=cur.get("filename", ""),
                        chat_template_name=cur.get("chat_template_name", ""),
                        has_system_role=cur.get("has_system_role", True),
                        activated_at=cur.get("activated_at", ""),
                    ),
                    history=[
                        MigrationHistoryEntry(
                            from_model=h.get("from", ""),
                            to_model=h.get("to", ""),
                            migrated_at=h.get("migrated_at", ""),
                            lora_archived_to=h.get("lora_archived_to", ""),
                        )
                        for h in (raw.get("history", []) or [])
                    ],
                )
                self._components[name] = comp
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to load model_state.json: %s", e)

    def save(self) -> None:
        """model_state.json をディスクに保存"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "current": {
                "filename": self._current.filename,
                "chat_template_name": self._current.chat_template_name,
                "has_system_role": self._current.has_system_role,
                "activated_at": self._current.activated_at,
            },
            "lora_compatible": self._lora_compatible,
            "migration_history": [
                {
                    "from": h.from_model,
                    "to": h.to_model,
                    "migrated_at": h.migrated_at,
                    "lora_archived_to": h.lora_archived_to,
                }
                for h in self._migration_history
            ],
            "components": {
                name: {
                    "current": {
                        "filename": comp.current.filename,
                        "chat_template_name": comp.current.chat_template_name,
                        "has_system_role": comp.current.has_system_role,
                        "activated_at": comp.current.activated_at,
                    },
                    "history": [
                        {
                            "from": h.from_model,
                            "to": h.to_model,
                            "migrated_at": h.migrated_at,
                            "lora_archived_to": h.lora_archived_to,
                        }
                        for h in comp.history
                    ],
                }
                for name, comp in self._components.items()
            },
        }
        atomic_write_text(
            self.path,
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ── 更新操作 ──

    def update_current(
        self,
        filename: str,
        chat_template_name: str = "",
        has_system_role: bool = True,
    ) -> None:
        """現在のモデル情報を更新"""
        self._current = ModelCurrent(
            filename=filename,
            chat_template_name=chat_template_name,
            has_system_role=has_system_role,
            activated_at=_now(),
        )

    def add_migration(
        self,
        from_model: str,
        to_model: str,
        lora_archived_to: str = "",
    ) -> None:
        """移行履歴にエントリを追加"""
        self._migration_history.append(MigrationHistoryEntry(
            from_model=from_model,
            to_model=to_model,
            migrated_at=_now(),
            lora_archived_to=lora_archived_to,
        ))

    def get_last_migration(self) -> MigrationHistoryEntry | None:
        """直前の移行履歴を取得"""
        if not self._migration_history:
            return None
        return self._migration_history[-1]

    def initialize_from_config(self, config: dict) -> None:
        """model_state.json が存在しない場合に config.yaml から初期化"""
        model_paths = config.get("model_paths", {}) or {}
        base_model = model_paths.get("base_model") or ""
        filename = Path(base_model).name if base_model else ""
        changed = False
        if not self._current.filename:
            self._current = ModelCurrent(
                filename=filename,
                activated_at=_now(),
            )
            changed = True
            logger.info("ModelState initialized from config: %s", filename)

        # コンポーネント
        for name in ALL_COMPONENTS:
            comp = self._components[name]
            if comp.current.filename:
                continue
            cfg_key = COMPONENT_CONFIG_KEY[name]
            raw = model_paths.get(cfg_key, "")
            if not raw:
                continue
            comp.current = ModelCurrent(
                filename=Path(raw).name,
                activated_at=_now(),
            )
            changed = True
            logger.info(
                "ModelState component initialized: %s = %s",
                name, comp.current.filename,
            )

        if changed:
            self.save()


def detect_mismatches(
    model_state: "ModelState", config: dict,
) -> dict[str, dict[str, str]]:
    """config.yaml の model_paths と model_state.json の current filename を比較する。

    base_model と各 component (embed) について、
    config と model_state の双方が非空かつ basename が異なるキーだけを返す。
    片方でも空 (初回起動で未初期化等) のキーは誤検知を避けるため除外する。

    Returns:
        ``{config_key: {"model_state": <filename>, "config": <filename>}}``。
        ``config_key`` は ``"base_model"`` または component の config キー
        (``embed_model``)。
    """
    model_paths = config.get("model_paths", {}) or {}
    result: dict[str, dict[str, str]] = {}

    ms_base = model_state.current_filename
    cfg_base = Path(model_paths.get("base_model") or "" or "").name
    if ms_base and cfg_base and ms_base != cfg_base:
        result["base_model"] = {"model_state": ms_base, "config": cfg_base}

    for component, cfg_key in COMPONENT_CONFIG_KEY.items():
        ms_name = model_state.get_component_current_filename(component)
        cfg_name = Path(model_paths.get(cfg_key, "") or "").name
        if ms_name and cfg_name and ms_name != cfg_name:
            result[cfg_key] = {"model_state": ms_name, "config": cfg_name}

    return result


# ────────────────────────────────────────────
# MigrationResult
# ────────────────────────────────────────────


@dataclass
class MigrationResult:
    """移行結果"""
    dry_run: bool = False
    old_model: str = ""
    new_model: str = ""
    lora_action: str = "archived"
    data_summary: dict = field(default_factory=dict)
    calibration: dict | None = None
    recommendations: list[str] = field(default_factory=list)


# ────────────────────────────────────────────
# ModelMigrator: 移行処理の実行
# ────────────────────────────────────────────


class ModelMigrator:
    """ベースモデル移行処理（§22.4.2 フロー）"""

    def __init__(
        self,
        config: dict,
        project_root: Path,
        model_state: ModelState,
        experience_buf=None,
        prompt_manager=None,
        eval_core_manager=None,
        learning_scheduler=None,
        short_term_memory=None,
        vector_store=None,
        cartridge_manager=None,
    ):
        self.config = config
        self.project_root = project_root
        self.model_state = model_state
        self.experience_buf = experience_buf
        self.prompt_manager = prompt_manager
        self.eval_core_manager = eval_core_manager
        self.learning_scheduler = learning_scheduler
        self.short_term_memory = short_term_memory
        self._vector_store = vector_store
        self._cartridge_manager = cartridge_manager

    def migrate(
        self,
        new_model_path: str,
        *,
        try_lora: bool = False,
        regenerate_context: bool = False,
        dry_run: bool = False,
    ) -> MigrationResult:
        """移行を実行（§22.4.2 Step 1〜9）

        Args:
            new_model_path: 新モデルの GGUF ファイルパス
            try_lora: LoRA 互換性テストを試みるか
            regenerate_context: context_description を再生成するか
            dry_run: ドライラン（変更しない）

        Returns:
            MigrationResult

        Raises:
            MigrationError: 事前検証エラー
            MigrationBusyError: 学習サイクル実行中
        """
        resolved_path = Path(new_model_path)
        if not resolved_path.is_absolute():
            resolved_path = self.project_root / resolved_path

        new_model_filename = resolved_path.name
        old_model_filename = self._get_current_filename()

        result = MigrationResult(
            dry_run=dry_run,
            old_model=old_model_filename,
            new_model=new_model_filename,
        )

        # Step 1: 事前検証
        self._validate(resolved_path)

        # データ集計
        result.data_summary = self._gather_data_summary()

        if dry_run:
            # partition 有効時 (既定) は実 migrate が LoRA アーカイブを skip する
            # ため dry_run も "kept" を返す (過大表示防止)。data_summary も
            # _gather_data_summary 側で partition-aware に集計済み。
            result.lora_action = (
                "kept" if (self._is_partitioned() or try_lora) else "archived"
            )
            result.recommendations = (
                self._known_issue_recommendations(resolved_path)
                + self._build_recommendations(dry_run=True)
            )
            return result

        # base 学習パーティション有効時は、Step 3-5 の flat 無効化 (LoRA アーカイブ /
        # 経験 perplexity リセット / プロンプト候補クリア) を行わない。学習データは
        # モデル別パーティションで保全されており、これらは旧モデルの保全データを
        # 破壊する (戻したとき復元できなくなる)。新モデルのパーティションは別途
        # 起動時 _activate_learning_partition で activate され、空なら一から学習する。
        partitioned = self._is_partitioned()

        if partitioned:
            logger.info(
                "Migration under partition_by_base_model: skipping flat "
                "LoRA-archive / experience-reset / prompt-meta-clear "
                "(per-model partitions preserve old-model learning)",
            )
            lora_action = "kept"
            result.lora_action = lora_action
        else:
            # Step 3: LoRA アーカイブ
            lora_action = self._archive_lora(old_model_filename, try_lora)
            result.lora_action = lora_action

            # Step 4: 経験バッファ更新
            self._update_experience_buffer(old_model_filename)

            # Step 5: プロンプトメタ情報更新
            self._update_prompt_meta(old_model_filename)

            # Step 6: コア評価セット準備
            self._reset_eval_core()

        # Step 7 (部分): config.yaml 更新
        self._update_config(new_model_path)

        # Step 8: メモリノート処理（オプション）
        if regenerate_context:
            self._mark_context_regeneration()

        # Step 9: model_state 更新
        archive_dir = f"local/lora_archive/{Path(old_model_filename).stem}/"
        self.model_state.add_migration(
            from_model=old_model_filename,
            to_model=new_model_filename,
            lora_archived_to=archive_dir if lora_action != "kept" else "",
        )
        self.model_state.update_current(filename=new_model_filename)
        self.model_state.lora_compatible = (try_lora and lora_action == "kept")
        self.model_state.save()

        known_issues = self._known_issue_recommendations(resolved_path)
        result.recommendations = known_issues + self._build_recommendations(
            dry_run=False,
        )
        for issue in self._target_known_issues(resolved_path):
            logger.warning(
                "Target model %s has a known issue: %s",
                new_model_filename, issue,
            )
        logger.info(
            "Migration completed: %s -> %s (lora: %s)",
            old_model_filename, new_model_filename, lora_action,
        )
        return result

    def rollback(self, target_model: str | None = None) -> dict:
        """ロールバック処理（§22.6）

        Args:
            target_model: ロールバック先モデル名。省略時は直前の移行元

        Returns:
            {"rolled_back_to": str, "lora_restored": bool}

        Raises:
            MigrationError: 履歴なし / ロールバック不能
        """
        last = self.model_state.get_last_migration()
        if last is None:
            raise MigrationError("No migration history found")

        rollback_target = target_model or last.from_model
        if not rollback_target:
            raise MigrationError("Cannot determine rollback target model")

        # LoRA 復元
        lora_restored = self._restore_lora(rollback_target)

        # config.yaml 更新
        old_base_model = self.config.get("model_paths", {}).get("base_model") or ""
        model_dir = Path(old_base_model).parent if old_base_model else Path("models")
        rollback_path = str(model_dir / rollback_target)
        self._update_config(rollback_path)

        # プロンプト model_calibrated_for 更新
        if self.prompt_manager:
            for mode in self.prompt_manager.MODES:
                try:
                    meta = self.prompt_manager.get_meta(mode)
                    meta.model_calibrated_for = rollback_target
                    self.prompt_manager._save_meta(mode)
                except ValueError:
                    continue

        # model_state 更新
        self.model_state.update_current(filename=rollback_target)
        self.model_state.lora_compatible = lora_restored
        self.model_state.save()

        logger.info(
            "Rollback completed: -> %s (lora_restored=%s)",
            rollback_target, lora_restored,
        )
        return {
            "rolled_back_to": rollback_target,
            "lora_restored": lora_restored,
        }

    # ── コンポーネント移行 ──

    def migrate_component(
        self,
        component: str,
        new_model_path: str,
        *,
        dry_run: bool = False,
    ) -> MigrationResult:
        """embedding モデルを切り替える

        base モデルと違い、経験バッファ・プロンプトメタなどのパーティション系
        付帯処理は不要 (embed の学習データは f_04_self_learning.md
        §1.2 のとおり元々 flat 共有でモデル別パーティション化されない)。
        LoRA のみ :meth:`_archive_component_lora_if_incompatible` で新モデルとの
        arch 整合性を確認し、不一致時のみアーカイブする。config.yaml 更新と
        model_state 記録も行う。実際の llama-server 再起動とクライアント
        差し替えは L2 で対応する。
        """
        if component not in ALL_COMPONENTS:
            raise MigrationError(
                f"Unknown component: {component}. "
                f"Expected one of {ALL_COMPONENTS}",
            )

        resolved = Path(new_model_path)
        if not resolved.is_absolute():
            resolved = self.project_root / resolved

        new_filename = resolved.name
        old_filename = self._get_component_current(component)

        result = MigrationResult(
            dry_run=dry_run,
            old_model=old_filename,
            new_model=new_filename,
            lora_action="n/a",
        )

        # 検証
        if not resolved.exists():
            raise MigrationError(f"New model file not found: {resolved}")
        if not resolved.is_file():
            raise MigrationError(f"Not a file: {resolved}")
        if (
            self.learning_scheduler
            and getattr(self.learning_scheduler, "running", False)
        ):
            raise MigrationBusyError("Learning cycle is currently running")

        if dry_run:
            result.recommendations = (
                self._known_issue_recommendations(resolved)
                + self._build_component_recommendations(component, dry_run=True)
            )
            return result

        # config.yaml 更新
        self._update_component_config(component, new_model_path)

        # LoRA: 新モデルと不適合 (arch / hidden size 不一致、または判定不能) の
        # 場合のみアーカイブ。適合時は f_04_self_learning.md §1.2 の flat 共有
        # 方針どおり persist する。
        lora_action = self._archive_component_lora_if_incompatible(
            component, old_filename, resolved,
        )
        result.lora_action = lora_action

        # model_state 更新
        self.model_state.add_component_migration(
            component, old_filename, new_filename,
            lora_archived_to=(
                f"local/{_COMPONENT_LORA_KEYS[component][2]}/"
                f"{Path(old_filename).stem}/"
                if lora_action == "archived" else ""
            ),
        )
        self.model_state.update_component_current(component, new_filename)
        self.model_state.save()

        result.recommendations = (
            self._known_issue_recommendations(resolved)
            + self._build_component_recommendations(
                component, dry_run=False,
                model_changed=old_filename != new_filename,
            )
        )
        for issue in self._target_known_issues(resolved):
            logger.warning(
                "Target %s model %s has a known issue: %s",
                component, new_filename, issue,
            )
        logger.info(
            "Component migration completed: %s: %s -> %s",
            component, old_filename, new_filename,
        )
        return result

    def rollback_component(
        self, component: str, target_model: str | None = None,
    ) -> dict:
        """コンポーネントモデルをロールバック"""
        if component not in ALL_COMPONENTS:
            raise MigrationError(f"Unknown component: {component}")

        last = self.model_state.get_component_last_migration(component)
        if last is None:
            raise MigrationError(
                f"No migration history for component: {component}",
            )
        rollback_target = target_model or last.from_model
        if not rollback_target:
            raise MigrationError(
                f"Cannot determine rollback target for {component}",
            )

        # 既存の config 値を流用してパスを推定
        cfg_key = COMPONENT_CONFIG_KEY[component]
        old_path = self.config.get("model_paths", {}).get(cfg_key, "")
        model_dir = Path(old_path).parent if old_path else Path("models")
        rollback_path = str(model_dir / rollback_target)

        self._update_component_config(component, rollback_path)

        adapter_key, versions_key, archive_root, _default_adapter = (
            _COMPONENT_LORA_KEYS[component]
        )
        lora_restored = self._restore_lora(
            rollback_target,
            adapter_key=adapter_key, versions_key=versions_key,
            archive_root=archive_root,
        )

        self.model_state.update_component_current(component, rollback_target)
        self.model_state.save()

        logger.info(
            "Component rollback completed: %s -> %s (lora_restored=%s)",
            component, rollback_target, lora_restored,
        )
        return {
            "rolled_back_to": rollback_target,
            "lora_restored": lora_restored,
        }

    def _get_component_current(self, component: str) -> str:
        cur = self.model_state.get_component_current_filename(component)
        if cur:
            return cur
        cfg_key = COMPONENT_CONFIG_KEY[component]
        raw = self.config.get("model_paths", {}).get(cfg_key, "")
        return Path(raw).name if raw else "unknown"

    def _archive_component_lora_if_incompatible(
        self, component: str, old_model_name: str, new_model_path: Path,
    ) -> str:
        """新モデルと既存 LoRA の互換性を判定し、不適合 (または判定不能) の
        ときのみ退避する。

        適合時は f_04_self_learning.md §1.2 の「embed の学習は
        モデル別パーティション化されず flat に共有される」方針どおり LoRA を
        persist させる (無条件アーカイブだと同一 arch 内でのモデル切替
        (量子化違い等) でも毎回学習を破棄してしまい、この設計意図を壊す)。

        判定は launch_llama.py の :func:`lora_compatible_with_model` を起動側
        ガード (``_lora_compatible``) と共有する。``general.architecture`` の
        一致に加え、adapter の全 ``*.lora_a`` / ``*.lora_b`` テンソルをモデル
        側の対応 weight 実形状と突合する — 同一 arch でもサイズ違い
        (例: gemma-4-E2B 1536 vs E4B 2560) や head 構成違いの LoRA を残すと
        llama-server が tensor 形状不一致でプロセスごと落ちるため、arch
        文字列の一致だけでは適合と言えない。arch 判定不能を安全側で
        アーカイブするのは起動側の fail-closed 方針と対称。
        """
        adapter_key, versions_key, archive_root, default_adapter = (
            _COMPONENT_LORA_KEYS[component]
        )
        lp = self.config.get("local_paths", {})
        lora_path = self._resolve_path(lp.get(adapter_key, default_adapter))
        if not lora_path.exists():
            return "n/a"

        try:
            from scripts.launch_llama import (
                lora_compatible_with_model,
                read_gguf_metadata,
            )
        except Exception as exc:
            logger.warning(
                "component LoRA compatibility check: launch_llama import "
                "failed: %s", exc,
            )
            return self._archive_lora(
                old_model_name, False,
                adapter_key=adapter_key, versions_key=versions_key,
                archive_root=archive_root,
            )

        compatible, reason = lora_compatible_with_model(
            new_model_path, lora_path,
        )
        if compatible:
            logger.info(
                "Component LoRA compatible (%s), keeping: %s",
                reason, lora_path,
            )
            # 系統 stamp の無いレガシーアダプタは arch/形状しか検証できて
            # いない。モデルが実際に変わる切替では「学習元不明のまま持ち
            # 越す」ことを観測可能にする (挙動は従来どおり keep)。
            if (
                old_model_name != new_model_path.name
                and not read_gguf_metadata(lora_path).get("trained_on_model")
            ):
                logger.warning(
                    "Component LoRA lineage unverifiable (no trained-on "
                    "stamp); kept by arch/shape check only: %s", lora_path,
                )
            return "kept"

        logger.info(
            "Component LoRA incompatible (%s), archiving: %s",
            reason, lora_path,
        )
        return self._archive_lora(
            old_model_name, False,
            adapter_key=adapter_key, versions_key=versions_key,
            archive_root=archive_root,
        )

    def _target_known_issues(self, resolved_path: Path) -> list[str]:
        """切替先モデルのプロファイルに宣言された既知の弱点を返す。

        切替**前**に伝えるのが目的。起動時の品質プローブ
        (:mod:`backend.free.llm.quality_probe`) は事後の観測で、切替を戻すには
        もう一度 migrate + 再起動が要る。プロファイルに実測済みの弱点があるなら
        ``--dry-run`` の時点で出すのが最も安い。

        モデル別層 (by-model) が効くので、同 arch でもサイズ・量子化ごとに
        別の弱点を宣言できる。プロファイルに ``quality_baseline`` が無い / 読めない
        場合は空。判断材料が無いことを警告にはしない (未知 ≠ 悪い)。
        """
        try:
            from scripts.launch_llama import load_model_profile_for

            profile = load_model_profile_for(resolved_path, self.project_root)
        except Exception as exc:
            logger.debug(
                "known-issue lookup failed for %s: %s", resolved_path, exc,
            )
            return []
        raw = (profile or {}).get("quality_baseline")
        if not isinstance(raw, dict):
            return []
        issues = raw.get("known_issues")
        return [str(i) for i in issues] if isinstance(issues, list) else []

    def _known_issue_recommendations(self, resolved_path: Path) -> list[str]:
        """既知の弱点を推奨アクション文へ整形する (無ければ空リスト)。"""
        issues = self._target_known_issues(resolved_path)
        if not issues:
            return []
        return [
            f"切替先モデルには既知の弱点が報告されています: {issue}"
            for issue in issues
        ] + [
            "切替後の起動時に出力品質プローブが走ります "
            "(結果は GET /api/model/quality)",
        ]

    #: プロファイルの ``embedding.<予約キー>`` → 転写先 config セクションと、
    #: そこで受け付けるキーの許可リスト。
    #:
    #: **なぜ profile に置くか** — ここに並ぶのはすべて「2 つの埋め込みの
    #: コサインを絶対値で比べる」閾値で、到達可能なスコア域が埋め込みモデル
    #: ごとに大きく違う。実測 (2026-08-30、同一ペアで 3 モデル比較):
    #:
    #:     無関係ペアの cos 中央値   LFM2.5 0.105 / Qwen3 0.273 / bge-m3 0.459
    #:
    #: つまり LFM2.5 で「無関係」を意味する 0.3 は、bge-m3 では **全件が超える**。
    #: 較正 (memory_threshold_calibration) が面倒を見るのは rag.* と
    #: memory.relevance_min_score だけで、ここに並ぶ値は **どこにも自動追従が
    #: 無く、モデルを替えると旧モデル前提の値が黙って残る**。
    #:
    #: リコール閾値 (tools.*) は 2026-06-29 に同じ方式で profile 化済みで、
    #: 本表はその一般化。許可リスト方式にしてあるので、プロファイルが
    #: 無関係な config を書き換えることはできない。
    _EMBEDDING_SCALED_THRESHOLDS: dict[str, tuple[str, ...]] = {
        "recall": (
            "url_recall_min_score",
            "executable_command_recall_min_score",
        ),
        "memory": (
            "note_link_threshold",
            "conflict_similarity_threshold",
            "attribute_similarity_threshold",
            "min_merge_similarity",
        ),
        "rag": (
            "cartridge_gate_threshold",
        ),
    }

    #: 予約キー → config のトップレベルセクション名。
    _THRESHOLD_SECTION: dict[str, str] = {
        "recall": "tools",
        "memory": "memory",
        "rag": "rag",
    }

    def _resolve_embedding_profile_params(self, resolved_path: Path) -> dict:
        """新 embed モデルの ``embedding.*`` パラメータを解決する。

        ``dim`` は GGUF の ``embedding_length`` (権威・必ず GGUF 由来)、
        ``model_name`` は GGUF ファイル名 stem、
        ``query_template`` / ``doc_template`` / ``instructions`` /
        ``max_length`` / ``pooling`` / ``context_size`` はプロファイルの
        ``embedding:`` ブロック (``models/profiles/<arch>.yaml`` + モデル別層
        ``by-model/<GGUF stem>.yaml``) から取る。プロファイルに embedding
        ブロックが無い場合はテンプレート系を据え置き (WARNING)。
        embed component-migrate / rollback の config 同期に使う。

        ``embedding.recall`` サブブロック (URL/コマンドリコールの sim 閾値) が
        あれば予約キー ``"recall"`` に入れて返す。これは ``embedding.*`` では
        なく ``tools.*`` 設定なので、:meth:`_update_component_config` 側で
        pop して ``tools`` へルーティングする。

        返すのは「設定すべきキーだけ」。GGUF 読取失敗時は dim を省く
        (既存 embedding.dim を温存) — 幅を誤って書くと VectorStore /
        dimension_check を壊すため。
        """
        params: dict = {}
        try:
            from scripts.launch_llama import load_model_profile_for, read_gguf_metadata
        except Exception as exc:
            logger.warning("embed config sync: launch_llama import failed: %s", exc)
            return params

        try:
            meta = read_gguf_metadata(resolved_path)
        except Exception as exc:
            logger.warning(
                "embed config sync: GGUF read failed for %s: %s", resolved_path, exc,
            )
            meta = {}

        dim = meta.get("embedding_length")
        if dim:
            params["dim"] = int(dim)
        else:
            logger.warning(
                "embed config sync: GGUF embedding_length unreadable for %s; "
                "keeping existing embedding.dim", resolved_path,
            )

        params["model_name"] = resolved_path.stem

        try:
            profile = load_model_profile_for(resolved_path, self.project_root)
        except Exception as exc:
            logger.warning(
                "embed config sync: profile load failed for %s: %s", resolved_path, exc,
            )
            profile = {}
        emb_prof = (profile or {}).get("embedding")
        if isinstance(emb_prof, dict) and emb_prof:
            # query/doc テンプレ・instructions・max_length (1 入力の推奨上限)・
            # pooling (llama-server --pooling へ転写、CLS pooling 系モデル向け)
            # のみ同期。context_size/batch_size/ubatch_size はサーバ側 KV/バッチ
            # 資源で、並列スロット分を要し model 固有でないため同期しない
            # (config/schema 既定に委ねる)。
            for key in (
                "query_template", "doc_template", "instructions", "max_length",
                "pooling",
            ):
                if key in emb_prof:
                    params[key] = emb_prof[key]
            # コサインを直接比べる閾値は **埋め込みモデルの sim 分布に依存する
            # model 固有値**。embedding.* ではない別セクションへ送るため、
            # 予約キーに退避して :meth:`_update_component_config` でルーティングする。
            for reserved, allowed in self._EMBEDDING_SCALED_THRESHOLDS.items():
                block = emb_prof.get(reserved)
                if not isinstance(block, dict):
                    continue
                picked = {k: block[k] for k in allowed if k in block}
                if picked:
                    params[reserved] = picked
        else:
            logger.warning(
                "embed config sync: %s has no embedding profile; query/doc "
                "templates + instructions left unchanged — review embedding.* manually",
                resolved_path.name,
            )
        return params

    #: プロファイルのキー → config 上の **実際の位置** (トップレベルからのパス)。
    #: ここに無いキーは ``<section>.<key>`` へそのまま書く。
    #:
    #: config は素の平坦な辞書ではなくネストしたスキーマなので、位置を間違えると
    #: ``extra_forbidden`` で **起動しなくなる**。実インシデント (2026-08-30):
    #: ``attribute_similarity_threshold`` を ``memory.*`` 直下へ書いてしまい、
    #: 切替後の backend が Config validation failed で落ちた。
    #: :class:`TestEmbeddingScaledThresholdSync` が実スキーマで検証する。
    _THRESHOLD_CONFIG_PATH: dict[str, tuple[str, ...]] = {
        "cartridge_gate_threshold": ("rag", "cartridge_gate", "threshold"),
        "attribute_similarity_threshold": (
            "memory", "conflict", "attribute_similarity_threshold",
        ),
        "min_merge_similarity": (
            "memory", "conflict_resolver", "min_merge_similarity",
        ),
    }

    #: 実行時にプロファイルから解決されるキー。config 側が **明示的に
    #: ``None``** (= プロファイル追随を選んでいる) なら転写しない — 書くと
    #: その時点の値で固定され、以後プロファイル (by-model 層) を直しても
    #: 追随しなくなる。未記載 (キー自体が無い) は従来どおり転写する。
    _RUNTIME_PROFILE_RESOLVED: frozenset[str] = frozenset({
        "cartridge_gate_threshold",
    })

    @classmethod
    def _apply_routed(cls, target: dict, routed: dict[str, dict]) -> None:
        """予約キー由来の閾値を config ツリーの **正しい位置** へ書き込む。

        プロファイルに値が無いキーは ``routed`` に含まれないので、``None`` や
        既定値 (0.3 等) を config へ書くことはない。
        """
        for section, values in routed.items():
            for key, value in values.items():
                if value is None:
                    continue
                path = cls._THRESHOLD_CONFIG_PATH.get(key, (section, key))
                node = target
                for part in path[:-1]:
                    node = node.setdefault(part, {})
                if (
                    key in cls._RUNTIME_PROFILE_RESOLVED
                    and path[-1] in node
                    and node[path[-1]] is None
                ):
                    continue
                node[path[-1]] = value

    def _update_component_config(
        self, component: str, new_model_path: str,
    ) -> None:
        config_path = self.project_root / "config.yaml"
        if not config_path.exists():
            logger.warning("config.yaml not found, skipping update")
            return
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        cfg.setdefault("model_paths", {})[COMPONENT_CONFIG_KEY[component]] = (
            new_model_path
        )
        # embedding 切替時は新モデルの embedding.* (dim/model_name/テンプレ/instruction)
        # を同期する。これをやらないと旧モデルの prefix スキームで新モデルが駆動され
        # RAG が静かに劣化する (同 dim swap では dimension_check も検知しない)。
        emb_params: dict = {}
        routed: dict[str, dict] = {}
        if component == "embedding":
            resolved = Path(new_model_path)
            if not resolved.is_absolute():
                resolved = self.project_root / resolved
            emb_params = self._resolve_embedding_profile_params(resolved)
            # コサインスケール依存の閾値は embedding.* ではなく、それぞれの
            # セクション (tools / memory / rag) へルーティングする。
            for reserved, section in self._THRESHOLD_SECTION.items():
                picked = emb_params.pop(reserved, {})
                if picked:
                    routed[section] = picked
            self._apply_routed(cfg, routed)
            if emb_params:
                cfg.setdefault("embedding", {}).update(emb_params)
        # config.yaml が truncate 途中で壊れると起動不能になる。atomic + fsync。
        atomic_write_text(
            config_path,
            yaml.dump(
                cfg,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            ),
            fsync=True,
        )
        # in-memory config も同期 (restart+rebind が同じ singleton を再読する)
        self.config.setdefault("model_paths", {})[
            COMPONENT_CONFIG_KEY[component]
        ] = new_model_path
        if emb_params:
            self.config.setdefault("embedding", {}).update(emb_params)
            logger.info(
                "embed config synced: embedding.%s", sorted(emb_params.keys()),
            )
        if routed:
            self._apply_routed(self.config, routed)
            logger.info(
                "embedding-scaled thresholds synced: %s",
                {sec: sorted(vals) for sec, vals in routed.items()},
            )
        logger.info(
            "config.yaml updated: model_paths.%s = %s",
            COMPONENT_CONFIG_KEY[component], new_model_path,
        )

    def _build_component_recommendations(
        self, component: str, *, dry_run: bool, model_changed: bool = True,
    ) -> list[str]:
        if dry_run:
            return [
                "ドライラン完了。--dry-run を外して実行すると切替が反映されます",
            ]
        recs = [
            f"{component} モデルの llama-server 再起動が必要です。"
            "process_manager.enabled=true で自動再起動されます "
            f"(無効時は手動再起動、または POST /api/model/process/{component}/restart)",
        ]
        if component == "embedding" and model_changed:
            recs.append(
                "RAG ベクトルストアの再構築も必要です。'evoref reindex' "
                "(または POST /api/rag/reindex) を実行してください。"
                "実行するまで検索結果は信頼できません。",
            )
            recs.append(
                "SemMem ファクトの埋め込みも再構築が必要です。"
                "'python scripts/evorefmem_cli.py reembed-facts --apply' "
                "を実行してください。",
            )
        return recs

    # ── 内部メソッド ──

    def _get_current_filename(self) -> str:
        """現在のモデルファイル名を取得"""
        if self.model_state.current_filename:
            return self.model_state.current_filename
        base_model = self.config.get("model_paths", {}).get("base_model") or ""
        return Path(base_model).name if base_model else "unknown"

    def _validate(self, new_model_path: Path) -> None:
        """Step 1: 事前検証"""
        if not new_model_path.exists():
            raise MigrationError(
                f"New model file not found: {new_model_path}"
            )
        if not new_model_path.is_file():
            raise MigrationError(f"Not a file: {new_model_path}")

        # 学習サイクル実行中チェック
        if self.learning_scheduler and self.learning_scheduler.running:
            raise MigrationBusyError("Learning cycle is currently running")

    def _is_partitioned(self) -> bool:
        """base 学習パーティション有効か (既定 True)。

        有効時は migrate() が flat 無効化 (LoRA アーカイブ / 経験 perplexity
        リセット / プロンプト候補クリア) を skip する。dry_run の lora_action と
        data_summary もこれに合わせて非破壊側へ倒し、過大表示を防ぐ。
        """
        return bool(
            self.config.get("learning", {}).get("partition_by_base_model", True),
        )

    def _gather_data_summary(self) -> dict:
        """移行対象データの集計

        partition_by_base_model 有効時 (既定) は flat 無効化を skip するため、
        破壊対象カウント (perplexity_reset / prompts_modes) は 0 / 空で返す。
        migrate() の partitioned 分岐と一致させ dry_run の過大表示を防ぐ。
        experience_entries / memory_notes / rag_chunks / cartridges は保持
        データ件数の情報表示なので partition に依らず集計する。
        """
        partitioned = self._is_partitioned()
        summary: dict = {
            "memory_notes": 0,
            "experience_entries": 0,
            "perplexity_reset": 0,
            "rag_chunks": 0,
            "cartridges": 0,
            "prompts_modes": [],
        }

        if self.experience_buf:
            summary["experience_entries"] = self.experience_buf.count
            if not partitioned:
                summary["perplexity_reset"] = sum(
                    1 for e in self.experience_buf.entries
                    if e.signals.perplexity is not None
                )

        if self.prompt_manager and not partitioned:
            summary["prompts_modes"] = list(self.prompt_manager.MODES)

        if self.short_term_memory:
            summary["memory_notes"] = len(
                getattr(self.short_term_memory, "notes", [])
            )

        # RAG / カートリッジは非依存データのため集計のみ
        if self._vector_store:
            summary["rag_chunks"] = self._vector_store.count
        if self._cartridge_manager:
            summary["cartridges"] = len(getattr(self._cartridge_manager, "installed", {}))

        return summary

    def _archive_lora(
        self,
        old_model_name: str,
        try_lora: bool,
        *,
        adapter_key: str = "lora_adapter",
        versions_key: str = "lora_versions_dir",
        archive_root: str = "lora_archive",
    ) -> str:
        """Step 3: LoRA アーカイブ

        base 用の既定キーワード引数はそのまま (完全後方互換)。embed
        用は :data:`_COMPONENT_LORA_KEYS` のキーを渡して呼ぶ。
        """
        lp = self.config.get("local_paths", {})
        lora_path = self._resolve_path(
            lp.get(adapter_key, "local/models/adapter.gguf")
        )
        lora_versions_dir = self._resolve_path(
            lp.get(versions_key, "local/lora_versions/")
        )

        if not lora_path.exists():
            logger.info("No LoRA adapter found, skipping archive")
            return "archived"

        if try_lora:
            logger.info("--try-lora: keeping LoRA for compatibility test")
            return "kept"

        archive_dir = (
            self.project_root / "local" / archive_root
            / Path(old_model_name).stem
        )
        archive_dir.mkdir(parents=True, exist_ok=True)

        # adapter.gguf コピー
        shutil.copy2(str(lora_path), str(archive_dir / "adapter.gguf"))
        logger.info(
            "LoRA archived: %s -> %s",
            lora_path, archive_dir / "adapter.gguf",
        )

        # lora_versions/ コピー
        if lora_versions_dir.exists() and any(lora_versions_dir.iterdir()):
            versions_archive = archive_dir / "versions"
            if versions_archive.exists():
                shutil.rmtree(str(versions_archive))
            shutil.copytree(str(lora_versions_dir), str(versions_archive))
            logger.info(
                "LoRA versions archived: %s -> %s",
                lora_versions_dir, versions_archive,
            )

        # 元の LoRA を削除
        lora_path.unlink()
        logger.info("Original LoRA adapter removed: %s", lora_path)

        # lora_versions を空にする (.gitkeep は tracked な構造保持ファイルのため残す)
        if lora_versions_dir.exists():
            for f in lora_versions_dir.iterdir():
                if f.is_file():
                    if f.name == ".gitkeep":
                        continue
                    f.unlink()
                elif f.is_dir():
                    shutil.rmtree(str(f))
            logger.info("LoRA versions cleared: %s", lora_versions_dir)

        return "archived"

    def _update_experience_buffer(self, old_model_name: str) -> None:
        """Step 4: 経験バッファ更新"""
        if self.experience_buf is None:
            return

        for entry in self.experience_buf.entries:
            if not entry.base_model:
                entry.base_model = old_model_name
            entry.signals.perplexity = None

        # 永続化。in-memory バッファは **active パーティション** の内容なので、
        # flat パス (local/experience.json) へ書くとパーティション側が古いまま
        # 残り、shutdown で上書きされて移行結果が消える (2026-09-05 監査)。
        # active パーティションへ書く。flat パス (local/experience.json) へ書くと
        # パーティション側が古いまま残り、shutdown で上書きされて移行結果が
        # 消える (2026-09-05 監査)。グローバル resolver 未初期化 (単体テスト等)
        # では従来の flat パスへ倒す。
        try:
            from backend.config import get_path_resolver

            exp_file = get_path_resolver().resolve_learning("experience_file")
        except RuntimeError:
            exp_file = self._resolve_path(
                self.config.get("local_paths", {}).get(
                    "experience_file", "local/experience.json",
                ),
            )
        self.experience_buf.save(exp_file)

        logger.info(
            "Experience buffer updated: %d entries, perplexity reset",
            self.experience_buf.count,
        )

    def _update_prompt_meta(self, old_model_name: str) -> None:
        """Step 5: プロンプトメタ情報更新"""
        if self.prompt_manager is None:
            return

        for mode in self.prompt_manager.MODES:
            try:
                meta = self.prompt_manager.get_meta(mode)
                meta.model_calibrated_for = old_model_name
                meta.candidates = []
                self.prompt_manager._save_meta(mode)
                logger.info(
                    "Prompt meta updated: %s.meta.json "
                    "(model_calibrated_for=%s)",
                    mode, old_model_name,
                )
            except ValueError:
                continue

    def _reset_eval_core(self) -> None:
        """Step 6: コア評価セット準備"""
        if self.eval_core_manager is None:
            return

        eval_set = self.eval_core_manager.load()
        for case in eval_set.cases:
            case.max_perplexity = None
        eval_set.version += 1
        self.eval_core_manager.save(eval_set)
        logger.info("Eval core reset: %d cases", len(eval_set.cases))

    def _update_config(self, new_model_path: str) -> None:
        """Step 7 (部分): config.yaml の base_model を更新"""
        config_path = self.project_root / "config.yaml"
        if not config_path.exists():
            logger.warning("config.yaml not found, skipping update")
            return

        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        cfg.setdefault("model_paths", {})["base_model"] = new_model_path

        # config.yaml が truncate 途中で壊れると起動不能になる。atomic + fsync。
        atomic_write_text(
            config_path,
            yaml.dump(
                cfg,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            ),
            fsync=True,
        )

        # in-memory config も同期 (_update_component_config と対称)。
        # 未同期だと移行後 restart 前に get_config() が旧 base_model を返し、
        # /api/model/reload が model_state.current を旧モデル名で上書きして
        # 再起動時に model_state↔config の mismatch ERROR を生む。
        self.config.setdefault("model_paths", {})["base_model"] = new_model_path

        logger.info(
            "config.yaml updated: model_paths.base_model = %s",
            new_model_path,
        )

    def _mark_context_regeneration(self) -> None:
        """Step 8: メモリノートの context_description 再生成マーク"""
        if self.short_term_memory is None:
            return

        count = 0
        for note in getattr(self.short_term_memory, "notes", []):
            if getattr(note, "context_description", ""):
                note.evolution_pending = True
                count += 1

        if count > 0:
            logger.info(
                "Marked %d notes for context regeneration", count,
            )

    def _restore_lora(
        self,
        target_model: str,
        *,
        adapter_key: str = "lora_adapter",
        versions_key: str = "lora_versions_dir",
        archive_root: str = "lora_archive",
    ) -> bool:
        """ロールバック時の LoRA 復元

        base 用の既定キーワード引数はそのまま (完全後方互換)。embed
        用は :data:`_COMPONENT_LORA_KEYS` のキーを渡して呼ぶ。
        """
        lp = self.config.get("local_paths", {})
        lora_path = self._resolve_path(
            lp.get(adapter_key, "local/models/adapter.gguf")
        )
        lora_versions_dir = self._resolve_path(
            lp.get(versions_key, "local/lora_versions/")
        )

        archive_dir = (
            self.project_root / "local" / archive_root
            / Path(target_model).stem
        )

        if not archive_dir.exists():
            logger.info("No LoRA archive found for %s", target_model)
            return False

        lora_restored = False

        # adapter.gguf 復元
        archived_adapter = archive_dir / "adapter.gguf"
        if archived_adapter.exists():
            lora_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(archived_adapter), str(lora_path))
            lora_restored = True
            logger.info(
                "LoRA adapter restored: %s -> %s",
                archived_adapter, lora_path,
            )

        # versions 復元
        archived_versions = archive_dir / "versions"
        if archived_versions.exists():
            if lora_versions_dir.exists():
                shutil.rmtree(str(lora_versions_dir))
            shutil.copytree(str(archived_versions), str(lora_versions_dir))
            logger.info("LoRA versions restored: %s", lora_versions_dir)

        return lora_restored

    def _build_recommendations(self, *, dry_run: bool) -> list[str]:
        """推奨アクションを生成"""
        if dry_run:
            return [
                "ドライラン完了。--dry-run を外して実行すると移行が実行されます",
            ]
        return [
            "llama-server を新モデルで再起動してください",
            "通常通り使用を開始してください（経験が自動蓄積されます）",
            "プロンプトの再最適化: evoref optimize --level1 で手動実行可能",
        ]

    def _resolve_path(self, raw: str) -> Path:
        """パスを絶対パスに解決"""
        path = Path(raw)
        return path if path.is_absolute() else self.project_root / path


# ────────────────────────────────────────────
# 例外クラス
# ────────────────────────────────────────────


class MigrationError(Exception):
    """移行エラー（400 Bad Request に対応）"""
    pass


class MigrationBusyError(MigrationError):
    """学習サイクル実行中エラー（409 Conflict に対応）"""
    pass


# ────────────────────────────────────────────
# ヘルパー
# ────────────────────────────────────────────


