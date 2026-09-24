"""モデル移行: model_state.json 管理 + 移行処理

設計書 docs/22_base_model_migration.md に準拠。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from backend.io import atomic_write_text
from backend.io.codec import codec_for, persisted
from backend.io.format_registry import FormatSpec, register_format
from backend.io.versioned import VersionedJsonFile
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

# config.yaml の model_paths 配下で model_state.json と同期されるキー。
# これらは migrate API (POST /api/model/migrate, /api/model/{component}/migrate)
# 経由でしか変更できない。config を直書きすると model_state.json と desync し、
# 起動時に mismatch を起こすため API 層で遮断する。create_model は model_state
# 非追跡 (未指定時は base_model にフォールバック) のため対象外。
MODEL_STATE_TRACKED_KEYS: frozenset[str] = frozenset(
    ("base_model", *COMPONENT_CONFIG_KEY.values()),
)


# ────────────────────────────────────────────
# ModelState: <data_root>/store/model_state.json 管理
# ────────────────────────────────────────────


@persisted()
@dataclass
class ModelCurrent:
    """現在のベースモデル情報"""
    filename: str = ""
    chat_template_name: str = ""
    has_system_role: bool = True
    activated_at: str = ""
    #: この版が知らないキー (書き戻しでそのまま戻す)。
    _extra: dict[str, Any] | None = None


@persisted()
@dataclass
class MigrationHistoryEntry:
    """移行履歴の 1 エントリ (ディスク上のキーは ``from`` / ``to``、:data:`_HISTORY_KEYS`)"""
    from_model: str = ""
    to_model: str = ""
    migrated_at: str = ""
    _extra: dict[str, Any] | None = None


@persisted()
@dataclass
class ComponentState:
    """embedding の current + history"""
    current: ModelCurrent = field(default_factory=ModelCurrent)
    history: list[MigrationHistoryEntry] = field(default_factory=list)
    _extra: dict[str, Any] | None = None


@persisted()
@dataclass
class ModelStatePayload:
    """``model_state.json`` の payload (コーデックの表)。"""
    current: ModelCurrent = field(default_factory=ModelCurrent)
    migration_history: list[MigrationHistoryEntry] = field(default_factory=list)
    components: dict[str, ComponentState] = field(default_factory=dict)
    _extra: dict[str, Any] | None = None


_PAYLOAD_CODEC = codec_for(ModelStatePayload)

#: 移行履歴の項目のディスク上のキー → フィールド名 (``from`` は Python の予約語で
#: フィールド名にできないので、コーデックの前後で付け替える)。
_HISTORY_KEYS: dict[str, str] = {"from": "from_model", "to": "to_model"}

MODEL_STATE_FORMAT = register_format(FormatSpec(
    format_id="model_state",
    version=1,
    klass="sot",
    writers=frozenset({"free"}),
    path_key="store/model_state.json",
    retention="rewritten in place; migration history is kept",
    records=(ModelStatePayload,),
))


def _rename_history(payload: Any, keys: dict[str, str]) -> Any:
    """payload の移行履歴の項目 (トップと各コンポーネント) のキーを ``keys`` で付け替えた写し。"""
    if not isinstance(payload, dict):
        return payload

    def with_entries(obj: dict[str, Any], key: str) -> dict[str, Any]:
        value = obj.get(key)
        if not isinstance(value, list):
            return obj
        renamed = [
            {keys.get(k, k): v for k, v in item.items()} if isinstance(item, dict) else item
            for item in value
        ]
        return {**obj, key: renamed}

    out = with_entries(payload, "migration_history")
    components = out.get("components")
    if isinstance(components, dict):
        out = {**out, "components": {
            name: with_entries(comp, "history") if isinstance(comp, dict) else comp
            for name, comp in components.items()
        }}
    return out


class ModelState(VersionedJsonFile):
    """``model_state.json`` (``PathResolver`` の ``model_state_file``) の読み書き管理 (封筒付き、c_05 §0.7.1 ``model_state``)。

    G1 の封筒でないファイル (G0 の素の JSON) / 新しい版は読まずに readonly で
    続け、``save`` はファイルを書き換えない。壊れたファイルは
    ``model_state.json.corrupt-<stamp>`` へ退避して空の状態から始める。
    保存の失敗 (OSError 等) は従来どおり送出する。
    """

    FORMAT = MODEL_STATE_FORMAT
    RAISE_ON_SAVE_ERROR = True
    _state_logger = logger

    def __init__(self, state_path: Path):
        super().__init__(state_path)
        self._current = ModelCurrent()
        self._migration_history: list[MigrationHistoryEntry] = []
        self._components: dict[str, ComponentState] = {
            name: ComponentState() for name in ALL_COMPONENTS
        }
        #: payload の未知キー (書き戻しでそのまま戻す)。
        self._extra: dict[str, Any] | None = None
        self.load()

    # ── プロパティ ──

    @property
    def current_filename(self) -> str:
        return self._current.filename

    @property
    def current(self) -> ModelCurrent:
        return self._current

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

    def add_component_migration(self, name: str, from_model: str, to_model: str) -> None:
        comp = self.get_component(name)
        comp.history.append(MigrationHistoryEntry(
            from_model=from_model,
            to_model=to_model,
            migrated_at=_now(),
        ))

    def get_component_last_migration(
        self, name: str,
    ) -> MigrationHistoryEntry | None:
        comp = self.get_component(name)
        return comp.history[-1] if comp.history else None

    # ── 永続化 (VersionedJsonFile) ──

    def _from_payload(self, payload: Any) -> None:
        """封筒の payload から状態を復元する (形が違えば送出して corrupt 扱い)。

        知らないコンポーネントも未知キーと同じく持ち続けて書き戻す。
        """
        data = _PAYLOAD_CODEC.decode(_rename_history(payload, _HISTORY_KEYS))
        for name in ALL_COMPONENTS:
            data.components.setdefault(name, ComponentState())
        self._current = data.current
        self._migration_history = data.migration_history
        self._components = data.components
        self._extra = data._extra

    def _to_payload(self) -> dict[str, Any]:
        payload = _PAYLOAD_CODEC.encode(ModelStatePayload(
            current=self._current,
            migration_history=self._migration_history,
            components=self._components,
            _extra=self._extra,
        ))
        return _rename_history(payload, {v: k for k, v in _HISTORY_KEYS.items()})

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

    def add_migration(self, from_model: str, to_model: str) -> None:
        """移行履歴にエントリを追加"""
        self._migration_history.append(MigrationHistoryEntry(
            from_model=from_model,
            to_model=to_model,
            migrated_at=_now(),
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
        learning_scheduler=None,
        episodic_memory=None,
        vector_store=None,
        cartridge_manager=None,
    ):
        self.config = config
        self.project_root = project_root
        self.model_state = model_state
        self.experience_buf = experience_buf
        self.prompt_manager = prompt_manager
        self.learning_scheduler = learning_scheduler
        self.episodic_memory = episodic_memory
        self._vector_store = vector_store
        self._cartridge_manager = cartridge_manager

    def migrate(
        self,
        new_model_path: str,
        *,
        regenerate_context: bool = False,
        dry_run: bool = False,
    ) -> MigrationResult:
        """移行を実行（§22.4.2 Step 1〜9）

        Args:
            new_model_path: 新モデルの GGUF ファイルパス
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

        # 学習データ (LoRA・経験・プロンプト) は model_key ごとのパーティションで
        # 保全される (c_05 §0.5.12)。旧モデルのものを退避・初期化する処理は無い —
        # 新しいモデルのパーティションは起動時 / rebind で束ねられ、空なら一から学習する。
        result.lora_action = "kept"
        if dry_run:
            result.recommendations = (
                self._known_issue_recommendations(resolved_path)
                + self._build_recommendations(dry_run=True)
            )
            return result

        # Step 7 (部分): config.yaml 更新
        self._update_config(new_model_path)

        # Step 8: メモリノート処理（オプション）
        if regenerate_context:
            self._mark_context_regeneration()

        # Step 9: model_state 更新
        self.model_state.add_migration(
            from_model=old_model_filename,
            to_model=new_model_filename,
        )
        self.model_state.update_current(filename=new_model_filename)
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
            old_model_filename, new_model_filename, result.lora_action,
        )
        return result

    def rollback(self, target_model: str | None = None) -> dict:
        """ロールバック処理（§22.6）

        Args:
            target_model: ロールバック先モデル名。省略時は直前の移行元

        Returns:
            {"rolled_back_to": str}

        Raises:
            MigrationError: 履歴なし / ロールバック不能
        """
        last = self.model_state.get_last_migration()
        if last is None:
            raise MigrationError("No migration history found")

        rollback_target = target_model or last.from_model
        if not rollback_target:
            raise MigrationError("Cannot determine rollback target model")

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
        self.model_state.save()

        logger.info("Rollback completed: -> %s", rollback_target)
        return {"rolled_back_to": rollback_target}

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
        埋め込みモデルは LoRA を持たない (embed LoRA は G1 で撤去) ので
        ``lora_action`` は常に ``"n/a"``。config.yaml 更新と model_state 記録を行う。
        実際の llama-server 再起動とクライアント差し替えは L2 で対応する。
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

        # model_state 更新
        self.model_state.add_component_migration(component, old_filename, new_filename)
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

        self.model_state.update_component_current(component, rollback_target)
        self.model_state.save()

        logger.info(
            "Component rollback completed: %s -> %s", component, rollback_target,
        )
        return {"rolled_back_to": rollback_target}

    def _get_component_current(self, component: str) -> str:
        cur = self.model_state.get_component_current_filename(component)
        if cur:
            return cur
        cfg_key = COMPONENT_CONFIG_KEY[component]
        raw = self.config.get("model_paths", {}).get(cfg_key, "")
        return Path(raw).name if raw else "unknown"

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
            "injection_relevance_min_score",
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
        # 注入ゲートは rag ではなく memory.injection が実体 (schemas/memory.py)。
        "injection_relevance_min_score": (
            "memory", "injection", "relevance_min_score",
        ),
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
        "injection_relevance_min_score",
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
                "POST /api/model/reembed-facts を実行してください。",
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

    def _gather_data_summary(self) -> dict:
        """移行対象データの集計

        学習データは model_key のパーティションで保全され移行で壊さないので、
        破壊対象カウント (perplexity_reset / prompts_modes) は常に 0 / 空。
        experience_entries / memory_notes / rag_chunks / cartridges は保持
        データ件数の情報表示。
        """
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

        if self.episodic_memory is not None:
            summary["memory_notes"] = len(self.episodic_memory)

        # RAG / カートリッジは非依存データのため集計のみ
        if self._vector_store:
            summary["rag_chunks"] = self._vector_store.count
        if self._cartridge_manager:
            summary["cartridges"] = len(getattr(self._cartridge_manager, "installed", {}))

        return summary

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
        """Step 8: ノートの ``context_description`` 再生成マークを立てる。

        ``context_description`` はモデルが書いた要約なので、ベースモデルを
        替えたら作り直す。印は ``attrs.evolution_pending`` への ``patch`` で、
        次の sleep-time Step 7 が拾う。
        """
        episodic = self.episodic_memory
        if episodic is None:
            return

        count = 0
        for note in episodic.iter_notes(tier="short"):
            if not note.context_description:
                continue
            attrs = {"evolution_pending": True}
            if episodic.patch_note(note.id, attrs=attrs) is not None:
                count += 1

        if count > 0:
            logger.info(
                "Marked %d notes for context regeneration", count,
            )

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


