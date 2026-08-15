"""base 学習データの flat→(model×mode) パーティション 一度きり非破壊移行。

``learning.partition_by_base_model`` 有効化に伴い、既存の flat レイアウト
(``local/experience.json`` / ``local/prompts/{mode}.md`` / ``local/models/adapter.gguf``
等) を、現在の base モデルのパーティション ``local/learning/<stem>/`` 配下へ
**コピー** する (原本は残置 = ロールバック可能 / 非破壊)。マーカー
``local/learning/.partition_migrated_v1`` で一度だけ実行する。

設計上の約束:
- experience エントリは ``base_model`` タグ別にバケツ分けし、別モデルのエントリは
  そのモデルのパーティションへ振り分ける (クロスモデル履歴を非破壊保持)。
- LoRA / 制御ベクトル / learning_state は mode-blind なのでモデルルート直下。
- ``--develop=evolve`` 中は LogIngestor の inode 整合のため移行を見送る (p_03 §9.4)。

Pro 型を import しない純ファイル操作 (Free-safe)。Pro の LoRA / cvector も
存在すればコピーする。SemMem ``learn.*`` ファクトの再キーは SemMem 構築後に
別途行う (:meth:`rekey_semmem_learn_facts`)。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from backend.log_config import get_logger
from backend.utils import utc_now

if TYPE_CHECKING:
    from backend.config import PathResolver

logger = get_logger("core.learning_partition_migrator")

MARKER_NAME = ".partition_migrated_v1"
# level2_adapter_partition=="model_mode" を初めて有効化した際の
# 「既存 (model のみ) パーティション済みアダプタ → "chat" バケット」移行マーカー。
# MARKER_NAME (flat→partition) とは独立したライフサイクル: partition_by_base_model
# は起動時から有効な可能性が高いが、level2_adapter_partition はユーザーが後から
# "model" → "model_mode" に切り替えるケースがあり、発火タイミングが異なるため。
ADAPTER_MODE_MARKER_NAME = ".adapter_mode_migrated_v1"

# flat prompts_dir から base パーティションへ移すファイル名。
_BASE_PROMPT_FILES = (
    "chat.md", "chat.meta.json",
    "create.md", "create.meta.json",
    "learning_state.json",
    "level1_session_active.json",
    "exploration_state.json",
    "policy_evolver_state.json",
    "fewshot_pool.json",
    "generation_param_ratios.json",
)
# 同じく移すサブディレクトリ。
_BASE_PROMPT_SUBDIRS = ("level1_history",)
# history/ 配下で移す base プロンプト版。
_BASE_HISTORY_GLOBS = ("chat_v*.md", "create_v*.md")


class LearningPartitionMigrator:
    """flat→パーティション 一度きり非破壊移行。"""

    def __init__(
        self,
        resolver: "PathResolver",
        cfg: dict[str, Any],
        *,
        develop_level: str | None = None,
    ) -> None:
        self._resolver = resolver
        self._cfg = cfg
        self._develop_level = develop_level

    # ── マーカー ─────────────────────────────────────────

    def _marker_path(self) -> Path:
        return self._resolver.resolve_local("learning_dir") / MARKER_NAME

    def already_migrated(self) -> bool:
        return self._marker_path().exists()

    def _write_marker(self, stem: str, counts: dict[str, int]) -> None:
        marker = self._marker_path()
        marker.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "migrated_at": utc_now(),
            "model_stem": stem,
            "counts": counts,
            "develop_level": self._develop_level or "off",
        }
        marker.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ── メイン ───────────────────────────────────────────

    def migrate_if_needed(self, producer_filename: str) -> bool:
        """必要なら flat データを **producer** モデルのパーティションへコピーする。

        producer = flat データを生成したモデル (``ModelState.current_filename``、
        前回起動モデル)。active (config) モデルとは異なりうる: config を新モデルへ
        変えた直後の初回起動では、flat データは旧 producer のパーティションへ移り、
        active(新) パーティションは空のまま = ゼロから学習になる (仕様通り)。
        experience は各エントリの ``base_model`` タグ別に振り分ける。

        Args:
            producer_filename: flat データの生成元 base モデル GGUF ファイル名
                (空なら config base_model 名)。

        Returns:
            移行を実行したら ``True``、スキップ (無効 / 済 / evolve) なら ``False``。
        """
        if not self._resolver.partition_enabled:
            return False
        if self.already_migrated():
            return False
        if self._develop_level == "evolve":
            logger.warning(
                "Learning partition migration skipped during --develop=evolve "
                "(deferred to next normal startup to keep LogIngestor inode stable)",
            )
            return False

        stem = Path(producer_filename).stem if producer_filename else ""
        if not stem:
            base_model = self._cfg.get("model_paths", {}).get("base_model") or ""
            stem = Path(base_model).stem
        if not stem:
            logger.warning(
                "Learning partition migration skipped: no resolvable base model stem",
            )
            return False

        counts: dict[str, int] = {}
        try:
            counts["experience"] = self._migrate_experience(stem)
            counts["prompts"] = self._migrate_base_prompts(stem)
            counts["adapters"] = self._migrate_mode_blind_artifacts(stem)
        except Exception as exc:
            # 移行失敗で起動を止めない。マーカーを書かず次回再試行する。
            logger.warning("Learning partition migration failed (will retry): %s", exc)
            return False

        self._write_marker(stem, counts)
        logger.info(
            "Learning partition migration complete: stem=%s counts=%s "
            "(flat originals retained for rollback)",
            stem, counts,
        )
        return True

    # ── experience (model 別バケツ分け) ─────────────────

    def _migrate_experience(self, current_stem: str) -> int:
        """flat experience.json を base_model タグ別にバケツ分けしてコピーする。

        各エントリの ``base_model`` (GGUF ファイル名) の stem でパーティションを
        決め、未タグは現行 stem に倒す。mode タグはエントリ内に保持したまま。
        """
        src = self._resolver.resolve_local("experience_file")
        if not src.exists():
            return 0
        try:
            payload = json.loads(src.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("experience migration: unreadable %s: %s", src, exc)
            return 0
        entries = payload.get("entries") if isinstance(payload, dict) else None
        if not isinstance(entries, list):
            return 0

        buckets: dict[str, list] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            tag = str(entry.get("base_model") or "" or "")
            stem = Path(tag).stem if tag else current_stem
            buckets.setdefault(stem or current_stem, []).append(entry)

        written = 0
        for stem, items in buckets.items():
            dst = self._resolver.learning_path_for("experience_file", stem)
            if dst.exists():
                continue  # 既にパーティションに存在 = 冪等スキップ
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(
                json.dumps({"entries": items}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            written += len(items)
        return written

    # ── base プロンプト + 派生状態 ──────────────────────

    def _migrate_base_prompts(self, stem: str) -> int:
        """flat ``local/prompts/`` の base プロンプト + 派生状態をコピーする。

        ``{mode}.md`` / ``.meta.json`` / learning_state / L1 セッション /
        exploration / policy_evolver / fewshot_pool / ratios と
        ``history/{mode}_v*.md`` / ``level1_history/`` を対象。
        """
        src_dir = self._resolver.resolve_local("prompts_dir")
        dst_dir = self._resolver.learning_path_for("prompts_dir", stem)
        if not src_dir.exists() or src_dir.resolve() == dst_dir.resolve():
            return 0
        copied = 0
        dst_dir.mkdir(parents=True, exist_ok=True)

        for name in _BASE_PROMPT_FILES:
            s = src_dir / name
            d = dst_dir / name
            if s.is_file() and not d.exists():
                shutil.copy2(s, d)
                copied += 1

        for sub in _BASE_PROMPT_SUBDIRS:
            s = src_dir / sub
            d = dst_dir / sub
            if s.is_dir() and not d.exists():
                shutil.copytree(s, d)
                copied += 1

        # history/{mode}_v*.md (base プロンプト版のみ)
        src_hist = src_dir / "history"
        if src_hist.is_dir():
            dst_hist = dst_dir / "history"
            for pattern in _BASE_HISTORY_GLOBS:
                for s in src_hist.glob(pattern):
                    d = dst_hist / s.name
                    if s.is_file() and not d.exists():
                        d.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(s, d)
                        copied += 1
        return copied

    # ── mode-blind: LoRA / 制御ベクトル ─────────────────

    def _migrate_mode_blind_artifacts(self, stem: str) -> int:
        """base LoRA / 制御ベクトル / cvector 作業ディレクトリをコピーする。

        いずれも mode-blind なのでモデルルート直下 (``models/`` / ``cvector/``)。
        Pro 未配置なら原本が無く no-op。
        """
        copied = 0
        for key in (
            "lora_adapter",
            "lora_versions_dir",
            "control_vector_adapter",
            "control_vector_versions_dir",
            "cvector_work_dir",
        ):
            src = self._resolver.resolve_local(key)
            dst = self._resolver.learning_path_for(key, stem)
            if src.resolve() == dst.resolve():
                continue
            if src.is_file() and not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                copied += 1
            elif src.is_dir() and not dst.exists():
                shutil.copytree(src, dst)
                copied += 1
        return copied

    # ── adapter mode-partition (chat/create 分離) 初回移行 ──

    def _adapter_mode_marker_path(self) -> Path:
        return self._resolver.resolve_local("learning_dir") / ADAPTER_MODE_MARKER_NAME

    def adapter_mode_already_migrated(self) -> bool:
        return self._adapter_mode_marker_path().exists()

    def migrate_adapter_partition_mode_if_needed(self, stem: str) -> bool:
        """``level2_adapter_partition=="model_mode"`` 初回有効化時の非破壊移行。

        既存の (model のみ) パーティション済み base LoRA を "chat" バケットへ
        コピーする(原本は残置)。既定モードが "chat" (``AppState.current_mode`` /
        Free CLI 既定と一致) なので、既存の共有アダプタは主に chat セッションの
        経験を反映していると見なすのが最も実態に近い、という前提。

        ``level2_adapter_partition=="model_mode"`` はスキーマ側で
        ``partition_by_base_model=true`` を要求しているため、コピー元は常に
        "model" スキームのパーティション済みパス
        (``learning_path_for(key, stem)``、mode 無し) になる — flat レイアウトからの
        直接移行は考慮不要 (既存の :meth:`migrate_if_needed` が別途担う領域)。
        """
        if self._resolver.adapter_partition_mode != "model_mode":
            return False
        if self.adapter_mode_already_migrated():
            return False

        copied = 0
        for key in ("lora_adapter", "lora_versions_dir", "lora_spsa_checkpoint"):
            src = self._resolver.learning_path_for(key, stem)
            dst = self._resolver.learning_path_for(key, stem, mode="chat")
            copied += self._copy_if_missing(src, dst)

        self._write_adapter_mode_marker(stem, copied)
        logger.info(
            "Adapter mode-partition migration complete: stem=%s copied=%d "
            "(model-partitioned/flat originals retained for rollback)",
            stem, copied,
        )
        return True

    @staticmethod
    def _copy_if_missing(src: Path, dst: Path) -> int:
        if src.resolve() == dst.resolve():
            return 0
        if src.is_file() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            return 1
        if src.is_dir() and not dst.exists():
            shutil.copytree(src, dst)
            return 1
        return 0

    def _write_adapter_mode_marker(self, stem: str, copied: int) -> None:
        marker = self._adapter_mode_marker_path()
        marker.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "migrated_at": utc_now(),
            "model_stem": stem,
            "target_mode": "chat",
            "copied": copied,
        }
        marker.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
