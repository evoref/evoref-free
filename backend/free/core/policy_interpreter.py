"""PolicyInterpreter: ポリシーファイルによるパラメータ外部化

ハードコード閾値・config 固定パラメータを JSON ポリシーファイルに
外部化し、O(1) の dict lookup で参照する仕組みを提供する。

LLM 不要。推論パスへの追加遅延: < 0.1ms。
Stage 4（PolicyEvolver）がこの外部化を前提とする。
"""

from __future__ import annotations

import copy
import json
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from backend.io import atomic_write_text
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.debug_logger import DebugLogger
    from backend.free.memory.semantic.store import SemanticFactStore

logger = get_logger("core.policy_interpreter")

# SemMem subject 先頭プレフィックス。owner pillar は EvorefLearn。
LEARN_POLICY_SUBJECT_PREFIX = "learn.policy."

PolicySource = Literal["yaml", "semmem", "hybrid"]

# ポリシーファイル名 → ドメイン名のマッピング
_POLICY_FILES: dict[str, str] = {
    "router_policy.json": "router",
    "memory_policy.json": "memory",
    "search_policy.json": "search",
    "agent_policy.json": "agent",
    "long_form_policy.json": "long_form",
    "learning_policy.json": "learning",
}

# 型変換関数
_TYPE_CONVERTERS: dict[str, type] = {
    "int": int,
    "float": float,
    "bool": bool,
    "str": str,
}


def _default_policies() -> dict[str, dict]:
    """全ドメインのデフォルトポリシー定義を返す"""
    return {
        "router": {
            "version": 1,
            "domain": "router",
            "params": {
                "chat": {
                    "rag_score_threshold": 0.8,
                    "short_query_min_tokens": 3,
                    "short_query_max_tokens": 10,
                    "short_query_max_chars": 20,
                },
                "create": {
                    "rag_score_threshold": 0.8,
                    "short_query_min_tokens": 3,
                    "short_query_max_tokens": 10,
                    "short_query_max_chars": 20,
                },
            },
            "constraints": {
                "rag_score_threshold": {"min": 0.0, "max": 1.0, "type": "float"},
                "short_query_min_tokens": {"min": 1, "max": 20, "type": "int"},
                "short_query_max_tokens": {"min": 2, "max": 50, "type": "int"},
                "short_query_max_chars": {"min": 5, "max": 100, "type": "int"},
            },
        },
        "memory": {
            "version": 1,
            "domain": "memory",
            "params": {
                "chat": {
                    "fade_alpha": 0.4,
                    "fade_beta": 0.3,
                    "fade_gamma": 0.3,
                    "fade_threshold": 0.15,
                    "decay_days": 7,
                    "eviction_ratio": 0.2,
                    "conflict_similarity_threshold": 0.85,
                    "conflict_batch_size": 5,
                },
                "create": {
                    "fade_alpha": 0.4,
                    "fade_beta": 0.3,
                    "fade_gamma": 0.3,
                    "fade_threshold": 0.15,
                    "decay_days": 7,
                    "eviction_ratio": 0.2,
                    "conflict_similarity_threshold": 0.85,
                    "conflict_batch_size": 5,
                },
            },
            "constraints": {
                "fade_alpha": {"min": 0.0, "max": 1.0, "type": "float"},
                "fade_beta": {"min": 0.0, "max": 1.0, "type": "float"},
                "fade_gamma": {"min": 0.0, "max": 1.0, "type": "float"},
                "fade_threshold": {"min": 0.0, "max": 1.0, "type": "float"},
                "decay_days": {"min": 1, "max": 365, "type": "int"},
                "eviction_ratio": {"min": 0.01, "max": 0.5, "type": "float"},
                "conflict_similarity_threshold": {"min": 0.0, "max": 1.0, "type": "float"},
                "conflict_batch_size": {"min": 1, "max": 50, "type": "int"},
            },
        },
        "search": {
            "version": 1,
            "domain": "search",
            "params": {
                "chat": {
                    "stm_top_k": 3,
                    "noise_sigma": 0.05,
                    "bm25_weight": 0.3,
                    "vector_weight": 0.7,
                    "rrf_k": 60,
                    "top_k": 5,
                    "candidates_multiplier": 3,
                    "rescore_candidates": 50,
                    "hybrid_search": True,
                    "salience_w_query_relevance": 0.35,
                    "salience_w_tfidf": 0.20,
                    "salience_w_entity_density": 0.15,
                    "salience_w_info_density": 0.20,
                    "salience_w_position_bias": 0.10,
                },
                "create": {
                    "stm_top_k": 5,
                    "noise_sigma": 0.03,
                    "bm25_weight": 0.4,
                    "vector_weight": 0.6,
                    "rrf_k": 60,
                    "top_k": 5,
                    "candidates_multiplier": 3,
                    "rescore_candidates": 50,
                    "hybrid_search": True,
                    "salience_w_query_relevance": 0.35,
                    "salience_w_tfidf": 0.20,
                    "salience_w_entity_density": 0.15,
                    "salience_w_info_density": 0.20,
                    "salience_w_position_bias": 0.10,
                },
            },
            "constraints": {
                "stm_top_k": {"min": 1, "max": 20, "type": "int"},
                "noise_sigma": {"min": 0.0, "max": 0.5, "type": "float"},
                "bm25_weight": {"min": 0.0, "max": 1.0, "type": "float"},
                "vector_weight": {"min": 0.0, "max": 1.0, "type": "float"},
                "rrf_k": {"min": 1, "max": 200, "type": "int"},
                # max=50 だと L1 進化が品質向上なしに取得件数だけ膨張させる
                # (2026-07-15: top_k=13 まで学習しコンテキストを圧迫)。
                # iGPU の prefill 速度とコンテキスト予算に対し 10 を上限とする。
                "top_k": {"min": 1, "max": 10, "type": "int"},
                "candidates_multiplier": {"min": 1, "max": 10, "type": "int"},
                "rescore_candidates": {"min": 0, "max": 200, "type": "int"},
                "hybrid_search": {"type": "bool"},
                "salience_w_query_relevance": {"min": 0.0, "max": 1.0, "type": "float"},
                "salience_w_tfidf": {"min": 0.0, "max": 1.0, "type": "float"},
                "salience_w_entity_density": {"min": 0.0, "max": 1.0, "type": "float"},
                "salience_w_info_density": {"min": 0.0, "max": 1.0, "type": "float"},
                "salience_w_position_bias": {"min": 0.0, "max": 1.0, "type": "float"},
            },
        },
        "agent": {
            "version": 1,
            "domain": "agent",
            "params": {
                "chat": {
                    "step_compaction_rag_lines": 2,
                    "step_compaction_command_head_tail": 5,
                    "file_skeleton_threshold": 30,
                    "meta_cognitive_min_budget": 512,
                },
                "create": {
                    "step_compaction_rag_lines": 2,
                    "step_compaction_command_head_tail": 5,
                    "file_skeleton_threshold": 30,
                    "meta_cognitive_min_budget": 512,
                },
            },
            "constraints": {
                "step_compaction_rag_lines": {"min": 1, "max": 20, "type": "int"},
                "step_compaction_command_head_tail": {"min": 1, "max": 30, "type": "int"},
                "file_skeleton_threshold": {"min": 5, "max": 200, "type": "int"},
                "meta_cognitive_min_budget": {"min": 128, "max": 4096, "type": "int"},
            },
        },
        "long_form": {
            "version": 1,
            "domain": "long_form",
            "params": {
                "chat": {
                    "unit_max_tokens": 2000,
                    "unit_target_tokens": 800,
                    "extend_threshold_ratio": 0.7,
                    "max_extend_rounds": 10,
                },
                "create": {
                    "unit_max_tokens": 2000,
                    "unit_target_tokens": 800,
                    "extend_threshold_ratio": 0.7,
                    "max_extend_rounds": 10,
                },
            },
            "constraints": {
                "unit_max_tokens": {"min": 256, "max": 8192, "type": "int"},
                # min=128 だと L1 進化が下限に張り付き、parse_plan の 200 トークン
                # クランプと組み合わさって全ユニットが機械的に倍分割される
                # (2026-07-15: 13→26 units / 464 秒)。512 を床にする。
                "unit_target_tokens": {"min": 512, "max": 4096, "type": "int"},
                "extend_threshold_ratio": {"min": 0.1, "max": 1.0, "type": "float"},
                "max_extend_rounds": {"min": 1, "max": 50, "type": "int"},
            },
        },
        "learning": {
            "version": 1,
            "domain": "learning",
            "params": {
                "_default": {
                    "level1_min_experiences": 20,
                    "level1_generations": 10,
                    "level1_population_size": 5,
                    "level1_idle_minutes": 30,
                },
            },
            "constraints": {
                "level1_min_experiences": {"min": 1, "max": 200, "type": "int"},
                "level1_generations": {"min": 1, "max": 100, "type": "int"},
                "level1_population_size": {"min": 2, "max": 50, "type": "int"},
                "level1_idle_minutes": {"min": 1, "max": 1440, "type": "int"},
            },
        },
    }


def _clamp(value: int | float, constraint: dict) -> int | float:
    """制約に基づいて値をクランプする"""
    ctype = constraint.get("type", "float")
    converter = _TYPE_CONVERTERS.get(ctype, float)

    if ctype == "bool":
        return bool(value)

    result = converter(value)
    if "min" in constraint:
        result = max(converter(constraint["min"]), result)
    if "max" in constraint:
        result = min(converter(constraint["max"]), result)
    return result


class PolicyInterpreter:
    """ポリシーファイルによるパラメータの O(1) 参照

    起動時に local/policies/*.json を読み込み、メモリ上に保持する。
    get() は dict lookup のみで推論パスから安全に呼べる。
    apply_delta() / rollback() は Stage 4（PolicyEvolver）で使用する。
    """

    def __init__(
        self,
        policies_dir: str | Path,
        *,
        policy_source: PolicySource = "yaml",
        semmem_stores: list[SemanticFactStore] | None = None,
        policy_activation_min_confidence: float = 0.7,
        debug_logger: "DebugLogger | None" = None,
        base_model_id: str = "",
    ):
        """
        Args:
            policies_dir: ``local/policies/*.json`` のディレクトリ
            policy_source: ``yaml`` (従来動作) / ``hybrid`` (YAML seed +
                SemMem 上書き) / ``semmem`` (導入予定。現状は
                ``hybrid`` と同等動作)。デフォルトは ``yaml`` で、既存テスト
                および従来挙動を破壊しない
            semmem_stores: SemMem 上の active な ``policy`` ファクトを供給
                するストア群。複数渡された場合は順番に適用し、後勝ちで上書き
                する (典型: ``[global_store, project_store]`` を渡してプロ
                ジェクトスコープを優先させる)。``hybrid`` / ``semmem`` 時
                のみ利用される。
            policy_activation_min_confidence: SemMem 上の policy ファクトを
                active と見なす最小 ``confidence``。これ未満は YAML 値が
                温存される
                policy ファクトの active 判定境界 (decision_point=
                ``policy_activation_boundary``) を ``decision.jsonl`` に
                記録する。``evolve`` レベル限定で実発火、それ以外は no-op。
        """
        self._policies_dir = Path(policies_dir)
        self._lock = threading.Lock()

        # ドメイン → { params: {mode: {key: value}}, constraints: {...}, ... }
        self._data: dict[str, dict] = {}

        # ロールバック用スナップショット: (domain, mode) → params snapshot
        self._snapshots: dict[tuple[str, str], dict] = {}

        # SemMem ローダ設定
        self._policy_source: PolicySource = policy_source
        self._semmem_stores: list[SemanticFactStore] = list(semmem_stores or [])
        self._policy_activation_min_confidence = float(
            policy_activation_min_confidence,
        )

        self._debug_logger = debug_logger
        # base 学習パーティションの active モデルスラグ。空 = partition 無効
        # (``learn.policy.<mode>.*`` レガシー subject を適用)。set_base_model_id で
        # 切替時に更新し、当該モデルの policy ファクトのみを適用する。
        self._base_model_id: str = base_model_id

        self._load_all()
        if self._policy_source != "yaml":
            self._apply_semmem_overrides()

    def set_base_model_id(self, base_model_id: str) -> None:
        """base 学習パーティションの active モデルスラグを差し替えて再適用する。

        モデル切替時に :func:`backend.factory._learning_rebind.rebind_base_learning`
        から呼ばれる。``set_semmem_stores`` と同様、変更後に即
        :meth:`_apply_semmem_overrides` を再走させ、前モデルの override を
        当該モデルのもので置き換える (YAML 再ロード後の呼び出しが前提)。
        """
        with self._lock:
            self._base_model_id = base_model_id or ""
            if self._policy_source != "yaml":
                self._apply_semmem_overrides()

    def _active_policy_prefix(self) -> str:
        """active モデルを織り込んだ policy ファクト検索 prefix を返す。

        partition 有効 (``_base_model_id`` 非空) 時は
        ``learn.policy.<model>.`` を返し、当該モデルの policy のみを対象にする。
        無効時はレガシー ``learn.policy.`` (全モデル) を返す。
        """
        if self._base_model_id:
            return f"{LEARN_POLICY_SUBJECT_PREFIX}{self._base_model_id}."
        return LEARN_POLICY_SUBJECT_PREFIX

    def get(self, domain: str, key: str, mode: str = "chat") -> Any:
        """パラメータ値を O(1) で取得する

        Args:
            domain: ポリシードメイン ("router", "memory", "search", ...)
            key: パラメータ名
            mode: "chat" | "create"（learning ドメインは無視される）

        Returns:
            パラメータ値

        Raises:
            KeyError: ドメインまたはキーが存在しない場合
        """
        policy = self._data.get(domain)
        if policy is None:
            raise KeyError(f"Unknown policy domain: {domain}")

        params = policy.get("params", {})

        # learning ドメインはモード区別なし（_default キーを使用）
        if "_default" in params:
            mode_params = params["_default"]
        else:
            mode_params = params.get(mode)
            if mode_params is None:
                raise KeyError(f"Unknown mode '{mode}' in domain '{domain}'")

        if key not in mode_params:
            raise KeyError(f"Unknown key '{key}' in domain '{domain}' mode '{mode}'")

        return mode_params[key]

    def get_all(self, domain: str, mode: str = "chat") -> dict:
        """指定ドメイン・モードの全パラメータを返す"""
        policy = self._data.get(domain)
        if policy is None:
            raise KeyError(f"Unknown policy domain: {domain}")

        params = policy.get("params", {})
        if "_default" in params:
            return dict(params["_default"])
        mode_params = params.get(mode, {})
        return dict(mode_params)

    def get_constraints(self, domain: str) -> dict:
        """指定ドメインの constraints を返す (読み取り専用想定の浅いコピー)。

        PolicyEvolver の摂動範囲算出用。``_data`` 直アクセスを避けるための公開 API。
        """
        policy = self._data.get(domain)
        if policy is None:
            raise KeyError(f"Unknown policy domain: {domain}")
        return dict(policy.get("constraints", {}))

    def restore_params(self, domain: str, snapshot: dict, mode: str = "chat") -> None:
        """与えた params スナップショットで当該 mode を丸ごと置換する。

        PolicyEvolver が best_fitness 更新時に取得した params で復元するための API。
        単段の apply_delta スナップショット (``_snapshots``) は当該キーを破棄し、
        二重管理を残さない (``rollback`` と同じ ``_default`` モード分岐を踏襲)。
        """
        with self._lock:
            policy = self._data.get(domain)
            if policy is None:
                raise KeyError(f"Unknown policy domain: {domain}")
            params = policy["params"]
            actual_mode = "_default" if "_default" in params else mode
            params[actual_mode] = copy.deepcopy(snapshot)
            self._snapshots.pop((domain, mode), None)
            logger.info(
                "Policy params restored to best snapshot: domain=%s, mode=%s",
                domain, mode,
            )

    def reload(self) -> None:
        """ポリシーファイルを再読み込みする。

        ``policy_source`` が ``hybrid`` / ``semmem`` の場合は YAML 再ロード後に
        SemMem の active な ``policy`` ファクトで再上書きする。
        """
        with self._lock:
            self._load_all()
            if self._policy_source != "yaml":
                self._apply_semmem_overrides()
        logger.info(
            "Policies reloaded from %s (source=%s)",
            self._policies_dir, self._policy_source,
        )

    # ── SemMem ローダ ────────────────────────────────

    @property
    def policy_source(self) -> PolicySource:
        """現在の policy_source モード"""
        return self._policy_source

    def set_semmem_stores(self, stores: list[SemanticFactStore]) -> None:
        """SemMem ストアを差し替えて即座に再上書きする (テスト・動的更新用)"""
        with self._lock:
            self._semmem_stores = list(stores)
            if self._policy_source != "yaml":
                self._apply_semmem_overrides()

    def _apply_semmem_overrides(self) -> int:
        """SemMem 上の active な ``learn.policy.*`` ファクトで YAML を上書きする。

        ストアは ``self._semmem_stores`` の順番で適用し、後勝ちで上書きする
        (典型: ``[global, project]`` でプロジェクトスコープを優先させる)。

        評価対象のファクトは以下の条件を満たすものに限る:
        - ``type == "policy"``
        - ``superseded_by is None`` (active = supersession チェーンの末端)
        - ``confidence >= policy_activation_min_confidence``
        - ``subject`` が ``learn.policy.<mode>.<domain>.<param_path>`` の形式

        ``object`` フィールドは JSON 値としてパースする (``"5"`` → ``5``、
        ``"true"`` → ``True``、文字列リテラルは ``"\"foo\""`` のように JSON
        エンコードしておく)。パースに失敗した場合は raw 文字列としてフォール
        バック適用する。型は ``constraints`` に基づき clamp する。

        Returns:
            適用に成功した override 件数 (デバッグ・テスト用)
        """
        if not self._semmem_stores:
            logger.debug("SemMem stores empty, no override applied")
            return 0

        applied = 0
        skipped_unknown_subject = 0
        skipped_below_confidence = 0
        skipped_unknown_target = 0

        # partition 有効時は active モデルの policy のみを対象にする
        # (``learn.policy.<model>.``)。他モデルの override が混入すると分割が
        # 形骸化するため、検索 prefix 自体でモデルをゲートする。
        search_prefix = self._active_policy_prefix()

        # スコア順ではなく、ストアの順番 → ファクトの created_at 順で適用する
        # ことで、後勝ちの規則を保ちつつ各ストア内の進化履歴も時系列で反映する
        for store in self._semmem_stores:
            facts = store.search_by_pillar_prefix(
                search_prefix, include_superseded=False,
            )
            facts.sort(key=lambda f: f.created_at)
            for fact in facts:
                if fact.type != "policy":
                    continue
                if fact.confidence < self._policy_activation_min_confidence:
                    skipped_below_confidence += 1
                    continue
                parsed = self._parse_learn_policy_subject(
                    fact.subject, self._base_model_id,
                )
                if parsed is None:
                    skipped_unknown_subject += 1
                    continue
                mode, domain, key = parsed
                if not self._apply_single_override(domain, mode, key, fact.object):
                    skipped_unknown_target += 1
                    continue
                applied += 1

        logger.info(
            "PolicyInterpreter SemMem overrides: applied=%d "
            "skipped_below_confidence=%d skipped_unknown_subject=%d "
            "skipped_unknown_target=%d (source=%s, min_conf=%.2f)",
            applied,
            skipped_below_confidence,
            skipped_unknown_subject,
            skipped_unknown_target,
            self._policy_source,
            self._policy_activation_min_confidence,
        )
        # に記録 (active vs inactive の境界遷移ではなく、サイクル全体の判定結果サマリ)。
        if (
            self._debug_logger is not None
            and (applied > 0 or skipped_below_confidence > 0)
        ):
            self._debug_logger.log_decision(
                decision_point="policy_activation_boundary",
                chosen="active" if applied > 0 else "inactive",
                candidates=["active", "inactive"],
                reason="semmem_override_evaluation",
                context={
                    "applied": applied,
                    "skipped_below_confidence": skipped_below_confidence,
                    "skipped_unknown_subject": skipped_unknown_subject,
                    "skipped_unknown_target": skipped_unknown_target,
                    "policy_source": self._policy_source,
                    "min_confidence": self._policy_activation_min_confidence,
                },
                scope="cycle",
            )
        return applied

    @staticmethod
    def _parse_learn_policy_subject(
        subject: str,
        base_model_id: str = "",
    ) -> tuple[str, str, str] | None:
        """policy subject を ``(mode, domain, key)`` に分解する。

        partition 有効 (``base_model_id`` 非空) 時は新形式
        ``learn.policy.<model>.<mode>.<domain>.<param_path>`` を期待し、モデル
        セグメントを剥がしてから ``<mode>.<domain>.<key>`` を読む (モデル不一致は
        ``None``)。無効時はレガシー ``learn.policy.<mode>.<domain>.<param_path>``。

        ``param_path`` はドット区切り (将来のネスト対応) を許容するが、現状
        ``params[mode][key]`` は単一 key のみのため、ドット含む場合は丸ごと key。
        """
        prefix = (
            f"{LEARN_POLICY_SUBJECT_PREFIX}{base_model_id}."
            if base_model_id
            else LEARN_POLICY_SUBJECT_PREFIX
        )
        if not subject.startswith(prefix):
            return None
        rest = subject[len(prefix):]
        parts = rest.split(".")
        if len(parts) < 3:
            return None
        mode = parts[0]
        domain = parts[1]
        key = ".".join(parts[2:])
        if not mode or not domain or not key:
            return None
        return mode, domain, key

    def _apply_single_override(
        self,
        domain: str,
        mode: str,
        key: str,
        raw_object: str,
    ) -> bool:
        """単一の SemMem ファクトを self._data に適用する。

        Returns:
            True: 適用成功 / False: ターゲット未知 (ドメイン or モード or key)
        """
        policy = self._data.get(domain)
        if policy is None:
            logger.debug(
                "SemMem override skipped: unknown domain '%s' (key=%s)",
                domain, key,
            )
            return False
        params = policy.get("params", {})
        # learning ドメイン等は ``_default`` キー
        if "_default" in params:
            actual_mode = "_default"
        elif mode in params:
            actual_mode = mode
        else:
            logger.debug(
                "SemMem override skipped: unknown mode '%s' in domain '%s'",
                mode, domain,
            )
            return False
        mode_params = params[actual_mode]
        if key not in mode_params:
            logger.debug(
                "SemMem override skipped: unknown key '%s' in domain '%s' mode '%s'",
                key, domain, actual_mode,
            )
            return False

        value = self._coerce_object_value(raw_object)
        constraints = policy.get("constraints", {})
        if key in constraints:
            try:
                value = _clamp(value, constraints[key])
            except (TypeError, ValueError) as exc:
                # 破損ファクト (object が JSON として不正 / 型変換不能) は
                # 黙って YAML デフォルトを温存する。推論パスを壊さ
                # ないことを最優先する。
                logger.warning(
                    "SemMem override skipped: corrupted value for "
                    "domain=%s mode=%s key=%s raw=%r (%s)",
                    domain, actual_mode, key, raw_object, exc,
                )
                return False
        mode_params[key] = value
        logger.debug(
            "SemMem override applied: domain=%s mode=%s key=%s value=%r",
            domain, actual_mode, key, value,
        )
        return True

    @staticmethod
    def _coerce_object_value(raw: Any) -> Any:
        """SemMem の ``object`` フィールド (str) を JSON 値として復元する。

        ``object`` は ``str`` 想定だが、すでに数値型やブール値が直接入って
        いた場合はそのまま返す。文字列なら ``json.loads`` を試み、失敗時は
        raw 文字列としてフォールバックする。
        """
        if not isinstance(raw, str):
            return raw
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return raw

    def apply_delta(self, domain: str, delta: dict, mode: str = "chat") -> None:
        """パラメータの差分更新を適用する（Stage 4: PolicyEvolver 用）

        制約違反時は clamp で安全範囲に丸める。
        適用前にスナップショットを保存し、rollback() で復元可能にする。

        Args:
            domain: ポリシードメイン
            delta: {key: new_value, ...}
            mode: "chat" | "create"
        """
        with self._lock:
            policy = self._data.get(domain)
            if policy is None:
                raise KeyError(f"Unknown policy domain: {domain}")

            params = policy["params"]
            if "_default" in params:
                actual_mode = "_default"
            else:
                actual_mode = mode

            mode_params = params.get(actual_mode)
            if mode_params is None:
                raise KeyError(f"Unknown mode '{mode}' in domain '{domain}'")

            constraints = policy.get("constraints", {})

            # スナップショット保存
            self._snapshots[(domain, mode)] = copy.deepcopy(mode_params)

            # デルタ適用（制約付き）
            for key, value in delta.items():
                if key not in mode_params:
                    logger.warning(
                        "apply_delta: unknown key '%s' in domain '%s', skipping",
                        key, domain,
                    )
                    continue
                if key in constraints:
                    value = _clamp(value, constraints[key])
                mode_params[key] = value

            logger.info(
                "Policy delta applied: domain=%s, mode=%s, keys=%s",
                domain, mode, list(delta.keys()),
            )

    def rollback(self, domain: str, mode: str = "chat") -> bool:
        """直前の apply_delta を元に戻す

        Returns:
            True: ロールバック成功, False: スナップショットなし
        """
        with self._lock:
            snapshot = self._snapshots.pop((domain, mode), None)
            if snapshot is None:
                logger.warning(
                    "No snapshot for rollback: domain=%s, mode=%s", domain, mode,
                )
                return False

            policy = self._data.get(domain)
            if policy is None:
                return False

            params = policy["params"]
            actual_mode = "_default" if "_default" in params else mode
            params[actual_mode] = snapshot

            logger.info("Policy rolled back: domain=%s, mode=%s", domain, mode)
            return True

    def save(self) -> None:
        """全ポリシーをファイルに保存する"""
        self._policies_dir.mkdir(parents=True, exist_ok=True)

        with self._lock:
            for filename, domain in _POLICY_FILES.items():
                if domain not in self._data:
                    continue
                path = self._policies_dir / filename
                atomic_write_text(
                    path,
                    json.dumps(self._data[domain], ensure_ascii=False, indent=2),
                )

        logger.info("Policies saved to %s", self._policies_dir)

    def _load_all(self) -> None:
        """全ポリシーファイルを読み込む（存在しない場合はデフォルト生成）"""
        defaults = _default_policies()
        self._policies_dir.mkdir(parents=True, exist_ok=True)

        for filename, domain in _POLICY_FILES.items():
            path = self._policies_dir / filename
            if path.exists():
                try:
                    with open(path, encoding="utf-8") as f:
                        data = json.load(f)
                    # デフォルトとマージ（新しいキーの追加に対応）
                    self._data[domain] = self._merge_with_defaults(
                        data, defaults.get(domain, {}),
                    )
                    logger.debug("Loaded policy: %s", path)
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning(
                        "Failed to load %s, using defaults: %s", path, e,
                    )
                    self._data[domain] = copy.deepcopy(defaults[domain])
            else:
                # デフォルトで生成
                self._data[domain] = copy.deepcopy(defaults[domain])
                try:
                    atomic_write_text(
                        path,
                        json.dumps(self._data[domain], ensure_ascii=False, indent=2),
                    )
                    logger.info("Created default policy: %s", path)
                except OSError as e:
                    logger.warning("Failed to write default policy %s: %s", path, e)

    @staticmethod
    def _merge_with_defaults(loaded: dict, defaults: dict) -> dict:
        """ロード済みデータにデフォルト値をマージする

        ポリシーファイルに新しいキーが追加された場合、
        既存ファイルにないキーをデフォルトで補完する。
        """
        result = copy.deepcopy(loaded)

        # constraints のマージ
        if "constraints" in defaults:
            existing_constraints = result.get("constraints", {})
            for k, v in defaults["constraints"].items():
                if k not in existing_constraints:
                    existing_constraints[k] = v
            result["constraints"] = existing_constraints

        # params のマージ（モード別）
        if "params" in defaults:
            existing_params = result.get("params", {})
            for mode, mode_defaults in defaults["params"].items():
                if mode not in existing_params:
                    existing_params[mode] = copy.deepcopy(mode_defaults)
                else:
                    for k, v in mode_defaults.items():
                        if k not in existing_params[mode]:
                            existing_params[mode][k] = v
            result["params"] = existing_params

        return result
