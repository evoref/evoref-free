"""GenerationParamEvolver: 生成パラメータのデルタベース進化ロジック

ExperienceBuffer のフィードバックスコアでモード別の生成パラメータを
小さなデルタで調整する。LLM 不要のルールベース進化。
"""

import random
from pathlib import Path

from backend.free.learning.generation_delta_store import GenerationDeltaStore
from backend.log_config import get_logger

logger = get_logger("learning.generation_param_evolver")

# デルタ範囲制約
DELTA_RANGES = {
    "temperature_delta": (-0.2, 0.2),
    "top_p_delta": (-0.1, 0.1),
    "top_k_delta": (-10, 10),
    "presence_penalty_delta": (-0.3, 0.3),
}

# パラメータ値の範囲制約
PARAM_CLAMPS = {
    "temperature": (0.0, 2.0),
    "top_p": (0.0, 1.0),
    "top_k": (0, 1000),
    "presence_penalty": (-2.0, 2.0),
}

# デルタファイルのデフォルトパス
DEFAULT_DELTA_FILE = "local/generation_deltas.json"


class GenerationParamEvolver:
    """生成パラメータのデルタベース進化

    現在の設定値に対して小さなデルタを生成し、
    ExperienceBuffer のフィードバックスコアで評価・選択する。
    """

    def __init__(self, delta_file: Path | None = None):
        self._delta_file = delta_file
        self._deltas: dict[str, dict[str, float]] = {}
        if delta_file is not None:
            loaded = GenerationDeltaStore.load(delta_file)
            if loaded is not None:
                self._deltas = loaded

    def save_deltas(self) -> None:
        """デルタを永続化する (infra 層 `GenerationDeltaStore` に委譲)"""
        if self._delta_file is None:
            return
        try:
            GenerationDeltaStore.save(self._deltas, self._delta_file)
        except OSError as e:
            logger.warning("Failed to save generation deltas: %s", e)

    def get_deltas(self, mode: str) -> dict[str, float]:
        """指定モードのデルタを取得"""
        return dict(self._deltas.get(mode, {}))

    def evolve(
        self,
        mode: str,
        experiences: list[dict],
        population_size: int = 5,
    ) -> dict:
        """モードの生成パラメータデルタを進化させる

        Args:
            mode: "chat" または "create"
            experiences: 経験バッファのエントリ（signals 含む）
            population_size: 候補数

        Returns:
            {"improved": bool, "fitness_before": float, "fitness_after": float,
             "deltas": dict}
        """
        # モード別経験のみフィルタ
        mode_exp = [e for e in experiences if e.get("mode") == mode]
        if not mode_exp:
            return {
                "improved": False,
                "fitness_before": 0.5,
                "fitness_after": 0.5,
                "deltas": {},
            }

        # 現在のデルタ。現行も候補と同一尺度 (_evaluate_with_deltas) で評価して
        # 比較を対称化する。現行を方向ボーナス無しの素評価にすると、候補側にだけ
        # ボーナスが乗って improved が常時 True になる (偽の改善)。current が空 {}
        # なら全 *_delta=0.0 で方向ボーナスは乗らず素のシグナルフィットネスと一致。
        current = self._deltas.get(mode, {})
        current_fitness = self._evaluate_with_deltas(mode_exp, current)

        best_deltas = dict(current)
        best_fitness = current_fitness

        # 候補を生成して評価
        for _ in range(population_size):
            candidate = self._generate_candidate(current)
            # フィットネスはデルタの方向性で評価
            fitness = self._evaluate_with_deltas(mode_exp, candidate)
            if fitness > best_fitness:
                best_fitness = fitness
                best_deltas = candidate

        improved = best_fitness > current_fitness
        if improved:
            self._deltas[mode] = best_deltas
            self.save_deltas()
            logger.info(
                "Mode %s generation params evolved: fitness %.4f -> %.4f, deltas=%s",
                mode, current_fitness, best_fitness, best_deltas,
            )
        else:
            # 「改善しなかった」と「そもそも評価できない」を区別して出す。
            # 後者は経験レコードに生成パラメータが残っていない限り解消しない。
            reason = (
                "no_improvement" if self._has_outcome_signal(mode_exp)
                else "no_outcome_signal"
            )
            logger.info(
                "Mode %s generation params: %s (fitness %.4f, n=%d)",
                mode, reason, current_fitness, len(mode_exp),
            )
            return {
                "improved": False,
                "fitness_before": current_fitness,
                "fitness_after": best_fitness,
                "deltas": dict(current),
                "reason": reason,
            }

        return {
            "improved": improved,
            "fitness_before": current_fitness,
            "fitness_after": best_fitness,
            "deltas": best_deltas,
            "reason": "improved",
        }

    def _generate_candidate(self, current: dict[str, float]) -> dict[str, float]:
        """現在のデルタに小さな摂動を加えた候補を生成"""
        candidate: dict[str, float] = {}
        for param, (min_d, max_d) in DELTA_RANGES.items():
            current_val = current.get(param, 0.0)
            # 正規分布の摂動
            sigma = (max_d - min_d) * 0.2
            new_val = current_val + random.gauss(0, sigma)
            # デルタ範囲制約でクランプ
            new_val = max(min_d, min(max_d, new_val))
            candidate[param] = round(new_val, 4)
        return candidate

    #: フィットネスに使う **欠陥シグナル** と重み。
    #:
    #: 旧実装は ``conversation_ended`` を加点 (+1.0) の主項にしていたが、この
    #: シグナルは実データで 201/205 = **98% が True** で情報量がほぼ無い。さらに
    #: ``turn_outcome`` は 205/205 が ``"success"`` の **恒真** な成功判定だった
    #: (2026-08-16 実測)。結果フィットネスは 0.987 に張り付き、改善余地は 0.0129 しか
    #: 残らない = Level 1 は実質的に評価不能な状態で回っていた。
    #:
    #: ここでは「観測された欠陥の率」だけを見る。1.0 は「欠陥が観測されなかった」
    #: を意味し、加点シグナルの恒真性に引きずられない。
    _DEFECT_WEIGHTS = {
        "user_correction": 1.0,
        "assistant_self_retraction": 1.0,
        "rephrased_query": 0.6,
        "tool_routing_false_negative": 0.5,
        "tool_routing_false_positive": 0.5,
    }

    def _evaluate_fitness(self, experiences: list[dict]) -> float:
        """観測された欠陥の率からフィットネスを計算する (1.0 = 欠陥なし)。"""
        if not experiences:
            return 0.5
        defects = 0.0
        for exp in experiences:
            signals = exp.get("signals") or {}
            for key, weight in self._DEFECT_WEIGHTS.items():
                if signals.get(key):
                    defects += weight
        return max(0.0, min(1.0, 1.0 - defects / len(experiences)))

    def _has_outcome_signal(self, experiences: list[dict]) -> bool:
        """欠陥シグナルが 1 件でも立っているか (= 評価に使える分散があるか)。"""
        return any(
            (exp.get("signals") or {}).get(key)
            for exp in experiences
            for key in self._DEFECT_WEIGHTS
        )

    def _evaluate_with_deltas(
        self, experiences: list[dict], deltas: dict[str, float],  # noqa: ARG002
    ) -> float:
        """デルタ適用時のフィットネス。**方向ボーナスは廃止した**。

        旧実装は「訂正が少ない → 温度を上げても安全なので +0.02」のような
        **デルタの符号だけを見た加点** を返していた。実際にそのデルタで生成した
        結果を見ているわけではないので、改善の根拠がゼロのまま ``improved=True``
        を宣言しうる。加点幅 (+0.02〜+0.05) が実データの改善余地 (0.0129) より
        大きく、事実上「符号でスコアが決まる」状態だった。

        経験レコードに **その応答を生成したときのパラメータが残っていない** ため、
        現状は候補と現行を区別する材料が無い。区別できないことを素直に返し、
        :meth:`evolve` 側で ``reason`` として可視化する (でっち上げの改善より、
        評価不能であることが見えている方が良い)。
        """
        return self._evaluate_fitness(experiences)


def apply_deltas(params: dict, deltas: dict[str, float]) -> dict:
    """生成パラメータにデルタを適用してクランプする

    Args:
        params: 基本パラメータ dict（temperature, top_p, top_k, presence_penalty）
        deltas: デルタ dict（temperature_delta, top_p_delta, ...）

    Returns:
        デルタ適用済みパラメータ dict
    """
    result = dict(params)
    for param, (min_val, max_val) in PARAM_CLAMPS.items():
        delta_key = f"{param}_delta"
        if delta_key in deltas:
            current = result.get(param)
            if current is not None:
                new_val = current + deltas[delta_key]
                if isinstance(min_val, int):
                    new_val = int(max(min_val, min(max_val, round(new_val))))
                else:
                    new_val = round(max(min_val, min(max_val, new_val)), 4)
                result[param] = new_val
    return result
