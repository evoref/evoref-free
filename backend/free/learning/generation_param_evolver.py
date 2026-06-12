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
            mode: "chat" または "coding"
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
            logger.info(
                "Mode %s generation params: no improvement (%.4f)",
                mode, current_fitness,
            )

        return {
            "improved": improved,
            "fitness_before": current_fitness,
            "fitness_after": best_fitness,
            "deltas": best_deltas if improved else dict(current),
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

    def _evaluate_fitness(self, experiences: list[dict]) -> float:
        """経験バッファのシグナルからフィットネスを計算"""
        if not experiences:
            return 0.5

        score = 0.0
        for exp in experiences:
            signals = exp.get("signals", {})
            # 肯定的シグナル
            if signals.get("conversation_ended"):
                score += 1.0
            # 否定的シグナル
            if signals.get("user_correction"):
                score -= 0.8
            if signals.get("rephrased_query"):
                score -= 0.5

        return max(0.0, min(1.0, (score / len(experiences) + 1) / 2))

    def _evaluate_with_deltas(
        self, experiences: list[dict], deltas: dict[str, float],
    ) -> float:
        """デルタ適用時のフィットネスを推定

        実際に LLM を呼び出すわけではなく、デルタの方向性と
        経験シグナルの相関からフィットネスを推定する。
        """
        base_fitness = self._evaluate_fitness(experiences)

        # デルタの方向性ボーナス/ペナルティ
        adjustment = 0.0

        temp_delta = deltas.get("temperature_delta", 0.0)
        # 訂正が多い場合: temperature を下げる方向が良い
        correction_rate = sum(
            1 for e in experiences
            if e.get("signals", {}).get("user_correction")
        ) / max(len(experiences), 1)

        if correction_rate > 0.3 and temp_delta < 0:
            adjustment += 0.05  # 訂正が多い → 温度を下げると改善
        elif correction_rate < 0.1 and temp_delta > 0:
            adjustment += 0.02  # 訂正が少ない → 多様性を上げても安全

        # presence_penalty: 繰り返しが多い場合に上げる方向が良い
        pp_delta = deltas.get("presence_penalty_delta", 0.0)
        if pp_delta > 0 and correction_rate < 0.2:
            adjustment += 0.02  # 繰り返し抑制

        return max(0.0, min(1.0, base_fitness + adjustment))


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
