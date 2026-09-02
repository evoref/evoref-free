"""GenerationParamEvolver: 生成パラメータのデルタベース進化ロジック

ExperienceBuffer のフィードバックスコアでモード別の生成パラメータを
小さなデルタで調整する。LLM 不要のルールベース進化。
"""

from pathlib import Path

from backend.free.learning.fitness import (
    DEFECT_WEIGHTS,
    defect_rate_fitness,
    has_defect_signal,
)
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

# デルタファイルのデフォルトパス (実配置は PathResolver ``generation_deltas_file``)
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

    def rebind_delta_file(self, delta_file: Path) -> None:
        """base モデル切替でデルタファイルを新パーティションへ向け直し再ロードする。

        旧パーティションのデルタは採用時点で ``save_deltas`` 済なので退避不要。
        新ファイル未存在なら空 (= 既定パラメータ) から始める。読み手
        (``config.get_generation_params``) は PathResolver 経由で毎回解決するため
        こちらは書き手側のパスを揃えるだけでよい。
        """
        self._delta_file = delta_file
        loaded = GenerationDeltaStore.load(delta_file)
        self._deltas = loaded if loaded is not None else {}

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
        population_size: int = 5,  # noqa: ARG002 — 候補生成は停止中 (下記)
    ) -> dict:
        """モードの生成パラメータデルタを評価する (候補生成は **停止中**)。

        経験レコードには「その応答を生成したときの生成パラメータ」が残っていない
        ため、候補デルタごとに異なる fitness を出す材料が無い。旧実装は候補と
        現行を同じ ``_evaluate_fitness`` で採点しており (デルタ無視)、候補が
        現行を strict に上回ることは構造的に不可能 = 採用ゲートは永久に閉じた
        まま候補生成だけが走っていた (2026-09-02 監査 L-A2)。でっち上げの改善
        より評価不能が見えている方が良いので、候補生成を行わず
        ``skipped=True / reason="no_outcome_signal"`` を返す。per-candidate の
        outcome シグナル (経験レコードへの生成パラメータ記録) が配線されたら
        候補生成を戻す。

        Returns:
            {"improved": False, "skipped": True, "reason": "no_outcome_signal",
             "fitness_before": float, "fitness_after": float, "deltas": dict}
        """
        mode_exp = [e for e in experiences if e.get("mode") == mode]
        current = dict(self._deltas.get(mode, {}))
        fitness = self._evaluate_fitness(mode_exp)
        logger.info(
            "Mode %s generation params: no_outcome_signal "
            "(candidate generation disabled; fitness %.4f, n=%d)",
            mode, fitness, len(mode_exp),
        )
        return {
            "improved": False,
            "skipped": True,
            "reason": "no_outcome_signal",
            "fitness_before": fitness,
            "fitness_after": fitness,
            "deltas": current,
        }

    #: フィットネスに使う **欠陥シグナル** と重み (共有定義 ``fitness.DEFECT_WEIGHTS``)。
    #:
    #: 旧実装は ``conversation_ended`` を加点 (+1.0) の主項にしていたが、この
    #: シグナルは実データで 201/205 = **98% が True** で情報量がほぼ無い。さらに
    #: ``turn_outcome`` は 205/205 が ``"success"`` の **恒真** な成功判定だった
    #: (2026-08-16 実測)。結果フィットネスは 0.987 に張り付き、改善余地は 0.0129 しか
    #: 残らない = Level 1 は実質的に評価不能な状態で回っていた。
    _DEFECT_WEIGHTS = DEFECT_WEIGHTS

    def _evaluate_fitness(self, experiences: list[dict]) -> float:
        """観測された欠陥の率からフィットネスを計算する (1.0 = 欠陥なし、空は 0.5)。"""
        value = defect_rate_fitness(experiences, weights=self._DEFECT_WEIGHTS)
        return 0.5 if value is None else value

    def _has_outcome_signal(self, experiences: list[dict]) -> bool:
        """欠陥シグナルが 1 件でも立っているか (= 評価に使える分散があるか)。"""
        return has_defect_signal(experiences, weights=self._DEFECT_WEIGHTS)


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
