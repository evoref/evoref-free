"""SPSA (Simultaneous Perturbation Stochastic Approximation) 最適化"""

from collections.abc import Callable

import numpy as np

from backend.log_config import get_logger

logger = get_logger("optimizer.spsa")


class SPSAOptimizer:
    """勾配フリー SPSA 最適化

    Spall (1992) の SPSA アルゴリズムに基づく。
    LoRA パラメータの微調整に使用する。
    """

    def __init__(
        self,
        a: float = 0.1,
        c: float = 0.01,
        alpha: float = 0.602,
        gamma: float = 0.101,
        A: float = 10.0,
    ):
        """
        Args:
            a: 学習率の初期スケール
            c: 摂動量の初期スケール
            alpha: 学習率の減衰指数
            gamma: 摂動量の減衰指数
            A: 学習率の安定化定数（初期イテレーションでの急激な更新を防ぐ）
        """
        self.a = a
        self.c = c
        self.alpha = alpha
        self.gamma = gamma
        self.A = A

    def _decay_schedule(self, iteration: int) -> tuple[float, float]:
        """学習率と摂動量の減衰スケジュール

        Args:
            iteration: 現在のイテレーション（1始まり）

        Returns:
            (a_k, c_k) 減衰後の学習率と摂動量
        """
        k = iteration
        a_k = self.a / (k + self.A) ** self.alpha
        c_k = self.c / k ** self.gamma
        return a_k, c_k

    def _compute_gradient(
        self,
        params: np.ndarray,
        eval_func: Callable[[np.ndarray], float],
        c_k: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """同時摂動による勾配近似

        Bernoulli ±1 の摂動ベクトルを生成し、
        2回の関数評価で全パラメータの勾配を同時推定する。

        Args:
            params: 現在のパラメータベクトル
            eval_func: 評価関数（小さいほど良い）
            c_k: 摂動量
            rng: 乱数生成器

        Returns:
            勾配近似ベクトル
        """
        # Bernoulli ±1 摂動
        delta = rng.choice([-1.0, 1.0], size=params.shape)
        perturbation = c_k * delta

        # 2方向の評価
        loss_plus = eval_func(params + perturbation)
        loss_minus = eval_func(params - perturbation)

        # 勾配近似
        gradient = (loss_plus - loss_minus) / (2.0 * perturbation)

        return gradient

    def optimize(
        self,
        eval_func: Callable[[np.ndarray], float],
        initial_params: np.ndarray,
        iterations: int = 500,
        seed: int | None = None,
        callback: Callable[[int, np.ndarray, float], None] | None = None,
    ) -> tuple[np.ndarray, float]:
        """SPSA 最適化ループ

        Args:
            eval_func: 評価関数（小さいほど良い = 最小化）
            initial_params: 初期パラメータ
            iterations: イテレーション数
            seed: 乱数シード
            callback: 各イテレーション後のコールバック (iteration, params, loss)

        Returns:
            (最適化後パラメータ, 最終損失値)
        """
        rng = np.random.default_rng(seed)
        params = initial_params.copy().astype(np.float64)
        best_params = params.copy()
        best_loss = eval_func(params)

        logger.info(
            "SPSA optimization: %d params, %d iterations, initial loss=%.6f",
            len(params), iterations, best_loss,
        )

        for k in range(1, iterations + 1):
            a_k, c_k = self._decay_schedule(k)

            # 勾配近似
            gradient = self._compute_gradient(params, eval_func, c_k, rng)

            # パラメータ更新
            params = params - a_k * gradient

            # 現在の損失
            current_loss = eval_func(params)

            # 最良結果の追跡
            if current_loss < best_loss:
                best_loss = current_loss
                best_params = params.copy()

            if callback:
                callback(k, params, current_loss)

            if k % 100 == 0 or k == iterations:
                logger.info(
                    "SPSA iter %d/%d: loss=%.6f, best=%.6f, a_k=%.6f, c_k=%.6f",
                    k, iterations, current_loss, best_loss, a_k, c_k,
                )

        return best_params, best_loss
