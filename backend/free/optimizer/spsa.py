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
        invalid_loss: float | None = None,
    ) -> tuple[np.ndarray, float, float, np.ndarray]:
        """同時摂動による勾配近似

        Bernoulli ±1 の摂動ベクトルを生成し、
        2回の関数評価で全パラメータの勾配を同時推定する。

        Args:
            params: 現在のパラメータベクトル
            eval_func: 評価関数（小さいほど良い）
            c_k: 摂動量
            rng: 乱数生成器
            invalid_loss: 評価不能を表す番兵値（例: 候補サーバ起動失敗）。±いずれかの
                探索点がこの値なら有限差分が暴発するため勾配を 0 にして当該反復の
                更新をスキップする。None なら従来どおり常に勾配を計算する。

        Returns:
            (勾配近似ベクトル, loss_plus, loss_minus, perturbation)。呼出側が探索点
            (params ± perturbation) を追加評価なしで再構成し best 追跡できるよう
            損失・摂動も返す。
        """
        # Bernoulli ±1 摂動
        delta = rng.choice([-1.0, 1.0], size=params.shape)
        perturbation = c_k * delta

        # 2方向の評価
        loss_plus = eval_func(params + perturbation)
        loss_minus = eval_func(params - perturbation)

        # 勾配近似（片側でも番兵値なら 0 勾配でスキップ）
        if invalid_loss is not None and (
            abs(loss_plus - invalid_loss) < 1e-9 or abs(loss_minus - invalid_loss) < 1e-9
        ):
            gradient = np.zeros_like(params)
        else:
            gradient = (loss_plus - loss_minus) / (2.0 * perturbation)

        return gradient, loss_plus, loss_minus, perturbation

    def optimize(
        self,
        eval_func: Callable[[np.ndarray], float],
        initial_params: np.ndarray,
        iterations: int = 500,
        seed: int | None = None,
        callback: Callable[[int, np.ndarray, float], None] | None = None,
        initial_loss: float | None = None,
        eval_current_params: bool = True,
        invalid_loss: float | None = None,
        early_stop_patience: int | None = None,
    ) -> tuple[np.ndarray, float]:
        """SPSA 最適化ループ

        Args:
            eval_func: 評価関数（小さいほど良い = 最小化）
            initial_params: 初期パラメータ
            iterations: イテレーション数
            seed: 乱数シード
            callback: 各イテレーション後のコールバック (iteration, params, loss)
            initial_loss: 初期点の損失（事前計算済み）。指定すると初期評価を 1 回省く
                （高コスト eval — 例: 候補 LoRA の実サーバ評価 — の重複を避ける）。
            eval_current_params: 各反復で更新後パラメータを追加評価して best 追跡する
                （既定 True = 従来挙動、3 eval/iter）。False なら追加評価を省き、2 つの
                探索点 (probe) の良い方で best を更新する（2 eval/iter）。高コスト eval
                での反復あたり起動回数を 1/3 削減する。
            invalid_loss: 評価不能を表す番兵値。探索点がこの値なら勾配 0 でスキップし、
                best 追跡からも除外する（番兵による暴発・誤採用を防ぐ）。
            early_stop_patience: best が更新されないまま連続でこの反復数を超えたら
                打ち切る（None / 0 で無効 = 従来挙動）。1 反復が候補 LoRA の実サーバ
                起動 2 回に相当する高コスト eval で、探索が空振りしていることが
                確定している後半を回し切らないための天井。

        Returns:
            (最適化後パラメータ, 最終損失値)
        """
        rng = np.random.default_rng(seed)
        params = initial_params.copy().astype(np.float64)
        best_params = params.copy()
        best_loss = initial_loss if initial_loss is not None else eval_func(params)

        logger.info(
            "SPSA optimization: %d params, %d iterations, initial loss=%.6f",
            len(params), iterations, best_loss,
        )

        patience = int(early_stop_patience or 0)
        stagnant = 0
        # 進捗ログの間隔。no-op eval (500 反復規模) は 100 ごとで足りるが、
        # 実 eval (30 反復、1 反復 ≈ 3 分) では完走まで 90 分無音になり、
        # 生きているのか止まっているのか外から読めない (2026-09-05 実機)。
        log_every = 1 if iterations <= 50 else 100

        for k in range(1, iterations + 1):
            a_k, c_k = self._decay_schedule(k)

            # 勾配近似（探索点の損失・摂動も受け取る）
            gradient, loss_plus, loss_minus, perturbation = self._compute_gradient(
                params, eval_func, c_k, rng, invalid_loss=invalid_loss,
            )

            # パラメータ更新（番兵スキップ時は gradient=0 で据え置き）
            prev_params = params
            params = prev_params - a_k * gradient

            improved = False
            if eval_current_params:
                # 更新後パラメータを評価して best 追跡（従来挙動）
                current_loss = eval_func(params)
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_params = params.copy()
                    improved = True
                cb_loss = current_loss
            else:
                # 追加評価を省き、2 探索点 (prev_params ± perturbation) の良い方で
                # best 更新。番兵値の探索点は best 候補から除外する。
                for ploss, ppoint in (
                    (loss_plus, prev_params + perturbation),
                    (loss_minus, prev_params - perturbation),
                ):
                    if invalid_loss is not None and abs(ploss - invalid_loss) < 1e-9:
                        continue
                    if ploss < best_loss:
                        best_loss = ploss
                        best_params = ppoint.copy()
                        improved = True
                cb_loss = min(loss_plus, loss_minus)

            if callback:
                callback(k, params, cb_loss)

            if k % log_every == 0 or k == iterations:
                logger.info(
                    "SPSA iter %d/%d: loss=%.6f, best=%.6f, a_k=%.6f, c_k=%.6f",
                    k, iterations, cb_loss, best_loss, a_k, c_k,
                )

            stagnant = 0 if improved else stagnant + 1
            if patience and stagnant >= patience:
                logger.info(
                    "SPSA early stop at iter %d/%d: best=%.6f unchanged for "
                    "%d iterations",
                    k, iterations, best_loss, stagnant,
                )
                break

        return best_params, best_loss
