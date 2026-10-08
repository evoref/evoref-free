"""GenerationParamEvolver: 生成パラメータのデルタベース進化ロジック

ExperienceBuffer のフィードバックスコアでモード別の生成パラメータを
小さなデルタで調整する。LLM 不要のルールベース進化。
"""

import random
from pathlib import Path

from backend.free.learning.fitness import (
    GENERATION_DEFECT_WEIGHTS,
    defect_rate_fitness,
)
from backend.free.learning.generation_delta_store import GenerationDeltaStore
from backend.free.learning.pairwise_gate import (
    PAIRWISE_NET_WINS,
    net_wins,
    noise_floor,
    pairwise_verdict,
    tally,
)
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

    def persisted_deltas(self, mode: str) -> dict[str, float]:
        """ディスク上 (本番の生成が読む側) のそのモードのデルタ。

        ファイルを持たない構成ではメモリ上のデルタ。採用後監視が「採用した版が
        まだ効いているか」を見るのに使う (手で書き換えられていたら監視を打ち切る)。
        """
        if self._delta_file is None:
            return self.get_deltas(mode)
        # load_mode の mtime キャッシュは粗い時刻の FS で直後の書込みを見逃しうる
        loaded = GenerationDeltaStore.load(self._delta_file)
        return dict((loaded or {}).get(mode, {}))

    def restore_deltas(self, mode: str, deltas: dict[str, float]) -> None:
        """そのモードのデルタを ``deltas`` に戻して永続化する (採用後監視の巻き戻し)。"""
        self._deltas[mode] = dict(deltas)
        self.save_deltas()

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

    #: フィットネスに使う **欠陥シグナル** と重み
    #: (共有定義 ``fitness.GENERATION_DEFECT_WEIGHTS``)。
    #:
    #: 旧実装は ``conversation_ended`` を加点 (+1.0) の主項にしていたが、この
    #: シグナルは実データで 201/205 = **98% が True** で情報量がほぼ無い。
    #:
    #: 2026-09-21: 共有表 (``DEFECT_WEIGHTS``) から **生成パラメータ専用の表**へ
    #: 切り替えた。動かすのは temperature / top_p / top_k だけなので、直接効く
    #: 出力の崩れ (``outcome_broken_output``) を重く、sampling では直らない
    #: ルーティングの失敗 (``outcome_execution``) を軽くする。責任のない失敗で
    #: 温度を動かすと、直らないまま制約の端まで振れる。
    _DEFECT_WEIGHTS = GENERATION_DEFECT_WEIGHTS

    def _evaluate_fitness(self, experiences: list[dict]) -> float:
        """観測された欠陥の率からフィットネスを計算する (1.0 = 欠陥なし、空は 0.5)。"""
        value = defect_rate_fitness(experiences, weights=self._DEFECT_WEIGHTS)
        return 0.5 if value is None else value

    def propose_delta(
        self,
        mode: str,
        rng: random.Random | None = None,
        available: set[str] | None = None,
    ) -> dict:
        """現行デルタに 1 つだけガウス摂動を載せた候補を返す。

        動かすのは 1 パラメータだけにする — 実測ゲートは 3 アームの実生成で
        高価なので、1 サイクルで 1 軸だけ動かし、どの軸が効いたかを
        採用記録から後で追えるようにする。

        Args:
            available: 実際に存在する基本パラメータ名。**指定必須に近い** —
                ``apply_deltas`` は基本パラメータに無いキーのデルタを黙って
                捨てるので、ここで絞らないと「現行と同一のパラメータ同士を
                比べて、勝ったから採用する」という空の採用が起きうる。
        """
        r = rng or random
        current = dict(self._deltas.get(mode, {}))
        keys = [
            k for k in DELTA_RANGES
            if available is None or k.removesuffix("_delta") in available
        ]
        if not keys:
            return current
        key = r.choice(keys)
        lo, hi = DELTA_RANGES[key]
        span = (hi - lo) / 2.0
        base = float(current.get(key, 0.0))
        moved = max(lo, min(hi, base + r.gauss(0.0, span * 0.25)))
        if key == "top_k_delta":
            moved = int(round(moved))
        else:
            moved = round(moved, 4)
        if moved == base:
            return current
        candidate = dict(current)
        candidate[key] = moved
        return candidate

    async def evolve_measured(
        self,
        mode: str,
        experiences: list[dict],
        *,
        base_params: dict,
        prompt_text: str,
        evaluator,
        cases: list,
        samples: list | None = None,
        min_net_wins: int = 1,
        rng: random.Random | None = None,
    ) -> dict:
        """候補デルタを **実生成で比べて** 採否を決める (f_04 §4.7)。

        ``evolve`` が候補生成を止めていた理由は「経験レコードに生成時の
        パラメータが残っていないので候補ごとの fitness を出せない」だった。
        sampling パラメータは prompt と違って **その場で再現できる**ので、
        記録を当てにせず現行 / 候補の両方をその場で生成して比べればよい。

        採用条件は prompt 側の一対比較ゲートと同じ不等式 —
        ``純勝ち >= min_net_wins`` かつ ``純勝ち > 雑音フロア``
        (:func:`~backend.free.learning.pairwise_gate.pairwise_verdict`)。フロアは
        **現行パラメータ同士** のカナリアで採るので、温度が高くて揺れる構成
        ほど自動的に高い要求になる。温度 > 0 なら候補も 2 回引き、純勝ちは
        2 回の小さい方、フロアは現行同士と候補同士の大きい方で見る (候補側の
        揺れ。2 回目は 1 回目が条件を満たしたときだけ引く)。

        純勝ちは ``cases`` (**失敗ケースだけ**) で数え、``samples`` (成功ターンの
        標本) は無退行の検査に使う (プロンプト側と同じ、f_04 §10.1 #2 後半)。
        標本が 0 件なら ``no_regression_samples``、標本で 1 敗なら
        ``sample_regression`` で採用しない。

        Returns:
            ``{"improved": bool, "skipped": bool, "reason": str, "deltas": dict,
            "candidate": dict, "wins"/"losses"/"ties"/"noise_floor"/"net_wins": int,
            "second_draw": bool}`` (wins/losses/ties は 1 回目の引き)
        """
        current = dict(self._deltas.get(mode, {}))
        out: dict = {
            "improved": False, "skipped": True, "reason": "",
            "deltas": current, "candidate": {},
            "wins": 0, "losses": 0, "ties": 0, "noise_floor": 0,
            # 採否は実測で決めるが、そのとき観測されていた欠陥率も残す
            # (後から「どんな状態で何を試したか」を追えるようにする)。
            "fitness": self._evaluate_fitness(experiences),
        }
        if evaluator is None or not callable(
            getattr(evaluator, "compare_generation_params", None),
        ):
            out["reason"] = "no_evaluator"
            return out
        if not cases:
            out["reason"] = "no_cases"
            return out
        candidate = self.propose_delta(mode, rng, available=set(base_params))
        if candidate == current:
            out["reason"] = "no_candidate"
            return out
        current_params = apply_deltas(base_params, current)
        candidate_params = apply_deltas(base_params, candidate)
        if current_params == candidate_params:
            # デルタが基本パラメータに届いていない (存在しないキー / 丸めで同値)。
            # 同じパラメータ同士を比べると雑音だけを測って採用しうるので止める。
            out["reason"] = "candidate_not_measurable"
            return out
        out["candidate"] = candidate
        if not samples:
            # 無退行を測れない候補は採用しない (プロンプト側と同じ、f_04 §8
            # 禁則 7)。失敗ケースの生成コストを払う前に止める。
            out["reason"] = "no_regression_samples"
            return out

        def _first_draw_passes(outcomes: dict, canary: dict) -> bool:
            # 1 回目で不採用が決まる候補には 2 回目の生成コストを払わない
            return pairwise_verdict(
                net_wins(outcomes), noise_floor(canary), min_net_wins,
            ) == PAIRWISE_NET_WINS

        try:
            outcomes, outcomes_b, canary, canary_cand = (
                await evaluator.compare_generation_params(
                    prompt_text, current_params, candidate_params, cases,
                    measure_second_draw=_first_draw_passes,
                )
            )
        except Exception as exc:  # noqa: BLE001 - 実測失敗は不採用で継続
            logger.warning(
                "Mode %s generation params: measured gate failed: %r", mode, exc,
            )
            out["reason"] = "gate_error"
            return out
        if not outcomes:
            out["reason"] = "insufficient_measured_cases"
            return out
        wins, losses, ties = tally(outcomes)
        net = wins - losses
        noise = noise_floor(canary)
        if outcomes_b is not None:
            # 候補も 2 回引いた (温度 > 0): 純勝ちは小さい方、雑音は現行同士と
            # 候補同士の大きい方。1 回の引きの当たりで採用しない (f_04 §4.7)。
            net = min(net, net_wins(outcomes_b))
            noise = max(noise, noise_floor(canary_cand or {}))
        out.update({
            "wins": wins, "losses": losses, "ties": ties, "noise_floor": noise,
            "net_wins": net, "second_draw": outcomes_b is not None,
            # ここまで来たら **実測は走っている**。採用しなくても skipped では
            # ない — scheduler は `_executed_phases` (実行して改善なし) と
            # `_noop_phases` (対象が無く実行していない) をこのフラグで分ける
            # ので、混ぜると「3 アーム 18 回の実生成を 7 分かけて回したのに
            # 実行していないことになる」(2026-09-21 実機で発覚)。
            "skipped": False,
        })
        out["reason"] = pairwise_verdict(net, noise, min_net_wins)
        if out["reason"] == PAIRWISE_NET_WINS:
            out["reason"] = await self._check_sample_regression(
                mode, evaluator, prompt_text, current_params, candidate_params,
                samples, out,
            )
        if out["reason"] == PAIRWISE_NET_WINS:
            self._deltas[mode] = candidate
            self.save_deltas()
            out.update({"improved": True, "deltas": candidate})
        logger.info(
            "Mode %s generation params: wins=%d losses=%d net=%d noise_floor=%d "
            "second_draw=%s candidate=%s (%s)",
            mode, wins, losses, net, noise, outcomes_b is not None, candidate,
            out["reason"],
        )
        return out

    @staticmethod
    async def _check_sample_regression(
        mode: str, evaluator, prompt_text: str, current_params: dict,
        candidate_params: dict, samples: list, out: dict,
    ) -> str:
        """失敗側で採用候補になった候補を標本で再生成し、無退行かを判定する。

        プロンプト側 (``LearningScheduler._check_sample_regression``) と同じ規則
        — 半数以上を測れなければ ``insufficient_measured_cases``、1 敗でもあれば
        ``sample_regression``。候補を 2 回引いたときは、どちらかの引きでの負けを
        数える (負けの多い方)。標本の勝敗は ``out["sample_*"]`` に入れる。
        """
        try:
            first, second, _, _ = await evaluator.compare_generation_params(
                prompt_text, current_params, candidate_params, samples,
            )
        except Exception as exc:  # noqa: BLE001 - 実測失敗は不採用で継続
            logger.warning(
                "Mode %s generation params: regression samples failed: %r", mode, exc,
            )
            return "gate_error"
        wins, losses, ties = tally(first)
        if second is not None:
            losses = max(losses, tally(second)[1])
        out.update({"sample_wins": wins, "sample_losses": losses, "sample_ties": ties})
        if len(first) < max(1, (len(samples) + 1) // 2):
            return "insufficient_measured_cases"
        if losses > 0:
            return "sample_regression"
        return PAIRWISE_NET_WINS


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
