"""GenerationParamEvolver の JSON 永続化

`backend.free.learning.generation_param_evolver.GenerationParamEvolver` から
ドメインロジックを分離するための infra 層。`GenerationDeltaStore` はモード別
デルタ辞書のシリアライズ / デシリアライズと JSON ファイル I/O のみを担い、
評価ロジック (フィットネス算出、候補生成、デルタ採択) は持たない。

レイヤー責務:
- `GenerationParamEvolver`  — ドメイン (フィットネス評価、候補生成、デルタ進化)
- `GenerationDeltaStore`    — インフラ (JSON 永続化、ファイル I/O)

このため `GenerationDeltaStore` は import 時に `GenerationParamEvolver` を
参照せず、純粋な dict 構造のみに依存する (循環依存防止 + 単体テスト可能性確保)。
"""

from __future__ import annotations

import json
from pathlib import Path

from backend.log_config import get_logger

logger = get_logger("learning.generation_delta_store")


# モード -> {param_delta: float} の入れ子辞書型エイリアス
type DeltaMap = dict[str, dict[str, float]]


class GenerationDeltaStore:
    """GenerationParamEvolver の純粋な永続化担当

    全メソッドが副作用ゼロ (ファイル I/O 以外) かつ I/O は static method として
    実装。インスタンス状態を持たないため、テストで stub 化が容易。
    """

    @staticmethod
    def serialize(deltas: DeltaMap) -> DeltaMap:
        """モード別デルタ辞書を JSON-serializable な dict に変換する (deep copy)。

        参照共有を避けるため新しい dict を返却する。
        """
        return {mode: dict(params) for mode, params in deltas.items()}

    @staticmethod
    def deserialize(data: dict) -> DeltaMap:
        """生 JSON dict から DeltaMap を再構築する。

        欠損やスキーマ不一致は寛容に処理し、float 化できないデルタは無視する。
        後方互換 (古い JSON フォーマット) を維持する。
        """
        result: DeltaMap = {}
        if not isinstance(data, dict):
            return result
        for mode, params in data.items():
            if not isinstance(params, dict):
                continue
            mode_deltas: dict[str, float] = {}
            for key, value in params.items():
                try:
                    mode_deltas[str(key)] = float(value)
                except (TypeError, ValueError):
                    logger.warning(
                        "Ignoring non-numeric delta for mode=%s key=%s value=%r",
                        mode, key, value,
                    )
            result[str(mode)] = mode_deltas
        return result

    @staticmethod
    def save(deltas: DeltaMap, path: str | Path) -> None:
        """`deltas` を JSON ファイルに書き出す。親ディレクトリは自動作成。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = GenerationDeltaStore.serialize(deltas)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Saved generation deltas (%d modes) to %s", len(payload), path)

    @staticmethod
    def load(path: str | Path) -> DeltaMap | None:
        """JSON ファイルから DeltaMap を読み込む。

        ファイルが存在しない場合は `None` を返す (空辞書とは区別する)。
        呼び出し側は `None` を「ファイル未存在 = 既存状態を保持」と解釈できる。
        パース失敗時も `None` を返し、警告ログを出力する。
        """
        path = Path(path)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to load generation deltas from %s: %s", path, e)
            return None
        deltas = GenerationDeltaStore.deserialize(data)
        logger.info("Loaded generation deltas (%d modes) from %s", len(deltas), path)
        return deltas

    @staticmethod
    def load_mode(path: str | Path, mode: str) -> dict[str, float]:
        """指定モードのデルタのみを返す薄いヘルパー (config.py 用途)。

        ファイル未存在・モード未登録時は空辞書を返す。
        """
        loaded = GenerationDeltaStore.load(path)
        if loaded is None:
            return {}
        return dict(loaded.get(mode, {}))
