"""アシスト purpose 別 timeout の自己較正値の永続化 (model-keyed)

`AssistModelClient` が観測した ReadTimeout から反応的に引き上げた purpose 別
timeout 天井を、アシストモデルの GGUF ファイル名でキー化して JSON 永続化する。
モデルを切り替えた際に別モデルの較正値を誤用しないよう、ファイル名が一致する
entry のみをロードする。

レイヤー責務:
- `AssistModelClient`        — ドメイン (レイテンシ観測、天井引き上げ判定)
- `AssistCalibrationStore`   — インフラ (JSON 永続化、ファイル I/O)

このため `AssistCalibrationStore` は import 時にドメインを参照せず、純粋な dict
構造のみに依存する (循環依存防止 + 単体テスト容易性確保)。
"""

from __future__ import annotations

import json
from pathlib import Path

from backend.io.atomic import atomic_write_text
from backend.log_config import get_logger
from backend.utils import utc_now

logger = get_logger("llm.assist_calibration_store")


class AssistCalibrationStore:
    """アシスト較正値の純粋な永続化担当 (model-keyed)

    JSON 構造::

        {
          "<assist_model_filename>": {
            "timeouts": {"<purpose>": <float seconds>, ...},
            "calibrated_at": "<ISO8601 Z>",
            "source": "reactive"
          },
          ...
        }

    全メソッドが static で、ファイル I/O 以外の副作用を持たない。
    """

    @staticmethod
    def load_all(path: str | Path) -> dict:
        """JSON 全体を読み込む。未存在 / 破損 / 型不一致時は空 dict を返す。"""
        p = Path(path)
        if not p.exists():
            return {}
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to load assist calibration from %s: %s", p, e)
            return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def load_timeouts(path: str | Path, model_filename: str) -> dict[str, float]:
        """指定モデルの purpose 別 timeout 較正値のみを返す。

        ファイル未存在 / モデル未登録 / 型不一致は空 dict を返す。float 化できない
        値は警告して無視する (後方互換 + 破損耐性)。
        """
        if not model_filename:
            return {}
        entry = AssistCalibrationStore.load_all(path).get(model_filename)
        if not isinstance(entry, dict):
            return {}
        raw = entry.get("timeouts")
        if not isinstance(raw, dict):
            return {}
        result: dict[str, float] = {}
        for purpose, value in raw.items():
            try:
                result[str(purpose)] = float(value)
            except (TypeError, ValueError):
                logger.warning(
                    "Ignoring non-numeric calibrated timeout "
                    "(model=%s purpose=%s value=%r)",
                    model_filename, purpose, value,
                )
        return result

    @staticmethod
    def save_timeouts(
        path: str | Path, model_filename: str, timeouts: dict[str, float],
    ) -> None:
        """指定モデルの purpose 別 timeout 較正値を書き出す。

        他モデルの entry は保持したまま当該モデルの entry のみ差し替える。親
        ディレクトリは自動作成。書き込みは ``atomic_write_text`` (tmp +
        ``os.replace``) で部分書き込みを防止する。
        """
        if not model_filename:
            return
        p = Path(path)
        data = AssistCalibrationStore.load_all(p)
        data[model_filename] = {
            "timeouts": {str(k): float(v) for k, v in timeouts.items()},
            "calibrated_at": utc_now(),
            "source": "reactive",
        }
        atomic_write_text(
            p,
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "Saved assist calibration for model=%s (%d purposes) to %s",
            model_filename, len(timeouts), p,
        )
