"""JSON-backed state persistence base class

重複と特定された 4 ファイル
(`level0_instant.ExperienceBuffer` / `fewshot_pool.FewShotPool` /
`policy_evolver.PolicyEvolver` / `exploration_controller.ExplorationController`)
の `save` / `load` 共通ボイラープレート (mkdir / JSON dump / 例外ロギング /
ファイル不在 no-op) を集約する template-method 基底クラス。

## 設計

サブクラスは 2 つの抽象メソッドを実装するだけで永続化の全責務を満たせる:

- :meth:`_to_payload` — 自身の状態を JSON シリアライズ可能なオブジェクトに変換
- :meth:`_from_payload` — JSON からデコードしたオブジェクトで自身の状態を復元

加えて、サブクラス固有のログメッセージや成功時カスタマイズに以下のフックが
オプショナルで利用可能:

- :meth:`_on_save_success` — 保存成功時 (パスを受け取る)
- :meth:`_on_load_success` — ロード成功時 (パスを受け取る)
- :meth:`_on_load_missing` — ロード時にファイルが存在しなかった場合

## エラーハンドリング方針

`save` は `OSError` / `TypeError` / `ValueError` を、`load` は読み込みで
`OSError` / `json.JSONDecodeError` を、デシリアライズ (`_from_payload`) では
あらゆる `Exception` を捕捉し、`_state_logger` (クラス変数で差し替え可能、
デフォルトはモジュール logger) に WARNING を出力した後 silently return する。

これは元の 4 実装のうち多数派 (3 / 4) が採用していたパターンで、
学習サブシステムの save/load 失敗が呼び出し側のホットパス (チャット応答 /
sleep 更新) を巻き添えにしないための安全策。失敗時の挙動を変えたい場合は
サブクラス側で `save` / `load` をオーバーライドする。

"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Union

from backend.io import atomic_write_text
from backend.log_config import get_logger

#: JSON ペイロードとして許容する型 (`json.dump` で扱える dict / list の合成)
JsonPayload = Union[dict[str, Any], list[Any]]

_default_logger = get_logger("learning.json_state_store")


class JsonStateStore:
    """JSON ファイルへの save/load を共通化する template-method 基底クラス。

    サブクラスは :meth:`_to_payload` と :meth:`_from_payload` を実装すること。
    継承時は `__init__` をそのまま使ってよい (本クラスは状態を持たない)。
    """

    #: サブクラスが上書き可能なロガー (デフォルトはモジュール logger)。
    _state_logger: logging.Logger | None = None

    # ── public API (template methods) ──

    def save(self, path: str | Path) -> None:
        """状態を `path` に JSON で永続化する (:func:`atomic_write_text` 経由)。

        親ディレクトリは ``AtomicWriter`` が自動作成。書き込み途中で例外が
        発生した場合は宛先ファイルが不変のまま残る (半壊 JSON を残さない)。
        シリアライズ / 書き込みに失敗した場合は WARNING ログを出して silently
        return する。
        """
        path = Path(path)
        try:
            payload = self._to_payload()
            text = json.dumps(payload, ensure_ascii=False, indent=2)
            atomic_write_text(path, text)
        except (OSError, TypeError, ValueError) as e:
            self._store_logger().warning(
                "Failed to save %s to %s: %s",
                type(self).__name__, path, e,
            )
            return
        self._on_save_success(path)

    def load(self, path: str | Path) -> None:
        """`path` から状態を復元する。

        ファイルが存在しない場合は :meth:`_on_load_missing` を呼び silently return。
        読み込み / パース / デシリアライズに失敗した場合は WARNING を出して return。
        """
        path = Path(path)
        if not path.exists():
            self._on_load_missing(path)
            return
        try:
            text = path.read_text(encoding="utf-8")
            payload = json.loads(text)
        except (OSError, json.JSONDecodeError) as e:
            self._store_logger().warning(
                "Failed to read %s from %s: %s",
                type(self).__name__, path, e,
            )
            return
        try:
            self._from_payload(payload)
        except Exception as e:
            # 型エラー 3 種に限っていたが、``AttributeError`` (要素が dict でない)
            # 等が素通りして起動を落とし、半分読んだ状態を次の save が上書き
            # していた (2026-09-02 監査 R-B1)。復元失敗は種類を問わず WARNING で
            # 止め、呼出側のホットパスへ伝播させない。
            self._store_logger().warning(
                "Failed to deserialize %s from %s: %s",
                type(self).__name__, path, e,
            )
            return
        self._on_load_success(path)

    # ── 抽象メソッド (subclass MUST override) ──

    def _to_payload(self) -> JsonPayload:
        """自身の状態を JSON シリアライズ可能なオブジェクトに変換する。"""
        raise NotImplementedError

    def _from_payload(self, payload: JsonPayload) -> None:
        """JSON ペイロードから自身の状態を復元する。"""
        raise NotImplementedError

    # ── 任意フック (subclass MAY override) ──

    def _on_save_success(self, path: Path) -> None:
        """保存成功時のフック。デフォルトは無音。"""

    def _on_load_success(self, path: Path) -> None:
        """ロード成功時のフック。デフォルトは無音。"""

    def _on_load_missing(self, path: Path) -> None:
        """ロード時にファイルが存在しなかった場合のフック。デフォルトは無音。"""

    # ── ロガー解決 ──

    def _store_logger(self) -> logging.Logger:
        return self._state_logger or _default_logger
