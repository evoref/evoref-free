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

#: 「読むことを拒否した」を表す番兵 (``None`` は正当なペイロードなので使えない)。
_REFUSED = object()


def read_payload(path: str | Path) -> Any:
    """永続化ファイルから **ペイロードだけ** を読む (エンベロープを剥がす)。

    エンベロープ以前の素のペイロードもそのまま返す。ストアを経由せずに中身を
    見たい移行スクリプト / テスト用。
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "payload" in raw and "schema_version" in raw:
        return raw["payload"]
    return raw


def write_payload(
    path: str | Path, payload: JsonPayload, *, schema_version: int = 1,
    component: str = "",
) -> None:
    """ペイロードをエンベロープで包んで書き出す (テスト / 移行スクリプト用)。"""
    from backend.utils import utc_now

    envelope = {
        "schema_version": schema_version,
        "written_at": utc_now(),
        "producer": {"component": component, "app_version": _app_version()},
        "payload": payload,
    }
    atomic_write_text(
        Path(path), json.dumps(envelope, ensure_ascii=False, indent=2),
    )


def _app_version() -> str:
    """アプリ版数 (取得できなければ空文字)。producer 記録用。"""
    try:
        from backend.version import get_runtime_version

        return str(get_runtime_version())
    except Exception:
        return ""


class JsonStateStore:
    """JSON ファイルへの save/load を共通化する template-method 基底クラス。

    サブクラスは :meth:`_to_payload` と :meth:`_from_payload` を実装すること。
    継承時は `__init__` をそのまま使ってよい (本クラスは状態を持たない)。

    書き出しは **永続化エンベロープ** (c_05 §0.5) で包む::

        {"schema_version": 1, "written_at": "...Z",
         "producer": {"component": "...", "app_version": "..."},
         "payload": <_to_payload() の返り値>}

    読み込みはエンベロープと素のペイロード (旧形式) の両方を受け付ける。
    ``schema_version`` が :attr:`SCHEMA_VERSION` より **新しい** ファイルは
    読まず、以後の ``save`` も拒否する — 知らないフィールドを落としたまま
    旧版で書き戻すと新版の内容を破壊するため (2026-09-05 監査)。
    """

    #: サブクラスが上書き可能なロガー (デフォルトはモジュール logger)。
    _state_logger: logging.Logger | None = None

    #: このストアのレコード版。フィールドの意味を変える / 必須キーを増やす
    #: 変更で上げ、対応する migrator を用意する。
    SCHEMA_VERSION: int = 1

    #: ディスク上のファイルが未対応の新しい版で、書き戻すと壊す状態。
    _downgrade_blocked: bool = False

    # ── public API (template methods) ──

    def save(self, path: str | Path) -> None:
        """状態を `path` に JSON で永続化する (:func:`atomic_write_text` 経由)。

        親ディレクトリは ``AtomicWriter`` が自動作成。書き込み途中で例外が
        発生した場合は宛先ファイルが不変のまま残る (半壊 JSON を残さない)。
        シリアライズ / 書き込みに失敗した場合は WARNING ログを出して silently
        return する。
        """
        path = Path(path)
        if self._downgrade_blocked:
            self._store_logger().warning(
                "Skipping save of %s to %s: on-disk file is a newer schema "
                "version and would be downgraded",
                type(self).__name__, path,
            )
            return
        try:
            payload = self._to_payload()
            text = json.dumps(
                self._wrap_payload(payload), ensure_ascii=False, indent=2,
            )
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
            raw = json.loads(text)
        except (OSError, json.JSONDecodeError) as e:
            self._store_logger().warning(
                "Failed to read %s from %s: %s",
                type(self).__name__, path, e,
            )
            return
        payload = self._unwrap_payload(raw, path)
        if payload is _REFUSED:
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

    # ── 永続化エンベロープ ──

    def _wrap_payload(self, payload: JsonPayload) -> dict[str, Any]:
        """ペイロードを永続化エンベロープで包む。"""
        from backend.utils import utc_now

        return {
            "schema_version": self.SCHEMA_VERSION,
            "written_at": utc_now(),
            "producer": {
                "component": type(self).__name__,
                "app_version": _app_version(),
            },
            "payload": payload,
        }

    def _unwrap_payload(self, raw: Any, path: Path) -> Any:
        """エンベロープを剥がす。旧形式 (素のペイロード) はそのまま返す。

        未対応の新しい版なら :data:`_REFUSED` を返し、以後の ``save`` も
        止める (旧版で書き戻して壊さないため)。
        """
        if not (isinstance(raw, dict) and "payload" in raw and "schema_version" in raw):
            return raw  # 旧形式 (エンベロープ以前)
        version = raw.get("schema_version")
        if isinstance(version, int) and version > self.SCHEMA_VERSION:
            self._downgrade_blocked = True
            self._store_logger().error(
                "Refusing to load %s from %s: schema_version %s is newer than "
                "supported %s. Running with current state; the file is left "
                "untouched.",
                type(self).__name__, path, version, self.SCHEMA_VERSION,
            )
            return _REFUSED
        return raw["payload"]

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
