"""ファイル書出し Writer の共通テンプレート

Free 版 5 形式 (plaintext / html / csv_tsv / json_yaml / latex) は同一の
``write`` / ``write_to_bytes`` パターンを持つ:

1. ``ext = path.suffix.lower()`` で拡張子取得
2. コンテンツをレンダリング (テキスト or バイト)
3. ``path.parent.mkdir(parents=True, exist_ok=True)`` してファイル書き出し
4. ``OSError`` を ``ExportError("write_error", ...)`` に変換
5. サイズを計測して ``WriteResult`` を返す

このテンプレート部分を ``BytesWriterBase`` に集約する。
サブクラスは ``_render_bytes`` のみを実装すればよい。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from backend.export.base import ExportContent, ExportError, WriteResult
from backend.log_config import get_logger

logger = get_logger("export._writer_base")


class BytesWriterBase(ABC):
    """ファイル書出しの共通テンプレート (バイト出力ベース)

    ``FileWriter`` プロトコルを満たすため、サブクラスは以下を実装する:

    - ``extensions`` プロパティ
    - ``_render_bytes(content, ext) -> bytes`` メソッド

    ``requires`` と ``is_available`` はデフォルト実装 (依存なし) を提供。
    必要に応じてサブクラスでオーバーライドする。
    """

    @property
    @abstractmethod
    def extensions(self) -> frozenset[str]:
        """対応する拡張子集合"""

    @property
    def requires(self) -> list[str]:
        """必要な pip パッケージ名 (デフォルト: なし)"""
        return []

    def is_available(self) -> bool:
        """必要ライブラリがインストール済みか (デフォルト: True)"""
        return True

    @abstractmethod
    def _render_bytes(self, content: ExportContent, ext: str) -> bytes:
        """コンテンツを書出し用バイト列にレンダリングする

        Args:
            content: 書出し対象
            ext: 出力ファイル拡張子 (例: ``".html"``)。
                writer が単一拡張子のみ対応する場合は無視してよい。
        """

    def write(self, content: ExportContent, path: Path) -> WriteResult:
        """ファイルに書き出す共通実装"""
        from backend.export.media import BASE_DIR_METADATA_KEY

        ext = path.suffix.lower()
        # 相対画像パスの基準は **出力先のディレクトリ** (プロセスの CWD ではない)。
        # f_11 §4.1。write_to_bytes には出力先が無いので設定されない。
        if content.metadata is None:
            content.metadata = {}
        content.metadata[BASE_DIR_METADATA_KEY] = str(path.parent)
        data = self._render_bytes(content, ext)

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        except OSError as e:
            raise ExportError("write_error", f"Failed to write {path}: {e}")

        size = len(data)
        logger.debug(
            "%s: %s -> %d bytes", type(self).__name__, path.name, size,
        )
        # テンプレート継承の来歴 (c_16 §4.5.2 / f_11 §9.1)。テンプレート未使用の
        # ときは何も足さない。選ばれていたのに writer が継承を宣言しなかった場合
        # (拡張子の不一致 / 継承に対応しない形式) は ``template_applied=False`` に
        # 倒す — 「選ばれたが効かなかった」を「未使用」と区別できなくしない。
        result_metadata: dict[str, object] = {}
        # ``template_applied`` は **体裁の継承元 (template_base) が渡されたとき**の
        # 成否。構成だけの様式 (outline のみ) は ``template`` の来歴だけを持つ。
        if "template" in content.metadata:
            result_metadata["template"] = content.metadata["template"]
        if "template_base" in content.metadata:
            result_metadata["template_applied"] = (
                content.metadata.get("template_applied") is True
            )
        elif "template_applied" in content.metadata:
            result_metadata["template_applied"] = content.metadata["template_applied"]
        return WriteResult(path=path, size_bytes=size, metadata=result_metadata)

    def write_to_bytes(self, content: ExportContent, ext: str) -> bytes:
        """バイトデータとして書き出す (API ダウンロード用)"""
        return self._render_bytes(content, ext)
