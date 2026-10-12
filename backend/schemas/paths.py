"""モデル・ローカル状態のパス系スキーマ"""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.config_normalize import reject_retired_keys


class ModelPathsConfig(BaseModel):
    """モデルファイルパス

    標準 2 モデル (base/embed) と任意のクリエイト用 (create)・リランカー (rerank) を
    明示フィールドとして定義する。読み手の無いキーは typo とみなして ``extra="forbid"``
    で拒否する (G0 で撤去したモデル種は ``evoref config normalize`` が先に落とす)。
    """

    model_config = ConfigDict(extra="forbid")

    base_model: str = Field(default="models/gemma-4-12b-it-qat-q4_0.gguf", min_length=1)
    embed_model: str = "models/Qwen3-Embedding-0.6B-Q8_0.gguf"
    create_model: str | None = Field(
        default=None,
        description="クリエイトモード用 GGUF パス。未指定 (None / 空文字列) の場合は base_model にフォールバック",
    )
    rerank_model: str | None = Field(
        default=None,
        description="リランカー (rag.rerank) 用 GGUF パス。未指定なら rerank 用 llama-server は起動しない",
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_model_kinds(cls, data: object) -> object:
        """撤去したモデル種 (assist_model 等) は normalize の案内付きで拒否する (未知キーの一般の拒否より先)。"""
        if isinstance(data, dict):
            reject_retired_keys("model_paths", data)
        return data


class LocalPathsConfig(BaseModel):
    """利用者が変えられるデータパス (c_05 §0.2: ``outputs_dir`` だけ)。

    それ以外の置き場はデータ根 (``--data-root`` / ``EVOREF_DATA_ROOT`` /
    ``<install_root>/userdata``) から ``PathResolver.LAYOUT`` で導出する。
    未知のキーは typo とみなして ``extra="forbid"`` で拒否する。
    """

    model_config = ConfigDict(extra="forbid")

    # チャット / long_form が生成した成果物の既定の書込み先。ユーザーが
    # ディレクトリを指定しなかった裸のファイル名 (``compose.yaml``) はここへ
    # 落とす。指定しないとプロセスの CWD (= リポジトリ直下) に書かれる
    # (2026-09-08 監査 F-05)。明示パスを与えられた書込みはこの外へ出る。
    outputs_dir: str = Field(
        default="outputs/",
        min_length=1,
        description="ディレクトリ指定の無い生成物の既定の書込み先 (相対ならデータ根基準)。",
    )
