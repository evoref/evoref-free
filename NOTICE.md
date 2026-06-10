# NOTICE

evoref (Free)
Copyright 2026 hiromimie

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the [LICENSE-FREE](LICENSE-FREE) file
at the root of this repository, or at:

    http://www.apache.org/licenses/LICENSE-2.0

Pro 拡張 (`backend/pro/` / `frontend/src/lib/pro/`) はプロプライエタリ配布で、
契約者向けバンドルにのみ含まれ、別途 [LICENSE-PRO](LICENSE-PRO) (Pro EULA) が
適用されます。詳細は [docs/e_01_licensing.md](docs/e_01_licensing.md) を参照。

---

# Third-Party Notices

本プロジェクトは以下のサードパーティ製モデル・ソフトウェアを使用しています。
モデル本体は同梱せず、`scripts/setup.sh` / `setup.bat` (内部で `scripts/download_model.py` を呼ぶ) が
Hugging Face から `models/` 配下にダウンロードします。モデル自体の改変は行っていません。

`config.yaml` の `model_paths` を変更することで、別の GGUF モデルにも差し替え可能です。
標準構成は `config.yaml.example` を参照してください。

---

## Qwen3.5-9B (ベースモデル — 既定)

- **提供元**: Qwen Team (Alibaba Cloud) / GGUF 配布: Unsloth
- **リポジトリ**: https://huggingface.co/unsloth/Qwen3.5-9B-GGUF
- **ライセンス**: Apache License 2.0
- **用途**: チャット / コーディングモードの応答生成 (llama-server で実行)。

### Attribution / 帰属表示

> Qwen3.5-9B is developed and provided by [Qwen Team](https://huggingface.co/Qwen) (Alibaba Cloud).
> Licensed under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

---

## Qwen3.5-4B (アシストモデル — 既定)

- **提供元**: Qwen Team (Alibaba Cloud) / GGUF 配布: Unsloth
- **リポジトリ**: https://huggingface.co/unsloth/Qwen3.5-4B-GGUF
- **ライセンス**: Apache License 2.0
- **用途**: メモリ抽出 / RAG 補助 / 要約 / 判定 等 (ベースモデルとは別 llama-server インスタンスで実行)。

### Attribution / 帰属表示

> Qwen3.5-4B is developed and provided by [Qwen Team](https://huggingface.co/Qwen) (Alibaba Cloud).
> Licensed under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

---

## Qwen3-Embedding-0.6B (埋め込みモデル — 既定)

- **提供元**: Qwen Team (Alibaba Cloud)
- **リポジトリ**: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF
- **ライセンス**: Apache License 2.0
- **用途**: テキストの埋め込み (embedding) 生成。RAG / SemMem / カートリッジで使用 (1024 次元)。

### Attribution / 帰属表示

> Qwen3-Embedding-0.6B is developed and provided by [Qwen Team](https://huggingface.co/Qwen) (Alibaba Cloud).
> Licensed under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

---

## 過去に使用していたモデル (現在は不使用)

### BGE-M3

- **提供元**: BAAI
- **リポジトリ**: https://huggingface.co/BAAI/bge-m3
- **ライセンス**: MIT License
- **備考**: 以前の埋め込みモデル候補として評価。Qwen3-Embedding-0.6B を既定に切り替え済 (`config.yaml` で引き続き選択可能)。

### ruri-v3-130m

- **提供元**: cl-nagoya (名古屋大学)
- **リポジトリ**: https://huggingface.co/cl-nagoya/ruri-v3-130m
- **ライセンス**: CC BY-SA 4.0 (Creative Commons Attribution-ShareAlike 4.0 International)
- **備考**: 初期に評価していた埋め込みモデル。CC BY-SA 4.0 の Copyleft 制約とライセンス互換性を踏まえ、現在は使用していない。
