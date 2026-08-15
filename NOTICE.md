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
モデル本体は同梱せず、また自動ダウンロードも行いません。利用者が GGUF を
`models/` 配下へ手動配置します (`scripts/download_model.py` は配置状況の確認のみ)。
モデル自体の改変は行っていません。

`config.yaml` の `model_paths` を変更することで、別の GGUF モデルにも差し替え可能です。
標準構成は `config.yaml.example` を参照してください。

---

## Qwen3.5-9B (ベースモデル — 既定)

- **提供元**: Qwen Team (Alibaba Cloud) / GGUF 配布: Unsloth
- **リポジトリ**: https://huggingface.co/unsloth/Qwen3.5-9B-GGUF
- **ライセンス**: Apache License 2.0
- **用途**: チャット / クリエイトモードの応答生成 (llama-server で実行)。

### Attribution / 帰属表示

> Qwen3.5-9B is developed and provided by [Qwen Team](https://huggingface.co/Qwen) (Alibaba Cloud).
> Licensed under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

---

## Qwen3.5-4B (補助タスク — 既定)

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

## BGE-M3 (埋め込みモデル — 代替選択肢)

- **提供元**: BAAI (Beijing Academy of Artificial Intelligence)
- **リポジトリ**: https://huggingface.co/BAAI/bge-m3
  (GGUF 配布例: https://huggingface.co/ggml-org/bge-m3-Q8_0-GGUF)
- **ライセンス**: MIT License
- **用途**: テキストの埋め込み (embedding) 生成 (1024 次元)。Qwen3-Embedding-0.6B の
  代替として `model_paths.embed_model` を差し替えることで選択可能 (既定は変更していない)。

### Attribution / 帰属表示

> BGE-M3 is developed and provided by [BAAI](https://huggingface.co/BAAI).
> Licensed under the [MIT License](https://opensource.org/licenses/MIT).

---

## 過去に使用していたモデル (現在は不使用)

### ruri-v3-130m

- **提供元**: cl-nagoya (名古屋大学)
- **リポジトリ**: https://huggingface.co/cl-nagoya/ruri-v3-130m
- **ライセンス**: CC BY-SA 4.0 (Creative Commons Attribution-ShareAlike 4.0 International)
- **備考**: 初期に評価していた埋め込みモデル。CC BY-SA 4.0 の Copyleft 制約とライセンス互換性を踏まえ、現在は使用していない。
