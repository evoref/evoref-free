# evoref (Free Edition)

evoref は **完全ローカル動作** の自己進化型 LLM アシスタントです。外部 API・クラウドに一切依存せず、llama.cpp (llama-server) 上で GGUF モデルを動かし、RAG・メモリ・自律実行ループ・自己学習を統合します。

会話を重ねるほど、経験記録とプロンプト進化を通じてあなたの環境に適応していきます。

---

## 動作条件

| 項目 | 要件 |
|------|------|
| OS | Windows 11 / macOS / Linux |
| Python | 3.12 以上 |
| Node.js | npm 同梱の LTS |
| Git | 最新版 |
| llama-server | [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) の build **b8946 以上**を推奨 |
| GPU / VRAM | GPU 推奨だが **CPU only でも動作** (`llama.gpu_layers: 0`)。Qwen3.5-9B Q4_K_M + KV キャッシュで VRAM **3〜5GB** が目安。埋め込み・リランカーは既定で CPU 配置 |
| ディスク | GGUF モデル計 数 GB (base ~2.6GB / assist ~1.4GB / embedding / reranker) |

llama-server は setup では導入されません。**別途インストールして PATH を通す**必要があります。

---

## インストール

### 初回セットアップ

```powershell
# Windows / PowerShell
.\scripts\setup.bat
```

```bash
# macOS / Linux
./scripts/setup.sh
```

setup は以下を一括実行します:

1. Python 仮想環境 (`.venv`) 作成
2. Python 依存インストール (`backend/requirements.txt` + `pip install -e .`)
3. フロントエンド依存 (`npm install`)
4. `config.yaml.example` → `config.yaml` コピー
5. モデルダウンロード (対話式 Y/n): base (Qwen3.5-9B) / assist (Qwen3.5-4B) / embedding (Qwen3-Embedding-0.6B) / reranker (Qwen3-Reranker-4B)
6. `local/` データディレクトリ生成

#### オプション

```powershell
.\scripts\setup.bat --shared-path <NAS_PATH>   # 複数 PC で models/ を共有 (DL スキップ)
.\scripts\setup.bat --force                    # .venv / config.yaml / models を強制再構築
```

---

## 起動

```powershell
# 1. config.yaml を編集 (gpu_layers / context_size をハードウェアに合わせる)
# 2. 一括起動 (llama-server 群 + FastAPI + SvelteKit)
.\scripts\evoref-ctl.bat start
.\scripts\evoref-ctl.bat status
.\scripts\evoref-ctl.bat stop
```

CLI (venv 有効化後):

```powershell
evoref serve   # backend + llama.cpp サーバ群
evoref chat    # 対話モード
evoref gui     # 既定ブラウザで Web UI を開く
```

### ポート

| サービス | ポート |
|----------|--------|
| Web UI | 5173 |
| backend (API) | 8000 |
| llama-server (base / assist / embed / rerank) | 8080 / 8081 / 8082 / 8083 |

---

## ライセンス

Apache-2.0 ([LICENSE](LICENSE))。サードパーティモデルの帰属は [NOTICE.md](NOTICE.md) を参照。
</content>
