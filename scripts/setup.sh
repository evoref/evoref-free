#!/usr/bin/env bash
# evoref 初回セットアップスクリプト (macOS / Linux)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_lib.sh"
cd "$PROJECT_ROOT"

# ── 引数解析 ──
SHARED_PATH=""
FORCE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --shared-path)
            if [[ -z "${2:-}" ]]; then
                echo "ERROR: --shared-path requires a path argument"
                echo "Usage: setup.sh [--shared-path <path>] [--force]"
                exit 1
            fi
            SHARED_PATH="$2"
            shift 2
            ;;
        --force)
            FORCE=1
            shift
            ;;
        -h|--help)
            echo "Usage: setup.sh [--shared-path <path>] [--force]"
            echo ""
            echo "Options:"
            echo "  --shared-path <path>  NAS shared path for multi-PC setup"
            echo "                        Uses shared models/ (no model checks)"
            echo "  --force               Force reinstall (recreate .venv, reinstall packages,"
            echo "                        overwrite config.yaml)"
            echo "  -h, --help            Show this help message"
            exit 0
            ;;
        *)
            echo "ERROR: Unknown option: $1"
            echo "Usage: setup.sh [--shared-path <path>] [--force]"
            exit 1
            ;;
    esac
done

# ── 依存コマンド確認 ──
check_dependency python3 "python.org or brew install python3" || exit 1
check_dependency npm "nodejs.org or brew install node" || exit 1
check_dependency git "git-scm.com or brew install git" || exit 1

# ── モデルパス検証 ──
if [[ -n "$SHARED_PATH" ]]; then
    if [[ -d "$SHARED_PATH" ]]; then
        SHARED_PATH="$(cd "$SHARED_PATH" && pwd)"
    else
        echo "ERROR: Shared path does not exist: $SHARED_PATH"
        exit 1
    fi
    if [[ ! -r "$SHARED_PATH" ]]; then
        echo "ERROR: Shared path is not readable: $SHARED_PATH"
        exit 1
    fi
    if [[ ! -d "$SHARED_PATH/models" ]]; then
        echo "ERROR: models/ directory not found in shared path: $SHARED_PATH"
        echo "Expected structure:"
        echo "  $SHARED_PATH/"
        echo "  └── models/    (GGUF model files)"
        exit 1
    fi
fi

echo "=== evoref setup ==="
echo "Project root: $PROJECT_ROOT"
if [[ -n "$SHARED_PATH" ]]; then
    echo "Shared path:  $SHARED_PATH"
fi
if [[ -n "$FORCE" ]]; then
    echo "Mode:          FORCE REINSTALL"
fi
echo ""

# ── 1. Python 仮想環境 ──
echo "[1/6] Creating Python virtual environment..."
if [[ -n "$FORCE" ]] && [ -d ".venv" ]; then
    echo "  [force] Removing existing .venv..."
    rm -rf .venv
fi
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "  Created .venv"
else
    echo "  .venv already exists, skipping"
fi

source .venv/bin/activate

# ── 2. Python パッケージ ──
echo "[2/6] Installing Python dependencies..."
python -m pip install --upgrade pip -q
if [[ -n "$FORCE" ]]; then
    python -m pip install --force-reinstall -r backend/requirements.txt -q
    python -m pip install --force-reinstall -e . -q
else
    python -m pip install -r backend/requirements.txt -q
    python -m pip install -e . -q
fi
echo "  Done"

# ── 3. フロントエンド依存 ──
echo "[3/6] Installing frontend dependencies..."
if [[ -n "$FORCE" ]] && [ -d "frontend/node_modules" ]; then
    echo "  [force] Removing existing node_modules..."
    rm -rf frontend/node_modules
fi
cd frontend
npm install --silent
cd "$PROJECT_ROOT"
echo "  Done"

# ── 4. config.yaml ──
echo "[4/6] Setting up config.yaml..."
if [[ -n "$FORCE" ]] && [ -f "config.yaml" ]; then
    echo "  [force] Overwriting existing config.yaml"
    rm config.yaml
fi
if [ ! -f "config.yaml" ]; then
    if [ -f "config.yaml.example" ]; then
        cp config.yaml.example config.yaml
        echo "  Copied config.yaml.example -> config.yaml"
    else
        echo "  WARNING: config.yaml.example not found. Please create config.yaml manually."
    fi
else
    echo "  config.yaml already exists, skipping"
fi

if [[ -n "$SHARED_PATH" ]] && [ -f "config.yaml" ]; then
    # GGUF モデルファイルを検索
    GGUF_PATH=""
    for f in "$SHARED_PATH"/models/*.gguf; do
        if [[ -e "$f" ]]; then
            GGUF_PATH="$f"
            echo "  Found model: $(basename "$GGUF_PATH")"
            break
        fi
    done
    if [[ -z "$GGUF_PATH" ]]; then
        GGUF_PATH="$SHARED_PATH/models/gemma-4-12b-it-qat-q4_0.gguf"
        echo "  WARNING: No .gguf file found in $SHARED_PATH/models/"
        echo "           Please update model_paths.base_model in config.yaml manually."
    fi
    # config.yaml の model_paths を更新
    python scripts/configure_shared_path.py "$SHARED_PATH" "$GGUF_PATH"
    echo "  Updated config.yaml with shared paths"
fi

# ── 5. モデル配置チェック ──
echo "[5/6] Checking models..."
if [[ -n "$SHARED_PATH" ]]; then
    echo "  Using shared path. Models expected at: $SHARED_PATH/models/"
else
    # 自動ダウンロードは廃止。GGUF は models/ へ手動配置する。
    # 未配置でも setup は継続させる (チェックは情報提供のみ)。
    python scripts/download_model.py || true
fi

# ── 6. ローカルディレクトリ ──
echo "[6/6] Creating local directories..."
ensure_directories
echo "  Done"

echo ""
echo "=== Setup complete ==="
echo ""
if [[ -n "$SHARED_PATH" ]]; then
    echo "Shared path configured: $SHARED_PATH"
    echo ""
    echo "Next steps:"
    echo "  1. Install llama-server from https://github.com/ggml-org/llama.cpp/releases"
    echo "  2. Verify config.yaml settings (gpu_layers, context_size, etc.)"
    echo "  3. Run: ./scripts/evoref-ctl.sh start"
else
    echo "Next steps:"
    echo "  1. Install llama-server from https://github.com/ggml-org/llama.cpp/releases"
    echo "  2. Edit config.yaml to set gpu_layers and context_size for your hardware"
    echo "  3. Run: ./scripts/evoref-ctl.sh start"
fi
echo ""
