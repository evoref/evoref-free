#!/usr/bin/env bash
# evoref サービス管理スクリプト (macOS / Linux)
# Usage: evoref-ctl.sh {start|stop|restart|status}
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_lib.sh"
cd "$PROJECT_ROOT"

PIDS=()

# ── サービス起動 ──
start_all() {
    # 依存コマンド確認
    check_dependency python3 "python.org or brew install python3" || exit 1
    check_dependency npm "nodejs.org or brew install node" || exit 1

    activate_venv
    setup_utf8

    # 版 (config_version) の無い旧形式の config.yaml は起動前に一度だけ直す
    # (config.yaml.g0-<stamp> へ退避)。別の evoref がロックを持っていれば拒否。
    python -m backend.free.cli.main config normalize --if-needed || exit 1

    # PC の環境が変わっていれば llama-server を起こす前に 1 回だけ確認する
    # (標準入力が端末でなければ聞かず pending のまま。失敗しても起動は止めない)。
    python -m backend.free.cli.main tune --startup-check || true

    # 前回の rerank 起動断念の印を --all を起こす前に消す (--all 側でも消すが、埋め込みが llama-cpp で
    # ない構成では下の --wait-embed が即座に返り、--all が消す前に --wait-rerank が古い印を読みうる)
    python scripts/launch_llama.py config.yaml --clear-rerank-abandoned || true

    echo "[start] Starting llama-server (base + embedding + rerank if enabled)..."
    python scripts/launch_llama.py config.yaml --all &
    PIDS+=($!)
    sleep 3

    # 埋め込みサーバを待つ。embedding.gpu_layers: auto で PC か埋め込みモデルが変わった回は
    # 配置 (GPU / CPU) の判別が先に走り、GPU で起動しなければ CPU で起こし直すので長めに待つ。
    python scripts/launch_llama.py config.yaml --wait-embed 240

    # リランカー (rag.rerank、既定 on。モデルファイルがあるときだけ起動) の自己テスト (PC が変わったときだけ) を待つ。
    # backend は起動時に結果を 1 回読むだけなので先に終わらせる。off なら即終了。
    python scripts/launch_llama.py config.yaml --wait-rerank 240

    echo "[start] Starting FastAPI backend on :8000..."
    uvicorn backend.main:app --host 127.0.0.1 --port 8000 &
    PIDS+=($!)

    echo "[start] Starting SvelteKit dev server on :5173..."
    cd frontend
    npm run dev -- --host 127.0.0.1 &
    PIDS+=($!)
    cd "$PROJECT_ROOT"

    echo ""
    echo "=== evoref is running ==="
    echo "  Web UI:   http://localhost:5173"
    echo "  API:      http://localhost:8000"
    echo "  llama:    http://localhost:8080 (base)"
    echo "  embed:    http://localhost:8082 (embedding, if llama-cpp)"
    echo "  rerank:   http://localhost:8083 (reranker, if rag.rerank.mode is not off)"
    echo ""
    echo "Press Ctrl+C to stop all services"

    wait
}

# ── サービス停止 ──
stop_all() {
    echo "[stop] Stopping evoref services..."

    # 起動スクリプトを先に止める (配置の判別中に llama-server だけ止めると、起動スクリプトが
    # 判別を諦めて本番の埋め込みサーバを起こしてしまう)
    pkill -f "scripts/launch_llama.py" 2>/dev/null || true

    pkill -f "llama-server" 2>/dev/null \
        && echo "  llama-server stopped" \
        || echo "  llama-server not running"

    pkill -f "uvicorn backend.main:app" 2>/dev/null \
        && echo "  FastAPI backend stopped" \
        || echo "  FastAPI backend not running"

    pkill -f "vite.*--host" 2>/dev/null \
        && echo "  SvelteKit frontend stopped" \
        || echo "  SvelteKit frontend not running"

    echo ""
    echo "=== All services stopped ==="
}

# ── サービス状態表示 ──
show_status() {
    echo "=== evoref service status ==="

    local llama_pid
    llama_pid=$(pgrep -f "llama-server" 2>/dev/null | head -1) || true
    if [[ -n "$llama_pid" ]]; then
        echo "  llama-server:       running (PID $llama_pid)"
    else
        echo "  llama-server:       stopped"
    fi

    local uvicorn_pid
    uvicorn_pid=$(pgrep -f "uvicorn backend.main" 2>/dev/null | head -1) || true
    if [[ -n "$uvicorn_pid" ]]; then
        echo "  FastAPI backend:    running (PID $uvicorn_pid)"
    else
        echo "  FastAPI backend:    stopped"
    fi

    local vite_pid
    vite_pid=$(pgrep -f "vite.*--host" 2>/dev/null | head -1) || true
    if [[ -n "$vite_pid" ]]; then
        echo "  SvelteKit frontend: running (PID $vite_pid)"
    else
        echo "  SvelteKit frontend: stopped"
    fi
}

# ── クリーンアップ（Ctrl+C / 終了時） ──
cleanup() {
    if [[ ${#PIDS[@]} -gt 0 ]]; then
        echo ""
        echo "[stop] Shutting down..."
        for pid in "${PIDS[@]}"; do
            kill "$pid" 2>/dev/null || true
        done
        wait 2>/dev/null || true
        echo "[stop] All processes stopped"
    fi
}

trap cleanup EXIT INT TERM

# ── ヘルプ表示 ──
usage() {
    echo "Usage: $(basename "$0") {start|stop|restart|status}"
    echo ""
    echo "Commands:"
    echo "  start    Start all services (llama-server, FastAPI, SvelteKit)"
    echo "  stop     Stop all running services"
    echo "  restart  Restart all services"
    echo "  status   Show status of all services"
    exit 1
}

# ── メイン ──
case "${1:-}" in
    start)   start_all ;;
    stop)    stop_all ;;
    restart) stop_all; sleep 1; start_all ;;
    status)  show_status ;;
    *)       usage ;;
esac
