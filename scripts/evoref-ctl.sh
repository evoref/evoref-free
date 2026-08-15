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

    echo "[start] Starting llama-server (base + embedding)..."
    python scripts/launch_llama.py config.yaml --all &
    PIDS+=($!)
    sleep 3

    echo "[start] Starting FastAPI backend on :8000..."
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
    PIDS+=($!)

    echo "[start] Starting SvelteKit dev server on :5173..."
    cd frontend
    npm run dev -- --host 0.0.0.0 &
    PIDS+=($!)
    cd "$PROJECT_ROOT"

    echo ""
    echo "=== evoref is running ==="
    echo "  Web UI:   http://localhost:5173"
    echo "  API:      http://localhost:8000"
    echo "  llama:    http://localhost:8080 (base)"
    echo "  embed:    http://localhost:8082 (embedding, if llama-cpp)"
    echo ""
    echo "Press Ctrl+C to stop all services"

    wait
}

# ── サービス停止 ──
stop_all() {
    echo "[stop] Stopping evoref services..."

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
