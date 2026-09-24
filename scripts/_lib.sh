#!/usr/bin/env bash
# evoref 共通関数ライブラリ
# 他のスクリプトから source して使用する

# ── プロジェクトルート解決 ──
# _lib.sh は常に scripts/ 配下にあるため、その親がプロジェクトルート
_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$_LIB_DIR/.." && pwd)"

# ── 依存コマンド確認 ──
# usage: check_dependency <command> [install_hint]
# 戻り値: 0=存在, 1=不在（エラーメッセージ出力済み）
check_dependency() {
    local cmd="$1"
    local install_hint="${2:-}"
    if ! command -v "$cmd" &>/dev/null; then
        echo "ERROR: '$cmd' is not installed."
        if [[ -n "$install_hint" ]]; then
            echo "  Install: $install_hint"
        fi
        return 1
    fi
    return 0
}

# ── 仮想環境の有効化 ──
activate_venv() {
    if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
        source "$PROJECT_ROOT/.venv/bin/activate"
    fi
}

# ── UTF-8 環境設定 ──
setup_utf8() {
    export PYTHONUTF8=1
}

# ── データ根のディレクトリ一括作成 ──
# データ根は backend.data_root が決める (EVOREF_DATA_ROOT → <install_root>/userdata)。
# 不正な指定はそこで拒否される。local/ (G0) は作らない・動かさない・消さない。
ensure_directories() {
    local data_root
    data_root="$(cd "$PROJECT_ROOT" && python -c 'from backend.data_root import resolve_data_root; print(resolve_data_root())')" || {
        echo "ERROR: could not resolve the data root (check EVOREF_DATA_ROOT)"
        return 1
    }
    echo "  Data root: $data_root"
    local dirs=(store logs outputs themes profiles tmp run cache)
    for dir in "${dirs[@]}"; do
        mkdir -p "$data_root/$dir"
    done
}
