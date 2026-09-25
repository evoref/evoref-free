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
# 不正な指定はそこで拒否される。レイアウト (世代フォルダ g<N>/store・g<N>/cache を含む) は
# PathResolver.ensure_local_dirs が作る。
ensure_directories() {
    local data_root
    data_root="$(cd "$PROJECT_ROOT" && python -c 'from backend.config import PathResolver; from backend.data_root import install_root, resolve_data_root; r = PathResolver({}, install_root(), data_root=resolve_data_root()); r.ensure_local_dirs(); print(r.data_root)')" || {
        echo "ERROR: could not resolve the data root (check EVOREF_DATA_ROOT)"
        return 1
    }
    echo "  Data root: $data_root"
}
