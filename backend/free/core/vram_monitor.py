"""VRAM 使用量モニタ

``scripts/launch_llama.py::estimate_vram_usage_mb`` が返す推定値を基本とし、
``nvidia-smi`` + ``LlamaProcessManager`` の PID レジストリが両方揃っている
場合のみ実測値で上書きする 2 段構成のフォールバックチェーンを提供する。

設計上の特徴:
    - ``nvidia-smi`` が存在しない / subprocess が失敗した環境 (AMD / Apple
      Silicon / Intel GPU / CI) では推定値のみを返す。エラーにはしない。
    - ``LlamaProcessManager`` が存在しない (``process_manager.enabled: false``)
      場合でも推定値表示は機能する。外部起動 (``launch_llama.py``) の場合は
      PID 追跡不可のため実測値は得られないが、推定値は問題なく返る。
    - ヒットした PID のモデルのみ ``source="actual"`` で実測値を返し、それ以外は
      ``source="estimate"`` で推定値を返す。トップレベル ``source`` は
      「全件が実測だった場合のみ ``actual``、それ以外は ``estimate``」。

チャット応答パスでは呼ばれないためパフォーマンス要件は緩い
(サブプロセス ``nvidia-smi`` 呼び出し許容)。
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.free.core.llama_process_manager import LlamaProcessManager

logger = get_logger("core.vram_monitor")


# 3 種モデルの固定順序 (GUI 表示順と合わせる)
MODEL_NAMES: tuple[str, ...] = ("base", "embed")


# LlamaProcessManager の component 名は "embedding"、本モジュールの GUI 表示 key は
# "embed"。nvidia-smi 突合時のみの差分なので局所マップで解消する。
_MODEL_TO_COMPONENT: dict[str, str] = {
    "base": "base",
    "embed": "embedding",
}


def _load_launch_llama(project_root: Path):
    """``scripts/launch_llama.py`` を動的ロードする

    LlamaProcessManager と同じ importlib 方式。sys.path 汚染を避けるため
    毎回 spec から load する。失敗時は None を返す。
    """
    launch_py = project_root / "scripts" / "launch_llama.py"
    if not launch_py.exists():
        logger.warning("launch_llama.py not found: %s", launch_py)
        return None
    try:
        spec = importlib.util.spec_from_file_location("_launch_llama_vram", launch_py)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception as e:
        logger.warning("Failed to load launch_llama.py: %s", e)
        return None


def nvidia_smi_available() -> bool:
    """``nvidia-smi`` が PATH に存在するか"""
    return shutil.which("nvidia-smi") is not None


def _parse_nvidia_smi_csv(output: str) -> dict[int, int]:
    """``nvidia-smi --query-compute-apps`` の CSV 出力を dict[pid]=used_mb にパース

    期待する行フォーマット (ヘッダ無し):
        ``<pid>, <used_memory_MiB>``

    例::

        12345, 1024 MiB
        12346, 512 MiB

    単位 ``MiB`` が付く場合と付かない場合があるため strip で両対応。
    空行 / パース失敗行は無視する。
    """
    result: dict[int, int] = {}
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        pid_str = parts[0]
        mem_str = parts[1]
        try:
            pid = int(pid_str)
        except ValueError:
            continue
        # "1024 MiB" / "1024" / "[N/A]" 等に対応
        mem_digits = "".join(ch for ch in mem_str if ch.isdigit())
        if not mem_digits:
            continue
        try:
            mem_mb = int(mem_digits)
        except ValueError:
            continue
        result[pid] = mem_mb
    return result


def nvidia_smi_snapshot(timeout_sec: float = 3.0) -> dict[int, int] | None:
    """``nvidia-smi --query-compute-apps=pid,used_memory`` を実行して dict で返す

    Returns:
        dict[pid, used_memory_mb]。

        - ``nvidia-smi`` が PATH に無い
        - subprocess 失敗 / タイムアウト
        - 出力が空

        のいずれでも ``None`` を返す (呼び出し側で推定値フォールバック)。
    """
    if not nvidia_smi_available():
        return None
    cmd = [
        "nvidia-smi",
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        logger.debug("nvidia-smi invocation failed: %s", e)
        return None
    if proc.returncode != 0:
        logger.debug(
            "nvidia-smi returned non-zero (%s): %s", proc.returncode, proc.stderr,
        )
        return None
    return _parse_nvidia_smi_csv(proc.stdout)


def _get_managed_pids(
    process_manager: LlamaProcessManager | None,
) -> dict[str, int]:
    """LlamaProcessManager から 3 モデルの PID を抽出する

    Returns:
        dict[model_name ("base"/"embed"), pid]。
        manager が None / 該当 component が未管理の場合はその key を含まない。
    """
    result: dict[str, int] = {}
    if process_manager is None:
        return result
    for model_name in MODEL_NAMES:
        component = _MODEL_TO_COMPONENT[model_name]
        try:
            entry = process_manager.get_entry(component)
        except Exception:
            entry = None
        if entry is None or entry.proc is None:
            continue
        pid = entry.proc.pid
        if pid is None:
            continue
        # プロセスがまだ生きているか軽くチェック
        if entry.proc.poll() is not None:
            continue
        result[model_name] = int(pid)
    return result


def collect_vram_status(
    cfg: dict,
    project_root: Path | None = None,
    *,
    process_manager: LlamaProcessManager | None = None,
    smi_snapshot_override: dict[int, int] | None = None,
) -> dict:
    """VRAM 使用量のスナップショットを生成する

    Args:
        cfg: ``config.yaml`` 全体の dict
        project_root: プロジェクトルート (省略時は ``Path.cwd()``)
        process_manager: LlamaProcessManager インスタンス (PID 取得用)
        smi_snapshot_override: テスト用に nvidia-smi 出力を差し込む場合の map

    Returns:
        ``GET /api/system/vram_status`` のレスポンス構造と同じ dict::

            {
              "source": "estimate" | "actual" | "mixed",
              "measurement_available": bool,
              "nvidia_smi_available": bool,
              "process_manager_enabled": bool,
              "models": [
                {"name": "base", "present": True, "vram_mb": 1105,
                 "gpu_layers": 999, "model_mb": 1105, "source": "estimate",
                 "placement": "GPU", "pid": null},
                ...
              ],
              "total_mb": int,
              "budget_mb": int | null,
              "over_budget": bool
            }

    推定値は ``scripts/launch_llama.py::estimate_vram_usage_mb`` を流用する。
    読み込みに失敗した場合はすべての ``present=False`` / ``vram_mb=0`` を返す
    (API 層で空配列を返すのではなくフォールバックを保証)。
    """
    if project_root is None:
        project_root = Path.cwd()

    mod = _load_launch_llama(project_root)

    # 推定値取得 (失敗時は全 present=False の骨格)
    if mod is None:
        estimates: dict[str, dict] = {
            name: {
                "model_mb": None, "gpu_layers": 0, "vram_mb": 0,
                "present": False, "path": "",
            }
            for name in MODEL_NAMES
        }
    else:
        try:
            # API ポーリング (10 秒間隔) で呼ばれるため
            # llama-fit-params subprocess を毎回起動するオーバーヘッドを
            # 避けるため Tier 2 (GGUF サイズ) のみ使う。Tier 1 は
            # ``scripts/launch_llama.py --all`` でのみ実行する。
            estimates = mod.estimate_vram_usage_mb(
                cfg, project_root, prefer_fit_params=False,
            )
        except Exception as e:
            logger.warning("estimate_vram_usage_mb failed: %s", e)
            estimates = {
                name: {
                    "model_mb": None, "gpu_layers": 0, "vram_mb": 0,
                    "present": False, "path": "",
                }
                for name in MODEL_NAMES
            }

    # ``estimate_vram_usage_mb`` は ``model_paths.<key>`` が空文字列のとき
    # ``project_root / ""`` = ``project_root`` 自身のサイズ (ディレクトリ、
    # Windows では 0) を拾ってしまい present=True になる。これを防ぐため、
    # cfg.model_paths の該当キーが空/未設定なら強制的に present=False に補正する。
    model_paths_cfg = cfg.get("model_paths", {}) or {}
    _KEY_MAP = {
        "base": "base_model",
        "embed": "embed_model",
    }
    for name, key in _KEY_MAP.items():
        raw_path = model_paths_cfg.get(key, "")
        if not raw_path:
            if name in estimates:
                estimates[name]["present"] = False
                estimates[name]["vram_mb"] = 0
                estimates[name]["model_mb"] = None

    # nvidia-smi 実測値取得
    smi_snapshot = (
        smi_snapshot_override
        if smi_snapshot_override is not None
        else nvidia_smi_snapshot()
    )
    smi_ok = smi_snapshot is not None and nvidia_smi_available()
    # override が明示されている場合はテスト目的なので measurement_available 判定に使う
    if smi_snapshot_override is not None:
        smi_ok = True

    # LlamaProcessManager から PID 取得
    pid_map = _get_managed_pids(process_manager)
    process_manager_enabled = process_manager is not None

    # 4 モデルを仕様通りの順序で並べる
    model_entries: list[dict] = []
    actual_count = 0
    present_count = 0
    for name in MODEL_NAMES:
        est = estimates.get(name, {})
        present = bool(est.get("present"))
        gpu_layers = int(est.get("gpu_layers", 0) or 0)
        model_mb = est.get("model_mb")
        est_vram_mb = int(est.get("vram_mb", 0) or 0)
        placement = "GPU" if gpu_layers > 0 else "CPU"

        pid: int | None = pid_map.get(name)
        # 実測値は smi_snapshot + pid が両方揃っているケースのみ
        actual_vram: int | None = None
        if (
            smi_snapshot is not None
            and pid is not None
            and pid in smi_snapshot
        ):
            actual_vram = int(smi_snapshot[pid])

        # source / vram_mb を決定
        if actual_vram is not None:
            source = "actual"
            vram_mb = actual_vram
            actual_count += 1
        else:
            source = "estimate"
            vram_mb = est_vram_mb

        if present:
            present_count += 1

        model_entries.append({
            "name": name,
            "present": present,
            "vram_mb": vram_mb,
            "gpu_layers": gpu_layers,
            "model_mb": model_mb,
            "source": source,
            "placement": placement if present else "none",
            "pid": pid,
        })

    total_mb = sum(m["vram_mb"] for m in model_entries)
    runtime_cfg = cfg.get("runtime", {}) or {}
    budget_mb_raw = runtime_cfg.get("total_vram_budget_mb")
    budget_mb = int(budget_mb_raw) if budget_mb_raw is not None else None
    over_budget = bool(
        budget_mb is not None and total_mb > budget_mb,
    )

    # トップレベル source: 全件実測なら "actual"、1 件以上実測なら "mixed"、それ以外 "estimate"
    if present_count > 0 and actual_count == present_count:
        top_source = "actual"
    elif actual_count > 0:
        top_source = "mixed"
    else:
        top_source = "estimate"

    measurement_available = smi_ok and process_manager_enabled

    return {
        "source": top_source,
        "measurement_available": measurement_available,
        "nvidia_smi_available": bool(smi_ok),
        "process_manager_enabled": process_manager_enabled,
        "models": model_entries,
        "total_mb": total_mb,
        "budget_mb": budget_mb,
        "over_budget": over_budget,
    }
