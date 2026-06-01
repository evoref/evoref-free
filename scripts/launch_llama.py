"""llama-server 起動スクリプト

config.yaml の llama / embedding / reranker セクションから起動コマンドを組み立て、
サブプロセスとして llama-server を起動する。

サブコマンド:
  (なし)     ベースモデル llama-server のみ起動
  --all      ベース + エンベッド + リランカー（設定時）を一括起動
  --embed    エンベッド用 llama-server のみ起動
  --reranker リランカー用 llama-server のみ起動

  --all 起動時に ``runtime.total_vram_budget_mb`` (config.yaml) を参照し、
  GPU オフロード対象モデルの VRAM 使用量推定合計が予算を超過する場合は
  警告してアボートする。``--force`` で強制起動可能。
  埋め込み / リランカー の ``-ngl`` 既定値は 0 (CPU フォールバック) とし、
  GPU 割り当ては ``embedding.gpu_layers`` / ``reranker.gpu_layers`` による
  明示 opt-in でのみ有効化される。

  起動時に ``llama-server --version`` を実行し build 番号をログに出力する。
  ``runtime.min_llamacpp_build`` 未満を検出すると stderr に警告を出し、
  ``runtime.enforce_min_llamacpp_build: true`` の場合は exit code 3 で
  アボートする。バイナリが build 番号を露出しないカスタムビルドの場合は
  検出失敗 (None) として警告のみで継続。

  llama.cpp
  slot 退避機構 (``--cache-ram`` / ``--cache-idle-slots``) を全 4 サーバ
  (base / assist / embed / reranker) で明示制御する。``cache_ram_mib`` /
  ``cache_idle_slots`` を ``config.yaml`` から駆動し、上流のデフォルト
  (8192 MiB / true) を黙従する状態を解消する。``slots > 1`` かつ
  ``cache_ram_mib > 0`` の場合は idle slot offload が動作するよう
  ``--kv-unified`` を自動付与する (``kv_unified`` で明示 override 可)。

  バイナリの ``-fitp on`` フラグを使い、device 別の使用 MiB (model /
  context / compute) を取得して VRAM 推定を 2 段構え化する。Tier 1 が
  バイナリ未存在 / タイムアウトで失敗した場合は GGUF ファイルサイズ
  ベースの Tier 2 (旧来挙動) にサイレントフォールバックする。
  ``runtime.fit_params_enabled: false`` で Tier 1 を一律スキップ可能。
  ``runtime.total_vram_budget_mb`` 未設定時は Tier 1 結果から 10%
  ヘッドルームを足した推奨予算を起動ログに表示する。

  ``--reasoning-budget`` / ``--reasoning-budget-message`` を
  ``build_assist_cmd`` から付与し、thinking モデル (Qwen3 / Gemma-4 等)
  をアシスト用途で使用する際の token 浪費を OS-fence で抑止する。
  従来の per-request ``chat_template_kwargs.enable_thinking=false``
  との二重防御で、サーバ側 fence 失敗時にもフォールバック可能。
  self-speculative decoding (``--spec-default`` / ``--spec-type ngram-*``) を
  Pro 限定で有効化する。``evoref code`` モード
  の base モデルが対象。``EVOREF_EDITION`` 環境変数 + ``backend.pro`` パッケー
  ジ存在で Pro 判定し、Free 環境では ``llama.speculative.enabled=true`` でも
  warning + 無効化する。assist / embed / reranker は対象外 (Issue 文 §スコープ)。
"""

import os
import re
import struct
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import httpx
import yaml


def _resolve_embed_gpu_layers(cfg: dict) -> int:
    """埋め込み用 ``-ngl`` の解決

    ``embedding.gpu_layers`` が明示されていればそれを、未指定の場合は CPU
    フォールバックとして 0 を返す。ベースモデル ``llama.gpu_layers`` には
    追従しない (GPU 割り当ては opt-in)。
    """
    emb_cfg = cfg.get("embedding", {}) or {}
    gpu_layers = emb_cfg.get("gpu_layers")
    if gpu_layers is None:
        return 0
    return int(gpu_layers)


def _resolve_reranker_gpu_layers(cfg: dict) -> int:
    """リランカー用 ``-ngl`` の解決

    ``reranker.gpu_layers`` が明示されていればそれを、未指定の場合は CPU
    フォールバックとして 0 を返す。
    """
    reranker_cfg = cfg.get("reranker", {}) or {}
    gpu_layers = reranker_cfg.get("gpu_layers")
    if gpu_layers is None:
        return 0
    return int(gpu_layers)


def _resolve_assist_gpu_layers(cfg: dict, project_root: Path | None = None) -> int:
    """アシスト用 ``-ngl`` の解決 (従来挙動: llama 継承)

    ``assist_model.local.gpu_layers`` または ``llama.gpu_layers`` が
    ``"auto"`` の場合は ``_resolve_auto_gpu_layers`` 経由でキャッシュ済み
    計算結果 (assist 用) を使用する。project_root が未指定の場合は
    auto 計算を skip し既定 999 にフォールバック。
    """
    assist_cfg = cfg.get("assist_model", {}) or {}
    local_cfg = assist_cfg.get("local", {}) or {}
    gpu_layers = local_cfg.get("gpu_layers")
    if gpu_layers is None:
        gpu_layers = (cfg.get("llama", {}) or {}).get("gpu_layers", 999)
    if isinstance(gpu_layers, str) and gpu_layers == "auto":
        if project_root is not None:
            cache = _resolve_auto_gpu_layers(cfg, project_root)
            if cache is not None:
                return int(cache["assist_ngl"])
        return 999  # auto 不能 → 安全側で全 offload (既存挙動)
    return int(gpu_layers)


def _resolve_base_gpu_layers(cfg: dict, project_root: Path | None = None) -> int:
    """ベース ``-ngl`` の解決。

    ``llama.gpu_layers`` が ``"auto"`` の場合は ``_resolve_auto_gpu_layers``
    経由でキャッシュ済み計算結果 (base 用) を使用する。project_root が
    未指定の場合は auto 計算を skip し既定 999 にフォールバック。
    数値指定時は従来挙動 (int に変換してそのまま返す)。
    """
    gpu_layers = (cfg.get("llama", {}) or {}).get("gpu_layers", 999)
    if isinstance(gpu_layers, str) and gpu_layers == "auto":
        if project_root is not None:
            cache = _resolve_auto_gpu_layers(cfg, project_root)
            if cache is not None:
                return int(cache["base_ngl"])
        return 999  # auto 不能 → 安全側で全 offload (既存挙動)
    return int(gpu_layers)


# ── cache-ram / cache-idle-slots / kv-unified 解決 ──────


def _resolve_cache_ram_mib(section_cfg: dict, default: int) -> int:
    """``cache_ram_mib`` の解決。

    値の意味は llama.cpp 上流に従う::
        -1: 無制限 (RAM ある限り全 idle slot を退避)
         0: disable (idle slot offload OFF、上流デフォルト 8192 を打ち消す)
        >0: その MiB 数を上限に RAM へ退避バッファを確保

    ``config.yaml`` 未指定時は呼び出し側が指定した ``default`` を返す。
    """
    value = section_cfg.get("cache_ram_mib", default)
    return int(value)


def _resolve_cache_idle_slots(section_cfg: dict, default: bool = True) -> bool:
    """``cache_idle_slots`` の解決

    True (上流デフォルト) は idle slot を offload 対象にする。False は
    ``--no-cache-idle-slots`` を明示付与して機構自体を OFF にする。
    """
    value = section_cfg.get("cache_idle_slots", default)
    return bool(value)


def _resolve_kv_unified(
    section_cfg: dict, slots: int, cache_ram_mib: int,
) -> bool:
    """``kv_unified`` の解決

    上流挙動:
      - ``-np 1`` (slots 自動 / 単一 slot): unified KV cache がデフォルト ON
      - ``-np >1`` (明示複数 slot): デフォルトで非 unified 配置 (slot 毎に
        独立 KV cell プール) になり、``--cache-ram`` の idle slot offload が
        動作しない

    したがって ``cache_ram_mib > 0 AND slots > 1`` の場合は ``--kv-unified``
    を自動付与する (auto)。``kv_unified`` を ``true``/``false`` で明示指定
    した場合は auto 判定を上書きしてその値をそのまま尊重する。
    """
    explicit = section_cfg.get("kv_unified")
    if explicit is None:
        return cache_ram_mib > 0 and slots > 1
    return bool(explicit)


def _append_cache_ram_args(
    cmd: list[str], section_cfg: dict, *, default_mib: int,
) -> None:
    """``--cache-ram`` / ``--no-cache-idle-slots`` を ``cmd`` に追加する
    (chat slots を持つ base / assist 用)。

    ``cache_ram_mib`` は常に明示付与し、上流デフォルト 8192 の黙従を
    回避する。``cache_idle_slots: false`` のときのみ ``--no-cache-idle-slots``
    を付与する (上流デフォルト true は何も付けない)。
    """
    cache_ram_mib = _resolve_cache_ram_mib(section_cfg, default=default_mib)
    cmd += ["--cache-ram", str(cache_ram_mib)]
    if not _resolve_cache_idle_slots(section_cfg, default=True):
        cmd += ["--no-cache-idle-slots"]


def _append_kv_unified_args(
    cmd: list[str], section_cfg: dict, *, slots: int, default_mib: int,
) -> None:
    """``--kv-unified`` を必要に応じて ``cmd`` に追加する

    ``cache_ram_mib > 0 AND slots > 1`` の場合に自動付与する。``kv_unified``
    を明示指定した場合は auto 判定を上書きする。
    """
    cache_ram_mib = _resolve_cache_ram_mib(section_cfg, default=default_mib)
    if _resolve_kv_unified(section_cfg, slots=slots, cache_ram_mib=cache_ram_mib):
        cmd += ["--kv-unified"]


# ── reasoning OS-fence 解決 ─────────────────────────────


_VALID_REASONING_MODES: frozenset[str] = frozenset({"on", "off", "auto"})


def _append_reasoning_args(cmd: list[str], section_cfg: dict) -> None:
    """``-rea`` / ``--reasoning-budget`` / ``--reasoning-budget-message``
    を ``cmd`` に追加する

    ``"on"`` / ``"off"`` / ``"auto"`` のいずれか (大文字小文字非依存)、
    ``None`` / 空文字列なら付与しない (上流 default に追従)。

    ``reasoning_budget_default`` の意味は上流に従う::

        -1: 制限なし (--reasoning-budget を付与しない、上流既定挙動)
         0: 即終了 (思考トリガー直後に thinking 終了 grammar)
        N>0: N トークンで delayed-launch grammar が発動

    ``reasoning_budget_message`` は budget 超過時に thinking 終了タグの
    直前へ挿入される文字列。空文字列なら付与しない。

    config schema (``AssistModelLocalConfig``) で値域は検証済みのため、
    ここでは型・空文字列のみチェックする。
    """
    reasoning = section_cfg.get("reasoning")
    if isinstance(reasoning, str):
        normalized = reasoning.strip().lower()
        if normalized in _VALID_REASONING_MODES:
            cmd += ["-rea", normalized]

    budget = section_cfg.get("reasoning_budget_default", -1)
    try:
        budget_int = int(budget)
    except (TypeError, ValueError):
        budget_int = -1
    if budget_int >= 0:
        cmd += ["--reasoning-budget", str(budget_int)]

    message = section_cfg.get("reasoning_budget_message", "")
    if isinstance(message, str) and message:
        cmd += ["--reasoning-budget-message", message]


# ── self-speculative decoding 解決 ───────────────────────


_VALID_SPECULATIVE_MODES: frozenset[str] = frozenset(
    {
        "default",
        "ngram-mod",
        "ngram-cache",
        "ngram-simple",
        "ngram-map-k",
        "ngram-map-k4v",
        "draft-model",
    }
)


def _resolve_pro_edition() -> bool:
    """Pro エディションかどうかを返す

    判定優先順位は ``backend/free/cli/cli_mode.py::is_cli_pro_edition`` と
    同一だが、``scripts/`` 配下から ``backend.edition`` への top-level 依存
    を増やさないため、本関数内に最小実装を持つ。

    1. ``EVOREF_EDITION`` 環境変数 (``"free"`` / ``"pro"``、大文字小文字非依存)
    2. ``backend.pro`` パッケージが import 可能か (パッケージ同梱判定)

    どちらでも判定できない場合は Free 扱い。
    """
    edition_env = os.environ.get("EVOREF_EDITION", "").strip().lower()
    if edition_env == "free":
        return False
    if edition_env == "pro":
        return True
    try:
        import importlib.util

        return importlib.util.find_spec("backend.pro") is not None
    except (ModuleNotFoundError, ValueError, ImportError):
        return False


def _resolve_draft_model_path(
    spec_cfg: dict, project_root: Path,
) -> Path | None:
    """``draft_model_path`` を絶対パスに解決する

    未設定 / 空文字列の場合は ``None``。相対パスは ``project_root`` 起点。
    """
    raw = spec_cfg.get("draft_model_path")
    if not raw:
        return None
    path = Path(str(raw))
    if not path.is_absolute():
        path = project_root / path
    return path


def build_speculative_args(
    llama_cfg: dict,
    *,
    is_pro: bool,
    project_root: Path | None = None,
    warn: Callable[[str], None] | None = None,
) -> list[str]:
    """``llama.speculative`` セクションから ``--spec-*`` 系フラグを組み立てる。


    Pro 限定機能。``is_pro=False`` で ``enabled=true`` の場合は warning を発し
    フラグを付与しない。``enabled=false`` の場合は warning なしで空リストを返す。

    mode 別の挙動:

    - ``"default"``: ``--spec-default`` 単独 (上流プリセット)。
      ``draft_max`` / ``draft_min`` / ``draft_p_min`` / ``ngram_size_n`` /
      ``ngram_size_m`` を付加すれば上書き可能。
    - ``"ngram-mod"`` / ``"ngram-cache"`` / ``"ngram-simple"`` /
      ``"ngram-map-k"`` / ``"ngram-map-k4v"``: ``--spec-type <mode>`` +
      明示パラメータ。
    - ``"draft-model"``: ``-md <path>`` + ``-cd`` (任意) + ``-ngld`` (任意) +
      ``--draft-max`` / ``--draft-min``。

    Args:
        llama_cfg: ``cfg["llama"]`` セクション dict。
        is_pro: Pro エディションかどうか。:func:`_resolve_pro_edition` の結果。
        project_root: ``draft_model_path`` の相対パス解決起点。``None`` なら
            CWD。
        warn: warning 出力先 (``print(..., file=sys.stderr)`` 等)。テスト用に
            差し替え可能。``None`` なら何もしない。

    Returns:
        ``llama-server`` に追加すべき引数のリスト。``enabled=false`` ある
        いは Free 判定で警告のみの場合は ``[]``。
    """
    spec_cfg = llama_cfg.get("speculative") or {}
    if not spec_cfg.get("enabled", False):
        return []

    if not is_pro:
        if warn is not None:
            warn(
                "[launch] WARNING: llama.speculative.enabled=true but Pro "
                "edition is not active (EVOREF_EDITION != 'pro' and "
                "backend.pro not installed). Skipping --spec-* flags "
                "."
            )
        return []

    mode = str(spec_cfg.get("mode", "default")).strip().lower()
    if mode not in _VALID_SPECULATIVE_MODES:
        if warn is not None:
            warn(
                f"[launch] WARNING: invalid llama.speculative.mode={mode!r}; "
                "skipping --spec-* flags (expected one of "
                f"{sorted(_VALID_SPECULATIVE_MODES)})."
            )
        return []

    args: list[str] = []
    draft_max = spec_cfg.get("draft_max")
    draft_min = spec_cfg.get("draft_min")
    draft_p_min = spec_cfg.get("draft_p_min")
    ctx_size_draft = spec_cfg.get("ctx_size_draft", 0)
    ngram_n = spec_cfg.get("ngram_size_n")
    ngram_m = spec_cfg.get("ngram_size_m")

    if mode == "default":
        args += ["--spec-default"]
    elif mode == "draft-model":
        if project_root is None:
            project_root = Path.cwd()
        draft_path = _resolve_draft_model_path(spec_cfg, project_root)
        if draft_path is None:
            if warn is not None:
                warn(
                    "[launch] WARNING: llama.speculative.mode='draft-model' "
                    "requires llama.speculative.draft_model_path. Skipping "
                    "--spec-* flags."
                )
            return []
        args += ["-md", str(draft_path)]
        if int(ctx_size_draft) > 0:
            args += ["-cd", str(int(ctx_size_draft))]
        ngld = spec_cfg.get("gpu_layers_draft")
        if ngld is not None:
            ngld_str = str(ngld).strip()
            if ngld_str:
                args += ["-ngld", ngld_str]
    else:
        # ngram-* 系
        args += ["--spec-type", mode]
        if int(ctx_size_draft) > 0:
            args += ["-cd", str(int(ctx_size_draft))]

    # 共通フラグ (default / ngram-* / draft-model 全てで上流が受理)
    if draft_max is not None:
        args += ["--draft-max", str(int(draft_max))]
    if draft_min is not None:
        args += ["--draft-min", str(int(draft_min))]
    if draft_p_min is not None:
        args += ["--draft-p-min", str(float(draft_p_min))]

    # ngram 系のみ ngram_size_n / ngram_size_m を付与する。``default`` は
    # 上流プリセット (n=24 / min=48 / max=64) に従わせるため、明示指定が
    # ある場合のみ後段で上書きする。
    if mode in ("default", "ngram-mod", "ngram-cache", "ngram-simple",
                "ngram-map-k", "ngram-map-k4v"):
        if ngram_n is not None:
            args += ["--spec-ngram-size-n", str(int(ngram_n))]
        if ngram_m is not None:
            args += ["--spec-ngram-size-m", str(int(ngram_m))]

    return args


def build_llama_cmd(
    cfg: dict,
    project_root: Path | None = None,
    *,
    model_override: str | None = None,
) -> list[str]:
    """config.yaml の llama セクションから起動コマンドを生成"""
    if "llama" not in cfg:
        raise ValueError("config.yaml に 'llama' セクションがありません")
    lc = cfg["llama"]
    sp = cfg.get("model_paths", {})
    lp = cfg.get("local_paths", {})

    if project_root is None:
        project_root = Path.cwd()

    # ベースモデルパス解決（model_override 指定時はそちらを優先）
    base_model = model_override or sp.get("base_model", "models/base_model.gguf")
    base_model_path = Path(base_model)
    if not base_model_path.is_absolute():
        base_model_path = project_root / base_model_path

    cmd = [
        "llama-server",
        "-m", str(base_model_path),
        "--port", str(lc.get("port", 8080)),
        "-c", str(lc.get("context_size", 4096)),
        "-ngl", str(_resolve_base_gpu_layers(cfg, project_root)),
        "-b", str(lc.get("batch_size", 512)),
    ]

    # LoRA アダプタ（存在する場合のみ）
    lora_path = lp.get("lora_adapter", "local/models/adapter.gguf")
    lora_full = Path(lora_path)
    if not lora_full.is_absolute():
        lora_full = project_root / lora_full
    if lora_full.exists():
        cmd += ["--lora", str(lora_full)]

    # オプション
    threads = lc.get("threads", 0)
    if threads > 0:
        cmd += ["-t", str(threads)]
    flash_attn = lc.get("flash_attn", True)
    if flash_attn is not False:
        fa_value = flash_attn if isinstance(flash_attn, str) else "on"
        cmd += ["-fa", fa_value]
    if lc.get("mlock", False):
        cmd += ["--mlock"]

    # KVキャッシュ最適化
    # slots は常に ``-np`` で明示する。未指定だと新しい llama-server が
    # n_parallel=auto (=4) を選び、slots=1 (単一スロット = 省メモリ) の意図が
    # 効かなくなるため (slots レバーが slots=1 で no-op 化していた)。
    slots = max(1, int(lc.get("slots", 1) or 1))
    cmd += ["-np", str(slots)]
    cache_type_k = lc.get("cache_type_k")
    if cache_type_k and cache_type_k != "f16":
        cmd += ["--cache-type-k", str(cache_type_k)]
    cache_type_v = lc.get("cache_type_v")
    if cache_type_v and cache_type_v != "f16":
        cmd += ["--cache-type-v", str(cache_type_v)]

    # idle slot offload。base は agentic ワークロード前提で
    # 既定 4096 MiB の RAM 退避バッファを確保する。slots>1 のときは
    # ``--cache-ram`` が動作するよう ``--kv-unified`` を自動付与する。
    _append_cache_ram_args(cmd, lc, default_mib=4096)
    _append_kv_unified_args(cmd, lc, slots=int(slots), default_mib=4096)

    # self-speculative decoding。Pro 限定機能。Free 判定では
    # ``enabled=true`` でも warning + フラグ未付与で素通りさせる。
    spec_args = build_speculative_args(
        lc,
        is_pro=_resolve_pro_edition(),
        project_root=project_root,
        warn=lambda msg: print(msg, file=sys.stderr),
    )
    cmd += spec_args

    # モデル arch 別の自動フラグ (--jinja / --reasoning-format / MoE)。
    # extra_args の直前に挿入し、ユーザー指定 (extra_args) を最終 override とする。
    cmd += resolve_auto_model_flags(
        cfg,
        base_model_path,
        fixed_flags={
            "-m", "--port", "-c", "-ngl", "-b", "-t", "-fa", "--mlock",
            "-np", "--cache-type-k", "--cache-type-v", "--cache-ram",
            "--kv-unified", "--lora",
        },
        project_root=project_root,
        warn=lambda m: print(m, file=sys.stderr),
    )

    # 追加オプション
    cmd += lc.get("extra_args", [])

    return cmd


def build_embed_cmd(cfg: dict, project_root: Path | None = None) -> list[str] | None:
    """config.yaml の embedding セクションから埋め込み用 llama-server コマンドを生成

    embedding.backend は llama-cpp のみサポート。未知の値は None を返す。
    ``-ngl`` 既定値は 0
    """
    emb_cfg = cfg.get("embedding", {})
    if emb_cfg.get("backend", "llama-cpp") != "llama-cpp":
        return None

    if project_root is None:
        project_root = Path.cwd()

    sp = cfg.get("model_paths", {})
    lp = cfg.get("local_paths", {})

    # エンベッドモデルパス（model_paths.embed_model or デフォルト）
    embed_model = sp.get("embed_model", "models/qwen3-embedding.gguf")
    embed_model_path = Path(embed_model)
    if not embed_model_path.is_absolute():
        embed_model_path = project_root / embed_model_path

    port = emb_cfg.get("llama_port", 8082)

    cmd = [
        "llama-server",
        "-m", str(embed_model_path),
        "--port", str(port),
        "--embedding",
        "-ngl", str(_resolve_embed_gpu_layers(cfg)),
    ]

    # 文脈長。モデル既定 n_ctx (Qwen3-Embedding=32768) は過剰なので
    # embedding.context_size (既定 8192 = max_length) に縮小して KV を節約する。
    # max_length を下回ると長い入力で 500 になるため context_size >= max_length を維持する。
    context_size = emb_cfg.get("context_size", 8192)
    cmd += ["-c", str(context_size)]

    # エンベッド LoRA アダプタ（存在する場合のみ）
    embed_lora = lp.get("embed_lora_adapter", "local/models/embed_adapter.gguf")
    embed_lora_path = Path(embed_lora)
    if not embed_lora_path.is_absolute():
        embed_lora_path = project_root / embed_lora_path
    if embed_lora_path.exists():
        cmd += ["--lora", str(embed_lora_path)]

    # 物理バッチサイズ。長い STM ノート (>512 tok) の埋め込みリクエストが
    # llama-server デフォルト batch=512 のままだと 500 エラーで落ちるため、
    # config.yaml の embedding.batch_size / ubatch_size を明示的に渡す。
    batch_size = emb_cfg.get("batch_size")
    if batch_size is not None:
        cmd += ["-b", str(batch_size)]
    ubatch_size = emb_cfg.get("ubatch_size")
    if ubatch_size is not None:
        cmd += ["-ub", str(ubatch_size)]

    # idle slot offload は埋め込みでは無意味 (chat slots 不使用) なので
    # 既定 0 で明示 disable する。``cache_ram_mib`` を 0 以外に
    # 指定された場合のみ付与値を変更する。
    cache_ram_mib = _resolve_cache_ram_mib(emb_cfg, default=0)
    cmd += ["--cache-ram", str(cache_ram_mib)]

    return cmd


def build_assist_cmd(cfg: dict, project_root: Path | None = None) -> list[str] | None:
    """config.yaml の assist_model セクションからアシストモデル用 llama-server コマンドを生成

    assist_model.local セクションがあり、model_paths.assist_model が存在する場合のみコマンドを返す。
    """
    assist_cfg = cfg.get("assist_model", {})
    local_cfg = assist_cfg.get("local", {})
    if not local_cfg:
        return None

    if project_root is None:
        project_root = Path.cwd()

    sp = cfg.get("model_paths", {})

    # アシストモデルパス
    assist_model = sp.get("assist_model", "")
    if not assist_model:
        return None
    assist_model_path = Path(assist_model)
    if not assist_model_path.is_absolute():
        assist_model_path = project_root / assist_model_path

    port = local_cfg.get("port", 8081)

    # GPU layers: assist 側で明示指定があればそちらを優先し、なければ
    # llama セクションの値にフォールバックする（従来挙動）。
    # ``"auto"`` 指定時は _resolve_auto_gpu_layers がキャッシュ済み計算結果を返す。
    base_llama_cfg = cfg.get("llama", {})
    gpu_layers = _resolve_assist_gpu_layers(cfg, project_root)

    cmd = [
        "llama-server",
        "-m", str(assist_model_path),
        "--port", str(port),
        "-c", str(local_cfg.get("context_size", 8192)),
        "-ngl", str(gpu_layers),
    ]

    # 物理バッチサイズ（指定があれば）
    batch_size = local_cfg.get("batch_size")
    if batch_size is not None:
        cmd += ["-b", str(batch_size)]
    ubatch_size = local_cfg.get("ubatch_size")
    if ubatch_size is not None:
        cmd += ["-ub", str(ubatch_size)]

    # スレッド数（0 は指定しない = llama-server デフォルト）
    threads = local_cfg.get("threads", 0)
    if threads and threads > 0:
        cmd += ["-t", str(threads)]

    # Flash Attention: assist 側で明示指定があればそちらを優先し、
    # なければ llama セクションの値にフォールバックする（従来挙動）。
    flash_attn = local_cfg.get("flash_attn")
    if flash_attn is None:
        flash_attn = base_llama_cfg.get("flash_attn", True)
    if flash_attn is not False:
        fa_value = flash_attn if isinstance(flash_attn, str) else "on"
        cmd += ["-fa", fa_value]

    # mlock
    if local_cfg.get("mlock", False):
        cmd += ["--mlock"]

    # 並列スロット数。常に ``-np`` で明示する (build_llama_cmd と同じ理由:
    # 未指定だと n_parallel=auto で単一スロット意図が崩れる)。
    slots = max(1, int(local_cfg.get("slots", 1) or 1))
    cmd += ["-np", str(slots)]

    # KV キャッシュ量子化（指定があれば）
    cache_type_k = local_cfg.get("cache_type_k")
    if cache_type_k and cache_type_k != "f16":
        cmd += ["--cache-type-k", str(cache_type_k)]
    cache_type_v = local_cfg.get("cache_type_v")
    if cache_type_v and cache_type_v != "f16":
        cmd += ["--cache-type-v", str(cache_type_v)]

    # 共通 prefix 自動再利用（0 は無効）
    cache_reuse = local_cfg.get("cache_reuse", 0)
    if cache_reuse and int(cache_reuse) > 0:
        cmd += ["--cache-reuse", str(int(cache_reuse))]

    # idle slot offload。assist 既定は 2048 MiB
    # アシストモデルは purpose 別セマフォで多重化されるため slots>1 で
    # 運用するケースもあり、その場合 ``--kv-unified`` を自動付与する。
    _append_cache_ram_args(cmd, local_cfg, default_mib=2048)
    _append_kv_unified_args(cmd, local_cfg, slots=int(slots), default_mib=2048)

    # reasoning OS-fence。thinking モデルをアシスト用途で
    # 使う場合の token 浪費を起動時点で抑止する。リクエスト側の
    # ``chat_template_kwargs.enable_thinking=false`` (assist_client.py) と
    # の二重防御。
    _append_reasoning_args(cmd, local_cfg)

    # debug ルート: サーバ側 chat 解析の完全スキップ
    # jinja template が指定されていても reasoning / tool calls を
    # ``message.content`` 一本化する。``<think>...</think>`` の除去は
    # Python 側 ``_ReasoningFilter`` に集約できるため
    # 障害切り分け / 学習サイクルで Python 側に処理を寄せたい用途で
    # 有効化する。通常運用では jinja の利点を失うため false を推奨。
    if local_cfg.get("skip_chat_parsing", False):
        cmd += ["--skip-chat-parsing"]

    # モデル arch 別の自動フラグ。base と同様 extra_args の直前に挿入し、
    # llama.auto_model_flags で base と一括制御する。
    cmd += resolve_auto_model_flags(
        cfg,
        assist_model_path,
        fixed_flags={
            "-m", "--port", "-c", "-ngl", "-b", "-ub", "-t", "-fa", "--mlock",
            "-np", "--cache-type-k", "--cache-type-v", "--cache-reuse",
            "--cache-ram", "--kv-unified", "-rea", "--reasoning-budget",
            "--reasoning-budget-message", "--skip-chat-parsing",
        },
        project_root=project_root,
        warn=lambda m: print(m, file=sys.stderr),
    )

    # 追加オプション
    extra_args = local_cfg.get("extra_args", [])
    if extra_args:
        cmd += list(extra_args)

    return cmd


def build_reranker_cmd(cfg: dict, project_root: Path | None = None) -> list[str] | None:
    """config.yaml の reranker セクションからリランカー用 llama-server コマンドを生成

    reranker が有効で backend が llama-cpp の場合のみコマンドを返す。
    ``-ngl`` 既定値は 0
    """
    reranker_cfg = cfg.get("reranker", {})
    if not reranker_cfg.get("enabled", False):
        return None
    if reranker_cfg.get("backend", "llama-cpp") != "llama-cpp":
        return None

    if project_root is None:
        project_root = Path.cwd()

    sp = cfg.get("model_paths", {})

    # リランカーモデルパス
    reranker_model = sp.get("reranker_model", "models/reranker.gguf")
    reranker_model_path = Path(reranker_model)
    if not reranker_model_path.is_absolute():
        reranker_model_path = project_root / reranker_model_path

    port = reranker_cfg.get("port", 8083)

    cmd = [
        "llama-server",
        "-m", str(reranker_model_path),
        "--port", str(port),
        "--reranking",
        "-ngl", str(_resolve_reranker_gpu_layers(cfg)),
    ]

    # idle slot offload は短時間推論を大量に行うリランカーでは恩恵が薄く、
    # 長時間運用時のキャッシュ破損リスクを排除するため既定 0 で明示 disable
    # する
    cache_ram_mib = _resolve_cache_ram_mib(reranker_cfg, default=0)
    cmd += ["--cache-ram", str(cache_ram_mib)]

    # 物理バッチサイズ (config 駆動)。指定が無ければ llama-server のデフォルトに任せる。
    # rag.chunk_size を超える入力で 500 エラーになるのを防ぐため、reranker.batch_size /
    # reranker.ubatch_size で明示的にチューニングする。
    batch_size = reranker_cfg.get("batch_size")
    if batch_size is not None:
        cmd += ["-b", str(batch_size)]
    ubatch_size = reranker_cfg.get("ubatch_size")
    if ubatch_size is not None:
        cmd += ["-ub", str(ubatch_size)]

    # 追加オプション（ユーザーによるチューニング用）
    extra_args = reranker_cfg.get("extra_args", [])
    if extra_args:
        cmd += list(extra_args)

    return cmd


# ── VRAM 予算検査 ─────────────────────────────────


def _resolve_model_path(
    cfg: dict, key: str, default: str, project_root: Path,
) -> Path:
    """model_paths.<key> を絶対パスに解決する"""
    sp = cfg.get("model_paths", {}) or {}
    value = sp.get(key, default) or default
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path


def _file_size_mb(path: Path) -> int | None:
    """ファイルサイズ (MB, 四捨五入)。存在しない場合は None。"""
    try:
        size_bytes = path.stat().st_size
    except OSError:
        return None
    return int(round(size_bytes / (1024 * 1024)))


# ── GGUF ヘッダ parser ──────────────────────────────────
# rationale: ``gguf`` Python パッケージは未導入のため、``<arch>.block_count``
# 1 つだけ抽出するための最小 parser を struct で実装する。完全な GGUF reader
# は不要で、key の string match だけが本実装の責務。
# Spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md (v2/v3)

_GGUF_MAGIC = b"GGUF"

# GGUF metadata value type enum (上流 ggml/gguf.py より)
_GGUF_TYPE_UINT8 = 0
_GGUF_TYPE_INT8 = 1
_GGUF_TYPE_UINT16 = 2
_GGUF_TYPE_INT16 = 3
_GGUF_TYPE_UINT32 = 4
_GGUF_TYPE_INT32 = 5
_GGUF_TYPE_FLOAT32 = 6
_GGUF_TYPE_BOOL = 7
_GGUF_TYPE_STRING = 8
_GGUF_TYPE_ARRAY = 9
_GGUF_TYPE_UINT64 = 10
_GGUF_TYPE_INT64 = 11
_GGUF_TYPE_FLOAT64 = 12


def _gguf_read_scalar(f, vtype: int) -> object:
    """GGUF metadata の単一スカラー値を読み出す (ARRAY は外側で扱う)。"""
    if vtype == _GGUF_TYPE_UINT8:
        return struct.unpack("<B", f.read(1))[0]
    if vtype == _GGUF_TYPE_INT8:
        return struct.unpack("<b", f.read(1))[0]
    if vtype == _GGUF_TYPE_UINT16:
        return struct.unpack("<H", f.read(2))[0]
    if vtype == _GGUF_TYPE_INT16:
        return struct.unpack("<h", f.read(2))[0]
    if vtype == _GGUF_TYPE_UINT32:
        return struct.unpack("<I", f.read(4))[0]
    if vtype == _GGUF_TYPE_INT32:
        return struct.unpack("<i", f.read(4))[0]
    if vtype == _GGUF_TYPE_FLOAT32:
        return struct.unpack("<f", f.read(4))[0]
    if vtype == _GGUF_TYPE_BOOL:
        return struct.unpack("<B", f.read(1))[0] != 0
    if vtype == _GGUF_TYPE_UINT64:
        return struct.unpack("<Q", f.read(8))[0]
    if vtype == _GGUF_TYPE_INT64:
        return struct.unpack("<q", f.read(8))[0]
    if vtype == _GGUF_TYPE_FLOAT64:
        return struct.unpack("<d", f.read(8))[0]
    if vtype == _GGUF_TYPE_STRING:
        (slen,) = struct.unpack("<Q", f.read(8))
        return f.read(slen).decode("utf-8", errors="replace")
    raise ValueError(f"unsupported GGUF scalar type: {vtype}")


def _gguf_skip_value(f, vtype: int) -> None:
    """GGUF metadata 値を読み飛ばす (block_count 抽出に不要な値の高速 skip)。"""
    if vtype == _GGUF_TYPE_ARRAY:
        (etype,) = struct.unpack("<I", f.read(4))
        (count,) = struct.unpack("<Q", f.read(8))
        for _ in range(count):
            _gguf_skip_value(f, etype)
        return
    # スカラーは読み捨て (decode コストを避けるため STRING も length+seek で skip)
    if vtype == _GGUF_TYPE_STRING:
        (slen,) = struct.unpack("<Q", f.read(8))
        f.seek(slen, 1)
        return
    # 固定長スカラー
    sizes = {
        _GGUF_TYPE_UINT8: 1, _GGUF_TYPE_INT8: 1, _GGUF_TYPE_BOOL: 1,
        _GGUF_TYPE_UINT16: 2, _GGUF_TYPE_INT16: 2,
        _GGUF_TYPE_UINT32: 4, _GGUF_TYPE_INT32: 4, _GGUF_TYPE_FLOAT32: 4,
        _GGUF_TYPE_UINT64: 8, _GGUF_TYPE_INT64: 8, _GGUF_TYPE_FLOAT64: 8,
    }
    if vtype in sizes:
        f.seek(sizes[vtype], 1)
        return
    raise ValueError(f"unsupported GGUF value type: {vtype}")


def _read_gguf_layer_count(gguf_path: Path) -> int | None:
    """GGUF ヘッダから ``<arch>.block_count`` を読み出す。

    パース失敗・キー不在・I/O エラーはすべて ``None`` 返却で呼び出し側に
    フォールバック判断を委ねる。呼び出し側 (`_resolve_auto_gpu_layers`) は
    None 受領で auto-tune を諦め既定 999 を維持する設計。
    """
    try:
        with gguf_path.open("rb") as f:
            magic = f.read(4)
            if magic != _GGUF_MAGIC:
                return None
            (version,) = struct.unpack("<I", f.read(4))
            if version < 2:
                # v1 は key prefix が異なるため対象外 (実用上殆ど存在しない)
                return None
            n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
            _ = n_tensors  # 未使用
            for _ in range(n_kv):
                (key_len,) = struct.unpack("<Q", f.read(8))
                key = f.read(key_len).decode("utf-8", errors="replace")
                (vtype,) = struct.unpack("<I", f.read(4))
                if key.endswith(".block_count"):
                    # ``<arch>.block_count`` は uint32/uint64 のいずれか
                    value = _gguf_read_scalar(f, vtype)
                    try:
                        return int(value)
                    except (TypeError, ValueError):
                        return None
                _gguf_skip_value(f, vtype)
    except (OSError, struct.error, UnicodeDecodeError, ValueError):
        return None
    return None


def read_gguf_metadata(gguf_path: Path) -> dict:
    """GGUF ヘッダから起動フラグ決定に必要なメタデータを 1 パスで読む。

    Returns (失敗時・キー不在時も同じ shape):
        ``{"architecture": str | None, "context_length": int | None,
           "expert_count": int, "has_chat_template": bool,
           "block_count" / "head_count_kv" / "head_count" / "key_length" /
           "value_length" / "embedding_length": int | None}``

    後半 6 キーは KV キャッシュ VRAM 推定 (``estimate_kv_cache_mb``) 用。
    ``expert_count`` は対応キーが無い dense モデルで 0。パース失敗・I/O
    エラーは安全な既定値を返し、呼び出し側 (default プロファイル /
    フラグ無付与) に委ねる。``_read_gguf_layer_count`` と同じ struct ベース
    最小 parser を流用する。backend (sampling 注入経路) からも import される
    ため public 名とする。
    """
    result: dict = {
        "architecture": None,
        "context_length": None,
        "expert_count": 0,
        "has_chat_template": False,
        # KV キャッシュ VRAM 推定用 (estimate_kv_cache_mb)
        "block_count": None,
        "head_count_kv": None,
        "head_count": None,
        "key_length": None,
        "value_length": None,
        "embedding_length": None,
    }
    try:
        with gguf_path.open("rb") as f:
            magic = f.read(4)
            if magic != _GGUF_MAGIC:
                return result
            (version,) = struct.unpack("<I", f.read(4))
            if version < 2:
                return result
            n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
            _ = n_tensors  # 未使用
            for _ in range(n_kv):
                (key_len,) = struct.unpack("<Q", f.read(8))
                key = f.read(key_len).decode("utf-8", errors="replace")
                (vtype,) = struct.unpack("<I", f.read(4))
                if key == "general.architecture":
                    val = _gguf_read_scalar(f, vtype)
                    result["architecture"] = str(val) if val is not None else None
                elif key.endswith(".context_length"):
                    try:
                        result["context_length"] = int(_gguf_read_scalar(f, vtype))
                    except (TypeError, ValueError):
                        pass
                elif key.endswith(".expert_count"):
                    try:
                        result["expert_count"] = int(_gguf_read_scalar(f, vtype))
                    except (TypeError, ValueError):
                        pass
                elif key.endswith(".block_count"):
                    try:
                        result["block_count"] = int(_gguf_read_scalar(f, vtype))
                    except (TypeError, ValueError):
                        pass
                elif key.endswith(".attention.head_count_kv"):
                    try:
                        result["head_count_kv"] = int(_gguf_read_scalar(f, vtype))
                    except (TypeError, ValueError):
                        pass
                elif key.endswith(".attention.head_count"):
                    try:
                        result["head_count"] = int(_gguf_read_scalar(f, vtype))
                    except (TypeError, ValueError):
                        pass
                elif key.endswith(".attention.key_length"):
                    try:
                        result["key_length"] = int(_gguf_read_scalar(f, vtype))
                    except (TypeError, ValueError):
                        pass
                elif key.endswith(".attention.value_length"):
                    try:
                        result["value_length"] = int(_gguf_read_scalar(f, vtype))
                    except (TypeError, ValueError):
                        pass
                elif key.endswith(".embedding_length"):
                    try:
                        result["embedding_length"] = int(_gguf_read_scalar(f, vtype))
                    except (TypeError, ValueError):
                        pass
                elif key == "tokenizer.chat_template":
                    result["has_chat_template"] = True
                    _gguf_skip_value(f, vtype)
                else:
                    _gguf_skip_value(f, vtype)
    except (OSError, struct.error, UnicodeDecodeError, ValueError):
        return result
    return result


# モデルプロファイル: arch 単位の起動フラグ / sampling 既定。
# 同梱ベース (tracked, models/profiles/) + local override の 2 段階解決。
# default フォールバックは持たない (プロファイルの無い arch はフラグ無付与)。
# ``models/`` 自体は .gitignore 対象だが models/profiles/ は再包含例外で tracked。
_MODEL_PROFILE_BASE_DIR = "models/profiles"
_MODEL_PROFILE_OVERRIDE_DIR = "local/profiles"


def load_model_profile(arch: str | None, project_root: Path) -> dict:
    """arch 名からモデルプロファイル dict を解決する (2 段階)。

    解決順: ``local/profiles/<arch>.yaml`` (user override) →
    同梱 ``models/profiles/<arch>.yaml`` (base)。いずれも無い / ``arch`` が
    None の場合は ``{}`` (フォールバックなし)。override は wholesale 置換
    (deep merge しない)。
    """
    if not arch:
        return {}

    override_dir = project_root / _MODEL_PROFILE_OVERRIDE_DIR
    base_dir = project_root / _MODEL_PROFILE_BASE_DIR
    for base in (override_dir, base_dir):
        path = base / f"{arch}.yaml"
        if path.exists():
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError):
                continue
            if isinstance(data, dict):
                return data
    return {}


def _dedupe_flags(candidate: list[str], fixed_flags: set[str]) -> list[str]:
    """既に固定フラグで出力済みのフラグを candidate から除外する。

    ``--xxx value`` / ``--flag`` (値なし) のペア単位で扱い、``fixed_flags`` に
    含まれるフラグは引数ごとスキップする。
    """
    out: list[str] = []
    i = 0
    n = len(candidate)
    while i < n:
        tok = candidate[i]
        if tok.startswith("-"):
            has_value = (i + 1 < n) and not candidate[i + 1].startswith("-")
            if tok in fixed_flags:
                i += 2 if has_value else 1
                continue
            out.append(tok)
            if has_value:
                out.append(candidate[i + 1])
                i += 2
            else:
                i += 1
        else:
            out.append(tok)
            i += 1
    return out


def resolve_auto_model_flags(
    cfg: dict,
    model_path: Path,
    *,
    fixed_flags: set[str],
    project_root: Path,
    warn: Callable[[str], None] | None = None,
) -> list[str]:
    """GGUF メタデータ + arch プロファイルから llama-server 起動フラグを決定する。

    - ``llama.auto_model_flags`` が false なら何も付与しない (現行挙動)。
    - profile.launch_flags をベースに、MoE (GGUF が expert を報告 +
      profile.moe.enabled + n_cpu_moe が明示 int) なら ``--n-cpu-moe N`` を付与。
    - ``fixed_flags`` に既出のフラグは除外して二重付与を避ける。
    - GGUF 読取失敗 / arch 不明時は default プロファイルにフォールバック。
    """
    lc = cfg.get("llama", {}) or {}
    if not lc.get("auto_model_flags", True):
        return []

    meta = read_gguf_metadata(model_path)
    profile = load_model_profile(meta.get("architecture"), project_root)
    if not profile:
        return []

    candidate: list[str] = list(profile.get("launch_flags", []) or [])

    # MoE: GGUF が expert を報告し、プロファイルが MoE 有効かつ n_cpu_moe を
    # 明示している場合のみ --n-cpu-moe を付与 (auto=null では推測しない)。
    moe_cfg = profile.get("moe", {}) or {}
    expert_count = int(meta.get("expert_count", 0) or 0)
    n_cpu_moe = moe_cfg.get("n_cpu_moe")
    if moe_cfg.get("enabled", False) and expert_count > 0 and n_cpu_moe is not None:
        candidate += ["--n-cpu-moe", str(int(n_cpu_moe))]

    # context 乖離は警告のみ (-c は config 優先で自動上書きしない)。
    ctx_train = meta.get("context_length")
    cfg_ctx = lc.get("context_size")
    if warn and ctx_train and cfg_ctx and int(cfg_ctx) > int(ctx_train):
        warn(
            f"[launch] WARNING: llama.context_size={cfg_ctx} exceeds model "
            f"n_ctx_train={ctx_train}; consider lowering context_size"
        )

    return _dedupe_flags(candidate, fixed_flags)


# KV キャッシュ量子化タイプ別の 1 要素あたりバイト数 (概算)。
# llama.cpp のブロックサイズ由来 (q8_0=34B/32, q5_1=24B/32, q4_1=20B/32 ...)。
_KV_BYTES_PER_ELEM: dict[str, float] = {
    "f32": 4.0, "f16": 2.0, "bf16": 2.0,
    "q8_0": 1.0625, "q5_1": 0.75, "q5_0": 0.6875,
    "q4_1": 0.625, "q4_0": 0.5625,
}


def _kv_bytes_per_elem(cache_type: str | None) -> float:
    """KV キャッシュ量子化タイプ → 1 要素バイト数。未指定/不明は f16 (2.0)。"""
    if not cache_type:
        return 2.0
    return _KV_BYTES_PER_ELEM.get(str(cache_type).lower(), 2.0)


def estimate_kv_cache_mb(
    meta: dict,
    n_ctx: int,
    cache_type_k: str | None,
    cache_type_v: str | None,
) -> int | None:
    """GGUF メタデータと文脈長から KV キャッシュ VRAM 量 (MiB) を概算する。

    ``KV(bytes) = n_ctx × block_count × head_count_kv
                  × (key_length × bpe(K) + value_length × bpe(V))``

    必要メタデータ (block_count / head_count_kv / key_length 等) が欠ける、
    または ``n_ctx<=0`` の場合は ``None`` を返し、呼び出し側で KV 加算を
    スキップさせる (= 従来のファイルサイズのみ推定にフォールバック)。

    ``--kv-unified`` (slots=1 含む) を前提に slots 倍は掛けない (unified KV は
    n_ctx 総量で頭打ちになるため)。host RAM 退避 (``--cache-ram``) は VRAM 側
    では差し引かない (概算であり安全側に多めに見積もる)。
    """
    if not n_ctx or n_ctx <= 0:
        return None
    n_layers = meta.get("block_count")
    n_head_kv = meta.get("head_count_kv")
    head_dim_k = meta.get("key_length")
    head_dim_v = meta.get("value_length") or head_dim_k
    # key_length 欠落時は embedding_length / head_count で head_dim を代替
    if not head_dim_k:
        n_embd = meta.get("embedding_length")
        n_head = meta.get("head_count")
        if n_embd and n_head:
            head_dim_k = n_embd // n_head
            head_dim_v = head_dim_k
    if not (n_layers and n_head_kv and head_dim_k and head_dim_v):
        return None
    bytes_k = n_ctx * n_layers * n_head_kv * head_dim_k * _kv_bytes_per_elem(cache_type_k)
    bytes_v = n_ctx * n_layers * n_head_kv * head_dim_v * _kv_bytes_per_elem(cache_type_v)
    return int(round((bytes_k + bytes_v) / (1024 * 1024)))


_gguf_meta_cache: dict[tuple[str, int, int], dict] = {}


def _read_gguf_metadata_cached(path: Path) -> dict:
    """``read_gguf_metadata`` の (path, size, mtime) キャッシュ付きラッパ。

    VRAM モニタは 10 秒間隔でポーリングするため、同一モデルファイルの
    ヘッダを毎回パースしないようプロセス内でキャッシュする。ファイルが
    差し替われば (size/mtime 変化) キーが変わり自動で再読込される。
    """
    try:
        st = path.stat()
        key = (str(path), st.st_size, st.st_mtime_ns)
    except OSError:
        return read_gguf_metadata(path)
    cached = _gguf_meta_cache.get(key)
    if cached is None:
        cached = read_gguf_metadata(path)
        _gguf_meta_cache[key] = cached
    return cached


def _estimate_via_gguf_size(
    cfg: dict, project_root: Path,
) -> dict[str, dict]:
    """Tier 2: GGUF ファイルサイズベースの粗い VRAM 見積り

    - ``gpu_layers == 0``: 0 MB (CPU 完全配置)
    - ``gpu_layers > 0``:  ファイルサイズ全量 (全層を GPU へオフロードと仮定)

    各 entry には ``estimated_via="gguf-size"`` が付き、Tier 1 結果と区別可能。
    ``context_mb`` / ``compute_mb`` / ``device`` は Tier 2 では不明のため None。
    """
    result: dict[str, dict] = {}

    # ベース
    base_path = _resolve_model_path(cfg, "base_model", "", project_root)
    base_ngl = _resolve_base_gpu_layers(cfg, project_root)
    base_size = _file_size_mb(base_path)
    base_vram = (base_size or 0) if base_ngl > 0 else 0

    # Pro かつ ``llama.speculative.mode == "draft-model"`` の
    # ときは draft GGUF のサイズも base の VRAM 推定に加算する。
    # ngram 系 (default / ngram-*) は n-gram テーブルのみで軽量のため
    # 加算対象外。Free / disabled の場合は加算しない。
    draft_model_mb: int | None = None
    draft_model_path_str: str | None = None
    spec_cfg = (cfg.get("llama") or {}).get("speculative") or {}
    if (
        spec_cfg.get("enabled", False)
        and str(spec_cfg.get("mode", "default")).strip().lower() == "draft-model"
        and _resolve_pro_edition()
    ):
        draft_path = _resolve_draft_model_path(spec_cfg, project_root)
        if draft_path is not None:
            draft_model_path_str = str(draft_path)
            draft_size_mb = _file_size_mb(draft_path)
            if draft_size_mb is not None:
                draft_model_mb = draft_size_mb
                # gpu_layers_draft が 0 のときのみ CPU 配置とみなす。
                # null=auto / "all" / 正数はすべて GPU 配置扱い。
                ngld_raw = spec_cfg.get("gpu_layers_draft")
                if ngld_raw is None or (
                    isinstance(ngld_raw, str) and ngld_raw.strip().lower() == "all"
                ) or int(ngld_raw) > 0:
                    base_vram += draft_size_mb

    # KV キャッシュ VRAM を加算 (GPU 配置時のみ)。GGUF メタデータが
    # 読めない場合は None となり加算しない (従来のサイズのみ推定に縮退)。
    base_kv_mb: int | None = None
    if base_ngl > 0 and base_size is not None:
        lc = cfg.get("llama", {}) or {}
        base_kv_mb = estimate_kv_cache_mb(
            _read_gguf_metadata_cached(base_path),
            int(lc.get("context_size", 4096) or 4096),
            lc.get("cache_type_k"),
            lc.get("cache_type_v"),
        )
        if base_kv_mb:
            base_vram += base_kv_mb

    result["base"] = {
        "model_mb": base_size,
        "gpu_layers": base_ngl,
        "vram_mb": base_vram,
        "present": base_size is not None,
        "path": str(base_path),
        "context_mb": base_kv_mb,
        "compute_mb": None,
        "device": None,
        "estimated_via": "gguf-size",
        "draft_model_mb": draft_model_mb,
        "draft_model_path": draft_model_path_str,
    }

    # アシスト
    assist_cfg = cfg.get("assist_model", {}) or {}
    local_cfg = assist_cfg.get("local") or {}
    has_assist_local = bool(local_cfg) and bool(
        (cfg.get("model_paths") or {}).get("assist_model")
    )
    if has_assist_local:
        assist_path = _resolve_model_path(cfg, "assist_model", "", project_root)
        assist_ngl = _resolve_assist_gpu_layers(cfg, project_root)
        assist_size = _file_size_mb(assist_path)
        assist_vram = (assist_size or 0) if assist_ngl > 0 else 0
        # KV キャッシュ VRAM を加算 (assist は cache_type 未指定時 f16)
        assist_kv_mb: int | None = None
        if assist_ngl > 0 and assist_size is not None:
            assist_kv_mb = estimate_kv_cache_mb(
                _read_gguf_metadata_cached(assist_path),
                int(local_cfg.get("context_size", 8192) or 8192),
                local_cfg.get("cache_type_k"),
                local_cfg.get("cache_type_v"),
            )
            if assist_kv_mb:
                assist_vram += assist_kv_mb
        result["assist"] = {
            "model_mb": assist_size,
            "gpu_layers": assist_ngl,
            "vram_mb": assist_vram,
            "present": assist_size is not None,
            "path": str(assist_path),
            "context_mb": assist_kv_mb,
            "compute_mb": None,
            "device": None,
            "estimated_via": "gguf-size",
        }
    else:
        result["assist"] = {
            "model_mb": None, "gpu_layers": 0, "vram_mb": 0,
            "present": False, "path": "",
            "context_mb": None, "compute_mb": None, "device": None,
            "estimated_via": "gguf-size",
        }

    # 埋め込み
    emb_cfg = cfg.get("embedding", {}) or {}
    if emb_cfg.get("backend", "llama-cpp") == "llama-cpp":
        embed_path = _resolve_model_path(cfg, "embed_model", "", project_root)
        embed_ngl = _resolve_embed_gpu_layers(cfg)
        embed_size = _file_size_mb(embed_path)
        result["embed"] = {
            "model_mb": embed_size,
            "gpu_layers": embed_ngl,
            "vram_mb": (embed_size or 0) if embed_ngl > 0 else 0,
            "present": embed_size is not None,
            "path": str(embed_path),
            "context_mb": None,
            "compute_mb": None,
            "device": None,
            "estimated_via": "gguf-size",
        }
    else:
        result["embed"] = {
            "model_mb": None, "gpu_layers": 0, "vram_mb": 0,
            "present": False, "path": "",
            "context_mb": None, "compute_mb": None, "device": None,
            "estimated_via": "gguf-size",
        }

    # リランカー
    reranker_cfg = cfg.get("reranker", {}) or {}
    reranker_enabled = bool(reranker_cfg.get("enabled", False))
    reranker_backend_ok = reranker_cfg.get("backend", "llama-cpp") == "llama-cpp"
    if reranker_enabled and reranker_backend_ok:
        rr_path = _resolve_model_path(cfg, "reranker_model", "", project_root)
        rr_ngl = _resolve_reranker_gpu_layers(cfg)
        rr_size = _file_size_mb(rr_path)
        result["reranker"] = {
            "model_mb": rr_size,
            "gpu_layers": rr_ngl,
            "vram_mb": (rr_size or 0) if rr_ngl > 0 else 0,
            "present": rr_size is not None,
            "path": str(rr_path),
            "context_mb": None,
            "compute_mb": None,
            "device": None,
            "estimated_via": "gguf-size",
        }
    else:
        result["reranker"] = {
            "model_mb": None, "gpu_layers": 0, "vram_mb": 0,
            "present": False, "path": "",
            "context_mb": None, "compute_mb": None, "device": None,
            "estimated_via": "gguf-size",
        }

    return result


# ── llama-fit-params Tier 1 推定 ─────────────────────


# 上流 ``llama-fit-params --fit-print on`` の出力 (1 行 1 device)::
#
#     0.00.196.882 I main: printing estimated memory in MiB to stdout (device, model, context, compute) ...
#     MTL0 7401 814 517
#     host 1280 0 154
#     CUDA0 4200 480 120
#
# device label は英数字 + 任意のサフィックス (``MTL0`` / ``CUDA0`` /
# ``Vulkan0`` / ``ROCm0`` / ``host``)。MiB 値は非負整数 3 つ。
_FIT_PARAMS_LINE_RE = re.compile(
    r"^([A-Za-z][A-Za-z0-9_]*)\s+(\d+)\s+(\d+)\s+(\d+)\s*$"
)


def _parse_fit_params_output(text: str) -> dict | None:
    """``llama-fit-params --fit-print on`` の stdout から GPU 集計値を抽出する。

    戻り値 (1 つ以上 GPU device 行が見つかった場合)::

        {"device": "CUDA0", "model_mb": 4200,
         "context_mb": 480, "compute_mb": 120}

    複数 GPU device がある場合は MiB 値を合算し、device 文字列は ``+`` 連結。
    GPU device 行が 1 つも無い (CPU only / 解析不能) 場合は ``None``。
    ``host`` 行 (CPU 側のシステム RAM 使用量) は集計対象外。
    """
    if not text:
        return None
    gpu_entries: list[dict] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _FIT_PARAMS_LINE_RE.match(line)
        if not match:
            continue
        label, model_s, ctx_s, comp_s = match.groups()
        if label.lower() == "host":
            continue
        gpu_entries.append({
            "device": label,
            "model_mb": int(model_s),
            "context_mb": int(ctx_s),
            "compute_mb": int(comp_s),
        })
    if not gpu_entries:
        return None
    if len(gpu_entries) == 1:
        return gpu_entries[0]
    return {
        "device": "+".join(e["device"] for e in gpu_entries),
        "model_mb": sum(e["model_mb"] for e in gpu_entries),
        "context_mb": sum(e["context_mb"] for e in gpu_entries),
        "compute_mb": sum(e["compute_mb"] for e in gpu_entries),
    }


# ── Vulkan/CUDA/ROCm device の物理容量 parser ──────────────
# rationale: ``llama-fit-params`` (および llama-server) の起動ログには
# 利用可能な device 列挙行が以下 2 フォーマットのいずれかで出る:
#   A. llama-server: "Vulkan0 : AMD Radeon(TM) 890M Graphics (32909 MiB, 31264 MiB free)"
#      → total と free 両方取れる
#   B. llama-fit-params -v: "using device Vulkan0 (AMD Radeon(TM) 890M Graphics) (unknown id) - 31264 MiB free"
#      → free のみ。total は不明 → 安全側で free を total として扱う
# 判定ロジック側では total を主軸に使う。
_DEVICE_MEM_RE_FULL = re.compile(
    # フォーマット A: 行頭 device 名 + `:` + 製品名 + `(NNN MiB, NNN MiB free)`
    r"^([A-Za-z][A-Za-z0-9_]*)\s*:\s*.+\((\d+)\s*MiB,\s*(\d+)\s*MiB\s*free\)\s*$"
)
_DEVICE_MEM_RE_USING = re.compile(
    # フォーマット B: 行中 "using device <name> (...)... - NNN MiB free"
    r"using device ([A-Za-z][A-Za-z0-9_]*)\s*\(.+?\).*?-\s*(\d+)\s*MiB\s*free"
)


def _parse_device_memory(text: str) -> dict[str, tuple[int, int]]:
    """device 列挙行から ``{device_name: (total_mib, free_mib)}`` を返す。

    フォーマット A (total + free) を優先採用、見つからない device はフォーマット B
    (free のみ) で補完する。フォーマット B では total が取れないため free を
    total として扱う (= 起動前空きを基準にした安全側評価)。

    マッチ 0 件で空 dict を返す (呼び出し側で auto-tune を諦めるシグナル)。
    """
    result: dict[str, tuple[int, int]] = {}
    for raw in (text or "").splitlines():
        line = raw.strip()
        m = _DEVICE_MEM_RE_FULL.match(line)
        if m:
            result[m.group(1)] = (int(m.group(2)), int(m.group(3)))
            continue
        m = _DEVICE_MEM_RE_USING.search(line)
        if m and m.group(1) not in result:
            free = int(m.group(2))
            result[m.group(1)] = (free, free)
    return result


def _resolve_context_size_for(cfg: dict, name: str) -> int:
    """Tier 1 推定で ``-c`` に渡す context_size を解決する。

    各モデルの context 設定を優先し、未設定時はモデル種別ごとに妥当な
    既定値 (チャット系 4096 / アシスト 8192 / 埋め込み・リランカー 8192)
    を採用する。
    """
    if name == "base":
        return int((cfg.get("llama") or {}).get("context_size", 4096))
    if name == "assist":
        return int(
            ((cfg.get("assist_model") or {}).get("local") or {}).get(
                "context_size", 8192,
            )
        )
    if name == "embed":
        return int((cfg.get("embedding") or {}).get("context_size", 8192))
    if name == "reranker":
        return int((cfg.get("reranker") or {}).get("context_size", 8192))
    return 4096


def _run_fit_params(
    binary: str,
    model_path: str,
    context_size: int,
    gpu_layers: int,
    timeout: float,
) -> dict | None:
    """``llama-fit-params -fitp on`` を実行し GPU 集計値を返す

    バイナリ未存在 / タイムアウト / 終了コード非 0 / stdout 解析失敗の
    いずれもサイレントに ``None`` を返す。呼び出し側で Tier 2 にフォールバック。

    モデルが存在しないパスの場合 ``llama-fit-params`` 自体が即時非 0 終了
    するため、呼び出し前に ``Path.exists()`` で防御するのが望ましい。
    """
    cmd = [
        binary,
        "-m", str(model_path),
        "-c", str(int(context_size)),
        "--fit-print", "on",
        "-ngl", str(int(gpu_layers)),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None
    if result.returncode != 0:
        return None
    return _parse_fit_params_output(result.stdout or "")


def _run_fit_params_with_meta(
    binary: str,
    model_path: str,
    context_size: int,
    gpu_layers: int,
    timeout: float,
) -> tuple[dict | None, dict[str, tuple[int, int]]]:
    """``_run_fit_params`` の拡張版: device memory 情報も併せて返す。

    返値: ``(vram_estimate, device_memory)`` のタプル。
      - vram_estimate: ``_parse_fit_params_output`` と同じ dict (失敗時 None)
      - device_memory: ``{device_name: (total_mib, free_mib)}`` (空 dict 可)

    device 列挙行は実装によって stdout / stderr のどちらに出るか不定なので
    両 stream を結合して走査する。``-v`` (verbose) を追加して device 行
    (``using device <name> (...) - NNN MiB free``) を確実に出力させる
    (verbose なしでは device 行が省略されるバージョンがあるため)。
    Tier 1 推定 (``_run_fit_params``) には影響しないよう本関数を分離。
    """
    cmd = [
        binary,
        "-m", str(model_path),
        "-c", str(int(context_size)),
        "--fit-print", "on",
        "-ngl", str(int(gpu_layers)),
        "-v",  # verbose: device 列挙行を確実に出力させる
    ]
    try:
        # encoding を utf-8 + errors=replace で明示。Windows の cp932 デフォルト
        # では verbose 出力中の非 ASCII バイト (GGUF メタなど) で UnicodeDecodeError
        # が発生する。
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None, {}
    combined = (result.stdout or "") + "\n" + (result.stderr or "")
    device_mem = _parse_device_memory(combined)
    if result.returncode != 0:
        return None, device_mem
    return _parse_fit_params_output(result.stdout or ""), device_mem


# ── 自動段階縮小ロジック ──────────────────────────────
# rationale: base/assist 両方を同じ ratio で縮小する。異 ratio
# (base のみ縮小) は会話応答速度が顕著に落ちる base を優先犠牲にする形に
# なるため UX 上 NG とした。
_AUTO_NGL_RATIOS: tuple[float, ...] = (1.0, 0.8, 0.6, 0.4, 0.0)


def _scaled_vram_mb(full_estimate: dict, ratio: float) -> int:
    """ngl 縮小後の VRAM 推定 MiB を返す。

    rationale: ``model_mb`` のみ ratio に線形比例で縮小、``context_mb`` /
    ``compute_mb`` は安全側で据置 (= 多めに見積もる)。実際は ngl=0 で
    ctx/compute も大幅減るが、上振れさせて「縮小不足で起動後 OOM」を回避。
    ratio=0 のときだけ全成分 0 (= CPU 配置)。
    """
    if ratio <= 0.0:
        return 0
    model_mb = int(full_estimate.get("model_mb", 0) or 0)
    ctx_mb = int(full_estimate.get("context_mb", 0) or 0)
    compute_mb = int(full_estimate.get("compute_mb", 0) or 0)
    return int(round(model_mb * ratio)) + ctx_mb + compute_mb


def _calc_auto_gpu_layers(
    *,
    base_layers_total: int,
    assist_layers_total: int,
    base_full_estimate: dict,
    assist_full_estimate: dict,
    gpu_total_mib: int,
    headroom_mib: int,
) -> tuple[int, int, str]:
    """base/assist 両方の ``-ngl`` を同じ ratio で段階縮小、最初に予算内に
    収まる組を返す。

    Args:
        base_layers_total: base モデルの全 layer 数 (GGUF block_count)
        assist_layers_total: assist モデルの全 layer 数
        base_full_estimate: ngl=999 (全 offload) 時の Tier 1 推定
            ``{"model_mb", "context_mb", "compute_mb"}``
        assist_full_estimate: 同上、assist 側
        gpu_total_mib: GPU 物理容量 (MiB)
        headroom_mib: Vulkan host buffer 予約 (MiB)

    Returns:
        ``(base_ngl, assist_ngl, reason_str)`` のタプル。
        ratio=1.0 採用時は ``(999, 999, ...)``、CPU フォールバック時は
        ``(0, 0, ...)`` を返す。
    """
    budget = max(gpu_total_mib - headroom_mib, 0)
    for ratio in _AUTO_NGL_RATIOS:
        if ratio >= 1.0:
            base_ngl = 999
            assist_ngl = 999
        else:
            base_ngl = int(round(base_layers_total * ratio))
            assist_ngl = int(round(assist_layers_total * ratio))
        base_vram = _scaled_vram_mb(base_full_estimate, ratio)
        assist_vram = _scaled_vram_mb(assist_full_estimate, ratio)
        total = base_vram + assist_vram
        if total <= budget:
            reason = (
                f"ratio={ratio:.0%} base={base_ngl}/{base_layers_total} "
                f"assist={assist_ngl}/{assist_layers_total} "
                f"est={total}MiB budget={budget}MiB "
                f"(gpu_total={gpu_total_mib}MiB - headroom={headroom_mib}MiB)"
            )
            return base_ngl, assist_ngl, reason
    # ratio=0.0 でも収まらない (異常状態) → CPU フォールバック
    reason = (
        f"all-CPU fallback: estimate exceeds budget even at ratio=0 "
        f"(budget={budget}MiB gpu_total={gpu_total_mib}MiB "
        f"headroom={headroom_mib}MiB)"
    )
    return 0, 0, reason


# ── auto エントリ: GGUF parse + fit-params + 段階縮小 + キャッシュ ──
_AUTO_NGL_CACHE_KEY = "__auto_ngl_cache__"


def _resolve_auto_gpu_layers(
    cfg: dict, project_root: Path,
) -> dict[str, int] | None:
    """``gpu_layers="auto"`` 指定時の base/assist 自動段階縮小を実行。

    1 度だけ計算し ``cfg[_AUTO_NGL_CACHE_KEY]`` に dict をキャッシュする。
    複数回の呼び出し (base / assist 解決時) でも GGUF parse + fit-params
    subprocess は 1 度ずつしか走らない。

    ``runtime.gpu_auto_tune_enabled: false`` のときは即座に None を返し、
    呼び出し側は既定 999 にフォールバックする。GGUF / fit-params のいずれかが
    失敗した場合も None を返し WARNING を 1 行出して同様にフォールバック。

    Returns:
        ``{"base_ngl": int, "assist_ngl": int, "reason": str}`` または None。
    """
    # 既にキャッシュ済みなら再利用
    cached = cfg.get(_AUTO_NGL_CACHE_KEY)
    if cached is not None:
        return cached if cached else None  # 空 dict は失敗マーカー

    runtime_cfg = cfg.get("runtime", {}) or {}
    if not runtime_cfg.get("gpu_auto_tune_enabled", True):
        cfg[_AUTO_NGL_CACHE_KEY] = {}  # 失敗マーカー (再計算抑止)
        print(
            "[launch] WARNING: gpu_layers='auto' specified but "
            "runtime.gpu_auto_tune_enabled=false; falling back to -ngl 999"
        )
        return None

    binary = runtime_cfg.get("fit_params_binary", "llama-fit-params")
    timeout = float(runtime_cfg.get("fit_params_timeout_sec", 10.0))
    headroom = int(runtime_cfg.get("vulkan_host_buffer_headroom_mib", 4096))

    # base / assist モデルパス
    base_model_path = _resolve_model_path(cfg, "base_model", "", project_root)
    assist_local = (cfg.get("assist_model") or {}).get("local") or {}
    assist_model_rel = assist_local.get("model") or (cfg.get("model_paths") or {}).get(
        "assist_model", ""
    )
    assist_model_path = (
        project_root / assist_model_rel
        if assist_model_rel and not Path(assist_model_rel).is_absolute()
        else Path(assist_model_rel)
        if assist_model_rel
        else None
    )

    if not base_model_path.exists() or assist_model_path is None or not assist_model_path.exists():
        cfg[_AUTO_NGL_CACHE_KEY] = {}
        print(
            "[launch] WARNING: gpu_layers='auto' but model file(s) missing; "
            "falling back to -ngl 999"
        )
        return None

    # GGUF から layer 数取得
    base_layers = _read_gguf_layer_count(base_model_path)
    assist_layers = _read_gguf_layer_count(assist_model_path)
    if base_layers is None or assist_layers is None:
        cfg[_AUTO_NGL_CACHE_KEY] = {}
        print(
            f"[launch] WARNING: gpu_layers='auto' but failed to read "
            f"block_count (base={base_layers}, assist={assist_layers}); "
            f"falling back to -ngl 999"
        )
        return None

    # fit-params で全 offload 時の VRAM 推定 + device 容量を取得
    base_ctx = _resolve_context_size_for(cfg, "base")
    assist_ctx = _resolve_context_size_for(cfg, "assist")
    base_full, base_devmem = _run_fit_params_with_meta(
        binary, str(base_model_path), base_ctx, 999, timeout,
    )
    assist_full, assist_devmem = _run_fit_params_with_meta(
        binary, str(assist_model_path), assist_ctx, 999, timeout,
    )
    if base_full is None or assist_full is None:
        cfg[_AUTO_NGL_CACHE_KEY] = {}
        print(
            "[launch] WARNING: gpu_layers='auto' but llama-fit-params failed; "
            "falling back to -ngl 999"
        )
        return None

    # device 容量 (両方を merge し、最も大きな total を持つ GPU device を採用)
    merged_devmem = {**base_devmem, **assist_devmem}
    # ``host`` 行は CPU 側システム RAM のためスキップ
    gpu_devices = {k: v for k, v in merged_devmem.items() if k.lower() != "host"}
    if not gpu_devices:
        cfg[_AUTO_NGL_CACHE_KEY] = {}
        print(
            "[launch] WARNING: gpu_layers='auto' but no GPU device detected "
            "from fit-params output; falling back to -ngl 999"
        )
        return None
    # 最大 total を持つ device を採用 (multi-GPU の場合は最大 1 枚に基づく
    # 保守的判定。共有 iGPU 環境では 1 枚しかないため影響なし)
    gpu_total_mib = max(total for total, _free in gpu_devices.values())

    base_ngl, assist_ngl, reason = _calc_auto_gpu_layers(
        base_layers_total=base_layers,
        assist_layers_total=assist_layers,
        base_full_estimate=base_full,
        assist_full_estimate=assist_full,
        gpu_total_mib=gpu_total_mib,
        headroom_mib=headroom,
    )

    print(
        f"[launch] auto-tuned gpu_layers: base={base_ngl}/{base_layers}, "
        f"assist={assist_ngl}/{assist_layers}"
    )
    print(f"[launch]   reason: {reason}")

    result = {
        "base_ngl": base_ngl,
        "assist_ngl": assist_ngl,
        "reason": reason,
    }
    cfg[_AUTO_NGL_CACHE_KEY] = result
    return result


def _try_estimate_via_fit_params(
    cfg: dict, base_estimates: dict[str, dict],
) -> dict[str, dict]:
    """Tier 1: ``llama-fit-params`` で base_estimates を上書きする

    各 entry のうち ``present=True`` かつ ``gpu_layers > 0`` のモデルについて
    のみ Tier 1 を試行し、成功した entry のみ ``estimated_via=llama-fit-params``
    に置き換える。失敗した entry は base_estimates (Tier 2) のまま。

    バイナリ自体が PATH に無い場合は最初の試行で None が返り、以降の
    モデルもサイレントにスキップされる (各呼び出しで ``FileNotFoundError``
    が拾われる)。
    """
    runtime_cfg = cfg.get("runtime", {}) or {}
    binary = runtime_cfg.get("fit_params_binary", "llama-fit-params")
    timeout = float(runtime_cfg.get("fit_params_timeout_sec", 10.0))

    enriched: dict[str, dict] = dict(base_estimates)
    for name, entry in base_estimates.items():
        if not entry.get("present"):
            continue
        if int(entry.get("gpu_layers", 0)) <= 0:
            continue
        path = entry.get("path") or ""
        if not path:
            continue
        ctx = _resolve_context_size_for(cfg, name)
        ngl = int(entry["gpu_layers"])
        tier1 = _run_fit_params(binary, path, ctx, ngl, timeout)
        if tier1 is None:
            continue
        enriched[name] = {
            **entry,
            "model_mb": tier1["model_mb"],
            "context_mb": tier1["context_mb"],
            "compute_mb": tier1["compute_mb"],
            "device": tier1["device"],
            "vram_mb": (
                tier1["model_mb"]
                + tier1["context_mb"]
                + tier1["compute_mb"]
            ),
            "estimated_via": "llama-fit-params",
        }
    return enriched


def estimate_vram_usage_mb(
    cfg: dict,
    project_root: Path | None = None,
    *,
    prefer_fit_params: bool = True,
) -> dict[str, dict]:
    """各 llama-server の GPU VRAM 使用量を推定する

    返却値は以下の形のネスト dict (各 entry に Tier 1/2 共通キーが揃う)::

        {
            "base": {
                "model_mb": 4200, "gpu_layers": 999, "vram_mb": 4800,
                "context_mb": 480, "compute_mb": 120, "device": "CUDA0",
                "present": True, "path": "...",
                "estimated_via": "llama-fit-params",
            },
            ...
        }

    2 段構え:

    - **Tier 1** (preferred): ``llama-fit-params -fitp on`` を ``-m PATH
      -c CTX -ngl NGL`` で実行し、device 別の使用 MiB (model / context /
      compute) を取得。``vram_mb = model + context + compute``。
    - **Tier 2** (fallback): GGUF ファイルサイズ (MB) を上限とする旧来
      ヒューリスティック。``gpu_layers > 0`` ならファイルサイズ全量、
      ``gpu_layers == 0`` なら 0 MB。``context_mb`` / ``compute_mb`` /
      ``device`` は ``None``。

    Tier 1 はバイナリ未存在 / タイムアウト / 解析失敗時にサイレントに
    Tier 2 にフォールバックする。``prefer_fit_params=False`` または
    ``runtime.fit_params_enabled=false`` で Tier 1 を完全スキップ。

    ファイルが存在しない / モデル未設定 / エディションで無効化されている
    場合は ``present: False`` を返し、合算から除外される。
    """
    if project_root is None:
        project_root = Path.cwd()

    base_result = _estimate_via_gguf_size(cfg, project_root)

    if not prefer_fit_params:
        return base_result
    runtime_cfg = cfg.get("runtime", {}) or {}
    if not runtime_cfg.get("fit_params_enabled", True):
        return base_result

    return _try_estimate_via_fit_params(cfg, base_result)


def suggest_total_vram_budget_mb(
    estimates: dict[str, dict], *, headroom_ratio: float = 0.1,
) -> int | None:
    """Tier 1 結果から推奨 ``runtime.total_vram_budget_mb`` を計算する

    Tier 1 (``estimated_via == "llama-fit-params"``) で見積もられた entry が
    1 つ以上ある場合のみ算出する (Tier 2 のみの結果は精度が荒すぎるため
    推奨値を出さない)。``headroom_ratio`` (既定 10%) のヘッドルームを
    付加した整数 MiB を返す。
    """
    has_tier1 = any(
        e.get("estimated_via") == "llama-fit-params" for e in estimates.values()
    )
    if not has_tier1:
        return None
    total = sum(int(e.get("vram_mb", 0)) for e in estimates.values())
    if total <= 0:
        return None
    return int(round(total * (1.0 + headroom_ratio)))


def format_placement_summary(estimates: dict[str, dict]) -> list[str]:
    """配置サマリを人間可読な複数行文字列として整形する

    Tier 1 (llama-fit-params) entry は device / context / compute も
    併せて表示する::

        base      : GPU CUDA0  ngl=999  model_size=4200MB  ctx=480MB  compute=120MB  est_vram=4800MB

    Tier 2 (gguf-size) entry は旧フォーマットを踏襲する::

        base      : GPU ngl=999 model_size= 4800MB est_vram=4800MB

    base entry に ``draft_model_mb``
    が含まれる場合は subline として draft GGUF 情報を追加表示する。
    """
    lines: list[str] = []
    for name in ("base", "assist", "embed", "reranker"):
        entry = estimates.get(name, {})
        if not entry.get("present"):
            lines.append(
                f"  {name:<9s} : (skipped — not configured / model file missing)"
            )
            continue
        ngl = entry.get("gpu_layers", 0)
        placement = "GPU" if ngl > 0 else "CPU"
        model_mb = entry.get("model_mb")
        vram_mb = entry.get("vram_mb", 0)
        model_str = f"{model_mb}MB" if model_mb is not None else "?MB"
        ctx_mb = entry.get("context_mb")
        comp_mb = entry.get("compute_mb")
        device = entry.get("device")
        if ctx_mb is not None and comp_mb is not None:
            device_str = f" {device}" if device else ""
            lines.append(
                f"  {name:<9s} : {placement}{device_str}  "
                f"ngl={ngl:<4d}  model_size={model_str:>7s}  "
                f"ctx={ctx_mb}MB  compute={comp_mb}MB  est_vram={vram_mb}MB"
            )
        else:
            lines.append(
                f"  {name:<9s} : {placement:<3s} ngl={ngl:<4d} "
                f"model_size={model_str:>7s} est_vram={vram_mb}MB"
            )
        # draft model 加算 subline (mode="draft-model" 時のみ)
        draft_mb = entry.get("draft_model_mb")
        if draft_mb is not None:
            draft_path = entry.get("draft_model_path") or "(unset)"
            lines.append(
                f"  {'':<9s}   + draft: "
                f"model_size={draft_mb}MB  path={draft_path}"
            )
    return lines


def check_vram_budget(
    cfg: dict, project_root: Path | None = None, *, force: bool = False,
) -> tuple[bool, int, int | None, dict[str, dict], str]:
    """VRAM 予算との比較を行う

    Returns:
        (ok, total_vram_mb, budget_mb, estimates, message)

        - ``ok``: True なら起動継続可。False なら超過中 (force=True でも True にはならない)
        - ``total_vram_mb``: 推定 VRAM 合計
        - ``budget_mb``: ``runtime.total_vram_budget_mb`` (未設定なら None)
        - ``estimates``: モデル別内訳 (``estimate_vram_usage_mb`` の返り値)
        - ``message``: 人間可読なログ用メッセージ
    """
    estimates = estimate_vram_usage_mb(cfg, project_root)
    total_vram_mb = sum(e.get("vram_mb", 0) for e in estimates.values())
    runtime_cfg = cfg.get("runtime", {}) or {}
    budget_mb = runtime_cfg.get("total_vram_budget_mb")

    # Tier 1 (llama-fit-params) を 1 つでも採用できたかをサマリ末尾に表示する。
    has_tier1 = any(
        e.get("estimated_via") == "llama-fit-params" for e in estimates.values()
    )

    lines = ["[launch] GPU/CPU placement summary:"]
    if has_tier1:
        lines[0] = (
            "[launch] GPU/CPU placement summary (via llama-fit-params, Tier 1):"
        )
    lines.extend(format_placement_summary(estimates))
    lines.append(f"  total estimated VRAM: {total_vram_mb} MB")
    if budget_mb is None:
        lines.append(
            "  runtime.total_vram_budget_mb: (not set — skipping VRAM budget check)"
        )
    else:
        lines.append(f"  runtime.total_vram_budget_mb: {budget_mb} MB")

    # Tier 1 結果から 10% headroom 付き推奨値を提示する
    if budget_mb is None:
        suggested = suggest_total_vram_budget_mb(estimates)
        if suggested is not None:
            lines.append(
                f"  Suggested: set runtime.total_vram_budget_mb={suggested} "
                "(current+10% headroom)"
            )
        return True, total_vram_mb, None, estimates, "\n".join(lines)

    if total_vram_mb > int(budget_mb):
        over = total_vram_mb - int(budget_mb)
        lines.append(
            f"[launch] WARNING: estimated VRAM {total_vram_mb} MB exceeds budget "
            f"{budget_mb} MB (over by {over} MB)"
        )
        if force:
            lines.append(
                "[launch] --force specified: continuing despite VRAM budget overrun"
            )
            return True, total_vram_mb, int(budget_mb), estimates, "\n".join(lines)
        lines.append(
            "[launch] Aborting. Re-run with --force to override, or set "
            "embedding.gpu_layers / reranker.gpu_layers to 0 (CPU fallback) "
            "or raise runtime.total_vram_budget_mb to continue."
        )
        return False, total_vram_mb, int(budget_mb), estimates, "\n".join(lines)

    lines.append(f"[launch] VRAM budget OK ({total_vram_mb} MB / {budget_mb} MB)")
    return True, total_vram_mb, int(budget_mb), estimates, "\n".join(lines)


# ── llama-server バージョン検査 ──────────────────────


# 上流公式リリース (`b<N>`)、タグ無しビルド (`build = N (commit)`)、
# 最近の `--version` ヘッダ (`version: N (commit)`) の 3 形式に対応する。
# 数字部のみを 10 進整数として抽出する。
_BUILD_NUMBER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bb(\d+)\b"),                          # b8946
    re.compile(r"\bbuild\s*[=:]\s*(\d+)\b"),            # build = 8946 / build: 8946
    re.compile(r"\bversion:\s*(\d+)\s*\("),             # version: 8946 (commit)
)


def _parse_build_number(text: str) -> int | None:
    """``llama-server --version`` の出力テキストから build 番号 (整数) を抽出する。

    `b8946` / `build = 8946 (commit)` / `version: 8946 (commit)` の 3 形式に対応。
    どのパターンにもマッチしない場合は None を返す (build 番号を露出しない
    カスタムビルドへのフォールバック)。
    """
    if not text:
        return None
    for pattern in _BUILD_NUMBER_PATTERNS:
        match = pattern.search(text)
        if match:
            try:
                return int(match.group(1))
            except ValueError:  # pragma: no cover - 正規表現が \d+ なので通常通らない
                continue
    return None


def _parse_build_requirement(value: str | int | None) -> int | None:
    """``runtime.min_llamacpp_build`` の値を整数の build 番号として正規化する。

    受理する形式:
      - ``"b8946"`` / ``"B8946"`` (大文字小文字非依存)
      - ``"8946"`` (整数文字列)
      - ``8946`` (整数)

    解釈不能な場合は None を返す (要件チェックをスキップ)。
    """
    if value is None:
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    text = str(value).strip()
    if not text:
        return None
    if text[0] in ("b", "B"):
        text = text[1:]
    if not text.isdigit():
        return None
    return int(text)


def _probe_llamacpp_build(
    binary: str = "llama-server", timeout: float = 2.0,
) -> int | None:
    """``llama-server --version`` を実行し build 番号を抽出する

    タイムアウト / バイナリ不存在 / 抽出失敗のいずれもサイレントに None を返す。
    バイナリが build 番号を露出しないカスタムビルドにも対応するため、抽出
    失敗を「要件未満」とは扱わない (呼び出し側で警告のみで継続)。
    """
    try:
        result = subprocess.run(
            [binary, "--version"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None
    combined = (result.stdout or "") + "\n" + (result.stderr or "")
    return _parse_build_number(combined)


def check_llamacpp_build(
    cfg: dict, *, probe: Callable[[], int | None] | None = None,
) -> tuple[bool, int | None, int | None, list[str]]:
    """llama-server の build 番号を取得し最低要件と比較する

    Args:
        cfg: ``config.yaml`` をパースした dict。
        probe: build 番号を返す callable (テスト差し込み用)。
            ``None`` のとき ``_probe_llamacpp_build()`` を使う。

    Returns:
        (ok, detected_build, required_build, messages):
            - ``ok``: 起動継続可なら True。``enforce_min_llamacpp_build: true``
              かつ ``detected < required`` のときのみ False。
            - ``detected_build``: 検出値 (None なら抽出失敗 = カスタムビルド扱い)
            - ``required_build``: 要件値 (None なら検査スキップ)
            - ``messages``: 人間可読なログ用メッセージ (1 行 1 要素)。
    """
    runtime_cfg = cfg.get("runtime", {}) or {}
    required = _parse_build_requirement(runtime_cfg.get("min_llamacpp_build"))
    enforce = bool(runtime_cfg.get("enforce_min_llamacpp_build", False))

    probe_fn = probe or _probe_llamacpp_build
    detected = probe_fn()

    messages: list[str] = []
    if detected is None:
        messages.append(
            "[launch] llama-server build: (unknown — custom build or "
            "version flag did not expose a build number)"
        )
    else:
        messages.append(f"[launch] llama-server build: b{detected}")

    if required is None:
        return True, detected, None, messages

    if detected is None:
        messages.append(
            f"[launch] WARNING: cannot verify llama-server >= b{required} "
            "(build number not detected). Continuing without enforcement."
        )
        return True, detected, required, messages

    if detected < required:
        messages.append(
            f"[launch] WARNING: llama-server build b{detected} < required "
            f"b{required}."
        )
        if enforce:
            messages.append(
                "[launch] runtime.enforce_min_llamacpp_build=true: aborting. "
                "Update llama.cpp (git pull && cmake --build build "
                "--target llama-server) or set the flag to false to continue."
            )
            return False, detected, required, messages
        messages.append(
            "[launch] runtime.enforce_min_llamacpp_build=false: continuing "
            "with warning only. Set true to enforce hard abort."
        )
        return True, detected, required, messages

    messages.append(
        f"[launch] llama-server build b{detected} >= required b{required} (OK)"
    )
    return True, detected, required, messages


def wait_for_health(host: str, port: int, timeout: int = 30) -> bool:
    """llama-server のヘルスチェックをポーリング"""
    url = f"http://{host}:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = httpx.get(url, timeout=2.0)
            if resp.status_code == 200:
                return True
        except (httpx.ConnectError, httpx.TimeoutException):
            pass
        time.sleep(1.0)
    return False


def _start_and_wait(
    cmd: list[str], name: str, host: str, port: int,
    *, env: dict | None = None,
) -> subprocess.Popen:
    """llama-server プロセスを起動しヘルスチェック"""
    print(f"[launch] {name}: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, env=env)
    print(f"[launch] Waiting for {name} at {host}:{port}...")
    if wait_for_health(host, port):
        print(f"[launch] {name} is ready")
    else:
        print(f"[launch] WARNING: {name} health check timed out")
    return proc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Launch llama-server instances")
    parser.add_argument("config", nargs="?", default="config.yaml", help="Config file path")
    parser.add_argument("--all", action="store_true", help="Launch all configured servers")
    parser.add_argument("--embed", action="store_true", help="Launch embedding server only")
    parser.add_argument("--assist", action="store_true", help="Launch assist model server only")
    parser.add_argument("--reranker", action="store_true", help="Launch reranker server only")
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "VRAM 予算超過時も強制起動する。--all と併用することを想定"
        ),
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"[launch] ERROR: config file not found: {cfg_path}")
        sys.exit(1)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    project_root = cfg_path.parent

    procs: list[subprocess.Popen] = []

    launch_base = not args.embed and not args.assist and not args.reranker
    launch_embed = args.embed or args.all
    launch_assist = args.assist or args.all
    launch_reranker = args.reranker or args.all

    # llama-server バージョン検査。--all / 個別起動を問わず
    # llama-server バイナリを起動する全パスで一度だけ build 番号を確認する。
    # ベース / アシスト / 埋め込み / リランカーは同一バイナリを使うため、
    # 起動前に 1 回プローブして INFO ログに出力する。
    build_ok, _detected, _required, build_messages = check_llamacpp_build(cfg)
    for line in build_messages:
        if "WARNING" in line or "aborting" in line:
            print(line, file=sys.stderr)
        else:
            print(line)
    if not build_ok:
        sys.exit(3)

    # VRAM 予算検査は --all (全モデル一括起動) 時のみ実行する。
    # 個別起動 (--embed / --assist / --reranker) では既存プロセスとの
    # 合算が読み取れないため検査をスキップする。
    if args.all:
        ok, _total, _budget, _estimates, message = check_vram_budget(
            cfg, project_root, force=args.force,
        )
        print(message)
        if not ok:
            sys.exit(2)

    try:
        # ベースモデル
        if launch_base:
            cmd = build_llama_cmd(cfg, project_root)
            host = cfg["llama"].get("host", "localhost")
            port = cfg["llama"].get("port", 8080)
            procs.append(_start_and_wait(cmd, "base-model", host, port))

        # アシストモデル
        if launch_assist:
            assist_cmd = build_assist_cmd(cfg, project_root)
            if assist_cmd:
                assist_cfg = cfg.get("assist_model", {}).get("local", {})
                host = assist_cfg.get("host", "127.0.0.1")
                port = assist_cfg.get("port", 8081)
                procs.append(_start_and_wait(assist_cmd, "assist-model", host, port))
            else:
                print("[launch] Assist model not configured, skipping")

        # エンベッド
        if launch_embed:
            embed_cmd = build_embed_cmd(cfg, project_root)
            if embed_cmd:
                emb_cfg = cfg.get("embedding", {})
                host = emb_cfg.get("llama_host", "localhost")
                port = emb_cfg.get("llama_port", 8082)
                # 注意: -ngl 0 (CPU モード) でも llama.cpp は Vulkan バックエンドを
                # 初期化し model loading 時に host (pinned) buffer を要求する。
                # buffer size が Vulkan device->max_buffer_size を超えると warning
                # (ggml_vulkan: Failed to allocate pinned memory) が出るが CPU buffer に
                # 自動フォールバックするため機能影響は無い (ggml-vulkan.cpp:14079 / :2671)。
                procs.append(_start_and_wait(embed_cmd, "embedding", host, port))
            else:
                print("[launch] Embedding backend is not llama-cpp, skipping")

        # リランカー
        if launch_reranker:
            reranker_cmd = build_reranker_cmd(cfg, project_root)
            if reranker_cmd:
                reranker_cfg = cfg.get("reranker", {})
                host = reranker_cfg.get("host", "localhost")
                port = reranker_cfg.get("port", 8083)
                # embed と同じ pinned-memory warning が出る可能性あり (詳細はembed側
                # コメント参照)。CPU buffer フォールバックで動作継続。
                procs.append(_start_and_wait(reranker_cmd, "reranker", host, port))
            else:
                print("[launch] Reranker not configured or not llama-cpp, skipping")

        if procs:
            for proc in procs:
                proc.wait()
    except KeyboardInterrupt:
        for proc in procs:
            proc.terminate()
        for proc in procs:
            proc.wait()
