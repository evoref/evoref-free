"""llama-server 起動スクリプト

config.yaml の llama / embedding セクションから起動コマンドを組み立て、
サブプロセスとして llama-server を起動する。

サブコマンド:
  (なし)     ベースモデル llama-server のみ起動
  --all      ベース + アシスト + エンベッドを一括起動
  --embed    エンベッド用 llama-server のみ起動

  --all 起動時に ``runtime.total_vram_budget_mb`` (config.yaml) を参照し、
  GPU オフロード対象モデルの VRAM 使用量推定合計が予算を超過する場合は
  警告してアボートする。``--force`` で強制起動可能。
  埋め込みの ``-ngl`` 既定値は 0 (CPU フォールバック) とし、
  GPU 割り当ては ``embedding.gpu_layers`` による
  明示 opt-in でのみ有効化される。

  起動時に ``llama-server --version`` を実行し build 番号をログに出力する。
  ``runtime.min_llamacpp_build`` 未満を検出すると stderr に警告を出し、
  ``runtime.enforce_min_llamacpp_build: true`` の場合は exit code 3 で
  アボートする。バイナリが build 番号を露出しないカスタムビルドの場合は
  検出失敗 (None) として警告のみで継続。

  llama.cpp
  slot 退避機構 (``--cache-ram`` / ``--cache-idle-slots``) を全 3 サーバ
  (base / assist / embed) で明示制御する。``cache_ram_mib`` /
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
  warning + 無効化する。assist / embed は対象外 (Issue 文 §スコープ)。
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


def _resolve_kv_unified(section_cfg: dict, slots: int) -> bool:
    """``kv_unified`` の解決

    上流挙動:
      - ``-np 1`` (slots 自動 / 単一 slot): unified KV cache がデフォルト ON
      - ``-np >1`` (明示複数 slot): デフォルトで非 unified 配置 (slot 毎に
        独立 KV cell プール) になり、(a) ``--cache-ram`` の idle slot offload が
        動作せず、(b) per-seq context が ``n_ctx / n_parallel`` に分割される
        (``-c 8192 -np 2`` なら各 slot 4096 に半減)

    したがって ``slots > 1`` の場合は ``--kv-unified`` を自動付与する。unified KV
    は multi-slot でも VRAM を増やさずに各シーケンスへ full ``n_ctx`` を与え
    (総 KV セル数は同じ)、かつ ``--cache-ram`` の動作前提でもある。``kv_unified``
    を ``true``/``false`` で明示指定した場合は auto 判定を上書きして尊重する。

    NOTE: 旧実装は auto 条件を ``cache_ram_mib > 0 AND slots > 1`` としていたが、
    ``cache_ram_mib: 0`` + ``slots > 1`` (例: assist) で per-seq context が黙って
    半減する footgun があったため、cache-ram から切り離した。
    """
    explicit = section_cfg.get("kv_unified")
    if explicit is None:
        return slots > 1
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
    cmd: list[str], section_cfg: dict, *, slots: int,
) -> None:
    """``--kv-unified`` を必要に応じて ``cmd`` に追加する

    ``slots > 1`` の場合に自動付与する (``kv_unified`` の明示指定で上書き可)。
    """
    if _resolve_kv_unified(section_cfg, slots=slots):
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


def _profile_reasoning_for_model(model_path: Path, project_root: Path) -> dict:
    """モデルパスから arch プロファイルの ``reasoning`` セクション (raw dict) を返す。

    GGUF arch 検出 → ``load_model_profile`` で解決。失敗 / 不在時は ``{}``。
    検証は backend 側 ``_resolve_profile_reasoning`` (ProfileReasoningConfig) で行うため
    ここでは raw を返す (docs/c_15)。
    """
    try:
        meta = read_gguf_metadata(model_path)
        profile = load_model_profile(meta.get("architecture"), project_root)
        r = profile.get("reasoning")
        return r if isinstance(r, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _append_base_reasoning_args(cmd: list[str], reasoning: dict) -> None:
    """ベースモデルの ``profile.reasoning`` から server 側 reasoning 起動フラグを付与する。

    ``server_control: true`` (llama.cpp が reasoning を認識する Qwen3/DeepSeek 系) の
    ときのみ、思考の上限化 (``--reasoning-budget``) と budget message を付与する。
    ``--reasoning-format`` は profile ``launch_flags`` 由来を尊重し、on/off は
    ``enable_thinking`` (リクエスト側 ``_build_payload``) で制御するため、ここでは emit しない。
    ``server_control: false`` (``thinking=0`` = lfm2moe 等) は no-op
    (クライアント側 ``_ReasoningFilter`` で扱う、docs/c_15)。
    """
    if not reasoning.get("server_control"):
        return
    try:
        budget = int(reasoning.get("budget_default", -1))
    except (TypeError, ValueError):
        budget = -1
    if budget >= 0:
        cmd += ["--reasoning-budget", str(budget)]
    message = reasoning.get("budget_message", "")
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


def build_mtp_args(
    model_cfg: dict,
    model_path: Path,
    *,
    warn: Callable[[str], None] | None = None,
) -> list[str]:
    """``mtp`` セクションから MTP self-speculative の ``--spec-*`` フラグを組み立てる。

    Free / Pro 共通 (Pro 限定の :func:`build_speculative_args` とは別系統)。
    モデル自身の MTP ヘッド (NextN 層) を draft に使うため外部 draft モデル不要。

    **MTP ヘッド内蔵モデルでのみ有効**。GGUF メタデータ
    ``<arch>.nextn_predict_layers > 0`` で判定し、非対応モデルには warning を
    出してフラグを付与しない (graceful degrade)。``enabled=false`` / ``mtp``
    未設定なら warning なしで ``[]``。

    Args:
        model_cfg: base なら ``cfg["llama"]``、assist なら
            ``cfg["assist_model"]["local"]`` の dict。``mtp`` サブ dict を読む。
        model_path: 解決済みモデル GGUF パス (MTP ヘッド検出用)。
        warn: warning 出力先。``None`` なら何もしない。

    Returns:
        ``llama-server`` に追加すべき引数のリスト。無効 / 非対応時は ``[]``。
    """
    mtp_cfg = model_cfg.get("mtp") or {}
    if not mtp_cfg.get("enabled", False):
        return []

    meta = read_gguf_metadata(model_path)
    if int(meta.get("nextn_predict_layers", 0) or 0) <= 0:
        if warn is not None:
            warn(
                "[launch] WARNING: mtp.enabled=true but the model has no MTP "
                f"heads (nextn_predict_layers=0): {model_path.name}. Skipping "
                "--spec-type draft-mtp (MTP requires a model such as "
                "Qwen3.5/3.6 with a built-in NextN layer)."
            )
        return []

    draft_n_max = int(mtp_cfg.get("draft_n_max", 3) or 3)
    return ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(draft_n_max)]


def lora_compatible_with_model(
    model_path: Path,
    lora_path: Path,
) -> tuple[bool, str]:
    """LoRA アダプタがモデルに適用可能かを GGUF メタデータから判定する。

    判定は 3 段階:

    1. ``general.architecture`` の一致。どちらか読めない場合も不適合扱い
       (fail-closed) — arch すら読めないファイルはアダプタとして信用
       できないため。
    2. 全テンソル形状突合 (:func:`_lora_shape_mismatch`): アダプタの全
       ``*.lora_a`` / ``*.lora_b`` をモデル側の対応 weight テンソルの
       実形状と照合する。同一 arch でもサイズ違い (例: gemma-4-E2B 1536
       vs E4B 2560) や head 構成違い、block 数を超えるターゲットは
       llama-server が "tensor has incorrect shape" でコンテキスト生成に
       失敗しプロセスごと落ちるため、arch 一致だけでは適合と言えない。
    3. 系統 (lineage) チェック: アダプタに ``evoref.trained_on_model``
       (学習元モデルの filename stem、Level 2 トレーナーが刻む) がある
       場合、現モデルの stem と完全一致 (大文字小文字無視) を要求する。
       LoRA は特定の重みへの差分なので、同一 arch・同一形状でも別モデル
       (例: Qwen3.5-9B と同 arch の別 finetune) に当てると silent な品質
       摂動になる。GGUF の identity メタデータ (``general.name`` 等) は
       量子化違いの同一モデルすら識別できない品質のため (qat 変換由来の
       ゴミ名等)、再量子化も「系統不明」として不適合側に倒す (誤って
       捨てる実害は次の Level 2 サイクルで再学習される軽微なもの、誤って
       適用する実害は診断困難な品質摂動であるため)。

    形状が判定不能 (LoRA テンソル無し / どちらかのテンソル情報節が
    読めない) の場合、および stamp が無いレガシーアダプタの系統は
    fail-open (arch + 形状のみで判定)。arch 側の fail-closed との非対称は
    意図的 — テンソル命名の異なる正当なアダプタを誤って捨てないため。
    Free エディションはトレーナーを持たず stamp 付きアダプタを生成しない
    ので、系統チェックは Free では実質 inert。

    起動側 (:func:`_lora_compatible`) と model migration 側
    (``backend.free.core.model_migration``) の両層で共有する単一の述語。

    Returns:
        ``(compatible, reason)``。reason はログ向けの英語短文。
    """
    model_meta = read_gguf_metadata(model_path)
    model_arch = model_meta.get("architecture")
    lora_meta = read_gguf_metadata(lora_path)
    lora_arch = lora_meta.get("architecture")
    if not model_arch or not lora_arch:
        return False, (
            f"architecture unreadable (model={model_arch or '?'}, "
            f"lora={lora_arch or '?'})"
        )
    if model_arch != lora_arch:
        return False, (
            f"architecture mismatch (model={model_arch}, lora={lora_arch})"
        )

    mismatch = _lora_shape_mismatch(model_path, lora_path)
    if mismatch is not None:
        return False, mismatch

    stamp = lora_meta.get("trained_on_model")
    if stamp and stamp.casefold() != model_path.stem.casefold():
        return False, (
            f"lineage mismatch (adapter trained on {stamp}, "
            f"model is {model_path.stem})"
        )
    return True, f"architecture={model_arch}"


def _lora_compatible(
    model_path: Path,
    lora_path: Path,
    *,
    warn: Callable[[str], None] | None = None,
) -> bool:
    """LoRA アダプタとモデルの互換性 (arch + 形状) を判定する。

    モデル切替 (``POST /api/model/{component}/migrate`` 等) 後に旧モデル向け
    LoRA が残存すると、llama-server が arch 不一致または tensor 形状不一致で
    コンテキスト生成に失敗しプロセスごと落ちる。付与直前に
    :func:`lora_compatible_with_model` で検証し、不適合は warning を出して
    ``--lora`` を諦める (``build_mtp_args`` と同じ「非対応なら機能を諦める」
    方針)。誤ってスキップする実害は警告ログのみだが、誤って付与する実害は
    プロセスクラッシュそのものであるため、この非対称性を正当化する。
    """
    ok, reason = lora_compatible_with_model(model_path, lora_path)
    if not ok and warn is not None:
        warn(
            f"[launch] WARNING: incompatible LoRA ({reason}): "
            f"{lora_path.name}. Skipping --lora."
        )
    return ok


def _cvector_compatible(
    model_path: Path,
    cvec_path: Path,
    *,
    warn: Callable[[str], None] | None = None,
) -> bool:
    """control vector の direction 次元とモデルの ``embedding_length`` を照合する。

    次元不一致の control vector を ``--control-vector`` で渡すと llama-server
    がロード失敗でプロセスごと落ちる (LoRA の形状不一致と同型)。cvector GGUF
    は ``general.architecture`` を持たないことがあるため arch 照合はせず、
    ``direction.<n>`` テンソルの ne0 とモデル ``embedding_length`` の比較のみ
    行う。判定不能 (テンソル情報不読 / direction テンソル無し /
    embedding_length 不明) は従来どおり適用する (fail-open)。
    """
    cvec_shapes = read_gguf_tensor_shapes(cvec_path)
    if not cvec_shapes:
        return True
    model_emb = read_gguf_metadata(model_path).get("embedding_length")
    if model_emb is None:
        return True
    for name, dims in sorted(cvec_shapes.items()):
        if name.startswith("direction.") and dims and dims[0] != model_emb:
            if warn is not None:
                warn(
                    "[launch] WARNING: incompatible control vector "
                    f"(direction dim={dims[0]}, model embedding_length="
                    f"{model_emb}): {cvec_path.name}. "
                    "Skipping --control-vector."
                )
            return False
    return True


def build_llama_cmd(
    cfg: dict,
    project_root: Path | None = None,
    *,
    model_override: str | None = None,
    lora_override: str | Path | None = None,
    port_override: int | None = None,
) -> list[str]:
    """config.yaml の llama セクションから起動コマンドを生成

    Level 2 base=spsa-real-eval の候補評価では ``lora_override`` (候補 GGUF LoRA
    パス) と ``port_override`` (スクラッチポート) を指定して ephemeral サーバを
    起動する (build_assist_cmd と同じパターン)。いずれも未指定 (通常運用) の
    ときは従来挙動と完全に等価。
    """
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

    port = port_override if port_override is not None else lc.get("port", 8080)

    cmd = [
        "llama-server",
        "-m", str(base_model_path),
        "--port", str(port),
        "-c", str(resolve_context_size_for(
            cfg, "base", project_root, model_override=model_override,
        )),
        "-ngl", str(_resolve_base_gpu_layers(cfg, project_root)),
        "-b", str(lc.get("batch_size", 512)),
    ]

    # LoRA アダプタ。lora_override (Level 2 base=spsa-real-eval 候補評価) は無条件で
    # 付与する (候補 GGUF は harness が起動直前に書き出すため exists チェックしない)。
    # 通常運用は既存アダプタが存在し、かつモデルと arch が一致する場合のみ付与する。
    if lora_override is not None:
        cmd += ["--lora", str(Path(lora_override))]
    else:
        lora_path = lp.get("lora_adapter", "local/models/adapter.gguf")
        lora_full = Path(lora_path)
        if not lora_full.is_absolute():
            lora_full = project_root / lora_full
        if lora_full.exists() and _lora_compatible(
            base_model_path, lora_full, warn=lambda m: print(m, file=sys.stderr),
        ):
            cmd += ["--lora", str(lora_full)]

    # Level 2 base=C: control vector (残差ストリーム操舵)。
    # learning.level2_base_method=='cvector' かつファイルが存在するときのみ適用する。
    # config 駆動のみ (backend.pro / backend.edition の import は行わない = 横断スクリプト)。
    # Free は control_vector.gguf を生成しないので inert。適用は次回起動時。
    learning_cfg = cfg.get("learning", {}) or {}
    if learning_cfg.get("level2_base_method") == "cvector":
        cvec_path = lp.get("control_vector_adapter", "local/models/control_vector.gguf")
        cvec_full = Path(cvec_path)
        if not cvec_full.is_absolute():
            cvec_full = project_root / cvec_full
        if cvec_full.exists() and _cvector_compatible(
            base_model_path, cvec_full, warn=lambda m: print(m, file=sys.stderr),
        ):
            # 既定 1.0。``or`` フォールバックは使わない (0.0 は診断用の正当な値で、
            # falsy 畳み込みすると 1.0=full strength に化けるため)。dict 既定が欠損を
            # 補い、schema が非 Optional float なので None は来ない。
            scale = float(learning_cfg.get("cvector_scale", 1.0))
            if scale == 1.0:
                # --control-vector は FNAME のみ取るため、Windows のドライブレター
                # (E:) のコロンとも衝突しない (絶対パスで安全)。
                cmd += ["--control-vector", str(cvec_full)]
            else:
                # スケール指定時のみ --control-vector-scaled FNAME:SCALE。FNAME に
                # Windows 絶対パス (E:\...) を渡すとドライブレターのコロンが FNAME:SCALE
                # のセパレータと衝突するため、project_root 相対 POSIX パスを渡す
                # (llama-server は CWD=project_root 基準で解決; standalone 起動は
                # _start_and_wait が cwd=project_root を設定)。project_root 外の絶対
                # パスは衝突を避けられないため fail-fast する。
                try:
                    rel = cvec_full.relative_to(project_root).as_posix()
                except ValueError as e:
                    raise ValueError(
                        "cvector_scale != 1.0 requires control_vector_adapter to "
                        "resolve under project_root (relative path); an absolute path "
                        "outside project_root collides with the --control-vector-scaled "
                        "FNAME:SCALE separator.",
                    ) from e
                cmd += ["--control-vector-scaled", f"{rel}:{scale}"]
            layer_range = str(learning_cfg.get("cvector_layer_range", "") or "").strip()
            if layer_range:
                parts = layer_range.replace(",", " ").split()
                if len(parts) == 2:
                    cmd += ["--control-vector-layer-range", parts[0], parts[1]]

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

    # 共通 prefix 自動再利用（0 は無効）。多ターン chat で system/RAG 接頭辞の
    # KV を再 prefill せず再利用する。assist 側 build_assist_cmd と同じ扱い。
    # 注: SWA モデル (gemma-4 等) では llama.cpp が cache_reuse を自動無効化する
    # ため no-op (フラグ付与は無害)。非 SWA base のみ実効。
    cache_reuse = lc.get("cache_reuse", 0)
    if cache_reuse and int(cache_reuse) > 0:
        cmd += ["--cache-reuse", str(int(cache_reuse))]

    # idle slot offload。base は agentic ワークロード前提で
    # 既定 4096 MiB の RAM 退避バッファを確保する。slots>1 のときは
    # ``--cache-ram`` が動作するよう ``--kv-unified`` を自動付与する。
    _append_cache_ram_args(cmd, lc, default_mib=4096)
    _append_kv_unified_args(cmd, lc, slots=int(slots))

    # MTP (Multi-Token Prediction) self-speculative。Free/Pro 共通。MTP ヘッド
    # 内蔵モデルでのみ有効 (非対応は warning + 素通り)。MTP と Pro speculative は
    # どちらも ``--spec-type`` を出力するため排他: MTP が実効なら speculative を
    # スキップする (両 enabled なら MTP 優先 + warning)。
    mtp_args = build_mtp_args(
        lc,
        base_model_path,
        warn=lambda msg: print(msg, file=sys.stderr),
    )
    if mtp_args:
        cmd += mtp_args
        if (lc.get("speculative") or {}).get("enabled", False):
            print(
                "[launch] WARNING: both llama.mtp.enabled and "
                "llama.speculative.enabled are true; MTP takes precedence and "
                "--spec-* (speculative) flags are skipped.",
                file=sys.stderr,
            )
    else:
        # self-speculative decoding。Pro 限定機能。Free 判定では
        # ``enabled=true`` でも warning + フラグ未付与で素通りさせる。
        spec_args = build_speculative_args(
            lc,
            is_pro=_resolve_pro_edition(),
            project_root=project_root,
            warn=lambda msg: print(msg, file=sys.stderr),
        )
        cmd += spec_args

    # profile.reasoning (server_control:true = Qwen3/DeepSeek 系) から server 側
    # reasoning 起動フラグ (--reasoning-budget 等) を付与する。resolve_auto_model_flags
    # の前に置き fixed_flags で重複を防ぐ (docs/c_15)。on/off は enable_thinking
    # (リクエスト側 _build_payload) が担い、thinking=0 モデルは no-op。
    _append_base_reasoning_args(
        cmd, _profile_reasoning_for_model(base_model_path, project_root),
    )

    # モデル arch 別の自動フラグ (--jinja / --reasoning-format / MoE)。
    # extra_args の直前に挿入し、ユーザー指定 (extra_args) を最終 override とする。
    cmd += resolve_auto_model_flags(
        cfg,
        base_model_path,
        fixed_flags={
            "-m", "--port", "-c", "-ngl", "-b", "-t", "-fa", "--mlock",
            "-np", "--cache-type-k", "--cache-type-v", "--cache-reuse", "--cache-ram",
            "--kv-unified", "--lora", "--reasoning-budget", "--reasoning-budget-message",
            "--control-vector", "--control-vector-scaled", "--control-vector-layer-range",
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

    # Pooling 方式。embedding.pooling が明示されている場合のみ --pooling を
    # 付与し、未設定なら llama-server のモデル既定 pooling に委ねる (既存
    # モデルの挙動を変えない)。BGE-M3 (arch "bert") は CLS pooling が正しく、
    # embed 切替時に models/profiles/bert.yaml から config.yaml へ自動転写
    # される (値は EmbeddingConfig 側で検証済みのためここでは素通しする)。
    pooling = emb_cfg.get("pooling")
    if pooling:
        cmd += ["--pooling", str(pooling)]

    # 文脈長。モデル既定 n_ctx (Qwen3-Embedding=32768) は過剰なので
    # embedding.context_size (既定 8192 = max_length) に縮小して KV を節約する。
    # max_length を下回ると長い入力で 500 になるため context_size >= max_length を維持する。
    context_size = emb_cfg.get("context_size", 8192)
    cmd += ["-c", str(context_size)]

    max_length = emb_cfg.get("max_length", 8192)
    if int(context_size) < int(max_length):
        print(
            f"[launch] WARNING: embedding.context_size={context_size} is below "
            f"max_length={max_length}; long embedding inputs will be rejected. "
            f"Consider raising embedding.context_size to at least {max_length}.",
            file=sys.stderr,
        )

    # エンベッド LoRA アダプタ（存在し、かつモデルと arch が一致する場合のみ）
    embed_lora = lp.get("embed_lora_adapter", "local/models/embed_adapter.gguf")
    embed_lora_path = Path(embed_lora)
    if not embed_lora_path.is_absolute():
        embed_lora_path = project_root / embed_lora_path
    if embed_lora_path.exists() and _lora_compatible(
        embed_model_path, embed_lora_path, warn=lambda m: print(m, file=sys.stderr),
    ):
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

    # スレッド数。0 (既定) は省略して llama.cpp 自動検出 (全物理コア)。CPU 埋め込み時に
    # base/assist と CPU を分け合うため、明示値でヘッドルームを残せる。
    threads = emb_cfg.get("threads", 0)
    if threads and threads > 0:
        cmd += ["-t", str(threads)]

    # 並列スロットは常に -np で明示する (未指定だと n_parallel=auto=4 になり
    # slots × context_size の KV を無駄に確保する)。埋め込みは概ね逐次のため既定 2。
    slots = max(1, int(emb_cfg.get("slots", 2) or 2))
    cmd += ["-np", str(slots)]

    # slots > 1 では --kv-unified 無しだと per-seq context が n_ctx/slots に
    # 黙って分割される (base/assist と同じ footgun、_resolve_kv_unified 参照)。
    _append_kv_unified_args(cmd, emb_cfg, slots=slots)

    return cmd


def build_assist_cmd(
    cfg: dict,
    project_root: Path | None = None,
    *,
    lora_override: str | Path | None = None,
    port_override: int | None = None,
    model_override: str | None = None,
) -> list[str] | None:
    """config.yaml の assist_model セクションからアシストモデル用 llama-server コマンドを生成

    assist_model.local セクションがあり、model_paths.assist_model が存在する場合のみコマンドを返す。

    Level 2 assist=B の候補評価では ``lora_override`` (候補 GGUF LoRA パス) と
    ``port_override`` (スクラッチポート) を指定して ephemeral サーバを起動する。
    ``model_override`` は chat/coding モード切替時 (``model_paths.assist_coding_model``)
    に一時的に別 GGUF を読み込ませる用途 (build_llama_cmd と同じパターン)。
    いずれも未指定 (通常運用) のときは従来挙動と完全に等価。
    """
    assist_cfg = cfg.get("assist_model", {})
    local_cfg = assist_cfg.get("local", {})
    if not local_cfg:
        return None

    if project_root is None:
        project_root = Path.cwd()

    sp = cfg.get("model_paths", {})

    # アシストモデルパス（model_override 指定時はそちらを優先）
    assist_model = model_override or sp.get("assist_model", "")
    if not assist_model:
        return None
    assist_model_path = Path(assist_model)
    if not assist_model_path.is_absolute():
        assist_model_path = project_root / assist_model_path

    port = port_override if port_override is not None else local_cfg.get("port", 8081)

    # GPU layers: assist 側で明示指定があればそちらを優先し、なければ
    # llama セクションの値にフォールバックする（従来挙動）。
    # ``"auto"`` 指定時は _resolve_auto_gpu_layers がキャッシュ済み計算結果を返す。
    base_llama_cfg = cfg.get("llama", {})
    gpu_layers = _resolve_assist_gpu_layers(cfg, project_root)

    cmd = [
        "llama-server",
        "-m", str(assist_model_path),
        "--port", str(port),
        "-c", str(resolve_context_size_for(
            cfg, "assist", project_root, model_override=model_override,
        )),
        "-ngl", str(gpu_layers),
    ]

    # LoRA アダプタ。lora_override (Level 2 assist=B 候補評価) は無条件で付与する
    # (候補 GGUF は harness が起動直前に書き出すため exists チェックしない)。
    # 通常運用は local_paths.assist_lora_adapter が存在するときのみ付与し、採用済み
    # assist LoRA を次回起動で反映する。model_override (chat/coding モード切替) 時は
    # lora_override が明示されない限り LoRA を付けない — assist_lora_adapter は
    # chat 用アシストの重みに対して学習されたものであり、arch が異なりうる
    # coding 用モデルに無条件添付すると shape mismatch で起動失敗しうるため。
    if lora_override is not None:
        cmd += ["--lora", str(Path(lora_override))]
    elif model_override is None:
        lp = cfg.get("local_paths", {})
        assist_lora = lp.get("assist_lora_adapter", "local/models/assist_adapter.gguf")
        assist_lora_full = Path(assist_lora)
        if not assist_lora_full.is_absolute():
            assist_lora_full = project_root / assist_lora_full
        if assist_lora_full.exists() and _lora_compatible(
            assist_model_path, assist_lora_full, warn=lambda m: print(m, file=sys.stderr),
        ):
            cmd += ["--lora", str(assist_lora_full)]

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
    _append_kv_unified_args(cmd, local_cfg, slots=int(slots))

    # MTP (Multi-Token Prediction) self-speculative。Free/Pro 共通。MTP ヘッド
    # 内蔵モデルでのみ有効 (非対応は warning + 素通り)。assist は speculative を
    # 持たないため排他考慮は不要。
    cmd += build_mtp_args(
        local_cfg,
        assist_model_path,
        warn=lambda msg: print(msg, file=sys.stderr),
    )

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
            "--reasoning-budget-message", "--skip-chat-parsing", "--lora",
        },
        project_root=project_root,
        warn=lambda m: print(m, file=sys.stderr),
    )

    # 追加オプション
    extra_args = local_cfg.get("extra_args", [])
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


def _gguf_read_scalar_or_skip(f, vtype: int) -> object:
    """スカラーは値を返す。ARRAY (ハイブリッド/MoE の per-layer 値等) は
    バイト列を消費して ``None`` を返し、ストリーム同期を維持する。

    ``_gguf_read_scalar`` は ARRAY を扱えず即 ``ValueError`` を送出するが、
    その際に配列バイト列を一切消費しないため、呼び出し側が例外を握りつぶすと
    ファイルポインタが取り残されて以降のパースが全て desync する
    (例: LFM2-MoE の ``attention.head_count_kv`` は ARRAY[int32])。
    """
    if vtype == _GGUF_TYPE_ARRAY:
        _gguf_skip_value(f, vtype)
        return None
    return _gguf_read_scalar(f, vtype)


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
    except (OSError, struct.error, UnicodeDecodeError, ValueError, MemoryError):
        return None
    return None


def read_gguf_metadata(gguf_path: Path) -> dict:
    """GGUF ヘッダから起動フラグ決定に必要なメタデータを 1 パスで読む。

    Returns (失敗時・キー不在時も同じ shape):
        ``{"architecture": str | None, "context_length": int | None,
           "expert_count": int, "nextn_predict_layers": int,
           "has_chat_template": bool,
           "block_count" / "head_count_kv" / "head_count" / "key_length" /
           "value_length" / "embedding_length": int | None,
           "trained_on_model": str | None}``

    ``trained_on_model`` は evoref 独自 KV ``evoref.trained_on_model``
    (Level 2 トレーナーが LoRA アダプタに刻む学習元モデルの filename stem)。
    :func:`lora_compatible_with_model` の系統チェックに使う。
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
        # MTP (Multi-Token Prediction) ヘッド数。``<arch>.nextn_predict_layers``。
        # >0 で MTP 対応 (Qwen3.5/3.6 等)。0 / 不在で非対応。
        "nextn_predict_layers": 0,
        "has_chat_template": False,
        # KV キャッシュ VRAM 推定用 (estimate_kv_cache_mb)
        "block_count": None,
        "head_count_kv": None,
        "head_count": None,
        "key_length": None,
        "value_length": None,
        "embedding_length": None,
        # LoRA アダプタ専用の evoref 独自 KV (学習元モデルの filename stem)
        "trained_on_model": None,
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
                    val = _gguf_read_scalar_or_skip(f, vtype)
                    result["architecture"] = str(val) if val is not None else None
                elif key.endswith(".context_length"):
                    try:
                        result["context_length"] = int(_gguf_read_scalar_or_skip(f, vtype))
                    except (TypeError, ValueError):
                        pass
                elif key.endswith(".expert_count"):
                    try:
                        result["expert_count"] = int(_gguf_read_scalar_or_skip(f, vtype))
                    except (TypeError, ValueError):
                        pass
                elif key.endswith(".nextn_predict_layers"):
                    try:
                        result["nextn_predict_layers"] = int(
                            _gguf_read_scalar_or_skip(f, vtype)
                        )
                    except (TypeError, ValueError):
                        pass
                elif key.endswith(".block_count"):
                    try:
                        result["block_count"] = int(_gguf_read_scalar_or_skip(f, vtype))
                    except (TypeError, ValueError):
                        pass
                elif key.endswith(".attention.head_count_kv"):
                    try:
                        result["head_count_kv"] = int(_gguf_read_scalar_or_skip(f, vtype))
                    except (TypeError, ValueError):
                        pass
                elif key.endswith(".attention.head_count"):
                    try:
                        result["head_count"] = int(_gguf_read_scalar_or_skip(f, vtype))
                    except (TypeError, ValueError):
                        pass
                elif key.endswith(".attention.key_length"):
                    try:
                        result["key_length"] = int(_gguf_read_scalar_or_skip(f, vtype))
                    except (TypeError, ValueError):
                        pass
                elif key.endswith(".attention.value_length"):
                    try:
                        result["value_length"] = int(_gguf_read_scalar_or_skip(f, vtype))
                    except (TypeError, ValueError):
                        pass
                elif key.endswith(".embedding_length"):
                    try:
                        result["embedding_length"] = int(_gguf_read_scalar_or_skip(f, vtype))
                    except (TypeError, ValueError):
                        pass
                elif key == "tokenizer.chat_template":
                    result["has_chat_template"] = True
                    _gguf_skip_value(f, vtype)
                elif key == "evoref.trained_on_model":
                    val = _gguf_read_scalar_or_skip(f, vtype)
                    result["trained_on_model"] = (
                        str(val) if val is not None else None
                    )
                else:
                    _gguf_skip_value(f, vtype)
    except (OSError, struct.error, UnicodeDecodeError, ValueError, MemoryError):
        # MemoryError は desync で巨大な length を読んだ場合の backstop。
        # docstring の「パース失敗は安全な既定値を返す」契約に合わせる。
        return result
    return result


def read_gguf_tensor_shapes(gguf_path: Path) -> dict[str, tuple[int, ...]] | None:
    """GGUF のテンソル情報節から ``{name: dims}`` を読む。

    テンソル情報 (name / n_dims / dims / type / offset) はヘッダ内
    (KV 節の直後) にあるため、数 GB のモデルでも読むのは先頭の数十 KB のみ。
    dims は ggml の ne 順 (ne0=in_features, ne1=out_features)。

    LoRA アダプタの GGUF は KV メタデータに次元情報を持たない
    (``general.architecture`` / ``general.type`` / ``adapter.*`` のみ) ため、
    モデルとの形状互換はここから判定するしかない。パース失敗・I/O エラーは
    ``None`` を返し、呼び出し側は「判定不能」として扱う。
    """
    try:
        with gguf_path.open("rb") as f:
            magic = f.read(4)
            if magic != _GGUF_MAGIC:
                return None
            (version,) = struct.unpack("<I", f.read(4))
            if version < 2:
                return None
            n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
            # KV 節を正確に消費してテンソル情報節の先頭に位置合わせする
            for _ in range(n_kv):
                (key_len,) = struct.unpack("<Q", f.read(8))
                f.seek(key_len, 1)
                (vtype,) = struct.unpack("<I", f.read(4))
                _gguf_skip_value(f, vtype)
            shapes: dict[str, tuple[int, ...]] = {}
            for _ in range(n_tensors):
                (name_len,) = struct.unpack("<Q", f.read(8))
                name = f.read(name_len).decode("utf-8", errors="replace")
                (n_dims,) = struct.unpack("<I", f.read(4))
                dims = struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims))
                f.seek(12, 1)  # type (uint32) + offset (uint64)
                shapes[name] = tuple(int(d) for d in dims)
            return shapes
    except (OSError, struct.error, UnicodeDecodeError, ValueError, MemoryError):
        return None


_LORA_SUFFIX_A = ".lora_a"
_LORA_SUFFIX_B = ".lora_b"


def _lora_shape_mismatch(model_path: Path, lora_path: Path) -> str | None:
    """LoRA の全ターゲットテンソルをモデル実形状と突合し、不一致理由を返す。

    ggml の行列は ne0=in_features, ne1=out_features で格納される。対象
    weight W (in, out) に対し lora_a は (in, r)、lora_b は (r, out) なので、
    全ターゲットについて ``lora_a.ne0 == W.ne0`` かつ ``lora_b.ne1 == W.ne1``
    を要求する。ターゲットがモデルに存在しない (block 数の少ないモデルへ
    深い層の adapter を当てる等) 場合も不一致。hidden size 違いだけでなく
    head 構成違い (out_features) も検出できる。

    ``None`` は「不一致の証拠なし」— 全ターゲット照合済みで一致したか、
    判定不能 (どちらかのテンソル情報が読めない / LoRA テンソルが無い) かの
    いずれか。判定不能を fail-open にする理由は
    :func:`lora_compatible_with_model` の docstring 参照。
    """
    lora_shapes = read_gguf_tensor_shapes(lora_path)
    if not lora_shapes:
        return None
    targets: dict[str, dict[str, tuple[int, ...]]] = {}
    for name, dims in lora_shapes.items():
        if name.endswith(_LORA_SUFFIX_A):
            targets.setdefault(name[: -len(_LORA_SUFFIX_A)], {})["a"] = dims
        elif name.endswith(_LORA_SUFFIX_B):
            targets.setdefault(name[: -len(_LORA_SUFFIX_B)], {})["b"] = dims
    if not targets:
        return None
    model_shapes = read_gguf_tensor_shapes(model_path)
    if model_shapes is None:
        return None
    for target, ab in sorted(targets.items()):
        model_dims = model_shapes.get(target)
        if model_dims is None:
            return f"lora target tensor not in model: {target}"
        a = ab.get("a")
        b = ab.get("b")
        if a and model_dims and a[0] != model_dims[0]:
            return (
                f"in_features mismatch at {target} "
                f"(lora_a={a[0]}, model={model_dims[0]})"
            )
        if b and len(b) >= 2 and len(model_dims) >= 2 and b[-1] != model_dims[-1]:
            return (
                f"out_features mismatch at {target} "
                f"(lora_b={b[-1]}, model={model_dims[-1]})"
            )
    return None


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
            resolve_context_size_for(cfg, "base", project_root),
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
                resolve_context_size_for(cfg, "assist", project_root),
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


# config 明示も arch プロファイル宣言も無い場合の context_size slot 別既定。
# backend/config.py::_CONTEXT_SIZE_FALLBACK と一致させること (サーバ ``-c`` と
# ランタイム token budget の値を揃えるため)。
_CONTEXT_SIZE_DEFAULTS: dict[str, int] = {
    "base": 8192, "assist": 8192, "embed": 8192,
}


def _profile_context_size_for_model(
    model_path: Path, project_root: Path,
) -> int | None:
    """モデルパスから arch プロファイルの ``context_size`` を返す。

    GGUF arch 検出 → ``load_model_profile`` で解決。未宣言 / 512 未満 / 不正値 /
    読取失敗はすべて ``None`` (呼び出し側で slot 別既定にフォールバック)。
    """
    try:
        meta = read_gguf_metadata(model_path)
        profile = load_model_profile(meta.get("architecture"), project_root)
        raw = profile.get("context_size")
        if raw is None:
            return None
        value = int(raw)
        return value if value >= 512 else None
    except Exception:  # noqa: BLE001
        return None


def resolve_context_size_for(
    cfg: dict, name: str, project_root: Path | None = None,
    *, model_override: str | None = None,
) -> int:
    """slot の ``-c`` (context_size) を解決する。

    優先順位 (docs/c_15、profile=arch 既定): config 明示 > arch プロファイル
    ``context_size`` > slot 別既定 (``_CONTEXT_SIZE_DEFAULTS``)。config 側
    (``llama.context_size`` / ``assist_model.local.context_size``) が ``None``
    (未指定) のときのみ profile を参照する。プロファイル参照は base / assist の
    み、かつ ``project_root`` 指定時のみ (未指定なら config + 既定で解決)。
    embed は profile 非対象 (従来挙動)。

    ``model_override`` 指定時 (例: /api/mode/switch で coding_model に差し替えて
    base を再起動する経路) は、その実モデルの profile から ``context_size`` を
    引く。これにより ``-m`` で渡すモデルと ``-c`` が一致する (旧実装は常に
    base_model profile から ``-c`` を引いていた)。``llama.context_size`` の明示は
    手動 pin として override より優先する。
    """
    default = _CONTEXT_SIZE_DEFAULTS.get(name, 8192)
    if name == "base":
        explicit = (cfg.get("llama") or {}).get("context_size")
        model_key = "base_model"
    elif name == "assist":
        explicit = ((cfg.get("assist_model") or {}).get("local") or {}).get(
            "context_size",
        )
        model_key = "assist_model"
    elif name == "embed":
        return int((cfg.get("embedding") or {}).get("context_size", default))
    else:
        return default

    if explicit is not None:
        return int(explicit)
    if project_root is not None:
        model_rel = model_override or (cfg.get("model_paths") or {}).get(model_key, "")
        if model_rel:
            model_path = Path(model_rel)
            if not model_path.is_absolute():
                model_path = project_root / model_path
            profile_ctx = _profile_context_size_for_model(model_path, project_root)
            if profile_ctx is not None:
                return profile_ctx
    return default


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
    base_ctx = resolve_context_size_for(cfg, "base", project_root)
    assist_ctx = resolve_context_size_for(cfg, "assist", project_root)
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
    project_root: Path | None = None,
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
        ctx = resolve_context_size_for(cfg, name, project_root)
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

    return _try_estimate_via_fit_params(cfg, base_result, project_root)


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
    for name in ("base", "assist", "embed"):
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
            "embedding.gpu_layers to 0 (CPU fallback) "
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


def _props_model_matches(host: str, port: int, expected_model_id: str) -> bool:
    """``/props`` の load 済みモデルが ``expected_model_id`` と一致するか。

    モデル切替の再起動で、旧サーバが生き残って ``/health`` 200 を返す/新サーバが
    まだ別モデルをロード中、といったケースを弾くための同一性検証。``/props`` の
    ``model_alias`` → ``model_path`` → ``default_generation_settings.model`` の順で
    load 済みモデルを引き、basename で比較する (alias がフルパス/別名のことがある)。
    """
    try:
        resp = httpx.get(f"http://{host}:{port}/props", timeout=2.0)
        if resp.status_code != 200:
            return False
        props = resp.json()
    except (httpx.ConnectError, httpx.TimeoutException, ValueError):
        return False
    loaded = props.get("model_alias") or props.get("model_path") or ""
    if not loaded:
        gen = props.get("default_generation_settings")
        if isinstance(gen, dict):
            loaded = gen.get("model", "")
    if not loaded:
        return False
    return Path(str(loaded)).name == Path(expected_model_id).name


def wait_for_health(
    host: str, port: int, timeout: int = 30,
    expected_model_id: str | None = None,
) -> bool:
    """llama-server のヘルスチェックをポーリング。

    ``expected_model_id`` を渡すと ``/health`` 200 に加えて ``/props`` の load 済み
    モデルが一致するまで待つ。モデル切替の再起動で、旧サーバの 200 や新サーバの
    ロード途中を成功と誤判定しないための同一性検証 (``None`` なら従来挙動)。
    """
    url = f"http://{host}:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = httpx.get(url, timeout=2.0)
            if resp.status_code == 200:
                if expected_model_id is None:
                    return True
                if _props_model_matches(host, port, expected_model_id):
                    return True
        except (httpx.ConnectError, httpx.TimeoutException):
            pass
        time.sleep(1.0)
    return False


def _extract_model_basename(cmd: list[str]) -> str | None:
    """``cmd`` 中の ``-m`` 引数の basename を返す

    ``build_llama_cmd`` / ``build_assist_cmd`` / ``build_embed_cmd`` はいずれも
    ``"-m", str(model_path)`` の並びでモデルパスを積むため、実際に spawn される
    値から抽出する (cfg の再読込による二重管理を避ける)。
    """
    if "-m" not in cmd:
        return None
    return Path(cmd[cmd.index("-m") + 1]).name


def _start_and_wait(
    cmd: list[str], name: str, host: str, port: int,
    *, env: dict | None = None, cwd: str | Path | None = None,
    expected_model_id: str | None = None, timeout: int = 30,
) -> subprocess.Popen:
    """llama-server プロセスを起動しヘルスチェック

    ``cwd`` を渡すと相対パス引数 (例: --control-vector-scaled の相対 FNAME) が
    project_root 基準で解決される。canonical 起動 (service_manager) は cwd 設定済で、
    本 standalone 経路でも揃える。``None`` は親プロセスの CWD を継承 (従来挙動)。

    ``expected_model_id`` を渡すと ``wait_for_health`` が ``/props`` の load 済み
    モデルとの同一性まで検証する。port 占有中の旧プロセスが応答して誤って
    ready 判定されるのを防ぐ。
    """
    print(f"[launch] {name}: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, env=env, cwd=cwd)

    # Popen 直後の即死検知 (port bind 失敗等)。wait_for_health のフルタイムアウト
    # 待ちを避け、旧プロセスが port を握ったまま新プロセスが起動失敗したケースを
    # 早期に切り分ける。
    time.sleep(0.5)
    if proc.poll() is not None:
        print(
            f"[launch] ERROR: {name} exited immediately "
            f"(exit code={proc.returncode}). Likely port bind conflict or "
            "invalid model path.",
            file=sys.stderr,
        )
        return proc

    print(f"[launch] Waiting for {name} at {host}:{port}...")
    if wait_for_health(host, port, timeout, expected_model_id):
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

    launch_base = not args.embed and not args.assist
    launch_embed = args.embed or args.all
    launch_assist = args.assist or args.all

    # llama-server バージョン検査。--all / 個別起動を問わず
    # llama-server バイナリを起動する全パスで一度だけ build 番号を確認する。
    # ベース / アシスト / 埋め込みは同一バイナリを使うため、
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
    # 個別起動 (--embed / --assist) では既存プロセスとの
    # 合算が読み取れないため検査をスキップする。
    if args.all:
        ok, _total, _budget, _estimates, message = check_vram_budget(
            cfg, project_root, force=args.force,
        )
        print(message)
        if not ok:
            sys.exit(2)

    # モデルロードに時間がかかる大型 GGUF を考慮し、health タイムアウトは
    # process_manager.health_timeout (mode.py の base 切替と同じ既定値) を使う。
    health_timeout = int((cfg.get("process_manager") or {}).get("health_timeout", 120))

    try:
        # ベースモデル
        if launch_base:
            cmd = build_llama_cmd(cfg, project_root)
            host = cfg["llama"].get("host", "localhost")
            port = cfg["llama"].get("port", 8080)
            procs.append(_start_and_wait(
                cmd, "base-model", host, port, cwd=project_root,
                expected_model_id=_extract_model_basename(cmd), timeout=health_timeout,
            ))

        # アシストモデル
        if launch_assist:
            assist_cmd = build_assist_cmd(cfg, project_root)
            if assist_cmd:
                assist_cfg = cfg.get("assist_model", {}).get("local", {})
                host = assist_cfg.get("host", "127.0.0.1")
                port = assist_cfg.get("port", 8081)
                procs.append(_start_and_wait(
                    assist_cmd, "assist-model", host, port, cwd=project_root,
                    expected_model_id=_extract_model_basename(assist_cmd), timeout=health_timeout,
                ))
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
                procs.append(_start_and_wait(
                    embed_cmd, "embedding", host, port, cwd=project_root,
                    expected_model_id=_extract_model_basename(embed_cmd), timeout=health_timeout,
                ))
            else:
                print("[launch] Embedding backend is not llama-cpp, skipping")

        if procs:
            for proc in procs:
                proc.wait()
    except KeyboardInterrupt:
        for proc in procs:
            proc.terminate()
        for proc in procs:
            proc.wait()
