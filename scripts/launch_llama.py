"""llama-server 起動スクリプト

config.yaml の llama / embedding セクションから起動コマンドを組み立て、
サブプロセスとして llama-server を起動する。

サブコマンド:
  (なし)     ベースモデル llama-server のみ起動
  --all      ベース + エンベッドを一括起動
  --embed    エンベッド用 llama-server のみ起動
  --rerank-selftest  リランカーの自己テストを手動で測り直す (c_16 §7.2.1)
  --embed-placement  埋め込みサーバの配置 (GPU / CPU) を手動で判別し直す (c_16 §7.2.2)

  --all では ``rag.rerank.mode`` が off 以外なら rerank 用 llama-server (既定 :8083) も
  起動する。自己テストは PC の指紋が保存結果と違うときだけ走らせる。

  --all 起動時に ``runtime.total_vram_budget_mb`` (config.yaml) を参照し、
  GPU オフロード対象モデルの VRAM 使用量推定合計が予算を超過する場合は
  警告してアボートする。``--force`` で強制起動可能。
  埋め込みの ``-ngl`` は ``embedding.gpu_layers`` で、``null`` は 0 (CPU)、整数はそのまま、
  ``auto`` は PC の指紋が変わったときだけ一時ポートで CPU と GPU を測って決める (c_16 §7.2.2)。

  起動時に ``llama-server --version`` を実行し build 番号をログに出力する。
  ``runtime.min_llamacpp_build`` 未満を検出すると stderr に警告を出し、
  ``runtime.enforce_min_llamacpp_build: true`` の場合は exit code 3 で
  アボートする。バイナリが build 番号を露出しないカスタムビルドの場合は
  検出失敗 (None) として警告のみで継続。

  llama.cpp
  slot 退避機構 (``--cache-ram`` / ``--cache-idle-slots``) を全サーバ
  (base / embed) で明示制御する。``cache_ram_mib`` /
  ``cache_idle_slots`` を ``config.yaml`` から駆動し、上流のデフォルト
  (8192 MiB / true) を黙従する状態を解消する。``slots > 1`` かつ
  ``cache_ram_mib > 0`` の場合は idle slot offload が動作するよう
  ``--kv-unified`` を自動付与する (``kv_unified`` で明示 override 可)。
  base ではコンテキスト checkpoint (``--ctx-checkpoints`` /
  ``--checkpoint-min-step``) も ``ctx_checkpoints`` /
  ``checkpoint_min_step`` から常に明示付与する (hybrid recurrent モデルで
  prompt 分岐時の全量 re-prefill を避けるため、上流既定 8192 に黙従しない)。

  バイナリの ``-fitp on`` フラグを使い、device 別の使用 MiB (model /
  context / compute) を取得して VRAM 推定を 2 段構え化する。Tier 1 が
  バイナリ未存在 / タイムアウトで失敗した場合は GGUF ファイルサイズ
  ベースの Tier 2 (旧来挙動) にサイレントフォールバックする。
  ``runtime.fit_params_enabled: false`` で Tier 1 を一律スキップ可能。
  ``runtime.total_vram_budget_mb`` 未設定時は Tier 1 結果から 10%
  ヘッドルームを足した推奨予算を起動ログに表示する。

  ``--reasoning-budget`` / ``--reasoning-budget-message`` を
  各 build_*_cmd から付与し、thinking モデル (Qwen3 / Gemma-4 等)
  を補助タスク用途で使用する際の token 浪費を OS-fence で抑止する。
  従来の per-request ``chat_template_kwargs.enable_thinking=false``
  との二重防御で、サーバ側 fence 失敗時にもフォールバック可能。
  self-speculative decoding (``--spec-default`` / ``--spec-type ngram-*``) を
  Pro 限定で有効化する。``evoref create`` モード
  の base モデルが対象。``EVOREF_EDITION`` 環境変数 + ``backend.pro`` パッケー
  ジ存在で Pro 判定し、Free 環境では ``llama.speculative.enabled=true`` でも
  warning + 無効化する。embed は対象外 (Issue 文 §スコープ)。
"""

import contextlib
import importlib
import os
import re
import struct
import subprocess
import sys
import time
import types
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx
import yaml


def _resolve_embed_gpu_layers(cfg: dict, project_root: Path | None = None) -> int:
    """埋め込み用 ``-ngl`` の解決 (判別はしない。保存結果を読むだけ、c_16 §7.2.2)。

    ``embedding.gpu_layers`` が整数ならそれを、``null`` (未指定) なら CPU の 0 を返す。
    ``auto`` は保存済みの判別結果 (:func:`saved_embed_placement`) の配置で、無い・PC が違えば 0。
    ベースモデル ``llama.gpu_layers`` には追従しない。
    """
    return saved_embed_placement(cfg, project_root).gpu_layers


def _import_canonical() -> bool:
    """正規の ``scripts.launch_llama`` を import できたか (``__main__`` では試さない — 起動の挙動を変えない)。"""
    if __name__ == "__main__":
        return False
    try:
        importlib.import_module("scripts.launch_llama")
    except ImportError:
        return False
    return True


def _tuned(cfg: dict, project_root: Path, key: str):
    """環境調整の項目 ``key`` の起動時の値 (``resolve_tuned``、c_16 §7.2.3)。backend が無ければ ``None``。

    明示値はそのまま、``auto`` は保存済みの調整結果 (この PC) → その場の見積り (保存する) → 保守側の値。
    環境移行の確認待ちの間の ctx / ngl / VRAM 予算は、保存しない一時の見積り
    (``base_model.resolve_or_provisional``。起動スクリプトは ``--list-devices`` を読めるので)。
    """
    # ``python scripts/launch_llama.py`` で起動されたとき (``__main__``) は ``scripts.launch_llama`` を
    # import できない。環境調整はその名前で GGUF / ``--list-devices`` の実装を引くので、自分自身を登録しておく。
    # backend がファイルから読み込んだ複製 (``_launch_llama*``) は、正規の ``scripts.launch_llama`` を
    # import できるならそれを使い、複製を同じ名前で登録しない — 同名のモジュールが 2 つになり、正規の
    # モジュールへの差し替え (``monkeypatch.setattr("scripts.launch_llama.…")``) と ``from scripts.launch_llama
    # import …`` が別のものを指した (並列の全体回帰で test_model_migration が揺れた)
    if "scripts.launch_llama" not in sys.modules and not _import_canonical():
        me = sys.modules.get(__name__)
        if me is None:  # spec_from_file_location で読んで sys.modules に載せていない複製
            me = types.ModuleType("scripts.launch_llama")
            me.__dict__.update(globals())
        sys.modules["scripts.launch_llama"] = me
    try:
        from backend.free.core.tuning.base_model import resolve_or_provisional
    except ImportError:
        return None
    # 予約 (稼働中の画面の実行) の再計算は base が止まっているときだけ (載っている間は空きが過小)
    return resolve_or_provisional(cfg, key, project_root=project_root, can_recompute=lambda: not _base_ready(cfg))


def _tuned_effective(cfg: dict, project_root: Path, name: str):
    """キーごとに反映する環境調整の項目 (``threads`` / ``ram_params`` / ``batch``) の実効値。

    項目モジュールの ``effective_*`` が明示値と調整値を合わせる (明示値はそのまま)。全キーが明示なら
    調整しない。backend が無ければ ``None`` (呼び手は config の値と従来の既定で起動する)。
    """
    from importlib import import_module

    try:
        mod = import_module(f"backend.free.core.tuning.tuners.{name}")
    except ImportError:
        return None
    effective = {"threads": "effective_threads", "ram_params": "effective_ram_params", "batch": "effective_batch"}[name]
    tuned = None
    if len(mod.manual_fields(cfg)) < len(mod.CONFIG_KEYS):
        tuned = _tuned(cfg, project_root, mod.KEY)
    return getattr(mod, effective)(cfg, tuned)


#: backend が無い配布形態で ``llama.cache_ram_mib`` が ``auto`` / 無しのときの値。正本は
#: ``backend/free/core/tuning/tuners/ram_params.py`` の ``FALLBACK`` (0 = RAM を食わない側。テストが一致を見る)。
_UNTUNED_CACHE_RAM_MIB = 0


def _explicit_int(value: object) -> int | None:
    """config の値が明示の整数ならその値 (``auto`` / ``null`` / bool は ``None``)。"""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _aux_threads(cfg: dict, project_root: Path, part: str, *, gpu: bool) -> int:
    """埋め込み (``embed``) / リランカー (``rerank``) を ``gpu`` 配置で起こすときの ``-t`` (0 は付けない)。

    明示の正の整数はそのまま。``0`` は環境調整の項目 threads の配分 (c_16 §7.2.3)。
    """
    section = (cfg.get("embedding") or {}) if part == "embed" else _rerank_cfg(cfg)
    plan = _tuned_effective(cfg, project_root, "threads")
    if plan is None:
        return max(0, _explicit_int(section.get("threads")) or 0)
    return plan.aux(part, "gpu" if gpu else "cpu")


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
    ``cache_ram_mib: 0`` + ``slots > 1`` で per-seq context が黙って
    半減する footgun があったため、cache-ram から切り離した。
    """
    explicit = section_cfg.get("kv_unified")
    if explicit is None:
        return slots > 1
    return bool(explicit)


def _append_cache_ram_args(
    cmd: list[str], section_cfg: dict, *, cache_ram_mib: int,
) -> None:
    """``--cache-ram`` / ``--no-cache-idle-slots`` を ``cmd`` に追加する
    (chat slots を持つ base 用)。

    ``cache_ram_mib`` (解決済み) は常に明示付与し、上流デフォルト 8192 の黙従を
    回避する。``cache_idle_slots: false`` のときのみ ``--no-cache-idle-slots``
    を付与する (上流デフォルト true は何も付けない)。
    """
    cmd += ["--cache-ram", str(int(cache_ram_mib))]
    if not _resolve_cache_idle_slots(section_cfg, default=True):
        cmd += ["--no-cache-idle-slots"]


# ── slots (auto) 解決 ─────────────────────────────────
#: ``llama.slots: auto`` の床。0 = chat / 1 = background / 2 = classifier。
SLOTS_BASE = 3
#: 4 本目 = long_form のユニット生成専用 (``LocalClient.longform_slot``)。
SLOTS_WITH_LONG_FORM = 4
#: auto で 4 本目を切る最小 context_size。
#:
#: unified KV (slots>1 で自動付与) ではスロットは VRAM を分け合わず、**同じ n_ctx
#: セル** を分け合う (KV 本体は n_ctx 総量で頭打ち、増えるのは hybrid arch の
#: 再帰状態だけ)。llama-server はセルが尽きると idle スロットを id 順に 1 本ずつ
#: purge する (server-context.cpp ``try_clear_idle_slots``) ので、最初に消えるのは
#: チャット接頭辞 (slot 0)。4 本目が効くのは、チャットのプロンプト (working
#: 4352 + system + 注入 ≈ 6〜7K)、ユニット 1 本 (prompt + unit_max_tokens 2000
#: ≈ 3.5〜4K)、背景 / 分類器の残滓 (≈ 2〜3K) が同時に n_ctx に収まるときだけ。
#: 8192 では収まらず (purge で今と同じ挙動になるだけ)、16384 以上で収まる。
LONG_FORM_SLOT_MIN_CTX = 16384
_RESOLVED_SLOTS_KEY = "__resolved_slots__"


def _per_seq_state_mb(
    cfg: dict, project_root: Path, n_ctx: int, model_override: str | None,
) -> int:
    """スロット 1 本ぶんの文脈メモリ増分 (MiB)。

    unified KV では KV 本体は増えず、hybrid arch (Qwen3.5/3.8 等) の再帰状態
    だけがシーケンス毎に確保される (純 attention モデルなら 0)。GGUF が無い /
    メタデータ不足なら 0 (= 判定に効かせない)。
    """
    sp = cfg.get("model_paths", {}) or {}
    path = Path(model_override or sp.get("base_model", ""))
    if not str(path):
        return 0
    if not path.is_absolute():
        path = project_root / path
    if not path.exists():
        return 0
    lc = cfg.get("llama", {}) or {}
    meta = _read_gguf_metadata_cached(path)
    with_lf = estimate_kv_cache_mb(
        meta, n_ctx, lc.get("cache_type_k"), lc.get("cache_type_v"),
        n_seq=SLOTS_WITH_LONG_FORM,
    )
    base = estimate_kv_cache_mb(
        meta, n_ctx, lc.get("cache_type_k"), lc.get("cache_type_v"),
        n_seq=SLOTS_BASE,
    )
    if with_lf is None or base is None:
        return 0
    return max(0, int(with_lf) - int(base))


def resolve_base_slots(
    cfg: dict, project_root: Path | None = None, *, model_override: str | None = None,
) -> int:
    """``llama.slots`` を実数に解決する (``auto`` は 3 か 4)。

    解決結果は ``cfg`` に 1 度だけキャッシュし、``-np`` / VRAM 推定 (再帰状態の
    シーケンス数) / 予算超過時の降格 (:func:`check_vram_budget`) が同じ値を見る。
    backend 側は config を解釈せず、llama-server ``/props`` の ``total_slots``
    を実数として受け取る (``client_builder.build_local_client``)。
    """
    key = model_override or ""
    cache = cfg.setdefault(_RESOLVED_SLOTS_KEY, {})
    if key in cache:
        return int(cache[key]["slots"])
    lc = cfg.get("llama", {}) or {}
    raw = lc.get("slots", "auto")
    if raw != "auto":
        slots = max(1, int(raw or 1))
        cache[key] = {"slots": slots, "auto": False, "per_seq_mb": 0}
        return slots
    project_root = project_root or Path.cwd()
    n_ctx = resolve_context_size_for(
        cfg, "base", project_root, model_override=model_override,
    )
    per_seq_mb = _per_seq_state_mb(cfg, project_root, n_ctx, model_override)
    if n_ctx < LONG_FORM_SLOT_MIN_CTX:
        slots = SLOTS_BASE
        reason = (
            f"n_ctx {n_ctx} < {LONG_FORM_SLOT_MIN_CTX}: a long_form slot could not "
            "keep the chat prefix resident (unified KV shares the cells)"
        )
    else:
        slots = SLOTS_WITH_LONG_FORM
        reason = (
            f"n_ctx {n_ctx} >= {LONG_FORM_SLOT_MIN_CTX}: dedicated long_form slot "
            f"(+{per_seq_mb} MiB per-sequence state)"
        )
    cache[key] = {"slots": slots, "auto": True, "per_seq_mb": per_seq_mb}
    print(f"[launch] llama.slots=auto -> {slots} ({reason})")
    return slots


def _append_kv_unified_args(
    cmd: list[str], section_cfg: dict, *, slots: int,
) -> None:
    """``--kv-unified`` を必要に応じて ``cmd`` に追加する

    ``slots > 1`` の場合に自動付与する (``kv_unified`` の明示指定で上書き可)。
    """
    if _resolve_kv_unified(section_cfg, slots=slots):
        cmd += ["--kv-unified"]


# ── reasoning 解決 (プロファイル由来) ─────────────────────


def _profile_reasoning_for_model(model_path: Path, project_root: Path) -> dict:
    """モデルパスからプロファイルの ``reasoning`` セクション (raw dict) を返す。

    ``load_model_profile_for`` で解決 (arch 層 + モデル別層)。失敗 / 不在時は ``{}``。
    検証は backend 側 ``_normalize_profile`` (ProfileReasoningConfig) で行うため
    ここでは raw を返す (docs/c_15)。
    """
    try:
        profile = load_model_profile_for(model_path, project_root)
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

# 上流 llama.cpp は 2026-05 に speculative 系フラグを mode 別へ分割し、総称名
# (``--draft-max`` / ``--draft-min`` / ``--spec-ngram-size-n`` /
# ``--spec-ngram-size-m``) を **削除** した。旧名を渡すと llama-server は
# "the argument has been removed" で **exit 1 になり起動しない**ため、mode から
# 実フラグを引く。``-cd`` (draft 文脈長) は代替フラグごと消滅している。
# 対応は installed llama-server の ``--help`` で実測して決めた (2026-08-15)。

# ``--spec-ngram-mod-*`` を使う mode。``default`` (= ``--spec-default``) は上流
# プリセットが ngram-mod ベース (n-match 24 / n-min 48 / n-max 64) のため同じ扱い。
_NGRAM_MOD_MODES: frozenset[str] = frozenset({"default", "ngram-mod"})

# ``--spec-<mode>-size-n`` / ``-size-m`` を持つ mode (mode 名がそのままフラグに入る)。
# ``ngram-cache`` は専用パラメータを持たない。
_NGRAM_SIZED_MODES: frozenset[str] = frozenset(
    {"ngram-simple", "ngram-map-k", "ngram-map-k4v"}
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


def _data_path(cfg: dict, project_root: Path, key: str) -> Path | None:
    """データ根 (c_05 §0.2) 配下の ``key`` を ``PathResolver`` で解決する。

    データ根は ``EVOREF_DATA_ROOT`` (``--data-root`` を反映) →
    ``<project_root>/userdata``。``local_paths`` の個別キーは撤去済みなので読まない。
    ``backend`` を import できない単体実行 (editable install 前) ではデータ根が
    決まらないため ``None`` (データ側のアダプタ / override は無いものとして扱う)。
    """
    try:
        from backend.config import PathResolver
    except ImportError:
        return None
    return PathResolver(cfg, project_root).resolve_local(key)


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
      ``draft_max`` / ``draft_min`` / ``draft_p_min`` / ``ngram_size_n`` を
      付加すれば上書き可能。
    - ``"ngram-mod"`` / ``"ngram-cache"`` / ``"ngram-simple"`` /
      ``"ngram-map-k"`` / ``"ngram-map-k4v"``: ``--spec-type <mode>`` +
      明示パラメータ。
    - ``"draft-model"``: ``-md <path>`` + ``-ngld`` (任意) +
      ``--spec-draft-n-max`` / ``--spec-draft-n-min``。

    パラメータ → 実フラグの対応は mode 依存 (``_NGRAM_MOD_MODES`` /
    ``_NGRAM_SIZED_MODES`` のコメント参照)。対応フラグを持たない組み合わせ
    (``ngram_size_m`` × ngram-mod 系 / ``ngram-cache`` の n・m / ``ctx_size_draft``)
    は warning を出して落とす (graceful degrade。起動は止めない)。

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

    # draft 文脈長 (旧 ``-cd``) は上流から代替フラグごと消滅した。
    if int(ctx_size_draft) > 0 and warn is not None:
        warn(
            "[launch] WARNING: llama.speculative.ctx_size_draft is ignored; "
            "upstream llama-server no longer accepts -cd / --ctx-size-draft."
        )

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
        ngld = spec_cfg.get("gpu_layers_draft")
        if ngld is not None:
            ngld_str = str(ngld).strip()
            if ngld_str:
                args += ["-ngld", ngld_str]
    else:
        # ngram-* 系
        args += ["--spec-type", mode]

    # draft トークン数の上下限。ngram-mod 系は lookup 長の上下限
    # (``--spec-ngram-mod-n-min/-n-max``)、それ以外は draft 長そのもの
    # (``--spec-draft-n-min/-n-max``)。旧総称名 ``--draft-max`` /
    # ``--draft-min`` は上流で削除済み。
    if mode in _NGRAM_MOD_MODES:
        max_flag, min_flag = "--spec-ngram-mod-n-max", "--spec-ngram-mod-n-min"
    else:
        max_flag, min_flag = "--spec-draft-n-max", "--spec-draft-n-min"
    if draft_max is not None:
        args += [max_flag, str(int(draft_max))]
    if draft_min is not None:
        args += [min_flag, str(int(draft_min))]
    # ``--draft-p-min`` は ``--spec-draft-p-min`` の alias として上流に現存する。
    if draft_p_min is not None:
        args += ["--draft-p-min", str(float(draft_p_min))]

    # ngram 系パラメータ。``--spec-ngram-size-n`` / ``-size-m`` は上流で削除済みの
    # ため mode 別フラグへ振り分ける。``default`` は上流プリセット
    # (n-match 24 / n-min 48 / n-max 64) に従わせ、明示指定がある場合のみ上書き。
    if mode in _NGRAM_MOD_MODES:
        if ngram_n is not None:
            args += ["--spec-ngram-mod-n-match", str(int(ngram_n))]
        if ngram_m is not None and warn is not None:
            warn(
                "[launch] WARNING: llama.speculative.ngram_size_m is ignored for "
                f"mode={mode!r}; upstream ngram-mod has no draft-length parameter "
                "(only --spec-ngram-mod-n-match / -n-min / -n-max)."
            )
    elif mode in _NGRAM_SIZED_MODES:
        if ngram_n is not None:
            args += [f"--spec-{mode}-size-n", str(int(ngram_n))]
        if ngram_m is not None:
            args += [f"--spec-{mode}-size-m", str(int(ngram_m))]
    elif mode == "ngram-cache" and warn is not None:
        if ngram_n is not None or ngram_m is not None:
            warn(
                "[launch] WARNING: llama.speculative.ngram_size_n / ngram_size_m "
                "are ignored for mode='ngram-cache'; upstream has no "
                "--spec-ngram-cache-* parameters."
            )

    return args


def _warn_cache_reuse_inert(
    meta: dict,
    cache_reuse: int,
    *,
    warn: Callable[[str], None] | None = None,
) -> bool:
    """``cache_reuse`` が上流で無効化される構成なら warning を出す。

    hybrid recurrent モデル (線形 attention 層を持つ Qwen3.5/3.8 等) では
    llama-server が起動時に ``cache_reuse is not supported by this context,
    it will be disabled`` を出して機構ごと無効化する。フラグ付与自体は無害
    だが、**config に値が入っているのに効いていない**ことが evoref 側の
    ログからは分からず、遅延調査で「prefix 再利用しているはず」という誤った
    前提を置く原因になる (2026-08-15 ライブ監査)。

    判定は GGUF の recurrent メタデータ (``ssm.*`` / ``full_attention_interval``)
    のみで行う。SWA (gemma-4 等) も上流で無効化されるが GGUF から確実に
    判別できないため対象外とし、既知の事実としてコメントに留める。

    Returns:
        warning を出したかどうか (テスト用)。
    """
    if cache_reuse <= 0:
        return False
    interval = int(meta.get("full_attention_interval") or 0)
    is_recurrent = bool(meta.get("ssm_state_size")) or interval > 1
    if not is_recurrent:
        return False
    if warn is not None:
        warn(
            f"[launch] WARNING: llama.cache_reuse={cache_reuse} has no effect on "
            "this model: it has recurrent (linear-attention) layers and "
            "llama-server disables cache_reuse for such contexts."
        )
    return True


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
        model_cfg: base なら ``cfg["llama"]`` の dict。``mtp`` サブ dict を読む。
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
    3. 系統 (lineage) チェック: アダプタに ``evoref.trained_on_model_key``
       (学習元モデルの ``model_key``、c_05 §0.5.7。Level 2 トレーナーが刻む)
       がある場合、現モデルの ``model_key`` との一致を要求する。
       LoRA は特定の重みへの差分なので、同一 arch・同一形状でも別モデル
       (例: Qwen3.5-9B と同 arch の別 finetune) に当てると silent な品質
       摂動になる。model_key は重みの標本から導くので、ファイル名が同じ
       別モデルも、再量子化した同じモデルも別の系統になる (誤って捨てる
       実害は次の Level 2 サイクルで再学習される軽微なもの、誤って適用する
       実害は診断困難な品質摂動であるため)。

    形状が判定不能 (LoRA テンソル無し / どちらかのテンソル情報節が
    読めない) の場合、および stamp が無いレガシーアダプタの系統は
    fail-open (arch + 形状のみで判定)。arch 側の fail-closed との非対称は
    意図的 — テンソル命名の異なる正当なアダプタを誤って捨てないため。
    起動側 (``backend.pro.adapters``) と LoRA 一覧 API
    (``backend.pro.api.lora``) で共有する単一の述語。

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

    stamp = lora_meta.get("trained_on_model_key")
    if stamp:
        from backend.model_key import model_key_for

        model_key = model_key_for(model_path)
        if stamp != model_key:
            return False, (
                f"lineage mismatch (adapter trained on {stamp}, "
                f"model {model_path.name} is {model_key})"
            )
    return True, f"architecture={model_arch}"


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
    control_vector_override: str | Path | None = None,
    port_override: int | None = None,
    context_override: int | None = None,
    slots_override: int | None = None,
    no_warmup: bool = False,
) -> list[str]:
    """config.yaml の llama セクションから起動コマンドを生成

    Level 2 base=spsa-real-eval の候補評価では ``lora_override`` (候補 GGUF LoRA
    パス) と ``port_override`` (スクラッチポート) を指定して ephemeral サーバを
    起動する。いずれも未指定 (通常運用) の
    ときは従来挙動と完全に等価。

    通常運用のアダプタ (``lora_override`` / ``control_vector_override``) は
    呼出元が Pro のハンドラ ``adapter_paths_for_launch`` から得て渡す
    (``backend.free.core.launch_adapters``)。このモジュールは config から
    アダプタを解決しない — ``None`` なら付けない (Free は常に無し)。

    ``context_override`` / ``slots_override`` / ``no_warmup`` は Level 2 候補評価
    のような **使い捨てサーバ** 用。通常運用の値 (base は既定 8192 × 2 slot) を
    そのまま継承すると、64 トークンの応答を数件取るためだけに本番同等の KV
    キャッシュを確保・解放する動作を候補ごと (1 サイクル数十回) 繰り返すことに
    なる。未指定 (通常運用) のときは従来挙動と完全に等価。
    """
    if "llama" not in cfg:
        raise ValueError("config.yaml に 'llama' セクションがありません")
    lc = cfg["llama"]
    sp = cfg.get("model_paths", {})

    if project_root is None:
        project_root = Path.cwd()

    # ベースモデルパス解決（model_override 指定時はそちらを優先）
    base_model = model_override or sp.get("base_model", "models/gemma-4-12b-it-qat-q4_0.gguf")
    base_model_path = Path(base_model)
    if not base_model_path.is_absolute():
        base_model_path = project_root / base_model_path

    port = port_override if port_override is not None else lc.get("port", 8080)

    # 環境調整の値 (ngl / b・ub / threads / RAM 系) は ``model_paths.base_model`` について決めたもの。
    # 別モデルの ``model_override`` (create_model 等) には持ち込まず、config の明示値と従来の既定で組む
    # (ctx は resolve_context_size_for がそのモデルの profile を引く)
    tuned_ok = _is_base_model(cfg, project_root, model_override)
    if tuned_ok:
        ngl = _resolve_base_gpu_layers(cfg, project_root)
    else:
        ngl = _explicit_int(lc.get("gpu_layers"))
        ngl = 999 if ngl is None else ngl  # auto / 無しは従来の既定 (全層)

    cmd = [
        "llama-server",
        "-m", str(base_model_path),
        "--port", str(port),
        "-c", str(context_override or resolve_context_size_for(
            cfg, "base", project_root, model_override=model_override,
        )),
        "-ngl", str(ngl),
    ]
    # -b / -ub: 明示値はそのまま、auto / 無しは環境調整の項目 batch (c_16 §7.2.3)
    batch = _tuned_effective(cfg, project_root, "batch") if tuned_ok else None
    if batch is not None:
        cmd += ["-b", str(batch.batch_size), "-ub", str(batch.ubatch_size)]
    else:
        cmd += ["-b", str(_explicit_int(lc.get("batch_size")) or 512)]
        if _explicit_int(lc.get("ubatch_size")):
            cmd += ["-ub", str(lc["ubatch_size"])]

    # LoRA アダプタ。呼出元が解決・検証済み (Level 2 候補評価の GGUF は harness が
    # 起動直前に書き出すため exists チェックしない)。
    if lora_override is not None:
        cmd += ["--lora", str(Path(lora_override))]

    # Level 2 base=C: control vector (残差ストリーム操舵)。呼出元が解決・検証済み。
    # scale / 層範囲は config (learning.cvector_*) から読む。
    if control_vector_override is not None:
        learning_cfg = cfg.get("learning", {}) or {}
        cvec_full = Path(control_vector_override)
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
                    "cvector_scale != 1.0 requires the control vector to "
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
    # -t / -tb: 明示の正の整数はそのまま (-tb は付けない)、0 は環境調整の項目 threads の配分 (c_16 §7.2.3)
    plan = _tuned_effective(cfg, project_root, "threads") if tuned_ok else None
    threads = plan.base if plan is not None else max(0, _explicit_int(lc.get("threads")) or 0)
    if threads > 0:
        cmd += ["-t", str(threads)]
    if plan is not None and plan.batch > 0:
        cmd += ["-tb", str(plan.batch)]
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
    slots = max(1, int(slots_override or resolve_base_slots(
        cfg, project_root, model_override=model_override,
    )))
    cmd += ["-np", str(slots)]
    if no_warmup:
        cmd += ["--no-warmup"]
    cache_type_k = lc.get("cache_type_k")
    if cache_type_k and cache_type_k != "f16":
        cmd += ["--cache-type-k", str(cache_type_k)]
    cache_type_v = lc.get("cache_type_v")
    if cache_type_v and cache_type_v != "f16":
        cmd += ["--cache-type-v", str(cache_type_v)]

    # 共通 prefix 自動再利用（0 は無効）。多ターン chat で system/RAG 接頭辞の
    # KV を再 prefill せず再利用する。
    # 注: SWA モデル (gemma-4 等) と hybrid recurrent モデル (Qwen3.5/3.8 等) では
    # llama.cpp が cache_reuse を自動無効化するため no-op (フラグ付与は無害)。
    # hybrid recurrent モデルでは上流が機構ごと無効化するので、フラグ自体を
    # 付けない (warning は残す)。「config に値があるのに効いていない」状態を
    # 起動コマンドの見た目で誤読させないため (2026-09-11)。
    cache_reuse = lc.get("cache_reuse", 0)
    if cache_reuse and int(cache_reuse) > 0:
        inert = _warn_cache_reuse_inert(
            _read_gguf_metadata_cached(base_model_path),
            int(cache_reuse),
            warn=lambda msg: print(msg, file=sys.stderr),
        )
        if not inert:
            # 明示値は尊重する (黙って上書きしない) が、実働する構成では警告する。
            print(
                f"[launch] WARNING: llama.cache_reuse={int(cache_reuse)} corrupted slot KV "
                "on a dense model in a 2026-09-27 A/B test (llama.cpp d834d44, "
                "Qwen2.5-Coder-14B: cached prompts returned only newlines while "
                "cache_prompt=false was fine; reuse=0 was clean). Set llama.cache_reuse: 0 "
                "unless you have verified it on this build (docs c_10 §4).",
                file=sys.stderr,
            )
            cmd += ["--cache-reuse", str(int(cache_reuse))]

    # コンテキスト checkpoint。hybrid recurrent モデルは部分巻き戻し不可で、
    # 上流既定 min-step 8192 だと n_ctx=8192 で中間 checkpoint が生まれず、
    # 前回の最終 user 位置より手前で prompt が分岐すると system prompt ごと
    # 全量 re-prefill になる。-np と同様に常に明示する (上流既定へ黙従しない)。
    # checkpoint 数と cache-ram は明示値はそのまま、auto / 無しは環境調整の項目 ram_params (空き RAM から)。
    ram = _tuned_effective(cfg, project_root, "ram_params") if tuned_ok else None
    checkpoints = ram.ctx_checkpoints if ram is not None else _explicit_int(lc.get("ctx_checkpoints"))
    cmd += ["--ctx-checkpoints", str(8 if checkpoints is None else checkpoints)]
    cmd += [
        "--checkpoint-min-step", str(int(lc.get("checkpoint_min_step", 256))),
    ]

    # idle slot offload / prompt cache の RAM 上限。slots>1 のときは
    # ``--cache-ram`` が動作するよう ``--kv-unified`` を自動付与する。
    cache_ram = ram.cache_ram_mib if ram is not None else _explicit_int(lc.get("cache_ram_mib"))
    _append_cache_ram_args(cmd, lc, cache_ram_mib=_UNTUNED_CACHE_RAM_MIB if cache_ram is None else cache_ram)
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
            "-m", "--port", "-c", "-ngl", "-b", "-ub", "-t", "-tb", "-fa", "--mlock",
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


def _embed_model_path(cfg: dict, project_root: Path) -> Path:
    """埋め込みモデルの絶対パス (``model_paths.embed_model`` or 既定、存在は見ない)。"""
    embed_model = (cfg.get("model_paths", {}) or {}).get(
        "embed_model", "models/Qwen3-Embedding-0.6B-Q8_0.gguf",
    )
    embed_model_path = Path(embed_model)
    if not embed_model_path.is_absolute():
        embed_model_path = project_root / embed_model_path
    return embed_model_path


def build_embed_cmd(cfg: dict, project_root: Path | None = None) -> list[str] | None:
    """config.yaml の embedding セクションから埋め込み用 llama-server コマンドを生成

    embedding.backend は llama-cpp のみサポート。未知の値は None を返す。
    ``-ngl`` は ``embedding.gpu_layers`` (``null`` → 0 / 整数 / ``auto`` は保存済みの判別結果、
    c_16 §7.2.2)。判別はしない — 判別して起動するのは持ち主のプロセスの :func:`start_embed_server`。
    GPU で起動できなかったときの CPU の再試行は :func:`embed_cpu_fallback_cmd`。
    """
    emb_cfg = cfg.get("embedding", {})
    if emb_cfg.get("backend", "llama-cpp") != "llama-cpp":
        return None

    if project_root is None:
        project_root = Path.cwd()

    gpu_layers = _resolve_embed_gpu_layers(cfg, project_root)
    return _embed_cmd(
        cfg, project_root, port=int(emb_cfg.get("llama_port", 8082)), gpu_layers=gpu_layers,
    )


def _embed_batch_sizes(cfg: dict, project_root: Path, *, cpu: bool = False) -> tuple[int, int]:
    """埋め込みの ``(-b, -ub)``。backend と同じ解決 (環境調整の項目 ``embed_params`` + 明示値、c_16 §7.2.3)。

    明示値が ``ubatch < max_length`` 等を破っていれば警告だけ出す。backend が無ければ明示値、
    ``auto`` / 未設定は ``max_length`` (不変条件の下限)。``cpu`` (CPU で起こす) のとき、GPU 配置を前提に
    VRAM の空きで広げた調整値は不変条件の下限 (``max_length``) に戻す (計算バッファを RAM に積まない)。
    調整値の理由が ``warn`` (計算バッファが空きに収まらない) なら、その提案を警告に出す。
    """
    emb_cfg = cfg.get("embedding", {}) or {}
    resolved = _tuned(cfg, project_root, "embed_params")
    if resolved is None:
        floor = max(int(emb_cfg.get("max_length", 8192)), 512)

        def pick(key: str) -> int:
            raw = emb_cfg.get(key)
            return floor if raw is None or raw == "auto" else int(raw)

        return pick("batch_size"), pick("ubatch_size")
    from backend.free.core.tuning.tuners.embed_params import effective_embed_params, ubatch_floor

    params = effective_embed_params(emb_cfg, resolved.value)
    for warning in params.warnings:
        print(f"[launch] WARNING: {warning}", file=sys.stderr)
    if "warn:" in (resolved.reason or ""):
        print(f"[launch] WARNING: embedding ubatch: {resolved.reason}", file=sys.stderr)
    batch, ubatch = params.batch_size, params.ubatch_size
    if cpu and isinstance(resolved.value, dict) and resolved.value.get("placement") == "gpu":
        if params.sources.get("ubatch_size") == "tuned":
            ubatch = min(ubatch, ubatch_floor(int(emb_cfg.get("max_length") or 8192)))
        if params.sources.get("batch_size") == "tuned":
            batch = ubatch
    return batch, ubatch


def _forget_tuned(cfg: dict, key: str) -> None:
    """1 回の起動の中で覚えた ``key`` の解決 (``resolve_tuned`` のメモ) を捨てる (前提が変わったとき)。"""
    try:
        from backend.free.core.tuning.resolve import CACHE_KEY
    except ImportError:
        return
    memo = cfg.get(CACHE_KEY)
    if isinstance(memo, dict):
        memo.pop(key, None)


def _embed_cmd(cfg: dict, project_root: Path, *, port: int, gpu_layers: int) -> list[str]:
    """埋め込み用 llama-server のコマンド (ポートと ``-ngl`` 以外は設定から。判別の一時サーバも同じ引数)。"""
    emb_cfg = cfg.get("embedding", {})
    embed_model_path = _embed_model_path(cfg, project_root)

    cmd = [
        "llama-server",
        "-m", str(embed_model_path),
        "--port", str(port),
        "--embedding",
        "-ngl", str(gpu_layers),
    ]

    # Pooling 方式。embedding.pooling が明示されている場合のみ --pooling を
    # 付与し、未設定なら llama-server のモデル既定 pooling に委ねる (既存
    # モデルの挙動を変えない)。BGE-M3 (arch "bert") は CLS pooling が正しく、
    # embed 切替時に models/profiles/bert.yaml から config.yaml へ自動転写
    # される (値は EmbeddingConfig 側で検証済みのためここでは素通しする)。
    #
    # config 未設定時のみプロファイルの embedding.pooling へフォールバックする。
    # migrate 経由の切替では転写済みなので発火せず、config.yaml の
    # model_paths.embed_model を手編集して差し替えた場合の安全網として効く
    # (転写と同じプロファイルを読むので値は分岐しない)。
    pooling = emb_cfg.get("pooling")
    if not pooling and (cfg.get("llama", {}) or {}).get("auto_model_flags", True):
        emb_profile = load_model_profile_for(embed_model_path, project_root).get(
            "embedding",
        )
        if isinstance(emb_profile, dict):
            pooling = emb_profile.get("pooling")
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

    # 物理バッチサイズ。長い STM ノート (>512 tok) の埋め込みリクエストが
    # llama-server デフォルト batch=512 のままだと 500 エラーで落ちるため、常に明示する
    # (auto は環境調整の値で ubatch >= max_length を保証、明示値はそのまま。c_16 §7.2.3)。
    batch_size, ubatch_size = _embed_batch_sizes(cfg, project_root, cpu=int(gpu_layers) == 0)
    cmd += ["-b", str(batch_size), "-ub", str(ubatch_size)]

    # idle slot offload は埋め込みでは無意味 (chat slots 不使用) なので
    # 既定 0 で明示 disable する。``cache_ram_mib`` を 0 以外に
    # 指定された場合のみ付与値を変更する。
    cache_ram_mib = _resolve_cache_ram_mib(emb_cfg, default=0)
    cmd += ["--cache-ram", str(cache_ram_mib)]

    # スレッド数。明示の正の整数はそのまま。0 (既定) は環境調整の項目 threads が物理コア数を
    # base と分け合った値 (GPU 配置は 2。決められなければ省略して llama.cpp の既定)。
    threads = _aux_threads(cfg, project_root, "embed", gpu=int(gpu_layers) != 0)
    if threads > 0:
        cmd += ["-t", str(threads)]

    # 並列スロットは常に -np で明示する (未指定だと n_parallel=auto=4 になり
    # slots × context_size の KV を無駄に確保する)。埋め込みは概ね逐次のため既定 2。
    slots = max(1, int(emb_cfg.get("slots", 2) or 2))
    cmd += ["-np", str(slots)]

    # slots > 1 では --kv-unified 無しだと per-seq context が n_ctx/slots に
    # 黙って分割される (base と同じ footgun、_resolve_kv_unified 参照)。
    _append_kv_unified_args(cmd, emb_cfg, slots=slots)

    return cmd


# ── rerank (リランカー、c_16 §7.2.1) ───────────────────────
# 実測 (japanese-bge-reranker-v2-m3 q8_0、890M iGPU): -ngl 999 -fa on -c 8192 -b 8192
# -ub 2048 -np 1 -t 2 --cache-ram 0 が最適。-ub は 1 組 (query + doc) の最大長以上が
# 必須 (512 では 500 エラー)。

RERANK_DEFAULT_PORT = 8083
#: GPU 配置の判定で、モデルサイズに足す計算バッファの見積り (MiB)。
#: 実測ではなく見積り (埋め込みと同じ値)。RAM 余裕の判定 (:func:`decide_rerank_memory`) と
#: VRAM の合算 (:func:`estimate_rerank_vram`) で共有する。
RERANK_COMPUTE_MARGIN_MIB = 1024
#: GPU 配置で threads=0 (自動) のときの -t (GPU が計算するので少なくてよい、実測最適)。
RERANK_GPU_DEFAULT_THREADS = 2
#: GPU 配置の本番サーバの health 待ち (秒)。超えたら CPU で起こし直す (埋め込みと同じ値)。
RERANK_GPU_START_TIMEOUT_SEC = 60
#: evoref-ctl が rerank の準備を待つとき、起動の待ちに足す自己テスト本体の余裕 (秒)。
RERANK_SELFTEST_ALLOWANCE_SEC = 60


@dataclass(frozen=True)
class RerankPlacement:
    """rerank サーバの配置。``kind`` は ``gpu`` / ``cpu``、``reason`` は判定の根拠 (英語)。"""

    kind: str
    gpu_layers: int
    reason: str = ""


def _rerank_cfg(cfg: dict) -> dict:
    return ((cfg.get("rag") or {}).get("rerank") or {})


def rerank_mode(cfg: dict) -> str:
    """``rag.rerank.mode`` (正規化は ``backend.schemas.rag.rerank_mode_of`` の 1 実装)。"""
    from backend.schemas.rag import rerank_mode_of

    return rerank_mode_of(cfg)


def rerank_port(cfg: dict) -> int:
    return int(_rerank_cfg(cfg).get("port", RERANK_DEFAULT_PORT))


def resolve_rerank_model_path(cfg: dict, project_root: Path) -> Path | None:
    """``model_paths.rerank_model`` の絶対パス。未設定なら ``None`` (存在は見ない)。"""
    raw = (cfg.get("model_paths") or {}).get("rerank_model")
    if not raw:
        return None
    path = Path(str(raw))
    return path if path.is_absolute() else project_root / path


def rerank_launchable(cfg: dict, project_root: Path) -> tuple[bool, str]:
    """rerank サーバを起動する条件 (mode != off・モデル設定済み・ファイルあり) と、起動しない理由。"""
    if rerank_mode(cfg) == "off":
        return False, "rag.rerank.mode is off"
    model = resolve_rerank_model_path(cfg, project_root)
    if model is None:
        return False, "model_paths.rerank_model is not set"
    if not model.is_file():
        return False, f"rerank model not found: {model}"
    return True, ""


@dataclass(frozen=True)
class GpuFit:
    """GPU に載るかの判定 (リランカーと埋め込みで共有する 1 実装)。

    ``status`` は ``no_gpu`` / ``unknown_size`` / ``fits`` / ``short``。
    """

    status: str
    device: str = ""
    free_mib: int = 0
    need_mib: int = 0


def gpu_fit(
    device_memory: dict[str, tuple[int, int]], model_mb: int | None, *, margin_mib: int, reserve_mib: int = 0,
) -> GpuFit:
    """GPU デバイス (``host`` 以外) の空きの最大が ``model_mb + margin_mib + reserve_mib`` 以上か (純関数)。"""
    gpus = {k: v for k, v in device_memory.items() if k.lower() != "host"}
    if not gpus:
        return GpuFit("no_gpu")
    if model_mb is None:
        return GpuFit("unknown_size")
    name, (_total, free) = max(gpus.items(), key=lambda kv: kv[1][1])
    need = model_mb + margin_mib + max(0, reserve_mib)
    return GpuFit("fits" if free >= need else "short", name, free, need)


def decide_rerank_placement(
    gpu_layers_cfg: object,
    device_memory: dict[str, tuple[int, int]],
    model_mb: int | None,
    *,
    compute_margin_mib: int = RERANK_COMPUTE_MARGIN_MIB,
) -> RerankPlacement:
    """配置を決める (純関数)。

    明示の整数ならそれに従う (0 = CPU)。``auto`` は GPU デバイス (``host`` 以外) の空きの
    最大がモデルサイズ + ``compute_margin_mib`` 以上なら GPU (``-ngl 999``)、でなければ CPU。
    """
    if not (isinstance(gpu_layers_cfg, str) and gpu_layers_cfg == "auto"):
        ngl = int(gpu_layers_cfg)  # type: ignore[call-overload]
        if ngl > 0:
            return RerankPlacement("gpu", ngl, "explicit gpu_layers")
        return RerankPlacement("cpu", 0, "explicit gpu_layers=0")
    fit = gpu_fit(device_memory, model_mb, margin_mib=compute_margin_mib)
    if fit.status == "no_gpu":
        return RerankPlacement("cpu", 0, "no GPU device")
    if fit.status == "unknown_size":
        return RerankPlacement("cpu", 0, "model size unknown")
    if fit.status == "fits":
        return RerankPlacement("gpu", 999, f"{fit.device} free {fit.free_mib} MiB >= need {fit.need_mib} MiB")
    return RerankPlacement("cpu", 0, f"{fit.device} free {fit.free_mib} MiB < need {fit.need_mib} MiB")


def build_rerank_cmd(
    cfg: dict, project_root: Path | None, placement: RerankPlacement,
) -> list[str] | None:
    """rerank 用 llama-server (``--reranking``) の起動コマンド。

    ``rag.rerank.mode`` が off / ``model_paths.rerank_model`` 未設定 / ファイル無しなら
    ``None`` (起動しない)。モデルパスは空白・括弧を含んでも argv の 1 要素のまま渡す。
    """
    if project_root is None:
        project_root = Path.cwd()
    ok, why = rerank_launchable(cfg, project_root)
    if not ok:
        print(f"[launch] rerank server not started: {why}")
        return None
    model = resolve_rerank_model_path(cfg, project_root)
    assert model is not None
    cmd = [
        "llama-server",
        "-m", str(model),
        "--port", str(rerank_port(cfg)),
        "--reranking",
        "-c", "8192",
        "-b", "8192",
        # 1 組 (query + doc) の最大長以上が必須 (512 では 500 エラー)
        "-ub", "2048",
        "-np", "1",
        "--cache-ram", "0",
        "-ngl", str(placement.gpu_layers),
    ]
    # 明示の正の整数はそのまま。0 は環境調整の項目 threads の配分 (決められなければ GPU は 2、CPU は省略)
    threads = _aux_threads(cfg, project_root, "rerank", gpu=placement.kind == "gpu")
    if placement.kind == "gpu":
        cmd += ["-fa", "on", "-t", str(threads or RERANK_GPU_DEFAULT_THREADS)]
    elif threads > 0:
        cmd += ["-t", str(threads)]
    return cmd


def rerank_cpu_fallback_cmd(cfg: dict, project_root: Path | None, cmd: list[str] | None) -> list[str] | None:
    """GPU 配置の rerank サーバが起動しなかったときに起こし直す CPU (``-ngl 0``) のコマンド。

    ``rag.rerank.gpu_layers: auto`` で ``-ngl`` が 0 以外のときだけ (明示の整数は利用者の選択なので
    そのまま)。それ以外は ``None``。保存結果は書き換えない (起動の失敗は環境起因)。全部の本番起動経路
    (--all / serve / ctl / server_control / LlamaProcessManager) がこの 1 実装を使う。
    """
    if not cmd or "-ngl" not in cmd or _explicit_gpu_layers(_rerank_cfg(cfg)) is not None:
        return None
    i = cmd.index("-ngl") + 1
    if i >= len(cmd) or cmd[i] == "0":
        return None
    return build_rerank_cmd(cfg, project_root, RerankPlacement("cpu", 0, "cpu_fallback_after_gpu_start_failed"))


def decide_rerank_memory(
    available_mb: int, model_mb: int | None, *, compute_margin_mib: int = RERANK_COMPUTE_MARGIN_MIB,
) -> tuple[bool, str]:
    """自己テストの前の RAM 余裕 (純関数)。利用可能な物理メモリ ≥ モデル + 計算バッファの見積り か。

    ``available_mb`` が 0 (取れなかった) / ``model_mb`` が不明なら判定せず可。iGPU は VRAM が共有 RAM
    なので GPU 配置でも見る。不足の理由は ``low_memory: ...``。
    """
    if available_mb <= 0 or model_mb is None:
        return True, ""
    need = model_mb + compute_margin_mib
    if available_mb >= need:
        return True, ""
    return False, f"low_memory: available {available_mb} MiB < need {need} MiB (model {model_mb} + buffer {compute_margin_mib})"


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
           "trained_on_model_key": str | None}``

    ``trained_on_model_key`` は evoref 独自 KV ``evoref.trained_on_model_key``
    (Level 2 トレーナーが LoRA アダプタに刻む学習元モデルの ``model_key``)。
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
        # hybrid (full attention + 線形 attention) 判定用。``<arch>.full_attention_interval``
        # が N なら N 層に 1 層だけが KV を持ち、残りは固定長の再帰状態を持つ
        # (Qwen3.5/3.8 の Gated DeltaNet 等)。不在 = 全層 full attention。
        "full_attention_interval": None,
        # 再帰状態のサイズ (``<arch>.ssm.state_size`` / ``<arch>.ssm.inner_size``)。
        "ssm_state_size": None,
        "ssm_inner_size": None,
        # 再帰層の畳み込み状態 (``<arch>.ssm.conv_kernel`` / ``<arch>.ssm.group_count``)。
        # checkpoint 1 つの状態量の見積り (estimate_checkpoint_state_mb) に使う。
        "ssm_conv_kernel": None,
        "ssm_group_count": None,
        # SWA の窓 (``<arch>.attention.sliding_window``)。不在 = SWA なし。
        "sliding_window": None,
        # LoRA アダプタ専用の evoref 独自 KV (学習元モデルの model_key)
        "trained_on_model_key": None,
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
                elif key.endswith(".full_attention_interval"):
                    try:
                        result["full_attention_interval"] = int(
                            _gguf_read_scalar_or_skip(f, vtype)
                        )
                    except (TypeError, ValueError):
                        pass
                elif key.endswith(".ssm.state_size"):
                    try:
                        result["ssm_state_size"] = int(_gguf_read_scalar_or_skip(f, vtype))
                    except (TypeError, ValueError):
                        pass
                elif key.endswith(".ssm.inner_size"):
                    try:
                        result["ssm_inner_size"] = int(_gguf_read_scalar_or_skip(f, vtype))
                    except (TypeError, ValueError):
                        pass
                elif key.endswith(".ssm.conv_kernel"):
                    try:
                        result["ssm_conv_kernel"] = int(_gguf_read_scalar_or_skip(f, vtype))
                    except (TypeError, ValueError):
                        pass
                elif key.endswith(".ssm.group_count"):
                    try:
                        result["ssm_group_count"] = int(_gguf_read_scalar_or_skip(f, vtype))
                    except (TypeError, ValueError):
                        pass
                elif key.endswith(".attention.sliding_window"):
                    try:
                        result["sliding_window"] = int(_gguf_read_scalar_or_skip(f, vtype))
                    except (TypeError, ValueError):
                        pass
                elif key == "tokenizer.chat_template":
                    result["has_chat_template"] = True
                    _gguf_skip_value(f, vtype)
                elif key == "evoref.trained_on_model_key":
                    val = _gguf_read_scalar_or_skip(f, vtype)
                    result["trained_on_model_key"] = (
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


# モデルプロファイル: arch 単位 + GGUF ファイル名単位の起動フラグ / sampling 既定。
# 同梱ベース (tracked, models/profiles/) + データ根の override (<data_root>/profiles/)
# の 2 段階 × arch 層 /
# モデル別層 (by-model/) の 2 スコープ。default フォールバックは持たない
# (プロファイルの無い arch はフラグ無付与)。
# ``models/`` 自体は .gitignore 対象だが models/profiles/ は再包含例外で tracked
# (``by-model/`` サブディレクトリにも同じ再包含が効く)。
_MODEL_PROFILE_BASE_DIR = "models/profiles"
_MODEL_PROFILE_BY_MODEL_SUBDIR = "by-model"


def _load_profile_layer(project_root: Path, *parts: str) -> tuple[dict, Path | None]:
    """1 スコープ分のプロファイルを解決する (データ根の override → 同梱 base)。

    最初に読めた YAML を wholesale で採用し ``(data, path)`` を返す。読取失敗 /
    dict でない場合は次の候補へ進み、どこにも無ければ ``({}, None)``。
    """
    for base in (
        _data_path({}, project_root, "profiles_dir"),
        project_root / _MODEL_PROFILE_BASE_DIR,
    ):
        if base is None:
            continue
        path = base.joinpath(*parts)
        if not path.exists():
            continue
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            continue
        if isinstance(data, dict):
            return data, path
    return {}, None


def _deep_merge(base: dict, over: dict) -> dict:
    """``over`` を ``base`` に重ねた新しい dict を返す (dict のみ再帰マージ)。

    list・スカラー・明示 ``None`` は置換する。``launch_flags`` は順序と排他性
    (``--reasoning-format auto`` 等) を持つため連結してはならず、明示 ``null`` は
    下位層の宣言を消す唯一の手段になる。
    """
    merged = dict(base)
    for key, value in over.items():
        current = merged.get(key)
        merged[key] = (
            _deep_merge(current, value)
            if isinstance(current, dict) and isinstance(value, dict)
            else value
        )
    return merged


def load_model_profile(arch: str | None, project_root: Path) -> dict:
    """arch 名からモデルプロファイル dict を解決する (2 段階)。

    解決順: ``<data_root>/profiles/<arch>.yaml`` (user override) →
    同梱 ``models/profiles/<arch>.yaml`` (base)。いずれも無い / ``arch`` が
    None の場合は ``{}`` (フォールバックなし)。override は wholesale 置換
    (deep merge しない)。
    """
    if not arch:
        return {}
    return _load_profile_layer(project_root, f"{arch}.yaml")[0]


def load_model_profile_for(
    model_path: Path, project_root: Path, *,
    warn: Callable[[str], None] | None = None,
) -> dict:
    """GGUF パスから有効なモデルプロファイルを解決する (プロファイル解決の入口)。

    解決順:
      ① ``models/profiles/<arch>.yaml``          同梱・arch 既定
      ② ``<data_root>/profiles/<arch>.yaml``     ①を wholesale 置換
      ③ ``models/profiles/by-model/<stem>.yaml`` 同梱・モデル固有
      ④ ``<data_root>/profiles/by-model/<stem>.yaml``  ③を wholesale 置換

    有効プロファイル = ``deep_merge(arch 層, モデル別層)``。同一スコープ内は
    wholesale 置換 (arch override の既存挙動を変えない)、スコープ跨ぎのみ
    deep merge なのでモデル別層には差分だけ書けばよい。``<stem>`` は GGUF
    ファイル名の拡張子なし部分をそのままファイル名として引く (正規化や
    あいまい照合はしない。大文字小文字の区別はファイルシステム依存)。
    GGUF 読取に失敗しても (arch 不明でも) モデル別層は適用される。
    """
    # ヘッダ解析はキャッシュ経由 (bge-m3 は語彙 25 万で 1 回 1 秒超。チャット
    # 応答パスの注入ゲートが毎ターン呼ぶため、素の read では TTFT に乗る)。
    try:
        arch = _read_gguf_metadata_cached(model_path).get("architecture")
    except Exception:  # noqa: BLE001
        arch = None
    profile = load_model_profile(arch, project_root)

    by_model, path = _load_profile_layer(
        project_root, _MODEL_PROFILE_BY_MODEL_SUBDIR, f"{model_path.stem}.yaml",
    )
    if not by_model:
        return profile
    if warn:
        warn(f"[launch] per-model profile applied: {path}")
    return _deep_merge(profile, by_model)


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
    """GGUF メタデータ + モデルプロファイルから llama-server 起動フラグを決定する。

    - ``llama.auto_model_flags`` が false なら何も付与しない (現行挙動)。
    - profile.launch_flags をベースに、MoE (GGUF が expert を報告 +
      profile.moe.enabled + n_cpu_moe が明示 int) なら ``--n-cpu-moe N`` を付与。
    - ``fixed_flags`` に既出のフラグは除外して二重付与を避ける。
    - GGUF 読取失敗 / arch 不明時はモデル別層のみが適用される。
    """
    lc = cfg.get("llama", {}) or {}
    if not lc.get("auto_model_flags", True):
        return []

    meta = read_gguf_metadata(model_path)
    profile = load_model_profile_for(model_path, project_root, warn=warn)
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
    if warn and ctx_train and isinstance(cfg_ctx, int) and cfg_ctx > int(ctx_train):
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
    *,
    n_seq: int = 1,
) -> int | None:
    """GGUF メタデータと文脈長から文脈メモリ VRAM 量 (MiB) を概算する。

    ``KV(bytes) = n_ctx × n_attn_layers × head_count_kv
                  × (key_length × bpe(K) + value_length × bpe(V))``

    ``n_attn_layers`` は **KV を実際に持つ層数**。全層 full attention なら
    ブロック数そのものだが、以下の 2 つを差し引く:

    - **MTP (NextN) 層** (``nextn_predict_layers``): draft 用の追加ヘッドで
      KV も再帰状態も持たない。
    - **線形 attention 層** (``full_attention_interval``): N 層に 1 層だけが
      full attention の hybrid arch (Qwen3.5/3.8 の Gated DeltaNet 等) では
      残り (N-1)/N の層が KV を持たず、代わりに文脈長に依存しない固定長の
      再帰状態を持つ。この分は ``ssm_state_size × ssm_inner_size × 4B (f32)``
      をシーケンス数倍して加算する (KV と違いシーケンス毎に確保されるため)。

    必要メタデータ (block_count / head_count_kv / key_length 等) が欠ける、
    または ``n_ctx<=0`` の場合は ``None`` を返し、呼び出し側で加算を
    スキップさせる (= 従来のファイルサイズのみ推定にフォールバック)。

    ``--kv-unified`` (slots=1 含む) を前提に KV 側は slots 倍しない (unified KV は
    n_ctx 総量で頭打ちになるため)。``n_seq`` が効くのは再帰状態の項だけ。
    host RAM 退避 (``--cache-ram``) は VRAM 側では差し引かない。

    実測突合 (2026-08-15, Qwen3.8-27B-Q4_K_M / n_ctx 8192 / f16,
    ``llama-fit-params --fit-print on`` の context 列):
    実測 661 / 811 / 1110 MiB (-np 1 / 2 / 4) に対し本式は 656 / 800 / 1088 MiB。
    差分は再帰層の畳み込み状態 (層あたり ~0.12 MiB) を省いている分で、
    GGUF に conv 次元が無いため意図的に落としている (誤差 1〜2%)。
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

    n_blocks = int(n_layers) - int(meta.get("nextn_predict_layers") or 0)
    if n_blocks <= 0:
        return None
    interval = int(meta.get("full_attention_interval") or 0)
    n_attn = n_blocks // interval if interval > 1 else n_blocks
    if n_attn <= 0:
        return None

    bytes_k = n_ctx * n_attn * n_head_kv * head_dim_k * _kv_bytes_per_elem(cache_type_k)
    bytes_v = n_ctx * n_attn * n_head_kv * head_dim_v * _kv_bytes_per_elem(cache_type_v)
    total = bytes_k + bytes_v

    n_recurrent = n_blocks - n_attn
    state_size = meta.get("ssm_state_size")
    inner_size = meta.get("ssm_inner_size")
    if n_recurrent > 0 and state_size and inner_size:
        total += (
            n_recurrent * int(state_size) * int(inner_size) * 4 * max(int(n_seq), 1)
        )

    return int(round(total / (1024 * 1024)))


def estimate_checkpoint_state_mb(
    meta: dict,
    n_ctx: int,
    cache_type_k: str | None,
    cache_type_v: str | None,
) -> int | None:
    """コンテキスト checkpoint 1 つ (1 スロット) がホスト RAM に持つ状態量 (MiB) を GGUF から見積もる。

    llama-server が checkpoint を作るのは部分巻き戻しのできないモデルだけで、保存するのは
    KV 全体ではなく巻き戻せない部分だけ:

    - **再帰層** (hybrid recurrent / 純 SSM): ``estimate_kv_cache_mb`` の再帰状態と同じ
      ``n_recurrent × ssm_state_size × ssm_inner_size × 4B (f32)`` に、畳み込み状態
      ``n_recurrent × (ssm_conv_kernel - 1) × (ssm_inner_size + 2 × ssm_group_count × ssm_state_size) × 4B``
      を足す (文脈長に依存しない)。実測 (2026-10-09, Qwen3.6-35B-A3B: 30 再帰層 / state 128 /
      inner 4096 / conv 4 / group 16) の 62.8 MiB と本式の 62.8 MiB が一致する。
    - **SWA**: 窓 ``min(sliding_window, n_ctx)`` 分の KV。どの層が SWA かは GGUF から確実に
      取れないので全層で数える (多め = RAM を保守側に見る)。

    どちらでもない純 attention モデルは checkpoint を作らないので 0。必要なメタデータが
    欠けて見積れなければ ``None`` (呼び出し側で概算へ縮退する)。
    """
    n_layers = meta.get("block_count")
    if not n_layers:
        return None
    n_blocks = int(n_layers) - int(meta.get("nextn_predict_layers") or 0)
    if n_blocks <= 0:
        return None
    interval = int(meta.get("full_attention_interval") or 0)
    state_size = meta.get("ssm_state_size")
    window = int(meta.get("sliding_window") or 0)
    is_recurrent = bool(state_size) or interval > 1
    if not is_recurrent and window <= 0:
        return 0

    total = 0
    if is_recurrent:
        inner_size = meta.get("ssm_inner_size")
        if not (state_size and inner_size):
            return None
        n_attn = n_blocks // interval if interval > 1 else 0
        n_recurrent = n_blocks - n_attn
        total += n_recurrent * int(state_size) * int(inner_size) * 4
        conv_kernel = int(meta.get("ssm_conv_kernel") or 0)
        if conv_kernel > 1:
            conv_dim = int(inner_size) + 2 * int(meta.get("ssm_group_count") or 0) * int(state_size)
            total += n_recurrent * (conv_kernel - 1) * conv_dim * 4
    if window > 0:
        n_head_kv = meta.get("head_count_kv")
        head_dim_k = meta.get("key_length")
        head_dim_v = meta.get("value_length") or head_dim_k
        if not head_dim_k:
            n_embd = meta.get("embedding_length")
            n_head = meta.get("head_count")
            if n_embd and n_head:
                head_dim_k = head_dim_v = n_embd // n_head
        if not (n_head_kv and head_dim_k and head_dim_v and n_ctx and n_ctx > 0):
            return None
        tokens = min(window, int(n_ctx))
        total += tokens * n_blocks * int(n_head_kv) * (
            head_dim_k * _kv_bytes_per_elem(cache_type_k) + head_dim_v * _kv_bytes_per_elem(cache_type_v)
        )
    return max(1, int(-(-total // (1024 * 1024))))


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
    if base_size and 0 < base_ngl < 999:
        # 層の一部だけを載せる (``gpu_layers: auto`` が空きに合わせて縮めた) ときは重みも層の割合で
        # 見積もる。全量で数えると、空きから導いた予算 (c_16 §7.2.3) の検査で必ず超過になる。
        layers = _read_gguf_layer_count(base_path)
        if layers and base_ngl < layers:
            base_vram = int(round(base_size * base_ngl / layers))

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
            # hybrid arch の再帰状態はスロット毎に確保されるため slots を渡す
            # (KV 側は --kv-unified 前提で slots 倍しない)。
            n_seq=resolve_base_slots(cfg, project_root),
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

    # 埋め込み
    emb_cfg = cfg.get("embedding", {}) or {}
    if emb_cfg.get("backend", "llama-cpp") == "llama-cpp":
        embed_path = _resolve_model_path(cfg, "embed_model", "", project_root)
        embed_ngl = _resolve_embed_gpu_layers(cfg, project_root)
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


_DEVICE_NAME_RE = re.compile(
    # "Vulkan0: AMD Radeon(TM) 890M Graphics (48923 MiB, 46477 MiB free)" (--list-devices)
    r"^([A-Za-z][A-Za-z0-9_]*)\s*:\s*(.+?)\s*\(\d+\s*MiB,\s*\d+\s*MiB\s*free\)\s*$"
)


def _parse_device_names(text: str) -> list[str]:
    """device 列挙行 (フォーマット A) から ``"<device>: <製品名>"`` の一覧を返す。

    容量 (空きは刻々と変わる) は含めない — PC の指紋 (リランカー自己テスト) に使うため。
    """
    names: list[str] = []
    for raw in (text or "").splitlines():
        m = _DEVICE_NAME_RE.match(raw.strip())
        if m and m.group(1).lower() != "host":
            names.append(f"{m.group(1)}: {m.group(2)}")
    return names


def _list_llama_devices(binary: str = "llama-server", timeout: float = 30.0) -> str:
    """``llama-server --list-devices`` の出力 (stdout + stderr)。失敗は空文字列。"""
    try:
        result = subprocess.run(
            [binary, "--list-devices"],
            capture_output=True, timeout=timeout, check=False,
            encoding="utf-8", errors="replace",
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return ""
    return (result.stdout or "") + "\n" + (result.stderr or "")


# config 明示も arch プロファイル宣言も無い場合の context_size slot 別既定。
# backend/config.py::_CONTEXT_SIZE_FALLBACK と一致させること (サーバ ``-c`` と
# ランタイム token budget の値を揃えるため)。
_CONTEXT_SIZE_DEFAULTS: dict[str, int] = {
    "base": 8192, "embed": 8192,
}


def _profile_context_size_for_model(
    model_path: Path, project_root: Path,
) -> int | None:
    """モデルパスからプロファイルの ``context_size`` を返す。

    ``load_model_profile_for`` で解決 (arch 層 + モデル別層)。未宣言 / 512 未満 /
    不正値 / 読取失敗はすべて ``None`` (呼び出し側で slot 別既定にフォールバック)。
    """
    try:
        profile = load_model_profile_for(model_path, project_root)
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
    (``llama.context_size``) が ``None`` (未指定) のときのみ profile を参照
    する。プロファイル参照は base のみ、かつ ``project_root`` 指定時のみ
    (未指定なら config + 既定で解決)。
    embed は profile 非対象 (従来挙動)。

    ``model_override`` 指定時 (例: /api/mode/switch で create_model に差し替えて
    base を再起動する経路) は、その実モデルの profile から ``context_size`` を
    引く。これにより ``-m`` で渡すモデルと ``-c`` が一致する (旧実装は常に
    base_model profile から ``-c`` を引いていた)。``llama.context_size`` の明示は
    手動 pin として override より優先する。

    ``auto`` / ``null`` (c_16 §7.2.3) は環境調整の項目 ctx — 保存済みの調整結果 (この PC・この
    base モデル) → その場の見積り (空き VRAM / RAM に収まる最大、上限は profile) → 保守側 8192。
    調整は ``model_paths.base_model`` について決めるので、別モデルの ``model_override`` は従来どおり
    そのモデルの profile を引く。backend が無い配布物も従来どおり profile。
    """
    default = _CONTEXT_SIZE_DEFAULTS.get(name, 8192)
    if name == "base":
        explicit = (cfg.get("llama") or {}).get("context_size")
        model_key = "base_model"
    elif name == "embed":
        return int((cfg.get("embedding") or {}).get("context_size", default))
    else:
        return default

    if explicit is not None and explicit != "auto":
        return int(explicit)
    if project_root is not None:
        base_rel = (cfg.get("model_paths") or {}).get(model_key, "")
        model_rel = model_override or base_rel
        if model_rel:
            model_path = Path(model_rel)
            if not model_path.is_absolute():
                model_path = project_root / model_path
            same_as_base = not model_override or (
                bool(base_rel) and _same_path(model_path, project_root / Path(base_rel))
            )
            if same_as_base:
                tuned = _tuned(cfg, project_root, "ctx")
                if tuned is not None and isinstance(tuned.value, int):
                    return int(tuned.value)
            profile_ctx = _profile_context_size_for_model(model_path, project_root)
            if profile_ctx is not None:
                return profile_ctx
    return default


def _is_base_model(cfg: dict, project_root: Path, model_override: str | None) -> bool:
    """``model_override`` が無いか、``model_paths.base_model`` と同じファイルか (環境調整の値を使ってよいか)。"""
    if not model_override:
        return True
    base_rel = (cfg.get("model_paths") or {}).get("base_model", "")
    if not base_rel:
        return False

    def absolute(rel: str) -> Path:
        path = Path(rel)
        return path if path.is_absolute() else project_root / path

    return _same_path(absolute(model_override), absolute(base_rel))


def _same_path(a: Path, b: Path) -> bool:
    """2 つのパスが同じファイルを指すか (存在しなくても表記を正規化して比べる)。"""
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


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


# ── auto エントリ: 環境調整の項目 ngl (c_16 §7.2.3) + キャッシュ ──
_AUTO_NGL_CACHE_KEY = "__auto_ngl_cache__"


def _resolve_auto_gpu_layers(
    cfg: dict, project_root: Path,
) -> dict[str, int] | None:
    """``gpu_layers="auto"`` 指定時の base の ``-ngl`` (環境調整の項目 ngl)。

    保存済みの調整結果 (この PC・このモデル) → その場の見積り (GPU の **空き** から段階縮小、
    ``backend.free.core.tuning.tuners.ngl``。保存する) の順に決める (``resolve_tuned``)。
    環境移行の確認待ちの間は保存しない一時の見積り (取れなければ全層の 40%、``tuners.ngl.conservative_ngl``)。
    決められない (モデルが読めない / backend が無い) ときは None を返し、
    呼び出し側は既定 999 にフォールバックする。``runtime.gpu_auto_tune_enabled: false`` も None。

    1 度だけ計算し ``cfg[_AUTO_NGL_CACHE_KEY]`` に dict をキャッシュする。

    Returns:
        ``{"base_ngl": int, "reason": str}`` または None。
    """
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

    resolved = _tuned(cfg, project_root, "ngl")
    if resolved is None or resolved.source == "fallback" or not isinstance(resolved.value, int):
        cfg[_AUTO_NGL_CACHE_KEY] = {}
        why = "auto-tune unavailable" if resolved is None else resolved.reason
        print(
            f"[launch] WARNING: gpu_layers='auto' could not be tuned ({why}); "
            "falling back to -ngl 999"
        )
        return None

    reason = f"{resolved.source}: {resolved.reason}"
    if resolved.source == "provisional":
        print(
            f"[launch] WARNING: gpu_layers='auto' uses an unsaved estimate (base={resolved.value}) until the "
            "re-tune after the PC change is confirmed. Run `evoref tune --startup-check` or `evoref tune run`"
        )
    print(f"[launch] auto-tuned gpu_layers: base={resolved.value}")
    print(f"[launch]   reason: {reason}")
    result = {"base_ngl": int(resolved.value), "reason": reason}
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


def _saved_rerank_known_invalid(cfg: dict, project_root: Path, rr: dict, explicit: int) -> bool:
    """明示の ``gpu_layers`` と同じ配置の保存結果が無効で、起動しないと分かっているか (VRAM を足さない)。

    保存結果が無い / 配置が違う (測り直す) / 測り直す失敗 (環境起因・``too_slow``) を数えている途中 (また試す) は ``False``。
    """
    path = rerank_selftest_path(cfg, project_root)
    if path is None:
        return False
    try:
        from backend.free.rag import rerank_selftest as st
    except ImportError:
        return False
    saved, _status = st.load_selftest_result(path)
    if saved is None or saved.gpu_layers != explicit:
        return False
    if not saved.enabled and st.is_retryable_failure(saved.reason) and (
        saved.environmental_streak < st.ENVIRONMENTAL_FAILURE_LIMIT
    ):
        return False
    enabled, _candidates, _reason = _usable_saved(st, saved, rr)
    return not enabled


def estimate_rerank_vram(cfg: dict, project_root: Path) -> dict:
    """rerank の VRAM 見積り (Tier 2)。GPU 配置のときだけ ``present`` True で合算に入る。

    起動対象 (:func:`rerank_launchable`) で、明示の ``gpu_layers > 0`` (保存結果が同じ配置で無効と
    分かっていれば除く) か、``auto`` で保存済みの自己テストが GPU 配置のとき。mode off / モデル無し / CPU 配置 / 未テストの ``auto`` は 0 MB・
    ``present`` False (埋め込みの未判別 = CPU と同じ扱い)。``vram_mb`` はモデル + 計算バッファの見積り。
    """
    model = resolve_rerank_model_path(cfg, project_root)
    entry = {
        "model_mb": None, "gpu_layers": 0, "vram_mb": 0, "present": False,
        "path": str(model) if model is not None else "",
        "context_mb": None, "compute_mb": None, "device": None,
        "estimated_via": "gguf-size",
    }
    if not rerank_launchable(cfg, project_root)[0] or model is None:
        return entry
    rr = _rerank_cfg(cfg)
    explicit = _explicit_gpu_layers(rr)
    if explicit is not None:
        ngl = explicit
        if ngl > 0 and _saved_rerank_known_invalid(cfg, project_root, rr, explicit):
            ngl = 0
    else:
        saved = saved_rerank_placement(cfg, project_root)
        ngl = saved.gpu_layers if saved is not None and saved.kind == "gpu" else 0
    size = _file_size_mb(model)
    entry["model_mb"] = size
    entry["gpu_layers"] = ngl
    if ngl > 0 and size is not None:
        entry["compute_mb"] = RERANK_COMPUTE_MARGIN_MIB
        entry["vram_mb"] = size + RERANK_COMPUTE_MARGIN_MIB
        entry["present"] = True
    return entry


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

    runtime_cfg = cfg.get("runtime", {}) or {}
    if prefer_fit_params and runtime_cfg.get("fit_params_enabled", True):
        result = _try_estimate_via_fit_params(cfg, base_result, project_root)
    else:
        result = base_result
    # rerank は常に Tier 2 (サイズ + 計算バッファの見積り)。GPU 配置のときだけ present
    result["rerank"] = estimate_rerank_vram(cfg, project_root)
    return result


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
    for name in ("base", "embed"):
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
    rerank = estimates.get("rerank") or {}
    if rerank.get("present"):
        lines.append(
            f"  {'rerank':<9s} : GPU ngl={rerank.get('gpu_layers', 0):<4d} "
            f"model_size={rerank.get('model_mb')}MB + buffer={rerank.get('compute_mb')}MB "
            f"est_vram={rerank.get('vram_mb', 0)}MB"
        )
    return lines


def _vram_budget_mb(cfg: dict, project_root: Path | None) -> tuple[int | None, str]:
    """VRAM 予算と表示の注記。``runtime.total_vram_budget_mb`` が ``null`` なら環境調整の項目
    vram_budget (GPU の空き × 安全率、c_16 §7.2.3)。GPU が無い・決められなければ ``None`` (検査しない)。
    """
    raw = (cfg.get("runtime") or {}).get("total_vram_budget_mb")
    if raw is not None:
        return int(raw), ""
    if project_root is None:
        return None, ""
    tuned = _tuned(cfg, project_root, "vram_budget")
    if tuned is None or tuned.value is None:
        return None, ""
    return int(tuned.value), f" (auto-tuned: {tuned.reason})"


def _vram_budget_is_explicit(cfg: dict) -> bool:
    """``runtime.total_vram_budget_mb`` を利用者が数値で明示したか (``null`` は自動で導いた目安)。"""
    return (cfg.get("runtime") or {}).get("total_vram_budget_mb") is not None


def check_vram_budget(
    cfg: dict, project_root: Path | None = None, *, force: bool = False,
) -> tuple[bool, int, int | None, dict[str, dict], str]:
    """VRAM 予算との比較を行う

    Returns:
        (ok, total_vram_mb, budget_mb, estimates, message)

        - ``ok``: True なら起動継続可。False は **明示の予算** を超過中 (``force`` なら True)。
          自動で導いた予算 (``null``) の超過は警告だけで True
        - ``total_vram_mb``: 判定に使った VRAM 合計 (base + embed + GPU 配置の rerank)
        - ``budget_mb``: ``runtime.total_vram_budget_mb`` (未設定なら None)
        - ``estimates``: モデル別内訳 (``estimate_vram_usage_mb`` の返り値)
        - ``message``: 人間可読なログ用メッセージ
    """
    estimates = estimate_vram_usage_mb(cfg, project_root)
    total_vram_mb = sum(e.get("vram_mb", 0) for e in estimates.values())
    budget_mb, budget_note = _vram_budget_mb(cfg, project_root)

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
        lines.append(f"  runtime.total_vram_budget_mb: {budget_mb} MB{budget_note}")

    # Tier 1 結果から 10% headroom 付き推奨値を提示する
    if budget_mb is None:
        suggested = suggest_total_vram_budget_mb(estimates)
        if suggested is not None:
            lines.append(
                f"  Suggested: set runtime.total_vram_budget_mb={suggested} "
                "(current+10% headroom)"
            )
        return True, total_vram_mb, None, estimates, "\n".join(lines)

    # 自動で導いた予算 (``null`` → 項目 vram_budget = 空き × 0.9) は目安: 超えても警告だけで止めない。
    # ctx・b・ub の見積りは「空き - ヘッドルーム」まで詰めるので、空き × 0.9 の予算とは食い違いうる
    # (12 GB の単体 GPU で 8B・ctx 32768・ub 1024 なら必要 ≈ 11.2 GB > 予算 10.8 GB)。中断と slots の降格は
    # 利用者が明示した予算のときだけ (``null`` は以前「検査しない」だったので、既存の --all を止めない)。
    explicit_budget = _vram_budget_is_explicit(cfg)

    # ``llama.slots: auto`` が 4 本目 (long_form 専用) を選んだせいで予算を
    # 超えるなら、そのぶん (hybrid arch の再帰状態 1 シーケンス分) を諦めて 3 に
    # 降格する。明示 ``slots: 4`` は降格しない (超過は従来どおり警告 / アボート)。
    resolved = (cfg.get(_RESOLVED_SLOTS_KEY) or {}).get("")
    if (
        explicit_budget and resolved is not None and resolved.get("auto")
        and int(resolved.get("slots", 0)) == SLOTS_WITH_LONG_FORM
        and total_vram_mb > int(budget_mb)
        and total_vram_mb - int(resolved.get("per_seq_mb", 0)) <= int(budget_mb)
    ):
        per_seq_mb = int(resolved.get("per_seq_mb", 0))
        resolved["slots"] = SLOTS_BASE
        total_vram_mb -= per_seq_mb
        if "base" in estimates:
            estimates["base"]["vram_mb"] = int(estimates["base"].get("vram_mb", 0)) - per_seq_mb
        lines.append(
            f"[launch] llama.slots=auto: dropping the long_form slot (-{per_seq_mb} MB) "
            f"to fit runtime.total_vram_budget_mb={budget_mb}; using {SLOTS_BASE} slots"
        )

    if total_vram_mb > int(budget_mb):
        over = total_vram_mb - int(budget_mb)
        lines.append(
            f"[launch] WARNING: estimated VRAM {total_vram_mb} MB exceeds budget "
            f"{budget_mb} MB (over by {over} MB)"
        )
        if not explicit_budget:
            lines.append(
                "[launch] WARNING: auto-derived budget exceeded; continuing. "
                "Set runtime.total_vram_budget_mb to enforce"
            )
            return True, total_vram_mb, int(budget_mb), estimates, "\n".join(lines)
        if force:
            lines.append(
                "[launch] --force specified: continuing despite VRAM budget overrun"
            )
            return True, total_vram_mb, int(budget_mb), estimates, "\n".join(lines)
        lines.append(
            "[launch] Aborting. Re-run with --force to override, or set "
            "embedding.gpu_layers to 0 (CPU fallback) "
            "(or rag.rerank.gpu_layers: 0 when rerank is in the breakdown) "
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


def _llama_version_text(binary: str = "llama-server", timeout: float = 2.0) -> str | None:
    """``llama-server --version`` の出力 (stdout + stderr)。起動できなければ ``None``。"""
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
    return (result.stdout or "") + "\n" + (result.stderr or "")


def _probe_llamacpp_build(
    binary: str = "llama-server", timeout: float = 2.0,
) -> int | None:
    """``llama-server --version`` を実行し build 番号を抽出する

    タイムアウト / バイナリ不存在 / 抽出失敗のいずれもサイレントに None を返す。
    バイナリが build 番号を露出しないカスタムビルドにも対応するため、抽出
    失敗を「要件未満」とは扱わない (呼び出し側で警告のみで継続)。
    """
    combined = _llama_version_text(binary, timeout)
    if combined is None:
        return None
    return _parse_build_number(combined)


def _llama_server_version(binary: str = "llama-server", timeout: float = 5.0) -> str:
    """``llama-server --version`` の ``version:`` 行の値 (リランカー自己テストの記録用、無ければ空)。"""
    for line in (_llama_version_text(binary, timeout) or "").splitlines():
        if line.strip().startswith("version:"):
            return line.strip()[len("version:"):].strip()
    return ""


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


def _wait_health(ports: dict[str, int], timeout_sec: float) -> None:
    """各サーバの ``/health`` が 200 を返すまで待つ (超過は WARNING を出して続行)。"""
    for name, port in ports.items():
        if wait_for_health("localhost", port, timeout=int(timeout_sec)):
            print(f"[launch] {name} (port {port}) is healthy")
        else:
            print(f"[launch] WARNING: {name} (port {port}) health check timed out, proceeding anyway")


#: 起動の health 待ちの下限 (``backend.free.llm._base_client.HEALTH_WAIT_FLOOR_SEC`` と同じ値。
#: backend を import できない配布形態のときだけの保険で、通常は backend の 1 実装が決める)。
_HEALTH_WAIT_FALLBACK_SEC = 120


def _health_wait(
    cfg: dict, model_path: "str | Path | None", project_root: Path | None = None,
    *, floor: float = _HEALTH_WAIT_FALLBACK_SEC, label: str = "llama-server",
) -> int:
    """起動の health 待ちの秒数 (全経路の 1 実装 ``resolve_health_wait`` を呼ぶ、c_16 §7.2.3)。

    config の ``process_manager.health_timeout`` が整数ならそのまま、``auto`` / 無しなら
    GGUF サイズ連動 (``floor`` 以上、上限 600)。
    """
    try:
        from backend.free.llm._base_client import health_wait_for_cfg
    except ImportError:
        configured = (cfg.get("process_manager") or {}).get("health_timeout", "auto")
        return int(configured) if isinstance(configured, int) and configured > 0 else int(floor)
    return health_wait_for_cfg(cfg, model_path, project_root=project_root, floor=floor, label=label)


def _sized_wait(model_path: "str | Path | None", floor: float, *, label: str) -> int:
    """GPU 起動待ち / 判別プローブの待ち (``floor`` 以上でモデルサイズ連動。config の明示値は見ない)。

    ``floor`` は現行の値 (60 / 30) で、「GPU で起動しない PC」を速く検知する意図のため下げない。
    延びるのは大型モデルだけ。呼び出し側は本番の待ちとの ``min`` を取る。
    """
    try:
        from backend.free.llm._base_client import resolve_health_wait
    except ImportError:
        return int(floor)
    return resolve_health_wait("auto", model_path, floor=floor, label=label)


def _model_path_of_cmd(cmd: list[str] | None) -> str | None:
    """``cmd`` 中の ``-m`` の値 (起動するモデルのパス。無ければ ``None``)。"""
    if cmd and "-m" in cmd and cmd.index("-m") + 1 < len(cmd):
        return str(cmd[cmd.index("-m") + 1])
    return None


def _extract_model_basename(cmd: list[str]) -> str | None:
    """``cmd`` 中の ``-m`` 引数の basename を返す

    ``build_llama_cmd`` / ``build_embed_cmd`` はいずれも
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


# ── rerank の起動と自己テスト (c_16 §7.2.1) ─────────────────
# 自己テストは rerank サーバを所有するプロセス (この起動スクリプト / evoref serve) で
# 走らせる: GPU で退化したら CPU で起動し直す必要があり、プロセスの持ち主でないと
# 止めて起こし直せない。backend はその結果ファイルを読むだけ。

RERANK_HOST = "127.0.0.1"


@dataclass
class RerankLaunch:
    """:func:`start_rerank_server` の結果。``proc`` は起動したまま残した rerank サーバ。"""

    proc: subprocess.Popen | None
    #: 使った / 書いた自己テストの結果 (``RerankSelftestResult``)。起動しなかった前段の理由では ``None``。
    result: object | None
    #: この呼出で自己テストを走らせたか。
    tested: bool
    #: 起動しなかった / 無効にした理由 (英語)。
    message: str = ""


def _stop_proc(proc: subprocess.Popen, timeout: float = 10.0) -> None:
    """``proc`` を止めて待つ (止まらなければ kill)。"""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _port_answers(host: str, port: int) -> bool:
    """``host:port`` の ``/health`` が何か答えるか (既に別のサーバが居る)。"""
    try:
        httpx.get(f"http://{host}:{port}/health", timeout=1.0)
    except (httpx.ConnectError, httpx.TimeoutException):
        return False
    except httpx.HTTPError:
        return True
    return True


def rerank_selftest_path(cfg: dict, project_root: Path) -> Path | None:
    """自己テスト結果の置き場 (``PathResolver.LAYOUT["rerank_selftest_file"]``)。"""
    return _data_path(cfg, project_root, "rerank_selftest_file")


#: 起動スクリプトが rerank の起動を断念した印のファイル名 (``run_dir`` の下)。中身は理由 1 行。
RERANK_ABANDONED_FILE = "rerank_abandoned"


def rerank_abandoned_path(cfg: dict, project_root: Path) -> Path | None:
    """起動断念の印の置き場 (``PathResolver.LAYOUT["run_dir"]`` の下)。ctl の ``--wait-rerank`` が見る。"""
    run_dir = _data_path(cfg, project_root, "run_dir")
    return None if run_dir is None else run_dir / RERANK_ABANDONED_FILE


def clear_rerank_abandoned(cfg: dict, project_root: Path) -> None:
    """古い起動断念の印を消す (前回の印で今回の待ちを誤って打ち切らない)。"""
    path = rerank_abandoned_path(cfg, project_root)
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError as e:
        print(f"[launch] WARNING: failed to remove {path}: {e}", file=sys.stderr)


def mark_rerank_abandoned(cfg: dict, project_root: Path, reason: str) -> None:
    """rerank の起動を断念した印を残す (別プロセスの :func:`wait_rerank_ready` が即 ``disabled`` を返す)。"""
    path = rerank_abandoned_path(cfg, project_root)
    if path is None:
        return
    try:
        from backend.io.atomic import atomic_write_text

        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, f"{reason or 'not_started'}\n")
    except (ImportError, OSError) as e:
        print(f"[launch] WARNING: failed to write {path}: {e}", file=sys.stderr)


def _explicit_gpu_layers(rr: dict) -> int | None:
    """``rag.rerank.gpu_layers`` が明示の整数ならその値、``auto`` なら ``None``。"""
    raw = rr.get("gpu_layers", "auto")
    if isinstance(raw, str) and raw == "auto":
        return None
    return int(raw)


def _usable_saved(st, saved, rr: dict) -> tuple[bool, int, str]:
    """保存済みの結果を **現在の** 締切・候補数で読み直す (再テストしない)。"""
    return st.effective_candidates(
        saved,
        deadline_ms=int(rr.get("deadline_ms", 1000)),
        max_candidates=int(rr.get("max_candidates", 12)),
        min_candidates=int(rr.get("min_candidates", 3)),
    )


def saved_rerank_placement(cfg: dict, project_root: Path) -> RerankPlacement | None:
    """保存済みの自己テストが (現在の設定で) 有効なら、その配置 (再起動の経路用。測り直さない)。

    明示の ``gpu_layers`` (整数) が保存済みの配置と食い違うときは ``None`` — その配置は
    未テストなので起こさない (測り直すのは起動スクリプト、理由 ``placement_setting_changed``)。
    """
    path = rerank_selftest_path(cfg, project_root)
    if path is None:
        return None
    try:
        from backend.free.rag import rerank_selftest as st
    except ImportError:
        return None
    saved, _status = st.load_selftest_result(path)
    if saved is None or saved.placement not in ("gpu", "cpu"):
        return None
    rr = _rerank_cfg(cfg)
    explicit = _explicit_gpu_layers(rr)
    if explicit is not None and explicit != saved.gpu_layers:
        print(
            f"[launch] rerank not started: gpu_layers={explicit} differs from the self-tested "
            f"placement ({saved.gpu_layers}); re-run the launcher to test it",
        )
        return None
    enabled, _candidates, _reason = _usable_saved(st, saved, rr)
    if not enabled:
        return None
    return RerankPlacement(saved.placement, saved.gpu_layers, "saved self-test")


def _tune_gate(cfg: dict, project_root: Path, pc, why: str, *, force: bool = False) -> tuple[bool, str]:
    """環境移行の確認 (c_16 §7.2.3) — 測り直す直前に見る。(測ってよいか, 理由)。

    判定は backend の 1 実装 (``backend.free.core.tuning.gate.should_measure``) に任せる。``force``
    (手動の --embed-placement / --rerank-selftest) は常に測り、現在の PC に ``accepted`` を記録する。
    確認を挟むのは PC が変わった / 結果が無い回だけ (新規インストールは ``fresh`` で従来どおり測る)。
    backend を import できなければ従来どおり測る。
    """
    try:
        from backend.free.core.tuning import gate as tg
        from backend.free.core.tuning.store import resolve_tune_paths
    except ImportError:
        return True, "gate_unavailable"
    paths = resolve_tune_paths(cfg, project_root)
    if force:
        tg.record_decision(paths.auto_tune, "accepted", pc)
        return True, "forced"
    if why not in tg.GATED_REASONS:
        return True, why
    return tg.should_measure(tg.load_gate(cfg, project_root, pc, paths=paths))


def _rerank_health_once(host: str, port: int, expected_model_id: str | None) -> bool:
    """``/health`` が 200 で、``/props`` のモデルが一致するか (1 回だけ見る)。"""
    try:
        resp = httpx.get(f"http://{host}:{port}/health", timeout=2.0)
    except (httpx.ConnectError, httpx.TimeoutException):
        return False
    if resp.status_code != 200:
        return False
    return expected_model_id is None or _props_model_matches(host, port, expected_model_id)


def _wait_healthy_or_dead(
    proc: subprocess.Popen,
    host: str,
    port: int,
    timeout: float,
    expected_model_id: str | None,
    *,
    probe: Callable[[str, int, str | None], bool] = _rerank_health_once,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> bool:
    """health を待つ。プロセスが死んだら **その場で** ``False`` (timeout 全部は待たない)。"""
    deadline = clock() + timeout
    while True:
        if proc.poll() is not None:
            return False
        if probe(host, port, expected_model_id):
            return True
        if clock() >= deadline:
            return False
        sleep(1.0)


def start_rerank_server(cfg: dict, project_root: Path, **kwargs) -> RerankLaunch:
    """rerank サーバを起動する (:func:`_start_rerank_server`)。起動を断念した回は印を残す。

    始める時点で古い印を消し、起動対象 (:func:`rerank_launchable`) なのにサーバを残さず返る回
    (保存結果が無効・起動失敗・自己テスト失敗・確認待ち・メモリ不足 等) は ``run/`` に印を書く。
    別プロセスの ctl の ``--wait-rerank`` (:func:`wait_rerank_ready`) はそれを見て待ちを打ち切る。
    """
    clear_rerank_abandoned(cfg, project_root)
    try:
        launched = _start_rerank_server(cfg, project_root, **kwargs)
    except BaseException:
        # 例外 (中断を含む) で抜けた回も待ち側を 240 秒待たせない
        _mark_rerank_abandoned_if_launchable(cfg, project_root, "launcher_error")
        raise
    if launched.proc is None:
        _mark_rerank_abandoned_if_launchable(cfg, project_root, launched.message)
    return launched


def _mark_rerank_abandoned_if_launchable(cfg: dict, project_root: Path, reason: str) -> None:
    """起動対象 (mode off / モデル無しでない) のときだけ起動断念の印を書く。"""
    if rerank_launchable(cfg, project_root)[0]:
        mark_rerank_abandoned(cfg, project_root, reason)


def _start_rerank_server(
    cfg: dict,
    project_root: Path,
    *,
    force_selftest: bool = False,
    popen_kwargs: dict | None = None,
    popen: Callable[..., subprocess.Popen] = subprocess.Popen,
    list_devices: Callable[[], str] = _list_llama_devices,
    health: Callable[..., bool] = _wait_healthy_or_dead,
    measure: Callable[..., object] | None = None,
    port_busy: Callable[[str, int], bool] = _port_answers,
    server_version: Callable[[], str] = _llama_server_version,
    health_timeout: int | None = None,
    available_memory: Callable[[], int] | None = None,
) -> RerankLaunch:
    """rerank サーバを起動する。PC の指紋が保存結果と違うときだけ自己テストを走らせる。

    - mode off / モデル未設定 / ファイル無し → 起動しない (INFO)。
    - 指紋が一致 → 保存値の配置で起動。候補数は保存済みの ms/件と **現在の** 締切・候補数で
      引き直す (再テストしない)。無効なら起動しない。モデルが変わっていても再テストせず WARNING。
    - 一致しない / 結果が無い / 読めない / ``force_selftest`` / 明示の ``gpu_layers`` が
      保存済みの配置と違う (``placement_setting_changed``、唯一の例外) → 配置を決めて起動して
      測る。GPU で退化・起動失敗したら CPU で 1 回だけ起動し直す (``gpu_layers: auto`` の
      ときだけ。遅すぎは再試行しない)。
    - 自己テストの前に利用可能な物理メモリがモデル + 計算バッファ未満なら起動せず ``low_memory``
      (保存も数えもしない。次の起動でまた試す)。
    - 結果は ``cache/rerank_selftest.json`` へ書く。環境起因の失敗 (``server_unhealthy`` /
      ``http_error:*``) が混じった回は、連続回数を ``environmental_streak`` に数えて保存し、
      ``ENVIRONMENTAL_FAILURE_LIMIT`` 回に届くまで次の起動でまた試す (届いたら無効が確定)。
    - 保存済みの配置が GPU (``gpu_layers: auto``) で起動しなければ CPU で 1 回だけ起こし直す
      (:func:`rerank_cpu_fallback_cmd`)。保存結果は書き換えない。

    起動・health・自己テストの失敗はチャットを止めない (rerank を無効にして続ける)。
    """
    ok, why = rerank_launchable(cfg, project_root)
    if not ok:
        print(f"[launch] rerank server not started: {why}")
        return RerankLaunch(None, None, False, why)
    try:
        from backend.free.rag import rerank_selftest as st
        from backend.model_key import model_key_for
        from backend.utils import utc_now
    except ImportError as e:
        msg = f"backend is not importable ({e}); the rerank self-test cannot run"
        print(f"[launch] WARNING: rerank server not started: {msg}", file=sys.stderr)
        return RerankLaunch(None, None, False, msg)
    path = rerank_selftest_path(cfg, project_root)
    if path is None:
        msg = "data root is not resolvable"
        print(f"[launch] WARNING: rerank server not started: {msg}", file=sys.stderr)
        return RerankLaunch(None, None, False, msg)

    port = rerank_port(cfg)
    if port_busy(RERANK_HOST, port):
        msg = f"port {port} is already in use; stop the old rerank server first (evoref-ctl stop)"
        print(f"[launch] WARNING: rerank server not started: {msg}", file=sys.stderr)
        return RerankLaunch(None, None, False, msg)

    rr = _rerank_cfg(cfg)
    explicit = _explicit_gpu_layers(rr)
    deadline_ms = int(rr.get("deadline_ms", 1000))
    max_candidates = int(rr.get("max_candidates", 12))
    min_candidates = int(rr.get("min_candidates", 3))
    model = resolve_rerank_model_path(cfg, project_root)
    assert model is not None
    model_key = model_key_for(model)
    timeout = health_timeout or _health_wait(cfg, model, project_root, label="rerank")
    kwargs = {"cwd": project_root, **(popen_kwargs or {})}

    devices_text = list_devices()
    pc = st.collect_pc_info(_parse_device_names(devices_text))
    fingerprint = pc.digest
    saved, status = st.load_selftest_result(path)
    needed, why_test = st.selftest_needed(
        saved, fingerprint, force=force_selftest, explicit_gpu_layers=explicit,
    )

    def spawn(
        placement: RerankPlacement, cmd: list[str] | None = None, wait: float = timeout,
    ) -> subprocess.Popen | None:
        cmd = cmd or build_rerank_cmd(cfg, project_root, placement)
        if cmd is None:
            return None
        print(f"[launch] rerank ({placement.kind}): {' '.join(cmd)}")
        try:
            proc = popen(cmd, **kwargs)
        except (FileNotFoundError, OSError) as e:
            print(f"[launch] WARNING: failed to spawn rerank server: {e}", file=sys.stderr)
            return None
        if health(proc, RERANK_HOST, port, wait, model.name):
            return proc
        print(f"[launch] WARNING: rerank server ({placement.kind}) did not become healthy", file=sys.stderr)
        if placement.kind == "gpu" and saved_embed_placement(cfg, project_root).kind == "gpu":
            print(
                "[launch] WARNING: the embedding server is also on GPU (cache/embed_placement.json); "
                "GPU memory may be short for both. The reranker is disabled for this run (retrieval "
                "falls back to cosine order); set embedding.gpu_layers: 0 to keep the GPU for the reranker",
                file=sys.stderr,
            )
        _stop_proc(proc)
        return None

    if not needed and saved is not None:
        if saved.model_key and saved.model_key != model_key:
            print(
                f"[launch] WARNING: model_paths.rerank_model changed since the self-test "
                f"({saved.model_file} -> {model.name}); not re-testing (only a PC change re-tests). "
                "Run --rerank-selftest to measure the new model",
                file=sys.stderr,
            )
        enabled, candidates, reason = _usable_saved(st, saved, rr)
        if not enabled:
            print(f"[launch] rerank disabled by the saved self-test ({reason}); not started")
            return RerankLaunch(None, saved, False, reason)
        placement = RerankPlacement(saved.placement, saved.gpu_layers, "saved self-test")
        gpu_cmd = build_rerank_cmd(cfg, project_root, placement)
        fallback = rerank_cpu_fallback_cmd(cfg, project_root, gpu_cmd)
        gpu_wait = _sized_wait(model, RERANK_GPU_START_TIMEOUT_SEC, label="rerank GPU start")
        proc = spawn(placement, gpu_cmd, min(timeout, gpu_wait) if fallback else timeout)
        if proc is None and fallback is not None:
            print(
                "[launch] WARNING: rerank server on GPU did not start; restarting it on CPU (-ngl 0). "
                "The saved self-test is kept (a start failure is environmental); candidates were "
                "sized for the GPU, so the deadline may cut some queries to cosine order",
                file=sys.stderr,
            )
            placement = RerankPlacement("cpu", 0, "cpu_fallback_after_gpu_start_failed")
            proc = spawn(placement, fallback)
        if proc is None:
            return RerankLaunch(None, saved, False, "server_unhealthy")
        print(
            f"[launch] rerank ready on :{port} ({placement.kind}, "
            f"{saved.ms_per_doc or 0:.0f} ms/doc, {candidates} candidates at self-test length, "
            f"token budget {st.rerank_token_budget(saved.ms_per_doc, deadline_ms=deadline_ms)}; saved self-test)",
        )
        return RerankLaunch(proc, saved, False, "")

    tune_ok, tune_why = _tune_gate(cfg, project_root, pc, why_test, force=force_selftest)
    if not tune_ok:
        print(
            f"[launch] rerank not started: {tune_why} (the PC changed since the saved self-test and "
            "the re-measure is not confirmed; run `evoref tune --startup-check` or `evoref tune run`)",
        )
        return RerankLaunch(None, None, False, tune_why)
    print(f"[launch] rerank self-test: {why_test} (saved result: {status})")
    mem_ok, mem_detail = decide_rerank_memory(
        (available_memory or st.available_physical_memory_mb)(), _file_size_mb(model),
    )
    if not mem_ok:
        print(f"[launch] WARNING: rerank self-test skipped, {mem_detail}; will retry next start", file=sys.stderr)
        return RerankLaunch(None, None, False, st.LOW_MEMORY)
    measure_fn = measure or st.measure_rerank_server
    budget = st.warm_budget_ms(deadline_ms, min_candidates)
    first = decide_rerank_placement(
        rr.get("gpu_layers", "auto"), _parse_device_memory(devices_text), _file_size_mb(model),
    )
    print(f"[launch] rerank placement: {first.kind} ({first.reason})")
    attempts = [first]
    if first.kind == "gpu" and explicit is None:
        attempts.append(RerankPlacement("cpu", 0, "retry on CPU after the GPU attempt failed"))

    common = {
        "fingerprint": fingerprint,
        "model_key": model_key,
        "model_file": model.name,
        "llama_server_version": server_version(),
        "pc": pc,
    }
    threads = int(rr.get("threads", 0) or 0)
    last = None
    for placement in attempts:
        proc = spawn(placement)
        if proc is None:
            last = st.RerankSelftestResult(
                enabled=False, reason="server_unhealthy", placement=placement.kind,
                gpu_layers=placement.gpu_layers, threads=threads, tested_at=utc_now(), **common,
            )
            continue
        measurement = measure_fn(f"http://{RERANK_HOST}:{port}", warm_budget=budget)
        verdict = st.evaluate_measurement(
            measurement, deadline_ms=deadline_ms,
            max_candidates=max_candidates, min_candidates=min_candidates,
        )
        last = st.RerankSelftestResult(
            enabled=verdict.enabled, reason=verdict.reason, placement=placement.kind,
            gpu_layers=placement.gpu_layers, threads=threads,
            ms_per_doc=verdict.ms_per_doc, candidates=verdict.candidates, margin=verdict.margin,
            scores=list(getattr(measurement, "scores", [])),
            prompt_tokens=getattr(measurement, "prompt_tokens", None),
            tested_at=utc_now(), **common,
        )
        if verdict.enabled:
            if not st.save_selftest_result(path, last):
                print(f"[launch] WARNING: failed to save the rerank self-test to {path}", file=sys.stderr)
            print(
                f"[launch] rerank self-test passed on {placement.kind}: "
                f"{verdict.ms_per_doc or 0:.0f} ms/doc, {verdict.candidates} candidates at self-test length, "
                f"token budget {st.rerank_token_budget(verdict.ms_per_doc, deadline_ms=deadline_ms)}, "
                f"margin {verdict.margin or 0:.2f}",
            )
            return RerankLaunch(proc, last, True, "")
        print(
            f"[launch] WARNING: rerank self-test failed on {placement.kind}: {verdict.reason}",
            file=sys.stderr,
        )
        _stop_proc(proc)
        if not verdict.retry_on_cpu:
            break

    assert last is not None
    # 環境起因かは最後の試行の結果で決める (GPU が環境起因でも CPU が too_slow なら too_slow として数える)
    if st.is_environmental_failure(last.reason) and saved is not None and saved.fingerprint == fingerprint and saved.enabled:
        print(
            f"[launch] rerank disabled for this run: {last.reason} (environmental; kept the previous result "
            "of this PC, not overwritten)",
        )
    elif st.is_retryable_failure(last.reason):
        last.environmental_streak = st.next_environmental_streak(saved, fingerprint)
        if not st.save_selftest_result(path, last):
            print(f"[launch] WARNING: failed to save the rerank self-test to {path}", file=sys.stderr)
        if last.environmental_streak >= st.ENVIRONMENTAL_FAILURE_LIMIT:
            print(
                f"[launch] rerank disabled: {last.reason} ({last.environmental_streak} possibly temporary failures "
                "in a row; not re-testing until the PC changes or --rerank-selftest)",
            )
        else:
            print(
                f"[launch] rerank disabled for this run: {last.reason} "
                f"(possibly temporary failure {last.environmental_streak}/{st.ENVIRONMENTAL_FAILURE_LIMIT}; "
                "will retry next start)",
            )
    else:
        if not st.save_selftest_result(path, last):
            print(f"[launch] WARNING: failed to save the rerank self-test to {path}", file=sys.stderr)
        print(f"[launch] rerank disabled: {last.reason}")
    return RerankLaunch(None, last, True, last.reason)


def wait_rerank_ready(
    cfg: dict, project_root: Path, timeout_sec: float,
    *, sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic,
    list_devices: Callable[[], str] = _list_llama_devices,
) -> str:
    """別プロセスの起動スクリプトが rerank の自己テストを終えるまで待つ (evoref-ctl 用)。

    保存結果の指紋がこの PC と一致し、有効なら ``/health`` が 200 になるまで。mode off /
    モデル無しは即座に返る。環境移行の確認待ち (``pending`` / ``declined``、c_16 §7.2.3) で起動スクリプトが
    測らない回も待たずに ``disabled``。起動スクリプトが起動を断念した印 (:func:`mark_rerank_abandoned`) が
    あれば、その時点で ``disabled`` を返す。戻り値は状態 (``off`` / ``ready`` / ``disabled`` / ``timeout``)。
    """
    ok, _why = rerank_launchable(cfg, project_root)
    path = rerank_selftest_path(cfg, project_root)
    if not ok or path is None:
        return "off"
    try:
        from backend.free.rag import rerank_selftest as st
    except ImportError:
        return "off"
    pc = st.collect_pc_info(_parse_device_names(list_devices()))
    fingerprint = pc.digest
    rr = _rerank_cfg(cfg)
    explicit = _explicit_gpu_layers(rr)
    port = rerank_port(cfg)
    deadline = clock() + timeout_sec
    gate_checked = False
    abandoned = rerank_abandoned_path(cfg, project_root)
    while True:
        if abandoned is not None and abandoned.exists():
            return "disabled"
        saved, _status = st.load_selftest_result(path)
        needed, why = st.selftest_needed(saved, fingerprint, explicit_gpu_layers=explicit)
        if needed and not gate_checked:
            # 確認待ち (pending / declined) の間は起動スクリプトが測らないので待たない (c_16 §7.2.3)
            gate_checked = True
            if not _tune_gate(cfg, project_root, pc, why)[0]:
                return "disabled"
        if saved is not None and not needed:
            if not _usable_saved(st, saved, rr)[0]:
                return "disabled"
            try:
                if httpx.get(f"http://{RERANK_HOST}:{port}/health", timeout=1.0).status_code == 200:
                    return "ready"
            except (httpx.ConnectError, httpx.TimeoutException):
                pass
        if clock() >= deadline:
            return "timeout"
        sleep(1.0)


# ── 埋め込みサーバの配置 (GPU / CPU) の判別と起動 (c_16 §7.2.2) ─────
# 判別は埋め込みサーバを起こす持ち主のプロセス (この起動スクリプト / evoref serve / auto-serve) だけが
# :func:`start_embed_server` から走らせる。一時ポートで CPU と GPU を起こして測り、本番のポートは
# 触らない。再起動の経路と backend は保存結果を読むだけ。GPU 配置の本番サーバが起動しなければ、
# どの経路も :func:`embed_cpu_fallback_cmd` で CPU に 1 回だけ起こし直す (埋め込みは必須機能)。

EMBED_PROBE_HOST = "127.0.0.1"
#: GPU 候補の判定で、埋め込みモデルのサイズに足す計算バッファの見積り (MiB)。
EMBED_COMPUTE_MARGIN_MIB = 1024
#: 判別の一時サーバの health 待ち (秒)。ハングした GPU で起動時間を膨らませない。
EMBED_PROBE_HEALTH_TIMEOUT_SEC = 30
#: GPU 配置の本番サーバの health 待ち (秒)。超えたら CPU で起こし直す。
EMBED_GPU_START_TIMEOUT_SEC = 60


@dataclass(frozen=True)
class EmbedPlacement:
    """埋め込みサーバの配置。``kind`` は ``gpu`` / ``cpu``、``reason`` は根拠 (英語の識別子)。"""

    kind: str
    gpu_layers: int
    reason: str = ""


def embed_placement_path(cfg: dict, project_root: Path) -> Path | None:
    """判別結果の置き場 (``PathResolver.LAYOUT["embed_placement_file"]``)。"""
    return _data_path(cfg, project_root, "embed_placement_file")


def _embed_setting_is_auto(cfg: dict) -> bool:
    return (cfg.get("embedding") or {}).get("gpu_layers") == "auto"


def _embed_model_key(cfg: dict, project_root: Path) -> str | None:
    """埋め込みモデルの ``model_key`` (ファイルが無い / backend が無ければ ``None``)。"""
    model = _embed_model_path(cfg, project_root)
    if not model.is_file():
        return None
    try:
        from backend.model_key import model_key_for
    except ImportError:
        return None
    return model_key_for(model)


def _placement_from_status(status) -> EmbedPlacement:
    return EmbedPlacement(status.placement, int(status.gpu_layers), status.reason)


def saved_embed_placement(cfg: dict, project_root: Path | None) -> EmbedPlacement:
    """設定と保存結果から配置を決める (判別しない。再起動の経路・VRAM 見積り・``build_embed_cmd`` 用)。

    backend と同じ純関数 ``embed_placement.resolve_embed_placement_status`` を使う。``auto`` で
    結果が無い / 読めない / PC が違う (GPU 名は比べない) / 埋め込みモデルが違うなら CPU。
    """
    raw = (cfg.get("embedding") or {}).get("gpu_layers")
    try:
        from backend.free.rag import embed_placement as ep
        from backend.free.rag.rerank_selftest import collect_pc_info
    except ImportError:
        ngl = 0 if raw is None or raw == "auto" else int(raw)
        return EmbedPlacement("gpu" if ngl != 0 else "cpu", ngl, "backend not importable")
    if ep.gpu_layers_setting(raw) != "auto":
        return _placement_from_status(ep.resolve_embed_placement_status(raw, None))
    path = embed_placement_path(cfg, project_root) if project_root is not None else None
    saved = ep.load_placement_result(path)[0] if path is not None else None
    model_key = _embed_model_key(cfg, project_root) if saved is not None and project_root is not None else None
    return _placement_from_status(ep.resolve_embed_placement_status(
        raw, saved, current_pc=collect_pc_info([]), current_model_key=model_key,
    ))


def embed_cpu_fallback_cmd(
    cfg: dict, cmd: list[str], project_root: Path | None = None,
) -> list[str] | None:
    """GPU 配置の埋め込みサーバが起動しなかったときに起こし直す CPU (``-ngl 0``) のコマンド。

    ``embedding.gpu_layers: auto`` で ``-ngl`` が 0 以外のときだけ (明示の整数は利用者の選択なので
    そのまま。リランカーの CPU 再試行と同じ)。それ以外は ``None``。保存結果は書き換えない
    (起動の失敗は環境起因)。全部の本番起動経路 (--all / --embed / serve / auto-serve /
    server_control / LlamaProcessManager) がこの 1 実装を使う。

    コマンドは ``-ngl`` だけを差し替えず、CPU 配置として **組み直す** (:func:`_embed_cmd` の ``gpu_layers=0``。
    リランカーの :func:`rerank_cpu_fallback_cmd` と同じ): GPU 配置の ``-t 2`` (GPU_AUX_THREADS) と VRAM で
    広げた ``-ub`` を CPU に持ち込むと、CPU の埋め込みが 2 スレッドで詰まりクエリが timeout する。
    ポートは ``cmd`` の ``--port``。``project_root`` が無ければ backend のプロジェクト根 (無ければ CWD)。
    """
    if not _embed_setting_is_auto(cfg) or "-ngl" not in cmd:
        return None
    i = cmd.index("-ngl") + 1
    if i >= len(cmd) or cmd[i] == "0":
        return None
    if project_root is None:
        try:
            from backend.config import get_project_root

            project_root = Path(get_project_root())
        except Exception:  # noqa: BLE001 - backend が無い / 読めない配布物は CWD (build_embed_cmd と同じ)
            project_root = Path.cwd()
    port_at = cmd.index("--port") + 1 if "--port" in cmd else -1
    port = int(cmd[port_at]) if 0 < port_at < len(cmd) else int((cfg.get("embedding") or {}).get("llama_port", 8082))
    return _embed_cmd(cfg, project_root, port=port, gpu_layers=0)


def decide_embed_gpu_candidate(
    device_memory: dict[str, tuple[int, int]],
    model_mb: int | None,
    *,
    reserve_mib: int = 0,
    compute_margin_mib: int = EMBED_COMPUTE_MARGIN_MIB,
) -> tuple[bool, str]:
    """GPU で測る価値があるか (純関数)。GPU の空きの最大 ≥ モデル + 計算バッファ + ``reserve_mib``。

    ``reserve_mib`` はまだ載っていない base の見積り (先に埋め込みが VRAM を取って base が OOM
    しないように)。理由: ``no_gpu_device`` / ``model_size_unknown`` / ``gpu_memory_short: ...``。
    判定の式はリランカーと共有する (:func:`gpu_fit`)。
    """
    fit = gpu_fit(device_memory, model_mb, margin_mib=compute_margin_mib, reserve_mib=reserve_mib)
    if fit.status == "no_gpu":
        return False, "no_gpu_device"
    if fit.status == "unknown_size":
        return False, "model_size_unknown"
    detail = (
        f"{fit.device} free {fit.free_mib} MiB, need {fit.need_mib} MiB "
        f"(reserve {max(0, reserve_mib)} MiB for base)"
    )
    if fit.status == "fits":
        return True, detail
    return False, f"gpu_memory_short: {detail}"


def embed_gpu_within_budget(cfg: dict, project_root: Path, embed_model_mb: int | None) -> tuple[bool, str]:
    """埋め込みを GPU に見積もっても ``runtime.total_vram_budget_mb`` に収まるか (未設定なら常に可)。

    ``--all`` の予算検査 (:func:`check_vram_budget`) は判別の前に走り、未判別の埋め込みを CPU と
    見積もるので、判別する回はここで GPU 側に見積もり直す。理由: ``vram_budget_exceeded: ...``。
    """
    budget, _note = _vram_budget_mb(cfg, project_root)
    if budget is None:
        return True, ""
    try:
        estimates = _estimate_via_gguf_size(cfg, project_root)
    except (OSError, ValueError, KeyError):
        return True, ""
    others = sum(int(e.get("vram_mb") or 0) for name, e in estimates.items() if name != "embed")
    total = others + int(embed_model_mb or 0)
    if total > int(budget):
        return False, f"vram_budget_exceeded: {total} MB with the embed on GPU > budget {int(budget)} MB"
    return True, ""


def _base_ready(cfg: dict) -> bool:
    """base の llama-server が ``/health`` 200 を返すか (載り終わっていれば空きに反映済み)。"""
    lc = cfg.get("llama", {}) or {}
    url = f"http://{lc.get('host', '127.0.0.1')}:{lc.get('port', 8080)}/health"
    try:
        return httpx.get(url, timeout=1.0).status_code == 200
    except httpx.HTTPError:
        return False


def _wait_base_ready(
    cfg: dict, timeout_sec: float, base_ready: Callable[[dict], bool],
    *, sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic,
) -> bool:
    """base が載り終わるまで待つ (判別する回だけ。空きと p50 を base の読み込みと競合させない)。"""
    deadline = clock() + max(0.0, timeout_sec)
    while True:
        if base_ready(cfg):
            return True
        if clock() >= deadline:
            return False
        sleep(1.0)


def _base_reserve_mib(cfg: dict, project_root: Path, base_ready: Callable[[dict], bool]) -> int:
    """まだ載っていない base の VRAM 見積り (Tier 2)。載り終わっていれば 0。"""
    if base_ready(cfg):
        return 0
    try:
        return int(_estimate_via_gguf_size(cfg, project_root)["base"]["vram_mb"] or 0)
    except (OSError, ValueError, KeyError):
        return 0


def _free_port() -> int:
    """OS に空きポートを 1 つ選ばせる (判別の一時サーバ用)。"""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((EMBED_PROBE_HOST, 0))
        return int(sock.getsockname()[1])


def ensure_embed_placement(
    cfg: dict,
    project_root: Path,
    *,
    force: bool = False,
    wait_base_sec: float = 0.0,
    popen: Callable[..., subprocess.Popen] = subprocess.Popen,
    list_devices: Callable[[], str] = _list_llama_devices,
    health: Callable[..., bool] = _wait_healthy_or_dead,
    measure: Callable[..., object] | None = None,
    server_version: Callable[[], str] = _llama_server_version,
    free_port: Callable[[], int] = _free_port,
    base_ready: Callable[[dict], bool] = _base_ready,
    probe_timeout: int | None = None,
) -> EmbedPlacement:
    """``embedding.gpu_layers: auto`` の配置を返す。PC か埋め込みモデルが変わったときだけ判別する (c_16 §7.2.2)。

    - ``null`` / 整数 → 設定どおり (判別しない・保存結果も見ない)。
    - 指紋とモデルが保存結果と一致 (GPU の起動失敗を数えている途中でない) → 保存結果の配置。
    - それ以外 / ``force`` → GPU デバイスが無ければ測らずに CPU (保存)。あれば base が載り終わるのを
      ``wait_base_sec`` まで待ち、空き・予算で GPU 候補か決め、候補なら一時ポートで CPU (``-ngl 0``) と
      GPU (``-ngl 999``) を起こして測る。空きの不足・予算超過は状態なので保存しない (測らないので
      次の起動の費用は ``--list-devices`` だけ)。GPU の一時サーバが起動しないのは
      ``GPU_UNHEALTHY_LIMIT`` 回続くまで数えるだけ。どの失敗もチャットを止めない (CPU で返す)。
    - PC が変わった / 結果が無い回は判別の前に環境移行の確認 (:func:`_tune_gate`、c_16 §7.2.3) を見て、
      確認待ち (``pending`` / ``declined``) なら測らずに CPU (理由 ``tune_pending`` / ``tune_declined``)。
      ``force`` は常に測り、``accepted`` を記録する。
    """
    raw = (cfg.get("embedding") or {}).get("gpu_layers")
    try:
        from backend.free.rag import embed_placement as ep
        from backend.free.rag import rerank_selftest as st
        from backend.utils import utc_now
    except ImportError as e:
        print(f"[launch] WARNING: embed placement not decided (backend is not importable: {e})", file=sys.stderr)
        return saved_embed_placement(cfg, project_root)
    if ep.gpu_layers_setting(raw) != "auto":
        return saved_embed_placement(cfg, project_root)
    path = embed_placement_path(cfg, project_root)
    if path is None:
        print("[launch] WARNING: embed placement not decided (data root is not resolvable); using CPU", file=sys.stderr)
        return EmbedPlacement("cpu", 0, "data_root_unresolvable")
    model = _embed_model_path(cfg, project_root)
    model_key = _embed_model_key(cfg, project_root)
    if model_key is None:
        print(f"[launch] WARNING: embed model not found ({model}); embed placement not decided", file=sys.stderr)
        return EmbedPlacement("cpu", 0, "model_not_found")

    devices_text = list_devices()
    pc = st.collect_pc_info(_parse_device_names(devices_text))
    fingerprint = pc.digest
    saved, status = ep.load_placement_result(path)
    needed, why = ep.placement_needed(saved, fingerprint, force=force, model_key=model_key)
    if not needed and saved is not None:
        print(f"[launch] embed placement: {saved.placement} (saved: {saved.reason})")
        return EmbedPlacement(saved.placement, saved.gpu_layers, saved.reason)

    tune_ok, tune_why = _tune_gate(cfg, project_root, pc, why, force=force)
    if not tune_ok:
        print(
            f"[launch] embed placement not measured: {tune_why} (the PC changed and the re-measure is not "
            "confirmed); the embedding server runs on CPU. Run `evoref tune --startup-check` or `evoref tune run`",
        )
        return EmbedPlacement("cpu", 0, tune_why)
    print(f"[launch] embed placement: deciding ({why}; saved result: {status})")
    emb_cfg = cfg.get("embedding", {}) or {}
    dim = int(emb_cfg.get("dim", 1024))
    model_mb = _file_size_mb(model)
    if probe_timeout is None:
        probe_timeout = _sized_wait(model, EMBED_PROBE_HEALTH_TIMEOUT_SEC, label="embed probe")
    device_memory = _parse_device_memory(devices_text)
    has_gpu = gpu_fit(device_memory, model_mb, margin_mib=0).status != "no_gpu"
    if has_gpu and wait_base_sec > 0 and not base_ready(cfg):
        print(f"[launch] embed placement: waiting for the base model to load (up to {wait_base_sec:.0f}s)")
        _wait_base_ready(cfg, wait_base_sec, base_ready)
        device_memory = _parse_device_memory(list_devices())
    reserve = _base_reserve_mib(cfg, project_root, base_ready) if has_gpu else 0
    gpu_ok, candidate_reason = decide_embed_gpu_candidate(device_memory, model_mb, reserve_mib=reserve)
    if gpu_ok:
        gpu_ok, budget_reason = embed_gpu_within_budget(cfg, project_root, model_mb)
        candidate_reason = budget_reason or candidate_reason
    print(f"[launch] embed placement: GPU candidate={gpu_ok} ({candidate_reason})")
    measure_fn = measure or ep.measure_embed_server

    def run_probe(gpu_layers: int):
        port = free_port()
        cmd = _embed_cmd(cfg, project_root, port=port, gpu_layers=gpu_layers)
        print(f"[launch] embed probe (-ngl {gpu_layers}) on :{port}")
        try:
            proc = popen(cmd, cwd=project_root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (FileNotFoundError, OSError) as e:
            print(f"[launch] WARNING: failed to spawn the embed probe server: {e}", file=sys.stderr)
            return ep.EmbedProbe(error=ep.SERVER_UNHEALTHY)
        try:
            if not health(proc, EMBED_PROBE_HOST, port, probe_timeout, model.name):
                return ep.EmbedProbe(error=ep.SERVER_UNHEALTHY)
            return measure_fn(f"http://{EMBED_PROBE_HOST}:{port}")
        finally:
            _stop_proc(proc)

    cpu = run_probe(0) if gpu_ok else None
    gpu = run_probe(999) if gpu_ok else None
    decision = ep.decide_embed_placement(
        cpu, gpu, gpu_candidate_reason=candidate_reason, expected_dim=dim,
        prior_unhealthy_streak=ep.prior_unhealthy_streak(saved, fingerprint, model_key),
    )
    placement = EmbedPlacement(decision.placement, decision.gpu_layers, decision.reason)
    cos = f"{decision.cosine_min:.6f}" if decision.cosine_min is not None else "-"
    streak = f", GPU start failures {decision.gpu_unhealthy_streak}" if decision.gpu_unhealthy_streak else ""
    print(
        f"[launch] embed placement: {decision.placement} ({decision.reason}; p50 cpu "
        f"{decision.cpu_p50_ms or 0:.0f} ms / gpu {decision.gpu_p50_ms or 0:.0f} ms, cosine min {cos}{streak})",
    )
    if not decision.save:
        print("[launch] embed placement not saved (a state, not a PC property); will decide again next start")
        return placement
    result = ep.EmbedPlacementResult(
        fingerprint=fingerprint, placement=decision.placement, gpu_layers=decision.gpu_layers,
        reason=decision.reason, cpu_p50_ms=decision.cpu_p50_ms, gpu_p50_ms=decision.gpu_p50_ms,
        cosine_min=decision.cosine_min, dim=dim, decided_at=utc_now(),
        gpu_unhealthy_streak=decision.gpu_unhealthy_streak,
        model_key=model_key, model_file=model.name,
        llama_server_version=server_version(), pc=pc,
    )
    if not ep.save_placement_result(path, result):
        print(f"[launch] WARNING: failed to save the embed placement to {path}", file=sys.stderr)
    # 一時サーバの引数を決めたときの埋め込みの調整値は判別前の材料なので、本番の起動で見積り直す
    _forget_tuned(cfg, "embed_params")
    return placement


@dataclass
class EmbedLaunch:
    """:func:`start_embed_server` の結果。``proc`` は起動したまま残した本番の埋め込みサーバ。"""

    proc: subprocess.Popen | None
    placement: EmbedPlacement
    healthy: bool
    #: GPU で起動できず CPU に起こし直した。
    fell_back: bool = False


def start_embed_server(
    cfg: dict,
    project_root: Path,
    *,
    popen_kwargs: dict | None = None,
    wait_base_sec: float = 0.0,
    health_timeout: int | None = None,
    popen: Callable[..., subprocess.Popen] = subprocess.Popen,
    health: Callable[..., bool] = _wait_healthy_or_dead,
    placement_fn: Callable[..., EmbedPlacement] | None = None,
) -> EmbedLaunch | None:
    """埋め込みサーバを本番のポートで起動する (持ち主のプロセスの入口。``embedding.backend`` が
    llama-cpp でなければ ``None``)。

    ``auto`` なら先に配置を判別し (:func:`ensure_embed_placement`、PC かモデルが変わったときだけ)、
    GPU 配置のサーバが即死 / ``EMBED_GPU_START_TIMEOUT_SEC`` 以内に health を通らなければ止めて
    CPU (``-ngl 0``) で 1 回だけ起こし直す (:func:`embed_cpu_fallback_cmd`)。保存結果は書き換えない。
    最後の試行は health を通らなくてもプロセスを残す (従来どおり、呼び手の health 待ちに任せる)。
    """
    emb_cfg = cfg.get("embedding", {}) or {}
    if emb_cfg.get("backend", "llama-cpp") != "llama-cpp":
        return None
    placement = (placement_fn or ensure_embed_placement)(cfg, project_root, wait_base_sec=wait_base_sec)
    port = int(emb_cfg.get("llama_port", 8082))
    host = emb_cfg.get("llama_host", "localhost")
    model_name = _embed_model_path(cfg, project_root).name
    embed_model = _embed_model_path(cfg, project_root)
    timeout = health_timeout or _health_wait(cfg, embed_model, project_root, label="embedding")
    gpu_wait = _sized_wait(embed_model, EMBED_GPU_START_TIMEOUT_SEC, label="embedding GPU start")
    kwargs = {"cwd": project_root, **(popen_kwargs or {})}

    cmd = _embed_cmd(cfg, project_root, port=port, gpu_layers=placement.gpu_layers)
    fallback = embed_cpu_fallback_cmd(cfg, cmd, project_root)
    attempts = [(cmd, placement)]
    if fallback is not None:
        attempts.append((fallback, EmbedPlacement("cpu", 0, "cpu_fallback_after_gpu_start_failed")))
    proc = None
    for i, (attempt_cmd, attempt) in enumerate(attempts):
        last = i == len(attempts) - 1
        print(f"[launch] embedding ({attempt.kind}): {' '.join(attempt_cmd)}")
        try:
            proc = popen(attempt_cmd, **kwargs)
        except (FileNotFoundError, OSError) as e:
            print(f"[launch] WARNING: failed to spawn the embedding server: {e}", file=sys.stderr)
            proc = None
            continue
        wait = timeout if last else min(timeout, gpu_wait)
        if health(proc, host, port, wait, model_name):
            print(f"[launch] embedding is ready on :{port} ({attempt.kind})")
            return EmbedLaunch(proc, attempt, True, fell_back=i > 0)
        if not last:
            print(
                f"[launch] WARNING: embedding server on GPU did not become healthy within {wait}s; "
                "restarting it on CPU (-ngl 0). The saved placement is kept (a start failure is environmental)",
                file=sys.stderr,
            )
            _stop_proc(proc)
            continue
        print(f"[launch] WARNING: embedding server ({attempt.kind}) health check timed out", file=sys.stderr)
        return EmbedLaunch(proc, attempt, False, fell_back=i > 0)
    return EmbedLaunch(proc, attempts[-1][1], False, fell_back=len(attempts) > 1)


#: 環境移行の確認待ちで配置を判別しない回の埋め込みの health 待ち (秒)。判別の分 (240 秒) は待たず、
#: 他のサーバと同じ 60 秒にする (CPU の本番サーバは起こすので、backend より先に待つ意味は残る)。
EMBED_UNTUNED_WAIT_SEC = 60.0


def _embed_tune_blocked(cfg: dict, project_root: Path, list_devices: Callable[[], str]) -> bool:
    """埋め込みの配置を確認待ち (``pending`` / ``declined``) のため判別しない回か (c_16 §7.2.3)。"""
    if not _embed_setting_is_auto(cfg):
        return False
    path = embed_placement_path(cfg, project_root)
    try:
        from backend.free.rag import embed_placement as ep
        from backend.free.rag import rerank_selftest as st
    except ImportError:
        return False
    if path is None:
        return False
    pc = st.collect_pc_info(_parse_device_names(list_devices()))
    saved, _status = ep.load_placement_result(path)
    needed, why = ep.placement_needed(saved, pc.digest, model_key=_embed_model_key(cfg, project_root))
    return needed and not _tune_gate(cfg, project_root, pc, why)[0]


def _embed_total_wait(cfg: dict, project_root: Path) -> int:
    """evoref-ctl が埋め込みの準備を待つ最長 = 判別プローブ 2 回 + GPU 起動待ち + CPU 起こし直しの待ち。

    各項は :func:`start_embed_server` と同じ関数で導く (下限どうしの和は従来の 240 秒)。
    """
    model = _embed_model_path(cfg, project_root)
    return (
        2 * _sized_wait(model, EMBED_PROBE_HEALTH_TIMEOUT_SEC, label="embed probe")
        + _sized_wait(model, EMBED_GPU_START_TIMEOUT_SEC, label="embedding GPU start")
        + _health_wait(cfg, model, project_root, label="embedding")
    )


def _rerank_total_wait(cfg: dict, project_root: Path) -> int:
    """evoref-ctl が rerank の準備を待つ最長 = GPU 起動待ち + CPU 起こし直しの待ち + 自己テスト本体の余裕。

    下限どうしの和は従来の 240 秒。
    """
    model = resolve_rerank_model_path(cfg, project_root)
    return (
        _sized_wait(model, RERANK_GPU_START_TIMEOUT_SEC, label="rerank GPU start")
        + _health_wait(cfg, model, project_root, label="rerank")
        + RERANK_SELFTEST_ALLOWANCE_SEC
    )


def wait_embed_ready(
    cfg: dict, timeout_sec: float,
    *, sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic,
    project_root: Path | None = None, list_devices: Callable[[], str] = _list_llama_devices,
) -> str:
    """埋め込みサーバの ``/health`` が 200 になるまで待つ (evoref-ctl 用。初回の判別と CPU の再起動を含めて待つ)。

    ``project_root`` を渡すと、環境移行の確認待ち (``pending`` / ``declined``、c_16 §7.2.3) で判別しない回は
    待ちを :data:`EMBED_UNTUNED_WAIT_SEC` までに縮める (判別の分の 240 秒を空費しない)。
    戻り値は ``off`` (llama-cpp でない) / ``ready`` / ``timeout``。
    """
    emb = cfg.get("embedding", {}) or {}
    if emb.get("backend", "llama-cpp") != "llama-cpp":
        return "off"
    if project_root is not None and _embed_tune_blocked(cfg, project_root, list_devices):
        print("[launch] embedding: the auto-tune is not confirmed, so no placement probe runs; waiting for the CPU server only")
        timeout_sec = min(timeout_sec, EMBED_UNTUNED_WAIT_SEC)
    url = f"http://{emb.get('llama_host', 'localhost')}:{int(emb.get('llama_port', 8082))}/health"
    deadline = clock() + timeout_sec
    while True:
        try:
            if httpx.get(url, timeout=1.0).status_code == 200:
                return "ready"
        except httpx.HTTPError:
            pass
        if clock() >= deadline:
            return "timeout"
        sleep(1.0)


if __name__ == "__main__":
    import argparse

    # 配置サマリ等に含まれる非 ASCII 記号 (em-dash 等) は、標準出力がファイルへ
    # リダイレクトされた Windows 環境では cp932 に落ちて UnicodeEncodeError で
    # 落ちる。**サーバを 1 つも起動しないまま**死ぬので、表示は best-effort に倒す。
    for _stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError):
            _stream.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(description="Launch llama-server instances")
    parser.add_argument("config", nargs="?", default="config.yaml", help="Config file path")
    parser.add_argument("--all", action="store_true", help="Launch all configured servers")
    parser.add_argument("--embed", action="store_true", help="Launch embedding server only")
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "VRAM 予算超過時も強制起動する。--all と併用することを想定"
        ),
    )
    parser.add_argument(
        "--lora", default=None, metavar="PATH",
        help="base に当てる LoRA (呼出元が解決・検証済み。backend.free.cli.llama_launcher が渡す)",
    )
    parser.add_argument(
        "--control-vector", default=None, metavar="PATH",
        help="base に当てる control vector (呼出元が解決・検証済み)",
    )
    parser.add_argument(
        "--print-health-ports",
        action="store_true",
        help=(
            "起動せず、--all で実際に立ち上がるサーバの 'name=port' を 1 行ずつ "
            "出力して終了する。evoref-ctl が health 待ち対象を決めるのに使う"
        ),
    )
    parser.add_argument(
        "--wait-health",
        type=float,
        default=None,
        metavar="SECONDS",
        help=(
            "起動せず、--all で立ち上がるサーバの /health が 200 を返すまで待って終了する "
            "(evoref-ctl の health 待ち。以前は powershell のループで、標準入力が"
            "端末でない呼出元からは 'Input redirection is not supported' で落ちた)"
        ),
    )
    parser.add_argument(
        "--rerank-selftest",
        action="store_true",
        help=(
            "リランカーの自己テストを手動で測り直す (c_16 §7.2.1)。rerank サーバを起動して測り、"
            "結果を cache/rerank_selftest.json に書いて止める。rerank のポートが使用中なら拒否する "
            "(先に evoref-ctl stop)"
        ),
    )
    parser.add_argument(
        "--embed-placement",
        action="store_true",
        help=(
            "埋め込みサーバの配置 (GPU / CPU) を手動で判別し直す (c_16 §7.2.2、embedding.gpu_layers: auto の"
            "ときだけ)。一時ポートで CPU と GPU を起こして測り、cache/embed_placement.json に書いて終わる"
        ),
    )
    parser.add_argument(
        "--wait-embed",
        type=float,
        default=None,
        metavar="SECONDS",
        help=(
            "起動せず、別プロセスの --all が立てる埋め込みサーバの /health が 200 になるまで待って終了する "
            "(evoref-ctl 用。初回の配置の判別と GPU 失敗時の CPU 再起動を含めて待つ)"
        ),
    )
    parser.add_argument(
        "--wait-rerank",
        type=float,
        default=None,
        metavar="SECONDS",
        help=(
            "起動せず、別プロセスの --all が rerank の自己テストを終えて (有効なら /health 200 まで) "
            "準備できるまで待って終了する (evoref-ctl 用。rerank が off なら即終了)"
        ),
    )
    parser.add_argument(
        "--clear-rerank-abandoned",
        action="store_true",
        help=(
            "前回の rerank 起動断念の印 (run/rerank_abandoned) を消して終了する (evoref-ctl 用。"
            "--all を背後で起こす前に呼び、--wait-rerank が古い印で待ちを打ち切らないようにする)"
        ),
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"[launch] ERROR: config file not found: {cfg_path}")
        sys.exit(1)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    project_root = cfg_path.parent

    health_ports: dict[str, int] = {"base": int(cfg.get("llama", {}).get("port", 8080))}
    _emb = cfg.get("embedding", {}) or {}
    if _emb.get("backend", "llama-cpp") == "llama-cpp":
        health_ports["embed"] = int(_emb.get("llama_port", 8082))

    if args.print_health_ports:
        for name, port in health_ports.items():
            print(f"{name}={port}")
        sys.exit(0)

    if args.wait_health is not None:
        _wait_health(health_ports, float(args.wait_health))
        sys.exit(0)

    if args.clear_rerank_abandoned:
        clear_rerank_abandoned(cfg, project_root)
        sys.exit(0)

    if args.wait_rerank is not None:
        # 引数は下限。大型モデルでは起動の待ちの導出値 (GGUF サイズ連動) まで延ばす
        state = wait_rerank_ready(
            cfg, project_root, max(float(args.wait_rerank), _rerank_total_wait(cfg, project_root)),
        )
        if state == "timeout":
            print("[launch] WARNING: rerank self-test / health did not finish in time, proceeding anyway")
        elif state != "off":
            print(f"[launch] rerank: {state}")
        sys.exit(0)

    if args.wait_embed is not None:
        state = wait_embed_ready(
            cfg, max(float(args.wait_embed), _embed_total_wait(cfg, project_root)), project_root=project_root,
        )
        if state == "timeout":
            print("[launch] WARNING: embedding server did not become healthy in time, proceeding anyway")
        elif state != "off":
            print(f"[launch] embedding: {state}")
        sys.exit(0)

    if args.embed_placement:
        decided = ensure_embed_placement(cfg, project_root, force=True)
        print(f"[launch] embed placement: {decided.kind} (-ngl {decided.gpu_layers}, {decided.reason})")
        sys.exit(0)

    if args.rerank_selftest:
        launched = start_rerank_server(cfg, project_root, force_selftest=True)
        if launched.proc is not None:
            _stop_proc(launched.proc)
        sys.exit(0 if launched.result is not None else 1)

    procs: list[subprocess.Popen] = []
    if args.all:
        # ctl の --wait-rerank は base / embed の health 待ちの後に始まる。それより前 (base を起こす前) に
        # 前回の起動断念の印を消し、今回の rerank を始める前に古い印で待ちを打ち切らせない
        clear_rerank_abandoned(cfg, project_root)

    launch_base = not args.embed
    launch_embed = args.embed or args.all

    # llama-server バージョン検査。--all / 個別起動を問わず
    # llama-server バイナリを起動する全パスで一度だけ build 番号を確認する。
    # ベース / 埋め込みは同一バイナリを使うため、
    # 起動前に 1 回プローブして INFO ログに出力する。
    build_ok, _detected, _required, build_messages = check_llamacpp_build(cfg)
    for line in build_messages:
        if "WARNING" in line or "aborting" in line:
            print(line, file=sys.stderr)
        else:
            print(line)
    if not build_ok:
        if args.all:
            _mark_rerank_abandoned_if_launchable(cfg, project_root, "llama_server_build_check_failed")
        sys.exit(3)

    # VRAM 予算検査は --all (全モデル一括起動) 時のみ実行する。
    # 個別起動 (--embed) では既存プロセスとの
    # 合算が読み取れないため検査をスキップする。
    if args.all:
        ok, _total, _budget, _estimates, message = check_vram_budget(
            cfg, project_root, force=args.force,
        )
        print(message)
        if not ok:
            _mark_rerank_abandoned_if_launchable(cfg, project_root, "vram_budget_exceeded")
            sys.exit(2)

    try:
        # ベースモデル
        if launch_base:
            cmd = build_llama_cmd(
                cfg, project_root,
                lora_override=args.lora,
                control_vector_override=args.control_vector,
            )
            host = cfg["llama"].get("host", "localhost")
            port = cfg["llama"].get("port", 8080)
            # health 待ちは process_manager.health_timeout (整数は明示値、auto / 無しは GGUF サイズ連動)
            procs.append(_start_and_wait(
                cmd, "base-model", host, port, cwd=project_root,
                expected_model_id=_extract_model_basename(cmd),
                timeout=_health_wait(cfg, _model_path_of_cmd(cmd), project_root, label="base"),
            ))

        # エンベッド
        if launch_embed:
            # 注意: -ngl 0 (CPU モード) でも llama.cpp は Vulkan バックエンドを
            # 初期化し model loading 時に host (pinned) buffer を要求する。
            # buffer size が Vulkan device->max_buffer_size を超えると warning
            # (ggml_vulkan: Failed to allocate pinned memory) が出るが CPU buffer に
            # 自動フォールバックするため機能影響は無い (ggml-vulkan.cpp:14079 / :2671)。
            # auto なら配置を判別し (base は上で health 済み)、GPU で起動しなければ CPU で起こし直す。
            embed = start_embed_server(cfg, project_root)
            if embed is None:
                print("[launch] Embedding backend is not llama-cpp, skipping")
            elif embed.proc is not None:
                procs.append(embed.proc)

        # リランカー (rag.rerank.mode が off 以外のときだけ)。自己テストは PC が変わったときだけ。
        # 起動・自己テストの失敗は他のサーバを止めない (rerank を無効にして続ける)。
        if args.all:
            rerank = start_rerank_server(cfg, project_root)
            if rerank.proc is not None:
                procs.append(rerank.proc)

        if procs:
            for proc in procs:
                proc.wait()
    except KeyboardInterrupt:
        for proc in procs:
            proc.terminate()
        for proc in procs:
            proc.wait()
