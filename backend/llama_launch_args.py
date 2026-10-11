"""llama-server の起動オプション文字列 (``launch_args``) の解釈 — 起動スクリプトと backend で共有する 1 実装。

``llama.launch_args`` / ``embedding.launch_args`` / ``rag.rerank.launch_args`` は利用者が書く
llama-server の引数 (例 ``-c 8192 -ngl 99 -fa on``)。規則 (c_16 §7.2.4):

- 自動で組み立てたコマンドから、利用者が指定したのと同じフラグ (短縮 / 長形 / ``--no-`` 形は同じものと
  みなす) を値ごと除いてから、利用者の引数を末尾に足す (後勝ちに頼らない)。
- 環境調整 (c_16 §7.2.3) と backend の読み手が見る値 (``-c`` → ``context_size`` 等) は設定のキーへ
  写す (:func:`derive_values`)。写した項目は明示値と同じ扱いになり、環境調整はその項目を決めない。
- ``-m`` / ``--port`` / ``--host`` は拒否する (モデルのパスとポートは別の設定が正本)。
- 設定 API (GUI) から書けるのは :func:`api_disallowed_flags` が通すフラグだけ (性能・メモリの調整)。
  ファイル・ネットワークに触れるフラグは config.yaml を手で編集する (c_06 §1.5 の能力キーと同じ扱い)。

標準ライブラリだけに依存する (``scripts/launch_llama.py`` からも import する)。
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Callable

__all__ = [
    "LAUNCH_ARGS_SECTIONS",
    "LaunchArgsError",
    "api_disallowed_flags",
    "apply_launch_args",
    "canonical_flag",
    "derive_into",
    "derive_values",
    "derived_keys",
    "merge_launch_args",
    "parse_launch_args",
]


class LaunchArgsError(ValueError):
    """``launch_args`` を解釈できない (引用符の閉じ忘れ / 予約フラグ / 値の型違い)。"""


#: ``launch_args`` を持つ設定の節 (種別 → config dict 内の位置)。
LAUNCH_ARGS_SECTIONS: dict[str, tuple[str, ...]] = {
    "llama": ("llama",),
    "embedding": ("embedding",),
    "rerank": ("rag", "rerank"),
}

#: 同じ意味のフラグの別名。先頭が正規名。``--no-X`` は ``--X`` と同じ組 (:func:`canonical_flag`)。
_ALIAS_GROUPS: tuple[tuple[str, ...], ...] = (
    ("-m", "--model"),
    ("-c", "--ctx-size"),
    ("-ngl", "--gpu-layers", "--n-gpu-layers"),
    ("-t", "--threads"),
    ("-tb", "--threads-batch"),
    ("-b", "--batch-size"),
    ("-ub", "--ubatch-size"),
    ("-fa", "--flash-attn"),
    ("-np", "--parallel"),
    ("-ctk", "--cache-type-k"),
    ("-ctv", "--cache-type-v"),
    ("-cram", "--cache-ram"),
    ("-kvu", "--kv-unified"),
    ("--ctx-checkpoints", "--swa-checkpoints"),
    ("-rea", "--reasoning"),
    ("-ot", "--override-tensor"),
    ("-cmoe", "--cpu-moe"),
    ("-ncmoe", "--n-cpu-moe"),
    ("-cb", "--cont-batching"),
    ("-nocb", "--no-cont-batching"),
    ("-dev", "--device"),
    ("-sm", "--split-mode"),
    ("-ts", "--tensor-split"),
    ("-mg", "--main-gpu"),
    ("-nkvo", "--no-kv-offload"),
    ("-kvo", "--kv-offload"),
    ("-dt", "--defrag-thold"),
    ("-n", "--n-predict"),
    ("-to", "--timeout"),
)
_CANONICAL: dict[str, str] = {alias: group[0] for group in _ALIAS_GROUPS for alias in group}
#: ``-nocb`` / ``-nkvo`` は ``--no-`` 形の短縮。肯定形と同じ組にまとめる。
_CANONICAL.update({"-nocb": "-cb", "--no-cont-batching": "-cb", "-nkvo": "-kvo", "--no-kv-offload": "-kvo"})

#: 指定を拒否するフラグ (正規名)。
_RESERVED: frozenset[str] = frozenset({"-m", "--port", "--host"})

#: 設定 API から書けるフラグ (正規名)。性能・メモリ・推論の挙動だけで、ファイルやネットワークに触れない。
_API_ALLOWED: frozenset[str] = frozenset({
    "-c", "-ngl", "-t", "-tb", "-b", "-ub", "-fa", "--mlock", "--mmap", "-np", "-ctk", "-ctv", "-cram",
    "--cache-reuse", "-kvu", "--cache-idle-slots", "--ctx-checkpoints", "--checkpoint-min-step", "--swa-full",
    "-rea", "--reasoning-budget", "--reasoning-format", "--jinja", "-ot", "-cmoe", "-ncmoe", "-cb", "-dev",
    "-sm", "-ts", "-mg", "-kvo", "-dt", "-n", "--pooling", "--numa", "--prio", "--prio-batch", "--poll",
    "--poll-batch", "--keep", "--rope-scaling", "--rope-scale", "--rope-freq-base", "--rope-freq-scale",
    "--yarn-orig-ctx", "--yarn-ext-factor", "--yarn-attn-factor", "--yarn-beta-slow", "--yarn-beta-fast",
    "--temp", "--top-k", "--top-p", "--min-p", "--repeat-penalty", "--presence-penalty",
    "--frequency-penalty", "--cache-prompt", "--warmup", "--op-offload", "--kv-offload", "-to",
    "--slot-prompt-similarity", "--threads-http", "--context-shift", "--webui", "--metrics",
})
#: 推測デコードのフラグは名前が版で変わるので接頭辞で通す (ファイル・モデルを指すものを除く)。
_API_ALLOWED_PREFIXES = ("--spec-", "--draft-")
_PATHY_WORDS = ("file", "path", "model", "url", "dir", "lora", "vector", "mmproj", "log", "save", "ssl", "key")

_NUMBER = re.compile(r"^-?\d+(\.\d+)?$")


def _is_flag(token: str) -> bool:
    """フラグか (``-`` で始まり、負の数ではない)。"""
    return token.startswith("-") and len(token) > 1 and not _NUMBER.match(token)


def canonical_flag(token: str) -> str:
    """フラグの正規名 (``--ctx-size=8192`` → ``-c``、``--no-mmap`` → ``--mmap``)。"""
    name = token.split("=", 1)[0] if token.startswith("--") else token
    if name in _CANONICAL:
        return _CANONICAL[name]
    if name.startswith("--no-"):
        positive = "--" + name[len("--no-"):]
        return _CANONICAL.get(positive, positive)
    return name


def parse_launch_args(text: str | None) -> list[str]:
    """``launch_args`` を argv に分ける。予約フラグ (``-m`` / ``--port`` / ``--host``) は拒否する。

    引用符は POSIX 流 (空白を含む値は ``"..."`` で囲む)。バックスラッシュはエスケープにせず
    そのまま残す (Windows のパス ``C:\\models\\x.gguf`` を壊さない)。改行は空白と同じ。
    """
    if not text or not text.strip():
        return []
    lexer = shlex.shlex(text, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    lexer.escape = ""  # バックスラッシュは文字のまま (Windows のパス)
    try:
        argv = list(lexer)
    except ValueError as e:
        raise LaunchArgsError(f"launch_args could not be parsed: {e}") from e
    reserved = sorted({t for t in argv if _is_flag(t) and canonical_flag(t) in _RESERVED})
    if reserved:
        raise LaunchArgsError(
            f"launch_args must not set {', '.join(reserved)} "
            "(the model path and the port come from model_paths / the port settings)"
        )
    return argv


def _flags(argv: list[str]) -> set[str]:
    return {canonical_flag(t) for t in argv if _is_flag(t)}


def merge_launch_args(cmd: list[str], argv: list[str]) -> list[str]:
    """自動で組み立てた ``cmd`` から ``argv`` と同じフラグを値ごと除き、``argv`` を末尾に足す。

    ``cmd[0]`` (実行ファイル) は残す。``cmd`` のフラグの直後の、フラグでない要素をその値とみなす
    (自動のコマンドは ``フラグ [値]`` の並びだけで組まれている)。``argv`` が空なら ``cmd`` をそのまま返す。
    """
    if not argv:
        return cmd
    user = _flags(argv)
    out = cmd[:1]
    i = 1
    while i < len(cmd):
        token = cmd[i]
        if _is_flag(token) and canonical_flag(token) in user:
            i += 1
            if i < len(cmd) and not _is_flag(cmd[i]):
                i += 1
            continue
        out.append(token)
        i += 1
    return out + list(argv)


def _value_of(argv: list[str], canon: str) -> str | None:
    """最後に指定された ``canon`` の値 (``--flag=値`` も読む)。値が無ければ ``None``。"""
    found: str | None = None
    for i, token in enumerate(argv):
        if not _is_flag(token) or canonical_flag(token) != canon:
            continue
        if token.startswith("--") and "=" in token:
            found = token.split("=", 1)[1]
        elif i + 1 < len(argv) and not _is_flag(argv[i + 1]):
            found = argv[i + 1]
    return found


def _last_switch(argv: list[str], canon: str) -> bool | None:
    """真偽のフラグ (``--X`` / ``--no-X`` / 別名) の最後の指定。無ければ ``None``。"""
    found: bool | None = None
    for token in argv:
        if _is_flag(token) and canonical_flag(token) == canon:
            found = not (token.startswith("--no-") or token in ("-nocb", "-nkvo"))
    return found


def _int(flag: str) -> Callable[[str], int]:
    def parse(value: str) -> int:
        try:
            return int(value)
        except ValueError:
            raise LaunchArgsError(f"launch_args: {flag} needs an integer, got {value!r}") from None
    return parse


def _ngl(value: str) -> int | None:
    if value == "all":
        return 999
    if value == "auto":
        return None  # llama-server 側の自動に任せる (設定へは写さない)
    return _int("-ngl")(value)


#: 写す値 (種別 → 正規名 → (設定のキー, 値の変換))。値の範囲は各スキーマの検証が見る。
#: ``-ctk`` / ``-ctv`` は写さない (スキーマの許す型は上流より狭く、上流の正しい値を拒否してしまう)。
_DERIVED: dict[str, dict[str, tuple[str, Callable[[str], object]]]] = {
    "llama": {
        "-c": ("context_size", _int("-c")),
        "-ngl": ("gpu_layers", _ngl),
        "-np": ("slots", _int("-np")),
        "-t": ("threads", _int("-t")),
        "-b": ("batch_size", _int("-b")),
        "-ub": ("ubatch_size", _int("-ub")),
        "-cram": ("cache_ram_mib", _int("--cache-ram")),
        "--ctx-checkpoints": ("ctx_checkpoints", _int("--ctx-checkpoints")),
    },
    "embedding": {
        "-c": ("context_size", _int("-c")),
        "-ngl": ("gpu_layers", _ngl),
        "-np": ("slots", _int("-np")),
        "-t": ("threads", _int("-t")),
        "-b": ("batch_size", _int("-b")),
        "-ub": ("ubatch_size", _int("-ub")),
        "-cram": ("cache_ram_mib", _int("--cache-ram")),
    },
    "rerank": {
        "-ngl": ("gpu_layers", _ngl),
        "-t": ("threads", _int("-t")),
    },
}
#: 真偽のフラグから写す値 (種別 → 正規名 → 設定のキー)。
_DERIVED_SWITCHES: dict[str, dict[str, str]] = {"llama": {"-kvu": "kv_unified"}}


def derive_values(kind: str, argv: list[str]) -> dict[str, object]:
    """``argv`` から設定のキーへ写す値 (``-c 8192`` → ``{"context_size": 8192}``)。"""
    out: dict[str, object] = {}
    for canon, (key, convert) in _DERIVED[kind].items():
        raw = _value_of(argv, canon)
        if raw is None:
            continue
        value = convert(raw)
        if value is not None:
            out[key] = value
    for canon, key in _DERIVED_SWITCHES.get(kind, {}).items():
        switch = _last_switch(argv, canon)
        if switch is not None:
            out[key] = switch
    return out


def derived_keys(kind: str, text: str | None) -> set[str]:
    """``text`` (``launch_args``) が写す設定のキー。解釈できなければ空 (検証は別の場所が落とす)。"""
    try:
        return set(derive_values(kind, parse_launch_args(text)))
    except LaunchArgsError:
        return set()


def derive_into(data: object, kind: str) -> object:
    """スキーマの ``mode="before"`` 検証用: ``launch_args`` が写す値で ``data`` のキーを上書きした dict。"""
    if not isinstance(data, dict):
        return data
    text = data.get("launch_args")
    if not text:
        return data
    if not isinstance(text, str):
        raise LaunchArgsError("launch_args must be a string")
    return {**data, **derive_values(kind, parse_launch_args(text))}


def apply_launch_args(cfg: dict) -> dict:
    """検証前の config dict に、3 つの節の ``launch_args`` が写す値をその場で書き込む (冪等)。

    起動スクリプトは config.yaml を素の dict で読むので、環境調整と組み立てが明示値を見るよう
    最初に 1 回呼ぶ。検証済みの config (backend) には同じ値が既に入っている。
    """
    for kind, path in LAUNCH_ARGS_SECTIONS.items():
        node: object = cfg
        for key in path:
            node = node.get(key) if isinstance(node, dict) else None
        if isinstance(node, dict) and node.get("launch_args"):
            node.update(derive_values(kind, parse_launch_args(str(node["launch_args"]))))
    return cfg


def api_disallowed_flags(argv: list[str]) -> list[str]:
    """設定 API からは書けないフラグ (元の綴り、重複なし)。空なら全部書ける。"""
    out: list[str] = []
    for token in argv:
        if not _is_flag(token):
            continue
        canon = canonical_flag(token)
        allowed = canon in _API_ALLOWED or (
            canon.startswith(_API_ALLOWED_PREFIXES) and not any(w in canon for w in _PATHY_WORDS)
        )
        spelled = token.split("=", 1)[0]
        if not allowed and spelled not in out:
            out.append(spelled)
    return out
