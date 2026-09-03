"""ツール定義・登録・実行レジストリ"""

from __future__ import annotations

import asyncio
import functools
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from backend.log_config import get_logger

from backend.free.agent.meta_cognitive_tool_io import (
    tool_result_lacks_information,
    tool_result_succeeded,
)
from backend.free.agent.tool_ledger import record_current

logger = get_logger("agent.tools_registry")


@dataclass
class ToolDefinition:
    """ツール定義

    ``hidden=True`` のツールは ``get_descriptions_text()`` (LLM プロンプトの
    ツール一覧) に出さない。judge 等のコード側注入専用ツール
    (例: run_command_readonly) 向けで、search_history の session_id パラメータ
    非公開 (parameters から省く) のツール版に相当する。
    """
    name: str
    func: Callable
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    modes: list[str] = field(default_factory=lambda: ["chat", "create"])
    hidden: bool = False
    #: 実行タイムアウト (秒)。None なら呼出側の既定
    #: (chat_constants.TOOL_EXECUTION_TIMEOUT_SEC) を使う。内部で LLM 生成を
    #: 行うツールは既定 30 秒では足りない (実測 2026-07-26: draft_document が
    #: 会議テンプレート生成で 30 秒に達し、タイムアウト文言がそのまま回答に
    #: なった)。低速な環境ほど顕著なため、ツール側で宣言できるようにする。
    timeout_sec: float | None = None
    #: 「このモードで使えるツールは何か」に **載せる** モード。既定は
    #: :attr:`modes` と同じ。
    #:
    #: ``modes`` が制御するのは **選択** (分類器のメニュー / ``is_available``)
    #: だけで、``execute()`` はモードを強制しない。そのため
    #: 「``modes`` には無いが、そのモードの別の層から実際に実行される」ツールが
    #: 生まれる。実インシデント (2026-08-27 ライブ監査): chat で
    #: ``write_file`` を 3 回実行した直後に「使えるツールの一覧」を尋ねると、
    #: ``write_file`` だけが抜けた一覧が返った (``modes=["create"]`` のため)。
    #: 同じ会話の別の問いには「write_file を 3 回呼んだ」と正しく答えており、
    #: **目録だけが実態とずれていた**。
    #:
    #: 選択側を広げると分類器が直接そのツールを選べるようになり、経由していた
    #: 層のガード (パス解決 / action_blocked) を飛ばすため、**目録だけを**
    #: 実態に合わせる。
    inventory_modes: list[str] | None = None

    def listed_in(self, mode: str | None) -> bool:
        """``mode`` の capability summary に載せるべきか (純粋関数)。"""
        if not mode:
            return True
        return mode in (self.inventory_modes or self.modes)


def _record_tool_issue(name: str, succeeded: bool, rendered: str) -> None:
    """ツール実行の不首尾を issue 台帳へ落とす。

    「成功したが該当 0 件」(``search_history`` / ``search_code`` の空振り) は
    実行としては成功だが、ユーザーから見れば見つからなかったターン。監査では
    7 回空振りし、そのうえで「見つからなかった項目はありません」と答えていた。
    空振りの判定は ``meta_cognitive_tool_io._TOOL_EMPTY_RESULT_PREFIXES`` が
    SSOT (成否判定と同じ表を見る)。
    """
    from backend.free.agent.issue_ledger import record_current_issue

    if not succeeded:
        record_current_issue("tool_failed", f"{name}: {rendered[:80]}")
        return
    if tool_result_lacks_information(name, rendered):
        record_current_issue("tool_empty", f"{name}: {rendered[:80]}")

#: ファイルを対象にするツールと、そのパス引数の名前。
#: 台帳へ入れるのは **成功した実行だけ** — 失敗したパスを「直前のファイル」に
#: すると、次の暗黙参照が存在しないファイルへ向く。
_FILE_PATH_ARGS: dict[str, tuple[str, ...]] = {
    "write_file": ("path", "file_path", "filename"),
    "read_file": ("path", "file_path", "filename"),
}


def _resolve_bare_filename(name: str, kwargs: dict) -> None:
    """裸のファイル名を、この会話で使っているディレクトリへ寄せる。

    ``kwargs`` は **その場で書き換える** ので、実行にも台帳の記録にも
    解決後のパスが載る (台帳が相対パスのままだと、次ターンの突合が
    できず存在しないパスを「読んだ」と答える)。詳細は
    ``file_ledger.resolve_against_recent_dir``。
    """
    arg_names = _FILE_PATH_ARGS.get(name)
    if not arg_names:
        return
    from backend.free.agent.file_ledger import resolve_current_against_recent_dir

    for arg in arg_names:
        value = kwargs.get(arg)
        if isinstance(value, str) and value.strip():
            kwargs[arg] = resolve_current_against_recent_dir(value)
            return


def _record_touched_file(name: str, succeeded: bool, kwargs: dict) -> None:
    """ファイル操作のパスを file 台帳へ落とす。"""
    if not succeeded:
        return
    arg_names = _FILE_PATH_ARGS.get(name)
    if not arg_names:
        return
    from backend.free.agent.file_ledger import record_current_file

    for arg in arg_names:
        value = kwargs.get(arg)
        if isinstance(value, str) and value.strip():
            record_current_file(value)
            return


@functools.lru_cache(maxsize=256)
def _signature_params(func: Callable) -> tuple[inspect.Parameter, ...]:
    """ツール関数の (self / cls / *args / **kwargs を除いた) 引数を宣言順で返す。

    シグネチャ取得不可 (C 実装等) なら空。``required_args`` /
    ``required_params`` / ``_validate_args`` / ``get_descriptions_text`` が
    それぞれ別々にシグネチャを走査していた重複の合流点。
    """
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return ()
    return tuple(
        p for name, p in sig.parameters.items()
        if name not in ("self", "cls")
        and p.kind not in (
            inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD,
        )
    )


def _accepts_var_kwargs(func: Callable) -> bool:
    """関数が ``**kwargs`` を受けるか (未知引数を落としてよいかの判定)。"""
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return True
    return any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )


def _coerce_declared_type(value: Any, declared: str) -> Any:
    """``parameters`` の型宣言に従って文字列値を単純型へ寄せる。

    文法制約 JSON の分類器はスキーマ上 string しか出せない経路があり、
    ``max_depth="1"`` / ``timeout="5"`` のような値がそのまま関数へ届く。
    変換できなければ ``ValueError``。
    """
    if not isinstance(value, str):
        return value
    text = value.strip()
    if declared == "integer":
        return int(text)
    if declared == "number":
        return float(text)
    if declared == "boolean":
        lowered = text.lower()
        if lowered in ("true", "1", "yes"):
            return True
        if lowered in ("false", "0", "no"):
            return False
        raise ValueError(text)
    return value


def _summarize_args_for_log(kwargs: dict[str, Any]) -> str:
    """ログ向けに引数値を 80 文字へ切り詰める (write_file の本文全文を落とさない)。"""
    parts = []
    for k, v in kwargs.items():
        text = repr(v)
        if len(text) > 80:
            text = text[:80] + f"...({len(text)} chars)"
        parts.append(f"{k}={text}")
    return ", ".join(parts)


class ToolsRegistry:
    """ツールの登録・取得・実行を管理"""

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def register(
        self,
        name: str,
        func: Callable,
        description: str,
        parameters: dict[str, Any] | None = None,
        modes: list[str] | None = None,
        hidden: bool = False,
        timeout_sec: float | None = None,
        inventory_modes: list[str] | None = None,
    ) -> None:
        """ツールを登録

        ``inventory_modes`` は「使えるツール一覧」に載せるモード
        (既定は ``modes``)。選択可否は変えずに目録だけ広げたいときに使う
        (:attr:`ToolDefinition.inventory_modes` の説明を参照)。
        """
        self._tools[name] = ToolDefinition(
            name=name,
            func=func,
            description=description,
            parameters=parameters or {},
            modes=modes or ["chat", "create"],
            hidden=hidden,
            timeout_sec=timeout_sec,
            inventory_modes=inventory_modes,
        )
        logger.info("Registered tool: %s", name)

    def timeout_for(self, name: str, default: float) -> float:
        """ツールの実行タイムアウトを返す (未宣言なら ``default``)。"""
        tool = self._tools.get(name)
        if tool is None or tool.timeout_sec is None:
            return default
        return tool.timeout_sec

    def has(self, name: str) -> bool:
        """ツールが登録済みかどうか"""
        return name in self._tools

    def get(self, name: str) -> ToolDefinition | None:
        """ツール定義を取得"""
        return self._tools.get(name)

    def required_args(self, name: str) -> tuple[str, ...]:
        """``name`` の **必須引数** を宣言順で返す (未登録なら空)。

        判定は実装関数のシグネチャ — **既定値を持たない引数が必須**。
        ``parameters`` の宣言には required/optional の区別が無いので、
        そちらから required を読むと ``list_directory`` の ``max_depth``
        (既定 3) まで必須になってしまう。

        文法制約 JSON の分類器が arg を 1 つしか生成しないため、必須が 2 つ
        以上あるツールは呼び出しが必ず失敗する。その事前判定に使う
        (``tool_call_judge._drop_if_required_args_missing``)。
        """
        tool = self._tools.get(name)
        if tool is None:
            return ()
        declared = set(tool.parameters or {})
        return tuple(
            p.name for p in _signature_params(tool.func)
            if p.default is inspect.Parameter.empty
            and (not declared or p.name in declared)
        )

    def is_available(self, name: str, mode: str) -> bool:
        """ツールが登録済みかつ現在の ``mode`` で利用可能か。

        ``has(name)`` (存在チェックのみ) と異なり ``ToolDefinition.modes`` も
        考慮する。tool_call_judge.py の各判定層 (rule/learned/executable
        fallback/recall) と deliberative.py の実行前ゲートが、同じ「存在 +
        mode 適合」判定をそれぞれ個別に書いていた重複を解消するための共通口
        (2026-07-18 レビューで指摘)。
        """
        tool_def = self._tools.get(name)
        return tool_def is not None and mode in tool_def.modes

    def list_names(self) -> list[str]:
        """登録済みツール名を登録順で返す。

        文法制約ツール分類の enum 構築 (``grammar_tool_classifier.
        available_tool_names``) が全ツールを走査するための公開口。``hidden`` は
        ここでは除外しない — hidden は「プロンプトのツール一覧に出さない」印で
        あって「使わせない」印ではないため、絞り込みは呼出側の責務。
        """
        return list(self._tools)

    def required_params(self, name: str) -> set[str]:
        """ツールの必須引数名。未登録なら空集合。

        呼出側が「この引数一式でこのツールを撃てるか」を判断するための公開口
        (ToolCallJudge の兄弟ツール載せ替えが使う)。
        """
        tool = self._tools.get(name)
        return self._required_param_names(tool) if tool is not None else set()

    def get_descriptions_text(self, mode: str | None = None) -> str:
        """ツール説明をテキスト形式で返す（プロンプト注入用）

        各ツールについて、シグネチャ行に続けて parameter ごとに required/optional
        と description を 1 行ずつ展開する。LLM が optional 引数を省略する判断や、
        ISO 日付などの例示を読めるようにするため。
        """
        lines = []
        for tool in self._tools.values():
            if tool.hidden:
                continue
            if mode and mode not in tool.modes:
                continue
            required = self._required_param_names(tool)
            sig_parts = []
            for k, v in tool.parameters.items():
                ptype = v.get("type", "any")
                marker = "" if k in required else "?"
                sig_parts.append(f"{k}{marker}: {ptype}")
            sig_text = ", ".join(sig_parts)
            lines.append(f"- {tool.name}({sig_text}): {tool.description}")
            for k, v in tool.parameters.items():
                desc = v.get("description", "")
                if not desc:
                    continue
                req_marker = "required" if k in required else "optional"
                lines.append(f"    - {k} ({req_marker}): {desc}")
        return "\n".join(lines)

    def get_capability_summary(self, mode: str | None = None) -> str:
        """ユーザー向けのツール一覧 (名前 + 説明のみ、1 行 1 ツール)。

        ``get_descriptions_text`` と違い引数シグネチャを展開せず、``hidden``
        も含める。hidden は「モデルへのツールメニューに出さない」印であって
        「ユーザーに隠す」印ではなく、実行結果は UI の Agentic ステップに
        そのまま表示されている。「何のツールが使えるか」を尋ねられたときに
        hidden を落とすと、実際に走っている ``run_command_readonly`` などが
        一覧から消えて回答が実態とずれる。
        """
        lines = []
        for tool in self._tools.values():
            if not tool.listed_in(mode):
                continue
            lines.append(f"- {tool.name}: {tool.description}")
        return "\n".join(lines)

    @staticmethod
    def _required_param_names(tool: ToolDefinition) -> set[str]:
        """ツール関数シグネチャから default なしパラメータ名を抽出"""
        return {
            p.name for p in _signature_params(tool.func)
            if p.default is inspect.Parameter.empty
        }

    async def execute(self, name: str, **kwargs: Any) -> Any:
        """ツールを実行

        引数不足時は TypeError を発生させず、エラー文字列を返す。
        """
        tool = self._tools.get(name)
        if tool is None:
            raise ValueError(f"Unknown tool: {name}")

        # 引数のバリデーション（全ツール共通）。未知引数の除去と型寄せは
        # kwargs をその場で書き換える。
        validation_error = self._validate_args(tool, kwargs)
        if validation_error:
            logger.warning("Tool arg validation failed: %s - %s", name, validation_error)
            # 実行に至らなかった撃ち方も「実行しようとした」記録として残す。
            # 台帳が空だとモデルは「そのツールは使っていません」と答える。
            record_current(name, False, reason="invalid_args")
            _record_tool_issue(name, False, validation_error)
            return f"Error: {validation_error}"

        # 裸のファイル名はプロセスの cwd ではなく「この会話のディレクトリ」へ。
        # ログにも解決後のパスが出るよう、要約の前に掛ける。
        _resolve_bare_filename(name, kwargs)

        logger.info("Executing tool: %s(%s)", name, _summarize_args_for_log(kwargs))

        try:
            if inspect.iscoroutinefunction(tool.func):
                result = await tool.func(**kwargs)
            else:
                # 同期関数はスレッドプールで実行し、イベントループをブロックしない
                result = await asyncio.to_thread(tool.func, **kwargs)
        except asyncio.CancelledError:
            # 呼出側の ``asyncio.wait_for`` による timeout もここへ届く
            # (キャンセルは await 中のこのコルーチンに配送される)。呼出側で
            # 記録すると経路ごとに取りこぼすので、合流点であるここで残す。
            record_current(name, False, reason="timeout")
            _record_tool_issue(name, False, "timeout")
            raise
        except Exception as e:
            record_current(name, False, reason="error")
            _record_tool_issue(name, False, str(e))
            raise

        # 自己申告の根拠 (「この会話で実際に何を実行したか」)。実行経路は 6 箇所に
        # 分かれているので、記録は **唯一の合流点であるここ** で行う。呼出側に
        # 配ると必ず取りこぼす (詳細は tool_ledger._current_target のコメント)。
        succeeded = tool_result_succeeded(name, str(result))
        record_current(name, succeeded)
        # 不首尾も同じ合流点で記録する。自己申告の問い (「見つからなかった
        # 項目はありましたか」) に対し、監査では 7 回すべて「ありません」と
        # 答えていた — 会話履歴にツールの成否も空振りも残らないため。
        _record_tool_issue(name, succeeded, str(result))

        # 「保存したファイルを読んで」型の暗黙参照を解決する材料。
        # パスはクエリの文字列からは取れないので、実行時に覚えておく
        # (file_ledger のモジュール docstring を参照)。
        _record_touched_file(name, succeeded, kwargs)
        return result

    @staticmethod
    def _validate_args(tool: ToolDefinition, kwargs: dict[str, Any]) -> str | None:
        """関数シグネチャと ``parameters`` 宣言に基づいて引数を検証・正規化する。

        - 既定値なしの引数が欠落 / ``None`` なら必須引数エラー
        - 既定値ありの引数に ``None`` が来たら落として既定値に任せる
        - シグネチャに無い引数は WARNING を出して落とす (``**kwargs`` を受ける
          関数は除く)。以前は ``TypeError`` で実行ごと失敗していた
        - ``parameters`` で integer / number / boolean と宣言された引数に文字列が
          来たら変換する。変換できなければエラー

        ``kwargs`` は **その場で書き換える**。

        Returns:
            エラーメッセージ。問題なければ None。
        """
        params = _signature_params(tool.func)
        if not params and _accepts_var_kwargs(tool.func):
            return None  # シグネチャ取得不可 / 全て **kwargs ならスキップ

        missing: list[str] = []
        for param in params:
            if param.default is inspect.Parameter.empty:
                if param.name not in kwargs or kwargs[param.name] is None:
                    missing.append(param.name)
            elif param.name in kwargs and kwargs[param.name] is None:
                del kwargs[param.name]

        if missing:
            # LLM が引数を修正できるようにシグネチャ情報を含める
            expected_parts: list[str] = []
            for p in params:
                ann = (
                    p.annotation.__name__
                    if p.annotation is not inspect.Parameter.empty
                    and hasattr(p.annotation, "__name__")
                    else "any"
                )
                if p.default is inspect.Parameter.empty:
                    expected_parts.append(f"{p.name}: {ann}")
                else:
                    expected_parts.append(f"{p.name}: {ann} = {p.default!r}")
            return (
                f"Missing required argument(s): {', '.join(missing)}. "
                f"Expected: {tool.name}({', '.join(expected_parts)})"
            )

        if not _accepts_var_kwargs(tool.func):
            known = {p.name for p in params}
            for extra in [k for k in kwargs if k not in known]:
                logger.warning(
                    "Dropping unknown argument for tool %s: %s", tool.name, extra,
                )
                del kwargs[extra]

        for key, spec in (tool.parameters or {}).items():
            if key not in kwargs or not isinstance(spec, dict):
                continue
            declared = spec.get("type")
            if declared not in ("integer", "number", "boolean"):
                continue
            try:
                kwargs[key] = _coerce_declared_type(kwargs[key], declared)
            except (TypeError, ValueError):
                return (
                    f"Invalid value for argument '{key}': expected {declared}, "
                    f"got {kwargs[key]!r}"
                )
        return None

    @property
    def count(self) -> int:
        return len(self._tools)
