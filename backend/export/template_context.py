"""ターンスコープの選択済みテンプレート (c_16 §4.5.2 / f_11 §9.1)。

``write_file`` ツールの引数は増やさない契約 (chat.py が判定点 / 明示指定で
確定したテンプレートを、ターンの間だけ ``contextvars`` 経由で運ぶ)。
``trace_id`` / ``private`` (``backend.trace_context``) と同じ理由でここに置く
— 発火地点 (chat.py) と消費地点 (``filesystem._write_rich_document``) の間に
関数引数で配れない層 (ツール実行の合流点) が挟まるため。

セット / リセットは呼出側の責務 (chat.py がターンの開始と終わりで行う)。
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "SelectedTemplate",
    "applied_template_key",
    "clear_selected_template",
    "defer_template_provenance",
    "get_selected_template",
    "get_template_hint",
    "mark_template_applied",
    "set_selected_template",
    "set_template_hint",
]


@dataclass(frozen=True, slots=True)
class SelectedTemplate:
    """このターンで適用するテンプレート (c_16 §4.5.2)。"""

    #: 継承元ファイルの絶対パス (corpus の版ディレクトリ内、読み取り専用)。
    #: ``outline`` だけを持つエントリ (構成だけ決める様式) では ``None``。
    base_path: Path | None
    #: 来歴に刻む版込みの鍵 (``<package_id>@<version>:<entry_id>``、c_05 §0.6)。
    provenance_key: str
    #: 章立て (構成テンプレート) ファイルの絶対パス。無ければ ``None``
    #: (f_08 §3.1.1)。
    outline_path: Path | None = None
    #: 帳票の列定義 (生 dict、c_16 §4.5.2)。無ければ ``None`` (体裁の継承 /
    #: 構成テンプレートのみの様式、f_11 §9.1)。あれば帳票の穴埋め (f_11 §9.2)。
    fields: "tuple[dict, ...] | None" = None


_current: ContextVar[SelectedTemplate | None] = ContextVar(
    "evoref_selected_template", default=None,
)


@dataclass(slots=True)
class _TemplateUsage:
    """「選ばれた」だけでなく実際に適用されたか (来歴、c_05 §0.6) — ターンに 1 つ。

    contextvar に入れるのは **この可変オブジェクトへの参照**で、値そのものではない。
    理由は 2 つ (どちらも 2026-09-20 のレビューで、値を直接 contextvar に入れた初版が
    実際の流れでは効かないと分かった):

    - contextvar の ``set`` は子タスクから親へ伝わらない。create の制作ターンは
      detached な子タスクで ``write_file`` を呼ぶので、そこで立てた印は親側の
      経験記録から見えない。参照を共有すれば、コピーされた context からも同じ
      オブジェクトを書き換えられる。
    - 長文経路は **経験を記録してからファイルを書く** (``chat_stream_long_form``)。
      記録の時点では適用したかどうかがまだ決まっていないので、記録側は来歴の
      入れ先 (``.template`` を持つオブジェクト) を預け、適用が決まった時点で
      後から埋める (:func:`defer_template_provenance`)。経験は sleep-time が
      書き戻すまでメモリ上のバッファにあるので、後からの書き込みが永続化に間に合う。
    """

    applied_key: str = ""
    pending: list[Any] = field(default_factory=list)


_usage: ContextVar[_TemplateUsage | None] = ContextVar(
    "evoref_template_usage", default=None,
)

#: このターンの様式候補通知 (``template_hint`` SSE フレーム、c_17 §3.8)。
#: ``None`` = 出さない。``_current`` / ``_usage`` と同じくターンスコープの
#: contextvar (発火地点 chat.py と消費地点 ``_respond`` の間に関数引数で
#: 配れない層 (層ごとのディスパッチ分岐) が挟まるため)。
_hint: ContextVar[dict | None] = ContextVar("evoref_template_hint", default=None)


def set_template_hint(payload: dict | None) -> None:
    """このターンの様式候補通知を確定する (``None`` で「出さない」)。"""
    _hint.set(payload)


def get_template_hint() -> dict | None:
    """現在のコンテキストの様式候補通知 (無ければ ``None``)。"""
    return _hint.get()


def set_selected_template(template: SelectedTemplate | None) -> None:
    """このターンで適用するテンプレートを確定する (``None`` で「無し」)。"""
    _current.set(template)
    _usage.set(_TemplateUsage() if template is not None else None)


def get_selected_template() -> SelectedTemplate | None:
    """現在のコンテキストの選択済みテンプレート (未選択なら ``None``)。"""
    return _current.get()


def mark_template_applied() -> None:
    """選択済みテンプレートを実際に適用したことを記録する (未選択なら no-op)。

    先に預けられていた来歴の入れ先 (:func:`defer_template_provenance`) も埋める。
    """
    selected = _current.get()
    usage = _usage.get()
    if selected is None or usage is None:
        return
    usage.applied_key = selected.provenance_key
    for ref in usage.pending:
        if not getattr(ref, "template", ""):
            ref.template = usage.applied_key
    usage.pending.clear()


def applied_template_key() -> str:
    """適用済みなら来歴鍵 (``<package_id>@<version>:<entry_id>``)、未適用なら空文字。"""
    usage = _usage.get()
    return usage.applied_key if usage is not None else ""


def defer_template_provenance(ref: Any) -> None:
    """来歴の入れ先を預ける — 適用が後から決まったら ``ref.template`` を埋める。

    様式が選ばれていないターンでは何もしない (``ref`` に触らない)。既に適用済み
    ならその場で埋める。``ref`` は ``template`` 属性を持つ可変オブジェクト
    (経験レコードの ``GenerationConfigRef``)。
    """
    usage = _usage.get()
    if usage is None or getattr(ref, "template", ""):
        return
    if usage.applied_key:
        ref.template = usage.applied_key
    else:
        usage.pending.append(ref)


def clear_selected_template() -> None:
    """ターン終了で必ず呼ぶ (次のターン・次のリクエストへ持ち越さない)。

    context の参照を外すだけで、預かった入れ先 (``pending``) は消さない —
    detached な子タスクが親のターン終了後に書き出すことがあり、その子の context は
    同じ ``_TemplateUsage`` をまだ指している。様式候補通知 (``_hint``) も
    同じターンの寿命なのでここで一緒に外す。
    """
    _current.set(None)
    _usage.set(None)
    _hint.set(None)
