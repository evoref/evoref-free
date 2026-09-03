"""補助タスクの失敗を「その処理単位」に紐づけて数える台帳 (横断基盤)。

**なぜ要るか**: sleep-time / 学習の各ステージは aux のタイムアウトを個別に
握って WARNING を出し、縮退したまま先へ進む。設計としては正しい (1 ステージの
失敗でサイクル全体を落とさない) が、その結果 **サイクルは「成功」で終わる**。

実測 (2026-09-03 ライブ監査): ``outcome_2026-09-03.jsonl`` は 456 件すべてが
``success: true`` だった。同じ時間帯に aux 生成タイムアウトが 14 件、
``MetaCognitive process failed`` が 1 件、ユーザーに見えた ``stream_timeout``
が 1 件あったにもかかわらず。この記録を ``log_ingestor`` 経由で MDP エピソード
として学習へ流すと、**報酬信号が定数**になり自己進化が乱歩する
(既知の「fitness が恒真だと選択圧がゼロ」と同じ形)。

**設計**: 失敗を検知できる唯一の場所 (``AuxClient``) で記録し、結末を書く側
(outcome エミッタ) が読む。各ステージに引数を通して回らないので、ステージを
足したときの付け忘れが起きない。``contextvars`` はタスク生成時にコピーされる
ので、``asyncio.create_task`` の子まで同じ台帳を共有する。
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

__all__ = [
    "aux_failure_scope",
    "record_aux_failure",
    "aux_failure_signals",
    "current_aux_failures",
]

_ledger: ContextVar[list[dict] | None] = ContextVar("aux_failure_ledger", default=None)


@contextmanager
def aux_failure_scope() -> Iterator[list[dict]]:
    """この with 区間で起きた aux 失敗を集める台帳を開く。

    入れ子にした場合、内側のスコープが外側から独立する (区間ごとの結末を
    書きたいのだから、内側の失敗を外側にも二重計上しない)。
    """
    entries: list[dict] = []
    token = _ledger.set(entries)
    try:
        yield entries
    finally:
        _ledger.reset(token)


def record_aux_failure(purpose: str, reason: str) -> None:
    """aux 呼び出しの失敗を、開いていれば現在の台帳へ記録する。

    スコープが開いていなければ何もしない (計測していない区間まで
    グローバルに溜め込むと、どの処理単位の失敗か分からなくなる)。
    """
    entries = _ledger.get()
    if entries is None:
        return
    entries.append({"purpose": purpose or "<unspecified>", "reason": reason})


def open_aux_failure_ledger() -> None:
    """このリクエスト (タスク) 用の台帳を開く。``issue_ledger_scope`` と同じ形。

    ``with`` を張れないリクエストハンドラ用。contextvar はタスクごとに
    独立しているので、reset しなくても他リクエストへ漏れない。
    """
    _ledger.set([])


def current_aux_failures() -> list[dict]:
    """開いている台帳の現在値 (コピー)。開いていなければ空。

    結末を書く側が「自分でスコープを持ち回らずに」読むための入口。
    スコープを開くのはリクエストやサイクルの入口 1 箇所で足りる。
    """
    entries = _ledger.get()
    return list(entries) if entries else []


def aux_failure_signals(entries: list[dict]) -> dict:
    """台帳を outcome の ``quality_signals`` へ載せる形へ畳む。

    件数と purpose 別内訳を返す。空なら ``aux_failures: 0`` だけ返す —
    「測ったうえでゼロ」と「測っていない」を区別できるようにするため、
    キー自体は必ず出す。
    """
    if not entries:
        return {"aux_failures": 0}
    by_purpose: dict[str, int] = {}
    for e in entries:
        by_purpose[e["purpose"]] = by_purpose.get(e["purpose"], 0) + 1
    return {
        "aux_failures": len(entries),
        "aux_failure_purposes": by_purpose,
    }
