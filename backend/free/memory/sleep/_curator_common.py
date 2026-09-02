"""sleep-time キュレーターの共通処理。

``url_curator`` / ``executable_command_curator`` / ``assertion_curator`` は
「直近ノートから対を集める → 補助タスクで採点 → 既存ファクトと突合 → 書き戻す」
という同型のパイプラインで、下記のヘルパを各ファイルへ byte 一致で書き写して
いた。片方だけ直すと採点や subject の体裁がずれるため 1 箇所へ集約する。

**subject の組み立て自体は共有しない**: url は host、executable_command は
正規化 mode と、prefix の決め方が本質的に異なる。共通なのは
「正規化済みテキストの sha1 先頭 12 桁」までなので、そこだけを
``subject_digest()`` として切り出す。
"""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.free.memory.stores.short_term import MemoryNote

#: プロンプトへ載せる質問の最大文字数。
QUERY_PROMPT_LIMIT = 1000
#: プロンプトへ載せる応答の最大文字数。
ANSWER_PROMPT_LIMIT = 2000

_BARE_SCORE_RE = re.compile(r"-?\d+(?:\.\d+)?")


def truncate_for_prompt(text: str, limit: int) -> str:
    """``limit`` を超えたら切り詰めて ``...(truncated)`` を付ける (純粋関数)。"""
    if len(text) <= limit:
        return text
    return text[:limit] + "...(truncated)"


def public_notes(notes: list["MemoryNote"]) -> list["MemoryNote"]:
    """private セッション由来のノートを落とす (純粋関数)。

    private ターンは ``memory_only`` で動き、LTM / SemMem への昇格をしない契約
    (``config.yaml`` の ``memory.private``)。Step 8 の抽出器はこれを
    ``extractors/base.py`` の ``if note.private: skip`` で守っているが、後から
    足された Step 8.4 / 8.5 / 8.6 の 3 キュレーターは **同じ
    ``short_term.notes.values()`` を受け取りながら同じガードを持っていなかった**。

    実害 (2026-09-01 監査で再現): private ターンで実行した
    ``run_command_readonly`` が ``mem.world.executable_command.chat.<digest>`` の
    ``world_fact`` として書き込まれ、しかも生成側が ``private`` を引き継がない
    ため ``fact.private=False`` になる。``MemoryInjector._classify_fact`` の
    ``if fact.private: return None`` にも掛からず、``ToolCallJudge`` の
    コマンドリコールが **後続の通常セッションから引き当てる**。漏れる中身は
    private ターンで踏んだ URL・実行したコマンド文字列・そのときの質問文。

    3 キュレーターが各々ガードを書くと 4 度目の抜けを作るので、ここを SSOT に
    して全員が入口で必ず通す。呼出側 (``sleep_update``) も渡す前に落とす
    (二重防御)。
    """
    return [n for n in notes if not getattr(n, "private", False)]


def subject_digest(normalized: str) -> str:
    """正規化済みテキストから subject 用の短いダイジェストを作る (純粋関数)。

    prefix の付け方は用途ごとに異なるため、ここではダイジェストのみを返す。
    """
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]


def coerce_bare_score(content: str | None) -> str | None:
    """裸の数値応答 ("0.7" 等) から最初の float トークンを取り出す。

    LFM2 系 hybrid/recurrent モデルは response_format(json_schema) を強制できず、
    スキーマ ({"score": ...}) ではなくスカラ ("0.7") を返すことがある。その場合に
    後段の ``float()`` へ渡せる文字列を返す (見つからなければ ``None``)。
    """
    if not content:
        return None
    m = _BARE_SCORE_RE.search(content)
    return m.group(0) if m else None


def build_scoring_prompt(query: str, answer: str, **extra: str) -> str:
    """採点用 user プロンプトを組み立てる (純粋関数)。

    ``QUESTION`` / ``ASSISTANT_ANSWER`` は 3 兄弟で共通。末尾へ載せる文脈行
    (``URL`` / ``JUDGEMENT`` / ``COMMAND`` 等) はキュレーターごとに異なるため
    キーワード引数で受け取り、渡された順に追記する。
    """
    lines = [
        f"QUESTION: {truncate_for_prompt(query, QUERY_PROMPT_LIMIT)}",
        f"ASSISTANT_ANSWER: {truncate_for_prompt(answer, ANSWER_PROMPT_LIMIT)}",
    ]
    lines.extend(f"{label}: {value}" for label, value in extra.items())
    return "\n".join(lines) + "\n"
