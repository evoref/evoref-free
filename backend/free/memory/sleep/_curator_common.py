"""sleep-time キュレーター 3 兄弟の共通処理。

``url_curator`` / ``rag_judge_curator`` / ``executable_command_curator`` は
「直近ノートから対を集める → 補助タスクで採点 → 既存ファクトと突合 → 書き戻す」
という同型のパイプラインで、下記のヘルパを各ファイルへ byte 一致で書き写して
いた。片方だけ直すと採点や subject の体裁がずれるため 1 箇所へ集約する。

**subject の組み立て自体は共有しない**: url は host、rag_judge は種別テーブル、
executable_command は正規化 mode と、prefix の決め方が本質的に異なる。共通なのは
「正規化済みテキストの sha1 先頭 12 桁」までなので、そこだけを
``subject_digest()`` として切り出す。
"""

from __future__ import annotations

import hashlib
import re

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
