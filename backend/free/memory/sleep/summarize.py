"""Step 8-9: 未要約セッションの LLM 要約生成

``sleep_update.SleepTimeWorker._step8_9_summarize_sessions`` として実装されていた
要約生成ロジックを独立 module に切り出したもの。

処理は ``HistoryManager.index.sessions`` のうち ``summary is None`` かつ
``session.turns`` が存在するセッションを対象に、補助タスクで 1-2 文の要約を
生成し、 続いて embedder で要約埋め込みベクトルを計算する。
生成した要約はセッションファイル + インデックスに、ベクトルは埋め込みモデルごとの
束 (``history/embeddings/<model>.npy`` + id 表) に永続化する (c_05 §2.1)。

本 module は EvorefMem pillar 内部扱いで、LLM 呼び出しは caller から受け取った
``llm_client`` に閉じる (EvorefGen pillar の Protocol に準拠する抽象 client)。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from backend.config import get_path_resolver, mode_base_model_raw
from backend.free.core.intent_vocab import split_sentences
from backend.free.core.session_mode import normalize_session_mode
from backend.free.core.text_quality import detect_lang
from backend.log_config import get_logger
from backend.trace_context import run_in_executor_with_context
from backend.utils import parse_utc

if TYPE_CHECKING:
    from backend.free.rag.embedding_backend import EmbeddingBackend

logger = get_logger("memory.sleep.summarize")

#: 要約済みセッションを作り直す最小の追加ターン数。
RESUMMARIZE_MIN_TURNS = 10

#: 要約プロンプトへ載せるターン数 (末尾から)。
_SUMMARY_TURN_WINDOW = 20

#: 要約の出力上限。切れたら倍にして 1 回だけ作り直す (:func:`_generate_summary`)。
_SUMMARY_MAX_TOKENS = 128

#: 作り直しても切れた会話の ``{session_id: その時点の turn_count}``。会話が伸びる
#: まで要約に出さない — 256 でも切れる会話が毎回の Full で 2 回ずつ永久に再試行
#: されていた (2026-09-27 独立レビュー)。プロセス内だけの印 (再起動で 1 回だけ
#: 再試行する) で、セッション索引の形式は変えない。
_TRUNCATED_AT: dict[str, int] = {}

#: 作り直しても行為主体の無い文が残った会話 (``_TRUNCATED_AT`` と同じ扱いの印)。
_UNATTRIBUTED_AT: dict[str, int] = {}

#: 要約の各文が名指すべき行為主体 (要約の入力 ``<role>: <発話>`` の 2 者)。
#: 文に主語が無いと、アシスタントの提案 (「浅草や渋谷を訪れ…」) がユーザーの
#: 行動として読める (2026-10-10: 31 要約中 8 件が主語なし、うち 2 件がそう読めた)。
#: 照合は小文字化した文に対して行う (「ユーザ」は長音の有無の両方を受ける)。
_ACTORS: tuple[str, ...] = ("ユーザ", "アシスタント", "user", "assistant")

_SUMMARY_PROMPT = (
    "以下の会話を1-2文で要約してください。各文は「ユーザーは」または「アシスタントは」"
    "で始め、誰の発言・行動かを明示してください (例: ユーザーは〜を尋ね、"
    "アシスタントは〜を提案した)。アシスタントの提案や説明を、ユーザーが実際に"
    "行ったこととして書かないでください。"
)
_SUMMARY_RETRY_NOTE = (
    "前回の要約には主語の無い文がありました。すべての文に「ユーザー」か"
    "「アシスタント」を主語として書いてください。"
)


def chat_summary_skip_modes() -> frozenset[str]:
    """要約を見送るセッションの mode (配信モデルが chat のモデルと違う間は ``chat``)。

    create モードで ``create_model`` が別のモデルだと、Full の要約は chat の会話も
    そのモデルが作る (2026-09-27 監査 M3: Coder-14B が chat の会話を要約した)。
    chat のモデルに戻ってから作る。解決できなければ見送らない (従来どおり)。
    """
    try:
        resolver = get_path_resolver()
        active = resolver.active_mode
        if active == "chat":
            return frozenset()
        serving = mode_base_model_raw(resolver.models, active, default="")
        chat = mode_base_model_raw(resolver.models, "chat", default="")
    except Exception as exc:  # noqa: BLE001 - 解決できなければ従来どおり要約する
        logger.debug("Failed to resolve the serving model for summaries: %s", exc)
        return frozenset()
    if serving and chat and Path(serving).name != Path(chat).name:
        return frozenset({"chat"})
    return frozenset()


def unattributed_sentences(summary: str) -> list[str]:
    """行為主体 (ユーザー / アシスタント) を名指さない文 (無ければ空)。

    語形の一覧ではなく構造の検査 — 要約の入力は 2 者の発話だけなので、各文が
    どちらの行為かを名指していれば、提案と実行の取り違えは文面に表れる。
    """
    sentences = (s.strip() for s in split_sentences(summary))
    return [
        s for s in sentences
        if s and not any(actor in s.lower() for actor in _ACTORS)
    ]


async def _generate_summary(
    llm_client: Any, turns_text: str, session_id: str, *, retry_note: str = "",
) -> str | None:
    """要約を 1 つ作る。``finish_reason=length`` なら上限を倍にして 1 回だけ作り直す。

    それでも切れたら ``None`` (保存しない)。切れた要約が保存・昇格され、途中で
    途切れた文と誤答の日付が事実として運ばれた (2026-09-27 監査 M3)。
    """
    instruction = f"{_SUMMARY_PROMPT}{retry_note}"
    for max_tokens in (_SUMMARY_MAX_TOKENS, _SUMMARY_MAX_TOKENS * 2):
        result = await llm_client.generate(
            messages=[{
                "role": "user",
                "content": f"{instruction}\n\n{turns_text}",
            }],
            stream=False,
            max_tokens=max_tokens,
            purpose="summarize",
            id_slot=getattr(llm_client, "background_slot", -1),
        )
        choice = result["choices"][0]
        if str(choice.get("finish_reason") or "") != "length":
            return choice["message"]["content"].strip()
        logger.warning(
            "Summary for session %s was truncated at max_tokens=%d; not saved as is",
            session_id, max_tokens,
        )
    return None


async def _generate_attributed_summary(
    llm_client: Any, turns_text: str, session_id: str,
) -> tuple[str | None, str]:
    """主語の検査を通った要約と、通らなかった理由 (``truncated`` / ``unattributed``)。

    主語の無い文が残れば 1 回だけ主語を求めて作り直し、それでも残れば保存しない。
    """
    summary = await _generate_summary(llm_client, turns_text, session_id)
    if summary is None:
        return None, "truncated"
    if not unattributed_sentences(summary):
        return summary, ""
    logger.info(
        "Summary for session %s has a sentence without an actor; regenerating once",
        session_id,
    )
    summary = await _generate_summary(
        llm_client, turns_text, session_id, retry_note=_SUMMARY_RETRY_NOTE,
    )
    if summary is None:
        return None, "truncated"
    if unattributed_sentences(summary):
        logger.warning(
            "Summary for session %s still has a sentence without an actor; not saved",
            session_id,
        )
        return None, "unattributed"
    return summary, ""


def _held_back(entry: object, skip_modes: frozenset[str]) -> bool:
    """このサイクルでは要約に出さない会話か (mode の見送り / 作り直しても保存できなかった)。"""
    if skip_modes and normalize_session_mode(getattr(entry, "mode", None)) in skip_modes:
        return True
    turn_count = int(getattr(entry, "turn_count", 0) or 0)
    session_id = getattr(entry, "session_id", "")
    return (
        _TRUNCATED_AT.get(session_id) == turn_count
        or _UNATTRIBUTED_AT.get(session_id) == turn_count
    )


def oldest_pending_summary(
    *, skip_modes: frozenset[str] = frozenset(), now: float | None = None,
) -> tuple[int, str | None]:
    """要約待ちのうち、会話が止まってから最も長く待っている会話の ``(経過秒, session_id)``。

    会話が止まった時刻はセッションの ``ended_at`` (ターンを保存するたびに更新される)、
    無ければ ``started_at``。新しい永続化は持たない。待ちが無い / 読めないときは ``(0, None)``。
    """
    from backend.free.history.history_manager import get_history_manager

    try:
        mgr = get_history_manager()
        index = mgr._load_index()
    except Exception as exc:  # noqa: BLE001 - 観測のための読み出しで落とさない
        logger.debug("Failed to read the history index for summary ages: %s", exc)
        return 0, None
    now = time.time() if now is None else now
    oldest: tuple[int, str | None] = (0, None)
    for entry in index.sessions:
        if not needs_summary(entry) or _held_back(entry, skip_modes):
            continue
        try:
            session = mgr.get_session(entry.session_id)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to read session %s: %s", entry.session_id, exc)
            continue
        if session is None or not session.turns:
            continue
        stopped = parse_utc(session.ended_at or session.started_at)
        if stopped is None:
            continue
        age = max(0, int(now - stopped.timestamp()))
        if oldest[1] is None or age > oldest[0]:
            oldest = (age, entry.session_id)
    return oldest


def needs_summary(entry: object) -> bool:
    """要約の対象か: 未要約、または要約後に ``RESUMMARIZE_MIN_TURNS`` 以上伸びた。

    選別・持ち越し件数のログ・死活監視の入力件数が **同じ条件** を見る。以前は
    持ち越し件数だけ「1 ターンでも伸びたら対象」で数えており、選別 (10 ターン
    以上) と食い違ってログの件数が過大に出ていた (2026-09-21)。
    """
    summary = getattr(entry, "summary", None)
    if summary is None:
        return True
    turn_count = int(getattr(entry, "turn_count", 0) or 0)
    summary_turn_count = int(getattr(entry, "summary_turn_count", 0) or 0)
    return turn_count - summary_turn_count >= RESUMMARIZE_MIN_TURNS


def count_sessions_needing_summary() -> int:
    """要約の対象になっているセッション数 (Step 8-9 の入力件数)。失敗時は 0。"""
    from backend.free.history.history_manager import get_history_manager

    try:
        index = get_history_manager()._load_index()
    except Exception as exc:  # noqa: BLE001 - 観測のための読み出しで落とさない
        logger.debug("Failed to count sessions needing summary: %s", exc)
        return 0
    return sum(1 for entry in index.sessions if needs_summary(entry))


async def summarize_unsummarized_sessions(
    llm_client: Any,
    embedder: "EmbeddingBackend",
    *,
    batch_size: int = 5,
    is_cancelled: Callable[[], bool] | None = None,
    should_pause: Callable[[], bool] | None = None,
    skip_modes: frozenset[str] = frozenset(),
    first_session_id: str | None = None,
) -> int:
    """未要約セッションに LLM 要約 + 埋め込みベクトルを生成する。

    処理手順:

    1. :func:`~backend.free.history.history_manager.get_history_manager` で
       シングルトンを取得 (失敗時は warning ログ + ``0`` 返却)。
    2. インデックスから ``summary is None`` のセッションを順次取得
       (1 サイクルあたり ``batch_size`` 件まで)。
    3. LLM に「以下の会話を 1-2 文で要約してください」プロンプト (各文に
       ユーザー / アシスタントの主語を求める) を投げ、末尾 20 ターン (各 200 文字まで)
       を入力とする。主語の無い文が残れば 1 回だけ作り直し、それでも残れば保存しない
       (:func:`_generate_attributed_summary`)。
    4. 生成された要約を :meth:`HistoryManager.update_session_fields` で
       セッションへ書く (書き手スレッドの上で追記ログと一緒に畳むので、要約中に
       届いたターンを落とさない。索引も同時に更新される)。
    5. embedder で要約の埋め込みを作り
       :meth:`HistoryManager.put_summary_embedding` で束へ入れる。

    Args:
        llm_client: 要約生成に使う LLM クライアント。``generate`` async メソッド
            が必須 (``messages=..., stream=False, max_tokens=128,
            id_slot=<background_slot>`` を受け付けること)。
        embedder: 要約の埋め込みベクトルを生成する
            :class:`~backend.free.rag.embedding_backend.EmbeddingBackend`。
        batch_size: 1 サイクルで要約する最大セッション数
            (config ``history.summary_batch_size``)。``0`` 以下は無制限。
        is_cancelled: キャンセル判定コールバック (``True`` で途中中断)。
        should_pause: ``True`` を返したらセッション境界でループを打ち切る
            協調 yield。未要約のセッションは ``summary`` が付かないままなので
            次サイクルが拾う。
        skip_modes: このサイクルで要約しないセッションの mode
            (:func:`chat_summary_skip_modes`)。見送ったセッションは次サイクルが拾う。
        first_session_id: 先に要約する会話 (待ちすぎた最古の会話、
            :func:`oldest_pending_summary`)。打ち切らせない 1 回をこの会話に充てる。

    Returns:
        実際に要約を生成できたセッション数。
    """
    from backend.free.history.history_manager import get_history_manager

    try:
        mgr = get_history_manager()
    except Exception as exc:
        logger.warning("Failed to init HistoryManager for step 8-9: %s", exc)
        return 0

    index = mgr._load_index()
    summarized = 0
    #: should_pause 発火時に「まだ要約が必要な件数」を報告するための分母。
    pending_total = sum(1 for e in index.sessions if needs_summary(e))
    attempted = 0
    entries = list(index.sessions)
    if first_session_id is not None:
        entries.sort(key=lambda e: getattr(e, "session_id", None) != first_session_id)

    for entry in entries:
        if is_cancelled is not None and is_cancelled():
            break
        # 協調 yield: チャット生成が走っている間はセッション境界で手を止める
        # (note_evolver と同じ実測。CLAUDE.md 不変則 #1)。
        if should_pause is not None and should_pause():
            remaining = pending_total - attempted
            if remaining > 0:
                logger.info(
                    "Step 8-9 summarization paused for the user turn: "
                    "%d session(s) left pending for the next cycle", remaining,
                )
            break
        if batch_size > 0 and summarized >= batch_size:
            break
        # 未要約、または要約後に会話が伸びたセッションを対象にする。
        # 自動保存は毎ターン走るため、会話途中で要約が付くことがある。その要約を
        # 恒久化すると後半の訂正が要約に載らず、search_history 経由で訂正前の値が
        # 「独立した根拠」として再注入される (2026-07-26 ライブ検証: 火曜→水曜と
        # 訂正済みの予約が過去セッションの要約から火曜へ巻き戻った)。
        if not needs_summary(entry):
            # 1〜2 ターン伸びるたびに作り直さない — 同じセッションが 20 サイクルで
            # 26 回要約されていた (2026-09-12 実測、入力は末尾 20 ターン固定)。
            continue
        if _held_back(entry, skip_modes):
            continue
        turn_count = int(getattr(entry, "turn_count", 0) or 0)

        session = mgr.get_session(entry.session_id)
        if session is None or not session.turns:
            continue

        attempted += 1

        # 入力は **末尾** の 20 ターン。先頭 20 ターン固定だと、再要約の条件
        # (会話が伸びた) を満たしても入力が変わらず、20 ターン目以降の訂正は
        # 要約に決して載らなかった (2026-09-02 監査 H7)。
        turns_text = "\n".join(
            f"{t.get('role', 'user')}: {t.get('content', '')[:200]}"
            for t in session.turns[-_SUMMARY_TURN_WINDOW:]
        )
        try:
            summary, reason = await _generate_attributed_summary(
                llm_client, turns_text, entry.session_id,
            )
            if summary is None:
                held = _TRUNCATED_AT if reason == "truncated" else _UNATTRIBUTED_AT
                held[entry.session_id] = turn_count
                continue
            _TRUNCATED_AT.pop(entry.session_id, None)
            _UNATTRIBUTED_AT.pop(entry.session_id, None)
            emb = await embedder.embed([summary], is_query=False)
            # 要約の基にしたターン数を刻む。会話がここから伸びたら次回作り直す。
            fields: dict[str, Any] = {
                "summary": summary, "summary_turn_count": len(session.turns),
            }
            if not session.lang:
                fields["lang"] = detect_lang(summary)
            # 書き手スレッドの結果を待つので、ループの外で呼ぶ (f_02 §4.3)。
            written = await run_in_executor_with_context(
                asyncio.get_running_loop(), None,
                partial(mgr.update_session_fields, entry.session_id, **fields),
            )
            if not written:
                continue
            # ベクトルは埋め込みモデルごとの束へ (埋め込みモデルの model_key が鍵、無い構成は
            # model_name。無記名だと替えた後に新旧のベクトルが次元一致だけで見分けられない)。
            mgr.put_summary_embedding(
                entry.session_id, emb[0].tolist(),
                getattr(embedder, "model_key", "") or embedder.model_name(),
            )
            summarized += 1
        except Exception as exc:
            if isinstance(exc, TimeoutError) and getattr(exc, "contended", False):
                # チャットに打ち切られた。次のセッションも次のターンに打ち切られるだけなので、
                # 残りは次の窓へ回す (ここで続けると未要約の数だけ毎ターン無駄な dispatch が
                # 走り、2026-10-06 ライブ監査では 1 サイクルで 5〜6 回の警告になった)。
                logger.info(
                    "Step 8-9 summarization preempted by a chat request; "
                    "%d session(s) left pending for the next window",
                    pending_total - attempted + 1,
                )
                break
            logger.warning(
                "Failed to summarize session %s: %s", entry.session_id, exc,
            )

    if summarized > 0:
        logger.info("Summarized %d sessions in step 8-9", summarized)

    return summarized


__all__ = ["summarize_unsummarized_sessions"]
