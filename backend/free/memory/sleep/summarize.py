"""Step 8-9: 未要約セッションの LLM 要約生成

``sleep_update.SleepTimeWorker._step8_9_summarize_sessions`` として実装されていた
要約生成ロジックを独立 module に切り出したもの。

処理は ``HistoryManager.index.sessions`` のうち ``summary is None`` かつ
``session.turns`` が存在するセッションを対象に、補助タスクで 1-2 文の要約を
生成し、 続いて embedder で要約埋め込みベクトルを計算する。
生成した要約とベクトルはセッションファイル + インデックスに永続化する。

本 module は EvorefMem pillar 内部扱いで、LLM 呼び出しは caller から受け取った
``llm_client`` に閉じる (EvorefGen pillar の Protocol に準拠する抽象 client)。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.free.rag.embedding_backend import EmbeddingBackend

logger = get_logger("memory.sleep.summarize")

#: 要約プロンプトへ載せるターン数 (末尾から)。
_SUMMARY_TURN_WINDOW = 20


async def summarize_unsummarized_sessions(
    llm_client: Any,
    embedder: "EmbeddingBackend",
    *,
    batch_size: int = 5,
    is_cancelled: Callable[[], bool] | None = None,
) -> int:
    """未要約セッションに LLM 要約 + 埋め込みベクトルを生成する。

    処理手順:

    1. :func:`~backend.free.history.history_manager.get_history_manager` で
       シングルトンを取得 (失敗時は warning ログ + ``0`` 返却)。
    2. インデックスから ``summary is None`` のセッションを順次取得
       (1 サイクルあたり ``batch_size`` 件まで)。
    3. LLM に「以下の会話を 1-2 文で要約してください」プロンプトを投げ、
       末尾 20 ターン (各 200 文字まで) を入力とする。
    4. 生成された要約をセッションオブジェクトに保存し、続けて embedder で
       埋め込みベクトルを作り ``summary_embedding`` に格納する。
    5. :meth:`HistoryManager.save_session` で永続化、インデックス更新。
    6. 最後に ``_save_index`` で index.json を再書き込み。

    Args:
        llm_client: 要約生成に使う LLM クライアント。``generate`` async メソッド
            が必須 (``messages=..., stream=False, max_tokens=128,
            id_slot=<background_slot>`` を受け付けること)。
        embedder: 要約の埋め込みベクトルを生成する
            :class:`~backend.free.rag.embedding_backend.EmbeddingBackend`。
        batch_size: 1 サイクルで要約する最大セッション数
            (config ``history.summary_batch_size``)。``0`` 以下は無制限。
        is_cancelled: キャンセル判定コールバック (``True`` で途中中断)。

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

    for entry in index.sessions:
        if is_cancelled is not None and is_cancelled():
            break
        if batch_size > 0 and summarized >= batch_size:
            break
        # 未要約、または要約後に会話が伸びたセッションを対象にする。
        # 自動保存は毎ターン走るため、会話途中で要約が付くことがある。その要約を
        # 恒久化すると後半の訂正が要約に載らず、search_history 経由で訂正前の値が
        # 「独立した根拠」として再注入される (2026-07-26 ライブ検証: 火曜→水曜と
        # 訂正済みの予約が過去セッションの要約から火曜へ巻き戻った)。
        if entry.summary is not None and entry.turn_count <= entry.summary_turn_count:
            continue

        session = mgr.get_session(entry.session_id)
        if session is None or not session.turns:
            continue

        # 入力は **末尾** の 20 ターン。先頭 20 ターン固定だと、再要約の条件
        # (会話が伸びた) を満たしても入力が変わらず、20 ターン目以降の訂正は
        # 要約に決して載らなかった (2026-09-02 監査 H7)。
        turns_text = "\n".join(
            f"{t.get('role', 'user')}: {t.get('content', '')[:200]}"
            for t in session.turns[-_SUMMARY_TURN_WINDOW:]
        )
        try:
            result = await llm_client.generate(
                messages=[{
                    "role": "user",
                    "content": f"以下の会話を1-2文で要約してください:\n\n{turns_text}",
                }],
                stream=False,
                max_tokens=128,
                purpose="summarize",
                id_slot=getattr(llm_client, "background_slot", -1),
            )
            session.summary = (
                result["choices"][0]["message"]["content"].strip()
            )
            # 要約の基にしたターン数を刻む。会話がここから伸びたら次回作り直す。
            session.summary_turn_count = len(session.turns)

            emb = await embedder.embed([session.summary], is_query=False)
            session.summary_embedding = emb[0].tolist()

            mgr.save_session(session)

            entry.summary = session.summary
            entry.summary_turn_count = session.summary_turn_count
            summarized += 1
        except Exception as exc:
            logger.warning(
                "Failed to summarize session %s: %s", entry.session_id, exc,
            )

    if summarized > 0:
        mgr._save_index(index)
        logger.info("Summarized %d sessions in step 8-9", summarized)

    return summarized


__all__ = ["summarize_unsummarized_sessions"]
