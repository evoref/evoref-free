"""統合検索パイプライン: エピソード記憶 + カートリッジ + Self-RAG（asyncio.gather 並列検索）"""

from __future__ import annotations

import asyncio
import re
import zlib
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from backend.exceptions import RAGError
from backend.i18n_helper import prompt_locale
from backend.log_config import get_logger
from backend.free.core.date_math_cue import (
    day_count_closed_in_query,
    day_count_is_the_only_cue,
    query_has_date_math_cue,
)
from backend.free.core.inference import eligible_rag_indices
from backend.free.core.intent_vocab import (
    RECALL_IN_SESSION_MIN_USER_TURNS,
    excludes_current_conversation,
    only_proximal_recall,
    only_session_ordinal_recall,
    points_to_past_session,
    recall_form_points_into_conversation,
    refers_to_ongoing_session,
    refers_to_previous_output,
    session_user_turns,
)
from backend.free.core.session_mode import is_create_mode
from backend.free.memory.corrections import corrections_by_target
from backend.free.memory.episodic.note import record_mode
from backend.trace_context import run_in_executor_with_context
from backend.utils import utc_now_dt
from backend.free.constants import (
    SEARCH_HISTORY_CURRENT_SESSION_HEADER,
    SEARCH_HISTORY_NO_RESULTS_PREFIX,
    SEARCH_HISTORY_OTHER_SESSIONS_HEADER,
)
from backend.free.rag.chunk_content_gate import ChunkContentGate, GateConfig
from backend.free.rag.evidence.types import compute_claim_key
from backend.free.rag.rerank_llamacpp import degenerate_reason, fit_rerank_pool, rerank_pair_cost
from backend.free.core.inference import SUMMARY_TAIL_RE
from backend.free.core.intent_vocab import NUMERAL_HINT_RE
from backend.free.rag.self_rag_judge import (
    QualityThresholds,
    RetrievalNecessityJudge,
    RetrievalQualityJudge,
)

if TYPE_CHECKING:

    from backend.free.core.policy_interpreter import PolicyInterpreter
    from backend.free.core.stage_timer import StageTimer
    from backend.free.rag.judge_usage_tracker import JudgeUsageTracker

logger = get_logger("memory.search_pipeline")

# STM / LTM 用スレッドプール（設計書 4.10.1）
_search_executor = ThreadPoolExecutor(max_workers=3)

#: カートリッジ検索専用のスレッドプール。
#:
#: ``asyncio.wait_for`` はコルーチン側を打ち切るだけで、executor へ投げた
#: 関数は走り続ける。STM / LTM と同じプールを共有していると、タイムアウトした
#: カートリッジ検索がワーカーを占有したまま次ターンの STM / LTM 検索を待たせる
#: (3 ワーカーが全部塞がれば記憶検索そのものが止まる)。専用プールへ隔離し、
#: 影響をカートリッジ層の中に閉じる。
_cartridge_executor = ThreadPoolExecutor(max_workers=2)

#: ストア横断の共通レコード ``(id, cosine, score, text)`` (c_16 §6.3 / §7.2)。
#:
#: 3 ストア (episodic / semantic / corpus) の ``search()`` はすべて **素の
#: cosine** (ゲート用) と **順位式のスコア** (``cos × freshness × confidence ×
#: store_prior``、ストアの中で計算済み) を分けて返す。層ごとに違う式で並べる
#: / 層内正規化で揃える / RRF で混ぜる、はすべて廃止した (c_16 §7.2)。
type StoreEntry = tuple[str, float, float, str]


def gate_view(entries: list[StoreEntry]) -> list[tuple[str, float, str]]:
    """ゲート用の射影 ``(id, cosine, text)`` を cosine 降順で返す。

    品質判定 / content gate / relevance floor は cosine スケール前提で閾値が
    決まっている (c_16 §7.1「ゲートは素の cosine のみ」)。
    """
    view = [(cid, cos, text) for cid, cos, _score, text in entries]
    view.sort(key=lambda item: -item[1])
    return view


def rank_view(entries: list[StoreEntry]) -> list[tuple[str, float, str]]:
    """順位付け用の射影 ``(id, score, text)`` (入力の並びを保つ)。"""
    return [(cid, score, text) for cid, _cos, score, text in entries]


@dataclass(frozen=True)
class SearchUsage:
    """採用した id の「使った」記録を、注入が確定するまで保留する (f_01 §8.1 Step 7.7)。

    検索しただけで記録すると、結果を捨てた回 (投機検索を取り消した reactive /
    軽量パス) でも ``last_used_at`` と疑似クエリの lazy 生成対象 (f_01 §6.4) が
    汚れる。記録するのは注入を組んだ消費側 (``chat._build_messages_with_search``)
    で、:meth:`commit` を 1 回呼ぶ。バッファはプロセス内で、ディスクへは
    sleep-time が書く (c_16 §2.1)。
    """

    episodic: Any
    cartridge_mgr: Any
    sources: tuple[tuple[str, float, str], ...]
    record_corpus_hits: bool
    #: 転置索引の上位候補 (採用に関わらず lazy 生成の対象、f_01 §6.4)。
    pq_seeds: list[str] = field(default_factory=list)

    def commit(self) -> None:
        """採用した id を episodic の ``usage`` と corpus の lazy 生成対象へ積む。"""
        if self.pq_seeds and self.cartridge_mgr is not None:
            try:
                self.cartridge_mgr.record_pq_hits(self.pq_seeds)
            except Exception as e:  # noqa: BLE001 — 観測のための記録で応答を止めない
                logger.warning("Failed to record lexical pq seeds: %s", e)
        _record_episodic_usage(self.episodic, list(self.sources))
        if self.record_corpus_hits:
            _record_corpus_hits(self.cartridge_mgr, list(self.sources))


@dataclass
class SearchResult:
    """統合検索の結果"""
    sources: list[tuple[str, float, str]] = field(default_factory=list)
    quality: str = "low"
    from_memory: bool = False
    skipped: bool = False
    #: ``sources`` に採用されたチャンクの **生スコア** (cosine スケール) の最大値。
    #: ``sources`` 側のスコアは順位式 (``cos × freshness × confidence ×
    #: store_prior``) の値で、cosine スケールではないため検索品質の観測値には
    #: 使えない。品質判定・gate と同じ「判定は生スコア」の不変則
    #: (docs/f_01 §8.3 / c_16 §7.1) に合わせるための観測用フィールド。
    top_raw_score: float | None = None

    #: このターンで実際に注入した Evidence の id (``<store>:<evidence_id>``、
    #: c_16 §5.5)。``corpus:`` は corpus パッケージ由来、``episodic:`` は
    #: 会話ノート由来。学習帰属 (``GenerationConfigRef.evidence_ids``) の材料。
    evidence_ids: list[str] = field(default_factory=list)

    #: 疑似クエリの関連性ゲート (f_01 §6.6) で corpus を引かなかった turn。
    #: Level 0 の ``gen_config.corpus_gated`` へ (「注入ゼロ」の理由の区別)。
    corpus_gated: bool = False
    #: 採用した corpus チャンクのうち疑似クエリ索引経由で拾った件数。
    pseudo_derived: int = 0
    #: 転置索引の上位候補 (``<pkg>:<ev>``、採用の有無に関わらず)。抑止応答の
    #: turn で取りこぼした問いの種 (f_01 §6.4 の misses) に使う。
    lexical_candidate_ids: list[str] = field(default_factory=list)
    #: corpus の候補はあったのに棒で全部落ち、1 件も注入できなかった turn
    #: (f_01 §6.4 の misses の第 2 の引き金。作話の温床なので問いを種にする)。
    corpus_starved: bool = False
    #: 採用した id の「使った」記録。注入を確定した消費側が ``commit()`` する。
    usage: SearchUsage | None = None
    #: 関連度の床 (Step 6.5) で落とした episodic の id (疑似クエリ・corpus を除く)。
    #: 記憶の注入 (``MemoryInjector``) が同じノートを低い棒で拾い直さないために
    #: 渡す (2026-10-05 trace 8232204694b3: 床 0.670 で落ちたノートが注入の棒
    #: 0.617 で戻り、別の会話の旅程が載った)。
    episodic_rejected_ids: list[str] = field(default_factory=list)


def _resolve_fetch_multiplier(cfg: dict) -> int:
    """候補拡張倍率を解決する。

    ``rag.fetch_multiplier`` (既定 1 = 拡張なし) を用いて各層の取得件数を
    ``top_k * N`` へ広げ、「広く取って絞る」第1段として候補プールを確保する。
    値は [1, 5] にクランプする。
    """
    rag_cfg = cfg.get("rag") or {}
    multiplier = rag_cfg.get("fetch_multiplier", 1)
    try:
        multiplier = int(multiplier)
    except (TypeError, ValueError):
        multiplier = 1
    return max(1, min(5, multiplier))


def _resolve_rescore_candidates(rag_cfg: dict) -> int:
    """``rag.rescore_candidates`` (int8 粗検索後の float32 rescore 候補数)。

    0 以下 / 不正値は 0 (= ``VectorStore.search`` の内部既定 ``max(50, top_k*3)``)。
    """
    try:
        value = int(rag_cfg.get("rescore_candidates", 0) or 0)
    except (TypeError, ValueError):
        return 0
    return max(0, value)


def _resolve_search_params(
    policy: PolicyInterpreter | None,
    rag_cfg: dict,
    mode: str,
) -> tuple[int, int, float]:
    """ポリシー優先で `(top_k, stm_top_k, noise_sigma)` を解決する。

    ポリシー未設定 / キー欠落時は config + ハードコードのデフォルトへフォールバック。
    """
    top_k = rag_cfg.get("top_k", 5)
    stm_top_k = 3
    noise_sigma = 0.05
    if policy is None:
        return top_k, stm_top_k, noise_sigma
    try:
        top_k = policy.get("search", "top_k", mode)
    except KeyError:
        pass
    try:
        stm_top_k = policy.get("search", "stm_top_k", mode)
    except KeyError:
        pass
    try:
        noise_sigma = policy.get("search", "noise_sigma", mode)
    except KeyError:
        pass
    return top_k, stm_top_k, noise_sigma


def adopted_top_k(policy: PolicyInterpreter | None, rag_cfg: dict, mode: str) -> int:
    """Step 7 が採用する件数 (``top_k``) を返す (問い分解の影の空き枠、f_01 §8.1 の 7.9)。"""
    return int(_resolve_search_params(policy, rag_cfg, mode)[0])


#: ``compress_turn(style="summary")`` の圧縮マーク (ja / en) と末尾の元文字数。
#: 生成側 (``backend.utils.compress_turn``) は ja 固定なので、こちらは両方を
#: 剥がせるようにしておく (locale 切替で剥がし漏れを起こさないため)。
_SUMMARY_MARKS: tuple[str, ...] = ("[要約] ", "[summary] ")

#: 「同じ質問の繰り返し」と見なす最小文字数。短い相槌 (「ありがとう」「はい」) は
#: 何度でも出るので、これを繰り返し扱いにすると assistant ノートを不当に落とす。
_REPEAT_MIN_CHARS = 12


def _normalize_utterance(text: str) -> str:
    """発話の同一判定用の正規化 (空白除去 + 圧縮マーク剥がし)。"""
    body = (text or "").strip()
    for mark in _SUMMARY_MARKS:
        if body.startswith(mark):
            body = body[len(mark):]
            break
    body = SUMMARY_TAIL_RE.sub("", body)
    return "".join(body.split())


def _is_repeat_of_a_stored_turn(query: str, notes: list) -> bool:
    """今回のクエリが、保存済みの user 発話の焼き直しか (純粋関数)。

    「前にも同じことを聞いた」状態を検出する。この状態でだけ、検索結果に含まれる
    assistant の発話 = **その質問への前回の回答** が意味を持ってしまう
    (モデルがそれを丸写しする)。
    """
    q = _normalize_utterance(query)
    if len(q) < _REPEAT_MIN_CHARS:
        return False
    for note in notes:
        if getattr(note, "source", "user") != "user":
            continue
        c = _normalize_utterance(getattr(note, "content", "") or "")
        if not c:
            continue
        if c == q or c.startswith(q) or q.startswith(c):
            return True
    return False


def query_repeats_a_stored_turn(episodic, query: str) -> bool:
    """短期ノートを走査して「前にも同じことを聞いたか」を判定する。

    層検索は ``asyncio.gather`` で並列に走るため、検索結果を待ってから扱いを
    決めることはできない。判定に必要なのはベクトル検索ではなく保存済みノートの
    文字列一致だけなので、層を起動する **前** に同期で済ませる
    (``short`` tier のノートは snapshot 単位でキャッシュされる)。
    """
    if episodic is None or not query:
        return False
    try:
        notes = episodic.short_notes()
    except Exception:
        return False
    return _is_repeat_of_a_stored_turn(query, notes)




#: 訂正済みノートに付ける注記 (``i18n.prompt_locale`` 別)。SemMem 側の
#: ``(訂正後の記録)`` (``pipeline.injector._render_fact``) と対になる。
#: en 側を変えるときは ``core.inference._normalize_for_frame_dedup`` も揃える。
_SUPERSEDED_MARKS: dict[str, str] = {
    "ja": "（後に訂正された古い値）",
    "en": " (superseded by a later correction)",
}
#: 従来名 (ja)。テスト / 外部参照の後方互換用。
_SUPERSEDED_MARK = _SUPERSEDED_MARKS["ja"]


def _superseded_mark() -> str:
    """prompt_locale に応じた訂正済み注記 (未知 locale は ja)。"""
    return _SUPERSEDED_MARKS.get(prompt_locale(), _SUPERSEDED_MARKS["ja"])


#: 前チャンクの文脈の見出し語 (f_01 §8.1 の 7.65)。
PREVIOUS_CONTEXT_MARK = "(前の文脈) "


def attach_previous_chunk_context(
    cartridge_mgr, sources: list[tuple[str, float, str]], *, tail_chars: int = 300,
) -> list[tuple[str, float, str]]:
    """採用した corpus チャンクごとに直前チャンクの末尾を随伴させる (f_01 §8.1 の 7.65)。

    chunker v3 で節の途中から始まるチャンクは減ったが、文単位で詰めた長い節では
    答えの主語が直前のチャンクに残る。実測 (v3、golden 135 件): 0.830 → 0.859。
    随伴は **同じ文書・同じ大節** の直前だけ、既に採用済みなら足さない。置き場は元チャンクの
    **直前** (本文の流れのとおり。予算選別は元の並びを保つので、末尾に足すとプロンプトの最後に
    元チャンクから離れて残る)。
    """
    if tail_chars <= 0 or cartridge_mgr is None or not sources:
        return sources
    fn = getattr(cartridge_mgr, "previous_chunk_context", None)
    if fn is None:
        return sources
    present = {cid for cid, _, _ in sources}
    out: list[tuple[str, float, str]] = []
    added = 0
    for entry in sources:
        cid, score, _text = entry
        if _store_of(cid) != "corpus":
            out.append(entry)
            continue
        try:
            found = fn(cid, tail_chars)
        except Exception as e:  # noqa: BLE001 — 随伴で応答を止めない
            logger.debug("previous chunk context skipped for %s: %s", cid, e)
            found = None
        if found:
            prev_id, prev_text = found
            if prev_id not in present and prev_text:
                present.add(prev_id)
                out.append((prev_id, score, PREVIOUS_CONTEXT_MARK + prev_text))
                added += 1
        out.append(entry)
    if added:
        logger.info(
            "Attached %d previous-chunk context(s) (tail %d chars) to corpus references",
            added, tail_chars,
        )
    return out


def attach_superseding_corrections(
    episodic, sources: list[tuple[str, float, str]],
    correction_trail: dict[str, str] | None = None,
) -> list[tuple[str, float, str]]:
    """採用済みチャンクのうち訂正されたものに注記を付け、訂正本文を随伴させる。

    relevance floor は **生の cosine** に掛かるため、訂正ノートにスコア加点を
    しても救済できない (``_PIN_RETRIEVAL_BOOST`` 方式では届かない)。訂正は
    省略形で話題語を落としているぶん、対象より必ず類似度が低くなる。

    実インシデント (2026-08-19 ライブ監査): 新セッションで
    「あさひプロジェクトの締切はいつでしたか？」に対し、訂正**前**の
    「…締切は9月30日です。」が 0.7294 で採用され、訂正
    「訂正します、締切は10月15日に変更になりました。」は floor 0.302 に
    届かず落ちて、**訂正前の値が回答された**。

    被訂正ノートを落とすのではなく **両方残す**。訂正は話題語を落としている
    ため単独では「何の締切か」が失われる。

    宛先は 2 つの情報源の **和** で決める:

    1. ``corrections.corrections_by_target`` — 同一セッションの user ノート
       同士を語の重なりで結ぶ。短期シャードのノートだけを見る。
    2. ``correction_trail`` — SemMem の世代 (``superseded_by`` +
       ``from_correction``) から引いた ``{被訂正ノート id: 現在値}``。
       **セッションにも短期 / 長期の別にも依存しない。**

    1 だけだと、セッションを跨いだ訂正が 1 件も解けない。実インシデント
    (2026-09-14 監査 F-01): 自己紹介 (T01) の「建設会社で施工管理」が別
    セッション (T07) で訂正されたのに注記が付かず、``[参考情報]`` に原文が
    素のまま載って、モデルが *「参考情報1の建設会社を優先して記載しています」*
    と **明示的に訂正前を採用** した。記憶がセッションを跨いで持続する製品で、
    訂正だけ跨げないのは構造的な穴だった。
    """
    if episodic is None or not sources:
        return sources
    try:
        notes = episodic.short_notes()
    except Exception:
        notes = []
    kept_ids = {cid for cid, _, _ in sources}
    links = [
        (target_id, getattr(corr, "id", ""), getattr(corr, "content", "") or "")
        for target_id, corr in corrections_by_target(notes).items()
        if target_id in kept_ids
    ]
    # SemMem 由来の宛先を足す (同一セッションで解けた分は上書きしない —
    # そちらは訂正発話の本文を随伴でき、文脈が濃い)。
    resolved = {target_id for target_id, _, _ in links}
    for target_id, current in (correction_trail or {}).items():
        if target_id in kept_ids and target_id not in resolved and current:
            # 訂正ノート本体は手元に無いので、現在値の言明を随伴させる。
            # id は注記用の合成キー (episodic のノート id と衝突しない)。
            links.append((target_id, f"semmem:{target_id}", current))
    if not links:
        return sources

    marked = {sid for sid, _, _ in links}
    present = {cid for cid, _, _ in sources}
    mark = _superseded_mark()
    out: list[tuple[str, float, str]] = []
    for chunk_id, score, text in sources:
        if chunk_id in marked and not (text or "").rstrip().endswith(mark):
            text = f"{(text or '').rstrip()}{mark}"
        out.append((chunk_id, score, text))
    for superseded_id, corr_id, corr_text in links:
        if corr_id and corr_id not in present and corr_text:
            present.add(corr_id)
            score = next(
                (s for cid, s, _ in sources if cid == superseded_id), 0.0,
            )
            out.append((corr_id, score, corr_text))
    logger.info(
        "Attached %d correction note(s) alongside superseded reference(s): %s",
        len(links), ", ".join(sid for sid, _, _ in links),
    )
    return out

async def _search_episodic_layer(
    episodic, query: str, query_vec: np.ndarray, top_k: int,
    drop_past_answers: bool = False,
    threshold: float = 0.0,
    own_session: str | None = None,
    create_session: str | None = None,
    mode: str = "chat",
    swapped_sink: set[str] | None = None,
    session_id: str | None = None,
    user_turns: int | None = None,
) -> list[StoreEntry]:
    """エピソード記憶 (``short`` → ``long``) を 1 回で引く。

    ``create_session`` (create モードのターンのセッション id) が与えられたら、
    別セッションの create ターンのノートを外す (:func:`_drop_other_create_sessions`)。

    ``own_session`` が与えられたら、provenance の ``session_id`` が一致する
    ノートだけ残す。自分の直前の出力を指す問い (「今の 4 つの回答は…」) は
    それを出したセッションのノートしか根拠になり得ず、別セッションのヒットは
    どれほど似ていても誤り (f_01 §8.1、2026-09-12 (b))。

    旧 STM 層 + LTM 層の置き換え (c_16 §4.1)。``EpisodicStore.search`` が
    ``short`` シャードを先に、続いて直近 12 か月の ``long`` シャードを引き、
    順位式 (``cos × freshness × confidence × store_prior``) で並べる。

    Args:
        swapped_sink: 問いを答えへ差し替えた行 (assistant 由来) の id を足す先
            (:func:`_answers_for_question_only_hits`)。
        session_id: 現在のセッション id。与えられたら、別セッションの問いを
            答えへ差し替えるのは、今の問いが過去の会話を指し、かつ問いか答えが今の問いの
            内容語を持つときだけにする (:func:`_answers_for_question_only_hits`)。
            ``None`` なら出所を比べない。
        user_turns: 会話の利用者の発話数 (今回を含む、``session_user_turns``)。
            1 ターン目は近接語・位置語も過去の会話を指すと読む
            (:func:`~backend.free.core.intent_vocab.points_to_past_session`)。
            ``None`` なら先行ターンがあるものとして読む。
        threshold: **素の cosine** のゲート (c_16 §7.1)。較正が効いている
            ときだけ呼出側が較正済み ``relevance`` を渡す。較正が無い構成で
            静的閾値を渡すと、埋め込みモデル次第で到達不能になり黙って全件
            落とすため 0.0 のままにする (呼出側 :func:`_store_cosine_gate`)。

    Returns:
        ``(id, cosine, score, text)`` のリスト。``cosine`` はゲート用の素の
        値、``score`` は順位式の値。品質判定 / floor / content gate は
        cosine スケール前提で閾値が決まっている (c_16 §7.1)。

    ``retracted`` / ``superseded`` (訂正で畳まれたノート、要約に吸収された
    ノート) はストア側のアクティブマスクが落とすので、呼出側で退役 id を
    集める必要は無くなった (c_16 §8: ``_collect_retired_note_ids`` の廃止)。
    """
    if episodic is None:
        return []
    loop = asyncio.get_running_loop()
    try:
        hits = await run_in_executor_with_context(
            loop, _search_executor,
            lambda: episodic.search(
                query, query_vec, top_k, threshold=threshold,
            ),
        )
    except asyncio.CancelledError:
        raise
    except (RuntimeError, ValueError, TypeError, OSError) as e:
        logger.warning("Episodic search failed: %s", e)
        return []

    if own_session is not None:
        before = len(hits)
        hits = [h for h in hits if _hit_session(h) == own_session]
        if len(hits) != before:
            logger.info(
                "Episodic: the query refers to our own previous output; dropped "
                "%d note(s) from other sessions (kept %d of session %s)",
                before - len(hits), len(hits), own_session,
            )
    if create_session is not None:
        hits = _drop_other_create_sessions(hits, create_session)
    if drop_past_answers:
        before = len(hits)
        hits = [h for h in hits if h.record.origin == "user"]
        if len(hits) != before:
            logger.info(
                "Episodic: this query repeats an earlier turn; dropped %d "
                "assistant note(s) so the previous answer is not handed back "
                "as reference material", before - len(hits),
            )
    else:
        hits = _answers_for_question_only_hits(
            episodic, hits, query, mode, swapped_sink=swapped_sink,
            session_id=session_id, user_turns=user_turns,
        )
    hits = _latest_statement_per_slot(hits)

    entries: list[StoreEntry] = []
    n_dropped = n_cleaned = 0
    for hit in hits:
        text = hit.text
        if _is_injected_output(text):
            n_dropped += 1
            continue
        trimmed = _trim_file_payload(text)
        cleaned = _sanitize_episode_chunk(trimmed)
        if cleaned is None:
            n_dropped += 1
            continue
        if cleaned != text:
            n_cleaned += 1
        entries.append((hit.id, hit.cosine, hit.score, cleaned))
    if n_cleaned or n_dropped:
        logger.info(
            "Episodic: stripped internal ids from %d note(s), dropped %d "
            "that carried nothing but ids or our own injected output",
            n_cleaned, n_dropped,
        )
    logger.debug(
        "Episodic layer: %d hit(s), score=[%s] cosine=[%s]",
        len(entries),
        ", ".join(f"{e[2]:.3f}" for e in entries),
        ", ".join(f"{e[1]:.3f}" for e in entries),
    )
    return entries


def episodic_session_scope(
    query: str, session_id: str, *, user_turns: int = 0,
) -> str | None:
    """エピソード検索を自セッションに閉じるべき問いなら、そのセッション id を返す。

    自分の直前の出力を指す問いは、それを出したセッションの外に根拠を持たない。
    進行中の会話を指す問い (「ここまでをまとめて」) も同じ — 別セッションの
    回答が似ていると、それを丸写しした (2026-09-21 ライブ監査 C10: セキュリティ
    の会話のまとめに、別セッションの Docker 本番チェックリストが返った)。

    ツール判定のガードが「答えは進行中の会話にある」と結論する 2 つの問いも
    同じ述語で閉じる (不変則 #14(a))。ガードだけがそう結論すると、エピソード
    検索が別セッションの答えを「さっきの話」として差し出す (2026-10-05 ライブ
    監査: 「さっきの距離は何マイル？」に別セッションのマラソンの換算「約 26.22
    マイル」が返った)。

    ガードは履歴検索を撃つと決まった問いにしか掛からないが、ここは全ターンで判定
    するので、近接語・位置語が会話の中のものを指す形
    (:func:`recall_form_points_into_conversation`、「さっきの距離」「最初に私が
    聞いた」) に限る。素の語 (「今朝娘が熱を出した」「最後に、息子の誕生日は」
    「asyncio で最初に終わったタスク」) では閉じない。そのうえで:

    - 履歴参照語が近接語だけ (:func:`only_proximal_recall`)。
    - 会話内の位置だけ (:func:`only_session_ordinal_recall`) で、現在の会話を除外
      していない (``tool_judge_guards._suppress_ordinal_recall_within_session`` と同じ規則)。

    どちらも会話の利用者の発話数 ``user_turns`` (今回を含む、
    :func:`~backend.free.core.intent_vocab.session_user_turns`) が
    :data:`RECALL_IN_SESSION_MIN_USER_TURNS` 以上のときだけ — ガードと同じ数え方と
    閾値。1 ターン目の「さっき」は指す先が会話の中に無い。

    「以前」「前回」「昨日」のような過去セッションを指す語があればどちらの述語も
    偽なので、別セッションも従来どおり引く。
    """
    if refers_to_previous_output(query) or refers_to_ongoing_session(query):
        return session_id
    if user_turns < RECALL_IN_SESSION_MIN_USER_TURNS:
        return None
    if not recall_form_points_into_conversation(query):
        return None
    if only_proximal_recall(query):
        return session_id
    if only_session_ordinal_recall(query) and not excludes_current_conversation(query):
        return session_id
    return None


def _hit_session(hit) -> str:
    """ヒットのノートが書かれたセッション id (provenance[0].session_id、無ければ "")。"""
    prov = getattr(hit.record, "provenance", None) or []
    first = prov[0] if prov and isinstance(prov[0], Mapping) else {}
    return str(first.get("session_id") or "")


def _drop_other_create_sessions(hits: list, session_id: str) -> list:
    """別セッションの create ターンのノートを外す (create モードのターン用、f_01 §8.1)。

    過去の制作依頼とその配信報告は、依頼が違えば別の成果物の構成でしかない。
    載せると骨組みがそれを再現し、誤った回の報告が次の回の参考になって自己増幅
    する (2026-09-27 再監査 trace 8e21c7fba11b: 家計簿 CLI の依頼に過去 3 回の
    家計簿の依頼と「… kakeibo_live2 の models.py … に書き込みました」が入り、
    成果物が依頼に無い ``kakeibo_live2`` フォルダの下に作られた)。判定は出所の
    メタデータ (``attrs.mode`` と provenance の ``session_id``) だけ。同じセッション
    のノートと chat ターンのノートは残す。問いを答えへ差し替える
    (:func:`_answers_for_question_only_hits`) 前に掛けるので、対ごと消える。
    """
    kept = [
        h for h in hits
        if not is_create_mode(record_mode(h.record)) or _hit_session(h) == session_id
    ]
    if len(kept) != len(hits):
        logger.info(
            "Episodic: dropped %d note(s) from create turns of other sessions "
            "(create turn of session %s)", len(hits) - len(kept), session_id,
        )
    return kept


def _latest_statement_per_slot(hits: list) -> list:
    """同じ単値の属性スロットを述べるノートは、**最新の利用者の発言** だけ残す。

    版を作る前・Full の抽出前は、言い直した値 (「好きな飲み物は紅茶です」) と
    前の値 (「…コーヒーです」) が両方ノートとして当たり、cosine がほぼ同じなら
    古い方が先に並んでモデルがそれを答える (2026-09-23 実機: 別セッションで
    言い直した後も「コーヒーです」)。ファクト側の「1 スロット 1 値」(注入前の
    畳み込み) と同じ規則をノートに掛ける。

    アシスタントのノートは勝者にしない — 古い値を答えた応答がいちばん新しい
    ノートになり、それが次の想起で勝つと誤りが自己増幅する。スロットに利用者の
    発言があれば、同じスロットのアシスタントのノートは落とす。利用者の発言が
    無いスロットはそのまま (比べる根拠が無い)。

    sleep-time が検証して適用した編集 (``attrs.slot_edit_slot``、「趣味に写真を
    加えて、キャンプは外してください。」) も利用者の最新の言明に数える
    (:func:`~backend.free.memory.notes.note_builder.note_state_slot`)。ただし
    編集の発話は差分しか持たないので、編集が勝ったスロットは **編集の元になった
    最新の言明** も残す (編集だけだと「写真を加えて」から釣りが読めない。編集後の
    全体は SemMem の行が運ぶ)。
    """
    if len(hits) < 2:
        return hits
    from backend.free.memory.notes.note_builder import (
        note_state_slot,
        restated_attribute_slot,
    )
    from backend.utils import parse_utc

    def stamp(hit) -> float:
        try:
            parsed = parse_utc(str(getattr(hit.record, "observed_at", "") or ""))
        except (TypeError, ValueError):
            return 0.0
        return parsed.timestamp() if parsed is not None else 0.0

    def edit_slot(hit) -> str | None:
        attrs = getattr(hit.record, "attrs", None) or {}
        return attrs.get("slot_edit_slot") if isinstance(attrs, Mapping) else None

    slot_of = {hit.id: note_state_slot(hit.text, edit_slot(hit)) for hit in hits}
    latest: dict[str, tuple[float, str]] = {}
    #: スロットごとの最新の **値を述べた** 言明 (編集の発話を除く)
    latest_statement: dict[str, tuple[float, str]] = {}
    for hit in hits:
        slot = slot_of[hit.id]
        if slot is None or hit.record.origin != "user":
            continue
        when = stamp(hit)
        if slot not in latest or when > latest[slot][0]:
            latest[slot] = (when, hit.id)
        if restated_attribute_slot(hit.text) == slot and (
            slot not in latest_statement or when > latest_statement[slot][0]
        ):
            latest_statement[slot] = (when, hit.id)
    if not latest:
        return hits
    winners = {hit_id for _when, hit_id in latest.values()}
    winners |= {hit_id for _when, hit_id in latest_statement.values()}
    # 最新の言明より後の編集は全部残す (「写真を加えて」→「キャンプは外して」の
    # 2 ターンを 1 つに絞ると、片方の差分が読めなくなる)
    for hit in hits:
        slot = slot_of[hit.id]
        if (
            slot is not None
            and hit.record.origin == "user"
            and restated_attribute_slot(hit.text) != slot
            and stamp(hit) > latest_statement.get(slot, (0.0, ""))[0]
        ):
            winners.add(hit.id)
    kept = [
        hit for hit in hits
        if slot_of[hit.id] not in latest or hit.id in winners
    ]
    if len(kept) != len(hits):
        logger.info(
            "Episodic: kept the latest user statement per attribute slot "
            "(%s); dropped %d older or echoed note(s)",
            ", ".join(sorted(latest)), len(hits) - len(kept),
        )
    return kept


def _answers_for_question_only_hits(
    episodic, hits: list, query: str = "", mode: str = "chat",
    *, swapped_sink: set[str] | None = None, session_id: str | None = None,
    user_turns: int | None = None,
) -> list:
    """問いだけの user ノートが当たったら、その **答え** の assistant ノートを差し出す。

    過去セッションのユーザーの問い (「発表の日付とテーマを確認させてください」)
    は本文に事実を持たないので注入側 (``inference._eligible_rag_indices``) が
    捨てる。しかしその問いが最類似で当たるということは、**問いに対して返した
    答えこそ** が今回の問いの証拠。捨てるだけだと、答え (「発表日は 10 月 22 日、
    テーマは『顧客オンボーディングの自動化』です」) が cos 0.78 の隣にあるのに
    「テーマは確認できていません」に落ちた (2026-09-10 ライブ監査 (g) G-05、
    「ここまで話した内容を 5 項目で」→ 家族構成の要約も同型)。

    隣接は取り込み時に刻んだ ``answered_by`` (``episodic.ingest``) で引く
    (走査しない)。答えのノートが無い / 退役済み / 自分が注入した出力なら
    元の問いのまま (後段が従来どおり捨てる)。順位 (cosine / score) は問いの
    ものを継ぐ — 当たったのは問いであり、答えはその付属。

    差し替えは origin 規則 (assistant のノートを落とす) の迂回路なので、次の
    2 つでは差し替えない (2026-09-30 実機 QC01、f_02 §8.3):

    - 問いが利用者の属性を尋ねている (``MemoryInjector._asked_attributes`` が
      空でない)。個人の値の出所は利用者の発話と SemMem だけにする
      (:func:`_latest_statement_per_slot` が assistant を勝者にしないのと同じ)。
      「私の犬の名前は何?」に過去の答え「猫の名前はきなこです」を渡していた。
    - 答えの本文が値を述べていない (問いの側と同じ既存の判定)。問い返しや
      「参考情報には記載がありません」を証拠として手渡さない。
    - 問いのノートが **別のセッション** のもので (``session_id`` が与えられ、ノートに
      出所のセッションが記録されているとき)、今回の問いが過去の会話を指していない
      (:func:`~backend.free.core.intent_vocab.points_to_past_session` が偽、「前に教えて
      もらった」「前回」「以前」が無い。会話の 1 ターン目 (``user_turns`` が
      ``RECALL_IN_SESSION_MIN_USER_TURNS`` 未満) は「さっき」「最初に聞いた」も過去の
      会話を指すと読む — :func:`episodic_session_scope` と同じ数え方)。別の会話の答えはその会話の前提に条件づけられた
      自分の出力で、一般知識の問いならモデルが今の前提で答え直せる。載せると前提ごと
      写される (2026-10-05 ライブ監査 trace d8ee403d199c: 寝つきの会話の「カフェインは
      何時までに控えるべき？」に、就寝 2 時の別の会話の答え「午後 9 時以前」が渡り、
      そのまま答えた。不変則 #15)。利用者の値は利用者の発話と SemMem が運ぶ。
    - (過去の会話を指す問いでも) 問いのノートが **別のセッション** のもので、問いと答えのどちらの本文も今回の問いの
      内容語 (:func:`~backend.free.core.query_anchors.query_anchors` を
      :func:`~backend.free.core.query_anchors.mentions_anchor` で語として照合) を
      1 つも持たない。内容語が無い問いには掛けない。問いの言い回しが似ているだけで、
      答えは別の話題についてのもの (2026-10-05 ライブ監査: 英語学習の会話の「もう少し
      カジュアルな言い方は？」に、別セッションの「もう少し短く、丁寧すぎない表現に
      してください。」の答え = 納期遅延のお詫びメールが渡り、無関係なメールを書いた)。
      差し替えなかった問いは後段が従来どおり捨てる。同じセッションの問いと、過去の会話を
      指す問いに内容語で繋がる答え (「前に教えてもらったテーマは」→「発表日は…テーマは…」)
      は差し替え、別セッションの答えには出所ヘッダに「別の会話」を添える。

    差し替えた行は assistant 由来なので、c_16 §7.6 の出所ヘッダを本文の頭に付け、id を
    ``swapped_sink`` へ足す。corpus が答えている回は ``unified_search`` がそれを外す
    (:func:`_yield_past_answers_to_corpus`、不変則 #15)。
    """
    if not hits or episodic is None:
        return hits
    from backend.free.core.session_mode import normalize_session_mode
    from backend.free.memory.pipeline.injector import MemoryInjector, past_answer_lead

    asked = MemoryInjector._asked_attributes(query, normalize_session_mode(mode))
    if asked:
        if any(
            h.record.origin == "user" and (getattr(h.record, "attrs", None) or {}).get("answered_by")
            for h in hits
        ):
            logger.info(
                "Episodic: the query asks for user attribute(s) (%s); question-only "
                "hits are not replaced by our own answers",
                ", ".join(sorted(asked)),
            )
        return hits
    from backend.free.core.query_anchors import mentions_anchor, query_anchors, word_anchors
    from backend.free.core.text_quality import (
        abstains_on_reference_material,
        carries_no_assertion,
        states_no_user_value,
    )
    from backend.free.memory.episodic.store import EpisodicHit
    from backend.free.rag.evidence.store import is_active

    now_epoch = utc_now_dt().timestamp()
    # 照合する語が無い問い (「私の好きな色は？」) は門を掛けない (判定の根拠が無い)。
    anchors = word_anchors(query_anchors(query))
    query_points_to_past = points_to_past_session(query, user_turns=user_turns)
    out = []
    swapped = 0
    seen_ids = {h.id for h in hits}
    for hit in hits:
        record = hit.record
        answer_id = (getattr(record, "attrs", None) or {}).get("answered_by")
        if (
            record.origin != "user"
            or not answer_id
            or answer_id in seen_ids
            or not (carries_no_assertion(hit.text) or states_no_user_value(hit.text))
        ):
            out.append(hit)
            continue
        answer = episodic.get(str(answer_id))
        if (
            answer is None
            or answer.origin != "assistant"
            or not is_active(answer, now_epoch)
            or not (answer.text or "").strip()
            or _is_injected_output(answer.text)
            or carries_no_assertion(answer.text)
            or states_no_user_value(answer.text)
            or abstains_on_reference_material(answer.text)
        ):
            out.append(hit)
            continue
        # 出所のセッションが記録に無いノートは「別の会話」と決めつけない (門も注記も掛けない)。
        hit_session = _hit_session(hit)
        other_session = session_id is not None and bool(hit_session) and hit_session != session_id
        if other_session and not query_points_to_past:
            logger.info(
                "Episodic: kept question-only hit %s unswapped; its answer %s is from "
                "another session (%s) and the query does not point to a past conversation",
                hit.id, answer.id, hit_session,
            )
            out.append(hit)
            continue
        if (
            other_session
            and anchors
            and not mentions_anchor(f"{hit.text}\n{answer.text}", anchors)
        ):
            logger.info(
                "Episodic: kept question-only hit %s unswapped; it and its answer %s "
                "are from another session (%s) and share no content word with the "
                "query (anchors=%s)",
                hit.id, answer.id, hit_session, ", ".join(sorted(anchors)),
            )
            out.append(hit)
            continue
        question = hit.text.strip()
        lead = past_answer_lead(
            str(getattr(answer, "observed_at", "") or ""), question,
            other_session=other_session,
        )
        out.append(EpisodicHit(answer, hit.cosine, hit.score, f"{lead} {answer.text.strip()}"))
        seen_ids.add(answer.id)
        if swapped_sink is not None:
            swapped_sink.add(answer.id)
        swapped += 1
    if swapped:
        logger.info(
            "Episodic: %d question-only hit(s) replaced by the answer they received",
            swapped,
        )
    return out


def _corpus_keep_floor(
    merged_raw: Sequence[tuple[str, float, str]],
    pseudo_ids: set[str],
    rag_cfg: dict,
    corpus_thresholds: QualityThresholds,
) -> tuple[set[str], float]:
    """Step 6.5 の corpus 本体 (席を含む、疑似クエリ由来を除く) の id と、その棒を返す。

    相対の棒の top1 は同じストアの候補から取る (f_01 §6.6)。
    """
    corpus_ids = {
        cid for cid, _, _ in merged_raw
        if cid not in pseudo_ids and _store_of(cid) == "corpus"
    }
    floor = _resolve_keep_floor(
        rag_cfg, corpus_thresholds,
        top_raw_score=max((s for cid, s, _ in merged_raw if cid in corpus_ids), default=0.0),
    )
    return corpus_ids, floor


def _yield_past_answers_to_corpus(
    answer_ids: set[str],
    merged_entries: list[StoreEntry],
    episodic_extras: list[StoreEntry],
    *,
    pseudo_ids: set[str],
    floor_pseudo: float,
    own_session: str | None,
    rag_cfg: dict,
    corpus_thresholds: QualityThresholds,
    debug_logger=None,
    query: str = "",
) -> tuple[list[StoreEntry], list[StoreEntry], set[str]]:
    """載せられる corpus の候補が棒を越えるなら、差し替えた過去の答えを両方の幅から外す。

    過去の答えは問いの cosine を継ぐので、corpus の正解より上に並んでモデルがそれに
    従う (2026-10-03 実機 run14: 誤答「3年」が cos 0.994 で第 8 条「2 年」より先)。
    自分の出力が埋めてよいのは現在の証拠が棄権したときだけ (不変則 #15)。

    ``merged_entries`` は Step 4.9 の資格判定を通った後の候補。棒は Step 6.5 と同じ —
    本体と席は :func:`_corpus_keep_floor`、疑似クエリ由来は ``floor_pseudo``。資格判定の
    前の top1 で決めると、問いだけのチャンクが top1 の回に答えも corpus も載らない。
    戻り値は ``(merged_entries, episodic_extras, 外した id)``。
    """
    present = answer_ids & {e[0] for e in (*merged_entries, *episodic_extras)}
    if not present:
        return merged_entries, episodic_extras, set()
    raw = gate_view(merged_entries)
    corpus_ids, corpus_floor = _corpus_keep_floor(raw, pseudo_ids, rag_cfg, corpus_thresholds)
    body_top = max((s for cid, s, _ in raw if cid in corpus_ids), default=None)
    pq_top = max((s for cid, s, _ in raw if cid in pseudo_ids), default=None)
    if own_session is not None:
        reason = "conversation_target"
    elif body_top is None and pq_top is None:
        reason = "no_corpus"
    elif (body_top is not None and body_top >= corpus_floor) or (
        pq_top is not None and pq_top >= floor_pseudo
    ):
        reason = "corpus_passes_floor"
    else:
        reason = "below_floor"
    drop = reason == "corpus_passes_floor"
    if drop:
        logger.info(
            "Episodic: dropped %d past answer(s) swapped in for question-only hits; "
            "a corpus candidate passes its floor (body %s >= %.3f / pseudo %s >= %.3f)",
            len(present), body_top, corpus_floor, pq_top, floor_pseudo,
        )
    if debug_logger is not None:
        debug_logger.log_decision(
            decision_point="past_answer_yields_to_corpus",
            chosen="drop" if drop else "keep",
            candidates=["drop", "keep"],
            reason=reason,
            context={
                "query": query[:80],
                ("dropped_ids" if drop else "kept_ids"): sorted(present),
                "corpus_top_cosine": None if body_top is None else round(float(body_top), 4),
                "corpus_floor": round(float(corpus_floor), 4),
                "pq_top_cosine": None if pq_top is None else round(float(pq_top), 4),
                "pseudo_floor": round(float(floor_pseudo), 4),
            },
        )
    if not drop:
        return merged_entries, episodic_extras, set()
    return (
        [e for e in merged_entries if e[0] not in present],
        [e for e in episodic_extras if e[0] not in present],
        present,
    )


def _log_unadopted_yield(
    dropped: set[str],
    final_sources: Sequence[tuple[str, float, str]],
    pseudo_ids: set[str],
    *,
    debug_logger=None,
    query: str = "",
) -> None:
    """答えを外したのに採用に corpus が 1 件も載らなかった回を記録する (戻さない)。

    棒は Step 6.5 と同じなので、ここに来るのは corpus が順位で席を取れなかった回だけ。
    答えを戻すと再順位段 (c_16 §7.2.1) が保つストアの位置の集合が崩れるので、記録に留める。
    """
    if not dropped or any(
        cid in pseudo_ids or _store_of(cid) == "corpus" for cid, _, _ in final_sources
    ):
        return
    final_ids = [cid for cid, _, _ in final_sources]
    logger.warning(
        "Episodic: past answer(s) %s were dropped for corpus evidence, but no corpus "
        "candidate was adopted (final=%s)", sorted(dropped), final_ids,
    )
    if debug_logger is not None:
        debug_logger.log_decision(
            decision_point="past_answer_yield_unadopted",
            chosen="logged",
            candidates=["logged"],
            reason="corpus_not_adopted",
            context={"query": query[:80], "dropped_ids": sorted(dropped), "final_ids": final_ids},
        )


#: 「注入するために組み立てたテキスト」だけに現れるマーカー。これを含む LTM
#: チャンクは、こちらが生成した提示用の文字列が記憶へ回り込んだもの。
#:
#: 取り込み側は ``MDPIngester`` が塞いだが (記憶読み出しツールのエピソードは
#: 昇格させない)、**既に取り込まれたベクトルは残る**。生成側の修正だけでは
#: 既存データが直らないので、読込側でも同じルールを適用する (STM の
#: ``_sanitize_tags`` と同じ遡及修復の形)。
#:
#: 実害 (2026-08-16 再測定): ``[以下は**今回の会話**の記録です]`` を含む
#: mdp_trace が LTM に入っており、**「今回の会話」というラベルごと別セッションへ
#: 持ち越される**。読んだモデルは他人の会話を自分の会話として帰属する。
_INJECTED_OUTPUT_MARKERS: tuple[str, ...] = (
    SEARCH_HISTORY_CURRENT_SESSION_HEADER,
    SEARCH_HISTORY_OTHER_SESSIONS_HEADER,
    SEARCH_HISTORY_NO_RESULTS_PREFIX,
)


def _is_injected_output(text: str) -> bool:
    """こちらが注入用に組み立てた文字列か (純粋関数)。"""
    return any(marker in text for marker in _INJECTED_OUTPUT_MARKERS)


#: 既に LTM へ入っている mdp_trace の ``result=[file: …]`` に続くファイル本文。
#: メタ行の閉じ ``]`` から、次のフィールド区切り (``; actions=`` 等) までを切る。
_MDP_FILE_PAYLOAD_RE = re.compile(
    r"(result=\[file:[^\]]*\])(?:(?!;\s*(?:actions|conversation)=).)*",
    re.DOTALL,
)


def _trim_file_payload(text: str) -> str:
    """エピソード記憶チャンクからファイル本文を落とす (純粋関数)。

    取り込み側は ``MDPIngester._summarize_observation`` が塞いだが、**既に
    取り込まれたベクトルには本文が入ったまま**残る。生成側の修正だけでは既存
    データが直らないので、読込側でも同じルールを適用する (STM の
    ``_sanitize_tags`` / ``_INJECTED_OUTPUT_MARKERS`` と同じ遡及修復の形)。

    実インシデント (2026-08-16 動作検証): 「README.md は存在しますか？」に対し
    ツールは 1 行しか読んでいない (``start_line=1, end_line=1``) のに応答は全文
    ダンプのままだった。プロンプトの ``[参考情報 3]`` に
    ``[mdp_trace] episode=...; result=[file: ... | lines: 121 | chars: 3331]``
    に続けて README 本文がまるごと入った、**旧仕様のままのチャンク**があった。
    ツール側 (PR #436/#439)、few-shot 側 (PR #446)、STM ノート側 (PR #447) を
    塞いでも、この経路が残っている限り同じダンプが再生産される。

    メタ行 (``lines`` / ``chars``) は残すので「そのファイルを読んだ / 何行何文字
    だった」は保てる。
    """
    return _MDP_FILE_PAYLOAD_RE.sub(lambda m: m.group(1), text)


#: エピソード記憶チャンクのフィールド境界。``MDPIngester.to_memory_note`` は
#: ``"; "`` で連結する。``task`` 本文に ``;`` が混じっても壊れないよう、
#: 既知のキーで始まらない断片は直前のフィールドへ戻す (下記 ``_split_mdp_fields``)。
_MDP_FIELD_KEY_RE = re.compile(r"^(episode|outcome|task|result|actions|conversation)=")

#: 根拠枠に出さないフィールド。``episode`` / ``conversation`` は内部 ID で、
#: 後者は別会話の UUID をそのまま露出する。``outcome`` は「ツールが壊れなかったか」
#: であって「役に立ったか」ではない (``agent.deliberative._trace_tool_episode`` は
#: 0 件検索も ``success`` にし、有用性は ``reward`` 側で表す) ため、根拠枠に出すと
#: 読み手には検索が当たったように見える。
_MDP_INTERNAL_FIELDS = frozenset({"episode", "outcome", "conversation"})

#: 残っていれば情報があると見なすフィールド。``actions`` は含めない —
#: 「どのツールが動いたか」だけでは「何のために / 何が出たか」が無く、根拠枠に
#: 置いても読み手には使えない (実例: ``actions=search_history`` だけのチャンク)。
_MDP_INFORMATIVE_FIELDS = frozenset({"task", "result"})

_MDP_MARKER = "[mdp_trace]"


def _split_mdp_fields(body: str) -> list[str]:
    """``"; "`` 連結のフィールド列へ分解する (``task`` 内の ``;`` を壊さない)。"""
    fields: list[str] = []
    for seg in body.split(";"):
        if _MDP_FIELD_KEY_RE.match(seg.strip()) or not fields:
            fields.append(seg.strip())
        else:
            fields[-1] = f"{fields[-1]};{seg}"
    return [f for f in fields if f]


def _sanitize_episode_chunk(text: str) -> str | None:
    """エピソード記憶チャンクから内部識別子を落とす (純粋関数)。

    ``[参考情報 N]`` はユーザーに見える根拠枠なので、``episode=ep_xxx`` /
    ``conversation=<uuid>`` / ``outcome=success`` のような内部テレメトリを
    そのまま出さない。``task`` / ``result`` / ``actions`` は残す
    (「何をしてどうなったか」はエピソード想起の本体)。

    実インシデント (2026-08-16 ライブ監査 ターン30): 解約率の数値目標を尋ねた
    ターンの ``[参考情報 2]`` が
    ``[mdp_trace] episode=ep_4720a1a5; outcome=success;``
    ``result=[file: E:\\tmp\\事業メモ.md | lines: 7 | chars: 451 …];``
    ``actions=read_file; conversation=3d818afc-…``
    だった。ローカル絶対パスと別会話の UUID が根拠枠へ露出していた。

    Returns:
        整形後のテキスト。識別子しか無く情報が残らない場合は ``None``
        (呼び出し側がチャンクごと落とす)。
    """
    if _MDP_MARKER not in text:
        return text
    body = text.split(_MDP_MARKER, 1)[1].strip()
    kept: list[str] = []
    informative = False
    for field in _split_mdp_fields(body):
        m = _MDP_FIELD_KEY_RE.match(field)
        key = m.group(1) if m else ""
        if key in _MDP_INTERNAL_FIELDS:
            continue
        if key in _MDP_INFORMATIVE_FIELDS and field[len(key) + 1:].strip():
            informative = True
        kept.append(field)
    if not informative:
        return None
    return "; ".join(kept)


def _resolve_corpus_thresholds(
    cartridge_mgr, cfg: dict, memory_thresholds: QualityThresholds,
) -> tuple[QualityThresholds, float | None, bool, float | None]:
    """corpus 層に使う棒と、疑似クエリの関連性ゲートを掛けるかを決める (f_01 §6.6)。

    Returns:
        ``(corpus の QualityThresholds, pq_gate, veto, on_topic_bar)``。
        ``veto`` は ``pq_gate`` だけで corpus ごと拒否してよいか (充足率 ≥
        ``gate_min_coverage`` かつ較正の ``pq_veto_allowed``)。``on_topic_bar`` は
        拒否しないときの OR で本体側に使う棒 — 較正の ``on_topic_threshold``
        (``match_top1_p25``、プロファイルの棒で頭を抑える)、問いの少ない較正では
        プロファイルの棒、どちらも
        無ければ ``None`` (呼び出し側が confidence に倒す)。未較正は
        ``(記憶側の棒, None, False, None)``。

    有効条件は 2 段 (2026-09-12 (b) で分離):

    1. **棒** (``relevance`` / ``support`` / ``confidence`` と、疑似クエリ由来の
       floor に使う ``pq_gate``) は ``threshold_mode: auto`` かつ corpus 側の
       較正 (疑似クエリ + カナリア) が済んでいれば **充足率に関わらず** 使う。
       棒の妥当性は標本数 (``compute_calibration`` の最小ノート / 最小クエリ)
       で決まり、corpus 全体に対する割合には依存しない。
    2. **関連性ゲート** (Step 3d の「pq_gate を越える疑似クエリが無ければ corpus
       ごと引かない」という拒否) だけ充足率 ≥ ``rag.pseudo_query.gate_min_coverage``
       を要る。疑似クエリを持たないチャンクへの問いは pq が無いのが当然で、
       充足率が低い間に拒否すると正解を落とす。充足していても、較正の正側と
       null 側が重なる (``pq_overlap``) か問いが ``MIN_PQ_FOR_TOPIC_GATE`` 未満なら
       拒否しない (``pq_veto_allowed``、2026-09-30)。
    3. ``on_topic_bar`` は問いが ``MIN_PQ_FOR_TOPIC_GATE`` 未満の較正では
       埋め込みプロファイルの棒 (``injection_relevance_min_score``) に倒し
       (``on_topic_calibrated``)、それ以上の較正では ``min(較正の p25,
       プロファイルの棒)``。プロファイルに値が無ければ較正値のまま。

    以前は両方を充足率で縛っていたため、大きな corpus を入れ直すと充足 50% まで
    (1070 チャンクで約 3 時間のアイドル) 記憶側の棒 0.64 に倒れ、その間 corpus の
    注入がほぼゼロだった。未較正なら記憶側の棒 (従来動作) と ``None`` / ``False``。
    """
    rag_cfg = cfg.get("rag") or {}
    if str((rag_cfg.get("self_rag") or {}).get("threshold_mode", "auto")) != "auto":
        return memory_thresholds, None, False, None
    if cartridge_mgr is None or not hasattr(cartridge_mgr, "corpus_calibration"):
        return memory_thresholds, None, False, None
    if not _pseudo_query_enabled(cfg):
        return memory_thresholds, None, False, None
    try:
        calibration = cartridge_mgr.corpus_calibration()
        coverage = float(cartridge_mgr.pq_coverage())
    except Exception as e:  # noqa: BLE001 — 較正の読み出しで検索を止めない
        logger.warning("Corpus calibration lookup failed: %s", e)
        return memory_thresholds, None, False, None
    if not calibration:
        return memory_thresholds, None, False, None
    min_coverage = float(
        ((rag_cfg.get("pseudo_query") or {}).get("gate_min_coverage", 0.5)),
    )
    pq_gate = calibration.get("pq_gate")
    thresholds = QualityThresholds.from_config(rag_cfg, calibration=calibration)
    # 重なった較正 / 問いの少ない較正は拒否しない (低充足率と同じ OR に倒す)。
    veto = (
        pq_gate is not None and coverage >= min_coverage
        and bool(calibration.get("pq_veto_allowed", True))
    )
    on_topic = calibration.get("on_topic_threshold")
    profile_bar = _profile_absolute_floor()
    if not calibration.get("on_topic_calibrated", True):
        # 問いが少ない p25 は揺れる — 較正前と同じ埋め込みプロファイルの棒に倒す。
        if profile_bar is not None:
            on_topic = profile_bar
    elif on_topic is not None and profile_bar is not None:
        # p25 は定義上正しい問いの 25% を落とし、合成問いの正側は実際の問いより
        # cos が高く出る。棒を上げる側にだけはプロファイルを越えさせない (2026-10-03)。
        on_topic = min(float(on_topic), profile_bar)
    on_topic_bar = float(on_topic) if on_topic is not None else None
    return thresholds, (float(pq_gate) if pq_gate is not None else None), veto, on_topic_bar


def _pseudo_query_may_lead(cartridge_mgr, rag_cfg: dict) -> bool:
    """疑似クエリ由来を先頭に置いてよいか = 充足率 ≥ ``gate_min_coverage`` (f_01 §6.3)。

    充足率を持たないマネージャ (テストのスタブ等) は従来どおり先頭に置く。
    """
    probe = getattr(cartridge_mgr, "pq_coverage", None)
    if probe is None:
        return True
    try:
        coverage = float(probe())
    except Exception:  # noqa: BLE001 — 観測の失敗で並びを変えない
        return True
    min_coverage = float(
        ((rag_cfg.get("pseudo_query") or {}).get("gate_min_coverage", 0.5)),
    )
    return coverage >= min_coverage


def _pseudo_query_enabled(cfg: dict) -> bool:
    """``rag.pseudo_query.enabled`` (既定 True、f_01 §6.5)。"""
    section = (cfg.get("rag") or {}).get("pseudo_query") or {}
    return bool(section.get("enabled", True))


#: 取りこぼしの種にする転置索引の上位件数 (f_01 §6.4 の misses)。
LEXICAL_MISS_CANDIDATES = 5


async def _search_lexical_seat_layer(
    cartridge_mgr, query_text: str, query_vec: np.ndarray,
    exclude_ids: list[str], seats: int, timeout_ms: int = 0,
) -> list[StoreEntry]:
    """転置索引の席 (f_01 §8.1 の 4.3)。corpus 層の結果に無い語彙上位を引く。"""
    if seats <= 0 or not query_text or cartridge_mgr is None:
        return []
    fn = getattr(cartridge_mgr, "search_detailed_lexical_seat", None)
    if fn is None:
        return []
    loop = asyncio.get_running_loop()
    try:
        coro = run_in_executor_with_context(
            loop, _cartridge_executor,
            lambda: fn(query_vec, query_text, exclude_ids, seats),
        )
        raw = await asyncio.wait_for(coro, timeout=timeout_ms / 1000.0) if timeout_ms > 0 else await coro
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001 — 席の探索で応答を止めない
        logger.warning("Lexical seat search failed: %s", e)
        return []
    return [(str(e[0]), float(e[1]), float(e[2]), e[3]) for e in raw if len(e) >= 4]


async def _search_pseudo_query_layer(
    cartridge_mgr, query_vec: np.ndarray, top_k: int, timeout_ms: int = 0,
) -> tuple[list[StoreEntry], set[str]]:
    """corpus の疑似クエリ索引 (f_01 §6.3)。``(id, cosine, score, text)`` の列と、
    取りこぼした問いの言い換え (``from_hint``) が最良だった id の集合を返す。

    並びは問い↔問いの cosine 順。``cosine`` は問い↔問いの値 (較正済みの棒は
    クエリ↔発話の分布なのでこちらがスケールに合う)、``score`` は対象チャンク
    本体の順位式の値。失敗 / タイムアウトは空 — 本体の corpus 層と同じく、
    1 層の失敗で他層を巻き込まない。
    """
    if cartridge_mgr is None or not hasattr(cartridge_mgr, "search_detailed_pq"):
        return [], set()
    loop = asyncio.get_running_loop()
    try:
        coro = run_in_executor_with_context(
            loop, _cartridge_executor,
            lambda: cartridge_mgr.search_detailed_pq(query_vec, top_k),
        )
        if timeout_ms > 0:
            raw = await asyncio.wait_for(coro, timeout=timeout_ms / 1000.0)
        else:
            raw = await coro
        entries: list[StoreEntry] = []
        hinted: set[str] = set()
        for item in raw:
            cid, cosine, score, text = item[0], item[1], item[2], item[3]
            entries.append((cid, float(cosine), float(score), text))
            if len(item) > 4 and item[4]:
                hinted.add(cid)
        logger.debug("Step 3c pseudo-query: %d results", len(entries))
        return entries, hinted
    except asyncio.TimeoutError:
        logger.warning("Pseudo-query search timed out after %d ms", timeout_ms)
        return [], set()
    except asyncio.CancelledError:
        raise
    except (RAGError, RuntimeError, ValueError, TypeError, OSError) as e:
        logger.warning("Pseudo-query search failed: %s", e)
        return [], set()


def _interleave_pseudo(
    pseudo_entries: list[StoreEntry], merged: list[StoreEntry],
) -> list[StoreEntry]:
    """疑似クエリ由来を先頭に、交互 1 件ずつ id で畳む (f_01 §6.3)。

    順位式 1 本 (c_16 §7.2) の外側の **位置融合**。問い↔問いの cosine は
    問い↔文書より高く、score へ流すと top-k を占有する (実測 0.605 vs 0.775)
    ので、スケールの違う 2 列は位置で合わせる。
    """
    if not pseudo_entries:
        return merged
    out: list[StoreEntry] = []
    seen: set[str] = set()
    for rank in range(max(len(pseudo_entries), len(merged))):
        for column in (pseudo_entries, merged):
            if rank < len(column) and column[rank][0] not in seen:
                seen.add(column[rank][0])
                out.append(column[rank])
    return out


async def _search_corpus_layer(
    cartridge_mgr, query_vec: np.ndarray, top_k: int,
    timeout_ms: int = 0,
    rescore_candidates: int = 0,
    query_text: str = "",
    pq_seeds: list[str] | None = None,
) -> list[StoreEntry]:
    """corpus (旧カートリッジ) 検索。``(id, cosine, score, text)`` を返す。

    ``pq_seeds`` は転置索引の上位候補 (疑似クエリの lazy 生成対象の種) の受け皿。
    積むのは注入を確定した消費側 (:class:`SearchUsage`)。

    ``query_text`` は転置索引の候補生成 (c_16 §6.3) に渡す。空だとベクトル
    候補だけになり、固有語の問いを取りこぼす (f_01 §8.1、2026-09-12 (b))。

    ``score`` は ``CorpusStore.search`` が **ストアの中で** 計算した順位式
    (``cos × freshness × confidence × store_prior``、c_16 §7.2) の値。
    ``store_prior`` は ``memory.evidence.ranking.store_prior.corpus`` と
    ``corpus/manifest.json`` の ``store_prior_overrides`` から解決される。
    ゲート用の ``cosine`` を分けて持つのは、品質判定 / floor / content gate が
    cosine スケール前提で閾値を決めているため — 合成スコアを閾値に流すと
    **``store_prior`` が閾値を偽装する** (c_16 §7.1)。

    ``search_detailed`` (cosine と score を分けて返す) があればそれを使い、
    無ければ ``search`` の戻り (score のみ) を両方に使う。

    ``timeout_ms`` が 1 以上の場合、検索全体にタイムアウトを適用する。
    タイムアウト時は空を返し、チャット応答を止めないようにする。
    マネージャ未設定 / 失敗時 (:class:`~backend.exceptions.RAGError` を含む —
    次元不一致のパッケージが 1 つあるだけで ``asyncio.gather`` ごと落ち、
    そのターンの記憶が全部消えていた) も空。
    """
    if cartridge_mgr is None or not hasattr(cartridge_mgr, "search"):
        return []
    loop = asyncio.get_running_loop()
    detailed = getattr(cartridge_mgr, "search_detailed", None)
    extra = {"rescore_candidates": rescore_candidates} if rescore_candidates > 0 else {}
    if query_text:
        extra["query_text"] = query_text
        if pq_seeds is not None:
            extra["pq_seeds"] = pq_seeds
    try:
        if detailed is not None:
            coro = run_in_executor_with_context(
                loop, _cartridge_executor,
                lambda: detailed(query_vec, top_k, **extra),
            )
        else:
            coro = run_in_executor_with_context(
                loop, _cartridge_executor,
                lambda: cartridge_mgr.search(query_vec, top_k, **extra),
            )
        if timeout_ms > 0:
            raw = await asyncio.wait_for(coro, timeout=timeout_ms / 1000.0)
        else:
            raw = await coro
        entries: list[StoreEntry] = []
        for entry in raw:
            if len(entry) >= 4:
                cid, cosine, score, text = entry[0], entry[1], entry[2], entry[3]
                entries.append((cid, float(cosine), float(score), text))
            else:
                cid, score, text = entry[0], entry[1], entry[2]
                entries.append((cid, float(score), float(score), text))
        logger.debug("Step 3b corpus: %d results", len(entries))
        return entries
    except asyncio.TimeoutError:
        logger.warning(
            "Corpus search timed out after %d ms (L3); "
            "returning empty results to keep chat responsive",
            timeout_ms,
        )
        return []
    except asyncio.CancelledError:
        raise
    except (RAGError, RuntimeError, ValueError, TypeError, OSError) as e:
        logger.warning("Corpus search failed: %s", e)
        return []


async def _try_quality_expansion(
    quality: str,
    merged: list[StoreEntry],
    quality_judge: RetrievalQualityJudge,
    query: str,
    query_vec: np.ndarray,
    working_mem,
    episodic,
    top_k: int,
    noise_sigma: float,
) -> tuple[list[StoreEntry], str]:
    """品質不足 (low) 時にクエリ拡張で再検索を試みる。`(merged, quality)` を返す。

    品質判定は **素の cosine** に対して行う (c_16 §7.1) ので、判定へ渡すのは
    :func:`gate_view` の射影。
    """
    if quality != "low" or not merged:
        return merged, quality
    logger.debug("Quality low — attempting query expansion")
    expanded_results = await _expand_and_research(
        query, query_vec, working_mem, episodic, top_k,
        noise_sigma=noise_sigma,
    )
    if not expanded_results:
        return merged, quality
    merged = _merge_results(merged, expanded_results)
    quality = quality_judge.judge(gate_view(merged))
    logger.debug("After expansion: %d results, quality=%s", len(merged), quality)
    return merged, quality


def _log_memory_search_state(
    debug_logger,
    context_count: int,
    episodic_results: list,
    semmem_stats: dict | None = None,
) -> None:
    """DebugLogger 設定時のみメモリ検索状態を記録する。

    ``semmem_stats`` が与えられた場合は memory.jsonl の ``semmem`` フィールド
    として埋め込む
    """
    if debug_logger is None:
        return
    debug_logger.log_memory_state(
        session_id="unified_search",
        memory_dump={
            "working_turns": context_count,
            "episodic_notes": len(episodic_results),
        },
        semmem_stats=semmem_stats,
    )


#: 較正が効いていないときに使う「top 相対」の棒の比率。
#:
#: 静的な絶対閾値は埋め込みモデルを替えると**到達不能になって黙って全部落とす**。
#: このプロジェクトは同じ壊れ方を 2 度している:
#:
#: - ``relevance_threshold: 0.65`` — Qwen3-Embedding 前提。LFM2.5 で記憶採用 0 件
#: - ``low_quality_keep_floor: 0.40`` — 旧 STM combined スケール前提。2026-08-16
#:   の実測で観測最大 0.381 を上回り、45 候補の **通過 0 件**
#:
#: 較正 (``threshold_mode: auto``) が効いていれば実データから棒が決まるので
#: この問題は起きない。効かない条件 (ノート数不足 / ``manual``) だけが脆い。
#: そこで「静的値」と「その検索の top のβ倍」の**低い方**を採る。静的値より
#: 厳しくはならず、静的値がそのスケールで到達不能なときだけ緩む。
_RELATIVE_FLOOR_RATIO = 0.6

#: 相対フロアの下限 (絶対値)。
#:
#: 相対フロアだけだと「**どれも無関係な検索でも top の 60% は残る**」ことになり、
#: 「関連が無い」を「一番マシなもの」へすり替えてしまう。相対は静的値を緩める
#: 方向にだけ働かせ、ノイズ帯より下へは落とさない。
#: 実測 (2026-08-16、LFM2.5-Embedding): 無関係ペアの類似度は全ペア中央値 0.105 /
#: p90 0.238 で、関連ペアは 0.467 だった。
#:
#: 旧値 0.15 は「ノイズの **中央値** より上」を根拠にしていたが、中央値より上と
#: いうことは **ノイズの約半分が通る** ということで、「ノイズ帯より下へは落とさ
#: ない」という上のポリシー宣言と食い違っていた。相対の棒は
#: ``max(absolute, top_raw_score * ratio)`` なので top1 は定義上必ず越える —
#: つまり **この絶対値だけが「このターンには関連する記憶が無い」を表現できる
#: 唯一の手段** で、そこがノイズ帯の中にあると「関連なし」を表現できない。
#:
#: 実測 (2026-08-23 ライブ監査セット 1): 較正が未成立の状態 (新規のデータ根) で
#: 実効フロアは 0.15〜0.22 まで下がり、「√2 を小数第 6 位まで教えてください」
#: に大阪出張の予定が cos 0.1735 で、「税抜き 12,800 円の消費税」に同じ予定が
#: 0.1863 / 0.1656 で ``[参考情報]`` として注入された。同セッションで本当に
#: 関連するチャンク (猫の名前の問いに対する猫の記録) は 0.344 / 0.3377 で、
#: ノイズ帯とは分離していた。
#:
#: そこで宣言どおりノイズ分布の p90 に置く。関連ペア (0.467、実測の一致は
#: 0.34 以上) は十分上に残る。
#:
#: ⚠ この値は **旧 STM combined スケール** のもの。実際に使う棒は
#: :func:`_uncalibrated_absolute_min` が埋め込みプロファイルから引き、
#: プロファイルに値が無いときだけこの定数へ倒れる。
_RELATIVE_FLOOR_ABSOLUTE_MIN = 0.24


def _profile_absolute_floor() -> float | None:
    """アクティブな埋め込みプロファイルの絶対の棒 (値が無ければ ``None``)。"""
    from backend.free.rag.memory_threshold_calibration import (
        profile_embedding_threshold,
    )

    value = profile_embedding_threshold("rag", "injection_relevance_min_score")
    return None if value is None else float(value)


def _uncalibrated_absolute_min() -> float:
    """較正が効く前の絶対の棒 (アクティブな埋め込みプロファイル由来)。

    ``MemoryInjector`` が既に同じことをしている
    (:meth:`~backend.free.memory.pipeline.injector.MemoryInjector._resolve_relevance_thresholds`)
    — 較正が ``MIN_NOTES`` 溜まるまで走らない窓で、静的値が別モデル前提だと
    **ノイズが全部通る**。あちらはプロファイルの
    ``rag.injection_relevance_min_score`` (bert = 0.54) へ追随させて直したが、
    **エピソード検索側のこの棒だけが旧スケールの定数 (0.24) のまま残っていた**
    — 同じ判定の読み手が 2 つあり、片方だけ直っていた形。

    実測 (2026-09-16 ライブ監査): 全リセット直後の 13 ターンは実効フロアが
    0.25〜0.32 で、bge-m3 のノイズ床 (無関係ペア 0.29〜0.47) の下にあり、
    ``[参考情報]`` の 4〜5 枠が毎ターン無関係なノートで埋まった。

    棒そのものは呼出側で ``min(configured, …)`` に掛けられるので、
    **ユーザーが config で明示した値を超えて厳しくはならない**。
    """
    from backend.free.rag.memory_threshold_calibration import (
        profile_embedding_threshold,
    )

    from_profile = profile_embedding_threshold(
        "rag", "injection_relevance_min_score",
    )
    if from_profile is None:
        return _RELATIVE_FLOOR_ABSOLUTE_MIN
    return float(from_profile)

#: 「そのターンの最良証拠」に対する相対の棒 (既定値)。
#:
#: :data:`_RELATIVE_FLOOR_RATIO` とは **符号が逆** なので混同しないこと。
#: あちらは静的閾値がそのスケールで到達不能なときに floor を **緩める**
#: 到達性の保険。こちらは top1 と比べて明らかに弱いチャンクを **落とす** 棒。
#:
#: なぜ絶対値の棒だけでは足りないか: 較正が効いているとき floor は
#: ``relevance`` = **background_p95** (ノイズ分布の 95 パーセンタイル) になる。
#: 構造上ノイズの 5% が通る棒で、「関連しているか」ではなく「ノイズより上か」
#: しか見ていない。実測 (2026-08-19、chat 56 ターン / 採用 183 チャンク、
#: 較正値 background_p95=0.302 / match_top1_p25=0.475): 採用スコアは p25 0.352 /
#: 中央値 0.401 で、**75% が「真の一致の下位 25%」より下**。実サンプルでは
#: 他人のペルソナを含む挨拶文が 0.32〜0.40 で 5 件通っていた。
#:
#: なぜ絶対値を上げるのではなく相対にするか (同じ 42 ターンでの実測):
#:   絶対 0.40 -> 採用 51%、**15/42 ターンが空になる**
#:   相対 0.75 -> 採用 73%、**空になるターンは 0**
#: top1 は定義上必ず棒を越えるので、相対は検索結果を空にしない。
#: 「ノイズより上か」(絶対) と「このクエリで取れた最良証拠と比べられるか」
#: (相対) は別の問いで、候補集合が「このクエリの検索結果」である RAG 側では
#: 後者が意味を持つ (注入側 MemoryInjector が相対を採らない理由は
#: ``_resolve_relevance_thresholds`` の docstring 参照 — あちらの候補は
#: ストア全件なので、無関係な集合の top に対する相対は意味を持たない)。
#: 0.0 で無効化。
_RELATIVE_KEEP_RATIO = 0.75

#: **弱い結果集合** (top1 が較正済み ``confidence`` = 真の一致の中央値に届かない)
#: に対して、絶対の棒を ``relevance`` から ``confidence`` へこの割合だけ寄せる。
#: 絶対の棒は背景 p95 なので、ストアが育つと定義上ノイズの 5% が毎ターン
#: 通り、**関連する内容が 1 つも無いクエリ** でも「最良のノイズ」が
#: ``[参考情報]`` に載る (2026-09-05 ライブ監査 F-17: API キーの保管場所の
#: 質問に勤怠管理の日付設計とビーフシチューの煮込み時間が参考情報として
#: 入った)。強い一致があるターン (top1 ≥ confidence) には掛けない —
#: そこでは相対の棒 (0.75 × top1) が弱い兄弟を既に落としている。
#: 0.0 で無効化。
_WEAK_SET_MARGIN = 0.35


def _resolve_relative_keep_ratio(rag_cfg: dict) -> float:
    """``relative_keep_ratio`` を [0, 1] にクランプして返す。"""
    self_rag = rag_cfg.get("self_rag") or {}
    try:
        ratio = float(self_rag.get("relative_keep_ratio", _RELATIVE_KEEP_RATIO))
    except (TypeError, ValueError):
        ratio = _RELATIVE_KEEP_RATIO
    return max(0.0, min(1.0, ratio))


def _pseudo_floor(
    rag_cfg: dict, corpus_thresholds: QualityThresholds, pq_gate: float | None,
) -> float:
    """疑似クエリ由来の行の棒 (f_01 §6.3)。

    較正済みなら ``pq_gate``、無ければ :func:`_resolve_keep_floor` の絶対の棒
    (相対の棒・弱い集合の上乗せ無し)。フロアを切った構成 (``low_quality_keep_floor``
    が 0) では他の棒と同じく 0.0。
    """
    if float((rag_cfg.get("self_rag") or {}).get("low_quality_keep_floor", 0.0)) <= 0.0:
        return 0.0
    if pq_gate is not None:
        return pq_gate
    return _resolve_keep_floor(rag_cfg, corpus_thresholds, top_raw_score=0.0)


def _resolve_keep_floor(
    rag_cfg: dict,
    thresholds: QualityThresholds,
    top_raw_score: float = 0.0,
) -> float:
    """``low_quality_keep_floor`` を ``threshold_mode`` に従って解決する。

    棒は 2 段で決まる。

    1. **絶対の棒** — ``auto`` かつ較正が効いていれば較正済み ``relevance``、
       ``manual`` / 較正なしでは config の静的値。静的値は埋め込みスケールが
       変わると到達不能になるので ``top_raw_score`` との相対で緩める
       (:data:`_RELATIVE_FLOOR_RATIO`、**緩める方向**)。緩和の定数は旧スケール
       由来で、埋め込みを替えると逆に全通しになる — 未較正の窓を短くするのが
       対処で、窓の中の挙動は変えない (2026-09-14 監査 F-04、下の枝を参照)。
    2. **相対の棒** — そのターンの ``top_raw_score`` の
       :data:`_RELATIVE_KEEP_RATIO` 倍 (**絞る方向**)。絶対の棒は
       「ノイズより上か」しか見ておらず、較正時は定義上ノイズの 5% が通る。

    最終的な floor は 1 と 2 の **高い方**。top1 は定義上 2 を越えるため、
    相対の棒で結果集合が空になることはない。config が ``0.0`` (= 従来どおり
    クエリ単位の全件破棄) を明示している場合は較正より優先して尊重する。

    3. **弱い結果集合の上乗せ** — 較正が効いていて top1 が ``confidence``
       に届かないときだけ、絶対の棒を ``relevance`` と ``confidence`` の間
       (:data:`_WEAK_SET_MARGIN`) へ上げる。これは top1 も越えられない
       ことがあり、その場合は結果集合が空になる (= 何も載せない方が正しい)。
    """
    self_rag = rag_cfg.get("self_rag") or {}
    configured = float(self_rag.get("low_quality_keep_floor", 0.0))
    if configured <= 0.0:
        return 0.0
    from backend.free.rag.memory_threshold_calibration import get_active_calibration

    calibrated = (
        str(self_rag.get("threshold_mode", "auto")) == "auto"
        and get_active_calibration() is not None
    )
    profile_floor = (
        _profile_absolute_floor()
        if str(self_rag.get("threshold_mode", "auto")) == "auto" and not calibrated
        else None
    )
    if calibrated:
        absolute = float(thresholds.relevance)
        confidence = float(thresholds.confidence)
        if (
            _WEAK_SET_MARGIN > 0.0
            and confidence > absolute
            and 0.0 < top_raw_score < confidence
        ):
            absolute = absolute + (confidence - absolute) * _WEAK_SET_MARGIN
    elif profile_floor is not None:
        # ``auto`` で較正が未確定の窓は **埋め込みプロファイルの棒** をそのまま
        # 使う (2026-09-17 監査)。以前は下の枝で ``min(configured, プロファイル)``
        # に挟んでいたが、``configured`` はスキーマ既定の 0.40 (旧スケール) が
        # ほぼ全環境に入っているため **プロファイルが一度も効いていなかった** —
        # 全リセット後 20 ターンの較正未確定窓で、算術の追い質問に cosine
        # 0.42〜0.47 の無関係なノート (訂正前の値を含む) が 5 件ずつ載り、
        # 約 1,000 トークンの再 prefill で TTFT 40〜54 秒になった。プロファイル値は
        # その埋め込みモデル用に置いた値なので「静的値が別スケールで到達不能」
        # (2026-08-16) には当たらない。``manual`` は従来どおり config を尊重する。
        absolute = profile_floor
    elif top_raw_score > 0.0:
        # **この枝は「較正が未確定」= 棒がスケール非依存でない状態。** 緩和の
        # 定数 (:data:`_RELATIVE_FLOOR_ABSOLUTE_MIN` = 0.24) は旧 STM combined
        # スケールのノイズ p90 で、bge-m3 (ノイズ p50 0.365 / p95 0.584) では
        # ノイズ帯の下にあり実質全通しになる。実測 (2026-09-14 監査 F-04/F-06):
        # 較正確定前の 35 ターンで cosine 0.386 の無関係チャンクが 5 件
        # ``[参考情報]`` に載り、その窓で「evoref の 4 つの pillar は？」が
        # ``Evoref Core / UI / State / Utils`` と **完全に捏造** された
        # (較正後の同型の問いは「記載はありません」と正答している)。
        #
        # ここを厳しくすると「静的値が到達不能で注入ゼロ」という逆の壊れ方
        # (2026-08-16 で実測) に戻るので、**棒ではなく較正の遅れを直す** —
        # ``SleepTimeWorker`` の Light にも較正リトライを置き、確定を Full 待ち
        # にしない (``_run_light_locked``)。この枝が使われる窓を数ターンに
        # 縮めるのが正しい対処で、窓の中の挙動は従来どおり「緩めて救う」。
        #
        # ただし **緩める下限は現行の埋め込みスケールで置く**
        # (:func:`_uncalibrated_absolute_min`、2026-09-16 監査)。定数 0.24 の
        # ままだと bge-m3 のノイズ床の下なので「緩めて救う」ではなく全通しに
        # なる。``min(configured, …)`` で挟むので config の明示値より厳しくは
        # ならず、到達不能側へは倒れない。
        relaxed = min(configured, top_raw_score * _RELATIVE_FLOOR_RATIO)
        absolute = max(relaxed, min(configured, _uncalibrated_absolute_min()))
    else:
        absolute = configured

    # 絶対の棒を通ったうえで、そのターンの最良証拠と比べて弱いものを落とす。
    # top1 は定義上必ず越えるので結果集合が空になることはない。
    ratio = _resolve_relative_keep_ratio(rag_cfg)
    if ratio > 0.0 and top_raw_score > 0.0:
        return max(absolute, top_raw_score * ratio)
    return absolute


def _store_cosine_gate(
    rag_cfg: dict, thresholds: QualityThresholds,
) -> float:
    """各ストアの ``search(threshold=…)`` へ渡す **素の cosine** の棒。

    c_16 §7.1 は「ゲートは素の cosine のみ、ストア別閾値はモデルプロファイル
    同期」と定める。同期の実体は
    :mod:`backend.free.rag.memory_threshold_calibration` (実ストアの
    クエリ↔レコード分布から ``relevance`` を導く) で、これが効いているときだけ
    棒を渡す。

    較正が無い構成で config の静的値 (既定 0.65) を渡してはいけない。静的な
    絶対閾値は埋め込みモデルを替えると到達不能になり **黙って全件落とす** —
    本リポジトリは既に 2 度同じ壊れ方をしている
    (:data:`_RELATIVE_FLOOR_RATIO` の説明を参照)。較正が無いターンは 0.0 を
    返し、到達性の保険を持つ :func:`_resolve_keep_floor` に一本化する。
    """
    self_rag = rag_cfg.get("self_rag") or {}
    if str(self_rag.get("threshold_mode", "auto")) != "auto":
        return 0.0
    from backend.free.rag.memory_threshold_calibration import get_active_calibration

    if get_active_calibration() is None:
        return 0.0
    return max(0.0, float(thresholds.relevance))


async def unified_search(
    query: str,
    query_vec: np.ndarray,
    working_mem,
    episodic,
    cartridge_mgr=None,
    config: dict | None = None,
    aux_client=None,
    debug_logger=None,
    mode: str = "chat",
    policy: PolicyInterpreter | None = None,
    timer: "StageTimer | None" = None,
    semmem_stats: dict | None = None,
    *,
    session_id: str = "default",
    judge_tracker: "JudgeUsageTracker | None" = None,
    corpus_mode: str = "auto",
    correction_trail: dict[str, str] | None = None,
    correction_successors: Mapping[str, Sequence[str]] | None = None,
    skip_gate=None,
    reranker: "Reranker | None" = None,
    on_corpus_evidence: Callable[[int], None] | None = None,
) -> SearchResult:
    """統合検索パイプライン: Self-RAG + エピソード記憶 + corpus

    **SemMem (semantic) はここでは検索しない。** 融合するのはエピソード記憶と
    corpus の 2 ストアだけで、``semmem_stats`` はログ用の受け渡しにすぎない。
    semantic がプロンプトへ載る経路は ``chat_service.build_semmem_injection`` →
    :class:`~backend.free.memory.pipeline.injector.MemoryInjector` の **完全に
    別系統** (全件 + 関連度ゲート + Tier パッキング) で、top_k 融合を通らない。
    2 系統に分かれているのは、semantic が「属性スロットの現在値」を扱うのに対し
    RAG は「チャンクの関連度」を扱うからで、注入枠 (``[関連する記憶]`` /
    ``[参考情報]``) そのものが別だからでもある (2026-09-01 監査 F12)。
    **順位式は 3 ストアで共通の 1 本** (c_16 §7.2) で、semantic 側も
    ``SemanticStore.search()`` の同じスコアで並べる。

    ストアをまたぐ融合 (:func:`_merge_results`) は RRF でも層内正規化でもない
    — 各ストアの ``search()`` が返す ``score`` (= ``cos × freshness ×
    confidence × store_prior``) をそのまま降順に並べ、``claim_key`` が同じ
    言明を 1 件へ畳む。``rag.score_normalization`` / ``rag.rrf_k`` /
    カートリッジ ``priority`` は廃止した (c_16 §7.2 / §8)。

    ゲートは **素の cosine のみ** (c_16 §7.1)。較正 (``threshold_mode: auto`` +
    ``memory_threshold_calibration``) が効いているときは各ストアの
    ``search(threshold=…)`` に較正済み ``relevance`` を渡し、効いていない構成
    では 0.0 のまま後段の :func:`_resolve_keep_floor` に任せる
    (:func:`_store_cosine_gate`)。

    判定はルールベース + ベクトル演算で完結する (LLM 呼び出しゼロ)。
    エピソード記憶 / corpus 検索を asyncio.gather で並列実行する。
    ``rag.fetch_multiplier`` が 2 以上の場合、取得件数を `top_k * N` に拡張する。

    Args:
        session_id: content gate の発火カウンタキー。
            ``run_search_pipeline`` が ``WorkingMemory.session_id`` か
            フロントエンド指定 session_id を渡す。
        judge_tracker: content gate (create モード) のセッション単位
            カウンタ。``None`` なら上限を評価しない (テスト経路互換)。
        correction_trail: SemMem の世代から引いた
            ``{被訂正ノート id: 現在値の言明}``。``[参考情報]`` に載る訂正前の
            参照へ ``（訂正済み）`` を付け、現在値を随伴させるために使う。
            セッションを跨いだ訂正はこちらでしか解けない
            (:func:`attach_superseding_corrections`、2026-09-14 監査 F-01)。
        correction_successors: SemMem の世代から引いた
            ``{被訂正ノート id: 訂正後の根拠ノート id 列}``。再順位段が episodic を
            並べ替えた回に、採用した訂正の組を訂正後が上になるよう入れ替えるために使う
            (:func:`enforce_correction_order`、f_02 §5.3)。
        reranker: 再順位段 (c_16 §7.2.1)。``rag.rerank.mode: on`` かつ自己テストを
            通ったときだけ渡る (``GenPillar.reranker``)。``None`` なら従来どおり。
        on_corpus_evidence: 採用に載る corpus の件数を、再順位段 (Step 6.8) の **前**
            に知らせる (:func:`early_corpus_evidence`)。件数が再順位の結果に依存する
            回と、Step 6.8 に届かない回 (necessity skip / 採用ゼロ) は呼ばない。
    """
    cfg = config or {}
    rag_cfg = cfg.get("rag", {})
    top_k, stm_top_k, noise_sigma = _resolve_search_params(policy, rag_cfg, mode)
    multiplier = _resolve_fetch_multiplier(cfg)
    fetch_k = top_k * multiplier
    # 再順位段が動くときだけ、候補プール用に corpus と episodic を候補数まで広く引く。
    # 席の除外・関連性ゲート・floor の top1・スロットごとの最新は元の fetch_k 幅の結果で
    # 決め、広げて増えた行はプールの候補にだけ使う (c_16 §7.2.1 の第 2 / 第 4 段階)。
    # 遮断器の冷却中は再順位段が無いのと同じ (広く引いても並べ替えずに捨てるだけ)。
    if reranker is not None and not rerank_active(reranker):
        logger.debug("unified_search: reranker skipped (circuit open)")
        reranker = None
    corpus_fetch_k = max(fetch_k, _rerank_candidates(reranker))
    episodic_fetch_k = corpus_fetch_k
    rescore_candidates = _resolve_rescore_candidates(rag_cfg)
    logger.debug(
        "unified_search: query=%r, top_k=%d, fetch_k=%d (mult=%d, corpus=%d)",
        query[:80], top_k, fetch_k, multiplier, corpus_fetch_k,
    )

    # Step 1: Self-RAG 検索必要性判定 (純ルール、uncertain は retrieve に倒す)
    necessity_judge = RetrievalNecessityJudge()
    full_context = working_mem.get_context()
    context_count = len(full_context)
    # 「答えは今の窓の中にある」という前提で skip するルール (自明質問の
    # セッション自己参照枝 / 十分コンテキスト) は、WorkingMemory が 1 件でも
    # 押し出した時点で前提が崩れる。押し出し後も ``context_count`` は上限付近
    # で張り付くため、そのままだとセッションの残り全部で記憶検索が skip され
    # 続ける (2026-08-23 ライブ監査: 35/94 ターン)。
    window_complete = int(getattr(working_mem, "session_evicted_turns", 0) or 0) == 0
    # Step 1 全体を計測する。ここが未計測だったため「遅い検索の 89.7% が
    # 内訳不明」という状態が続き、実際の支配要因が見えていなかった
    # (2026-08-01 プロファイリング)。
    if timer is not None:
        timer.start("necessity_ms")
    necessity = necessity_judge.judge_rule_only(
        query, context_count=context_count, window_complete=window_complete,
    )
    # 規則が skip と言ったターンだけ事例の近傍に確認を取る。反対されたら
    # ``uncertain`` へ降ろす — 検索を強制するのではなく、埋め込みリコールへ
    # 回す安全側の値 (backend.free.rag.retrieval_skip_gate の説明を参照)。
    # ``query_vec`` を渡すので埋め込みの往復は起きない。
    if skip_gate is not None and necessity == "skip":
        try:
            necessity = await skip_gate.confirm(
                query, necessity=necessity, query_vec=query_vec,
            )
        except Exception as e:  # pragma: no cover - 縮退で吸収する
            logger.info("Retrieval skip gate failed: %s", e)
    # 旧 ``judge()`` の 2 値へ正規化する (uncertain は安全側の retrieve)。
    if necessity == "uncertain":
        necessity = "retrieve"
    if timer is not None:
        timer.stop("necessity_ms")
    logger.debug(
        "Step 1 necessity: %s (context_count=%d, window_complete=%s)",
        necessity, context_count, window_complete,
    )
    # `fetch` は外部 fetch_url 委譲シグナル — RAG パイプラインは `skip` と同等に
    # 即終了し、ToolCallJudge / fetch_url ツールに委ねる。
    if necessity in ("skip", "fetch"):
        logger.info(
            "Search skipped (necessity=%s) for query: %s", necessity, query[:50],
        )
        return SearchResult(skipped=True, from_memory=True)

    # Step 2-3: エピソード記憶 / corpus を asyncio.gather で並列実行
    # fetch_multiplier >= 2 のときは fetch_k 件を取得し、後段で top_k に絞る
    cart_timeout_ms = int(rag_cfg.get("cartridge_search_timeout_ms", 3000))
    # 取得そのものの所要。necessity ゲートの費用対効果を測るために分けて計る
    # (ゲートが取得より高くつく状態を検出できるようにする)。
    if timer is not None:
        timer.start("retrieval_ms")
    # 「前にも同じことを聞いたか」は層検索の結果ではなく保存済みノートの文字列
    # 一致で決まるので、gather の前に同期で確定させる (層は並列に走るため、
    # STM の結果を待ってから LTM の扱いを決めることはできない)。
    drop_past_answers = query_repeats_a_stored_turn(episodic, query)
    if drop_past_answers:
        logger.info(
            "This query repeats an earlier turn; past answers will be kept out "
            "of the reference block (query=%r)", query[:50],
        )
    # ゲートは **素の cosine** を全ストア共通の較正済みの棒で掛ける (c_16 §7.1)。
    thresholds = QualityThresholds.from_config(rag_cfg)
    store_gate = _store_cosine_gate(rag_cfg, thresholds)
    pseudo_enabled = _pseudo_query_enabled(cfg)
    corpus_thresholds, pq_gate, pq_veto, on_topic_bar = _resolve_corpus_thresholds(
        cartridge_mgr, cfg, thresholds,
    )
    if on_topic_bar is None:
        on_topic_bar = corpus_thresholds.confidence
    user_turns = session_user_turns(full_context)
    own_session = episodic_session_scope(query, session_id, user_turns=user_turns)
    # 日付演算の問い (営業日 / 日目 / 週間後…) はツールが答える。文書側に根拠は
    # 無く、設計書に同じ例文があると cosine では区別できない (2026-09-14: 09-12
    # の corpus 誤射 8/46 のうち 6 件がこれで、1 ターン 1088 tok の prefill)。
    # corpus 層と疑似クエリ層を引かず、episodic (直前の回答の日付) は残す。
    # 参加モード (f_01 §8.1): off = このターンは文書を引かない、on = 問い側の
    # 抑止を掛けない、auto = 問いの性質で決める。ロード状態はグローバルなので
    # (`/load` `/unload`)、雑談中に大きなコーパスへ毎ターン払わない手段が
    # これしかない。
    if corpus_mode == "off":
        skip_corpus = True
        logger.info("Corpus layer skipped: corpus_mode=off for this turn")
    elif corpus_mode == "on":
        skip_corpus = False
    else:
        skip_corpus = corpus_layer_skipped_for_query(query)
        if skip_corpus:
            logger.info(
                "Corpus layer skipped: the query is date arithmetic answered by a tool: %s",
                query[:50],
            )
    widen_corpus = corpus_fetch_k > fetch_k and not skip_corpus
    pq_seeds: list[str] = []
    widen_episodic = episodic_fetch_k > fetch_k and episodic is not None
    swapped_answer_ids: set[str] = set()
    (
        epi_entries, corpus_entries, (pseudo_entries, hinted_pq_ids), wide_corpus, wide_episodic,
    ) = await asyncio.gather(
        _search_episodic_layer(
            episodic, query, query_vec, fetch_k, drop_past_answers,
            threshold=store_gate, own_session=own_session,
            create_session=session_id if is_create_mode(mode) else None,
            mode=mode, swapped_sink=swapped_answer_ids, session_id=session_id,
            user_turns=user_turns,
        ),
        _empty_corpus_layer() if skip_corpus else _search_corpus_layer(
            cartridge_mgr, query_vec, fetch_k, timeout_ms=cart_timeout_ms,
            rescore_candidates=rescore_candidates, query_text=query, pq_seeds=pq_seeds,
        ),
        _search_pseudo_query_layer(
            cartridge_mgr, query_vec, fetch_k, timeout_ms=cart_timeout_ms,
        ) if pseudo_enabled and not skip_corpus else _empty_pseudo_layer(),
        _search_corpus_layer(
            cartridge_mgr, query_vec, corpus_fetch_k, timeout_ms=cart_timeout_ms,
            rescore_candidates=rescore_candidates, query_text=query, pq_seeds=pq_seeds,
        ) if widen_corpus else _empty_corpus_layer(),
        _search_episodic_layer(
            episodic, query, query_vec, episodic_fetch_k, drop_past_answers,
            threshold=store_gate, own_session=own_session,
            create_session=session_id if is_create_mode(mode) else None,
            mode=mode, swapped_sink=swapped_answer_ids, session_id=session_id,
            user_turns=user_turns,
        ) if widen_episodic else _empty_corpus_layer(),
    )
    if timer is not None:
        timer.stop("retrieval_ms")

    # Step 3c-2: 転置索引の席 (f_01 §8.1 の 4.3)。corpus 層の結果に入らなかった
    # 語彙上位で、本体 cosine が corpus の relevance の棒以上のものだけ。
    seat_entries: list[StoreEntry] = []
    lexical_candidate_ids: list[str] = []
    seats = int(rag_cfg.get("lexical_seats", 1) or 0)
    if seats > 0 and query and (corpus_entries or not cartridge_mgr is None):
        # 走査は 1 回: 上位 LEXICAL_MISS_CANDIDATES 件を取り、採用済みを除いた
        # 先頭 seats 件が席、全体は取りこぼしの種 (misses) の候補になる。
        lexical_top = await _search_lexical_seat_layer(
            cartridge_mgr, query, query_vec, [], LEXICAL_MISS_CANDIDATES,
            timeout_ms=cart_timeout_ms,
        )
        lexical_candidate_ids = [e[0] for e in lexical_top]
        taken = {entry[0] for entry in corpus_entries}
        candidates = [e for e in lexical_top if e[0] not in taken][:seats]
        # 棒は他の corpus チャンクと同じ keep floor (絶対の棒 ∨ 本体 top1 の相対)。
        seat_bar = _resolve_keep_floor(
            rag_cfg, corpus_thresholds,
            top_raw_score=max((e[1] for e in corpus_entries), default=0.0),
        )
        seat_entries = [e for e in candidates if e[1] >= seat_bar]
        if seat_entries:
            logger.info(
                "Lexical seat: %d chunk(s) placed ahead by the inverted index (cosine %s >= bar %.3f)",
                len(seat_entries), ", ".join(f"{e[1]:.3f}" for e in seat_entries), seat_bar,
            )
            corpus_entries = list(corpus_entries) + seat_entries

    # Step 3d: 疑似クエリの関連性ゲート (f_01 §6.6)。この問いに棒を越える
    # 疑似クエリが 1 本も無ければ、corpus は今回の話題と無関係と見て本体側の
    # 候補ごと引かない。本体 cosine では無関係な問いと正解が重なるが
    # (off-topic top1 max 0.575 / 正解 p25 0.56)、問い↔問いは分離する。
    corpus_gated = False
    if pq_gate is not None and (corpus_entries or pseudo_entries):
        top_pq = max((entry[1] for entry in pseudo_entries), default=0.0)
        passes = top_pq >= pq_gate
        top_body = max((entry[1] for entry in corpus_entries), default=0.0)
        if not passes and not pq_veto:
            # 充足率が低い間は疑似クエリを持たないチャンクへの問いが pq を
            # 持てないので、本体 top1 が正解 top1 の p25 (on_topic) を越える
            # ことも、転置索引の席が立ったこと (固有語の一致) も「話題が合う」
            # 信号として認める (f_01 §6.6 / §8.1 の 4.3、2026-09-12 (b))。
            # 較正が重なる / 問いが少ないときも同じ OR に倒す (2026-09-30)。
            passes = top_body >= on_topic_bar or bool(seat_entries)
        if not passes:
            logger.info(
                "Pseudo-query gate: corpus skipped (top pq cosine %.3f < gate %.3f, "
                "top body %.3f < on_topic %.3f, veto=%s) for query: %s",
                top_pq, pq_gate, top_body, on_topic_bar, pq_veto,
                query[:50],
            )
            corpus_entries = []
            pseudo_entries = []
            corpus_gated = True

    # Step 4: ストア横断のマージ。順位式は 1 本 (c_16 §7.2) なので、層内正規化
    # (``rag.score_normalization``) も RRF も要らない。``merged`` は順位付け用
    # (score 降順)、``merged_raw`` は同じ集合を **素の cosine** で見た射影
    # (品質判定 / gate / floor 用)。2 つは同じ id 集合を指すので、片方で落とした
    # ものをもう片方へ射影する下流の手順が id で完全に揃う。
    merged_entries = _merge_results(epi_entries, corpus_entries)
    # 広く引いた corpus のうち元の幅に無かった行 (Step 6.8 のプール候補だけに使う)。
    step4_ids = {entry[0] for entry in merged_entries} | {entry[0] for entry in pseudo_entries}
    pool_extras = [] if corpus_gated else [e for e in wide_corpus if e[0] not in step4_ids]
    episodic_pool_extras = [e for e in wide_episodic if e[0] not in step4_ids]
    # Step 4.2 / 4.3: 疑似クエリ由来と転置索引の席を先頭に位置で interleave
    # (f_01 §6.3 / §8.1 の 4.3)。席は疑似クエリの後ろ。
    # 疑似クエリ由来を先頭に置くのは充足率 ≥ gate_min_coverage のときだけ —
    # 充足が薄い間は正解チャンクに問いが無く、近くの別チャンクの問いが先頭を
    # 取って recall@5 が 0.617 → 0.511 に落ちる (f_01 §6.3、2026-09-12 (b))。
    # 未充足の間は本体の後ろに並べる (floor は従来どおり pq_gate)。
    pq_lead = _pseudo_query_may_lead(cartridge_mgr, rag_cfg)
    if not pq_lead and pseudo_entries:
        # 本体の後ろに並べても品質 low の経路が cosine 降順で並べ直すため先頭へ
        # 戻る。未充足の間は候補から外す (実測: 「後ろ」と「無し」は同値 0.617)。
        # 例外は取りこぼした問いの言い換え (from_hint) が最良だったもの —
        # その問いの答えのチャンクに、その問いから作った言い換えなので、近くの
        # 別チャンクの問いが勝つ失敗形には当たらない (f_01 §6.3)。
        kept = [e for e in pseudo_entries if e[0] in hinted_pq_ids]
        logger.debug(
            "Pseudo-query entries dropped from the candidates (%d of %d, coverage "
            "below gate_min_coverage; %d hinted kept)",
            len(pseudo_entries) - len(kept), len(pseudo_entries), len(kept),
        )
        pseudo_entries = kept
    # 疑似クエリ由来の列に入れるのは、自身の棒 (Step 6.5 の floor_pseudo) を越える
    # 行だけ (f_01 §6.3)。越えない行を先頭に置くと、同じチャンクの本体の行が id の
    # 畳み込みで消え、その行も floor で落ちて本体 top1 の正解ごと注入されない。
    floor_pseudo = _pseudo_floor(rag_cfg, corpus_thresholds, pq_gate)
    if pseudo_entries and floor_pseudo > 0.0:
        pseudo_entries = [e for e in pseudo_entries if e[1] >= floor_pseudo]
    head = list(pseudo_entries) + [e for e in seat_entries if not corpus_gated]
    merged_entries = _interleave_pseudo(head, merged_entries)
    merged = rank_view(merged_entries)
    merged_raw = gate_view(merged_entries)
    logger.debug("Step 4 merge: %d unique results after dedup", len(merged_raw))

    # Step 4.5: 取得直後の内容精査ゲート — 低価値 chunk を pruning し、後続の
    # 品質判定 / クエリ拡張の候補数を縮小する。
    # create mode を主対象 (chat mode は近似重複除去のみ)。marginal band の
    # prose のみ aux で 1 回関連性判定する (aux 無/cap 超過/error は純ルール)。
    # gate は生スコア (relevance_floor は cosine 前提) で判定するため merged_raw に
    # 適用し、残った chunk_id 集合を正規化側 merged にも射影する。
    gate_cfg = GateConfig.from_rag_cfg(rag_cfg)
    if gate_cfg.enabled and merged_raw:
        if timer is not None:
            timer.start("content_gate_ms")
        try:
            merged_raw = await ChunkContentGate(
                gate_cfg, debug_logger=debug_logger,
            ).filter(
                query, merged_raw, mode,
                aux_client=aux_client,
                tracker=judge_tracker,
                session_id=session_id,
                # 表のチャンクを前置きを除いた行で比べるため (f_01 §3.1.5)
                attrs_of=getattr(cartridge_mgr, "chunk_attrs", None),
            )
        finally:
            if timer is not None:
                timer.stop("content_gate_ms")
        kept = {cid for cid, _, _ in merged_raw}
        merged_entries = [e for e in merged_entries if e[0] in kept]
        merged = rank_view(merged_entries)
        merged_raw = gate_view(merged_entries)
        logger.debug("Step 4.5 content gate: %d results after prune", len(merged_raw))

    # Step 5: Self-RAG 品質判定 (ベクトル閾値、< 0.1ms)
    # 判定は常に生スコア (merged_raw) に対して行う。品質3閾値は cosine 分布前提の
    # ため、順位式のスコアを渡すと閾値の意味が崩れる (c_16 §7.1)。
    # decision.jsonl に記録 (decision_point=``self_rag_judge_path``)
    # 品質は「載せられる候補」で判定する (2026-09-14)。問いだけのチャンク /
    # 今回の問いの反復 / 原文と要約の重複は組立側 (``inference``) で落ちるのに、
    # ここではそれらが top1 になって high を立て、低スコアの corpus を high 枠で
    # 通していた (09-12: top1 は 53/68 が episodic で、多くは同じ問いのノート)。
    eligible = set(
        eligible_rag_indices([t for _, _, t in gate_view(merged_entries)], query),
    )
    if len(eligible) != len(merged_entries):
        logger.debug(
            "Step 4.9 eligibility: %d/%d candidates can be injected",
            len(eligible), len(merged_entries),
        )
        merged_entries = [e for i, e in enumerate(merged_entries) if i in eligible]
        merged = rank_view(merged_entries)
        merged_raw = gate_view(merged_entries)
    # Step 4.95: 問いを答えへ差し替えた行 (自分の過去の応答) は、載せられる corpus の候補が
    # 棒を越える回は外す (不変則 #15、c_16 §7.3 #4)。資格判定 (4.9) の後に置くのは、そこで
    # 落ちる corpus の top1 (問いだけのチャンク等) で決めないため。品質判定・episodic の
    # 相対 floor・再順位段より前に外すので、どれもこの行を見ない。会話そのものを指す問い
    # (``own_session`` が立つ) は答えが会話の中にしか無いので外さない。
    pseudo_ids = {entry[0] for entry in pseudo_entries}
    yielded_answer_ids: set[str] = set()
    if swapped_answer_ids:
        merged_entries, episodic_pool_extras, yielded_answer_ids = _yield_past_answers_to_corpus(
            swapped_answer_ids, merged_entries, episodic_pool_extras,
            pseudo_ids=pseudo_ids, floor_pseudo=floor_pseudo, own_session=own_session,
            rag_cfg=rag_cfg, corpus_thresholds=corpus_thresholds,
            debug_logger=debug_logger, query=query,
        )
        if yielded_answer_ids:
            epi_entries = [e for e in epi_entries if e[0] not in yielded_answer_ids]
            merged = rank_view(merged_entries)
            merged_raw = gate_view(merged_entries)
    quality_judge = RetrievalQualityJudge(thresholds, debug_logger=debug_logger)
    quality = quality_judge.judge(merged_raw)
    logger.debug("Step 5 quality: %s", quality)

    # Step 6: 品質不足時のクエリ拡張フォールバック (生スコアで再検索)。
    expanded, quality = await _try_quality_expansion(
        quality, merged_entries, quality_judge, query, query_vec,
        working_mem, episodic, top_k, noise_sigma,
    )
    if expanded is not merged_entries:
        merged_entries = expanded
        merged = rank_view(merged_entries)
        merged_raw = gate_view(merged_entries)

    # Step 6.5: 品質 low の結果はそのままでは添付しない。クエリ拡張 (Step 6) を
    # 経ても low のままなら、無関連チャンクをコンテキストへ注入する害の方が大きい
    # (2026-07-15: final_quality=low の 13 件がそのまま添付され、内容は全て
    # 無関連の過去雑談ノートだった)。「low と判定したのに全件添付」を塞ぐ。
    #
    # ただし判定は merged 全体に対する **単一スカラ (top_score)** で、これは
    # 「この質問に記憶が要るか」を弁別しない。実測 (2026-08-12、STM 94 / LTM 103、
    # 監査 24 クエリ): 記憶が要るクエリの top_score 中央値 0.472 に対し、
    # 要らないクエリは 0.541 と **逆転** していた。閾値をどこに置いても
    # 誤注入 >= recall になる一方、正解ノート自体は 5 プローブ中 3 件で
    # merged 1 位に来ていた (0.547 / 0.605 / 0.544)。つまり検索は当たっており、
    # 全件破棄だけが効いていた (31 ターンの実会話で採用 0 件)。
    #
    # そこで「クエリ単位の全件破棄」ではなく「チャンク単位のフロア」で絞る。
    # 実測のフロア別 recall / 漏れ (正解が残る件数 / 記憶不要クエリの通過チャンク
    # 数): 0.45 -> 3/5・平均 1.3 件、0.40 -> 4/5・平均 1.3 件、0.35 -> 5/5・平均
    # 2.8 件。上記インシデントの 13 件とは桁が違う。
    #
    # フロアは生スコア (cosine スケール) 前提なので merged_raw で判定し、
    # 残った chunk_id を正規化側 merged に射影する (Step 4.5 と同じ形)。
    # 0.0 で無効化 = 従来どおり「low はクエリ単位で全件破棄 / それ以外は全通し」。
    # **進化対象にはしない**: このゲートはモデルが何を見るかを決めるため、
    # モデル自身の出力由来の turn_outcome で自動調整すると閉ループになる。
    #
    # フロアの値は threshold_mode に従う。``auto`` (較正が効いている) では
    # **較正済み relevance と同じ棒**を使う。静的既定 0.40 は旧スケール
    # (STM の combined) で決めた値で、cosine スケールでは実測 need の下限 0.338 /
    # p25 0.367 を上回り、救済すべき単発ヒットを捨てる。relevance と同じ棒に
    # すれば「関連性の棒を越えたチャンクだけを残す」という一貫した意味になる。
    # フロアは **集計判定と独立に常時**掛ける。``quality`` は merged 全体に対する
    # 単一スカラで、「セットとして使えるか」しか見ない。top1 が強ければ high に
    # なり、**同じ検索の 2〜8 位が無関連でも全部通っていた**。
    #
    # 実測 (2026-08-16 再測定、chat 23 ターン): quality は high 12 / medium 7 /
    # low 4 で、フロアが掛かったのは low の 4 ターンだけ。残り 19 ターン (83%) は
    # merge 8 件が無条件で全通しになり、``[参考情報]`` の **67% (58/87 件) /
    # 73% (8,711/11,904 tok)** が別セッションの mdp_trace ログで埋まっていた。
    # それらの対クエリ類似度は平均 0.119 で、較正済み閾値 0.259 を **1 件も
    # 超えていない**。閾値の分離性能自体は健全で、同じ実測で関連チャンクは 0.467、
    # 無関連は 0.196 以下だった。効いていなかったのは適用範囲だけ。
    #
    # ``quality`` の役割は「フロアを掛けるか」ではなく「フロアを通ったものが
    # 1 件も無いときにどう扱うか」に限定する。
    # 疑似クエリ由来 (f_01 §6.3) は cosine が問い↔問いのスケールなので、
    # (a) 相対の棒の top1 には数えない — 本体側の弱い兄弟を高い棒で消さない、
    # (b) 自身は絶対の棒 (top1 を渡さない = 相対 0 / 弱い集合の上乗せ無し)
    # だけを越えればよい。同じ棒に 2 つのスケールを流さないための分岐。
    # 棒はストア別 (f_01 §6.6): episodic は記憶側の較正、corpus 本体は corpus 側の
    # 較正 (未較正なら記憶側)。相対の棒の top1 も同じストアの候補から取る。
    corpus_ids, floor_corpus = _corpus_keep_floor(
        merged_raw, pseudo_ids, rag_cfg, corpus_thresholds,
    )
    floor_epi = _resolve_keep_floor(
        rag_cfg, thresholds,
        top_raw_score=max(
            (s for cid, s, _ in merged_raw if cid not in pseudo_ids and cid not in corpus_ids),
            default=0.0,
        ),
    )

    def _floor_for(cid: str) -> float:
        if cid in pseudo_ids:
            return floor_pseudo
        return floor_corpus if cid in corpus_ids else floor_epi

    # 記録する棒は **候補が居るストアのもの** だけにする。候補ゼロのストアの棒
    # (``top_raw_score=0`` から導かれるので常に静的値) まで max() に入れると、
    # 1 件も掛かっていない棒がログに載り、採用値がそれを割って見える
    # (2026-09-16 ライブ監査 F-04)。判定そのものは従来どおり ``_floor_for``。
    applied_floors: dict[str, float] = {}
    for cid, _score, _ in merged_raw:
        if cid in pseudo_ids:
            applied_floors["pseudo"] = floor_pseudo
        elif cid in corpus_ids:
            applied_floors["corpus"] = floor_corpus
        else:
            applied_floors["episodic"] = floor_epi
    floor = max(floor_epi, floor_corpus)
    logged_floor = max(applied_floors.values(), default=floor)
    episodic_rejected: list[str] = []
    if floor > 0.0:
        kept_ids = {
            cid for cid, score, _ in merged_raw if score >= _floor_for(cid)
        }
        episodic_rejected = [
            cid for cid, _, _ in merged_raw
            if cid not in kept_ids and cid not in pseudo_ids and cid not in corpus_ids
        ]
        passed = [t for t in merged if t[0] in kept_ids]
        if len(passed) != len(merged):
            logger.info(
                "Relevance floor: %d/%d chunks passed floors=%s "
                "(quality=%s) for query: %s",
                len(passed), len(merged),
                {k: round(v, 3) for k, v in applied_floors.items()},
                quality, query[:50],
            )
        if debug_logger is not None:
            debug_logger.log_rag_selection(
                query=query,
                quality=quality,
                floor=logged_floor,
                floors=applied_floors,
                kept=[(cid, s) for cid, s, _ in merged_raw if cid in kept_ids],
                rejected=[
                    (cid, s) for cid, s, _ in merged_raw if cid not in kept_ids
                ],
            )
        merged = passed
    elif quality == "low":
        # フロア無効 (0.0) の構成では従来どおり「low はクエリ単位で全件破棄」。
        logger.info(
            "Search results discarded (quality=low, floor disabled) "
            "for query: %s", query[:50],
        )
        merged = []

    if not merged:
        _log_memory_search_state(
            debug_logger, context_count, epi_entries,
            semmem_stats=semmem_stats,
        )
        return SearchResult(
            sources=[],
            quality=quality,
            from_memory=bool(epi_entries),
            corpus_gated=corpus_gated,
            lexical_candidate_ids=lexical_candidate_ids,
            # corpus の候補があったのに棒で全部落ちた (f_01 §6.4 の misses の引き金)。
            # 較正済みでゲートを通った場合だけ (未較正では話題の判定が無い)。
            corpus_starved=pq_gate is not None and bool(corpus_ids),
            episodic_rejected_ids=episodic_rejected,
            # 採用ゼロでも転置索引の種は積む (採用に関わらず、f_01 §6.4)。
            usage=SearchUsage(
                episodic=episodic, cartridge_mgr=cartridge_mgr, sources=(),
                record_corpus_hits=False, pq_seeds=pq_seeds,
            ),
        )

    # 採用に載る corpus の有無は再順位の前に決まる (位置の集合を保つので)。reactive の
    # 証拠による昇格 (docs/f_03 §2.2) が再順位の完了を待たずに判定できるよう知らせる。
    if on_corpus_evidence is not None:
        early = early_corpus_evidence(
            merged, top_k,
            seat_ids={entry[0] for entry in seat_entries},
            corpus_gated=corpus_gated,
            rerank_pending=reranker is not None,
            has_pool_extras=bool(pool_extras),
        )
        if early is not None:
            on_corpus_evidence(early)

    # Step 6.8: 再順位段 (c_16 §7.2.1)。floor を通った corpus / episodic の候補だけを
    # 1 回の呼出で並べ替え、各ストアが占める位置の集合は変えない (ストア間の配分・
    # store_prior・freshness を保つ)。
    reranked_ids: set[str] = set()
    outcome: RerankOutcome | None = None
    if reranker is not None:
        extras = _eligible_pool_extras(query, pool_extras, merged, floor=floor_corpus)
        episodic_extras = _eligible_episodic_extras(
            query, episodic_pool_extras, [*merged, *extras], floor=floor_epi,
        )
        outcome = await _rerank_corpus(
            query, merged, reranker, extras=extras, episodic_extras=episodic_extras, timer=timer,
        )
        merged = outcome.merged
        if outcome.scores is not None:
            reranked_ids = set(outcome.pool_ids)
            # 採用された広げた幅の corpus 行も「corpus の候補」に数える (corpus_starved の判定用)。
            corpus_ids |= {cid for cid, _, _ in extras} & {cid for cid, _, _ in merged}

    # Step 7: 最終順位付け (merged は順位式のスコア降順。Step 6.8 が並べ替えたらその順) から top_k 件を採用。
    # カートリッジ公平性保証 (旧 Step 7.5) と語彙アンカーの随伴 (旧 Step 7.55) は
    # 廃止した (c_16 §7)。前者は ``priority`` 由来の席の奪い合いを補正するための
    # 装置で、順位式が 1 本になり ``store_prior`` が明示の係数になった今は、
    # 「席を取れなかった」= 「順位式で負けた」であって補正する理由が無い。後者は
    # 候補生成が転置索引を ``EvidenceStore`` の内部へ取り込んだ結果 (c_16 §6.3:
    # lexical はスコアを持ち込まない)、どの候補が語彙由来かが外へ出てこなくなった。
    final_sources = merged[:top_k]
    # 転置索引の席 (f_01 §8.1 の 4.3): 棒を越えて残った席が中間の並べ替えで
    # top_k の外へ出ていたら、末尾の 1 席と入れ替える (席は位置の予約であって
    # スコアではない)。再順位段のプールに入って判定を受けた席は保証しない
    # (実測: 保証すると r@5 63.7 → 62.5、c_16 §7.2.1)。
    guarded_seats = {entry[0] for entry in seat_entries} - reranked_ids
    if guarded_seats and not corpus_gated:
        seat_ids = guarded_seats
        chosen = {cid for cid, _, _ in final_sources}
        for entry in merged[top_k:]:
            if entry[0] in seat_ids and entry[0] not in chosen and final_sources:
                final_sources = final_sources[:-1] + [entry]
                chosen.add(entry[0])
                if len(chosen & seat_ids) >= len(seat_ids):
                    break

    # Step 7.2: 6.8 が episodic を並べ替えた回だけ、採用した訂正の組を訂正後が上になるよう
    # 入れ替える (f_02 §5.3)。top_k の後に掛けるので採用の集合は変わらない — 前に掛けると
    # 訂正前が席から押し出され、話題語を持たない訂正だけが残る。訂正前だけの組は 7.6 に任せる。
    swapped: list[tuple[str, str]] = []
    if outcome is not None and outcome.episodic_scores is not None:
        final_sources, swapped = enforce_correction_order(
            final_sources, correction_pairs(correction_successors),
        )
    _log_rerank_applied(debug_logger, outcome, swapped)
    _log_unadopted_yield(
        yielded_answer_ids, final_sources, pseudo_ids, debug_logger=debug_logger, query=query,
    )

    # Step 7.6: 採用ノートが後続の訂正で上書きされているなら、訂正も一緒に出す。
    # top_k で切った **後** に足す — 訂正は席を争う候補ではなく随伴情報であり、
    # floor / top_k のどちらで落ちても「訂正前の値だけが残る」状態になる。
    final_sources = attach_superseding_corrections(
        episodic, final_sources, correction_trail,
    )
    # Step 7.65: 採用した corpus チャンクに直前チャンク (同文書・同大節) の末尾を
    # 「(前の文脈)」として随伴させる (f_01 §8.1 の 7.65)。席を争わない随伴。
    final_sources = attach_previous_chunk_context(
        cartridge_mgr, final_sources,
        tail_chars=int(rag_cfg.get("previous_context_chars", 300) or 0),
    )

    # Step 7.7: 採用した id の「使った」記録は消費側が注入を確定してから積む
    # (SearchUsage)。ここで積むと結果を捨てた回まで記録される。

    logger.info(
        "Search completed: %d results, quality=%s, from_memory=%s",
        len(final_sources), quality, bool(epi_entries),
    )
    _log_memory_search_state(
        debug_logger, context_count, epi_entries,
        semmem_stats=semmem_stats,
    )

    return SearchResult(
        sources=final_sources,
        quality=quality,
        from_memory=bool(epi_entries),
        top_raw_score=_top_raw_score(
            final_sources, merged_raw,
            # 広げた幅は採用された追加行の cosine 用。元の幅と重なる id は元の幅の値を採る (後勝ち)。
            body_entries=[*wide_corpus, *wide_episodic, *epi_entries, *corpus_entries],
            pseudo_ids=pseudo_ids,
        ),
        evidence_ids=evidence_ids_of(final_sources, _corpus_versions(cartridge_mgr)),
        corpus_gated=corpus_gated,
        pseudo_derived=sum(1 for cid, _, _ in final_sources if cid in pseudo_ids),
        lexical_candidate_ids=lexical_candidate_ids,
        # 較正済みのゲートを通った (話題は合う) のに注入ゼロ。未較正の構成では
        # 「話題が合う」の判定が無いので立てない (無関係な問いに注記を付けない)。
        corpus_starved=pq_gate is not None and bool(corpus_ids) and not any(
            cid in corpus_ids for cid, _, _ in final_sources
        ),
        episodic_rejected_ids=episodic_rejected,
        usage=SearchUsage(
            episodic=episodic, cartridge_mgr=cartridge_mgr,
            sources=tuple(final_sources), record_corpus_hits=pseudo_enabled,
            pq_seeds=pq_seeds,
        ),
    )


def early_corpus_evidence(
    merged: Sequence[tuple[str, float, str]],
    top_k: int,
    *,
    seat_ids: set[str],
    corpus_gated: bool,
    rerank_pending: bool,
    has_pool_extras: bool,
) -> int | None:
    """Step 7 の採用に載る corpus の件数を、Step 6.8 (再順位段) の前に決める。

    決まらなければ ``None``。判定は最終の採用 (``SearchResult.evidence_ids`` の corpus) と
    同じでなければならない (reactive の昇格の shadow / on / 最終採用を食い違わせない):

    - ``corpus_gated`` は 0 (呼び手の ``_corpus_evidence`` も gated を 0 と数える)。
    - ``merged[:top_k]`` に corpus があれば、その件数で確定する。再順位段は各ストアの
      位置の集合を保ち (追加行は ``merged`` の後ろに足すので ``top_k`` 内の corpus の位置は
      減らない)、席の保証 (Step 7) は末尾 1 席を corpus の席と入れ替えるだけで、7.2 は
      採用の中の入れ替え、7.6 は episodic の訂正、7.65 は随伴 (呼び手が件数から除く)。
    - 先頭に corpus が無い回: 再順位段が無ければ、``top_k`` の外の席が Step 7 で入るか
      (1 件) / 入らないか (0) で確定する。再順位段があると、席の保証 (並べ替えた席は
      保証しない) と追加行 (``merged`` が ``top_k`` に満たないとき ``top_k`` 内へ入る) が
      再順位の成否で変わるので、その可能性がある回だけ ``None`` (完了を待つ)。
    """
    if corpus_gated:
        return 0
    head = sum(1 for cid, _score, _text in merged[:top_k] if _store_of(cid) == "corpus")
    if head:
        return head
    tail_seat = bool(merged[:top_k]) and any(
        cid in seat_ids for cid, _score, _text in merged[top_k:]
    )
    if not rerank_pending:
        return 1 if tail_seat else 0
    extras_may_enter = len(merged) < top_k and has_pool_extras
    if tail_seat or extras_may_enter:
        return None
    return 0


class Reranker(Protocol):
    """再順位段のクライアント (EvorefGen の ``RerankClient`` が満たす面)。

    ``candidates`` は 1 回に送る件数の上限、``token_budget`` は近似トークンの予算 (``None`` なら件数だけ)。
    """

    candidates: int
    token_budget: float | None

    async def rerank(
        self, query: str, documents: Sequence[str], *, ids: Sequence[str] | None = None,
    ) -> list[float] | None: ...


def rerank_active(reranker: Reranker | None) -> bool:
    """再順位段がこのターンで動きうるか。無い / 遮断器の冷却中 (送っても弾かれる) なら偽。

    冷却が明けて試しを送れる間は真 (送らないと試しが起きず復帰しない)。
    """
    return reranker is not None and not getattr(reranker, "breaker_blocking", False)


def _rerank_candidates(reranker: Reranker | None) -> int:
    """1 回に並べ替える件数の上限 (``rag.rerank.max_candidates``)。再順位段が無い / 遮断中なら 0。

    実際に送る件数はトークン予算でこれより少なくなりうる (:func:`_rerank_corpus`)。
    """
    if not rerank_active(reranker):
        return 0
    return max(0, int(reranker.candidates))


def store_rerank_positions(
    merged: Sequence[tuple[str, float, str]], limit: int, store: str,
) -> list[int]:
    """``merged`` の中で ``store`` の項目が占める位置 (現順位で上位 ``limit`` 件)。"""
    positions = [i for i, (cid, _score, _text) in enumerate(merged) if _store_of(cid) == store]
    return positions[:max(0, limit)]


def corpus_rerank_positions(
    merged: Sequence[tuple[str, float, str]], limit: int,
) -> list[int]:
    """``merged`` の中で corpus 項目が占める位置 (現順位で上位 ``limit`` 件)。"""
    return store_rerank_positions(merged, limit, "corpus")


def apply_store_rerank(
    merged: Sequence[tuple[str, float, str]],
    scores: Sequence[float],
    limit: int,
    store: str,
) -> list[tuple[str, float, str]]:
    """``store`` の位置の集合を保ったまま、候補プール (上位 ``limit`` 件) を再順位で並べる。

    他のストアの項目と位置は動かさない。``store`` の位置には「プールを再順位の降順 (同点は
    現順位) → プール外の同じストアの項目を現順位」で詰める。``scores`` はプールの入力順。
    動くのは id と本文だけで、順位式のスコアは位置に残す — 後段 (``SalienceRanker`` の予算選別) が
    並べ替える前の順位のスコアで選ばないよう、位置とスコアの単調性を保つ。
    """
    positions = [i for i, (cid, _score, _text) in enumerate(merged) if _store_of(cid) == store]
    pool = positions[:max(0, limit)]
    if len(scores) != len(pool):
        raise ValueError(f"rerank scores ({len(scores)}) do not match the pool ({len(pool)})")
    order = sorted(range(len(pool)), key=lambda j: -scores[j])
    sequence = [pool[j] for j in order] + positions[len(pool):]
    out = list(merged)
    for slot, source in zip(positions, sequence):
        cid, _score, text = merged[source]
        out[slot] = (cid, merged[slot][1], text)
    return out


def apply_corpus_rerank(
    merged: Sequence[tuple[str, float, str]],
    scores: Sequence[float],
    limit: int,
) -> list[tuple[str, float, str]]:
    """corpus の位置の集合を保ったまま、候補プール (上位 ``limit`` 件) を再順位で並べる。"""
    return apply_store_rerank(merged, scores, limit, "corpus")


def _eligible_pool_extras(
    query: str,
    extras: Sequence[StoreEntry],
    merged: Sequence[tuple[str, float, str]],
    *,
    floor: float,
) -> list[tuple[str, float, str]]:
    """広く引いた層の追加行のうち、プールに足してよいもの (``(id, score, text)``)。

    元の幅の候補と同じ関所を掛ける: そのストアの floor (素の cosine)、注入できる本文か
    (``eligible_rag_indices``)、``merged`` と同じ言明でないか (``claim_key``)。並びは順位式の降順。
    """
    seen_ids = {cid for cid, _score, _text in merged}
    seen_claims = {_claim_key_of(text) for _cid, _score, text in merged if text}
    passed = [e for e in extras if e[0] not in seen_ids and e[1] >= floor]
    eligible = set(eligible_rag_indices([e[3] for e in passed], query))
    out: list[tuple[str, float, str]] = []
    for i, (cid, _cos, score, text) in enumerate(sorted(passed, key=lambda e: -e[2])):
        claim = _claim_key_of(text) if text else ""
        if i not in eligible or (claim and claim in seen_claims):
            continue
        seen_ids.add(cid)
        if claim:
            seen_claims.add(claim)
        out.append((cid, score, text))
    return out


def _eligible_episodic_extras(
    query: str,
    extras: Sequence[StoreEntry],
    merged: Sequence[tuple[str, float, str]],
    *,
    floor: float,
) -> list[tuple[str, float, str]]:
    """広く引いた episodic の追加行のうち、プールに足してよいもの。

    :func:`_eligible_pool_extras` の関所に加えて、元の幅の episodic と **同じ属性スロット**
    を述べる行は入れない — 「スロットごとの最新の言明」(:func:`_latest_statement_per_slot`)
    は元の幅で決めたので、広げた幅から同じスロットの別の言明を足すと、元の幅では 1 件に
    絞った値が 2 件並ぶ (c_16 §7.2.1 の第 4 段階)。
    """
    from backend.free.memory.notes.note_builder import restated_attribute_slot

    candidates = [e for e in extras if _store_of(e[0]) == "episodic"]
    passed = _eligible_pool_extras(query, candidates, merged, floor=floor)
    if not passed:
        return passed
    taken_slots = {
        slot for cid, _score, text in merged
        if _store_of(cid) == "episodic" and (slot := restated_attribute_slot(text)) is not None
    }
    # 広げた幅の中の重複は層 (``_latest_statement_per_slot``) が既に 1 スロット 1 件に畳んでいる。
    return [e for e in passed if restated_attribute_slot(e[2]) not in taken_slots]


@dataclass(frozen=True)
class RerankOutcome:
    """Step 6.8 の結果。``scores`` は corpus のプールの入力順 (``pool_ids`` と同じ並び)。

    corpus を並べなかった (候補 2 件未満) / 縮退した回は ``scores`` が ``None``。``episodic_scores`` / ``episodic_pool_ids``
    は episodic のプールの同じ値 (並べなかった回は ``None`` / 空)。縮退した回の ``merged`` は入力のまま。
    """

    merged: list[tuple[str, float, str]]
    pool_ids: tuple[str, ...] = ()
    scores: tuple[float, ...] | None = None
    episodic_pool_ids: tuple[str, ...] = ()
    episodic_scores: tuple[float, ...] | None = None
    #: 件数の上限・予算で切る前の、ストアごとの送る資格のある行数 (``merged`` + 追加行)。
    eligible: Mapping[str, int] = field(default_factory=dict)
    #: ストアごとに送った件数 (退化で縮退したストアも送った数のまま)。
    sent: Mapping[str, int] = field(default_factory=dict)
    #: 送った組のコストの合計 (:func:`rerank_pair_cost`、予算と同じ単位)。
    sent_tokens: float = 0.0
    token_budget: float | None = None
    max_candidates: int = 0


def _pool_order(pool_ids: Sequence[str], scores: Sequence[float] | None) -> list[str]:
    """プールの id を再順位の降順 (同点は現順位) に並べる。スコアが無ければ空。"""
    if scores is None:
        return []
    return [pool_ids[j] for j in sorted(range(len(pool_ids)), key=lambda j: -scores[j])]


async def _rerank_corpus(
    query: str,
    merged: list[tuple[str, float, str]],
    reranker: Reranker,
    *,
    extras: Sequence[tuple[str, float, str]] = (),
    episodic_extras: Sequence[tuple[str, float, str]] = (),
    timer: "StageTimer | None" = None,
) -> RerankOutcome:
    """Step 6.8 の本体。corpus と episodic のプールを 1 回の呼出で並べ替える (c_16 §7.2.1)。

    ``extras`` / ``episodic_extras`` (元の幅に無かった追加行) は ``merged`` の後ろに置いて
    プールの候補にし、縮退した回は捨てる (結果は再順位段が無いときと同じ)。総数は
    ``candidates`` 件と ``token_budget`` (近似トークン) の早い方まで (:func:`fit_rerank_pool`) で、
    corpus が現順位で先に取り、episodic は残りの枠と残りの予算を取る (corpus のプールと
    注入の確認の較正の前提を変えないため)。各ストアとも 2 件未満なら送らない。
    位置の集合はストアごとに保つ。失敗・締切超過・退化 (``rerank`` が ``None``) は現順位のまま
    (チャットを止めない、e_03 §4.4)。
    """
    limit = _rerank_candidates(reranker)
    widened = (
        list(merged)
        + [e for e in extras if _store_of(e[0]) == "corpus"]
        + [e for e in episodic_extras if _store_of(e[0]) == "episodic"]
    )
    budget = reranker.token_budget
    corpus_all = store_rerank_positions(widened, len(widened), "corpus")
    epi_all = store_rerank_positions(widened, len(widened), "episodic")
    n_corpus = fit_rerank_pool(
        query, [widened[i][2] for i in corpus_all], max_count=limit, token_budget=budget,
    )
    positions = corpus_all[:n_corpus] if n_corpus >= 2 else []
    used = sum(rerank_pair_cost(query, widened[i][2]) for i in positions)
    n_epi = fit_rerank_pool(
        query, [widened[i][2] for i in epi_all], max_count=limit - len(positions),
        token_budget=None if budget is None else budget - used,
    )
    epi_positions = epi_all[:n_epi] if n_epi >= 2 else []
    if not positions and not epi_positions:
        return RerankOutcome(merged=merged)
    sent = positions + epi_positions
    pool_stats = {
        "eligible": {"corpus": len(corpus_all), "episodic": len(epi_all)},
        "sent": {"corpus": len(positions), "episodic": len(epi_positions)},
        "sent_tokens": used + sum(rerank_pair_cost(query, widened[i][2]) for i in epi_positions),
        "token_budget": budget,
        "max_candidates": limit,
    }
    if timer is not None:
        timer.start("rerank_ms")
    try:
        scores = await reranker.rerank(
            query,
            [widened[i][2] for i in sent],
            ids=[widened[i][0] for i in sent],
        )
    except Exception as e:  # noqa: BLE001 — 再順位の失敗で応答を止めない
        logger.warning("Rerank failed, keeping the current order: %s: %s", type(e).__name__, e)
        scores = None
    finally:
        if timer is not None:
            timer.stop("rerank_ms")
    if scores is None or len(scores) != len(sent):
        return RerankOutcome(merged=merged, **pool_stats)
    corpus_scores = [float(s) for s in scores[:len(positions)]]
    epi_scores = [float(s) for s in scores[len(positions):]]
    # 呼出全体では健全でも、ストアの中で退化 (全 0 / 同値) していればそのストアは縮退扱い
    # (現順位のまま、追加行は入れない)。
    corpus_degenerate = bool(positions) and degenerate_reason(
        corpus_scores, documents=[widened[i][2] for i in positions],
    ) is not None
    epi_degenerate = bool(epi_positions) and degenerate_reason(
        epi_scores, documents=[widened[i][2] for i in epi_positions],
    ) is not None
    if corpus_degenerate or epi_degenerate:
        logger.info(
            "Rerank: degenerate scores inside a store (corpus=%s, episodic=%s); keeping its order",
            corpus_degenerate, epi_degenerate,
        )
    if corpus_degenerate:
        positions, corpus_scores = [], []
    if epi_degenerate:
        epi_positions, epi_scores = [], []
    if not positions and not epi_positions:
        return RerankOutcome(merged=merged, **pool_stats)
    applied = positions + epi_positions
    out = widened
    if positions:
        out = apply_store_rerank(out, corpus_scores, len(positions), "corpus")
    if epi_positions:
        out = apply_store_rerank(out, epi_scores, len(epi_positions), "episodic")
    # 追加行のうちプールに送らなかったものは判定を受けていないので入れない (補充しない)。
    applied_set = set(applied)
    unsent_extras = {
        widened[i][0] for i in range(len(merged), len(widened)) if i not in applied_set
    }
    if unsent_extras:
        out = [e for e in out if e[0] not in unsent_extras]
    logger.info(
        "Rerank: %d corpus / %d episodic candidate(s) reordered (%d from the widened fetch; "
        "eligible %d / %d, ~%.0f tokens of budget %s, cap %d)",
        len(positions), len(epi_positions), sum(1 for i in applied if i >= len(merged)),
        len(corpus_all), len(epi_all), pool_stats["sent_tokens"],
        "none" if budget is None else f"{budget:.0f}", limit,
    )
    return RerankOutcome(
        merged=out,
        pool_ids=tuple(widened[i][0] for i in positions),
        scores=tuple(corpus_scores) if positions else None,
        episodic_pool_ids=tuple(widened[i][0] for i in epi_positions),
        episodic_scores=tuple(epi_scores) if epi_positions else None,
        **pool_stats,
    )


def correction_pairs(
    correction_successors: Mapping[str, Sequence[str]] | None,
) -> list[tuple[str, str]]:
    """訂正の組 ``(訂正前のノート id, 訂正後のノート id)`` (f_02 §5.3)。

    SemMem の世代から引いた後継台帳 (``{被訂正ノート id: 訂正後の根拠ノート id 列}``) **だけ**
    を使う。検証済みの訂正 (``from_correction``) だけが世代を作り、宛先も抽出が決めたもの
    (不変則 #12 / #13)。同一セッションの ``corrections_by_target`` は訂正かどうかも宛先も
    字句の候補なので、順位を入れ替える根拠にしない (7.6 の注記には従来どおり使う)。
    自分自身を指す組は捨てる。
    """
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for before, afters in (correction_successors or {}).items():
        for after in afters:
            pair = (str(before), str(after))
            if not pair[0] or not pair[1] or pair[0] == pair[1] or pair in seen:
                continue
            seen.add(pair)
            pairs.append(pair)
    return pairs


def enforce_correction_order(
    sources: Sequence[tuple[str, float, str]],
    pairs: Sequence[tuple[str, str]],
) -> tuple[list[tuple[str, float, str]], list[tuple[str, str]]]:
    """訂正の組が両方 ``sources`` にあれば、訂正後が訂正前より上になるよう 2 つの位置を入れ替える。

    入れ替えるのは組の 2 つの位置の id と本文だけで、順位式のスコアは位置に残す
    (:func:`apply_store_rerank` と同じ契約)。集合と他の項目の位置は変えない (多値スロットの兄弟の
    ノートは組に入らないので動かない、不変則 #13)。訂正前だけがある組は触らない (手順 7.6 が
    注記と随伴を付ける)。互いに逆向きの組 (矛盾) は両方とも使わない。それより長い循環でも
    止まるよう走査は組の数 + 1 回で打ち切り、入れ替えた組は 1 回だけ数える。

    Returns:
        ``(並べ直した sources, 入れ替えた組 (訂正前, 訂正後))``。
    """
    out = list(sources)
    swapped: list[tuple[str, str]] = []
    pair_set = set(pairs)
    pairs = [(b, a) for b, a in pairs if (a, b) not in pair_set]
    if not pairs or len(out) < 2:
        return out, swapped
    for _ in range(len(pairs) + 1):
        changed = False
        for before, after in pairs:
            pos = {cid: i for i, (cid, _score, _text) in enumerate(out)}
            i, j = pos.get(before), pos.get(after)
            if i is None or j is None or j < i:
                continue
            (bid, b_score, btext), (aid, a_score, atext) = out[i], out[j]
            out[i], out[j] = (aid, b_score, atext), (bid, a_score, btext)
            if (before, after) not in swapped:
                swapped.append((before, after))
            changed = True
        if not changed:
            break
    if swapped:
        logger.info(
            "Rerank: moved %d correction(s) above the note(s) they supersede", len(swapped),
        )
    return out, swapped


def _log_rerank_applied(
    debug_logger, outcome: RerankOutcome | None, swapped: Sequence[tuple[str, str]],
) -> None:
    """``op="rerank_apply"``: ストア別の前後 id と訂正の入れ替え (本文は出さない)。

    あわせて、件数の上限・予算で切る前のストア別の適格件数・送った件数・送った近似トークンを残す。
    """
    if debug_logger is None or outcome is None:
        return
    by_store: dict[str, dict[str, list[str]]] = {}
    if outcome.scores is not None:
        by_store["corpus"] = {
            "before": list(outcome.pool_ids),
            "after": _pool_order(outcome.pool_ids, outcome.scores),
        }
    if outcome.episodic_scores is not None:
        by_store["episodic"] = {
            "before": list(outcome.episodic_pool_ids),
            "after": _pool_order(outcome.episodic_pool_ids, outcome.episodic_scores),
        }
    if not by_store:
        return
    debug_logger.log_rerank_applied(
        by_store=by_store, correction_swaps=len(swapped), swapped_pairs=list(swapped),
        eligible=dict(outcome.eligible), sent=dict(outcome.sent),
        sent_tokens=round(outcome.sent_tokens),
        token_budget=None if outcome.token_budget is None else round(outcome.token_budget),
        max_candidates=outcome.max_candidates,
    )


async def _empty_pseudo_layer() -> tuple[list[StoreEntry], set[str]]:
    return [], set()


#: 日付演算が閉じている印 (数字 / 漢数字)。agent 側の同名判定と同じ文字集合。


async def _empty_corpus_layer() -> list[StoreEntry]:
    return []


def corpus_layer_skipped_for_query(query: str) -> bool:
    """corpus (文書) 層を引かない問いか (純粋関数)。

    日付演算の手掛かり (:func:`query_has_date_math_cue`) に **数量** (数字 /
    漢数字) を伴う問いはツールが答える閉じた演算なので、文書に根拠を求めない。
    照応 (「その日」) と組み合わさる形が典型で、設計書に同じ例文があると
    較正済みの棒でも通ってしまう。手掛かり語だけの問い (「設計書では祝日の
    扱いをどう決めていますか」) は文書への問いでありうるので切らない。

    手掛かりが **日数の問いだけ** のときは、発話だけで両端が閉じる (日付がある /
    今年の残り) ときに限る。「設計書 v2 の締切まであと何日？」は数量 (v2) を
    持つが終点は文書にある — 日数の語彙を手掛かりに合成した 2026-09-27 に、
    こうした問いで corpus を引かなくなっていた (独立レビュー)。
    """
    text = query or ""
    if day_count_is_the_only_cue(text):
        return day_count_closed_in_query(text)
    return query_has_date_math_cue(text) and bool(NUMERAL_HINT_RE.search(text))


def _record_corpus_hits(
    cartridge_mgr, sources: list[tuple[str, float, str]],
) -> None:
    """注入した corpus チャンク id を疑似クエリの lazy 生成対象へ溜める (f_01 §6.4)。

    プロセス内バッファで、ディスクには sleep-time Step 5.9 が書く。
    """
    if cartridge_mgr is None or not hasattr(cartridge_mgr, "record_pq_hits"):
        return
    ids = [cid for cid, _, _ in sources if _store_of(cid) == "corpus"]
    if not ids:
        return
    try:
        cartridge_mgr.record_pq_hits(ids)
    except Exception as e:  # noqa: BLE001 — 観測のための記録で応答を止めない
        logger.warning("Failed to record corpus hits: %s", e)


def _record_episodic_usage(
    episodic, sources: list[tuple[str, float, str]],
) -> None:
    """注入したエピソード記憶の id をプロセス内バッファへ溜める。

    ディスクには触らない (書き込みは sleep-time)。カートリッジ由来の id は
    ストアに無いので黙って落ちる。
    """
    if episodic is None or not sources:
        return
    usage = getattr(episodic, "usage", None)
    if usage is None:
        return
    try:
        usage.add_many([
            cid for cid, _, _ in sources if episodic.get(cid) is not None
        ])
    except Exception as e:  # noqa: BLE001 — 観測のための記録で応答を止めない
        logger.warning("Failed to record episodic usage: %s", e)


def _top_raw_score(
    final_sources: list[tuple[str, float, str]],
    merged_raw: list[tuple[str, float, str]],
    *,
    body_entries: list[StoreEntry] | None = None,
    pseudo_ids: set[str] | None = None,
) -> float | None:
    """採用チャンクの生スコア (cosine スケール) の最大値を返す。

    `final_sources` のスコアは順位式 (``cos × freshness × confidence ×
    store_prior``) なので観測値には使えない。`merged_raw` 側の素の cosine を
    chunk_id で引き直す。gate/拡張で `merged_raw` から落ちた chunk は引けない
    ので単に除外する。1 件も引けなければ `None`。
    """
    if not final_sources:
        return None
    # 疑似クエリ由来 (f_01 §6.3) の ``merged_raw`` の値は問い↔問いの cosine で
    # スケールが違う。``rag_top1_score`` は「検索器の top-1 cosine」(f_04) として
    # Level 0 経験へ記録され few-shot の fitness / phase3 の入力になるので、
    # 本体側の候補にも載っていればその cosine を採り、無ければ観測値から外す。
    body_by_id = {cid: cosine for cid, cosine, _score, _text in (body_entries or [])}
    excluded = pseudo_ids or set()
    raw_by_id = {
        cid: score for cid, score, _ in merged_raw if cid not in excluded
    }
    scores = [
        body_by_id[cid] if cid in body_by_id else raw_by_id[cid]
        for cid, _, _ in final_sources
        if cid in body_by_id or cid in raw_by_id
    ]
    return max(scores) if scores else None


def _store_of(chunk_id: str) -> str:
    """id からストア名を判定する (c_16 §5.5 の ``<store>:`` 接頭辞用)。

    corpus のチャンク id は ``CartridgeManager.search_detailed`` が
    ``"<package_id>:<evidence_id>"`` 形式で返す。episodic のノート id は
    ``ev_…`` で ``":"`` を含まない。
    """
    return "corpus" if ":" in chunk_id else "episodic"


def _corpus_versions(cartridge_mgr) -> dict[str, str]:
    """ロード中の corpus パッケージ id → 版 (参照 ``corpus:<pkg>@<ver>:<id>`` 用)。"""
    loaded = getattr(cartridge_mgr, "loaded", None) if cartridge_mgr is not None else None
    if not isinstance(loaded, dict):
        return {}
    out: dict[str, str] = {}
    for package_id, package in loaded.items():
        version = getattr(getattr(package, "meta", None), "version", None)
        if isinstance(version, str):
            out[str(package_id)] = version
    return out


def evidence_ids_of(
    sources: list[tuple[str, float, str]],
    corpus_versions: dict[str, str] | None = None,
) -> list[str]:
    """注入した ``sources`` を Evidence の参照形式へ写す (c_16 §5.5 / c_05 §0.5.5)。

    会話ノートは ``episodic:<evidence_id>``。corpus は **必ず**
    ``corpus:<package_id>@<version>:<evidence_id>`` — チャンク id は内容由来で
    版を跨いで同じ値になり、code_node はルートを跨ぐと同じ id になるので、
    パッケージと版を持たない bare な id 参照は禁止 (c_05 §0.5.5)。検索結果の
    chunk id は ``<package_id>:<evidence_id>`` なので、版は ``corpus_versions``
    (ロード中のパッケージ) から引く。引けなければ版は空にする (``@:``)。
    """
    versions = corpus_versions or {}
    out: list[str] = []
    seen: set[str] = set()
    for chunk_id, _score, _text in sources:
        store = _store_of(chunk_id)
        if store == "corpus":
            package_id, _, evidence_id = chunk_id.partition(":")
            tagged = f"corpus:{package_id}@{versions.get(package_id, '')}:{evidence_id}"
        else:
            tagged = f"{store}:{chunk_id}"
        if tagged not in seen:
            seen.add(tagged)
            out.append(tagged)
    return out


def _claim_key_of(text: str) -> str:
    """本文から ``claim_key`` を作る (c_16 §3.4)。

    レコード側の ``claim_key`` はストアの中にあり、``search()`` の戻りには
    載らない (載せると層ごとに別の鍵の作り方が生まれる)。ここは注入直前の
    畳み込みなので、**同じ正規化関数** を本文へ掛けて鍵を作り直す。
    """
    return compute_claim_key(text)


def _merge_results(*result_lists: list[StoreEntry]) -> list[StoreEntry]:
    """ストア横断でスコア降順にマージし、id と ``claim_key`` で 1 件へ畳む。

    c_16 §7.2 のとおり **1 本の順位式** (``cos × freshness × confidence ×
    store_prior``、各ストアの ``search()`` が計算済み) をそのまま降順に並べる。
    層内正規化 (``rag.score_normalization``) も RRF も使わない — 層ごとに違う
    スケールを揃える必要がそもそも無くなった。

    畳み込みは c_16 §7.3 の意味で **ストアをまたぐ**: 同じ言明が episodic の
    ノートと corpus のチャンクの両方に載っていれば、スコアの高い方だけを残す。
    ストア内の ``claim_key`` 畳み込み (``ranking.collapse``) は同じストアの中
    でしか効かないため、ここが横断の唯一の関所になる。
    """
    ordered: list[StoreEntry] = []
    for results in result_lists:
        ordered.extend(results)
    ordered.sort(key=lambda item: -item[2])

    seen_ids: set[str] = set()
    seen_claims: set[str] = set()
    merged: list[StoreEntry] = []
    collapsed = 0
    for entry in ordered:
        chunk_id, _cosine, _score, text = entry
        if chunk_id in seen_ids:
            continue
        claim = _claim_key_of(text) if text else ""
        if claim and claim in seen_claims:
            collapsed += 1
            continue
        seen_ids.add(chunk_id)
        if claim:
            seen_claims.add(claim)
        merged.append(entry)
    if collapsed:
        logger.info(
            "Merge: collapsed %d duplicate claim(s) across stores", collapsed,
        )
    return merged


def _expansion_seed(query: str) -> int:
    """クエリ文字列から決定論的な乱数シードを作る (純粋関数)。"""
    return zlib.crc32((query or "").encode("utf-8")) & 0xFFFFFFFF


async def _expand_and_research(
    query: str,
    query_vec: np.ndarray,
    working_mem,
    episodic,
    top_k: int,
    noise_sigma: float = 0.05,
) -> list[StoreEntry]:
    """クエリベクトル摂動による簡易再検索（LLM なし）。

    直近の会話コンテキストがある場合に限り、クエリベクトルを微小ノイズで
    摂動させて近傍を再取得する。会話コンテキストの内容自体は (キーワード抽出
    等で) 検索条件に反映しない簡易拡張。

    摂動は **クエリ文字列から導いたシード** で生成する (:func:`_expansion_seed`)。
    チャット応答パスの補助判定は決定論層で行う不変則 (CLAUDE.md §6 #1) に
    従い、同じクエリには同じ拡張結果を返す (未シードの乱数だと再現も
    テストもできない)。
    """
    # 直近 3 ターンに非空の発話があるときだけ拡張する。
    has_context = any(
        (turn.get("content", "") or "").strip()
        for turn in working_mem.get_context()[-3:]
    )
    if not has_context or episodic is None:
        return []

    # クエリベクトルを少し摂動させて再検索（簡易的な拡張、決定論）
    rng = np.random.default_rng(_expansion_seed(query))
    noise = rng.standard_normal(query_vec.shape).astype(np.float32) * noise_sigma
    expanded_vec = query_vec + noise
    norm = np.linalg.norm(expanded_vec)
    if norm > 0:
        expanded_vec = expanded_vec / norm

    loop = asyncio.get_running_loop()
    try:
        hits = await run_in_executor_with_context(
            loop, _search_executor,
            lambda: episodic.search(query, expanded_vec, top_k),
        )
    except asyncio.CancelledError:
        raise
    except (RuntimeError, ValueError, TypeError, OSError) as e:
        logger.warning("Expanded search failed: %s", e)
        return []
    return [(hit.id, hit.cosine, hit.score, hit.text) for hit in hits]
