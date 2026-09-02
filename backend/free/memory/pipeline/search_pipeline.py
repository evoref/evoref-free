"""統合検索パイプライン: 3層メモリ + Self-RAG（asyncio.gather 並列検索）"""

from __future__ import annotations

import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from backend.log_config import get_logger
from backend.free.memory.corrections import corrections_by_target
from backend.trace_context import run_in_executor_with_context
from backend.free.constants import (
    SEARCH_HISTORY_CURRENT_SESSION_HEADER,
    SEARCH_HISTORY_NO_RESULTS_PREFIX,
    SEARCH_HISTORY_OTHER_SESSIONS_HEADER,
)
from backend.free.rag.chunk_content_gate import ChunkContentGate, GateConfig
from backend.free.rag.self_rag_judge import (
    QualityThresholds,
    RetrievalNecessityJudge,
    RetrievalQualityJudge,
)

if TYPE_CHECKING:

    from backend.debug_logger import DebugLogger
    from backend.free.core.policy_interpreter import PolicyInterpreter
    from backend.free.core.stage_timer import StageTimer
    from backend.free.rag.judge_usage_tracker import JudgeUsageTracker
    from backend.free.rag.lazy_contextual import LazyContextualPrefixService

logger = get_logger("memory.search_pipeline")

# STM / LTM / カートリッジ用スレッドプール（設計書 4.10.1）
_search_executor = ThreadPoolExecutor(max_workers=3)


@dataclass
class SearchResult:
    """統合検索の結果"""
    sources: list[tuple[str, float, str]] = field(default_factory=list)
    quality: str = "low"
    from_memory: bool = False
    skipped: bool = False
    #: ``sources`` に採用されたチャンクの **生スコア** (cosine スケール) の最大値。
    #: ``sources`` 側のスコアは ``score_normalization`` 適用後で、``minmax`` では
    #: 層内 max が定義上 1.0 に張り付くため、検索品質の観測値としては使えない
    #: (Level 0 の ``rag_top1_score`` が 21 ターン全てで厳密に 1.0 になり、
    #: それを目的関数にする embed_instruction / policy `search` ドメインの
    #: fitness が定数化していた)。品質判定・gate と同じ「判定は生スコア」の
    #: 不変則 (docs/f_01 §8.3) に合わせるための観測用フィールド。
    top_raw_score: float | None = None


def _resolve_fetch_multiplier(cfg: dict) -> int:
    """候補拡張倍率を解決する。

    ``rag.fetch_multiplier`` (既定 1 = 拡張なし) を用いて LTM / カートリッジの
    取得件数を ``top_k * N`` へ広げ、「広く取って絞る」第1段として候補プールを
    確保する (STM は ``stm_top_k`` 固定で拡張対象外)。値は [1, 5] にクランプする。
    """
    rag_cfg = cfg.get("rag") or {}
    multiplier = rag_cfg.get("fetch_multiplier", 1)
    try:
        multiplier = int(multiplier)
    except (TypeError, ValueError):
        multiplier = 1
    return max(1, min(5, multiplier))


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


#: ``compress_turn(style="summary")`` の圧縮マークと末尾の元文字数。
_SUMMARY_MARK = "[要約] "
_SUMMARY_TAIL_RE = re.compile(r"…（\d+文字）\s*$")

#: 「同じ質問の繰り返し」と見なす最小文字数。短い相槌 (「ありがとう」「はい」) は
#: 何度でも出るので、これを繰り返し扱いにすると assistant ノートを不当に落とす。
_REPEAT_MIN_CHARS = 12


def _normalize_utterance(text: str) -> str:
    """発話の同一判定用の正規化 (空白除去 + 圧縮マーク剥がし)。"""
    body = (text or "").strip()
    if body.startswith(_SUMMARY_MARK):
        body = body[len(_SUMMARY_MARK):]
    body = _SUMMARY_TAIL_RE.sub("", body)
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


def query_repeats_a_stored_turn(short_term, query: str) -> bool:
    """STM 全体を走査して「前にも同じことを聞いたか」を判定する。

    層検索は ``asyncio.gather`` で並列に走るため、STM の検索結果を待ってから
    LTM の扱いを決めることはできない。判定に必要なのはベクトル検索ではなく
    保存済みノートの文字列一致だけなので、層を起動する **前** に同期で済ませる
    (ノートは上限 100 件程度、実測 1ms 未満)。
    """
    if short_term is None or not query:
        return False
    try:
        notes = list(getattr(short_term, "notes", {}).values())
    except Exception:
        return False
    return _is_repeat_of_a_stored_turn(query, notes)




#: 訂正済みノートに付ける注記。SemMem 側の ``(訂正後の記録)``
#: (``pipeline.injector._render_fact``) と対になる。
_SUPERSEDED_MARK = "（訂正済み）"


def attach_superseding_corrections(
    short_term, sources: list[tuple[str, float, str]],
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
    """
    if short_term is None or not sources:
        return sources
    try:
        notes = list(getattr(short_term, "notes", {}).values())
    except Exception:
        return sources
    kept_ids = {cid for cid, _, _ in sources}
    links = [
        (target_id, getattr(corr, "id", ""), getattr(corr, "content", "") or "")
        for target_id, corr in corrections_by_target(notes).items()
        if target_id in kept_ids
    ]
    if not links:
        return sources

    marked = {sid for sid, _, _ in links}
    present = {cid for cid, _, _ in sources}
    out: list[tuple[str, float, str]] = []
    for chunk_id, score, text in sources:
        if chunk_id in marked and not (text or "").rstrip().endswith(_SUPERSEDED_MARK):
            text = f"{(text or '').rstrip()}{_SUPERSEDED_MARK}"
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

async def _search_stm_layer(
    short_term, query_vec: np.ndarray, stm_top_k: int,
    drop_past_answers: bool = False,
    retired_note_ids: set[str] | None = None,
) -> tuple[list[tuple[str, float, str]], list[tuple[str, float, str]]]:
    """Layer 2 短期記憶検索 (< 1ms)。``(順位付け用, ゲート用)`` を返す。

    STM のスコアは ``類似度 × 0.6 + LightMem × 0.4`` (+ pin 加点) で、これは
    順位付けには有用だがゲート判定には使えない。品質判定 / floor / content gate
    は **cosine スケール前提**で閾値が決まっており、LTM は素の cosine を返す。
    STM だけ別スケールを流すと閾値が層ごとに違う意味になるため、ゲート用には
    素の cosine を載せた同順の別リストを返す (docs/f_02 §品質ゲート)。

    失敗時は両方とも空リスト。
    """
    loop = asyncio.get_running_loop()
    try:
        hits = await run_in_executor_with_context(
            loop, _search_executor,
            short_term.retrieve_top_k_detailed, query_vec, stm_top_k,
        )
        # 同じ質問を前にもしていた場合だけ、assistant ノート (= その質問への
        # 前回の回答) を落とす。詳細は _is_repeat_of_a_stored_turn を参照。
        if drop_past_answers:
            before = len(hits)
            hits = [
                h for h in hits
                if getattr(h[0], "source", "user") == "user"
            ]
            if len(hits) != before:
                logger.info(
                    "Step 2 STM: this query repeats an earlier turn; dropped "
                    "%d assistant note(s) so the previous answer is not "
                    "handed back as reference material",
                    before - len(hits),
                )
        # 値が supersede された発話は現在値として参照させない。
        # MemoryInjector 側 ([関連する記憶]) は別途落としているが、STM は
        # **2 経路で注入される** — ここを塞がないと同じノートが [参考情報]
        # として出る (2026-08-27 の実機検証: injector が 3 件落とした同じ
        # ターンで「データベース管理者」が返り続けた)。
        if retired_note_ids:
            before = len(hits)
            hits = [h for h in hits if h[0].id not in retired_note_ids]
            if len(hits) != before:
                logger.info(
                    "Step 2 STM: dropped %d note(s) whose value was "
                    "superseded by a later correction",
                    before - len(hits),
                )
        ranked = [(note.id, combined, note.content) for note, combined, _ in hits]
        gated = [(note.id, relevance, note.content) for note, _, relevance in hits]
        logger.debug(
            "Step 2 STM: %d hits, combined=[%s], relevance=[%s]",
            len(ranked),
            ", ".join(f"{s:.3f}" for _, s, _ in ranked),
            ", ".join(f"{s:.3f}" for _, s, _ in gated),
        )
        return ranked, gated
    except asyncio.CancelledError:
        raise
    except (RuntimeError, ValueError, TypeError) as e:
        logger.warning("STM search failed: %s", e)
        return [], []


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


def _drop_past_answers(
    long_term, results: list[tuple[str, float, str]],
) -> list[tuple[str, float, str]]:
    """assistant 由来のチャンクを落とす (繰り返し質問のときだけ呼ぶ)。

    発話者は ``absorb_from_short_term`` が ``note_meta`` に残しているので、
    ``LongTermMemory.chunk_source`` で読める。素性は 2026-08-09 に「検索側で
    どう扱うかは観測できるようになってから実測で決める」として保存され、
    読み側では使われていなかった。

    実測 (2026-08-16): STM 84 件のうち assistant 39 件 (46%)。技術質問では上位
    5 件中 3〜4 件が過去の回答で、それらは **有用な参照** だった (asyncio の
    コード例など)。したがって一律には落とさず、「前にも同じことを聞いていた」
    ターンに限る — その場合に返るのは同じ質問への前回の回答であり、モデルは
    それを丸写しする。
    """
    kept: list[tuple[str, float, str]] = []
    for cid, score, text in results:
        if long_term.chunk_source(cid) == "assistant":
            continue
        kept.append((cid, score, text))
    if len(kept) != len(results):
        logger.info(
            "Step 3 LTM: this query repeats an earlier turn; dropped %d "
            "assistant chunk(s) so the previous answer is not handed back "
            "as reference material",
            len(results) - len(kept),
        )
    return kept


async def _search_ltm_layer(
    long_term, query_vec: np.ndarray, top_k: int,
    drop_past_answers: bool = False,
    query_text: str = "",
) -> tuple[list[tuple[str, float, str]], frozenset[str]]:
    """Layer 3 長期記憶検索 (< 5ms)。LTM 未設定 / 失敗時は空リスト。

    ``search_hybrid`` があれば **ベクトル + BM25 の RRF** で候補を集める。
    スコアは常に素のコサインなので、後段の品質判定 / フロアの閾値は不変。

    注入用に組み立てた文字列が回り込んだチャンクはここで落とす
    (:data:`_INJECTED_OUTPUT_MARKERS`)。

    Returns:
        ``(結果リスト, 語彙アンカーの chunk_id 集合)``。
    """
    if long_term is None:
        return [], frozenset()
    loop = asyncio.get_running_loop()
    anchors: frozenset[str] = frozenset()
    try:
        hybrid = getattr(long_term, "search_hybrid", None)
        if hybrid is not None and query_text:
            results, anchors = await run_in_executor_with_context(
                loop, _search_executor, hybrid, query_vec, query_text, top_k,
            )
        else:
            results = await run_in_executor_with_context(
                loop, _search_executor, long_term.search, query_vec, top_k,
            )
        if drop_past_answers:
            results = _drop_past_answers(long_term, results)
        kept = [r for r in results if not _is_injected_output(r[2])]
        if len(kept) != len(results):
            logger.info(
                "Step 3 LTM: dropped %d chunk(s) that are our own injected "
                "output (search_history render) from episodic memory",
                len(results) - len(kept),
            )
        trimmed = [(cid, score, _trim_file_payload(text)) for cid, score, text in kept]
        n_trimmed = sum(1 for a, b in zip(kept, trimmed) if a[2] != b[2])
        if n_trimmed:
            logger.info(
                "Step 3 LTM: trimmed file payload from %d legacy episodic "
                "chunk(s) (ingested before the summariser)", n_trimmed,
            )
        # エピソード記憶の内部識別子 (episode / conversation / outcome) を落とす。
        # 識別子だけで情報が残らないチャンクは丸ごと捨てる。
        sanitized: list[tuple[str, float, str]] = []
        n_dropped = n_cleaned = 0
        for cid, score, text in trimmed:
            cleaned = _sanitize_episode_chunk(text)
            if cleaned is None:
                n_dropped += 1
                continue
            if cleaned != text:
                n_cleaned += 1
            sanitized.append((cid, score, cleaned))
        if n_cleaned or n_dropped:
            logger.info(
                "Step 3 LTM: stripped internal ids from %d episodic chunk(s), "
                "dropped %d that carried nothing but ids",
                n_cleaned, n_dropped,
            )
        results = sanitized
        kept_ids = {cid for cid, _, _ in results}
        logger.debug("Step 3 LTM: %d results", len(results))
        return results, frozenset(anchors & kept_ids)
    except asyncio.CancelledError:
        raise
    except (RuntimeError, ValueError, TypeError, OSError) as e:
        logger.warning("LTM search failed: %s", e)
        return [], frozenset()


async def _search_cartridge_layer(
    cartridge_mgr, query_vec: np.ndarray, top_k: int,
    timeout_ms: int = 0,
) -> list[tuple[str, float, str]]:
    """カートリッジ検索。マネージャ未設定 / 失敗時は空リスト。

    ``timeout_ms`` が 1 以上の場合、検索全体にタイムアウトを適用する
。タイムアウト時は空リストを返し、チャット応答を
    止めないようにする。
    """
    if cartridge_mgr is None or not hasattr(cartridge_mgr, "search"):
        return []
    loop = asyncio.get_running_loop()
    try:
        coro = run_in_executor_with_context(
            loop, _search_executor, cartridge_mgr.search, query_vec, top_k,
        )
        if timeout_ms > 0:
            results = await asyncio.wait_for(coro, timeout=timeout_ms / 1000.0)
        else:
            results = await coro
        logger.debug("Step 3b cartridge: %d results", len(results))
        return results
    except asyncio.TimeoutError:
        logger.warning(
            "Cartridge search timed out after %d ms (L3); "
            "returning empty results to keep chat responsive",
            timeout_ms,
        )
        return []
    except asyncio.CancelledError:
        raise
    except (RuntimeError, ValueError, TypeError, OSError) as e:
        logger.warning("Cartridge search failed: %s", e)
        return []


async def _try_quality_expansion(
    quality: str,
    merged: list[tuple[str, float, str]],
    quality_judge: RetrievalQualityJudge,
    query: str,
    query_vec: np.ndarray,
    working_mem,
    long_term,
    top_k: int,
    noise_sigma: float,
) -> tuple[list[tuple[str, float, str]], str]:
    """品質不足 (low) 時にクエリ拡張で再検索を試みる。`(merged, quality)` を返す。"""
    if quality != "low" or not merged:
        return merged, quality
    logger.debug("Quality low — attempting query expansion")
    expanded_results = await _expand_and_research(
        query, query_vec, working_mem, long_term, top_k,
        noise_sigma=noise_sigma,
    )
    if not expanded_results:
        return merged, quality
    merged = _merge_results(merged, expanded_results, [])
    quality = quality_judge.judge(merged)
    logger.debug("After expansion: %d results, quality=%s", len(merged), quality)
    return merged, quality


def _log_memory_search_state(
    debug_logger,
    context_count: int,
    stm_results: list,
    ltm_results: list,
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
            "stm_notes": len(stm_results),
            "ltm_vectors": len(ltm_results),
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
#: 実測 (2026-08-23 ライブ監査セット 1): 較正が未成立の状態 (新規 local/) で
#: 実効フロアは 0.15〜0.22 まで下がり、「√2 を小数第 6 位まで教えてください」
#: に大阪出張の予定が cos 0.1735 で、「税抜き 12,800 円の消費税」に同じ予定が
#: 0.1863 / 0.1656 で ``[参考情報]`` として注入された。同セッションで本当に
#: 関連するチャンク (猫の名前の問いに対する猫の記録) は 0.344 / 0.3377 で、
#: ノイズ帯とは分離していた。
#:
#: そこで宣言どおりノイズ分布の p90 に置く。関連ペア (0.467、実測の一致は
#: 0.34 以上) は十分上に残る。
_RELATIVE_FLOOR_ABSOLUTE_MIN = 0.24

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


def _resolve_relative_keep_ratio(rag_cfg: dict) -> float:
    """``relative_keep_ratio`` を [0, 1] にクランプして返す。"""
    self_rag = rag_cfg.get("self_rag") or {}
    try:
        ratio = float(self_rag.get("relative_keep_ratio", _RELATIVE_KEEP_RATIO))
    except (TypeError, ValueError):
        ratio = _RELATIVE_KEEP_RATIO
    return max(0.0, min(1.0, ratio))


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
       (:data:`_RELATIVE_FLOOR_RATIO`、**緩める方向**)。
    2. **相対の棒** — そのターンの ``top_raw_score`` の
       :data:`_RELATIVE_KEEP_RATIO` 倍 (**絞る方向**)。絶対の棒は
       「ノイズより上か」しか見ておらず、較正時は定義上ノイズの 5% が通る。

    最終的な floor は 1 と 2 の **高い方**。top1 は定義上 2 を越えるため、
    相対の棒で結果集合が空になることはない。config が ``0.0`` (= 従来どおり
    クエリ単位の全件破棄) を明示している場合は較正より優先して尊重する。
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
    if calibrated:
        absolute = float(thresholds.relevance)
    elif top_raw_score > 0.0:
        relaxed = min(configured, top_raw_score * _RELATIVE_FLOOR_RATIO)
        absolute = max(relaxed, min(configured, _RELATIVE_FLOOR_ABSOLUTE_MIN))
    else:
        absolute = configured

    # 絶対の棒を通ったうえで、そのターンの最良証拠と比べて弱いものを落とす。
    # top1 は定義上必ず越えるので結果集合が空になることはない。
    ratio = _resolve_relative_keep_ratio(rag_cfg)
    if ratio > 0.0 and top_raw_score > 0.0:
        return max(absolute, top_raw_score * ratio)
    return absolute


async def unified_search(
    query: str,
    query_vec: np.ndarray,
    working_mem,
    short_term,
    long_term,
    cartridge_mgr=None,
    config: dict | None = None,
    aux_client=None,
    debug_logger=None,
    mode: str = "chat",
    policy: PolicyInterpreter | None = None,
    timer: "StageTimer | None" = None,
    semmem_stats: dict | None = None,
    lazy_contextual: "LazyContextualPrefixService | None" = None,
    *,
    session_id: str = "default",
    judge_tracker: "JudgeUsageTracker | None" = None,
    retired_note_ids: set[str] | None = None,
) -> SearchResult:
    """統合検索パイプライン: Self-RAG + 3層メモリ

    **SemMem はここでは検索しない。** 融合するのは STM / LTM / カートリッジの
    3 層だけで、``semmem_stats`` はログ用の受け渡しにすぎない。SemMem が
    プロンプトへ載る経路は ``chat_service.build_semmem_injection`` →
    :class:`~backend.free.memory.pipeline.injector.MemoryInjector` の **完全に
    別系統** (全件 + 関連度ゲート + Tier パッキング) で、RRF も top_k 融合も
    通らない。2 系統に分かれているのは、SemMem が「属性スロットの現在値」を
    扱うのに対し RAG は「チャンクの関連度」を扱うからで、順位付けとして混ぜる
    対象ではない (2026-09-01 監査 F12 で設計書 §8 の記述を実装へ合わせた)。

    層をまたぐ融合 (``_merge_results``) は **RRF ではない** — スコア降順の
    マージ + 先勝ち dedup (``score_normalization`` を指定した場合は層内正規化を
    挟む)。RRF は LTM 層の **内部** (``search_hybrid``) にのみある。

    判定はルールベース + ベクトル演算で完結する (LLM 呼び出しゼロ)。
    STM / LTM / カートリッジ検索を asyncio.gather で並列実行する。
    ``rag.fetch_multiplier`` が 2 以上の場合、LTM / カートリッジの取得件数を
    `top_k * N` に拡張する (STM は `stm_top_k` 固定)。

    Args:
        session_id: content gate の発火カウンタキー。
            ``run_search_pipeline`` が ``WorkingMemory.session_id`` か
            フロントエンド指定 session_id を渡す。
        judge_tracker: content gate (create モード) のセッション単位
            カウンタ。``None`` なら上限を評価しない (テスト経路互換)。
    """
    cfg = config or {}
    rag_cfg = cfg.get("rag", {})
    top_k, stm_top_k, noise_sigma = _resolve_search_params(policy, rag_cfg, mode)
    multiplier = _resolve_fetch_multiplier(cfg)
    fetch_k = top_k * multiplier
    logger.debug(
        "unified_search: query=%r, top_k=%d, fetch_k=%d (mult=%d)",
        query[:80], top_k, fetch_k, multiplier,
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
    # 末尾ターンは現在のユーザクエリ自身 (chat service が search 前に
    # WorkingMemory.add_turn 済) なので、補助タスクプロンプトの "最新のクエリ"
    # と重複しないよう除外する。末尾が user role でない (テスト経路等) なら
    # 全件をそのまま渡す。
    if full_context and full_context[-1].get("role") == "user":
        recent_context = full_context[:-1]
    else:
        recent_context = full_context
    # Step 1 全体を計測する。ここが未計測だったため「遅い検索の 89.7% が
    # 内訳不明」という状態が続き、実際の支配要因が見えていなかった
    # (2026-08-01 プロファイリング)。
    if timer is not None:
        timer.start("necessity_ms")
    necessity = necessity_judge.judge(
        query, context_count=context_count, window_complete=window_complete,
    )
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

    # Step 2-3: STM / LTM / カートリッジを asyncio.gather で並列実行（設計書 4.10.1）
    # fetch_multiplier >= 2 のときは fetch_k 件を取得し、後段で top_k に絞る
    cart_timeout_ms = int(rag_cfg.get("cartridge_search_timeout_ms", 3000))
    # 取得そのものの所要。necessity ゲートの費用対効果を測るために分けて計る
    # (ゲートが取得より高くつく状態を検出できるようにする)。
    if timer is not None:
        timer.start("retrieval_ms")
    # 「前にも同じことを聞いたか」は層検索の結果ではなく保存済みノートの文字列
    # 一致で決まるので、gather の前に同期で確定させる (層は並列に走るため、
    # STM の結果を待ってから LTM の扱いを決めることはできない)。
    drop_past_answers = query_repeats_a_stored_turn(short_term, query)
    if drop_past_answers:
        logger.info(
            "This query repeats an earlier turn; past answers will be kept out "
            "of the reference block (query=%r)", query[:50],
        )
    stm_pair, ltm_pair, cart_results = await asyncio.gather(
        _search_stm_layer(
            short_term, query_vec, stm_top_k, drop_past_answers,
            retired_note_ids=retired_note_ids,
        ),
        _search_ltm_layer(
            long_term, query_vec, fetch_k, drop_past_answers, query,
        ),
        _search_cartridge_layer(
            cartridge_mgr, query_vec, fetch_k, timeout_ms=cart_timeout_ms,
        ),
    )
    # STM は順位付け用 (combined) とゲート用 (素の cosine) の 2 系統を返す。
    stm_results, stm_gate_results = stm_pair
    # LTM は結果と「語彙アンカー」(クエリの希少語を実際に含む chunk_id) を返す。
    ltm_results, lexical_anchors = ltm_pair
    if timer is not None:
        timer.stop("retrieval_ms")

    # Lazy Contextual Retrieval — LTM hit について on-demand で
    # プレフィックス生成を非同期タスクとして起動する。fire-and-forget なので
    # 本 retrieval のレイテンシには影響せず、次回以降の同一 chunk ヒット時に
    # contextual_text が使えるようになる。
    if lazy_contextual is not None and lazy_contextual.is_active and ltm_results:
        ltm_chunk_ids = [cid for cid, _, _ in ltm_results]
        asyncio.create_task(
            lazy_contextual.on_retrieval_hits(ltm_chunk_ids),
            name=f"lazy_contextual_hits[{len(ltm_chunk_ids)}]",
        )

    # Step 4: 結果マージ。merged_raw は生スコア (品質判定 / gate / クエリ拡張用)、
    # merged は最終順位付け用。score_normalization でクロスレイヤ正規化を適用し、
    # STM/LTM/cartridge の異種スコアスケールの歪みを吸収する。
    # merged_raw は STM のゲート用スコア (素の cosine) を使う。LTM / cartridge は
    # 元から cosine なので、これで 3 層すべてが同じスケールに揃う。merged 側は
    # 従来どおり STM の combined (LightMem / pin 込み) で順位付けする。
    norm = rag_cfg.get("score_normalization", "none")
    merged_raw = _merge_results(stm_gate_results, ltm_results, cart_results)
    if norm == "none":
        # 正規化なしのときは **3 層とも素の cosine** で並べる。
        #
        # 以前は STM だけ combined (cosine*0.6 + LightMem*0.4 + pin 加点) を
        # 順位付けに使っていた。層ごとにスケールが違う値を 1 本の降順ソートへ
        # 混ぜると、LightMem 項のぶん STM が LTM / カートリッジより系統的に
        # 上へ来る。``merged[:top_k]`` は層をまたいだ順位で切るので、これは
        # 「新しい会話ノートが、より関連する長期記憶を押しのける」形で効く。
        #
        # LightMem の役割は失われない。STM の内部で **どのノートを候補に
        # するか** (``retrieve_top_k_detailed`` が combined 順に stm_top_k 件)
        # を決めるのが本来の仕事で、層をまたいだ関連性の比較は素の cosine の
        # 担当。ゲート (品質判定 / floor / content gate) が既に素の cosine を
        # 使っているのと同じ理由。
        merged = _merge_results(stm_gate_results, ltm_results, cart_results)
    else:
        merged = _merge_results(
            stm_results, ltm_results, cart_results, normalization=norm,
        )
    logger.debug(
        "Step 4 merge: %d unique results after dedup (normalization=%s)",
        len(merged_raw), norm,
    )

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
            )
        finally:
            if timer is not None:
                timer.stop("content_gate_ms")
        kept = {cid for cid, _, _ in merged_raw}
        merged = [t for t in merged if t[0] in kept]
        logger.debug("Step 4.5 content gate: %d results after prune", len(merged_raw))

    # Step 5: Self-RAG 品質判定 (ベクトル閾値、< 0.1ms)
    # 判定は常に生スコア (merged_raw) に対して行う。品質3閾値は cosine 分布前提の
    # ため、正規化スコアを渡すと閾値の意味が崩れる。
    thresholds = QualityThresholds.from_config(rag_cfg)
    # decision.jsonl に記録 (decision_point=``self_rag_judge_path``)
    quality_judge = RetrievalQualityJudge(thresholds, debug_logger=debug_logger)
    quality = quality_judge.judge(merged_raw)
    logger.debug("Step 5 quality: %s", quality)

    # Step 6: 品質不足時のクエリ拡張フォールバック (生スコアで再検索)。
    # 拡張が発火した場合 (quality=="low" の稀ケース) は結果集合が変わるため、
    # 正規化側 merged も生順 (拡張結果込み) にフォールバックして確実に届ける。
    expanded, quality = await _try_quality_expansion(
        quality, merged_raw, quality_judge, query, query_vec,
        working_mem, long_term, top_k, noise_sigma,
    )
    if expanded is not merged_raw:
        merged_raw = expanded
        merged = expanded

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
    floor = _resolve_keep_floor(
        rag_cfg, thresholds,
        top_raw_score=max((s for _, s, _ in merged_raw), default=0.0),
    )
    if floor > 0.0:
        kept_ids = {cid for cid, score, _ in merged_raw if score >= floor}
        # 語彙アンカーの免除。
        #
        # フロアはコサインの棒なので、**密ベクトルが原理的に苦手なもの**
        # (型番・パス・エラーコード・固有名詞のような literal) は、ユーザーが
        # 名指ししていても低いコサインのまま落ちる。BM25 側の df から
        # 「コーパスの 1% 未満にしか現れないトークン」を希少語と定義し、
        # そのトークンを **実際に含む** チャンクだけをフロアから免除する。
        #
        # 相対バーではなく df を使うのは、相対バーだと top1 が定義上必ず越えて
        # しまうため (注入側で同じ理由から相対フロアを採らなかった。
        # ``MemoryInjector._resolve_relevance_thresholds`` のコメント参照)。
        # df 比はコーパス規模にも埋め込みモデルにも依存しないので、静的な絶対
        # 閾値が差し替えで到達不能になる事故を繰り返さない。
        anchored = {cid for cid in lexical_anchors if cid not in kept_ids}
        if anchored:
            kept_ids |= anchored
            logger.info(
                "Relevance floor: %d chunk(s) exempted as lexical anchors "
                "(query names a rare literal they contain) for query: %s",
                len(anchored), query[:50],
            )
        passed = [t for t in merged if t[0] in kept_ids]
        if len(passed) != len(merged):
            logger.info(
                "Relevance floor: %d/%d chunks passed floor=%.2f "
                "(quality=%s) for query: %s",
                len(passed), len(merged), floor, quality, query[:50],
            )
        if debug_logger is not None:
            debug_logger.log_rag_selection(
                query=query,
                quality=quality,
                floor=floor,
                kept=[(cid, s) for cid, s, _ in merged_raw if cid in kept_ids],
                rejected=[
                    (cid, s) for cid, s, _ in merged_raw if cid not in kept_ids
                ],
            )
        merged = passed
        # 公平性保証 (Step 7.5) は cart_results から未代表カートリッジを
        # 強制注入するため、フロアを通していないチャンクが裏口から戻る。
        # カートリッジ側にも同じ棒を掛ける (関連性の棒は経路で変わらない)。
        cart_results = [c for c in cart_results if c[0] in kept_ids]
    elif quality == "low":
        # フロア無効 (0.0) の構成では従来どおり「low はクエリ単位で全件破棄」。
        logger.info(
            "Search results discarded (quality=low, floor disabled) "
            "for query: %s", query[:50],
        )
        merged = []

    if not merged:
        _log_memory_search_state(
            debug_logger, context_count, stm_results, ltm_results,
            semmem_stats=semmem_stats,
        )
        return SearchResult(
            sources=[],
            quality=quality,
            from_memory=bool(stm_results),
        )

    # Step 7: 最終順位付け (merged はスコア降順) から top_k 件を採用
    final_sources = merged[:top_k]

    # Step 7.5: カートリッジ公平性保証
    # ロード済みカートリッジが上位選別で完全に欠落するのを防ぐ
    final_sources = _ensure_cartridge_fairness(
        final_sources, cart_results, top_k,
    )

    # Step 7.55: 語彙アンカーは席を争わせない。
    #
    # ``_merge_results`` は最終的にスコア (= cosine) 降順で並べ直すため、LTM 側の
    # RRF 順は「候補集合に入る/入らない」にしか効かない。密ベクトルが苦手な
    # literal (型番・パス・エラーコード) は cosine が低いままなので、フロアを
    # 免除しても ``merged[:top_k]`` で切られて結局届かない。訂正の随伴注入
    # (Step 7.6) と同じ形で、**top_k を切った後に**足す。
    final_sources = _attach_lexical_anchors(
        final_sources, merged, lexical_anchors,
    )

    # Step 7.6: 採用ノートが後続の訂正で上書きされているなら、訂正も一緒に出す。
    # top_k で切った **後** に足す — 訂正は席を争う候補ではなく随伴情報であり、
    # floor / top_k のどちらで落ちても「訂正前の値だけが残る」状態になる。
    final_sources = attach_superseding_corrections(short_term, final_sources)

    logger.info(
        "Search completed: %d results, quality=%s, from_memory=%s",
        len(final_sources), quality, bool(stm_results),
    )
    _log_memory_search_state(
        debug_logger, context_count, stm_results, ltm_results,
        semmem_stats=semmem_stats,
    )

    return SearchResult(
        sources=final_sources,
        quality=quality,
        from_memory=bool(stm_results),
        top_raw_score=_top_raw_score(final_sources, merged_raw),
    )


#: ``_attach_lexical_anchors`` が随伴注入する上限。コンテキストを膨らませない
#: ための保守的な値 (context rot: 意味的に近いが無関係な文脈ほど有害)。
MAX_LEXICAL_ANCHOR_ATTACHMENTS: int = 2


def _attach_lexical_anchors(
    final_sources: list[tuple[str, float, str]],
    merged: list[tuple[str, float, str]],
    lexical_anchors: frozenset[str],
    limit: int = MAX_LEXICAL_ANCHOR_ATTACHMENTS,
) -> list[tuple[str, float, str]]:
    """語彙アンカーのうち top_k から漏れたものを末尾へ足す (純粋関数)。

    ユーザーが希少な literal を名指ししているのに、それを含むチャンクが
    cosine 順位だけで落ちる状態を防ぐ。上限付きなので注入量は増えすぎない。
    """
    if not lexical_anchors:
        return final_sources
    present = {cid for cid, _, _ in final_sources}
    extra = [
        entry for entry in merged
        if entry[0] in lexical_anchors and entry[0] not in present
    ][:limit]
    if not extra:
        return final_sources
    logger.info(
        "Attached %d lexical anchor chunk(s) that fell outside top_k", len(extra),
    )
    return final_sources + extra


def _top_raw_score(
    final_sources: list[tuple[str, float, str]],
    merged_raw: list[tuple[str, float, str]],
) -> float | None:
    """採用チャンクの生スコア (cosine スケール) の最大値を返す。

    `final_sources` のスコアは `score_normalization` 適用後なので観測値に使えない
    (`minmax` では先頭が定義上 1.0)。`merged_raw` 側の生スコアを chunk_id で
    引き直す。gate/拡張で `merged_raw` から落ちた chunk (カートリッジ公平性で
    裏口から戻った等) は引けないので単に除外する。1 件も引けなければ `None`。
    """
    if not final_sources:
        return None
    raw_by_id = {cid: score for cid, score, _ in merged_raw}
    scores = [raw_by_id[cid] for cid, _, _ in final_sources if cid in raw_by_id]
    return max(scores) if scores else None


def _cartridge_id_of(chunk_id: str) -> str | None:
    """カートリッジ由来のチャンク ID から cart_id 部分を抽出する。

    `cartridge_manager.search` は ``"<cart_id>:<original_chunk_id>"`` 形式で
    chunk_id を返す。STM/LTM 由来の chunk_id は通常 ":" を含まないため、
    プレフィックス分離で十分判別可能。
    """
    if ":" not in chunk_id:
        return None
    return chunk_id.split(":", 1)[0]


def _ensure_cartridge_fairness(
    final: list[tuple[str, float, str]],
    cart_results: list[tuple[str, float, str]],
    top_k: int,
) -> list[tuple[str, float, str]]:
    """ロード済みカートリッジの公平な代表を保証する

    `cart_results` (cartridge_manager.search の生出力) に含まれる各
    カートリッジのうち、最終結果 `final` から完全に欠落しているものがあれば、
    その入力チャンクの先頭 (= ラウンドロビン優先順位の最高) を最終結果に
    強制的に含める。

    グローバルスコアソートが特定カートリッジを完全に除外する
    挙動を補正することが目的。複数カートリッジが存在しないケースでは
    `final` をそのまま返す (no-op)。

    既に `final` の長さが `top_k` 未満なら末尾追加し、満杯の場合は最低
    スコアの非カートリッジ要素または別カートリッジで重複代表されている
    要素を差し替える。

    注入エントリのスコアは `final` 内の最低スコアへ揃える。`final` は
    score_normalization 適用済み [0,1] であり、
    `cart_results` の生スコア (cosine*priority、無上限) を持ち込むと下流の
    SalienceRanker min-max が歪むため。強制注入 = 最下位相当の意味付け。
    """
    if not cart_results or not final:
        return final

    floor_score = min(s for _, s, _ in final)

    input_carts: list[str] = []
    seen_input: set[str] = set()
    cart_top_chunk: dict[str, tuple[str, float, str]] = {}
    for entry in cart_results:
        cid = _cartridge_id_of(entry[0])
        if cid is None:
            continue
        if cid not in seen_input:
            seen_input.add(cid)
            input_carts.append(cid)
            cart_top_chunk[cid] = entry

    if len(input_carts) < 2:
        # カートリッジが 1 つ以下なら公平性問題は起きない
        return final

    output_carts = {
        c
        for cid, _, _ in final
        for c in (_cartridge_id_of(cid),)
        if c is not None
    }

    missing = [c for c in input_carts if c not in output_carts]
    if not missing:
        return final

    # 既存の chunk_id 集合 (重複混入防止)
    existing_ids = {cid for cid, _, _ in final}
    result = list(final)

    for cart_id in missing:
        raw = cart_top_chunk[cart_id]
        if raw[0] in existing_ids:
            continue
        chunk = (raw[0], floor_score, raw[2])
        if len(result) < top_k:
            result.append(chunk)
            existing_ids.add(chunk[0])
            output_carts.add(cart_id)
            continue
        # 満杯: 差し替え対象を選ぶ
        # 優先度1: 非カートリッジ (STM/LTM) のうち最低スコア
        # 優先度2: 既に他で代表されているカートリッジの最低スコア
        replace_idx = _find_replaceable_index(result, output_carts)
        if replace_idx is None:
            continue
        result[replace_idx] = chunk
        existing_ids.add(chunk[0])
        output_carts.add(cart_id)

    logger.debug(
        "cartridge fairness: input_carts=%s, missing=%s, applied=%d",
        input_carts, missing, len(missing),
    )
    return result


def _find_replaceable_index(
    final: list[tuple[str, float, str]],
    output_carts: set[str],  # noqa: ARG001
) -> int | None:
    """差し替え可能な要素のインデックスを返す

    優先度:
        1. 非カートリッジ要素 (STM/LTM 由来) のうち最終位置 (≒最低スコア)
        2. 複数カートリッジが代表されている場合の重複代表側の最終位置

    どれも該当しない場合 (1 カートリッジしかなく満杯) は None。
    """
    # 優先度 1: 非カートリッジ要素を末尾から探す
    for i in range(len(final) - 1, -1, -1):
        if _cartridge_id_of(final[i][0]) is None:
            return i

    # 優先度 2: 同じカートリッジが複数回代表されているケースで末尾を差し替え
    cart_counts: dict[str, int] = {}
    for cid, _, _ in final:
        c = _cartridge_id_of(cid)
        if c is not None:
            cart_counts[c] = cart_counts.get(c, 0) + 1

    for i in range(len(final) - 1, -1, -1):
        c = _cartridge_id_of(final[i][0])
        if c is not None and cart_counts.get(c, 0) > 1:
            cart_counts[c] -= 1
            return i

    return None


def _normalize_layer(
    layer: list[tuple[str, float, str]],
    method: str,
) -> list[tuple[str, float, str]]:
    """1 層 (STM/LTM/cartridge) 内でスコアを正規化する。

    ``minmax``: 層内 min-max で [0, 1] へ写像。レンジ 0 (1 件 / 全同値) は
        0 に潰さず一律 1.0 とする (単独層を層間で不利にしないため)。
    ``rank``: 1 位=1.0 … 最下位=1/n の線形ランク減衰。n=1 は 1.0。
    ``none`` / 未知: 入力をそのまま返す (no-op)。空層は空リスト。

    combined スコアをスカラとして正規化するため、STM の blended
    (cosine*0.6+lightmem*0.4) や cartridge の cosine*priority の内訳は壊さない
    (層内相対順位は不変、層間スケールのみ揃う)。cartridge priority は無上限だが
    層内 min-max が層内相対順位を保ったまま [0,1] に収めるため歪みを吸収する。
    """
    n = len(layer)
    if n == 0:
        return []
    if method == "minmax":
        scores = [s for _, s, _ in layer]
        lo = min(scores)
        hi = max(scores)
        rng = hi - lo
        if rng <= 1e-12:
            return [(cid, 1.0, text) for cid, _, text in layer]
        return [(cid, (s - lo) / rng, text) for cid, s, text in layer]
    if method == "rank":
        order = sorted(range(n), key=lambda i: -layer[i][1])
        norm = [0.0] * n
        for rank, idx in enumerate(order):
            norm[idx] = (n - rank) / n
        return [(layer[i][0], norm[i], layer[i][2]) for i in range(n)]
    return layer


def _merge_results(
    *result_lists: list[tuple[str, float, str]],
    normalization: str = "none",
) -> list[tuple[str, float, str]]:
    """結果をマージしてスコア降順でソート（重複排除）。

    ``normalization`` が ``minmax`` / ``rank`` のとき、各 result_list を 1 層と
    みなして層内でスコアを正規化してからマージし、STM/LTM/cartridge の異種
    スコアスケールを吸収する。``none`` (既定) は従来どおり生スコアでマージする。
    重複は最初に出現した層の要素を採用する (先勝ち、スコアに非依存=現状一致)。
    """
    seen: set[str] = set()
    merged: list[tuple[str, float, str]] = []

    for results in result_lists:
        for chunk_id, score, text in _normalize_layer(results, normalization):
            if chunk_id not in seen:
                seen.add(chunk_id)
                merged.append((chunk_id, score, text))

    merged.sort(key=lambda x: -x[1])
    return merged


async def _expand_and_research(
    query: str,  # noqa: ARG001
    query_vec: np.ndarray,
    working_mem,
    long_term,
    top_k: int,
    noise_sigma: float = 0.05,
) -> list[tuple[str, float, str]]:
    """クエリベクトル摂動による簡易再検索（LLM なし）。

    直近の会話コンテキストがある場合に限り、クエリベクトルを微小ノイズで
    摂動させて近傍を再取得する。会話コンテキストの内容自体は (キーワード抽出
    等で) 検索条件に反映しない簡易拡張。
    """
    # 直近 3 ターンに非空の発話があるときだけ拡張する。
    has_context = any(
        (turn.get("content", "") or "").strip()
        for turn in working_mem.get_context()[-3:]
    )
    if not has_context or long_term is None:
        return []

    # クエリベクトルを少し摂動させて再検索（簡易的な拡張）
    noise = np.random.randn(*query_vec.shape).astype(np.float32) * noise_sigma
    expanded_vec = query_vec + noise
    norm = np.linalg.norm(expanded_vec)
    if norm > 0:
        expanded_vec = expanded_vec / norm

    loop = asyncio.get_running_loop()
    try:
        return await run_in_executor_with_context(
            loop, _search_executor, long_term.search, expanded_vec, top_k,
        )
    except asyncio.CancelledError:
        raise
    except (RuntimeError, ValueError, TypeError, OSError) as e:
        logger.warning("Expanded search failed: %s", e)
        return []
