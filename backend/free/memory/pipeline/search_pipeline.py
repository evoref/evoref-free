"""統合検索パイプライン: 3層メモリ + Self-RAG（asyncio.gather 並列検索）"""

from __future__ import annotations

import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from backend.log_config import get_logger
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
    from backend.free.memory.views.mem import MemFactView
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


async def _search_stm_layer(
    short_term, query_vec: np.ndarray, stm_top_k: int,
    drop_past_answers: bool = False,
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
) -> list[tuple[str, float, str]]:
    """Layer 3 長期記憶検索 (< 5ms)。LTM 未設定 / 失敗時は空リスト。

    注入用に組み立てた文字列が回り込んだチャンクはここで落とす
    (:data:`_INJECTED_OUTPUT_MARKERS`)。
    """
    if long_term is None:
        return []
    loop = asyncio.get_running_loop()
    try:
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
        logger.debug("Step 3 LTM: %d results", len(results))
        return results
    except asyncio.CancelledError:
        raise
    except (RuntimeError, ValueError, TypeError, OSError) as e:
        logger.warning("LTM search failed: %s", e)
        return []


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


def _resolve_quality_judge_cfg(rag_cfg: dict) -> dict:
    """``rag.self_rag.quality_judge`` セクションを安全に取り出す

    セクション欠落時はデフォルト値相当の dict を返し、呼び出し側で
    キーの存在チェックを不要にする。旧 ``rag.aux_judge_enabled``
    フラット構造は廃止済み (後方互換なし)。
    """
    self_rag_cfg = rag_cfg.get("self_rag") or {}
    aj_cfg = self_rag_cfg.get("quality_judge") or {}
    return {
        "enabled": bool(aj_cfg.get("enabled", True)),
        "max_top_score": float(aj_cfg.get("max_top_score", 0.75)),
        "max_per_session": int(aj_cfg.get("max_per_session", 5)),
        "max_per_query": int(aj_cfg.get("max_per_query", 1)),
        "only_when_quality": list(aj_cfg.get("only_when_quality", ["medium"])),
    }


def _resolve_necessity_judge_cfg(rag_cfg: dict) -> dict:
    """``rag.self_rag.necessity_judge`` セクションを安全に取り出す

    検索必要性のハイブリッド判定 (ルール → uncertain 時のみ補助タスク) を
    制御する。``only_when_quality`` は ``JudgeUsageTracker`` 互換の
    キーで、本機能の quality ラベルは ``"uncertain"`` のみを使う。

    既定は無効 (``SelfRagNecessityJudgeConfig`` の説明を参照)。実測で
    ゲート (3,407ms) がゲート対象の retrieval (21ms) の 165 倍高くついていた。
    """
    self_rag_cfg = rag_cfg.get("self_rag") or {}
    an_cfg = self_rag_cfg.get("necessity_judge") or {}
    return {
        "enabled": bool(an_cfg.get("enabled", False)),
        "max_per_session": int(an_cfg.get("max_per_session", 10)),
        "max_per_query": int(an_cfg.get("max_per_query", 1)),
        "only_when_quality": list(an_cfg.get("only_when_quality", ["uncertain"])),
        "timeout_s": float(an_cfg.get("timeout_s", 5.0)),
        "context_turns": int(an_cfg.get("context_turns", 2)),
    }


def _resolve_necessity_recall_cfg(rag_cfg: dict) -> dict:
    """``rag.self_rag.necessity_recall`` セクションを安全に取り出す

    embedding 決定論的リコール (``rag_judge_recall.try_recall_necessity``) の
    閾値設定。url/executable_command recall と同型。
    """
    cfg = ((rag_cfg.get("self_rag") or {}).get("necessity_recall")) or {}
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "topk": int(cfg.get("topk", 5)),
        "min_score": float(cfg.get("min_score", 0.75)),
        "min_record_score": float(cfg.get("min_record_score", 0.65)),
        "ttl_days": int(cfg.get("ttl_days", 14)),
    }


def _resolve_quality_recall_cfg(rag_cfg: dict) -> dict:
    """``rag.self_rag.quality_recall`` セクションを安全に取り出す (necessity_recall と対称)。"""
    cfg = ((rag_cfg.get("self_rag") or {}).get("quality_recall")) or {}
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "topk": int(cfg.get("topk", 5)),
        "min_score": float(cfg.get("min_score", 0.75)),
        "min_record_score": float(cfg.get("min_record_score", 0.65)),
        "ttl_days": int(cfg.get("ttl_days", 14)),
    }


async def _maybe_recall_quality(
    quality: str,
    query: str,
    rag_cfg: dict,
    *,
    debug_logger: "DebugLogger | None" = None,
    mem_view: "MemFactView | None" = None,
    query_vec: np.ndarray | None = None,
) -> str:
    """marginal 境界の品質ラベルを embedding 決定論的リコールで上書きする。

    対象は ``rag.self_rag.quality_judge.only_when_quality`` (既定 ``medium``)
    の帯のみ。リコールが当たらなければルールベース判定をそのまま返す。

    LLM による再判定は撤去済み。実測 (2026-08-12、``op="quality_judge"`` 244 件)
    では実際に結果が変わる medium→low は全機会の 2.0% で、格下げの取り戻しより
    誤格下げで検索結果を落とすリスクの方が大きいと判断した (docs/c_14 §1.2.6)。
    """
    aj_cfg = _resolve_quality_judge_cfg(rag_cfg)
    if quality not in aj_cfg["only_when_quality"]:
        return quality

    quality_recall_cfg = _resolve_quality_recall_cfg(rag_cfg)
    if not (
        quality_recall_cfg["enabled"]
        and mem_view is not None
        and query_vec is not None
    ):
        return quality

    from backend.free.memory.pipeline.rag_judge_recall import try_recall_quality
    recalled = await try_recall_quality(query_vec, mem_view, quality_recall_cfg)
    if recalled is None:
        return quality

    label, recall_sim = recalled
    logger.debug("Step 5b quality recall: %s (sim=%.3f)", label, recall_sim)
    if debug_logger is not None:
        debug_logger.log_decision(
            decision_point="self_rag_judge_path",
            chosen="embedding_recall",
            candidates=["rule_based", "embedding_recall"],
            reason="embedding_recall_matched",
            context={
                "rule_based_quality": quality,
                "recalled_quality": label,
                "similarity": recall_sim,
                "query_preview": query[:80],
            },
            scope="request",
        )
    return label


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
#: p90 0.238 で、関連ペアは 0.467 だった。0.15 はノイズの中央値より上、
#: 関連の下限より十分下に位置する。
_RELATIVE_FLOOR_ABSOLUTE_MIN = 0.15


def _resolve_keep_floor(
    rag_cfg: dict,
    thresholds: QualityThresholds,
    top_raw_score: float = 0.0,
) -> float:
    """``low_quality_keep_floor`` を ``threshold_mode`` に従って解決する。

    ``auto`` かつ較正が効いているときは較正済み ``relevance`` と同じ棒を使う
    (「関連性の棒を越えたチャンクは集計判定が low でも残す」)。``manual`` /
    較正なしでは config の静的値を使うが、そのままだと埋め込みスケールが
    変わったときに到達不能になるため、``top_raw_score`` との相対でも抑える
    (:data:`_RELATIVE_FLOOR_RATIO`)。config が ``0.0`` (= 従来どおりクエリ単位の
    全件破棄) を明示している場合は較正より優先して尊重する。
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
        return float(thresholds.relevance)
    if top_raw_score > 0.0:
        relaxed = min(configured, top_raw_score * _RELATIVE_FLOOR_RATIO)
        return max(relaxed, min(configured, _RELATIVE_FLOOR_ABSOLUTE_MIN))
    return configured


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
    mem_view: "MemFactView | None" = None,
) -> SearchResult:
    """統合検索パイプライン: Self-RAG + 3層メモリ

    判定はルールベース + ベクトル演算で完結する (LLM 呼び出しゼロ)。
    marginal 帯 (既定 ``medium``) の品質ラベルだけは embedding 決定論的
    リコールで上書きしうる。
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

    # Step 1: Self-RAG 検索必要性判定 (rule + uncertain 時のみ embedding リコール)
    necessity_judge = RetrievalNecessityJudge()
    full_context = working_mem.get_context()
    context_count = len(full_context)
    # 末尾ターンは現在のユーザクエリ自身 (chat service が search 前に
    # WorkingMemory.add_turn 済) なので、補助タスクプロンプトの "最新のクエリ"
    # と重複しないよう除外する。末尾が user role でない (テスト経路等) なら
    # 全件をそのまま渡す。
    if full_context and full_context[-1].get("role") == "user":
        recent_context = full_context[:-1]
    else:
        recent_context = full_context
    necessity_cfg = _resolve_necessity_judge_cfg(rag_cfg)
    necessity_recall_cfg = _resolve_necessity_recall_cfg(rag_cfg)

    # Step 1 全体を計測する。ここが未計測だったため「遅い検索の 89.7% が
    # 内訳不明」という状態が続き、実際の支配要因 (necessity aux の往復) が
    # 見えていなかった (2026-08-01 プロファイリング)。
    if timer is not None:
        timer.start("necessity_ms")
    rule_necessity = necessity_judge.judge_rule_only(query, context_count)
    necessity: str | None = None
    if rule_necessity != "uncertain":
        necessity = rule_necessity
    elif (
        # リコールは necessity 判定そのもののキャッシュなので、判定を無効に
        # したらキャッシュも効かせない。切り離すと「判定は止めたのに過去の
        # 判定結果が skip を出し続ける」状態になり、無効化の意図が骨抜きになる。
        # しかもキュレータは閾値 (min_record_score) を超えたものだけ残すため、
        # 「retrieve は不要だった」と採点されやすい現行ルーブリックの下では
        # skip 判定が選択的に蓄積される (2026-08-01 実測: retrieve 判定の
        # 採点は 0.0 で捨てられた)。無効時に生かすと退行方向にしか働かない。
        necessity_cfg["enabled"]
        and necessity_recall_cfg["enabled"]
        and mem_view is not None
    ):
        from backend.free.memory.pipeline.rag_judge_recall import try_recall_necessity
        recalled = await try_recall_necessity(query_vec, mem_view, necessity_recall_cfg)
        if recalled is not None:
            necessity, recall_sim = recalled
            logger.info(
                "Necessity recall: %s (sim=%.3f, query=%r)",
                necessity, recall_sim, query[:50],
            )
            if debug_logger is not None:
                debug_logger.log_decision(
                    decision_point="self_rag_necessity_path",
                    chosen=necessity,
                    candidates=["retrieve", "fetch", "skip"],
                    reason="embedding_recall_matched",
                    context={"similarity": recall_sim},
                    scope="request",
                )
    if necessity is None:
        necessity = necessity_judge.judge(query, context_count=context_count)
    if timer is not None:
        timer.stop("necessity_ms")
    logger.debug("Step 1 necessity: %s (context_count=%d)", necessity, context_count)
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
    stm_pair, ltm_results, cart_results = await asyncio.gather(
        _search_stm_layer(short_term, query_vec, stm_top_k, drop_past_answers),
        _search_ltm_layer(long_term, query_vec, fetch_k, drop_past_answers),
        _search_cartridge_layer(
            cartridge_mgr, query_vec, fetch_k, timeout_ms=cart_timeout_ms,
        ),
    )
    # STM は順位付け用 (combined) とゲート用 (素の cosine) の 2 系統を返す。
    stm_results, stm_gate_results = stm_pair
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
        merged = _merge_results(stm_results, ltm_results, cart_results)
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
        merged_raw = await ChunkContentGate(
            gate_cfg, debug_logger=debug_logger,
        ).filter(
            query, merged_raw, mode,
            aux_client=aux_client,
            tracker=judge_tracker,
            session_id=session_id,
        )
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
    if timer is not None:
        timer.start("content_gate_ms")
    try:
        quality = await _maybe_recall_quality(
            quality, query, rag_cfg,
            debug_logger=debug_logger,
            mem_view=mem_view,
            query_vec=query_vec,
        )
    finally:
        if timer is not None:
            timer.stop("content_gate_ms")

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
                kept=[(cid, s) for cid, s, _ in merged_raw if s >= floor],
                rejected=[(cid, s) for cid, s, _ in merged_raw if s < floor],
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
