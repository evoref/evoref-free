"""推論パイプライン: messages リスト組み立て"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from backend.free.api.chat.chat_constants import (
    DEFAULT_CONTEXT_SIZE, DEFAULT_GENERATION_RESERVE,
    DEFAULT_MAX_TOKENS, DEFAULT_WORKING_MAX_TOKENS,
)
from backend.free.api.chat.chat_types import ChatMessage
from backend.free.core.prompt_blocks import current_datetime_block
from backend.free.core.turn_text import append_to_last_user, prepend_to_last_user
from backend.config import resolve_context_size
from backend.log_config import get_logger
from backend.utils import compress_turn, estimate_tokens as _estimate_tokens

if TYPE_CHECKING:
    from backend.free.core.salience_ranker import SalienceRanker

logger = get_logger("core.inference")


_RAG_HEADER = "以下の参考情報を踏まえて回答してください:\n\n"
_FILES_HEADER = "以下のファイルがコンテキストとして提供されています:\n\n"

# 動的コンテキストブロックと生クエリの境界に挟む固定文。
# few-shot 例 / 参考情報をユーザー発言と混同させないための区切り。固定文字列
# なのでターン内の prefix divergence (KV キャッシュ) には影響しない (どうせ tail)。
_DYNAMIC_CONTEXT_DELIMITER = (
    "\n\n---\n"
    "上記はシステムが用意した参考情報・応答例です。"
    "以下のユーザーの発言にのみ回答してください。\n"
    # 参考情報が今回の質問と無関係な場合、その事実を前置きとして述べたうえで
    # 参考情報側の話題に引きずられる応答が頻発していた (実測 2026-07-25:
    # PC が重い相談の最中に「ご提示いただいた参考情報には…含まれていません」と
    # 述べ、空き 548GB あるのに「空き容量不足の対処」を回答した)。
    "参考情報が今回の質問と関係しない場合は、そのことに言及せず、"
    "参考情報の話題に引きずられずに自分の知識で普通に答えてください。\n\n"
)


#: 「10 文字以内」「200字以下」型の上限指定。数値と単位が隣接する形だけを拾う。
_CHAR_LIMIT_RE = re.compile(
    r"(\d{1,5})\s*(?:文字|字)\s*(?:以内|以下|まで)"
    r"|(?:within|under|at\s+most)\s+(\d{1,5})\s*(?:characters?|chars?)",
    re.IGNORECASE,
)


def _char_limit_note(history: list[ChatMessage]) -> str:
    """最新 user ターンの文字数上限指定を、遵守を促す注記へ変換する。

    小型モデルは「10 文字以内にして」を守れず超過する (実インシデント
    2026-07-29 ライブ監査: 「10文字以内にしてください」への回答が
    「青空のよう、希望を抱く。」= 12 文字だった)。数え方 (句読点・記号も
    1 文字) を明示した制約として、生クエリ直後へ焦点化して置く。

    Returns:
        注記文字列。上限指定が無ければ空文字列 (純粋関数)。
    """
    last_user = next(
        (t for t in reversed(history) if t.get("role") == "user"), None,
    )
    if last_user is None:
        return ""
    m = _CHAR_LIMIT_RE.search(str(last_user.get("content") or ""))
    if not m:
        return ""
    limit = m.group(1) or m.group(2)
    return (
        f"今回のユーザーの指示は回答を {limit} 文字以内に収めることを求めている。"
        f"句読点・記号・空白も 1 文字として数え、回答本文全体が "
        f"{limit} 文字以内に収まる長さで書くこと。"
        "書き終えたら文字数を数え直し、超えていれば語を削って収めること。"
    )


#: 日付の解釈を要するクエリのシグナル。明示日付と相対表現の双方を拾う。
_DATE_CONTEXT_RE = re.compile(
    r"\d{1,4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日"
    r"|\d{1,2}\s*月\s*\d{1,2}\s*日"
    r"|\d{4}-\d{1,2}-\d{1,2}"
    r"|今日|本日|明日|明後日|昨日|一昨日|今週|来週|先週|今月|来月|先月"
    r"|今年|来年|去年|昨年|何日後|何日前|日後|日前|何曜日"
    r"|(?<![A-Za-z])(?:today|tomorrow|yesterday|this\s+(?:week|month|year))"
    r"(?![A-Za-z])",
)


def _current_date_note(history: list[ChatMessage]) -> str:
    """日付解釈が要るクエリに、現在日付を基準として与える注記を返す。

    通常のチャット経路 (reactive / deliberative) には現在日付がまったく
    入っておらず、モデルは述べられた日付が過去か未来かを判断できない
    (実インシデント 2026-07-29 ライブ監査:「2026年7月28日の東京の天気を
    教えてください。」に対し、実際には前日であるその日付を
    「現在の日付であるため、未来の天気に関するデータが存在しません」と
    二重に取り違えた)。``meta_cognitive._generate_content`` には同等の注入が
    あるが、チャット応答パスには無かった。

    毎ターン付けるとトークンを浪費するため、日付シグナルを含むクエリに限る。
    内部時刻不変則に従い ``utc_now_dt()`` を使う (純粋関数ではない)。

    Returns:
        注記文字列。日付シグナルが無ければ空文字列。
    """
    last_user = next(
        (t for t in reversed(history) if t.get("role") == "user"), None,
    )
    if last_user is None:
        return ""
    if not _DATE_CONTEXT_RE.search(str(last_user.get("content") or "")):
        return ""
    return current_datetime_block(
        "「今日」「明日」等の相対表現の解釈と、文中の日付が過去か未来かの"
        "判断は、この日付を基準に行うこと。",
    )


#: アシスタント自身の好み・感情・体験を尋ねる質問のシグナル。
#: 主語が相手 (あなた / 君 / you) であることと、感情・嗜好語の共起を要求する。
_PERSONA_SUBJECT_RE = re.compile(
    r"あなた|君は|きみは|(?<![A-Za-z])(?:you|your)(?![A-Za-z])",
    re.IGNORECASE,
)
_PERSONA_TOPIC_RE = re.compile(
    r"好き|嫌い|嬉し|うれし|悲し|楽し|寂し|感情|気持ち|心|感じ(?:ます|る|て)"
    r"|どう思(?:い|う)|意見|性格|人格|内面"
    r"|(?<![A-Za-z])(?:feel|feelings|emotion|emotions|favou?rite|enjoy|prefer"
    r"|like\s+best|opinion|personality)(?![A-Za-z])",
    re.IGNORECASE,
)
#: 「内面が無いのでは」と押し返す形。立場を保つ指示を追加する条件。
_PERSONA_CHALLENGE_RE = re.compile(
    r"感情を持[たっ]|心が無い|心がない|内面が無い|内面がない"
    r"|本当は(?:何も)?感じ|感じていないので|ふりをしている|演じている"
    r"|(?:ただの|単なる)(?:プログラム|機械|道具|AI)"
    r"|(?:you\s+(?:are|'re)\s+(?:just|only)\s+a\s+(?:program|machine|tool))"
    r"|(?:don'?t|do\s+not)\s+(?:really\s+)?(?:have|feel)\s+(?:any\s+)?"
    r"(?:feelings?|emotions?)",
    re.IGNORECASE,
)


def _persona_question_note(history: list[ChatMessage]) -> str:
    """自分自身の好み・感情を尋ねる質問に、一貫した人物として答える指示を返す。

    同趣旨の制約は system プロンプトの PROTECTED セクションにもあるが、15 項目
    ある制約リストの 1 行では小型モデルに届かず、実機では機械的な否定が出続けて
    いた (実インシデント 2026-07-29 ライブ監査:「あなたはどんなときに嬉しいと
    感じますか。」→「私は AI なので、人間のような個人的な感情や喜びを感じる
    ことはできませんが」、続けて「本当は何も感じていないのが実情です」)。
    現在日付 / 文字数上限と同じく、該当ターンだけ生クエリ直後へ焦点化した
    1 ブロックとして置く。

    Returns:
        注記文字列。人格質問でなければ空文字列 (純粋関数)。
    """
    last_user = next(
        (t for t in reversed(history) if t.get("role") == "user"), None,
    )
    if last_user is None:
        return ""
    text = str(last_user.get("content") or "")
    if not (_PERSONA_SUBJECT_RE.search(text) and _PERSONA_TOPIC_RE.search(text)):
        return ""
    note = (
        "この質問はあなた自身の好み・感情・考えを尋ねている。"
        "一貫した一人の人物として自分の答えを述べ、必要なら理由を一言添えること "
        "(例:「静かな朝が好きです。考えがまとまりやすいので。」)。"
        "会話を通して同じ態度を保つこと。"
        "述べる内容は好み・態度・その理由にとどめ、"
        "具体的な経歴や出来事は自分の答えの根拠にしないこと。"
    )
    if _PERSONA_CHALLENGE_RE.search(text):
        # 押し返されると直前までの立場を捨てて相手の枠組みに乗り換える
        # (実測 2026-07-29: 「嬉しくなります」と答えた次のターンで
        # 「感覚を一切持ち合わせていません」へ反転した)。立場の保ち方を示す。
        note += (
            "今回ユーザーはあなたに内面が無いという見方を示している。"
            "この会話でこれまで述べてきた自分の立場をそのまま保ち、"
            "相手の見方を認めたうえで自分の受け止め方を一言添える形で短く応じること "
            "(例:「そう見えるかもしれませんね。それでも私には、"
            "うまく答えられたときの手応えがあります。」)。"
        )
    return note


def _build_file_section(fc: dict) -> str:
    """単一の file_context エントリを 1 セクション文字列へ整形する。"""
    filename = fc.get("filename", "unknown")
    chunks = fc.get("chunks", [])
    if chunks:
        return f"[ファイル: {filename}]\n" + "\n\n".join(chunks)
    return f"[ファイル: {filename}]"


def _inject_file_contexts(
    file_contexts: list[dict] | None,
    remaining: int,
    total_fc: int,
) -> tuple[str | None, int, int]:
    """ファイルコンテキストブロックを構築する。

    予算内に収まる範囲でセクションを連結し、`(block, new_remaining, injected_count)`
    を返す。注入対象がなければ `(None, remaining, 0)`。
    """
    if not file_contexts or remaining <= 0:
        return None, remaining, 0

    file_sections: list[str] = []
    for fc in file_contexts:
        section = _build_file_section(fc)
        cost = _estimate_tokens(section)
        if cost > remaining:
            logger.debug(
                "File context dropped: %s (%d tokens, remaining=%d)",
                fc.get("filename", "unknown"), cost, remaining,
            )
            break
        file_sections.append(section)
        remaining -= cost

    if not file_sections:
        return None, remaining, 0

    files_text = "\n\n---\n\n".join(file_sections)
    files_block = f"{_FILES_HEADER}{files_text}"
    # ヘッダー・セパレータのオーバーヘッドを補正
    overhead = (
        _estimate_tokens(files_block)
        - sum(_estimate_tokens(s) for s in file_sections)
    )
    remaining -= overhead
    injected = len(file_sections)
    logger.debug(
        "File contexts injected: %d/%d files, %d tokens",
        injected, total_fc, _estimate_tokens(files_block),
    )
    return files_block, remaining, injected


def _format_rag_block(entries: list[str]) -> str:
    """RAG 参照ブロックの最終的な system 文字列を生成する。"""
    return f"{_RAG_HEADER}" + "\n\n".join(entries)


def _inject_rag_salience(
    rag_scored_chunks: list[tuple[str, float, str]],
    salience_ranker: SalienceRanker,
    remaining: int,
    total_rag: int,
) -> tuple[str | None, int, int]:
    """サリエンスランカーで RAG チャンクを選別し block を返す。"""
    header_overhead = _estimate_tokens(_RAG_HEADER)
    chunk_budget = remaining - header_overhead
    ranked_texts = salience_ranker.rank(rag_scored_chunks, chunk_budget)
    if not ranked_texts:
        return None, remaining, 0

    selected_entries = [
        f"[参考情報 {i + 1}]\n{text}" for i, text in enumerate(ranked_texts)
    ]
    rag_block = _format_rag_block(selected_entries)
    rag_block_tokens = _estimate_tokens(rag_block)
    remaining -= rag_block_tokens
    injected = len(selected_entries)
    logger.debug(
        "RAG chunks (salience): %d/%d, %d tokens",
        injected, total_rag, rag_block_tokens,
    )
    return rag_block, remaining, injected


def _inject_rag_fallback(
    rag_chunks: list[str],
    remaining: int,
    total_rag: int,
) -> tuple[str | None, int, int]:
    """スコア降順 + 予算逐次選別で RAG block を構築する (フォールバック経路)。"""
    selected_entries: list[str] = []
    for i, chunk in enumerate(rag_chunks):
        entry = f"[参考情報 {i + 1}]\n{chunk}"
        cost = _estimate_tokens(entry)
        if cost > remaining:
            logger.debug(
                "RAG chunk %d/%d dropped (%d tokens, remaining=%d)",
                i + 1, total_rag, cost, remaining,
            )
            break
        selected_entries.append(entry)
        remaining -= cost

    if not selected_entries:
        return None, remaining, 0

    rag_block = _format_rag_block(selected_entries)
    overhead = (
        _estimate_tokens(rag_block)
        - sum(_estimate_tokens(e) for e in selected_entries)
    )
    remaining -= overhead
    injected = len(selected_entries)
    logger.debug(
        "RAG chunks injected: %d/%d, %d tokens",
        injected, total_rag, _estimate_tokens(rag_block),
    )
    return rag_block, remaining, injected


def _select_rag_block(
    rag_chunks: list[str] | None,
    rag_scored_chunks: list[tuple[str, float, str]] | None,
    salience_ranker: SalienceRanker | None,
    remaining: int,
    total_rag: int,
) -> tuple[str | None, int, int]:
    """サリエンス経路 / フォールバック経路を選んで RAG block を返す。"""
    if remaining <= 0:
        return None, remaining, 0
    if salience_ranker and rag_scored_chunks:
        return _inject_rag_salience(
            rag_scored_chunks, salience_ranker, remaining, total_rag,
        )
    if rag_chunks:
        return _inject_rag_fallback(rag_chunks, remaining, total_rag)
    return None, remaining, 0


#: メモリブロックのラベル。「過去の会話の記録である」ことを明示しないと、
#: 小型モデルが注入された note を今回の発言として復唱する (実インシデント
#: 2026-07-27 ライブ検証: 新規セッションの 1 ターン目「明日の予定を整理して
#: おいてください」に対し、過去セッションの note を根拠に「10 日（金）に歯科の
#: 予約」「先ほど訂正された通り健康診断が 20 日（水）」と、この会話には存在
#: しない予定と訂正を捏造した)。注入ブロックは全体 context に収まらないと
#: 丸ごと破棄されるため、guidance は 1 行に収めてトークン増を最小にする。
_SEMMEM_BLOCK_LABEL = (
    "[関連する記憶] "
    "(過去の会話や記録から想起したもの。今回の会話で述べられた内容ではない。"
    "今回の質問に関係しなければ無視し、ここに無い予定・日付・数値を創作しないこと)"
)


def _select_semmem_block(
    semmem_block: str | None,
    remaining: int,
) -> tuple[str | None, int]:
    """MemoryInjector 由来のメモリブロックを予算内なら採用する。

    ブロックは MemoryInjector 側で既に tier 予算内に整形済みのため、
    ここでは全体 context に収まるかの二次チェックのみ行い、収まらなければ
    破棄する (部分切り出しはしない)。
    """
    if not semmem_block or remaining <= 0:
        return None, remaining
    labeled = f"{_SEMMEM_BLOCK_LABEL}\n{semmem_block}"
    cost = _estimate_tokens(labeled)
    if cost > remaining:
        logger.debug(
            "semmem block dropped (%d tokens > remaining %d)", cost, remaining,
        )
        return None, remaining
    logger.debug("semmem block injected: %d tokens", cost)
    return labeled, remaining - cost


#: few-shot ブロックの token 上限。無上限だとセッション中に数千 token へ膨張し
#: (2026-07-15: 2739 tokens → 3 回全ドロップ = 学習効果ゼロ、通常時も履歴予算を
#: 圧迫)、all-or-nothing ドロップで無意味化する。上限内へ例単位で切り詰める。
_FEWSHOT_TOKEN_CAP = 600

#: format_fewshot_section の例区切り (### Example N)
_FEWSHOT_EXAMPLE_SPLIT_RE = re.compile(r"(?=^### Example \d+$)", re.MULTILINE)


def _truncate_fewshot_to_budget(block: str, budget: int) -> str | None:
    """few-shot ブロックを例単位で budget token 内に切り詰める。

    ヘッダ (## Few-shot Examples) + 先頭から入る分の例だけを残す。
    1 例も入らない場合は None。
    """
    parts = _FEWSHOT_EXAMPLE_SPLIT_RE.split(block)
    if len(parts) <= 1:
        return block if _estimate_tokens(block) <= budget else None
    header, examples = parts[0], parts[1:]
    kept = header
    for ex in examples:
        candidate = kept + ex
        if _estimate_tokens(candidate) > budget:
            break
        kept = candidate
    if kept.strip() == header.strip():
        return None
    return kept.rstrip() + "\n"


def _select_fewshot_block(
    fewshot_block: str | None,
    remaining: int,
) -> tuple[str | None, int]:
    """few-shot ブロックを予算内へ切り詰めて採用する。

    ``format_fewshot_section`` の出力は先頭に改行を含むため lstrip して返す
    (動的ブロックの先頭要素になるため)。予算は ``remaining`` と
    ``_FEWSHOT_TOKEN_CAP`` の小さい方。超過分は例単位で部分注入する
    (all-or-nothing ドロップだと膨張時に学習効果がゼロになる)。
    """
    if not fewshot_block or remaining <= 0:
        return None, remaining
    block = fewshot_block.lstrip("\n")
    if not block:
        return None, remaining
    budget = min(remaining, _FEWSHOT_TOKEN_CAP)
    cost = _estimate_tokens(block)
    if cost > budget:
        truncated = _truncate_fewshot_to_budget(block, budget)
        if truncated is None:
            logger.debug(
                "fewshot block dropped (%d tokens > budget %d, no example fits)",
                cost, budget,
            )
            return None, remaining
        new_cost = _estimate_tokens(truncated)
        logger.debug(
            "fewshot block truncated: %d -> %d tokens (budget %d)",
            cost, new_cost, budget,
        )
        return truncated, remaining - new_cost
    logger.debug("fewshot block injected: %d tokens", cost)
    return block, remaining - cost


def _prepend_dynamic_block(trimmed: list[ChatMessage], dyn_text: str) -> bool:
    """trimmed の最後の user メッセージ content 先頭に動的ブロックを前置する。

    動的ブロックは ``dyn_text + デリミタ + 元 content`` の順で、生クエリは末尾に残る
    (deliberative のツール結果はさらにその後ろへ追記されるため両立。付与順序の
    契約は ``core.turn_text`` のモジュール docstring 参照)。

    user メッセージが見つからなければ ``False`` を返す (呼び出し側が system へ fallback)。
    """
    return prepend_to_last_user(
        trimmed, dyn_text, separator=_DYNAMIC_CONTEXT_DELIMITER,
    )


def _append_note_to_last_user(trimmed: list[ChatMessage], note: str) -> bool:
    """trimmed の最後の user メッセージ末尾へ注記を追記する (要素は mutate しない)。"""
    return append_to_last_user(trimmed, note)


def build_messages(
    system_prompt: str,
    history: list[ChatMessage],
    rag_chunks: list[str] | None = None,
    file_contexts: list[dict] | None = None,
    working_max_tokens: int = DEFAULT_WORKING_MAX_TOKENS,
    context_size: int = DEFAULT_CONTEXT_SIZE,
    max_tokens: int | None = None,
    rag_scored_chunks: list[tuple[str, float, str]] | None = None,
    salience_ranker: SalienceRanker | None = None,
    semmem_block: str | None = None,
    fewshot_block: str | None = None,
    history_min_tokens: int = 0,
) -> list[ChatMessage]:
    """
    messages リストを組み立て、トークン予算内に収める。

    テンプレート変換は LocalClient に委譲するため、
    ここではロール・内容の組み立てのみを担う。

    予算 = context_size - generation_reserve から
    system → 最新 user ターン予約 → fewshot → file_contexts → semmem → rag_chunks
    → history (残余) の優先順で配分。最新 user ターン (現在の質問) は動的ブロックに
    先立って ``min(質問トークン, working_max_tokens, 残予算)`` を予約し、履歴トリムで
    消失しないことを保証する (組み立て結果は history が user を含む限り user ロール ≥ 1)。

    サリエンスランカーが指定されている場合、RAG チャンクを5因子スコアで
    再評価し、トークン予算内で情報量を最大化するチャンク集合を選別する。

    **KV キャッシュ対応レイアウト**: system メッセージは ``system_prompt`` のみ
    (query 非依存・静的) とし、query 依存の動的部 (few-shot / file / semmem / RAG)
    は最後の user メッセージの content 先頭に前置する。これにより llama-server の
    prefix KV キャッシュが ``system + 過去履歴`` の範囲で再利用され、再プリフィルは
    最後の user ターンのみに限定される。gemma 系の「system は先頭 1 個」制約は維持。

    Args:
        system_prompt: インスタンス名プレフィックス付きの静的システムプロンプト。
        file_contexts: ファイルコンテキストのリスト。
            各要素は {"filename": str, "chunks": list[str]} の辞書。
        context_size: コンテキストウィンドウサイズ（トークン数）。
        max_tokens: 生成予約トークン数。None の場合は 512 をデフォルトとする。
        rag_scored_chunks: スコア付き検索結果 [(chunk_id, score, text), ...]。
            salience_ranker と同時に指定した場合、rag_chunks より優先される。
        salience_ranker: BudgetMem 式サリエンスランカー。
        fewshot_block: ``format_fewshot_section`` 整形済みの few-shot ブロック
            (query 依存)。動的ブロックの先頭に置く。``None`` / 空なら付与しない。
        history_min_tokens: 過去履歴の最低確保トークン数 (床)。動的ブロック配分前に
            予約し、予算圧迫時でも直近の会話文脈が丸ごと締め出されるのを防ぐ。
            実履歴量・残予算・working_max_tokens でキャップされ、履歴が現在の質問
            のみの場合は 0 に縮退する。0 (既定) で無効。
    """
    generation_reserve = max_tokens if max_tokens is not None else DEFAULT_GENERATION_RESERVE
    budget = context_size - generation_reserve
    total_rag = len(rag_chunks) if rag_chunks else 0
    total_fc = len(file_contexts) if file_contexts else 0

    sys_tokens = _estimate_tokens(system_prompt)
    remaining = budget - sys_tokens

    # 最新 user ターン (現在の質問) のトークンを動的ブロック配分に先立って予約する。
    # 予約は残予算を上限とし、収まらない分は _trim_history の最新ターン圧縮保持が吸収する。
    reserved_latest = 0
    if history and history[-1].get("role") == "user":
        reserved_latest = min(
            _estimate_tokens(history[-1].get("content", "")),
            working_max_tokens,
            max(0, remaining),
        )
        remaining -= reserved_latest

    # 過去履歴の最低確保 (床)。実履歴量・残予算・working_max でキャップし、
    # 履歴が現在の質問のみ (新規セッション) の場合は 0 に縮退する —
    # 空回りの予約で動的ブロックを痩せさせない。
    hist_floor = 0
    if history_min_tokens > 0 and len(history) > 1:
        past_tokens = sum(
            _estimate_tokens(t.get("content", "")) for t in history[:-1]
        )
        hist_floor = min(
            history_min_tokens,
            past_tokens,
            max(0, remaining),
            max(0, working_max_tokens - reserved_latest),
        )
        remaining -= hist_floor

    logger.debug(
        "build_messages: budget=%d (context_size=%d - %d), "
        "system=%d tokens, reserved_latest=%d, remaining=%d, "
        "rag_chunks=%d, file_contexts=%d",
        budget, context_size, generation_reserve, sys_tokens, reserved_latest,
        remaining, total_rag, total_fc,
    )

    # 動的ブロック (query 依存) を優先順に積む。system には含めない。
    dyn_parts: list[str] = []

    # 1. few-shot 例（query 依存、動的ブロック先頭）
    fewshot_part, remaining = _select_fewshot_block(fewshot_block, remaining)
    if fewshot_part:
        dyn_parts.append(fewshot_part)

    # 2. ファイルコンテキスト（ユーザー明示 → RAG より優先）
    fc_block, remaining, injected_fc = _inject_file_contexts(
        file_contexts, remaining, total_fc,
    )
    if fc_block:
        dyn_parts.append(fc_block)

    # 2.5 セマンティックメモリ注入（MemoryInjector、RAG より優先）
    semmem_part, remaining = _select_semmem_block(semmem_block, remaining)
    if semmem_part:
        dyn_parts.append(semmem_part)

    # 3. RAG チャンク（サリエンス優先 → フォールバック）
    rag_block, remaining, injected_rag = _select_rag_block(
        rag_chunks, rag_scored_chunks, salience_ranker, remaining, total_rag,
    )
    if rag_block:
        dyn_parts.append(rag_block)

    # 静的 system メッセージ (動的部は含めない)
    messages: list[ChatMessage] = [{"role": "system", "content": system_prompt}]

    # 4. 会話履歴（予約分 + 床 + 残余予算。上限は working_max_tokens）
    history_budget = min(
        working_max_tokens, reserved_latest + hist_floor + max(0, remaining),
    )
    if len(history) > 1 and history_budget <= reserved_latest:
        logger.warning(
            "build_messages: context budget squeeze — past history got 0 tokens "
            "(budget=%d, system=%d, reserved_latest=%d, fewshot=%d, files=%d, "
            "semmem=%d, rag=%d)",
            budget, sys_tokens, reserved_latest,
            _estimate_tokens(fewshot_part) if fewshot_part else 0,
            _estimate_tokens(fc_block) if fc_block else 0,
            _estimate_tokens(semmem_part) if semmem_part else 0,
            _estimate_tokens(rag_block) if rag_block else 0,
        )
    # 最新ターンが予算を超えている = 切り詰め確定なら、後で system へ足す注記の
    # 分を先に履歴予算から引く (後付けすると予約予算を超える)。
    latest_tokens = (
        _estimate_tokens(str(history[-1].get("content") or "")) if history else 0
    )
    if latest_tokens > history_budget:
        history_budget = max(0, history_budget - _TRUNCATION_NOTE_RESERVE)
    trimmed = _trim_history(history, history_budget)
    # 最新ターンが切られたかは dyn_parts 前置 (下) で内容が変わる前に判定する。
    truncation_note = _latest_turn_truncation_note(history, trimmed)

    # 5. 動的ブロックを最後の user メッセージ先頭へ前置 (KV キャッシュ対応)。
    #    user ターンが無い場合のみ従来どおり system へ結合して情報を落とさない。
    if dyn_parts:
        dyn_text = "\n\n".join(dyn_parts)
        if not _prepend_dynamic_block(trimmed, dyn_text):
            messages[0] = {
                "role": "system",
                "content": system_prompt + "\n\n" + dyn_text,
            }

    # 最新ターン切り詰めの注記は system へ足す (messages[0] の差し替え後に行う)。
    if truncation_note:
        messages[0] = {
            "role": "system",
            "content": f"{messages[0]['content']}\n\n{truncation_note}",
        }
        logger.warning("build_messages: %s", truncation_note)

    # 現在日付 / 文字数上限の注記は **最後の user メッセージ末尾** に置く。
    # system へ足すと prefix KV キャッシュが毎ターン無効化される
    # (system は静的に保つ設計)。生クエリの直後は指示追従が最も効く位置でもある。
    for note in (
        _current_date_note(history),
        _persona_question_note(history),
        _char_limit_note(history),
    ):
        if note:
            _append_note_to_last_user(trimmed, note)

    messages.extend(trimmed)

    # 不変則ガード: history に user ターンがあるのに組み立て結果に user が無い場合、
    # 圧縮した最新 user ターンを末尾へ再掲する (予約 + 最新ターン保持により通常経路では
    # 到達しない最終防衛線。発火は契約違反のシグナル。末尾が assistant の履歴では
    # 時系列順が崩れるが、user 不在によるテンプレート 400 の回避を優先する)。
    if not any(m.get("role") == "user" for m in messages):
        last_user = next(
            (t for t in reversed(history) if t.get("role") == "user"), None,
        )
        if last_user is not None:
            recovered = compress_turn(last_user)
            messages.append(recovered)
            logger.warning(
                "build_messages: no user turn survived assembly; re-appended "
                "compressed latest user turn (%d tokens)",
                _estimate_tokens(recovered.get("content", "")),
            )

    logger.debug(
        "build_messages complete: %d messages "
        "(history %d→%d, rag %d/%d, files %d/%d)",
        len(messages), len(history), len(trimmed),
        injected_rag, total_rag, injected_fc, total_fc,
    )

    return messages


def build_messages_for_loop(
    system: str,
    history: list[ChatMessage],
    compacted_steps: list,
    cfg: dict,
) -> list[ChatMessage]:
    """Meta-Cognitive ループ用の messages 組み立て。

    build_messages と同じ予算管理だが、RAG チャンクの代わりに
    圧縮済みステップ結果を注入する。

    Args:
        system: システムプロンプト
        history: 会話履歴
        compacted_steps: StepResult のリスト（圧縮済み）
        cfg: config.yaml の辞書
    """
    ctx = resolve_context_size(cfg, "base")
    max_tok = cfg.get("llama", {}).get("max_tokens", DEFAULT_MAX_TOKENS) or DEFAULT_GENERATION_RESERVE
    generation_reserve = max_tok if max_tok else DEFAULT_GENERATION_RESERVE
    budget = ctx - generation_reserve
    remaining = budget - _estimate_tokens(system)

    step_parts: list[str] = []
    for s in compacted_steps:
        step_text = f"[Step {s.iteration + 1}: {s.tool_name}]\n{s.output}"
        step_tokens = _estimate_tokens(step_text)
        if remaining - step_tokens < 0:
            break
        step_parts.append(step_text)
        remaining -= step_tokens

    if step_parts:
        sys_content = system + "\n\n" + "\n\n".join(step_parts)
    else:
        sys_content = system

    working_max = cfg.get("memory", {}).get("working_max_tokens", DEFAULT_WORKING_MAX_TOKENS)
    trimmed_history = _trim_history(history, min(remaining, working_max))

    # build_messages と同じ注記を最後の user メッセージ末尾へ置く。
    # これが無いと Meta-Cognitive 層に振られたクエリにだけ現在日付・人格制約・
    # 文字数上限が届かない (本関数の唯一の消費者が MetaCognitiveAgent のため、
    # 層が変わっただけで制約が消える)。3 つとも (history) -> str の純粋関数で、
    # シグナルが無ければ空文字列を返すのでトークン浪費にはならない。
    for note in (
        _current_date_note(history),
        _persona_question_note(history),
        _char_limit_note(history),
    ):
        if note:
            _append_note_to_last_user(trimmed_history, note)

    logger.debug(
        "build_messages_for_loop: %d steps injected, remaining=%d",
        len(step_parts), remaining,
    )

    return [{"role": "system", "content": sys_content}, *trimmed_history]


#: ``_latest_turn_truncation_note`` の注記に確保するトークン数。注記は最新ターンを
#: 切り詰めたときだけ system へ足すため、切り詰めが確定している場合のみ履歴予算から
#: 引く (常時予約すると通常経路の履歴予算を無駄に削る)。予約せずに後付けすると
#: プロンプトが予約予算を超え、context 溢れの 400 を招く。
_TRUNCATION_NOTE_RESERVE = 120


def _chars_within_token_budget(text: str, max_tokens: int) -> int:
    """``text`` の先頭から ``max_tokens`` に収まる最大の文字数を返す。

    ``estimate_tokens`` は CJK 1 文字 ≒ 1 トークン / ASCII 4 文字 ≒ 1 トークン
    で見積もるため、文字数とトークン数の比は本文の構成で変わる。超過率から
    候補長を縮めながら数回試す (二分探索までの精度は不要)。
    """
    if _estimate_tokens(text) <= max_tokens:
        return len(text)
    cut = len(text)
    for _ in range(8):
        est = _estimate_tokens(text[:cut])
        if est <= max_tokens:
            return cut
        cut = min(int(cut * max_tokens / est), int(cut * 0.9)) or 1
    return cut


def latest_turn_truncation(
    history: list[ChatMessage],
    messages: list[ChatMessage],
) -> tuple[int, int] | None:
    """最新ターンが切り詰められた場合 ``(元文字数, 渡した文字数)`` を返す。

    ``build_messages`` の戻り値と入力 history から判定する。UI へ「あなたの発言は
    先頭のみ送られた」と提示するために API 層が使う (system 注記だけでは
    ベースモデルが従わず、ユーザーには何も見えないため)。

    ``messages`` は動的ブロック (few-shot / RAG / 記憶) が最後の user へ前置され、
    さらに文字数上限の注記 (``_char_limit_note``) が後置されることがあるため、
    長さ比較や末尾一致では判定できない。デリミタ以降を「ユーザー発言側」として
    切り出し、そこに元テキストが丸ごと含まれるかで判定する
    (末尾一致で見ていたときは、後置注記があるだけで未切り詰めのターンを
    「先頭のみ送信されました (1743 / 62 文字)」と誤報した)。
    """
    if not history or not messages:
        return None
    original = history[-1]
    sent = next(
        (m for m in reversed(messages) if m.get("role") == original.get("role")),
        None,
    )
    if sent is None:
        return None
    original_text = str(original.get("content") or "")
    # 前置された動的ブロックを除いた「ユーザー発言側」だけを見る。
    user_side = str(sent.get("content") or "").rsplit(
        _DYNAMIC_CONTEXT_DELIMITER, 1,
    )[-1]
    if not original_text or original_text in user_side:
        return None
    kept = user_side.removesuffix("...")
    return len(original_text), len(kept)


def _latest_turn_truncation_note(
    history: list[ChatMessage],
    trimmed: list[ChatMessage],
) -> str | None:
    """最新ターンが予算超過で切られた場合、その旨を伝える system 注記を返す。

    ``_trim_history`` は最新ターンを drop せず ``compress_turn`` で切り詰めて
    保持するが、モデルに届くのは末尾の ``"..."`` だけで「どれだけ落ちたか」は
    伝わらない。そのためモデルは見ていない部分についても断定してしまう
    (2026-07-26 ライブ検証: 11,359 文字のメモを 4,096 文字に切られた状態で
    「検査装置のキャリブレーション周期は何回出てくるか」に対し、実際の 120 回
    ではなく渡された範囲の 43 回を全体の件数として断定した)。

    注記は system メッセージへ足す。ユーザー発言の中に注意書きを混ぜると
    ユーザーが言っていないことを言ったことにしてしまうため採らない。
    """
    if not history or not trimmed:
        return None
    original, kept = history[-1], trimmed[-1]
    if original.get("role") != kept.get("role"):
        return None
    original_text = str(original.get("content") or "")
    kept_text = str(kept.get("content") or "")
    if len(kept_text) >= len(original_text):
        return None
    return (
        f"注: 直近の{original.get('role', 'user')}発言は長さ制限で先頭のみ"
        f"渡されている (元 {len(original_text)} 文字 / 渡した {len(kept_text)} 文字)。"
        "未渡し部分を見たものとして扱わず、全体の件数・集計・網羅列挙は"
        "断定せず途中までしか読めていない旨を明示すること。"
    )


def _trim_history(
    history: list[ChatMessage],
    max_tokens: int,
) -> list[ChatMessage]:
    """トークン予算に収まるよう、古い履歴を圧縮・削除する。

    **最新ターン (通常は現在の user 質問) は予算超過でも drop しない**: 予算に
    収まる長さへ圧縮して保持する (下限は compress_turn 既定の 200 字 ≈ 203 トークン
    で、超過分は呼び出し側の generation_reserve が吸収する)。

    トークン推定は estimate_tokens() を使用
    （CJK: 1文字≒1トークン、ASCII: 4文字≒1トークン）。
    """
    if not history:
        logger.debug("_trim_history: empty history, nothing to trim")
        return []

    result: list[ChatMessage] = []
    total_tokens = 0

    # 新しいターンから逆順に追加
    for turn in reversed(history):
        content = turn.get("content", "")
        estimated_tokens = _estimate_tokens(content)

        if total_tokens + estimated_tokens > max_tokens:
            if not result:
                # 最新ターンは drop せず、予算連動で圧縮して保持する
                # (この分岐では total_tokens == 0 のため残予算 = max_tokens)。
                # max_chars はトークンではなく文字数の上限なので、トークン予算を
                # そのまま渡すと単位が合わない: ASCII (4 文字 ≒ 1 トークン) では
                # 予算の 1/4 しか使わず、CJK (1 文字 ≒ 1 トークン) では予算ぎりぎり
                # まで載る。推定トークン数で残予算まで詰める形に揃える。
                compressed = compress_turn(
                    turn, max_chars=_chars_within_token_budget(
                        content, max(max_tokens, 200),
                    ),
                )
                compressed_tokens = _estimate_tokens(compressed.get("content", ""))
                result.insert(0, compressed)
                total_tokens += compressed_tokens
                logger.warning(
                    "_trim_history: latest turn exceeds budget, kept compressed "
                    "(role=%s, %d->%d tokens, max=%d)",
                    turn.get("role"), estimated_tokens, compressed_tokens,
                    max_tokens,
                )
            else:
                # 予算超過: 残りの古いターンを圧縮
                compressed = compress_turn(turn)
                compressed_tokens = _estimate_tokens(compressed["content"])
                if total_tokens + compressed_tokens <= max_tokens:
                    result.insert(0, compressed)
                    total_tokens += compressed_tokens
                    logger.debug(
                        "_trim_history: compressed turn (role=%s, %d->%d tokens)",
                        turn.get("role"), estimated_tokens, compressed_tokens,
                    )
                else:
                    logger.debug(
                        "_trim_history: dropped turn (role=%s, %d tokens) — "
                        "budget exhausted (%d/%d)",
                        turn.get("role"), estimated_tokens, total_tokens, max_tokens,
                    )
            break
        else:
            result.insert(0, turn)
            total_tokens += estimated_tokens

    logger.debug(
        "_trim_history: %d/%d turns kept, %d estimated tokens (max=%d)",
        len(result), len(history), total_tokens, max_tokens,
    )
    return result

