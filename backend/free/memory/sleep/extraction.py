"""Step 8: SemanticFact Extractor orchestration

``sleep_update.SleepTimeWorker._step8_extract_facts`` として
実装された Extractor 起動ロジックを独立 module に切り出したもの。

処理は 3 段階で構成される:

1. :class:`~backend.free.memory.extractors.chat.ChatExtractor`
   → ``global`` スコープに ``personal_fact`` / ``world_fact`` / ``preference`` /
   ``emotion`` / ``opinion`` を追記
2. :class:`~backend.free.memory.extractors.create.CreateExtractor`
   → ``project:<id>`` スコープに ``project`` / ``decision`` / ``commitment`` /
   ``create_task`` / ``create`` を追記
3. :class:`~backend.free.memory.extractors.mdp_trace.MDPTraceExtractor`
   → ``project:<id>`` スコープに ``failure_pattern`` / ``decision`` を追記
   (config で disable 可)

上記はすべて **regex / 決定論** の抽出で、語形が 1 つ外れるとその属性の
ファクトが 0 件になる。取りこぼしを埋める第 2 段
(:func:`extract_and_split_semantic_facts` → ``sleep.personal_fact_curator``)
は sleep-time の補助タスクで発話を属性ごとの逐語 span に分ける。

本 module は EvorefMem pillar 内部扱いのため SemanticFactStore を直接参照する。
"""

from __future__ import annotations

import re

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from backend.free.memory.notes.note_builder import is_single_valued_subject
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.free.memory.extractors import (
        ExtractionResult,
        MDPTraceExtractor,
    )
    from backend.free.memory.semantic.store import SemanticFactStore
    from backend.free.memory.episodic.note import MemoryNote
    from backend.free.memory.notes.subject_canonicalizer import SubjectCanonicalizer

logger = get_logger("memory.sleep.extraction")


def persist_facts(
    store: "SemanticFactStore",
    result: "ExtractionResult",
    label: str,
) -> int:
    """``ExtractionResult`` のファクトを ``SemanticFactStore`` に書き込む。

    重複 ID 衝突 (極めてまれ) や書き込み失敗は warning ログにとどめ、
    sleep-time 全体は止めない。

    Args:
        store: 書き込み先ストア。
        result: ``BaseExtractor.extract`` の戻り値。
        label: ログ用ラベル (``"chat"`` / ``"create"`` / ``"mdp_trace"`` 等)。

    Returns:
        実際に書き込まれたファクト数。
    """
    written = 0
    persisted: list = []
    # 同じスロット名 (subject の末尾) と同じ本文を持つファクトが kind 違いで
    # 並ぶことがある (color / beverage は personal_fact と preference の両節に
    # あり、一人称の申告は両タグに当たる)。同じ値を 2 件 live にしない
    # (2026-09-11 (k): mem.personal.color と mem.preference.color が二重)。
    seen_values: set[tuple[str, str]] = set()
    for fact in result.facts:
        slug = (getattr(fact, "subject", "") or "").rsplit(".", 1)[-1]
        key = (slug, "".join(str(getattr(fact, "object", "") or "").split()))
        if slug and key in seen_values:
            logger.debug(
                "Step 8 [%s]: skipped duplicate value for slot %s (%s)",
                label, slug, fact.subject,
            )
            continue
        seen_values.add(key)
        try:
            store.add_fact(fact)
            written += 1
            persisted.append(fact)
        except Exception as exc:
            logger.warning(
                "Step 8 [%s]: failed to add fact %s: %s",
                label, fact.id, exc,
            )
    _supersede_corrected_slots(store, persisted, label)
    _retire_assertions_contradicted_by_change(store, persisted, label)
    if written:
        logger.debug("Step 8 [%s]: persisted %d facts", label, written)
    return written


#: 「<旧> ではなく <新>」の旧値 span。``ではなく`` の直前で、主題・目的語の
#: 助詞か読点から後ろを旧値とみなす (助詞が無ければ文頭から)。
_OLD_VALUE_BEFORE_NEGATION_RE = re.compile(
    r"(?:^|[はがをもに、，,。．])\s*(?P<old>[^、，,。はがをも]{2,40}?)\s*(?:ではなく|じゃなく)",
)
_ASSERTION_SUBJECT_PREFIX = "mem.world.assertion."


def _old_value_spans(fact: object) -> list[str]:
    """変更 / 訂正ファクトが無効化する旧値の逐語 span (正規化済み)。"""
    from backend.free.core.correction_verdict import norm_span

    spans: list[str] = []
    # ``statement`` は訂正形を新値だけに畳んである (``_reduce_correction_statement``)
    # ので旧値は **object (原文)** にしか無い。両方を見る (実機再検証で
    # statement だけを見て 0 件だった)。
    for text in (
        str(getattr(fact, "object", "") or ""),
        str(getattr(fact, "statement", None) or ""),
    ):
        for m in _OLD_VALUE_BEFORE_NEGATION_RE.finditer(text):
            old = norm_span(m.group("old"))
            if len(old) >= 2 and old not in spans:
                spans.append(old)
    return spans


def _retire_assertions_contradicted_by_change(
    store: "SemanticFactStore", persisted: list, label: str,
) -> int:
    """属性スロットへ入った変更が、同じ旧値を持つ world assertion も畳む。

    「来週の木曜日に顧客へ提案書を送る予定です」は Step 8 が型付けできず
    assertion curator が ``mem.world.assertion.proposal_submission_plan``
    に置く。次の「送付は来週の木曜日ではなく、再来週の月曜日にします」は
    ``変わりました`` / ``ではなく`` で personal_fact (schedule) に型付け
    されるので curator には届かず、assertion 側は旧日付のまま live で残り、
    別セッションの「提案書を送る予定日は」に旧日付が注入された
    (2026-09-10 ライブ監査 (i) I-15)。訂正の宛先を subject で結べないので
    **旧値の逐語 span** で結ぶ (値アンカーと同じ根拠)。同じセッション由来の
    assertion に限る (別会話の同じ言い回しまで消さない)。

    Returns:
        supersede した assertion 数。
    """
    from backend.free.core.correction_verdict import norm_span

    superseded = 0
    search = getattr(store, "search_by_pillar_prefix", None)
    if search is None:
        return 0
    for fact in persisted:
        subject = str(getattr(fact, "subject", "") or "")
        if subject.startswith(_ASSERTION_SUBJECT_PREFIX):
            continue
        spans = _old_value_spans(fact)
        if not spans:
            continue
        sessions = set(getattr(fact, "session_ids", None) or ())
        try:
            assertions = search(_ASSERTION_SUBJECT_PREFIX, include_superseded=False)
        except Exception as exc:  # noqa: BLE001 - 読めなければ何も畳まない
            logger.warning("Step 8 [%s]: failed to list assertions: %s", label, exc)
            return superseded
        for old in assertions:
            if old.superseded_by or old.id == fact.id:
                continue
            if sessions and not (set(getattr(old, "session_ids", None) or ()) & sessions):
                continue
            if getattr(old, "created_at", 0.0) > getattr(fact, "created_at", 0.0):
                continue
            body = norm_span(str(getattr(old, "object", "") or ""))
            if not any(span in body for span in spans):
                continue
            try:
                store.supersede(old.id, fact.id)
                superseded += 1
                logger.info(
                    "Step 8 [%s]: assertion %s superseded by change %s (%s)",
                    label, old.subject, fact.id, fact.subject,
                )
            except (KeyError, ValueError) as exc:
                logger.warning(
                    "Step 8 [%s]: failed to supersede %s -> %s: %s",
                    label, old.id, fact.id, exc,
                )
    return superseded


def _supersede_corrected_slots(
    store: "SemanticFactStore", persisted: list, label: str,
) -> int:
    """訂正ファクトを書いたら、同じスロットの旧世代を supersede する。

    ``SemanticFactStore.supersede`` は既に存在するが、**抽出経路からは
    一度も呼ばれていなかった**。呼んでいたのはセッション要約の昇格
    (``sleep.promotion``)、競合解決 (``conflict_review`` /
    ``SemanticConflictResolver``)、learn / loop の書き戻しだけで、
    チャット由来の属性ファクトは新旧が live のまま積み上がる。

    実データ (2026-08-27 ライブ監査、``semantic/global/facts.jsonl`` 12 件):
    全件が ``supersedes: []`` / ``superseded_by: None``。``from_correction``
    は 4 件立っているのに旧世代が 1 つも無効化されていない。結果、

    - ``mem.personal.occupation`` に「データベース管理者」と
      「ネットワークエンジニア」が同時に live
    - ``mem.personal.name`` に 3 世代 (テスト太郎 / 御堂 陽介 / 田中) が live

    となり、新規セッションでの想起が **同じ問いに毎回ちがう値** を返した
    (名前は 1 回目「御堂 陽介」/ 2 回目「田中」)。注入側の
    ``_collapse_to_current_values`` は 1 スロット 1 値へ畳むが、
    **畳む前に検索へ乗るのは全世代**で、埋め込み検索の上位に旧値が来れば
    その時点で負ける。ストア側で世代を閉じるのが本筋。

    畳む条件は ``from_correction`` が立っているか、**スロットが単値と宣言されて
    いる** こと。訂正でない再言明まで無条件に supersede すると、``pet`` のように
    1 人が複数値を持ちうるスロットで正当な値を落とすため、一括では畳まない。

    単値スロット (``fact_attributes.yaml`` の ``single_valued: true``) を条件に
    加えたのは、**訂正ではない更新** が旧値を live のまま残していたから。
    実データ (2026-08-29 ライブ監査、``semantic/global/facts.jsonl``):

    - 「先月、横浜から札幌に引っ越しました」→ ``from_correction`` が立たず、
      ``mem.personal.location`` に 横浜 / 札幌 / 名古屋 が **3 つとも live**
    - その結果、次セッションの想起が旧値を返した
      (T28: 「39歳, **横浜市**, **ソフトウェアエンジニア**」/ T29: 更新前の出張日程)
    - 自己検査も「古い情報は含まれていません」と旧値を最新だと保証した

    ``single_valued`` を宣言していないスロットの挙動は一切変わらない
    (既定 ``False``)。

    Returns:
        supersede した旧ファクト数。
    """
    superseded = 0
    #: 同一バッチで同じスロットへ複数の値が書かれたとき、**勝者を 1 つに決めて
    #: から** 畳む。素朴に「新規ファクトごとに他の live を supersede する」と、
    #: 同じ subject の 2 件が **互いを supersede** して live が 0 件になる。
    #:
    #: 実データ (2026-08-29 クリーンストア検証): 「今は千葉に住んでいます」と
    #: 「先週、千葉から神戸に引っ越しました」が同じ Full で抽出され、
    #: ``mem.personal.location`` の **2 件とも SUPERSEDED** になった
    #: (``occupation`` も同様)。想起は「確認できていません」に落ちる。
    #: 勝者は ``persisted`` の **最後** に来たもの (抽出順 = 発話順)。
    #:
    #: **畳む側になり得る全ファクトを勝者表に載せる。** 以前は
    #: ``is_single_valued_subject`` のものだけを載せていたため、``from_correction``
    #: で畳みに来る多値スロットのファクトは ``winners.get`` が None になり、
    #: 同じ subject の訂正 2 件が揃って supersede ループへ入って **互いを
    #: supersede** した。実データ (2026-08-30 ライブ監査): デプロイ訂正と
    #: インスタンスタイプ訂正が同じ ``mem.personal.birthday`` (多値) に載り、
    #: ``sf_17c4c4ac9939 <-> sf_9f97d02bb6b0`` の 2-閉路になってスロットの live が
    #: 0 件になった。この関数の入口条件と勝者表の条件は同じでなければならない。
    def _collapses(fact: object) -> bool:
        # 検証済み訂正 / 単値スロット / 本人の値更新 (「本社ではなく名古屋支社」
        # を旧値の言明へ置換した行、J-03) が旧世代を畳む側に回る。
        return bool(
            getattr(fact, "from_correction", False)
            or getattr(fact, "value_update", False)
            or is_single_valued_subject(getattr(fact, "subject", "") or "")
        )

    winners: dict[tuple[str, str], object] = {}
    for fact in persisted:
        if not _collapses(fact):
            continue
        winners[(fact.subject, fact.predicate)] = fact
    for fact in persisted:
        if not _collapses(fact):
            continue
        # 単値スロットは勝者だけが畳む側に回る (敗者は何も supersede しない)。
        winner = winners.get((fact.subject, fact.predicate))
        if winner is not None and winner is not fact:
            continue
        try:
            siblings = store.search_by_subject(
                fact.subject, include_superseded=False,
            )
        except Exception as exc:
            logger.warning(
                "Step 8 [%s]: failed to list slot %s: %s",
                label, fact.subject, exc,
            )
            continue
        # 本人の値更新は **旧値を含む世代だけ** を畳む。schedule のような並列多値
        # スロットで全兄弟を畳むと、別の予定 (提案書の送付日) まで消える。
        update_span = str(getattr(fact, "value_update", "") or "")
        if update_span:
            from backend.free.core.correction_verdict import norm_span

            needle = norm_span(update_span)
            siblings = [
                o for o in siblings
                if needle and needle in norm_span(str(getattr(o, "object", "") or ""))
            ]
        for old in siblings:
            if old.id == fact.id or old.predicate != fact.predicate:
                continue
            if old.superseded_by:
                continue
            # **発話時刻の順で畳む** (書込み順ではない)。古いノートを後から
            # 読み直す経路 (Step 8.3 の属性分割 / 再抽出) が、新しい発話由来の
            # live 値を古い証拠で上書きしてはならない。実データ (2026-09-08
            # 検証): 午前の A/B ノート「境川サイクリングロードをよく走ります」
            # を Step 8.3 が location に割り当て、直前の自己紹介の「金沢市」を
            # supersede した。``created_at`` は発話時刻 (永続形の ``as_of``)。
            #
            # **訂正 (``from_correction``) も例外にしない** (2026-09-08 夜の
            # 監査 G-04)。訂正が無効化できるのは *自分より前の発言* だけで、
            # 後の発言まで消せてよい理由が無い。実データ: 午前 02:34Z の
            # ノートが引用中の「色が違う」で訂正候補になり、11:51Z の
            # occupation ファクトを supersede していた。
            if getattr(old, "created_at", 0.0) > getattr(fact, "created_at", 0.0):
                try:
                    store.supersede(fact.id, old.id)
                    superseded += 1
                    logger.info(
                        "Step 8 [%s]: newer live value kept for %s; older "
                        "evidence %s superseded on arrival",
                        label, fact.subject, fact.id,
                    )
                except (KeyError, ValueError) as exc:
                    logger.warning(
                        "Step 8 [%s]: failed to supersede %s -> %s: %s",
                        label, fact.id, old.id, exc,
                    )
                break
            try:
                store.supersede(old.id, fact.id)
                superseded += 1
            except (KeyError, ValueError) as exc:
                # 昇格側と同じ扱い — 書き込み自体は成立しているので警告に留め、
                # 残った旧世代は競合解決 / TTL に委ねる。
                logger.warning(
                    "Step 8 [%s]: failed to supersede %s -> %s: %s",
                    label, old.id, fact.id, exc,
                )
    if superseded:
        logger.info(
            "Step 8 [%s]: superseded %d stale slot value(s) by corrections",
            label, superseded,
        )
    return superseded


#: ``mem.<kind>.<attr>`` の ``kind`` → ``FactType``。値アンカー用の逆引き。
_FACT_TYPE_BY_KIND: dict[str, str] = {
    "personal": "personal_fact",
    "world": "world_fact",
    "preference": "preference",
    "emotion": "emotion",
    "opinion": "opinion",
}


def collect_live_attribute_values(
    store: "SemanticFactStore",
) -> dict[tuple[str, str], tuple[str, ...]]:
    """live ファクトから ``{(fact_type, 属性スロット): (現在値, ...)}`` を組む。

    属性語を落とした訂正 (「さっき名古屋と言いましたが、正しくは横浜です。」)
    の宛先を決めるための材料。詳細は
    :func:`~backend.free.memory.extractors.chat.
    resolve_value_anchored_attributes`。

    ``user`` スロットは除く — 属性が解決できなかったファクトの受け皿なので、
    そこを名指しても宛先を絞れない。
    """
    values: dict[tuple[str, str], list[str]] = {}
    try:
        facts = store.all_facts(include_superseded=False)
    except Exception as exc:
        logger.warning("Step 8: failed to read live facts: %s", exc)
        return {}
    for fact in facts:
        parts = (fact.subject or "").split(".")
        if len(parts) != 3 or parts[0] != "mem":
            continue
        fact_type = _FACT_TYPE_BY_KIND.get(parts[1])
        attr = parts[2]
        if not fact_type or attr == "user":
            continue
        text = (fact.text or "").strip()
        if text:
            values.setdefault((fact_type, attr), []).append(text)
    return {key: tuple(vals) for key, vals in values.items()}


def _mdp_extract_state_path(agent_trace_dir: Path | None) -> Path | None:
    """Step 8 の処理済み episode_id 永続先 (``local/memory/mdp_extract_state.json``)。

    ``ensure_mdp_ingester`` と同じく ``memory_dir`` を優先し、resolver が使えない
    場合は ``agent_trace_dir`` 配下へフォールバックする。
    """
    try:
        from backend.config import get_path_resolver
        return Path(get_path_resolver().resolve_local("memory_dir")) / "mdp_extract_state.json"
    except Exception:
        if agent_trace_dir is None:
            return None
        return Path(agent_trace_dir) / "mdp_extract_state.json"


def _drop_facts_with_existing_subject(
    store: "SemanticFactStore",
    result: "ExtractionResult",
) -> int:
    """既に同一 subject の active fact が存在する ``decision`` 候補を除外する。

    ``MDPTraceExtractor`` は ``decision`` を ``mem.decision.<episode_id>``
    (エピソード毎に一意) で生成する。プロセス再起動で抽出器の in-memory
    ``_processed_episode_ids`` が失われると同一エピソードが再抽出されるが、
    既存 subject を弾くことで新しい ``fact_id`` での重複追記を防ぐ (store が
    dedup の永続状態を兼ねる)。``chat`` / ``create`` 抽出器は同一 subject の再
    アサートで内容を更新する設計のため、この dedup は MDP 経路にのみ適用する。

    ``failure_pattern`` (loop 所有) は呼出側で事前に分離され
    :func:`_persist_failure_patterns_via_view` が ``LoopFactView`` 経由で
    signature 単位の in-place occurrences 加算として書くため、本 dedup には
    渡らない (別エピソードでの同一 signature 再発を弾くと再発頻度が失われる)。
    よって本関数の対象は ``decision`` のみ。

    Returns:
        除外した件数。
    """
    if not result.facts:
        return 0
    kept = []
    dropped = 0
    for fact in result.facts:
        # decision のみ subject 一意性 dedup。failure_pattern は Step 13 に委ねる。
        if fact.type == "decision" and store.search_by_subject(
            fact.subject, include_superseded=False,
        ):
            dropped += 1
            continue
        kept.append(fact)
    if dropped:
        result.facts = kept
        logger.debug(
            "Step 8 [mdp_trace]: skipped %d duplicate-subject facts", dropped,
        )
    return dropped


def _persist_failure_patterns_via_view(
    project_store: "SemanticFactStore",
    failure_facts: list,
    project_id: str,
) -> int:
    """MDP 由来の ``failure_pattern`` を ``LoopFactView`` 経由で書き込む。

    ``failure_pattern`` は loop 所有 FactType のため、mem pillar が
    ``store.add_fact`` で直書きすると ownership enforcement を素通りする。
    :meth:`LoopFactView.write_failure_pattern` 経由にすることで owner 検証を
    通し、同一 signature を **in-place で occurrences 加算** する
    (failure_consolidator が LoopFactView を使うのと同じ前例)。MDP 抽出器が
    組み立てた JSON object (``error_type`` / ``normalized_file_path`` /
    ``last_actions`` / ``outcomes_history``) を分解して低レベル API に渡す。

    Returns:
        書き込んだ failure_pattern 数。
    """
    if not failure_facts:
        return 0
    from backend.free.memory.views.loop import LoopFactView

    view = LoopFactView(stores=[project_store], writeback_store=project_store)
    written = 0
    for f in failure_facts:
        signature = f.failure_signature or ""
        if not signature:
            continue
        try:
            payload = json.loads(f.object)
        except (json.JSONDecodeError, TypeError):
            payload = {}
        outcomes = payload.get("outcomes_history") or []
        try:
            view.write_failure_pattern(
                project_id=project_id,
                signature=signature,
                error_type=str(payload.get("error_type", "")),
                normalized_file_path=str(payload.get("normalized_file_path", "")),
                last_actions=list(payload.get("last_actions") or []),
                outcome_label=str(outcomes[0]) if outcomes else None,
                trace_id=f.trace_id,
            )
            written += 1
        except Exception as exc:
            logger.warning(
                "Step 8 [mdp_trace]: failure_pattern write_via_view failed "
                "(sig=%s): %s", signature, exc,
            )
    if written:
        logger.debug(
            "Step 8 [mdp_trace]: persisted %d failure_pattern via LoopFactView",
            written,
        )
    return written


def extract_semantic_facts(
    notes: list["MemoryNote"],
    *,
    config: dict | None,
    store_provider: Callable[[str], "SemanticFactStore | None"] | None,
    current_project_id: str | None,
    agent_trace_dir: Path | None,
    subject_canonicalizer: "SubjectCanonicalizer | None",
    mdp_trace_extractor: "MDPTraceExtractor | None" = None,
    mdp_trace_extractor_factory: Callable[[], "MDPTraceExtractor"] | None = None,
    verification_available: bool = False,
) -> tuple[int, "MDPTraceExtractor | None"]:
    """Step 8: ChatExtractor / CreateExtractor / MDPTraceExtractor を順次実行する。

    Guards:

    - ``memory.facts.enable_extraction = False`` → no-op (``0``)
    - ``store_provider`` が ``None`` → no-op (``0``)
    - ``global`` store 取得失敗 → Chat skip
    - ``current_project_id`` 未設定 / project store 取得失敗 → Create / MDP skip

    Args:
        notes: 対象ノート群 (通常は ``EpisodicWorkspace.notes.values()`` のリスト)。
        config: ``memory.facts`` 配下の設定を含む設定 dict。
        store_provider: ``scope`` → ``SemanticFactStore`` を返すコールバック。
        current_project_id: 現在のプロジェクト ID。
        agent_trace_dir: ``agent_trace*.jsonl`` のディレクトリ (MDP 抽出用)。
        subject_canonicalizer: subject の正規化器
        mdp_trace_extractor: 既存の MDPTraceExtractor インスタンス
            (プロセス内で episode の二重抽出を防ぐためワーカー側で保持するもの)。
        mdp_trace_extractor_factory: ``mdp_trace_extractor`` が ``None``
            の場合に新規生成するファクトリ。未指定時は
            :class:`MDPTraceExtractor` を直接 import して生成する。

    Returns:
        ``(total_extracted, mdp_trace_extractor)`` のペア。
        第二要素は caller にキャッシュして再利用させるためのもの (初回実行で
        生成したインスタンスを返す; 2 回目以降は同じインスタンスが戻る)。
    """
    cfg_facts = (config or {}).get("memory", {}).get("facts", {}) or {}
    if not cfg_facts.get("enable_extraction", True):
        logger.debug("Step 8: extraction disabled by config")
        return 0, mdp_trace_extractor
    if store_provider is None:
        logger.debug("Step 8: no semantic store provider, skipping")
        return 0, mdp_trace_extractor

    from backend.free.memory.extractors import (
        ChatExtractor,
        CreateExtractor,
        ExtractionContext,
        MDPTraceExtractor,
    )

    max_per_session_cfg = cfg_facts.get("extraction_max_per_session", {}) or {}
    ctx = ExtractionContext(
        project_id=current_project_id,
        agent_trace_dir=agent_trace_dir,
        max_per_session={
            "chat": int(max_per_session_cfg.get("chat", 10)),
            "create": int(max_per_session_cfg.get("create", 5)),
        },
        max_pinned_per_session=int(
            cfg_facts.get("extraction_max_pinned_per_session", -1),
        ),
        canonicalizer=subject_canonicalizer,
    )

    total_extracted = 0
    try:
        global_store = store_provider("global")
    except Exception as exc:
        logger.warning("Step 8: failed to obtain global store: %s", exc)
        global_store = None

    # ── 1. ChatExtractor → global ──
    if global_store is not None:
        # 属性語を落とした訂正の宛先を決めるため、既存スロットの現在値を渡す
        # (chat.resolve_value_anchored_attributes の説明を参照)。
        ctx.live_attribute_values = collect_live_attribute_values(global_store)
        # Step 8.0 が検証できる構成なら、未検証の訂正候補は据え置く (H-12)。
        ctx.defer_unverified_corrections = bool(verification_available)
        chat_result = ChatExtractor().extract(notes, ctx)
        if chat_result.notes_deferred:
            logger.info(
                "Step 8: %d correction candidate(s) deferred until verified",
                chat_result.notes_deferred,
            )
        total_extracted += persist_facts(global_store, chat_result, "chat")

    # ── 2. CreateExtractor → project ──
    project_store = None
    if current_project_id:
        try:
            project_store = store_provider(f"project:{current_project_id}")
        except Exception as exc:
            logger.warning("Step 8: failed to obtain project store: %s", exc)

    if project_store is not None:
        create_result = CreateExtractor().extract(notes, ctx)
        total_extracted += persist_facts(project_store, create_result, "create")

    # ── 3. MDPTraceExtractor → project ──
    if (
        project_store is not None
        and bool(cfg_facts.get("extract_from_mdp_trace", True))
    ):
        if mdp_trace_extractor is None:
            if mdp_trace_extractor_factory is not None:
                mdp_trace_extractor = mdp_trace_extractor_factory()
            else:
                mdp_trace_extractor = MDPTraceExtractor(
                    state_path=_mdp_extract_state_path(agent_trace_dir),
                )
        mdp_result = mdp_trace_extractor.extract(notes, ctx)
        # failure_pattern は loop 所有なので LoopFactView 経由で書く (ownership
        # 準拠 + signature 単位の in-place occurrences 加算)。decision は mem の
        # store 直書き経路 (subject 一意 dedup) のまま。
        failure_facts = [f for f in mdp_result.facts if f.type == "failure_pattern"]
        mdp_result.facts = [
            f for f in mdp_result.facts if f.type != "failure_pattern"
        ]
        _drop_facts_with_existing_subject(project_store, mdp_result)
        total_extracted += persist_facts(
            project_store, mdp_result, "mdp_trace",
        )
        total_extracted += _persist_failure_patterns_via_view(
            project_store, failure_facts, current_project_id,
        )

    if total_extracted:
        logger.info("Step 8: extracted %d facts", total_extracted)
    else:
        logger.debug("Step 8: no facts extracted")
    return total_extracted, mdp_trace_extractor


async def extract_and_split_semantic_facts(
    notes: list["MemoryNote"],
    *,
    aux_client=None,
    embedder=None,
    profile_id: str = "default",
    **extract_kwargs,
) -> tuple[int, "MDPTraceExtractor | None"]:
    """Step 8 (regex 抽出) → Step 8.3 (LLM 属性分割) を 1 本にした入口。

    **regex が型付けし、LLM が分割・補完する** 二段構え。第 2 段は
    :mod:`backend.free.memory.sleep.personal_fact_curator` が担う (発火条件・
    検証規則・実インシデントはそちらの docstring)。

    分割を Step 8 の直後に置くのは、regex が同じサイクルで書いたファクトを
    ``note.extracted_fact_ids`` から引き当てて **粗い object を狭め直す**
    ため。順序が逆だと、あとから走る regex が同じスロットへ粗い値を書き足す。

    第 2 段は ``aux_client`` / ``embedder`` が無ければ何もしない (degraded)。
    したがって既存の同期入口 :func:`extract_semantic_facts` と挙動は変わらず、
    呼出側は補助タスクを渡せるようになったときだけ二段目を得る。

    Args:
        notes: 対象ノート群。
        aux_client: 属性分割に使う補助タスククライアント。``None`` で第 2 段
            を no-op にする。
        embedder: 分割で作るファクトの埋め込み生成器。
        profile_id: 書込先 fact の profile_id。
        extract_kwargs: :func:`extract_semantic_facts` へそのまま渡す。

    Returns:
        ``(total_extracted, mdp_trace_extractor)``。件数は 2 段の合計。
    """
    from backend.free.memory.sleep.personal_fact_curator import (
        curate_personal_facts,
    )

    total, mdp_trace_extractor = extract_semantic_facts(notes, **extract_kwargs)
    total += await curate_personal_facts(
        notes,
        store_provider=extract_kwargs.get("store_provider"),
        aux_client=aux_client,
        embedder=embedder,
        profile_id=profile_id,
    )
    return total, mdp_trace_extractor


__all__ = [
    "collect_live_attribute_values",
    "extract_and_split_semantic_facts",
    "extract_semantic_facts",
    "persist_facts",
]
