"""Layer 1: ワーキングメモリ（直近Nターンのインメモリ管理）"""

import time
from uuid import uuid4

from backend.free.memory.stores.fact_slate import SessionFactSlate
from backend.log_config import get_logger
from backend.utils import estimate_tokens as _estimate_tokens

logger = get_logger("memory.working")

#: 既に一度警告した (context_size, 設定値) の組。毎ターン同じ WARNING を
#: 出さないためのデデュープ (WorkingMemory はセッションごとに作られる)。
_reconcile_warned: set[tuple[int, int]] = set()


def _reconcile_working_max_tokens(config: dict, mem: dict) -> int:
    """WM の窓を、プロンプト側が実際に載せられる上限へ丸める。

    **窓を決める主体を 1 つにする。** WM は ``working_max_tokens`` まで溜める
    一方、``build_messages`` は ``context_size`` から生成予約・system・動的
    ブロック予約を引いた残りしか載せられない。設定値が構造的に達成不能だと、
    WM のブロック押し出しと ``_trim_history`` の切り落としが別々のタイミングで
    先頭を動かし、接頭辞 KV キャッシュが崩れる回数が倍になる。

    実測 (2026-09-03 ライブ監査、11:00 以降に 12 回):
    ``working_max_tokens=4352 exceeds the prompt history budget=2486
    (context_size=8192, generation_reserve=1024, system=2279, dyn_reserve=1600)``
    — 設定値は context_size=8192 の下でどうやっても満たせない。従来は WARNING で
    「config を下げろ」と促すだけで、実際には毎ターン二重に切られていた。

    元の設定値が達成可能ならそのまま返す (縮めない)。

    **context_size が分からないときは丸めない。** 素の dict で作る呼出
    (テスト / 部分 config) では ``resolve_context_size`` が既定値を返すが、
    それは推測であって実機の窓ではない。推測で窓を縮めると、設定と無関係に
    挙動が変わる。``llama`` セクションを持つ config だけを対象にする。
    """
    configured = int(mem.get("working_max_tokens", 4096) or 4096)
    if not (config.get("llama") or {}):
        return configured
    try:
        from backend.config import resolve_context_size
        from backend.free.core.prompt_budget import resolve_budgets
    except Exception:
        return configured
    try:
        context_size = int(resolve_context_size(config, "base") or 0)
    except Exception:
        context_size = 0
    if context_size <= 0:
        return configured
    # 予算関数は 1 つ (c_02 §6.3)。system はレンダラが守る上限
    # (``prompt.system_max_share``) で見積もる — WM はレンダ前に作られる。
    budgets = resolve_budgets(config, context_size)
    ceiling = budgets.working_max_tokens
    if configured <= ceiling:
        return configured
    key = (context_size, configured)
    if key not in _reconcile_warned:
        _reconcile_warned.add(key)
        logger.warning(
            "memory.working_max_tokens=%d cannot fit the prompt budget "
            "(context_size=%d, generation_reserve=%d, system_max=%d, "
            "dyn_reserve=%d, fact_slate=%d); clamping the working-memory window "
            "to %d so a single component decides it",
            configured, context_size, budgets.generation_reserve,
            budgets.system_max_tokens, budgets.dyn_reserve,
            budgets.fact_slate_tokens, ceiling,
        )
    return ceiling

#: トークン上限を超えたときに落とす下限水位 (``max_tokens`` に対する比)。
#: 上限ちょうどで止めると次ターンで即また超えて毎ターン先頭が動く。
#:
#: この比は「次に上限へ達するまでの猶予ターン数」を決める。窓の先頭が動くと
#: llama-server の接頭辞 KV キャッシュが system プロンプト以降まるごと無効化
#: され、プロンプト全体が再 prefill される。hybrid (recurrent) アーキテクチャの
#: モデルでは ``--cache-reuse`` による部分再利用も無効化されるため、逃げ道が無い。
#:
#: 2026-08-16 ライブ監査 (40 ターン / Qwen3.8-27B / n_ctx 8192) の実測:
#:
#:   押し出しが起きたターン   n=9  中央値 141.4 秒
#:   起きなかったターン       n=31 中央値  42.7 秒
#:
#: ツール発火の有無で層別しても 141.4 秒 vs 44.9 秒 (3.1 倍) で、押し出しは
#: 全 40 ターンの 22.5% でしか起きないのに LLM 総時間 2,848 秒の 48% (1,373 秒)
#: を占めていた。1 回あたり約 +96 秒の純損失。
#:
#: 同セッションの実発話長 (80 メッセージ / 平均 91 tok) で水位を振ると、
#: 先頭が動いた回数は 0.8 → 0.6 で **7 回 → 4 回** (出荷既定の 30/2048 構成)。
#: 保持メッセージ数の平均は 15.2 → 14.1 とわずかに下がるだけで、割に合う。
_TOKEN_EVICT_KEEP_RATIO = 0.6

#: turn dict に立てる「STM へ転送済み」の印。``snapshot_unabsorbed`` が立て、
#: ``_evict_oldest`` / ``clear`` が転送バッファへの二重積みを抑止するのに使う。
#: turn dict はそのまま ``ShortTermMemory.absorb`` へ渡るが、absorb は必要な
#: キーだけを読むため未知キーが増えても影響しない。
_ABSORBED_KEY = "absorbed"


class WorkingMemory:
    """Layer 1: ゼロレイテンシの会話コンテキスト"""

    def __init__(self, config: dict):
        mem = config.get("memory", {})
        # フォールバックはスキーマ既定 (backend/schemas/memory.py) と同じ値。
        # validate_config は既定値を実体化するので本番では常にキーがあるが、
        # 素の dict で作るテストが別の窓で走らないよう揃える。
        self.max_turns: int = mem.get("working_max_turns", 256)
        self.max_tokens: int = _reconcile_working_max_tokens(config, mem)
        #: 上限に達したときに **まとめて** 押し出すターン数 (ヒステリシス)。
        #:
        #: 1 ターンずつ押し出すと、窓の先頭が毎ターン 1 つずれる。プロンプトは
        #: ``system + 履歴`` の順に組み立てられ、llama-server は共通接頭辞の
        #: KV キャッシュを再利用するため、先頭がずれると **system 以降が毎ターン
        #: 全部無効化** される。実測 (2026-08-09 ライブ監査、base=gemma-4-12b):
        #:
        #:   prompt eval time = 27,712 ms / 2,898 tokens   ← 固定 28 秒
        #:          eval time =  7,105 ms /    54 tokens   ← decode は正常
        #:   selected slot by LCP similarity, f_sim_best = 0.366
        #:
        #: 連続ターンのプロンプト差分を取ると共通接頭辞は毎回きっちり
        #: system プロンプト長 (1,834 文字) で、その直後から食い違っていた。
        #: 8〜14 トークンしか出さない短い応答でも 28〜40 秒かかっていた。
        #:
        #: まとめて押し出すと、次に上限へ達するまでの数ターンは先頭が動かず
        #: 接頭辞が再利用できる。代償は保持ターン数の平均が
        #: ``max_turns - block/2`` に下がること。
        self.evict_block: int = max(
            1, int(mem.get("working_evict_block", 6)),
        )
        self.turns: list[dict] = []
        self.session_id: str = uuid4().hex[:8]
        self._evicted: list[dict] = []  # Layer 2 転送用バッファ
        #: 押し出したターンの要点表 (f_02 §1.2)。窓の補助であって記憶層ではない。
        self.fact_slate = SessionFactSlate()
        #: 現在のセッションで押し出したターン数 (``clear()`` でリセット)。
        #: ``_evicted`` は ``drain_evicted()`` で吸い出されるため残高を見ても
        #: 「このセッションで会話の前半が視界から落ちたか」は分からない。
        #: 会話全体を走査しないと答えられない質問 (「この会話で依頼した
        #: ファイル操作を全部」等) で、見えていない範囲を「無い」と断定させない
        #: ための注記を出すかどうかの判定に使う (2026-08-05 ライブ監査:
        #: ターン19 で行ったファイル書き込みが 30 メッセージ窓から外れた状態で
        #: 「この会話で依頼したファイル操作を全部」と聞かれ、
        #: 「ファイル操作はありません」と断言した)。
        self.session_evicted_turns: int = 0
        #: このセッションで最初に届いた user 発話 (``clear()`` でリセット)。
        #:
        #: 「この会話で最初に何を言ったか」は並び順で決まる事実なので、検索にも
        #: モデルの読解にも委ねる理由が無い。ところが窓から押し出されると
        #: ``turns`` からも ``_evicted`` からも消えるため、押し出し後は
        #: **注記を出して降りる** しかなかった (``agent.deliberative.
        #: _append_session_position_fact``)。
        #:
        #: 実インシデント (2026-08-16 ライブ監査 ターン34):
        #: 「今日の会話のいちばん最初、私は何の話をした？」に対し、正解
        #: 「おはよう。今朝はけっこう冷え込んでるね…」は ``[参考情報 2]`` として
        #: **プロンプトに載っていた** のに、併記された「冒頭 42 件は見えていない・
        #: 断定するな」の注記が勝ち、窓の先頭 (「SaaS の解約率を…」) を
        #: 「確認できる範囲での最古」として答えた。1 文字保持しておけば
        #: 決定論で確定できる。
        self.session_first_user_turn: str = ""

    def add_turn(
        self,
        role: str,
        content: str,
        *,
        private: bool = False,
        mode: str | None = None,
        project_id: str | None = None,
        source: str | None = None,
        tool_command: str | None = None,
        tool_command_name: str | None = None,
        tool_command_success: bool | None = None,
        tool_command_source: str | None = None,
        tool_command_query: str | None = None,
        correction: bool = False,
    ) -> None:
        """ターンを追加し、トークン上限を超えたら圧縮・押し出し

        EvorefMem 拡張:
        ``private=True`` のターンは ``MemoryNote.private=True`` で吸収され、
        STM 以降 (LTM / SemMem) には伝播しない。``mode`` / ``project_id`` /
        ``source`` も任意で turn dict に格納し、``ShortTermMemory.absorb``
        が読み取る。``tool_command`` / ``tool_command_name`` /
        ``tool_command_success`` は run_command 実行ターンの learning 用メタで、
        sleep-time の executable_command_curator が参照する (それ以外は None)。
        ``tool_command_query`` は当該コマンドを発火させたユーザークエリで、
        curator が STM 走査で対応付けを推測しないために持たせる。

        ``correction=True`` は「この発話がユーザー自身の値の言い直し」であることを
        表す (判定は :func:`backend.free.agent.feedback.restates_a_value`。
        アシスタントの誤りの指摘と、ユーザー自身の申告訂正の両方を含む)。
        ``MemoryNote.is_correction`` → ``SemanticFact.from_correction`` と伝播し、
        (a) 抽出器が直前の名前付き属性を継承して訂正が対象と同じスロットへ入り、
        (b) sleep-time の競合解決が「同一セッション内だから微妙ケース」として
        pending へ落とすのを免除する。訂正は会話中に起きるので、この印が無いと
        **いちばん確度の高い訂正がいちばん自動解決されない**。
        """
        est_tokens = _estimate_tokens(content)
        logger.debug(
            "add_turn: role=%s, content_len=%d, est_tokens=%d, "
            "turns_before=%d, session=%s, private=%s",
            role, len(content), est_tokens, len(self.turns), self.session_id,
            private,
        )
        turn: dict = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
        }
        if private:
            turn["private"] = True
        if mode is not None:
            turn["mode"] = mode
        if project_id is not None:
            turn["project_id"] = project_id
        if source is not None:
            turn["source"] = source
        if tool_command is not None:
            turn["tool_command"] = tool_command
        if tool_command_name is not None:
            turn["tool_command_name"] = tool_command_name
        if tool_command_success is not None:
            turn["tool_command_success"] = tool_command_success
        if tool_command_source is not None:
            turn["tool_command_source"] = tool_command_source
        if tool_command_query is not None:
            turn["tool_command_query"] = tool_command_query
        if correction:
            turn["is_correction"] = True
        if role == "user" and not self.session_first_user_turn and content.strip():
            self.session_first_user_turn = content
        self.turns.append(turn)
        self._enforce_limits()

    def get_context(self) -> list[dict]:
        """推論時のコンテキスト取得（0ms: 配列参照のみ）"""
        return self.turns

    def get_messages(self) -> list[dict]:
        """LLM用 messages 形式で返す（role + content のみ）"""
        return [{"role": t["role"], "content": t["content"]} for t in self.turns]

    def snapshot_unabsorbed(self) -> list[dict]:
        """まだ STM へ渡していないターンを **非破壊で** 返し、転送済みに印を付ける。

        f_02 §1.2 経路 (c)。窓 (``working_max_turns`` / ``working_max_tokens``)
        に収まる長さの会話は押し出しが起きず、セッションが終わるまで STM
        ノートを 1 件も生まない。sleep-time Step 8 の入力は STM ノートなので、
        その間に走った Full は入力が空のまま ``facts_extracted=0`` になる
        (2026-08-18 ライブ監査: 21 ターンの会話が再起動まで 1 件も
        ファクト化されなかった)。Full の直前にここを呼び、WM を保ったまま
        STM 側だけ先に埋める。

        WM からは取り除かない (会話 context を壊さない)。後で同じターンが
        押し出されても二重に吸収しないよう ``absorbed`` フラグを立て、
        :meth:`_evict_oldest` / :meth:`clear` が転送バッファへ積むのを抑止する。
        """
        pending = [t for t in self.turns if not t.get(_ABSORBED_KEY)]
        for turn in pending:
            turn[_ABSORBED_KEY] = True
        if pending:
            logger.debug(
                "snapshot_unabsorbed: %d/%d turns handed to Layer 2 (WM retained)",
                len(pending), len(self.turns),
            )
        return pending

    def clear(self) -> None:
        """コンテキストクリア"""
        # 既に snapshot で STM へ渡したターンは転送バッファへ積まない
        # (二重ノートになる)。
        pending = [t for t in self.turns if not t.get(_ABSORBED_KEY)]
        logger.debug(
            "clear: evicting %d/%d turns to transfer buffer",
            len(pending), len(self.turns),
        )
        self._evicted.extend(pending)
        self.turns.clear()
        # clear() はセッション切替時に呼ばれる。新しいセッションでは「前半が
        # 落ちている」状態ではないのでカウンタも畳む。
        self.session_evicted_turns = 0
        self.session_first_user_turn = ""

    def drain_evicted(self) -> list[dict]:
        """Layer 2 への転送用: 押し出されたターンを取得してバッファをクリア"""
        evicted = self._evicted[:]
        self._evicted.clear()
        if evicted:
            logger.debug("drain_evicted: %d turns transferred to Layer 2", len(evicted))
        return evicted

    def _enforce_limits(self) -> None:
        """ターン数・トークン上限を強制

        どちらの上限も、超えたら **下限水位までまとめて** 落とす
        (``evict_block`` の説明を参照)。1 件ずつ削ると窓の先頭が毎ターン動き、
        llama-server の接頭辞 KV キャッシュが毎回無効化されて prefill が
        応答時間を支配する。
        """
        # ターン数制限
        if len(self.turns) > self.max_turns:
            # 一度に窓の半分より多くは捨てない。小さい ``max_turns`` (テスト値や
            # 極小構成) で block が窓を上回ると、1 回のオーバーフローで会話が
            # ほぼ空になる。
            block = max(1, min(self.evict_block, max(1, self.max_turns // 2)))
            target = max(1, self.max_turns - block + 1)
            logger.debug(
                "_enforce_limits: turn count %d > max %d, evicting to %d "
                "(block=%d)",
                len(self.turns), self.max_turns, target, block,
            )
            while len(self.turns) > target:
                self._evict_oldest()

        # トークン制限
        total = self._total_tokens()
        if total <= self.max_tokens:
            return
        # 上限ちょうどまでしか削らないと次ターンで即また超える。窓の先頭を
        # 動かす回数を減らすため下限水位 (既定 60%) まで落とす。
        token_target = max(1, int(self.max_tokens * _TOKEN_EVICT_KEEP_RATIO))
        logger.debug(
            "_enforce_limits: tokens %d > max %d, compressing/evicting to %d",
            total, self.max_tokens, token_target,
        )
        # 先頭を ``[要約]`` へ書き換えてから押し出す 2 段構えは廃止した。
        # 圧縮しても次の反復で必ず押し出されるため (2026-08-16 ライブ監査:
        # ``compressing oldest turn`` 23 回に対し ``evicting compressed turn``
        # も 23 回 = 生き残りゼロ)、保持量は 1 件も増えないまま副作用だけが残る:
        #   1. 窓の先頭 (system 直後) を書き換えるので llama-server の接頭辞
        #      KV キャッシュが押し出しとは別に無効化される。
        #   2. 同じターンが原文と ``[要約]`` の 2 回 ``_evicted`` へ積まれ、
        #      STM に同一発話のノートが 2 本できる。そこから起こしたファクトは
        #      statement が完全一致するため「未解決の競合」として恒久 pending 化し、
        #      無関係な質問のプロンプトにまで注入される (実データ:
        #      sf_127618cb29fb 原文 / sf_852c5461ce09 ``[要約]``、同一秒・同一 subject)。
        # 押し出しのみにすると窓の先頭は単調に前進し、転送も 1 ターン 1 回になる。
        while self._total_tokens() > token_target and len(self.turns) > 1:
            logger.debug(
                "_enforce_limits: evicting oldest turn (role=%s, %d chars)",
                self.turns[0].get("role"), len(self.turns[0].get("content", "")),
            )
            self._evict_oldest()

    def _evict_oldest(self) -> None:
        """最古ターンを押し出し

        押し出し後に先頭へ露出した assistant ターンは、対になる user ターン
        (直前に押し出された) を失った孤立発言であり、次回 LLM 呼出時に
        messages 配列の先頭 (system 直後) へ verbatim で渡ってしまう。
        文脈を失ったまま目立つ位置に居座ると、後続ターンで LLM がこの
        孤立発言をそのまま複製する逐語的自己反復を招くため (2026-07-18
        の会話ログで実際に発生・確認済み)、連鎖的に押し出す。

        ただし連鎖は ``len(self.turns) > 1`` の間のみ行う。1 件を残して
        止めないと、token 予算ループの ``while ... and len(self.turns) > 1``
        ガード (押し出しは常に高々 1 ターンという前提で書かれている) が
        破れ、1 回の呼出で残り 2 ターンが両方消え、直前に追加したばかりの
        ターンごと会話履歴が空になる回帰を招く (レビューで判明)。孤立
        ターンが最後の 1 件になった場合はそのまま残し、次の add_turn で
        新しい user ターンが追加された後の押し出しで改めて連鎖させる。
        """
        if self.turns:
            evicted = self.turns.pop(0)
            # snapshot 済みのターンは転送バッファへ積まない (二重ノート防止)。
            # 窓から落ちた事実は変わらないのでカウンタは通常どおり進める。
            if not evicted.get(_ABSORBED_KEY):
                self._evicted.append(evicted)
            # 要点表は STM 転送とは別 (f_02 §1.2): snapshot 済みでも窓から
            # 消える事実は同じなので、転送バッファを経由せずここで渡す。
            # 実機 (2026-09-03): 全ターンが snapshot 済みで転送バッファが空の
            # まま押し出され、スレートが一度も埋まらなかった。
            self.fact_slate.absorb([evicted])
            self.session_evicted_turns += 1
        while len(self.turns) > 1 and self.turns[0].get("role") == "assistant":
            orphan = self.turns.pop(0)
            if not orphan.get(_ABSORBED_KEY):
                self._evicted.append(orphan)
            self.session_evicted_turns += 1

    def _total_tokens(self) -> int:
        """全ターンの推定トークン数"""
        return sum(_estimate_tokens(t["content"]) for t in self.turns)


#: ``WorkingMemoryRegistry.drop`` の ``drain_to`` 既定値 (コンストラクタで
#: 渡した STM を使う) を表す番兵。``None`` は「転送せず捨てる」の意味で使う。
_DRAIN_DEFAULT = object()


class WorkingMemoryRegistry:
    """セッション別 :class:`WorkingMemory` の台帳 (LRU 上限付き)。

    以前は WM がプロセスに 1 つしか無く、``prepare_memory_context`` がセッション
    切替のたびに窓を空にして STM へ流していた。並行セッションでは A の生成中に
    B が窓を奪い、A の応答が B の会話に載る (2026-09-02 監査 A-5 の迂回ガード
    が要った理由)。ここでは session_id ごとに独立した WM を持ち、切替という
    概念そのものを無くす。

    - :meth:`get` は無ければ作る。触ったセッションを LRU の末尾へ動かし、
      ``memory.working_max_sessions`` を超えたら最古のセッションを
      **セッション終了と同じ経路** (``clear()`` → drain ハンドラ) で STM へ
      流してから落とす (f_02 §1.2 経路 (b))。
    - drain ハンドラは api 層のエコー落とし規則 (``chat_recorder.
      drain_evicted_to_stm``) を持つため **注入** で受ける
      (``SleepTimeScheduler.set_pre_full_flush`` と同じ配線パターン)。未注入
      なら ``stm.absorb`` を直に呼ぶ縮退動作。
    - :class:`WorkingMemory` 自体の意味 (窓 / 押し出しブロック / ``absorbed`` /
      ``session_first_user_turn`` / ``session_evicted_turns``) は変えない。
    """

    def __init__(
        self,
        config: dict,
        *,
        drain_to=None,
        drain_handler=None,
    ) -> None:
        mem = config.get("memory", {})
        # フォールバックはスキーマ既定 (backend/schemas/memory.py) と同じ値。
        self.max_sessions: int = max(1, int(mem.get("working_max_sessions", 8)))
        self._config = config
        self._drain_to = drain_to
        self._drain_handler = drain_handler
        self._sessions: dict[str, WorkingMemory] = {}
        # legacy 読み手 (session_id を持たない統計 API 等) が 1 件も無いときに
        # 見る空の窓。台帳には載せないので drain / snapshot の対象にならない。
        self._scratch: WorkingMemory | None = None

    # ── 取得 ────────────────────────────────────────────────────────

    def get(self, session_id: str) -> WorkingMemory:
        """``session_id`` の WM を返す (無ければ作る)。LRU の末尾へ動かす。"""
        wm = self._sessions.pop(session_id, None)
        if wm is None:
            wm = WorkingMemory(self._config)
            wm.session_id = session_id
            logger.debug(
                "registry: new working memory for session %s (active=%d)",
                session_id, len(self._sessions) + 1,
            )
        self._sessions[session_id] = wm
        self._evict_over_capacity(keep=session_id)
        return wm

    def peek(self, session_id: str) -> WorkingMemory | None:
        """作らずに返す (無ければ ``None``)。LRU 順は動かさない。"""
        return self._sessions.get(session_id)

    def current(self) -> WorkingMemory:
        """最後に触ったセッションの WM (legacy: session_id を持たない読み手向け)。

        並行セッション下では「誰の窓か」が定まらないので、応答パスでは必ず
        :meth:`get` を session_id 付きで使うこと。1 件も無ければ台帳外の空の
        窓を返す (統計 API が ``len(wm.turns)`` を読めるように)。
        """
        if self._sessions:
            return next(reversed(self._sessions.values()))
        if self._scratch is None:
            self._scratch = WorkingMemory(self._config)
        return self._scratch

    def active_sessions(self) -> list[str]:
        """台帳にあるセッション ID (古い順)。"""
        return list(self._sessions)

    def __len__(self) -> int:
        return len(self._sessions)

    # ── 終了 / 転送 ────────────────────────────────────────────────

    def drop(self, session_id: str, *, drain_to=_DRAIN_DEFAULT) -> WorkingMemory | None:
        """セッションの WM を台帳から外す。

        ``drain_to`` に STM を渡すと (既定はコンストラクタの ``drain_to``)、
        まだ STM へ渡していないターンを **そのセッション ID で** 吸収してから
        落とす (f_02 §1.2 経路 (b): ``clear()`` → drain)。``None`` なら捨てる。
        無ければ ``None`` を返す。
        """
        wm = self._sessions.pop(session_id, None)
        if wm is None:
            return None
        target = self._drain_to if drain_to is _DRAIN_DEFAULT else drain_to
        pending = len(wm.turns)
        # ``clear()`` を先に実行してから drain する。逆順だと窓超過で既に押し
        # 出された分しか拾えず、会話本体は転送バッファに滞留する。
        wm.clear()
        if target is not None:
            self._drain(wm, target, session_id)
        logger.debug(
            "registry: dropped session %s (%d turns pending, drained=%s, active=%d)",
            session_id, pending, target is not None, len(self._sessions),
        )
        return wm

    def drain_all(self, drain_to=_DRAIN_DEFAULT) -> int:
        """全セッションを :meth:`drop` する (プロセス終了時)。落とした件数を返す。"""
        sids = self.active_sessions()
        for sid in sids:
            self.drop(sid, drain_to=drain_to)
        return len(sids)

    def snapshot_all_unabsorbed(self) -> list[tuple[str, list[dict]]]:
        """全セッションの未転送ターンを **非破壊で** 返す (Full 直前の経路 (c))。

        ``[(session_id, turns), ...]`` を古いセッション順で返す。ターンには
        ``absorbed`` の印が立つので、後で押し出されても二重吸収されない。
        未転送が無いセッションは含めない。
        """
        out: list[tuple[str, list[dict]]] = []
        for sid, wm in self._sessions.items():
            pending = wm.snapshot_unabsorbed()
            if pending:
                out.append((sid, pending))
        return out

    # ── 内部 ────────────────────────────────────────────────────────

    def _evict_over_capacity(self, *, keep: str) -> None:
        while len(self._sessions) > self.max_sessions:
            oldest = next(iter(self._sessions))
            if oldest == keep:
                # 上限 1 で自分しか居ない、等。取ったばかりの窓は落とさない。
                break
            logger.info(
                "registry: session cap %d reached, draining LRU session %s to STM",
                self.max_sessions, oldest,
            )
            self.drop(oldest)

    def _drain(self, wm: WorkingMemory, stm, session_id: str) -> None:
        if self._drain_handler is not None:
            self._drain_handler(wm, stm, session_id)
            return
        for turn in wm.drain_evicted():
            stm.absorb(turn, session_id)
