"""Layer 1: ワーキングメモリ（直近Nターンのインメモリ管理）"""

import time
from uuid import uuid4

from backend.log_config import get_logger
from backend.utils import estimate_tokens as _estimate_tokens

logger = get_logger("memory.working")

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
        self.max_tokens: int = mem.get("working_max_tokens", 4096)
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
            self.session_evicted_turns += 1
        while len(self.turns) > 1 and self.turns[0].get("role") == "assistant":
            orphan = self.turns.pop(0)
            if not orphan.get(_ABSORBED_KEY):
                self._evicted.append(orphan)
            self.session_evicted_turns += 1

    def _total_tokens(self) -> int:
        """全ターンの推定トークン数"""
        return sum(_estimate_tokens(t["content"]) for t in self.turns)
