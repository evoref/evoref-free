"""Layer 1: ワーキングメモリ（直近Nターンのインメモリ管理）"""

import time
from uuid import uuid4

from backend.log_config import get_logger
from backend.utils import compress_turn, estimate_tokens as _estimate_tokens

logger = get_logger("memory.working")

#: トークン上限を超えたときに落とす下限水位 (``max_tokens`` に対する比)。
#: 上限ちょうどで止めると次ターンで即また超えて毎ターン先頭が動く。
_TOKEN_EVICT_KEEP_RATIO = 0.8


class WorkingMemory:
    """Layer 1: ゼロレイテンシの会話コンテキスト"""

    def __init__(self, config: dict):
        mem = config.get("memory", {})
        self.max_turns: int = mem.get("working_max_turns", 30)
        self.max_tokens: int = mem.get("working_max_tokens", 2048)
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
        self.active_notes: list[str] = []
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
        self.turns.append(turn)
        self._enforce_limits()

    def get_context(self) -> list[dict]:
        """推論時のコンテキスト取得（0ms: 配列参照のみ）"""
        return self.turns

    def get_messages(self) -> list[dict]:
        """LLM用 messages 形式で返す（role + content のみ）"""
        return [{"role": t["role"], "content": t["content"]} for t in self.turns]

    def clear(self) -> None:
        """コンテキストクリア"""
        logger.debug("clear: evicting all %d turns to transfer buffer", len(self.turns))
        self._evicted.extend(self.turns)
        self.turns.clear()
        # clear() はセッション切替時に呼ばれる。新しいセッションでは「前半が
        # 落ちている」状態ではないのでカウンタも畳む。
        self.session_evicted_turns = 0

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
        # 動かす回数を減らすため下限水位 (既定 80%) まで落とす。
        token_target = max(1, int(self.max_tokens * _TOKEN_EVICT_KEEP_RATIO))
        logger.debug(
            "_enforce_limits: tokens %d > max %d, compressing/evicting to %d",
            total, self.max_tokens, token_target,
        )
        while self._total_tokens() > token_target and len(self.turns) > 1:
            oldest = self.turns[0]
            if not oldest.get("compressed"):
                # 未圧縮なら圧縮で対処
                logger.debug(
                    "_enforce_limits: compressing oldest turn (role=%s, %d chars)",
                    oldest.get("role"), len(oldest.get("content", "")),
                )
                self._evicted.append(oldest)  # 原文を転送バッファに
                self.turns[0] = compress_turn(oldest, max_chars=60, style="summary")
            else:
                # 圧縮済みは押し出し
                logger.debug("_enforce_limits: evicting compressed turn")
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
            self._evicted.append(evicted)
            self.session_evicted_turns += 1
        while len(self.turns) > 1 and self.turns[0].get("role") == "assistant":
            orphan = self.turns.pop(0)
            self._evicted.append(orphan)
            self.session_evicted_turns += 1

    def _total_tokens(self) -> int:
        """全ターンの推定トークン数"""
        return sum(_estimate_tokens(t["content"]) for t in self.turns)
