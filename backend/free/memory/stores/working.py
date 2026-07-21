"""Layer 1: ワーキングメモリ（直近Nターンのインメモリ管理）"""

import time
from uuid import uuid4

from backend.log_config import get_logger
from backend.utils import compress_turn, estimate_tokens as _estimate_tokens

logger = get_logger("memory.working")


class WorkingMemory:
    """Layer 1: ゼロレイテンシの会話コンテキスト"""

    def __init__(self, config: dict):
        mem = config.get("memory", {})
        self.max_turns: int = mem.get("working_max_turns", 10)
        self.max_tokens: int = mem.get("working_max_tokens", 2048)
        self.turns: list[dict] = []
        self.active_notes: list[str] = []
        self.session_id: str = uuid4().hex[:8]
        self._evicted: list[dict] = []  # Layer 2 転送用バッファ

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
    ) -> None:
        """ターンを追加し、トークン上限を超えたら圧縮・押し出し

        EvorefMem 拡張:
        ``private=True`` のターンは ``MemoryNote.private=True`` で吸収され、
        STM 以降 (LTM / SemMem) には伝播しない。``mode`` / ``project_id`` /
        ``source`` も任意で turn dict に格納し、``ShortTermMemory.absorb``
        が読み取る。``tool_command`` / ``tool_command_name`` /
        ``tool_command_success`` は run_command 実行ターンの learning 用メタで、
        sleep-time の executable_command_curator が参照する (それ以外は None)。
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

    def drain_evicted(self) -> list[dict]:
        """Layer 2 への転送用: 押し出されたターンを取得してバッファをクリア"""
        evicted = self._evicted[:]
        self._evicted.clear()
        if evicted:
            logger.debug("drain_evicted: %d turns transferred to Layer 2", len(evicted))
        return evicted

    def _enforce_limits(self) -> None:
        """ターン数・トークン上限を強制"""
        # ターン数制限
        while len(self.turns) > self.max_turns:
            logger.debug(
                "_enforce_limits: turn count %d > max %d, evicting oldest",
                len(self.turns), self.max_turns,
            )
            self._evict_oldest()

        # トークン制限
        total = self._total_tokens()
        if total > self.max_tokens:
            logger.debug(
                "_enforce_limits: tokens %d > max %d, compressing/evicting",
                total, self.max_tokens,
            )
        while self._total_tokens() > self.max_tokens and len(self.turns) > 1:
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
        while len(self.turns) > 1 and self.turns[0].get("role") == "assistant":
            orphan = self.turns.pop(0)
            self._evicted.append(orphan)

    def _total_tokens(self) -> int:
        """全ターンの推定トークン数"""
        return sum(_estimate_tokens(t["content"]) for t in self.turns)
