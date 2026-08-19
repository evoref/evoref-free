"""Meta-Cognitive ループのトークン予算計算（router.py / meta_cognitive.py 共有）。

``_can_use_meta_cognitive`` (router.py) と ``MetaCognitiveAgent.__init__``
(meta_cognitive.py) が独立に同じ式をハードコードしていた重複を解消する。
"""

from __future__ import annotations

from backend.config import resolve_context_size_for_mode

# 生成出力用の予約トークン数。``MetaCognitiveAgent._execute_max_tokens =
# max(ctx_size - 512, 1024)`` と同値であることから「1ターンの生成出力予約」
# と推測されるが、この意味づけを明記した設計文書は見つかっていない。
OUTPUT_RESERVE_TOKENS = 512

# ツールループの固定オーバーヘッド予約トークン数。用途は使用箇所からの
# 推測であり、根拠となる設計文書は未確認。
LOOP_OVERHEAD_TOKENS = 400

# 送信直前ガード (``LocalClient._enforce_context_budget``) が確保する余白の
# 見積り。ガードは ``budget = n_ctx - 安全マージン - max_tokens`` でプロンプト
# 側の上限を決めるため、``max_tokens`` を ``n_ctx - プロンプト長`` いっぱいまで
# 積むと budget がプロンプト長を必ず下回り、**毎回プロンプト側が中略される**
# (実測 2026-08-19 ライブ監査: ファイル書き込みの本文生成で
# ``est=900 > budget=828`` となり user メッセージが 2135 → 1621 文字へ削られた)。
# 内訳は安全マージン 192 + メッセージ毎オーバーヘッド (4 tok × 最大 16 通)。
SEND_GUARD_RESERVE_TOKENS = 256


def resolve_meta_cognitive_loop_budget(config: dict, mode: str = "chat") -> int:
    """Meta-Cognitive ループが使えるトークン予算を計算する。

    ``ctx_size - OUTPUT_RESERVE_TOKENS - LOOP_OVERHEAD_TOKENS - history_budget``。
    """
    ctx_size = resolve_context_size_for_mode(config, mode)
    history_budget = config.get("memory", {}).get("working_max_tokens", 2048)
    return ctx_size - OUTPUT_RESERVE_TOKENS - LOOP_OVERHEAD_TOKENS - history_budget
