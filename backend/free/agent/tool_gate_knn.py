"""ツール要否の二値ゲート (埋め込み exemplar 近傍法)。

ツール判定の最終層 (文法制約 JSON 分類器) を撃つかどうかを決める門。従来は
正規表現 ``_query_has_tool_signal`` が担っていたが、**実クエリ 137 件の
ベンチで recall 66.2% しかなく、ツールが要るクエリの 3 分の 1 を落として
いた** (`backend/free/agent/tests/data/tool_gate_bench.jsonl`)。

穴が集中していたのは:

- ``file-read`` 6/9 取りこぼし — 「保存したファイルの中身を見せて」型。
  ツール名もパスも書かれないため正規表現に引っかからない。
- ``arith-challenge`` 5/6 — 「その計算、〜ではないですか」型の訂正要求。
- ``unit-conversion`` 2/5 — 「何MBになりますか」等。

どれも監査で繰り返し実害 (ファイル内容の捏造 / 訂正要求への暗算) が出ている型。

埋め込み近傍なら同じベンチで **k=5 の leave-one-out で recall 98.5%**
(取りこぼし 1 件)。precision は 84.9% → 81.7% と下がるが、137 ターンあたり
無駄な分類器呼出が 8 → 15 件 (+7) 増えるだけで、22 件の取りこぼしを回収できる。

**「どのツールか」は判定しない**。それは検証済みの分類器 (19/20) の仕事で、
ここは要否だけを見る。二値なので exemplar が少なくて済み、確定した判断から
育てられる。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from backend.log_config import get_logger

logger = get_logger("agent.tool_gate_knn")

#: 近傍投票数。ベンチでは k=1 が recall 92.6%、k=5 が 98.5%。取りこぼしの
#: コスト (誤答) が無駄撃ちのコスト (分類器 1 回) より高いので k=5 を採る。
DEFAULT_K = 5

#: 同梱 exemplar (tracked)。ユーザ override は ``local/triggers`` と同じ
#: 2 段階構成にせず、まず同梱のみ。育成経路は今後 Level 1 側で足す。
_DEFAULTS_FILE = Path(__file__).parent / "_defaults" / "tool_gate_exemplars.jsonl"

#: exemplar 埋め込みのクエリ整形。検索と同じテンプレートを使う (質問文と
#: 平叙文で埋め込みが非対称なため、判定側も同じ側に揃える)。
_QUERY_TEMPLATE = "query: {query}"


def load_exemplars(path: Path | None = None) -> list[tuple[str, str]]:
    """``(query, label)`` の配列を読む。``label`` は ``tool`` / ``none``。"""
    src = path or _DEFAULTS_FILE
    if not src.exists():
        logger.warning("Tool gate exemplars not found: %s", src)
        return []
    out: list[tuple[str, str]] = []
    for line in src.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except (ValueError, TypeError):
            continue
        if "_comment" in rec:
            continue
        q = rec.get("query")
        label = rec.get("label")
        if isinstance(q, str) and q.strip() and label in ("tool", "none"):
            out.append((q.strip(), label))
    return out


class ToolGateKNN:
    """exemplar 近傍でツール要否を判定する。

    埋め込みは起動後に一度だけ生成する (137 件で実測 ~5.6 秒)。生成が終わる
    までは :meth:`is_ready` が ``False`` を返し、呼出側は従来の正規表現ゲートへ
    縮退する — 起動直後の数ターンのために応答を待たせない。
    """

    def __init__(
        self,
        embedder,
        *,
        k: int = DEFAULT_K,
        exemplars: list[tuple[str, str]] | None = None,
    ) -> None:
        self._embedder = embedder
        self._k = max(1, int(k))
        self._exemplars = exemplars if exemplars is not None else load_exemplars()
        self._vectors: np.ndarray | None = None
        self._labels: list[str] = []

    def is_ready(self) -> bool:
        """判定に使える状態か。``False`` なら呼出側は従来ゲートへ縮退する。"""
        return self._vectors is not None and len(self._labels) >= self._k

    async def warmup(self) -> bool:
        """exemplar を埋め込む。成功で ``True``。失敗しても例外は投げない。"""
        if self._vectors is not None:
            return True
        if self._embedder is None or not self._exemplars:
            logger.info(
                "Tool gate kNN not warmed up: embedder=%s exemplars=%d",
                self._embedder is not None, len(self._exemplars),
            )
            return False
        try:
            texts = [
                _QUERY_TEMPLATE.format(query=q) for q, _ in self._exemplars
            ]
            vecs = await self._embedder.embed(texts, is_query=True, mode="chat")
            mat = np.asarray(vecs, dtype=np.float32)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._vectors = (mat / norms).astype(np.float32)
            self._labels = [label for _, label in self._exemplars]
            logger.info(
                "Tool gate kNN ready: %d exemplars (tool=%d, none=%d), k=%d",
                len(self._labels), self._labels.count("tool"),
                self._labels.count("none"), self._k,
            )
            return True
        except Exception as e:  # pragma: no cover - 縮退で吸収する
            logger.warning("Tool gate kNN warmup failed: %s", e)
            return False

    async def needs_tool(self, query: str) -> bool | None:
        """ツールが要りそうなら ``True``。判定できなければ ``None``。

        ``None`` は「この門では決められない」を意味し、呼出側は従来の
        正規表現ゲートへ縮退する (誤って閉じない)。
        """
        if not self.is_ready() or not query.strip():
            return None
        try:
            qv = await self._embedder.embed_query(
                _QUERY_TEMPLATE.format(query=query), mode="chat",
            )
            q = np.asarray(qv, dtype=np.float32)
            norm = float(np.linalg.norm(q))
            if norm == 0.0:
                return None
            q = q / norm
        except Exception as e:
            logger.info("Tool gate kNN embed failed: %s", e)
            return None

        assert self._vectors is not None
        if self._vectors.shape[1] != q.shape[0]:
            logger.warning(
                "Tool gate kNN dim mismatch (exemplars=%d, query=%d); "
                "falling back to the rule gate",
                self._vectors.shape[1], q.shape[0],
            )
            return None

        sims = self._vectors @ q
        k = min(self._k, sims.shape[0])
        idx = np.argpartition(sims, -k)[-k:]
        votes = [self._labels[int(i)] for i in idx]
        needed = votes.count("tool") * 2 > k
        logger.debug(
            "Tool gate kNN: %s (votes tool=%d/%d) for %r",
            needed, votes.count("tool"), k, query[:50],
        )
        return needed
