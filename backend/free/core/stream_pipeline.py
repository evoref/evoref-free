"""ストリーミングフィルタパイプライン

複数のフィルタをチェーンして適用するパイプライン。
StreamThinkingFilter + HeadBufferFilter の組み合わせを宣言的に構成する。
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class StreamFilter(Protocol):
    """ストリーミングフィルタのプロトコル

    全てのストリーミングフィルタはこのプロトコルを実装する。
    process() でトークンを逐次処理し、flush() で残りバッファを排出する。
    """

    def process(self, token: str) -> str:
        """トークンをフィルタリングする

        Args:
            token: 入力トークン

        Returns:
            出力すべきテキスト。フィルタリングで除去された場合は空文字列。
        """
        ...

    def flush(self) -> str:
        """残りのバッファをフラッシュする

        Returns:
            バッファに残っていたテキスト。なければ空文字列。
        """
        ...


class StreamPipeline:
    """フィルタをチェーンして適用するパイプライン

    フィルタは登録順に適用される。前段のフィルタが空文字列を返した場合、
    後段のフィルタは呼ばれない（短絡評価）。

    Usage:
        pipeline = StreamPipeline([thinking_filter, head_filter])
        for token in stream:
            output = pipeline.process(token)
            if output:
                yield output
        remaining = pipeline.flush()
        if remaining:
            yield remaining
    """

    def __init__(self, filters: list[StreamFilter]) -> None:
        self._filters = list(filters)

    def process(self, token: str) -> str:
        """トークンを全フィルタに順次適用する

        Args:
            token: 入力トークン

        Returns:
            全フィルタ適用後の出力テキスト。除去された場合は空文字列。
        """
        result = token
        for f in self._filters:
            result = f.process(result)
            if not result:
                return ""
        return result

    def flush(self) -> str:
        """全フィルタのバッファをフラッシュする

        各フィルタの flush 出力を後続フィルタに通してから結合する。

        Returns:
            全フィルタのフラッシュ結果を結合したテキスト。
        """
        parts: list[str] = []
        for i, f in enumerate(self._filters):
            remaining = f.flush()
            if remaining:
                # 後続フィルタを通す
                for j in range(i + 1, len(self._filters)):
                    remaining = self._filters[j].process(remaining)
                    if not remaining:
                        break
                if remaining:
                    parts.append(remaining)
        return "".join(parts)
