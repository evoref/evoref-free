"""SSE フレーム生成を一元化する

各ストリーミング関数（reactive / deliberative / meta_cognitive / long_form）で
個別に実装されていた SSE フレーム生成を統一する。

フレーム形式は c_05_data_schemas.md §12 に準拠。
"""

import json


class SSEFrameBuilder:
    """SSE フレーム生成の統一ビルダー

    全ストリーミング関数が共有するフレーム生成ロジックを一元管理する。
    フレーム種別ごとにメソッドを提供し、JSON 構造の不整合を防止する。

    Usage:
        sse = SSEFrameBuilder()
        yield sse.agent_layer("deliberative")
        yield sse.token("こんにちは")
        yield sse.token_info({"used": 100, "limit": 4096, "pct": 2})
        yield sse.done()
    """

    @staticmethod
    def token(text: str) -> str:
        """トークンフレーム: LLM 生成トークンの逐次送信

        Args:
            text: 生成されたトークン（1〜数文字）
        """
        return f"data: {json.dumps({'token': text}, ensure_ascii=False)}\n\n"

    @staticmethod
    def step(step_data: dict) -> str:
        """ステップフレーム: エージェント処理の進捗通知

        Args:
            step_data: {"type": str, "detail": str, "status": str, ...}
        """
        return f"data: {json.dumps({'step': step_data}, ensure_ascii=False)}\n\n"

    @staticmethod
    def agent_layer(layer: str) -> str:
        """エージェント層フレーム: 応答元レイヤーの通知

        Args:
            layer: "reactive" | "deliberative" | "meta_cognitive"
        """
        return f"data: {json.dumps({'agent_layer': layer}, ensure_ascii=False)}\n\n"

    @staticmethod
    def editor_route(target: str) -> str:
        """エディタ振り分けフレーム: 生成コードの出力先通知（coding モードのみ）

        ストリーム冒頭でトークンより先に 1 度だけ送信し、フロントエンドが
        コードブロックをエディタへ流すかチャットに表示するかを決める。

        Args:
            target: "editor" (既定、エディタへ流す) | "chat" (チャットに表示)
        """
        return f"data: {json.dumps({'editor_route': {'target': target}}, ensure_ascii=False)}\n\n"

    @staticmethod
    def token_info(info: dict) -> str:
        """終端情報フレーム: トークン使用状況

        Args:
            info: {"used": int, "limit": int, "pct": int, "instance_name"?: str}
        """
        return f"data: {json.dumps({'token_info': info}, ensure_ascii=False)}\n\n"

    @staticmethod
    def rag_debug(chunks: list[dict], search_time_ms: float) -> str:
        """RAG デバッグフレーム: 検索結果チャンクの可視化（debug.enabled 時のみ送信）

        Args:
            chunks: [{"source": str, "score": float, "preview": str}, ...]
            search_time_ms: 検索パイプラインの所要時間（ミリ秒）
        """
        payload = {
            "rag_debug": {
                "chunks": chunks,
                "search_time_ms": round(search_time_ms, 1),
            }
        }
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    @staticmethod
    def error(msg: str) -> str:
        """エラーフレーム

        Args:
            msg: エラーメッセージ
        """
        return f"data: {json.dumps({'error': msg})}\n\n"

    @staticmethod
    def error_with_code(code: str, message: str, **context) -> str:
        """構造化エラーフレーム（コード付き）

        Args:
            code: エラーコード（例: E0513）
            message: エラーメッセージ
            **context: 追加のコンテキスト情報
        """
        payload = {"error": {"code": code, "message": message, "context": context}}
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    @staticmethod
    def result(payload: dict) -> str:
        """完了フレーム: 最終レスポンスペイロード

        SSE ストリームの末尾近く ([DONE] の前) で 1 度だけ送信される、
        非ストリーミング API と同形のレスポンス本体。

        Args:
            payload: API レスポンスと同等の辞書
        """
        return f"data: {json.dumps({'result': payload}, ensure_ascii=False)}\n\n"

    @staticmethod
    def done() -> str:
        """終端フレーム: ストリーム終了マーカー"""
        return "data: [DONE]\n\n"

    @staticmethod
    def keepalive() -> str:
        """キープアライブコメント（SSE コメント形式）

        クライアントのタイムアウト防止用。JSON データではなく
        SSE 仕様のコメント行として送信する。
        """
        return ": keepalive\n\n"
