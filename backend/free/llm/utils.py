"""LLM レスポンスユーティリティ

OpenAI 互換レスポンスからのデータ抽出や、テキスト截断などの共通処理。
"""

from __future__ import annotations


def extract_content(response: dict) -> str:
    """OpenAI 互換レスポンスからコンテンツを抽出"""
    choices = response.get("choices") or [{}]
    return (
        choices[0]
        .get("message", {})
        .get("content", "")
    )
