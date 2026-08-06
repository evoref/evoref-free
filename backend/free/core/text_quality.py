"""生成テキストの決定論的な品質チェック (pillar 非依存の共有基盤)

モデル差し替えで静かに劣化する「表記の崩れ」を、LLM 採点に頼らず決定論で検出する。
判定器をここに集約するのは、**同じ崩れを 2 箇所で別々に定義すると片方だけ直る**
ためで、実際に以下 2 系統が同一の判定を必要とする:

- :mod:`backend.free.learning.fewshot_pool` — 崩れた応答を手本に採用しない (入口ゲート)
- :mod:`backend.free.llm.quality_probe` — モデル切替時に崩れを検出する (事前ゲート)

アシスト採点は使わない。小型モデルは自分と同種の崩れを問題と認識できず、実測で
空白混入例の quality 平均 0.80 に対し正常例 0.89 と 0.09 しか差が付かなかった
(混入例に 0.95 が 4 件)。決定論でのみ分離できる。
"""

from __future__ import annotations

import re

#: 日本語の語間に混じった空白。
#:
#: 正常な日本語では和文文字が空白で分かたれることはない (実測: Qwen3.5-9B 時代の
#: 応答 96 件中 0 件)。一方 gemma-4-12b では 76〜83% に混入し、``temperature=0.0``
#: の貪欲法でも再現した — サンプリングではなく出力分布そのものの性質。
_JA_INTERWORD_SPACE_RE = re.compile(r"[ぁ-んァ-ヶ一-龥][ 　]+[ぁ-んァ-ヶ一-龥]")

#: コードブロック。整形済みコードやログ引用では半角空白が正常に現れる。
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)

#: 和文文字。応答が日本語かどうかの判定に使う。
_JA_CHAR_RE = re.compile(r"[ぁ-んァ-ヶ一-龥]")

#: 日本語判定の下限。これ未満の和文文字しか無い応答は英語応答等とみなし、
#: 語間空白チェックの母数から外す (英文の空白を誤検出しないため)。
_JA_MIN_CHARS = 20


def has_broken_ja_spacing(text: str) -> bool:
    """日本語部分に不自然な語間空白が混じっているかを判定する (純粋関数)。

    コードブロック内は対象外。整形済みコードやログの引用で半角空白が現れるのは
    正常なため、fence で囲まれた領域を除いてから判定する。
    """
    outside = _CODE_FENCE_RE.sub("\n", text)
    return bool(_JA_INTERWORD_SPACE_RE.search(outside))


def is_japanese_text(text: str) -> bool:
    """語間空白チェックの母数に含めてよい日本語応答かを判定する。

    和文文字が :data:`_JA_MIN_CHARS` 未満の応答 (英語応答 / 空応答 / コードのみ)
    は False。日本語で答えていないモデルを「空白混入 0%」と誤って合格させない
    ため、母数側でも呼び出し元が本関数で足切りする。
    """
    outside = _CODE_FENCE_RE.sub("\n", text)
    return len(_JA_CHAR_RE.findall(outside)) >= _JA_MIN_CHARS


#: 逐語エコー判定に載せるユーザー発言の下限文字数。これ未満は「はい」「OK」
#: 等の短い相槌で、応答が偶然同じ語で始まっただけの誤検出になりうる。
_ECHO_MIN_QUERY_CHARS = 8


def strip_echoed_query(response: str, query: str) -> str:
    """応答冒頭に混じったユーザー発言の逐語コピーを取り除く (純粋関数)。

    ユーザーの問いをそのまま繰り返してから答える崩れが実在する。単発なら
    見た目が悪いだけだが、その応答が記憶ノートとして保存されると、同じ問いで
    想起されて再生産され、繰り返し回数が増えていく (実インシデント
    2026-08-04 ライブ監査:「今日は何曜日ですか。」に対し同文を 5 回返して
    答えが出ない状態まで悪化した。汚染ノートを除去したら 5/5 で解消)。

    ``query`` が :data:`_ECHO_MIN_QUERY_CHARS` 未満のときは何もしない。
    """
    q = (query or "").strip()
    if len(q) < _ECHO_MIN_QUERY_CHARS:
        return response
    out = response.lstrip()
    while out.startswith(q):
        out = out[len(q):].lstrip()
    return out


#: 復唱の前に許す前置きの長さ。呼びかけ (「小川さん、」) を挟んでから復唱する
#: 形が実在するため、位置 0 の一致だけでは取りこぼす (実測 2026-08-04:
#: 「私の名前を覚えていますか。」に対し「小川さん、私の名前を覚えていますか。」)。
#: 短い前置きに限るのは、問いを引用してから答える正常な応答を巻き込まないため。
_ECHO_MAX_LEAD_CHARS = 16


def is_query_echo(response: str, query: str) -> bool:
    """応答がユーザー発言の逐語繰り返しだけで中身を持たないかを判定する。"""
    q = (query or "").strip()
    if len(q) < _ECHO_MIN_QUERY_CHARS:
        return False
    if not strip_echoed_query(response, q).strip():
        return True
    idx = response.find(q)
    if 0 < idx <= _ECHO_MAX_LEAD_CHARS:
        return not strip_echoed_query(response[idx:], q).strip()
    return False


#: 本文の後ろに付く定型の締め文。system プロンプトの出力形式が明示的に禁止して
#: いる (「応答の末尾に自己紹介・挨拶・『他にご質問はありますか?』等の定型文を
#: 追加しない」) にもかかわらず実際に出る。禁止だけでは消えないのは、違反応答が
#: 手本として採用され再生産されるため (実インシデント 2026-08-04 ライブ監査:
#: fitness 0.889 の最上位帯で few-shot に載っていた)。
_BOILERPLATE_CLOSING_RE = re.compile(
    r"(?:"
    r"他に(?:も)?(?:ご質問|ご不明な点|お困りのこと|何か)[^。\n]{0,12}"
    r"|何か(?:他に)?お手伝いできること[^。\n]{0,8}"
    r"|(?:Is|Was) there anything else"
    r"|Let me know if you (?:need|have)"
    r")[^。\n]{0,12}[?？]\s*$",
)

#: 締め文を除いた本文がこれ未満なら、応答そのものが定型の挨拶とみなす。挨拶への
#: 反射応答 (``agent.reactive``) は締め文が本体なので、違反として扱わない。
_CLOSING_MIN_BODY_CHARS = 24


def has_boilerplate_closing(text: str) -> bool:
    """本文の末尾に禁止された定型の締め文が付いているかを判定する (純粋関数)。

    挨拶だけの短い応答は対象外。締め文が応答の本体である反射応答まで違反に
    数えると、正常な挨拶が手本から一律に落ちてしまう。
    """
    body = text.rstrip()
    m = _BOILERPLATE_CLOSING_RE.search(body)
    if m is None:
        return False
    return len(body[:m.start()].strip()) >= _CLOSING_MIN_BODY_CHARS


#: 終端記号を保ったまま文へ切る。
_SENTENCE_RE = re.compile(r"[^。．.!！?？\n]+[。．.!！?？]?")

#: 疑問文の語尾。日本語は語尾が疑問を担い、疑問符が無いことが多い。
_INTERROGATIVE_TAIL_RE = re.compile(
    r"(?:ですか|ますか|でしょうか|ありますか|いますか|ませんか|だろうか)"
    r"[。．.]?$"
    r"|[?？]\s*$",
)


def carries_no_assertion(text: str) -> bool:
    """本文が疑問だけで、知識としての主張を含まないかを判定する (純粋関数)。

    過去セッションのユーザー発言はそのまま記憶ノートになる。問いだけのノート
    (「今日は何曜日ですか。」) は答えを含まないのに ``(過去の記録)`` として
    想起され、モデルがそれを回答として出力してしまう (実インシデント
    2026-08-04 ライブ監査)。1 文でも断定・依頼が混じっていれば False を返す
    ため、「私の誕生日は 3 月 14 日です。あと何日ですか。」のような
    事実を含む発言は残る。
    """
    sentences = [s.strip() for s in _SENTENCE_RE.findall(text or "")]
    sentences = [s for s in sentences if s]
    if not sentences:
        return True
    return all(_INTERROGATIVE_TAIL_RE.search(s) for s in sentences)


__all__ = [
    "carries_no_assertion",
    "has_boilerplate_closing",
    "has_broken_ja_spacing",
    "is_japanese_text",
    "is_query_echo",
    "strip_echoed_query",
]
