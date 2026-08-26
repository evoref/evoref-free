"""``local/`` に保存済みの「出力上限」注記をアシスタント本文から取り除く。

2026-08-25 まで、``finish_reason="length"`` の開示注記は content ストリームへ
``yield`` されており、応答本文の一部として履歴 / STM / 経験バッファへ保存されて
いた。保存された注記はモデル自身の出力として次ターンのプロンプトに入り、
「続けて」で **注記ごと逐語復唱** される (実インシデント: 2 ターン連続で同一の
末尾ブロック + 注記が返った)。

注記の出所はコード側で塞いだ (``StreamOutcome`` / ``sse.output_truncated``) が、
**既に保存された分は残る**。few-shot / RAG / 履歴経由で再生産されるため、
このスクリプトで浄化する。

使い方 (**サービス全停止中に実行すること**)::

    .\\scripts\\evoref-ctl.bat stop
    python scripts\\strip_truncation_notice.py            # dry-run (既定)
    python scripts\\strip_truncation_notice.py --apply    # 実際に書き換える

``--apply`` は書き換え前に ``<file>.bak`` を作る。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import sys

#: 注記の JSON エスケープ表記パターン。保存形は必ず
#: ``…本文\n\n> ⚠ <警告文>`` で、値の終端 (``"``) か次の ``\n`` の手前で終わる。
#:
#: 文字クラスから ``"`` と ``\`` を外すのが要点で、これで **1 つの JSON 文字列値
#: の内側** から出られなくなる。物理行で区切る書き方 (``[^\n]*``) は使えない —
#: history / experience は 1 レコードが 1 物理行なので、行末まで飲み込んで
#: 値の閉じ引用符ごと消し、JSON を壊す (実際に壊れることを確認済み)。
#:
#: 警告文は i18n 3 系統 (ja / en 完全形 / en フォールバック) を先頭語で束ねる。
NOTICE_RE = re.compile(
    r"(?:\\n)*>[ 　]*⚠[^\"\\]*?"
    r"(?:出力上限に達したため|Output token limit reached)"
    r"[^\"\\]*",
)

#: 走査対象。``local/logs`` は診断記録なので触らない (注記が出た事実の証跡)。
SCAN_SUFFIXES = (".json", ".jsonl")


def strip_notice(text: str) -> tuple[str, int]:
    """``text`` から注記を除去し ``(結果, 除去数)`` を返す。

    JSON を **文字列として** 処理する。パースして書き戻すとキー順・数値表記・
    インデントが変わり、巨大な差分になって「何を消したか」が読めなくなる。
    :data:`NOTICE_RE` は JSON 文字列値の内側で閉じるので構造は壊れない
    (呼出側が ``json.loads`` で検証する)。
    """
    return NOTICE_RE.subn("", text)


def _parses(path: Path, text: str) -> bool:
    """書き換え後のテキストが元と同じ形式で読めるか。"""
    try:
        if path.suffix.lower() == ".jsonl":
            for line in text.splitlines():
                if line.strip():
                    json.loads(line)
        else:
            json.loads(text)
    except ValueError:
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true",
        help="実際に書き換える (既定は dry-run)",
    )
    parser.add_argument(
        "--root", default="local",
        help="走査するルート (既定: local)",
    )
    args = parser.parse_args(argv)

    root = Path(args.root)
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 1

    changed = 0
    removed = 0
    failed = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SCAN_SUFFIXES:
            continue
        if "logs" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        new_text, n = strip_notice(text)
        if n == 0:
            continue
        # 書き戻す前に必ず構造を検証する。正規表現が JSON 文字列値の外へ
        # はみ出したらここで止まる (壊れたストアを黙って残さない)。
        if not _parses(path, new_text):
            print(f"SKIP (would break JSON): {path}", file=sys.stderr)
            failed += 1
            continue
        changed += 1
        removed += n
        print(f"{'strip' if args.apply else 'would strip'} {n:3d} from {path}")
        if not args.apply:
            continue
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
        path.write_text(new_text, encoding="utf-8", newline="")

    verb = "removed" if args.apply else "would remove"
    print(f"{verb} {removed} notice(s) across {changed} file(s)")
    if failed:
        print(f"{failed} file(s) skipped to avoid breaking JSON", file=sys.stderr)
        return 1
    if not args.apply and changed:
        print("re-run with --apply (services must be stopped first)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
