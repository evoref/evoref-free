"""モデルダウンロードスクリプト

Hugging Face から GGUF モデルをダウンロードする。
埋め込みモデルは llama.cpp 経由で使用する GGUF 版のみをサポートする。
"""

import argparse
import sys
import urllib.request
from pathlib import Path


# ── デフォルトモデル定義 ──

EMBED_GGUF_MODEL = {
    "repo": "Qwen/Qwen3-Embedding-0.6B-GGUF",
    "file": "Qwen3-Embedding-0.6B-Q8_0.gguf",
    "dest": "models",
}

LLM_MODEL = {
    "repo": "unsloth/Qwen3.5-9B-GGUF",
    "file": "Qwen3.5-9B-Q4_K_M.gguf",
    "dest": "models",
}

ASSIST_MODEL = {
    "repo": "unsloth/Qwen3.5-4B-GGUF",
    "file": "Qwen3.5-4B-Q4_K_M.gguf",
    "dest": "models",
}

HF_BASE = "https://huggingface.co"


def download_file(url: str, dest: Path, desc: str = "", force: bool = False) -> None:
    """ファイルをダウンロード（プログレス付き）"""
    if dest.exists():
        if not force:
            print(f"  [skip] {desc or dest.name} already exists")
            return
        print(f"  [force] Removing existing {desc or dest.name}")
        dest.unlink()

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [download] {desc or dest.name}...")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "evoref/0.1.0"})
        with urllib.request.urlopen(req) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1MB

            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded * 100 // total
                        mb = downloaded / (1024 * 1024)
                        total_mb = total / (1024 * 1024)
                        print(f"\r  [{pct:3d}%] {mb:.1f} / {total_mb:.1f} MB", end="", flush=True)

            print()  # newline after progress
    except Exception as e:
        # 失敗時は不完全ファイルを削除
        if dest.exists():
            dest.unlink()
        raise RuntimeError(f"Download failed: {e}") from e


def download_embed(project_root: Path, force: bool = False) -> None:
    """埋め込みモデル（GGUF）をダウンロード"""
    repo = EMBED_GGUF_MODEL["repo"]
    filename = EMBED_GGUF_MODEL["file"]
    dest = project_root / EMBED_GGUF_MODEL["dest"] / filename

    print(f"Downloading embedding model (GGUF): {repo}/{filename}")
    url = f"{HF_BASE}/{repo}/resolve/main/{filename}"
    download_file(url, dest, filename, force=force)
    print("  Embedding model (GGUF) ready")


def download_llm(project_root: Path, force: bool = False) -> None:
    """GGUF モデルをダウンロード"""
    repo = LLM_MODEL["repo"]
    filename = LLM_MODEL["file"]
    dest = project_root / LLM_MODEL["dest"] / filename

    print(f"Downloading LLM model: {repo}/{filename}")
    url = f"{HF_BASE}/{repo}/resolve/main/{filename}"
    download_file(url, dest, filename, force=force)
    print("  LLM model ready")


def download_assist(project_root: Path, force: bool = False) -> None:
    """アシストモデル（GGUF）をダウンロード"""
    repo = ASSIST_MODEL["repo"]
    filename = ASSIST_MODEL["file"]
    dest = project_root / ASSIST_MODEL["dest"] / filename

    print(f"Downloading assist model: {repo}/{filename}")
    url = f"{HF_BASE}/{repo}/resolve/main/{filename}"
    download_file(url, dest, filename, force=force)
    print("  Assist model ready")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download models for evoref")
    parser.add_argument("--embed", action="store_true", help="Download embedding model (GGUF)")
    parser.add_argument("--llm", action="store_true", help="Download LLM model (GGUF)")
    parser.add_argument("--assist", action="store_true", help="Download assist model (GGUF)")
    parser.add_argument("--all", action="store_true", help="Download all models")
    parser.add_argument("--force", action="store_true", help="Force re-download even if files already exist")
    args = parser.parse_args()

    if not (args.embed or args.llm or args.assist or args.all):
        parser.print_help()
        sys.exit(1)

    project_root = Path(__file__).parent.parent

    force = args.force
    if force:
        print("[force] Force re-download mode enabled")

    try:
        if args.embed or args.all:
            download_embed(project_root, force=force)
        if args.llm or args.all:
            download_llm(project_root, force=force)
        if args.assist or args.all:
            download_assist(project_root, force=force)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print("\nAll downloads complete.")


if __name__ == "__main__":
    main()
