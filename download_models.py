#!/usr/bin/env python3
"""Download GGUF models to the HuggingFace cache used by llama.cpp.

Uses the huggingface_hub library to download models into the same
cache directory (~/.cache/huggingface/hub) that llama.cpp reads from.

Usage:
    # First-time setup (installs huggingface_hub):
    uv run download_models.py --install

    # Download all models:
    uv run download_models.py

    # Dry run (show what would be downloaded):
    uv run download_models.py --dry-run

    # Download only a specific group:
    uv run download_models.py --group gemma-4
    uv run download_models.py --group qwen3.5 --group qwen3.6

    # List available groups:
    uv run download_models.py --list-groups
"""

import argparse
import sys
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from huggingface_hub import list_repo_files, hf_hub_download, scan_cache_dir

    HAS_HF = True
except ImportError:
    HAS_HF = False

from results import load_models

MODELS = load_models()


def find_matching_files(repo_id: str, tag: str) -> list[str]:
    """List GGUF files in a repo matching the given quant tag."""
    try:
        all_files = list_repo_files(repo_id)
    except Exception as e:
        print(f"  [ERROR] Could not list files in {repo_id}: {e}", file=sys.stderr)
        return []

    pattern = re.compile(re.escape(tag) + r"[.-]", re.IGNORECASE)
    matches = [
        f
        for f in all_files
        if f.endswith(".gguf")
        and pattern.search(f)
        and "mmproj" not in f.lower()
        and "imatrix" not in f.lower()
    ]

    if not matches:
        ud_pattern = re.compile(r"UD-" + re.escape(tag) + r"[.-]", re.IGNORECASE)
        matches = [
            f
            for f in all_files
            if f.endswith(".gguf")
            and ud_pattern.search(f)
            and "mmproj" not in f.lower()
            and "imatrix" not in f.lower()
        ]

    if not matches:
        pattern2 = re.compile(r"[-.]" + re.escape(tag) + r"[-.]", re.IGNORECASE)
        matches = [
            f
            for f in all_files
            if f.endswith(".gguf")
            and pattern2.search(f)
            and "mmproj" not in f.lower()
            and "imatrix" not in f.lower()
        ]

    if not matches:
        pattern3 = re.compile(r"[-.]" + re.escape(tag) + r"\.gguf$", re.IGNORECASE)
        matches = [
            f
            for f in all_files
            if f.endswith(".gguf")
            and pattern3.search(f)
            and "mmproj" not in f.lower()
            and "imatrix" not in f.lower()
        ]

    return sorted(matches)


def find_mmproj_files(repo_id: str) -> list[str]:
    """Find the best mmproj file in a repo (prefer F16, fall back to BF16, then any)."""
    try:
        all_files = list_repo_files(repo_id)
    except Exception:
        return []
    mmproj_files = sorted([f for f in all_files if f.endswith(".gguf") and "mmproj" in f.lower()])
    if not mmproj_files:
        return []
    for preferred in ["mmproj-F16.gguf", "mmproj-f16.gguf"]:
        for f in mmproj_files:
            if f.endswith(preferred):
                return [f]
    for preferred in ["mmproj-BF16.gguf", "mmproj-bf16.gguf"]:
        for f in mmproj_files:
            if f.endswith(preferred):
                return [f]
    return [mmproj_files[0]]


def is_in_cache(repo_id: str, filenames: list[str]) -> bool:
    """Check if all files are already in the HF cache."""
    try:
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_id == repo_id:
                cached_files = {rf.file_name for rev in repo.revisions for rf in rev.files}
                return all(f.split("/")[-1] in cached_files for f in filenames)
    except Exception:
        pass
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Download GGUF models to the HF cache used by llama.cpp"
    )
    parser.add_argument(
        "-n", "--dry-run", action="store_true", help="Show what would be downloaded"
    )
    parser.add_argument(
        "-g",
        "--group",
        action="append",
        default=[],
        help="Only download models matching this group prefix",
    )
    parser.add_argument(
        "--list-groups", action="store_true", help="List available groups and exit"
    )
    parser.add_argument(
        "-p",
        "--parallel",
        type=int,
        default=4,
        help="Number of parallel downloads (default: 4)",
    )
    parser.add_argument("--install", action="store_true", help="Install huggingface_hub via uv")
    args = parser.parse_args()

    if args.install:
        import subprocess

        subprocess.run(["uv", "pip", "install", "huggingface_hub"], check=True)
        print("Installed huggingface_hub.")
        return

    if args.list_groups:
        groups = sorted(set(g for _, _, g in MODELS))
        print("Available groups:")
        for g in groups:
            count = sum(1 for _, _, gr in MODELS if gr == g)
            print(f"  {g} ({count} variants)")
        return

    if not HAS_HF:
        print("Error: huggingface_hub not installed.", file=sys.stderr)
        print("Install with: uv pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    print("=== llama.cpp Model Downloader ===")
    print(f"Total model variants: {len(MODELS)}")
    if args.group:
        print(f"Filtering to groups: {', '.join(args.group)}")
    print()

    stats = {"cached": 0, "downloaded": 0, "failed": 0, "skipped": 0, "missing": 0}

    tasks = []
    for i, (repo_id, tag, group) in enumerate(MODELS, 1):
        if args.group and not any(group.startswith(g) for g in args.group):
            stats["skipped"] += 1
            continue

        label = f"{repo_id}:{tag}"
        print(f"[{i}/{len(MODELS)}] {label}")

        files = find_matching_files(repo_id, tag)
        if not files:
            print(f"  [MISSING] No files matching '{tag}' in {repo_id}")
            stats["missing"] += 1
            print()
            continue

        mmproj_files = find_mmproj_files(repo_id)
        all_files = files + mmproj_files

        if is_in_cache(repo_id, all_files):
            print(f"  [CACHED] Already in cache ({len(files)} model + {len(mmproj_files)} mmproj)")
            stats["cached"] += 1
            print()
            continue

        print("  Files to download:")
        for f in files:
            print(f"    - {f}")
        for f in mmproj_files:
            print(f"    - {f} (mmproj)")

        if args.dry_run:
            print(f"  [DRY RUN] Would download {len(all_files)} file(s)")
            stats["downloaded"] += 1
            print()
            continue

        tasks.append((repo_id, all_files, label))

    if not tasks:
        print("=== Done ===")
        print(
            f"Cached: {stats['cached']}, Downloaded: {stats['downloaded']}, "
            f"Missing: {stats['missing']}, Failed: {stats['failed']}, Skipped: {stats['skipped']}"
        )
        return

    print(f"\n=== Downloading {len(tasks)} model(s) with {args.parallel} parallel worker(s) ===\n")

    def download_task(item):
        repo_id, all_files, label = item
        try:
            for f in all_files:
                hf_hub_download(repo_id=repo_id, filename=f)
            return (label, True, None)
        except Exception as e:
            return (label, False, e)

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(download_task, t): t for t in tasks}
        for future in as_completed(futures):
            label, ok, err = future.result()
            if ok:
                print(f"  [OK] {label}")
                stats["downloaded"] += 1
            else:
                print(f"  [FAILED] {label}: {err}", file=sys.stderr)
                stats["failed"] += 1

    print("=== Done ===")
    print(
        f"Cached: {stats['cached']}, Downloaded: {stats['downloaded']}, "
        f"Missing: {stats['missing']}, Failed: {stats['failed']}, Skipped: {stats['skipped']}"
    )


if __name__ == "__main__":
    main()
