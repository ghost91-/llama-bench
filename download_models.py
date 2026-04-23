#!/usr/bin/env python3
"""Download GGUF models to the HuggingFace cache used by llama.cpp.

Uses the huggingface_hub library to download models into the same
cache directory (~/.cache/huggingface/hub) that llama.cpp reads from.

Usage:
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

from huggingface_hub import list_repo_files, scan_cache_dir, snapshot_download

from hf_gguf import find_best_mmproj_file, find_matching_model_files
from results import load_models

MODELS = load_models()
def get_repo_files(repo_id: str, repo_files_cache: dict[str, list[str] | None]) -> list[str] | None:
    if repo_id not in repo_files_cache:
        try:
            repo_files_cache[repo_id] = list_repo_files(repo_id)
        except Exception as e:
            print(f"  [ERROR] Could not list files in {repo_id}: {e}", file=sys.stderr)
            repo_files_cache[repo_id] = None
    return repo_files_cache[repo_id]


def build_cache_index() -> dict[str, set[str]]:
    """Index cached HF files by repo id."""
    try:
        cache_info = scan_cache_dir()
        return {
            repo.repo_id: {rf.file_name for rev in repo.revisions for rf in rev.files}
            for repo in cache_info.repos
        }
    except Exception:
        return {}


def is_in_cache(repo_id: str, filenames: list[str], cache_index: dict[str, set[str]]) -> bool:
    """Check if all files are already in the HF cache."""
    cached_files = cache_index.get(repo_id)
    if cached_files is None:
        return False
    return all(f.split("/")[-1] in cached_files for f in filenames)


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
    args = parser.parse_args()

    if args.list_groups:
        groups = sorted(set(g for _, _, g in MODELS))
        print("Available groups:")
        for g in groups:
            count = sum(1 for _, _, gr in MODELS if gr == g)
            print(f"  {g} ({count} variants)")
        return

    print("=== llama.cpp Model Downloader ===")
    print(f"Total model variants: {len(MODELS)}")
    if args.group:
        print(f"Filtering to groups: {', '.join(args.group)}")
    print()

    stats = {"cached": 0, "downloaded": 0, "failed": 0, "skipped": 0, "missing": 0}
    cache_index = build_cache_index()
    repo_files_cache: dict[str, list[str] | None] = {}

    repo_tasks = {}
    for i, (repo_id, tag, group) in enumerate(MODELS, 1):
        if args.group and not any(group.startswith(g) for g in args.group):
            stats["skipped"] += 1
            continue

        label = f"{repo_id}:{tag}"
        print(f"[{i}/{len(MODELS)}] {label}")

        repo_files = get_repo_files(repo_id, repo_files_cache)
        if repo_files is None:
            stats["failed"] += 1
            print()
            continue

        files = find_matching_model_files(repo_files, tag)
        if not files:
            print(f"  [MISSING] No files matching '{tag}' in {repo_id}")
            stats["missing"] += 1
            print()
            continue

        mmproj = find_best_mmproj_file(repo_files, files[0])
        mmproj_files = [mmproj] if mmproj else []
        all_files = files + mmproj_files

        if is_in_cache(repo_id, all_files, cache_index):
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

        repo_task = repo_tasks.setdefault(repo_id, {"files": set(), "labels": []})
        repo_task["files"].update(all_files)
        repo_task["labels"].append(label)

    if not repo_tasks:
        print("=== Done ===")
        print(
            f"Cached: {stats['cached']}, Downloaded: {stats['downloaded']}, "
            f"Missing: {stats['missing']}, Failed: {stats['failed']}, Skipped: {stats['skipped']}"
        )
        return

    total_models = sum(len(task["labels"]) for task in repo_tasks.values())
    print(f"\n=== Downloading {total_models} model(s) from {len(repo_tasks)} repo(s) ===\n")

    for i, (repo_id, task) in enumerate(repo_tasks.items(), 1):
        files = sorted(task["files"])
        labels = task["labels"]
        print(
            f"[{i}/{len(repo_tasks)}] {repo_id} "
            f"({len(files)} file(s), {len(labels)} model(s))"
        )
        try:
            snapshot_download(repo_id=repo_id, allow_patterns=files)
            for label in labels:
                print(f"  [OK] {label}")
                stats["downloaded"] += 1
        except Exception as err:
            for label in labels:
                print(f"  [FAILED] {label}: {err}", file=sys.stderr)
                stats["failed"] += 1
        print()

    print("=== Done ===")
    print(
        f"Cached: {stats['cached']}, Downloaded: {stats['downloaded']}, "
        f"Missing: {stats['missing']}, Failed: {stats['failed']}, Skipped: {stats['skipped']}"
    )


if __name__ == "__main__":
    main()
