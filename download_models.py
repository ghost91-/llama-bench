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
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import list_repo_files, snapshot_download

from hf_gguf import find_best_mmproj_file, find_matching_model_files
from results import load_models


def get_repo_files(
    repo_id: str, repo_files_cache: dict[str, list[str] | None]
) -> list[str] | None:
    if repo_id not in repo_files_cache:
        try:
            repo_files_cache[repo_id] = list_repo_files(repo_id)
        except Exception as e:
            print(f"  [ERROR] Could not list files in {repo_id}: {e}", file=sys.stderr)
            repo_files_cache[repo_id] = None
    return repo_files_cache[repo_id]



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
        help="Number of repos to download in parallel (default: 4)",
    )
    args = parser.parse_args()

    if args.parallel < 1:
        parser.error("--parallel must be >= 1")

    models = load_models()

    if args.list_groups:
        groups = sorted(set(g for _, _, g in models))
        print("Available groups:")
        for g in groups:
            count = sum(1 for _, _, gr in models if gr == g)
            print(f"  {g} ({count} variants)")
        return

    print("=== llama.cpp Model Downloader ===")
    print(f"Total model variants: {len(models)}")
    if args.group:
        print(f"Filtering to groups: {', '.join(args.group)}")
    print()

    stats = {"downloaded": 0, "failed": 0, "skipped": 0, "missing": 0}
    repo_files_cache: dict[str, list[str] | None] = {}

    repo_tasks = {}
    for i, (repo_id, tag, group) in enumerate(models, 1):
        if args.group and not any(group.startswith(g) for g in args.group):
            stats["skipped"] += 1
            continue

        label = f"{repo_id}:{tag}"
        print(f"[{i}/{len(models)}] {label}")

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

        if args.dry_run:
            print(f"  [DRY RUN] Would download/verify {len(all_files)} file(s)")
            stats["downloaded"] += 1
            print()
            continue

        repo_task = repo_tasks.setdefault(repo_id, {"files": set(), "labels": []})
        repo_task["files"].update(all_files)
        repo_task["labels"].append(label)

    if not repo_tasks:
        print(
            f"=== Done ===\n"
            f"Downloaded: {stats['downloaded']}, "
            f"Missing: {stats['missing']}, Failed: {stats['failed']}, Skipped: {stats['skipped']}"
        )
        return

    total_models = sum(len(task["labels"]) for task in repo_tasks.values())
    print(
        f"\n=== Downloading {total_models} model(s) from {len(repo_tasks)} repo(s) "
        f"with {args.parallel} parallel repo worker(s) ===\n"
    )

    repo_items = list(repo_tasks.items())
    for i, (repo_id, task) in enumerate(repo_items, 1):
        print(
            f"[{i}/{len(repo_items)}] {repo_id} ({len(task['files'])} file(s), {len(task['labels'])} model(s))"
        )

    def download_repo(item):
        repo_id, task = item
        snapshot_download(repo_id=repo_id, allow_patterns=sorted(task["files"]))
        return task["labels"]

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(download_repo, item): item for item in repo_items}
        for future in as_completed(futures):
            try:
                labels = future.result()
                for label in labels:
                    print(f"  [OK] {label}")
                    stats["downloaded"] += 1
            except Exception as err:
                _repo_id, task = futures[future]
                for label in task["labels"]:
                    print(f"  [FAILED] {label}: {err}", file=sys.stderr)
                    stats["failed"] += 1
            print()

    print("=== Done ===")
    print(
        f"Downloaded: {stats['downloaded']}, "
        f"Missing: {stats['missing']}, Failed: {stats['failed']}, Skipped: {stats['skipped']}"
    )


if __name__ == "__main__":
    main()
