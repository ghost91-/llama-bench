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
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Protocol, TypedDict, cast

import huggingface_hub
from huggingface_hub import list_repo_files, scan_cache_dir

from llama_bench.gguf_cache import desired_gguf_files
from llama_bench.results import load_models
from llama_bench.schema_types import HFCacheInfo


class RepoDownloadTask(TypedDict):
    files: set[str]
    labels: list[str]


class SnapshotDownloadFn(Protocol):
    def __call__(self, repo_id: str, *, allow_patterns: list[str] | None = None) -> str: ...


snapshot_download_fn = cast(SnapshotDownloadFn, huggingface_hub.snapshot_download)


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



def evict_old_revisions() -> int:
    try:
        cache_info = cast(HFCacheInfo, scan_cache_dir())
    except Exception:
        return 0
    old_hashes: list[str] = []
    for repo in cache_info.repos:
        if len(repo.revisions) <= 1:
            continue
        revs = sorted(repo.revisions, key=lambda r: r.last_modified, reverse=True)
        old_hashes.extend(r.commit_hash for r in revs[1:])
    if not old_hashes:
        return 0
    strategy = cache_info.delete_revisions(*old_hashes)
    strategy.execute()
    return strategy.expected_freed_size


def print_stats(stats: dict[str, int]) -> None:
    print("=== Done ===")
    print(
        f"Downloaded: {stats['downloaded']}, "
        f"Missing: {stats['missing']}, Failed: {stats['failed']}, Skipped: {stats['skipped']}"
    )


def main() -> None:
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
        groups = Counter(g for _, _, g in models)
        print("Available groups:")
        for g, count in sorted(groups.items()):
            print(f"  {g} ({count} variants)")
        return

    print("=== llama.cpp Model Downloader ===")
    print(f"Total model variants: {len(models)}")
    if args.group:
        print(f"Filtering to groups: {', '.join(args.group)}")
    print()

    freed = evict_old_revisions()
    if freed > 0:
        freed_gib = freed / 1024**3
        print(f"Evicted old cache revisions, freed {freed_gib:.1f} GiB")
    print()

    stats: dict[str, int] = {"downloaded": 0, "failed": 0, "skipped": 0, "missing": 0}
    repo_files_cache: dict[str, list[str] | None] = {}

    repo_tasks: dict[str, RepoDownloadTask] = {}
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

        files = desired_gguf_files(repo_files, tag)
        if not files:
            print(f"  [MISSING] No files matching '{tag}' in {repo_id}")
            stats["missing"] += 1
            print()
            continue

        if args.dry_run:
            print(f"  [DRY RUN] Would download/verify {len(files)} file(s)")
            stats["downloaded"] += 1
            print()
            continue

        repo_task = repo_tasks.setdefault(repo_id, RepoDownloadTask(files=set(), labels=[]))
        repo_task["files"].update(files)
        repo_task["labels"].append(label)

    if not repo_tasks:
        print_stats(stats)
        return

    total_models = sum(len(task["labels"]) for task in repo_tasks.values())
    print(
        f"\n=== Downloading {total_models} model(s) from {len(repo_tasks)} repo(s) "
        f"with {args.parallel} parallel repo worker(s) ===\n"
    )

    repo_items: list[tuple[str, RepoDownloadTask]] = list(repo_tasks.items())
    for i, (repo_id, task) in enumerate(repo_items, 1):
        print(
            f"[{i}/{len(repo_items)}] {repo_id} ({len(task['files'])} file(s), {len(task['labels'])} model(s))"
        )

    def download_repo(item: tuple[str, RepoDownloadTask]) -> list[str]:
        repo_id, task = item
        snapshot_download_fn(repo_id=repo_id, allow_patterns=sorted(task["files"]))
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

    freed = evict_old_revisions()
    if freed > 0:
        freed_gib = freed / 1024**3
        print(f"Evicted old cache revisions, freed {freed_gib:.1f} GiB")

    print_stats(stats)


if __name__ == "__main__":
    main()
