#!/usr/bin/env python3
"""Remove cached HF model files that are not needed by models.toml.

Usage:
    # Preview deletions:
    uv run cleanup_model_cache.py --dry-run

    # Delete unlisted cached models:
    uv run cleanup_model_cache.py
"""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import cast

from huggingface_hub import scan_cache_dir

from llama_bench.gguf_cache import desired_gguf_files
from llama_bench.results import load_models
from llama_bench.schema_types import (
    CachedFileInfo,
    CachedRepoInfo,
    CachedRevisionInfo,
    HFCacheInfo,
    ModelRecord,
)


def format_size(size: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def build_desired_tags(models: list[ModelRecord]) -> dict[str, list[str]]:
    desired_tags: defaultdict[str, list[str]] = defaultdict(list)
    for repo_id, tag, _group, _pinned in models:
        desired_tags[repo_id].append(tag)
    return dict(desired_tags)


def build_keep_files_by_repo(
    cache_info: HFCacheInfo, desired_tags: dict[str, list[str]]
) -> dict[str, set[str]]:
    keep_files_by_repo: dict[str, set[str]] = {}
    for repo in cache_info.repos:
        if repo.repo_type != "model" or repo.repo_id not in desired_tags:
            continue
        repo_files: list[str] = sorted(
            {file.file_name for revision in repo.revisions for file in revision.files}
        )
        if not any(file.endswith(".gguf") for file in repo_files):
            continue
        keep_files: set[str] = set()
        for tag in desired_tags[repo.repo_id]:
            keep_files.update(desired_gguf_files(repo_files, tag))
        keep_files_by_repo[repo.repo_id] = keep_files
    return keep_files_by_repo


def repo_has_cached_gguf(repo: CachedRepoInfo) -> bool:
    return any(
        file.file_name.endswith(".gguf") for revision in repo.revisions for file in revision.files
    )


def prune_empty_dirs(start: Path, stop_before: Path) -> None:
    current = start
    while current != stop_before:
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove cached HF GGUF files that are not listed in models.toml"
    )
    parser.add_argument("-n", "--dry-run", action="store_true", help="Show what would be deleted")
    args = parser.parse_args()

    models = load_models()
    cache_info = cast(HFCacheInfo, scan_cache_dir())
    desired_tags = build_desired_tags(models)
    keep_files_by_repo = build_keep_files_by_repo(cache_info, desired_tags)

    revisions_to_delete: list[str] = []
    file_entries_to_delete: list[tuple[CachedRepoInfo, CachedRevisionInfo, CachedFileInfo]] = []
    blobs_to_keep: set[Path] = set()
    seen_file_paths: set[Path] = set()

    for repo in sorted(cache_info.repos, key=lambda repo: repo.repo_id):
        if repo.repo_type != "model" or not repo_has_cached_gguf(repo):
            continue
        if repo.repo_id not in desired_tags:
            revisions_to_delete.extend(revision.commit_hash for revision in repo.revisions)
            continue

        keep_files = keep_files_by_repo.get(repo.repo_id, set())
        for revision in repo.revisions:
            for file in revision.files:
                if file.file_path in seen_file_paths:
                    continue
                seen_file_paths.add(file.file_path)
                if not file.file_name.endswith(".gguf") or file.file_name in keep_files:
                    blobs_to_keep.add(file.blob_path)
                    continue
                file_entries_to_delete.append((repo, revision, file))

    delete_strategy = cache_info.delete_revisions(*revisions_to_delete)
    blobs_to_delete: dict[Path, int] = {}
    for _repo, _revision, file in file_entries_to_delete:
        if file.blob_path not in blobs_to_keep:
            blobs_to_delete[file.blob_path] = file.size_on_disk

    partial_freed_size = sum(blobs_to_delete.values())
    total_freed_size = delete_strategy.expected_freed_size + partial_freed_size

    print("=== HF Model Cache Cleanup ===")
    print(f"Configured model variants: {len(models)}")
    print(f"Unlisted cached repos to delete: {len(delete_strategy.repos)}")
    print(f"Extra cached GGUF files to delete: {len(file_entries_to_delete)}")
    print(f"Expected space to free: {format_size(total_freed_size)}")
    print()

    if delete_strategy.repos:
        print("Cached repos to delete:")
        for path in sorted(delete_strategy.repos):
            print(f"  {path}")
        print()

    if file_entries_to_delete:
        print("Extra cached GGUF files to delete:")
        for repo, _revision, file in sorted(
            file_entries_to_delete, key=lambda item: (item[0].repo_id, item[2].file_name)
        ):
            print(f"  {repo.repo_id}: {file.file_name}")
        print()

    if not delete_strategy.repos and not file_entries_to_delete:
        print("Nothing to delete.")
        return

    if args.dry_run:
        print("Dry run only. No files were deleted.")
        return

    delete_strategy.execute()

    for _repo, revision, file in file_entries_to_delete:
        if file.file_path.exists() or file.file_path.is_symlink():
            file.file_path.unlink()
            prune_empty_dirs(file.file_path.parent, revision.snapshot_path.parent)

    for blob_path in sorted(blobs_to_delete):
        if blob_path.exists():
            blob_path.unlink()

    print(f"Deleted cached files. Freed about {format_size(total_freed_size)}.")


if __name__ == "__main__":
    main()
