import os
from pathlib import Path

import llama_bench.gguf_cache as gguf_cache


def test_desired_gguf_files_returns_matching_models_and_best_mmproj() -> None:
    repo_files = [
        "model-Q4_K_M-00002-of-00002.gguf",
        "model-Q4_K_M-00001-of-00002.gguf",
        "model-Q5_K_M.gguf",
        "mmproj-Q8_0.gguf",
        "mmproj-Q4_K.gguf",
        "README.md",
    ]

    assert gguf_cache.desired_gguf_files(repo_files, "Q4_K_M") == [
        "model-Q4_K_M-00001-of-00002.gguf",
        "model-Q4_K_M-00002-of-00002.gguf",
        "mmproj-Q4_K.gguf",
    ]


def test_desired_gguf_files_returns_empty_without_matching_model() -> None:
    assert (
        gguf_cache.desired_gguf_files(["model-Q5_K_M.gguf", "mmproj-Q4_K.gguf"], "Q4_K_M")
        == []
    )


def test_local_gguf_files_indexes_snapshot_relative_paths(tmp_path: Path) -> None:
    snapshots = tmp_path / "models--org--repo" / "snapshots"
    model = snapshots / "rev" / "nested" / "model-Q4_K_M.gguf"
    model.parent.mkdir(parents=True)
    model.write_text("", encoding="utf-8")

    assert gguf_cache.local_gguf_files("org/repo", tmp_path) == {
        "nested/model-Q4_K_M.gguf": model
    }
    assert gguf_cache.local_gguf_files("org/missing", tmp_path) is None


def test_local_gguf_files_prefers_newest_duplicate_snapshot_relpath(
    tmp_path: Path,
) -> None:
    snapshots = tmp_path / "models--org--repo" / "snapshots"
    first = snapshots / "first" / "model-Q4_K_M.gguf"
    second = snapshots / "second" / "model-Q4_K_M.gguf"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text("", encoding="utf-8")
    second.write_text("", encoding="utf-8")
    second_mtime = second.stat().st_mtime_ns
    os.utime(first, ns=(second_mtime, second_mtime + 1_000_000_000))

    files = gguf_cache.local_gguf_files("org/repo", tmp_path)

    assert files is not None
    assert list(files) == ["model-Q4_K_M.gguf"]
    assert files["model-Q4_K_M.gguf"] == first


def test_local_gguf_files_breaks_duplicate_mtime_ties_by_path(tmp_path: Path) -> None:
    snapshots = tmp_path / "models--org--repo" / "snapshots"
    first = snapshots / "first" / "model-Q4_K_M.gguf"
    second = snapshots / "second" / "model-Q4_K_M.gguf"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text("", encoding="utf-8")
    second.write_text("", encoding="utf-8")
    first_stat = first.stat()
    os.utime(second, ns=(first_stat.st_atime_ns, first_stat.st_mtime_ns))

    files = gguf_cache.local_gguf_files("org/repo", tmp_path)

    assert files is not None
    assert list(files) == ["model-Q4_K_M.gguf"]
    assert files["model-Q4_K_M.gguf"] == first
