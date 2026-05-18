from dataclasses import dataclass
from pathlib import Path
from typing import cast

from pytest import CaptureFixture, MonkeyPatch

import cleanup_model_cache
from llama_bench.schema_types import CachedRepoInfo, HFCacheInfo


@dataclass(frozen=True)
class FakeFile:
    file_name: str
    file_path: Path
    blob_path: Path
    size_on_disk: int


@dataclass
class FakeCleanupRevision:
    commit_hash: str
    files: list[FakeFile]
    last_modified: float
    snapshot_path: Path


@dataclass
class FakeCleanupRepo:
    repo_type: str
    repo_id: str
    revisions: list[FakeCleanupRevision]


class FakeCleanupStrategy:
    def __init__(self, revisions: tuple[str, ...]) -> None:
        self.expected_freed_size = 100 * len(revisions)
        self.repos = [Path(f"/cache/{revision}") for revision in revisions]
        self.executed = False

    def execute(self) -> None:
        self.executed = True


class FakeCleanupCacheInfo:
    def __init__(self, repos: list[FakeCleanupRepo]) -> None:
        self.repos = repos
        self.deleted: tuple[str, ...] = ()
        self.strategy = FakeCleanupStrategy(())

    def delete_revisions(self, *revisions: str) -> FakeCleanupStrategy:
        self.deleted = revisions
        self.strategy = FakeCleanupStrategy(revisions)
        return self.strategy


def test_format_size_and_desired_tags() -> None:
    assert cleanup_model_cache.format_size(0) == "0 B"
    assert cleanup_model_cache.format_size(1023) == "1023 B"
    assert cleanup_model_cache.format_size(1024) == "1.0 KiB"
    assert cleanup_model_cache.format_size(2 * 1024**3) == "2.0 GiB"
    assert cleanup_model_cache.build_desired_tags(
        [("repo/a", "Q4_K_M", "group", False), ("repo/a", "Q5_K_M", "group", True)]
    ) == {"repo/a": ["Q4_K_M", "Q5_K_M"]}


def test_build_keep_files_by_repo_keeps_matching_models_and_best_mmproj(tmp_path: Path) -> None:
    revision = FakeCleanupRevision(
        "rev",
        [
            FakeFile("model-Q4_K_M.gguf", tmp_path / "model", tmp_path / "blob-model", 10),
            FakeFile("model-Q5_K_M.gguf", tmp_path / "model5", tmp_path / "blob-model5", 10),
            FakeFile("mmproj-Q4_K.gguf", tmp_path / "mmproj", tmp_path / "blob-mmproj", 10),
        ],
        1.0,
        tmp_path / "snapshots" / "rev",
    )
    cache_info = FakeCleanupCacheInfo([FakeCleanupRepo("model", "repo/a", [revision])])

    assert cleanup_model_cache.build_keep_files_by_repo(cast(HFCacheInfo, cache_info), {"repo/a": ["Q4_K_M"]}) == {
        "repo/a": {"model-Q4_K_M.gguf", "mmproj-Q4_K.gguf"}
    }


def test_repo_has_cached_gguf_only_checks_model_files(tmp_path: Path) -> None:
    repo = FakeCleanupRepo(
        "model",
        "repo/a",
        [
            FakeCleanupRevision(
                "rev",
                [FakeFile("README.md", tmp_path / "README.md", tmp_path / "blob", 1)],
                1.0,
                tmp_path,
            )
        ],
    )
    assert cleanup_model_cache.repo_has_cached_gguf(cast(CachedRepoInfo, repo)) is False
    repo.revisions[0].files.append(FakeFile("model.gguf", tmp_path / "model.gguf", tmp_path / "blob2", 1))
    assert cleanup_model_cache.repo_has_cached_gguf(cast(CachedRepoInfo, repo)) is True


def test_prune_empty_dirs_stops_before_boundary(tmp_path: Path) -> None:
    stop = tmp_path / "snapshots"
    leaf = stop / "rev" / "nested"
    leaf.mkdir(parents=True)

    cleanup_model_cache.prune_empty_dirs(leaf, stop)

    assert not leaf.exists()
    assert stop.exists()


def _make_cache_for_main(tmp_path: Path) -> tuple[FakeCleanupCacheInfo, Path, Path, Path]:
    wanted_file = tmp_path / "models--repo--a" / "snapshots" / "rev" / "model-Q4_K_M.gguf"
    extra_file = tmp_path / "models--repo--a" / "snapshots" / "rev" / "model-Q8_0.gguf"
    shared_file = tmp_path / "models--repo--a" / "snapshots" / "rev" / "duplicate-Q6_K.gguf"
    for path in (wanted_file, extra_file, shared_file):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("data", encoding="utf-8")
    wanted_blob = tmp_path / "blobs" / "wanted"
    extra_blob = tmp_path / "blobs" / "extra"
    shared_blob = tmp_path / "blobs" / "shared"
    for path in (wanted_blob, extra_blob, shared_blob):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("blob", encoding="utf-8")

    listed_repo = FakeCleanupRepo(
        "model",
        "repo/a",
        [
            FakeCleanupRevision(
                "rev-a",
                [
                    FakeFile("model-Q4_K_M.gguf", wanted_file, wanted_blob, 4),
                    FakeFile("model-Q8_0.gguf", extra_file, extra_blob, 8),
                    FakeFile("duplicate-Q6_K.gguf", shared_file, wanted_blob, 16),
                ],
                1.0,
                tmp_path / "models--repo--a" / "snapshots" / "rev",
            )
        ],
    )
    unlisted_repo = FakeCleanupRepo(
        "model",
        "repo/old",
        [
            FakeCleanupRevision(
                "rev-old",
                [FakeFile("old-Q4_K_M.gguf", tmp_path / "old", shared_blob, 12)],
                1.0,
                tmp_path / "old-snapshot",
            )
        ],
    )
    return FakeCleanupCacheInfo([listed_repo, unlisted_repo]), extra_file, extra_blob, wanted_blob


def test_main_dry_run_reports_without_deleting(
    monkeypatch: MonkeyPatch, tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    cache_info, extra_file, extra_blob, _wanted_blob = _make_cache_for_main(tmp_path)
    monkeypatch.setattr(cleanup_model_cache, "load_models", lambda: [("repo/a", "Q4_K_M", "g", False)])
    monkeypatch.setattr(cleanup_model_cache, "scan_cache_dir", lambda: cache_info)
    monkeypatch.setattr("sys.argv", ["cleanup_model_cache.py", "--dry-run"])

    cleanup_model_cache.main()

    captured = capsys.readouterr()
    assert cache_info.deleted == ("rev-old",)
    assert "Extra cached GGUF files to delete: 2" in captured.out
    assert "Dry run only. No files were deleted." in captured.out
    assert extra_file.exists()
    assert extra_blob.exists()
    assert cache_info.strategy.executed is False


def test_main_deletes_extra_files_and_unreferenced_blobs(
    monkeypatch: MonkeyPatch, tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    cache_info, extra_file, extra_blob, wanted_blob = _make_cache_for_main(tmp_path)
    monkeypatch.setattr(cleanup_model_cache, "load_models", lambda: [("repo/a", "Q4_K_M", "g", False)])
    monkeypatch.setattr(cleanup_model_cache, "scan_cache_dir", lambda: cache_info)
    monkeypatch.setattr("sys.argv", ["cleanup_model_cache.py"])

    cleanup_model_cache.main()

    captured = capsys.readouterr()
    assert cache_info.strategy.executed is True
    assert not extra_file.exists()
    assert not extra_blob.exists()
    assert wanted_blob.exists()
    assert "Deleted cached files. Freed about" in captured.out


def test_main_reports_nothing_to_delete(
    monkeypatch: MonkeyPatch, tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    model_file = tmp_path / "models--repo--a" / "snapshots" / "rev" / "model-Q4_K_M.gguf"
    model_file.parent.mkdir(parents=True)
    model_file.write_text("data", encoding="utf-8")
    blob = tmp_path / "blobs" / "model"
    blob.parent.mkdir()
    blob.write_text("blob", encoding="utf-8")
    cache_info = FakeCleanupCacheInfo(
        [
            FakeCleanupRepo(
                "model",
                "repo/a",
                [
                    FakeCleanupRevision(
                        "rev-a",
                        [FakeFile("model-Q4_K_M.gguf", model_file, blob, 4)],
                        1.0,
                        model_file.parent,
                    )
                ],
            )
        ]
    )
    monkeypatch.setattr(cleanup_model_cache, "load_models", lambda: [("repo/a", "Q4_K_M", "g", False)])
    monkeypatch.setattr(cleanup_model_cache, "scan_cache_dir", lambda: cache_info)
    monkeypatch.setattr("sys.argv", ["cleanup_model_cache.py"])

    cleanup_model_cache.main()

    captured = capsys.readouterr()
    assert cache_info.deleted == ()
    assert cache_info.strategy.executed is False
    assert "Nothing to delete." in captured.out


def test_main_ignores_non_model_repos_and_deletes_unlisted_model_repos(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    model_revision = FakeCleanupRevision(
        "rev-model",
        [FakeFile("model-Q4_K_M.gguf", tmp_path / "model.gguf", tmp_path / "model-blob", 4)],
        1.0,
        tmp_path / "model-snapshot",
    )
    dataset_revision = FakeCleanupRevision(
        "rev-dataset",
        [FakeFile("data-Q4_K_M.gguf", tmp_path / "data.gguf", tmp_path / "data-blob", 4)],
        1.0,
        tmp_path / "dataset-snapshot",
    )
    cache_info = FakeCleanupCacheInfo(
        [
            FakeCleanupRepo("dataset", "repo/data", [dataset_revision]),
            FakeCleanupRepo("model", "repo/old", [model_revision]),
        ]
    )

    def load_no_models() -> list[tuple[str, str, str, bool]]:
        return []

    monkeypatch.setattr(cleanup_model_cache, "load_models", load_no_models)
    monkeypatch.setattr(cleanup_model_cache, "scan_cache_dir", lambda: cache_info)
    monkeypatch.setattr("sys.argv", ["cleanup_model_cache.py"])

    cleanup_model_cache.main()

    assert cache_info.deleted == ("rev-model",)
    assert cache_info.strategy.executed is True


def test_main_deletes_stale_ggufs_when_listed_repo_has_no_matching_quant(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    stale_file = tmp_path / "models--repo--a" / "snapshots" / "rev" / "model-Q8_0.gguf"
    readme_file = stale_file.with_name("README.md")
    stale_file.parent.mkdir(parents=True)
    stale_file.write_text("data", encoding="utf-8")
    readme_file.write_text("readme", encoding="utf-8")
    stale_blob = tmp_path / "blobs" / "stale"
    readme_blob = tmp_path / "blobs" / "readme"
    stale_blob.parent.mkdir()
    stale_blob.write_text("blob", encoding="utf-8")
    readme_blob.write_text("blob", encoding="utf-8")
    cache_info = FakeCleanupCacheInfo(
        [
            FakeCleanupRepo(
                "model",
                "repo/a",
                [
                    FakeCleanupRevision(
                        "rev-a",
                        [
                            FakeFile("model-Q8_0.gguf", stale_file, stale_blob, 8),
                            FakeFile("README.md", readme_file, readme_blob, 1),
                        ],
                        1.0,
                        stale_file.parent,
                    )
                ],
            )
        ]
    )
    monkeypatch.setattr(cleanup_model_cache, "load_models", lambda: [("repo/a", "Q4_K_M", "g", False)])
    monkeypatch.setattr(cleanup_model_cache, "scan_cache_dir", lambda: cache_info)
    monkeypatch.setattr("sys.argv", ["cleanup_model_cache.py"])

    cleanup_model_cache.main()

    assert not stale_file.exists()
    assert not stale_blob.exists()
    assert readme_file.exists()
    assert readme_blob.exists()


def test_main_deletes_stale_mmproj_and_retains_selected_mmproj(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    snapshot = tmp_path / "models--repo--a" / "snapshots" / "rev"
    model_file = snapshot / "model-Q4_K_M.gguf"
    selected_mmproj = snapshot / "mmproj-Q4_K.gguf"
    stale_mmproj = snapshot / "old-mmproj-Q8_0.gguf"
    for path in (model_file, selected_mmproj, stale_mmproj):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("data", encoding="utf-8")
    model_blob = tmp_path / "blobs" / "model"
    selected_blob = tmp_path / "blobs" / "selected-mmproj"
    stale_blob = tmp_path / "blobs" / "stale-mmproj"
    for path in (model_blob, selected_blob, stale_blob):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("blob", encoding="utf-8")
    cache_info = FakeCleanupCacheInfo(
        [
            FakeCleanupRepo(
                "model",
                "repo/a",
                [
                    FakeCleanupRevision(
                        "rev-a",
                        [
                            FakeFile("model-Q4_K_M.gguf", model_file, model_blob, 4),
                            FakeFile("mmproj-Q4_K.gguf", selected_mmproj, selected_blob, 4),
                            FakeFile("old-mmproj-Q8_0.gguf", stale_mmproj, stale_blob, 8),
                        ],
                        1.0,
                        snapshot,
                    )
                ],
            )
        ]
    )
    monkeypatch.setattr(cleanup_model_cache, "load_models", lambda: [("repo/a", "Q4_K_M", "g", False)])
    monkeypatch.setattr(cleanup_model_cache, "scan_cache_dir", lambda: cache_info)
    monkeypatch.setattr("sys.argv", ["cleanup_model_cache.py"])

    cleanup_model_cache.main()

    assert model_file.exists()
    assert selected_mmproj.exists()
    assert selected_blob.exists()
    assert not stale_mmproj.exists()
    assert not stale_blob.exists()


def test_main_skips_duplicate_file_paths(
    monkeypatch: MonkeyPatch, tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    stale_file = tmp_path / "models--repo--a" / "snapshots" / "rev" / "model-Q8_0.gguf"
    stale_file.parent.mkdir(parents=True)
    stale_file.write_text("data", encoding="utf-8")
    blob = tmp_path / "blobs" / "stale"
    blob.parent.mkdir()
    blob.write_text("blob", encoding="utf-8")
    cache_info = FakeCleanupCacheInfo(
        [
            FakeCleanupRepo(
                "model",
                "repo/a",
                [
                    FakeCleanupRevision(
                        "rev-a",
                        [
                            FakeFile("model-Q8_0.gguf", stale_file, blob, 8),
                            FakeFile("copy-Q8_0.gguf", stale_file, blob, 8),
                        ],
                        1.0,
                        stale_file.parent,
                    )
                ],
            )
        ]
    )
    monkeypatch.setattr(cleanup_model_cache, "load_models", lambda: [("repo/a", "Q4_K_M", "g", False)])
    monkeypatch.setattr(cleanup_model_cache, "scan_cache_dir", lambda: cache_info)
    monkeypatch.setattr("sys.argv", ["cleanup_model_cache.py"])

    cleanup_model_cache.main()

    captured = capsys.readouterr()
    assert "Extra cached GGUF files to delete: 1" in captured.out
    assert not stale_file.exists()


def test_main_unlinks_dangling_symlink(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    symlink = tmp_path / "models--repo--a" / "snapshots" / "rev" / "model-Q8_0.gguf"
    symlink.parent.mkdir(parents=True)
    symlink.symlink_to(tmp_path / "missing-blob")
    blob = tmp_path / "blobs" / "stale"
    blob.parent.mkdir()
    blob.write_text("blob", encoding="utf-8")
    cache_info = FakeCleanupCacheInfo(
        [
            FakeCleanupRepo(
                "model",
                "repo/a",
                [
                    FakeCleanupRevision(
                        "rev-a",
                        [FakeFile("model-Q8_0.gguf", symlink, blob, 8)],
                        1.0,
                        symlink.parent,
                    )
                ],
            )
        ]
    )
    monkeypatch.setattr(cleanup_model_cache, "load_models", lambda: [("repo/a", "Q4_K_M", "g", False)])
    monkeypatch.setattr(cleanup_model_cache, "scan_cache_dir", lambda: cache_info)
    monkeypatch.setattr("sys.argv", ["cleanup_model_cache.py"])

    cleanup_model_cache.main()

    assert not symlink.exists()
    assert not symlink.is_symlink()


def test_main_actual_deletion_prunes_empty_snapshot_dirs(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    snapshot = tmp_path / "models--repo--a" / "snapshots" / "rev"
    stale_file = snapshot / "nested" / "model-Q8_0.gguf"
    stale_file.parent.mkdir(parents=True)
    stale_file.write_text("data", encoding="utf-8")
    blob = tmp_path / "blobs" / "stale"
    blob.parent.mkdir()
    blob.write_text("blob", encoding="utf-8")
    cache_info = FakeCleanupCacheInfo(
        [
            FakeCleanupRepo(
                "model",
                "repo/a",
                [
                    FakeCleanupRevision(
                        "rev-a",
                        [FakeFile("model-Q8_0.gguf", stale_file, blob, 8)],
                        1.0,
                        snapshot,
                    )
                ],
            )
        ]
    )
    monkeypatch.setattr(cleanup_model_cache, "load_models", lambda: [("repo/a", "Q4_K_M", "g", False)])
    monkeypatch.setattr(cleanup_model_cache, "scan_cache_dir", lambda: cache_info)
    monkeypatch.setattr("sys.argv", ["cleanup_model_cache.py"])

    cleanup_model_cache.main()

    assert not stale_file.exists()
    assert not snapshot.exists()
    assert snapshot.parent.exists()
