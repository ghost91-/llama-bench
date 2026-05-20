from dataclasses import dataclass
from pathlib import Path

import pytest
from pytest import CaptureFixture, MonkeyPatch

import download_models


@dataclass
class FakeRevision:
    commit_hash: str
    last_modified: float


@dataclass
class FakeRepo:
    revisions: list[FakeRevision]


class FakeDeleteStrategy:
    def __init__(self, expected_freed_size: int = 1234) -> None:
        self.expected_freed_size = expected_freed_size
        self.repos: list[Path] = []
        self.executed = False

    def execute(self) -> None:
        self.executed = True


class FakeCacheInfo:
    def __init__(self) -> None:
        self.repos = [
            FakeRepo([FakeRevision("new", 20.0), FakeRevision("old", 10.0)]),
            FakeRepo([FakeRevision("only", 1.0)]),
        ]
        self.deleted: tuple[str, ...] = ()
        self.strategy = FakeDeleteStrategy()

    def delete_revisions(self, *revisions: str) -> FakeDeleteStrategy:
        self.deleted = revisions
        return self.strategy


class FakeSingleRevisionCacheInfo:
    def __init__(self) -> None:
        self.repos = [FakeRepo([FakeRevision("only", 1.0)])]
        self.delete_called = False

    def delete_revisions(self, *revisions: str) -> FakeDeleteStrategy:
        self.delete_called = True
        raise AssertionError(f"delete_revisions called with {revisions!r}")


def fail_evict_old_revisions() -> int:
    raise AssertionError("eviction should not run")


def fail_list_repo_files(_repo_id: str) -> list[str]:
    raise AssertionError("listing should not run")


def fail_snapshot_download(
    _repo_id: str, *, allow_patterns: list[str] | None = None
) -> str:
    raise AssertionError("download should not run")


def q4_repo_files(_repo_id: str) -> list[str]:
    return ["model-Q4_K_M.gguf"]


def no_repo_files(_repo_id: str) -> list[str]:
    return []


def test_get_repo_files_caches_success_and_failure(monkeypatch: MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_list_repo_files(repo_id: str) -> list[str]:
        calls.append(repo_id)
        if repo_id == "bad/repo":
            raise RuntimeError("boom")
        return ["model-Q4_K_M.gguf"]

    monkeypatch.setattr(download_models, "list_repo_files", fake_list_repo_files)
    cache: dict[str, list[str] | None] = {}

    assert download_models.get_repo_files("ok/repo", cache) == ["model-Q4_K_M.gguf"]
    assert download_models.get_repo_files("ok/repo", cache) == ["model-Q4_K_M.gguf"]
    assert download_models.get_repo_files("bad/repo", cache) is None
    assert download_models.get_repo_files("bad/repo", cache) is None
    assert calls == ["ok/repo", "bad/repo"]


def test_get_repo_files_failure_writes_stderr(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    def fake_list_repo_files(_repo_id: str) -> list[str]:
        raise RuntimeError("repo unavailable")

    monkeypatch.setattr(download_models, "list_repo_files", fake_list_repo_files)

    assert download_models.get_repo_files("bad/repo", {}) is None

    captured = capsys.readouterr()
    assert "[ERROR] Could not list files in bad/repo: repo unavailable" in captured.err


def test_evict_old_revisions_deletes_all_but_latest(monkeypatch: MonkeyPatch) -> None:
    cache_info = FakeCacheInfo()
    monkeypatch.setattr(download_models, "scan_cache_dir", lambda: cache_info)

    assert download_models.evict_old_revisions() == 1234
    assert cache_info.deleted == ("old",)
    assert cache_info.strategy.executed is True


def test_evict_old_revisions_returns_zero_on_scan_failure(monkeypatch: MonkeyPatch) -> None:
    def fail_scan() -> FakeCacheInfo:
        raise RuntimeError("cache broken")

    monkeypatch.setattr(download_models, "scan_cache_dir", fail_scan)

    assert download_models.evict_old_revisions() == 0


def test_evict_old_revisions_returns_zero_without_old_revisions(monkeypatch: MonkeyPatch) -> None:
    cache_info = FakeSingleRevisionCacheInfo()
    monkeypatch.setattr(download_models, "scan_cache_dir", lambda: cache_info)

    assert download_models.evict_old_revisions() == 0
    assert cache_info.delete_called is False


def test_main_rejects_parallel_zero(monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]) -> None:
    monkeypatch.setattr("sys.argv", ["download_models.py", "--parallel", "0"])

    with pytest.raises(SystemExit) as exc_info:
        download_models.main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "--parallel must be >= 1" in captured.err


def test_main_lists_groups(monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]) -> None:
    monkeypatch.setattr(
        download_models,
        "load_models",
        lambda: [
            ("repo/a", "Q4_K_M", "qwen"),
            ("repo/b", "Q5_K_M", "qwen"),
            ("repo/c", "Q4_K_M", "gemma"),
        ],
    )
    monkeypatch.setattr(
        download_models,
        "evict_old_revisions",
        fail_evict_old_revisions,
    )
    monkeypatch.setattr(download_models, "list_repo_files", fail_list_repo_files)
    monkeypatch.setattr(download_models, "snapshot_download_fn", fail_snapshot_download)
    monkeypatch.setattr("sys.argv", ["download_models.py", "--list-groups"])

    download_models.main()

    captured = capsys.readouterr()
    assert "gemma (1 variants)" in captured.out
    assert "qwen (2 variants)" in captured.out


def test_main_dry_run_filters_groups_and_counts_missing(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        download_models,
        "load_models",
        lambda: [
            ("repo/a", "Q4_K_M", "qwen3"),
            ("repo/b", "Q5_K_M", "gemma"),
            ("repo/c", "Q8_0", "qwen3"),
        ],
    )
    monkeypatch.setattr(download_models, "evict_old_revisions", lambda: 0)

    def fake_list_repo_files(repo_id: str) -> list[str]:
        return ["model-Q4_K_M.gguf", "mmproj-Q4_K.gguf"] if repo_id == "repo/a" else []

    monkeypatch.setattr(download_models, "list_repo_files", fake_list_repo_files)
    monkeypatch.setattr("sys.argv", ["download_models.py", "--dry-run", "--group", "qwen"])

    download_models.main()

    captured = capsys.readouterr()
    assert "[DRY RUN] Would download/verify 2 file(s)" in captured.out
    assert "[MISSING] No files matching 'Q8_0'" in captured.out
    assert "Downloaded: 1, Missing: 1, Failed: 0, Skipped: 1" in captured.out


def test_main_groups_downloads_by_repo_and_reports_failures(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    downloaded: list[tuple[str, list[str] | None]] = []
    monkeypatch.setattr(
        download_models,
        "load_models",
        lambda: [
            ("repo/a", "Q4_K_M", "qwen"),
            ("repo/a", "Q5_K_M", "qwen"),
            ("repo/b", "Q4_K_M", "qwen"),
        ],
    )
    monkeypatch.setattr(download_models, "evict_old_revisions", lambda: 0)

    def fake_list_repo_files(repo_id: str) -> list[str]:
        if repo_id == "repo/a":
            return ["model-Q4_K_M.gguf", "model-Q5_K_M.gguf", "mmproj-Q4_K.gguf"]
        return ["model-Q4_K_M.gguf"]

    monkeypatch.setattr(download_models, "list_repo_files", fake_list_repo_files)

    def fake_snapshot_download(repo_id: str, *, allow_patterns: list[str] | None = None) -> str:
        downloaded.append((repo_id, allow_patterns))
        if repo_id == "repo/b":
            raise RuntimeError("download failed")
        return "/tmp/snapshot"

    monkeypatch.setattr(download_models, "snapshot_download_fn", fake_snapshot_download)
    monkeypatch.setattr("sys.argv", ["download_models.py", "--parallel", "1"])

    download_models.main()

    captured = capsys.readouterr()
    assert ("repo/a", ["mmproj-Q4_K.gguf", "model-Q4_K_M.gguf", "model-Q5_K_M.gguf"]) in downloaded
    assert ("repo/b", ["model-Q4_K_M.gguf"]) in downloaded
    assert "[OK] repo/a:Q4_K_M" in captured.out
    assert "[OK] repo/a:Q5_K_M" in captured.out
    assert "[FAILED] repo/b:Q4_K_M: download failed" in captured.err
    assert "Downloaded: 2, Missing: 0, Failed: 1, Skipped: 0" in captured.out


def test_main_repo_list_failure_continues(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        download_models,
        "load_models",
        lambda: [
            ("repo/bad", "Q4_K_M", "qwen"),
            ("repo/good", "Q4_K_M", "qwen"),
        ],
    )
    monkeypatch.setattr(download_models, "evict_old_revisions", lambda: 0)

    def fake_list_repo_files(repo_id: str) -> list[str]:
        if repo_id == "repo/bad":
            raise RuntimeError("repo list failed")
        return ["model-Q4_K_M.gguf"]

    monkeypatch.setattr(download_models, "list_repo_files", fake_list_repo_files)
    monkeypatch.setattr("sys.argv", ["download_models.py", "--dry-run"])

    download_models.main()

    captured = capsys.readouterr()
    assert "[ERROR] Could not list files in repo/bad: repo list failed" in captured.err
    assert "[DRY RUN] Would download/verify 1 file(s)" in captured.out
    assert "Downloaded: 1, Missing: 0, Failed: 1, Skipped: 0" in captured.out


def test_main_sorts_and_deduplicates_allow_patterns(monkeypatch: MonkeyPatch) -> None:
    downloaded: list[tuple[str, list[str] | None]] = []
    monkeypatch.setattr(
        download_models,
        "load_models",
        lambda: [
            ("repo/a", "Q4_K_M", "qwen"),
            ("repo/a", "Q4_K_M", "qwen"),
        ],
    )
    monkeypatch.setattr(download_models, "evict_old_revisions", lambda: 0)

    def fake_list_repo_files(_repo_id: str) -> list[str]:
        return [
            "z-model-Q4_K_M.gguf",
            "a-model-Q4_K_M.gguf",
            "a-model-Q4_K_M.gguf",
            "mmproj-Q4_K.gguf",
        ]

    monkeypatch.setattr(download_models, "list_repo_files", fake_list_repo_files)

    def fake_snapshot_download(repo_id: str, *, allow_patterns: list[str] | None = None) -> str:
        downloaded.append((repo_id, allow_patterns))
        return "/tmp/snapshot"

    monkeypatch.setattr(download_models, "snapshot_download_fn", fake_snapshot_download)
    monkeypatch.setattr("sys.argv", ["download_models.py", "--parallel", "1"])

    download_models.main()

    assert downloaded == [
        ("repo/a", ["a-model-Q4_K_M.gguf", "mmproj-Q4_K.gguf", "z-model-Q4_K_M.gguf"])
    ]


def test_main_accepts_multiple_group_prefixes(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        download_models,
        "load_models",
        lambda: [
            ("repo/a", "Q4_K_M", "qwen3.5"),
            ("repo/b", "Q4_K_M", "gemma-4"),
            ("repo/c", "Q4_K_M", "qwen2"),
        ],
    )
    monkeypatch.setattr(download_models, "evict_old_revisions", lambda: 0)
    monkeypatch.setattr(download_models, "list_repo_files", q4_repo_files)
    monkeypatch.setattr(
        "sys.argv", ["download_models.py", "--dry-run", "--group", "qwen3", "--group", "gemma"]
    )

    download_models.main()

    captured = capsys.readouterr()
    assert "[1/3] repo/a:Q4_K_M" in captured.out
    assert "[2/3] repo/b:Q4_K_M" in captured.out
    assert "repo/c:Q4_K_M" not in captured.out
    assert "Downloaded: 2, Missing: 0, Failed: 0, Skipped: 1" in captured.out


def test_main_no_repo_tasks_prints_final_counts(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        download_models,
        "load_models",
        lambda: [("repo/a", "Q4_K_M", "qwen")],
    )
    monkeypatch.setattr(download_models, "evict_old_revisions", lambda: 0)
    monkeypatch.setattr(download_models, "list_repo_files", no_repo_files)
    monkeypatch.setattr(download_models, "snapshot_download_fn", fail_snapshot_download)
    monkeypatch.setattr("sys.argv", ["download_models.py"])

    download_models.main()

    captured = capsys.readouterr()
    assert "[MISSING] No files matching 'Q4_K_M'" in captured.out
    assert "Downloaded: 0, Missing: 1, Failed: 0, Skipped: 0" in captured.out


def test_main_evicts_before_and_after_download(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    evictions = [0, 2 * 1024**3]
    monkeypatch.setattr(
        download_models,
        "load_models",
        lambda: [("repo/a", "Q4_K_M", "qwen")],
    )
    monkeypatch.setattr(download_models, "evict_old_revisions", lambda: evictions.pop(0))
    monkeypatch.setattr(download_models, "list_repo_files", q4_repo_files)

    def fake_snapshot_download(
        _repo_id: str, *, allow_patterns: list[str] | None = None
    ) -> str:
        return "/tmp/snapshot"

    monkeypatch.setattr(download_models, "snapshot_download_fn", fake_snapshot_download)
    monkeypatch.setattr("sys.argv", ["download_models.py", "--parallel", "1"])

    download_models.main()

    captured = capsys.readouterr()
    assert evictions == []
    assert "Evicted old cache revisions, freed 2.0 GiB" in captured.out
