from collections.abc import Collection
from pathlib import Path
from typing import Literal, Protocol, TypedDict, TypeAlias

ModelRecord: TypeAlias = tuple[str, str, str]
ResultRow: TypeAlias = dict[str, str]


class ReasoningDetails(TypedDict):
    switchable: bool
    efforts: str | None


ReasoningCapability: TypeAlias = ReasoningDetails | Literal[False]


class Capabilities(TypedDict):
    vision: bool
    reasoning: ReasoningCapability


class ScanEntry(TypedDict):
    fit_target: int
    ctx: int
    ngl: int
    offload: int | None
    ot: str | None
    scan_ts: str


UbatchEntries: TypeAlias = dict[str, ScanEntry]


class ModeCacheEntry(TypedDict):
    ubatch_sizes: UbatchEntries


class ModelScanCacheEntry(TypedDict, total=False):
    mmproj: str
    moe: bool
    max_ctx: int
    max_ctx_ts: str
    caps: Capabilities
    text: ModeCacheEntry
    vision: ModeCacheEntry


ScanCache: TypeAlias = dict[str, ModelScanCacheEntry]


class CachedFileInfo(Protocol):
    file_name: str
    file_path: Path
    blob_path: Path
    size_on_disk: int


class CachedRevisionInfo(Protocol):
    commit_hash: str
    files: Collection[CachedFileInfo]
    last_modified: float
    snapshot_path: Path


class CachedRepoInfo(Protocol):
    repo_type: str
    repo_id: str
    revisions: Collection[CachedRevisionInfo]


class DeleteCacheStrategy(Protocol):
    expected_freed_size: int
    repos: Collection[Path]

    def execute(self) -> None: ...


class HFCacheInfo(Protocol):
    repos: Collection[CachedRepoInfo]

    def delete_revisions(self, *revisions: str) -> DeleteCacheStrategy: ...
