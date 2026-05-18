from pathlib import Path
from collections.abc import Sequence

from llama_bench.hf_gguf import find_best_mmproj_file, find_matching_model_files

HF_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"


def repo_snapshot_dir(repo: str, cache_dir: Path | None = None) -> Path:
    root = HF_CACHE_DIR if cache_dir is None else cache_dir
    return root / f"models--{repo.replace('/', '--')}" / "snapshots"


def _snapshot_relative_path(repo_dir: Path, path: Path) -> str:
    parts = path.relative_to(repo_dir).parts
    return Path(*parts[1:]).as_posix()


def local_gguf_files(repo: str, cache_dir: Path | None = None) -> dict[str, Path] | None:
    repo_dir = repo_snapshot_dir(repo, cache_dir)
    if not repo_dir.exists():
        return None

    candidates: dict[str, Path] = {}
    for path in repo_dir.rglob("*.gguf"):
        rel = _snapshot_relative_path(repo_dir, path)
        existing = candidates.get(rel)
        if existing is None or _prefer_cache_path(path, existing):
            candidates[rel] = path
    return candidates


def _prefer_cache_path(path: Path, existing: Path) -> bool:
    path_mtime = path.stat().st_mtime_ns
    existing_mtime = existing.stat().st_mtime_ns
    return path_mtime > existing_mtime or (
        path_mtime == existing_mtime and path.as_posix() < existing.as_posix()
    )


def desired_gguf_files(repo_files: Sequence[str], quant: str) -> list[str]:
    model_files = find_matching_model_files(list(repo_files), quant)
    if not model_files:
        return []

    mmproj = find_best_mmproj_file(list(repo_files), model_files[0])
    return model_files + ([mmproj] if mmproj else [])
