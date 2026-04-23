from pathlib import Path

from gguf import GGUFReader
from hf_gguf import find_best_mmproj_file, find_matching_model_files

HF_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"


def _snapshot_relative_path(repo_dir: Path, path: Path) -> str:
    parts = path.relative_to(repo_dir).parts
    return Path(*parts[1:]).as_posix()


def find_local_gguf_path(tag):
    repo, quant = tag.split(":", 1)
    repo_dir = HF_CACHE_DIR / f"models--{repo.replace('/', '--')}" / "snapshots"
    if not repo_dir.exists():
        return None

    candidates = {}
    for path in repo_dir.rglob("*.gguf"):
        rel = _snapshot_relative_path(repo_dir, path)
        candidates.setdefault(rel, path)

    matches = find_matching_model_files(sorted(candidates), quant)
    if not matches:
        return None

    return candidates[matches[0]]


def get_mmproj_size_mib(tag):
    repo = tag.split(":")[0] if ":" in tag else tag
    repo_dir = HF_CACHE_DIR / f"models--{repo.replace('/', '--')}" / "snapshots"
    if not repo_dir.exists():
        return 0

    repo_files = {}
    for path in repo_dir.rglob("*.gguf"):
        if path.suffix != ".gguf":
            continue
        rel = _snapshot_relative_path(repo_dir, path)
        repo_files.setdefault(rel, path)

    quant = tag.split(":", 1)[1] if ":" in tag else ""
    model_files = find_matching_model_files(sorted(repo_files), quant)
    if not model_files:
        return 0

    mmproj = find_best_mmproj_file(sorted(repo_files), model_files[0])
    if mmproj is None:
        return 0

    best = repo_files[mmproj].resolve()
    if not best.exists():
        return 0
    return best.stat().st_size // (1024 * 1024)


def get_max_ctx_from_gguf(tag):
    path = find_local_gguf_path(tag)
    if path is None:
        return None

    reader = GGUFReader(str(path))
    arch_field = reader.get_field("general.architecture")
    if arch_field is not None:
        arch = arch_field.contents()
        ctx_field = reader.get_field(f"{arch}.context_length")
        if ctx_field is not None:
            return int(ctx_field.contents())

    ctx_field = reader.get_field("general.context_length")
    if ctx_field is not None:
        return int(ctx_field.contents())

    for key in reader.fields.keys():
        if key.endswith(".context_length"):
            field = reader.get_field(key)
            if field is not None:
                return int(field.contents())

    return None


def detect_capabilities(tag):
    path = find_local_gguf_path(tag)
    if path is None:
        return {"vision": "?", "reasoning": "?", "switchable": "?", "effort": "?"}

    reader = GGUFReader(str(path))
    vision = False
    reasoning = False
    switchable = False
    effort_levels = []

    ct_field = reader.get_field("tokenizer.chat_template")
    if ct_field:
        try:
            val = ct_field.contents()
            if isinstance(val, str):
                vl = val.lower()
                vision = any(
                    w in vl
                    for w in [
                        "<|image",
                        "image_url",
                        "boi_token",
                        "eoi_token",
                        "<image>",
                        "image_pad",
                        "<|vision",
                    ]
                )
                reasoning = any(
                    w in vl
                    for w in [
                        "enable_thinking",
                        "reasoning_content",
                        "think>",
                        "</think",
                        "<think",
                        "[think]",
                    ]
                )
                switchable = "enable_thinking" in vl
                if "reasoning_effort" in vl:
                    if (
                        "none" in val.lower()
                        and "high" in val.lower()
                        and "medium" not in val.lower()
                    ):
                        effort_levels = ["none", "high"]
                    elif "medium" in val.lower():
                        effort_levels = ["low", "medium", "high"]
                    elif "low" in val.lower():
                        effort_levels = ["low", "high"]
                elif "low_effort" in vl:
                    effort_levels = ["low", "high"]
        except Exception:
            pass

    if not vision:
        tags_field = reader.get_field("general.tags")
        if tags_field:
            try:
                tags_val = tags_field.contents()
                if isinstance(tags_val, (list, tuple)):
                    vision = any(
                        "image" in str(t).lower()
                        or "vision" in str(t).lower()
                        or "any-to-any" in str(t).lower()
                        for t in tags_val
                    )
            except Exception:
                pass

    effort_str = "/".join(effort_levels) if effort_levels else "-"
    return {
        "vision": "yes" if vision else "no",
        "reasoning": "yes" if reasoning else "no",
        "switchable": "yes" if switchable else ("no" if reasoning else "-"),
        "effort": effort_str,
    }
