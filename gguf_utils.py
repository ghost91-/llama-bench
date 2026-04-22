import re
from pathlib import Path

from gguf import GGUFReader

HF_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"


def find_local_gguf_path(tag):
    repo, quant = tag.split(":", 1)
    repo_dir = HF_CACHE_DIR / f"models--{repo.replace('/', '--')}" / "snapshots"
    if not repo_dir.exists():
        return None

    candidates = []
    for path in repo_dir.rglob("*.gguf"):
        name = path.name
        lower = name.lower()
        if "mmproj" in lower or "imatrix" in lower:
            continue
        candidates.append(path)

    patterns = [
        re.compile(re.escape(quant) + r"[.-]", re.IGNORECASE),
        re.compile(r"UD-" + re.escape(quant) + r"[.-]", re.IGNORECASE),
        re.compile(r"[-.]" + re.escape(quant) + r"[-.]", re.IGNORECASE),
        re.compile(r"[-.]" + re.escape(quant) + r"\.gguf$", re.IGNORECASE),
    ]

    matches = []
    for pattern in patterns:
        matches = [p for p in candidates if pattern.search(p.name)]
        if matches:
            break

    if not matches:
        return None

    matches = sorted(matches, key=lambda p: ("00001-of" not in p.name, str(p)))
    return matches[0]


def get_mmproj_size_mib(tag):
    repo = tag.split(":")[0] if ":" in tag else tag
    repo_dir = HF_CACHE_DIR / f"models--{repo.replace('/', '--')}" / "snapshots"
    if not repo_dir.exists():
        return 0

    best = None
    for path in repo_dir.rglob("mmproj*"):
        if path.is_symlink() or path.suffix != ".gguf":
            real = path.resolve()
            if real.exists() and (best is None or real.stat().st_size > best.stat().st_size):
                best = real
        elif path.is_file() and (best is None or path.stat().st_size > best.stat().st_size):
            best = path

    if best is None:
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
