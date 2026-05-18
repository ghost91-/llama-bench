import re
from pathlib import Path
from typing import cast

from gguf import GGUFReader
from llama_bench.gguf_cache import desired_gguf_files, local_gguf_files
from llama_bench.model_identity import identity_from_tag
from llama_bench.schema_types import Capabilities, ReasoningDetails

VISION_TEMPLATE_TOKENS = (
    "<|image",
    "image_url",
    "boi_token",
    "eoi_token",
    "<image>",
    "image_pad",
    "<|vision",
)
VISION_TAG_TOKENS = ("image", "vision", "any-to-any")


def find_local_gguf_path(tag: str) -> Path | None:
    identity = identity_from_tag(tag)
    candidates = local_gguf_files(identity.repo)
    if candidates is None:
        return None

    matches = desired_gguf_files(sorted(candidates), identity.quant)
    if not matches:
        return None

    return candidates[matches[0]]


def get_mmproj_size_mib(tag: str) -> int:
    identity = identity_from_tag(tag, require_quant=False)
    repo_files = local_gguf_files(identity.repo)
    if repo_files is None:
        return 0

    files = desired_gguf_files(sorted(repo_files), identity.quant)
    if not files:
        return 0

    mmproj = files[-1]
    if "mmproj" not in mmproj.lower():
        return 0

    best = repo_files[mmproj].resolve()
    if not best.exists():
        return 0
    return best.stat().st_size // (1024 * 1024)


def get_max_ctx_from_gguf(tag: str) -> int | None:
    path = find_local_gguf_path(tag)
    if path is None:
        return None

    reader = GGUFReader(str(path))
    arch_field = reader.get_field("general.architecture")
    if arch_field is not None:
        arch = cast(str, arch_field.contents())
        ctx_field = reader.get_field(f"{arch}.context_length")
        if ctx_field is not None:
            return int(cast(int | str, ctx_field.contents()))

    ctx_field = reader.get_field("general.context_length")
    if ctx_field is not None:
        return int(cast(int | str, ctx_field.contents()))

    for key in reader.fields.keys():
        if key.endswith(".context_length"):
            field = reader.get_field(key)
            if field is not None:
                return int(cast(int | str, field.contents()))

    return None


def _has_experts(reader: GGUFReader) -> bool:
    arch_field = reader.get_field("general.architecture")
    if arch_field is not None:
        arch = cast(str, arch_field.contents())
        for key in (f"{arch}.expert_count", f"{arch}.expert_used_count"):
            field = reader.get_field(key)
            if field is not None and int(field.contents()) > 0:
                return True

    for key in reader.fields.keys():
        if key.endswith((".expert_count", ".expert_used_count")):
            field = reader.get_field(key)
            if field is not None and int(field.contents()) > 0:
                return True

    return False


def is_moe_model(tag: str) -> bool:
    path = find_local_gguf_path(tag)
    if path is None:
        return False

    reader = GGUFReader(str(path))
    return _has_experts(reader)


def detect_capabilities(tag: str) -> Capabilities:
    path = find_local_gguf_path(tag)
    if path is None:
        return {"vision": False, "reasoning": False}

    reader = GGUFReader(str(path))
    vision = False
    reasoning = False
    switchable = False
    effort_levels: list[str] = []

    ct_field = reader.get_field("tokenizer.chat_template")
    if ct_field:
        try:
            val = ct_field.contents()
        except (UnicodeDecodeError, IndexError, TypeError, ValueError):
            val = None
        if isinstance(val, str):
            vl = val.lower()
            vision = any(token in vl for token in VISION_TEMPLATE_TOKENS)
            reasoning = bool(
                re.search(
                    r"(?:enable_thinking|reasoning_content|\[think\]|</?think(?=[\s>/]))",
                    vl,
                )
            )
            if "reasoning_effort" in vl or "low_effort" in vl:
                reasoning = True
            switchable = (
                "enable_thinking" in vl or "reasoning_effort" in vl or "low_effort" in vl
            )
            if "reasoning_effort" in vl:
                if "none" in vl and "high" in vl and "medium" not in vl:
                    effort_levels = ["none", "high"]
                elif "medium" in vl:
                    effort_levels = ["low", "medium", "high"]
                elif "low" in vl:
                    effort_levels = ["low", "high"]
            elif "low_effort" in vl:
                effort_levels = ["low", "high"]

    if not vision:
        tags_field = reader.get_field("general.tags")
        if tags_field:
            try:
                tags_val = tags_field.contents()
            except (UnicodeDecodeError, IndexError, TypeError, ValueError):
                tags_val = None
            if isinstance(tags_val, (list, tuple)):
                vision = any(
                    any(token in str(t).lower() for token in VISION_TAG_TOKENS)
                    for t in cast(list[object] | tuple[object, ...], tags_val)
                )

    reasoning_info: ReasoningDetails | None = None
    if reasoning:
        reasoning_info = {
            "switchable": switchable,
            "efforts": "|".join(effort_levels) if effort_levels else None,
        }
    return {
        "vision": vision,
        "reasoning": reasoning_info if reasoning_info is not None else False,
    }
