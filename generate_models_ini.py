#!/usr/bin/env python3
import argparse
import json
import os
import sys

from gguf_utils import (
    find_local_gguf_path,
)
from sampler_config import FAMILY_DESCRIPTIONS, SAMPLER_CONFIG
from results import (
    MODELS_FILE,
    display_name_from_tag,
    load_models,
)
from scan_cache import load_scan_cache


def parse_scan_cache(cache):
    results = {}
    for tag, entry in cache.items():
        display_name = display_name_from_tag(tag)
        repo = tag.split(":")[0]
        provider = repo.split("/")[0]
        quant = tag.split(":")[1]
        key = (display_name, quant, provider)
        text = entry.get("text", {})
        vision = entry.get("vision", {})
        results[key] = {
            "actual_ctx": text.get("ctx"),
            "fit_target": text.get("fit_target"),
            "ngl": text.get("ngl"),
            "ubatch": text.get("ubatch"),
            "vision": entry.get("has_vision", "no") == "yes",
            "mmproj": entry.get("mmproj", ""),
            "vision_fit_target": vision.get("fit_target"),
            "vision_ctx": vision.get("ctx"),
            "vision_ngl": vision.get("ngl"),
            "vision_ubatch": vision.get("ubatch"),
        }
    return results


def sampler_summary(group, skip_keys=None):
    cfg = SAMPLER_CONFIG.get(group, {})
    parts = []
    for k, v in cfg.items():
        if skip_keys and k in skip_keys:
            continue
        if k == "chat-template-kwargs":
            try:
                ctk = json.loads(v)
                for ck, cv in ctk.items():
                    val = "true" if cv is True else "false" if cv is False else cv
                    parts.append(f"{ck}={val}")
            except (json.JSONDecodeError, TypeError):
                pass
            continue
        nk = k.replace("-", "_")
        parts.append(f"{nk}={v}")
    return ", ".join(parts)


def format_sampler_settings(group, skip_keys=None):
    cfg = SAMPLER_CONFIG.get(group, {})
    props = []
    for k, v in cfg.items():
        if skip_keys and k in skip_keys:
            continue
        props.append((k, v))
    return props

def generate_ini(models, results, output, dry_run):
    sections = []
    current_group = None
    for repo_id, quant_tag, group, pinned in models:
        if not pinned:
            continue
        full_tag = f"{repo_id}:{quant_tag}"
        if find_local_gguf_path(full_tag) is None:
            print(f"WARNING: {full_tag} not found on disk, skipping", file=sys.stderr)
            continue
        display_name = display_name_from_tag(full_tag)
        provider_prefix = repo_id.split("/")[0]
        entry = results.get((display_name, quant_tag, provider_prefix))
        if not entry:
            print(f"WARNING: {full_tag} has no scan results, skipping", file=sys.stderr)
            continue
        is_vision_capable = entry["vision"]
        text_ctx = entry["actual_ctx"]
        text_fit_target = entry["fit_target"]
        text_ngl = entry["ngl"]
        text_ubatch = entry["ubatch"]
        vision_fit_target = entry["vision_fit_target"]
        vision_ctx = entry["vision_ctx"]
        vision_ngl = entry["vision_ngl"]
        vision_ubatch = entry["vision_ubatch"]
        skip_keys = {"ubatch-size"} if text_ubatch is not None or vision_ubatch is not None else None
        if group != current_group:
            current_group = group
            desc = FAMILY_DESCRIPTIONS.get(group, group)
            summary = sampler_summary(group, skip_keys=skip_keys)
            sections.append({"type": "comment", "text": f"; {desc} — {summary}"})
        need_vision_section = is_vision_capable and (
            vision_ctx is not None
            and (
                vision_ctx != text_ctx
                or vision_ngl != text_ngl
                or vision_ubatch != text_ubatch
            )
        )
        text_props = []
        text_props.append(("hf", full_tag))
        if text_ctx is not None:
            text_props.append(("ctx-size", str(text_ctx)))
        if text_fit_target is not None and not (is_vision_capable and not need_vision_section):
            text_props.append(("fit-target", str(text_fit_target)))
        if text_ubatch is not None:
            text_props.append(("ubatch-size", str(text_ubatch)))
        if is_vision_capable and not need_vision_section:
            text_props.append(("mmproj-auto", "on"))
            text_props.append(("mmproj-offload", "on"))
            if vision_fit_target is not None:
                text_props.append(("fit-target", str(vision_fit_target)))
        text_props.extend(format_sampler_settings(group, skip_keys=skip_keys))
        sections.append({"type": "section", "name": full_tag, "props": text_props})
        if need_vision_section:
            vision_props = []
            vision_props.append(("hf", full_tag))
            if vision_ctx is not None:
                vision_props.append(("ctx-size", str(vision_ctx)))
            if vision_fit_target is not None:
                vision_props.append(("fit-target", str(vision_fit_target)))
            if vision_ubatch is not None:
                vision_props.append(("ubatch-size", str(vision_ubatch)))
            vision_props.append(("mmproj-auto", "on"))
            vision_props.append(("mmproj-offload", "on"))
            vision_props.extend(format_sampler_settings(group, skip_keys=skip_keys))
            sections.append(
                {"type": "section", "name": f"{full_tag}:vision", "props": vision_props}
            )
    lines = []
    lines.append("version = 1")
    lines.append("")
    lines.append("[*]")
    lines.append("fit = on")
    lines.append("fit-ctx = 5000")
    lines.append("flash-attn = on")
    lines.append("parallel = 1")
    for sec in sections:
        lines.append("")
        if sec["type"] == "comment":
            lines.append(sec["text"])
        elif sec["type"] == "section":
            lines.append(f"[{sec['name']}]")
            for prop in sec["props"]:
                k, v = prop
                lines.append(f"{k} = {v}")
    lines.append("")
    content = "\n".join(lines)
    if dry_run:
        print(content, end="")
    else:
        output_dir = os.path.dirname(output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output, "w") as f:
            f.write(content)
        print(f"Wrote {output}")


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate models.ini from benchmark results and sampler config"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print to stdout")
    parser.add_argument("--output", default=MODELS_FILE, help="Output file path")
    args = parser.parse_args()

    models = load_models()
    cache = load_scan_cache()
    results = parse_scan_cache(cache)
    generate_ini(models, results, args.output, args.dry_run)


if __name__ == "__main__":
    main()
