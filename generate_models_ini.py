#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from gguf_utils import (  # noqa: E402
    detect_capabilities,
    find_local_gguf_path,
    get_mmproj_size_mib,
)
from sampler_config import FAMILY_DESCRIPTIONS, SAMPLER_CONFIG  # noqa: E402
from results import (  # noqa: E402
    MODELS_FILE,
    RESULTS_FILE,
    display_name_from_tag,
    load_models,
)


def parse_ctx(val):
    if not val or val == "-":
        return None
    val = val.strip().lower()
    if val.endswith("k"):
        return int(float(val[:-1]) * 1000)
    return int(val)


def parse_ngl(val):
    if not val or val == "-":
        return None
    val = val.strip().lower()
    if val == "all":
        return -1
    return int(val)


def parse_results_table(filepath):
    results = {}
    if not os.path.exists(filepath):
        return results
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            display_name = row["model"]
            quant = row["quant"]
            provider = row["provider"]
            key = (display_name, quant, provider)
            entry = {
                "actual_ctx": parse_ctx(row["ctx"]),
                "ngl": parse_ngl(row["ngl"]),
                "vision": row["vision"].strip().lower() == "yes",
                "mmproj": row["mmproj"],
                "vision_ctx": parse_ctx(row["vctx"]),
                "vision_ngl": parse_ngl(row["vngl"]),
            }
            results[key] = entry
    return results


def sampler_summary(group):
    cfg = SAMPLER_CONFIG.get(group, {})
    parts = []
    for k, v in cfg.items():
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


def format_sampler_settings(group):
    cfg = SAMPLER_CONFIG.get(group, {})
    props = []
    for k, v in cfg.items():
        props.append((k, v))
    return props


def generate_ini(models, results, output, dry_run):
    sections = []
    current_group = None
    for repo_id, quant_tag, group in models:
        full_tag = f"{repo_id}:{quant_tag}"
        if find_local_gguf_path(full_tag) is None:
            print(f"WARNING: {full_tag} not found on disk, skipping", file=sys.stderr)
            continue
        if group != current_group:
            current_group = group
            desc = FAMILY_DESCRIPTIONS.get(group, group)
            summary = sampler_summary(group)
            sections.append({"type": "comment", "text": f"; {desc} — {summary}"})
        display_name = display_name_from_tag(full_tag)
        provider_prefix = repo_id.split("/")[0]
        entry = results.get((display_name, quant_tag, provider_prefix))
        if entry:
            is_vision_capable = entry["vision"]
            text_ctx = entry["actual_ctx"]
            text_ngl = entry["ngl"]
            vision_ctx = entry["vision_ctx"]
            vision_ngl = entry["vision_ngl"]
        else:
            caps = detect_capabilities(full_tag)
            is_vision_capable = caps["vision"] == "yes"
            text_ctx = None
            text_ngl = None
            vision_ctx = None
            vision_ngl = None
        need_vision_section = is_vision_capable and (
            vision_ctx is not None and (vision_ctx != text_ctx or vision_ngl != text_ngl)
        )
        mmproj_mib = get_mmproj_size_mib(full_tag) if is_vision_capable else 0
        text_props = []
        text_props.append(("hf", full_tag))
        if text_ctx is not None:
            text_props.append(("ctx-size", str(text_ctx)))
        if is_vision_capable and not need_vision_section:
            text_props.append(("mmproj-auto", "on"))
        text_props.extend(format_sampler_settings(group))
        sections.append({"type": "section", "name": full_tag, "props": text_props})
        if need_vision_section:
            vision_props = []
            vision_props.append(("hf", full_tag))
            if vision_ctx is not None:
                vision_props.append(("ctx-size", str(vision_ctx)))
            vision_props.append(("mmproj-auto", "on"))
            vision_props.append(("mmproj-offload", "on"))
            if mmproj_mib > 0:
                vision_props.append(("fit-target", str(512 + mmproj_mib)))
            vision_props.extend(format_sampler_settings(group))
            sections.append(
                {"type": "section", "name": f"{full_tag}:vision", "props": vision_props}
            )
    lines = []
    lines.append("version = 1")
    lines.append("")
    lines.append("[*]")
    lines.append("fit = on")
    lines.append("fit-ctx = 5000")
    lines.append("fit-target = 512")
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
        os.makedirs(os.path.dirname(output), exist_ok=True)
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
    results = parse_results_table(RESULTS_FILE)
    generate_ini(models, results, args.output, args.dry_run)


if __name__ == "__main__":
    main()
