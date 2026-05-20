#!/usr/bin/env python3
import argparse
import csv
import os
import sys
from collections.abc import Callable, Sequence
from typing import TypeAlias, TypedDict

from llama_bench.consolidation import LabelledConfig
from llama_bench.gguf_utils import find_local_gguf_path
from llama_bench.model_identity import render_model_tag, result_key_from_parts
from llama_bench.results import MODELS_FILE, RESULTS_FILE, load_models, parse_ctx
from llama_bench.sampler_config import SAMPLER_CONFIG
from llama_bench.selection import load_candidates, select_profiles
from select_configs import build_labelled_configs

SERVER_PARALLEL = 1
SERVER_BATCH_FLOOR = 4096


class IniSection(TypedDict):
    name: str
    props: list[tuple[str, str]]
    comment: str


WarningCallback: TypeAlias = Callable[[str], None]

FitLookup: TypeAlias = dict[tuple[str, str, str, str, int, int], int]


def _server_batch_size(ubatch: int) -> int:
    return max(SERVER_BATCH_FLOOR, SERVER_PARALLEL * ubatch)


def _append_fit_props(
    props: list[tuple[str, str]],
    ctx: int | None,
    fit_target: int | None,
    ubatch: int | None,
) -> None:
    if ctx is not None:
        props.append(("ctx-size", str(ctx)))
    if fit_target is not None:
        props.append(("fit-target", str(fit_target)))
    if ubatch is not None:
        props.append(("ubatch-size", str(ubatch)))
        props.append(("batch-size", str(_server_batch_size(ubatch))))


def _parse_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def load_fit_lookup(results_file: str = RESULTS_FILE) -> FitLookup:
    lookup: FitLookup = {}
    if not os.path.exists(results_file):
        return lookup
    with open(results_file, newline="") as f:
        for row in csv.DictReader(f):
            mode = row.get("mode")
            if mode not in ("text", "vision"):
                continue
            ctx = parse_ctx(row.get("ctx"))
            ubatch = _parse_int(row.get("ubatch"))
            fit_target = _parse_int(row.get("fit_target"))
            if ctx is None or ubatch is None or fit_target is None:
                continue
            key = (row.get("model", ""), row.get("quant", ""), row.get("provider", ""), mode, ubatch, ctx)
            lookup[key] = fit_target
    return lookup


def load_repo_lookup() -> dict[tuple[str, str, str], str]:
    repo_by_key: dict[tuple[str, str, str], str] = {}
    for repo, quant, _group in load_models():
        key = result_key_from_parts(repo, quant)
        repo_by_key[key] = repo
    return repo_by_key


def format_sampler_settings(group: str, skip_keys: set[str] | None = None) -> list[tuple[str, str]]:
    cfg = SAMPLER_CONFIG.get(group, {})
    props: list[tuple[str, str]] = []
    for k, v in cfg.items():
        if skip_keys and k in skip_keys:
            continue
        props.append((k, v))
    return props


def _print_warning(message: str) -> None:
    print(message, file=sys.stderr)


def build_ini_sections(
    configs: Sequence[LabelledConfig],
    fit_lookup: FitLookup,
    repo_lookup: dict[tuple[str, str, str], str],
    gguf_exists_fn: Callable[[str], object | None] = find_local_gguf_path,
    warn: WarningCallback = _print_warning,
) -> list[IniSection]:
    raw: list[IniSection] = []
    for config in configs:
        key = config.key
        group, _provider, quant, mode, ubatch, ctx, _pp_tps, _tg_tps = key
        candidate = config.entries[0].scored.candidate
        result_key = (candidate.model, candidate.quant, candidate.provider)
        repo = repo_lookup.get(result_key)
        if repo is None:
            warn(f"WARNING: no repo for {result_key}, skipping")
            continue
        full_tag = render_model_tag(repo, quant)
        if gguf_exists_fn(full_tag) is None:
            warn(f"WARNING: {full_tag} not found on disk, skipping")
            continue
        fit_key = (candidate.model, candidate.quant, candidate.provider, mode, ubatch, ctx)
        fit_target = fit_lookup.get(fit_key)
        props: list[tuple[str, str]] = []
        props.append(("hf", full_tag))
        _append_fit_props(props, ctx, fit_target, ubatch)
        if mode == "vision":
            props.append(("mmproj-auto", "on"))
            props.append(("mmproj-offload", "on"))
        skip_keys = {"ubatch-size"}
        props.extend(format_sampler_settings(group, skip_keys=skip_keys))
        raw.append({"name": config.label, "props": props, "comment": config.description})
    return _merge_vision_sections(raw)


def _merge_vision_sections(sections: list[IniSection]) -> list[IniSection]:
    by_model_key: dict[tuple[str, str, str, str], list[int]] = {}
    for idx, sec in enumerate(sections):
        props = dict(sec["props"])
        hf = props.get("hf", "")
        ctx = props.get("ctx-size", "")
        ubatch = props.get("ubatch-size", "")
        by_model_key.setdefault((hf, ctx, ubatch, "text"), []).append(idx)
        by_model_key.setdefault((hf, ctx, ubatch, "vision"), []).append(idx)

    merged_indices: set[int] = set()
    result: list[IniSection] = []
    for idx, sec in enumerate(sections):
        if idx in merged_indices:
            continue
        props = dict(sec["props"])
        mode = "vision" if "mmproj-auto" in props else "text"
        if mode == "text":
            hf = props.get("hf", "")
            ctx = props.get("ctx-size", "")
            ubatch = props.get("ubatch-size", "")
            vision_indices = by_model_key.get((hf, ctx, ubatch, "vision"), [])
            vision_idx = next((i for i in vision_indices if i > idx and i not in merged_indices), None)
            if vision_idx is not None:
                vision_props = dict(sections[vision_idx]["props"])
                vision_fit_target = vision_props.get("fit-target")
                merged = _apply_vision_merge(sec, vision_fit_target)
                merged_indices.add(vision_idx)
                result.append(merged)
                continue
        result.append(sec)
    return result


def _apply_vision_merge(text_section: IniSection, vision_fit_target: str | None) -> IniSection:
    props: list[tuple[str, str]] = []
    text_props = dict(text_section["props"])
    for k, v in text_section["props"]:
        if k == "fit-target" and vision_fit_target is not None:
            props.append((k, vision_fit_target))
        else:
            props.append((k, v))
    if "mmproj-auto" not in text_props:
        props.append(("mmproj-auto", "on"))
    if "mmproj-offload" not in text_props:
        props.append(("mmproj-offload", "on"))
    if vision_fit_target is not None and "fit-target" not in text_props:
        props.append(("fit-target", vision_fit_target))
    return {"name": text_section["name"], "props": props, "comment": text_section["comment"]}


def render_ini(sections: Sequence[IniSection]) -> str:
    lines: list[str] = []
    lines.append("version = 1")
    lines.append("")
    lines.append("[*]")
    lines.append("fit = on")
    lines.append("fit-ctx = 5000")
    lines.append("flash-attn = on")
    lines.append(f"parallel = {SERVER_PARALLEL}")
    lines.append("no-mmproj = on")
    lines.append("no-mmap = on")
    lines.append(f"batch-size = {_server_batch_size(512)}")
    for sec in sections:
        lines.append("")
        if sec["comment"]:
            lines.append(f"; {sec['comment']}")
        lines.append(f"[{sec['name']}]")
        for k, v in sec["props"]:
            lines.append(f"{k} = {v}")
    lines.append("")
    return "\n".join(lines)


def generate_ini(
    configs: Sequence[LabelledConfig],
    fit_lookup: FitLookup,
    repo_lookup: dict[tuple[str, str, str], str],
    output: str,
    dry_run: bool,
) -> None:
    content = render_ini(
        build_ini_sections(configs, fit_lookup, repo_lookup, gguf_exists_fn=find_local_gguf_path)
    )
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
    parser.add_argument(
        "--max-configs-per-group", type=int, default=5, help="Soft consolidation target per model group"
    )
    args = parser.parse_args()

    candidates = load_candidates()
    selections = select_profiles(candidates)
    configs = build_labelled_configs(
        selections, consolidate=True, max_configs_per_group=args.max_configs_per_group
    )
    fit_lookup = load_fit_lookup()
    repo_lookup = load_repo_lookup()
    generate_ini(configs, fit_lookup, repo_lookup, args.output, args.dry_run)


if __name__ == "__main__":
    main()
