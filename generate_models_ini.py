#!/usr/bin/env python3
import argparse
import csv
import os
import sys
from collections.abc import Callable, Sequence
from typing import Literal, TypeAlias, TypedDict

from llama_bench.gguf_utils import (
    find_local_gguf_path,
)
from llama_bench.model_identity import ResultKey, render_model_tag, result_key_from_parts
from llama_bench.results import (
    MODELS_FILE,
    PP_COL,
    RESULTS_FILE,
    TG_COL,
    load_models,
    parse_ctx,
)
from llama_bench.sampler_config import SAMPLER_CONFIG
from llama_bench.schema_types import ModelRecord

# -b (n_batch) differs from scan/bench (which use max(BENCH_BATCH, ubatch)):
# llama-server batches concurrent requests; each can submit up to -ub tokens per decode call.
# Setting -b = SERVER_PARALLEL * ubatch ensures one llama_decode() call can process all
# parallel slots at full throughput. Compute buffer is sized by -ub, so -b has zero VRAM
# impact — this is purely a scheduling/throughput optimisation.
SERVER_PARALLEL = 4
Mode: TypeAlias = Literal["text", "vision"]


class SelectedResult(TypedDict):
    ctx: int
    fit_target: int
    ngl: int | None
    ubatch: int
    pp4096_tps: float
    tg128_tps: float


class ResultSummary(TypedDict, total=False):
    text: SelectedResult
    vision: SelectedResult


ParsedResults: TypeAlias = dict[ResultKey, ResultSummary]


class IniSection(TypedDict):
    name: str
    props: list[tuple[str, str]]


WarningCallback: TypeAlias = Callable[[str], None]


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
        batch_size = SERVER_PARALLEL * ubatch
        props.append(("batch-size", str(batch_size)))


def _parse_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _parse_ngl(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    if value == "all":
        return -1
    return int(value)


def _result_from_csv_row(row: dict[str, str]) -> SelectedResult | None:
    ctx = parse_ctx(row.get("ctx"))
    fit_target = _parse_int(row.get("fit_target"))
    ngl = _parse_ngl(row.get("ngl"))
    ubatch = _parse_int(row.get("ubatch"))
    pp_tps = _parse_float(row.get(PP_COL))
    tg_tps = _parse_float(row.get(TG_COL))
    if ctx is None or fit_target is None or ubatch is None or pp_tps is None or tg_tps is None:
        return None
    return {
        "ctx": ctx,
        "fit_target": fit_target,
        "ngl": ngl,
        "ubatch": ubatch,
        "pp4096_tps": pp_tps,
        "tg128_tps": tg_tps,
    }


def select_result_row(rows: Sequence[SelectedResult], mode: Mode) -> SelectedResult | None:
    if not rows:
        return None

    max_ctx = max(row["ctx"] for row in rows)
    max_ctx_rows = [row for row in rows if row["ctx"] == max_ctx]
    best_max_ctx_pp = max(row["pp4096_tps"] for row in max_ctx_rows)
    if max_ctx >= 125_000:
        floor = 100_000 if mode == "vision" else 125_000
    elif max_ctx >= 100_000 and best_max_ctx_pp >= 500:
        floor = max_ctx
    elif max_ctx >= 75_000:
        floor = 50_000
    else:
        floor = max_ctx

    remaining = [row for row in rows if row["ctx"] >= floor]
    if not remaining:
        return None
    best_tg = max(row["tg128_tps"] for row in remaining)
    remaining = [row for row in remaining if row["tg128_tps"] >= best_tg * 0.9]
    return max(remaining, key=lambda row: (row["pp4096_tps"], row["ctx"], row["ubatch"]))


def load_result_summary(results_file: str = RESULTS_FILE) -> ParsedResults:
    grouped: dict[tuple[ResultKey, Mode], list[SelectedResult]] = {}
    if not os.path.exists(results_file):
        return {}

    with open(results_file, newline="") as f:
        for row in csv.DictReader(f):
            mode_value = row.get("mode")
            if mode_value not in ("text", "vision"):
                continue
            selected = _result_from_csv_row(row)
            if selected is None:
                continue
            key = (row.get("model", ""), row.get("quant", ""), row.get("provider", ""))
            grouped.setdefault((key, mode_value), []).append(selected)

    results: ParsedResults = {}
    for (key, mode), rows in grouped.items():
        selected = select_result_row(rows, mode)
        if selected is None:
            continue
        results.setdefault(key, {})[mode] = selected
    return results


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
    models: list[ModelRecord],
    results: ParsedResults,
    gguf_exists_fn: Callable[[str], object | None] = find_local_gguf_path,
    warn: WarningCallback = _print_warning,
) -> list[IniSection]:
    sections: list[IniSection] = []
    for repo_id, quant_tag, group, pinned in models:
        if not pinned:
            continue
        full_tag = render_model_tag(repo_id, quant_tag)
        if gguf_exists_fn(full_tag) is None:
            warn(f"WARNING: {full_tag} not found on disk, skipping")
            continue
        entry = results.get(result_key_from_parts(repo_id, quant_tag))
        text = entry.get("text") if entry else None
        if text is None:
            warn(f"WARNING: {full_tag} has no benchmark results, skipping")
            continue
        vision = entry.get("vision") if entry else None
        text_ctx = text["ctx"]
        text_fit_target = text["fit_target"]
        text_ngl = text["ngl"]
        text_ubatch = text["ubatch"]
        vision_fit_target = vision["fit_target"] if vision else None
        vision_ctx = vision["ctx"] if vision else None
        vision_ngl = vision["ngl"] if vision else None
        vision_ubatch = vision["ubatch"] if vision else None
        skip_keys = {"ubatch-size"}
        need_vision_section = vision is not None and (
            vision_ctx is not None
            and (
                vision_ctx != text_ctx
                or vision_ngl != text_ngl
                or vision_ubatch != text_ubatch
            )
        )
        text_props: list[tuple[str, str]] = []
        text_props.append(("hf", full_tag))
        text_section_fit_target = None if vision is not None and not need_vision_section else text_fit_target
        _append_fit_props(text_props, text_ctx, text_section_fit_target, text_ubatch)
        if vision is not None and not need_vision_section:
            text_props.append(("mmproj-auto", "on"))
            text_props.append(("mmproj-offload", "on"))
            if vision_fit_target is not None:
                text_props.append(("fit-target", str(vision_fit_target)))
        text_props.extend(format_sampler_settings(group, skip_keys=skip_keys))
        sections.append({"name": full_tag, "props": text_props})
        if need_vision_section:
            vision_props: list[tuple[str, str]] = []
            vision_props.append(("hf", full_tag))
            _append_fit_props(vision_props, vision_ctx, vision_fit_target, vision_ubatch)
            vision_props.append(("mmproj-auto", "on"))
            vision_props.append(("mmproj-offload", "on"))
            vision_props.extend(format_sampler_settings(group, skip_keys=skip_keys))
            sections.append({"name": f"{full_tag}:vision", "props": vision_props})
    return sections


def render_ini(sections: Sequence[IniSection]) -> str:
    lines: list[str] = []
    lines.append("version = 1")
    lines.append("")
    lines.append("[*]")
    lines.append("fit = on")
    lines.append("fit-ctx = 5000")
    lines.append("flash-attn = on")
    lines.append("parallel = 4")
    lines.append(f"batch-size = {SERVER_PARALLEL * 512}")
    for sec in sections:
        lines.append("")
        lines.append(f"[{sec['name']}]")
        for k, v in sec["props"]:
            lines.append(f"{k} = {v}")
    lines.append("")
    return "\n".join(lines)


def generate_ini(models: list[ModelRecord], results: ParsedResults, output: str, dry_run: bool) -> None:
    content = render_ini(build_ini_sections(models, results, gguf_exists_fn=find_local_gguf_path))
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
    results = load_result_summary()
    generate_ini(models, results, args.output, args.dry_run)


if __name__ == "__main__":
    main()
