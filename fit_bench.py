#!/usr/bin/env python3
"""Pick and benchmark the best context size using llama-fit-params + llama-bench.

Selection rules:
1. Dense models: choose the highest context that still fits fully in VRAM (`-ngl all`).
2. MoE models: choose the highest context that still keeps the maximum achievable `ngl`.

Candidate contexts use a coarse ladder: 5k, 10k, 20k, 30k, 40k, 50k, 75k, 100k,
125k, 150k, 175k, 200k, then 50k steps upward. After the coarse scan, a refinement
pass fills 25k gaps near the boundary where ngl drops below the best.

Note: for MoE models, llama-fit-params uses a greedy overflow strategy per partial
layer that can produce non-monotonic ngl (ngl can go back up at higher context).
The scan must therefore run to completion to find the true max ngl.

Usage:
    python fit-bench.py unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_M
    python fit-bench.py unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_M --list
"""

import argparse
import csv
import fcntl
import io
import os
import re
import subprocess
import sys
import time

from gguf_utils import detect_capabilities, get_max_ctx_from_gguf, get_mmproj_size_mib
from results import (
    BENCH_PP,
    BENCH_TG,
    PP_COL,
    PP_STDDEV_COL,
    RESULTS_FILE,
    TG_COL,
    TG_STDDEV_COL,
    VPP_COL,
    VPP_STDDEV_COL,
    VTG_COL,
    VTG_STDDEV_COL,
    append_result_row,
    display_name_from_tag,
    format_ctx,
    format_mmproj,
    format_ngl,
    format_params,
    load_models,
    parse_ctx,
    sort_results_file,
)

FIT_TARGET = 128
FLASH_ATTN = 1
REPS = 20
BENCH_BATCH = 2048
FIT_UBATCH_DENSE = 512
FIT_UBATCH_MOE = 1024
FIT_PARAMS_TIMEOUT = 600
BENCH_TIMEOUT = 900

_log_file = None
_run_lock = None


def log(message=""):
    line = f"[{time.strftime('%H:%M:%S')}] {message}" if message else ""
    print(line, flush=True)
    if _log_file and line:
        with open(_log_file, "a") as f:
            f.write(line + "\n")


def set_log_file(path):
    global _log_file
    _log_file = path
    if path:
        open(path, "w").close()


def acquire_run_lock():
    global _run_lock
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if runtime_dir:
        lock_dir = os.path.join(runtime_dir, "llama-bench")
    else:
        lock_dir = os.path.join("/tmp", f"llama-bench-{os.getuid()}")
    os.makedirs(lock_dir, exist_ok=True)
    lock_path = os.path.join(lock_dir, "fit_bench.lock")
    lock_file = open(lock_path, "w")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.close()
        return False
    lock_file.write(f"{os.getpid()}\n")
    lock_file.flush()
    _run_lock = lock_file
    return True


def get_fit_params(tag, target_ctx, fit_target=None, ubatch=FIT_UBATCH_DENSE):
    if fit_target is None:
        fit_target = FIT_TARGET
    try:
        r = subprocess.run(
            [
                "llama-fit-params",
                "-hf",
                tag,
                "-c",
                str(target_ctx),
                "--fit-target",
                str(fit_target),
                "-ub",
                str(ubatch),
                "-fa",
                str(FLASH_ATTN),
            ],
            capture_output=True,
            text=True,
            timeout=FIT_PARAMS_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        log(f"fit-params timed out for ctx={format_ctx(target_ctx)}")
        return None
    except Exception as e:
        log(f"fit-params failed for ctx={format_ctx(target_ctx)}: {e}")
        return None
    for line in r.stdout.strip().splitlines():
        line = line.strip()
        if line.startswith("-c "):
            return line
    return None


def get_max_ctx(tag, fit_target=FIT_TARGET, prio=None):
    start = time.monotonic()
    log(f"resolving max context for {tag}")
    try:
        ctx = get_max_ctx_from_gguf(tag)
        if ctx is not None:
            log(
                f"max context resolved via GGUF metadata in {time.monotonic() - start:.1f}s: {format_ctx(ctx)}"
            )
            return ctx
    except Exception as e:
        log(f"GGUF metadata lookup failed: {e}")

    try:
        r = subprocess.run(
            [
                "llama-fit-params",
                "-hf",
                tag,
                "-c",
                "1",
                "--fit-target",
                str(fit_target),
                "-fa",
                str(FLASH_ATTN),
                "-lv",
                "4",
            ],
            capture_output=True,
            text=True,
            timeout=FIT_PARAMS_TIMEOUT,
        )
        m = re.search(r"n_ctx_train\s*=\s*(\d+)", r.stderr + r.stdout)
        if m:
            log(
                f"max context resolved via llama-fit-params in {time.monotonic() - start:.1f}s: {format_ctx(int(m.group(1)))}"
            )
            return int(m.group(1))
    except subprocess.TimeoutExpired:
        log("llama-fit-params max-context probe timed out")

    try:
        r2 = subprocess.run(
            [
                "llama-bench",
                "-hf",
                tag,
                "-fa",
                "1",
                "-pg",
                "512,128",
                "-o",
                "csv",
                "-r",
                "1",
                "-v",
            ] + (["--prio", str(prio)] if prio is not None else []),
            capture_output=True,
            text=True,
            timeout=BENCH_TIMEOUT,
        )
        m2 = re.search(r"n_ctx_train\s*=\s*(\d+)", r2.stderr + r2.stdout)
        if m2:
            log(
                f"max context resolved via llama-bench in {time.monotonic() - start:.1f}s: {format_ctx(int(m2.group(1)))}"
            )
            return int(m2.group(1))
    except subprocess.TimeoutExpired:
        log("llama-bench max-context probe timed out")

    log(f"failed to resolve max context after {time.monotonic() - start:.1f}s")
    return None


def build_ctx_list(max_ctx):
    if not max_ctx:
        return [
            5000,
            10000,
            20000,
            30000,
            40000,
            50000,
            75000,
            100000,
            125000,
            150000,
            175000,
            200000,
        ]

    steps = [
        ctx
        for ctx in [
            5000,
            10000,
            20000,
            30000,
            40000,
            50000,
            75000,
            100000,
            125000,
            150000,
            175000,
            200000,
        ]
        if ctx <= max_ctx
    ]
    ctx = 250000
    while ctx <= max_ctx:
        steps.append(ctx)
        ctx += 50000

    if max_ctx not in steps:
        steps.append(max_ctx)

    return steps


def fit_target_mib(mmproj_mib=0):
    return FIT_TARGET + mmproj_mib


def parse_int_field(value):
    if not value or value in ("-", "?"):
        return None
    return int(value)


def parse_ngl_field(value):
    if not value or value in ("-", "?"):
        return None
    value = value.strip().lower()
    if value == "all":
        return -1
    return int(value)


def load_existing_fit_choice(tag, fit_target, vision_mode):
    if not os.path.exists(RESULTS_FILE):
        return None

    display_name = display_name_from_tag(tag)
    repo = tag.split(":")[0] if ":" in tag else tag
    provider = repo.split("/")[0] if "/" in repo else repo
    quant = tag.split(":")[1] if ":" in tag else ""

    fit_target_col = "vfit_target" if vision_mode else "fit_target"
    ctx_col = "vctx" if vision_mode else "ctx"
    ngl_col = "vngl" if vision_mode else "ngl"
    ubatch_col = "vubatch" if vision_mode else "ubatch"
    moe_col = "vmoe_cpu" if vision_mode else "moe_cpu"
    moe_raw_col = "vmoe_cpu_raw" if vision_mode else "moe_cpu_raw"

    with open(RESULTS_FILE, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (
                row.get("model") != display_name
                or row.get("quant") != quant
                or row.get("provider") != provider
            ):
                continue

            stored_fit_target = parse_int_field(row.get(fit_target_col, ""))
            if stored_fit_target != fit_target:
                return None

            ctx = parse_ctx(row.get(ctx_col, ""))
            ngl = parse_ngl_field(row.get(ngl_col, ""))
            ubatch = parse_int_field(row.get(ubatch_col, ""))
            if ctx is None or ngl is None or ubatch is None:
                return None

            ot_raw = row.get(moe_raw_col, "").strip() or None
            ot = row.get(moe_col, "").strip() or ot_label(ot_raw)
            if vision_mode and row.get("model_type") == "MoE" and ot_raw is None:
                return None
            if ot and ot != "no" and ot_raw is None:
                return None

            return {
                "target_ctx": ctx,
                "ctx": ctx,
                "ngl": ngl,
                "ubatch": ubatch,
                "ot": ot or "no",
                "ot_raw": ot_raw,
            }

    return None


def parse_fit_params(params_str):
    parts = params_str.split()
    c = ngl = ot = None
    i = 0
    while i < len(parts):
        if parts[i] == "-c" and i + 1 < len(parts):
            c = int(parts[i + 1])
            i += 2
        elif parts[i] == "-ngl" and i + 1 < len(parts):
            ngl = int(parts[i + 1])
            i += 2
        elif parts[i] == "-ot":
            i += 1
            ot_parts = []
            while i < len(parts) and not parts[i].startswith("-"):
                ot_parts.append(parts[i])
                i += 1
            ot = " ".join(ot_parts).strip('"')
        else:
            i += 1
    return c, ngl, ot


def result_matches_target(result, target_ngl):
    return result["ctx"] is not None and result["ngl"] == target_ngl


def build_refinement_ctx_list(results, chosen, max_ctx, target_ngl):
    if not chosen or not max_ctx or chosen["ctx"] is None or chosen["ctx"] >= max_ctx:
        return []

    valid = sorted(
        [r for r in results if r["ctx"] is not None and r["ngl"] is not None],
        key=lambda r: r["ctx"],
    )

    upper = None
    for result in valid:
        if result["ctx"] <= chosen["ctx"]:
            continue
        if not result_matches_target(result, target_ngl):
            upper = result["ctx"]
            break

    if upper is None:
        upper = max_ctx

    if upper <= chosen["ctx"] + 25000:
        return []

    targets = []
    ctx = chosen["ctx"] + 25000
    while ctx < upper:
        targets.append(ctx)
        ctx += 25000

    if upper not in targets and upper != chosen["ctx"]:
        targets.append(upper)

    scanned = {r["target_ctx"] for r in results}
    return [ctx for ctx in targets if ctx not in scanned]


def merge_scan_results(results, extra_results):
    merged = {r["target_ctx"]: r for r in results}
    for result in extra_results:
        merged[result["target_ctx"]] = result
    return [merged[key] for key in sorted(merged)]


def ot_to_bench_arg(ot):
    if not ot:
        return []
    return ["-ot", ot.replace(",", ";")]


def ot_label(ot):
    if not ot:
        return "no"
    n = ot.count(",") + 1
    return f"yes({n})"


def parse_bench_row(output, target_prompt, target_gen, target_depth):
    reader = csv.DictReader(io.StringIO(output))
    size_bytes = ""
    n_params = ""
    model_name = ""
    for row in reader:
        if not size_bytes:
            size_bytes = row.get("model_size", "").strip('"')
        if not model_name:
            model_name = row.get("model_type", "").strip('"')
        if not n_params:
            n_params = row.get("model_n_params", "").strip('"')
        if (
            row.get("n_prompt", "").strip('"') == str(target_prompt)
            and row.get("n_gen", "").strip('"') == str(target_gen)
            and row.get("n_depth", "").strip('"') == str(target_depth)
        ):
            return {
                "model_name": model_name,
                "size_bytes": size_bytes,
                "n_params": n_params,
                "speed": row.get("avg_ts", "").strip('"'),
                "stddev": row.get("stddev_ts", "").strip('"'),
            }
    return None


def build_fit_result(target_ctx, ubatch, params_str):
    if not params_str:
        return {
            "target_ctx": target_ctx,
            "ctx": None,
            "ngl": None,
            "ubatch": ubatch,
            "ot_raw": None,
            "ot": None,
        }

    ctx, ngl, ot = parse_fit_params(params_str)
    return {
        "target_ctx": target_ctx,
        "ctx": ctx,
        "ngl": ngl,
        "ubatch": ubatch,
        "ot_raw": ot,
        "ot": ot_label(ot),
    }


def scan_fit_configs(
    tag,
    ctx_targets,
    fit_target=None,
    ubatch=FIT_UBATCH_DENSE,
    stop_on_match=None,
    existing_results=None,
):
    results = list(existing_results or [])
    existing_by_ctx = {r["target_ctx"]: r for r in results}
    total = len(ctx_targets)
    for i, target_ctx in enumerate(ctx_targets, start=1):
        if target_ctx in existing_by_ctx:
            log(
                f"fit scan {i}/{total}: target ctx={format_ctx(target_ctx)} ub={ubatch} (reusing probe)"
            )
            if stop_on_match and stop_on_match(existing_by_ctx[target_ctx]):
                break
            continue
        log(f"fit scan {i}/{total}: target ctx={format_ctx(target_ctx)} ub={ubatch}")
        params_str = get_fit_params(tag, target_ctx, fit_target=fit_target, ubatch=ubatch)
        result = build_fit_result(target_ctx, ubatch, params_str)
        if not params_str:
            log("fit-params failed")
            results.append(result)
            continue

        log(
            f"fit result: ctx={format_ctx(result['ctx'])} ngl={format_ngl(result['ngl'])} moe={result['ot']}"
        )
        results.append(result)
        if stop_on_match and stop_on_match(result):
            break
    return results


def ngl_rank(ngl):
    if ngl is None:
        return -1
    if ngl == -1:
        return 10**9
    return ngl


def prefer_result(left, right):
    if left is None:
        return right
    if right is None:
        return left
    left_key = (ngl_rank(left["ngl"]), left.get("ubatch", 0), left["ctx"] or 0)
    right_key = (ngl_rank(right["ngl"]), right.get("ubatch", 0), right["ctx"] or 0)
    return left if left_key >= right_key else right


def probe_fit_config(tag, target_ctx, fit_target, ubatch):
    params_str = get_fit_params(tag, target_ctx, fit_target=fit_target, ubatch=ubatch)
    if not params_str:
        log(f"reference probe failed: ctx={format_ctx(target_ctx)} ub={ubatch}")
        return None

    result = build_fit_result(target_ctx, ubatch, params_str)
    log(
        f"reference probe: ctx={format_ctx(target_ctx)} ub={ubatch} ngl={format_ngl(result['ngl'])} moe={result['ot']}"
    )
    return result


def try_reuse_existing_fit_choice(tag, fit_target, max_ctx, vision_mode, forced_ubatch=None):
    existing = load_existing_fit_choice(tag, fit_target, vision_mode)
    if existing is None:
        return None

    if max_ctx is not None and existing["ctx"] > max_ctx:
        log(
            f"stored fit choice exceeds current max ctx: ctx={format_ctx(existing['ctx'])} max={format_ctx(max_ctx)}"
        )
        return None

    if forced_ubatch is not None and forced_ubatch != existing["ubatch"]:
        log(
            f"stored fit choice uses ub={existing['ubatch']} but --ubatch={forced_ubatch}; falling back to scan"
        )
        return None

    mode_label = "vision" if vision_mode else "text"
    log(
        "reusing stored fit choice from results: "
        f"mode={mode_label} ctx={format_ctx(existing['ctx'])} ngl={format_ngl(existing['ngl'])} ub={existing['ubatch']}"
    )

    reason = f"Reused stored {mode_label} fit choice from {os.path.basename(RESULTS_FILE)}"
    return [existing], existing, reason, bool(existing["ot_raw"])


def result_below_target(result, target_ngl):
    return result["ctx"] is not None and result["ngl"] is not None and result["ngl"] < target_ngl


def choose_target_result(results, target_ngl, descending):
    ordered = sorted(results, key=lambda r: r["target_ctx"], reverse=descending)
    if descending:
        return next((r for r in ordered if result_matches_target(r, target_ngl)), None)

    chosen = None
    for result in ordered:
        if result_matches_target(result, target_ngl):
            chosen = result
        elif result_below_target(result, target_ngl):
            break
    return chosen


def run_target_scan(
    tag,
    ctx_targets,
    fit_target,
    max_ctx,
    ubatch,
    target_ngl,
    descending,
    probe_result=None,
):
    stop_on_match = (
        (lambda result: result_matches_target(result, target_ngl))
        if descending
        else (lambda result: result_below_target(result, target_ngl))
    )
    results = scan_fit_configs(
        tag,
        sorted(ctx_targets, reverse=descending),
        fit_target=fit_target,
        ubatch=ubatch,
        stop_on_match=stop_on_match,
        existing_results=[probe_result] if probe_result is not None else None,
    )
    chosen = choose_target_result(results, target_ngl, descending)

    refinement_targets = build_refinement_ctx_list(results, chosen, max_ctx, target_ngl)
    if refinement_targets:
        log()
        log(f"Refinement contexts: {', '.join(format_ctx(c) for c in refinement_targets)}")
        extra_results = scan_fit_configs(
            tag,
            sorted(refinement_targets, reverse=descending),
            fit_target=fit_target,
            ubatch=ubatch,
            stop_on_match=stop_on_match,
        )
        results = merge_scan_results(results, extra_results)
        chosen = choose_target_result(results, target_ngl, descending)
    return results, chosen


def reason_for_target_scan(chosen, target_ngl, is_moe):
    if chosen and result_matches_target(chosen, target_ngl):
        if is_moe:
            return f"MoE: highest context that keeps max ngl ({format_ngl(target_ngl)})"
        if target_ngl == -1:
            return "Dense: highest context that still fits fully in VRAM"
        return (
            "Dense fallback: 5k probe does not fit fully in VRAM, "
            f"using highest context at max ngl ({format_ngl(target_ngl)})"
        )
    return "No matching fit-params result"


def fallback_scan_strategy(tag, ctx_targets, fit_target, ubatch):
    results = scan_fit_configs(
        tag, sorted(ctx_targets, reverse=True), fit_target=fit_target, ubatch=ubatch
    )
    chosen, reason, is_moe = select_best_result(results)
    return results, chosen, reason, is_moe


def finalize_target_scan(results, chosen, target_ngl, is_moe):
    if chosen is None:
        chosen, reason, is_moe = select_best_result(results)
        return results, chosen, reason, is_moe
    return results, chosen, reason_for_target_scan(chosen, target_ngl, is_moe), is_moe


def resolve_probe_strategy(probe_dense, probe_moe):
    probes = [p for p in (probe_dense, probe_moe) if p is not None and p["ngl"] is not None]
    if not probes:
        return None

    is_moe = any(p["ot_raw"] for p in probes)
    if is_moe:
        target_ngl = max(p["ngl"] for p in probes)
    else:
        target_ngl = -1 if any(p["ngl"] == -1 for p in probes) else max(p["ngl"] for p in probes)

    selected_probe = next(
        (p for p in (probe_moe, probe_dense) if p is not None and p["ngl"] == target_ngl),
        prefer_result(probe_moe, probe_dense),
    )
    return selected_probe, target_ngl, is_moe


def choose_scan_strategy(tag, ctx_targets, fit_target, max_ctx, forced_ubatch=None):
    probe_ctx = ctx_targets[0]
    if forced_ubatch is not None:
        probe = probe_fit_config(tag, probe_ctx, fit_target, forced_ubatch)
        if probe is None or probe["ngl"] is None:
            return fallback_scan_strategy(tag, ctx_targets, fit_target, forced_ubatch)

        is_moe = bool(probe["ot_raw"])
        target_ngl = probe["ngl"]
        results, chosen = run_target_scan(
            tag,
            ctx_targets,
            fit_target,
            max_ctx,
            forced_ubatch,
            target_ngl,
            descending=is_moe or target_ngl == -1,
            probe_result=probe,
        )
        return finalize_target_scan(results, chosen, target_ngl, is_moe)

    probe_dense = probe_fit_config(tag, probe_ctx, fit_target, FIT_UBATCH_DENSE)
    probe_moe = probe_fit_config(tag, probe_ctx, fit_target, FIT_UBATCH_MOE)
    strategy = resolve_probe_strategy(probe_dense, probe_moe)

    if strategy is None:
        log("reference probes failed; falling back to descending scan with ub=512")
        return fallback_scan_strategy(tag, ctx_targets, fit_target, FIT_UBATCH_DENSE)

    selected_probe, target_ngl, is_moe = strategy
    selected_ubatch = selected_probe["ubatch"]
    model_type = "MoE" if is_moe else "Dense"
    log(
        f"{model_type} model: target ngl={format_ngl(target_ngl)} -> using ub={selected_ubatch}"
    )

    results, chosen = run_target_scan(
        tag,
        ctx_targets,
        fit_target,
        max_ctx,
        selected_ubatch,
        target_ngl,
        descending=is_moe or target_ngl == -1,
        probe_result=selected_probe,
    )
    return finalize_target_scan(results, chosen, target_ngl, is_moe)


def select_best_result(results):
    valid = [r for r in results if r["ctx"] is not None and r["ngl"] is not None]
    if not valid:
        return None, "No valid fit-params results", False

    is_moe = any(r["ot_raw"] for r in valid)

    if is_moe:
        max_ngl = max(r["ngl"] for r in valid)
        matches = [r for r in valid if r["ngl"] == max_ngl]
        chosen = max(matches, key=lambda r: r["ctx"])
        return chosen, f"MoE: highest context that keeps max ngl ({format_ngl(max_ngl)})", True

    full_vram = [r for r in valid if r["ngl"] == -1]
    if full_vram:
        chosen = max(full_vram, key=lambda r: r["ctx"])
        return chosen, "Dense: highest context that still fits fully in VRAM", False

    max_ngl = max(r["ngl"] for r in valid)
    matches = [r for r in valid if r["ngl"] == max_ngl]
    chosen = max(matches, key=lambda r: r["ctx"])
    return (
        chosen,
        f"Dense fallback: no full-VRAM fit found, using highest context at max ngl ({format_ngl(max_ngl)})",
        False,
    )


def run_bench(tag, ngl, ot, ubatch, reps=REPS, prio=None):
    ngl_arg = 99 if ngl == -1 else ngl
    log(
        f"starting llama-bench for {tag} with ngl={format_ngl(ngl)} ub={ubatch} reps={reps}"
    )
    base_cmd = [
        "llama-bench",
        "-hf",
        tag,
        "-fa",
        str(FLASH_ATTN),
        "-ngl",
        str(ngl_arg),
        "-b",
        str(BENCH_BATCH),
        "-ub",
        str(ubatch),
        "-o",
        "csv",
        "-r",
        str(reps),
    ]
    if prio is not None:
        base_cmd += ["--prio", str(prio)]
    if ot:
        base_cmd += ot_to_bench_arg(ot)

    cmd = base_cmd + ["-p", str(BENCH_PP), "-n", str(BENCH_TG), "-d", "0"]

    run = subprocess.run(cmd, capture_output=True, text=True, timeout=BENCH_TIMEOUT)
    if run.returncode != 0 and not run.stdout.strip():
        return None

    pp_result = parse_bench_row(run.stdout, BENCH_PP, 0, 0)
    tg_result = parse_bench_row(run.stdout, 0, BENCH_TG, 0)
    if pp_result is None or tg_result is None:
        return None

    return {
        "model_name": pp_result["model_name"] or tg_result["model_name"],
        "size_bytes": pp_result["size_bytes"] or tg_result["size_bytes"],
        "n_params": pp_result["n_params"] or tg_result["n_params"],
        "pp_speed": pp_result["speed"],
        "pp_stddev": pp_result["stddev"],
        "tg_speed": tg_result["speed"],
        "tg_stddev": tg_result["stddev"],
    }


def print_markdown_table(headers, rows):
    widths = [len(header) for header in headers]
    normalized_rows = []
    for row in rows:
        normalized = ["" if value is None else str(value) for value in row]
        normalized_rows.append(normalized)
        for i, value in enumerate(normalized):
            widths[i] = max(widths[i], len(value))

    def format_row(row):
        cells = [f"{value:<{widths[i]}}" for i, value in enumerate(row)]
        return f"| {' | '.join(cells)} |"

    separator = "|-" + "-|-".join("-" * width for width in widths) + "-|"
    print(format_row(headers), flush=True)
    print(separator, flush=True)
    for row in normalized_rows:
        print(format_row(row), flush=True)


def print_scan_table(scan_results, chosen):
    print("Fit scan:", flush=True)
    rows = []
    for r in scan_results:
        selected = "yes" if chosen and r["target_ctx"] == chosen["target_ctx"] else ""
        rows.append(
            [
                format_ctx(r["target_ctx"]),
                format_ctx(r["ctx"]),
                format_ngl(r["ngl"]),
                r.get("ubatch", "?"),
                r["ot"] if r["ot"] else "?",
                selected,
            ]
        )
    print_markdown_table(["Target Ctx", "Actual Ctx", "ngl", "ub", "MoE CPU", "Selected"], rows)


def print_summary(display_name, quant, provider, size_gib, chosen, is_moe, bench_result):
    print(flush=True)
    print("=" * 90, flush=True)
    print(f"Results for {display_name} ({quant}, {provider}, {size_gib} GiB)", flush=True)
    print("RTX 4070 Laptop (8GB VRAM), 64GB RAM, -fa on", flush=True)
    print(flush=True)
    headers = [
        "Type",
        "Target Ctx",
        "Actual Ctx",
        "ngl",
        "ub",
        "MoE CPU",
        f"pp{BENCH_PP} (t/s)",
        f"tg{BENCH_TG} (t/s)",
    ]
    rows = []
    if chosen:
        pp_f = (
            f"{float(bench_result['pp_speed']):.1f}"
            if bench_result and bench_result.get("pp_speed")
            else ""
        )
        tg_f = (
            f"{float(bench_result['tg_speed']):.1f}"
            if bench_result and bench_result.get("tg_speed")
            else ""
        )
        model_kind = "MoE" if is_moe else "Dense"
        rows.append(
            [
                model_kind,
                format_ctx(chosen["target_ctx"]),
                format_ctx(chosen["ctx"]),
                format_ngl(chosen["ngl"]),
                chosen.get("ubatch", "?"),
                chosen["ot"],
                pp_f,
                tg_f,
            ]
        )
    print_markdown_table(headers, rows)


def write_result_row(tag, chosen, is_moe, bench_result, caps, vision_mode, reps, fit_target):
    display_name = display_name_from_tag(tag)
    repo = tag.split(":")[0] if ":" in tag else tag
    provider = repo.split("/")[0] if "/" in repo else repo
    quant = tag.split(":")[1] if ":" in tag else ""
    model_kind = "MoE" if is_moe else "Dense"
    mmproj_mib = get_mmproj_size_mib(tag)
    mmproj_col = format_mmproj(mmproj_mib)
    size_gib = ""
    n_params = ""
    if bench_result:
        if bench_result["size_bytes"]:
            size_gib = f"{int(bench_result['size_bytes']) / 1024**3:.2f}"
        if bench_result["n_params"]:
            n_params = format_params(bench_result["n_params"])
    pp_f = (
        f"{float(bench_result['pp_speed']):.1f}"
        if bench_result and bench_result.get("pp_speed")
        else ""
    )
    pp_stddev_f = (
        f"{float(bench_result['pp_stddev']):.1f}"
        if bench_result and bench_result.get("pp_stddev")
        else ""
    )
    tg_f = (
        f"{float(bench_result['tg_speed']):.1f}"
        if bench_result and bench_result.get("tg_speed")
        else ""
    )
    tg_stddev_f = (
        f"{float(bench_result['tg_stddev']):.1f}"
        if bench_result and bench_result.get("tg_stddev")
        else ""
    )
    ctx_val = format_ctx(chosen["ctx"])
    ngl_val = format_ngl(chosen["ngl"])
    moe_val = chosen["ot"]
    ubatch_val = str(chosen.get("ubatch", "")) if chosen.get("ubatch") is not None else ""

    row = {
        "model": display_name,
        "quant": quant,
        "provider": provider,
        "size_gib": size_gib,
        "params": n_params,
        "model_type": model_kind,
        "mmproj": mmproj_col,
        "vision": caps["vision"],
        "reason": caps["reasoning"],
        "switch": caps["switchable"],
        "effort": caps["effort"],
    }

    if vision_mode:
        row["fit_target"] = ""
        row["vfit_target"] = str(fit_target)
        row["vctx"] = ctx_val
        row["vngl"] = ngl_val
        row["vubatch"] = ubatch_val
        row["vmoe_cpu"] = moe_val
        row["vmoe_cpu_raw"] = chosen["ot_raw"] or ""
        row[VPP_COL] = pp_f
        row[VPP_STDDEV_COL] = pp_stddev_f
        row[VTG_COL] = tg_f
        row[VTG_STDDEV_COL] = tg_stddev_f
        row["vreps"] = str(reps)
        row["ctx"] = ""
        row["ngl"] = ""
        row["ubatch"] = ""
        row["moe_cpu"] = ""
        row["moe_cpu_raw"] = ""
        row[PP_COL] = ""
        row[PP_STDDEV_COL] = ""
        row[TG_COL] = ""
        row[TG_STDDEV_COL] = ""
        row["reps"] = ""
    else:
        row["fit_target"] = str(fit_target)
        row["vfit_target"] = ""
        row["ctx"] = ctx_val
        row["ngl"] = ngl_val
        row["ubatch"] = ubatch_val
        row["moe_cpu"] = moe_val
        row["moe_cpu_raw"] = chosen["ot_raw"] or ""
        row[PP_COL] = pp_f
        row[PP_STDDEV_COL] = pp_stddev_f
        row[TG_COL] = tg_f
        row[TG_STDDEV_COL] = tg_stddev_f
        row["reps"] = str(reps)
        row["vctx"] = ""
        row["vngl"] = ""
        row["vubatch"] = ""
        row["vmoe_cpu"] = ""
        row["vmoe_cpu_raw"] = ""
        row[VPP_COL] = ""
        row[VPP_STDDEV_COL] = ""
        row[VTG_COL] = ""
        row[VTG_STDDEV_COL] = ""
        row["vreps"] = ""

    append_result_row(row)
    sort_results_file()
    log(f"Results appended to {RESULTS_FILE}")


def benchmark_tag(tag, args):
    start_time = time.monotonic()
    quant = tag.split(":")[1] if ":" in tag else ""
    display_name = display_name_from_tag(tag)

    mmproj_mib = get_mmproj_size_mib(tag)
    if args.vision and mmproj_mib == 0:
        log("skipping non-vision model in vision mode")
        return

    max_ctx = get_max_ctx(tag, prio=args.prio)
    if args.vision:
        fit_target = fit_target_mib(mmproj_mib)
        log(f"vision model detected: mmproj={mmproj_mib} MiB, fit-target={fit_target} MiB")
    else:
        fit_target = fit_target_mib()
        mmproj_mib = get_mmproj_size_mib(tag)
        if mmproj_mib > 0:
            log(
                f"vision model detected (mmproj={mmproj_mib} MiB) — running in text mode (fit-target={fit_target})"
            )

    ctx_targets = build_ctx_list(max_ctx)

    log(f"Model: {tag}")
    log(f"Max ctx: {format_ctx(max_ctx)}")
    log(f"Candidate contexts: {', '.join(format_ctx(c) for c in ctx_targets)}")
    log()

    reused_choice = try_reuse_existing_fit_choice(
        tag,
        fit_target,
        max_ctx,
        args.vision,
        forced_ubatch=args.ubatch,
    )

    if reused_choice is not None:
        scan_results, chosen, reason, is_moe = reused_choice
    else:
        scan_results, chosen, reason, is_moe = choose_scan_strategy(
            tag,
            ctx_targets,
            fit_target,
            max_ctx,
            forced_ubatch=args.ubatch,
        )

    print_scan_table(scan_results, chosen)

    log()
    log(f"Selection: {reason}")
    if chosen:
        log(
            f"Chosen context: {format_ctx(chosen['ctx'])} (target {format_ctx(chosen['target_ctx'])}, ngl {format_ngl(chosen['ngl'])}, ub {chosen.get('ubatch', '?')})"
        )

    size_gib = "?"
    bench_result = None

    if args.list or not chosen:
        pass
    else:
        log()
        log("Benchmark:")
        log(
            f"ctx={format_ctx(chosen['ctx'])} ngl={format_ngl(chosen['ngl'])} ub={chosen.get('ubatch', '?')} ot={chosen['ot']}"
        )
        bench_start = time.monotonic()
        bench_result = run_bench(
            tag,
            chosen["ngl"],
            chosen["ot_raw"],
            chosen.get("ubatch", FIT_UBATCH_DENSE) if args.ubatch is None else args.ubatch,
            reps=args.reps,
            prio=args.prio,
        )
        if bench_result is None:
            log("benchmark failed")
        else:
            pp_f = f"{float(bench_result['pp_speed']):.1f}" if bench_result["pp_speed"] else "?"
            pp_stddev_f = (
                f"{float(bench_result['pp_stddev']):.1f}" if bench_result["pp_stddev"] else "?"
            )
            tg_f = f"{float(bench_result['tg_speed']):.1f}" if bench_result["tg_speed"] else "?"
            tg_stddev_f = (
                f"{float(bench_result['tg_stddev']):.1f}" if bench_result["tg_stddev"] else "?"
            )
            log(
                f"benchmark complete in {time.monotonic() - bench_start:.1f}s: pp{BENCH_PP}={pp_f}±{pp_stddev_f} tg{BENCH_TG}={tg_f}±{tg_stddev_f}"
            )
            if bench_result["size_bytes"]:
                size_gib = f"{int(bench_result['size_bytes']) / 1024**3:.2f}"

    repo = tag.split(":")[0] if ":" in tag else tag
    provider = repo.split("/")[0] if "/" in repo else repo
    print_summary(display_name, quant, provider, size_gib, chosen, is_moe, bench_result)

    if not args.list and chosen:
        caps = detect_capabilities(tag)
        log(
            f"Capabilities: vision={caps['vision']} reasoning={caps['reasoning']} switchable={caps['switchable']} effort={caps['effort']}"
        )
        write_result_row(
            tag,
            chosen,
            is_moe,
            bench_result,
            caps,
            vision_mode=args.vision,
            reps=args.reps,
            fit_target=fit_target,
        )

    log(f"Finished model in {time.monotonic() - start_time:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Pick and benchmark the best context size")
    parser.add_argument("tags", nargs="*", help="HF repo:quant tags")
    parser.add_argument(
        "--all", action="store_true", help="Run sequentially for all tags from models.toml"
    )
    parser.add_argument("-r", "--reps", type=int, default=REPS, help="Repetitions per test")
    parser.add_argument(
        "--list", action="store_true", help="Only list fit params, don't benchmark"
    )
    parser.add_argument(
        "--vision", action="store_true", help="Benchmark with mmproj VRAM budget (vision mode)"
    )
    parser.add_argument("-p", "--provider", action="append", help="Only benchmark models from this provider (e.g. unsloth)")
    parser.add_argument("-g", "--group", action="append", help="Only benchmark models in this group (e.g. qwen3.6-35b-a3b)")
    parser.add_argument("-ub", "--ubatch", type=int, default=None, help="Force ubatch value (e.g. 512 or 1024)")
    parser.add_argument(
        "--prio",
        type=int,
        choices=[-1, 0, 1, 2, 3],
        default=None,
        help="Pass llama-bench process priority through (--prio -1|0|1|2|3)",
    )
    parser.add_argument("--log-file", help="Write timestamped progress logs to this file")
    args = parser.parse_args()
    if not acquire_run_lock():
        print("fit_bench.py is already running; refusing parallel execution", file=sys.stderr)
        return 1
    set_log_file(args.log_file)

    if args.all and args.tags:
        parser.error("cannot use --all with explicit tags")
    if args.all:
        tags = [f"{repo}:{quant}" for repo, quant, _, _ in load_models()]
    elif args.tags:
        tags = args.tags
    elif args.provider or args.group:
        models = load_models()
        if args.provider:
            models = [m for m in models if m[0].split("/")[0] in args.provider]
        if args.group:
            models = [m for m in models if m[2] in args.group]
        tags = [f"{repo}:{quant}" for repo, quant, _, _ in models]
    else:
        tags = []

    if args.provider:
        tags = [t for t in tags if t.split("/")[0].split(":")[0] in args.provider]
    if args.group:
        all_models = load_models()
        group_repos = {m[0] for m in all_models if m[2] in args.group}
        tags = [t for t in tags if t.split(":")[0] in group_repos]

    if not tags:
        parser.error("provide at least one tag or use --all")

    total = len(tags)
    for i, tag in enumerate(tags, start=1):
        if total > 1:
            log(f"[{i}/{total}] {tag}")
        benchmark_tag(tag, args)
        if total > 1 and i != total:
            print(flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
