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
import io
import re
import subprocess
import time

from gguf_utils import detect_capabilities, get_max_ctx_from_gguf, get_mmproj_size_mib
from results import (
    BENCH_PP,
    BENCH_TG,
    RESULTS_FILE,
    append_result_row,
    display_name_from_tag,
    format_ctx,
    format_mmproj,
    format_ngl,
    format_params,
    load_tags,
    sort_results_file,
)

FIT_TARGET = 512
FLASH_ATTN = 1
REPS = 3
FIT_PARAMS_TIMEOUT = 600
BENCH_TIMEOUT = 900

_log_file = None


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


def get_fit_params(tag, target_ctx, fit_target=None):
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


def get_max_ctx(tag):
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
                str(FIT_TARGET),
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
            ],
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


def result_keeps_best_ngl(result, chosen, is_moe):
    if result["ctx"] is None or result["ngl"] is None:
        return False
    if is_moe:
        return result["ngl"] == chosen["ngl"]
    if chosen["ngl"] == -1:
        return result["ngl"] == -1
    return result["ngl"] == chosen["ngl"]


def build_refinement_ctx_list(results, chosen, max_ctx, is_moe):
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
        if not result_keeps_best_ngl(result, chosen, is_moe):
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


def scan_fit_configs(tag, ctx_targets, fit_target=None):
    results = []
    total = len(ctx_targets)
    for i, target_ctx in enumerate(ctx_targets, start=1):
        log(f"fit scan {i}/{total}: target ctx={format_ctx(target_ctx)}")
        params_str = get_fit_params(tag, target_ctx, fit_target=fit_target)
        if not params_str:
            log("fit-params failed")
            results.append(
                {"target_ctx": target_ctx, "ctx": None, "ngl": None, "ot_raw": None, "ot": None}
            )
            continue

        ctx, ngl, ot = parse_fit_params(params_str)
        log(f"fit result: ctx={format_ctx(ctx)} ngl={format_ngl(ngl)} moe={ot_label(ot)}")
        results.append(
            {"target_ctx": target_ctx, "ctx": ctx, "ngl": ngl, "ot_raw": ot, "ot": ot_label(ot)}
        )
    return results


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


def run_bench(tag, ngl, ot, reps=REPS):
    ngl_arg = 99 if ngl == -1 else ngl
    log(f"starting llama-bench for {tag} with ngl={format_ngl(ngl)} reps={reps}")
    cmd = [
        "llama-bench",
        "-hf",
        tag,
        "-fa",
        str(FLASH_ATTN),
        "-ngl",
        str(ngl_arg),
        "-pg",
        f"{BENCH_PP},{BENCH_TG}",
        "-o",
        "csv",
        "-r",
        str(reps),
    ]
    if ot:
        cmd += ot_to_bench_arg(ot)

    r = subprocess.run(cmd, capture_output=True, text=True, timeout=BENCH_TIMEOUT)
    if r.returncode != 0 and not r.stdout.strip():
        return None

    reader = csv.reader(io.StringIO(r.stdout))
    header = next(reader, None)
    if not header:
        return None

    n_gen_idx = header.index("n_gen") if "n_gen" in header else -1
    avg_ts_idx = header.index("avg_ts") if "avg_ts" in header else -1
    model_size_idx = header.index("model_size") if "model_size" in header else -1
    model_type_idx = header.index("model_type") if "model_type" in header else -1
    n_params_idx = header.index("model_n_params") if "model_n_params" in header else -1

    size_bytes = n_params = model_name = ""
    pp_speed = tg_speed = None

    for row in reader:
        if len(row) <= max(n_gen_idx, avg_ts_idx, model_size_idx):
            continue
        if model_size_idx >= 0 and not size_bytes:
            size_bytes = row[model_size_idx].strip('"')
        if model_type_idx >= 0 and not model_name:
            model_name = row[model_type_idx].strip('"')
        if n_params_idx >= 0 and not n_params:
            n_params = row[n_params_idx].strip('"')
        if n_gen_idx >= 0 and avg_ts_idx >= 0:
            n_gen = row[n_gen_idx].strip('"')
            ts = row[avg_ts_idx].strip('"')
            if n_gen == "0" and pp_speed is None:
                pp_speed = ts
            elif n_gen != "0" and tg_speed is None:
                tg_speed = ts

    if pp_speed is None and tg_speed is None:
        return None

    return {
        "model_name": model_name,
        "size_bytes": size_bytes,
        "n_params": n_params,
        "pp_speed": pp_speed,
        "tg_speed": tg_speed,
    }


def print_scan_table(scan_results, chosen):
    print("Fit scan:", flush=True)
    print("| Target Ctx | Actual Ctx | ngl | MoE CPU | Selected |", flush=True)
    print("|---|---|---|---|---|", flush=True)
    for r in scan_results:
        selected = "yes" if chosen and r["target_ctx"] == chosen["target_ctx"] else ""
        print(
            f"| {format_ctx(r['target_ctx'])} | {format_ctx(r['ctx'])} | {format_ngl(r['ngl'])} | {r['ot'] if r['ot'] else '?'} | {selected} |",
            flush=True,
        )


def print_summary(display_name, quant, provider, size_gib, chosen, is_moe, bench_result):
    print(flush=True)
    print("=" * 90, flush=True)
    print(f"Results for {display_name} ({quant}, {provider}, {size_gib} GiB)", flush=True)
    print("RTX 4070 Laptop (8GB VRAM), 64GB RAM, -fa on", flush=True)
    print(flush=True)
    header = f"| Type | Target Ctx | Actual Ctx | ngl | MoE CPU | pp{BENCH_PP} (t/s) | tg{BENCH_TG} (t/s) |"
    sep = "|---|---|---|---|---|---|---|"
    print(header, flush=True)
    print(sep, flush=True)
    if chosen:
        pp_f = (
            f"{float(bench_result['pp_speed']):.1f}"
            if bench_result and bench_result.get("pp_speed")
            else "-"
        )
        tg_f = (
            f"{float(bench_result['tg_speed']):.1f}"
            if bench_result and bench_result.get("tg_speed")
            else "-"
        )
        model_kind = "MoE" if is_moe else "Dense"
        print(
            f"| {model_kind} | {format_ctx(chosen['target_ctx'])} | {format_ctx(chosen['ctx'])} | {format_ngl(chosen['ngl'])} | {chosen['ot']} | {pp_f} | {tg_f} |",
            flush=True,
        )


def write_result_row(tag, chosen, is_moe, bench_result, caps, vision_mode):
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
    tg_f = (
        f"{float(bench_result['tg_speed']):.1f}"
        if bench_result and bench_result.get("tg_speed")
        else ""
    )
    ctx_val = format_ctx(chosen["ctx"])
    ngl_val = format_ngl(chosen["ngl"])
    moe_val = chosen["ot"]

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
        row["vctx"] = ctx_val
        row["vngl"] = ngl_val
        row["vpp2048_tps"] = pp_f
        row["vtg512_tps"] = tg_f
        row["ctx"] = ""
        row["ngl"] = ""
        row["moe_cpu"] = ""
        row["pp2048_tps"] = ""
        row["tg512_tps"] = ""
    else:
        row["ctx"] = ctx_val
        row["ngl"] = ngl_val
        row["moe_cpu"] = moe_val
        row["pp2048_tps"] = pp_f
        row["tg512_tps"] = tg_f
        row["vctx"] = ""
        row["vngl"] = ""
        row["vpp2048_tps"] = ""
        row["vtg512_tps"] = ""

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

    max_ctx = get_max_ctx(tag)
    if args.vision:
        fit_target = FIT_TARGET + mmproj_mib
        log(f"vision model detected: mmproj={mmproj_mib} MiB, fit-target={fit_target} MiB")
    else:
        fit_target = FIT_TARGET
        mmproj_mib = get_mmproj_size_mib(tag)
        if mmproj_mib > 0:
            log(
                f"vision model detected (mmproj={mmproj_mib} MiB) — running in text mode (fit-target={FIT_TARGET})"
            )

    ctx_targets = build_ctx_list(max_ctx)

    log(f"Model: {tag}")
    log(f"Max ctx: {format_ctx(max_ctx)}")
    log(f"Candidate contexts: {', '.join(format_ctx(c) for c in ctx_targets)}")
    log()

    scan_results = scan_fit_configs(tag, ctx_targets, fit_target=fit_target)
    chosen, reason, is_moe = select_best_result(scan_results)

    refinement_targets = build_refinement_ctx_list(scan_results, chosen, max_ctx, is_moe)
    if refinement_targets:
        log()
        log(f"Refinement contexts: {', '.join(format_ctx(c) for c in refinement_targets)}")
        extra_results = scan_fit_configs(tag, refinement_targets, fit_target=fit_target)
        scan_results = merge_scan_results(scan_results, extra_results)
        chosen, reason, is_moe = select_best_result(scan_results)

    print_scan_table(scan_results, chosen)

    log()
    log(f"Selection: {reason}")
    if chosen:
        log(
            f"Chosen context: {format_ctx(chosen['ctx'])} (target {format_ctx(chosen['target_ctx'])}, ngl {format_ngl(chosen['ngl'])})"
        )

    size_gib = "?"
    bench_result = None

    if args.list or not chosen:
        pass
    else:
        log()
        log("Benchmark:")
        log(f"ctx={format_ctx(chosen['ctx'])} ngl={format_ngl(chosen['ngl'])} ot={chosen['ot']}")
        bench_start = time.monotonic()
        bench_result = run_bench(tag, chosen["ngl"], chosen["ot_raw"], reps=args.reps)
        if bench_result is None:
            log("benchmark failed")
        else:
            pp_f = f"{float(bench_result['pp_speed']):.1f}" if bench_result["pp_speed"] else "?"
            tg_f = f"{float(bench_result['tg_speed']):.1f}" if bench_result["tg_speed"] else "?"
            log(
                f"benchmark complete in {time.monotonic() - bench_start:.1f}s: pp{BENCH_PP}={pp_f} tg{BENCH_TG}={tg_f}"
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
        write_result_row(tag, chosen, is_moe, bench_result, caps, vision_mode=args.vision)

    log(f"Finished model in {time.monotonic() - start_time:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Pick and benchmark the best context size")
    parser.add_argument("tags", nargs="*", help="HF repo:quant tags")
    parser.add_argument(
        "--all", action="store_true", help="Run sequentially for all tags from models.toml"
    )
    parser.add_argument("--reps", type=int, default=REPS, help="Repetitions per test")
    parser.add_argument(
        "--list", action="store_true", help="Only list fit params, don't benchmark"
    )
    parser.add_argument(
        "--vision", action="store_true", help="Benchmark with mmproj VRAM budget (vision mode)"
    )
    parser.add_argument("--log-file", help="Write timestamped progress logs to this file")
    args = parser.parse_args()

    set_log_file(args.log_file)

    if args.all:
        tags = load_tags()
    else:
        tags = args.tags

    if not tags:
        parser.error("provide at least one tag or use --all")

    total = len(tags)
    for i, tag in enumerate(tags, start=1):
        if total > 1:
            log(f"[{i}/{total}] {tag}")
        benchmark_tag(tag, args)
        if total > 1 and i != total:
            print(flush=True)


if __name__ == "__main__":
    main()
