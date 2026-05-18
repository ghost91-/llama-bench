#!/usr/bin/env python3
"""Pick and benchmark the best context size using llama-fit-params + llama-bench.

Selection rules:
1. Dense models: choose the highest context that still fits fully in VRAM (`-ngl all`).
2. MoE models: choose the highest context that still keeps the maximum achievable `ngl`.

Candidate contexts use a coarse ladder: 5k, 10k, 20k, 30k, 40k, 50k, 75k, 100k,
125k, 150k, 175k, 200k, then 50k steps up to 300k, then 100k steps upward. After
the coarse scan, a refinement pass fills 25k gaps near the boundary where ngl drops
below the best.

Note: for MoE models, llama-fit-params uses a greedy overflow strategy per partial
layer that can produce non-monotonic ngl (ngl can go back up at higher context).
The scan must therefore run to completion to find the true max ngl.

MoE models are scanned at multiple ubatch sizes (512, 1024, 2048, 4096) to capture
the context-size vs prompt-processing-speed trade-off. Dense models use ub=512 only.

Usage:
    python fit_bench.py unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_M
    python fit_bench.py unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_M --scan
"""

import argparse
import csv
import fcntl
import io
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from functools import cache
from typing import IO, Callable, Sequence, TypeAlias, TypedDict, cast

from llama_bench.gguf_utils import (
    detect_capabilities,
    get_max_ctx_from_gguf,
    get_mmproj_size_mib,
    is_moe_model,
)
from llama_bench.model_identity import identity_from_tag, render_model_tag
from llama_bench.results import (
    BENCH_PP,
    BENCH_TG,
    PP_COL,
    PP_STDDEV_COL,
    RESULTS_FILE,
    TG_COL,
    TG_STDDEV_COL,
    append_result_row,
    format_ctx,
    format_mmproj,
    format_ngl,
    format_params,
    get_bench_ts,
    load_models,
    sort_results_file,
)
from llama_bench.schema_types import Capabilities, ReasoningDetails, ResultRow, ScanCache, ScanEntry
from llama_bench.scan_cache import (
    get_capabilities,
    get_cached_max_ctx,
    get_model_moe,
    get_reusable_scan_entry,
    get_scan_entry,
    load_scan_cache,
    save_scan_cache,
    set_cached_max_ctx,
    set_model_moe,
    set_scan_entry,
    SCAN_CACHE_FILE,
)

FIT_TARGET = 256
FLASH_ATTN = 1
REPS = 20
BENCH_BATCH = 4096
FIT_UBATCH_DENSE = 512
MOE_UBATCH_SIZES = [512, 1024, 2048, 4096]
VALID_UBATCH_SIZES = sorted(set([FIT_UBATCH_DENSE] + MOE_UBATCH_SIZES))
FIT_PARAMS_TIMEOUT = 600
BENCH_TIMEOUT = 1800
BENCH_FAILURE_OUTPUT_LINES = 25
MODEL_BOUNDARY = "#" * 100
MODEL_START_MARKER = "### MODEL START"
MODEL_END_MARKER = "### MODEL END"
BASE_CTX_STEPS = [
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


class FitResult(TypedDict):
    target_ctx: int
    ctx: int | None
    ngl: int | None
    ubatch: int
    offload: int | None
    ot: str | None


class BenchMetricRow(TypedDict):
    size_bytes: str
    n_params: str
    speed: str
    stddev: str


class BenchResult(TypedDict):
    size_bytes: str
    n_params: str
    pp_speed: str
    pp_stddev: str
    tg_speed: str
    tg_stddev: str


class Args(argparse.Namespace):
    tags: list[str]
    reps: int
    scan: bool
    rescan: str | None
    rebench: str | None
    vision: bool
    provider: list[str] | None
    group: list[str] | None
    ubatch: int | None
    prio: int | None
    log_file: str | None
    print_commands: bool
    rescan_cutoff: datetime | None
    rebench_cutoff: datetime | None


StopOnMatch: TypeAlias = Callable[[FitResult], bool]
ScanStrategyResult: TypeAlias = tuple[list[FitResult], FitResult | None, str, bool]
ScanContextProvider: TypeAlias = Callable[[], tuple[int | None, Sequence[int]]]

_log_file: str | None = None
_run_lock: IO[str] | None = None


def _parse_resume_age(age_str: str) -> datetime | None:
    m = re.fullmatch(r"(\d+)([mhd])", age_str)
    if not m:
        return None
    value, unit = int(m.group(1)), m.group(2)
    deltas = {"m": timedelta(minutes=value), "h": timedelta(hours=value), "d": timedelta(days=value)}
    return datetime.now(timezone.utc) - deltas[unit]


def log(message: str = "") -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {message}" if message else ""
    print(line, flush=True)
    if _log_file and line:
        with open(_log_file, "a") as f:
            f.write(line + "\n")


def log_model_header(
    tag: str,
    mode: str,
    fit_target: int | None,
    mmproj_mib: int | None,
    index: int | None,
    total: int | None,
) -> None:
    prefix = f"MODEL {index}/{total}" if index is not None and total is not None else "MODEL"
    log()
    log(MODEL_BOUNDARY)
    log(f"{MODEL_START_MARKER} | label={prefix} | mode={mode} | tag={tag}")
    if fit_target is not None and mmproj_mib is not None:
        log(f"### MODEL META  | fit_target_mib={fit_target} | mmproj_mib={mmproj_mib}")
    log(MODEL_BOUNDARY)


def log_ubatch_header(ubatch: int) -> None:
    log(f"ubatch | start | value={ubatch}")


def log_model_footer(elapsed: float) -> None:
    log(f"{MODEL_END_MARKER} | elapsed={elapsed:.1f}s")
    log(MODEL_BOUNDARY)


def log_process_output(label: str, output: str) -> None:
    lines = output.strip().splitlines()
    if not lines:
        log(f"bench | {label} | empty=true")
        return
    omitted = max(0, len(lines) - BENCH_FAILURE_OUTPUT_LINES)
    log(f"bench | {label} | begin")
    if omitted:
        log(f"bench | {label} | omitted_lines={omitted}")
    for line in lines[-BENCH_FAILURE_OUTPUT_LINES:]:
        log(f"bench | {label} | {line}")


def log_bench_failure(run: subprocess.CompletedProcess[str], reason: str) -> None:
    log(f"bench | fail | reason={reason} | return_code={run.returncode}")
    log_process_output("stderr", run.stderr)
    log_process_output("stdout", run.stdout)


def set_log_file(path: str | None) -> None:
    global _log_file
    _log_file = path
    if path:
        open(path, "w").close()


def acquire_run_lock() -> bool:
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


def get_fit_params(
    tag: str, target_ctx: int, fit_target: int | None = None, ubatch: int = FIT_UBATCH_DENSE
) -> str | None:
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
                "-b",
                str(max(BENCH_BATCH, ubatch)),
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
        log(f"fit_params | timeout | ctx={format_ctx(target_ctx)}")
        return None
    except Exception as e:
        log(f"fit_params | fail | ctx={format_ctx(target_ctx)} | error={e}")
        return None
    for line in r.stdout.strip().splitlines():
        line = line.strip()
        if line.startswith("-c "):
            return line
    return None


def get_max_ctx(tag: str, fit_target: int = FIT_TARGET, prio: int | None = None) -> int | None:
    start = time.monotonic()
    log(f"max_ctx | resolve | tag={tag}")
    try:
        ctx = get_max_ctx_from_gguf(tag)
        if ctx is not None:
            log(
                f"max_ctx | resolved | source=gguf | elapsed={time.monotonic() - start:.1f}s | ctx={format_ctx(ctx)}"
            )
            return ctx
    except Exception as e:
        log(f"max_ctx | source_fail | source=gguf | error={e}")

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
                f"max_ctx | resolved | source=llama-fit-params | elapsed={time.monotonic() - start:.1f}s | ctx={format_ctx(int(m.group(1)))}"
            )
            return int(m.group(1))
    except subprocess.TimeoutExpired:
        log("max_ctx | source_timeout | source=llama-fit-params")

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
                f"max_ctx | resolved | source=llama-bench | elapsed={time.monotonic() - start:.1f}s | ctx={format_ctx(int(m2.group(1)))}"
            )
            return int(m2.group(1))
    except subprocess.TimeoutExpired:
        log("max_ctx | source_timeout | source=llama-bench")

    log(f"max_ctx | fail | elapsed={time.monotonic() - start:.1f}s")
    return None


def get_cached_or_resolve_max_ctx(tag: str, args: Args, cache: ScanCache) -> int | None:
    cached = get_cached_max_ctx(cache, tag, args.rescan_cutoff)
    if cached is not None:
        log(f"max_ctx | cache_hit | ctx={format_ctx(cached)}")
        return cached

    max_ctx = get_max_ctx(tag, prio=args.prio)
    if max_ctx is not None:
        set_cached_max_ctx(
            cache,
            tag,
            max_ctx,
            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S%z"),
        )
        save_scan_cache(cache)
    return max_ctx


def build_ctx_list(max_ctx: int | None) -> list[int]:
    if not max_ctx:
        return BASE_CTX_STEPS.copy()

    steps = [ctx for ctx in BASE_CTX_STEPS if ctx <= max_ctx]
    ctx = 250000
    while ctx <= max_ctx and ctx <= 300000:
        steps.append(ctx)
        ctx += 50000

    ctx = 400000
    while ctx <= max_ctx:
        steps.append(ctx)
        ctx += 100000

    if max_ctx not in steps:
        steps.append(max_ctx)

    return steps


def load_existing_fit_choice(
    tag: str,
    fit_target: int,
    vision_mode: bool,
    ubatch: int,
    cache: ScanCache,
) -> FitResult | None:
    entry = get_scan_entry(cache, tag, vision_mode, ubatch)
    if entry is None:
        return None

    if entry.get("fit_target") != fit_target:
        return None

    ctx = int(entry["ctx"])
    ngl = entry["ngl"]

    ot = entry.get("ot")
    offload = entry.get("offload")
    if (offload is None) != (ot is None):
        return None

    return {
        "target_ctx": ctx,
        "ctx": ctx,
        "ngl": ngl,
        "ubatch": ubatch,
        "offload": offload,
        "ot": ot,
    }


def parse_fit_params(params_str: str) -> tuple[int | None, int | None, str | None]:
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
            ot_parts: list[str] = []
            while i < len(parts) and not parts[i].startswith("-"):
                ot_parts.append(parts[i])
                i += 1
            ot = " ".join(ot_parts).strip('"')
        else:
            i += 1
    return c, ngl, ot


def fit_ctx(result: FitResult) -> int:
    ctx = result["ctx"]
    if ctx is None:
        raise ValueError("fit result is missing ctx")
    return ctx


def fit_ngl(result: FitResult) -> int:
    ngl = result["ngl"]
    if ngl is None:
        raise ValueError("fit result is missing ngl")
    return ngl


def result_matches_target(result: FitResult, target_ngl: int) -> bool:
    return result["ctx"] is not None and result["ngl"] == target_ngl


def build_refinement_ctx_list(
    results: Sequence[FitResult], chosen: FitResult | None, max_ctx: int | None, target_ngl: int
) -> list[int]:
    if not chosen or not max_ctx or chosen["ctx"] is None or chosen["ctx"] >= max_ctx:
        return []

    chosen_ctx = fit_ctx(chosen)
    valid = sorted([r for r in results if r["ctx"] is not None and r["ngl"] is not None], key=fit_ctx)

    upper: int | None = None
    for result in valid:
        result_ctx = fit_ctx(result)
        if result_ctx <= chosen_ctx:
            continue
        if not result_matches_target(result, target_ngl):
            upper = result_ctx
            break

    if upper is None:
        upper = max_ctx

    if upper <= chosen_ctx + 25000:
        return []

    targets: list[int] = []
    ctx = chosen_ctx + 25000
    while ctx < upper:
        targets.append(ctx)
        ctx += 25000

    if upper not in targets and upper != chosen_ctx:
        targets.append(upper)

    scanned = {r["target_ctx"] for r in results}
    return [ctx for ctx in targets if ctx not in scanned]


def merge_scan_results(results: Sequence[FitResult], extra_results: Sequence[FitResult]) -> list[FitResult]:
    merged = {r["target_ctx"]: r for r in results}
    for result in extra_results:
        merged[result["target_ctx"]] = result
    return [merged[key] for key in sorted(merged)]


def ot_to_bench_arg(ot: str | None) -> list[str]:
    if not ot:
        return []
    return ["-ot", ot.replace(",", ";")]


def count_offload(ot: str | None) -> int | None:
    if not ot:
        return None
    return ot.count(",") + 1


def format_offload(offload: int | None) -> str:
    if offload is None:
        return ""
    return str(offload)


def format_bench_metric(value: str | None, missing: str = "") -> str:
    if not value:
        return missing
    return f"{float(value):.1f}"


def bench_is_fresh(tag: str, mode: str, ubatch: int, cutoff: datetime) -> bool:
    ts = get_bench_ts(tag, mode=mode, ubatch=ubatch)
    return ts is not None and ts >= cutoff


def expected_rebench_ubatches(args: Args, cache: ScanCache, tag: str) -> Sequence[int] | None:
    if args.ubatch is not None:
        return [args.ubatch]

    model_is_moe = get_model_moe(cache, tag)
    if model_is_moe is None:
        return None
    if model_is_moe:
        return MOE_UBATCH_SIZES
    return [FIT_UBATCH_DENSE]


def should_skip_rebench_model(tag: str, args: Args, cache: ScanCache, mode: str) -> bool:
    cutoff = args.rebench_cutoff
    if cutoff is None:
        return False

    ubatches = expected_rebench_ubatches(args, cache, tag)
    if ubatches is None:
        return False
    return all(bench_is_fresh(tag, mode, ubatch, cutoff) for ubatch in ubatches)


def parse_bench_row(
    output: str, target_prompt: int, target_gen: int, target_depth: int
) -> BenchMetricRow | None:
    reader = csv.DictReader(io.StringIO(output))
    size_bytes = ""
    n_params = ""
    for row in reader:
        def field(name: str) -> str:
            return row.get(name, "").strip('"')

        if not size_bytes:
            size_bytes = field("model_size")
        if not n_params:
            n_params = field("model_n_params")
        if (
            field("n_prompt") == str(target_prompt)
            and field("n_gen") == str(target_gen)
            and field("n_depth") == str(target_depth)
        ):
            return {
                "size_bytes": size_bytes,
                "n_params": n_params,
                "speed": field("avg_ts"),
                "stddev": field("stddev_ts"),
            }
    return None


def build_fit_result(target_ctx: int, ubatch: int, params_str: str | None) -> FitResult:
    ctx: int | None = None
    ngl: int | None = None
    ot: str | None = None
    if params_str:
        ctx, ngl, ot = parse_fit_params(params_str)
    return {
        "target_ctx": target_ctx,
        "ctx": ctx,
        "ngl": ngl,
        "ubatch": ubatch,
        "offload": count_offload(ot),
        "ot": ot,
    }


def scan_fit_configs(
    tag: str,
    ctx_targets: Sequence[int],
    fit_target: int | None = None,
    ubatch: int = FIT_UBATCH_DENSE,
    stop_on_match: StopOnMatch | None = None,
    existing_results: Sequence[FitResult] | None = None,
) -> list[FitResult]:
    results = list(existing_results or [])
    existing_by_ctx = {r["target_ctx"]: r for r in results}
    total = len(ctx_targets)
    for i, target_ctx in enumerate(ctx_targets, start=1):
        if target_ctx in existing_by_ctx:
            log(
                f"scan | step | index={i}/{total} | target={format_ctx(target_ctx)} | ub={ubatch} | source=probe"
            )
            if stop_on_match and stop_on_match(existing_by_ctx[target_ctx]):
                break
            continue
        log(f"scan | step | index={i}/{total} | target={format_ctx(target_ctx)} | ub={ubatch}")
        params_str = get_fit_params(tag, target_ctx, fit_target=fit_target, ubatch=ubatch)
        result = build_fit_result(target_ctx, ubatch, params_str)
        if not params_str:
            log("scan | fail | reason=no_fit_params")
            results.append(result)
            continue

        log(
            f"scan | result | ctx={format_ctx(result['ctx'])} | ngl={format_ngl(result['ngl'])} | offload={format_offload(result['offload'])}"
        )
        results.append(result)
        if stop_on_match and stop_on_match(result):
            break
    return results


def ngl_rank(ngl: int | None) -> int:
    if ngl is None:
        return -1
    if ngl == -1:
        return 10**9
    return ngl


def prefer_result(left: FitResult | None, right: FitResult | None) -> FitResult | None:
    if left is None:
        return right
    if right is None:
        return left
    left_key = (ngl_rank(left["ngl"]), left.get("ubatch", 0), left["ctx"] or 0)
    right_key = (ngl_rank(right["ngl"]), right.get("ubatch", 0), right["ctx"] or 0)
    return left if left_key >= right_key else right


def probe_fit_config(tag: str, target_ctx: int, fit_target: int, ubatch: int) -> FitResult | None:
    params_str = get_fit_params(tag, target_ctx, fit_target=fit_target, ubatch=ubatch)
    if not params_str:
        log(f"probe | fail | target={format_ctx(target_ctx)} | ub={ubatch}")
        return None

    result = build_fit_result(target_ctx, ubatch, params_str)
    log(
        f"probe | ok | target={format_ctx(target_ctx)} | ub={ubatch} | ngl={format_ngl(result['ngl'])} | offload={format_offload(result['offload'])}"
    )
    return result


def result_below_target(result: FitResult, target_ngl: int) -> bool:
    return result["ctx"] is not None and result["ngl"] is not None and result["ngl"] < target_ngl


def choose_target_result(
    results: Sequence[FitResult], target_ngl: int, descending: bool
) -> FitResult | None:
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
    tag: str,
    ctx_targets: Sequence[int],
    fit_target: int,
    max_ctx: int | None,
    ubatch: int,
    target_ngl: int,
    descending: bool,
    probe_result: FitResult | None = None,
) -> tuple[list[FitResult], FitResult | None]:
    def stop_on_match(result: FitResult) -> bool:
        if descending:
            return result_matches_target(result, target_ngl)
        return result_below_target(result, target_ngl)

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
        log(f"scan | refine | candidates={','.join(format_ctx(c) for c in refinement_targets)}")
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


def _reasoning_details(caps: Capabilities) -> ReasoningDetails | None:
    reasoning = caps["reasoning"]
    return reasoning if reasoning is not False else None


def reason_for_target_scan(chosen: FitResult | None, target_ngl: int, is_moe: bool) -> str:
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


def fallback_scan_strategy(
    tag: str, ctx_targets: Sequence[int], fit_target: int, ubatch: int, is_moe: bool
) -> ScanStrategyResult:
    results = scan_fit_configs(
        tag, sorted(ctx_targets, reverse=True), fit_target=fit_target, ubatch=ubatch
    )
    chosen, reason = select_best_result(results, is_moe)
    return results, chosen, reason, is_moe


def finalize_target_scan(
    results: list[FitResult], chosen: FitResult | None, target_ngl: int, is_moe: bool
) -> ScanStrategyResult:
    if chosen is None:
        chosen, reason = select_best_result(results, is_moe)
        return results, chosen, reason, is_moe
    return results, chosen, reason_for_target_scan(chosen, target_ngl, is_moe), is_moe


def resolve_probe_strategy(
    probe_dense: FitResult | None, probe_moe: FitResult | None, is_moe: bool
) -> tuple[FitResult, int, bool] | None:
    probes = [p for p in (probe_dense, probe_moe) if p is not None and p["ngl"] is not None]
    if not probes:
        return None

    if is_moe:
        target_ngl = max(fit_ngl(p) for p in probes)
    else:
        target_ngl = -1 if any(fit_ngl(p) == -1 for p in probes) else max(fit_ngl(p) for p in probes)

    selected_probe = next(
        (p for p in (probe_moe, probe_dense) if p is not None and p["ngl"] == target_ngl),
        prefer_result(probe_moe, probe_dense),
    )
    if selected_probe is None:
        return None
    return selected_probe, target_ngl, is_moe


def choose_scan_strategy(
    tag: str,
    ctx_targets: Sequence[int],
    fit_target: int,
    max_ctx: int | None,
    is_moe: bool,
    forced_ubatch: int | None = None,
) -> ScanStrategyResult:
    probe_ctx = ctx_targets[0]
    if forced_ubatch is not None:
        probe = probe_fit_config(tag, probe_ctx, fit_target, forced_ubatch)
        if probe is None or probe["ngl"] is None:
            return fallback_scan_strategy(tag, ctx_targets, fit_target, forced_ubatch, is_moe)

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
    probe_moe = probe_fit_config(tag, probe_ctx, fit_target, MOE_UBATCH_SIZES[1])
    strategy = resolve_probe_strategy(probe_dense, probe_moe, is_moe)

    if strategy is None:
        log("scan | fallback | reason=probes_failed | strategy=descending | ub=512")
        return fallback_scan_strategy(tag, ctx_targets, fit_target, FIT_UBATCH_DENSE, is_moe)

    selected_probe, target_ngl, is_moe = strategy
    selected_ubatch = selected_probe["ubatch"]
    model_type = "MoE" if is_moe else "Dense"
    log(
        f"scan | strategy | type={model_type} | target_ngl={format_ngl(target_ngl)} | ub={selected_ubatch}"
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


def select_best_result(results: Sequence[FitResult], is_moe: bool) -> tuple[FitResult | None, str]:
    valid = [r for r in results if r["ctx"] is not None and r["ngl"] is not None]
    if not valid:
        return None, "No valid fit-params results"

    if is_moe:
        max_ngl = max(fit_ngl(r) for r in valid)
        matches = [r for r in valid if r["ngl"] == max_ngl]
        chosen = max(matches, key=fit_ctx)
        return chosen, f"MoE: highest context that keeps max ngl ({format_ngl(max_ngl)})"

    full_vram = [r for r in valid if r["ngl"] == -1]
    if full_vram:
        chosen = max(full_vram, key=fit_ctx)
        return chosen, "Dense: highest context that still fits fully in VRAM"

    max_ngl = max(fit_ngl(r) for r in valid)
    matches = [r for r in valid if r["ngl"] == max_ngl]
    chosen = max(matches, key=fit_ctx)
    return (
        chosen,
        f"Dense fallback: no full-VRAM fit found, using highest context at max ngl ({format_ngl(max_ngl)})",
    )


def build_bench_command(
    tag: str, ngl: int, ot: str | None, ubatch: int, reps: int, prio: int | None
) -> list[str]:
    ngl_arg = 99 if ngl == -1 else ngl
    batch = max(BENCH_BATCH, ubatch)
    cmd = [
        "llama-bench",
        "-hf",
        tag,
        "-fa",
        str(FLASH_ATTN),
        "-ngl",
        str(ngl_arg),
        "-b",
        str(batch),
        "-ub",
        str(ubatch),
        "-o",
        "csv",
        "-r",
        str(reps),
    ]
    if prio is not None:
        cmd += ["--prio", str(prio)]
    if ot:
        cmd += ot_to_bench_arg(ot)
    return cmd + ["-p", str(BENCH_PP), "-n", str(BENCH_TG), "-d", "0"]


def run_bench(
    tag: str, ngl: int, ot: str | None, ubatch: int, reps: int = REPS, prio: int | None = None
) -> BenchResult | None:
    cmd = build_bench_command(tag, ngl, ot, ubatch, reps, prio)

    try:
        run = subprocess.run(cmd, capture_output=True, text=True, timeout=BENCH_TIMEOUT)
    except subprocess.TimeoutExpired:
        log(f"bench | timeout | timeout_s={BENCH_TIMEOUT}")
        return None
    if run.returncode != 0 and not run.stdout.strip():
        log_bench_failure(run, "process exited without stdout")
        return None

    pp_result = parse_bench_row(run.stdout, BENCH_PP, 0, 0)
    tg_result = parse_bench_row(run.stdout, 0, BENCH_TG, 0)
    if pp_result is None or tg_result is None:
        log_bench_failure(run, "missing expected CSV result rows")
        return None

    return {
        "size_bytes": pp_result["size_bytes"] or tg_result["size_bytes"],
        "n_params": pp_result["n_params"] or tg_result["n_params"],
        "pp_speed": pp_result["speed"],
        "pp_stddev": pp_result["stddev"],
        "tg_speed": tg_result["speed"],
        "tg_stddev": tg_result["stddev"],
    }


def print_markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str | int | None]]) -> None:
    widths = [len(header) for header in headers]
    normalized_rows: list[list[str]] = []
    for row in rows:
        normalized = ["" if value is None else str(value) for value in row]
        normalized_rows.append(normalized)
        for i, value in enumerate(normalized):
            widths[i] = max(widths[i], len(value))

    def format_row(row: Sequence[str]) -> str:
        cells = [f"{value:<{widths[i]}}" for i, value in enumerate(row)]
        return f"| {' | '.join(cells)} |"

    separator = "|-" + "-|-".join("-" * width for width in widths) + "-|"
    log(format_row(headers))
    log(separator)
    for row in normalized_rows:
        log(format_row(row))


def print_scan_table(scan_results: Sequence[FitResult], chosen: FitResult | None) -> None:
    log("scan | table")
    rows: list[list[str | int | None]] = []
    for r in scan_results:
        selected = "*" if chosen and r["target_ctx"] == chosen["target_ctx"] else ""
        rows.append(
            [
                format_ctx(r["target_ctx"]),
                format_ctx(r["ctx"]),
                format_ngl(r["ngl"]),
                r.get("ubatch", "?"),
                format_offload(r.get("offload")),
                selected,
            ]
        )
    print_markdown_table(["Target Ctx", "Actual Ctx", "ngl", "ub", "Offload", "Selected"], rows)


def print_summary(
    display_name: str,
    quant: str,
    provider: str,
    size_gib: str,
    chosen: FitResult | None,
    is_moe: bool,
    bench_result: BenchResult | None,
) -> None:
    log("result | summary")
    result_label = f"{quant}, {provider}"
    if size_gib:
        result_label = f"{result_label}, {size_gib} GiB"
    log(f"result | model | name={display_name} | label={result_label}")
    log(f"result | system | summary={system_summary_line()}")
    headers = [
        "Type",
        "Target Ctx",
        "Actual Ctx",
        "ngl",
        "ub",
        "Offload",
        f"pp{BENCH_PP} (t/s)",
        f"tg{BENCH_TG} (t/s)",
    ]
    rows: list[list[str | int | None]] = []
    if chosen:
        chosen_ubatch = chosen["ubatch"]
        pp_f = format_bench_metric(bench_result.get("pp_speed") if bench_result else None)
        tg_f = format_bench_metric(bench_result.get("tg_speed") if bench_result else None)
        model_kind = "MoE" if is_moe else "Dense"
        rows.append(
            [
                model_kind,
                format_ctx(chosen["target_ctx"]),
                format_ctx(chosen["ctx"]),
                format_ngl(chosen["ngl"]),
                chosen_ubatch,
                format_offload(chosen.get("offload")),
                pp_f,
                tg_f,
            ]
        )
    print_markdown_table(headers, rows)


@cache
def system_summary_line() -> str:
    parts: list[str] = []
    gpu = gpu_summary()
    ram = ram_summary()
    if gpu:
        parts.append(gpu)
    if ram:
        parts.append(ram)
    parts.append(f"-fa {'on' if FLASH_ATTN else 'off'}")
    return ", ".join(parts)


def gpu_summary() -> str | None:
    return nvidia_proc_gpu_summary()


def nvidia_proc_gpu_summary(root: str = "/proc/driver/nvidia/gpus") -> str | None:
    if not os.path.isdir(root):
        return None
    try:
        gpu_dirs = sorted(os.listdir(root))
    except OSError:
        return None
    for gpu_dir in gpu_dirs:
        info_path = os.path.join(root, gpu_dir, "information")
        try:
            with open(info_path) as f:
                lines = f.readlines()
        except OSError:
            continue
        model = None
        for line in lines:
            if line.startswith("Model:"):
                model = line.split(":", 1)[1].strip()
                break
        if model:
            return model
    return None

def ram_summary() -> str | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError):
        return None
    if pages <= 0 or page_size <= 0:
        return None
    ram_gib = round(pages * page_size / 1024**3)
    return f"{ram_gib} GiB RAM"


def write_result_row(
    tag: str,
    chosen: FitResult,
    is_moe: bool,
    bench_result: BenchResult | None,
    _caps: Capabilities,
    mode: str,
    fit_target: int,
    ubatch: int,
    reps: int,
) -> None:
    identity = identity_from_tag(tag, require_quant=False)
    size_gib = ""
    n_params = ""
    if bench_result:
        if bench_result["size_bytes"]:
            size_gib = f"{int(bench_result['size_bytes']) / 1024**3:.2f}"
        if bench_result["n_params"]:
            n_params = format_params(bench_result["n_params"])
    pp_f = format_bench_metric(bench_result.get("pp_speed") if bench_result else None)
    pp_stddev_f = format_bench_metric(bench_result.get("pp_stddev") if bench_result else None)
    tg_f = format_bench_metric(bench_result.get("tg_speed") if bench_result else None)
    tg_stddev_f = format_bench_metric(bench_result.get("tg_stddev") if bench_result else None)
    ctx_val = format_ctx(fit_ctx(chosen))
    ngl_val = format_ngl(fit_ngl(chosen))
    offload_val = format_offload(chosen.get("offload"))

    row: ResultRow = {
        "model": identity.display_name,
        "quant": identity.quant,
        "provider": identity.provider,
        "mode": mode,
        "size_gib": size_gib,
        "params": n_params,
        "moe": "true" if is_moe else "false",
        "fit_target": str(fit_target),
        "ctx": ctx_val,
        "ngl": ngl_val,
        "ubatch": str(ubatch),
        "offload": offload_val,
        PP_COL: pp_f,
        PP_STDDEV_COL: pp_stddev_f,
        TG_COL: tg_f,
        TG_STDDEV_COL: tg_stddev_f,
        "reps": str(reps),
        "bench_ts": time.strftime("%Y-%m-%dT%H:%M:%S%z") if bench_result else "",
    }

    append_result_row(row)
    sort_results_file()
    log(f"result | write | file={RESULTS_FILE}")


def write_scan_cache_entry(
    cache: ScanCache,
    tag: str,
    vision: bool,
    fit_target: int,
    chosen: FitResult,
    ubatch: int,
    mmproj_mib: int,
    is_moe: bool,
    caps: Capabilities | None = None,
) -> Capabilities:
    ctx = fit_ctx(chosen)
    ngl = fit_ngl(chosen)
    if caps is None:
        caps = detect_capabilities(tag)
    scan_entry: ScanEntry = {
        "fit_target": fit_target,
        "ctx": ctx,
        "ngl": ngl,
        "offload": chosen.get("offload"),
        "ot": chosen.get("ot"),
        "scan_ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    set_model_moe(cache, tag, is_moe)
    set_scan_entry(
        cache, tag, vision, ubatch, scan_entry,
        mmproj=format_mmproj(mmproj_mib), caps=caps,
    )
    save_scan_cache(cache)
    return caps


def scan_and_bench_ubatch(
    tag: str,
    args: Args,
    cache: ScanCache,
    fit_target: int,
    ubatch: int,
    mode: str,
    mmproj_mib: int,
    model_is_moe: bool,
    max_ctx: int | None = None,
    ctx_targets: Sequence[int] | None = None,
    scan_contexts: ScanContextProvider | None = None,
    caps: Capabilities | None = None,
) -> tuple[FitResult | None, bool, BenchResult | None, bool]:
    vision_mode = mode == "vision"
    mode_label = mode

    cached_entry = get_reusable_scan_entry(
        cache,
        tag,
        vision_mode,
        ubatch,
        fit_target,
        rescan_cutoff=args.rescan_cutoff,
    )
    need_scan = cached_entry is None

    scan_results: list[FitResult] | None = None
    chosen: FitResult | None = None
    reason = ""
    is_moe = model_is_moe
    did_scan = False

    if not need_scan and cached_entry is not None:
        existing = load_existing_fit_choice(tag, fit_target, vision_mode, ubatch, cache)
        if existing is not None:
            scan_results = [existing]
            chosen = existing
            reason = f"Reused cached {mode_label} fit choice"
            is_moe = model_is_moe
            log(
                f"cache | hit | mode={mode_label} | ub={ubatch} | ctx={format_ctx(chosen['ctx'])} | ngl={format_ngl(chosen['ngl'])}"
            )
            if args.scan:
                log(
                    f"scan | skip | reason=cache_hit | mode={mode_label} | ub={ubatch} | force=--rescan"
                )
        else:
            need_scan = True

    if need_scan:
        if ctx_targets is None:
            if scan_contexts is not None:
                max_ctx, ctx_targets = scan_contexts()
            else:
                if max_ctx is None:
                    max_ctx = get_max_ctx(tag, prio=args.prio)
                ctx_targets = build_ctx_list(max_ctx)
        elif max_ctx is None:
            max_ctx = get_max_ctx(tag, prio=args.prio)

        log(f"scan | plan | max_ctx={format_ctx(max_ctx)}")
        log(f"scan | plan | candidates={','.join(format_ctx(c) for c in ctx_targets)}")

        scan_results, chosen, reason, is_moe = choose_scan_strategy(
            tag,
            ctx_targets,
            fit_target,
            max_ctx,
            model_is_moe,
            forced_ubatch=ubatch,
        )
        did_scan = True

    if scan_results:
        print_scan_table(scan_results, chosen)

    log(f"select | done | reason={reason}")
    if chosen:
        log(
            f"select | chosen | ctx={format_ctx(chosen['ctx'])} | target={format_ctx(chosen['target_ctx'])} | ngl={format_ngl(chosen['ngl'])} | ub={chosen.get('ubatch', '?')}"
        )

    bench_result = None

    if args.scan or not chosen:
        if chosen and did_scan:
            write_scan_cache_entry(
                cache,
                tag,
                vision_mode,
                fit_target,
                chosen,
                ubatch,
                mmproj_mib,
                is_moe,
                caps=caps,
            )
            log(f"cache | write | file={SCAN_CACHE_FILE}")
        return chosen, is_moe, bench_result, did_scan

    if args.rebench_cutoff is not None and bench_is_fresh(tag, mode, ubatch, args.rebench_cutoff):
        log(f"bench | skip | reason=fresh_result | rebench={args.rebench}")
        if did_scan:
            if caps is None:
                caps = detect_capabilities(tag)
            write_scan_cache_entry(
                cache,
                tag,
                vision_mode,
                fit_target,
                chosen,
                ubatch,
                mmproj_mib,
                is_moe,
                caps=caps,
            )
            log(f"cache | write | file={SCAN_CACHE_FILE}")
        return chosen, is_moe, bench_result, did_scan

    chosen_ngl = fit_ngl(chosen)
    cmd = build_bench_command(
        tag, chosen_ngl, chosen["ot"], ubatch, reps=args.reps, prio=args.prio
    )
    log(
        f"bench | start | ctx={format_ctx(chosen['ctx'])} | ngl={format_ngl(chosen['ngl'])} | "
        f"ub={ubatch} | offload={format_offload(chosen.get('offload'))} | reps={args.reps}"
    )
    if args.print_commands:
        print(shlex.join(cmd), flush=True)
        return chosen, is_moe, None, did_scan

    bench_start = time.monotonic()
    bench_result = run_bench(
        tag,
        chosen_ngl,
        chosen["ot"],
        ubatch,
        reps=args.reps,
        prio=args.prio,
    )
    if bench_result is not None:
        pp_f = format_bench_metric(bench_result["pp_speed"], missing="?")
        pp_stddev_f = format_bench_metric(bench_result["pp_stddev"], missing="?")
        tg_f = format_bench_metric(bench_result["tg_speed"], missing="?")
        tg_stddev_f = format_bench_metric(bench_result["tg_stddev"], missing="?")
        log(
            f"bench | ok | elapsed={time.monotonic() - bench_start:.1f}s | pp{BENCH_PP}={pp_f}±{pp_stddev_f} | tg{BENCH_TG}={tg_f}±{tg_stddev_f}"
        )

    if caps is None:
        caps = detect_capabilities(tag)
    if did_scan:
        caps = write_scan_cache_entry(
            cache,
            tag,
            vision_mode,
            fit_target,
            chosen,
            ubatch,
            mmproj_mib,
            is_moe,
            caps=caps,
        )
        reasoning = _reasoning_details(caps)
        reasoning_str = "true" if reasoning is not None else "false"
        switchable_str = "true" if reasoning is not None and reasoning.get("switchable", False) else ""
        effort_str = (reasoning.get("efforts") or "") if reasoning is not None else ""
        log(
            f"metadata | caps | vision={caps['vision']} | reasoning={reasoning_str} | switchable={switchable_str} | efforts={effort_str}"
        )

    if bench_result is not None:
        write_result_row(
            tag,
            chosen,
            is_moe,
            bench_result,
            caps,
            mode=mode,
            fit_target=fit_target,
            ubatch=ubatch,
            reps=args.reps,
        )

    return chosen, is_moe, bench_result, did_scan


def benchmark_tag(
    tag: str, args: Args, cache: ScanCache, index: int | None = None, total: int | None = None
) -> None:
    start_time = time.monotonic()
    identity = identity_from_tag(tag, require_quant=False)
    refresh_metadata = args.rescan_cutoff is not None

    mmproj_mib = get_mmproj_size_mib(tag)
    mode = "vision" if args.vision else "text"

    if args.vision:
        fit_target = FIT_TARGET + mmproj_mib
    else:
        fit_target = FIT_TARGET

    log_model_header(tag, mode, fit_target, mmproj_mib, index, total)
    if args.vision and mmproj_mib == 0:
        log("model | skip | reason=non_vision_model | mode=vision")
        return
    if not args.vision and mmproj_mib > 0:
        log("model | meta | vision_capable=true | running=text")

    caps = detect_capabilities(tag) if refresh_metadata else (get_capabilities(cache, tag) or detect_capabilities(tag))
    model_is_moe = is_moe_model(tag) if refresh_metadata else get_model_moe(cache, tag)
    if model_is_moe is None:
        model_is_moe = is_moe_model(tag)
        set_model_moe(cache, tag, model_is_moe)
        save_scan_cache(cache)

    scan_max_ctx: int | None = None
    scan_ctx_targets: Sequence[int] | None = None
    scan_contexts_resolved = False

    def get_scan_contexts() -> tuple[int | None, Sequence[int]]:
        nonlocal scan_max_ctx, scan_ctx_targets, scan_contexts_resolved
        if not scan_contexts_resolved:
            scan_max_ctx = get_cached_or_resolve_max_ctx(tag, args, cache)
            scan_ctx_targets = build_ctx_list(scan_max_ctx)
            scan_contexts_resolved = True
        assert scan_ctx_targets is not None
        return scan_max_ctx, scan_ctx_targets

    if args.ubatch is not None:
        ubatch_sizes = [args.ubatch]
        is_moe = None
    else:
        ubatch_sizes = list(MOE_UBATCH_SIZES)
        is_moe = None

    last_chosen = None
    last_is_moe = False
    last_bench_result = None

    for i, ubatch in enumerate(ubatch_sizes):
        log_ubatch_header(ubatch)
        chosen, is_moe, bench_result, _ = scan_and_bench_ubatch(
            tag, args, cache, fit_target, ubatch, mode, mmproj_mib,
            model_is_moe, scan_contexts=get_scan_contexts, caps=caps,
        )
        if chosen is not None:
            last_chosen = chosen
            last_is_moe = is_moe
            last_bench_result = bench_result
        if i == 0 and not args.ubatch and not is_moe:
            log("model | skip | dense=true | reason=remaining_ubatches_not_needed")
            break

    size_gib = ""
    if last_bench_result and last_bench_result.get("size_bytes"):
        size_gib = f"{int(last_bench_result['size_bytes']) / 1024**3:.2f}"
    print_summary(
        identity.display_name,
        identity.quant,
        identity.provider,
        size_gib,
        last_chosen,
        last_is_moe,
        last_bench_result,
    )

    log_model_footer(time.monotonic() - start_time)


def main() -> int:
    parser = argparse.ArgumentParser(description="Pick and benchmark the best context size")
    parser.add_argument("tags", nargs="*", help="HF repo:quant tags")
    parser.add_argument("-r", "--reps", type=int, default=REPS, help="Repetitions per test")
    parser.add_argument(
        "--scan", action="store_true", help="Only scan fit params, don't benchmark"
    )
    parser.add_argument(
        "--rescan",
        type=str,
        default=None,
        metavar="AGE",
        help="Re-scan if scan_ts is older than AGE (e.g. 24h, 7d, 30m). Default: use cache.",
    )
    parser.add_argument(
        "--rebench",
        type=str,
        default=None,
        metavar="AGE",
        help="Re-bench if bench_ts is older than AGE (e.g. 24h, 7d, 30m). Default: always bench.",
    )
    parser.add_argument(
        "--vision", action="store_true", help="Benchmark with mmproj VRAM budget (vision mode)"
    )
    parser.add_argument("-p", "--provider", action="append", help="Only benchmark models from this provider (e.g. unsloth)")
    parser.add_argument("-g", "--group", action="append", help="Only benchmark models in this group (e.g. qwen3.6-35b-a3b)")
    parser.add_argument(
        "-ub",
        "--ubatch",
        type=int,
        default=None,
        choices=VALID_UBATCH_SIZES,
        help=f"Force ubatch value (one of: {', '.join(str(u) for u in VALID_UBATCH_SIZES)})",
    )
    parser.add_argument(
        "--prio",
        type=int,
        choices=[-1, 0, 1, 2, 3],
        default=None,
        help="Pass llama-bench process priority through (--prio -1|0|1|2|3)",
    )
    parser.add_argument(
        "--print-commands", action="store_true", help="Print llama-bench commands and exit without benchmarking"
    )
    parser.add_argument("--log-file", help="Write timestamped progress logs to this file")
    args = cast(Args, parser.parse_args())
    if not args.print_commands:
        if not acquire_run_lock():
            print("fit_bench.py is already running; refusing parallel execution", file=sys.stderr)
            return 1
    set_log_file(args.log_file)

    if args.scan and args.rebench is not None:
        parser.error("--scan and --rebench are incompatible (--scan doesn't benchmark)")
    if args.print_commands and args.scan:
        parser.error("--print-commands and --scan are incompatible")
    if args.print_commands and args.rebench is not None:
        parser.error("--print-commands and --rebench are incompatible")

    if args.scan and args.reps != REPS:
        log("warning | ignored_arg | arg=--reps | reason=scan_mode")
    if args.scan and args.prio is not None:
        log("warning | ignored_arg | arg=--prio | reason=scan_mode")

    rescan_cutoff = None
    if args.rescan:
        rescan_cutoff = _parse_resume_age(args.rescan)
        if rescan_cutoff is None:
            parser.error(f"invalid --rescan age: {args.rescan!r} (use e.g. 24h, 7d, 30m)")

    rebench_cutoff = None
    if args.rebench is not None:
        rebench_cutoff = _parse_resume_age(args.rebench)
        if rebench_cutoff is None:
            parser.error(f"invalid --rebench age: {args.rebench!r} (use e.g. 24h, 7d, 30m)")

    args.rescan_cutoff = rescan_cutoff
    args.rebench_cutoff = rebench_cutoff

    if args.tags:
        tags = args.tags
    else:
        models = load_models()
        if args.provider:
            models = [m for m in models if m[0].split("/")[0] in args.provider]
        if args.group:
            models = [m for m in models if m[2] in args.group]
        tags = [render_model_tag(repo, quant) for repo, quant, _, _ in models]

    if not tags:
        parser.error("no models matched — provide tags or use --provider/--group filters")

    cache = load_scan_cache()

    mode = "vision" if args.vision else "text"

    total = len(tags)
    for i, tag in enumerate(tags, start=1):
        if rebench_cutoff is not None and not args.scan:
            if should_skip_rebench_model(tag, args, cache, mode):
                log_model_header(tag, mode, None, None, i if total > 1 else None, total if total > 1 else None)
                log(f"model | skip | reason=all_required_ubatches_fresh | rebench={args.rebench}")
                continue
        try:
            benchmark_tag(tag, args, cache, i if total > 1 else None, total if total > 1 else None)
        except Exception as exc:
            log(f"model | error | tag={tag} | error={exc}")
        if total > 1 and i != total:
            print(flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
