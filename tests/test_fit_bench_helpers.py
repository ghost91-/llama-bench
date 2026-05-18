# pyright: reportPrivateUsage=false
import subprocess
from datetime import datetime, timedelta, timezone
from typing import Mapping

import pytest
from pytest import MonkeyPatch

import fit_bench
from llama_bench.schema_types import Capabilities, ScanCache


def make_fit_result(
    target_ctx: int,
    ctx: int | None,
    ngl: int | None,
    ubatch: int = 512,
    ot: str | None = None,
) -> fit_bench.FitResult:
    return {
        "target_ctx": target_ctx,
        "ctx": ctx,
        "ngl": ngl,
        "ubatch": ubatch,
        "offload": fit_bench.count_offload(ot),
        "ot": ot,
    }


def make_args(scan: bool = False, ubatch: int | None = None) -> fit_bench.Args:
    args = fit_bench.Args()
    args.tags = []
    args.reps = 3
    args.scan = scan
    args.rescan = None
    args.rebench = None
    args.vision = False
    args.provider = None
    args.group = None
    args.ubatch = ubatch
    args.prio = 1
    args.log_file = None
    args.print_commands = False
    args.rescan_cutoff = None
    args.rebench_cutoff = None
    return args


def noop_log(_message: str = "") -> None:
    pass


def noop_print_scan_table(
    _scan_results: list[fit_bench.FitResult], _chosen: fit_bench.FitResult | None
) -> None:
    pass


def test_should_skip_rebench_model_requires_all_cached_moe_ubatches(
    monkeypatch: MonkeyPatch,
) -> None:
    args = make_args()
    cutoff = datetime(2026, 1, 2, tzinfo=timezone.utc)
    args.rebench_cutoff = cutoff
    cache: ScanCache = {"repo/model:Q4_K_M": {"moe": True}}
    fresh = cutoff + timedelta(hours=1)
    timestamps = {512: fresh, 2048: fresh, 4096: fresh}

    def fake_get_bench_ts(_tag: str, mode: str = "text", ubatch: int | None = None) -> datetime | None:
        assert mode == "text"
        assert ubatch is not None
        return timestamps.get(ubatch)

    monkeypatch.setattr(fit_bench, "get_bench_ts", fake_get_bench_ts)

    assert fit_bench.should_skip_rebench_model("repo/model:Q4_K_M", args, cache, "text") is False


def test_should_skip_rebench_model_skips_when_all_cached_moe_ubatches_are_fresh(
    monkeypatch: MonkeyPatch,
) -> None:
    args = make_args()
    cutoff = datetime(2026, 1, 2, tzinfo=timezone.utc)
    args.rebench_cutoff = cutoff
    cache: ScanCache = {"repo/model:Q4_K_M": {"moe": True}}
    fresh = cutoff + timedelta(hours=1)

    def fake_get_bench_ts(_tag: str, mode: str = "text", ubatch: int | None = None) -> datetime | None:
        assert mode == "text"
        assert ubatch is not None
        return fresh

    monkeypatch.setattr(fit_bench, "get_bench_ts", fake_get_bench_ts)

    assert fit_bench.should_skip_rebench_model("repo/model:Q4_K_M", args, cache, "text") is True


def test_should_skip_rebench_model_checks_only_dense_ubatch(monkeypatch: MonkeyPatch) -> None:
    args = make_args()
    cutoff = datetime(2026, 1, 2, tzinfo=timezone.utc)
    args.rebench_cutoff = cutoff
    cache: ScanCache = {"repo/model:Q4_K_M": {"moe": False}}
    checked: list[int] = []

    def fake_get_bench_ts(_tag: str, mode: str = "text", ubatch: int | None = None) -> datetime | None:
        assert mode == "text"
        assert ubatch is not None
        checked.append(ubatch)
        return cutoff + timedelta(hours=1)

    monkeypatch.setattr(fit_bench, "get_bench_ts", fake_get_bench_ts)

    assert fit_bench.should_skip_rebench_model("repo/model:Q4_K_M", args, cache, "text") is True
    assert checked == [512]


def test_should_skip_rebench_model_does_not_skip_when_moe_status_unknown(
    monkeypatch: MonkeyPatch,
) -> None:
    args = make_args()
    args.rebench_cutoff = datetime(2026, 1, 2, tzinfo=timezone.utc)

    def fail_get_bench_ts(_tag: str, mode: str = "text", ubatch: int | None = None) -> datetime | None:
        del mode, ubatch
        raise AssertionError("timestamps should not be checked without known expected ubatches")

    monkeypatch.setattr(fit_bench, "get_bench_ts", fail_get_bench_ts)

    assert fit_bench.should_skip_rebench_model("repo/model:Q4_K_M", args, {}, "text") is False


def test_system_summary_line_uses_detected_hardware(monkeypatch: MonkeyPatch) -> None:
    fit_bench.system_summary_line.cache_clear()

    def fake_sysconf(name: str) -> int:
        values = {
            "SC_PHYS_PAGES": 16 * 1024**3 // 4096,
            "SC_PAGE_SIZE": 4096,
        }
        return values[name]

    monkeypatch.setattr(fit_bench, "nvidia_proc_gpu_summary", lambda: "NVIDIA RTX Test")
    monkeypatch.setattr(fit_bench.os, "sysconf", fake_sysconf)

    assert fit_bench.system_summary_line() == "NVIDIA RTX Test, 16 GiB RAM, -fa on"
    fit_bench.system_summary_line.cache_clear()


def test_system_summary_line_handles_missing_hardware(monkeypatch: MonkeyPatch) -> None:
    fit_bench.system_summary_line.cache_clear()

    def fake_sysconf(_name: str) -> int:
        raise ValueError("missing")

    monkeypatch.setattr(fit_bench, "nvidia_proc_gpu_summary", lambda: None)
    monkeypatch.setattr(fit_bench.os, "sysconf", fake_sysconf)

    assert fit_bench.system_summary_line() == "-fa on"
    fit_bench.system_summary_line.cache_clear()


def test_build_ctx_list_includes_max_and_extended_steps() -> None:
    assert fit_bench.build_ctx_list(None)[:4] == [5000, 10000, 20000, 30000]
    assert fit_bench.build_ctx_list(360000) == [
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
        250000,
        300000,
        360000,
    ]


def test_get_max_ctx_uses_gguf_before_subprocess_fallbacks(monkeypatch: MonkeyPatch) -> None:
    subprocess_calls = 0

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal subprocess_calls
        subprocess_calls += 1
        return subprocess.CompletedProcess([], 0, stdout="", stderr="")

    def fake_get_max_ctx_from_gguf(_tag: str) -> int:
        return 32768

    monkeypatch.setattr(fit_bench, "get_max_ctx_from_gguf", fake_get_max_ctx_from_gguf)
    monkeypatch.setattr(fit_bench.subprocess, "run", fake_run)
    monkeypatch.setattr(fit_bench, "log", noop_log)

    assert fit_bench.get_max_ctx("repo/model:Q4_K_M") == 32768
    assert subprocess_calls == 0


def test_get_max_ctx_falls_back_from_fit_params_to_llama_bench(
    monkeypatch: MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def fake_run(
        cmd: list[str], capture_output: bool, text: bool, timeout: int
    ) -> subprocess.CompletedProcess[str]:
        del capture_output, text, timeout
        commands.append(cmd)
        if cmd[0] == "llama-fit-params":
            return subprocess.CompletedProcess(cmd, 0, stdout="no context", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="n_ctx_train = 65536", stderr="")

    def fake_get_max_ctx_from_gguf(_tag: str) -> None:
        return None

    monkeypatch.setattr(fit_bench, "get_max_ctx_from_gguf", fake_get_max_ctx_from_gguf)
    monkeypatch.setattr(fit_bench.subprocess, "run", fake_run)
    monkeypatch.setattr(fit_bench, "log", noop_log)

    assert fit_bench.get_max_ctx("repo/model:Q4_K_M", fit_target=256, prio=3) == 65536
    assert commands[0][:5] == ["llama-fit-params", "-hf", "repo/model:Q4_K_M", "-c", "1"]
    assert commands[0][commands[0].index("--fit-target") + 1] == "256"
    assert commands[1][0] == "llama-bench"
    assert commands[1][-2:] == ["--prio", "3"]


def test_get_max_ctx_returns_none_after_timeouts(monkeypatch: MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_run(
        cmd: list[str], capture_output: bool, text: bool, timeout: int
    ) -> subprocess.CompletedProcess[str]:
        del capture_output, text
        calls.append(cmd[0])
        raise subprocess.TimeoutExpired(cmd[0], timeout)

    def fake_get_max_ctx_from_gguf(_tag: str) -> None:
        return None

    monkeypatch.setattr(fit_bench, "get_max_ctx_from_gguf", fake_get_max_ctx_from_gguf)
    monkeypatch.setattr(fit_bench.subprocess, "run", fake_run)
    monkeypatch.setattr(fit_bench, "log", noop_log)

    assert fit_bench.get_max_ctx("repo/model:Q4_K_M") is None
    assert calls == ["llama-fit-params", "llama-bench"]


def test_get_cached_or_resolve_max_ctx_uses_and_writes_scan_cache(
    monkeypatch: MonkeyPatch,
) -> None:
    args = make_args()
    cache: ScanCache = {
        "repo/cached:Q4_K_M": {
            "max_ctx": 32768,
            "max_ctx_ts": "2026-01-01T00:00:00+0000",
        }
    }
    calls: list[str] = []
    saves = 0

    def fake_get_max_ctx(tag: str, fit_target: int = fit_bench.FIT_TARGET, prio: int | None = None) -> int:
        del fit_target, prio
        calls.append(tag)
        return 65536

    def fake_save_scan_cache(_cache: ScanCache) -> None:
        nonlocal saves
        saves += 1

    monkeypatch.setattr(fit_bench, "get_max_ctx", fake_get_max_ctx)
    monkeypatch.setattr(fit_bench, "save_scan_cache", fake_save_scan_cache)
    monkeypatch.setattr(fit_bench, "log", noop_log)

    assert fit_bench.get_cached_or_resolve_max_ctx("repo/cached:Q4_K_M", args, cache) == 32768
    assert fit_bench.get_cached_or_resolve_max_ctx("repo/new:Q4_K_M", args, cache) == 65536
    assert calls == ["repo/new:Q4_K_M"]
    assert saves == 1
    assert cache["repo/new:Q4_K_M"].get("max_ctx") == 65536


def test_parse_fit_params_and_build_fit_result() -> None:
    params = '-c 8192 -ngl -1 -ot "0,1,2"'

    assert fit_bench.parse_fit_params(params) == (8192, -1, "0,1,2")
    assert fit_bench.build_fit_result(10000, 512, params) == {
        "target_ctx": 10000,
        "ctx": 8192,
        "ngl": -1,
        "ubatch": 512,
        "offload": 3,
        "ot": "0,1,2",
    }
    assert fit_bench.build_fit_result(10000, 512, None) == {
        "target_ctx": 10000,
        "ctx": None,
        "ngl": None,
        "ubatch": 512,
        "offload": None,
        "ot": None,
    }


def test_build_refinement_ctx_list_fills_gap_before_drop() -> None:
    results: list[fit_bench.FitResult] = [
        {"target_ctx": 5000, "ctx": 5000, "ngl": -1, "ubatch": 512, "offload": None, "ot": None},
        {
            "target_ctx": 50000,
            "ctx": 50000,
            "ngl": -1,
            "ubatch": 512,
            "offload": None,
            "ot": None,
        },
        {
            "target_ctx": 100000,
            "ctx": 100000,
            "ngl": 72,
            "ubatch": 512,
            "offload": None,
            "ot": None,
        },
    ]

    assert fit_bench.build_refinement_ctx_list(results, results[1], 120000, -1) == [75000]


def test_strategy_helpers_choose_expected_results() -> None:
    probe_dense: fit_bench.FitResult = {
        "target_ctx": 5000,
        "ctx": 5000,
        "ngl": -1,
        "ubatch": 512,
        "offload": None,
        "ot": None,
    }
    probe_moe: fit_bench.FitResult = {
        "target_ctx": 5000,
        "ctx": 5000,
        "ngl": 84,
        "ubatch": 1024,
        "offload": None,
        "ot": None,
    }
    dense_results: list[fit_bench.FitResult] = [
        probe_dense,
        {
            "target_ctx": 10000,
            "ctx": 10000,
            "ngl": -1,
            "ubatch": 512,
            "offload": None,
            "ot": None,
        },
        {
            "target_ctx": 20000,
            "ctx": 20000,
            "ngl": 80,
            "ubatch": 512,
            "offload": None,
            "ot": None,
        },
    ]

    assert fit_bench.resolve_probe_strategy(probe_dense, probe_moe, is_moe=False) == (
        probe_dense,
        -1,
        False,
    )
    assert fit_bench.resolve_probe_strategy(probe_dense, probe_moe, is_moe=True) == (
        probe_moe,
        84,
        True,
    )
    assert fit_bench.choose_target_result(dense_results, -1, descending=False) == dense_results[1]
    assert fit_bench.select_best_result(dense_results, is_moe=False) == (
        dense_results[1],
        "Dense: highest context that still fits fully in VRAM",
    )


def test_select_best_result_handles_moe_non_monotonic_and_dense_fallback() -> None:
    results = [
        make_fit_result(5000, 5000, 84),
        make_fit_result(10000, 10000, 80),
        make_fit_result(20000, 20000, 84),
        make_fit_result(30000, 30000, 82),
    ]

    assert fit_bench.select_best_result(results, is_moe=True) == (
        results[2],
        "MoE: highest context that keeps max ngl (84)",
    )
    assert fit_bench.select_best_result(results, is_moe=False) == (
        results[2],
        "Dense fallback: no full-VRAM fit found, using highest context at max ngl (84)",
    )


def test_choose_target_result_descending_and_ascending_early_stop_edges() -> None:
    results = [
        make_fit_result(5000, 5000, 84),
        make_fit_result(10000, 10000, 80),
        make_fit_result(20000, 20000, 84),
    ]

    assert fit_bench.choose_target_result(results, 84, descending=True) == results[2]
    assert fit_bench.choose_target_result(results, 84, descending=False) == results[0]


def test_resume_age_parser_accepts_supported_units() -> None:
    before = datetime.now(timezone.utc)
    cutoff = fit_bench._parse_resume_age("30m")
    after = datetime.now(timezone.utc)

    assert cutoff is not None
    assert before.timestamp() - 30 * 60 <= cutoff.timestamp() <= after.timestamp()
    assert fit_bench._parse_resume_age("2h") is not None
    assert fit_bench._parse_resume_age("7d") is not None
    assert fit_bench._parse_resume_age("1w") is None
    assert fit_bench._parse_resume_age("bad") is None


def test_fit_result_accessors_raise_for_missing_values() -> None:
    missing: fit_bench.FitResult = {
        "target_ctx": 5000,
        "ctx": None,
        "ngl": None,
        "ubatch": 512,
        "offload": None,
        "ot": None,
    }

    try:
        fit_bench.fit_ctx(missing)
    except ValueError as exc:
        assert "ctx" in str(exc)
    else:
        raise AssertionError("fit_ctx should reject missing ctx")

    try:
        fit_bench.fit_ngl(missing)
    except ValueError as exc:
        assert "ngl" in str(exc)
    else:
        raise AssertionError("fit_ngl should reject missing ngl")


def test_load_existing_fit_choice_validates_fit_target_and_ot_consistency() -> None:
    cache: ScanCache = {
        "repo/model:Q4_K_M": {
            "text": {
                "ubatch_sizes": {
                    "512": {
                        "fit_target": fit_bench.FIT_TARGET,
                        "ctx": 8192,
                        "ngl": -1,
                        "offload": 3,
                        "ot": "0,1,2",
                        "scan_ts": "2026-01-01T00:00:00+00:00",
                    },
                    "1024": {
                        "fit_target": fit_bench.FIT_TARGET + 1,
                        "ctx": 4096,
                        "ngl": 72,
                        "offload": 2,
                        "ot": None,
                        "scan_ts": "2026-01-01T00:00:00+00:00",
                    },
                    "2048": {
                        "fit_target": fit_bench.FIT_TARGET,
                        "ctx": 8192,
                        "ngl": -1,
                        "offload": None,
                        "ot": "0,1,2",
                        "scan_ts": "2026-01-01T00:00:00+00:00",
                    },
                }
            }
        }
    }

    assert fit_bench.load_existing_fit_choice(
        "repo/model:Q4_K_M", fit_bench.FIT_TARGET, False, 512, cache=cache
    ) == {
        "target_ctx": 8192,
        "ctx": 8192,
        "ngl": -1,
        "ubatch": 512,
        "offload": 3,
        "ot": "0,1,2",
    }
    assert (
        fit_bench.load_existing_fit_choice(
            "repo/model:Q4_K_M", fit_bench.FIT_TARGET, False, 1024, cache=cache
        )
        is None
    )
    assert (
        fit_bench.load_existing_fit_choice(
            "repo/model:Q4_K_M", fit_bench.FIT_TARGET, False, 2048, cache=cache
        )
        is None
    )
    assert fit_bench.load_existing_fit_choice("repo/model:Q4_K_M", 999, False, 512, cache=cache) is None


def test_merge_prefer_and_format_helpers() -> None:
    low: fit_bench.FitResult = {
        "target_ctx": 5000,
        "ctx": 5000,
        "ngl": 40,
        "ubatch": 512,
        "offload": None,
        "ot": None,
    }
    high: fit_bench.FitResult = {
        "target_ctx": 10000,
        "ctx": 10000,
        "ngl": -1,
        "ubatch": 1024,
        "offload": 2,
        "ot": "0,1",
    }
    replacement: fit_bench.FitResult = {
        "target_ctx": 5000,
        "ctx": 6000,
        "ngl": 40,
        "ubatch": 512,
        "offload": None,
        "ot": None,
    }

    assert fit_bench.merge_scan_results([low, high], [replacement]) == [replacement, high]
    assert fit_bench.ngl_rank(None) == -1
    assert fit_bench.ngl_rank(-1) > fit_bench.ngl_rank(999)
    assert fit_bench.prefer_result(low, high) is high
    assert fit_bench.prefer_result(None, low) is low
    assert fit_bench.ot_to_bench_arg("0,1") == ["-ot", "0;1"]
    assert fit_bench.ot_to_bench_arg(None) == []
    assert fit_bench.count_offload(None) is None
    assert fit_bench.format_offload(None) == ""
    assert fit_bench.format_offload(3) == "3"


def test_parse_bench_row_extracts_matching_metric_and_metadata() -> None:
    output = (
        'model_type,model_size,model_n_params,n_prompt,n_gen,n_depth,avg_ts,stddev_ts\n'
        '"foo","1073741824","7000000000","4096","0","0","12.5","0.3"\n'
        '"foo","1073741824","7000000000","0","128","0","25.0","0.8"\n'
    )

    assert fit_bench.parse_bench_row(output, 0, 128, 0) == {
        "size_bytes": "1073741824",
        "n_params": "7000000000",
        "speed": "25.0",
        "stddev": "0.8",
    }
    assert fit_bench.parse_bench_row(output, 1, 1, 0) is None


def test_build_bench_command_uses_full_gpu_ngl_prio_ot_and_larger_ubatch() -> None:
    assert fit_bench.build_bench_command(
        "repo/model:Q4_K_M", -1, "0,1", 8192, reps=7, prio=2
    ) == [
        "llama-bench",
        "-hf",
        "repo/model:Q4_K_M",
        "-fa",
        "1",
        "-ngl",
        "99",
        "-b",
        "8192",
        "-ub",
        "8192",
        "-o",
        "csv",
        "-r",
        "7",
        "--prio",
        "2",
        "-ot",
        "0;1",
        "-p",
        "4096",
        "-n",
        "128",
        "-d",
        "0",
    ]


def test_build_bench_command_uses_minimum_batch_and_omits_optional_args() -> None:
    cmd = fit_bench.build_bench_command("repo/model:Q4_K_M", 72, None, 512, reps=20, prio=None)

    assert cmd[cmd.index("-b") + 1] == "4096"
    assert cmd[cmd.index("-ub") + 1] == "512"
    assert cmd[cmd.index("-ngl") + 1] == "72"
    assert "--prio" not in cmd
    assert "-ot" not in cmd


def test_run_bench_builds_command_and_parses_metrics(monkeypatch: MonkeyPatch) -> None:
    calls: list[list[str]] = []
    output = (
        "model_type,model_size,model_n_params,n_prompt,n_gen,n_depth,avg_ts,stddev_ts\n"
        '"foo","1073741824","7000000000","4096","0","0","12.5","0.3"\n'
        '"foo","1073741824","7000000000","0","128","0","25.0","0.8"\n'
    )

    def fake_run(
        cmd: list[str],
        capture_output: bool,
        text: bool,
        timeout: int,
    ) -> subprocess.CompletedProcess[str]:
        del capture_output, text
        calls.append(cmd)
        assert timeout == fit_bench.BENCH_TIMEOUT
        return subprocess.CompletedProcess(cmd, 0, stdout=output, stderr="")

    monkeypatch.setattr(fit_bench.subprocess, "run", fake_run)

    assert fit_bench.run_bench(
        "repo/model:Q4_K_M", -1, "0,1", 8192, reps=7, prio=2
    ) == {
        "size_bytes": "1073741824",
        "n_params": "7000000000",
        "pp_speed": "12.5",
        "pp_stddev": "0.3",
        "tg_speed": "25.0",
        "tg_stddev": "0.8",
    }
    assert calls == [
        [
            "llama-bench",
            "-hf",
            "repo/model:Q4_K_M",
            "-fa",
            "1",
            "-ngl",
            "99",
            "-b",
            "8192",
            "-ub",
            "8192",
            "-o",
            "csv",
            "-r",
            "7",
            "--prio",
            "2",
            "-ot",
            "0;1",
            "-p",
            "4096",
            "-n",
            "128",
            "-d",
            "0",
        ]
    ]


def test_run_bench_uses_minimum_batch_for_small_ubatch(monkeypatch: MonkeyPatch) -> None:
    calls: list[list[str]] = []
    output = (
        "model_type,model_size,model_n_params,n_prompt,n_gen,n_depth,avg_ts,stddev_ts\n"
        '"foo","1","2","4096","0","0","1","0"\n'
        '"foo","1","2","0","128","0","2","0"\n'
    )

    def fake_run(
        cmd: list[str], capture_output: bool, text: bool, timeout: int
    ) -> subprocess.CompletedProcess[str]:
        del capture_output, text, timeout
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=output, stderr="")

    monkeypatch.setattr(fit_bench.subprocess, "run", fake_run)

    assert fit_bench.run_bench("repo/model:Q4_K_M", 72, None, 512) is not None
    assert calls[0][calls[0].index("-b") + 1] == "4096"
    assert calls[0][calls[0].index("-ngl") + 1] == "72"
    assert "--prio" not in calls[0]
    assert "-ot" not in calls[0]


def test_run_bench_failure_modes_return_none(monkeypatch: MonkeyPatch) -> None:
    log_lines: list[str] = []
    monkeypatch.setattr(fit_bench, "log", log_lines.append)

    def timeout_run(
        cmd: list[str], capture_output: bool, text: bool, timeout: int
    ) -> subprocess.CompletedProcess[str]:
        del cmd, capture_output, text
        raise subprocess.TimeoutExpired("llama-bench", timeout)

    monkeypatch.setattr(fit_bench.subprocess, "run", timeout_run)
    assert fit_bench.run_bench("repo/model:Q4_K_M", 72, None, 512) is None

    def nonzero_no_stdout_run(
        cmd: list[str], capture_output: bool, text: bool, timeout: int
    ) -> subprocess.CompletedProcess[str]:
        del capture_output, text, timeout
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="failed")

    monkeypatch.setattr(fit_bench.subprocess, "run", nonzero_no_stdout_run)
    assert fit_bench.run_bench("repo/model:Q4_K_M", 72, None, 512) is None
    assert "bench | fail | reason=process exited without stdout | return_code=1" in log_lines
    assert "bench | stderr | failed" in log_lines

    def missing_rows_run(
        cmd: list[str], capture_output: bool, text: bool, timeout: int
    ) -> subprocess.CompletedProcess[str]:
        del capture_output, text, timeout
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout="model_type,model_size,model_n_params,n_prompt,n_gen,n_depth,avg_ts,stddev_ts\n",
            stderr="",
        )

    monkeypatch.setattr(fit_bench.subprocess, "run", missing_rows_run)
    assert fit_bench.run_bench("repo/model:Q4_K_M", 72, None, 512) is None
    assert "bench | fail | reason=missing expected CSV result rows | return_code=0" in log_lines
    assert "bench | stdout | begin" in log_lines


def test_scan_fit_configs_reuses_existing_results_and_stops_on_match(
    monkeypatch: MonkeyPatch,
) -> None:
    calls: list[int] = []

    def fake_get_fit_params(
        _tag: str, target_ctx: int, fit_target: int | None = None, ubatch: int = 512
    ) -> str | None:
        calls.append(target_ctx)
        return f"-c {target_ctx} -ngl -1 -ot 0,1"

    def stop_on_full_vram(result: fit_bench.FitResult) -> bool:
        return result["ngl"] == -1

    monkeypatch.setattr(fit_bench, "get_fit_params", fake_get_fit_params)
    monkeypatch.setattr(fit_bench, "log", noop_log)
    existing: fit_bench.FitResult = {
        "target_ctx": 5000,
        "ctx": 5000,
        "ngl": 10,
        "ubatch": 512,
        "offload": None,
        "ot": None,
    }

    results = fit_bench.scan_fit_configs(
        "repo/model:Q4_K_M",
        [5000, 10000, 20000],
        existing_results=[existing],
        stop_on_match=stop_on_full_vram,
    )

    assert calls == [10000]
    assert results == [
        existing,
        {"target_ctx": 10000, "ctx": 10000, "ngl": -1, "ubatch": 512, "offload": 2, "ot": "0,1"},
    ]


def test_scan_fit_configs_records_failure_when_fit_params_returns_none(
    monkeypatch: MonkeyPatch,
) -> None:
    def fake_get_fit_params(
        _tag: str, target_ctx: int, fit_target: int | None = None, ubatch: int = 512
    ) -> None:
        del target_ctx, fit_target, ubatch
        return None

    monkeypatch.setattr(fit_bench, "get_fit_params", fake_get_fit_params)
    monkeypatch.setattr(fit_bench, "log", noop_log)

    assert fit_bench.scan_fit_configs("repo/model:Q4_K_M", [5000]) == [
        make_fit_result(5000, None, None)
    ]


def test_fallback_scan_strategy_scans_all_targets_for_moe_non_monotonic(
    monkeypatch: MonkeyPatch,
) -> None:
    scanned_targets: list[int] = []

    def fake_get_fit_params(
        _tag: str, target_ctx: int, fit_target: int | None = None, ubatch: int = 512
    ) -> str:
        del fit_target, ubatch
        scanned_targets.append(target_ctx)
        ngl_by_ctx = {30000: 82, 20000: 84, 10000: 80, 5000: 84}
        return f"-c {target_ctx} -ngl {ngl_by_ctx[target_ctx]}"

    monkeypatch.setattr(fit_bench, "get_fit_params", fake_get_fit_params)
    monkeypatch.setattr(fit_bench, "log", noop_log)

    results, chosen, reason, is_moe = fit_bench.fallback_scan_strategy(
        "repo/model:Q4_K_M", [5000, 10000, 20000, 30000], 128, 512, True
    )

    assert scanned_targets == [30000, 20000, 10000, 5000]
    assert chosen == make_fit_result(20000, 20000, 84)
    assert reason == "MoE: highest context that keeps max ngl (84)"
    assert is_moe is True
    assert [result["target_ctx"] for result in results] == scanned_targets


def test_run_target_scan_adds_refinement_results(monkeypatch: MonkeyPatch) -> None:
    calls: list[list[int]] = []

    def fake_scan_fit_configs(
        _tag: str,
        ctx_targets: list[int] | tuple[int, ...],
        fit_target: int | None = None,
        ubatch: int = 512,
        stop_on_match: fit_bench.StopOnMatch | None = None,
        existing_results: list[fit_bench.FitResult] | None = None,
    ) -> list[fit_bench.FitResult]:
        del fit_target, ubatch, stop_on_match, existing_results
        calls.append(list(ctx_targets))
        return [
            {
                "target_ctx": target,
                "ctx": target,
                "ngl": -1 if target <= 50000 else 72,
                "ubatch": 512,
                "offload": None,
                "ot": None,
            }
            for target in ctx_targets
        ]

    monkeypatch.setattr(fit_bench, "scan_fit_configs", fake_scan_fit_configs)
    monkeypatch.setattr(fit_bench, "log", noop_log)

    results, chosen = fit_bench.run_target_scan(
        "repo/model:Q4_K_M",
        [5000, 50000, 100000],
        fit_target=128,
        max_ctx=100000,
        ubatch=512,
        target_ngl=-1,
        descending=False,
    )

    assert calls == [[5000, 50000, 100000], [75000]]
    assert chosen is not None
    assert chosen["ctx"] == 50000
    assert [result["target_ctx"] for result in results] == [5000, 50000, 75000, 100000]


def test_choose_scan_strategy_uses_forced_ubatch_probe(monkeypatch: MonkeyPatch) -> None:
    probe: fit_bench.FitResult = {
        "target_ctx": 5000,
        "ctx": 5000,
        "ngl": 84,
        "ubatch": 2048,
        "offload": None,
        "ot": None,
    }
    chosen: fit_bench.FitResult = {
        "target_ctx": 20000,
        "ctx": 20000,
        "ngl": 84,
        "ubatch": 2048,
        "offload": None,
        "ot": None,
    }
    run_args: list[tuple[int, int, bool]] = []

    def fake_probe_fit_config(
        _tag: str, _target_ctx: int, _fit_target: int, _ubatch: int
    ) -> fit_bench.FitResult:
        return probe

    monkeypatch.setattr(fit_bench, "probe_fit_config", fake_probe_fit_config)

    def fake_run_target_scan(
        _tag: str,
        _ctx_targets: list[int],
        _fit_target: int,
        _max_ctx: int,
        ubatch: int,
        target_ngl: int,
        descending: bool,
        probe_result: fit_bench.FitResult | None = None,
    ) -> tuple[list[fit_bench.FitResult], fit_bench.FitResult | None]:
        del probe_result
        run_args.append((ubatch, target_ngl, descending))
        return [probe, chosen], chosen

    monkeypatch.setattr(fit_bench, "run_target_scan", fake_run_target_scan)

    assert fit_bench.choose_scan_strategy(
        "repo/model:Q4_K_M", [5000, 20000], 128, 20000, is_moe=True, forced_ubatch=2048
    ) == (
        [probe, chosen],
        chosen,
        "MoE: highest context that keeps max ngl (84)",
        True,
    )
    assert run_args == [(2048, 84, True)]


def test_choose_scan_strategy_unforced_uses_dense_full_vram_probe(
    monkeypatch: MonkeyPatch,
) -> None:
    probe_calls: list[int] = []
    run_args: list[tuple[int, int, bool, fit_bench.FitResult | None]] = []
    dense_probe = make_fit_result(5000, 5000, -1, ubatch=512)
    moe_probe = make_fit_result(5000, 5000, 84, ubatch=1024)
    chosen = make_fit_result(50000, 50000, -1, ubatch=512)

    def fake_probe_fit_config(
        _tag: str, _target_ctx: int, _fit_target: int, ubatch: int
    ) -> fit_bench.FitResult:
        probe_calls.append(ubatch)
        return dense_probe if ubatch == 512 else moe_probe

    def fake_run_target_scan(
        _tag: str,
        _ctx_targets: list[int] | tuple[int, ...],
        _fit_target: int,
        _max_ctx: int | None,
        ubatch: int,
        target_ngl: int,
        descending: bool,
        probe_result: fit_bench.FitResult | None = None,
    ) -> tuple[list[fit_bench.FitResult], fit_bench.FitResult | None]:
        del _ctx_targets
        run_args.append((ubatch, target_ngl, descending, probe_result))
        return [dense_probe, chosen], chosen

    monkeypatch.setattr(fit_bench, "probe_fit_config", fake_probe_fit_config)
    monkeypatch.setattr(fit_bench, "run_target_scan", fake_run_target_scan)
    monkeypatch.setattr(fit_bench, "log", noop_log)

    assert fit_bench.choose_scan_strategy(
        "repo/model:Q4_K_M", [5000, 50000], 128, 50000, is_moe=False
    ) == (
        [dense_probe, chosen],
        chosen,
        "Dense: highest context that still fits fully in VRAM",
        False,
    )
    assert probe_calls == [512, 1024]
    assert run_args == [(512, -1, True, dense_probe)]


def test_choose_scan_strategy_unforced_uses_moe_max_ngl_probe(
    monkeypatch: MonkeyPatch,
) -> None:
    dense_probe = make_fit_result(5000, 5000, 80, ubatch=512)
    moe_probe = make_fit_result(5000, 5000, 84, ubatch=1024)
    run_args: list[tuple[int, int, bool, fit_bench.FitResult | None]] = []

    def fake_probe_fit_config(
        _tag: str, _target_ctx: int, _fit_target: int, ubatch: int
    ) -> fit_bench.FitResult:
        return dense_probe if ubatch == 512 else moe_probe

    def fake_run_target_scan(
        _tag: str,
        _ctx_targets: list[int] | tuple[int, ...],
        _fit_target: int,
        _max_ctx: int | None,
        ubatch: int,
        target_ngl: int,
        descending: bool,
        probe_result: fit_bench.FitResult | None = None,
    ) -> tuple[list[fit_bench.FitResult], fit_bench.FitResult | None]:
        del _ctx_targets
        run_args.append((ubatch, target_ngl, descending, probe_result))
        return [moe_probe], moe_probe

    monkeypatch.setattr(fit_bench, "probe_fit_config", fake_probe_fit_config)
    monkeypatch.setattr(fit_bench, "run_target_scan", fake_run_target_scan)
    monkeypatch.setattr(fit_bench, "log", noop_log)

    assert fit_bench.choose_scan_strategy(
        "repo/model:Q4_K_M", [5000, 50000], 128, 50000, is_moe=True
    ) == (
        [moe_probe],
        moe_probe,
        "MoE: highest context that keeps max ngl (84)",
        True,
    )
    assert run_args == [(1024, 84, True, moe_probe)]


def test_choose_scan_strategy_probe_failure_falls_back_to_descending_dense_scan(
    monkeypatch: MonkeyPatch,
) -> None:
    scanned: list[tuple[list[int], int]] = []

    def fake_scan_fit_configs(
        _tag: str,
        ctx_targets: list[int] | tuple[int, ...],
        fit_target: int | None = None,
        ubatch: int = 512,
        stop_on_match: fit_bench.StopOnMatch | None = None,
        existing_results: list[fit_bench.FitResult] | None = None,
    ) -> list[fit_bench.FitResult]:
        del fit_target, stop_on_match, existing_results
        scanned.append((list(ctx_targets), ubatch))
        return [make_fit_result(10000, 10000, 72, ubatch), make_fit_result(5000, 5000, 72, ubatch)]

    def fake_probe_fit_config(
        _tag: str, _target_ctx: int, _fit_target: int, _ubatch: int
    ) -> None:
        return None

    monkeypatch.setattr(fit_bench, "probe_fit_config", fake_probe_fit_config)
    monkeypatch.setattr(fit_bench, "scan_fit_configs", fake_scan_fit_configs)
    monkeypatch.setattr(fit_bench, "log", noop_log)

    results, chosen, reason, is_moe = fit_bench.choose_scan_strategy(
        "repo/model:Q4_K_M", [5000, 10000], 128, 10000, is_moe=False
    )

    assert scanned == [([10000, 5000], 512)]
    assert results[0]["target_ctx"] == 10000
    assert chosen == results[0]
    assert reason == "Dense fallback: no full-VRAM fit found, using highest context at max ngl (72)"
    assert is_moe is False


def test_choose_scan_strategy_dense_probe_below_full_vram_uses_dense_fallback_reason(
    monkeypatch: MonkeyPatch,
) -> None:
    probe = make_fit_result(5000, 5000, 72, ubatch=512)
    chosen = make_fit_result(20000, 20000, 72, ubatch=512)

    def fake_probe_fit_config(
        _tag: str, _target_ctx: int, _fit_target: int, ubatch: int
    ) -> fit_bench.FitResult:
        return probe if ubatch == 512 else make_fit_result(5000, 5000, 70, ubatch=1024)

    def fake_run_target_scan(
        _tag: str,
        _ctx_targets: list[int] | tuple[int, ...],
        _fit_target: int,
        _max_ctx: int | None,
        ubatch: int,
        target_ngl: int,
        descending: bool,
        probe_result: fit_bench.FitResult | None = None,
    ) -> tuple[list[fit_bench.FitResult], fit_bench.FitResult | None]:
        del _ctx_targets, ubatch, target_ngl, descending, probe_result
        return [probe, chosen], chosen

    monkeypatch.setattr(fit_bench, "probe_fit_config", fake_probe_fit_config)
    monkeypatch.setattr(fit_bench, "run_target_scan", fake_run_target_scan)
    monkeypatch.setattr(fit_bench, "log", noop_log)

    assert fit_bench.choose_scan_strategy(
        "repo/model:Q4_K_M", [5000, 20000], 128, 20000, is_moe=False
    ) == (
        [probe, chosen],
        chosen,
        "Dense fallback: 5k probe does not fit fully in VRAM, using highest context at max ngl (72)",
        False,
    )


def test_write_result_row_formats_benchmark(monkeypatch: MonkeyPatch) -> None:
    rows: list[dict[str, str]] = []

    def fake_append_result_row(row: Mapping[str, str | None]) -> None:
        rows.append({key: "" if value is None else value for key, value in row.items()})

    monkeypatch.setattr(fit_bench, "append_result_row", fake_append_result_row)
    monkeypatch.setattr(fit_bench, "sort_results_file", lambda: None)
    monkeypatch.setattr(fit_bench, "log", noop_log)
    chosen: fit_bench.FitResult = {
        "target_ctx": 8192,
        "ctx": 8192,
        "ngl": -1,
        "ubatch": 512,
        "offload": 3,
        "ot": "0,1,2",
    }
    bench: fit_bench.BenchResult = {
        "size_bytes": str(2 * 1024**3),
        "n_params": "7000000000",
        "pp_speed": "12.34",
        "pp_stddev": "0.56",
        "tg_speed": "24.68",
        "tg_stddev": "1.23",
    }
    caps: Capabilities = {"vision": True, "reasoning": {"switchable": True, "efforts": "low|high"}}

    fit_bench.write_result_row(
        "unsloth/Foo-GGUF:Q4_K_M",
        chosen,
        True,
        bench,
        caps,
        mode="vision",
        fit_target=192,
        ubatch=512,
        reps=20,
    )

    assert rows[0]["model"] == "Foo"
    assert rows[0]["size_gib"] == "2.00"
    assert rows[0]["params"] == "7B"
    assert rows[0]["moe"] == "true"
    assert rows[0]["ngl"] == "all"
    assert rows[0][fit_bench.PP_COL] == "12.3"
    assert rows[0][fit_bench.TG_COL] == "24.7"
    assert rows[0]["fit_target"] == "192"
    assert rows[0]["bench_ts"]


def test_write_result_row_rejects_missing_chosen_ctx(monkeypatch: MonkeyPatch) -> None:
    def fake_append_result_row(_row: Mapping[str, str | None]) -> None:
        pass

    monkeypatch.setattr(fit_bench, "append_result_row", fake_append_result_row)
    monkeypatch.setattr(fit_bench, "sort_results_file", lambda: None)

    with pytest.raises(ValueError, match="ctx"):
        fit_bench.write_result_row(
            "unsloth/Foo-GGUF:Q4_K_M",
            make_fit_result(8192, None, -1),
            False,
            None,
            {"vision": False, "reasoning": False},
            mode="text",
            fit_target=128,
            ubatch=512,
            reps=20,
        )


def test_write_scan_cache_entry_detects_caps_and_saves(monkeypatch: MonkeyPatch) -> None:
    saved: list[ScanCache] = []
    cache: ScanCache = {}
    chosen: fit_bench.FitResult = {
        "target_ctx": 4096,
        "ctx": 4096,
        "ngl": 72,
        "ubatch": 1024,
        "offload": None,
        "ot": None,
    }
    caps: Capabilities = {"vision": False, "reasoning": False}
    def fake_detect_capabilities(_tag: str) -> Capabilities:
        return caps

    def fake_save_scan_cache(value: ScanCache) -> None:
        saved.append(value.copy())

    monkeypatch.setattr(fit_bench, "detect_capabilities", fake_detect_capabilities)
    monkeypatch.setattr(fit_bench, "save_scan_cache", fake_save_scan_cache)

    assert fit_bench.write_scan_cache_entry(
        cache,
        "repo/model:Q4_K_M",
        False,
        fit_bench.FIT_TARGET,
        chosen,
        1024,
        64,
        True,
    ) == caps
    assert saved
    entry = cache["repo/model:Q4_K_M"]
    assert entry.get("mmproj") == "64M"
    assert entry.get("moe") is True
    assert entry.get("caps") == caps
    text_entry = entry.get("text")
    assert text_entry is not None
    assert text_entry["ubatch_sizes"]["1024"]["ctx"] == 4096
    assert "moe" not in text_entry["ubatch_sizes"]["1024"]


def test_write_scan_cache_entry_rejects_missing_chosen_ctx(monkeypatch: MonkeyPatch) -> None:
    def fake_detect_capabilities(_tag: str) -> Capabilities:
        return {"vision": False, "reasoning": False}

    def fake_save_scan_cache(_cache: ScanCache) -> None:
        pass

    monkeypatch.setattr(fit_bench, "detect_capabilities", fake_detect_capabilities)
    monkeypatch.setattr(fit_bench, "save_scan_cache", fake_save_scan_cache)
    cache: ScanCache = {}

    with pytest.raises(ValueError, match="ctx"):
        fit_bench.write_scan_cache_entry(
            cache,
            "repo/model:Q4_K_M",
            False,
        fit_bench.FIT_TARGET,
            make_fit_result(4096, None, 72),
            1024,
            64,
            False,
        )

    assert cache == {}


def test_scan_and_bench_ubatch_scans_benchmarks_and_writes_outputs(
    monkeypatch: MonkeyPatch,
) -> None:
    cache: ScanCache = {}
    args = make_args()
    chosen = make_fit_result(20000, 20000, -1, ubatch=512, ot="0,1")
    caps: Capabilities = {"vision": False, "reasoning": False}
    calls: dict[str, int] = {"cache": 0, "row": 0}

    def fake_choose_scan_strategy(
        _tag: str,
        _ctx_targets: list[int] | tuple[int, ...],
        _fit_target: int,
        _max_ctx: int | None,
        _is_moe: bool,
        forced_ubatch: int | None = None,
    ) -> fit_bench.ScanStrategyResult:
        assert forced_ubatch == 512
        return [chosen], chosen, "selected", False

    def fake_run_bench(
        _tag: str,
        ngl: int,
        ot: str | None,
        ubatch: int,
        reps: int = fit_bench.REPS,
        prio: int | None = None,
    ) -> fit_bench.BenchResult:
        assert (ngl, ot, ubatch, reps, prio) == (-1, "0,1", 512, 3, 1)
        return {
            "size_bytes": "1",
            "n_params": "2",
            "pp_speed": "1",
            "pp_stddev": "0",
            "tg_speed": "2",
            "tg_stddev": "0",
        }

    def fake_write_scan_cache_entry(
        _cache: ScanCache,
        _tag: str,
        vision: bool,
        fit_target: int,
        cache_chosen: fit_bench.FitResult,
        ubatch: int,
        mmproj_mib: int,
        is_moe: bool,
        caps: Capabilities | None = None,
    ) -> Capabilities:
        assert (vision, fit_target, cache_chosen, ubatch, mmproj_mib, is_moe, caps) == (
            False,
            fit_bench.FIT_TARGET,
            chosen,
            512,
            0,
            False,
            {"vision": False, "reasoning": False},
        )
        calls["cache"] += 1
        return {"vision": False, "reasoning": False}

    def fake_write_result_row(
        _tag: str,
        row_chosen: fit_bench.FitResult,
        is_moe: bool,
        bench_result: fit_bench.BenchResult | None,
        row_caps: Capabilities,
        mode: str,
        fit_target: int,
        ubatch: int,
        reps: int,
    ) -> None:
        assert row_chosen == chosen
        assert is_moe is False
        assert bench_result is not None
        assert row_caps == caps
        assert (mode, fit_target, ubatch, reps) == ("text", fit_bench.FIT_TARGET, 512, 3)
        calls["row"] += 1

    monkeypatch.setattr(fit_bench, "choose_scan_strategy", fake_choose_scan_strategy)
    monkeypatch.setattr(fit_bench, "run_bench", fake_run_bench)
    monkeypatch.setattr(fit_bench, "write_scan_cache_entry", fake_write_scan_cache_entry)
    monkeypatch.setattr(fit_bench, "write_result_row", fake_write_result_row)
    monkeypatch.setattr(fit_bench, "print_scan_table", noop_print_scan_table)
    monkeypatch.setattr(fit_bench, "log", noop_log)

    assert fit_bench.scan_and_bench_ubatch(
        "repo/model:Q4_K_M",
        args,
        cache,
        fit_bench.FIT_TARGET,
        512,
        "text",
        0,
        False,
        max_ctx=20000,
        ctx_targets=[5000, 20000],
        caps=caps,
    ) == (chosen, False, {
        "size_bytes": "1",
        "n_params": "2",
        "pp_speed": "1",
        "pp_stddev": "0",
        "tg_speed": "2",
        "tg_stddev": "0",
    }, True)
    assert calls == {"cache": 1, "row": 1}


def test_scan_and_bench_ubatch_scan_only_writes_cache_without_benchmark(
    monkeypatch: MonkeyPatch,
) -> None:
    args = make_args(scan=True)
    chosen = make_fit_result(10000, 10000, 72, ubatch=1024)
    calls: list[str] = []

    def fake_choose_scan_strategy(
        _tag: str,
        _ctx_targets: list[int] | tuple[int, ...],
        _fit_target: int,
        _max_ctx: int | None,
        _is_moe: bool,
        forced_ubatch: int | None = None,
    ) -> fit_bench.ScanStrategyResult:
        del forced_ubatch
        return [chosen], chosen, "selected", True

    def fake_run_bench(
        _tag: str,
        _ngl: int,
        _ot: str | None,
        _ubatch: int,
        reps: int = fit_bench.REPS,
        prio: int | None = None,
    ) -> None:
        del reps, prio
        calls.append("bench")

    def fake_write_result_row(
        _tag: str,
        _chosen: fit_bench.FitResult,
        _is_moe: bool,
        _bench_result: fit_bench.BenchResult | None,
        _caps: Capabilities,
        _mode: str,
        _fit_target: int,
        _ubatch: int,
        _reps: int,
    ) -> None:
        calls.append("row")

    def fake_write_scan_cache_entry(
        _cache: ScanCache,
        _tag: str,
        _vision: bool,
        _fit_target: int,
        _chosen: fit_bench.FitResult,
        _ubatch: int,
        _mmproj_mib: int,
        _is_moe: bool,
        caps: Capabilities | None = None,
    ) -> Capabilities:
        del caps
        calls.append("cache")
        return {"vision": False, "reasoning": False}

    monkeypatch.setattr(
        fit_bench,
        "choose_scan_strategy",
        fake_choose_scan_strategy,
    )
    monkeypatch.setattr(fit_bench, "run_bench", fake_run_bench)
    monkeypatch.setattr(fit_bench, "write_result_row", fake_write_result_row)
    monkeypatch.setattr(
        fit_bench,
        "write_scan_cache_entry",
        fake_write_scan_cache_entry,
    )
    monkeypatch.setattr(fit_bench, "print_scan_table", noop_print_scan_table)
    monkeypatch.setattr(fit_bench, "log", noop_log)

    assert fit_bench.scan_and_bench_ubatch(
        "repo/model:Q4_K_M",
        args,
        {},
        fit_bench.FIT_TARGET,
        1024,
        "text",
        0,
        True,
        max_ctx=10000,
        ctx_targets=[5000, 10000],
    ) == (chosen, True, None, True)
    assert calls == ["cache"]


def test_scan_and_bench_ubatch_scan_only_reuses_valid_cached_entry(
    monkeypatch: MonkeyPatch,
) -> None:
    args = make_args(scan=True)
    cache: ScanCache = {
        "repo/model:Q4_K_M": {
            "text": {
                "ubatch_sizes": {
                    "512": {
                        "fit_target": fit_bench.FIT_TARGET,
                        "ctx": 8192,
                        "ngl": -1,
                        "offload": 2,
                        "ot": "0,1",
                        "scan_ts": "2026-01-01T00:00:00+00:00",
                    }
                }
            }
        }
    }

    def fail_choose_scan_strategy(
        _tag: str,
        _ctx_targets: list[int] | tuple[int, ...],
        _fit_target: int,
        _max_ctx: int | None,
        _is_moe: bool,
        forced_ubatch: int | None = None,
    ) -> fit_bench.ScanStrategyResult:
        del forced_ubatch
        raise AssertionError("choose_scan_strategy should not be called")

    def fail_run_bench(
        _tag: str,
        _ngl: int,
        _ot: str | None,
        _ubatch: int,
        reps: int = fit_bench.REPS,
        prio: int | None = None,
    ) -> fit_bench.BenchResult:
        del reps, prio
        raise AssertionError("run_bench should not be called")

    def fail_write_scan_cache_entry(
        _cache: ScanCache,
        _tag: str,
        _vision: bool,
        _fit_target: int,
        _chosen: fit_bench.FitResult,
        _ubatch: int,
        _mmproj_mib: int,
        _is_moe: bool,
        caps: Capabilities | None = None,
    ) -> Capabilities:
        del caps
        raise AssertionError("write_scan_cache_entry should not be called")

    monkeypatch.setattr(fit_bench, "choose_scan_strategy", fail_choose_scan_strategy)
    monkeypatch.setattr(fit_bench, "run_bench", fail_run_bench)
    monkeypatch.setattr(fit_bench, "write_scan_cache_entry", fail_write_scan_cache_entry)
    monkeypatch.setattr(fit_bench, "print_scan_table", noop_print_scan_table)
    monkeypatch.setattr(fit_bench, "log", noop_log)

    assert fit_bench.scan_and_bench_ubatch(
        "repo/model:Q4_K_M",
        args,
        cache,
        fit_bench.FIT_TARGET,
        512,
        "text",
        0,
        True,
        max_ctx=8192,
        ctx_targets=[8192],
    ) == (
        {
            "target_ctx": 8192,
            "ctx": 8192,
            "ngl": -1,
            "ubatch": 512,
            "offload": 2,
            "ot": "0,1",
        },
        True,
        None,
        False,
    )


def test_scan_and_bench_ubatch_skips_fresh_rebench_for_specific_ubatch(
    monkeypatch: MonkeyPatch,
) -> None:
    cutoff = datetime(2026, 1, 2, tzinfo=timezone.utc)
    args = make_args()
    args.rebench = "24h"
    args.rebench_cutoff = cutoff
    cache: ScanCache = {
        "repo/model:Q4_K_M": {
            "text": {
                "ubatch_sizes": {
                    "1024": {
                        "fit_target": fit_bench.FIT_TARGET,
                        "ctx": 8192,
                        "ngl": 84,
                        "offload": 2,
                        "ot": "0,1",
                        "scan_ts": "2026-01-01T00:00:00+00:00",
                    }
                }
            }
        }
    }
    chosen = {
        "target_ctx": 8192,
        "ctx": 8192,
        "ngl": 84,
        "ubatch": 1024,
        "offload": 2,
        "ot": "0,1",
    }

    def fake_get_bench_ts(_tag: str, mode: str = "text", ubatch: int | None = None) -> datetime | None:
        assert (mode, ubatch) == ("text", 1024)
        return cutoff + timedelta(hours=1)

    def fail_choose_scan_strategy(
        _tag: str,
        _ctx_targets: list[int] | tuple[int, ...],
        _fit_target: int,
        _max_ctx: int | None,
        _is_moe: bool,
        forced_ubatch: int | None = None,
    ) -> fit_bench.ScanStrategyResult:
        del forced_ubatch
        raise AssertionError("choose_scan_strategy should not be called")

    def fail_run_bench(
        _tag: str,
        _ngl: int,
        _ot: str | None,
        _ubatch: int,
        reps: int = fit_bench.REPS,
        prio: int | None = None,
    ) -> fit_bench.BenchResult:
        del reps, prio
        raise AssertionError("run_bench should not be called")

    def fail_write_result_row(
        _tag: str,
        _chosen: fit_bench.FitResult,
        _is_moe: bool,
        _bench_result: fit_bench.BenchResult | None,
        _caps: Capabilities,
        _mode: str,
        _fit_target: int,
        _ubatch: int,
        _reps: int,
    ) -> None:
        raise AssertionError("write_result_row should not be called")

    monkeypatch.setattr(fit_bench, "get_bench_ts", fake_get_bench_ts)
    monkeypatch.setattr(fit_bench, "choose_scan_strategy", fail_choose_scan_strategy)
    monkeypatch.setattr(fit_bench, "run_bench", fail_run_bench)
    monkeypatch.setattr(fit_bench, "write_result_row", fail_write_result_row)
    monkeypatch.setattr(fit_bench, "print_scan_table", noop_print_scan_table)
    monkeypatch.setattr(fit_bench, "log", noop_log)

    assert fit_bench.scan_and_bench_ubatch(
        "repo/model:Q4_K_M",
        args,
        cache,
            fit_bench.FIT_TARGET,
        1024,
        "text",
        0,
        True,
        max_ctx=8192,
        ctx_targets=[8192],
    ) == (chosen, True, None, False)


def test_benchmark_tag_reuses_cached_ubatch_without_resolving_max_ctx(
    monkeypatch: MonkeyPatch,
) -> None:
    args = make_args(ubatch=512)
    caps: Capabilities = {"vision": False, "reasoning": False}
    cache: ScanCache = {
        "repo/model:Q4_K_M": {
            "moe": True,
            "caps": caps,
            "text": {
                "ubatch_sizes": {
                    "512": {
                        "fit_target": fit_bench.FIT_TARGET,
                        "ctx": 8192,
                        "ngl": -1,
                        "offload": None,
                        "ot": None,
                        "scan_ts": "2026-01-01T00:00:00+00:00",
                    }
                }
            },
        }
    }
    bench_calls: list[int] = []
    rows: list[int] = []

    def fail_get_max_ctx(
        _tag: str, fit_target: int = fit_bench.FIT_TARGET, prio: int | None = None
    ) -> int:
        del fit_target, prio
        raise AssertionError("get_max_ctx should not be called")

    def fail_choose_scan_strategy(
        _tag: str,
        _ctx_targets: list[int] | tuple[int, ...],
        _fit_target: int,
        _max_ctx: int | None,
        _is_moe: bool,
        forced_ubatch: int | None = None,
    ) -> fit_bench.ScanStrategyResult:
        del forced_ubatch
        raise AssertionError("choose_scan_strategy should not be called")

    def fake_run_bench(
        _tag: str,
        _ngl: int,
        _ot: str | None,
        ubatch: int,
        reps: int = fit_bench.REPS,
        prio: int | None = None,
    ) -> fit_bench.BenchResult:
        del reps, prio
        bench_calls.append(ubatch)
        return {
            "size_bytes": "1",
            "n_params": "2",
            "pp_speed": "1",
            "pp_stddev": "0",
            "tg_speed": "2",
            "tg_stddev": "0",
        }

    def fake_write_result_row(
        _tag: str,
        _chosen: fit_bench.FitResult,
        _is_moe: bool,
        _bench_result: fit_bench.BenchResult | None,
        _caps: Capabilities,
        mode: str,
        fit_target: int,
        ubatch: int,
        reps: int,
    ) -> None:
        del mode, fit_target, reps
        rows.append(ubatch)

    def fail_write_scan_cache_entry(
        _cache: ScanCache,
        _tag: str,
        _vision: bool,
        _fit_target: int,
        _chosen: fit_bench.FitResult,
        _ubatch: int,
        _mmproj_mib: int,
        _is_moe: bool,
        caps: Capabilities | None = None,
    ) -> Capabilities:
        del caps
        raise AssertionError("write_scan_cache_entry should not be called")

    def fake_get_mmproj_size_mib(_tag: str) -> int:
        return 0

    def fake_print_summary(
        _display_name: str,
        _quant: str,
        _provider: str,
        _size_gib: str,
        _chosen: fit_bench.FitResult | None,
        _is_moe: bool,
        _bench_result: fit_bench.BenchResult | None,
    ) -> None:
        pass

    monkeypatch.setattr(fit_bench, "get_mmproj_size_mib", fake_get_mmproj_size_mib)
    monkeypatch.setattr(fit_bench, "get_max_ctx", fail_get_max_ctx)
    monkeypatch.setattr(fit_bench, "choose_scan_strategy", fail_choose_scan_strategy)
    monkeypatch.setattr(fit_bench, "run_bench", fake_run_bench)
    monkeypatch.setattr(fit_bench, "write_result_row", fake_write_result_row)
    monkeypatch.setattr(fit_bench, "write_scan_cache_entry", fail_write_scan_cache_entry)
    monkeypatch.setattr(fit_bench, "print_scan_table", noop_print_scan_table)
    monkeypatch.setattr(fit_bench, "print_summary", fake_print_summary)
    monkeypatch.setattr(fit_bench, "log", noop_log)

    fit_bench.benchmark_tag("repo/model:Q4_K_M", args, cache)

    assert bench_calls == [512]
    assert rows == [512]


def test_benchmark_tag_reuses_cached_moe_ubatch_then_resolves_max_ctx_once(
    monkeypatch: MonkeyPatch,
) -> None:
    args = make_args()
    caps: Capabilities = {"vision": False, "reasoning": False}
    cache: ScanCache = {
        "repo/model:Q4_K_M": {
            "moe": True,
            "caps": caps,
            "text": {
                "ubatch_sizes": {
                    "512": {
                        "fit_target": fit_bench.FIT_TARGET,
                        "ctx": 8192,
                        "ngl": 84,
                        "offload": None,
                        "ot": None,
                        "scan_ts": "2026-01-01T00:00:00+00:00",
                    }
                }
            },
        }
    }
    max_calls: list[str] = []
    scan_calls: list[int] = []
    bench_calls: list[int] = []

    def fake_get_max_ctx(
        tag: str, fit_target: int = fit_bench.FIT_TARGET, prio: int | None = None
    ) -> int:
        del fit_target, prio
        max_calls.append(tag)
        return 20000

    def fake_choose_scan_strategy(
        _tag: str,
        ctx_targets: list[int] | tuple[int, ...],
        _fit_target: int,
        max_ctx: int | None,
        _is_moe: bool,
        forced_ubatch: int | None = None,
    ) -> fit_bench.ScanStrategyResult:
        assert list(ctx_targets) == [5000, 10000, 20000]
        assert max_ctx == 20000
        assert forced_ubatch is not None
        scan_calls.append(forced_ubatch)
        chosen = make_fit_result(20000, 20000, 84, ubatch=forced_ubatch)
        return [chosen], chosen, "selected", True

    def fake_run_bench(
        _tag: str,
        _ngl: int,
        _ot: str | None,
        ubatch: int,
        reps: int = fit_bench.REPS,
        prio: int | None = None,
    ) -> fit_bench.BenchResult:
        del reps, prio
        bench_calls.append(ubatch)
        return {
            "size_bytes": "1",
            "n_params": "2",
            "pp_speed": "1",
            "pp_stddev": "0",
            "tg_speed": "2",
            "tg_stddev": "0",
        }

    def fake_write_scan_cache_entry(
        _cache: ScanCache,
        _tag: str,
        _vision: bool,
        _fit_target: int,
        _chosen: fit_bench.FitResult,
        _ubatch: int,
        _mmproj_mib: int,
        _is_moe: bool,
        caps: Capabilities | None = None,
    ) -> Capabilities:
        assert caps is not None
        return caps

    def fake_get_mmproj_size_mib(_tag: str) -> int:
        return 0

    def fake_write_result_row(
        _tag: str,
        _chosen: fit_bench.FitResult,
        _is_moe: bool,
        _bench_result: fit_bench.BenchResult | None,
        _caps: Capabilities,
        mode: str,
        fit_target: int,
        ubatch: int,
        reps: int,
    ) -> None:
        del mode, fit_target, ubatch, reps

    def fake_print_summary(
        _display_name: str,
        _quant: str,
        _provider: str,
        _size_gib: str,
        _chosen: fit_bench.FitResult | None,
        _is_moe: bool,
        _bench_result: fit_bench.BenchResult | None,
    ) -> None:
        pass

    monkeypatch.setattr(fit_bench, "get_mmproj_size_mib", fake_get_mmproj_size_mib)
    monkeypatch.setattr(fit_bench, "get_max_ctx", fake_get_max_ctx)
    monkeypatch.setattr(fit_bench, "choose_scan_strategy", fake_choose_scan_strategy)
    monkeypatch.setattr(fit_bench, "run_bench", fake_run_bench)
    monkeypatch.setattr(fit_bench, "write_result_row", fake_write_result_row)
    monkeypatch.setattr(fit_bench, "write_scan_cache_entry", fake_write_scan_cache_entry)
    monkeypatch.setattr(fit_bench, "print_scan_table", noop_print_scan_table)
    monkeypatch.setattr(fit_bench, "print_summary", fake_print_summary)
    monkeypatch.setattr(fit_bench, "log", noop_log)

    fit_bench.benchmark_tag("repo/model:Q4_K_M", args, cache)

    assert max_calls == ["repo/model:Q4_K_M"]
    assert scan_calls == [1024, 2048, 4096]
    assert bench_calls == [512, 1024, 2048, 4096]


def test_benchmark_tag_skips_non_vision_model_in_vision_mode(monkeypatch: MonkeyPatch) -> None:
    args = make_args()
    args.vision = True
    called: list[str] = []

    def fake_get_mmproj_size_mib(_tag: str) -> int:
        return 0

    def fake_scan_and_bench_ubatch(
        _tag: str,
        _args: fit_bench.Args,
        _cache: ScanCache,
        _fit_target: int,
        _ubatch: int,
        _mode: str,
        _mmproj_mib: int,
        _is_moe_model: bool,
        max_ctx: int | None = None,
        ctx_targets: list[int] | tuple[int, ...] | None = None,
        caps: Capabilities | None = None,
    ) -> tuple[fit_bench.FitResult | None, bool, fit_bench.BenchResult | None, bool]:
        del max_ctx, ctx_targets, caps
        called.append("scan")
        return None, False, None, False

    monkeypatch.setattr(fit_bench, "get_mmproj_size_mib", fake_get_mmproj_size_mib)
    monkeypatch.setattr(fit_bench, "scan_and_bench_ubatch", fake_scan_and_bench_ubatch)
    monkeypatch.setattr(fit_bench, "log", noop_log)

    fit_bench.benchmark_tag("repo/model:Q4_K_M", args, {})

    assert called == []
