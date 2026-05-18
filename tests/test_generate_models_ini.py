import csv
from pathlib import Path

from pytest import CaptureFixture, MonkeyPatch

import generate_models_ini
from llama_bench import results as bench_results


def _fake_gguf_path(tag: str) -> Path:
    return Path("/tmp/model.gguf")


def _parse_ini_sections(content: str) -> dict[str, dict[str, str]]:
    sections: dict[str, dict[str, str]] = {}
    current: str | None = None
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            sections[current] = {}
            continue
        if current is not None and "=" in line:
            key, value = line.split("=", 1)
            sections[current][key.strip()] = value.strip()
    return sections


def _selected(ctx: int, ubatch: int, pp: float, tg: float) -> generate_models_ini.SelectedResult:
    return {
        "ctx": ctx,
        "fit_target": 128,
        "ngl": -1,
        "ubatch": ubatch,
        "pp4096_tps": pp,
        "tg128_tps": tg,
    }


def test_load_result_summary_reads_result_csv(tmp_path: Path) -> None:
    results_file = tmp_path / "fit-bench-results.csv"
    with results_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=bench_results.CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerow(
            {
                "model": "Foo",
                "quant": "Q4_K_M",
                "provider": "unsloth",
                "mode": "text",
                "fit_target": "128",
                "ctx": "50k",
                "ngl": "all",
                "ubatch": "512",
                bench_results.PP_COL: "1000.0",
                bench_results.TG_COL: "100.0",
            }
        )

    parsed = generate_models_ini.load_result_summary(str(results_file))

    assert parsed[("Foo", "Q4_K_M", "unsloth")].get("text") == {
        "ctx": 50000,
        "fit_target": 128,
        "ngl": -1,
        "ubatch": 512,
        "pp4096_tps": 1000.0,
        "tg128_tps": 100.0,
    }


def test_select_result_row_gpt_oss_120b_like_floor_50k_best_pp() -> None:
    rows = [
        _selected(50_000, 4096, 900, 100),
        _selected(75_000, 1024, 500, 100),
        _selected(40_000, 4096, 1200, 100),
    ]

    assert generate_models_ini.select_result_row(rows, "text") == rows[0]


def test_select_result_row_glm_like_keeps_fast_100k_max_ctx() -> None:
    rows = [
        _selected(75_000, 4096, 900, 100),
        _selected(100_000, 1024, 600, 100),
    ]

    assert generate_models_ini.select_result_row(rows, "text") == rows[1]


def test_select_result_row_gemma_vision_like_floor_100k() -> None:
    rows = [
        _selected(75_000, 4096, 1200, 100),
        _selected(125_000, 1024, 800, 100),
    ]

    assert generate_models_ini.select_result_row(rows, "vision") == rows[1]


def test_select_result_row_qwen_text_like_floor_125k() -> None:
    rows = [
        _selected(100_000, 1024, 1200, 100),
        _selected(150_000, 4096, 900, 100),
        _selected(125_000, 1024, 800, 100),
    ]

    assert generate_models_ini.select_result_row(rows, "text") == rows[1]


def test_generate_ini_merges_same_vision_profile_into_main_section(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "models.ini"
    monkeypatch.setattr(generate_models_ini, "find_local_gguf_path", _fake_gguf_path)

    models = [("unsloth/Qwen3.5-9B-GGUF", "Q4_K_M", "qwen3.5-9b", True)]
    results: generate_models_ini.ParsedResults = {
        ("Qwen3.5-9B", "Q4_K_M", "unsloth"): {
            "text": _selected(8192, 1024, 1000, 100),
            "vision": {
                "ctx": 8192,
                "fit_target": 192,
                "ngl": -1,
                "ubatch": 1024,
                "pp4096_tps": 1000,
                "tg128_tps": 100,
            },
        }
    }

    generate_models_ini.generate_ini(models, results, str(output), dry_run=False)

    content = output.read_text()
    sections = _parse_ini_sections(content)

    assert "unsloth/Qwen3.5-9B-GGUF:Q4_K_M" in sections
    assert "unsloth/Qwen3.5-9B-GGUF:Q4_K_M:vision" not in sections
    section = sections["unsloth/Qwen3.5-9B-GGUF:Q4_K_M"]
    assert section["ctx-size"] == "8192"
    assert section["fit-target"] == "192"
    assert section["ubatch-size"] == "1024"
    assert section["mmproj-auto"] == "on"
    assert section["mmproj-offload"] == "on"


def test_generate_ini_writes_separate_vision_section_when_needed(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "models.ini"
    monkeypatch.setattr(generate_models_ini, "find_local_gguf_path", _fake_gguf_path)

    models = [("unsloth/Qwen3.5-9B-GGUF", "Q4_K_M", "qwen3.5-9b", True)]
    results: generate_models_ini.ParsedResults = {
        ("Qwen3.5-9B", "Q4_K_M", "unsloth"): {
            "text": _selected(8192, 1024, 1000, 100),
            "vision": {
                "ctx": 4096,
                "fit_target": 192,
                "ngl": 72,
                "ubatch": 8192,
                "pp4096_tps": 1000,
                "tg128_tps": 100,
            },
        }
    }

    generate_models_ini.generate_ini(models, results, str(output), dry_run=False)

    content = output.read_text()
    sections = _parse_ini_sections(content)

    text_section = sections["unsloth/Qwen3.5-9B-GGUF:Q4_K_M"]
    vision_section = sections["unsloth/Qwen3.5-9B-GGUF:Q4_K_M:vision"]
    assert text_section["ctx-size"] == "8192"
    assert text_section["fit-target"] == "128"
    assert text_section["ubatch-size"] == "1024"
    assert text_section["batch-size"] == "4096"
    assert "mmproj-auto" not in text_section
    assert vision_section["ctx-size"] == "4096"
    assert vision_section["fit-target"] == "192"
    assert vision_section["ubatch-size"] == "8192"
    assert vision_section["batch-size"] == "32768"
    assert vision_section["mmproj-auto"] == "on"


def test_generate_ini_writes_section_local_text_only_values(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "models.ini"
    monkeypatch.setattr(generate_models_ini, "find_local_gguf_path", _fake_gguf_path)
    monkeypatch.setattr(generate_models_ini, "SAMPLER_CONFIG", {"family": {"temp": "0.7"}})
    models = [("unsloth/TextOnly-GGUF", "Q4_K_M", "family", True)]
    results: generate_models_ini.ParsedResults = {
        ("TextOnly", "Q4_K_M", "unsloth"): {"text": _selected(4096, 512, 1000, 100)}
    }

    generate_models_ini.generate_ini(models, results, str(output), dry_run=False)

    sections = _parse_ini_sections(output.read_text())
    section = sections["unsloth/TextOnly-GGUF:Q4_K_M"]
    assert section["ctx-size"] == "4096"
    assert section["fit-target"] == "128"
    assert section["ubatch-size"] == "512"
    assert section["batch-size"] == "2048"
    assert section["temp"] == "0.7"
    assert "mmproj-auto" not in section
    assert "unsloth/TextOnly-GGUF:Q4_K_M:vision" not in sections


def test_generate_ini_suppresses_sampler_ubatch_when_scan_has_ubatch(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "models.ini"
    monkeypatch.setattr(generate_models_ini, "find_local_gguf_path", _fake_gguf_path)
    monkeypatch.setattr(
        generate_models_ini,
        "SAMPLER_CONFIG",
        {"family": {"ubatch-size": "128", "temp": "0.7"}},
    )
    models = [("unsloth/Foo-GGUF", "Q4_K_M", "family", True)]
    results: generate_models_ini.ParsedResults = {
        ("Foo", "Q4_K_M", "unsloth"): {"text": _selected(4096, 1024, 1000, 100)}
    }

    generate_models_ini.generate_ini(models, results, str(output), dry_run=False)

    section = _parse_ini_sections(output.read_text())["unsloth/Foo-GGUF:Q4_K_M"]
    assert section["ubatch-size"] == "1024"
    assert section["temp"] == "0.7"


def test_build_ini_sections_plans_sections(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(generate_models_ini, "SAMPLER_CONFIG", {"family": {"temp": "0.7"}})
    models = [("unsloth/Foo-GGUF", "Q4_K_M", "family", True)]
    results: generate_models_ini.ParsedResults = {
        ("Foo", "Q4_K_M", "unsloth"): {"text": _selected(4096, 512, 1000, 100)}
    }
    warnings: list[str] = []

    sections = generate_models_ini.build_ini_sections(
        models,
        results,
        gguf_exists_fn=lambda _tag: Path("/tmp/model.gguf"),
        warn=warnings.append,
    )

    assert warnings == []
    assert sections == [
        {
            "name": "unsloth/Foo-GGUF:Q4_K_M",
            "props": [
                ("hf", "unsloth/Foo-GGUF:Q4_K_M"),
                ("ctx-size", "4096"),
                ("fit-target", "128"),
                ("ubatch-size", "512"),
                ("batch-size", "2048"),
                ("temp", "0.7"),
            ],
        },
    ]


def test_render_ini_renders_sections() -> None:
    content = generate_models_ini.render_ini(
        [
            {
                "name": "unsloth/Foo-GGUF:Q4_K_M",
                "props": [("hf", "unsloth/Foo-GGUF:Q4_K_M"), ("ctx-size", "4096")],
            },
        ]
    )

    assert content == (
        "version = 1\n"
        "\n"
        "[*]\n"
        "fit = on\n"
        "fit-ctx = 5000\n"
        "flash-attn = on\n"
        "parallel = 4\n"
        "batch-size = 2048\n"
        "\n"
        "[unsloth/Foo-GGUF:Q4_K_M]\n"
        "hf = unsloth/Foo-GGUF:Q4_K_M\n"
        "ctx-size = 4096\n"
    )


def test_batch_size_is_server_parallel_times_ubatch(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "models.ini"
    monkeypatch.setattr(generate_models_ini, "find_local_gguf_path", _fake_gguf_path)
    monkeypatch.setattr(generate_models_ini, "SAMPLER_CONFIG", {})
    models = [("unsloth/MoE-GGUF", "Q4_K_M", "moe", True)]
    results: generate_models_ini.ParsedResults = {
        ("MoE", "Q4_K_M", "unsloth"): {"text": _selected(4096, 2048, 1000, 100)}
    }

    generate_models_ini.generate_ini(models, results, str(output), dry_run=False)

    section = _parse_ini_sections(output.read_text())["unsloth/MoE-GGUF:Q4_K_M"]
    assert section["ubatch-size"] == "2048"
    assert section["batch-size"] == str(generate_models_ini.SERVER_PARALLEL * 2048)


def test_format_sampler_settings_skips_keys(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        generate_models_ini,
        "SAMPLER_CONFIG",
        {
            "family": {
                "temperature": "0.7",
                "ubatch-size": "512",
                "chat-template-kwargs": '{"enable_thinking": true, "low_effort": false}',
            }
        },
    )

    assert generate_models_ini.format_sampler_settings("family", skip_keys={"temperature"}) == [
        ("ubatch-size", "512"),
        ("chat-template-kwargs", '{"enable_thinking": true, "low_effort": false}'),
    ]


def test_generate_ini_dry_run_skips_unpinned_missing_and_unscanned_models(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    found = {"unsloth/Foo-GGUF:Q4_K_M", "unsloth/Bar-GGUF:Q4_K_M"}

    def fake_find_local_gguf_path(tag: str) -> Path | None:
        return Path("/tmp/model.gguf") if tag in found else None

    monkeypatch.setattr(generate_models_ini, "find_local_gguf_path", fake_find_local_gguf_path)
    monkeypatch.setattr(generate_models_ini, "SAMPLER_CONFIG", {"foo": {"temperature": "0.6"}})
    models = [
        ("unsloth/Foo-GGUF", "Q4_K_M", "foo", True),
        ("unsloth/Bar-GGUF", "Q4_K_M", "foo", True),
        ("unsloth/Baz-GGUF", "Q4_K_M", "foo", False),
    ]
    results: generate_models_ini.ParsedResults = {
        ("Foo", "Q4_K_M", "unsloth"): {"text": _selected(4096, 512, 1000, 100)}
    }

    generate_models_ini.generate_ini(models, results, "ignored.ini", dry_run=True)

    captured = capsys.readouterr()
    assert "version = 1" in captured.out
    assert "[unsloth/Foo-GGUF:Q4_K_M]" in captured.out
    assert "temperature = 0.6" in captured.out
    assert "unsloth/Bar-GGUF:Q4_K_M has no benchmark results" in captured.err
    assert "Baz" not in captured.out
