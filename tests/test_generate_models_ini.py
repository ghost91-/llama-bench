# pyright: reportPrivateUsage=false

import csv
from pathlib import Path
from typing import Literal, cast

import pytest
from pytest import CaptureFixture, MonkeyPatch

import generate_models_ini
from llama_bench.results import PP_COL, TG_COL
from llama_bench.schema_types import ScanCache


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


def _bench_config(
    *,
    group: str = "foo-group",
    repo: str = "unsloth/Foo-GGUF",
    provider: str = "unsloth",
    quant: str = "Q4_K_M",
    mode: Literal["text", "vision"] = "text",
    ubatch: int = 512,
    ctx: int = 128_000,
    pp_tps: float = 100.0,
    tg_tps: float = 20.0,
    model: str = "Foo",
    fit_target: int | None = 128,
) -> generate_models_ini.BenchConfig:
    return generate_models_ini.BenchConfig(
        group=group,
        model=model,
        repo=repo,
        provider=provider,
        quant=quant,
        mode=mode,
        ubatch=ubatch,
        ctx=ctx,
        fit_target=fit_target,
        pp_tps=pp_tps,
        tg_tps=tg_tps,
    )


def test_build_ini_sections_emits_section_per_bench_config(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(generate_models_ini, "SAMPLER_CONFIG", {"foo-group": {"temp": "0.7"}})
    configs = [
        _bench_config(group="foo-group", quant="Q4_K_M", ubatch=512, ctx=128_000),
    ]

    sections = generate_models_ini.build_ini_sections(
        configs,
        gguf_exists_fn=lambda _tag: Path("/tmp/model.gguf"),
    )

    assert len(sections) == 1
    assert sections[0]["name"] == "foo-group-unsloth-q4-k-m-text-ub512"
    assert sections[0]["comment"] != ""
    props = dict(sections[0]["props"])
    assert props["hf"] == "unsloth/Foo-GGUF:Q4_K_M"
    assert props["ctx-size"] == "128000"
    assert props["fit-target"] == "128"
    assert props["ubatch-size"] == "512"
    assert props["batch-size"] == str(generate_models_ini._server_batch_size(512))
    assert props["temp"] == "0.7"


def test_build_ini_sections_adds_vision_props_for_vision_mode(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(generate_models_ini, "SAMPLER_CONFIG", {})
    configs = [
        _bench_config(
            group="foo-group",
            mode="vision",
            ctx=64_000,
            fit_target=192,
        ),
    ]

    sections = generate_models_ini.build_ini_sections(
        configs,
        gguf_exists_fn=lambda _tag: Path("/tmp/model.gguf"),
    )

    assert len(sections) == 1
    props = dict(sections[0]["props"])
    assert props["mmproj-auto"] == "on"
    assert props["mmproj-offload"] == "on"
    assert props["fit-target"] == "192"


def test_build_ini_sections_text_mode_has_no_mmproj(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(generate_models_ini, "SAMPLER_CONFIG", {})
    configs = [
        _bench_config(mode="text"),
    ]

    sections = generate_models_ini.build_ini_sections(
        configs,
        gguf_exists_fn=lambda _tag: Path("/tmp/model.gguf"),
    )

    props = dict(sections[0]["props"])
    assert "mmproj-auto" not in props
    assert "mmproj-offload" not in props


def test_build_ini_sections_orders_by_group_then_provider(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(generate_models_ini, "SAMPLER_CONFIG", {})
    configs = [
        _bench_config(group="z-group", provider="unsloth", repo="unsloth/Foo-GGUF"),
        _bench_config(group="a-group", provider="unsloth", repo="unsloth/Foo-GGUF"),
        _bench_config(group="a-group", provider="bartowski", repo="bartowski/Foo-GGUF"),
    ]

    sections = generate_models_ini.build_ini_sections(
        configs,
        gguf_exists_fn=lambda _tag: Path("/tmp/model.gguf"),
    )

    assert [section["name"] for section in sections] == [
        "a-group-bartowski-q4-k-m-text-ub512",
        "a-group-unsloth-q4-k-m-text-ub512",
        "z-group-unsloth-q4-k-m-text-ub512",
    ]


def test_build_ini_sections_fails_on_missing_gguf() -> None:
    configs = [
        _bench_config(),
    ]

    with pytest.raises(generate_models_ini.MissingSelectedModelError) as exc_info:
        generate_models_ini.build_ini_sections(
            configs,
            gguf_exists_fn=lambda _tag: None,
        )

    assert "not found on disk" in str(exc_info.value)


def test_render_ini_includes_comments_and_sections() -> None:
    sections: list[generate_models_ini.IniSection] = [
        {
            "name": "foo-group-agentic",
            "props": [("hf", "unsloth/Foo-GGUF:Q4_K_M"), ("ctx-size", "128000")],
            "comment": "Agentic coding default. text, ctx 128k, pp 100, tg 20.",
        },
    ]

    content = generate_models_ini.render_ini(sections)

    assert "version = 1" in content
    assert "; Agentic coding default." in content
    assert "[foo-group-agentic]" in content
    assert "hf = unsloth/Foo-GGUF:Q4_K_M" in content
    assert "ctx-size = 128000" in content


def test_render_ini_omits_empty_comment() -> None:
    sections: list[generate_models_ini.IniSection] = [
        {
            "name": "test",
            "props": [("hf", "x:y")],
            "comment": "",
        },
    ]

    content = generate_models_ini.render_ini(sections)

    assert ";" not in content.split("[test]")[0].split("\n")[-1]


def test_render_ini_header() -> None:
    content = generate_models_ini.render_ini([])

    assert content.startswith("version = 1\n")
    assert "[*]" in content
    assert "fit = on" in content
    assert "flash-attn = on" in content
    assert f"parallel = {generate_models_ini.SERVER_PARALLEL}" in content
    assert f"batch-size = {generate_models_ini._server_batch_size(512)}" in content


def test_format_sampler_settings_skips_keys(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        generate_models_ini,
        "SAMPLER_CONFIG",
        {
            "family": {
                "temperature": "0.7",
                "ubatch-size": "512",
                "reasoning": "on",
                "chat-template-kwargs": '{"low_effort": false}',
            }
        },
    )

    assert generate_models_ini.format_sampler_settings("family", skip_keys={"temperature"}) == [
        ("ubatch-size", "512"),
        ("reasoning", "on"),
        ("chat-template-kwargs", '{"low_effort": false}'),
    ]


def test_load_ini_selections_reads_models_toml(tmp_path: Path) -> None:
    models_file = tmp_path / "models.toml"
    models_file.write_text(
        """
[[models]]
repo = "unsloth/Foo-GGUF"
quant = "Q4_K_M"
group = "foo-group"
ini = [
  { mode = "text", ubatch = 512 },
  { mode = "vision", ubatch = 2048 },
]
""".lstrip()
    )

    selections = generate_models_ini.load_ini_selections(str(models_file))

    assert selections == [
        generate_models_ini.IniSelection(
            "foo-group", "Foo", "unsloth/Foo-GGUF", "unsloth", "Q4_K_M", "text", 512,
        ),
        generate_models_ini.IniSelection(
            "foo-group", "Foo", "unsloth/Foo-GGUF", "unsloth", "Q4_K_M", "vision", 2048,
        ),
    ]


def test_load_ini_selections_rejects_invalid_fields(tmp_path: Path) -> None:
    models_file = tmp_path / "models.toml"
    models_file.write_text(
        """
[[models]]
repo = "unsloth/Foo-GGUF"
quant = "Q4_K_M"
group = "foo-group"
ini = [{ mode = "text", ubatch = 512, ctx = 128000 }]
""".lstrip()
    )

    with pytest.raises(generate_models_ini.InvalidSelectionError) as exc_info:
        generate_models_ini.load_ini_selections(str(models_file))

    assert "unknown fields: ctx" in str(exc_info.value)


def test_load_bench_configs_rejects_duplicate_selection_rows(
    monkeypatch: MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        generate_models_ini,
        "load_models",
        lambda: [("unsloth/Foo-GGUF", "Q4_K_M", "foo-group")],
    )
    results_file = tmp_path / "fit-bench-results.csv"
    with results_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "quant", "provider", "mode", "fit_target", "ctx", "ubatch",
            PP_COL, TG_COL,
        ])
        writer.writeheader()
        writer.writerow({
            "model": "Foo", "quant": "Q4_K_M", "provider": "unsloth",
            "mode": "text", "fit_target": "128", "ctx": "50k", "ubatch": "512",
            PP_COL: "1000.0", TG_COL: "100.0",
        })
        writer.writerow({
            "model": "Foo", "quant": "Q4_K_M", "provider": "unsloth",
            "mode": "text", "fit_target": "192", "ctx": "64k", "ubatch": "512",
            PP_COL: "1100.0", TG_COL: "110.0",
        })

    with pytest.raises(generate_models_ini.InvalidSelectionError) as exc_info:
        generate_models_ini.load_bench_configs(str(results_file))

    assert "Duplicate benchmark rows" in str(exc_info.value)


def test_resolve_manual_selections_falls_back_to_scan_cache(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        generate_models_ini,
        "load_scan_cache",
        lambda: {
            "unsloth/Foo-GGUF:Q4_K_M": {
                "text": {"ubatch_sizes": {"512": {"ctx": 64000, "fit_target": 128}}},
            },
        },
    )
    selection = generate_models_ini.IniSelection(
        "foo-group", "Foo", "unsloth/Foo-GGUF", "unsloth", "Q4_K_M", "text", 512,
    )

    configs = generate_models_ini.resolve_manual_selections([selection], [])

    assert configs == [
        generate_models_ini.BenchConfig(
            "foo-group",
            "Foo",
            "unsloth/Foo-GGUF",
            "unsloth",
            "Q4_K_M",
            "text",
            512,
            64000,
            128,
            0.0,
            0.0,
        )
    ]


def test_resolve_manual_selections_matches_model_name() -> None:
    configs = [
        _bench_config(model="Foo"),
        _bench_config(model="Bar"),
    ]
    selection = generate_models_ini.IniSelection(
        "foo-group", "Bar", "unsloth/Bar-GGUF", "unsloth", "Q4_K_M", "text", 512,
    )

    selected = generate_models_ini.resolve_manual_selections([selection], configs)

    assert [config.model for config in selected] == ["Bar"]


def test_add_free_vision_configs_adds_matching_bench_config() -> None:
    text_config = _bench_config(mode="text", ubatch=512, ctx=128_000, fit_target=256)
    vision_config = _bench_config(mode="vision", ubatch=512, ctx=128_000, fit_target=453)

    configs = generate_models_ini.add_free_vision_configs([text_config], [text_config, vision_config])

    assert configs == [text_config, vision_config]


def test_add_free_vision_configs_ignores_different_ctx_or_ubatch(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(generate_models_ini, "load_scan_cache", lambda: cast(ScanCache, {}))
    text_config = _bench_config(mode="text", ubatch=512, ctx=128_000, fit_target=256)
    different_ctx = _bench_config(mode="vision", ubatch=512, ctx=100_000, fit_target=453)
    different_ubatch = _bench_config(mode="vision", ubatch=1024, ctx=128_000, fit_target=453)

    configs = generate_models_ini.add_free_vision_configs([text_config], [different_ctx, different_ubatch])

    assert configs == [text_config]


def test_add_free_vision_configs_falls_back_to_matching_scan_cache(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        generate_models_ini,
        "load_scan_cache",
        lambda: {
            "unsloth/Foo-GGUF:Q4_K_M": {
                "vision": {"ubatch_sizes": {"512": {"ctx": 128000, "fit_target": 453}}},
            },
        },
    )
    text_config = _bench_config(mode="text", ubatch=512, ctx=128_000, fit_target=256)

    configs = generate_models_ini.add_free_vision_configs([text_config], [])

    assert configs == [
        text_config,
        generate_models_ini.BenchConfig(
            "foo-group",
            "Foo",
            "unsloth/Foo-GGUF",
            "unsloth",
            "Q4_K_M",
            "vision",
            512,
            128000,
            453,
            0.0,
            0.0,
        ),
    ]


def test_batch_size_uses_server_batch_rule(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(generate_models_ini, "SAMPLER_CONFIG", {})
    configs = [
        _bench_config(group="moe", ubatch=2048),
    ]

    sections = generate_models_ini.build_ini_sections(
        configs,
        gguf_exists_fn=lambda _tag: Path("/tmp/model.gguf"),
    )

    props = dict(sections[0]["props"])
    assert props["ubatch-size"] == "2048"
    assert props["batch-size"] == str(generate_models_ini._server_batch_size(2048))


def test_generate_ini_dry_run_prints_content(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture[str],
) -> None:
    monkeypatch.setattr(generate_models_ini, "SAMPLER_CONFIG", {})
    monkeypatch.setattr(generate_models_ini, "find_local_gguf_path", _fake_gguf_path)
    configs = [
        _bench_config(quant="Q4_K_M", ubatch=512, ctx=128_000),
    ]

    generate_models_ini.generate_ini(configs, "ignored.ini", dry_run=True)

    captured = capsys.readouterr()
    assert "version = 1" in captured.out
    assert "[foo-group-unsloth-q4-k-m-text-ub512]" in captured.out
    assert "hf = unsloth/Foo-GGUF:Q4_K_M" in captured.out


def test_generate_ini_writes_file(
    monkeypatch: MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(generate_models_ini, "SAMPLER_CONFIG", {})
    monkeypatch.setattr(generate_models_ini, "find_local_gguf_path", _fake_gguf_path)
    output = tmp_path / "models.ini"
    configs = [
        _bench_config(quant="Q4_K_M", ubatch=512, ctx=128_000),
    ]

    generate_models_ini.generate_ini(configs, str(output), dry_run=False)

    content = output.read_text()
    sections = _parse_ini_sections(content)
    assert "foo-group-unsloth-q4-k-m-text-ub512" in sections
    assert sections["foo-group-unsloth-q4-k-m-text-ub512"]["hf"] == "unsloth/Foo-GGUF:Q4_K_M"


def test_fit_target_missing_gracefully_omitted(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(generate_models_ini, "SAMPLER_CONFIG", {})
    configs = [
        _bench_config(ctx=64_000, fit_target=None),
    ]

    sections = generate_models_ini.build_ini_sections(
        configs,
        gguf_exists_fn=lambda _tag: Path("/tmp/model.gguf"),
    )

    props = dict(sections[0]["props"])
    assert "fit-target" not in props
    assert props["ctx-size"] == "64000"


def test_vision_merged_into_text_section_when_config_matches(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(generate_models_ini, "SAMPLER_CONFIG", {})
    text_config = _bench_config(
        mode="text", ubatch=512, ctx=128_000, fit_target=256,
    )
    vision_config = _bench_config(
        mode="vision", ubatch=512, ctx=128_000, fit_target=453,
    )

    sections = generate_models_ini.build_ini_sections(
        [text_config, vision_config],
        gguf_exists_fn=lambda _tag: Path("/tmp/model.gguf"),
    )

    assert len(sections) == 1
    assert sections[0]["name"] == "foo-group-unsloth-q4-k-m-text-ub512"
    props = dict(sections[0]["props"])
    assert props["fit-target"] == "453"
    assert props["mmproj-auto"] == "on"
    assert props["mmproj-offload"] == "on"


def test_vision_section_kept_separate_when_config_differs(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(generate_models_ini, "SAMPLER_CONFIG", {})
    text_config = _bench_config(
        mode="text", ubatch=2048, ctx=100_000, fit_target=256,
    )
    vision_config = _bench_config(
        mode="vision", ubatch=512, ctx=100_000, fit_target=453,
    )

    sections = generate_models_ini.build_ini_sections(
        [text_config, vision_config],
        gguf_exists_fn=lambda _tag: Path("/tmp/model.gguf"),
    )

    assert len(sections) == 2
    text_props = dict(sections[0]["props"])
    vision_props = dict(sections[1]["props"])
    assert "mmproj-auto" not in text_props
    assert vision_props["mmproj-auto"] == "on"
    assert text_props["ubatch-size"] == "2048"
    assert vision_props["ubatch-size"] == "512"


def test_text_sections_with_same_fit_shape_are_not_merged(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(generate_models_ini, "SAMPLER_CONFIG", {})
    first_text = _bench_config(
        mode="text", ubatch=512, ctx=128_000, pp_tps=100.0, tg_tps=20.0,
    )
    second_text = _bench_config(
        mode="text", ubatch=512, ctx=128_000, pp_tps=101.0, tg_tps=21.0,
    )

    sections = generate_models_ini.build_ini_sections(
        [first_text, second_text],
        gguf_exists_fn=lambda _tag: Path("/tmp/model.gguf"),
    )

    assert [section["name"] for section in sections] == [
        "foo-group-unsloth-q4-k-m-text-ub512",
        "foo-group-unsloth-q4-k-m-text-ub512-2",
    ]
    assert all("mmproj-auto" not in dict(section["props"]) for section in sections)
