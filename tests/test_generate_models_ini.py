import csv
from pathlib import Path
from typing import Literal

from pytest import CaptureFixture, MonkeyPatch

import generate_models_ini
from llama_bench.consolidation import ConfigKey, LabelledConfig, ReportEntry
from llama_bench.results import PP_COL, TG_COL
from llama_bench.selection import Candidate, ProfileSelection, Quality, ScoredCandidate


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


def _candidate(
    *,
    group: str = "foo-group",
    model: str = "Foo",
    quant: str = "Q4_K_M",
    provider: str = "unsloth",
    mode: Literal["text", "vision"] = "text",
    ctx: int = 128_000,
    ubatch: int = 512,
    pp_tps: float = 100.0,
    tg_tps: float = 20.0,
) -> Candidate:
    return Candidate(
        group=group,
        model=model,
        quant=quant,
        provider=provider,
        mode=mode,
        ctx=ctx,
        ubatch=ubatch,
        pp_tps=pp_tps,
        tg_tps=tg_tps,
        params=8_000_000_000,
        size_gib=4.0,
        kld=None,
    )


def _scored(candidate: Candidate, quality_score: float = 0.75) -> ScoredCandidate:
    return ScoredCandidate(
        candidate=candidate,
        quality=Quality(score=quality_score, source="quant-proxy", kld=None),
        score=0.8,
    )


def _selection(
    group: str = "foo-group",
    profile: str = "agentic-coding",
    scored: ScoredCandidate | None = None,
) -> ProfileSelection:
    return ProfileSelection(
        group=group,
        profile=profile,
        recommendation=scored,
        alternatives={},
        skipped_reason=None if scored else "no results",
    )


def _labelled_config(
    *,
    label: str = "foo-group-agentic",
    description: str = "Agentic coding default. text, ctx 128k, pp 100, tg 20.",
    group: str = "foo-group",
    provider: str = "unsloth",
    quant: str = "Q4_K_M",
    mode: Literal["text", "vision"] = "text",
    ubatch: int = 512,
    ctx: int = 128_000,
    pp_tps: float = 100.0,
    tg_tps: float = 20.0,
    model: str = "Foo",
) -> LabelledConfig:
    cand = _candidate(
        group=group, model=model, quant=quant, provider=provider, mode=mode,
        ctx=ctx, ubatch=ubatch, pp_tps=pp_tps, tg_tps=tg_tps,
    )
    scored = _scored(cand)
    sel = _selection(group=group, scored=scored)
    entry = ReportEntry(sel, "recommended", scored)
    key: ConfigKey = (group, provider, quant, mode, ubatch, ctx, pp_tps, tg_tps)
    return LabelledConfig(label=label, description=description, key=key, entries=(entry,))


def test_build_ini_sections_emits_section_per_labelled_config(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(generate_models_ini, "SAMPLER_CONFIG", {"foo-group": {"temp": "0.7"}})
    configs = [
        _labelled_config(
            label="foo-group-agentic",
            group="foo-group",
            quant="Q4_K_M",
            ubatch=512,
            ctx=128_000,
        ),
    ]
    fit_lookup: generate_models_ini.FitLookup = {
        ("Foo", "Q4_K_M", "unsloth", "text", 512, 128_000): 128,
    }
    repo_lookup: dict[tuple[str, str, str], str] = {
        ("Foo", "Q4_K_M", "unsloth"): "unsloth/Foo-GGUF",
    }
    warnings: list[str] = []

    sections = generate_models_ini.build_ini_sections(
        configs, fit_lookup, repo_lookup,
        gguf_exists_fn=lambda _tag: Path("/tmp/model.gguf"),
        warn=warnings.append,
    )

    assert warnings == []
    assert len(sections) == 1
    assert sections[0]["name"] == "foo-group-agentic"
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
        _labelled_config(
            label="foo-group-vision",
            group="foo-group",
            mode="vision",
            ctx=64_000,
        ),
    ]
    fit_lookup: generate_models_ini.FitLookup = {
        ("Foo", "Q4_K_M", "unsloth", "vision", 512, 64_000): 192,
    }
    repo_lookup: dict[tuple[str, str, str], str] = {
        ("Foo", "Q4_K_M", "unsloth"): "unsloth/Foo-GGUF",
    }

    sections = generate_models_ini.build_ini_sections(
        configs, fit_lookup, repo_lookup,
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
        _labelled_config(label="foo-group-chat", mode="text"),
    ]
    fit_lookup: generate_models_ini.FitLookup = {
        ("Foo", "Q4_K_M", "unsloth", "text", 512, 128_000): 128,
    }
    repo_lookup: dict[tuple[str, str, str], str] = {
        ("Foo", "Q4_K_M", "unsloth"): "unsloth/Foo-GGUF",
    }

    sections = generate_models_ini.build_ini_sections(
        configs, fit_lookup, repo_lookup,
        gguf_exists_fn=lambda _tag: Path("/tmp/model.gguf"),
    )

    props = dict(sections[0]["props"])
    assert "mmproj-auto" not in props
    assert "mmproj-offload" not in props


def test_build_ini_sections_warns_on_missing_repo() -> None:
    configs = [
        _labelled_config(model="Missing", quant="Q5_K_M", provider="unknown"),
    ]
    fit_lookup: generate_models_ini.FitLookup = {}
    repo_lookup: dict[tuple[str, str, str], str] = {}
    warnings: list[str] = []

    sections = generate_models_ini.build_ini_sections(
        configs, fit_lookup, repo_lookup,
        gguf_exists_fn=lambda _tag: Path("/tmp/model.gguf"),
        warn=warnings.append,
    )

    assert sections == []
    assert any("no repo" in w for w in warnings)


def test_build_ini_sections_warns_on_missing_gguf() -> None:
    configs = [
        _labelled_config(),
    ]
    fit_lookup: generate_models_ini.FitLookup = {
        ("Foo", "Q4_K_M", "unsloth", "text", 512, 128_000): 128,
    }
    repo_lookup: dict[tuple[str, str, str], str] = {
        ("Foo", "Q4_K_M", "unsloth"): "unsloth/Foo-GGUF",
    }
    warnings: list[str] = []

    sections = generate_models_ini.build_ini_sections(
        configs, fit_lookup, repo_lookup,
        gguf_exists_fn=lambda _tag: None,
        warn=warnings.append,
    )

    assert sections == []
    assert any("not found on disk" in w for w in warnings)


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


def test_load_fit_lookup_reads_results_csv(tmp_path: Path) -> None:
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

    lookup = generate_models_ini.load_fit_lookup(str(results_file))

    assert lookup[("Foo", "Q4_K_M", "unsloth", "text", 512, 50000)] == 128


def test_batch_size_uses_server_batch_rule(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(generate_models_ini, "SAMPLER_CONFIG", {})
    configs = [
        _labelled_config(label="moe-group-chat", group="moe", ubatch=2048),
    ]
    fit_lookup: generate_models_ini.FitLookup = {
        ("Foo", "Q4_K_M", "unsloth", "text", 2048, 128_000): 128,
    }
    repo_lookup: dict[tuple[str, str, str], str] = {
        ("Foo", "Q4_K_M", "unsloth"): "unsloth/Foo-GGUF",
    }

    sections = generate_models_ini.build_ini_sections(
        configs, fit_lookup, repo_lookup,
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
        _labelled_config(label="foo-group-agentic", quant="Q4_K_M", ubatch=512, ctx=128_000),
    ]
    fit_lookup: generate_models_ini.FitLookup = {
        ("Foo", "Q4_K_M", "unsloth", "text", 512, 128_000): 128,
    }
    repo_lookup: dict[tuple[str, str, str], str] = {
        ("Foo", "Q4_K_M", "unsloth"): "unsloth/Foo-GGUF",
    }

    generate_models_ini.generate_ini(
        configs, fit_lookup, repo_lookup, "ignored.ini", dry_run=True
    )

    captured = capsys.readouterr()
    assert "version = 1" in captured.out
    assert "[foo-group-agentic]" in captured.out
    assert "hf = unsloth/Foo-GGUF:Q4_K_M" in captured.out


def test_generate_ini_writes_file(
    monkeypatch: MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(generate_models_ini, "SAMPLER_CONFIG", {})
    monkeypatch.setattr(generate_models_ini, "find_local_gguf_path", _fake_gguf_path)
    output = tmp_path / "models.ini"
    configs = [
        _labelled_config(label="foo-group-agentic", quant="Q4_K_M", ubatch=512, ctx=128_000),
    ]
    fit_lookup: generate_models_ini.FitLookup = {
        ("Foo", "Q4_K_M", "unsloth", "text", 512, 128_000): 128,
    }
    repo_lookup: dict[tuple[str, str, str], str] = {
        ("Foo", "Q4_K_M", "unsloth"): "unsloth/Foo-GGUF",
    }

    generate_models_ini.generate_ini(
        configs, fit_lookup, repo_lookup, str(output), dry_run=False
    )

    content = output.read_text()
    sections = _parse_ini_sections(content)
    assert "foo-group-agentic" in sections
    assert sections["foo-group-agentic"]["hf"] == "unsloth/Foo-GGUF:Q4_K_M"


def test_fit_target_missing_gracefully_omitted(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(generate_models_ini, "SAMPLER_CONFIG", {})
    configs = [
        _labelled_config(label="foo-group-chat", ctx=64_000),
    ]
    fit_lookup: generate_models_ini.FitLookup = {}
    repo_lookup: dict[tuple[str, str, str], str] = {
        ("Foo", "Q4_K_M", "unsloth"): "unsloth/Foo-GGUF",
    }

    sections = generate_models_ini.build_ini_sections(
        configs, fit_lookup, repo_lookup,
        gguf_exists_fn=lambda _tag: Path("/tmp/model.gguf"),
    )

    props = dict(sections[0]["props"])
    assert "fit-target" not in props
    assert props["ctx-size"] == "64000"


def test_vision_merged_into_text_section_when_config_matches(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(generate_models_ini, "SAMPLER_CONFIG", {})
    text_config = _labelled_config(
        label="foo-group-agentic", mode="text", ubatch=512, ctx=128_000,
    )
    vision_config = _labelled_config(
        label="foo-group-vision", mode="vision", ubatch=512, ctx=128_000,
    )
    fit_lookup: generate_models_ini.FitLookup = {
        ("Foo", "Q4_K_M", "unsloth", "text", 512, 128_000): 256,
        ("Foo", "Q4_K_M", "unsloth", "vision", 512, 128_000): 453,
    }
    repo_lookup: dict[tuple[str, str, str], str] = {
        ("Foo", "Q4_K_M", "unsloth"): "unsloth/Foo-GGUF",
    }

    sections = generate_models_ini.build_ini_sections(
        [text_config, vision_config], fit_lookup, repo_lookup,
        gguf_exists_fn=lambda _tag: Path("/tmp/model.gguf"),
    )

    assert len(sections) == 1
    assert sections[0]["name"] == "foo-group-agentic"
    props = dict(sections[0]["props"])
    assert props["fit-target"] == "453"
    assert props["mmproj-auto"] == "on"
    assert props["mmproj-offload"] == "on"


def test_vision_section_kept_separate_when_config_differs(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(generate_models_ini, "SAMPLER_CONFIG", {})
    text_config = _labelled_config(
        label="foo-group-agentic", mode="text", ubatch=2048, ctx=100_000,
    )
    vision_config = _labelled_config(
        label="foo-group-vision", mode="vision", ubatch=512, ctx=100_000,
    )
    fit_lookup: generate_models_ini.FitLookup = {
        ("Foo", "Q4_K_M", "unsloth", "text", 2048, 100_000): 256,
        ("Foo", "Q4_K_M", "unsloth", "vision", 512, 100_000): 453,
    }
    repo_lookup: dict[tuple[str, str, str], str] = {
        ("Foo", "Q4_K_M", "unsloth"): "unsloth/Foo-GGUF",
    }

    sections = generate_models_ini.build_ini_sections(
        [text_config, vision_config], fit_lookup, repo_lookup,
        gguf_exists_fn=lambda _tag: Path("/tmp/model.gguf"),
    )

    assert len(sections) == 2
    text_props = dict(sections[0]["props"])
    vision_props = dict(sections[1]["props"])
    assert "mmproj-auto" not in text_props
    assert vision_props["mmproj-auto"] == "on"
    assert text_props["ubatch-size"] == "2048"
    assert vision_props["ubatch-size"] == "512"
