# pyright: reportPrivateUsage=false

import csv
from pathlib import Path

from pytest import MonkeyPatch

import llama_bench.selection as selection
from llama_bench.capabilities import ModelCapabilities
from llama_bench.results import PP_COL, TG_COL


def write_results(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "model",
        "quant",
        "provider",
        "mode",
        "size_gib",
        "params",
        "ctx",
        "ubatch",
        PP_COL,
        TG_COL,
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def result_row(
    *,
    quant: str = "Q4_K_M",
    provider: str = "unsloth",
    mode: str = "text",
    ctx: str = "128k",
    ubatch: str = "512",
    pp: str = "100.0",
    tg: str = "20.0",
    params: str = "8B",
) -> dict[str, str]:
    return {
        "model": "Foo",
        "quant": quant,
        "provider": provider,
        "mode": mode,
        "size_gib": "4.0",
        "params": params,
        "ctx": ctx,
        "ubatch": ubatch,
        PP_COL: pp,
        TG_COL: tg,
    }


def candidate(
    *,
    group: str = "foo-group",
    quant: str = "Q4_K_M",
    provider: str = "unsloth",
    ctx: int = 128_000,
    pp: float = 100.0,
    tg: float = 20.0,
    kld: float | None = None,
    params: int | None = 8_000_000_000,
) -> selection.Candidate:
    return selection.Candidate(
        group=group,
        model="Foo",
        quant=quant,
        provider=provider,
        mode="text",
        ctx=ctx,
        ubatch=512,
        pp_tps=pp,
        tg_tps=tg,
        params=params,
        size_gib=4.0,
        kld=kld,
    )


def capabilities(
    *,
    coding: int = 4,
    reasoning: int = 4,
    tools: int = 4,
    vision: int = 0,
    writing: int = 4,
    multilingual: int = 4,
) -> ModelCapabilities:
    values: ModelCapabilities = {
        "coding": coding,
        "reasoning": reasoning,
        "tools": tools,
        "vision": vision,
        "writing": writing,
        "multilingual": multilingual,
    }
    return values


def test_load_candidates_joins_models_and_kld(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    results = tmp_path / "fit-bench-results.csv"
    kld = tmp_path / "kld-results.csv"
    write_results(results, [result_row(), result_row(quant="Q5_K_M")])
    kld.write_text("model,quant,provider,kld\nFoo,Q4_K_M,unsloth,0.02\n", encoding="utf-8")
    monkeypatch.setattr(
        selection,
        "model_groups",
        lambda: {("Foo", "Q4_K_M", "unsloth"): "foo-group"},
    )

    rows = selection.load_candidates(str(results), str(kld))

    assert [(row.group, row.quant, row.kld) for row in rows] == [("foo-group", "Q4_K_M", 0.02)]


def test_estimate_quality_prefers_measured_then_estimated_then_proxy() -> None:
    measured = candidate(quant="Q4_K_M", kld=0.02)
    same_quant_missing = candidate(quant="Q4_K_M", kld=None)
    no_kld = candidate(quant="IQ2_M", kld=None, params=2_000_000_000)

    assert selection.estimate_quality(measured, [measured]).source == "measured-kld"

    estimated = selection.estimate_quality(same_quant_missing, [measured, same_quant_missing])
    assert estimated.source == "estimated-kld"
    assert estimated.kld == 0.02

    proxy = selection.estimate_quality(no_kld, [no_kld])
    assert proxy.source == "quant-proxy"
    assert proxy.score == 0.05


def test_estimate_quality_extrapolates_unmeasured_higher_quant_from_curve() -> None:
    q5 = candidate(quant="Q5_K_M", kld=0.01)
    q6 = candidate(quant="Q6_K", kld=0.006)
    q8 = candidate(quant="Q8_0", kld=None)

    estimated = selection.estimate_quality(q8, [q5, q6, q8])

    assert estimated.source == "estimated-kld"
    assert estimated.kld is not None
    assert estimated.kld < 0.006


def test_estimate_quality_interpolates_unmeasured_quant_from_curve() -> None:
    q4 = candidate(quant="Q4_K_M", kld=0.02)
    q6 = candidate(quant="Q6_K", kld=0.006)
    q5 = candidate(quant="Q5_K_M", kld=None)

    estimated = selection.estimate_quality(q5, [q4, q5, q6])

    assert estimated.source == "estimated-kld"
    assert estimated.kld is not None
    assert 0.006 < estimated.kld < 0.02


def test_estimate_quality_knows_non_ud_xl_quant_order() -> None:
    q5 = candidate(quant="Q5_K_XL", kld=0.0069)
    q6 = candidate(quant="Q6_K_XL", kld=0.0041)
    q8 = candidate(quant="Q8_K_XL", kld=None)

    estimated = selection.estimate_quality(q8, [q5, q6, q8])

    assert estimated.source == "estimated-kld"
    assert estimated.kld is not None
    assert estimated.kld < 0.0041


def test_kld_quality_penalises_agentic_coding_loss_strongly() -> None:
    good_candidate = candidate(quant="Q6_K", kld=0.00486)
    poor_candidate = candidate(quant="IQ4_XS", kld=0.0275)
    scored = selection.score_candidates(
        [good_candidate, poor_candidate], selection.PROFILES["agentic-coding"]
    )
    quality_by_quant = {row.candidate.quant: row.quality.score for row in scored}

    assert quality_by_quant["Q6_K"] == 1.0
    assert quality_by_quant["IQ4_XS"] == 0.0


def test_context_score_measures_headroom_above_profile_floor() -> None:
    assert selection._context_score(96_000, 96_000, 225_000) == 0.0
    assert selection._context_score(225_000, 96_000, 225_000) == 1.0
    assert 0.0 < selection._context_score(150_000, 96_000, 225_000) < 1.0


def test_proxy_quality_uses_absolute_quant_ladder_not_available_rows() -> None:
    low = candidate(quant="Q4_K_M", kld=None)
    high = candidate(quant="UD-Q4_K_XL", kld=None)

    scored = selection.score_candidates([low, high], selection.PROFILES["small-fast-tasks"])
    quality_by_quant = {row.candidate.quant: row.quality.score for row in scored}

    assert 0.0 < quality_by_quant["Q4_K_M"] < quality_by_quant["UD-Q4_K_XL"] < 1.0


def test_quant_proxy_penalises_low_quants_more_for_tiny_models() -> None:
    tiny = candidate(quant="Q4_K_M", params=752_000_000)
    larger = candidate(quant="Q4_K_M", params=8_000_000_000)

    tiny_quality = selection.estimate_quality(tiny, [tiny]).score
    larger_quality = selection.estimate_quality(larger, [larger]).score

    assert tiny_quality < larger_quality


def test_quant_proxy_penalises_low_quants_for_small_models_globally() -> None:
    low = candidate(quant="UD-Q4_K_XL", params=2_000_000_000)
    high = candidate(quant="UD-Q5_K_XL", params=2_000_000_000)

    quality_by_quant = {
        row.candidate.quant: row.quality.score
        for row in selection.score_candidates([low, high], selection.PROFILES["regular-quick-chat"])
    }

    assert quality_by_quant["UD-Q5_K_XL"] > quality_by_quant["UD-Q4_K_XL"]


def test_select_for_group_estimates_quality_from_full_group_not_filtered_subset() -> None:
    profile = selection.PROFILES["regular-quick-chat"]
    low = candidate(quant="Q3_K_M", ctx=50_000, tg=18.0, kld=0.0585)
    mid = candidate(quant="IQ4_XS", ctx=50_000, tg=17.0, kld=0.0234)
    unmeasured = candidate(quant="Q4_K_S", ctx=30_000, tg=16.0, kld=None)
    higher_measured = candidate(quant="Q4_K_M", ctx=10_000, tg=12.0, kld=0.0182)

    result = selection.select_for_group(
        [low, mid, unmeasured, higher_measured], profile, capabilities(reasoning=3, writing=3)
    )

    assert result.recommendation is not None
    assert result.recommendation.candidate != unmeasured
    best_quality_kld = result.alternatives["best-quality"].quality.kld
    higher_measured_kld = higher_measured.kld
    assert best_quality_kld is not None
    assert higher_measured_kld is not None
    assert best_quality_kld > higher_measured_kld


def test_estimate_quality_uses_same_provider_kld_curve() -> None:
    same_provider_low = candidate(provider="bartowski", quant="Q3_K_M", kld=0.0585)
    same_provider_high = candidate(provider="bartowski", quant="Q4_K_M", kld=0.0182)
    other_provider_high = candidate(provider="unsloth", quant="Q5_K_M", kld=0.0069)
    unmeasured = candidate(provider="bartowski", quant="Q4_K_S", kld=None)

    estimated = selection.estimate_quality(
        unmeasured, [same_provider_low, same_provider_high, other_provider_high, unmeasured]
    )

    assert estimated.source == "estimated-kld"
    same_provider_high_kld = same_provider_high.kld
    assert estimated.kld is not None
    assert same_provider_high_kld is not None
    assert estimated.kld > same_provider_high_kld


def test_estimate_quality_falls_back_to_cross_provider_kld_average() -> None:
    exact_other_provider = candidate(provider="provider-a", quant="Q4_K_S", kld=0.021)
    lower_other_provider = candidate(provider="provider-b", quant="Q3_K_M", kld=0.0585)
    higher_other_provider = candidate(provider="provider-b", quant="Q4_K_M", kld=0.0182)
    unmeasured = candidate(provider="provider-c", quant="Q4_K_S", kld=None)

    estimated = selection.estimate_quality(
        unmeasured,
        [exact_other_provider, lower_other_provider, higher_other_provider, unmeasured],
    )

    provider_b_estimate = selection._estimate_kld_for_provider(
        "Q4_K_S", [lower_other_provider, higher_other_provider], allow_extrapolate=True
    )
    assert estimated.source == "cross-provider-kld"
    assert provider_b_estimate is not None
    assert estimated.kld is not None
    assert abs(estimated.kld - (0.021 + provider_b_estimate) / 2) < 1e-12


def test_estimate_quality_prefers_close_provider_same_quant_before_extrapolation() -> None:
    same_provider_q4 = candidate(provider="AesSedai", quant="Q4_K_M", kld=0.0122)
    same_provider_q5 = candidate(provider="AesSedai", quant="Q5_K_M", kld=0.00671)
    close_provider_q4 = candidate(provider="unsloth", quant="UD-Q4_K_M", kld=0.0125)
    close_provider_q5 = candidate(provider="unsloth", quant="UD-Q5_K_M", kld=0.00666)
    close_provider_q6 = candidate(provider="unsloth", quant="UD-Q6_K", kld=0.00526)
    unmeasured = candidate(provider="AesSedai", quant="Q6_K", kld=None)

    estimated = selection.estimate_quality(
        unmeasured,
        [
            same_provider_q4,
            same_provider_q5,
            close_provider_q4,
            close_provider_q5,
            close_provider_q6,
            unmeasured,
        ],
    )

    assert estimated.source == "close-provider-kld"
    assert estimated.kld == 0.00526


def test_select_for_group_recommends_scored_candidate() -> None:
    profile = selection.PROFILES["regular-quick-chat"]
    slow_high_quality = candidate(quant="Q5_K_M", pp=90.0, tg=10.0, kld=0.01)
    fast_lower_quality = candidate(quant="Q4_K_M", pp=100.0, tg=40.0, kld=0.05)

    result = selection.select_for_group(
        [slow_high_quality, fast_lower_quality], profile, capabilities(reasoning=3, writing=3)
    )

    assert result.recommendation is not None
    assert result.recommendation.candidate == fast_lower_quality
    assert "best-quality" in result.alternatives


def test_alternatives_are_near_tradeoffs_not_extremes() -> None:
    profile = selection.PROFILES["agentic-coding"]
    recommendation = candidate(quant="Q5_K_M", ctx=150_000, pp=970.0, tg=33.0, kld=0.009)
    faster_too_lossy = candidate(quant="IQ4_XS", ctx=150_000, pp=1200.0, tg=42.0, kld=0.0275)
    faster_nearby = candidate(quant="Q5_K_S", ctx=150_000, pp=990.0, tg=34.0, kld=0.010)
    more_context = candidate(quant="Q6_K", ctx=175_000, pp=535.0, tg=31.0, kld=0.008)

    result = selection.select_for_group(
        [recommendation, faster_too_lossy, faster_nearby, more_context], profile, capabilities()
    )

    assert result.recommendation is not None
    assert result.alternatives["faster"].candidate == faster_nearby
    assert result.alternatives["more-context"].candidate == more_context
    assert result.alternatives["fastest"].candidate == faster_too_lossy
    assert "best-quality" in result.alternatives


def test_agentic_coding_prefers_much_better_pp_when_quality_is_close() -> None:
    profile = selection.PROFILES["agentic-coding"]
    low_pp = candidate(quant="Q6_K_L", ctx=175_000, pp=375.0, tg=27.0, kld=0.0064)
    high_pp = candidate(quant="Q6_K", ctx=150_000, pp=655.0, tg=31.0, kld=0.0069)
    best_quality = candidate(quant="Q8_0", ctx=125_000, pp=570.0, tg=24.0, kld=0.0053)
    poor_quality = candidate(quant="Q4_K_M", ctx=150_000, pp=900.0, tg=30.0, kld=0.02)

    result = selection.select_for_group(
        [low_pp, high_pp, best_quality, poor_quality], profile, capabilities()
    )

    assert result.recommendation is not None
    assert result.recommendation.candidate == high_pp


def test_profile_controls_reported_alternatives() -> None:
    profile = selection.PROFILES["slow-max-intelligence-chat"]
    recommendation = candidate(quant="Q8_0", ctx=128_000, pp=400.0, tg=25.0, kld=0.005)
    fastest = candidate(quant="IQ4_XS", ctx=128_000, pp=1200.0, tg=42.0, kld=0.03)
    more_context = candidate(quant="Q6_K", ctx=175_000, pp=350.0, tg=30.0, kld=0.007)

    result = selection.select_for_group([recommendation, fastest, more_context], profile, capabilities())

    assert "fastest" not in result.alternatives
    assert "more-context" in result.alternatives


def test_data_extraction_is_text_only() -> None:
    profile = selection.PROFILES["data-extraction"]
    text = candidate(quant="Q5_K_M", ctx=50_000, pp=100.0, tg=20.0)
    vision = selection.Candidate(
        group="foo-group",
        model="Foo",
        quant="Q8_0",
        provider="unsloth",
        mode="vision",
        ctx=50_000,
        ubatch=512,
        pp_tps=1000.0,
        tg_tps=100.0,
        params=8_000_000_000,
        size_gib=4.0,
        kld=None,
    )

    result = selection.select_for_group([text, vision], profile, capabilities(reasoning=3, tools=3))

    assert result.recommendation is not None
    assert result.recommendation.candidate == text


def test_select_for_group_reports_capability_and_context_skips() -> None:
    coding_profile = selection.PROFILES["agentic-coding"]

    low_cap = selection.select_for_group([candidate()], coding_profile, capabilities(coding=3))
    assert low_cap.recommendation is None
    assert low_cap.skipped_reason == "coding 3 < 4"

    low_ctx = selection.select_for_group([candidate(ctx=32_000)], coding_profile, capabilities())
    assert low_ctx.recommendation is None
    assert low_ctx.skipped_reason is not None
    assert "max ctx is 32000" in low_ctx.skipped_reason


def test_agentic_coding_softly_penalises_configs_below_tg_target() -> None:
    profile = selection.PROFILES["agentic-coding"]
    below_target = candidate(ctx=128_000, pp=800.0, tg=29.0)
    at_target = candidate(ctx=128_000, pp=790.0, tg=30.0)

    result = selection.select_for_group([below_target, at_target], profile, capabilities())

    assert result.recommendation is not None
    assert result.recommendation.candidate == at_target


def test_vision_deep_rejects_painfully_slow_large_model_candidates() -> None:
    profile = selection.PROFILES["vision-deep"]
    slow_high_context = selection.Candidate(
        group="foo-group",
        model="Foo",
        quant="Q4_K_S",
        provider="unsloth",
        mode="vision",
        ctx=50_000,
        ubatch=2048,
        pp_tps=288.0,
        tg_tps=8.3,
        params=122_000_000_000,
        size_gib=60.0,
        kld=None,
    )
    usable_lower_context = selection.Candidate(
        group="foo-group",
        model="Foo",
        quant="Q3_K_M",
        provider="bartowski",
        mode="vision",
        ctx=30_000,
        ubatch=512,
        pp_tps=138.0,
        tg_tps=18.2,
        params=122_000_000_000,
        size_gib=60.0,
        kld=None,
    )

    result = selection.select_for_group(
        [slow_high_context, usable_lower_context], profile, capabilities(vision=5, reasoning=5)
    )

    assert result.recommendation is not None
    assert result.recommendation.candidate == usable_lower_context


def test_small_fast_tasks_requires_high_local_speed() -> None:
    profile = selection.PROFILES["small-fast-tasks"]
    slow = candidate(ctx=50_000, pp=800.0, tg=34.0)

    result = selection.select_for_group([slow], profile, capabilities(reasoning=1))

    assert result.recommendation is None
    assert result.skipped_reason is not None
    assert "pp>=2500" in result.skipped_reason
    assert "tg>=70" in result.skipped_reason


def test_vision_fast_requires_high_vision_speed() -> None:
    profile = selection.PROFILES["vision-fast"]
    slow = selection.Candidate(
        group="foo-group",
        model="Foo",
        quant="Q4_K_M",
        provider="unsloth",
        mode="vision",
        ctx=50_000,
        ubatch=2048,
        pp_tps=300.0,
        tg_tps=8.0,
        params=8_000_000_000,
        size_gib=4.0,
        kld=None,
    )

    result = selection.select_for_group([slow], profile, capabilities(vision=4, reasoning=4))

    assert result.recommendation is None
    assert result.skipped_reason is not None
    assert "pp>=500" in result.skipped_reason
    assert "tg>=70" in result.skipped_reason
