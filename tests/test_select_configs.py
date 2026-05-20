import select_configs
import llama_bench.selection as selection


def test_render_table_includes_recommendations_and_skips() -> None:
    candidate = selection.Candidate(
        group="foo-group",
        model="Foo",
        quant="Q4_K_M",
        provider="unsloth",
        mode="text",
        ctx=128_000,
        ubatch=512,
        pp_tps=100.0,
        tg_tps=20.0,
        params=8_000_000_000,
        size_gib=4.0,
        kld=0.02,
    )
    ok = selection.ProfileSelection(
        group="foo-group",
        profile="regular-quick-chat",
        recommendation=selection.ScoredCandidate(
            candidate=candidate,
            quality=selection.Quality(score=0.75, source="measured-kld", kld=0.02),
            score=0.8,
        ),
        alternatives={},
        skipped_reason=None,
    )
    skipped = selection.ProfileSelection(
        group="bar-group",
        profile="agentic-coding",
        recommendation=None,
        alternatives={},
        skipped_reason="coding 3 < 4",
    )

    rendered = select_configs.render_table([ok, skipped])

    assert "profile" in rendered
    assert "regular-quick-chat" in rendered
    assert "foo-group" in rendered
    assert "0.75/measured-kld/0.02" in rendered
    assert "coding 3 < 4" in rendered


def test_render_markdown_skips_duplicate_alternatives() -> None:
    candidate = selection.Candidate(
        group="foo-group",
        model="Foo",
        quant="Q4_K_M",
        provider="unsloth",
        mode="text",
        ctx=128_000,
        ubatch=512,
        pp_tps=100.0,
        tg_tps=20.0,
        params=8_000_000_000,
        size_gib=4.0,
        kld=0.02,
    )
    duplicate = selection.ScoredCandidate(
        candidate=candidate,
        quality=selection.Quality(score=0.75, source="measured-kld", kld=0.02),
        score=0.8,
    )
    selection_result = selection.ProfileSelection(
        group="foo-group",
        profile="regular-quick-chat",
        recommendation=duplicate,
        alternatives={"faster": duplicate, "fastest": duplicate},
        skipped_reason=None,
    )

    rendered = select_configs.render_markdown([selection_result])

    assert rendered.count("foo-group") == 1
    assert "faster" not in rendered
    assert "fastest" not in rendered


def test_render_markdown_collapses_near_duplicate_speed_alternatives() -> None:
    recommended_candidate = selection.Candidate(
        group="foo-group",
        model="Foo",
        quant="Q4_K_M",
        provider="unsloth",
        mode="text",
        ctx=150_000,
        ubatch=1024,
        pp_tps=365.5,
        tg_tps=33.8,
        params=8_000_000_000,
        size_gib=4.0,
        kld=None,
    )
    faster_candidate = selection.Candidate(
        group="foo-group",
        model="Foo",
        quant="Q4_K_M",
        provider="bartowski",
        mode="text",
        ctx=100_000,
        ubatch=2048,
        pp_tps=559.4,
        tg_tps=33.3,
        params=8_000_000_000,
        size_gib=4.0,
        kld=None,
    )
    fastest_candidate = selection.Candidate(
        group="foo-group",
        model="Foo",
        quant="Q4_K_M",
        provider="unsloth",
        mode="text",
        ctx=100_000,
        ubatch=2048,
        pp_tps=547.7,
        tg_tps=34.1,
        params=8_000_000_000,
        size_gib=4.0,
        kld=None,
    )
    higher_quality_candidate = selection.Candidate(
        group="foo-group",
        model="Foo",
        quant="UD-Q4_K_XL",
        provider="unsloth",
        mode="text",
        ctx=100_000,
        ubatch=2048,
        pp_tps=551.7,
        tg_tps=26.3,
        params=8_000_000_000,
        size_gib=4.0,
        kld=None,
    )
    recommendation = selection.ScoredCandidate(
        candidate=recommended_candidate,
        quality=selection.Quality(score=0.69, source="quant-proxy", kld=None),
        score=0.77,
    )
    selection_result = selection.ProfileSelection(
        group="foo-group",
        profile="agentic-coding",
        recommendation=recommendation,
        alternatives={
            "faster": selection.ScoredCandidate(
                candidate=faster_candidate,
                quality=selection.Quality(score=0.69, source="quant-proxy", kld=None),
                score=0.753,
            ),
            "higher-quality": selection.ScoredCandidate(
                candidate=higher_quality_candidate,
                quality=selection.Quality(score=0.77, source="quant-proxy", kld=None),
                score=0.744,
            ),
            "fastest": selection.ScoredCandidate(
                candidate=fastest_candidate,
                quality=selection.Quality(score=0.69, source="quant-proxy", kld=None),
                score=0.749,
            ),
        },
        skipped_reason=None,
    )

    rendered = select_configs.render_markdown([selection_result])

    assert "recommended" in rendered
    assert "fastest" in rendered
    assert "faster" not in rendered
    assert "higher-quality" not in rendered
    assert "best-quality" not in rendered


def test_render_markdown_drops_redundant_q8_k_xl_quality_alternative() -> None:
    recommended_candidate = selection.Candidate(
        group="foo-group",
        model="Foo",
        quant="Q8_0",
        provider="bartowski",
        mode="text",
        ctx=128_000,
        ubatch=512,
        pp_tps=100.0,
        tg_tps=90.0,
        params=800_000_000,
        size_gib=1.0,
        kld=None,
    )
    q8_k_xl_candidate = selection.Candidate(
        group="foo-group",
        model="Foo",
        quant="UD-Q8_K_XL",
        provider="unsloth",
        mode="text",
        ctx=128_000,
        ubatch=512,
        pp_tps=98.0,
        tg_tps=80.0,
        params=800_000_000,
        size_gib=1.0,
        kld=None,
    )
    recommendation = selection.ScoredCandidate(
        candidate=recommended_candidate,
        quality=selection.Quality(score=0.92, source="quant-proxy", kld=None),
        score=0.9,
    )
    alternative = selection.ScoredCandidate(
        candidate=q8_k_xl_candidate,
        quality=selection.Quality(score=1.0, source="quant-proxy", kld=None),
        score=0.85,
    )
    selection_result = selection.ProfileSelection(
        group="foo-group",
        profile="small-fast-tasks",
        recommendation=recommendation,
        alternatives={"best-quality": alternative},
        skipped_reason=None,
    )

    rendered = select_configs.render_markdown([selection_result])

    assert "recommended" in rendered
    assert "UD-Q8_K_XL" not in rendered
    assert "best-quality" not in rendered


def test_render_reverse_markdown_groups_uses_by_configuration() -> None:
    candidate = selection.Candidate(
        group="foo-group",
        model="Foo",
        quant="Q4_K_M",
        provider="unsloth",
        mode="text",
        ctx=128_000,
        ubatch=512,
        pp_tps=100.0,
        tg_tps=20.0,
        params=8_000_000_000,
        size_gib=4.0,
        kld=0.02,
    )
    scored = selection.ScoredCandidate(
        candidate=candidate,
        quality=selection.Quality(score=0.75, source="measured-kld", kld=0.02),
        score=0.8,
    )
    selections = [
        selection.ProfileSelection(
            group="foo-group",
            profile="regular-quick-chat",
            recommendation=scored,
            alternatives={},
            skipped_reason=None,
        ),
        selection.ProfileSelection(
            group="foo-group",
            profile="data-extraction",
            recommendation=scored,
            alternatives={},
            skipped_reason=None,
        ),
    ]

    rendered = select_configs.render_reverse_markdown(selections)

    assert "| Label | Description | Group |" in rendered
    assert "foo-group-chat" in rendered
    assert "regular-quick-chat: recommended<br>data-extraction: recommended" in rendered
    assert rendered.count("| foo-group |") == 1
    assert "| 2 |" in rendered


def test_build_labelled_configs_derives_meaningful_label_and_description() -> None:
    candidate = selection.Candidate(
        group="foo-group",
        model="Foo",
        quant="Q4_K_M",
        provider="unsloth",
        mode="text",
        ctx=128_000,
        ubatch=512,
        pp_tps=100.0,
        tg_tps=20.0,
        params=8_000_000_000,
        size_gib=4.0,
        kld=0.02,
    )
    scored = selection.ScoredCandidate(
        candidate=candidate,
        quality=selection.Quality(score=0.75, source="measured-kld", kld=0.02),
        score=0.8,
    )
    selections = [
        selection.ProfileSelection(
            group="foo-group",
            profile="agentic-coding",
            recommendation=scored,
            alternatives={},
            skipped_reason=None,
        ),
        selection.ProfileSelection(
            group="foo-group",
            profile="writing-and-polish",
            recommendation=scored,
            alternatives={},
            skipped_reason=None,
        ),
    ]

    configs = select_configs.build_labelled_configs(selections)

    assert [config.label for config in configs] == ["foo-group-agentic"]
    assert configs[0].description.startswith("Agentic coding default; also Writing")


def test_build_labelled_configs_disambiguates_duplicate_labels() -> None:
    low_ctx = selection.Candidate(
        group="foo-group",
        model="Foo",
        quant="Q4_K_M",
        provider="unsloth",
        mode="text",
        ctx=128_000,
        ubatch=512,
        pp_tps=100.0,
        tg_tps=20.0,
        params=8_000_000_000,
        size_gib=4.0,
        kld=0.02,
    )
    high_ctx = selection.Candidate(
        group="foo-group",
        model="Foo",
        quant="Q5_K_M",
        provider="unsloth",
        mode="text",
        ctx=256_000,
        ubatch=512,
        pp_tps=90.0,
        tg_tps=18.0,
        params=8_000_000_000,
        size_gib=4.0,
        kld=0.01,
    )
    low_scored = selection.ScoredCandidate(
        candidate=low_ctx,
        quality=selection.Quality(score=0.75, source="measured-kld", kld=0.02),
        score=0.8,
    )
    high_scored = selection.ScoredCandidate(
        candidate=high_ctx,
        quality=selection.Quality(score=0.85, source="measured-kld", kld=0.01),
        score=0.75,
    )
    selections = [
        selection.ProfileSelection(
            group="foo-group",
            profile="agentic-coding",
            recommendation=low_scored,
            alternatives={},
            skipped_reason=None,
        ),
        selection.ProfileSelection(
            group="foo-group",
            profile="agentic-coding",
            recommendation=high_scored,
            alternatives={},
            skipped_reason=None,
        ),
    ]

    configs = select_configs.build_labelled_configs(selections)

    assert [config.label for config in configs] == ["foo-group-agentic", "foo-group-agentic-q5-k-m"]


def test_consolidation_with_custom_profile() -> None:
    from llama_bench.consolidation import satisfies_profile_constraints, ReportEntry
    from llama_bench.selection import Profile

    custom_profile = Profile(
        name="custom-test",
        min_ctx=64_000,
        modes=("text",),
        min_capabilities={},
        weights={"quality": 0.4, "ctx": 0.3, "pp": 0.15, "tg": 0.15},
        alternatives=("faster",),
    )
    custom_candidate = selection.Candidate(
        group="foo-group",
        model="Foo",
        quant="Q4_K_M",
        provider="unsloth",
        mode="text",
        ctx=128_000,
        ubatch=512,
        pp_tps=100.0,
        tg_tps=20.0,
        params=8_000_000_000,
        size_gib=4.0,
        kld=None,
    )
    custom_scored = selection.ScoredCandidate(
        candidate=custom_candidate,
        quality=selection.Quality(score=0.75, source="quant-proxy", kld=None),
        score=0.8,
    )
    custom_selection = selection.ProfileSelection(
        group="foo-group",
        profile="custom-test",
        recommendation=custom_scored,
        alternatives={},
        skipped_reason=None,
        profile_config=custom_profile,
    )
    entry = ReportEntry(custom_selection, "recommended", custom_scored)

    passing = selection.Candidate(
        group="foo-group",
        model="Foo",
        quant="Q5_K_M",
        provider="unsloth",
        mode="text",
        ctx=100_000,
        ubatch=512,
        pp_tps=90.0,
        tg_tps=18.0,
        params=8_000_000_000,
        size_gib=4.0,
        kld=None,
    )
    passing_scored = selection.ScoredCandidate(
        candidate=passing,
        quality=selection.Quality(score=0.80, source="quant-proxy", kld=None),
        score=0.78,
    )
    assert satisfies_profile_constraints(entry, passing_scored) is True

    failing_candidate = selection.Candidate(
        group="foo-group",
        model="Foo",
        quant="Q3_K_M",
        provider="unsloth",
        mode="vision",
        ctx=30_000,
        ubatch=512,
        pp_tps=50.0,
        tg_tps=10.0,
        params=8_000_000_000,
        size_gib=4.0,
        kld=None,
    )
    failing_scored = selection.ScoredCandidate(
        candidate=failing_candidate,
        quality=selection.Quality(score=0.50, source="quant-proxy", kld=None),
        score=0.5,
    )
    assert satisfies_profile_constraints(entry, failing_scored) is False


def test_consolidation_preserves_profile_hard_speed_constraints() -> None:
    chat_candidate = selection.Candidate(
        group="foo-group",
        model="Foo",
        quant="Q4_K_M",
        provider="unsloth",
        mode="text",
        ctx=128_000,
        ubatch=512,
        pp_tps=4500.0,
        tg_tps=67.5,
        params=8_000_000_000,
        size_gib=4.0,
        kld=None,
    )
    fast_candidate = selection.Candidate(
        group="foo-group",
        model="Foo",
        quant="Q4_K_S",
        provider="unsloth",
        mode="text",
        ctx=128_000,
        ubatch=512,
        pp_tps=4520.0,
        tg_tps=70.0,
        params=8_000_000_000,
        size_gib=4.0,
        kld=None,
    )
    chat_scored = selection.ScoredCandidate(
        candidate=chat_candidate,
        quality=selection.Quality(score=0.76, source="quant-proxy", kld=None),
        score=0.91,
    )
    fast_scored = selection.ScoredCandidate(
        candidate=fast_candidate,
        quality=selection.Quality(score=0.68, source="quant-proxy", kld=None),
        score=0.89,
    )
    selections = [
        selection.ProfileSelection(
            group="foo-group",
            profile="regular-quick-chat",
            recommendation=chat_scored,
            alternatives={},
            skipped_reason=None,
            profile_config=selection.PROFILES["regular-quick-chat"],
        ),
        selection.ProfileSelection(
            group="foo-group",
            profile="small-fast-tasks",
            recommendation=fast_scored,
            alternatives={},
            skipped_reason=None,
            profile_config=selection.PROFILES["small-fast-tasks"],
        ),
    ]

    configs = select_configs.build_labelled_configs(selections, consolidate=True)

    assert len(configs) == 2
    fast_config = next(config for config in configs if config.entries[0].selection.profile == "small-fast-tasks")
    assert fast_config.key[7] == 70.0
