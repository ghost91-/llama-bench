# pyright: reportPrivateUsage=false, reportUnknownMemberType=false

import csv
import sys
from collections import Counter
from pathlib import Path

from pytest import MonkeyPatch

import plot_metrics
import llama_bench.results as results


def write_results(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "model",
        "quant",
        "provider",
        "mode",
        "size_gib",
        "ctx",
        "ubatch",
        results.PP_COL,
        results.PP_STDDEV_COL,
        results.TG_COL,
        results.TG_STDDEV_COL,
        "reps",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_result_row(
    *,
    model: str = "Foo",
    quant: str = "Q4_K_M",
    provider: str = "unsloth",
    mode: str = "text",
    ctx: str = "8k",
    ubatch: str = "512",
    pp: str = "10.0",
    tg: str = "20.0",
) -> dict[str, str]:
    return {
        "model": model,
        "quant": quant,
        "provider": provider,
        "mode": mode,
        "size_gib": "3.5",
        "ctx": ctx,
        "ubatch": ubatch,
        results.PP_COL: pp,
        results.PP_STDDEV_COL: "1.0",
        results.TG_COL: tg,
        results.TG_STDDEV_COL: "2.0",
        "reps": "4",
    }


def make_metric_row(
    *,
    quant: str = "Q4_K_M",
    provider: str = "unsloth",
    mode: plot_metrics.Mode = "text",
    ctx: int = 8000,
    pp: float = 10.0,
    tg: float = 20.0,
    kld: float | None = 0.2,
) -> plot_metrics.MetricRow:
    return plot_metrics.MetricRow(
        model="Foo",
        quant=quant,
        provider=provider,
        mode=mode,
        group="foo-group",
        ctx=ctx,
        size_gib=3.5,
        pp_tps=pp,
        pp_stddev_tps=1.0,
        tg_tps=tg,
        tg_stddev_tps=2.0,
        reps=4,
        ubatch=512,
        kld=kld,
    )


def test_load_metric_rows_joins_kld_and_model_group(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    results_file = tmp_path / "fit-bench-results.csv"
    kld_file = tmp_path / "kld-results.csv"
    write_results(
        results_file,
        [
            make_result_row(ctx="8k"),
            make_result_row(quant="Q5_K_M", mode="vision", ctx="16k", pp="11.0", tg="21.0"),
        ],
    )
    kld_file.write_text("model,quant,provider,kld\nFoo,Q4_K_M,unsloth,0.123\n", encoding="utf-8")
    monkeypatch.setattr(
        plot_metrics,
        "model_groups",
        lambda: {("Foo", "Q4_K_M", "unsloth"): "foo-group"},
    )

    rows = plot_metrics.load_metric_rows(str(results_file), str(kld_file))

    assert [(row.quant, row.mode, row.ctx, row.group, row.kld) for row in rows] == [
        ("Q4_K_M", "text", 8000, "foo-group", 0.123),
    ]


def test_load_metric_rows_skips_rows_not_in_models_toml(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    results_file = tmp_path / "fit-bench-results.csv"
    kld_file = tmp_path / "kld-results.csv"
    write_results(
        results_file,
        [
            make_result_row(model="Foo", quant="Q4_K_M", provider="unsloth"),
            make_result_row(model="Bar", quant="Q5_K_M", provider="unsloth"),
        ],
    )
    kld_file.write_text("model,quant,provider,kld\nFoo,Q4_K_M,unsloth,0.123\n", encoding="utf-8")
    monkeypatch.setattr(
        plot_metrics,
        "model_groups",
        lambda: {("Foo", "Q4_K_M", "unsloth"): "foo-group"},
    )

    rows = plot_metrics.load_metric_rows(str(results_file), str(kld_file))

    assert [row.model for row in rows] == ["Foo"]


def test_load_metric_rows_filters_mode_and_ubatch(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    results_file = tmp_path / "fit-bench-results.csv"
    kld_file = tmp_path / "missing-kld.csv"
    write_results(
        results_file,
        [
            make_result_row(mode="text", ubatch="512"),
            make_result_row(mode="vision", ubatch="1024"),
        ],
    )

    monkeypatch.setattr(
        plot_metrics,
        "model_groups",
        lambda: {("Foo", "Q4_K_M", "unsloth"): "foo-group"},
    )

    rows = plot_metrics.load_metric_rows(str(results_file), str(kld_file), mode="vision", ubatch=1024)

    assert [(row.mode, row.ubatch) for row in rows] == [("vision", 1024)]


def test_load_kld_skips_blank_and_zero_and_negative_kld(tmp_path: Path) -> None:
    kld_file = tmp_path / "kld-results.csv"
    kld_file.write_text(
        "model,quant,provider,kld\nFoo,Q4_K_M,unsloth,0.123\n"
        "Foo,Q5_K_M,unsloth,\n"
        "Foo,Q6_K,unsloth,0.0\n"
        "Foo,Q8_0,unsloth,-0.01\n"
        "Bar,Q4_K_M,unsloth,bad\n",
        encoding="utf-8",
    )

    rows = plot_metrics.load_kld(str(kld_file))

    assert [(row.model, row.quant, row.kld) for row in rows] == [
        ("Foo", "Q4_K_M", 0.123),
    ]


def test_filter_rows_applies_model_group_provider_and_mode() -> None:
    rows = [
        make_metric_row(provider="unsloth", mode="text"),
        make_metric_row(provider="bartowski", mode="vision"),
        make_metric_row(provider="unsloth", mode="vision", kld=None),
    ]

    filtered = plot_metrics.filter_rows(
        rows,
        models=["Foo"],
        groups=["foo-group"],
        providers=["unsloth"],
        show_text=True,
        show_vision=False,
    )

    assert filtered == [rows[0]]


def test_filter_rows_accepts_multiple_groups_and_providers() -> None:
    rows = [
        make_metric_row(provider="unsloth"),
        make_metric_row(provider="bartowski"),
        make_metric_row(provider="AesSedai"),
    ]

    filtered = plot_metrics.filter_rows(
        rows,
        groups=["foo-group"],
        providers=["unsloth", "bartowski"],
    )

    assert filtered == rows[:2]


def test_kld_colored_scatter_preserves_mode_markers(tmp_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PathCollection

    rows = [
        make_metric_row(provider="unsloth", mode="text", kld=0.1),
        make_metric_row(provider="bartowski", mode="vision", pp=11.0, tg=21.0, kld=0.2),
    ]
    fig, ax = plt.subplots()
    scatter = plot_metrics._plot_kld_colored_scatter(ax, rows, lambda row: row.ctx, lambda row: row.pp_tps)
    plt.close(fig)

    assert scatter is not None
    assert isinstance(scatter, PathCollection)
    kld_collections = [c for c in ax.collections if isinstance(c, PathCollection)]
    assert len(kld_collections) >= 2


def test_label_omits_provider() -> None:
    row = make_metric_row(quant="Q4_K_M", provider="unsloth", mode="vision")

    assert plot_metrics._label(row) == "Q4_K_M vision ub=512"


def test_kld_tick_formatter_uses_decimal_notation() -> None:
    assert plot_metrics._format_kld_tick(0.005, 0) == "0.005"
    assert plot_metrics._format_kld_tick(0.02, 0) == "0.02"
    assert plot_metrics._format_kld_tick(1.0, 0) == "1"
    assert plot_metrics._format_kld_tick(2.5, 0) == "2.5"


def test_ctx_vs_speed_plot_writes_png(tmp_path: Path) -> None:
    out_path = plot_metrics.plot_ctx_vs_speed("Foo", [make_metric_row()], str(tmp_path))

    assert out_path == str(tmp_path / "Foo-ctx-vs-speed.png")
    assert Path(out_path).exists()


def test_speed_map_plot_writes_png_without_kld(tmp_path: Path) -> None:
    rows = [
        make_metric_row(provider="unsloth", mode="text", kld=None),
        make_metric_row(provider="bartowski", mode="vision", pp=11.0, tg=21.0, kld=None),
    ]

    out_path = plot_metrics.plot_speed_map("Foo", rows, str(tmp_path))

    assert out_path == str(tmp_path / "Foo-speed-map.png")
    assert Path(out_path).exists()


def test_ctx_pp_kld_plot_writes_png_with_partial_kld(tmp_path: Path) -> None:
    rows = [
        make_metric_row(provider="unsloth", mode="text", kld=0.1),
        make_metric_row(provider="bartowski", mode="vision", ctx=16000, pp=11.0, kld=None),
    ]

    out_path = plot_metrics.plot_ctx_pp_kld("Foo", rows, str(tmp_path))

    assert out_path is not None
    assert out_path == str(tmp_path / "Foo-ctx-pp-kld.png")
    assert Path(out_path).exists()


def test_ctx_pp_kld_plot_skips_without_kld(tmp_path: Path) -> None:
    out_path = plot_metrics.plot_ctx_pp_kld("Foo", [make_metric_row(kld=None)], str(tmp_path))

    assert out_path is None


def test_speed_map_plot_writes_png_with_partial_kld(tmp_path: Path) -> None:
    rows = [
        make_metric_row(provider="unsloth", mode="text", kld=0.1),
        make_metric_row(provider="bartowski", mode="vision", pp=11.0, tg=21.0, kld=None),
    ]

    out_path = plot_metrics.plot_speed_map("Foo", rows, str(tmp_path))

    assert out_path == str(tmp_path / "Foo-speed-map.png")
    assert Path(out_path).exists()


def test_real_metric_rows_match_snapshot_inventory() -> None:
    rows = plot_metrics.load_metric_rows()

    assert len(rows) == 684
    assert Counter(row.mode for row in rows) == {"text": 369, "vision": 315}
    assert sum(row.kld is not None for row in rows) == 407

    by_group = Counter(row.group for row in rows)
    assert by_group["gemma-4-26b-a4b"] == 176
    assert by_group["qwen3.5-122b-a10b"] == 86
    assert by_group["qwen3.6-35b-a3b"] == 208


def test_main_plot_inventory_for_group(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[str, str, int]] = []

    def fake_plotter(model: str, rows: list[plot_metrics.MetricRow], out_dir: str) -> str:
        calls.append((model, out_dir, len(rows)))
        return str(Path(out_dir) / f"{model}-fake.png")

    monkeypatch.setattr(
        plot_metrics,
        "PLOTTERS",
        {
            "kld-vs-bench": fake_plotter,
            "ctx-vs-speed": fake_plotter,
            "ctx-pp-kld": fake_plotter,
            "speed-map": fake_plotter,
        },
    )
    monkeypatch.setattr(plot_metrics, "SCRIPT_DIR", str(tmp_path))
    monkeypatch.setattr(
        sys,
        "argv",
        ["plot_metrics.py", "--plot", "all", "--group", "qwen3.5-122b-a10b"],
    )

    plot_metrics.main()

    assert calls == [("Qwen3.5-122B-A10B", str(tmp_path), 86)] * 4
