# pyright: reportPrivateUsage=false
import csv
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

import pytest
from pytest import CaptureFixture, MonkeyPatch

import plot_kld
import llama_bench.results as results
from llama_bench.schema_types import ResultRow


class AxisFake(plot_kld.AxisProtocol):
    def __init__(self) -> None:
        self.major_locator: object | None = None
        self.minor_locator: object | None = None
        self.major_formatter: object | None = None
        self.minor_formatter: object | None = None

    def set_major_locator(self, locator: object) -> None:
        self.major_locator = locator

    def set_minor_locator(self, locator: object) -> None:
        self.minor_locator = locator

    def set_major_formatter(self, formatter: object) -> None:
        self.major_formatter = formatter

    def set_minor_formatter(self, formatter: object) -> None:
        self.minor_formatter = formatter


class ErrorbarCall(NamedTuple):
    x: list[int | float]
    y: list[float]
    xerr: list[float]
    fmt: str
    ecolor: str
    elinewidth: float
    capsize: int
    alpha: float
    zorder: float


class ScatterCall(NamedTuple):
    x: list[int | float]
    y: list[float]
    color: str
    marker: str
    alpha: float
    zorder: int
    label: str


class AnnotateCall(NamedTuple):
    text: str
    xy: tuple[int | float, float]
    textcoords: str
    xytext: tuple[int, int]
    fontsize: int
    ha: str
    color: str
    alpha: float


class AxesFake(plot_kld.AxesProtocol):
    def __init__(self) -> None:
        self.xaxis_fake = AxisFake()
        self.yaxis_fake = AxisFake()
        self.xaxis = self.xaxis_fake
        self.yaxis = self.yaxis_fake
        self.errorbars: list[ErrorbarCall] = []
        self.scatters: list[ScatterCall] = []
        self.annotations: list[AnnotateCall] = []
        self.xlabels: list[tuple[str, int]] = []
        self.ylabels: list[tuple[str, int]] = []
        self.yscales: list[str] = []
        self.tick_param_calls: list[tuple[str, str, int, int]] = []
        self.grid_calls: list[tuple[bool, float, str]] = []
        self.invert_calls = 0
        self.legend_calls: list[tuple[int, str]] = []

    def errorbar(
        self,
        x: Iterable[int | float],
        y: Iterable[float],
        *,
        xerr: Iterable[float],
        fmt: str,
        ecolor: str,
        elinewidth: float,
        capsize: int,
        alpha: float,
        zorder: float,
    ) -> object:
        self.errorbars.append(
            ErrorbarCall(list(x), list(y), list(xerr), fmt, ecolor, elinewidth, capsize, alpha, zorder)
        )
        return object()

    def scatter(
        self,
        x: Iterable[int | float],
        y: Iterable[float],
        *,
        color: str,
        marker: str,
        alpha: float,
        zorder: int,
        label: str,
    ) -> object:
        self.scatters.append(ScatterCall(list(x), list(y), color, marker, alpha, zorder, label))
        return object()

    def annotate(
        self,
        text: str,
        xy: tuple[int | float, float],
        *,
        textcoords: str,
        xytext: tuple[int, int],
        fontsize: int,
        ha: str,
        color: str,
        alpha: float,
    ) -> object:
        self.annotations.append(AnnotateCall(text, xy, textcoords, xytext, fontsize, ha, color, alpha))
        return object()

    def set_xlabel(self, xlabel: str, *, fontsize: int) -> None:
        self.xlabels.append((xlabel, fontsize))

    def set_ylabel(self, ylabel: str, *, fontsize: int) -> None:
        self.ylabels.append((ylabel, fontsize))

    def set_yscale(self, value: str) -> None:
        self.yscales.append(value)

    def tick_params(self, *, axis: str, which: str, labelsize: int, length: int) -> None:
        self.tick_param_calls.append((axis, which, labelsize, length))

    def grid(self, visible: bool, *, alpha: float, which: str) -> None:
        self.grid_calls.append((visible, alpha, which))

    def invert_xaxis(self) -> None:
        self.invert_calls += 1

    def legend(self, *, fontsize: int, loc: str) -> object:
        self.legend_calls.append((fontsize, loc))
        return object()


class AxesGridFake:
    def __init__(self) -> None:
        self.axes = [AxesFake(), AxesFake(), AxesFake(), AxesFake()]

    @property
    def flat(self) -> Iterable[AxesFake]:
        return self.axes


class FigureFake:
    def __init__(self) -> None:
        self.titles: list[tuple[str, int, str]] = []
        self.texts: list[tuple[float, float, str, str, int, str, str]] = []
        self.tight_layout_rects: list[tuple[float, float, float, float]] = []
        self.saved: list[tuple[str, int]] = []

    def suptitle(self, title: str, *, fontsize: int, fontweight: str) -> object:
        self.titles.append((title, fontsize, fontweight))
        return object()

    def text(
        self,
        x: float,
        y: float,
        s: str,
        *,
        ha: str,
        fontsize: int,
        fontstyle: str,
        color: str,
    ) -> object:
        self.texts.append((x, y, s, ha, fontsize, fontstyle, color))
        return object()

    def tight_layout(self, *, rect: tuple[float, float, float, float]) -> None:
        self.tight_layout_rects.append(rect)

    def savefig(self, fname: str, *, dpi: int) -> None:
        self.saved.append((fname, dpi))


def make_plot_row(
    quant: str = "Q4_K_M",
    provider: str = "unsloth",
    mode: str = "text",
    kld: float = 0.1,
    ctx: int = 8192,
    size_gib: float = 3.5,
    pp4096_tps: float = 10.0,
    pp4096_stddev_tps: float | None = 1.0,
    tg128_tps: float = 20.0,
    tg128_stddev_tps: float | None = 2.0,
    reps: int | None = 4,
    ubatch: int = 512,
) -> plot_kld.PlotRow:
    return {
        "quant": quant,
        "provider": provider,
        "kld": kld,
        "ctx": ctx,
        "size_gib": size_gib,
        "pp4096_tps": pp4096_tps,
        "pp4096_stddev_tps": pp4096_stddev_tps,
        "tg128_tps": tg128_tps,
        "tg128_stddev_tps": tg128_stddev_tps,
        "reps": reps,
        "mode": mode,
        "ubatch": ubatch,
    }


def install_plot_fakes(monkeypatch: MonkeyPatch) -> tuple[FigureFake, AxesGridFake, list[object]]:
    fig = FigureFake()
    axes_grid = AxesGridFake()
    closed: list[object] = []

    def fake_subplots(nrows: int, ncols: int, *, figsize: tuple[int, int]) -> tuple[FigureFake, AxesGridFake]:
        assert (nrows, ncols, figsize) == (2, 2, (14, 10))
        return fig, axes_grid

    def fake_close(closed_fig: object) -> None:
        closed.append(closed_fig)

    monkeypatch.setattr(plot_kld.plt, "subplots", fake_subplots)
    monkeypatch.setattr(plot_kld.plt, "close", fake_close)
    return fig, axes_grid, closed


def test_load_bench_filters_mode_and_ubatch(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    bench_file = tmp_path / "fit-bench-results.csv"
    monkeypatch.setattr(plot_kld, "RESULTS_FILE", str(bench_file))
    with bench_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "mode", "ubatch"])
        writer.writeheader()
        writer.writerow({"model": "Foo", "mode": "text", "ubatch": "512"})
        writer.writerow({"model": "Foo", "mode": "vision", "ubatch": "1024"})

    assert plot_kld.load_bench(mode="text") == [{"model": "Foo", "mode": "text", "ubatch": "512"}]
    assert plot_kld.load_bench(ubatch=1024) == [
        {"model": "Foo", "mode": "vision", "ubatch": "1024"}
    ]


def test_load_kld_parses_rows_and_defaults_blank_values(tmp_path: Path) -> None:
    kld_file = tmp_path / "kld-results.csv"
    kld_file.write_text(
        "model,quant,provider,kld\nFoo,Q4_K_M,unsloth,0.123\n,,bartowski,\n",
        encoding="utf-8",
    )

    assert plot_kld.load_kld(str(kld_file)) == [
        {"model": "Foo", "quant": "Q4_K_M", "provider": "unsloth", "kld": 0.123},
        {"model": "", "quant": "", "provider": "bartowski", "kld": 0.0},
    ]


def test_merge_kld_bench_filters_and_sorts_complete_matching_rows() -> None:
    kld_rows: list[plot_kld.KldRow] = [
        {"model": "Foo", "quant": "Q5_K_M", "provider": "unsloth", "kld": 0.2},
        {"model": "Foo", "quant": "Q4_K_M", "provider": "unsloth", "kld": 0.1},
        {"model": "Foo", "quant": "Q8_0", "provider": "unsloth", "kld": 0.3},
    ]
    bench_rows: list[ResultRow] = [
        {
            "model": "Foo",
            "quant": "Q5_K_M",
            "provider": "unsloth",
            "mode": "text",
            "ubatch": "512",
            "ctx": "8k",
            "size_gib": "4.5",
            results.PP_COL: "10.0",
            results.PP_STDDEV_COL: "1.0",
            results.TG_COL: "20.0",
            results.TG_STDDEV_COL: "2.0",
            "reps": "16",
        },
        {
            "model": "Foo",
            "quant": "Q4_K_M",
            "provider": "unsloth",
            "mode": "text",
            "ubatch": "512",
            "ctx": "16k",
            "size_gib": "3.5",
            results.PP_COL: "11.0",
            results.PP_STDDEV_COL: "",
            results.TG_COL: "21.0",
            results.TG_STDDEV_COL: "",
            "reps": "",
        },
        {
            "model": "Foo",
            "quant": "Q8_0",
            "provider": "unsloth",
            "mode": "vision",
            "ubatch": "512",
            "ctx": "16k",
            "size_gib": "7.0",
            results.PP_COL: "8.0",
            results.PP_STDDEV_COL: "1.0",
            results.TG_COL: "15.0",
            results.TG_STDDEV_COL: "2.0",
            "reps": "4",
        },
    ]

    merged = plot_kld.merge_kld_bench(kld_rows, bench_rows, "Foo", bench_ubatch=512)

    assert [row["quant"] for row in merged] == ["Q4_K_M", "Q5_K_M"]
    assert merged[0]["ctx"] == 16000
    assert merged[0]["pp4096_stddev_tps"] is None
    assert merged[0]["reps"] is None
    assert merged[0]["ubatch"] == 512
    assert merged[1]["tg128_stddev_tps"] == 2.0


def test_merge_kld_bench_keeps_separate_ubatch_rows() -> None:
    kld_rows: list[plot_kld.KldRow] = [
        {"model": "Foo", "quant": "Q4_K_M", "provider": "unsloth", "kld": 0.1}
    ]
    bench_rows: list[ResultRow] = [
        {
            "model": "Foo",
            "quant": "Q4_K_M",
            "provider": "unsloth",
            "mode": "text",
            "ubatch": ubatch,
            "ctx": ctx,
            "size_gib": "3.5",
            results.PP_COL: "11.0",
            results.PP_STDDEV_COL: "1.0",
            results.TG_COL: "21.0",
            results.TG_STDDEV_COL: "2.0",
            "reps": "4",
        }
        for ubatch, ctx in [("512", "8k"), ("1024", "16k")]
    ]

    merged = plot_kld.merge_kld_bench(kld_rows, bench_rows, "Foo")

    assert [(row["ubatch"], row["ctx"]) for row in merged] == [(512, 8000), (1024, 16000)]
    assert [(row["ubatch"], row["ctx"]) for row in plot_kld.merge_kld_bench(kld_rows, bench_rows, "Foo", bench_ubatch=1024)] == [(1024, 16000)]


def test_merge_kld_bench_skips_incomplete_rows_and_rejects_old_schema() -> None:
    kld_rows: list[plot_kld.KldRow] = [
        {"model": "Foo", "quant": "Q4_K_M", "provider": "unsloth", "kld": 0.1}
    ]
    incomplete: list[ResultRow] = [
        {
            "model": "Foo",
            "quant": "Q4_K_M",
            "provider": "unsloth",
            "mode": "text",
            "ubatch": "512",
            "ctx": "",
            "size_gib": "3.5",
            results.PP_COL: "11.0",
            results.PP_STDDEV_COL: "1.0",
            results.TG_COL: "21.0",
            results.TG_STDDEV_COL: "2.0",
            "reps": "4",
        }
    ]
    assert plot_kld.merge_kld_bench(kld_rows, incomplete, "Foo") == []

    legacy_unknown_ctx = [{**incomplete[0], "ctx": "?"}]
    with pytest.raises(ValueError, match="invalid ctx value"):
        plot_kld.merge_kld_bench(kld_rows, legacy_unknown_ctx, "Foo")

    old_schema: list[ResultRow] = [
        {
            "model": "Foo",
            "quant": "Q4_K_M",
            "provider": "unsloth",
            "mode": "text",
            "ctx": "4k",
            "size_gib": "3.5",
        }
    ]
    with pytest.raises(ValueError, match="missing required new benchmark columns"):
        plot_kld.merge_kld_bench(kld_rows, old_schema, "Foo")


def test_metric_and_format_helpers() -> None:
    row = make_plot_row(tg128_stddev_tps=None)

    assert plot_kld._metric_value(row, "ctx") == 8192
    assert plot_kld._metric_value(row, "size_gib") == 3.5
    assert plot_kld._metric_error(row, "pp4096_stddev_tps") == 1.0
    assert plot_kld._metric_error(row, "tg128_stddev_tps") is None
    assert plot_kld._ci95(2.0, 4) == 1.96
    assert plot_kld._ci95(None, 4) == 0.0
    assert plot_kld._format_log_major(0.001, 0) == "1e-03"
    assert plot_kld._format_log_minor(0.5, 0) == "0.5"
    assert plot_kld._format_ctx_tick(12000, 0) == "12k"


def test_plot_series_empty_rows_noop() -> None:
    ax = AxesFake()

    plot_kld._plot_series(ax, [], "ctx", "red", "o", "empty")

    assert ax.errorbars == []
    assert ax.scatters == []
    assert ax.annotations == []


def test_plot_series_scatter_and_annotations() -> None:
    ax = AxesFake()
    rows = [make_plot_row("Q4_K_M", kld=0.1, ctx=8192), make_plot_row("Q5_K_M", kld=0.2, ctx=4096)]

    plot_kld._plot_series(ax, rows, "ctx", "#123456", "D", "unsloth")

    assert ax.scatters == [ScatterCall([8192, 4096], [0.1, 0.2], "#123456", "D", 1.0, 3, "unsloth")]
    assert ax.annotations == [
        AnnotateCall("Q4_K_M ub=512", (8192, 0.1), "offset points", (-5, 5), 6, "right", "#123456", 1.0),
        AnnotateCall("Q5_K_M ub=512", (4096, 0.2), "offset points", (-5, 5), 6, "right", "#123456", 1.0),
    ]


def test_plot_series_ci_errorbars_with_reps() -> None:
    ax = AxesFake()
    rows = [
        make_plot_row("Q4_K_M", pp4096_tps=10.0, pp4096_stddev_tps=2.0, reps=4),
        make_plot_row("Q5_K_M", pp4096_tps=20.0, pp4096_stddev_tps=3.0, reps=9),
    ]

    plot_kld._plot_series(
        ax,
        rows,
        "pp4096_tps",
        "#123456",
        "o",
        "speed",
        err_field="pp4096_stddev_tps",
    )

    assert ax.errorbars == [
        ErrorbarCall([10.0, 20.0], [0.1, 0.1], [1.96, 1.96], "none", "#123456", 1.0, 2, 0.6, 2.5)
    ]


def test_plot_series_missing_stddev_or_reps_uses_zero_errorbars() -> None:
    ax = AxesFake()
    rows = [
        make_plot_row("Q4_K_M", pp4096_tps=10.0, pp4096_stddev_tps=None, reps=4),
        make_plot_row("Q5_K_M", pp4096_tps=20.0, pp4096_stddev_tps=2.0, reps=None),
    ]

    plot_kld._plot_series(
        ax,
        rows,
        "pp4096_tps",
        "#123456",
        "o",
        "speed",
        err_field="pp4096_stddev_tps",
    )

    assert ax.errorbars == [
        ErrorbarCall([10.0, 20.0], [0.1, 0.1], [0.0, 0.0], "none", "#123456", 1.0, 2, 0.6, 2.5)
    ]


def test_plot_model_saves_closes_and_sets_title_suffixes(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    fig, _axes_grid, closed = install_plot_fakes(monkeypatch)

    plot_kld.plot_model("Foo", [make_plot_row()], str(tmp_path), show_vision=False)

    assert fig.saved == [(str(tmp_path / "Foo-kld-vs-bench.png"), 300)]
    assert closed == [fig]
    assert fig.titles == [("KLD vs Bench Metrics — Foo (text only)", 14, "bold")]

    fig, _axes_grid, closed = install_plot_fakes(monkeypatch)

    plot_kld.plot_model("Foo", [make_plot_row(mode="vision")], str(tmp_path), show_text=False)

    assert fig.titles == [("KLD vs Bench Metrics — Foo (vision only)", 14, "bold")]
    assert closed == [fig]


def test_plot_model_groups_providers_in_known_then_default_order(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    _fig, axes_grid, _closed = install_plot_fakes(monkeypatch)
    rows = [
        make_plot_row("Q4_K_M", "zzz", ctx=1000),
        make_plot_row("Q5_K_M", "bartowski", ctx=2000),
        make_plot_row("Q6_K", "unsloth", ctx=3000),
        make_plot_row("Q8_0", "AesSedai", ctx=4000),
    ]

    plot_kld.plot_model("Foo", rows, str(tmp_path))

    first_axis = axes_grid.axes[0]

    assert [call.label for call in first_axis.scatters] == [
        "unsloth (text)",
        "bartowski (text)",
        "AesSedai (text)",
        "zzz (text)",
    ]
    assert [call.color for call in first_axis.scatters] == ["#2166AC", "#B2182B", "#1B7837", "#888888"]
    assert [call.marker for call in first_axis.scatters] == ["o", "s", "D", "o"]


def test_plot_model_configures_axes_and_context_formatter(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    fig, axes_grid, _closed = install_plot_fakes(monkeypatch)

    plot_kld.plot_model("Foo", [make_plot_row()], str(tmp_path))

    labels = [axis.xlabels[0][0] for axis in axes_grid.axes]

    assert labels == [
        "Context Size (tokens)",
        "Model Size (GiB)",
        f"Prompt Processing Speed (pp{results.BENCH_PP}, t/s)",
        f"Generation Speed (tg{results.BENCH_TG}, t/s)",
    ]
    assert [axis.ylabels for axis in axes_grid.axes] == [[("KLD", 9)]] * 4
    assert [axis.yscales for axis in axes_grid.axes] == [["log"]] * 4
    assert [axis.grid_calls for axis in axes_grid.axes] == [[(True, 0.3, "major"), (True, 0.15, "minor")]] * 4
    assert [axis.legend_calls for axis in axes_grid.axes] == [[(7, "best")]] * 4
    assert axes_grid.axes[0].xaxis_fake.major_formatter is not None
    assert [axis.invert_calls for axis in axes_grid.axes] == [0, 0, 0, 0]
    assert fig.texts == [
        (0.5, 0.01, "Error bars: 95% CI (±1.96 × σ / √n)", "center", 8, "italic", "#555555")
    ]
    assert fig.tight_layout_rects == [(0.0, 0.03, 1.0, 0.95)]


def test_plot_model_plots_vision_rows_with_same_metric_specs(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    _fig, axes_grid, _closed = install_plot_fakes(monkeypatch)
    rows = [make_plot_row(mode="vision", ctx=1234, pp4096_tps=11.0, tg128_tps=22.0)]

    plot_kld.plot_model("Foo", rows, str(tmp_path))

    assert axes_grid.axes[0].scatters == [ScatterCall([1234], [0.1], "#2166AC", "^", 0.45, 3, "unsloth (vision)")]
    assert axes_grid.axes[2].scatters == [ScatterCall([11.0], [0.1], "#2166AC", "^", 0.45, 3, "unsloth (vision)")]
    assert axes_grid.axes[3].scatters == [ScatterCall([22.0], [0.1], "#2166AC", "^", 0.45, 3, "unsloth (vision)")]
    assert axes_grid.axes[2].errorbars[0].xerr == [0.98]
    assert axes_grid.axes[3].errorbars[0].xerr == [1.96]


def test_main_reports_missing_files(monkeypatch: MonkeyPatch, tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    missing_bench = tmp_path / "missing-bench.csv"
    kld_file = tmp_path / "kld-results.csv"
    kld_file.write_text("model,quant,provider,kld\n", encoding="utf-8")
    monkeypatch.setattr(plot_kld, "RESULTS_FILE", str(missing_bench))
    monkeypatch.setattr(plot_kld, "KLD_FILE", str(kld_file))
    monkeypatch.setattr(sys, "argv", ["plot_kld.py"])

    plot_kld.main()

    assert capsys.readouterr().out == f"Bench results file not found: {missing_bench}\n"

    bench_file = tmp_path / "fit-bench-results.csv"
    missing_kld = tmp_path / "missing-kld.csv"
    bench_file.write_text("model,mode,ubatch\n", encoding="utf-8")
    monkeypatch.setattr(plot_kld, "RESULTS_FILE", str(bench_file))
    monkeypatch.setattr(plot_kld, "KLD_FILE", str(missing_kld))

    plot_kld.main()

    assert capsys.readouterr().out == f"KLD file not found: {missing_kld}\n"


def test_main_filters_provider_and_plots_matches(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    bench_file = tmp_path / "fit-bench-results.csv"
    kld_file = tmp_path / "kld-results.csv"
    monkeypatch.setattr(plot_kld, "RESULTS_FILE", str(bench_file))
    monkeypatch.setattr(plot_kld, "KLD_FILE", str(kld_file))
    monkeypatch.setattr(plot_kld, "SCRIPT_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["plot_kld.py", "--provider", "bartowski"])
    kld_file.write_text(
        "model,quant,provider,kld\nFoo,Q4_K_M,unsloth,0.1\nFoo,Q5_K_M,bartowski,0.2\n",
        encoding="utf-8",
    )
    fieldnames = [
        "model",
        "quant",
        "provider",
        "mode",
        "ubatch",
        "ctx",
        "size_gib",
        results.PP_COL,
        results.PP_STDDEV_COL,
        results.TG_COL,
        results.TG_STDDEV_COL,
        "reps",
    ]
    with bench_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "model": "Foo",
                "quant": "Q4_K_M",
                "provider": "unsloth",
                "mode": "text",
                "ubatch": "512",
                "ctx": "4k",
                "size_gib": "3.0",
                results.PP_COL: "10.0",
                results.PP_STDDEV_COL: "1.0",
                results.TG_COL: "20.0",
                results.TG_STDDEV_COL: "2.0",
                "reps": "4",
            }
        )
        writer.writerow(
            {
                "model": "Foo",
                "quant": "Q5_K_M",
                "provider": "bartowski",
                "mode": "text",
                "ubatch": "512",
                "ctx": "8k",
                "size_gib": "4.0",
                results.PP_COL: "11.0",
                results.PP_STDDEV_COL: "1.0",
                results.TG_COL: "21.0",
                results.TG_STDDEV_COL: "2.0",
                "reps": "4",
            }
        )
    plotted: list[tuple[str, list[plot_kld.PlotRow], str, bool, bool]] = []

    def fake_plot_model(
        model_name: str,
        data: list[plot_kld.PlotRow],
        out_dir: str,
        show_text: bool = True,
        show_vision: bool = True,
    ) -> None:
        plotted.append((model_name, data, out_dir, show_text, show_vision))

    monkeypatch.setattr(plot_kld, "plot_model", fake_plot_model)

    plot_kld.main()

    assert len(plotted) == 1
    assert plotted[0][0] == "Foo"
    assert [row["provider"] for row in plotted[0][1]] == ["bartowski"]
    assert plotted[0][2:] == (str(tmp_path), True, True)


def test_main_reports_no_matching_provider(monkeypatch: MonkeyPatch, tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    bench_file = tmp_path / "fit-bench-results.csv"
    kld_file = tmp_path / "kld-results.csv"
    monkeypatch.setattr(plot_kld, "RESULTS_FILE", str(bench_file))
    monkeypatch.setattr(plot_kld, "KLD_FILE", str(kld_file))
    monkeypatch.setattr(sys, "argv", ["plot_kld.py", "--provider", "missing"])
    kld_file.write_text("model,quant,provider,kld\nFoo,Q4_K_M,unsloth,0.1\n", encoding="utf-8")
    bench_file.write_text(
        "model,quant,provider,mode,ubatch,ctx,size_gib,pp4096_tps,pp4096_stddev_tps,tg128_tps,tg128_stddev_tps,reps\n"
        "Foo,Q4_K_M,unsloth,text,512,4k,3.0,10.0,1.0,20.0,2.0,4\n",
        encoding="utf-8",
    )

    plot_kld.main()

    assert capsys.readouterr().out == "No matching bench data for Foo (provider=missing)\n"


def test_main_forwards_mode_ubatch_and_reports_no_matching_data(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    bench_file = tmp_path / "fit-bench-results.csv"
    kld_file = tmp_path / "kld-results.csv"
    bench_file.write_text("model,mode,ubatch\n", encoding="utf-8")
    kld_file.write_text("model,quant,provider,kld\n", encoding="utf-8")
    monkeypatch.setattr(plot_kld, "RESULTS_FILE", str(bench_file))
    monkeypatch.setattr(plot_kld, "KLD_FILE", str(kld_file))
    monkeypatch.setattr(sys, "argv", ["plot_kld.py", "--mode", "vision", "--ubatch", "1024"])
    load_calls: list[tuple[str | None, int | None]] = []
    merge_calls: list[tuple[str, int | None]] = []

    def fake_load_bench(mode: str | None = None, ubatch: int | None = None) -> list[ResultRow]:
        load_calls.append((mode, ubatch))
        return []

    def fake_load_kld(path: str) -> list[plot_kld.KldRow]:
        assert path == str(kld_file)
        return [{"model": "Foo", "quant": "Q4_K_M", "provider": "unsloth", "kld": 0.1}]

    def fake_merge_kld_bench(
        kld_rows: list[plot_kld.KldRow],
        bench_rows: list[ResultRow],
        model_name: str,
        bench_mode: str = "text",
        bench_ubatch: int | None = None,
    ) -> list[plot_kld.PlotRow]:
        assert kld_rows == [{"model": "Foo", "quant": "Q4_K_M", "provider": "unsloth", "kld": 0.1}]
        assert bench_rows == []
        assert model_name == "Foo"
        merge_calls.append((bench_mode, bench_ubatch))
        return []

    monkeypatch.setattr(plot_kld, "load_bench", fake_load_bench)
    monkeypatch.setattr(plot_kld, "load_kld", fake_load_kld)
    monkeypatch.setattr(plot_kld, "merge_kld_bench", fake_merge_kld_bench)

    plot_kld.main()

    assert load_calls == [("vision", 1024)]
    assert merge_calls == [("vision", 1024)]
    assert capsys.readouterr().out == "No matching bench data for Foo\n"
