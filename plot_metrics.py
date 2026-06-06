#!/usr/bin/env python3
# pyright: reportUnknownMemberType=false
# All matplotlib method stubs use **kwargs: Unknown, which triggers this on every
# ax.scatter / ax.legend / fig.savefig etc. call. Not actionable without typed wrappers.
"""Plot benchmark trade-offs for quantized models."""

import argparse
import csv
import math
import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.collections import PathCollection
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from llama_bench.model_identity import canonical_result_model
from llama_bench.quant_order import quant_sort_key
from llama_bench.results import (
    BENCH_PP,
    BENCH_TG,
    PP_COL,
    PP_STDDEV_COL,
    RESULTS_FILE,
    TG_COL,
    TG_STDDEV_COL,
    model_groups,
    parse_ctx,
)


PlotKind = Literal[
    "quality-tradeoffs",
    "quality-tradeoffs-compact",
    "ctx-vs-speed",
    "speed-map",
]
Mode = Literal["text", "vision"]
PlotStyle = Literal["analysis", "blog"]
CtxSpeedPanels = Literal["all", "pp-tg"]


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KLD_FILE = os.path.join(SCRIPT_DIR, "kld-results.csv")
DEFAULT_PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")

PROVIDER_STYLES = {
    "unsloth": "#2166AC",
    "bartowski": "#B2182B",
    "AesSedai": "#1B7837",
    "byteshape": "#984EA3",
    "mudler": "#FF7F00",
    "ggml-org": "#E41A1C",
}
MODE_MARKERS = {"text": "o", "vision": "^"}
DEFAULT_COLOR = "#888888"
POINT_ALPHA = 0.8
KLD_POINT_ALPHA = 0.75
MISSING_KLD_ALPHA = 0.65
QUALITY_TRADEOFFS_FIGSIZE = (36, 20)
CTX_VS_SPEED_FIGSIZE = (32, 12)
SPEED_MAP_FIGSIZE = (22, 16)

plot_style: PlotStyle = "analysis"
ctx_speed_panels: CtxSpeedPanels = "all"

DISPLAY_NAMES = {
    "gemma-4-26B-A4B": "Gemma 4 26B (A4B)",
    "Qwen3.6-35B-A3B": "Qwen 3.6 35B (A3B)",
}


@dataclass(frozen=True)
class MetricRow:
    model: str
    quant: str
    provider: str
    mode: Mode
    group: str
    ctx: int
    size_gib: float
    pp_tps: float
    pp_stddev_tps: float | None
    tg_tps: float
    tg_stddev_tps: float | None
    reps: int | None
    ubatch: int
    kld: float | None


@dataclass(frozen=True)
class KldRow:
    model: str
    quant: str
    provider: str
    kld: float


def load_kld(path: str) -> list[KldRow]:
    rows: list[KldRow] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            kld_str = row.get("kld", "").strip()
            if not kld_str:
                continue
            try:
                kld_val = float(kld_str)
            except ValueError:
                continue
            if kld_val <= 0:
                continue
            rows.append(
                KldRow(
                    model=row["model"] or "",
                    quant=row["quant"] or "",
                    provider=row["provider"] or "",
                    kld=kld_val,
                )
            )
    return rows


def _get_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    return float(value) if value else None


def load_metric_rows(
    results_file: str = RESULTS_FILE,
    kld_file: str = KLD_FILE,
    *,
    mode: Mode | None = None,
    ubatches: Sequence[int] | None = None,
) -> list[MetricRow]:
    ubatch_filter = {str(value) for value in ubatches} if ubatches is not None else None
    kld_by_key: dict[tuple[str, str, str], float] = {}
    if os.path.exists(kld_file):
        for row in load_kld(kld_file):
            kld_by_key[(row.model, row.quant, row.provider)] = row.kld

    groups = model_groups()
    rows: list[MetricRow] = []
    with open(results_file, newline="") as f:
        for row in csv.DictReader(f):
            normalized = {key: value or "" for key, value in row.items()}
            row_mode = normalized.get("mode")
            if row_mode not in ("text", "vision"):
                continue
            if mode is not None and row_mode != mode:
                continue
            if ubatch_filter is not None and normalized.get("ubatch") not in ubatch_filter:
                continue

            ctx = parse_ctx(normalized.get("ctx"))
            size = _get_float(normalized, "size_gib")
            pp = _get_float(normalized, PP_COL)
            tg = _get_float(normalized, TG_COL)
            if (
                ctx is None
                or size is None
                or pp is None
                or tg is None
                or not normalized.get("ubatch")
            ):
                continue

            provider = normalized.get("provider", "")
            model = canonical_result_model(normalized.get("model", ""), provider)
            key = (model, normalized.get("quant", ""), provider)
            group = groups.get(key)
            if group is None:
                continue
            rows.append(
                MetricRow(
                    model=key[0],
                    quant=key[1],
                    provider=key[2],
                    mode=row_mode,
                    group=group,
                    ctx=ctx,
                    size_gib=size,
                    pp_tps=pp,
                    pp_stddev_tps=_get_float(normalized, PP_STDDEV_COL),
                    tg_tps=tg,
                    tg_stddev_tps=_get_float(normalized, TG_STDDEV_COL),
                    reps=int(normalized["reps"]) if normalized.get("reps") else None,
                    ubatch=int(normalized["ubatch"]),
                    kld=kld_by_key.get(key),
                )
            )
    return sorted(
        rows,
        key=lambda row: (row.model, quant_sort_key(row.quant), row.provider, row.mode, row.ubatch),
    )


def filter_rows(
    rows: Iterable[MetricRow],
    *,
    models: Sequence[str] | None = None,
    groups: Sequence[str] | None = None,
    providers: Sequence[str] | None = None,
    show_text: bool = True,
    show_vision: bool = True,
    min_ctx: int | None = None,
    min_pp: float | None = None,
    min_tg: float | None = None,
) -> list[MetricRow]:
    filtered: list[MetricRow] = []
    for row in rows:
        if models is not None and row.model not in models:
            continue
        if groups is not None and row.group not in groups:
            continue
        if providers is not None and row.provider not in providers:
            continue
        if row.mode == "text" and not show_text:
            continue
        if row.mode == "vision" and not show_vision:
            continue
        if min_ctx is not None and row.ctx < min_ctx:
            continue
        if min_pp is not None and row.pp_tps < min_pp:
            continue
        if min_tg is not None and row.tg_tps < min_tg:
            continue
        filtered.append(row)
    return filtered


def _ci95(stddev: float | None, reps: int | None) -> float:
    if stddev is None or reps is None or reps < 1:
        return 0.0
    return 1.96 * stddev / (reps**0.5)


def _format_ctx_tick(value: float, _: int) -> str:
    return f"{value / 1000:.0f}k" if value >= 1000 else f"{value:.0f}"


def _format_ctx_axis(axis: Axis) -> None:
    axis.set_major_locator(mticker.MultipleLocator(25_000))
    axis.set_major_formatter(mticker.FuncFormatter(_format_ctx_tick))


def _format_kld_tick(value: float, _: int) -> str:
    if value >= 1:
        text = f"{value:.1f}"
    elif value >= 0.1:
        text = f"{value:.2f}"
    elif value >= 0.01:
        text = f"{value:.3f}"
    else:
        text = f"{value:.4f}"
    return text.rstrip("0").rstrip(".")


def _kld_ticks_from_values(values: Sequence[float], inner_count: int = 5) -> list[float]:
    unique = sorted({float(value) for value in values if value > 0})
    if not unique:
        return []
    if len(unique) <= inner_count + 2:
        return unique

    interior = unique[1:-1]
    selected = [unique[0]]
    log_min = math.log(unique[0])
    log_max = math.log(unique[-1])
    for i in range(inner_count):
        target = math.exp(log_min + (log_max - log_min) * (i + 1) / (inner_count + 1))
        nearest = min(interior, key=lambda value: abs(math.log(value) - math.log(target)))
        selected.append(nearest)
    selected.append(unique[-1])

    deduped: list[float] = []
    for tick in selected:
        if not deduped or not math.isclose(tick, deduped[-1], rel_tol=1e-9, abs_tol=1e-12):
            deduped.append(tick)
    return deduped


def _format_kld_axis(axis: Axis) -> None:
    axis.set_major_formatter(mticker.FuncFormatter(_format_kld_tick))
    axis.set_minor_formatter(mticker.NullFormatter())


def _label(row: MetricRow) -> str:
    return f"{row.quant} {row.mode} ub={row.ubatch}"


def _quant_mode_label(row: MetricRow) -> str:
    return f"{row.quant} {row.mode}"


def _provider_color(row: MetricRow) -> str:
    return PROVIDER_STYLES.get(row.provider, DEFAULT_COLOR)


def _is_blog_style() -> bool:
    return plot_style == "blog"


def _style_rc_params() -> dict[str, object]:
    params: dict[str, object] = {
        "figure.facecolor": "#ffffff",
        "savefig.facecolor": "#ffffff",
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Figtree",
            "Inter",
            "Aptos",
            "Arial",
            "Helvetica",
            "Liberation Sans",
            "DejaVu Sans",
        ],
        "axes.labelcolor": "#1f2937",
        "text.color": "#111827",
        "xtick.color": "#4b5563",
        "ytick.color": "#4b5563",
    }
    if _is_blog_style():
        params["axes.facecolor"] = "#fcfcff"
    return params


def _should_draw_error_bars() -> bool:
    return not _is_blog_style()


def _point_size() -> float:
    return 75.0 if _is_blog_style() else 60.0


def _kld_point_size() -> float:
    return 82.0 if _is_blog_style() else 70.0


def _annotation_fontsize() -> int:
    return 7 if _is_blog_style() else 6


def _label_fn(rows: Sequence[MetricRow]) -> Callable[[MetricRow], str]:
    same_mode = len({row.mode for row in rows}) == 1
    same_ubatch = len({row.ubatch for row in rows}) == 1
    if same_mode and same_ubatch:
        return lambda row: row.quant
    if same_mode:
        return lambda row: f"{row.quant} ub={row.ubatch}"
    if same_ubatch:
        return lambda row: f"{row.quant} {row.mode}"
    return _label


def _combined_speed_values(rows: Sequence[MetricRow]) -> dict[MetricRow, float]:
    max_pp = max((row.pp_tps for row in rows), default=0.0)
    max_tg = max((row.tg_tps for row in rows), default=0.0)
    if max_pp <= 0.0 or max_tg <= 0.0:
        return {row: 0.0 for row in rows}
    values: dict[MetricRow, float] = {}
    for row in rows:
        pp_norm = row.pp_tps / max_pp
        tg_norm = row.tg_tps / max_tg
        if pp_norm <= 0.0 or tg_norm <= 0.0:
            values[row] = 0.0
        else:
            values[row] = 2.0 / ((1.0 / pp_norm) + (1.0 / tg_norm))
    return values


def _annotate_all(
    ax: Axes,
    rows: Iterable[MetricRow],
    x: Callable[[MetricRow], float],
    y: Callable[[MetricRow], float],
    label: Callable[[MetricRow], str] = _label,
) -> None:
    x_min, x_max = ax.get_xlim()
    x_span = max(x_max - x_min, 1e-9)
    for row in rows:
        x_pos = x(row)
        rel_x = (x_pos - x_min) / x_span
        if rel_x >= 0.82:
            xytext = (-4, 4)
            ha = "right"
        else:
            xytext = (4, 4)
            ha = "left"
        ax.annotate(
            label(row),
            (x_pos, y(row)),
            textcoords="offset points",
            xytext=xytext,
            ha=ha,
            fontsize=_annotation_fontsize(),
            alpha=POINT_ALPHA,
            clip_on=True,
        )


def _annotate(
    ax: Axes,
    rows: Sequence[MetricRow],
    x: Callable[[MetricRow], float],
    y: Callable[[MetricRow], float],
    *,
    label: Callable[[MetricRow], str],
) -> None:
    _annotate_all(ax, rows, x, y, label)


def _finish_axes(ax: Axes) -> None:
    ax.grid(True, alpha=0.3, which="major")
    ax.grid(True, alpha=0.15, which="minor")
    ax.tick_params(axis="both", labelsize=9 if _is_blog_style() else 8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)


def _provider_legend_handles(rows: Sequence[MetricRow]) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor=PROVIDER_STYLES.get(provider, DEFAULT_COLOR),
            markeredgewidth=1.5,
            markersize=7,
            label=provider,
        )
        for provider in sorted({row.provider for row in rows})
    ]


def _add_provider_edge_legend(ax: Axes, rows: Sequence[MetricRow]) -> None:
    handles = _provider_legend_handles(rows)
    legend = ax.legend(
        handles=handles,
        title="Provider",
        fontsize=8 if _is_blog_style() else 7,
        title_fontsize=9 if _is_blog_style() else 8,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=False,
    )
    ax.add_artist(legend)


def _add_provider_figure_legend(fig: Figure, rows: Sequence[MetricRow]) -> None:
    handles = _provider_legend_handles(rows)
    fig.legend(
        handles=handles,
        title="Provider",
        fontsize=8 if _is_blog_style() else 7,
        title_fontsize=9 if _is_blog_style() else 8,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=max(1, len(handles)),
        frameon=False,
    )


def _add_kld_colorbar(fig: Figure, ax: Axes, scatter: PathCollection) -> None:
    array = scatter.get_array()
    values = [float(value) for value in array.tolist()] if array is not None else []
    ticks = _kld_ticks_from_values(values)
    colorbar = fig.colorbar(
        scatter,
        ax=ax,
        label="KLD (log scale, lower is better)",
        pad=0.03,
        fraction=0.05,
        ticks=ticks,
    )
    colorbar.formatter = mticker.FuncFormatter(_format_kld_tick)
    colorbar.update_ticks()
    colorbar.ax.minorticks_off()
    colorbar.ax.yaxis.get_offset_text().set_visible(False)


def _plot_provider_mode_points(
    ax: Axes,
    rows: Sequence[MetricRow],
    x: Callable[[MetricRow], float],
    y: Callable[[MetricRow], float],
) -> None:
    for provider in sorted({row.provider for row in rows}):
        for mode in ("text", "vision"):
            subset = [row for row in rows if row.provider == provider and row.mode == mode]
            if not subset:
                continue
            ax.scatter(
                [x(row) for row in subset],
                [y(row) for row in subset],
                s=_point_size(),
                c=_provider_color(subset[0]),
                marker=MODE_MARKERS[mode],
                alpha=POINT_ALPHA,
                edgecolors="none",
                linewidths=0.0,
                label=f"{provider} ({mode})",
                zorder=3,
            )


def _dedupe_kld_size_rows(rows: Sequence[MetricRow]) -> list[MetricRow]:
    by_key: dict[tuple[str, str, str, Mode, str, float, float | None], MetricRow] = {}
    for row in rows:
        key = (row.model, row.quant, row.provider, row.mode, row.group, row.size_gib, row.kld)
        current = by_key.get(key)
        if current is None or row.ubatch < current.ubatch:
            by_key[key] = row
    return sorted(
        by_key.values(),
        key=lambda row: (row.model, quant_sort_key(row.quant), row.provider, row.mode),
    )


def plot_quality_tradeoffs(model: str, rows: Sequence[MetricRow], out_dir: str) -> str | None:
    with matplotlib.rc_context(_style_rc_params()):
        kld_rows = [row for row in rows if row.kld is not None]
        if not kld_rows:
            return None
        fig, axes = plt.subplots(2, 3, figsize=QUALITY_TRADEOFFS_FIGSIZE)
        combined_speed = _combined_speed_values(rows)
        label_fn = _label_fn(kld_rows)
        specs: list[tuple[str, Callable[[MetricRow], float], str, bool]] = [
            ("Context Size (tokens)", lambda row: float(row.ctx), "ctx", False),
            ("Model Size (GiB)", lambda row: row.size_gib, "size", False),
            ("", lambda _row: 0.0, "empty", False),
            (f"Prompt Processing Speed (pp{BENCH_PP}, t/s)", lambda row: row.pp_tps, "pp", True),
            (f"Generation Speed (tg{BENCH_TG}, t/s)", lambda row: row.tg_tps, "tg", True),
            ("Combined Speed (harmonic, 0-1)", lambda row: combined_speed[row], "combined", False),
        ]
        flat_axes = list(axes.flat)
        for ax, (xlabel, x_value, field, has_error) in zip(flat_axes, specs):
            if field == "empty":
                ax.axis("off")
                continue
            plot_rows = _dedupe_kld_size_rows(kld_rows) if field == "size" else kld_rows
            _plot_provider_mode_points(
                ax,
                plot_rows,
                x_value,
                lambda row: row.kld or 0.0,
            )
            if has_error and _should_draw_error_bars():
                for row in plot_rows:
                    stddev = row.pp_stddev_tps if field == "pp" else row.tg_stddev_tps
                    ax.errorbar(
                        x_value(row),
                        row.kld or 0.0,
                        xerr=_ci95(stddev, row.reps),
                        fmt="none",
                        ecolor=_provider_color(row),
                        alpha=0.35,
                    )
            point_label = _quant_mode_label if field == "size" else label_fn
            _annotate(
                ax,
                plot_rows,
                x_value,
                lambda row: row.kld or 0.0,
                label=point_label,
            )
            ax.set_xlabel(xlabel)
            ax.set_ylabel("KLD")
            ax.set_yscale("log")
            _format_kld_axis(ax.yaxis)
            if field == "ctx":
                _format_ctx_axis(ax.xaxis)
            _finish_axes(ax)
            ax.legend(fontsize=8 if _is_blog_style() else 7, loc="best")
        return _save(fig, model, "quality-tradeoffs", out_dir)


def plot_quality_tradeoffs_compact(
    model: str, rows: Sequence[MetricRow], out_dir: str
) -> str | None:
    with matplotlib.rc_context(_style_rc_params()):
        kld_rows = [row for row in rows if row.kld is not None]
        if not kld_rows:
            return None
        fig, axes = plt.subplots(1, 3, figsize=(28, 10))
        label_fn = _label_fn(kld_rows)
        specs: list[tuple[Axes, str, Callable[[MetricRow], float], str, bool]] = [
            (axes[0], "Context Size (tokens)", lambda row: float(row.ctx), "ctx", False),
            (
                axes[1],
                f"Prompt Processing Speed (pp{BENCH_PP}, t/s)",
                lambda row: row.pp_tps,
                "pp",
                True,
            ),
            (axes[2], f"Generation Speed (tg{BENCH_TG}, t/s)", lambda row: row.tg_tps, "tg", True),
        ]
        for ax, xlabel, x_value, field, has_error in specs:
            _plot_provider_mode_points(
                ax,
                kld_rows,
                x_value,
                lambda row: row.kld or 0.0,
            )
            if has_error and _should_draw_error_bars():
                for row in kld_rows:
                    stddev = row.pp_stddev_tps if field == "pp" else row.tg_stddev_tps
                    ax.errorbar(
                        x_value(row),
                        row.kld or 0.0,
                        xerr=_ci95(stddev, row.reps),
                        fmt="none",
                        ecolor=_provider_color(row),
                        alpha=0.35,
                    )
            _annotate(
                ax,
                kld_rows,
                x_value,
                lambda row: row.kld or 0.0,
                label=label_fn,
            )
            ax.set_xlabel(xlabel)
            ax.set_ylabel("KLD")
            ax.set_yscale("log")
            _format_kld_axis(ax.yaxis)
            if field == "ctx":
                _format_ctx_axis(ax.xaxis)
            _finish_axes(ax)
        _add_provider_figure_legend(fig, kld_rows)
        return _save(fig, model, "quality-tradeoffs-compact", out_dir)


def plot_ctx_vs_speed(model: str, rows: Sequence[MetricRow], out_dir: str) -> str:
    with matplotlib.rc_context(_style_rc_params()):
        combined_speed = _combined_speed_values(rows)
        panel_specs: list[
            tuple[str, Callable[[MetricRow], float], Callable[[MetricRow], float | None], str]
        ] = [
            (
                f"Prompt Processing Speed (pp{BENCH_PP}, t/s)",
                lambda row: row.pp_tps,
                lambda row: row.pp_stddev_tps,
                "pp",
            ),
            (
                f"Generation Speed (tg{BENCH_TG}, t/s)",
                lambda row: row.tg_tps,
                lambda row: row.tg_stddev_tps,
                "tg",
            ),
            (
                "Combined Speed (harmonic, 0-1)",
                lambda row: combined_speed[row],
                lambda _row: None,
                "combined",
            ),
        ]
        if ctx_speed_panels == "pp-tg":
            panel_specs = [spec for spec in panel_specs if spec[3] != "combined"]
        fig, axes = plt.subplots(
            1,
            len(panel_specs),
            figsize=(22, 12) if len(panel_specs) == 2 else CTX_VS_SPEED_FIGSIZE,
        )
        axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]
        label_fn = _label_fn(rows)
        colorbar_scatter: PathCollection | None = None
        for ax, (ylabel, speed, stddev, _field) in zip(axes_list, panel_specs):
            scatter = _plot_kld_colored_scatter(ax, rows, lambda row: float(row.ctx), speed)
            if scatter is None:
                _plot_provider_mode_points(ax, rows, lambda row: float(row.ctx), speed)
            else:
                colorbar_scatter = scatter
            if _should_draw_error_bars():
                for row in rows:
                    ax.errorbar(
                        row.ctx,
                        speed(row),
                        yerr=_ci95(stddev(row), row.reps),
                        fmt="none",
                        ecolor=_provider_color(row),
                        alpha=0.35,
                    )
            _annotate(ax, rows, lambda row: float(row.ctx), speed, label=label_fn)
            ax.set_xlabel("Context Size (tokens)")
            ax.set_ylabel(ylabel)
            _format_ctx_axis(ax.xaxis)
            _finish_axes(ax)
        if colorbar_scatter is not None:
            _add_kld_colorbar(fig, axes_list[-1], colorbar_scatter)
            _add_provider_figure_legend(fig, rows)
        else:
            _add_provider_figure_legend(fig, rows)
        return _save(fig, model, "ctx-vs-speed", out_dir)


def _plot_kld_colored_scatter(
    ax: Axes,
    rows: Sequence[MetricRow],
    x_fn: Callable[[MetricRow], float],
    y_fn: Callable[[MetricRow], float],
) -> PathCollection | None:
    kld_rows = [row for row in rows if row.kld is not None]
    if not kld_rows:
        return None
    klds = [row.kld or 0.0 for row in kld_rows]
    norm = LogNorm(vmin=max(min(klds), 1e-6), vmax=max(klds))
    last_scatter: PathCollection | None = None
    for mode in ("text", "vision"):
        subset = [row for row in kld_rows if row.mode == mode]
        if not subset:
            continue
        mode_klds = [row.kld or 0.0 for row in subset]
        scatter = ax.scatter(
            [x_fn(row) for row in subset],
            [y_fn(row) for row in subset],
            s=_kld_point_size(),
            c=mode_klds,
            cmap="RdYlGn_r",
            norm=norm,
            marker=MODE_MARKERS[mode],
            alpha=KLD_POINT_ALPHA,
            edgecolors=[_provider_color(row) for row in subset],
            linewidths=1.2,
            zorder=3,
        )
        last_scatter = scatter
    missing_kld_rows = [row for row in rows if row.kld is None]
    for provider in sorted({row.provider for row in missing_kld_rows}):
        for mode in ("text", "vision"):
            subset = [
                row for row in missing_kld_rows if row.provider == provider and row.mode == mode
            ]
            if not subset:
                continue
            ax.scatter(
                [x_fn(row) for row in subset],
                [y_fn(row) for row in subset],
                s=_kld_point_size(),
                c="#eeeeee",
                marker=MODE_MARKERS[mode],
                alpha=MISSING_KLD_ALPHA,
                edgecolors=_provider_color(subset[0]),
                linewidths=1.2,
                zorder=2.5,
            )
    return last_scatter


def plot_speed_map(model: str, rows: Sequence[MetricRow], out_dir: str) -> str:
    with matplotlib.rc_context(_style_rc_params()):
        fig, ax = plt.subplots(1, 1, figsize=SPEED_MAP_FIGSIZE)
        scatter = _plot_kld_colored_scatter(
            ax,
            rows,
            lambda row: row.pp_tps,
            lambda row: row.tg_tps,
        )
        if scatter is not None:
            _add_kld_colorbar(fig, ax, scatter)
            _add_provider_edge_legend(ax, rows)
        else:
            _plot_provider_mode_points(
                ax,
                rows,
                lambda row: row.pp_tps,
                lambda row: row.tg_tps,
            )
            legend = ax.legend(fontsize=7, loc="best")
            ax.add_artist(legend)
        _annotate(ax, rows, lambda row: row.pp_tps, lambda row: row.tg_tps, label=_label_fn(rows))
        ax.set_xlabel(f"Prompt Processing Speed (pp{BENCH_PP}, t/s)")
        ax.set_ylabel(f"Generation Speed (tg{BENCH_TG}, t/s)")
        _finish_axes(ax)
        return _save(fig, model, "speed-map", out_dir)


def _save(fig: Figure, model: str, suffix: str, out_dir: str) -> str:
    display = DISPLAY_NAMES.get(model, model)
    fig.suptitle(
        f"{display} — {suffix.replace('-', ' ').title()}",
        fontsize=16 if _is_blog_style() else 14,
        fontweight="bold",
    )
    if _is_blog_style():
        fig.subplots_adjust(top=0.88, right=0.92, bottom=0.09, wspace=0.10)
    else:
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    os.makedirs(out_dir, exist_ok=True)
    file_suffix = f"{suffix}-{plot_style}" if _is_blog_style() else suffix
    out_path = os.path.join(out_dir, f"{model}-{file_suffix}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


PLOTTERS: dict[PlotKind, Callable[[str, Sequence[MetricRow], str], str | None]] = {
    "quality-tradeoffs": plot_quality_tradeoffs,
    "quality-tradeoffs-compact": plot_quality_tradeoffs_compact,
    "ctx-vs-speed": plot_ctx_vs_speed,
    "speed-map": plot_speed_map,
}


def _selected_plots(value: str) -> list[PlotKind]:
    if value == "all":
        return list(PLOTTERS)
    if value not in PLOTTERS:
        raise argparse.ArgumentTypeError(f"unknown plot kind: {value}")
    return [value]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot benchmark metric trade-offs.")
    parser.add_argument(
        "--plot",
        type=_selected_plots,
        default=list(PLOTTERS),
        help="Plot kind: quality-tradeoffs, ctx-vs-speed, speed-map, or all",
    )
    parser.add_argument(
        "--model", action="append", help="Only plot this benchmark model display name; repeatable"
    )
    parser.add_argument(
        "-g",
        "--group",
        action="append",
        help="Only plot models in this models.toml group; repeatable",
    )
    parser.add_argument(
        "-p", "--provider", action="append", help="Only plot models from this provider; repeatable"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["text", "vision"],
        default=None,
        help="Filter CSV rows by mode",
    )
    parser.add_argument(
        "--ubatch", type=int, action="append", help="Filter by ubatch size; repeatable"
    )
    parser.add_argument(
        "--style",
        choices=["analysis", "blog"],
        default="analysis",
        help="Presentation style for generated plots",
    )
    parser.add_argument(
        "--ctx-speed-panels",
        choices=["all", "pp-tg"],
        default="all",
        help="Panels to include in ctx-vs-speed plots",
    )
    parser.add_argument(
        "--out-dir", default=DEFAULT_PLOTS_DIR, help="Root output directory for generated plots"
    )
    parser.add_argument("--no-text", action="store_true", help="Hide text-mode data")
    parser.add_argument("--no-vision", action="store_true", help="Hide vision-mode data")
    parser.add_argument("--min-ctx", type=int, default=None, help="Minimum context size (tokens)")
    parser.add_argument(
        "--min-pp", type=float, default=None, help="Minimum prompt processing speed (t/s)"
    )
    parser.add_argument(
        "--min-tg", type=float, default=None, help="Minimum generation speed (t/s)"
    )
    args = parser.parse_args()

    if not os.path.exists(RESULTS_FILE):
        print(f"Bench results file not found: {RESULTS_FILE}")
        return

    global plot_style, ctx_speed_panels
    plot_style = args.style
    ctx_speed_panels = args.ctx_speed_panels

    rows = load_metric_rows(mode=args.mode, ubatches=args.ubatch)
    rows = filter_rows(
        rows,
        models=args.model,
        groups=args.group,
        providers=args.provider,
        show_text=not args.no_text,
        show_vision=not args.no_vision,
        min_ctx=args.min_ctx,
        min_pp=args.min_pp,
        min_tg=args.min_tg,
    )

    by_model: dict[str, list[MetricRow]] = {}
    for row in rows:
        by_model.setdefault(row.model, []).append(row)

    if not by_model:
        print("No matching bench data")
        return

    for model, model_rows in by_model.items():
        group = model_rows[0].group
        out_dir = os.path.join(args.out_dir, group)
        for plot_kind in args.plot:
            out_path = PLOTTERS[plot_kind](model, model_rows, out_dir)
            if out_path is None:
                print(f"No KLD data for {model}; skipped {plot_kind}")
                continue
            print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
