#!/usr/bin/env python3
"""Plot KLD vs benchmark metrics for quantized models."""

import argparse
import csv
import os
from typing import Iterable, Literal, Mapping, NamedTuple, Protocol, TypeAlias, TypedDict, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.figure import Figure

from llama_bench.quant_order import QUANT_ORDER
from llama_bench.results import (
    BENCH_PP,
    BENCH_TG,
    PP_COL,
    PP_STDDEV_COL,
    RESULTS_FILE,
    TG_COL,
    TG_STDDEV_COL,
    parse_ctx,
)
from llama_bench.schema_types import ResultRow


MetricField: TypeAlias = Literal["ctx", "size_gib", "pp4096_tps", "tg128_tps"]
MetricErrorField: TypeAlias = Literal["pp4096_stddev_tps", "tg128_stddev_tps"]
class AxisProtocol(Protocol):
    def set_major_locator(self, locator: object) -> None: ...

    def set_minor_locator(self, locator: object) -> None: ...

    def set_major_formatter(self, formatter: object) -> None: ...

    def set_minor_formatter(self, formatter: object) -> None: ...


class AxesProtocol(Protocol):
    xaxis: AxisProtocol
    yaxis: AxisProtocol

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
    ) -> object: ...

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
    ) -> object: ...

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
    ) -> object: ...

    def set_xlabel(self, xlabel: str, *, fontsize: int) -> None: ...

    def set_ylabel(self, ylabel: str, *, fontsize: int) -> None: ...

    def set_yscale(self, value: str) -> None: ...

    def tick_params(self, *, axis: str, which: str, labelsize: int, length: int) -> None: ...

    def grid(self, visible: bool, *, alpha: float, which: str) -> None: ...

    def invert_xaxis(self) -> None: ...

    def legend(self, *, fontsize: int, loc: str) -> object: ...


class AxesGridProtocol(Protocol):
    @property
    def flat(self) -> Iterable[AxesProtocol]: ...


class FigureProtocol(Protocol):
    def suptitle(self, title: str, *, fontsize: int, fontweight: str) -> object: ...

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
    ) -> object: ...

    def tight_layout(self, *, rect: tuple[float, float, float, float]) -> None: ...

    def savefig(self, fname: str, *, dpi: int) -> None: ...


class SubplotsProtocol(Protocol):
    def __call__(
        self, nrows: int, ncols: int, *, figsize: tuple[int, int]
    ) -> tuple[Figure, object]: ...


class CloseProtocol(Protocol):
    def __call__(self, fig: Figure) -> None: ...


class SubplotSpec(NamedTuple):
    field: MetricField
    err: MetricErrorField | None
    xlabel: str
    xytext: tuple[int, int]
    ha: str


class KldRow(TypedDict):
    model: str
    quant: str
    provider: str
    kld: float


class PlotRow(TypedDict):
    quant: str
    provider: str
    kld: float
    ctx: int
    size_gib: float
    pp4096_tps: float
    pp4096_stddev_tps: float | None
    tg128_tps: float
    tg128_stddev_tps: float | None
    reps: int | None
    mode: str
    ubatch: int


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KLD_FILE = os.path.join(SCRIPT_DIR, "kld-results.csv")

PROVIDER_STYLES = {
    "unsloth": ("#2166AC", "o", "^"),
    "bartowski": ("#B2182B", "s", "v"),
    "AesSedai": ("#1B7837", "D", "P"),
}

DEFAULT_STYLE = ("#888888", "o", "^")

DISPLAY_NAMES = {
    "gemma-4-26B-A4B": "Gemma 4 26B (A4B)",
    "Qwen3.6-35B-A3B": "Qwen 3.6 35B (A3B)",
}


def load_bench(mode: str | None = None, ubatch: int | None = None) -> list[ResultRow]:
    rows: list[ResultRow] = []
    with open(RESULTS_FILE, newline="") as f:
        for row in csv.DictReader(f):
            normalized_row: ResultRow = {key: value or "" for key, value in row.items()}
            if mode is not None and normalized_row.get("mode") != mode:
                continue
            if ubatch is not None and normalized_row.get("ubatch") != str(ubatch):
                continue
            rows.append(normalized_row)
    return rows


def load_kld(path: str) -> list[KldRow]:
    rows: list[KldRow] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "model": row["model"] or "",
                    "quant": row["quant"] or "",
                    "provider": row["provider"] or "",
                    "kld": float(row["kld"] or 0.0),
                }
            )
    return rows


def _get_float(row: Mapping[str, str], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key, "")
        if value != "":
            return float(value)
    return None


def _require_metric_columns(row: Mapping[str, str]) -> None:
    required = [PP_COL, PP_STDDEV_COL, TG_COL, TG_STDDEV_COL]
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(
            "fit-bench-results.csv is missing required new benchmark columns: "
            + ", ".join(missing)
        )


def _metric_value(row: PlotRow, field: MetricField) -> int | float:
    if field == "ctx":
        return row["ctx"]
    if field == "size_gib":
        return row["size_gib"]
    if field == "pp4096_tps":
        return row["pp4096_tps"]
    return row["tg128_tps"]


def _metric_error(row: PlotRow, field: MetricErrorField) -> float | None:
    if field == "pp4096_stddev_tps":
        return row["pp4096_stddev_tps"]
    return row["tg128_stddev_tps"]


def merge_kld_bench(
    kld_rows: list[KldRow],
    bench_rows: list[ResultRow],
    model_name: str,
    bench_mode: str = "text",
    bench_ubatch: int | None = None,
) -> list[PlotRow]:
    merged: list[PlotRow] = []
    kld_by_key = {(row["quant"], row["provider"]): row for row in kld_rows}

    for bench_row in bench_rows:
        if bench_row["model"] != model_name:
            continue
        if bench_row.get("mode") != bench_mode:
            continue
        if bench_ubatch is not None and bench_row.get("ubatch") != str(bench_ubatch):
            continue
        kld_row = kld_by_key.get((bench_row["quant"], bench_row["provider"]))
        if kld_row is None:
            continue
        _require_metric_columns(bench_row)
        ctx = parse_ctx(bench_row["ctx"])
        size = float(bench_row["size_gib"]) if bench_row["size_gib"] else None
        pp = _get_float(bench_row, PP_COL)
        pp_stddev = _get_float(bench_row, PP_STDDEV_COL)
        tg = _get_float(bench_row, TG_COL)
        tg_stddev = _get_float(bench_row, TG_STDDEV_COL)
        reps = int(bench_row["reps"]) if bench_row.get("reps") else None
        ubatch = int(bench_row["ubatch"]) if bench_row.get("ubatch") else None
        if ctx is None or size is None or pp is None or tg is None or ubatch is None:
            continue
        plot_row: PlotRow = {
            "quant": kld_row["quant"],
            "provider": kld_row["provider"],
            "kld": kld_row["kld"],
            "ctx": ctx,
            "size_gib": size,
            "pp4096_tps": pp,
            "pp4096_stddev_tps": pp_stddev,
            "tg128_tps": tg,
            "tg128_stddev_tps": tg_stddev,
            "reps": reps,
            "mode": bench_row.get("mode") or "text",
            "ubatch": ubatch,
        }
        merged.append(plot_row)
    merged.sort(key=lambda row: (QUANT_ORDER.get(row["quant"], 99), row["ubatch"]))
    return merged


CI_95_FACTOR = 1.96


def _ci95(stddev: float | None, reps: int | None) -> float:
    if stddev is None or reps is None or reps < 1:
        return 0.0
    return CI_95_FACTOR * stddev / (reps**0.5)


def _plot_series(
    ax: AxesProtocol,
    rows: list[PlotRow],
    field: MetricField,
    color: str,
    marker: str,
    label: str,
    alpha: float = 1.0,
    err_field: MetricErrorField | None = None,
    xytext: tuple[int, int] = (-5, 5),
    ha: str = "right",
    fontsize: int = 6,
) -> None:
    xs = [_metric_value(row, field) for row in rows]
    ys = [row["kld"] for row in rows]
    labels = [f"{row['quant']} ub={row['ubatch']}" for row in rows]
    if not xs:
        return
    if err_field is not None:
        xerr = [_ci95(_metric_error(row, err_field), row["reps"]) for row in rows]
        ax.errorbar(
            xs,
            ys,
            xerr=xerr,
            fmt="none",
            ecolor=color,
            elinewidth=1.0,
            capsize=2,
            alpha=alpha * 0.6,
            zorder=2.5,
        )
    ax.scatter(xs, ys, color=color, marker=marker, alpha=alpha, zorder=3, label=label)
    for x, y, point_label in zip(xs, ys, labels):
        ax.annotate(
            point_label,
            (x, y),
            textcoords="offset points",
            xytext=xytext,
            fontsize=fontsize,
            ha=ha,
            color=color,
            alpha=max(alpha, 0.6),
        )


def _format_log_major(value: float, _: int) -> str:
    return f"{value:g}" if value >= 0.01 else f"{value:.0e}"


def _format_log_minor(value: float, _: int) -> str:
    return f"{value:g}" if value < 1 else ""


def _format_ctx_tick(value: float, _: int) -> str:
    return f"{value / 1000:.0f}k" if value >= 1000 else f"{value:.0f}"


def plot_model(
    model_name: str,
    data: list[PlotRow],
    out_dir: str,
    show_text: bool = True,
    show_vision: bool = True,
) -> None:
    display = DISPLAY_NAMES.get(model_name, model_name)
    subplots = cast(SubplotsProtocol, plt.subplots)
    close = cast(CloseProtocol, plt.close)
    raw_fig, raw_axes = subplots(2, 2, figsize=(14, 10))
    fig = cast(FigureProtocol, raw_fig)
    axes_grid = cast(AxesGridProtocol, raw_axes)

    suffix = ""
    if not show_text:
        suffix = " (vision only)"
    elif not show_vision:
        suffix = " (text only)"
    fig.suptitle(f"KLD vs Bench Metrics — {display}{suffix}", fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        0.01,
        "Error bars: 95% CI (±1.96 × σ / √n)",
        ha="center",
        fontsize=8,
        fontstyle="italic",
        color="#555555",
    )

    by_provider: dict[str, list[PlotRow]] = {}
    for row in data:
        by_provider.setdefault(row["provider"], []).append(row)

    def provider_sort_key(provider: str) -> int:
        return {"unsloth": 0, "bartowski": 1, "AesSedai": 2}.get(provider, 9)

    plot_specs: list[SubplotSpec] = [
        SubplotSpec("ctx", None, "Context Size (tokens)", (5, 5), "left"),
        SubplotSpec("size_gib", None, "Model Size (GiB)", (5, 5), "left"),
        SubplotSpec("pp4096_tps", "pp4096_stddev_tps", f"Prompt Processing Speed (pp{BENCH_PP}, t/s)", (5, -5), "left"),
        SubplotSpec("tg128_tps", "tg128_stddev_tps", f"Generation Speed (tg{BENCH_TG}, t/s)", (5, -5), "left"),
    ]

    for ax, spec in zip(axes_grid.flat, plot_specs):
        for provider in sorted(by_provider, key=provider_sort_key):
            rows = by_provider[provider]
            color, text_marker, vision_marker = PROVIDER_STYLES.get(provider, DEFAULT_STYLE)
            text_data = [row for row in rows if row["mode"] == "text"] if show_text else []
            vision_data = [row for row in rows if row["mode"] == "vision"] if show_vision else []
            if text_data:
                text_label = f"{provider} (text)" if show_vision else provider
                _plot_series(
                    ax,
                    text_data,
                    spec.field,
                    color,
                    text_marker,
                    text_label,
                    err_field=spec.err,
                    xytext=spec.xytext,
                    ha=spec.ha,
                )
            if vision_data:
                vision_label = f"{provider} (vision)" if show_text else provider
                _plot_series(
                    ax,
                    vision_data,
                    spec.field,
                    color,
                    vision_marker,
                    vision_label,
                    alpha=0.45,
                    err_field=spec.err,
                    xytext=spec.xytext,
                    ha=spec.ha,
                )

        ax.set_xlabel(spec.xlabel, fontsize=9)
        ax.set_ylabel("KLD", fontsize=9)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(mticker.LogLocator(numticks=30))
        ax.yaxis.set_minor_locator(mticker.LogLocator(subs=[2, 3, 4, 5, 6, 7, 8, 9], numticks=30))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_format_log_major))
        ax.yaxis.set_minor_formatter(mticker.FuncFormatter(_format_log_minor))
        ax.tick_params(axis="y", which="minor", labelsize=5, length=2)
        ax.grid(True, alpha=0.3, which="major")
        ax.grid(True, alpha=0.15, which="minor")
        if spec.field == "ctx":
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(_format_ctx_tick))
        ax.legend(fontsize=7, loc="best")

    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
    out_path = os.path.join(out_dir, f"{model_name}-kld-vs-bench.png")
    fig.savefig(out_path, dpi=300)
    close(raw_fig)
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot KLD vs bench metrics.")
    parser.add_argument("--no-text", action="store_true", help="Hide text-mode data")
    parser.add_argument("--no-vision", action="store_true", help="Hide vision-mode data")
    parser.add_argument("--provider", type=str, default=None, help="Only plot a single provider (e.g. unsloth)")
    parser.add_argument("--mode", type=str, choices=["text", "vision"], default=None, help="Filter CSV rows by mode")
    parser.add_argument("--ubatch", type=int, default=None, help="Filter CSV rows by ubatch size")
    args = parser.parse_args()

    show_text = not args.no_text
    show_vision = not args.no_vision

    out_dir = SCRIPT_DIR
    if not os.path.exists(RESULTS_FILE):
        print(f"Bench results file not found: {RESULTS_FILE}")
        return

    bench_rows = load_bench(mode=args.mode, ubatch=args.ubatch)

    if not os.path.exists(KLD_FILE):
        print(f"KLD file not found: {KLD_FILE}")
        return

    kld_rows = load_kld(KLD_FILE)
    by_model: dict[str, list[KldRow]] = {}
    for row in kld_rows:
        by_model.setdefault(row["model"], []).append(row)

    for model_name, model_kld_rows in by_model.items():
        bench_mode = args.mode or "text"
        merged_rows = merge_kld_bench(
            model_kld_rows,
            bench_rows,
            model_name,
            bench_mode=bench_mode,
            bench_ubatch=args.ubatch,
        )
        if not merged_rows:
            print(f"No matching bench data for {model_name}")
            continue
        if args.provider:
            merged_rows = [row for row in merged_rows if row["provider"] == args.provider]
            if not merged_rows:
                print(f"No matching bench data for {model_name} (provider={args.provider})")
                continue
        print(f"{model_name}: {len(merged_rows)} matched quants")
        plot_model(model_name, merged_rows, out_dir, show_text=show_text, show_vision=show_vision)


if __name__ == "__main__":
    main()
