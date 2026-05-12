#!/usr/bin/env python3
"""Plot KLD vs benchmark metrics for quantized models."""

import argparse
import csv
import os
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from quant_order import QUANT_ORDER
from results import (
    BENCH_PP,
    BENCH_TG,
    PP_COL,
    PP_STDDEV_COL,
    RESULTS_FILE,
    TG_COL,
    TG_STDDEV_COL,
    VPP_COL,
    VPP_STDDEV_COL,
    VTG_COL,
    VTG_STDDEV_COL,
    parse_ctx,
)


class SubplotSpec(NamedTuple):
    text_field: str
    text_err: str | None
    vis_field: str | None
    vis_err: str | None
    xlabel: str
    large_dir: str
    text_reps: str | None
    vis_reps: str | None
    xytext: tuple[int, int]
    ha: str

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


def load_bench():
    rows = []
    with open(RESULTS_FILE, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def load_kld(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            row["kld"] = float(row["kld"])
            rows.append(row)
    return rows


def _get_float(row, *keys):
    for key in keys:
        value = row.get(key, "")
        if value not in ("", "-"):
            return float(value)
    return None


def _require_metric_columns(row):
    required = [PP_COL, PP_STDDEV_COL, TG_COL, TG_STDDEV_COL]
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(
            "fit-bench-results.csv is missing required new benchmark columns: "
            + ", ".join(missing)
        )


def merge_kld_bench(kld_rows, bench_rows, model_name):
    merged = []
    bench_by_key = {}
    for b in bench_rows:
        if b["model"] != model_name:
            continue
        key = (b["quant"], b["provider"])
        bench_by_key[key] = b

    for k in kld_rows:
        key = (k["quant"], k["provider"])
        b = bench_by_key.get(key)
        if b is None:
            continue
        _require_metric_columns(b)
        ctx = parse_ctx(b["ctx"])
        size = float(b["size_gib"]) if b["size_gib"] else None
        pp = _get_float(b, PP_COL)
        pp_stddev = _get_float(b, PP_STDDEV_COL)
        tg = _get_float(b, TG_COL)
        tg_stddev = _get_float(b, TG_STDDEV_COL)
        reps = int(b["reps"]) if b.get("reps") else None
        vctx = parse_ctx(b.get("vctx", ""))
        vpp = _get_float(b, VPP_COL)
        vpp_stddev = _get_float(b, VPP_STDDEV_COL)
        vtg = _get_float(b, VTG_COL)
        vtg_stddev = _get_float(b, VTG_STDDEV_COL)
        vreps = int(b.get("vreps")) if b.get("vreps") else None
        if ctx is None or size is None or pp is None or tg is None:
            continue
        merged.append(
            {
                "quant": k["quant"],
                "provider": k["provider"],
                "kld": k["kld"],
                "ctx": ctx,
                "size_gib": size,
                PP_COL: pp,
                PP_STDDEV_COL: pp_stddev,
                TG_COL: tg,
                TG_STDDEV_COL: tg_stddev,
                "reps": reps,
                "vctx": vctx,
                VPP_COL: vpp,
                VPP_STDDEV_COL: vpp_stddev,
                VTG_COL: vtg,
                VTG_STDDEV_COL: vtg_stddev,
                "vreps": vreps,
            }
        )
    merged.sort(key=lambda r: QUANT_ORDER.get(r["quant"], 99))
    return merged


CI_95_FACTOR = 1.96


def _ci95(stddev, reps):
    if stddev is None or reps is None or reps < 1:
        return 0
    return CI_95_FACTOR * stddev / (reps**0.5)


def _plot_series(ax, rows, field, color, marker, label, alpha=1.0, err_field=None, reps_field=None, xytext=(-5, 5), ha="right", fontsize=6):
    plot_rows = [r for r in rows if r[field] is not None]
    xs = [r[field] for r in plot_rows]
    ys = [r["kld"] for r in plot_rows]
    quants = [r["quant"] for r in plot_rows]
    if not xs:
        return
    if err_field:
        if reps_field:
            xerr = [_ci95(r.get(err_field), r.get(reps_field)) for r in plot_rows]
        else:
            xerr = [r.get(err_field) or 0 for r in plot_rows]
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
    for x, y, q in zip(xs, ys, quants):
        ax.annotate(
            q,
            (x, y),
            textcoords="offset points",
            xytext=xytext,
            fontsize=fontsize,
            ha=ha,
            color=color,
            alpha=max(alpha, 0.6),
        )


def plot_model(model_name, data, out_dir, show_text=True, show_vision=True):
    display = DISPLAY_NAMES.get(model_name, model_name)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    suffix = ""
    if not show_text:
        suffix = " (vision only)"
    elif not show_vision:
        suffix = " (text only)"
    fig.suptitle(f"KLD vs Bench Metrics — {display}{suffix}", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.01, "Error bars: 95% CI (±1.96 × σ / √n)", ha="center", fontsize=8, fontstyle="italic", color="#555555")

    by_provider = {}
    for r in data:
        by_provider.setdefault(r["provider"], []).append(r)

    def provider_sort_key(provider):
        return {"unsloth": 0, "bartowski": 1, "AesSedai": 2}.get(provider, 9)

    plot_specs = [
        SubplotSpec("ctx", None, "vctx", None, "Context Size (tokens)", "right", None, None, (5, 5), "left"),
        SubplotSpec("size_gib", None, "size_gib", None, "Model Size (GiB)", "right", None, None, (5, 5), "left"),
        SubplotSpec(PP_COL, PP_STDDEV_COL, VPP_COL, VPP_STDDEV_COL, f"Prompt Processing Speed (pp{BENCH_PP}, t/s)", "right", "reps", "vreps", (5, -5), "left"),
        SubplotSpec(TG_COL, TG_STDDEV_COL, VTG_COL, VTG_STDDEV_COL, f"Generation Speed (tg{BENCH_TG}, t/s)", "right", "reps", "vreps", (5, -5), "left"),
    ]

    for ax, spec in zip(axes.flat, plot_specs):
        for prov in sorted(by_provider, key=provider_sort_key):
            rows = by_provider[prov]
            color, tm, vm = PROVIDER_STYLES.get(prov, DEFAULT_STYLE)
            if show_text:
                text_label = f"{prov} (text)" if show_vision else prov
                _plot_series(ax, rows, spec.text_field, color, tm, text_label, err_field=spec.text_err, reps_field=spec.text_reps, xytext=spec.xytext, ha=spec.ha)
            if spec.vis_field and show_vision:
                vis_label = f"{prov} (vision)" if show_text else prov
                _plot_series(ax, rows, spec.vis_field, color, vm, vis_label, alpha=0.45, err_field=spec.vis_err, reps_field=spec.vis_reps, xytext=spec.xytext, ha=spec.ha)

        ax.set_xlabel(spec.xlabel, fontsize=9)
        ax.set_ylabel("KLD", fontsize=9)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(mticker.LogLocator(numticks=30))
        ax.yaxis.set_minor_locator(mticker.LogLocator(subs=[2, 3, 4, 5, 6, 7, 8, 9], numticks=30))
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v:g}" if v >= 0.01 else f"{v:.0e}")
        )
        ax.yaxis.set_minor_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}" if v < 1 else ""))
        ax.tick_params(axis="y", which="minor", labelsize=5, length=2)
        ax.grid(True, alpha=0.3, which="major")
        ax.grid(True, alpha=0.15, which="minor")
        if spec.text_field in ("ctx", "vctx"):
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k" if v >= 1000 else f"{v:.0f}")
            )
        if spec.large_dir == "left":
            ax.invert_xaxis()
        ax.legend(fontsize=7, loc="best")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join(out_dir, f"{model_name}-kld-vs-bench.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot KLD vs bench metrics.")
    parser.add_argument("--no-text", action="store_true", help="Hide text-mode data")
    parser.add_argument("--no-vision", action="store_true", help="Hide vision-mode data")
    parser.add_argument("--provider", type=str, default=None, help="Only plot a single provider (e.g. unsloth)")
    args = parser.parse_args()

    show_text = not args.no_text
    show_vision = not args.no_vision

    out_dir = SCRIPT_DIR
    bench = load_bench()

    if not os.path.exists(KLD_FILE):
        print(f"KLD file not found: {KLD_FILE}")
        return

    kld_all = load_kld(KLD_FILE)
    by_model = {}
    for row in kld_all:
        by_model.setdefault(row["model"], []).append(row)

    for model_name, kld_rows in by_model.items():
        merged = merge_kld_bench(kld_rows, bench, model_name)
        if not merged:
            print(f"No matching bench data for {model_name}")
            continue
        if args.provider:
            merged = [r for r in merged if r["provider"] == args.provider]
            if not merged:
                print(f"No matching bench data for {model_name} (provider={args.provider})")
                continue
        print(f"{model_name}: {len(merged)} matched quants")
        plot_model(model_name, merged, out_dir, show_text=show_text, show_vision=show_vision)


if __name__ == "__main__":
    main()
