#!/usr/bin/env python3
"""Plot KLD vs benchmark metrics for quantized models."""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

QUANT_ORDER = {
    "IQ1_M": 0,
    "IQ2_XXS": 1,
    "IQ2_XSS": 2,
    "IQ2_XS": 3,
    "IQ2_S": 4,
    "IQ2_M": 5,
    "Q2_K": 6,
    "Q2_K_L": 7,
    "UD-IQ2_XXS": 8,
    "UD-IQ2_XSS": 9,
    "UD-IQ2_M": 10,
    "UD-Q2_K_XL": 11,
    "IQ3_XXS": 12,
    "IQ3_XS": 13,
    "IQ3_S": 14,
    "IQ3_M": 15,
    "Q3_K_S": 16,
    "Q3_K_M": 17,
    "Q3_K_L": 18,
    "Q3_K_XL": 19,
    "UD-IQ3_XXS": 20,
    "UD-IQ3_S": 21,
    "UD-Q3_K_S": 22,
    "UD-Q3_K_M": 23,
    "UD-Q3_K_XL": 24,
    "IQ4_XS": 25,
    "IQ4_NL": 26,
    "UD-IQ4_XS": 27,
    "UD-IQ4_NL": 28,
    "UD-IQ4_NL_XL": 29,
    "MXFP4_MOE": 30,
    "Q4_0": 31,
    "Q4_1": 32,
    "Q4_K_S": 33,
    "Q4_K_M": 34,
    "Q4_K_L": 35,
    "UD-Q4_K_S": 36,
    "UD-Q4_K_M": 37,
    "UD-Q4_K_XL": 38,
    "Q5_K_S": 39,
    "Q5_K_M": 40,
    "Q5_K_L": 41,
    "UD-Q5_K_S": 42,
    "UD-Q5_K_M": 43,
    "UD-Q5_K_XL": 44,
    "Q6_K": 45,
    "Q6_K_L": 46,
    "UD-Q6_K": 47,
    "UD-Q6_K_XL": 48,
    "Q8_0": 49,
    "UD-Q8_K_XL": 50,
}

PROVIDER_COLORS = {
    "unsloth": "#2166AC",
    "bartowski": "#B2182B",
    "AesSedai": "#1B7837",
}
TEXT_MARKERS = {
    "unsloth": "o",
    "bartowski": "s",
    "AesSedai": "D",
}
VISION_MARKERS = {
    "unsloth": "^",
    "bartowski": "v",
    "AesSedai": "P",
}

BENCH_FILE = os.path.join(SCRIPT_DIR, "fit-bench-results.csv")
KLD_FILES = {
    "gemma-4-26B-A4B": os.path.join(SCRIPT_DIR, "Gemma4-26B-A4B-KLD.csv"),
    "Qwen3.6-35B-A3B": os.path.join(SCRIPT_DIR, "Qwen3.6-35B-A3B-KLD.csv"),
}

DISPLAY_NAMES = {
    "gemma-4-26B-A4B": "Gemma 4 26B (A4B)",
    "Qwen3.6-35B-A3B": "Qwen 3.6 35B (A3B)",
}


def parse_ctx(s):
    if not s or s == "-":
        return None
    s = s.strip()
    if s.endswith("k"):
        return int(float(s[:-1]) * 1000)
    return int(s)


def load_bench():
    rows = []
    with open(BENCH_FILE, newline="") as f:
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
        ctx = parse_ctx(b["ctx"])
        size = float(b["size_gib"]) if b["size_gib"] else None
        pp = float(b["pp2048_tps"]) if b["pp2048_tps"] else None
        tg = float(b["tg512_tps"]) if b["tg512_tps"] else None
        vctx = parse_ctx(b.get("vctx", ""))
        vpp = float(b["vpp2048_tps"]) if b.get("vpp2048_tps", "") not in ("", "-") else None
        vtg = float(b["vtg512_tps"]) if b.get("vtg512_tps", "") not in ("", "-") else None
        if ctx is None or size is None or pp is None or tg is None:
            continue
        merged.append({
            "quant": k["quant"],
            "provider": k["provider"],
            "kld": k["kld"],
            "ctx": ctx,
            "size_gib": size,
            "pp2048_tps": pp,
            "tg512_tps": tg,
            "vctx": vctx,
            "vpp2048_tps": vpp,
            "vtg512_tps": vtg,
        })
    merged.sort(key=lambda r: QUANT_ORDER.get(r["quant"], 99))
    return merged


def _plot_series(ax, rows, field, color, marker, label, linestyle="-", alpha=1.0):
    xs = [r[field] for r in rows if r[field] is not None]
    ys = [r["kld"] for r in rows if r[field] is not None]
    quants = [r["quant"] for r in rows if r[field] is not None]
    if not xs:
        return
    sorted_pairs = sorted(zip(xs, ys))
    sx = [p[0] for p in sorted_pairs]
    sy = [p[1] for p in sorted_pairs]
    ax.plot(sx, sy, color=color, linewidth=1.2, alpha=alpha * 0.5, linestyle=linestyle, zorder=2)
    ax.scatter(xs, ys, color=color, marker=marker, s=60, alpha=alpha, zorder=3, label=label)
    for x, y, q in zip(xs, ys, quants):
        ax.annotate(
            q,
            (x, y),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=6,
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

    by_provider = {}
    for r in data:
        by_provider.setdefault(r["provider"], []).append(r)

    prov_sort = lambda p: {"unsloth": 0, "bartowski": 1, "AesSedai": 2}.get(p, 9)

    plot_specs = [
        ("ctx", "vctx", "Context Size (tokens)", "left", True),
        ("size_gib", None, "Model Size (GiB)", "right", False),
        ("pp2048_tps", "vpp2048_tps", "Prompt Processing Speed (pp2048, t/s)", "left", True),
        ("tg512_tps", "vtg512_tps", "Generation Speed (tg512, t/s)", "left", True),
    ]

    for ax, (text_field, vis_field, xlabel, large_dir, has_vision) in zip(axes.flat, plot_specs):
        for prov in sorted(by_provider, key=prov_sort):
            rows = by_provider[prov]
            color = PROVIDER_COLORS.get(prov, "#888888")
            tm = TEXT_MARKERS.get(prov, "o")
            vm = VISION_MARKERS.get(prov, "^")
            if show_text:
                text_label = f"{prov} (text)" if show_vision else prov
                _plot_series(ax, rows, text_field, color, tm, text_label)
            if has_vision and vis_field and show_vision:
                vis_label = f"{prov} (vision)" if show_text else prov
                _plot_series(ax, rows, vis_field, color, vm, vis_label, linestyle="--", alpha=0.45)

        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("KLD", fontsize=9)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(mticker.LogLocator(numticks=30))
        ax.yaxis.set_minor_locator(mticker.LogLocator(subs=[2, 3, 4, 5, 6, 7, 8, 9], numticks=30))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}" if v >= 0.01 else f"{v:.0e}"))
        ax.yaxis.set_minor_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}" if v < 1 else ""))
        ax.tick_params(axis="y", which="minor", labelsize=5, length=2)
        ax.grid(True, alpha=0.3, which="major")
        ax.grid(True, alpha=0.15, which="minor")
        if text_field == "ctx" or text_field == "vctx":
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k" if v >= 1000 else f"{v:.0f}"))
        if large_dir == "left":
            ax.invert_xaxis()
        ax.legend(fontsize=7, loc="best")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(out_dir, f"{model_name}-kld-vs-bench.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot KLD vs bench metrics.")
    parser.add_argument("--no-text", action="store_true", help="Hide text-mode data")
    parser.add_argument("--no-vision", action="store_true", help="Hide vision-mode data")
    args = parser.parse_args()

    show_text = not args.no_text
    show_vision = not args.no_vision

    out_dir = SCRIPT_DIR
    bench = load_bench()

    for model_name, kld_path in KLD_FILES.items():
        if not os.path.exists(kld_path):
            print(f"Skipping {model_name}: {kld_path} not found")
            continue
        kld = load_kld(kld_path)
        merged = merge_kld_bench(kld, bench, model_name)
        if not merged:
            print(f"No matching bench data for {model_name}")
            continue
        print(f"{model_name}: {len(merged)} matched quants")
        plot_model(model_name, merged, out_dir, show_text=show_text, show_vision=show_vision)


if __name__ == "__main__":
    main()
