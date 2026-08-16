#!/usr/bin/env python3
"""Plot Kairos performance relative to hand-written / standard implementations.

Main figure: pct_of_ref (= ref_ms / kairos_ms) vs time-step T, one line per
pattern. y=100% dashed reference; 80-105% band = "reaches / matches the
opponent". Optional right panel: absolute ms vs T (Kairos vs opponent).

Reads eval_handwritten.csv produced by dev_tests/eval_handwritten.py.
Only rows with correctness_pass=True and finite timings are plotted.
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# Palette consistent with dev_tests/plot_eval_three_taxes.py.
PATTERNS = ("conv_bn_lif", "linear_lif", "selective_scan", "convlstm")
COLORS = {
    "conv_bn_lif": "#B5761F",
    "linear_lif": "#2E7D6F",
    "selective_scan": "#882255",
    "convlstm": "#4A6FA5",
}
LABELS = {
    "conv_bn_lif": "conv+bn+lif (vs SpikingJelly)",
    "linear_lif": "linear+lif (vs SpikingJelly)",
    "selective_scan": "selective scan (vs Mamba)",
    "convlstm": "ConvLSTM (vs standard cell)",
}
SHORT = {
    "conv_bn_lif": "conv+bn+lif",
    "linear_lif": "linear+lif",
    "selective_scan": "sel. scan",
    "convlstm": "ConvLSTM",
}
MARKERS = {
    "conv_bn_lif": "o",
    "linear_lif": "s",
    "selective_scan": "^",
    "convlstm": "D",
}

# Diverging ramp for the pct-of-ref heatmap: slower (<100, warm orange) ->
# ~parity (light neutral) -> faster (>100, teal). Orange<->teal is a CVD-safe
# diverging pair; the neutral midpoint sits at 100%.
DIVERGING = LinearSegmentedColormap.from_list(
    "slower_faster", ["#B5761F", "#F3EEE6", "#2E7D6F"]
)
# Two categorical series for eager vs cudagraph (blue/orange = CVD-safe pair).
MODE_COLORS = {"eager": "#4A6FA5", "cudagraph": "#B5761F"}
MODE_LABELS = {"eager": "eager (per-call latency)", "cudagraph": "CUDA-graph (kernel-only)"}


def read_rows(path: Path, mode: str = "eager"):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out = []
    for row in rows:
        # CSVs without a mode column (older runs) are all eager.
        row_mode = row.get("mode") or "eager"
        if row_mode != mode:
            continue
        if str(row.get("correctness_pass")).lower() not in ("true", "1"):
            continue
        try:
            T = int(float(row["T"]))
            pct = float(row["pct_of_ref"])
            k = float(row["kairos_ms_mean"])
            r = float(row["ref_ms_mean"])
        except (ValueError, KeyError):
            continue
        if not (math.isfinite(pct) and math.isfinite(k) and math.isfinite(r)):
            continue
        out.append({"pattern": row["pattern"], "T": T, "pct": pct, "kairos": k, "ref": r})
    return out


def read_all(path: Path):
    """All correctness-passing rows, keeping the `mode` column (eager/cudagraph)."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out = []
    for row in rows:
        if str(row.get("correctness_pass")).lower() not in ("true", "1"):
            continue
        try:
            T = int(float(row["T"]))
            pct = float(row["pct_of_ref"])
            k = float(row["kairos_ms_mean"])
            r = float(row["ref_ms_mean"])
        except (ValueError, KeyError):
            continue
        if not (math.isfinite(pct) and math.isfinite(k) and math.isfinite(r)):
            continue
        out.append({
            "pattern": row["pattern"], "mode": row.get("mode") or "eager",
            "T": T, "pct": pct, "kairos": k, "ref": r,
        })
    return out


def series(rows, pattern, field):
    pts = sorted([(r["T"], r[field]) for r in rows if r["pattern"] == pattern])
    if not pts:
        return None, None
    xs, ys = zip(*pts)
    return np.array(xs), np.array(ys)


def set_style():
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.5,
            "legend.fontsize": 7,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot(rows, output: Path, with_abs: bool):
    set_style()
    all_T = sorted({r["T"] for r in rows})
    if not all_T:
        raise ValueError("no correctness-passing timed rows to plot")

    ncols = 2 if with_abs else 1
    figsize = (7.1, 2.7) if with_abs else (3.5, 2.7)
    fig, axes = plt.subplots(1, ncols, figsize=figsize, constrained_layout=True)
    pct_ax = axes[0] if with_abs else axes

    # Main panel: pct_of_ref vs T.
    pct_ax.axhline(100.0, color="#999999", linestyle="--", linewidth=0.9, zorder=1)
    for pattern in PATTERNS:
        xs, ys = series(rows, pattern, "pct")
        if xs is None:
            continue
        pct_ax.plot(
            xs, ys, color=COLORS[pattern], marker=MARKERS[pattern], markersize=3.5,
            label=LABELS[pattern], zorder=3,
        )
    pct_ax.set_xscale("log", base=2)
    pct_ax.set_xticks(all_T)
    pct_ax.set_xticklabels([str(t) for t in all_T])
    pct_ax.set_xlabel("time steps T")
    pct_ax.set_ylabel("% of hand-written / standard  (ref_ms / kairos_ms)")
    pct_ax.grid(True, color="#EDF1F5", linewidth=0.65)
    pct_ax.set_axisbelow(True)
    pct_ax.set_title("(a) Kairos relative performance" if with_abs else None)

    # Optional panel: absolute ms.
    if with_abs:
        abs_ax = axes[1]
        for pattern in PATTERNS:
            xs, kys = series(rows, pattern, "kairos")
            _, rys = series(rows, pattern, "ref")
            if xs is None:
                continue
            abs_ax.plot(xs, kys, color=COLORS[pattern], marker=MARKERS[pattern],
                        markersize=3.5, linestyle="-", label=f"{pattern} Kairos")
            abs_ax.plot(xs, rys, color=COLORS[pattern], marker=MARKERS[pattern],
                        markersize=3.5, linestyle=":", alpha=0.7)
        abs_ax.set_xscale("log", base=2)
        abs_ax.set_yscale("log")
        abs_ax.set_xticks(all_T)
        abs_ax.set_xticklabels([str(t) for t in all_T])
        abs_ax.set_xlabel("time steps T")
        abs_ax.set_ylabel("latency (ms)")
        abs_ax.grid(True, color="#EDF1F5", linewidth=0.65)
        abs_ax.set_axisbelow(True)
        abs_ax.set_title("(b) Absolute latency (solid=Kairos, dotted=opponent)")

    handles, labels = pct_ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="outside upper center")
    fig.savefig(output, format="pdf")
    plt.close(fig)


# Heatmap row labels and mode titles (paper-facing names).
HMAP_ROWS = {
    "conv_bn_lif": "spiking conv.",
    "linear_lif": "spiking MLP",
    "selective_scan": "sel. scan",
    "convlstm": "ConvLSTM",
}
HMAP_TITLES = {"eager": "no CUDA-graph", "cudagraph": "CUDA-graph"}


def plot_heatmap(rows, output: Path):
    """Style B: pattern x T matrix, cell colour = pct_of_ref on a diverging ramp
    centred at 100% (parity). Two panels side by side (no-CUDA-graph, CUDA-graph)
    sized for a single column. Every cell is annotated with the actual percentage,
    so extreme values stay legible while the colour conveys the win/loss field and
    the scan crossover at a glance."""
    set_style()
    modes = ["eager", "cudagraph"]
    all_T = sorted({r["T"] for r in rows})
    if not all_T:
        raise ValueError("no correctness-passing timed rows to plot")
    # Diverging scale centred at parity. vmin=70 gives the sub-100 (Kairos slower)
    # band real colourbar space so it can be ticked; values above 250% clip to the
    # darkest teal but their number is still printed in-cell.
    norm = TwoSlopeNorm(vcenter=100.0, vmin=70.0, vmax=250.0)

    # Keep the compact single-column width, but reduce the height now that each
    # 4x4 matrix is rendered with square (rather than vertically stretched) cells.
    fig, axes = plt.subplots(1, 2, figsize=(3.85, 1.93))
    fig.subplots_adjust(left=0.21, right=0.84, bottom=0.23, top=0.82, wspace=0.15)
    im = None
    for ax, mode in zip(axes, modes):
        grid = np.full((len(PATTERNS), len(all_T)), np.nan)
        for i, p in enumerate(PATTERNS):
            for j, T in enumerate(all_T):
                vals = [r["pct"] for r in rows if r["mode"] == mode and r["pattern"] == p and r["T"] == T]
                if vals:
                    grid[i, j] = vals[0]
        im = ax.imshow(grid, aspect="equal", cmap=DIVERGING, norm=norm)
        ax.set_xticks(range(len(all_T)), [str(t) for t in all_T], fontsize=5.5)
        if mode == "eager":
            ax.set_yticks(range(len(PATTERNS)), [HMAP_ROWS[p] for p in PATTERNS],
                          fontsize=7, rotation=0, ha="right", va="center")
        else:
            ax.set_yticks([])
        ax.set_title(HMAP_TITLES[mode], fontsize=8)
        ax.set_xlabel("time steps T", fontsize=7.5)
        for i in range(len(PATTERNS)):
            for j in range(len(all_T)):
                v = grid[i, j]
                if np.isnan(v):
                    continue
                rgba = DIVERGING(norm(v))
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=4.6,
                        color="white" if lum < 0.5 else "#1A1A1A")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
    # Anchor the colourbar to the heatmap axes itself, so its height exactly
    # matches the coloured matrix rather than including titles/x-axis labels.
    cax = axes[1].inset_axes([1.07, 0.0, 0.065, 1.0])
    cb = fig.colorbar(im, cax=cax, orientation="vertical", extend="max",
                      ticks=[70, 85, 100, 150, 200, 250])
    cb.set_label("% of reference", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    fig.savefig(output, format=output.suffix.lstrip(".") or "pdf",
                bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def plot_facets(rows, output: Path):
    """Style C: 2x2 small multiples, one panel per pattern; each panel overlays
    the eager and CUDA-graph curves vs T with a 100% parity reference. Reads the
    launch-overhead story per pattern (eager high -> cudagraph near parity) and
    the scan crossover, which the single overlaid line chart cannot show."""
    set_style()
    all_T = sorted({r["T"] for r in rows})
    if not all_T:
        raise ValueError("no correctness-passing timed rows to plot")
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.6), constrained_layout=True, sharex=True)
    for ax, p in zip(axes.flat, PATTERNS):
        ax.axhline(100.0, color="#999999", linestyle="--", linewidth=0.9, zorder=1)
        for mode in ("eager", "cudagraph"):
            pts = sorted([(r["T"], r["pct"]) for r in rows if r["pattern"] == p and r["mode"] == mode])
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=MODE_COLORS[mode], marker="o", markersize=3.2,
                    label=MODE_LABELS[mode], zorder=3)
        ax.set_xscale("log", base=2)
        ax.set_xticks(all_T, [str(t) for t in all_T])
        ax.set_title(LABELS[p], fontsize=8)
        ax.grid(True, color="#EDF1F5", linewidth=0.65)
        ax.set_axisbelow(True)
    for ax in axes[-1]:
        ax.set_xlabel("time steps T")
    for ax in axes[:, 0]:
        ax.set_ylabel("% of hand-written")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="outside upper center")
    fig.savefig(output, format=output.suffix.lstrip(".") or "pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="eval_handwritten.csv")
    parser.add_argument("--output", default="fig_handwritten.pdf")
    parser.add_argument("--style", choices=("lines", "heatmap", "facets"), default="lines",
                        help="lines: pattern lines vs T (one --mode); "
                             "heatmap: pattern x T grid, both modes; "
                             "facets: 2x2 per-pattern, eager vs cudagraph")
    parser.add_argument("--with-abs", action="store_true",
                        help="[lines only] add right panel with absolute latency vs T")
    parser.add_argument("--mode", choices=("eager", "cudagraph"), default="eager",
                        help="[lines only] which timing mode's rows to plot")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.style == "lines":
        rows = read_rows(Path(args.input), mode=args.mode)
        plot(rows, output, args.with_abs)
        print(f"[write] {output}  style=lines mode={args.mode} ({len(rows)} rows)")
    elif args.style == "heatmap":
        rows = read_all(Path(args.input))
        plot_heatmap(rows, output)
        print(f"[write] {output}  style=heatmap ({len(rows)} rows, both modes)")
    else:
        rows = read_all(Path(args.input))
        plot_facets(rows, output)
        print(f"[write] {output}  style=facets ({len(rows)} rows, both modes)")


if __name__ == "__main__":
    main()
