#!/usr/bin/env python3
"""Create paper-ready vector figures for the three-taxes experiment."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COL = {
    "per_step": "#B5761F",
    "batched": "#8A9199",
    "fused": "#2E7D6F",
    "theory": "#8A9199",
    "roofline": "#33475B",
    "accent": "#882255",
}
LABEL = {"per_step": "Per-step", "batched": "Batched-only", "fused": "Fused"}
MARKER = {"per_step": "o", "batched": "s", "fused": "^"}


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["T"] = int(row["T"])
        for key in row:
            if key not in {"mode", "T"}:
                row[key] = number(row[key])
    return rows


def series(rows, mode, field):
    selected = sorted((row for row in rows if row["mode"] == mode), key=lambda x: x["T"])
    return np.array([row["T"] for row in selected]), np.array([row[field] for row in selected])


def configure():
    plt.rcParams.update(
        {
            "font.size": 7.5,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 6.8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "lines.linewidth": 1.35,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_taxes(rows, output: Path):
    fig, ax = plt.subplots(figsize=(3.45, 2.45), constrained_layout=True)
    complete = any(math.isfinite(row["dram_total_bytes"]) for row in rows)
    for mode in ("per_step", "batched", "fused"):
        x, y = series(rows, mode, "dram_total_bytes")
        ax.plot(x, y / 1e9, marker=MARKER[mode], color=COL[mode], label=LABEL[mode])
    theory_rows = sorted(
        (row for row in rows if row["mode"] == "per_step"), key=lambda x: x["T"]
    )
    tx = np.array([row["T"] for row in theory_rows])
    ty = np.array([row["min_bytes_theory"] for row in theory_rows]) / 1e9
    ax.plot(tx, ty, "--", color=COL["theory"], label="HBM lower bound")
    px, py = series(rows, "per_step", "dram_total_bytes")
    fx, fy = series(rows, "fused", "dram_total_bytes")
    if complete and np.all(np.isfinite(py)) and np.all(np.isfinite(fy)):
        ax.fill_between(px, fy / 1e9, py / 1e9, color=COL["per_step"], alpha=0.12)
    ax.set_xscale("log", base=2)
    ax.set_xticks(tx, [str(value) for value in tx])
    ax.set_xlabel("Timesteps (T)")
    ax.set_ylabel("HBM traffic (GB)")
    ax.grid(True, color="#E6E6E6", linewidth=0.55)

    kernels = ax.twinx()
    for mode, linestyle in (("per_step", ":"), ("fused", "-.")):
        x, y = series(rows, mode, "kernel_count")
        kernels.plot(
            x,
            y,
            linestyle,
            marker=MARKER[mode],
            color=COL[mode],
            alpha=0.85,
            label=f"{LABEL[mode]} launches",
        )
    if complete:
        kernels.set_yscale("log")
    kernels.set_ylabel("Kernel launches")
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = kernels.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, loc="upper left", frameon=False, ncol=2)
    if not complete:
        ax.text(
            0.5,
            0.48,
            "NCU metrics not collected",
            transform=ax.transAxes,
            ha="center",
            color=COL["accent"],
        )
    fig.savefig(output, format="pdf")
    plt.close(fig)


def plot_roofline(rows, metadata, output: Path, representative_t: int):
    bandwidth = float(metadata["peak_bandwidth_gbs"]) * 1e9
    peak = float(metadata["peak_compute_tflops"]) * 1e12
    ridge = peak / bandwidth
    intensities = [row["arith_intensity"] for row in rows if math.isfinite(row["arith_intensity"])]
    xmin = min(intensities) / 2 if intensities else ridge / 100
    xmax = max(max(intensities) * 2 if intensities else ridge * 10, ridge * 10)
    x = np.logspace(math.log10(max(xmin, 1e-3)), math.log10(xmax), 300)
    roof = np.minimum(x * bandwidth, peak)

    fig, ax = plt.subplots(figsize=(3.45, 2.45), constrained_layout=True)
    ax.plot(x, roof / 1e12, color=COL["roofline"], label="Configured roofline")
    ax.axvline(ridge, color=COL["roofline"], linestyle=":", linewidth=1)
    ax.text(ridge * 1.08, peak / 1e12 * 0.18, f"Ridge\n{ridge:.1f} FLOP/B", color=COL["roofline"])
    points = [row for row in rows if row["T"] == representative_t]
    complete = False
    for row in points:
        ai = row["arith_intensity"]
        latency = row["time_ms_mean"]
        if not (math.isfinite(ai) and math.isfinite(latency) and latency > 0):
            continue
        problem = metadata["problem"]
        flops = (
            2
            * problem["cout"]
            * problem["height"]
            * problem["width"]
            * problem["cin"]
            * 9
            * representative_t
            * problem["batch"]
        )
        performance = flops / (latency * 1e-3) / 1e12
        mode = row["mode"]
        ax.scatter(
            ai,
            performance,
            s=31,
            marker=MARKER[mode],
            color=COL[mode],
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
            label=f"{LABEL[mode]} (T={representative_t})",
        )
        complete = True
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("Performance (TFLOP/s)")
    ax.grid(True, which="both", color="#E6E6E6", linewidth=0.55)
    ax.legend(loc="lower right", frameon=False)
    if not complete:
        ax.text(
            0.5,
            0.42,
            "NCU metrics not collected",
            transform=ax.transAxes,
            ha="center",
            color=COL["accent"],
        )
    fig.savefig(output, format="pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="test/motivation_three_taxes")
    parser.add_argument("--representative-t", type=int, default=16)
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    rows = read_rows(input_dir / "three_taxes_by_T.csv")
    metadata = json.loads((input_dir / "metadata.json").read_text(encoding="utf-8"))
    configure()
    plot_taxes(rows, input_dir / "fig_motivation_taxes.pdf")
    plot_roofline(
        rows,
        metadata,
        input_dir / "fig_motivation_roofline.pdf",
        args.representative_t,
    )
    print(f"[write] {input_dir / 'fig_motivation_taxes.pdf'}")
    print(f"[write] {input_dir / 'fig_motivation_roofline.pdf'}")


if __name__ == "__main__":
    main()

# python dev_tests/plot_motivation.py \
#   --input-dir test/motivation_three_taxes \
#   --representative-t 16