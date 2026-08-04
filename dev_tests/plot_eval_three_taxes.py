#!/usr/bin/env python3
"""Plot real-model HBM traffic and execution time for four execution modes."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODELS = ("resnet18", "spiketransformer", "mamba", "convlstm")
MODES = ("per_step", "batched", "fused", "multi_stream")
COLORS = {
    "per_step": "#B5761F",
    "batched": "#8A9199",
    "fused": "#2E7D6F",
    "multi_stream": "#882255",
}
LABELS = {
    "per_step": "Per-step",
    "batched": "Graph-batched",
    "fused": "Fused",
    "multi_stream": "Multi-stream",
}


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in (
            "T",
            "batch",
            "dram_total_bytes",
            "min_bytes_theory",
            "ratio_to_lower_bound",
            "time_ms_mean",
            "time_ms_std",
        ):
            row[key] = float(row[key])
    return rows


def plot(rows, output: Path):
    lookup = {(row["model"], row["mode"]): row for row in rows}
    missing = [(model, mode) for model in MODELS for mode in MODES if (model, mode) not in lookup]
    if missing:
        raise ValueError(f"CSV is missing required cases: {missing}")
    traffic = np.array(
        [[lookup[model, mode]["dram_total_bytes"] / 1e9 for model in MODELS] for mode in MODES]
    )
    latency = np.array(
        [[lookup[model, mode]["time_ms_mean"] for model in MODELS] for mode in MODES]
    )
    if not (np.all(np.isfinite(traffic)) and np.all(np.isfinite(latency))):
        raise ValueError("HBM traffic or latency contains non-finite values")

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    x = np.arange(len(MODELS))
    width = 0.19
    fig, (traffic_ax, latency_ax) = plt.subplots(
        1, 2, figsize=(7.1, 2.65), constrained_layout=True
    )
    offsets = (np.arange(len(MODES)) - (len(MODES) - 1) / 2) * width
    for index, mode in enumerate(MODES):
        for ax, values in ((traffic_ax, traffic), (latency_ax, latency)):
            ax.bar(
                x + offsets[index],
                values[index],
                width,
                color=COLORS[mode],
                edgecolor="white",
                linewidth=0.6,
                label=LABELS[mode],
            )
    for ax in (traffic_ax, latency_ax):
        ax.set_xticks(x, MODELS)
        ax.grid(axis="y", color="#EDF1F5", linewidth=0.65)
        ax.set_axisbelow(True)
    traffic_ax.set_ylabel("HBM traffic (GB)")
    traffic_ax.set_title("(a) Data movement")
    latency_ax.set_ylabel("Latency (ms)")
    latency_ax.set_title("(b) Execution time")
    handles, labels = traffic_ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=4, loc="outside upper center")
    fig.savefig(output, format="pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="test/eval_three_taxes/eval_three_taxes.csv")
    parser.add_argument("--output", default="test/eval_three_taxes/fig_eval_three_taxes.pdf")
    args = parser.parse_args()
    rows = read_rows(Path(args.input))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plot(rows, output)
    print(f"[write] {output}")


if __name__ == "__main__":
    main()
