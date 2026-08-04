#!/usr/bin/env python3
"""Plot best searched runtime for Kairos W/P/R ablations."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODELS = ("resnet18", "spiketransformer", "mamba", "convlstm")
VARIANTS = ("full", "P1", "R1", "W1")
LABELS = {"full": "Full", "P1": "P=1", "R1": "R=1", "W1": "W=1"}
COLORS = {"full": "#2E7D6F", "P1": "#2C5F8A", "R1": "#6B4E8F", "W1": "#8A9199"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="test/eval_tiling_ablation/eval_tiling_ablation.csv")
    parser.add_argument("--output", default="test/eval_tiling_ablation/fig_tiling_ablation.pdf")
    args = parser.parse_args()
    with Path(args.input).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    lookup = {(row["model"], row["variant"]): float(row["slowdown_vs_full"]) for row in rows}
    missing = [(m, v) for m in MODELS for v in VARIANTS if (m, v) not in lookup]
    if missing:
        raise ValueError(f"missing ablation rows: {missing}")
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8.5, "legend.fontsize": 7.5, "pdf.fonttype": 42})
    x = np.arange(len(MODELS))
    width = 0.19
    fig, ax = plt.subplots(figsize=(7.0, 2.55), constrained_layout=True)
    for index, variant in enumerate(VARIANTS):
        values = [lookup[(model, variant)] for model in MODELS]
        ax.bar(
            x + (index - 1.5) * width,
            values,
            width,
            color=COLORS[variant],
            edgecolor="white",
            linewidth=0.6,
            label=LABELS[variant],
        )
    ax.axhline(1.0, color="#33475B", linewidth=1.0)
    ax.set_xticks(x, MODELS)
    ax.set_ylabel("Runtime normalized to searched full config")
    ax.grid(axis="y", color="#EDF1F5", linewidth=0.65)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=4, loc="upper center")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf")
    plt.close(fig)
    print(f"[write] {output}")


if __name__ == "__main__":
    main()
