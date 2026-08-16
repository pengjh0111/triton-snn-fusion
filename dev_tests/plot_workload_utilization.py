#!/usr/bin/env python3
"""Plot compact eager/compile utilization bars for a single paper column."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODEL_LABELS = {
    "resnet18": "ResNet18",
    "mamba": "Mamba",
    "spiketransformer": "SpikeTrans.",
    "convlstm": "ConvLSTM",
}
MODE_STYLE = {
    "eager": {"label": "Eager", "color": "#7B8794", "hatch": ""},
    "compile": {"label": "torch.compile", "color": "#355F8A", "hatch": "///"},
}
PANELS = (
    ("achieved_sm_pct", "SM throughput", "#4C78A8"),
    ("achieved_memory_bandwidth_pct", "DRAM bandwidth", "#3D8B74"),
    ("gpu_launch_idle_pct", "Launch / idle", "#8A6D8F"),
)


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, _, _ in PANELS:
            row[key] = float(row[key])
    return rows


def plot(rows, output: Path):
    models = list(dict.fromkeys(row["model"] for row in rows))
    by_case = {(row["model"], row["mode"]): row for row in rows}
    missing = [
        (model, mode)
        for model in models
        for mode in MODE_STYLE
        if (model, mode) not in by_case
    ]
    if missing:
        raise ValueError(f"missing cases: {missing}")

    plt.rcParams.update(
        {
            "font.size": 7.2,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.7,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(3.45, 3.55),
        sharex=True,
        gridspec_kw={"hspace": 0.13},
    )
    x = np.arange(len(models), dtype=float)
    width = 0.34
    offsets = {"eager": -width / 2, "compile": width / 2}

    for panel_index, (ax, (field, label, color)) in enumerate(zip(axes, PANELS)):
        for mode, style in MODE_STYLE.items():
            values = [by_case[(model, mode)][field] for model in models]
            bars = ax.bar(
                x + offsets[mode],
                values,
                width=width,
                label=style["label"],
                color=color if mode == "eager" else "white",
                edgecolor=color,
                hatch=style["hatch"],
                linewidth=0.75,
                zorder=3,
            )
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + (0.8 if field != "gpu_launch_idle_pct" else 0.12),
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=5.7,
                    color="#34495E",
                )
        ax.set_ylabel(f"{label}\n(% of peak)" if panel_index < 2 else f"{label} (%)")
        ax.grid(axis="y", color="#E8EDF2", linewidth=0.55, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if field == "gpu_launch_idle_pct":
            ax.set_ylim(90, 100)
            ax.set_yticks([90, 95, 100])
        else:
            ymax = max(by_case[(model, mode)][field] for model in models for mode in MODE_STYLE)
            ax.set_ylim(0, max(35, np.ceil((ymax + 7) / 10) * 10))

    axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.28),
        ncol=2,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.5,
    )
    axes[-1].set_xticks(x, [MODEL_LABELS.get(model, model) for model in models])
    axes[-1].tick_params(axis="x", pad=2)
    fig.subplots_adjust(left=0.22, right=0.99, top=0.93, bottom=0.11)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf")
    fig.savefig(output.with_suffix(".png"), dpi=300)
    plt.close(fig)


def plot_compile_only_combined(rows, output: Path):
    """One compact grouped panel: three metrics for torch.compile only."""
    compile_rows = [row for row in rows if row["mode"] == "compile"]
    models = list(dict.fromkeys(row["model"] for row in compile_rows))
    by_model = {row["model"]: row for row in compile_rows}
    if not models:
        raise ValueError("input contains no compile rows")

    metrics = (
        ("achieved_sm_pct", "SM throughput", "#3E6D9C"),
        ("achieved_memory_bandwidth_pct", "DRAM bandwidth", "#3D8B74"),
        ("gpu_launch_idle_pct", "GPU idle (% of span)", "#80658A"),
    )
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.labelsize": 11.0,
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 7.6,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(3.45, 2.5))
    x = np.arange(len(models), dtype=float)
    width = 0.245
    offsets = (-width, 0.0, width)
    for offset, (field, label, color) in zip(offsets, metrics):
        values = [by_model[model][field] for model in models]
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            label=label,
            zorder=3,
        )
        for bar, value in zip(bars, values):
            is_idle = field == "gpu_launch_idle_pct"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value - 3.0 if is_idle else value + 1.2,
                f"{value:.1f}",
                ha="center",
                va="top" if is_idle else "bottom",
                fontsize=6.4,
                color="white" if is_idle else "#33475B",
                rotation=90 if not is_idle else 0,
            )

    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("Percentage (%)")
    ax.set_xticks(x, [MODEL_LABELS.get(model, model) for model in models])
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", pad=2)
    ax.grid(axis="y", color="#E8EDF2", linewidth=0.55, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    order = [0, 2, 1]
    ax.legend(
        [handles[index] for index in order],
        [labels[index] for index in order],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        handlelength=1.15,
        handletextpad=0.35,
        columnspacing=1.0,
        borderaxespad=0,
    )
    fig.subplots_adjust(left=0.18, right=0.995, top=0.76, bottom=0.22)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf")
    fig.savefig(output.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="test/workload_utilization_elapsed/workload_utilization.csv",
    )
    parser.add_argument(
        "--output",
        default="test/fig_workload_utilization.pdf",
    )
    parser.add_argument(
        "--compile-only-combined",
        action="store_true",
        help="Plot one grouped panel with SM, DRAM, and GPU-idle compile data.",
    )
    args = parser.parse_args()
    rows = read_rows(Path(args.input))
    if args.compile_only_combined:
        plot_compile_only_combined(rows, Path(args.output))
    else:
        plot(rows, Path(args.output))
    print(f"[write] {args.output}")


if __name__ == "__main__":
    main()
