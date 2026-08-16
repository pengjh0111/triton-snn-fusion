#!/usr/bin/env python3
"""Append compact compile utilization as panel (c) to the motivation figure."""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dev_tests.plot_motivation import COL, LABEL, MARKER, read_rows, series


MODEL_LABELS = {
    "resnet18": "ResNet18",
    "mamba": "Mamba",
    "spiketransformer": "SpikeTrans.",
    "convlstm": "ConvLSTM",
}


def read_utilization(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["mode"] == "compile"]
    for row in rows:
        for field in (
            "achieved_sm_pct",
            "achieved_memory_bandwidth_pct",
            "gpu_launch_idle_pct",
        ):
            row[field] = float(row[field])
    return rows


def draw(args):
    motivation_dir = Path(args.motivation_dir)
    rows = read_rows(motivation_dir / "three_taxes_by_T.csv")
    multilayer = read_rows(motivation_dir / "three_taxes_multilayer.csv")
    utilization = read_utilization(Path(args.utilization_csv))

    style = {
        "font.size": 10.8,
        "axes.labelsize": 11.3,
        "axes.titlesize": 11.7,
        "legend.fontsize": 8.3,
        "xtick.labelsize": 9.1,
        "ytick.labelsize": 9.1,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.color": COL["ink"],
        "axes.labelcolor": COL["ink"],
        "axes.titlecolor": COL["ink"],
        "xtick.color": COL["ink"],
        "ytick.color": COL["ink"],
    }
    with plt.rc_context(style):
        fig = plt.figure(figsize=(7.15, 2.28))
        grid = fig.add_gridspec(
            1, 3, width_ratios=(1.08, 1.08, 1.0), wspace=0.68,
            left=0.07, right=0.995, bottom=0.24, top=0.64,
        )
        tax_ax, stack_ax, util_ax = (fig.add_subplot(grid[0, i]) for i in range(3))

        # (a) Temporal accumulation, matching the original combined figure.
        for mode in ("per_step", "batched", "fused"):
            x, traffic = series(rows, mode, "dram_total_bytes")
            tax_ax.plot(
                x, traffic / 1e9, marker=MARKER[mode], markersize=4.2,
                linewidth=1.15, color=COL[mode], label=LABEL[mode],
            )
        theory = sorted(
            (row for row in rows if row["mode"] == "per_step"),
            key=lambda row: row["T"],
        )
        timesteps = np.array([row["T"] for row in theory])
        per_x, per_traffic = series(rows, "per_step", "dram_total_bytes")
        _, fused_traffic = series(rows, "fused", "dram_total_bytes")
        tax_ax.fill_between(
            per_x, fused_traffic / 1e9, per_traffic / 1e9,
            color=COL["spatial_fill"], alpha=0.75, linewidth=0,
        )
        tax_ax.set_xscale("log", base=2)
        tax_ax.set_xticks(timesteps, [str(value) for value in timesteps])
        tax_ax.set_xlabel("Timesteps (T)", labelpad=2)
        tax_ax.set_ylabel("HBM (GB)", labelpad=2)
        tax_ax.set_title("(a) Temporal accumulation", pad=3)
        tax_ax.grid(True, color=COL["grid"], linewidth=0.5)
        tax_launch = tax_ax.twinx()
        for mode, linestyle in (("per_step", ":"), ("fused", "-.")):
            x, launches = series(rows, mode, "kernel_count")
            tax_launch.plot(
                x, launches, linestyle, marker=MARKER[mode], markersize=3.8,
                linewidth=1.0, color=COL[mode], alpha=0.72,
                label=f"{LABEL[mode]} launches",
            )
        tax_launch.set_yscale("log")
        tax_launch.set_yticks([1, 10, 100, 1000])
        tax_launch.set_ylabel(
            "Launches", color=COL["muted"], labelpad=0, fontsize=11.3
        )
        tax_launch.tick_params(
            axis="y", colors=COL["muted"], pad=1, labelsize=9.1
        )

        # (b) Cross-layer accumulation.
        for mode in ("per_step", "batched", "fused"):
            selected = sorted(
                (row for row in multilayer if row["mode"] == mode),
                key=lambda row: row["layer_count"],
            )
            stack_ax.plot(
                [row["layer_count"] for row in selected],
                [row["dram_total_bytes"] / 1e9 for row in selected],
                marker=MARKER[mode], markersize=4.2, linewidth=1.15,
                color=COL[mode], label=LABEL[mode],
            )
        per_stack = sorted(
            (row for row in multilayer if row["mode"] == "per_step"),
            key=lambda row: row["layer_count"],
        )
        fused_stack = sorted(
            (row for row in multilayer if row["mode"] == "fused"),
            key=lambda row: row["layer_count"],
        )
        depths = np.array([row["layer_count"] for row in per_stack])
        stack_ax.fill_between(
            depths,
            np.array([row["dram_total_bytes"] for row in fused_stack]) / 1e9,
            np.array([row["dram_total_bytes"] for row in per_stack]) / 1e9,
            color=COL["spatial_fill"], alpha=0.75, linewidth=0,
        )
        stack_ax.set_xticks(depths)
        stack_ax.set_xlabel("Stack depth (layers)", labelpad=2)
        stack_ax.set_ylabel("HBM (GB)", labelpad=2)
        stack_ax.set_title("(b) Cross-layer accumulation", pad=3)
        stack_ax.grid(True, color=COL["grid"], linewidth=0.5)
        stack_launch = stack_ax.twinx()
        for mode, linestyle in (("per_step", ":"), ("fused", "-.")):
            selected = sorted(
                (row for row in multilayer if row["mode"] == mode),
                key=lambda row: row["layer_count"],
            )
            stack_launch.plot(
                [row["layer_count"] for row in selected],
                [row["kernel_count"] for row in selected],
                linestyle, marker=MARKER[mode], markersize=3.8,
                linewidth=1.0, color=COL[mode], alpha=0.72,
            )
        stack_launch.set_yscale("log")
        stack_launch.set_yticks([1, 10, 100, 1000])
        stack_launch.set_ylabel(
            "Launches", color=COL["muted"], labelpad=0, fontsize=11.3
        )
        stack_launch.tick_params(
            axis="y", colors=COL["muted"], pad=1, labelsize=9.1
        )
        # Shift both axes of panel (b) together; doing this before twinx()
        # would let the secondary axis snap back to the GridSpec position.
        stack_pos = stack_ax.get_position()
        shifted_stack_pos = [
            stack_pos.x0 - 0.012,
            stack_pos.y0,
            stack_pos.width,
            stack_pos.height,
        ]
        stack_ax.set_position(shifted_stack_pos)
        stack_launch.set_position(shifted_stack_pos)

        # (c) Compile-only utilization, exactly the same axes height as (a)/(b).
        metrics = (
            ("achieved_sm_pct", "SM throughput", "#3E6D9C"),
            ("achieved_memory_bandwidth_pct", "DRAM bandwidth", "#3D8B74"),
            ("gpu_launch_idle_pct", "GPU idle (% of span)", "#80658A"),
        )
        ux = np.arange(len(utilization), dtype=float)
        width = 0.25
        for offset, (field, label, color) in zip((-width, 0, width), metrics):
            util_ax.bar(
                ux + offset, [row[field] for row in utilization], width,
                color=color, edgecolor="white", linewidth=0.35,
                label=label, zorder=3,
            )
        util_ax.set_ylim(0, 100)
        util_ax.set_yticks([0, 20, 40, 60, 80, 100])
        util_ax.set_ylabel("Percentage (%)", labelpad=2)
        util_ax.set_xticks(
            ux, [MODEL_LABELS.get(row["model"], row["model"]) for row in utilization]
        )
        plt.setp(
            util_ax.get_xticklabels(), rotation=32, ha="right", rotation_mode="anchor",
            fontsize=7.7,
        )
        util_ax.set_title("(c) Utilization", pad=3)
        util_ax.grid(axis="y", color=COL["grid"], linewidth=0.5, zorder=0)
        util_ax.spines["top"].set_visible(False)
        util_ax.spines["right"].set_visible(False)

        mode_handles, mode_labels = tax_ax.get_legend_handles_labels()
        launch_handles, launch_labels = tax_launch.get_legend_handles_labels()
        util_handles, util_labels = util_ax.get_legend_handles_labels()
        fig.legend(
            mode_handles + launch_handles + util_handles,
            mode_labels + launch_labels + util_labels,
            loc="upper center", bbox_to_anchor=(0.5, 0.995),
            ncol=4, frameon=False, columnspacing=1.1,
            handlelength=1.5, handletextpad=0.4,
        )

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, format="pdf")
        fig.savefig(output.with_suffix(".png"), dpi=300)
        plt.close(fig)
        print(f"[write] {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--motivation-dir", default="test/motivation_three_taxes"
    )
    parser.add_argument(
        "--utilization-csv",
        default="test/workload_utilization_elapsed/workload_utilization.csv",
    )
    parser.add_argument(
        "--output",
        default="test/fig_motivation_combined_with_utilization.pdf",
    )
    draw(parser.parse_args())


if __name__ == "__main__":
    main()
