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
from matplotlib.lines import Line2D


COL = {
    "per_step": "#2C5F8A",
    "compiled": "#7C7067",
    "batched": "#2E7D6F",
    "fused": "#6B4E8F",
    "theory": "#8A9199",
    "roofline": "#33475B",
    "accent": "#882255",
    "spatial_fill": "#DCE9F5",
    "temporal_fill": "#E6DFF0",
    "fused_fill": "#EAF4F1",
    "ink": "#33475B",
    "muted": "#8A9199",
    "grid": "#EDF1F5",
}
LABEL = {
    "per_step": "Per-step",
    "compiled": "torch.compile",
    "batched": "Batched-only",
    "fused": "Fused",
}
MARKER = {"per_step": "o", "compiled": "D", "batched": "s", "fused": "^"}


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


def expected_flops(metadata, timesteps, layers=1):
    problem = metadata["problem"]
    kernel = problem.get("kernel", 3)
    return (
        2
        * problem["cout"]
        * problem["cin"]
        * kernel
        * kernel
        * problem["height"]
        * problem["width"]
        * timesteps
        * problem["batch"]
        * layers
    )


def audit_derived_metrics(rows, metadata, layer_field=None):
    for row in rows:
        layers = int(row[layer_field]) if layer_field else 1
        formula_flops = expected_flops(metadata, row["T"], layers)
        csv_flops = row.get("flops_total", math.nan)
        if math.isfinite(csv_flops) and not math.isclose(
            csv_flops, formula_flops, rel_tol=0, abs_tol=0
        ):
            raise ValueError(
                f"FLOP mismatch for {row['mode']} T={row['T']} L={layers}: "
                f"CSV={csv_flops}, formula={formula_flops}"
            )
        row["flops_total"] = float(formula_flops)
        traffic = row["dram_total_bytes"]
        latency = row["time_ms_mean"]
        row["arith_intensity"] = (
            formula_flops / traffic
            if math.isfinite(traffic) and traffic > 0
            else math.nan
        )
        row["achieved_tflops"] = (
            formula_flops / (latency * 1e9)
            if math.isfinite(latency) and latency > 0
            else math.nan
        )


def print_roofline_audit(rows, metadata, representative_t):
    problem = metadata["problem"]
    print(
        "[FLOP audit] "
        f"T={representative_t} B={problem['batch']} "
        f"Cin={problem['cin']} Cout={problem['cout']} "
        f"K={problem.get('kernel', 3)} Hout={problem['height']} "
        f"Wout={problem['width']} "
        f"FLOP={expected_flops(metadata, representative_t) / 1e9:.6f} GFLOP"
    )
    for row in rows:
        if row["T"] != representative_t:
            continue
        print(
            f"  {row['mode']}: DRAM={row['dram_total_bytes'] / 1e6:.6f} MB "
            f"AI={row['arith_intensity']:.6f} FLOP/B "
            f"time={row['time_ms_mean']:.6f} ms "
            f"performance={row['achieved_tflops']:.6f} TFLOP/s"
        )


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
            "axes.edgecolor": COL["ink"],
            "axes.labelcolor": COL["ink"],
            "axes.titlecolor": COL["ink"],
            "text.color": COL["ink"],
            "xtick.color": COL["ink"],
            "ytick.color": COL["ink"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_taxes(rows, output: Path):
    fig, ax = plt.subplots(figsize=(3.45, 2.45), constrained_layout=True)
    complete = any(math.isfinite(row["dram_total_bytes"]) for row in rows)
    modes = tuple(
        mode
        for mode in ("per_step", "compiled", "batched", "fused")
        if any(row["mode"] == mode for row in rows)
    )
    for mode in modes:
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
    ax.grid(True, color=COL["grid"], linewidth=0.55)

    kernels = ax.twinx()
    for mode, linestyle in (
        ("per_step", ":"),
        ("compiled", "--"),
        ("fused", "-."),
    ):
        if mode not in modes:
            continue
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
        performance = row["achieved_tflops"]
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
    ax.grid(True, which="both", color=COL["grid"], linewidth=0.55)
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


def plot_multilayer(rows, output: Path):
    fig, (traffic_ax, time_ax) = plt.subplots(
        1, 2, figsize=(7.0, 2.4), constrained_layout=True
    )
    modes = tuple(
        mode
        for mode in ("per_step", "compiled", "batched", "fused")
        if any(row["mode"] == mode for row in rows)
    )
    for mode in modes:
        selected = sorted(
            (row for row in rows if row["mode"] == mode),
            key=lambda row: row["layer_count"],
        )
        x = np.array([row["layer_count"] for row in selected])
        traffic = np.array([row["dram_total_bytes"] for row in selected])
        latency = np.array([row["time_ms_mean"] for row in selected])
        traffic_ax.plot(
            x,
            traffic / 1e9,
            marker=MARKER[mode],
            color=COL[mode],
            label=LABEL[mode],
        )
        time_ax.plot(
            x,
            latency,
            marker=MARKER[mode],
            color=COL[mode],
            label=LABEL[mode],
        )
    theory = sorted(
        (row for row in rows if row["mode"] == "per_step"),
        key=lambda row: row["layer_count"],
    )
    traffic_ax.plot(
        [row["layer_count"] for row in theory],
        [row["min_bytes_theory"] / 1e9 for row in theory],
        "--",
        color=COL["theory"],
        label="HBM lower bound",
    )
    for ax in (traffic_ax, time_ax):
        ax.set_xlabel("Stack depth (layers)")
        ax.set_xticks(sorted({int(row["layer_count"]) for row in rows}))
        ax.grid(True, color=COL["grid"], linewidth=0.55)
    traffic_ax.set_ylabel("HBM traffic (GB)")
    traffic_ax.set_title("(a) Main evidence: HBM traffic")
    time_ax.set_ylabel("Latency (ms)")
    time_ax.set_title("(b) Auxiliary: end-to-end latency")
    traffic_ax.legend(frameon=False, loc="upper left")
    fig.savefig(output, format="pdf")
    plt.close(fig)


def plot_combined(
    rows,
    multilayer_rows,
    output: Path,
    compiled_rows=None,
    compiled_multilayer_rows=None,
    compiled_config=None,
):
    """Combine the timestep tax plot and multilayer traffic plot in one row."""
    combined_style = {
        "font.size": 13.0625,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 12.75,
        "xtick.labelsize": 12.125,
        "ytick.labelsize": 12.125,
    }
    with plt.rc_context(combined_style):
        fig, (tax_ax, stack_ax) = plt.subplots(
            1, 2, figsize=(7.15, 2.9), constrained_layout=True
        )

        for mode in ("per_step", "batched", "fused"):
            x, traffic = series(rows, mode, "dram_total_bytes")
            tax_ax.plot(
                x,
                traffic / 1e9,
                marker=MARKER[mode],
                color=COL[mode],
                label=LABEL[mode],
            )
        if compiled_rows:
            x, traffic = series(compiled_rows, "compiled", "dram_total_bytes")
            tax_ax.plot(
                x,
                traffic / 1e9,
                linestyle="none",
                marker=MARKER["compiled"],
                markersize=6.5,
                color=COL["compiled"],
                label=LABEL["compiled"],
                zorder=5,
            )
        theory_rows = sorted(
            (row for row in rows if row["mode"] == "per_step"),
            key=lambda row: row["T"],
        )
        timesteps = np.array([row["T"] for row in theory_rows])
        per_x, per_traffic = series(rows, "per_step", "dram_total_bytes")
        _, fused_traffic = series(rows, "fused", "dram_total_bytes")
        if np.all(np.isfinite(per_traffic)) and np.all(np.isfinite(fused_traffic)):
            tax_ax.fill_between(
                per_x,
                fused_traffic / 1e9,
                per_traffic / 1e9,
                color=COL["spatial_fill"],
                alpha=0.75,
                linewidth=0,
            )
        tax_ax.set_xscale("log", base=2)
        tax_ax.set_xticks(timesteps, [str(value) for value in timesteps])
        tax_ax.set_xlabel("Timesteps (T)")
        tax_ax.set_ylabel("HBM traffic (GB)")
        tax_ax.set_yticks([0, 1, 2])
        tax_ax.set_title("(a) Temporal accumulation")
        tax_ax.grid(True, color=COL["grid"], linewidth=0.55)

        launches_ax = tax_ax.twinx()
        for mode, linestyle in (("per_step", ":"), ("fused", "-.")):
            x, launches = series(rows, mode, "kernel_count")
            launches_ax.plot(
                x,
                launches,
                linestyle,
                marker=MARKER[mode],
                color=COL[mode],
                alpha=0.72,
                label=f"{LABEL[mode]} launches",
            )
        if compiled_rows:
            x, launches = series(compiled_rows, "compiled", "kernel_count")
            launches_ax.plot(
                x,
                launches,
                linestyle="none",
                marker=MARKER["compiled"],
                markersize=6.5,
                color=COL["compiled"],
                alpha=0.9,
                label=f"{LABEL['compiled']} launches",
                zorder=5,
            )
        launches_ax.set_yscale("log")
        launches_ax.set_yticks([1, 10, 100, 1000])
        launches_ax.set_ylabel("Kernel launches", color=COL["muted"])
        launches_ax.tick_params(axis="y", colors=COL["muted"], labelsize=12.125)

        for mode in ("per_step", "batched", "fused"):
            selected = sorted(
                (row for row in multilayer_rows if row["mode"] == mode),
                key=lambda row: row["layer_count"],
            )
            depths = np.array([row["layer_count"] for row in selected])
            traffic = np.array([row["dram_total_bytes"] for row in selected])
            stack_ax.plot(
                depths,
                traffic / 1e9,
                marker=MARKER[mode],
                color=COL[mode],
                label=LABEL[mode],
            )
        if compiled_multilayer_rows:
            selected = sorted(
                (
                    row
                    for row in compiled_multilayer_rows
                    if row["mode"] == "compiled"
                ),
                key=lambda row: row["layer_count"],
            )
            stack_ax.plot(
                [row["layer_count"] for row in selected],
                [row["dram_total_bytes"] / 1e9 for row in selected],
                marker=MARKER["compiled"],
                color=COL["compiled"],
                label=LABEL["compiled"],
                zorder=5,
            )
        per_stack = sorted(
            (row for row in multilayer_rows if row["mode"] == "per_step"),
            key=lambda row: row["layer_count"],
        )
        fused_stack = sorted(
            (row for row in multilayer_rows if row["mode"] == "fused"),
            key=lambda row: row["layer_count"],
        )
        stack_depths = np.array([row["layer_count"] for row in per_stack])
        per_stack_traffic = np.array(
            [row["dram_total_bytes"] for row in per_stack]
        )
        fused_stack_traffic = np.array(
            [row["dram_total_bytes"] for row in fused_stack]
        )
        if (
            np.array_equal(
                stack_depths,
                np.array([row["layer_count"] for row in fused_stack]),
            )
            and np.all(np.isfinite(per_stack_traffic))
            and np.all(np.isfinite(fused_stack_traffic))
        ):
            stack_ax.fill_between(
                stack_depths,
                fused_stack_traffic / 1e9,
                per_stack_traffic / 1e9,
                color=COL["spatial_fill"],
                alpha=0.75,
                linewidth=0,
            )
        stack_ax.set_xlabel("Stack depth (layers)")
        stack_ax.set_ylabel("HBM traffic (GB)")
        stack_ax.set_yticks([0, 1, 2])
        stack_ax.set_title("(b) Cross-layer accumulation")
        stack_ax.set_xticks(
            sorted({int(row["layer_count"]) for row in multilayer_rows})
        )
        stack_ax.grid(True, color=COL["grid"], linewidth=0.55)

        stack_launches_ax = stack_ax.twinx()
        for mode, linestyle in (("per_step", ":"), ("fused", "-.")):
            selected = sorted(
                (row for row in multilayer_rows if row["mode"] == mode),
                key=lambda row: row["layer_count"],
            )
            stack_launches_ax.plot(
                [row["layer_count"] for row in selected],
                [row["kernel_count"] for row in selected],
                linestyle,
                marker=MARKER[mode],
                color=COL[mode],
                alpha=0.72,
            )
        if compiled_multilayer_rows:
            selected = sorted(
                (
                    row
                    for row in compiled_multilayer_rows
                    if row["mode"] == "compiled"
                ),
                key=lambda row: row["layer_count"],
            )
            stack_launches_ax.plot(
                [row["layer_count"] for row in selected],
                [row["kernel_count"] for row in selected],
                linestyle="--",
                marker=MARKER["compiled"],
                color=COL["compiled"],
                alpha=0.9,
                label=f"{LABEL['compiled']} launches",
                zorder=5,
            )
        stack_launches_ax.set_yscale("log")
        stack_launches_ax.set_yticks([1, 10, 100, 1000])
        stack_launches_ax.set_ylabel("Kernel launches", color=COL["muted"])
        stack_launches_ax.tick_params(
            axis="y", colors=COL["muted"], labelsize=12.125
        )

        handles, labels = tax_ax.get_legend_handles_labels()
        launch_handles, launch_labels = launches_ax.get_legend_handles_labels()
        stack_handles, stack_labels = stack_ax.get_legend_handles_labels()
        combined = {}
        for handle, label in zip(
            handles + launch_handles + stack_handles,
            labels + launch_labels + stack_labels,
        ):
            combined.setdefault(label, handle)
        legend_handles = list(combined.values())
        legend_labels = list(combined.keys())
        if compiled_config:
            fig.text(
                0.5,
                0.006,
                compiled_config,
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=COL["muted"],
            )
        fig.legend(
            legend_handles,
            legend_labels,
            loc="outside upper center",
            ncol=3,
            frameon=False,
        )
        fig.savefig(output, format="pdf")
        plt.close(fig)


def plot_combined_no_launches(rows, multilayer_rows, output: Path):
    """Combined HBM-traffic plot without launch-count overlays."""
    style = {
        "font.size": 13.0625,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 12.75,
        "xtick.labelsize": 12.125,
        "ytick.labelsize": 12.125,
    }
    modes = tuple(
        mode
        for mode in ("per_step", "compiled", "batched", "fused")
        if any(row["mode"] == mode for row in rows)
    )
    with plt.rc_context(style):
        fig, (tax_ax, stack_ax) = plt.subplots(
            1, 2, figsize=(7.15, 2.55), constrained_layout=True
        )
        for mode in modes:
            x, traffic = series(rows, mode, "dram_total_bytes")
            tax_ax.plot(
                x, traffic / 1e9,
                marker=MARKER[mode], markersize=4.0, linewidth=1.1,
                color=COL[mode], label=LABEL[mode],
                zorder=5 if mode == "compiled" else 3,
            )
        theory_rows = sorted(
            (row for row in rows if row["mode"] == "per_step"),
            key=lambda row: row["T"],
        )
        timesteps = np.array([row["T"] for row in theory_rows])
        per_x, per_traffic = series(rows, "per_step", "dram_total_bytes")
        _, fused_traffic = series(rows, "fused", "dram_total_bytes")
        if np.all(np.isfinite(per_traffic)) and np.all(np.isfinite(fused_traffic)):
            tax_ax.fill_between(
                per_x, fused_traffic / 1e9, per_traffic / 1e9,
                color=COL["spatial_fill"], alpha=0.75, linewidth=0,
            )
        tax_ax.set_xscale("log", base=2)
        tax_ax.set_xticks(timesteps, [str(value) for value in timesteps])
        tax_ax.set_xlabel("Timesteps (T)")
        tax_ax.set_ylabel("HBM traffic (GB)")
        tax_ax.set_title("(a) Temporal accumulation")
        tax_ax.grid(True, color=COL["grid"], linewidth=0.55)

        for mode in modes:
            selected = sorted(
                (row for row in multilayer_rows if row["mode"] == mode),
                key=lambda row: row["layer_count"],
            )
            if not selected:
                continue
            stack_ax.plot(
                [row["layer_count"] for row in selected],
                [row["dram_total_bytes"] / 1e9 for row in selected],
                marker=MARKER[mode], markersize=4.0, linewidth=1.1,
                color=COL[mode], label=LABEL[mode],
                zorder=5 if mode == "compiled" else 3,
            )
        per_stack = sorted(
            (row for row in multilayer_rows if row["mode"] == "per_step"),
            key=lambda row: row["layer_count"],
        )
        fused_stack = sorted(
            (row for row in multilayer_rows if row["mode"] == "fused"),
            key=lambda row: row["layer_count"],
        )
        depths = np.array([row["layer_count"] for row in per_stack])
        if np.array_equal(depths, np.array([row["layer_count"] for row in fused_stack])):
            stack_ax.fill_between(
                depths,
                np.array([row["dram_total_bytes"] for row in fused_stack]) / 1e9,
                np.array([row["dram_total_bytes"] for row in per_stack]) / 1e9,
                color=COL["spatial_fill"], alpha=0.75, linewidth=0,
            )
        stack_ax.set_xlabel("Stack depth (layers)")
        stack_ax.set_ylabel("HBM traffic (GB)")
        stack_ax.set_title("(b) Cross-layer accumulation")
        stack_ax.set_xticks(sorted({int(row["layer_count"]) for row in multilayer_rows}))
        stack_ax.grid(True, color=COL["grid"], linewidth=0.55)

        handles, labels = tax_ax.get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="outside upper center", ncol=4, frameon=False
        )
        fig.savefig(output, format="pdf")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="test/motivation_three_taxes")
    parser.add_argument("--representative-t", type=int, default=16)
    parser.add_argument(
        "--compiled-data-dir",
        default=None,
        help="Optional directory containing newly collected compiled rows.",
    )
    parser.add_argument(
        "--combined-output",
        default=None,
        help="Output path for the combined plot; defaults to the legacy filename.",
    )
    parser.add_argument(
        "--combined-only",
        action="store_true",
        help="Only render the combined plot, preserving the other PDFs.",
    )
    parser.add_argument(
        "--combined-no-launches",
        action="store_true",
        help="Render the combined plot without launch-count curves or axes.",
    )
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    rows = read_rows(input_dir / "three_taxes_by_T.csv")
    metadata = json.loads((input_dir / "metadata.json").read_text(encoding="utf-8"))
    audit_derived_metrics(rows, metadata)
    print_roofline_audit(rows, metadata, args.representative_t)
    configure()
    if not args.combined_only:
        plot_taxes(rows, input_dir / "fig_motivation_taxes.pdf")
        plot_roofline(
            rows,
            metadata,
            input_dir / "fig_motivation_roofline.pdf",
            args.representative_t,
        )
    multilayer_path = input_dir / "three_taxes_multilayer.csv"
    if multilayer_path.exists():
        multilayer_rows = read_rows(multilayer_path)
        audit_derived_metrics(multilayer_rows, metadata, layer_field="layer_count")
        if not args.combined_only:
            plot_multilayer(
                multilayer_rows, input_dir / "fig_motivation_multilayer.pdf"
            )
        combined_output = (
            Path(args.combined_output)
            if args.combined_output
            else input_dir / "fig_motivation_combined.pdf"
        )
        compiled_rows = (
            rows if any(row["mode"] == "compiled" for row in rows) else None
        )
        compiled_multilayer_rows = (
            multilayer_rows
            if any(row["mode"] == "compiled" for row in multilayer_rows)
            else None
        )
        compiled_config = None
        if args.compiled_data_dir:
            compiled_dir = Path(args.compiled_data_dir)
            compiled_rows = read_rows(compiled_dir / "three_taxes_by_T.csv")
            compiled_multilayer_rows = read_rows(
                compiled_dir / "three_taxes_multilayer.csv"
            )
            compiled_metadata = json.loads(
                (compiled_dir / "metadata.json").read_text(encoding="utf-8")
            )
            audit_derived_metrics(compiled_rows, compiled_metadata)
            audit_derived_metrics(
                compiled_multilayer_rows,
                compiled_metadata,
                layer_field="layer_count",
            )
            problem = compiled_metadata["problem"]
            compiled_config = (
                "torch.compile measurements: "
                f"{compiled_metadata['dtype'].upper()}, B={problem['batch']}, "
                f"T={compiled_metadata['multilayer_t']}"
            )
        if args.combined_no_launches:
            plot_combined_no_launches(rows, multilayer_rows, combined_output)
        else:
            plot_combined(
                rows,
                multilayer_rows,
                combined_output,
                compiled_rows=compiled_rows,
                compiled_multilayer_rows=compiled_multilayer_rows,
                compiled_config=compiled_config,
            )
    if not args.combined_only:
        print(f"[write] {input_dir / 'fig_motivation_taxes.pdf'}")
        print(f"[write] {input_dir / 'fig_motivation_roofline.pdf'}")
    if multilayer_path.exists():
        if not args.combined_only:
            print(f"[write] {input_dir / 'fig_motivation_multilayer.pdf'}")
        print(f"[write] {combined_output}")


if __name__ == "__main__":
    main()

# python dev_tests/plot_motivation.py \
#   --input-dir test/motivation_three_taxes \
#   --representative-t 16
