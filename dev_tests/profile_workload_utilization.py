#!/usr/bin/env python3
"""Profile compute, HBM, and GPU-idle utilization for real workloads.

The default experiment compares PyTorch eager and torch.compile/Inductor for
ResNet18, Mamba, SpikeTransformer, and ConvLSTM at T=16, batch size 4, and
FP32. Nsight Compute supplies peak-relative hardware throughput. An optional
Nsight Systems pass measures the union of GPU-kernel intervals, from which
the between-kernel idle share is derived.

Use --dry-run to validate and print profiler commands without loading models
or running any CUDA work.
"""

import argparse
import csv
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

import runtime.snn_custom_ops as snn_custom_ops
from dev_tests import eval_three_taxes as workload
from benchmarks.validate_kairos_baselines import (
    SequenceInputLoopWrapper,
    SingleStepModeLoopWrapper,
    make_model_input,
    model_input_mode,
)


MODELS = ("resnet18", "mamba", "spiketransformer", "convlstm")
MODES = ("eager", "compile")
# Use the kernel's elapsed cycles as the denominator.  The ``..._active``
# variants normalize by cycles in which the corresponding unit was already
# active; on Blackwell, gpu__dram_throughput...active is consequently 100%
# for every kernel and is not an achieved-percent-of-peak measurement.
TENSOR_METRIC = "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"
SM_METRIC = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
DRAM_METRIC = "dram__throughput.avg.pct_of_peak_sustained_elapsed"
DURATION_METRIC = "gpu__time_duration.sum"
NCU_METRICS = (TENSOR_METRIC, SM_METRIC, DRAM_METRIC, DURATION_METRIC)
CSV_COLUMNS = (
    "model",
    "mode",
    "T",
    "batch",
    "compute_metric",
    "achieved_compute_pct",
    "achieved_tensor_core_pct",
    "achieved_sm_pct",
    "achieved_memory_bandwidth_pct",
    "gpu_kernel_busy_pct",
    "gpu_launch_idle_pct",
    "kernel_count",
)


def _number(text: str) -> Optional[float]:
    text = (text or "").strip().replace(",", "")
    if not text or text.lower() in {"n/a", "nan"}:
        return None
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)
    return float(match.group(0)) if match else None


def _raw_ncu_rows(path: Path) -> List[Dict[str, str]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header = next(
        (
            index
            for index, line in enumerate(lines)
            if '"Kernel Name"' in line
            and ('"Metric Name"' in line or f'"{SM_METRIC}"' in line)
        ),
        None,
    )
    if header is None:
        raise RuntimeError(f"NCU CSV has no raw metric table: {path}")
    return list(csv.DictReader(lines[header:]))


def parse_ncu_utilization(path: Path) -> Dict[str, float]:
    rows = _raw_ncu_rows(path)
    per_launch: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    if rows and "Metric Name" in rows[0]:
        for row in rows:
            launch = (
                row.get("Process ID", ""),
                row.get("Kernel Name", ""),
                row.get("ID", row.get("Kernel ID", "")),
            )
            metric = row.get("Metric Name", "")
            value = _number(row.get("Metric Value", ""))
            if value is not None:
                per_launch.setdefault(launch, {})[metric] = value
    else:
        for index, row in enumerate(rows):
            if not row.get("Kernel Name", "").strip():
                continue
            launch = (
                row.get("Process ID", ""),
                row.get("Kernel Name", ""),
                row.get("ID", str(index)),
            )
            metrics = {
                metric: value
                for metric in NCU_METRICS
                if (value := _number(row.get(metric, ""))) is not None
            }
            if metrics:
                per_launch[launch] = metrics

    def weighted(metric: str) -> float:
        numerator = denominator = 0.0
        for metrics in per_launch.values():
            value = metrics.get(metric)
            if value is None:
                continue
            duration = metrics.get(DURATION_METRIC, 1.0)
            numerator += value * duration
            denominator += duration
        return numerator / denominator if denominator else math.nan

    tensor_pct = weighted(TENSOR_METRIC)
    sm_pct = weighted(SM_METRIC)
    return {
        "achieved_tensor_core_pct": tensor_pct,
        "achieved_sm_pct": sm_pct,
        "achieved_memory_bandwidth_pct": weighted(DRAM_METRIC),
        "kernel_count": float(len(per_launch)),
    }


def _child_command(args, model: str, mode: str) -> List[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--profile-child",
        "--profile-model",
        model,
        "--profile-mode",
        mode,
        "--T",
        str(args.T),
        "--batch",
        str(args.batch),
        "--dtype",
        args.dtype,
        "--seed",
        str(args.seed),
        "--profile-warmup",
        str(args.profile_warmup),
        "--out-dir",
        str(args.out_dir),
    ]


def _run_checked(cmd: Sequence[str], timeout: int) -> None:
    completed = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"profiler command failed with exit={completed.returncode}:\n"
            f"{' '.join(map(str, cmd))}\n{completed.stdout[-6000:]}"
        )


def _check_ncu_permissions() -> None:
    """Fail early when the NVIDIA driver restricts counters to administrators."""
    params = Path("/proc/driver/nvidia/params")
    if os.geteuid() == 0 or not params.exists():
        return
    text = params.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"^RmProfilingAdminOnly:\s*(\d+)", text, re.MULTILINE)
    if match and match.group(1) == "1":
        raise RuntimeError(
            "NCU cannot access GPU performance counters: the NVIDIA driver has "
            "RmProfilingAdminOnly=1. Run this script with `sudo -E`, or configure "
            "the driver with NVreg_RestrictProfilingToAdminUsers=0 and reload/reboot."
        )


def collect_ncu(args, model: str, mode: str) -> Tuple[Dict[str, float], List[str]]:
    _check_ncu_permissions()
    ncu = workload.resolve_ncu_path(args.ncu_path)
    if ncu is None:
        raise RuntimeError(f"NCU executable not found: {args.ncu_path}")
    report = Path(args.out_dir) / "ncu" / f"{model}_{mode}.csv"
    report.parent.mkdir(parents=True, exist_ok=True)
    def command(metrics: Sequence[str]) -> List[str]:
        return [
            ncu,
            "--target-processes",
            "all",
            "--profile-from-start",
            "off",
            "--csv",
            "--page",
            "raw",
            "--metrics",
            ",".join(metrics),
            "--log-file",
            str(report),
            *_child_command(args, model, mode),
        ]

    cmd = command(NCU_METRICS)
    try:
        _run_checked(cmd, args.timeout_sec)
    except RuntimeError as tensor_error:
        # The tensor-pipe counter is architecture-dependent. Keep the
        # experiment usable on GPUs that expose only the generic SM counter.
        fallback_metrics = (SM_METRIC, DRAM_METRIC, DURATION_METRIC)
        cmd = command(fallback_metrics)
        try:
            _run_checked(cmd, args.timeout_sec)
        except RuntimeError as fallback_error:
            report_text = (
                report.read_text(encoding="utf-8", errors="replace")
                if report.exists()
                else ""
            )
            raise RuntimeError(
                f"{fallback_error}\nNCU report output:\n{report_text[-6000:]}"
            ) from tensor_error
    return parse_ncu_utilization(report), cmd


def _kernel_table(connection: sqlite3.Connection) -> str:
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    for name in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL"):
        if name in tables:
            return name
    raise RuntimeError("Nsight Systems SQLite export contains no CUDA kernel table")


def parse_nsys_idle(sqlite_path: Path) -> Dict[str, float]:
    with sqlite3.connect(sqlite_path) as connection:
        table = _kernel_table(connection)
        intervals = [
            (int(start), int(end))
            for start, end in connection.execute(
                f'SELECT start, end FROM "{table}" WHERE end > start ORDER BY start'
            )
        ]
    if not intervals:
        return {
            "gpu_kernel_busy_pct": math.nan,
            "gpu_launch_idle_pct": math.nan,
            "nsys_kernel_count": 0.0,
        }
    merged: List[List[int]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    busy = sum(end - start for start, end in merged)
    span = merged[-1][1] - merged[0][0]
    busy_pct = 100.0 * busy / span if span else 100.0
    return {
        "gpu_kernel_busy_pct": busy_pct,
        "gpu_launch_idle_pct": max(0.0, 100.0 - busy_pct),
        "nsys_kernel_count": float(len(intervals)),
    }


def collect_nsys(args, model: str, mode: str) -> Tuple[Dict[str, float], List[str]]:
    nsys = shutil.which(args.nsys_path)
    if nsys is None:
        raise RuntimeError(f"Nsight Systems executable not found: {args.nsys_path}")
    prefix = Path(args.out_dir) / "nsys" / f"{model}_{mode}"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        nsys,
        "profile",
        "--force-overwrite=true",
        "--trace=cuda",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--export=sqlite",
        "--output",
        str(prefix),
        *_child_command(args, model, mode),
    ]
    _run_checked(cmd, args.timeout_sec)
    sqlite_path = prefix.with_suffix(".sqlite")
    if not sqlite_path.exists():
        raise RuntimeError(f"Nsight Systems did not create {sqlite_path}")
    return parse_nsys_idle(sqlite_path), cmd


def write_csv(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_baseline_runnable(args, model_name: str, mode: str):
    # Reuse the workload shapes and model factory, but deliberately bypass
    # every Kairos graph-rewrite backend. Both modes execute the same wrapper;
    # the only difference is eager dispatch versus vanilla Inductor compile.
    bench_args, _ = workload._benchmark_namespace(
        args, model_name, "per_step", profile=True
    )
    bench_args.fused_op_backend = "torch"
    dtype = workload.runtime_bench.resolve_dtype(bench_args.dtype)
    torch.manual_seed(bench_args.seed)
    torch.cuda.manual_seed_all(bench_args.seed)
    base_model = workload._make_base_model(model_name, bench_args, dtype)
    x = make_model_input(model_name, bench_args, dtype)
    wrapper_cls = (
        SequenceInputLoopWrapper
        if model_input_mode(model_name) == "sequence"
        else SingleStepModeLoopWrapper
    )
    wrapped = wrapper_cls(base_model, bench_args.T).to(
        device="cuda", dtype=dtype
    ).eval()
    snn_custom_ops.configure_fused_op(
        backend="torch",
        strict_triton=False,
        verbose=False,
        use_triton_autotune=False,
    )
    runnable = (
        wrapped
        if mode == "eager"
        else torch.compile(
            wrapped,
            backend="inductor",
            fullgraph=False,
            dynamic=False,
        )
    )
    return runnable, wrapped, x


def profile_child(args) -> None:
    runnable, model, x = build_baseline_runnable(
        args, args.profile_model, args.profile_mode
    )
    workload.warmup(runnable, model, x, args.profile_warmup)
    torch.cuda.cudart().cudaProfilerStart()
    workload.invoke(runnable, model, x)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


def main(args) -> None:
    if args.profile_child:
        profile_child(args)
        return
    cases = [(model, mode) for model in args.models for mode in args.modes]
    if args.dry_run:
        for model, mode in cases:
            child = _child_command(args, model, mode)
            report = Path(args.out_dir) / "ncu" / f"{model}_{mode}.csv"
            prefix = Path(args.out_dir) / "nsys" / f"{model}_{mode}"
            print(f"[{model}/{mode}]")
            print(
                "NCU:",
                " ".join(
                    [
                        args.ncu_path,
                        "--target-processes", "all",
                        "--profile-from-start", "off",
                        "--csv", "--page", "raw",
                        "--metrics", ",".join(NCU_METRICS),
                        "--log-file", str(report),
                        *child,
                    ]
                ),
            )
            if not args.no_nsys:
                print(
                    "Nsight Systems:",
                    " ".join(
                        [
                            args.nsys_path, "profile", "--force-overwrite=true",
                            "--trace=cuda", "--capture-range=cudaProfilerApi",
                            "--capture-range-end=stop", "--export=sqlite",
                            "--output", str(prefix), *child,
                        ]
                    ),
                )
        return

    rows = []
    commands = {}
    for model, mode in cases:
        print(f"[profile] model={model} mode={mode} T={args.T} batch={args.batch}")
        metrics, ncu_cmd = collect_ncu(args, model, mode)
        idle = {
            "gpu_kernel_busy_pct": math.nan,
            "gpu_launch_idle_pct": math.nan,
        }
        nsys_cmd = None
        if not args.no_nsys:
            idle, nsys_cmd = collect_nsys(args, model, mode)
        tensor_pct = metrics["achieved_tensor_core_pct"]
        compute_metric = "tensor_core" if math.isfinite(tensor_pct) else "sm"
        achieved_compute = (
            tensor_pct if compute_metric == "tensor_core" else metrics["achieved_sm_pct"]
        )
        row = {
            "model": model,
            "mode": mode,
            "T": args.T,
            "batch": args.batch,
            "compute_metric": compute_metric,
            "achieved_compute_pct": achieved_compute,
            **metrics,
            **idle,
        }
        rows.append(row)
        commands[f"{model}/{mode}"] = {"ncu": ncu_cmd, "nsys": nsys_cmd}
        write_csv(Path(args.out_dir) / "workload_utilization.csv", rows)
        print(
            f"  compute={achieved_compute:.2f}% ({compute_metric}) "
            f"memory={metrics['achieved_memory_bandwidth_pct']:.2f}% "
            f"idle={idle['gpu_launch_idle_pct']:.2f}%"
        )
    metadata = {
        "T": args.T,
        "batch": args.batch,
        "dtype": args.dtype,
        "models": args.models,
        "modes": args.modes,
        "ncu_metrics": list(NCU_METRICS),
        "aggregation": "kernel-duration-weighted mean",
        "utilization_denominator": "per-kernel elapsed cycles; launch/CPU gaps are reported separately as idle",
        "idle_definition": "gaps in the union of GPU kernel intervals between first kernel start and last kernel end",
        "commands": commands,
    }
    (Path(args.out_dir) / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--T", type=int, default=16)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp32")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--profile-warmup", type=int, default=2)
    parser.add_argument("--ncu-path", default="ncu")
    parser.add_argument("--nsys-path", default="nsys")
    parser.add_argument("--no-nsys", action="store_true")
    parser.add_argument("--timeout-sec", type=int, default=7200)
    parser.add_argument("--out-dir", default="test/workload_utilization")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--profile-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--profile-model", choices=MODELS, help=argparse.SUPPRESS)
    parser.add_argument("--profile-mode", choices=MODES, help=argparse.SUPPRESS)
    parser.set_defaults(disable_fused_cudagraph=True, temporal_window=None)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
