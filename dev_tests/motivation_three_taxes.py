#!/usr/bin/env python3
"""Measure the three memory/launch taxes of step-wise Conv-BN-LIF execution.

The script has two execution layers:
* the parent benchmarks correctness and CUDA-event latency;
* an NCU child executes exactly one profiled forward after compilation/warmup.

BatchNorm is folded into convolution parameters, as it is for inference in
Kairos. All three modes therefore consume identical folded weights and bias.
"""

import argparse
import atexit
import csv
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

from runtime.snn_custom_ops import lif_forward_state_torch
from runtime.triton_convlif_backend import (
    run_triton_fused_temporal_conv_lif_state_packed_out,
)
from runtime.triton_temporal_lif_backend import run_triton_fused_temporal_lif_state


MODES = ("per_step", "batched", "fused")
NCU_METRICS = (
    "dram__bytes_op_read.sum",
    "dram__bytes_op_write.sum",
    "sm__throughput.avg.pct_of_peak_sustained_active",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_active",
    "gpu__time_duration.sum",
)
CSV_COLUMNS = (
    "mode",
    "T",
    "dram_read_bytes",
    "dram_write_bytes",
    "dram_total_bytes",
    "min_bytes_theory",
    "kernel_count",
    "arith_intensity",
    "sm_tput_pct",
    "dram_tput_pct",
    "time_ms_mean",
    "time_ms_std",
)


@dataclass
class Problem:
    batch: int
    cin: int
    cout: int
    height: int
    width: int
    kernel: int = 3
    padding: int = 1

    @property
    def output_elements_per_step(self) -> int:
        return self.batch * self.cout * self.height * self.width

    @property
    def weight_elements(self) -> int:
        return self.cout * self.cin * self.kernel * self.kernel

    def flops(self, timesteps: int) -> int:
        return (
            2
            * self.cout
            * self.height
            * self.width
            * self.cin
            * self.kernel
            * self.kernel
            * timesteps
            * self.batch
        )

    def min_bytes(self, timesteps: int, element_bytes: int) -> int:
        input_elems = timesteps * self.batch * self.cin * self.height * self.width
        output_elems = timesteps * self.output_elements_per_step
        return (input_elems + output_elems + self.weight_elements) * element_bytes


@dataclass
class Inputs:
    x: torch.Tensor
    weight: torch.Tensor
    bias: torch.Tensor
    v0: torch.Tensor


def dtype_from_name(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "fp32": torch.float32}[name]


def fold_batch_norm(
    weight: torch.Tensor,
    conv_bias: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    scale = gamma * torch.rsqrt(running_var + eps)
    folded_weight = weight * scale[:, None, None, None]
    folded_bias = beta + (conv_bias - running_mean) * scale
    return folded_weight.contiguous(), folded_bias.contiguous()


def make_inputs(problem: Problem, timesteps: int, dtype: torch.dtype, device: str, seed: int) -> Inputs:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    scale = 0.02
    x = torch.randn(
        timesteps,
        problem.batch,
        problem.cin,
        problem.height,
        problem.width,
        generator=generator,
        device=device,
        dtype=dtype,
    ) * scale
    weight = torch.randn(
        problem.cout,
        problem.cin,
        problem.kernel,
        problem.kernel,
        generator=generator,
        device=device,
        dtype=dtype,
    ) * scale
    conv_bias = torch.randn(
        problem.cout, generator=generator, device=device, dtype=dtype
    ) * scale
    gamma = 0.9 + torch.rand(
        problem.cout, generator=generator, device=device, dtype=dtype
    ) * 0.2
    beta = torch.randn(
        problem.cout, generator=generator, device=device, dtype=dtype
    ) * scale
    running_mean = torch.randn(
        problem.cout, generator=generator, device=device, dtype=dtype
    ) * scale
    running_var = 0.5 + torch.rand(
        problem.cout, generator=generator, device=device, dtype=dtype
    )
    weight, bias = fold_batch_norm(
        weight, conv_bias, gamma, beta, running_mean, running_var, eps=1e-5
    )
    v0 = torch.zeros(
        problem.batch,
        problem.cout,
        problem.height,
        problem.width,
        device=device,
        dtype=dtype,
    )
    return Inputs(x.contiguous(), weight, bias, v0)


def per_step_forward(data: Inputs, problem: Problem) -> Tuple[torch.Tensor, torch.Tensor]:
    v = data.v0
    spikes = []
    for step in range(int(data.x.shape[0])):
        preact = F.conv2d(
            data.x[step],
            data.weight,
            data.bias,
            stride=1,
            padding=problem.padding,
        )
        spike, v = lif_forward_state_torch(preact, v, 1.0, 0.0, 2.0, False)
        spikes.append(spike)
    return torch.stack(spikes), v


def temporal_ranges(timesteps: int, max_window: int):
    """Decompose T into supported power-of-two windows, largest first."""
    start = 0
    while start < timesteps:
        remaining = timesteps - start
        size = min(max_window, 1 << (remaining.bit_length() - 1))
        yield start, start + size
        start += size


def _lif_in_windows(preact: torch.Tensor, v0: torch.Tensor, window: int):
    v = v0
    chunks = []
    for start, end in temporal_ranges(int(preact.shape[0]), window):
        result = run_triton_fused_temporal_lif_state(
            preact[start:end],
            v,
            1.0,
            0.0,
            2.0,
            False,
            strict=True,
            use_autotune=False,
        )
        chunks.append(result.spikes)
        v = result.v_next
    return (chunks[0] if len(chunks) == 1 else torch.cat(chunks)), v


def batched_forward(
    data: Inputs, problem: Problem, temporal_window: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    timesteps, batch = data.x.shape[:2]
    preact = F.conv2d(
        data.x.flatten(0, 1),
        data.weight,
        data.bias,
        stride=1,
        padding=problem.padding,
    ).view(timesteps, batch, problem.cout, problem.height, problem.width)
    return _lif_in_windows(preact, data.v0, temporal_window)


def fused_forward(
    data: Inputs, problem: Problem, temporal_window: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    timesteps = int(data.x.shape[0])
    spikes = torch.empty(
        timesteps,
        problem.batch,
        problem.cout,
        problem.height,
        problem.width,
        device=data.x.device,
        dtype=data.x.dtype,
    )
    v = data.v0
    for start, end in temporal_ranges(timesteps, temporal_window):
        v_out = torch.empty_like(v)
        result = run_triton_fused_temporal_conv_lif_state_packed_out(
            data.x[start:end],
            data.weight,
            data.bias,
            v,
            [1, 1],
            [problem.padding, problem.padding],
            [1, 1],
            1,
            1.0,
            0.0,
            2.0,
            False,
            spikes[start:end],
            v_out,
            strict=True,
            use_autotune=False,
        )
        v = result.v_next
    return spikes, v


def mode_callable(
    mode: str, data: Inputs, problem: Problem, temporal_window: int
) -> Callable[[], Tuple[torch.Tensor, torch.Tensor]]:
    if mode == "per_step":
        return lambda: per_step_forward(data, problem)
    if mode == "batched":
        return lambda: batched_forward(data, problem, temporal_window)
    if mode == "fused":
        return lambda: fused_forward(data, problem, temporal_window)
    raise ValueError(f"unknown mode: {mode}")


def time_cuda(fn: Callable, warmup: int, repeat: int) -> Tuple[float, float]:
    with torch.no_grad():
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return statistics.mean(samples), statistics.pstdev(samples)


def correctness(
    outputs: Dict[str, Tuple[torch.Tensor, torch.Tensor]], dtype_name: str
) -> Dict[str, Dict[str, float]]:
    reference = outputs["per_step"]
    atol = 2e-2 if dtype_name == "fp16" else 5e-2
    rtol = 2e-2 if dtype_name == "fp16" else 1e-2
    checks = {}
    for mode, (spikes, state) in outputs.items():
        spike_diff = (spikes - reference[0]).abs()
        state_diff = (state - reference[1]).abs()
        spike_mismatch = (spikes != reference[0]).float().mean().item()
        state_ok = torch.allclose(state, reference[1], atol=atol, rtol=rtol)
        # Tensor-core accumulation can move values across the hard spike
        # threshold. Keep the established project-level fp16 tolerance.
        spike_ok = spike_mismatch <= (1e-3 if dtype_name == "fp16" else 5e-3)
        checks[mode] = {
            "ok": bool(spike_ok and state_ok),
            "spike_mismatch_ratio": float(spike_mismatch),
            "spike_max_abs": float(spike_diff.max().item()),
            "state_max_abs": float(state_diff.max().item()),
        }
    return checks


def _number(text: str) -> Optional[float]:
    text = text.strip().replace(",", "")
    if not text or text.lower() in {"n/a", "nan"}:
        return None
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)
    return float(match.group(0)) if match else None


def parse_ncu_csv(path: Path) -> Dict[str, float]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header_index = next(
        (
            index
            for index, line in enumerate(lines)
            if '"Kernel Name"' in line
            and (
                '"Metric Name"' in line
                or '"dram__bytes_op_read.sum"' in line
            )
        ),
        None,
    )
    if header_index is None:
        raise RuntimeError(f"NCU CSV has no raw metric table: {path}")
    rows = list(csv.DictReader(lines[header_index:]))
    if rows and "Metric Name" not in rows[0]:
        return _parse_ncu_wide_rows(rows)

    launches = set()
    read_bytes = 0.0
    write_bytes = 0.0
    weighted = {"sm": [0.0, 0.0], "dram": [0.0, 0.0]}
    per_launch: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    for row in rows:
        launch = (
            row.get("Process ID", ""),
            row.get("Kernel Name", ""),
            row.get("ID", row.get("Kernel ID", "")),
        )
        launches.add(launch)
        metric = row.get("Metric Name", "")
        value = _number(row.get("Metric Value", ""))
        if value is None:
            continue
        per_launch.setdefault(launch, {})[metric] = value
        if metric == "dram__bytes_op_read.sum":
            read_bytes += value
        elif metric == "dram__bytes_op_write.sum":
            write_bytes += value
    for metrics in per_launch.values():
        duration = metrics.get("gpu__time_duration.sum", 1.0)
        for key, metric in (
            ("sm", "sm__throughput.avg.pct_of_peak_sustained_active"),
            ("dram", "gpu__dram_throughput.avg.pct_of_peak_sustained_active"),
        ):
            if metric in metrics:
                weighted[key][0] += metrics[metric] * duration
                weighted[key][1] += duration
    return {
        "dram_read_bytes": read_bytes,
        "dram_write_bytes": write_bytes,
        "kernel_count": float(len(launches)),
        "sm_tput_pct": (
            weighted["sm"][0] / weighted["sm"][1] if weighted["sm"][1] else math.nan
        ),
        "dram_tput_pct": (
            weighted["dram"][0] / weighted["dram"][1]
            if weighted["dram"][1]
            else math.nan
        ),
    }


def _parse_ncu_wide_rows(rows: List[Dict[str, str]]) -> Dict[str, float]:
    read_metric = "dram__bytes_op_read.sum"
    write_metric = "dram__bytes_op_write.sum"
    duration_metric = "gpu__time_duration.sum"
    sm_metric = "sm__throughput.avg.pct_of_peak_sustained_active"
    dram_metric = "gpu__dram_throughput.avg.pct_of_peak_sustained_active"
    kernel_rows = [
        row
        for row in rows
        if row.get("ID", "").strip() and row.get("Kernel Name", "").strip()
    ]
    if not kernel_rows:
        raise RuntimeError("NCU wide CSV contains no kernel rows")

    read_bytes = sum(_number(row.get(read_metric, "")) or 0.0 for row in kernel_rows)
    write_bytes = sum(_number(row.get(write_metric, "")) or 0.0 for row in kernel_rows)
    weighted = {"sm": [0.0, 0.0], "dram": [0.0, 0.0]}
    for row in kernel_rows:
        duration = _number(row.get(duration_metric, "")) or 0.0
        for key, metric in (("sm", sm_metric), ("dram", dram_metric)):
            value = _number(row.get(metric, ""))
            if value is not None and duration > 0:
                weighted[key][0] += value * duration
                weighted[key][1] += duration
    return {
        "dram_read_bytes": read_bytes,
        "dram_write_bytes": write_bytes,
        "kernel_count": float(len(kernel_rows)),
        "sm_tput_pct": (
            weighted["sm"][0] / weighted["sm"][1] if weighted["sm"][1] else math.nan
        ),
        "dram_tput_pct": (
            weighted["dram"][0] / weighted["dram"][1]
            if weighted["dram"][1]
            else math.nan
        ),
    }


def collect_ncu(
    args, mode: str, timesteps: int, out_dir: Path
) -> Dict[str, float]:
    ncu = resolve_ncu_path(args.ncu_path)
    if ncu is None:
        raise RuntimeError(
            f"NCU executable not found: {args.ncu_path}. "
            "Pass an absolute path with --ncu-path; sudo may replace PATH via secure_path."
        )
    report = out_dir / "ncu" / f"{mode}_T{timesteps}.csv"
    report.parent.mkdir(parents=True, exist_ok=True)
    child = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--profile-child",
        "--mode",
        mode,
        "--t-values",
        str(timesteps),
        "--batch-size",
        str(args.batch_size),
        "--cin",
        str(args.cin),
        "--cout",
        str(args.cout),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--dtype",
        args.dtype,
        "--temporal-window",
        str(args.temporal_window),
        "--seed",
        str(args.seed),
    ]
    cmd = [
        ncu,
        "--target-processes",
        "all",
        "--profile-from-start",
        "off",
        "--csv",
        "--page",
        "raw",
        "--metrics",
        ",".join(NCU_METRICS),
        "--log-file",
        str(report),
        *child,
    ]
    completed = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=args.ncu_timeout_sec,
        check=False,
    )
    if completed.returncode != 0:
        report_tail = ""
        if report.exists():
            report_tail = report.read_text(
                encoding="utf-8", errors="replace"
            )[-4000:]
        raise RuntimeError(
            f"NCU failed for mode={mode}, T={timesteps}, exit={completed.returncode}\n"
            + completed.stdout[-4000:]
            + report_tail
        )
    return parse_ncu_csv(report)


def profile_child(args) -> None:
    problem = problem_from_args(args)
    timesteps = args.t_values[0]
    data = make_inputs(problem, timesteps, dtype_from_name(args.dtype), "cuda", args.seed)
    fn = mode_callable(args.mode, data, problem, args.temporal_window)
    with torch.no_grad():
        fn()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    with torch.no_grad():
        fn()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


def resolve_ncu_path(requested: str) -> Optional[str]:
    resolved = shutil.which(requested)
    if resolved is not None:
        return resolved
    requested_path = Path(requested)
    if requested_path.is_absolute() and requested_path.is_file():
        return str(requested_path)
    for candidate in (
        Path("/usr/local/cuda/bin/ncu"),
        Path("/usr/local/cuda-13.1/bin/ncu"),
        Path("/opt/nvidia/nsight-compute/ncu"),
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def write_csv(path: Path, rows: Iterable[Dict], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_breakdown(path: Path, rows: List[Dict], problem: Problem, args) -> None:
    fixed_t = args.breakdown_t
    element_bytes = torch.empty((), dtype=dtype_from_name(args.dtype)).element_size()
    selected = [row for row in rows if int(row["T"]) == fixed_t]
    output_bytes = fixed_t * problem.output_elements_per_step * element_bytes
    recurrent_state_tax = (
        2 * max(fixed_t - 1, 0) * problem.output_elements_per_step * element_bytes
    )
    weight_tax = max(fixed_t - 1, 0) * problem.weight_elements * element_bytes
    columns = (
        "mode",
        "T",
        "component",
        "estimated_bytes",
        "method",
    )
    breakdown = []
    for row in selected:
        measured = float(row["dram_total_bytes"])
        components = {
            "weight_reload": weight_tax if row["mode"] == "per_step" else 0,
            "preact_roundtrip": 2 * output_bytes if row["mode"] != "fused" else 0,
            "state_roundtrip": recurrent_state_tax if row["mode"] == "per_step" else 0,
        }
        known = sum(components.values())
        components["other"] = max(measured - known, 0.0) if math.isfinite(measured) else math.nan
        for component, value in components.items():
            breakdown.append(
                {
                    "mode": row["mode"],
                    "T": fixed_t,
                    "component": component,
                    "estimated_bytes": value,
                    "method": "analytical_tax_model; other=max(NCU-total-model,0)",
                }
            )
    write_csv(path, breakdown, columns)


def write_paper_summary(path: Path, rows: List[Dict], args) -> None:
    selected = {
        row["mode"]: row for row in rows if int(row["T"]) == args.breakdown_t
    }
    if len(selected) != len(MODES):
        path.write_text(
            f"No complete T={args.breakdown_t} row set; choose --breakdown-t from --t-values.\n",
            encoding="utf-8",
        )
        return
    per_step = selected["per_step"]
    fused = selected["fused"]
    measured = math.isfinite(float(per_step["dram_total_bytes"]))
    if measured:
        per_ratio = float(per_step["dram_total_bytes"]) / float(
            per_step["min_bytes_theory"]
        )
        fused_ratio = float(fused["dram_total_bytes"]) / float(
            fused["min_bytes_theory"]
        )
        text = (
            f"At T={args.breakdown_t}, per-step execution transferred "
            f"{per_ratio:.2f}x the theoretical minimum HBM traffic, while "
            f"Kairos fusion transferred {fused_ratio:.2f}x. Kernel launches "
            f"fell from {float(per_step['kernel_count']):.0f} to "
            f"{float(fused['kernel_count']):.0f}. Arithmetic intensity moved "
            f"from {float(per_step['arith_intensity']):.2f} to "
            f"{float(fused['arith_intensity']):.2f} FLOP/byte.\n"
        )
    else:
        text = (
            f"T={args.breakdown_t} timing and correctness were collected, but "
            "NCU performance-counter data was not requested. Run with "
            "--collect-ncu before using this experiment in the paper.\n"
        )
    path.write_text(text, encoding="utf-8")


def lock_gpu_clock(clock_mhz: Optional[int]) -> None:
    if clock_mhz is None:
        return
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        raise RuntimeError("nvidia-smi is required by --lock-gpu-clock-mhz")
    subprocess.run(
        [nvidia_smi, "-lgc", f"{clock_mhz},{clock_mhz}"],
        check=True,
    )

    def reset():
        subprocess.run([nvidia_smi, "-rgc"], check=False)

    atexit.register(reset)


def restore_sudo_output_ownership(path: Path) -> None:
    sudo_uid = os.environ.get("SUDO_UID")
    sudo_gid = os.environ.get("SUDO_GID")
    if sudo_uid is None or sudo_gid is None:
        return
    uid, gid = int(sudo_uid), int(sudo_gid)

    def restore():
        if not path.exists():
            return
        for child in path.rglob("*"):
            try:
                os.chown(child, uid, gid)
            except FileNotFoundError:
                pass
        os.chown(path, uid, gid)

    atexit.register(restore)


def environment_metadata(args) -> Dict:
    gpu = torch.cuda.get_device_properties(0)
    nvidia_smi = shutil.which("nvidia-smi")
    clocks = ""
    if nvidia_smi:
        proc = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=clocks.current.graphics,clocks.applications.graphics,"
                "clocks.max.graphics,persistence_mode",
                "--format=csv,noheader",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        clocks = proc.stdout.strip()
    return {
        "command": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": gpu.name,
        "gpu_compute_capability": f"{gpu.major}.{gpu.minor}",
        "ncu_path": resolve_ncu_path(args.ncu_path),
        "ncu_metrics": list(NCU_METRICS),
        "clock_state": clocks,
        "clock_lock_requested": args.lock_gpu_clock_mhz is not None,
        "requested_gpu_clock_mhz": args.lock_gpu_clock_mhz,
        "bn_handling": "inference BatchNorm folded into convolution weight and bias",
        "kernel_count_method": "unique NCU raw-report launch rows",
        "throughput_aggregation": "gpu__time_duration.sum weighted mean across launches",
        "breakdown_method": "analytical mechanism attribution; not direct hardware attribution",
        "peak_bandwidth_gbs": args.peak_bandwidth_gbs,
        "peak_compute_tflops": args.peak_compute_tflops,
    }


def problem_from_args(args) -> Problem:
    return Problem(args.batch_size, args.cin, args.cout, args.height, args.width)


def main(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if len(args.t_values) != len(set(args.t_values)) or any(t <= 0 for t in args.t_values):
        raise ValueError("--t-values must contain unique positive integers")
    if args.temporal_window not in (1, 2, 4, 8, 16):
        raise ValueError("--temporal-window must be one of 1,2,4,8,16")
    if args.profile_child:
        profile_child(args)
        return

    lock_gpu_clock(args.lock_gpu_clock_mhz)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    problem = problem_from_args(args)
    out_dir = Path(args.out_dir)
    restore_sudo_output_ownership(out_dir)
    rows: List[Dict] = []
    correctness_rows = []
    element_bytes = torch.empty((), dtype=dtype_from_name(args.dtype)).element_size()

    for timesteps in args.t_values:
        print(f"[case] T={timesteps}")
        data = make_inputs(
            problem, timesteps, dtype_from_name(args.dtype), "cuda", args.seed
        )
        outputs = {}
        for mode in MODES:
            fn = mode_callable(mode, data, problem, args.temporal_window)
            with torch.no_grad():
                outputs[mode] = fn()
        torch.cuda.synchronize()
        checks = correctness(outputs, args.dtype)
        for mode, check in checks.items():
            correctness_rows.append({"mode": mode, "T": timesteps, **check})
        failed = [mode for mode, check in checks.items() if not check["ok"]]
        if failed:
            raise AssertionError(f"correctness failed at T={timesteps}: {failed}; {checks}")

        for mode in MODES:
            fn = mode_callable(mode, data, problem, args.temporal_window)
            mean_ms, std_ms = time_cuda(fn, args.warmup, args.repeat)
            profile = {
                "dram_read_bytes": math.nan,
                "dram_write_bytes": math.nan,
                "kernel_count": math.nan,
                "sm_tput_pct": math.nan,
                "dram_tput_pct": math.nan,
            }
            if args.collect_ncu:
                profile = collect_ncu(args, mode, timesteps, out_dir)
            total_bytes = profile["dram_read_bytes"] + profile["dram_write_bytes"]
            intensity = (
                problem.flops(timesteps) / total_bytes
                if math.isfinite(total_bytes) and total_bytes > 0
                else math.nan
            )
            rows.append(
                {
                    "mode": mode,
                    "T": timesteps,
                    **profile,
                    "dram_total_bytes": total_bytes,
                    "min_bytes_theory": problem.min_bytes(timesteps, element_bytes),
                    "arith_intensity": intensity,
                    "time_ms_mean": mean_ms,
                    "time_ms_std": std_ms,
                }
            )
            print(
                f"  {mode:<9} {mean_ms:.4f} +/- {std_ms:.4f} ms "
                f"DRAM={total_bytes if math.isfinite(total_bytes) else 'NCU-not-collected'}"
            )

    write_csv(out_dir / "three_taxes_by_T.csv", rows, CSV_COLUMNS)
    write_csv(
        out_dir / "correctness.csv",
        correctness_rows,
        ("mode", "T", "ok", "spike_mismatch_ratio", "spike_max_abs", "state_max_abs"),
    )
    write_breakdown(out_dir / "three_taxes_breakdown.csv", rows, problem, args)
    write_paper_summary(out_dir / "paper_summary.md", rows, args)
    metadata = environment_metadata(args)
    metadata["problem"] = problem.__dict__
    metadata["t_values"] = args.t_values
    metadata["dtype"] = args.dtype
    metadata["temporal_window"] = args.temporal_window
    metadata["correctness_passed"] = True
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"[write] {out_dir / 'three_taxes_by_T.csv'}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Kairos motivation experiment: quantify three step-wise execution taxes."
    )
    parser.add_argument("--t-values", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cin", type=int, default=128)
    parser.add_argument("--cout", type=int, default=128)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--temporal-window", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out-dir", default="test/motivation_three_taxes")
    parser.add_argument("--collect-ncu", action="store_true")
    parser.add_argument("--ncu-path", default="ncu")
    parser.add_argument("--ncu-timeout-sec", type=int, default=3600)
    parser.add_argument("--breakdown-t", type=int, default=16)
    parser.add_argument("--peak-bandwidth-gbs", type=float, default=1792.0)
    parser.add_argument(
        "--peak-compute-tflops",
        type=float,
        default=419.2,
        help="Configured dense fp16 Tensor Core peak; override for the measured GPU/dtype.",
    )
    parser.add_argument("--lock-gpu-clock-mhz", type=int, default=None)
    parser.add_argument("--profile-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=MODES, help=argparse.SUPPRESS)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

# sudo -E "$(command -v python)" dev_tests/motivation_three_taxes.py \
#   --collect-ncu \
#   --ncu-path /usr/local/cuda-13.1/bin/ncu \
#   --lock-gpu-clock-mhz 2900 \
#   --peak-bandwidth-gbs 1792 \
#   --peak-compute-tflops 419.2 \
#   --out-dir test/motivation_three_taxes