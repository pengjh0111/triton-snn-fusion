import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Callable, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from test.test_fused_convlif_kernel_configs import CASES, _dtype_from_arg, _pair, make_case_tensors, torch_temporal_ref
from runtime.triton_convlif_backend import (
    classify_conv_lif_config,
    run_triton_fused_conv_lif_state,
    run_triton_fused_temporal_conv_lif_state,
)


def time_cuda(fn: Callable, warmup: int, repeat: int) -> Dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    times_sorted = sorted(times)
    return {
        "mean_ms": float(statistics.mean(times)),
        "p50_ms": float(times_sorted[len(times_sorted) // 2]),
        "p90_ms": float(times_sorted[int(0.9 * (len(times_sorted) - 1))]),
        "min_ms": float(min(times)),
    }


def run_case(case: Dict, T: int, device: str, dtype_name: str, warmup: int, repeat: int, use_autotune: bool) -> Dict:
    stride = _pair(case["stride"])
    padding = _pair(case["padding"])
    dtype = _dtype_from_arg(dtype_name)
    x_seq, weight, bias, v0 = make_case_tensors(case, T, device, dtype=dtype)
    xs = [x_seq[i] for i in range(T)]
    kernel_key = classify_conv_lif_config(weight, stride, padding, [1, 1], 1)

    def torch_fn():
        return torch_temporal_ref(x_seq, weight, bias, v0, stride, padding)

    def torch_compile_body(x_seq_arg, weight_arg, bias_arg, v0_arg):
        return torch_temporal_ref(x_seq_arg, weight_arg, bias_arg, v0_arg, stride, padding)

    compiled_torch_fn = torch.compile(torch_compile_body, fullgraph=False, dynamic=False)

    def torch_compile_fn():
        return compiled_torch_fn(x_seq, weight, bias, v0)

    def triton_fn():
        if T == 1:
            return run_triton_fused_conv_lif_state(
                x_seq[0],
                weight,
                bias,
                v0,
                stride,
                padding,
                [1, 1],
                1,
                1.0,
                0.0,
                2.0,
                False,
                use_autotune=use_autotune,
            )
        return run_triton_fused_temporal_conv_lif_state(
            xs,
            weight,
            bias,
            v0,
            stride,
            padding,
            [1, 1],
            1,
            1.0,
            0.0,
            2.0,
            False,
            use_autotune=use_autotune,
        )

    torch_stats = time_cuda(torch_fn, warmup, repeat)
    torch_compile_stats = time_cuda(torch_compile_fn, warmup, repeat)
    sample_result = triton_fn()
    torch.cuda.synchronize()
    triton_stats = time_cuda(triton_fn, warmup, repeat)
    temporal_config = sample_result.kernel_temporal_config if hasattr(sample_result, "kernel_temporal_config") else None
    speedup = torch_stats["mean_ms"] / triton_stats["mean_ms"] if triton_stats["mean_ms"] > 0 else 0.0
    speedup_vs_compile = (
        torch_compile_stats["mean_ms"] / triton_stats["mean_ms"] if triton_stats["mean_ms"] > 0 else 0.0
    )
    return {
        "case": case["name"],
        "T": T,
        "dtype": dtype_name,
        "shape": [case["N"], case["Cin"], case["H"], case["W"]],
        "out_channels": case["Cout"],
        "kernel": case["K"],
        "stride": case["stride"],
        "padding": case["padding"],
        "kernel_key": kernel_key,
        "use_autotune": bool(use_autotune),
        "torch": torch_stats,
        "torch_compile": torch_compile_stats,
        "triton": triton_stats,
        "speedup": float(speedup),
        "speedup_vs_compile": float(speedup_vs_compile),
        "kernel_temporal_config": temporal_config,
        "BTILE_T": None if temporal_config is None else temporal_config.get("BTILE_T"),
        "REUSE_GROUPS": None if temporal_config is None else temporal_config.get("REUSE_GROUPS"),
        "kernel_temporal_window": None if temporal_config is None else temporal_config.get("kernel_temporal_window"),
    }


def print_table(rows: List[Dict]):
    print("| case | T | dtype | key | K | stride | eager ms | compile ms | triton ms | speedup vs compile | BTILE_T | REUSE_GROUPS | kernel window |")
    print("|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['case']} | {row['T']} | {row['dtype']} | {row['kernel_key']} | {row['kernel']} | {row['stride']} | "
            f"{row['torch']['mean_ms']:.3f} | {row['torch_compile']['mean_ms']:.3f} | "
            f"{row['triton']['mean_ms']:.3f} | {row['speedup_vs_compile']:.3f} | "
            f"{row['BTILE_T']} | {row['REUSE_GROUPS']} | {row['kernel_temporal_window']} |"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Chronos fused ConvLIF Triton kernel configs.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--out-dir", default="kernel_config_benchmark")
    parser.add_argument("--quick", action="store_true", help="Run a reduced T set for smoke testing.")
    parser.add_argument("--all-t", action="store_true", help="Benchmark T=1,2,4,8,16 instead of the default T=1,16 set.")
    parser.add_argument("--t-values", type=int, nargs="+", default=None, help="Override temporal lengths to benchmark.")
    parser.add_argument("--case-names", nargs="+", default=None, help="Benchmark only the named CASES entries.")
    parser.add_argument("--use-autotune", action="store_true", help="Benchmark through autotuned temporal dispatch.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)

    if args.t_values is not None:
        t_values = args.t_values
    elif args.quick:
        t_values = [1]
    elif args.all_t:
        t_values = [1, 2, 4, 8, 16]
    else:
        t_values = [1, 16]
    cases = CASES
    if args.case_names is not None:
        wanted = set(args.case_names)
        cases = [case for case in CASES if case["name"] in wanted]
        missing = sorted(wanted - {case["name"] for case in cases})
        if missing:
            raise ValueError(f"unknown case name(s): {missing}")
    rows = []
    for case in cases:
        for T in t_values:
            print(f"[BENCH] {case['name']} T={T} dtype={args.dtype}")
            rows.append(run_case(case, T, args.device, args.dtype, args.warmup, args.repeat, use_autotune=args.use_autotune))

    print_table(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "kernel_config_benchmark.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
