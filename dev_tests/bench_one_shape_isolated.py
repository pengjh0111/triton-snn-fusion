"""Benchmark a single (T, leading, in_features, out_features) shape in an
isolated process: naive / old-TC / new-codegen. Meant to be invoked once per
shape from a driver loop so each measurement gets a fresh process (no
cross-shape GPU/allocator/cache interference) -- run under locked GPU clocks
(`sudo nvidia-smi -lgc <freq>,<freq>`) for reproducibility; the in-process
multi-shape variant (dev_tests/bench_temporal_batched_linear_lif.py) showed
unstable, sometimes contradictory results even after adding a clock-warmup
burst, so this is the authoritative measurement path.

Prints one line: label T rows in_features out_features naive_ms tc_ms
codegen_ms cfg_text
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

import runtime.snn_custom_ops as snn_custom_ops
from kernels.generated_temporal_transformer_lif_kernels import run_temporal_batched_linear_lif as run_tc
from kernels.benchmark_batched_linear_lif_temporal_general import (
    run_fused_batched_linear_lif as run_codegen,
    get_autotune_best_config,
)


def naive_baseline(x_seq, weight, bias, v_init, v_threshold, v_reset, tau):
    v = v_init
    spikes = []
    for t in range(int(x_seq.shape[0])):
        current = F.linear(x_seq[t], weight, bias)
        spike, v = snn_custom_ops.lif_forward_state_torch(current, v, v_threshold, v_reset, tau, False)
        spikes.append(spike)
    return torch.stack(spikes, dim=0), v


def warm_up_gpu_clocks(seconds=3.0):
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        a @ b
    torch.cuda.synchronize()


def time_cuda(fn, warmup=50, rep=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / rep


def time_cuda_median(fn, warmup=50, rep=200, trials=5):
    samples = [time_cuda(fn, warmup=warmup, rep=rep) for _ in range(trials)]
    return statistics.median(samples), min(samples), max(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--T", type=int, required=True)
    parser.add_argument("--leading", required=True, help="comma-separated leading dims, e.g. 1,197")
    parser.add_argument("--in-features", type=int, required=True)
    parser.add_argument("--out-features", type=int, required=True)
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--rep", type=int, default=200)
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    leading = tuple(int(x) for x in args.leading.split(","))
    rows = 1
    for d in leading:
        rows *= d

    torch.manual_seed(0)
    x_seq = (torch.randn((args.T,) + leading + (args.in_features,), device="cuda", dtype=dtype) * 0.04).contiguous()
    weight = (torch.randn(args.out_features, args.in_features, device="cuda", dtype=dtype) * 0.02).contiguous()
    bias = (torch.randn(args.out_features, device="cuda", dtype=dtype) * 0.01).contiguous()
    v_init = torch.zeros(leading + (args.out_features,), device="cuda", dtype=dtype)

    warm_up_gpu_clocks(3.0)

    # Prime autotune search outside of timing.
    run_codegen(x_seq, weight, bias, v_init, 1.0, 0.0, 2.0)
    run_tc(x_seq, weight, bias, v_init, 1.0, 0.0, 2.0)
    torch.cuda.synchronize()

    naive_ms, naive_min, naive_max = time_cuda_median(lambda: naive_baseline(x_seq, weight, bias, v_init, 1.0, 0.0, 2.0), rep=args.rep)
    tc_ms, tc_min, tc_max = time_cuda_median(lambda: run_tc(x_seq, weight, bias, v_init, 1.0, 0.0, 2.0), rep=args.rep)
    codegen_ms, codegen_min, codegen_max = time_cuda_median(lambda: run_codegen(x_seq, weight, bias, v_init, 1.0, 0.0, 2.0), rep=args.rep)

    cfg = get_autotune_best_config()
    cfg_text = (
        f"BM={cfg['BLOCK_M']}_BN={cfg['BLOCK_N']}_BK={cfg['BLOCK_K']}_BT={cfg['BTILE_T']}_RG={cfg['REUSE_GROUPS']}"
        if cfg is not None
        else "unavailable"
    )
    print(
        f"RESULT {args.label} T={args.T} rows={rows} shape={args.in_features}->{args.out_features} "
        f"naive_ms={naive_ms:.4f}[{naive_min:.4f},{naive_max:.4f}] "
        f"tc_ms={tc_ms:.4f}[{tc_min:.4f},{tc_max:.4f}] "
        f"codegen_ms={codegen_ms:.4f}[{codegen_min:.4f},{codegen_max:.4f}] "
        f"codegen_vs_tc={tc_ms/codegen_ms:.3f}x codegen_vs_naive={naive_ms/codegen_ms:.3f}x cfg={cfg_text}"
    )


if __name__ == "__main__":
    main()
