"""Performance comparison for batched temporal Linear+LIF:
old single-window TC kernel vs. new codegen (Conv-style BTILE_T x REUSE_GROUPS
double-layer temporal schedule) vs. a naive per-timestep PyTorch baseline.

Reports the autotune-selected (BTILE_T, REUSE_GROUPS, BLOCK_M/N/K) config per
shape for the new kernel so we can confirm the double-layer schedule is
actually being selected rather than degenerating back to a TC-equivalent
(BTILE_T*REUSE_GROUPS == T but effectively REUSE_GROUPS == 1) config.
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


def warm_up_gpu_clocks(seconds: float = 3.0):
    # Consumer GPUs (e.g. this RTX 5090) idle at a few hundred MHz and only
    # ramp to boost clocks under sustained load; without this, whichever
    # shape happens to run first/after an idle gap gets an inflated reading
    # purely from clock ramp-up, not kernel cost. Force boost clocks with a
    # large matmul burst before any measurement.
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        a @ b
    torch.cuda.synchronize()


def time_cuda(fn, warmup=30, rep=100):
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


def time_cuda_median(fn, warmup=30, rep=100, trials=3):
    samples = [time_cuda(fn, warmup=warmup, rep=rep) for _ in range(trials)]
    return statistics.median(samples), (max(samples) - min(samples))


SHAPES = [
    # label, T, leading, in_features, out_features
    ("qkv_rows1_T16", 16, (1, 1), 768, 2304),
    ("qkv_rows16_T16", 16, (1, 16), 768, 2304),
    ("qkv_seq197_T4", 4, (1, 197), 768, 2304),
    ("ffn_rows16_T8", 8, (1, 16), 768, 3072),
    ("ffn_batch8seq128_T4", 4, (8, 128), 4096, 4096),
    ("small_rows1_T16", 16, (1, 1), 256, 1024),
    ("large_rows_T2", 2, (32, 128), 1024, 1024),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--rep", type=int, default=50)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    print("Warming up GPU boost clocks (large matmul burst, ~3s) before any measurement...")
    warm_up_gpu_clocks(3.0)

    header = (
        f"{'label':<22}{'T':>3} {'rows':>8} {'shape':>14} {'naive(ms)':>12} {'tc(ms)':>14} "
        f"{'codegen(ms)':>16} {'codegen/tc':>11} {'codegen/naive':>14}"
    )
    print(header)
    for label, T, leading, in_features, out_features in SHAPES:
        torch.manual_seed(0)
        x_seq = (torch.randn((T,) + leading + (in_features,), device="cuda", dtype=dtype) * 0.04).contiguous()
        weight = (torch.randn(out_features, in_features, device="cuda", dtype=dtype) * 0.02).contiguous()
        bias = (torch.randn(out_features, device="cuda", dtype=dtype) * 0.01).contiguous()
        v_init = torch.zeros(leading + (out_features,), device="cuda", dtype=dtype)
        rows = 1
        for d in leading:
            rows *= d

        # Prime autotune search (first call) outside of timing.
        run_codegen(x_seq, weight, bias, v_init, 1.0, 0.0, 2.0)
        run_tc(x_seq, weight, bias, v_init, 1.0, 0.0, 2.0)

        naive_ms, naive_spread = time_cuda_median(lambda: naive_baseline(x_seq, weight, bias, v_init, 1.0, 0.0, 2.0), rep=args.rep)
        tc_ms, tc_spread = time_cuda_median(lambda: run_tc(x_seq, weight, bias, v_init, 1.0, 0.0, 2.0), rep=args.rep)
        codegen_ms, codegen_spread = time_cuda_median(lambda: run_codegen(x_seq, weight, bias, v_init, 1.0, 0.0, 2.0), rep=args.rep)
        cfg = get_autotune_best_config()
        cfg_text = (
            f"BM={cfg['BLOCK_M']} BN={cfg['BLOCK_N']} BK={cfg['BLOCK_K']} "
            f"BTILE_T={cfg['BTILE_T']} REUSE_GROUPS={cfg['REUSE_GROUPS']} window={cfg['kernel_temporal_window']}"
            if cfg is not None
            else "unavailable"
        )
        print(
            f"{label:<22}{T:>3} {rows:>8} {in_features:>6}->{out_features:<6} "
            f"{naive_ms:>8.4f}±{naive_spread:<3.3f} {tc_ms:>8.4f}±{tc_spread:<4.3f} "
            f"{codegen_ms:>10.4f}±{codegen_spread:<4.3f} "
            f"{tc_ms / codegen_ms:>10.2f}x {naive_ms / codegen_ms:>13.2f}x   cfg: {cfg_text}"
        )


if __name__ == "__main__":
    main()
