"""For rows=1, T=16, 768->2304 (qkv_rows1_T16 target scenario) autotune picked
BTILE_T=2, REUSE_GROUPS=1 (window=2) and came out slower than the old TC
kernel. Manually sweep all valid (BTILE_T, REUSE_GROUPS) schedules via the
specialized (non-autotune) kernel path with a few spatial configs to see
whether a larger window actually would have won, i.e. whether autotune's
internal noisy benchmarking under-explored this shape.
"""
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from kernels.benchmark_batched_linear_lif_temporal_general import (
    run_fused_batched_linear_lif as run_codegen,
    valid_temporal_schedules,
    LINEAR_AUTOTUNE_SPATIAL_CONFIGS,
)
from kernels.generated_temporal_transformer_lif_kernels import run_temporal_batched_linear_lif as run_tc


def warm_up_gpu_clocks(seconds=3.0):
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        a @ b
    torch.cuda.synchronize()


def time_cuda(fn, warmup=50, rep=300):
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


def time_cuda_median(fn, trials=5, **kwargs):
    return statistics.median([time_cuda(fn, **kwargs) for _ in range(trials)])


def main():
    torch.manual_seed(0)
    T, rows, in_features, out_features = 16, 1, 768, 2304
    dtype = torch.float16
    leading = (1, 1)
    x_seq = (torch.randn((T,) + leading + (in_features,), device="cuda", dtype=dtype) * 0.04).contiguous()
    weight = (torch.randn(out_features, in_features, device="cuda", dtype=dtype) * 0.02).contiguous()
    bias = (torch.randn(out_features, device="cuda", dtype=dtype) * 0.01).contiguous()
    v_init = torch.zeros(leading + (out_features,), device="cuda", dtype=dtype)

    warm_up_gpu_clocks(3.0)

    tc_ms = time_cuda_median(lambda: run_tc(x_seq, weight, bias, v_init, 1.0, 0.0, 2.0))
    print(f"tc (autotuned) = {tc_ms:.4f} ms\n")

    print(f"{'BTILE_T':>8} {'REUSE_GROUPS':>13} {'BLOCK_M':>8} {'BLOCK_N':>8} {'BLOCK_K':>8} {'ms':>10} {'vs_tc':>8}")
    results = []
    for btile_t, reuse_groups in valid_temporal_schedules(T):
        for spatial in LINEAR_AUTOTUNE_SPATIAL_CONFIGS:
            try:
                ms = time_cuda_median(
                    lambda bt=btile_t, rg=reuse_groups, cfg=spatial: run_codegen(
                        x_seq, weight, bias, v_init, 1.0, 0.0, 2.0,
                        use_autotune=False, btile_t=bt, reuse_groups=rg, spatial_config=cfg,
                    ),
                    trials=3, rep=200,
                )
            except Exception:
                continue
            results.append((btile_t, reuse_groups, spatial, ms))
            print(f"{btile_t:>8} {reuse_groups:>13} {spatial['BLOCK_M']:>8} {spatial['BLOCK_N']:>8} {spatial['BLOCK_K']:>8} {ms:>10.4f} {tc_ms/ms:>7.3f}x")

    results.sort(key=lambda r: r[3])
    print("\nTop 5 fastest manual configs:")
    for btile_t, reuse_groups, spatial, ms in results[:5]:
        print(f"  BTILE_T={btile_t} REUSE_GROUPS={reuse_groups} {spatial} ms={ms:.4f} vs_tc={tc_ms/ms:.3f}x")


if __name__ == "__main__":
    main()
