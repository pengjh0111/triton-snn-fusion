"""Measure Triton autotune cost for the batched temporal Linear+LIF codegen
kernel: cold-search wall time (with an empty Triton cache), warm-cache launch
overhead, and config-pool size after pruning -- compared against the conv
backend's own single-kernel-key first-search cost as a baseline.

Usage:
    TRITON_CACHE_DIR=/tmp/fresh_cache python3 dev_tests/bench_batched_linear_lif_autotune_cost.py
"""

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

# Representative shapes: the actual Linear layers in KairosSpikeTransformer
# (benchmarks/validate_kairos_baselines.py, dim=256 heads=8 mlp_ratio=4
# input_dim=768) crossed with (rows, T) combos covering small-batch/large-T
# (the target scenario) and the model's default batch=16, seq=256, T=16.
LINEAR_SHAPES = [
    ("input_proj", 768, 256),
    ("qkv", 256, 768),
    ("attn_proj", 256, 256),
    ("fc1_up", 256, 1024),
    ("fc2_down", 1024, 256),
    ("classifier", 256, 100),
]

ROW_T_CASES = [
    ("default_b16_seq256", 16 * 256, 16),
    ("small_batch_b1_seq256", 1 * 256, 16),
    ("small_batch_b1_seq16", 1 * 16, 16),
    ("large_batch_b64_seq256", 64 * 256, 16),
]


def time_call(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--skip-conv-baseline", action="store_true")
    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    from kernels.benchmark_batched_linear_lif_temporal_general import (
        run_fused_batched_linear_lif,
        _autotuned_kernels,
        _make_autotune_configs,
        _prune_batched_linear_configs,
    )

    total_raw_configs = len(_make_autotune_configs())
    print(f"[config-pool] total raw configs (pre-shape-prune): {total_raw_configs}")

    print("\n=== batched_linear_lif codegen kernel: cold-search wall time per shape ===")
    total_cold = 0.0
    for label, in_features, out_features in LINEAR_SHAPES:
        for row_label, rows, T in ROW_T_CASES:
            x_seq = torch.randn(T, rows, in_features, device="cuda", dtype=dtype) * 0.05
            # reshape to rank>=4 as the custom op requires: split rows into (rows,1)
            x_seq = x_seq.view(T, rows, 1, in_features)
            weight = torch.randn(out_features, in_features, device="cuda", dtype=dtype) * 0.02
            bias = torch.randn(out_features, device="cuda", dtype=dtype) * 0.01
            v_init = torch.zeros(rows, 1, out_features, device="cuda", dtype=dtype)

            pruned = _prune_batched_linear_configs(
                _make_autotune_configs(),
                {"T_STEPS": T, "rows": rows, "USE_TF32": dtype == torch.float32},
            )

            elapsed = time_call(lambda: run_fused_batched_linear_lif(x_seq, weight, bias, v_init, 1.0, 0.0, 2.0))
            total_cold += elapsed
            print(f"{label:<12} {row_label:<24} rows={rows:<6} T={T:<3} shape={in_features}->{out_features:<5} "
                  f"pruned_configs={len(pruned):<4} cold_search_s={elapsed:.3f}")

    print(f"\nTotal cold-search time across {len(LINEAR_SHAPES) * len(ROW_T_CASES)} (shape, rows, T) keys: {total_cold:.2f}s")

    print("\n=== warm-cache re-launch overhead (same keys, cache now populated) ===")
    total_warm = 0.0
    for label, in_features, out_features in LINEAR_SHAPES:
        for row_label, rows, T in ROW_T_CASES:
            x_seq = torch.randn(T, rows, in_features, device="cuda", dtype=dtype) * 0.05
            x_seq = x_seq.view(T, rows, 1, in_features)
            weight = torch.randn(out_features, in_features, device="cuda", dtype=dtype) * 0.02
            bias = torch.randn(out_features, device="cuda", dtype=dtype) * 0.01
            v_init = torch.zeros(rows, 1, out_features, device="cuda", dtype=dtype)
            elapsed = time_call(lambda: run_fused_batched_linear_lif(x_seq, weight, bias, v_init, 1.0, 0.0, 2.0))
            total_warm += elapsed
            print(f"{label:<12} {row_label:<24} rows={rows:<6} T={T:<3} warm_relaunch_s={elapsed:.4f}")
    print(f"\nTotal warm-cache re-launch time: {total_warm:.3f}s (this is the steady-state per-*process* cost; "
          f"in-process autotune results are additionally cached in the triton.autotune object itself, so a "
          f"second call with the *same* key inside the same process is microseconds, not measured here -- "
          f"this measures a fresh process picking up the on-disk cache).")

    if not args.skip_conv_baseline:
        print("\n=== conv backend baseline: single kernel_key first-search wall time ===")
        from kernels.benchmark_conv_lif_temporal_general import (
            run_fused_temporal_general_autotuned,
            ProblemShape,
            build_input_sequence,
            make_conv,
        )
        shape = ProblemShape(16, 4, 64, 128, 56, 56)
        conv = make_conv(shape, dtype=dtype)
        x_seq = build_input_sequence(shape, dtype=dtype)
        elapsed = time_call(lambda: run_fused_temporal_general_autotuned(x_seq, conv.weight, conv.bias))
        print(f"conv k3_s1_p1 mid-early shape cold-search: {elapsed:.3f}s")


if __name__ == "__main__":
    main()
