"""Regression matrix for the batch=1/temporal-window=1 correctness bug found
in round 4 of the MobileNet optimization task: at window=1, the temporal
mean-rewrite pass (_rewrite_temporal_sum_div_to_mean in
compiler/fx_temporal_spatial_canonicalize.py) could misidentify a getitem
into an unrelated multi-output node as a per-timestep temporal-stack getitem
(any group of size 1 trivially passes the "indices == range(len(group))"
check regardless of what the getitem source actually is), producing an
illegal `.sum(dim=0)` call on a Python tuple instead of a tensor.

This test compiles the full FX rewrite pipeline (matching
run_full_validation.sh's production config) across a batch x
temporal-fuse-window matrix and compares against the eager reference
numerically, to catch this bug class permanently.
"""
import argparse
import copy
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

import runtime.snn_custom_ops as snn_custom_ops


def build_and_compare(model_name, batch_size, window, T, height, width, model_channels, dtype, tmp_dir):
    from benchmarks.benchmark_kairos_runtime import parse_args
    from benchmarks.validate_kairos_baselines import (
        RewriteCounters, make_resnet_layer, make_rewrite_backend, make_model_input, SingleStepModeLoopWrapper,
    )

    argv = [
        "x", "--models", model_name, "--T", str(T), "--batch-size", str(batch_size),
        "--height", str(height), "--width", str(width), "--model-channels", str(model_channels),
        "--device", "cuda", "--dtype", dtype,
        "--fused-op-backend", "triton", "--rewrite-backend-mode", "standalone",
        "--fx-standalone-streams", "32", "--fx-standalone-cudagraph", "--fx-standalone-schedule-policy", "ready",
        "--enable-temporal-rewrite", "--enable-temporal-schedule",
        "--enable-spatial-batching", "--spatial-batching-ops", "conv", "bn", "add", "maxpool", "avgpool", "flatten", "linear",
        "--cudagraph-mode", "reduce-overhead",
        "--temporal-fuse-window", str(window), "--temporal-schedule-window", str(window),
        "--max-patterns", "1000000", "--warmup", "1", "--repeat", "1",
    ]
    old_argv = sys.argv
    try:
        sys.argv = argv
        args = parse_args()
    finally:
        sys.argv = old_argv

    torch_dtype = torch.float16 if dtype == "fp16" else torch.float32
    torch.manual_seed(2026)
    base_layer_s = make_resnet_layer(
        model_name, allow_resnet32_fallback=True, step_mode="s",
        model_channels=args.model_channels, lif_impl=args.lif_impl,
        sequence_length=args.sequence_length, transformer_depth=args.transformer_depth,
        transformer_dim=args.transformer_dim, transformer_heads=args.transformer_heads,
        transformer_input_dim=args.transformer_input_dim, transformer_vocab_size=args.transformer_vocab_size,
        transformer_num_classes=args.transformer_num_classes,
    ).to(device=args.device, dtype=torch_dtype).eval()

    eager_model = SingleStepModeLoopWrapper(copy.deepcopy(base_layer_s), args.T).to(device=args.device, dtype=torch_dtype).eval()
    compiled_model = SingleStepModeLoopWrapper(copy.deepcopy(base_layer_s), args.T).to(device=args.device, dtype=torch_dtype).eval()
    x = make_model_input(model_name, args, torch_dtype)

    with torch.no_grad():
        ref_out = eager_model(x)

    counters = RewriteCounters()
    backend = make_rewrite_backend(args, tmp_dir, counters)
    torch._dynamo.reset()
    compiled = torch.compile(compiled_model, backend=backend, fullgraph=False, dynamic=False)

    snn_custom_ops.configure_fused_op("triton", strict_triton=False, verbose=False)
    with torch.no_grad():
        got_out = compiled(x)
    torch.cuda.synchronize()

    max_err = (ref_out - got_out).abs().max().item()
    mean_err = (ref_out - got_out).abs().mean().item()
    allclose = torch.allclose(ref_out, got_out, rtol=1e-2, atol=1e-2)
    return allclose, max_err, mean_err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["mobilenetv2"])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--windows", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--T", type=int, default=16)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--model-channels", type=int, default=64)
    parser.add_argument("--dtype", default="fp32")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    tmp_root = Path("/tmp/kairos_batch_window_matrix")
    tmp_root.mkdir(parents=True, exist_ok=True)

    results = []
    for model_name in args.models:
        for batch_size in args.batch_sizes:
            for window in args.windows:
                if args.T % window != 0:
                    continue
                label = f"{model_name}_b{batch_size}_w{window}"
                print(f"\n### {label} ###")
                try:
                    allclose, max_err, mean_err = build_and_compare(
                        model_name, batch_size, window, args.T, args.height, args.width,
                        args.model_channels, args.dtype, tmp_root / label,
                    )
                    status = "PASS" if allclose else "NUMERIC_FAIL"
                    print(f"[{status}] {label} max_err={max_err:.4e} mean_err={mean_err:.4e}")
                    results.append((label, status, max_err, mean_err))
                except Exception as exc:
                    print(f"[CRASH] {label}: {type(exc).__name__}: {exc}")
                    traceback.print_exc()
                    results.append((label, "CRASH", None, None))

    print("\n\n========== MATRIX SUMMARY ==========")
    n_fail = 0
    for label, status, max_err, mean_err in results:
        if status != "PASS":
            n_fail += 1
        me = f"{max_err:.4e}" if max_err is not None else "-"
        print(f"{label:<30}{status:<15}{me}")
    print(f"\n{len(results) - n_fail}/{len(results)} PASS")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
