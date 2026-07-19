"""Task 0: category-attributed profiler trace for MobileNetV1/V2 (and a
ResNet18 control) through the actual compiled 'standalone' execution path
(not eager), matching run_full_validation.sh's production config.

Categorizes each profiled op by CPU-side node name pattern into:
  - dwconv_lif      : fused depthwise conv+LIF Triton kernel
  - pwconv_lif      : fused pointwise (1x1) conv+LIF Triton kernel
  - regular_conv_lif: fused regular (k3/k5/k7/k11, non-depthwise-non-1x1) conv+LIF
  - unfused_conv_bn : conv2d/batch_norm not going through any fused kernel
                       (this is where MobileNetV2's LIF-less project convs land)
  - residual_add    : elementwise add (stack-based or per-timestep)
  - glue            : stack/getitem/reshape/movedim etc.
  - other           : anything else

Also reports wall-clock (CUDA event) time vs GPU self-time sum, to quantify
CPU-side/launch-interpreter overhead (the "gap" category).
"""
import argparse
import copy
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.profiler import profile, ProfilerActivity

import runtime.snn_custom_ops as snn_custom_ops


def build_compiled(model_name, T, batch_size, height, width, model_channels, dtype, tmp_dir, rewrite_backend_mode="standalone", enable_spatial_batching=True):
    from benchmarks.benchmark_kairos_runtime import parse_args
    from benchmarks.validate_kairos_baselines import (
        RewriteCounters, make_resnet_layer, make_rewrite_backend, make_model_input, SingleStepModeLoopWrapper,
    )

    argv = [
        "x", "--models", model_name, "--T", str(T), "--batch-size", str(batch_size),
        "--height", str(height), "--width", str(width), "--model-channels", str(model_channels),
        "--device", "cuda", "--dtype", dtype,
        "--fused-op-backend", "triton", "--rewrite-backend-mode", rewrite_backend_mode,
        "--fx-standalone-streams", "32", "--fx-standalone-cudagraph", "--fx-standalone-schedule-policy", "ready",
        "--enable-temporal-rewrite", "--enable-temporal-schedule",
        "--cudagraph-mode", "reduce-overhead",
        "--temporal-fuse-window", "4", "--temporal-schedule-window", "4",
        "--max-patterns", "1000000", "--warmup", "1", "--repeat", "1",
    ]
    if enable_spatial_batching:
        argv += ["--enable-spatial-batching", "--spatial-batching-ops", "conv", "bn", "add", "maxpool", "avgpool", "flatten", "linear"]
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
    model = SingleStepModeLoopWrapper(copy.deepcopy(base_layer_s), args.T).to(device=args.device, dtype=torch_dtype).eval()
    x = make_model_input(model_name, args, torch_dtype)

    counters = RewriteCounters()
    backend = make_rewrite_backend(args, tmp_dir, counters)
    torch._dynamo.reset()
    compiled = torch.compile(model, backend=backend, fullgraph=False, dynamic=False)

    snn_custom_ops.configure_fused_op("triton", strict_triton=False, verbose=False)
    with torch.no_grad():
        compiled(x)
    torch.cuda.synchronize()
    return compiled, x


CATEGORY_RULES = [
    ("dwconv_lif", ("depthwise",)),
    ("pwconv_lif", ("k1_s1_p0", "pointwise")),
    ("classifier_linear_lif", ("temporal_linear_lif",)),
    ("regular_conv_lif", ("fused_temporal_conv", "fused_conv_lif", "conv_lif_temporal_general")),
    ("residual_add", ("vectorized_elementwise", "aten::add", "elementwise_kernel")),
    ("unfused_bn", ("batch_norm", "native_batch_norm", "cudnn::bn", "bn_fw", "bn_bw")),
    ("unfused_conv_bn", ("cutlass", "conv", "convolution", "cudnn_convolution", "implicit_gemm", "wgrad", "dgrad", "fprop")),
    ("copy_memcpy", ("memcpy", "aten::copy_", "aten::contiguous", "aten::clone")),
    ("glue", ("stack", "getitem", "reshape", "movedim", "view", "transpose", "cat", "chunk", "catarray")),
    ("reduce_pool", ("reduce_kernel", "avg_pool", "adaptive_avg", "max_pool")),
]


def categorize(name: str) -> str:
    lname = name.lower()
    for cat, keywords in CATEGORY_RULES:
        if any(k in lname for k in keywords):
            return cat
    return "other"


def profile_case(label, model_name, T, batch_size, height, width, model_channels, dtype, tmp_root, reps=10, enable_spatial_batching=True):
    print(f"\n########## {label} ({model_name}, T={T}, batch={batch_size}, {height}x{width}, ch={model_channels}, {dtype}, spatial_batching={enable_spatial_batching}) ##########")
    compiled, x = build_compiled(model_name, T, batch_size, height, width, model_channels, dtype, tmp_root / label, enable_spatial_batching=enable_spatial_batching)

    for _ in range(5):
        with torch.no_grad():
            compiled(x)
    torch.cuda.synchronize()

    # Wall-clock via per-iteration CUDA events (pre-allocated, reused --
    # allocating a fresh Event object per iteration was observed to correlate
    # with GPU memory growth under the cudagraph "reduce-overhead" path at
    # T=16/batch=8 production scale, eventually OOMing after ~40 calls even
    # though the same config is stable at reps=10 with a single time-window
    # measurement; not fully root-caused, flagged separately in the report).
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for i in range(reps):
        starts[i].record()
        with torch.no_grad():
            compiled(x)
        ends[i].record()
    torch.cuda.synchronize()
    iter_ms = [starts[i].elapsed_time(ends[i]) for i in range(reps)]
    iter_ms_t = torch.tensor(iter_ms)
    wall_ms = iter_ms_t.mean().item()
    wall_ms_std = iter_ms_t.std().item()
    print(f"wall_ms mean={wall_ms:.4f} std={wall_ms_std:.4f} (n={reps})")

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            with torch.no_grad():
                compiled(x)
        torch.cuda.synchronize()

    events = prof.key_averages()
    totals = defaultdict(float)
    counts = defaultdict(int)
    for e in events:
        # "Torch-Compiled Region" (and similar dynamo/inductor wrapper scope
        # markers) get a self_device_time_total that DOUBLE-COUNTS the real
        # kernels launched underneath them (verified: excluding these makes
        # the category sum match profiler's own "Self CUDA time total"
        # exactly; including them does not and can even produce a negative
        # CPU-gap, which is impossible).
        if "Compiled Region" in e.key or "CompiledFunction" in e.key:
            continue
        cat = categorize(e.key)
        totals[cat] += e.self_device_time_total
        counts[cat] += e.count

    gpu_total_us = sum(totals.values())
    gpu_total_ms_per_iter = gpu_total_us / 1000.0 / reps
    print(f"wall_ms/iter={wall_ms:.4f}  gpu_self_time_ms/iter={gpu_total_ms_per_iter:.4f}  cpu_gap_ms/iter={wall_ms - gpu_total_ms_per_iter:.4f}")
    print(f"{'category':<20}{'gpu_us_total':>14}{'share%':>10}{'#calls':>10}")
    for cat, _ in CATEGORY_RULES + [("other", ())]:
        if totals[cat] == 0 and counts[cat] == 0:
            continue
        share = 100.0 * totals[cat] / gpu_total_us if gpu_total_us else 0.0
        print(f"{cat:<20}{totals[cat]:>14.1f}{share:>9.2f}%{counts[cat]:>10}")

    print("\nTop 20 ops by self CUDA time:")
    print(events.table(sort_by="self_cuda_time_total", row_limit=20))

    return {
        "label": label, "wall_ms": wall_ms, "wall_ms_std": wall_ms_std, "gpu_ms": gpu_total_ms_per_iter,
        "cpu_gap_ms": wall_ms - gpu_total_ms_per_iter,
        "category_us": dict(totals), "category_counts": dict(counts),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["mobilenetv2", "mobilenetv1", "resnet18"])
    parser.add_argument("--T", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--model-channels", type=int, default=64)
    parser.add_argument("--dtype", default="fp32")
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--no-spatial-batching", action="store_true", help="disable Route 1 (spatial batching) to measure its true baseline delta")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    tmp_root = Path("/tmp/kairos_mobilenet_profile")
    tmp_root.mkdir(parents=True, exist_ok=True)

    results = []
    for model_name in args.models:
        result = profile_case(
            model_name, model_name, args.T, args.batch_size, args.height, args.width,
            args.model_channels, args.dtype, tmp_root, reps=args.reps,
            enable_spatial_batching=not args.no_spatial_batching,
        )
        results.append(result)

    print("\n\n========== SUMMARY ==========")
    print(f"{'model':<16}{'wall_ms':>10}{'wall_std':>10}{'gpu_ms':>10}{'cpu_gap_ms':>12}")
    for r in results:
        print(f"{r['label']:<16}{r['wall_ms']:>10.3f}{r['wall_ms_std']:>10.3f}{r['gpu_ms']:>10.3f}{r['cpu_gap_ms']:>12.3f}")


if __name__ == "__main__":
    main()
