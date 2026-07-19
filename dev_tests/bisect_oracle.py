"""Round 5 P0 oracle: fixed-methodology MobileNetV2 timing at a given T,
excluding compile/autotune time from the timed region (separately reported).

Usage: python dev_tests/bisect_oracle.py --T 4 --batch-size 4 [--reps 100] [--warmup 20]
Prints: COMPILE_TIME_S=<x> WALL_MS_MEAN=<x> WALL_MS_STD=<x>
"""
import argparse
import copy
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

import runtime.snn_custom_ops as snn_custom_ops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--model-channels", type=int, default=64)
    parser.add_argument("--dtype", default="fp32")
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--window", type=int, default=None, help="temporal-fuse-window; defaults to T")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    from benchmarks.benchmark_kairos_runtime import parse_args
    from benchmarks.validate_kairos_baselines import (
        RewriteCounters, make_resnet_layer, make_rewrite_backend, make_model_input, SingleStepModeLoopWrapper,
    )

    window = args.window if args.window is not None else args.T

    argv = [
        "x", "--models", "mobilenetv2", "--T", str(args.T), "--batch-size", str(args.batch_size),
        "--height", str(args.height), "--width", str(args.width), "--model-channels", str(args.model_channels),
        "--device", "cuda", "--dtype", args.dtype,
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
        parsed = parse_args()
    finally:
        sys.argv = old_argv

    torch_dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    torch.manual_seed(2026)
    base_layer_s = make_resnet_layer(
        "mobilenetv2", allow_resnet32_fallback=True, step_mode="s",
        model_channels=parsed.model_channels, lif_impl=parsed.lif_impl,
        sequence_length=parsed.sequence_length, transformer_depth=parsed.transformer_depth,
        transformer_dim=parsed.transformer_dim, transformer_heads=parsed.transformer_heads,
        transformer_input_dim=parsed.transformer_input_dim, transformer_vocab_size=parsed.transformer_vocab_size,
        transformer_num_classes=parsed.transformer_num_classes,
    ).to(device=parsed.device, dtype=torch_dtype).eval()
    model = SingleStepModeLoopWrapper(copy.deepcopy(base_layer_s), parsed.T).to(device=parsed.device, dtype=torch_dtype).eval()
    x = make_model_input("mobilenetv2", parsed, torch_dtype)

    counters = RewriteCounters()
    tmp_dir = Path(f"/tmp/kairos_bisect_oracle/T{args.T}_b{args.batch_size}_w{window}")
    backend = make_rewrite_backend(parsed, tmp_dir, counters)
    torch._dynamo.reset()
    compiled = torch.compile(model, backend=backend, fullgraph=False, dynamic=False)

    snn_custom_ops.configure_fused_op("triton", strict_triton=False, verbose=False)

    compile_start = time.time()
    with torch.no_grad():
        compiled(x)  # triggers dynamo trace + rewrite + first Triton compile/autotune
    torch.cuda.synchronize()
    compile_time_s = time.time() - compile_start

    # Extra warmup beyond the compile-triggering call, still outside the timed region.
    for _ in range(args.warmup):
        with torch.no_grad():
            compiled(x)
    torch.cuda.synchronize()

    iter_ms = []
    for _ in range(args.reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            compiled(x)
        end.record()
        torch.cuda.synchronize()
        iter_ms.append(start.elapsed_time(end))
    iter_ms_t = torch.tensor(iter_ms)
    wall_ms_mean = iter_ms_t.mean().item()
    wall_ms_std = iter_ms_t.std().item()

    print(f"COMPILE_TIME_S={compile_time_s:.3f} WALL_MS_MEAN={wall_ms_mean:.4f} WALL_MS_STD={wall_ms_std:.4f}")


if __name__ == "__main__":
    main()
