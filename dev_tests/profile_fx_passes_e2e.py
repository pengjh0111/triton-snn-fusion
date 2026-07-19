"""Profiler attribution: drive the actual compiled 'standalone' FX-rewritten
execution path (not eager) for the full KairosSpikeTransformer, with and
without the post-fuse passes, and report attention-kernel time share.
"""
import argparse
import copy
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.profiler import profile, ProfilerActivity

import runtime.snn_custom_ops as snn_custom_ops


def build_and_compile(pass_env, tmp_dir):
    from benchmarks.benchmark_kairos_runtime import parse_args
    from benchmarks.validate_kairos_baselines import (
        RewriteCounters, make_resnet_layer, make_rewrite_backend, make_model_input, SingleStepModeLoopWrapper,
    )

    all_vars = ("KAIROS_PASS_SDPA", "KAIROS_PASS_STACK_CSE", "KAIROS_PASS_VINIT_CLEANUP", "KAIROS_PASS_CLASSIFIER_BATCH")
    for v in all_vars:
        os.environ.pop(v, None)
    for v, enabled in pass_env.items():
        os.environ[v] = "1" if enabled else "0"

    argv = [
        "x", "--models", "spiketransformer", "--T", "16", "--batch-size", "16",
        "--device", "cuda", "--dtype", "fp16",
        "--fused-op-backend", "triton", "--rewrite-backend-mode", "standalone",
        "--fx-standalone-streams", "1",
        "--enable-temporal-rewrite", "--enable-temporal-schedule",
        "--temporal-fuse-window", "4", "--temporal-schedule-window", "4",
        "--max-patterns", "1000000", "--warmup", "1", "--repeat", "1",
    ]
    old_argv = sys.argv
    try:
        sys.argv = argv
        args = parse_args()
    finally:
        sys.argv = old_argv

    dtype = torch.float16
    torch.manual_seed(2026)
    base_layer_s = make_resnet_layer(
        "spiketransformer", allow_resnet32_fallback=True, step_mode="s",
        model_channels=64, lif_impl=args.lif_impl, sequence_length=args.sequence_length,
        transformer_depth=args.transformer_depth, transformer_dim=args.transformer_dim,
        transformer_heads=args.transformer_heads, transformer_input_dim=args.transformer_input_dim,
        transformer_vocab_size=args.transformer_vocab_size, transformer_num_classes=args.transformer_num_classes,
    ).to(device=args.device, dtype=dtype).eval()
    model = SingleStepModeLoopWrapper(copy.deepcopy(base_layer_s), args.T).to(device=args.device, dtype=dtype).eval()
    x = make_model_input("spiketransformer", args, dtype)

    counters = RewriteCounters()
    backend = make_rewrite_backend(args, tmp_dir, counters)
    torch._dynamo.reset()
    compiled = torch.compile(model, backend=backend, fullgraph=False, dynamic=False)

    snn_custom_ops.configure_fused_op("triton", strict_triton=False, verbose=False)
    with torch.no_grad():
        compiled(x)  # trigger compile + autotune
    torch.cuda.synchronize()
    return compiled, x


def profile_case(label, pass_env, tmp_root, reps=10):
    compiled, x = build_and_compile(pass_env, tmp_root / label)
    for _ in range(5):
        with torch.no_grad():
            compiled(x)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            with torch.no_grad():
                compiled(x)
        torch.cuda.synchronize()

    events = prof.key_averages()
    total = sum(e.self_device_time_total for e in events)
    attn_keywords = ("matmul", "bmm", "softmax", "scaled_dot_product_attention", "attention", "cutlass", "flash")
    attn_time = sum(e.self_device_time_total for e in events if any(k in e.key.lower() for k in attn_keywords))
    print(f"\n=== {label} ===")
    print(f"total_self_device_time_us={total:.1f} attn_related_self_device_time_us={attn_time:.1f} share={100.0*attn_time/total:.2f}%")
    print(events.table(sort_by="self_cuda_time_total", row_limit=12))
    return total, attn_time


def main():
    tmp_root = Path("/tmp/kairos_fx_pass_profile")
    tmp_root.mkdir(parents=True, exist_ok=True)
    profile_case("baseline_off", {}, tmp_root)
    profile_case("all_on", {"KAIROS_PASS_SDPA": True, "KAIROS_PASS_STACK_CSE": True, "KAIROS_PASS_VINIT_CLEANUP": True, "KAIROS_PASS_CLASSIFIER_BATCH": True}, tmp_root)


if __name__ == "__main__":
    main()
