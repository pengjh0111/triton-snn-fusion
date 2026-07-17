"""Task 2.2/2.3: end-to-end numerical consistency (codegen vs tc backend) and
torch.profiler-based operator-share attribution for ChronosSpikeTransformer,
run eagerly (no torch.compile) so profiler traces stay simple to interpret.
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from spikingjelly.activation_based import functional

import runtime.snn_custom_ops as snn_custom_ops
from benchmarks.validate_chronos_baselines import ChronosSpikeTransformer, make_model_input


class Args:
    batch_size = 16
    sequence_length = 256
    transformer_input_dim = 768
    device = "cuda"


def build_model(dtype):
    torch.manual_seed(2026)
    model = ChronosSpikeTransformer(
        input_dim=768, dim=256, depth=8, heads=8, num_classes=100,
        lif_impl="chronos", step_mode="m",
    ).to(device="cuda", dtype=dtype).eval()
    return model


def run_multistep(model, x_seq):
    functional.reset_net(model)
    with torch.no_grad():
        return model(x_seq)


def main():
    dtype = torch.float16
    T = 16
    torch.manual_seed(0)
    x = make_model_input("spiketransformer", Args(), dtype)
    x_seq = x.unsqueeze(0).repeat(T, 1, 1, 1)

    model = build_model(dtype)

    print("=== Task 2.2: numerical consistency (codegen vs tc backend) ===")
    snn_custom_ops.configure_fused_op("triton", strict_triton=True, verbose=False)

    os.environ["CHRONOS_BATCHED_LINEAR_LIF_BACKEND"] = "codegen"
    out_codegen = run_multistep(model, x_seq)

    os.environ["CHRONOS_BATCHED_LINEAR_LIF_BACKEND"] = "tc"
    out_tc = run_multistep(model, x_seq)

    max_abs_diff = (out_codegen.float() - out_tc.float()).abs().max().item()
    mean_abs_diff = (out_codegen.float() - out_tc.float()).abs().mean().item()
    allclose = torch.allclose(out_codegen.float(), out_tc.float(), atol=2e-2, rtol=2e-2)
    print(f"logits shape={tuple(out_codegen.shape)} max_abs_diff={max_abs_diff:.4e} mean_abs_diff={mean_abs_diff:.4e} allclose(atol=2e-2)={allclose}")

    print("\n=== Task 2.3: op-share profiling (codegen backend) ===")
    os.environ["CHRONOS_BATCHED_LINEAR_LIF_BACKEND"] = "codegen"
    functional.reset_net(model)
    # warmup (also completes autotune search so profiler doesn't capture compile time)
    for _ in range(5):
        run_multistep(model, x_seq)
    torch.cuda.synchronize()

    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(10):
            run_multistep(model, x_seq)
        torch.cuda.synchronize()

    events = prof.key_averages()
    total_cuda_us = sum(e.self_device_time_total for e in events)
    fused_linear_lif_us = sum(
        e.self_device_time_total for e in events
        if "fused_temporal_batched_linear" in e.key or "batched_linear_lif" in e.key.lower()
    )
    print(f"total_self_cuda_time_us={total_cuda_us:.1f} fused_linear_lif_self_cuda_time_us={fused_linear_lif_us:.1f} "
          f"share={100.0*fused_linear_lif_us/total_cuda_us:.2f}%")
    print("\nTop 15 CUDA ops by self time:")
    print(events.table(sort_by="self_cuda_time_total", row_limit=15))


if __name__ == "__main__":
    main()
