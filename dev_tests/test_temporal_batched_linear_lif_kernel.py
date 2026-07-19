"""Correctness tests for the batched temporal Linear+LIF custom ops
(fused_temporal_batched_linear_lif_state / fused_temporal_batched_linear_add_lif_state),
covering both the new codegen kernel (kernels/generated_temporal_batched_linear_lif_kernel.py,
default backend) and the older TC kernel
(kernels/generated_temporal_transformer_lif_kernels.py, backend="tc") kept for regression
comparison. Both are checked against a naive per-timestep PyTorch reference.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import runtime.snn_custom_ops as snn_custom_ops


def resolve_dtype(name: str):
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(name)


def reference(x_seq, weight, bias, v_init, tau, detach_reset, residual_seq=None):
    v = v_init
    spikes = []
    for t in range(int(x_seq.shape[0])):
        current = F.linear(x_seq[t], weight, bias)
        if residual_seq is not None:
            current = current + residual_seq[t]
        spike, v = snn_custom_ops.lif_forward_state_torch(current, v, 1.0, 0.0, tau, detach_reset)
        spikes.append(spike)
    return torch.stack(spikes, dim=0), v


def run_case(
    device,
    dtype,
    T,
    leading,
    in_features,
    out_features,
    bias_enabled,
    has_residual,
    tau,
    detach_reset,
    backend,
):
    torch.manual_seed(2026 + T + in_features + out_features + len(leading))
    x_seq = (torch.randn((T,) + leading + (in_features,), device=device, dtype=dtype) * 0.04).contiguous()
    weight = (torch.randn(out_features, in_features, device=device, dtype=dtype) * 0.02).contiguous()
    bias = (
        (torch.randn(out_features, device=device, dtype=dtype) * 0.01).contiguous()
        if bias_enabled
        else None
    )
    v_init = torch.zeros(leading + (out_features,), device=device, dtype=dtype)
    residual_seq = None
    if has_residual:
        residual_seq = (torch.randn((T,) + leading + (out_features,), device=device, dtype=dtype) * 0.03).contiguous()

    ref_spike, ref_v = reference(x_seq, weight, bias, v_init, tau, detach_reset, residual_seq=residual_seq)

    os.environ["KAIROS_BATCHED_LINEAR_LIF_BACKEND"] = backend
    snn_custom_ops.reset_fused_op_call_stats()
    snn_custom_ops.configure_fused_op("triton", strict_triton=True, verbose=False)
    try:
        if has_residual:
            out_spike, out_v = torch.ops.snn_custom.fused_temporal_batched_linear_add_lif_state.default(
                x_seq, residual_seq, weight, bias, v_init, 1.0, 0.0, tau, detach_reset,
            )
        else:
            out_spike, out_v = torch.ops.snn_custom.fused_temporal_batched_linear_lif_state.default(
                x_seq, weight, bias, v_init, 1.0, 0.0, tau, detach_reset,
            )
    finally:
        os.environ.pop("KAIROS_BATCHED_LINEAR_LIF_BACKEND", None)

    atol = 2e-2 if dtype == torch.float16 else 2e-3
    rtol = 2e-2 if dtype == torch.float16 else 2e-3
    spike_mismatch = (out_spike.float() != ref_spike.float()).float().mean().item()
    v_allclose = torch.allclose(out_v.float(), ref_v.float(), atol=atol, rtol=rtol)
    max_v = (out_v.float() - ref_v.float()).abs().max().item()
    # LIF is a hard threshold -- floating point noise near the threshold can
    # flip a spike bit even when both implementations are "correct". Mirror
    # the tolerance policy used by the rank-3 kernel test.
    ok = spike_mismatch < 1e-2 and v_allclose
    tag = (
        f"backend={backend:<7} T={T:<2} leading={leading!s:<10} shape={in_features}->{out_features:<5} "
        f"bias={bias_enabled} residual={has_residual} tau={tau:g} detach={detach_reset} dtype={dtype}"
    )
    print(f"{'OK  ' if ok else 'FAIL'} {tag} spike_mismatch={spike_mismatch:.4%} max_v_err={max_v:.3e}")
    if not ok:
        raise AssertionError(f"failed case: {tag} spike_mismatch={spike_mismatch:.4%} max_v_err={max_v:.3e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--T", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--backends", nargs="+", default=["codegen", "tc"])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--lif-param-coverage",
        action="store_true",
        help="Also cover tau=1/tau>1 and detach_reset True/False.",
    )
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for strict Triton batched temporal Linear+LIF tests")
    dtype = resolve_dtype(args.dtype)

    leading_cases = [(1, 1), (1, 16), (8, 8), (2, 3, 5)]
    shape_cases = [(384, 1000), (768, 768), (4096, 4096), (300, 700)]
    if args.quick:
        leading_cases = leading_cases[:2]
        shape_cases = shape_cases[:2]

    lif_params = [(2.0, False)]
    if args.lif_param_coverage:
        lif_params = [(1.0, False), (1.0, True), (2.0, False), (2.0, True)]

    for backend in args.backends:
        for T in args.T:
            for leading in leading_cases:
                for in_features, out_features in shape_cases:
                    for bias_enabled in (False, True):
                        for has_residual in (False, True):
                            for tau, detach_reset in lif_params:
                                run_case(
                                    args.device,
                                    dtype,
                                    T,
                                    leading,
                                    in_features,
                                    out_features,
                                    bias_enabled,
                                    has_residual,
                                    tau,
                                    detach_reset,
                                    backend,
                                )


if __name__ == "__main__":
    main()
