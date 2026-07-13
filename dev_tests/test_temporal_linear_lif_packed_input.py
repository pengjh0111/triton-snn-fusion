import argparse
import sys
from pathlib import Path

import torch

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


def run_case(device, dtype, T, batch, in_features, out_features, bias_enabled, tau, detach_reset, strict_triton):
    torch.manual_seed(6060 + T + batch + in_features + out_features + int(bias_enabled))
    x_seq = (torch.randn(T, batch, in_features, device=device, dtype=dtype) * 0.04).contiguous()
    xs = [x_seq[t] for t in range(T)]
    weight = (torch.randn(out_features, in_features, device=device, dtype=dtype) * 0.02).contiguous()
    bias = (
        (torch.randn(out_features, device=device, dtype=dtype) * 0.01).contiguous()
        if bias_enabled
        else None
    )
    v_init = torch.zeros(batch, out_features, device=device, dtype=dtype)

    snn_custom_ops.configure_fused_op("triton", strict_triton=strict_triton, verbose=False)

    snn_custom_ops.reset_fused_op_call_stats()
    old_spike, old_v = torch.ops.snn_custom.fused_temporal_linear_lif_state.default(
        xs,
        weight,
        bias,
        v_init,
        1.0,
        0.0,
        tau,
        detach_reset,
    )
    old_stats = snn_custom_ops.get_fused_op_call_stats()

    snn_custom_ops.reset_fused_op_call_stats()
    packed_spike, packed_v = torch.ops.snn_custom.fused_temporal_linear_lif_state_packed.default(
        x_seq,
        weight,
        bias,
        v_init,
        1.0,
        0.0,
        tau,
        detach_reset,
    )
    packed_stats = snn_custom_ops.get_fused_op_call_stats()

    atol = 1e-2 if dtype == torch.float16 else 2e-3
    rtol = 1e-2 if dtype == torch.float16 else 2e-3
    spike_ok = torch.allclose(packed_spike, old_spike, atol=atol, rtol=rtol)
    v_ok = torch.allclose(packed_v, old_v, atol=atol, rtol=rtol)
    max_spike = (packed_spike - old_spike).abs().max().item()
    max_v = (packed_v - old_v).abs().max().item()
    packed_hit = packed_stats.get("temporal_linear_lif_packed_triton", 0)
    packed_fallback = packed_stats.get("temporal_linear_lif_packed_fallback", 0)
    packed_stack = packed_stats.get("temporal_linear_lif_stack_materialized", 0)
    old_stack = old_stats.get("temporal_linear_lif_stack_materialized", 0)
    ok = spike_ok and v_ok and packed_hit > 0 and packed_fallback == 0 and packed_stack == 0
    print(
        f"T={T:<2} batch={batch:<2} shape={in_features}->{out_features:<5} "
        f"bias={bias_enabled} tau={tau:g} detach={detach_reset} dtype={dtype} "
        f"allclose={ok} max_spike={max_spike:.3e} max_v={max_v:.3e} "
        f"packed_hit={packed_hit} packed_fallback={packed_fallback} "
        f"packed_stack={packed_stack} old_stack={old_stack}"
    )
    if not ok:
        raise AssertionError(
            f"packed Linear+LIF failed: spike_ok={spike_ok}, v_ok={v_ok}, "
            f"packed_stats={packed_stats}, old_stats={old_stats}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--T", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--strict-triton", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for packed temporal Linear+LIF tests")
    dtype = resolve_dtype(args.dtype)
    shapes = [(2, 64, 32), (2, 4096, 4096)]
    if args.quick:
        shapes = [(2, 64, 32)]
        args.T = [1, 4]
    for T in args.T:
        for batch, in_features, out_features in shapes:
            for bias_enabled in (False, True):
                for tau in (1.0, 2.0):
                    for detach_reset in (False, True):
                        run_case(
                            args.device,
                            dtype,
                            T,
                            batch,
                            in_features,
                            out_features,
                            bias_enabled,
                            tau,
                            detach_reset,
                            args.strict_triton,
                        )


if __name__ == "__main__":
    main()
