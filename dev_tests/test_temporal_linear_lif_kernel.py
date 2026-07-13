import argparse
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


def reference(xs, weight, bias, v_init, tau, detach_reset):
    v = v_init
    spikes = []
    for x in xs:
        y = F.linear(x, weight, bias)
        spike, v = snn_custom_ops.lif_forward_state_torch(
            y,
            v,
            1.0,
            0.0,
            tau,
            detach_reset,
        )
        spikes.append(spike)
    return torch.stack(spikes, dim=0), v


def run_case(
    device,
    dtype,
    T,
    batch,
    in_features,
    out_features,
    bias_enabled,
    tau,
    detach_reset,
    strict_triton,
):
    torch.manual_seed(2026 + T + batch + in_features + out_features)
    xs = [
        (torch.randn(batch, in_features, device=device, dtype=dtype) * 0.04).contiguous()
        for _ in range(T)
    ]
    weight = (torch.randn(out_features, in_features, device=device, dtype=dtype) * 0.02).contiguous()
    bias = (
        (torch.randn(out_features, device=device, dtype=dtype) * 0.01).contiguous()
        if bias_enabled
        else None
    )
    v_init = torch.zeros(batch, out_features, device=device, dtype=dtype)

    ref_spike, ref_v = reference(xs, weight, bias, v_init, tau, detach_reset)
    snn_custom_ops.reset_fused_op_call_stats()
    snn_custom_ops.configure_fused_op("triton", strict_triton=strict_triton, verbose=False)
    out_spike, out_v = torch.ops.snn_custom.fused_temporal_linear_lif_state.default(
        xs,
        weight,
        bias,
        v_init,
        1.0,
        0.0,
        tau,
        detach_reset,
    )
    atol = 1e-2 if dtype == torch.float16 else 2e-3
    rtol = 1e-2 if dtype == torch.float16 else 2e-3
    spike_allclose = torch.allclose(out_spike, ref_spike, atol=atol, rtol=rtol)
    v_allclose = torch.allclose(out_v, ref_v, atol=atol, rtol=rtol)
    max_spike = (out_spike - ref_spike).abs().max().item()
    max_v = (out_v - ref_v).abs().max().item()
    stats = snn_custom_ops.get_fused_op_call_stats()
    ok = spike_allclose and v_allclose and stats.get("temporal_linear_lif_triton", 0) > 0
    print(
        f"T={T:<2} batch={batch:<2} shape={in_features}->{out_features:<5} "
        f"bias={bias_enabled} tau={tau:g} detach={detach_reset} "
        f"dtype={dtype} allclose={ok} "
        f"max_spike={max_spike:.3e} max_v={max_v:.3e} "
        f"triton={stats.get('temporal_linear_lif_triton', 0)} "
        f"fallback={stats.get('temporal_linear_lif_fallback', 0)}"
    )
    if not ok:
        raise AssertionError(
            f"failed case T={T}, batch={batch}, shape={in_features}->{out_features}, "
            f"bias={bias_enabled}, tau={tau}, detach_reset={detach_reset}, "
            f"spike_allclose={spike_allclose}, v_allclose={v_allclose}, stats={stats}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--T", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--batch-size", nargs="+", type=int, default=[1, 4, 16])
    parser.add_argument("--strict-triton", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--lif-param-coverage",
        action="store_true",
        help="Also cover tau=1/tau>1 and detach_reset True/False.",
    )
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for strict Triton temporal Linear+LIF tests")
    dtype = resolve_dtype(args.dtype)
    cases = [
        (512, 512),
        (1024, 512),
        (4096, 4096),
        (4096, 1000),
        (25088, 4096),
    ]
    if args.quick:
        cases = cases[:2]
        args.batch_size = [1, 4]
    lif_params = [(2.0, False)]
    if args.lif_param_coverage:
        lif_params = [(1.0, False), (1.0, True), (2.0, False), (2.0, True)]
    for T in args.T:
        for batch in args.batch_size:
            for in_features, out_features in cases:
                if in_features >= 25088 and batch > 4:
                    continue
                for bias_enabled in (False, True):
                    for tau, detach_reset in lif_params:
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
