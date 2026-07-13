import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime import snn_custom_ops


def _dtype_from_arg(name: str):
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(name)


def _reference(x_seq, v_init, v_threshold, v_reset, tau, detach_reset):
    v = v_init
    spikes = []
    for t in range(int(x_seq.shape[0])):
        spike, v = snn_custom_ops.lif_forward_state_torch(
            x_seq[t],
            v,
            v_threshold,
            v_reset,
            tau,
            detach_reset,
        )
        spikes.append(spike)
    return torch.stack(spikes, dim=0), v


def _make_input(T: int, shape: Tuple[int, int, int, int], device: str, dtype: torch.dtype):
    torch.manual_seed(2026 + T + shape[1])
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(2026 + T + shape[1])
    # Keep values away from pathological overflow while still creating spikes.
    x_seq = torch.randn((T,) + shape, device=device, dtype=dtype) * 0.75 + 0.15
    v_init = torch.zeros(shape, device=device, dtype=dtype)
    return x_seq.contiguous(), v_init.contiguous()


def _sync(device: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def run_case(T: int, shape: Tuple[int, int, int, int], args) -> Dict:
    dtype = _dtype_from_arg(args.dtype)
    x_seq, v_init = _make_input(T, shape, args.device, dtype)
    snn_custom_ops.configure_fused_op(
        backend=args.backend,
        strict_triton=args.strict_triton,
        verbose=args.verbose,
    )
    snn_custom_ops.reset_fused_op_call_stats()

    with torch.no_grad():
        ref_spike, ref_v = _reference(x_seq, v_init, args.v_threshold, args.v_reset, args.tau, False)
        out_spike, out_v = torch.ops.snn_custom.fused_temporal_lif_state.default(
            x_seq,
            v_init,
            float(args.v_threshold),
            float(args.v_reset),
            float(args.tau),
            False,
        )
    _sync(args.device)

    spike_diff = (ref_spike - out_spike).abs()
    v_diff = (ref_v - out_v).abs()
    spike_allclose = torch.allclose(ref_spike, out_spike, rtol=args.rtol, atol=args.atol)
    v_allclose = torch.allclose(ref_v, out_v, rtol=args.rtol, atol=args.atol)
    mismatch_ratio = (ref_spike != out_spike).to(torch.float32).mean().item()
    v_bad_ratio = (v_diff > args.atol + args.rtol * ref_v.abs()).to(torch.float32).mean().item()
    stats = snn_custom_ops.get_fused_op_call_stats()
    spike_ok = bool(spike_allclose or mismatch_ratio <= args.spike_mismatch_tol)
    v_ok = bool(v_allclose or v_bad_ratio <= args.spike_mismatch_tol)
    ok = bool(spike_ok and v_ok and not torch.isnan(out_spike).any() and not torch.isnan(out_v).any())
    return {
        "T": T,
        "shape": list(shape),
        "dtype": args.dtype,
        "ok": ok,
        "spike_allclose": bool(spike_allclose),
        "spike_ok": bool(spike_ok),
        "v_allclose": bool(v_allclose),
        "v_ok": bool(v_ok),
        "max_abs_err_spike": float(spike_diff.max().item()),
        "max_abs_err_v": float(v_diff.max().item()),
        "mismatch_ratio": float(mismatch_ratio),
        "v_bad_ratio": float(v_bad_ratio),
        "temporal_lif_triton": int(stats.get("temporal_lif_triton", 0)),
        "temporal_lif_fallback": int(stats.get("temporal_lif_fallback", 0)),
        "fallback_reasons": stats.get("fallback_reasons", {}),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Correctness test for standalone Chronos temporal LIF Triton kernel.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--backend", choices=("torch", "triton"), default="triton")
    parser.add_argument("--T", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--strict-triton", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--v-threshold", type=float, default=1.0)
    parser.add_argument("--v-reset", type=float, default=0.0)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--rtol", type=float, default=None)
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--spike-mismatch-tol", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    if args.rtol is None:
        args.rtol = 1e-2 if args.dtype == "fp16" else 1e-5
    if args.atol is None:
        args.atol = 1e-2 if args.dtype == "fp16" else 1e-5
    if args.spike_mismatch_tol is None:
        args.spike_mismatch_tol = 1e-3 if args.dtype == "fp16" else 0.0

    shapes: List[Tuple[int, int, int, int]] = [
        (args.batch_size, args.channels, args.height, args.width),
        (args.batch_size, 64, 56, 56),
    ]
    seen = set()
    rows = []
    for shape in shapes:
        if shape in seen:
            continue
        seen.add(shape)
        for T in args.T:
            row = run_case(T, shape, args)
            rows.append(row)
            status = "PASS" if row["ok"] else "FAIL"
            print(
                f"[{status}] dtype={row['dtype']} T={T} shape={shape} "
                f"max_spike={row['max_abs_err_spike']:.3e} max_v={row['max_abs_err_v']:.3e} "
                f"mismatch={row['mismatch_ratio']:.3e} v_bad={row['v_bad_ratio']:.3e} "
                f"triton={row['temporal_lif_triton']} fallback={row['temporal_lif_fallback']}"
            )

    failed = [row for row in rows if not row["ok"]]
    if failed:
        raise SystemExit(f"{len(failed)} temporal LIF correctness case(s) failed")


if __name__ == "__main__":
    main()
