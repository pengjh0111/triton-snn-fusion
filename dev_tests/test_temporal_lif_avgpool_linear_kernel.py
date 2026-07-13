import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import runtime.snn_custom_ops as snn_custom_ops


def dtype_from_arg(name):
    return torch.float16 if name == "fp16" else torch.float32


def run_case(T, shape, classes, args):
    dtype = dtype_from_arg(args.dtype)
    torch.manual_seed(2026 + T + shape[1])
    torch.cuda.manual_seed_all(2026 + T + shape[1])
    x = (torch.randn((T,) + shape, device=args.device, dtype=dtype) * 0.75 + 0.15).contiguous()
    v = torch.zeros(shape, device=args.device, dtype=dtype)
    w = (torch.randn(classes, shape[1], device=args.device, dtype=dtype) * 0.05).contiguous()
    b = (torch.randn(classes, device=args.device, dtype=dtype) * 0.05).contiguous()
    snn_custom_ops.configure_fused_op("triton", strict_triton=args.strict_triton, verbose=args.verbose)
    snn_custom_ops.reset_fused_op_call_stats()
    with torch.no_grad():
        ref_out, ref_v = snn_custom_ops.fused_temporal_lif_avgpool_linear_torch(x, v, w, b, 1.0, 0.0, 2.0, False)
        out, out_v = torch.ops.snn_custom.fused_temporal_lif_avgpool_linear.default(x, v, w, b, 1.0, 0.0, 2.0, False)
    torch.cuda.synchronize()
    atol = 1e-2 if args.dtype == "fp16" else 1e-4
    rtol = 1e-2 if args.dtype == "fp16" else 1e-4
    out_diff = (ref_out - out).abs()
    v_diff = (ref_v - out_v).abs()
    out_allclose = torch.allclose(ref_out, out, atol=atol, rtol=rtol)
    v_allclose = torch.allclose(ref_v, out_v, atol=atol, rtol=rtol)
    v_mismatch_ratio = (v_diff > atol).float().mean().item()
    v_ok = v_allclose or (args.dtype == "fp16" and v_mismatch_ratio <= args.max_v_mismatch_ratio)
    ok = out_allclose and v_ok and torch.isfinite(out).all().item() and torch.isfinite(out_v).all().item()
    stats = snn_custom_ops.get_fused_op_call_stats()
    ok = ok and stats.get("temporal_lif_avgpool_linear_triton", 0) > 0 and stats.get("temporal_lif_avgpool_linear_fallback", 0) == 0
    print(
        f"[{'PASS' if ok else 'FAIL'}] dtype={args.dtype} T={T} shape={shape} classes={classes} "
        f"max_out={out_diff.max().item():.3e} max_v={v_diff.max().item():.3e} "
        f"out_allclose={out_allclose} v_allclose={v_allclose} v_mismatch_ratio={v_mismatch_ratio:.3e} "
        f"triton={stats.get('temporal_lif_avgpool_linear_triton', 0)} fallback={stats.get('temporal_lif_avgpool_linear_fallback', 0)}"
    )
    if not ok:
        raise RuntimeError(f"temporal LIF avgpool linear failed: stats={stats}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--T", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--max-v-mismatch-ratio", type=float, default=1e-3)
    parser.add_argument("--strict-triton", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    shapes = [(args.batch_size, args.channels, args.height, args.width), (args.batch_size, 64, 7, 7)]
    for shape in shapes:
        for T in args.T:
            run_case(T, shape, args.classes, args)


if __name__ == "__main__":
    main()
