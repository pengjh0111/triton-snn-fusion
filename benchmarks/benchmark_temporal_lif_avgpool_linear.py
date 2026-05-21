import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import runtime.snn_custom_ops as snn_custom_ops


SHAPES = [(16, 64, 56, 56), (16, 128, 28, 28), (16, 256, 14, 14), (16, 512, 7, 7)]


def dtype_from_arg(name):
    return torch.float16 if name == "fp16" else torch.float32


def ref_avgpool_linear(x, v, w, b):
    return snn_custom_ops.fused_temporal_lif_avgpool_linear_torch(x, v, w, b, 1.0, 0.0, 2.0, False)


def time_cuda(fn, warmup, repeat):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    return {"mean_ms": statistics.mean(times), "min_ms": min(times), "p50_ms": sorted(times)[len(times)//2]}


def run_case(T, shape, args):
    dtype = dtype_from_arg(args.dtype)
    classes = args.classes
    x = (torch.randn((T,) + shape, device="cuda", dtype=dtype) * 0.75 + 0.15).contiguous()
    v = torch.zeros(shape, device="cuda", dtype=dtype)
    w = (torch.randn(classes, shape[1], device="cuda", dtype=dtype) * 0.05).contiguous()
    b = (torch.randn(classes, device="cuda", dtype=dtype) * 0.05).contiguous()

    compiled = torch.compile(lambda a, vv, ww, bb: ref_avgpool_linear(a, vv, ww, bb), fullgraph=False, dynamic=False)
    snn_custom_ops.configure_fused_op("triton", strict_triton=args.strict_triton, verbose=args.verbose)

    def eager_fn():
        return ref_avgpool_linear(x, v, w, b)

    def compile_fn():
        return compiled(x, v, w, b)

    def triton_fn():
        return torch.ops.snn_custom.fused_temporal_lif_avgpool_linear.default(x, v, w, b, 1.0, 0.0, 2.0, False)

    with torch.no_grad():
        ref = eager_fn()
        got = triton_fn()
    torch.cuda.synchronize()
    atol = 1e-2 if args.dtype == "fp16" else 1e-4
    rtol = 1e-2 if args.dtype == "fp16" else 1e-4
    out_diff = (ref[0] - got[0]).abs()
    v_diff = (ref[1] - got[1]).abs()
    out_allclose = torch.allclose(ref[0], got[0], atol=atol, rtol=rtol)
    v_allclose = torch.allclose(ref[1], got[1], atol=atol, rtol=rtol)
    v_mismatch_ratio = (v_diff > atol).float().mean().item()
    allclose = out_allclose and (v_allclose or (args.dtype == "fp16" and v_mismatch_ratio <= args.max_v_mismatch_ratio))
    max_err = max(out_diff.max().item(), v_diff.max().item())
    eager = time_cuda(eager_fn, args.warmup, args.repeat)
    comp = time_cuda(compile_fn, args.warmup, args.repeat)
    snn_custom_ops.reset_fused_op_call_stats()
    tri = time_cuda(triton_fn, args.warmup, args.repeat)
    stats = snn_custom_ops.get_fused_op_call_stats()
    return {
        "T": T,
        "shape": list(shape),
        "dtype": args.dtype,
        "classes": classes,
        "eager_loop_ms": eager["mean_ms"],
        "compile_loop_ms": comp["mean_ms"],
        "temporal_lif_avgpool_linear_triton_ms": tri["mean_ms"],
        "speedup_vs_eager": eager["mean_ms"] / tri["mean_ms"],
        "speedup_vs_compile": comp["mean_ms"] / tri["mean_ms"],
        "allclose": bool(allclose),
        "out_allclose": bool(out_allclose),
        "v_allclose": bool(v_allclose),
        "v_mismatch_ratio": float(v_mismatch_ratio),
        "max_abs_error_out": float(out_diff.max().item()),
        "max_abs_error_v": float(v_diff.max().item()),
        "max_abs_error": float(max_err),
        "triton_hit": stats.get("temporal_lif_avgpool_linear_triton", 0),
        "fallback": stats.get("temporal_lif_avgpool_linear_fallback", 0),
        "fallback_reasons": stats.get("fallback_reasons", {}),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--T", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--classes", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--strict-triton", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--max-v-mismatch-ratio", type=float, default=1e-3)
    parser.add_argument("--out-dir", default="temporal_lif_avgpool_linear_benchmark")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    shapes = [SHAPES[-1]] if args.quick else SHAPES
    rows = []
    for shape in shapes:
        for T in args.T:
            print(f"[BENCH] dtype={args.dtype} T={T} shape={shape}")
            rows.append(run_case(T, shape, args))
    print("| T | shape | dtype | eager | compile | triton | speedup compile | allclose | out_allclose | v_mismatch | hit | fallback |")
    print("|---:|---|---|---:|---:|---:|---:|---|---|---:|---:|---:|")
    for r in rows:
        print(f"| {r['T']} | {r['shape']} | {r['dtype']} | {r['eager_loop_ms']:.3f} | {r['compile_loop_ms']:.3f} | {r['temporal_lif_avgpool_linear_triton_ms']:.3f} | {r['speedup_vs_compile']:.3f} | {r['allclose']} | {r['out_allclose']} | {r['v_mismatch_ratio']:.3e} | {r['triton_hit']} | {r['fallback']} |")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "temporal_lif_avgpool_linear_benchmark.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
