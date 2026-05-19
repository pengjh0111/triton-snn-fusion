import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime import snn_custom_ops

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


RESNET_LIKE_SHAPES = [
    (16, 64, 56, 56),
    (16, 128, 28, 28),
    (16, 256, 14, 14),
    (16, 512, 7, 7),
]


def _dtype_from_arg(name: str):
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(name)


def _sync():
    torch.cuda.synchronize()


def _time_cuda(fn: Callable, warmup: int, repeat: int) -> Dict[str, float]:
    for _ in range(warmup):
        fn()
    _sync()
    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        _sync()
        times.append(float(start.elapsed_time(end)))
    sorted_times = sorted(times)
    return {
        "mean_ms": float(statistics.mean(times)),
        "p50_ms": float(sorted_times[len(sorted_times) // 2]),
        "p90_ms": float(sorted_times[int(0.9 * (len(sorted_times) - 1))]),
        "min_ms": float(min(times)),
    }


def temporal_lif_loop_torch(x_seq, v_init, v_threshold: float, v_reset: float, tau: float):
    v = v_init
    spikes = []
    for t in range(int(x_seq.shape[0])):
        spike, v = snn_custom_ops.lif_forward_state_torch(x_seq[t], v, v_threshold, v_reset, tau, False)
        spikes.append(spike)
    return torch.stack(spikes, dim=0), v


def _make_input(T: int, shape: Tuple[int, int, int, int], dtype: torch.dtype):
    torch.manual_seed(2026 + T + shape[1])
    torch.cuda.manual_seed_all(2026 + T + shape[1])
    x_seq = torch.randn((T,) + shape, device="cuda", dtype=dtype) * 0.75 + 0.15
    v_init = torch.zeros(shape, device="cuda", dtype=dtype)
    return x_seq.contiguous(), v_init.contiguous()


def run_case(T: int, shape: Tuple[int, int, int, int], args) -> Dict:
    dtype = _dtype_from_arg(args.dtype)
    x_seq, v_init = _make_input(T, shape, dtype)
    v_threshold = 1.0
    v_reset = 0.0
    tau = 2.0

    def eager_fn():
        return temporal_lif_loop_torch(x_seq, v_init, v_threshold, v_reset, tau)

    def compiled_body(x_arg, v_arg):
        return temporal_lif_loop_torch(x_arg, v_arg, v_threshold, v_reset, tau)

    compiled_body = torch.compile(compiled_body, fullgraph=False, dynamic=False)

    def compile_fn():
        return compiled_body(x_seq, v_init)

    snn_custom_ops.configure_fused_op(
        backend="triton",
        strict_triton=args.strict_triton,
        verbose=args.verbose,
    )

    def triton_fn():
        return torch.ops.snn_custom.fused_temporal_lif_state.default(
            x_seq,
            v_init,
            v_threshold,
            v_reset,
            tau,
            False,
        )

    with torch.no_grad():
        ref_spike, ref_v = eager_fn()
        out_spike, out_v = triton_fn()
    _sync()
    atol = 1e-2 if args.dtype == "fp16" else 1e-5
    rtol = 1e-2 if args.dtype == "fp16" else 1e-5
    spike_diff = (ref_spike - out_spike).abs()
    v_diff = (ref_v - out_v).abs()
    max_abs_err = max(spike_diff.max().item(), v_diff.max().item())
    mismatch_ratio = (ref_spike != out_spike).to(torch.float32).mean().item()
    v_bad_ratio = (v_diff > atol + rtol * ref_v.abs()).to(torch.float32).mean().item()
    spike_ok = torch.allclose(ref_spike, out_spike, rtol=rtol, atol=atol) or (
        args.dtype == "fp16" and mismatch_ratio <= args.spike_mismatch_tol
    )
    v_ok = torch.allclose(ref_v, out_v, rtol=rtol, atol=atol) or (
        args.dtype == "fp16" and v_bad_ratio <= args.spike_mismatch_tol
    )
    allclose = bool(spike_ok and v_ok)

    eager_stats = _time_cuda(eager_fn, args.warmup, args.repeat)
    compile_stats = _time_cuda(compile_fn, args.warmup, args.repeat)
    snn_custom_ops.reset_fused_op_call_stats()
    triton_stats = _time_cuda(triton_fn, args.warmup, args.repeat)
    stats = snn_custom_ops.get_fused_op_call_stats()

    return {
        "T": T,
        "shape": list(shape),
        "dtype": args.dtype,
        "eager_loop": eager_stats,
        "compile_loop": compile_stats,
        "temporal_lif_triton": triton_stats,
        "speedup_vs_eager_loop": eager_stats["mean_ms"] / triton_stats["mean_ms"],
        "speedup_vs_compile_loop": compile_stats["mean_ms"] / triton_stats["mean_ms"],
        "allclose": bool(allclose),
        "max_abs_err": float(max_abs_err),
        "spike_mismatch_ratio": float(mismatch_ratio),
        "v_bad_ratio": float(v_bad_ratio),
        "triton_hit": int(stats.get("temporal_lif_triton", 0)),
        "fallback": int(stats.get("temporal_lif_fallback", 0)),
        "fallback_reasons": stats.get("fallback_reasons", {}),
    }


def _print_table(rows: List[Dict]):
    print("| T | shape | dtype | eager_loop_ms | compile_loop_ms | temporal_lif_triton_ms | speedup eager | speedup compile | allclose | triton_hit | fallback |")
    print("|---:|---|---|---:|---:|---:|---:|---:|---|---:|---:|")
    for row in rows:
        print(
            f"| {row['T']} | {row['shape']} | {row['dtype']} | "
            f"{row['eager_loop']['mean_ms']:.4f} | {row['compile_loop']['mean_ms']:.4f} | "
            f"{row['temporal_lif_triton']['mean_ms']:.4f} | "
            f"{row['speedup_vs_eager_loop']:.3f} | {row['speedup_vs_compile_loop']:.3f} | "
            f"{row['allclose']} | {row['triton_hit']} | {row['fallback']} |"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark standalone Chronos temporal LIF kernel.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--T", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--height", type=int, default=56)
    parser.add_argument("--width", type=int, default=56)
    parser.add_argument("--single-shape", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--strict-triton", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--spike-mismatch-tol", type=float, default=1e-3)
    parser.add_argument("--out-dir", default="temporal_lif_kernel_benchmark")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires --device cuda and torch.cuda.is_available()")
    shapes = [(args.batch_size, args.channels, args.height, args.width)] if args.single_shape else RESNET_LIKE_SHAPES
    rows = []
    print(
        "[Benchmark Config] "
        f"dtype={args.dtype} TF32_matmul={torch.backends.cuda.matmul.allow_tf32} "
        f"TF32_cudnn={torch.backends.cudnn.allow_tf32} precision={torch.get_float32_matmul_precision()}"
    )
    for shape in shapes:
        for T in args.T:
            print(f"[BENCH] dtype={args.dtype} T={T} shape={shape}")
            rows.append(run_case(T, shape, args))
    _print_table(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "temporal_lif_kernel_benchmark.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"[WRITE] {out_dir / 'temporal_lif_kernel_benchmark.json'}")


if __name__ == "__main__":
    main()
