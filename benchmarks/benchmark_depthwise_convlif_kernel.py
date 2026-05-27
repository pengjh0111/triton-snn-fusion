import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Callable, Dict

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.snn_custom_ops import lif_forward_state_torch
from runtime.triton_convlif_backend import (
    run_triton_fused_temporal_conv_lif_state,
)


def _dtype(name: str):
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


def _lif_loop(x_seq, weight, bias, v_init, stride: int):
    v = v_init
    spikes = []
    groups = x_seq.shape[2]
    for t in range(x_seq.shape[0]):
        y = F.conv2d(x_seq[t], weight, bias, stride=(stride, stride), padding=(1, 1), groups=groups)
        spike, v = lif_forward_state_torch(y, v, 1.0, 0.0, 2.0, False)
        spikes.append(spike)
    return torch.stack(spikes, dim=0), v


def _make_case(args, stride: int, dtype: torch.dtype):
    torch.manual_seed(2026 + stride + args.T + args.channels)
    x_seq = (torch.randn(args.T, args.batch_size, args.channels, args.height, args.width, device="cuda", dtype=dtype) * 0.03).contiguous()
    weight = (torch.randn(args.channels, 1, 3, 3, device="cuda", dtype=dtype) * 0.03).contiguous()
    bias = (torch.randn(args.channels, device="cuda", dtype=dtype) * 0.03).contiguous()
    v_init = torch.tensor(0.0, device="cuda", dtype=dtype)
    return x_seq, weight, bias, v_init


def run_case(args, stride: int) -> Dict:
    dtype = _dtype(args.dtype)
    x_seq, weight, bias, v_init = _make_case(args, stride, dtype)
    xs = [x_seq[t] for t in range(args.T)]

    def eager_fn():
        return _lif_loop(x_seq, weight, bias, v_init, stride)

    def fixed_fn():
        return run_triton_fused_temporal_conv_lif_state(
            xs,
            weight,
            bias,
            v_init,
            [stride, stride],
            [1, 1],
            [1, 1],
            args.channels,
            1.0,
            0.0,
            2.0,
            False,
            use_autotune=False,
        )

    def autotuned_fn():
        return run_triton_fused_temporal_conv_lif_state(
            xs,
            weight,
            bias,
            v_init,
            [stride, stride],
            [1, 1],
            [1, 1],
            args.channels,
            1.0,
            0.0,
            2.0,
            False,
            use_autotune=True,
        )

    ref_spike, ref_v = eager_fn()
    auto = fixed_fn() if args.skip_autotune else autotuned_fn()
    _sync()
    atol = 1e-2 if args.dtype == "fp16" else 1e-5
    rtol = 1e-2 if args.dtype == "fp16" else 1e-5
    allclose = torch.allclose(auto.spikes, ref_spike, atol=atol, rtol=rtol) and torch.allclose(auto.v_next, ref_v, atol=atol, rtol=rtol)

    eager = _time_cuda(eager_fn, args.warmup, args.repeat)
    if args.skip_compile_loop:
        compiled = {"mean_ms": None, "p50_ms": None, "p90_ms": None, "min_ms": None}
    else:
        compiled_body = torch.compile(eager_fn, fullgraph=False, dynamic=False)
        compiled = _time_cuda(compiled_body, args.warmup, args.repeat)
    fixed = _time_cuda(fixed_fn, args.warmup, args.repeat)
    if args.skip_autotune:
        autotuned = {"mean_ms": None, "p50_ms": None, "p90_ms": None, "min_ms": None}
    else:
        autotuned = _time_cuda(autotuned_fn, args.warmup, args.repeat)
    comparison_ms = fixed["mean_ms"] if autotuned["mean_ms"] is None else autotuned["mean_ms"]
    return {
        "T": args.T,
        "shape": [args.batch_size, args.channels, args.height, args.width],
        "stride": stride,
        "dtype": args.dtype,
        "allclose": bool(allclose),
        "kernel_key": auto.kernel_key,
        "config": auto.kernel_temporal_config,
        "eager_loop": eager,
        "compile_loop": compiled,
        "fixed_triton": fixed,
        "autotuned_triton": autotuned,
        "speedup_vs_eager": eager["mean_ms"] / comparison_ms,
        "speedup_vs_compile": None if compiled["mean_ms"] is None else compiled["mean_ms"] / comparison_ms,
        "speedup_vs_fixed": None if autotuned["mean_ms"] is None else fixed["mean_ms"] / autotuned["mean_ms"],
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark depthwise temporal ConvLIF kernels.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("fp32", "fp16"), default="fp16")
    parser.add_argument("--T", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--skip-compile-loop", action="store_true")
    parser.add_argument("--skip-autotune", action="store_true")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    if args.device != "cuda":
        raise SystemExit("depthwise benchmark requires --device cuda")
    rows = [run_case(args, stride) for stride in (1, 2)]
    print("| stride | key | config | eager_ms | compile_ms | fixed_ms | autotuned_ms | speedup eager | speedup compile | speedup fixed | allclose |")
    print("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        compile_ms = "skip" if row["compile_loop"]["mean_ms"] is None else f"{row['compile_loop']['mean_ms']:.4f}"
        autotuned_ms = "skip" if row["autotuned_triton"]["mean_ms"] is None else f"{row['autotuned_triton']['mean_ms']:.4f}"
        speedup_compile = "skip" if row["speedup_vs_compile"] is None else f"{row['speedup_vs_compile']:.3f}"
        speedup_fixed = "skip" if row["speedup_vs_fixed"] is None else f"{row['speedup_vs_fixed']:.3f}"
        print(
            f"| {row['stride']} | {row['kernel_key']} | {row['config']} | "
            f"{row['eager_loop']['mean_ms']:.4f} | {compile_ms} | "
            f"{row['fixed_triton']['mean_ms']:.4f} | {autotuned_ms} | "
            f"{row['speedup_vs_eager']:.3f} | {speedup_compile} | "
            f"{speedup_fixed} | {row['allclose']} |"
        )
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
