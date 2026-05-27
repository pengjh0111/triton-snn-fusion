import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import runtime.snn_custom_ops as snn_custom_ops
from kernels.generated_temporal_linear_lif_kernel import run_fused_temporal_linear_lif_state_kernel


def resolve_dtype(name: str):
    return torch.float16 if name == "fp16" else torch.float32


def sync(device: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def time_ms(fn, warmup: int, repeat: int, device: str):
    for _ in range(warmup):
        fn()
    sync(device)
    samples = []
    for _ in range(repeat):
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return {
        "mean": statistics.mean(samples),
        "p50": statistics.median(samples),
        "min": min(samples),
        "p90": sorted(samples)[int(0.9 * (len(samples) - 1))],
    }


def eager_loop(xs, weight, bias, v_init):
    v = v_init
    spikes = []
    for x in xs:
        y = F.linear(x, weight, bias)
        spike, v = snn_custom_ops.lif_forward_state_torch(y, v, 1.0, 0.0, 2.0, False)
        spikes.append(spike)
    return torch.stack(spikes, dim=0), v


def run_case(args, dtype, T, batch, in_features, out_features):
    torch.manual_seed(2026 + T + batch + in_features + out_features)
    xs = [(torch.randn(batch, in_features, device=args.device, dtype=dtype) * 0.04).contiguous() for _ in range(T)]
    x_seq = torch.stack(xs, dim=0).contiguous()
    weight = (torch.randn(out_features, in_features, device=args.device, dtype=dtype) * 0.02).contiguous()
    bias = (torch.randn(out_features, device=args.device, dtype=dtype) * 0.01).contiguous()
    v_init = torch.zeros(batch, out_features, device=args.device, dtype=dtype)

    compiled_loop = torch.compile(lambda: eager_loop(xs, weight, bias, v_init), backend="inductor")

    snn_custom_ops.configure_fused_op("triton", strict_triton=args.strict_triton, verbose=False)
    ref_spike, ref_v = eager_loop(xs, weight, bias, v_init)
    snn_custom_ops.reset_fused_op_call_stats()
    old_spike, old_v = torch.ops.snn_custom.fused_temporal_linear_lif_state.default(
        xs, weight, bias, v_init, 1.0, 0.0, 2.0, False
    )
    old_stats = snn_custom_ops.get_fused_op_call_stats()
    snn_custom_ops.reset_fused_op_call_stats()
    tri_spike, tri_v = torch.ops.snn_custom.fused_temporal_linear_lif_state_packed.default(
        x_seq, weight, bias, v_init, 1.0, 0.0, 2.0, False
    )
    packed_op_stats = snn_custom_ops.get_fused_op_call_stats()
    packed_spike, packed_v, packed_diag = run_fused_temporal_linear_lif_state_kernel(
        x_seq,
        weight,
        bias,
        v_init,
        1.0,
        0.0,
        2.0,
        False,
        use_autotune=True,
    )
    fixed_spike, fixed_v, fixed_diag = run_fused_temporal_linear_lif_state_kernel(
        x_seq,
        weight,
        bias,
        v_init,
        1.0,
        0.0,
        2.0,
        False,
        use_autotune=False,
    )
    atol = 1e-2 if dtype == torch.float16 else 2e-3
    rtol = 1e-2 if dtype == torch.float16 else 2e-3
    allclose = torch.allclose(tri_spike, ref_spike, atol=atol, rtol=rtol) and torch.allclose(
        tri_v, ref_v, atol=atol, rtol=rtol
    )
    stats = snn_custom_ops.get_fused_op_call_stats()

    eager_ms = time_ms(lambda: eager_loop(xs, weight, bias, v_init), args.warmup, args.repeat, args.device)
    compile_ms = time_ms(lambda: compiled_loop(), args.warmup, args.repeat, args.device)
    old_custom_triton_ms = time_ms(
        lambda: torch.ops.snn_custom.fused_temporal_linear_lif_state.default(
            xs, weight, bias, v_init, 1.0, 0.0, 2.0, False
        ),
        args.warmup,
        args.repeat,
        args.device,
    )
    packed_custom_triton_ms = time_ms(
        lambda: torch.ops.snn_custom.fused_temporal_linear_lif_state_packed.default(
            x_seq, weight, bias, v_init, 1.0, 0.0, 2.0, False
        ),
        args.warmup,
        args.repeat,
        args.device,
    )
    triton_ms = time_ms(
        lambda: run_fused_temporal_linear_lif_state_kernel(
            x_seq,
            weight,
            bias,
            v_init,
            1.0,
            0.0,
            2.0,
            False,
            use_autotune=True,
        )[:2],
        args.warmup,
        args.repeat,
        args.device,
    )
    fixed_ms = time_ms(
        lambda: run_fused_temporal_linear_lif_state_kernel(
            x_seq,
            weight,
            bias,
            v_init,
            1.0,
            0.0,
            2.0,
            False,
            use_autotune=False,
        )[:2],
        args.warmup,
        args.repeat,
        args.device,
    )
    return {
        "T": T,
        "batch": batch,
        "in_features": in_features,
        "out_features": out_features,
        "dtype": args.dtype,
        "eager_loop_ms": eager_ms,
        "compile_loop_ms": compile_ms,
        "fixed_triton_ms": fixed_ms,
        "old_tensor_list_op_triton_ms": old_custom_triton_ms,
        "packed_op_triton_ms": packed_custom_triton_ms,
        "temporal_linear_lif_triton_ms": triton_ms,
        "speedup_vs_eager": eager_ms["mean"] / triton_ms["mean"] if triton_ms["mean"] else None,
        "speedup_vs_compile": compile_ms["mean"] / triton_ms["mean"] if triton_ms["mean"] else None,
        "speedup_vs_fixed_triton": fixed_ms["mean"] / triton_ms["mean"] if triton_ms["mean"] else None,
        "allclose": bool(allclose),
        "old_op_allclose": bool(
            torch.allclose(old_spike, ref_spike, atol=atol, rtol=rtol)
            and torch.allclose(old_v, ref_v, atol=atol, rtol=rtol)
        ),
        "packed_allclose": bool(
            torch.allclose(packed_spike, ref_spike, atol=atol, rtol=rtol)
            and torch.allclose(packed_v, ref_v, atol=atol, rtol=rtol)
        ),
        "fixed_allclose": bool(
            torch.allclose(fixed_spike, ref_spike, atol=atol, rtol=rtol)
            and torch.allclose(fixed_v, ref_v, atol=atol, rtol=rtol)
        ),
        "max_abs_err_spike": (tri_spike - ref_spike).abs().max().item(),
        "max_abs_err_v": (tri_v - ref_v).abs().max().item(),
        "triton_hit": packed_op_stats.get("temporal_linear_lif_packed_triton", 0),
        "fallback": packed_op_stats.get("temporal_linear_lif_packed_fallback", 0),
        "old_stack_materialized": old_stats.get("temporal_linear_lif_stack_materialized", 0),
        "packed_stack_materialized": packed_op_stats.get("temporal_linear_lif_stack_materialized", 0),
        "fixed_config": fixed_diag,
        "autotuned_config": packed_diag,
        "kernel_temporal_configs": packed_op_stats.get("kernel_temporal_configs", {}),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--T", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--batch-size", nargs="+", type=int, default=[1, 4, 16])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--strict-triton", action="store_true")
    parser.add_argument("--use-autotune", action="store_true", default=True)
    parser.add_argument("--disable-autotune", action="store_true")
    parser.add_argument("--fixed-config", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out-dir", default="/tmp/chronos_temporal_linear_lif_benchmark")
    args = parser.parse_args()
    dtype = resolve_dtype(args.dtype)
    cases = [(512, 512), (1024, 512), (4096, 4096), (4096, 1000), (25088, 4096)]
    if args.quick:
        cases = cases[:2]
        args.T = [4, 16]
        args.batch_size = [1, 4]

    results = []
    print("| T | batch | shape | dtype | eager ms | compile ms | fixed ms | direct packed ms | old op ms | packed op ms | speedup eager | speedup compile | speedup fixed | allclose | packed op stack | old op stack | hit | fallback | config |")
    print("|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|")
    for T in args.T:
        for batch in args.batch_size:
            for in_features, out_features in cases:
                if in_features >= 25088 and batch > 4:
                    continue
                result = run_case(args, dtype, T, batch, in_features, out_features)
                results.append(result)
                print(
                    f"| {T} | {batch} | {in_features}->{out_features} | {args.dtype} | "
                    f"{result['eager_loop_ms']['mean']:.3f} | {result['compile_loop_ms']['mean']:.3f} | "
                    f"{result['fixed_triton_ms']['mean']:.3f} | "
                    f"{result['temporal_linear_lif_triton_ms']['mean']:.3f} | "
                    f"{result['old_tensor_list_op_triton_ms']['mean']:.3f} | "
                    f"{result['packed_op_triton_ms']['mean']:.3f} | "
                    f"{result['speedup_vs_eager']:.2f} | {result['speedup_vs_compile']:.2f} | "
                    f"{result['speedup_vs_fixed_triton']:.2f} | "
                    f"{result['allclose']} | "
                    f"{result['packed_stack_materialized']} | {result['old_stack_materialized']} | "
                    f"{result['triton_hit']} | {result['fallback']} | "
                    f"{result['autotuned_config']} |"
                )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "temporal_linear_lif_benchmark.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[WRITE] {out_dir / 'temporal_linear_lif_benchmark.json'}")


if __name__ == "__main__":
    main()
