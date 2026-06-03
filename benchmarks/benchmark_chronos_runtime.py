import argparse
import copy
import json
import statistics
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

import runtime.snn_custom_ops as snn_custom_ops
from compiler.chronos_compile import (
    build_chronos_compile_config,
    compile_with_chronos_options,
    diff_compile_counters,
    snapshot_compile_counters,
    summarize_cudagraph_check,
)
from benchmarks.validate_chronos_baselines import (
    CHRONOS_MODEL_CHOICES,
    LIF_IMPL_CHOICES,
    MultiStepModeWrapper,
    SingleStepModeLoopWrapper,
    RewriteCounters,
    make_resnet_layer,
    make_rewrite_backend,
    reset_lif_modules,
    synchronize_if_needed,
)


@dataclass
class BenchResult:
    name: str
    ok: bool
    mean_ms: Optional[float] = None
    min_ms: Optional[float] = None
    max_ms: Optional[float] = None
    p50_ms: Optional[float] = None
    p90_ms: Optional[float] = None
    repeat: int = 0
    error: str = ""


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    idx = int(round((len(values) - 1) * q))
    return values[idx]


def resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {dtype_name}")


def prepare_runnable(name, model, compile_mode, backend, device, args):
    model.eval()
    reset_lif_modules(model)

    if compile_mode:
        runnable = compile_with_chronos_options(
            model,
            backend=backend if backend is not None else "inductor",
            enable_cudagraphs=args.enable_cudagraphs,
            cudagraph_mode=args.cudagraph_mode,
            fullgraph=False,
            dynamic=False,
        )
    else:
        runnable = model

    return runnable


def _mark_cudagraph_step(args):
    if not getattr(args, "enable_cudagraphs", False):
        return
    mark_step = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
    if mark_step is not None:
        mark_step()


def compile_and_warmup(runnable, model, x, device, warmup, args):
    _mark_cudagraph_step(args)
    reset_lif_modules(model)

    synchronize_if_needed(device)
    with torch.no_grad():
        _ = runnable(x)
    synchronize_if_needed(device)

    for _ in range(warmup):
        _mark_cudagraph_step(args)
        reset_lif_modules(model)

        synchronize_if_needed(device)
        with torch.no_grad():
            _ = runnable(x)
        synchronize_if_needed(device)


def benchmark_runnable(name, runnable, model, x, device, repeat, args):
    times = []

    try:
        for _ in range(repeat):
            _mark_cudagraph_step(args)
            reset_lif_modules(model)

            synchronize_if_needed(device)

            t0 = time.perf_counter()

            with torch.no_grad():
                _ = runnable(x)

            synchronize_if_needed(device)

            t1 = time.perf_counter()

            times.append((t1 - t0) * 1000.0)

        return BenchResult(
            name=name,
            ok=True,
            mean_ms=statistics.mean(times),
            min_ms=min(times),
            max_ms=max(times),
            p50_ms=percentile(times, 0.50),
            p90_ms=percentile(times, 0.90),
            repeat=repeat,
        )

    except Exception:
        return BenchResult(
            name=name,
            ok=False,
            repeat=repeat,
            error=traceback.format_exc(),
        )


def run_case(case_name, model, x, device, compile_mode, backend, warmup, repeat, args):
    print(f"[BENCH] {case_name}")

    try:
        runnable = prepare_runnable(
            case_name,
            model,
            compile_mode,
            backend,
            device,
            args,
        )

        compile_and_warmup(
            runnable,
            model,
            x,
            device,
            warmup,
            args,
        )

        return benchmark_runnable(
            case_name,
            runnable,
            model,
            x,
            device,
            repeat,
            args,
        )

    except Exception:
        return BenchResult(
            name=case_name,
            ok=False,
            repeat=repeat,
            error=traceback.format_exc(),
        )


def benchmark_one_model(model_name: str, args) -> Dict[str, Any]:
    print(f"\n================ {model_name} ================")

    dtype = resolve_dtype(args.dtype)
    _, compile_config = build_chronos_compile_config(
        backend="inductor",
        enable_cudagraphs=args.enable_cudagraphs,
        cudagraph_mode=args.cudagraph_mode,
        fullgraph=False,
        dynamic=False,
    )
    print(
        "[Baseline Config] "
        f"dtype={args.dtype} "
        f"matmul_allow_tf32={torch.backends.cuda.matmul.allow_tf32} "
        f"cudnn_allow_tf32={torch.backends.cudnn.allow_tf32} "
        f"float32_matmul_precision={torch.get_float32_matmul_precision()}"
    )
    print(f"[Compile Summary Config] {compile_config}")

    torch.manual_seed(args.seed)

    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)

    base_layer_s = make_resnet_layer(
        model_name,
        allow_resnet32_fallback=not args.require_direct_resnet32_api,
        step_mode="s",
        model_channels=args.model_channels,
        lif_impl=args.lif_impl,
    )
    base_layer_m = make_resnet_layer(
        model_name,
        allow_resnet32_fallback=not args.require_direct_resnet32_api,
        step_mode="m",
        model_channels=args.model_channels,
        lif_impl=args.lif_impl,
    )

    base_layer_s = base_layer_s.to(
        device=args.device,
        dtype=dtype,
    ).eval()
    base_layer_m = base_layer_m.to(
        device=args.device,
        dtype=dtype,
    ).eval()

    x = torch.randn(
        args.batch_size,
        3,
        args.height,
        args.width,
        device=args.device,
        dtype=dtype,
    )

    snn_custom_ops.configure_fused_op(
        backend=args.fused_op_backend,
        strict_triton=args.strict_triton,
        verbose=args.print_fused_op_calls,
        use_triton_autotune=not args.disable_triton_autotune,
    )

    # out_dir = Path(args.out_dir) / model_name
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = {}
    execution_modes = {}

    if args.include_s_cases and not args.chronos_only:
        cases["baseline_single_step_mode_eager"] = (
            SingleStepModeLoopWrapper(copy.deepcopy(base_layer_s), args.T).to(
                device=args.device,
                dtype=dtype,
            ).eval(),
            False,
            None,
        )
        execution_modes["baseline_single_step_mode_eager"] = "single_step_mode_loop"

        cases["baseline_single_step_mode_compile"] = (
            SingleStepModeLoopWrapper(copy.deepcopy(base_layer_s), args.T).to(
                device=args.device,
                dtype=dtype,
            ).eval(),
            True,
            None,
        )
        execution_modes["baseline_single_step_mode_compile"] = "single_step_mode_loop"

    if not args.chronos_only:
        cases["baseline_multi_step_mode_eager"] = (
            MultiStepModeWrapper(
                copy.deepcopy(base_layer_m),
                args.T,
            ).to(
                device=args.device,
                dtype=dtype,
            ).eval(),
            False,
            None,
        )
        execution_modes["baseline_multi_step_mode_eager"] = "multi_step_mode_native"

        cases["baseline_multi_step_mode_compile"] = (
            MultiStepModeWrapper(
                copy.deepcopy(base_layer_m),
                args.T,
            ).to(
                device=args.device,
                dtype=dtype,
            ).eval(),
            True,
            None,
        )
        execution_modes["baseline_multi_step_mode_compile"] = "multi_step_mode_native"

    chronos_rewrite_counters = {}

    if not args.baseline_only:
        if args.sweep_temporal_windows:
            candidate_windows = args.temporal_window_candidates
        else:
            candidate_windows = [args.temporal_fuse_window]

        candidate_windows = [
            w for w in candidate_windows
            if w <= args.T and args.T % w == 0
        ]

        for tw in candidate_windows:
            local_args = copy.deepcopy(args)
            local_args.temporal_fuse_window = tw

            if local_args.temporal_schedule_window is None:
                local_args.temporal_schedule_window = tw

            rewrite_counters = RewriteCounters()

            chronos_backend = make_rewrite_backend(
                local_args,
                out_dir / f"chronos_single_step_loop_compile_w{tw}",
                rewrite_counters,
            )

            case_name = f"chronos_single_step_loop_compile_w{tw}"

            cases[case_name] = (
                SingleStepModeLoopWrapper(
                    copy.deepcopy(base_layer_s),
                    local_args.T,
                ).to(
                    device=local_args.device,
                    dtype=dtype,
                ).eval(),
                True,
                chronos_backend,
            )
            execution_modes[case_name] = "chronos_single_step_loop_temporal_fusion"

            chronos_rewrite_counters[case_name] = rewrite_counters

    results = {}
    summary_rows = []
    fused_stats_by_case = {}
    cudagraph_status_by_case = {}

    for case_name, (model, compile_mode, backend) in cases.items():
        snn_custom_ops.reset_fused_op_call_stats()
        counter_before = snapshot_compile_counters()
        graph_count_before = None
        if case_name in chronos_rewrite_counters:
            graph_count_before = chronos_rewrite_counters[case_name].captured_graphs

        result = run_case(
            case_name,
            model,
            x,
            args.device,
            compile_mode,
            backend,
            args.warmup,
            args.repeat,
            args,
        )

        case_fused_stats = snn_custom_ops.get_fused_op_call_stats()
        counter_after = snapshot_compile_counters()
        counter_diff = diff_compile_counters(counter_before, counter_after)
        graph_count = None
        if case_name in chronos_rewrite_counters:
            graph_count = chronos_rewrite_counters[case_name].captured_graphs - int(graph_count_before or 0)
        else:
            graph_count = counter_diff.get("stats", {}).get("unique_graphs")
        cudagraph_status = summarize_cudagraph_check(
            model=model_name,
            case=case_name,
            compile_config=compile_config,
            compile_mode=compile_mode,
            device=args.device,
            graph_count=graph_count,
            counter_diff=counter_diff,
        )
        cudagraph_status_by_case[case_name] = cudagraph_status
        fused_stats_by_case[case_name] = case_fused_stats
        results[case_name] = asdict(result)
        results[case_name]["execution_mode"] = execution_modes.get(case_name, "")
        results[case_name]["cudagraph_status"] = cudagraph_status

        if result.ok:
            print(
                f"  {case_name:28s} "
                f"mean={result.mean_ms:.3f} ms "
                f"p50={result.p50_ms:.3f} ms "
                f"p90={result.p90_ms:.3f} ms "
                f"min={result.min_ms:.3f} ms"
            )

            summary_rows.append({
                "case": case_name,
                "mean_ms": result.mean_ms,
                "p50_ms": result.p50_ms,
                "p90_ms": result.p90_ms,
                "fused_stats": case_fused_stats,
            })

        else:
            print(f"  {case_name:28s} FAIL")

            if result.error:
                print(result.error.splitlines()[-1])

    fused_stats = snn_custom_ops.get_fused_op_call_stats()

    print("\n[AUTOTUNE SUMMARY]")
    print(f"{'case':32s} {'mean(ms)':>12s} {'speedup':>12s}")

    baseline = None

    if "baseline_multi_step_mode_compile" in results:
        baseline = results["baseline_multi_step_mode_compile"]["mean_ms"]

    sorted_rows = sorted(summary_rows, key=lambda x: x["mean_ms"])

    for row in sorted_rows:
        speedup = ""

        if baseline is not None:
            speedup = f"{baseline / row['mean_ms']:.3f}x"

        print(
            f"{row['case']:32s} "
            f"{row['mean_ms']:12.3f} "
            f"{speedup:>12s}"
        )

        stats = row.get("fused_stats") or {}

        if stats.get("total", 0):
            print(
                f"{'':32s} triton={stats.get('triton', 0)} "
                f"fallback={stats.get('fallback', 0)} "
                f"temporal_triton={stats.get('temporal_triton', 0)} "
                f"temporal_fallback={stats.get('temporal_fallback', 0)} "
                f"fallback_reasons={stats.get('fallback_reasons', {})}"
            )

            kernel_temporal_configs = stats.get("kernel_temporal_configs")
            if kernel_temporal_configs:
                print(
                    f"{'':32s} "
                    f"kernel_temporal_configs={kernel_temporal_configs}"
                )

    best_case = None

    if sorted_rows:
        best_case = sorted_rows[0]

        print(
            f"\n[BEST CONFIG] "
            f"{best_case['case']} "
            f"mean={best_case['mean_ms']:.3f} ms"
        )

    payload = {
        "model": model_name,
        "input_shape": [
            args.batch_size,
            3,
            args.height,
            args.width,
        ],
        "model_channels": args.model_channels,
        "lif_impl": args.lif_impl,
        "T": args.T,
        "dtype": args.dtype,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "fused_op_backend": args.fused_op_backend,
        "enable_cudagraphs": args.enable_cudagraphs,
        "cudagraph_mode": args.cudagraph_mode,
        "compile_mode": compile_config["compile_mode"],
        "compile_options": compile_config["compile_options"],
        "candidate_windows": candidate_windows,
        "execution_mode": execution_modes,
        "results": results,
        "chronos_rewrite_counters": {
            k: asdict(v)
            for k, v in chronos_rewrite_counters.items()
        },
        "best_case": best_case,
        "fused_op_call_stats_last_case": fused_stats,
        "fused_op_call_stats_by_case": fused_stats_by_case,
        "cudagraph_status_by_case": cudagraph_status_by_case,
    }

    write_path = out_dir / "benchmark_summary.json"

    write_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"\n  wrote: {write_path}")

    return payload


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chronos runtime benchmark with temporal autotune."
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=["resnet18"],
        choices=CHRONOS_MODEL_CHOICES,
    )

    parser.add_argument(
        "--model-channels",
        type=int,
        default=64,
        help="Base channel width for handcrafted alexnet/zfnet models.",
    )

    parser.add_argument(
        "--lif-impl",
        choices=LIF_IMPL_CHOICES,
        default="chronos",
        help="LIF implementation used when constructing benchmark models.",
    )

    parser.add_argument("--T", type=int, default=16)

    parser.add_argument("--batch-size", type=int, default=16)

    parser.add_argument("--height", type=int, default=224)

    parser.add_argument("--width", type=int, default=224)

    parser.add_argument("--device", default="cuda")

    parser.add_argument(
        "--dtype",
        choices=("fp32", "fp16"),
        default="fp32",
    )

    parser.add_argument("--warmup", type=int, default=10)

    parser.add_argument("--repeat", type=int, default=50)

    parser.add_argument(
        "--fused-op-backend",
        choices=("torch", "triton"),
        default="triton",
    )

    parser.add_argument(
        "--rewrite-backend-mode",
        choices=("eager", "inductor", "standalone"),
        default="inductor",
    )
    parser.add_argument("--fx-standalone-streams", type=int, default=1)
    parser.add_argument("--fx-standalone-cudagraph", action="store_true")
    parser.add_argument("--fx-standalone-debug", action="store_true")
    parser.add_argument("--fx-standalone-schedule-policy", choices=("topo", "ready"), default="topo")

    parser.add_argument("--strict-triton", action="store_true")

    parser.add_argument("--print-fused-op-calls", action="store_true")

    parser.add_argument("--disable-triton-autotune", action="store_true")

    parser.add_argument("--enable-temporal-rewrite", action="store_true")

    parser.add_argument("--enable-temporal-schedule", action="store_true")

    parser.add_argument("--temporal-fuse-window", type=int, default=8)

    parser.add_argument("--temporal-schedule-window", type=int, default=None)

    parser.add_argument("--temporal-allow-tail", action="store_true")

    parser.add_argument("--temporal-schedule-dump", action="store_true")

    parser.add_argument("--temporal-schedule-strict", action="store_true")

    parser.add_argument("--enable-spatial-batching", action="store_true")

    parser.add_argument(
        "--spatial-batching-ops",
        nargs="+",
        default=["conv", "bn", "add", "maxpool", "avgpool", "flatten", "linear", "elementwise", "view"],
        choices=["conv", "bn", "add", "maxpool", "linear", "flatten", "avgpool", "elementwise", "view"],
    )

    parser.add_argument("--spatial-batching-dump", action="store_true")

    parser.add_argument("--spatial-batching-strict", action="store_true")

    parser.add_argument("--disable-spatial-batching-chain", action="store_true", help=argparse.SUPPRESS)

    parser.add_argument("--enable-cudagraphs", action="store_true")

    parser.add_argument(
        "--cudagraph-mode",
        choices=("reduce-overhead", "triton-option", "both"),
        default="reduce-overhead",
    )

    parser.add_argument(
        "--sweep-temporal-windows",
        action="store_true",
    )

    parser.add_argument(
        "--temporal-window-candidates",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16],
    )

    parser.add_argument("--disable-rewrite", action="store_true")

    parser.add_argument("--disable-conv-bn-lif", action="store_true")

    parser.add_argument("--disable-temporal-lif-avgpool-linear-rewrite", action="store_true")
    parser.add_argument(
        "--disable-temporal-lif-tail-rewrite",
        action="store_true",
        dest="disable_temporal_lif_avgpool_linear_rewrite",
        help=argparse.SUPPRESS,
    )

    parser.add_argument("--disable-temporal-lif-rewrite", action="store_true")
    parser.add_argument("--disable-temporal-linear-lif-rewrite", action="store_true")
    parser.add_argument("--drop-intermediate-states", action="store_true")
    parser.add_argument("--enable-temporal-mean-rewrite", action="store_true")

    parser.add_argument("--max-patterns", type=int, default=1000)

    parser.add_argument("--include-s-cases", action="store_true")

    parser.add_argument(
        "--chronos-only",
        action="store_true",
        help="Run only Chronos temporal-fusion cases and skip all baseline cases.",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Run only baseline cases and skip Chronos temporal-fusion cases.",
    )

    parser.add_argument("--require-direct-resnet32-api", action="store_true")

    parser.add_argument("--out-dir", default="chronos_benchmark")

    parser.add_argument("--seed", type=int, default=2026)

    parser.add_argument("--rtol", type=float, default=1e-4)

    parser.add_argument("--atol", type=float, default=1e-4)

    return parser.parse_args()


def main():
    args = parse_args()
    if args.chronos_only and args.baseline_only:
        raise ValueError("--chronos-only and --baseline-only are mutually exclusive")
    if args.rewrite_backend_mode == "standalone" and args.fx_standalone_cudagraph and args.enable_cudagraphs:
        print(
            "[FX_STANDALONE] warning: disabling outer --enable-cudagraphs because "
            "--fx-standalone-cudagraph captures the standalone executor internally"
        )
        args.enable_cudagraphs = False

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False"
        )

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    all_payloads = {}

    for model_name in args.models:
        all_payloads[model_name] = benchmark_one_model(
            model_name,
            args,
        )

    aggregate_path = (
        Path(args.out_dir) / "benchmark_summary_all.json"
    )

    aggregate_path.write_text(
        json.dumps(all_payloads, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"\nWrote aggregate benchmark summary: {aggregate_path}")


if __name__ == "__main__":
    main()
