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

import runtime.snn_custom_ops as snn_custom_ops
from test.models_for_fx_test import reset_custom_stateful_lif_modules
from benchmarks.validate_chronos_baselines import (
    MultiStepWrapper,
    SingleStepWrapper,
    RewriteCounters,
    make_resnet_layer,
    make_rewrite_backend,
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


def prepare_runnable(name, model, compile_mode, backend, device):
    model.eval()
    reset_custom_stateful_lif_modules(model)

    if compile_mode:
        runnable = torch.compile(
            model,
            backend=backend if backend is not None else "inductor",
            fullgraph=False,
            dynamic=False,
        )
    else:
        runnable = model

    return runnable


def compile_and_warmup(runnable, model, x, device, warmup):
    # compile trigger
    reset_custom_stateful_lif_modules(model)

    synchronize_if_needed(device)
    with torch.no_grad():
        _ = runnable(x)
    synchronize_if_needed(device)

    # runtime warmup
    for _ in range(warmup):
        reset_custom_stateful_lif_modules(model)

        synchronize_if_needed(device)
        with torch.no_grad():
            _ = runnable(x)
        synchronize_if_needed(device)


def benchmark_runnable(name, runnable, model, x, device, repeat):
    times = []

    try:
        for _ in range(repeat):
            reset_custom_stateful_lif_modules(model)

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


def run_case(case_name, model, x, device, compile_mode, backend, warmup, repeat):
    print(f"[BENCH] {case_name}")

    try:
        runnable = prepare_runnable(
            case_name,
            model,
            compile_mode,
            backend,
            device,
        )

        compile_and_warmup(
            runnable,
            model,
            x,
            device,
            warmup,
        )

        return benchmark_runnable(
            case_name,
            runnable,
            model,
            x,
            device,
            repeat,
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

    torch.manual_seed(args.seed)

    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)

    base_layer = make_resnet_layer(
        model_name,
        allow_resnet32_fallback=not args.require_direct_resnet32_api,
    )

    base_layer = base_layer.to(
        device=args.device,
        dtype=torch.float32,
    ).eval()

    x = torch.randn(
        args.batch_size,
        3,
        args.height,
        args.width,
        device=args.device,
        dtype=torch.float32,
    )

    snn_custom_ops.configure_fused_op(
        backend=args.fused_op_backend,
        strict_triton=args.strict_triton,
        verbose=args.print_fused_op_calls,
    )

    out_dir = Path(args.out_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = {}

    #
    # baseline single-step
    #
    if args.include_s_cases:
        cases["baseline_s_eager"] = (
            SingleStepWrapper(copy.deepcopy(base_layer)).to(args.device).eval(),
            False,
            None,
        )

        cases["baseline_s_compile"] = (
            SingleStepWrapper(copy.deepcopy(base_layer)).to(args.device).eval(),
            True,
            None,
        )

    #
    # baseline multi-step
    #
    cases["baseline_m_eager"] = (
        MultiStepWrapper(
            copy.deepcopy(base_layer),
            args.T,
        ).to(args.device).eval(),
        False,
        None,
    )

    cases["baseline_m_compile"] = (
        MultiStepWrapper(
            copy.deepcopy(base_layer),
            args.T,
        ).to(args.device).eval(),
        True,
        None,
    )

    #
    # outer temporal autotune
    #
    if args.sweep_temporal_windows:
        candidate_windows = args.temporal_window_candidates
    else:
        candidate_windows = [args.temporal_fuse_window]

    #
    # filter invalid windows
    #
    candidate_windows = [
        w for w in candidate_windows
        if w <= args.T and args.T % w == 0
    ]

    chronos_rewrite_counters = {}

    for tw in candidate_windows:
        local_args = copy.deepcopy(args)

        local_args.temporal_fuse_window = tw

        #
        # schedule window follows temporal window
        #
        if local_args.temporal_schedule_window is None:
            local_args.temporal_schedule_window = tw

        rewrite_counters = RewriteCounters()

        chronos_backend = make_rewrite_backend(
            local_args,
            out_dir / f"chronos_m_compile_w{tw}",
            rewrite_counters,
        )

        case_name = f"chronos_m_compile_w{tw}"

        cases[case_name] = (
            MultiStepWrapper(
                copy.deepcopy(base_layer),
                local_args.T,
            ).to(local_args.device).eval(),
            True,
            chronos_backend,
        )

        chronos_rewrite_counters[case_name] = rewrite_counters

    #
    # run benchmark
    #
    results = {}
    summary_rows = []
    fused_stats_by_case = {}

    for case_name, (model, compile_mode, backend) in cases.items():
        snn_custom_ops.reset_fused_op_call_stats()

        result = run_case(
            case_name,
            model,
            x,
            args.device,
            compile_mode,
            backend,
            args.warmup,
            args.repeat,
        )
        case_fused_stats = snn_custom_ops.get_fused_op_call_stats()
        fused_stats_by_case[case_name] = case_fused_stats

        results[case_name] = asdict(result)

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

    #
    # autotune summary
    #
    print("\n[AUTOTUNE SUMMARY]")
    print(f"{'case':32s} {'mean(ms)':>12s} {'speedup':>12s}")

    baseline = None

    if "baseline_m_compile" in results:
        baseline = results["baseline_m_compile"]["mean_ms"]

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

    best_case = None

    if sorted_rows:
        best_case = sorted_rows[0]

        print(
            f"\n[BEST CONFIG] "
            f"{best_case['case']} "
            f"mean={best_case['mean_ms']:.3f} ms"
        )

    #
    # dump json
    #
    payload = {
        "model": model_name,
        "input_shape": [
            args.batch_size,
            3,
            args.height,
            args.width,
        ],
        "T": args.T,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "fused_op_backend": args.fused_op_backend,
        "candidate_windows": candidate_windows,
        "results": results,
        "chronos_rewrite_counters": {
            k: asdict(v)
            for k, v in chronos_rewrite_counters.items()
        },
        "best_case": best_case,
        "fused_op_call_stats_last_case": fused_stats,
        "fused_op_call_stats_by_case": fused_stats_by_case,
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

    #
    # model
    #
    parser.add_argument(
        "--models",
        nargs="+",
        default=["resnet18"],
        choices=["resnet18", "resnet32"],
    )

    parser.add_argument("--T", type=int, default=16)

    parser.add_argument("--batch-size", type=int, default=16)

    parser.add_argument("--height", type=int, default=224)

    parser.add_argument("--width", type=int, default=224)

    parser.add_argument("--device", default="cuda")

    #
    # benchmark
    #
    parser.add_argument("--warmup", type=int, default=10)

    parser.add_argument("--repeat", type=int, default=50)

    #
    # backend
    #
    parser.add_argument(
        "--fused-op-backend",
        choices=("torch", "triton"),
        default="triton",
    )

    parser.add_argument(
        "--rewrite-backend-mode",
        choices=("eager", "inductor"),
        default="inductor",
    )

    parser.add_argument("--strict-triton", action="store_true")

    parser.add_argument("--print-fused-op-calls", action="store_true")

    #
    # temporal rewrite
    #
    parser.add_argument("--enable-temporal-rewrite", action="store_true")

    parser.add_argument("--enable-temporal-schedule", action="store_true")

    parser.add_argument("--temporal-fuse-window", type=int, default=8)

    parser.add_argument("--temporal-schedule-window", type=int, default=None)

    parser.add_argument("--temporal-allow-tail", action="store_true")

    parser.add_argument("--temporal-schedule-dump", action="store_true")

    parser.add_argument("--temporal-schedule-strict", action="store_true")

    #
    # outer autotune
    #
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

    #
    # rewrite control
    #
    parser.add_argument("--disable-rewrite", action="store_true")

    parser.add_argument("--disable-conv-bn-lif", action="store_true")

    parser.add_argument("--max-patterns", type=int, default=1000)

    #
    # misc
    #
    parser.add_argument("--include-s-cases", action="store_true")

    parser.add_argument("--require-direct-resnet32-api", action="store_true")

    parser.add_argument("--out-dir", default="chronos_benchmark")

    parser.add_argument("--seed", type=int, default=2026)

    #
    # dummy fields required by imported backend
    #
    parser.add_argument("--rtol", type=float, default=1e-4)

    parser.add_argument("--atol", type=float, default=1e-4)

    return parser.parse_args()


def main():
    args = parse_args()

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
