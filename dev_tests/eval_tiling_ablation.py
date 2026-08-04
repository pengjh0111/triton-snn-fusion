#!/usr/bin/env python3
"""Ablate Kairos temporal window, time-pack, and weight-reuse knobs."""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MODELS = ("resnet18", "alexnet", "spiketransformer", "mamba")
# P1/R1 were merged into a single "time reuse disabled" variant (TR1): forcing
# BTILE_T=1 AND REUSE_GROUPS=1 simultaneously, then re-searching W the same
# way "full" does. Splitting P and R separately added little signal --
# whichever factor is left free, the autotuner mostly recovers the loss by
# increasing the other one under the shared register budget (paper Eq. 4) --
# so this ablation now asks the coarser, more decision-relevant question:
# "what does disabling temporal reuse entirely cost, at the schedule's own
# best window?"
VARIANTS = ("full", "TR1", "W1")


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def child_main(args):
    cache_root = Path(args.out_dir) / "cache" / args.child_model / args.child_variant / f"W{args.child_window}"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(cache_root / "triton")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_root / "inductor")
    if args.child_variant == "TR1":
        os.environ["KAIROS_EVAL_FORCE_BTILE_T"] = "1"
        os.environ["KAIROS_EVAL_FORCE_REUSE_GROUPS"] = "1"

    import torch
    import runtime.snn_custom_ops as snn_custom_ops
    import eval_three_taxes as et
    from eval_three_taxes import build_runnable, invoke, time_cuda

    if args.child_model == "spiketransformer":
        # Ablation-only, this-script-only workaround (does not touch
        # compiler/ or eval_three_taxes.py): rewrite_transformer_spatial_batching
        # (compiler/fx_spatial_batching.py) has a pre-existing bug that only
        # SpikeTransformer's attention/layer_norm rewrite path reaches -- it
        # can't determine a fused LIF op's own output shape when
        # temporal_window < T (FakeTensorProp only ran once, on the graph
        # before that fused op node existed, so its shape metadata is
        # empty), so the reshape it builds targets the wrong window size and
        # crashes at runtime for any window narrower than T. Confirmed via a
        # direct repro at W=2, T=16.
        #
        # This ablation isolates a DIFFERENT layer entirely -- the fused
        # custom op's own tiling autotune (BTILE_T / REUSE_GROUPS), read
        # back via kernel_temporal_configs -- and doesn't depend on spatial
        # batching's cross-timestep candidate batching having run at all, so
        # disabling it here for SpikeTransformer only sidesteps the crash
        # rather than fixing the (orthogonal to this ablation) framework
        # pass. resnet18/alexnet/mamba are unaffected: this only patches the
        # CLI construction for this one model, in this one child process.
        real_namespace = et._benchmark_namespace

        def _namespace_without_spatial_batching(a, model, mode, profile):
            bench_args, cli = real_namespace(a, model, mode, profile)
            bench_args.enable_spatial_batching = False
            return bench_args, cli

        et._benchmark_namespace = _namespace_without_spatial_batching

    eval_args = SimpleNamespace(
        T=args.T,
        batch=args.batch,
        dtype=args.dtype,
        seed=args.seed,
        out_dir=str(Path(args.out_dir) / "compile_ablation"),
        disable_fused_cudagraph=args.disable_cudagraph,
        temporal_window=args.child_window,
    )
    snn_custom_ops.reset_fused_op_call_stats()
    runnable, model, x, _, counters, _, cli = build_runnable(
        eval_args, args.child_model, "multi_stream", profile=False
    )
    mean_ms, std_ms = time_cuda(runnable, model, x, args.warmup, args.repeat)
    output = invoke(runnable, model, x)
    torch.cuda.synchronize()
    stats = snn_custom_ops.get_fused_op_call_stats()
    temporal_configs = stats.get("kernel_temporal_configs", {})
    pr_config_keys = [key for key in temporal_configs if "BTILE_T=" in key]
    if args.child_variant == "TR1":
        constraint_observed = bool(pr_config_keys) and all(
            "BTILE_T=1:REUSE_GROUPS=1:" in key for key in pr_config_keys
        )
    else:
        constraint_observed = bool(pr_config_keys)

    # Ablation-only, this-script-only instrumentation shim (does not touch
    # runtime/ or kernels/ on disk): SpikeTransformer's fused batched-linear-
    # LIF custom op (kernels/benchmark_batched_linear_lif_temporal_general.py,
    # the default "codegen" backend behind snn_custom.fused_temporal_batched_
    # linear_lif_state) DOES read and enforce KAIROS_EVAL_FORCE_BTILE_T /
    # KAIROS_EVAL_FORCE_REUSE_GROUPS in its autotune config pruning
    # (_prune_batched_linear_configs raises RuntimeError if no candidate
    # config satisfies a forced constraint) -- but its runtime/snn_custom_ops.py
    # _impl wrapper never calls _record_kernel_temporal_config, so
    # kernel_temporal_configs stays empty for this model and the check above
    # always reads False here, regardless of whether the constraint actually
    # applied. (A first attempt at fixing this wrapped
    # _prune_batched_linear_configs's module attribute in-process, but that
    # doesn't work: triton.autotune(..., prune_configs_by={"early_config_prune":
    # _prune_batched_linear_configs}) captures the function object once, at
    # module-import time, so rebinding the module attribute afterward never
    # reaches the already-constructed Autotoner.) Instead, read back the
    # Autotuner's own actual choice via get_autotune_best_config() -- an
    # existing, already-used-elsewhere helper in that same kernel module --
    # after this run completes.
    tr1_shim_observed = None
    if args.child_model == "spiketransformer" and args.child_variant == "TR1":
        if snn_custom_ops._batched_linear_lif_backend() == "tc":
            # The "tc" fallback backend's own config pruning
            # (kernels/generated_temporal_transformer_lif_kernels.py) has no
            # REUSE_GROUPS concept in its default path, so a combined P=1+R=1
            # constraint can't be verified the same way there. Not expected
            # to be active (default backend is "codegen"); leave unobserved
            # rather than guess.
            tr1_shim_observed = None
        else:
            import kernels.benchmark_batched_linear_lif_temporal_general as kblt

            best_config = kblt.get_autotune_best_config()
            forced_p = int(os.environ["KAIROS_EVAL_FORCE_BTILE_T"])
            forced_r = int(os.environ["KAIROS_EVAL_FORCE_REUSE_GROUPS"])
            tr1_shim_observed = bool(
                best_config is not None
                and best_config.get("BTILE_T") is not None
                and best_config.get("REUSE_GROUPS") is not None
                and int(best_config["BTILE_T"]) == forced_p
                and int(best_config["REUSE_GROUPS"]) == forced_r
            )
        if tr1_shim_observed:
            constraint_observed = True
    output_path = Path(args.child_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output.detach().float().cpu(), output_path)
    payload = {
        "model": args.child_model,
        "variant": args.child_variant,
        "W": args.child_window,
        "time_ms_mean": mean_ms,
        "time_ms_std": std_ms,
        "kernel_temporal_configs": temporal_configs,
        "constraint_observed": constraint_observed,
        "constraint_observed_via_shim": tr1_shim_observed,
        "benchmark_cli": cli,
        "rewrite_counters": vars(counters),
        "output_path": str(output_path),
    }
    Path(args.child_result).write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def run_candidate(args, model, variant, window, out_dir):
    case_dir = out_dir / "cases" / model / variant / f"W{window}"
    result_path = case_dir / "result.json"
    output_path = case_dir / "output.pt"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--child-model",
        model,
        "--child-variant",
        variant,
        "--child-window",
        str(window),
        "--child-result",
        str(result_path),
        "--child-output",
        str(output_path),
        "--T",
        str(args.T),
        "--batch",
        str(args.batch),
        "--dtype",
        args.dtype,
        "--warmup",
        str(args.warmup),
        "--repeat",
        str(args.repeat),
        "--seed",
        str(args.seed),
        "--out-dir",
        str(out_dir),
    ]
    if args.disable_cudagraph:
        cmd.append("--disable-cudagraph")
    completed = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=args.timeout_sec,
        check=False,
    )
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "run.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"ablation child failed for {model}/{variant}/W{window}:\n"
            f"{completed.stdout[-5000:]}"
        )
    return json.loads(result_path.read_text(encoding="utf-8"))


def main(args):
    if args.child:
        child_main(args)
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    out_dir = Path(args.out_dir)
    candidates = []
    summaries = []
    metadata = {
        "T": args.T,
        "batch": args.batch,
        "dtype": args.dtype,
        "models": args.models,
        "searched_windows": args.windows,
        "variants": {
            "full": "Search W with unconstrained P/R autotune",
            "TR1": "Search W with time reuse disabled (KAIROS_EVAL_FORCE_BTILE_T=1 AND KAIROS_EVAL_FORCE_REUSE_GROUPS=1)",
            "W1": "Fix graph/region W=1 with unconstrained P/R",
        },
        "default_path_impact": "None; P/R constraints are inactive unless the evaluation-only environment variables are explicitly set.",
        "cases": {},
    }
    run_variants = args.variants
    for model in args.models:
        print(f"[model] {model}")
        reference = None
        for variant in run_variants:
            windows = [1] if variant == "W1" else args.windows
            for window in windows:
                print(f"  [candidate] {variant} W={window}")
                payload = run_candidate(args, model, variant, window, out_dir)
                output = torch.load(payload["output_path"], map_location="cpu", weights_only=True)
                if reference is None:
                    reference = output
                max_abs = float((output - reference).abs().max()) if output.numel() else 0.0
                ok = bool(torch.allclose(output, reference, rtol=args.rtol, atol=args.atol))
                if not ok:
                    raise AssertionError(
                        f"correctness failed for {model}/{variant}/W{window}: max_abs={max_abs}"
                    )
                row = {
                    "model": model,
                    "variant": variant,
                    "W": window,
                    "time_ms_mean": payload["time_ms_mean"],
                    "time_ms_std": payload["time_ms_std"],
                    "correctness_ok": ok,
                    "max_abs": max_abs,
                    "constraint_observed": payload["constraint_observed"],
                    "kernel_temporal_configs": json.dumps(payload["kernel_temporal_configs"], sort_keys=True),
                }
                candidates.append(row)
                metadata["cases"][f"{model}/{variant}/W{window}"] = payload
                write_csv(
                    out_dir / "eval_tiling_ablation_candidates.csv",
                    candidates,
                    tuple(row.keys()),
                )
        model_rows = [row for row in candidates if row["model"] == model]
        best = {
            variant: min(
                (row for row in model_rows if row["variant"] == variant),
                key=lambda row: row["time_ms_mean"],
            )
            for variant in run_variants
        }
        # "full" is only used as an in-script ratio baseline when it was
        # actually part of this run. When --variants only asks for TR1 (the
        # common case: "full" already has an externally supplied reference
        # time, don't re-measure it), slowdown_vs_full is left as None here
        # -- it gets computed downstream against the externally supplied
        # Full number instead, same as tiling_data.py already does.
        baseline = best["full"]["time_ms_mean"] if "full" in best else None
        for variant in run_variants:
            summaries.append(
                {
                    **best[variant],
                    "slowdown_vs_full": (
                        best[variant]["time_ms_mean"] / baseline if baseline is not None else None
                    ),
                    "evidence_type": (
                        "controlled"
                        if variant != "TR1" or best[variant]["constraint_observed"]
                        else "observational_only"
                    ),
                }
            )
        write_csv(
            out_dir / "eval_tiling_ablation.csv",
            summaries,
            tuple(summaries[0].keys()),
        )
        (out_dir / "ablation_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--windows", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--T", type=int, default=16)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp32")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--disable-cudagraph", action="store_true")
    parser.add_argument("--timeout-sec", type=int, default=7200)
    parser.add_argument("--out-dir", default="test/eval_tiling_ablation")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--child-model", choices=MODELS, help=argparse.SUPPRESS)
    parser.add_argument("--child-variant", choices=VARIANTS, help=argparse.SUPPRESS)
    parser.add_argument("--child-window", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--child-result", help=argparse.SUPPRESS)
    parser.add_argument("--child-output", help=argparse.SUPPRESS)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
