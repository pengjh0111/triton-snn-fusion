# benchmarks/benchmark_tvm_metaschedule_runtime.py

import argparse
import inspect
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CHRONOS_MODEL_CHOICES = [
    "resnet18",
    "resnet34",
    "resnet32",
    "alexnet",
    "zfnet",
    "vgg11",
    "vgg16",
    "mobilenetv1",
    "mobilenetv2",
]
LIF_IMPL_CHOICES = ["chronos", "spikingjelly"]


def resolve_np_dtype(precision: str):
    import numpy as np

    if precision == "fp16":
        return np.float16
    if precision in ("fp32", "tf32"):
        return np.float32
    raise ValueError(precision)


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    idx = int(round((len(values) - 1) * q))
    return values[idx]


def summarize_ms(values):
    values = [float(v) for v in values]
    if not values:
        return {}
    return {
        "mean": float(sum(values) / len(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "p50": float(percentile(values, 0.50)),
        "p90": float(percentile(values, 0.90)),
    }


def import_tvm_deps():
    import numpy as np
    import onnx
    import tvm
    from tvm import relay
    from tvm.contrib import graph_executor
    import tvm.meta_schedule as ms

    return np, onnx, tvm, relay, graph_executor, ms


def import_export_onnx():
    from benchmarks.benchmark_tensorrt_runtime import export_onnx

    return export_onnx


def call_with_supported_kwargs(fn, **kwargs):
    sig = inspect.signature(fn)
    accepted = {}
    has_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )
    for key, value in kwargs.items():
        if has_var_kwargs or key in sig.parameters:
            accepted[key] = value
    return fn(**accepted)


def get_relay_integration(ms):
    relay_integration = getattr(ms, "relay_integration", None)
    tune_relay = getattr(relay_integration, "tune_relay", None) if relay_integration else None
    compile_relay = getattr(relay_integration, "compile_relay", None) if relay_integration else None

    if tune_relay is None:
        tune_relay = getattr(ms, "tune_relay", None)
    if compile_relay is None:
        compile_relay = getattr(ms, "compile_relay", None)

    if tune_relay is None or compile_relay is None:
        raise RuntimeError(
            "This TVM build does not expose meta_schedule tune_relay/compile_relay APIs."
        )
    return tune_relay, compile_relay


def tune_and_build_with_metaschedule(
    mod,
    params,
    target,
    work_dir: Path,
    max_trials_global: int,
    num_trials_per_iter: int,
    runner: str,
    builder: str,
    ms,
):
    tune_relay, compile_relay = get_relay_integration(ms)
    work_dir.mkdir(parents=True, exist_ok=True)

    database = call_with_supported_kwargs(
        tune_relay,
        mod=mod,
        target=target,
        params=params,
        work_dir=str(work_dir),
        max_trials_global=max_trials_global,
        num_trials_per_iter=num_trials_per_iter,
        runner=runner,
        builder=builder,
    )

    lib = call_with_supported_kwargs(
        compile_relay,
        database=database,
        mod=mod,
        target=target,
        params=params,
    )
    return database, lib


def build_without_tuning(mod, params, target, relay, tvm):
    with tvm.transform.PassContext(opt_level=3):
        return relay.build(mod, target=target, params=params)


def load_onnx_as_relay(onnx_path: Path, input_shape, relay, onnx):
    onnx_model = onnx.load(onnx_path.as_posix())
    shape_dict = {"input": input_shape}
    mod, params = relay.frontend.from_onnx(
        onnx_model,
        shape=shape_dict,
        freeze_params=True,
    )
    return mod, params


def run_tvm_graph_executor(lib, input_data, repeat: int, number: int, dev, graph_executor):
    module = graph_executor.GraphModule(lib["default"](dev))
    module.set_input("input", input_data)
    module.run()

    timer = module.module.time_evaluator(
        "run",
        dev,
        number=number,
        repeat=repeat,
    )
    prof = timer()
    per_run_ms = [float(t) * 1000.0 / float(number) for t in prof.results]
    return {
        "repeat": repeat,
        "number": number,
        "latency_ms": summarize_ms(per_run_ms),
        "raw_latency_ms": per_run_ms,
    }


def benchmark_onnx_with_tvm(
    onnx_path: Path,
    precision: str,
    batch_size: int,
    height: int,
    width: int,
    target_text: str,
    dev_id: int,
    max_trials_global: int,
    num_trials_per_iter: int,
    runner: str,
    builder: str,
    repeat: int,
    number: int,
    out_dir: Path,
):
    result: Dict[str, Any] = {
        "ok": False,
        "tvm_import_ok": False,
        "tvm_tune_ok": False,
        "tvm_compile_ok": False,
        "tvm_benchmark_ok": False,
        "onnx_path": str(onnx_path),
        "precision": precision,
        "target": target_text,
        "max_trials_global": max_trials_global,
        "num_trials_per_iter": num_trials_per_iter,
        "runner": runner,
        "builder": builder,
        "parsed": {},
        "error": "",
    }

    try:
        np, onnx, tvm, relay, graph_executor, ms = import_tvm_deps()
        target = tvm.target.Target(target_text)
        dev = tvm.cuda(dev_id)
        if not dev.exist:
            raise RuntimeError(f"TVM CUDA device {dev_id} is not available")

        input_shape = (batch_size, 3, height, width)
        mod, params = load_onnx_as_relay(onnx_path, input_shape, relay, onnx)
        result["tvm_import_ok"] = True

        tune_dir = out_dir / "ms_work_dir"
        if max_trials_global > 0:
            _, lib = tune_and_build_with_metaschedule(
                mod=mod,
                params=params,
                target=target,
                work_dir=tune_dir,
                max_trials_global=max_trials_global,
                num_trials_per_iter=num_trials_per_iter,
                runner=runner,
                builder=builder,
                ms=ms,
            )
            result["tvm_tune_ok"] = True
        else:
            lib = build_without_tuning(mod, params, target, relay, tvm)
            result["tvm_tune_ok"] = False
            result["note"] = "max_trials_global <= 0, built with Relay opt_level=3 without MetaSchedule tuning"

        result["tvm_compile_ok"] = True

        lib_path = out_dir / f"{onnx_path.stem}_tvm.so"
        try:
            lib.export_library(lib_path.as_posix())
            result["lib_path"] = str(lib_path)
        except Exception as exc:
            result["lib_export_error"] = str(exc)

        dtype = resolve_np_dtype(precision)
        rng = np.random.default_rng(0)
        input_data = rng.standard_normal(input_shape).astype(dtype)
        bench = run_tvm_graph_executor(
            lib=lib,
            input_data=input_data,
            repeat=repeat,
            number=number,
            dev=dev,
            graph_executor=graph_executor,
        )
        result["parsed"] = bench
        result["tvm_benchmark_ok"] = True
        result["ok"] = True
        return result

    except Exception:
        result["error"] = traceback.format_exc()
        print("[TVM METASCHEDULE FAILED]")
        print(result["error"])
        return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--models", nargs="+", default=["resnet18"], choices=CHRONOS_MODEL_CHOICES)
    parser.add_argument(
        "--execution-modes",
        nargs="+",
        default=["single_step_mode"],
        choices=["single_step_mode"],
    )
    parser.add_argument(
        "--precisions",
        nargs="+",
        default=["fp32", "tf32", "fp16"],
        choices=["fp32", "tf32", "fp16"],
    )
    parser.add_argument("--T", type=int, default=16)
    parser.add_argument("--model-channels", type=int, default=64)
    parser.add_argument("--lif-impl", choices=LIF_IMPL_CHOICES, default="chronos")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--out-dir", default="test/tvm_metaschedule_validation")

    parser.add_argument("--target", default="cuda")
    parser.add_argument("--dev-id", type=int, default=0)
    parser.add_argument("--max-trials-global", type=int, default=1024)
    parser.add_argument("--num-trials-per-iter", type=int, default=64)
    parser.add_argument("--runner", default="local")
    parser.add_argument("--builder", default="local")
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--number", type=int, default=10)
    parser.add_argument(
        "--onnx-path",
        default="",
        help="Optional pre-exported single-step ONNX. Only valid when one model/mode/precision is selected.",
    )

    args = parser.parse_args()
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for model_name in args.models:
        all_results[model_name] = {}
        for execution_mode in args.execution_modes:
            all_results[model_name][execution_mode] = {}
            for precision in args.precisions:
                print("=" * 80)
                print(
                    f"[RUN] model={model_name} mode={execution_mode} "
                    f"precision={precision} target={args.target}"
                )
                print("=" * 80)

                run_dir = out_root / model_name / execution_mode / precision
                run_dir.mkdir(parents=True, exist_ok=True)

                if args.onnx_path:
                    onnx_path = Path(args.onnx_path)
                    export_result = {
                        "ok": onnx_path.exists(),
                        "onnx_export_ok": onnx_path.exists(),
                        "onnx_path": str(onnx_path),
                        "precision": precision,
                        "model_channels": args.model_channels,
                        "lif_impl": args.lif_impl,
                        "custom_lif": args.lif_impl == "chronos",
                        "wrapper": "SingleStepModeLoopWrapper",
                        "error": "" if onnx_path.exists() else f"ONNX path does not exist: {onnx_path}",
                    }
                else:
                    export_onnx = import_export_onnx()
                    export_result = export_onnx(
                        model_name=model_name,
                        execution_mode=execution_mode,
                        precision=precision,
                        model_channels=args.model_channels,
                        lif_impl=args.lif_impl,
                        T=args.T,
                        batch_size=args.batch_size,
                        height=args.height,
                        width=args.width,
                        opset=args.opset,
                        out_dir=run_dir,
                    )

                if not export_result["onnx_export_ok"]:
                    result = {
                        **export_result,
                        "tvm_import_ok": False,
                        "tvm_tune_ok": False,
                        "tvm_compile_ok": False,
                        "tvm_benchmark_ok": False,
                        "parsed": {},
                    }
                else:
                    tvm_result = benchmark_onnx_with_tvm(
                        onnx_path=Path(export_result["onnx_path"]),
                        precision=precision,
                        batch_size=args.batch_size,
                        height=args.height,
                        width=args.width,
                        target_text=args.target,
                        dev_id=args.dev_id,
                        max_trials_global=args.max_trials_global,
                        num_trials_per_iter=args.num_trials_per_iter,
                        runner=args.runner,
                        builder=args.builder,
                        repeat=args.repeat,
                        number=args.number,
                        out_dir=run_dir,
                    )
                    result = {
                        **export_result,
                        **tvm_result,
                        "onnx_export_ok": export_result["onnx_export_ok"],
                        "custom_lif": export_result.get("custom_lif"),
                        "model_channels": export_result.get("model_channels"),
                        "lif_impl": export_result.get("lif_impl"),
                        "wrapper": export_result.get("wrapper"),
                        "ok": export_result["onnx_export_ok"] and tvm_result["ok"],
                    }

                all_results[model_name][execution_mode][precision] = result
                summary_path = run_dir / "summary.json"
                summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                print(f"[WRITE] {summary_path}")

    aggregate_path = out_root / "tvm_metaschedule_summary_all.json"
    aggregate_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    print("=" * 80)
    print("[DONE]")
    print(f"[WRITE] {aggregate_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
