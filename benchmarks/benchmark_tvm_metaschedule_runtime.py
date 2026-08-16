# benchmarks/benchmark_tvm_metaschedule_runtime.py

import argparse
import contextlib
import inspect
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

KAIROS_MODEL_CHOICES = [
    "resnet18",
    "resnet34",
    "resnet32",
    "alexnet",
    "zfnet",
    "vgg11",
    "vgg16",
    "mobilenetv1",
    "mobilenetv2",
    "spiketransformer",
    "spikebert",
    "convlstm",
    "mamba",
    "deepspeech2",
    "nafnet",
    "bsrn",
]
LIF_IMPL_CHOICES = ["kairos", "spikingjelly"]


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


def build_relay_pass_config(fuse_max_depth: Optional[int]) -> Dict[str, Any]:
    """Base pass_config used by extract_tasks/compile_relay, with an optional
    override for relay.FuseOps.max_depth (TVM's C++ default is 256, which lets
    long chains of injective ops -- e.g. the dense+normalize+LIF-surrogate
    chains in these spiking models -- get fused into a single sketch-generation
    target whose AutoInline candidate search can take upwards of many hours to
    even initialize (see e.g. task "fused_nn_dense_subtract_divide_add_
    greater_equal_cast_cast_greater_where_subtract..."); capping it keeps
    fused subgraphs small enough to tune in reasonable time).
    """
    pass_config: Dict[str, Any] = {
        "relay.backend.use_meta_schedule": True,
        "relay.backend.tir_converter": "default",
    }
    if fuse_max_depth is not None and fuse_max_depth > 0:
        pass_config["relay.FuseOps.max_depth"] = fuse_max_depth
    return pass_config


def build_fallback_pass_config(fuse_max_depth: Optional[int]) -> Dict[str, Any]:
    """pass_config for build_without_tuning, which deliberately skips
    MetaSchedule and has no meta_schedule.Database in scope. It must NOT
    include relay.backend.use_meta_schedule=True (unlike
    build_relay_pass_config) -- doing so makes the TE compiler look for a
    database context that was never entered and hard-fail with
    'use_meta_schedule is enabled ... but no meta_schedule.Database context is
    provided'. Only relay.FuseOps.max_depth is relevant here.
    """
    pass_config: Dict[str, Any] = {}
    if fuse_max_depth is not None and fuse_max_depth > 0:
        pass_config["relay.FuseOps.max_depth"] = fuse_max_depth
    return pass_config


@contextlib.contextmanager
def patched_extract_tasks(ms, pass_config: Dict[str, Any]):
    """tune_relay() and find_untuned_tasks() both call extract_tasks() without
    forwarding a pass_config, so extract_tasks always falls back to its own
    hardcoded default (no relay.FuseOps.max_depth override). Neither tune_relay
    nor find_untuned_tasks accepts pass_config either, so the only way to make
    task extraction respect our fuse-depth cap is to substitute extract_tasks
    for the duration of the call. compile_relay() does accept pass_config
    directly, so it isn't patched here -- callers should pass pass_config to it
    explicitly, using the same dict, so tuning-time and compile-time fusion
    boundaries match.
    """
    relay_integration = ms.relay_integration
    original_extract_tasks = relay_integration.extract_tasks

    def extract_tasks_with_pass_config(mod, target, params, **kwargs):
        kwargs.setdefault("pass_config", pass_config)
        return original_extract_tasks(mod, target, params, **kwargs)

    relay_integration.extract_tasks = extract_tasks_with_pass_config
    try:
        yield
    finally:
        relay_integration.extract_tasks = original_extract_tasks


def build_recycling_local_runner(ms, timeout_sec=30.0, maximum_process_uses=16):
    """A LocalRunner whose single worker process gets recycled periodically.

    ms.runner.LocalRunner hardcodes a single PopenPoolExecutor worker that is
    only recycled when a candidate times out. A candidate that crashes with a
    CUDA-level error (e.g. "CUDA: misaligned address") leaves that worker
    process alive at the OS level, but its CUDA context is left permanently
    unusable -- every later measurement submitted to the same process then
    silently fails too, for the rest of the tuning run, even though those
    later candidates' schedules were never themselves at fault. Recycling the
    process every `maximum_process_uses` trials bounds how many trials a
    single bad candidate can take down with it.
    """
    import subprocess

    from tvm.contrib.popen_pool import PopenPoolExecutor

    runner = ms.runner.LocalRunner(timeout_sec=timeout_sec)
    runner.pool = PopenPoolExecutor(
        max_workers=1,
        timeout=timeout_sec,
        initializer=None,
        maximum_process_uses=maximum_process_uses,
        stderr=subprocess.DEVNULL,
    )
    return runner


def find_untuned_tasks(mod, target, params, database, ms):
    """Return task names for which MetaSchedule never recorded a measured schedule.

    tune_relay always registers a database workload entry for every extracted
    task, but only tasks with at least one successfully measured candidate get
    a tuning record. Tasks with zero records make compile_relay fall back to
    an unscheduled (unbound) PrimFunc, which fails VerifyMemory on a CUDA
    target instead of producing executable code.
    """
    extract_tasks = getattr(ms.relay_integration, "extract_tasks", None)
    if extract_tasks is None:
        return []

    untuned = []
    for task in extract_tasks(mod, target, params):
        task_mod = task.dispatched[0]
        if not database.has_workload(task_mod):
            untuned.append(task.task_name)
            continue
        workload = database.commit_workload(task_mod)
        if not database.get_top_k(workload, 1):
            untuned.append(task.task_name)
    return untuned


def tune_and_build_with_metaschedule(
    mod,
    params,
    target,
    work_dir: Path,
    max_trials_global: int,
    max_trials_per_task: int,
    num_trials_per_iter: int,
    runner: str,
    builder: str,
    builder_timeout_sec: float,
    ms,
    relay,
    tvm,
    runner_timeout_sec: float = 30.0,
    runner_max_process_uses: int = 16,
    untuned_retry_trials: int = 512,
    fuse_max_depth: Optional[int] = None,
):
    tune_relay, compile_relay = get_relay_integration(ms)
    work_dir.mkdir(parents=True, exist_ok=True)
    builder_config = (
        ms.builder.LocalBuilder(timeout_sec=builder_timeout_sec)
        if builder == "local"
        else builder
    )
    pass_config = build_relay_pass_config(fuse_max_depth)

    def make_runner_config():
        if runner == "local":
            return build_recycling_local_runner(
                ms, timeout_sec=runner_timeout_sec, maximum_process_uses=runner_max_process_uses
            )
        return runner

    with patched_extract_tasks(ms, pass_config):
        database = call_with_supported_kwargs(
            tune_relay,
            mod=mod,
            target=target,
            params=params,
            work_dir=str(work_dir),
            max_trials_global=max_trials_global,
            max_trials_per_task=max_trials_per_task,
            num_trials_per_iter=num_trials_per_iter,
            runner=make_runner_config(),
            builder=builder_config,
        )

        untuned_tasks = find_untuned_tasks(mod, target, params, database, ms)
        if untuned_tasks and untuned_retry_trials > 0:
            # A poisoned CUDA context (see build_recycling_local_runner) is a
            # likely cause of a task ending up with zero measured trials even
            # though its schedule itself is fine. work_dir/database already
            # persisted every trial that DID succeed, so re-entering tune_relay
            # on the same work_dir with a brand-new runner process gives the
            # still-untuned tasks a fresh, unpoisoned shot before we give up on
            # them and discard the rest of the (successfully tuned) database.
            print(
                f"[TVM METASCHEDULE] {len(untuned_tasks)} task(s) still lack a measured "
                f"schedule; retrying with a fresh runner process and a "
                f"{untuned_retry_trials}-trial top-up budget:"
            )
            for name in untuned_tasks:
                print(f"  - {name}")
            database = call_with_supported_kwargs(
                tune_relay,
                mod=mod,
                target=target,
                params=params,
                work_dir=str(work_dir),
                max_trials_global=untuned_retry_trials,
                max_trials_per_task=max_trials_per_task,
                num_trials_per_iter=num_trials_per_iter,
                runner=make_runner_config(),
                builder=builder_config,
            )
            untuned_tasks = find_untuned_tasks(mod, target, params, database, ms)

        if untuned_tasks:
            # Don't throw away the tuned schedules for every other task just
            # because a handful of tasks never got a measured record.
            # compile_relay()'s TE compiler already does a per-PrimFunc
            # database lookup (see te_compiler_cache.cc): a miss logs a
            # warning and falls back to TVM's default (untuned) schedule for
            # *that one task only*, leaving every tuned task's kernel intact.
            print(
                f"[TVM METASCHEDULE] {len(untuned_tasks)} task(s) still have no measured "
                "schedule in the tuning database after retry; compile_relay() will fall "
                "back to TVM's default (untuned) schedule for just these kernels while "
                "every other task keeps its MetaSchedule-tuned kernel:"
            )
            for name in untuned_tasks:
                print(f"  - {name}")

        try:
            lib = call_with_supported_kwargs(
                compile_relay,
                database=database,
                mod=mod,
                target=target,
                params=params,
                pass_config=pass_config,
            )
        except Exception:
            if not untuned_tasks:
                raise
            # Some untuned tasks can leave compile_relay() with an
            # unscheduled/unbound PrimFunc that fails a codegen-time
            # verification pass (e.g. VerifyMemory) instead of degrading
            # gracefully -- see find_untuned_tasks()'s docstring. Only in
            # that case do we fall all the way back to a plain, non-tuned
            # build of the whole module so we still produce something
            # executable.
            print(
                "[TVM METASCHEDULE] Per-task fallback via compile_relay() could not "
                "produce a valid build (likely an unscheduled PrimFunc for one of the "
                "untuned tasks above); falling back to a non-MetaSchedule Relay build "
                "(opt_level=3) for the entire module as a last resort:"
            )
            traceback.print_exc()
            lib = build_without_tuning(
                mod, params, target, relay, tvm, build_fallback_pass_config(fuse_max_depth)
            )
    return database, lib, untuned_tasks


def build_without_tuning(mod, params, target, relay, tvm, pass_config: Optional[Dict[str, Any]] = None):
    with tvm.transform.PassContext(opt_level=3, config=pass_config or {}):
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
    per_run_ms = [float(t) * 1000.0 for t in prof.results]

    # Scheduling overhead: CPU-side dispatch time for module.run() before the
    # device sync, measured separately from time_evaluator's own device-synced
    # timing above.
    schedule_samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        module.run()
        t1 = time.perf_counter()
        dev.sync()
        schedule_samples.append((t1 - t0) * 1000.0)

    return {
        "repeat": repeat,
        "number": number,
        "latency_ms": summarize_ms(per_run_ms),
        "raw_latency_ms": per_run_ms,
        "schedule_ms": summarize_ms(schedule_samples),
    }


def benchmark_onnx_with_tvm(
    onnx_path: Path,
    model_name: str,
    precision: str,
    batch_size: int,
    height: int,
    width: int,
    sequence_length: int,
    transformer_input_dim: int,
    transformer_vocab_size: int,
    target_text: str,
    dev_id: int,
    max_trials_global: int,
    max_trials_per_task: int,
    num_trials_per_iter: int,
    runner: str,
    builder: str,
    builder_timeout_sec: float,
    T: int,
    repeat: int,
    number: int,
    out_dir: Path,
    fuse_max_depth: Optional[int] = None,
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
        "max_trials_per_task": max_trials_per_task,
        "num_trials_per_iter": num_trials_per_iter,
        "runner": runner,
        "builder": builder,
        "builder_timeout_sec": builder_timeout_sec,
        "fuse_max_depth": fuse_max_depth,
        "parsed": {},
        "error": "",
    }

    try:
        np, onnx, tvm, relay, graph_executor, ms = import_tvm_deps()
        dev = tvm.cuda(dev_id)
        if not dev.exist:
            raise RuntimeError(f"TVM CUDA device {dev_id} is not available")
        target = (
            tvm.target.Target.from_device(dev)
            if target_text.strip() == "cuda"
            else tvm.target.Target(target_text)
        )

        if model_name == "spiketransformer":
            input_shape = (batch_size, sequence_length, transformer_input_dim)
        elif model_name == "spikebert":
            input_shape = (batch_size, sequence_length)
        elif model_name == "convlstm":
            # matches export_onnx's own hardcoded convlstm input (Phase A
            # scope: not threaded through per-model CLI flags yet).
            input_shape = (T, batch_size, 1, 64, 64)
        elif model_name == "mamba":
            input_shape = (T, batch_size, 768)
        elif model_name == "deepspeech2":
            input_shape = (batch_size, 1, 161, 2 * T)
        else:
            input_shape = (batch_size, 3, height, width)
        mod, params = load_onnx_as_relay(onnx_path, input_shape, relay, onnx)
        result["tvm_import_ok"] = True

        tune_dir = out_dir / "ms_work_dir"
        if max_trials_global > 0:
            _, lib, untuned_tasks = tune_and_build_with_metaschedule(
                mod=mod,
                params=params,
                target=target,
                work_dir=tune_dir,
                max_trials_global=max_trials_global,
                max_trials_per_task=max_trials_per_task,
                num_trials_per_iter=num_trials_per_iter,
                runner=runner,
                builder=builder,
                builder_timeout_sec=builder_timeout_sec,
                ms=ms,
                relay=relay,
                tvm=tvm,
                fuse_max_depth=fuse_max_depth,
            )
            result["tvm_tune_ok"] = True
            result["tvm_tune_untuned_tasks"] = untuned_tasks
            if untuned_tasks:
                result["tvm_compile_fallback"] = "opt_level=3 (no MetaSchedule schedule for all tasks)"
        else:
            lib = build_without_tuning(
                mod, params, target, relay, tvm, build_fallback_pass_config(fuse_max_depth)
            )
            result["tvm_tune_ok"] = False
            result["note"] = "max_trials_global <= 0, built with Relay opt_level=3 without MetaSchedule tuning"

        result["tvm_compile_ok"] = True

        lib_path = out_dir / f"{onnx_path.stem}_tvm.so"
        try:
            lib.export_library(lib_path.as_posix())
            result["lib_path"] = str(lib_path)
        except Exception as exc:
            result["lib_export_error"] = str(exc)

        rng = np.random.default_rng(0)
        if model_name == "spikebert":
            input_data = rng.integers(
                0, transformer_vocab_size, size=input_shape, dtype=np.int64
            )
        else:
            dtype = resolve_np_dtype(precision)
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

    parser.add_argument("--models", nargs="+", default=["resnet18"], choices=KAIROS_MODEL_CHOICES)
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
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--transformer-depth", type=int, default=8)
    parser.add_argument("--transformer-dim", type=int, default=256)
    parser.add_argument("--transformer-heads", type=int, default=8)
    parser.add_argument("--transformer-input-dim", type=int, default=768)
    parser.add_argument("--transformer-vocab-size", type=int, default=30522)
    parser.add_argument("--transformer-num-classes", type=int, default=100)
    parser.add_argument("--lif-impl", choices=LIF_IMPL_CHOICES, default="kairos")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--out-dir", default="test/tvm_metaschedule_validation")

    parser.add_argument("--target", default="cuda")
    parser.add_argument("--dev-id", type=int, default=0)
    parser.add_argument("--max-trials-global", type=int, default=16384)
    parser.add_argument("--max-trials-per-task", type=int, default=512)
    parser.add_argument("--num-trials-per-iter", type=int, default=64)
    parser.add_argument("--runner", default="local")
    parser.add_argument("--builder", default="local")
    parser.add_argument("--builder-timeout-sec", type=float, default=300.0)
    parser.add_argument(
        "--fuse-max-depth",
        type=int,
        default=10,
        help="Override for relay.FuseOps.max_depth (TVM's C++ default is 256). "
        "Caps how many ops FuseOps can chain into a single fused subgraph before "
        "MetaSchedule tunes it; long injective chains (e.g. dense+normalize+LIF "
        "surrogate chains) left uncapped can make AutoInline's design-space "
        "generation take many hours to even initialize for a single task. "
        "Use <=0 to keep TVM's built-in default.",
    )
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
                        "custom_lif": args.lif_impl == "kairos",
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
                        sequence_length=args.sequence_length,
                        transformer_depth=args.transformer_depth,
                        transformer_dim=args.transformer_dim,
                        transformer_heads=args.transformer_heads,
                        transformer_input_dim=args.transformer_input_dim,
                        transformer_vocab_size=args.transformer_vocab_size,
                        transformer_num_classes=args.transformer_num_classes,
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
                        model_name=model_name,
                        precision=precision,
                        batch_size=args.batch_size,
                        height=args.height,
                        width=args.width,
                        sequence_length=args.sequence_length,
                        transformer_input_dim=args.transformer_input_dim,
                        transformer_vocab_size=args.transformer_vocab_size,
                        target_text=args.target,
                        dev_id=args.dev_id,
                        max_trials_global=args.max_trials_global,
                        max_trials_per_task=args.max_trials_per_task,
                        num_trials_per_iter=args.num_trials_per_iter,
                        runner=args.runner,
                        builder=args.builder,
                        builder_timeout_sec=args.builder_timeout_sec,
                        T=args.T,
                        repeat=args.repeat,
                        number=args.number,
                        out_dir=run_dir,
                        fuse_max_depth=args.fuse_max_depth,
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
