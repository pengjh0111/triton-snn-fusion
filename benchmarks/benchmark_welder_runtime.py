"""Compile and benchmark the 13 Kairos workloads with Welder (OSDI'23).

Model construction and ONNX export follow benchmark_bladedisc_runtime.py's
build_case()/export_onnx() exactly (same single-step/sequence wrapper
dispatch as the TVM and TensorRT baselines) -- reused directly rather than
re-derived, since it already returns an export-ready (LIF-decomposed) model.

Unlike TVM (in-process Relay API) and TensorRT (trtexec CLI), Welder is
driven as a sequence of subprocess calls against the `nnfusion` binary, a
separate `run_compiler` module invocation, and a generated CMake project --
mirroring nnfusion/artifacts/tune_welder.py and nnfusion/testing/run_welder.py
from the welder branch of https://github.com/microsoft/nnfusion. The
`run_compiler` step must run under a dedicated Python environment that has
Welder's own (patched) TVM fork on PYTHONPATH; the rest of this driver runs
under the normal Chronos interpreter, same as the other baselines.
"""

import argparse
import copy
import ctypes
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.benchmark_tensorrt_runtime import (  # noqa: E402
    onnx_graph_contains_custom_lif,
    resolve_dtype,
)
from benchmarks.benchmark_bladedisc_runtime import (  # noqa: E402
    build_case,
    export_onnx,
)
from benchmarks.validate_kairos_baselines import (  # noqa: E402
    KAIROS_MODEL_CHOICES,
    LIF_IMPL_CHOICES,
    reset_lif_modules,
)


################################################################################
# defaults for the RTX 5090 welder build (see docs/welder_rtx5090_setup, or
# the incremental-build notes in this PR, for how these were produced)
################################################################################

DEFAULT_WELDER_REPO = "/data/nnfusion"
DEFAULT_WELDER_TVM_PYTHONPATH = "/data/welder-deps/tvm-welder/python"
DEFAULT_NNFUSION_BIN_DIR = "/data/welder-deps/nnfusion-welder-cpp/build/src/tools/nnfusion"
DEFAULT_WELDER_PYTHON = "/data/welder-deps/envs/welder-5090-venv/bin/python"
DEFAULT_CUDA_HOME = "/usr/local/cuda-12.8"


################################################################################
# helpers
################################################################################

def summarize_ms(times) -> Dict[str, float]:
    arr = np.asarray(times, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def run_logged(cmd, cwd, env, log_path: Path, timeout=None):
    """Returns (returncode, elapsed_seconds). returncode is None on timeout."""
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n$ " + " ".join(str(c) for c in cmd) + "\n")
        f.flush()
        start = time.monotonic()
        try:
            proc = subprocess.run(
                cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT,
                timeout=timeout, check=False,
            )
            return proc.returncode, time.monotonic() - start
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - start
            f.write(f"\n[TIMEOUT] after {timeout}s\n")
            return None, elapsed


def infer_output_shape(model, x) -> Dict[str, Any]:
    reset_lif_modules(model)
    with torch.no_grad():
        y = model(x)
    return {"shape": list(y.shape), "dtype": str(y.dtype).replace("torch.", "")}


TORCH_DTYPE_BY_NAME = {
    "float32": torch.float32,
    "float16": torch.float16,
    "int64": torch.int64,
}


################################################################################
# welder compile pipeline (subprocess-driven, mirrors artifacts/tune_welder.py
# and testing/run_welder.py from the welder branch of nnfusion/nnfusion)
################################################################################

def compile_with_welder(
    work_dir: Path,
    arch: str,
    topk: int,
    gpu_device: int,
    welder_repo: str,
    welder_python: str,
    welder_tvm_pythonpath: str,
    nnfusion_bin_dir: str,
    cuda_home: str,
    skip_dot: bool,
    no_tc: bool,
    compile_timeout_sec: int,
) -> Dict[str, Any]:
    # Must be absolute: every subprocess below runs with cwd=work_dir, and a
    # relative work_dir (the common case, since --out-dir defaults to a
    # relative path) would otherwise get resolved a second time against
    # itself wherever it's later passed as a standalone path argument (e.g.
    # cmake -S/-B), doubling the directory prefix.
    work_dir = work_dir.resolve()
    log_path = work_dir / "welder_compile.log"

    base_env = os.environ.copy()
    base_env["PATH"] = f"{cuda_home}/bin:{nnfusion_bin_dir}:{base_env.get('PATH', '')}"
    # welder/utils.py hardcodes "~/cutlass/include" (not CPLUS_INCLUDE_PATH) for
    # its own kernel-profiling nvcc calls, but the *final* model's generated
    # CMakeLists.txt has no -I for cutlass at all -- nvcc/gcc only find it via
    # this env var, which the reference Docker setup exported process-wide.
    base_env["CPLUS_INCLUDE_PATH"] = os.path.expanduser(
        f"~/cutlass/include:{base_env.get('CPLUS_INCLUDE_PATH', '')}"
    )

    nnfusion_cmd1 = ["nnfusion", "model.onnx", "-f", "onnx", "-ftune_output_file=model.json"]
    nnfusion_cmd3 = ["nnfusion", "model.onnx", "-f", "onnx", "-ftune_output_file=/dev/null",
                      "-ftune_input_file=tuned.json"]
    if no_tc:
        nnfusion_cmd1.append("-ftc_rewrite=0")
        nnfusion_cmd3.append("-ftc_rewrite=0")
    if skip_dot:
        nnfusion_cmd1.append("-ffusion_skiplist=Dot")
        nnfusion_cmd3.append("-ffusion_skiplist=Dot")

    # Per-stage wall-clock time (seconds). "run_compiler" is welder's actual
    # autotuning cost (compiling+profiling ~topk candidates per fusion group
    # on the real GPU); the rest is ordinary ONNX-import/codegen/nvcc-build
    # overhead, reported separately since callers may only care about one or
    # the other. Populated incrementally so a failed/timed-out run still
    # reports how long it ran before failing.
    stage_seconds: Dict[str, float] = {}

    def stage_result(ok: bool, stage: str, returncode) -> Dict[str, Any]:
        stage_seconds["total"] = sum(stage_seconds.values())
        return {
            "ok": ok, "stage": stage, "returncode": returncode,
            "log_path": str(log_path), "stage_seconds": dict(stage_seconds),
        }

    rc, elapsed = run_logged(nnfusion_cmd1, work_dir, base_env, log_path)
    stage_seconds["nnfusion_tune_export"] = elapsed
    if rc != 0:
        return stage_result(False, "nnfusion_tune_export", rc)

    # run_compiler needs Welder's own (patched) TVM fork on PYTHONPATH, and
    # must run isolated from any ambient user-site packages (this box has a
    # numpy>=2 / mismatched-deps stack in ~/.local that breaks the old TVM
    # fork's ctypes/numpy glue).
    compiler_env = base_env.copy()
    existing_pp = compiler_env.get("PYTHONPATH", "")
    compiler_env["PYTHONPATH"] = os.pathsep.join(
        p for p in [welder_tvm_pythonpath, f"{welder_repo}/python", existing_pp] if p
    )
    compiler_env["PYTHONNOUSERSITE"] = "1"

    run_compiler_cmd = [
        welder_python, "-m", "run_compiler", "model.json", "tuned.json",
        "--topk", str(topk), "--arch", arch, "--device", str(gpu_device),
    ]
    rc, elapsed = run_logged(run_compiler_cmd, work_dir, compiler_env, log_path, timeout=compile_timeout_sec)
    stage_seconds["run_compiler"] = elapsed
    if rc != 0:
        return stage_result(False, "run_compiler", rc)

    rc, elapsed = run_logged(nnfusion_cmd3, work_dir, base_env, log_path)
    stage_seconds["nnfusion_codegen"] = elapsed
    if rc != 0:
        return stage_result(False, "nnfusion_codegen", rc)

    codegen_dir = work_dir / "nnfusion_rt" / "cuda_codegen"
    build_dir = codegen_dir / "build"
    if build_dir.exists():
        run_logged(["rm", "-rf", str(build_dir)], work_dir, base_env, log_path)

    rc, elapsed = run_logged(["cmake", "-S", str(codegen_dir), "-B", str(build_dir)], work_dir, base_env, log_path)
    stage_seconds["cmake_configure"] = elapsed
    if rc != 0:
        return stage_result(False, "cmake_configure", rc)

    nproc = os.cpu_count() or 4
    rc, elapsed = run_logged(["make", "-C", str(build_dir), f"-j{nproc}"], work_dir, base_env, log_path)
    stage_seconds["cmake_build"] = elapsed
    if rc != 0:
        return stage_result(False, "cmake_build", rc)

    so_path = build_dir / "libnnfusion_naive_rt.so"
    if not so_path.exists():
        return stage_result(False, "artifact_missing", 0)

    stage_seconds["total"] = sum(stage_seconds.values())
    return {
        "ok": True, "stage": "done", "returncode": 0, "log_path": str(log_path),
        "so_path": str(so_path), "codegen_dir": str(codegen_dir),
        "stage_seconds": dict(stage_seconds),
    }


################################################################################
# benchmark the compiled .so (ctypes, mirrors testing/run_welder.py)
################################################################################

def benchmark_shared_lib(
    so_path: Path,
    codegen_dir: Path,
    x: torch.Tensor,
    output_shape,
    output_dtype: torch.dtype,
    gpu_device: int,
    warmup_sec: float,
    iters: int,
) -> Dict[str, Any]:
    cuda_rt = ctypes.CDLL("libcudart.so")
    lib = ctypes.CDLL(str(so_path))

    cur_dir = os.getcwd()
    os.chdir(codegen_dir)
    try:
        lib.cuda_init()
        y = torch.empty(output_shape, dtype=output_dtype, device=f"cuda:{gpu_device}")
        args = [ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(y.data_ptr())]

        def run_once():
            tic = time.monotonic_ns()
            lib.kernel_entry(*args)
            mid = time.monotonic_ns()
            cuda_rt.cudaDeviceSynchronize()
            end = time.monotonic_ns()
            return (mid - tic) / 1e6, (end - tic) / 1e6

        st = time.time()
        while time.time() - st < warmup_sec:
            run_once()

        dispatch_times, total_times = [], []
        for _ in range(iters):
            d, t = run_once()
            dispatch_times.append(d)
            total_times.append(t)
    finally:
        os.chdir(cur_dir)

    return {
        "latency_ms": summarize_ms(total_times),
        "schedule_ms": summarize_ms(dispatch_times),
    }


################################################################################
# one (model, execution_mode, precision) case
################################################################################

def run_case(model_name: str, execution_mode: str, precision: str, args) -> Dict[str, Any]:
    dtype = resolve_dtype(precision)

    result: Dict[str, Any] = {
        "ok": False,
        "model": model_name,
        "execution_mode": execution_mode,
        "precision": precision,
        "batch_size": args.batch_size,
        "time_steps": args.T,
        "arch": args.arch,
        "topk": args.topk,
        "onnx_export_ok": False,
        "welder_compile_ok": False,
        "benchmark_ok": False,
        "error": "",
    }

    if execution_mode != "single_step_mode":
        result["error"] = f"unsupported execution mode: {execution_mode}"
        return result

    run_dir = Path(args.out_dir) / model_name / execution_mode / precision
    run_dir.mkdir(parents=True, exist_ok=True)

    #
    # build model + input (reused from the BladeDISC baseline's build_case,
    # which already applies the same export-only LIF decomposition as TVM
    # and TensorRT) and export ONNX
    #
    try:
        model, x, wrapper_name, replaced = build_case(model_name, args, dtype)
        result["wrapper"] = wrapper_name
        result["export_custom_lif_replaced"] = replaced
        result["input_shape"] = list(x.shape)

        onnx_path = run_dir / f"{model_name}_{execution_mode}_T{args.T}_{precision}.onnx"
        export_onnx(model, x, onnx_path, args.opset)
        result["onnx_path"] = str(onnx_path)
        result["onnx_export_ok"] = True
        result["graph_contains_custom_lif"] = onnx_graph_contains_custom_lif(onnx_path)

        io_info = infer_output_shape(model, x)
        result["output_shape"] = io_info["shape"]
        result["output_dtype"] = io_info["dtype"]
    except Exception:
        result["error"] = traceback.format_exc()
        return result

    #
    # welder workspace: PREFIX dir containing model.onnx, matching
    # artifacts/tune_welder.py's expected layout
    #
    workspace = run_dir / "welder_workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    workspace_onnx = workspace / "model.onnx"
    workspace_onnx.write_bytes(onnx_path.read_bytes())

    compile_result = compile_with_welder(
        work_dir=workspace,
        arch=args.arch,
        topk=args.topk,
        gpu_device=args.gpu_device,
        welder_repo=args.welder_repo,
        welder_python=args.welder_python,
        welder_tvm_pythonpath=args.welder_tvm_pythonpath,
        nnfusion_bin_dir=args.nnfusion_bin_dir,
        cuda_home=args.cuda_home,
        skip_dot=args.skip_dot,
        no_tc=args.no_tc,
        compile_timeout_sec=args.compile_timeout_sec,
    )
    result["welder_compile"] = compile_result
    result["welder_compile_ok"] = compile_result["ok"]
    # Convenient top-level aliases into welder_compile.stage_seconds:
    # autotune_seconds is welder's actual autotuning cost (run_compiler --topk
    # candidates compiled+profiled on the real GPU); welder_compile_seconds_total
    # additionally includes ordinary ONNX-import/codegen/nvcc-build overhead.
    stage_seconds = compile_result.get("stage_seconds", {})
    result["autotune_seconds"] = stage_seconds.get("run_compiler")
    result["welder_compile_seconds_total"] = stage_seconds.get("total")
    if not compile_result["ok"]:
        return result

    #
    # benchmark
    #
    try:
        parsed = benchmark_shared_lib(
            so_path=Path(compile_result["so_path"]),
            codegen_dir=Path(compile_result["codegen_dir"]),
            x=x,
            output_shape=result["output_shape"],
            output_dtype=TORCH_DTYPE_BY_NAME[result["output_dtype"]],
            gpu_device=args.gpu_device,
            warmup_sec=args.warmup_sec,
            iters=args.iters,
        )
        result["parsed"] = parsed
        result["benchmark_ok"] = True
        result["ok"] = True
    except Exception:
        result["error"] = traceback.format_exc()

    return result


################################################################################
# main
################################################################################

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--models", nargs="+", choices=KAIROS_MODEL_CHOICES, default=["resnet18"])
    p.add_argument("--execution-modes", nargs="+", default=["single_step_mode"],
                    choices=["single_step_mode"])
    p.add_argument("--precisions", nargs="+", default=["fp32", "tf32", "fp16"],
                    choices=["fp32", "tf32", "fp16"])

    p.add_argument("--T", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--height", type=int, default=224)
    p.add_argument("--width", type=int, default=224)
    p.add_argument("--model-channels", type=int, default=64)
    p.add_argument("--lif-impl", choices=LIF_IMPL_CHOICES, default="kairos")
    p.add_argument("--sequence-length", type=int, default=256)
    p.add_argument("--transformer-depth", type=int, default=8)
    p.add_argument("--transformer-dim", type=int, default=256)
    p.add_argument("--transformer-heads", type=int, default=8)
    p.add_argument("--transformer-input-dim", type=int, default=768)
    p.add_argument("--transformer-vocab-size", type=int, default=30522)
    p.add_argument("--transformer-num-classes", type=int, default=100)
    p.add_argument("--convlstm-in-channels", type=int, default=1)
    p.add_argument("--convlstm-hidden-channels", type=int, default=64)
    p.add_argument("--convlstm-num-layers", type=int, default=2)
    p.add_argument("--convlstm-height", type=int, default=64)
    p.add_argument("--convlstm-width", type=int, default=64)
    p.add_argument("--mamba-d-model", type=int, default=768)
    p.add_argument("--mamba-n-layer", type=int, default=24)
    p.add_argument("--mamba-d-inner", type=int, default=1536)
    p.add_argument("--mamba-d-state", type=int, default=16)
    p.add_argument("--mamba-d-conv", type=int, default=4)
    p.add_argument("--mamba-dt-rank", type=int, default=48)
    p.add_argument("--deepspeech2-freq-bins", type=int, default=161)
    p.add_argument("--deepspeech2-conv-channels", type=int, default=32)
    p.add_argument("--deepspeech2-gru-hidden", type=int, default=800)
    p.add_argument("--deepspeech2-gru-layers", type=int, default=3)

    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--out-dir", default="test/welder_validation")

    # welder-specific
    p.add_argument("--topk", type=int, default=20, help="Number of tuning trials per subgraph.")
    p.add_argument("--arch", default="RTX5090")
    p.add_argument("--gpu-device", type=int, default=0)
    p.add_argument("--skip-dot", action="store_true")
    p.add_argument("--no-tc", action="store_true")
    p.add_argument(
        "--compile-timeout-sec", type=int, default=7200,
        help="Welder's autotuner compiles+profiles ~topk candidates per fusion "
             "group on the real GPU; for a full model at topk=20 this routinely "
             "takes over an hour (~14 min was observed at topk=3), not seconds.",
    )
    p.add_argument("--warmup-sec", type=float, default=1.0)
    p.add_argument("--iters", type=int, default=100)

    p.add_argument("--welder-repo", default=DEFAULT_WELDER_REPO)
    p.add_argument("--welder-python", default=DEFAULT_WELDER_PYTHON)
    p.add_argument("--welder-tvm-pythonpath", default=DEFAULT_WELDER_TVM_PYTHONPATH)
    p.add_argument("--nnfusion-bin-dir", default=DEFAULT_NNFUSION_BIN_DIR)
    p.add_argument("--cuda-home", default=DEFAULT_CUDA_HOME)

    args = p.parse_args()
    # build_case()/make_model_input() index into args by attribute, expecting
    # a torch device string (e.g. "cuda:0"), separate from --gpu-device (an
    # int used for welder's own --device flag and the nnfusion CLI).
    args.device = f"cuda:{args.gpu_device}"
    return args


def main():
    args = parse_args()
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, Any] = {}

    for model_name in args.models:
        all_results[model_name] = {}
        for execution_mode in args.execution_modes:
            all_results[model_name][execution_mode] = {}
            for precision in args.precisions:
                print("=" * 80)
                print(f"[RUN] model={model_name} mode={execution_mode} precision={precision}")
                print("=" * 80)

                result = run_case(model_name, execution_mode, precision, args)
                all_results[model_name][execution_mode][precision] = result

                run_dir = out_root / model_name / execution_mode / precision
                run_dir.mkdir(parents=True, exist_ok=True)
                summary_path = run_dir / "summary.json"
                summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                print(f"[WRITE] {summary_path}")

    aggregate_path = out_root / "welder_summary_all.json"
    aggregate_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print("=" * 80)
    print("[DONE]")
    print(f"[WRITE] {aggregate_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
