#!/usr/bin/env python3
"""Benchmark end-to-end ONNX Runtime CUDA execution for Chronos workloads."""

import argparse
import json
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.benchmark_tensorrt_runtime import export_onnx
from benchmarks.validate_kairos_baselines import KAIROS_MODEL_CHOICES, LIF_IMPL_CHOICES


# PyTorch's legacy exporter currently lowers the relative-axis expression in
# KairosSpikeTransformerBlock to an invalid ONNX Transpose permutation (for
# example, [-3, 0, 3, 1, 4]). TensorRT accepts that extension, but ORT follows
# the ONNX schema and rejects it during session creation. Keep the compatibility
# rewrite local to this benchmark so other backends and the shared model remain
# untouched.
ORT_NEGATIVE_TRANSPOSE_FIX_MODELS = {"spiketransformer", "spikebert"}


def percentile(values: List[float], q: float) -> float:
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * q))
    return ordered[index]


def torch_dtype(precision: str) -> torch.dtype:
    return torch.float16 if precision == "fp16" else torch.float32


def numpy_dtype(dtype: torch.dtype):
    return np.float16 if dtype == torch.float16 else np.float32


def make_input(model_name: str, shape: List[int], dtype: torch.dtype, vocab_size: int):
    if model_name == "spikebert":
        return torch.randint(
            0, vocab_size, shape, device="cuda", dtype=torch.int64
        ).contiguous()
    return torch.randn(shape, device="cuda", dtype=dtype).contiguous()


def normalize_negative_transpose_perms(onnx_path: Path) -> int:
    """Replace negative Transpose axes with equivalent ONNX-positive axes."""
    import onnx

    model = onnx.load_model(str(onnx_path), load_external_data=True)
    repaired = 0

    def visit_graph(graph) -> None:
        nonlocal repaired
        for node in graph.node:
            if node.op_type == "Transpose":
                perm_attr = next((attr for attr in node.attribute if attr.name == "perm"), None)
                if perm_attr is not None:
                    rank = len(perm_attr.ints)
                    old_perm = list(perm_attr.ints)
                    new_perm = [axis + rank if axis < 0 else axis for axis in old_perm]
                    if any(axis < 0 or axis >= rank for axis in new_perm):
                        raise ValueError(
                            f"Transpose {node.name!r} has out-of-range perm {old_perm}"
                        )
                    if sorted(new_perm) != list(range(rank)):
                        raise ValueError(
                            f"Transpose {node.name!r} has non-permutation perm {old_perm}"
                        )
                    if new_perm != old_perm:
                        del perm_attr.ints[:]
                        perm_attr.ints.extend(new_perm)
                        repaired += 1
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    visit_graph(attr.g)
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    for subgraph in attr.graphs:
                        visit_graph(subgraph)

    visit_graph(model.graph)
    if repaired:
        temporary_path = onnx_path.with_name(f".{onnx_path.name}.ort-fix.tmp")
        onnx.save_model(model, str(temporary_path))
        temporary_path.replace(onnx_path)
    return repaired


def provider_options(args, precision: str) -> Dict[str, str]:
    return {
        "device_id": str(args.device_id),
        "cudnn_conv_algo_search": args.cudnn_conv_algo_search,
        "do_copy_in_default_stream": "1",
        "use_tf32": "1" if precision == "tf32" else "0",
    }


def synchronize_binding(binding) -> None:
    sync = getattr(binding, "synchronize_outputs", None)
    if sync is not None:
        sync()
    else:
        torch.cuda.synchronize()


def bind_cuda_io(session, tensor: torch.Tensor):
    binding = session.io_binding()
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    element_type = np.int64 if tensor.dtype == torch.int64 else numpy_dtype(tensor.dtype)
    binding.bind_input(
        name=input_meta.name,
        device_type="cuda",
        device_id=tensor.device.index or 0,
        element_type=element_type,
        shape=tuple(tensor.shape),
        buffer_ptr=tensor.data_ptr(),
    )
    binding.bind_output(output_meta.name, "cuda", tensor.device.index or 0)
    return binding


def timed_run(session, binding) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    session.run_with_iobinding(binding)
    synchronize_binding(binding)
    return (time.perf_counter() - start) * 1000.0


def benchmark_onnxruntime(onnx_path: Path, model_name: str, precision: str, args):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        return {
            "ok": False,
            "error": f"onnxruntime-gpu is required: {exc}",
        }

    available = ort.get_available_providers()
    if "CUDAExecutionProvider" not in available:
        return {
            "ok": False,
            "error": "CUDAExecutionProvider is unavailable; install a compatible onnxruntime-gpu build",
            "available_providers": available,
        }

    options = provider_options(args, precision)
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.enable_profiling = bool(args.enable_ort_profiling)
    session_options.log_severity_level = args.ort_log_severity

    session_start = time.perf_counter()
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=session_options,
        providers=[("CUDAExecutionProvider", options)],
    )
    session_creation_ms = (time.perf_counter() - session_start) * 1000.0

    input_shape = [int(dim) for dim in session.get_inputs()[0].shape]
    input_tensor = make_input(
        model_name, input_shape, torch_dtype(precision), args.transformer_vocab_size
    )
    binding = bind_cuda_io(session, input_tensor)

    first_inference_ms = timed_run(session, binding)
    warmup_start = time.perf_counter()
    warmup_iterations = 0
    while (time.perf_counter() - warmup_start) * 1000.0 < args.warmup_ms:
        timed_run(session, binding)
        warmup_iterations += 1
    warmup_total_ms = (time.perf_counter() - warmup_start) * 1000.0

    latencies = []
    measured_start = time.perf_counter()
    while time.perf_counter() - measured_start < args.duration_sec:
        latencies.append(timed_run(session, binding))
    measured_wall_sec = time.perf_counter() - measured_start
    if not latencies:
        raise RuntimeError("measurement produced no samples")

    steady_median = statistics.median(latencies)
    # CUDA EP's exhaustive/heuristic cuDNN algorithm selection and other lazy
    # initialization normally occur on the first inference. Report the excess
    # over steady-state median explicitly; session construction is separate.
    autotune_overhead_ms = max(0.0, first_inference_ms - steady_median)
    profile_path = session.end_profiling() if args.enable_ort_profiling else None
    return {
        "ok": True,
        "onnxruntime_version": ort.__version__,
        "available_providers": available,
        "active_providers": session.get_providers(),
        "provider_options_requested": options,
        "graph_optimization_level": "ORT_ENABLE_ALL",
        "input_shape": input_shape,
        "input_dtype": str(input_tensor.dtype),
        "io_binding": "CUDA input and CUDA output",
        "session_creation_ms": session_creation_ms,
        "first_inference_ms": first_inference_ms,
        "autotune_overhead_ms": autotune_overhead_ms,
        "autotune_overhead_method": "max(first_inference_ms - steady_state_median_ms, 0)",
        "warmup_total_ms": warmup_total_ms,
        "warmup_iterations": warmup_iterations,
        "measurement_wall_sec": measured_wall_sec,
        "measurement_iterations": len(latencies),
        "throughput_qps": len(latencies) / measured_wall_sec,
        "latency_ms": {
            "mean": statistics.mean(latencies),
            "min": min(latencies),
            "max": max(latencies),
            "median": steady_median,
            "p90": percentile(latencies, 0.90),
            "p99": percentile(latencies, 0.99),
            "std": statistics.pstdev(latencies),
        },
        "ort_profile_path": profile_path,
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_case(args, model_name: str, execution_mode: str, precision: str):
    run_dir = Path(args.out_dir) / model_name / execution_mode / precision
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[RUN] model={model_name} mode={execution_mode} precision={precision}")

    export_start = time.perf_counter()
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
    onnx_export_ms = (time.perf_counter() - export_start) * 1000.0
    ort_compat_rewrite = {
        "applied": False,
        "negative_transpose_perms_fixed": 0,
        "overhead_ms": 0.0,
    }
    runtime_result: Dict[str, Any] = {"ok": False, "error": "ONNX export failed"}
    if export_result.get("onnx_export_ok"):
        try:
            if model_name in ORT_NEGATIVE_TRANSPOSE_FIX_MODELS:
                rewrite_start = time.perf_counter()
                fixed = normalize_negative_transpose_perms(Path(export_result["onnx_path"]))
                ort_compat_rewrite = {
                    "applied": True,
                    "negative_transpose_perms_fixed": fixed,
                    "overhead_ms": (time.perf_counter() - rewrite_start) * 1000.0,
                }
                print(
                    "[ORT COMPAT REWRITE] "
                    f"negative_transpose_perms_fixed={fixed}"
                )
            runtime_result = benchmark_onnxruntime(
                Path(export_result["onnx_path"]), model_name, precision, args
            )
        except Exception:
            runtime_result = {"ok": False, "error": traceback.format_exc()}
    result = {
        **export_result,
        "runtime": runtime_result,
        "onnx_export_ms": onnx_export_ms,
        "ort_compat_rewrite": ort_compat_rewrite,
        "onnxruntime_ok": bool(runtime_result.get("ok")),
        "ok": bool(export_result.get("onnx_export_ok") and runtime_result.get("ok")),
    }
    write_json(run_dir / "summary.json", result)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=KAIROS_MODEL_CHOICES, default=["resnet18"])
    parser.add_argument("--execution-modes", nargs="+", choices=("single_step_mode",), default=["single_step_mode"])
    parser.add_argument("--precisions", nargs="+", choices=("fp32", "tf32", "fp16"), default=["tf32"])
    parser.add_argument("--lif-impl", choices=LIF_IMPL_CHOICES, default="kairos")
    parser.add_argument("--T", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--model-channels", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--transformer-depth", type=int, default=8)
    parser.add_argument("--transformer-dim", type=int, default=256)
    parser.add_argument("--transformer-heads", type=int, default=8)
    parser.add_argument("--transformer-input-dim", type=int, default=768)
    parser.add_argument("--transformer-vocab-size", type=int, default=30522)
    parser.add_argument("--transformer-num-classes", type=int, default=100)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--cudnn-conv-algo-search", choices=("EXHAUSTIVE", "HEURISTIC", "DEFAULT"), default="EXHAUSTIVE")
    parser.add_argument("--warmup-ms", type=int, default=2000)
    parser.add_argument("--duration-sec", type=float, default=10.0)
    parser.add_argument("--enable-ort-profiling", action="store_true")
    parser.add_argument("--ort-log-severity", type=int, choices=range(5), default=2)
    parser.add_argument("--out-dir", default="test/onnxruntime_validation")
    return parser.parse_args()


def main():
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    all_results = {}
    for model_name in args.models:
        all_results[model_name] = {}
        for execution_mode in args.execution_modes:
            all_results[model_name][execution_mode] = {}
            for precision in args.precisions:
                result = run_case(args, model_name, execution_mode, precision)
                all_results[model_name][execution_mode][precision] = result
                write_json(Path(args.out_dir) / "onnxruntime_summary_all.json", all_results)
    print(f"[WRITE] {Path(args.out_dir) / 'onnxruntime_summary_all.json'}")


if __name__ == "__main__":
    main()
