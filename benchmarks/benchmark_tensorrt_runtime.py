# benchmarks/benchmark_tensorrt_runtime.py

import argparse
import copy
import json
import re
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Dict, Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.validate_chronos_baselines import (
    CHRONOS_MODEL_CHOICES,
    LIF_IMPL_CHOICES,
    SingleStepModeLoopWrapper,
    make_resnet_layer,
    reset_lif_modules,
)
from test.models_for_fx_test import CustomStatefulIFNode


################################################################################
# helpers
################################################################################

def resolve_dtype(dtype: str):
    if dtype == "fp16":
        return torch.float16
    elif dtype in ("fp32", "tf32"):
        return torch.float32
    else:
        raise ValueError(dtype)


class ExportableStatefulIFNode(CustomStatefulIFNode):
    """ONNX/TensorRT export-only decomposition of Chronos stateful LIF."""

    def forward(self, x):
        self.reset_state_if_needed(x)
        if float(self.tau) <= 1.0:
            v_before_spike = self.v + x
        else:
            v_before_spike = self.v + (x - self.v) / float(self.tau)

        spike = (v_before_spike >= float(self.v_threshold)).to(x.dtype)
        spike_for_reset = spike.detach() if bool(self.detach_reset) else spike

        if float(self.v_reset) < 0:
            v_next = v_before_spike - spike_for_reset * float(self.v_threshold)
        else:
            v_next = torch.where(
                spike_for_reset > 0,
                torch.full_like(v_before_spike, float(self.v_reset)),
                v_before_spike,
            )

        self.v = v_next
        return spike


def _make_exportable_lif_node(module: CustomStatefulIFNode) -> ExportableStatefulIFNode:
    exportable = ExportableStatefulIFNode(
        v_threshold=module.v_threshold,
        v_reset=module.v_reset,
        tau=module.tau,
        detach_reset=module.detach_reset,
        surrogate_function=module.surrogate_function,
        step_mode=module.step_mode,
    )
    exportable.v = module.v.detach().clone()
    exportable.train(module.training)
    return exportable


def replace_custom_lif_for_export(module: torch.nn.Module) -> int:
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, CustomStatefulIFNode) and not isinstance(child, ExportableStatefulIFNode):
            setattr(module, name, _make_exportable_lif_node(child))
            replaced += 1
        else:
            replaced += replace_custom_lif_for_export(child)
    return replaced


def onnx_graph_contains_custom_lif(onnx_path: Path) -> bool:
    try:
        blob = onnx_path.read_bytes()
    except OSError:
        return False
    return b"snn_custom" in blob or b"lif_forward_state" in blob


################################################################################
# export onnx
################################################################################

def export_onnx(
    model_name: str,
    execution_mode: str,
    precision: str,
    model_channels: int,
    lif_impl: str,
    T: int,
    batch_size: int,
    height: int,
    width: int,
    opset: int,
    out_dir: Path,
):
    dtype = resolve_dtype(precision)
    onnx_path = (
        out_dir
        / f"{model_name}_{execution_mode}_T{T}_{precision}.onnx"
    )

    result = {
        "ok": False,
        "onnx_export_ok": False,
        "onnx_path": str(onnx_path),
        "precision": precision,
        "model_channels": model_channels,
        "lif_impl": lif_impl,
        "custom_lif": lif_impl == "chronos",
        "wrapper": "SingleStepModeLoopWrapper",
        "graph_contains_custom_lif": False,
        "export_rewrite_custom_lif": lif_impl == "chronos",
        "export_custom_lif_replaced": 0,
        "export_graph_contains_custom_op": None,
        "error": "",
    }

    if execution_mode == "single_step_mode":

        layer = make_resnet_layer(
            model_name=model_name,
            allow_resnet32_fallback=True,
            step_mode="s",
            model_channels=model_channels,
            lif_impl=lif_impl,
        )
        model = SingleStepModeLoopWrapper(
            layer=layer,
            T=T,
        )

    else:
        raise ValueError(execution_mode)

    model = model.cuda().to(dtype).eval()
    result["graph_contains_custom_lif"] = any(
        module.__class__.__name__ == "CustomStatefulIFNode"
        for module in model.modules()
    )

    print("[MODEL DEBUG]")
    print(model)
    graph = getattr(model, "graph", None)
    if graph is not None:
        print("[MODEL GRAPH]")
        print(graph)
    print(
        "[MODEL DEBUG] "
        f"custom_lif={result['graph_contains_custom_lif']} "
        "custom_lif_op=torch.ops.snn_custom.lif_forward_state.default "
        "wrapper=SingleStepModeLoopWrapper"
    )

    x = torch.randn(
        batch_size,
        3,
        height,
        width,
        device="cuda",
        dtype=dtype,
    )

    reset_lif_modules(model)
    model_for_export = copy.deepcopy(model)
    replaced = replace_custom_lif_for_export(model_for_export)
    reset_lif_modules(model_for_export)
    result["export_custom_lif_replaced"] = replaced
    result["export_graph_contains_custom_op"] = any(
        module.__class__.__name__ == "CustomStatefulIFNode"
        for module in model_for_export.modules()
    )
    print("[EXPORT REWRITE]")
    print(f"custom_lif_modules_replaced={replaced}")
    print(f"export_graph_contains_custom_op={result['export_graph_contains_custom_op']}")

    print(f"[EXPORT ONNX] {onnx_path}")

    try:
        with torch.no_grad():

            torch.onnx.export(
                model_for_export,
                (x,),
                onnx_path.as_posix(),
                input_names=["input"],
                output_names=["output"],
                opset_version=opset,
                dynamo=False,
                do_constant_folding=False,
                optimize=False,
            )
    except Exception as exc:
        error = traceback.format_exc()
        print("[ONNX EXPORT FAILED]")
        print(error)
        result["error"] = error or str(exc)
        return result

    result["ok"] = True
    result["onnx_export_ok"] = True
    result["export_graph_contains_custom_op"] = onnx_graph_contains_custom_lif(onnx_path)
    print(f"[EXPORT CHECK] onnx_contains_custom_lif={result['export_graph_contains_custom_op']}")
    return result


################################################################################
# trtexec
################################################################################

def parse_trtexec_output(text: str) -> Dict[str, Any]:

    out = {}

    latency_match = re.search(
        r"Latency:\s*min = ([\d.]+) ms, max = ([\d.]+) ms, "
        r"mean = ([\d.]+) ms, median = ([\d.]+) ms",
        text,
    )

    if latency_match:
        out["latency_ms"] = {
            "min": float(latency_match.group(1)),
            "max": float(latency_match.group(2)),
            "mean": float(latency_match.group(3)),
            "median": float(latency_match.group(4)),
        }

    gpu_match = re.search(
        r"GPU Compute Time:\s*min = ([\d.]+) ms, "
        r"max = ([\d.]+) ms, "
        r"mean = ([\d.]+) ms, "
        r"median = ([\d.]+) ms",
        text,
    )

    if gpu_match:
        out["gpu_compute_ms"] = {
            "min": float(gpu_match.group(1)),
            "max": float(gpu_match.group(2)),
            "mean": float(gpu_match.group(3)),
            "median": float(gpu_match.group(4)),
        }

    throughput_match = re.search(
        r"Throughput:\s*([\d.]+)\s*qps",
        text,
    )

    if throughput_match:
        out["throughput_qps"] = float(
            throughput_match.group(1)
        )

    return out


def run_trtexec(
    onnx_path: Path,
    precision: str,
    workspace_mb: int,
    warmup_ms: int,
    duration_sec: int,
    out_dir: Path,
):
    engine_path = (
        out_dir
        / f"{onnx_path.stem}_{precision}.engine"
    )

    log_path = (
        out_dir
        / f"{onnx_path.stem}_{precision}.log"
    )

    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--memPoolSize=workspace:{workspace_mb}",
        f"--warmUp={warmup_ms}",
        f"--duration={duration_sec}",
        # "--builderOptimizationLevel=1",
        "--separateProfileRun",
    ]

    #
    # precision
    #

    if precision == "fp16":
        cmd.append("--fp16")

    elif precision == "tf32":
        pass

    elif precision == "fp32":
        cmd.append("--noTF32")

    print("[TRTEXEC]")
    print(" ".join(cmd))

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        text = f"trtexec not found: {exc}"
        log_path.write_text(text, encoding="utf-8")
        return {
            "ok": False,
            "returncode": None,
            "onnx_path": str(onnx_path),
            "engine_path": str(engine_path),
            "log_path": str(log_path),
            "precision": precision,
            "parsed": {},
            "error": text,
        }

    log_path.write_text(
        proc.stdout,
        encoding="utf-8",
    )

    parsed = parse_trtexec_output(proc.stdout)

    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "onnx_path": str(onnx_path),
        "engine_path": str(engine_path),
        "log_path": str(log_path),
        "precision": precision,
        "parsed": parsed,
    }


################################################################################
# main
################################################################################

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models",
        nargs="+",
        default=["resnet18"],
        choices=CHRONOS_MODEL_CHOICES,
    )

    parser.add_argument(
        "--execution-modes",
        nargs="+",
        default=[
            "single_step_mode",
        ],
        choices=[
            "single_step_mode",
        ],
    )

    parser.add_argument(
        "--precisions",
        nargs="+",
        default=[
            "fp32",
            "tf32",
            "fp16",
        ],
        choices=[
            "fp32",
            "tf32",
            "fp16",
        ],
    )

    parser.add_argument("--T", type=int, default=16)

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

    parser.add_argument("--batch-size", type=int, default=16)

    parser.add_argument("--height", type=int, default=224)

    parser.add_argument("--width", type=int, default=224)

    parser.add_argument("--opset", type=int, default=17)

    parser.add_argument("--workspace-mb", type=int, default=4096)

    parser.add_argument("--warmup-ms", type=int, default=2000)

    parser.add_argument("--duration-sec", type=int, default=10)

    parser.add_argument(
        "--out-dir",
        default="test/tensorrt_validation",
    )

    args = parser.parse_args()

    out_root = Path(args.out_dir)

    out_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results = {}

    for model_name in args.models:

        all_results[model_name] = {}

        for execution_mode in args.execution_modes:

            all_results[model_name][execution_mode] = {}

            for precision in args.precisions:

                print("=" * 80)
                print(
                    f"[RUN] "
                    f"model={model_name} "
                    f"mode={execution_mode} "
                    f"precision={precision}"
                )
                print("=" * 80)

                run_dir = (
                    out_root
                    / model_name
                    / execution_mode
                    / precision
                )

                run_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                #
                # export onnx
                #

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
                        "trtexec_ok": False,
                        "returncode": None,
                        "engine_path": "",
                        "log_path": "",
                        "parsed": {},
                    }
                    all_results[model_name][execution_mode][precision] = result

                    summary_path = (
                        run_dir / "summary.json"
                    )

                    summary_path.write_text(
                        json.dumps(result, indent=2),
                        encoding="utf-8",
                    )

                    print(f"[WRITE] {summary_path}")
                    continue

                #
                # trtexec
                #

                trt_result = run_trtexec(
                    onnx_path=Path(export_result["onnx_path"]),
                    precision=precision,
                    workspace_mb=args.workspace_mb,
                    warmup_ms=args.warmup_ms,
                    duration_sec=args.duration_sec,
                    out_dir=run_dir,
                )
                result = {
                    **export_result,
                    **trt_result,
                    "onnx_export_ok": export_result["onnx_export_ok"],
                    "custom_lif": export_result["custom_lif"],
                    "model_channels": export_result["model_channels"],
                    "lif_impl": export_result["lif_impl"],
                    "wrapper": export_result["wrapper"],
                    "graph_contains_custom_lif": export_result["graph_contains_custom_lif"],
                    "export_rewrite_custom_lif": export_result["export_rewrite_custom_lif"],
                    "export_custom_lif_replaced": export_result["export_custom_lif_replaced"],
                    "export_graph_contains_custom_op": export_result["export_graph_contains_custom_op"],
                    "ok": export_result["onnx_export_ok"] and trt_result["ok"],
                    "trtexec_ok": trt_result["ok"],
                }

                all_results[model_name][execution_mode][precision] = result

                summary_path = (
                    run_dir / "summary.json"
                )

                summary_path.write_text(
                    json.dumps(result, indent=2),
                    encoding="utf-8",
                )

                print(f"[WRITE] {summary_path}")

    aggregate_path = (
        out_root / "tensorrt_summary_all.json"
    )

    aggregate_path.write_text(
        json.dumps(all_results, indent=2),
        encoding="utf-8",
    )

    print("=" * 80)
    print("[DONE]")
    print(f"[WRITE] {aggregate_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
