import argparse
import copy
import json
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from spikingjelly.activation_based import functional, surrogate
from spikingjelly.activation_based.model import spiking_resnet

import runtime.snn_custom_ops as snn_custom_ops
from compiler.fx_lif_rewrite import (
    count_fused_conv_lif_state_nodes,
    count_lif_state_nodes,
    match_conv_bn_lif_state,
    match_conv_lif_state,
    rewrite_conv_bn_lif_state_to_fused,
    rewrite_conv_lif_state_to_fused,
)
from compiler.fx_lif_temporal_rewrite import (
    collect_conv_bn_lif_state_patterns,
    count_fused_temporal_conv_lif_state_nodes,
    dump_temporal_patterns,
    dump_temporal_rewrite_log,
    dump_temporal_windows,
    group_temporal_patterns,
    make_temporal_windows,
    rewrite_temporal_conv_bn_lif_state_to_fused,
)
from compiler.fx_temporal_scheduler import reorder_fx_graph_by_temporal_windows
from test.models_for_fx_test import CustomStatefulIFNode, reset_custom_stateful_lif_modules


@dataclass
class RunResult:
    name: str
    ok: bool
    shape: Optional[List[int]] = None
    dtype: Optional[str] = None
    elapsed_ms: Optional[float] = None
    max_abs_diff: Optional[float] = None
    mean_abs_diff: Optional[float] = None
    allclose: Optional[bool] = None
    error: str = ""


@dataclass
class RewriteCounters:
    captured_graphs: int = 0
    lif_state_nodes: int = 0
    direct_matches: int = 0
    conv_bn_matches: int = 0
    direct_replaced: int = 0
    conv_bn_replaced: int = 0
    fused_state_nodes: int = 0
    fused_temporal_state_nodes: int = 0
    temporal_groups: int = 0
    temporal_windows: int = 0
    temporal_replaced_windows: int = 0
    temporal_replaced_patterns: int = 0
    temporal_skipped_windows: int = 0
    single_step_replaced_patterns: int = 0
    temporal_schedule_ok: bool = False
    temporal_schedule_windows: int = 0
    temporal_schedule_moved_nodes: int = 0
    temporal_schedule_reason: str = ""


class SingleStepWrapper(nn.Module):
    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        return self.layer(x)


class MultiStepWrapper(nn.Module):
    def __init__(self, layer: nn.Module, T: int):
        super().__init__()
        self.layer = layer
        self.T = T

    def forward(self, x):
        out_spikes_counter = 0
        for _ in range(self.T):
            out_spikes_counter = out_spikes_counter + self.layer(x)
        return out_spikes_counter / self.T


def _make_spiking_resnet32_from_blocks():
    # Some SpikingJelly versions do not export spiking_resnet32. This fallback
    # keeps the validation script usable while reporting that the direct API was absent.
    return spiking_resnet.SpikingResNet(
        spiking_resnet.BasicBlock,
        [3, 4, 5, 3],
        spiking_neuron=CustomStatefulIFNode,
        surrogate_function=surrogate.ATan(),
    )


def make_resnet_layer(model_name: str, allow_resnet32_fallback: bool) -> nn.Module:
    if model_name == "resnet18":
        layer = spiking_resnet.spiking_resnet18(
            pretrained=False,
            spiking_neuron=CustomStatefulIFNode,
            surrogate_function=surrogate.ATan(),
        )
    elif model_name == "resnet32":
        if hasattr(spiking_resnet, "spiking_resnet32"):
            layer = spiking_resnet.spiking_resnet32(
                pretrained=False,
                spiking_neuron=CustomStatefulIFNode,
                surrogate_function=surrogate.ATan(),
            )
        elif allow_resnet32_fallback:
            print("[WARN] spiking_resnet.spiking_resnet32 is not available; using SpikingResNet BasicBlock [3,4,5,3] fallback.")
            layer = _make_spiking_resnet32_from_blocks()
        else:
            raise RuntimeError("spiking_resnet.spiking_resnet32 is not available in this SpikingJelly install")
    else:
        raise ValueError(f"unsupported model: {model_name}")

    functional.set_step_mode(layer, step_mode="s")
    return layer


def build_placeholder_values(gm: torch.fx.GraphModule, example_inputs) -> Dict[torch.fx.Node, Any]:
    placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
    return {node: value for node, value in zip(placeholders, example_inputs)}


def save_graph_files(gm: torch.fx.GraphModule, out_dir: Path, prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{prefix}_fx.py").write_text(gm.code, encoding="utf-8")
    (out_dir / f"{prefix}_fx.txt").write_text(str(gm.graph), encoding="utf-8")


def make_rewrite_backend(args, graph_dir: Path, counters: RewriteCounters):
    def backend(gm: torch.fx.GraphModule, example_inputs):
        graph_idx = counters.captured_graphs
        counters.captured_graphs += 1
        local_dir = graph_dir if graph_idx == 0 else graph_dir / f"graph_{graph_idx}"
        local_dir.mkdir(parents=True, exist_ok=True)
        save_graph_files(gm, local_dir, "original")

        placeholder_values = build_placeholder_values(gm, example_inputs)
        lif_state_count = count_lif_state_nodes(gm)
        temporal_replaced_patterns = 0
        temporal_log: List[str] = []

        temporal_patterns = collect_conv_bn_lif_state_patterns(gm) if not args.disable_conv_bn_lif else []
        if args.enable_temporal_schedule and temporal_patterns:
            schedule_window = args.temporal_schedule_window or args.temporal_fuse_window
            schedule_result = reorder_fx_graph_by_temporal_windows(
                gm,
                args.T,
                schedule_window,
                temporal_patterns,
                dump_dir=local_dir if args.temporal_schedule_dump else None,
                strict=args.temporal_schedule_strict,
            )
            counters.temporal_schedule_ok = schedule_result.ok
            counters.temporal_schedule_windows += schedule_result.scheduled_windows
            counters.temporal_schedule_moved_nodes += schedule_result.moved_nodes
            counters.temporal_schedule_reason = schedule_result.reason
            if schedule_result.ok:
                temporal_patterns = collect_conv_bn_lif_state_patterns(gm)
            elif args.temporal_schedule_strict:
                raise RuntimeError(schedule_result.reason)
            else:
                print(f"[SCHEDULE][FALLBACK] {schedule_result.reason}")

        if args.enable_temporal_rewrite and args.temporal_fuse_window > 1 and not args.disable_conv_bn_lif:
            temporal_groups = group_temporal_patterns(temporal_patterns)
            temporal_windows = make_temporal_windows(
                temporal_groups,
                args.temporal_fuse_window,
                args.temporal_allow_tail,
            )
            dump_temporal_patterns(temporal_groups, local_dir / "temporal_patterns.txt")
            dump_temporal_windows(temporal_windows, local_dir / "temporal_windows.txt")
            counters.temporal_groups += len(temporal_groups)
            counters.temporal_windows += len(temporal_windows)
            if args.disable_rewrite:
                temporal_log.append("SKIP: --disable-rewrite enabled")
            else:
                temporal_stats = rewrite_temporal_conv_bn_lif_state_to_fused(
                    gm,
                    temporal_windows,
                    placeholder_values,
                    args.max_patterns,
                )
                temporal_replaced_patterns = temporal_stats.temporal_replaced_patterns
                temporal_log.extend(temporal_stats.log)
                counters.temporal_replaced_windows += temporal_stats.temporal_replaced_windows
                counters.temporal_replaced_patterns += temporal_stats.temporal_replaced_patterns
                counters.temporal_skipped_windows += temporal_stats.temporal_skipped_windows
            dump_temporal_rewrite_log(temporal_log, local_dir / "temporal_rewrite_log.txt")

        direct_matches = match_conv_lif_state(gm)
        conv_bn_matches = []
        if not args.disable_conv_bn_lif:
            conv_bn_matches = match_conv_bn_lif_state(gm)

        direct_replaced = 0
        conv_bn_replaced = 0
        if not args.disable_rewrite:
            remaining = max(0, args.max_patterns - temporal_replaced_patterns)
            conv_bn_replaced = rewrite_conv_bn_lif_state_to_fused(
                gm,
                conv_bn_matches,
                placeholder_values,
                remaining,
            )
            remaining = max(0, remaining - conv_bn_replaced)
            direct_replaced = rewrite_conv_lif_state_to_fused(
                gm,
                direct_matches,
                placeholder_values,
                remaining,
            )
        else:
            gm.graph.lint()
            gm.recompile()

        fused_state_count = count_fused_conv_lif_state_nodes(gm)
        fused_temporal_state_count = count_fused_temporal_conv_lif_state_nodes(gm)
        save_graph_files(gm, local_dir, "rewritten")

        counters.lif_state_nodes += lif_state_count
        counters.direct_matches += len(direct_matches)
        counters.conv_bn_matches += len(conv_bn_matches)
        counters.direct_replaced += direct_replaced
        counters.conv_bn_replaced += conv_bn_replaced
        counters.fused_state_nodes += fused_state_count
        counters.fused_temporal_state_nodes += fused_temporal_state_count
        counters.single_step_replaced_patterns += direct_replaced + conv_bn_replaced

        if args.rewrite_backend_mode == "eager":
            return gm.forward
        gm.meta.pop("dynamo_compile_id", None)
        if hasattr(gm, "_param_name_to_source"):
            delattr(gm, "_param_name_to_source")
        return torch._inductor.compile(gm, example_inputs)

    return backend


def synchronize_if_needed(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def run_model(name: str, model: nn.Module, x: torch.Tensor, device: str, compile_mode: bool, backend=None) -> RunResult:
    try:
        model.eval()
        reset_custom_stateful_lif_modules(model)
        runnable = model
        if compile_mode:
            runnable = torch.compile(model, backend=backend if backend is not None else "inductor", fullgraph=False, dynamic=False)
        synchronize_if_needed(device)
        start = time.perf_counter()
        with torch.no_grad():
            out = runnable(x)
        synchronize_if_needed(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if isinstance(out, (tuple, list)):
            out = out[0]
        return RunResult(
            name=name,
            ok=True,
            shape=list(out.shape),
            dtype=str(out.dtype),
            elapsed_ms=elapsed_ms,
        ), out.detach()
    except Exception:
        return RunResult(name=name, ok=False, error=traceback.format_exc()), None


def compare_to(result: RunResult, out: Optional[torch.Tensor], ref: Optional[torch.Tensor], rtol: float, atol: float):
    if out is None or ref is None or not result.ok:
        return
    diff = (out - ref).abs()
    result.max_abs_diff = diff.max().item()
    result.mean_abs_diff = diff.mean().item()
    result.allclose = torch.allclose(out, ref, rtol=rtol, atol=atol)


def write_summary(path: Path, payload: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def validate_one_model(model_name: str, args) -> Dict[str, Any]:
    print(f"\n================ {model_name} ================")
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)

    base_layer = make_resnet_layer(model_name, allow_resnet32_fallback=not args.require_direct_resnet32_api)
    base_layer = base_layer.to(device=args.device, dtype=torch.float32).eval()
    x = torch.randn(args.batch_size, 3, args.height, args.width, device=args.device, dtype=torch.float32)

    models = {
        "baseline_s_eager": SingleStepWrapper(copy.deepcopy(base_layer)).to(args.device).eval(),
        "baseline_s_compile": SingleStepWrapper(copy.deepcopy(base_layer)).to(args.device).eval(),
        "baseline_m_eager": MultiStepWrapper(copy.deepcopy(base_layer), args.T).to(args.device).eval(),
        "baseline_m_compile": MultiStepWrapper(copy.deepcopy(base_layer), args.T).to(args.device).eval(),
        "rewrite_s_compile": SingleStepWrapper(copy.deepcopy(base_layer)).to(args.device).eval(),
        "rewrite_m_compile": MultiStepWrapper(copy.deepcopy(base_layer), args.T).to(args.device).eval(),
    }

    snn_custom_ops.configure_fused_op(
        backend=args.fused_op_backend,
        strict_triton=args.strict_triton,
        verbose=args.print_fused_op_calls,
    )
    snn_custom_ops.reset_fused_op_call_stats()

    out_dir = Path(args.out_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, RunResult] = {}
    outputs: Dict[str, Optional[torch.Tensor]] = {}

    for case_name, compile_mode, backend in [
        ("baseline_s_eager", False, None),
        ("baseline_s_compile", True, None),
        ("baseline_m_eager", False, None),
        ("baseline_m_compile", True, None),
    ]:
        print(f"[RUN] {model_name}/{case_name}")
        result, out = run_model(case_name, models[case_name], x, args.device, compile_mode, backend)
        results[case_name] = result
        outputs[case_name] = out
        if not result.ok:
            print(f"[FAIL] {case_name}: {result.error.splitlines()[-1] if result.error else 'unknown error'}")

    rewrite_counters: Dict[str, RewriteCounters] = {
        "rewrite_s_compile": RewriteCounters(),
        "rewrite_m_compile": RewriteCounters(),
    }
    for case_name, ref_name in [
        ("rewrite_s_compile", "baseline_s_eager"),
        ("rewrite_m_compile", "baseline_m_eager"),
    ]:
        print(f"[RUN] {model_name}/{case_name}")
        backend = make_rewrite_backend(args, out_dir / case_name, rewrite_counters[case_name])
        result, out = run_model(case_name, models[case_name], x, args.device, True, backend)
        results[case_name] = result
        outputs[case_name] = out
        if not result.ok:
            print(f"[FAIL] {case_name}: {result.error.splitlines()[-1] if result.error else 'unknown error'}")

    compare_pairs = {
        "baseline_s_compile": "baseline_s_eager",
        "baseline_m_compile": "baseline_m_eager",
        "rewrite_s_compile": "baseline_s_eager",
        "rewrite_m_compile": "baseline_m_eager",
    }
    for case_name, ref_name in compare_pairs.items():
        compare_to(results[case_name], outputs[case_name], outputs[ref_name], args.rtol, args.atol)

    call_stats = snn_custom_ops.get_fused_op_call_stats()
    payload = {
        "model": model_name,
        "input_shape": [args.batch_size, 3, args.height, args.width],
        "T": args.T,
        "temporal_fuse_window": args.temporal_fuse_window,
        "enable_temporal_rewrite": args.enable_temporal_rewrite,
        "fused_op_backend": args.fused_op_backend,
        "results": {name: asdict(result) for name, result in results.items()},
        "rewrite_counters": {name: asdict(counters) for name, counters in rewrite_counters.items()},
        "fused_op_call_stats": call_stats,
    }
    write_summary(out_dir / "summary.json", payload)

    print(f"\n[SUMMARY] {model_name}")
    for name in [
        "baseline_s_eager",
        "baseline_s_compile",
        "baseline_m_eager",
        "baseline_m_compile",
        "rewrite_s_compile",
        "rewrite_m_compile",
    ]:
        result = results[name]
        status = "OK" if result.ok else "FAIL"
        diff = ""
        if result.max_abs_diff is not None:
            diff = f" max={result.max_abs_diff:.3e} mean={result.mean_abs_diff:.3e} allclose={result.allclose}"
        print(f"  {name}: {status}{diff}")
    print(f"  rewrite_s counters: {asdict(rewrite_counters['rewrite_s_compile'])}")
    print(f"  rewrite_m counters: {asdict(rewrite_counters['rewrite_m_compile'])}")
    print(f"  temporal_fuse_window: {args.temporal_fuse_window}")
    print(f"  fused calls: {call_stats}")
    print(f"  wrote: {out_dir / 'summary.json'}")
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description="Validate Chronos FX Conv+BN+LIF rewrite against baseline s/m eager/compile.")
    parser.add_argument("--models", nargs="+", default=["resnet18", "resnet32"], choices=["resnet18", "resnet32"])
    parser.add_argument("--T", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fused-op-backend", choices=("torch", "triton"), default="torch")
    parser.add_argument("--rewrite-backend-mode", choices=("eager", "inductor"), default="inductor")
    parser.add_argument("--strict-triton", action="store_true")
    parser.add_argument("--disable-rewrite", action="store_true")
    parser.add_argument("--disable-conv-bn-lif", action="store_true")
    parser.add_argument("--enable-temporal-rewrite", action="store_true")
    parser.add_argument("--temporal-fuse-window", type=int, default=1)
    parser.add_argument("--temporal-allow-tail", action="store_true")
    parser.add_argument("--enable-temporal-schedule", action="store_true")
    parser.add_argument("--temporal-schedule-window", type=int, default=None)
    parser.add_argument("--temporal-schedule-dump", action="store_true")
    parser.add_argument("--temporal-schedule-strict", action="store_true")
    parser.add_argument("--max-patterns", type=int, default=1)
    parser.add_argument("--print-fused-op-calls", action="store_true")
    parser.add_argument("--require-direct-resnet32-api", action="store_true")
    parser.add_argument("--out-dir", default="chronos_baseline_validation")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    all_payloads = {}
    for model_name in args.models:
        try:
            all_payloads[model_name] = validate_one_model(model_name, args)
        except Exception:
            print(f"[MODEL FAIL] {model_name}")
            traceback.print_exc()
            all_payloads[model_name] = {"model": model_name, "error": traceback.format_exc()}

    write_summary(Path(args.out_dir) / "summary_all.json", all_payloads)
    print(f"\nWrote aggregate summary: {Path(args.out_dir) / 'summary_all.json'}")


if __name__ == "__main__":
    main()
