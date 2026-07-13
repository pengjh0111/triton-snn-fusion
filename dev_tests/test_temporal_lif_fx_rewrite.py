import argparse
import sys
import traceback
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import runtime.snn_custom_ops as snn_custom_ops
from compiler.fx_lif_temporal_rewrite import (
    collect_standalone_lif_state_patterns,
    count_fused_temporal_lif_state_nodes,
    group_temporal_lif_patterns,
    make_temporal_lif_windows,
    rewrite_temporal_lif_state_to_fused,
)
from compiler.fx_temporal_annotation import annotate_temporal_metadata
from benchmarks.helpers.models_for_fx import CustomStatefulIFNode, reset_custom_stateful_lif_modules


class TinyStandaloneTemporalLIF(nn.Module):
    def __init__(self, T: int):
        super().__init__()
        self.T = int(T)
        self.lif = CustomStatefulIFNode(v_threshold=1.0, v_reset=0.0, tau=2.0)

    def forward(self, x):
        outs = []
        for t in range(self.T):
            outs.append(self.lif(x + float(t % 3) * 0.125))
        return torch.stack(outs, dim=0)


def _dtype(name: str):
    return torch.float16 if name == "fp16" else torch.float32


def _sync(device: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def make_backend(args, counters: Dict[str, int], T: int):
    def backend(gm: torch.fx.GraphModule, example_inputs):
        annotate_temporal_metadata(gm, T, T, strict=False)
        patterns = collect_standalone_lif_state_patterns(gm)
        groups = group_temporal_lif_patterns(patterns)
        windows = make_temporal_lif_windows(groups, T, allow_tail=True)
        stats = rewrite_temporal_lif_state_to_fused(gm, windows, max_patterns=100000)
        gm.graph.lint()
        gm.recompile()
        counters["patterns"] += len(patterns)
        counters["windows"] += len(windows)
        counters["rewritten_windows"] += stats.temporal_lif_rewritten_windows
        counters["replaced_patterns"] += stats.temporal_lif_replaced_patterns
        counters["skipped_windows"] += stats.temporal_lif_skipped_windows
        counters["fused_nodes"] += count_fused_temporal_lif_state_nodes(gm)
        return gm.forward

    return backend


def run_case(T: int, args):
    dtype = _dtype(args.dtype)
    model = TinyStandaloneTemporalLIF(T).to(device=args.device, dtype=dtype).eval()
    x = torch.ones(args.batch_size, args.channels, args.height, args.width, device=args.device, dtype=dtype) * 1.25
    counters = {
        "patterns": 0,
        "windows": 0,
        "rewritten_windows": 0,
        "replaced_patterns": 0,
        "skipped_windows": 0,
        "fused_nodes": 0,
    }
    snn_custom_ops.configure_fused_op(
        backend=args.backend,
        strict_triton=args.strict_triton,
        verbose=args.verbose,
    )
    snn_custom_ops.reset_fused_op_call_stats()
    compiled = torch.compile(model, backend=make_backend(args, counters, T), fullgraph=False, dynamic=False)
    reset_custom_stateful_lif_modules(model)
    with torch.no_grad():
        eager = model(x)
    reset_custom_stateful_lif_modules(model)
    with torch.no_grad():
        rewritten = compiled(x)
    _sync(args.device)
    diff = (eager - rewritten).abs()
    allclose = torch.allclose(eager, rewritten, rtol=args.rtol, atol=args.atol)
    call_stats = snn_custom_ops.get_fused_op_call_stats()
    ok = bool(allclose and counters["rewritten_windows"] > 0)
    if args.backend == "triton" and args.device.startswith("cuda"):
        ok = ok and call_stats.get("temporal_lif_triton", 0) > 0 and call_stats.get("temporal_lif_fallback", 0) == 0
    print(
        f"[{'PASS' if ok else 'FAIL'}] dtype={args.dtype} T={T} "
        f"patterns={counters['patterns']} windows={counters['windows']} "
        f"rewritten={counters['rewritten_windows']} fused_nodes={counters['fused_nodes']} "
        f"max={diff.max().item():.3e} allclose={allclose} "
        f"temporal_lif_triton={call_stats.get('temporal_lif_triton', 0)} "
        f"temporal_lif_fallback={call_stats.get('temporal_lif_fallback', 0)}"
    )
    if not ok:
        raise RuntimeError(
            f"failed T={T}: counters={counters}, call_stats={call_stats}, max_diff={diff.max().item()}, allclose={allclose}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="FX rewrite test for standalone temporal LIF fusion.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--backend", choices=("torch", "triton"), default="triton")
    parser.add_argument("--T", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--strict-triton", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--rtol", type=float, default=None)
    parser.add_argument("--atol", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    if args.rtol is None:
        args.rtol = 1e-2 if args.dtype == "fp16" else 1e-5
    if args.atol is None:
        args.atol = 1e-2 if args.dtype == "fp16" else 1e-5
    if args.device == "cpu":
        args.backend = "torch"
    failed = 0
    for T in args.T:
        try:
            run_case(T, args)
        except Exception:
            failed += 1
            traceback.print_exc()
    if failed:
        raise SystemExit(f"{failed} FX temporal LIF rewrite case(s) failed")


if __name__ == "__main__":
    main()
