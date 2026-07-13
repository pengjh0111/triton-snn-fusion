import argparse
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

import runtime.snn_custom_ops as snn_custom_ops
from compiler.fx_lif_temporal_rewrite import (
    collect_conv_bn_add_lif_state_patterns,
    count_fused_temporal_conv_add_lif_state_nodes,
    group_temporal_residual_patterns,
    make_temporal_residual_windows,
    rewrite_temporal_conv_bn_add_lif_state_to_fused,
)
from benchmarks.helpers.models_for_fx import CustomStatefulIFNode, reset_custom_stateful_lif_modules


class ConvBnAddLifLoop(nn.Module):
    def __init__(self, T: int, stride: int, residual_left: bool):
        super().__init__()
        self.T = T
        self.residual_left = residual_left
        self.conv = nn.Conv2d(4, 4, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(4)
        self.lif = CustomStatefulIFNode(v_threshold=1.0, v_reset=0.0, tau=2.0)

    def _residual(self, x):
        if self.conv.stride == (2, 2):
            return F.avg_pool2d(x, 2, 2)
        return x

    def forward(self, x):
        out = 0
        residual = self._residual(x)
        for _ in range(self.T):
            y = self.bn(self.conv(x))
            y = residual + y if self.residual_left else y + residual
            out = out + self.lif(y)
        return out / self.T


def build_placeholder_values(gm: torch.fx.GraphModule, example_inputs):
    placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
    return {node: value for node, value in zip(placeholders, example_inputs)}


def make_backend(args, counters):
    def backend(gm: torch.fx.GraphModule, example_inputs):
        placeholder_values = build_placeholder_values(gm, example_inputs)
        patterns = collect_conv_bn_add_lif_state_patterns(gm)
        groups = group_temporal_residual_patterns(patterns)
        windows = make_temporal_residual_windows(groups, args.temporal_fuse_window, allow_tail=True)
        stats = rewrite_temporal_conv_bn_add_lif_state_to_fused(
            gm,
            windows,
            placeholder_values,
            max_patterns=args.max_patterns,
        )
        counters["patterns"] += len(patterns)
        counters["windows"] += len(windows)
        counters["replaced_windows"] += stats.temporal_residual_replaced_windows
        counters["replaced_patterns"] += stats.temporal_residual_replaced_patterns
        counters["skipped_windows"] += stats.temporal_residual_skipped_windows
        counters["fused_nodes"] += count_fused_temporal_conv_add_lif_state_nodes(gm)
        gm.graph.lint()
        gm.recompile()
        return gm.forward

    return backend


def run_case(args, T: int, stride: int, residual_left: bool):
    torch._dynamo.reset()
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    model = ConvBnAddLifLoop(T=T, stride=stride, residual_left=residual_left).to(args.device, dtype=dtype).eval()
    x = torch.randn(args.batch_size, 4, args.height, args.width, device=args.device, dtype=dtype) * 0.25
    counters = {
        "patterns": 0,
        "windows": 0,
        "replaced_windows": 0,
        "replaced_patterns": 0,
        "skipped_windows": 0,
        "fused_nodes": 0,
    }
    compiled = torch.compile(model, backend=make_backend(args, counters), fullgraph=False, dynamic=False)
    with torch.no_grad():
        reset_custom_stateful_lif_modules(model)
        eager = model(x)
        reset_custom_stateful_lif_modules(model)
        actual = compiled(x)
    diff = (eager - actual).abs()
    allclose = torch.allclose(eager, actual, rtol=args.rtol, atol=args.atol)
    print(
        f"T={T:<2d} stride={stride} residual_left={residual_left} "
        f"patterns={counters['patterns']} replaced={counters['replaced_patterns']} "
        f"fused_nodes={counters['fused_nodes']} max={diff.max().item():.3e} "
        f"mean={diff.mean().item():.3e} allclose={allclose}"
    )
    if not allclose:
        raise AssertionError("compiled output differs from eager reference")
    if T > 1 and counters["replaced_patterns"] == 0:
        raise AssertionError("expected temporal residual rewrite to replace at least one pattern")


def parse_args():
    parser = argparse.ArgumentParser(description="Conv+BN+residual-add+LIF temporal rewrite correctness test.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--fused-op-backend", choices=("torch", "triton"), default="torch")
    parser.add_argument("--strict-triton", action="store_true")
    parser.add_argument("--print-fused-op-calls", action="store_true")
    parser.add_argument("--T", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--temporal-fuse-window", type=int, default=16)
    parser.add_argument("--max-patterns", type=int, default=1000)
    parser.add_argument("--rtol", type=float, default=None)
    parser.add_argument("--atol", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if args.rtol is None:
        args.rtol = 1e-2 if args.dtype == "fp16" else 1e-4
    if args.atol is None:
        args.atol = 1e-2 if args.dtype == "fp16" else 1e-4
    snn_custom_ops.configure_fused_op(
        backend=args.fused_op_backend,
        strict_triton=args.strict_triton,
        verbose=args.print_fused_op_calls,
    )
    base_window = args.temporal_fuse_window
    for T in args.T:
        args.temporal_fuse_window = max(1, min(base_window, T))
        for stride in (1, 2):
            for residual_left in (False, True):
                run_case(args, T=T, stride=stride, residual_left=residual_left)
    print("fused_op_call_stats:", snn_custom_ops.get_fused_op_call_stats())


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
