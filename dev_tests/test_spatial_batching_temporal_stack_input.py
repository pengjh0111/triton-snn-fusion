import argparse
import copy
import operator
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import runtime.snn_custom_ops  # noqa: F401 - registers custom ops
from compiler.fx_spatial_batching import apply_spatial_batching
from compiler.fx_temporal_spatial_canonicalize import canonicalize_temporal_spatial_ir


def temporal_lif_stack(x_seq, v_init):
    spike_stack, _v_last = torch.ops.snn_custom.fused_temporal_lif_state.default(
        x_seq,
        v_init,
        1.0,
        0.0,
        2.0,
        False,
    )
    return spike_stack


class TemporalStackConvNet(nn.Module):
    def __init__(self, T: int):
        super().__init__()
        self.T = T
        self.weight = nn.Parameter(torch.randn(5, 3, 3, 3) * 0.1)
        self.bias = nn.Parameter(torch.randn(5) * 0.1)

    def forward(self, x_seq, v_init):
        stack = temporal_lif_stack(x_seq, v_init)
        return torch.stack([F.conv2d(stack[t], self.weight, self.bias, stride=1, padding=1) for t in range(self.T)], dim=0)


class TemporalStackBNNet(nn.Module):
    def __init__(self, T: int):
        super().__init__()
        self.T = T
        self.weight = nn.Parameter(torch.randn(3) * 0.1 + 1.0)
        self.bias = nn.Parameter(torch.randn(3) * 0.1)
        self.register_buffer("running_mean", torch.randn(3) * 0.1)
        self.register_buffer("running_var", torch.rand(3) + 0.5)

    def forward(self, x_seq, v_init):
        stack = temporal_lif_stack(x_seq, v_init)
        return torch.stack(
            [
                F.batch_norm(
                    stack[t],
                    self.running_mean,
                    self.running_var,
                    self.weight,
                    self.bias,
                    False,
                    0.1,
                    1e-5,
                )
                for t in range(self.T)
            ],
            dim=0,
        )


class TemporalStackPoolNet(nn.Module):
    def __init__(self, T: int, adaptive: bool):
        super().__init__()
        self.T = T
        self.adaptive = adaptive

    def forward(self, x_seq, v_init):
        stack = temporal_lif_stack(x_seq, v_init)
        if self.adaptive:
            outs = [F.adaptive_avg_pool2d(stack[t], (2, 2)) for t in range(self.T)]
        else:
            outs = [F.max_pool2d(stack[t], kernel_size=2, stride=2) for t in range(self.T)]
        return torch.stack(outs, dim=0)


class TemporalStackAddNet(nn.Module):
    def __init__(self, T: int):
        super().__init__()
        self.T = T

    def forward(self, x_seq, v_init, y_seq, y_v_init):
        a = temporal_lif_stack(x_seq, v_init)
        b = temporal_lif_stack(y_seq, y_v_init)
        return torch.stack([a[t] + b[t] for t in range(self.T)], dim=0)


class TemporalStackFlattenLinearNet(nn.Module):
    def __init__(self, T: int, in_features: int):
        super().__init__()
        self.T = T
        self.weight = nn.Parameter(torch.randn(7, in_features) * 0.1)
        self.bias = nn.Parameter(torch.randn(7) * 0.1)

    def forward(self, x_seq, v_init):
        stack = temporal_lif_stack(x_seq, v_init)
        outs = []
        for t in range(self.T):
            x = torch.flatten(stack[t], 1)
            outs.append(F.linear(x, self.weight, self.bias))
        return torch.stack(outs, dim=0)


class PreviousChunkChainNet(nn.Module):
    def __init__(self, T: int, h: int, w: int):
        super().__init__()
        self.T = T
        self.conv_weight = nn.Parameter(torch.randn(4, 3, 3, 3) * 0.1)
        self.conv_bias = nn.Parameter(torch.randn(4) * 0.1)
        self.bn_weight = nn.Parameter(torch.randn(4) * 0.1 + 1.0)
        self.bn_bias = nn.Parameter(torch.randn(4) * 0.1)
        self.register_buffer("running_mean", torch.randn(4) * 0.1)
        self.register_buffer("running_var", torch.rand(4) + 0.5)
        self.linear_weight = nn.Parameter(torch.randn(6, 4 * h * w) * 0.1)
        self.linear_bias = nn.Parameter(torch.randn(6) * 0.1)

    def forward(self, x_seq, v_init):
        stack = temporal_lif_stack(x_seq, v_init)
        outs = []
        for t in range(self.T):
            x = F.conv2d(stack[t], self.conv_weight, self.conv_bias, stride=1, padding=1)
            x = F.batch_norm(x, self.running_mean, self.running_var, self.bn_weight, self.bn_bias, False, 0.1, 1e-5)
            x = torch.flatten(x, 1)
            x = F.linear(x, self.linear_weight, self.linear_bias)
            outs.append(x)
        return torch.stack(outs, dim=0)


def _count_nodes(gm: torch.fx.GraphModule):
    counts = {"flatten": 0, "cat": 0, "chunk": 0, "getitem": 0, "conv": 0, "bn": 0, "linear": 0}
    for node in gm.graph.nodes:
        text = str(node.target)
        if node.op == "call_function" and (node.target is torch.flatten or "flatten" in text):
            counts["flatten"] += 1
        if node.op == "call_function" and node.target is torch.cat:
            counts["cat"] += 1
        if node.op == "call_function" and node.target is torch.chunk:
            counts["chunk"] += 1
        if node.op == "call_function" and node.target is operator.getitem:
            counts["getitem"] += 1
        if node.op == "call_function" and "conv2d" in text:
            counts["conv"] += 1
        if node.op == "call_function" and "batch_norm" in text:
            counts["bn"] += 1
        if node.op == "call_function" and "linear" in text:
            counts["linear"] += 1
    return counts


def _run_model_case(name, model, inputs, enabled_ops, dtype, out_dir):
    gm = torch.fx.symbolic_trace(copy.deepcopy(model).eval())
    gm = gm.to(device=inputs[0].device, dtype=dtype)
    ref_model = copy.deepcopy(model).eval()
    with torch.no_grad():
        ref = ref_model(*inputs)

    stats = apply_spatial_batching(
        gm,
        temporal_window=model.T,
        enabled_ops=enabled_ops,
        dump_dir=out_dir / name if out_dir is not None else None,
        strict=True,
        enable_chain=False,
    )
    canonicalize_temporal_spatial_ir(gm, strict=True)
    gm.graph.lint()
    gm.recompile()

    with torch.no_grad():
        out = gm(*inputs)

    tol = 1e-2 if dtype == torch.float16 else 1e-5
    allclose = torch.allclose(out, ref, rtol=tol, atol=tol)
    counts = _count_nodes(gm)
    print(f"{name}: stats={stats}")
    print(f"{name}: counts={counts}")
    print(f"{name}: allclose={allclose} max_abs={(out - ref).abs().max().item():.3e}")
    assert allclose
    assert stats.spatial_batched_ops > 0
    assert counts["cat"] == 0
    return stats, counts


def run_case(args):
    torch.manual_seed(2026)
    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    out_dir = Path(args.out_dir) if args.out_dir else None
    x_seq = torch.randn(args.T, args.batch_size, 3, args.height, args.width, device=device, dtype=dtype)
    v_init = torch.zeros(args.batch_size, 3, args.height, args.width, device=device, dtype=dtype)
    y_seq = torch.randn_like(x_seq)
    y_v_init = torch.zeros_like(v_init)

    stats, counts = _run_model_case("conv", TemporalStackConvNet(args.T).to(device=device, dtype=dtype), (x_seq, v_init), ["conv"], dtype, out_dir)
    assert stats.spatial_temporal_stack_groups >= 1 and stats.spatial_batched_conv >= args.T and counts["conv"] == 1

    stats, counts = _run_model_case("bn", TemporalStackBNNet(args.T).to(device=device, dtype=dtype), (x_seq, v_init), ["bn"], dtype, out_dir)
    assert stats.spatial_temporal_stack_bn_groups >= 1 and stats.spatial_batched_bn >= args.T and counts["bn"] == 1

    stats, _counts = _run_model_case("maxpool", TemporalStackPoolNet(args.T, adaptive=False).to(device=device, dtype=dtype), (x_seq, v_init), ["maxpool"], dtype, out_dir)
    assert stats.spatial_temporal_stack_pool_groups >= 1 and stats.spatial_batched_maxpool >= args.T

    stats, _counts = _run_model_case("adaptive_avgpool", TemporalStackPoolNet(args.T, adaptive=True).to(device=device, dtype=dtype), (x_seq, v_init), ["avgpool"], dtype, out_dir)
    assert stats.spatial_temporal_stack_pool_groups >= 1 and stats.spatial_batched_adaptive_avgpool >= args.T

    stats, _counts = _run_model_case("add", TemporalStackAddNet(args.T).to(device=device, dtype=dtype), (x_seq, v_init, y_seq, y_v_init), ["add"], dtype, out_dir)
    assert stats.spatial_temporal_stack_add_groups >= 1 and stats.spatial_batched_add >= args.T

    in_features = 3 * args.height * args.width
    stats, counts = _run_model_case(
        "flatten_linear",
        TemporalStackFlattenLinearNet(args.T, in_features).to(device=device, dtype=dtype),
        (x_seq, v_init),
        ["flatten", "linear"],
        dtype,
        out_dir,
    )
    assert stats.spatial_temporal_stack_flatten_groups >= 1
    assert stats.spatial_previous_batched_groups >= 1
    assert stats.spatial_batched_flatten >= args.T and stats.spatial_batched_linear >= args.T
    assert counts["linear"] == 1

    stats, counts = _run_model_case(
        "previous_chunk_chain",
        PreviousChunkChainNet(args.T, args.height, args.width).to(device=device, dtype=dtype),
        (x_seq, v_init),
        ["conv", "bn", "flatten", "linear"],
        dtype,
        out_dir,
    )
    assert stats.spatial_temporal_stack_groups >= 1
    assert stats.spatial_previous_batched_groups >= 3
    assert stats.spatial_chunk_cat_avoided >= 3
    assert stats.spatial_batched_conv >= args.T
    assert stats.spatial_batched_bn >= args.T
    assert stats.spatial_batched_flatten >= args.T
    assert stats.spatial_batched_linear >= args.T
    assert counts["conv"] == 1 and counts["bn"] == 1 and counts["linear"] == 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--T", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable; skipping")
        return
    run_case(args)


if __name__ == "__main__":
    main()
