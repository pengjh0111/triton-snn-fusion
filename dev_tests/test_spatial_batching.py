import argparse
import operator
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx.passes.shape_prop import ShapeProp

import runtime.snn_custom_ops  # noqa: F401 - ensure custom op registration
from compiler.fx_spatial_batching import apply_spatial_batching
from compiler.fx_spatial_batching import _candidate_kind
from compiler.fx_temporal_annotation import annotate_temporal_metadata


class TemporalMaxPoolNet(nn.Module):
    def __init__(self, T: int):
        super().__init__()
        self.T = T

    def forward(self, x):
        inputs = []
        for t in range(self.T):
            inputs.append(x + float(t))
        out = 0
        for value in inputs:
            y = F.max_pool2d(value, 2, 2)
            out = out + y
        return out


class TemporalLinearNet(nn.Module):
    def __init__(self, T: int, features: int):
        super().__init__()
        self.T = T
        self.fc = nn.Linear(features, features)

    def forward(self, x):
        inputs = []
        for t in range(self.T):
            inputs.append(x + float(t))
        out = 0
        for value in inputs:
            y = self.fc(value)
            out = out + y
        return out


class TemporalMaxPoolChainNet(nn.Module):
    def __init__(self, T: int):
        super().__init__()
        self.T = T

    def forward(self, x):
        inputs = []
        for t in range(self.T):
            inputs.append(x + float(t))
        out = 0
        for value in inputs:
            y = F.max_pool2d(value, 2, 2)
            z = F.max_pool2d(y, 2, 2)
            out = out + z
        return out


class TemporalAvgPoolFlattenLinearNet(nn.Module):
    def __init__(self, T: int, channels: int):
        super().__init__()
        self.T = T
        self.fc = nn.Linear(channels, channels)

    def forward(self, x):
        inputs = []
        for t in range(self.T):
            inputs.append(x + float(t))
        out = 0
        for value in inputs:
            y = F.adaptive_avg_pool2d(value, (1, 1))
            y = torch.flatten(y, 1)
            y = self.fc(y)
            out = out + y
        return out


class ConvLIFPriorityNet(nn.Module):
    def __init__(self, T: int):
        super().__init__()
        self.T = T
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.register_buffer("v", torch.tensor(0.0), persistent=False)

    def forward(self, x):
        v = self.v
        spikes = []
        for _ in range(self.T):
            y = self.conv(x)
            if v.dim() == 0 or v.shape != y.shape or v.device != y.device or v.dtype != y.dtype:
                v = torch.zeros_like(y)
            spike, v = torch.ops.snn_custom.lif_forward_state.default(y, v, 1.0, 0.0, 2.0, False)
            spikes.append(spike)
        out = 0
        for spike in spikes:
            out = out + self.pool(spike)
        self.v = v
        return out


def _is_candidate_node(node: torch.fx.Node, op_names) -> bool:
    return _candidate_kind(None, node, op_names) is not None


def _tag_timesteps_for_test(gm: torch.fx.GraphModule, op_names, T: int):
    candidates = [node for node in gm.graph.nodes if _candidate_kind(None, node, op_names) is not None]
    if candidates and len(candidates) % T == 0:
        per_timestep = len(candidates) // T
        for index, node in enumerate(candidates):
            setattr(node, "_chronos_timestep", index // per_timestep)
        return len(candidates)

    count_by_kind = {}
    for node in gm.graph.nodes:
        kind = _candidate_kind(None, node, op_names)
        if kind is None:
            continue
        count = count_by_kind.get(kind, 0)
        setattr(node, "_chronos_timestep", count % T)
        count_by_kind[kind] = count + 1
    return sum(count_by_kind.values())


def _count_graph_ops(gm: torch.fx.GraphModule):
    cat_count = 0
    chunk_count = 0
    getitem_count = 0
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target is torch.cat:
            cat_count += 1
        if node.op == "call_function" and node.target is torch.chunk:
            chunk_count += 1
        if node.op == "call_function" and node.target is operator.getitem:
            getitem_count += 1
    return {"cat": cat_count, "chunk": chunk_count, "getitem": getitem_count}


def _make_backend(T: int, op_names, out_dir: Path, enable_chain: bool = True):
    stats = {}

    def backend(gm: torch.fx.GraphModule, example_inputs):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "original_fx.py").write_text(gm.code, encoding="utf-8")
        tagged = _tag_timesteps_for_test(gm, op_names, T)
        annotation_stats = annotate_temporal_metadata(gm, T, T, strict=True)
        ShapeProp(gm).propagate(*example_inputs)
        batching_stats = apply_spatial_batching(
            gm,
            T,
            op_names,
            dump_dir=out_dir,
            strict=True,
            enable_chain=enable_chain,
        )
        (out_dir / "rewritten_fx.py").write_text(gm.code, encoding="utf-8")
        stats["tagged"] = tagged
        stats["annotated"] = annotation_stats.temporal_annotated_nodes
        stats["annotation_missing"] = annotation_stats.temporal_annotation_missing
        stats["batched"] = batching_stats.spatial_batched_ops
        stats["chains"] = batching_stats.spatial_batch_chains
        stats["chain_groups"] = batching_stats.spatial_chain_groups
        stats["cat_eliminated"] = batching_stats.spatial_cat_eliminated
        stats["chunk_eliminated"] = batching_stats.spatial_chunk_eliminated
        stats["graph_ops"] = _count_graph_ops(gm)
        return gm.forward

    return backend, stats


def run_case(
    name: str,
    model: nn.Module,
    x: torch.Tensor,
    T: int,
    op_names,
    out_dir: Path,
    rtol: float,
    atol: float,
    enable_chain: bool = True,
    expect_chain: bool = False,
):
    backend, stats = _make_backend(T, op_names, out_dir / name, enable_chain=enable_chain)
    compiled = torch.compile(model, backend=backend, fullgraph=False, dynamic=False)
    with torch.no_grad():
        if hasattr(model, "v"):
            model.v = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        eager = model(x)
        if hasattr(model, "v"):
            model.v = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        compiled_out = compiled(x)
    diff = (eager - compiled_out).abs()
    allclose = torch.allclose(eager, compiled_out, rtol=rtol, atol=atol)
    print(
        f"[SPATIAL_BATCHING_TEST] {name}: allclose={allclose} "
        f"max={diff.max().item():.6e} mean={diff.mean().item():.6e} stats={stats}"
    )
    if not allclose:
        raise AssertionError(f"{name} output mismatch")
    if stats.get("batched", 0) < T:
        raise AssertionError(f"{name} did not batch expected ops: {stats}")
    if expect_chain and stats.get("chains", 0) < 1:
        raise AssertionError(f"{name} did not form expected chain: {stats}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Minimal correctness tests for Chronos FX spatial batching.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--T", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--out-dir", default="/tmp/chronos_spatial_batching_unit")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")

    device = args.device
    out_dir = Path(args.out_dir)

    try:
        torch.manual_seed(2026)
        maxpool_model = TemporalMaxPoolNet(args.T).to(device).eval()
        maxpool_x = torch.randn(args.batch_size, 3, 16, 16, device=device)
        run_case("maxpool", maxpool_model, maxpool_x, args.T, ["maxpool"], out_dir, args.rtol, args.atol)

        linear_model = TemporalLinearNet(args.T, 8).to(device).eval()
        linear_x = torch.randn(args.batch_size, 8, device=device)
        run_case("linear", linear_model, linear_x, args.T, ["linear"], out_dir, args.rtol, args.atol)

        maxpool_chain_model = TemporalMaxPoolChainNet(args.T).to(device).eval()
        maxpool_chain_x = torch.randn(args.batch_size, 3, 32, 32, device=device)
        chain_stats = run_case(
            "maxpool_chain",
            maxpool_chain_model,
            maxpool_chain_x,
            args.T,
            ["maxpool"],
            out_dir,
            args.rtol,
            args.atol,
            expect_chain=True,
        )
        if chain_stats["graph_ops"]["cat"] >= 2 or chain_stats["graph_ops"]["chunk"] >= 2:
            raise AssertionError(f"maxpool_chain did not eliminate intermediate cat/chunk: {chain_stats}")

        avg_flat_linear_model = TemporalAvgPoolFlattenLinearNet(args.T, 3).to(device).eval()
        avg_flat_linear_x = torch.randn(args.batch_size, 3, 8, 8, device=device)
        afl_stats = run_case(
            "avgpool_flatten_linear_chain",
            avg_flat_linear_model,
            avg_flat_linear_x,
            args.T,
            ["avgpool", "flatten", "linear"],
            out_dir,
            args.rtol,
            args.atol,
            expect_chain=True,
        )
        if afl_stats["chains"] < 1 or afl_stats["chain_groups"] < 3:
            raise AssertionError(f"avgpool_flatten_linear_chain did not build 3-stage chain: {afl_stats}")
        if afl_stats["graph_ops"]["cat"] >= 3 or afl_stats["graph_ops"]["chunk"] >= 3:
            raise AssertionError(f"avgpool_flatten_linear_chain did not reduce cat/chunk count: {afl_stats}")

        priority_model = ConvLIFPriorityNet(args.T).to(device).eval()
        priority_x = torch.randn(args.batch_size, 3, 16, 16, device=device)
        run_case("conv_lif_priority_pool", priority_model, priority_x, args.T, ["maxpool"], out_dir, args.rtol, args.atol)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
