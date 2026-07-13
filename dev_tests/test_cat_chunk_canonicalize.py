import argparse
import operator
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

from compiler.fx_temporal_spatial_canonicalize import canonicalize_temporal_spatial_ir


class ChunkCatNet(nn.Module):
    def forward(self, x):
        chunks = torch.chunk(x, 4, dim=0)
        return torch.cat([chunks[0], chunks[1], chunks[2], chunks[3]], dim=0)


class CatChunkNet(nn.Module):
    def forward(self, a, b, c, d):
        y = torch.cat([a, b, c, d], dim=0)
        chunks = torch.chunk(y, 4, dim=0)
        return chunks[0] + chunks[1] + chunks[2] + chunks[3]


def count_ops(gm):
    counts = {"cat": 0, "chunk": 0, "getitem": 0}
    for node in gm.graph.nodes:
        if node.op == "call_function" and (node.target is torch.cat or "cat" in str(node.target)):
            counts["cat"] += 1
        if node.op == "call_function" and (node.target is torch.chunk or "chunk" in str(node.target)):
            counts["chunk"] += 1
        if node.op == "call_function" and node.target is operator.getitem:
            counts["getitem"] += 1
    return counts


def run_backend_case(model, inputs, expect_removed: str):
    captured = {}

    def backend(gm, example_inputs):
        before = count_ops(gm)
        stats = canonicalize_temporal_spatial_ir(gm, strict=True)
        after = count_ops(gm)
        captured["before"] = before
        captured["after"] = after
        captured["stats"] = stats
        gm.graph.lint()
        gm.recompile()
        return gm.forward

    compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    with torch.no_grad():
        ref = model(*inputs)
        got = compiled(*inputs)
    if not torch.allclose(ref, got):
        raise AssertionError("canonicalized output differs from eager")
    stats = captured["stats"]
    print(f"{model.__class__.__name__}: before={captured['before']} after={captured['after']} stats={stats}")
    if expect_removed == "chunk_cat" and stats.canonicalize_chunk_cat_removed < 1:
        raise AssertionError("expected chunk->cat canonicalization")
    if expect_removed == "cat_chunk" and stats.canonicalize_cat_chunk_removed < 1:
        raise AssertionError("expected cat->chunk canonicalization")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch._dynamo.reset()
    x = torch.randn(8, 3, 4, 4, device=args.device)
    run_backend_case(ChunkCatNet().to(args.device).eval(), (x,), "chunk_cat")
    xs = tuple(torch.randn(2, 3, 4, 4, device=args.device) for _ in range(4))
    run_backend_case(CatChunkNet().to(args.device).eval(), xs, "cat_chunk")


if __name__ == "__main__":
    main()
