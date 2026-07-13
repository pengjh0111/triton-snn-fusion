import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from compiler.fx_spatial_batching import apply_spatial_batching
from compiler.fx_temporal_spatial_canonicalize import canonicalize_temporal_spatial_ir


class PerOpSpatialNet(nn.Module):
    def __init__(self, kind: str, T: int):
        super().__init__()
        self.kind = kind
        self.T = T
        self.conv = nn.Conv2d(3, 3, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(3)
        self.fc = nn.Linear(3 * 8 * 8, 7)
        self.register_buffer("residual", torch.randn(1, 3, 8, 8) * 0.01)

    def _op(self, x):
        if self.kind == "conv":
            return self.conv(x)
        if self.kind == "bn":
            return self.bn(x)
        if self.kind == "add":
            return x + self.residual
        if self.kind == "maxpool":
            return F.max_pool2d(x, 1, 1)
        if self.kind == "avgpool":
            return F.adaptive_avg_pool2d(x, (8, 8))
        if self.kind == "flatten":
            return torch.flatten(x, 1)
        if self.kind == "linear":
            return self.fc(x)
        if self.kind == "elementwise":
            return F.relu(x)
        raise ValueError(self.kind)

    def forward(self, x):
        out = 0
        for _ in range(self.T):
            y = self._op(x)
            out = out + y.sum()
        return out


def run_case(kind: str, args):
    torch._dynamo.reset()
    model = PerOpSpatialNet(kind, args.T).to(args.device).eval()
    if kind == "linear":
        x = torch.randn(args.batch_size, 3 * 8 * 8, device=args.device)
    else:
        x = torch.randn(args.batch_size, 3, 8, 8, device=args.device)
    captured = {}

    def backend(gm, example_inputs):
        timestep = 0
        for node in gm.graph.nodes:
            text = str(node.target)
            meta = node.meta.get("tensor_meta") or node.meta.get("val")
            shape = getattr(meta, "shape", None)
            if shape is None and isinstance(meta, torch.Tensor):
                shape = meta.shape
            tensor_rank = len(shape) if shape is not None else None
            is_kind = node.name.startswith("y")
            if is_kind:
                node.meta["chronos_timestep"] = timestep
                node.meta["chronos_window_id"] = 0
                node.meta["chronos_occurrence"] = 0
                timestep += 1
        stats = apply_spatial_batching(
            gm,
            args.T,
            [kind],
            strict=True,
            enable_chain=False,
        )
        canon = canonicalize_temporal_spatial_ir(gm, strict=True)
        captured["stats"] = stats
        captured["canon"] = canon
        gm.graph.lint()
        gm.recompile()
        return gm.forward

    compiled = torch.compile(model, backend=backend, fullgraph=False, dynamic=False)
    with torch.no_grad():
        ref = model(x)
        got = compiled(x)
    allclose = torch.allclose(ref, got, rtol=1e-4, atol=1e-4)
    stats = captured.get("stats")
    print(f"kind={kind} batched_ops={stats.spatial_batched_ops if stats else None} allclose={allclose}")
    if not allclose:
        raise AssertionError(f"{kind} output differs")
    if stats is None or stats.spatial_batched_ops < args.T:
        raise AssertionError(f"{kind} did not batch per timestep op: {stats}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--T", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--ops",
        nargs="+",
        default=["conv", "bn", "add", "maxpool", "avgpool", "flatten", "linear", "elementwise"],
    )
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    for kind in args.ops:
        run_case(kind, args)


if __name__ == "__main__":
    main()
