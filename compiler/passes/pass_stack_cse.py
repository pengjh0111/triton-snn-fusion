"""Pass B: common-subexpression-elimination for torch.stack([x, x, ..., x],
dim) nodes whose entire input list is one node repeated -- e.g. the model
input being stacked T times to feed the first fused batched Linear+LIF call.
When the same (source node, list length, dim) triple shows up as more than
one distinct stack node in the graph, only the first is kept and the rest
are redirected to it.

Toggle: env var CHRONOS_PASS_STACK_CSE (default: on; set to "0"/"false" to
disable -- see compiler/passes/registry.py).

Verification level: bit-exact (this is a pure dedup of identical
computations; the kept node produces exactly the tensor the removed ones
would have).

Scope note -- this is first-tier CSE only. It does NOT attempt the second,
higher-value tier: replacing stack([x]*T, 0) with a zero-copy expand()
broadcast view (stride 0 along the new leading dim) and teaching the
batched_linear_lif kernel to accept that non-contiguous input directly
(stride_t=0 support in the Triton kernel's index arithmetic). That is a
kernel-level change and explicitly out of scope here.

Before attempting that second tier, verify: does the runtime wrapper
re-materialize an expand() view via .contiguous() before it ever reaches the
kernel? As of this writing,
kernels/benchmark_batched_linear_lif_temporal_general.py's
run_fused_batched_linear_lif calls `x_seq = x_seq.contiguous()` at the top,
which would immediately copy an expand() view back into a full [T, ...]
materialized tensor -- i.e. the second tier has *no* memory-traffic benefit
until that contiguous() call is either removed or made conditional on the
input already being expand-broadcastable, and the kernel's own address
arithmetic is extended to support a stride_t=0 X load. Do not implement the
second tier without first re-checking that wrapper.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch


@dataclass
class StackCsePassStats:
    stack_nodes_seen: int = 0
    duplicate_input_stacks: int = 0
    groups: int = 0
    merged: int = 0


def _is_stack(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and node.target is torch.stack


def _stack_tensors_and_dim(node: torch.fx.Node):
    tensors = node.args[0] if node.args else node.kwargs.get("tensors")
    if len(node.args) >= 2:
        dim = node.args[1]
    else:
        dim = node.kwargs.get("dim", 0)
    return tensors, dim


def apply_stack_cse_pass(gm: torch.fx.GraphModule) -> StackCsePassStats:
    stats = StackCsePassStats()
    graph = gm.graph

    groups: Dict[Tuple[torch.fx.Node, int, object], List[torch.fx.Node]] = {}
    for node in graph.nodes:
        if not _is_stack(node):
            continue
        stats.stack_nodes_seen += 1
        tensors, dim = _stack_tensors_and_dim(node)
        if not isinstance(tensors, (list, tuple)) or len(tensors) == 0:
            continue
        first = tensors[0]
        if not isinstance(first, torch.fx.Node):
            continue
        if not all(t is first for t in tensors):
            continue
        stats.duplicate_input_stacks += 1
        key = (first, len(tensors), dim)
        groups.setdefault(key, []).append(node)

    for nodes in groups.values():
        if len(nodes) < 2:
            continue
        stats.groups += 1
        keeper = nodes[0]
        for dup in nodes[1:]:
            dup.replace_all_uses_with(keeper)
            stats.merged += 1

    graph.eliminate_dead_code()
    graph.lint()
    gm.recompile()
    return stats
