"""Pass C: replace the "add-then-zeros_like" anti-pattern used to build a
throwaway shaped-zero v_init argument for fused_temporal_add_lif_state calls
with the scalar new_zeros(()) idiom the fuse rewrite already uses elsewhere
in the same graph for the batched Linear+LIF fused ops' v_init.

Matched pattern:
    add_n  = spike_a + spike_b
    zl_n   = torch.zeros_like(add_n)
    result = torch.ops.snn_custom.fused_temporal_add_lif_state.default(
        spike_a_stack, spike_b_stack, zl_n, ...)

Replacement: zl_n -> spike_a_stack.new_zeros(()) (a scalar tensor). add_n and
zl_n lose their only consumer and are removed by dead-code elimination.

Toggle: env var KAIROS_PASS_VINIT_CLEANUP (default: on; set to "0"/"false"
to disable -- see compiler/passes/registry.py).

Verification level: bit-exact. `fused_temporal_add_lif_state_torch` in
runtime/snn_custom_ops.py already special-cases a scalar v_init
(`v_init.dim() == 0`) as "start from `torch.zeros_like(lhs_seq[0])`" -- which
is precisely the value `zeros_like(spike_a_t0 + spike_b_t0)` computes today,
since zeros_like only reads shape/dtype/device off its input and the
computed sum's *value* is discarded entirely. The Triton kernel path
(kernels/generated_temporal_transformer_lif_kernels.py /
kernels/generated_temporal_batched_linear_lif_kernel.py) takes this same
V_INIT_IS_SCALAR branch for every other batched Linear+LIF call in this
graph already (see the `new_zeros(())` calls already present in the rewrite
output), so this pass only removes a redundant, wasted `add` -- it does not
exercise any new code path.
"""

import operator
from dataclasses import dataclass, field
from typing import Dict

import torch


def _add_lif_target():
    # Resolved lazily (not at module import time): the snn_custom op library
    # is only registered once runtime/snn_custom_ops.py has been imported
    # somewhere in the process, which may happen after this module is first
    # imported.
    return torch.ops.snn_custom.fused_temporal_add_lif_state.default


@dataclass
class VInitCleanupPassStats:
    candidates: int = 0
    replaced: int = 0
    skipped: int = 0
    skip_reasons: Dict[str, int] = field(default_factory=dict)

    def skip(self, reason: str):
        self.skipped += 1
        self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + 1


def _is_zeros_like(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and node.target is torch.zeros_like


def _is_add(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and node.target is operator.add


def apply_v_init_cleanup_pass(gm: torch.fx.GraphModule) -> VInitCleanupPassStats:
    stats = VInitCleanupPassStats()
    graph = gm.graph
    add_lif_target = _add_lif_target()

    for node in list(graph.nodes):
        if node.op != "call_function" or node.target is not add_lif_target:
            continue
        if len(node.args) < 3:
            continue
        lhs_seq, _rhs_seq, v_init = node.args[0], node.args[1], node.args[2]
        stats.candidates += 1

        if not isinstance(v_init, torch.fx.Node) or not _is_zeros_like(v_init):
            stats.skip("v_init_not_zeros_like")
            continue
        if any(k in v_init.kwargs for k in ("dtype", "layout", "device", "memory_format")):
            stats.skip("zeros_like_has_overrides")
            continue
        if len(v_init.users) != 1:
            stats.skip("zeros_like_extra_users")
            continue

        add_n = v_init.args[0] if v_init.args else v_init.kwargs.get("input")
        if not isinstance(add_n, torch.fx.Node) or not _is_add(add_n):
            stats.skip("zeros_like_input_not_add")
            continue
        if len(add_n.users) != 1:
            stats.skip("add_has_extra_users")
            continue
        if len(add_n.args) != 2 or not all(isinstance(a, torch.fx.Node) for a in add_n.args):
            stats.skip("add_operands_not_nodes")
            continue

        with graph.inserting_before(node):
            scalar_zero = graph.call_method("new_zeros", args=(lhs_seq, ()))

        new_args = list(node.args)
        new_args[2] = scalar_zero
        node.args = tuple(new_args)
        stats.replaced += 1

    graph.eliminate_dead_code()
    graph.lint()
    gm.recompile()
    return stats
