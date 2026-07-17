"""Pass D: batch the classifier head's per-timestep tail
(layer_norm -> mean(dim=-2) -> linear, accumulated by a chain of adds) into
one call over the whole spike stack plus a single sum(dim=0).

Matched pattern, per timestep t = 0..T-1, all reading from the same source
spike-stack node via operator.getitem(stack, t):
    x_t   = layer_norm(spike_t, normalized_shape, w, b, eps)   # same w/b/eps for every t
    m_t   = x_t.mean(dim=-2)
    l_t   = linear(m_t, W_cls, bias)                           # same W_cls/bias for every t
    acc_t = acc_{t-1} + l_t     (acc_{-1} := whatever the first add's other operand is)

Replacement:
    x_all = layer_norm(spike_stack, normalized_shape, w, b, eps)   # [T, ..., F] at once
    m_all = x_all.mean(dim=-2)                                     # [T, ..., F]
    l_all = linear(m_all, W_cls, bias)                             # [T, ..., C]
    acc   = base + l_all.sum(dim=0)                                # base = acc_{-1} above

Toggle: env var CHRONOS_PASS_CLASSIFIER_BATCH (default: on; set to
"0"/"false" to disable -- see compiler/passes/registry.py).

Verification level: tolerance, not bit-exact. F.layer_norm normalizes over
the trailing `normalized_shape` dims only, `.mean(dim=-2)` reduces one
non-trailing dim, and F.linear operates independently per leading-dim row --
all three functions are literally the same computation evaluated at each
leading-dim (here: timestep) index, whether invoked T separate times with
input shape [...] or once with input shape [T, ...] (an extra, independent
leading dim does not change what any of the three ops compute per row). The
only numerical difference is float summation order: the original graph
accumulates as a strictly sequential chain
`(((base + l_0) + l_1) + l_2) + l_3`, whereas `l_all.sum(dim=0)` performs
its own (typically pairwise/tree) reduction internally -- floating-point
addition is not associative, so results can differ by a few ULPs. Hence
tolerance-level verification (final logits within rtol/atol, argmax
unchanged), not bit-exact.
"""

import operator
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch


@dataclass
class ClassifierBatchPassStats:
    stack_groups_examined: int = 0
    groups_found: int = 0
    groups_replaced: int = 0
    skipped: int = 0
    skip_reasons: Dict[str, int] = field(default_factory=dict)

    def skip(self, reason: str):
        self.skipped += 1
        self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + 1


def _is_getitem(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and node.target is operator.getitem


def _is_layer_norm(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and node.target is torch.nn.functional.layer_norm


def _is_mean(node: torch.fx.Node) -> bool:
    return node.op == "call_method" and node.target == "mean"


def _is_linear(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and node.target in (
        torch.nn.functional.linear,
        torch._C._nn.linear,
    )


def _is_add(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and node.target is operator.add


def _single_user(node: torch.fx.Node) -> bool:
    return len(node.users) == 1


def _find_stack_getitem_groups(gm: torch.fx.GraphModule):
    groups: Dict[torch.fx.Node, Dict[int, torch.fx.Node]] = {}
    for node in gm.graph.nodes:
        if not _is_getitem(node):
            continue
        if len(node.args) != 2:
            continue
        source, index = node.args
        if not isinstance(source, torch.fx.Node) or not isinstance(index, int):
            continue
        groups.setdefault(source, {})[index] = node
    return groups


def _match_chain_from_getitem(t_node: torch.fx.Node, stats: ClassifierBatchPassStats) -> Optional[dict]:
    if not _single_user(t_node):
        stats.skip("getitem_extra_users")
        return None
    ln_node = next(iter(t_node.users))
    if not _is_layer_norm(ln_node) or not _single_user(ln_node):
        return None
    if len(ln_node.args) < 2:
        return None
    normalized_shape = ln_node.args[1]
    weight = ln_node.args[2] if len(ln_node.args) > 2 else ln_node.kwargs.get("weight")
    bias = ln_node.args[3] if len(ln_node.args) > 3 else ln_node.kwargs.get("bias")
    eps = ln_node.args[4] if len(ln_node.args) > 4 else ln_node.kwargs.get("eps", 1e-05)

    mean_node = next(iter(ln_node.users))
    if not _is_mean(mean_node) or not _single_user(mean_node):
        return None
    mean_dim = mean_node.kwargs.get("dim", mean_node.args[1] if len(mean_node.args) > 1 else None)
    if mean_dim != -2:
        return None

    linear_node = next(iter(mean_node.users))
    if not _is_linear(linear_node) or not _single_user(linear_node):
        return None
    if len(linear_node.args) < 2:
        return None
    lin_weight = linear_node.args[1]
    lin_bias = linear_node.args[2] if len(linear_node.args) > 2 else linear_node.kwargs.get("bias")

    add_node = next(iter(linear_node.users))
    if not _is_add(add_node):
        return None

    return {
        "ln": ln_node,
        "mean": mean_node,
        "linear": linear_node,
        "add": add_node,
        "normalized_shape": normalized_shape,
        "ln_weight": weight,
        "ln_bias": bias,
        "eps": eps,
        "lin_weight": lin_weight,
        "lin_bias": lin_bias,
    }


def _try_match_group(index_to_getitem: Dict[int, torch.fx.Node], stats: ClassifierBatchPassStats) -> Optional[dict]:
    T = len(index_to_getitem)
    if T < 2:
        return None
    if set(index_to_getitem.keys()) != set(range(T)):
        stats.skip("indices_not_0_to_T-1")
        return None

    chains = []
    for i in range(T):
        info = _match_chain_from_getitem(index_to_getitem[i], stats)
        if info is None:
            stats.skip("chain_shape_mismatch")
            return None
        chains.append(info)

    ref = chains[0]
    for info in chains[1:]:
        if info["normalized_shape"] != ref["normalized_shape"]:
            stats.skip("normalized_shape_mismatch")
            return None
        if info["ln_weight"] is not ref["ln_weight"] or info["ln_bias"] is not ref["ln_bias"]:
            stats.skip("layer_norm_params_mismatch")
            return None
        if info["eps"] != ref["eps"]:
            stats.skip("eps_mismatch")
            return None
        if info["lin_weight"] is not ref["lin_weight"] or info["lin_bias"] is not ref["lin_bias"]:
            stats.skip("linear_params_mismatch")
            return None

    adds = [c["add"] for c in chains]
    linears = [c["linear"] for c in chains]
    base = None
    prev_add = None
    for i in range(T):
        add_node = adds[i]
        if len(add_node.args) != 2:
            stats.skip("add_arity")
            return None
        a, b = add_node.args
        if i == 0:
            if b is linears[i]:
                base = a
            elif a is linears[i]:
                base = b
            else:
                stats.skip("add0_operand_mismatch")
                return None
        else:
            ok = (a is prev_add and b is linears[i]) or (b is prev_add and a is linears[i])
            if not ok:
                stats.skip("addN_operand_mismatch")
                return None
        if i < T - 1 and len(add_node.users) != 1:
            stats.skip("intermediate_add_extra_users")
            return None
        prev_add = add_node

    return {"chains": chains, "base": base, "final_add": adds[-1], "ref": ref, "T": T}


def apply_classifier_batch_pass(gm: torch.fx.GraphModule) -> ClassifierBatchPassStats:
    stats = ClassifierBatchPassStats()
    graph = gm.graph

    groups = _find_stack_getitem_groups(gm)
    for stack_node, index_map in groups.items():
        stats.stack_groups_examined += 1
        match = _try_match_group(index_map, stats)
        if match is None:
            continue
        stats.groups_found += 1

        ref = match["ref"]
        final_add = match["final_add"]

        with graph.inserting_before(final_add):
            ln_all = graph.call_function(
                ref["ln"].target,
                args=(stack_node, ref["normalized_shape"], ref["ln_weight"], ref["ln_bias"], ref["eps"]),
            )
            mean_all = graph.call_method("mean", args=(ln_all,), kwargs={"dim": -2})
            linear_all = graph.call_function(
                ref["linear"].target,
                args=(mean_all, ref["lin_weight"], ref["lin_bias"]),
            )
            summed = graph.call_method("sum", args=(linear_all,), kwargs={"dim": 0})
            folded = summed if match["base"] is None else graph.call_function(operator.add, args=(match["base"], summed))

        final_add.replace_all_uses_with(folded)
        stats.groups_replaced += 1

    graph.eliminate_dead_code()
    graph.lint()
    gm.recompile()
    return stats
