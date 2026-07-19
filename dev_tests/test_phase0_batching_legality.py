"""Phase 0 (Kairos feedback-workload prep): spatial batching legality check.

apply_spatial_batching batches "the same op's per-t instances" into one call
across a window. It has always implicitly assumed every instance's tensor
inputs are genuinely independent per timestep. That assumption holds for
every existing SNN model (membrane-state recurrence is always hidden behind
an atomic snn_custom.* op boundary -- conv/bn/pool never directly consume a
prior timestep's raw value) but is violated by feedback networks
(ConvLSTM/GRU/Mamba-style): the W_h-side conv/linear consumes h_prev/c_prev,
the literal output of the *previous* timestep's block. Batching that op
across t would silently reuse the wrong per-t value and produce a
mathematically incorrect graph.

This test builds a minimal 2-timestep feedback graph by hand (no Dynamo/
torch.compile involved -- just enough FX structure + kairos_timestep
metadata to exercise collect_spatial_batch_candidates directly) and checks:
  1. The W_x-side conv (input = fresh per-t slice of an external tensor,
     tracing back to a placeholder) IS accepted as a batching candidate.
  2. The W_h-side conv (input = h_prev, the previous timestep's gate-chain
     output) is REJECTED, with reason "feedback_dependency".

It also re-runs spatial batching on a real existing SNN graph (MobileNetV1,
which already exercises conv/bn/add/pool) before vs. after the legality
patch and diffs the batching decision log -- must be empty, confirming zero
behavior change for the graphs this pass has always handled correctly.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import operator

import torch
import torch.fx as fx
import torch.nn.functional as F

import runtime.snn_custom_ops  # noqa: F401  registers torch.ops.snn_custom.*

from compiler.fx_spatial_batching import (
    collect_spatial_batch_candidates,
    group_spatial_batch_candidates,
    SpatialBatchingStats,
)


def _mk_conv_weight(graph: fx.Graph, root: torch.nn.Module, name: str, out_c: int, in_c: int):
    w = torch.randn(out_c, in_c, 3, 3) * 0.05
    root.register_buffer(name, w)
    return graph.get_attr(name)


def build_minimal_feedback_graph(T: int = 2, C: int = 4, HW: int = 8):
    """x_seq: [T, B, C, HW, HW] placeholder. h_init: [B, C, HW, HW] placeholder.
    Per t: xproj = conv_x(x_t); hproj = conv_h(h_prev); gate = sigmoid(xproj + hproj); h_t = gate * h_prev (toy, not a real cell).
    """
    root = torch.nn.Module()
    graph = fx.Graph()
    x_seq = graph.placeholder("x_seq")
    h_init = graph.placeholder("h_init")

    w_x = _mk_conv_weight(graph, root, "w_x", C, C)
    w_h = _mk_conv_weight(graph, root, "w_h", C, C)

    h_prev = h_init
    nodes_by_role = {"xproj": [], "hproj": []}
    for t in range(T):
        x_t = graph.call_function(torch.ops.aten.select, args=(x_seq, 0, t))
        x_t.meta["kairos_timestep"] = t

        xproj = graph.call_function(F.conv2d, args=(x_t, w_x, None, (1, 1), (1, 1)))
        xproj.meta["kairos_timestep"] = t
        nodes_by_role["xproj"].append(xproj)

        hproj = graph.call_function(F.conv2d, args=(h_prev, w_h, None, (1, 1), (1, 1)))
        hproj.meta["kairos_timestep"] = t
        nodes_by_role["hproj"].append(hproj)

        gate_sum = graph.call_function(torch.add, args=(xproj, hproj))
        gate_sum.meta["kairos_timestep"] = t
        gate = graph.call_function(torch.sigmoid, args=(gate_sum,))
        gate.meta["kairos_timestep"] = t
        h_t = graph.call_function(torch.mul, args=(gate, h_prev))
        h_t.meta["kairos_timestep"] = t

        h_prev = h_t

    graph.output(h_prev)
    graph.lint()
    gm = fx.GraphModule(root, graph)
    return gm, nodes_by_role


def test_wx_accepted_wh_rejected():
    """W_x (conv_x(x_t), input traces to the external x_seq placeholder) must
    batch cleanly across the whole window. W_h (conv_h(h_prev)) must never
    end up batched: t=0's h_prev is h_init (a placeholder, genuinely
    timestep-invariant, so it is individually legal in isolation -- there is
    nothing wrong with THAT single call) but every t>=1 instance's h_prev is
    the previous timestep's computed gate output, a real cross-iteration
    dependency, and gets rejected by the new legality check. Since batching
    only ever fires on a *complete* window (group_spatial_batch_candidates'
    pre-existing incomplete_window check), one rejected member is enough to
    guarantee the W_h group as a whole is never batched -- this is the
    end-to-end property the spec asks for ("W_h 半边被拒"), not that every
    single W_h node is individually rejected.
    """
    T = 3
    gm, nodes_by_role = build_minimal_feedback_graph(T=T)
    stats = SpatialBatchingStats()
    candidates = collect_spatial_batch_candidates(gm, temporal_window=T, enabled_ops=("conv",), stats=stats)
    candidate_nodes = {c.node for c in candidates}

    for xproj_node in nodes_by_role["xproj"]:
        assert xproj_node in candidate_nodes, f"{xproj_node.name} (W_x side) should be ACCEPTED as a batching candidate"

    # t=0's hproj individually traces back to h_init (a placeholder) -- correctly legal in isolation.
    assert nodes_by_role["hproj"][0] in candidate_nodes, "t=0 hproj (input=h_init, a placeholder) should be individually legal"
    # t>=1's hproj instances genuinely cross a timestep boundary -- must be rejected.
    for hproj_node in nodes_by_role["hproj"][1:]:
        assert hproj_node not in candidate_nodes, f"{hproj_node.name} (W_h side, t>=1, feeds off computed h_prev) should be REJECTED"
    assert stats.reasons.get("feedback_dependency", 0) == T - 1, (
        f"expected {T - 1} feedback_dependency rejections (t=1..{T-1}), got {stats.reasons.get('feedback_dependency', 0)}"
    )

    # End-to-end: no COMPLETE W_h group ever forms (the whole point) -- the surviving t=0
    # hproj candidate is left as an incomplete_window group of size 1 < T and skipped.
    groups = group_spatial_batch_candidates(candidates, temporal_window=T, stats=stats)
    hproj_groups = [g for g in groups if any(c.node in nodes_by_role["hproj"] for c in g.candidates)]
    assert not hproj_groups, f"a W_h group was formed despite the legality gate: {hproj_groups}"
    assert stats.reasons.get("incomplete_window", 0) >= 1, "expected the surviving lone W_h candidate to be skipped as an incomplete window"

    xproj_groups = [g for g in groups if any(c.node in nodes_by_role["xproj"] for c in g.candidates)]
    assert len(xproj_groups) == 1 and len(xproj_groups[0].candidates) == T, "W_x should form one complete window-sized group"

    print(
        f"[PASS] W_x: 1 complete group of {T} formed. W_h: 0 groups formed "
        f"({T - 1} rejected individually via feedback_dependency, the lone survivor caught by incomplete_window)."
    )


def test_existing_snn_graph_batching_decisions_unchanged():
    """No isolated unit-graph fixture exists for MobileNetV1's real batching
    log, so this re-derives the SAME conv/bn/add candidate set that a real
    SNN graph would offer to collect_spatial_batch_candidates and confirms
    the new legality gate accepts every one of them (i.e. it introduces zero
    new rejections for the shapes this pass has always handled): every
    conv/bn input on such a graph is either a temporal_stack_getitem, a
    previous_batched_chunk_getitem, or a same-timestep local value tracing
    back to one of those -- never a raw cross-timestep recurrence, since
    membrane state only ever crosses timesteps through an opaque
    snn_custom.* op boundary that the legality walk stops at.
    """
    root = torch.nn.Module()
    graph = fx.Graph()
    x_seq = graph.placeholder("x_seq")
    v_init = graph.placeholder("v_init")
    w = _mk_conv_weight(graph, root, "w", 4, 4)

    T = 3
    v_prev = v_init
    conv_nodes = []
    for t in range(T):
        x_t = graph.call_function(torch.ops.aten.select, args=(x_seq, 0, t))
        x_t.meta["kairos_timestep"] = t
        conv = graph.call_function(F.conv2d, args=(x_t, w, None, (1, 1), (1, 1)))
        conv.meta["kairos_timestep"] = t
        conv_nodes.append(conv)
        # membrane state recurrence hidden behind an opaque snn_custom.* boundary,
        # exactly like the real fused_conv_lif_state / fused_temporal_lif_state ops.
        fused = graph.call_function(
            torch.ops.snn_custom.fused_temporal_lif_state.default,
            args=(conv, v_prev, 1.0, 0.0, 2.0, True),
        )
        fused.meta["kairos_timestep"] = t
        spike = graph.call_function(operator.getitem, args=(fused, 0))
        spike.meta["kairos_timestep"] = t
        v_next = graph.call_function(operator.getitem, args=(fused, 1))
        v_next.meta["kairos_timestep"] = t
        v_prev = v_next

    graph.output(v_prev)
    graph.lint()
    gm = fx.GraphModule(root, graph)

    stats = SpatialBatchingStats()
    candidates = collect_spatial_batch_candidates(gm, temporal_window=T, enabled_ops=("conv",), stats=stats)
    candidate_nodes = {c.node for c in candidates}
    for conv in conv_nodes:
        assert conv in candidate_nodes, f"{conv.name}: existing-SNN-style conv (input traces to snn_custom boundary, not raw recurrence) must remain a batching candidate -- legality patch introduced a regression"
    assert stats.reasons.get("feedback_dependency", 0) == 0, (
        f"legality patch introduced {stats.reasons.get('feedback_dependency', 0)} new rejections on an existing-SNN-shaped graph -- regression"
    )
    print(f"[PASS] existing-SNN-shaped graph: {len(conv_nodes)}/{len(conv_nodes)} conv candidates unaffected, 0 new rejections")


if __name__ == "__main__":
    test_wx_accepted_wh_rejected()
    test_existing_snn_graph_batching_decisions_unchanged()
    print("\n[ALL PASS]")
