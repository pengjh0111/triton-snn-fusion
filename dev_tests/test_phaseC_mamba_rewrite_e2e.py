"""End-to-end verification for rewrite_mamba_scan_to_fused, covering the
"Mamba verification checklist" items that dev_tests/test_phaseB_selective_scan.py
(kernel-only correctness) does not: window-spanning graph rewrite correctness,
idempotency, cross-window ssm_state threading, FIFO t-order preservation, and
layer-major scheduling (layer0's every timestep precedes layer1's first).

Uses real Dynamo graph capture (torch.compile with a custom backend), exactly
like dev_tests/test_temporal_lif_fx_rewrite.py's pattern: run the eager model
once for a reference, then mutate the captured GraphModule inside the backend
and compare.
"""
import operator
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import runtime.snn_custom_ops as snn_custom_ops
from benchmarks.validate_kairos_baselines import KairosMamba, SequenceInputLoopWrapper
from compiler.fx_lif_temporal_rewrite import (
    _match_mamba_scan_step,
    _param_like_name,
    collect_mamba_scan_patterns,
    rewrite_mamba_scan_to_fused,
)
from compiler.fx_temporal_annotation import annotate_temporal_metadata
from compiler.fx_temporal_scheduler import reorder_fx_graph_by_temporal_windows

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _layer_index(layer_key: str) -> int:
    match = re.search(r"blocks[_.](?:modules[_.])?(\d+)", layer_key)
    if match is None:
        raise ValueError(f"could not extract layer index from {layer_key!r}")
    return int(match.group(1))


def _build_model(n_layer: int = 2):
    torch.manual_seed(0)
    model = KairosMamba(
        d_model=16, n_layer=n_layer, d_inner=32, d_state=8, d_conv=4, dt_rank=4, num_classes=5,
    ).to(DEVICE).eval()
    return model


def run_case(T: int, window: int, batch_size: int = 3) -> Dict:
    model = _build_model(n_layer=2)
    wrapper = SequenceInputLoopWrapper(model, T).to(DEVICE).eval()
    x = torch.randn(T, batch_size, model.d_model, device=DEVICE)

    with torch.no_grad():
        eager = wrapper(x)

    captured: Dict = {}

    def backend(gm: torch.fx.GraphModule, example_inputs):
        annotate_temporal_metadata(gm, window, T, strict=False)
        patterns = collect_mamba_scan_patterns(gm)
        schedule_result = reorder_fx_graph_by_temporal_windows(gm, T, window, patterns)
        assert schedule_result.ok, f"scheduling failed: {schedule_result.reason}"

        # Snapshot the post-reorder, pre-rewrite scan-step matches for the
        # FIFO / layer-major position-log checks below (must happen before
        # rewrite mutates the graph out from under these nodes).
        order = {node: index for index, node in enumerate(gm.graph.nodes)}
        pre_rewrite_matches: List[Tuple[str, int, int]] = []
        for node in list(gm.graph.nodes):
            if node.target is not torch.exp:
                continue
            match = _match_mamba_scan_step(node)
            if match is None:
                continue
            layer_key = _param_like_name(match["A"])
            timestep = node.meta.get("kairos_timestep")
            if layer_key is not None and isinstance(timestep, int):
                pre_rewrite_matches.append((layer_key, timestep, order[node]))
        captured["pre_rewrite_matches"] = pre_rewrite_matches

        nodes_before = len(gm.graph.nodes)
        replaced_1 = rewrite_mamba_scan_to_fused(gm, window)
        nodes_after = len(gm.graph.nodes)
        replaced_2 = rewrite_mamba_scan_to_fused(gm, window)  # idempotency probe

        fused_nodes = [n for n in gm.graph.nodes if n.target is torch.ops.snn_custom.fused_temporal_selective_scan]
        post_order = {node: index for index, node in enumerate(gm.graph.nodes)}

        captured["gm"] = gm
        captured["replaced_1"] = replaced_1
        captured["replaced_2"] = replaced_2
        captured["nodes_before"] = nodes_before
        captured["nodes_after"] = nodes_after
        captured["fused_nodes"] = fused_nodes
        captured["post_order"] = post_order

        gm.graph.lint()
        gm.recompile()
        return gm.forward

    compiled = torch.compile(wrapper, backend=backend, fullgraph=False, dynamic=False)
    with torch.no_grad():
        rewritten = compiled(x)

    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()

    diff = (eager - rewritten).abs()
    captured["eager"] = eager
    captured["rewritten"] = rewritten
    captured["max_err"] = float(diff.max().item())
    captured["allclose"] = torch.allclose(eager, rewritten, rtol=1e-4, atol=1e-4)
    return captured


def test_single_window_correctness_and_idempotency():
    T = window = 4
    result = run_case(T, window)
    n_layer = 2
    expected_replaced = n_layer * T
    assert result["replaced_1"] == expected_replaced, (
        f"expected {expected_replaced} replaced instances, got {result['replaced_1']}"
    )
    assert result["replaced_2"] == 0, f"second pass should be idempotent, got {result['replaced_2']}"
    assert result["allclose"], f"rewritten output diverged from eager, max_err={result['max_err']}"
    assert result["nodes_after"] < result["nodes_before"], "fused graph should have fewer nodes than pre-rewrite"
    print(
        f"[PASS] single-window T={T} window={window}: replaced={result['replaced_1']} "
        f"idempotent_second_pass={result['replaced_2']} max_err={result['max_err']:.3e} "
        f"nodes {result['nodes_before']}->{result['nodes_after']}"
    )


def test_cross_window_ssm_state_threading():
    T, window = 8, 4
    result = run_case(T, window)
    n_layer = 2
    expected_replaced = n_layer * T
    assert result["replaced_1"] == expected_replaced, (
        f"expected {expected_replaced} replaced instances, got {result['replaced_1']}"
    )
    assert result["replaced_2"] == 0, f"second pass should be idempotent, got {result['replaced_2']}"
    assert result["allclose"], f"rewritten output diverged from eager, max_err={result['max_err']}"

    fused_nodes = result["fused_nodes"]
    assert len(fused_nodes) == n_layer * (T // window), (
        f"expected {n_layer * (T // window)} fused calls (2 layers x 2 windows), got {len(fused_nodes)}"
    )

    post_order = result["post_order"]
    by_layer: Dict[int, List[torch.fx.Node]] = {}
    for node in fused_nodes:
        a_node = node.args[4]
        layer_key = _param_like_name(a_node)
        assert layer_key is not None
        by_layer.setdefault(id(a_node), []).append(node)

    assert len(by_layer) == n_layer, f"expected {n_layer} distinct layers among fused calls, got {len(by_layer)}"

    for a_id, nodes in by_layer.items():
        nodes = sorted(nodes, key=lambda n: post_order[n])
        assert len(nodes) == T // window
        first_window, second_window = nodes[0], nodes[1]

        first_state_arg = first_window.args[6]
        # window0's ssm_state_prev must NOT be another fused call's h_final --
        # it's the genuine external/initial state for this layer.
        assert not (
            isinstance(first_state_arg, torch.fx.Node)
            and first_state_arg.target is operator.getitem
            and first_state_arg.args[0] in fused_nodes
        ), "window0's ssm_state_prev should be the initial state, not another window's h_final"

        second_state_arg = second_window.args[6]
        assert isinstance(second_state_arg, torch.fx.Node), "window1's ssm_state_prev must be a node"
        assert second_state_arg.target is operator.getitem, (
            f"window1's ssm_state_prev should be a getitem(h_final), got target={second_state_arg.target}"
        )
        assert second_state_arg.args == (first_window, 1), (
            "window1's ssm_state_prev must be exactly window0's own fused call's h_final "
            f"(getitem(fused,1)); got args={second_state_arg.args}, window0 fused node={first_window}"
        )

    print(
        f"[PASS] cross-window ssm_state threading T={T} window={window}: "
        f"{len(fused_nodes)} fused calls across {n_layer} layers, each window1 correctly "
        f"threads from its own window0's h_final. max_err={result['max_err']:.3e}"
    )


def test_fifo_t_order_and_layer_major_scheduling():
    T, window = 8, 4
    result = run_case(T, window)
    matches = result["pre_rewrite_matches"]
    assert matches, "no scan-step matches captured"

    by_layer: Dict[str, List[Tuple[int, int]]] = {}
    for layer_key, timestep, pos in matches:
        by_layer.setdefault(layer_key, []).append((timestep, pos))

    assert len(by_layer) == 2, f"expected 2 layers, got {len(by_layer)}"

    # FIFO / t-order preservation: within a single layer's own window(s),
    # increasing timestep must map to strictly increasing graph position.
    for layer_key, pairs in by_layer.items():
        pairs_sorted_by_t = sorted(pairs, key=lambda p: p[0])
        positions = [pos for _, pos in pairs_sorted_by_t]
        assert positions == sorted(positions), (
            f"layer {layer_key}: timestep order does not match graph position order: {pairs_sorted_by_t}"
        )

    # Layer-major grouping: window_id is the scheduler's outer grouping (it
    # iterates window_id, then lays out each window's layers), so the
    # invariant is "within a given window, layer0's instances precede
    # layer1's" -- not globally across all T. For T==window_size (single
    # window, see test_single_window_correctness_and_idempotency) these
    # coincide; for T>window_size they don't, since layer0's *next* window
    # legitimately comes after layer1's *current* window.
    ordered_layers = sorted(by_layer.keys(), key=_layer_index)
    layer0_key, layer1_key = ordered_layers[0], ordered_layers[1]
    by_window: Dict[int, Dict[str, List[int]]] = {}
    for layer_key, pairs in by_layer.items():
        for timestep, pos in pairs:
            window_id = timestep // window
            by_window.setdefault(window_id, {}).setdefault(layer_key, []).append(pos)

    for window_id, per_layer_pos in sorted(by_window.items()):
        layer0_max_pos = max(per_layer_pos[layer0_key])
        layer1_min_pos = min(per_layer_pos[layer1_key])
        assert layer0_max_pos < layer1_min_pos, (
            f"window {window_id}: layer-major violation: layer0 max position {layer0_max_pos} >= "
            f"layer1 min position {layer1_min_pos}"
        )
    print(
        f"[PASS] FIFO t-order preserved within each layer; layer-major scheduling confirmed "
        f"within each of {len(by_window)} windows"
    )


if __name__ == "__main__":
    snn_custom_ops.configure_fused_op(backend="triton", strict_triton=False, verbose=False)
    test_single_window_correctness_and_idempotency()
    test_cross_window_ssm_state_threading()
    test_fifo_t_order_and_layer_major_scheduling()
    print("[ALL PASS]")
