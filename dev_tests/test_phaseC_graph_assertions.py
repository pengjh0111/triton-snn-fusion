"""Phase C step 3: post-rewrite graph assertions for all three Kairos
workloads. Confirms each rewriter doesn't just run without error but
actually eliminates the raw per-timestep gate chain it targets (zero
leftover sigmoid/tanh for ConvLSTM/GRU; zero leftover unfused scan-step
exp() for Mamba) and replaces it with exactly the expected number of fused
custom-op calls.
"""
import sys
from pathlib import Path
from typing import Dict

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import runtime.snn_custom_ops as snn_custom_ops
from benchmarks.validate_chronos_baselines import (
    ChronosConvLSTM,
    ChronosDeepSpeech2,
    ChronosMamba,
    SequenceInputLoopWrapper,
)
from compiler.fx_lif_temporal_rewrite import (
    _match_mamba_scan_step,
    collect_mamba_scan_patterns,
    rewrite_convlstm_cell_to_fused,
    rewrite_gru_cell_to_fused,
    rewrite_mamba_scan_to_fused,
)
from compiler.fx_temporal_annotation import annotate_temporal_metadata
from compiler.fx_temporal_scheduler import reorder_fx_graph_by_temporal_windows

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _count_raw_sigmoid_tanh(gm: torch.fx.GraphModule) -> Dict[str, int]:
    counts = {"sigmoid": 0, "tanh": 0}
    for node in gm.graph.nodes:
        if node.target is torch.sigmoid or (node.op == "call_method" and node.target == "sigmoid"):
            counts["sigmoid"] += 1
        if node.target is torch.tanh or (node.op == "call_method" and node.target == "tanh"):
            counts["tanh"] += 1
    return counts


def _count_calls(gm: torch.fx.GraphModule, target) -> int:
    return sum(1 for node in gm.graph.nodes if node.target is target)


def test_convlstm_graph_assertions():
    torch.manual_seed(0)
    T = 4
    n_layer = 2
    model = ChronosConvLSTM(
        in_channels=2, hidden_channels=4, num_layers=n_layer, num_classes=3, height=8, width=8,
    ).to(DEVICE).eval()
    wrapper = SequenceInputLoopWrapper(model, T).to(DEVICE).eval()
    x = torch.randn(T, 2, 2, 8, 8, device=DEVICE)

    captured: Dict = {}

    def backend(gm: torch.fx.GraphModule, example_inputs):
        annotate_temporal_metadata(gm, T, T, strict=False)
        before = _count_raw_sigmoid_tanh(gm)
        replaced = rewrite_convlstm_cell_to_fused(gm)
        after = _count_raw_sigmoid_tanh(gm)
        fused_calls = _count_calls(gm, torch.ops.snn_custom.fused_convlstm_cell)
        captured.update(before=before, after=after, replaced=replaced, fused_calls=fused_calls)
        gm.graph.lint()
        gm.recompile()
        return gm.forward

    compiled = torch.compile(wrapper, backend=backend, fullgraph=False, dynamic=False)
    with torch.no_grad():
        compiled(x)

    assert captured["replaced"] == n_layer * T, captured
    assert captured["before"]["sigmoid"] > 0 and captured["before"]["tanh"] > 0, "sanity: raw gate chain must exist pre-rewrite"
    assert captured["after"] == {"sigmoid": 0, "tanh": 0}, f"leftover raw gate nodes after rewrite: {captured['after']}"
    assert captured["fused_calls"] == n_layer * T, captured
    print(
        f"[PASS] ConvLSTM graph assertions: replaced={captured['replaced']} "
        f"fused_calls={captured['fused_calls']} sigmoid/tanh {captured['before']}->{captured['after']}"
    )


def test_gru_graph_assertions():
    torch.manual_seed(0)
    T = 4
    n_layer = 2
    model = ChronosDeepSpeech2(
        freq_bins=33, conv_channels=4, gru_hidden=16, gru_layers=n_layer, num_classes=7,
    ).to(DEVICE).eval()
    wrapper = SequenceInputLoopWrapper(model, T).to(DEVICE).eval()
    x = torch.randn(2, 1, 33, 2 * T, device=DEVICE)

    captured: Dict = {}

    def backend(gm: torch.fx.GraphModule, example_inputs):
        annotate_temporal_metadata(gm, T, T, strict=False)
        before = _count_raw_sigmoid_tanh(gm)
        replaced = rewrite_gru_cell_to_fused(gm)
        after = _count_raw_sigmoid_tanh(gm)
        fused_calls = _count_calls(gm, torch.ops.snn_custom.fused_gru_cell)
        captured.update(before=before, after=after, replaced=replaced, fused_calls=fused_calls)
        gm.graph.lint()
        gm.recompile()
        return gm.forward

    compiled = torch.compile(wrapper, backend=backend, fullgraph=False, dynamic=False)
    with torch.no_grad():
        compiled(x)

    assert captured["replaced"] == n_layer * T, captured
    assert captured["before"]["sigmoid"] > 0 and captured["before"]["tanh"] > 0, "sanity: raw gate chain must exist pre-rewrite"
    assert captured["after"] == {"sigmoid": 0, "tanh": 0}, f"leftover raw gate nodes after rewrite: {captured['after']}"
    assert captured["fused_calls"] == n_layer * T, captured
    print(
        f"[PASS] GRU (DeepSpeech2) graph assertions: replaced={captured['replaced']} "
        f"fused_calls={captured['fused_calls']} sigmoid/tanh {captured['before']}->{captured['after']}"
    )


def test_mamba_graph_assertions():
    torch.manual_seed(0)
    T = window = 4
    n_layer = 2
    model = ChronosMamba(
        d_model=16, n_layer=n_layer, d_inner=32, d_state=8, d_conv=4, dt_rank=4, num_classes=5,
    ).to(DEVICE).eval()
    wrapper = SequenceInputLoopWrapper(model, T).to(DEVICE).eval()
    x = torch.randn(T, 2, model.d_model, device=DEVICE)

    captured: Dict = {}

    def backend(gm: torch.fx.GraphModule, example_inputs):
        annotate_temporal_metadata(gm, window, T, strict=False)
        patterns = collect_mamba_scan_patterns(gm)
        schedule_result = reorder_fx_graph_by_temporal_windows(gm, T, window, patterns)
        assert schedule_result.ok, schedule_result.reason

        exp_before = sum(1 for n in gm.graph.nodes if n.target is torch.exp)
        replaced = rewrite_mamba_scan_to_fused(gm, window)

        leftover_unfused_scan_steps = 0
        for node in gm.graph.nodes:
            if node.target is torch.exp and _match_mamba_scan_step(node) is not None:
                leftover_unfused_scan_steps += 1
        fused_calls = _count_calls(gm, torch.ops.snn_custom.fused_temporal_selective_scan)
        captured.update(
            exp_before=exp_before,
            replaced=replaced,
            leftover_unfused_scan_steps=leftover_unfused_scan_steps,
            fused_calls=fused_calls,
        )
        gm.graph.lint()
        gm.recompile()
        return gm.forward

    compiled = torch.compile(wrapper, backend=backend, fullgraph=False, dynamic=False)
    with torch.no_grad():
        compiled(x)

    assert captured["exp_before"] >= n_layer * T, captured
    assert captured["replaced"] == n_layer * T, captured
    assert captured["leftover_unfused_scan_steps"] == 0, (
        f"leftover unfused scan-step exp() nodes after rewrite: {captured['leftover_unfused_scan_steps']}"
    )
    assert captured["fused_calls"] == n_layer * (T // window), captured
    print(
        f"[PASS] Mamba graph assertions: replaced={captured['replaced']} "
        f"fused_calls={captured['fused_calls']} "
        f"leftover_unfused_scan_steps={captured['leftover_unfused_scan_steps']}"
    )


if __name__ == "__main__":
    snn_custom_ops.configure_fused_op(backend="triton", strict_triton=False, verbose=False)
    test_convlstm_graph_assertions()
    test_gru_graph_assertions()
    test_mamba_graph_assertions()
    print("[ALL PASS]")
