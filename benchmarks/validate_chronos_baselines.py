import argparse
import copy
import json
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from spikingjelly.activation_based import functional, layer, neuron, surrogate
from spikingjelly.activation_based.model import spiking_resnet, spiking_vgg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

import runtime.snn_custom_ops as snn_custom_ops
from runtime.fx_standalone_executor import (
    build_fx_standalone_backend,
    get_last_cudagraph_status as get_fx_standalone_cudagraph_status,
    prune_graph_output_v_final_states,
)
from compiler.chronos_compile import (
    build_chronos_compile_config,
    compile_with_chronos_options,
    diff_compile_counters,
    snapshot_compile_counters,
    summarize_cudagraph_check,
)
from compiler.fx_lif_rewrite import (
    count_fused_conv_lif_state_nodes,
    count_lif_state_nodes,
    match_conv_bn_lif_state,
    match_conv_lif_state,
    rewrite_conv_bn_lif_state_to_fused,
    rewrite_conv_lif_state_to_fused,
)
from compiler.fx_lif_temporal_rewrite import (
    collect_conv_bn_lif_state_patterns,
    collect_conv_bn_add_lif_state_patterns,
    collect_standalone_lif_state_patterns,
    collect_temporal_linear_lif_state_patterns,
    collect_temporal_lif_avgpool_linear_patterns,
    count_fused_temporal_conv_add_lif_state_nodes,
    count_fused_temporal_conv_lif_state_nodes,
    count_fused_temporal_lif_state_nodes,
    count_fused_temporal_linear_lif_state_nodes,
    count_fused_temporal_lif_avgpool_linear_nodes,
    dump_temporal_patterns,
    dump_temporal_lif_avgpool_linear_patterns,
    dump_temporal_lif_avgpool_linear_windows,
    dump_temporal_linear_lif_patterns,
    dump_temporal_linear_lif_windows,
    dump_temporal_rewrite_log,
    dump_temporal_windows,
    group_temporal_patterns,
    group_temporal_residual_patterns,
    group_temporal_lif_patterns,
    group_temporal_linear_lif_patterns,
    group_temporal_lif_avgpool_linear_patterns,
    make_temporal_windows,
    make_temporal_residual_windows,
    make_temporal_lif_windows,
    make_temporal_linear_lif_windows,
    make_temporal_lif_avgpool_linear_windows,
    rewrite_temporal_conv_bn_add_lif_state_to_fused,
    rewrite_temporal_conv_bn_lif_state_to_fused,
    rewrite_temporal_lif_state_to_fused,
    rewrite_temporal_linear_lif_state_to_fused,
    rewrite_temporal_lif_avgpool_linear_to_fused,
    collect_mamba_scan_patterns,
    rewrite_convlstm_cell_to_fused,
    rewrite_gru_cell_to_fused,
    rewrite_mamba_scan_to_fused,
)
from compiler.fx_spatial_batching import apply_spatial_batching
from compiler.fx_temporal_annotation import annotate_temporal_metadata
from compiler.fx_temporal_graph_validation import (
    analyze_temporal_graph,
    dump_temporal_graph_validation,
    print_temporal_graph_summary,
)
from compiler.fx_temporal_scheduler import reorder_fx_graph_by_temporal_windows
from compiler.fx_temporal_spatial_canonicalize import canonicalize_temporal_spatial_ir
from benchmarks.helpers.models_for_fx import CustomStatefulIFNode, reset_custom_stateful_lif_modules


LIF_IMPL_CHOICES = ("chronos", "spikingjelly")


@dataclass
class RunResult:
    name: str
    ok: bool
    shape: Optional[List[int]] = None
    dtype: Optional[str] = None
    elapsed_ms: Optional[float] = None
    max_abs_diff: Optional[float] = None
    mean_abs_diff: Optional[float] = None
    allclose: Optional[bool] = None
    error: str = ""


@dataclass
class RewriteCounters:
    captured_graphs: int = 0
    lif_state_nodes: int = 0
    direct_matches: int = 0
    conv_bn_matches: int = 0
    direct_replaced: int = 0
    conv_bn_replaced: int = 0
    fused_state_nodes: int = 0
    fused_temporal_state_nodes: int = 0
    fused_temporal_residual_state_nodes: int = 0
    fused_temporal_lif_state_nodes: int = 0
    fused_temporal_linear_lif_state_nodes: int = 0
    fused_temporal_lif_avgpool_linear_nodes: int = 0
    fused_temporal_lif_tail_nodes: int = 0
    temporal_groups: int = 0
    temporal_windows: int = 0
    temporal_replaced_windows: int = 0
    temporal_replaced_patterns: int = 0
    temporal_skipped_windows: int = 0
    temporal_residual_groups: int = 0
    temporal_residual_windows: int = 0
    temporal_residual_total_windows: int = 0
    temporal_residual_replaced_windows: int = 0
    temporal_residual_rewritten_windows: int = 0
    temporal_residual_replaced_patterns: int = 0
    temporal_residual_skipped_windows: int = 0
    temporal_residual_remapped_spike_external_users: int = 0
    temporal_residual_unremappable_external_users: int = 0
    temporal_residual_skip_reasons: Dict[str, int] = field(default_factory=dict)
    residual_fuse_skip_reasons: Dict[str, int] = field(default_factory=dict)
    temporal_lif_windows: int = 0
    temporal_lif_total_windows: int = 0
    temporal_lif_rewritten_windows: int = 0
    temporal_lif_replaced_patterns: int = 0
    temporal_lif_skipped_windows: int = 0
    temporal_lif_remapped_spike_external_users: int = 0
    temporal_lif_unremappable_external_users: int = 0
    temporal_lif_skip_reasons: Dict[str, int] = field(default_factory=dict)
    linear_lif_patterns: int = 0
    temporal_linear_lif_windows: int = 0
    temporal_linear_lif_total_windows: int = 0
    temporal_linear_lif_rewritten_windows: int = 0
    temporal_linear_lif_replaced_patterns: int = 0
    temporal_linear_lif_skipped_windows: int = 0
    temporal_linear_lif_remapped_spike_external_users: int = 0
    temporal_linear_lif_unremappable_external_users: int = 0
    temporal_linear_lif_skip_reasons: Dict[str, int] = field(default_factory=dict)
    temporal_lif_avgpool_linear_windows: int = 0
    temporal_lif_avgpool_linear_total_windows: int = 0
    temporal_lif_avgpool_linear_rewritten_windows: int = 0
    temporal_lif_avgpool_linear_replaced_patterns: int = 0
    temporal_lif_avgpool_linear_skipped_windows: int = 0
    temporal_lif_avgpool_linear_skip_reasons: Dict[str, int] = field(default_factory=dict)
    # Deprecated compatibility fields; mirrors temporal_lif_avgpool_linear_*.
    temporal_lif_tail_windows: int = 0
    temporal_lif_tail_total_windows: int = 0
    temporal_lif_tail_rewritten_windows: int = 0
    temporal_lif_tail_replaced_patterns: int = 0
    temporal_lif_tail_skipped_windows: int = 0
    temporal_lif_tail_skip_reasons: Dict[str, int] = field(default_factory=dict)
    single_step_replaced_patterns: int = 0
    temporal_schedule_ok: bool = False
    temporal_schedule_windows: int = 0
    temporal_schedule_moved_nodes: int = 0
    temporal_schedule_reason: str = ""
    temporal_annotated_nodes: int = 0
    temporal_annotation_missing: int = 0
    temporal_annotation_roles: Dict[str, int] = field(default_factory=dict)
    temporal_annotation_windows: Dict[int, int] = field(default_factory=dict)
    temporal_annotation_reasons: Dict[str, int] = field(default_factory=dict)
    spatial_batch_groups: int = 0
    spatial_batched_ops: int = 0
    spatial_batch_chains: int = 0
    spatial_chain_groups: int = 0
    spatial_cat_eliminated: int = 0
    spatial_chunk_eliminated: int = 0
    spatial_batched_conv: int = 0
    spatial_batched_bn: int = 0
    spatial_batched_add: int = 0
    spatial_batched_pool: int = 0
    spatial_batched_maxpool: int = 0
    spatial_batched_avgpool: int = 0
    spatial_batched_adaptive_avgpool: int = 0
    spatial_batched_flatten: int = 0
    spatial_batched_linear: int = 0
    spatial_batched_elementwise: int = 0
    spatial_batched_layer_norm: int = 0
    spatial_batched_attention: int = 0
    spatial_temporal_stack_bn_groups: int = 0
    spatial_temporal_stack_add_groups: int = 0
    spatial_temporal_stack_pool_groups: int = 0
    spatial_temporal_stack_flatten_groups: int = 0
    spatial_temporal_stack_linear_groups: int = 0
    spatial_temporal_stack_groups: int = 0
    spatial_temporal_stack_flatten_inputs: int = 0
    spatial_cat_avoided_by_temporal_stack_flatten: int = 0
    spatial_previous_batched_groups: int = 0
    spatial_reused_previous_batched_inputs: int = 0
    spatial_chunk_cat_avoided: int = 0
    spatial_batch_skipped: int = 0
    spatial_batch_reasons: Dict[str, int] = field(default_factory=dict)
    canonicalize_cat_chunk_removed: int = 0
    canonicalize_chunk_cat_removed: int = 0
    canonicalize_getitem_cat_removed: int = 0
    canonicalize_stack_getitem_removed: int = 0
    canonicalize_getitem_stack_removed: int = 0
    canonicalize_stack_chunk_removed: int = 0
    canonicalize_getitem_stack_chunk_removed: int = 0
    canonicalize_cat_linear_chunk_removed: int = 0
    canonicalize_cat_linear_chunk_getitem_replaced: int = 0
    temporal_mean_rewrites: int = 0
    temporal_mean_removed_getitems: int = 0
    temporal_mean_removed_adds: int = 0
    state_prune_removed_final_return_states: int = 0
    ir_nodes_before: int = 0
    ir_nodes_after: int = 0
    ir_getitem_before: int = 0
    ir_getitem_after: int = 0
    ir_add_before: int = 0
    ir_add_after: int = 0
    ir_div_before: int = 0
    ir_div_after: int = 0
    ir_returned_states_before: int = 0
    ir_returned_states_after: int = 0
    canonicalize_view_folded: int = 0
    canonicalize_dead_nodes_removed: int = 0
    canonicalize_final_cat_count: int = 0
    canonicalize_final_chunk_count: int = 0
    canonicalize_final_getitem_count: int = 0
    temporal_graph_getitem_count: int = 0
    temporal_graph_getitem_from_temporal: int = 0
    temporal_graph_materialized_timestep_tensors: int = 0
    temporal_graph_fragmentation_paths: int = 0
    convlstm_replaced_patterns: int = 0
    gru_replaced_patterns: int = 0
    mamba_replaced_patterns: int = 0
    mamba_schedule_ok: bool = True
    mamba_schedule_reason: str = ""


class SingleStepModeLoopWrapper(nn.Module):
    def __init__(self, layer: nn.Module, T: int):
        super().__init__()
        self.layer = layer
        self.T = T

    def forward(self, x):
        out_spikes_counter = 0
        for _ in range(self.T):
            out_spikes_counter = out_spikes_counter + self.layer(x)
        return out_spikes_counter / self.T


class MultiStepModeWrapper(nn.Module):
    def __init__(self, layer: nn.Module, T: int):
        super().__init__()
        self.layer = layer
        self.T = T

    def forward(self, x):
        x_seq = x.unsqueeze(0).repeat((self.T,) + (1,) * x.dim())
        y_seq = self.layer(x_seq)
        return y_seq.mean(0)


SingleStepWrapper = SingleStepModeLoopWrapper
MultiStepWrapper = MultiStepModeWrapper


class SequenceInputLoopWrapper(nn.Module):
    """Sequence-input analogue of SingleStepModeLoopWrapper: the T-loop
    lives here, not inside the model. The one structural difference from
    SingleStepModeLoopWrapper is that each iteration feeds a genuinely
    different x_seq[t] slice instead of replicating the same x -- these
    models (ConvLSTM/Mamba/DeepSpeech2) need real per-t input, unlike the
    rate-coded SNN convention where every step sees identical input and
    only internal membrane state varies.

    model must expose:
      - step(x_t, *state) -> (y_t, *new_state)   (explicit state I/O, the
        exact shape a single-step ONNX export needs)
      - init_state(batch_size, device, dtype) -> tuple(state)
      - optionally frontend(x) -> x_seq for models whose "outside the
        temporal region" preprocessing (e.g. DeepSpeech2's conv frontend)
        needs to run once, in full batch, before the loop; models without
        one are given their input directly as x_seq.
    """

    def __init__(self, model: nn.Module, T: int):
        super().__init__()
        self.model = model
        self.T = T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_seq = self.model.frontend(x) if hasattr(self.model, "frontend") else x
        batch_size = x_seq.shape[1]
        state = self.model.init_state(batch_size, x_seq.device, x_seq.dtype)
        outputs = []
        for t in range(self.T):
            result = self.model.step(x_seq[t], *state)
            y_t, state = result[0], result[1:]
            outputs.append(y_t)
        return torch.stack(outputs, dim=0)


class SequenceInputMultiStepWrapper(SequenceInputLoopWrapper):
    """Sequence-input analogue of MultiStepModeWrapper. The existing SNN
    "multi-step" convention calls a step_mode="m" module ONCE with a
    [T,...]-shaped tensor and lets it vectorize/loop over T internally.
    Genuinely-recurrent models (LSTM/GRU/SSM state) have no such vectorized
    fast path -- an explicit T-loop is unavoidable either way -- so this
    class is identical to SequenceInputLoopWrapper. It exists as a
    separately-named subclass purely so callers that dispatch on wrapper
    *kind* (single-step vs multi-step case family, mirroring
    SingleStepModeLoopWrapper/MultiStepModeWrapper) don't need a special
    case for sequence-input models.
    """


CHRONOS_MODEL_CHOICES = [
    "resnet18",
    "resnet34",
    "resnet32",
    "alexnet",
    "zfnet",
    "vgg11",
    "vgg16",
    "mobilenetv1",
    "mobilenetv2",
    "spiketransformer",
    "spikebert",
    "convlstm",
    "mamba",
    "deepspeech2",
]

# Per-model input convention, declared once here and consulted everywhere a
# wrapper/input/case-construction decision depends on it (make_model_input,
# benchmark_one_model's wrapper selection) instead of scattered model_name
# checks. "static_replicate": the existing SNN convention -- a single
# [B,...] input is reused for all T steps (SingleStepModeLoopWrapper /
# MultiStepModeWrapper), output varies only via internal membrane state.
# "sequence": a genuine [T,B,...] (or frontend-preprocessed) input with
# different data per timestep (SequenceInputLoopWrapper /
# SequenceInputMultiStepWrapper). Every existing model defaults to
# static_replicate; only the three new workloads use sequence.
CHRONOS_MODEL_INPUT_MODE: Dict[str, str] = {
    "convlstm": "sequence",
    "mamba": "sequence",
    "deepspeech2": "sequence",
}


def model_input_mode(model_name: str) -> str:
    return CHRONOS_MODEL_INPUT_MODE.get(model_name, "static_replicate")


def _lif_node_class(lif_impl: str):
    if lif_impl == "chronos":
        return CustomStatefulIFNode
    if lif_impl == "spikingjelly":
        return neuron.LIFNode
    raise ValueError(f"unsupported lif_impl={lif_impl}, expected one of {LIF_IMPL_CHOICES}")


def _make_lif_node(lif_impl: str = "chronos") -> nn.Module:
    if lif_impl == "chronos":
        return CustomStatefulIFNode(
            v_threshold=1.0,
            v_reset=0.0,
            tau=2.0,
            surrogate_function=surrogate.ATan(),
        )
    if lif_impl == "spikingjelly":
        return neuron.LIFNode(
            tau=2.0,
            v_threshold=1.0,
            v_reset=0.0,
            surrogate_function=surrogate.ATan(),
        )
    raise ValueError(f"unsupported lif_impl={lif_impl}, expected one of {LIF_IMPL_CHOICES}")


def reset_lif_modules(model: nn.Module):
    reset_custom_stateful_lif_modules(model)
    try:
        functional.reset_net(model)
    except Exception:
        pass


class ChronosAlexZFNet(nn.Module):
    def __init__(
        self,
        *,
        kind: str,
        channels: int = 64,
        num_classes: int = 10,
        input_channels: int = 3,
        step_mode: str = "s",
        lif_impl: str = "chronos",
    ):
        super().__init__()
        self.kind = kind
        self.step_mode = step_mode
        c = int(channels)

        if kind == "alexnet":
            conv_specs = [
                (input_channels, c, 11, 4, 2, False),
                (c, 3 * c, 5, 1, 2, False),
                (3 * c, 6 * c, 3, 1, 1, False),
                (6 * c, 4 * c, 3, 1, 1, False),
                (4 * c, 4 * c, 3, 1, 1, False),
            ]
            pool_specs = {
                0: nn.MaxPool2d(3, stride=2),
                1: nn.MaxPool2d(3, stride=2),
                4: nn.MaxPool2d(3, stride=2),
            }
            avgpool = nn.AdaptiveAvgPool2d((6, 6))
            classifier_in = 4 * c * 6 * 6
        elif kind == "zfnet":
            conv_specs = [
                (input_channels, c, 7, 2, 3, False),
                (c, 3 * c, 5, 1, 2, False),
                (3 * c, 6 * c, 3, 1, 1, False),
                (6 * c, 4 * c, 3, 1, 1, False),
                (4 * c, 4 * c, 3, 1, 1, False),
            ]
            pool_specs = {
                0: nn.MaxPool2d(3, stride=2, padding=1),
                1: nn.MaxPool2d(3, stride=2, padding=1),
                4: nn.MaxPool2d(3, stride=2, padding=1),
            }
            avgpool = nn.AdaptiveAvgPool2d((1, 1))
            classifier_in = 4 * c
        else:
            raise ValueError(f"unsupported Alex/ZF kind: {kind}")

        features = []
        for idx, (cin, cout, kernel_size, stride, padding, bias) in enumerate(conv_specs):
            features.extend([
                nn.Conv2d(cin, cout, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(cout),
                _make_lif_node(lif_impl),
            ])
            if idx in pool_specs:
                features.append(pool_specs[idx])

        self.features = nn.Sequential(*features)
        self.avgpool = avgpool
        self.flatten = nn.Flatten(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(classifier_in, 64 * c, bias=False),
            _make_lif_node(lif_impl),
            nn.Dropout(0.5),
            nn.Linear(64 * c, 64 * c, bias=False),
            _make_lif_node(lif_impl),
            nn.Linear(64 * c, num_classes, bias=False),
            _make_lif_node(lif_impl),
        )

    def _forward_single(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        return self.classifier(x)

    def forward(self, x):
        if x.dim() == 5:
            return torch.stack([self._forward_single(x[t]) for t in range(x.shape[0])], dim=0)
        return self._forward_single(x)


def _chronos_if_node() -> CustomStatefulIFNode:
    return _make_lif_node("chronos")


def _scaled_channels(base_channels: int, channels: int) -> int:
    return max(1, int(base_channels * channels / 32))


class ChronosMobileNetV1(nn.Module):
    def __init__(
        self,
        *,
        channels: int = 64,
        num_classes: int = 10,
        input_channels: int = 3,
        step_mode: str = "s",
        lif_impl: str = "chronos",
    ):
        super().__init__()
        self.step_mode = step_mode

        ch_32 = _scaled_channels(32, channels)
        ch_64 = _scaled_channels(64, channels)
        ch_128 = _scaled_channels(128, channels)
        ch_256 = _scaled_channels(256, channels)
        ch_512 = _scaled_channels(512, channels)
        ch_1024 = _scaled_channels(1024, channels)

        modules = [
            layer.Conv2d(input_channels, ch_32, kernel_size=3, stride=2, padding=1, bias=False),
            layer.BatchNorm2d(ch_32),
            _make_lif_node(lif_impl),
        ]

        def add_depthwise_pointwise(in_ch: int, out_ch: int, stride: int):
            modules.extend(
                [
                    layer.Conv2d(
                        in_ch,
                        in_ch,
                        kernel_size=3,
                        stride=stride,
                        padding=1,
                        groups=in_ch,
                        bias=False,
                    ),
                    layer.BatchNorm2d(in_ch),
                    _make_lif_node(lif_impl),
                    layer.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
                    layer.BatchNorm2d(out_ch),
                    _make_lif_node(lif_impl),
                ]
            )

        add_depthwise_pointwise(ch_32, ch_64, stride=1)
        add_depthwise_pointwise(ch_64, ch_128, stride=2)
        add_depthwise_pointwise(ch_128, ch_128, stride=1)
        add_depthwise_pointwise(ch_128, ch_256, stride=2)
        add_depthwise_pointwise(ch_256, ch_256, stride=1)
        add_depthwise_pointwise(ch_256, ch_512, stride=2)
        for _ in range(5):
            add_depthwise_pointwise(ch_512, ch_512, stride=1)
        add_depthwise_pointwise(ch_512, ch_1024, stride=2)
        add_depthwise_pointwise(ch_1024, ch_1024, stride=1)

        modules.extend(
            [
                layer.AdaptiveAvgPool2d((1, 1)),
                layer.Flatten(),
                layer.Dropout(0.2),
                layer.Linear(ch_1024, num_classes, bias=False),
                _make_lif_node(lif_impl),
            ]
        )
        self.layer = nn.Sequential(*modules)
        functional.set_step_mode(self.layer, step_mode=step_mode)

    def forward(self, x: torch.Tensor):
        return self.layer(x)


class ChronosSpikingInvertedResidual(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, expand_ratio: int, step_mode: str = "s", lif_impl: str = "chronos"):
        super().__init__()
        if stride not in (1, 2):
            raise ValueError(f"unsupported MobileNetV2 stride: {stride}")
        hidden_dim = int(in_ch * expand_ratio)
        self.use_res_connect = stride == 1 and in_ch == out_ch

        modules = []
        if expand_ratio != 1:
            modules.extend(
                [
                    layer.Conv2d(in_ch, hidden_dim, kernel_size=1, bias=False),
                    layer.BatchNorm2d(hidden_dim),
                    _make_lif_node(lif_impl),
                ]
            )
        modules.extend(
            [
                layer.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=hidden_dim,
                    bias=False,
                ),
                layer.BatchNorm2d(hidden_dim),
                _make_lif_node(lif_impl),
                layer.Conv2d(hidden_dim, out_ch, kernel_size=1, bias=False),
                layer.BatchNorm2d(out_ch),
            ]
        )
        self.conv = nn.Sequential(*modules)
        functional.set_step_mode(self.conv, step_mode=step_mode)

    def forward(self, x: torch.Tensor):
        out = self.conv(x)
        if self.use_res_connect:
            return x + out
        return out


class ChronosMobileNetV2(nn.Module):
    def __init__(
        self,
        *,
        channels: int = 64,
        num_classes: int = 10,
        input_channels: int = 3,
        step_mode: str = "s",
        lif_impl: str = "chronos",
    ):
        super().__init__()
        self.step_mode = step_mode

        ch_32 = _scaled_channels(32, channels)
        ch_16 = _scaled_channels(16, channels)
        ch_24 = _scaled_channels(24, channels)
        ch_32b = _scaled_channels(32, channels)
        ch_64 = _scaled_channels(64, channels)
        ch_96 = _scaled_channels(96, channels)
        ch_160 = _scaled_channels(160, channels)
        ch_320 = _scaled_channels(320, channels)
        ch_1280 = _scaled_channels(1280, channels)

        modules = [
            layer.Conv2d(input_channels, ch_32, kernel_size=3, stride=2, padding=1, bias=False),
            layer.BatchNorm2d(ch_32),
            _make_lif_node(lif_impl),
        ]

        cfg = [
            (1, ch_16, 1, 1),
            (6, ch_24, 2, 2),
            (6, ch_32b, 3, 2),
            (6, ch_64, 4, 2),
            (6, ch_96, 3, 1),
            (6, ch_160, 3, 2),
            (6, ch_320, 1, 1),
        ]
        in_ch = ch_32
        for expand_ratio, out_ch, repeats, first_stride in cfg:
            for idx in range(repeats):
                stride = first_stride if idx == 0 else 1
                modules.append(
                    ChronosSpikingInvertedResidual(
                        in_ch,
                        out_ch,
                        stride,
                        expand_ratio,
                        step_mode=step_mode,
                        lif_impl=lif_impl,
                    )
                )
                in_ch = out_ch

        modules.extend(
            [
                layer.Conv2d(in_ch, ch_1280, kernel_size=1, bias=False),
                layer.BatchNorm2d(ch_1280),
                _make_lif_node(lif_impl),
                layer.AdaptiveAvgPool2d((1, 1)),
                layer.Flatten(),
                layer.Dropout(0.2),
                layer.Linear(ch_1280, num_classes, bias=False),
                _make_lif_node(lif_impl),
            ]
        )
        self.layer = nn.Sequential(*modules)
        functional.set_step_mode(self.layer, step_mode=step_mode)

    def forward(self, x: torch.Tensor):
        return self.layer(x)


class ChronosConvLSTMCellEager(nn.Module):
    """Single-step ConvLSTM cell body (Shi et al. 2015, no peephole)."""

    def __init__(self, in_ch: int, hidden_ch: int):
        super().__init__()
        self.hidden_ch = hidden_ch
        self.conv_x = nn.Conv2d(in_ch, 4 * hidden_ch, kernel_size=3, padding=1, bias=True)
        self.conv_h = nn.Conv2d(hidden_ch, 4 * hidden_ch, kernel_size=3, padding=1, bias=False)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor):
        xproj = self.conv_x(x_t)
        hproj = self.conv_h(h_prev)
        i, f, g, o = torch.chunk(xproj + hproj, 4, dim=1)
        c_t = torch.sigmoid(f) * c_prev + torch.sigmoid(i) * torch.tanh(g)
        h_t = torch.sigmoid(o) * torch.tanh(c_t)
        return h_t, c_t


class ChronosConvLSTM(nn.Module):
    """Step-form model (sequence-input analogue of the SNN step_mode="s"
    convention above): step() processes one timestep given explicit state
    and returns (output, *new_state); the T-loop lives in
    SequenceInputLoopWrapper/SequenceInputMultiStepWrapper, not here. State
    is a flat tuple of tensors (h0, c0, h1, c1, ...) rather than the SNN
    modules' hidden module-buffer state, since Kairos's FX passes (Phase 0's
    batching legality check, Phase C's annotation extension) need h/c
    visible as explicit graph values threaded across timesteps -- this is
    also exactly the state-I/O shape a single-step ONNX export needs.
    """

    def __init__(
        self,
        *,
        in_channels: int = 1,
        hidden_channels: int = 64,
        num_layers: int = 2,
        num_classes: int = 10,
        height: int = 64,
        width: int = 64,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.height = height
        self.width = width
        self.cells = nn.ModuleList(
            ChronosConvLSTMCellEager(in_channels if i == 0 else hidden_channels, hidden_channels)
            for i in range(num_layers)
        )
        self.head = nn.Conv2d(hidden_channels, num_classes, kernel_size=1, bias=True)

    def init_state(self, batch_size: int, device, dtype) -> tuple:
        state = []
        for _ in range(self.num_layers):
            state.append(torch.zeros(batch_size, self.hidden_channels, self.height, self.width, device=device, dtype=dtype))
            state.append(torch.zeros(batch_size, self.hidden_channels, self.height, self.width, device=device, dtype=dtype))
        return tuple(state)

    def step(self, x_t: torch.Tensor, *state: torch.Tensor) -> tuple:
        layer_input = x_t
        new_state = []
        for layer_idx, cell in enumerate(self.cells):
            h_prev, c_prev = state[2 * layer_idx], state[2 * layer_idx + 1]
            h_t, c_t = cell(layer_input, h_prev, c_prev)
            new_state.append(h_t)
            new_state.append(c_t)
            layer_input = h_t
        y_t = self.head(layer_input)
        return (y_t, *new_state)


class ChronosSpikeTransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        heads: int,
        mlp_ratio: int = 4,
        lif_impl: str = "chronos",
    ):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"transformer dim={dim} must be divisible by heads={heads}")
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = self.dim // self.heads
        self.scale = self.head_dim ** -0.5
        mlp_dim = self.dim * int(mlp_ratio)

        self.norm1 = nn.LayerNorm(self.dim)
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=False)
        self.proj = nn.Linear(self.dim, self.dim, bias=False)
        self.attn_if = _make_lif_node(lif_impl)

        self.norm2 = nn.LayerNorm(self.dim)
        self.fc1 = nn.Linear(self.dim, mlp_dim, bias=False)
        self.fc1_if = _make_lif_node(lif_impl)
        self.fc2 = nn.Linear(mlp_dim, self.dim, bias=False)
        self.fc2_if = _make_lif_node(lif_impl)
        self.mlp_res_if = _make_lif_node(lif_impl)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.norm1(x)
        qkv = self.qkv(y)
        leading_shape = qkv.shape[:-2]
        seq_len = qkv.shape[-2]
        qkv = qkv.reshape(*leading_shape, seq_len, 3, self.heads, self.head_dim)
        qkv = qkv.movedim(-3, 0).transpose(-3, -2)
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        y = torch.matmul(attn, v)
        y = y.transpose(-3, -2).reshape(*leading_shape, seq_len, self.dim)
        y = self.proj(y)
        x = self.attn_if(residual + y)

        residual = x
        y = self.norm2(x)
        y = self.fc1_if(self.fc1(y))
        y = self.fc2_if(self.fc2(y))
        x = self.mlp_res_if(residual + y)
        return x


class ChronosMambaBlockEager(nn.Module):
    """Single-step minimal Mamba block (selective SSM), random init, no
    pretrained weights -- pure performance test. State (conv_state,
    ssm_state) is threaded explicitly by the outer T-loop in ChronosMamba,
    matching ChronosConvLSTMCellEager's convention above.

    conv_state is carried as the full d_conv-wide FIFO window (matching the
    "# FIFO [B,1536,4]" comment in the canonical spec literally): each step
    drops the oldest tap and appends the new x, keeping the window at
    d_conv elements before and after the update, rather than a d_conv-1
    "conceptual carry" that would require reconstructing the 4th tap
    on every entry -- self-consistent with the update line
    conv_state = cat(conv_state[:,:,1:], x) being 4-wide on both sides.
    """

    def __init__(self, d_model: int, d_inner: int, d_state: int, d_conv: int, dt_rank: int):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = dt_rank

        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv1d_w = nn.Parameter(torch.randn(d_inner, d_conv) * 0.05)
        self.conv1d_b = nn.Parameter(torch.zeros(d_inner))
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        # S4D-real-style negative init keeps exp(dt*A) bounded even though
        # this is a random-init performance test, not a trained checkpoint.
        self.A = nn.Parameter(-torch.rand(d_inner, d_state) - 0.5)
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x_t, conv_state, ssm_state):
        xz = self.in_proj(x_t)
        x, z = torch.chunk(xz, 2, dim=-1)

        conv_state = torch.cat([conv_state[:, :, 1:], x.unsqueeze(-1)], dim=-1)
        x = F.silu((conv_state * self.conv1d_w).sum(-1) + self.conv1d_b)

        dt_bc = self.x_proj(x)
        dt_in, B_ssm, C_ssm = torch.split(dt_bc, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt_in))

        hA = torch.exp(dt.unsqueeze(-1) * self.A)
        ssm_state = hA * ssm_state + (dt.unsqueeze(-1) * B_ssm.unsqueeze(1)) * x.unsqueeze(-1)
        y = (ssm_state * C_ssm.unsqueeze(1)).sum(-1) + self.D * x

        out = self.out_proj(y * F.silu(z))
        return out, conv_state, ssm_state


class ChronosMamba(nn.Module):
    def __init__(
        self,
        *,
        d_model: int = 768,
        n_layer: int = 24,
        d_inner: int = 1536,
        d_state: int = 16,
        d_conv: int = 4,
        dt_rank: int = 48,
        num_classes: int = 10,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_state = d_state
        self.d_conv = d_conv
        self.n_layer = n_layer
        self.blocks = nn.ModuleList(
            ChronosMambaBlockEager(d_model, d_inner, d_state, d_conv, dt_rank) for _ in range(n_layer)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(n_layer))
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes, bias=False)

    def init_state(self, batch_size: int, device, dtype) -> tuple:
        state = []
        for _ in range(self.n_layer):
            state.append(torch.zeros(batch_size, self.d_inner, self.d_conv, device=device, dtype=dtype))
            state.append(torch.zeros(batch_size, self.d_inner, self.d_state, device=device, dtype=dtype))
        return tuple(state)

    def step(self, x_t: torch.Tensor, *state: torch.Tensor) -> tuple:
        # x_t: [B, d_model] -- already-embedded input; embedding/lm_head
        # deliberately do not enter the temporal region (per spec).
        h = x_t
        new_state = []
        for layer_idx in range(self.n_layer):
            conv_state, ssm_state = state[2 * layer_idx], state[2 * layer_idx + 1]
            residual = h
            y = self.norms[layer_idx](h)
            y, conv_state, ssm_state = self.blocks[layer_idx](y, conv_state, ssm_state)
            h = residual + y
            new_state.append(conv_state)
            new_state.append(ssm_state)
        y_t = self.head(self.final_norm(h))
        return (y_t, *new_state)


class ChronosSpikeTransformer(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = 768,
        dim: int = 256,
        depth: int = 8,
        heads: int = 8,
        num_classes: int = 100,
        lif_impl: str = "chronos",
        step_mode: str = "s",
    ):
        super().__init__()
        self.step_mode = step_mode
        self.input_proj = nn.Linear(int(input_dim), int(dim), bias=False)
        self.input_if = _make_lif_node(lif_impl)
        self.blocks = nn.ModuleList(
            [
                ChronosSpikeTransformerBlock(dim=int(dim), heads=int(heads), lif_impl=lif_impl)
                for _ in range(int(depth))
            ]
        )
        self.norm = nn.LayerNorm(int(dim))
        self.classifier = nn.Linear(int(dim), int(num_classes), bias=False)
        functional.set_step_mode(self.input_if, step_mode=step_mode)
        functional.set_step_mode(self.blocks, step_mode=step_mode)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_if(self.input_proj(x))
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x.mean(dim=-2)
        return self.classifier(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)


class ChronosGRUCellEager(nn.Module):
    """Single-step GRUCell body, PyTorch GRUCell gate order [r|z|n]. State
    (h) threaded explicitly by the outer T-loop in ChronosDeepSpeech2,
    matching ChronosConvLSTMCellEager / ChronosMambaBlockEager above.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.w_x = nn.Linear(input_size, 3 * hidden_size, bias=True)
        self.w_h = nn.Linear(hidden_size, 3 * hidden_size, bias=True)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        xproj = self.w_x(x_t)
        hproj = self.w_h(h_prev)
        xproj_r, xproj_z, xproj_n = torch.chunk(xproj, 3, dim=-1)
        hproj_r, hproj_z, hproj_n = torch.chunk(hproj, 3, dim=-1)
        r = torch.sigmoid(xproj_r + hproj_r)
        z = torch.sigmoid(xproj_z + hproj_z)
        n = torch.tanh(xproj_n + r * hproj_n)
        h_t = (1 - z) * n + z * h_prev
        return h_t


class ChronosDeepSpeech2(nn.Module):
    """conv frontend (2 layers, executed as a single full-batch call outside
    the temporal region -- not inside the timestep loop) -> 3-layer
    unidirectional GRU (hidden=800) -> FC(29). T_in (spectrogram time bins)
    is derived from the target T so the conv frontend's time-stride math
    lands exactly on T output frames: with stride_t=(2,1) across the two
    conv layers and "same"-style frequency/time padding, T_in = 2*T is
    exact (see build_model_input's deepspeech2 case).
    """

    def __init__(
        self,
        *,
        freq_bins: int = 161,
        conv_channels: int = 32,
        gru_hidden: int = 800,
        gru_layers: int = 3,
        num_classes: int = 29,
    ):
        super().__init__()
        self.freq_bins = freq_bins
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers

        self.conv1 = nn.Conv2d(1, conv_channels, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5))
        self.bn1 = nn.BatchNorm2d(conv_channels)
        self.conv2 = nn.Conv2d(conv_channels, conv_channels, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5))
        self.bn2 = nn.BatchNorm2d(conv_channels)

        freq_out = (freq_bins + 2 * 20 - 41) // 2 + 1
        freq_out = (freq_out + 2 * 10 - 21) // 2 + 1
        gru_input_size = conv_channels * freq_out

        self.gru_cells = nn.ModuleList(
            ChronosGRUCellEager(gru_input_size if i == 0 else gru_hidden, gru_hidden) for i in range(gru_layers)
        )
        self.fc = nn.Linear(gru_hidden, num_classes, bias=True)

    def frontend(self, spectrogram: torch.Tensor) -> torch.Tensor:
        # spectrogram: [B, 1, freq_bins, T_in] -- full-batch conv frontend,
        # deliberately outside the timestep loop / temporal region (per
        # spec). SequenceInputLoopWrapper calls this once before the T-loop
        # (mirroring "循环外整批执行"); models without a frontend (ConvLSTM,
        # Mamba) are given their x_seq directly and skip this hook.
        feat = F.relu(self.bn1(self.conv1(spectrogram)))
        feat = F.relu(self.bn2(self.conv2(feat)))
        B, C, Freq, T = feat.shape
        return feat.permute(3, 0, 1, 2).reshape(T, B, C * Freq)

    def init_state(self, batch_size: int, device, dtype) -> tuple:
        return tuple(
            torch.zeros(batch_size, self.gru_hidden, device=device, dtype=dtype) for _ in range(self.gru_layers)
        )

    def step(self, x_t: torch.Tensor, *state: torch.Tensor) -> tuple:
        layer_input = x_t
        new_state = []
        for layer_idx, cell in enumerate(self.gru_cells):
            h_t = cell(layer_input, state[layer_idx])
            new_state.append(h_t)
            layer_input = h_t
        y_t = self.fc(layer_input)
        return (y_t, *new_state)


class ChronosSpikeBERT(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int = 30522,
        dim: int = 256,
        depth: int = 8,
        heads: int = 8,
        num_classes: int = 100,
        lif_impl: str = "chronos",
        step_mode: str = "s",
    ):
        super().__init__()
        self.step_mode = step_mode
        self.embedding = nn.Embedding(int(vocab_size), int(dim))
        self.embedding_if = _make_lif_node(lif_impl)
        self.blocks = nn.ModuleList(
            [
                ChronosSpikeTransformerBlock(dim=int(dim), heads=int(heads), lif_impl=lif_impl)
                for _ in range(int(depth))
            ]
        )
        self.norm = nn.LayerNorm(int(dim))
        self.classifier = nn.Linear(int(dim), int(num_classes), bias=False)
        functional.set_step_mode(self.embedding_if, step_mode=step_mode)
        functional.set_step_mode(self.blocks, step_mode=step_mode)

    def _forward_impl(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding_if(self.embedding(token_ids))
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x.mean(dim=-2)
        return self.classifier(x)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(token_ids)


def make_model_input(model_name: str, args, dtype: torch.dtype) -> torch.Tensor:
    if model_name == "spiketransformer":
        return torch.randn(
            args.batch_size,
            args.sequence_length,
            args.transformer_input_dim,
            device=args.device,
            dtype=dtype,
        )
    if model_name == "spikebert":
        return torch.randint(
            low=0,
            high=args.transformer_vocab_size,
            size=(args.batch_size, args.sequence_length),
            device=args.device,
            dtype=torch.int64,
        )
    if model_name == "convlstm":
        # [T, B, C, H, W], independently random per timestep -- a genuine
        # sequence, unlike the repeated-same-frame convention below.
        return torch.randn(
            args.T,
            args.batch_size,
            args.convlstm_in_channels,
            args.convlstm_height,
            args.convlstm_width,
            device=args.device,
            dtype=dtype,
        )
    if model_name == "mamba":
        # [T, B, d_model] -- already-embedded per spec (embedding not in the
        # temporal region).
        return torch.randn(
            args.T,
            args.batch_size,
            args.mamba_d_model,
            device=args.device,
            dtype=dtype,
        )
    if model_name == "deepspeech2":
        # [B, 1, freq_bins, T_in]; T_in = 2*T is exact for the conv
        # frontend's stride math (see ChronosDeepSpeech2.frontend).
        t_in = 2 * args.T
        return torch.randn(
            args.batch_size,
            1,
            args.deepspeech2_freq_bins,
            t_in,
            device=args.device,
            dtype=dtype,
        )
    return torch.randn(args.batch_size, 3, args.height, args.width, device=args.device, dtype=dtype)


def _make_vgg_layer(model_name: str, step_mode: str, lif_impl: str = "chronos") -> nn.Module:
    spiking_neuron = _lif_node_class(lif_impl)
    if model_name == "vgg11":
        layer = spiking_vgg.spiking_vgg11_bn(
            pretrained=False,
            num_classes=10,
            spiking_neuron=spiking_neuron,
            surrogate_function=surrogate.ATan(),
        )
    elif model_name == "vgg16":
        layer = spiking_vgg.spiking_vgg16_bn(
            pretrained=False,
            num_classes=10,
            spiking_neuron=spiking_neuron,
            surrogate_function=surrogate.ATan(),
        )
    else:
        raise ValueError(f"unsupported VGG model: {model_name}")
    functional.set_step_mode(layer, step_mode=step_mode)
    return layer


def make_resnet_layer(
    model_name: str,
    allow_resnet32_fallback: bool,
    step_mode: str = "s",
    model_channels: int = 64,
    lif_impl: str = "chronos",
    sequence_length: int = 256,
    transformer_depth: int = 8,
    transformer_dim: int = 256,
    transformer_heads: int = 8,
    transformer_input_dim: int = 768,
    transformer_vocab_size: int = 30522,
    transformer_num_classes: int = 100,
    convlstm_in_channels: int = 1,
    convlstm_hidden_channels: int = 64,
    convlstm_num_layers: int = 2,
    convlstm_height: int = 64,
    convlstm_width: int = 64,
    mamba_d_model: int = 768,
    mamba_n_layer: int = 24,
    mamba_d_inner: int = 1536,
    mamba_d_state: int = 16,
    mamba_d_conv: int = 4,
    mamba_dt_rank: int = 48,
    deepspeech2_freq_bins: int = 161,
    deepspeech2_conv_channels: int = 32,
    deepspeech2_gru_hidden: int = 800,
    deepspeech2_gru_layers: int = 3,
) -> nn.Module:
    # step_mode is a spikingjelly concept (stateful module buffers) that
    # doesn't apply to these explicit-state, step()/init_state()-form
    # models -- the wrapper (SequenceInputLoopWrapper /
    # SequenceInputMultiStepWrapper), not step_mode, decides how they're
    # driven, so the same construction serves both "s" and "m" call sites.
    if model_name == "convlstm":
        return ChronosConvLSTM(
            in_channels=convlstm_in_channels,
            hidden_channels=convlstm_hidden_channels,
            num_layers=convlstm_num_layers,
            num_classes=10,
            height=convlstm_height,
            width=convlstm_width,
        )
    if model_name == "mamba":
        return ChronosMamba(
            d_model=mamba_d_model,
            n_layer=mamba_n_layer,
            d_inner=mamba_d_inner,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            dt_rank=mamba_dt_rank,
            num_classes=10,
        )
    if model_name == "deepspeech2":
        return ChronosDeepSpeech2(
            freq_bins=deepspeech2_freq_bins,
            conv_channels=deepspeech2_conv_channels,
            gru_hidden=deepspeech2_gru_hidden,
            gru_layers=deepspeech2_gru_layers,
            num_classes=29,
        )

    spiking_neuron = _lif_node_class(lif_impl)
    if model_name == "resnet18":
        layer = spiking_resnet.spiking_resnet18(
            pretrained=False,
            spiking_neuron=spiking_neuron,
            surrogate_function=surrogate.ATan(),
        )
    elif model_name in ("resnet34", "resnet32"):
        if model_name == "resnet32":
            print("[WARN] resnet32 is deprecated typo; using spiking_resnet34 instead.")
        layer = spiking_resnet.spiking_resnet34(
            pretrained=False,
            spiking_neuron=spiking_neuron,
            surrogate_function=surrogate.ATan(),
        )
    elif model_name in ("alexnet", "zfnet"):
        return ChronosAlexZFNet(
            kind=model_name,
            channels=model_channels,
            num_classes=10,
            step_mode=step_mode,
            lif_impl=lif_impl,
        )
    elif model_name == "mobilenetv1":
        return ChronosMobileNetV1(
            channels=model_channels,
            num_classes=10,
            step_mode=step_mode,
            lif_impl=lif_impl,
        )
    elif model_name == "mobilenetv2":
        return ChronosMobileNetV2(
            channels=model_channels,
            num_classes=10,
            step_mode=step_mode,
            lif_impl=lif_impl,
        )
    elif model_name == "spiketransformer":
        return ChronosSpikeTransformer(
            input_dim=transformer_input_dim,
            dim=transformer_dim,
            depth=transformer_depth,
            heads=transformer_heads,
            num_classes=transformer_num_classes,
            step_mode=step_mode,
            lif_impl=lif_impl,
        )
    elif model_name == "spikebert":
        return ChronosSpikeBERT(
            vocab_size=transformer_vocab_size,
            dim=transformer_dim,
            depth=transformer_depth,
            heads=transformer_heads,
            num_classes=transformer_num_classes,
            step_mode=step_mode,
            lif_impl=lif_impl,
        )
    elif model_name in ("vgg11", "vgg16"):
        return _make_vgg_layer(model_name, step_mode, lif_impl=lif_impl)
    else:
        raise ValueError(f"unsupported model: {model_name}")

    functional.set_step_mode(layer, step_mode=step_mode)
    return layer


def build_placeholder_values(gm: torch.fx.GraphModule, example_inputs) -> Dict[torch.fx.Node, Any]:
    placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
    return {node: value for node, value in zip(placeholders, example_inputs)}


def save_graph_files(gm: torch.fx.GraphModule, out_dir: Path, prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{prefix}_fx.py").write_text(gm.code, encoding="utf-8")
    (out_dir / f"{prefix}_fx.txt").write_text(str(gm.graph), encoding="utf-8")


def inductor_options_from_compile_kwargs(compile_kwargs: Dict[str, Any]):
    options = compile_kwargs.get("options")
    if options is None and compile_kwargs.get("mode") == "reduce-overhead":
        options = {"triton.cudagraphs": True}
    return options


def make_rewrite_backend(args, graph_dir: Path, counters: RewriteCounters):
    def backend(gm: torch.fx.GraphModule, example_inputs, **compile_kwargs):
        graph_idx = counters.captured_graphs
        counters.captured_graphs += 1
        local_dir = graph_dir if graph_idx == 0 else graph_dir / f"graph_{graph_idx}"
        local_dir.mkdir(parents=True, exist_ok=True)
        save_graph_files(gm, local_dir, "original")

        placeholder_values = build_placeholder_values(gm, example_inputs)
        try:
            FakeTensorProp(gm).propagate(*example_inputs)
        except Exception as exc:
            print(f"[FX_SHAPE_PROP][SKIP] {type(exc).__name__}: {exc}")
        lif_state_count = count_lif_state_nodes(gm)
        temporal_replaced_patterns = 0
        temporal_log: List[str] = []

        annotation_window = args.temporal_schedule_window or args.temporal_fuse_window
        annotation_stats = annotate_temporal_metadata(
            gm,
            annotation_window,
            args.T,
            strict=False,
        )
        counters.temporal_annotated_nodes += annotation_stats.temporal_annotated_nodes
        counters.temporal_annotation_missing += annotation_stats.temporal_annotation_missing
        for role, count in annotation_stats.temporal_annotation_roles.items():
            counters.temporal_annotation_roles[role] = counters.temporal_annotation_roles.get(role, 0) + count
        for window_id, count in annotation_stats.temporal_annotation_windows.items():
            counters.temporal_annotation_windows[window_id] = counters.temporal_annotation_windows.get(window_id, 0) + count
        for reason, count in annotation_stats.temporal_annotation_reasons.items():
            counters.temporal_annotation_reasons[reason] = (
                counters.temporal_annotation_reasons.get(reason, 0) + count
            )
        print(
            f"[TEMPORAL_ANNOTATION] annotated={annotation_stats.temporal_annotated_nodes} "
            f"missing={annotation_stats.temporal_annotation_missing} "
            f"roles={annotation_stats.temporal_annotation_roles}"
        )

        # Kairos workload rewrites (ConvLSTM/Mamba/DeepSpeech2-GRU). Each is
        # independently toggleable and, per annotate_temporal_metadata's
        # mutual-exclusivity note, their pattern collectors only ever match
        # one of these three workloads' own gate chains -- so on every
        # existing SNN model this whole block finds nothing and silently
        # no-ops, exactly like the LIF-specific passes below no-op on these
        # three new workloads.
        if not args.disable_rewrite:
            if not args.disable_convlstm_rewrite:
                convlstm_replaced = rewrite_convlstm_cell_to_fused(gm)
                counters.convlstm_replaced_patterns += convlstm_replaced
                temporal_replaced_patterns += convlstm_replaced

            if not args.disable_gru_rewrite:
                gru_replaced = rewrite_gru_cell_to_fused(gm)
                counters.gru_replaced_patterns += gru_replaced
                temporal_replaced_patterns += gru_replaced

            if not args.disable_mamba_rewrite and args.temporal_fuse_window > 1:
                mamba_patterns = collect_mamba_scan_patterns(gm)
                if mamba_patterns:
                    mamba_schedule_window = args.temporal_schedule_window or args.temporal_fuse_window
                    mamba_schedule_result = reorder_fx_graph_by_temporal_windows(
                        gm,
                        args.T,
                        mamba_schedule_window,
                        mamba_patterns,
                        dump_dir=local_dir if args.temporal_schedule_dump else None,
                        strict=args.temporal_schedule_strict,
                    )
                    counters.mamba_schedule_ok = mamba_schedule_result.ok
                    counters.mamba_schedule_reason = mamba_schedule_result.reason
                    if not mamba_schedule_result.ok:
                        if args.temporal_schedule_strict:
                            raise RuntimeError(mamba_schedule_result.reason)
                        print(f"[MAMBA_SCHEDULE][FALLBACK] {mamba_schedule_result.reason}")
                    else:
                        mamba_replaced = rewrite_mamba_scan_to_fused(gm, args.temporal_fuse_window)
                        counters.mamba_replaced_patterns += mamba_replaced
                        temporal_replaced_patterns += mamba_replaced

            if counters.convlstm_replaced_patterns or counters.gru_replaced_patterns or counters.mamba_replaced_patterns:
                print(
                    f"[KAIROS_REWRITE] convlstm={counters.convlstm_replaced_patterns} "
                    f"gru={counters.gru_replaced_patterns} mamba={counters.mamba_replaced_patterns} "
                    f"mamba_schedule_ok={counters.mamba_schedule_ok}"
                )

        temporal_patterns = collect_conv_bn_lif_state_patterns(gm) if not args.disable_conv_bn_lif else []
        residual_patterns = collect_conv_bn_add_lif_state_patterns(gm) if not args.disable_conv_bn_lif else []
        if args.enable_temporal_schedule and temporal_patterns:
            schedule_window = args.temporal_schedule_window or args.temporal_fuse_window
            schedule_result = reorder_fx_graph_by_temporal_windows(
                gm,
                args.T,
                schedule_window,
                temporal_patterns,
                dump_dir=local_dir if args.temporal_schedule_dump else None,
                strict=args.temporal_schedule_strict,
            )
            counters.temporal_schedule_ok = schedule_result.ok
            counters.temporal_schedule_windows += schedule_result.scheduled_windows
            counters.temporal_schedule_moved_nodes += schedule_result.moved_nodes
            counters.temporal_schedule_reason = schedule_result.reason
            if schedule_result.ok:
                temporal_patterns = collect_conv_bn_lif_state_patterns(gm)
                residual_patterns = collect_conv_bn_add_lif_state_patterns(gm)
            elif args.temporal_schedule_strict:
                raise RuntimeError(schedule_result.reason)
            else:
                print(f"[SCHEDULE][FALLBACK] {schedule_result.reason}")

        if args.enable_temporal_rewrite and args.temporal_fuse_window > 1 and not args.disable_conv_bn_lif:
            temporal_groups = group_temporal_patterns(temporal_patterns)
            temporal_windows = make_temporal_windows(
                temporal_groups,
                args.temporal_fuse_window,
                args.temporal_allow_tail,
            )
            dump_temporal_patterns(temporal_groups, local_dir / "temporal_patterns.txt")
            dump_temporal_windows(temporal_windows, local_dir / "temporal_windows.txt")
            counters.temporal_groups += len(temporal_groups)
            counters.temporal_windows += len(temporal_windows)
            if args.disable_rewrite:
                temporal_log.append("SKIP: --disable-rewrite enabled")
            else:
                temporal_stats = rewrite_temporal_conv_bn_lif_state_to_fused(
                    gm,
                    temporal_windows,
                    placeholder_values,
                    args.max_patterns,
                )
                temporal_replaced_patterns = temporal_stats.temporal_replaced_patterns
                temporal_log.extend(temporal_stats.log)
                counters.temporal_replaced_windows += temporal_stats.temporal_replaced_windows
                counters.temporal_replaced_patterns += temporal_stats.temporal_replaced_patterns
                counters.temporal_skipped_windows += temporal_stats.temporal_skipped_windows

            residual_patterns = collect_conv_bn_add_lif_state_patterns(gm)
            residual_groups = group_temporal_residual_patterns(residual_patterns)
            residual_windows = make_temporal_residual_windows(
                residual_groups,
                args.temporal_fuse_window,
                args.temporal_allow_tail,
            )
            counters.temporal_residual_groups += len(residual_groups)
            counters.temporal_residual_windows += len(residual_windows)
            counters.temporal_residual_total_windows += len(residual_windows)
            if not args.disable_rewrite and residual_windows:
                residual_stats = rewrite_temporal_conv_bn_add_lif_state_to_fused(
                    gm,
                    residual_windows,
                    placeholder_values,
                    max(0, args.max_patterns - temporal_replaced_patterns),
                )
                temporal_replaced_patterns += residual_stats.temporal_residual_replaced_patterns
                counters.temporal_residual_replaced_windows += residual_stats.temporal_residual_replaced_windows
                counters.temporal_residual_rewritten_windows += residual_stats.temporal_residual_replaced_windows
                counters.temporal_residual_replaced_patterns += residual_stats.temporal_residual_replaced_patterns
                counters.temporal_residual_skipped_windows += residual_stats.temporal_residual_skipped_windows
                counters.temporal_residual_remapped_spike_external_users += (
                    residual_stats.temporal_residual_remapped_spike_external_users
                )
                counters.temporal_residual_unremappable_external_users += (
                    residual_stats.temporal_residual_unremappable_external_users
                )
                for reason, count in residual_stats.residual_fuse_skip_reasons.items():
                    counters.temporal_residual_skip_reasons[reason] = (
                        counters.temporal_residual_skip_reasons.get(reason, 0) + count
                    )
                    counters.residual_fuse_skip_reasons[reason] = (
                        counters.residual_fuse_skip_reasons.get(reason, 0) + count
                )
                temporal_log.extend(residual_stats.log)

            if not args.disable_temporal_linear_lif_rewrite:
                linear_lif_patterns = collect_temporal_linear_lif_state_patterns(gm)
                linear_lif_groups = group_temporal_linear_lif_patterns(linear_lif_patterns)
                linear_lif_windows = make_temporal_linear_lif_windows(
                    linear_lif_groups,
                    args.temporal_fuse_window,
                    args.temporal_allow_tail,
                )
                dump_temporal_linear_lif_patterns(linear_lif_groups, local_dir / "temporal_linear_lif_patterns.txt")
                dump_temporal_linear_lif_windows(linear_lif_windows, local_dir / "temporal_linear_lif_windows.txt")
                counters.linear_lif_patterns += len(linear_lif_patterns)
                counters.temporal_linear_lif_windows += len(linear_lif_windows)
                counters.temporal_linear_lif_total_windows += len(linear_lif_windows)
                if not args.disable_rewrite and linear_lif_windows:
                    linear_lif_stats = rewrite_temporal_linear_lif_state_to_fused(
                        gm,
                        linear_lif_windows,
                        max(0, args.max_patterns - temporal_replaced_patterns),
                    )
                    temporal_replaced_patterns += linear_lif_stats.temporal_linear_lif_replaced_patterns
                    counters.temporal_linear_lif_rewritten_windows += (
                        linear_lif_stats.temporal_linear_lif_rewritten_windows
                    )
                    counters.temporal_linear_lif_replaced_patterns += (
                        linear_lif_stats.temporal_linear_lif_replaced_patterns
                    )
                    counters.temporal_linear_lif_skipped_windows += (
                        linear_lif_stats.temporal_linear_lif_skipped_windows
                    )
                    counters.temporal_linear_lif_remapped_spike_external_users += (
                        linear_lif_stats.temporal_linear_lif_remapped_spike_external_users
                    )
                    counters.temporal_linear_lif_unremappable_external_users += (
                        linear_lif_stats.temporal_linear_lif_unremappable_external_users
                    )
                    for reason, count in linear_lif_stats.temporal_linear_lif_skip_reasons.items():
                        counters.temporal_linear_lif_skip_reasons[reason] = (
                            counters.temporal_linear_lif_skip_reasons.get(reason, 0) + count
                        )
                    temporal_log.extend(linear_lif_stats.log)

            if not args.disable_temporal_lif_avgpool_linear_rewrite:
                avgpool_linear_patterns = collect_temporal_lif_avgpool_linear_patterns(gm)
                avgpool_linear_groups = group_temporal_lif_avgpool_linear_patterns(avgpool_linear_patterns)
                avgpool_linear_windows = make_temporal_lif_avgpool_linear_windows(
                    avgpool_linear_groups,
                    args.temporal_fuse_window,
                    args.temporal_allow_tail,
                )
                dump_temporal_lif_avgpool_linear_patterns(avgpool_linear_groups, local_dir / "temporal_lif_avgpool_linear_patterns.txt")
                dump_temporal_lif_avgpool_linear_windows(avgpool_linear_windows, local_dir / "temporal_lif_avgpool_linear_windows.txt")
                counters.temporal_lif_avgpool_linear_windows += len(avgpool_linear_windows)
                counters.temporal_lif_avgpool_linear_total_windows += len(avgpool_linear_windows)
                counters.temporal_lif_tail_windows += len(avgpool_linear_windows)
                counters.temporal_lif_tail_total_windows += len(avgpool_linear_windows)
                if not args.disable_rewrite and avgpool_linear_windows:
                    avgpool_linear_stats = rewrite_temporal_lif_avgpool_linear_to_fused(
                        gm,
                        avgpool_linear_windows,
                        max(0, args.max_patterns - temporal_replaced_patterns),
                    )
                    temporal_replaced_patterns += avgpool_linear_stats.temporal_lif_avgpool_linear_replaced_patterns
                    counters.temporal_lif_avgpool_linear_rewritten_windows += avgpool_linear_stats.temporal_lif_avgpool_linear_rewritten_windows
                    counters.temporal_lif_avgpool_linear_replaced_patterns += avgpool_linear_stats.temporal_lif_avgpool_linear_replaced_patterns
                    counters.temporal_lif_avgpool_linear_skipped_windows += avgpool_linear_stats.temporal_lif_avgpool_linear_skipped_windows
                    counters.temporal_lif_tail_rewritten_windows += avgpool_linear_stats.temporal_lif_avgpool_linear_rewritten_windows
                    counters.temporal_lif_tail_replaced_patterns += avgpool_linear_stats.temporal_lif_avgpool_linear_replaced_patterns
                    counters.temporal_lif_tail_skipped_windows += avgpool_linear_stats.temporal_lif_avgpool_linear_skipped_windows
                    for reason, count in avgpool_linear_stats.temporal_lif_avgpool_linear_skip_reasons.items():
                        counters.temporal_lif_avgpool_linear_skip_reasons[reason] = (
                            counters.temporal_lif_avgpool_linear_skip_reasons.get(reason, 0) + count
                        )
                        counters.temporal_lif_tail_skip_reasons[reason] = (
                            counters.temporal_lif_tail_skip_reasons.get(reason, 0) + count
                        )
                    temporal_log.extend(avgpool_linear_stats.log)

            if not args.disable_temporal_lif_rewrite:
                lif_patterns = collect_standalone_lif_state_patterns(gm)
                lif_groups = group_temporal_lif_patterns(lif_patterns)
                lif_windows = make_temporal_lif_windows(
                    lif_groups,
                    args.temporal_fuse_window,
                    args.temporal_allow_tail,
                )
                counters.temporal_lif_windows += len(lif_windows)
                counters.temporal_lif_total_windows += len(lif_windows)
                if not args.disable_rewrite and lif_windows:
                    lif_stats = rewrite_temporal_lif_state_to_fused(
                        gm,
                        lif_windows,
                        max(0, args.max_patterns - temporal_replaced_patterns),
                    )
                    temporal_replaced_patterns += lif_stats.temporal_lif_replaced_patterns
                    counters.temporal_lif_rewritten_windows += lif_stats.temporal_lif_rewritten_windows
                    counters.temporal_lif_replaced_patterns += lif_stats.temporal_lif_replaced_patterns
                    counters.temporal_lif_skipped_windows += lif_stats.temporal_lif_skipped_windows
                    counters.temporal_lif_remapped_spike_external_users += (
                        lif_stats.temporal_lif_remapped_spike_external_users
                    )
                    counters.temporal_lif_unremappable_external_users += (
                        lif_stats.temporal_lif_unremappable_external_users
                    )
                    for reason, count in lif_stats.temporal_lif_skip_reasons.items():
                        counters.temporal_lif_skip_reasons[reason] = (
                            counters.temporal_lif_skip_reasons.get(reason, 0) + count
                        )
                    temporal_log.extend(lif_stats.log)
            dump_temporal_rewrite_log(temporal_log, local_dir / "temporal_rewrite_log.txt")

        direct_matches = match_conv_lif_state(gm)
        conv_bn_matches = []
        if not args.disable_conv_bn_lif:
            conv_bn_matches = match_conv_bn_lif_state(gm)

        direct_replaced = 0
        conv_bn_replaced = 0
        if not args.disable_rewrite:
            remaining = max(0, args.max_patterns - temporal_replaced_patterns)
            conv_bn_replaced = rewrite_conv_bn_lif_state_to_fused(
                gm,
                conv_bn_matches,
                placeholder_values,
                remaining,
            )
            remaining = max(0, remaining - conv_bn_replaced)
            direct_replaced = rewrite_conv_lif_state_to_fused(
                gm,
                direct_matches,
                placeholder_values,
                remaining,
            )
        else:
            gm.graph.lint()
            gm.recompile()

        if args.enable_spatial_batching and not args.disable_rewrite:
            try:
                spatial_window = args.temporal_schedule_window or args.temporal_fuse_window
                spatial_stats = apply_spatial_batching(
                    gm,
                    spatial_window,
                    args.spatial_batching_ops,
                    dump_dir=local_dir if args.spatial_batching_dump else None,
                    strict=args.spatial_batching_strict,
                    enable_chain=False,
                )
                counters.spatial_batch_groups += spatial_stats.spatial_batch_groups
                counters.spatial_batched_ops += spatial_stats.spatial_batched_ops
                counters.spatial_batch_chains += spatial_stats.spatial_batch_chains
                counters.spatial_chain_groups += spatial_stats.spatial_chain_groups
                counters.spatial_cat_eliminated += spatial_stats.spatial_cat_eliminated
                counters.spatial_chunk_eliminated += spatial_stats.spatial_chunk_eliminated
                counters.spatial_batched_conv += spatial_stats.spatial_batched_conv
                counters.spatial_batched_bn += spatial_stats.spatial_batched_bn
                counters.spatial_batched_add += spatial_stats.spatial_batched_add
                counters.spatial_batched_pool += spatial_stats.spatial_batched_pool
                counters.spatial_batched_maxpool += spatial_stats.spatial_batched_maxpool
                counters.spatial_batched_avgpool += spatial_stats.spatial_batched_avgpool
                counters.spatial_batched_adaptive_avgpool += spatial_stats.spatial_batched_adaptive_avgpool
                counters.spatial_batched_flatten += spatial_stats.spatial_batched_flatten
                counters.spatial_batched_linear += spatial_stats.spatial_batched_linear
                counters.spatial_batched_elementwise += spatial_stats.spatial_batched_elementwise
                counters.spatial_batched_layer_norm += spatial_stats.spatial_batched_layer_norm
                counters.spatial_batched_attention += spatial_stats.spatial_batched_attention
                counters.spatial_temporal_stack_bn_groups += spatial_stats.spatial_temporal_stack_bn_groups
                counters.spatial_temporal_stack_add_groups += spatial_stats.spatial_temporal_stack_add_groups
                counters.spatial_temporal_stack_pool_groups += spatial_stats.spatial_temporal_stack_pool_groups
                counters.spatial_temporal_stack_flatten_groups += spatial_stats.spatial_temporal_stack_flatten_groups
                counters.spatial_temporal_stack_linear_groups += spatial_stats.spatial_temporal_stack_linear_groups
                counters.spatial_temporal_stack_groups += spatial_stats.spatial_temporal_stack_groups
                counters.spatial_temporal_stack_flatten_inputs += spatial_stats.spatial_temporal_stack_flatten_inputs
                counters.spatial_cat_avoided_by_temporal_stack_flatten += (
                    spatial_stats.spatial_cat_avoided_by_temporal_stack_flatten
                )
                counters.spatial_previous_batched_groups += spatial_stats.spatial_previous_batched_groups
                counters.spatial_reused_previous_batched_inputs += spatial_stats.spatial_reused_previous_batched_inputs
                counters.spatial_chunk_cat_avoided += spatial_stats.spatial_chunk_cat_avoided
                counters.spatial_batch_skipped += spatial_stats.spatial_batch_skipped
                for reason, count in spatial_stats.reasons.items():
                    counters.spatial_batch_reasons[reason] = (
                        counters.spatial_batch_reasons.get(reason, 0) + count
                    )
            except Exception:
                if args.spatial_batching_strict:
                    raise
                print("WARNING: spatial batching failed; continuing with the current graph.")
                traceback.print_exc()

        canonicalize_stats = canonicalize_temporal_spatial_ir(
            gm,
            dump_dir=local_dir,
            strict=False,
            rewrite_temporal_mean=not getattr(args, "disable_temporal_mean_rewrite", False),
            drop_intermediate_states=getattr(args, "drop_intermediate_states", False),
        )
        counters.canonicalize_cat_chunk_removed += canonicalize_stats.canonicalize_cat_chunk_removed
        counters.canonicalize_chunk_cat_removed += canonicalize_stats.canonicalize_chunk_cat_removed
        counters.canonicalize_getitem_cat_removed += canonicalize_stats.canonicalize_getitem_cat_removed
        counters.canonicalize_stack_getitem_removed += canonicalize_stats.canonicalize_stack_getitem_removed
        counters.canonicalize_getitem_stack_removed += canonicalize_stats.canonicalize_getitem_stack_removed
        counters.canonicalize_stack_chunk_removed += canonicalize_stats.canonicalize_stack_chunk_removed
        counters.canonicalize_getitem_stack_chunk_removed += canonicalize_stats.canonicalize_getitem_stack_chunk_removed
        counters.canonicalize_cat_linear_chunk_removed += canonicalize_stats.canonicalize_cat_linear_chunk_removed
        counters.canonicalize_cat_linear_chunk_getitem_replaced += (
            canonicalize_stats.canonicalize_cat_linear_chunk_getitem_replaced
        )
        counters.temporal_mean_rewrites += canonicalize_stats.temporal_mean_rewrites
        counters.temporal_mean_removed_getitems += canonicalize_stats.temporal_mean_removed_getitems
        counters.temporal_mean_removed_adds += canonicalize_stats.temporal_mean_removed_adds
        counters.state_prune_removed_final_return_states += canonicalize_stats.state_prune_removed_final_return_states
        counters.ir_nodes_before += canonicalize_stats.ir_nodes_before
        counters.ir_nodes_after += canonicalize_stats.ir_nodes_after
        counters.ir_getitem_before += canonicalize_stats.ir_getitem_before
        counters.ir_getitem_after += canonicalize_stats.ir_getitem_after
        counters.ir_add_before += canonicalize_stats.ir_add_before
        counters.ir_add_after += canonicalize_stats.ir_add_after
        counters.ir_div_before += canonicalize_stats.ir_div_before
        counters.ir_div_after += canonicalize_stats.ir_div_after
        counters.ir_returned_states_before += canonicalize_stats.ir_returned_states_before
        counters.ir_returned_states_after += canonicalize_stats.ir_returned_states_after
        counters.canonicalize_view_folded += canonicalize_stats.canonicalize_view_folded
        counters.canonicalize_dead_nodes_removed += canonicalize_stats.canonicalize_dead_nodes_removed
        counters.canonicalize_final_cat_count += canonicalize_stats.final_cat_count
        counters.canonicalize_final_chunk_count += canonicalize_stats.final_chunk_count
        counters.canonicalize_final_getitem_count += canonicalize_stats.final_getitem_count

        temporal_graph_stats = analyze_temporal_graph(gm)
        print_temporal_graph_summary(temporal_graph_stats)
        dump_temporal_graph_validation(temporal_graph_stats, local_dir / "temporal_graph_validation.json")
        counters.temporal_graph_getitem_count += temporal_graph_stats.getitem_count
        counters.temporal_graph_getitem_from_temporal += temporal_graph_stats.getitem_from_temporal
        counters.temporal_graph_materialized_timestep_tensors += temporal_graph_stats.materialized_timestep_tensors
        counters.temporal_graph_fragmentation_paths += len(temporal_graph_stats.fragmentation_paths)

        fused_state_count = count_fused_conv_lif_state_nodes(gm)
        fused_temporal_state_count = count_fused_temporal_conv_lif_state_nodes(gm)
        fused_temporal_residual_state_count = count_fused_temporal_conv_add_lif_state_nodes(gm)
        fused_temporal_lif_state_count = count_fused_temporal_lif_state_nodes(gm)
        fused_temporal_linear_lif_state_count = count_fused_temporal_linear_lif_state_nodes(gm)
        fused_temporal_lif_avgpool_linear_count = count_fused_temporal_lif_avgpool_linear_nodes(gm)
        save_graph_files(gm, local_dir, "rewritten")

        from compiler.passes.registry import apply_post_fuse_passes

        post_fuse_pass_stats = apply_post_fuse_passes(gm)
        if post_fuse_pass_stats:
            save_graph_files(gm, local_dir, "post_fuse_optimized")

        counters.lif_state_nodes += lif_state_count
        counters.direct_matches += len(direct_matches)
        counters.conv_bn_matches += len(conv_bn_matches)
        counters.direct_replaced += direct_replaced
        counters.conv_bn_replaced += conv_bn_replaced
        counters.fused_state_nodes += fused_state_count
        counters.fused_temporal_state_nodes += fused_temporal_state_count
        counters.fused_temporal_residual_state_nodes += fused_temporal_residual_state_count
        counters.fused_temporal_lif_state_nodes += fused_temporal_lif_state_count
        counters.fused_temporal_linear_lif_state_nodes += fused_temporal_linear_lif_state_count
        counters.fused_temporal_lif_avgpool_linear_nodes += fused_temporal_lif_avgpool_linear_count
        counters.fused_temporal_lif_tail_nodes += fused_temporal_lif_avgpool_linear_count
        counters.single_step_replaced_patterns += direct_replaced + conv_bn_replaced

        if args.rewrite_backend_mode == "eager":
            return gm.forward
        if args.rewrite_backend_mode == "standalone":
            removed_v_final_outputs = prune_graph_output_v_final_states(gm)
            if removed_v_final_outputs:
                print(f"[STATE_PRUNE] removed_v_final_outputs={removed_v_final_outputs}")
            return build_fx_standalone_backend(
                gm,
                num_streams=args.fx_standalone_streams,
                use_cuda_graph=args.fx_standalone_cudagraph,
                example_inputs=tuple(example_inputs),
                debug=args.fx_standalone_debug,
                schedule_policy=args.fx_standalone_schedule_policy,
            )
        gm.meta.pop("dynamo_compile_id", None)
        if hasattr(gm, "_param_name_to_source"):
            delattr(gm, "_param_name_to_source")
        return torch._inductor.compile(
            gm,
            example_inputs,
            options=inductor_options_from_compile_kwargs(compile_kwargs),
        )

    return backend


def synchronize_if_needed(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def run_model(name: str, model: nn.Module, x: torch.Tensor, device: str, compile_mode: bool, args, backend=None) -> RunResult:
    try:
        model.eval()
        reset_lif_modules(model)
        runnable = model
        if compile_mode:
            runnable = compile_with_chronos_options(
                model,
                backend=backend if backend is not None else "inductor",
                enable_cudagraphs=args.enable_cudagraphs,
                cudagraph_mode=args.cudagraph_mode,
                fullgraph=False,
                dynamic=False,
            )
        synchronize_if_needed(device)
        start = time.perf_counter()
        with torch.no_grad():
            out = runnable(x)
        synchronize_if_needed(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if isinstance(out, (tuple, list)):
            out = out[0]
        return RunResult(
            name=name,
            ok=True,
            shape=list(out.shape),
            dtype=str(out.dtype),
            elapsed_ms=elapsed_ms,
        ), out.detach().clone()
    except Exception:
        return RunResult(name=name, ok=False, error=traceback.format_exc()), None


def compare_to(result: RunResult, out: Optional[torch.Tensor], ref: Optional[torch.Tensor], rtol: float, atol: float):
    if out is None or ref is None or not result.ok:
        return
    diff = (out - ref).abs()
    result.max_abs_diff = diff.max().item()
    result.mean_abs_diff = diff.mean().item()
    result.allclose = torch.allclose(out, ref, rtol=rtol, atol=atol)


def write_summary(path: Path, payload: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def validate_one_model(model_name: str, args) -> Dict[str, Any]:
    print(f"\n================ {model_name} ================")
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    print(
        "[Baseline Config] "
        f"dtype={args.dtype} "
        f"matmul_allow_tf32={torch.backends.cuda.matmul.allow_tf32} "
        f"cudnn_allow_tf32={torch.backends.cudnn.allow_tf32} "
        f"float32_matmul_precision={torch.get_float32_matmul_precision()}"
    )
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)

    base_layer_s = make_resnet_layer(
        model_name,
        allow_resnet32_fallback=not args.require_direct_resnet32_api,
        step_mode="s",
        model_channels=args.model_channels,
        lif_impl=args.lif_impl,
        sequence_length=args.sequence_length,
        transformer_depth=args.transformer_depth,
        transformer_dim=args.transformer_dim,
        transformer_heads=args.transformer_heads,
        transformer_input_dim=args.transformer_input_dim,
        transformer_vocab_size=args.transformer_vocab_size,
        transformer_num_classes=args.transformer_num_classes,
    ).to(device=args.device, dtype=dtype).eval()
    base_layer_m = make_resnet_layer(
        model_name,
        allow_resnet32_fallback=not args.require_direct_resnet32_api,
        step_mode="m",
        model_channels=args.model_channels,
        lif_impl=args.lif_impl,
        sequence_length=args.sequence_length,
        transformer_depth=args.transformer_depth,
        transformer_dim=args.transformer_dim,
        transformer_heads=args.transformer_heads,
        transformer_input_dim=args.transformer_input_dim,
        transformer_vocab_size=args.transformer_vocab_size,
        transformer_num_classes=args.transformer_num_classes,
    ).to(device=args.device, dtype=dtype).eval()
    x = make_model_input(model_name, args, dtype)

    models = {
        "baseline_s_eager": SingleStepModeLoopWrapper(copy.deepcopy(base_layer_s), args.T).to(args.device).eval(),
        "baseline_s_compile": SingleStepModeLoopWrapper(copy.deepcopy(base_layer_s), args.T).to(args.device).eval(),
        "baseline_m_eager": MultiStepModeWrapper(copy.deepcopy(base_layer_m), args.T).to(args.device).eval(),
        "baseline_m_compile": MultiStepModeWrapper(copy.deepcopy(base_layer_m), args.T).to(args.device).eval(),
        "rewrite_s_compile": SingleStepModeLoopWrapper(copy.deepcopy(base_layer_s), args.T).to(args.device).eval(),
        "rewrite_m_compile": MultiStepModeWrapper(copy.deepcopy(base_layer_m), args.T).to(args.device).eval(),
    }

    snn_custom_ops.configure_fused_op(
        backend=args.fused_op_backend,
        strict_triton=args.strict_triton,
        verbose=args.print_fused_op_calls,
        use_triton_autotune=not args.disable_triton_autotune,
    )
    snn_custom_ops.reset_fused_op_call_stats()

    out_dir = Path(args.out_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, RunResult] = {}
    outputs: Dict[str, Optional[torch.Tensor]] = {}
    cudagraph_status_by_case: Dict[str, Dict[str, Any]] = {}
    fx_standalone_cudagraph_status_by_case: Dict[str, Dict[str, Any]] = {}
    _, baseline_compile_config = build_chronos_compile_config(
        backend="inductor",
        enable_cudagraphs=args.enable_cudagraphs,
        cudagraph_mode=args.cudagraph_mode,
        fullgraph=False,
        dynamic=False,
    )
    _, rewrite_compile_config = build_chronos_compile_config(
        backend=args.rewrite_backend_mode,
        enable_cudagraphs=args.enable_cudagraphs,
        cudagraph_mode=args.cudagraph_mode,
        fullgraph=False,
        dynamic=False,
    )

    for case_name, compile_mode, backend in [
        ("baseline_s_eager", False, None),
        ("baseline_s_compile", True, None),
        ("baseline_m_eager", False, None),
        ("baseline_m_compile", True, None),
    ]:
        print(f"[RUN] {model_name}/{case_name}")
        counter_before = snapshot_compile_counters()
        result, out = run_model(case_name, models[case_name], x, args.device, compile_mode, args, backend)
        counter_diff = diff_compile_counters(counter_before, snapshot_compile_counters())
        graph_count = counter_diff.get("stats", {}).get("unique_graphs") if compile_mode else None
        cudagraph_status_by_case[case_name] = summarize_cudagraph_check(
            model=model_name,
            case=case_name,
            compile_config=baseline_compile_config,
            compile_mode=compile_mode,
            device=args.device,
            graph_count=graph_count,
            counter_diff=counter_diff,
        )
        results[case_name] = result
        outputs[case_name] = out
        fx_standalone_cudagraph_status_by_case[case_name] = {}
        if not result.ok:
            print(f"[FAIL] {case_name}: {result.error.splitlines()[-1] if result.error else 'unknown error'}")

    rewrite_counters: Dict[str, RewriteCounters] = {
        "rewrite_s_compile": RewriteCounters(),
        "rewrite_m_compile": RewriteCounters(),
    }
    for case_name, ref_name in [
        ("rewrite_s_compile", "baseline_s_eager"),
        ("rewrite_m_compile", "baseline_m_eager"),
    ]:
        print(f"[RUN] {model_name}/{case_name}")
        backend = make_rewrite_backend(args, out_dir / case_name, rewrite_counters[case_name])
        counter_before = snapshot_compile_counters()
        graph_count_before = rewrite_counters[case_name].captured_graphs
        result, out = run_model(case_name, models[case_name], x, args.device, True, args, backend)
        counter_diff = diff_compile_counters(counter_before, snapshot_compile_counters())
        graph_count = rewrite_counters[case_name].captured_graphs - graph_count_before
        cudagraph_status_by_case[case_name] = summarize_cudagraph_check(
            model=model_name,
            case=case_name,
            compile_config=rewrite_compile_config,
            compile_mode=True,
            device=args.device,
            graph_count=graph_count,
            counter_diff=counter_diff,
        )
        fx_standalone_cudagraph_status_by_case[case_name] = (
            get_fx_standalone_cudagraph_status()
            if args.rewrite_backend_mode == "standalone"
            else {}
        )
        results[case_name] = result
        outputs[case_name] = out
        if not result.ok:
            print(f"[FAIL] {case_name}: {result.error.splitlines()[-1] if result.error else 'unknown error'}")

    compare_pairs = {
        "baseline_s_compile": "baseline_s_eager",
        "baseline_m_compile": "baseline_m_eager",
        "rewrite_s_compile": "baseline_s_eager",
        "rewrite_m_compile": "baseline_m_eager",
    }
    for case_name, ref_name in compare_pairs.items():
        compare_to(results[case_name], outputs[case_name], outputs[ref_name], args.rtol, args.atol)

    call_stats = snn_custom_ops.get_fused_op_call_stats()
    payload = {
        "model": model_name,
        "input_shape": list(x.shape),
        "model_channels": args.model_channels,
        "lif_impl": args.lif_impl,
        "dtype": args.dtype,
        "T": args.T,
        "temporal_fuse_window": args.temporal_fuse_window,
        "enable_temporal_rewrite": args.enable_temporal_rewrite,
        "fused_op_backend": args.fused_op_backend,
        "enable_cudagraphs": args.enable_cudagraphs,
        "cudagraph_mode": args.cudagraph_mode,
        "compile_mode": baseline_compile_config["compile_mode"],
        "compile_options": baseline_compile_config["compile_options"],
        "baseline_compile_config": baseline_compile_config,
        "rewrite_compile_config": rewrite_compile_config,
        "results": {name: asdict(result) for name, result in results.items()},
        "rewrite_counters": {name: asdict(counters) for name, counters in rewrite_counters.items()},
        "fused_op_call_stats": call_stats,
        "cudagraph_status_by_case": cudagraph_status_by_case,
        "fx_standalone_cudagraph_status_by_case": fx_standalone_cudagraph_status_by_case,
    }
    write_summary(out_dir / "summary.json", payload)

    print(f"\n[SUMMARY] {model_name}")
    for name in [
        "baseline_s_eager",
        "baseline_s_compile",
        "baseline_m_eager",
        "baseline_m_compile",
        "rewrite_s_compile",
        "rewrite_m_compile",
    ]:
        result = results[name]
        status = "OK" if result.ok else "FAIL"
        diff = ""
        if result.max_abs_diff is not None:
            diff = f" max={result.max_abs_diff:.3e} mean={result.mean_abs_diff:.3e} allclose={result.allclose}"
        print(f"  {name}: {status}{diff}")
    print(f"  rewrite_s counters: {asdict(rewrite_counters['rewrite_s_compile'])}")
    print(f"  rewrite_m counters: {asdict(rewrite_counters['rewrite_m_compile'])}")
    print(f"  temporal_fuse_window: {args.temporal_fuse_window}")
    print(f"  compile config: {compile_config}")
    print(f"  fused calls: {call_stats}")
    print(f"  wrote: {out_dir / 'summary.json'}")
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description="Validate Chronos FX Conv+BN+LIF rewrite against baseline s/m eager/compile.")
    parser.add_argument("--models", nargs="+", default=["resnet18", "resnet34"], choices=CHRONOS_MODEL_CHOICES)
    parser.add_argument("--model-channels", type=int, default=64, help="Base channel width for handcrafted alexnet/zfnet models.")
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--transformer-depth", type=int, default=8)
    parser.add_argument("--transformer-dim", type=int, default=256)
    parser.add_argument("--transformer-heads", type=int, default=8)
    parser.add_argument("--transformer-input-dim", type=int, default=768)
    parser.add_argument("--transformer-vocab-size", type=int, default=30522)
    parser.add_argument("--transformer-num-classes", type=int, default=100)
    parser.add_argument("--lif-impl", choices=LIF_IMPL_CHOICES, default="chronos", help="LIF implementation used when constructing benchmark models.")
    parser.add_argument("--T", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--fused-op-backend", choices=("torch", "triton"), default="torch")
    parser.add_argument("--rewrite-backend-mode", choices=("eager", "inductor", "standalone"), default="inductor")
    parser.add_argument("--fx-standalone-streams", type=int, default=1)
    parser.add_argument("--fx-standalone-cudagraph", action="store_true")
    parser.add_argument("--fx-standalone-debug", action="store_true")
    parser.add_argument("--fx-standalone-schedule-policy", choices=("topo", "ready"), default="topo")
    parser.add_argument("--strict-triton", action="store_true")
    parser.add_argument("--disable-triton-autotune", action="store_true")
    parser.add_argument("--disable-rewrite", action="store_true")
    parser.add_argument("--disable-conv-bn-lif", action="store_true")
    parser.add_argument("--disable-temporal-lif-avgpool-linear-rewrite", action="store_true")
    parser.add_argument(
        "--disable-temporal-lif-tail-rewrite",
        action="store_true",
        dest="disable_temporal_lif_avgpool_linear_rewrite",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--disable-temporal-lif-rewrite", action="store_true")
    parser.add_argument("--disable-temporal-linear-lif-rewrite", action="store_true")
    parser.add_argument("--drop-intermediate-states", action="store_true")
    parser.add_argument("--enable-temporal-mean-rewrite", action="store_true")
    parser.add_argument("--disable-temporal-mean-rewrite", action="store_true")
    parser.add_argument("--enable-temporal-rewrite", action="store_true")
    parser.add_argument("--temporal-fuse-window", type=int, default=1)
    parser.add_argument("--temporal-allow-tail", action="store_true")
    parser.add_argument("--enable-temporal-schedule", action="store_true")
    parser.add_argument("--temporal-schedule-window", type=int, default=None)
    parser.add_argument("--temporal-schedule-dump", action="store_true")
    parser.add_argument("--temporal-schedule-strict", action="store_true")
    parser.add_argument("--enable-spatial-batching", action="store_true")
    parser.add_argument(
        "--spatial-batching-ops",
        nargs="+",
        default=["conv", "bn", "add", "maxpool", "avgpool", "flatten", "linear", "elementwise", "view"],
        choices=["conv", "bn", "add", "maxpool", "linear", "flatten", "avgpool", "elementwise", "view"],
    )
    parser.add_argument("--spatial-batching-dump", action="store_true")
    parser.add_argument("--spatial-batching-strict", action="store_true")
    parser.add_argument("--disable-spatial-batching-chain", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--enable-cudagraphs", action="store_true")
    parser.add_argument("--cudagraph-mode", choices=("reduce-overhead", "triton-option", "both"), default="reduce-overhead")
    parser.add_argument("--max-patterns", type=int, default=1)
    parser.add_argument("--print-fused-op-calls", action="store_true")
    parser.add_argument("--require-direct-resnet32-api", action="store_true")
    parser.add_argument("--out-dir", default="chronos_baseline_validation")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.rewrite_backend_mode == "standalone" and args.fx_standalone_cudagraph and args.enable_cudagraphs:
        print(
            "[FX_STANDALONE] warning: disabling outer --enable-cudagraphs because "
            "--fx-standalone-cudagraph captures the standalone executor internally"
        )
        args.enable_cudagraphs = False
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    if args.dtype == "fp16" and args.rtol == 1e-4 and args.atol == 1e-4:
        args.rtol = 1e-2
        args.atol = 1e-2
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    all_payloads = {}
    for model_name in args.models:
        try:
            all_payloads[model_name] = validate_one_model(model_name, args)
        except Exception:
            print(f"[MODEL FAIL] {model_name}")
            traceback.print_exc()
            all_payloads[model_name] = {"model": model_name, "error": traceback.format_exc()}

    write_summary(Path(args.out_dir) / "summary_all.json", all_payloads)
    print(f"\nWrote aggregate summary: {Path(args.out_dir) / 'summary_all.json'}")


if __name__ == "__main__":
    main()
