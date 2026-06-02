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
from spikingjelly.activation_based import functional, layer, neuron, surrogate
from spikingjelly.activation_based.model import spiking_resnet, spiking_vgg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

import runtime.snn_custom_ops as snn_custom_ops
from runtime.fx_standalone_executor import (
    build_fx_standalone_backend,
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
from test.models_for_fx_test import CustomStatefulIFNode, reset_custom_stateful_lif_modules


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
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        y_seq = self.layer(x_seq)
        return y_seq.mean(0)


SingleStepWrapper = SingleStepModeLoopWrapper
MultiStepWrapper = MultiStepModeWrapper


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
]


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
) -> nn.Module:
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
            rewrite_temporal_mean=getattr(args, "enable_temporal_mean_rewrite", False),
            drop_intermediate_states=getattr(args, "drop_intermediate_states", False),
        )
        counters.canonicalize_cat_chunk_removed += canonicalize_stats.canonicalize_cat_chunk_removed
        counters.canonicalize_chunk_cat_removed += canonicalize_stats.canonicalize_chunk_cat_removed
        counters.canonicalize_getitem_cat_removed += canonicalize_stats.canonicalize_getitem_cat_removed
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
    ).to(device=args.device, dtype=dtype).eval()
    base_layer_m = make_resnet_layer(
        model_name,
        allow_resnet32_fallback=not args.require_direct_resnet32_api,
        step_mode="m",
        model_channels=args.model_channels,
        lif_impl=args.lif_impl,
    ).to(device=args.device, dtype=dtype).eval()
    x = torch.randn(args.batch_size, 3, args.height, args.width, device=args.device, dtype=dtype)

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
    _, compile_config = build_chronos_compile_config(
        backend="inductor",
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
            compile_config=compile_config,
            compile_mode=compile_mode,
            device=args.device,
            graph_count=graph_count,
            counter_diff=counter_diff,
        )
        results[case_name] = result
        outputs[case_name] = out
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
            compile_config=compile_config,
            compile_mode=True,
            device=args.device,
            graph_count=graph_count,
            counter_diff=counter_diff,
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
        "input_shape": [args.batch_size, 3, args.height, args.width],
        "model_channels": args.model_channels,
        "lif_impl": args.lif_impl,
        "dtype": args.dtype,
        "T": args.T,
        "temporal_fuse_window": args.temporal_fuse_window,
        "enable_temporal_rewrite": args.enable_temporal_rewrite,
        "fused_op_backend": args.fused_op_backend,
        "enable_cudagraphs": args.enable_cudagraphs,
        "cudagraph_mode": args.cudagraph_mode,
        "compile_mode": compile_config["compile_mode"],
        "compile_options": compile_config["compile_options"],
        "results": {name: asdict(result) for name, result in results.items()},
        "rewrite_counters": {name: asdict(counters) for name, counters in rewrite_counters.items()},
        "fused_op_call_stats": call_stats,
        "cudagraph_status_by_case": cudagraph_status_by_case,
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
