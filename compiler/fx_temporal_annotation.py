from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

import runtime.snn_custom_ops  # noqa: F401
from compiler.fx_lif_rewrite import (
    is_batch_norm_node,
    is_conv_node,
    is_custom_lif_state_node,
    is_fused_conv_lif_state_node,
)
from compiler.fx_lif_temporal_rewrite import collect_conv_bn_lif_state_patterns
from compiler.fx_temporal_scheduler import split_fx_graph_into_timesteps


@dataclass
class TemporalAnnotationStats:
    temporal_annotated_nodes: int = 0
    temporal_annotation_missing: int = 0
    temporal_annotation_roles: Dict[str, int] = field(default_factory=dict)
    temporal_annotation_windows: Dict[int, int] = field(default_factory=dict)
    temporal_annotation_reasons: Dict[str, int] = field(default_factory=dict)

    def role(self, value: str):
        self.temporal_annotation_roles[value] = self.temporal_annotation_roles.get(value, 0) + 1

    def window(self, value: int):
        self.temporal_annotation_windows[value] = self.temporal_annotation_windows.get(value, 0) + 1

    def missing(self, reason: str):
        self.temporal_annotation_missing += 1
        self.temporal_annotation_reasons[reason] = self.temporal_annotation_reasons.get(reason, 0) + 1


def _target_text(node: torch.fx.Node) -> str:
    return str(node.target)


def _is_spatial_node(node: torch.fx.Node) -> bool:
    text = _target_text(node)
    return (
        "max_pool2d" in text
        or "adaptive_avg_pool2d" in text
        or "avg_pool2d" in text
        or "flatten" in text
        or "linear" in text
        or node.target is torch.flatten
        or node.target is F.linear
    )


def _is_snn_node(node: torch.fx.Node) -> bool:
    text = _target_text(node)
    return "snn_custom." in text or "lif" in text.lower()


def _freeze(value):
    if isinstance(value, torch.fx.Node):
        return ("node", value.name)
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze(item)) for key, item in value.items()))
    return value


def _node_signature(node: torch.fx.Node, role: str) -> Tuple[Any, ...]:
    if role == "spatial":
        text = _target_text(node)
        if "max_pool2d" in text:
            return ("maxpool", str(node.target), _freeze(node.args[1:]), _freeze(node.kwargs))
        if "adaptive_avg_pool2d" in text:
            return ("adaptive_avg_pool", str(node.target), _freeze(node.args[1:]), _freeze(node.kwargs))
        if "flatten" in text or node.target is torch.flatten:
            args = list(node.args)
            start_dim = args[1] if len(args) > 1 else node.kwargs.get("start_dim", 0)
            end_dim = args[2] if len(args) > 2 else node.kwargs.get("end_dim", -1)
            return ("flatten", int(start_dim), int(end_dim))
        if "linear" in text or node.target is F.linear:
            return ("linear", str(node.target), _freeze(node.args[1:]), _freeze(node.kwargs))
    return (role, node.op, str(node.target))


def _role_for_node(
    gm: torch.fx.GraphModule,
    node: torch.fx.Node,
    fused_candidate_nodes,
) -> str:
    if node in fused_candidate_nodes:
        return "fused_candidate"
    if _is_snn_node(node) or is_custom_lif_state_node(node) or is_fused_conv_lif_state_node(node):
        return "snn"
    if _is_spatial_node(node):
        return "spatial"
    if is_conv_node(gm, node) or is_batch_norm_node(gm, node):
        return "fused_candidate"
    return "other"


def _write_meta(
    node: torch.fx.Node,
    timestep: int,
    temporal_window: int,
    occurrence: int,
    role: str,
):
    window_id = timestep // temporal_window if temporal_window > 0 else 0
    node.meta["chronos_timestep"] = timestep
    node.meta["chronos_window_id"] = window_id
    node.meta["chronos_occurrence"] = occurrence
    node.meta["chronos_role"] = role
    setattr(node, "_chronos_timestep", timestep)
    setattr(node, "_chronos_window_id", window_id)
    setattr(node, "_chronos_occurrence", occurrence)
    setattr(node, "_chronos_role", role)
    return window_id


def annotate_temporal_metadata(
    gm: torch.fx.GraphModule,
    temporal_window: int,
    T: int,
    *,
    strict: bool = False,
) -> TemporalAnnotationStats:
    stats = TemporalAnnotationStats()
    if T <= 0 or temporal_window <= 0:
        stats.missing("invalid_T_or_window")
        if strict:
            raise RuntimeError("T and temporal_window must be positive")
        return stats

    temporal_patterns = collect_conv_bn_lif_state_patterns(gm)
    fused_candidate_nodes = set()
    for pattern in temporal_patterns:
        fused_candidate_nodes.update(
            [
                pattern.conv_node,
                pattern.bn_node,
                pattern.lif_node,
                pattern.spike_getitem,
                pattern.v_getitem,
            ]
        )

    timestep_blocks = split_fx_graph_into_timesteps(gm, T, temporal_patterns)
    if len(timestep_blocks) == T:
        for timestep, block in enumerate(timestep_blocks):
            occurrence_counts: Dict[Tuple[Any, ...], int] = {}
            for node in block:
                role = _role_for_node(gm, node, fused_candidate_nodes)
                signature = _node_signature(node, role)
                occurrence = occurrence_counts.get(signature, 0)
                occurrence_counts[signature] = occurrence + 1
                window_id = _write_meta(node, timestep, temporal_window, occurrence, role)
                stats.temporal_annotated_nodes += 1
                stats.role(role)
                stats.window(window_id)
        return stats

    # Compatibility path: preserve explicit annotations already set by tests or
    # previous scheduling passes, and make them visible through node.meta.
    occurrence_counts: Dict[Tuple[int, Tuple[Any, ...]], int] = {}
    for node in gm.graph.nodes:
        timestep = getattr(node, "_chronos_timestep", None)
        if not isinstance(timestep, int):
            stats.missing("no_timestep_block_or_legacy_attr")
            continue
        role = getattr(node, "_chronos_role", _role_for_node(gm, node, fused_candidate_nodes))
        occurrence = getattr(node, "_chronos_occurrence", None)
        if not isinstance(occurrence, int):
            signature = _node_signature(node, role)
            key = (timestep, signature)
            occurrence = occurrence_counts.get(key, 0)
            occurrence_counts[key] = occurrence + 1
        window_id = _write_meta(node, timestep, temporal_window, int(occurrence), role)
        stats.temporal_annotated_nodes += 1
        stats.role(role)
        stats.window(window_id)

    if stats.temporal_annotated_nodes == 0 and strict:
        raise RuntimeError("failed to annotate temporal metadata")
    return stats
