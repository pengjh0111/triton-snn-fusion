import json
import operator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import torch


def _target_text(node: torch.fx.Node) -> str:
    return str(node.target)


def _is_getitem(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and node.target is operator.getitem


def _is_call_named(node: torch.fx.Node, name: str) -> bool:
    return node.op == "call_function" and (node.target is getattr(torch, name, None) or name in _target_text(node))


def _is_temporal_custom_node(node: torch.fx.Node) -> bool:
    text = _target_text(node)
    return "snn_custom.fused_temporal_" in text


def _is_temporal_stack_getitem(node: torch.fx.Node) -> bool:
    if not _is_getitem(node) or not node.args:
        return False
    source = node.args[0]
    if not isinstance(source, torch.fx.Node):
        return False
    if _is_temporal_custom_node(source):
        return True
    if not _is_getitem(source) or not source.args:
        return False
    tuple_source = source.args[0]
    return isinstance(tuple_source, torch.fx.Node) and _is_temporal_custom_node(tuple_source)


def _is_temporal_stack_output(node: torch.fx.Node) -> bool:
    if not _is_getitem(node) or len(node.args) < 2 or node.args[1] != 0:
        return False
    source = node.args[0]
    return isinstance(source, torch.fx.Node) and _is_temporal_custom_node(source)


def _is_batched_layout_projection(node: torch.fx.Node) -> bool:
    if node.meta.get("chronos_temporal_layout") == "batched_tn":
        return True
    return node.op == "call_method" and node.target == "flatten" and node.meta.get("chronos_temporal_layout") == "batched_tn"


def _is_spatial_consumer(node: torch.fx.Node) -> bool:
    text = _target_text(node)
    spatial_tokens = (
        "conv2d",
        "batch_norm",
        "max_pool2d",
        "avg_pool",
        "adaptive_avg_pool",
        "flatten",
        "linear",
    )
    return any(token in text for token in spatial_tokens)


@dataclass
class TemporalGraphValidationStats:
    temporal_tensor_edges: int = 0
    temporal_op_nodes: int = 0
    temporal_batched_output_nodes: int = 0
    temporal_stack_output_nodes: int = 0
    getitem_count: int = 0
    getitem_from_temporal: int = 0
    materialized_timestep_tensors: int = 0
    cat_after_temporal_getitem: int = 0
    torch_stack_count: int = 0
    torch_cat_count: int = 0
    torch_chunk_count: int = 0
    fragmentation_paths: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def analyze_temporal_graph(gm: torch.fx.GraphModule) -> TemporalGraphValidationStats:
    stats = TemporalGraphValidationStats()
    for node in gm.graph.nodes:
        text = _target_text(node)
        if _is_temporal_custom_node(node):
            stats.temporal_op_nodes += 1
        if node.meta.get("chronos_temporal_layout") == "batched_tn":
            stats.temporal_batched_output_nodes += 1
        if node.meta.get("chronos_temporal_layout") == "stack" or _is_temporal_stack_output(node):
            stats.temporal_stack_output_nodes += 1
        if _is_getitem(node):
            stats.getitem_count += 1
            if _is_temporal_stack_getitem(node):
                stats.getitem_from_temporal += 1
                if node.args and isinstance(node.args[0], torch.fx.Node) and _is_getitem(node.args[0]):
                    stats.materialized_timestep_tensors += 1
        if _is_call_named(node, "stack"):
            stats.torch_stack_count += 1
        if _is_call_named(node, "cat"):
            stats.torch_cat_count += 1
        if _is_call_named(node, "chunk"):
            stats.torch_chunk_count += 1

    for node in gm.graph.nodes:
        if not _is_temporal_stack_getitem(node):
            continue
        for user in node.users:
            if _is_batched_layout_projection(user):
                continue
            if _is_spatial_consumer(user):
                path = {
                    "temporal_getitem": node.name,
                    "consumer": user.name,
                    "consumer_target": _target_text(user),
                }
                stats.fragmentation_paths.append(path)
                stats.warnings.append(
                    f"[TEMPORAL_FRAGMENTATION] {node.name} -> {user.name} ({_target_text(user)})"
                )
            if user.op == "call_function" and (user.target is torch.cat or "cat" in _target_text(user)):
                stats.cat_after_temporal_getitem += 1

    stats.temporal_tensor_edges = sum(
        1
        for node in gm.graph.nodes
        if _is_temporal_custom_node(node)
        for user in node.users
        if not _is_getitem(user)
    )
    return stats


def print_temporal_graph_summary(stats: TemporalGraphValidationStats):
    print(
        "[TEMPORAL_GRAPH] "
        f"temporal_tensor_edges={stats.temporal_tensor_edges} "
        f"temporal_op_nodes={stats.temporal_op_nodes} "
        f"temporal_batched_output_nodes={stats.temporal_batched_output_nodes} "
        f"temporal_stack_output_nodes={stats.temporal_stack_output_nodes} "
        f"getitem={stats.getitem_count} "
        f"getitem_from_temporal={stats.getitem_from_temporal} "
        f"materialized_timestep_tensors={stats.materialized_timestep_tensors} "
        f"stack={stats.torch_stack_count} cat={stats.torch_cat_count} chunk={stats.torch_chunk_count} "
        f"cat_after_temporal_getitem={stats.cat_after_temporal_getitem}"
    )
    for warning in stats.warnings[:20]:
        print(warning)
    if len(stats.warnings) > 20:
        print(f"[TEMPORAL_FRAGMENTATION] ... {len(stats.warnings) - 20} more")


def dump_temporal_graph_validation(stats: TemporalGraphValidationStats, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(stats), indent=2, sort_keys=True), encoding="utf-8")
