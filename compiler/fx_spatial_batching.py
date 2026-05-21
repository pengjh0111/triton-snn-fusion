import operator
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from compiler.fx_lif_rewrite import _parse_conv_call_args, is_conv_node


@dataclass
class TemporalStackInput:
    temporal_tuple: torch.fx.Node
    spike_stack: torch.fx.Node
    getitem_node: torch.fx.Node
    timestep: int
    source_op: str


@dataclass
class BatchedChunkInput:
    batched_node: torch.fx.Node
    chunk_node: torch.fx.Node
    getitem_node: torch.fx.Node
    timestep: int
    chunks: int
    dim: int


@dataclass
class SpatialBatchCandidate:
    node: torch.fx.Node
    kind: str
    signature: Tuple[Any, ...]
    input_node: torch.fx.Node
    timestep: int
    window_id: int
    occurrence: int
    shape: Tuple[Any, ...]
    dtype: str
    input_kind: str = "plain"
    temporal_stack_input: Optional[TemporalStackInput] = None
    temporal_stack_inputs: Tuple[TemporalStackInput, ...] = field(default_factory=tuple)
    previous_batched_input: Optional[BatchedChunkInput] = None
    previous_batched_inputs: Tuple[BatchedChunkInput, ...] = field(default_factory=tuple)


@dataclass
class SpatialBatchGroup:
    kind: str
    signature: Tuple[Any, ...]
    window_id: int
    occurrence: int
    candidates: List[SpatialBatchCandidate]


@dataclass
class SpatialBatchingStats:
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
    reasons: Dict[str, int] = field(default_factory=dict)
    log: List[str] = field(default_factory=list)

    def skip(self, reason: str, message: str):
        self.spatial_batch_skipped += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1
        self.log.append(f"SKIP[{reason}] {message}")


def _target_text(target) -> str:
    return str(target)


def _node_sort_key(gm: torch.fx.GraphModule):
    order = {node: index for index, node in enumerate(gm.graph.nodes)}
    return lambda node: order[node]


def _get_chronos_meta(node: torch.fx.Node, key: str, default=None):
    meta_key = f"chronos_{key}"
    if meta_key in node.meta:
        return node.meta[meta_key]
    return getattr(node, f"_chronos_{key}", default)


def _collect_input_nodes(obj) -> List[torch.fx.Node]:
    out: List[torch.fx.Node] = []
    if isinstance(obj, torch.fx.Node):
        out.append(obj)
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            out.extend(_collect_input_nodes(item))
    elif isinstance(obj, dict):
        for item in obj.values():
            out.extend(_collect_input_nodes(item))
    return out


def _get_tensor_shape_dtype(node: torch.fx.Node) -> Tuple[Optional[Tuple[int, ...]], Optional[str]]:
    meta = node.meta.get("tensor_meta") or node.meta.get("val")
    if meta is None:
        return None, None
    shape = getattr(meta, "shape", None)
    dtype = getattr(meta, "dtype", None)
    if shape is not None:
        return tuple(int(dim) for dim in shape), str(dtype)
    if isinstance(meta, torch.Tensor):
        return tuple(int(dim) for dim in meta.shape), str(meta.dtype)
    return None, None


def _is_stateful_or_fused_snn_node(node: torch.fx.Node) -> bool:
    text = _target_text(node.target)
    return "snn_custom." in text or "lif" in text.lower()


def _getitem_index(node: torch.fx.Node) -> Optional[int]:
    if node.op != "call_function" or node.target is not operator.getitem or len(node.args) < 2:
        return None
    index = node.args[1]
    if isinstance(index, int):
        return index
    if isinstance(index, slice):
        return None
    try:
        return int(index)
    except Exception:
        return None


def _match_temporal_stack_getitem(node: torch.fx.Node) -> Optional[TemporalStackInput]:
    timestep = _getitem_index(node)
    if timestep is None or not node.args or not isinstance(node.args[0], torch.fx.Node):
        return None

    spike_stack = node.args[0]
    stack_index = _getitem_index(spike_stack)
    if stack_index != 0 or not spike_stack.args or not isinstance(spike_stack.args[0], torch.fx.Node):
        return None

    temporal_tuple = spike_stack.args[0]
    if temporal_tuple.op != "call_function":
        return None
    target_text = _target_text(temporal_tuple.target)
    if "snn_custom.fused_temporal_" not in target_text and "snn_custom::fused_temporal_" not in target_text:
        return None

    return TemporalStackInput(
        temporal_tuple=temporal_tuple,
        spike_stack=spike_stack,
        getitem_node=node,
        timestep=timestep,
        source_op=target_text,
    )


def _match_batched_chunk_getitem(node: torch.fx.Node) -> Optional[BatchedChunkInput]:
    timestep = _getitem_index(node)
    if timestep is None or not node.args or not isinstance(node.args[0], torch.fx.Node):
        return None

    chunk_node = node.args[0]
    if chunk_node.op != "call_function" or chunk_node.target is not torch.chunk or len(chunk_node.args) < 2:
        return None
    batched_node = chunk_node.args[0]
    chunks = chunk_node.args[1]
    dim = chunk_node.args[2] if len(chunk_node.args) > 2 else chunk_node.kwargs.get("dim", 0)
    if not isinstance(batched_node, torch.fx.Node):
        return None
    try:
        chunks = int(chunks)
        dim = int(dim)
    except Exception:
        return None
    if dim != 0:
        return None
    return BatchedChunkInput(
        batched_node=batched_node,
        chunk_node=chunk_node,
        getitem_node=node,
        timestep=timestep,
        chunks=chunks,
        dim=dim,
    )


def _is_maxpool_node(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    if node.op == "call_module":
        try:
            return isinstance(gm.get_submodule(str(node.target)), nn.MaxPool2d)
        except AttributeError:
            return False
    if node.op != "call_function":
        return False
    text = _target_text(node.target)
    return node.target is F.max_pool2d or "max_pool2d" in text


def _is_avgpool_node(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    if node.op == "call_module":
        try:
            return isinstance(gm.get_submodule(str(node.target)), (nn.AvgPool2d, nn.AdaptiveAvgPool2d))
        except AttributeError:
            return False
    if node.op != "call_function":
        return False
    text = _target_text(node.target)
    return node.target is F.avg_pool2d or "avg_pool2d" in text


def _is_linear_node(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    if node.op == "call_module":
        try:
            return isinstance(gm.get_submodule(str(node.target)), nn.Linear)
        except AttributeError:
            return False
    if node.op != "call_function":
        return False
    text = _target_text(node.target)
    return node.target is F.linear or "linear" in text


def _is_batch_norm_node(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    if node.op == "call_module":
        try:
            return isinstance(gm.get_submodule(str(node.target)), nn.BatchNorm2d)
        except AttributeError:
            return False
    if node.op != "call_function":
        return False
    text = _target_text(node.target)
    return node.target is F.batch_norm or "batch_norm" in text


def _batch_norm_is_eval(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    if node.op == "call_module":
        try:
            return not bool(gm.get_submodule(str(node.target)).training)
        except AttributeError:
            return False
    if node.op != "call_function":
        return False
    if "training" in node.kwargs:
        return bool(node.kwargs["training"]) is False
    if len(node.args) > 5:
        return bool(node.args[5]) is False
    return False


def _is_add_node(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and (
        node.target in (operator.add, torch.add) or "aten.add" in _target_text(node.target)
    )


def _is_relu_node(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    if node.op == "call_module":
        try:
            return isinstance(gm.get_submodule(str(node.target)), nn.ReLU)
        except AttributeError:
            return False
    if node.op == "call_function":
        return node.target is F.relu or node.target is torch.relu or "relu" in _target_text(node.target)
    if node.op == "call_method":
        return str(node.target) == "relu"
    return False


def _is_view_like_node(node: torch.fx.Node) -> bool:
    if node.op == "call_method":
        return str(node.target) in {"view", "reshape", "contiguous"}
    if node.op == "call_function":
        text = _target_text(node.target)
        return node.target in (torch.reshape,) or "reshape" in text or "view" in text
    return False


def _is_flatten_node(node: torch.fx.Node) -> bool:
    if node.op == "call_function":
        return node.target is torch.flatten or "flatten" in _target_text(node.target)
    if node.op == "call_method":
        return str(node.target) == "flatten"
    return False


def _is_adaptive_avg_pool_node(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    if node.op == "call_module":
        try:
            return isinstance(gm.get_submodule(str(node.target)), nn.AdaptiveAvgPool2d)
        except AttributeError:
            return False
    if node.op != "call_function":
        return False
    text = _target_text(node.target)
    return node.target is F.adaptive_avg_pool2d or "adaptive_avg_pool2d" in text


def _node_ref(value):
    if isinstance(value, torch.fx.Node):
        if value.op == "get_attr":
            return (value.op, str(value.target))
        return (value.op, str(value.target), value.name)
    if isinstance(value, (tuple, list)):
        return tuple(_node_ref(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _node_ref(item)) for key, item in value.items()))
    return value


def _signature_without_input(node: torch.fx.Node, kind: str) -> Tuple[Any, ...]:
    if kind == "conv":
        try:
            _inp, weight, bias, stride, padding, dilation, groups = _parse_conv_call_args(node)
            return (
                node.op,
                "conv2d",
                _node_ref(weight),
                _node_ref(bias),
                tuple(stride),
                tuple(padding),
                tuple(dilation),
                int(groups),
            )
        except Exception:
            return (node.op, str(node.target), kind, tuple(_node_ref(arg) for arg in node.args[1:]))
    if kind == "flatten":
        start_dim, end_dim = _flatten_dims(node)
        return (node.op, "flatten", kind, start_dim, end_dim)
    if kind == "add":
        args = tuple("<tensor>" if isinstance(arg, torch.fx.Node) else _node_ref(arg) for arg in node.args)
        kwargs = tuple(sorted((key, _node_ref(value)) for key, value in node.kwargs.items()))
        return (node.op, "add", kind, args, kwargs)
    args = tuple(_node_ref(arg) for arg in node.args[1:])
    kwargs = tuple(sorted((key, _node_ref(value)) for key, value in node.kwargs.items()))
    return (node.op, str(node.target), kind, args, kwargs)


def _flatten_dims(node: torch.fx.Node) -> Tuple[int, int]:
    if node.op == "call_method":
        args = list(node.args)
        start_dim = args[1] if len(args) > 1 else node.kwargs.get("start_dim", 0)
        end_dim = args[2] if len(args) > 2 else node.kwargs.get("end_dim", -1)
        return int(start_dim), int(end_dim)
    args = list(node.args)
    start_dim = args[1] if len(args) > 1 else node.kwargs.get("start_dim", 0)
    end_dim = args[2] if len(args) > 2 else node.kwargs.get("end_dim", -1)
    return int(start_dim), int(end_dim)


def _flatten_is_batch_safe(node: torch.fx.Node) -> bool:
    start_dim, _end_dim = _flatten_dims(node)
    return start_dim == 1


def _candidate_kind(gm: torch.fx.GraphModule, node: torch.fx.Node, enabled_ops: Sequence[str]) -> Optional[str]:
    enabled = set(enabled_ops)
    if "conv" in enabled and is_conv_node(gm, node):
        return "conv"
    if "bn" in enabled and _is_batch_norm_node(gm, node):
        return "bn"
    if "add" in enabled and _is_add_node(node):
        return "add"
    if "maxpool" in enabled and _is_maxpool_node(gm, node):
        return "maxpool"
    if "avgpool" in enabled and _is_avgpool_node(gm, node):
        if _is_adaptive_avg_pool_node(gm, node):
            return "adaptive_avg_pool"
        return "avg_pool"
    if "linear" in enabled and _is_linear_node(gm, node):
        return "linear"
    if "flatten" in enabled and _is_flatten_node(node):
        return "flatten"
    if "avgpool" in enabled and _is_adaptive_avg_pool_node(gm, node):
        return "adaptive_avg_pool"
    if "elementwise" in enabled and _is_relu_node(gm, node):
        return "relu"
    if "view" in enabled and _is_view_like_node(node):
        return "view"
    return None


def _candidate_input(node: torch.fx.Node) -> Optional[torch.fx.Node]:
    if not node.args:
        return None
    if _is_add_node(node):
        lhs = node.args[0] if len(node.args) > 0 else None
        rhs = node.args[1] if len(node.args) > 1 else None
        if isinstance(lhs, torch.fx.Node):
            return lhs
        if isinstance(rhs, torch.fx.Node):
            return rhs
        return None
    first = node.args[0]
    return first if isinstance(first, torch.fx.Node) else None


def _candidate_tensor_inputs(node: torch.fx.Node, kind: str) -> Tuple[torch.fx.Node, ...]:
    if kind == "add":
        return tuple(arg for arg in node.args[:2] if isinstance(arg, torch.fx.Node))
    input_node = _candidate_input(node)
    return (input_node,) if input_node is not None else ()


def _is_generated_spatial_batching_node(node: torch.fx.Node) -> bool:
    if node.meta.get("chronos_origin") == "temporal_stack_flatten":
        return True
    name = str(node.name)
    return (
        "_spatial_batch_" in name
        or "_temporal_stack_flatten" in name
        or name.endswith("_spatial_batch_cat")
        or name.endswith("_chunks")
    )


def _extract_candidate(
    gm: torch.fx.GraphModule,
    node: torch.fx.Node,
    enabled_ops: Sequence[str],
    temporal_window: int,
    occurrence_counts: Dict[Tuple[int, Tuple[Any, ...]], int],
    stats: SpatialBatchingStats,
) -> Optional[SpatialBatchCandidate]:
    if _is_stateful_or_fused_snn_node(node):
        return None
    if _is_generated_spatial_batching_node(node):
        return None
    kind = _candidate_kind(gm, node, enabled_ops)
    if kind is None:
        return None
    if kind == "flatten" and not _flatten_is_batch_safe(node):
        stats.skip("unsafe_flatten", f"node={node.name} flatten start_dim must be 1")
        return None
    if kind == "bn" and not _batch_norm_is_eval(gm, node):
        stats.skip("batch_norm_training", f"node={node.name} batch_norm training must be False")
        return None
    input_node = _candidate_input(node)
    if input_node is None:
        stats.skip("missing_input", f"node={node.name} kind={kind} has no tensor input node")
        return None
    tensor_inputs = _candidate_tensor_inputs(node, kind)
    temporal_stack_inputs = tuple(
        item for item in (_match_temporal_stack_getitem(input_item) for input_item in tensor_inputs) if item is not None
    )
    previous_batched_inputs = tuple(
        item for item in (_match_batched_chunk_getitem(input_item) for input_item in tensor_inputs) if item is not None
    )
    if kind == "add":
        if temporal_stack_inputs and previous_batched_inputs:
            stats.skip("add_mixed_batched_source_kinds", f"node={node.name} add mixes temporal stack and previous batched inputs")
            return None
        batchable_inputs = len(temporal_stack_inputs) + len(previous_batched_inputs)
        if batchable_inputs and batchable_inputs != len(tensor_inputs):
            stats.skip("add_mixed_batchable_inputs", f"node={node.name} add has unsupported mixed temporal/plain inputs")
            return None
        if len(temporal_stack_inputs) == 2 and temporal_stack_inputs[0].timestep != temporal_stack_inputs[1].timestep:
            stats.skip("temporal_stack_add_timestep_mismatch", f"node={node.name} add temporal stack timesteps differ")
            return None
        if len(previous_batched_inputs) == 2 and previous_batched_inputs[0].timestep != previous_batched_inputs[1].timestep:
            stats.skip("previous_batched_add_timestep_mismatch", f"node={node.name} add previous chunk timesteps differ")
            return None
    primary_temporal_stack = _match_temporal_stack_getitem(input_node)
    primary_previous_batched = _match_batched_chunk_getitem(input_node)
    timestep = _get_chronos_meta(node, "timestep")
    if not isinstance(timestep, int) and primary_temporal_stack is not None:
        timestep = primary_temporal_stack.timestep
    if not isinstance(timestep, int) and primary_previous_batched is not None:
        timestep = primary_previous_batched.timestep
    if not isinstance(timestep, int):
        stats.skip("missing_timestep", f"node={node.name} kind={kind} has no _chronos_timestep")
        return None
    shape, dtype = _get_tensor_shape_dtype(node)
    if shape is None or dtype is None:
        shape, dtype = _get_tensor_shape_dtype(input_node)
    if shape is None or dtype is None:
        shape, dtype = ("unknown",), "unknown"
    if len(shape) == 0:
        stats.skip("scalar_output", f"node={node.name} kind={kind} output is scalar")
        return None
    if node.op == "call_function" and "return_indices" in node.kwargs and node.kwargs["return_indices"]:
        stats.skip("tuple_output", f"node={node.name} kind={kind} returns indices")
        return None
    if kind == "add" and (temporal_stack_inputs or previous_batched_inputs):
        stack_refs = tuple((item.spike_stack.name, item.source_op) for item in temporal_stack_inputs)
        previous_refs = tuple(item.batched_node.name for item in previous_batched_inputs)
        kwargs = tuple(sorted((key, _node_ref(value)) for key, value in node.kwargs.items()))
        signature = (node.op, "add", kind, stack_refs, previous_refs, kwargs)
    else:
        signature = _signature_without_input(node, kind)
    occurrence = _get_chronos_meta(node, "occurrence")
    if not isinstance(occurrence, int):
        count_key = (timestep, signature)
        occurrence = occurrence_counts.get(count_key, 0)
        occurrence_counts[count_key] = occurrence + 1
    window_id = _get_chronos_meta(node, "window_id")
    if not isinstance(window_id, int):
        window_id = timestep // temporal_window
    return SpatialBatchCandidate(
        node=node,
        kind=kind,
        signature=signature,
        input_node=input_node,
        timestep=timestep,
        window_id=window_id,
        occurrence=occurrence,
        shape=shape,
        dtype=dtype,
        input_kind=(
            "temporal_stack_getitem"
            if primary_temporal_stack is not None
            else "previous_batched_chunk_getitem"
            if primary_previous_batched is not None
            else "plain"
        ),
        temporal_stack_input=primary_temporal_stack,
        temporal_stack_inputs=temporal_stack_inputs,
        previous_batched_input=primary_previous_batched,
        previous_batched_inputs=previous_batched_inputs,
    )


def collect_spatial_batch_candidates(
    gm: torch.fx.GraphModule,
    temporal_window: int,
    enabled_ops: Sequence[str],
    stats: SpatialBatchingStats,
) -> List[SpatialBatchCandidate]:
    occurrence_counts: Dict[Tuple[int, Tuple[Any, ...]], int] = {}
    candidates: List[SpatialBatchCandidate] = []
    for node in gm.graph.nodes:
        candidate = _extract_candidate(gm, node, enabled_ops, temporal_window, occurrence_counts, stats)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def group_spatial_batch_candidates(
    candidates: Iterable[SpatialBatchCandidate],
    temporal_window: int,
    stats: SpatialBatchingStats,
) -> List[SpatialBatchGroup]:
    grouped: Dict[Tuple[Any, ...], List[SpatialBatchCandidate]] = {}
    for candidate in candidates:
        temporal_stack_key = ()
        if candidate.temporal_stack_input is not None:
            temporal_stack_key = (candidate.temporal_stack_input.spike_stack.name,)
        if candidate.kind == "add" and candidate.temporal_stack_inputs:
            temporal_stack_key = tuple(item.spike_stack.name for item in candidate.temporal_stack_inputs)
        previous_batched_key = ()
        if candidate.previous_batched_input is not None:
            previous_batched_key = (candidate.previous_batched_input.batched_node.name,)
        if candidate.kind == "add" and candidate.previous_batched_inputs:
            previous_batched_key = tuple(item.batched_node.name for item in candidate.previous_batched_inputs)
        key = (candidate.window_id, candidate.occurrence, candidate.signature, temporal_stack_key, previous_batched_key)
        grouped.setdefault(key, []).append(candidate)

    groups: List[SpatialBatchGroup] = []
    for _key, items in grouped.items():
        items = sorted(items, key=lambda item: item.timestep)
        first = items[0]
        if len(items) != temporal_window:
            stats.skip(
                "incomplete_window",
                f"kind={first.kind} window={first.window_id} occurrence={first.occurrence} "
                f"size={len(items)} expected={temporal_window}",
            )
            continue
        expected_timesteps = list(range(first.window_id * temporal_window, (first.window_id + 1) * temporal_window))
        if [item.timestep for item in items] != expected_timesteps:
            stats.skip(
                "non_contiguous_timesteps",
                f"kind={first.kind} window={first.window_id} timesteps={[item.timestep for item in items]}",
            )
            continue
        if first.temporal_stack_input is not None:
            stack_node = first.temporal_stack_input.spike_stack
            stack_timesteps = [item.temporal_stack_input.timestep if item.temporal_stack_input is not None else None for item in items]
            if any(item.temporal_stack_input is None or item.temporal_stack_input.spike_stack is not stack_node for item in items):
                stats.skip("temporal_stack_source_mismatch", f"kind={first.kind} window={first.window_id}")
                continue
            if stack_timesteps != list(range(temporal_window)):
                stats.skip(
                    "temporal_stack_timestep_mismatch",
                    f"kind={first.kind} window={first.window_id} stack_timesteps={stack_timesteps}",
                )
                continue
        if first.previous_batched_input is not None:
            batched_node = first.previous_batched_input.batched_node
            chunk_node = first.previous_batched_input.chunk_node
            chunk_timesteps = [
                item.previous_batched_input.timestep if item.previous_batched_input is not None else None for item in items
            ]
            if any(
                item.previous_batched_input is None
                or item.previous_batched_input.batched_node is not batched_node
                or item.previous_batched_input.chunk_node is not chunk_node
                for item in items
            ):
                stats.skip("previous_batched_source_mismatch", f"kind={first.kind} window={first.window_id}")
                continue
            if chunk_timesteps != list(range(temporal_window)):
                stats.skip(
                    "previous_batched_timestep_mismatch",
                    f"kind={first.kind} window={first.window_id} chunk_timesteps={chunk_timesteps}",
                )
                continue
        if first.kind == "add" and first.temporal_stack_inputs:
            expected_num_inputs = len(first.temporal_stack_inputs)
            stack_nodes = tuple(item.spike_stack for item in first.temporal_stack_inputs)
            for item in items:
                if len(item.temporal_stack_inputs) != expected_num_inputs:
                    stats.skip("temporal_stack_add_input_count_mismatch", f"node={item.node.name}")
                    break
                if tuple(stack_item.spike_stack for stack_item in item.temporal_stack_inputs) != stack_nodes:
                    stats.skip("temporal_stack_add_source_mismatch", f"node={item.node.name}")
                    break
                if any(stack_item.timestep != item.timestep for stack_item in item.temporal_stack_inputs):
                    stats.skip("temporal_stack_add_timestep_mismatch", f"node={item.node.name}")
                    break
            else:
                pass
            if any(
                len(item.temporal_stack_inputs) != expected_num_inputs
                or tuple(stack_item.spike_stack for stack_item in item.temporal_stack_inputs) != stack_nodes
                or any(stack_item.timestep != item.timestep for stack_item in item.temporal_stack_inputs)
                for item in items
            ):
                continue
        if first.kind == "add" and first.previous_batched_inputs:
            expected_num_inputs = len(first.previous_batched_inputs)
            batched_nodes = tuple(item.batched_node for item in first.previous_batched_inputs)
            if any(
                len(item.previous_batched_inputs) != expected_num_inputs
                or tuple(chunk_item.batched_node for chunk_item in item.previous_batched_inputs) != batched_nodes
                or any(chunk_item.timestep != item.timestep for chunk_item in item.previous_batched_inputs)
                for item in items
            ):
                stats.skip("previous_batched_add_source_mismatch", f"kind={first.kind} window={first.window_id}")
                continue
        if any(item.shape != first.shape or item.dtype != first.dtype for item in items):
            stats.skip("incompatible_meta", f"kind={first.kind} window={first.window_id} occurrence={first.occurrence}")
            continue
        groups.append(
            SpatialBatchGroup(
                kind=first.kind,
                signature=first.signature,
                window_id=first.window_id,
                occurrence=first.occurrence,
                candidates=items,
            )
        )
    return groups


def _all_inputs_available_before(gm: torch.fx.GraphModule, inputs: List[torch.fx.Node], before: torch.fx.Node) -> Tuple[bool, str]:
    order = {node: index for index, node in enumerate(gm.graph.nodes)}
    before_order = order[before]
    for node in inputs:
        if order.get(node, before_order + 1) >= before_order:
            return False, f"input {node.name} is not defined before insertion point {before.name}"
    return True, ""


def _has_internal_data_dependency(group: SpatialBatchGroup) -> Tuple[bool, str]:
    nodes = {candidate.node for candidate in group.candidates}
    for candidate in group.candidates:
        deps = set(_collect_input_nodes((candidate.node.args, candidate.node.kwargs)))
        internal = deps & nodes
        if internal:
            return True, f"node {candidate.node.name} depends on {[node.name for node in internal]}"
    return False, ""


def _make_batched_call(gm: torch.fx.GraphModule, group: SpatialBatchGroup, input_node: torch.fx.Node) -> torch.fx.Node:
    return _make_batched_call_with_inputs(gm, group, (input_node,))


def _make_batched_call_with_inputs(
    gm: torch.fx.GraphModule,
    group: SpatialBatchGroup,
    input_nodes: Sequence[torch.fx.Node],
) -> torch.fx.Node:
    first = group.candidates[0].node
    if group.kind == "add":
        args = list(first.args)
        replacement_iter = iter(input_nodes)
        for index, arg in enumerate(args):
            if isinstance(arg, torch.fx.Node):
                try:
                    args[index] = next(replacement_iter)
                except StopIteration:
                    break
        return gm.graph.call_function(first.target, args=tuple(args), kwargs=dict(first.kwargs))
    input_node = input_nodes[0]
    new_args = (input_node,) + tuple(first.args[1:])
    if first.op == "call_module":
        return gm.graph.call_module(first.target, args=new_args, kwargs=dict(first.kwargs))
    if first.op == "call_method":
        return gm.graph.call_method(first.target, args=new_args, kwargs=dict(first.kwargs))
    return gm.graph.call_function(first.target, args=new_args, kwargs=dict(first.kwargs))


def _group_input_tuple(group: SpatialBatchGroup) -> Tuple[torch.fx.Node, ...]:
    return tuple(candidate.input_node for candidate in group.candidates)


def _temporal_stack_nodes_for_group(group: SpatialBatchGroup) -> Tuple[torch.fx.Node, ...]:
    first = group.candidates[0]
    if first.kind == "add" and first.temporal_stack_inputs:
        return tuple(item.spike_stack for item in first.temporal_stack_inputs)
    if first.temporal_stack_input is not None:
        return (first.temporal_stack_input.spike_stack,)
    return ()


def _previous_batched_nodes_for_group(group: SpatialBatchGroup) -> Tuple[torch.fx.Node, ...]:
    first = group.candidates[0]
    if first.kind == "add" and first.previous_batched_inputs:
        return tuple(item.batched_node for item in first.previous_batched_inputs)
    if first.previous_batched_input is not None:
        return (first.previous_batched_input.batched_node,)
    return ()


def _group_uses_temporal_stack_getitems(group: SpatialBatchGroup) -> bool:
    return bool(_temporal_stack_nodes_for_group(group))


def _make_temporal_stack_flatten(
    gm: torch.fx.GraphModule,
    stack_node: torch.fx.Node,
    name_prefix: str,
    temporal_window: int,
) -> torch.fx.Node:
    flatten_node = gm.graph.call_function(torch.flatten, args=(stack_node, 0, 1))
    flatten_node.name = f"{name_prefix}_temporal_stack_flatten"
    flatten_node.meta["chronos_temporal_layout"] = "batched_tn"
    flatten_node.meta["chronos_T"] = temporal_window
    flatten_node.meta["chronos_origin"] = "temporal_stack_flatten"
    flatten_node.meta["chronos_source_stack"] = stack_node.name
    return flatten_node


def rewrite_spatial_batch_group(
    gm: torch.fx.GraphModule,
    group: SpatialBatchGroup,
    stats: Optional[SpatialBatchingStats] = None,
) -> Tuple[bool, str]:
    first_node = group.candidates[0].node
    temporal_stack_nodes = _temporal_stack_nodes_for_group(group)
    previous_batched_nodes = _previous_batched_nodes_for_group(group)
    if temporal_stack_nodes:
        input_nodes = list(temporal_stack_nodes)
    elif previous_batched_nodes:
        input_nodes = list(previous_batched_nodes)
    else:
        input_nodes = [candidate.input_node for candidate in group.candidates]
    ok, reason = _all_inputs_available_before(gm, input_nodes, first_node)
    if not ok:
        return False, reason
    has_dep, reason = _has_internal_data_dependency(group)
    if has_dep:
        return False, reason

    with gm.graph.inserting_before(first_node):
        if temporal_stack_nodes:
            batched_inputs = tuple(
                _make_temporal_stack_flatten(gm, stack_node, f"{first_node.name}_{index}", len(group.candidates))
                for index, stack_node in enumerate(temporal_stack_nodes)
            )
            if stats is not None:
                stats.spatial_temporal_stack_groups += 1
                stats.spatial_temporal_stack_flatten_inputs += len(batched_inputs)
                stats.spatial_cat_avoided_by_temporal_stack_flatten += 1
                if group.kind == "bn":
                    stats.spatial_temporal_stack_bn_groups += 1
                elif group.kind == "add":
                    stats.spatial_temporal_stack_add_groups += 1
                elif group.kind in {"maxpool", "avg_pool", "adaptive_avg_pool"}:
                    stats.spatial_temporal_stack_pool_groups += 1
                elif group.kind == "flatten":
                    stats.spatial_temporal_stack_flatten_groups += 1
                elif group.kind == "linear":
                    stats.spatial_temporal_stack_linear_groups += 1
        elif previous_batched_nodes:
            batched_inputs = previous_batched_nodes
            if stats is not None:
                stats.spatial_previous_batched_groups += 1
                stats.spatial_reused_previous_batched_inputs += len(batched_inputs)
                stats.spatial_chunk_cat_avoided += 1
        else:
            cat_node = gm.graph.call_function(torch.cat, args=([candidate.input_node for candidate in group.candidates], 0))
            cat_node.name = f"{first_node.name}_spatial_batch_cat"
            batched_inputs = (cat_node,)
        batched_node = _make_batched_call_with_inputs(gm, group, batched_inputs)
        batched_node.name = f"{first_node.name}_spatial_batch_{group.kind}"
        chunks_node = gm.graph.call_function(torch.chunk, args=(batched_node, len(group.candidates), 0))
        chunks_node.name = f"{batched_node.name}_chunks"
        chunk_nodes = []
        for index, _candidate in enumerate(group.candidates):
            chunk_node = gm.graph.call_function(operator.getitem, args=(chunks_node, index))
            chunk_node.name = f"{batched_node.name}_t{index}"
            chunk_nodes.append(chunk_node)

    for candidate, chunk_node in zip(group.candidates, chunk_nodes):
        candidate.node.replace_all_uses_with(chunk_node)

    order = {node: index for index, node in enumerate(gm.graph.nodes)}
    for candidate in sorted(group.candidates, key=lambda item: order[item.node], reverse=True):
        if len(candidate.node.users) != 0:
            return False, f"node {candidate.node.name} still has users after replacement"
        gm.graph.erase_node(candidate.node)

    return True, ""


def dump_spatial_batching(groups: List[SpatialBatchGroup], stats: SpatialBatchingStats, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"groups={stats.spatial_batch_groups}",
        f"batched_ops={stats.spatial_batched_ops}",
        f"chains={stats.spatial_batch_chains}",
        f"chain_groups={stats.spatial_chain_groups}",
        f"cat_eliminated={stats.spatial_cat_eliminated}",
        f"chunk_eliminated={stats.spatial_chunk_eliminated}",
        f"batched_conv={stats.spatial_batched_conv}",
        f"batched_bn={stats.spatial_batched_bn}",
        f"batched_add={stats.spatial_batched_add}",
        f"batched_pool={stats.spatial_batched_pool}",
        f"batched_maxpool={stats.spatial_batched_maxpool}",
        f"batched_avgpool={stats.spatial_batched_avgpool}",
        f"batched_adaptive_avgpool={stats.spatial_batched_adaptive_avgpool}",
        f"batched_flatten={stats.spatial_batched_flatten}",
        f"batched_linear={stats.spatial_batched_linear}",
        f"batched_elementwise={stats.spatial_batched_elementwise}",
        f"temporal_stack_bn_groups={stats.spatial_temporal_stack_bn_groups}",
        f"temporal_stack_add_groups={stats.spatial_temporal_stack_add_groups}",
        f"temporal_stack_pool_groups={stats.spatial_temporal_stack_pool_groups}",
        f"temporal_stack_flatten_groups={stats.spatial_temporal_stack_flatten_groups}",
        f"temporal_stack_linear_groups={stats.spatial_temporal_stack_linear_groups}",
        f"temporal_stack_groups={stats.spatial_temporal_stack_groups}",
        f"temporal_stack_flatten_inputs={stats.spatial_temporal_stack_flatten_inputs}",
        f"cat_avoided_by_temporal_stack_flatten={stats.spatial_cat_avoided_by_temporal_stack_flatten}",
        f"previous_batched_groups={stats.spatial_previous_batched_groups}",
        f"reused_previous_batched_inputs={stats.spatial_reused_previous_batched_inputs}",
        f"chunk_cat_avoided={stats.spatial_chunk_cat_avoided}",
        f"skipped={stats.spatial_batch_skipped}",
        f"reasons={stats.reasons}",
        "",
    ]
    for index, group in enumerate(groups):
        lines.append(
            f"group_{index}: kind={group.kind} window={group.window_id} occurrence={group.occurrence} "
            f"nodes={[candidate.node.name for candidate in group.candidates]}"
        )
    if stats.log:
        lines.append("")
        lines.append("log:")
        lines.extend(f"  {line}" for line in stats.log)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def apply_spatial_batching(
    gm: torch.fx.GraphModule,
    temporal_window: int,
    enabled_ops: Sequence[str],
    dump_dir: Optional[Path] = None,
    strict: bool = False,
    enable_chain: bool = False,
) -> SpatialBatchingStats:
    stats = SpatialBatchingStats()
    if temporal_window <= 1:
        stats.skip("window_le_1", f"temporal_window={temporal_window}")
        return stats

    try:
        if enable_chain:
            stats.skip("chain_disabled", "chain-aware spatial batching is deprecated; using per-op batching")
        all_groups: List[SpatialBatchGroup] = []
        max_iter = 8
        for _iteration in range(max_iter):
            before_rewrites = stats.spatial_batched_ops
            candidates = collect_spatial_batch_candidates(gm, temporal_window, enabled_ops, stats)
            groups = group_spatial_batch_candidates(candidates, temporal_window, stats)
            stats.spatial_batch_groups += len(groups)
            all_groups.extend(groups)

            for group in groups:
                ok, reason = rewrite_spatial_batch_group(gm, group, stats)
                if not ok:
                    stats.skip("rewrite_skip", f"kind={group.kind} window={group.window_id}: {reason}")
                    continue
                count = len(group.candidates)
                stats.spatial_batched_ops += count
                if group.kind == "conv":
                    stats.spatial_batched_conv += count
                elif group.kind == "bn":
                    stats.spatial_batched_bn += count
                elif group.kind == "add":
                    stats.spatial_batched_add += count
                elif group.kind in {"maxpool", "avg_pool", "adaptive_avg_pool"}:
                    stats.spatial_batched_pool += count
                    if group.kind == "maxpool":
                        stats.spatial_batched_maxpool += count
                    elif group.kind == "avg_pool":
                        stats.spatial_batched_avgpool += count
                    else:
                        stats.spatial_batched_adaptive_avgpool += count
                elif group.kind == "flatten":
                    stats.spatial_batched_flatten += count
                elif group.kind == "linear":
                    stats.spatial_batched_linear += count
                else:
                    stats.spatial_batched_elementwise += count
                message = (
                    f"[SPATIAL_BATCHING][REWRITE] kind={group.kind} window={group.window_id} "
                    f"occurrence={group.occurrence} size={count}"
                )
                stats.log.append(message)
                print(message)

            gm.graph.eliminate_dead_code()
            gm.graph.lint()
            gm.recompile()
            if stats.spatial_batched_ops == before_rewrites:
                break

        gm.graph.lint()
        gm.recompile()

        if dump_dir is not None:
            dump_spatial_batching(all_groups, stats, dump_dir / "spatial_batching.txt")

        print(f"[SPATIAL_BATCHING] groups={stats.spatial_batch_groups}")
        print(f"[SPATIAL_BATCHING] batched_ops={stats.spatial_batched_ops}")
        print(f"[SPATIAL_BATCHING] chains={stats.spatial_batch_chains}")
        print(f"[SPATIAL_BATCHING] chain_groups={stats.spatial_chain_groups}")
        print(f"[SPATIAL_BATCHING] cat_eliminated={stats.spatial_cat_eliminated}")
        print(f"[SPATIAL_BATCHING] chunk_eliminated={stats.spatial_chunk_eliminated}")
        print(
            "[SPATIAL_BATCHING] by_kind="
            f"conv={stats.spatial_batched_conv} bn={stats.spatial_batched_bn} "
            f"add={stats.spatial_batched_add} pool={stats.spatial_batched_pool} "
            f"flatten={stats.spatial_batched_flatten} linear={stats.spatial_batched_linear} "
            f"elementwise={stats.spatial_batched_elementwise}"
        )
        print(
            "[SPATIAL_BATCHING] pool_detail="
            f"max={stats.spatial_batched_maxpool} avg={stats.spatial_batched_avgpool} "
            f"adaptive_avg={stats.spatial_batched_adaptive_avgpool}"
        )
        print(
            "[SPATIAL_BATCHING] temporal_stack="
            f"groups={stats.spatial_temporal_stack_groups} "
            f"flatten_inputs={stats.spatial_temporal_stack_flatten_inputs} "
            f"cat_avoided={stats.spatial_cat_avoided_by_temporal_stack_flatten}"
        )
        print(
            "[SPATIAL_BATCHING] previous_batched="
            f"groups={stats.spatial_previous_batched_groups} "
            f"inputs={stats.spatial_reused_previous_batched_inputs} "
            f"chunk_cat_avoided={stats.spatial_chunk_cat_avoided}"
        )
        print(f"[SPATIAL_BATCHING] skipped={stats.spatial_batch_skipped}")
        print(f"[SPATIAL_BATCHING] reasons={stats.reasons}")
        return stats
    except Exception as exc:
        if strict:
            raise
        stats.skip("exception", str(exc))
        print(f"[SPATIAL_BATCHING][SKIP] {exc}")
        traceback.print_exc()
        try:
            gm.graph.lint()
            gm.recompile()
        except Exception:
            traceback.print_exc()
        if dump_dir is not None:
            dump_spatial_batching([], stats, dump_dir / "spatial_batching.txt")
        return stats
