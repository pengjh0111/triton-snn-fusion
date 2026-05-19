import operator
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    spatial_batch_skipped: int = 0
    reasons: Dict[str, int] = field(default_factory=dict)
    log: List[str] = field(default_factory=list)

    def skip(self, reason: str, message: str):
        self.spatial_batch_skipped += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1
        self.log.append(f"SKIP[{reason}] {message}")


@dataclass
class SpatialBatchChain:
    groups: List[SpatialBatchGroup]


@dataclass
class BatchedGroupRewriteResult:
    group: SpatialBatchGroup
    batched_output_node: torch.fx.Node
    chunk_node: Optional[torch.fx.Node] = None
    getitem_nodes: List[torch.fx.Node] = field(default_factory=list)
    output_tuple_key: Tuple[torch.fx.Node, ...] = field(default_factory=tuple)


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
        return (value.op, str(value.target), value.name)
    if isinstance(value, (tuple, list)):
        return tuple(_node_ref(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _node_ref(item)) for key, item in value.items()))
    return value


def _signature_without_input(node: torch.fx.Node, kind: str) -> Tuple[Any, ...]:
    if kind == "flatten":
        start_dim, end_dim = _flatten_dims(node)
        return (node.op, "flatten", kind, start_dim, end_dim)
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
    if "maxpool" in enabled and _is_maxpool_node(gm, node):
        return "maxpool"
    if "linear" in enabled and _is_linear_node(gm, node):
        return "linear"
    if "flatten" in enabled and _is_flatten_node(node):
        return "flatten"
    if "avgpool" in enabled and _is_adaptive_avg_pool_node(gm, node):
        return "adaptive_avg_pool"
    return None


def _candidate_input(node: torch.fx.Node) -> Optional[torch.fx.Node]:
    if not node.args:
        return None
    first = node.args[0]
    return first if isinstance(first, torch.fx.Node) else None


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
    kind = _candidate_kind(gm, node, enabled_ops)
    if kind is None:
        return None
    if kind == "flatten" and not _flatten_is_batch_safe(node):
        stats.skip("unsafe_flatten", f"node={node.name} flatten start_dim must be 1")
        return None
    timestep = _get_chronos_meta(node, "timestep")
    if not isinstance(timestep, int):
        stats.skip("missing_timestep", f"node={node.name} kind={kind} has no _chronos_timestep")
        return None
    input_node = _candidate_input(node)
    if input_node is None:
        stats.skip("missing_input", f"node={node.name} kind={kind} has no tensor input node")
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
        key = (candidate.window_id, candidate.occurrence, candidate.signature)
        grouped.setdefault(key, []).append(candidate)

    groups: List[SpatialBatchGroup] = []
    for (_window_id, _occurrence, _signature), items in grouped.items():
        items = sorted(items, key=lambda item: item.timestep)
        if len(items) != temporal_window:
            stats.skip(
                "incomplete_window",
                f"kind={items[0].kind} window={items[0].window_id} occurrence={items[0].occurrence} "
                f"size={len(items)} expected={temporal_window}",
            )
            continue
        expected_timesteps = list(range(items[0].window_id * temporal_window, (items[0].window_id + 1) * temporal_window))
        if [item.timestep for item in items] != expected_timesteps:
            stats.skip(
                "non_contiguous_timesteps",
                f"kind={items[0].kind} window={items[0].window_id} timesteps={[item.timestep for item in items]}",
            )
            continue
        first = items[0]
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


def _external_users_for_chain(chain: SpatialBatchChain) -> List[torch.fx.Node]:
    chain_nodes = {candidate.node for group in chain.groups for candidate in group.candidates}
    out: List[torch.fx.Node] = []
    seen = set()
    for group in chain.groups:
        for candidate in group.candidates:
            for user in candidate.node.users:
                if user in chain_nodes or user in seen:
                    continue
                out.append(user)
                seen.add(user)
    return out


def _select_chain_insertion_point(
    gm: torch.fx.GraphModule,
    chain: SpatialBatchChain,
) -> Tuple[Optional[torch.fx.Node], str, str]:
    order = {node: index for index, node in enumerate(gm.graph.nodes)}
    root = chain.groups[0]
    first_candidate = root.candidates[0].node
    input_nodes = list(_group_input_tuple(root))
    if not input_nodes:
        return None, "before", "chain has no input nodes"

    max_input = max(input_nodes, key=lambda node: order.get(node, -1))
    max_input_order = order.get(max_input, -1)
    min_candidate_order = min(order[candidate.node] for group in chain.groups for candidate in group.candidates)
    if max_input_order < min_candidate_order:
        return first_candidate, "before", ""

    external_users = _external_users_for_chain(chain)
    early_users = [user.name for user in external_users if order.get(user, max_input_order + 1) <= max_input_order]
    if early_users:
        return None, "before", f"external users before max input {max_input.name}: {early_users}"
    return max_input, "after", ""


def _has_internal_data_dependency(group: SpatialBatchGroup) -> Tuple[bool, str]:
    nodes = {candidate.node for candidate in group.candidates}
    for candidate in group.candidates:
        deps = set(_collect_input_nodes((candidate.node.args, candidate.node.kwargs)))
        internal = deps & nodes
        if internal:
            return True, f"node {candidate.node.name} depends on {[node.name for node in internal]}"
    return False, ""


def _make_batched_call(gm: torch.fx.GraphModule, group: SpatialBatchGroup, input_node: torch.fx.Node) -> torch.fx.Node:
    first = group.candidates[0].node
    new_args = (input_node,) + tuple(first.args[1:])
    if first.op == "call_module":
        return gm.graph.call_module(first.target, args=new_args, kwargs=dict(first.kwargs))
    if first.op == "call_method":
        return gm.graph.call_method(first.target, args=new_args, kwargs=dict(first.kwargs))
    return gm.graph.call_function(first.target, args=new_args, kwargs=dict(first.kwargs))


def _group_input_tuple(group: SpatialBatchGroup) -> Tuple[torch.fx.Node, ...]:
    return tuple(candidate.input_node for candidate in group.candidates)


def _group_output_tuple(group: SpatialBatchGroup) -> Tuple[torch.fx.Node, ...]:
    return tuple(candidate.node for candidate in group.candidates)


def _ordered_groups(gm: torch.fx.GraphModule, groups: List[SpatialBatchGroup]) -> List[SpatialBatchGroup]:
    order = {node: index for index, node in enumerate(gm.graph.nodes)}
    return sorted(groups, key=lambda group: order[group.candidates[0].node])


def build_spatial_batch_chains(gm: torch.fx.GraphModule, groups: List[SpatialBatchGroup]) -> List[SpatialBatchChain]:
    ordered = _ordered_groups(gm, groups)
    output_to_index = {_group_output_tuple(group): index for index, group in enumerate(ordered)}
    predecessor: Dict[int, int] = {}
    successors: Dict[int, List[int]] = {index: [] for index in range(len(ordered))}

    for index, group in enumerate(ordered):
        pred_index = output_to_index.get(_group_input_tuple(group))
        if pred_index is None:
            continue
        if pred_index >= index:
            continue
        predecessor[index] = pred_index
        successors[pred_index].append(index)

    consumed = set()
    chains: List[SpatialBatchChain] = []
    for index, group in enumerate(ordered):
        if index in consumed:
            continue
        if index in predecessor and len(successors.get(predecessor[index], [])) == 1:
            continue
        chain_indices = [index]
        consumed.add(index)
        current = index
        while len(successors.get(current, [])) == 1:
            nxt = successors[current][0]
            if predecessor.get(nxt) != current or nxt in consumed:
                break
            chain_indices.append(nxt)
            consumed.add(nxt)
            current = nxt
        chains.append(SpatialBatchChain(groups=[ordered[item] for item in chain_indices]))
    return chains


def _next_group_candidate_nodes(chain: SpatialBatchChain, group_index: int) -> set:
    if group_index + 1 >= len(chain.groups):
        return set()
    return {candidate.node for candidate in chain.groups[group_index + 1].candidates}


def _group_has_external_users(chain: SpatialBatchChain, group_index: int) -> bool:
    allowed = _next_group_candidate_nodes(chain, group_index)
    for candidate in chain.groups[group_index].candidates:
        for user in candidate.node.users:
            if user not in allowed:
                return True
    return False


def _materialize_group_chunks(
    gm: torch.fx.GraphModule,
    group: SpatialBatchGroup,
    batched_node: torch.fx.Node,
) -> BatchedGroupRewriteResult:
    chunks_node = gm.graph.call_function(torch.chunk, args=(batched_node, len(group.candidates), 0))
    chunks_node.name = f"{batched_node.name}_chunks"
    chunk_nodes = []
    for index, _candidate in enumerate(group.candidates):
        chunk_node = gm.graph.call_function(operator.getitem, args=(chunks_node, index))
        chunk_node.name = f"{batched_node.name}_t{index}"
        chunk_nodes.append(chunk_node)
    for candidate, chunk_node in zip(group.candidates, chunk_nodes):
        candidate.node.replace_all_uses_with(chunk_node)
    return BatchedGroupRewriteResult(
        group=group,
        batched_output_node=batched_node,
        chunk_node=chunks_node,
        getitem_nodes=chunk_nodes,
        output_tuple_key=_group_output_tuple(group),
    )


def rewrite_spatial_batch_chain(
    gm: torch.fx.GraphModule,
    chain: SpatialBatchChain,
    stats: SpatialBatchingStats,
) -> Tuple[bool, str]:
    root = chain.groups[0]
    first_node = root.candidates[0].node
    input_nodes = list(_group_input_tuple(root))
    insert_node, insert_mode, reason = _select_chain_insertion_point(gm, chain)
    if insert_node is None:
        return False, reason

    for group in chain.groups:
        has_dep, reason = _has_internal_data_dependency(group)
        if has_dep:
            return False, reason

    batched_results: List[BatchedGroupRewriteResult] = []
    materialized = 0
    insert_context = gm.graph.inserting_before(insert_node) if insert_mode == "before" else gm.graph.inserting_after(insert_node)
    with insert_context:
        cat_node = gm.graph.call_function(torch.cat, args=(input_nodes, 0))
        cat_node.name = f"{first_node.name}_spatial_batch_cat"
        current_input = cat_node
        for group_index, group in enumerate(chain.groups):
            batched_node = _make_batched_call(gm, group, current_input)
            batched_node.name = f"{group.candidates[0].node.name}_spatial_batch_{group.kind}"
            needs_chunks = group_index == len(chain.groups) - 1 or _group_has_external_users(chain, group_index)
            if needs_chunks:
                result = _materialize_group_chunks(gm, group, batched_node)
                materialized += 1
            else:
                result = BatchedGroupRewriteResult(
                    group=group,
                    batched_output_node=batched_node,
                    output_tuple_key=_group_output_tuple(group),
                )
            batched_results.append(result)
            current_input = batched_node

    order = {node: index for index, node in enumerate(gm.graph.nodes)}
    original_nodes = []
    for group in chain.groups:
        original_nodes.extend(candidate.node for candidate in group.candidates)
    for node in sorted(set(original_nodes), key=lambda item: order[item], reverse=True):
        if len(node.users) != 0:
            return False, f"node {node.name} still has users after replacement"
        gm.graph.erase_node(node)

    chain_len = len(chain.groups)
    if chain_len > 1:
        stats.spatial_batch_chains += 1
        stats.spatial_chain_groups += chain_len
        stats.spatial_cat_eliminated += chain_len - 1
        stats.spatial_chunk_eliminated += chain_len - materialized
        kinds = "->".join(group.kind for group in chain.groups)
        size = len(root.candidates)
        message = (
            f"[SPATIAL_BATCHING][CHAIN] groups={chain_len} kind={kinds} "
            f"window={root.window_id} size={size}"
        )
        stats.log.append(message)
        print(message)
        if chain_len - 1 or chain_len - materialized:
            elim = (
                f"[SPATIAL_BATCHING][ELIMINATE] cat={chain_len - 1} "
                f"chunk={chain_len - materialized}"
            )
            stats.log.append(elim)
            print(elim)
    return True, ""


def rewrite_spatial_batch_group(gm: torch.fx.GraphModule, group: SpatialBatchGroup) -> Tuple[bool, str]:
    first_node = group.candidates[0].node
    input_nodes = [candidate.input_node for candidate in group.candidates]
    ok, reason = _all_inputs_available_before(gm, input_nodes, first_node)
    if not ok:
        return False, reason
    has_dep, reason = _has_internal_data_dependency(group)
    if has_dep:
        return False, reason

    with gm.graph.inserting_before(first_node):
        cat_node = gm.graph.call_function(torch.cat, args=(input_nodes, 0))
        cat_node.name = f"{first_node.name}_spatial_batch_cat"
        batched_node = _make_batched_call(gm, group, cat_node)
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
    enable_chain: bool = True,
) -> SpatialBatchingStats:
    stats = SpatialBatchingStats()
    if temporal_window <= 1:
        stats.skip("window_le_1", f"temporal_window={temporal_window}")
        return stats

    try:
        candidates = collect_spatial_batch_candidates(gm, temporal_window, enabled_ops, stats)
        groups = group_spatial_batch_candidates(candidates, temporal_window, stats)
        stats.spatial_batch_groups = len(groups)

        if enable_chain:
            chains = build_spatial_batch_chains(gm, groups)
            for chain in chains:
                ok, reason = rewrite_spatial_batch_chain(gm, chain, stats)
                if not ok:
                    group = chain.groups[0]
                    stats.skip("rewrite_skip", f"kind={group.kind} window={group.window_id}: {reason}")
                    continue
                batched = sum(len(group.candidates) for group in chain.groups)
                stats.spatial_batched_ops += batched
                head = chain.groups[0]
                message = (
                    f"[SPATIAL_BATCHING][REWRITE] kind={'->'.join(group.kind for group in chain.groups)} "
                    f"window={head.window_id} occurrence={head.occurrence} ops={batched}"
                )
                stats.log.append(message)
                print(message)
        else:
            for group in groups:
                ok, reason = rewrite_spatial_batch_group(gm, group)
                if not ok:
                    stats.skip("rewrite_skip", f"kind={group.kind} window={group.window_id}: {reason}")
                    continue
                stats.spatial_batched_ops += len(group.candidates)
                message = (
                    f"[SPATIAL_BATCHING][REWRITE] kind={group.kind} window={group.window_id} "
                    f"occurrence={group.occurrence} size={len(group.candidates)}"
                )
                stats.log.append(message)
                print(message)

        gm.graph.lint()
        gm.recompile()

        if dump_dir is not None:
            dump_spatial_batching(groups, stats, dump_dir / "spatial_batching.txt")

        print(f"[SPATIAL_BATCHING] groups={stats.spatial_batch_groups}")
        print(f"[SPATIAL_BATCHING] batched_ops={stats.spatial_batched_ops}")
        print(f"[SPATIAL_BATCHING] chains={stats.spatial_batch_chains}")
        print(f"[SPATIAL_BATCHING] chain_groups={stats.spatial_chain_groups}")
        print(f"[SPATIAL_BATCHING] cat_eliminated={stats.spatial_cat_eliminated}")
        print(f"[SPATIAL_BATCHING] chunk_eliminated={stats.spatial_chunk_eliminated}")
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
