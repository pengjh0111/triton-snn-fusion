import operator
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class TemporalSpatialCanonicalizeStats:
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
    state_prune_enabled: bool = False
    state_prune_removed_final_return_states: int = 0
    state_prune_kept_states: int = 0
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
    iterations: int = 0
    final_cat_count: int = 0
    final_chunk_count: int = 0
    final_getitem_count: int = 0
    skipped: int = 0
    reasons: Dict[str, int] = field(default_factory=dict)
    log: List[str] = field(default_factory=list)

    def skip(self, reason: str, message: str):
        self.skipped += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1
        self.log.append(f"SKIP[{reason}] {message}")


def _target_text(node: torch.fx.Node) -> str:
    return str(node.target)


def _is_getitem(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and node.target is operator.getitem


def _getitem_index(node: torch.fx.Node):
    if not _is_getitem(node) or len(node.args) < 2:
        return None
    return node.args[1]


def _is_cat(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and (node.target is torch.cat or "cat" in _target_text(node))


def _is_chunk(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and (node.target is torch.chunk or "chunk" in _target_text(node))


def _is_stack(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and (node.target is torch.stack or "stack" in _target_text(node))


def _is_linear_call(node: torch.fx.Node) -> bool:
    if node.op != "call_function":
        return False
    text = _target_text(node)
    return "linear" in text and "snn_custom" not in text


def _is_add(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and (
        node.target is operator.add or node.target is torch.add or "add" in _target_text(node)
    )


def _is_div(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and (
        node.target is operator.truediv
        or node.target is torch.div
        or "truediv" in _target_text(node)
        or "div" in _target_text(node)
    )


def _chunk_input(node: torch.fx.Node):
    if not _is_chunk(node) or not node.args:
        return None
    return node.args[0] if isinstance(node.args[0], torch.fx.Node) else None


def _chunk_count(node: torch.fx.Node) -> Optional[int]:
    if not _is_chunk(node):
        return None
    if len(node.args) > 1:
        try:
            return int(node.args[1])
        except Exception:
            return None
    if "chunks" in node.kwargs:
        try:
            return int(node.kwargs["chunks"])
        except Exception:
            return None
    return None


def _cat_inputs(node: torch.fx.Node) -> Optional[List[torch.fx.Node]]:
    if not _is_cat(node) or not node.args:
        return None
    first = node.args[0]
    if not isinstance(first, (tuple, list)):
        return None
    if not all(isinstance(item, torch.fx.Node) for item in first):
        return None
    return list(first)


def _cat_dim(node: torch.fx.Node) -> Optional[int]:
    if not _is_cat(node):
        return None
    dim = node.kwargs.get("dim", None)
    if dim is None and len(node.args) > 1:
        dim = node.args[1]
    if dim is None:
        dim = 0
    return dim if isinstance(dim, int) else None


def _stack_inputs_and_dim(node: torch.fx.Node) -> Tuple[Optional[List[torch.fx.Node]], Optional[int]]:
    if not _is_stack(node) or not node.args:
        return None, None
    first = node.args[0]
    if not isinstance(first, (tuple, list)):
        return None, None
    if not all(isinstance(item, torch.fx.Node) for item in first):
        return None, None
    dim = node.kwargs.get("dim", None)
    if dim is None and len(node.args) > 1:
        dim = node.args[1]
    if dim is None:
        dim = 0
    if not isinstance(dim, int):
        return None, None
    return list(first), dim


def _ordered_getitems_from_same_source(inputs: List[torch.fx.Node]) -> Optional[torch.fx.Node]:
    if not inputs:
        return None
    if not all(_is_getitem(item) and isinstance(item.args[0], torch.fx.Node) for item in inputs):
        return None
    source = inputs[0].args[0]
    if any(item.args[0] is not source for item in inputs):
        return None
    if [_getitem_index(item) for item in inputs] != list(range(len(inputs))):
        return None
    return source


def _collect_chunk_getitems(chunk_node: torch.fx.Node) -> Optional[Dict[int, torch.fx.Node]]:
    out: Dict[int, torch.fx.Node] = {}
    for user in list(chunk_node.users):
        if not _is_getitem(user):
            return None
        idx = _getitem_index(user)
        if not isinstance(idx, int):
            return None
        out[idx] = user
    return out


def _replace_chunk_of_cat(gm: torch.fx.GraphModule, stats: TemporalSpatialCanonicalizeStats) -> bool:
    changed = False
    for chunk in list(gm.graph.nodes):
        cat = _chunk_input(chunk)
        if not isinstance(cat, torch.fx.Node) or not _is_cat(cat):
            continue
        inputs = _cat_inputs(cat)
        count = _chunk_count(chunk)
        getitems = _collect_chunk_getitems(chunk)
        if inputs is None or count != len(inputs) or getitems is None:
            continue
        if sorted(getitems) != list(range(len(inputs))):
            continue
        for idx, original in enumerate(inputs):
            getitems[idx].replace_all_uses_with(original)
        for idx in sorted(getitems, reverse=True):
            if len(getitems[idx].users) == 0:
                gm.graph.erase_node(getitems[idx])
        if len(chunk.users) == 0:
            gm.graph.erase_node(chunk)
        if len(cat.users) == 0:
            gm.graph.erase_node(cat)
        stats.canonicalize_cat_chunk_removed += 1
        changed = True
    return changed


def _replace_cat_of_chunk(gm: torch.fx.GraphModule, stats: TemporalSpatialCanonicalizeStats) -> bool:
    changed = False
    for cat in list(gm.graph.nodes):
        if not _is_cat(cat):
            continue
        inputs = _cat_inputs(cat)
        if not inputs:
            continue
        if not all(_is_getitem(item) and isinstance(item.args[0], torch.fx.Node) for item in inputs):
            continue
        chunk = inputs[0].args[0]
        if not isinstance(chunk, torch.fx.Node) or not _is_chunk(chunk):
            continue
        if any(item.args[0] is not chunk for item in inputs):
            continue
        if [_getitem_index(item) for item in inputs] != list(range(len(inputs))):
            continue
        count = _chunk_count(chunk)
        source = _chunk_input(chunk)
        if count != len(inputs) or source is None:
            continue
        cat.replace_all_uses_with(source)
        if len(cat.users) == 0:
            gm.graph.erase_node(cat)
        for item in reversed(inputs):
            if len(item.users) == 0:
                gm.graph.erase_node(item)
        if len(chunk.users) == 0:
            gm.graph.erase_node(chunk)
        stats.canonicalize_chunk_cat_removed += 1
        stats.canonicalize_getitem_cat_removed += len(inputs)
        changed = True
    return changed


def _replace_stack_of_getitems(gm: torch.fx.GraphModule, stats: TemporalSpatialCanonicalizeStats) -> bool:
    changed = False
    for stack in list(gm.graph.nodes):
        inputs, dim = _stack_inputs_and_dim(stack)
        if not inputs or dim != 0:
            continue
        if not all(_is_getitem(item) and isinstance(item.args[0], torch.fx.Node) for item in inputs):
            continue
        source = inputs[0].args[0]
        if isinstance(source, torch.fx.Node) and _is_chunk(source):
            continue
        if any(item.args[0] is not source for item in inputs):
            continue
        indices = [_getitem_index(item) for item in inputs]
        if indices != list(range(len(inputs))):
            continue

        stack.replace_all_uses_with(source)
        if len(stack.users) == 0:
            gm.graph.erase_node(stack)

        removed_getitems = 0
        for item in reversed(inputs):
            if len(item.users) == 0:
                gm.graph.erase_node(item)
                removed_getitems += 1

        stats.canonicalize_stack_getitem_removed += 1
        stats.canonicalize_getitem_stack_removed += removed_getitems
        changed = True
    return changed


def _replace_stack_of_chunk_getitems(gm: torch.fx.GraphModule, stats: TemporalSpatialCanonicalizeStats) -> bool:
    changed = False
    for stack in list(gm.graph.nodes):
        inputs, dim = _stack_inputs_and_dim(stack)
        if not inputs or dim != 0:
            continue
        chunk = _ordered_getitems_from_same_source(inputs)
        if not isinstance(chunk, torch.fx.Node) or not _is_chunk(chunk):
            continue
        source = _chunk_input(chunk)
        count = _chunk_count(chunk)
        if source is None or count != len(inputs):
            continue

        with gm.graph.inserting_before(stack):
            replacement = gm.graph.call_method("unflatten", args=(source, 0, (len(inputs), -1)))
            replacement.meta.update(getattr(stack, "meta", {}))
        stack.replace_all_uses_with(replacement)
        if len(stack.users) == 0:
            gm.graph.erase_node(stack)

        removed_getitems = 0
        for item in reversed(inputs):
            if len(item.users) == 0:
                gm.graph.erase_node(item)
                removed_getitems += 1
        if len(chunk.users) == 0:
            gm.graph.erase_node(chunk)

        stats.canonicalize_stack_chunk_removed += 1
        stats.canonicalize_getitem_stack_chunk_removed += removed_getitems
        changed = True
    return changed


def _replace_cat_linear_chunk_with_batched_linear(
    gm: torch.fx.GraphModule,
    stats: TemporalSpatialCanonicalizeStats,
) -> bool:
    changed = False
    for chunk in list(gm.graph.nodes):
        if not _is_chunk(chunk):
            continue
        linear = _chunk_input(chunk)
        if not isinstance(linear, torch.fx.Node) or not _is_linear_call(linear):
            continue
        if not linear.args or not isinstance(linear.args[0], torch.fx.Node):
            continue
        cat = linear.args[0]
        if not isinstance(cat, torch.fx.Node) or not _is_cat(cat) or _cat_dim(cat) != 0:
            continue
        if len(cat.users) != 1 or len(linear.users) != 1:
            continue

        inputs = _cat_inputs(cat)
        if not inputs:
            continue
        source = _ordered_getitems_from_same_source(inputs)
        if not isinstance(source, torch.fx.Node):
            continue
        if _chunk_count(chunk) != len(inputs):
            continue
        getitems = _collect_chunk_getitems(chunk)
        if getitems is None or sorted(getitems) != list(range(len(inputs))):
            continue

        with gm.graph.inserting_before(linear):
            batched_linear = gm.graph.call_function(
                linear.target,
                args=(source, *linear.args[1:]),
                kwargs=dict(linear.kwargs),
            )
            batched_linear.meta.update(getattr(linear, "meta", {}))

        replaced_getitems = 0
        for idx in range(len(inputs)):
            old_getitem = getitems[idx]
            with gm.graph.inserting_before(old_getitem):
                new_getitem = gm.graph.call_function(operator.getitem, args=(batched_linear, idx))
                new_getitem.meta.update(getattr(old_getitem, "meta", {}))
            old_getitem.replace_all_uses_with(new_getitem)
            if len(old_getitem.users) == 0:
                gm.graph.erase_node(old_getitem)
            replaced_getitems += 1

        if len(chunk.users) == 0:
            gm.graph.erase_node(chunk)
        if len(linear.users) == 0:
            gm.graph.erase_node(linear)
        if len(cat.users) == 0:
            gm.graph.erase_node(cat)
        for item in reversed(inputs):
            if len(item.users) == 0:
                gm.graph.erase_node(item)

        stats.canonicalize_cat_linear_chunk_removed += 1
        stats.canonicalize_cat_linear_chunk_getitem_replaced += replaced_getitems
        changed = True
    return changed


def _count_nodes(gm: torch.fx.GraphModule):
    cats = chunks = getitems = 0
    for node in gm.graph.nodes:
        if _is_cat(node):
            cats += 1
        elif _is_chunk(node):
            chunks += 1
        elif _is_getitem(node):
            getitems += 1
    return cats, chunks, getitems


def _output_node(gm: torch.fx.GraphModule) -> Optional[torch.fx.Node]:
    for node in reversed(list(gm.graph.nodes)):
        if node.op == "output":
            return node
    return None


def _output_values(gm: torch.fx.GraphModule) -> Tuple[Any, ...]:
    output = _output_node(gm)
    if output is None or not output.args:
        return tuple()
    value = output.args[0]
    if isinstance(value, tuple):
        return value
    return (value,)


def _is_state_output_node(node: Any) -> bool:
    if not isinstance(node, torch.fx.Node):
        return False
    name = node.name
    return "v_final" in name or "v_next" in name or name.endswith("_v")


def _count_returned_states(gm: torch.fx.GraphModule) -> int:
    values = _output_values(gm)
    return sum(1 for value in values[1:] if _is_state_output_node(value))


def _count_ir_stats(gm: torch.fx.GraphModule) -> Dict[str, int]:
    nodes = getitems = adds = divs = 0
    for node in gm.graph.nodes:
        nodes += 1
        if _is_getitem(node):
            getitems += 1
        elif _is_add(node):
            adds += 1
        elif _is_div(node):
            divs += 1
    return {
        "nodes": nodes,
        "getitem": getitems,
        "add": adds,
        "div": divs,
        "returned_states": _count_returned_states(gm),
    }


def _count_all_nodes(gm: torch.fx.GraphModule) -> int:
    return sum(1 for _ in gm.graph.nodes)


def _as_number(value):
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, torch.fx.Node) and value.op == "get_attr":
        return None
    return None


def _collect_add_terms(node: Any) -> Optional[Tuple[List[torch.fx.Node], List[torch.fx.Node]]]:
    terms: List[torch.fx.Node] = []
    adds: List[torch.fx.Node] = []

    def visit(value: Any) -> bool:
        if isinstance(value, torch.fx.Node) and _is_add(value):
            adds.append(value)
            if len(value.args) < 2:
                return False
            return visit(value.args[0]) and visit(value.args[1])
        if isinstance(value, torch.fx.Node):
            terms.append(value)
            return True
        number = _as_number(value)
        return number == 0

    if not visit(node):
        return None
    return terms, adds


def _is_temporal_stack_timestep_getitem(node: torch.fx.Node) -> bool:
    return _is_getitem(node) and isinstance(_getitem_index(node), int) and isinstance(node.args[0], torch.fx.Node)


def _reduce_temporal_getitem_source(
    gm: torch.fx.GraphModule,
    source: torch.fx.Node,
    temporal_len: int,
    *,
    reduce: str,
) -> torch.fx.Node:
    if _is_chunk(source):
        chunk_input = _chunk_input(source)
        if chunk_input is None:
            raise ValueError("chunk source has no tensor input")
        view = gm.graph.call_method("unflatten", args=(chunk_input, 0, (temporal_len, -1)))
        return gm.graph.call_method(reduce, args=(view,), kwargs={"dim": 0})
    return gm.graph.call_method(reduce, args=(source,), kwargs={"dim": 0})


def _rewrite_temporal_sum_div_to_mean(
    gm: torch.fx.GraphModule,
    stats: TemporalSpatialCanonicalizeStats,
) -> bool:
    changed = False
    for div in list(gm.graph.nodes):
        if not _is_div(div) or len(div.args) < 2:
            continue
        divisor = _as_number(div.args[1])
        if divisor is None:
            continue
        collected = _collect_add_terms(div.args[0])
        if collected is None:
            continue
        terms, adds = collected
        if not terms:
            continue
        if not all(isinstance(term, torch.fx.Node) and _is_temporal_stack_timestep_getitem(term) for term in terms):
            continue
        if int(divisor) != len(terms) or float(divisor) != float(len(terms)):
            continue
        if any(len(term.users) != 1 for term in terms):
            continue

        grouped_terms: Dict[torch.fx.Node, List[torch.fx.Node]] = {}
        for term in terms:
            grouped_terms.setdefault(term.args[0], []).append(term)
        if not grouped_terms:
            continue
        valid_groups = True
        for source, group in grouped_terms.items():
            indices = [_getitem_index(term) for term in group]
            if sorted(indices) != list(range(len(group))):
                valid_groups = False
                break
            if _is_chunk(source) and _chunk_count(source) != len(group):
                valid_groups = False
                break
            # A group of size 1 trivially satisfies the indices check above
            # regardless of what `source` actually is (any single index
            # equals range(1)), so at window=1 (each timestep forming its
            # own degenerate size-1 group) this loop alone cannot reject a
            # getitem into an unrelated multi-output node -- e.g. a fused
            # op's (spike, v_final) tuple getitem(0)/getitem(1) being
            # misread as a "temporal timestep getitem". _reduce_temporal_
            # getitem_source's non-chunk branch calls source.sum(dim=0)
            # unconditionally, which is only valid when source is a genuine
            # torch.stack of temporal_len tensors; require that explicitly.
            if not _is_chunk(source) and not _is_stack(source):
                valid_groups = False
                break
        if not valid_groups:
            continue

        with gm.graph.inserting_before(div):
            if len(grouped_terms) == 1:
                stack = next(iter(grouped_terms))
                replacement = _reduce_temporal_getitem_source(
                    gm,
                    stack,
                    len(terms),
                    reduce="mean",
                )
            else:
                partial_sums = [
                    _reduce_temporal_getitem_source(
                        gm,
                        stack,
                        len(grouped_terms[stack]),
                        reduce="sum",
                    )
                    for stack in grouped_terms
                ]
                replacement = partial_sums[0]
                for partial in partial_sums[1:]:
                    replacement = gm.graph.call_function(operator.add, args=(replacement, partial))
                replacement = gm.graph.call_function(operator.truediv, args=(replacement, len(terms)))
            replacement.meta.update(getattr(div, "meta", {}))
        div.replace_all_uses_with(replacement)
        if len(div.users) == 0:
            gm.graph.erase_node(div)
        for add in reversed(adds):
            if len(add.users) == 0:
                gm.graph.erase_node(add)
        removed_getitems = len(terms)
        for term in reversed(terms):
            if len(term.users) == 0:
                gm.graph.erase_node(term)
        stats.temporal_mean_rewrites += 1
        stats.temporal_mean_removed_getitems += removed_getitems
        stats.temporal_mean_removed_adds += len(adds)
        print(
            f"[KAIROS_TEMPORAL_MEAN_REWRITE] matched=True T={len(terms)} "
            f"stacks={len(grouped_terms)} removed_getitems={removed_getitems} removed_adds={len(adds)}"
        )
        changed = True
    if not changed:
        print("[KAIROS_TEMPORAL_MEAN_REWRITE] matched=False")
    return changed


def _prune_final_return_states(
    gm: torch.fx.GraphModule,
    stats: TemporalSpatialCanonicalizeStats,
    *,
    enabled: bool,
    preserve_output_contract: bool,
) -> bool:
    stats.state_prune_enabled = bool(enabled)
    print(f"[KAIROS_STATE_PRUNE] enabled={bool(enabled)}")
    if not enabled:
        stats.state_prune_kept_states = _count_returned_states(gm)
        print(
            f"[KAIROS_STATE_PRUNE] removed_final_return_states=0 "
            f"kept_states={stats.state_prune_kept_states} reason_kept=disabled"
        )
        return False

    output = _output_node(gm)
    values = _output_values(gm)
    if output is None or len(values) <= 1:
        print("[KAIROS_STATE_PRUNE] removed_final_return_states=0 kept_states=0 reason_kept=no_tuple_outputs")
        return False

    returned_states = sum(1 for value in values[1:] if _is_state_output_node(value))
    if preserve_output_contract and returned_states:
        stats.state_prune_kept_states = returned_states
        print(
            "[KAIROS_STATE_PRUNE] removed_final_return_states=0 "
            f"kept_states={returned_states} "
            "reason_kept=dynamo_state_output_contract"
        )
        return False

    last_state_idx = None
    for idx, value in enumerate(values[1:], start=1):
        if _is_state_output_node(value):
            last_state_idx = idx

    kept = [values[0]]
    removed = 0
    kept_states = 0
    for idx, value in enumerate(values[1:], start=1):
        if _is_state_output_node(value):
            if idx == last_state_idx:
                kept.append(value)
                kept_states += 1
            else:
                removed += 1
        else:
            kept.append(value)
            if _is_state_output_node(value):
                kept_states += 1
    if removed == 0:
        print(f"[KAIROS_STATE_PRUNE] removed_final_return_states=0 kept_states={kept_states} reason_kept=no_state_outputs")
        return False
    output.args = (tuple(kept),)
    stats.state_prune_removed_final_return_states += removed
    stats.state_prune_kept_states = kept_states
    print(
        f"[KAIROS_STATE_PRUNE] removed_final_return_states={removed} "
        f"kept_states={kept_states} reason_kept=final_output_or_non_state"
    )
    return True


def canonicalize_temporal_spatial_ir(
    gm: torch.fx.GraphModule,
    *,
    max_iter: int = 8,
    dump_dir: Optional[Path] = None,
    strict: bool = False,
    rewrite_temporal_mean: bool = True,
    canonicalize_stack_getitem: bool = True,
    canonicalize_chunk_stack: bool = True,
    canonicalize_cat_linear_chunk: bool = True,
    drop_intermediate_states: bool = False,
    preserve_output_contract: bool = True,
) -> TemporalSpatialCanonicalizeStats:
    stats = TemporalSpatialCanonicalizeStats()
    try:
        before_stats = _count_ir_stats(gm)
        stats.ir_nodes_before = before_stats["nodes"]
        stats.ir_getitem_before = before_stats["getitem"]
        stats.ir_add_before = before_stats["add"]
        stats.ir_div_before = before_stats["div"]
        stats.ir_returned_states_before = before_stats["returned_states"]
        print(
            "[IR_STATS_BEFORE] "
            f"num_nodes={stats.ir_nodes_before} getitem_nodes={stats.ir_getitem_before} "
            f"add_nodes={stats.ir_add_before} div_nodes={stats.ir_div_before} "
            f"returned_states={stats.ir_returned_states_before}"
        )
        changed_once = False
        changed_once |= _prune_final_return_states(
            gm,
            stats,
            enabled=drop_intermediate_states,
            preserve_output_contract=preserve_output_contract,
        )
        for iteration in range(max_iter):
            stats.iterations = iteration + 1
            changed = changed_once
            changed_once = False
            if canonicalize_chunk_stack:
                changed |= _replace_stack_of_chunk_getitems(gm, stats)
            if canonicalize_stack_getitem:
                changed |= _replace_stack_of_getitems(gm, stats)
            if canonicalize_cat_linear_chunk:
                changed |= _replace_cat_linear_chunk_with_batched_linear(gm, stats)
            changed |= _replace_cat_of_chunk(gm, stats)
            changed |= _replace_chunk_of_cat(gm, stats)
            if rewrite_temporal_mean:
                changed |= _rewrite_temporal_sum_div_to_mean(gm, stats)
            else:
                print("[KAIROS_TEMPORAL_MEAN_REWRITE] enabled=False")
            before = _count_all_nodes(gm)
            gm.graph.eliminate_dead_code()
            after = _count_all_nodes(gm)
            if after < before:
                stats.canonicalize_dead_nodes_removed += before - after
                changed = True
            gm.graph.lint()
            gm.recompile()
            if not changed:
                break
        stats.final_cat_count, stats.final_chunk_count, stats.final_getitem_count = _count_nodes(gm)
        after_stats = _count_ir_stats(gm)
        stats.ir_nodes_after = after_stats["nodes"]
        stats.ir_getitem_after = after_stats["getitem"]
        stats.ir_add_after = after_stats["add"]
        stats.ir_div_after = after_stats["div"]
        stats.ir_returned_states_after = after_stats["returned_states"]
        print(
            "[IR_STATS_AFTER] "
            f"num_nodes={stats.ir_nodes_after} getitem_nodes={stats.ir_getitem_after} "
            f"add_nodes={stats.ir_add_after} div_nodes={stats.ir_div_after} "
            f"returned_states={stats.ir_returned_states_after}"
        )
        message = (
            "[CANONICALIZE] "
            f"cat_chunk_removed={stats.canonicalize_cat_chunk_removed} "
            f"chunk_cat_removed={stats.canonicalize_chunk_cat_removed} "
            f"getitem_cat_removed={stats.canonicalize_getitem_cat_removed} "
            f"stack_getitem_removed={stats.canonicalize_stack_getitem_removed} "
            f"getitem_stack_removed={stats.canonicalize_getitem_stack_removed} "
            f"stack_chunk_removed={stats.canonicalize_stack_chunk_removed} "
            f"getitem_stack_chunk_removed={stats.canonicalize_getitem_stack_chunk_removed} "
            f"cat_linear_chunk_removed={stats.canonicalize_cat_linear_chunk_removed} "
            f"cat_linear_chunk_getitem_replaced={stats.canonicalize_cat_linear_chunk_getitem_replaced} "
            f"temporal_mean_rewrites={stats.temporal_mean_rewrites} "
            f"state_pruned={stats.state_prune_removed_final_return_states} "
            f"dead={stats.canonicalize_dead_nodes_removed} "
            f"final_cat={stats.final_cat_count} final_chunk={stats.final_chunk_count} "
            f"final_getitem={stats.final_getitem_count}"
        )
        stats.log.append(message)
        print(message)
    except Exception as exc:
        if strict:
            raise
        stats.skip("exception", str(exc))
        print(f"[CANONICALIZE][SKIP] {exc}")
        traceback.print_exc()
    if dump_dir is not None:
        dump_dir.mkdir(parents=True, exist_ok=True)
        (dump_dir / "temporal_spatial_canonicalize.txt").write_text(
            "\n".join(stats.log + [f"stats={asdict(stats)}"]) + "\n",
            encoding="utf-8",
        )
    return stats
