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
        for group in grouped_terms.values():
            indices = [_getitem_index(term) for term in group]
            if sorted(indices) != list(range(len(group))):
                valid_groups = False
                break
        if not valid_groups:
            continue

        with gm.graph.inserting_before(div):
            if len(grouped_terms) == 1:
                stack = next(iter(grouped_terms))
                replacement = gm.graph.call_method("mean", args=(stack,), kwargs={"dim": 0})
            else:
                partial_sums = [
                    gm.graph.call_method("sum", args=(stack,), kwargs={"dim": 0})
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
            f"[CHRONOS_TEMPORAL_MEAN_REWRITE] matched=True T={len(terms)} "
            f"stacks={len(grouped_terms)} removed_getitems={removed_getitems} removed_adds={len(adds)}"
        )
        changed = True
    if not changed:
        print("[CHRONOS_TEMPORAL_MEAN_REWRITE] matched=False")
    return changed


def _prune_final_return_states(
    gm: torch.fx.GraphModule,
    stats: TemporalSpatialCanonicalizeStats,
    *,
    enabled: bool,
    preserve_output_contract: bool,
) -> bool:
    stats.state_prune_enabled = bool(enabled)
    print(f"[CHRONOS_STATE_PRUNE] enabled={bool(enabled)}")
    if not enabled:
        stats.state_prune_kept_states = _count_returned_states(gm)
        print(
            f"[CHRONOS_STATE_PRUNE] removed_final_return_states=0 "
            f"kept_states={stats.state_prune_kept_states} reason_kept=disabled"
        )
        return False

    output = _output_node(gm)
    values = _output_values(gm)
    if output is None or len(values) <= 1:
        print("[CHRONOS_STATE_PRUNE] removed_final_return_states=0 kept_states=0 reason_kept=no_tuple_outputs")
        return False

    returned_states = sum(1 for value in values[1:] if _is_state_output_node(value))
    if preserve_output_contract and returned_states:
        stats.state_prune_kept_states = returned_states
        print(
            "[CHRONOS_STATE_PRUNE] removed_final_return_states=0 "
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
        print(f"[CHRONOS_STATE_PRUNE] removed_final_return_states=0 kept_states={kept_states} reason_kept=no_state_outputs")
        return False
    output.args = (tuple(kept),)
    stats.state_prune_removed_final_return_states += removed
    stats.state_prune_kept_states = kept_states
    print(
        f"[CHRONOS_STATE_PRUNE] removed_final_return_states={removed} "
        f"kept_states={kept_states} reason_kept=final_output_or_non_state"
    )
    return True


def canonicalize_temporal_spatial_ir(
    gm: torch.fx.GraphModule,
    *,
    max_iter: int = 8,
    dump_dir: Optional[Path] = None,
    strict: bool = False,
    rewrite_temporal_mean: bool = False,
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
        if rewrite_temporal_mean:
            changed_once |= _rewrite_temporal_sum_div_to_mean(gm, stats)
        else:
            print("[CHRONOS_TEMPORAL_MEAN_REWRITE] enabled=False")
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
            changed |= _replace_cat_of_chunk(gm, stats)
            changed |= _replace_chunk_of_cat(gm, stats)
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
