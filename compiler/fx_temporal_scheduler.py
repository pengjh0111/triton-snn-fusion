from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from compiler.fx_lif_temporal_rewrite import TemporalPattern, group_temporal_patterns


@dataclass
class NodeScheduleInfo:
    timestep_index: int
    layer_index: int
    layer_id: Optional[str]
    pattern_role: Optional[str]
    original_order: int


@dataclass
class TemporalScheduleResult:
    ok: bool
    scheduled_windows: int = 0
    moved_nodes: int = 0
    reason: str = ""
    node_info: Dict[str, Any] = field(default_factory=dict)


def collect_input_nodes(obj) -> List[torch.fx.Node]:
    out: List[torch.fx.Node] = []
    if isinstance(obj, torch.fx.Node):
        out.append(obj)
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            out.extend(collect_input_nodes(item))
    elif isinstance(obj, dict):
        for item in obj.values():
            out.extend(collect_input_nodes(item))
    return out


def _node_order(gm: torch.fx.GraphModule) -> Dict[torch.fx.Node, int]:
    return {node: index for index, node in enumerate(gm.graph.nodes)}


def _body_nodes(gm: torch.fx.GraphModule) -> List[torch.fx.Node]:
    return [node for node in gm.graph.nodes if node.op not in ("placeholder", "get_attr", "output")]


def _select_marker_group(patterns: List[TemporalPattern], T: int, order: Dict[torch.fx.Node, int]):
    groups = group_temporal_patterns(patterns)
    candidates = [group for group in groups if len(group.patterns) >= T]
    if not candidates:
        return None
    return min(candidates, key=lambda group: order[group.patterns[0].conv_node])


def split_fx_graph_into_timesteps(
    gm: torch.fx.GraphModule,
    T: int,
    temporal_patterns: Optional[List[TemporalPattern]] = None,
) -> List[List[torch.fx.Node]]:
    if T <= 0:
        return []
    temporal_patterns = temporal_patterns or []
    order = _node_order(gm)
    marker_group = _select_marker_group(temporal_patterns, T, order)
    if marker_group is None:
        return []

    markers = sorted(marker_group.patterns[:T], key=lambda pattern: order[pattern.conv_node])
    body_nodes = _body_nodes(gm)
    marker_orders = [order[pattern.conv_node] for pattern in markers]
    blocks: List[List[torch.fx.Node]] = []
    for index, start_order in enumerate(marker_orders):
        end_order = marker_orders[index + 1] if index + 1 < len(marker_orders) else float("inf")
        block = [node for node in body_nodes if start_order <= order[node] < end_order]
        blocks.append(block)
    return blocks if len(blocks) == T else []


def annotate_nodes_with_layer_and_timestep(
    gm: torch.fx.GraphModule,
    timestep_blocks: List[List[torch.fx.Node]],
    temporal_patterns: List[TemporalPattern],
) -> Dict[torch.fx.Node, NodeScheduleInfo]:
    order = _node_order(gm)
    info: Dict[torch.fx.Node, NodeScheduleInfo] = {}
    for timestep_index, block in enumerate(timestep_blocks):
        for local_index, node in enumerate(block):
            info[node] = NodeScheduleInfo(
                timestep_index=timestep_index,
                layer_index=local_index * 10,
                layer_id=None,
                pattern_role=None,
                original_order=order[node],
            )

    first_seen_layer_ids: List[str] = []
    seen: Set[str] = set()
    for pattern in sorted(temporal_patterns, key=lambda pattern: order[pattern.conv_node]):
        if pattern.layer_id not in seen:
            first_seen_layer_ids.append(pattern.layer_id)
            seen.add(pattern.layer_id)
    layer_rank = {layer_id: index for index, layer_id in enumerate(first_seen_layer_ids)}

    for pattern in temporal_patterns:
        base = layer_rank.get(pattern.layer_id, 0) * 1000
        roles = [
            (pattern.conv_node, "conv", 0),
            (pattern.bn_node, "bn", 1),
            (pattern.lif_node, "lif", 3),
            (pattern.spike_getitem, "spike", 4),
            (pattern.v_getitem, "v_next", 5),
        ]
        for node, role, offset in roles:
            if node not in info:
                continue
            info[node] = NodeScheduleInfo(
                timestep_index=info[node].timestep_index,
                layer_index=base + offset,
                layer_id=pattern.layer_id,
                pattern_role=role,
                original_order=order[node],
            )
        v_prev = pattern.lif_node.args[1] if len(pattern.lif_node.args) > 1 else None
        if isinstance(v_prev, torch.fx.Node) and v_prev in info:
            info[v_prev] = NodeScheduleInfo(
                timestep_index=info[v_prev].timestep_index,
                layer_index=base + 2,
                layer_id=pattern.layer_id,
                pattern_role="v_prev",
                original_order=order[v_prev],
            )
    return info


def build_scheduled_order_for_window(
    nodes: List[torch.fx.Node],
    info: Dict[torch.fx.Node, NodeScheduleInfo],
) -> List[torch.fx.Node]:
    window_set = set(nodes)
    deps: Dict[torch.fx.Node, Set[torch.fx.Node]] = {}
    users: Dict[torch.fx.Node, Set[torch.fx.Node]] = {node: set() for node in nodes}
    for node in nodes:
        node_deps = {dep for dep in collect_input_nodes((node.args, node.kwargs)) if dep in window_set}
        deps[node] = node_deps
        for dep in node_deps:
            users.setdefault(dep, set()).add(node)

    emitted: Set[torch.fx.Node] = set()
    ready: Set[torch.fx.Node] = {node for node in nodes if not deps[node]}
    scheduled: List[torch.fx.Node] = []

    def priority(node: torch.fx.Node):
        node_info = info[node]
        return (node_info.layer_index, node_info.timestep_index, node_info.original_order)

    while ready:
        node = min(ready, key=priority)
        ready.remove(node)
        emitted.add(node)
        scheduled.append(node)
        for user in users.get(node, ()):
            if user in emitted:
                continue
            if deps[user].issubset(emitted):
                ready.add(user)

    if len(scheduled) != len(nodes):
        blocked = [node.name for node in nodes if node not in emitted][:20]
        raise RuntimeError(f"could not schedule all nodes; blocked={blocked}")
    return scheduled


def _dump_schedule_order(path: Path, nodes: List[torch.fx.Node], info: Dict[torch.fx.Node, NodeScheduleInfo]):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for index, node in enumerate(nodes):
        node_info = info.get(node)
        if node_info is None:
            suffix = "timestep=None layer=None layer_id=None role=None"
        else:
            suffix = (
                f"timestep={node_info.timestep_index} layer={node_info.layer_index} "
                f"layer_id={node_info.layer_id} role={node_info.pattern_role}"
            )
        lines.append(f"order={index} name={node.name} op={node.op} target={node.target} {suffix}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _dump_schedule_windows(path: Path, windows: List[Tuple[int, int, List[torch.fx.Node]]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for window_id, start_t, nodes in windows:
        end_t = start_t + 1
        if nodes:
            timesteps = sorted({getattr(node, "_chronos_timestep", start_t) for node in nodes})
            if timesteps:
                end_t = timesteps[-1]
        lines.append(f"window_{window_id}: timesteps={start_t}..{end_t} nodes={len(nodes)}")
        lines.append(f"  first={nodes[0].name if nodes else '<none>'}")
        lines.append(f"  last={nodes[-1].name if nodes else '<none>'}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _dump_node_info(path: Path, info: Dict[torch.fx.Node, NodeScheduleInfo]):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for node, node_info in sorted(info.items(), key=lambda item: item[1].original_order):
        lines.append(
            f"name={node.name} original={node_info.original_order} timestep={node_info.timestep_index} "
            f"layer={node_info.layer_index} layer_id={node_info.layer_id} role={node_info.pattern_role}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def reorder_fx_graph_by_temporal_windows(
    gm: torch.fx.GraphModule,
    T: int,
    window_size: int,
    temporal_patterns: List[TemporalPattern],
    dump_dir: Optional[Path] = None,
    strict: bool = False,
) -> TemporalScheduleResult:
    if window_size <= 1:
        return TemporalScheduleResult(ok=True, reason="window_size <= 1; no scheduling needed")

    try:
        original_nodes = list(gm.graph.nodes)
        order = _node_order(gm)
        timestep_blocks = split_fx_graph_into_timesteps(gm, T, temporal_patterns)
        if len(timestep_blocks) != T:
            raise RuntimeError(f"could not split graph into {T} timestep blocks; got {len(timestep_blocks)}")

        info = annotate_nodes_with_layer_and_timestep(gm, timestep_blocks, temporal_patterns)
        if dump_dir is not None:
            _dump_schedule_order(dump_dir / "temporal_schedule_before.txt", original_nodes, info)
            _dump_node_info(dump_dir / "temporal_schedule_node_info.txt", info)

        covered: Set[torch.fx.Node] = set()
        scheduled_windows: List[Tuple[int, int, List[torch.fx.Node]]] = []
        for window_id, start in enumerate(range(0, T, window_size)):
            blocks = timestep_blocks[start : start + window_size]
            if not blocks:
                continue
            window_nodes = [node for block in blocks for node in block]
            for timestep_offset, block in enumerate(blocks):
                for node in block:
                    setattr(node, "_chronos_timestep", start + timestep_offset)
            scheduled = build_scheduled_order_for_window(window_nodes, info)
            scheduled_windows.append((window_id, start, scheduled))
            covered.update(scheduled)

        params = [node for node in original_nodes if node.op in ("placeholder", "get_attr")]
        outputs = [node for node in original_nodes if node.op == "output"]
        non_special_uncovered = [
            node for node in original_nodes if node.op not in ("placeholder", "get_attr", "output") and node not in covered
        ]
        scheduled_body = [node for _window_id, _start, nodes in scheduled_windows for node in nodes]
        desired_order = params + non_special_uncovered + scheduled_body + outputs

        seen: Set[torch.fx.Node] = set()
        deduped_order: List[torch.fx.Node] = []
        for node in desired_order:
            if node in seen:
                continue
            deduped_order.append(node)
            seen.add(node)
        if len(deduped_order) != len(original_nodes):
            missing = [node.name for node in original_nodes if node not in seen]
            raise RuntimeError(f"scheduled order lost nodes: {missing[:20]}")

        previous = deduped_order[0]
        for node in deduped_order[1:]:
            previous.append(node)
            previous = node

        gm.graph.lint()
        gm.recompile()

        after_nodes = list(gm.graph.nodes)
        moved_nodes = sum(1 for old, new in zip(original_nodes, after_nodes) if old is not new)
        if dump_dir is not None:
            _dump_schedule_order(dump_dir / "temporal_schedule_after.txt", after_nodes, info)
            _dump_schedule_windows(dump_dir / "temporal_schedule_windows.txt", scheduled_windows)

        return TemporalScheduleResult(
            ok=True,
            scheduled_windows=len(scheduled_windows),
            moved_nodes=moved_nodes,
            reason="",
            node_info={node.name: vars(node_info) for node, node_info in info.items()},
        )
    except Exception as exc:
        reason = str(exc)
        if strict:
            raise
        print(f"[SCHEDULE][SKIP] {reason}")
        return TemporalScheduleResult(ok=False, reason=reason)
