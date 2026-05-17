import operator
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

import runtime.snn_custom_ops  # noqa: F401 - ensure custom op registration
from compiler.fx_lif_rewrite import (
    _insert_get_attr_before,
    _is_zeros_like_of,
    _parse_conv_call_args,
    add_tensor_attr,
    extract_batch_norm_params,
    extract_conv2d_tensors,
    find_tuple_getitems,
    fold_bn_into_conv_params,
    is_batch_norm_inference_node,
    is_conv_node,
    is_custom_lif_state_node,
)


@dataclass
class TemporalPattern:
    layer_id: str
    timestep_index: int
    conv_node: torch.fx.Node
    bn_node: torch.fx.Node
    lif_node: torch.fx.Node
    spike_getitem: torch.fx.Node
    v_getitem: torch.fx.Node
    conv_input: torch.fx.Node
    conv_weight_key: str
    bn_key: str
    v_prev_node: torch.fx.Node
    v_next_node: torch.fx.Node
    lif_params: Tuple[Any, Any, Any, Any]


@dataclass
class TemporalGroup:
    layer_id: str
    patterns: List[TemporalPattern]


@dataclass
class TemporalWindow:
    layer_id: str
    window_id: int
    patterns: List[TemporalPattern]


@dataclass
class TemporalRewriteStats:
    temporal_groups: int = 0
    temporal_windows: int = 0
    temporal_replaced_windows: int = 0
    temporal_replaced_patterns: int = 0
    temporal_skipped_windows: int = 0
    single_step_replaced_patterns: int = 0
    log: List[str] = field(default_factory=list)


def _node_key(value) -> Optional[str]:
    if value is None:
        return "None"
    if isinstance(value, torch.fx.Node):
        if value.op == "placeholder":
            return f"placeholder:{value.name}"
        if value.op == "get_attr":
            return f"get_attr:{value.target}"
        return f"{value.op}:{value.name}"
    if isinstance(value, torch.Tensor):
        return f"tensor:{tuple(value.shape)}:{value.dtype}:{value.device}"
    return repr(value)


def _extract_conv_graph_args(conv_node: torch.fx.Node):
    if conv_node.op == "call_module":
        return conv_node.args[0], f"module:{conv_node.target}.weight"
    if conv_node.op == "call_function":
        conv_input, weight_arg, _bias_arg, _stride, _padding, _dilation, _groups = _parse_conv_call_args(conv_node)
        return conv_input, _node_key(weight_arg)
    return None, None


def _extract_bn_key(bn_node: torch.fx.Node) -> str:
    if bn_node.op == "call_module":
        return f"module:{bn_node.target}.running_mean|module:{bn_node.target}.running_var"
    args = list(bn_node.args)
    running_mean = args[1] if len(args) > 1 else bn_node.kwargs.get("running_mean")
    running_var = args[2] if len(args) > 2 else bn_node.kwargs.get("running_var")
    return f"{_node_key(running_mean)}|{_node_key(running_var)}"


def extract_layer_id(gm: torch.fx.GraphModule, conv_node: torch.fx.Node, bn_node: torch.fx.Node) -> Optional[str]:
    _conv_input, conv_weight_key = _extract_conv_graph_args(conv_node)
    if not conv_weight_key:
        return None
    bn_key = _extract_bn_key(bn_node)
    return f"{conv_weight_key}|{bn_key}"


def _is_batch_norm_node(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    return is_batch_norm_inference_node(gm, node)


def _lif_state_is_usable(lif_node: torch.fx.Node) -> Tuple[bool, str]:
    getitems = find_tuple_getitems(lif_node)
    if 0 not in getitems:
        return False, "missing spike getitem[0]"
    if 1 not in getitems:
        return False, "missing v_next getitem[1]"
    non_getitem_users = [user.name for user in lif_node.users if not (user.op == "call_function" and user.target is operator.getitem)]
    if non_getitem_users:
        return False, f"lif_state has non-getitem users {non_getitem_users}"
    return True, ""


def collect_conv_bn_lif_state_patterns(gm: torch.fx.GraphModule) -> List[TemporalPattern]:
    raw: List[Tuple[str, torch.fx.Node, torch.fx.Node, torch.fx.Node]] = []
    for node in gm.graph.nodes:
        if not is_conv_node(gm, node):
            continue
        conv_users = list(node.users)
        if len(conv_users) != 1:
            continue
        bn_node = conv_users[0]
        if not _is_batch_norm_node(gm, bn_node):
            continue
        lif_candidates = [user for user in bn_node.users if is_custom_lif_state_node(user)]
        if len(lif_candidates) != 1:
            continue
        lif_node = lif_candidates[0]
        ok, reason = _lif_state_is_usable(lif_node)
        if not ok:
            print(f"[SKIP][TEMPORAL] lif={lif_node.name}: {reason}")
            continue
        layer_id = extract_layer_id(gm, node, bn_node)
        if layer_id is None:
            print(f"[SKIP][TEMPORAL] conv={node.name}: cannot extract layer_id")
            continue
        raw.append((layer_id, node, bn_node, lif_node))

    counts: Dict[str, int] = {}
    patterns: List[TemporalPattern] = []
    for layer_id, conv_node, bn_node, lif_node in raw:
        getitems = find_tuple_getitems(lif_node)
        conv_input, conv_weight_key = _extract_conv_graph_args(conv_node)
        bn_key = _extract_bn_key(bn_node)
        timestep_index = counts.get(layer_id, 0)
        counts[layer_id] = timestep_index + 1
        patterns.append(
            TemporalPattern(
                layer_id=layer_id,
                timestep_index=timestep_index,
                conv_node=conv_node,
                bn_node=bn_node,
                lif_node=lif_node,
                spike_getitem=getitems[0],
                v_getitem=getitems[1],
                conv_input=conv_input,
                conv_weight_key=conv_weight_key or "",
                bn_key=bn_key,
                v_prev_node=lif_node.args[1],
                v_next_node=getitems[1],
                lif_params=tuple(lif_node.args[2:6]),
            )
        )
    return patterns


def group_temporal_patterns(patterns: List[TemporalPattern]) -> List[TemporalGroup]:
    grouped: Dict[str, List[TemporalPattern]] = {}
    for pattern in patterns:
        grouped.setdefault(pattern.layer_id, []).append(pattern)
    return [TemporalGroup(layer_id=layer_id, patterns=items) for layer_id, items in grouped.items()]


def check_temporal_state_chain(patterns: List[TemporalPattern]) -> Tuple[bool, str]:
    for prev, nxt in zip(patterns, patterns[1:]):
        if nxt.v_prev_node is prev.v_getitem:
            continue
        return False, f"{prev.v_getitem.name} does not feed {nxt.lif_node.name} v_prev"
    return True, ""


def make_temporal_windows(groups: List[TemporalGroup], window_size: int, allow_tail: bool) -> List[TemporalWindow]:
    if window_size <= 1:
        return []
    windows: List[TemporalWindow] = []
    for group in groups:
        ok, reason = check_temporal_state_chain(group.patterns)
        if not ok:
            print(f"[SKIP][TEMPORAL] layer={group.layer_id}: state chain not continuous: {reason}")
            continue
        window_id = 0
        for start in range(0, len(group.patterns), window_size):
            chunk = group.patterns[start : start + window_size]
            if len(chunk) < window_size and not allow_tail:
                print(f"[SKIP][TEMPORAL] layer={group.layer_id}: tail size={len(chunk)} < window={window_size}")
                continue
            if len(chunk) <= 1:
                continue
            windows.append(TemporalWindow(layer_id=group.layer_id, window_id=window_id, patterns=chunk))
            window_id += 1
    return windows


def _same_lif_params(patterns: List[TemporalPattern]) -> bool:
    first = patterns[0].lif_params
    return all(pattern.lif_params == first for pattern in patterns)


def _middle_v_next_has_no_external_uses(window: TemporalWindow) -> Tuple[bool, str]:
    patterns = window.patterns
    for idx, pattern in enumerate(patterns[:-1]):
        allowed = patterns[idx + 1].lif_node
        external = [user.name for user in pattern.v_getitem.users if user is not allowed]
        if external:
            return False, f"middle v_next {pattern.v_getitem.name} has external users {external}"
    return True, ""


def _erase_if_unused(gm: torch.fx.GraphModule, node: torch.fx.Node):
    if len(node.users) == 0:
        gm.graph.erase_node(node)


def _materialize_scalar_zero_v_init(gm: torch.fx.GraphModule, before: torch.fx.Node, like_tensor: torch.Tensor):
    zero = torch.tensor(0.0, device=like_tensor.device, dtype=like_tensor.dtype)
    attr = add_tensor_attr(gm, "_fx_zero_scalar_v_init", zero)
    return _insert_get_attr_before(gm, before, attr)


def _all_inputs_available_before(gm: torch.fx.GraphModule, inputs: List[Any], before: torch.fx.Node) -> Tuple[bool, str]:
    order = {node: index for index, node in enumerate(gm.graph.nodes)}
    before_order = order[before]
    for value in inputs:
        if isinstance(value, torch.fx.Node) and order.get(value, before_order + 1) >= before_order:
            return False, f"input {value.name} is not defined before insertion point {before.name}"
    return True, ""


def _cleanup_window_nodes(gm: torch.fx.GraphModule, window: TemporalWindow):
    order = {node: index for index, node in enumerate(gm.graph.nodes)}
    candidates = []
    for pattern in window.patterns:
        v_prev = pattern.lif_node.args[1] if len(pattern.lif_node.args) > 1 else None
        candidates.extend([pattern.spike_getitem, pattern.v_getitem, pattern.lif_node])
        if isinstance(v_prev, torch.fx.Node):
            candidates.append(v_prev)
        candidates.extend([pattern.bn_node, pattern.conv_node])
    unique = []
    seen = set()
    for node in candidates:
        if isinstance(node, torch.fx.Node) and node not in seen and node in order:
            unique.append(node)
            seen.add(node)
    for node in sorted(unique, key=lambda n: order[n], reverse=True):
        _erase_if_unused(gm, node)


def rewrite_temporal_conv_bn_lif_state_to_fused(
    gm: torch.fx.GraphModule,
    temporal_windows: List[TemporalWindow],
    placeholder_values,
    max_patterns: int,
) -> TemporalRewriteStats:
    stats = TemporalRewriteStats(
        temporal_groups=len({window.layer_id for window in temporal_windows}),
        temporal_windows=len(temporal_windows),
    )
    replaced_patterns = 0
    for window in temporal_windows:
        patterns = window.patterns
        if replaced_patterns + len(patterns) > max_patterns:
            reason = "max-patterns limit reached"
            stats.temporal_skipped_windows += 1
            stats.log.append(f"SKIP layer={window.layer_id} window={window.window_id}: {reason}")
            print(f"[SKIP][TEMPORAL] layer={window.layer_id}, window={window.window_id}: {reason}")
            continue
        try:
            if not _same_lif_params(patterns):
                raise ValueError("lif params differ inside temporal window")
            ok, reason = check_temporal_state_chain(patterns)
            if not ok:
                raise ValueError(f"state chain not continuous: {reason}")
            ok, reason = _middle_v_next_has_no_external_uses(window)
            if not ok:
                raise ValueError(reason)

            first = patterns[0]
            conv_input, conv_weight, conv_bias, stride, padding, dilation, groups = extract_conv2d_tensors(
                gm, first.conv_node, placeholder_values
            )
            running_mean, running_var, bn_weight, bn_bias, training, eps = extract_batch_norm_params(
                gm, first.bn_node, placeholder_values
            )
            if training is not False:
                raise ValueError("batch_norm training is not False")
            folded_weight, folded_bias = fold_bn_into_conv_params(
                conv_weight,
                conv_bias,
                running_mean,
                running_var,
                bn_weight,
                bn_bias,
                eps,
            )

            v_init = first.lif_node.args[1]
            if isinstance(v_init, torch.fx.Node) and _is_zeros_like_of(v_init, first.bn_node):
                v_init = _materialize_scalar_zero_v_init(gm, first.conv_node, folded_weight)

            weight_attr = add_tensor_attr(gm, "_fx_temporal_folded_conv_bn_weight", folded_weight)
            bias_attr = add_tensor_attr(gm, "_fx_temporal_folded_conv_bn_bias", folded_bias)
            weight_node = _insert_get_attr_before(gm, first.conv_node, weight_attr)
            bias_node = _insert_get_attr_before(gm, first.conv_node, bias_attr)
            xs = [pattern.conv_input for pattern in patterns]
            ok, reason = _all_inputs_available_before(gm, xs + [v_init, weight_node, bias_node], first.conv_node)
            if not ok:
                raise ValueError("scheduled inputs are not available before first conv; " + reason)
            v_threshold, v_reset, tau, detach_reset = first.lif_params

            with gm.graph.inserting_before(first.conv_node):
                temporal_tuple = gm.graph.call_function(
                    torch.ops.snn_custom.fused_temporal_conv_lif_state.default,
                    args=(
                        xs,
                        weight_node,
                        bias_node,
                        v_init,
                        stride,
                        padding,
                        dilation,
                        groups,
                        v_threshold,
                        v_reset,
                        tau,
                        detach_reset,
                    ),
                )
                temporal_tuple.name = f"{first.conv_node.name}_temporal_fused_conv_lif_state"
                spike_stack = gm.graph.call_function(operator.getitem, args=(temporal_tuple, 0))
                spike_stack.name = f"{temporal_tuple.name}_spike_stack"
                v_final = gm.graph.call_function(operator.getitem, args=(temporal_tuple, 1))
                v_final.name = f"{temporal_tuple.name}_v_final"

            for index, pattern in enumerate(patterns):
                with gm.graph.inserting_before(pattern.spike_getitem):
                    spike_k = gm.graph.call_function(operator.getitem, args=(spike_stack, index))
                    spike_k.name = f"{temporal_tuple.name}_spike_t{index}"
                pattern.spike_getitem.replace_all_uses_with(spike_k)

            patterns[-1].v_getitem.replace_all_uses_with(v_final)
            _cleanup_window_nodes(gm, window)

            stats.temporal_replaced_windows += 1
            stats.temporal_replaced_patterns += len(patterns)
            replaced_patterns += len(patterns)
            message = (
                f"[REWRITE][TEMPORAL] layer={window.layer_id}, window={window.window_id}, "
                f"size={len(patterns)}, first={patterns[0].lif_node.name}, last={patterns[-1].lif_node.name}"
            )
            stats.log.append(message)
            print(message)
        except Exception as exc:
            stats.temporal_skipped_windows += 1
            message = f"SKIP layer={window.layer_id} window={window.window_id}: {exc}"
            stats.log.append(message)
            print(f"[SKIP][TEMPORAL] {message}")
            if not isinstance(exc, ValueError):
                traceback.print_exc()

    gm.graph.lint()
    gm.recompile()
    return stats


def dump_temporal_patterns(groups: List[TemporalGroup], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for group in groups:
        lines.append(f"layer_id: {group.layer_id}")
        lines.append(f"  count={len(group.patterns)}")
        for pattern in group.patterns:
            lines.append(
                "  "
                f"pattern_{pattern.timestep_index}: conv={pattern.conv_node.name}, bn={pattern.bn_node.name}, "
                f"lif={pattern.lif_node.name}, spike={pattern.spike_getitem.name}, v={pattern.v_getitem.name}, "
                f"v_prev={getattr(pattern.v_prev_node, 'name', pattern.v_prev_node)}"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def dump_temporal_windows(windows: List[TemporalWindow], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for index, window in enumerate(windows):
        lines.append(f"window_{index}:")
        lines.append(f"  layer_id={window.layer_id}")
        lines.append(f"  window_id={window.window_id}")
        lines.append(f"  size={len(window.patterns)}")
        lines.append(f"  patterns={[pattern.lif_node.name for pattern in window.patterns]}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def dump_temporal_rewrite_log(log: List[str], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(log) + ("\n" if log else ""), encoding="utf-8")


def count_fused_temporal_conv_lif_state_nodes(gm: torch.fx.GraphModule) -> int:
    return sum(
        1
        for node in gm.graph.nodes
        if node.op == "call_function" and str(node.target) == "snn_custom.fused_temporal_conv_lif_state.default"
    )
