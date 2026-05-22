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
class TemporalResidualPattern:
    layer_id: str
    timestep_index: int
    conv_node: torch.fx.Node
    bn_node: torch.fx.Node
    add_node: torch.fx.Node
    residual_node: torch.fx.Node
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
class TemporalResidualGroup:
    layer_id: str
    patterns: List[TemporalResidualPattern]


@dataclass
class TemporalResidualWindow:
    layer_id: str
    window_id: int
    patterns: List[TemporalResidualPattern]


@dataclass
class TemporalLifPattern:
    layer_id: str
    timestep_index: int
    window_id: int
    lif_node: torch.fx.Node
    input_node: torch.fx.Node
    v_prev_node: torch.fx.Node
    spike_getitem: torch.fx.Node
    v_getitem: torch.fx.Node
    v_next_node: torch.fx.Node
    lif_params: Tuple[Any, Any, Any, Any]
    occurrence: int
    shape_key: str


@dataclass
class TemporalLifGroup:
    layer_id: str
    patterns: List[TemporalLifPattern]


@dataclass
class TemporalLifWindow:
    layer_id: str
    window_id: int
    patterns: List[TemporalLifPattern]


@dataclass
class TemporalLifAvgPoolLinearPattern:
    layer_id: str
    timestep_index: int
    window_id: int
    lif_node: torch.fx.Node
    input_node: torch.fx.Node
    v_prev_node: torch.fx.Node
    spike_getitem: torch.fx.Node
    v_getitem: torch.fx.Node
    v_next_node: torch.fx.Node
    pool_node: torch.fx.Node
    flatten_node: torch.fx.Node
    linear_node: torch.fx.Node
    acc_node: torch.fx.Node
    acc_prev: Any
    fc_weight: Any
    fc_bias: Any
    lif_params: Tuple[Any, Any, Any, Any]
    occurrence: int
    shape_key: str


@dataclass
class TemporalLifAvgPoolLinearGroup:
    layer_id: str
    patterns: List[TemporalLifAvgPoolLinearPattern]


@dataclass
class TemporalLifAvgPoolLinearWindow:
    layer_id: str
    window_id: int
    patterns: List[TemporalLifAvgPoolLinearPattern]


@dataclass
class TemporalRewriteStats:
    temporal_groups: int = 0
    temporal_windows: int = 0
    temporal_replaced_windows: int = 0
    temporal_replaced_patterns: int = 0
    temporal_skipped_windows: int = 0
    single_step_replaced_patterns: int = 0
    log: List[str] = field(default_factory=list)


@dataclass
class TemporalResidualRewriteStats:
    temporal_residual_groups: int = 0
    temporal_residual_windows: int = 0
    temporal_residual_total_windows: int = 0
    temporal_residual_replaced_windows: int = 0
    temporal_residual_rewritten_windows: int = 0
    temporal_residual_replaced_patterns: int = 0
    temporal_residual_skipped_windows: int = 0
    temporal_residual_remapped_spike_external_users: int = 0
    temporal_residual_unremappable_external_users: int = 0
    residual_fuse_skip_reasons: Dict[str, int] = field(default_factory=dict)
    log: List[str] = field(default_factory=list)

    def skip(self, window: TemporalResidualWindow, reason: str):
        self.temporal_residual_skipped_windows += 1
        self.residual_fuse_skip_reasons[reason] = self.residual_fuse_skip_reasons.get(reason, 0) + 1
        message = f"SKIP layer={window.layer_id} window={window.window_id}: {reason}"
        self.log.append(message)
        print(f"[SKIP][TEMPORAL_RESADD] {message}")


@dataclass
class TemporalLifRewriteStats:
    temporal_lif_groups: int = 0
    temporal_lif_windows: int = 0
    temporal_lif_total_windows: int = 0
    temporal_lif_rewritten_windows: int = 0
    temporal_lif_replaced_patterns: int = 0
    temporal_lif_skipped_windows: int = 0
    temporal_lif_remapped_spike_external_users: int = 0
    temporal_lif_unremappable_external_users: int = 0
    temporal_lif_skip_reasons: Dict[str, int] = field(default_factory=dict)
    log: List[str] = field(default_factory=list)

    def skip(self, window: TemporalLifWindow, reason: str):
        self.temporal_lif_skipped_windows += 1
        self.temporal_lif_skip_reasons[reason] = self.temporal_lif_skip_reasons.get(reason, 0) + 1
        message = f"SKIP layer={window.layer_id} window={window.window_id}: {reason}"
        self.log.append(message)
        print(f"[SKIP][TEMPORAL_LIF] {message}")


@dataclass
class TemporalLifAvgPoolLinearRewriteStats:
    temporal_lif_avgpool_linear_groups: int = 0
    temporal_lif_avgpool_linear_windows: int = 0
    temporal_lif_avgpool_linear_total_windows: int = 0
    temporal_lif_avgpool_linear_rewritten_windows: int = 0
    temporal_lif_avgpool_linear_replaced_patterns: int = 0
    temporal_lif_avgpool_linear_skipped_windows: int = 0
    temporal_lif_avgpool_linear_skip_reasons: Dict[str, int] = field(default_factory=dict)
    log: List[str] = field(default_factory=list)

    def skip(self, window: TemporalLifAvgPoolLinearWindow, reason: str):
        self.temporal_lif_avgpool_linear_skipped_windows += 1
        self.temporal_lif_avgpool_linear_skip_reasons[reason] = self.temporal_lif_avgpool_linear_skip_reasons.get(reason, 0) + 1
        message = f"SKIP layer={window.layer_id} window={window.window_id}: {reason}"
        self.log.append(message)
        print(f"[SKIP][TEMPORAL_LIF_AVGPOOL_LINEAR] {message}")


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


def _is_add_node(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and node.target in (operator.add, operator.iadd, torch.add)


def _find_bn_residual_add_user(bn_node: torch.fx.Node):
    add_users = [user for user in bn_node.users if _is_add_node(user)]
    if len(add_users) != 1:
        return None, None
    add_node = add_users[0]
    args = list(add_node.args)
    if len(args) < 2:
        return None, None
    if args[0] is bn_node and isinstance(args[1], torch.fx.Node):
        return add_node, args[1]
    if args[1] is bn_node and isinstance(args[0], torch.fx.Node):
        return add_node, args[0]
    return None, None


def _producer_users_are_add_lif_zero(producer: torch.fx.Node, add_node: torch.fx.Node, lif_node: torch.fx.Node) -> Tuple[bool, str]:
    for user in producer.users:
        if user is add_node:
            continue
        return False, f"producer has unsupported user {user.name}"
    for user in add_node.users:
        if user is lif_node:
            continue
        if _is_zeros_like_of(user, add_node) and len(lif_node.args) > 1 and lif_node.args[1] is user:
            continue
        return False, f"add has unsupported user {user.name}"
    return True, ""


def collect_conv_bn_add_lif_state_patterns(gm: torch.fx.GraphModule) -> List[TemporalResidualPattern]:
    raw: List[Tuple[str, torch.fx.Node, torch.fx.Node, torch.fx.Node, torch.fx.Node, torch.fx.Node]] = []
    for node in gm.graph.nodes:
        if not is_conv_node(gm, node):
            continue
        conv_users = list(node.users)
        if len(conv_users) != 1:
            continue
        bn_node = conv_users[0]
        if not _is_batch_norm_node(gm, bn_node):
            continue
        add_node, residual_node = _find_bn_residual_add_user(bn_node)
        if add_node is None or residual_node is None:
            continue
        lif_candidates = [user for user in add_node.users if is_custom_lif_state_node(user)]
        if len(lif_candidates) != 1:
            continue
        lif_node = lif_candidates[0]
        ok, reason = _lif_state_is_usable(lif_node)
        if not ok:
            print(f"[SKIP][TEMPORAL_RESADD] lif={lif_node.name}: {reason}")
            continue
        ok, reason = _producer_users_are_add_lif_zero(bn_node, add_node, lif_node)
        if not ok:
            print(f"[SKIP][TEMPORAL_RESADD] bn={bn_node.name}, add={add_node.name}: {reason}")
            continue
        layer_id = extract_layer_id(gm, node, bn_node)
        if layer_id is None:
            print(f"[SKIP][TEMPORAL_RESADD] conv={node.name}: cannot extract layer_id")
            continue
        raw.append((f"resadd|{layer_id}", node, bn_node, add_node, residual_node, lif_node))

    # A ResNet downsample block can present the same add->lif through both the
    # main conv2/bn2 branch and the downsample conv/bn branch.  This pass only
    # fuses the main branch and treats the other input as residual, so keep one
    # producer per add/lif and prefer non-downsample parameter sources.
    deduped: List[Tuple[str, torch.fx.Node, torch.fx.Node, torch.fx.Node, torch.fx.Node, torch.fx.Node]] = []
    by_add_lif: Dict[Tuple[torch.fx.Node, torch.fx.Node], List[Tuple[str, torch.fx.Node, torch.fx.Node, torch.fx.Node, torch.fx.Node, torch.fx.Node]]] = {}
    for item in raw:
        _layer_id, _conv_node, _bn_node, add_node, _residual_node, lif_node = item
        by_add_lif.setdefault((add_node, lif_node), []).append(item)
    for (_add_node, _lif_node), items in by_add_lif.items():
        def branch_priority(item):
            _layer_id, conv_node, _bn_node, _add_node, _residual_node, _lif_node = item
            _conv_input, conv_weight_key = _extract_conv_graph_args(conv_node)
            return (1 if "downsample" in str(conv_weight_key) else 0, str(conv_weight_key))

        chosen = sorted(items, key=branch_priority)[0]
        if len(items) > 1:
            skipped = [item[1].name for item in items if item is not chosen]
            print(
                f"[SKIP][TEMPORAL_RESADD] add={_add_node.name}: duplicate producers {skipped}; "
                f"using conv={chosen[1].name}"
            )
        deduped.append(chosen)
    raw = deduped

    counts: Dict[str, int] = {}
    patterns: List[TemporalResidualPattern] = []
    for layer_id, conv_node, bn_node, add_node, residual_node, lif_node in raw:
        getitems = find_tuple_getitems(lif_node)
        conv_input, conv_weight_key = _extract_conv_graph_args(conv_node)
        bn_key = _extract_bn_key(bn_node)
        timestep_index = counts.get(layer_id, 0)
        counts[layer_id] = timestep_index + 1
        patterns.append(
            TemporalResidualPattern(
                layer_id=layer_id,
                timestep_index=timestep_index,
                conv_node=conv_node,
                bn_node=bn_node,
                add_node=add_node,
                residual_node=residual_node,
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


def _chronos_meta(node: torch.fx.Node, key: str, default=None):
    if key in node.meta:
        return node.meta[key]
    return getattr(node, f"_chronos_{key}", default)


def _shape_key_from_node(node: torch.fx.Node) -> str:
    meta = node.meta.get("tensor_meta") or node.meta.get("val")
    shape = getattr(meta, "shape", None)
    dtype = getattr(meta, "dtype", None)
    if shape is not None:
        return f"shape={tuple(shape)}|dtype={dtype}"
    if isinstance(meta, torch.Tensor):
        return f"shape={tuple(meta.shape)}|dtype={meta.dtype}"
    return "shape=<unknown>"


def _is_linear_output_node(node: torch.fx.Node) -> bool:
    if node.op != "call_function":
        return False
    target_text = str(node.target)
    return node.target is F.linear or "linear" in target_text


def collect_standalone_lif_state_patterns(
    gm: torch.fx.GraphModule,
    excluded_lif_nodes=None,
) -> List[TemporalLifPattern]:
    excluded = set(excluded_lif_nodes or [])
    fallback_counts: Dict[str, int] = {}
    patterns: List[TemporalLifPattern] = []
    for node in gm.graph.nodes:
        if node in excluded or not is_custom_lif_state_node(node):
            continue
        ok, reason = _lif_state_is_usable(node)
        if not ok:
            print(f"[SKIP][TEMPORAL_LIF] lif={node.name}: {reason}")
            continue
        if len(node.args) < 6 or not isinstance(node.args[0], torch.fx.Node):
            print(f"[SKIP][TEMPORAL_LIF] lif={node.name}: unsupported lif args")
            continue
        if str(node.target) in (
            "snn_custom.fused_temporal_conv_lif_state.default",
            "snn_custom.fused_temporal_conv_add_lif_state.default",
            "snn_custom.fused_temporal_lif_state.default",
        ):
            continue
        if _is_linear_output_node(node.args[0]):
            print(
                f"[SKIP][TEMPORAL_LIF] lif={node.name}: "
                "linear-output LIF is rank-2 and fused_temporal_lif_state currently requires [T,N,C,H,W]"
            )
            continue

        timestep = _chronos_meta(node, "timestep", None)
        window_id = _chronos_meta(node, "window_id", None)
        occurrence = _chronos_meta(node, "occurrence", None)
        if not isinstance(timestep, int):
            fallback_key = "standalone_lif_fallback"
            timestep = fallback_counts.get(fallback_key, 0)
            fallback_counts[fallback_key] = timestep + 1
        if not isinstance(window_id, int):
            window_id = 0
        if not isinstance(occurrence, int):
            occurrence = 0

        getitems = find_tuple_getitems(node)
        input_node = node.args[0]
        shape_key = _shape_key_from_node(input_node)
        lif_params = tuple(node.args[2:6])
        layer_id = f"standalone_lif|occurrence={occurrence}|{shape_key}|params={repr(lif_params)}"
        patterns.append(
            TemporalLifPattern(
                layer_id=layer_id,
                timestep_index=int(timestep),
                window_id=int(window_id),
                lif_node=node,
                input_node=input_node,
                v_prev_node=node.args[1],
                spike_getitem=getitems[0],
                v_getitem=getitems[1],
                v_next_node=getitems[1],
                lif_params=lif_params,
                occurrence=int(occurrence),
                shape_key=shape_key,
            )
        )
    return patterns


def _is_adaptive_avg_pool_1x1(node: torch.fx.Node) -> bool:
    if node.op != "call_function" or node.target is not F.adaptive_avg_pool2d:
        return False
    output_size = node.args[1] if len(node.args) > 1 else node.kwargs.get("output_size")
    return output_size in ((1, 1), [1, 1], 1)


def _is_flatten_batch_preserving(node: torch.fx.Node) -> bool:
    if node.op == "call_function" and node.target is torch.flatten:
        start_dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("start_dim", 0)
        end_dim = node.args[2] if len(node.args) > 2 else node.kwargs.get("end_dim", -1)
        return int(start_dim) == 1 and int(end_dim) == -1
    if node.op == "call_method" and node.target == "flatten":
        start_dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("start_dim", 0)
        end_dim = node.args[2] if len(node.args) > 2 else node.kwargs.get("end_dim", -1)
        return int(start_dim) == 1 and int(end_dim) == -1
    return False


def _is_linear_node(node: torch.fx.Node) -> bool:
    if node.op != "call_function":
        return False
    return node.target in (torch._C._nn.linear, F.linear)


def _single_user_node(node: torch.fx.Node) -> Optional[torch.fx.Node]:
    users = list(node.users)
    return users[0] if len(users) == 1 else None


def _extract_linear_weight_bias(linear_node: torch.fx.Node) -> Tuple[Any, Any]:
    weight = linear_node.args[1] if len(linear_node.args) > 1 else linear_node.kwargs.get("weight")
    bias = linear_node.args[2] if len(linear_node.args) > 2 else linear_node.kwargs.get("bias", None)
    return weight, bias


def _find_accumulator_add_user(linear_node: torch.fx.Node) -> Tuple[Optional[torch.fx.Node], Any]:
    add_users = [user for user in linear_node.users if _is_add_node(user)]
    if len(add_users) != 1:
        return None, None
    add_node = add_users[0]
    args = list(add_node.args)
    if len(args) < 2:
        return None, None
    if args[0] is linear_node:
        return add_node, args[1]
    if args[1] is linear_node:
        return add_node, args[0]
    return None, None


def collect_temporal_lif_avgpool_linear_patterns(
    gm: torch.fx.GraphModule,
    excluded_lif_nodes=None,
) -> List[TemporalLifAvgPoolLinearPattern]:
    excluded = set(excluded_lif_nodes or [])
    fallback_counts: Dict[str, int] = {}
    patterns: List[TemporalLifAvgPoolLinearPattern] = []
    for node in gm.graph.nodes:
        if node in excluded or not is_custom_lif_state_node(node):
            continue
        ok, reason = _lif_state_is_usable(node)
        if not ok:
            print(f"[SKIP][TEMPORAL_LIF_AVGPOOL_LINEAR] lif={node.name}: {reason}")
            continue
        if len(node.args) < 6 or not isinstance(node.args[0], torch.fx.Node):
            print(f"[SKIP][TEMPORAL_LIF_AVGPOOL_LINEAR] lif={node.name}: unsupported lif args")
            continue

        getitems = find_tuple_getitems(node)
        spike = getitems[0]
        pool = _single_user_node(spike)
        if pool is None or not _is_adaptive_avg_pool_1x1(pool):
            continue
        flatten = _single_user_node(pool)
        if flatten is None or not _is_flatten_batch_preserving(flatten):
            continue
        linear = _single_user_node(flatten)
        if linear is None or not _is_linear_node(linear):
            continue
        acc_node, acc_prev = _find_accumulator_add_user(linear)
        if acc_node is None:
            continue

        timestep = _chronos_meta(node, "timestep", None)
        window_id = _chronos_meta(node, "window_id", None)
        occurrence = _chronos_meta(node, "occurrence", None)
        if not isinstance(timestep, int):
            fallback_key = "temporal_lif_avgpool_linear_fallback"
            timestep = fallback_counts.get(fallback_key, 0)
            fallback_counts[fallback_key] = timestep + 1
        if not isinstance(window_id, int):
            window_id = 0
        if not isinstance(occurrence, int):
            occurrence = 0

        weight, bias = _extract_linear_weight_bias(linear)
        input_node = node.args[0]
        shape_key = _shape_key_from_node(input_node)
        lif_params = tuple(node.args[2:6])
        layer_id = (
            f"lif_avgpool_linear|occurrence={occurrence}|{shape_key}|fc={_node_key(weight)}|"
            f"bias={_node_key(bias)}|params={repr(lif_params)}"
        )
        patterns.append(
            TemporalLifAvgPoolLinearPattern(
                layer_id=layer_id,
                timestep_index=int(timestep),
                window_id=int(window_id),
                lif_node=node,
                input_node=input_node,
                v_prev_node=node.args[1],
                spike_getitem=spike,
                v_getitem=getitems[1],
                v_next_node=getitems[1],
                pool_node=pool,
                flatten_node=flatten,
                linear_node=linear,
                acc_node=acc_node,
                acc_prev=acc_prev,
                fc_weight=weight,
                fc_bias=bias,
                lif_params=lif_params,
                occurrence=int(occurrence),
                shape_key=shape_key,
            )
        )
    return patterns


def group_temporal_patterns(patterns: List[TemporalPattern]) -> List[TemporalGroup]:
    grouped: Dict[str, List[TemporalPattern]] = {}
    for pattern in patterns:
        grouped.setdefault(pattern.layer_id, []).append(pattern)
    return [TemporalGroup(layer_id=layer_id, patterns=items) for layer_id, items in grouped.items()]


def group_temporal_residual_patterns(patterns: List[TemporalResidualPattern]) -> List[TemporalResidualGroup]:
    grouped: Dict[str, List[TemporalResidualPattern]] = {}
    for pattern in patterns:
        grouped.setdefault(pattern.layer_id, []).append(pattern)
    return [TemporalResidualGroup(layer_id=layer_id, patterns=items) for layer_id, items in grouped.items()]


def group_temporal_lif_patterns(patterns: List[TemporalLifPattern]) -> List[TemporalLifGroup]:
    grouped: Dict[str, List[TemporalLifPattern]] = {}
    for pattern in patterns:
        grouped.setdefault(pattern.layer_id, []).append(pattern)
    groups = []
    for layer_id, items in grouped.items():
        groups.append(TemporalLifGroup(layer_id=layer_id, patterns=sorted(items, key=lambda p: p.timestep_index)))
    return groups


def group_temporal_lif_avgpool_linear_patterns(patterns: List[TemporalLifAvgPoolLinearPattern]) -> List[TemporalLifAvgPoolLinearGroup]:
    grouped: Dict[str, List[TemporalLifAvgPoolLinearPattern]] = {}
    for pattern in patterns:
        grouped.setdefault(pattern.layer_id, []).append(pattern)
    groups = []
    for layer_id, items in grouped.items():
        groups.append(TemporalLifAvgPoolLinearGroup(layer_id=layer_id, patterns=sorted(items, key=lambda p: p.timestep_index)))
    return groups


def check_temporal_state_chain(patterns: List[TemporalPattern]) -> Tuple[bool, str]:
    for prev, nxt in zip(patterns, patterns[1:]):
        if nxt.v_prev_node is prev.v_getitem:
            continue
        return False, f"{prev.v_getitem.name} does not feed {nxt.lif_node.name} v_prev"
    return True, ""


def check_temporal_residual_state_chain(patterns: List[TemporalResidualPattern]) -> Tuple[bool, str]:
    for prev, nxt in zip(patterns, patterns[1:]):
        if nxt.v_prev_node is prev.v_getitem:
            continue
        return False, f"{prev.v_getitem.name} does not feed {nxt.lif_node.name} v_prev"
    return True, ""


def check_temporal_lif_state_chain(patterns: List[TemporalLifPattern]) -> Tuple[bool, str]:
    for prev, nxt in zip(patterns, patterns[1:]):
        if nxt.v_prev_node is prev.v_getitem:
            continue
        return False, f"{prev.v_getitem.name} does not feed {nxt.lif_node.name} v_prev"
    return True, ""


def check_temporal_lif_avgpool_linear_state_and_acc_chain(patterns: List[TemporalLifAvgPoolLinearPattern]) -> Tuple[bool, str]:
    for prev, nxt in zip(patterns, patterns[1:]):
        if nxt.v_prev_node is not prev.v_getitem:
            return False, f"{prev.v_getitem.name} does not feed {nxt.lif_node.name} v_prev"
        if nxt.acc_prev is not prev.acc_node:
            return False, f"{prev.acc_node.name} does not feed {nxt.acc_node.name} accumulator"
    return True, ""


def make_temporal_windows(groups: List[TemporalGroup], window_size: int, allow_tail: bool) -> List[TemporalWindow]:
    if window_size < 1:
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


def make_temporal_residual_windows(
    groups: List[TemporalResidualGroup],
    window_size: int,
    allow_tail: bool,
) -> List[TemporalResidualWindow]:
    if window_size < 1:
        return []
    windows: List[TemporalResidualWindow] = []
    for group in groups:
        ok, reason = check_temporal_residual_state_chain(group.patterns)
        if not ok:
            print(f"[SKIP][TEMPORAL_RESADD] layer={group.layer_id}: state chain not continuous: {reason}")
            continue
        window_id = 0
        for start in range(0, len(group.patterns), window_size):
            chunk = group.patterns[start : start + window_size]
            if len(chunk) < window_size and not allow_tail:
                print(f"[SKIP][TEMPORAL_RESADD] layer={group.layer_id}: tail size={len(chunk)} < window={window_size}")
                continue
            if len(chunk) <= 1:
                continue
            windows.append(TemporalResidualWindow(layer_id=group.layer_id, window_id=window_id, patterns=chunk))
            window_id += 1
    return windows


def make_temporal_lif_windows(
    groups: List[TemporalLifGroup],
    window_size: int,
    allow_tail: bool,
) -> List[TemporalLifWindow]:
    if window_size < 1:
        return []
    windows: List[TemporalLifWindow] = []
    for group in groups:
        by_window: Dict[int, List[TemporalLifPattern]] = {}
        for pattern in group.patterns:
            by_window.setdefault(pattern.window_id, []).append(pattern)
        for window_id, items in sorted(by_window.items()):
            items = sorted(items, key=lambda pattern: pattern.timestep_index)
            if len(items) < window_size and not allow_tail:
                print(f"[SKIP][TEMPORAL_LIF] layer={group.layer_id}: tail size={len(items)} < window={window_size}")
                continue
            expected = list(range(items[0].timestep_index, items[0].timestep_index + len(items)))
            actual = [pattern.timestep_index for pattern in items]
            if actual != expected:
                print(f"[SKIP][TEMPORAL_LIF] layer={group.layer_id}: timesteps not continuous: {actual}")
                continue
            ok, reason = check_temporal_lif_state_chain(items)
            if not ok:
                print(f"[SKIP][TEMPORAL_LIF] layer={group.layer_id}: state chain not continuous: {reason}")
                continue
            windows.append(TemporalLifWindow(layer_id=group.layer_id, window_id=window_id, patterns=items))
    return windows


def make_temporal_lif_avgpool_linear_windows(
    groups: List[TemporalLifAvgPoolLinearGroup],
    window_size: int,
    allow_tail: bool,
) -> List[TemporalLifAvgPoolLinearWindow]:
    if window_size < 1:
        return []
    windows: List[TemporalLifAvgPoolLinearWindow] = []
    for group in groups:
        by_window: Dict[int, List[TemporalLifAvgPoolLinearPattern]] = {}
        for pattern in group.patterns:
            by_window.setdefault(pattern.window_id, []).append(pattern)
        for window_id, items in sorted(by_window.items()):
            items = sorted(items, key=lambda pattern: pattern.timestep_index)
            if len(items) < window_size and not allow_tail:
                print(f"[SKIP][TEMPORAL_LIF_AVGPOOL_LINEAR] layer={group.layer_id}: tail size={len(items)} < window={window_size}")
                continue
            if len(items) <= 1:
                continue
            expected = list(range(items[0].timestep_index, items[0].timestep_index + len(items)))
            actual = [pattern.timestep_index for pattern in items]
            if actual != expected:
                print(f"[SKIP][TEMPORAL_LIF_AVGPOOL_LINEAR] layer={group.layer_id}: timesteps not continuous: {actual}")
                continue
            ok, reason = check_temporal_lif_avgpool_linear_state_and_acc_chain(items)
            if not ok:
                print(f"[SKIP][TEMPORAL_LIF_AVGPOOL_LINEAR] layer={group.layer_id}: chain not continuous: {reason}")
                continue
            windows.append(TemporalLifAvgPoolLinearWindow(layer_id=group.layer_id, window_id=window_id, patterns=items))
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


def _lif_middle_v_next_has_no_external_uses(window: TemporalLifWindow) -> Tuple[bool, str]:
    patterns = window.patterns
    for idx, pattern in enumerate(patterns[:-1]):
        allowed = patterns[idx + 1].lif_node
        external = [user.name for user in pattern.v_getitem.users if user is not allowed]
        if external:
            return False, f"middle v_next {pattern.v_getitem.name} has external users {external}"
    return True, ""


def _same_standalone_lif_params(patterns: List[TemporalLifPattern]) -> bool:
    first = patterns[0].lif_params
    return all(pattern.lif_params == first for pattern in patterns)


def _same_standalone_lif_shapes(patterns: List[TemporalLifPattern]) -> bool:
    first = patterns[0].shape_key
    return all(pattern.shape_key == first for pattern in patterns)


def _same_lif_avgpool_linear_params(patterns: List[TemporalLifAvgPoolLinearPattern]) -> bool:
    first = patterns[0]
    return all(
        pattern.lif_params == first.lif_params
        and _node_key(pattern.fc_weight) == _node_key(first.fc_weight)
        and _node_key(pattern.fc_bias) == _node_key(first.fc_bias)
        for pattern in patterns
    )


def _same_lif_avgpool_linear_shapes(patterns: List[TemporalLifAvgPoolLinearPattern]) -> bool:
    first = patterns[0].shape_key
    return all(pattern.shape_key == first for pattern in patterns)


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


def _all_inputs_available_for_node(gm: torch.fx.GraphModule, inputs: List[Any], node: torch.fx.Node) -> Tuple[bool, str]:
    order = {fx_node: index for index, fx_node in enumerate(gm.graph.nodes)}
    node_order = order[node]
    for value in inputs:
        if isinstance(value, torch.fx.Node) and order.get(value, node_order + 1) >= node_order:
            return False, f"input {value.name} is not defined before node {node.name}"
    return True, ""


def _iter_arg_nodes(value):
    if isinstance(value, torch.fx.Node):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_arg_nodes(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_arg_nodes(item)


def _node_reaches_any_input(source: torch.fx.Node, inputs: List[Any]) -> bool:
    stack = [node for value in inputs for node in _iter_arg_nodes(value)]
    seen = set()
    while stack:
        node = stack.pop()
        if node is source:
            return True
        if node in seen:
            continue
        seen.add(node)
        stack.extend(_iter_arg_nodes(node.args))
        stack.extend(_iter_arg_nodes(node.kwargs))
    return False


def _unique_nodes(nodes: List[torch.fx.Node]) -> List[torch.fx.Node]:
    out = []
    seen = set()
    for node in nodes:
        if node not in seen:
            out.append(node)
            seen.add(node)
    return out


def _resolved_replacement_node(node):
    seen = set()
    while isinstance(node, torch.fx.Node) and "chronos_replacement_node" in node.meta:
        if node in seen:
            break
        seen.add(node)
        replacement = node.meta.get("chronos_replacement_node")
        if not isinstance(replacement, torch.fx.Node):
            break
        node = replacement
    return node


def _external_spike_users_by_pattern(patterns, replaceable: set) -> Dict[torch.fx.Node, List[torch.fx.Node]]:
    out: Dict[torch.fx.Node, List[torch.fx.Node]] = {}
    for pattern in patterns:
        users = [user for user in pattern.spike_getitem.users if user not in replaceable]
        out[pattern.spike_getitem] = _unique_nodes(users)
    return out


def _unremappable_spike_external_user_reason(
    spike_external_users: Dict[torch.fx.Node, List[torch.fx.Node]],
    inputs: List[Any],
) -> str:
    for spike_node, users in spike_external_users.items():
        for user in users:
            if _node_reaches_any_input(user, inputs):
                return (
                    f"replacement would create cycle: external spike user {user.name} "
                    f"from {spike_node.name} produces a fused-op input"
                )
    return ""


def _move_early_remapped_users_after(
    gm: torch.fx.GraphModule,
    users: List[torch.fx.Node],
    anchor: torch.fx.Node,
):
    order = {node: index for index, node in enumerate(gm.graph.nodes)}
    anchor_order = order[anchor]
    to_move = set()
    stack = [user for user in users if order.get(user, anchor_order + 1) <= anchor_order]
    while stack:
        node = stack.pop()
        if node in to_move or node is anchor or node.op == "output":
            continue
        to_move.add(node)
        for user in node.users:
            if order.get(user, anchor_order + 1) <= anchor_order:
                stack.append(user)

    prev = anchor
    for node in sorted(to_move, key=lambda item: order[item]):
        prev.append(node)
        prev = node


def _replaceable_residual_window_nodes(window: TemporalResidualWindow) -> set:
    nodes = set()
    for pattern in window.patterns:
        nodes.update(
            [
                pattern.conv_node,
                pattern.bn_node,
                pattern.add_node,
                pattern.lif_node,
                pattern.spike_getitem,
                pattern.v_getitem,
            ]
        )
        v_prev = pattern.lif_node.args[1] if len(pattern.lif_node.args) > 1 else None
        if isinstance(v_prev, torch.fx.Node) and _is_zeros_like_of(v_prev, pattern.add_node):
            nodes.add(v_prev)
    return nodes


def _external_residual_window_users(window: TemporalResidualWindow) -> List[torch.fx.Node]:
    replaceable = _replaceable_residual_window_nodes(window)
    users: List[torch.fx.Node] = []
    for user in window.patterns[-1].v_getitem.users:
        if user not in replaceable:
            users.append(user)
    return users


def _select_residual_temporal_insert_anchor(
    gm: torch.fx.GraphModule,
    window: TemporalResidualWindow,
    inputs: List[Any],
) -> Tuple[Optional[torch.fx.Node], str, str]:
    """Choose a legal insertion point for residual temporal fusion.

    Residual-add windows often consume upstream temporal-fused spike getitems
    that are materialized after the first conv in this window.  In that case
    inserting before the first conv would violate FX topological order.  This
    helper moves the fused op just after the latest real input.  External
    v_next users remain strict; external spike users are handled later by
    remapping them to fused spike_stack[t] and moving early consumers.
    """
    order = {node: index for index, node in enumerate(gm.graph.nodes)}
    first = window.patterns[0].conv_node
    first_order = order[first]
    replaceable = _replaceable_residual_window_nodes(window)
    input_nodes = [value for value in inputs if isinstance(value, torch.fx.Node)]

    for node in input_nodes:
        if node in replaceable:
            return None, "skip", f"input {node.name} is produced by nodes being replaced"
    spike_external_users = _external_spike_users_by_pattern(window.patterns, replaceable)
    unremappable = _unremappable_spike_external_user_reason(spike_external_users, inputs)
    if unremappable:
        return None, "skip", unremappable

    late_inputs = [node for node in input_nodes if order.get(node, -1) >= first_order]
    if not late_inputs:
        return first, "before", ""

    anchor = max(late_inputs, key=lambda node: order[node])
    anchor_order = order[anchor]
    early_users = [
        user.name
        for user in _external_residual_window_users(window)
        if order.get(user, anchor_order + 1) <= anchor_order
    ]
    if early_users:
        return (
            None,
            "skip",
            f"external users {early_users} appear before latest input {anchor.name}",
        )
    return anchor, "after", ""


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


def _replaceable_lif_window_nodes(window: TemporalLifWindow) -> set:
    nodes = set()
    for pattern in window.patterns:
        nodes.update([pattern.lif_node, pattern.spike_getitem, pattern.v_getitem])
    return nodes


def _external_lif_window_users(window: TemporalLifWindow) -> List[torch.fx.Node]:
    replaceable = _replaceable_lif_window_nodes(window)
    users: List[torch.fx.Node] = []
    for user in window.patterns[-1].v_getitem.users:
        if user not in replaceable:
            users.append(user)
    return users


def _select_lif_temporal_insert_anchor(
    gm: torch.fx.GraphModule,
    window: TemporalLifWindow,
    inputs: List[Any],
) -> Tuple[Optional[torch.fx.Node], str, str]:
    order = {node: index for index, node in enumerate(gm.graph.nodes)}
    first = window.patterns[0].lif_node
    first_order = order[first]
    replaceable = _replaceable_lif_window_nodes(window)
    input_nodes = [value for value in inputs if isinstance(value, torch.fx.Node)]
    for node in input_nodes:
        if node in replaceable:
            return None, "skip", f"input {node.name} is produced by nodes being replaced"
    spike_external_users = _external_spike_users_by_pattern(window.patterns, replaceable)
    unremappable = _unremappable_spike_external_user_reason(spike_external_users, inputs)
    if unremappable:
        return None, "skip", unremappable
    late_inputs = [node for node in input_nodes if order.get(node, -1) >= first_order]
    if not late_inputs:
        return first, "before", ""
    anchor = max(late_inputs, key=lambda node: order[node])
    anchor_order = order[anchor]
    early_users = [
        user.name
        for user in _external_lif_window_users(window)
        if order.get(user, anchor_order + 1) <= anchor_order
    ]
    if early_users:
        return None, "skip", f"external users {early_users} appear before latest input {anchor.name}"
    return anchor, "after", ""


def _cleanup_lif_window_nodes(gm: torch.fx.GraphModule, window: TemporalLifWindow):
    order = {node: index for index, node in enumerate(gm.graph.nodes)}
    candidates = []
    for pattern in window.patterns:
        candidates.extend([pattern.spike_getitem, pattern.v_getitem, pattern.lif_node])
    unique = []
    seen = set()
    for node in candidates:
        if isinstance(node, torch.fx.Node) and node not in seen and node in order:
            unique.append(node)
            seen.add(node)
    for node in sorted(unique, key=lambda n: order[n], reverse=True):
        _erase_if_unused(gm, node)


def _replaceable_lif_avgpool_linear_window_nodes(window: TemporalLifAvgPoolLinearWindow) -> set:
    nodes = set()
    for pattern in window.patterns:
        nodes.update(
            [
                pattern.lif_node,
                pattern.spike_getitem,
                pattern.v_getitem,
                pattern.pool_node,
                pattern.flatten_node,
                pattern.linear_node,
                pattern.acc_node,
            ]
        )
    return nodes


def _external_lif_avgpool_linear_window_users(window: TemporalLifAvgPoolLinearWindow) -> List[torch.fx.Node]:
    replaceable = _replaceable_lif_avgpool_linear_window_nodes(window)
    users: List[torch.fx.Node] = []
    for pattern in window.patterns:
        for node in (pattern.spike_getitem, pattern.pool_node, pattern.flatten_node, pattern.linear_node):
            for user in node.users:
                if user not in replaceable:
                    users.append(user)
    for user in window.patterns[-1].v_getitem.users:
        if user not in replaceable:
            users.append(user)
    for user in window.patterns[-1].acc_node.users:
        if user not in replaceable:
            users.append(user)
    return users


def _lif_avgpool_linear_middle_nodes_have_no_external_uses(window: TemporalLifAvgPoolLinearWindow) -> Tuple[bool, str]:
    replaceable = _replaceable_lif_avgpool_linear_window_nodes(window)
    for pattern in window.patterns[:-1]:
        for user in pattern.v_getitem.users:
            if user not in replaceable:
                return False, f"middle v_next {pattern.v_getitem.name} has external user {user.name}"
        for user in pattern.acc_node.users:
            if user not in replaceable:
                return False, f"middle accumulator {pattern.acc_node.name} has external user {user.name}"
    return True, ""


def _select_lif_avgpool_linear_temporal_insert_anchor(
    gm: torch.fx.GraphModule,
    window: TemporalLifAvgPoolLinearWindow,
    inputs: List[Any],
) -> Tuple[Optional[torch.fx.Node], str, str]:
    order = {node: index for index, node in enumerate(gm.graph.nodes)}
    first = window.patterns[0].lif_node
    first_order = order[first]
    replaceable = _replaceable_lif_avgpool_linear_window_nodes(window)
    input_nodes = [value for value in inputs if isinstance(value, torch.fx.Node)]
    for node in input_nodes:
        if node in replaceable:
            return None, "skip", f"input {node.name} is produced by nodes being replaced"
    late_inputs = [node for node in input_nodes if order.get(node, -1) >= first_order]
    if not late_inputs:
        return first, "before", ""
    anchor = max(late_inputs, key=lambda node: order[node])
    anchor_order = order[anchor]
    early_users = [
        user.name
        for user in _external_lif_avgpool_linear_window_users(window)
        if order.get(user, anchor_order + 1) <= anchor_order
    ]
    if early_users:
        return None, "skip", f"external users {early_users} appear before latest input {anchor.name}"
    return anchor, "after", ""


def _cleanup_lif_avgpool_linear_window_nodes(gm: torch.fx.GraphModule, window: TemporalLifAvgPoolLinearWindow):
    order = {node: index for index, node in enumerate(gm.graph.nodes)}
    candidates = []
    for pattern in window.patterns:
        candidates.extend(
            [
                pattern.acc_node,
                pattern.linear_node,
                pattern.flatten_node,
                pattern.pool_node,
                pattern.spike_getitem,
                pattern.v_getitem,
                pattern.lif_node,
            ]
        )
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

            v_init = _resolved_replacement_node(first.lif_node.args[1])
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
                pattern.spike_getitem.meta["chronos_replacement_node"] = spike_k
                pattern.spike_getitem.replace_all_uses_with(spike_k)

            patterns[-1].v_getitem.meta["chronos_replacement_node"] = v_final
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


def _residual_middle_v_next_has_no_external_uses(window: TemporalResidualWindow) -> Tuple[bool, str]:
    patterns = window.patterns
    for idx, pattern in enumerate(patterns[:-1]):
        allowed = patterns[idx + 1].lif_node
        external = [user.name for user in pattern.v_getitem.users if user is not allowed]
        if external:
            return False, f"middle v_next {pattern.v_getitem.name} has external users {external}"
    return True, ""


def _residual_shapes_compatible(patterns: List[TemporalResidualPattern]) -> Tuple[bool, str]:
    for pattern in patterns:
        bn_meta = pattern.bn_node.meta.get("tensor_meta") or pattern.bn_node.meta.get("val")
        residual_meta = pattern.residual_node.meta.get("tensor_meta") or pattern.residual_node.meta.get("val")
        bn_shape = tuple(getattr(bn_meta, "shape", getattr(bn_meta, "shape", ()))) if bn_meta is not None else None
        residual_shape = (
            tuple(getattr(residual_meta, "shape", getattr(residual_meta, "shape", ())))
            if residual_meta is not None
            else None
        )
        if bn_shape and residual_shape and bn_shape != residual_shape:
            return False, f"shape mismatch bn={bn_shape} residual={residual_shape} at add={pattern.add_node.name}"
    return True, ""


def _cleanup_residual_window_nodes(gm: torch.fx.GraphModule, window: TemporalResidualWindow):
    order = {node: index for index, node in enumerate(gm.graph.nodes)}
    candidates = []
    for pattern in window.patterns:
        v_prev = pattern.lif_node.args[1] if len(pattern.lif_node.args) > 1 else None
        candidates.extend([pattern.spike_getitem, pattern.v_getitem, pattern.lif_node])
        if isinstance(v_prev, torch.fx.Node):
            candidates.append(v_prev)
        candidates.extend([pattern.add_node, pattern.bn_node, pattern.conv_node])
    unique = []
    seen = set()
    for node in candidates:
        if isinstance(node, torch.fx.Node) and node not in seen and node in order:
            unique.append(node)
            seen.add(node)
    for node in sorted(unique, key=lambda n: order[n], reverse=True):
        _erase_if_unused(gm, node)


def rewrite_temporal_conv_bn_add_lif_state_to_fused(
    gm: torch.fx.GraphModule,
    temporal_windows: List[TemporalResidualWindow],
    placeholder_values,
    max_patterns: int,
) -> TemporalResidualRewriteStats:
    stats = TemporalResidualRewriteStats(
        temporal_residual_groups=len({window.layer_id for window in temporal_windows}),
        temporal_residual_windows=len(temporal_windows),
        temporal_residual_total_windows=len(temporal_windows),
    )
    replaced_patterns = 0
    for window in temporal_windows:
        patterns = window.patterns
        if replaced_patterns + len(patterns) > max_patterns:
            stats.skip(window, "max_patterns")
            continue
        try:
            if not _same_lif_params(patterns):
                stats.skip(window, "lif params differ inside temporal residual window")
                continue
            ok, reason = check_temporal_residual_state_chain(patterns)
            if not ok:
                stats.skip(window, f"state chain not continuous: {reason}")
                continue
            ok, reason = _residual_middle_v_next_has_no_external_uses(window)
            if not ok:
                stats.skip(window, reason)
                continue
            ok, reason = _residual_shapes_compatible(patterns)
            if not ok:
                stats.skip(window, reason)
                continue
            spike_external_users = _external_spike_users_by_pattern(
                patterns,
                _replaceable_residual_window_nodes(window),
            )
            remappable_spike_users = _unique_nodes(
                [user for users in spike_external_users.values() for user in users]
            )

            first = patterns[0]
            conv_input, conv_weight, conv_bias, stride, padding, dilation, groups = extract_conv2d_tensors(
                gm, first.conv_node, placeholder_values
            )
            running_mean, running_var, bn_weight, bn_bias, training, eps = extract_batch_norm_params(
                gm, first.bn_node, placeholder_values
            )
            if training is not False:
                stats.skip(window, "batch_norm training is not False")
                continue
            folded_weight, folded_bias = fold_bn_into_conv_params(
                conv_weight,
                conv_bias,
                running_mean,
                running_var,
                bn_weight,
                bn_bias,
                eps,
            )

            v_init = _resolved_replacement_node(first.lif_node.args[1])
            if isinstance(v_init, torch.fx.Node) and _is_zeros_like_of(v_init, first.add_node):
                v_init = _materialize_scalar_zero_v_init(gm, first.conv_node, folded_weight)

            xs = [_resolved_replacement_node(pattern.conv_input) for pattern in patterns]
            residuals = [_resolved_replacement_node(pattern.residual_node) for pattern in patterns]
            anchor, insert_mode, reason = _select_residual_temporal_insert_anchor(
                gm,
                window,
                xs + residuals + [v_init],
            )
            if anchor is None:
                stats.skip(window, "cannot find legal temporal residual insertion point; " + reason)
                continue

            weight_attr = add_tensor_attr(gm, "_fx_temporal_resadd_folded_conv_bn_weight", folded_weight)
            bias_attr = add_tensor_attr(gm, "_fx_temporal_resadd_folded_conv_bn_bias", folded_bias)
            weight_node = _insert_get_attr_before(gm, first.conv_node, weight_attr)
            bias_node = _insert_get_attr_before(gm, first.conv_node, bias_attr)
            anchor, insert_mode, reason = _select_residual_temporal_insert_anchor(
                gm,
                window,
                xs + residuals + [v_init, weight_node, bias_node],
            )
            if anchor is None:
                if "replacement would create cycle" in reason:
                    stats.temporal_residual_unremappable_external_users += len(remappable_spike_users)
                stats.skip(window, "cannot find legal temporal residual insertion point; " + reason)
                continue
            v_threshold, v_reset, tau, detach_reset = first.lif_params

            insert_ctx = gm.graph.inserting_before(anchor) if insert_mode == "before" else gm.graph.inserting_after(anchor)
            with insert_ctx:
                temporal_tuple = gm.graph.call_function(
                    torch.ops.snn_custom.fused_temporal_conv_add_lif_state.default,
                    args=(
                        xs,
                        residuals,
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
                temporal_tuple.name = f"{first.conv_node.name}_temporal_fused_conv_bn_add_lif_state"

            ok, reason = _all_inputs_available_for_node(
                gm,
                xs + residuals + [weight_node, bias_node, v_init],
                temporal_tuple,
            )
            if not ok:
                gm.graph.erase_node(temporal_tuple)
                stats.skip(window, "cannot find legal temporal residual insertion point; " + reason)
                continue

            with gm.graph.inserting_after(temporal_tuple):
                spike_stack = gm.graph.call_function(operator.getitem, args=(temporal_tuple, 0))
                spike_stack.name = f"{temporal_tuple.name}_spike_stack"
            with gm.graph.inserting_after(spike_stack):
                v_final = gm.graph.call_function(operator.getitem, args=(temporal_tuple, 1))
                v_final.name = f"{temporal_tuple.name}_v_final"

            prev_insert = v_final
            for index, pattern in enumerate(patterns):
                with gm.graph.inserting_after(prev_insert):
                    spike_k = gm.graph.call_function(operator.getitem, args=(spike_stack, index))
                    spike_k.name = f"{temporal_tuple.name}_spike_t{index}"
                prev_insert = spike_k
                pattern.spike_getitem.meta["chronos_replacement_node"] = spike_k
                pattern.spike_getitem.replace_all_uses_with(spike_k)

            patterns[-1].v_getitem.meta["chronos_replacement_node"] = v_final
            patterns[-1].v_getitem.replace_all_uses_with(v_final)
            if remappable_spike_users:
                _move_early_remapped_users_after(gm, remappable_spike_users, prev_insert)
                stats.temporal_residual_remapped_spike_external_users += len(remappable_spike_users)
            _cleanup_residual_window_nodes(gm, window)

            stats.temporal_residual_replaced_windows += 1
            stats.temporal_residual_rewritten_windows += 1
            stats.temporal_residual_replaced_patterns += len(patterns)
            replaced_patterns += len(patterns)
            message = (
                f"[REWRITE][TEMPORAL_RESADD] layer={window.layer_id}, window={window.window_id}, "
                f"size={len(patterns)}, first={patterns[0].lif_node.name}, last={patterns[-1].lif_node.name}"
            )
            stats.log.append(message)
            print(message)
        except Exception as exc:
            reason = str(exc)
            stats.skip(window, reason)
            if not isinstance(exc, ValueError):
                traceback.print_exc()

    try:
        gm.graph.lint()
        gm.recompile()
    except Exception:
        print("[WARN][TEMPORAL_RESADD] graph lint/recompile failed after residual rewrite; preserving exception for caller")
        traceback.print_exc()
        raise
    return stats


def rewrite_temporal_lif_state_to_fused(
    gm: torch.fx.GraphModule,
    temporal_windows: List[TemporalLifWindow],
    max_patterns: int,
) -> TemporalLifRewriteStats:
    stats = TemporalLifRewriteStats(
        temporal_lif_groups=len({window.layer_id for window in temporal_windows}),
        temporal_lif_windows=len(temporal_windows),
        temporal_lif_total_windows=len(temporal_windows),
    )
    replaced_patterns = 0
    for window in temporal_windows:
        patterns = window.patterns
        if replaced_patterns + len(patterns) > max_patterns:
            stats.skip(window, "max_patterns")
            continue
        try:
            if not _same_standalone_lif_params(patterns):
                stats.skip(window, "lif params differ inside temporal lif window")
                continue
            if not _same_standalone_lif_shapes(patterns):
                stats.skip(window, "input shapes differ inside temporal lif window")
                continue
            ok, reason = check_temporal_lif_state_chain(patterns)
            if not ok:
                stats.skip(window, f"state chain not continuous: {reason}")
                continue
            ok, reason = _lif_middle_v_next_has_no_external_uses(window)
            if not ok:
                stats.skip(window, reason)
                continue
            spike_external_users = _external_spike_users_by_pattern(
                patterns,
                _replaceable_lif_window_nodes(window),
            )
            remappable_spike_users = _unique_nodes(
                [user for users in spike_external_users.values() for user in users]
            )

            first = patterns[0]
            xs = [_resolved_replacement_node(pattern.input_node) for pattern in patterns]
            v_init = _resolved_replacement_node(first.v_prev_node)
            anchor, insert_mode, reason = _select_lif_temporal_insert_anchor(gm, window, xs + [v_init])
            if anchor is None:
                if "replacement would create cycle" in reason:
                    stats.temporal_lif_unremappable_external_users += len(remappable_spike_users)
                stats.skip(window, "cannot find legal temporal lif insertion point; " + reason)
                continue
            v_threshold, v_reset, tau, detach_reset = first.lif_params

            insert_ctx = gm.graph.inserting_before(anchor) if insert_mode == "before" else gm.graph.inserting_after(anchor)
            with insert_ctx:
                x_seq = gm.graph.call_function(torch.stack, args=(xs,), kwargs={"dim": 0})
                x_seq.name = f"{first.lif_node.name}_temporal_lif_x_seq"
            with gm.graph.inserting_after(x_seq):
                temporal_tuple = gm.graph.call_function(
                    torch.ops.snn_custom.fused_temporal_lif_state.default,
                    args=(x_seq, v_init, v_threshold, v_reset, tau, detach_reset),
                )
                temporal_tuple.name = f"{first.lif_node.name}_temporal_fused_lif_state"

            ok, reason = _all_inputs_available_for_node(gm, xs + [v_init], x_seq)
            if not ok:
                gm.graph.erase_node(temporal_tuple)
                gm.graph.erase_node(x_seq)
                stats.skip(window, "cannot find legal temporal lif insertion point; " + reason)
                continue
            ok, reason = _all_inputs_available_for_node(gm, [x_seq, v_init], temporal_tuple)
            if not ok:
                gm.graph.erase_node(temporal_tuple)
                gm.graph.erase_node(x_seq)
                stats.skip(window, "cannot find legal temporal lif insertion point; " + reason)
                continue

            with gm.graph.inserting_after(temporal_tuple):
                spike_stack = gm.graph.call_function(operator.getitem, args=(temporal_tuple, 0))
                spike_stack.name = f"{temporal_tuple.name}_spike_stack"
            with gm.graph.inserting_after(spike_stack):
                v_final = gm.graph.call_function(operator.getitem, args=(temporal_tuple, 1))
                v_final.name = f"{temporal_tuple.name}_v_final"

            prev_insert = v_final
            for index, pattern in enumerate(patterns):
                with gm.graph.inserting_after(prev_insert):
                    spike_k = gm.graph.call_function(operator.getitem, args=(spike_stack, index))
                    spike_k.name = f"{temporal_tuple.name}_spike_t{index}"
                prev_insert = spike_k
                pattern.spike_getitem.meta["chronos_replacement_node"] = spike_k
                pattern.spike_getitem.replace_all_uses_with(spike_k)

            patterns[-1].v_getitem.meta["chronos_replacement_node"] = v_final
            patterns[-1].v_getitem.replace_all_uses_with(v_final)
            if remappable_spike_users:
                _move_early_remapped_users_after(gm, remappable_spike_users, prev_insert)
                stats.temporal_lif_remapped_spike_external_users += len(remappable_spike_users)
            _cleanup_lif_window_nodes(gm, window)

            stats.temporal_lif_rewritten_windows += 1
            stats.temporal_lif_replaced_patterns += len(patterns)
            replaced_patterns += len(patterns)
            message = (
                f"[REWRITE][TEMPORAL_LIF] layer={window.layer_id}, window={window.window_id}, "
                f"size={len(patterns)}, first={patterns[0].lif_node.name}, last={patterns[-1].lif_node.name}"
            )
            stats.log.append(message)
            print(message)
        except Exception as exc:
            reason = str(exc)
            stats.skip(window, reason)
            if not isinstance(exc, ValueError):
                traceback.print_exc()

    try:
        gm.graph.lint()
        gm.recompile()
    except Exception:
        print("[WARN][TEMPORAL_LIF] graph lint/recompile failed after standalone LIF rewrite")
        traceback.print_exc()
        raise
    return stats


def rewrite_temporal_lif_avgpool_linear_to_fused(
    gm: torch.fx.GraphModule,
    temporal_windows: List[TemporalLifAvgPoolLinearWindow],
    max_patterns: int,
) -> TemporalLifAvgPoolLinearRewriteStats:
    stats = TemporalLifAvgPoolLinearRewriteStats(
        temporal_lif_avgpool_linear_groups=len({window.layer_id for window in temporal_windows}),
        temporal_lif_avgpool_linear_windows=len(temporal_windows),
        temporal_lif_avgpool_linear_total_windows=len(temporal_windows),
    )
    replaced_patterns = 0
    for window in temporal_windows:
        patterns = window.patterns
        if replaced_patterns + len(patterns) > max_patterns:
            stats.skip(window, "max_patterns")
            continue
        try:
            if not _same_lif_avgpool_linear_params(patterns):
                stats.skip(window, "lif avgpool-linear params differ inside window")
                continue
            if not _same_lif_avgpool_linear_shapes(patterns):
                stats.skip(window, "lif avgpool-linear input shapes differ inside window")
                continue
            ok, reason = check_temporal_lif_avgpool_linear_state_and_acc_chain(patterns)
            if not ok:
                stats.skip(window, f"state/accumulator chain not continuous: {reason}")
                continue
            ok, reason = _lif_avgpool_linear_middle_nodes_have_no_external_uses(window)
            if not ok:
                stats.skip(window, reason)
                continue

            first = patterns[0]
            last = patterns[-1]
            xs = [pattern.input_node for pattern in patterns]
            v_init = first.v_prev_node
            fc_bias = first.fc_bias
            fc_inputs = [first.fc_weight]
            if isinstance(fc_bias, torch.fx.Node):
                fc_inputs.append(fc_bias)
            acc_prev = first.acc_prev
            inputs = xs + [v_init] + fc_inputs
            if isinstance(acc_prev, torch.fx.Node):
                inputs.append(acc_prev)
            anchor, insert_mode, reason = _select_lif_avgpool_linear_temporal_insert_anchor(gm, window, inputs)
            if anchor is None:
                stats.skip(window, "cannot find legal temporal lif avgpool-linear insertion point; " + reason)
                continue
            v_threshold, v_reset, tau, detach_reset = first.lif_params

            insert_ctx = gm.graph.inserting_before(anchor) if insert_mode == "before" else gm.graph.inserting_after(anchor)
            with insert_ctx:
                x_seq = gm.graph.call_function(torch.stack, args=(xs,), kwargs={"dim": 0})
                x_seq.name = f"{first.lif_node.name}_temporal_lif_avgpool_linear_x_seq"
            with gm.graph.inserting_after(x_seq):
                temporal_tuple = gm.graph.call_function(
                    torch.ops.snn_custom.fused_temporal_lif_avgpool_linear.default,
                    args=(x_seq, v_init, first.fc_weight, fc_bias, v_threshold, v_reset, tau, detach_reset),
                )
                temporal_tuple.name = f"{first.lif_node.name}_temporal_fused_lif_avgpool_linear"

            ok, reason = _all_inputs_available_for_node(gm, xs, x_seq)
            if not ok:
                gm.graph.erase_node(temporal_tuple)
                gm.graph.erase_node(x_seq)
                stats.skip(window, "cannot find legal temporal lif avgpool-linear insertion point; " + reason)
                continue
            ok, reason = _all_inputs_available_for_node(gm, [x_seq, v_init, first.fc_weight, fc_bias], temporal_tuple)
            if not ok:
                gm.graph.erase_node(temporal_tuple)
                gm.graph.erase_node(x_seq)
                stats.skip(window, "cannot find legal temporal lif avgpool-linear insertion point; " + reason)
                continue

            with gm.graph.inserting_after(temporal_tuple):
                out_sum = gm.graph.call_function(operator.getitem, args=(temporal_tuple, 0))
                out_sum.name = f"{temporal_tuple.name}_out_sum"
            with gm.graph.inserting_after(out_sum):
                v_final = gm.graph.call_function(operator.getitem, args=(temporal_tuple, 1))
                v_final.name = f"{temporal_tuple.name}_v_final"

            final_acc = out_sum
            if isinstance(acc_prev, torch.fx.Node):
                with gm.graph.inserting_after(v_final):
                    final_acc = gm.graph.call_function(operator.add, args=(acc_prev, out_sum))
                    final_acc.name = f"{temporal_tuple.name}_accumulated"
                ok, reason = _all_inputs_available_for_node(gm, [acc_prev, out_sum], final_acc)
                if not ok:
                    gm.graph.erase_node(final_acc)
                    gm.graph.erase_node(v_final)
                    gm.graph.erase_node(out_sum)
                    gm.graph.erase_node(temporal_tuple)
                    gm.graph.erase_node(x_seq)
                    stats.skip(window, "cannot find legal temporal lif avgpool-linear insertion point; " + reason)
                    continue
            elif acc_prev not in (0, 0.0, None):
                with gm.graph.inserting_after(v_final):
                    final_acc = gm.graph.call_function(operator.add, args=(acc_prev, out_sum))
                    final_acc.name = f"{temporal_tuple.name}_accumulated"

            last.acc_node.replace_all_uses_with(final_acc)
            last.v_getitem.replace_all_uses_with(v_final)
            _cleanup_lif_avgpool_linear_window_nodes(gm, window)

            stats.temporal_lif_avgpool_linear_rewritten_windows += 1
            stats.temporal_lif_avgpool_linear_replaced_patterns += len(patterns)
            replaced_patterns += len(patterns)
            message = (
                f"[REWRITE][TEMPORAL_LIF_AVGPOOL_LINEAR] layer={window.layer_id}, window={window.window_id}, "
                f"size={len(patterns)}, first={patterns[0].lif_node.name}, last={patterns[-1].lif_node.name}"
            )
            stats.log.append(message)
            print(message)
        except Exception as exc:
            reason = str(exc)
            stats.skip(window, reason)
            if not isinstance(exc, ValueError):
                traceback.print_exc()

    try:
        gm.graph.lint()
        gm.recompile()
    except Exception:
        print("[WARN][TEMPORAL_LIF_AVGPOOL_LINEAR] graph lint/recompile failed after temporal LIF avgpool linear rewrite")
        traceback.print_exc()
        raise
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


def dump_temporal_lif_avgpool_linear_patterns(groups: List[TemporalLifAvgPoolLinearGroup], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for group in groups:
        lines.append(f"layer_id: {group.layer_id}")
        lines.append(f"  count={len(group.patterns)}")
        for pattern in group.patterns:
            lines.append(
                "  "
                f"pattern_{pattern.timestep_index}: lif={pattern.lif_node.name}, pool={pattern.pool_node.name}, "
                f"flatten={pattern.flatten_node.name}, linear={pattern.linear_node.name}, acc={pattern.acc_node.name}, "
                f"v={pattern.v_getitem.name}, v_prev={getattr(pattern.v_prev_node, 'name', pattern.v_prev_node)}"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def dump_temporal_lif_avgpool_linear_windows(windows: List[TemporalLifAvgPoolLinearWindow], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for index, window in enumerate(windows):
        lines.append(f"window_{index}:")
        lines.append(f"  layer_id={window.layer_id}")
        lines.append(f"  window_id={window.window_id}")
        lines.append(f"  size={len(window.patterns)}")
        lines.append(f"  patterns={[pattern.lif_node.name for pattern in window.patterns]}")
        lines.append(f"  acc_nodes={[pattern.acc_node.name for pattern in window.patterns]}")
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


def count_fused_temporal_conv_add_lif_state_nodes(gm: torch.fx.GraphModule) -> int:
    return sum(
        1
        for node in gm.graph.nodes
        if node.op == "call_function" and str(node.target) == "snn_custom.fused_temporal_conv_add_lif_state.default"
    )


def count_fused_temporal_lif_state_nodes(gm: torch.fx.GraphModule) -> int:
    return sum(
        1
        for node in gm.graph.nodes
        if node.op == "call_function" and str(node.target) == "snn_custom.fused_temporal_lif_state.default"
    )


def count_fused_temporal_lif_avgpool_linear_nodes(gm: torch.fx.GraphModule) -> int:
    return sum(
        1
        for node in gm.graph.nodes
        if node.op == "call_function" and str(node.target) == "snn_custom.fused_temporal_lif_avgpool_linear.default"
    )


# Deprecated compatibility aliases for downstream scripts that still import the
# old classifier-tail names. New code should use the avgpool-linear names above.
TemporalLifTailPattern = TemporalLifAvgPoolLinearPattern
TemporalLifTailGroup = TemporalLifAvgPoolLinearGroup
TemporalLifTailWindow = TemporalLifAvgPoolLinearWindow
TemporalLifTailRewriteStats = TemporalLifAvgPoolLinearRewriteStats
collect_temporal_lif_tail_patterns = collect_temporal_lif_avgpool_linear_patterns
group_temporal_lif_tail_patterns = group_temporal_lif_avgpool_linear_patterns
make_temporal_lif_tail_windows = make_temporal_lif_avgpool_linear_windows
rewrite_temporal_lif_tail_to_fused = rewrite_temporal_lif_avgpool_linear_to_fused
dump_temporal_lif_tail_patterns = dump_temporal_lif_avgpool_linear_patterns
dump_temporal_lif_tail_windows = dump_temporal_lif_avgpool_linear_windows
count_fused_temporal_lif_tail_nodes = count_fused_temporal_lif_avgpool_linear_nodes
