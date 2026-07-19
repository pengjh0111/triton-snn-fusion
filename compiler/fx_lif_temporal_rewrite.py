import operator
import os
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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
    # Anchor + closure representation (Phase C scheduling fix): patterns for
    # workloads whose instance subgraph is a DAG, not the LIF triple's short
    # linear chain (Mamba's x/z fork-join, unsqueeze branches, etc.), set
    # these instead of relying on the 5 fixed role slots above, which have
    # no way to express arbitrary topology or length. When non-empty,
    # annotate_nodes_with_layer_and_timestep computes this instance's full
    # member set as the graph closure between the two boundaries (forward-
    # reachable from input_boundary, backward-reachable from
    # output_boundary, bounded so the walk never crosses into another
    # instance's own boundary nodes) and gives every member uniform,
    # layer-grouped scheduling priority -- instead of only the explicitly
    # named nodes. Empty (the default) preserves the original 5-role path
    # unchanged for LIF patterns.
    input_boundary: Tuple[torch.fx.Node, ...] = ()
    output_boundary: Tuple[torch.fx.Node, ...] = ()


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
    add_node: Optional[torch.fx.Node] = None
    add_lhs: Optional[torch.fx.Node] = None
    add_rhs: Optional[torch.fx.Node] = None


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
class TemporalLinearLifPattern:
    layer_id: str
    timestep_index: int
    window_id: int
    linear_node: torch.fx.Node
    linear_input: torch.fx.Node
    weight: Any
    bias: Any
    lif_node: torch.fx.Node
    v_prev_node: torch.fx.Node
    spike_getitem: torch.fx.Node
    v_getitem: torch.fx.Node
    v_next_node: torch.fx.Node
    lif_params: Tuple[Any, Any, Any, Any]
    occurrence: int
    input_shape_key: str
    output_shape_key: str
    add_node: Optional[torch.fx.Node] = None
    residual_node: Optional[torch.fx.Node] = None


@dataclass
class TemporalLinearLifGroup:
    layer_id: str
    patterns: List[TemporalLinearLifPattern]


@dataclass
class TemporalLinearLifWindow:
    layer_id: str
    window_id: int
    patterns: List[TemporalLinearLifPattern]


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
class TemporalLinearLifRewriteStats:
    temporal_linear_lif_groups: int = 0
    temporal_linear_lif_windows: int = 0
    temporal_linear_lif_total_windows: int = 0
    temporal_linear_lif_rewritten_windows: int = 0
    temporal_linear_lif_replaced_patterns: int = 0
    temporal_linear_lif_skipped_windows: int = 0
    temporal_linear_lif_remapped_spike_external_users: int = 0
    temporal_linear_lif_unremappable_external_users: int = 0
    temporal_linear_lif_skip_reasons: Dict[str, int] = field(default_factory=dict)
    log: List[str] = field(default_factory=list)

    def skip(self, window: TemporalLinearLifWindow, reason: str):
        self.temporal_linear_lif_skipped_windows += 1
        self.temporal_linear_lif_skip_reasons[reason] = self.temporal_linear_lif_skip_reasons.get(reason, 0) + 1
        message = f"SKIP layer={window.layer_id} window={window.window_id}: {reason}"
        self.log.append(message)
        print(f"[SKIP][TEMPORAL_LINEAR_LIF] {message}")


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


def _get_chronos_meta(node: torch.fx.Node, key: str, default=None):
    meta_key = f"chronos_{key}"
    if meta_key in node.meta:
        return node.meta[meta_key]
    return getattr(node, f"_chronos_{key}", default)


def _param_like_name(node: Any) -> Optional[str]:
    """Identify a placeholder/get_attr node carrying a module parameter --
    torch.compile lifts parameters to placeholder nodes (confirmed via real
    FX dumps: original_fx.txt shows every nn.Parameter as
    `placeholder[target=L_self_...parameters_weight_]`, not get_attr), so
    both ops must be checked to find a layer-identifying node."""
    if not isinstance(node, torch.fx.Node) or node.op not in ("get_attr", "placeholder"):
        return None
    return str(node.target)


def _make_synthetic_temporal_pattern(
    anchor: torch.fx.Node,
    layer_id: str,
    conv_input: torch.fx.Node,
    input_boundary: Tuple[torch.fx.Node, ...] = (),
    output_boundary: Tuple[torch.fx.Node, ...] = (),
) -> "TemporalPattern":
    """Builds a TemporalPattern for split_fx_graph_into_timesteps /
    annotate_temporal_metadata to consume for a non-LIF recurrent workload
    (ConvLSTM/Mamba/DeepSpeech2). Those two callers only ever read
    pattern.conv_node (the per-timestep anchor used to find block
    boundaries) and pattern.layer_id (to group an anchor's T recurring
    instances together, mirroring how conv_weight_key/bn_key identify a
    conv+bn+lif layer's instances) -- the other TemporalPattern fields exist
    only for the LIF-specific fusion rewrite passes, which never consume
    patterns from this function, so they're filled with the anchor node
    itself / harmless placeholders rather than real conv/bn/lif nodes.
    input_boundary/output_boundary opt into the closure-based scheduling
    path in annotate_nodes_with_layer_and_timestep (see TemporalPattern's
    docstring) for patterns whose per-instance subgraph is too long/DAG-
    shaped for the 5 fixed role slots to usefully cover.
    """
    return TemporalPattern(
        layer_id=layer_id,
        timestep_index=0,
        input_boundary=input_boundary,
        output_boundary=output_boundary,
        conv_node=anchor,
        bn_node=anchor,
        lif_node=anchor,
        spike_getitem=anchor,
        v_getitem=anchor,
        conv_input=conv_input,
        conv_weight_key=layer_id,
        bn_key=layer_id,
        v_prev_node=anchor,
        v_next_node=anchor,
        lif_params=(None, None, None, None),
    )


def collect_convlstm_cell_patterns(gm: torch.fx.GraphModule) -> List[TemporalPattern]:
    """Per-timestep anchor for ConvLSTM: xproj = conv_x(x_t) -- the first op
    of each timestep's cell (ChronosConvLSTMCellEager.forward: xproj is
    computed before hproj/chunk/everything else), matching the convention
    conv_node uses for LIF patterns (the conv node is likewise each
    timestep's first op) -- split_fx_graph_into_timesteps positions block
    boundaries starting *at* the marker node, so anchoring on anything but
    the first op of a timestep leaves that timestep's preceding nodes
    orphaned into the previous block (confirmed via a real trace: anchoring
    on the chunk node instead put xproj/hproj/add in the wrong block and
    the very first timestep's ones outside any block at all). Verified by
    confirming this conv2d's *output* reaches a torch.chunk(_, 4, dim=1) --
    i.e. it really is a ConvLSTM gate-projection conv, not an unrelated one
    -- via a bounded forward walk, without anchoring on the chunk itself.
    """
    patterns: List[TemporalPattern] = []
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target not in (torch.conv2d, F.conv2d):
            continue
        weight_x = node.args[1] if len(node.args) > 1 else None
        layer_key = _param_like_name(weight_x)
        if layer_key is None or "conv_x" not in layer_key:
            continue
        # forward-walk a few hops to confirm this conv's output reaches an
        # add -> chunk(_,4,dim=1), the ConvLSTM gate-split signature.
        found_chunk = False
        frontier = list(node.users)
        for _ in range(4):
            next_frontier = []
            for user in frontier:
                if user.target is torch.chunk and len(user.args) > 1 and user.args[1] == 4 and user.kwargs.get("dim") == 1:
                    found_chunk = True
                    break
                next_frontier.extend(user.users)
            if found_chunk:
                break
            frontier = next_frontier
        if not found_chunk:
            continue
        patterns.append(_make_synthetic_temporal_pattern(node, f"convlstm_cell:{layer_key}", node.args[0]))
    return patterns


def collect_mamba_scan_patterns(gm: torch.fx.GraphModule) -> List[TemporalPattern]:
    """Per-timestep anchor for Mamba: y = self.norms[layer_idx](h) -- the
    true first op of each timestep's block (ChronosMamba.step: "residual =
    h; y = self.norms[layer_idx](h)"), *not* xz = in_proj(y) as an earlier
    version of this collector anchored on. For layer 0 specifically, h is
    directly x_t (an external per-t value), so the LayerNorm call sits
    between x_t's getitem and in_proj -- anchoring on in_proj left this one
    real op orphaned in the previous timestep's block (confirmed via a real
    trace: layer 0's in_proj at t>=1 was incorrectly flagged
    feedback_dependency because its *own* input, the LayerNorm, inherited
    the off-by-one; layer 1's in_proj was unaffected since its LayerNorm
    input is layer 0's same-timestep output, not an external getitem).
    Verified by confirming this layer_norm's output reaches, within a few
    hops, a linear whose weight contains "in_proj" feeding a
    torch.chunk(_,2,dim=-1) (the xz -> x,z split) -- i.e. it really is one
    of the per-block pre-norms, not final_norm (which precedes the head
    linear instead, and only exists once per timestep, not once per layer).
    """
    patterns: List[TemporalPattern] = []
    layer_norm_targets = (torch.nn.functional.layer_norm, F.layer_norm)
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target not in layer_norm_targets:
            continue
        weight = node.args[2] if len(node.args) > 2 else node.kwargs.get("weight")
        layer_key = _param_like_name(weight)
        if layer_key is None:
            continue
        found_in_proj_chunk = False
        frontier = list(node.users)
        for _ in range(4):
            next_frontier = []
            for user in frontier:
                if user.target in (torch._C._nn.linear, F.linear):
                    user_weight = user.args[1] if len(user.args) > 1 else None
                    user_weight_key = _param_like_name(user_weight)
                    if user_weight_key is not None and "in_proj" in user_weight_key:
                        for grandchild in user.users:
                            if (
                                grandchild.target is torch.chunk
                                and len(grandchild.args) > 1
                                and grandchild.args[1] == 2
                                and grandchild.kwargs.get("dim") == -1
                            ):
                                found_in_proj_chunk = True
                                break
                if found_in_proj_chunk:
                    break
                next_frontier.extend(user.users)
            if found_in_proj_chunk:
                break
            frontier = next_frontier
        if not found_in_proj_chunk:
            continue
        patterns.append((node, layer_key))

    # Correlate each LayerNorm anchor with its own instance's scan chain
    # (the reorder_fx_graph_by_temporal_windows scheduler needs role nodes
    # spanning the *scan chain itself*, not just the LayerNorm anchor, to
    # give those nodes scheduling priority -- otherwise the window-spanning
    # rewrite's stack/fused-call insertion point can violate topological
    # order, since same-timestep, cross-layer consumers of a scan's y
    # output (e.g. layer N+1's norm(h) at the same t) are not guaranteed to
    # sort after the whole window's scan chain by default. Both lists occur
    # in the same relative graph order (each layer-timestep's LayerNorm is
    # immediately followed, a bounded number of hops later, by that same
    # instance's scan) since Dynamo's unrolled trace is deterministic and
    # uniform, so pairing by position index is exact here (verified against
    # a real trace) without an expensive full reachability search.
    hA_nodes = [n for n in gm.graph.nodes if n.target is torch.exp and _match_mamba_scan_step(n) is not None]
    result: List[TemporalPattern] = []
    for (anchor, layer_key), hA_node in zip(patterns, hA_nodes):
        match = _match_mamba_scan_step(hA_node)
        input_boundary = (anchor.args[0],) if isinstance(anchor.args[0], torch.fx.Node) else ()
        output_boundary = (match["y"], match["ssm_state_new"]) if match is not None else ()
        pattern = _make_synthetic_temporal_pattern(
            anchor, f"mamba_scan:{layer_key}", anchor.args[0],
            input_boundary=input_boundary, output_boundary=output_boundary,
        )
        if match is not None:
            pattern.bn_node = match["hA"]
            pattern.lif_node = match["ssm_state_new"]
            pattern.spike_getitem = match["y"]
            pattern.v_getitem = match["y"]
        result.append(pattern)
    return result


def collect_gru_cell_patterns(gm: torch.fx.GraphModule) -> List[TemporalPattern]:
    """Per-timestep anchor for the DeepSpeech2 GRU stack: xproj = w_x(x_t) --
    the first op of each timestep's cell (ChronosGRUCellEager.forward:
    xproj computed before hproj/chunk/everything else). Verified by
    confirming this linear's output reaches a torch.chunk(_,3,dim=-1), the
    xproj -> r,z,n split signature.
    """
    patterns: List[TemporalPattern] = []
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target not in (torch._C._nn.linear, F.linear):
            continue
        weight_x = node.args[1] if len(node.args) > 1 else None
        layer_key = _param_like_name(weight_x)
        if layer_key is None or "w_x" not in layer_key:
            continue
        found_chunk = False
        frontier = list(node.users)
        for _ in range(3):
            next_frontier = []
            for user in frontier:
                if user.target is torch.chunk and len(user.args) > 1 and user.args[1] == 3 and user.kwargs.get("dim") == -1:
                    found_chunk = True
                    break
                next_frontier.extend(user.users)
            if found_chunk:
                break
            frontier = next_frontier
        if not found_chunk:
            continue
        patterns.append(_make_synthetic_temporal_pattern(node, f"gru_cell:{layer_key}", node.args[0]))
    return patterns


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


def _node_rank(node: torch.fx.Node) -> Optional[int]:
    meta = node.meta.get("tensor_meta") or node.meta.get("val")
    shape = getattr(meta, "shape", None)
    return len(shape) if shape is not None else None


def _temporal_value_source_key(node: torch.fx.Node) -> str:
    """Collapse timestep getitems to their shared temporal producer."""
    current = node
    seen = set()
    while isinstance(current, torch.fx.Node) and current not in seen:
        seen.add(current)
        replacement = current.meta.get("chronos_replacement_node")
        if isinstance(replacement, torch.fx.Node):
            current = replacement
            continue
        if current.op == "call_function" and current.target is operator.getitem and current.args:
            parent = current.args[0]
            if isinstance(parent, torch.fx.Node):
                current = parent
                continue
        break
    return _node_key(current) if isinstance(current, torch.fx.Node) else repr(current)


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
            "snn_custom.fused_temporal_pointwise_conv_lif_state.default",
            "snn_custom.fused_temporal_depthwise_conv_lif_state.default",
            "snn_custom.fused_temporal_conv_lif_state_depthwise.default",
            "snn_custom.fused_temporal_conv_add_lif_state.default",
            "snn_custom.fused_temporal_conv_add_lif_state_depthwise.default",
            "snn_custom.fused_temporal_lif_state.default",
        ):
            continue
        input_node = node.args[0]
        add_node = input_node if _is_add_node(input_node) else None
        add_lhs = add_node.args[0] if add_node is not None and len(add_node.args) >= 2 else None
        add_rhs = add_node.args[1] if add_node is not None and len(add_node.args) >= 2 else None
        if _is_linear_output_node(input_node):
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
        shape_key = _shape_key_from_node(input_node)
        lif_params = tuple(node.args[2:6])
        source_key = ""
        if add_node is not None and isinstance(add_lhs, torch.fx.Node) and isinstance(add_rhs, torch.fx.Node):
            source_key = (
                f"|add_sources={_temporal_value_source_key(add_lhs)}+"
                f"{_temporal_value_source_key(add_rhs)}"
            )
        layer_id = f"standalone_lif|occurrence={occurrence}|{shape_key}|params={repr(lif_params)}{source_key}"
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
                add_node=add_node,
                add_lhs=add_lhs,
                add_rhs=add_rhs,
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
    if node.target in (torch._C._nn.linear, F.linear):
        return True
    target_text = str(node.target)
    return "aten.linear" in target_text or target_text.endswith(".linear.default")


def _getitem_index(node: torch.fx.Node):
    if node.op != "call_function" or node.target is not operator.getitem or len(node.args) < 2:
        return None
    index = node.args[1]
    if isinstance(index, slice):
        return None
    try:
        return int(index)
    except (TypeError, ValueError):
        return None


def _is_temporal_stack_tensor(node: torch.fx.Node) -> bool:
    if not isinstance(node, torch.fx.Node):
        return False
    if node.meta.get("chronos_temporal_layout") == "stack":
        return True
    if node.op != "call_function" or node.target is not operator.getitem:
        return False
    if _getitem_index(node) != 0 or not node.args or not isinstance(node.args[0], torch.fx.Node):
        return False
    producer = node.args[0]
    if producer.op != "call_function":
        return False
    return "snn_custom.fused_temporal_" in str(producer.target)


def _match_temporal_stack_avgpool_flatten(node: torch.fx.Node) -> Optional[Tuple[torch.fx.Node, torch.fx.Node, torch.fx.Node, int]]:
    if not isinstance(node, torch.fx.Node) or not _is_flatten_batch_preserving(node):
        return None
    pool_node = node.args[0] if node.args and isinstance(node.args[0], torch.fx.Node) else None
    if pool_node is None or not _is_adaptive_avg_pool_1x1(pool_node):
        return None
    spike_t = pool_node.args[0] if pool_node.args and isinstance(pool_node.args[0], torch.fx.Node) else None
    if spike_t is None or spike_t.op != "call_function" or spike_t.target is not operator.getitem:
        return None
    timestep = _getitem_index(spike_t)
    if timestep is None:
        return None
    stack_node = spike_t.args[0] if spike_t.args and isinstance(spike_t.args[0], torch.fx.Node) else None
    if stack_node is None or not _is_temporal_stack_tensor(stack_node):
        return None
    return stack_node, spike_t, pool_node, int(timestep)


def _single_user_node(node: torch.fx.Node) -> Optional[torch.fx.Node]:
    users = list(node.users)
    return users[0] if len(users) == 1 else None


def _extract_linear_weight_bias(linear_node: torch.fx.Node) -> Tuple[Any, Any]:
    weight = linear_node.args[1] if len(linear_node.args) > 1 else linear_node.kwargs.get("weight")
    bias = linear_node.args[2] if len(linear_node.args) > 2 else linear_node.kwargs.get("bias", None)
    return weight, bias


def _extract_linear_input(linear_node: torch.fx.Node):
    if len(linear_node.args) > 0:
        return linear_node.args[0]
    return linear_node.kwargs.get("input")


def collect_temporal_linear_lif_state_patterns(
    gm: torch.fx.GraphModule,
    excluded_lif_nodes=None,
) -> List[TemporalLinearLifPattern]:
    excluded = set(excluded_lif_nodes or [])
    fallback_counts: Dict[str, int] = {}
    patterns: List[TemporalLinearLifPattern] = []
    for node in gm.graph.nodes:
        if node in excluded or not is_custom_lif_state_node(node):
            continue
        ok, reason = _lif_state_is_usable(node)
        if not ok:
            print(f"[SKIP][TEMPORAL_LINEAR_LIF] lif={node.name}: {reason}")
            continue
        if len(node.args) < 6 or not isinstance(node.args[0], torch.fx.Node):
            print(f"[SKIP][TEMPORAL_LINEAR_LIF] lif={node.name}: unsupported lif args")
            continue
        lif_input = node.args[0]
        add_node = lif_input if _is_add_node(lif_input) else None
        residual_node = None
        linear_node = lif_input
        if add_node is not None and len(add_node.args) >= 2:
            lhs, rhs = add_node.args[:2]
            if isinstance(lhs, torch.fx.Node) and _is_linear_node(lhs):
                linear_node, residual_node = lhs, rhs
            elif isinstance(rhs, torch.fx.Node) and _is_linear_node(rhs):
                linear_node, residual_node = rhs, lhs
        if not isinstance(linear_node, torch.fx.Node) or not _is_linear_node(linear_node):
            continue
        linear_input = _extract_linear_input(linear_node)
        if not isinstance(linear_input, torch.fx.Node):
            print(f"[SKIP][TEMPORAL_LINEAR_LIF] linear={linear_node.name}: unsupported linear input")
            continue
        linear_external = []
        for user in linear_node.users:
            if user is node or user is add_node:
                continue
            if _is_zeros_like_of(user, linear_node) and len(node.args) > 1 and node.args[1] is user:
                continue
            linear_external.append(user.name)
        if linear_external:
            print(
                f"[SKIP][TEMPORAL_LINEAR_LIF] linear={linear_node.name}: "
                f"linear output has external users {linear_external}"
            )
            continue

        timestep = _chronos_meta(node, "timestep", None)
        window_id = _chronos_meta(node, "window_id", None)
        occurrence = _chronos_meta(node, "occurrence", None)
        if not isinstance(timestep, int):
            fallback_key = f"linear_lif|{_node_key(_extract_linear_weight_bias(linear_node)[0])}"
            timestep = fallback_counts.get(fallback_key, 0)
            fallback_counts[fallback_key] = timestep + 1
        if not isinstance(window_id, int):
            window_id = 0
        if not isinstance(occurrence, int):
            occurrence = 0

        weight, bias = _extract_linear_weight_bias(linear_node)
        getitems = find_tuple_getitems(node)
        lif_params = tuple(node.args[2:6])
        input_shape_key = _shape_key_from_node(linear_input)
        output_shape_key = _shape_key_from_node(linear_node)
        layer_id = (
            f"linear_lif|occurrence={occurrence}|weight={_node_key(weight)}|bias={_node_key(bias)}|"
            f"input={input_shape_key}|output={output_shape_key}|params={repr(lif_params)}"
        )
        patterns.append(
            TemporalLinearLifPattern(
                layer_id=layer_id,
                timestep_index=int(timestep),
                window_id=int(window_id),
                linear_node=linear_node,
                linear_input=linear_input,
                weight=weight,
                bias=bias,
                lif_node=node,
                v_prev_node=node.args[1],
                spike_getitem=getitems[0],
                v_getitem=getitems[1],
                v_next_node=getitems[1],
                lif_params=lif_params,
                occurrence=int(occurrence),
                input_shape_key=input_shape_key,
                output_shape_key=output_shape_key,
                add_node=add_node,
                residual_node=residual_node if isinstance(residual_node, torch.fx.Node) else None,
            )
        )
    return patterns


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


def group_temporal_linear_lif_patterns(patterns: List[TemporalLinearLifPattern]) -> List[TemporalLinearLifGroup]:
    grouped: Dict[str, List[TemporalLinearLifPattern]] = {}
    for pattern in patterns:
        grouped.setdefault(pattern.layer_id, []).append(pattern)
    groups = []
    for layer_id, items in grouped.items():
        groups.append(TemporalLinearLifGroup(layer_id=layer_id, patterns=sorted(items, key=lambda p: p.timestep_index)))
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


def check_temporal_linear_lif_state_chain(patterns: List[TemporalLinearLifPattern]) -> Tuple[bool, str]:
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


def make_temporal_linear_lif_windows(
    groups: List[TemporalLinearLifGroup],
    window_size: int,
    allow_tail: bool,
) -> List[TemporalLinearLifWindow]:
    if window_size < 1:
        return []
    windows: List[TemporalLinearLifWindow] = []
    for group in groups:
        group_window_ids = {pattern.window_id for pattern in group.patterns}
        group_timesteps = [pattern.timestep_index for pattern in group.patterns]
        # Transformer-style graphs may not have Conv/BN/LIF markers, so the
        # generic temporal annotation pass cannot split the FX graph into
        # timestep blocks. In that case collection falls back to monotonic
        # timestep indices but leaves window_id at 0. Split those groups by the
        # requested linear temporal window here so --temporal-fuse-window still
        # changes the final linear-LIF IR.
        use_timestep_window = (
            len(group_window_ids) == 1
            and len(group_timesteps) > window_size
            and max(group_timesteps, default=0) >= window_size
        )
        by_window: Dict[int, List[TemporalLinearLifPattern]] = {}
        for pattern in group.patterns:
            window_id = pattern.timestep_index // window_size if use_timestep_window else pattern.window_id
            by_window.setdefault(window_id, []).append(pattern)
        for window_id, items in sorted(by_window.items()):
            items = sorted(items, key=lambda pattern: pattern.timestep_index)
            if len(items) < window_size and not allow_tail:
                print(f"[SKIP][TEMPORAL_LINEAR_LIF] layer={group.layer_id}: tail size={len(items)} < window={window_size}")
                continue
            if len(items) <= 1:
                continue
            expected = list(range(items[0].timestep_index, items[0].timestep_index + len(items)))
            actual = [pattern.timestep_index for pattern in items]
            if actual != expected:
                print(f"[SKIP][TEMPORAL_LINEAR_LIF] layer={group.layer_id}: timesteps not continuous: {actual}")
                continue
            ok, reason = check_temporal_linear_lif_state_chain(items)
            if not ok:
                print(f"[SKIP][TEMPORAL_LINEAR_LIF] layer={group.layer_id}: state chain not continuous: {reason}")
                continue
            windows.append(TemporalLinearLifWindow(layer_id=group.layer_id, window_id=window_id, patterns=items))
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


def _linear_lif_middle_v_next_has_no_external_uses(window: TemporalLinearLifWindow) -> Tuple[bool, str]:
    patterns = window.patterns
    for idx, pattern in enumerate(patterns[:-1]):
        allowed = patterns[idx + 1].lif_node
        external = [user.name for user in pattern.v_getitem.users if user is not allowed]
        if external:
            return False, f"middle v_next {pattern.v_getitem.name} has external users {external}"
    return True, ""


def _same_linear_lif_params(patterns: List[TemporalLinearLifPattern]) -> bool:
    first = patterns[0]
    return all(
        pattern.lif_params == first.lif_params
        and _node_key(pattern.weight) == _node_key(first.weight)
        and _node_key(pattern.bias) == _node_key(first.bias)
        for pattern in patterns
    )


def _same_linear_lif_shapes(patterns: List[TemporalLinearLifPattern]) -> bool:
    first = patterns[0]
    return all(
        pattern.input_shape_key == first.input_shape_key and pattern.output_shape_key == first.output_shape_key
        for pattern in patterns
    )


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


def _replaceable_temporal_window_nodes(window: TemporalWindow) -> set:
    nodes = set()
    for pattern in window.patterns:
        nodes.update([pattern.conv_node, pattern.bn_node, pattern.lif_node, pattern.spike_getitem, pattern.v_getitem])
        v_prev = pattern.lif_node.args[1] if len(pattern.lif_node.args) > 1 else None
        if isinstance(v_prev, torch.fx.Node) and _is_zeros_like_of(v_prev, pattern.bn_node):
            nodes.add(v_prev)
    return nodes


def _select_temporal_conv_lif_insert_anchor(
    gm: torch.fx.GraphModule,
    window: TemporalWindow,
    inputs: List[Any],
) -> Tuple[Optional[torch.fx.Node], str, str]:
    order = {node: index for index, node in enumerate(gm.graph.nodes)}
    first = window.patterns[0].conv_node
    first_order = order[first]
    replaceable = _replaceable_temporal_window_nodes(window)
    input_nodes = [value for value in inputs if isinstance(value, torch.fx.Node)]
    for node in input_nodes:
        if node in replaceable:
            return None, "skip", f"input {node.name} is produced by nodes being replaced"

    late_inputs = [node for node in input_nodes if order.get(node, -1) >= first_order]
    if not late_inputs:
        return first, "before", ""

    anchor = max(late_inputs, key=lambda node: order[node])
    anchor_order = order[anchor]
    first_spike_order = min(order[pattern.spike_getitem] for pattern in window.patterns)
    if anchor_order >= first_spike_order:
        return (
            None,
            "skip",
            f"latest input {anchor.name} appears after first replaceable spike {window.patterns[0].spike_getitem.name}",
        )
    return anchor, "after", ""


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
        candidates.extend([pattern.spike_getitem, pattern.v_getitem, pattern.lif_node, pattern.add_node])
    unique = []
    seen = set()
    for node in candidates:
        if isinstance(node, torch.fx.Node) and node not in seen and node in order:
            unique.append(node)
            seen.add(node)
    for node in sorted(unique, key=lambda n: order[n], reverse=True):
        _erase_if_unused(gm, node)


def _replaceable_linear_lif_window_nodes(window: TemporalLinearLifWindow) -> set:
    nodes = set()
    for pattern in window.patterns:
        nodes.update([pattern.linear_node, pattern.add_node, pattern.lif_node, pattern.spike_getitem, pattern.v_getitem])
        v_prev = pattern.lif_node.args[1] if len(pattern.lif_node.args) > 1 else None
        if isinstance(v_prev, torch.fx.Node) and _is_zeros_like_of(v_prev, pattern.linear_node):
            nodes.add(v_prev)
    return nodes


def _external_linear_lif_window_users(window: TemporalLinearLifWindow) -> List[torch.fx.Node]:
    replaceable = _replaceable_linear_lif_window_nodes(window)
    users: List[torch.fx.Node] = []
    for user in window.patterns[-1].v_getitem.users:
        if user not in replaceable:
            users.append(user)
    return users


def _select_linear_lif_temporal_insert_anchor(
    gm: torch.fx.GraphModule,
    window: TemporalLinearLifWindow,
    inputs: List[Any],
) -> Tuple[Optional[torch.fx.Node], str, str]:
    order = {node: index for index, node in enumerate(gm.graph.nodes)}
    first = window.patterns[0].linear_node
    first_order = order[first]
    replaceable = _replaceable_linear_lif_window_nodes(window)
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
        for user in _external_linear_lif_window_users(window)
        if order.get(user, anchor_order + 1) <= anchor_order
    ]
    if early_users:
        return None, "skip", f"external users {early_users} appear before latest input {anchor.name}"
    return anchor, "after", ""


def _cleanup_linear_lif_window_nodes(gm: torch.fx.GraphModule, window: TemporalLinearLifWindow):
    order = {node: index for index, node in enumerate(gm.graph.nodes)}
    candidates = []
    for pattern in window.patterns:
        candidates.extend([pattern.spike_getitem, pattern.v_getitem, pattern.lif_node, pattern.add_node, pattern.linear_node])
        v_prev = pattern.lif_node.args[1] if len(pattern.lif_node.args) > 1 else None
        if isinstance(v_prev, torch.fx.Node) and _is_zeros_like_of(v_prev, pattern.linear_node):
            candidates.append(v_prev)
    unique = []
    seen = set()
    for node in candidates:
        if isinstance(node, torch.fx.Node) and node not in seen and node in order:
            unique.append(node)
            seen.add(node)
    for node in sorted(unique, key=lambda n: order[n], reverse=True):
        _erase_if_unused(gm, node)


_TEMPORAL_FUSED_STATE_OP_TARGETS = {
    "snn_custom.fused_temporal_conv_lif_state.default",
    "snn_custom.fused_temporal_pointwise_conv_lif_state.default",
    "snn_custom.fused_temporal_depthwise_conv_lif_state.default",
    "snn_custom.fused_temporal_conv_lif_state_depthwise.default",
    "snn_custom.fused_temporal_conv_add_lif_state.default",
    "snn_custom.fused_temporal_conv_add_lif_state_depthwise.default",
    "snn_custom.fused_temporal_lif_state.default",
    "snn_custom.fused_temporal_linear_lif_state.default",
    "snn_custom.fused_temporal_linear_lif_state_packed.default",
    "snn_custom.fused_temporal_batched_linear_lif_state.default",
    "snn_custom.fused_temporal_batched_linear_add_lif_state.default",
    "snn_custom.fused_temporal_add_lif_state.default",
    "snn_custom.fused_temporal_lif_avgpool_linear.default",
    "snn_custom.fused_temporal_lif_tail.default",
}


def _is_temporal_fused_v_final_node(node) -> bool:
    if not isinstance(node, torch.fx.Node) or node.op != "call_function":
        return False
    if _getitem_index(node) != 1 or not node.args:
        return False
    producer = node.args[0]
    return (
        isinstance(producer, torch.fx.Node)
        and producer.op == "call_function"
        and str(producer.target) in _TEMPORAL_FUSED_STATE_OP_TARGETS
    )


def _materialize_temporal_fused_v_init(
    gm: torch.fx.GraphModule,
    v_init,
    *,
    origin: str,
):
    return v_init


def _materialize_temporal_fused_v_final(
    gm: torch.fx.GraphModule,
    v_final: torch.fx.Node,
    *,
    enabled: bool,
    origin: str,
) -> torch.fx.Node:
    return v_final


def _insert_size_after(gm: torch.fx.GraphModule, tensor_node: torch.fx.Node, dim: int, prev: torch.fx.Node) -> torch.fx.Node:
    with gm.graph.inserting_after(prev):
        size_node = gm.graph.call_method("size", args=(tensor_node, dim))
    return size_node


def _as_pair(value) -> Tuple[int, int]:
    if isinstance(value, int):
        return int(value), int(value)
    return int(value[0]), int(value[1])


def classify_conv_kind(conv_module, folded_weight, stride, padding, dilation, groups) -> str:
    del conv_module
    if not isinstance(folded_weight, torch.Tensor) or folded_weight.dim() != 4:
        return "regular"
    stride_pair = _as_pair(stride)
    padding_pair = _as_pair(padding)
    dilation_pair = _as_pair(dilation)
    groups = int(groups)
    out_channels = int(folded_weight.shape[0])
    weight_in_channels = int(folded_weight.shape[1])
    kernel_h = int(folded_weight.shape[2])
    kernel_w = int(folded_weight.shape[3])
    in_channels = int(weight_in_channels * groups)

    is_depthwise = (
        groups == in_channels
        and out_channels == in_channels
        and weight_in_channels == 1
        and kernel_h == 3
        and kernel_w == 3
        and dilation_pair == (1, 1)
        and padding_pair == (1, 1)
        and stride_pair in ((1, 1), (2, 2))
    )
    if is_depthwise:
        return "depthwise"

    is_pointwise = (
        groups == 1
        and kernel_h == 1
        and kernel_w == 1
        and padding_pair == (0, 0)
        and dilation_pair == (1, 1)
    )
    if is_pointwise:
        return "pointwise"

    return "regular"


def _temporal_conv_lif_op_for_kind(kind: str):
    if kind == "depthwise":
        return torch.ops.snn_custom.fused_temporal_depthwise_conv_lif_state.default
    if kind == "pointwise":
        return torch.ops.snn_custom.fused_temporal_pointwise_conv_lif_state.default
    return torch.ops.snn_custom.fused_temporal_conv_lif_state.default


def _temporal_conv_add_lif_op_for_kind(kind: str):
    if kind == "depthwise":
        return torch.ops.snn_custom.fused_temporal_conv_add_lif_state_depthwise.default
    return torch.ops.snn_custom.fused_temporal_conv_add_lif_state.default


def _debug_conv_classify(name: str, kind: str, folded_weight, stride, padding, groups) -> None:
    if os.environ.get("CHRONOS_FX_CONV_CLASSIFY", "0") != "1":
        return
    weight_shape = tuple(folded_weight.shape) if isinstance(folded_weight, torch.Tensor) else None
    print(
        f"[CHRONOS_FX_CONV_CLASSIFY] name={name} kind={kind} "
        f"weight_shape={weight_shape} stride={_as_pair(stride)} padding={_as_pair(padding)} groups={int(groups)}"
    )


def _annotate_chronos_fused_temporal_node(
    node: torch.fx.Node,
    *,
    op_kind: str,
    layer_id: str,
    window_id: int,
    patterns: List[Any],
) -> None:
    timesteps = [int(getattr(pattern, "timestep_index", index)) for index, pattern in enumerate(patterns)]
    if timesteps:
        time_range = (min(timesteps), max(timesteps) + 1)
    else:
        time_range = (0, 0)
    node.meta["chronos_op_kind"] = op_kind
    node.meta["chronos_layer_id"] = layer_id
    node.meta["chronos_window_id"] = int(window_id)
    node.meta["chronos_time_range"] = time_range
    node.meta["chronos_window_size"] = len(patterns)


def _insert_binary_after(
    gm: torch.fx.GraphModule,
    target,
    lhs,
    rhs,
    prev: torch.fx.Node,
    name: str,
) -> torch.fx.Node:
    with gm.graph.inserting_after(prev):
        node = gm.graph.call_function(target, args=(lhs, rhs))
        node.name = name
    return node


def _insert_conv_out_dim_after(
    gm: torch.fx.GraphModule,
    input_dim,
    *,
    kernel: int,
    stride: int,
    padding: int,
    dilation: int,
    prev: torch.fx.Node,
    name: str,
) -> torch.fx.Node:
    # (input + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1
    node = _insert_binary_after(gm, operator.add, input_dim, 2 * int(padding), prev, f"{name}_pad")
    node = _insert_binary_after(gm, operator.sub, node, int(dilation) * (int(kernel) - 1), node, f"{name}_kernel")
    node = _insert_binary_after(gm, operator.sub, node, 1, node, f"{name}_minus_one")
    node = _insert_binary_after(gm, operator.floordiv, node, int(stride), node, f"{name}_div")
    node = _insert_binary_after(gm, operator.add, node, 1, node, name)
    return node


def _insert_temporal_conv_lif_out_buffers(
    gm: torch.fx.GraphModule,
    x_seq: torch.fx.Node,
    *,
    T: int,
    out_channels: int,
    kernel_hw: Tuple[int, int],
    stride,
    padding,
    dilation,
) -> Tuple[torch.fx.Node, torch.fx.Node]:
    stride_h, stride_w = _as_pair(stride)
    pad_h, pad_w = _as_pair(padding)
    dil_h, dil_w = _as_pair(dilation)
    k_h, k_w = int(kernel_hw[0]), int(kernel_hw[1])
    n = _insert_size_after(gm, x_seq, 1, x_seq)
    h = _insert_size_after(gm, x_seq, 3, n)
    w = _insert_size_after(gm, x_seq, 4, h)
    out_h = _insert_conv_out_dim_after(
        gm,
        h,
        kernel=k_h,
        stride=stride_h,
        padding=pad_h,
        dilation=dil_h,
        prev=w,
        name=f"{x_seq.name}_out_h",
    )
    out_w = _insert_conv_out_dim_after(
        gm,
        w,
        kernel=k_w,
        stride=stride_w,
        padding=pad_w,
        dilation=dil_w,
        prev=out_h,
        name=f"{x_seq.name}_out_w",
    )
    with gm.graph.inserting_after(out_w):
        spike_out = gm.graph.call_method("new_empty", args=(x_seq, (int(T), n, int(out_channels), out_h, out_w)))
        spike_out.name = f"{x_seq.name}_spike_out"
    with gm.graph.inserting_after(spike_out):
        v_out = gm.graph.call_method("new_empty", args=(x_seq, (n, int(out_channels), out_h, out_w)))
        v_out.name = f"{x_seq.name}_v_out"
    return spike_out, v_out


def _insert_temporal_linear_lif_out_buffers(
    gm: torch.fx.GraphModule,
    x_seq: torch.fx.Node,
    weight,
    *,
    T: int,
) -> Tuple[torch.fx.Node, torch.fx.Node]:
    with gm.graph.inserting_after(x_seq):
        x_shape = gm.graph.call_method("size", args=(x_seq,))
        x_shape.name = f"{x_seq.name}_shape"
    if isinstance(weight, torch.fx.Node):
        out_features = _insert_size_after(gm, weight, 0, x_shape)
        prev = out_features
    else:
        out_features = int(weight.shape[0])
        prev = x_shape
    with gm.graph.inserting_after(prev):
        spike_prefix = gm.graph.call_function(operator.getitem, args=(x_shape, slice(None, -1, None)))
        spike_prefix.name = f"{x_seq.name}_spike_prefix"
    with gm.graph.inserting_after(spike_prefix):
        spike_shape = gm.graph.call_function(operator.add, args=(spike_prefix, (out_features,)))
        spike_shape.name = f"{x_seq.name}_spike_shape"
    with gm.graph.inserting_after(spike_shape):
        v_prefix = gm.graph.call_function(operator.getitem, args=(x_shape, slice(1, -1, None)))
        v_prefix.name = f"{x_seq.name}_v_prefix"
    with gm.graph.inserting_after(v_prefix):
        v_shape = gm.graph.call_function(operator.add, args=(v_prefix, (out_features,)))
        v_shape.name = f"{x_seq.name}_v_shape"
    with gm.graph.inserting_after(v_shape):
        spike_out = gm.graph.call_method("new_empty", args=(x_seq, spike_shape))
        spike_out.name = f"{x_seq.name}_spike_out"
    with gm.graph.inserting_after(spike_out):
        v_out = gm.graph.call_method("new_empty", args=(x_seq, v_shape))
        v_out.name = f"{x_seq.name}_v_out"
    return spike_out, v_out


def _clone_temporal_state_after(
    gm: torch.fx.GraphModule,
    state_node: torch.fx.Node,
    after_node: torch.fx.Node,
    *,
    name: str,
) -> torch.fx.Node:
    with gm.graph.inserting_after(after_node):
        cloned = gm.graph.call_function(torch.clone, args=(state_node,))
        cloned.name = name
        cloned.meta["chronos_origin"] = "temporal_fused_state_cudagraph_pool_clone"
    return cloned


def _is_get_attr_scalar_zero(gm: torch.fx.GraphModule, node) -> bool:
    if not isinstance(node, torch.fx.Node) or node.op != "get_attr":
        return False
    try:
        value = getattr(gm, str(node.target))
    except AttributeError:
        return False
    if not isinstance(value, torch.Tensor) or value.numel() != 1:
        return False
    try:
        return float(value.detach().cpu().item()) == 0.0
    except Exception:
        return False


def _materialize_zero_scalar_like_after(
    gm: torch.fx.GraphModule,
    like_node: torch.fx.Node,
    after_node: torch.fx.Node,
    *,
    name: str,
) -> torch.fx.Node:
    with gm.graph.inserting_after(after_node):
        zero = gm.graph.call_method("new_zeros", args=(like_node, ()))
        zero.name = name
        zero.meta["chronos_origin"] = "temporal_linear_lif_device_scalar_v_init"
    return zero


def _make_temporal_avgpool_flatten_x_seq(
    gm: torch.fx.GraphModule,
    patterns: List[TemporalLinearLifPattern],
    anchor: torch.fx.Node,
    insert_mode: str,
) -> Tuple[Optional[torch.fx.Node], List[torch.fx.Node], str]:
    matches = []
    for pattern in patterns:
        match = _match_temporal_stack_avgpool_flatten(pattern.linear_input)
        if match is None:
            return None, [], "linear input is not temporal_stack[t] -> adaptive_avg_pool2d(1x1) -> flatten"
        matches.append(match)

    stack_node = matches[0][0]
    if any(match[0] is not stack_node for match in matches):
        return None, [], "linear inputs come from different temporal stacks"
    timesteps = [match[3] for match in matches]
    if timesteps != list(range(len(patterns))):
        return None, [], f"temporal stack timesteps are not contiguous from zero: {timesteps}"

    insert_ctx = gm.graph.inserting_before(anchor) if insert_mode == "before" else gm.graph.inserting_after(anchor)
    with insert_ctx:
        stack_batched = gm.graph.call_function(torch.flatten, args=(stack_node, 0, 1))
        stack_batched.name = f"{stack_node.name}_temporal_avgpool_input_batched"
        stack_batched.meta["chronos_temporal_layout"] = "batched_tn"
        stack_batched.meta["chronos_origin"] = "temporal_avgpool_flatten_propagation"
        stack_batched.meta["chronos_T"] = len(patterns)

    with gm.graph.inserting_after(stack_batched):
        pooled_batched = gm.graph.call_function(F.adaptive_avg_pool2d, args=(stack_batched, (1, 1)))
        pooled_batched.name = f"{stack_node.name}_temporal_avgpool_batched"
        pooled_batched.meta["chronos_temporal_layout"] = "batched_tn"
        pooled_batched.meta["chronos_origin"] = "temporal_avgpool_flatten_propagation"
        pooled_batched.meta["chronos_T"] = len(patterns)

    with gm.graph.inserting_after(pooled_batched):
        flat_batched = gm.graph.call_function(torch.flatten, args=(pooled_batched, 1, -1))
        flat_batched.name = f"{stack_node.name}_temporal_flatten_batched"
        flat_batched.meta["chronos_temporal_layout"] = "batched_tn"
        flat_batched.meta["chronos_origin"] = "temporal_avgpool_flatten_propagation"
        flat_batched.meta["chronos_T"] = len(patterns)

    with gm.graph.inserting_after(flat_batched):
        x_seq = gm.graph.call_function(torch.unflatten, args=(flat_batched, 0, (len(patterns), -1)))
        x_seq.name = f"{patterns[0].linear_node.name}_temporal_avgpool_flatten_x_seq"
        x_seq.meta["chronos_temporal_layout"] = "stack"
        x_seq.meta["chronos_origin"] = "temporal_avgpool_flatten_propagation"
        x_seq.meta["chronos_T"] = len(patterns)

    cleanup_nodes = []
    for _stack, spike_t, pool_node, _timestep in matches:
        cleanup_nodes.extend([spike_t, pool_node])
    cleanup_nodes.extend([pattern.linear_input for pattern in patterns])
    return x_seq, cleanup_nodes, ""


def _cleanup_extra_nodes(gm: torch.fx.GraphModule, nodes: List[torch.fx.Node]):
    order = {node: index for index, node in enumerate(gm.graph.nodes)}
    unique = []
    seen = set()
    for node in nodes:
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
            v_init = _materialize_temporal_fused_v_init(
                gm,
                v_init,
                origin="temporal_conv_lif_cross_window_v_clone",
            )

            weight_attr = add_tensor_attr(gm, "_fx_temporal_folded_conv_bn_weight", folded_weight)
            bias_attr = add_tensor_attr(gm, "_fx_temporal_folded_conv_bn_bias", folded_bias)
            weight_node = _insert_get_attr_before(gm, first.conv_node, weight_attr)
            bias_node = _insert_get_attr_before(gm, first.conv_node, bias_attr)
            xs = [_resolved_replacement_node(pattern.conv_input) for pattern in patterns]
            fused_inputs = xs + [v_init, weight_node, bias_node]
            anchor, insert_mode, reason = _select_temporal_conv_lif_insert_anchor(gm, window, fused_inputs)
            if anchor is None:
                raise ValueError("cannot find legal temporal conv lif insertion point; " + reason)
            v_threshold, v_reset, tau, detach_reset = first.lif_params
            conv_kind = classify_conv_kind(None, folded_weight, stride, padding, dilation, groups)
            conv_op = _temporal_conv_lif_op_for_kind(conv_kind)
            _debug_conv_classify(first.conv_node.name, conv_kind, folded_weight, stride, padding, groups)

            insert_ctx = gm.graph.inserting_before(anchor) if insert_mode == "before" else gm.graph.inserting_after(anchor)
            with insert_ctx:
                temporal_tuple = gm.graph.call_function(
                    conv_op,
                    args=(xs, weight_node, bias_node, v_init, stride, padding, dilation, groups, v_threshold, v_reset, tau, detach_reset),
                )
                temporal_tuple.name = f"{first.conv_node.name}_temporal_fused_{conv_kind}_conv_lif_state"
                _annotate_chronos_fused_temporal_node(
                    temporal_tuple,
                    op_kind=conv_kind,
                    layer_id=window.layer_id,
                    window_id=window.window_id,
                    patterns=patterns,
                )

            ok, reason = _all_inputs_available_for_node(gm, xs + [v_init, weight_node, bias_node], temporal_tuple)
            if not ok:
                gm.graph.erase_node(temporal_tuple)
                raise ValueError("cannot find legal temporal conv lif insertion point; " + reason)

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
            v_init = _materialize_temporal_fused_v_init(
                gm,
                v_init,
                origin="temporal_residual_conv_lif_cross_window_v_clone",
            )

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
            conv_kind = classify_conv_kind(None, folded_weight, stride, padding, dilation, groups)
            conv_op = _temporal_conv_add_lif_op_for_kind(conv_kind)
            _debug_conv_classify(first.conv_node.name, conv_kind, folded_weight, stride, padding, groups)

            insert_ctx = gm.graph.inserting_before(anchor) if insert_mode == "before" else gm.graph.inserting_after(anchor)
            with insert_ctx:
                temporal_tuple = gm.graph.call_function(
                    conv_op,
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
                _annotate_chronos_fused_temporal_node(
                    temporal_tuple,
                    op_kind="residual",
                    layer_id=window.layer_id,
                    window_id=window.window_id,
                    patterns=patterns,
                )

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
            v_final = _materialize_temporal_fused_v_final(
                gm,
                v_final,
                enabled=len(patterns) > 1,
                origin="temporal_residual_conv_lif_v_final_graph_output_clone",
            )

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
            use_add_lif = all(
                pattern.add_node is not None
                and isinstance(pattern.add_lhs, torch.fx.Node)
                and isinstance(pattern.add_rhs, torch.fx.Node)
                for pattern in patterns
            )
            lhs_values = [_resolved_replacement_node(pattern.add_lhs) for pattern in patterns] if use_add_lif else []
            rhs_values = [_resolved_replacement_node(pattern.add_rhs) for pattern in patterns] if use_add_lif else []
            v_init = _resolved_replacement_node(first.v_prev_node)
            v_init = _materialize_temporal_fused_v_init(
                gm,
                v_init,
                origin="temporal_lif_cross_window_v_clone",
            )
            data_inputs = lhs_values + rhs_values if use_add_lif else xs
            anchor, insert_mode, reason = _select_lif_temporal_insert_anchor(gm, window, data_inputs + [v_init])
            if anchor is None:
                if "replacement would create cycle" in reason:
                    stats.temporal_lif_unremappable_external_users += len(remappable_spike_users)
                stats.skip(window, "cannot find legal temporal lif insertion point; " + reason)
                continue
            v_threshold, v_reset, tau, detach_reset = first.lif_params

            insert_ctx = gm.graph.inserting_before(anchor) if insert_mode == "before" else gm.graph.inserting_after(anchor)
            with insert_ctx:
                x_seq = gm.graph.call_function(torch.stack, args=(lhs_values if use_add_lif else xs,), kwargs={"dim": 0})
                x_seq.name = f"{first.lif_node.name}_temporal_lif_x_seq"
            with gm.graph.inserting_after(x_seq):
                if use_add_lif:
                    rhs_seq = gm.graph.call_function(torch.stack, args=(rhs_values,), kwargs={"dim": 0})
                    rhs_seq.name = f"{first.lif_node.name}_temporal_add_lif_rhs_seq"
            temporal_anchor = rhs_seq if use_add_lif else x_seq
            with gm.graph.inserting_after(temporal_anchor):
                op = (
                    torch.ops.snn_custom.fused_temporal_add_lif_state.default
                    if use_add_lif else torch.ops.snn_custom.fused_temporal_lif_state.default
                )
                op_args = (
                    (x_seq, rhs_seq, v_init, v_threshold, v_reset, tau, detach_reset)
                    if use_add_lif else (x_seq, v_init, v_threshold, v_reset, tau, detach_reset)
                )
                temporal_tuple = gm.graph.call_function(op, args=op_args)
                temporal_tuple.name = f"{first.lif_node.name}_temporal_fused_lif_state"
                _annotate_chronos_fused_temporal_node(
                    temporal_tuple,
                    op_kind="lif",
                    layer_id=window.layer_id,
                    window_id=window.window_id,
                    patterns=patterns,
                )

            ok, reason = _all_inputs_available_for_node(gm, data_inputs + [v_init], x_seq)
            if not ok:
                gm.graph.erase_node(temporal_tuple)
                gm.graph.erase_node(x_seq)
                stats.skip(window, "cannot find legal temporal lif insertion point; " + reason)
                continue
            temporal_inputs = [x_seq, v_init] + ([rhs_seq] if use_add_lif else [])
            ok, reason = _all_inputs_available_for_node(gm, temporal_inputs, temporal_tuple)
            if not ok:
                gm.graph.erase_node(temporal_tuple)
                if use_add_lif and len(rhs_seq.users) == 0:
                    gm.graph.erase_node(rhs_seq)
                gm.graph.erase_node(x_seq)
                stats.skip(window, "cannot find legal temporal lif insertion point; " + reason)
                continue

            with gm.graph.inserting_after(temporal_tuple):
                spike_stack = gm.graph.call_function(operator.getitem, args=(temporal_tuple, 0))
                spike_stack.name = f"{temporal_tuple.name}_spike_stack"
            with gm.graph.inserting_after(spike_stack):
                v_final = gm.graph.call_function(operator.getitem, args=(temporal_tuple, 1))
                v_final.name = f"{temporal_tuple.name}_v_final"
            v_final = _materialize_temporal_fused_v_final(
                gm,
                v_final,
                enabled=len(patterns) > 1,
                origin="temporal_lif_v_final_graph_output_clone",
            )

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


def rewrite_temporal_linear_lif_state_to_fused(
    gm: torch.fx.GraphModule,
    temporal_windows: List[TemporalLinearLifWindow],
    max_patterns: int,
) -> TemporalLinearLifRewriteStats:
    stats = TemporalLinearLifRewriteStats(
        temporal_linear_lif_groups=len({window.layer_id for window in temporal_windows}),
        temporal_linear_lif_windows=len(temporal_windows),
        temporal_linear_lif_total_windows=len(temporal_windows),
    )
    replaced_patterns = 0
    for window in temporal_windows:
        patterns = window.patterns
        if replaced_patterns + len(patterns) > max_patterns:
            stats.skip(window, "max_patterns")
            continue
        try:
            if not _same_linear_lif_params(patterns):
                stats.skip(window, "linear/lif params differ inside temporal linear lif window")
                continue
            if not _same_linear_lif_shapes(patterns):
                stats.skip(window, "linear input/output shapes differ inside temporal linear lif window")
                continue
            ok, reason = check_temporal_linear_lif_state_chain(patterns)
            if not ok:
                stats.skip(window, f"state chain not continuous: {reason}")
                continue
            ok, reason = _linear_lif_middle_v_next_has_no_external_uses(window)
            if not ok:
                stats.skip(window, reason)
                continue
            spike_external_users = _external_spike_users_by_pattern(
                patterns,
                _replaceable_linear_lif_window_nodes(window),
            )
            remappable_spike_users = _unique_nodes(
                [user for users in spike_external_users.values() for user in users]
            )

            first = patterns[0]
            xs = [_resolved_replacement_node(pattern.linear_input) for pattern in patterns]
            use_linear_add_lif = all(pattern.add_node is not None and pattern.residual_node is not None for pattern in patterns)
            residuals = [_resolved_replacement_node(pattern.residual_node) for pattern in patterns] if use_linear_add_lif else []
            use_batched_linear = (_node_rank(first.linear_input) or 0) >= 3
            weight = _resolved_replacement_node(first.weight)
            bias = _resolved_replacement_node(first.bias) if isinstance(first.bias, torch.fx.Node) else first.bias
            v_init = _resolved_replacement_node(first.v_prev_node)
            if isinstance(v_init, torch.fx.Node) and _is_zeros_like_of(v_init, first.linear_node):
                zero_attr = add_tensor_attr(gm, "_fx_temporal_linear_lif_zero_scalar_v_init", torch.tensor(0.0))
                v_init = _insert_get_attr_before(gm, first.linear_node, zero_attr)
            v_init = _materialize_temporal_fused_v_init(
                gm,
                v_init,
                origin="temporal_linear_lif_cross_window_v_clone",
            )

            temporal_stack_match = [_match_temporal_stack_avgpool_flatten(pattern.linear_input) for pattern in patterns]
            use_temporal_avgpool_flatten = (
                all(match is not None for match in temporal_stack_match)
                and len({match[0] for match in temporal_stack_match if match is not None}) == 1
                and [match[3] for match in temporal_stack_match if match is not None] == list(range(len(patterns)))
            )
            linear_data_inputs = [temporal_stack_match[0][0]] if use_temporal_avgpool_flatten else xs

            input_values = linear_data_inputs + residuals + [weight, v_init]
            if isinstance(bias, torch.fx.Node):
                input_values.append(bias)
            anchor, insert_mode, reason = _select_linear_lif_temporal_insert_anchor(gm, window, input_values)
            if anchor is None:
                if "replacement would create cycle" in reason:
                    stats.temporal_linear_lif_unremappable_external_users += len(remappable_spike_users)
                stats.skip(window, "cannot find legal temporal linear lif insertion point; " + reason)
                continue
            v_threshold, v_reset, tau, detach_reset = first.lif_params

            extra_cleanup_nodes: List[torch.fx.Node] = []
            if use_temporal_avgpool_flatten:
                x_seq, extra_cleanup_nodes, reason = _make_temporal_avgpool_flatten_x_seq(
                    gm,
                    patterns,
                    anchor,
                    insert_mode,
                )
                if x_seq is None:
                    stats.skip(window, "cannot propagate temporal avgpool/flatten: " + reason)
                    continue
            else:
                insert_ctx = gm.graph.inserting_before(anchor) if insert_mode == "before" else gm.graph.inserting_after(anchor)
                with insert_ctx:
                    x_seq = gm.graph.call_function(torch.stack, args=(xs, 0))
                    x_seq.name = f"{first.linear_node.name}_temporal_linear_lif_x_seq"
                    x_seq.meta["chronos_temporal_layout"] = "stack"
                    x_seq.meta["chronos_origin"] = "temporal_linear_lif_packed_input"
                    x_seq.meta["chronos_T"] = len(xs)

            materialize_scalar_v_init = _is_get_attr_scalar_zero(gm, v_init)
            if use_batched_linear:
                if materialize_scalar_v_init:
                    v_init = _materialize_zero_scalar_like_after(
                        gm, x_seq, x_seq,
                        name=f"{first.linear_node.name}_temporal_batched_linear_lif_v_init_device",
                    )
                batched_insert_after = v_init if materialize_scalar_v_init else x_seq
                with gm.graph.inserting_after(batched_insert_after):
                    if use_linear_add_lif:
                        residual_seq = gm.graph.call_function(torch.stack, args=(residuals, 0))
                        residual_seq.name = f"{first.linear_node.name}_temporal_linear_add_lif_residual_seq"
                temporal_insert_after = residual_seq if use_linear_add_lif else batched_insert_after
                with gm.graph.inserting_after(temporal_insert_after):
                    temporal_target = (
                        torch.ops.snn_custom.fused_temporal_batched_linear_add_lif_state.default
                        if use_linear_add_lif
                        else torch.ops.snn_custom.fused_temporal_batched_linear_lif_state.default
                    )
                    temporal_args = (
                        (x_seq, residual_seq, weight, bias, v_init, v_threshold, v_reset, tau, detach_reset)
                        if use_linear_add_lif
                        else (x_seq, weight, bias, v_init, v_threshold, v_reset, tau, detach_reset)
                    )
                    temporal_op = gm.graph.call_function(temporal_target, args=temporal_args)
                    temporal_op.name = f"{first.linear_node.name}_temporal_fused_batched_linear_lif_state"
                with gm.graph.inserting_after(temporal_op):
                    spike_stack = gm.graph.call_function(operator.getitem, args=(temporal_op, 0))
                    spike_stack.name = f"{temporal_op.name}_spike_stack"
                with gm.graph.inserting_after(spike_stack):
                    v_final = gm.graph.call_function(operator.getitem, args=(temporal_op, 1))
                    v_final.name = f"{temporal_op.name}_v_final"
                _annotate_chronos_fused_temporal_node(
                    temporal_op, op_kind="linear", layer_id=window.layer_id,
                    window_id=window.window_id, patterns=patterns,
                )
                v_final_for_users = v_final
            else:
                spike_stack, v_final = _insert_temporal_linear_lif_out_buffers(gm, x_seq, weight, T=len(patterns))
                spike_stack.name = f"{first.linear_node.name}_temporal_fused_linear_lif_state_spike_stack"
                v_final.name = f"{first.linear_node.name}_temporal_fused_linear_lif_state_v_final"
                temporal_insert_after = v_final
                if materialize_scalar_v_init:
                    v_init = _materialize_zero_scalar_like_after(
                        gm, x_seq, v_final,
                        name=f"{first.linear_node.name}_temporal_linear_lif_v_init_device",
                    )
                    temporal_insert_after = v_init
                with gm.graph.inserting_after(temporal_insert_after):
                    temporal_op = gm.graph.call_function(
                        torch.ops.snn_custom.fused_temporal_linear_lif_state_packed_out.default,
                        args=(x_seq, weight, bias, v_init, v_threshold, v_reset, tau, detach_reset, spike_stack, v_final),
                    )
                    temporal_op.name = f"{first.linear_node.name}_temporal_fused_linear_lif_state_out"
                    _annotate_chronos_fused_temporal_node(
                        temporal_op, op_kind="linear", layer_id=window.layer_id,
                        window_id=window.window_id, patterns=patterns,
                    )
                v_final_for_users = _clone_temporal_state_after(
                    gm, v_final, temporal_op, name=f"{v_final.name}_pool_safe",
                )

            x_seq_inputs = linear_data_inputs if use_temporal_avgpool_flatten else xs
            ok, reason = _all_inputs_available_for_node(gm, x_seq_inputs, x_seq)
            if not ok:
                if len(temporal_op.users) == 0:
                    gm.graph.erase_node(temporal_op)
                if len(x_seq.users) == 0:
                    gm.graph.erase_node(x_seq)
                stats.skip(window, "cannot find legal temporal linear lif insertion point; " + reason)
                continue
            fused_input_values = [x_seq, weight, v_init] + residuals
            if isinstance(bias, torch.fx.Node):
                fused_input_values.append(bias)
            ok, reason = _all_inputs_available_for_node(gm, fused_input_values, temporal_op)
            if not ok:
                if len(temporal_op.users) == 0:
                    gm.graph.erase_node(temporal_op)
                if len(x_seq.users) == 0:
                    gm.graph.erase_node(x_seq)
                stats.skip(window, "cannot find legal temporal linear lif insertion point; " + reason)
                continue

            prev_insert = v_final_for_users
            for index, pattern in enumerate(patterns):
                with gm.graph.inserting_after(prev_insert):
                    spike_k = gm.graph.call_function(operator.getitem, args=(spike_stack, index))
                    spike_k.name = f"{temporal_op.name}_spike_t{index}"
                prev_insert = spike_k
                pattern.spike_getitem.meta["chronos_replacement_node"] = spike_k
                pattern.spike_getitem.replace_all_uses_with(spike_k)

            patterns[-1].v_getitem.meta["chronos_replacement_node"] = v_final_for_users
            patterns[-1].v_getitem.replace_all_uses_with(v_final_for_users)
            if remappable_spike_users:
                _move_early_remapped_users_after(gm, remappable_spike_users, prev_insert)
                stats.temporal_linear_lif_remapped_spike_external_users += len(remappable_spike_users)
            _cleanup_linear_lif_window_nodes(gm, window)
            if extra_cleanup_nodes:
                _cleanup_extra_nodes(gm, extra_cleanup_nodes)

            stats.temporal_linear_lif_rewritten_windows += 1
            stats.temporal_linear_lif_replaced_patterns += len(patterns)
            replaced_patterns += len(patterns)
            message = (
                f"[REWRITE][TEMPORAL_LINEAR_LIF] layer={window.layer_id}, window={window.window_id}, "
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
        print("[WARN][TEMPORAL_LINEAR_LIF] graph lint/recompile failed after temporal Linear+LIF rewrite")
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
            v_init = _materialize_temporal_fused_v_init(
                gm,
                v_init,
                origin="temporal_lif_avgpool_linear_cross_window_v_clone",
            )
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
                _annotate_chronos_fused_temporal_node(
                    temporal_tuple,
                    op_kind="linear",
                    layer_id=window.layer_id,
                    window_id=window.window_id,
                    patterns=patterns,
                )

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
            v_final = _materialize_temporal_fused_v_final(
                gm,
                v_final,
                enabled=len(patterns) > 1,
                origin="temporal_lif_avgpool_linear_v_final_graph_output_clone",
            )

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


def dump_temporal_linear_lif_patterns(groups: List[TemporalLinearLifGroup], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for group in groups:
        lines.append(f"layer_id: {group.layer_id}")
        lines.append(f"  count={len(group.patterns)}")
        for pattern in group.patterns:
            lines.append(
                "  "
                f"pattern_{pattern.timestep_index}: linear={pattern.linear_node.name}, "
                f"lif={pattern.lif_node.name}, spike={pattern.spike_getitem.name}, "
                f"v={pattern.v_getitem.name}, v_prev={getattr(pattern.v_prev_node, 'name', pattern.v_prev_node)}"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def dump_temporal_linear_lif_windows(windows: List[TemporalLinearLifWindow], path: Path):
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
        if node.op == "call_function"
        and str(node.target)
        in {
            "snn_custom.fused_temporal_conv_lif_state.default",
            "snn_custom.fused_temporal_pointwise_conv_lif_state.default",
            "snn_custom.fused_temporal_depthwise_conv_lif_state.default",
            "snn_custom.fused_temporal_conv_lif_state_depthwise.default",
            "snn_custom.fused_temporal_conv_lif_state_packed_out.default",
        }
    )


def count_fused_temporal_conv_add_lif_state_nodes(gm: torch.fx.GraphModule) -> int:
    return sum(
        1
        for node in gm.graph.nodes
        if node.op == "call_function"
        and str(node.target)
        in {
            "snn_custom.fused_temporal_conv_add_lif_state.default",
            "snn_custom.fused_temporal_conv_add_lif_state_depthwise.default",
        }
    )


def count_fused_temporal_lif_state_nodes(gm: torch.fx.GraphModule) -> int:
    return sum(
        1
        for node in gm.graph.nodes
        if node.op == "call_function"
        and str(node.target) in {
            "snn_custom.fused_temporal_lif_state.default",
            "snn_custom.fused_temporal_add_lif_state.default",
        }
    )


def count_fused_temporal_lif_avgpool_linear_nodes(gm: torch.fx.GraphModule) -> int:
    return sum(
        1
        for node in gm.graph.nodes
        if node.op == "call_function" and str(node.target) == "snn_custom.fused_temporal_lif_avgpool_linear.default"
    )


def count_fused_temporal_linear_lif_state_nodes(gm: torch.fx.GraphModule) -> int:
    return sum(
        1
        for node in gm.graph.nodes
        if node.op == "call_function"
        and str(node.target)
        in {
            "snn_custom.fused_temporal_linear_lif_state.default",
            "snn_custom.fused_temporal_linear_lif_state_packed.default",
            "snn_custom.fused_temporal_linear_lif_state_packed_out.default",
            "snn_custom.fused_temporal_batched_linear_lif_state.default",
            "snn_custom.fused_temporal_batched_linear_add_lif_state.default",
        }
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


################################################################################
# Kairos sequence-input workload rewriters (Phase C step 2): swap each
# matched cell's raw gate-chain subgraph for the corresponding Phase B
# custom op call. Independent of the LIF fusion machinery above --
# self-contained pattern match + graph.eliminate_dead_code() cleanup rather
# than the TemporalWindow/_cleanup_window_nodes bookkeeping those passes use,
# since these ops don't share that machinery's per-field shape
# (spike_getitem/v_getitem etc. don't apply to a plain elementwise cell).
# Every match step requires len(users)==1 on the intermediate node being
# consumed next: conservative-by-construction, matching the "middle value
# has a consumer outside the chain => skip" rule these passes must follow --
# any extra consumer breaks the single-user check and the whole node is
# left untouched.
################################################################################


def rewrite_convlstm_cell_to_fused(gm: torch.fx.GraphModule) -> int:
    """Match ChronosConvLSTMCellEager.forward's gate chain (chunk(4,dim=1) of
    xproj+hproj through to h_t/c_t; see collect_convlstm_cell_patterns for
    the anchor-half of this pattern) and replace it with a single
    fused_convlstm_cell(gates_sum, c_prev) call. Idempotent: once rewritten,
    the chunk/sigmoid/tanh nodes this looks for no longer exist, so a second
    pass over the same graph finds nothing.
    """
    replaced = 0
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target is not torch.chunk:
            continue
        if len(node.args) < 2 or node.args[1] != 4 or node.kwargs.get("dim") != 1:
            continue
        if len(node.users) != 4:
            continue
        add_node = node.args[0]
        if not isinstance(add_node, torch.fx.Node) or add_node.target not in (operator.add, torch.add):
            continue
        if len(add_node.users) != 1:
            continue

        getitems: Dict[int, torch.fx.Node] = {}
        ok = True
        for user in node.users:
            idx = _getitem_index(user)
            if idx is None or idx in getitems:
                ok = False
                break
            getitems[idx] = user
        if not ok or set(getitems.keys()) != {0, 1, 2, 3}:
            continue
        i_node, f_node, g_node, o_node = getitems[0], getitems[1], getitems[2], getitems[3]
        if any(len(gn.users) != 1 for gn in (i_node, f_node, g_node, o_node)):
            continue

        sig_f = next(iter(f_node.users))
        if sig_f.target is not torch.sigmoid or len(sig_f.users) != 1:
            continue
        mul_fc = next(iter(sig_f.users))
        if mul_fc.target not in (operator.mul, torch.mul) or len(mul_fc.users) != 1 or len(mul_fc.args) < 2:
            continue
        if mul_fc.args[0] is sig_f:
            c_prev_node = mul_fc.args[1]
        elif mul_fc.args[1] is sig_f:
            c_prev_node = mul_fc.args[0]
        else:
            continue
        if not isinstance(c_prev_node, torch.fx.Node):
            continue

        sig_i = next(iter(i_node.users))
        if sig_i.target is not torch.sigmoid or len(sig_i.users) != 1:
            continue
        tanh_g = next(iter(g_node.users))
        if tanh_g.target is not torch.tanh or len(tanh_g.users) != 1:
            continue
        mul_ig_candidates = set(sig_i.users) & set(tanh_g.users)
        if len(mul_ig_candidates) != 1:
            continue
        mul_ig = next(iter(mul_ig_candidates))
        if mul_ig.target not in (operator.mul, torch.mul) or len(mul_ig.users) != 1:
            continue

        c_t_candidates = set(mul_fc.users) & set(mul_ig.users)
        if len(c_t_candidates) != 1:
            continue
        c_t_node = next(iter(c_t_candidates))
        if c_t_node.target not in (operator.add, torch.add):
            continue

        sig_o = next(iter(o_node.users))
        if sig_o.target is not torch.sigmoid or len(sig_o.users) != 1:
            continue
        tanh_c_t_candidates = [u for u in c_t_node.users if u.target is torch.tanh]
        if len(tanh_c_t_candidates) != 1 or len(tanh_c_t_candidates[0].users) != 1:
            continue
        tanh_c_t = tanh_c_t_candidates[0]
        h_t_candidates = set(sig_o.users) & set(tanh_c_t.users)
        if len(h_t_candidates) != 1:
            continue
        h_t_node = next(iter(h_t_candidates))
        if h_t_node.target not in (operator.mul, torch.mul):
            continue

        with gm.graph.inserting_before(node):
            fused = gm.graph.call_function(
                torch.ops.snn_custom.fused_convlstm_cell,
                args=(add_node, c_prev_node),
            )
            fused.name = f"{node.name}_fused_convlstm_cell"
            new_h = gm.graph.call_function(operator.getitem, args=(fused, 0))
            new_h.name = f"{node.name}_fused_h_t"
            new_c = gm.graph.call_function(operator.getitem, args=(fused, 1))
            new_c.name = f"{node.name}_fused_c_t"

        h_t_node.replace_all_uses_with(new_h)
        c_t_node.replace_all_uses_with(new_c)
        replaced += 1

    if replaced:
        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()
    return replaced


def rewrite_gru_cell_to_fused(gm: torch.fx.GraphModule) -> int:
    """Match ChronosGRUCellEager.forward's gate chain (two chunk(3,dim=-1)
    splits of xproj/hproj through to h_t; see collect_gru_cell_patterns for
    the anchor-half of this pattern) and replace it with a single
    fused_gru_cell(xproj, hproj, h_prev) call.
    """
    replaced = 0
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target is not torch.chunk:
            continue
        if len(node.args) < 2 or node.args[1] != 3 or node.kwargs.get("dim") != -1:
            continue
        if len(node.users) != 3:
            continue
        hproj_node = node.args[0]
        if not isinstance(hproj_node, torch.fx.Node) or hproj_node.target not in (torch._C._nn.linear, F.linear):
            continue

        getitems: Dict[int, torch.fx.Node] = {}
        ok = True
        for user in node.users:
            idx = _getitem_index(user)
            if idx is None or idx in getitems:
                ok = False
                break
            getitems[idx] = user
        if not ok or set(getitems.keys()) != {0, 1, 2}:
            continue
        hproj_r, hproj_z, hproj_n = getitems[0], getitems[1], getitems[2]

        # hproj_r/hproj_z feed an add with the sibling xproj chunk's r/z;
        # hproj_n feeds a mul with r first. Use hproj_r's add partner to
        # locate the xproj chunk (and confirm it is genuinely a chunk(3,-1)
        # of a linear call, i.e. xproj, not some unrelated node).
        if len(hproj_r.users) != 1 or len(hproj_z.users) != 1 or len(hproj_n.users) != 1:
            continue
        add_r = next(iter(hproj_r.users))
        if add_r.target not in (operator.add, torch.add) or len(add_r.users) != 1 or len(add_r.args) < 2:
            continue
        xproj_r = add_r.args[0] if add_r.args[1] is hproj_r else (add_r.args[1] if add_r.args[0] is hproj_r else None)
        if not isinstance(xproj_r, torch.fx.Node):
            continue
        xproj_chunk = xproj_r.args[0] if _getitem_index(xproj_r) == 0 and isinstance(xproj_r.args[0], torch.fx.Node) else None
        if xproj_chunk is None or xproj_chunk.target is not torch.chunk:
            continue
        if len(xproj_chunk.args) < 2 or xproj_chunk.args[1] != 3 or xproj_chunk.kwargs.get("dim") != -1:
            continue
        xproj_node = xproj_chunk.args[0]
        if not isinstance(xproj_node, torch.fx.Node) or xproj_node.target not in (torch._C._nn.linear, F.linear):
            continue
        xproj_getitems: Dict[int, torch.fx.Node] = {}
        ok = True
        for user in xproj_chunk.users:
            idx = _getitem_index(user)
            if idx is None or idx in xproj_getitems:
                ok = False
                break
            xproj_getitems[idx] = user
        if not ok or set(xproj_getitems.keys()) != {0, 1, 2}:
            continue
        xproj_r2, xproj_z, xproj_n = xproj_getitems[0], xproj_getitems[1], xproj_getitems[2]
        if xproj_r2 is not xproj_r:
            continue
        if any(len(gn.users) != 1 for gn in (xproj_r, xproj_z, xproj_n)):
            continue

        sig_r = add_r
        if len(sig_r.users) != 1:
            continue
        r_node = next(iter(sig_r.users))
        if r_node.target is not torch.sigmoid or len(r_node.users) != 1:
            continue

        add_z_candidates = set(xproj_z.users) & set(hproj_z.users)
        if len(add_z_candidates) != 1:
            continue
        add_z = next(iter(add_z_candidates))
        if add_z.target not in (operator.add, torch.add) or len(add_z.users) != 1:
            continue
        z_node = next(iter(add_z.users))
        # z legitimately feeds both (1-z) and z*h_prev -- exactly 2 users,
        # unlike every other intermediate in this chain which feeds exactly
        # one downstream op.
        if z_node.target is not torch.sigmoid or len(z_node.users) != 2:
            continue

        mul_r_hn_candidates = set(r_node.users) & set(hproj_n.users)
        if len(mul_r_hn_candidates) != 1:
            continue
        mul_r_hn = next(iter(mul_r_hn_candidates))
        if mul_r_hn.target not in (operator.mul, torch.mul) or len(mul_r_hn.users) != 1:
            continue
        add_n_candidates = set(xproj_n.users) & set(mul_r_hn.users)
        if len(add_n_candidates) != 1:
            continue
        add_n = next(iter(add_n_candidates))
        if add_n.target not in (operator.add, torch.add) or len(add_n.users) != 1:
            continue
        n_node = next(iter(add_n.users))
        if n_node.target is not torch.tanh or len(n_node.users) != 1:
            continue

        # h_t = (1-z)*n + z*h_prev
        one_minus_z_candidates = [u for u in z_node.users if u.target in (operator.sub, torch.sub)]
        if len(one_minus_z_candidates) != 1:
            continue
        one_minus_z = one_minus_z_candidates[0]
        if len(one_minus_z.users) != 1:
            continue
        mul_1mz_n_candidates = set(one_minus_z.users) & set(n_node.users)
        if len(mul_1mz_n_candidates) != 1:
            continue
        mul_1mz_n = next(iter(mul_1mz_n_candidates))
        if mul_1mz_n.target not in (operator.mul, torch.mul) or len(mul_1mz_n.users) != 1:
            continue
        mul_z_h_candidates = [u for u in z_node.users if u.target in (operator.mul, torch.mul)]
        if len(mul_z_h_candidates) != 1:
            continue
        mul_z_h = mul_z_h_candidates[0]
        if len(mul_z_h.args) < 2 or len(mul_z_h.users) != 1:
            continue
        h_prev_node = mul_z_h.args[0] if mul_z_h.args[1] is z_node else (mul_z_h.args[1] if mul_z_h.args[0] is z_node else None)
        if not isinstance(h_prev_node, torch.fx.Node):
            continue
        h_t_candidates = set(mul_1mz_n.users) & set(mul_z_h.users)
        if len(h_t_candidates) != 1:
            continue
        h_t_node = next(iter(h_t_candidates))
        if h_t_node.target not in (operator.add, torch.add):
            continue

        with gm.graph.inserting_before(xproj_chunk):
            fused = gm.graph.call_function(
                torch.ops.snn_custom.fused_gru_cell,
                args=(xproj_node, hproj_node, h_prev_node),
            )
            fused.name = f"{node.name}_fused_gru_cell"

        h_t_node.replace_all_uses_with(fused)
        replaced += 1

    if replaced:
        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()
    return replaced


def _match_mamba_scan_step(hA_node: torch.fx.Node) -> Optional[Dict[str, torch.fx.Node]]:
    """hA_node must be torch.exp(dt.unsqueeze(-1) * A); verifies and extracts
    the full canonical selective-scan step exactly as
    ChronosMambaBlockEager.forward computes it:
        hA = exp(dt.unsqueeze(-1) * A)
        ssm_state_new = hA*ssm_state_prev + (dt.unsqueeze(-1)*B_ssm.unsqueeze(1))*x.unsqueeze(-1)
        y = (ssm_state_new*C_ssm.unsqueeze(1)).sum(-1) + D*x
    Every intermediate is checked for single-user except dt (legitimately
    unsqueezed twice) and x (legitimately consumed by both the scan and,
    outside this pattern, x_proj/D-skip -- not itself checked here since
    those extra uses are outside what this function inspects). Returns None
    on any mismatch (conservative, matching the other two rewriters' style).
    """
    if hA_node.target is not torch.exp or len(hA_node.args) < 1 or len(hA_node.users) != 1:
        return None
    mul_1 = hA_node.args[0]
    if not isinstance(mul_1, torch.fx.Node) or mul_1.target not in (operator.mul, torch.mul) or len(mul_1.args) < 2:
        return None

    def _is_unsqueeze(n):
        return isinstance(n, torch.fx.Node) and n.op == "call_method" and n.target == "unsqueeze"

    a0, a1 = mul_1.args[0], mul_1.args[1]
    if _is_unsqueeze(a0) and _param_like_name(a1) is not None:
        unsqueeze_dt_1, a_node = a0, a1
    elif _is_unsqueeze(a1) and _param_like_name(a0) is not None:
        unsqueeze_dt_1, a_node = a1, a0
    else:
        return None
    if len(unsqueeze_dt_1.args) < 1 or len(unsqueeze_dt_1.users) != 1:
        return None
    dt_node = unsqueeze_dt_1.args[0]
    if not isinstance(dt_node, torch.fx.Node):
        return None

    mul_2 = next(iter(hA_node.users))
    if mul_2.target not in (operator.mul, torch.mul) or len(mul_2.args) < 2 or len(mul_2.users) != 1:
        return None
    ssm_state_prev = mul_2.args[0] if mul_2.args[1] is hA_node else (mul_2.args[1] if mul_2.args[0] is hA_node else None)
    if not isinstance(ssm_state_prev, torch.fx.Node):
        return None

    dt_other_users = [u for u in dt_node.users if u is not unsqueeze_dt_1]
    if len(dt_other_users) != 1:
        return None
    unsqueeze_dt_2 = dt_other_users[0]
    if not _is_unsqueeze(unsqueeze_dt_2) or len(unsqueeze_dt_2.users) != 1:
        return None
    mul_3 = next(iter(unsqueeze_dt_2.users))
    if mul_3.target not in (operator.mul, torch.mul) or len(mul_3.args) < 2 or len(mul_3.users) != 1:
        return None
    unsqueeze_b = mul_3.args[0] if mul_3.args[1] is unsqueeze_dt_2 else (mul_3.args[1] if mul_3.args[0] is unsqueeze_dt_2 else None)
    if not _is_unsqueeze(unsqueeze_b) or len(unsqueeze_b.args) < 1 or len(unsqueeze_b.users) != 1:
        return None
    b_ssm_node = unsqueeze_b.args[0]

    mul_4 = next(iter(mul_3.users))
    if mul_4.target not in (operator.mul, torch.mul) or len(mul_4.args) < 2 or len(mul_4.users) != 1:
        return None
    unsqueeze_x = mul_4.args[0] if mul_4.args[1] is mul_3 else (mul_4.args[1] if mul_4.args[0] is mul_3 else None)
    if not _is_unsqueeze(unsqueeze_x) or len(unsqueeze_x.args) < 1:
        return None
    x_node = unsqueeze_x.args[0]

    ssm_state_new_candidates = set(mul_2.users) & set(mul_4.users)
    if len(ssm_state_new_candidates) != 1:
        return None
    ssm_state_new = next(iter(ssm_state_new_candidates))
    if ssm_state_new.target not in (operator.add, torch.add):
        return None

    # ssm_state_new legitimately has two mul-consumers: this timestep's
    # y-computation (ssm_state_new * C_ssm.unsqueeze(1)) *and*, when this
    # instance isn't the last in its window, the next timestep's
    # hA_next * ssm_state_new (structurally identical to this step's own
    # mul_2). Disambiguate by requiring the other operand to be an
    # unsqueeze -- the next-timestep one's other operand is an exp() output.
    mul_5_candidates = []
    for u in ssm_state_new.users:
        if u.target not in (operator.mul, torch.mul) or len(u.args) < 2:
            continue
        other = u.args[0] if u.args[1] is ssm_state_new else (u.args[1] if u.args[0] is ssm_state_new else None)
        if _is_unsqueeze(other):
            mul_5_candidates.append((u, other))
    if len(mul_5_candidates) != 1:
        return None
    mul_5, unsqueeze_c = mul_5_candidates[0]
    if len(mul_5.users) != 1 or len(unsqueeze_c.args) < 1 or len(unsqueeze_c.users) != 1:
        return None
    c_ssm_node = unsqueeze_c.args[0]

    sum_2 = next(iter(mul_5.users))
    if sum_2.op != "call_method" or sum_2.target != "sum" or len(sum_2.users) != 1:
        return None
    y_add = next(iter(sum_2.users))
    if y_add.target not in (operator.add, torch.add) or len(y_add.args) < 2:
        return None
    mul_6 = y_add.args[0] if y_add.args[1] is sum_2 else (y_add.args[1] if y_add.args[0] is sum_2 else None)
    if not isinstance(mul_6, torch.fx.Node) or mul_6.target not in (operator.mul, torch.mul) or len(mul_6.args) < 2:
        return None
    d0, d1 = mul_6.args[0], mul_6.args[1]
    if _param_like_name(d0) is not None and d1 is x_node:
        d_node = d0
    elif _param_like_name(d1) is not None and d0 is x_node:
        d_node = d1
    else:
        return None

    return dict(
        hA=hA_node, dt=dt_node, A=a_node, ssm_state_prev=ssm_state_prev,
        B_ssm=b_ssm_node, x=x_node, ssm_state_new=ssm_state_new,
        C_ssm=c_ssm_node, D=d_node, y=y_add,
    )


def _mamba_window_stack_inputs_legal(
    matches: List[Dict[str, torch.fx.Node]],
    window_start_timestep: int,
) -> bool:
    """Phase-0-style reachability legality check, transplanted to this
    rewriter's own stack inputs. Stacking x/dt/B_ssm/C_ssm ahead of a single
    window-spanning fused call is only valid because those four tensors are,
    in Mamba's canonical form, computable independently of the scan's own
    outputs within the window -- only ssm_state threads across timesteps,
    and that threading is handled *inside* the fused kernel via
    ssm_state_prev, not by the stack. If a future model variant fed y or
    ssm_state_new from one timestep in this window back into another
    timestep's x/dt/B/C (a feedback-type recurrence Mamba's canonical form
    doesn't have), stacking would silently compute on stale/wrong values
    since the whole stack is materialized before the fused call runs. This
    walks backward from each stack input, bounded exactly like
    _is_batching_source_legal in fx_spatial_batching.py: stop at
    placeholder/get_attr, or at any node already committed before this
    window (timestep < window_start_timestep, e.g. the incoming
    ssm_state_prev chain) -- and reject if that walk ever reaches this
    window's own y/ssm_state_new.
    """
    forbidden: Set[torch.fx.Node] = set()
    for m in matches:
        forbidden.add(m["y"])
        forbidden.add(m["ssm_state_new"])

    visited: Set[torch.fx.Node] = set()
    frontier: List[torch.fx.Node] = []
    for m in matches:
        frontier.extend([m["x"], m["dt"], m["B_ssm"], m["C_ssm"]])

    while frontier:
        node = frontier.pop()
        if node in visited:
            continue
        visited.add(node)
        if node in forbidden:
            return False
        if node.op in ("placeholder", "get_attr"):
            continue
        node_timestep = _get_chronos_meta(node, "timestep")
        if isinstance(node_timestep, int) and node_timestep < window_start_timestep:
            continue
        frontier.extend(node.all_input_nodes)
    return True


def rewrite_mamba_scan_to_fused(gm: torch.fx.GraphModule, window_size: int) -> int:
    """Groups matched per-timestep scan steps (see _match_mamba_scan_step) by
    (A parameter identity, window), and for every complete window of
    window_size consecutive instances, stacks their x/dt/B_ssm/C_ssm into
    [window,...] tensors, replaces the whole per-step chain with a single
    fused_temporal_selective_scan call spanning the window, and unstacks y
    back into per-t values -- this is what actually gets Phase B's kernel
    its ~48x microbenchmarked win (a single launch for the window instead of
    one tiny elementwise op sequence per timestep), unlike a one-shot
    per-instance swap. Requires annotate_temporal_metadata to have already
    run (uses chronos_timestep to order instances within a window; falls
    back to graph position order if unannotated).
    """
    order = {node: index for index, node in enumerate(gm.graph.nodes)}
    groups: Dict[Tuple[str, int], List[Tuple[int, Dict[str, torch.fx.Node]]]] = {}
    for node in list(gm.graph.nodes):
        if node.target is not torch.exp:
            continue
        match = _match_mamba_scan_step(node)
        if match is None:
            continue
        layer_key = _param_like_name(match["A"])
        if layer_key is None:
            continue
        timestep = _get_chronos_meta(node, "timestep")
        if not isinstance(timestep, int):
            timestep = order[node]
        window_id = timestep // window_size
        groups.setdefault((layer_key, window_id), []).append((timestep, match))

    replaced = 0
    # Process windows in per-layer chronological order (sorted by
    # (layer_key, window_id), not raw dict insertion order) so that when a
    # layer has multiple windows (T > window_size), each later window's
    # ssm_state_prev can be rewired to the *previous* window's fused
    # h_final output below -- otherwise it would keep pointing at the raw
    # pre-rewrite node, which after the previous window's
    # replace_all_uses_with(h_final) has 0 users and would normally be
    # dead-code-eliminated; re-using it here would instead resurrect that
    # entire unfused per-timestep chain alongside the new fused call,
    # silently doubling the compute for that window and leaving h_final
    # unused.
    state_override: Dict[str, torch.fx.Node] = {}
    for (layer_key, _window_id), items in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if len(items) != window_size:
            continue
        items = sorted(items, key=lambda pair: pair[0])
        matches = [m for _, m in items]
        if any(m["A"] is not matches[0]["A"] for m in matches):
            continue
        window_start_timestep = items[0][0]
        if not _mamba_window_stack_inputs_legal(matches, window_start_timestep):
            continue

        ssm_state_prev_node = state_override.get(layer_key, matches[0]["ssm_state_prev"])

        anchor = matches[-1]["hA"]
        with gm.graph.inserting_before(anchor):
            x_stack = gm.graph.call_function(torch.stack, args=([m["x"] for m in matches],), kwargs={"dim": 0})
            x_stack.name = f"{anchor.name}_x_stack"
            dt_stack = gm.graph.call_function(torch.stack, args=([m["dt"] for m in matches],), kwargs={"dim": 0})
            dt_stack.name = f"{anchor.name}_dt_stack"
            b_stack = gm.graph.call_function(torch.stack, args=([m["B_ssm"] for m in matches],), kwargs={"dim": 0})
            b_stack.name = f"{anchor.name}_b_stack"
            c_stack = gm.graph.call_function(torch.stack, args=([m["C_ssm"] for m in matches],), kwargs={"dim": 0})
            c_stack.name = f"{anchor.name}_c_stack"
            fused = gm.graph.call_function(
                torch.ops.snn_custom.fused_temporal_selective_scan,
                args=(x_stack, dt_stack, b_stack, c_stack, matches[0]["A"], matches[0]["D"], ssm_state_prev_node),
            )
            fused.name = f"{anchor.name}_fused_scan"
            y_seq = gm.graph.call_function(operator.getitem, args=(fused, 0))
            y_seq.name = f"{anchor.name}_y_seq"
            h_final = gm.graph.call_function(operator.getitem, args=(fused, 1))
            h_final.name = f"{anchor.name}_h_final"
            y_nodes = []
            for index in range(window_size):
                y_t = gm.graph.call_function(operator.getitem, args=(y_seq, index))
                y_t.name = f"{anchor.name}_y_t{index}"
                y_nodes.append(y_t)

        for m, y_t in zip(matches, y_nodes):
            m["y"].replace_all_uses_with(y_t)
        matches[-1]["ssm_state_new"].replace_all_uses_with(h_final)
        state_override[layer_key] = h_final
        replaced += len(matches)

    if replaced:
        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()
    return replaced
