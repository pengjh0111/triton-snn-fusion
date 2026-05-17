import operator
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import runtime.snn_custom_ops  # noqa: F401 - ensures custom op registration before matching/rewrite


FOLDED_ATTR_COUNTER = 0


@dataclass
class RewriteStats:
    direct_matches: int = 0
    conv_bn_matches: int = 0
    direct_replaced: int = 0
    conv_bn_replaced: int = 0
    lif_state_nodes: int = 0
    fused_state_nodes: int = 0


def _target_text(target) -> str:
    return str(target)


def is_aten_convolution_target(target) -> bool:
    return target is torch.ops.aten.convolution.default or str(target) == "aten.convolution.default"


def is_conv_node(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    if node.op == "call_module":
        try:
            return isinstance(gm.get_submodule(str(node.target)), nn.Conv2d)
        except AttributeError:
            return False
    if node.op == "call_function":
        return node.target in (F.conv2d, torch.conv2d) or is_aten_convolution_target(node.target)
    return False


def is_batch_norm_node(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    if node.op == "call_module":
        try:
            return isinstance(gm.get_submodule(str(node.target)), nn.BatchNorm2d)
        except AttributeError:
            return False
    return node.op == "call_function" and node.target is F.batch_norm


def batch_norm_training_arg(node: torch.fx.Node) -> Any:
    if node.op == "call_module":
        return False
    if "training" in node.kwargs:
        return node.kwargs["training"]
    if len(node.args) > 5:
        return node.args[5]
    return None


def is_batch_norm_inference_node(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    if not is_batch_norm_node(gm, node):
        return False
    training = batch_norm_training_arg(node)
    return training is False


def is_custom_lif_state_node(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and str(node.target) == "snn_custom.lif_forward_state.default"


def is_fused_conv_lif_state_node(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and str(node.target) == "snn_custom.fused_conv_lif_state.default"


def is_getitem_node(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and node.target is operator.getitem


def get_getitem_index(node: torch.fx.Node):
    if not is_getitem_node(node) or len(node.args) < 2:
        return None
    return node.args[1]


def find_tuple_getitems(tuple_node: torch.fx.Node) -> Dict[int, torch.fx.Node]:
    out: Dict[int, torch.fx.Node] = {}
    for user in tuple_node.users:
        if is_getitem_node(user):
            index = get_getitem_index(user)
            if isinstance(index, int):
                out[index] = user
    return out


def _lif_state_is_rewriteable(lif_node: torch.fx.Node) -> Tuple[bool, str]:
    getitems = find_tuple_getitems(lif_node)
    if 0 not in getitems:
        return False, "missing getitem[0] spike user"
    if 1 not in getitems:
        return False, "missing getitem[1] v_next user"
    non_getitem_users = [user.name for user in lif_node.users if not is_getitem_node(user)]
    if non_getitem_users:
        return False, f"has non-getitem users {non_getitem_users}"
    return True, ""


def _is_zeros_like_of(node: torch.fx.Node, producer: torch.fx.Node) -> bool:
    if node.op != "call_function" or len(node.args) < 1 or node.args[0] is not producer:
        return False
    return node.target is torch.zeros_like or "zeros_like" in str(node.target)


def _producer_has_only_lif_and_optional_zero_vinit(producer: torch.fx.Node, lif_node: torch.fx.Node) -> Tuple[bool, str]:
    for user in producer.users:
        if user is lif_node:
            continue
        if _is_zeros_like_of(user, producer) and len(lif_node.args) > 1 and lif_node.args[1] is user:
            continue
        return False, f"producer has unsupported user {user.name}"
    return True, ""


def match_conv_lif_state(gm: torch.fx.GraphModule) -> List[Tuple[torch.fx.Node, torch.fx.Node]]:
    matches = []
    for node in gm.graph.nodes:
        if not is_conv_node(gm, node):
            continue
        lif_candidates = [user for user in node.users if is_custom_lif_state_node(user)]
        if len(lif_candidates) != 1:
            continue
        lif_node = lif_candidates[0]
        ok, reason = _producer_has_only_lif_and_optional_zero_vinit(node, lif_node)
        if not ok:
            print(f"[SKIP] conv={node.name}: {reason}")
            continue
        ok, reason = _lif_state_is_rewriteable(lif_node)
        if not ok:
            print(f"[SKIP] lif_state={lif_node.name}: {reason}")
            continue
        print(f"[MATCH] conv -> lif_state found: conv={node.name}, lif={lif_node.name}")
        matches.append((node, lif_node))
    return matches


def match_conv_bn_lif_state(gm: torch.fx.GraphModule) -> List[Tuple[torch.fx.Node, torch.fx.Node, torch.fx.Node]]:
    matches = []
    for node in gm.graph.nodes:
        if not is_conv_node(gm, node):
            continue
        conv_users = list(node.users)
        if len(conv_users) != 1:
            continue
        bn_node = conv_users[0]
        if not is_batch_norm_node(gm, bn_node):
            continue
        if not is_batch_norm_inference_node(gm, bn_node):
            print(f"[SKIP] bn={bn_node.name}: batch_norm training flag is not statically False")
            continue
        lif_candidates = [user for user in bn_node.users if is_custom_lif_state_node(user)]
        if len(lif_candidates) != 1:
            continue
        lif_node = lif_candidates[0]
        ok, reason = _producer_has_only_lif_and_optional_zero_vinit(bn_node, lif_node)
        if not ok:
            print(f"[SKIP] bn={bn_node.name}: {reason}")
            continue
        ok, reason = _lif_state_is_rewriteable(lif_node)
        if not ok:
            print(f"[SKIP] lif_state={lif_node.name}: {reason}")
            continue
        print(
            "[MATCH] conv -> batch_norm -> lif_state found: "
            f"conv={node.name}, bn={bn_node.name}, lif={lif_node.name}"
        )
        matches.append((node, bn_node, lif_node))
    return matches


def get_attr_value(gm: torch.fx.GraphModule, target: str):
    value = gm
    for atom in target.split("."):
        value = getattr(value, atom)
    return value


def resolve_tensor_from_node(gm: torch.fx.GraphModule, node_or_value, placeholder_values: Dict[torch.fx.Node, Any]):
    if node_or_value is None:
        return None
    if isinstance(node_or_value, torch.Tensor):
        return node_or_value
    if isinstance(node_or_value, torch.fx.Node):
        if node_or_value.op == "get_attr":
            value = get_attr_value(gm, str(node_or_value.target))
            return value if isinstance(value, torch.Tensor) else None
        if node_or_value.op == "placeholder":
            value = placeholder_values.get(node_or_value)
            return value if isinstance(value, torch.Tensor) else None
    return None


def _insert_get_attr_before(gm: torch.fx.GraphModule, before: torch.fx.Node, target: str):
    with gm.graph.inserting_before(before):
        return gm.graph.get_attr(target)


def add_tensor_attr(gm: torch.fx.GraphModule, name_prefix: str, tensor: torch.Tensor) -> str:
    global FOLDED_ATTR_COUNTER
    safe_prefix = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name_prefix)
    while True:
        attr_name = f"{safe_prefix}_{FOLDED_ATTR_COUNTER}"
        FOLDED_ATTR_COUNTER += 1
        if not hasattr(gm, attr_name):
            break
    gm.register_buffer(attr_name, tensor.detach().clone())
    return attr_name


def _normalize_pair(value, default) -> List[int]:
    if value is None:
        return list(default)
    if isinstance(value, int):
        return [value, value]
    return list(value)


def _parse_conv_call_args(conv_node: torch.fx.Node):
    args = list(conv_node.args)
    if len(args) < 2:
        raise ValueError("conv call has too few args")
    conv_input = args[0]
    weight = args[1]
    bias = args[2] if len(args) > 2 else None
    stride = _normalize_pair(args[3] if len(args) > 3 else None, (1, 1))
    padding = _normalize_pair(args[4] if len(args) > 4 else None, (0, 0))
    dilation = _normalize_pair(args[5] if len(args) > 5 else None, (1, 1))
    groups = int(args[6]) if len(args) > 6 else 1
    if is_aten_convolution_target(conv_node.target):
        if len(args) < 9:
            raise ValueError("aten.convolution call has too few args")
        transposed = bool(args[6])
        if transposed:
            raise ValueError("transposed aten.convolution is not supported")
        groups = int(args[8])
    return conv_input, weight, bias, stride, padding, dilation, groups


def extract_conv2d_tensors(gm: torch.fx.GraphModule, conv_node: torch.fx.Node, placeholder_values):
    if conv_node.op == "call_module":
        conv = gm.get_submodule(str(conv_node.target))
        if not isinstance(conv, nn.Conv2d):
            raise ValueError("call_module target is not nn.Conv2d")
        return (
            conv_node.args[0],
            conv.weight,
            conv.bias,
            list(conv.stride),
            list(conv.padding),
            list(conv.dilation),
            int(conv.groups),
        )

    if conv_node.op == "call_function" and (
        conv_node.target in (F.conv2d, torch.conv2d) or is_aten_convolution_target(conv_node.target)
    ):
        conv_input, weight_arg, bias_arg, stride, padding, dilation, groups = _parse_conv_call_args(conv_node)
        weight = resolve_tensor_from_node(gm, weight_arg, placeholder_values)
        if weight is None:
            raise ValueError("conv weight is placeholder/get_attr that cannot be resolved")
        bias = resolve_tensor_from_node(gm, bias_arg, placeholder_values)
        if bias_arg is not None and bias is None:
            raise ValueError("conv bias cannot be resolved")
        return conv_input, weight, bias, stride, padding, dilation, groups

    raise ValueError(f"unsupported conv target={conv_node.target}")


def extract_conv2d_graph_args(gm: torch.fx.GraphModule, conv_node: torch.fx.Node, before: torch.fx.Node, placeholder_values):
    if conv_node.op == "call_module":
        conv = gm.get_submodule(str(conv_node.target))
        if not isinstance(conv, nn.Conv2d):
            raise ValueError("call_module target is not nn.Conv2d")
        conv_input = conv_node.args[0]
        weight_node = _insert_get_attr_before(gm, before, f"{conv_node.target}.weight")
        if conv.bias is not None:
            bias_node = _insert_get_attr_before(gm, before, f"{conv_node.target}.bias")
        else:
            bias_attr = add_tensor_attr(gm, "_fx_zero_conv_bias", torch.zeros(conv.out_channels, device=conv.weight.device, dtype=conv.weight.dtype))
            bias_node = _insert_get_attr_before(gm, before, bias_attr)
        return conv_input, weight_node, bias_node, list(conv.stride), list(conv.padding), list(conv.dilation), int(conv.groups)

    if conv_node.op == "call_function" and (
        conv_node.target in (F.conv2d, torch.conv2d) or is_aten_convolution_target(conv_node.target)
    ):
        conv_input, weight_node, bias_node, stride, padding, dilation, groups = _parse_conv_call_args(conv_node)
        if bias_node is None:
            weight = resolve_tensor_from_node(gm, weight_node, placeholder_values)
            if weight is None:
                raise ValueError("cannot create zero bias without resolving conv weight")
            bias_attr = add_tensor_attr(gm, "_fx_zero_conv_bias", torch.zeros(weight.shape[0], device=weight.device, dtype=weight.dtype))
            bias_node = _insert_get_attr_before(gm, before, bias_attr)
        return conv_input, weight_node, bias_node, stride, padding, dilation, groups

    raise ValueError(f"unsupported conv target={conv_node.target}")


def extract_batch_norm_params(gm: torch.fx.GraphModule, bn_node: torch.fx.Node, placeholder_values):
    if bn_node.op == "call_module":
        bn = gm.get_submodule(str(bn_node.target))
        if not isinstance(bn, nn.BatchNorm2d):
            raise ValueError("call_module target is not nn.BatchNorm2d")
        if bn.training:
            raise ValueError("BatchNorm2d module is in training mode")
        return bn.running_mean, bn.running_var, bn.weight, bn.bias, False, float(bn.eps)

    if bn_node.op == "call_function" and bn_node.target is F.batch_norm:
        args = list(bn_node.args)
        running_mean_arg = args[1] if len(args) > 1 else bn_node.kwargs.get("running_mean")
        running_var_arg = args[2] if len(args) > 2 else bn_node.kwargs.get("running_var")
        weight_arg = args[3] if len(args) > 3 else bn_node.kwargs.get("weight")
        bias_arg = args[4] if len(args) > 4 else bn_node.kwargs.get("bias")
        training = args[5] if len(args) > 5 else bn_node.kwargs.get("training", False)
        eps = args[7] if len(args) > 7 else bn_node.kwargs.get("eps", 1e-5)
        if training is not False:
            raise ValueError(f"batch_norm training must be False, got {training}")
        running_mean = resolve_tensor_from_node(gm, running_mean_arg, placeholder_values)
        running_var = resolve_tensor_from_node(gm, running_var_arg, placeholder_values)
        bn_weight = resolve_tensor_from_node(gm, weight_arg, placeholder_values)
        bn_bias = resolve_tensor_from_node(gm, bias_arg, placeholder_values)
        if running_mean is None or running_var is None:
            raise ValueError("running_mean/running_var cannot be resolved")
        return running_mean, running_var, bn_weight, bn_bias, False, float(eps)

    raise ValueError(f"unsupported batch_norm target={bn_node.target}")


def fold_bn_into_conv_params(conv_weight, conv_bias, running_mean, running_var, bn_weight, bn_bias, eps):
    if conv_bias is None:
        conv_bias = torch.zeros_like(running_mean)
    if bn_weight is None:
        bn_weight = torch.ones_like(running_mean)
    if bn_bias is None:
        bn_bias = torch.zeros_like(running_mean)

    conv_weight = conv_weight.to(dtype=running_mean.dtype)
    conv_bias = conv_bias.to(dtype=running_mean.dtype)
    scale = bn_weight / torch.sqrt(running_var + float(eps))
    folded_weight = conv_weight * scale.reshape([-1, 1, 1, 1])
    folded_bias = (conv_bias - running_mean) * scale + bn_bias
    return folded_weight, folded_bias


def _conv2d_output_shape_from_tensors(x: torch.Tensor, weight: torch.Tensor, stride, padding, dilation):
    batch = x.shape[0]
    out_channels = weight.shape[0]
    height = x.shape[2]
    width = x.shape[3]
    kernel_h = weight.shape[2]
    kernel_w = weight.shape[3]
    out_h = (height + 2 * padding[0] - dilation[0] * (kernel_h - 1) - 1) // stride[0] + 1
    out_w = (width + 2 * padding[1] - dilation[1] * (kernel_w - 1) - 1) // stride[1] + 1
    return batch, out_channels, out_h, out_w


def _maybe_materialize_zero_v_prev(
    gm: torch.fx.GraphModule,
    lif_node: torch.fx.Node,
    producer: torch.fx.Node,
    conv_input,
    conv_weight: torch.Tensor,
    stride,
    padding,
    dilation,
    placeholder_values,
):
    v_prev = lif_node.args[1]
    if not (isinstance(v_prev, torch.fx.Node) and _is_zeros_like_of(v_prev, producer)):
        return v_prev, None

    x_value = resolve_tensor_from_node(gm, conv_input, placeholder_values)
    if x_value is None:
        print(f"[SKIP] lif_state={lif_node.name}: cannot materialize zero v_prev without concrete conv input")
        return v_prev, None

    out_shape = _conv2d_output_shape_from_tensors(x_value, conv_weight, stride, padding, dilation)
    zero_v = torch.zeros(out_shape, device=x_value.device, dtype=x_value.dtype)
    attr = add_tensor_attr(gm, "_fx_zero_v_prev", zero_v)
    return _insert_get_attr_before(gm, lif_node, attr), v_prev


def _replace_lif_state_uses_with_fused(gm: torch.fx.GraphModule, lif_node: torch.fx.Node, fused_tuple: torch.fx.Node):
    getitems = find_tuple_getitems(lif_node)
    if 0 not in getitems or 1 not in getitems:
        raise ValueError("lif_state must have both getitem[0] and getitem[1]")
    old_spike = getitems[0]
    old_v_next = getitems[1]

    with gm.graph.inserting_before(old_spike):
        fused_spike = gm.graph.call_function(operator.getitem, args=(fused_tuple, 0))
        fused_spike.name = f"{fused_tuple.name}_spike"
    with gm.graph.inserting_before(old_v_next):
        fused_v_next = gm.graph.call_function(operator.getitem, args=(fused_tuple, 1))
        fused_v_next.name = f"{fused_tuple.name}_v_next"

    old_spike.replace_all_uses_with(fused_spike)
    old_v_next.replace_all_uses_with(fused_v_next)

    if len(old_spike.users) == 0:
        gm.graph.erase_node(old_spike)
    if len(old_v_next.users) == 0:
        gm.graph.erase_node(old_v_next)
    if len(lif_node.users) == 0:
        gm.graph.erase_node(lif_node)


def rewrite_conv_lif_state_to_fused(gm: torch.fx.GraphModule, matches, placeholder_values, max_patterns: int) -> int:
    replaced = 0
    for conv_node, lif_node in matches:
        if replaced >= max_patterns:
            print(f"[SKIP] max-patterns reached for conv={conv_node.name}, lif={lif_node.name}")
            continue
        try:
            if len(lif_node.args) < 6 or lif_node.args[0] is not conv_node:
                print(f"[SKIP] lif_state={lif_node.name}: unexpected lif args")
                continue
            v_prev, v_threshold, v_reset, tau, detach_reset = lif_node.args[1:6]
            tensor_conv_input, conv_weight, _conv_bias, tensor_stride, tensor_padding, tensor_dilation, _tensor_groups = (
                extract_conv2d_tensors(gm, conv_node, placeholder_values)
            )
            conv_input, weight_node, bias_node, stride, padding, dilation, groups = extract_conv2d_graph_args(
                gm, conv_node, lif_node, placeholder_values
            )
            v_prev, zero_v_node = _maybe_materialize_zero_v_prev(
                gm,
                lif_node,
                conv_node,
                tensor_conv_input,
                conv_weight,
                tensor_stride,
                tensor_padding,
                tensor_dilation,
                placeholder_values,
            )
            with gm.graph.inserting_before(lif_node):
                fused_tuple = gm.graph.call_function(
                    torch.ops.snn_custom.fused_conv_lif_state.default,
                    args=(
                        conv_input,
                        weight_node,
                        bias_node,
                        v_prev,
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
                fused_tuple.name = f"{conv_node.name}_fused_conv_lif_state"
            _replace_lif_state_uses_with_fused(gm, lif_node, fused_tuple)
            if zero_v_node is not None and len(zero_v_node.users) == 0:
                gm.graph.erase_node(zero_v_node)
            if len(conv_node.users) == 0:
                gm.graph.erase_node(conv_node)
            replaced += 1
            print("[REWRITE] replaced conv -> lif_state with fused_conv_lif_state")
            print(f"          conv={conv_node.name}, fused={fused_tuple.name}")
        except Exception as exc:
            print(f"[SKIP] rewrite failed for conv={conv_node.name}, lif={lif_node.name}: {exc}")
            traceback.print_exc()
    gm.graph.lint()
    gm.recompile()
    return replaced


def rewrite_conv_bn_lif_state_to_fused(gm: torch.fx.GraphModule, matches, placeholder_values, max_patterns: int) -> int:
    replaced = 0
    for conv_node, bn_node, lif_node in matches:
        if replaced >= max_patterns:
            print(f"[SKIP] max-patterns reached for conv={conv_node.name}, bn={bn_node.name}, lif={lif_node.name}")
            continue
        try:
            if len(lif_node.args) < 6 or lif_node.args[0] is not bn_node:
                print(f"[SKIP] lif_state={lif_node.name}: unexpected lif args")
                continue
            conv_input, conv_weight, conv_bias, stride, padding, dilation, groups = extract_conv2d_tensors(
                gm, conv_node, placeholder_values
            )
            running_mean, running_var, bn_weight, bn_bias, training, eps = extract_batch_norm_params(
                gm, bn_node, placeholder_values
            )
            if training is not False:
                print(f"[SKIP] bn={bn_node.name}: training is not False")
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
            v_prev, v_threshold, v_reset, tau, detach_reset = lif_node.args[1:6]
            v_prev, zero_v_node = _maybe_materialize_zero_v_prev(
                gm,
                lif_node,
                bn_node,
                conv_input,
                folded_weight,
                stride,
                padding,
                dilation,
                placeholder_values,
            )
            weight_attr = add_tensor_attr(gm, "_fx_folded_conv_bn_weight", folded_weight)
            bias_attr = add_tensor_attr(gm, "_fx_folded_conv_bn_bias", folded_bias)
            weight_node = _insert_get_attr_before(gm, lif_node, weight_attr)
            bias_node = _insert_get_attr_before(gm, lif_node, bias_attr)
            with gm.graph.inserting_before(lif_node):
                fused_tuple = gm.graph.call_function(
                    torch.ops.snn_custom.fused_conv_lif_state.default,
                    args=(
                        conv_input,
                        weight_node,
                        bias_node,
                        v_prev,
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
                fused_tuple.name = f"{conv_node.name}_bn_fused_conv_lif_state"
            _replace_lif_state_uses_with_fused(gm, lif_node, fused_tuple)
            if zero_v_node is not None and len(zero_v_node.users) == 0:
                gm.graph.erase_node(zero_v_node)
            if len(bn_node.users) == 0:
                gm.graph.erase_node(bn_node)
            if len(conv_node.users) == 0:
                gm.graph.erase_node(conv_node)
            replaced += 1
            print("[REWRITE] replaced conv -> batch_norm -> lif_state with fused_conv_lif_state")
            print(f"          conv={conv_node.name}, bn={bn_node.name}, fused={fused_tuple.name}")
        except Exception as exc:
            print(f"[SKIP] rewrite failed for conv={conv_node.name}, bn={bn_node.name}, lif={lif_node.name}: {exc}")
            traceback.print_exc()
    gm.graph.lint()
    gm.recompile()
    return replaced


def count_lif_state_nodes(gm: torch.fx.GraphModule) -> int:
    return sum(1 for node in gm.graph.nodes if is_custom_lif_state_node(node))


def count_fused_conv_lif_state_nodes(gm: torch.fx.GraphModule) -> int:
    return sum(1 for node in gm.graph.nodes if is_fused_conv_lif_state_node(node))
