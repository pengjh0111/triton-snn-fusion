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
    # "Mixed" dual-operand resolution (add/mul only): set when exactly one of
    # the two tensor operands is stack/chain-sourced (temporal_stack_inputs or
    # previous_batched_inputs has length 1, not 2) and the OTHER operand is a
    # plain, legally-per-timestep value (passed _is_batching_source_legal but
    # isn't itself a recognized stack/chain source) -- e.g. Mamba's
    # `y * silu(z)`, where y comes from the fused scan's per-window stack and
    # z is freshly computed each timestep from this same layer's own in_proj.
    # mixed_plain_operand_index is which of node.args[0]/args[1] is the plain
    # side (0 or 1); mixed_plain_input is that operand's node for THIS
    # candidate specifically (grouped across the window and cat'd at rewrite
    # time -- see _mixed_operand_inputs_for_group).
    mixed_plain_operand_index: Optional[int] = None
    mixed_plain_input: Optional[torch.fx.Node] = None


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
    spatial_batched_layer_norm: int = 0
    spatial_batched_attention: int = 0
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
    spatial_mixed_operand_groups: int = 0
    spatial_batched_mul: int = 0
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


def _get_kairos_meta(node: torch.fx.Node, key: str, default=None):
    meta_key = f"kairos_{key}"
    if meta_key in node.meta:
        return node.meta[meta_key]
    return getattr(node, f"_kairos_{key}", default)


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


def _is_fused_temporal_stack_source(node: torch.fx.Node) -> bool:
    """True iff node is the SNN-native fused-temporal-op getitem pattern
    _match_temporal_stack_getitem produces stack sources from: node itself
    is spike_stack = getitem(temporal_tuple, 0), so the fused_temporal_*
    call being tested for is node.args[0] (temporal_tuple), *not* node
    itself -- node.op is always "call_function" targeting
    operator.getitem, never the fused op directly (confirmed by a real
    MobileNetV2 regression this exact confusion caused: checking node
    itself made this always return False for genuine SNN stack sources
    too, routing them through the window-slice path meant only for
    _match_external_sequence_getitem's sources, and narrowing an
    already-exactly-window-sized tensor at a nonzero window offset went
    out of bounds).

    Used at rewrite time to decide whether a temporal-stack source needs
    the window-slice step _match_external_sequence_getitem's sources
    require -- see that function's docstring for why the two cases can't
    share the same flatten path unchanged. By construction the two
    matchers' outputs are disjoint (_match_temporal_stack_getitem only
    returns non-None when this predicate holds; _match_external_sequence_getitem
    only matches placeholders, never a fused_temporal_* getitem), so this
    check safely tells them apart without adding any new fields to
    TemporalStackInput.
    """
    if node.op != "call_function" or not node.args or not isinstance(node.args[0], torch.fx.Node):
        return False
    temporal_tuple = node.args[0]
    if temporal_tuple.op != "call_function":
        return False
    target_text = _target_text(temporal_tuple.target)
    return "snn_custom.fused_temporal_" in target_text or "snn_custom::fused_temporal_" in target_text


def _match_external_sequence_getitem(node: torch.fx.Node, temporal_window: int) -> Optional[TemporalStackInput]:
    """Recognizes x_t = x_seq[t] -- a single getitem indexing directly into
    the model's own top-level input placeholder (a genuine whole-sequence
    [T, ...] tensor for a SequenceInputLoopWrapper-style model, e.g.
    ConvLSTM/Mamba) -- the Kairos sequence-input workloads' analogue of
    _match_temporal_stack_getitem's SNN-specific double-getitem-from-a-
    fused-op pattern, letting the batched-group rewrite reference the
    original whole-sequence tensor directly instead of needing every
    per-t getitem "available before" a single insertion point (the whole
    point of the "stack shortcut": t=1's x_seq[1] is already sitting right
    next to t=0's x_seq[0] in one pre-existing tensor -- there's nothing to
    wait for or re-stack).

    Deliberately restricted to source.op == "placeholder" -- a real
    function argument, unambiguously never anything else -- rather than
    the broader "no timestep annotation" heuristic tried first. That
    broader version produced two confirmed false positives on existing SNN
    models (both lack _kairos_meta timestep annotation for the same
    reason this case does -- they're materialized mid-pass, before
    annotate_temporal_metadata ever saw them): a prior spatial-batching
    iteration's own batched intermediate, and a `new_empty`-allocated
    per-timestep output buffer inside the existing fused_linear_lif_state
    rewrite (name pattern *_temporal_fused_linear_lif_state_spike_stack).
    Blacklisting each pattern as discovered doesn't bound the search space
    with any confidence; placeholder-only does, at the cost of not
    covering KairosDeepSpeech2's "frontend output" case (a computed node,
    not a placeholder) -- a real, scoped gap, not a silent one, and safe
    to leave for a follow-up since DeepSpeech2's win is dominated by the
    GRU cell rewrite itself, not this batching layer.

    Unlike the SNN case, this source's own dim0 is the *full* T, not just
    one window's worth (a fused_temporal_* op only ever produces one
    window's stack at a time, so its getitem-0 output is already exactly
    window-sized) -- rewrite_spatial_batch_group's temporal-stack branch
    must narrow to [window_start:window_start+window] before flattening,
    or the flatten merges the wrong element count into the batch dim
    (confirmed via a real trace: window=4 out of T=16 produced a
    4x-oversized flatten and a downstream shape-mismatch crash). See
    _is_fused_temporal_stack_source, which is how the rewrite step tells
    this case apart from the SNN one to know whether to narrow first.

    TemporalStackInput.timestep is window-*relative* (0..window-1), not the
    getitem's raw absolute index into the full T-length source -- the one
    place that field is consumed (group_spatial_batch_candidates's
    "stack_timesteps != list(range(temporal_window))" completeness check)
    hardcodes a 0-based-per-window expectation, which is trivially true for
    the SNN case (a fused_temporal_* op's own stack output only ever holds
    one window's worth of data, so its raw getitem indices are already
    0..window-1) but not for this whole-sequence case, where raw indices
    range over the full T (confirmed via a real trace: only window_id=0
    happened to pass this check by coincidence, since 0..window-1 vs
    0..window-1 only lines up for the very first window).
    """
    absolute_timestep = _getitem_index(node)
    if absolute_timestep is None or not node.args or not isinstance(node.args[0], torch.fx.Node):
        return None
    source = node.args[0]
    if source.op != "placeholder":
        return None
    return TemporalStackInput(
        temporal_tuple=source,
        spike_stack=source,
        getitem_node=node,
        timestep=absolute_timestep % temporal_window if temporal_window > 0 else absolute_timestep,
        source_op="external_sequence_input",
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


@dataclass
class TransformerAttentionItem:
    output_node: torch.fx.Node
    input_node: torch.fx.Node
    layer_norm_node: torch.fx.Node
    qkv_linear_node: torch.fx.Node
    qkv_reshape_node: torch.fx.Node
    qkv_movedim_node: torch.fx.Node
    qkv_transpose_node: torch.fx.Node
    q_node: torch.fx.Node
    k_node: torch.fx.Node
    v_node: torch.fx.Node
    k_transpose_node: torch.fx.Node
    qk_matmul_node: torch.fx.Node
    scale_node: torch.fx.Node
    softmax_node: torch.fx.Node
    av_matmul_node: torch.fx.Node
    out_transpose_node: torch.fx.Node
    timestep: int
    source_kind: str
    source_node: torch.fx.Node
    normalized_shape: Any
    norm_weight: Any
    norm_bias: Any
    norm_eps: Any
    qkv_weight: Any
    qkv_bias: Any
    reshape_args: Tuple[Any, ...]
    scale: Any
    softmax_dim: Any
    output_shape_args: Tuple[Any, ...]


@dataclass
class LayerNormStackItem:
    output_node: torch.fx.Node
    input_node: torch.fx.Node
    timestep: int
    source_kind: str
    source_node: torch.fx.Node
    normalized_shape: Any
    norm_weight: Any
    norm_bias: Any
    norm_eps: Any


def _call_method_target(node: torch.fx.Node, name: str) -> bool:
    return node.op == "call_method" and str(node.target) == name


def _single_tensor_arg(node: torch.fx.Node) -> Optional[torch.fx.Node]:
    if not node.args or not isinstance(node.args[0], torch.fx.Node):
        return None
    return node.args[0]


def _stack_items(node: torch.fx.Node) -> Optional[Tuple[List[torch.fx.Node], int]]:
    if not _is_stack_node(node) or not node.args:
        return None
    values = node.args[0]
    if not isinstance(values, (list, tuple)) or not values:
        return None
    if not all(isinstance(item, torch.fx.Node) for item in values):
        return None
    dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
    try:
        dim = int(dim)
    except Exception:
        return None
    if dim != 0:
        return None
    return list(values), dim


def _node_temporal_source(node: torch.fx.Node) -> Optional[Tuple[str, torch.fx.Node, int]]:
    temporal_stack = _match_temporal_stack_getitem(node)
    if temporal_stack is not None:
        return ("temporal_stack", temporal_stack.spike_stack, temporal_stack.timestep)
    previous = _match_batched_chunk_getitem(node)
    if previous is not None:
        return ("previous_batched", previous.batched_node, previous.timestep)
    timestep = _get_kairos_meta(node, "timestep")
    if isinstance(timestep, int):
        return ("plain", node, timestep)
    return None


def _layer_norm_args(node: torch.fx.Node) -> Optional[Tuple[Any, Any, Any, Any]]:
    if not _is_layer_norm_node(node) or len(node.args) < 2:
        return None
    normalized_shape = node.args[1]
    weight = node.args[2] if len(node.args) > 2 else node.kwargs.get("weight", None)
    bias = node.args[3] if len(node.args) > 3 else node.kwargs.get("bias", None)
    eps = node.args[4] if len(node.args) > 4 else node.kwargs.get("eps", 1e-5)
    return normalized_shape, weight, bias, eps


def _match_layer_norm_stack_item(node: torch.fx.Node) -> Optional[LayerNormStackItem]:
    args = _layer_norm_args(node)
    if args is None:
        return None
    input_node = _single_tensor_arg(node)
    if input_node is None:
        return None
    source = _node_temporal_source(input_node)
    if source is None:
        return None
    source_kind, source_node, timestep = source
    normalized_shape, weight, bias, eps = args
    return LayerNormStackItem(
        output_node=node,
        input_node=input_node,
        timestep=timestep,
        source_kind=source_kind,
        source_node=source_node,
        normalized_shape=normalized_shape,
        norm_weight=weight,
        norm_bias=bias,
        norm_eps=eps,
    )


def _linear_args(node: torch.fx.Node) -> Optional[Tuple[torch.fx.Node, Any, Any]]:
    if not _is_linear_node(None, node):  # gm is not needed for call_function nodes.
        return None
    if len(node.args) < 2 or not isinstance(node.args[0], torch.fx.Node):
        return None
    weight = node.args[1]
    bias = node.args[2] if len(node.args) > 2 else node.kwargs.get("bias", None)
    return node.args[0], weight, bias


def _match_transformer_attention_output(node: torch.fx.Node) -> Optional[TransformerAttentionItem]:
    if not _call_method_target(node, "reshape"):
        return None
    if len(node.args) < 2:
        return None
    output_shape_args = tuple(node.args[1:])
    out_transpose = _single_tensor_arg(node)
    if out_transpose is None or not _call_method_target(out_transpose, "transpose"):
        return None
    if tuple(out_transpose.args[1:3]) != (-3, -2):
        return None
    av_matmul = _single_tensor_arg(out_transpose)
    if av_matmul is None or not _is_matmul_node(av_matmul) or len(av_matmul.args) < 2:
        return None
    softmax_node, v_node = av_matmul.args[:2]
    if not isinstance(softmax_node, torch.fx.Node) or not isinstance(v_node, torch.fx.Node):
        return None
    if not _is_softmax_node(softmax_node):
        return None
    softmax_dim = softmax_node.kwargs.get("dim", None)
    if softmax_dim is None and len(softmax_node.args) > 1:
        softmax_dim = softmax_node.args[1]
    scale_node = _single_tensor_arg(softmax_node)
    if scale_node is None or scale_node.op != "call_function" or scale_node.target is not operator.mul:
        return None
    if len(scale_node.args) != 2:
        return None
    qk_matmul = None
    scale = None
    if isinstance(scale_node.args[0], torch.fx.Node):
        qk_matmul = scale_node.args[0]
        scale = scale_node.args[1]
    elif isinstance(scale_node.args[1], torch.fx.Node):
        qk_matmul = scale_node.args[1]
        scale = scale_node.args[0]
    if qk_matmul is None or not _is_matmul_node(qk_matmul) or len(qk_matmul.args) < 2:
        return None
    q_node, k_transpose = qk_matmul.args[:2]
    if not isinstance(q_node, torch.fx.Node) or not isinstance(k_transpose, torch.fx.Node):
        return None
    if not _call_method_target(k_transpose, "transpose") or tuple(k_transpose.args[1:3]) != (-2, -1):
        return None
    k_node = _single_tensor_arg(k_transpose)
    if k_node is None:
        return None

    def _qkv_getitem(item: torch.fx.Node, expected_index: int) -> Optional[torch.fx.Node]:
        if item.op != "call_function" or item.target is not operator.getitem or len(item.args) < 2:
            return None
        if item.args[1] != expected_index or not isinstance(item.args[0], torch.fx.Node):
            return None
        return item.args[0]

    qkv_transpose = _qkv_getitem(q_node, 0)
    if qkv_transpose is None or _qkv_getitem(k_node, 1) is not qkv_transpose or _qkv_getitem(v_node, 2) is not qkv_transpose:
        return None
    if not _call_method_target(qkv_transpose, "transpose") or tuple(qkv_transpose.args[1:3]) != (-3, -2):
        return None
    movedim = _single_tensor_arg(qkv_transpose)
    if movedim is None or not _call_method_target(movedim, "movedim") or tuple(movedim.args[1:3]) != (-3, 0):
        return None
    qkv_reshape = _single_tensor_arg(movedim)
    if qkv_reshape is None or not _call_method_target(qkv_reshape, "reshape") or len(qkv_reshape.args) < 2:
        return None
    reshape_args = tuple(qkv_reshape.args[1:])
    qkv_linear = _single_tensor_arg(qkv_reshape)
    if qkv_linear is None:
        return None
    linear = _linear_args(qkv_linear)
    if linear is None:
        return None
    layer_norm_node, qkv_weight, qkv_bias = linear
    ln_args = _layer_norm_args(layer_norm_node)
    if ln_args is None:
        return None
    input_node = _single_tensor_arg(layer_norm_node)
    if input_node is None:
        return None
    source = _node_temporal_source(input_node)
    if source is None:
        return None
    source_kind, source_node, timestep = source
    normalized_shape, norm_weight, norm_bias, norm_eps = ln_args
    return TransformerAttentionItem(
        output_node=node,
        input_node=input_node,
        layer_norm_node=layer_norm_node,
        qkv_linear_node=qkv_linear,
        qkv_reshape_node=qkv_reshape,
        qkv_movedim_node=movedim,
        qkv_transpose_node=qkv_transpose,
        q_node=q_node,
        k_node=k_node,
        v_node=v_node,
        k_transpose_node=k_transpose,
        qk_matmul_node=qk_matmul,
        scale_node=scale_node,
        softmax_node=softmax_node,
        av_matmul_node=av_matmul,
        out_transpose_node=out_transpose,
        timestep=timestep,
        source_kind=source_kind,
        source_node=source_node,
        normalized_shape=normalized_shape,
        norm_weight=norm_weight,
        norm_bias=norm_bias,
        norm_eps=norm_eps,
        qkv_weight=qkv_weight,
        qkv_bias=qkv_bias,
        reshape_args=reshape_args,
        scale=scale,
        softmax_dim=softmax_dim,
        output_shape_args=output_shape_args,
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


def _is_layer_norm_node(node: torch.fx.Node) -> bool:
    if node.op != "call_function":
        return False
    text = _target_text(node.target)
    return node.target is F.layer_norm or "layer_norm" in text


def _is_matmul_node(node: torch.fx.Node) -> bool:
    if node.op != "call_function":
        return False
    text = _target_text(node.target)
    return node.target is torch.matmul or "matmul" in text


def _is_softmax_node(node: torch.fx.Node) -> bool:
    if node.op != "call_function":
        return False
    text = _target_text(node.target)
    return node.target is torch.softmax or "softmax" in text


def _is_stack_node(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and node.target is torch.stack


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


def _is_mul_node(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and (
        node.target in (operator.mul, torch.mul) or "aten.mul" in _target_text(node.target)
    )


# "add" and "mul" are both plain elementwise binary ops where BOTH operands
# vary per timestep (unlike conv/bn/linear/layer_norm, where only the single
# tensor input varies and weight/bias/eps are timestep-invariant params) --
# they share every candidate-extraction/grouping/rewrite code path below
# (_extract_candidate's dual-tensor-input handling, group_spatial_batch_candidates'
# add-specific grouping keys, rewrite_spatial_batch_group/_make_batched_call_with_inputs's
# "add" branch), so DUAL_OPERAND_KINDS is the single switch controlling which
# kinds take that path instead of the single-input path every other kind uses.
DUAL_OPERAND_KINDS = ("add", "mul")


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
        text = _target_text(node.target)
        return node.target is torch.flatten or ("flatten" in text and "unflatten" not in text)
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
    if kind in DUAL_OPERAND_KINDS:
        args = tuple("<tensor>" if isinstance(arg, torch.fx.Node) else _node_ref(arg) for arg in node.args)
        kwargs = tuple(sorted((key, _node_ref(value)) for key, value in node.kwargs.items()))
        return (node.op, kind, args, kwargs)
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
    if "mul" in enabled and _is_mul_node(node):
        return "mul"
    if "layer_norm" in enabled and _is_layer_norm_node(node):
        return "layer_norm"
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
    if _is_add_node(node) or _is_mul_node(node):
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
    if kind in DUAL_OPERAND_KINDS:
        return tuple(arg for arg in node.args[:2] if isinstance(arg, torch.fx.Node))
    input_node = _candidate_input(node)
    return (input_node,) if input_node is not None else ()


def _is_generated_spatial_batching_node(node: torch.fx.Node) -> bool:
    if node.meta.get("kairos_origin") == "temporal_stack_flatten":
        return True
    name = str(node.name)
    return (
        "_spatial_batch_" in name
        or "_temporal_stack_flatten" in name
        or name.endswith("_spatial_batch_cat")
        or name.endswith("_chunks")
    )


_BATCH_LEGALITY_MAX_VISITED = 512


def _is_batching_source_legal(source_node: torch.fx.Node, candidate_timestep: int) -> bool:
    """Reject batching a candidate op across its t-instances if one of its
    tensor inputs is not actually independent per timestep, i.e. it
    transitively depends on a node produced by a *different* timestep's
    block without going through a recognized "safe" boundary first.

    This is the general form of the bug feedback networks (ConvLSTM/GRU/
    Mamba-style W_h-side conv or linear consuming h_prev/c_prev/ssm_state)
    trigger: existing SNN models never hit this path because every
    conv/bn/pool input is always either the externally supplied per-t stack
    (_match_temporal_stack_getitem), an already-batched-and-rechunked prior
    layer's output (_match_batched_chunk_getitem), or a value produced
    within the SAME timestep's block -- membrane-state recurrence in SNNs is
    always hidden behind an atomic snn_custom.* op boundary
    (_is_stateful_or_fused_snn_node), which this walk treats as opaque and
    does not descend into (its output is validated as same-timestep by
    construction, so what it does internally with cross-timestep state is
    irrelevant to callers).

    Bounded BFS: get_attr/placeholder (params/buffers, timestep-invariant),
    a temporal_stack_getitem match, a previous_batched_chunk_getitem match,
    or an snn_custom.*/lif op boundary all stop the walk on that branch as
    "safe". A node whose own kairos_timestep annotation differs from
    candidate_timestep proves a genuine cross-iteration dependency and
    fails the whole check immediately. Nodes at the same timestep (or
    unannotated, e.g. plain constants) are transparently walked through.
    """
    visited = set()
    frontier = [source_node]
    visited_count = 0
    while frontier:
        node = frontier.pop()
        if node in visited:
            continue
        visited.add(node)
        visited_count += 1
        if visited_count > _BATCH_LEGALITY_MAX_VISITED:
            # Could not prove safety within the search budget -- conservatively reject.
            return False
        if node.op in ("get_attr", "placeholder"):
            continue
        if _match_temporal_stack_getitem(node) is not None:
            continue
        if _match_batched_chunk_getitem(node) is not None:
            continue
        if _is_stateful_or_fused_snn_node(node):
            continue
        if (
            _getitem_index(node) is not None
            and isinstance(node.args[0], torch.fx.Node)
            and (
                node.args[0].op == "placeholder"
                or _get_kairos_meta(node.args[0], "timestep") is None
            )
        ):
            # A direct getitem(external_sequence_source, constant_t) where
            # the source is either the raw placeholder or, more generally,
            # any node with no kairos_timestep annotation at all -- i.e. a
            # value computed *outside* every per-timestep block (before the
            # first marker), such as KairosDeepSpeech2.frontend()'s conv
            # output reshaped once before the T-loop indexes into it: that
            # reshape sits outside all blocks by construction (nothing
            # anchors it), so checking specifically for op=="placeholder"
            # missed it and produced the exact same off-by-one symptom this
            # boundary condition was added to fix, one level removed. This
            # is the "sequence" input-mode analogue of
            # _match_temporal_stack_getitem's specific fused_temporal_*-stack
            # pattern: a compile-time-constant slice of an externally
            # supplied full sequence tensor is genuinely independent per t
            # regardless of which timestep's block split_fx_graph_into_timesteps
            # happened to assign it to (the select necessarily sits just
            # before the block boundary marker that consumes it, so it is
            # annotated with the *previous* timestep even though its data is
            # this timestep's -- a boundary-adjacency artifact, not a real
            # cross-iteration dependency).
            continue
        node_timestep = _get_kairos_meta(node, "timestep")
        if isinstance(node_timestep, int):
            if node_timestep != candidate_timestep:
                return False
            # Same-timestep match: this node's own annotation already
            # certifies it as a genuinely-this-t value regardless of what
            # its own inputs are -- if ITS computation were itself unsafe to
            # batch (e.g. it directly consumes a cross-timestep value), that
            # gets caught independently when IT is evaluated as its own
            # candidate. Batching legality is a per-op property, not
            # transitive over the whole ancestry: e.g. layer1's xproj(h_t)
            # is safe to batch across t even though h_t's own production
            # internally reads layer0's h_{t-1} -- h_t itself, as the value
            # xproj consumes, is correctly this-t's data. Continuing to
            # descend here would incorrectly reject every same-block
            # consumer of any node whose *own* history happens to touch a
            # prior timestep anywhere upstream.
            continue
        for arg in list(node.args) + list(node.kwargs.values()):
            frontier.extend(_collect_input_nodes(arg))
    return True


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
        item
        for item in (
            _match_temporal_stack_getitem(input_item) or _match_external_sequence_getitem(input_item, temporal_window)
            for input_item in tensor_inputs
        )
        if item is not None
    )
    previous_batched_inputs = tuple(
        item for item in (_match_batched_chunk_getitem(input_item) for input_item in tensor_inputs) if item is not None
    )
    mixed_plain_operand_index = None
    mixed_plain_input = None
    if kind in DUAL_OPERAND_KINDS:
        if len(tensor_inputs) != 2:
            stats.skip(f"{kind}_requires_two_tensor_inputs", f"node={node.name} {kind} must have two tensor inputs")
            return None
        for input_item in tensor_inputs:
            if (
                _is_generated_spatial_batching_node(input_item)
                and _match_temporal_stack_getitem(input_item) is None
                and _match_batched_chunk_getitem(input_item) is None
                and _match_external_sequence_getitem(input_item, temporal_window) is None
            ):
                stats.skip(
                    f"{kind}_direct_generated_batched_input",
                    f"node={node.name} {kind} input {input_item.name} is already batched but not chunk-indexed",
                )
                return None
        if temporal_stack_inputs and previous_batched_inputs:
            stats.skip(f"{kind}_mixed_batched_source_kinds", f"node={node.name} {kind} mixes temporal stack and previous batched inputs")
            return None
        batchable_inputs = len(temporal_stack_inputs) + len(previous_batched_inputs)
        if batchable_inputs and batchable_inputs != len(tensor_inputs):
            if batchable_inputs != 1:
                stats.skip(f"{kind}_mixed_batchable_inputs", f"node={node.name} {kind} has unsupported mixed temporal/plain inputs")
                return None
            # Exactly one operand is stack/chain-sourced (e.g. Mamba's
            # `y * silu(z)`: y comes from the scan's per-window stack, z is
            # freshly computed this timestep from this same layer's own
            # in_proj). Find which operand index that is; the OTHER operand
            # resolves at rewrite time by cat-ing all T candidates' own
            # values for that operand instead of requiring it to itself be
            # a recognized stack/chain source -- see
            # _mixed_operand_inputs_for_group. Still requires the plain
            # operand to be legally per-timestep (same-timestep-derived, not
            # a genuine cross-iteration/state dependency).
            stack_or_chain_index = None
            for index, input_item in enumerate(tensor_inputs):
                if (
                    _match_temporal_stack_getitem(input_item)
                    or _match_external_sequence_getitem(input_item, temporal_window)
                    or _match_batched_chunk_getitem(input_item)
                ):
                    stack_or_chain_index = index
                    break
            if stack_or_chain_index is None:
                stats.skip(f"{kind}_mixed_batchable_inputs", f"node={node.name} {kind} has unsupported mixed temporal/plain inputs")
                return None
            plain_index = 1 - stack_or_chain_index
            plain_operand = tensor_inputs[plain_index]
            node_timestep = _get_kairos_meta(node, "timestep")
            if not isinstance(node_timestep, int):
                stats.skip(
                    f"{kind}_mixed_missing_timestep",
                    f"node={node.name} {kind} plain operand {plain_operand.name} has no timestep to check legality against",
                )
                return None
            if not _is_batching_source_legal(plain_operand, node_timestep):
                stats.skip(
                    "feedback_dependency",
                    f"node={node.name} kind={kind} input={plain_operand.name} (mixed-resolution plain operand) depends "
                    f"on a different timestep's block (cross-iteration recurrence)",
                )
                return None
            mixed_plain_operand_index = plain_index
            mixed_plain_input = plain_operand
        if len(temporal_stack_inputs) == 2 and temporal_stack_inputs[0].timestep != temporal_stack_inputs[1].timestep:
            stats.skip(f"temporal_stack_{kind}_timestep_mismatch", f"node={node.name} {kind} temporal stack timesteps differ")
            return None
        if len(previous_batched_inputs) == 2 and previous_batched_inputs[0].timestep != previous_batched_inputs[1].timestep:
            stats.skip(f"previous_batched_{kind}_timestep_mismatch", f"node={node.name} {kind} previous chunk timesteps differ")
            return None
    primary_temporal_stack = _match_temporal_stack_getitem(input_node) or _match_external_sequence_getitem(input_node, temporal_window)
    primary_previous_batched = _match_batched_chunk_getitem(input_node)
    timestep = _get_kairos_meta(node, "timestep")
    if not isinstance(timestep, int) and primary_temporal_stack is not None:
        timestep = primary_temporal_stack.timestep
    if not isinstance(timestep, int) and primary_previous_batched is not None:
        timestep = primary_previous_batched.timestep
    if not isinstance(timestep, int):
        stats.skip("missing_timestep", f"node={node.name} kind={kind} has no _kairos_timestep")
        return None
    for tensor_input in tensor_inputs:
        if (
            _match_temporal_stack_getitem(tensor_input) is not None
            or _match_batched_chunk_getitem(tensor_input) is not None
            or _match_external_sequence_getitem(tensor_input, temporal_window) is not None
        ):
            continue
        if not _is_batching_source_legal(tensor_input, timestep):
            stats.skip(
                "feedback_dependency",
                f"node={node.name} kind={kind} input={tensor_input.name} depends on a different "
                f"timestep's block (cross-iteration recurrence, e.g. W_h-side conv/linear consuming "
                f"h_prev) -- batching would silently reuse the wrong per-t value",
            )
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
    if kind in DUAL_OPERAND_KINDS and (temporal_stack_inputs or previous_batched_inputs):
        stack_refs = tuple((item.spike_stack.name, item.source_op) for item in temporal_stack_inputs)
        previous_refs = tuple(item.batched_node.name for item in previous_batched_inputs)
        kwargs = tuple(sorted((key, _node_ref(value)) for key, value in node.kwargs.items()))
        signature = (node.op, kind, stack_refs, previous_refs, mixed_plain_operand_index, kwargs)
    else:
        signature = _signature_without_input(node, kind)
    occurrence = _get_kairos_meta(node, "occurrence")
    if not isinstance(occurrence, int):
        count_key = (timestep, signature)
        occurrence = occurrence_counts.get(count_key, 0)
        occurrence_counts[count_key] = occurrence + 1
    window_id = _get_kairos_meta(node, "window_id")
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
        mixed_plain_operand_index=mixed_plain_operand_index,
        mixed_plain_input=mixed_plain_input,
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
        if candidate.kind in DUAL_OPERAND_KINDS and candidate.temporal_stack_inputs:
            temporal_stack_key = tuple(item.spike_stack.name for item in candidate.temporal_stack_inputs)
        previous_batched_key = ()
        if candidate.previous_batched_input is not None:
            previous_batched_key = (candidate.previous_batched_input.batched_node.name,)
        if candidate.kind in DUAL_OPERAND_KINDS and candidate.previous_batched_inputs:
            previous_batched_key = tuple(item.batched_node.name for item in candidate.previous_batched_inputs)
        key = (
            candidate.window_id,
            candidate.occurrence,
            candidate.signature,
            temporal_stack_key,
            previous_batched_key,
            candidate.mixed_plain_operand_index,
        )
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
        if first.kind in DUAL_OPERAND_KINDS and first.temporal_stack_inputs:
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
        if first.kind in DUAL_OPERAND_KINDS and first.previous_batched_inputs:
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
    if group.kind in DUAL_OPERAND_KINDS:
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
    if first.kind in DUAL_OPERAND_KINDS and first.temporal_stack_inputs:
        return tuple(item.spike_stack for item in first.temporal_stack_inputs)
    if first.temporal_stack_input is not None:
        return (first.temporal_stack_input.spike_stack,)
    return ()


def _previous_batched_nodes_for_group(group: SpatialBatchGroup) -> Tuple[torch.fx.Node, ...]:
    first = group.candidates[0]
    if first.kind in DUAL_OPERAND_KINDS and first.previous_batched_inputs:
        return tuple(item.batched_node for item in first.previous_batched_inputs)
    if first.previous_batched_input is not None:
        return (first.previous_batched_input.batched_node,)
    return ()


def _plain_add_operand_inputs_for_group(group: SpatialBatchGroup) -> Tuple[Tuple[torch.fx.Node, ...], ...]:
    if group.kind not in DUAL_OPERAND_KINDS:
        return ()
    first = group.candidates[0]
    if first.temporal_stack_inputs or first.previous_batched_inputs:
        return ()
    per_candidate_inputs = [_candidate_tensor_inputs(candidate.node, group.kind) for candidate in group.candidates]
    if any(len(inputs) != 2 for inputs in per_candidate_inputs):
        return ()
    return tuple(tuple(inputs[index] for inputs in per_candidate_inputs) for index in range(2))


def _mixed_operand_inputs_for_group(group: SpatialBatchGroup) -> Optional[Tuple[torch.fx.Node, ...]]:
    """For DUAL_OPERAND_KINDS groups (add/mul) where exactly one operand is
    stack/chain-sourced and the other is a plain per-timestep value (see
    SpatialBatchCandidate.mixed_plain_operand_index / _extract_candidate's
    mixed-resolution branch, e.g. Mamba's `y * silu(z)`): returns the plain
    operand's own per-candidate nodes, in the SAME timestep order as
    group.candidates, ready to be cat'd at rewrite time. None if this isn't
    a mixed-resolution group (the two existing all-batchable / all-plain
    resolutions handle those; this is strictly the third case).
    """
    if group.kind not in DUAL_OPERAND_KINDS:
        return None
    first = group.candidates[0]
    if first.mixed_plain_operand_index is None:
        return None
    return tuple(candidate.mixed_plain_input for candidate in group.candidates)


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
    flatten_node.meta["kairos_temporal_layout"] = "batched_tn"
    flatten_node.meta["kairos_T"] = temporal_window
    flatten_node.meta["kairos_origin"] = "temporal_stack_flatten"
    flatten_node.meta["kairos_source_stack"] = stack_node.name
    return flatten_node


def _make_temporal_stack_batched_input(
    gm: torch.fx.GraphModule,
    stack_node: torch.fx.Node,
    name_prefix: str,
    window_start: int,
    temporal_window: int,
) -> torch.fx.Node:
    """Builds the batched [window*B, ...] input for one temporal-stack
    source, narrowing to this window's slice first when the source is a
    whole-sequence tensor (see _match_external_sequence_getitem /
    _is_fused_temporal_stack_source) -- an SNN fused-temporal-op source is
    already exactly window-sized so it flattens directly, unchanged from
    before this function existed.
    """
    if _is_fused_temporal_stack_source(stack_node):
        return _make_temporal_stack_flatten(gm, stack_node, name_prefix, temporal_window)
    narrowed = gm.graph.call_function(torch.narrow, args=(stack_node, 0, window_start, temporal_window))
    narrowed.name = f"{name_prefix}_window_slice"
    narrowed.meta["kairos_origin"] = "external_sequence_window_slice"
    return _make_temporal_stack_flatten(gm, narrowed, name_prefix, temporal_window)


def rewrite_spatial_batch_group(
    gm: torch.fx.GraphModule,
    group: SpatialBatchGroup,
    stats: Optional[SpatialBatchingStats] = None,
) -> Tuple[bool, str]:
    first_node = group.candidates[0].node
    temporal_stack_nodes = _temporal_stack_nodes_for_group(group)
    previous_batched_nodes = _previous_batched_nodes_for_group(group)
    plain_add_operand_inputs = _plain_add_operand_inputs_for_group(group)
    mixed_plain_inputs = _mixed_operand_inputs_for_group(group)
    if mixed_plain_inputs is not None:
        # The stack/chain-sourced operand is already a fully materialized
        # window tensor (available by construction, same as the pure
        # temporal_stack_nodes/previous_batched_nodes cases below) -- only
        # the plain operand's own T per-candidate values need the
        # availability check, since those get cat'd fresh at rewrite time.
        input_nodes = list(mixed_plain_inputs)
    elif temporal_stack_nodes:
        input_nodes = list(temporal_stack_nodes)
    elif previous_batched_nodes:
        input_nodes = list(previous_batched_nodes)
    elif plain_add_operand_inputs:
        input_nodes = [node for operand_inputs in plain_add_operand_inputs for node in operand_inputs]
    else:
        input_nodes = [candidate.input_node for candidate in group.candidates]
    ok, reason = _all_inputs_available_before(gm, input_nodes, first_node)
    if not ok:
        return False, reason
    has_dep, reason = _has_internal_data_dependency(group)
    if has_dep:
        return False, reason

    with gm.graph.inserting_before(first_node):
        if mixed_plain_inputs is not None:
            plain_index = group.candidates[0].mixed_plain_operand_index
            if temporal_stack_nodes:
                window_start = group.window_id * len(group.candidates)
                stack_side = _make_temporal_stack_batched_input(
                    gm, temporal_stack_nodes[0], f"{first_node.name}_stack", window_start, len(group.candidates)
                )
                if stats is not None:
                    stats.spatial_temporal_stack_groups += 1
                    stats.spatial_temporal_stack_flatten_inputs += 1
                    stats.spatial_cat_avoided_by_temporal_stack_flatten += 1
            else:
                stack_side = previous_batched_nodes[0]
                if stats is not None:
                    stats.spatial_previous_batched_groups += 1
                    stats.spatial_reused_previous_batched_inputs += 1
                    stats.spatial_chunk_cat_avoided += 1
            plain_cat = gm.graph.call_function(torch.cat, args=(list(mixed_plain_inputs), 0))
            plain_cat.name = f"{first_node.name}_spatial_batch_{group.kind}_mixed_plain_cat"
            batched_inputs_list: List[Optional[torch.fx.Node]] = [None, None]
            batched_inputs_list[plain_index] = plain_cat
            batched_inputs_list[1 - plain_index] = stack_side
            batched_inputs = tuple(batched_inputs_list)
            if stats is not None:
                stats.spatial_mixed_operand_groups += 1
        elif temporal_stack_nodes:
            window_start = group.window_id * len(group.candidates)
            batched_inputs = tuple(
                _make_temporal_stack_batched_input(
                    gm, stack_node, f"{first_node.name}_{index}", window_start, len(group.candidates)
                )
                for index, stack_node in enumerate(temporal_stack_nodes)
            )
            if stats is not None:
                stats.spatial_temporal_stack_groups += 1
                stats.spatial_temporal_stack_flatten_inputs += len(batched_inputs)
                stats.spatial_cat_avoided_by_temporal_stack_flatten += 1
                if group.kind == "bn":
                    stats.spatial_temporal_stack_bn_groups += 1
                elif group.kind in DUAL_OPERAND_KINDS:
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
        elif plain_add_operand_inputs:
            batched_inputs = []
            for operand_index, operand_inputs in enumerate(plain_add_operand_inputs):
                cat_node = gm.graph.call_function(torch.cat, args=(list(operand_inputs), 0))
                cat_node.name = f"{first_node.name}_spatial_batch_add_operand{operand_index}_cat"
                batched_inputs.append(cat_node)
            batched_inputs = tuple(batched_inputs)
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


def _transformer_item_signature(item: TransformerAttentionItem) -> Tuple[Any, ...]:
    return (
        _node_ref(item.source_node),
        _node_ref(item.normalized_shape),
        _node_ref(item.norm_weight),
        _node_ref(item.norm_bias),
        item.norm_eps,
        _node_ref(item.qkv_weight),
        _node_ref(item.qkv_bias),
        item.reshape_args,
        item.scale,
        item.softmax_dim,
        item.output_shape_args,
    )


def _layer_norm_item_signature(item: LayerNormStackItem) -> Tuple[Any, ...]:
    return (
        _node_ref(item.source_node),
        _node_ref(item.normalized_shape),
        _node_ref(item.norm_weight),
        _node_ref(item.norm_bias),
        item.norm_eps,
    )


def _sequence_source_for_items(
    gm: torch.fx.GraphModule,
    items: Sequence[Any],
    insertion_point: torch.fx.Node,
) -> Optional[torch.fx.Node]:
    first = items[0]
    if first.source_kind in {"temporal_stack", "previous_batched"}:
        return first.source_node
    if first.source_kind == "plain":
        with gm.graph.inserting_before(insertion_point):
            stack = gm.graph.call_function(torch.stack, args=([item.input_node for item in items], 0))
            stack.name = f"{insertion_point.name}_transformer_input_stack"
            stack.meta["kairos_origin"] = "transformer_spatial_batch_input_stack"
            return stack
    return None


def _annotate_transformer_temporal_nodes(items: Sequence[TransformerAttentionItem], temporal_window: int, window_id: int):
    for occurrence, item in enumerate(items):
        for node in (
            item.layer_norm_node,
            item.qkv_linear_node,
            item.qkv_reshape_node,
            item.qkv_movedim_node,
            item.qkv_transpose_node,
            item.q_node,
            item.k_node,
            item.v_node,
            item.k_transpose_node,
            item.qk_matmul_node,
            item.scale_node,
            item.softmax_node,
            item.av_matmul_node,
            item.out_transpose_node,
            item.output_node,
        ):
            node.meta["kairos_timestep"] = item.timestep
            node.meta["kairos_window_id"] = window_id
            node.meta["kairos_occurrence"] = occurrence
            node.meta["kairos_role"] = "transformer_attention"
            setattr(node, "_kairos_timestep", item.timestep)
            setattr(node, "_kairos_window_id", window_id)
            setattr(node, "_kairos_occurrence", occurrence)
            setattr(node, "_kairos_role", "transformer_attention")


def _replace_attention_zero_state_users(
    gm: torch.fx.GraphModule,
    old_stack: torch.fx.Node,
    new_x_seq: torch.fx.Node,
):
    for user in list(old_stack.users):
        if user.op != "call_function":
            continue
        target_text = _target_text(user.target)
        if "fused_temporal_batched_linear_add_lif_state" not in target_text:
            continue
        args = list(user.args)
        if len(args) < 5 or args[0] is not old_stack:
            continue
        v_init = args[4]
        if not isinstance(v_init, torch.fx.Node):
            continue
        if v_init.op != "call_function" or v_init.target is not torch.zeros_like:
            continue
        with gm.graph.inserting_before(user):
            scalar_zero = gm.graph.call_method("new_zeros", args=(new_x_seq, ()))
            scalar_zero.name = f"{user.name}_spatial_batch_scalar_v_init"
        args[4] = scalar_zero
        user.args = tuple(args)


def _rewrite_transformer_attention_stack(
    gm: torch.fx.GraphModule,
    stack_node: torch.fx.Node,
    temporal_window: int,
    stats: SpatialBatchingStats,
) -> bool:
    stack = _stack_items(stack_node)
    if stack is None:
        return False
    values, _dim = stack
    if len(values) != temporal_window:
        return False
    items = [_match_transformer_attention_output(value) for value in values]
    if any(item is None for item in items):
        return False
    items = [item for item in items if item is not None]
    items = sorted(items, key=lambda item: item.timestep)
    first = items[0]
    expected_timesteps = list(range(first.timestep, first.timestep + temporal_window))
    if [item.timestep for item in items] != expected_timesteps:
        stats.skip("transformer_attention_timesteps", f"stack={stack_node.name} timesteps={[item.timestep for item in items]}")
        return False
    if any(_transformer_item_signature(item) != _transformer_item_signature(first) for item in items):
        stats.skip("transformer_attention_signature", f"stack={stack_node.name} incompatible signatures")
        return False
    window_id = first.timestep // temporal_window if temporal_window > 0 else 0
    x_seq = _sequence_source_for_items(gm, items, stack_node)
    if x_seq is None:
        return False

    _annotate_transformer_temporal_nodes(items, temporal_window, window_id)

    qkv_shape = (temporal_window,) + tuple(first.reshape_args)
    output_shape = (temporal_window,) + tuple(first.output_shape_args)
    with gm.graph.inserting_before(stack_node):
        ln = gm.graph.call_function(
            F.layer_norm,
            args=(x_seq, first.normalized_shape, first.norm_weight, first.norm_bias, first.norm_eps),
        )
        ln.name = f"{stack_node.name}_spatial_batch_attention_layer_norm"
        qkv = gm.graph.call_function(F.linear, args=(ln, first.qkv_weight, first.qkv_bias))
        qkv.name = f"{stack_node.name}_spatial_batch_attention_qkv"
        qkv_reshape = gm.graph.call_method("reshape", args=(qkv,) + qkv_shape)
        qkv_reshape.name = f"{stack_node.name}_spatial_batch_attention_qkv_reshape"
        movedim = gm.graph.call_method("movedim", args=(qkv_reshape, -3, 0))
        movedim.name = f"{stack_node.name}_spatial_batch_attention_movedim"
        qkv_transpose = gm.graph.call_method("transpose", args=(movedim, -3, -2))
        qkv_transpose.name = f"{stack_node.name}_spatial_batch_attention_qkv_transpose"
        q = gm.graph.call_function(operator.getitem, args=(qkv_transpose, 0))
        q.name = f"{stack_node.name}_spatial_batch_attention_q"
        k = gm.graph.call_function(operator.getitem, args=(qkv_transpose, 1))
        k.name = f"{stack_node.name}_spatial_batch_attention_k"
        v = gm.graph.call_function(operator.getitem, args=(qkv_transpose, 2))
        v.name = f"{stack_node.name}_spatial_batch_attention_v"
        k_t = gm.graph.call_method("transpose", args=(k, -2, -1))
        k_t.name = f"{stack_node.name}_spatial_batch_attention_k_t"
        qk = gm.graph.call_function(torch.matmul, args=(q, k_t))
        qk.name = f"{stack_node.name}_spatial_batch_attention_qk"
        scaled = gm.graph.call_function(operator.mul, args=(qk, first.scale))
        scaled.name = f"{stack_node.name}_spatial_batch_attention_scaled"
        attn = gm.graph.call_function(torch.softmax, args=(scaled,), kwargs={"dim": first.softmax_dim})
        attn.name = f"{stack_node.name}_spatial_batch_attention_softmax"
        av = gm.graph.call_function(torch.matmul, args=(attn, v))
        av.name = f"{stack_node.name}_spatial_batch_attention_av"
        out_t = gm.graph.call_method("transpose", args=(av, -3, -2))
        out_t.name = f"{stack_node.name}_spatial_batch_attention_out_transpose"
        out = gm.graph.call_method("reshape", args=(out_t,) + output_shape)
        out.name = f"{stack_node.name}_spatial_batch_attention"

    _replace_attention_zero_state_users(gm, stack_node, out)
    stack_node.replace_all_uses_with(out)
    if len(stack_node.users) == 0:
        gm.graph.erase_node(stack_node)
    stats.spatial_batch_groups += 1
    stats.spatial_batched_ops += temporal_window
    stats.spatial_batched_attention += temporal_window
    stats.log.append(
        f"[SPATIAL_BATCHING][TRANSFORMER_ATTENTION] stack={stack_node.name} "
        f"window={window_id} size={temporal_window}"
    )
    print(stats.log[-1])
    return True


def _rewrite_layer_norm_stack(
    gm: torch.fx.GraphModule,
    stack_node: torch.fx.Node,
    temporal_window: int,
    stats: SpatialBatchingStats,
) -> bool:
    stack = _stack_items(stack_node)
    if stack is None:
        return False
    values, _dim = stack
    if len(values) != temporal_window:
        return False
    items = [_match_layer_norm_stack_item(value) for value in values]
    if any(item is None for item in items):
        return False
    items = [item for item in items if item is not None]
    items = sorted(items, key=lambda item: item.timestep)
    first = items[0]
    expected_timesteps = list(range(first.timestep, first.timestep + temporal_window))
    if [item.timestep for item in items] != expected_timesteps:
        stats.skip("layer_norm_timesteps", f"stack={stack_node.name} timesteps={[item.timestep for item in items]}")
        return False
    if any(_layer_norm_item_signature(item) != _layer_norm_item_signature(first) for item in items):
        stats.skip("layer_norm_signature", f"stack={stack_node.name} incompatible signatures")
        return False
    window_id = first.timestep // temporal_window if temporal_window > 0 else 0
    x_seq = _sequence_source_for_items(gm, items, stack_node)
    if x_seq is None:
        return False
    for occurrence, item in enumerate(items):
        item.output_node.meta["kairos_timestep"] = item.timestep
        item.output_node.meta["kairos_window_id"] = window_id
        item.output_node.meta["kairos_occurrence"] = occurrence
        item.output_node.meta["kairos_role"] = "transformer_layer_norm"

    with gm.graph.inserting_before(stack_node):
        ln = gm.graph.call_function(
            F.layer_norm,
            args=(x_seq, first.normalized_shape, first.norm_weight, first.norm_bias, first.norm_eps),
        )
        ln.name = f"{stack_node.name}_spatial_batch_layer_norm"
        ln.meta["kairos_origin"] = "transformer_spatial_batch_layer_norm"

    stack_node.replace_all_uses_with(ln)
    if len(stack_node.users) == 0:
        gm.graph.erase_node(stack_node)
    stats.spatial_batch_groups += 1
    stats.spatial_batched_ops += temporal_window
    stats.spatial_batched_layer_norm += temporal_window
    stats.log.append(
        f"[SPATIAL_BATCHING][LAYER_NORM] stack={stack_node.name} window={window_id} size={temporal_window}"
    )
    print(stats.log[-1])
    return True


def rewrite_transformer_spatial_batching(
    gm: torch.fx.GraphModule,
    temporal_window: int,
    stats: SpatialBatchingStats,
) -> int:
    rewrites = 0
    # Attention stacks must run before generic layer_norm stack rewrites,
    # otherwise the strict attention chain no longer exists.
    for node in list(gm.graph.nodes):
        if _rewrite_transformer_attention_stack(gm, node, temporal_window, stats):
            rewrites += 1
    if rewrites:
        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()
    for node in list(gm.graph.nodes):
        if _rewrite_layer_norm_stack(gm, node, temporal_window, stats):
            rewrites += 1
    if rewrites:
        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()
    return rewrites


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
        f"batched_layer_norm={stats.spatial_batched_layer_norm}",
        f"batched_attention={stats.spatial_batched_attention}",
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
    max_iter: int = 8,
) -> SpatialBatchingStats:
    stats = SpatialBatchingStats()
    if temporal_window <= 1:
        stats.skip("window_le_1", f"temporal_window={temporal_window}")
        return stats

    try:
        if enable_chain:
            stats.skip("chain_disabled", "chain-aware spatial batching is deprecated; using per-op batching")
        rewrite_transformer_spatial_batching(gm, temporal_window, stats)
        all_groups: List[SpatialBatchGroup] = []
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
                elif group.kind == "mul":
                    stats.spatial_batched_mul += count
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
                elif group.kind == "layer_norm":
                    stats.spatial_batched_layer_norm += count
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
            f"elementwise={stats.spatial_batched_elementwise} "
            f"layer_norm={stats.spatial_batched_layer_norm} attention={stats.spatial_batched_attention}"
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
