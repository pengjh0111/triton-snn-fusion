"""Pass A: replace matmul -> scale -> softmax(dim=-1) -> matmul attention
patterns with torch.nn.functional.scaled_dot_product_attention (SDPA).

Toggle: env var CHRONOS_PASS_SDPA (default: on; set to "0"/"false" to
disable -- see compiler/passes/registry.py).

Verification level: tolerance (SDPA's internal reduction order over the
softmax/matmul differs from the naive path, so outputs are close but not
bit-identical).

Why (benefit rationale): the unfused pattern fully materializes the
[T, B, H, S, S] attention score tensor at every occurrence -- for this
model's default shapes that is roughly 33M elements written by matmul, read
+ written by the scale-multiply, read + written by softmax, and read again
by the second matmul. That is the single largest per-op memory-traffic hot
spot anywhere in the post-fuse graph (there are 8 attention blocks x T
timesteps of these in the full model). SDPA fuses the whole sequence into
one kernel that never materializes the full [.., S, S] score matrix, turning
an O(S^2) memory-bound chain of 4 kernel launches into effectively one
compute-bound one.

CUDA-graph note (resolved): an earlier version of this pass called bare
torch.nn.functional.scaled_dot_product_attention and let PyTorch's dispatcher
auto-select a backend. Under `--fx-standalone-cudagraph` (multi-stream
capture/replay), that auto-selected FLASH_ATTENTION backend for fp16 turned
this pass into a net regression (+14% to +115% depending on dtype/config,
confirmed via direct A/B ablation) even though CUDA graph capture itself
always succeeded -- the captured FLASH_ATTENTION kernel simply replayed
inefficiently at this model's shapes. Pinning EFFICIENT_ATTENTION instead
(see `_sdpa_capture_safe` below) resolved it and turned Pass A into a solid
net win in every tested (dtype x cudagraph) configuration, including a
9.305ms vs 18.277ms (Pass A off) result for fp16 + cudagraph.
"""

import operator
from dataclasses import dataclass, field
from typing import Dict

import torch


def _sdpa_capture_safe(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    """scaled_dot_product_attention with the backend pinned to
    EFFICIENT_ATTENTION instead of letting PyTorch's dispatcher auto-select.

    Root-caused via a direct ablation (see the pass-A cudagraph-regression
    investigation): under `--fx-standalone-cudagraph`, PyTorch's automatic
    SDPA backend selection picks FLASH_ATTENTION for fp16 inputs, since
    that's normally the fastest *eager* choice. But FLASH_ATTENTION's kernel
    does not replay efficiently once captured into a CUDA graph at this
    model's (small-sequence-length, many-batched-instances) shapes -- an
    isolated capture+replay microbenchmark measured FLASH_ATTENTION replay
    at ~1.5x the naive 4-op path's replayed time, while EFFICIENT_ATTENTION
    replayed at parity with the naive path (within ~2%). CUDA graph capture
    itself still succeeds either way (confirmed via
    runtime/fx_standalone_executor.py's captured/fallback status) -- this is
    not a capture failure, it's a backend whose kernel is capture-*safe* but
    not capture-*efficient*. Pinning the backend avoids the dispatcher
    picking the eager-fast/replay-slow option.

    EFFICIENT_ATTENTION only accepts rank-4 (batch, heads, seq, head_dim) q/k/v
    -- it raises "No available kernel" on rank-3 or rank-5+ inputs, which auto
    dispatch (and FLASH_ATTENTION) handle transparently by treating any extra
    leading dims as batch. In this model's actual post-spatial-batching graph,
    q/k/v arrive as rank-5 (an extra leading dim from batching multiple
    repeat-instances together), so this flattens every leading dim except
    `heads` into one batch dim before the call and restores the original
    shape after.
    """
    orig_shape = q.shape
    if q.dim() == 4:
        with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION]):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)
    heads, seq_q, head_dim = orig_shape[-3], orig_shape[-2], orig_shape[-1]
    seq_kv = k.shape[-2]
    leading = orig_shape[:-3]
    q2 = q.reshape(-1, heads, seq_q, head_dim)
    k2 = k.reshape(-1, heads, seq_kv, head_dim)
    v2 = v.reshape(-1, heads, seq_kv, head_dim)
    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION]):
        out = torch.nn.functional.scaled_dot_product_attention(q2, k2, v2, scale=scale)
    return out.reshape(*leading, heads, seq_q, head_dim)


@dataclass
class SdpaPassStats:
    matched: int = 0
    replaced: int = 0
    skipped: int = 0
    skip_reasons: Dict[str, int] = field(default_factory=dict)

    def skip(self, reason: str):
        self.skipped += 1
        self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + 1


def _is_softmax(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and node.target in (torch.softmax, torch.nn.functional.softmax)


def _is_matmul(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and node.target is torch.matmul


def _is_mul(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and node.target is operator.mul


def _softmax_dim(node: torch.fx.Node):
    if "dim" in node.kwargs:
        return node.kwargs["dim"]
    if len(node.args) >= 2:
        return node.args[1]
    return None


def _split_node_and_constant(a, b):
    """Given a 2-tuple of operator.mul args, return (node_operand,
    constant_operand) if exactly one side is an fx.Node and the other is a
    plain scalar; otherwise return (None, None)."""
    a_is_node = isinstance(a, torch.fx.Node)
    b_is_node = isinstance(b, torch.fx.Node)
    if a_is_node and not b_is_node:
        return a, b
    if b_is_node and not a_is_node:
        return b, a
    return None, None


def apply_sdpa_pass(gm: torch.fx.GraphModule) -> SdpaPassStats:
    stats = SdpaPassStats()
    graph = gm.graph

    for node in list(graph.nodes):
        if not _is_softmax(node):
            continue
        if _softmax_dim(node) != -1:
            stats.skip("dim_not_neg1")
            continue
        if not node.args:
            stats.skip("softmax_no_positional_input")
            continue

        mul_node = node.args[0]
        if not isinstance(mul_node, torch.fx.Node) or not _is_mul(mul_node):
            stats.skip("no_scale_mul")
            continue
        if len(mul_node.users) != 1:
            stats.skip("scaled_has_extra_users")
            continue
        if len(mul_node.args) != 2:
            stats.skip("mul_arity")
            continue

        mm_node, scale = _split_node_and_constant(*mul_node.args)
        if mm_node is None:
            stats.skip("mul_operand_shape")
            continue
        if not _is_matmul(mm_node):
            stats.skip("scaled_input_not_matmul")
            continue
        if len(mm_node.users) != 1:
            stats.skip("qk_matmul_has_extra_users")
            continue
        if len(mm_node.args) != 2:
            stats.skip("qk_matmul_arity")
            continue

        q_node, k_transposed_node = mm_node.args
        if not isinstance(q_node, torch.fx.Node) or not isinstance(k_transposed_node, torch.fx.Node):
            stats.skip("qk_operands_not_nodes")
            continue

        is_recognized_transpose = (
            k_transposed_node.op == "call_method"
            and k_transposed_node.target == "transpose"
            and len(k_transposed_node.args) == 3
            and set(k_transposed_node.args[1:]) == {-2, -1}
        )
        if not is_recognized_transpose:
            stats.skip("k_not_recognized_transpose")
            continue
        k_node = k_transposed_node.args[0]
        if not isinstance(k_node, torch.fx.Node):
            stats.skip("k_operand_not_node")
            continue
        if len(k_transposed_node.users) != 1:
            stats.skip("k_transpose_has_extra_users")
            continue

        if len(node.users) != 1:
            stats.skip("softmax_has_extra_users")
            continue
        av_node = next(iter(node.users))
        if not _is_matmul(av_node):
            stats.skip("no_av_matmul")
            continue
        if len(av_node.args) != 2 or av_node.args[0] is not node:
            stats.skip("av_matmul_shape")
            continue
        v_node = av_node.args[1]
        if not isinstance(v_node, torch.fx.Node):
            stats.skip("v_not_node")
            continue

        stats.matched += 1

        with graph.inserting_before(av_node):
            sdpa_node = graph.call_function(
                _sdpa_capture_safe,
                args=(q_node, k_node, v_node),
                kwargs={"scale": scale},
            )
        av_node.replace_all_uses_with(sdpa_node)
        stats.replaced += 1

    graph.eliminate_dead_code()
    graph.lint()
    gm.recompile()
    return stats
