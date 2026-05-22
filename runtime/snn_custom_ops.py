from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple
from collections import Counter
import os

import torch
import torch.nn.functional as F

from runtime.triton_convlif_backend import (
    check_triton_support,
    run_triton_fused_conv_lif_state,
    run_triton_fused_temporal_conv_add_lif_state,
    run_triton_fused_temporal_conv_lif_state,
)
from runtime.triton_temporal_lif_backend import run_triton_fused_temporal_lif_state
from runtime.triton_temporal_lif_avgpool_linear_backend import run_triton_fused_temporal_lif_avgpool_linear


TORCH_LIBRARY_HANDLES = []


@dataclass
class FusedOpConfig:
    backend: str = "torch"
    strict_triton: bool = False
    verbose: bool = False


_CONFIG = FusedOpConfig()
_CALL_STATS: Dict[str, Any] = {
    "total": 0,
    "triton": 0,
    "fallback": 0,
    "temporal_total": 0,
    "temporal_triton": 0,
    "temporal_fallback": 0,
    "single_k3_s1_p1": 0,
    "single_k3_s2_p1": 0,
    "single_k5_s1_p2": 0,
    "single_k7_s2_p3": 0,
    "single_k11_s4_p2": 0,
    "temporal_k3_s1_p1": 0,
    "temporal_k3_s2_p1": 0,
    "temporal_k5_s1_p2": 0,
    "temporal_k7_s2_p3": 0,
    "temporal_k11_s4_p2": 0,
    "temporal_residual_total": 0,
    "temporal_residual_triton": 0,
    "temporal_residual_fallback": 0,
    "temporal_lif_total": 0,
    "temporal_lif_triton": 0,
    "temporal_lif_fallback": 0,
    "temporal_lif_avgpool_linear_total": 0,
    "temporal_lif_avgpool_linear_triton": 0,
    "temporal_lif_avgpool_linear_fallback": 0,
    # Deprecated compatibility fields; mirrors temporal_lif_avgpool_linear_*.
    "temporal_lif_tail_total": 0,
    "temporal_lif_tail_triton": 0,
    "temporal_lif_tail_fallback": 0,
    "temporal_batched_output_total": 0,
    "kernel_temporal_configs": {},
}

_FALLBACK_REASON_STATS = Counter()


def _as_pair(value):
    if isinstance(value, int):
        return (value, value)
    return (int(value[0]), int(value[1]))


def _shape_tuple(x):
    if isinstance(x, torch.Tensor):
        return tuple(x.shape)
    return None


def _conv_shape_desc(x, weight, bias, v, stride, padding, dilation, groups, temporal_len=None):
    if x is None or weight is None:
        return "shape=<unknown>"

    stride = _as_pair(stride)
    padding = _as_pair(padding)
    dilation = _as_pair(dilation)

    N, Cin, H, W = tuple(x.shape)
    Cout, _, KH, KW = tuple(weight.shape)

    return (
        f"T={temporal_len if temporal_len is not None else 1}, "
        f"x={tuple(x.shape)}, weight={tuple(weight.shape)}, bias={_shape_tuple(bias)}, "
        f"v={_shape_tuple(v)}, N={N}, Cin={Cin}, Cout={Cout}, H={H}, W={W}, "
        f"K=({KH},{KW}), stride={stride}, padding={padding}, dilation={dilation}, "
        f"groups={groups}, dtype={x.dtype}, device={x.device}"
    )


def _reason_key(reasons):
    if not reasons:
        return "unknown"
    msg = "; ".join(str(r) for r in reasons)
    if "not implemented" in msg:
        return "not_implemented"
    if "stride" in msg:
        return "unsupported_stride"
    if "padding" in msg:
        return "unsupported_padding"
    if "3x3" in msg or "kernel" in msg:
        return "unsupported_kernel_size"
    if "dilation" in msg:
        return "unsupported_dilation"
    if "groups" in msg:
        return "unsupported_groups"
    if "float32" in msg or "dtype" in msg:
        return "unsupported_dtype"
    if "bias" in msg:
        return "missing_or_unsupported_bias"
    if "v_prev" in msg or "membrane" in msg or "v_init" in msg:
        return "unsupported_membrane_state"
    if "threshold" in msg or "V_THRESHOLD" in msg:
        return "unsupported_threshold"
    if "V_RESET" in msg or "v_reset" in msg:
        return "unsupported_reset"
    if "tau" in msg:
        return "unsupported_tau"
    if "detach_reset" in msg:
        return "unsupported_detach_reset"
    return "other_unsupported"


def _record_fallback(kind: str, reasons, shape_desc: str):
    keys = []
    for reason in reasons or ["unknown"]:
        text = str(reason)
        if ":" in text and text.split(":", 1)[0].startswith("unsupported_"):
            keys.append(text.split(":", 1)[0])
        else:
            keys.append(_reason_key([text]))
    for key in sorted(set(keys)):
        full_key = f"{kind}:{key}"
        _FALLBACK_REASON_STATS[full_key] += 1

    if _CONFIG.verbose:
        reason_text = "; ".join(str(r) for r in reasons) if reasons else "unknown"
        print(f"[TRITON][FALLBACK][{kind}] reason_key={','.join(sorted(set(keys)))}; reason={reason_text}; {shape_desc}")


def configure_fused_op(backend: str = "torch", strict_triton: bool = False, verbose: bool = False):
    if backend not in ("torch", "triton"):
        raise ValueError(f"unsupported fused op backend: {backend}")
    _CONFIG.backend = backend
    _CONFIG.strict_triton = bool(strict_triton)
    _CONFIG.verbose = bool(verbose)


def reset_fused_op_call_stats():
    for key in _CALL_STATS:
        if isinstance(_CALL_STATS[key], dict):
            _CALL_STATS[key].clear()
        else:
            _CALL_STATS[key] = 0
    _FALLBACK_REASON_STATS.clear()


def get_fused_op_call_stats() -> Dict[str, Any]:
    out = {
        key: dict(value) if isinstance(value, dict) else value
        for key, value in _CALL_STATS.items()
    }
    out["fallback_reasons"] = dict(_FALLBACK_REASON_STATS)
    return out


def get_kernel_temporal_configs() -> Dict[str, int]:
    """Get kernel temporal configuration statistics."""
    return dict(_CALL_STATS.get("kernel_temporal_configs", {}))


def reset_fused_op_call_stats():
    """Reset fused op call statistics for per-case tracking."""
    global _CALL_STATS, _FALLBACK_REASON_STATS
    _CALL_STATS = {
        "total": 0,
        "triton": 0,
        "fallback": 0,
        "temporal_total": 0,
        "temporal_triton": 0,
        "temporal_fallback": 0,
        "single_k3_s1_p1": 0,
        "single_k3_s2_p1": 0,
        "single_k5_s1_p2": 0,
        "single_k7_s2_p3": 0,
        "single_k11_s4_p2": 0,
        "temporal_k3_s1_p1": 0,
        "temporal_k3_s2_p1": 0,
        "temporal_k5_s1_p2": 0,
        "temporal_k7_s2_p3": 0,
        "temporal_k11_s4_p2": 0,
        "temporal_residual_total": 0,
        "temporal_residual_triton": 0,
        "temporal_residual_fallback": 0,
        "temporal_lif_total": 0,
        "temporal_lif_triton": 0,
        "temporal_lif_fallback": 0,
        "temporal_lif_avgpool_linear_total": 0,
        "temporal_lif_avgpool_linear_triton": 0,
        "temporal_lif_avgpool_linear_fallback": 0,
        # Deprecated compatibility fields; mirrors temporal_lif_avgpool_linear_*.
        "temporal_lif_tail_total": 0,
        "temporal_lif_tail_triton": 0,
        "temporal_lif_tail_fallback": 0,
        "temporal_batched_output_total": 0,
        "kernel_temporal_configs": {},
    }
    _FALLBACK_REASON_STATS.clear()


def _record_kernel_temporal_config(kind: str, kernel_key: str, config, compute_dtype: str = "float32"):
    if not config:
        return
    btile_t = config.get("BTILE_T")
    reuse_groups = config.get("REUSE_GROUPS")
    window = config.get("kernel_temporal_window")
    key = f"{kind}:{kernel_key}:compute_dtype={compute_dtype}:BTILE_T={btile_t}:REUSE_GROUPS={reuse_groups}:window={window}"
    configs = _CALL_STATS["kernel_temporal_configs"]
    configs[key] = configs.get(key, 0) + 1


def _strict_temporal_lif_triton_enabled() -> bool:
    return bool(_CONFIG.strict_triton) or os.environ.get("CHRONOS_STRICT_TEMPORAL_LIF_TRITON", "0") == "1"


def _ensure_v_prev(x: torch.Tensor, v_prev: torch.Tensor) -> torch.Tensor:
    if v_prev.dim() == 0 or tuple(v_prev.shape) != tuple(x.shape) or v_prev.device != x.device or v_prev.dtype != x.dtype:
        return torch.zeros_like(x)
    return v_prev


def lif_forward_state_torch(
    x,
    v_prev,
    v_threshold: float,
    v_reset: float,
    tau: float,
    detach_reset: bool,
):
    v_prev = _ensure_v_prev(x, v_prev)
    if float(tau) <= 1.0:
        v_before_spike = v_prev + x
    else:
        v_before_spike = v_prev + (x - v_prev) / float(tau)

    spike = (v_before_spike >= float(v_threshold)).to(x.dtype)
    spike_for_reset = spike.detach() if bool(detach_reset) else spike

    if float(v_reset) < 0:
        v_next = v_before_spike - spike_for_reset * float(v_threshold)
    else:
        v_next = torch.where(
            spike_for_reset.bool(),
            torch.full_like(v_before_spike, float(v_reset)),
            v_before_spike,
        )
    return spike, v_next


def fused_conv_lif_state_torch(
    x,
    weight,
    bias,
    v_prev,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    groups: int,
    v_threshold: float,
    v_reset: float,
    tau: float,
    detach_reset: bool,
):
    conv_out = F.conv2d(x, weight, bias, stride, padding, dilation, groups)
    return lif_forward_state_torch(conv_out, v_prev, v_threshold, v_reset, tau, detach_reset)


def fused_temporal_conv_lif_state_torch(
    xs,
    weight,
    bias,
    v_init,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    groups: int,
    v_threshold: float,
    v_reset: float,
    tau: float,
    detach_reset: bool,
):
    if len(xs) == 0:
        raise RuntimeError("fused_temporal_conv_lif_state requires at least one input tensor")

    v = v_init
    spikes = []
    for x in xs:
        conv_out = F.conv2d(x, weight, bias, stride, padding, dilation, groups)
        spike, v = lif_forward_state_torch(conv_out, v, v_threshold, v_reset, tau, detach_reset)
        spikes.append(spike)
    return torch.stack(spikes, dim=0), v


def fused_temporal_conv_add_lif_state_torch(
    xs,
    residuals,
    weight,
    bias,
    v_init,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    groups: int,
    v_threshold: float,
    v_reset: float,
    tau: float,
    detach_reset: bool,
):
    if len(xs) == 0:
        raise RuntimeError("fused_temporal_conv_add_lif_state requires at least one input tensor")
    if len(xs) != len(residuals):
        raise RuntimeError("xs and residuals must have the same temporal length")

    v = v_init
    spikes = []
    for x, residual in zip(xs, residuals):
        conv_out = F.conv2d(x, weight, bias, stride, padding, dilation, groups)
        lif_in = conv_out + residual
        spike, v = lif_forward_state_torch(lif_in, v, v_threshold, v_reset, tau, detach_reset)
        spikes.append(spike)
    return torch.stack(spikes, dim=0), v


def fused_temporal_lif_state_torch(
    x_seq,
    v_init,
    v_threshold: float,
    v_reset: float,
    tau: float,
    detach_reset: bool,
):
    if x_seq.dim() != 5:
        raise RuntimeError(f"fused_temporal_lif_state requires x_seq [T,N,C,H,W], got dim={x_seq.dim()}")
    v = v_init
    spikes = []
    for t in range(int(x_seq.shape[0])):
        spike, v = lif_forward_state_torch(x_seq[t], v, v_threshold, v_reset, tau, detach_reset)
        spikes.append(spike)
    return torch.stack(spikes, dim=0), v


def fused_temporal_lif_avgpool_linear_torch(
    x_seq,
    v_init,
    fc_weight,
    fc_bias,
    v_threshold: float,
    v_reset: float,
    tau: float,
    detach_reset: bool,
):
    if x_seq.dim() != 5:
        raise RuntimeError(f"fused_temporal_lif_avgpool_linear requires x_seq [T,N,C,H,W], got dim={x_seq.dim()}")
    v = v_init
    out_sum = None
    for t in range(int(x_seq.shape[0])):
        spike, v = lif_forward_state_torch(x_seq[t], v, v_threshold, v_reset, tau, detach_reset)
        pooled = F.adaptive_avg_pool2d(spike, (1, 1)).flatten(1)
        logits = F.linear(pooled, fc_weight, fc_bias if isinstance(fc_bias, torch.Tensor) and fc_bias.numel() > 0 else None)
        out_sum = logits if out_sum is None else out_sum + logits
    return out_sum, v


# Deprecated compatibility alias for older callers.
fused_temporal_lif_tail_torch = fused_temporal_lif_avgpool_linear_torch


def _conv2d_output_shape(x, weight, stride, padding, dilation) -> Tuple[int, int, int, int]:
    batch, _, height, width = x.shape
    out_channels = weight.shape[0]
    kernel_h = weight.shape[2]
    kernel_w = weight.shape[3]
    out_h = (height + 2 * padding[0] - dilation[0] * (kernel_h - 1) - 1) // stride[0] + 1
    out_w = (width + 2 * padding[1] - dilation[1] * (kernel_w - 1) - 1) // stride[1] + 1
    return batch, out_channels, out_h, out_w


def _lif_forward_state_meta(x, v_prev, v_threshold: float, v_reset: float, tau: float, detach_reset: bool):
    return x.new_empty(x.shape), x.new_empty(x.shape)


def _fused_conv_lif_state_meta(
    x,
    weight,
    bias,
    v_prev,
    stride,
    padding,
    dilation,
    groups: int,
    v_threshold: float,
    v_reset: float,
    tau: float,
    detach_reset: bool,
):
    out_shape = _conv2d_output_shape(x, weight, stride, padding, dilation)
    out = x.new_empty(out_shape)
    return out, out.new_empty(out_shape)


def _fused_temporal_conv_lif_state_meta(
    xs,
    weight,
    bias,
    v_init,
    stride,
    padding,
    dilation,
    groups: int,
    v_threshold: float,
    v_reset: float,
    tau: float,
    detach_reset: bool,
):
    if len(xs) == 0:
        raise RuntimeError("fused_temporal_conv_lif_state requires at least one input tensor")
    out_shape = _conv2d_output_shape(xs[0], weight, stride, padding, dilation)
    spike_stack = xs[0].new_empty((len(xs),) + tuple(out_shape))
    v_final = xs[0].new_empty(out_shape)
    return spike_stack, v_final


def _fused_temporal_conv_lif_state_batched_tn_meta(
    xs,
    weight,
    bias,
    v_init,
    stride,
    padding,
    dilation,
    groups: int,
    v_threshold: float,
    v_reset: float,
    tau: float,
    detach_reset: bool,
):
    if len(xs) == 0:
        raise RuntimeError("fused_temporal_conv_lif_state_batched_tn requires at least one input tensor")
    out_shape = _conv2d_output_shape(xs[0], weight, stride, padding, dilation)
    n, c, h, w = tuple(out_shape)
    spike_batched = xs[0].new_empty((len(xs) * n, c, h, w))
    v_final = xs[0].new_empty(out_shape)
    return spike_batched, v_final


def _fused_temporal_conv_add_lif_state_meta(
    xs,
    residuals,
    weight,
    bias,
    v_init,
    stride,
    padding,
    dilation,
    groups: int,
    v_threshold: float,
    v_reset: float,
    tau: float,
    detach_reset: bool,
):
    if len(xs) == 0:
        raise RuntimeError("fused_temporal_conv_add_lif_state requires at least one input tensor")
    out_shape = _conv2d_output_shape(xs[0], weight, stride, padding, dilation)
    spike_stack = xs[0].new_empty((len(xs),) + tuple(out_shape))
    v_final = xs[0].new_empty(out_shape)
    return spike_stack, v_final


def _fused_temporal_conv_add_lif_state_batched_tn_meta(
    xs,
    residuals,
    weight,
    bias,
    v_init,
    stride,
    padding,
    dilation,
    groups: int,
    v_threshold: float,
    v_reset: float,
    tau: float,
    detach_reset: bool,
):
    if len(xs) == 0:
        raise RuntimeError("fused_temporal_conv_add_lif_state_batched_tn requires at least one input tensor")
    out_shape = _conv2d_output_shape(xs[0], weight, stride, padding, dilation)
    n, c, h, w = tuple(out_shape)
    spike_batched = xs[0].new_empty((len(xs) * n, c, h, w))
    v_final = xs[0].new_empty(out_shape)
    return spike_batched, v_final


def _fused_temporal_lif_state_meta(
    x_seq,
    v_init,
    v_threshold: float,
    v_reset: float,
    tau: float,
    detach_reset: bool,
):
    if x_seq.dim() != 5:
        raise RuntimeError(f"fused_temporal_lif_state requires x_seq [T,N,C,H,W], got dim={x_seq.dim()}")
    return x_seq.new_empty(x_seq.shape), x_seq.new_empty(x_seq.shape[1:])


def _fused_temporal_lif_avgpool_linear_meta(
    x_seq,
    v_init,
    fc_weight,
    fc_bias,
    v_threshold: float,
    v_reset: float,
    tau: float,
    detach_reset: bool,
):
    if x_seq.dim() != 5:
        raise RuntimeError(f"fused_temporal_lif_avgpool_linear requires x_seq [T,N,C,H,W], got dim={x_seq.dim()}")
    return x_seq.new_empty((x_seq.shape[1], fc_weight.shape[0])), x_seq.new_empty(x_seq.shape[1:])


def _lif_forward_state_impl(x, v_prev, v_threshold: float, v_reset: float, tau: float, detach_reset: bool):
    return lif_forward_state_torch(x, v_prev, v_threshold, v_reset, tau, detach_reset)


def _fused_conv_lif_state_impl(
    x,
    weight,
    bias,
    v_prev,
    stride,
    padding,
    dilation,
    groups: int,
    v_threshold: float,
    v_reset: float,
    tau: float,
    detach_reset: bool,
):
    _CALL_STATS["total"] += 1
    shape_desc = _conv_shape_desc(x, weight, bias, v_prev, stride, padding, dilation, groups)

    if _CONFIG.backend == "triton" and x.is_cuda:
        reasons = check_triton_support(
            x,
            weight,
            bias,
            v_prev,
            stride,
            padding,
            dilation,
            groups,
            v_threshold,
            v_reset,
            tau,
            detach_reset,
        )
        if not reasons:
            try:
                # Determine compute dtype based on input tensor dtype
                compute_dtype = "float16" if x.dtype == torch.float16 else "float32"
                
                result = run_triton_fused_conv_lif_state(
                    x,
                    weight,
                    bias,
                    v_prev,
                    stride,
                    padding,
                    dilation,
                    groups,
                    v_threshold,
                    v_reset,
                    tau,
                    detach_reset,
                    strict=_CONFIG.strict_triton,
                    verbose=_CONFIG.verbose,
                    compute_dtype=compute_dtype,
                )
                _CALL_STATS["triton"] += 1
                _CALL_STATS[f"single_{result.kernel_key}"] = _CALL_STATS.get(f"single_{result.kernel_key}", 0) + 1
                _record_kernel_temporal_config("single", result.kernel_key, result.kernel_temporal_config, compute_dtype)
                if _CONFIG.verbose:
                    if result.kernel_diagnostics:
                        print(
                            "[Kernel Config] "
                            f"compute_dtype={result.kernel_diagnostics.get('compute_dtype')} "
                            f"accumulator_dtype={result.kernel_diagnostics.get('accumulator_dtype')} "
                            f"membrane_dtype={result.kernel_diagnostics.get('membrane_dtype')} "
                            f"tf32_enabled={result.kernel_diagnostics.get('tf32_enabled')} "
                            f"tensor_core_usage_mode={result.kernel_diagnostics.get('tensor_core_usage_mode')}"
                        )
                    print(f"[TRITON][HIT][single][{result.kernel_key}] {shape_desc}")
                return result.spikes, result.v_next
            except Exception as exc:
                _CALL_STATS["fallback"] += 1
                _record_fallback("single_runtime_error", [f"Triton call failed: {exc}"], shape_desc)
                if _CONFIG.strict_triton:
                    raise
        else:
            _CALL_STATS["fallback"] += 1
            _record_fallback("single", reasons, shape_desc)
            if _CONFIG.strict_triton:
                raise RuntimeError("[TRITON][STRICT] " + "; ".join(reasons))
    else:
        _CALL_STATS["fallback"] += 1
        reason = "backend is not triton" if _CONFIG.backend != "triton" else "x is not CUDA"
        _record_fallback("single_dispatch", [reason], shape_desc)

    return fused_conv_lif_state_torch(
        x,
        weight,
        bias,
        v_prev,
        stride,
        padding,
        dilation,
        groups,
        v_threshold,
        v_reset,
        tau,
        detach_reset,
    )


def _fused_temporal_conv_lif_state_impl(
    xs,
    weight,
    bias,
    v_init,
    stride,
    padding,
    dilation,
    groups: int,
    v_threshold: float,
    v_reset: float,
    tau: float,
    detach_reset: bool,
):
    _CALL_STATS["total"] += 1
    _CALL_STATS["temporal_total"] += 1
    first_x = xs[0] if len(xs) > 0 else None
    shape_desc = _conv_shape_desc(
        first_x,
        weight,
        bias,
        v_init,
        stride,
        padding,
        dilation,
        groups,
        temporal_len=len(xs) if xs is not None else None,
    )

    if _CONFIG.backend == "triton" and first_x is not None and first_x.is_cuda:
        try:
            # Determine compute dtype based on input tensor dtype
            compute_dtype = "float16" if first_x.dtype == torch.float16 else "float32"
            
            result = run_triton_fused_temporal_conv_lif_state(
                xs,
                weight,
                bias,
                v_init,
                stride,
                padding,
                dilation,
                groups,
                v_threshold,
                v_reset,
                tau,
                detach_reset,
                strict=_CONFIG.strict_triton,
                verbose=_CONFIG.verbose,
                compute_dtype=compute_dtype,
            )
            _CALL_STATS["triton"] += 1
            _CALL_STATS["temporal_triton"] += 1
            _CALL_STATS[f"temporal_{result.kernel_key}"] = _CALL_STATS.get(f"temporal_{result.kernel_key}", 0) + 1
            _record_kernel_temporal_config("temporal", result.kernel_key, result.kernel_temporal_config, compute_dtype)
            if _CONFIG.verbose:
                if result.kernel_diagnostics:
                    print(
                        "[Kernel Config] "
                        f"compute_dtype={result.kernel_diagnostics.get('compute_dtype')} "
                        f"accumulator_dtype={result.kernel_diagnostics.get('accumulator_dtype')} "
                        f"membrane_dtype={result.kernel_diagnostics.get('membrane_dtype')} "
                        f"tf32_enabled={result.kernel_diagnostics.get('tf32_enabled')} "
                        f"tensor_core_usage_mode={result.kernel_diagnostics.get('tensor_core_usage_mode')}"
                    )
                print(f"[TRITON][HIT][temporal][{result.kernel_key}] {shape_desc}")
            return result.spikes, result.v_next
        except Exception as exc:
            _CALL_STATS["fallback"] += 1
            _CALL_STATS["temporal_fallback"] += 1
            _record_fallback("temporal", [str(exc)], shape_desc)
            if _strict_temporal_lif_triton_enabled():
                raise
    else:
        _CALL_STATS["fallback"] += 1
        _CALL_STATS["temporal_fallback"] += 1
        reason = "backend is not triton" if _CONFIG.backend != "triton" else "first_x is not CUDA or xs is empty"
        _record_fallback("temporal_dispatch", [reason], shape_desc)

    return fused_temporal_conv_lif_state_torch(
        xs,
        weight,
        bias,
        v_init,
        stride,
        padding,
        dilation,
        groups,
        v_threshold,
        v_reset,
        tau,
        detach_reset,
    )


def _flatten_temporal_stack_to_batched_tn(spike_stack: torch.Tensor) -> torch.Tensor:
    if spike_stack.dim() != 5:
        raise RuntimeError(f"expected spike stack [T,N,C,H,W], got shape={tuple(spike_stack.shape)}")
    return spike_stack.flatten(0, 1)


def _fused_temporal_conv_lif_state_batched_tn_impl(
    xs,
    weight,
    bias,
    v_init,
    stride,
    padding,
    dilation,
    groups: int,
    v_threshold: float,
    v_reset: float,
    tau: float,
    detach_reset: bool,
):
    _CALL_STATS["temporal_batched_output_total"] += 1
    spike_stack, v_final = _fused_temporal_conv_lif_state_impl(
        xs,
        weight,
        bias,
        v_init,
        stride,
        padding,
        dilation,
        groups,
        v_threshold,
        v_reset,
        tau,
        detach_reset,
    )
    return _flatten_temporal_stack_to_batched_tn(spike_stack), v_final


def _fused_temporal_conv_add_lif_state_impl(
    xs,
    residuals,
    weight,
    bias,
    v_init,
    stride,
    padding,
    dilation,
    groups: int,
    v_threshold: float,
    v_reset: float,
    tau: float,
    detach_reset: bool,
):
    _CALL_STATS["total"] += 1
    _CALL_STATS["temporal_total"] += 1
    _CALL_STATS["temporal_residual_total"] += 1
    first_x = xs[0] if len(xs) > 0 else None
    shape_desc = _conv_shape_desc(
        first_x,
        weight,
        bias,
        v_init,
        stride,
        padding,
        dilation,
        groups,
        temporal_len=len(xs) if xs is not None else None,
    )

    if _CONFIG.backend == "triton" and first_x is not None and first_x.is_cuda:
        try:
            compute_dtype = "float16" if first_x.dtype == torch.float16 else "float32"
            result = run_triton_fused_temporal_conv_add_lif_state(
                xs,
                residuals,
                weight,
                bias,
                v_init,
                stride,
                padding,
                dilation,
                groups,
                v_threshold,
                v_reset,
                tau,
                detach_reset,
                strict=_CONFIG.strict_triton,
                verbose=_CONFIG.verbose,
                compute_dtype=compute_dtype,
            )
            _CALL_STATS["triton"] += 1
            _CALL_STATS["temporal_triton"] += 1
            _CALL_STATS["temporal_residual_triton"] += 1
            _CALL_STATS[f"temporal_residual_{result.kernel_key}"] = (
                _CALL_STATS.get(f"temporal_residual_{result.kernel_key}", 0) + 1
            )
            _record_kernel_temporal_config("temporal_residual", result.kernel_key, result.kernel_temporal_config, compute_dtype)
            if _CONFIG.verbose:
                if result.kernel_diagnostics:
                    print(
                        "[Kernel Config] "
                        f"compute_dtype={result.kernel_diagnostics.get('compute_dtype')} "
                        f"accumulator_dtype={result.kernel_diagnostics.get('accumulator_dtype')} "
                        f"membrane_dtype={result.kernel_diagnostics.get('membrane_dtype')} "
                        f"tf32_enabled={result.kernel_diagnostics.get('tf32_enabled')} "
                        f"tensor_core_usage_mode={result.kernel_diagnostics.get('tensor_core_usage_mode')} "
                        f"residual_add={result.kernel_diagnostics.get('residual_add')}"
                    )
                print(f"[TRITON][HIT][temporal_residual][{result.kernel_key}] {shape_desc}")
            return result.spikes, result.v_next
        except Exception as exc:
            _CALL_STATS["fallback"] += 1
            _CALL_STATS["temporal_fallback"] += 1
            _CALL_STATS["temporal_residual_fallback"] += 1
            _record_fallback("temporal_residual", [str(exc)], shape_desc)
            if _strict_temporal_lif_triton_enabled():
                raise
    else:
        _CALL_STATS["fallback"] += 1
        _CALL_STATS["temporal_fallback"] += 1
        _CALL_STATS["temporal_residual_fallback"] += 1
        reason = "backend is not triton" if _CONFIG.backend != "triton" else "first_x is not CUDA or xs is empty"
        _record_fallback("temporal_residual_dispatch", [reason], shape_desc)

    return fused_temporal_conv_add_lif_state_torch(
        xs,
        residuals,
        weight,
        bias,
        v_init,
        stride,
        padding,
        dilation,
        groups,
        v_threshold,
        v_reset,
        tau,
        detach_reset,
    )


def _fused_temporal_conv_add_lif_state_batched_tn_impl(
    xs,
    residuals,
    weight,
    bias,
    v_init,
    stride,
    padding,
    dilation,
    groups: int,
    v_threshold: float,
    v_reset: float,
    tau: float,
    detach_reset: bool,
):
    _CALL_STATS["temporal_batched_output_total"] += 1
    spike_stack, v_final = _fused_temporal_conv_add_lif_state_impl(
        xs,
        residuals,
        weight,
        bias,
        v_init,
        stride,
        padding,
        dilation,
        groups,
        v_threshold,
        v_reset,
        tau,
        detach_reset,
    )
    return _flatten_temporal_stack_to_batched_tn(spike_stack), v_final


def _lif_shape_desc(x_seq, v_init):
    if not isinstance(x_seq, torch.Tensor):
        return "shape=<unknown>"
    return (
        f"T={int(x_seq.shape[0]) if x_seq.dim() > 0 else '<unknown>'}, "
        f"x_seq={tuple(x_seq.shape)}, v={_shape_tuple(v_init)}, "
        f"dtype={x_seq.dtype}, device={x_seq.device}"
    )


def _fused_temporal_lif_state_impl(
    x_seq,
    v_init,
    v_threshold: float,
    v_reset: float,
    tau: float,
    detach_reset: bool,
):
    _CALL_STATS["total"] += 1
    _CALL_STATS["temporal_total"] += 1
    _CALL_STATS["temporal_lif_total"] += 1
    shape_desc = _lif_shape_desc(x_seq, v_init)

    if _CONFIG.backend == "triton" and isinstance(x_seq, torch.Tensor) and x_seq.is_cuda:
        try:
            result = run_triton_fused_temporal_lif_state(
                x_seq,
                v_init,
                v_threshold,
                v_reset,
                tau,
                detach_reset,
                strict=_CONFIG.strict_triton,
                verbose=_CONFIG.verbose,
            )
            _CALL_STATS["triton"] += 1
            _CALL_STATS["temporal_triton"] += 1
            _CALL_STATS["temporal_lif_triton"] += 1
            if _CONFIG.verbose:
                diag = result.kernel_diagnostics or {}
                print(
                    "[TRITON][HIT][temporal_lif] "
                    f"compute_dtype={diag.get('compute_dtype')} "
                    f"membrane_dtype={diag.get('membrane_dtype')} "
                    f"T={diag.get('T')} numel_per_step={diag.get('numel_per_step')} "
                    f"{shape_desc}"
                )
            return result.spikes, result.v_next
        except Exception as exc:
            _CALL_STATS["fallback"] += 1
            _CALL_STATS["temporal_fallback"] += 1
            _CALL_STATS["temporal_lif_fallback"] += 1
            _record_fallback("temporal_lif", [str(exc)], shape_desc)
            if _CONFIG.strict_triton:
                raise
    else:
        _CALL_STATS["fallback"] += 1
        _CALL_STATS["temporal_fallback"] += 1
        _CALL_STATS["temporal_lif_fallback"] += 1
        reason = "backend is not triton" if _CONFIG.backend != "triton" else "x_seq is not CUDA"
        _record_fallback("temporal_lif_dispatch", [reason], shape_desc)

    return fused_temporal_lif_state_torch(
        x_seq,
        v_init,
        v_threshold,
        v_reset,
        tau,
        detach_reset,
    )


def _fused_temporal_lif_avgpool_linear_impl(
    x_seq,
    v_init,
    fc_weight,
    fc_bias,
    v_threshold: float,
    v_reset: float,
    tau: float,
    detach_reset: bool,
):
    _CALL_STATS["total"] += 1
    _CALL_STATS["temporal_total"] += 1
    _CALL_STATS["temporal_lif_avgpool_linear_total"] += 1
    _CALL_STATS["temporal_lif_tail_total"] += 1
    shape_desc = _lif_shape_desc(x_seq, v_init)
    if _CONFIG.backend == "triton" and isinstance(x_seq, torch.Tensor) and x_seq.is_cuda:
        try:
            result = run_triton_fused_temporal_lif_avgpool_linear(
                x_seq,
                v_init,
                fc_weight,
                fc_bias,
                v_threshold,
                v_reset,
                tau,
                detach_reset,
                strict=_CONFIG.strict_triton,
                verbose=_CONFIG.verbose,
            )
            _CALL_STATS["triton"] += 1
            _CALL_STATS["temporal_triton"] += 1
            _CALL_STATS["temporal_lif_avgpool_linear_triton"] += 1
            _CALL_STATS["temporal_lif_tail_triton"] += 1
            if _CONFIG.verbose:
                print(f"[TRITON][HIT][temporal_lif_avgpool_linear] {result.kernel_diagnostics} {shape_desc}")
            return result.out_sum, result.v_next
        except Exception as exc:
            _CALL_STATS["fallback"] += 1
            _CALL_STATS["temporal_fallback"] += 1
            _CALL_STATS["temporal_lif_avgpool_linear_fallback"] += 1
            _CALL_STATS["temporal_lif_tail_fallback"] += 1
            _record_fallback("temporal_lif_avgpool_linear", [str(exc)], shape_desc)
            if _CONFIG.strict_triton:
                raise
    else:
        _CALL_STATS["fallback"] += 1
        _CALL_STATS["temporal_fallback"] += 1
        _CALL_STATS["temporal_lif_avgpool_linear_fallback"] += 1
        _CALL_STATS["temporal_lif_tail_fallback"] += 1
        reason = "backend is not triton" if _CONFIG.backend != "triton" else "x_seq is not CUDA"
        _record_fallback("temporal_lif_avgpool_linear_dispatch", [reason], shape_desc)
    return fused_temporal_lif_avgpool_linear_torch(
        x_seq,
        v_init,
        fc_weight,
        fc_bias,
        v_threshold,
        v_reset,
        tau,
        detach_reset,
    )


def register_snn_custom_ops():
    try:
        def_lib = torch.library.Library("snn_custom", "DEF")
        def_lib.define(
            "lif_forward_state("
            "Tensor x, Tensor v_prev, float v_threshold, float v_reset, float tau, bool detach_reset"
            ") -> (Tensor, Tensor)"
        )
        def_lib.define(
            "fused_conv_lif_state("
            "Tensor x, Tensor weight, Tensor bias, Tensor v_prev, int[] stride, int[] padding, int[] dilation, "
            "int groups, float v_threshold, float v_reset, float tau, bool detach_reset"
            ") -> (Tensor, Tensor)"
        )
        def_lib.define(
            "fused_temporal_conv_lif_state("
            "Tensor[] xs, Tensor weight, Tensor bias, Tensor v_init, int[] stride, int[] padding, int[] dilation, "
            "int groups, float v_threshold, float v_reset, float tau, bool detach_reset"
            ") -> (Tensor, Tensor)"
        )
        def_lib.define(
            "fused_temporal_conv_lif_state_batched_tn("
            "Tensor[] xs, Tensor weight, Tensor bias, Tensor v_init, int[] stride, int[] padding, int[] dilation, "
            "int groups, float v_threshold, float v_reset, float tau, bool detach_reset"
            ") -> (Tensor, Tensor)"
        )
        def_lib.define(
            "fused_temporal_conv_add_lif_state("
            "Tensor[] xs, Tensor[] residuals, Tensor weight, Tensor bias, Tensor v_init, "
            "int[] stride, int[] padding, int[] dilation, int groups, "
            "float v_threshold, float v_reset, float tau, bool detach_reset"
            ") -> (Tensor, Tensor)"
        )
        def_lib.define(
            "fused_temporal_conv_add_lif_state_batched_tn("
            "Tensor[] xs, Tensor[] residuals, Tensor weight, Tensor bias, Tensor v_init, "
            "int[] stride, int[] padding, int[] dilation, int groups, "
            "float v_threshold, float v_reset, float tau, bool detach_reset"
            ") -> (Tensor, Tensor)"
        )
        def_lib.define(
            "fused_temporal_lif_state("
            "Tensor x_seq, Tensor v_init, float v_threshold, float v_reset, float tau, bool detach_reset"
            ") -> (Tensor, Tensor)"
        )
        def_lib.define(
            "fused_temporal_lif_avgpool_linear("
            "Tensor x_seq, Tensor v_init, Tensor fc_weight, Tensor fc_bias, "
            "float v_threshold, float v_reset, float tau, bool detach_reset"
            ") -> (Tensor, Tensor)"
        )
        # Deprecated compatibility schema for older scripts.
        def_lib.define(
            "fused_temporal_lif_tail("
            "Tensor x_seq, Tensor v_init, Tensor fc_weight, Tensor fc_bias, "
            "float v_threshold, float v_reset, float tau, bool detach_reset"
            ") -> (Tensor, Tensor)"
        )
        TORCH_LIBRARY_HANDLES.append(def_lib)
    except RuntimeError:
        pass

    try:
        impl_lib = torch.library.Library("snn_custom", "IMPL")
        impl_lib.impl("lif_forward_state", _lif_forward_state_impl, "CPU")
        impl_lib.impl("lif_forward_state", _lif_forward_state_impl, "CUDA")
        impl_lib.impl("lif_forward_state", _lif_forward_state_meta, "Meta")
        impl_lib.impl("fused_conv_lif_state", _fused_conv_lif_state_impl, "CPU")
        impl_lib.impl("fused_conv_lif_state", _fused_conv_lif_state_impl, "CUDA")
        impl_lib.impl("fused_conv_lif_state", _fused_conv_lif_state_meta, "Meta")
        impl_lib.impl("fused_temporal_conv_lif_state", _fused_temporal_conv_lif_state_impl, "CPU")
        impl_lib.impl("fused_temporal_conv_lif_state", _fused_temporal_conv_lif_state_impl, "CUDA")
        impl_lib.impl("fused_temporal_conv_lif_state", _fused_temporal_conv_lif_state_meta, "Meta")
        impl_lib.impl("fused_temporal_conv_lif_state_batched_tn", _fused_temporal_conv_lif_state_batched_tn_impl, "CPU")
        impl_lib.impl("fused_temporal_conv_lif_state_batched_tn", _fused_temporal_conv_lif_state_batched_tn_impl, "CUDA")
        impl_lib.impl("fused_temporal_conv_lif_state_batched_tn", _fused_temporal_conv_lif_state_batched_tn_meta, "Meta")
        impl_lib.impl("fused_temporal_conv_add_lif_state", _fused_temporal_conv_add_lif_state_impl, "CPU")
        impl_lib.impl("fused_temporal_conv_add_lif_state", _fused_temporal_conv_add_lif_state_impl, "CUDA")
        impl_lib.impl("fused_temporal_conv_add_lif_state", _fused_temporal_conv_add_lif_state_meta, "Meta")
        impl_lib.impl("fused_temporal_conv_add_lif_state_batched_tn", _fused_temporal_conv_add_lif_state_batched_tn_impl, "CPU")
        impl_lib.impl("fused_temporal_conv_add_lif_state_batched_tn", _fused_temporal_conv_add_lif_state_batched_tn_impl, "CUDA")
        impl_lib.impl("fused_temporal_conv_add_lif_state_batched_tn", _fused_temporal_conv_add_lif_state_batched_tn_meta, "Meta")
        impl_lib.impl("fused_temporal_lif_state", _fused_temporal_lif_state_impl, "CPU")
        impl_lib.impl("fused_temporal_lif_state", _fused_temporal_lif_state_impl, "CUDA")
        impl_lib.impl("fused_temporal_lif_state", _fused_temporal_lif_state_meta, "Meta")
        impl_lib.impl("fused_temporal_lif_avgpool_linear", _fused_temporal_lif_avgpool_linear_impl, "CPU")
        impl_lib.impl("fused_temporal_lif_avgpool_linear", _fused_temporal_lif_avgpool_linear_impl, "CUDA")
        impl_lib.impl("fused_temporal_lif_avgpool_linear", _fused_temporal_lif_avgpool_linear_meta, "Meta")
        impl_lib.impl("fused_temporal_lif_tail", _fused_temporal_lif_avgpool_linear_impl, "CPU")
        impl_lib.impl("fused_temporal_lif_tail", _fused_temporal_lif_avgpool_linear_impl, "CUDA")
        impl_lib.impl("fused_temporal_lif_tail", _fused_temporal_lif_avgpool_linear_meta, "Meta")
        TORCH_LIBRARY_HANDLES.append(impl_lib)
    except RuntimeError:
        pass


register_snn_custom_ops()
