import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


@dataclass
class TritonTemporalLinearLIFResult:
    spikes: torch.Tensor
    v_next: torch.Tensor
    used_triton: bool
    kernel_key: str = "temporal_linear_lif"
    fallback_reason: str = ""
    kernel_temporal_config: Optional[Dict] = None
    kernel_diagnostics: Optional[Dict] = None


def strict_temporal_linear_lif_enabled(strict: bool = False) -> bool:
    return bool(strict) or os.environ.get("CHRONOS_STRICT_TEMPORAL_LINEAR_LIF_TRITON", "0") == "1"


def check_temporal_linear_lif_support(
    x_seq,
    weight,
    bias,
    v_init,
    v_threshold: float,
    v_reset: float,
    tau: float,
    detach_reset: bool,
) -> List[str]:
    reasons: List[str] = []
    if not isinstance(x_seq, torch.Tensor):
        return ["x_seq must be a tensor"]
    if not x_seq.is_cuda:
        reasons.append("unsupported_device: x_seq is not CUDA")
    if x_seq.dim() != 3:
        reasons.append(f"unsupported_rank: x_seq.dim must be 3, got {x_seq.dim()}")
    if x_seq.dtype not in (torch.float32, torch.float16):
        reasons.append(f"unsupported_dtype: x_seq dtype must be float32 or float16, got {x_seq.dtype}")
    if x_seq.dim() >= 1 and int(x_seq.shape[0]) not in (1, 2, 4, 8, 16):
        reasons.append(f"unsupported_temporal_length: T must be one of 1,2,4,8,16, got {int(x_seq.shape[0])}")

    if not isinstance(weight, torch.Tensor):
        reasons.append("weight must be a tensor")
    elif weight.dim() != 2:
        reasons.append(f"unsupported_weight_rank: weight.dim must be 2, got {weight.dim()}")
    elif weight.device != x_seq.device or weight.dtype != x_seq.dtype:
        reasons.append(f"unsupported_dtype: weight device/dtype must match x_seq, got {weight.device}/{weight.dtype}")
    elif x_seq.dim() == 3 and int(weight.shape[1]) != int(x_seq.shape[2]):
        reasons.append(f"unsupported_shape: weight in_features {int(weight.shape[1])} != x_seq {int(x_seq.shape[2])}")

    if isinstance(bias, torch.Tensor) and bias.numel() > 0:
        if weight is not None and hasattr(weight, "shape") and (bias.dim() != 1 or int(bias.shape[0]) != int(weight.shape[0])):
            reasons.append(f"unsupported_bias: bias shape {tuple(bias.shape)} does not match out_features")
        if bias.device != x_seq.device or bias.dtype != x_seq.dtype:
            reasons.append(f"unsupported_dtype: bias device/dtype must match x_seq, got {bias.device}/{bias.dtype}")

    if not isinstance(v_init, torch.Tensor):
        reasons.append("v_init must be a tensor")
    elif v_init.dim() != 0:
        if x_seq.dim() == 3 and isinstance(weight, torch.Tensor) and weight.dim() == 2:
            expected = (int(x_seq.shape[1]), int(weight.shape[0]))
            if tuple(v_init.shape) != expected:
                reasons.append(f"unsupported_membrane_state: v_init shape {tuple(v_init.shape)} does not match {expected}")
        if v_init.device != x_seq.device or v_init.dtype != x_seq.dtype:
            reasons.append(f"unsupported_dtype: v_init device/dtype must match x_seq, got {v_init.device}/{v_init.dtype}")
    if float(tau) == 0.0:
        reasons.append("unsupported_tau: tau must be nonzero")
    return reasons


def run_triton_fused_temporal_linear_lif_state(
    xs,
    weight,
    bias,
    v_init,
    v_threshold: float,
    v_reset: float,
    tau: float,
    detach_reset: bool,
    strict: bool = False,
    verbose: bool = False,
) -> TritonTemporalLinearLIFResult:
    stack_materialized = False
    if isinstance(xs, torch.Tensor):
        return run_triton_fused_temporal_linear_lif_state_packed(
            xs,
            weight,
            bias,
            v_init,
            v_threshold,
            v_reset,
            tau,
            detach_reset,
            strict=strict,
            verbose=verbose,
        )
    elif isinstance(xs, (tuple, list)) and len(xs) > 0:
        x_seq = torch.stack(list(xs), dim=0).contiguous()
        stack_materialized = True
    else:
        raise RuntimeError("fused temporal linear LIF requires a non-empty Tensor list or [T,N,F] tensor")
    return _run_triton_fused_temporal_linear_lif_state_packed_impl(
        x_seq,
        weight,
        bias,
        v_init,
        v_threshold,
        v_reset,
        tau,
        detach_reset,
        strict=strict,
        verbose=verbose,
        stack_materialized=stack_materialized,
    )


def run_triton_fused_temporal_linear_lif_state_packed(
    x_seq,
    weight,
    bias,
    v_init,
    v_threshold: float,
    v_reset: float,
    tau: float,
    detach_reset: bool,
    strict: bool = False,
    verbose: bool = False,
) -> TritonTemporalLinearLIFResult:
    return _run_triton_fused_temporal_linear_lif_state_packed_impl(
        x_seq.contiguous() if isinstance(x_seq, torch.Tensor) else x_seq,
        weight,
        bias,
        v_init,
        v_threshold,
        v_reset,
        tau,
        detach_reset,
        strict=strict,
        verbose=verbose,
        stack_materialized=False,
    )


def run_triton_fused_temporal_linear_lif_state_packed_out(
    x_seq,
    weight,
    bias,
    v_init,
    v_threshold: float,
    v_reset: float,
    tau: float,
    detach_reset: bool,
    spike_out,
    v_out,
    strict: bool = False,
    verbose: bool = False,
) -> TritonTemporalLinearLIFResult:
    return _run_triton_fused_temporal_linear_lif_state_packed_impl(
        x_seq.contiguous() if isinstance(x_seq, torch.Tensor) and not x_seq.is_contiguous() else x_seq,
        weight,
        bias,
        v_init,
        v_threshold,
        v_reset,
        tau,
        detach_reset,
        strict=strict,
        verbose=verbose,
        stack_materialized=False,
        spike_out=spike_out,
        v_out=v_out,
    )


def _run_triton_fused_temporal_linear_lif_state_packed_impl(
    x_seq,
    weight,
    bias,
    v_init,
    v_threshold: float,
    v_reset: float,
    tau: float,
    detach_reset: bool,
    strict: bool = False,
    verbose: bool = False,
    stack_materialized: bool = False,
    spike_out=None,
    v_out=None,
) -> TritonTemporalLinearLIFResult:
    reasons = check_temporal_linear_lif_support(
        x_seq,
        weight,
        bias,
        v_init,
        v_threshold,
        v_reset,
        tau,
        detach_reset,
    )
    strict = strict_temporal_linear_lif_enabled(strict)
    if reasons:
        if verbose:
            print(f"[TRITON][FALLBACK][temporal_linear_lif] {'; '.join(reasons)}")
        if strict:
            raise RuntimeError("[TRITON][STRICT][temporal_linear_lif] " + "; ".join(reasons))
        raise RuntimeError("; ".join(reasons))

    try:
        from kernels.generated_temporal_linear_lif_kernel import run_fused_temporal_linear_lif_state_kernel

        spikes, v_next, diagnostics = run_fused_temporal_linear_lif_state_kernel(
            x_seq,
            weight,
            bias,
            v_init,
            v_threshold,
            v_reset,
            tau,
            detach_reset,
            spike_seq_out=spike_out,
            v_last_out=v_out,
        )
        diagnostics = {
            "kernel_kind": "temporal_linear_lif",
            **diagnostics,
            "membrane_dtype": str(x_seq.dtype),
            "spike_dtype": str(x_seq.dtype),
            "stack_materialized": stack_materialized,
        }
        return TritonTemporalLinearLIFResult(
            spikes=spikes,
            v_next=v_next,
            used_triton=True,
            kernel_key="temporal_linear_lif",
            kernel_temporal_config=diagnostics,
            kernel_diagnostics=diagnostics,
        )
    except Exception as exc:
        if verbose:
            print(f"[TRITON][FALLBACK][temporal_linear_lif] Triton kernel call failed: {exc}")
        if strict:
            raise
        raise RuntimeError(f"temporal Linear+LIF Triton kernel call failed: {exc}") from exc
