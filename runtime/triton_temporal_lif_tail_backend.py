import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


@dataclass
class TritonTemporalLIFTailResult:
    out_sum: torch.Tensor
    v_next: torch.Tensor
    used_triton: bool
    kernel_key: str = "temporal_lif_tail"
    fallback_reason: str = ""
    kernel_diagnostics: Optional[Dict] = None


def strict_temporal_lif_tail_enabled(strict: bool = False) -> bool:
    return bool(strict) or os.environ.get("CHRONOS_STRICT_TEMPORAL_LIF_TAIL_TRITON", "0") == "1"


def check_temporal_lif_tail_support(x_seq, v_init, fc_weight, fc_bias, tau) -> List[str]:
    reasons: List[str] = []
    if not isinstance(x_seq, torch.Tensor):
        return ["x_seq must be a tensor"]
    if not x_seq.is_cuda:
        reasons.append("unsupported_device: x_seq is not CUDA")
    if x_seq.dim() != 5:
        reasons.append(f"unsupported_rank: x_seq.dim must be 5, got {x_seq.dim()}")
    if x_seq.dtype not in (torch.float32, torch.float16):
        reasons.append(f"unsupported_dtype: x_seq dtype must be float32 or float16, got {x_seq.dtype}")
    if x_seq.dim() == 5 and int(x_seq.shape[0]) not in (1, 2, 4, 8, 16):
        reasons.append(f"unsupported_temporal_length: T must be one of 1,2,4,8,16, got {int(x_seq.shape[0])}")
    if not isinstance(v_init, torch.Tensor):
        reasons.append("v_init must be a tensor")
    elif v_init.dim() != 0:
        if x_seq.dim() == 5 and tuple(v_init.shape) != tuple(x_seq.shape[1:]):
            reasons.append(f"unsupported_membrane_state: v_init shape {tuple(v_init.shape)} does not match {tuple(x_seq.shape[1:])}")
        if v_init.device != x_seq.device or v_init.dtype != x_seq.dtype:
            reasons.append("unsupported_dtype: v_init device/dtype must match x_seq")
    if not isinstance(fc_weight, torch.Tensor) or fc_weight.dim() != 2:
        reasons.append("fc_weight must be a 2D tensor")
    elif x_seq.dim() == 5:
        if int(fc_weight.shape[1]) != int(x_seq.shape[2]):
            reasons.append(f"unsupported_linear: fc in_features {fc_weight.shape[1]} must match channels {x_seq.shape[2]}")
        if fc_weight.device != x_seq.device or fc_weight.dtype != x_seq.dtype:
            reasons.append("unsupported_dtype: fc_weight device/dtype must match x_seq")
    if isinstance(fc_bias, torch.Tensor) and fc_bias.numel() > 0:
        if fc_weight is not None and fc_bias.numel() != fc_weight.shape[0]:
            reasons.append("unsupported_linear: fc_bias shape does not match out_features")
        if fc_bias.device != x_seq.device or fc_bias.dtype != x_seq.dtype:
            reasons.append("unsupported_dtype: fc_bias device/dtype must match x_seq")
    if float(tau) == 0.0:
        reasons.append("unsupported_tau: tau must be nonzero")
    return reasons


def run_triton_fused_temporal_lif_tail(
    x_seq,
    v_init,
    fc_weight,
    fc_bias,
    v_threshold: float,
    v_reset: float,
    tau: float,
    detach_reset: bool,
    strict: bool = False,
    verbose: bool = False,
) -> TritonTemporalLIFTailResult:
    reasons = check_temporal_lif_tail_support(x_seq, v_init, fc_weight, fc_bias, tau)
    strict = strict_temporal_lif_tail_enabled(strict)
    if reasons:
        if verbose:
            print(f"[TRITON][FALLBACK][temporal_lif_tail] {'; '.join(reasons)}")
        if strict:
            raise RuntimeError("[TRITON][STRICT][temporal_lif_tail] " + "; ".join(reasons))
        raise RuntimeError("; ".join(reasons))
    try:
        from kernels.generated_temporal_lif_tail_kernel import run_fused_temporal_lif_tail_kernel

        out_sum, v_next = run_fused_temporal_lif_tail_kernel(
            x_seq,
            v_init,
            fc_weight,
            fc_bias,
            v_threshold,
            v_reset,
            tau,
            detach_reset,
        )
        diagnostics = {
            "kernel_kind": "standalone_temporal_lif_tail",
            "compute_dtype": "float16" if x_seq.dtype == torch.float16 else "float32",
            "T": int(x_seq.shape[0]),
            "shape": tuple(x_seq.shape),
            "classes": int(fc_weight.shape[0]),
        }
        return TritonTemporalLIFTailResult(out_sum, v_next, True, kernel_diagnostics=diagnostics)
    except Exception as exc:
        if verbose:
            print(f"[TRITON][FALLBACK][temporal_lif_tail] Triton kernel call failed: {exc}")
        if strict:
            raise
        raise RuntimeError(f"temporal LIF tail Triton kernel call failed: {exc}") from exc
