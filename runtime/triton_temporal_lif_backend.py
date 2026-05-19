import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


@dataclass
class TritonTemporalLIFResult:
    spikes: torch.Tensor
    v_next: torch.Tensor
    used_triton: bool
    kernel_key: str = "temporal_lif"
    fallback_reason: str = ""
    kernel_diagnostics: Optional[Dict] = None


def strict_temporal_lif_enabled(strict: bool = False) -> bool:
    return bool(strict) or os.environ.get("CHRONOS_STRICT_TEMPORAL_LIF_TRITON", "0") == "1"


def check_temporal_lif_support(
    x_seq,
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
    if x_seq.dim() != 5:
        reasons.append(f"unsupported_rank: x_seq.dim must be 5, got {x_seq.dim()}")
    if x_seq.dtype not in (torch.float32, torch.float16):
        reasons.append(f"unsupported_dtype: x_seq dtype must be float32 or float16, got {x_seq.dtype}")
    if x_seq.dim() >= 1 and int(x_seq.shape[0]) not in (1, 2, 4, 8, 16):
        reasons.append(f"unsupported_temporal_length: T must be one of 1,2,4,8,16, got {int(x_seq.shape[0])}")
    if not isinstance(v_init, torch.Tensor):
        reasons.append("v_init must be a tensor")
    elif v_init.dim() != 0:
        if x_seq.dim() == 5 and tuple(v_init.shape) != tuple(x_seq.shape[1:]):
            reasons.append(f"unsupported_membrane_state: v_init shape {tuple(v_init.shape)} does not match {tuple(x_seq.shape[1:])}")
        if v_init.device != x_seq.device or v_init.dtype != x_seq.dtype:
            reasons.append(
                f"unsupported_dtype: v_init device/dtype must match x_seq, got {v_init.device}/{v_init.dtype}"
            )
    if float(tau) == 0.0:
        reasons.append("unsupported_tau: tau must be nonzero")
    # detach_reset is forward-value equivalent in inference; keep the argument for schema compatibility.
    return reasons


def run_triton_fused_temporal_lif_state(
    x_seq,
    v_init,
    v_threshold: float,
    v_reset: float,
    tau: float,
    detach_reset: bool,
    strict: bool = False,
    verbose: bool = False,
) -> TritonTemporalLIFResult:
    reasons = check_temporal_lif_support(x_seq, v_init, v_threshold, v_reset, tau, detach_reset)
    strict = strict_temporal_lif_enabled(strict)
    if reasons:
        if verbose:
            print(f"[TRITON][FALLBACK][temporal_lif] {'; '.join(reasons)}")
        if strict:
            raise RuntimeError("[TRITON][STRICT][temporal_lif] " + "; ".join(reasons))
        raise RuntimeError("; ".join(reasons))

    try:
        from kernels.generated_temporal_lif_kernel import run_fused_temporal_lif_state_kernel

        spikes, v_next = run_fused_temporal_lif_state_kernel(
            x_seq,
            v_init,
            v_threshold,
            v_reset,
            tau,
            detach_reset,
        )
        diagnostics = {
            "kernel_kind": "standalone_temporal_lif",
            "compute_dtype": "float16" if x_seq.dtype == torch.float16 else "float32",
            "membrane_dtype": str(x_seq.dtype),
            "spike_dtype": str(x_seq.dtype),
            "T": int(x_seq.shape[0]),
            "numel_per_step": int(x_seq[0].numel()),
        }
        return TritonTemporalLIFResult(
            spikes=spikes,
            v_next=v_next,
            used_triton=True,
            kernel_key="temporal_lif",
            kernel_diagnostics=diagnostics,
        )
    except Exception as exc:
        if verbose:
            print(f"[TRITON][FALLBACK][temporal_lif] Triton kernel call failed: {exc}")
        if strict:
            raise
        raise RuntimeError(f"temporal LIF Triton kernel call failed: {exc}") from exc
