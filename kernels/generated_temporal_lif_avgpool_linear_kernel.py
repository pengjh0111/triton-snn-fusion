from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _temporal_lif_pool_kernel(
    x_seq,
    v_init,
    pooled,
    v_last,
    total_elements: tl.constexpr,
    nc_total: tl.constexpr,
    chw: tl.constexpr,
    hw: tl.constexpr,
    channels: tl.constexpr,
    v_threshold,
    v_reset,
    tau_inv,
    T: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TAU_LE_ONE: tl.constexpr,
    SOFT_RESET: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    c = (offsets % chw) // hw
    n = offsets // chw
    nc_offsets = n * channels + c
    pool_scale = 1.0 / hw

    v = tl.load(v_init + offsets, mask=mask, other=0.0)
    for t in tl.static_range(0, T):
        x = tl.load(x_seq + t * total_elements + offsets, mask=mask, other=0.0)
        if TAU_LE_ONE:
            v_before_spike = v + x
        else:
            v_before_spike = v + (x - v) * tau_inv

        pred = v_before_spike >= v_threshold
        spike = pred.to(tl.float32)
        if SOFT_RESET:
            v = v_before_spike - spike * v_threshold
        else:
            v = tl.where(pred, v_before_spike * 0.0 + v_reset, v_before_spike)
        tl.atomic_add(pooled + t * nc_total + nc_offsets, spike * pool_scale, mask=mask, sem="relaxed")

    tl.store(v_last + offsets, v, mask=mask)


@triton.jit
def _temporal_lif_avgpool_linear_linear_kernel(
    pooled,
    fc_weight,
    fc_bias,
    out_sum,
    n_classes: tl.constexpr,
    batch: tl.constexpr,
    channels: tl.constexpr,
    T: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0)
    k_offsets = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    c_offsets = tl.arange(0, BLOCK_C)
    k_mask = k_offsets < n_classes

    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for t in tl.static_range(0, T):
        for c_start in tl.static_range(0, channels, BLOCK_C):
            cs = c_start + c_offsets
            c_mask = cs < channels
            p = tl.load(pooled + (t * batch + n) * channels + cs, mask=c_mask, other=0.0)
            w = tl.load(fc_weight + k_offsets[:, None] * channels + cs[None, :], mask=k_mask[:, None] & c_mask[None, :], other=0.0)
            acc += tl.sum(w * p[None, :], axis=1)

    bias = tl.load(fc_bias + k_offsets, mask=k_mask, other=0.0)
    acc += bias * T
    tl.store(out_sum + n * n_classes + k_offsets, acc, mask=k_mask)


@triton.jit
def _temporal_lif_avgpool_linear_linear_kernel_nobias(
    pooled,
    fc_weight,
    out_sum,
    n_classes: tl.constexpr,
    batch: tl.constexpr,
    channels: tl.constexpr,
    T: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0)
    k_offsets = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    c_offsets = tl.arange(0, BLOCK_C)
    k_mask = k_offsets < n_classes

    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for t in tl.static_range(0, T):
        for c_start in tl.static_range(0, channels, BLOCK_C):
            cs = c_start + c_offsets
            c_mask = cs < channels
            p = tl.load(pooled + (t * batch + n) * channels + cs, mask=c_mask, other=0.0)
            w = tl.load(fc_weight + k_offsets[:, None] * channels + cs[None, :], mask=k_mask[:, None] & c_mask[None, :], other=0.0)
            acc += tl.sum(w * p[None, :], axis=1)

    tl.store(out_sum + n * n_classes + k_offsets, acc, mask=k_mask)


def _block_size(total_elements: int) -> int:
    return 512 if total_elements < 512 * 1024 else 1024


def run_fused_temporal_lif_avgpool_linear_kernel(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    fc_weight: torch.Tensor,
    fc_bias: torch.Tensor,
    v_threshold: float,
    v_reset: float,
    tau: float,
    detach_reset: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if x_seq.dim() != 5:
        raise RuntimeError(f"x_seq must be [T,N,C,H,W], got {tuple(x_seq.shape)}")
    if not x_seq.is_cuda:
        raise RuntimeError("x_seq must be CUDA")
    if x_seq.dtype not in (torch.float32, torch.float16):
        raise RuntimeError(f"unsupported dtype {x_seq.dtype}")
    if fc_weight.dim() != 2:
        raise RuntimeError("fc_weight must be [classes, channels]")
    x_seq = x_seq.contiguous()
    if v_init.dim() == 0:
        v_init = torch.zeros_like(x_seq[0])
    else:
        v_init = v_init.contiguous()
    fc_weight = fc_weight.contiguous()
    if fc_bias is not None and fc_bias.numel() > 0:
        fc_bias = fc_bias.contiguous()

    T, N, C, H, W = [int(v) for v in x_seq.shape]
    classes = int(fc_weight.shape[0])
    if int(fc_weight.shape[1]) != C:
        raise RuntimeError(f"fc_weight in_features {fc_weight.shape[1]} does not match channels {C}")
    if T not in (1, 2, 4, 8, 16):
        raise RuntimeError(f"unsupported T={T}")
    if tuple(v_init.shape) != (N, C, H, W):
        raise RuntimeError(f"v_init shape {tuple(v_init.shape)} does not match {(N, C, H, W)}")

    pooled = torch.empty((T, N, C), device=x_seq.device, dtype=torch.float32)
    pooled.zero_()
    v_last = torch.empty_like(x_seq[0])
    out_sum = torch.empty((N, classes), device=x_seq.device, dtype=x_seq.dtype)
    total_elements = int(x_seq[0].numel())
    block = _block_size(total_elements)
    tau_inv = 1.0 if float(tau) == 0.0 else 1.0 / float(tau)

    _temporal_lif_pool_kernel[(triton.cdiv(total_elements, block),)](
        x_seq,
        v_init,
        pooled,
        v_last,
        total_elements,
        N * C,
        C * H * W,
        H * W,
        C,
        float(v_threshold),
        float(v_reset),
        float(tau_inv),
        T=T,
        BLOCK_SIZE=block,
        TAU_LE_ONE=float(tau) <= 1.0,
        SOFT_RESET=float(v_reset) < 0.0,
    )

    grid = (N, triton.cdiv(classes, 32))
    if fc_bias is not None and fc_bias.numel() > 0:
        _temporal_lif_avgpool_linear_linear_kernel[grid](
            pooled,
            fc_weight,
            fc_bias,
            out_sum,
            classes,
            N,
            C,
            T=T,
            BLOCK_K=32,
            BLOCK_C=triton.next_power_of_2(C),
        )
    else:
        _temporal_lif_avgpool_linear_linear_kernel_nobias[grid](
            pooled,
            fc_weight,
            out_sum,
            classes,
            N,
            C,
            T=T,
            BLOCK_K=32,
            BLOCK_C=triton.next_power_of_2(C),
        )
    return out_sum, v_last
