from typing import Tuple

import torch
import triton
import triton.language as tl


def _make_autotune_configs():
    configs = []
    for block_size in (32, 64, 128):
        for num_warps in (2, 4, 8):
            configs.append(
                triton.Config(
                    {"BLOCK_C": block_size},
                    num_warps=num_warps,
                    num_stages=2,
                )
            )
    return configs


def _prune_scan_configs(configs, named_args, **kwargs):
    d_inner = int(named_args.get("d_inner", 1))
    valid = [c for c in configs if int(c.all_kwargs()["BLOCK_C"]) <= max(32, d_inner)]
    return valid or configs[:1]


@triton.autotune(
    configs=_make_autotune_configs(),
    key=["batch", "d_inner", "d_state", "T"],
    prune_configs_by={"early_config_prune": _prune_scan_configs},
    cache_results=True,
)
@triton.jit
def _fused_temporal_selective_scan_kernel(
    x_ptr,
    dt_ptr,
    b_ptr,
    c_ptr,
    a_ptr,
    d_ptr,
    h_init_ptr,
    y_ptr,
    h_final_ptr,
    batch,
    d_inner,
    T: tl.constexpr,
    D_STATE: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Selective-scan recurrence (Mamba), matching the canonical eager loop
    body exactly:
        hA = exp(dt * A)
        state = hA * state + (dt * B) * x
        y = sum(state * C, -1) + D * x
    x/dt/y: [T, batch, d_inner] contiguous. B_ssm/C_ssm: [T, batch, D_STATE].
    A: [d_inner, D_STATE]. D: [d_inner]. h_init/h_final: [batch, d_inner, D_STATE].

    grid = (batch, cdiv(d_inner, BLOCK_C)): each program owns one batch
    element's BLOCK_C-wide channel slice and keeps its [BLOCK_C, D_STATE]
    state resident in registers across the whole T loop (register residency
    across T is exactly the mechanism the fused_temporal_general conv/LIF
    kernels already use -- see kernels/benchmark_conv_lif_temporal_general.py
    -- applied here to the SSM recurrence instead of LIF membrane state).
    State accumulates in fp32 regardless of x's dtype since exp(dt*A) is
    numerically sensitive to precision loss.
    """
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    c_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offsets < d_inner

    s_offsets = tl.arange(0, D_STATE)

    a_ptrs = a_ptr + c_offsets[:, None] * D_STATE + s_offsets[None, :]
    a_block = tl.load(a_ptrs, mask=c_mask[:, None], other=0.0).to(tl.float32)
    d_block = tl.load(d_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)

    h_init_ptrs = h_init_ptr + pid_b * d_inner * D_STATE + c_offsets[:, None] * D_STATE + s_offsets[None, :]
    state = tl.load(h_init_ptrs, mask=c_mask[:, None], other=0.0).to(tl.float32)

    for t in tl.static_range(0, T):
        base_tb_c = t * batch * d_inner + pid_b * d_inner
        x_t = tl.load(x_ptr + base_tb_c + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        dt_t = tl.load(dt_ptr + base_tb_c + c_offsets, mask=c_mask, other=0.0).to(tl.float32)

        base_tb_s = t * batch * D_STATE + pid_b * D_STATE
        b_t = tl.load(b_ptr + base_tb_s + s_offsets).to(tl.float32)
        c_t = tl.load(c_ptr + base_tb_s + s_offsets).to(tl.float32)

        hA = tl.exp(dt_t[:, None] * a_block)
        state = hA * state + (dt_t[:, None] * b_t[None, :]) * x_t[:, None]
        y_t = tl.sum(state * c_t[None, :], axis=1) + d_block * x_t

        tl.store(y_ptr + base_tb_c + c_offsets, y_t, mask=c_mask)

    h_final_ptrs = h_final_ptr + pid_b * d_inner * D_STATE + c_offsets[:, None] * D_STATE + s_offsets[None, :]
    tl.store(h_final_ptrs, state, mask=c_mask[:, None])


def run_fused_temporal_selective_scan_kernel(
    x_seq: torch.Tensor,
    dt_seq: torch.Tensor,
    b_seq: torch.Tensor,
    c_seq: torch.Tensor,
    A: torch.Tensor,
    D: torch.Tensor,
    h_init: torch.Tensor,
    block_c: int = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if x_seq.dim() != 3:
        raise RuntimeError(f"x_seq must have shape [T,B,d_inner], got dim={x_seq.dim()}")
    T, batch, d_inner = x_seq.shape
    if b_seq.dim() != 3:
        raise RuntimeError(f"b_seq must have shape [T,B,d_state], got dim={b_seq.dim()}")
    d_state = b_seq.shape[2]
    if T not in (1, 2, 4, 8, 16):
        raise RuntimeError(f"unsupported temporal length T={T}; expected one of 1,2,4,8,16")
    if not x_seq.is_cuda:
        raise RuntimeError("x_seq must be a CUDA tensor")

    x_seq = x_seq.contiguous()
    dt_seq = dt_seq.contiguous()
    b_seq = b_seq.contiguous()
    c_seq = c_seq.contiguous()
    A = A.contiguous()
    D = D.contiguous()
    h_init = h_init.contiguous()

    y_seq = torch.empty_like(x_seq)
    h_final = torch.empty_like(h_init)

    grid = lambda meta: (batch, triton.cdiv(d_inner, meta["BLOCK_C"]))
    kwargs = {}
    if block_c is not None:
        kwargs["BLOCK_C"] = int(block_c)
    _fused_temporal_selective_scan_kernel[grid](
        x_seq,
        dt_seq,
        b_seq,
        c_seq,
        A,
        D,
        h_init,
        y_seq,
        h_final,
        batch,
        d_inner,
        T=T,
        D_STATE=d_state,
        **kwargs,
    )
    return y_seq, h_final


def get_selective_scan_best_config():
    best_config = getattr(_fused_temporal_selective_scan_kernel, "best_config", None)
    if best_config is None:
        return None
    values = best_config.all_kwargs()
    return {
        "kernel_key": "temporal_selective_scan",
        "BLOCK_C": values.get("BLOCK_C"),
        "num_warps": values.get("num_warps"),
        "num_stages": values.get("num_stages"),
    }
