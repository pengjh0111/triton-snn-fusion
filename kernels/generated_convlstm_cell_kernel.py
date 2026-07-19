from typing import Tuple

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice


def _make_autotune_configs():
    configs = []
    for block_size, num_warps in (
        (128, 2),
        (256, 4),
        (512, 4),
        (1024, 4),
        (1024, 8),
    ):
        configs.append(
            triton.Config(
                {"BLOCK_SIZE": block_size},
                num_warps=num_warps,
                num_stages=2,
            )
        )
    return configs


def _prune_convlstm_cell_configs(configs, named_args, **kwargs):
    total_elements = int(named_args.get("total_elements", 1))
    valid = [c for c in configs if not (total_elements < 64 * 1024 and int(c.all_kwargs()["BLOCK_SIZE"]) > 512)]
    return valid or configs[:1]


@triton.autotune(
    configs=_make_autotune_configs(),
    key=["total_elements"],
    prune_configs_by={"early_config_prune": _prune_convlstm_cell_configs},
    cache_results=True,
)
@triton.jit
def _fused_convlstm_cell_kernel(
    gates_sum_ptr,
    c_prev_ptr,
    h_out_ptr,
    c_out_ptr,
    total_elements: tl.constexpr,
    channel_size,
    BLOCK_SIZE: tl.constexpr,
):
    """gates_sum: [B, 4C, H, W] contiguous NCHW (i|f|g|o chunked along dim=1,
    matching torch.chunk(xproj+hproj, 4, dim=1)). c_prev/h_out/c_out: [B,C,H,W].
    channel_size = C*H*W; a flat offset into the [B,C,H,W]-sized output maps
    to gates_sum's four channel-groups at gates_base + k*channel_size for
    k in (0,1,2,3) = (i,f,g,o), where gates_base re-bases the per-batch
    offset from a C-wide batch stride to a 4C-wide one.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    batch_idx = offsets // channel_size
    within_batch = offsets % channel_size
    gates_base = batch_idx * (4 * channel_size) + within_batch

    i_gate = tl.load(gates_sum_ptr + gates_base, mask=mask, other=0.0)
    f_gate = tl.load(gates_sum_ptr + gates_base + channel_size, mask=mask, other=0.0)
    g_gate = tl.load(gates_sum_ptr + gates_base + 2 * channel_size, mask=mask, other=0.0)
    o_gate = tl.load(gates_sum_ptr + gates_base + 3 * channel_size, mask=mask, other=0.0)
    c_prev = tl.load(c_prev_ptr + offsets, mask=mask, other=0.0)

    i_sig = tl.sigmoid(i_gate)
    f_sig = tl.sigmoid(f_gate)
    o_sig = tl.sigmoid(o_gate)
    g_tanh = libdevice.tanh(g_gate)

    c_t = f_sig * c_prev + i_sig * g_tanh
    h_t = o_sig * libdevice.tanh(c_t)

    tl.store(c_out_ptr + offsets, c_t, mask=mask)
    tl.store(h_out_ptr + offsets, h_t, mask=mask)


def run_fused_convlstm_cell_kernel(
    gates_sum: torch.Tensor,
    c_prev: torch.Tensor,
    block_size: int = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not gates_sum.is_cuda or not c_prev.is_cuda:
        raise RuntimeError("gates_sum and c_prev must be CUDA tensors")
    if gates_sum.dim() != 4:
        raise RuntimeError(f"gates_sum must have shape [B,4C,H,W], got dim={gates_sum.dim()}")
    B, four_c, H, W = gates_sum.shape
    if four_c % 4 != 0:
        raise RuntimeError(f"gates_sum channel dim must be divisible by 4, got {four_c}")
    C = four_c // 4
    if tuple(c_prev.shape) != (B, C, H, W):
        raise RuntimeError(f"c_prev shape {tuple(c_prev.shape)} does not match expected {(B, C, H, W)}")
    if gates_sum.dtype != c_prev.dtype or gates_sum.device != c_prev.device:
        raise RuntimeError("gates_sum/c_prev dtype and device must match")

    gates_sum = gates_sum.contiguous()
    c_prev = c_prev.contiguous()
    h_out = torch.empty_like(c_prev)
    c_out = torch.empty_like(c_prev)

    channel_size = C * H * W
    total_elements = B * channel_size

    grid = lambda meta: (triton.cdiv(total_elements, meta["BLOCK_SIZE"]),)
    kwargs = {}
    if block_size is not None:
        kwargs["BLOCK_SIZE"] = int(block_size)
    _fused_convlstm_cell_kernel[grid](
        gates_sum,
        c_prev,
        h_out,
        c_out,
        total_elements,
        channel_size,
        **kwargs,
    )
    return h_out, c_out


def get_convlstm_cell_best_config():
    best_config = getattr(_fused_convlstm_cell_kernel, "best_config", None)
    if best_config is None:
        return None
    values = best_config.all_kwargs()
    return {
        "kernel_key": "convlstm_cell",
        "BLOCK_SIZE": values.get("BLOCK_SIZE"),
        "num_warps": values.get("num_warps"),
        "num_stages": values.get("num_stages"),
    }
