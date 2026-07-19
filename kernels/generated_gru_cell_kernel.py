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


def _prune_gru_cell_configs(configs, named_args, **kwargs):
    total_elements = int(named_args.get("total_elements", 1))
    valid = [c for c in configs if not (total_elements < 64 * 1024 and int(c.all_kwargs()["BLOCK_SIZE"]) > 512)]
    return valid or configs[:1]


@triton.autotune(
    configs=_make_autotune_configs(),
    key=["total_elements"],
    prune_configs_by={"early_config_prune": _prune_gru_cell_configs},
    cache_results=True,
)
@triton.jit
def _fused_gru_cell_kernel(
    xproj_ptr,
    hproj_ptr,
    h_prev_ptr,
    h_out_ptr,
    total_elements: tl.constexpr,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    """xproj/hproj: [B, 3H] contiguous, gate order [r|z|n] (PyTorch GRUCell
    convention). h_prev/h_out: [B, H]. A flat offset into the [B,H]-sized
    h_prev/h_out maps to xproj/hproj's three gate slots at
    gates_base + k*H for k in (0,1,2) = (r,z,n), where gates_base re-bases
    the per-batch offset from an H-wide stride to a 3H-wide one -- same
    channel-group indexing scheme as fused_convlstm_cell.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    batch_idx = offsets // hidden_size
    within_batch = offsets % hidden_size
    gates_base = batch_idx * (3 * hidden_size) + within_batch

    xproj_r = tl.load(xproj_ptr + gates_base, mask=mask, other=0.0)
    xproj_z = tl.load(xproj_ptr + gates_base + hidden_size, mask=mask, other=0.0)
    xproj_n = tl.load(xproj_ptr + gates_base + 2 * hidden_size, mask=mask, other=0.0)
    hproj_r = tl.load(hproj_ptr + gates_base, mask=mask, other=0.0)
    hproj_z = tl.load(hproj_ptr + gates_base + hidden_size, mask=mask, other=0.0)
    hproj_n = tl.load(hproj_ptr + gates_base + 2 * hidden_size, mask=mask, other=0.0)
    h_prev = tl.load(h_prev_ptr + offsets, mask=mask, other=0.0)

    r = tl.sigmoid(xproj_r + hproj_r)
    z = tl.sigmoid(xproj_z + hproj_z)
    n = libdevice.tanh(xproj_n + r * hproj_n)
    h_t = (1.0 - z) * n + z * h_prev

    tl.store(h_out_ptr + offsets, h_t, mask=mask)


def run_fused_gru_cell_kernel(
    xproj: torch.Tensor,
    hproj: torch.Tensor,
    h_prev: torch.Tensor,
    block_size: int = None,
) -> torch.Tensor:
    if not xproj.is_cuda or not hproj.is_cuda or not h_prev.is_cuda:
        raise RuntimeError("xproj/hproj/h_prev must be CUDA tensors")
    if xproj.dim() != 2:
        raise RuntimeError(f"xproj must have shape [B,3H], got dim={xproj.dim()}")
    B, three_h = xproj.shape
    if three_h % 3 != 0:
        raise RuntimeError(f"xproj feature dim must be divisible by 3, got {three_h}")
    H = three_h // 3
    if tuple(h_prev.shape) != (B, H):
        raise RuntimeError(f"h_prev shape {tuple(h_prev.shape)} does not match expected {(B, H)}")
    if tuple(hproj.shape) != (B, three_h):
        raise RuntimeError(f"hproj shape {tuple(hproj.shape)} does not match xproj shape {tuple(xproj.shape)}")

    xproj = xproj.contiguous()
    hproj = hproj.contiguous()
    h_prev = h_prev.contiguous()
    h_out = torch.empty_like(h_prev)

    total_elements = B * H
    grid = lambda meta: (triton.cdiv(total_elements, meta["BLOCK_SIZE"]),)
    kwargs = {}
    if block_size is not None:
        kwargs["BLOCK_SIZE"] = int(block_size)
    _fused_gru_cell_kernel[grid](
        xproj,
        hproj,
        h_prev,
        h_out,
        total_elements,
        H,
        **kwargs,
    )
    return h_out


def get_gru_cell_best_config():
    best_config = getattr(_fused_gru_cell_kernel, "best_config", None)
    if best_config is None:
        return None
    values = best_config.all_kwargs()
    return {
        "kernel_key": "gru_cell",
        "BLOCK_SIZE": values.get("BLOCK_SIZE"),
        "num_warps": values.get("num_warps"),
        "num_stages": values.get("num_stages"),
    }
