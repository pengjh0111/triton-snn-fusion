from typing import Tuple

import torch
import triton
import triton.language as tl


def _make_autotune_configs():
    configs = []
    for block_size, num_warps in (
        (128, 4),
        (256, 4),
        (512, 4),
        (1024, 4),
    ):
        configs.append(
            triton.Config(
                {"BLOCK_SIZE": block_size},
                num_warps=num_warps,
                num_stages=2,
            )
        )
    return configs


def _prune_temporal_lif_configs(configs, named_args, **kwargs):
    T = int(named_args.get("T", 1))
    total_elements = int(named_args.get("total_elements", 1))
    valid = []
    for config in configs:
        values = config.all_kwargs()
        if total_elements < 64 * 1024 and int(values["BLOCK_SIZE"]) > 512:
            continue
        valid.append(config)
    return valid or configs[:1]


def _temporal_lif_config_dict(best_config, *, T, total_elements, dtype):
    if best_config is None:
        return None
    values = best_config.all_kwargs()
    return {
        "kernel_key": "temporal_lif",
        "BLOCK_SIZE": values.get("BLOCK_SIZE"),
        "num_warps": values.get("num_warps"),
        "num_stages": values.get("num_stages"),
        "T": int(T),
        "total_elements": int(total_elements),
        "dtype": str(dtype),
    }


@triton.autotune(
    configs=_make_autotune_configs(),
    key=["total_elements", "T"],
    prune_configs_by={"early_config_prune": _prune_temporal_lif_configs},
    cache_results=True,
)
@triton.jit
def _fused_temporal_lif_state_kernel(
    x_seq,
    v_init,
    spike_seq,
    v_last,
    total_elements: tl.constexpr,
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

        tl.store(spike_seq + t * total_elements + offsets, spike, mask=mask)

    tl.store(v_last + offsets, v, mask=mask)


def run_fused_temporal_lif_state_kernel(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    v_threshold: float,
    v_reset: float,
    tau: float,
    detach_reset: bool,
    block_size: int = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if x_seq.dim() != 5:
        raise RuntimeError(f"x_seq must have shape [T, N, C, H, W], got dim={x_seq.dim()}")
    if not x_seq.is_cuda:
        raise RuntimeError("x_seq must be a CUDA tensor")
    if x_seq.dtype not in (torch.float32, torch.float16):
        raise RuntimeError(f"x_seq dtype must be float32 or float16, got {x_seq.dtype}")
    if bool(detach_reset):
        # detach_reset only affects autograd; this forward-only kernel has no backward.
        pass

    x_seq = x_seq.contiguous()
    if v_init.dim() == 0:
        v_init = torch.zeros_like(x_seq[0])
    elif tuple(v_init.shape) != tuple(x_seq.shape[1:]):
        raise RuntimeError(f"v_init shape {tuple(v_init.shape)} does not match x_seq[0] shape {tuple(x_seq.shape[1:])}")
    elif v_init.device != x_seq.device or v_init.dtype != x_seq.dtype:
        raise RuntimeError(f"v_init device/dtype must match x_seq, got {v_init.device}/{v_init.dtype}")
    else:
        v_init = v_init.contiguous()

    spike_seq = torch.empty_like(x_seq)
    v_last = torch.empty_like(x_seq[0])
    total_elements = int(x_seq[0].numel())
    T = int(x_seq.shape[0])
    if T not in (1, 2, 4, 8, 16):
        raise RuntimeError(f"unsupported temporal length T={T}; expected one of 1,2,4,8,16")

    tau_value = float(tau)
    tau_inv = 1.0 if tau_value == 0.0 else 1.0 / tau_value
    grid = lambda meta: (triton.cdiv(total_elements, meta["BLOCK_SIZE"]),)
    kwargs = {}
    if block_size is not None:
        kwargs["BLOCK_SIZE"] = int(block_size)
    _fused_temporal_lif_state_kernel[grid](
        x_seq,
        v_init,
        spike_seq,
        v_last,
        total_elements,
        float(v_threshold),
        float(v_reset),
        float(tau_inv),
        T=T,
        TAU_LE_ONE=tau_value <= 1.0,
        SOFT_RESET=float(v_reset) < 0.0,
        **kwargs,
    )
    return spike_seq, v_last


def get_temporal_lif_best_config(*, T: int = None, total_elements: int = None, dtype=None):
    best_config = getattr(_fused_temporal_lif_state_kernel, "best_config", None)
    return _temporal_lif_config_dict(
        best_config,
        T=T or 0,
        total_elements=total_elements or 0,
        dtype=dtype,
    )
