from typing import Tuple

import torch
import triton
import triton.language as tl


TEMPORAL_POW2_CANDIDATES = (1, 2, 4, 8, 16)


def _make_autotune_configs():
    configs = []
    for btile_t, reuse_groups in (
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
        (4, 1),
        (4, 2),
        (8, 1),
        (16, 1),
    ):
        for block_m, block_n, block_k, num_warps in (
            (8, 32, 64, 4),
            (16, 32, 64, 4),
            (16, 64, 64, 4),
            (16, 64, 128, 4),
            (32, 32, 64, 4),
        ):
            configs.append(
                triton.Config(
                    {
                        "BTILE_T": btile_t,
                        "REUSE_GROUPS": reuse_groups,
                        "BLOCK_M": block_m,
                        "BLOCK_N": block_n,
                        "BLOCK_K": block_k,
                    },
                    num_warps=num_warps,
                    num_stages=3,
                )
            )
    return configs


def _prune_temporal_linear_lif_configs(configs, named_args, **kwargs):
    T = int(named_args.get("T", 1))
    in_features = int(named_args.get("in_features", 1))
    out_features = int(named_args.get("out_features", 1))
    valid = []
    for config in configs:
        values = config.all_kwargs()
        temporal_window = int(values["BTILE_T"]) * int(values["REUSE_GROUPS"])
        if temporal_window > T:
            continue
        if T % temporal_window != 0:
            continue
        if int(values["BLOCK_K"]) > triton.next_power_of_2(in_features) * 2:
            continue
        if out_features < 1024 and int(values["BLOCK_N"]) > 64:
            continue
        valid.append(config)
    return valid or configs[:1]


def _temporal_linear_lif_config_dict(best_config, *, T, n_batches, in_features, out_features, dtype):
    if best_config is None:
        return None
    values = best_config.all_kwargs()
    btile_t = values.get("BTILE_T")
    reuse_groups = values.get("REUSE_GROUPS")
    window = int(btile_t) * int(reuse_groups) if btile_t is not None and reuse_groups is not None else None
    return {
        "kernel_key": "temporal_linear_lif",
        "BLOCK_M": values.get("BLOCK_M"),
        "BLOCK_N": values.get("BLOCK_N"),
        "BLOCK_K": values.get("BLOCK_K"),
        "BTILE_T": btile_t,
        "REUSE_GROUPS": reuse_groups,
        "kernel_temporal_window": window,
        "num_warps": values.get("num_warps"),
        "num_stages": values.get("num_stages"),
        "T": int(T),
        "n_batches": int(n_batches),
        "in_features": int(in_features),
        "out_features": int(out_features),
        "dtype": str(dtype),
    }


@triton.autotune(
    configs=_make_autotune_configs(),
    key=["n_batches", "in_features", "out_features", "T", "USE_TF32", "AUTOTUNE_VERSION"],
    prune_configs_by={"early_config_prune": _prune_temporal_linear_lif_configs},
    cache_results=True,
)
@triton.jit
def _fused_temporal_linear_lif_state_kernel(
    x_seq,
    weight,
    bias,
    v_init,
    spike_seq,
    v_last,
    n_batches: tl.constexpr,
    in_features: tl.constexpr,
    out_features: tl.constexpr,
    v_threshold,
    v_reset,
    tau_inv,
    T: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BTILE_T: tl.constexpr,
    REUSE_GROUPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    TAU_LE_ONE: tl.constexpr,
    SOFT_RESET: tl.constexpr,
    USE_TF32: tl.constexpr,
    V_INIT_IS_SCALAR: tl.constexpr,
    AUTOTUNE_VERSION: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < n_batches
    col_mask = cols < out_features

    v_offsets = rows[:, None] * out_features + cols[None, :]
    if V_INIT_IS_SCALAR:
        v = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    else:
        v = tl.load(v_init + v_offsets, mask=row_mask[:, None] & col_mask[None, :], other=0.0)

    for group_start in tl.static_range(0, T, BTILE_T * REUSE_GROUPS):
        for reuse_group in tl.static_range(0, REUSE_GROUPS):
            for local_t in tl.static_range(0, BTILE_T):
                t = group_start + reuse_group * BTILE_T + local_t
                acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                for k0 in range(0, in_features, BLOCK_K):
                    ks = k0 + tl.arange(0, BLOCK_K)
                    a = tl.load(
                        x_seq + t * n_batches * in_features + rows[:, None] * in_features + ks[None, :],
                        mask=row_mask[:, None] & (ks[None, :] < in_features),
                        other=0.0,
                    )
                    b = tl.load(
                        weight + cols[None, :] * in_features + ks[:, None],
                        mask=(ks[:, None] < in_features) & col_mask[None, :],
                        other=0.0,
                    )
                    if USE_TF32:
                        acc = tl.dot(a, b, acc, input_precision="tf32")
                    else:
                        acc = tl.dot(a, b, acc)

                if HAS_BIAS:
                    bval = tl.load(bias + cols, mask=col_mask, other=0.0)
                    acc += bval[None, :]

                if TAU_LE_ONE:
                    v_before_spike = v + acc
                else:
                    v_before_spike = v + (acc - v) * tau_inv

                pred = v_before_spike >= v_threshold
                spike = pred.to(tl.float32)
                if SOFT_RESET:
                    v = v_before_spike - spike * v_threshold
                else:
                    v = tl.where(pred, v_before_spike * 0.0 + v_reset, v_before_spike)

                tl.store(
                    spike_seq + t * n_batches * out_features + v_offsets,
                    spike,
                    mask=row_mask[:, None] & col_mask[None, :],
                )

    tl.store(v_last + v_offsets, v, mask=row_mask[:, None] & col_mask[None, :])


_fused_temporal_linear_lif_state_kernel_fixed = _fused_temporal_linear_lif_state_kernel.fn


def _default_fixed_config(T: int, n_batches: int, in_features: int, out_features: int):
    btile_t, reuse_groups = 1, 1
    if out_features >= 2048 or in_features >= 4096:
        block_m, block_n, block_k, num_warps = 16, 32, 64, 4
    else:
        block_m, block_n, block_k, num_warps = 16, 64, 64, 4
    return {
        "BTILE_T": btile_t,
        "REUSE_GROUPS": reuse_groups,
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "num_warps": num_warps,
        "num_stages": 3,
    }


def run_fused_temporal_linear_lif_state_kernel(
    x_seq: torch.Tensor,
    weight: torch.Tensor,
    bias,
    v_init: torch.Tensor,
    v_threshold: float,
    v_reset: float,
    tau: float,
    detach_reset: bool,
    use_autotune: bool = True,
    fixed_config: dict = None,
    spike_seq_out: torch.Tensor = None,
    v_last_out: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    if x_seq.dim() != 3:
        raise RuntimeError(f"x_seq must have shape [T, N, in_features], got dim={x_seq.dim()}")
    if weight.dim() != 2:
        raise RuntimeError(f"weight must have shape [out_features, in_features], got dim={weight.dim()}")
    if not x_seq.is_cuda:
        raise RuntimeError("x_seq must be a CUDA tensor")
    if x_seq.dtype not in (torch.float32, torch.float16):
        raise RuntimeError(f"x_seq dtype must be float32 or float16, got {x_seq.dtype}")
    if weight.device != x_seq.device or weight.dtype != x_seq.dtype:
        raise RuntimeError(f"weight device/dtype must match x_seq, got {weight.device}/{weight.dtype}")
    if bool(detach_reset):
        pass

    T = int(x_seq.shape[0])
    n_batches = int(x_seq.shape[1])
    in_features = int(x_seq.shape[2])
    out_features = int(weight.shape[0])
    if int(weight.shape[1]) != in_features:
        raise RuntimeError(f"weight in_features {int(weight.shape[1])} does not match x_seq {in_features}")
    if T not in (1, 2, 4, 8, 16):
        raise RuntimeError(f"unsupported temporal length T={T}; expected one of 1,2,4,8,16")

    x_seq = x_seq.contiguous()
    weight = weight.contiguous()
    has_bias = isinstance(bias, torch.Tensor) and bias.numel() > 0
    if has_bias:
        if bias.dim() != 1 or int(bias.shape[0]) != out_features:
            raise RuntimeError(f"bias shape must be [{out_features}], got {tuple(bias.shape)}")
        if bias.device != x_seq.device or bias.dtype != x_seq.dtype:
            raise RuntimeError(f"bias device/dtype must match x_seq, got {bias.device}/{bias.dtype}")
        bias = bias.contiguous()
    else:
        bias = weight

    v_init_is_scalar = v_init.dim() == 0
    if v_init_is_scalar:
        if v_init.device != x_seq.device or v_init.dtype != x_seq.dtype:
            raise RuntimeError(f"v_init scalar device/dtype must match x_seq, got {v_init.device}/{v_init.dtype}")
        v_init = v_init.contiguous()
    elif tuple(v_init.shape) != (n_batches, out_features):
        raise RuntimeError(f"v_init shape {tuple(v_init.shape)} does not match {(n_batches, out_features)}")
    elif v_init.device != x_seq.device or v_init.dtype != x_seq.dtype:
        raise RuntimeError(f"v_init device/dtype must match x_seq, got {v_init.device}/{v_init.dtype}")
    else:
        v_init = v_init.contiguous()

    expected_spike_shape = (T, n_batches, out_features)
    expected_v_shape = (n_batches, out_features)
    if spike_seq_out is None:
        spike_seq = torch.empty(expected_spike_shape, device=x_seq.device, dtype=x_seq.dtype)
    else:
        if tuple(spike_seq_out.shape) != expected_spike_shape:
            raise RuntimeError(f"spike_seq_out shape {tuple(spike_seq_out.shape)} does not match {expected_spike_shape}")
        if spike_seq_out.device != x_seq.device or spike_seq_out.dtype != x_seq.dtype:
            raise RuntimeError("spike_seq_out device/dtype must match x_seq")
        spike_seq = spike_seq_out
    if v_last_out is None:
        v_last = torch.empty(expected_v_shape, device=x_seq.device, dtype=x_seq.dtype)
    else:
        if tuple(v_last_out.shape) != expected_v_shape:
            raise RuntimeError(f"v_last_out shape {tuple(v_last_out.shape)} does not match {expected_v_shape}")
        if v_last_out.device != x_seq.device or v_last_out.dtype != x_seq.dtype:
            raise RuntimeError("v_last_out device/dtype must match x_seq")
        v_last = v_last_out
    tau_value = float(tau)
    tau_inv = 1.0 if tau_value == 0.0 else 1.0 / tau_value
    use_tf32 = x_seq.dtype == torch.float32
    grid = lambda meta: (triton.cdiv(n_batches, meta["BLOCK_M"]), triton.cdiv(out_features, meta["BLOCK_N"]))
    kernel = _fused_temporal_linear_lif_state_kernel if use_autotune else _fused_temporal_linear_lif_state_kernel_fixed
    launch_kwargs = {}
    selected_config = None
    if not use_autotune:
        selected_config = dict(fixed_config or _default_fixed_config(T, n_batches, in_features, out_features))
        if T % (int(selected_config["BTILE_T"]) * int(selected_config["REUSE_GROUPS"])) != 0:
            selected_config["BTILE_T"] = 1
            selected_config["REUSE_GROUPS"] = 1
        launch_kwargs.update(
            BTILE_T=int(selected_config["BTILE_T"]),
            REUSE_GROUPS=int(selected_config["REUSE_GROUPS"]),
            BLOCK_M=int(selected_config["BLOCK_M"]),
            BLOCK_N=int(selected_config["BLOCK_N"]),
            BLOCK_K=int(selected_config["BLOCK_K"]),
            num_warps=int(selected_config.get("num_warps", 4)),
            num_stages=int(selected_config.get("num_stages", 3)),
        )
    kernel[grid](
        x_seq,
        weight,
        bias,
        v_init,
        spike_seq,
        v_last,
        n_batches=n_batches,
        in_features=in_features,
        out_features=out_features,
        v_threshold=float(v_threshold),
        v_reset=float(v_reset),
        tau_inv=float(tau_inv),
        T=T,
        HAS_BIAS=has_bias,
        TAU_LE_ONE=tau_value <= 1.0,
        SOFT_RESET=float(v_reset) < 0.0,
        USE_TF32=use_tf32,
        V_INIT_IS_SCALAR=v_init_is_scalar,
        AUTOTUNE_VERSION=2,
        **launch_kwargs,
    )
    if use_autotune:
        best_config = get_temporal_linear_lif_best_config(
            T=T,
            n_batches=n_batches,
            in_features=in_features,
            out_features=out_features,
            dtype=x_seq.dtype,
        )
    else:
        best_config = _temporal_linear_lif_config_dict(
            triton.Config(
                {
                    "BTILE_T": selected_config["BTILE_T"],
                    "REUSE_GROUPS": selected_config["REUSE_GROUPS"],
                    "BLOCK_M": selected_config["BLOCK_M"],
                    "BLOCK_N": selected_config["BLOCK_N"],
                    "BLOCK_K": selected_config["BLOCK_K"],
                },
                num_warps=selected_config.get("num_warps", 4),
                num_stages=selected_config.get("num_stages", 3),
            ),
            T=T,
            n_batches=n_batches,
            in_features=in_features,
            out_features=out_features,
            dtype=x_seq.dtype,
        )
    diagnostics = {
        **(best_config or {}),
        "autotune": bool(use_autotune),
        "T": T,
        "n_batches": n_batches,
        "in_features": in_features,
        "out_features": out_features,
        "compute_dtype": "float16" if x_seq.dtype == torch.float16 else "float32",
        "tensor_core_usage_mode": "fp16" if x_seq.dtype == torch.float16 else "tf32",
    }
    return spike_seq, v_last, diagnostics


def get_temporal_linear_lif_best_config(
    *,
    T: int = None,
    n_batches: int = None,
    in_features: int = None,
    out_features: int = None,
    dtype=None,
):
    best_config = getattr(_fused_temporal_linear_lif_state_kernel, "best_config", None)
    return _temporal_linear_lif_config_dict(
        best_config,
        T=T or 0,
        n_batches=n_batches or 0,
        in_features=in_features or 0,
        out_features=out_features or 0,
        dtype=dtype,
    )
