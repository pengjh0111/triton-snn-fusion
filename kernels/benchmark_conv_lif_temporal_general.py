import argparse
import copy
import linecache
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

os.environ.setdefault("TRITON_ALWAYS_COMPILE", "1")
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(os.getcwd(), "aot_result/triton_cache"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(os.getcwd(), "aot_result/inductor_cache"))
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

import torch
import torch.nn as nn
import triton
import triton.language as tl
from spikingjelly.activation_based import functional, layer, neuron, surrogate


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

DEVICE = "cuda"
TAU = 2.0
TAU_INV = 1.0 / TAU
V_THRESHOLD = 1.0
V_RESET = 0.0
TF32_SPIKE_ERR_LIMIT = 1e-3
TF32_V_MEAN_ERR_LIMIT = 1e-3
TF32_V_MAX_ERR_LIMIT = 5e-2
TEMPORAL_POW2_CANDIDATES = (1, 2, 4, 8, 16)
TEMPORAL_AUTOTUNE_SCHEDULES = (
    (1, 1),
    (1, 2),
    (2, 1),
    (1, 4),
    (2, 2),
    (4, 1),
    (1, 8),
    (2, 4),
    (4, 2),
    (8, 1),
    (1, 16),
    (2, 8),
    (4, 4),
    (8, 2),
    (16, 1),
)
MAX_REUSE_GROUPS = 16
DEFAULT_POINTWISE_ACC_ELEMS_LIMIT = 8192


@dataclass(frozen=True)
class ProblemShape:
    timesteps: int
    batch: int
    in_channels: int
    out_channels: int
    height: int
    width: int
    kernel_size: int = 3
    pad: int = 1


BASE_SHAPES = [
    ("first layer", ProblemShape(4, 4, 3, 64, 224, 224)),
    ("mid-early", ProblemShape(4, 4, 64, 128, 56, 56)),
    ("mid", ProblemShape(4, 4, 128, 128, 28, 28)),
    ("mid-late", ProblemShape(4, 4, 256, 256, 14, 14)),
    ("late", ProblemShape(4, 4, 512, 512, 7, 7)),
]


SPATIAL_CONFIGS = [
    {"BLOCK_M": 8, "BLOCK_OC": 32, "BLOCK_K": 32, "num_warps": 4, "num_stages": 2},
    {"BLOCK_M": 16, "BLOCK_OC": 32, "BLOCK_K": 32, "num_warps": 4, "num_stages": 2},
    {"BLOCK_M": 32, "BLOCK_OC": 32, "BLOCK_K": 32, "num_warps": 4, "num_stages": 2},
    {"BLOCK_M": 8, "BLOCK_OC": 64, "BLOCK_K": 32, "num_warps": 4, "num_stages": 2},
    {"BLOCK_M": 16, "BLOCK_OC": 64, "BLOCK_K": 32, "num_warps": 4, "num_stages": 2},
    {"BLOCK_M": 32, "BLOCK_OC": 64, "BLOCK_K": 32, "num_warps": 4, "num_stages": 2},
]

AUTOTUNE_SPATIAL_CONFIGS = [
    {"BLOCK_M": 16, "BLOCK_OC": 64, "BLOCK_K": 32, "num_warps": 4, "num_stages": 2},
    {"BLOCK_M": 32, "BLOCK_OC": 64, "BLOCK_K": 32, "num_warps": 4, "num_stages": 2},
]

POINTWISE_AUTOTUNE_CONFIGS = [
    # Keep pointwise independent from regular k3/k5/k7 configs. It is still a
    # GEMM-like reduction, but it should not inherit depthwise or halo-oriented
    # tuning decisions.
    {"BLOCK_M": 16, "BLOCK_OC": 64, "BLOCK_K": 32, "num_warps": 4, "num_stages": 2},
    {"BLOCK_M": 32, "BLOCK_OC": 64, "BLOCK_K": 32, "num_warps": 4, "num_stages": 2},
    {"BLOCK_M": 16, "BLOCK_OC": 128, "BLOCK_K": 16, "num_warps": 4, "num_stages": 2},
    {"BLOCK_M": 8, "BLOCK_OC": 128, "BLOCK_K": 64, "num_warps": 2, "num_stages": 2},
    {"BLOCK_M": 16, "BLOCK_OC": 128, "BLOCK_K": 64, "num_warps": 4, "num_stages": 2},
    {"BLOCK_M": 16, "BLOCK_OC": 128, "BLOCK_K": 32, "num_warps": 4, "num_stages": 2},
    {"BLOCK_M": 32, "BLOCK_OC": 128, "BLOCK_K": 32, "num_warps": 4, "num_stages": 2},
]

DWCONV_AUTOTUNE_CONFIGS = [
    # Depthwise conv is a stencil kernel, not a GEMM kernel. Tune spatial
    # rectangle and channel tile directly so the runtime key does not inherit
    # ordinary ConvLIF's reduction-oriented BLOCK_M/BLOCK_K schedule.
    {"BLOCK_H": 16, "BLOCK_W": 16, "BLOCK_C": 16, "PIXELS_PER_THREAD": 1, "num_warps": 4, "num_stages": 4},
    {"BLOCK_H": 8, "BLOCK_W": 8, "BLOCK_C": 16, "PIXELS_PER_THREAD": 1, "num_warps": 2, "num_stages": 3},
    {"BLOCK_H": 2, "BLOCK_W": 8, "BLOCK_C": 16, "PIXELS_PER_THREAD": 1, "num_warps": 2, "num_stages": 4},
    {"BLOCK_H": 4, "BLOCK_W": 8, "BLOCK_C": 16, "PIXELS_PER_THREAD": 1, "num_warps": 1, "num_stages": 3},
    {"BLOCK_H": 4, "BLOCK_W": 8, "BLOCK_C": 16, "PIXELS_PER_THREAD": 1, "num_warps": 4, "num_stages": 3},
    {"BLOCK_H": 2, "BLOCK_W": 8, "BLOCK_C": 16, "PIXELS_PER_THREAD": 1, "num_warps": 4, "num_stages": 5},
    {"BLOCK_H": 2, "BLOCK_W": 8, "BLOCK_C": 16, "PIXELS_PER_THREAD": 1, "num_warps": 1, "num_stages": 3},
    {"BLOCK_H": 8, "BLOCK_W": 8, "BLOCK_C": 16, "PIXELS_PER_THREAD": 1, "num_warps": 4, "num_stages": 3},
]


def _make_autotune_configs():
    configs = []
    for spatial_config in AUTOTUNE_SPATIAL_CONFIGS:
        for btile_t, reuse_groups in TEMPORAL_AUTOTUNE_SCHEDULES:
            configs.append(
                triton.Config(
                    {
                        "BLOCK_M": spatial_config["BLOCK_M"],
                        "BLOCK_OC": spatial_config["BLOCK_OC"],
                        "BLOCK_K": spatial_config["BLOCK_K"],
                        "BTILE_T": btile_t,
                        "REUSE_GROUPS": reuse_groups,
                    },
                    num_warps=spatial_config["num_warps"],
                    num_stages=spatial_config["num_stages"],
                )
            )
    return configs


def _make_pointwise_autotune_configs():
    configs = []
    for spatial_config in POINTWISE_AUTOTUNE_CONFIGS:
        for btile_t, reuse_groups in TEMPORAL_AUTOTUNE_SCHEDULES:
            configs.append(
                triton.Config(
                    {
                        "BLOCK_M": spatial_config["BLOCK_M"],
                        "BLOCK_OC": spatial_config["BLOCK_OC"],
                        "BLOCK_K": spatial_config["BLOCK_K"],
                        "BTILE_T": btile_t,
                        "REUSE_GROUPS": reuse_groups,
                    },
                    num_warps=spatial_config["num_warps"],
                    num_stages=spatial_config["num_stages"],
                )
            )
    return configs


def _make_dwconv_autotune_configs():
    configs = []
    for spatial_config in DWCONV_AUTOTUNE_CONFIGS:
        for btile_t, reuse_groups in TEMPORAL_AUTOTUNE_SCHEDULES:
            configs.append(
                triton.Config(
                    {
                        "BLOCK_C": spatial_config["BLOCK_C"],
                        "BLOCK_H": spatial_config["BLOCK_H"],
                        "BLOCK_W": spatial_config["BLOCK_W"],
                        "PIXELS_PER_THREAD": spatial_config["PIXELS_PER_THREAD"],
                        "BTILE_T": btile_t,
                        "REUSE_GROUPS": reuse_groups,
                    },
                    num_warps=spatial_config["num_warps"],
                    num_stages=spatial_config["num_stages"],
                )
            )
    return configs


def _prune_temporal_configs(configs, named_args, **kwargs):
    timesteps = int(named_args["T_STEPS"])
    return [
        config
        for config in configs
        if int(config.kwargs["BTILE_T"]) * int(config.kwargs["REUSE_GROUPS"]) <= timesteps
    ]


def _config_matches_spatial(config, desired: Dict[str, int], keys: Tuple[str, ...]) -> bool:
    for key in keys:
        if int(config.kwargs[key]) != int(desired[key]):
            return False
    return (
        int(getattr(config, "num_warps", desired["num_warps"])) == int(desired["num_warps"])
        and int(getattr(config, "num_stages", desired["num_stages"])) == int(desired["num_stages"])
    )


def _pointwise_acc_elems_limit() -> int:
    raw = os.environ.get("CHRONOS_POINTWISE_ACC_ELEMS_LIMIT")
    if raw is None:
        return DEFAULT_POINTWISE_ACC_ELEMS_LIMIT
    try:
        return int(raw)
    except ValueError:
        return DEFAULT_POINTWISE_ACC_ELEMS_LIMIT


def _pointwise_acc_elems(config) -> int:
    return (
        int(config.kwargs["BTILE_T"])
        * int(config.kwargs["REUSE_GROUPS"])
        * int(config.kwargs["BLOCK_M"])
        * int(config.kwargs["BLOCK_OC"])
    )


def _prune_pointwise_configs(configs, named_args, **kwargs):
    valid = _prune_temporal_configs(configs, named_args, **kwargs)
    if not valid:
        return valid

    acc_limit = _pointwise_acc_elems_limit()
    if acc_limit > 0:
        limited = [
            config
            for config in valid
            if _pointwise_acc_elems(config) <= acc_limit
        ]
        valid = limited or valid

    in_channels = int(named_args["in_channels"])
    out_channels = int(named_args["out_channels"])
    height = int(named_args["height"])
    width = int(named_args["width"])
    desired = _pointwise_config_for_shape(in_channels, out_channels, height, width)
    filtered = [
        config
        for config in valid
        if _config_matches_spatial(config, desired, ("BLOCK_M", "BLOCK_OC", "BLOCK_K"))
    ]
    return filtered or valid


def _prune_dwconv_configs(configs, named_args, **kwargs):
    valid = _prune_temporal_configs(configs, named_args, **kwargs)
    if not valid:
        return valid

    height = int(named_args["height"])
    width = int(named_args["width"])
    out_height = int(named_args["out_height"])
    out_width = int(named_args["out_width"])
    out_channels = int(named_args["out_channels"])
    kernel_key = "depthwise_k3_s1_p1" if (height, width) == (out_height, out_width) else "depthwise_k3_s2_p1"
    desired = _depthwise_config_for_shape(kernel_key, out_channels, height, width)

    filtered = [
        config
        for config in valid
        if _config_matches_spatial(config, desired, ("BLOCK_H", "BLOCK_W", "BLOCK_C", "PIXELS_PER_THREAD"))
    ]
    return filtered or valid


def kernel_dtype_diagnostics(dtype) -> Dict[str, object]:
    if dtype in (torch.float16, "fp16", "float16", "torch.float16"):
        compute_dtype = "fp16"
        accumulator_dtype = "fp16"
        membrane_dtype = "fp16"
        tensor_core_usage_mode = "fp16_tensor_cores"
    else:
        compute_dtype = "fp32/tf32"
        accumulator_dtype = "fp32"
        membrane_dtype = "fp32"
        tensor_core_usage_mode = "tf32_tensor_cores" if torch.backends.cuda.matmul.allow_tf32 else "fp32_cuda_cores"
    return {
        "compute_dtype": compute_dtype,
        "accumulator_dtype": accumulator_dtype,
        "membrane_dtype": membrane_dtype,
        "tf32_enabled": bool(torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32),
        "tensor_core_usage_mode": tensor_core_usage_mode,
    }


def format_kernel_dtype_diagnostics(dtype) -> str:
    diag = kernel_dtype_diagnostics(dtype)
    return (
        "[Kernel Config] "
        f"compute_dtype={diag['compute_dtype']} "
        f"accumulator_dtype={diag['accumulator_dtype']} "
        f"membrane_dtype={diag['membrane_dtype']} "
        f"tf32_enabled={diag['tf32_enabled']} "
        f"tensor_core_usage_mode={diag['tensor_core_usage_mode']}"
    )


def build_input_sequence(shape: ProblemShape, dtype=torch.float32) -> torch.Tensor:
    x = torch.randn(
        shape.timesteps * shape.batch,
        shape.in_channels,
        shape.height,
        shape.width,
        device=DEVICE,
        dtype=dtype,
    )
    return x.view(shape.timesteps, shape.batch, shape.in_channels, shape.height, shape.width)


def lif_update(v_old: torch.Tensor, synaptic: torch.Tensor, v_reset: float, tau_inv: float):
    v_new = v_old + (synaptic - (v_old - v_reset)) * tau_inv
    spikes = (v_new >= V_THRESHOLD).to(v_new.dtype)
    v_next = torch.where(spikes > 0.5, torch.full_like(v_new, v_reset), v_new)
    return spikes, v_next


class ConvLIFSingleStepBaseline(nn.Module):
    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        self.conv = conv
        self.lif = neuron.LIFNode(
            tau=TAU,
            decay_input=True,
            v_threshold=V_THRESHOLD,
            v_reset=V_RESET,
            surrogate_function=surrogate.ATan(),
            step_mode="s",
        )

    def forward(self, x_seq: torch.Tensor):
        timesteps, batch, _, height, width = x_seq.shape
        out_channels = self.conv.out_channels
        spikes = x_seq.new_empty((timesteps, batch, out_channels, height, width))
        for step in range(timesteps):
            spikes[step] = self.lif(self.conv(x_seq[step]))
        return spikes, self.lif.v.clone()


class ConvLIFMultiStepBaseline(nn.Module):
    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        self.layer = nn.Sequential(
            layer.Conv2d(
                conv.in_channels,
                conv.out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                bias=(conv.bias is not None),
            ),
            neuron.LIFNode(
                tau=TAU,
                decay_input=True,
                v_threshold=V_THRESHOLD,
                v_reset=V_RESET,
                surrogate_function=surrogate.ATan(),
            ),
        )
        self.layer.to(device=conv.weight.device, dtype=conv.weight.dtype)
        self.layer[0].weight.data.copy_(conv.weight.data)
        if conv.bias is not None:
            self.layer[0].bias.data.copy_(conv.bias.data)
        functional.set_step_mode(self, step_mode="m")

    def forward(self, x_seq: torch.Tensor):
        spikes = self.layer(x_seq)
        return spikes, self.layer[1].v.clone()


def reference_conv_lif(x_seq: torch.Tensor, conv: nn.Conv2d):
    timesteps, batch, _, height, width = x_seq.shape
    membrane = torch.zeros((batch, conv.out_channels, height, width), device=x_seq.device, dtype=x_seq.dtype)
    spikes = torch.empty((timesteps, batch, conv.out_channels, height, width), device=x_seq.device, dtype=x_seq.dtype)
    for step in range(timesteps):
        spikes[step], membrane = lif_update(membrane, conv(x_seq[step]), V_RESET, TAU_INV)
    return spikes, membrane


def make_conv(shape: ProblemShape, dtype=torch.float32):
    return nn.Conv2d(
        shape.in_channels,
        shape.out_channels,
        kernel_size=shape.kernel_size,
        stride=1,
        padding=shape.pad,
        bias=True,
    ).to(DEVICE, dtype=dtype).eval()


def make_reference(x_seq: torch.Tensor, conv: nn.Conv2d):
    return reference_conv_lif(x_seq.to(torch.float32), copy.deepcopy(conv).to(DEVICE, dtype=torch.float32).eval())


def summarize_correctness(name: str, spikes: torch.Tensor, membrane: torch.Tensor, ref_spikes: torch.Tensor, ref_v: torch.Tensor):
    spikes = spikes.to(torch.float32)
    membrane = membrane.to(torch.float32)
    ref_spikes = ref_spikes.to(torch.float32)
    ref_v = ref_v.to(torch.float32)
    return {
        "name": name,
        "spike_err": (spikes != ref_spikes).float().mean().item(),
        "v_max_err": (membrane - ref_v).abs().max().item(),
        "v_mean_err": (membrane - ref_v).abs().mean().item(),
    }


def validate_summary(summary: Dict[str, float]):
    spike_ok = summary["spike_err"] < TF32_SPIKE_ERR_LIMIT
    mean_ok = summary["v_mean_err"] < TF32_V_MEAN_ERR_LIMIT
    max_ok = summary["v_max_err"] < TF32_V_MAX_ERR_LIMIT
    if not (spike_ok and mean_ok and (max_ok or summary["spike_err"] > 0.0)):
        raise AssertionError(
            f"{summary['name']} failed: spike_err={summary['spike_err']:.3e}, "
            f"v_max_err={summary['v_max_err']:.3e}, v_mean_err={summary['v_mean_err']:.3e}"
        )


def _conv_out_hw(height: int, width: int, kernel_size: int, stride: int, pad: int) -> Tuple[int, int]:
    out_height = (height + 2 * pad - (kernel_size - 1) - 1) // stride + 1
    out_width = (width + 2 * pad - (kernel_size - 1) - 1) // stride + 1
    return out_height, out_width


def _alloc_outputs(
    x_seq: torch.Tensor,
    out_channels: int,
    out_height: int = None,
    out_width: int = None,
    v_init: torch.Tensor = None,
):
    timesteps, batch, _, height, width = x_seq.shape
    out_height = height if out_height is None else int(out_height)
    out_width = width if out_width is None else int(out_width)

    spikes = torch.empty(
        (timesteps, batch, out_channels, out_height, out_width),
        device=x_seq.device,
        dtype=x_seq.dtype,
    )

    expected_shape = (batch, out_channels, out_height, out_width)

    # None 或 scalar v_init 都表示初始零膜电位
    if v_init is None or (isinstance(v_init, torch.Tensor) and v_init.dim() == 0):
        membrane = torch.zeros(expected_shape, device=x_seq.device, dtype=x_seq.dtype)
    else:
        if not isinstance(v_init, torch.Tensor):
            raise TypeError("v_init must be a torch.Tensor or None")
        if tuple(v_init.shape) != expected_shape:
            raise ValueError(f"v_init shape {tuple(v_init.shape)} does not match expected {expected_shape}")
        if v_init.device != x_seq.device:
            raise ValueError(f"v_init device {v_init.device} does not match x_seq device {x_seq.device}")
        if v_init.dtype != x_seq.dtype:
            raise ValueError(f"v_init dtype {v_init.dtype} does not match x_seq dtype {x_seq.dtype}")
        membrane = v_init.contiguous().clone()

    return spikes, membrane


def _check_output_buffers(
    x_seq: torch.Tensor,
    spikes: torch.Tensor,
    membrane: torch.Tensor,
    out_channels: int,
    out_height: int,
    out_width: int,
    v_init: torch.Tensor = None,
):
    timesteps, batch, _, _, _ = x_seq.shape
    expected_spikes = (timesteps, batch, out_channels, out_height, out_width)
    expected_membrane = (batch, out_channels, out_height, out_width)
    if tuple(spikes.shape) != expected_spikes:
        raise ValueError(f"spikes shape {tuple(spikes.shape)} does not match expected {expected_spikes}")
    if tuple(membrane.shape) != expected_membrane:
        raise ValueError(f"membrane shape {tuple(membrane.shape)} does not match expected {expected_membrane}")
    if spikes.device != x_seq.device or membrane.device != x_seq.device:
        raise ValueError("output buffers must be on the same device as x_seq")
    if spikes.dtype != x_seq.dtype or membrane.dtype != x_seq.dtype:
        raise ValueError("output buffers must have the same dtype as x_seq")
    if v_init is None or (isinstance(v_init, torch.Tensor) and v_init.dim() == 0):
        membrane.zero_()
    else:
        if tuple(v_init.shape) != expected_membrane:
            raise ValueError(f"v_init shape {tuple(v_init.shape)} does not match expected {expected_membrane}")
        if v_init.device != x_seq.device or v_init.dtype != x_seq.dtype:
            raise ValueError("v_init must have the same device and dtype as x_seq")
        membrane.copy_(v_init)


KERNEL_VARIANTS = {
    "k1_s1_p0": {"function": "_fused_conv_lif_temporal_general_kernel_k1_s1_p0_impl", "kernel": 1, "stride": 1, "pad": 0},
    "k1_s1_p0_direct": {
        "function": "_fused_pointwise_conv_lif_temporal_kernel_k1_s1_p0_impl",
        "kernel": 1,
        "stride": 1,
        "pad": 0,
        "pointwise_direct": True,
    },
    "k3_s1_p1": {"function": "_fused_conv_lif_temporal_general_kernel_k3_s1_p1_impl", "kernel": 3, "stride": 1, "pad": 1},
    "k3_s2_p1": {"function": "_fused_conv_lif_temporal_general_kernel_k3_s2_p1_impl", "kernel": 3, "stride": 2, "pad": 1},
    "k5_s1_p2": {"function": "_fused_conv_lif_temporal_general_kernel_k5_s1_p2_impl", "kernel": 5, "stride": 1, "pad": 2},
    "k7_s2_p3": {"function": "_fused_conv_lif_temporal_general_kernel_k7_s2_p3_impl", "kernel": 7, "stride": 2, "pad": 3},
    "k11_s4_p2": {"function": "_fused_conv_lif_temporal_general_kernel_k11_s4_p2_impl", "kernel": 11, "stride": 4, "pad": 2},
    "depthwise_k3_s1_p1": {
        "function": "_fused_depthwise_conv_lif_temporal_kernel_k3_s1_p1_impl",
        "kernel": 3,
        "stride": 1,
        "pad": 1,
        "depthwise": True,
    },
    "depthwise_k3_s2_p1": {
        "function": "_fused_depthwise_conv_lif_temporal_kernel_k3_s2_p1_impl",
        "kernel": 3,
        "stride": 2,
        "pad": 1,
        "depthwise": True,
    },
}


def _is_depthwise_kernel_key(kernel_key: str) -> bool:
    return bool(KERNEL_VARIANTS.get(kernel_key, {}).get("depthwise", False))


def _is_pointwise_kernel_key(kernel_key: str) -> bool:
    return kernel_key in ("k1_s1_p0", "k1_s1_p0_direct")


def _default_config_for_key(kernel_key: str) -> Dict[str, int]:
    if _is_depthwise_kernel_key(kernel_key):
        return {"BLOCK_H": 4, "BLOCK_W": 8, "BLOCK_C": 16, "PIXELS_PER_THREAD": 1, "num_warps": 1, "num_stages": 3}
    if kernel_key in ("k1_s1_p0", "k1_s1_p0_direct"):
        return {"BLOCK_M": 16, "BLOCK_OC": 128, "BLOCK_K": 32, "num_warps": 4, "num_stages": 2}
    return {"BLOCK_M": 16, "BLOCK_OC": 64, "BLOCK_K": 32, "num_warps": 4, "num_stages": 2}


def _pointwise_config_for_shape(in_channels: int, out_channels: int, height: int, width: int) -> Dict[str, int]:
    # MobileNetV1 pointwise layers are bandwidth/occupancy sensitive across
    # spatial scales. Keep a compact selector so regular conv autotune remains
    # untouched while k1_s1_p0 does not inherit the old two-config default.
    if height <= 14 and out_channels >= 4 * in_channels:
        return {"BLOCK_M": 16, "BLOCK_OC": 128, "BLOCK_K": 32, "num_warps": 4, "num_stages": 2}
    if 14 <= height <= 56 and in_channels >= 4 * out_channels:
        return {"BLOCK_M": 32, "BLOCK_OC": 64, "BLOCK_K": 32, "num_warps": 4, "num_stages": 2}
    if height >= 112 and in_channels <= 64:
        return {"BLOCK_M": 16, "BLOCK_OC": 128, "BLOCK_K": 16, "num_warps": 4, "num_stages": 2}
    if height >= 56 and in_channels <= 128:
        return {"BLOCK_M": 8, "BLOCK_OC": 128, "BLOCK_K": 64, "num_warps": 2, "num_stages": 2}
    if height >= 28 and in_channels <= 256:
        return {"BLOCK_M": 16, "BLOCK_OC": 128, "BLOCK_K": 64, "num_warps": 4, "num_stages": 2}
    if height >= 28:
        return {"BLOCK_M": 16, "BLOCK_OC": 128, "BLOCK_K": 32, "num_warps": 4, "num_stages": 2}
    if height <= 14 and in_channels <= 512:
        return {"BLOCK_M": 32, "BLOCK_OC": 64, "BLOCK_K": 32, "num_warps": 4, "num_stages": 2}
    return {"BLOCK_M": 16, "BLOCK_OC": 128, "BLOCK_K": 32, "num_warps": 4, "num_stages": 2}


def _depthwise_config_for_shape(kernel_key: str, channels: int, height: int, width: int) -> Dict[str, int]:
    if kernel_key == "depthwise_k3_s1_p1":
        if height <= 14 and channels >= 768:
            return {"BLOCK_H": 4, "BLOCK_W": 8, "BLOCK_C": 16, "PIXELS_PER_THREAD": 1, "num_warps": 4, "num_stages": 3}
        if height >= 16 and channels <= 64:
            return {"BLOCK_H": 16, "BLOCK_W": 16, "BLOCK_C": 16, "PIXELS_PER_THREAD": 1, "num_warps": 4, "num_stages": 4}
        if height >= 8 and channels <= 256:
            return {"BLOCK_H": 2, "BLOCK_W": 8, "BLOCK_C": 16, "PIXELS_PER_THREAD": 1, "num_warps": 2, "num_stages": 4}
        if height >= 4 and channels <= 512:
            return {"BLOCK_H": 4, "BLOCK_W": 8, "BLOCK_C": 16, "PIXELS_PER_THREAD": 1, "num_warps": 4, "num_stages": 3}
        return {"BLOCK_H": 2, "BLOCK_W": 8, "BLOCK_C": 16, "PIXELS_PER_THREAD": 1, "num_warps": 1, "num_stages": 3}
    if kernel_key == "depthwise_k3_s2_p1":
        if height <= 14 and channels >= 768:
            return {"BLOCK_H": 8, "BLOCK_W": 8, "BLOCK_C": 16, "PIXELS_PER_THREAD": 1, "num_warps": 2, "num_stages": 3}
        if height >= 16 and channels <= 128:
            return {"BLOCK_H": 8, "BLOCK_W": 8, "BLOCK_C": 16, "PIXELS_PER_THREAD": 1, "num_warps": 2, "num_stages": 3}
        if height >= 8 and channels <= 256:
            return {"BLOCK_H": 4, "BLOCK_W": 8, "BLOCK_C": 16, "PIXELS_PER_THREAD": 1, "num_warps": 1, "num_stages": 3}
        if height >= 4 and channels <= 512:
            return {"BLOCK_H": 2, "BLOCK_W": 8, "BLOCK_C": 16, "PIXELS_PER_THREAD": 1, "num_warps": 4, "num_stages": 5}
        return {"BLOCK_H": 8, "BLOCK_W": 8, "BLOCK_C": 16, "PIXELS_PER_THREAD": 1, "num_warps": 4, "num_stages": 3}
    return _default_config_for_key(kernel_key)


def _emit_general_kernel_source(
    max_groups: int = MAX_REUSE_GROUPS,
    function_name: str = "_fused_conv_lif_temporal_general_kernel_impl",
    kernel_size: int = 3,
    stride: int = 1,
    pad: int = 1,
    residual_add: bool = False,
    pointwise_direct: bool = False,
) -> str:
    """Generate a static-unrolled Triton kernel.

    Triton tensors cannot be mutated through Python containers inside jit code
    (`acc_groups[g] = ...` is unsupported). This generator keeps one generalized
    pattern in Python and emits a static specialization body for REUSE_GROUPS up
    to 16 and BTILE_T up to 16, covering T=4/8/16 experiments without hand-writing
    separate kernels.
    """
    lines: List[str] = []

    def emit_split_tree(var_expr: str, levels: int, g: int, path: Tuple[int, ...], indent: str):
        if levels == 0:
            idx = 0
            for bit in path:
                idx = idx * 2 + bit
            lines.append(f"{indent}acc_g{g}_{idx} = {var_expr}")
            return

        suffix = "_".join(str(bit) for bit in path) or "root"
        lhs = f"split_g{g}_{suffix}_0"
        rhs = f"split_g{g}_{suffix}_1"
        permute_args = ", ".join(str(idx) for idx in (list(range(1, levels)) + [levels, levels + 1, 0]))
        lines.append(f"{indent}{lhs}, {rhs} = {var_expr}.permute({permute_args}).split()")
        emit_split_tree(lhs, levels - 1, g, path + (0,), indent)
        emit_split_tree(rhs, levels - 1, g, path + (1,), indent)

    def emit_btile_split(g: int, btile_t: int, keyword: str):
        indent = "            "
        levels = btile_t.bit_length() - 1
        lines.append(f"{indent}{keyword} BTILE_T == {btile_t}:")
        if btile_t == 1:
            lines.append(f"{indent}    acc_g{g}_0 = acc_g{g}")
            return
        shape = ", ".join(["2"] * levels + ["BLOCK_M", "BLOCK_OC"])
        emit_split_tree(f"acc_g{g}.reshape([{shape}])", levels, g, (), indent + "    ")

    if pointwise_direct and (kernel_size, stride, pad) != (1, 1, 0):
        raise ValueError("pointwise_direct emitter only supports k1_s1_p0")

    k_square = kernel_size * kernel_size
    lines.append(f"def {function_name}(")
    if residual_add:
        lines.append("    x_ptr, residual_ptr, w_ptr, b_ptr, v_ptr, spike_ptr,")
    else:
        lines.append("    x_ptr, w_ptr, b_ptr, v_ptr, spike_ptr,")
    lines.append("    num_batches, in_channels: tl.constexpr, out_channels, height, width, out_height, out_width,")
    lines.append("    v_threshold, v_reset, tau_inv,")
    lines.append("    T_STEPS: tl.constexpr,")
    lines.append("    BLOCK_M: tl.constexpr, BLOCK_OC: tl.constexpr, BLOCK_K: tl.constexpr,")
    lines.append("    BTILE_T: tl.constexpr, REUSE_GROUPS: tl.constexpr,")
    lines.append("    USE_TF32: tl.constexpr,")
    lines.append("):")
    lines.append("    pid_m = tl.program_id(0)")
    lines.append("    pid_oc = tl.program_id(1)")
    lines.append("    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)")
    lines.append("    oc_offsets = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)")
    lines.append("    m_mask = m_offsets < (num_batches * out_height * out_width)")
    lines.append("    oc_mask = oc_offsets < out_channels")
    lines.append("    pix_n = m_offsets // (out_height * out_width)")
    lines.append("    pix_hw = m_offsets % (out_height * out_width)")
    lines.append("    pix_h = pix_hw // out_width")
    lines.append("    pix_w = pix_hw % out_width")
    if pointwise_direct:
        lines.append("    K_TOTAL: tl.constexpr = in_channels")
    else:
        lines.append(f"    K_TOTAL: tl.constexpr = in_channels * {k_square}")
    lines.append("    BM_T: tl.constexpr = BLOCK_M * BTILE_T")
    lines.append("    WINDOW_T: tl.constexpr = BTILE_T * REUSE_GROUPS")
    lines.append("    if USE_TF32:")
    lines.append("        bias = tl.load(b_ptr + oc_offsets, mask=oc_mask, other=0.0).to(tl.float32)")
    lines.append("    else:")
    lines.append("        bias = tl.load(b_ptr + oc_offsets, mask=oc_mask, other=0.0).to(tl.float16)")
    lines.append("    v_offsets = pix_n[:, None] * (out_channels * out_height * out_width) + oc_offsets[None, :] * (out_height * out_width) + pix_hw[:, None]")
    lines.append("    if USE_TF32:")
    lines.append("        v_state = tl.load(v_ptr + v_offsets, mask=m_mask[:, None] & oc_mask[None, :], other=0.0).to(tl.float32)")
    lines.append("    else:")
    lines.append("        v_state = tl.load(v_ptr + v_offsets, mask=m_mask[:, None] & oc_mask[None, :], other=0.0).to(tl.float16)")
    lines.append("    cat_offsets = tl.arange(0, BM_T)")
    lines.append("    local_t = cat_offsets // BLOCK_M")
    lines.append("    local_m = cat_offsets % BLOCK_M")
    lines.append("    cat_m_offsets = pid_m * BLOCK_M + local_m")
    lines.append("    cat_m_mask = cat_m_offsets < (num_batches * out_height * out_width)")
    lines.append("    cat_pix_n = cat_m_offsets // (out_height * out_width)")
    lines.append("    cat_pix_hw = cat_m_offsets % (out_height * out_width)")
    if not pointwise_direct:
        lines.append("    cat_pix_h = cat_pix_hw // out_width")
        lines.append("    cat_pix_w = cat_pix_hw % out_width")
    lines.append("    for temporal_base in range(0, T_STEPS, WINDOW_T):")
    for g in range(max_groups):
        lines.append(f"        if REUSE_GROUPS >= {g + 1}:")
        lines.append("            if USE_TF32:")
        lines.append(f"                acc_g{g} = tl.zeros((BM_T, BLOCK_OC), dtype=tl.float32)")
        lines.append("            else:")
        lines.append(f"                acc_g{g} = tl.zeros((BM_T, BLOCK_OC), dtype=tl.float16)")
    lines.append("        for k_start in range(0, K_TOTAL, BLOCK_K):")
    lines.append("            k_offsets = k_start + tl.arange(0, BLOCK_K)")
    lines.append("            k_mask = k_offsets < K_TOTAL")
    if pointwise_direct:
        lines.append("            ci = k_offsets")
        lines.append("            w_offsets = oc_offsets[None, :] * in_channels + k_offsets[:, None]")
    else:
        lines.append(f"            ci = k_offsets // {k_square}")
        lines.append(f"            kk = k_offsets % {k_square}")
        lines.append(f"            kh = kk // {kernel_size}")
        lines.append(f"            kw = kk % {kernel_size}")
        lines.append(f"            ih = cat_pix_h[:, None] * {stride} + kh[None, :] - {pad}")
        lines.append(f"            iw = cat_pix_w[:, None] * {stride} + kw[None, :] - {pad}")
        lines.append("            in_bounds = (ih >= 0) & (ih < height) & (iw >= 0) & (iw < width)")
        lines.append("            w_offsets = oc_offsets[None, :] * K_TOTAL + k_offsets[:, None]")
    lines.append("            w_tile = tl.load(w_ptr + w_offsets, mask=k_mask[:, None] & oc_mask[None, :], other=0.0)")
    for g in range(max_groups):
        lines.append(f"            if REUSE_GROUPS >= {g + 1}:")
        lines.append(f"                step_g{g} = temporal_base + {g} * BTILE_T + local_t")
        if pointwise_direct:
            lines.append(f"                x_offsets_g{g} = (step_g{g}[:, None] * num_batches + cat_pix_n[:, None]) * in_channels * height * width + ci[None, :] * height * width + cat_pix_hw[:, None]")
            lines.append(f"                x_g{g} = tl.load(x_ptr + x_offsets_g{g}, mask=cat_m_mask[:, None] & (step_g{g}[:, None] < T_STEPS) & k_mask[None, :], other=0.0)")
        else:
            lines.append(f"                x_offsets_g{g} = (step_g{g}[:, None] * num_batches + cat_pix_n[:, None]) * in_channels * height * width + ci[None, :] * height * width + ih * width + iw")
            lines.append(f"                x_g{g} = tl.load(x_ptr + x_offsets_g{g}, mask=cat_m_mask[:, None] & (step_g{g}[:, None] < T_STEPS) & k_mask[None, :] & in_bounds, other=0.0)")
        lines.append("                if USE_TF32:")
        lines.append(f"                    acc_g{g} = tl.dot(x_g{g}, w_tile, acc_g{g}, input_precision='tf32')")
        lines.append("                else:")
        lines.append(f"                    acc_g{g} = tl.dot(x_g{g}, w_tile, acc_g{g}, out_dtype=tl.float16)")
    for g in range(max_groups):
        lines.append(f"        if REUSE_GROUPS >= {g + 1}:")
        for idx, btile_t in enumerate(TEMPORAL_POW2_CANDIDATES):
            emit_btile_split(g, btile_t, "if" if idx == 0 else "elif")
        for bt in range(max(TEMPORAL_POW2_CANDIDATES)):
            lines.append(f"            if BTILE_T >= {bt + 1}:")
            lines.append(f"                step = temporal_base + {g} * BTILE_T + {bt}")
            lines.append("                if step < T_STEPS:")
            if residual_add:
                lines.append("                    residual_offsets = (step * num_batches + pix_n[:, None]) * out_channels * out_height * out_width + oc_offsets[None, :] * (out_height * out_width) + pix_hw[:, None]")
                lines.append("                    if USE_TF32:")
                lines.append("                        residual_t = tl.load(residual_ptr + residual_offsets, mask=m_mask[:, None] & oc_mask[None, :], other=0.0).to(tl.float32)")
                lines.append("                    else:")
                lines.append("                        residual_t = tl.load(residual_ptr + residual_offsets, mask=m_mask[:, None] & oc_mask[None, :], other=0.0).to(tl.float16)")
            lines.append("                    if USE_TF32:")
            if residual_add:
                lines.append(f"                        acc_t = acc_g{g}_{bt} + bias[None, :] + residual_t")
            else:
                lines.append(f"                        acc_t = acc_g{g}_{bt} + bias[None, :]")
            lines.append("                        v_new = v_state + (acc_t - (v_state - v_reset)) * tau_inv")
            lines.append("                        spike = (v_new >= v_threshold).to(tl.float32)")
            lines.append("                        v_state = tl.where(spike > 0.5, v_reset, v_new)")
            lines.append("                    else:")
            if residual_add:
                lines.append(f"                        acc_t = (acc_g{g}_{bt} + bias[None, :] + residual_t).to(tl.float16)")
            else:
                lines.append(f"                        acc_t = (acc_g{g}_{bt} + bias[None, :]).to(tl.float16)")
            lines.append("                        v_new = (v_state + (acc_t - (v_state - v_reset)) * tau_inv).to(tl.float16)")
            lines.append("                        spike = (v_new >= v_threshold).to(tl.float16)")
            lines.append("                        v_state = tl.where(spike > 0.5, v_new * 0.0, v_new)")
            lines.append("                    spike_offsets = (step * num_batches + pix_n[:, None]) * out_channels * out_height * out_width + oc_offsets[None, :] * (out_height * out_width) + pix_hw[:, None]")
            lines.append("                    tl.store(spike_ptr + spike_offsets, spike, mask=m_mask[:, None] & oc_mask[None, :])")
    lines.append("    tl.store(v_ptr + v_offsets, v_state, mask=m_mask[:, None] & oc_mask[None, :])")
    return "\n".join(lines)


def _emit_depthwise_kernel_source(
    function_name: str,
    stride: int,
    pad: int,
    residual_add: bool = False,
) -> str:
    lines: List[str] = []
    lines.append(f"def {function_name}(")
    if residual_add:
        lines.append("    x_ptr, residual_ptr, w_ptr, b_ptr, v_ptr, spike_ptr,")
    else:
        lines.append("    x_ptr, w_ptr, b_ptr, v_ptr, spike_ptr,")
    lines.append("    num_batches, in_channels: tl.constexpr, out_channels, height, width, out_height, out_width,")
    lines.append("    v_threshold, v_reset, tau_inv,")
    lines.append("    T_STEPS: tl.constexpr,")
    lines.append("    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_C: tl.constexpr,")
    lines.append("    PIXELS_PER_THREAD: tl.constexpr,")
    lines.append("    BTILE_T: tl.constexpr, REUSE_GROUPS: tl.constexpr,")
    lines.append("    USE_TF32: tl.constexpr,")
    lines.append("):")
    lines.append("    pid_w = tl.program_id(0)")
    lines.append("    pid_hn = tl.program_id(1)")
    lines.append("    pid_c = tl.program_id(2)")
    lines.append("    BLOCK_W_ELEMS: tl.constexpr = BLOCK_W * PIXELS_PER_THREAD")
    lines.append("    w_offsets = pid_w * BLOCK_W_ELEMS + tl.arange(0, BLOCK_W_ELEMS)")
    lines.append("    hn_offsets = pid_hn * BLOCK_H + tl.arange(0, BLOCK_H)")
    lines.append("    c_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)")
    lines.append("    pix_n = hn_offsets // out_height")
    lines.append("    pix_h = hn_offsets % out_height")
    lines.append("    pix_w = w_offsets")
    lines.append("    spatial_mask = (pix_n[:, None] < num_batches) & (pix_w[None, :] < out_width)")
    lines.append("    c_mask = c_offsets < out_channels")
    lines.append("    out_hw = pix_h[:, None] * out_width + pix_w[None, :]")
    lines.append("    WINDOW_T: tl.constexpr = BTILE_T * REUSE_GROUPS")
    lines.append("    if USE_TF32:")
    lines.append("        bias = tl.load(b_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)")
    lines.append("    else:")
    lines.append("        bias = tl.load(b_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float16)")
    lines.append("    v_offsets = pix_n[:, None, None] * (out_channels * out_height * out_width) + c_offsets[None, None, :] * (out_height * out_width) + out_hw[:, :, None]")
    lines.append("    if USE_TF32:")
    lines.append("        v_state = tl.load(v_ptr + v_offsets, mask=spatial_mask[:, :, None] & c_mask[None, None, :], other=0.0).to(tl.float32)")
    lines.append("    else:")
    lines.append("        v_state = tl.load(v_ptr + v_offsets, mask=spatial_mask[:, :, None] & c_mask[None, None, :], other=0.0).to(tl.float16)")
    lines.append("    for temporal_base in range(0, T_STEPS, WINDOW_T):")
    for g in range(MAX_REUSE_GROUPS):
        lines.append(f"        if REUSE_GROUPS >= {g + 1}:")
        for bt in range(max(TEMPORAL_POW2_CANDIDATES)):
            lines.append(f"            if BTILE_T >= {bt + 1}:")
            lines.append(f"                step = temporal_base + {g} * BTILE_T + {bt}")
            lines.append("                if step < T_STEPS:")
            lines.append("                    if USE_TF32:")
            lines.append("                        acc = tl.zeros((BLOCK_H, BLOCK_W_ELEMS, BLOCK_C), dtype=tl.float32)")
            lines.append("                    else:")
            lines.append("                        acc = tl.zeros((BLOCK_H, BLOCK_W_ELEMS, BLOCK_C), dtype=tl.float16)")
            for ky in range(3):
                for kx in range(3):
                    lines.append(f"                    ih_{ky}_{kx} = pix_h[:, None, None] * {stride} + {ky} - {pad}")
                    lines.append(f"                    iw_{ky}_{kx} = pix_w[None, :, None] * {stride} + {kx} - {pad}")
                    lines.append(
                        f"                    in_bounds_{ky}_{kx} = "
                        f"(ih_{ky}_{kx} >= 0) & (ih_{ky}_{kx} < height) & "
                        f"(iw_{ky}_{kx} >= 0) & (iw_{ky}_{kx} < width)"
                    )
                    lines.append(
                        f"                    x_offsets_{ky}_{kx} = "
                        f"(step * num_batches + pix_n[:, None, None]) * in_channels * height * width + "
                        f"c_offsets[None, None, :] * height * width + ih_{ky}_{kx} * width + iw_{ky}_{kx}"
                    )
                    lines.append(f"                    w_offsets_{ky}_{kx} = c_offsets * 9 + {ky * 3 + kx}")
                    lines.append(
                        f"                    x_{ky}_{kx} = tl.load("
                        f"x_ptr + x_offsets_{ky}_{kx}, "
                        f"mask=spatial_mask[:, :, None] & c_mask[None, None, :] & in_bounds_{ky}_{kx}, other=0.0, cache_modifier='.ca')"
                    )
                    lines.append(
                        f"                    w_{ky}_{kx} = tl.load("
                        f"w_ptr + w_offsets_{ky}_{kx}, mask=c_mask, other=0.0)"
                    )
                    lines.append(f"                    acc += x_{ky}_{kx} * w_{ky}_{kx}[None, None, :]")
            if residual_add:
                lines.append("                    residual_offsets = (step * num_batches + pix_n[:, None, None]) * out_channels * out_height * out_width + c_offsets[None, None, :] * (out_height * out_width) + out_hw[:, :, None]")
                lines.append("                    if USE_TF32:")
                lines.append("                        residual_t = tl.load(residual_ptr + residual_offsets, mask=spatial_mask[:, :, None] & c_mask[None, None, :], other=0.0).to(tl.float32)")
                lines.append("                    else:")
                lines.append("                        residual_t = tl.load(residual_ptr + residual_offsets, mask=spatial_mask[:, :, None] & c_mask[None, None, :], other=0.0).to(tl.float16)")
            lines.append("                    if USE_TF32:")
            if residual_add:
                lines.append("                        acc_t = acc + bias[None, None, :] + residual_t")
            else:
                lines.append("                        acc_t = acc + bias[None, None, :]")
            lines.append("                        v_new = v_state + (acc_t - (v_state - v_reset)) * tau_inv")
            lines.append("                        spike = (v_new >= v_threshold).to(tl.float32)")
            lines.append("                        v_state = tl.where(spike > 0.5, v_reset, v_new)")
            lines.append("                    else:")
            if residual_add:
                lines.append("                        acc_t = (acc + bias[None, None, :] + residual_t).to(tl.float16)")
            else:
                lines.append("                        acc_t = (acc + bias[None, None, :]).to(tl.float16)")
            lines.append("                        v_new = (v_state + (acc_t - (v_state - v_reset)) * tau_inv).to(tl.float16)")
            lines.append("                        spike = (v_new >= v_threshold).to(tl.float16)")
            lines.append("                        v_state = tl.where(spike > 0.5, v_new * 0.0, v_new)")
            lines.append("                    spike_offsets = (step * num_batches + pix_n[:, None, None]) * out_channels * out_height * out_width + c_offsets[None, None, :] * (out_height * out_width) + out_hw[:, :, None]")
            lines.append("                    tl.store(spike_ptr + spike_offsets, spike, mask=spatial_mask[:, :, None] & c_mask[None, None, :])")
    lines.append("    tl.store(v_ptr + v_offsets, v_state, mask=spatial_mask[:, :, None] & c_mask[None, None, :])")
    return "\n".join(lines)


_kernel_namespace = {"tl": tl}
_kernel_sources = []
for _kernel_key, _variant in KERNEL_VARIANTS.items():
    if _variant.get("depthwise"):
        _kernel_sources.append(
            _emit_depthwise_kernel_source(
                function_name=_variant["function"],
                stride=_variant["stride"],
                pad=_variant["pad"],
            )
        )
        continue
    _kernel_sources.append(
        _emit_general_kernel_source(
            function_name=_variant["function"],
            kernel_size=_variant["kernel"],
            stride=_variant["stride"],
            pad=_variant["pad"],
            pointwise_direct=bool(_variant.get("pointwise_direct", False)),
        )
    )
for _kernel_key, _variant in KERNEL_VARIANTS.items():
    if _variant.get("depthwise"):
        _kernel_sources.append(
            _emit_depthwise_kernel_source(
                function_name=f"{_variant['function']}_resadd",
                stride=_variant["stride"],
                pad=_variant["pad"],
                residual_add=True,
            )
        )
        continue
    _kernel_sources.append(
        _emit_general_kernel_source(
            function_name=f"{_variant['function']}_resadd",
            kernel_size=_variant["kernel"],
            stride=_variant["stride"],
            pad=_variant["pad"],
            residual_add=True,
            pointwise_direct=bool(_variant.get("pointwise_direct", False)),
        )
    )
_kernel_sources.append("_fused_conv_lif_temporal_general_kernel_impl = _fused_conv_lif_temporal_general_kernel_k3_s1_p1_impl")
_kernel_source = "\n\n".join(_kernel_sources)
_kernel_filename = os.path.join(os.path.dirname(__file__), "generated_temporal_general_kernel.py")
with open(_kernel_filename, "w", encoding="utf-8") as kernel_file:
    kernel_file.write(_kernel_source)
    kernel_file.write("\n")
linecache.cache[_kernel_filename] = (
    len(_kernel_source),
    None,
    [line + "\n" for line in _kernel_source.splitlines()],
    _kernel_filename,
)
exec(compile(_kernel_source, _kernel_filename, "exec"), _kernel_namespace)
_kernel_fns = {
    kernel_key: _kernel_namespace[variant["function"]]
    for kernel_key, variant in KERNEL_VARIANTS.items()
}
_residual_kernel_fns = {
    kernel_key: _kernel_namespace[f"{variant['function']}_resadd"]
    for kernel_key, variant in KERNEL_VARIANTS.items()
}
_specialized_kernels = {
    kernel_key: triton.jit(kernel_fn)
    for kernel_key, kernel_fn in _kernel_fns.items()
}
_residual_specialized_kernels = {
    kernel_key: triton.jit(kernel_fn)
    for kernel_key, kernel_fn in _residual_kernel_fns.items()
}
_autotuned_kernels = {
    kernel_key: triton.autotune(
        configs=(
            _make_dwconv_autotune_configs()
            if KERNEL_VARIANTS[kernel_key].get("depthwise")
            else _make_pointwise_autotune_configs()
            if _is_pointwise_kernel_key(kernel_key)
            else _make_autotune_configs()
        ),
        key=[
            "num_batches",
            "in_channels",
            "out_channels",
            "height",
            "width",
            "out_height",
            "out_width",
            "T_STEPS",
            "USE_TF32",
        ],
        prune_configs_by={
            "early_config_prune": (
                _prune_dwconv_configs
                if KERNEL_VARIANTS[kernel_key].get("depthwise")
                else _prune_pointwise_configs
                if _is_pointwise_kernel_key(kernel_key)
                else _prune_temporal_configs
            )
        },
        reset_to_zero=["v_ptr"],
        cache_results=True,
    )(triton.jit(kernel_fn))
    for kernel_key, kernel_fn in _kernel_fns.items()
}
_residual_autotuned_kernels = {
    kernel_key: triton.autotune(
        configs=(
            _make_dwconv_autotune_configs()
            if KERNEL_VARIANTS[kernel_key].get("depthwise")
            else _make_pointwise_autotune_configs()
            if _is_pointwise_kernel_key(kernel_key)
            else _make_autotune_configs()
        ),
        key=[
            "num_batches",
            "in_channels",
            "out_channels",
            "height",
            "width",
            "out_height",
            "out_width",
            "T_STEPS",
            "USE_TF32",
        ],
        prune_configs_by={
            "early_config_prune": (
                _prune_dwconv_configs
                if KERNEL_VARIANTS[kernel_key].get("depthwise")
                else _prune_pointwise_configs
                if _is_pointwise_kernel_key(kernel_key)
                else _prune_temporal_configs
            )
        },
        reset_to_zero=["v_ptr"],
        cache_results=True,
    )(triton.jit(kernel_fn))
    for kernel_key, kernel_fn in _residual_kernel_fns.items()
}
_kernel_fn = _kernel_fns["k3_s1_p1"]
fused_conv_lif_temporal_general_specialized_kernel = _specialized_kernels["k3_s1_p1"]
fused_conv_lif_temporal_general_kernel = _autotuned_kernels["k3_s1_p1"]


def valid_temporal_schedules(timesteps: int):
    schedules = []
    for btile_t in TEMPORAL_POW2_CANDIDATES:
        for reuse_groups in TEMPORAL_POW2_CANDIDATES:
            if btile_t * reuse_groups <= timesteps:
                schedules.append((btile_t, reuse_groups))
    return schedules


def run_fused_temporal_general(
    x_seq: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    temporal_batch_size: int,
    reuse_groups: int,
    spatial_config: Dict[str, int] = None,
    kernel_key: str = "k3_s1_p1",
    v_init: torch.Tensor = None,
    spikes_out: torch.Tensor = None,
    membrane_out: torch.Tensor = None,
):
    if temporal_batch_size not in TEMPORAL_POW2_CANDIDATES:
        raise ValueError(f"temporal_batch_size must be one of {TEMPORAL_POW2_CANDIDATES}, got {temporal_batch_size}")
    if reuse_groups not in TEMPORAL_POW2_CANDIDATES:
        raise ValueError(f"reuse_groups must be one of {TEMPORAL_POW2_CANDIDATES}, got {reuse_groups}")
    if reuse_groups > MAX_REUSE_GROUPS:
        raise ValueError(f"reuse_groups={reuse_groups} exceeds generated max_groups={MAX_REUSE_GROUPS}")

    timesteps, batch, _, height, width = x_seq.shape
    if temporal_batch_size * reuse_groups > timesteps:
        raise ValueError(
            f"BTILE_T * REUSE_GROUPS must be <= T_STEPS, got "
            f"{temporal_batch_size} * {reuse_groups} > {timesteps}"
        )

    if kernel_key not in KERNEL_VARIANTS:
        raise ValueError(f"unsupported kernel_key={kernel_key}, expected one of {tuple(KERNEL_VARIANTS)}")

    in_channels = x_seq.shape[2] if _is_depthwise_kernel_key(kernel_key) else weight.shape[1]
    out_channels = weight.shape[0]
    variant = KERNEL_VARIANTS[kernel_key]
    out_height, out_width = _conv_out_hw(height, width, variant["kernel"], variant["stride"], variant["pad"])
    if spatial_config is None and _is_depthwise_kernel_key(kernel_key):
        spatial_config = _depthwise_config_for_shape(kernel_key, out_channels, height, width)
    elif spatial_config is None and _is_pointwise_kernel_key(kernel_key):
        spatial_config = _pointwise_config_for_shape(in_channels, out_channels, height, width)
    elif spatial_config is None:
        spatial_config = _default_config_for_key(kernel_key)
    x_flat = x_seq.reshape(timesteps * batch, in_channels, height, width).contiguous()
    if spikes_out is None or membrane_out is None:
        spikes, membrane = _alloc_outputs(x_seq, out_channels, out_height, out_width, v_init=v_init)
    else:
        spikes, membrane = spikes_out, membrane_out
        _check_output_buffers(x_seq, spikes, membrane, out_channels, out_height, out_width, v_init=v_init)
    kernel = _specialized_kernels[kernel_key]

    def grid(meta):
        if _is_depthwise_kernel_key(kernel_key):
            return (
                triton.cdiv(out_width, meta["BLOCK_W"] * meta["PIXELS_PER_THREAD"]),
                triton.cdiv(batch * out_height, meta["BLOCK_H"]),
                triton.cdiv(out_channels, meta["BLOCK_C"]),
            )
        return (
            triton.cdiv(batch * out_height * out_width, meta["BLOCK_M"]),
            triton.cdiv(out_channels, meta["BLOCK_OC"]),
        )

    if _is_depthwise_kernel_key(kernel_key):
        kernel[grid](
            x_flat,
            weight,
            bias,
            membrane,
            spikes,
            batch,
            in_channels,
            out_channels,
            height,
            width,
            out_height,
            out_width,
            V_THRESHOLD,
            V_RESET,
            TAU_INV,
            timesteps,
            BTILE_T=temporal_batch_size,
            REUSE_GROUPS=reuse_groups,
            BLOCK_H=spatial_config["BLOCK_H"],
            BLOCK_W=spatial_config["BLOCK_W"],
            BLOCK_C=spatial_config["BLOCK_C"],
            PIXELS_PER_THREAD=spatial_config.get("PIXELS_PER_THREAD", 1),
            USE_TF32=(x_seq.dtype == torch.float32),
            num_warps=spatial_config["num_warps"],
            num_stages=spatial_config["num_stages"],
        )
    else:
        kernel[grid](
            x_flat,
            weight,
            bias,
            membrane,
            spikes,
            batch,
            in_channels,
            out_channels,
            height,
            width,
            out_height,
            out_width,
            V_THRESHOLD,
            V_RESET,
            TAU_INV,
            timesteps,
            BTILE_T=temporal_batch_size,
            REUSE_GROUPS=reuse_groups,
            BLOCK_M=spatial_config["BLOCK_M"],
            BLOCK_OC=spatial_config["BLOCK_OC"],
            BLOCK_K=spatial_config["BLOCK_K"],
            USE_TF32=(x_seq.dtype == torch.float32),
            num_warps=spatial_config["num_warps"],
            num_stages=spatial_config["num_stages"],
        )
    return spikes, membrane


def _check_residual_seq(residual_seq: torch.Tensor, spikes: torch.Tensor, x_seq: torch.Tensor):
    if residual_seq.shape != spikes.shape:
        raise ValueError(f"residual_seq shape {tuple(residual_seq.shape)} does not match output shape {tuple(spikes.shape)}")
    if residual_seq.device != x_seq.device:
        raise ValueError(f"residual_seq device {residual_seq.device} does not match x_seq device {x_seq.device}")
    if residual_seq.dtype != x_seq.dtype:
        raise ValueError(f"residual_seq dtype {residual_seq.dtype} does not match x_seq dtype {x_seq.dtype}")


def run_fused_temporal_general_residual(
    x_seq: torch.Tensor,
    residual_seq: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    temporal_batch_size: int,
    reuse_groups: int,
    spatial_config: Dict[str, int] = None,
    kernel_key: str = "k3_s1_p1",
    v_init: torch.Tensor = None,
):
    if temporal_batch_size not in TEMPORAL_POW2_CANDIDATES:
        raise ValueError(f"temporal_batch_size must be one of {TEMPORAL_POW2_CANDIDATES}, got {temporal_batch_size}")
    if reuse_groups not in TEMPORAL_POW2_CANDIDATES:
        raise ValueError(f"reuse_groups must be one of {TEMPORAL_POW2_CANDIDATES}, got {reuse_groups}")
    if reuse_groups > MAX_REUSE_GROUPS:
        raise ValueError(f"reuse_groups={reuse_groups} exceeds generated max_groups={MAX_REUSE_GROUPS}")

    timesteps, batch, _, height, width = x_seq.shape
    if temporal_batch_size * reuse_groups > timesteps:
        raise ValueError(
            f"BTILE_T * REUSE_GROUPS must be <= T_STEPS, got "
            f"{temporal_batch_size} * {reuse_groups} > {timesteps}"
        )
    if kernel_key not in KERNEL_VARIANTS:
        raise ValueError(f"unsupported kernel_key={kernel_key}, expected one of {tuple(KERNEL_VARIANTS)}")

    in_channels = x_seq.shape[2] if _is_depthwise_kernel_key(kernel_key) else weight.shape[1]
    out_channels = weight.shape[0]
    variant = KERNEL_VARIANTS[kernel_key]
    out_height, out_width = _conv_out_hw(height, width, variant["kernel"], variant["stride"], variant["pad"])
    if spatial_config is None and _is_depthwise_kernel_key(kernel_key):
        spatial_config = _depthwise_config_for_shape(kernel_key, out_channels, height, width)
    elif spatial_config is None and _is_pointwise_kernel_key(kernel_key):
        spatial_config = _pointwise_config_for_shape(in_channels, out_channels, height, width)
    elif spatial_config is None:
        spatial_config = _default_config_for_key(kernel_key)
    x_flat = x_seq.reshape(timesteps * batch, in_channels, height, width).contiguous()
    spikes, membrane = _alloc_outputs(x_seq, out_channels, out_height, out_width, v_init=v_init)
    _check_residual_seq(residual_seq, spikes, x_seq)
    residual_flat = residual_seq.reshape(timesteps * batch, out_channels, out_height, out_width).contiguous()
    kernel = _residual_specialized_kernels[kernel_key]

    def grid(meta):
        if _is_depthwise_kernel_key(kernel_key):
            return (
                triton.cdiv(out_width, meta["BLOCK_W"] * meta["PIXELS_PER_THREAD"]),
                triton.cdiv(batch * out_height, meta["BLOCK_H"]),
                triton.cdiv(out_channels, meta["BLOCK_C"]),
            )
        return (
            triton.cdiv(batch * out_height * out_width, meta["BLOCK_M"]),
            triton.cdiv(out_channels, meta["BLOCK_OC"]),
        )

    if _is_depthwise_kernel_key(kernel_key):
        kernel[grid](
            x_flat,
            residual_flat,
            weight,
            bias,
            membrane,
            spikes,
            batch,
            in_channels,
            out_channels,
            height,
            width,
            out_height,
            out_width,
            V_THRESHOLD,
            V_RESET,
            TAU_INV,
            timesteps,
            BTILE_T=temporal_batch_size,
            REUSE_GROUPS=reuse_groups,
            BLOCK_H=spatial_config["BLOCK_H"],
            BLOCK_W=spatial_config["BLOCK_W"],
            BLOCK_C=spatial_config["BLOCK_C"],
            PIXELS_PER_THREAD=spatial_config.get("PIXELS_PER_THREAD", 1),
            USE_TF32=(x_seq.dtype == torch.float32),
            num_warps=spatial_config["num_warps"],
            num_stages=spatial_config["num_stages"],
        )
    else:
        kernel[grid](
            x_flat,
            residual_flat,
            weight,
            bias,
            membrane,
            spikes,
            batch,
            in_channels,
            out_channels,
            height,
            width,
            out_height,
            out_width,
            V_THRESHOLD,
            V_RESET,
            TAU_INV,
            timesteps,
            BTILE_T=temporal_batch_size,
            REUSE_GROUPS=reuse_groups,
            BLOCK_M=spatial_config["BLOCK_M"],
            BLOCK_OC=spatial_config["BLOCK_OC"],
            BLOCK_K=spatial_config["BLOCK_K"],
            USE_TF32=(x_seq.dtype == torch.float32),
            num_warps=spatial_config["num_warps"],
            num_stages=spatial_config["num_stages"],
        )
    return spikes, membrane


def run_fused_temporal_general_autotuned(
    x_seq: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    kernel_key: str = "k3_s1_p1",
    v_init: torch.Tensor = None,
    spikes_out: torch.Tensor = None,
    membrane_out: torch.Tensor = None,
):
    timesteps, batch, _, height, width = x_seq.shape
    if kernel_key not in KERNEL_VARIANTS:
        raise ValueError(f"unsupported kernel_key={kernel_key}, expected one of {tuple(KERNEL_VARIANTS)}")
    in_channels = x_seq.shape[2] if _is_depthwise_kernel_key(kernel_key) else weight.shape[1]
    out_channels = weight.shape[0]
    variant = KERNEL_VARIANTS[kernel_key]
    out_height, out_width = _conv_out_hw(height, width, variant["kernel"], variant["stride"], variant["pad"])
    x_flat = x_seq.reshape(timesteps * batch, in_channels, height, width).contiguous()
    if spikes_out is None or membrane_out is None:
        spikes, membrane = _alloc_outputs(x_seq, out_channels, out_height, out_width, v_init=v_init)
    else:
        spikes, membrane = spikes_out, membrane_out
        _check_output_buffers(x_seq, spikes, membrane, out_channels, out_height, out_width, v_init=v_init)
    kernel = _autotuned_kernels[kernel_key]

    def grid(meta):
        if _is_depthwise_kernel_key(kernel_key):
            return (
                triton.cdiv(out_width, meta["BLOCK_W"] * meta["PIXELS_PER_THREAD"]),
                triton.cdiv(batch * out_height, meta["BLOCK_H"]),
                triton.cdiv(out_channels, meta["BLOCK_C"]),
            )
        return (
            triton.cdiv(batch * out_height * out_width, meta["BLOCK_M"]),
            triton.cdiv(out_channels, meta["BLOCK_OC"]),
        )

    kernel[grid](
        x_flat,
        weight,
        bias,
        membrane,
        spikes,
        batch,
        in_channels,
        out_channels,
        height,
        width,
        out_height,
        out_width,
        V_THRESHOLD,
        V_RESET,
        TAU_INV,
        timesteps,
        USE_TF32=(x_seq.dtype == torch.float32),
    )
    return spikes, membrane


def run_fused_temporal_general_residual_autotuned(
    x_seq: torch.Tensor,
    residual_seq: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    kernel_key: str = "k3_s1_p1",
    v_init: torch.Tensor = None,
):
    timesteps, batch, _, height, width = x_seq.shape
    if kernel_key not in KERNEL_VARIANTS:
        raise ValueError(f"unsupported kernel_key={kernel_key}, expected one of {tuple(KERNEL_VARIANTS)}")
    in_channels = x_seq.shape[2] if _is_depthwise_kernel_key(kernel_key) else weight.shape[1]
    out_channels = weight.shape[0]
    variant = KERNEL_VARIANTS[kernel_key]
    out_height, out_width = _conv_out_hw(height, width, variant["kernel"], variant["stride"], variant["pad"])
    x_flat = x_seq.reshape(timesteps * batch, in_channels, height, width).contiguous()
    spikes, membrane = _alloc_outputs(x_seq, out_channels, out_height, out_width, v_init=v_init)
    _check_residual_seq(residual_seq, spikes, x_seq)
    residual_flat = residual_seq.reshape(timesteps * batch, out_channels, out_height, out_width).contiguous()
    kernel = _residual_autotuned_kernels[kernel_key]

    def grid(meta):
        if _is_depthwise_kernel_key(kernel_key):
            return (
                triton.cdiv(out_width, meta["BLOCK_W"] * meta["PIXELS_PER_THREAD"]),
                triton.cdiv(batch * out_height, meta["BLOCK_H"]),
                triton.cdiv(out_channels, meta["BLOCK_C"]),
            )
        return (
            triton.cdiv(batch * out_height * out_width, meta["BLOCK_M"]),
            triton.cdiv(out_channels, meta["BLOCK_OC"]),
        )

    kernel[grid](
        x_flat,
        residual_flat,
        weight,
        bias,
        membrane,
        spikes,
        batch,
        in_channels,
        out_channels,
        height,
        width,
        out_height,
        out_width,
        V_THRESHOLD,
        V_RESET,
        TAU_INV,
        timesteps,
        USE_TF32=(x_seq.dtype == torch.float32),
    )
    return spikes, membrane


def run_baseline(model, state_model: nn.Module, x_seq: torch.Tensor):
    with torch.no_grad():
        functional.reset_net(state_model)
        return model(x_seq)


def get_autotune_best_config(kernel_key: str = "k3_s1_p1", residual_add: bool = False):
    kernels = _residual_autotuned_kernels if residual_add else _autotuned_kernels
    if kernel_key not in kernels:
        return None
    kernel = kernels[kernel_key]
    best_config = getattr(kernel, "best_config", None)
    if best_config is None:
        return None
    all_kwargs = best_config.all_kwargs()
    btile_t = all_kwargs.get("BTILE_T")
    reuse_groups = all_kwargs.get("REUSE_GROUPS")
    kernel_temporal_window = None
    if btile_t is not None and reuse_groups is not None:
        kernel_temporal_window = int(btile_t) * int(reuse_groups)
    return {
        "kernel_key": kernel_key,
        "residual_add": bool(residual_add),
        "BLOCK_M": all_kwargs.get("BLOCK_M"),
        "BLOCK_OC": all_kwargs.get("BLOCK_OC"),
        "BLOCK_K": all_kwargs.get("BLOCK_K"),
        "BLOCK_H": all_kwargs.get("BLOCK_H"),
        "BLOCK_W": all_kwargs.get("BLOCK_W"),
        "BLOCK_C": all_kwargs.get("BLOCK_C"),
        "PIXELS_PER_THREAD": all_kwargs.get("PIXELS_PER_THREAD"),
        "BTILE_T": btile_t,
        "REUSE_GROUPS": reuse_groups,
        "kernel_temporal_window": kernel_temporal_window,
        "acc_elems": (
            int(btile_t) * int(reuse_groups) * int(all_kwargs.get("BLOCK_M")) * int(all_kwargs.get("BLOCK_OC"))
            if btile_t is not None
            and reuse_groups is not None
            and all_kwargs.get("BLOCK_M") is not None
            and all_kwargs.get("BLOCK_OC") is not None
            else None
        ),
        "num_warps": all_kwargs.get("num_warps"),
        "num_stages": all_kwargs.get("num_stages"),
    }


def time_cuda(fn, warmup: int = 10, rep: int = 50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / rep


def manual_search_general_kernel(
    x_seq: torch.Tensor,
    conv: nn.Conv2d,
    ref_spikes: torch.Tensor,
    ref_v: torch.Tensor,
    repetitions: int,
):
    candidates = []
    for btile_t, reuse_groups in valid_temporal_schedules(x_seq.shape[0]):
        for spatial_config in SPATIAL_CONFIGS:
            with torch.no_grad():
                spikes, membrane = run_fused_temporal_general(
                    x_seq,
                    conv.weight,
                    conv.bias,
                    temporal_batch_size=btile_t,
                    reuse_groups=reuse_groups,
                    spatial_config=spatial_config,
                )
            summary = summarize_correctness(
                f"general_B{btile_t}_R{reuse_groups}", spikes, membrane, ref_spikes, ref_v
            )
            validate_summary(summary)
            ms = time_cuda(
                lambda bt=btile_t, rg=reuse_groups, cfg=spatial_config: run_fused_temporal_general(
                    x_seq,
                    conv.weight,
                    conv.bias,
                    temporal_batch_size=bt,
                    reuse_groups=rg,
                    spatial_config=cfg,
                ),
                rep=repetitions,
            )
            candidates.append(
                {
                    "btile_t": btile_t,
                    "reuse_groups": reuse_groups,
                    "spatial_config": spatial_config,
                    "ms": ms,
                    "correctness": summary,
                }
            )
    return min(candidates, key=lambda row: row["ms"]), candidates


def benchmark_one(label: str, shape: ProblemShape, repetitions: int, manual_search: bool = False):
    torch.manual_seed(0)
    conv = make_conv(shape)
    x_seq = build_input_sequence(shape)
    ref_spikes, ref_v = make_reference(x_seq, conv)
    baseline_s_eager = ConvLIFSingleStepBaseline(copy.deepcopy(conv)).to(DEVICE).eval()
    baseline_s_compile_model = ConvLIFSingleStepBaseline(copy.deepcopy(conv)).to(DEVICE).eval()
    baseline_s_compile = torch.compile(baseline_s_compile_model, fullgraph=True, mode="max-autotune")
    baseline_m_eager = ConvLIFMultiStepBaseline(conv).to(DEVICE).eval()

    with torch.no_grad():
        baseline_s_eager_spikes, baseline_s_eager_v = run_baseline(
            baseline_s_eager, baseline_s_eager, x_seq
        )
        baseline_s_compile_spikes, baseline_s_compile_v = run_baseline(
            baseline_s_compile, baseline_s_compile_model, x_seq
        )
        baseline_m_eager_spikes, baseline_m_eager_v = run_baseline(
            baseline_m_eager, baseline_m_eager, x_seq
        )
        fused_auto_spikes, fused_auto_v = run_fused_temporal_general_autotuned(
            x_seq, conv.weight, conv.bias
        )

    baseline_s_eager_summary = summarize_correctness(
        "baseline_s_eager", baseline_s_eager_spikes, baseline_s_eager_v, ref_spikes, ref_v
    )
    baseline_s_compile_summary = summarize_correctness(
        "baseline_s_compile", baseline_s_compile_spikes, baseline_s_compile_v, ref_spikes, ref_v
    )
    baseline_m_eager_summary = summarize_correctness(
        "baseline_m_eager", baseline_m_eager_spikes, baseline_m_eager_v, ref_spikes, ref_v
    )
    fused_auto_summary = summarize_correctness(
        "fused_auto", fused_auto_spikes, fused_auto_v, ref_spikes, ref_v
    )
    validate_summary(baseline_s_eager_summary)
    validate_summary(baseline_s_compile_summary)
    validate_summary(baseline_m_eager_summary)
    validate_summary(fused_auto_summary)

    best_config = get_autotune_best_config()
    if best_config is None:
        print("WARNING: Triton autotune best_config is unavailable for fused_auto")

    baseline_s_eager_ms = time_cuda(
        lambda: run_baseline(baseline_s_eager, baseline_s_eager, x_seq), rep=repetitions
    )
    baseline_s_compile_ms = time_cuda(
        lambda: run_baseline(baseline_s_compile, baseline_s_compile_model, x_seq), rep=repetitions
    )
    baseline_m_eager_ms = time_cuda(
        lambda: run_baseline(baseline_m_eager, baseline_m_eager, x_seq), rep=repetitions
    )
    fused_auto_ms = time_cuda(
        lambda: run_fused_temporal_general_autotuned(x_seq, conv.weight, conv.bias), rep=repetitions
    )

    result = {
        "label": label,
        "shape": shape,
        "baseline_s_eager_ms": baseline_s_eager_ms,
        "baseline_s_compile_ms": baseline_s_compile_ms,
        "baseline_m_eager_ms": baseline_m_eager_ms,
        "fused_auto_ms": fused_auto_ms,
        "baseline_s_eager_correctness": baseline_s_eager_summary,
        "baseline_s_compile_correctness": baseline_s_compile_summary,
        "baseline_m_eager_correctness": baseline_m_eager_summary,
        "fused_auto_correctness": fused_auto_summary,
        "autotune_best_config": best_config,
    }
    if manual_search:
        manual_best, manual_candidates = manual_search_general_kernel(
            x_seq, conv, ref_spikes, ref_v, repetitions
        )
        result["manual_best"] = manual_best
        result["manual_candidates"] = manual_candidates
    return result


def print_result(result):
    shape = result["shape"]
    correctness = result["fused_auto_correctness"]
    cfg = result["autotune_best_config"]
    if cfg is None:
        cfg_text = "best_config=unavailable"
    else:
        cfg_text = (
            f"BM={cfg['BLOCK_M']} BOC={cfg['BLOCK_OC']} BK={cfg['BLOCK_K']} "
            f"BTILE_T={cfg['BTILE_T']} REUSE_GROUPS={cfg['REUSE_GROUPS']} "
            f"warps={cfg['num_warps']} stages={cfg['num_stages']}"
        )
    print(
        f"{result['label']:<12} T={shape.timesteps:<2} "
        f"s_eager={result['baseline_s_eager_ms']:.3f} ms "
        f"s_compile={result['baseline_s_compile_ms']:.3f} ms "
        f"m_eager={result['baseline_m_eager_ms']:.3f} ms "
        f"fused_auto={result['fused_auto_ms']:.3f} ms "
        f"auto/sc={result['baseline_s_compile_ms'] / result['fused_auto_ms']:.2f}x "
        f"auto/m={result['baseline_m_eager_ms'] / result['fused_auto_ms']:.2f}x "
        f"{cfg_text} "
        f"spk={correctness['spike_err']:.4%} vmax={correctness['v_max_err']:.2e}"
    )
    if "manual_best" in result:
        manual_best = result["manual_best"]
        manual_cfg = manual_best["spatial_config"]
        print(
            f"{'manual best':<12} T={shape.timesteps:<2} "
            f"manual={manual_best['ms']:.3f} ms "
            f"BTILE_T={manual_best['btile_t']} REUSE_GROUPS={manual_best['reuse_groups']} "
            f"BM={manual_cfg['BLOCK_M']} BOC={manual_cfg['BLOCK_OC']} BK={manual_cfg['BLOCK_K']} "
            f"warps={manual_cfg['num_warps']} stages={manual_cfg['num_stages']}"
        )


def main():
    parser = argparse.ArgumentParser(description="Generalized temporal Conv+LIF fusion benchmark")
    parser.add_argument("--bench", action="store_true", help="run benchmark")
    parser.add_argument("--check", action="store_true", help="run correctness checks")
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--timesteps-sweep", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--shape", choices=("mid", "late", "all"), default="mid")
    parser.add_argument("--manual-search", action="store_true", help="also run Python manual schedule search for debug")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    selected_shapes: List[Tuple[str, ProblemShape]]
    if args.shape == "all":
        selected_shapes = BASE_SHAPES
    else:
        selected_shapes = [(label, shape) for label, shape in BASE_SHAPES if label == args.shape]

    if args.check:
        for timesteps in (2, 4, 8, 16):
            shape = ProblemShape(timesteps, 2, 3, 8, 16, 16)
            torch.manual_seed(0)
            conv = make_conv(shape)
            x_seq = build_input_sequence(shape)
            ref_spikes, ref_v = make_reference(x_seq, conv)
            for btile_t, reuse_groups in valid_temporal_schedules(timesteps):
                spikes, membrane = run_fused_temporal_general(
                    x_seq,
                    conv.weight,
                    conv.bias,
                    temporal_batch_size=btile_t,
                    reuse_groups=reuse_groups,
                )
                summary = summarize_correctness(
                    f"T{timesteps}_B{btile_t}_R{reuse_groups}", spikes, membrane, ref_spikes, ref_v
                )
                validate_summary(summary)
                print(
                    f"check {summary['name']}: spk={summary['spike_err']:.4%} "
                    f"vmax={summary['v_max_err']:.3e} vmean={summary['v_mean_err']:.3e}"
                )
        return

    if not args.bench:
        args.bench = True

    if args.bench:
        print(f"PyTorch: {torch.__version__}")
        print(f"Triton: {triton.__version__}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("general pattern: temporal window = BTILE_T * REUSE_GROUPS")
        print(
            f"{'case':<12} {'T':<3} {'s_eager':>10} {'s_compile':>10} {'m_eager':>10} "
            f"{'fused_auto':>11} {'auto/sc':>8} {'auto/m':>8} autotune_config"
        )
        for base_label, base_shape in selected_shapes:
            for timesteps in args.timesteps_sweep:
                shape = ProblemShape(
                    timesteps,
                    base_shape.batch,
                    base_shape.in_channels,
                    base_shape.out_channels,
                    base_shape.height,
                    base_shape.width,
                )
                result = benchmark_one(base_label, shape, args.repetitions, manual_search=args.manual_search)
                print_result(result)


if __name__ == "__main__":
    main()
