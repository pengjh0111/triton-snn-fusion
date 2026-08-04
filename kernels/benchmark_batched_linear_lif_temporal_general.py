"""Generalized temporal batched Linear+LIF fusion kernel, mirroring the Conv+LIF
temporal schedule framework in benchmark_conv_lif_temporal_general.py.

This module is both the codegen script for
kernels/generated_temporal_batched_linear_lif_kernel.py and the runtime entry point
(`run_fused_batched_linear_lif`) used by runtime/snn_custom_ops.py. Regenerate the
kernel file by re-running this module's codegen block (it executes automatically on
import) -- do not hand-edit the generated file.

Run as a script for benchmarking/manual correctness checks:
    python -m kernels.benchmark_batched_linear_lif_temporal_general --check
    python -m kernels.benchmark_batched_linear_lif_temporal_general --bench
"""

import argparse
import linecache
import os
from typing import Dict, List, Tuple

os.environ.setdefault("TRITON_ALWAYS_COMPILE", "1")
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(os.getcwd(), "aot_result/triton_cache"))
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

DEVICE = "cuda"
TEMPORAL_POW2_CANDIDATES = (1, 2, 4, 8, 16)
MAX_REUSE_GROUPS = 16
DEFAULT_ACC_ELEMS_LIMIT = 32768
# A curated subset of (BTILE_T, REUSE_GROUPS) pairs covering temporal windows
# 1/2/4/8/16, mirroring TEMPORAL_AUTOTUNE_SCHEDULES in the conv codegen. Linear
# has no im2col overhead so we do not additionally prune out large BTILE_T the
# way conv's register budget forces -- the acc-elems limit below does that.
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

LINEAR_AUTOTUNE_SPATIAL_CONFIGS = [
    # The first 7 entries are the exact deduplicated (BLOCK_M, BLOCK_N,
    # BLOCK_K, num_warps, num_stages) spatial pool used by the old TC kernel's
    # _linear_configs() in generated_temporal_transformer_lif_kernels.py (its
    # 19-config list crossed with TC in {1,2,4} reduces to these 7 unique
    # spatial tuples). Kept byte-for-byte so this kernel's autotune space is a
    # strict superset of the old kernel's -- autotune should never need to
    # pick a worse spatial tile than TC could.
    {"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 32, "num_warps": 4, "num_stages": 2},
    {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64, "num_warps": 4, "num_stages": 3},
    {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 64, "num_warps": 4, "num_stages": 3},
    {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64, "num_warps": 4, "num_stages": 3},
    {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 4, "num_stages": 3},
    {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 64, "num_warps": 4, "num_stages": 3},
    {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "num_warps": 4, "num_stages": 3},
    # Extra tiles beyond the TC pool -- covers the very-small-rows and
    # large-BLOCK_N/BLOCK_K corners TC's sweep doesn't reach.
    {"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 4, "num_stages": 3},
    {"BLOCK_M": 8, "BLOCK_N": 64, "BLOCK_K": 128, "num_warps": 4, "num_stages": 3},
]


def _acc_elems_limit() -> int:
    raw = os.environ.get("KAIROS_BATCHED_LINEAR_ACC_ELEMS_LIMIT")
    if raw is None:
        return DEFAULT_ACC_ELEMS_LIMIT
    try:
        return int(raw)
    except ValueError:
        return DEFAULT_ACC_ELEMS_LIMIT


def _make_autotune_configs() -> List["triton.Config"]:
    configs = []
    for spatial in LINEAR_AUTOTUNE_SPATIAL_CONFIGS:
        for btile_t, reuse_groups in TEMPORAL_AUTOTUNE_SCHEDULES:
            configs.append(
                triton.Config(
                    {
                        "BLOCK_M": spatial["BLOCK_M"],
                        "BLOCK_N": spatial["BLOCK_N"],
                        "BLOCK_K": spatial["BLOCK_K"],
                        "BTILE_T": btile_t,
                        "REUSE_GROUPS": reuse_groups,
                    },
                    num_warps=spatial["num_warps"],
                    num_stages=spatial["num_stages"],
                )
            )
    return configs


def _acc_elems(config) -> int:
    kw = config.all_kwargs()
    return (
        int(kw["BTILE_T"])
        * int(kw["REUSE_GROUPS"])
        * int(kw["BLOCK_M"])
        * int(kw["BLOCK_N"])
    )


# Conservative headroom under the ~99-101KB opt-in shared memory limit typical
# of consumer/data-center GPUs. Each of the REUSE_GROUPS separate X loads in
# the K-loop gets its own multi-buffered (num_stages) shared memory staging
# area in Triton's software pipeliner, so REUSE_GROUPS multiplies shared
# memory pressure independently of the accumulator (register) budget that
# _acc_elems_limit already bounds -- large REUSE_GROUPS with num_stages>=2 can
# blow the hardware shared memory limit well before it blows the accumulator
# budget, so this needs its own guard.
DEFAULT_SHARED_MEM_BUDGET = 96 * 1024


def _shared_mem_budget() -> int:
    raw = os.environ.get("KAIROS_BATCHED_LINEAR_SHARED_MEM_BUDGET")
    if raw is None:
        return DEFAULT_SHARED_MEM_BUDGET
    try:
        return int(raw)
    except ValueError:
        return DEFAULT_SHARED_MEM_BUDGET


def _shared_mem_estimate(btile_t: int, reuse_groups: int, block_m: int, block_k: int, block_n: int, num_stages: int, dtype_bytes: int) -> int:
    # REUSE_GROUPS separately-staged X tiles of shape (BLOCK_M*BTILE_T, BLOCK_K)
    # plus one shared W tile of shape (BLOCK_K, BLOCK_N), each multi-buffered
    # num_stages deep by the software pipeliner.
    return int(num_stages) * int(block_k) * (int(reuse_groups) * int(block_m) * int(btile_t) + int(block_n)) * int(dtype_bytes)


def _config_shared_mem_estimate(config, dtype_bytes: int) -> int:
    kw = config.all_kwargs()
    return _shared_mem_estimate(
        int(kw["BTILE_T"]),
        int(kw["REUSE_GROUPS"]),
        int(kw["BLOCK_M"]),
        int(kw["BLOCK_K"]),
        int(kw["BLOCK_N"]),
        int(getattr(config, "num_stages", 3)),
        dtype_bytes,
    )


def _prune_batched_linear_configs(configs, named_args, **kwargs):
    merged = {**named_args, **kwargs}
    T_STEPS = int(merged["T_STEPS"])
    rows = int(merged["rows"])
    dtype_bytes = 4 if bool(merged.get("USE_TF32", True)) else 2

    window_valid = [
        config
        for config in configs
        if int(config.all_kwargs()["BTILE_T"]) * int(config.all_kwargs()["REUSE_GROUPS"]) <= T_STEPS
    ]
    forced_p = os.environ.get("KAIROS_EVAL_FORCE_BTILE_T")
    forced_r = os.environ.get("KAIROS_EVAL_FORCE_REUSE_GROUPS")
    if forced_p is not None:
        window_valid = [
            config for config in window_valid
            if int(config.all_kwargs()["BTILE_T"]) == int(forced_p)
        ]
    if forced_r is not None:
        window_valid = [
            config for config in window_valid
            if int(config.all_kwargs()["REUSE_GROUPS"]) == int(forced_r)
        ]
    if not window_valid:
        if forced_p is None and forced_r is None:
            return configs[:1]
        raise RuntimeError(
            "No batched-linear autotune config satisfies the evaluation "
            "BTILE_T/REUSE_GROUPS constraint"
        )

    shared_budget = _shared_mem_budget()
    shared_valid = [config for config in window_valid if _config_shared_mem_estimate(config, dtype_bytes) <= shared_budget]
    if not shared_valid:
        # Guaranteed-safe fallback: keep the single smallest-footprint config
        # rather than let an OutOfResources compile error propagate.
        shared_valid = [min(window_valid, key=lambda c: _config_shared_mem_estimate(c, dtype_bytes))]

    acc_limit = _acc_elems_limit()
    valid = shared_valid
    if acc_limit > 0:
        limited = [config for config in shared_valid if _acc_elems(config) <= acc_limit]
        valid = limited or shared_valid

    # NOTE: an earlier "rows-aware bias" here hard-excluded window==1 configs
    # whenever rows <= 128, on the assumption that small rows always benefits
    # from temporal batching. A manual schedule sweep (see
    # dev_tests/diag_rows1_T16_schedule_sweep.py) disproved this for the
    # rows=1 extreme: the genuinely fastest config there is BTILE_T=1,
    # REUSE_GROUPS=1 with a wide BLOCK_N -- at rows=1 almost the entire
    # BLOCK_M tile is padding regardless of BTILE_T, so folding more
    # timesteps into M or adding REUSE_GROUPS accumulators only adds
    # generated-code branch/instruction overhead without amortizing anything,
    # since weight-load reuse was never the bottleneck at this occupancy. A
    # hard filter that excludes a region already proven to win is not
    # acceptable pruning, so this now only applies the resource-budget limits
    # above (shared memory, accumulator elements) and leaves the full
    # window space, including window==1, for autotune to actually benchmark.
    return valid


def _default_spatial_config_for_shape(rows: int, in_features: int, out_features: int) -> Dict[str, int]:
    if rows <= 64:
        return {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64, "num_warps": 4, "num_stages": 3}
    if rows <= 1024:
        return {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64, "num_warps": 4, "num_stages": 3}
    return {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 64, "num_warps": 4, "num_stages": 3}


_FALLBACK_SAFE_SPATIAL_CONFIG = {"BLOCK_M": 8, "BLOCK_N": 32, "BLOCK_K": 32, "num_warps": 4, "num_stages": 2}


def _default_schedule_for_shape(T: int, rows: int, dtype_bytes: int = 4) -> Tuple[Dict[str, int], int, int]:
    """Pick a (spatial_config, BTILE_T, REUSE_GROUPS) triple for the
    non-autotune specialized kernel path that both matches the rows-aware
    temporal bias used by the autotune prune and stays within the shared
    memory budget -- unlike a schedule chosen in isolation from the spatial
    config, which can silently pick a REUSE_GROUPS/num_stages combination
    that overflows shared memory (see _shared_mem_estimate).
    """
    if T <= 1:
        return _default_spatial_config_for_shape(rows, 0, 0), 1, 1

    if rows <= 128:
        temporal_priority = [(1, rg) for rg in (16, 8, 4, 2, 1)]
    else:
        temporal_priority = [(bt, 1) for bt in (4, 2, 1)]

    spatial_priority = [_default_spatial_config_for_shape(rows, 0, 0)] + LINEAR_AUTOTUNE_SPATIAL_CONFIGS
    budget = _shared_mem_budget()
    for btile_t, reuse_groups in temporal_priority:
        if btile_t * reuse_groups > T:
            continue
        for cfg in spatial_priority:
            estimate = _shared_mem_estimate(
                btile_t, reuse_groups, cfg["BLOCK_M"], cfg["BLOCK_K"], cfg["BLOCK_N"], cfg.get("num_stages", 3), dtype_bytes
            )
            if estimate <= budget:
                return cfg, btile_t, reuse_groups
    return _FALLBACK_SAFE_SPATIAL_CONFIG, 1, 1


def _emit_batched_linear_lif_kernel_source(
    max_groups: int = MAX_REUSE_GROUPS,
    function_name: str = "_fused_batched_linear_lif_temporal_general_kernel_impl",
) -> str:
    """Generate a static-unrolled Triton kernel for batched temporal Linear+LIF.

    Mirrors _emit_general_kernel_source in benchmark_conv_lif_temporal_general.py:
    Triton tensors cannot be mutated through Python containers inside jit code
    (`acc_groups[g] = ...` is unsupported), so this generator keeps one
    generalized pattern in Python and emits a static specialization body for
    REUSE_GROUPS up to `max_groups` and BTILE_T up to 16.

    Unlike the conv generator, there is no im2col here -- X tiles are plain
    strided loads off x_ptr[T, rows, in_features] -- and HAS_RESIDUAL is a
    runtime constexpr on a single generated kernel (not a second `_resadd`
    function), so the plain Linear+LIF and Linear+Add+LIF custom ops share the
    same compiled template.
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
        shape = ", ".join(["2"] * levels + ["BLOCK_M", "BLOCK_N"])
        emit_split_tree(f"acc_g{g}.reshape([{shape}])", levels, g, (), indent + "    ")

    lines.append(f"def {function_name}(")
    lines.append("    x_ptr, residual_ptr, w_ptr, b_ptr, v_ptr, spike_ptr, v_last_ptr,")
    lines.append("    rows, in_features: tl.constexpr, out_features,")
    lines.append("    v_threshold, v_reset, tau_inv,")
    lines.append("    T_STEPS: tl.constexpr,")
    lines.append("    HAS_BIAS: tl.constexpr, HAS_RESIDUAL: tl.constexpr,")
    lines.append("    TAU_LE_ONE: tl.constexpr, SOFT_RESET: tl.constexpr,")
    lines.append("    V_INIT_IS_SCALAR: tl.constexpr, USE_TF32: tl.constexpr,")
    lines.append("    AUTOTUNE_VERSION: tl.constexpr,")
    lines.append("    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,")
    lines.append("    BTILE_T: tl.constexpr, REUSE_GROUPS: tl.constexpr,")
    lines.append("):")
    lines.append("    pid_m = tl.program_id(0)")
    lines.append("    pid_n = tl.program_id(1)")
    lines.append("    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)")
    lines.append("    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)")
    lines.append("    m_mask = m_offsets < rows")
    lines.append("    n_mask = n_offsets < out_features")
    lines.append("    BM_T: tl.constexpr = BLOCK_M * BTILE_T")
    lines.append("    WINDOW_T: tl.constexpr = BTILE_T * REUSE_GROUPS")
    lines.append("    if HAS_BIAS:")
    lines.append("        bias = tl.load(b_ptr + n_offsets, mask=n_mask, other=0.0).to(tl.float32)")
    lines.append("    out_offsets = m_offsets[:, None] * out_features + n_offsets[None, :]")
    lines.append("    if V_INIT_IS_SCALAR:")
    lines.append("        v_state = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)")
    lines.append("    else:")
    lines.append("        v_state = tl.load(v_ptr + out_offsets, mask=m_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float32)")
    lines.append("    cat_offsets = tl.arange(0, BM_T)")
    lines.append("    local_t = cat_offsets // BLOCK_M")
    lines.append("    local_m = cat_offsets % BLOCK_M")
    lines.append("    cat_m_offsets = pid_m * BLOCK_M + local_m")
    lines.append("    cat_m_mask = cat_m_offsets < rows")
    lines.append("    for temporal_base in range(0, T_STEPS, WINDOW_T):")
    for g in range(max_groups):
        lines.append(f"        if REUSE_GROUPS >= {g + 1}:")
        lines.append(f"            acc_g{g} = tl.zeros((BM_T, BLOCK_N), dtype=tl.float32)")
    lines.append("        for k_start in range(0, in_features, BLOCK_K):")
    lines.append("            k_offsets = k_start + tl.arange(0, BLOCK_K)")
    lines.append("            k_mask = k_offsets < in_features")
    lines.append("            w_offsets = n_offsets[None, :] * in_features + k_offsets[:, None]")
    lines.append("            w_tile = tl.load(w_ptr + w_offsets, mask=k_mask[:, None] & n_mask[None, :], other=0.0)")
    for g in range(max_groups):
        lines.append(f"            if REUSE_GROUPS >= {g + 1}:")
        lines.append(f"                step_g{g} = temporal_base + {g} * BTILE_T + local_t")
        lines.append(
            f"                x_offsets_g{g} = step_g{g}[:, None] * rows * in_features + "
            f"cat_m_offsets[:, None] * in_features + k_offsets[None, :]"
        )
        lines.append(
            f"                x_g{g} = tl.load(x_ptr + x_offsets_g{g}, "
            f"mask=cat_m_mask[:, None] & (step_g{g}[:, None] < T_STEPS) & k_mask[None, :], other=0.0)"
        )
        lines.append("                if USE_TF32:")
        lines.append(f"                    acc_g{g} = tl.dot(x_g{g}, w_tile, acc_g{g}, input_precision='tf32')")
        lines.append("                else:")
        lines.append(f"                    acc_g{g} = tl.dot(x_g{g}, w_tile, acc_g{g})")
    for g in range(max_groups):
        lines.append(f"        if REUSE_GROUPS >= {g + 1}:")
        for idx, btile_t in enumerate(TEMPORAL_POW2_CANDIDATES):
            emit_btile_split(g, btile_t, "if" if idx == 0 else "elif")
        for bt in range(max(TEMPORAL_POW2_CANDIDATES)):
            lines.append(f"            if BTILE_T >= {bt + 1}:")
            lines.append(f"                step = temporal_base + {g} * BTILE_T + {bt}")
            lines.append("                if step < T_STEPS:")
            lines.append(f"                    acc_t = acc_g{g}_{bt}")
            lines.append("                    if HAS_BIAS:")
            lines.append("                        acc_t += bias[None, :]")
            lines.append("                    if HAS_RESIDUAL:")
            lines.append("                        residual_offsets = step * rows * out_features + out_offsets")
            lines.append(
                "                        residual_t = tl.load(residual_ptr + residual_offsets, "
                "mask=m_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float32)"
            )
            lines.append("                        acc_t += residual_t")
            lines.append("                    if TAU_LE_ONE:")
            lines.append("                        v_before_spike = v_state + acc_t")
            lines.append("                    else:")
            lines.append("                        v_before_spike = v_state + (acc_t - v_state) * tau_inv")
            lines.append("                    pred = v_before_spike >= v_threshold")
            lines.append("                    spike = pred.to(tl.float32)")
            lines.append("                    if SOFT_RESET:")
            lines.append("                        v_state = v_before_spike - spike * v_threshold")
            lines.append("                    else:")
            lines.append("                        v_state = tl.where(pred, v_reset, v_before_spike)")
            lines.append("                    spike_offsets = step * rows * out_features + out_offsets")
            lines.append("                    tl.store(spike_ptr + spike_offsets, spike, mask=m_mask[:, None] & n_mask[None, :])")
    lines.append("    tl.store(v_last_ptr + out_offsets, v_state, mask=m_mask[:, None] & n_mask[None, :])")
    return "\n".join(lines)


_FUNCTION_NAME = "_fused_batched_linear_lif_temporal_general_kernel_impl"
_kernel_namespace = {"tl": tl}
_kernel_source = _emit_batched_linear_lif_kernel_source(function_name=_FUNCTION_NAME)
_kernel_filename = os.path.join(os.path.dirname(__file__), "generated_temporal_batched_linear_lif_kernel.py")
with open(_kernel_filename, "w", encoding="utf-8") as _kernel_file:
    _kernel_file.write(
        "# Generated by kernels/benchmark_batched_linear_lif_temporal_general.py\n"
        "# Regenerate with: python -m kernels.benchmark_batched_linear_lif_temporal_general --check\n"
        "# Do not hand-edit; edit the emitter (_emit_batched_linear_lif_kernel_source) instead.\n\n"
    )
    _kernel_file.write(_kernel_source)
    _kernel_file.write("\n")
linecache.cache[_kernel_filename] = (
    len(_kernel_source),
    None,
    [line + "\n" for line in _kernel_source.splitlines()],
    _kernel_filename,
)
exec(compile(_kernel_source, _kernel_filename, "exec"), _kernel_namespace)
_kernel_fn = _kernel_namespace[_FUNCTION_NAME]

_specialized_kernels = {"batched_linear_lif": triton.jit(_kernel_fn)}
_autotuned_kernels = {
    "batched_linear_lif": triton.autotune(
        configs=_make_autotune_configs(),
        key=["rows", "in_features", "out_features", "T_STEPS", "HAS_RESIDUAL", "HAS_BIAS", "USE_TF32", "AUTOTUNE_VERSION"],
        prune_configs_by={"early_config_prune": _prune_batched_linear_configs},
        cache_results=True,
    )(triton.jit(_kernel_fn))
}


def get_autotune_best_config():
    kernel = _autotuned_kernels["batched_linear_lif"]
    best_config = getattr(kernel, "best_config", None)
    if best_config is None:
        return None
    kw = best_config.all_kwargs()
    btile_t = kw.get("BTILE_T")
    reuse_groups = kw.get("REUSE_GROUPS")
    window = int(btile_t) * int(reuse_groups) if btile_t is not None and reuse_groups is not None else None
    return {
        "kernel_key": "batched_linear_lif",
        "BLOCK_M": kw.get("BLOCK_M"),
        "BLOCK_N": kw.get("BLOCK_N"),
        "BLOCK_K": kw.get("BLOCK_K"),
        "BTILE_T": btile_t,
        "REUSE_GROUPS": reuse_groups,
        "kernel_temporal_window": window,
        "acc_elems": (
            int(btile_t) * int(reuse_groups) * int(kw.get("BLOCK_M")) * int(kw.get("BLOCK_N"))
            if btile_t is not None and reuse_groups is not None and kw.get("BLOCK_M") is not None and kw.get("BLOCK_N") is not None
            else None
        ),
        "num_warps": kw.get("num_warps"),
        "num_stages": kw.get("num_stages"),
    }


def run_fused_batched_linear_lif(
    x_seq: torch.Tensor,
    weight: torch.Tensor,
    bias,
    v_init: torch.Tensor,
    v_threshold: float,
    v_reset: float,
    tau: float,
    *,
    residual_seq: torch.Tensor = None,
    use_autotune: bool = True,
    spatial_config: Dict[str, int] = None,
    btile_t: int = None,
    reuse_groups: int = None,
    spikes_out: torch.Tensor = None,
    v_last_out: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if x_seq.dim() < 4:
        raise RuntimeError(f"batched temporal Linear+LIF expects [T,...,Fin] with rank >= 4, got {x_seq.dim()}")
    if weight.dim() != 2 or int(x_seq.shape[-1]) != int(weight.shape[1]):
        raise RuntimeError("batched temporal Linear+LIF weight/input feature mismatch")
    x_seq = x_seq.contiguous()
    T = int(x_seq.shape[0])
    leading = tuple(x_seq.shape[1:-1])
    in_features = int(x_seq.shape[-1])
    rows = x_seq.numel() // (T * in_features)
    out_features = int(weight.shape[0])
    out_shape = (T,) + leading + (out_features,)
    state_shape = leading + (out_features,)

    if spikes_out is None:
        spikes = torch.empty(out_shape, device=x_seq.device, dtype=x_seq.dtype)
    else:
        if tuple(spikes_out.shape) != out_shape:
            raise RuntimeError(f"spikes_out shape {tuple(spikes_out.shape)} does not match expected {out_shape}")
        spikes = spikes_out
    if v_last_out is None:
        v_last = torch.empty(state_shape, device=x_seq.device, dtype=x_seq.dtype)
    else:
        if tuple(v_last_out.shape) != state_shape:
            raise RuntimeError(f"v_last_out shape {tuple(v_last_out.shape)} does not match expected {state_shape}")
        v_last = v_last_out

    residual = weight
    has_residual = residual_seq is not None
    if has_residual:
        if tuple(residual_seq.shape) != out_shape:
            raise RuntimeError(f"residual shape {tuple(residual_seq.shape)} does not match {out_shape}")
        residual = residual_seq.contiguous()

    has_bias = isinstance(bias, torch.Tensor) and bias.numel() > 0
    bias_arg = bias.contiguous() if has_bias else weight

    scalar_v = v_init.dim() == 0
    if not scalar_v and tuple(v_init.shape) != state_shape:
        raise RuntimeError(f"v_init shape {tuple(v_init.shape)} does not match {state_shape}")
    v_init_arg = v_init.contiguous()
    weight_c = weight.contiguous()

    grid = lambda meta: (triton.cdiv(rows, meta["BLOCK_M"]), triton.cdiv(out_features, meta["BLOCK_N"]))
    common_kwargs = dict(
        rows=rows,
        in_features=in_features,
        out_features=out_features,
        v_threshold=float(v_threshold),
        v_reset=float(v_reset),
        tau_inv=1.0 / float(tau),
        T_STEPS=T,
        HAS_BIAS=has_bias,
        HAS_RESIDUAL=has_residual,
        TAU_LE_ONE=float(tau) <= 1.0,
        SOFT_RESET=float(v_reset) < 0.0,
        V_INIT_IS_SCALAR=scalar_v,
        USE_TF32=x_seq.dtype == torch.float32,
        AUTOTUNE_VERSION=1,
    )

    if use_autotune:
        kernel = _autotuned_kernels["batched_linear_lif"]
        kernel[grid](
            x_seq, residual, weight_c, bias_arg, v_init_arg, spikes, v_last,
            **common_kwargs,
        )
    else:
        dtype_bytes = 4 if x_seq.dtype == torch.float32 else 2
        if spatial_config is None and (btile_t is None or reuse_groups is None):
            default_cfg, default_bt, default_rg = _default_schedule_for_shape(T, rows, dtype_bytes)
            cfg = default_cfg
            bt = btile_t if btile_t is not None else default_bt
            rg = reuse_groups if reuse_groups is not None else default_rg
        else:
            cfg = spatial_config or _default_spatial_config_for_shape(rows, in_features, out_features)
            bt, rg = btile_t, reuse_groups
            if bt is None or rg is None:
                _, default_bt, default_rg = _default_schedule_for_shape(T, rows, dtype_bytes)
                bt = bt if bt is not None else default_bt
                rg = rg if rg is not None else default_rg
        if bt * rg > T:
            bt, rg = 1, 1
        if bt > MAX_REUSE_GROUPS or rg > MAX_REUSE_GROUPS:
            raise ValueError(f"btile_t/reuse_groups exceed generated max_groups={MAX_REUSE_GROUPS}")
        kernel = _specialized_kernels["batched_linear_lif"]
        kernel[grid](
            x_seq, residual, weight_c, bias_arg, v_init_arg, spikes, v_last,
            BTILE_T=bt,
            REUSE_GROUPS=rg,
            BLOCK_M=cfg["BLOCK_M"],
            BLOCK_N=cfg["BLOCK_N"],
            BLOCK_K=cfg["BLOCK_K"],
            num_warps=cfg.get("num_warps", 4),
            num_stages=cfg.get("num_stages", 3),
            **common_kwargs,
        )
    return spikes, v_last


def valid_temporal_schedules(timesteps: int):
    schedules = []
    for btile_t in TEMPORAL_POW2_CANDIDATES:
        for reuse_groups in TEMPORAL_POW2_CANDIDATES:
            if btile_t * reuse_groups <= timesteps and btile_t * reuse_groups <= MAX_REUSE_GROUPS * 16:
                schedules.append((btile_t, reuse_groups))
    return schedules


def reference_batched_linear_lif(
    x_seq: torch.Tensor,
    weight: torch.Tensor,
    bias,
    v_init: torch.Tensor,
    v_threshold: float,
    v_reset: float,
    tau: float,
    residual_seq: torch.Tensor = None,
):
    T = int(x_seq.shape[0])
    if v_init.dim() == 0:
        first = F.linear(x_seq[0], weight, bias if isinstance(bias, torch.Tensor) and bias.numel() > 0 else None)
        v = torch.zeros_like(first)
    else:
        v = v_init.clone()
    spikes = []
    tau_inv = 1.0 / float(tau)
    for t in range(T):
        current = F.linear(x_seq[t], weight, bias if isinstance(bias, torch.Tensor) and bias.numel() > 0 else None)
        if residual_seq is not None:
            current = current + residual_seq[t]
        if float(tau) <= 1.0:
            v_before_spike = v + current
        else:
            v_before_spike = v + (current - v) * tau_inv
        spike = (v_before_spike >= float(v_threshold)).to(x_seq.dtype)
        if float(v_reset) < 0:
            v = v_before_spike - spike * float(v_threshold)
        else:
            v = torch.where(spike.bool(), torch.full_like(v_before_spike, float(v_reset)), v_before_spike)
        spikes.append(spike)
    return torch.stack(spikes, dim=0), v


def _time_cuda(fn, warmup: int = 10, rep: int = 50) -> float:
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


def _run_checks():
    torch.manual_seed(0)
    cases = [
        (1, (1, 1), 384, 1000),
        (2, (1, 16), 512, 512),
        (4, (1, 16), 384, 1000),
        (4, (8, 8), 4096, 4096),
        (8, (2, 3, 5), 300, 700),
        (16, (1, 4), 256, 256),
        (3, (1, 1), 128, 128),  # not a pow2 -- exercises the generic BTILE_T=1,REUSE_GROUPS=1 fallback path
    ]
    # Autotune sweeps re-search the whole config space per unique
    # (shape, dtype, HAS_BIAS, HAS_RESIDUAL) key, so exercise it once per case
    # (richest bias/residual combo) instead of the full matrix below -- the
    # matrix itself is covered via the (cheap, single-config) specialized path,
    # which drives the exact same generated kernel body.
    for T, leading, in_features, out_features in cases:
        x_seq = torch.randn((T,) + leading + (in_features,), device=DEVICE, dtype=torch.float16) * 0.05
        weight = torch.randn(out_features, in_features, device=DEVICE, dtype=torch.float16) * 0.02
        bias = torch.randn(out_features, device=DEVICE, dtype=torch.float16) * 0.01
        v_init = torch.zeros(leading + (out_features,), device=DEVICE, dtype=torch.float16)
        residual_seq = torch.randn((T,) + leading + (out_features,), device=DEVICE, dtype=torch.float16) * 0.05
        ref_spikes, ref_v = reference_batched_linear_lif(
            x_seq, weight, bias, v_init, 1.0, 0.0, 2.0, residual_seq=residual_seq
        )
        spikes, v_last = run_fused_batched_linear_lif(
            x_seq, weight, bias, v_init, 1.0, 0.0, 2.0, residual_seq=residual_seq, use_autotune=True,
        )
        spike_err = (spikes.float() != ref_spikes.float()).float().mean().item()
        v_err = (v_last.float() - ref_v.float()).abs().max().item()
        ok = spike_err < 1e-3 and v_err < 2e-2
        cfg = get_autotune_best_config()
        tag = f"T={T} leading={leading} shape={in_features}->{out_features} autotune=True cfg={cfg}"
        print(f"{'OK  ' if ok else 'FAIL'} {tag} spike_err={spike_err:.4%} v_err={v_err:.3e}")
        if not ok:
            raise AssertionError(f"correctness check failed: {tag}")

    for T, leading, in_features, out_features in cases:
        for has_bias in (False, True):
            for has_residual in (False, True):
                for dtype in (torch.float32, torch.float16):
                    x_seq = torch.randn((T,) + leading + (in_features,), device=DEVICE, dtype=dtype) * 0.05
                    weight = torch.randn(out_features, in_features, device=DEVICE, dtype=dtype) * 0.02
                    bias = torch.randn(out_features, device=DEVICE, dtype=dtype) * 0.01 if has_bias else torch.empty(0, device=DEVICE, dtype=dtype)
                    v_init = torch.zeros(leading + (out_features,), device=DEVICE, dtype=dtype)
                    residual_seq = None
                    if has_residual:
                        residual_seq = torch.randn((T,) + leading + (out_features,), device=DEVICE, dtype=dtype) * 0.05

                    ref_spikes, ref_v = reference_batched_linear_lif(
                        x_seq, weight, bias, v_init, 1.0, 0.0, 2.0, residual_seq=residual_seq
                    )
                    for bt, rg in valid_temporal_schedules(T)[:4] or [(1, 1)]:
                        spikes, v_last = run_fused_batched_linear_lif(
                            x_seq, weight, bias, v_init, 1.0, 0.0, 2.0,
                            residual_seq=residual_seq, use_autotune=False, btile_t=bt, reuse_groups=rg,
                        )
                        atol = 2e-2 if dtype == torch.float16 else 2e-3
                        spike_err = (spikes.float() != ref_spikes.float()).float().mean().item()
                        v_err = (v_last.float() - ref_v.float()).abs().max().item()
                        ok = spike_err < 1e-3 and v_err < atol
                        tag = (
                            f"T={T} leading={leading} shape={in_features}->{out_features} bias={has_bias} "
                            f"residual={has_residual} dtype={dtype} BTILE_T={bt} REUSE_GROUPS={rg}"
                        )
                        print(f"{'OK  ' if ok else 'FAIL'} {tag} spike_err={spike_err:.4%} v_err={v_err:.3e}")
                        if not ok:
                            raise AssertionError(f"correctness check failed: {tag}")


def main():
    parser = argparse.ArgumentParser(description="Generalized temporal batched Linear+LIF fusion benchmark")
    parser.add_argument("--check", action="store_true", help="run correctness checks")
    parser.add_argument("--bench", action="store_true", help="run a quick perf sweep")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    if args.check or not args.bench:
        _run_checks()

    if args.bench:
        shapes = [
            (16, (1, 1), 768, 3072),
            (16, (1, 16), 768, 3072),
            (16, (1, 197), 768, 768),
            (4, (8, 128), 4096, 4096),
        ]
        for T, leading, in_features, out_features in shapes:
            x_seq = torch.randn((T,) + leading + (in_features,), device=DEVICE, dtype=torch.float16) * 0.05
            weight = torch.randn(out_features, in_features, device=DEVICE, dtype=torch.float16) * 0.02
            bias = torch.randn(out_features, device=DEVICE, dtype=torch.float16) * 0.01
            v_init = torch.zeros(leading + (out_features,), device=DEVICE, dtype=torch.float16)
            ms = _time_cuda(lambda: run_fused_batched_linear_lif(x_seq, weight, bias, v_init, 1.0, 0.0, 2.0))
            cfg = get_autotune_best_config()
            print(f"T={T} leading={leading} shape={in_features}->{out_features} ms={ms:.4f} cfg={cfg}")


if __name__ == "__main__":
    main()
