import os
from typing import Tuple

import torch
import triton
import triton.language as tl


def _linear_configs():
    configs = []
    for tc in (1, 2, 4):
        for bm, bn, bk, warps, stages in (
            (16, 32, 32, 4, 2),
            (16, 64, 64, 4, 3),
            (32, 32, 64, 4, 3),
            (32, 64, 64, 4, 3),
            (32, 128, 64, 4, 3),
            (64, 32, 64, 4, 3),
            (64, 64, 64, 4, 3),
        ):
            # Keep the temporal accumulator pressure bounded.  The LIF state is
            # only [BM, BN], but the GEMM accumulator is [TC * BM, BN].
            if tc * bm * bn > 8192:
                continue
            configs.append(
                triton.Config(
                    {"TC": tc, "BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
                    num_warps=warps,
                    num_stages=stages,
                )
            )
    return configs


def _prune_linear_configs(configs, named_args, **kwargs):
    merged = {**named_args, **kwargs}
    T = int(merged["T"])
    in_features = int(merged["in_features"])
    out_features = int(merged["out_features"])
    valid = []
    for config in configs:
        values = config.all_kwargs()
        tc = int(values["TC"])
        block_k = int(values["BLOCK_K"])
        block_n = int(values["BLOCK_N"])
        if tc > T or T % tc != 0:
            continue
        forced_p = os.environ.get("KAIROS_EVAL_FORCE_BTILE_T")
        if forced_p is not None and tc != int(forced_p):
            continue
        if block_k > triton.next_power_of_2(in_features) * 2:
            continue
        if out_features < 1024 and block_n > 128:
            continue
        valid.append(config)
    assert valid, f"all configs pruned: T={T} in={in_features} out={out_features}"
    return valid


def _linear_convstyle4_configs():
    configs = []
    for btile_t, reuse_groups in ((1, 4), (2, 2), (4, 1)):
        for bm, bn, bk, warps, stages in (
            (16, 32, 32, 4, 2),
            (16, 64, 64, 4, 3),
            (32, 32, 64, 4, 3),
            (32, 64, 64, 4, 3),
            (32, 128, 64, 4, 3),
            (64, 32, 64, 4, 3),
            (64, 64, 64, 4, 3),
        ):
            if reuse_groups * btile_t * bm * bn > 16384:
                continue
            configs.append(
                triton.Config(
                    {
                        "BTILE_T": btile_t,
                        "REUSE_GROUPS": reuse_groups,
                        "BLOCK_M": bm,
                        "BLOCK_N": bn,
                        "BLOCK_K": bk,
                    },
                    num_warps=warps,
                    num_stages=stages,
                )
            )
    return configs


def _prune_linear_convstyle4_configs(configs, named_args, **kwargs):
    merged = {**named_args, **kwargs}
    T = int(merged["T"])
    in_features = int(merged["in_features"])
    out_features = int(merged["out_features"])
    valid = []
    for config in configs:
        values = config.all_kwargs()
        forced_p = os.environ.get("KAIROS_EVAL_FORCE_BTILE_T")
        forced_r = os.environ.get("KAIROS_EVAL_FORCE_REUSE_GROUPS")
        if forced_p is not None and int(values["BTILE_T"]) != int(forced_p):
            continue
        if forced_r is not None and int(values["REUSE_GROUPS"]) != int(forced_r):
            continue
        block_k = int(values["BLOCK_K"])
        block_n = int(values["BLOCK_N"])
        if T != 4:
            continue
        if block_k > triton.next_power_of_2(in_features) * 2:
            continue
        if out_features < 1024 and block_n > 128:
            continue
        valid.append(config)
    assert valid, f"all convstyle4 configs pruned: T={T} in={in_features} out={out_features}"
    return valid


@triton.autotune(
    configs=_linear_configs(),
    key=["rows", "in_features", "out_features", "T", "HAS_RESIDUAL", "HAS_BIAS", "AUTOTUNE_VERSION"],
    prune_configs_by={"early_config_prune": _prune_linear_configs},
    cache_results=True,
)
@triton.jit
def _temporal_batched_linear_lif_kernel(
    x_seq, residual_seq, weight, bias, v_init, spike_seq, v_last,
    rows: tl.constexpr, in_features: tl.constexpr, out_features: tl.constexpr,
    v_threshold, v_reset, tau_inv,
    T: tl.constexpr, HAS_BIAS: tl.constexpr, HAS_RESIDUAL: tl.constexpr,
    TAU_LE_ONE: tl.constexpr, SOFT_RESET: tl.constexpr, USE_TF32: tl.constexpr,
    V_INIT_IS_SCALAR: tl.constexpr,
    AUTOTUNE_VERSION: tl.constexpr,
    TC: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    ms = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    ns = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = ms < rows
    n_mask = ns < out_features
    out_offsets = ms[:, None] * out_features + ns[None, :]
    if V_INIT_IS_SCALAR:
        v = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    else:
        v = tl.load(v_init + out_offsets, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
    if HAS_BIAS:
        bias_tile = tl.load(bias + ns, mask=n_mask, other=0.0)

    MT: tl.constexpr = BLOCK_M * TC
    cat = tl.arange(0, MT)
    ct = cat // BLOCK_M
    cm = cat % BLOCK_M
    cat_ms = pid_m * BLOCK_M + cm
    cat_mask = cat_ms < rows

    for tb in tl.static_range(0, T, TC):
        acc = tl.zeros((MT, BLOCK_N), tl.float32)
        for k0 in range(0, in_features, BLOCK_K):
            ks = k0 + tl.arange(0, BLOCK_K)
            k_mask = ks < in_features
            x = tl.load(
                x_seq + (tb + ct[:, None]) * rows * in_features + cat_ms[:, None] * in_features + ks[None, :],
                mask=cat_mask[:, None] & k_mask[None, :], other=0.0,
            )
            w = tl.load(
                weight + ns[None, :] * in_features + ks[:, None],
                mask=k_mask[:, None] & n_mask[None, :], other=0.0,
            )
            if USE_TF32:
                acc = tl.dot(x, w, acc, input_precision="tf32")
            else:
                acc = tl.dot(x, w, acc)

        if TC == 1:
            acc0 = acc
            if HAS_BIAS:
                acc0 += bias_tile[None, :]
            if HAS_RESIDUAL:
                residual0 = tl.load(
                    residual_seq + tb * rows * out_features + out_offsets,
                    mask=m_mask[:, None] & n_mask[None, :], other=0.0,
                )
                acc0 += residual0
            if TAU_LE_ONE:
                v_before_spike = v + acc0
            else:
                v_before_spike = v + (acc0 - v) * tau_inv
            pred = v_before_spike >= v_threshold
            spike = pred.to(tl.float32)
            if SOFT_RESET:
                v = v_before_spike - spike * v_threshold
            else:
                v = tl.where(pred, v_reset, v_before_spike)
            tl.store(
                spike_seq + tb * rows * out_features + out_offsets,
                spike, mask=m_mask[:, None] & n_mask[None, :],
            )

        elif TC == 2:
            acc0, acc1 = acc.reshape([2, BLOCK_M, BLOCK_N]).permute(1, 2, 0).split()
            if HAS_BIAS:
                acc0 += bias_tile[None, :]
                acc1 += bias_tile[None, :]
            if HAS_RESIDUAL:
                residual0 = tl.load(
                    residual_seq + tb * rows * out_features + out_offsets,
                    mask=m_mask[:, None] & n_mask[None, :], other=0.0,
                )
                residual1 = tl.load(
                    residual_seq + (tb + 1) * rows * out_features + out_offsets,
                    mask=m_mask[:, None] & n_mask[None, :], other=0.0,
                )
                acc0 += residual0
                acc1 += residual1

            if TAU_LE_ONE:
                v_before_spike = v + acc0
            else:
                v_before_spike = v + (acc0 - v) * tau_inv
            pred = v_before_spike >= v_threshold
            spike = pred.to(tl.float32)
            if SOFT_RESET:
                v = v_before_spike - spike * v_threshold
            else:
                v = tl.where(pred, v_reset, v_before_spike)
            tl.store(
                spike_seq + tb * rows * out_features + out_offsets,
                spike, mask=m_mask[:, None] & n_mask[None, :],
            )

            if TAU_LE_ONE:
                v_before_spike = v + acc1
            else:
                v_before_spike = v + (acc1 - v) * tau_inv
            pred = v_before_spike >= v_threshold
            spike = pred.to(tl.float32)
            if SOFT_RESET:
                v = v_before_spike - spike * v_threshold
            else:
                v = tl.where(pred, v_reset, v_before_spike)
            tl.store(
                spike_seq + (tb + 1) * rows * out_features + out_offsets,
                spike, mask=m_mask[:, None] & n_mask[None, :],
            )

        else:
            a = acc.reshape([2, 2, BLOCK_M, BLOCK_N]).permute(2, 3, 0, 1)
            split_lo0, split_lo1 = a.split()
            acc0, acc2 = split_lo0.split()
            acc1, acc3 = split_lo1.split()

            if HAS_BIAS:
                acc0 += bias_tile[None, :]
                acc1 += bias_tile[None, :]
                acc2 += bias_tile[None, :]
                acc3 += bias_tile[None, :]
            if HAS_RESIDUAL:
                residual0 = tl.load(
                    residual_seq + tb * rows * out_features + out_offsets,
                    mask=m_mask[:, None] & n_mask[None, :], other=0.0,
                )
                residual1 = tl.load(
                    residual_seq + (tb + 1) * rows * out_features + out_offsets,
                    mask=m_mask[:, None] & n_mask[None, :], other=0.0,
                )
                residual2 = tl.load(
                    residual_seq + (tb + 2) * rows * out_features + out_offsets,
                    mask=m_mask[:, None] & n_mask[None, :], other=0.0,
                )
                residual3 = tl.load(
                    residual_seq + (tb + 3) * rows * out_features + out_offsets,
                    mask=m_mask[:, None] & n_mask[None, :], other=0.0,
                )
                acc0 += residual0
                acc1 += residual1
                acc2 += residual2
                acc3 += residual3

            if TAU_LE_ONE:
                v_before_spike = v + acc0
            else:
                v_before_spike = v + (acc0 - v) * tau_inv
            pred = v_before_spike >= v_threshold
            spike = pred.to(tl.float32)
            if SOFT_RESET:
                v = v_before_spike - spike * v_threshold
            else:
                v = tl.where(pred, v_reset, v_before_spike)
            tl.store(
                spike_seq + tb * rows * out_features + out_offsets,
                spike, mask=m_mask[:, None] & n_mask[None, :],
            )

            if TAU_LE_ONE:
                v_before_spike = v + acc1
            else:
                v_before_spike = v + (acc1 - v) * tau_inv
            pred = v_before_spike >= v_threshold
            spike = pred.to(tl.float32)
            if SOFT_RESET:
                v = v_before_spike - spike * v_threshold
            else:
                v = tl.where(pred, v_reset, v_before_spike)
            tl.store(
                spike_seq + (tb + 1) * rows * out_features + out_offsets,
                spike, mask=m_mask[:, None] & n_mask[None, :],
            )

            if TAU_LE_ONE:
                v_before_spike = v + acc2
            else:
                v_before_spike = v + (acc2 - v) * tau_inv
            pred = v_before_spike >= v_threshold
            spike = pred.to(tl.float32)
            if SOFT_RESET:
                v = v_before_spike - spike * v_threshold
            else:
                v = tl.where(pred, v_reset, v_before_spike)
            tl.store(
                spike_seq + (tb + 2) * rows * out_features + out_offsets,
                spike, mask=m_mask[:, None] & n_mask[None, :],
            )

            if TAU_LE_ONE:
                v_before_spike = v + acc3
            else:
                v_before_spike = v + (acc3 - v) * tau_inv
            pred = v_before_spike >= v_threshold
            spike = pred.to(tl.float32)
            if SOFT_RESET:
                v = v_before_spike - spike * v_threshold
            else:
                v = tl.where(pred, v_reset, v_before_spike)
            tl.store(
                spike_seq + (tb + 3) * rows * out_features + out_offsets,
                spike, mask=m_mask[:, None] & n_mask[None, :],
            )
    tl.store(v_last + out_offsets, v, mask=m_mask[:, None] & n_mask[None, :])


@triton.autotune(
    configs=_linear_convstyle4_configs(),
    key=["rows", "in_features", "out_features", "T", "HAS_RESIDUAL", "HAS_BIAS", "AUTOTUNE_VERSION"],
    prune_configs_by={"early_config_prune": _prune_linear_convstyle4_configs},
    cache_results=True,
)
@triton.jit
def _temporal_batched_linear_lif_convstyle4_kernel(
    x_seq, residual_seq, weight, bias, v_init, spike_seq, v_last,
    rows: tl.constexpr, in_features: tl.constexpr, out_features: tl.constexpr,
    v_threshold, v_reset, tau_inv,
    T: tl.constexpr, HAS_BIAS: tl.constexpr, HAS_RESIDUAL: tl.constexpr,
    TAU_LE_ONE: tl.constexpr, SOFT_RESET: tl.constexpr, USE_TF32: tl.constexpr,
    V_INIT_IS_SCALAR: tl.constexpr,
    AUTOTUNE_VERSION: tl.constexpr,
    BTILE_T: tl.constexpr, REUSE_GROUPS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    ms = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    ns = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = ms < rows
    n_mask = ns < out_features
    out_offsets = ms[:, None] * out_features + ns[None, :]

    if V_INIT_IS_SCALAR:
        v = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    else:
        v = tl.load(v_init + out_offsets, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
    if HAS_BIAS:
        bias_tile = tl.load(bias + ns, mask=n_mask, other=0.0)

    acc0 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    acc1 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    acc2 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    acc3 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for k0 in range(0, in_features, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        k_mask = ks < in_features
        w = tl.load(
            weight + ns[None, :] * in_features + ks[:, None],
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )

        if BTILE_T == 1:
            x0 = tl.load(
                x_seq + ms[:, None] * in_features + ks[None, :],
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            x1 = tl.load(
                x_seq + rows * in_features + ms[:, None] * in_features + ks[None, :],
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            x2 = tl.load(
                x_seq + 2 * rows * in_features + ms[:, None] * in_features + ks[None, :],
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            x3 = tl.load(
                x_seq + 3 * rows * in_features + ms[:, None] * in_features + ks[None, :],
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            if USE_TF32:
                acc0 = tl.dot(x0, w, acc0, input_precision="tf32")
                acc1 = tl.dot(x1, w, acc1, input_precision="tf32")
                acc2 = tl.dot(x2, w, acc2, input_precision="tf32")
                acc3 = tl.dot(x3, w, acc3, input_precision="tf32")
            else:
                acc0 = tl.dot(x0, w, acc0)
                acc1 = tl.dot(x1, w, acc1)
                acc2 = tl.dot(x2, w, acc2)
                acc3 = tl.dot(x3, w, acc3)

        elif BTILE_T == 2:
            MT2: tl.constexpr = BLOCK_M * 2
            cat2 = tl.arange(0, MT2)
            lt2 = cat2 // BLOCK_M
            lm2 = cat2 % BLOCK_M
            cat_ms2 = pid_m * BLOCK_M + lm2
            cat_mask2 = cat_ms2 < rows
            x01 = tl.load(
                x_seq + lt2[:, None] * rows * in_features + cat_ms2[:, None] * in_features + ks[None, :],
                mask=cat_mask2[:, None] & k_mask[None, :],
                other=0.0,
            )
            x23 = tl.load(
                x_seq + (2 + lt2[:, None]) * rows * in_features + cat_ms2[:, None] * in_features + ks[None, :],
                mask=cat_mask2[:, None] & k_mask[None, :],
                other=0.0,
            )
            acc01 = tl.zeros((MT2, BLOCK_N), tl.float32)
            acc23 = tl.zeros((MT2, BLOCK_N), tl.float32)
            if USE_TF32:
                acc01 = tl.dot(x01, w, acc01, input_precision="tf32")
                acc23 = tl.dot(x23, w, acc23, input_precision="tf32")
            else:
                acc01 = tl.dot(x01, w, acc01)
                acc23 = tl.dot(x23, w, acc23)
            a0, a1 = acc01.reshape([2, BLOCK_M, BLOCK_N]).permute(1, 2, 0).split()
            a2, a3 = acc23.reshape([2, BLOCK_M, BLOCK_N]).permute(1, 2, 0).split()
            acc0 += a0
            acc1 += a1
            acc2 += a2
            acc3 += a3

        else:
            MT4: tl.constexpr = BLOCK_M * 4
            cat4 = tl.arange(0, MT4)
            lt4 = cat4 // BLOCK_M
            lm4 = cat4 % BLOCK_M
            cat_ms4 = pid_m * BLOCK_M + lm4
            cat_mask4 = cat_ms4 < rows
            x = tl.load(
                x_seq + lt4[:, None] * rows * in_features + cat_ms4[:, None] * in_features + ks[None, :],
                mask=cat_mask4[:, None] & k_mask[None, :],
                other=0.0,
            )
            acc = tl.zeros((MT4, BLOCK_N), tl.float32)
            if USE_TF32:
                acc = tl.dot(x, w, acc, input_precision="tf32")
            else:
                acc = tl.dot(x, w, acc)
            a = acc.reshape([2, 2, BLOCK_M, BLOCK_N]).permute(2, 3, 0, 1)
            split_lo0, split_lo1 = a.split()
            a0, a2 = split_lo0.split()
            a1, a3 = split_lo1.split()
            acc0 += a0
            acc1 += a1
            acc2 += a2
            acc3 += a3

    if HAS_BIAS:
        acc0 += bias_tile[None, :]
        acc1 += bias_tile[None, :]
        acc2 += bias_tile[None, :]
        acc3 += bias_tile[None, :]
    if HAS_RESIDUAL:
        residual0 = tl.load(
            residual_seq + out_offsets,
            mask=m_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        residual1 = tl.load(
            residual_seq + rows * out_features + out_offsets,
            mask=m_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        residual2 = tl.load(
            residual_seq + 2 * rows * out_features + out_offsets,
            mask=m_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        residual3 = tl.load(
            residual_seq + 3 * rows * out_features + out_offsets,
            mask=m_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        acc0 += residual0
        acc1 += residual1
        acc2 += residual2
        acc3 += residual3

    if TAU_LE_ONE:
        v_before_spike = v + acc0
    else:
        v_before_spike = v + (acc0 - v) * tau_inv
    pred = v_before_spike >= v_threshold
    spike = pred.to(tl.float32)
    if SOFT_RESET:
        v = v_before_spike - spike * v_threshold
    else:
        v = tl.where(pred, v_reset, v_before_spike)
    tl.store(spike_seq + out_offsets, spike, mask=m_mask[:, None] & n_mask[None, :])

    if TAU_LE_ONE:
        v_before_spike = v + acc1
    else:
        v_before_spike = v + (acc1 - v) * tau_inv
    pred = v_before_spike >= v_threshold
    spike = pred.to(tl.float32)
    if SOFT_RESET:
        v = v_before_spike - spike * v_threshold
    else:
        v = tl.where(pred, v_reset, v_before_spike)
    tl.store(spike_seq + rows * out_features + out_offsets, spike, mask=m_mask[:, None] & n_mask[None, :])

    if TAU_LE_ONE:
        v_before_spike = v + acc2
    else:
        v_before_spike = v + (acc2 - v) * tau_inv
    pred = v_before_spike >= v_threshold
    spike = pred.to(tl.float32)
    if SOFT_RESET:
        v = v_before_spike - spike * v_threshold
    else:
        v = tl.where(pred, v_reset, v_before_spike)
    tl.store(spike_seq + 2 * rows * out_features + out_offsets, spike, mask=m_mask[:, None] & n_mask[None, :])

    if TAU_LE_ONE:
        v_before_spike = v + acc3
    else:
        v_before_spike = v + (acc3 - v) * tau_inv
    pred = v_before_spike >= v_threshold
    spike = pred.to(tl.float32)
    if SOFT_RESET:
        v = v_before_spike - spike * v_threshold
    else:
        v = tl.where(pred, v_reset, v_before_spike)
    tl.store(spike_seq + 3 * rows * out_features + out_offsets, spike, mask=m_mask[:, None] & n_mask[None, :])
    tl.store(v_last + out_offsets, v, mask=m_mask[:, None] & n_mask[None, :])


@triton.autotune(
    configs=[triton.Config({"BLOCK": block}, num_warps=warps) for block, warps in ((128, 4), (256, 4), (512, 8), (1024, 8))],
    key=["elements", "T"],
)
@triton.jit
def _temporal_add_lif_kernel(
    lhs_seq, rhs_seq, v_init, spike_seq, v_last,
    elements: tl.constexpr, v_threshold, v_reset, tau_inv,
    T: tl.constexpr, TAU_LE_ONE: tl.constexpr, SOFT_RESET: tl.constexpr,
    V_INIT_IS_SCALAR: tl.constexpr, BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < elements
    if V_INIT_IS_SCALAR:
        v = tl.zeros((BLOCK,), tl.float32)
    else:
        v = tl.load(v_init + offsets, mask=mask, other=0.0)
    for t in tl.static_range(0, T):
        base = t * elements + offsets
        current = tl.load(lhs_seq + base, mask=mask, other=0.0)
        current += tl.load(rhs_seq + base, mask=mask, other=0.0)
        if TAU_LE_ONE:
            v_before_spike = v + current
        else:
            v_before_spike = v + (current - v) * tau_inv
        pred = v_before_spike >= v_threshold
        spike = pred.to(tl.float32)
        if SOFT_RESET:
            v = v_before_spike - spike * v_threshold
        else:
            v = tl.where(pred, v_before_spike * 0.0 + v_reset, v_before_spike)
        tl.store(spike_seq + base, spike, mask=mask)
    tl.store(v_last + offsets, v, mask=mask)


def run_temporal_batched_linear_lif(
    x_seq: torch.Tensor,
    weight: torch.Tensor,
    bias,
    v_init: torch.Tensor,
    v_threshold: float,
    v_reset: float,
    tau: float,
    *,
    residual_seq: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if x_seq.dim() < 4:
        raise RuntimeError(f"batched temporal Linear+LIF expects [T,...,Fin] with rank >= 4, got {x_seq.dim()}")
    if weight.dim() != 2 or int(x_seq.shape[-1]) != int(weight.shape[1]):
        raise RuntimeError("batched temporal Linear+LIF weight/input feature mismatch")
    x_seq = x_seq.contiguous()
    T = int(x_seq.shape[0])
    leading = tuple(x_seq.shape[1:-1])
    rows = x_seq.numel() // (T * int(x_seq.shape[-1]))
    out_features = int(weight.shape[0])
    out_shape = (T,) + leading + (out_features,)
    state_shape = leading + (out_features,)
    spikes = torch.empty(out_shape, device=x_seq.device, dtype=x_seq.dtype)
    v_last = torch.empty(state_shape, device=x_seq.device, dtype=x_seq.dtype)
    residual = weight
    if residual_seq is not None:
        if tuple(residual_seq.shape) != out_shape:
            raise RuntimeError(f"residual shape {tuple(residual_seq.shape)} does not match {out_shape}")
        residual = residual_seq.contiguous()
    has_bias = isinstance(bias, torch.Tensor) and bias.numel() > 0
    bias_arg = bias.contiguous() if has_bias else weight
    scalar_v = v_init.dim() == 0
    if not scalar_v and tuple(v_init.shape) != state_shape:
        raise RuntimeError(f"v_init shape {tuple(v_init.shape)} does not match {state_shape}")
    grid = lambda meta: (triton.cdiv(rows, meta["BLOCK_M"]), triton.cdiv(out_features, meta["BLOCK_N"]))
    _temporal_batched_linear_lif_kernel[grid](
        x_seq, residual, weight.contiguous(), bias_arg, v_init.contiguous(), spikes, v_last,
        rows=rows, in_features=int(x_seq.shape[-1]), out_features=out_features,
        v_threshold=float(v_threshold), v_reset=float(v_reset), tau_inv=1.0 / float(tau),
        T=T, HAS_BIAS=has_bias, HAS_RESIDUAL=residual_seq is not None,
        TAU_LE_ONE=float(tau) <= 1.0, SOFT_RESET=float(v_reset) < 0.0,
        USE_TF32=x_seq.dtype == torch.float32, V_INIT_IS_SCALAR=scalar_v,
        AUTOTUNE_VERSION=2,
    )
    return spikes, v_last


def run_temporal_batched_linear_lif_convstyle4(
    # Deprecated: T=4-only experimental prototype for the double-layer
    # (BTILE_T x REUSE_GROUPS) temporal schedule. Superseded by the codegen
    # framework in kernels/benchmark_batched_linear_lif_temporal_general.py /
    # kernels/generated_temporal_batched_linear_lif_kernel.py, which supports
    # arbitrary T and is wired into the runtime by default. Kept only for
    # microbenchmark comparison; do not extend this function further.
    x_seq: torch.Tensor,
    weight: torch.Tensor,
    bias,
    v_init: torch.Tensor,
    v_threshold: float,
    v_reset: float,
    tau: float,
    *,
    residual_seq: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Experimental Linear+LIF kernel that mirrors the conv temporal schedule for T=4.

    This entry point is intentionally separate from the default runtime path so
    microbenchmarks can compare the scheduling strategy without changing whole
    model execution.
    """
    if x_seq.dim() < 4:
        raise RuntimeError(f"batched temporal Linear+LIF expects [T,...,Fin] with rank >= 4, got {x_seq.dim()}")
    if weight.dim() != 2 or int(x_seq.shape[-1]) != int(weight.shape[1]):
        raise RuntimeError("batched temporal Linear+LIF weight/input feature mismatch")
    x_seq = x_seq.contiguous()
    T = int(x_seq.shape[0])
    if T != 4:
        raise RuntimeError(f"convstyle4 Linear+LIF expects T=4, got T={T}")
    leading = tuple(x_seq.shape[1:-1])
    rows = x_seq.numel() // (T * int(x_seq.shape[-1]))
    out_features = int(weight.shape[0])
    out_shape = (T,) + leading + (out_features,)
    state_shape = leading + (out_features,)
    spikes = torch.empty(out_shape, device=x_seq.device, dtype=x_seq.dtype)
    v_last = torch.empty(state_shape, device=x_seq.device, dtype=x_seq.dtype)
    residual = weight
    if residual_seq is not None:
        if tuple(residual_seq.shape) != out_shape:
            raise RuntimeError(f"residual shape {tuple(residual_seq.shape)} does not match {out_shape}")
        residual = residual_seq.contiguous()
    has_bias = isinstance(bias, torch.Tensor) and bias.numel() > 0
    bias_arg = bias.contiguous() if has_bias else weight
    scalar_v = v_init.dim() == 0
    if not scalar_v and tuple(v_init.shape) != state_shape:
        raise RuntimeError(f"v_init shape {tuple(v_init.shape)} does not match {state_shape}")
    grid = lambda meta: (triton.cdiv(rows, meta["BLOCK_M"]), triton.cdiv(out_features, meta["BLOCK_N"]))
    _temporal_batched_linear_lif_convstyle4_kernel[grid](
        x_seq, residual, weight.contiguous(), bias_arg, v_init.contiguous(), spikes, v_last,
        rows=rows, in_features=int(x_seq.shape[-1]), out_features=out_features,
        v_threshold=float(v_threshold), v_reset=float(v_reset), tau_inv=1.0 / float(tau),
        T=T, HAS_BIAS=has_bias, HAS_RESIDUAL=residual_seq is not None,
        TAU_LE_ONE=float(tau) <= 1.0, SOFT_RESET=float(v_reset) < 0.0,
        USE_TF32=x_seq.dtype == torch.float32, V_INIT_IS_SCALAR=scalar_v,
        AUTOTUNE_VERSION=1,
    )
    return spikes, v_last


def run_temporal_add_lif(
    lhs_seq: torch.Tensor,
    rhs_seq: torch.Tensor,
    v_init: torch.Tensor,
    v_threshold: float,
    v_reset: float,
    tau: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if tuple(lhs_seq.shape) != tuple(rhs_seq.shape) or lhs_seq.dim() < 2:
        raise RuntimeError("temporal Add+LIF inputs must have equal [T,...] shapes")
    lhs_seq = lhs_seq.contiguous()
    rhs_seq = rhs_seq.contiguous()
    T = int(lhs_seq.shape[0])
    state_shape = tuple(lhs_seq.shape[1:])
    elements = lhs_seq[0].numel()
    spikes = torch.empty_like(lhs_seq)
    v_last = torch.empty(state_shape, device=lhs_seq.device, dtype=lhs_seq.dtype)
    scalar_v = v_init.dim() == 0
    if not scalar_v and tuple(v_init.shape) != state_shape:
        raise RuntimeError(f"v_init shape {tuple(v_init.shape)} does not match {state_shape}")
    grid = lambda meta: (triton.cdiv(elements, meta["BLOCK"]),)
    _temporal_add_lif_kernel[grid](
        lhs_seq, rhs_seq, v_init.contiguous(), spikes, v_last,
        elements=elements, v_threshold=float(v_threshold), v_reset=float(v_reset), tau_inv=1.0 / float(tau),
        T=T, TAU_LE_ONE=float(tau) <= 1.0, SOFT_RESET=float(v_reset) < 0.0,
        V_INIT_IS_SCALAR=scalar_v,
    )
    return spikes, v_last
