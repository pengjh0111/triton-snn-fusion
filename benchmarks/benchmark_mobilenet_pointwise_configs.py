#!/usr/bin/env python3
"""Benchmark MobileNetV1 1x1 temporal Conv+LIF Triton configs."""

import argparse
import itertools
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.validate_chronos_baselines import ChronosMobileNetV1
from kernels.benchmark_conv_lif_temporal_general import (
    _pointwise_config_for_shape,
    get_autotune_best_config,
    run_fused_temporal_general,
    run_fused_temporal_general_autotuned,
    time_cuda,
    valid_temporal_schedules,
)


@dataclass(frozen=True)
class PointwiseShape:
    index: int
    in_channels: int
    out_channels: int
    height: int
    width: int
    count: int = 1


def _conv_out(size: int, kernel: int, stride: int, padding: int, dilation: int = 1) -> int:
    return (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


def collect_mobilenet_pointwise_shapes(height: int, width: int, channels: int) -> List[PointwiseShape]:
    model = ChronosMobileNetV1(channels=channels, lif_impl="chronos").eval()
    shapes: List[PointwiseShape] = []
    cur_h, cur_w = height, width
    for idx, module in enumerate(model.layer):
        if not hasattr(module, "weight") or module.weight.dim() != 4:
            continue
        weight = module.weight
        kh, kw = weight.shape[2], weight.shape[3]
        stride_h, stride_w = module.stride
        pad_h, pad_w = module.padding
        groups = int(module.groups)
        if kh == 1 and kw == 1 and groups == 1:
            shapes.append(
                PointwiseShape(
                    index=idx,
                    in_channels=int(module.in_channels),
                    out_channels=int(module.out_channels),
                    height=cur_h,
                    width=cur_w,
                )
            )
        cur_h = _conv_out(cur_h, kh, stride_h, pad_h)
        cur_w = _conv_out(cur_w, kw, stride_w, pad_w)

    merged: Dict[Tuple[int, int, int, int], PointwiseShape] = {}
    for shape in shapes:
        key = (shape.in_channels, shape.out_channels, shape.height, shape.width)
        if key not in merged:
            merged[key] = shape
        else:
            prev = merged[key]
            merged[key] = PointwiseShape(
                prev.index,
                prev.in_channels,
                prev.out_channels,
                prev.height,
                prev.width,
                prev.count + 1,
            )
    return list(merged.values())


def _dtype(name: str) -> torch.dtype:
    return torch.float16 if name == "fp16" else torch.float32


def _candidate_configs() -> Iterable[Dict[str, int]]:
    for block_m, block_oc, block_k, warps in itertools.product(
        (8, 16, 32),
        (32, 64, 128),
        (16, 32, 64),
        (2, 4),
    ):
        yield {
            "BLOCK_M": block_m,
            "BLOCK_OC": block_oc,
            "BLOCK_K": block_k,
            "num_warps": warps,
            "num_stages": 2,
        }


def _make_inputs(shape: PointwiseShape, timesteps: int, batch: int, dtype: torch.dtype, seed: int):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed + shape.in_channels + shape.out_channels + shape.height)
    x_seq = (torch.randn(
        timesteps,
        batch,
        shape.in_channels,
        shape.height,
        shape.width,
        device="cuda",
        dtype=dtype,
        generator=generator,
    ) * 0.02).contiguous()
    weight = (torch.randn(
        shape.out_channels,
        shape.in_channels,
        1,
        1,
        device="cuda",
        dtype=dtype,
        generator=generator,
    ) * 0.02).contiguous()
    bias = (torch.randn(shape.out_channels, device="cuda", dtype=dtype, generator=generator) * 0.01).contiguous()
    return x_seq, weight, bias


def _torch_reference(x_seq: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    timesteps, batch, _, height, width = x_seq.shape
    membrane = torch.zeros((batch, weight.shape[0], height, width), device=x_seq.device, dtype=x_seq.dtype)
    spikes = []
    for step in range(timesteps):
        y = F.conv2d(x_seq[step], weight, bias)
        v_new = membrane + (y - membrane) * 0.5
        spike = (v_new >= 1.0).to(v_new.dtype)
        membrane = torch.where(spike > 0.5, torch.zeros_like(v_new), v_new)
        spikes.append(spike)
    return torch.stack(spikes, dim=0), membrane


def _correctness(spikes, membrane, ref_spikes, ref_membrane):
    spike_diff = (spikes - ref_spikes).abs().to(torch.float32)
    v_diff = (membrane - ref_membrane).abs().to(torch.float32)
    atol = 1e-2 if spikes.dtype == torch.float16 else 2e-3
    rtol = 1e-2 if spikes.dtype == torch.float16 else 2e-3
    return {
        "spike_max": float(spike_diff.max().item()),
        "spike_mean": float(spike_diff.mean().item()),
        "v_max": float(v_diff.max().item()),
        "v_mean": float(v_diff.mean().item()),
        "allclose": bool(
            torch.allclose(spikes, ref_spikes, atol=atol, rtol=rtol)
            and torch.allclose(membrane, ref_membrane, atol=atol, rtol=rtol)
        ),
    }


def _acc_elems(config: Dict[str, int], btile_t: int, reuse_groups: int) -> int:
    return int(btile_t) * int(reuse_groups) * int(config["BLOCK_M"]) * int(config["BLOCK_OC"])


def benchmark_shape(args, shape: PointwiseShape):
    dtype = _dtype(args.dtype)
    x_seq, weight, bias = _make_inputs(shape, args.T, args.batch_size, dtype, args.seed)
    ref_spikes, ref_membrane = _torch_reference(x_seq, weight, bias)
    old_config = {"BLOCK_M": 32, "BLOCK_OC": 64, "BLOCK_K": 32, "num_warps": 4, "num_stages": 2}
    selector_config = _pointwise_config_for_shape(
        shape.in_channels,
        shape.out_channels,
        shape.height,
        shape.width,
    )
    configs = [("old_fixed", old_config, args.T, 1), ("selector_fixed", selector_config, args.T, 1)]
    for limit in args.acc_limits:
        for btile_t, reuse_groups in valid_temporal_schedules(args.T):
            if btile_t not in (1, 2, 4):
                continue
            if _acc_elems(selector_config, btile_t, reuse_groups) <= limit:
                configs.append((f"selector_limit_{limit}", selector_config, btile_t, reuse_groups))
    if args.sweep:
        for idx, cfg in enumerate(_candidate_configs()):
            for btile_t, reuse_groups in valid_temporal_schedules(args.T):
                if btile_t not in (1, 2, 4):
                    continue
                for limit in args.acc_limits:
                    if _acc_elems(cfg, btile_t, reuse_groups) <= limit:
                        configs.append((f"sweep_limit_{limit}_{idx}", cfg, btile_t, reuse_groups))

    rows = []
    seen = set()
    for label, config, btile_t, reuse_groups in configs:
        key = (label, tuple(sorted(config.items())), btile_t, reuse_groups)
        if key in seen:
            continue
        seen.add(key)
        try:
            spikes, membrane = run_fused_temporal_general(
                x_seq,
                weight,
                bias,
                temporal_batch_size=btile_t,
                reuse_groups=reuse_groups,
                spatial_config=config,
                kernel_key="k1_s1_p0",
            )
            corr = _correctness(spikes, membrane, ref_spikes, ref_membrane)
            torch.cuda.synchronize()
            ms = time_cuda(
                lambda config=config, btile_t=btile_t, reuse_groups=reuse_groups: run_fused_temporal_general(
                    x_seq,
                    weight,
                    bias,
                    temporal_batch_size=btile_t,
                    reuse_groups=reuse_groups,
                    spatial_config=config,
                    kernel_key="k1_s1_p0",
                ),
                warmup=args.warmup,
                rep=args.repeat,
            )
            rows.append((ms, label, config, btile_t, reuse_groups, _acc_elems(config, btile_t, reuse_groups), corr))
        except Exception as exc:
            if args.verbose:
                print(f"[POINTWISE_SKIP] shape={shape} label={label} reason={type(exc).__name__}: {exc}")

    rows.sort(key=lambda row: row[0])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--dtype", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--acc-limits", type=int, nargs="+", default=[8192, 16384])
    parser.add_argument("--check-autotune", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    shapes = collect_mobilenet_pointwise_shapes(args.height, args.width, args.channels)
    old_weighted = 0.0
    selector_weighted = 0.0
    print(
        f"[POINTWISE_CONFIG_BENCH] T={args.T} batch={args.batch_size} "
        f"input={args.height}x{args.width} dtype={args.dtype} shapes={len(shapes)} acc_limits={args.acc_limits}"
    )
    for shape in shapes:
        print(
            f"[POINTWISE_SHAPE] layer={shape.index} T={args.T} N={args.batch_size} "
            f"Cin={shape.in_channels} Cout={shape.out_channels} H={shape.height} W={shape.width} count={shape.count}"
        )
        rows = benchmark_shape(args, shape)
        by_label = {}
        for ms, label, config, btile_t, reuse_groups, acc_elems, corr in rows:
            by_label.setdefault(label, (ms, config, btile_t, reuse_groups, acc_elems, corr))
        if "old_fixed" in by_label and "selector_fixed" in by_label:
            old_ms, old_config, _, _, _, old_corr = by_label["old_fixed"]
            selector_ms, selector_config, _, _, _, selector_corr = by_label["selector_fixed"]
            old_weighted += old_ms * shape.count
            selector_weighted += selector_ms * shape.count
            print(
                f"{shape.in_channels}->{shape.out_channels} {shape.height}x{shape.width} x{shape.count}: "
                f"old_fixed={old_ms:.4f}ms selector_fixed={selector_ms:.4f}ms "
                f"speedup={old_ms / selector_ms:.3f} cfg={selector_config} "
                f"corr_allclose={selector_corr['allclose']} v_max={selector_corr['v_max']:.3e}"
            )
        for limit in args.acc_limits:
            limit_rows = [row for row in rows if row[1] == f"selector_limit_{limit}"]
            if limit_rows:
                best_ms, best_label, best_config, btile_t, reuse_groups, acc_elems, corr = min(limit_rows, key=lambda row: row[0])
                print(
                    f"  limit={limit:<5} best={best_ms:.4f}ms BTILE_T={btile_t} REUSE_GROUPS={reuse_groups} "
                    f"acc_elems={acc_elems} cfg={best_config} allclose={corr['allclose']} v_max={corr['v_max']:.3e}"
                )
        if args.sweep and rows:
            best_ms, best_label, best_config, btile_t, reuse_groups, acc_elems, corr = rows[0]
            print(
                f"  sweep_best={best_ms:.4f}ms label={best_label} BTILE_T={btile_t} "
                f"REUSE_GROUPS={reuse_groups} acc_elems={acc_elems} cfg={best_config} "
                f"allclose={corr['allclose']} v_max={corr['v_max']:.3e}"
            )
        if args.check_autotune:
            dtype = _dtype(args.dtype)
            x_seq, weight, bias = _make_inputs(shape, args.T, args.batch_size, dtype, args.seed)
            for limit in args.acc_limits:
                os.environ["CHRONOS_POINTWISE_ACC_ELEMS_LIMIT"] = str(limit)
                run_fused_temporal_general_autotuned(x_seq, weight, bias, kernel_key="k1_s1_p0")
                torch.cuda.synchronize()
                print(f"  autotune_limit={limit} best_config={get_autotune_best_config('k1_s1_p0')}")

    if selector_weighted > 0:
        print(
            f"[POINTWISE_CONFIG_SUMMARY] old_weighted={old_weighted:.4f}ms "
            f"selector_weighted={selector_weighted:.4f}ms "
            f"speedup={old_weighted / selector_weighted:.3f}x"
        )


if __name__ == "__main__":
    main()
