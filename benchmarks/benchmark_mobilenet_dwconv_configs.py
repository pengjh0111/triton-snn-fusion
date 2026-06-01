import argparse
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.validate_chronos_baselines import ChronosMobileNetV1
from kernels.benchmark_conv_lif_temporal_general import run_fused_temporal_general


@dataclass(frozen=True)
class DWShape:
    name: str
    layer_index: int
    kernel_key: str
    T: int
    batch: int
    channels: int
    height: int
    width: int
    out_height: int
    out_width: int
    stride: int
    padding: int
    occurrences: int = 1


def _conv_out(size: int, kernel: int, stride: int, padding: int, dilation: int = 1) -> int:
    return (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


def collect_mobilenetv1_depthwise_shapes(
    *,
    model_channels: int,
    T: int,
    batch_size: int,
    height: int,
    width: int,
    lif_impl: str,
    unique: bool,
) -> List[DWShape]:
    model = ChronosMobileNetV1(channels=model_channels, step_mode="s", lif_impl=lif_impl)
    h, w = int(height), int(width)
    shapes: List[DWShape] = []
    for idx, module in enumerate(model.layer):
        if not all(hasattr(module, attr) for attr in ("in_channels", "out_channels", "kernel_size", "stride", "padding", "groups")):
            continue
        kh, kw = tuple(module.kernel_size)
        sh, sw = tuple(module.stride)
        ph, pw = tuple(module.padding)
        out_h = _conv_out(h, kh, sh, ph)
        out_w = _conv_out(w, kw, sw, pw)
        is_depthwise = (
            int(module.groups) == int(module.in_channels)
            and int(module.out_channels) == int(module.in_channels)
            and kh == 3
            and kw == 3
            and ph == 1
            and pw == 1
            and sh in (1, 2)
            and sw == sh
        )
        if is_depthwise:
            shapes.append(
                DWShape(
                    name=f"layer_{idx}",
                    layer_index=idx,
                    kernel_key=f"depthwise_k3_s{sh}_p1",
                    T=T,
                    batch=batch_size,
                    channels=int(module.in_channels),
                    height=h,
                    width=w,
                    out_height=out_h,
                    out_width=out_w,
                    stride=sh,
                    padding=ph,
                )
            )
        h, w = out_h, out_w
    if not unique:
        return shapes
    by_key: Dict[Tuple[int, int, int, int, int, str], DWShape] = {}
    counts: Dict[Tuple[int, int, int, int, int, str], int] = {}
    for shape in shapes:
        key = (shape.channels, shape.height, shape.width, shape.stride, shape.batch, shape.kernel_key)
        counts[key] = counts.get(key, 0) + 1
        if key not in by_key:
            by_key[key] = shape
    merged = []
    for key, shape in by_key.items():
        merged.append(
            DWShape(
                **{
                    **asdict(shape),
                    "name": f"{shape.name}_x{counts[key]}",
                    "occurrences": counts[key],
                }
            )
        )
    return merged


def iter_requested_configs() -> Iterable[Dict[str, int]]:
    for block_h in (2, 4, 8, 16):
        for block_w in (8, 16, 32, 64):
            for block_c in (1, 2, 4, 8, 16):
                for pixels_per_thread in (1, 2, 4):
                    for num_warps in (1, 2, 4):
                        for num_stages in (3, 4, 5):
                            yield {
                                "BLOCK_H": block_h,
                                "BLOCK_W": block_w,
                                "BLOCK_C": block_c,
                                "PIXELS_PER_THREAD": pixels_per_thread,
                                "num_warps": num_warps,
                                "num_stages": num_stages,
                            }


def _config_score(shape: DWShape, cfg: Dict[str, int]) -> Tuple[int, int, int, int]:
    # Prefer configs that are plausible for the shape near the front when a
    # candidate cap is used; full sweep still traverses every requested config.
    width_tile = cfg["BLOCK_W"] * cfg["PIXELS_PER_THREAD"]
    spatial = cfg["BLOCK_H"] * width_tile
    channel_gap = abs(cfg["BLOCK_C"] - min(16, max(1, 1 << int(math.log2(max(1, min(shape.channels, 16)))))))
    width_gap = abs(width_tile - min(64, max(8, shape.out_width)))
    height_gap = abs(cfg["BLOCK_H"] - min(16, max(2, shape.out_height)))
    return (channel_gap + width_gap + height_gap, abs(spatial - 128), cfg["num_warps"], cfg["num_stages"])


def make_configs(shape: DWShape, limit: Optional[int]) -> List[Dict[str, int]]:
    configs = sorted(iter_requested_configs(), key=lambda cfg: _config_score(shape, cfg))
    if limit is not None:
        configs = configs[:limit]
    return configs


def _sync():
    torch.cuda.synchronize()


def _time_cuda(fn, warmup: int, repeat: int) -> Dict[str, float]:
    for _ in range(warmup):
        fn()
    _sync()
    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        _sync()
        times.append(float(start.elapsed_time(end)))
    ordered = sorted(times)
    return {
        "mean_ms": float(statistics.mean(times)),
        "min_ms": float(min(times)),
        "p50_ms": float(ordered[len(ordered) // 2]),
        "p90_ms": float(ordered[int(0.9 * (len(ordered) - 1))]),
    }


def _make_inputs(shape: DWShape, dtype: torch.dtype, seed: int):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    x_seq = torch.randn(
        shape.T,
        shape.batch,
        shape.channels,
        shape.height,
        shape.width,
        device="cuda",
        dtype=dtype,
        generator=gen,
    ).contiguous()
    weight = torch.randn(shape.channels, 1, 3, 3, device="cuda", dtype=dtype, generator=gen).contiguous()
    bias = torch.randn(shape.channels, device="cuda", dtype=dtype, generator=gen).contiguous()
    v_init = torch.tensor(0.0, device="cuda", dtype=dtype)
    return x_seq, weight, bias, v_init


def benchmark_shape(shape: DWShape, args) -> Dict:
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    x_seq, weight, bias, v_init = _make_inputs(shape, dtype, seed=args.seed + shape.layer_index)
    configs = make_configs(shape, None if args.max_configs <= 0 else args.max_configs)
    rows = []
    errors = []
    for cfg_idx, cfg in enumerate(configs):
        def run():
            return run_fused_temporal_general(
                x_seq,
                weight,
                bias,
                temporal_batch_size=args.btile_t,
                reuse_groups=args.reuse_groups,
                spatial_config=cfg,
                kernel_key=shape.kernel_key,
                v_init=v_init,
            )

        try:
            run()
            _sync()
            timing = _time_cuda(run, args.warmup, args.repeat)
            rows.append({"config": cfg, **timing})
        except Exception as exc:
            errors.append({"config": cfg, "error": str(exc).splitlines()[0]})
        if args.progress and (cfg_idx + 1) % args.progress == 0:
            print(
                f"[DWCONV_SWEEP] {shape.name} {cfg_idx + 1}/{len(configs)} "
                f"tested best={min((r['mean_ms'] for r in rows), default=float('nan')):.4f} ms",
                flush=True,
            )
    rows.sort(key=lambda row: row["mean_ms"])
    return {
        "shape": asdict(shape),
        "kernel_key": shape.kernel_key,
        "tested_configs": len(rows),
        "failed_configs": len(errors),
        "best": rows[0] if rows else None,
        "top": rows[: args.topk],
        "errors": errors[: args.keep_errors],
    }


def main():
    parser = argparse.ArgumentParser(description="Sweep MobileNetV1 depthwise temporal ConvLIF Triton configs.")
    parser.add_argument("--model-channels", type=int, default=64)
    parser.add_argument("--lif-impl", default="chronos")
    parser.add_argument("--T", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--dtype", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--btile-t", type=int, default=1)
    parser.add_argument("--reuse-groups", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--max-configs", type=int, default=0, help="0 means full requested sweep.")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--keep-errors", type=int, default=8)
    parser.add_argument("--unique", action="store_true")
    parser.add_argument("--layer-index", type=int, action="append", default=[])
    parser.add_argument("--kernel-key", choices=("depthwise_k3_s1_p1", "depthwise_k3_s2_p1"), default="")
    parser.add_argument("--progress", type=int, default=0)
    parser.add_argument("--out", default="")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    shapes = collect_mobilenetv1_depthwise_shapes(
        model_channels=args.model_channels,
        T=args.T,
        batch_size=args.batch_size,
        height=args.height,
        width=args.width,
        lif_impl=args.lif_impl,
        unique=args.unique,
    )
    if args.layer_index:
        wanted = set(args.layer_index)
        shapes = [shape for shape in shapes if shape.layer_index in wanted]
    if args.kernel_key:
        shapes = [shape for shape in shapes if shape.kernel_key == args.kernel_key]
    print("[DWCONV_SHAPES]")
    for shape in shapes:
        print(
            f"{shape.name}: key={shape.kernel_key} T={shape.T} N={shape.batch} "
            f"C={shape.channels} HxW={shape.height}x{shape.width} "
            f"out={shape.out_height}x{shape.out_width} occurrences={shape.occurrences}",
            flush=True,
        )
    results = [benchmark_shape(shape, args) for shape in shapes]
    print("| layer | key | T,N,C,H,W | out | occ | tested | best_ms | best_config |")
    print("|---|---|---|---|---:|---:|---:|---|")
    for result in results:
        shape = result["shape"]
        best = result["best"] or {}
        print(
            f"| {shape['name']} | {result['kernel_key']} | "
            f"{shape['T']},{shape['batch']},{shape['channels']},{shape['height']},{shape['width']} | "
            f"{shape['out_height']}x{shape['out_width']} | {shape['occurrences']} | "
            f"{result['tested_configs']} | {best.get('mean_ms', float('nan')):.4f} | {best.get('config')} |"
        )
    payload = {
        "args": vars(args),
        "config_space": {
            "BLOCK_H": [2, 4, 8, 16],
            "BLOCK_W": [8, 16, 32, 64],
            "BLOCK_C": [1, 2, 4, 8, 16],
            "PIXELS_PER_THREAD": [1, 2, 4],
            "num_warps": [1, 2, 4],
            "num_stages": [3, 4, 5],
        },
        "results": results,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
