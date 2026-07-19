"""Step 0 (Tier-1 lowering spec): quantitative per-layer table for MobileNetV2.

For every LIF-fused conv layer (stem, expand-pw, dwconv, final 1280-pw --
i.e. every layer that goes through the fused_temporal_general kernel family
targeted by Tier 1; the LIF-less project-pw layers are NOT in this table,
they already run as plain conv2d/cudnn, confirmed via
dev_tests/profile_mobilenet_categories.py's "unfused_conv_bn" category),
this script:

  1. Builds the real x_seq/weight tensors for that layer's exact shape.
  2. Calls the real autotuned kernel entry point once, forcing Triton's
     autotuner to run its real (already-pruned by round 4's RG-K/min-CTA
     rules) search and pick a real winning config.
  3. Reads that winning config back and computes CTA count with the exact
     formula already used by the min-CTA pruning rule
     (_gemm_cta_count_fn / _dwconv_cta_count_fn), so the numbers here are
     guaranteed consistent with what the pruning rule itself sees.
  4. Reports waves = CTA_count / SM_count.

Channel/resolution progression below is derived directly from
KairosMobileNetV2 / KairosSpikingInvertedResidual in
benchmarks/validate_kairos_baselines.py (channels=64 default, matching
every benchmark run in this investigation).
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

import kernels.benchmark_conv_lif_temporal_general as K

T_STEPS = 16
SM_COUNT = torch.cuda.get_device_properties(0).multi_processor_count

# (name, kind, in_ch, out_ch_or_hidden, H_in, stride)
# kind: "general" (stem, k3s2p1) | "pw" (k1s1p0) | "dw" (depthwise k3, stride given)
LAYERS = [
    ("stem", "general", 3, 64, 224, 2),
]

_block_cfg = [
    # (in_ch, out_ch, stride, expand_ratio, H_in)
    (64, 32, 1, 1, 112),    # B1
    (32, 48, 2, 6, 112),    # B2
    (48, 48, 1, 6, 56),     # B3
    (48, 64, 2, 6, 56),     # B4
    (64, 64, 1, 6, 28),     # B5
    (64, 64, 1, 6, 28),     # B6
    (64, 128, 2, 6, 28),    # B7
    (128, 128, 1, 6, 14),   # B8
    (128, 128, 1, 6, 14),   # B9
    (128, 128, 1, 6, 14),   # B10
    (128, 192, 1, 6, 14),   # B11
    (192, 192, 1, 6, 14),   # B12
    (192, 192, 1, 6, 14),   # B13
    (192, 320, 2, 6, 14),   # B14
    (320, 320, 1, 6, 7),    # B15
    (320, 320, 1, 6, 7),    # B16
    (320, 640, 1, 6, 7),    # B17
]

for i, (in_ch, out_ch, stride, ratio, h_in) in enumerate(_block_cfg, start=1):
    hidden = in_ch * ratio
    if ratio != 1:
        LAYERS.append((f"B{i}_expand", "pw", in_ch, hidden, h_in, 1))
    LAYERS.append((f"B{i}_dwconv", "dw", hidden, hidden, h_in, stride))

LAYERS.append(("final_pw", "pw", 640, 2560, 7, 1))


def _conv_out_hw(h, w, k, s, p):
    return (h + 2 * p - k) // s + 1, (w + 2 * p - k) // s + 1


def run_one(name, kind, in_ch, out_ch, h_in, stride, batch, dtype=torch.float32):
    device = "cuda"
    if kind == "general":
        kernel_key = "k3_s2_p1"
        weight = torch.randn(out_ch, in_ch, 3, 3, device=device, dtype=dtype)
    elif kind == "pw":
        kernel_key = "k1_s1_p0"
        weight = torch.randn(out_ch, in_ch, 1, 1, device=device, dtype=dtype)
    elif kind == "dw":
        kernel_key = "depthwise_k3_s1_p1" if stride == 1 else "depthwise_k3_s2_p1"
        weight = torch.randn(out_ch, 1, 3, 3, device=device, dtype=dtype)
    else:
        raise ValueError(kind)

    x_seq = torch.randn(T_STEPS, batch, in_ch, h_in, h_in, device=device, dtype=dtype)
    bias = torch.zeros(out_ch, device=device, dtype=dtype)

    K.run_fused_temporal_general_autotuned(x_seq, weight, bias, kernel_key=kernel_key)
    torch.cuda.synchronize()

    kernel = K._autotuned_kernels[kernel_key]
    best_config = getattr(kernel, "best_config", None)

    variant = K.KERNEL_VARIANTS[kernel_key]
    oh, ow = _conv_out_hw(h_in, h_in, variant["kernel"], variant["stride"], variant["pad"])

    if kind == "dw":
        cta_fn = K._dwconv_cta_count_fn(
            {"num_batches": batch, "out_width": ow, "out_height": oh, "out_channels": out_ch}
        )
    else:
        cta_fn = K._gemm_cta_count_fn(
            {"num_batches": batch, "out_height": oh, "out_width": ow, "out_channels": out_ch}
        )

    if best_config is None:
        return dict(name=name, kind=kind, in_ch=in_ch, out_ch=out_ch, h_in=h_in, oh=oh, ow=ow,
                    stride=stride, batch=batch, m=batch * oh * ow, cta=None, waves=None, config="UNAVAILABLE")

    cta = cta_fn(best_config)
    waves = cta / SM_COUNT
    cfg_str = ",".join(f"{k}={v}" for k, v in sorted(best_config.kwargs.items()))
    cfg_str += f",warps={best_config.num_warps}"
    return dict(name=name, kind=kind, in_ch=in_ch, out_ch=out_ch, h_in=h_in, oh=oh, ow=ow,
                stride=stride, batch=batch, m=batch * oh * ow, cta=cta, waves=waves, config=cfg_str)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, nargs="+", default=[4, 16])
    args = parser.parse_args()

    print(f"SM_COUNT={SM_COUNT}", flush=True)
    for batch in args.batches:
        print(f"\n{'='*30} batch={batch} {'='*30}", flush=True)
        print(f"{'layer':<14}{'kind':<6}{'in':>6}{'out':>6}{'H':>5}{'OH':>5}{'OW':>5}{'M':>8}{'CTA':>8}{'waves':>8}  config")
        rows = []
        for name, kind, in_ch, out_ch, h_in, stride in LAYERS:
            r = run_one(name, kind, in_ch, out_ch, h_in, stride, batch)
            rows.append(r)
            waves_str = f"{r['waves']:.3f}" if r["waves"] is not None else "N/A"
            cta_str = str(r["cta"]) if r["cta"] is not None else "N/A"
            print(f"{r['name']:<14}{r['kind']:<6}{r['in_ch']:>6}{r['out_ch']:>6}{r['h_in']:>5}{r['oh']:>5}{r['ow']:>5}{r['m']:>8}{cta_str:>8}{waves_str:>8}  {r['config']}", flush=True)

        n_total = len(rows)
        n_lt2 = sum(1 for r in rows if r["waves"] is not None and r["waves"] < 2.0)
        n_lt1 = sum(1 for r in rows if r["waves"] is not None and r["waves"] < 1.0)
        print(f"\nlayers with waves<2: {n_lt2}/{n_total}   waves<1: {n_lt1}/{n_total}")


if __name__ == "__main__":
    main()
