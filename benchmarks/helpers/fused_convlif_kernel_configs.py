import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

from runtime.snn_custom_ops import lif_forward_state_torch
from runtime.triton_convlif_backend import (
    check_triton_support,
    classify_conv_lif_config,
    run_triton_fused_conv_lif_state,
    run_triton_fused_temporal_conv_add_lif_state,
    run_triton_fused_temporal_conv_lif_state,
    run_triton_fused_temporal_pointwise_conv_lif_state,
)


CASES = [
    {"name": "mobilenet_pointwise_1x1_s1", "N": 2, "Cin": 64, "Cout": 128, "H": 32, "W": 32, "K": 1, "stride": 1, "padding": 0},
    {"name": "alex_stem_11x11_s4", "N": 2, "Cin": 3, "Cout": 64, "H": 224, "W": 224, "K": 11, "stride": 4, "padding": 2},
    {"name": "zf_stem_7x7_s2", "N": 2, "Cin": 3, "Cout": 64, "H": 224, "W": 224, "K": 7, "stride": 2, "padding": 3},
    {"name": "alex_zf_5x5_s1", "N": 2, "Cin": 64, "Cout": 192, "H": 28, "W": 28, "K": 5, "stride": 1, "padding": 2},
    {"name": "small_5x5_s1", "N": 2, "Cin": 64, "Cout": 64, "H": 56, "W": 56, "K": 5, "stride": 1, "padding": 2},
    {"name": "res2_3x3_s1", "N": 2, "Cin": 64, "Cout": 64, "H": 56, "W": 56, "K": 3, "stride": 1, "padding": 1},
    {"name": "res3_3x3_s2", "N": 2, "Cin": 64, "Cout": 128, "H": 56, "W": 56, "K": 3, "stride": 2, "padding": 1},
]


def _pair(x):
    return [x, x]


def _dtype_from_arg(dtype_name: str):
    if dtype_name == "fp32":
        return torch.float32
    if dtype_name == "fp16":
        return torch.float16
    raise ValueError(f"unsupported dtype: {dtype_name}")


def make_case_tensors(case: Dict, T: int, device: str, dtype=torch.float32):
    # Keep magnitudes small so the legacy TF32 3x3/s1 path also satisfies a tight
    # allclose check while still exercising the complete conv/LIF dataflow.
    scale = 0.005
    x_seq = torch.randn(
        T,
        case["N"],
        case["Cin"],
        case["H"],
        case["W"],
        device=device,
        dtype=dtype,
    ) * scale
    weight = torch.randn(
        case["Cout"],
        case["Cin"],
        case["K"],
        case["K"],
        device=device,
        dtype=dtype,
    ) * scale
    bias = torch.randn(case["Cout"], device=device, dtype=dtype) * scale
    v0 = torch.tensor(0.0, device=device, dtype=dtype)
    with torch.no_grad():
        out_h = (case["H"] + 2 * case["padding"] - (case["K"] - 1) - 1) // case["stride"] + 1
        out_w = (case["W"] + 2 * case["padding"] - (case["K"] - 1) - 1) // case["stride"] + 1
    residual_seq = torch.randn(
        T,
        case["N"],
        case["Cout"],
        out_h,
        out_w,
        device=device,
        dtype=dtype,
    ) * scale
    return x_seq.contiguous(), weight.contiguous(), bias.contiguous(), v0, residual_seq.contiguous()


def torch_temporal_ref(x_seq, weight, bias, v_init, stride, padding, residual_seq=None):
    v = v_init
    spikes = []
    for step in range(x_seq.shape[0]):
        conv = F.conv2d(x_seq[step], weight, bias, stride, padding, (1, 1), 1)
        if residual_seq is not None:
            conv = conv + residual_seq[step]
        spike, v = lif_forward_state_torch(conv, v, 1.0, 0.0, 2.0, False)
        spikes.append(spike)
    return torch.stack(spikes, dim=0), v


def run_case(case: Dict, T: int, device: str, dtype_name: str, rtol: float, atol: float, use_autotune: bool, residual: bool) -> Dict:
    stride = _pair(case["stride"])
    padding = _pair(case["padding"])
    dtype = _dtype_from_arg(dtype_name)
    x_seq, weight, bias, v0, residual_seq = make_case_tensors(case, T, device, dtype=dtype)
    residual_arg = residual_seq if residual else None
    ref_spikes, ref_v = torch_temporal_ref(x_seq, weight, bias, v0, stride, padding, residual_arg)

    support_reasons = check_triton_support(
        x_seq[0],
        weight,
        bias,
        v0,
        stride,
        padding,
        [1, 1],
        1,
        1.0,
        0.0,
        2.0,
        False,
    )
    kernel_key = classify_conv_lif_config(weight, stride, padding, [1, 1], 1)
    backend_hit = False
    runtime_kernel_key = "unsupported"
    error = ""
    try:
        if residual:
            result = run_triton_fused_temporal_conv_add_lif_state(
                [x_seq[i] for i in range(T)],
                [residual_seq[i] for i in range(T)],
                weight,
                bias,
                v0,
                stride,
                padding,
                [1, 1],
                1,
                1.0,
                0.0,
                2.0,
                False,
                use_autotune=use_autotune,
            )
            got_spikes = result.spikes
            got_v = result.v_next
        elif T == 1:
            result = run_triton_fused_conv_lif_state(
                x_seq[0],
                weight,
                bias,
                v0,
                stride,
                padding,
                [1, 1],
                1,
                1.0,
                0.0,
                2.0,
                False,
                use_autotune=use_autotune,
            )
            got_spikes = result.spikes.unsqueeze(0)
            got_v = result.v_next
        else:
            runner = (
                run_triton_fused_temporal_pointwise_conv_lif_state
                if kernel_key == "k1_s1_p0"
                else run_triton_fused_temporal_conv_lif_state
            )
            result = runner(
                [x_seq[i] for i in range(T)],
                weight,
                bias,
                v0,
                stride,
                padding,
                [1, 1],
                1,
                1.0,
                0.0,
                2.0,
                False,
                use_autotune=use_autotune,
            )
            got_spikes = result.spikes
            got_v = result.v_next
        backend_hit = result.used_triton
        runtime_kernel_key = result.kernel_key
    except Exception as exc:
        got_spikes = torch.empty_like(ref_spikes)
        got_v = torch.empty_like(ref_v)
        error = str(exc)

    spike_diff = (got_spikes - ref_spikes).abs()
    v_diff = (got_v - ref_v).abs()
    spike_mismatch_ratio = (got_spikes != ref_spikes).to(torch.float32).mean().item()
    spike_ok = torch.allclose(got_spikes, ref_spikes, rtol=rtol, atol=atol)
    v_ok = torch.allclose(got_v, ref_v, rtol=rtol, atol=atol)
    return {
        "case": case["name"],
        "kind": "conv_add_lif" if residual else "conv_lif",
        "T": T,
        "dtype": dtype_name,
        "shape": [case["N"], case["Cin"], case["H"], case["W"]],
        "out_channels": case["Cout"],
        "kernel": case["K"],
        "stride": case["stride"],
        "padding": case["padding"],
        "kernel_key": kernel_key,
        "runtime_kernel_key": runtime_kernel_key,
        "max_abs_diff_spike": float(spike_diff.max().item()),
        "mean_abs_diff_spike": float(spike_diff.mean().item()),
        "spike_mismatch_ratio": float(spike_mismatch_ratio),
        "max_abs_diff_v": float(v_diff.max().item()),
        "mean_abs_diff_v": float(v_diff.mean().item()),
        "allclose": bool(spike_ok and v_ok),
        "backend_hit": bool(backend_hit),
        "support_reasons": support_reasons,
        "error": error,
    }


def print_table(rows: List[Dict]):
    print("| case | kind | T | dtype | shape | key | K | stride | padding | spike max | spike mismatch | v max | allclose | backend_hit |")
    print("|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|---|")
    for row in rows:
        print(
            f"| {row['case']} | {row['kind']} | {row['T']} | {row['dtype']} | {row['shape']} -> {row['out_channels']} | "
            f"{row['runtime_kernel_key']} | {row['kernel']} | {row['stride']} | {row['padding']} | "
            f"{row['max_abs_diff_spike']:.3e} | {row['spike_mismatch_ratio']:.3e} | {row['max_abs_diff_v']:.3e} | "
            f"{row['allclose']} | {row['backend_hit']} |"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Correctness matrix for Chronos fused ConvLIF Triton kernels.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--rtol", type=float, default=None)
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--out-dir", default="kernel_config_correctness")
    parser.add_argument("--quick", action="store_true", help="Run a reduced T set for smoke testing.")
    parser.add_argument(
        "--use-autotune",
        action="store_true",
        help="Run correctness through autotuned dispatch. Default uses a fixed temporal schedule for speed.",
    )
    parser.add_argument(
        "--skip-residual",
        action="store_true",
        help="Only test ConvLIF, not ConvAddLIF residual temporal kernels.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    torch.manual_seed(2026)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(2026)
    rtol = args.rtol if args.rtol is not None else (1e-2 if args.dtype == "fp16" else 1e-4)
    atol = args.atol if args.atol is not None else (1e-2 if args.dtype == "fp16" else 1e-4)

    t_values = [1, 2] if args.quick else [1, 2, 4, 8, 16]
    rows = []
    for case in CASES:
        for T in t_values:
            for residual in ([False] if args.skip_residual else [False, True]):
                kind = "conv_add_lif" if residual else "conv_lif"
                print(f"[RUN] {case['name']} {kind} T={T} dtype={args.dtype}")
                rows.append(run_case(case, T, args.device, args.dtype, rtol, atol, use_autotune=args.use_autotune, residual=residual))

    print_table(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "kernel_config_correctness.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    failed = [row for row in rows if not row["allclose"] or not row["backend_hit"]]
    if failed:
        raise SystemExit(f"{len(failed)} kernel config correctness cases failed")


if __name__ == "__main__":
    main()
