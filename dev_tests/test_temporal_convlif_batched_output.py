import argparse
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

import runtime.snn_custom_ops as snn_custom_ops


def _dtype(name: str):
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(name)


def _make_tensors(T: int, device: str, dtype: torch.dtype, stride: int):
    n, cin, cout, h, w = 2, 4, 5, 16, 16
    scale = 0.05
    xs = [
        (torch.randn(n, cin, h, w, device=device, dtype=dtype) * scale).contiguous()
        for _ in range(T)
    ]
    weight = (torch.randn(cout, cin, 3, 3, device=device, dtype=dtype) * scale).contiguous()
    bias = (torch.randn(cout, device=device, dtype=dtype) * scale).contiguous()
    out_h = (h + 2 - 3) // stride + 1
    out_w = (w + 2 - 3) // stride + 1
    residuals = [
        (torch.randn(n, cout, out_h, out_w, device=device, dtype=dtype) * scale).contiguous()
        for _ in range(T)
    ]
    v_init = torch.tensor(0.0, device=device, dtype=dtype)
    return xs, residuals, weight, bias, v_init


def _assert_layout(stack: torch.Tensor, batched: torch.Tensor, rtol: float, atol: float):
    expected = stack.flatten(0, 1)
    diff = (expected - batched).abs()
    allclose = torch.allclose(expected, batched, rtol=rtol, atol=atol)
    print(
        f"stack={tuple(stack.shape)} batched={tuple(batched.shape)} "
        f"max={diff.max().item():.3e} mean={diff.mean().item():.3e} allclose={allclose}"
    )
    if not allclose:
        raise AssertionError("batched_tn output does not match stack.flatten(0, 1)")


def run_case(args, T: int, stride: int, residual: bool):
    dtype = _dtype(args.dtype)
    xs, residuals, weight, bias, v_init = _make_tensors(T, args.device, dtype, stride)
    stride_pair = [stride, stride]
    padding = [1, 1]
    dilation = [1, 1]
    if residual:
        stack, v_stack = torch.ops.snn_custom.fused_temporal_conv_add_lif_state.default(
            xs,
            residuals,
            weight,
            bias,
            v_init,
            stride_pair,
            padding,
            dilation,
            1,
            1.0,
            0.0,
            2.0,
            False,
        )
        batched, v_batched = torch.ops.snn_custom.fused_temporal_conv_add_lif_state_batched_tn.default(
            xs,
            residuals,
            weight,
            bias,
            v_init,
            stride_pair,
            padding,
            dilation,
            1,
            1.0,
            0.0,
            2.0,
            False,
        )
    else:
        stack, v_stack = torch.ops.snn_custom.fused_temporal_conv_lif_state.default(
            xs,
            weight,
            bias,
            v_init,
            stride_pair,
            padding,
            dilation,
            1,
            1.0,
            0.0,
            2.0,
            False,
        )
        batched, v_batched = torch.ops.snn_custom.fused_temporal_conv_lif_state_batched_tn.default(
            xs,
            weight,
            bias,
            v_init,
            stride_pair,
            padding,
            dilation,
            1,
            1.0,
            0.0,
            2.0,
            False,
        )
    rtol = args.rtol if args.rtol is not None else (1e-2 if args.dtype == "fp16" else 1e-4)
    atol = args.atol if args.atol is not None else (1e-2 if args.dtype == "fp16" else 1e-4)
    print(f"T={T:<2d} stride={stride} residual={residual} dtype={args.dtype}")
    _assert_layout(stack, batched, rtol, atol)
    v_diff = (v_stack - v_batched).abs()
    v_ok = torch.allclose(v_stack, v_batched, rtol=rtol, atol=atol)
    print(f"v_final max={v_diff.max().item():.3e} mean={v_diff.mean().item():.3e} allclose={v_ok}")
    if not v_ok:
        raise AssertionError("batched_tn v_final differs from stack mode")


def parse_args():
    parser = argparse.ArgumentParser(description="Temporal ConvLIF batched_tn output layout correctness test.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--fused-op-backend", choices=("torch", "triton"), default="torch")
    parser.add_argument("--strict-triton", action="store_true")
    parser.add_argument("--print-fused-op-calls", action="store_true")
    parser.add_argument("--T", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--rtol", type=float, default=None)
    parser.add_argument("--atol", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    snn_custom_ops.configure_fused_op(
        backend=args.fused_op_backend,
        strict_triton=args.strict_triton,
        verbose=args.print_fused_op_calls,
    )
    snn_custom_ops.reset_fused_op_call_stats()
    for T in args.T:
        for stride in (1, 2):
            for residual in (False, True):
                run_case(args, T, stride, residual)
    print("fused_op_call_stats:", snn_custom_ops.get_fused_op_call_stats())


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
