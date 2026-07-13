import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.snn_custom_ops import lif_forward_state_torch
from runtime.triton_convlif_backend import (
    run_triton_fused_temporal_conv_add_lif_state,
    run_triton_fused_temporal_depthwise_conv_lif_state,
)


def _dtype(name: str):
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(name)


def _ref(x_seq, weight, bias, v_init, stride, residual_seq=None):
    v = v_init
    spikes = []
    for t in range(x_seq.shape[0]):
        y = F.conv2d(x_seq[t], weight, bias, stride=(stride, stride), padding=(1, 1), groups=x_seq.shape[2])
        if residual_seq is not None:
            y = y + residual_seq[t]
        spike, v = lif_forward_state_torch(y, v, 1.0, 0.0, 2.0, False)
        spikes.append(spike)
    return torch.stack(spikes, dim=0), v


def _make_case(T: int, stride: int, device: str, dtype: torch.dtype):
    torch.manual_seed(1000 + T * 10 + stride)
    N, C, H, W = 2, 32, 32, 32
    x_seq = (torch.randn(T, N, C, H, W, device=device, dtype=dtype) * 0.03).contiguous()
    weight = (torch.randn(C, 1, 3, 3, device=device, dtype=dtype) * 0.03).contiguous()
    bias = (torch.randn(C, device=device, dtype=dtype) * 0.03).contiguous()
    out_h = (H + 2 - 3) // stride + 1
    out_w = (W + 2 - 3) // stride + 1
    residual = (torch.randn(T, N, C, out_h, out_w, device=device, dtype=dtype) * 0.03).contiguous()
    v_init = torch.tensor(0.0, device=device, dtype=dtype)
    return x_seq, weight, bias, residual, v_init


def _check(T: int, stride: int, dtype_name: str, device: str, residual: bool, use_autotune: bool):
    dtype = _dtype(dtype_name)
    x_seq, weight, bias, residual_seq, v_init = _make_case(T, stride, device, dtype)
    residual_arg = residual_seq if residual else None
    ref_spike, ref_v = _ref(x_seq, weight, bias, v_init, stride, residual_arg)
    xs = [x_seq[t] for t in range(T)]
    if residual:
        result = run_triton_fused_temporal_conv_add_lif_state(
            xs,
            [residual_seq[t] for t in range(T)],
            weight,
            bias,
            v_init,
            [stride, stride],
            [1, 1],
            [1, 1],
            x_seq.shape[2],
            1.0,
            0.0,
            2.0,
            False,
            use_autotune=use_autotune,
        )
    else:
        result = run_triton_fused_temporal_depthwise_conv_lif_state(
            xs,
            weight,
            bias,
            v_init,
            [stride, stride],
            [1, 1],
            [1, 1],
            x_seq.shape[2],
            1.0,
            0.0,
            2.0,
            False,
            use_autotune=use_autotune,
        )
    torch.cuda.synchronize()
    atol = 1e-2 if dtype_name == "fp16" else 1e-5
    rtol = 1e-2 if dtype_name == "fp16" else 1e-5
    spike_ok = torch.allclose(result.spikes, ref_spike, atol=atol, rtol=rtol)
    v_ok = torch.allclose(result.v_next, ref_v, atol=atol, rtol=rtol)
    if not result.used_triton or not spike_ok or not v_ok:
        raise AssertionError(
            f"depthwise case failed T={T} stride={stride} residual={residual} "
            f"key={result.kernel_key} triton={result.used_triton} "
            f"spike_max={(result.spikes - ref_spike).abs().max().item()} "
            f"v_max={(result.v_next - ref_v).abs().max().item()}"
        )
    print(
        f"[PASS] dtype={dtype_name} T={T} stride={stride} residual={residual} "
        f"key={result.kernel_key} autotune={use_autotune}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("fp32", "fp16"), default="fp16")
    parser.add_argument("--T", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--use-autotune", action="store_true")
    parser.add_argument("--strict-triton", action="store_true")
    args = parser.parse_args()
    if args.device != "cuda":
        raise SystemExit("depthwise Triton test requires --device cuda")
    for T in args.T:
        for stride in (1, 2):
            for residual in (False, True):
                _check(T, stride, args.dtype, args.device, residual, args.use_autotune)


if __name__ == "__main__":
    main()
