"""Phase B (Kairos Mamba workload): fused_temporal_selective_scan kernel
unit test + microbench against the canonical eager step-by-step scan it
replaces (exp/mul/add/sum chain, matching KairosMambaBlockEager.forward's
scan portion, from hA=exp(...) through y=... ).
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

import runtime.snn_custom_ops as snn_custom_ops
from kernels.generated_temporal_selective_scan_kernel import run_fused_temporal_selective_scan_kernel


def eager_scan(x_seq, dt_seq, b_seq, c_seq, A, D, h_init):
    state = h_init.clone()
    ys = []
    for t in range(x_seq.shape[0]):
        x_t, dt_t, b_t, c_t = x_seq[t], dt_seq[t], b_seq[t], c_seq[t]
        hA = torch.exp(dt_t.unsqueeze(-1) * A)
        state = hA * state + (dt_t.unsqueeze(-1) * b_t.unsqueeze(1)) * x_t.unsqueeze(-1)
        y_t = (state * c_t.unsqueeze(1)).sum(-1) + D * x_t
        ys.append(y_t)
    return torch.stack(ys, dim=0), state


def _make_inputs(T, B, d_inner, d_state, seed):
    torch.manual_seed(seed)
    x_seq = torch.randn(T, B, d_inner, device="cuda") * 0.3
    dt_seq = torch.rand(T, B, d_inner, device="cuda") * 0.5 + 0.01
    b_seq = torch.randn(T, B, d_state, device="cuda") * 0.5
    c_seq = torch.randn(T, B, d_state, device="cuda") * 0.5
    A = -torch.rand(d_inner, d_state, device="cuda") - 0.5
    D = torch.ones(d_inner, device="cuda")
    h_init = torch.randn(B, d_inner, d_state, device="cuda") * 0.1
    return x_seq, dt_seq, b_seq, c_seq, A, D, h_init


def test_correctness():
    shapes = [(1, 1, 1, 16), (4, 3, 24, 16), (8, 2, 100, 16), (16, 4, 1536, 16)]
    all_pass = True
    for i, (T, B, d_inner, d_state) in enumerate(shapes):
        inputs = _make_inputs(T, B, d_inner, d_state, seed=i)
        y_ref, h_ref = eager_scan(*inputs)
        y_out, h_out = run_fused_temporal_selective_scan_kernel(*inputs)
        ok_y = torch.allclose(y_out, y_ref, atol=1e-3, rtol=1e-3)
        ok_h = torch.allclose(h_out, h_ref, atol=1e-3, rtol=1e-3)
        max_err_y = (y_out - y_ref).abs().max().item()
        max_err_h = (h_out - h_ref).abs().max().item()
        status = "PASS" if (ok_y and ok_h) else "FAIL"
        all_pass = all_pass and ok_y and ok_h
        print(f"[{status}] T={T} B={B} d_inner={d_inner} d_state={d_state} max_err_y={max_err_y:.2e} max_err_h={max_err_h:.2e}")
    assert all_pass
    print("[ALL PASS] correctness")


def test_custom_op_dispatch():
    snn_custom_ops.configure_fused_op("triton", strict_triton=True, verbose=False)
    inputs = _make_inputs(8, 2, 32, 16, seed=99)
    y_ref, h_ref = eager_scan(*inputs)
    y_out, h_out = torch.ops.snn_custom.fused_temporal_selective_scan(*inputs)
    assert torch.allclose(y_out, y_ref, atol=1e-3) and torch.allclose(h_out, h_ref, atol=1e-3)
    print("[PASS] custom_op dispatch matches eager reference")


def microbench():
    T, B, d_inner, d_state = 16, 16, 1536, 16  # matches KairosMamba spec defaults
    inputs = _make_inputs(T, B, d_inner, d_state, seed=7)

    def time_fn(fn, n=30, warmup=10):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        for i in range(n):
            starts[i].record()
            fn()
            ends[i].record()
        torch.cuda.synchronize()
        return torch.tensor([starts[i].elapsed_time(ends[i]) for i in range(n)]).mean().item()

    eager_ms = time_fn(lambda: eager_scan(*inputs))
    kernel_ms = time_fn(lambda: run_fused_temporal_selective_scan_kernel(*inputs))
    print(f"\n[MICROBENCH] T={T} B={B} d_inner={d_inner} d_state={d_state}")
    print(f"eager step-by-step scan: {eager_ms:.4f} ms")
    print(f"fused_temporal_selective_scan (triton): {kernel_ms:.4f} ms")
    print(f"speedup: {eager_ms / kernel_ms:.3f}x")


if __name__ == "__main__":
    test_correctness()
    test_custom_op_dispatch()
    microbench()
