"""Phase B (Kairos ConvLSTM workload): fused_convlstm_cell kernel unit test
+ microbench against the eager gate-chain it replaces (torch.chunk + 2x
sigmoid + tanh + mul + add + sigmoid + tanh + mul, matching
KairosConvLSTMCellEager.forward's second half, everything after
xproj+hproj is summed).
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

import runtime.snn_custom_ops as snn_custom_ops
from kernels.generated_convlstm_cell_kernel import run_fused_convlstm_cell_kernel


def eager_gate_chain(gates_sum, c_prev):
    i, f, g, o = torch.chunk(gates_sum, 4, dim=1)
    c_t = torch.sigmoid(f) * c_prev + torch.sigmoid(i) * torch.tanh(g)
    h_t = torch.sigmoid(o) * torch.tanh(c_t)
    return h_t, c_t


def test_correctness():
    torch.manual_seed(0)
    shapes = [(1, 1, 1, 1), (2, 4, 5, 5), (4, 64, 16, 16), (8, 320, 7, 7), (3, 17, 9, 13)]
    all_pass = True
    for B, C, H, W in shapes:
        gates_sum = torch.randn(B, 4 * C, H, W, device="cuda", dtype=torch.float32)
        c_prev = torch.randn(B, C, H, W, device="cuda", dtype=torch.float32)
        h_ref, c_ref = eager_gate_chain(gates_sum, c_prev)
        h_t, c_t = run_fused_convlstm_cell_kernel(gates_sum, c_prev)
        ok_h = torch.allclose(h_t, h_ref, atol=1e-5, rtol=1e-5)
        ok_c = torch.allclose(c_t, c_ref, atol=1e-5, rtol=1e-5)
        max_err_h = (h_t - h_ref).abs().max().item()
        max_err_c = (c_t - c_ref).abs().max().item()
        status = "PASS" if (ok_h and ok_c) else "FAIL"
        all_pass = all_pass and ok_h and ok_c
        print(f"[{status}] shape=(B={B},C={C},H={H},W={W}) max_err_h={max_err_h:.2e} max_err_c={max_err_c:.2e}")
    assert all_pass
    print("[ALL PASS] correctness")


def test_custom_op_dispatch():
    snn_custom_ops.configure_fused_op("triton", strict_triton=True, verbose=False)
    torch.manual_seed(1)
    gates_sum = torch.randn(2, 32, 8, 8, device="cuda")
    c_prev = torch.randn(2, 8, 8, 8, device="cuda")
    h_t, c_t = torch.ops.snn_custom.fused_convlstm_cell(gates_sum, c_prev)
    h_ref, c_ref = eager_gate_chain(gates_sum, c_prev)
    assert torch.allclose(h_t, h_ref, atol=1e-5) and torch.allclose(c_t, c_ref, atol=1e-5)
    print("[PASS] custom_op dispatch matches eager reference")


def microbench():
    torch.manual_seed(2)
    B, C, H, W = 16, 64, 64, 64  # matches KairosConvLSTM spec defaults
    gates_sum = torch.randn(B, 4 * C, H, W, device="cuda")
    c_prev = torch.randn(B, C, H, W, device="cuda")

    def time_fn(fn, n=50, warmup=10):
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

    eager_ms = time_fn(lambda: eager_gate_chain(gates_sum, c_prev))
    kernel_ms = time_fn(lambda: run_fused_convlstm_cell_kernel(gates_sum, c_prev))
    print(f"\n[MICROBENCH] shape=(B={B},C={C},H={H},W={W})")
    print(f"eager_gate_chain: {eager_ms:.4f} ms")
    print(f"fused_convlstm_cell (triton): {kernel_ms:.4f} ms")
    print(f"speedup: {eager_ms / kernel_ms:.3f}x")


if __name__ == "__main__":
    test_correctness()
    test_custom_op_dispatch()
    microbench()
