"""Phase B (Kairos DeepSpeech2 workload): fused_gru_cell kernel unit test
+ microbench against the eager gate-chain it replaces (matching
KairosGRUCellEager.forward: 2x chunk, 2x sigmoid, tanh, mul, add).
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

import runtime.snn_custom_ops as snn_custom_ops
from kernels.generated_gru_cell_kernel import run_fused_gru_cell_kernel


def eager_gate_chain(xproj, hproj, h_prev):
    xproj_r, xproj_z, xproj_n = torch.chunk(xproj, 3, dim=-1)
    hproj_r, hproj_z, hproj_n = torch.chunk(hproj, 3, dim=-1)
    r = torch.sigmoid(xproj_r + hproj_r)
    z = torch.sigmoid(xproj_z + hproj_z)
    n = torch.tanh(xproj_n + r * hproj_n)
    return (1 - z) * n + z * h_prev


def test_correctness():
    torch.manual_seed(0)
    shapes = [(1, 1), (2, 5), (4, 800), (16, 800), (3, 29)]
    all_pass = True
    for B, H in shapes:
        xproj = torch.randn(B, 3 * H, device="cuda")
        hproj = torch.randn(B, 3 * H, device="cuda")
        h_prev = torch.randn(B, H, device="cuda")
        h_ref = eager_gate_chain(xproj, hproj, h_prev)
        h_t = run_fused_gru_cell_kernel(xproj, hproj, h_prev)
        ok = torch.allclose(h_t, h_ref, atol=1e-5, rtol=1e-5)
        max_err = (h_t - h_ref).abs().max().item()
        status = "PASS" if ok else "FAIL"
        all_pass = all_pass and ok
        print(f"[{status}] B={B} H={H} max_err={max_err:.2e}")
    assert all_pass
    print("[ALL PASS] correctness")


def test_custom_op_dispatch():
    snn_custom_ops.configure_fused_op("triton", strict_triton=True, verbose=False)
    torch.manual_seed(1)
    xproj = torch.randn(4, 2400, device="cuda")
    hproj = torch.randn(4, 2400, device="cuda")
    h_prev = torch.randn(4, 800, device="cuda")
    h_t = torch.ops.snn_custom.fused_gru_cell(xproj, hproj, h_prev)
    h_ref = eager_gate_chain(xproj, hproj, h_prev)
    assert torch.allclose(h_t, h_ref, atol=1e-5)
    print("[PASS] custom_op dispatch matches eager reference")


def microbench():
    torch.manual_seed(2)
    B, H = 16, 800  # matches KairosDeepSpeech2 spec defaults
    xproj = torch.randn(B, 3 * H, device="cuda")
    hproj = torch.randn(B, 3 * H, device="cuda")
    h_prev = torch.randn(B, H, device="cuda")

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

    eager_ms = time_fn(lambda: eager_gate_chain(xproj, hproj, h_prev))
    kernel_ms = time_fn(lambda: run_fused_gru_cell_kernel(xproj, hproj, h_prev))
    print(f"\n[MICROBENCH] B={B} H={H}")
    print(f"eager_gate_chain: {eager_ms:.4f} ms")
    print(f"fused_gru_cell (triton): {kernel_ms:.4f} ms")
    print(f"speedup: {eager_ms / kernel_ms:.3f}x")


if __name__ == "__main__":
    test_correctness()
    test_custom_op_dispatch()
    microbench()
