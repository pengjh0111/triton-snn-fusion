"""TC kernel cold-search baseline for the same 6 model Linear shapes at the
default (batch=16,seq=256,T=16) scenario, for direct before/after comparison
against the new codegen kernel's Task 1.1 numbers.
"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from kernels.generated_temporal_transformer_lif_kernels import run_temporal_batched_linear_lif as run_tc

LINEAR_SHAPES = [
    ("input_proj", 768, 256),
    ("qkv", 256, 768),
    ("attn_proj", 256, 256),
    ("fc1_up", 256, 1024),
    ("fc2_down", 1024, 256),
    ("classifier", 256, 100),
]

def time_call(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t0

def main():
    T, rows = 16, 16 * 256
    dtype = torch.float16
    total = 0.0
    for label, in_features, out_features in LINEAR_SHAPES:
        x_seq = torch.randn(T, rows, 1, in_features, device="cuda", dtype=dtype) * 0.05
        weight = torch.randn(out_features, in_features, device="cuda", dtype=dtype) * 0.02
        bias = torch.randn(out_features, device="cuda", dtype=dtype) * 0.01
        v_init = torch.zeros(rows, 1, out_features, device="cuda", dtype=dtype)
        elapsed = time_call(lambda: run_tc(x_seq, weight, bias, v_init, 1.0, 0.0, 2.0))
        total += elapsed
        print(f"{label:<12} rows={rows} T={T} shape={in_features}->{out_features:<5} cold_search_s={elapsed:.3f}")
    print(f"\nTotal TC cold-search time across 6 shapes (default_b16_seq256 scenario): {total:.2f}s")

if __name__ == "__main__":
    main()
