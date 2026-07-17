"""Task 0.2 attribution diagnostic: rows=197, T=4, shape=768->2304 is still
<1.0x (codegen vs tc) even after the spatial config pool was expanded to a
strict superset of TC's own sweep. Dump both backends' selected autotune
config and Triton compiled-kernel resource metadata (registers, shared mem,
spills) to see whether the gap is scheduling (config choice) or fixed
per-launch overhead in the generated kernel body.
"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from kernels.generated_temporal_transformer_lif_kernels import (
    run_temporal_batched_linear_lif as run_tc,
    _temporal_batched_linear_lif_kernel,
)
from kernels.benchmark_batched_linear_lif_temporal_general import (
    run_fused_batched_linear_lif as run_codegen,
    _autotuned_kernels,
)


def warm_up_gpu_clocks(seconds=3.0):
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        a @ b
    torch.cuda.synchronize()


def time_cuda(fn, warmup=30, rep=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / rep


def dump_compiled_metadata(prefix, autotuner_or_kernel):
    cache = getattr(autotuner_or_kernel, "cache", None)
    if cache is None:
        print(f"{prefix}: no .cache attribute found")
        return
    print(f"{prefix}: {len(cache)} compiled variant(s) in cache")
    for key, compiled in cache.items():
        try:
            metadata = compiled.metadata
            n_regs = getattr(metadata, "num_regs", getattr(metadata, "n_regs", None))
            shared = getattr(metadata, "shared", None)
            spills = getattr(metadata, "num_spills", getattr(metadata, "n_spills", None))
            print(f"  key={key} n_regs={n_regs} shared_bytes={shared} spills={spills}")
        except Exception as exc:
            print(f"  key={key} <no metadata: {exc}>")


def main():
    torch.manual_seed(0)
    T, rows, in_features, out_features = 4, 197, 768, 2304
    dtype = torch.float16
    x_seq = (torch.randn(T, 1, rows, in_features, device="cuda", dtype=dtype) * 0.04).contiguous()
    weight = (torch.randn(out_features, in_features, device="cuda", dtype=dtype) * 0.02).contiguous()
    bias = (torch.randn(out_features, device="cuda", dtype=dtype) * 0.01).contiguous()
    v_init = torch.zeros(1, rows, out_features, device="cuda", dtype=dtype)

    warm_up_gpu_clocks(3.0)

    # Prime autotune caches.
    run_codegen(x_seq, weight, bias, v_init, 1.0, 0.0, 2.0)
    run_tc(x_seq, weight, bias, v_init, 1.0, 0.0, 2.0)

    tc_ms = time_cuda(lambda: run_tc(x_seq, weight, bias, v_init, 1.0, 0.0, 2.0))
    codegen_ms = time_cuda(lambda: run_codegen(x_seq, weight, bias, v_init, 1.0, 0.0, 2.0))
    print(f"tc_ms={tc_ms:.4f} codegen_ms={codegen_ms:.4f} ratio(tc/codegen)={tc_ms/codegen_ms:.3f}x")

    tc_best = getattr(_temporal_batched_linear_lif_kernel, "best_config", None)
    print(f"\nTC best_config: {tc_best.all_kwargs() if tc_best else None} "
          f"num_warps={getattr(tc_best,'num_warps',None)} num_stages={getattr(tc_best,'num_stages',None)}")

    codegen_kernel = _autotuned_kernels["batched_linear_lif"]
    codegen_best = getattr(codegen_kernel, "best_config", None)
    print(f"codegen best_config: {codegen_best.all_kwargs() if codegen_best else None} "
          f"num_warps={getattr(codegen_best,'num_warps',None)} num_stages={getattr(codegen_best,'num_stages',None)}")

    print()
    dump_compiled_metadata("TC kernel", _temporal_batched_linear_lif_kernel)
    dump_compiled_metadata("codegen kernel", codegen_kernel)

    # Grid size comparison -- rows=197 with different BLOCK_M can produce a
    # very different number of CTAs, which matters a lot at this small a
    # problem size (occupancy-bound, not compute-bound).
    if tc_best is not None:
        bm = int(tc_best.all_kwargs()["BLOCK_M"])
        bn = int(tc_best.all_kwargs()["BLOCK_N"])
        print(f"\nTC grid: cdiv({rows},{bm})={-(-rows//bm)} x cdiv({out_features},{bn})={-(-out_features//bn)} "
              f"= {(-(-rows//bm))*(-(-out_features//bn))} CTAs")
    if codegen_best is not None:
        bm = int(codegen_best.all_kwargs()["BLOCK_M"])
        bn = int(codegen_best.all_kwargs()["BLOCK_N"])
        print(f"codegen grid: cdiv({rows},{bm})={-(-rows//bm)} x cdiv({out_features},{bn})={-(-out_features//bn)} "
              f"= {(-(-rows//bm))*(-(-out_features//bn))} CTAs")


if __name__ == "__main__":
    main()
