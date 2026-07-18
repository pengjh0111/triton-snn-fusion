"""Step 1 (two-stage lowering, scoped-down plan): per-layer A/B adjudication.

Path A (status quo): the fused_temporal_general Triton kernel, autotune-best
config, grid=(M,OC) with T handled by an in-kernel loop.

Path B (two-stage, both components pre-existing -- no new kernel):
  Stage A: x_seq.reshape(T*B, C, H, W) -> F.conv2d -> reshape back to
           [T, B, C', OH, OW]  (this is exactly what
           compiler/fx_spatial_batching.py already does for LIF-less convs)
  Stage B: torch.ops.snn_custom.fused_temporal_lif_state(pre_act, v_init,
           v_threshold, v_reset, tau, detach_reset) -- the existing
           standalone LIF-scan custom op (runtime/triton_temporal_lif_backend.py),
           already registered CPU/CUDA/Meta, currently used for the
           classifier tail, not for conv layers.

For every distinct MobileNetV2 conv+LIF shape (reusing the exact layer list
from dev_tests/step0_layer_table.py) at batch in {4, 16}: times both paths
(locked clocks, CUDA events, N iters), cross-checks numerical equivalence
(spike + v_final, rtol=atol=1e-2 matching dev_tests/test_batch_window_correctness_matrix.py's
convention), and reports a table plus a profiler-weighted end-to-end upper
bound using the category shares already measured in
dev_tests/profile_mobilenet_categories.py (pwconv_lif=40.90%, dwconv_lif=17.97%,
regular_conv_lif=0.88% of GPU time at batch=4/T=16 -- rerun once for batch=16
if the decision gate is close).
"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

import runtime.snn_custom_ops as snn_custom_ops
import kernels.benchmark_conv_lif_temporal_general as K
from dev_tests.step0_layer_table import LAYERS, _conv_out_hw

T_STEPS = 16
V_THRESHOLD = 1.0
V_RESET = 0.0
TAU = 2.0
N_ITERS = 50
N_WARMUP = 10


def _make_weight(kind, in_ch, out_ch, device, dtype):
    if kind == "general":
        return torch.randn(out_ch, in_ch, 3, 3, device=device, dtype=dtype) * 0.05
    if kind == "pw":
        return torch.randn(out_ch, in_ch, 1, 1, device=device, dtype=dtype) * 0.05
    if kind == "dw":
        return torch.randn(out_ch, 1, 3, 3, device=device, dtype=dtype) * 0.05
    raise ValueError(kind)


def _conv2d_params(kind, stride):
    if kind == "general":
        return dict(stride=stride, padding=1)
    if kind == "pw":
        return dict(stride=1, padding=0)
    if kind == "dw":
        return dict(stride=stride, padding=1)
    raise ValueError(kind)


def _kernel_key(kind, stride):
    if kind == "general":
        return "k3_s2_p1"
    if kind == "pw":
        return "k1_s1_p0"
    if kind == "dw":
        return "depthwise_k3_s1_p1" if stride == 1 else "depthwise_k3_s2_p1"
    raise ValueError(kind)


def _time_cuda(fn, n_iters=N_ITERS, n_warmup=N_WARMUP):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iters)]
    for i in range(n_iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    ms = torch.tensor([starts[i].elapsed_time(ends[i]) for i in range(n_iters)])
    return ms.mean().item(), ms.std().item()


def bench_one(name, kind, in_ch, out_ch, h_in, stride, batch, device="cuda", dtype=torch.float32):
    weight = _make_weight(kind, in_ch, out_ch, device, dtype)
    bias = torch.zeros(out_ch, device=device, dtype=dtype)
    x_seq = torch.randn(T_STEPS, batch, in_ch, h_in, h_in, device=device, dtype=dtype) * 0.1

    variant = K.KERNEL_VARIANTS[_kernel_key(kind, stride)]
    oh, ow = _conv_out_hw(h_in, h_in, variant["kernel"], variant["stride"], variant["pad"])
    v_init_a = torch.zeros(batch, out_ch, oh, ow, device=device, dtype=dtype)
    v_init_b = torch.zeros(batch, out_ch, oh, ow, device=device, dtype=dtype)

    kernel_key = _kernel_key(kind, stride)

    def path_a():
        return K.run_fused_temporal_general_autotuned(
            x_seq, weight, bias, kernel_key=kernel_key, v_init=v_init_a.clone()
        )

    conv_kwargs = _conv2d_params(kind, stride)
    groups = out_ch if kind == "dw" else 1

    def path_b():
        x_flat = x_seq.reshape(T_STEPS * batch, in_ch, h_in, h_in)
        pre_act = F.conv2d(x_flat, weight, bias, groups=groups, **conv_kwargs)
        pre_act = pre_act.reshape(T_STEPS, batch, out_ch, oh, ow)
        return torch.ops.snn_custom.fused_temporal_lif_state(
            pre_act, v_init_b.clone(), V_THRESHOLD, V_RESET, TAU, True
        )

    # warm autotune cache / triton compile cache for both paths before timing
    spikes_a, v_a = path_a()
    spikes_b, v_b = path_b()
    torch.cuda.synchronize()

    max_err_spike = (spikes_a - spikes_b).abs().max().item()
    max_err_v = (v_a - v_b).abs().max().item()
    close = torch.allclose(spikes_a, spikes_b, rtol=1e-2, atol=1e-2) and torch.allclose(
        v_a, v_b, rtol=1e-2, atol=1e-2
    )

    a_ms, a_std = _time_cuda(lambda: path_a())
    b_ms, b_std = _time_cuda(lambda: path_b())

    return dict(
        name=name, kind=kind, in_ch=in_ch, out_ch=out_ch, h_in=h_in, oh=oh, ow=ow,
        batch=batch, a_ms=a_ms, a_std=a_std, b_ms=b_ms, b_std=b_std,
        winner="B" if b_ms < a_ms else "A", ratio=a_ms / b_ms if b_ms else float("inf"),
        numeric_close=close, max_err_spike=max_err_spike, max_err_v=max_err_v,
    )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, nargs="+", default=[4, 16])
    args = parser.parse_args()

    snn_custom_ops.configure_fused_op("triton", strict_triton=False, verbose=False)

    all_rows = []
    for batch in args.batches:
        print(f"\n{'='*30} batch={batch} {'='*30}", flush=True)
        print(f"{'layer':<14}{'kind':<6}{'shape':<26}{'A_ms':>9}{'B_ms':>9}{'winner':>8}{'ratio':>8}  numeric_ok  max_err(spk,v)", flush=True)
        for name, kind, in_ch, out_ch, h_in, stride in LAYERS:
            r = bench_one(name, kind, in_ch, out_ch, h_in, stride, batch)
            all_rows.append(r)
            shape_str = f"{r['in_ch']}->{r['out_ch']}@{r['h_in']}x{r['h_in']}"
            print(
                f"{r['name']:<14}{r['kind']:<6}{shape_str:<26}{r['a_ms']:>9.4f}{r['b_ms']:>9.4f}"
                f"{r['winner']:>8}{r['ratio']:>8.3f}  {str(r['numeric_close']):<10}"
                f"({r['max_err_spike']:.2e},{r['max_err_v']:.2e})",
                flush=True,
            )

    print("\n\n========== SUMMARY ==========")
    for batch in args.batches:
        rows = [r for r in all_rows if r["batch"] == batch]
        n_b_wins = sum(1 for r in rows if r["winner"] == "B")
        all_numeric_ok = all(r["numeric_close"] for r in rows)
        print(f"batch={batch}: B wins {n_b_wins}/{len(rows)} layers; all numeric checks pass: {all_numeric_ok}")


if __name__ == "__main__":
    main()
