"""Hand-written / standard-implementation comparison for Kairos fused kernels.

Pure diagnostic script (zero production-code changes). For each of four
computation *patterns* (a spatial operator + its state recurrence) it compares
the Kairos-generated fused kernel against the field-standard hand-written /
reference implementation of the *same* mathematical function:

    pattern           Kairos kernel                                   opponent
    ---------------   ---------------------------------------------   ------------------------------
    conv+bn+lif       run_triton_fused_temporal_conv_lif_state        SpikingJelly (cuDNN conv + fused LIF, cupy)
    linear+lif        run_fused_temporal_linear_lif_state_kernel      SpikingJelly (Linear + fused LIF, cupy)
    selective_scan    run_fused_temporal_selective_scan_kernel        mamba-ssm official selective_scan_fn
    convlstm          run_fused_convlstm_cell_kernel                  standard ConvLSTM cell (cuDNN conv + PyTorch pointwise gating)

Protocol per (pattern, T):
  1. correctness gate: run both implementations on identical inputs; only if
     outputs agree within tolerance (rtol=1e-3, relative tolerance for the
     non-linear scan/convlstm paths) do we time.
  2. timing: CUDA-event, warmup 20 / repeat 100, mean +/- std.
  3. primary axis: pct_of_ref = ref_ms / kairos_ms * 100
     (>100% => Kairos faster, 80-100% => close).

Missing opponents (spikingjelly / cupy / mamba-ssm) are recorded and their
patterns skipped, never silently faked.

The time_cuda / lock_gpu_clock / environment_metadata helpers mirror the ones
in dev_tests/motivation_three_taxes.py (reimplemented locally so this script
has no dependency on that module's heavy imports).
"""

import argparse
import atexit
import csv
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _install_numpy_compat_shim() -> None:
    """Restore the deprecated numpy scalar aliases removed in NumPy>=1.24.

    SpikingJelly's cupy backend arg-checker (auto_cuda/base.py) still uses
    ``np.int`` (and datasets use ``np.bool``); NumPy 2.x removed those aliases,
    raising AttributeError inside the fused-LIF kernel launch. Restoring the
    aliases to their documented builtin equivalents reproduces the pre-1.24
    behaviour exactly (this is the substitution NumPy's own error suggests),
    so the cupy opponent kernel runs unmodified. Only fills names that are
    actually missing.
    """
    for name, builtin in (("int", int), ("float", float), ("bool", bool), ("object", object)):
        if not hasattr(np, name):
            setattr(np, name, builtin)


_install_numpy_compat_shim()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
# Let cuDNN pick its fastest conv algo (benefits the SpikingJelly opponent's conv);
# the algo is chosen during warmup so it is stable/capturable during CUDA-graph capture.
torch.backends.cudnn.benchmark = True

# Kairos kernels under test (production code, imported read-only).
from runtime.triton_convlif_backend import run_triton_fused_temporal_conv_lif_state
from kernels.generated_temporal_linear_lif_kernel import (
    run_fused_temporal_linear_lif_state_kernel,
)
from kernels.generated_temporal_selective_scan_kernel import (
    run_fused_temporal_selective_scan_kernel,
)
from kernels.generated_convlstm_cell_kernel import run_fused_convlstm_cell_kernel

# ---------------------------------------------------------------------------
# LIF parameters. The conv+bn+lif Triton kernel hard-codes these as compile-time
# constants (see runtime/triton_convlif_backend.check_triton_support: tau=2.0,
# v_threshold=1.0, v_reset=0.0, detach_reset=False). We use the same values for
# the linear+lif path so both patterns share one LIF definition, and align the
# SpikingJelly opponent to them exactly (decay_input=True, hard reset).
# ---------------------------------------------------------------------------
LIF_TAU = 2.0
LIF_V_THRESHOLD = 1.0
LIF_V_RESET = 0.0
LIF_DETACH_RESET = False

SUPPORTED_T = (1, 2, 4, 8, 16)  # shared x-axis default.
# Per-kernel temporal-window limits. The linear+lif and selective_scan kernels
# hard-reject T outside {1,2,4,8,16} (an in-kernel check); conv+bn+lif and the
# looped convlstm cell handle larger windows (T=32/64), so those extend the axis.
ALLOWED_T = (1, 2, 4, 8, 16, 32, 64, 128)  # any T the CLI will accept
# Per-pattern temporal ceilings. The default {1,2,4,8,16} guard on the
# linear/scan kernels is conservative (their loops are tl.static_range unrolls);
# KAIROS_ALLOW_EXTENDED_T=1 opts into the verified extended sets:
#   * selective_scan -> up to 128 (verified vs mamba; T=128 is the crossover
#     where mamba's parallel scan overtakes, and compiles in ~55min once).
#   * linear_lif     -> up to 64 (T>=128 unverified + slow static-unroll compile).
#   * conv_bn_lif    -> up to 64 (static-unroll would risk ~hour compiles at 128).
#   * convlstm       -> any T (single-step cell looped; no unroll/compile cost).
if os.environ.get("KAIROS_ALLOW_EXTENDED_T", "") == "1":
    # Full sweep to 128 for all patterns (verified correct via each run's gate).
    # Note: linear/conv/scan use static-unroll temporal loops, so T=128 compiles
    # slowly (~tens of minutes each, cached afterward); convlstm loops with no
    # compile cost.
    SCAN_SUPPORTED_T = (1, 2, 4, 8, 16, 32, 64, 128)
    LINEAR_SUPPORTED_T = (1, 2, 4, 8, 16, 32, 64, 128)
    CONV_SUPPORTED_T = (1, 2, 4, 8, 16, 32, 64, 128)
else:
    SCAN_SUPPORTED_T = LINEAR_SUPPORTED_T = CONV_SUPPORTED_T = (1, 2, 4, 8, 16)


# ---------------------------------------------------------------------------
# Timing / clock-lock / environment helpers (mirror motivation_three_taxes.py).
# ---------------------------------------------------------------------------
def time_cuda(fn: Callable, warmup: int, repeat: int) -> Tuple[float, float]:
    # Compile / autotune happen on the first (untimed) call, outside the events.
    with torch.no_grad():
        fn()
    torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return statistics.mean(samples), statistics.pstdev(samples)


def _flatten_cuda_tensors(obj) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    if isinstance(obj, torch.Tensor):
        if obj.is_cuda:
            out.append(obj)
    elif isinstance(obj, (tuple, list)):
        for x in obj:
            out.extend(_flatten_cuda_tensors(x))
    return out


def time_cuda_graph(fn: Callable, warmup: int, repeat: int) -> Tuple[float, float]:
    """Capture fn() into a CUDA graph and time graph.replay().

    This removes CPU launch/framework overhead from *both* sides symmetrically,
    isolating pure GPU kernel-execution time. Requires that fn() launch all its
    work on the current (capture) stream and not synchronize during capture:
      * Triton/torch kernels do this automatically.
      * cupy (SpikingJelly's fused LIF) launches on cupy's own current stream, so
        its ref_fn binds cupy to torch.cuda.current_stream() via an ExternalStream
        (see build_conv_bn_lif / build_linear_lif); otherwise capture would miss
        the LIF kernel.
    A post-capture "empty capture" guard zeroes the outputs, replays once, and
    asserts the replay wrote non-zero results -- catching the silent failure mode
    where a kernel launched on the wrong stream was not captured.
    """
    # Warm up: autotune, torch.compile, and memory-pool population must finish
    # before capture (they allocate / synchronize, which is illegal mid-capture).
    with torch.no_grad():
        for _ in range(max(warmup, 5)):
            fn()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.no_grad():
        with torch.cuda.graph(graph):
            out = fn()
    torch.cuda.synchronize()

    # Empty-capture guard: replay must actually write the outputs.
    flat = _flatten_cuda_tensors(out)
    if not flat:
        raise RuntimeError("cudagraph: fn() returned no CUDA tensor to validate")
    for t in flat:
        t.zero_()
    graph.replay()
    torch.cuda.synchronize()
    wrote = any(bool((t.abs().sum() > 0).item()) for t in flat)
    if not wrote:
        raise RuntimeError(
            "cudagraph replay wrote all-zero outputs -- capture likely missed a "
            "kernel (e.g. cupy launched on the wrong stream)"
        )

    samples = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    # Free the graph's private pool before the next capture.
    del graph
    torch.cuda.synchronize()
    return statistics.mean(samples), statistics.pstdev(samples)


def install_cupy_torch_allocator() -> bool:
    """Route cupy allocations through torch's caching allocator.

    cupy (SpikingJelly's fused LIF backend) uses its own memory pool, which calls
    cudaMalloc on a pool miss -- illegal during CUDA-graph capture
    (cudaErrorStreamCaptureUnsupported). torch's caching allocator IS capture-aware
    (allocations during capture come from the graph's private pool), so bridging
    cupy onto it makes the cupy LIF kernel capturable. Global + idempotent."""
    try:
        import cupy
    except Exception:
        return False

    def _malloc(size, device_id):
        return int(torch.cuda.caching_allocator_alloc(size))

    def _free(ptr, device_id):
        torch.cuda.caching_allocator_delete(ptr)

    cupy.cuda.set_allocator(cupy.cuda.PythonFunctionAllocator(_malloc, _free).malloc)
    return True


def cupy_stream_ctx():
    """Bind cupy's current stream to torch's current stream, so cupy kernels
    (SpikingJelly's fused LIF) launch on the same stream torch is using -- and,
    critically, on the CUDA-graph capture stream during capture. No-op if cupy is
    unavailable. Must be entered at call time since the current stream changes."""
    try:
        import cupy

        return cupy.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream)
    except Exception:
        import contextlib

        return contextlib.nullcontext()


def lock_gpu_clock(clock_mhz: Optional[int]) -> None:
    """Best-effort clock lock. Changing clocks needs root; when the process is
    not root (the common case), this warns and continues rather than aborting --
    an external `sudo nvidia-smi -lgc` may already have locked the clock for the
    whole run, and the requested value is still recorded in the env metadata."""
    if clock_mhz is None:
        return
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        print("[warn] nvidia-smi not found; skipping in-process clock lock")
        return
    proc = subprocess.run(
        [nvidia_smi, "-lgc", f"{clock_mhz},{clock_mhz}"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    if proc.returncode == 0:
        atexit.register(lambda: subprocess.run([nvidia_smi, "-rgc"], check=False))
        print(f"[clock] locked GPU graphics clock to {clock_mhz} MHz")
    else:
        print(f"[warn] in-process clock lock failed (needs root?); continuing -- "
              f"rely on an external `sudo nvidia-smi -lgc {clock_mhz},{clock_mhz}`. "
              f"nvidia-smi said: {proc.stdout.strip()}")


def environment_metadata(args) -> Dict:
    gpu = torch.cuda.get_device_properties(0)
    nvidia_smi = shutil.which("nvidia-smi")
    clocks = ""
    if nvidia_smi:
        proc = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=clocks.current.graphics,clocks.applications.graphics,"
                "clocks.max.graphics,persistence_mode",
                "--format=csv,noheader",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        clocks = proc.stdout.strip()
    return {
        "command": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": gpu.name,
        "gpu_compute_capability": f"{gpu.major}.{gpu.minor}",
        "clock_state": clocks,
        "clock_lock_requested": args.lock_gpu_clock_mhz is not None,
        "requested_gpu_clock_mhz": args.lock_gpu_clock_mhz,
        "dtype": args.dtype,
        "batch": args.batch,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "rtol": args.rtol,
        "bn_handling": "inference BatchNorm folded into conv via torch.nn.utils.fuse_conv_bn_eval on both sides",
        "lif_params": {"tau": LIF_TAU, "v_threshold": LIF_V_THRESHOLD, "v_reset": LIF_V_RESET},
        "linear_tokens": args.linear_tokens,
        "linear_N": args.batch * args.linear_tokens,
        "linear_shape": f"{args.linear_in}->{args.linear_out}",
        "convlstm_ref": args.convlstm_ref,
        "use_cuda_graph": args.use_cuda_graph,
        "extended_T": os.environ.get("KAIROS_ALLOW_EXTENDED_T", "") == "1",
        "lif_state_atol": args.lif_state_atol,
        "per_pattern_dtype": {
            "conv_bn_lif": args.dtype,
            "linear_lif": args.dtype,
            "selective_scan": "fp32 (numerical stability)",
            "convlstm": "fp32 (kernel libdevice.tanh + fp32 workload)",
        },
        "size_provenance": "linear/conv shapes from the end-to-end models "
                           "(KairosSpikeTransformer dim=256 mlp_ratio=4 seq=256; SNN-ResNet18); "
                           "batch adjusted to the experiment's fixed batch",
        "tf32": {
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "note": "both Kairos and opponents use TF32 tensor cores for fp32 matmul/conv",
        },
    }


# ---------------------------------------------------------------------------
# Correctness helpers.
# ---------------------------------------------------------------------------
def close(a: torch.Tensor, b: torch.Tensor, rtol: float, atol: float):
    """allclose gate plus reported max-abs and near-zero-stable max-rel error.

    A pure relative metric explodes on state elements that legitimately pass
    through ~0 (e.g. a reset membrane), so the pass/fail decision uses
    torch.allclose (|a-b| <= atol + rtol*|b|); the relative number reported is
    computed only where |b| exceeds atol, purely for diagnostics.
    """
    a = a.float()
    b = b.float()
    ok = bool(torch.allclose(a, b, rtol=rtol, atol=atol))
    max_abs = float((a - b).abs().max().item())
    denom = b.abs()
    mask = denom > atol
    if bool(mask.any()):
        max_rel = float(((a - b).abs()[mask] / denom[mask]).max().item())
    else:
        max_rel = 0.0
    return ok, max_abs, max_rel


def frac_exceed(a: torch.Tensor, b: torch.Tensor, rtol: float, atol: float):
    """Fraction of elements violating allclose, and the max-abs error.

    For the LIF patterns the two kernels round the ``v >= v_threshold``
    comparison differently, so a handful of neurons within fp rounding of the
    threshold flip their spike decision; at those neurons the final membrane
    differs by ~v_threshold (one resets, one does not). That makes the *max*
    membrane error saturate at the threshold no matter how correct the bulk is,
    so the gate uses the *fraction* of disagreeing elements instead (the same
    boundary neurons that show up in the spike-mismatch fraction).
    """
    a = a.float()
    b = b.float()
    bad = (a - b).abs() > (atol + rtol * b.abs())
    return float(bad.float().mean().item()), float((a - b).abs().max().item())


def spike_mismatch_fraction(a: torch.Tensor, b: torch.Tensor) -> float:
    a = (a.float() > 0.5)
    b = (b.float() > 0.5)
    return float((a != b).float().mean().item())


@dataclass
class Record:
    pattern: str
    T: int
    batch: int
    mode: str = "eager"  # "eager" (per-call) or "cudagraph" (launch overhead removed)
    kairos_ms_mean: float = float("nan")
    kairos_ms_std: float = float("nan")
    ref_ms_mean: float = float("nan")
    ref_ms_std: float = float("nan")
    pct_of_ref: float = float("nan")
    correctness_pass: bool = False
    detail: str = ""

    def row(self) -> Dict:
        d = asdict(self)
        return d


CSV_COLUMNS = [
    "pattern",
    "T",
    "batch",
    "mode",
    "kairos_ms_mean",
    "kairos_ms_std",
    "ref_ms_mean",
    "ref_ms_std",
    "pct_of_ref",
    "correctness_pass",
    "detail",
]


# ---------------------------------------------------------------------------
# Opponent availability probes.
# ---------------------------------------------------------------------------
def probe_spikingjelly() -> Tuple[bool, str, object]:
    try:
        from spikingjelly.activation_based import layer, neuron  # noqa: F401
        import spikingjelly

        return True, f"spikingjelly {getattr(spikingjelly, '__version__', '?')}", (layer, neuron)
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, f"spikingjelly import failed: {exc}", None


def probe_cupy() -> Tuple[bool, str]:
    try:
        import cupy

        return True, f"cupy {getattr(cupy, '__version__', '?')}"
    except Exception as exc:  # pragma: no cover
        return False, f"cupy import failed: {exc}"


def probe_mamba() -> Tuple[bool, str, object]:
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

        return True, "mamba_ssm.selective_scan_fn (CUDA)", selective_scan_fn
    except Exception as exc_cuda:
        # Fall back to the reference (pure-PyTorch) implementation of the same
        # math if the CUDA extension is unavailable; record which one is used.
        try:
            from mamba_ssm.ops.selective_scan_interface import selective_scan_ref

            return True, f"mamba_ssm.selective_scan_ref (PyTorch fallback; CUDA import failed: {exc_cuda})", selective_scan_ref
        except Exception as exc_ref:
            return False, f"mamba_ssm import failed: {exc_cuda}; ref fallback failed: {exc_ref}", None


# ===========================================================================
# Pattern 1: conv + bn + lif  vs  SpikingJelly
# ===========================================================================
def build_conv_bn_lif(args, sj_modules, backend: str):
    layer, neuron = sj_modules
    device = args.device
    dtype = args.torch_dtype
    B, C, H, W = args.batch, args.conv_channels, args.conv_hw, args.conv_hw

    torch.manual_seed(args.seed)
    conv = nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1, bias=False)
    bn = nn.BatchNorm2d(C)
    # Give BN non-trivial running stats so the fusion is meaningful.
    bn.running_mean.normal_(0, 0.1)
    bn.running_var.uniform_(0.5, 1.5)
    bn.weight.data.uniform_(0.5, 1.5)
    bn.bias.data.normal_(0, 0.1)
    conv_bn = nn.Sequential(conv, bn).to(device=device, dtype=dtype).eval()
    fused_conv = torch.nn.utils.fuse_conv_bn_eval(conv_bn[0], conv_bn[1])
    # weight_gain lifts the synaptic drive so the LIF actually spikes (~20-30%
    # firing); both implementations share the identical scaled weight/bias.
    weight = (fused_conv.weight.detach() * args.weight_gain).contiguous()
    bias = (fused_conv.bias.detach() * args.weight_gain).contiguous()

    def make_case(T: int):
        torch.manual_seed(args.seed + T)
        x = (torch.randn(T, B, C, H, W, device=device, dtype=dtype) * args.input_scale).contiguous()
        v_init = torch.zeros(B, C, H, W, device=device, dtype=dtype)

        def kairos_fn():
            res = run_triton_fused_temporal_conv_lif_state(
                [x[t].contiguous() for t in range(T)],
                weight,
                bias,
                v_init,
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
                groups=1,
                v_threshold=LIF_V_THRESHOLD,
                v_reset=LIF_V_RESET,
                tau=LIF_TAU,
                detach_reset=LIF_DETACH_RESET,
                use_autotune=True,
            )
            return res.spikes, res.v_next

        # SpikingJelly opponent: SeqToANNContainer(conv) [cuDNN, time-folded]
        # + multi-step fused LIF (cupy backend). Same fused conv weights.
        from spikingjelly.activation_based import functional

        sj_conv = layer.SeqToANNContainer(
            nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1, bias=True)
        ).to(device=device, dtype=dtype)
        sj_conv[0].weight.data.copy_(weight)
        sj_conv[0].bias.data.copy_(bias)
        sj_lif = neuron.LIFNode(
            tau=LIF_TAU,
            decay_input=True,
            v_threshold=LIF_V_THRESHOLD,
            v_reset=LIF_V_RESET,
            step_mode="m",
            backend=backend,
        ).to(device=device, dtype=dtype)
        # Reset once here (not inside the timed loop): the neuron starts at v=0
        # for the first call, which is the correctness-gate call. Timing then
        # measures pure forward, symmetric with Kairos which reuses a cached
        # zero v_init and likewise pays no per-call reset. Compute is identical
        # whether v starts at 0 or accumulates, so excluding reset is fair.
        functional.reset_net(sj_lif)

        def ref_fn():
            with cupy_stream_ctx():
                gates = sj_conv(x)
                spikes = sj_lif(gates)
            return spikes, sj_lif.v

        def check_fn(kout, rout):
            ks, kv = kout
            rs, rv = rout
            mism = spike_mismatch_fraction(ks, rs)
            fire = float((rs.float() > 0.5).float().mean().item())
            v_bad, v_abs = frac_exceed(kv, rv, args.rtol, args.lif_state_atol)
            ok = mism <= args.spike_tol and v_bad <= args.state_tol
            return ok, f"fire={fire:.3f} spike_mismatch={mism:.2e} membrane_badfrac={v_bad:.2e} maxabs={v_abs:.2e}"

        return kairos_fn, ref_fn, check_fn

    return make_case


# ===========================================================================
# Pattern 2: linear + lif  vs  SpikingJelly
# ===========================================================================
def build_linear_lif(args, sj_modules, backend: str):
    layer, neuron = sj_modules
    device = args.device
    dtype = args.torch_dtype
    # A spiking-transformer/MLP linear layer processes N = batch * tokens
    # independent feature vectors. Keeping N = batch (=8) makes the GEMM tiny and
    # launch/framework-overhead bound; using a realistic token count puts it in a
    # compute-bound regime so the comparison reflects kernel efficiency + the
    # memory-traffic saved by fusion, not per-call overhead.
    N = args.batch * args.linear_tokens
    in_features = args.linear_in
    out_features = args.linear_out

    torch.manual_seed(args.seed)
    lin = nn.Linear(in_features, out_features, bias=True).to(device=device, dtype=dtype)
    # weight_gain lifts synaptic drive into a spiking regime (shared by both).
    weight = (lin.weight.detach() * args.weight_gain).contiguous()
    bias = (lin.bias.detach() * args.weight_gain).contiguous()

    def make_case(T: int):
        torch.manual_seed(args.seed + T)
        x = (torch.randn(T, N, in_features, device=device, dtype=dtype) * args.input_scale).contiguous()
        v_init = torch.zeros(N, out_features, device=device, dtype=dtype)

        def kairos_fn():
            spikes, v_last, _ = run_fused_temporal_linear_lif_state_kernel(
                x,
                weight,
                bias,
                v_init,
                v_threshold=LIF_V_THRESHOLD,
                v_reset=LIF_V_RESET,
                tau=LIF_TAU,
                detach_reset=LIF_DETACH_RESET,
                use_autotune=True,
            )
            return spikes, v_last

        from spikingjelly.activation_based import functional

        sj_lin = layer.SeqToANNContainer(nn.Linear(in_features, out_features, bias=True)).to(
            device=device, dtype=dtype
        )
        sj_lin[0].weight.data.copy_(weight)
        sj_lin[0].bias.data.copy_(bias)
        sj_lif = neuron.LIFNode(
            tau=LIF_TAU,
            decay_input=True,
            v_threshold=LIF_V_THRESHOLD,
            v_reset=LIF_V_RESET,
            step_mode="m",
            backend=backend,
        ).to(device=device, dtype=dtype)
        # Reset once (see conv+bn+lif note): correctness call starts at v=0,
        # timing measures pure forward, symmetric with Kairos.
        functional.reset_net(sj_lif)

        def ref_fn():
            with cupy_stream_ctx():
                gates = sj_lin(x)
                spikes = sj_lif(gates)
            return spikes, sj_lif.v

        def check_fn(kout, rout):
            ks, kv = kout
            rs, rv = rout
            mism = spike_mismatch_fraction(ks, rs)
            fire = float((rs.float() > 0.5).float().mean().item())
            v_bad, v_abs = frac_exceed(kv, rv, args.rtol, args.lif_state_atol)
            ok = mism <= args.spike_tol and v_bad <= args.state_tol
            return ok, f"fire={fire:.3f} spike_mismatch={mism:.2e} membrane_badfrac={v_bad:.2e} maxabs={v_abs:.2e}"

        return kairos_fn, ref_fn, check_fn

    return make_case


# ===========================================================================
# Pattern 3: selective scan  vs  official Mamba selective_scan_fn
# ===========================================================================
def build_selective_scan(args, selective_scan_fn):
    device = args.device
    # Selective scan accumulates state in fp32 (numerically sensitive exp); run
    # the whole pattern in fp32 for a clean correctness gate regardless of the
    # SNN-path dtype.
    dtype = torch.float32
    B = args.batch
    d_inner = args.d_inner
    d_state = args.d_state

    def make_case(T: int):
        torch.manual_seed(args.seed + T)
        x = torch.randn(T, B, d_inner, device=device, dtype=dtype)
        # dt strictly positive; A strictly negative -> stable exp(dt*A) decay.
        dt = torch.rand(T, B, d_inner, device=device, dtype=dtype) * 0.1 + 0.01
        b = torch.randn(T, B, d_state, device=device, dtype=dtype)
        c = torch.randn(T, B, d_state, device=device, dtype=dtype)
        A = -torch.rand(d_inner, d_state, device=device, dtype=dtype) - 0.1
        D = torch.randn(d_inner, device=device, dtype=dtype)
        h_init = torch.zeros(B, d_inner, d_state, device=device, dtype=dtype)

        def kairos_fn():
            y, h_final = run_fused_temporal_selective_scan_kernel(
                x, dt, b, c, A, D, h_init
            )
            return y, h_final

        # Official kernel layout is [B, d_inner, L] for u/delta and
        # [B, d_state, L] for B/C. Transpose so both compute the same function.
        u_m = x.permute(1, 2, 0).contiguous()       # [B, d_inner, T]
        delta_m = dt.permute(1, 2, 0).contiguous()  # [B, d_inner, T]
        B_m = b.permute(1, 2, 0).contiguous()       # [B, d_state, T]
        C_m = c.permute(1, 2, 0).contiguous()       # [B, d_state, T]

        def ref_fn():
            out = selective_scan_fn(
                u_m,
                delta_m,
                A,
                B_m,
                C_m,
                D=D,
                z=None,
                delta_bias=None,
                delta_softplus=False,
                return_last_state=True,
            )
            y_m, last_state = out  # y_m [B,d_inner,T], last_state [B,d_inner,d_state]
            y = y_m.permute(2, 0, 1).contiguous()  # -> [T,B,d_inner]
            return y, last_state

        def check_fn(kout, rout):
            ky, kh = kout
            ry, rh = rout
            ok_y, y_abs, y_rel = close(ky, ry, args.rtol, args.atol)
            ok_h, h_abs, h_rel = close(kh, rh, args.rtol, args.atol)
            ok = ok_y and ok_h
            return ok, f"y_maxabs={y_abs:.2e} rel={y_rel:.2e} h_maxabs={h_abs:.2e} rel={h_rel:.2e}"

        return kairos_fn, ref_fn, check_fn

    return make_case


# ===========================================================================
# Pattern 4: ConvLSTM cell  vs  standard PyTorch pointwise gating
# ===========================================================================
def build_convlstm(args):
    device = args.device
    # ConvLSTM runs fp32: the Kairos cell kernel uses libdevice.tanh (fp32/fp64
    # only), and the standard ConvLSTM workload (Shi 2015, precipitation
    # nowcasting) is fp32 -- unrelated to the SNN fp16 patterns.
    dtype = torch.float32
    B, C, H, W = args.batch, args.convlstm_channels, args.convlstm_hw, args.convlstm_hw

    # The convolution that produces gates_sum is identical on both sides (cuDNN);
    # it is computed once, OUTSIDE the timed region, so the measurement isolates
    # the two implementations of the gating cell over T steps.
    torch.manual_seed(args.seed)
    gate_conv = nn.Conv2d(C, 4 * C, kernel_size=3, stride=1, padding=1, bias=True).to(
        device=device, dtype=dtype
    )

    def make_case(T: int):
        torch.manual_seed(args.seed + T)
        # Pre-compute T gates_sum tensors via cuDNN conv (shared by both sides).
        with torch.no_grad():
            inputs = [
                torch.randn(B, C, H, W, device=device, dtype=dtype) for _ in range(T)
            ]
            gates_list = [gate_conv(inp).contiguous() for inp in inputs]
        c0 = torch.zeros(B, C, H, W, device=device, dtype=dtype)

        def kairos_fn():
            c_prev = c0
            h_last = None
            for t in range(T):
                h_last, c_prev = run_fused_convlstm_cell_kernel(gates_list[t], c_prev)
            return h_last, c_prev

        # Standard ConvLSTM gating cell (Shi 2015): pointwise sigmoid/tanh + cell
        # update. cuDNN has no primitive for this spatial gating, so the field
        # implements it as pointwise ops. Comparing a fused kernel to *unfused*
        # eager PyTorch would be a strawman, so the opponent is the same gating
        # under torch.compile (Inductor), which fuses the pointwise ops into a
        # single kernel -- fused-vs-fused, the defensible baseline. --convlstm-ref
        # eager falls back to unfused eager for reference. Chunk order i,f,g,o
        # matches the fused kernel's memory layout.
        def gate_step(gates, c_prev):
            i, f, g, o = torch.chunk(gates, 4, dim=1)
            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            g = torch.tanh(g)
            o = torch.sigmoid(o)
            c = f * c_prev + i * g
            h = o * torch.tanh(c)
            return h, c

        step_fn = torch.compile(gate_step) if args.convlstm_ref == "compile" else gate_step

        def ref_fn():
            c_prev = c0
            h_last = None
            for t in range(T):
                h_last, c_prev = step_fn(gates_list[t], c_prev)
            return h_last, c_prev

        def check_fn(kout, rout):
            kh, kc = kout
            rh, rc = rout
            ok_h, h_abs, h_rel = close(kh, rh, args.rtol, args.atol)
            ok_c, c_abs, c_rel = close(kc, rc, args.rtol, args.atol)
            ok = ok_h and ok_c
            return ok, f"h_maxabs={h_abs:.2e} rel={h_rel:.2e} c_maxabs={c_abs:.2e} rel={c_rel:.2e}"

        return kairos_fn, ref_fn, check_fn

    return make_case


# ===========================================================================
# Driver
# ===========================================================================
def _fill_timing(rec: Record, timer, kairos_fn, ref_fn, args):
    k_mean, k_std = timer(kairos_fn, args.warmup, args.repeat)
    r_mean, r_std = timer(ref_fn, args.warmup, args.repeat)
    rec.kairos_ms_mean = k_mean
    rec.kairos_ms_std = k_std
    rec.ref_ms_mean = r_mean
    rec.ref_ms_std = r_std
    rec.pct_of_ref = (r_mean / k_mean * 100.0) if k_mean > 0 else float("nan")
    return k_mean, k_std, r_mean, r_std


def run_pattern(name: str, make_case, args, notes: List[str], supported_t=None) -> List[Record]:
    records: List[Record] = []
    for T in args.t_values:
        rec = Record(pattern=name, T=T, batch=args.batch, mode="eager")
        if supported_t is not None and T not in supported_t:
            rec.detail = f"skipped: kernel supports only T in {sorted(supported_t)}"
            print(f"[{name}] T={T:<2d} SKIP  ({rec.detail})")
            records.append(rec)
            continue
        try:
            kairos_fn, ref_fn, check_fn = make_case(T)
            kout = kairos_fn()
            rout = ref_fn()
            torch.cuda.synchronize()
            ok, detail = check_fn(kout, rout)
            rec.correctness_pass = bool(ok)
            rec.detail = detail
            if not ok:
                print(f"[{name}] T={T:<2d} CORRECTNESS FAIL  {detail}  -> not timed")
                records.append(rec)
                continue
            k_mean, k_std, r_mean, r_std = _fill_timing(rec, time_cuda, kairos_fn, ref_fn, args)
            print(
                f"[{name}] T={T:<2d} eager      pass  kairos={k_mean:.4f}+/-{k_std:.4f}ms "
                f"ref={r_mean:.4f}+/-{r_std:.4f}ms  pct_of_ref={rec.pct_of_ref:.1f}%  ({detail})"
            )
        except Exception as exc:
            rec.detail = f"error: {exc}"
            print(f"[{name}] T={T:<2d} ERROR: {exc}")
            traceback.print_exc()
        records.append(rec)

        # Optional CUDA-graph timing (both sides), removing launch overhead.
        if args.use_cuda_graph and rec.correctness_pass:
            grec = Record(pattern=name, T=T, batch=args.batch, mode="cudagraph",
                          correctness_pass=True, detail=rec.detail)
            try:
                # Fresh callables so cudagraph captures against clean state.
                kfn2, rfn2, _ = make_case(T)
                k_mean, k_std, r_mean, r_std = _fill_timing(grec, time_cuda_graph, kfn2, rfn2, args)
                print(
                    f"[{name}] T={T:<2d} cudagraph  pass  kairos={k_mean:.4f}+/-{k_std:.4f}ms "
                    f"ref={r_mean:.4f}+/-{r_std:.4f}ms  pct_of_ref={grec.pct_of_ref:.1f}%"
                )
            except Exception as exc:
                grec.correctness_pass = False
                grec.detail = f"cudagraph capture failed: {exc}"
                print(f"[{name}] T={T:<2d} cudagraph  FAILED: {exc}")
                notes.append(f"{name} T={T} cudagraph: {exc}")
            records.append(grec)
    return records


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("fp32", "fp16"), default="fp32",
                   help="dtype for conv/linear LIF patterns (fp32 => TF32 tensor cores). "
                        "selective_scan and convlstm always run fp32.")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--t-values", nargs="+", type=int, default=list(SUPPORTED_T))
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repeat", type=int, default=100)
    p.add_argument("--use-cuda-graph", action="store_true",
                   help="also emit CUDA-graph-timed rows (mode=cudagraph) alongside eager, "
                        "capturing BOTH sides symmetrically to remove CPU launch/framework "
                        "overhead. SpikingJelly's cupy LIF is bound to torch's capture stream; "
                        "if a pattern's capture fails it is recorded, not silently faked.")
    p.add_argument("--rtol", type=float, default=1e-3)
    p.add_argument("--atol", type=float, default=1e-3,
                   help="absolute tolerance floor for allclose (stabilises near-zero state)")
    p.add_argument("--spike-tol", type=float, default=2e-3,
                   help="max fraction of spikes allowed to differ (near-threshold fp rounding)")
    p.add_argument("--state-tol", type=float, default=2e-3,
                   help="max fraction of membrane elements allowed to differ (boundary flips)")
    p.add_argument("--lif-state-atol", type=float, default=None,
                   help="membrane abs tolerance for LIF patterns. Kairos runs the conv/linear "
                        "matmul on tensor cores (TF32 for fp32, fp16 accumulation for fp16), so "
                        "the membrane agrees only to that floor while spike trains match near-exactly. "
                        "Default: 5e-3 for fp32, 2e-2 for fp16 (larger reduction error).")
    p.add_argument("--input-scale", type=float, default=3.0,
                   help="std of SNN input drive; with --weight-gain sets ~20-30%% firing")
    p.add_argument("--weight-gain", type=float, default=2.0,
                   help="multiplier on conv/linear weights to reach a spiking regime")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--lock-gpu-clock-mhz", type=int, default=None)
    p.add_argument("--sj-backend", choices=("cupy", "torch"), default="cupy",
                   help="SpikingJelly LIF backend for the opponent (cupy = hand-written fused kernel)")
    p.add_argument(
        "--patterns",
        nargs="+",
        default=["conv_bn_lif", "linear_lif", "selective_scan", "convlstm"],
        choices=["conv_bn_lif", "linear_lif", "selective_scan", "convlstm"],
    )
    # Problem-size knobs (real-model-ish defaults).
    p.add_argument("--conv-channels", type=int, default=128)
    p.add_argument("--conv-hw", type=int, default=32)
    # Sizes taken from the end-to-end model (KairosSpikeTransformer: dim=256,
    # mlp_ratio=4, seq=256; see dev_tests/bench_batched_linear_lif_autotune_cost.py).
    # Default is the FFN expansion layer fc1_up (256->1024), the compute-dominant
    # linear in a spiking transformer; N = batch*tokens with the model's seq=256.
    # There is deliberately NO 1024->1024 square layer here -- no model layer has
    # that shape, and it is the one regime where a Triton-fused GEMM loses to
    # cuBLAS, so benchmarking it would misrepresent the end-to-end workload.
    p.add_argument("--linear-in", type=int, default=256,
                   help="in_features of the representative linear layer (fc1_up=256)")
    p.add_argument("--linear-out", type=int, default=1024,
                   help="out_features of the representative linear layer (fc1_up=1024)")
    p.add_argument("--linear-tokens", type=int, default=256,
                   help="tokens (seq) per batch element; N=batch*tokens, from the model's seq=256")
    p.add_argument("--convlstm-ref", choices=("compile", "eager"), default="compile",
                   help="opponent for the gating cell: torch.compile (Inductor-fused, the "
                        "defensible baseline) or unfused eager PyTorch (reference only)")
    p.add_argument("--d-inner", type=int, default=1536,
                   help="Mamba-130M d_inner = 2*d_model = 2*768 = 1536 (real config)")
    p.add_argument("--d-state", type=int, default=16)
    p.add_argument("--convlstm-channels", type=int, default=128)
    p.add_argument("--convlstm-hw", type=int, default=32)
    p.add_argument("--out-csv", default=str(PROJECT_ROOT / "eval_handwritten.csv"))
    p.add_argument("--out-env", default=str(PROJECT_ROOT / "eval_handwritten_env.json"))
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    for t in args.t_values:
        if t not in ALLOWED_T:
            raise ValueError(f"T={t} not in allowed set {ALLOWED_T}")
    args.torch_dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    if args.lif_state_atol is None:
        # fp16 conv/linear matmul accumulates ~sqrt(K)*eps error over the K-wide
        # reduction, larger than TF32's; use a looser membrane floor for fp16.
        args.lif_state_atol = 2e-2 if args.dtype == "fp16" else 5e-3

    lock_gpu_clock(args.lock_gpu_clock_mhz)
    env = environment_metadata(args)

    notes: List[str] = []
    all_records: List[Record] = []

    # Opponent availability.
    sj_ok, sj_msg, sj_modules = probe_spikingjelly()
    cupy_ok, cupy_msg = probe_cupy()
    mamba_ok, mamba_msg, scan_fn = probe_mamba()
    notes.extend([sj_msg, cupy_msg, mamba_msg])
    print("=== opponent availability ===")
    print("  spikingjelly:", sj_msg)
    print("  cupy        :", cupy_msg)
    print("  mamba-ssm   :", mamba_msg)

    sj_backend = args.sj_backend
    if sj_backend == "cupy" and not cupy_ok:
        sj_backend = "torch"
        notes.append("cupy unavailable -> SpikingJelly LIF ran on backend='torch' (not the cupy fused kernel)")
        print("  [warn] cupy unavailable; falling back SpikingJelly LIF backend to 'torch'")

    if args.use_cuda_graph and cupy_ok:
        if install_cupy_torch_allocator():
            notes.append("cupy allocations bridged onto torch caching allocator for CUDA-graph capture")
            print("  [cudagraph] cupy -> torch allocator bridge installed")

    # Pattern 1: conv+bn+lif
    if "conv_bn_lif" in args.patterns:
        if sj_ok:
            print(f"\n--- conv+bn+lif vs SpikingJelly (LIF backend={sj_backend}) ---")
            all_records += run_pattern(
                "conv_bn_lif", build_conv_bn_lif(args, sj_modules, sj_backend), args, notes,
                supported_t=CONV_SUPPORTED_T,
            )
        else:
            print("\n--- conv+bn+lif SKIPPED (spikingjelly unavailable) ---")
            notes.append("conv_bn_lif skipped: spikingjelly unavailable")

    # Pattern 2: linear+lif
    if "linear_lif" in args.patterns:
        if sj_ok:
            print(f"\n--- linear+lif vs SpikingJelly (LIF backend={sj_backend}) ---")
            all_records += run_pattern(
                "linear_lif", build_linear_lif(args, sj_modules, sj_backend), args, notes,
                supported_t=LINEAR_SUPPORTED_T,
            )
        else:
            print("\n--- linear+lif SKIPPED (spikingjelly unavailable) ---")
            notes.append("linear_lif skipped: spikingjelly unavailable")

    # Pattern 3: selective scan
    if "selective_scan" in args.patterns:
        if mamba_ok:
            print("\n--- selective_scan vs mamba-ssm ---")
            all_records += run_pattern(
                "selective_scan", build_selective_scan(args, scan_fn), args, notes,
                supported_t=SCAN_SUPPORTED_T,
            )
        else:
            print("\n--- selective_scan SKIPPED (mamba-ssm unavailable) ---")
            notes.append("selective_scan skipped: mamba-ssm unavailable")

    # Pattern 4: convlstm
    if "convlstm" in args.patterns:
        print("\n--- convlstm vs standard PyTorch pointwise gating ---")
        all_records += run_pattern("convlstm", build_convlstm(args), args, notes)

    # Write CSV.
    out_csv = Path(args.out_csv)
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for rec in all_records:
            writer.writerow({k: rec.row()[k] for k in CSV_COLUMNS})
    print(f"\nwrote {out_csv}")

    # Write environment + notes sidecar.
    env["opponent_notes"] = notes
    env["sj_backend_used"] = sj_backend
    Path(args.out_env).write_text(json.dumps(env, indent=2, default=str))
    print(f"wrote {args.out_env}")

    # Summary.
    print("\n=== summary (pct_of_ref = ref_ms / kairos_ms) ===")
    for rec in all_records:
        status = "PASS" if rec.correctness_pass else "FAIL/SKIP"
        pct = f"{rec.pct_of_ref:6.1f}%" if rec.pct_of_ref == rec.pct_of_ref else "   n/a"
        print(f"  {rec.pattern:<16s} T={rec.T:<2d} {rec.mode:<9s} {status:<9s} {pct}  {rec.detail}")


if __name__ == "__main__":
    main()
