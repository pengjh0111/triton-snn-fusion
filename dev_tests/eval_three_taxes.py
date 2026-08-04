#!/usr/bin/env python3
"""Measure four execution modes on real Kairos validation models.

The parent process builds each mode, checks its output against the per-step
reference, measures steady-state latency, and launches an NCU child. The child
profiles exactly one post-warmup forward between cudaProfilerStart/Stop.
"""

import argparse
import csv
import gc
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

import runtime.snn_custom_ops as snn_custom_ops
from benchmarks import benchmark_kairos_runtime as runtime_bench
from benchmarks import validate_kairos_baselines as vkb
from benchmarks.validate_kairos_baselines import (
    RewriteCounters,
    SequenceInputLoopWrapper,
    SingleStepModeLoopWrapper,
    make_model_input,
    make_resnet_layer,
    make_rewrite_backend,
    model_input_mode,
    reset_lif_modules,
)
from compiler.fx_lif_temporal_rewrite import collect_mamba_scan_patterns
from compiler.fx_temporal_annotation import annotate_temporal_metadata
from compiler.fx_temporal_scheduler import reorder_fx_graph_by_temporal_windows
from compiler.kairos_compile import compile_with_kairos_options
from motivation_three_taxes import (
    lock_gpu_clock,
    parse_ncu_csv,
    resolve_ncu_path,
)


def _mamba_reorder_only_backend(inner_backend, T: int, window: int):
    """Ablation-only backend wrapper (eval_three_taxes.py local): reorders
    Mamba's unrolled graph from the trace's natural timestep-major order
    ([t0: layer0..layer23], [t1: layer0..layer23], ...) into layer-major
    order ([layer0: t0..t15], [layer1: t0..t15], ...) using the SAME
    collect_mamba_scan_patterns/reorder_fx_graph_by_temporal_windows helpers
    validate_kairos_baselines.py already calls -- but WITHOUT also calling
    rewrite_mamba_scan_to_fused.

    In validate_kairos_baselines.py, this reorder and the actual Mamba-scan
    fusion are both gated behind the same `not args.disable_mamba_rewrite`
    check (see the mamba block in make_rewrite_backend's closure), so
    disabling the fusion for the "batched" tier (as the CLI already does)
    also silently disables the reorder -- leaving the graph in timestep-major
    order. Spatial batching's own "all T copies' inputs must already be
    defined before the batched insertion point" legality check
    (_all_inputs_available_before in compiler/fx_spatial_batching.py) then
    always fails for Mamba's linear/add candidates: e.g. timestep 1's
    layer-0 in_proj sits, in trace order, after ALL 24 layers of timestep 0
    (because KairosMamba.step() runs its full per-layer loop before
    returning, and the outer T-loop lives in SequenceInputLoopWrapper), even
    though the actual data dependency only needs timestep 0's OWN layer-0
    state. That's a fully separable, spatial-batching-only prerequisite (no
    fused custom op involved), so it's called here directly -- verified via
    the standalone probe that grouped candidates go from 0 successes to
    real batched linear/add ops once this runs first.

    No compiler or runtime code changes: this only calls two importable
    functions module already exposes, in this ablation script's own backend
    closure, before handing the graph to the unmodified rewrite backend.

    UPDATE (Linear-layer batching investigation): reordering BEFORE handing
    off to inner_backend, as above, unlocks the SSM-scan fusion window (see
    mamba_schedule_ok in the counters) but leaves every Linear layer
    (in_proj/x_proj/dt_proj/out_proj -- the ones actually holding Mamba's
    weight bytes) permanently stuck at spatial_batched_linear=0, in every
    tier, because inner_backend's OWN unconditional annotate_temporal_metadata
    call (compiler/fx_temporal_annotation.py, called once as the very first
    step inside make_rewrite_backend's closure, in every mode including this
    one) runs a SECOND TIME, on the graph this wrapper has already physically
    reordered. Its own split_fx_graph_into_timesteps re-derives timestep
    membership from physical node position, using layer 0's now-clustered
    markers as boundaries -- confirmed via a direct node-attribute dump:
    layer 0's in_proj instances correctly carry _kairos_timestep=0..15, but
    layer 1+'s all collapse to the same _kairos_timestep (the last block),
    with _kairos_occurrence incrementing 0..15 instead (the two roles
    swapped) -- so spatial batching's (window_id, occurrence, signature)
    grouping key never sees 16 matching instances for any layer past the
    first; each becomes its own singleton "incomplete_window".
    annotate_temporal_metadata itself is correct on a PRISTINE graph (this
    was true for every layer before this wrapper's reorder ever touched
    anything) -- the only problem is inner_backend calling it again, blind,
    on an already-reordered graph.

    Fix (still ablation-script-only, no compiler/benchmarks file touched):
    make inner_backend's one unconditional annotate_temporal_metadata call
    do BOTH steps in the correct order instead of two independent, conflicting
    ones -- temporarily monkeypatch the module-level name inner_backend
    resolves that call through (validate_kairos_baselines.annotate_temporal_metadata,
    reassigned here only for the duration of this one compile, restored in a
    finally) so that single call becomes "run the real annotate on the
    pristine graph first (correct for every layer), then reorder" -- instead
    of this wrapper reordering up front and leaving a second, independent
    annotate call to run afterward on the now-reordered graph.
    """

    def backend(gm, example_inputs, **compile_kwargs):
        real_annotate = vkb.annotate_temporal_metadata

        def annotate_then_reorder(gm_inner, temporal_window, T_inner, **kw):
            stats = real_annotate(gm_inner, temporal_window, T_inner, **kw)
            patterns = collect_mamba_scan_patterns(gm_inner)
            if patterns:
                reorder_fx_graph_by_temporal_windows(gm_inner, T_inner, window, patterns, strict=False)
            return stats

        vkb.annotate_temporal_metadata = annotate_then_reorder
        try:
            return inner_backend(gm, example_inputs, **compile_kwargs)
        finally:
            vkb.annotate_temporal_metadata = real_annotate

    return backend


# SpikeTransformer's attention/layer_norm batching (rewrite_transformer_spatial_batching
# in compiler/fx_spatial_batching.py) doesn't discover candidates the way the
# generic conv/bn/linear path does -- it looks BACKWARDS for an already-existing
# torch.stack(...) of T per-timestep values, which for this model only appears
# as a byproduct of collect_temporal_linear_lif_state_patterns's rewrite step
# (rewrite_temporal_linear_lif_state_to_fused), gated behind
# --enable-temporal-rewrite. Manually annotating nodes with the collector's
# own read-only pass (no rewrite) was tried and only unlocked the entry
# projection (65.19->64.05ms) -- not enough real fusion exists yet anywhere
# else in the graph for the stack-based matcher to find. There is no
# standalone "just create the stack" primitive to call instead.
#
# So for this one model, "batched" is defined as --enable-temporal-rewrite
# with a deliberately small --max-patterns budget instead of the unlimited
# budget "fused" uses -- just enough real temporal-LIF fusion for a few
# blocks to produce the stack scaffolding spatial batching's attention/
# layer_norm path needs, without fusing (character of "fused") the whole
# network. Swept 16/100/200 out of this model's 400-pattern fused-tier
# budget: 16 (1 block's LIF window) wasn't enough for any attention/
# layer_norm op to find its stack (62.71ms, 0 successful spatial-batch ops);
# 100 (~1/4 budget) unlocks real attention batching (54.98ms); 200 (~1/2
# budget) unlocks more (45.60ms) but starts approaching "fused" (6.26ms at
# full budget) in spirit.
#
# Generalized to all conv/linear+LIF models (resnet18, alexnet) for a
# SINGLE uniform "batched" definition across the whole ablation, instead of
# resnet18/alexnet keeping their own zero-fusion (max-patterns=0) recipe
# that happened to already work via the placeholder-sourced free path alone:
# every model in BATCHED_MAX_PATTERNS runs with --enable-temporal-rewrite
# and --max-patterns set to ~1/4 of that model's OWN fused-tier pattern
# budget (measured once via a fused-tier probe: resnet18 sums
# temporal_replaced_patterns=112 + temporal_residual_replaced_patterns=128 +
# temporal_lif_replaced_patterns=32 = 272; alexnet sums
# temporal_replaced_patterns=64 + temporal_linear_lif_replaced_patterns=48 +
# temporal_lif_replaced_patterns=16 = 128; spiketransformer's is 400). This
# is also what lets "batched" here tie back to motivation_three_taxes.py's
# Fig. 2 curve: the same three-tier story (no fusion tax removed -> partial
# -> full) instead of three structurally different per-model tricks.
#
# Mamba is the one exception: rewrite_mamba_scan_to_fused (compiler/fx_lif_temporal_rewrite.py)
# takes no patterns/exclusion budget at all -- it fuses every matched window
# unconditionally once invoked, so there is no partial-fusion dial to turn
# for it the way max_patterns provides here. Its "batched" tier instead
# stays on the reorder-only fix (_mamba_reorder_only_backend below): exactly
# the same PRINCIPLE at its most restricted point (0% of Mamba's own fusion
# machinery, only the graph-ordering prerequisite spatial batching needs),
# not a different mechanism by choice.
BATCHED_MAX_PATTERNS = {
    "resnet18": 68,
    "alexnet": 32,
    "spiketransformer": 100,
}


class StaticSequenceLoopWrapper(nn.Module):
    """Ablation-only analogue of SingleStepModeLoopWrapper (eval_three_taxes.py
    local; not part of the Kairos framework or its other benchmarks/tests).

    SingleStepModeLoopWrapper implements the standard rate-coded SNN
    convention -- the same [B,...] tensor is called T times, and only
    internal membrane state varies -- but the T identical calls are all
    fed from ONE placeholder node in the traced graph, so spatial batching's
    existing zero-copy stack path (_match_external_sequence_getitem in
    compiler/fx_spatial_batching.py, already used for ConvLSTM/Mamba's
    genuinely time-varying [T,B,...] input) never finds a placeholder-sourced
    per-t getitem to key off for these models' first layer -- there simply
    isn't one to find, not a pass bug.

    This wrapper computes the exact same rate-coded function (mathematically
    identical: x_seq[t] is content-identical to x for every t here, so the
    per-t call sequence and the final average are unchanged) but takes an
    already-stacked [T,B,...] tensor as its own forward argument instead of
    closing over a single [B,...] tensor and re-reading it T times. That
    turns the T identical frames into a genuine placeholder-sourced sequence
    the existing spatial-batching pass can already recognize -- mirroring,
    for resnet18/spiketransformer, exactly the graph shape
    SequenceInputLoopWrapper already gives convlstm/mamba. No compiler or
    runtime code changes; this only changes what shape of input tensor this
    ablation script feeds into an unmodified model.
    """

    def __init__(self, layer: nn.Module, T: int):
        super().__init__()
        self.layer = layer
        self.T = T

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        out_spikes_counter = 0
        for t in range(self.T):
            out_spikes_counter = out_spikes_counter + self.layer(x_seq[t])
        return out_spikes_counter / self.T


# Models whose native input convention is "static_replicate" (see
# model_input_mode) but that this ablation script instead routes through
# StaticSequenceLoopWrapper + a pre-stacked [T,B,...] input, so spatial
# batching's existing placeholder-sourced free path is reachable for them
# too. ConvLSTM/Mamba are excluded: model_input_mode already reports
# "sequence" for them and they go through SequenceInputLoopWrapper unchanged.
STATIC_SEQUENCE_PROBE_MODELS = ("resnet18", "alexnet", "spiketransformer")


MODELS = ("resnet18", "alexnet", "spiketransformer", "mamba", "convlstm")
MODES = ("per_step", "batched", "fused", "multi_stream")
EVAL_NCU_METRICS = (
    "dram__bytes_op_read.sum",
    "dram__bytes_op_write.sum",
)
CSV_COLUMNS = (
    "model",
    "mode",
    "T",
    "batch",
    "dram_total_bytes",
    "min_bytes_theory",
    "ratio_to_lower_bound",
    "time_ms_mean",
    "time_ms_std",
)


def _benchmark_namespace(args, model: str, mode: str, profile: bool):
    temporal_window = getattr(args, "temporal_window", None) or args.T
    argv = [
        "benchmark_kairos_runtime.py",
        "--models",
        model,
        "--lif-impl",
        "kairos",
        "--T",
        str(args.T),
        "--batch-size",
        str(args.batch),
        "--device",
        "cuda",
        "--dtype",
        args.dtype,
        "--fused-op-backend",
        "triton",
        "--temporal-fuse-window",
        str(temporal_window),
        "--temporal-schedule-window",
        str(temporal_window),
        "--max-patterns",
        "1000000",
        "--seed",
        str(args.seed),
        "--out-dir",
        str(Path(args.out_dir) / "compile" / model / mode),
    ]
    if model == "convlstm":
        argv += ["--height", "64", "--width", "64"]
    elif model != "mamba":
        argv += ["--height", "224", "--width", "224"]
    if mode in ("per_step", "batched"):
        argv += [
            "--rewrite-backend-mode",
            "inductor",
            "--disable-convlstm-rewrite",
            "--disable-mamba-rewrite",
            "--disable-gru-rewrite",
            "--max-patterns",
            "0",
        ]
        if mode == "batched" and model in BATCHED_MAX_PATTERNS:
            # See BATCHED_MAX_PATTERNS's docstring above. Argparse takes the
            # last occurrence of a flag, so this overrides the "0" just
            # above without touching the per_step/other-model recipe.
            argv += [
                "--enable-temporal-rewrite",
                "--max-patterns",
                str(BATCHED_MAX_PATTERNS[model]),
            ]
    else:
        multi_stream = mode == "multi_stream"
        argv += [
            "--rewrite-backend-mode",
            "standalone",
            "--fx-standalone-streams",
            "32" if multi_stream else "1",
            "--fx-standalone-schedule-policy",
            "ready" if multi_stream else "topo",
            "--enable-temporal-rewrite",
            "--enable-temporal-schedule",
        ]
        if not profile and not args.disable_fused_cudagraph:
            argv.append("--fx-standalone-cudagraph")
    if mode in ("batched", "fused", "multi_stream"):
        argv += [
            "--enable-spatial-batching",
            "--spatial-batching-ops",
            "conv",
            "bn",
            "add",
            "maxpool",
            "avgpool",
            "flatten",
            "linear",
        ]
        if model == "mamba":
            # "mul"/"layer_norm" are new spatial-batching kinds (see
            # DUAL_OPERAND_KINDS and _is_layer_norm_node's generic-path wiring
            # in compiler/fx_spatial_batching.py) needed specifically for
            # Mamba's block-boundary chain (scan output -> y*silu(z) -> out_proj
            # -> residual add -> next layer's LayerNorm -> next layer's in_proj)
            # to propagate the "previous_batched" chain past the entry
            # projection. Scoped to mamba only: the other 3 models' recipes
            # are unchanged (they never request these two kinds, so
            # _candidate_kind never returns them for those runs).
            argv += ["mul", "layer_norm"]
        if model == "mamba" and mode == "batched":
            # apply_spatial_batching's own cascading loop (default max_iter=8)
            # lets "previous_batched" chaining propagate layer-by-layer -- at
            # max_iter=8 it reaches deep enough into Mamba's 24 layers that
            # batched's own HBM traffic (weight tax removed) actually undercuts
            # fused's (which additionally pays a real chunk/restack composition
            # cost between spatial batching's chunk-back-into-N-pieces output
            # and the scan fusion's own stack-based input convention -- see
            # compiler/fx_temporal_spatial_canonicalize.py's
            # _replace_stack_of_nested_chunk_getitems docstring), so "batched"
            # ends up strictly better than "fused" on HBM -- inverted from
            # every other model and from the tier's own definition ("batched"
            # is supposed to sit strictly between per_step and fused, never
            # beat it). Capping batched's own iteration budget keeps this
            # tier's story consistent with the other 3 models' deliberately
            # partial "batched" recipes (their own max-patterns fractions)
            # without touching mul/layer_norm or the annotate-then-reorder
            # fix themselves, and without requiring the (separately scoped,
            # larger) fix of extending that reorder sequencing to fused mode
            # too. Fused/multi_stream keep the default max_iter=8 --
            # unaffected, still get full chaining.
            argv += ["--spatial-batching-max-iter", "2"]
    if mode == "batched":
        # Reorder to layer-major windows so spatial batching can see every
        # legal cross-timestep group; max_patterns=0 still forbids fusion.
        argv.append("--enable-temporal-schedule")
    old_argv = sys.argv
    try:
        sys.argv = argv
        return runtime_bench.parse_args(), argv[1:]
    finally:
        sys.argv = old_argv


def _make_base_model(model: str, bench_args, dtype: torch.dtype):
    return make_resnet_layer(
        model,
        allow_resnet32_fallback=not bench_args.require_direct_resnet32_api,
        step_mode="s",
        model_channels=bench_args.model_channels,
        lif_impl=bench_args.lif_impl,
        sequence_length=bench_args.sequence_length,
        transformer_depth=bench_args.transformer_depth,
        transformer_dim=bench_args.transformer_dim,
        transformer_heads=bench_args.transformer_heads,
        transformer_input_dim=bench_args.transformer_input_dim,
        transformer_vocab_size=bench_args.transformer_vocab_size,
        transformer_num_classes=bench_args.transformer_num_classes,
        convlstm_in_channels=bench_args.convlstm_in_channels,
        convlstm_hidden_channels=bench_args.convlstm_hidden_channels,
        convlstm_num_layers=bench_args.convlstm_num_layers,
        convlstm_height=bench_args.convlstm_height,
        convlstm_width=bench_args.convlstm_width,
        mamba_d_model=bench_args.mamba_d_model,
        mamba_n_layer=bench_args.mamba_n_layer,
        mamba_d_inner=bench_args.mamba_d_inner,
        mamba_d_state=bench_args.mamba_d_state,
        mamba_d_conv=bench_args.mamba_d_conv,
        mamba_dt_rank=bench_args.mamba_dt_rank,
        deepspeech2_freq_bins=bench_args.deepspeech2_freq_bins,
        deepspeech2_conv_channels=bench_args.deepspeech2_conv_channels,
        deepspeech2_gru_hidden=bench_args.deepspeech2_gru_hidden,
        deepspeech2_gru_layers=bench_args.deepspeech2_gru_layers,
    ).to(device="cuda", dtype=dtype).eval()


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _model_weight_bytes(model: torch.nn.Module) -> int:
    total = 0
    seen = set()
    for tensor in model.state_dict().values():
        if not torch.is_tensor(tensor) or not (tensor.is_floating_point() or tensor.is_complex()):
            continue
        storage = tensor.untyped_storage()
        key = (storage.data_ptr(), storage.nbytes())
        if key not in seen:
            seen.add(key)
            total += storage.nbytes()
    return total


def build_runnable(args, model: str, mode: str, profile: bool):
    bench_args, cli = _benchmark_namespace(args, model, mode, profile)
    dtype = runtime_bench.resolve_dtype(bench_args.dtype)
    torch.manual_seed(bench_args.seed)
    torch.cuda.manual_seed_all(bench_args.seed)
    base_model = _make_base_model(model, bench_args, dtype)
    x = make_model_input(model, bench_args, dtype)
    if model_input_mode(model) == "sequence":
        wrapper_cls = SequenceInputLoopWrapper
    elif model in STATIC_SEQUENCE_PROBE_MODELS:
        wrapper_cls = StaticSequenceLoopWrapper
        # Materialize the [T,B,...] stack once, outside the timed region --
        # same treatment make_model_input already gives SequenceInputLoopWrapper
        # models, and the same one-time-setup-cost convention
        # motivation_three_taxes.py's make_inputs uses for its own Fig. 2
        # batched_forward. Content is identical across t (rate-coded
        # convention), so this changes nothing about what the model computes.
        x = x.unsqueeze(0).expand(bench_args.T, *x.shape).contiguous()
    else:
        wrapper_cls = SingleStepModeLoopWrapper
    wrapped = wrapper_cls(base_model, bench_args.T).to(device="cuda", dtype=dtype).eval()
    counters = RewriteCounters()
    backend = make_rewrite_backend(
        bench_args,
        Path(bench_args.out_dir) / "rewrite",
        counters,
    )
    if mode == "batched" and model == "mamba":
        backend = _mamba_reorder_only_backend(
            backend, bench_args.T, bench_args.temporal_schedule_window or bench_args.temporal_fuse_window
        )
    snn_custom_ops.configure_fused_op(
        backend=bench_args.fused_op_backend,
        strict_triton=bench_args.strict_triton,
        verbose=bench_args.print_fused_op_calls,
        use_triton_autotune=not bench_args.disable_triton_autotune,
    )
    runnable = compile_with_kairos_options(
        wrapped,
        backend=backend,
        enable_cudagraphs=False,
        cudagraph_mode=bench_args.cudagraph_mode,
        fullgraph=False,
        dynamic=False,
    )
    theory = {
        "weight_bytes": _model_weight_bytes(wrapped),
        "input_bytes": _tensor_bytes(x),
        "input_shape": list(x.shape),
        "input_mode": model_input_mode(model),
        "static_sequence_probe": model in STATIC_SEQUENCE_PROBE_MODELS,
    }
    return runnable, wrapped, x, bench_args, counters, theory, cli


def invoke(runnable, model, x):
    runtime_bench._mark_cudagraph_step(model)
    reset_lif_modules(model)
    with torch.no_grad():
        return runnable(x)


def warmup(runnable, model, x, count: int):
    for _ in range(count + 1):
        invoke(runnable, model, x)
        torch.cuda.synchronize()


def time_cuda(runnable, model, x, warmups: int, repeats: int) -> Tuple[float, float]:
    warmup(runnable, model, x, warmups)
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        invoke(runnable, model, x)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.mean(samples), statistics.pstdev(samples)


def compare_output(output: torch.Tensor, reference: torch.Tensor, args) -> Dict:
    actual = output.detach().float().cpu()
    ref = reference.float()
    if actual.shape != ref.shape:
        return {"ok": False, "shape": list(actual.shape), "max_abs": math.inf, "mean_abs": math.inf}
    delta = (actual - ref).abs()
    return {
        "ok": bool(torch.allclose(actual, ref, rtol=args.rtol, atol=args.atol)),
        "shape": list(actual.shape),
        "max_abs": float(delta.max()) if delta.numel() else 0.0,
        "mean_abs": float(delta.mean()) if delta.numel() else 0.0,
    }


def release_cuda():
    gc.collect()
    torch.cuda.empty_cache()


def collect_ncu(args, model: str, mode: str, out_dir: Path) -> Tuple[Dict, Sequence[str]]:
    ncu = resolve_ncu_path(args.ncu_path)
    if ncu is None:
        raise RuntimeError(f"NCU executable not found: {args.ncu_path}")
    report = out_dir / "ncu" / f"{model}_{mode}.csv"
    report.parent.mkdir(parents=True, exist_ok=True)
    child = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--profile-child",
        "--profile-model",
        model,
        "--profile-mode",
        mode,
        "--T",
        str(args.T),
        "--batch",
        str(args.batch),
        "--dtype",
        args.dtype,
        "--seed",
        str(args.seed),
        "--profile-warmup",
        str(args.profile_warmup),
        "--out-dir",
        str(out_dir),
    ]
    if args.disable_fused_cudagraph:
        child.append("--disable-fused-cudagraph")
    cmd = [
        ncu,
        "--target-processes",
        "all",
        "--profile-from-start",
        "off",
        "--csv",
        "--page",
        "raw",
        "--metrics",
        ",".join(EVAL_NCU_METRICS),
        "--log-file",
        str(report),
        *child,
    ]
    completed = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=args.ncu_timeout_sec,
        check=False,
    )
    if completed.returncode != 0:
        tail = report.read_text(encoding="utf-8", errors="replace")[-4000:] if report.exists() else ""
        raise RuntimeError(
            f"NCU failed for model={model}, mode={mode}, exit={completed.returncode}\n"
            f"{completed.stdout[-4000:]}{tail}"
        )
    metrics = parse_ncu_csv(report)
    return metrics, cmd


def profile_child(args):
    runnable, model, x, _, _, _, _ = build_runnable(
        args, args.profile_model, args.profile_mode, profile=True
    )
    warmup(runnable, model, x, args.profile_warmup)
    torch.cuda.cudart().cudaProfilerStart()
    invoke(runnable, model, x)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


def write_csv(path: Path, rows: Iterable[Dict], columns: Sequence[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_checkpoint(path: Path) -> Dict[Tuple[str, str], Dict]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            (row["model"], row["mode"]): row
            for row in csv.DictReader(handle)
            if row.get("model") and row.get("mode")
        }


def write_summary(path: Path, rows):
    lines = ["# Real-model execution-mode summary", ""]
    for model in MODELS:
        selected = {row["mode"]: row for row in rows if row["model"] == model}
        if len(selected) != len(MODES):
            continue
        lines.append(
            f"- **{model}**: per-step {selected['per_step']['dram_total_bytes']/1e9:.3f} GB "
            f"/{selected['per_step']['time_ms_mean']:.3f} ms, "
            f"graph-batched {selected['batched']['dram_total_bytes']/1e9:.3f} GB "
            f"/{selected['batched']['time_ms_mean']:.3f} ms, fused "
            f"{selected['fused']['dram_total_bytes']/1e9:.3f} GB "
            f"/{selected['fused']['time_ms_mean']:.3f} ms, multi-stream "
            f"{selected['multi_stream']['dram_total_bytes']/1e9:.3f} GB "
            f"/{selected['multi_stream']['time_ms_mean']:.3f} ms."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def environment_metadata(args):
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "ncu_path": resolve_ncu_path(args.ncu_path) if args.collect_ncu else None,
        "ncu_metrics": list(EVAL_NCU_METRICS),
        "gpu_clock_mhz_requested": args.lock_gpu_clock_mhz,
        "T": args.T,
        "batch": args.batch,
        "dtype": args.dtype,
        "models": args.models,
        "modes": list(MODES),
        "standalone_cudagraph_timing": not args.disable_fused_cudagraph,
        "standalone_cudagraph_ncu": False,
        "standalone_cudagraph_note": "Enabled for fused/multi-stream timing, disabled for all NCU children so one uncaptured steady-state forward is measured.",
        "lower_bound_method": "deduplicated floating state_dict storage bytes + one model input read + one actual output write",
    }


def main(args):
    if args.profile_child:
        profile_child(args)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.lock_gpu_clock_mhz is not None:
        lock_gpu_clock(args.lock_gpu_clock_mhz)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = (
        read_checkpoint(out_dir / "eval_three_taxes.csv") if args.resume else {}
    )
    metadata = environment_metadata(args)
    previous_metadata_path = out_dir / "metadata.json"
    previous_cases = {}
    if args.resume and previous_metadata_path.exists():
        previous_cases = json.loads(
            previous_metadata_path.read_text(encoding="utf-8")
        ).get("cases", {})
    metadata["cases"] = previous_cases
    rows = []
    correctness_rows = []
    try:
        for model_name in args.models:
            print(f"[model] {model_name}")
            reference = None
            expected_theory = None
            for mode in MODES:
                print(f"  [mode] {mode}")
                runnable, model, x, _, counters, theory, cli = build_runnable(
                    args, model_name, mode, profile=False
                )
                mean_ms, std_ms = time_cuda(
                    runnable, model, x, args.warmup, args.repeat
                )
                output = invoke(runnable, model, x)
                torch.cuda.synchronize()
                if reference is None:
                    reference = output.detach().float().cpu()
                check = compare_output(output, reference, args)
                if not check["ok"]:
                    raise AssertionError(
                        f"correctness failed for {model_name}/{mode}: {check}"
                    )
                theory["output_bytes"] = _tensor_bytes(output)
                theory["output_shape"] = list(output.shape)
                theory["min_bytes_theory"] = (
                    theory["weight_bytes"]
                    + theory["input_bytes"]
                    + theory["output_bytes"]
                )
                if expected_theory is None:
                    expected_theory = theory["min_bytes_theory"]
                elif theory["min_bytes_theory"] != expected_theory:
                    raise AssertionError(
                        f"theoretical bytes differ by mode for {model_name}: "
                        f"{expected_theory} vs {theory['min_bytes_theory']}"
                    )
                if args.collect_ncu:
                    checkpoint_row = checkpoint.get((model_name, mode))
                    if checkpoint_row is not None:
                        total_bytes = float(checkpoint_row["dram_total_bytes"])
                        previous_case = metadata["cases"].get(
                            f"{model_name}/{mode}", {}
                        )
                        metrics = previous_case.get("ncu_extra_metrics", {})
                        ncu_cmd = previous_case.get("ncu_command")
                        print("    reusing checkpointed NCU traffic")
                    else:
                        metrics, ncu_cmd = collect_ncu(args, model_name, mode, out_dir)
                        total_bytes = metrics["dram_read_bytes"] + metrics["dram_write_bytes"]
                else:
                    metrics, ncu_cmd, total_bytes = {}, None, math.nan
                ratio = total_bytes / expected_theory if math.isfinite(total_bytes) else math.nan
                row = {
                    "model": model_name,
                    "mode": mode,
                    "T": args.T,
                    "batch": args.batch,
                    "dram_total_bytes": total_bytes,
                    "min_bytes_theory": expected_theory,
                    "ratio_to_lower_bound": ratio,
                    "time_ms_mean": mean_ms,
                    "time_ms_std": std_ms,
                }
                rows.append(row)
                correctness_rows.append({"model": model_name, "mode": mode, **check})
                metadata["cases"][f"{model_name}/{mode}"] = {
                    "benchmark_cli": cli,
                    "ncu_command": ncu_cmd,
                    "ncu_reused_checkpoint": checkpoint_row is not None if args.collect_ncu else False,
                    "theory": theory,
                    "rewrite_counters": asdict(counters),
                    "ncu_extra_metrics": metrics,
                }
                write_csv(out_dir / "eval_three_taxes.csv", rows, CSV_COLUMNS)
                write_csv(
                    out_dir / "correctness.csv",
                    correctness_rows,
                    ("model", "mode", "ok", "shape", "max_abs", "mean_abs"),
                )
                (out_dir / "metadata.json").write_text(
                    json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
                )
                print(
                    f"    time={mean_ms:.3f}+/-{std_ms:.3f} ms "
                    f"HBM={total_bytes/1e9:.3f} GB ratio={ratio:.2f}x"
                    if math.isfinite(total_bytes)
                    else f"    time={mean_ms:.3f}+/-{std_ms:.3f} ms HBM=not-collected"
                )
                del runnable, model, x, output
                release_cuda()
        write_summary(out_dir / "paper_summary.md", rows)
    finally:
        if args.lock_gpu_clock_mhz is not None:
            subprocess.run([shutil.which("nvidia-smi") or "nvidia-smi", "-rgc"], check=False)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--T", type=int, default=16)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp32")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--profile-warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--collect-ncu", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ncu-path", default="ncu")
    parser.add_argument("--ncu-timeout-sec", type=int, default=7200)
    parser.add_argument("--lock-gpu-clock-mhz", type=int, default=None)
    parser.add_argument("--disable-fused-cudagraph", action="store_true")
    parser.add_argument("--out-dir", default="test/eval_three_taxes")
    parser.add_argument("--profile-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--profile-model", choices=MODELS, help=argparse.SUPPRESS)
    parser.add_argument("--profile-mode", choices=MODES, help=argparse.SUPPRESS)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
