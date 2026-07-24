"""Export the 13 Kairos workloads to ONNX and benchmark them with BladeDISC.

The ONNX graph follows benchmark_tensorrt_runtime.py exactly: a single-step
model is wrapped by a T-step loop (or its sequence-input counterpart), so one
invocation executes all T timesteps without changing the underlying model.
BladeDISC consumes the same PyTorch wrapper through TorchScript because
TorchBlade/DISC does not use an ONNX file as its frontend.
"""

import argparse
import copy
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch

# Keep DISC's runtime CUDA tools aligned with the cu128 PyTorch build. Without
# this, XLA may discover the system CUDA 13.1 fatbinary first; CUDA 13 removed
# the legacy --image option used by this BladeDISC revision.
os.environ.setdefault(
    "XLA_FLAGS", "--xla_gpu_cuda_data_dir=/usr/local/cuda-12.8"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.benchmark_tensorrt_runtime import (  # noqa: E402
    onnx_graph_contains_custom_lif,
    replace_custom_lif_for_export,
)
from benchmarks.validate_kairos_baselines import (  # noqa: E402
    LIF_IMPL_CHOICES,
    SequenceInputLoopWrapper,
    SingleStepModeLoopWrapper,
    make_model_input,
    make_resnet_layer,
    model_input_mode,
    reset_lif_modules,
)


WORKLOADS = [
    "resnet18", "resnet34", "alexnet", "zfnet", "vgg11", "vgg16",
    "mobilenetv1", "mobilenetv2", "spiketransformer", "spikebert",
    "convlstm", "mamba", "deepspeech2",
]


def percentile(values, q):
    values = sorted(values)
    return values[int(round((len(values) - 1) * q))]


def build_case(model_name, args, dtype):
    layer = make_resnet_layer(
        model_name=model_name,
        allow_resnet32_fallback=True,
        step_mode="s",
        model_channels=args.model_channels,
        lif_impl=args.lif_impl,
        sequence_length=args.sequence_length,
        transformer_depth=args.transformer_depth,
        transformer_dim=args.transformer_dim,
        transformer_heads=args.transformer_heads,
        transformer_input_dim=args.transformer_input_dim,
        transformer_vocab_size=args.transformer_vocab_size,
        transformer_num_classes=args.transformer_num_classes,
        convlstm_in_channels=args.convlstm_in_channels,
        convlstm_hidden_channels=args.convlstm_hidden_channels,
        convlstm_num_layers=args.convlstm_num_layers,
        convlstm_height=args.convlstm_height,
        convlstm_width=args.convlstm_width,
        mamba_d_model=args.mamba_d_model,
        mamba_n_layer=args.mamba_n_layer,
        mamba_d_inner=args.mamba_d_inner,
        mamba_d_state=args.mamba_d_state,
        mamba_d_conv=args.mamba_d_conv,
        mamba_dt_rank=args.mamba_dt_rank,
        deepspeech2_freq_bins=args.deepspeech2_freq_bins,
        deepspeech2_conv_channels=args.deepspeech2_conv_channels,
        deepspeech2_gru_hidden=args.deepspeech2_gru_hidden,
        deepspeech2_gru_layers=args.deepspeech2_gru_layers,
    )
    wrapper_cls = (
        SequenceInputLoopWrapper
        if model_input_mode(model_name) == "sequence"
        else SingleStepModeLoopWrapper
    )
    model = wrapper_cls(layer, args.T).to(args.device).to(dtype).eval()
    x = make_model_input(model_name, args, dtype)
    reset_lif_modules(model)
    # This is the same export-only decomposition used by TensorRT/TVM.
    model = copy.deepcopy(model)
    replaced = replace_custom_lif_for_export(model)
    reset_lif_modules(model)
    return model, x, wrapper_cls.__name__, replaced


def export_onnx(model, x, path, opset):
    reset_lif_modules(model)
    with torch.no_grad():
        torch.onnx.export(
            model,
            (x,),
            path.as_posix(),
            input_names=["input"],
            output_names=["output"],
            opset_version=opset,
            dynamo=False,
            do_constant_folding=False,
            optimize=False,
        )


def benchmark(model, x, warmup, repeat):
    for _ in range(warmup):
        reset_lif_modules(model)
        with torch.no_grad():
            model(x)
    torch.cuda.synchronize()
    samples = []
    schedule_samples = []
    for _ in range(repeat):
        reset_lif_modules(model)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            model(x)
        # Scheduling overhead: CPU-side dispatch time before the sync below,
        # distinct from the full wall-clock sample recorded after sync.
        dispatched = time.perf_counter()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
        schedule_samples.append((dispatched - start) * 1000.0)
    return {
        "mean_ms": statistics.mean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "p50_ms": percentile(samples, 0.50),
        "p90_ms": percentile(samples, 0.90),
        "repeat": repeat,
        "schedule_mean_ms": statistics.mean(schedule_samples),
        "schedule_min_ms": min(schedule_samples),
    }


def run_one(model_name, args):
    result = {"ok": False, "model": model_name}
    try:
        import torch_blade
        import torch_blade.mlir

        dtype = torch.float16 if args.dtype == "fp16" else torch.float32
        model, x, wrapper, replaced = build_case(model_name, args, dtype)
        run_dir = Path(args.out_dir) / model_name / args.dtype
        run_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = run_dir / f"{model_name}_single_step_mode_T{args.T}_{args.dtype}.onnx"
        export_onnx(model, x, onnx_path, args.opset)

        reset_lif_modules(model)
        with torch.no_grad():
            eager_out = model(x)
        reset_lif_modules(model)
        with torch.no_grad():
            traced_model = torch.jit.trace(
                model, (x,), strict=False, check_trace=False
            )
        cfg = torch_blade.Config()
        cfg.optimization_pipeline = torch_blade.mlir.backend_name()
        cfg.enable_static_shape = True
        with cfg:
            disc_model = torch_blade.optimize(
                traced_model, model_inputs=(x,)
            )
        disc_graph = str(disc_model.forward.graph)
        if "torch_blade.Engine" not in disc_graph:
            raise RuntimeError(
                "BladeDISC compilation fell back to TorchScript; "
                "optimized graph contains no torch_blade.Engine"
            )
        reset_lif_modules(model)
        with torch.no_grad():
            disc_out = disc_model(x)
        max_abs_diff = (eager_out - disc_out).abs().max().item()
        torch.jit.save(disc_model, (run_dir / "bladedisc.pt").as_posix())

        result.update({
            "ok": True,
            "wrapper": wrapper,
            "batch_size": args.batch_size,
            "time_steps": args.T,
            "dtype": args.dtype,
            "input_shape": list(x.shape),
            "onnx_path": str(onnx_path),
            "onnx_contains_custom_lif": onnx_graph_contains_custom_lif(onnx_path),
            "export_custom_lif_replaced": replaced,
            "max_abs_diff": max_abs_diff,
            "latency": benchmark(disc_model, x, args.warmup, args.repeat),
        })
    except Exception:
        result["error"] = traceback.format_exc()
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", choices=WORKLOADS, default=WORKLOADS)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--T", type=int, default=16, help="Number of single-step calls.")
    p.add_argument("--dtype", choices=("fp16", "fp32"), default="fp32")
    p.add_argument("--device", default="cuda")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeat", type=int, default=50)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--out-dir", default="test/bladedisc_validation")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--height", type=int, default=224)
    p.add_argument("--width", type=int, default=224)
    p.add_argument("--model-channels", type=int, default=64)
    p.add_argument("--lif-impl", choices=LIF_IMPL_CHOICES, default="kairos")
    p.add_argument("--sequence-length", type=int, default=256)
    p.add_argument("--transformer-depth", type=int, default=8)
    p.add_argument("--transformer-dim", type=int, default=256)
    p.add_argument("--transformer-heads", type=int, default=8)
    p.add_argument("--transformer-input-dim", type=int, default=768)
    p.add_argument("--transformer-vocab-size", type=int, default=30522)
    p.add_argument("--transformer-num-classes", type=int, default=100)
    p.add_argument("--convlstm-in-channels", type=int, default=1)
    p.add_argument("--convlstm-hidden-channels", type=int, default=64)
    p.add_argument("--convlstm-num-layers", type=int, default=2)
    p.add_argument("--convlstm-height", type=int, default=64)
    p.add_argument("--convlstm-width", type=int, default=64)
    p.add_argument("--mamba-d-model", type=int, default=768)
    p.add_argument("--mamba-n-layer", type=int, default=24)
    p.add_argument("--mamba-d-inner", type=int, default=1536)
    p.add_argument("--mamba-d-state", type=int, default=16)
    p.add_argument("--mamba-d-conv", type=int, default=4)
    p.add_argument("--mamba-dt-rank", type=int, default=48)
    p.add_argument("--deepspeech2-freq-bins", type=int, default=161)
    p.add_argument("--deepspeech2-conv-channels", type=int, default=32)
    p.add_argument("--deepspeech2-gru-hidden", type=int, default=800)
    p.add_argument("--deepspeech2-gru-layers", type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    if args.batch_size < 1 or args.T < 1:
        raise ValueError("--batch-size and --T must be positive")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    results = {name: run_one(name, args) for name in args.models}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = out / "summary.json"
    summary.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"[WRITE] {summary}")
    if not all(item["ok"] for item in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
