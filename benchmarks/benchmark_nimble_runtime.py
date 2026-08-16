#!/usr/bin/env python3
"""Benchmark Chronos workloads with the SNUSPL Nimble PyTorch fork."""

import argparse
import json
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.nimble_model_loader import load_workload_namespace


MODELS = (
    "resnet18", "resnet34", "resnet32", "alexnet", "zfnet", "vgg11",
    "vgg16", "mobilenetv1", "mobilenetv2", "spiketransformer",
    "spikebert", "convlstm", "mamba", "deepspeech2", "nafnet", "bsrn",
)


def percentile(values, q):
    ordered = sorted(values)
    return ordered[int(round((len(ordered) - 1) * q))]


def make_input(model_name, args, dtype):
    if model_name == "spiketransformer":
        return torch.randn(
            args.batch_size, args.sequence_length, args.transformer_input_dim,
            device="cuda", dtype=dtype,
        )
    if model_name == "spikebert":
        return torch.randint(
            0, args.transformer_vocab_size,
            (args.batch_size, args.sequence_length),
            device="cuda", dtype=torch.int64,
        )
    if model_name == "convlstm":
        return torch.randn(
            args.T, args.batch_size, args.convlstm_in_channels,
            args.convlstm_height, args.convlstm_width,
            device="cuda", dtype=dtype,
        )
    if model_name == "mamba":
        return torch.randn(
            args.T, args.batch_size, args.mamba_d_model,
            device="cuda", dtype=dtype,
        )
    if model_name == "deepspeech2":
        return torch.randn(
            args.batch_size, 1, args.deepspeech2_freq_bins, 2 * args.T,
            device="cuda", dtype=dtype,
        )
    return torch.randn(
        args.batch_size, 3, args.height, args.width,
        device="cuda", dtype=dtype,
    )


def build_case(model_name, args, workload, dtype):
    model = workload["make_resnet_layer"](
        model_name=model_name,
        allow_resnet32_fallback=True,
        step_mode="s",
        model_channels=args.model_channels,
        lif_impl="kairos",
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
        nafnet_width=args.nafnet_width,
        nafnet_enc_blk_nums=tuple(args.nafnet_enc_blk_nums),
        nafnet_middle_blk_num=args.nafnet_middle_blk_num,
        nafnet_dec_blk_nums=tuple(args.nafnet_dec_blk_nums),
        bsrn_num_feat=args.bsrn_num_feat,
        bsrn_num_block=args.bsrn_num_block,
    ).cuda().to(dtype).eval()
    wrapper_cls = (
        workload["SequenceInputLoopWrapper"]
        if workload["model_input_mode"](model_name) == "sequence"
        else workload["SingleStepModeLoopWrapper"]
    )
    return wrapper_cls(model, args.T).cuda().to(dtype).eval()


def run_model(model_name, args, workload):
    result = {
        "model": model_name,
        "ok": False,
        "T": args.T,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "use_multi_stream": args.use_multi_stream,
    }
    try:
        dtype = torch.float16 if args.dtype == "fp16" else torch.float32
        model = build_case(model_name, args, workload, dtype)
        x = make_input(model_name, args, dtype)
        torch.backends.cudnn.benchmark = True

        nimble_model = torch.cuda.Nimble(model)
        prepare_start = time.perf_counter()
        nimble_model.prepare(
            x,
            training=False,
            use_multi_stream=args.use_multi_stream,
            relaxed=args.relaxed_capture,
        )
        torch.cuda.synchronize()
        prepare_ms = (time.perf_counter() - prepare_start) * 1000.0

        first_start = torch.cuda.Event(enable_timing=True)
        first_end = torch.cuda.Event(enable_timing=True)
        first_start.record()
        nimble_model(x)
        first_end.record()
        first_end.synchronize()
        first_ms = first_start.elapsed_time(first_end)

        for _ in range(args.warmup):
            nimble_model(x)
        torch.cuda.synchronize()
        samples = []
        for _ in range(args.repeat):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            nimble_model(x)
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))

        median = statistics.median(samples)
        result.update({
            "ok": True,
            "nimble_prepare_ms": prepare_ms,
            "nimble_prepare_overhead_note": "includes conv algorithm selection, JIT rewrite, multi-stream scheduling, precapture, and CUDA graph capture",
            "first_replay_ms": first_ms,
            "first_replay_overhead_ms": max(0.0, first_ms - median),
            "latency_ms": {
                "mean": statistics.mean(samples),
                "std": statistics.pstdev(samples),
                "min": min(samples),
                "max": max(samples),
                "median": median,
                "p90": percentile(samples, 0.90),
                "p99": percentile(samples, 0.99),
            },
            "throughput_qps": 1000.0 / statistics.mean(samples),
        })
    except Exception:
        result["error"] = traceback.format_exc()
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=["resnet18"])
    parser.add_argument("--T", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--dtype", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--model-channels", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--transformer-depth", type=int, default=8)
    parser.add_argument("--transformer-dim", type=int, default=256)
    parser.add_argument("--transformer-heads", type=int, default=8)
    parser.add_argument("--transformer-input-dim", type=int, default=768)
    parser.add_argument("--transformer-vocab-size", type=int, default=30522)
    parser.add_argument("--transformer-num-classes", type=int, default=100)
    parser.add_argument("--convlstm-in-channels", type=int, default=1)
    parser.add_argument("--convlstm-hidden-channels", type=int, default=64)
    parser.add_argument("--convlstm-num-layers", type=int, default=2)
    parser.add_argument("--convlstm-height", type=int, default=64)
    parser.add_argument("--convlstm-width", type=int, default=64)
    parser.add_argument("--mamba-d-model", type=int, default=768)
    parser.add_argument("--mamba-n-layer", type=int, default=24)
    parser.add_argument("--mamba-d-inner", type=int, default=1536)
    parser.add_argument("--mamba-d-state", type=int, default=16)
    parser.add_argument("--mamba-d-conv", type=int, default=4)
    parser.add_argument("--mamba-dt-rank", type=int, default=48)
    parser.add_argument("--deepspeech2-freq-bins", type=int, default=161)
    parser.add_argument("--deepspeech2-conv-channels", type=int, default=32)
    parser.add_argument("--deepspeech2-gru-hidden", type=int, default=800)
    parser.add_argument("--deepspeech2-gru-layers", type=int, default=3)
    parser.add_argument("--nafnet-width", type=int, default=8)
    parser.add_argument("--nafnet-enc-blk-nums", nargs="+", type=int, default=[2, 2, 4, 8])
    parser.add_argument("--nafnet-middle-blk-num", type=int, default=12)
    parser.add_argument("--nafnet-dec-blk-nums", nargs="+", type=int, default=[2, 2, 2, 2])
    parser.add_argument("--bsrn-num-feat", type=int, default=16)
    parser.add_argument("--bsrn-num-block", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    stream_group = parser.add_mutually_exclusive_group()
    stream_group.add_argument("--use-multi-stream", dest="use_multi_stream", action="store_true")
    stream_group.add_argument("--no-use-multi-stream", dest="use_multi_stream", action="store_false")
    parser.set_defaults(use_multi_stream=True)
    parser.add_argument("--relaxed-capture", action="store_true")
    parser.add_argument("--out-dir", default="test/nimble_validation")
    return parser.parse_args()


def main():
    args = parse_args()
    if not hasattr(torch.cuda, "Nimble"):
        raise RuntimeError("This script must run with the SNUSPL Nimble PyTorch fork")
    workload = load_workload_namespace(PROJECT_ROOT)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for model_name in args.models:
        print("[Nimble] model={} T={} batch={}".format(model_name, args.T, args.batch_size))
        result = run_model(model_name, args, workload)
        results[model_name] = result
        (output_dir / "{}_summary.json".format(model_name)).write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print("  {}".format("OK" if result["ok"] else "FAIL"))
    (output_dir / "nimble_summary_all.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
