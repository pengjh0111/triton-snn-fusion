import argparse
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

import runtime.snn_custom_ops as snn_custom_ops
from compiler.kairos_compile import build_kairos_compile_config, compile_with_kairos_options
from compiler.fx_dag_analyzer import (
    build_fx_dag,
    dump_fx_dag_edges,
    dump_fx_dag_json,
    dump_fx_dag_regions,
    dump_fx_dag_text,
    find_fused_convlif_regions,
    summarize_fx_dag,
    try_dump_fx_dag_svg,
)
from compiler.fx_lif_rewrite import (
    RewriteStats,
    count_fused_conv_lif_state_nodes,
    count_lif_state_nodes,
    match_conv_bn_lif_state,
    match_conv_lif_state,
    rewrite_conv_bn_lif_state_to_fused,
    rewrite_conv_lif_state_to_fused,
)
from compiler.fx_lif_temporal_rewrite import (
    collect_conv_bn_add_lif_state_patterns,
    collect_conv_bn_lif_state_patterns,
    collect_standalone_lif_state_patterns,
    count_fused_temporal_conv_add_lif_state_nodes,
    count_fused_temporal_conv_lif_state_nodes,
    count_fused_temporal_lif_state_nodes,
    dump_temporal_patterns,
    dump_temporal_rewrite_log,
    dump_temporal_windows,
    group_temporal_residual_patterns,
    group_temporal_patterns,
    group_temporal_lif_patterns,
    make_temporal_residual_windows,
    make_temporal_windows,
    make_temporal_lif_windows,
    rewrite_temporal_conv_bn_add_lif_state_to_fused,
    rewrite_temporal_conv_bn_lif_state_to_fused,
    rewrite_temporal_lif_state_to_fused,
)
from compiler.fx_spatial_batching import apply_spatial_batching
from compiler.fx_temporal_annotation import annotate_temporal_metadata
from compiler.fx_temporal_graph_validation import (
    analyze_temporal_graph,
    dump_temporal_graph_validation,
    print_temporal_graph_summary,
)
from compiler.fx_temporal_scheduler import reorder_fx_graph_by_temporal_windows
from compiler.fx_temporal_spatial_canonicalize import canonicalize_temporal_spatial_ir
from benchmarks.helpers.models_for_fx import build_model, reset_custom_stateful_lif_modules


ARGS = None
OUT_DIR = Path("fx_convlif_inductor_debug")
CAPTURED_GRAPHS = 0
TOTAL_STATS = RewriteStats()


def tensor_meta_summary(node: torch.fx.Node) -> str:
    meta = node.meta.get("tensor_meta") or node.meta.get("val")
    if meta is None:
        return ""
    shape = getattr(meta, "shape", None)
    dtype = getattr(meta, "dtype", None)
    device = getattr(meta, "device", None)
    if shape is not None:
        return f"shape={tuple(shape)} dtype={dtype}"
    if isinstance(meta, torch.Tensor):
        return f"shape={tuple(meta.shape)} dtype={meta.dtype} device={meta.device}"
    return repr(meta)


def dump_fx_debug(gm: torch.fx.GraphModule, out_dir: Path):
    lines = []
    for node in gm.graph.nodes:
        users = [user.name for user in node.users]
        meta = tensor_meta_summary(node)
        block = (
            f"name={node.name}\n"
            f"  op={node.op}\n"
            f"  target={node.target}\n"
            f"  args={node.args}\n"
            f"  kwargs={node.kwargs}\n"
            f"  users={users}\n"
            f"  tensor_meta={meta}\n"
        )
        lines.append(block)
        print(block, end="")
        if ARGS.print_node_meta and node.meta:
            print(f"  raw_meta={node.meta}")
    (out_dir / "fx_debug_nodes.txt").write_text("\n".join(lines), encoding="utf-8")


def save_graph_files(gm: torch.fx.GraphModule, out_dir: Path, prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{prefix}_fx.py").write_text(gm.code, encoding="utf-8")
    (out_dir / f"{prefix}_fx.txt").write_text(str(gm.graph), encoding="utf-8")


def dump_rewritten_fx_dag(gm: torch.fx.GraphModule, graph_dir: Path):
    try:
        dag = build_fx_dag(gm)
        dump_fx_dag_text(dag, graph_dir / "fx_dag.txt")
        dump_fx_dag_edges(dag, graph_dir / "fx_dag_edges.txt")
        dump_fx_dag_json(dag, graph_dir / "fx_dag.json")
        try_dump_fx_dag_svg(dag, graph_dir / "fx_dag.dot", graph_dir / "fx_dag.svg")
        summary = summarize_fx_dag(dag)
        (graph_dir / "fx_dag_summary.txt").write_text(
            "\n".join(f"{key}: {value}" for key, value in summary.items()) + "\n",
            encoding="utf-8",
        )
        regions = find_fused_convlif_regions(dag)
        dump_fx_dag_regions(regions, graph_dir / "fx_dag_regions.txt")
        print("[DAG] summary:")
        for key, value in summary.items():
            print(f"      {key}: {value}")
        print(f"[DAG] fused ConvLIF regions: {len(regions)}")
    except Exception:
        print("WARNING: FX DAG dump failed. Continuing main flow.")
        traceback.print_exc()


def build_placeholder_values(gm: torch.fx.GraphModule, example_inputs) -> Dict[torch.fx.Node, Any]:
    placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
    return {node: value for node, value in zip(placeholders, example_inputs)}


def inductor_options_from_compile_kwargs(compile_kwargs: Dict[str, Any]):
    options = compile_kwargs.get("options")
    if options is None and compile_kwargs.get("mode") == "reduce-overhead":
        options = {"triton.cudagraphs": True}
    return options


def rewrite_backend(gm: torch.fx.GraphModule, example_inputs, **compile_kwargs):
    global CAPTURED_GRAPHS, TOTAL_STATS

    graph_idx = CAPTURED_GRAPHS
    CAPTURED_GRAPHS += 1
    graph_dir = OUT_DIR if graph_idx == 0 else OUT_DIR / f"graph_{graph_idx}"
    graph_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n========== Captured FX graph {graph_idx} ==========")
    save_graph_files(gm, graph_dir, "original")
    dump_fx_debug(gm, graph_dir)

    placeholder_values = build_placeholder_values(gm, example_inputs)
    lif_state_count = count_lif_state_nodes(gm)
    temporal_replaced_patterns = 0
    temporal_replaced_windows = 0
    temporal_skipped_windows = 0
    temporal_residual_replaced_patterns = 0
    temporal_residual_replaced_windows = 0
    temporal_residual_skipped_windows = 0
    temporal_lif_replaced_patterns = 0
    temporal_lif_rewritten_windows = 0
    temporal_lif_skipped_windows = 0

    annotation_window = ARGS.temporal_schedule_window or ARGS.temporal_fuse_window
    annotation_stats = annotate_temporal_metadata(
        gm,
        annotation_window,
        ARGS.T,
        strict=False,
    )
    print(
        f"[TEMPORAL_ANNOTATION] annotated={annotation_stats.temporal_annotated_nodes} "
        f"missing={annotation_stats.temporal_annotation_missing} "
        f"roles={annotation_stats.temporal_annotation_roles}"
    )

    temporal_patterns = collect_conv_bn_lif_state_patterns(gm) if not ARGS.disable_conv_bn_lif else []
    residual_patterns = collect_conv_bn_add_lif_state_patterns(gm) if not ARGS.disable_conv_bn_lif else []
    if ARGS.enable_temporal_schedule and temporal_patterns:
        schedule_window = ARGS.temporal_schedule_window or ARGS.temporal_fuse_window
        schedule_result = reorder_fx_graph_by_temporal_windows(
            gm,
            ARGS.T,
            schedule_window,
            temporal_patterns,
            dump_dir=graph_dir if ARGS.temporal_schedule_dump else None,
            strict=ARGS.temporal_schedule_strict,
        )
        print(
            f"[SCHEDULE] ok={schedule_result.ok}, windows={schedule_result.scheduled_windows}, "
            f"moved_nodes={schedule_result.moved_nodes}, reason={schedule_result.reason}"
        )
        if schedule_result.ok:
            temporal_patterns = collect_conv_bn_lif_state_patterns(gm)
            residual_patterns = collect_conv_bn_add_lif_state_patterns(gm)
        elif ARGS.temporal_schedule_strict:
            raise RuntimeError(schedule_result.reason)


    if ARGS.enable_temporal_rewrite and ARGS.temporal_fuse_window > 1 and not ARGS.disable_conv_bn_lif:
        temporal_groups = group_temporal_patterns(temporal_patterns)
        temporal_windows = make_temporal_windows(
            temporal_groups,
            ARGS.temporal_fuse_window,
            ARGS.temporal_allow_tail,
        )
        dump_temporal_patterns(temporal_groups, graph_dir / "temporal_patterns.txt")
        dump_temporal_windows(temporal_windows, graph_dir / "temporal_windows.txt")
        temporal_log = []
        if ARGS.disable_rewrite:
            temporal_log.append("SKIP: --disable-rewrite enabled")
        else:
            temporal_stats = rewrite_temporal_conv_bn_lif_state_to_fused(
                gm,
                temporal_windows,
                placeholder_values,
                ARGS.max_patterns,
            )
            temporal_replaced_patterns = temporal_stats.temporal_replaced_patterns
            temporal_replaced_windows = temporal_stats.temporal_replaced_windows
            temporal_skipped_windows = temporal_stats.temporal_skipped_windows
            temporal_log.extend(temporal_stats.log)

            residual_patterns = collect_conv_bn_add_lif_state_patterns(gm)
            residual_groups = group_temporal_residual_patterns(residual_patterns)
            residual_windows = make_temporal_residual_windows(
                residual_groups,
                ARGS.temporal_fuse_window,
                ARGS.temporal_allow_tail,
            )
            if residual_windows:
                residual_stats = rewrite_temporal_conv_bn_add_lif_state_to_fused(
                    gm,
                    residual_windows,
                    placeholder_values,
                    max(0, ARGS.max_patterns - temporal_replaced_patterns),
                )
                temporal_residual_replaced_patterns = residual_stats.temporal_residual_replaced_patterns
                temporal_residual_replaced_windows = residual_stats.temporal_residual_replaced_windows
                temporal_residual_skipped_windows = residual_stats.temporal_residual_skipped_windows
                temporal_replaced_patterns += residual_stats.temporal_residual_replaced_patterns
                temporal_log.extend(residual_stats.log)

            if not ARGS.disable_temporal_lif_rewrite:
                lif_patterns = collect_standalone_lif_state_patterns(gm)
                lif_groups = group_temporal_lif_patterns(lif_patterns)
                lif_windows = make_temporal_lif_windows(
                    lif_groups,
                    ARGS.temporal_fuse_window,
                    ARGS.temporal_allow_tail,
                )
                if lif_windows:
                    lif_stats = rewrite_temporal_lif_state_to_fused(
                        gm,
                        lif_windows,
                        max(0, ARGS.max_patterns - temporal_replaced_patterns),
                    )
                    temporal_lif_replaced_patterns = lif_stats.temporal_lif_replaced_patterns
                    temporal_lif_rewritten_windows = lif_stats.temporal_lif_rewritten_windows
                    temporal_lif_skipped_windows = lif_stats.temporal_lif_skipped_windows
                    temporal_replaced_patterns += lif_stats.temporal_lif_replaced_patterns
                    temporal_log.extend(lif_stats.log)
        dump_temporal_rewrite_log(temporal_log, graph_dir / "temporal_rewrite_log.txt")

    direct_matches = match_conv_lif_state(gm)
    conv_bn_matches = []
    if ARGS.disable_conv_bn_lif:
        print("[SKIP] --disable-conv-bn-lif enabled")
    else:
        conv_bn_matches = match_conv_bn_lif_state(gm)

    direct_replaced = 0
    conv_bn_replaced = 0
    if ARGS.disable_rewrite:
        print("[SKIP] --disable-rewrite enabled")
        gm.graph.lint()
        gm.recompile()
    else:
        try:
            remaining = max(0, ARGS.max_patterns - temporal_replaced_patterns)
            conv_bn_replaced = rewrite_conv_bn_lif_state_to_fused(
                gm,
                conv_bn_matches,
                placeholder_values,
                remaining,
            )
            remaining = max(0, remaining - conv_bn_replaced)
            direct_replaced = rewrite_conv_lif_state_to_fused(
                gm,
                direct_matches,
                placeholder_values,
                remaining,
            )
        except Exception:
            print("WARNING: rewrite failed; continuing with the current graph.")
            traceback.print_exc()
            gm.graph.lint()
            gm.recompile()

    if ARGS.enable_spatial_batching and not ARGS.disable_rewrite:
        try:
            spatial_window = ARGS.temporal_schedule_window or ARGS.temporal_fuse_window
            apply_spatial_batching(
                gm,
                spatial_window,
                ARGS.spatial_batching_ops,
                dump_dir=graph_dir if ARGS.spatial_batching_dump else None,
                strict=ARGS.spatial_batching_strict,
                enable_chain=False,
            )
        except Exception:
            if ARGS.spatial_batching_strict:
                raise
            print("WARNING: spatial batching failed; continuing with the current graph.")
            traceback.print_exc()

    canonicalize_temporal_spatial_ir(gm, dump_dir=graph_dir, strict=False)

    temporal_graph_stats = analyze_temporal_graph(gm)
    print_temporal_graph_summary(temporal_graph_stats)
    dump_temporal_graph_validation(temporal_graph_stats, graph_dir / "temporal_graph_validation.json")

    fused_state_count = count_fused_conv_lif_state_nodes(gm)
    fused_temporal_state_count = count_fused_temporal_conv_lif_state_nodes(gm)
    fused_temporal_residual_state_count = count_fused_temporal_conv_add_lif_state_nodes(gm)
    fused_temporal_lif_state_count = count_fused_temporal_lif_state_nodes(gm)
    save_graph_files(gm, graph_dir, "rewritten")
    if not ARGS.disable_dag_dump:
        dump_rewritten_fx_dag(gm, graph_dir)

    TOTAL_STATS.lif_state_nodes += lif_state_count
    TOTAL_STATS.direct_matches += len(direct_matches)
    TOTAL_STATS.conv_bn_matches += len(conv_bn_matches)
    TOTAL_STATS.direct_replaced += direct_replaced
    TOTAL_STATS.conv_bn_replaced += conv_bn_replaced
    TOTAL_STATS.fused_state_nodes += fused_state_count

    print(
        f"[STATS] lif_state={lif_state_count}, direct_matches={len(direct_matches)}, "
        f"conv_bn_matches={len(conv_bn_matches)}, direct_replaced={direct_replaced}, "
        f"conv_bn_replaced={conv_bn_replaced}, fused_state={fused_state_count}, "
        f"temporal_replaced_windows={temporal_replaced_windows}, "
        f"temporal_replaced_patterns={temporal_replaced_patterns}, "
        f"temporal_skipped_windows={temporal_skipped_windows}, "
        f"temporal_residual_replaced_windows={temporal_residual_replaced_windows}, "
        f"temporal_residual_replaced_patterns={temporal_residual_replaced_patterns}, "
        f"temporal_residual_skipped_windows={temporal_residual_skipped_windows}, "
        f"temporal_lif_rewritten_windows={temporal_lif_rewritten_windows}, "
        f"temporal_lif_replaced_patterns={temporal_lif_replaced_patterns}, "
        f"temporal_lif_skipped_windows={temporal_lif_skipped_windows}, "
        f"fused_temporal_state={fused_temporal_state_count}, "
        f"fused_temporal_residual_state={fused_temporal_residual_state_count}, "
        f"fused_temporal_lif_state={fused_temporal_lif_state_count}"
    )

    if ARGS.backend_mode == "eager":
        return gm.forward
    # Rewrites introduce closed-over placeholders/buffers that do not have
    # Dynamo source metadata. Strip the Dynamo-only source map before handing
    # the mutated graph to AOTAutograd/Inductor.
    gm.meta.pop("dynamo_compile_id", None)
    if hasattr(gm, "_param_name_to_source"):
        delattr(gm, "_param_name_to_source")
    return torch._inductor.compile(
        gm,
        example_inputs,
        options=inductor_options_from_compile_kwargs(compile_kwargs),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Stateful Conv+BN+LIF FX rewrite + Inductor test.")
    parser.add_argument("--model", choices=("tiny-stateful", "resnet18"), default="tiny-stateful")
    parser.add_argument("--T", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend-mode", choices=("eager", "inductor"), default="eager")
    parser.add_argument("--fused-op-backend", choices=("torch", "triton"), default="torch")
    parser.add_argument("--strict-triton", action="store_true")
    parser.add_argument("--disable-rewrite", action="store_true")
    parser.add_argument("--disable-conv-bn-lif", action="store_true")
    parser.add_argument("--disable-temporal-lif-rewrite", action="store_true")
    parser.add_argument("--enable-temporal-rewrite", action="store_true")
    parser.add_argument("--temporal-fuse-window", type=int, default=1)
    parser.add_argument("--temporal-allow-tail", action="store_true")
    parser.add_argument("--enable-temporal-schedule", action="store_true")
    parser.add_argument("--temporal-schedule-window", type=int, default=None)
    parser.add_argument("--temporal-schedule-dump", action="store_true")
    parser.add_argument("--temporal-schedule-strict", action="store_true")
    parser.add_argument("--enable-spatial-batching", action="store_true")
    parser.add_argument(
        "--spatial-batching-ops",
        nargs="+",
        default=["conv", "bn", "add", "maxpool", "avgpool", "flatten", "linear", "elementwise", "view"],
        choices=["conv", "bn", "add", "maxpool", "linear", "flatten", "avgpool", "elementwise", "view"],
    )
    parser.add_argument("--spatial-batching-dump", action="store_true")
    parser.add_argument("--spatial-batching-strict", action="store_true")
    parser.add_argument("--disable-spatial-batching-chain", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--enable-cudagraphs", action="store_true")
    parser.add_argument("--cudagraph-mode", choices=("reduce-overhead", "triton-option", "both"), default="reduce-overhead")
    parser.add_argument("--max-patterns", type=int, default=1)
    parser.add_argument("--print-fused-op-calls", action="store_true")
    parser.add_argument("--out-dir", default="fx_convlif_inductor_debug")
    parser.add_argument("--disable-dag-dump", action="store_true")
    parser.add_argument("--fullgraph", action="store_true")
    parser.add_argument("--print-node-meta", action="store_true")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    return parser.parse_args()


def main():
    global ARGS, OUT_DIR, TOTAL_STATS, CAPTURED_GRAPHS
    ARGS = parse_args()
    OUT_DIR = Path(ARGS.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TOTAL_STATS = RewriteStats()
    CAPTURED_GRAPHS = 0

    device = ARGS.device
    dtype = torch.float32
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device was requested, but torch.cuda.is_available() is False.")

    snn_custom_ops.configure_fused_op(
        backend=ARGS.fused_op_backend,
        strict_triton=ARGS.strict_triton,
        verbose=ARGS.print_fused_op_calls,
    )
    snn_custom_ops.reset_fused_op_call_stats()

    print("=== Stateful ConvLIF FX rewrite + Inductor prototype ===")
    print(f"model={ARGS.model}")
    print(f"device={device}")
    print(f"backend_mode={ARGS.backend_mode}")
    print(f"fused_op_backend={ARGS.fused_op_backend}, strict_triton={ARGS.strict_triton}")
    _, compile_config = build_kairos_compile_config(
        backend="rewrite_backend",
        enable_cudagraphs=ARGS.enable_cudagraphs,
        cudagraph_mode=ARGS.cudagraph_mode,
        fullgraph=ARGS.fullgraph,
        dynamic=False,
    )
    print(f"compile_config={compile_config}")
    print(f"input shape=({ARGS.batch_size}, 3, {ARGS.height}, {ARGS.width})")
    print(f"out dir={OUT_DIR.resolve()}")

    model = build_model(ARGS.model, ARGS.T).to(device=device, dtype=dtype).eval()
    x = torch.randn(ARGS.batch_size, 3, ARGS.height, ARGS.width, device=device, dtype=dtype)
    compiled_model = compile_with_kairos_options(
        model,
        backend=rewrite_backend,
        enable_cudagraphs=ARGS.enable_cudagraphs,
        cudagraph_mode=ARGS.cudagraph_mode,
        fullgraph=ARGS.fullgraph,
        dynamic=False,
    )

    allclose = False
    try:
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        with torch.no_grad():
            reset_custom_stateful_lif_modules(model)
            eager_out = model(x)
            reset_custom_stateful_lif_modules(model)
            compiled_out = compiled_model(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()

        diff = (eager_out - compiled_out).abs()
        allclose = torch.allclose(eager_out, compiled_out, rtol=ARGS.rtol, atol=ARGS.atol)
        print("\n=== Correctness ===")
        print(f"eager shape={tuple(eager_out.shape)} dtype={eager_out.dtype} device={eager_out.device}")
        print(f"compiled shape={tuple(compiled_out.shape)} dtype={compiled_out.dtype} device={compiled_out.device}")
        print(f"max abs diff: {diff.max().item():.6e}")
        print(f"mean abs diff: {diff.mean().item():.6e}")
        print(f"allclose(rtol={ARGS.rtol}, atol={ARGS.atol}): {allclose}")
    except Exception:
        print("\nERROR: stateful ConvLIF FX rewrite experiment failed. Full traceback:")
        traceback.print_exc()
        raise
    finally:
        if device.startswith("cuda") and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                traceback.print_exc()
        call_stats = snn_custom_ops.get_fused_op_call_stats()
        print("\n=== Summary ===")
        print(f"captured graphs: {CAPTURED_GRAPHS}")
        print(f"lif_state nodes: {TOTAL_STATS.lif_state_nodes}")
        print(f"direct conv_lif_state matches: {TOTAL_STATS.direct_matches}")
        print(f"conv_bn_lif_state matches: {TOTAL_STATS.conv_bn_matches}")
        print(f"direct replaced: {TOTAL_STATS.direct_replaced}")
        print(f"conv_bn_lif replaced: {TOTAL_STATS.conv_bn_replaced}")
        print(f"fused_conv_lif_state nodes: {TOTAL_STATS.fused_state_nodes}")
        print(f"fused op calls total: {call_stats['total']}")
        print(f"fused op calls triton: {call_stats['triton']}")
        print(f"fused op calls fallback: {call_stats['fallback']}")
        print(f"allclose: {allclose}")


if __name__ == "__main__":
    main()
