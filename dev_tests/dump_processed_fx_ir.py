import argparse
import json
import operator
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

import runtime.snn_custom_ops as snn_custom_ops

from benchmarks.validate_chronos_baselines import (
    SingleStepModeLoopWrapper,
    build_placeholder_values,
    make_resnet_layer,
)

from compiler.fx_lif_rewrite import count_lif_state_nodes

import compiler.fx_lif_temporal_rewrite as temporal_rewrite

from compiler.fx_lif_temporal_rewrite import (
    collect_conv_bn_add_lif_state_patterns,
    collect_conv_bn_lif_state_patterns,
    count_fused_temporal_conv_add_lif_state_nodes,
    count_fused_temporal_conv_lif_state_nodes,
    dump_temporal_patterns,
    dump_temporal_rewrite_log,
    dump_temporal_windows,
    group_temporal_patterns,
    group_temporal_residual_patterns,
    make_temporal_residual_windows,
    make_temporal_windows,
    rewrite_temporal_conv_bn_add_lif_state_to_fused,
    rewrite_temporal_conv_bn_lif_state_to_fused,
)

from compiler.fx_spatial_batching import (
    SpatialBatchingStats,
    apply_spatial_batching,
    collect_spatial_batch_candidates,
    group_spatial_batch_candidates,
)

from compiler.fx_temporal_annotation import annotate_temporal_metadata
from compiler.fx_temporal_graph_validation import (
    analyze_temporal_graph,
    dump_temporal_graph_validation,
    print_temporal_graph_summary,
)
from compiler.fx_temporal_scheduler import reorder_fx_graph_by_temporal_windows
from compiler.fx_temporal_spatial_canonicalize import canonicalize_temporal_spatial_ir
from benchmarks.helpers.models_for_fx import reset_custom_stateful_lif_modules


CAPTURED_GRAPHS = 0
SUMMARY: Dict[str, Any] = {}


def get_optional_fn(*names):
    for name in names:
        fn = getattr(temporal_rewrite, name, None)
        if fn is not None:
            return fn
    return None


collect_lif_avgpool_linear_patterns = get_optional_fn(
    "collect_temporal_lif_avgpool_linear_patterns",
    "collect_temporal_lif_tail_patterns",
)

group_lif_avgpool_linear_patterns = get_optional_fn(
    "group_temporal_lif_avgpool_linear_patterns",
    "group_temporal_lif_tail_patterns",
)

make_lif_avgpool_linear_windows = get_optional_fn(
    "make_temporal_lif_avgpool_linear_windows",
    "make_temporal_lif_tail_windows",
)

rewrite_lif_avgpool_linear_to_fused = get_optional_fn(
    "rewrite_temporal_lif_avgpool_linear_to_fused",
    "rewrite_temporal_lif_tail_to_fused",
)

count_fused_lif_avgpool_linear_nodes = get_optional_fn(
    "count_fused_temporal_lif_avgpool_linear_nodes",
    "count_fused_temporal_lif_tail_nodes",
)

collect_standalone_lif_patterns = get_optional_fn(
    "collect_standalone_lif_state_patterns",
)

group_standalone_lif_patterns = get_optional_fn(
    "group_temporal_lif_patterns",
)

make_standalone_lif_windows = get_optional_fn(
    "make_temporal_lif_windows",
)

rewrite_standalone_lif_to_fused = get_optional_fn(
    "rewrite_temporal_lif_state_to_fused",
)

count_fused_standalone_lif_nodes = get_optional_fn(
    "count_fused_temporal_lif_state_nodes",
)

collect_linear_lif_patterns = get_optional_fn("collect_temporal_linear_lif_state_patterns")
group_linear_lif_patterns = get_optional_fn("group_temporal_linear_lif_patterns")
make_linear_lif_windows = get_optional_fn("make_temporal_linear_lif_windows")
rewrite_linear_lif_to_fused = get_optional_fn("rewrite_temporal_linear_lif_state_to_fused")
count_fused_linear_lif_nodes = get_optional_fn("count_fused_temporal_linear_lif_state_nodes")


def resolve_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def save_fx(gm: torch.fx.GraphModule, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.py").write_text(gm.code, encoding="utf-8")
    (out_dir / f"{name}.txt").write_text(str(gm.graph), encoding="utf-8")
    print(f"[WRITE] {out_dir / (name + '.py')}")


def target_text(node: torch.fx.Node) -> str:
    return str(node.target)


def is_cat_node(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and (
        node.target is torch.cat or target_text(node) == "cat"
    )


def is_chunk_node(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and (
        node.target is torch.chunk or "chunk" in target_text(node)
    )


def is_getitem_node(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and node.target is operator.getitem


def count_graph_nodes(gm: torch.fx.GraphModule) -> Dict[str, Any]:
    counts = {
        "total_nodes": 0,
        "call_function": 0,
        "call_module": 0,
        "call_method": 0,
        "torch_cat": 0,
        "torch_chunk": 0,
        "operator_getitem": 0,
        "maxpool": 0,
        "avgpool": 0,
        "flatten": 0,
        "linear": 0,
        "lif_forward_state": 0,
        "fused_temporal_convlif": 0,
        "fused_temporal_residual_convlif": 0,
        "fused_temporal_lif_state": 0,
        "fused_temporal_lif_avgpool_linear": 0,
    }

    for node in gm.graph.nodes:
        counts["total_nodes"] += 1

        if node.op in ("call_function", "call_module", "call_method"):
            counts[node.op] += 1

        text = target_text(node)

        if is_cat_node(node):
            counts["torch_cat"] += 1
        if is_chunk_node(node):
            counts["torch_chunk"] += 1
        if is_getitem_node(node):
            counts["operator_getitem"] += 1

        if "max_pool" in text or "MaxPool" in text:
            counts["maxpool"] += 1
        if "avg_pool" in text or "AvgPool" in text or "adaptive_avg_pool" in text:
            counts["avgpool"] += 1
        if "flatten" in text:
            counts["flatten"] += 1
        if "linear" in text or "Linear" in text:
            counts["linear"] += 1

        if "snn_custom.lif_forward_state" in text:
            counts["lif_forward_state"] += 1
        if "snn_custom.fused_temporal_conv_lif_state" in text:
            counts["fused_temporal_convlif"] += 1
        if "snn_custom.fused_temporal_conv_add_lif_state" in text:
            counts["fused_temporal_residual_convlif"] += 1
        if "snn_custom.fused_temporal_lif_state" in text:
            counts["fused_temporal_lif_state"] += 1
        if (
            "snn_custom.fused_temporal_lif_avgpool_linear" in text
            or "snn_custom.fused_temporal_lif_tail" in text
        ):
            counts["fused_temporal_lif_avgpool_linear"] += 1

    return counts


def residual_lif_nodes(gm: torch.fx.GraphModule) -> List[Dict[str, Any]]:
    out = []
    for node in gm.graph.nodes:
        if "snn_custom.lif_forward_state" not in target_text(node):
            continue
        out.append(
            {
                "name": node.name,
                "target": target_text(node),
                "users": [user.name for user in node.users],
                "chronos_timestep": node.meta.get("chronos_timestep"),
                "chronos_window_id": node.meta.get("chronos_window_id"),
                "chronos_role": node.meta.get("chronos_role"),
            }
        )
    return out


def find_repeated_cat_chunk_paths(gm: torch.fx.GraphModule) -> List[Dict[str, str]]:
    paths = []
    for cat in gm.graph.nodes:
        if not is_cat_node(cat):
            continue
        for op_node in list(cat.users):
            for chunk in list(op_node.users):
                if not is_chunk_node(chunk):
                    continue
                for getitem in list(chunk.users):
                    if not is_getitem_node(getitem):
                        continue
                    for next_cat in list(getitem.users):
                        if is_cat_node(next_cat):
                            paths.append(
                                {
                                    "cat": cat.name,
                                    "op": op_node.name,
                                    "chunk": chunk.name,
                                    "getitem": getitem.name,
                                    "next_cat": next_cat.name,
                                }
                            )
    return paths


def stat_value(stats, *names, default=0):
    if stats is None:
        return default
    for name in names:
        if hasattr(stats, name):
            return getattr(stats, name)
    return default


def stat_dict(stats, *names):
    if stats is None:
        return {}
    for name in names:
        if hasattr(stats, name):
            return getattr(stats, name)
    return {}


def build_spatial_group_summary(groups) -> List[Dict[str, Any]]:
    out = []
    for index, group in enumerate(groups):
        out.append(
            {
                "index": index,
                "kind": group.kind,
                "window_id": group.window_id,
                "occurrence": group.occurrence,
                "candidate_nodes": [candidate.node.name for candidate in group.candidates],
                "timesteps": [candidate.timestep for candidate in group.candidates],
            }
        )
    return out


def write_summary(out_dir: Path, summary: Dict[str, Any]):
    json_path = out_dir / "analysis_summary.json"
    md_path = out_dir / "analysis_summary.md"

    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    lines = [
        "# Processed FX IR Summary",
        "",
        "## Final Counts",
    ]

    for key, value in summary.get("final_counts", {}).items():
        lines.append(f"- `{key}`: `{value}`")

    lines.extend(
        [
            "",
            "## Rewrite Stats",
            f"- ConvLIF rewritten windows: `{summary.get('temporal_replaced_windows')}`",
            f"- Residual ConvLIF rewritten windows: `{summary.get('temporal_residual_replaced_windows')}`",
            f"- Residual remapped spike external users: `{summary.get('temporal_residual_remapped_spike_external_users')}`",
            f"- LIF+AvgPool+Linear rewritten windows: `{summary.get('temporal_lif_avgpool_linear_rewritten_windows')}`",
            f"- Standalone LIF rewritten windows: `{summary.get('temporal_lif_rewritten_windows')}`",
            f"- Standalone LIF remapped spike external users: `{summary.get('temporal_lif_remapped_spike_external_users')}`",
            f"- Spatial batch groups: `{summary.get('spatial_batch_groups')}`",
            f"- Spatial batch chains: `{summary.get('spatial_batch_chains')}`",
            f"- Spatial temporal-stack groups: `{summary.get('spatial_temporal_stack_groups')}`",
            f"- Spatial temporal-stack flatten inputs: `{summary.get('spatial_temporal_stack_flatten_inputs')}`",
            f"- Spatial cat avoided by temporal-stack flatten: `{summary.get('spatial_cat_avoided_by_temporal_stack_flatten')}`",
            f"- Spatial previous-batched groups: `{summary.get('spatial_previous_batched_groups')}`",
            f"- Spatial chunk/cat avoided: `{summary.get('spatial_chunk_cat_avoided')}`",
            "",
            "## Structure Checks",
            f"- repeated cat/chunk/cat paths: `{len(summary.get('repeated_cat_chunk_cat_paths', []))}`",
            f"- remaining lif_forward_state nodes: `{len(summary.get('residual_lif_nodes', []))}`",
        ]
    )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[WRITE] {json_path}")
    print(f"[WRITE] {md_path}")


def safe_call(name: str, fn, *args, strict: bool = False, **kwargs):
    if fn is None:
        print(f"[SKIP] {name}: function not found")
        return None
    try:
        return fn(*args, **kwargs)
    except Exception:
        if strict:
            raise
        print(f"[ERROR][{name}]")
        traceback.print_exc()
        return None


def make_backend(args, out_dir: Path):
    def backend(gm: torch.fx.GraphModule, example_inputs, **_compile_kwargs):
        global CAPTURED_GRAPHS, SUMMARY

        graph_idx = CAPTURED_GRAPHS
        CAPTURED_GRAPHS += 1

        graph_dir = out_dir if graph_idx == 0 else out_dir / f"graph_{graph_idx}"
        graph_dir.mkdir(parents=True, exist_ok=True)

        if graph_idx != 0:
            save_fx(gm, graph_dir, "00_original_fx")
            gm.graph.lint()
            gm.recompile()
            return gm.forward

        save_fx(gm, graph_dir, "00_original_fx")

        original_counts = count_graph_nodes(gm)
        lif_state_nodes_before = count_lif_state_nodes(gm)
        placeholder_values = build_placeholder_values(gm, example_inputs)

        annotation_stats = safe_call(
            "temporal_annotation",
            annotate_temporal_metadata,
            gm,
            args.temporal_schedule_window or args.temporal_fuse_window,
            args.T,
            strict=args.strict,
        )
        save_fx(gm, graph_dir, "01_after_temporal_annotation_fx")

        temporal_patterns = collect_conv_bn_lif_state_patterns(gm)
        temporal_groups = group_temporal_patterns(temporal_patterns)
        dump_temporal_patterns(temporal_groups, graph_dir / "temporal_patterns.txt")

        schedule_result = safe_call(
            "temporal_schedule",
            reorder_fx_graph_by_temporal_windows,
            gm,
            args.T,
            args.temporal_schedule_window,
            temporal_patterns,
            dump_dir=graph_dir,
            strict=args.strict,
        )

        if schedule_result is not None and getattr(schedule_result, "ok", False):
            temporal_patterns = collect_conv_bn_lif_state_patterns(gm)
            temporal_groups = group_temporal_patterns(temporal_patterns)

        save_fx(gm, graph_dir, "02_after_temporal_schedule_fx")

        spatial_stats = SpatialBatchingStats()
        spatial_groups = []

        temporal_windows = make_temporal_windows(
            temporal_groups,
            args.temporal_fuse_window,
            allow_tail=False,
        )
        dump_temporal_windows(temporal_windows, graph_dir / "temporal_windows.txt")

        temporal_stats = safe_call(
            "rewrite_temporal_convlif",
            rewrite_temporal_conv_bn_lif_state_to_fused,
            gm,
            temporal_windows,
            placeholder_values,
            max_patterns=1000000,
            strict=args.strict,
        )

        if temporal_stats is not None:
            dump_temporal_rewrite_log(
                temporal_stats.log,
                graph_dir / "temporal_rewrite_log.txt",
            )

        gm.graph.lint()
        gm.recompile()
        save_fx(gm, graph_dir, "04_after_convlif_rewrite_fx")

        residual_patterns = collect_conv_bn_add_lif_state_patterns(gm)
        residual_groups = group_temporal_residual_patterns(residual_patterns)
        residual_windows = make_temporal_residual_windows(
            residual_groups,
            args.temporal_fuse_window,
            allow_tail=False,
        )

        residual_stats = safe_call(
            "rewrite_temporal_residual_convlif",
            rewrite_temporal_conv_bn_add_lif_state_to_fused,
            gm,
            residual_windows,
            placeholder_values,
            max_patterns=1000000,
            strict=args.strict,
        )

        gm.graph.lint()
        gm.recompile()
        save_fx(gm, graph_dir, "05_after_residual_convlif_rewrite_fx")

        linear_lif_stats = None
        linear_lif_groups = []
        linear_lif_windows = []

        if not args.disable_temporal_linear_lif_rewrite and collect_linear_lif_patterns is not None:
            linear_lif_patterns = safe_call(
                "collect_linear_lif_patterns",
                collect_linear_lif_patterns,
                gm,
                strict=args.strict,
            )
            if linear_lif_patterns is not None:
                linear_lif_groups = safe_call(
                    "group_linear_lif_patterns",
                    group_linear_lif_patterns,
                    linear_lif_patterns,
                    strict=args.strict,
                ) or []
                linear_lif_windows = safe_call(
                    "make_linear_lif_windows",
                    make_linear_lif_windows,
                    linear_lif_groups,
                    args.temporal_fuse_window,
                    allow_tail=False,
                    strict=args.strict,
                ) or []
                linear_lif_stats = safe_call(
                    "rewrite_linear_lif_to_fused",
                    rewrite_linear_lif_to_fused,
                    gm,
                    linear_lif_windows,
                    max_patterns=1000000,
                    strict=args.strict,
                )

        gm.graph.lint()
        gm.recompile()
        save_fx(gm, graph_dir, "06_after_linear_lif_rewrite_fx")

        lif_avgpool_linear_stats = None
        lif_avgpool_linear_groups = []
        lif_avgpool_linear_windows = []

        if not args.disable_temporal_lif_avgpool_linear_rewrite:
            lif_avgpool_linear_patterns = safe_call(
                "collect_lif_avgpool_linear_patterns",
                collect_lif_avgpool_linear_patterns,
                gm,
                strict=args.strict,
            )

            if lif_avgpool_linear_patterns is not None:
                lif_avgpool_linear_groups = safe_call(
                    "group_lif_avgpool_linear_patterns",
                    group_lif_avgpool_linear_patterns,
                    lif_avgpool_linear_patterns,
                    strict=args.strict,
                ) or []

                lif_avgpool_linear_windows = safe_call(
                    "make_lif_avgpool_linear_windows",
                    make_lif_avgpool_linear_windows,
                    lif_avgpool_linear_groups,
                    args.temporal_fuse_window,
                    allow_tail=False,
                    strict=args.strict,
                ) or []

                lif_avgpool_linear_stats = safe_call(
                    "rewrite_lif_avgpool_linear_to_fused",
                    rewrite_lif_avgpool_linear_to_fused,
                    gm,
                    lif_avgpool_linear_windows,
                    max_patterns=1000000,
                    strict=args.strict,
                )

        gm.graph.lint()
        gm.recompile()
        save_fx(gm, graph_dir, "07_after_lif_avgpool_linear_rewrite_fx")

        standalone_lif_stats = None
        standalone_lif_groups = []
        standalone_lif_windows = []

        if not args.disable_temporal_lif_rewrite:
            standalone_lif_patterns = safe_call(
                "collect_standalone_lif_patterns",
                collect_standalone_lif_patterns,
                gm,
                strict=args.strict,
            )

            if standalone_lif_patterns is not None:
                standalone_lif_groups = safe_call(
                    "group_standalone_lif_patterns",
                    group_standalone_lif_patterns,
                    standalone_lif_patterns,
                    strict=args.strict,
                ) or []

                standalone_lif_windows = safe_call(
                    "make_standalone_lif_windows",
                    make_standalone_lif_windows,
                    standalone_lif_groups,
                    args.temporal_fuse_window,
                    allow_tail=False,
                    strict=args.strict,
                ) or []

                standalone_lif_stats = safe_call(
                    "rewrite_standalone_lif_to_fused",
                    rewrite_standalone_lif_to_fused,
                    gm,
                    standalone_lif_windows,
                    max_patterns=1000000,
                    strict=args.strict,
                )

        gm.graph.lint()
        gm.recompile()
        save_fx(gm, graph_dir, "08_after_standalone_lif_rewrite_fx")

        if args.enable_spatial_batching:
            pre_stats = SpatialBatchingStats()
            candidates = safe_call(
                "collect_spatial_batch_candidates",
                collect_spatial_batch_candidates,
                gm,
                args.temporal_schedule_window or args.temporal_fuse_window,
                args.spatial_batching_ops,
                pre_stats,
                strict=args.strict,
            )
            if candidates is not None:
                spatial_groups = safe_call(
                    "group_spatial_batch_candidates",
                    group_spatial_batch_candidates,
                    candidates,
                    args.temporal_schedule_window or args.temporal_fuse_window,
                    pre_stats,
                    strict=args.strict,
                ) or []
            spatial_stats = safe_call(
                "apply_spatial_batching",
                apply_spatial_batching,
                gm,
                args.temporal_schedule_window or args.temporal_fuse_window,
                args.spatial_batching_ops,
                dump_dir=graph_dir,
                strict=args.strict,
                enable_chain=False,
            ) or SpatialBatchingStats()
        gm.graph.lint()
        gm.recompile()
        save_fx(gm, graph_dir, "09_after_spatial_batching_fx")

        canonicalize_stats = safe_call(
            "canonicalize_temporal_spatial_ir",
            canonicalize_temporal_spatial_ir,
            gm,
            dump_dir=graph_dir,
            strict=args.strict,
        )
        gm.graph.lint()
        gm.recompile()
        save_fx(gm, graph_dir, "10_after_canonicalize_fx")

        temporal_graph_stats = analyze_temporal_graph(gm)
        print_temporal_graph_summary(temporal_graph_stats)
        dump_temporal_graph_validation(temporal_graph_stats, graph_dir / "temporal_graph_validation.json")

        final_graph_path = graph_dir / "11_final_readable_graph.txt"
        final_graph_path.write_text(str(gm.graph), encoding="utf-8")
        print(f"[WRITE] {final_graph_path}")

        final_counts = count_graph_nodes(gm)
        repeated_paths = find_repeated_cat_chunk_paths(gm)
        residual_lifs = residual_lif_nodes(gm)

        SUMMARY = {
            "captured_graphs": CAPTURED_GRAPHS,
            "args": vars(args),
            "original_counts": original_counts,
            "final_counts": final_counts,
            "lif_state_nodes_before": lif_state_nodes_before,
            "fused_temporal_convlif_nodes": count_fused_temporal_conv_lif_state_nodes(gm),
            "fused_temporal_residual_convlif_nodes": count_fused_temporal_conv_add_lif_state_nodes(gm),
            "fused_temporal_lif_avgpool_linear_nodes": (
                count_fused_lif_avgpool_linear_nodes(gm)
                if count_fused_lif_avgpool_linear_nodes is not None
                else final_counts["fused_temporal_lif_avgpool_linear"]
            ),
            "fused_temporal_lif_nodes": (
                count_fused_standalone_lif_nodes(gm)
                if count_fused_standalone_lif_nodes is not None
                else final_counts["fused_temporal_lif_state"]
            ),
            "fused_temporal_linear_lif_nodes": (
                count_fused_linear_lif_nodes(gm)
                if count_fused_linear_lif_nodes is not None
                else final_counts.get("fused_temporal_linear_lif_state", 0)
            ),
            "temporal_annotation": asdict(annotation_stats) if annotation_stats is not None else None,
            "temporal_schedule": asdict(schedule_result) if schedule_result is not None else None,
            "temporal_groups": len(temporal_groups),
            "temporal_windows": len(temporal_windows),
            "temporal_replaced_windows": stat_value(temporal_stats, "temporal_replaced_windows"),
            "temporal_replaced_patterns": stat_value(temporal_stats, "temporal_replaced_patterns"),
            "temporal_skipped_windows": stat_value(temporal_stats, "temporal_skipped_windows"),
            "temporal_residual_groups": len(residual_groups),
            "temporal_residual_windows": len(residual_windows),
            "temporal_residual_replaced_windows": stat_value(
                residual_stats,
                "temporal_residual_replaced_windows",
                "temporal_residual_rewritten_windows",
            ),
            "temporal_residual_replaced_patterns": stat_value(
                residual_stats,
                "temporal_residual_replaced_patterns",
            ),
            "temporal_residual_skipped_windows": stat_value(
                residual_stats,
                "temporal_residual_skipped_windows",
            ),
            "temporal_residual_remapped_spike_external_users": stat_value(
                residual_stats,
                "temporal_residual_remapped_spike_external_users",
            ),
            "temporal_residual_unremappable_external_users": stat_value(
                residual_stats,
                "temporal_residual_unremappable_external_users",
            ),
            "temporal_residual_skip_reasons": stat_dict(
                residual_stats,
                "residual_fuse_skip_reasons",
                "temporal_residual_skip_reasons",
            ),
            "linear_lif_patterns": sum(len(group.patterns) for group in linear_lif_groups),
            "temporal_linear_lif_windows": len(linear_lif_windows),
            "temporal_linear_lif_rewritten_windows": stat_value(
                linear_lif_stats,
                "temporal_linear_lif_rewritten_windows",
            ),
            "temporal_linear_lif_replaced_patterns": stat_value(
                linear_lif_stats,
                "temporal_linear_lif_replaced_patterns",
            ),
            "temporal_linear_lif_skipped_windows": stat_value(
                linear_lif_stats,
                "temporal_linear_lif_skipped_windows",
            ),
            "temporal_linear_lif_skip_reasons": stat_dict(
                linear_lif_stats,
                "temporal_linear_lif_skip_reasons",
            ),
            "temporal_lif_avgpool_linear_groups": len(lif_avgpool_linear_groups),
            "temporal_lif_avgpool_linear_windows": len(lif_avgpool_linear_windows),
            "temporal_lif_avgpool_linear_rewritten_windows": stat_value(
                lif_avgpool_linear_stats,
                "temporal_lif_avgpool_linear_rewritten_windows",
                "temporal_lif_tail_rewritten_windows",
                "temporal_lif_avgpool_linear_replaced_windows",
                "temporal_lif_tail_replaced_windows",
            ),
            "temporal_lif_avgpool_linear_skipped_windows": stat_value(
                lif_avgpool_linear_stats,
                "temporal_lif_avgpool_linear_skipped_windows",
                "temporal_lif_tail_skipped_windows",
            ),
            "temporal_lif_avgpool_linear_skip_reasons": stat_dict(
                lif_avgpool_linear_stats,
                "temporal_lif_avgpool_linear_skip_reasons",
                "temporal_lif_tail_skip_reasons",
                "lif_avgpool_linear_skip_reasons",
                "lif_tail_skip_reasons",
            ),
            "temporal_lif_groups": len(standalone_lif_groups),
            "temporal_lif_windows": len(standalone_lif_windows),
            "temporal_lif_rewritten_windows": stat_value(
                standalone_lif_stats,
                "temporal_lif_rewritten_windows",
                "temporal_lif_replaced_windows",
            ),
            "temporal_lif_skipped_windows": stat_value(
                standalone_lif_stats,
                "temporal_lif_skipped_windows",
            ),
            "temporal_lif_remapped_spike_external_users": stat_value(
                standalone_lif_stats,
                "temporal_lif_remapped_spike_external_users",
            ),
            "temporal_lif_unremappable_external_users": stat_value(
                standalone_lif_stats,
                "temporal_lif_unremappable_external_users",
            ),
            "temporal_lif_skip_reasons": stat_dict(
                standalone_lif_stats,
                "temporal_lif_skip_reasons",
                "lif_fuse_skip_reasons",
            ),
            "spatial_batch_groups": spatial_stats.spatial_batch_groups,
            "spatial_batched_ops": spatial_stats.spatial_batched_ops,
            "spatial_batch_chains": spatial_stats.spatial_batch_chains,
            "spatial_chain_groups": spatial_stats.spatial_chain_groups,
            "spatial_cat_eliminated": spatial_stats.spatial_cat_eliminated,
            "spatial_chunk_eliminated": spatial_stats.spatial_chunk_eliminated,
            "spatial_batched_conv": spatial_stats.spatial_batched_conv,
            "spatial_batched_bn": spatial_stats.spatial_batched_bn,
            "spatial_batched_add": spatial_stats.spatial_batched_add,
            "spatial_batched_pool": spatial_stats.spatial_batched_pool,
            "spatial_batched_maxpool": spatial_stats.spatial_batched_maxpool,
            "spatial_batched_avgpool": spatial_stats.spatial_batched_avgpool,
            "spatial_batched_adaptive_avgpool": spatial_stats.spatial_batched_adaptive_avgpool,
            "spatial_batched_flatten": spatial_stats.spatial_batched_flatten,
            "spatial_batched_linear": spatial_stats.spatial_batched_linear,
            "spatial_batched_elementwise": spatial_stats.spatial_batched_elementwise,
            "spatial_temporal_stack_bn_groups": spatial_stats.spatial_temporal_stack_bn_groups,
            "spatial_temporal_stack_add_groups": spatial_stats.spatial_temporal_stack_add_groups,
            "spatial_temporal_stack_pool_groups": spatial_stats.spatial_temporal_stack_pool_groups,
            "spatial_temporal_stack_flatten_groups": spatial_stats.spatial_temporal_stack_flatten_groups,
            "spatial_temporal_stack_linear_groups": spatial_stats.spatial_temporal_stack_linear_groups,
            "spatial_temporal_stack_groups": spatial_stats.spatial_temporal_stack_groups,
            "spatial_temporal_stack_flatten_inputs": spatial_stats.spatial_temporal_stack_flatten_inputs,
            "spatial_cat_avoided_by_temporal_stack_flatten": spatial_stats.spatial_cat_avoided_by_temporal_stack_flatten,
            "spatial_previous_batched_groups": spatial_stats.spatial_previous_batched_groups,
            "spatial_reused_previous_batched_inputs": spatial_stats.spatial_reused_previous_batched_inputs,
            "spatial_chunk_cat_avoided": spatial_stats.spatial_chunk_cat_avoided,
            "spatial_batch_skipped": spatial_stats.spatial_batch_skipped,
            "spatial_batch_reasons": spatial_stats.reasons,
            "canonicalize": asdict(canonicalize_stats) if canonicalize_stats is not None else None,
            "temporal_graph_validation": asdict(temporal_graph_stats),
            "spatial_batch_group_details": build_spatial_group_summary(spatial_groups),
            "repeated_cat_chunk_cat_paths": repeated_paths,
            "residual_lif_nodes": residual_lifs,
        }

        write_summary(graph_dir, SUMMARY)

        return gm.forward

    return backend


def parse_args():
    parser = argparse.ArgumentParser(description="Dump processed Chronos FX IR.")

    parser.add_argument("--model", default="resnet18", choices=["resnet18", "resnet34", "mobilenetv1", "mobilenetv2"])
    parser.add_argument("--T", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("fp32", "fp16"), default="fp32")

    parser.add_argument("--temporal-fuse-window", type=int, default=16)
    parser.add_argument("--temporal-schedule-window", type=int, default=16)

    parser.add_argument("--disable-temporal-lif-avgpool-linear-rewrite", action="store_true")
    parser.add_argument("--disable-temporal-lif-rewrite", action="store_true")
    parser.add_argument("--disable-temporal-linear-lif-rewrite", action="store_true")

    parser.add_argument("--enable-spatial-batching", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument(
        "--spatial-batching-ops",
        nargs="+",
        default=["conv", "bn", "add", "maxpool", "avgpool", "flatten", "linear", "elementwise", "view"],
        choices=["conv", "bn", "add", "maxpool", "avgpool", "flatten", "linear", "elementwise", "view"],
    )

    parser.add_argument("--disable-spatial-batching-chain", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--out-dir", default="test/fx_ir_dump_processed")
    parser.add_argument("--strict", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = resolve_dtype(args.dtype)

    snn_custom_ops.configure_fused_op(
        backend="torch",
        strict_triton=False,
        verbose=False,
    )

    layer = make_resnet_layer(
        model_name=args.model,
        allow_resnet32_fallback=True,
        step_mode="s",
    ).to(device=args.device, dtype=dtype).eval()

    model = SingleStepModeLoopWrapper(
        layer=layer,
        T=args.T,
    ).to(device=args.device, dtype=dtype).eval()

    x = torch.randn(
        args.batch_size,
        3,
        args.height,
        args.width,
        device=args.device,
        dtype=dtype,
    )

    reset_custom_stateful_lif_modules(model)

    compiled = torch.compile(
        model,
        backend=make_backend(args, out_dir),
        fullgraph=False,
        dynamic=False,
    )

    with torch.no_grad():
        _ = compiled(x)

    if args.device.startswith("cuda"):
        torch.cuda.synchronize()

    if SUMMARY:
        final_counts = SUMMARY["final_counts"]

        print("\n=== FX IR Dump Summary ===")
        print(f"model: {args.model}")
        print(f"total nodes: {final_counts['total_nodes']}")
        print(
            "cat/chunk/getitem: "
            f"{final_counts['torch_cat']}/"
            f"{final_counts['torch_chunk']}/"
            f"{final_counts['operator_getitem']}"
        )
        print(f"remaining lif_forward_state: {final_counts['lif_forward_state']}")
        print(f"fused temporal ConvLIF: {final_counts['fused_temporal_convlif']}")
        print(f"fused temporal Residual ConvLIF: {final_counts['fused_temporal_residual_convlif']}")
        print(f"fused temporal LIF: {final_counts['fused_temporal_lif_state']}")
        print(
            "fused temporal LIF+AvgPool+Linear: "
            f"{final_counts['fused_temporal_lif_avgpool_linear']}"
        )
        print(f"spatial groups: {SUMMARY['spatial_batch_groups']}")
        print(f"spatial chains: {SUMMARY['spatial_batch_chains']}")
        print(f"repeated cat/chunk/cat paths: {len(SUMMARY['repeated_cat_chunk_cat_paths'])}")
        print(f"summary: {out_dir / 'analysis_summary.json'}")
    else:
        print("WARNING: no graph was captured.")


if __name__ == "__main__":
    main()
