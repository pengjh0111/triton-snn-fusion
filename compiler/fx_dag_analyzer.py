import json
import operator
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FxDagNode:
    node_id: int
    name: str
    op: str
    target: str
    kind: str
    inputs: List[str]
    users: List[str]
    args_repr: str
    kwargs_repr: str
    shape: Optional[Tuple[int, ...]]
    dtype: Optional[str]
    is_placeholder: bool
    is_output: bool
    is_param_or_buffer: bool
    is_conv: bool
    is_batch_norm: bool
    is_lif: bool
    is_fused_convlif: bool
    is_getitem: bool
    is_add: bool
    is_pool: bool
    is_linear: bool


@dataclass
class FxDag:
    nodes: Dict[str, FxDagNode]
    topo_order: List[str]
    edges: List[Tuple[str, str]]


def collect_input_nodes(obj) -> List[torch.fx.Node]:
    nodes = []
    if isinstance(obj, torch.fx.Node):
        return [obj]
    if isinstance(obj, (tuple, list)):
        for item in obj:
            nodes.extend(collect_input_nodes(item))
        return nodes
    if isinstance(obj, dict):
        for item in obj.values():
            nodes.extend(collect_input_nodes(item))
        return nodes
    return nodes


def _target_text(node: torch.fx.Node) -> str:
    return str(node.target)


def _is_aten(target, name: str) -> bool:
    return str(target) == name


def _is_module(gm: torch.fx.GraphModule, node: torch.fx.Node, module_type) -> bool:
    if node.op != "call_module":
        return False
    try:
        return isinstance(gm.get_submodule(str(node.target)), module_type)
    except AttributeError:
        return False


def _is_conv(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    if _is_module(gm, node, nn.Conv2d):
        return True
    if node.op != "call_function":
        return False
    return node.target in (F.conv2d, torch.conv2d) or _is_aten(node.target, "aten.convolution.default")


def _is_batch_norm(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    if _is_module(gm, node, nn.BatchNorm2d):
        return True
    return node.op == "call_function" and node.target is F.batch_norm


def _is_lif(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and str(node.target) in (
        "snn_custom.lif_forward.default",
        "snn_custom.lif_forward_state.default",
    )


def _is_lif_state(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and str(node.target) == "snn_custom.lif_forward_state.default"


def _is_fused_convlif(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and str(node.target) in (
        "snn_custom.fused_conv_lif_forward.default",
        "snn_custom.fused_conv_lif_forward_state.default",
        "snn_custom.fused_conv_lif_state.default",
        "snn_custom.fused_temporal_conv_lif_state.default",
    )


def _is_fused_convlif_state(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and str(node.target) in (
        "snn_custom.fused_conv_lif_forward_state.default",
        "snn_custom.fused_conv_lif_state.default",
        "snn_custom.fused_temporal_conv_lif_state.default",
    )


def _is_getitem(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and node.target is operator.getitem


def _is_add(node: torch.fx.Node) -> bool:
    if node.op != "call_function":
        return False
    return node.target in (operator.add, operator.iadd, torch.add) or str(node.target) in (
        "aten.add.Tensor",
        "<built-in function add>",
        "<built-in function iadd>",
    )


def _is_pool(node: torch.fx.Node) -> bool:
    if node.op == "call_module":
        return "Pool" in str(node.target)
    if node.op != "call_function":
        return False
    text = str(node.target)
    return "max_pool2d" in text or "adaptive_avg_pool2d" in text


def _is_linear(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    if _is_module(gm, node, nn.Linear):
        return True
    if node.op != "call_function":
        return False
    text = str(node.target)
    return node.target is F.linear or "linear" in text


def classify_node(gm: torch.fx.GraphModule, node: torch.fx.Node) -> str:
    if node.op == "placeholder":
        return "placeholder"
    if node.op == "output":
        return "output"
    if node.op == "get_attr":
        return "get_attr"
    if _is_conv(gm, node):
        return "conv2d"
    if _is_batch_norm(gm, node):
        return "batch_norm"
    if _is_lif_state(node):
        return "lif_state"
    if _is_lif(node):
        return "lif"
    if _is_fused_convlif_state(node):
        return "fused_convlif_state"
    if _is_fused_convlif(node):
        return "fused_convlif"
    if _is_getitem(node):
        return "getitem"
    if _is_add(node):
        return "add"
    if node.op == "call_function" and "relu" in str(node.target):
        return "relu"
    if _is_pool(node):
        return "pool"
    if node.op == "call_function" and "flatten" in str(node.target):
        return "flatten"
    if _is_linear(gm, node):
        return "linear"
    return "unknown"


def get_tensor_meta(node: torch.fx.Node) -> Tuple[Optional[Tuple[int, ...]], Optional[str]]:
    meta = node.meta.get("tensor_meta")
    if meta is not None:
        shape = getattr(meta, "shape", None)
        dtype = getattr(meta, "dtype", None)
        return tuple(shape) if shape is not None else None, str(dtype) if dtype is not None else None

    val = node.meta.get("val")
    if isinstance(val, torch.Tensor):
        return tuple(val.shape), str(val.dtype)
    if hasattr(val, "shape") and hasattr(val, "dtype"):
        return tuple(val.shape), str(val.dtype)
    return None, None


def build_fx_dag(gm: torch.fx.GraphModule) -> FxDag:
    dag_nodes: Dict[str, FxDagNode] = {}
    topo_order: List[str] = []
    edges: List[Tuple[str, str]] = []

    for node_id, node in enumerate(gm.graph.nodes):
        inputs = []
        for dep in collect_input_nodes((node.args, node.kwargs)):
            inputs.append(dep.name)
            edges.append((dep.name, node.name))

        users = [user.name for user in node.users]
        kind = classify_node(gm, node)
        shape, dtype = get_tensor_meta(node)
        dag_nodes[node.name] = FxDagNode(
            node_id=node_id,
            name=node.name,
            op=node.op,
            target=_target_text(node),
            kind=kind,
            inputs=inputs,
            users=users,
            args_repr=repr(node.args),
            kwargs_repr=repr(node.kwargs),
            shape=shape,
            dtype=dtype,
            is_placeholder=node.op == "placeholder",
            is_output=node.op == "output",
            is_param_or_buffer=node.op == "get_attr" or (node.op == "placeholder" and "parameters_" in node.name or "buffers_" in node.name),
            is_conv=kind == "conv2d",
            is_batch_norm=kind == "batch_norm",
            is_lif=kind in ("lif", "lif_state"),
            is_fused_convlif=kind in ("fused_convlif", "fused_convlif_state"),
            is_getitem=kind == "getitem",
            is_add=kind == "add",
            is_pool=kind == "pool",
            is_linear=kind == "linear",
        )
        topo_order.append(node.name)

    return FxDag(nodes=dag_nodes, topo_order=topo_order, edges=edges)


def _ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def dump_fx_dag_text(dag: FxDag, path: Path):
    _ensure_parent(path)
    lines = []
    for name in dag.topo_order:
        node = dag.nodes[name]
        lines.append(f"[{node.node_id}] name={node.name}")
        lines.append(f"    kind={node.kind}")
        lines.append(f"    op={node.op}")
        lines.append(f"    target={node.target}")
        lines.append(f"    shape={node.shape}")
        lines.append(f"    dtype={node.dtype}")
        lines.append(f"    inputs={node.inputs}")
        lines.append(f"    users={node.users}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def dump_fx_dag_edges(dag: FxDag, path: Path):
    _ensure_parent(path)
    path.write_text("\n".join(f"{src} -> {dst}" for src, dst in dag.edges) + "\n", encoding="utf-8")


def _jsonable(value):
    if isinstance(value, tuple):
        return list(value)
    return value


def dump_fx_dag_json(dag: FxDag, path: Path):
    _ensure_parent(path)
    payload = {
        "nodes": {name: {key: _jsonable(value) for key, value in asdict(node).items()} for name, node in dag.nodes.items()},
        "edges": [[src, dst] for src, dst in dag.edges],
        "topo_order": dag.topo_order,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def summarize_fx_dag(dag: FxDag) -> Dict[str, Any]:
    nodes = list(dag.nodes.values())
    return {
        "total_nodes": len(nodes),
        "total_edges": len(dag.edges),
        "num_placeholders": sum(node.is_placeholder for node in nodes),
        "num_get_attrs": sum(node.op == "get_attr" for node in nodes),
        "num_conv": sum(node.is_conv for node in nodes),
        "num_batch_norm": sum(node.is_batch_norm for node in nodes),
        "num_lif": sum(node.kind == "lif" for node in nodes),
        "num_lif_state": sum(node.kind == "lif_state" for node in nodes),
        "num_fused_convlif": sum(node.kind == "fused_convlif" for node in nodes),
        "num_fused_convlif_state": sum(node.kind == "fused_convlif_state" for node in nodes),
        "num_add": sum(node.is_add for node in nodes),
        "num_pool": sum(node.is_pool for node in nodes),
        "num_linear": sum(node.is_linear for node in nodes),
        "num_output": sum(node.is_output for node in nodes),
    }


def find_fused_convlif_regions(dag: FxDag) -> List[List[str]]:
    regions = []
    for name in dag.topo_order:
        if dag.nodes[name].kind in ("fused_convlif", "fused_convlif_state"):
            regions.append([name])
    return regions


def dump_fx_dag_regions(regions: List[List[str]], path: Path):
    _ensure_parent(path)
    lines = []
    for idx, region in enumerate(regions):
        lines.append(f"region_{idx}:")
        for node_name in region:
            lines.append(f"  {node_name}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _dot_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    return (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "")
    )


def _dot_color(kind: str) -> str:
    colors = {
        "placeholder": "lightgray",
        "get_attr": "gray",
        "conv2d": "lightblue",
        "batch_norm": "plum",
        "lif": "orange",
        "lif_state": "orange",
        "fused_convlif": "lightgreen",
        "fused_convlif_state": "lightgreen",
        "add": "khaki",
        "pool": "cyan",
        "linear": "pink",
        "output": "salmon",
        "unknown": "white",
    }
    return colors.get(kind, "white")


def dump_fx_dag_dot(dag: FxDag, path: Path):
    _ensure_parent(path)
    lines = [
        "digraph FX_DAG {",
        "    rankdir=LR;",
        '    node [shape=box, style="rounded,filled"];',
    ]
    for name in dag.topo_order:
        node = dag.nodes[name]
        label = f"{node.node_id}: {node.name}\\n{node.kind}\\nshape={node.shape}"
        lines.append(
            f'    "{_dot_escape(node.name)}" '
            f'[label="{_dot_escape(label)}", fillcolor="{_dot_color(node.kind)}"];'
        )
    for src, dst in dag.edges:
        lines.append(f'    "{_dot_escape(src)}" -> "{_dot_escape(dst)}";')
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def try_dump_fx_dag_svg(dag: FxDag, dot_path: Path, svg_path: Path):
    dump_fx_dag_dot(dag, dot_path)
    dot_bin = shutil.which("dot")
    if dot_bin is None:
        print("WARNING: graphviz 'dot' command not found; skipped FX DAG SVG generation.")
        return
    _ensure_parent(svg_path)
    try:
        subprocess.run([dot_bin, "-Tsvg", str(dot_path), "-o", str(svg_path)], check=True)
    except Exception as exc:
        print(f"WARNING: failed to generate FX DAG SVG: {exc}")
