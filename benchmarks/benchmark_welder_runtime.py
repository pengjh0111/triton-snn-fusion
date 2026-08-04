"""Compile and benchmark the 13 Kairos workloads with Welder (OSDI'23).

Model construction and ONNX export follow benchmark_bladedisc_runtime.py's
build_case()/export_onnx() exactly (same single-step/sequence wrapper
dispatch as the TVM and TensorRT baselines) -- reused directly rather than
re-derived, since it already returns an export-ready (LIF-decomposed) model.

Unlike TVM (in-process Relay API) and TensorRT (trtexec CLI), Welder is
driven as a sequence of subprocess calls against the `nnfusion` binary, a
separate `run_compiler` module invocation, and a generated CMake project --
mirroring nnfusion/artifacts/tune_welder.py and nnfusion/testing/run_welder.py
from the welder branch of https://github.com/microsoft/nnfusion. The
`run_compiler` step must run under a dedicated Python environment that has
Welder's own (patched) TVM fork on PYTHONPATH; the rest of this driver runs
under the normal Chronos interpreter, same as the other baselines.
"""

import argparse
import copy
import ctypes
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict

import numpy as np
import onnx
from onnx import helper as onnx_helper
from onnx import numpy_helper
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.benchmark_tensorrt_runtime import (  # noqa: E402
    onnx_graph_contains_custom_lif,
    resolve_dtype,
)
from benchmarks.benchmark_bladedisc_runtime import (  # noqa: E402
    build_case,
    export_onnx,
)
from benchmarks.validate_kairos_baselines import (  # noqa: E402
    KAIROS_MODEL_CHOICES,
    LIF_IMPL_CHOICES,
    reset_lif_modules,
)


################################################################################
# defaults for the RTX 5090 welder build (see docs/welder_rtx5090_setup, or
# the incremental-build notes in this PR, for how these were produced)
################################################################################

DEFAULT_WELDER_REPO = "/data/nnfusion"
DEFAULT_WELDER_TVM_PYTHONPATH = "/data/welder-deps/tvm-welder/python"
DEFAULT_NNFUSION_BIN_DIR = "/data/welder-deps/nnfusion-welder-cpp/build/src/tools/nnfusion"
DEFAULT_WELDER_PYTHON = "/data/welder-deps/envs/welder-5090-venv/bin/python"
DEFAULT_CUDA_HOME = "/usr/local/cuda-12.8"


################################################################################
# helpers
################################################################################

def summarize_ms(times) -> Dict[str, float]:
    arr = np.asarray(times, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def run_logged(cmd, cwd, env, log_path: Path, timeout=None):
    """Returns (returncode, elapsed_seconds). returncode is None on timeout."""
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n$ " + " ".join(str(c) for c in cmd) + "\n")
        f.flush()
        start = time.monotonic()
        try:
            proc = subprocess.run(
                cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT,
                timeout=timeout, check=False,
            )
            return proc.returncode, time.monotonic() - start
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - start
            f.write(f"\n[TIMEOUT] after {timeout}s\n")
            return None, elapsed


def infer_output_shape(model, x) -> Dict[str, Any]:
    reset_lif_modules(model)
    with torch.no_grad():
        y = model(x)
    return {"shape": list(y.shape), "dtype": str(y.dtype).replace("torch.", "")}


TORCH_DTYPE_BY_NAME = {
    "float32": torch.float32,
    "float16": torch.float16,
    "int64": torch.int64,
}


def save_onnx(model: onnx.ModelProto, onnx_path: Path) -> None:
    """Save a (possibly just-loaded-with-external-data) model back to onnx_path.

    onnx.load() inlines external-data tensors into the in-memory proto by
    default, so a plain onnx.save() on a large model (vgg/mobilenet's
    hundreds of MB of conv weights) tries to protobuf-serialize the whole
    thing as one string and hits protobuf's hard 2GB message-size limit
    (`EncodeError: Failed to serialize proto`) -- even though torch's own
    exporter avoided this by writing weights as external-data files
    alongside a small `.onnx` shell in the first place. save_as_external_data
    restores that split: the shell (structure + external-data pointers)
    serializes fine, and tensor bytes go straight to files via plain I/O,
    never passing through the single-string protobuf path.

    Deliberately does NOT pass convert_attribute=True: that would also
    externalize large Constant-node `value` attributes (which is where
    do_constant_folding=False leaves torch's weights, instead of graph
    initializers) and let this call succeed -- but NNFusion's importer only
    resolves external data for graph.initializer entries
    (graph_convert.cpp's move_external_to_rawdata is called exclusively from
    the `for (tensor : onnx_graph_proto->initializer())` loop, never for node
    attributes), so it would silently read those tensors back as empty
    (`Did not get the expected number of literals ... got 0`). Large weights
    need to already be initializers by the time this runs --
    promote_large_constants_to_initializers (below) does that, once, right
    after export.

    This pipeline calls save_onnx() repeatedly (once per rewrite pass that
    touches the model), always at the same external-data location -- but
    onnx's writer (external_data_helper.save_external_data) opens that file
    with "r+b" and seeks to EOF before writing, never truncating first. Left
    alone, every single call re-appends a full copy of every large tensor
    onto whatever the previous pass already wrote there, growing the file by
    roughly one model's worth of weights per pass (confirmed: this alone
    inflated vgg16/mobilenet's output to 20+GB and filled the disk).
    Unlinking any stale file from a prior save right before writing keeps
    each save's external-data file to exactly one copy.
    """
    external_data_path = onnx_path.with_name(onnx_path.name + ".data")
    external_data_path.unlink(missing_ok=True)
    onnx.save(
        model, str(onnx_path),
        save_as_external_data=True, all_tensors_to_one_file=True,
        location=onnx_path.name + ".data", size_threshold=1024,
    )


def promote_large_constants_to_initializers(onnx_path: Path, size_threshold: int = 1024) -> int:
    """Turn large Constant-node `value` attributes into graph initializers.

    do_constant_folding=False (needed so the other backends see the same
    LIF-decomposed graph) makes torch's exporter emit every weight tensor as
    a literal `Constant` node feeding its consumers, rather than a graph
    initializer -- fine for small tensors, but save_onnx's external-data path
    can only externalize initializers (see its docstring), so a large model's
    weights would otherwise sit fully inlined in memory on every subsequent
    onnx.load()/save() in this pipeline, hitting protobuf's 2GB message-size
    limit for vgg/mobilenet. A Constant node's output name and an
    initializer of the same name are interchangeable as far as every
    consumer's input edge is concerned, so replacing one with the other typed
    the same way changes nothing about what the graph computes.
    """
    model = onnx.load(str(onnx_path))
    graph = model.graph
    new_nodes = []
    promoted = 0
    for node in graph.node:
        if node.op_type == "Constant" and len(node.output) == 1:
            value_attr = next((a for a in node.attribute if a.name == "value"), None)
            if (
                value_attr is not None
                and value_attr.HasField("t")
                and value_attr.t.HasField("raw_data")
                and len(value_attr.t.raw_data) >= size_threshold
            ):
                promoted_tensor = onnx.TensorProto()
                promoted_tensor.CopyFrom(value_attr.t)
                promoted_tensor.name = node.output[0]
                graph.initializer.append(promoted_tensor)
                promoted += 1
                continue
        new_nodes.append(node)
    if promoted:
        del graph.node[:]
        graph.node.extend(new_nodes)
        save_onnx(model, onnx_path)
    return promoted


def replace_unsupported_comparison_ops(onnx_path: Path) -> int:
    """Rewrite GreaterOrEqual/LessOrEqual as Not(Less)/Not(Greater) in-place.

    NNFusion's ONNX frontend has no native translator for GreaterOrEqual (the
    LIF neuron threshold in every Kairos workload lowers to this op) -- but
    rather than lean solely on registering one there, mirror this graph-level
    rewrite first: it's what let a prior manual test (Chronos's test.py, a
    T=32 spiking resnet18) compile through welder successfully. welder's own
    Python-side fusion/tile-search engine appears to handle the far more
    common Less/Greater+Not idiom more robustly than a bare GreaterEq node,
    which independently reproduced two different welder-engine crashes on
    Kairos's and spikingjelly's LIF decompositions at T=16.
    """
    model = onnx.load(str(onnx_path))
    graph = model.graph
    new_nodes = []
    replaced = 0
    for node in graph.node:
        if node.op_type == "GreaterOrEqual":
            less_out = node.output[0] + "_lt_tmp"
            new_nodes.append(onnx_helper.make_node(
                "Less", inputs=list(node.input), outputs=[less_out],
                name=(node.name + "_less") if node.name else None,
            ))
            new_nodes.append(onnx_helper.make_node(
                "Not", inputs=[less_out], outputs=list(node.output),
                name=(node.name + "_not") if node.name else None,
            ))
            replaced += 1
        elif node.op_type == "LessOrEqual":
            gt_out = node.output[0] + "_gt_tmp"
            new_nodes.append(onnx_helper.make_node(
                "Greater", inputs=list(node.input), outputs=[gt_out],
                name=(node.name + "_greater") if node.name else None,
            ))
            new_nodes.append(onnx_helper.make_node(
                "Not", inputs=[gt_out], outputs=list(node.output),
                name=(node.name + "_not") if node.name else None,
            ))
            replaced += 1
        else:
            new_nodes.append(node)
    if replaced:
        del graph.node[:]
        graph.node.extend(new_nodes)
        save_onnx(model, onnx_path)
    return replaced


def replace_unsqueeze_axes_input_with_attribute(onnx_path: Path) -> int:
    """Rewrite opset>=13 Unsqueeze(data, axes) as opset<13 Unsqueeze(data){axes=...}.

    NNFusion's ONNX frontend still expects the pre-opset-13 encoding, where
    Unsqueeze's axes are a node attribute rather than a second input -- it
    hard-crashes (uncaught C++ exception, `unknown attribute 'axes'`) on
    every model whose Unsqueeze axes come in as an input tensor, which torch
    onnx export at opset>=13 always produces. Every axes input observed here
    traces back to a single, statically-known Constant feeding exactly one
    Unsqueeze, so this is a lossless re-encoding, not a graph edit: same op,
    same axes values, just moved from input to attribute. The now-orphaned
    Constant nodes are dropped afterward if nothing else still reads them.
    """
    model = onnx.load(str(onnx_path))
    graph = model.graph
    initializers = {i.name: i for i in graph.initializer}
    const_nodes = {n.output[0]: n for n in graph.node if n.op_type == "Constant"}

    def resolve_axes(name: str):
        if name in initializers:
            return numpy_helper.to_array(initializers[name]).astype(np.int64).tolist()
        if name in const_nodes:
            value_attr = next(
                (a for a in const_nodes[name].attribute if a.name == "value"), None
            )
            if value_attr is not None:
                return numpy_helper.to_array(value_attr.t).astype(np.int64).tolist()
        return None

    new_nodes = []
    replaced = 0
    for node in graph.node:
        if (
            node.op_type == "Unsqueeze"
            and len(node.input) == 2
            and not any(a.name == "axes" for a in node.attribute)
        ):
            axes = resolve_axes(node.input[1])
            if axes is not None:
                new_nodes.append(onnx_helper.make_node(
                    "Unsqueeze", inputs=[node.input[0]], outputs=list(node.output),
                    name=node.name, axes=axes,
                ))
                replaced += 1
                continue
        new_nodes.append(node)

    if replaced:
        still_used = {i for n in new_nodes for i in n.input}
        still_used.update(o.name for o in graph.output)
        new_nodes = [
            n for n in new_nodes
            if not (n.op_type == "Constant" and n.output[0] not in still_used)
        ]
        del graph.node[:]
        graph.node.extend(new_nodes)
        save_onnx(model, onnx_path)
    return replaced


def replace_split_sizes_input_with_attribute(onnx_path: Path) -> int:
    """Rewrite opset>=13 Split(data, split) as opset<13 Split(data){split=...}.

    Same story as Unsqueeze above, different op: opset>=13 moved Split's
    per-output sizes from a `split` attribute to an optional second input,
    and NNFusion's ONNX frontend still only understands the attribute form
    -- `Node (.../Split): unknown attribute 'split'` (mamba's block splits
    its SSM projection into three chunks this way). Every split-sizes input
    observed here traces back to a single, statically-known Constant feeding
    exactly one Split, so this is a lossless re-encoding, not a graph edit.
    """
    model = onnx.load(str(onnx_path))
    graph = model.graph
    initializers = {i.name: i for i in graph.initializer}
    const_nodes = {n.output[0]: n for n in graph.node if n.op_type == "Constant"}

    def resolve_split(name: str):
        if name in initializers:
            return numpy_helper.to_array(initializers[name]).astype(np.int64).tolist()
        if name in const_nodes:
            value_attr = next(
                (a for a in const_nodes[name].attribute if a.name == "value"), None
            )
            if value_attr is not None:
                return numpy_helper.to_array(value_attr.t).astype(np.int64).tolist()
        return None

    new_nodes = []
    replaced = 0
    for node in graph.node:
        if (
            node.op_type == "Split"
            and len(node.input) == 2
            and not any(a.name == "split" for a in node.attribute)
        ):
            split = resolve_split(node.input[1])
            if split is not None:
                axis = next((a.i for a in node.attribute if a.name == "axis"), 0)
                new_nodes.append(onnx_helper.make_node(
                    "Split", inputs=[node.input[0]], outputs=list(node.output),
                    name=node.name, split=split, axis=axis,
                ))
                replaced += 1
                continue
        new_nodes.append(node)

    if replaced:
        still_used = {i for n in new_nodes for i in n.input}
        still_used.update(o.name for o in graph.output)
        new_nodes = [
            n for n in new_nodes
            if not (n.op_type == "Constant" and n.output[0] not in still_used)
        ]
        del graph.node[:]
        graph.node.extend(new_nodes)
        save_onnx(model, onnx_path)
    return replaced


def replace_reducesum_axes_input_with_attribute(onnx_path: Path) -> int:
    """Rewrite opset>=13 ReduceSum(data, axes) as opset<13 ReduceSum(data){axes=...}.

    Same story as Unsqueeze/Split above, different op: opset>=13 moved
    ReduceSum's axes from an attribute to an optional second input, but
    NNFusion's frontend (op/reduce.hpp TranslateReduceSumOp) is registered
    only for opset 1 and only ever reads the `axes` attribute -- it never
    looks at a second input at all. Since torch onnx export at opset>=13
    always emits the axes-as-input form, every ReduceSum here silently falls
    through to nnfusion's "no axes attribute" default of reducing *all* axes
    instead of just the intended one (mamba's SSM does `.sum(-1)` over its
    d_state/d_conv axis only) -- corrupting the reduction shape and
    eventually crashing graph conversion with an unrelated-looking
    `autobroadcast_incompatible_shapes` error much further downstream. Every
    axes input observed here traces back to a single, statically-known
    Constant feeding exactly one ReduceSum, so this is a lossless
    re-encoding, not a graph edit: same op, same axes values and keepdims,
    just moved from input to attribute.
    """
    model = onnx.load(str(onnx_path))
    graph = model.graph
    initializers = {i.name: i for i in graph.initializer}
    const_nodes = {n.output[0]: n for n in graph.node if n.op_type == "Constant"}

    def resolve_axes(name: str):
        if name in initializers:
            return numpy_helper.to_array(initializers[name]).astype(np.int64).tolist()
        if name in const_nodes:
            value_attr = next(
                (a for a in const_nodes[name].attribute if a.name == "value"), None
            )
            if value_attr is not None:
                return numpy_helper.to_array(value_attr.t).astype(np.int64).tolist()
        return None

    new_nodes = []
    replaced = 0
    for node in graph.node:
        if (
            node.op_type == "ReduceSum"
            and len(node.input) == 2
            and not any(a.name == "axes" for a in node.attribute)
        ):
            axes = resolve_axes(node.input[1])
            if axes is not None:
                keepdims = next((a.i for a in node.attribute if a.name == "keepdims"), 1)
                new_nodes.append(onnx_helper.make_node(
                    "ReduceSum", inputs=[node.input[0]], outputs=list(node.output),
                    name=node.name, axes=axes, keepdims=keepdims,
                ))
                replaced += 1
                continue
        new_nodes.append(node)

    if replaced:
        still_used = {i for n in new_nodes for i in n.input}
        still_used.update(o.name for o in graph.output)
        new_nodes = [
            n for n in new_nodes
            if not (n.op_type == "Constant" and n.output[0] not in still_used)
        ]
        del graph.node[:]
        graph.node.extend(new_nodes)
        save_onnx(model, onnx_path)
    return replaced


def decompose_layernorm(onnx_path: Path) -> int:
    """Rewrite LayerNormalization nodes into their primitive-op decomposition.

    NNFusion's ONNX frontend imports LayerNormalization as an opaque
    GenericOp("LayerNorm") with no registered Antares IR translation, so
    welder's run_compiler crashes (`AssertionError: The computing expression
    doesn't start with proper prefix: - `) on every node the C++ import
    stage let through unflattened (spiketransformer/spikebert use it in
    every transformer block). Decomposing it into ReduceMean/Sub/Mul/Add/
    Sqrt/Div -- all primitives with an existing Antares IR translation --
    sidesteps that gap entirely. Only the single-output form (Y only, no
    Mean/InvStdDev) is handled since that's the only form torch onnx export
    produces here.
    """
    model = onnx.load(str(onnx_path))
    graph = model.graph

    new_nodes = []
    replaced = 0
    for node in graph.node:
        if node.op_type != "LayerNormalization" or len(node.output) != 1:
            new_nodes.append(node)
            continue

        x, scale, bias = node.input[0], node.input[1], node.input[2]
        axis = next((a.i for a in node.attribute if a.name == "axis"), -1)
        epsilon = next((a.f for a in node.attribute if a.name == "epsilon"), 1e-5)
        base = node.output[0]

        def t(suffix):
            return f"{base}/{suffix}"

        eps_const = onnx_helper.make_node(
            "Constant", inputs=[], outputs=[t("eps")], name=t("eps_const"),
            value=onnx_helper.make_tensor(t("eps_val"), onnx.TensorProto.FLOAT, [], [epsilon]),
        )
        mean = onnx_helper.make_node(
            "ReduceMean", inputs=[x], outputs=[t("mean")], name=t("ReduceMean"),
            axes=[axis], keepdims=1,
        )
        centered = onnx_helper.make_node(
            "Sub", inputs=[x, t("mean")], outputs=[t("centered")], name=t("Sub"),
        )
        squared = onnx_helper.make_node(
            "Mul", inputs=[t("centered"), t("centered")], outputs=[t("squared")], name=t("Mul_sq"),
        )
        var = onnx_helper.make_node(
            "ReduceMean", inputs=[t("squared")], outputs=[t("var")], name=t("ReduceMean_var"),
            axes=[axis], keepdims=1,
        )
        var_eps = onnx_helper.make_node(
            "Add", inputs=[t("var"), t("eps")], outputs=[t("var_eps")], name=t("Add_eps"),
        )
        std = onnx_helper.make_node(
            "Sqrt", inputs=[t("var_eps")], outputs=[t("std")], name=t("Sqrt"),
        )
        normed = onnx_helper.make_node(
            "Div", inputs=[t("centered"), t("std")], outputs=[t("normed")], name=t("Div"),
        )
        scaled = onnx_helper.make_node(
            "Mul", inputs=[t("normed"), scale], outputs=[t("scaled")], name=t("Mul_scale"),
        )
        biased = onnx_helper.make_node(
            "Add", inputs=[t("scaled"), bias], outputs=[base], name=t("Add_bias"),
        )

        new_nodes.extend([
            eps_const, mean, centered, squared, var, var_eps, std, normed, scaled, biased,
        ])
        replaced += 1

    if replaced:
        del graph.node[:]
        graph.node.extend(new_nodes)
        save_onnx(model, onnx_path)
    return replaced


def normalize_negative_transpose_perm(onnx_path: Path) -> int:
    """Rewrite negative-indexed Transpose `perm` attributes to their positive form.

    ONNX's spec allows negative axes in `perm` (e.g. `perm=[-3, 0, 3, 1, 4]`,
    which torch's exporter emits verbatim for source code that permutes with
    a negative dim, as spikebert/spiketransformer's QKV reshape does).
    NNFusion's Transpose shape-inference (numpy_transpose.cpp) doesn't
    normalize negative entries before using them as an axis order -- it reads
    -3 as a huge unsigned index and hard-crashes. `perm[i] + len(perm)` for
    `perm[i] < 0` is the exact same permutation (len(perm) == input rank,
    since Transpose's perm is always a full permutation of all dims), so this
    changes only the encoding nnfusion chokes on, not what the op computes.
    """
    model = onnx.load(str(onnx_path))
    graph = model.graph
    normalized = 0
    for node in graph.node:
        if node.op_type != "Transpose":
            continue
        for attr in node.attribute:
            if attr.name != "perm":
                continue
            rank = len(attr.ints)
            if any(p < 0 for p in attr.ints):
                fixed = [p + rank if p < 0 else p for p in attr.ints]
                del attr.ints[:]
                attr.ints.extend(fixed)
                normalized += 1
    if normalized:
        save_onnx(model, onnx_path)
    return normalized


_SHAPE_FOLDABLE_OPS = ("Shape", "Gather", "Add", "Sub", "Mul", "Div", "Cast", "Unsqueeze", "Concat")


def fold_static_shape_subgraphs(onnx_path: Path) -> int:
    """Collapse Shape/Gather/Add/Div/Mul chains (torch.chunk's dynamic split
    bounds) into plain Constants.

    Every model exported here (benchmark_bladedisc_runtime.export_onnx, no
    `dynamic_axes`) has a fully static input shape, so every `torch.chunk`/
    `torch.split` call -- traced with `do_constant_folding=False`, needed for
    the other backends -- lowers to a live Shape->Gather->Add->Div->Mul chain
    computing what is, for this fixed input, provably always the same
    integer: e.g. deepspeech2's GRU cell chunking its 2400-wide projection
    into three 800-wide gates, or mamba's block chunking its 3072-wide
    in_proj output into two 1536-wide halves. NNFusion has its own
    import-time evaluator for exactly this pattern (frontend/util/evaluator.hpp)
    but it mis-evaluates these two (a wrong Slice bound for deepspeech2, a
    Concat-axis mismatch for mamba) -- so fold it ourselves with ONNX's own
    shape inference (data_prop=True gives concrete values, not just shapes,
    since nothing here is dynamic) rather than lean on that fragile path.
    Only nodes whose output actually traces back to a `Shape` node are
    touched -- ordinary weight/initializer arithmetic is left alone.
    """
    model = onnx.load(str(onnx_path))
    graph = model.graph
    # infer_shapes(model, ...) unconditionally does model.SerializeToString()
    # on the full in-memory proto, which hits protobuf's 2GB message-size cap
    # on vgg/mobilenet (do_constant_folding=False means their weights sit as
    # Constant-node attributes rather than lean initializer references, so
    # the loaded model is far larger in memory than the on-disk, external-data
    # .onnx file). infer_shapes_path is the file-based variant built
    # specifically to support >2GB models; load_external_data=False on the
    # result since only shape/type info (value_info) is needed here, not the
    # actual tensor bytes.
    with tempfile.TemporaryDirectory() as tmp_dir:
        inferred_path = Path(tmp_dir) / "inferred.onnx"
        onnx.shape_inference.infer_shapes_path(
            str(onnx_path), str(inferred_path), strict_mode=True, data_prop=True,
        )
        inferred = onnx.load(str(inferred_path), load_external_data=False)
    static_shapes: Dict[str, list] = {}
    for vi in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output):
        dims = vi.type.tensor_type.shape.dim
        if dims and all(d.HasField("dim_value") for d in dims):
            static_shapes[vi.name] = [d.dim_value for d in dims]

    def get_attr(node, name, default=None):
        for a in node.attribute:
            if a.name == name:
                return onnx_helper.get_attribute_value(a)
        return default

    resolved: Dict[str, np.ndarray] = {
        i.name: numpy_helper.to_array(i) for i in graph.initializer
    }
    for node in graph.node:
        if node.op_type == "Constant":
            value_attr = next((a for a in node.attribute if a.name == "value"), None)
            if value_attr is not None:
                resolved[node.output[0]] = numpy_helper.to_array(value_attr.t)

    shape_derived = set()
    for node in graph.node:
        if not node.output or node.output[0] in resolved:
            continue
        out0 = node.output[0]
        try:
            if node.op_type == "Shape" and node.input[0] in static_shapes:
                dims = static_shapes[node.input[0]]
                start = get_attr(node, "start", 0)
                end = get_attr(node, "end", len(dims))
                resolved[out0] = np.array(dims[start:end], dtype=np.int64)
                shape_derived.add(out0)
            elif node.op_type == "Gather" and all(i in resolved for i in node.input):
                axis = get_attr(node, "axis", 0)
                resolved[out0] = np.take(resolved[node.input[0]], resolved[node.input[1]], axis=axis)
                if any(i in shape_derived for i in node.input):
                    shape_derived.add(out0)
            elif node.op_type in ("Add", "Sub", "Mul", "Div") and all(i in resolved for i in node.input):
                a, b = resolved[node.input[0]], resolved[node.input[1]]
                if node.op_type == "Add":
                    resolved[out0] = a + b
                elif node.op_type == "Sub":
                    resolved[out0] = a - b
                elif node.op_type == "Mul":
                    resolved[out0] = a * b
                else:
                    resolved[out0] = (a // b) if np.issubdtype(a.dtype, np.integer) else a / b
                if any(i in shape_derived for i in node.input):
                    shape_derived.add(out0)
            elif node.op_type == "Cast" and node.input[0] in resolved:
                np_dtype = onnx.helper.tensor_dtype_to_np_dtype(get_attr(node, "to"))
                resolved[out0] = resolved[node.input[0]].astype(np_dtype)
                if node.input[0] in shape_derived:
                    shape_derived.add(out0)
            elif node.op_type == "Unsqueeze" and node.input[0] in resolved:
                # axes is an attribute pre-opset-13, a second input from
                # opset-13 on -- this pass runs before
                # replace_unsqueeze_axes_input_with_attribute, so it's still
                # the latter here, but handle both since op ordering is an
                # implementation detail neither function should have to assume.
                if any(a.name == "axes" for a in node.attribute):
                    axes = get_attr(node, "axes")
                elif len(node.input) == 2 and node.input[1] in resolved:
                    axes = np.atleast_1d(resolved[node.input[1]]).tolist()
                else:
                    axes = None
                if axes is not None:
                    arr = resolved[node.input[0]]
                    for ax in sorted(axes):
                        arr = np.expand_dims(arr, ax)
                    resolved[out0] = arr
                    if node.input[0] in shape_derived:
                        shape_derived.add(out0)
            elif node.op_type == "Concat" and all(i in resolved for i in node.input):
                resolved[out0] = np.concatenate(
                    [np.atleast_1d(resolved[i]) for i in node.input], axis=get_attr(node, "axis", 0)
                )
                if any(i in shape_derived for i in node.input):
                    shape_derived.add(out0)
        except Exception:
            pass

    # Only replace the *boundary* nodes -- those with at least one consumer
    # outside the shape-derived clique (a real Slice/Concat/etc that needs
    # the concrete value). Everything strictly upstream of a boundary node
    # (Shape/Gather/Add/... feeding only other shape-derived nodes) is left
    # for the dead-code sweep below instead of also being materialized as
    # its own Constant -- collapsing thousands of near-duplicate per-timestep
    # scalars (one GRU cell's chunk-bounds chain, unrolled 16x by
    # SequenceInputLoopWrapper) down to just the handful nnfusion actually
    # needs was what let its own graph optimizer run without tripping over
    # the sheer node count this pass otherwise introduced.
    consumers: Dict[str, list] = {}
    for node in graph.node:
        for i in node.input:
            consumers.setdefault(i, []).append(node)

    def is_sink(name: str) -> bool:
        for c in consumers.get(name, []):
            c_out0 = c.output[0] if c.output else None
            if c_out0 not in shape_derived:
                return True
        return name in {o.name for o in graph.output}

    sinks = {name for name in shape_derived if is_sink(name)}

    new_nodes = []
    folded = 0
    for node in graph.node:
        out0 = node.output[0] if node.output else None
        if node.op_type in _SHAPE_FOLDABLE_OPS and out0 in sinks:
            value = resolved[out0]
            new_nodes.append(onnx_helper.make_node(
                "Constant", inputs=[], outputs=[out0], name=node.name,
                value=numpy_helper.from_array(value, name=out0 + "_folded_value"),
            ))
            folded += 1
        else:
            new_nodes.append(node)

    if folded:
        # Fixed-point dead-code sweep: dropping a now-unused upstream
        # shape-computation node can orphan its own inputs' producers in turn.
        graph_outputs = {o.name for o in graph.output}
        changed = True
        while changed:
            changed = False
            still_used = {i for n in new_nodes for i in n.input}
            still_used.update(graph_outputs)
            pruned = [
                n for n in new_nodes
                if not (n.op_type in _SHAPE_FOLDABLE_OPS + ("Constant",) and n.output[0] not in still_used)
            ]
            if len(pruned) != len(new_nodes):
                changed = True
            new_nodes = pruned
        del graph.node[:]
        graph.node.extend(new_nodes)
        save_onnx(model, onnx_path)
    return folded


################################################################################
# welder compile pipeline (subprocess-driven, mirrors artifacts/tune_welder.py
# and testing/run_welder.py from the welder branch of nnfusion/nnfusion)
################################################################################

def compile_with_welder(
    work_dir: Path,
    arch: str,
    topk: int,
    gpu_device: int,
    welder_repo: str,
    welder_python: str,
    welder_tvm_pythonpath: str,
    nnfusion_bin_dir: str,
    cuda_home: str,
    skip_dot: bool,
    no_tc: bool,
    nofusion: bool,
    compile_timeout_sec: int,
) -> Dict[str, Any]:
    # Must be absolute: every subprocess below runs with cwd=work_dir, and a
    # relative work_dir (the common case, since --out-dir defaults to a
    # relative path) would otherwise get resolved a second time against
    # itself wherever it's later passed as a standalone path argument (e.g.
    # cmake -S/-B), doubling the directory prefix.
    work_dir = work_dir.resolve()
    log_path = work_dir / "welder_compile.log"

    base_env = os.environ.copy()
    base_env["PATH"] = f"{cuda_home}/bin:{nnfusion_bin_dir}:{base_env.get('PATH', '')}"
    # welder/utils.py hardcodes "~/cutlass/include" (not CPLUS_INCLUDE_PATH) for
    # its own kernel-profiling nvcc calls, but the *final* model's generated
    # CMakeLists.txt has no -I for cutlass at all -- nvcc/gcc only find it via
    # this env var, which the reference Docker setup exported process-wide.
    base_env["CPLUS_INCLUDE_PATH"] = os.path.expanduser(
        f"~/cutlass/include:{base_env.get('CPLUS_INCLUDE_PATH', '')}"
    )

    # Must be the absolute path, not the bare "model.onnx" relative filename:
    # nnfusion resolves an external-data tensor's companion file as
    # `dirname(argv[1]) + "/" + location` (graph_convert.cpp), and computes
    # dirname("model.onnx") -- no "/" anywhere in the string -- as "" rather
    # than ".", so it ends up looking for the file at a bogus path like
    # "/deepspeech2_..._tf32.onnx.data" (looks filesystem-rooted) instead of
    # inside work_dir. A real directory component sidesteps that path-parsing
    # gap entirely, for both nnfusion invocations below.
    model_onnx_path = str(work_dir / "model.onnx")
    nnfusion_cmd1 = ["nnfusion", model_onnx_path, "-f", "onnx", "-ftune_output_file=model.json"]
    nnfusion_cmd3 = ["nnfusion", model_onnx_path, "-f", "onnx", "-ftune_output_file=/dev/null",
                      "-ftune_input_file=tuned.json"]
    if no_tc:
        nnfusion_cmd1.append("-ftc_rewrite=0")
        nnfusion_cmd3.append("-ftc_rewrite=0")
    if skip_dot:
        nnfusion_cmd1.append("-ffusion_skiplist=Dot")
        nnfusion_cmd3.append("-ffusion_skiplist=Dot")
    if nofusion:
        # welder's own --nofusion (passed to run_compiler below) only controls
        # whether welder's Python tuning search groups nodes into multi-op tile
        # schedules. It does NOT stop nnfusion's C++ RegisterFusionPass from
        # first merging elementwise op chains (e.g. long Broadcast_Add chains)
        # into a single Matched_Pattern graph node -- that pass runs in both
        # the model.json export (cmd1) and the final codegen (cmd3), and is
        # gated by this separate flag (register_fusion_pass.cpp:22,146).
        # Without it, welder's tuning search never even sees the individual
        # ops: they're already pre-fused into one node before it gets model.json.
        nnfusion_cmd1.append("-fnofuse=1")
        nnfusion_cmd3.append("-fnofuse=1")

    # GemmFusionPass (gemm_fusion_pass.cpp) hard-crashes on any graph where it
    # actually finds something to fuse: Graph::add_gnode_and_edge's null-check
    # is inverted (`NNFUSION_CHECK(gnode == nullptr)` -- graph.cpp:104 -- guards
    # against a null gnode by asserting one *is* null, so it throws on every
    # valid, freshly-built fused node instead). CNN-family models never hit
    # this path since GemmFusionPass has nothing to match; deepspeech2's GRU
    # gate matmuls do. `-fgemm_fusion=false` (its own dedicated, documented
    # flag) skips the pass outright -- a compile-time optimization toggle,
    # not a model change -- sidestepping the bug rather than patching nnfusion's
    # C++ and rebuilding it.
    nnfusion_cmd1.append("-fgemm_fusion=false")
    nnfusion_cmd3.append("-fgemm_fusion=false")

    # Per-stage wall-clock time (seconds). "run_compiler" is welder's actual
    # autotuning cost (compiling+profiling ~topk candidates per fusion group
    # on the real GPU); the rest is ordinary ONNX-import/codegen/nvcc-build
    # overhead, reported separately since callers may only care about one or
    # the other. Populated incrementally so a failed/timed-out run still
    # reports how long it ran before failing.
    stage_seconds: Dict[str, float] = {}

    def stage_result(ok: bool, stage: str, returncode) -> Dict[str, Any]:
        stage_seconds["total"] = sum(stage_seconds.values())
        return {
            "ok": ok, "stage": stage, "returncode": returncode,
            "log_path": str(log_path), "stage_seconds": dict(stage_seconds),
        }

    rc, elapsed = run_logged(nnfusion_cmd1, work_dir, base_env, log_path)
    stage_seconds["nnfusion_tune_export"] = elapsed
    if rc != 0:
        return stage_result(False, "nnfusion_tune_export", rc)

    # run_compiler needs Welder's own (patched) TVM fork on PYTHONPATH, and
    # must run isolated from any ambient user-site packages (this box has a
    # numpy>=2 / mismatched-deps stack in ~/.local that breaks the old TVM
    # fork's ctypes/numpy glue).
    compiler_env = base_env.copy()
    existing_pp = compiler_env.get("PYTHONPATH", "")
    compiler_env["PYTHONPATH"] = os.pathsep.join(
        p for p in [welder_tvm_pythonpath, f"{welder_repo}/python", existing_pp] if p
    )
    compiler_env["PYTHONNOUSERSITE"] = "1"

    run_compiler_cmd = [
        welder_python, "-m", "run_compiler", "model.json", "tuned.json",
        "--topk", str(topk), "--arch", arch, "--device", str(gpu_device),
    ]
    if nofusion:
        run_compiler_cmd.append("--nofusion")
    rc, elapsed = run_logged(run_compiler_cmd, work_dir, compiler_env, log_path, timeout=compile_timeout_sec)
    stage_seconds["run_compiler"] = elapsed
    if rc != 0:
        return stage_result(False, "run_compiler", rc)

    rc, elapsed = run_logged(nnfusion_cmd3, work_dir, base_env, log_path)
    stage_seconds["nnfusion_codegen"] = elapsed
    if rc != 0:
        return stage_result(False, "nnfusion_codegen", rc)

    codegen_dir = work_dir / "nnfusion_rt" / "cuda_codegen"
    build_dir = codegen_dir / "build"
    if build_dir.exists():
        run_logged(["rm", "-rf", str(build_dir)], work_dir, base_env, log_path)

    rc, elapsed = run_logged(["cmake", "-S", str(codegen_dir), "-B", str(build_dir)], work_dir, base_env, log_path)
    stage_seconds["cmake_configure"] = elapsed
    if rc != 0:
        return stage_result(False, "cmake_configure", rc)

    nproc = os.cpu_count() or 4
    rc, elapsed = run_logged(["make", "-C", str(build_dir), f"-j{nproc}"], work_dir, base_env, log_path)
    stage_seconds["cmake_build"] = elapsed
    if rc != 0:
        return stage_result(False, "cmake_build", rc)

    so_path = build_dir / "libnnfusion_naive_rt.so"
    if not so_path.exists():
        return stage_result(False, "artifact_missing", 0)

    stage_seconds["total"] = sum(stage_seconds.values())
    return {
        "ok": True, "stage": "done", "returncode": 0, "log_path": str(log_path),
        "so_path": str(so_path), "codegen_dir": str(codegen_dir),
        "stage_seconds": dict(stage_seconds),
    }


################################################################################
# benchmark the compiled .so (ctypes, mirrors testing/run_welder.py)
################################################################################

def benchmark_shared_lib(
    so_path: Path,
    codegen_dir: Path,
    x: torch.Tensor,
    output_shape,
    output_dtype: torch.dtype,
    gpu_device: int,
    warmup_sec: float,
    iters: int,
) -> Dict[str, Any]:
    cuda_rt = ctypes.CDLL("libcudart.so")
    lib = ctypes.CDLL(str(so_path))

    cur_dir = os.getcwd()
    os.chdir(codegen_dir)
    try:
        lib.cuda_init()
        y = torch.empty(output_shape, dtype=output_dtype, device=f"cuda:{gpu_device}")
        args = [ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(y.data_ptr())]

        def run_once():
            tic = time.monotonic_ns()
            lib.kernel_entry(*args)
            mid = time.monotonic_ns()
            cuda_rt.cudaDeviceSynchronize()
            end = time.monotonic_ns()
            return (mid - tic) / 1e6, (end - tic) / 1e6

        st = time.time()
        while time.time() - st < warmup_sec:
            run_once()

        dispatch_times, total_times = [], []
        for _ in range(iters):
            d, t = run_once()
            dispatch_times.append(d)
            total_times.append(t)
    finally:
        os.chdir(cur_dir)

    return {
        "latency_ms": summarize_ms(total_times),
        "schedule_ms": summarize_ms(dispatch_times),
    }


################################################################################
# one (model, execution_mode, precision) case
################################################################################

def run_case(model_name: str, execution_mode: str, precision: str, args) -> Dict[str, Any]:
    dtype = resolve_dtype(precision)

    result: Dict[str, Any] = {
        "ok": False,
        "model": model_name,
        "execution_mode": execution_mode,
        "precision": precision,
        "batch_size": args.batch_size,
        "time_steps": args.T,
        "arch": args.arch,
        "topk": args.topk,
        "onnx_export_ok": False,
        "welder_compile_ok": False,
        "benchmark_ok": False,
        "error": "",
    }

    if execution_mode != "single_step_mode":
        result["error"] = f"unsupported execution mode: {execution_mode}"
        return result

    run_dir = Path(args.out_dir) / model_name / execution_mode / precision
    run_dir.mkdir(parents=True, exist_ok=True)

    #
    # build model + input (reused from the BladeDISC baseline's build_case,
    # which already applies the same export-only LIF decomposition as TVM
    # and TensorRT) and export ONNX
    #
    try:
        model, x, wrapper_name, replaced = build_case(model_name, args, dtype)
        result["wrapper"] = wrapper_name
        result["export_custom_lif_replaced"] = replaced
        result["input_shape"] = list(x.shape)

        onnx_path = run_dir / f"{model_name}_{execution_mode}_T{args.T}_{precision}.onnx"
        export_onnx(model, x, onnx_path, args.opset)
        result["onnx_path"] = str(onnx_path)
        result["onnx_export_ok"] = True
        result["graph_contains_custom_lif"] = onnx_graph_contains_custom_lif(onnx_path)
        result["constants_promoted_to_initializers"] = promote_large_constants_to_initializers(onnx_path)
        result["comparison_ops_replaced"] = replace_unsupported_comparison_ops(onnx_path)
        result["transpose_perm_normalized"] = normalize_negative_transpose_perm(onnx_path)
        result["shape_subgraphs_folded"] = fold_static_shape_subgraphs(onnx_path)
        result["unsqueeze_axes_rewritten"] = replace_unsqueeze_axes_input_with_attribute(onnx_path)
        result["split_sizes_rewritten"] = replace_split_sizes_input_with_attribute(onnx_path)
        result["reducesum_axes_rewritten"] = replace_reducesum_axes_input_with_attribute(onnx_path)
        result["layernorm_decomposed"] = decompose_layernorm(onnx_path)

        io_info = infer_output_shape(model, x)
        result["output_shape"] = io_info["shape"]
        result["output_dtype"] = io_info["dtype"]
    except Exception:
        result["error"] = traceback.format_exc()
        return result

    #
    # welder workspace: PREFIX dir containing model.onnx, matching
    # artifacts/tune_welder.py's expected layout
    #
    workspace = run_dir / "welder_workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    workspace_onnx = workspace / "model.onnx"
    workspace_onnx.write_bytes(onnx_path.read_bytes())
    # save_onnx() (above) writes big models (vgg/mobilenet) with an external-data
    # companion file next to onnx_path, named onnx_path.name + ".data" -- that
    # literal name is what's baked into the model's external_data location, so
    # it has to be copied alongside the renamed "model.onnx" under the same name.
    onnx_data_path = onnx_path.with_name(onnx_path.name + ".data")
    if onnx_data_path.exists():
        (workspace / onnx_data_path.name).write_bytes(onnx_data_path.read_bytes())

    compile_result = compile_with_welder(
        work_dir=workspace,
        arch=args.arch,
        topk=args.topk,
        gpu_device=args.gpu_device,
        welder_repo=args.welder_repo,
        welder_python=args.welder_python,
        welder_tvm_pythonpath=args.welder_tvm_pythonpath,
        nnfusion_bin_dir=args.nnfusion_bin_dir,
        cuda_home=args.cuda_home,
        skip_dot=args.skip_dot,
        no_tc=args.no_tc,
        nofusion=args.nofusion,
        compile_timeout_sec=args.compile_timeout_sec,
    )
    result["welder_compile"] = compile_result
    result["welder_compile_ok"] = compile_result["ok"]
    # Convenient top-level aliases into welder_compile.stage_seconds:
    # autotune_seconds is welder's actual autotuning cost (run_compiler --topk
    # candidates compiled+profiled on the real GPU); welder_compile_seconds_total
    # additionally includes ordinary ONNX-import/codegen/nvcc-build overhead.
    stage_seconds = compile_result.get("stage_seconds", {})
    result["autotune_seconds"] = stage_seconds.get("run_compiler")
    result["welder_compile_seconds_total"] = stage_seconds.get("total")
    if not compile_result["ok"]:
        return result

    #
    # benchmark
    #
    try:
        parsed = benchmark_shared_lib(
            so_path=Path(compile_result["so_path"]),
            codegen_dir=Path(compile_result["codegen_dir"]),
            x=x,
            output_shape=result["output_shape"],
            output_dtype=TORCH_DTYPE_BY_NAME[result["output_dtype"]],
            gpu_device=args.gpu_device,
            warmup_sec=args.warmup_sec,
            iters=args.iters,
        )
        result["parsed"] = parsed
        result["benchmark_ok"] = True
        result["ok"] = True
    except Exception:
        result["error"] = traceback.format_exc()

    return result


################################################################################
# main
################################################################################

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--models", nargs="+", choices=KAIROS_MODEL_CHOICES, default=["resnet18"])
    p.add_argument("--execution-modes", nargs="+", default=["single_step_mode"],
                    choices=["single_step_mode"])
    p.add_argument("--precisions", nargs="+", default=["fp32", "tf32", "fp16"],
                    choices=["fp32", "tf32", "fp16"])

    p.add_argument("--T", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=16)
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

    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--out-dir", default="test/welder_validation")

    # welder-specific
    p.add_argument("--topk", type=int, default=20, help="Number of tuning trials per subgraph.")
    p.add_argument("--arch", default="RTX5090")
    p.add_argument("--gpu-device", type=int, default=0)
    p.add_argument("--skip-dot", action="store_true")
    p.add_argument("--no-tc", action="store_true")
    p.add_argument("--nofusion", action="store_true",
                    help="Pass --nofusion to welder's run_compiler: tune every op "
                         "individually instead of building multi-node fusion groups.")
    p.add_argument(
        "--compile-timeout-sec", type=int, default=7200,
        help="Welder's autotuner compiles+profiles ~topk candidates per fusion "
             "group on the real GPU; for a full model at topk=20 this routinely "
             "takes over an hour (~14 min was observed at topk=3), not seconds.",
    )
    p.add_argument("--warmup-sec", type=float, default=1.0)
    p.add_argument("--iters", type=int, default=100)

    p.add_argument("--welder-repo", default=DEFAULT_WELDER_REPO)
    p.add_argument("--welder-python", default=DEFAULT_WELDER_PYTHON)
    p.add_argument("--welder-tvm-pythonpath", default=DEFAULT_WELDER_TVM_PYTHONPATH)
    p.add_argument("--nnfusion-bin-dir", default=DEFAULT_NNFUSION_BIN_DIR)
    p.add_argument("--cuda-home", default=DEFAULT_CUDA_HOME)

    args = p.parse_args()
    # build_case()/make_model_input() index into args by attribute, expecting
    # a torch device string (e.g. "cuda:0"), separate from --gpu-device (an
    # int used for welder's own --device flag and the nnfusion CLI).
    args.device = f"cuda:{args.gpu_device}"
    return args


def main():
    args = parse_args()
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, Any] = {}

    for model_name in args.models:
        all_results[model_name] = {}
        for execution_mode in args.execution_modes:
            all_results[model_name][execution_mode] = {}
            for precision in args.precisions:
                print("=" * 80)
                print(f"[RUN] model={model_name} mode={execution_mode} precision={precision}")
                print("=" * 80)

                result = run_case(model_name, execution_mode, precision, args)
                all_results[model_name][execution_mode][precision] = result

                run_dir = out_root / model_name / execution_mode / precision
                run_dir.mkdir(parents=True, exist_ok=True)
                summary_path = run_dir / "summary.json"
                summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                print(f"[WRITE] {summary_path}")

    aggregate_path = out_root / "welder_summary_all.json"
    aggregate_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print("=" * 80)
    print("[DONE]")
    print(f"[WRITE] {aggregate_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
