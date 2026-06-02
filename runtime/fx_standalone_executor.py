"""Experimental FX-native executor for Chronos-rewritten GraphModules.

This module intentionally stays below the existing Inductor path: it executes
the final FX graph directly, optionally with producer-event multi-stream
scheduling and CUDA Graph replay.
"""

from __future__ import annotations

import operator
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import torch
import torch.fx
from torch.fx.node import map_arg


def _iter_nodes(value: Any) -> Iterable[torch.fx.Node]:
    if isinstance(value, torch.fx.Node):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_nodes(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_nodes(item)
    elif isinstance(value, slice):
        yield from _iter_nodes(value.start)
        yield from _iter_nodes(value.stop)
        yield from _iter_nodes(value.step)


def _summarize(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device})"
    if isinstance(value, torch.fx.Node):
        return f"Node(name={value.name}, op={value.op}, target={value.target})"
    if isinstance(value, (tuple, list)):
        return type(value).__name__ + "(" + ", ".join(_summarize(v) for v in list(value)[:4]) + ")"
    if isinstance(value, dict):
        keys = list(value.keys())[:4]
        return "dict(" + ", ".join(f"{k}={_summarize(value[k])}" for k in keys) + ")"
    return repr(value)


def _clone_static_input(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    return value


def _copy_input(dst: Any, src: Any) -> Any:
    if isinstance(dst, torch.Tensor):
        if not isinstance(src, torch.Tensor):
            raise TypeError(f"static input expects Tensor, got {type(src).__name__}")
        if tuple(dst.shape) != tuple(src.shape) or dst.dtype != src.dtype or dst.device != src.device:
            raise ValueError(
                "CUDA graph input shape/dtype/device mismatch: "
                f"expected shape={tuple(dst.shape)} dtype={dst.dtype} device={dst.device}, "
                f"got shape={tuple(src.shape)} dtype={src.dtype} device={src.device}"
            )
        dst.copy_(src)
        return dst
    return src


def _is_plain_python_scalar(value: Any) -> bool:
    return isinstance(value, (int, float, bool))


def _has_tensor_meta(node: torch.fx.Node) -> bool:
    value = node.meta.get("val")
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, (tuple, list)):
        return any(isinstance(item, torch.Tensor) for item in value)
    return False


def _has_known_non_tensor_meta(node: torch.fx.Node) -> bool:
    return "val" in node.meta and not _has_tensor_meta(node)


def _is_metadata_node(node: torch.fx.Node) -> bool:
    value = node.meta.get("val")
    if _is_plain_python_scalar(value):
        return True
    if node.op == "call_method" and node.target in ("size", "dim", "numel", "item"):
        return True
    if node.op == "call_function" and node.target is operator.getitem:
        users = tuple(node.users)
        if not users:
            return False
        return all(_has_known_non_tensor_meta(user) for user in users)
    return False


def _compile_value_resolver(value: Any) -> Callable[[Dict[torch.fx.Node, Any]], Any]:
    if isinstance(value, torch.fx.Node):
        return lambda env, node=value: env[node]
    if isinstance(value, tuple):
        resolvers = tuple(_compile_value_resolver(item) for item in value)
        return lambda env, resolvers=resolvers: tuple(resolve(env) for resolve in resolvers)
    if isinstance(value, list):
        resolvers = tuple(_compile_value_resolver(item) for item in value)
        return lambda env, resolvers=resolvers: [resolve(env) for resolve in resolvers]
    if isinstance(value, dict):
        resolvers = tuple((key, _compile_value_resolver(item)) for key, item in value.items())
        return lambda env, resolvers=resolvers: {key: resolve(env) for key, resolve in resolvers}
    if isinstance(value, slice):
        start = _compile_value_resolver(value.start)
        stop = _compile_value_resolver(value.stop)
        step = _compile_value_resolver(value.step)
        return lambda env, start=start, stop=stop, step=step: slice(start(env), stop(env), step(env))
    return lambda _env, value=value: value


def _compile_arg_resolver(
    node: torch.fx.Node,
) -> Callable[[Dict[torch.fx.Node, Any]], Tuple[tuple, dict]]:
    args_resolver = _compile_value_resolver(node.args)
    kwargs_resolver = _compile_value_resolver(node.kwargs)
    return lambda env, args_resolver=args_resolver, kwargs_resolver=kwargs_resolver: (
        args_resolver(env),
        kwargs_resolver(env),
    )


@dataclass
class _ExecStep:
    node: torch.fx.Node
    stream_idx: Optional[int]
    wait_events: List[torch.cuda.Event]
    record_event: Optional[torch.cuda.Event]
    node_kind: str
    arg_resolver: Callable[[Dict[torch.fx.Node, Any]], Tuple[tuple, dict]]


_METADATA_INLINE = "metadata_inline"
_SIDE_STREAM_COMPUTE = "side_stream_compute"
_MAIN_STREAM_COMPUTE = "main_stream_compute"


def _target_text(target: Any) -> str:
    return str(target)


def _target_short(target: Any, limit: int = 72) -> str:
    text = _target_text(target)
    text = text.replace("torch.ops.", "")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _is_chronos_fused_temporal_op(node: torch.fx.Node) -> bool:
    if node.op != "call_function":
        return False
    text = _target_text(node.target)
    return "snn_custom" in text and (
        "fused_temporal" in text
        or "fused_conv_lif_state" in text
    )


def _is_chronos_fused_temporal_conv_state_target(target: Any) -> bool:
    text = _target_text(target)
    return (
        "snn_custom.fused_temporal_conv_lif_state" in text
        or "snn_custom.fused_temporal_pointwise_conv_lif_state" in text
        or "snn_custom.fused_temporal_depthwise_conv_lif_state" in text
    ) and "packed_out" not in text and "conv_add" not in text


def _is_chronos_fused_temporal_conv_state_node(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and _is_chronos_fused_temporal_conv_state_target(node.target)


def _is_metadata_inline_node(node: torch.fx.Node) -> bool:
    if node.op == "get_attr":
        return True
    if node.op == "call_method" and node.target in ("size", "dim", "numel"):
        return True
    if node.op == "call_function":
        if node.target is operator.getitem:
            return True
    return False


def _classify_exec_node(node: torch.fx.Node) -> str:
    if _is_metadata_inline_node(node):
        return _METADATA_INLINE
    if _is_chronos_fused_temporal_op(node):
        return _SIDE_STREAM_COMPUTE
    if node.op in ("call_function", "call_method", "call_module"):
        return _SIDE_STREAM_COMPUTE
    return _MAIN_STREAM_COMPUTE


def _getitem_index(node: torch.fx.Node) -> Any:
    if node.op != "call_function" or node.target is not operator.getitem or len(node.args) < 2:
        return None
    return node.args[1]


def _is_prunable_v_final_output_value(value: Any, output_node: torch.fx.Node) -> bool:
    if not isinstance(value, torch.fx.Node):
        return False
    if value.op != "call_function" or value.target is not operator.getitem:
        return False
    if _getitem_index(value) != 1 or not value.args or not isinstance(value.args[0], torch.fx.Node):
        return False
    producer = value.args[0]
    if not _is_chronos_fused_temporal_op(producer):
        return False
    real_users = [user for user in value.users if user is not output_node]
    return len(real_users) == 0


def _prune_graph_output_v_final_states(gm: torch.fx.GraphModule) -> int:
    output_node = next((node for node in reversed(list(gm.graph.nodes)) if node.op == "output"), None)
    if output_node is None or not output_node.args:
        return 0
    value = output_node.args[0]
    if not isinstance(value, tuple):
        return 0
    kept = []
    removed = 0
    replacement = value[0] if len(value) > 0 else None
    for index, item in enumerate(value):
        if replacement is not None and index > 0 and _is_prunable_v_final_output_value(item, output_node):
            removed += 1
            kept.append(replacement)
            continue
        kept.append(item)
    if not removed:
        return 0
    output_node.args = (tuple(kept),)
    gm.graph.lint()
    gm.recompile()
    return removed


prune_graph_output_v_final_states = _prune_graph_output_v_final_states


def _conv_out_hw(height: int, width: int, weight: torch.Tensor, stride, padding, dilation) -> Tuple[int, int]:
    stride_h, stride_w = int(stride[0]), int(stride[1])
    pad_h, pad_w = int(padding[0]), int(padding[1])
    dil_h, dil_w = int(dilation[0]), int(dilation[1])
    kernel_h, kernel_w = int(weight.shape[2]), int(weight.shape[3])
    out_h = (height + 2 * pad_h - dil_h * (kernel_h - 1) - 1) // stride_h + 1
    out_w = (width + 2 * pad_w - dil_w * (kernel_w - 1) - 1) // stride_w + 1
    return out_h, out_w


class TensorBufferPool:
    def __init__(self, debug: bool = False):
        self.debug = debug
        self._pool: Dict[Tuple[Tuple[int, ...], torch.dtype, torch.device], List[torch.Tensor]] = defaultdict(list)
        self.allocated = 0
        self.reused = 0
        self.live = 0
        self.peak_live = 0

    def acquire(self, shape, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        key = (tuple(int(dim) for dim in shape), dtype, device)
        if self._pool[key]:
            tensor = self._pool[key].pop()
            self.reused += 1
        else:
            tensor = torch.empty(key[0], dtype=dtype, device=device)
            self.allocated += 1
        self.live += 1
        self.peak_live = max(self.peak_live, self.live)
        return tensor

    def release(self, tensor: torch.Tensor) -> None:
        if not isinstance(tensor, torch.Tensor):
            return
        key = (tuple(tensor.shape), tensor.dtype, tensor.device)
        self._pool[key].append(tensor)
        self.live = max(0, self.live - 1)

    def release_value(self, value: Any) -> None:
        if isinstance(value, torch.Tensor):
            self.release(value)
        elif isinstance(value, (tuple, list)):
            for item in value:
                self.release_value(item)

    def reset_stats(self) -> None:
        self.allocated = 0
        self.reused = 0
        self.live = 0
        self.peak_live = 0


class ChronosFXStandaloneExecutor:
    def __init__(
        self,
        gm: torch.fx.GraphModule,
        *,
        num_streams: int = 1,
        use_cuda_graph: bool = False,
        example_inputs: Optional[Tuple[Any, ...]] = None,
        debug: bool = False,
        schedule_policy: str = "topo",
    ):
        self.gm = gm
        self.removed_v_final_outputs = 0
        self.num_streams = max(1, int(num_streams))
        self.use_cuda_graph = bool(use_cuda_graph)
        self.debug = bool(debug)
        if schedule_policy not in ("topo", "ready"):
            raise ValueError(f"unsupported FX standalone schedule_policy={schedule_policy!r}")
        self.schedule_policy = schedule_policy
        self.nodes: List[torch.fx.Node] = list(gm.graph.nodes)
        self.placeholders = [node for node in self.nodes if node.op == "placeholder"]
        self.output_node = next((node for node in self.nodes if node.op == "output"), None)
        if self.output_node is None:
            raise RuntimeError("FX graph has no output node")

        self.deps: Dict[torch.fx.Node, Set[torch.fx.Node]] = {}
        self.users: Dict[torch.fx.Node, Set[torch.fx.Node]] = defaultdict(set)
        self.levels: Dict[torch.fx.Node, int] = {}
        self.groups: Dict[int, List[torch.fx.Node]] = defaultdict(list)
        self.executable_nodes: List[torch.fx.Node] = []
        self._build_dag()

        self.streams: List[torch.cuda.Stream] = []
        self.node_events: Dict[torch.fx.Node, torch.cuda.Event] = {}
        self.execution_plan: List[_ExecStep] = []
        self._plan_compiled = False
        self.node_kinds: Dict[torch.fx.Node, str] = {}
        self.producer_compute_deps: Dict[torch.fx.Node, Set[torch.fx.Node]] = {}
        self.compute_nodes: List[torch.fx.Node] = []
        self.compute_users: Dict[torch.fx.Node, Set[torch.fx.Node]] = defaultdict(set)
        self.node_to_stream: Dict[torch.fx.Node, Optional[int]] = {}
        self.requested_num_streams = self.num_streams
        self.effective_num_streams = self.num_streams
        self._affinity_hits = 0
        self._affinity_misses = 0
        self._same_stream_edges = 0
        self._cross_stream_edges = 0
        self._event_count = 0
        self.output_wait_events: List[torch.cuda.Event] = []
        self.output_dependency_nodes: Set[torch.fx.Node] = set()
        self._env_pruned_nodes = 0
        self._env_peak_live_tensor_count = 0
        self._env_peak_live_tensor_bytes = 0
        self.arg_resolvers: Dict[torch.fx.Node, Callable[[Dict[torch.fx.Node, Any]], Tuple[tuple, dict]]] = {}
        self.output_arg = self.output_node.args[0]
        self.pruned_output_arg = self.output_arg
        self.buffer_pool = TensorBufferPool(debug=self.debug)
        self._pool_owned_nodes: Set[torch.fx.Node] = set()
        self._conv_out_available = True
        self._conv_out_disable_reason = ""
        self._graph = None
        self._graph_captured = False
        self._graph_fallback_reason = ""
        self._static_inputs: Optional[Tuple[Any, ...]] = None
        self._static_output: Any = None

        if self.debug:
            edges = sum(len(v) for v in self.deps.values())
            print(
                f"[FX_STANDALONE] nodes={len(self.nodes)} num_streams={self.num_streams} "
                f"cuda_graph={self.use_cuda_graph} schedule_policy={self.schedule_policy}"
            )
            print(f"[STATE_PRUNE] removed_v_final_outputs={self.removed_v_final_outputs}")
            print(f"[FX_DAG] nodes={len(self.executable_nodes)} edges={edges} levels={len(self.groups)}")
            for level in sorted(self.groups):
                names = ",".join(node.name for node in self.groups[level])
                print(f"[FX_LEVEL_GROUP] level={level} nodes={names}")

        if self.num_streams == 1:
            self._analyze_compute_dependencies()
            self.execution_plan = self._build_execution_plan()
            self._plan_compiled = True

        if self.use_cuda_graph and example_inputs is not None:
            self._try_capture(tuple(example_inputs))

    def _build_dag(self) -> None:
        produced: Set[torch.fx.Node] = set()
        for node in self.nodes:
            deps = {dep for dep in _iter_nodes((node.args, node.kwargs)) if dep in produced}
            self.deps[node] = deps
            for dep in deps:
                self.users[dep].add(node)

            if node.op not in ("placeholder", "output"):
                self.executable_nodes.append(node)
                level = max((self.levels.get(dep, 0) + 1 for dep in deps), default=0)
                self.levels[node] = level
                self.groups[level].append(node)
            else:
                self.levels[node] = 0
            produced.add(node)

    def _is_prunable_v_final_output(self, value: Any) -> bool:
        if not isinstance(value, torch.fx.Node):
            return False
        if value.op != "call_function" or value.target is not operator.getitem:
            return False
        if _getitem_index(value) != 1 or not value.args or not isinstance(value.args[0], torch.fx.Node):
            return False
        producer = value.args[0]
        if not _is_chronos_fused_temporal_op(producer):
            return False
        real_users = [user for user in value.users if user is not self.output_node]
        return len(real_users) == 0

    def _compute_output_prune(self) -> None:
        value = self.output_arg
        if not isinstance(value, tuple):
            return
        kept = []
        removed = 0
        for index, item in enumerate(value):
            if index > 0 and self._is_prunable_v_final_output(item):
                removed += 1
                continue
            kept.append(item)
        if removed:
            self.pruned_output_arg = tuple(kept)
            self.removed_v_final_outputs = removed

    def _resolve(self, env: Dict[torch.fx.Node, Any], value: Any) -> Any:
        return map_arg(value, lambda node: env[node])

    def _fetch_attr(self, target: str) -> Any:
        atom = self.gm
        for item in target.split("."):
            atom = getattr(atom, item)
        return atom

    def _execute_node(self, node: torch.fx.Node, env: Dict[torch.fx.Node, Any]) -> Any:
        args = self._resolve(env, node.args)
        kwargs = self._resolve(env, node.kwargs)
        try:
            if node.op == "get_attr":
                return self._fetch_attr(node.target)
            if node.op == "call_function":
                return node.target(*args, **kwargs)
            if node.op == "call_method":
                if not args:
                    raise RuntimeError("call_method node has no receiver")
                receiver, *method_args = args
                return getattr(receiver, node.target)(*method_args, **kwargs)
            if node.op == "call_module":
                return self.gm.get_submodule(node.target)(*args, **kwargs)
        except Exception as exc:
            raise RuntimeError(
                "FX standalone executor failed at "
                f"name={node.name} op={node.op} target={node.target} "
                f"args={_summarize(args)} kwargs={_summarize(kwargs)} "
                f"exc_type={type(exc).__name__} exc={repr(exc)}"
            ) from exc
        raise RuntimeError(
            f"Unsupported FX node: name={node.name} op={node.op} target={node.target} "
            f"args={_summarize(node.args)} kwargs={_summarize(node.kwargs)}"
        )

    def _execute_node_resolved(
        self,
        node: torch.fx.Node,
        args: tuple,
        kwargs: dict,
    ) -> Any:
        try:
            if node.op == "get_attr":
                return self._fetch_attr(node.target)
            if node.op == "call_function":
                return node.target(*args, **kwargs)
            if node.op == "call_method":
                if not args:
                    raise RuntimeError("call_method node has no receiver")
                receiver, *rest = args
                return getattr(receiver, node.target)(*rest, **kwargs)
            if node.op == "call_module":
                return self.gm.get_submodule(node.target)(*args, **kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"FX standalone executor failed at name={node.name} "
                f"op={node.op} target={node.target} "
                f"args={_summarize(args)} kwargs={_summarize(kwargs)} "
                f"exc_type={type(exc).__name__} exc={repr(exc)}"
            ) from exc
        raise RuntimeError(
            f"Unsupported FX node: name={node.name} op={node.op} "
            f"target={node.target}"
        )

    def _bind_placeholders(self, inputs: Tuple[Any, ...]) -> Dict[torch.fx.Node, Any]:
        if len(inputs) != len(self.placeholders):
            raise ValueError(f"expected {len(self.placeholders)} inputs, got {len(inputs)}")
        return {node: value for node, value in zip(self.placeholders, inputs)}

    def _run_serial(self, inputs: Tuple[Any, ...]) -> Any:
        env = self._bind_placeholders(inputs)
        for node in self.nodes:
            if node.op in ("placeholder",):
                continue
            if node.op == "output":
                return self._resolve(env, self.pruned_output_arg)
            env[node] = self._execute_node(node, env)
        raise RuntimeError("FX graph execution reached end without output")

    def _analyze_compute_dependencies(self) -> None:
        self.node_kinds = {
            node: _classify_exec_node(node)
            for node in self.executable_nodes
        }
        self.arg_resolvers = {
            node: _compile_arg_resolver(node)
            for node in self.executable_nodes
        }
        self.producer_compute_deps = {}
        for node in self.nodes:
            producer_deps: Set[torch.fx.Node] = set()
            for dep in self.deps.get(node, ()):
                dep_kind = self.node_kinds.get(dep)
                if dep_kind in (_SIDE_STREAM_COMPUTE, _MAIN_STREAM_COMPUTE):
                    producer_deps.add(dep)
                else:
                    producer_deps.update(self.producer_compute_deps.get(dep, set()))
            self.producer_compute_deps[node] = producer_deps
        self.compute_nodes = [
            node
            for node in self.executable_nodes
            if self.node_kinds.get(node) in (_SIDE_STREAM_COMPUTE, _MAIN_STREAM_COMPUTE)
        ]
        self.compute_users = defaultdict(set)
        for node in self.compute_nodes:
            for dep in self.producer_compute_deps.get(node, ()):
                if dep in self.compute_nodes:
                    self.compute_users[dep].add(node)

    def _build_execution_plan(self) -> List[_ExecStep]:
        if self.schedule_policy == "ready" and self.effective_num_streams > 1:
            ordered_nodes = self._build_ready_schedule_order()
        else:
            ordered_nodes = list(self.compute_nodes)
        self._assign_streams(ordered_nodes)
        self._create_required_events()
        plan = self._make_execution_steps(ordered_nodes)
        output_producers = self._output_producer_nodes()
        self.output_dependency_nodes = set(output_producers)
        self.output_wait_events = [
            self.node_events[dep]
            for dep in output_producers
            if dep in self.node_events
        ]
        if self.debug:
            self._print_schedule_diagnostics()
            self._print_stream_stats(plan)
        return plan

    def _build_ready_schedule_order(self) -> List[torch.fx.Node]:
        ordered_nodes: List[torch.fx.Node] = []
        indegree: Dict[torch.fx.Node, int] = {
            node: sum(1 for dep in self.producer_compute_deps.get(node, ()) if dep in self.compute_nodes)
            for node in self.compute_nodes
        }
        ready = deque(node for node in self.compute_nodes if indegree[node] == 0)
        scheduled: Set[torch.fx.Node] = set()

        while ready:
            node = ready.popleft()
            if node in scheduled:
                continue
            scheduled.add(node)
            ordered_nodes.append(node)
            for user in sorted(self.compute_users.get(node, ()), key=lambda n: self.nodes.index(n)):
                indegree[user] -= 1
                if indegree[user] == 0:
                    ready.append(user)

        if len(scheduled) != len(self.compute_nodes):
            missing = [node.name for node in self.compute_nodes if node not in scheduled]
            raise RuntimeError(f"FX standalone ready scheduler did not schedule all compute nodes: {missing[:8]}")
        return ordered_nodes

    def _assign_streams(self, ordered_nodes: List[torch.fx.Node]) -> None:
        has_streams = bool(self.streams)
        stream_loads = [0 for _ in self.streams]
        self.node_to_stream = {}
        self._affinity_hits = 0
        self._affinity_misses = 0
        for node in ordered_nodes:
            node_kind = self.node_kinds.get(node, _classify_exec_node(node))
            if node_kind == _SIDE_STREAM_COMPUTE and has_streams:
                stream_idx = self._choose_stream(node, stream_loads)
                stream_loads[stream_idx] += 1
            else:
                stream_idx = None
            self.node_to_stream[node] = stream_idx

    def _make_execution_steps(self, ordered_nodes: List[torch.fx.Node]) -> List[_ExecStep]:
        plan: List[_ExecStep] = []
        for node in ordered_nodes:
            node_kind = self.node_kinds.get(node, _classify_exec_node(node))
            node_stream = self.node_to_stream.get(node)
            wait_events = []
            for dep in self.producer_compute_deps.get(node, ()):
                if dep not in self.node_events:
                    continue
                dep_stream = self.node_to_stream.get(dep)
                if dep_stream != node_stream:
                    wait_events.append(self.node_events[dep])
            plan.append(
                _ExecStep(
                    node=node,
                    stream_idx=node_stream,
                    wait_events=wait_events,
                    record_event=self.node_events.get(node),
                    node_kind=node_kind,
                    arg_resolver=self.arg_resolvers[node],
                )
            )
        return plan

    def _choose_stream(self, node: torch.fx.Node, stream_loads: List[int]) -> int:
        if not stream_loads:
            return 0
        producer_streams = [
            self.node_to_stream[dep]
            for dep in self.producer_compute_deps.get(node, ())
            if dep in self.node_to_stream and self.node_to_stream[dep] is not None
        ]
        min_stream = min(range(len(stream_loads)), key=lambda idx: stream_loads[idx])
        min_load = stream_loads[min_stream]
        if producer_streams:
            counts: Dict[int, int] = defaultdict(int)
            for stream_idx in producer_streams:
                counts[int(stream_idx)] += 1
            candidate = max(sorted(counts), key=lambda idx: (counts[idx], -stream_loads[idx]))
            imbalance_threshold = 2
            if stream_loads[candidate] <= min_load + imbalance_threshold:
                self._affinity_hits += 1
                if self.debug:
                    print(
                        f"[FX_STREAM_AFFINITY] node={node.name} hit=True stream={candidate} "
                        f"candidate_load={stream_loads[candidate]} min_stream={min_stream} min_load={min_load}"
                    )
                return candidate
            self._affinity_misses += 1
            if self.debug:
                print(
                    f"[FX_STREAM_AFFINITY] node={node.name} hit=False candidate={candidate} "
                    f"candidate_load={stream_loads[candidate]} min_stream={min_stream} min_load={min_load}"
                )
        return min_stream

    def _create_required_events(self) -> None:
        required: Set[torch.fx.Node] = set()
        same_stream_edges = 0
        cross_stream_edges = 0
        for node in self.compute_nodes:
            node_stream = self.node_to_stream.get(node)
            for dep in self.producer_compute_deps.get(node, ()):
                if dep not in self.compute_nodes:
                    continue
                dep_stream = self.node_to_stream.get(dep)
                if dep_stream == node_stream:
                    same_stream_edges += 1
                else:
                    cross_stream_edges += 1
                    required.add(dep)
        for node in self._output_producer_nodes():
            if self.node_to_stream.get(node) is not None:
                required.add(node)
        self.node_events = {
            node: torch.cuda.Event(enable_timing=False, blocking=False)
            for node in required
        }
        self._same_stream_edges = same_stream_edges
        self._cross_stream_edges = cross_stream_edges
        self._event_count = len(self.node_events)

    def _compute_levels(self) -> Tuple[Dict[torch.fx.Node, int], Dict[int, List[torch.fx.Node]]]:
        levels: Dict[torch.fx.Node, int] = {}
        groups: Dict[int, List[torch.fx.Node]] = defaultdict(list)
        for node in self.compute_nodes:
            deps = [dep for dep in self.producer_compute_deps.get(node, ()) if dep in levels]
            level = max((levels[dep] + 1 for dep in deps), default=0)
            levels[node] = level
            groups[level].append(node)
        return levels, groups

    def _max_compute_ready_width(self) -> int:
        _levels, groups = self._compute_levels()
        if not groups:
            return 1
        return max(len(nodes) for nodes in groups.values())

    def _print_schedule_diagnostics(self) -> None:
        metadata_count = sum(1 for kind in self.node_kinds.values() if kind == _METADATA_INLINE)
        side_count = sum(1 for kind in self.node_kinds.values() if kind == _SIDE_STREAM_COMPUTE)
        main_count = sum(1 for kind in self.node_kinds.values() if kind == _MAIN_STREAM_COMPUTE)
        compute_count = side_count + main_count
        print(
            f"[FX_SCHED_SUMMARY] fx_nodes={len(self.nodes)} metadata_inline={metadata_count} "
            f"compute_nodes={compute_count} side_stream_compute={side_count} "
            f"main_stream_compute={main_count} num_streams={self.num_streams} "
            f"cuda_graph={self.use_cuda_graph} schedule_policy={self.schedule_policy}"
        )
        levels, groups = self._compute_levels()
        edge_count = sum(
            1
            for node in self.compute_nodes
            for dep in self.producer_compute_deps.get(node, ())
            if dep in self.compute_nodes
        )
        widths = [len(groups[level]) for level in sorted(groups)]
        critical_path_len = (max(levels.values()) + 1) if levels else 0
        max_ready_width = max(widths) if widths else 0
        avg_ready_width = (sum(widths) / len(widths)) if widths else 0.0
        print(
            f"[FX_COMPUTE_DAG] compute_nodes={len(self.compute_nodes)} compute_edges={edge_count} "
            f"critical_path_len={critical_path_len} max_ready_width={max_ready_width} "
            f"avg_ready_width={avg_ready_width:.2f}"
        )
        for level in sorted(groups):
            entries = ",".join(
                f"{node.name}:{_target_short(node.target)}"
                for node in groups[level]
            )
            print(f"[FX_COMPUTE_LEVEL] level={level} width={len(groups[level])} nodes={entries}")
        fused_index = 0
        for node in self.compute_nodes:
            if not _is_chronos_fused_temporal_op(node):
                continue
            input_arg = node.args[0] if node.args else None
            if isinstance(input_arg, (tuple, list)):
                input_list_len: Any = len(input_arg)
                input_items = list(input_arg)
            else:
                input_list_len = "packed" if isinstance(input_arg, torch.fx.Node) else "unknown"
                input_items = [input_arg] if isinstance(input_arg, torch.fx.Node) else []
            shape_texts = []
            for item in input_items[:4]:
                val = item.meta.get("val") if isinstance(item, torch.fx.Node) else None
                if isinstance(val, torch.Tensor):
                    shape_texts.append(str(tuple(val.shape)))
                else:
                    shape_texts.append("unknown")
            if len(input_items) > 4:
                shape_texts.append("...")
            output_users = ",".join(user.name for user in node.users)
            producer_fused_deps = ",".join(
                dep.name
                for dep in self.producer_compute_deps.get(node, ())
                if _is_chronos_fused_temporal_op(dep)
            )
            print(
                f"[CHRONOS_FUSED_NODE] idx={fused_index} name={node.name} "
                f"target={_target_short(node.target)} input_list_len={input_list_len} "
                f"input_shapes={'|'.join(shape_texts)} output_users={output_users} "
                f"producer_fused_deps={producer_fused_deps}"
            )
            fused_index += 1
        self._print_chronos_window_diagnostics()

    def _producer_fused_deps(self, node: torch.fx.Node) -> Set[torch.fx.Node]:
        fused_deps: Set[torch.fx.Node] = set()
        stack = list(self.producer_compute_deps.get(node, ()))
        visited: Set[torch.fx.Node] = set()
        while stack:
            dep = stack.pop()
            if dep in visited:
                continue
            visited.add(dep)
            if _is_chronos_fused_temporal_op(dep):
                fused_deps.add(dep)
                continue
            stack.extend(self.producer_compute_deps.get(dep, ()))
        return fused_deps

    def _compute_fused_levels(
        self,
        fused_nodes: List[torch.fx.Node],
        fused_deps: Dict[torch.fx.Node, Set[torch.fx.Node]],
    ) -> Tuple[Dict[torch.fx.Node, int], Dict[int, List[torch.fx.Node]]]:
        fused_set = set(fused_nodes)
        levels: Dict[torch.fx.Node, int] = {}
        groups: Dict[int, List[torch.fx.Node]] = defaultdict(list)
        for node in fused_nodes:
            deps = [dep for dep in fused_deps.get(node, ()) if dep in fused_set and dep in levels]
            level = max((levels[dep] + 1 for dep in deps), default=0)
            levels[node] = level
            groups[level].append(node)
        return levels, groups

    def _max_fused_ready_width(self) -> int:
        fused_nodes = [node for node in self.compute_nodes if _is_chronos_fused_temporal_op(node)]
        if not fused_nodes:
            return 1
        fused_deps = {node: self._producer_fused_deps(node) for node in fused_nodes}
        _levels, groups = self._compute_fused_levels(fused_nodes, fused_deps)
        if not groups:
            return 1
        return max(len(nodes) for nodes in groups.values())

    def _chronos_input_list_len(self, node: torch.fx.Node) -> Any:
        input_arg = node.args[0] if node.args else None
        if isinstance(input_arg, (tuple, list)):
            return len(input_arg)
        if isinstance(input_arg, torch.fx.Node):
            value = input_arg.meta.get("val")
            if isinstance(value, torch.Tensor) and value.dim() > 0:
                return int(value.shape[0])
            return "packed"
        return "unknown"

    def _print_chronos_window_diagnostics(self) -> None:
        fused_nodes = [node for node in self.compute_nodes if _is_chronos_fused_temporal_op(node)]
        fused_deps = {node: self._producer_fused_deps(node) for node in fused_nodes}
        fused_levels, fused_groups = self._compute_fused_levels(fused_nodes, fused_deps)
        widths = [len(fused_groups[level]) for level in sorted(fused_groups)]
        max_width = max(widths) if widths else 0
        avg_width = (sum(widths) / len(widths)) if widths else 0.0
        critical_path_len = (max(fused_levels.values()) + 1) if fused_levels else 0
        layer_ids = {node.meta.get("chronos_layer_id", "unknown") for node in fused_nodes}
        window_ids = {node.meta.get("chronos_window_id", "unknown") for node in fused_nodes}
        edge_count = sum(len(deps) for deps in fused_deps.values())
        print(
            f"[CHRONOS_WINDOW_DAG_SUMMARY] fused_nodes={len(fused_nodes)} "
            f"num_layers={len(layer_ids)} num_windows={len(window_ids)} "
            f"fused_edges={edge_count} max_fused_ready_width={max_width} "
            f"avg_fused_ready_width={avg_width:.2f} fused_critical_path_len={critical_path_len}"
        )
        for level in sorted(fused_groups):
            entries = ", ".join(
                "("
                f"{node.meta.get('chronos_layer_id', 'unknown')},"
                f"{node.meta.get('chronos_window_id', 'unknown')},"
                f"{node.meta.get('chronos_op_kind', 'unknown')},"
                f"{node.name}"
                ")"
                for node in fused_groups[level]
            )
            print(f"[CHRONOS_FUSED_LEVEL] level={level} width={len(fused_groups[level])} nodes={entries}")
        for node in fused_nodes:
            stream_idx = self.node_to_stream.get(node)
            stream_text = "main" if stream_idx is None else str(stream_idx)
            producer_fused_deps = ",".join(dep.name for dep in sorted(fused_deps[node], key=lambda n: self.nodes.index(n)))
            print(
                f"[CHRONOS_WINDOW_NODE] name={node.name} "
                f"kind={node.meta.get('chronos_op_kind', 'unknown')} "
                f"layer_id={node.meta.get('chronos_layer_id', 'unknown')} "
                f"window_id={node.meta.get('chronos_window_id', 'unknown')} "
                f"time_range={node.meta.get('chronos_time_range', 'unknown')} "
                f"input_list_len={self._chronos_input_list_len(node)} "
                f"compute_level={fused_levels.get(node, 'unknown')} "
                f"stream={stream_text} "
                f"producer_fused_deps={producer_fused_deps}"
            )

    def _print_stream_stats(self, plan: List[_ExecStep]) -> None:
        stream_counts = [0 for _ in self.streams]
        main_steps = 0
        side_steps = 0
        for step in plan:
            if step.stream_idx is None:
                main_steps += 1
            else:
                stream_counts[step.stream_idx] += 1
                side_steps += 1
        stream_parts = " ".join(f"stream{idx}_steps={count}" for idx, count in enumerate(stream_counts))
        print(f"[FX_STREAM_STATS] {stream_parts} side_stream_steps={side_steps} main_stream_steps={main_steps}")
        print(
            f"[FX_EVENT_STATS] events={self._event_count} cross_stream_edges={self._cross_stream_edges} "
            f"same_stream_edges={self._same_stream_edges} affinity_hits={self._affinity_hits} "
            f"affinity_misses={self._affinity_misses}"
        )

    def _output_producer_nodes(self) -> Set[torch.fx.Node]:
        producers: Set[torch.fx.Node] = set()
        for node in _iter_nodes(self.pruned_output_arg):
            producers.update(self.producer_compute_deps.get(node, set()))
            if self.node_kinds.get(node) in (_SIDE_STREAM_COMPUTE, _MAIN_STREAM_COMPUTE):
                producers.add(node)
        return producers

    def _materialize_metadata_deps(self, node: torch.fx.Node, env: Dict[torch.fx.Node, Any]) -> None:
        for dep in self.deps.get(node, ()):
            if dep in env:
                continue
            if self.node_kinds.get(dep) != _METADATA_INLINE:
                continue
            self._materialize_metadata_deps(dep, env)
            args, kwargs = self.arg_resolvers[dep](env)
            env[dep] = self._execute_node_resolved(dep, args, kwargs)

    def _try_execute_fused_temporal_conv_out(
        self,
        node: torch.fx.Node,
        args: tuple,
        kwargs: dict,
    ) -> Optional[Any]:
        if not self._conv_out_available or kwargs or not _is_chronos_fused_temporal_conv_state_node(node):
            return None
        if len(args) != 12:
            return None
        xs, weight, bias, v_init, stride, padding, dilation, groups, v_threshold, v_reset, tau, detach_reset = args
        if not isinstance(xs, (list, tuple)) or not xs or not isinstance(xs[0], torch.Tensor):
            return None
        first = xs[0]
        if not first.is_cuda or not isinstance(v_init, torch.Tensor):
            return None
        spike_out = self.buffer_pool.acquire((len(xs),) + tuple(v_init.shape), first.dtype, first.device)
        v_out = self.buffer_pool.acquire(tuple(v_init.shape), v_init.dtype, v_init.device)
        try:
            x_seq = torch.stack(tuple(xs), dim=0).contiguous()
            from runtime.triton_convlif_backend import run_triton_fused_temporal_conv_lif_state_packed_out

            compute_dtype = "float16" if first.dtype == torch.float16 else "float32"
            run_triton_fused_temporal_conv_lif_state_packed_out(
                x_seq,
                weight,
                bias,
                v_init,
                stride,
                padding,
                dilation,
                groups,
                v_threshold,
                v_reset,
                tau,
                detach_reset,
                spike_out,
                v_out,
                strict=True,
                verbose=False,
                use_autotune=True,
                compute_dtype=compute_dtype,
            )
            self._pool_owned_nodes.add(node)
            return spike_out, v_out
        except Exception as exc:
            self._conv_out_available = False
            self._conv_out_disable_reason = f"{type(exc).__name__}: {repr(exc)}"
            if self.debug:
                print(f"[BUFFER_POOL] disable_conv_out reason={self._conv_out_disable_reason}")
            self.buffer_pool.release(spike_out)
            self.buffer_pool.release(v_out)
            return None

    def _execute_node_resolved_pooled(
        self,
        node: torch.fx.Node,
        args: tuple,
        kwargs: dict,
    ) -> Any:
        out_value = self._try_execute_fused_temporal_conv_out(node, args, kwargs)
        if out_value is not None:
            return out_value
        return self._execute_node_resolved(node, args, kwargs)

    def _release_node_if_dead(
        self,
        node: torch.fx.Node,
        env: Dict[torch.fx.Node, Any],
        remaining_uses: Dict[torch.fx.Node, int],
        retained_nodes: Set[torch.fx.Node],
    ) -> None:
        if node in retained_nodes or remaining_uses.get(node, 0) > 0 or node not in self._pool_owned_nodes:
            return
        if node in env:
            self.buffer_pool.release_value(env[node])
            self._pool_owned_nodes.discard(node)

    def _make_remaining_user_counts(self) -> Dict[torch.fx.Node, int]:
        remaining: Dict[torch.fx.Node, int] = defaultdict(int)
        for node in self.nodes:
            remaining[node] = len(node.users)
        return remaining

    def _is_env_prune_protected(self, node: torch.fx.Node) -> bool:
        return (
            node.op == "placeholder"
            or node.op == "get_attr"
            or node in self.output_dependency_nodes
            or node is self.output_node
        )

    def _prune_env_node_if_dead(
        self,
        node: torch.fx.Node,
        env: Dict[torch.fx.Node, Any],
        remaining_uses: Dict[torch.fx.Node, int],
    ) -> None:
        if remaining_uses.get(node, 0) > 0 or self._is_env_prune_protected(node) or node not in env:
            return
        del env[node]
        self._env_pruned_nodes += 1
        for dep in self.deps.get(node, ()):
            remaining_uses[dep] -= 1
            self._prune_env_node_if_dead(dep, env, remaining_uses)

    def _consume_env_inputs(
        self,
        node: torch.fx.Node,
        env: Dict[torch.fx.Node, Any],
        remaining_uses: Dict[torch.fx.Node, int],
    ) -> None:
        direct_deps = set(self.deps.get(node, ()))
        for dep in direct_deps:
            remaining_uses[dep] -= 1
            self._prune_env_node_if_dead(dep, env, remaining_uses)
        self._prune_env_node_if_dead(node, env, remaining_uses)

    def _value_tensor_stats(self, value: Any, seen: Set[int]) -> Tuple[int, int]:
        if isinstance(value, torch.Tensor):
            ident = id(value)
            if ident in seen:
                return 0, 0
            seen.add(ident)
            return 1, int(value.numel() * value.element_size())
        if isinstance(value, (tuple, list)):
            count = 0
            nbytes = 0
            for item in value:
                item_count, item_bytes = self._value_tensor_stats(item, seen)
                count += item_count
                nbytes += item_bytes
            return count, nbytes
        if isinstance(value, dict):
            count = 0
            nbytes = 0
            for item in value.values():
                item_count, item_bytes = self._value_tensor_stats(item, seen)
                count += item_count
                nbytes += item_bytes
            return count, nbytes
        return 0, 0

    def _update_env_live_stats(self, env: Dict[torch.fx.Node, Any]) -> None:
        if not self.debug:
            return
        seen: Set[int] = set()
        count = 0
        nbytes = 0
        for value in env.values():
            item_count, item_bytes = self._value_tensor_stats(value, seen)
            count += item_count
            nbytes += item_bytes
        self._env_peak_live_tensor_count = max(self._env_peak_live_tensor_count, count)
        self._env_peak_live_tensor_bytes = max(self._env_peak_live_tensor_bytes, nbytes)

    def _run_single_stream_pooled(self, inputs: Tuple[Any, ...]) -> Any:
        if not self._plan_compiled:
            self._analyze_compute_dependencies()
            self.execution_plan = self._build_execution_plan()
            self._plan_compiled = True
        self.buffer_pool.reset_stats()
        self._pool_owned_nodes.clear()
        env = self._bind_placeholders(inputs)
        retained_nodes = self._output_producer_nodes()
        remaining_uses: Dict[torch.fx.Node, int] = defaultdict(int)
        for step in self.execution_plan:
            for dep in self.producer_compute_deps.get(step.node, ()):
                remaining_uses[dep] += 1

        for step in self.execution_plan:
            self._materialize_metadata_deps(step.node, env)
            args, kwargs = step.arg_resolver(env)
            env[step.node] = self._execute_node_resolved_pooled(step.node, args, kwargs)
            for dep in self.producer_compute_deps.get(step.node, ()):
                remaining_uses[dep] -= 1
                self._release_node_if_dead(dep, env, remaining_uses, retained_nodes)
            self._release_node_if_dead(step.node, env, remaining_uses, retained_nodes)

        self._materialize_metadata_deps(self.output_node, env)
        output = self._resolve(env, self.pruned_output_arg)
        if self.debug:
            print(
                f"[BUFFER_POOL] allocated={self.buffer_pool.allocated} "
                f"reused={self.buffer_pool.reused} peak_live={self.buffer_pool.peak_live}"
            )
        return output

    def _ensure_cuda_schedule(self, inputs: Tuple[Any, ...]) -> bool:
        if self.num_streams <= 1 or not torch.cuda.is_available():
            return False
        tensors = [value for value in inputs if isinstance(value, torch.Tensor)]
        if not tensors or not any(t.is_cuda for t in tensors):
            return False
        if not self.streams:
            self._analyze_compute_dependencies()
            max_compute_ready_width = self._max_compute_ready_width()
            if max_compute_ready_width <= 1:
                effective = 1
            elif max_compute_ready_width == 2:
                effective = min(self.num_streams, 2)
            else:
                effective = min(self.num_streams, max_compute_ready_width)
            self.effective_num_streams = max(1, int(effective))
            if self.debug:
                print(
                    f"[FX_STREAM_CLAMP] requested={self.num_streams} effective={self.effective_num_streams} "
                    f"max_compute_ready_width={max_compute_ready_width}"
                )
            self.streams = [torch.cuda.Stream(device=tensors[0].device) for _ in range(self.effective_num_streams)]
            self._plan_compiled = False
        if not self._plan_compiled:
            if not self.node_kinds:
                self._analyze_compute_dependencies()
            self.execution_plan = self._build_execution_plan()
            self._plan_compiled = True
        return True

    def _run_multistream(self, inputs: Tuple[Any, ...]) -> Any:
        if not self._ensure_cuda_schedule(inputs):
            return self._run_serial(inputs)

        env = self._bind_placeholders(inputs)
        remaining_uses = self._make_remaining_user_counts()
        self._env_pruned_nodes = 0
        self._env_peak_live_tensor_count = 0
        self._env_peak_live_tensor_bytes = 0
        self._update_env_live_stats(env)
        main_stream = torch.cuda.current_stream()
        for stream in self.streams:
            stream.wait_stream(main_stream)

        for idx, step in enumerate(self.execution_plan):
            nvtx_label = f"FXNODE:{step.node.name}:{_target_short(step.node.target)}"
            if step.stream_idx is None:
                for event in step.wait_events:
                    main_stream.wait_event(event)
                self._materialize_metadata_deps(step.node, env)
                args, kwargs = step.arg_resolver(env)
                if self.debug:
                    torch.cuda.nvtx.range_push(f"{nvtx_label}:stream=main")
                try:
                    env[step.node] = self._execute_node_resolved(step.node, args, kwargs)
                finally:
                    if self.debug:
                        torch.cuda.nvtx.range_pop()
                if step.record_event is not None:
                    step.record_event.record(main_stream)
            else:
                stream = self.streams[step.stream_idx]
                for event in step.wait_events:
                    stream.wait_event(event)
                self._materialize_metadata_deps(step.node, env)
                with torch.cuda.stream(stream):
                    args, kwargs = step.arg_resolver(env)
                    if self.debug:
                        torch.cuda.nvtx.range_push(f"{nvtx_label}:stream={step.stream_idx}")
                    try:
                        env[step.node] = self._execute_node_resolved(step.node, args, kwargs)
                    finally:
                        if self.debug:
                            torch.cuda.nvtx.range_pop()
                if step.record_event is not None:
                    step.record_event.record(stream)

            # Keep scheduled execution conservative: FX metadata/getitem chains
            # can share tensor producers in ways that are not represented by
            # producer-event deps. Dynamic env pruning may delete those values
            # before a later metadata materialization resolves them.
            self._update_env_live_stats(env)

            if self.debug:
                producer_deps = ",".join(dep.name for dep in self.producer_compute_deps.get(step.node, ()))
                sid = "main" if step.stream_idx is None else step.stream_idx
                print(
                    f"[FX_STREAM_ASSIGN] idx={idx} node={step.node.name} kind={step.node_kind} "
                    f"stream={sid} producer_deps={producer_deps} target={_target_short(step.node.target)}"
                )

        for event in self.output_wait_events:
            main_stream.wait_event(event)
        self._materialize_metadata_deps(self.output_node, env)
        output = self._resolve(env, self.output_node.args[0])
        if self.debug:
            print(
                f"[FX_ENV_LIFETIME] peak_live_tensor_count={self._env_peak_live_tensor_count} "
                f"peak_live_tensor_bytes={self._env_peak_live_tensor_bytes} "
                f"pruned_nodes={self._env_pruned_nodes}"
            )
        return output

    def _run_uncaptured(self, inputs: Tuple[Any, ...]) -> Any:
        if self.num_streams > 1:
            return self._run_multistream(inputs)
        if any(isinstance(value, torch.Tensor) and value.is_cuda for value in inputs):
            return self._run_single_stream_pooled(inputs)
        return self._run_serial(inputs)

    def _try_capture(self, example_inputs: Tuple[Any, ...]) -> None:
        if not torch.cuda.is_available() or not any(
            isinstance(value, torch.Tensor) and value.is_cuda for value in example_inputs
        ):
            self._graph_fallback_reason = "CUDA graph requested without CUDA tensor inputs"
            if self.debug:
                print(f"[CUDA_GRAPH] enabled=True captured=False fallback=True reason={self._graph_fallback_reason}")
            return
        try:
            self._static_inputs = tuple(_clone_static_input(value) for value in example_inputs)
            if self.debug:
                device = next(value.device for value in example_inputs if isinstance(value, torch.Tensor) and value.is_cuda)
                print(
                    f"[MEMORY] before_capture allocated={torch.cuda.memory_allocated(device)} "
                    f"reserved={torch.cuda.memory_reserved(device)}"
                )
            for _ in range(2):
                warmup_output = self._run_uncaptured(self._static_inputs)
                del warmup_output
                self._static_output = None
            torch.cuda.synchronize()

            # streams and events must exist before capture.
            if self.num_streams > 1 and not self._ensure_cuda_schedule(self._static_inputs):
                self._graph_fallback_reason = "CUDA schedule could not be initialized before capture"
                if self.debug:
                    print(f"[CUDA_GRAPH] enabled=True captured=False fallback=True reason={self._graph_fallback_reason}")
                return
            if self.num_streams > 1 and not self.streams:
                self._graph_fallback_reason = "CUDA streams were not populated before capture"
                if self.debug:
                    print(f"[CUDA_GRAPH] enabled=True captured=False fallback=True reason={self._graph_fallback_reason}")
                return
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._static_output = self._run_uncaptured(self._static_inputs)
            self._graph = graph
            self._graph_captured = True
            if self.debug:
                device = next(value.device for value in example_inputs if isinstance(value, torch.Tensor) and value.is_cuda)
                print(
                    f"[MEMORY] after_capture allocated={torch.cuda.memory_allocated(device)} "
                    f"reserved={torch.cuda.memory_reserved(device)}"
                )
                print("[CUDA_GRAPH] enabled=True captured=True fallback=False reason=none")
        except Exception as exc:
            self._graph = None
            self._graph_captured = False
            self._graph_fallback_reason = f"{type(exc).__name__}: {exc}"
            if self.debug:
                print(
                    "[CUDA_GRAPH] enabled=True captured=False fallback=True "
                    f"reason={self._graph_fallback_reason}"
                )

    def __call__(self, *inputs: Any) -> Any:
        inputs_tuple = tuple(inputs)
        if self._graph_captured and self._graph is not None and self._static_inputs is not None:
            for dst, src in zip(self._static_inputs, inputs_tuple):
                _copy_input(dst, src)
            self._graph.replay()
            return self._static_output
        return self._run_uncaptured(inputs_tuple)


def build_fx_standalone_backend(
    gm: torch.fx.GraphModule,
    *,
    num_streams: int = 1,
    use_cuda_graph: bool = False,
    example_inputs: Optional[Tuple[Any, ...]] = None,
    debug: bool = False,
    schedule_policy: str = "topo",
):
    executor = ChronosFXStandaloneExecutor(
        gm,
        num_streams=num_streams,
        use_cuda_graph=use_cuda_graph,
        example_inputs=example_inputs,
        debug=debug,
        schedule_policy=schedule_policy,
    )
    return executor


def make_fx_standalone_torch_compile_backend(
    *,
    num_streams: int = 1,
    use_cuda_graph: bool = False,
    debug: bool = False,
    schedule_policy: str = "topo",
):
    def backend(gm: torch.fx.GraphModule, example_inputs, **_compile_kwargs):
        return build_fx_standalone_backend(
            gm,
            num_streams=num_streams,
            use_cuda_graph=use_cuda_graph,
            example_inputs=tuple(example_inputs),
            debug=debug,
            schedule_policy=schedule_policy,
        )

    return backend
