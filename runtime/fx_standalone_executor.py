"""Experimental FX-native executor for Chronos-rewritten GraphModules.

This module intentionally stays below the existing Inductor path: it executes
the final FX graph directly, optionally with a simple level-barrier multi-stream
schedule and CUDA Graph replay.
"""

from __future__ import annotations

import operator
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

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


class ChronosFXStandaloneExecutor:
    def __init__(
        self,
        gm: torch.fx.GraphModule,
        *,
        num_streams: int = 1,
        use_cuda_graph: bool = False,
        example_inputs: Optional[Tuple[Any, ...]] = None,
        debug: bool = False,
    ):
        self.gm = gm
        self.num_streams = max(1, int(num_streams))
        self.use_cuda_graph = bool(use_cuda_graph)
        self.debug = bool(debug)
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
        self._graph = None
        self._graph_captured = False
        self._graph_fallback_reason = ""
        self._static_inputs: Optional[Tuple[Any, ...]] = None
        self._static_output: Any = None

        if self.debug:
            edges = sum(len(v) for v in self.deps.values())
            print(
                f"[FX_STANDALONE] nodes={len(self.nodes)} num_streams={self.num_streams} "
                f"cuda_graph={self.use_cuda_graph}"
            )
            print(f"[FX_DAG] nodes={len(self.executable_nodes)} edges={edges} levels={len(self.groups)}")
            for level in sorted(self.groups):
                names = ",".join(node.name for node in self.groups[level])
                print(f"[FX_LEVEL_GROUP] level={level} nodes={names}")

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
                f"args={_summarize(args)} kwargs={_summarize(kwargs)}"
            ) from exc
        raise RuntimeError(
            f"Unsupported FX node: name={node.name} op={node.op} target={node.target} "
            f"args={_summarize(node.args)} kwargs={_summarize(node.kwargs)}"
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
                return self._resolve(env, node.args[0])
            env[node] = self._execute_node(node, env)
        raise RuntimeError("FX graph execution reached end without output")

    def _ensure_cuda_schedule(self, inputs: Tuple[Any, ...]) -> bool:
        if self.num_streams <= 1 or not torch.cuda.is_available():
            return False
        tensors = [value for value in inputs if isinstance(value, torch.Tensor)]
        if not tensors or not any(t.is_cuda for t in tensors):
            return False
        if not self.streams:
            self.streams = [torch.cuda.Stream(device=tensors[0].device) for _ in range(self.num_streams)]
            self.node_events = {
                node: torch.cuda.Event(enable_timing=False, blocking=False)
                for node in self.executable_nodes
            }
        return True

    def _run_multistream(self, inputs: Tuple[Any, ...]) -> Any:
        if not self._ensure_cuda_schedule(inputs):
            return self._run_serial(inputs)

        env = self._bind_placeholders(inputs)
        main_stream = torch.cuda.current_stream()
        for stream in self.streams:
            stream.wait_stream(main_stream)

        for level in sorted(self.groups):
            level_events = []
            for index, node in enumerate(self.groups[level]):
                stream = self.streams[index % len(self.streams)]
                if self.debug:
                    deps = ",".join(dep.name for dep in self.deps[node])
                    print(f"[STREAM_ASSIGN] level={level} node={node.name} stream={index % len(self.streams)} deps={deps}")
                with torch.cuda.stream(stream):
                    for dep in self.deps[node]:
                        event = self.node_events.get(dep)
                        if event is not None:
                            stream.wait_event(event)
                    env[node] = self._execute_node(node, env)
                    event = self.node_events[node]
                    event.record(stream)
                    level_events.append(event)
            for event in level_events:
                main_stream.wait_event(event)

        return self._resolve(env, self.output_node.args[0])

    def _run_uncaptured(self, inputs: Tuple[Any, ...]) -> Any:
        if self.num_streams > 1:
            return self._run_multistream(inputs)
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
            for _ in range(2):
                self._static_output = self._run_uncaptured(self._static_inputs)
            torch.cuda.synchronize()

            self._ensure_cuda_schedule(self._static_inputs)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._static_output = self._run_uncaptured(self._static_inputs)
            self._graph = graph
            self._graph_captured = True
            if self.debug:
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
):
    executor = ChronosFXStandaloneExecutor(
        gm,
        num_streams=num_streams,
        use_cuda_graph=use_cuda_graph,
        example_inputs=example_inputs,
        debug=debug,
    )
    return executor


def make_fx_standalone_torch_compile_backend(
    *,
    num_streams: int = 1,
    use_cuda_graph: bool = False,
    debug: bool = False,
):
    def backend(gm: torch.fx.GraphModule, example_inputs, **_compile_kwargs):
        return build_fx_standalone_backend(
            gm,
            num_streams=num_streams,
            use_cuda_graph=use_cuda_graph,
            example_inputs=tuple(example_inputs),
            debug=debug,
        )

    return backend

