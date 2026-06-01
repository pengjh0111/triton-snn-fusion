from typing import Any, Dict, Optional, Tuple

import torch


def _dynamo_counters():
    try:
        from torch._dynamo.utils import counters

        return counters
    except Exception:
        return {}


def snapshot_compile_counters() -> Dict[str, Dict[str, int]]:
    counters = _dynamo_counters()
    return {str(section): dict(values) for section, values in counters.items()}


def diff_compile_counters(
    before: Dict[str, Dict[str, int]],
    after: Dict[str, Dict[str, int]],
) -> Dict[str, Dict[str, int]]:
    diff: Dict[str, Dict[str, int]] = {}
    for section, values in after.items():
        base = before.get(section, {})
        section_diff = {
            key: int(value) - int(base.get(key, 0))
            for key, value in values.items()
            if int(value) - int(base.get(key, 0)) != 0
        }
        if section_diff:
            diff[section] = section_diff
    return diff


def summarize_cudagraph_check(
    *,
    model: str,
    case: str,
    compile_config: Dict[str, Any],
    compile_mode: bool,
    device: str,
    graph_count: Optional[int],
    counter_diff: Dict[str, Dict[str, int]],
    print_log: bool = True,
) -> Dict[str, Any]:
    inductor = counter_diff.get("inductor", {})
    graph_breaks = counter_diff.get("graph_break", {})
    stats = counter_diff.get("stats", {})
    cudagraph_counters = {
        key: value
        for key, value in inductor.items()
        if "cudagraph" in str(key).lower()
    }
    cudagraph_skip_counters = {
        key: value
        for key, value in cudagraph_counters.items()
        if any(token in str(key).lower() for token in ("skip", "fallback", "fail", "disable"))
    }
    cudagraph_recorded = any(
        "record" in str(key).lower() and int(value) > 0
        for key, value in cudagraph_counters.items()
    )
    cudagraph_expected = bool(
        compile_mode
        and compile_config.get("enable_cudagraphs")
        and str(device).startswith("cuda")
    )
    fallback_reasons = []
    if not compile_mode:
        fallback_reasons.append("case_not_compiled")
    if compile_mode and cudagraph_expected and not cudagraph_recorded:
        fallback_reasons.append("no_cudagraph_record_counter")
    for key, value in cudagraph_skip_counters.items():
        fallback_reasons.append(f"{key}={value}")
    if graph_count is not None and graph_count > 1:
        fallback_reasons.append(f"multiple_fx_graphs={graph_count}")
    for key, value in graph_breaks.items():
        fallback_reasons.append(f"graph_break:{key}={value}")

    full_graph = graph_count == 1 if graph_count is not None else None
    fallback_taken = bool(cudagraph_expected and fallback_reasons)
    status = {
        "model": model,
        "case": case,
        "compile_backend": compile_config.get("backend"),
        "compile_mode": compile_config.get("compile_mode"),
        "compile_options": compile_config.get("compile_options"),
        "cudagraph_expected": cudagraph_expected,
        "cudagraph_enabled": bool(cudagraph_recorded and not cudagraph_skip_counters),
        "fallback_taken": fallback_taken,
        "fallback_reason": "; ".join(fallback_reasons) if fallback_reasons else "",
        "full_graph": full_graph,
        "graph_count": graph_count,
        "graph_break_count": int(sum(graph_breaks.values())),
        "graph_breaks": graph_breaks,
        "cudagraph_counters": cudagraph_counters,
        "compile_stats": stats,
    }
    if print_log:
        print(f"[CHRONOS_CUDAGRAPH_CHECK] model={model} case={case}")
        print(f"[CHRONOS_CUDAGRAPH_CHECK] compile_backend={status['compile_backend']}")
        print(f"[CHRONOS_CUDAGRAPH_CHECK] cudagraph_expected={status['cudagraph_expected']}")
        print(f"[CHRONOS_CUDAGRAPH_CHECK] cudagraph_enabled={status['cudagraph_enabled']}")
        print(f"[CHRONOS_CUDAGRAPH_CHECK] fallback_taken={status['fallback_taken']}")
        print(f"[CHRONOS_CUDAGRAPH_CHECK] fallback_reason={status['fallback_reason'] or 'none'}")
        print(
            f"[GRAPH_STATUS] model={model} case={case} full_graph={status['full_graph']} "
            f"cudagraph={status['cudagraph_enabled']} fallback={status['fallback_taken']} "
            f"graph_breaks={status['graph_break_count']} reason={status['fallback_reason'] or 'none'}"
        )
    return status


def build_chronos_compile_config(
    *,
    backend: Any = "inductor",
    enable_cudagraphs: bool = False,
    cudagraph_mode: str = "reduce-overhead",
    fullgraph: bool = False,
    dynamic: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    mode: Optional[str] = None
    options: Optional[Dict[str, Any]] = None

    if enable_cudagraphs:
        if cudagraph_mode == "reduce-overhead":
            mode = "reduce-overhead"
        elif cudagraph_mode == "triton-option":
            options = {"triton.cudagraphs": True}
        elif cudagraph_mode == "both":
            mode = "reduce-overhead"
            options = {"triton.cudagraphs": True}
        else:
            raise ValueError(f"unsupported cudagraph_mode: {cudagraph_mode}")

    kwargs: Dict[str, Any] = {
        "backend": backend,
        "fullgraph": fullgraph,
        "dynamic": dynamic,
    }
    if mode is not None:
        kwargs["mode"] = mode
    if options is not None:
        kwargs["options"] = options

    config = {
        "enable_cudagraphs": bool(enable_cudagraphs),
        "cudagraph_mode": cudagraph_mode,
        "compile_mode": mode,
        "compile_options": options,
        "backend": str(backend),
        "fullgraph": fullgraph,
        "dynamic": dynamic,
    }
    return kwargs, config


def compile_with_chronos_options(
    model_or_fn,
    *,
    backend: Any = "inductor",
    enable_cudagraphs: bool = False,
    cudagraph_mode: str = "reduce-overhead",
    fullgraph: bool = False,
    dynamic: bool = False,
):
    kwargs, config = build_chronos_compile_config(
        backend=backend,
        enable_cudagraphs=enable_cudagraphs,
        cudagraph_mode=cudagraph_mode,
        fullgraph=fullgraph,
        dynamic=dynamic,
    )
    print(
        "[Compile Config] "
        f"enable_cudagraphs={config['enable_cudagraphs']} "
        f"cudagraph_mode={config['cudagraph_mode']} "
        f"mode={config['compile_mode']} "
        f"options={config['compile_options']}"
    )
    return torch.compile(model_or_fn, **kwargs)
