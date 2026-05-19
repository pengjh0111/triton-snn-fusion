from typing import Any, Dict, Optional, Tuple

import torch


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
