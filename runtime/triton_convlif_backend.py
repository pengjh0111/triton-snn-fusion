from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch


@dataclass
class TritonConvLIFResult:
    spikes: torch.Tensor
    v_next: torch.Tensor
    used_triton: bool
    kernel_key: str = "unsupported"
    fallback_reason: str = ""
    kernel_temporal_config: Optional[Dict] = None
    kernel_diagnostics: Optional[Dict] = None


def _as_pair(value) -> Tuple[int, int]:
    if isinstance(value, int):
        return value, value
    return int(value[0]), int(value[1])


def _conv2d_output_shape(x, weight, stride, padding, dilation) -> Tuple[int, int, int, int]:
    batch = x.shape[0]
    out_channels = weight.shape[0]
    height = x.shape[2]
    width = x.shape[3]
    stride_h, stride_w = _as_pair(stride)
    pad_h, pad_w = _as_pair(padding)
    dil_h, dil_w = _as_pair(dilation)
    kernel_h = weight.shape[2]
    kernel_w = weight.shape[3]
    out_h = (height + 2 * pad_h - dil_h * (kernel_h - 1) - 1) // stride_h + 1
    out_w = (width + 2 * pad_w - dil_w * (kernel_w - 1) - 1) // stride_w + 1
    return batch, out_channels, out_h, out_w


def _unsupported(reasons: List[str], strict: bool, verbose: bool) -> None:
    if reasons and verbose:
        print(f"[TRITON][FALLBACK] {'; '.join(reasons)}")
    if reasons and strict:
        raise RuntimeError("[TRITON][STRICT] " + "; ".join(reasons))


def _is_fast_legacy_shape(weight, stride, padding, dilation) -> bool:
    return classify_conv_lif_config(weight, stride, padding, dilation, 1) == "k3_s1_p1"


def classify_conv_lif_config(weight, stride, padding, dilation, groups) -> str:
    if weight is None or not isinstance(weight, torch.Tensor) or weight.dim() != 4:
        return "unsupported"
    if tuple(_as_pair(dilation)) != (1, 1):
        return "unsupported"

    stride_pair = tuple(_as_pair(stride))
    padding_pair = tuple(_as_pair(padding))
    kernel_pair = tuple(weight.shape[-2:])
    groups = int(groups)
    out_channels = int(weight.shape[0])
    weight_in_channels = int(weight.shape[1])
    if groups == out_channels and weight_in_channels == 1:
        if kernel_pair == (3, 3) and stride_pair == (1, 1) and padding_pair == (1, 1):
            return "depthwise_k3_s1_p1"
        if kernel_pair == (3, 3) and stride_pair == (2, 2) and padding_pair == (1, 1):
            return "depthwise_k3_s2_p1"
        return "unsupported"
    if groups != 1:
        return "unsupported"
    if kernel_pair == (1, 1) and stride_pair == (1, 1) and padding_pair == (0, 0):
        return "k1_s1_p0"
    if kernel_pair == (3, 3) and stride_pair == (1, 1) and padding_pair == (1, 1):
        return "k3_s1_p1"
    if kernel_pair == (3, 3) and stride_pair == (2, 2) and padding_pair == (1, 1):
        return "k3_s2_p1"
    if kernel_pair == (5, 5) and stride_pair == (1, 1) and padding_pair == (2, 2):
        return "k5_s1_p2"
    if kernel_pair == (7, 7) and stride_pair == (2, 2) and padding_pair == (3, 3):
        return "k7_s2_p3"
    if kernel_pair == (11, 11) and stride_pair == (4, 4) and padding_pair == (2, 2):
        return "k11_s4_p2"
    return "unsupported"


def classify_conv_lif_runtime_kind(weight, stride, padding, dilation, groups) -> str:
    kernel_key = classify_conv_lif_config(weight, stride, padding, dilation, groups)
    if kernel_key.startswith("depthwise_"):
        return "depthwise"
    if kernel_key == "k1_s1_p0":
        return "pointwise"
    if kernel_key != "unsupported":
        return "regular"
    return "unsupported"


SUPPORTED_CONV_LIF_CONFIGS = {
    (1, 1): {
        ((1, 1), (0, 0)): "k1_s1_p0",
    },
    (3, 3): {
        ((1, 1), (1, 1)): "k3_s1_p1",
        ((2, 2), (1, 1)): "k3_s2_p1",
        ((1, 1), (1, 1), "depthwise"): "depthwise_k3_s1_p1",
        ((2, 2), (1, 1), "depthwise"): "depthwise_k3_s2_p1",
    },
    (5, 5): {
        ((1, 1), (2, 2)): "k5_s1_p2",
    },
    (7, 7): {
        ((2, 2), (3, 3)): "k7_s2_p3",
    },
    (11, 11): {
        ((4, 4), (2, 2)): "k11_s4_p2",
    },
}


def check_triton_support(
    x,
    weight,
    bias,
    v_prev,
    stride,
    padding,
    dilation,
    groups,
    v_threshold,
    v_reset,
    tau,
    detach_reset,
) -> List[str]:
    reasons: List[str] = []
    if not x.is_cuda:
        reasons.append("unsupported_device: x is not CUDA")
    if bias is None:
        reasons.append("missing_or_unsupported_bias: bias must be a Tensor")
    supported_dtypes = (torch.float32, torch.float16)
    if x.dtype not in supported_dtypes:
        reasons.append(f"unsupported_dtype: x dtype must be float32 or float16, got {x.dtype}")
    if weight.dtype != x.dtype or (bias is not None and bias.dtype != x.dtype):
        reasons.append(
            f"unsupported_dtype: x/weight/bias dtypes must match, got "
            f"x={x.dtype}, weight={weight.dtype}, bias={getattr(bias, 'dtype', None)}"
        )
    if v_prev is not None and isinstance(v_prev, torch.Tensor) and v_prev.dtype != x.dtype:
        reasons.append(f"unsupported_dtype: v_prev dtype must match x dtype, got v_prev={v_prev.dtype}, x={x.dtype}")
    if x.dim() != 4:
        reasons.append(f"unsupported_rank: x.dim must be 4, got {x.dim()}")
    if weight.dim() != 4:
        reasons.append(f"unsupported_rank: weight.dim must be 4, got {weight.dim()}")
    groups = int(groups)
    in_channels = int(x.shape[1]) if x.dim() == 4 else None
    out_channels = int(weight.shape[0]) if weight.dim() == 4 else None
    weight_in_channels = int(weight.shape[1]) if weight.dim() == 4 else None
    is_depthwise = (
        in_channels is not None
        and out_channels is not None
        and weight_in_channels == 1
        and groups == in_channels == out_channels
    )
    if groups != 1 and not is_depthwise:
        reasons.append(f"unsupported_groups: groups must be 1 or depthwise, got {groups}")
    if tuple(_as_pair(dilation)) != (1, 1):
        reasons.append(f"unsupported_dilation: dilation must be (1, 1), got {tuple(_as_pair(dilation))}")
    stride_pair = tuple(_as_pair(stride))
    padding_pair = tuple(_as_pair(padding))
    kernel_pair = tuple(weight.shape[-2:]) if weight.dim() == 4 else None
    if kernel_pair is not None:
        supported_for_kernel = SUPPORTED_CONV_LIF_CONFIGS.get(kernel_pair)
        if supported_for_kernel is None:
            reasons.append(
                "unsupported_kernel: kernel must be one of "
                f"{tuple(SUPPORTED_CONV_LIF_CONFIGS)}, got {kernel_pair}"
            )
        elif (
            (stride_pair, padding_pair) not in supported_for_kernel
            and not (is_depthwise and (stride_pair, padding_pair, "depthwise") in supported_for_kernel)
        ):
            supported_strides = sorted({config[0] for config in supported_for_kernel})
            supported_paddings_for_stride = sorted(
                {config[1] for config in supported_for_kernel if config[0] == stride_pair}
            )
            if not supported_paddings_for_stride:
                reasons.append(
                    f"unsupported_stride: {kernel_pair[0]}x{kernel_pair[1]} kernel supports "
                    f"stride {supported_strides}, got {stride_pair}"
                )
            else:
                reasons.append(
                    f"unsupported_padding: {kernel_pair[0]}x{kernel_pair[1]} kernel with stride {stride_pair} "
                    f"supports padding {supported_paddings_for_stride}, got {padding_pair}"
                )
    if bool(detach_reset):
        reasons.append("unsupported_detach_reset: detach_reset=True is not supported by existing Triton kernel")
    if abs(float(v_threshold) - 1.0) > 1e-6:
        reasons.append("unsupported_threshold: existing kernel uses compile-time V_THRESHOLD=1.0")
    if abs(float(v_reset) - 0.0) > 1e-6:
        reasons.append("unsupported_reset: existing kernel uses compile-time V_RESET=0.0")
    if abs(float(tau) - 2.0) > 1e-6:
        reasons.append("unsupported_tau: existing kernel uses compile-time tau=2.0")

    if not reasons:
        out_shape = _conv2d_output_shape(x, weight, stride, padding, dilation)
        if v_prev.dim() == 0:
            pass
        elif tuple(v_prev.shape) != tuple(out_shape):
            reasons.append(f"v_prev shape {tuple(v_prev.shape)} does not match conv output shape {out_shape}")

    # Note: kernel reads initial membrane from v_ptr. Do not reject non-zero v_prev here.

    return reasons


def _run_triton_temporal_framework(
    kernel_key: str,
    x_seq: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    use_autotune: bool = True,
    v_init: torch.Tensor = None,
    spikes_out: torch.Tensor = None,
    v_out: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if use_autotune:
        from kernels.benchmark_conv_lif_temporal_general import run_fused_temporal_general_autotuned

        return run_fused_temporal_general_autotuned(
            x_seq,
            weight.contiguous(),
            bias.contiguous(),
            kernel_key=kernel_key,
            v_init=v_init,
            spikes_out=spikes_out,
            membrane_out=v_out,
        )

    from kernels.benchmark_conv_lif_temporal_general import run_fused_temporal_general

    return run_fused_temporal_general(
        x_seq,
        weight.contiguous(),
        bias.contiguous(),
        temporal_batch_size=1,
        reuse_groups=1,
        kernel_key=kernel_key,
        v_init=v_init,
        spikes_out=spikes_out,
        membrane_out=v_out,
    )


def _run_triton_temporal_by_key(
    kernel_key: str,
    x_seq: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    use_autotune: bool = True,
    v_init: torch.Tensor = None,
    spikes_out: torch.Tensor = None,
    v_out: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if kernel_key in (
        "k1_s1_p0",
        "k3_s1_p1",
        "k3_s2_p1",
        "k5_s1_p2",
        "k7_s2_p3",
        "k11_s4_p2",
        "depthwise_k3_s1_p1",
        "depthwise_k3_s2_p1",
    ):
        return _run_triton_temporal_framework(
            kernel_key,
            x_seq,
            weight,
            bias,
            use_autotune=use_autotune,
            v_init=v_init,
            spikes_out=spikes_out,
            v_out=v_out,
        )
    raise RuntimeError(f"unsupported dispatch key: {kernel_key}")


def _validate_temporal_conv_kind(kind: str, kernel_key: str) -> None:
    if kind == "regular" and kernel_key.startswith("depthwise_"):
        raise RuntimeError(f"regular conv op cannot dispatch depthwise kernel {kernel_key}")
    if kind == "regular" and kernel_key == "k1_s1_p0":
        raise RuntimeError("regular conv op cannot dispatch pointwise kernel k1_s1_p0")
    if kind == "pointwise" and kernel_key != "k1_s1_p0":
        raise RuntimeError(f"pointwise conv op requires k1_s1_p0 kernel, got {kernel_key}")
    if kind == "depthwise" and not kernel_key.startswith("depthwise_"):
        raise RuntimeError(f"depthwise conv op requires depthwise kernel, got {kernel_key}")


def _debug_dispatch(verbose: bool, op_name: str, backend: str, xs_or_seq, weight, stride, padding, groups) -> None:
    if not verbose:
        return
    if isinstance(xs_or_seq, torch.Tensor):
        x_shape = tuple(xs_or_seq.shape)
    elif isinstance(xs_or_seq, (list, tuple)) and len(xs_or_seq) > 0 and isinstance(xs_or_seq[0], torch.Tensor):
        x_shape = (len(xs_or_seq),) + tuple(xs_or_seq[0].shape)
    else:
        x_shape = None
    w_shape = tuple(weight.shape) if isinstance(weight, torch.Tensor) else None
    print(
        f"[CHRONOS_DISPATCH] op={op_name} backend={backend} "
        f"x_shape={x_shape} w_shape={w_shape} stride={tuple(_as_pair(stride))} "
        f"padding={tuple(_as_pair(padding))} groups={int(groups)}"
    )


def _get_kernel_temporal_config(kernel_key: str, residual_add: bool = False):
    try:
        from kernels.benchmark_conv_lif_temporal_general import get_autotune_best_config

        return get_autotune_best_config(kernel_key, residual_add=residual_add)
    except Exception:
        return None


def _get_kernel_diagnostics(dtype):
    try:
        from kernels.benchmark_conv_lif_temporal_general import kernel_dtype_diagnostics

        return kernel_dtype_diagnostics(dtype)
    except Exception:
        return None


def run_triton_fused_conv_lif_state(
    x,
    weight,
    bias,
    v_prev,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    groups: int,
    v_threshold: float,
    v_reset: float,
    tau: float,
    detach_reset: bool,
    strict: bool = False,
    verbose: bool = False,
    use_autotune: bool = True,
    compute_dtype: str = None,
) -> TritonConvLIFResult:
    reasons = check_triton_support(
        x,
        weight,
        bias,
        v_prev,
        stride,
        padding,
        dilation,
        groups,
        v_threshold,
        v_reset,
        tau,
        detach_reset,
    )
    if reasons:
        _unsupported(reasons, strict=strict, verbose=verbose)
        raise RuntimeError("; ".join(reasons))
    expected_compute_dtype = "float16" if x.dtype == torch.float16 else "float32"
    if compute_dtype is not None and compute_dtype != expected_compute_dtype:
        raise RuntimeError(
            f"compute_dtype={compute_dtype} does not match input dtype path {expected_compute_dtype}"
        )

    try:
        x_seq = x.unsqueeze(0).contiguous()
        kernel_key = classify_conv_lif_config(weight, stride, padding, dilation, groups)
        if kernel_key == "unsupported":
            raise RuntimeError("unsupported dispatch key after support check")
        spikes, membrane = _run_triton_temporal_by_key(kernel_key, x_seq, weight, bias, use_autotune=use_autotune, v_init=v_prev)
        return TritonConvLIFResult(
            spikes[0],
            membrane,
            True,
            kernel_key=kernel_key,
            kernel_temporal_config=_get_kernel_temporal_config(kernel_key),
            kernel_diagnostics=_get_kernel_diagnostics(x.dtype),
        )
    except Exception as exc:
        reason = f"Triton kernel call failed: {exc}"
        if verbose:
            print(f"[TRITON][FALLBACK][single] {reason}")
        raise


def run_triton_fused_temporal_conv_lif_state(
    xs,
    weight,
    bias,
    v_init,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    groups: int,
    v_threshold: float,
    v_reset: float,
    tau: float,
    detach_reset: bool,
    strict: bool = False,
    verbose: bool = False,
    use_autotune: bool = True,
    compute_dtype: str = None,
) -> TritonConvLIFResult:
    return _run_triton_fused_temporal_conv_lif_state_kind(
        "regular",
        "fused_temporal_conv_lif_state",
        xs,
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
        strict=strict,
        verbose=verbose,
        use_autotune=use_autotune,
        compute_dtype=compute_dtype,
    )


def run_triton_fused_temporal_pointwise_conv_lif_state(
    xs,
    weight,
    bias,
    v_init,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    groups: int,
    v_threshold: float,
    v_reset: float,
    tau: float,
    detach_reset: bool,
    strict: bool = False,
    verbose: bool = False,
    use_autotune: bool = True,
    compute_dtype: str = None,
) -> TritonConvLIFResult:
    return _run_triton_fused_temporal_conv_lif_state_kind(
        "pointwise",
        "fused_temporal_pointwise_conv_lif_state",
        xs,
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
        strict=strict,
        verbose=verbose,
        use_autotune=use_autotune,
        compute_dtype=compute_dtype,
    )


def run_triton_fused_temporal_depthwise_conv_lif_state(
    xs,
    weight,
    bias,
    v_init,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    groups: int,
    v_threshold: float,
    v_reset: float,
    tau: float,
    detach_reset: bool,
    strict: bool = False,
    verbose: bool = False,
    use_autotune: bool = True,
    compute_dtype: str = None,
) -> TritonConvLIFResult:
    return _run_triton_fused_temporal_conv_lif_state_kind(
        "depthwise",
        "fused_temporal_depthwise_conv_lif_state",
        xs,
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
        strict=strict,
        verbose=verbose,
        use_autotune=use_autotune,
        compute_dtype=compute_dtype,
    )


def _run_triton_fused_temporal_conv_lif_state_kind(
    kind: str,
    op_name: str,
    xs,
    weight,
    bias,
    v_init,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    groups: int,
    v_threshold: float,
    v_reset: float,
    tau: float,
    detach_reset: bool,
    strict: bool = False,
    verbose: bool = False,
    use_autotune: bool = True,
    compute_dtype: str = None,
) -> TritonConvLIFResult:
    reasons: List[str] = []
    if not isinstance(xs, (list, tuple)) or len(xs) == 0:
        reasons.append("xs must be a non-empty list/tuple of tensors")
    else:
        first = xs[0]
        for index, x in enumerate(xs):
            if not isinstance(x, torch.Tensor):
                reasons.append(f"xs[{index}] is not a tensor")
                continue
            if tuple(x.shape) != tuple(first.shape):
                reasons.append(f"xs[{index}] shape {tuple(x.shape)} differs from first shape {tuple(first.shape)}")
            if x.device != first.device or x.dtype != first.dtype:
                reasons.append(f"xs[{index}] device/dtype differs from first tensor")

    if reasons:
        _unsupported(reasons, strict=strict, verbose=verbose)
        raise RuntimeError("; ".join(reasons))

    first = xs[0]
    reasons.extend(
        check_triton_support(
            first,
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
        )
    )
    if reasons:
        _unsupported(reasons, strict=strict, verbose=verbose)
        raise RuntimeError("; ".join(reasons))
    first = xs[0]
    expected_compute_dtype = "float16" if first.dtype == torch.float16 else "float32"
    if compute_dtype is not None and compute_dtype != expected_compute_dtype:
        raise RuntimeError(
            f"compute_dtype={compute_dtype} does not match input dtype path {expected_compute_dtype}"
        )

    try:
        x_seq = torch.stack(tuple(xs), dim=0).contiguous()
        kernel_key = classify_conv_lif_config(weight, stride, padding, dilation, groups)
        if kernel_key == "unsupported":
            raise RuntimeError("unsupported dispatch key after support check")
        _validate_temporal_conv_kind(kind, kernel_key)
        _debug_dispatch(verbose, op_name, kernel_key, xs, weight, stride, padding, groups)
        spikes, membrane = _run_triton_temporal_by_key(kernel_key, x_seq, weight, bias, use_autotune=use_autotune, v_init=v_init)
        return TritonConvLIFResult(
            spikes,
            membrane,
            True,
            kernel_key=kernel_key,
            kernel_temporal_config=_get_kernel_temporal_config(kernel_key),
            kernel_diagnostics=_get_kernel_diagnostics(first.dtype),
        )
    except Exception as exc:
        reason = f"temporal Triton kernel call failed: {exc}"
        if verbose:
            print(f"[TRITON][FALLBACK][temporal] {reason}")
        raise


def run_triton_fused_temporal_conv_lif_state_packed_out(
    x_seq,
    weight,
    bias,
    v_init,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    groups: int,
    v_threshold: float,
    v_reset: float,
    tau: float,
    detach_reset: bool,
    spike_out,
    v_out,
    strict: bool = False,
    verbose: bool = False,
    use_autotune: bool = True,
    compute_dtype: str = None,
) -> TritonConvLIFResult:
    reasons: List[str] = []
    if not isinstance(x_seq, torch.Tensor):
        reasons.append("x_seq must be a tensor")
    elif x_seq.dim() != 5:
        reasons.append(f"x_seq must have rank 5 [T,N,C,H,W], got {x_seq.dim()}")
    elif not x_seq.is_contiguous():
        reasons.append("x_seq must be contiguous; create the packed stack in FX before calling the out op")
    if not isinstance(spike_out, torch.Tensor) or not isinstance(v_out, torch.Tensor):
        reasons.append("spike_out and v_out must be tensors")
    if reasons:
        _unsupported(reasons, strict=strict, verbose=verbose)
        raise RuntimeError("; ".join(reasons))

    first = x_seq[0]
    reasons.extend(
        check_triton_support(
            first,
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
        )
    )
    if reasons:
        _unsupported(reasons, strict=strict, verbose=verbose)
        raise RuntimeError("; ".join(reasons))
    expected_compute_dtype = "float16" if x_seq.dtype == torch.float16 else "float32"
    if compute_dtype is not None and compute_dtype != expected_compute_dtype:
        raise RuntimeError(
            f"compute_dtype={compute_dtype} does not match input dtype path {expected_compute_dtype}"
        )

    try:
        kernel_key = classify_conv_lif_config(weight, stride, padding, dilation, groups)
        if kernel_key == "unsupported":
            raise RuntimeError("unsupported dispatch key after support check")
        spikes, membrane = _run_triton_temporal_by_key(
            kernel_key,
            x_seq,
            weight,
            bias,
            use_autotune=use_autotune,
            v_init=v_init,
            spikes_out=spike_out,
            v_out=v_out,
        )
        return TritonConvLIFResult(
            spikes,
            membrane,
            True,
            kernel_key=kernel_key,
            kernel_temporal_config=_get_kernel_temporal_config(kernel_key),
            kernel_diagnostics=_get_kernel_diagnostics(x_seq.dtype),
        )
    except Exception as exc:
        reason = f"temporal Triton out kernel call failed: {exc}"
        if verbose:
            print(f"[TRITON][FALLBACK][temporal_out] {reason}")
        raise


def run_triton_fused_temporal_conv_add_lif_state(
    xs,
    residuals,
    weight,
    bias,
    v_init,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    groups: int,
    v_threshold: float,
    v_reset: float,
    tau: float,
    detach_reset: bool,
    strict: bool = False,
    verbose: bool = False,
    use_autotune: bool = True,
    compute_dtype: str = None,
) -> TritonConvLIFResult:
    reasons: List[str] = []
    if not isinstance(xs, (list, tuple)) or len(xs) == 0:
        reasons.append("xs must be a non-empty list/tuple of tensors")
    if not isinstance(residuals, (list, tuple)) or len(residuals) == 0:
        reasons.append("residuals must be a non-empty list/tuple of tensors")
    if not reasons and len(xs) != len(residuals):
        reasons.append(f"xs and residuals length mismatch: {len(xs)} vs {len(residuals)}")

    if not reasons:
        first = xs[0]
        for index, x in enumerate(xs):
            if not isinstance(x, torch.Tensor):
                reasons.append(f"xs[{index}] is not a tensor")
                continue
            if tuple(x.shape) != tuple(first.shape):
                reasons.append(f"xs[{index}] shape {tuple(x.shape)} differs from first shape {tuple(first.shape)}")
            if x.device != first.device or x.dtype != first.dtype:
                reasons.append(f"xs[{index}] device/dtype differs from first tensor")

        out_shape = _conv2d_output_shape(first, weight, stride, padding, dilation)
        for index, residual in enumerate(residuals):
            if not isinstance(residual, torch.Tensor):
                reasons.append(f"residuals[{index}] is not a tensor")
                continue
            if tuple(residual.shape) != tuple(out_shape):
                reasons.append(
                    f"residuals[{index}] shape {tuple(residual.shape)} does not match conv output shape {out_shape}"
                )
            if residual.device != first.device or residual.dtype != first.dtype:
                reasons.append(f"residuals[{index}] device/dtype differs from first tensor")

    if reasons:
        _unsupported(reasons, strict=strict, verbose=verbose)
        raise RuntimeError("; ".join(reasons))

    first = xs[0]
    reasons.extend(
        check_triton_support(
            first,
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
        )
    )
    if reasons:
        _unsupported(reasons, strict=strict, verbose=verbose)
        raise RuntimeError("; ".join(reasons))

    expected_compute_dtype = "float16" if first.dtype == torch.float16 else "float32"
    if compute_dtype is not None and compute_dtype != expected_compute_dtype:
        raise RuntimeError(
            f"compute_dtype={compute_dtype} does not match input dtype path {expected_compute_dtype}"
        )

    try:
        from kernels.benchmark_conv_lif_temporal_general import (
            run_fused_temporal_general_residual,
            run_fused_temporal_general_residual_autotuned,
        )

        x_seq = torch.stack(tuple(xs), dim=0).contiguous()
        residual_seq = torch.stack(tuple(residuals), dim=0).contiguous()
        kernel_key = classify_conv_lif_config(weight, stride, padding, dilation, groups)
        if kernel_key == "unsupported":
            raise RuntimeError("unsupported dispatch key after support check")
        if use_autotune:
            spikes, membrane = run_fused_temporal_general_residual_autotuned(
                x_seq,
                residual_seq,
                weight.contiguous(),
                bias.contiguous(),
                kernel_key=kernel_key,
                v_init=v_init,
            )
        else:
            spikes, membrane = run_fused_temporal_general_residual(
                x_seq,
                residual_seq,
                weight.contiguous(),
                bias.contiguous(),
                temporal_batch_size=1,
                reuse_groups=1,
                kernel_key=kernel_key,
                v_init=v_init,
            )
        diagnostics = _get_kernel_diagnostics(first.dtype) or {}
        diagnostics = dict(diagnostics)
        diagnostics["residual_add"] = True
        diagnostics["kernel_kind"] = "temporal_residual"
        return TritonConvLIFResult(
            spikes,
            membrane,
            True,
            kernel_key=kernel_key,
            kernel_temporal_config=_get_kernel_temporal_config(kernel_key, residual_add=True),
            kernel_diagnostics=diagnostics,
        )
    except Exception as exc:
        reason = f"temporal residual Triton kernel call failed: {exc}"
        if verbose:
            print(f"[TRITON][FALLBACK][temporal_residual] {reason}")
        raise
