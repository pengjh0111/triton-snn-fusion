"""Deprecated compatibility wrapper for temporal LIF avgpool-linear backend."""

from runtime.triton_temporal_lif_avgpool_linear_backend import (  # noqa: F401
    TritonTemporalLIFAvgPoolLinearResult,
    TritonTemporalLIFTailResult,
    check_temporal_lif_avgpool_linear_support,
    check_temporal_lif_tail_support,
    run_triton_fused_temporal_lif_avgpool_linear,
    run_triton_fused_temporal_lif_tail,
    strict_temporal_lif_avgpool_linear_enabled,
    strict_temporal_lif_tail_enabled,
)
