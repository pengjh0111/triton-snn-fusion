"""Deprecated compatibility wrapper for temporal LIF avgpool-linear kernel."""

from kernels.generated_temporal_lif_avgpool_linear_kernel import (  # noqa: F401
    run_fused_temporal_lif_avgpool_linear_kernel,
)

run_fused_temporal_lif_tail_kernel = run_fused_temporal_lif_avgpool_linear_kernel
