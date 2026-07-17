"""Independent, individually-toggleable FX post-fuse optimization passes.

Each pass module in this package operates on an already fuse-rewritten
torch.fx.GraphModule (i.e. after compiler/fx_lif_temporal_rewrite.py and
compiler/fx_temporal_spatial_canonicalize.py have run) and looks for a
narrow, local graph pattern left behind by that rewrite. Passes are wired
into benchmarks/validate_chronos_baselines.py's make_rewrite_backend via
compiler/passes/registry.py, right after the existing fuse-rewrite pipeline
finishes and before the graph is handed to whichever backend
(eager/standalone/inductor) consumes it. See registry.py for the toggle
env vars and execution order.
"""
