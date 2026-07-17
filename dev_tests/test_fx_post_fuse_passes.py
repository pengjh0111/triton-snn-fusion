"""Tests for the four independent FX post-fuse optimization passes in
compiler/passes/ (Pass A: SDPA, Pass B: stack CSE, Pass C: v_init cleanup,
Pass D: classifier head batching).

Each pass gets:
  - a graph-structure assertion (node-count/pattern checks), and
  - a numerical validation test at the level specified in the task spec
    (bit-exact for B/C, tolerance for A/D), comparing the real
    ChronosSpikeTransformer's output with the pass off vs on.

Pass B additionally gets a synthetic-graph unit test, because the real model
at the small size used here only ever produces ONE occurrence of its
duplicate-input stack pattern (one stack node per whole-model forward, not
multiple), so there's nothing to merge in an end-to-end capture at this
scale; the synthetic graph exercises the actual merge logic directly.

Run: python dev_tests/test_fx_post_fuse_passes.py [--quick]
"""

import argparse
import copy
import operator
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

import runtime.snn_custom_ops as snn_custom_ops
from compiler.passes.pass_sdpa import apply_sdpa_pass
from compiler.passes.pass_stack_cse import apply_stack_cse_pass
from compiler.passes.pass_v_init_cleanup import apply_v_init_cleanup_pass
from compiler.passes.pass_classifier_batch import apply_classifier_batch_pass
from compiler.passes.registry import apply_post_fuse_passes


FAILURES = []

# Explicit "all four passes off" baseline. Passing {} to _capture_gm_only
# would leave the env vars unset, which (since compiler/passes/registry.py
# defaults unset vars to enabled) now means "all on", not "all off" -- so
# every "_off" baseline in this file must set all four explicitly to False.
ALL_PASSES_OFF = {
    "CHRONOS_PASS_SDPA": False,
    "CHRONOS_PASS_STACK_CSE": False,
    "CHRONOS_PASS_VINIT_CLEANUP": False,
    "CHRONOS_PASS_CLASSIFIER_BATCH": False,
}


def check(name: str, condition: bool, detail: str = ""):
    status = "OK  " if condition else "FAIL"
    print(f"{status} {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(f"{name}: {detail}")


# --------------------------------------------------------------------------
# Real-model graph capture helper
# --------------------------------------------------------------------------

def _build_args(**overrides) -> argparse.Namespace:
    from benchmarks.benchmark_chronos_runtime import parse_args

    argv = [
        "benchmark_chronos_runtime.py",
        "--models", "spiketransformer",
        "--T", "4",
        "--batch-size", "2",
        "--sequence-length", "8",
        "--transformer-depth", "2",
        "--transformer-dim", "32",
        "--transformer-heads", "4",
        "--transformer-input-dim", "48",
        "--transformer-num-classes", "10",
        "--device", "cuda",
        "--dtype", "fp16",
        "--fused-op-backend", "triton",
        "--rewrite-backend-mode", "eager",
        "--enable-temporal-rewrite",
        "--enable-temporal-schedule",
        "--temporal-fuse-window", "4",
        "--temporal-schedule-window", "4",
        "--max-patterns", "1000000",
        "--warmup", "1",
        "--repeat", "1",
    ]
    old_argv = sys.argv
    try:
        sys.argv = argv
        args = parse_args()
    finally:
        sys.argv = old_argv
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _capture_gm_only(pass_env: dict, tmp_dir: Path, seed: int = 2026):
    """Build ChronosSpikeTransformer, run it once through torch.compile with
    make_rewrite_backend (eager mode), applying only the passes named True in
    pass_env. Returns (output_tensor, captured_gm) -- the GraphModule is
    recovered from the eager-mode backend's returned callable (gm.forward),
    via its __self__."""
    import os

    from benchmarks.validate_chronos_baselines import (
        RewriteCounters,
        make_resnet_layer,
        make_rewrite_backend,
        make_model_input,
        SingleStepModeLoopWrapper,
    )

    old_env = {}
    all_pass_vars = ("CHRONOS_PASS_SDPA", "CHRONOS_PASS_STACK_CSE", "CHRONOS_PASS_VINIT_CLEANUP", "CHRONOS_PASS_CLASSIFIER_BATCH")
    for var in all_pass_vars:
        old_env[var] = os.environ.get(var)
        os.environ.pop(var, None)
    for var, enabled in pass_env.items():
        os.environ[var] = "1" if enabled else "0"

    try:
        torch.manual_seed(seed)
        args = _build_args()
        dtype = torch.float16

        base_layer_s = make_resnet_layer(
            "spiketransformer",
            allow_resnet32_fallback=True,
            step_mode="s",
            model_channels=64,
            lif_impl=args.lif_impl,
            sequence_length=args.sequence_length,
            transformer_depth=args.transformer_depth,
            transformer_dim=args.transformer_dim,
            transformer_heads=args.transformer_heads,
            transformer_input_dim=args.transformer_input_dim,
            transformer_vocab_size=args.transformer_vocab_size,
            transformer_num_classes=args.transformer_num_classes,
        ).to(device=args.device, dtype=dtype).eval()

        model = SingleStepModeLoopWrapper(copy.deepcopy(base_layer_s), args.T).to(device=args.device, dtype=dtype).eval()
        x = make_model_input("spiketransformer", args, dtype)

        counters = RewriteCounters()
        captured = {}

        def capturing_backend(gm, example_inputs, **compile_kwargs):
            fwd = make_rewrite_backend(args, tmp_dir, counters)(gm, example_inputs, **compile_kwargs)
            captured["gm"] = fwd.__self__ if hasattr(fwd, "__self__") else None
            return fwd

        torch._dynamo.reset()
        compiled = torch.compile(model, backend=capturing_backend, fullgraph=False, dynamic=False)

        snn_custom_ops.configure_fused_op("triton", strict_triton=True, verbose=False)
        with torch.no_grad():
            out = compiled(x)

        return out.detach().clone(), captured.get("gm")
    finally:
        for var, value in old_env.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value


# --------------------------------------------------------------------------
# Pass B: synthetic graph unit test (bit-exact by construction)
# --------------------------------------------------------------------------

def test_pass_b_synthetic():
    print("\n=== Pass B (stack CSE): synthetic graph test ===")

    class M(torch.nn.Module):
        def forward(self, x):
            a = torch.stack([x, x, x, x], 0)
            b = torch.stack([x, x, x, x], 0)  # duplicate of `a`
            c = torch.stack([x, x], 0)  # different length -- must NOT merge with a/b
            return a.sum() + b.sum() + c.sum()

    gm = torch.fx.symbolic_trace(M())
    stack_nodes_before = sum(1 for n in gm.graph.nodes if n.op == "call_function" and n.target is torch.stack)
    check("pass_b_synthetic: 3 stack nodes before", stack_nodes_before == 3, f"got {stack_nodes_before}")

    stats = apply_stack_cse_pass(gm)
    check("pass_b_synthetic: 1 group found", stats.groups == 1, f"got {stats.groups}")
    check("pass_b_synthetic: 1 node merged", stats.merged == 1, f"got {stats.merged}")

    stack_nodes_after = sum(1 for n in gm.graph.nodes if n.op == "call_function" and n.target is torch.stack)
    check("pass_b_synthetic: 2 stack nodes after (4-elem group collapsed, 2-elem group untouched)", stack_nodes_after == 2, f"got {stack_nodes_after}")

    x = torch.randn(3)
    ref = M()(x)
    out = gm(x)
    check("pass_b_synthetic: bit-exact output", torch.equal(ref, out), f"ref={ref.item()} out={out.item()}")

    # Idempotency: running again on the already-merged graph must be a no-op.
    stats2 = apply_stack_cse_pass(gm)
    check("pass_b_synthetic: idempotent (no further merges)", stats2.merged == 0, f"got {stats2.merged}")


# --------------------------------------------------------------------------
# Real-model tests
# --------------------------------------------------------------------------

def _count_targets(gm, predicate):
    return sum(1 for n in gm.graph.nodes if predicate(n))


def test_pass_a_real_model(tmp_root: Path):
    print("\n=== Pass A (SDPA): real-model graph assertion + tolerance numerical test ===")
    out_off, gm_off = _capture_gm_only(ALL_PASSES_OFF, tmp_root / "a_off")
    out_on, gm_on = _capture_gm_only({**ALL_PASSES_OFF, "CHRONOS_PASS_SDPA": True}, tmp_root / "a_on")

    if gm_on is None:
        check("pass_a: captured graph module", False, "gm capture failed")
        return

    softmax_count = _count_targets(gm_on, lambda n: n.op == "call_function" and n.target in (torch.softmax, torch.nn.functional.softmax))
    check("pass_a: softmax node count == 0 after pass", softmax_count == 0, f"got {softmax_count}")

    from compiler.passes.pass_sdpa import _sdpa_capture_safe

    sdpa_count = _count_targets(gm_on, lambda n: n.op == "call_function" and n.target is _sdpa_capture_safe)
    check("pass_a: at least one scaled_dot_product_attention node introduced", sdpa_count > 0, f"got {sdpa_count}")

    allclose = torch.allclose(out_off.float(), out_on.float(), rtol=1e-2, atol=1e-3)
    argmax_match = torch.equal(out_off.argmax(dim=-1), out_on.argmax(dim=-1))
    max_diff = (out_off.float() - out_on.float()).abs().max().item()
    check("pass_a: logits within tolerance (rtol=1e-2, atol=1e-3)", allclose, f"max_abs_diff={max_diff:.4e}")
    check("pass_a: argmax classification unchanged", argmax_match)

    # Idempotency on the already-rewritten graph.
    stats_again = apply_sdpa_pass(gm_on)
    check("pass_a: idempotent (no further matches on already-rewritten graph)", stats_again.matched == 0, f"got {stats_again.matched}")


def test_pass_c_real_model(tmp_root: Path):
    print("\n=== Pass C (v_init cleanup): real-model graph assertion + bit-exact numerical test ===")
    out_off, gm_off = _capture_gm_only(ALL_PASSES_OFF, tmp_root / "c_off")
    out_on, gm_on = _capture_gm_only({**ALL_PASSES_OFF, "CHRONOS_PASS_VINIT_CLEANUP": True}, tmp_root / "c_on")

    if gm_on is None:
        check("pass_c: captured graph module", False, "gm capture failed")
        return

    add_lif_target = torch.ops.snn_custom.fused_temporal_add_lif_state.default
    bad_pattern_count = 0
    orphan_add_count = 0
    for node in gm_on.graph.nodes:
        if node.op == "call_function" and node.target is add_lif_target:
            v_init = node.args[2]
            if isinstance(v_init, torch.fx.Node) and v_init.op == "call_function" and v_init.target is torch.zeros_like:
                bad_pattern_count += 1
        if node.op == "call_function" and node.target is operator.add and len(node.users) == 0:
            orphan_add_count += 1

    check("pass_c: zeros_like->v_init pattern count == 0 for fused_temporal_add_lif_state", bad_pattern_count == 0, f"got {bad_pattern_count}")
    check("pass_c: no orphaned add nodes left in graph", orphan_add_count == 0, f"got {orphan_add_count}")

    exact = torch.equal(out_off, out_on)
    max_diff = (out_off.float() - out_on.float()).abs().max().item()
    check("pass_c: bit-exact output", exact, f"max_abs_diff={max_diff:.4e}")

    stats_again = apply_v_init_cleanup_pass(gm_on)
    check("pass_c: idempotent", stats_again.replaced == 0, f"got {stats_again.replaced}")


def test_pass_d_real_model(tmp_root: Path):
    print("\n=== Pass D (classifier batching): real-model graph assertion + tolerance numerical test ===")
    out_off, gm_off = _capture_gm_only(ALL_PASSES_OFF, tmp_root / "d_off")
    out_on, gm_on = _capture_gm_only({**ALL_PASSES_OFF, "CHRONOS_PASS_CLASSIFIER_BATCH": True}, tmp_root / "d_on")

    if gm_on is None:
        check("pass_d: captured graph module", False, "gm capture failed")
        return

    ln_count_off = _count_targets(gm_off, lambda n: n.op == "call_function" and n.target is torch.nn.functional.layer_norm)
    ln_count_on = _count_targets(gm_on, lambda n: n.op == "call_function" and n.target is torch.nn.functional.layer_norm)
    # Off graph: (depth) attention-block layer_norms (2 per block: norm1+norm2, x T timesteps) + T classifier-tail layer_norms.
    # On graph: same attention-block layer_norms untouched + 1 classifier-tail layer_norm.
    check("pass_d: layer_norm count strictly decreases", ln_count_on < ln_count_off, f"off={ln_count_off} on={ln_count_on}")

    sum_count = _count_targets(gm_on, lambda n: n.op == "call_method" and n.target == "sum")
    check("pass_d: at least one .sum(dim=0) introduced", sum_count > 0, f"got {sum_count}")

    allclose = torch.allclose(out_off.float(), out_on.float(), rtol=1e-2, atol=1e-3)
    argmax_match = torch.equal(out_off.argmax(dim=-1), out_on.argmax(dim=-1))
    max_diff = (out_off.float() - out_on.float()).abs().max().item()
    check("pass_d: logits within tolerance (rtol=1e-2, atol=1e-3)", allclose, f"max_abs_diff={max_diff:.4e}")
    check("pass_d: argmax classification unchanged", argmax_match)

    stats_again = apply_classifier_batch_pass(gm_on)
    check("pass_d: idempotent (no further groups found)", stats_again.groups_replaced == 0, f"got {stats_again.groups_replaced}")


def test_pass_b_real_model_smoke(tmp_root: Path):
    print("\n=== Pass B (stack CSE): real-model smoke test (idempotent, no crash, bit-exact if it does match) ===")
    out_off, gm_off = _capture_gm_only(ALL_PASSES_OFF, tmp_root / "b_off")
    out_on, gm_on = _capture_gm_only({**ALL_PASSES_OFF, "CHRONOS_PASS_STACK_CSE": True}, tmp_root / "b_on")
    if gm_on is None:
        check("pass_b_real: captured graph module", False, "gm capture failed")
        return
    exact = torch.equal(out_off, out_on)
    check("pass_b_real: bit-exact output (pass is a no-op or a safe merge at this model size)", exact)
    stats_again = apply_stack_cse_pass(gm_on)
    check("pass_b_real: idempotent", stats_again.merged == 0, f"got {stats_again.merged}")


def test_combined_all_passes(tmp_root: Path):
    print("\n=== Combined: all four passes enabled ===")
    out_off, _ = _capture_gm_only(ALL_PASSES_OFF, tmp_root / "combo_off")
    out_on, gm_on = _capture_gm_only(
        {
            "CHRONOS_PASS_SDPA": True,
            "CHRONOS_PASS_STACK_CSE": True,
            "CHRONOS_PASS_VINIT_CLEANUP": True,
            "CHRONOS_PASS_CLASSIFIER_BATCH": True,
        },
        tmp_root / "combo_on",
    )
    if gm_on is None:
        check("combined: captured graph module", False, "gm capture failed")
        return

    softmax_count = _count_targets(gm_on, lambda n: n.op == "call_function" and n.target in (torch.softmax, torch.nn.functional.softmax))
    check("combined: softmax eliminated", softmax_count == 0, f"got {softmax_count}")

    allclose = torch.allclose(out_off.float(), out_on.float(), rtol=1e-2, atol=1e-3)
    argmax_match = torch.equal(out_off.argmax(dim=-1), out_on.argmax(dim=-1))
    max_diff = (out_off.float() - out_on.float()).abs().max().item()
    check("combined: logits within tolerance", allclose, f"max_abs_diff={max_diff:.4e}")
    check("combined: argmax classification unchanged", argmax_match)

    # Re-running apply_post_fuse_passes on the already-optimized graph must be idempotent and not error.
    results_again = apply_post_fuse_passes(gm_on, sdpa=True, stack_cse=True, v_init_cleanup=True, classifier_batch=True, verbose=False)
    total_changes = sum(
        getattr(s, "replaced", 0) + getattr(s, "merged", 0) + getattr(s, "groups_replaced", 0)
        for s in results_again.values()
    )
    check("combined: idempotent (second run makes no further changes)", total_changes == 0, f"got {total_changes}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tmp-dir", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    tmp_root = Path(args.tmp_dir) if args.tmp_dir else Path("/tmp/chronos_fx_pass_tests")
    tmp_root.mkdir(parents=True, exist_ok=True)

    test_pass_b_synthetic()
    test_pass_b_real_model_smoke(tmp_root)
    test_pass_c_real_model(tmp_root)
    test_pass_a_real_model(tmp_root)
    test_pass_d_real_model(tmp_root)
    test_combined_all_passes(tmp_root)

    print(f"\n{'ALL PASSED' if not FAILURES else 'FAILURES:'}")
    for f in FAILURES:
        print(f"  - {f}")
    if FAILURES:
        sys.exit(1)


if __name__ == "__main__":
    main()
