"""Registration and toggling for the post-fuse optimization passes.

Env var toggles (checked at call time, so tests can flip them per-case):
    KAIROS_PASS_SDPA=1               Pass A: attention -> SDPA
    KAIROS_PASS_STACK_CSE=1          Pass B: duplicate-input stack CSE
    KAIROS_PASS_VINIT_CLEANUP=1      Pass C: zeros_like(add) -> new_zeros(())
    KAIROS_PASS_CLASSIFIER_BATCH=1   Pass D: classifier head batching

All four default ON (unset env var == enabled). Set the var to "0"/"false"/
"no"/"off" (case-insensitive) to explicitly disable one; any other value,
including unset, is treated as enabled.

Execution order: Pass B and C (graph simplification: CSE, dead-node removal)
run before Pass A and D (structural replacement) -- purely a convention
matching the task spec, not a correctness dependency; each pass only looks
for its own pattern and leaves everything else alone or skips silently.

apply_post_fuse_passes(gm) runs whichever of the four are enabled, in that
B, C, A, D order, and returns a dict of each enabled pass's stats object
(keyed by pass name) for logging/testing.
"""

import os
from typing import Dict, Optional

import torch

from compiler.passes.pass_sdpa import apply_sdpa_pass
from compiler.passes.pass_stack_cse import apply_stack_cse_pass
from compiler.passes.pass_v_init_cleanup import apply_v_init_cleanup_pass
from compiler.passes.pass_classifier_batch import apply_classifier_batch_pass

PASS_ENV_VARS = {
    "sdpa": "KAIROS_PASS_SDPA",
    "stack_cse": "KAIROS_PASS_STACK_CSE",
    "v_init_cleanup": "KAIROS_PASS_VINIT_CLEANUP",
    "classifier_batch": "KAIROS_PASS_CLASSIFIER_BATCH",
}

_TRUE_VALUES = {"1", "true", "yes", "on"}


def _env_enabled(var_name: str, override: Optional[bool]) -> bool:
    if override is not None:
        return bool(override)
    return os.environ.get(var_name, "1").strip().lower() in _TRUE_VALUES


def apply_post_fuse_passes(
    gm: torch.fx.GraphModule,
    *,
    sdpa: Optional[bool] = None,
    stack_cse: Optional[bool] = None,
    v_init_cleanup: Optional[bool] = None,
    classifier_batch: Optional[bool] = None,
    verbose: bool = True,
) -> Dict[str, object]:
    """Run the enabled post-fuse passes on `gm` in place, in B, C, A, D order.

    Each of the four keyword args, if not None, overrides the corresponding
    env var for this call (used by tests to enable a single pass in
    isolation without touching process-wide environment state).
    """
    results: Dict[str, object] = {}

    if _env_enabled(PASS_ENV_VARS["stack_cse"], stack_cse):
        stats = apply_stack_cse_pass(gm)
        results["stack_cse"] = stats
        if verbose:
            print(f"[PASS_B_STACK_CSE] groups={stats.groups} merged={stats.merged} seen={stats.stack_nodes_seen}")

    if _env_enabled(PASS_ENV_VARS["v_init_cleanup"], v_init_cleanup):
        stats = apply_v_init_cleanup_pass(gm)
        results["v_init_cleanup"] = stats
        if verbose:
            print(f"[PASS_C_VINIT_CLEANUP] candidates={stats.candidates} replaced={stats.replaced} skipped={stats.skipped}")

    if _env_enabled(PASS_ENV_VARS["sdpa"], sdpa):
        stats = apply_sdpa_pass(gm)
        results["sdpa"] = stats
        if verbose:
            print(f"[PASS_A_SDPA] matched={stats.matched} replaced={stats.replaced} skipped={stats.skipped}")

    if _env_enabled(PASS_ENV_VARS["classifier_batch"], classifier_batch):
        stats = apply_classifier_batch_pass(gm)
        results["classifier_batch"] = stats
        if verbose:
            print(f"[PASS_D_CLASSIFIER_BATCH] groups_found={stats.groups_found} replaced={stats.groups_replaced} skipped={stats.skipped}")

    return results
