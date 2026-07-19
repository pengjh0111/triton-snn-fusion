# Kairos 

**A temporal-fusion compiler for stateful neural networks on GPUs.**

Temporal networks — spiking neural networks (SNNs), ConvLSTM/ConvGRU, gated RNNs, and linear-recurrence models — share one structural property: a *spatial* operator (convolution, linear, depthwise/pointwise conv) repeated across time steps, coupled to a lightweight *temporal* operator that carries state (membrane potential, hidden state) from step to step. Executed naively, every time step re-reads the same weights from HBM, materializes its pre-activation tensor to global memory, and launches a serialized stream of tiny state-update kernels.

This project eliminates all three costs by **compiling the spatial operator, the normalization, and the temporal state recurrence of an entire time window into a single GPU kernel**, generated and autotuned by Triton.

## Core idea: time as a first-class tiling dimension

Classical GEMM kernels tile the (M, N, K) iteration space onto the GPU memory hierarchy. We extend the tile space with **T**, decomposed into two orthogonal schedule knobs:

- **`BTILE_T`** — packs multiple time steps into the M dimension of the MMA tile (`BM_T = BLOCK_M × BTILE_T`), converting temporal work into spatial parallelism. Critical when the spatial extent alone cannot fill the GPU (small batch, streaming inference).
- **`REUSE_GROUPS`** — replicates independent accumulator groups so that a single on-chip weight tile residency serves `BTILE_T × REUSE_GROUPS` time steps (`WINDOW_T`), amortizing HBM weight traffic without growing the MMA shape.

On top of the window sits a **stateful epilogue**: the recurrence (e.g., LIF membrane update, reset, and spike generation) is applied *in registers, in strict temporal order*, across the whole T loop. Membrane state never round-trips to HBM between steps; pre-activations are never materialized. This contract is what preserves the exact semantics of step-by-step execution while the schedule above it is free to reorganize the work.

Different spatial operators are different *address generators* feeding the same temporal schedule:

| Operator | Spatial addressing | Temporal schedule |
|---|---|---|
| Conv + BN + LIF | implicit GEMM (im2col) | full `BTILE_T × REUSE_GROUPS` |
| Linear + LIF (± residual) | dense GEMM | full `BTILE_T × REUSE_GROUPS` |
| Pointwise (1×1) conv + LIF | strided GEMM (no im2col) | full `BTILE_T × REUSE_GROUPS` |
| Depthwise conv + LIF | direct convolution | `BTILE_T` only (K≈9 makes weight-reuse scheduling moot — the framework discovers this) |

Which knobs pay off is workload-dependent and discovered by autotuning, not assumed.

## Engineering: how the kernels are built

Triton's JIT cannot index dynamic containers of accumulators, so the kernels are produced by a **source-level code generator**: an emitter writes fully unrolled kernel sources (`acc_g0 … acc_gN`, static split trees) guarded by `tl.constexpr` branches. Triton then compiles each autotune configuration into a static specialization in which unused paths are eliminated at compile time — large schedule search spaces without register-pressure blowups.

The autotune space covers `BLOCK_M / BLOCK_N / BLOCK_K / BTILE_T / REUSE_GROUPS / num_warps / num_stages`, pruned by register- and shared-memory-budget rules (accumulator footprint scales with `BM_T × BLOCK_N × REUSE_GROUPS`; software-pipelined staging scales with `REUSE_GROUPS × num_stages`).

## The compiler stack

Kernels alone are not enough; the surrounding graph must feed them efficiently.

1. **FX temporal rewrite** (`compiler/fx_lif_temporal_rewrite.py`) — detects spatial-op + state-update patterns in the traced graph and replaces the per-step chains with fused temporal custom ops operating on `[T, …]` sequence tensors. Dataflow between fused ops is sequence-native: no per-step stack/unbind churn between layers.
2. **Post-fuse optimization passes** (`compiler/passes/`) — independent, individually switchable, idempotent graph passes with structural assertions and per-pass numerical acceptance (bit-exact or logits-tolerance):
   - SDPA rewrite (replaces materialized attention score chains; capture-aware backend pinning — see note below)
   - input-stack CSE
   - state-init cleanup
   - classifier-head batching
   - T-fold-batch for stateless ops (folds the time loop of state-free conv/BN into the batch dimension — one kernel call instead of T)
3. **Runtime** (`runtime/`) — custom-op registration, backend dispatch with environment-variable overrides, a standalone multi-stream graph executor with CUDA Graphs capture.

> **A lesson worth keeping:** kernel rankings measured in eager mode do not transfer into CUDA graph replay. Capture freezes launch-time heuristics; a kernel that wins by adapting at launch (e.g., flash attention) can lose 1.5× once its capture-time decisions are replayed forever. Kernels entering a captured region are re-benchmarked *in the capture context*, and backends are pinned accordingly.

## Results (examples)

Measured on RTX 5090 (locked clocks, CUDA-event timing, warm autotune caches). Representative, not exhaustive:

- **Spiking transformer** (T=16, batch=16, depth=8, dim=256): end-to-end 55.9 ms → 27.3 ms (−51%) with fused kernels + all graph passes; attention-related kernel time −55% from eliminating score-tensor materialization.
- **Batched Linear+LIF kernel**: up to 3× over the single-level temporal-chunking baseline in small-rows × large-T regimes (43× over naive per-step), with the two-level temporal schedule selected by autotuning on most shapes.
- CNN backbones (ResNet/VGG/MobileNet families) supported via the conv/dw/pw kernel family; MobileNet-specific optimizations (direct dwconv, 1×1 fast path, stateless-op batching) tracked in the roadmap. <!-- TODO: update with final MobileNet numbers -->

## Quick start

```bash
git clone https://github.com/pengjh0111/triton-snn-fusion
cd triton-snn-fusion
pip install -r requirements.txt   # PyTorch ≥ 2.x, Triton ≥ 3.x

# Validate fused kernels + rewritten graph against eager baselines
python benchmarks/validate_kairos_baselines.py \
    --enable-temporal-rewrite \
    --rewrite-backend-mode standalone
```

### Key environment flags

| Flag | Effect |
|---|---|
| `KAIROS_BATCHED_LINEAR_LIF_BACKEND` | select linear+LIF kernel backend (`codegen` / `tc` fallback) |
| `KAIROS_PASS_SDPA` | SDPA rewrite pass |
| `KAIROS_PASS_STACK_CSE` | input-stack CSE pass |
| `KAIROS_PASS_VINIT_CLEANUP` | state-init cleanup pass |
| `KAIROS_PASS_CLASSIFIER_BATCH` | classifier-head batching pass |

<!-- TODO: keep this table in sync with compiler/passes/registry.py -->

## Repository layout

```
kernels/     kernel source emitters + generated Triton kernels (do not hand-edit generated files)
compiler/    FX temporal rewrite + post-fuse pass framework
runtime/     custom ops, backend dispatch, standalone multi-stream / CUDA-graph executor
benchmarks/  end-to-end validation and benchmarking entry points
dev_tests/   kernel- and pass-level correctness suites
```

## Relationship to Chronos

[Chronos](https://github.com/pengjh0111/Chronos) schedules temporal networks *across* kernels: it batches time-independent spatial operators and orchestrates them with a temporal-tile (tTile) abstraction over multi-stream tracks — while deliberately leaving spatial–temporal fusion on the table, since ad-hoc fusion would forfeit vendor-library performance. This project removes that limitation from below: it compiles the bounded spatio-temporal region *inside* a single kernel, with the state recurrence as an in-register epilogue. The two layers compose — a tTile-style scheduler can place these fused kernels.


## Citation

Paper in preparation. <!-- TODO: add citation once available -->

## License

<!-- TODO: choose and add a license file -->
