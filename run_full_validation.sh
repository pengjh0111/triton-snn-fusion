#!/bin/bash

set -e

cd /data/Triton-to-tile-IR/Tile_IR_Test/Chronos

OUT_ROOT=test/full_validation
mkdir -p ${OUT_ROOT}

# Candidate temporal fuse/schedule windows for the Kairos cases. These are
# no longer swept by an outer bash loop (one python3 process per window) --
# they're passed to benchmark_kairos_runtime.py's own
# --sweep-temporal-windows / --temporal-window-candidates, so window size is
# searched *inside* a single process alongside the rest of autotune, and the
# script picks whichever window actually measures fastest (best_case).
TEMPORAL_WINDOW_CANDIDATES=(1 2 4 8 16)
MODELS=("vgg11" "vgg16" "mobilenetv1" "mobilenetv2")
# MODELS=("convlstm" "mamba" "deepspeech2")

# Full per-case diagnostic output (annotation/spatial-batching/rewrite/pass
# stats, one line per fused window, etc.) is verbose and is always kept
# intact in each case's runtime.log via `tee` below. The console only needs
# progress banners, per-case timing, and the final autotune summary table --
# this awk filter narrows *console* output to that, without touching what
# gets written to the log file or requiring changes to the benchmark script.
concise_console() {
  awk '
    /Traceback|Error|error:/ { print; next }
    /^\[AUTOTUNE SUMMARY\]/ { show=1 }
    show { print; next }
    /^\[BENCH\]/ || /^  [A-Za-z0-9_]+ mean=/ { print }
  '
}

RUN_MODE=all
BASELINE_MAX_AUTOTUNE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --kairos-only)
      RUN_MODE=kairos
      shift
      ;;
    --baseline-only)
      RUN_MODE=baseline
      shift
      ;;
    --baseline-max-autotune)
      BASELINE_MAX_AUTOTUNE=1
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [--kairos-only | --baseline-only] [--baseline-max-autotune]"
      echo "  --kairos-only          run only Kairos standalone cases"
      echo "  --baseline-only        run only the four configured baseline cases"
      echo "  --baseline-max-autotune  compile baseline_*_compile cases with"
      echo "                         torch.compile mode='max-autotune' (off by"
      echo "                         default); has no effect on Kairos cases,"
      echo "                         which have their own autotune already"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--kairos-only | --baseline-only] [--baseline-max-autotune]" >&2
      exit 2
      ;;
  esac
done

if [[ "${RUN_MODE}" == "kairos" ]]; then
  CASE_ARGS=(--kairos-only)
elif [[ "${RUN_MODE}" == "baseline" ]]; then
  CASE_ARGS=(--baseline-only --include-s-cases)
else
  CASE_ARGS=(--include-s-cases)
fi

if [[ "${BASELINE_MAX_AUTOTUNE}" == "1" ]]; then
  CASE_ARGS+=(--baseline-max-autotune)
fi

echo "[RUN_MODE] ${RUN_MODE}"
echo "[BASELINE_MAX_AUTOTUNE] ${BASELINE_MAX_AUTOTUNE}"

############################################
# FP32 FULL VALIDATION
############################################

# for MODEL in "${MODELS[@]}"; do

  
#   # per-model batch size
  

#   if [[ "${MODEL}" == "vgg11" ]]; then
#     BATCH_SIZE=8
#   elif [[ "${MODEL}" == "vgg16" ]]; then
#     BATCH_SIZE=4
#   else
#     BATCH_SIZE=16
#   fi

#   for W in "${WINDOWS[@]}"; do

#     echo "========================================="
#     echo "[FP32] MODEL=${MODEL} WINDOW=${W} BATCH=${BATCH_SIZE}"
#     echo "========================================="

#     OUT_DIR=${OUT_ROOT}/${MODEL}/fp32_w${W}
#     mkdir -p ${OUT_DIR}

#     python3 benchmarks/benchmark_kairos_runtime.py \
#       --models ${MODEL} \
#       --lif-impl kairos \
#       --T 16 \
#       --batch-size ${BATCH_SIZE} \
#       --height 224 \
#       --width 224 \
#       --device cuda \
#       --dtype fp32 \
#       --fused-op-backend triton \
#       --rewrite-backend-mode inductor \
#       --enable-temporal-rewrite \
#       --enable-temporal-schedule \
#       --enable-spatial-batching \
#       --spatial-batching-ops conv bn add maxpool avgpool flatten linear \
#       --enable-cudagraphs \
#       --cudagraph-mode reduce-overhead \
#       --temporal-fuse-window ${W} \
#       --temporal-schedule-window ${W} \
#       --max-patterns 1000000 \
#       --warmup 20 \
#       --repeat 100 \
#       --include-s-cases \
#       --out-dir ${OUT_DIR} \
#       2>&1 | tee ${OUT_DIR}/runtime.log

#   done
# done

# multi-stream
for MODEL in "${MODELS[@]}"; do

  #
  # per-model batch size sweep
  #

  if [[ "${MODEL}" == "vgg11" ]]; then
    BATCH_SIZES=(4)
  elif [[ "${MODEL}" == "vgg16" ]]; then
    BATCH_SIZES=(4)
  else
    BATCH_SIZES=(4)
  fi

  # Per-model input shape. Image models use the generic --height/--width
  # (224x224). ConvLSTM has its own --convlstm-height/--convlstm-width
  # flags (already defaulted to 64x64 in make_model_input) -- the generic
  # --height/--width is never read for it, so pass the matching 64x64
  # here too rather than a misleading 224x224. Mamba/DeepSpeech2 are pure
  # sequence-input models (input built from --mamba-* / --deepspeech2-*
  # flags alone, dispatched via KAIROS_MODEL_INPUT_MODE registration
  # metadata) with no spatial height/width concept at all, so the flag is
  # omitted for them entirely rather than passed and ignored.
  if [[ "${MODEL}" == "convlstm" ]]; then
    HEIGHT_WIDTH_ARGS=(--height 64 --width 64)
  elif [[ "${MODEL}" == "mamba" || "${MODEL}" == "deepspeech2" ]]; then
    HEIGHT_WIDTH_ARGS=()
  else
    HEIGHT_WIDTH_ARGS=(--height 224 --width 224)
  fi

  # Window size is no longer a bash-level dimension: baseline mode has no
  # window concept at all (no Kairos cases are built), and kairos/all mode
  # hands the whole candidate list to a single process via
  # --sweep-temporal-windows so the window search itself is part of autotune.
  if [[ "${RUN_MODE}" == "baseline" ]]; then
    WINDOW_ARGS=(--temporal-fuse-window 1 --temporal-schedule-window 1)
  else
    WINDOW_ARGS=(--sweep-temporal-windows --temporal-window-candidates "${TEMPORAL_WINDOW_CANDIDATES[@]}")
  fi

  for BATCH_SIZE in "${BATCH_SIZES[@]}"; do

    if [[ "${RUN_MODE}" == "baseline" ]]; then
      OUT_DIR=${OUT_ROOT}/${MODEL}/fp32_b${BATCH_SIZE}_baseline
    else
      OUT_DIR=${OUT_ROOT}/${MODEL}/fp32_b${BATCH_SIZE}_wsweep
    fi

    echo "========================================="
    echo "[FP32] MODE=${RUN_MODE} MODEL=${MODEL} BATCH=${BATCH_SIZE}"
    echo "========================================="

    mkdir -p ${OUT_DIR}

    python3 benchmarks/benchmark_kairos_runtime.py \
      --models ${MODEL} \
      --lif-impl kairos \
      --T 16 \
      --batch-size ${BATCH_SIZE} \
      "${HEIGHT_WIDTH_ARGS[@]}" \
      --device cuda \
      --dtype fp32 \
      --fused-op-backend triton \
      --rewrite-backend-mode standalone \
      --fx-standalone-streams 32 \
      --fx-standalone-cudagraph \
      --fx-standalone-schedule-policy ready \
      --enable-temporal-rewrite \
      --enable-temporal-schedule \
      --enable-spatial-batching \
      --spatial-batching-ops conv bn add maxpool avgpool flatten linear \
      --cudagraph-mode reduce-overhead \
      "${WINDOW_ARGS[@]}" \
      --max-patterns 1000000 \
      --warmup 20 \
      --repeat 100 \
      "${CASE_ARGS[@]}" \
      --out-dir ${OUT_DIR} \
      2>&1 | tee ${OUT_DIR}/runtime.log | concise_console

  done
done

############################################
# AUTOTUNE OVERHEAD SUMMARY
############################################
# Each benchmark_summary.json written above (one per model/batch OUT_DIR)
# now carries a per-case "autotune_seconds" field -- the wall-clock time
# spent in compile_and_warmup() (torch.compile + Triton kernel autotuning),
# timed separately from the steady-state benchmark loop in
# benchmark_kairos_runtime.py:run_case(). In "wsweep" dirs, --sweep-temporal-
# windows produced one case per candidate window (e.g.
# kairos_single_step_loop_compile_w4), and the script's own "best_case" is
# whichever one measured fastest -- so the window size itself was part of
# the autotune search. The headline "autotune_seconds" recorded per workload
# below is that winning case's compile+warmup cost (not the sum across all
# candidate windows that were tried and discarded); the full per-candidate
# breakdown is kept alongside under "window_candidates" for reference.
AUTOTUNE_SUMMARY_JSON=${OUT_ROOT}/autotune_overhead_summary.json

python3 - "${OUT_ROOT}" "${AUTOTUNE_SUMMARY_JSON}" <<'PY'
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

out_root = Path(sys.argv[1])
summary_path = Path(sys.argv[2])

DIR_RE = re.compile(r"^fp32_b(?P<batch>\d+)_(?:wsweep|baseline)$")
CASE_WINDOW_RE = re.compile(r"_w(?P<window>\d+)$")

records = []
by_model = defaultdict(float)

for summary_file in sorted(out_root.glob("*/*/benchmark_summary.json")):
    case_dir = summary_file.parent
    model_dir = case_dir.parent

    m = DIR_RE.match(case_dir.name)
    if not m:
        continue

    mode = "baseline" if case_dir.name.endswith("_baseline") else "wsweep"
    batch_size = int(m.group("batch"))

    payload = json.loads(summary_file.read_text())
    results = payload.get("results", {})
    best_case_info = payload.get("best_case")

    window_candidates = []
    for case_name, case_result in results.items():
        cm = CASE_WINDOW_RE.search(case_name)
        if not cm:
            continue
        window_candidates.append({
            "window": int(cm.group("window")),
            "case": case_name,
            "ok": case_result.get("ok"),
            "autotune_seconds": case_result.get("autotune_seconds"),
            "mean_ms": case_result.get("mean_ms"),
            "is_best": bool(best_case_info) and case_name == best_case_info.get("case"),
        })
    window_candidates.sort(key=lambda c: c["window"])

    best_case_name = best_case_info.get("case") if best_case_info else None
    best_result = results.get(best_case_name, {}) if best_case_name else {}
    best_window = None
    if best_case_name:
        bm = CASE_WINDOW_RE.search(best_case_name)
        best_window = int(bm.group("window")) if bm else None
    autotune_seconds = best_result.get("autotune_seconds")

    records.append({
        "model": model_dir.name,
        "batch_size": batch_size,
        "mode": mode,
        "best_case": best_case_name,
        "window": best_window,
        "ok": best_result.get("ok"),
        "autotune_seconds": autotune_seconds,
        "mean_ms": best_result.get("mean_ms"),
        "window_candidates": window_candidates,
        "out_dir": str(case_dir),
    })

    if autotune_seconds is not None:
        by_model[model_dir.name] += autotune_seconds

report = {
    "workloads": records,
    "total_autotune_seconds_by_model": dict(by_model),
}

summary_path.write_text(json.dumps(report, indent=2, sort_keys=True))
print(f"[AUTOTUNE OVERHEAD] wrote {len(records)} workload records to {summary_path}")
PY

# ############################################
# # FP16 CORRECTNESS
# ############################################

# python3 benchmarks/helpers/fused_convlif_kernel_configs.py \
#   --device cuda \
#   --dtype fp16 \
#   --out-dir ${OUT_ROOT}/fp16_correctness \
#   2>&1 | tee ${OUT_ROOT}/fp16_correctness.log

# ############################################
# # FP16 BENCHMARK
# ############################################

# for MODEL in "${MODELS[@]}"; do

#   #
#   # per-model batch size
#   #

#   if [[ "${MODEL}" == "vgg11" ]]; then
#     BATCH_SIZE=8
#   elif [[ "${MODEL}" == "vgg16" ]]; then
#     BATCH_SIZE=4
#   else
#     BATCH_SIZE=16
#   fi

#   for W in "${WINDOWS[@]}"; do

#     echo "========================================="
#     echo "[FP16] MODEL=${MODEL} WINDOW=${W} BATCH=${BATCH_SIZE}"
#     echo "========================================="

#     OUT_DIR=${OUT_ROOT}/${MODEL}/fp16_w${W}
#     mkdir -p ${OUT_DIR}

#     python3 benchmarks/benchmark_kairos_runtime.py \
#       --models ${MODEL} \
#       --T 16 \
#       --batch-size ${BATCH_SIZE} \
#       --height 224 \
#       --width 224 \
#       --device cuda \
#       --dtype fp16 \
#       --fused-op-backend triton \
#       --rewrite-backend-mode inductor \
#       --enable-temporal-rewrite \
#       --enable-temporal-schedule \
#       --enable-spatial-batching \
#       --spatial-batching-ops conv bn add maxpool avgpool flatten linear \
#       --enable-cudagraphs \
#       --cudagraph-mode reduce-overhead \
#       --temporal-fuse-window ${W} \
#       --temporal-schedule-window ${W} \
#       --max-patterns 1000000 \
#       --warmup 20 \
#       --repeat 100 \
#       --include-s-cases \
#       --out-dir ${OUT_DIR} \
#       2>&1 | tee ${OUT_DIR}/runtime.log

#   done
# done

echo "========================================="
echo "ALL TESTS FINISHED"
echo "========================================="
