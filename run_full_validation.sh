#!/bin/bash

set -e

cd /data/Triton-to-tile-IR/Tile_IR_Test/Chronos

OUT_ROOT=test/full_validation
mkdir -p ${OUT_ROOT}

WINDOWS=(1 2 4 8 16)
# MODELS=("resnet18" "resnet34" "vgg11" "vgg16" "alexnet" "zfnet" "mobilenetv1" "mobilenetv2" "spiketransformer" "spikebert")
MODELS=("spiketransformer" "spikebert")

RUN_MODE=all
while [[ $# -gt 0 ]]; do
  case "$1" in
    --chronos-only)
      RUN_MODE=chronos
      shift
      ;;
    --baseline-only)
      RUN_MODE=baseline
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [--chronos-only | --baseline-only]"
      echo "  --chronos-only   run only Chronos standalone cases"
      echo "  --baseline-only  run only the four configured baseline cases"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--chronos-only | --baseline-only]" >&2
      exit 2
      ;;
  esac
done

if [[ "${RUN_MODE}" == "chronos" ]]; then
  CASE_ARGS=(--chronos-only)
elif [[ "${RUN_MODE}" == "baseline" ]]; then
  CASE_ARGS=(--baseline-only --include-s-cases)
else
  CASE_ARGS=(--include-s-cases)
fi

echo "[RUN_MODE] ${RUN_MODE}"

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

#     python3 benchmarks/benchmark_chronos_runtime.py \
#       --models ${MODEL} \
#       --lif-impl chronos \
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
    BATCH_SIZES=(1 4 8)
  elif [[ "${MODEL}" == "vgg16" ]]; then
    BATCH_SIZES=(1 4)
  else
    BATCH_SIZES=(1 4 8 16)
  fi

  if [[ "${RUN_MODE}" == "baseline" ]]; then
    ACTIVE_WINDOWS=("baseline")
  else
    ACTIVE_WINDOWS=("${WINDOWS[@]}")
  fi

  for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    for W_TOKEN in "${ACTIVE_WINDOWS[@]}"; do

      if [[ "${W_TOKEN}" == "baseline" ]]; then
        W=1
        OUT_DIR=${OUT_ROOT}/${MODEL}/fp32_b${BATCH_SIZE}_baseline
      else
        W=${W_TOKEN}
        OUT_DIR=${OUT_ROOT}/${MODEL}/fp32_b${BATCH_SIZE}_w${W}
      fi

      echo "========================================="
      echo "[FP32] MODE=${RUN_MODE} MODEL=${MODEL} WINDOW=${W_TOKEN} BATCH=${BATCH_SIZE}"
      echo "========================================="

      mkdir -p ${OUT_DIR}

      python3 benchmarks/benchmark_chronos_runtime.py \
        --models ${MODEL} \
        --lif-impl chronos \
        --T 16 \
        --batch-size ${BATCH_SIZE} \
        --height 224 \
        --width 224 \
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
        --temporal-fuse-window ${W} \
        --temporal-schedule-window ${W} \
        --max-patterns 1000000 \
        --warmup 20 \
        --repeat 100 \
        "${CASE_ARGS[@]}" \
        --out-dir ${OUT_DIR} \
        2>&1 | tee ${OUT_DIR}/runtime.log

    done
  done
done

# ############################################
# # FP16 CORRECTNESS
# ############################################

# python3 test/test_fused_convlif_kernel_configs.py \
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

#     python3 benchmarks/benchmark_chronos_runtime.py \
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
