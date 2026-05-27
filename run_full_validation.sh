#!/bin/bash

set -e

cd /data/Triton-to-tile-IR/Tile_IR_Test/Chronos

OUT_ROOT=test/full_validation
mkdir -p ${OUT_ROOT}

WINDOWS=(1 2 4 8 16)
# MODELS=("resnet18" "resnet34" "vgg11" "vgg16" "alexnet" "zfnet" "mobilenetv1" "mobilenetv2")
MODELS=("mobilenetv1")

############################################
# FP32 FULL VALIDATION
############################################

for MODEL in "${MODELS[@]}"; do

  #
  # per-model batch size
  #

  if [[ "${MODEL}" == "vgg11" ]]; then
    BATCH_SIZE=8
  elif [[ "${MODEL}" == "vgg16" ]]; then
    BATCH_SIZE=4
  else
    BATCH_SIZE=16
  fi

  for W in "${WINDOWS[@]}"; do

    echo "========================================="
    echo "[FP32] MODEL=${MODEL} WINDOW=${W} BATCH=${BATCH_SIZE}"
    echo "========================================="

    OUT_DIR=${OUT_ROOT}/${MODEL}/fp32_w${W}
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
      --rewrite-backend-mode inductor \
      --enable-temporal-rewrite \
      --enable-temporal-schedule \
      --enable-spatial-batching \
      --spatial-batching-ops conv bn add maxpool avgpool flatten linear \
      --enable-cudagraphs \
      --cudagraph-mode reduce-overhead \
      --temporal-fuse-window ${W} \
      --temporal-schedule-window ${W} \
      --max-patterns 1000000 \
      --warmup 20 \
      --repeat 100 \
      --include-s-cases \
      --out-dir ${OUT_DIR} \
      2>&1 | tee ${OUT_DIR}/runtime.log

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