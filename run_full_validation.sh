#!/bin/bash

set -e

cd /data/Triton-to-tile-IR/Tile_IR_Test/Chronos

OUT_ROOT=test/full_validation
mkdir -p ${OUT_ROOT}

WINDOWS=(1 2 4 8 16)
MODELS=("resnet18" "resnet34")

############################################
# FP32 FULL VALIDATION
############################################

for MODEL in "${MODELS[@]}"; do
  for W in "${WINDOWS[@]}"; do

    echo "========================================="
    echo "[FP32] MODEL=${MODEL} WINDOW=${W}"
    echo "========================================="

    OUT_DIR=${OUT_ROOT}/fp32_${MODEL}_w${W}
    mkdir -p ${OUT_DIR}

    python3 benchmarks/benchmark_chronos_runtime.py \
      --models ${MODEL} \
      --T 16 \
      --batch-size 16 \
      --height 224 \
      --width 224 \
      --device cuda \
      --dtype fp32 \
      --fused-op-backend triton \
      --rewrite-backend-mode inductor \
      --enable-temporal-rewrite \
      --enable-temporal-schedule \
      --enable-spatial-batching \
      --spatial-batching-ops maxpool avgpool flatten linear \
      --enable-cudagraphs \
      --cudagraph-mode reduce-overhead \
      --temporal-fuse-window ${W} \
      --temporal-schedule-window ${W} \
      --max-patterns 10000 \
      --warmup 20 \
      --repeat 100 \
      --include-s-cases \
      --print-fused-op-calls \
      --out-dir ${OUT_DIR} \
      2>&1 | tee ${OUT_DIR}/runtime.log

  done
done

############################################
# FP16 CORRECTNESS
############################################

python3 test/test_fused_convlif_kernel_configs.py \
  --device cuda \
  --dtype fp16 \
  --out-dir ${OUT_ROOT}/fp16_correctness \
  2>&1 | tee ${OUT_ROOT}/fp16_correctness.log

############################################
# FP16 BENCHMARK
############################################

for MODEL in "${MODELS[@]}"; do
  for W in "${WINDOWS[@]}"; do

    echo "========================================="
    echo "[FP16] MODEL=${MODEL} WINDOW=${W}"
    echo "========================================="

    OUT_DIR=${OUT_ROOT}/fp16_${MODEL}_w${W}
    mkdir -p ${OUT_DIR}

    python3 benchmarks/benchmark_chronos_runtime.py \
      --models ${MODEL} \
      --T 16 \
      --batch-size 16 \
      --height 224 \
      --width 224 \
      --device cuda \
      --dtype fp16 \
      --fused-op-backend triton \
      --rewrite-backend-mode inductor \
      --enable-temporal-rewrite \
      --enable-temporal-schedule \
      --enable-spatial-batching \
      --spatial-batching-ops maxpool avgpool flatten linear \
      --enable-cudagraphs \
      --cudagraph-mode reduce-overhead \
      --temporal-fuse-window ${W} \
      --temporal-schedule-window ${W} \
      --max-patterns 10000 \
      --warmup 20 \
      --repeat 100 \
      --include-s-cases \
      --print-fused-op-calls \
      --out-dir ${OUT_DIR} \
      2>&1 | tee ${OUT_DIR}/runtime.log

  done
done

echo "========================================="
echo "ALL TESTS FINISHED"
echo "========================================="