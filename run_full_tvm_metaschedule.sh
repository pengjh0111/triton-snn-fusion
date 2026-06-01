#!/bin/bash

set -e

cd /data/Triton-to-tile-IR/Tile_IR_Test/Chronos

OUT_ROOT=test/tvm_metaschedule_full_validation
mkdir -p ${OUT_ROOT}

MODELS=(
  "resnet18"
  "resnet34"
  "alexnet"
  "zfnet"
  "vgg11"
  "vgg16"
  "mobilenetv1"
  "mobilenetv2"
)

PRECISIONS=(
  "tf32"
)

############################################
# TVM METASCHEDULE FULL VALIDATION
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

  for PREC in "${PRECISIONS[@]}"; do

    echo "========================================="
    echo "[TVM MetaSchedule] MODEL=${MODEL} PREC=${PREC}"
    echo "========================================="

    OUT_DIR=${OUT_ROOT}/${MODEL}/${PREC}
    mkdir -p ${OUT_DIR}

python3 benchmarks/benchmark_tvm_metaschedule_runtime.py \
      --models ${MODEL} \
      --lif-impl chronos \
      --execution-modes single_step_mode \
      --precisions ${PREC} \
      --T 16 \
      --batch-size ${BATCH_SIZE} \
      --height 224 \
      --width 224 \
      --target cuda \
      --max-trials-global 1024 \
      --num-trials-per-iter 64 \
      --repeat 20 \
      --number 10 \
      --out-dir ${OUT_DIR} \
      2>&1 | tee ${OUT_DIR}/runtime.log

  done
done

echo "========================================="
echo "ALL TVM METASCHEDULE TESTS FINISHED"
echo "========================================="
