#!/bin/bash

set -euo pipefail

cd /data/Triton-to-tile-IR/Tile_IR_Test/Chronos

OUT_ROOT=test/onnxruntime_full_validation
BATCH_SIZE_OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --batch-size)
      if [[ $# -lt 2 || ! "$2" =~ ^[1-9][0-9]*$ ]]; then
        echo "--batch-size requires a positive integer" >&2
        exit 2
      fi
      BATCH_SIZE_OVERRIDE="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--batch-size N]"
      echo "  --batch-size N  use N for every model; otherwise use per-model defaults"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

MODELS=(
  "resnet18"
  "resnet34"
  "resnet32"
  "alexnet"
  "zfnet"
  "vgg11"
  "vgg16"
  "mobilenetv1"
  "mobilenetv2"
  "spiketransformer"
  "spikebert"
  "convlstm"
  "mamba"
  "deepspeech2"
  "nafnet"
  "bsrn"
)
PRECISIONS=("tf32")

mkdir -p "${OUT_ROOT}"

for MODEL in "${MODELS[@]}"; do
  if [[ -n "${BATCH_SIZE_OVERRIDE}" ]]; then
    BATCH_SIZE="${BATCH_SIZE_OVERRIDE}"
  elif [[ "${MODEL}" == "vgg11" ]]; then
    BATCH_SIZE=8
  elif [[ "${MODEL}" == "vgg16" || "${MODEL}" == "nafnet" || "${MODEL}" == "bsrn" ]]; then
    BATCH_SIZE=4
  else
    BATCH_SIZE=16
  fi

  for PREC in "${PRECISIONS[@]}"; do
    echo "========================================="
    echo "[ONNX Runtime] MODEL=${MODEL} PREC=${PREC} BATCH=${BATCH_SIZE}"
    echo "========================================="

    OUT_DIR="${OUT_ROOT}/${MODEL}/${PREC}_b${BATCH_SIZE}"
    mkdir -p "${OUT_DIR}"

    python3 benchmarks/benchmark_onnxruntime_runtime.py \
      --models "${MODEL}" \
      --lif-impl kairos \
      --execution-modes single_step_mode \
      --precisions "${PREC}" \
      --T 16 \
      --batch-size "${BATCH_SIZE}" \
      --height 224 \
      --width 224 \
      --sequence-length 256 \
      --transformer-depth 8 \
      --transformer-dim 256 \
      --transformer-heads 8 \
      --transformer-input-dim 768 \
      --transformer-vocab-size 30522 \
      --transformer-num-classes 100 \
      --cudnn-conv-algo-search EXHAUSTIVE \
      --warmup-ms 2000 \
      --duration-sec 10 \
      --out-dir "${OUT_DIR}" \
      2>&1 | tee "${OUT_DIR}/runtime.log"
  done
done

echo "========================================="
echo "ALL ONNX RUNTIME TESTS FINISHED"
echo "========================================="
