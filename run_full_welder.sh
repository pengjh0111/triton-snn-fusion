#!/bin/bash

set -e

cd /data/Triton-to-tile-IR/Tile_IR_Test/Chronos

OUT_ROOT=test/welder_full_validation
mkdir -p ${OUT_ROOT}

BATCH_SIZE_OVERRIDE=""
TOPK=20
ARCH="RTX5090"
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
    --topk)
      if [[ $# -lt 2 || ! "$2" =~ ^[1-9][0-9]*$ ]]; then
        echo "--topk requires a positive integer" >&2
        exit 2
      fi
      TOPK="$2"
      shift 2
      ;;
    --arch)
      if [[ $# -lt 2 ]]; then
        echo "--arch requires a value" >&2
        exit 2
      fi
      ARCH="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--batch-size N] [--topk K] [--arch NAME]"
      echo "  --batch-size N  use N for every model; otherwise use per-model defaults"
      echo "  --topk K        welder tuning trials per subgraph (default 20)"
      echo "  --arch NAME     welder arch profile (default RTX5090)"
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
)
PRECISIONS=(
  "tf32"
)

############################################
# WELDER FULL VALIDATION
############################################

for MODEL in "${MODELS[@]}"; do

  #
  # per-model batch size
  #

  if [[ -n "${BATCH_SIZE_OVERRIDE}" ]]; then
    BATCH_SIZE=${BATCH_SIZE_OVERRIDE}
  elif [[ "${MODEL}" == "vgg11" ]]; then
    BATCH_SIZE=8
  elif [[ "${MODEL}" == "vgg16" ]]; then
    BATCH_SIZE=4
  else
    BATCH_SIZE=16
  fi

  for PREC in "${PRECISIONS[@]}"; do

    echo "========================================="
    echo "[Welder] MODEL=${MODEL} PREC=${PREC} TOPK=${TOPK} ARCH=${ARCH}"
    echo "========================================="

    OUT_DIR=${OUT_ROOT}/${MODEL}/${PREC}_b${BATCH_SIZE}
    mkdir -p ${OUT_DIR}

    python3 benchmarks/benchmark_welder_runtime.py \
      --models ${MODEL} \
      --lif-impl kairos \
      --execution-modes single_step_mode \
      --precisions ${PREC} \
      --T 16 \
      --batch-size ${BATCH_SIZE} \
      --height 224 \
      --width 224 \
      --sequence-length 256 \
      --transformer-depth 8 \
      --transformer-dim 256 \
      --transformer-heads 8 \
      --transformer-input-dim 768 \
      --transformer-vocab-size 30522 \
      --transformer-num-classes 100 \
      --topk ${TOPK} \
      --arch ${ARCH} \
      --out-dir ${OUT_DIR} \
      2>&1 | tee ${OUT_DIR}/runtime.log

  done
done

echo "========================================="
echo "ALL WELDER TESTS FINISHED"
echo "========================================="
