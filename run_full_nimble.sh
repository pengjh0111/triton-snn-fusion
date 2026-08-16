#!/bin/bash

set -euo pipefail

cd /data/Triton-to-tile-IR/Tile_IR_Test/Chronos

CONDA_ENV="/data/conda-envs/nimble-py37"
OUT_ROOT=test/nimble_full_validation
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
    --conda-env)
      CONDA_ENV="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--batch-size N] [--conda-env NAME]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

MODELS=(
  "resnet18" "resnet34" "resnet32" "alexnet" "zfnet"
  "vgg11" "vgg16" "mobilenetv1" "mobilenetv2"
  "spiketransformer" "spikebert" "convlstm" "mamba"
  "deepspeech2" "nafnet" "bsrn"
)

mkdir -p "${OUT_ROOT}"

if [[ "${CONDA_ENV}" == */* ]]; then
  CONDA_RUN=(conda run -p "${CONDA_ENV}")
else
  CONDA_RUN=(conda run -n "${CONDA_ENV}")
fi

if ! "${CONDA_RUN[@]}" python -c 'import torch; assert hasattr(torch.cuda, "Nimble")'; then
  echo "Nimble is not installed correctly in conda env ${CONDA_ENV}" >&2
  exit 1
fi

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

  OUT_DIR="${OUT_ROOT}/${MODEL}/fp32_b${BATCH_SIZE}"
  mkdir -p "${OUT_DIR}"
  echo "========================================="
  echo "[Nimble] MODEL=${MODEL} PREC=fp32 BATCH=${BATCH_SIZE}"
  echo "========================================="

  "${CONDA_RUN[@]}" --no-capture-output \
    python benchmarks/benchmark_nimble_runtime.py \
      --models "${MODEL}" \
      --dtype fp32 \
      --T 16 \
      --batch-size "${BATCH_SIZE}" \
      --height 224 \
      --width 224 \
      --warmup 20 \
      --repeat 100 \
      --use-multi-stream \
      --out-dir "${OUT_DIR}" \
      2>&1 | tee "${OUT_DIR}/runtime.log"
done

echo "========================================="
echo "ALL NIMBLE TESTS FINISHED"
echo "========================================="
