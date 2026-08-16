#!/bin/bash

set -e

cd /data/Triton-to-tile-IR/Tile_IR_Test/Chronos

# This benchmark still uses Relay, which has been removed from /data/tvm.
# Keep its compatible TVM/Python environment isolated from the Tile IR TVM build.
TVM_COMPAT_ROOT=/data/tvm-relay-compat
TVM_PYTHON=/home/ubuntu/miniconda3/envs/tvm-build-venv/bin/python
TVM_SITE_PACKAGES=/home/ubuntu/miniconda3/envs/tvm-build-venv/lib/python3.11/site-packages

if [[ ! -x "${TVM_PYTHON}" || ! -f "${TVM_COMPAT_ROOT}/build/libtvm.so" ]]; then
  echo "Relay-compatible TVM environment is unavailable" >&2
  exit 1
fi

export PYTHONPATH="${TVM_COMPAT_ROOT}/python:${TVM_SITE_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}"
export TVM_LIBRARY_PATH="${TVM_COMPAT_ROOT}/build"
export TVM_DISABLE_EDITABLE=1

# Python's tempfile.mkdtemp() -- used by MetaSchedule's LocalBuilder for every
# build candidate, and by nvcc during CUDA compilation -- silently falls back
# to the current working directory whenever its write-probe of /tmp fails,
# which happens as soon as the root disk (/) fills up. That's what was
# dropping tmpXXXXXXXX dirs into this repo. Pin TMPDIR to the much roomier
# /data mount so build artifacts always land there instead, regardless of how
# full / gets.
TVM_TMPDIR=/data/tmp_tvm_metaschedule
mkdir -p "${TVM_TMPDIR}"
export TMPDIR="${TVM_TMPDIR}"

OUT_ROOT=test/tvm_metaschedule_full_validation
mkdir -p ${OUT_ROOT}

BATCH_SIZE_OVERRIDE=""
FUSE_MAX_DEPTH=10
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
    --fuse-max-depth)
      if [[ $# -lt 2 || ! "$2" =~ ^-?[0-9]+$ ]]; then
        echo "--fuse-max-depth requires an integer" >&2
        exit 2
      fi
      FUSE_MAX_DEPTH="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--batch-size N] [--fuse-max-depth N]"
      echo "  --batch-size N      use N for every model; otherwise use per-model defaults"
      echo "  --fuse-max-depth N  cap on relay.FuseOps.max_depth passed to MetaSchedule"
      echo "                      tuning/compile (default: 10; TVM's own default is 256;"
      echo "                      use <=0 to keep TVM's built-in default)"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

MODELS=(
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

  if [[ -n "${BATCH_SIZE_OVERRIDE}" ]]; then
    BATCH_SIZE=${BATCH_SIZE_OVERRIDE}
  elif [[ "${MODEL}" == "vgg11" ]]; then
    BATCH_SIZE=8
  elif [[ "${MODEL}" == "vgg16" || "${MODEL}" == "nafnet" || "${MODEL}" == "bsrn" ]]; then
    BATCH_SIZE=4
  else
    BATCH_SIZE=16
  fi

  for PREC in "${PRECISIONS[@]}"; do

    echo "========================================="
    echo "[TVM MetaSchedule] MODEL=${MODEL} PREC=${PREC}"
    echo "========================================="

    OUT_DIR=${OUT_ROOT}/${MODEL}/${PREC}_b${BATCH_SIZE}
    mkdir -p ${OUT_DIR}

"${TVM_PYTHON}" -S benchmarks/benchmark_tvm_metaschedule_runtime.py \
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
      --target cuda \
      --max-trials-global 65536 \
      --num-trials-per-iter 64 \
      --builder-timeout-sec 300 \
      --fuse-max-depth ${FUSE_MAX_DEPTH} \
      --repeat 20 \
      --number 10 \
      --out-dir ${OUT_DIR} \
      2>&1 | tee ${OUT_DIR}/runtime.log

  done
done

echo "========================================="
echo "ALL TVM METASCHEDULE TESTS FINISHED"
echo "========================================="
