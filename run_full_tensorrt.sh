#!/bin/bash

set -e

cd /data/Triton-to-tile-IR/Tile_IR_Test/Chronos

OUT_ROOT=test/tensorrt_full_validation
mkdir -p ${OUT_ROOT}

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
# MODELS=(
# "mamba"
# )
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
# TENSORRT FULL VALIDATION
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
    echo "[TensorRT] MODEL=${MODEL} PREC=${PREC}"
    echo "========================================="

    OUT_DIR=${OUT_ROOT}/${MODEL}/${PREC}_b${BATCH_SIZE}
    mkdir -p ${OUT_DIR}

python3 benchmarks/benchmark_tensorrt_runtime.py \
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
      --workspace-mb 4096 \
      --warmup-ms 2000 \
      --duration-sec 10 \
      --out-dir ${OUT_DIR} \
      2>&1 | tee ${OUT_DIR}/runtime.log

  done
done

############################################
# AUTOTUNE OVERHEAD SUMMARY
############################################
# trtexec reports its own engine-build time ("Engine built in X sec.") --
# that's the TensorRT builder's tactic autotuning/kernel-selection cost,
# parsed into parsed.autotune_seconds by parse_trtexec_output() in
# benchmark_tensorrt_runtime.py, and already present in each
# tensorrt_summary_all.json under result["parsed"]["autotune_seconds"].
# Walk all of them here and collect one flat JSON report of autotune
# overhead per workload (model/batch/precision/execution_mode).
AUTOTUNE_SUMMARY_JSON=${OUT_ROOT}/autotune_overhead_summary.json

python3 - "${OUT_ROOT}" "${AUTOTUNE_SUMMARY_JSON}" <<'PY'
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

out_root = Path(sys.argv[1])
summary_path = Path(sys.argv[2])

DIR_RE = re.compile(r"^(?P<prec>[a-z0-9]+)_b(?P<batch>\d+)$")

records = []
by_model = defaultdict(float)

for summary_file in sorted(out_root.glob("*/*/tensorrt_summary_all.json")):
    case_dir = summary_file.parent
    model_dir = case_dir.parent

    m = DIR_RE.match(case_dir.name)
    if not m:
        continue

    batch_size = int(m.group("batch"))
    precision = m.group("prec")

    payload = json.loads(summary_file.read_text())

    for model_name, by_exec_mode in payload.items():
        for execution_mode, by_precision in by_exec_mode.items():
            for prec_key, result in by_precision.items():
                parsed = result.get("parsed", {}) or {}
                autotune_seconds = parsed.get("autotune_seconds")

                records.append({
                    "model": model_name,
                    "batch_size": batch_size,
                    "precision": prec_key,
                    "execution_mode": execution_mode,
                    "ok": result.get("ok"),
                    "trtexec_ok": result.get("trtexec_ok"),
                    "autotune_seconds": autotune_seconds,
                    "mean_ms": parsed.get("latency_ms", {}).get("mean"),
                    "out_dir": str(case_dir),
                })

                if autotune_seconds is not None:
                    by_model[model_name] += autotune_seconds

report = {
    "workloads": records,
    "total_autotune_seconds_by_model": dict(by_model),
}

summary_path.write_text(json.dumps(report, indent=2, sort_keys=True))
print(f"[AUTOTUNE OVERHEAD] wrote {len(records)} workload records to {summary_path}")
PY

echo "========================================="
echo "ALL TENSORRT TESTS FINISHED"
echo "========================================="
