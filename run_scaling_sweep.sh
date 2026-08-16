#!/usr/bin/env bash

# Sweep batch size and timesteps across Chronos and all external baselines.
#
# The eight selectable backend groups are:
#   pytorch     four cases: S-eager, S-compile, M-eager, M-compile
#   kairos      all power-of-two temporal windows <= T
#   tvm         Relay MetaSchedule, matching run_full_tvm_metaschedule.sh
#   tensorrt    matching run_full_tensorrt.sh
#   welder      matching run_full_welder.sh
#   bladedisc   delegated to /data/run_bladedisc_5090.sh
#   onnxruntime matching run_full_onnxruntime.sh
#   nimble      legacy Nimble environment

set -uo pipefail

CHRONOS_ROOT="/data/Triton-to-tile-IR/Tile_IR_Test/Chronos"
cd "${CHRONOS_ROOT}"

ALL_BACKENDS=(pytorch kairos tvm tensorrt welder bladedisc onnxruntime nimble)
MODELS=(resnet18 alexnet spiketransformer mamba convlstm)
TIME_SWEEP=(4 8 16 32 64)
BATCH_SWEEP=(1 4 8 16 32 64)

OUT_ROOT="test/scaling_sweep"
WARMUP=20
REPEAT=100
DRY_RUN=0
SELECTED_BACKENDS=()

TVM_COMPAT_ROOT="/data/tvm-relay-compat"
TVM_PYTHON="/home/ubuntu/miniconda3/envs/tvm-build-venv/bin/python"
TVM_SITE_PACKAGES="/home/ubuntu/miniconda3/envs/tvm-build-venv/lib/python3.11/site-packages"
TVM_TMPDIR="/data/tmp_tvm_metaschedule"
NIMBLE_ENV="/data/conda-envs/nimble-py37"
BLADEDISC_RUNNER="/data/run_bladedisc_5090.sh"

usage() {
  cat <<'EOF'
Usage:
  ./run_scaling_sweep.sh --backends LIST [options]
  ./run_scaling_sweep.sh --pytorch --kairos [options]
  ./run_scaling_sweep.sh --all [options]

Backend selection (eight groups):
  --backends LIST     Comma-separated subset of:
                      pytorch,kairos,tvm,tensorrt,welder,bladedisc,
                      onnxruntime,nimble
  --pytorch           Four PyTorch cases (S/M eager and compile)
  --kairos            Kairos, sweeping windows 1,2,...,T
  --tvm               TVM MetaSchedule
  --tensorrt          TensorRT
  --welder            Welder
  --bladedisc         BladeDISC
  --onnxruntime       ONNX Runtime
  --nimble            Nimble
  --all               Select all eight groups

Sweep definition:
  1. batch=4, T in {4,8,16,32,64}
  2. T=16, batch in {1,4,8,16,32,64}
  The shared (batch=4,T=16) point runs once.

Other options:
  --models LIST       Comma-separated workload subset (default: all five)
  --out-root DIR      Output root (default: test/scaling_sweep)
  --warmup N          PyTorch/Kairos/Nimble warmup count (default: 20)
  --repeat N          PyTorch/Kairos/Nimble repeat count (default: 100)
  --nimble-env PATH   Nimble conda prefix/name
  --dry-run           Print commands without running workloads
  -h, --help          Show this help

Precision policy:
  PyTorch/Kairos/BladeDISC/Nimble use FP32 tensors and do not disable TF32.
  TVM/TensorRT/Welder/ONNX Runtime use their existing tf32 configuration.

Failures:
  A failed model/config is recorded in sweep_status.tsv and the sweep continues.
EOF
}

contains() {
  local needle="$1"
  shift
  local item
  for item in "$@"; do
    [[ "${item}" == "${needle}" ]] && return 0
  done
  return 1
}

add_backend() {
  local backend="$1"
  if ! contains "${backend}" "${ALL_BACKENDS[@]}"; then
    echo "Unknown backend: ${backend}" >&2
    exit 2
  fi
  if ! contains "${backend}" "${SELECTED_BACKENDS[@]}"; then
    SELECTED_BACKENDS+=("${backend}")
  fi
}

parse_csv_backends() {
  local values
  IFS=',' read -r -a values <<< "$1"
  local value
  for value in "${values[@]}"; do
    add_backend "${value}"
  done
}

parse_models() {
  local requested
  IFS=',' read -r -a requested <<< "$1"
  MODELS=()
  local model
  for model in "${requested[@]}"; do
    [[ "${model}" == "spiketransforme" ]] && model="spiketransformer"
    case "${model}" in
      resnet18|mamba|convlstm|spiketransformer) MODELS+=("${model}") ;;
      *) echo "Unsupported sweep workload: ${model}" >&2; exit 2 ;;
    esac
  done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backends) parse_csv_backends "$2"; shift 2 ;;
    --pytorch) add_backend pytorch; shift ;;
    --kairos) add_backend kairos; shift ;;
    --tvm) add_backend tvm; shift ;;
    --tensorrt) add_backend tensorrt; shift ;;
    --welder) add_backend welder; shift ;;
    --bladedisc) add_backend bladedisc; shift ;;
    --onnxruntime) add_backend onnxruntime; shift ;;
    --nimble) add_backend nimble; shift ;;
    --all) SELECTED_BACKENDS=("${ALL_BACKENDS[@]}"); shift ;;
    --models) parse_models "$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --repeat) REPEAT="$2"; shift 2 ;;
    --nimble-env) NIMBLE_ENV="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ${#SELECTED_BACKENDS[@]} -eq 0 ]]; then
  echo "No backend selected. Use --backends LIST, individual flags, or --all." >&2
  usage >&2
  exit 2
fi
if [[ ! "${WARMUP}" =~ ^[0-9]+$ || ! "${REPEAT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--warmup must be non-negative and --repeat must be positive" >&2
  exit 2
fi

mkdir -p "${OUT_ROOT}"
STATUS_FILE="${OUT_ROOT}/sweep_status.tsv"
if [[ ! -f "${STATUS_FILE}" ]]; then
  printf 'timestamp\tbackend\tmodel\tbatch\tT\tsweep\tstatus\texit_code\toutput_dir\n' > "${STATUS_FILE}"
fi

shape_args() {
  local model="$1"
  if [[ "${model}" == "convlstm" ]]; then
    SHAPE_ARGS=(--height 64 --width 64)
  elif [[ "${model}" == "mamba" ]]; then
    SHAPE_ARGS=()
  else
    SHAPE_ARGS=(--height 224 --width 224)
  fi
}

temporal_windows() {
  local timesteps="$1"
  TEMPORAL_WINDOWS=()
  local window=1
  while (( window <= timesteps )); do
    TEMPORAL_WINDOWS+=("${window}")
    window=$((window * 2))
  done
}

print_command() {
  printf ' %q' "$@"
  printf '\n'
}

execute_case() {
  local backend="$1" model="$2" batch="$3" timesteps="$4" sweep="$5"
  local case_dir="${OUT_ROOT}/${backend}/${model}/b${batch}_T${timesteps}"
  # Keep BladeDISC's directory convention so its JSON and this wrapper's log
  # land in the same location.
  if [[ "${backend}" == "bladedisc" ]]; then
    case_dir="${OUT_ROOT}/bladedisc/${model}/fp32_b${batch}_T${timesteps}"
  fi
  mkdir -p "${case_dir}"
  shape_args "${model}"
  temporal_windows "${timesteps}"

  local -a cmd=()
  case "${backend}" in
    pytorch)
      cmd=(python3 benchmarks/benchmark_kairos_runtime.py
        --models "${model}" --lif-impl kairos --T "${timesteps}"
        --batch-size "${batch}" "${SHAPE_ARGS[@]}" --device cuda --dtype fp32
        --fused-op-backend triton --rewrite-backend-mode standalone
        --warmup "${WARMUP}" --repeat "${REPEAT}"
        --baseline-only --include-s-cases --out-dir "${case_dir}")
      ;;
    kairos)
      cmd=(python3 benchmarks/benchmark_kairos_runtime.py
        --models "${model}" --lif-impl kairos --T "${timesteps}"
        --batch-size "${batch}" "${SHAPE_ARGS[@]}" --device cuda --dtype fp32
        --fused-op-backend triton --rewrite-backend-mode standalone
        --fx-standalone-streams 32 --fx-standalone-cudagraph
        --fx-standalone-schedule-policy ready
        --enable-temporal-rewrite --enable-temporal-schedule
        --enable-spatial-batching
        --spatial-batching-ops conv bn add maxpool avgpool flatten linear
        --cudagraph-mode reduce-overhead --max-patterns 1000000
        --sweep-temporal-windows
        --temporal-window-candidates "${TEMPORAL_WINDOWS[@]}"
        --warmup "${WARMUP}" --repeat "${REPEAT}"
        --kairos-only --out-dir "${case_dir}")
      ;;
    tvm)
      mkdir -p "${TVM_TMPDIR}"
      local tvm_pythonpath="${TVM_COMPAT_ROOT}/python:${TVM_SITE_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}"
      cmd=(env PYTHONPATH="${tvm_pythonpath}"
        TVM_LIBRARY_PATH="${TVM_COMPAT_ROOT}/build" TVM_DISABLE_EDITABLE=1
        TMPDIR="${TVM_TMPDIR}" "${TVM_PYTHON}" -S
        benchmarks/benchmark_tvm_metaschedule_runtime.py
        --models "${model}" --lif-impl kairos
        --execution-modes single_step_mode --precisions tf32
        --T "${timesteps}" --batch-size "${batch}" "${SHAPE_ARGS[@]}"
        --sequence-length 256 --transformer-depth 8 --transformer-dim 256
        --transformer-heads 8 --transformer-input-dim 768
        --transformer-vocab-size 30522 --transformer-num-classes 100
        --target cuda --max-trials-global 8192 --num-trials-per-iter 64
        --builder-timeout-sec 300 --fuse-max-depth 10
        --repeat 20 --number 10 --out-dir "${case_dir}")
      ;;
    tensorrt)
      cmd=(python3 benchmarks/benchmark_tensorrt_runtime.py
        --models "${model}" --lif-impl kairos
        --execution-modes single_step_mode --precisions tf32
        --T "${timesteps}" --batch-size "${batch}" "${SHAPE_ARGS[@]}"
        --sequence-length 256 --transformer-depth 8 --transformer-dim 256
        --transformer-heads 8 --transformer-input-dim 768
        --transformer-vocab-size 30522 --transformer-num-classes 100
        --workspace-mb 4096 --warmup-ms 2000 --duration-sec 10
        --out-dir "${case_dir}")
      ;;
    welder)
      cmd=(python3 benchmarks/benchmark_welder_runtime.py
        --models "${model}" --lif-impl kairos
        --execution-modes single_step_mode --precisions tf32
        --T "${timesteps}" --batch-size "${batch}" "${SHAPE_ARGS[@]}"
        --sequence-length 256 --transformer-depth 8 --transformer-dim 256
        --transformer-heads 8 --transformer-input-dim 768
        --transformer-vocab-size 30522 --transformer-num-classes 100
        --topk 20 --arch RTX5090 --skip-dot --out-dir "${case_dir}")
      ;;
    bladedisc)
      cmd=("${BLADEDISC_RUNNER}" --models "${model}" --batch-size "${batch}"
        --T "${timesteps}" --dtype fp32 --warmup "${WARMUP}"
        --repeat "${REPEAT}" --out-root "${OUT_ROOT}/bladedisc")
      ;;
    onnxruntime)
      cmd=(python3 benchmarks/benchmark_onnxruntime_runtime.py
        --models "${model}" --lif-impl kairos
        --execution-modes single_step_mode --precisions tf32
        --T "${timesteps}" --batch-size "${batch}" "${SHAPE_ARGS[@]}"
        --sequence-length 256 --transformer-depth 8 --transformer-dim 256
        --transformer-heads 8 --transformer-input-dim 768
        --transformer-vocab-size 30522 --transformer-num-classes 100
        --cudnn-conv-algo-search EXHAUSTIVE --warmup-ms 2000
        --duration-sec 10 --out-dir "${case_dir}")
      ;;
    nimble)
      local -a conda_run
      if [[ "${NIMBLE_ENV}" == */* ]]; then
        conda_run=(conda run -p "${NIMBLE_ENV}")
      else
        conda_run=(conda run -n "${NIMBLE_ENV}")
      fi
      cmd=("${conda_run[@]}" --no-capture-output python
        benchmarks/benchmark_nimble_runtime.py --models "${model}"
        --dtype fp32 --T "${timesteps}" --batch-size "${batch}"
        "${SHAPE_ARGS[@]}" --warmup "${WARMUP}" --repeat "${REPEAT}"
        --use-multi-stream --out-dir "${case_dir}")
      ;;
  esac

  echo
  echo "================================================================"
  echo "[SWEEP] backend=${backend} model=${model} batch=${batch} T=${timesteps} group=${sweep}"
  echo "[OUT]   ${case_dir}"
  echo "[CMD]"
  print_command "${cmd[@]}"
  if (( DRY_RUN )); then
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$(date -u +%FT%TZ)" "${backend}" "${model}" "${batch}" "${timesteps}" \
      "${sweep}" dry-run 0 "${case_dir}" >> "${STATUS_FILE}"
    return 0
  fi

  "${cmd[@]}" 2>&1 | tee "${case_dir}/runtime.log"
  local exit_code=${PIPESTATUS[0]}
  local status=ok
  [[ ${exit_code} -ne 0 ]] && status=failed
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -u +%FT%TZ)" "${backend}" "${model}" "${batch}" "${timesteps}" \
    "${sweep}" "${status}" "${exit_code}" "${case_dir}" >> "${STATUS_FILE}"
  if [[ ${exit_code} -ne 0 ]]; then
    echo "[WARN] failed with exit=${exit_code}; continuing" >&2
  fi
  return 0
}

echo "[BACKENDS] ${SELECTED_BACKENDS[*]}"
echo "[MODELS]   ${MODELS[*]}"
echo "[OUTPUT]   ${OUT_ROOT}"
(( DRY_RUN )) && echo "[MODE]     dry-run"

# Generate the union of both sweeps. Associative keys remove B=4,T=16 duplicate.
declare -A CASE_GROUP=()
for timesteps in "${TIME_SWEEP[@]}"; do
  CASE_GROUP["4:${timesteps}"]="time"
done
for batch in "${BATCH_SWEEP[@]}"; do
  key="${batch}:16"
  if [[ -n "${CASE_GROUP[$key]+x}" ]]; then
    CASE_GROUP["${key}"]="time+batch"
  else
    CASE_GROUP["${key}"]="batch"
  fi
done

# Preserve deterministic order: time sweep first, then new batch-sweep points.
CASES=()
for timesteps in "${TIME_SWEEP[@]}"; do CASES+=("4:${timesteps}"); done
for batch in "${BATCH_SWEEP[@]}"; do
  key="${batch}:16"
  contains "${key}" "${CASES[@]}" || CASES+=("${key}")
done

for backend in "${SELECTED_BACKENDS[@]}"; do
  for model in "${MODELS[@]}"; do
    for key in "${CASES[@]}"; do
      batch="${key%%:*}"
      timesteps="${key##*:}"
      execute_case "${backend}" "${model}" "${batch}" "${timesteps}" "${CASE_GROUP[$key]}"
    done
  done
done

echo
echo "================================================================"
echo "SWEEP COMPLETE"
echo "Status manifest: ${STATUS_FILE}"
echo "================================================================"
