#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
action="${1:-train}"
case "$action" in
    prepare|train|query|valid|test) if (( $# )); then shift; fi ;;
    -h|--help|help)
    cat <<'HELP'
Usage: bash run_meta.sh [prepare|train|query|valid|test] [OPTIONS]
  Default: train with configs/meta.json in a detached Docker container.
  Training options can be passed directly: ./run_meta.sh --resume
  Output is appended to run.log in the project directory.
  Training runs detached by default; other subcommands run in the foreground.

  prepare    Verify raw recordings and source annotation alignment
  train      Run waveform frontend meta-training
  query      Replay configured query-day metrics
  valid      Replay independent selection-day metrics
  test       Replay development-test on Days 12/13 (also scored during training)

Train options: --config configs/meta.json
               --steps N --targets X1,... --resume --plan-only
Launcher options: --device auto|cuda|cpu --dry-run --foreground
Environment: DATA_DIR, CACHE_DIR, PREPARED_DIR, OUTPUT_DIR, IMAGE_NAME,
             TRAIN_CPU_THREADS
HELP
        exit 0 ;;
    -*) action=train ;;
    *) echo "Unknown action: $action" >&2; exit 2 ;;
esac
background=false
[[ "$action" != train ]] || background=true
log_file="$project_dir/run.log"

data_dir="${DATA_DIR:-/data/users/jianchen/data/seizure_data}"
cache_dir="${CACHE_DIR:-/data/users/jianchen/data/seizure_data-cache/meta-learning}"
prepared_dir="${PREPARED_DIR:-$cache_dir/prepared}"
output_dir="${OUTPUT_DIR:-$project_dir/results/waveform-conv-meta}"
image_name="${IMAGE_NAME:-seizure-prediction-cnn:pytorch}"
device=auto
dry_run=false
arguments=()
while (( $# )); do
    case "$1" in
        --dry-run) dry_run=true; shift ;;
        --foreground) background=false; shift ;;
        --device)
            (( $# >= 2 )) || { echo '--device requires a value' >&2; exit 2; }
            device="$2"; shift 2 ;;
        *) arguments+=("$1"); shift ;;
    esac
done
for directory_name in data_dir cache_dir prepared_dir output_dir; do
    value="${!directory_name}"
    [[ "$value" == /* ]] || printf -v "$directory_name" '%s/%s' "$PWD" "$value"
done
prepared_mode=ro
if [[ "$action" == prepare ]]; then
    prepared_mode=rw
    python_args=(python prepare.py --data-dir /data/MouseLFP --output /prepared)
else
    python_args=(python train.py)
    [[ "$action" == train ]] || python_args=(python evaluate.py --split "$action")
    python_args+=(--data-dir /data/MouseLFP --prepared-dir /prepared
                  --cache-dir /cache --run-dir /output --device "$device")
fi
command_args=(docker run --rm --init --shm-size=1g
    --user "$(id -u):$(id -g)" -e HOME=/tmp
    -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1
    -e "OMP_NUM_THREADS=${TRAIN_CPU_THREADS:-4}" -e "MKL_NUM_THREADS=${TRAIN_CPU_THREADS:-4}"
    -e "OPENBLAS_NUM_THREADS=${TRAIN_CPU_THREADS:-4}"
    -w /workspace -v "$project_dir:/workspace:ro"
    -v "$data_dir:/data/MouseLFP:ro" -v "$cache_dir:/cache"
    -v "$prepared_dir:/prepared:$prepared_mode" -v "$output_dir:/output")
if [[ "$action" != prepare ]] && { [[ "$device" == cuda || "$device" == cuda:* ]] ||
    { [[ "$device" == auto ]] && command -v nvidia-smi >/dev/null; }; }; then
    command_args+=(--gpus all)
fi
if "$background"; then
    command_args+=(--detach -v "$log_file:/run-meta.log:rw" "$image_name"
        /bin/bash -c 'exec "$@" >> /run-meta.log 2>&1' run-meta)
else
    command_args+=("$image_name")
fi
command_args+=("${python_args[@]}" "${arguments[@]}")
if "$dry_run"; then
    printf '%q ' "${command_args[@]}"
    printf '\n'
    exit 0
fi
mkdir -p -- "$cache_dir" "$prepared_dir" "$output_dir"
if "$background"; then
    : >> "$log_file"
    container_id="$("${command_args[@]}")"
    printf 'Started detached training container: %s\nLog: %s\n' "$container_id" "$log_file"
    printf 'Check status: docker ps --filter id=%s\nStop: docker stop %s\n' "$container_id" "$container_id"
    exit 0
fi
exec "${command_args[@]}" >> "$log_file" 2>&1
