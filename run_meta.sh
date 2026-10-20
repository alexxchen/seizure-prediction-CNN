#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if (( $# == 0 )); then
    set -- train --config configs/meta.json
fi
action="${1:-help}"
if [[ "$action" == help || "$action" == --help || "$action" == -h ]]; then
    cat <<'HELP'
Usage: bash run_meta.sh [prepare|train|validate|test|check] [OPTIONS]
  No arguments: train with configs/meta.json in a detached Docker container.
  Output is appended to run.log in the project directory.
  Training runs detached by default; other subcommands run in the foreground.

  prepare    Verify raw recordings and source annotation alignment
  train      Run meta, pooled, or standalone log-STFT training
  validate   Replay query (Day 8) and valid (Day 11) metrics separately
  test       Replay held-out test on Days 12/13 (also scored during training)
  check      Run unit and synthetic integration tests

Train options: --config configs/meta.json --algorithm meta|pooled|log_stft
               --steps N --targets X1,... --resume --plan-only
Launcher options: --device auto|cuda|cpu --dry-run --foreground
Environment: DATA_DIR, CACHE_DIR, PREPARED_DIR, OUTPUT_DIR, IMAGE_NAME,
             TRAIN_CPU_THREADS
HELP
    exit 0
fi
shift
case "$action" in prepare|train|validate|test|check) ;; *) echo "Unknown action: $action" >&2; exit 2 ;; esac
background=false
[[ "$action" != train ]] || background=true
log_file="$project_dir/run.log"

data_dir="${DATA_DIR:-/data/users/jianchen/data/seizure_data}"
cache_dir="${CACHE_DIR:-/data/users/jianchen/data/seizure_data-cache/meta-learning}"
prepared_dir="${PREPARED_DIR:-$cache_dir/prepared}"
output_dir="${OUTPUT_DIR:-$project_dir/results/waveform-conv-meta-all}"
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
[[ "$action" != prepare ]] || prepared_mode=rw
command_args=(docker run --rm --init --shm-size=1g
    --user "$(id -u):$(id -g)" -e HOME=/tmp
    -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1
    -e "OMP_NUM_THREADS=${TRAIN_CPU_THREADS:-4}" -e "MKL_NUM_THREADS=${TRAIN_CPU_THREADS:-4}"
    -e "OPENBLAS_NUM_THREADS=${TRAIN_CPU_THREADS:-4}"
    -w /workspace -v "$project_dir:/workspace:ro"
    -v "$data_dir:/data/MouseLFP:ro" -v "$cache_dir:/cache"
    -v "$prepared_dir:/prepared:$prepared_mode" -v "$output_dir:/output")
if [[ "$action" == train || "$action" == validate || "$action" == test ]]; then
    if [[ "$device" == cuda || "$device" == cuda:* ]] || { [[ "$device" == auto ]] && command -v nvidia-smi >/dev/null; }; then
        command_args+=(--gpus all)
    fi
fi
if "$background"; then
    command_args+=(--detach -v "$log_file:/run-meta.log:rw")
fi
command_args+=("$image_name")
if "$background"; then
    command_args+=(/bin/bash -c 'exec "$@" >> /run-meta.log 2>&1' run-meta)
fi
case "$action" in
    prepare) command_args+=(python prepare.py --data-dir /data/MouseLFP --output /prepared "${arguments[@]}") ;;
    train) command_args+=(python train.py --data-dir /data/MouseLFP --prepared-dir /prepared
               --cache-dir /cache --run-dir /output --device "$device" "${arguments[@]}") ;;
    validate|test)
        split=validation
        [[ "$action" != test ]] || split=test
        command_args+=(python evaluate.py --data-dir /data/MouseLFP --prepared-dir /prepared
            --cache-dir /cache --run-dir /output --device "$device" --split "$split" "${arguments[@]}") ;;
    check) command_args+=(python -m unittest discover -s tests -v "${arguments[@]}") ;;
esac
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
