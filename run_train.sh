#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
data_dir="${DATA_DIR:-/data/users/jianchen/data/seizure_data}"
cache_dir="${CACHE_DIR:-/data/users/jianchen/data/seizure_data-cache}"
run_stamp="$(date -u +%Y%m%dT%H%M%SZ)-$$"
output_dir="${OUTPUT_DIR:-$project_dir/docker-output-mouse-training/$run_stamp}"
image_name="${IMAGE_NAME:-seizure-prediction-cnn:pytorch}"
dry_run=false
training_args=()
for argument in "$@"; do
    if [[ "$argument" == --dry-run ]]; then
        dry_run=true
    else
        training_args+=("$argument")
    fi
done
[[ "$data_dir" == /* ]] || data_dir="$PWD/$data_dir"
[[ "$cache_dir" == /* ]] || cache_dir="$PWD/$cache_dir"
[[ "$output_dir" == /* ]] || output_dir="$PWD/$output_dir"
command_args=(docker run --rm --init --gpus all --shm-size=1g
    --user "$(id -u):$(id -g)" -e HOME=/tmp
    -e "OPENBLAS_NUM_THREADS=${TRAIN_CPU_THREADS:-4}" -e "OMP_NUM_THREADS=${TRAIN_CPU_THREADS:-4}"
    -e "MPLCONFIGDIR=/tmp/matplotlib-$(id -u)"
    -v "$project_dir:/workspace:ro"
    -v "$data_dir:/data/MouseLFP:ro"
    -v "$cache_dir:/cache" -v "$output_dir:/output"
    "$image_name" python train_mouse_lfp.py
    --data-dir /data/MouseLFP --cache-dir /cache --result-dir /output/MouseLFP
    "${training_args[@]}")
if "$dry_run"; then
    printf 'Results: %s/MouseLFP\n' "$output_dir"
    printf '%q ' "${command_args[@]}"
    printf '\n'
    exit 0
fi
mkdir -p -- "$cache_dir" "$output_dir/MouseLFP"
printf 'Results: %s/MouseLFP\n' "$output_dir"
"${command_args[@]}" \
    2>&1 | tee -a "$output_dir/MouseLFP/scan.log"
