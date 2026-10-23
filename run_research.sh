#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
action="${1:-help}"
if [[ "$action" == help || "$action" == --help || "$action" == -h ]]; then
    cat <<'HELP'
Usage: ./run_research.sh [prepare|check|experiment|replay] [OPTIONS]
  prepare     Verify raw recording and annotation identities
  check       Verify fixed controls and run synthetic frontend contract checks
  experiment  Run one candidate and print metrics; the agent logs and keeps/discards it
  replay      Inference from --experiment ID --split query|valid|test

Experiment options: --description TEXT [--resume EXPERIMENT_ID]
                    --timeout-seconds N (default 12000; fixed epoch budget)
Launcher options: --device auto|cuda|cpu --dry-run
Environment: DATA_DIR, PREPARED_DIR, CACHE_DIR, OUTPUT_DIR, IMAGE_NAME,
             TRAIN_CPU_THREADS (default 4)
All operations run in the foreground. Read program.md before starting research.
Each experiment returns to the agent for result review and the next hypothesis.
The agent repeats until stopped and manages results.tsv and Git commits.
HELP
    exit 0
fi
shift
case "$action" in prepare|check|experiment|replay) ;; *) echo "Unknown action: $action" >&2; exit 2 ;; esac
data_dir="${DATA_DIR:-/data/users/jianchen/data/seizure_data}"
cache_dir="${CACHE_DIR:-/data/users/jianchen/data/seizure_data-cache/meta-learning}"
prepared_dir="${PREPARED_DIR:-$cache_dir/prepared}"
output_dir="${OUTPUT_DIR:-$project_dir/autoresearch-runs/meta-frontend}"
image_name="${IMAGE_NAME:-seizure-prediction-cnn:pytorch}"
device=auto
dry_run=false
arguments=()
while (( $# )); do
    case "$1" in
        --dry-run) dry_run=true; shift ;;
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
research_commit=unknown
if command -v git >/dev/null; then
    research_commit="$(git -C "$project_dir" rev-parse HEAD 2>/dev/null || true)"
fi
command_args=(docker run --rm --init --shm-size=1g
    --user "$(id -u):$(id -g)" -e HOME=/tmp
    -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1
    -e "AUTORESEARCH_COMMIT=${research_commit:-unknown}"
    -e "OMP_NUM_THREADS=${TRAIN_CPU_THREADS:-4}" -e "MKL_NUM_THREADS=${TRAIN_CPU_THREADS:-4}"
    -e "OPENBLAS_NUM_THREADS=${TRAIN_CPU_THREADS:-4}" -e MPLCONFIGDIR=/tmp/matplotlib
    -w /workspace -v "$project_dir:/workspace:ro")
if [[ "$action" != check ]]; then
    command_args+=(-v "$data_dir:/data/MouseLFP:ro" -v "$cache_dir:/cache"
                  -v "$prepared_dir:/prepared:$prepared_mode" -v "$output_dir:/output")
fi
if [[ "$action" == experiment || "$action" == replay ]]; then
    if [[ "$device" == cuda || "$device" == cuda:* ]] || { [[ "$device" == auto ]] && command -v nvidia-smi >/dev/null; }; then
        command_args+=(--gpus all)
    fi
fi
command_args+=("$image_name")
case "$action" in
    prepare) command_args+=(python prepare.py --data-dir /data/MouseLFP --output /prepared "${arguments[@]}") ;;
    check) command_args+=(python research.py check "${arguments[@]}") ;;
    *) command_args+=(python research.py "$action" --run-dir /output --data-dir /data/MouseLFP
                      --prepared-dir /prepared --cache-dir /cache --device "$device" "${arguments[@]}") ;;
esac
if "$dry_run"; then
    printf '%q ' "${command_args[@]}"
    printf '\n'
    exit 0
fi
if [[ "$action" != check ]]; then
    mkdir -p -- "$cache_dir" "$prepared_dir" "$output_dir"
fi
exec "${command_args[@]}"
