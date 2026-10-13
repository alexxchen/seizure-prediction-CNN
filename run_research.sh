#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
data_dir="${DATA_DIR:-/data/users/jianchen/data/seizure_data}"
prepared_dir="${PREPARED_DIR:-/data/users/jianchen/data/seizure_data-cache/autoresearch-raw-v1}"
run_dir="${OUTPUT_DIR:-$project_dir/autoresearch-runs/default}"
image_name="${IMAGE_NAME:-seizure-prediction-cnn:pytorch}"
action=experiment
device=cuda
dry_run=false
arguments=()

if [[ ${1:-} == prepare || ${1:-} == experiment || ${1:-} == finalize || ${1:-} == restore ]]; then
    action="$1"
    shift
fi
while (( $# )); do
    case "$1" in
        -h|--help)
            cat <<'HELP'
Usage: ./run_research.sh [prepare|experiment|restore|finalize] [OPTIONS]
  prepare      Prepare the fixed raw windows once (default: all seven mice)
  experiment   Run one recipe and compare validation accuracy (default action)
  restore      Restore the best recipe, backing up current training/model code
  finalize     Evaluate selected checkpoints on the fixed test set

Experiment options: --targets X1,X2,... --device cuda|cpu|auto
                    --budget-seconds 300 --seed 0 --description TEXT
Prepare options:    --targets X1,X2,...
Launcher option:    --dry-run
Environment: DATA_DIR, PREPARED_DIR, OUTPUT_DIR, IMAGE_NAME, TRAIN_CPU_THREADS
Read program.md to start an AI-driven research loop.
HELP
            exit 0 ;;
        --dry-run) dry_run=true; shift ;;
        --device)
            (( $# >= 2 )) || { echo '--device requires a value' >&2; exit 1; }
            device="$2"; arguments+=("$1" "$2"); shift 2 ;;
        *) arguments+=("$1"); shift ;;
    esac
done
[[ "$data_dir" == /* ]] || data_dir="$PWD/$data_dir"
[[ "$prepared_dir" == /* ]] || prepared_dir="$PWD/$prepared_dir"
[[ "$run_dir" == /* ]] || run_dir="$PWD/$run_dir"
if [[ "$action" == finalize && -f "$run_dir/session.json" ]]; then
    device="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["device"])' "$run_dir/session.json")"
fi
project_mode=ro
prepared_mode=ro
[[ "$action" != restore ]] || project_mode=rw
[[ "$action" != prepare ]] || prepared_mode=rw
research_commit=unknown
if command -v git >/dev/null; then
    research_commit="$(git -C "$project_dir" rev-parse --short HEAD 2>/dev/null || true)"
fi
command_args=(docker run --rm --init --shm-size=1g
    --user "$(id -u):$(id -g)" -e HOME=/tmp
    -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1
    -e "AUTORESEARCH_COMMIT=${research_commit:-unknown}"
    -e "OPENBLAS_NUM_THREADS=${TRAIN_CPU_THREADS:-4}"
    -e "OMP_NUM_THREADS=${TRAIN_CPU_THREADS:-4}" -e "MKL_NUM_THREADS=${TRAIN_CPU_THREADS:-4}"
    -e "MPLCONFIGDIR=/tmp/matplotlib-$(id -u)"
    -w /workspace -v "$project_dir:/workspace:$project_mode"
    -v "$prepared_dir:/prepared:$prepared_mode" -v "$run_dir:/output")
if [[ "$action" != prepare && "$action" != restore && "$device" != cpu ]]; then
    command_args+=(--gpus all)
fi
if [[ "$action" == prepare ]]; then
    command_args+=(-v "$data_dir:/data/MouseLFP:ro" "$image_name"
        python prepare.py --data-dir /data/MouseLFP --output /prepared "${arguments[@]}")
else
    command_args+=("$image_name" python evaluate.py "$action"
        --prepared-dir /prepared --run-dir /output "${arguments[@]}")
fi
if "$dry_run"; then
    printf '%q ' "${command_args[@]}"
    printf '\n'
    exit 0
fi
if [[ "$action" == prepare ]]; then
    [[ -d "$data_dir" ]] || { echo "Data directory missing: $data_dir" >&2; exit 1; }
    mkdir -p -- "$prepared_dir"
else
    [[ -f "$prepared_dir/manifest.json" ]] || { echo 'Prepare raw windows first: ./run_research.sh prepare' >&2; exit 1; }
fi
mkdir -p -- "$run_dir"
exec "${command_args[@]}"
