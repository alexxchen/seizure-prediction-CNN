#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
image_name="${IMAGE_NAME:-seizure-prediction-cnn:pytorch}"
data_dir="${DATA_DIR:-/data/users/jianchen/data/CHBMIT}"
cache_dir="${CACHE_DIR:-/data/users/jianchen/data/CHBMIT-cache}"
output_dir="${OUTPUT_DIR:-$project_dir/docker-output-pytorch}"
mode=cv
targets="${CHBMIT_TARGETS:-20}"
device="${DEVICE:-cuda}"
batch_size=32
num_workers=0
epochs=""
max_folds=""
seed=0
reuse_checkpoint=false
dry_run=false

usage() {
    cat <<'HELP'
Usage: ./run.sh [OPTIONS]

Run PyTorch CHBMIT cross-validation for subject 20 on CUDA by default.
  --mode cv|test     Select cross-validation or test on the last seizure groups
  --targets 20,21    Select downloaded subject IDs
  --device DEVICE   cuda, cuda:INDEX, auto, or cpu (default: cuda)
  --epochs N        Override 50 cv epochs or 100 test epochs
  --batch-size N    Training and inference batch size (default: 32)
  --num-workers N   DataLoader workers (default: 0)
  --seed N          Random seed (default: 0)
  --max-folds N     Limit cross-validation folds for a shorter experiment
  --reuse-checkpoint  Evaluate an existing PyTorch checkpoint
  --dry-run         Print the launch command without starting a container
  -h, --help        Show help

Environment overrides: IMAGE_NAME, DATA_DIR, CACHE_DIR, OUTPUT_DIR,
CHBMIT_TARGETS, DEVICE. DATA_DIR must contain chb20/ and the project CSV files.
GPU execution requires the NVIDIA Container Toolkit. Use --device cpu to
run without GPU access. CUDA requests fail explicitly if CUDA is unavailable.
HELP
}

fail() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

while (( $# )); do
    case "$1" in
        --mode|--targets|--device|--epochs|--batch-size|--num-workers|--seed|--max-folds)
            (( $# >= 2 )) || fail "$1 requires a value"
            case "$1" in
                --mode) mode="$2" ;;
                --targets) targets="$2" ;;
                --device) device="$2" ;;
                --epochs) epochs="$2" ;;
                --batch-size) batch_size="$2" ;;
                --num-workers) num_workers="$2" ;;
                --seed) seed="$2" ;;
                --max-folds) max_folds="$2" ;;
            esac
            shift 2
            ;;
        --reuse-checkpoint) reuse_checkpoint=true; shift ;;
        --dry-run) dry_run=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) fail "Unknown argument: $1 (use --help for usage)" ;;
    esac
done

[[ "$mode" == cv || "$mode" == test ]] || fail '--mode must be cv or test'
[[ "$device" == cpu || "$device" == auto || "$device" =~ ^cuda(:[0-9]+)?$ ]] || fail 'Invalid --device'
[[ "$targets" =~ ^[1-9][0-9]?(,[1-9][0-9]?)*$ ]] || fail '--targets must contain subject IDs, such as 20 or 20,21'
for value in "$batch_size" "${epochs:-1}" "${max_folds:-1}"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || fail 'Epochs, batch size, and fold limit must be positive integers'
done
for value in "$num_workers" "$seed"; do
    [[ "$value" =~ ^[0-9]+$ ]] || fail 'Worker count and seed must be nonnegative integers'
done
[[ -d "$data_dir" ]] || fail "Data directory does not exist: $data_dir"
IFS=',' read -r -a subject_ids <<< "$targets"
for subject_id in "${subject_ids[@]}"; do
    (( subject_id <= 23 )) || fail "Supported CHBMIT subject IDs range from 1 to 23: $subject_id"
    printf -v subject_dir 'chb%02d' "$subject_id"
    [[ -d "$data_dir/$subject_dir" ]] || fail "Subject data has not been downloaded: $data_dir/$subject_dir"
done
for metadata in segmentation.csv seizure_summary.csv special_interictal.csv; do
    [[ -f "$data_dir/$metadata" ]] || fail "Missing project annotation file: $data_dir/$metadata"
done

[[ "$data_dir" == /* ]] || data_dir="$PWD/$data_dir"
[[ "$cache_dir" == /* ]] || cache_dir="$PWD/$cache_dir"
[[ "$output_dir" == /* ]] || output_dir="$PWD/$output_dir"
command_args=(docker run --rm --init --shm-size=1g
    --user "$(id -u):$(id -g)" -e HOME=/tmp
    -e "MPLCONFIGDIR=/tmp/matplotlib-$(id -u)")
if [[ "$device" != cpu ]]; then
    command_args+=(--gpus all)
fi
command_args+=(
    -v "$data_dir:/data/CHBMIT:ro"
    -v "$cache_dir:/cache/CHBMIT"
    -v "$output_dir:/output"
    "$image_name"
    python main.py --mode "$mode" --dataset CHBMIT --targets "$targets"
    --device "$device" --batch-size "$batch_size" --num-workers "$num_workers" --seed "$seed"
    --data-dir /data/CHBMIT --cache-dir /cache/CHBMIT --result-dir /output/CHBMIT
)
[[ -z "$epochs" ]] || command_args+=(--epochs "$epochs")
[[ -z "$max_folds" ]] || command_args+=(--max-folds "$max_folds")
if "$reuse_checkpoint"; then
    command_args+=(--reuse-checkpoint)
fi
if "$dry_run"; then
    printf '%q ' "${command_args[@]}"
    printf '\n'
    exit 0
fi

command -v docker >/dev/null || fail 'Docker was not found'
docker info >/dev/null 2>&1 || fail 'Cannot connect to the Docker daemon'
docker image inspect "$image_name" >/dev/null 2>&1 || fail \
    "Image not found: $image_name. Run docker build -t $image_name . in the project directory first."
mkdir -p -- "$cache_dir" "$output_dir"
printf 'Starting PyTorch CHBMIT: mode=%s, targets=%s, device=%s\n' "$mode" "$targets" "$device"
printf 'Cache: %s\nResults: %s/CHBMIT\n' "$cache_dir" "$output_dir"
exec "${command_args[@]}"
