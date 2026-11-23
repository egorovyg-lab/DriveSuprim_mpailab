#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == --worker ]]; then
  runtime=$2
  out=$3
  prompt=${4:-baseline}
  exec 9>"$out/.predict.lock"
  flock -n 9 || { echo 'Prediction is already running for this run' >&2; exit 1; }
  export PYTHONPATH="$runtime/upstream"
  exec srun --ntasks=1 "$runtime/venv/bin/python" -m scripts.drex_drive.navsim_straight predict \
    --out "$out" --weights "$runtime/models/drex-max-v1" --prompt-variant "$prompt"
fi

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo"
runtime=${DREX_RUNTIME_ROOT:-/scratch/pr0201/drex_runtime}
out=$runtime/runs/${DREX_STRAIGHT_NAME:-navsim-straight-train-v1}
control=${DREX_TRANSFER_CONTROL:-$HOME/.ssh/drex-astra-transfer.sock}
host=${DREX_ASTRA_HOST:-egorov@93.180.62.99}
port=${DREX_ASTRA_PORT:-27434}

case "${1:-}" in
  progress-test)
    export DREX_STRAIGHT_NAME=${DREX_STRAIGHT_NAME:-navsim-straight-progress-v1}
    export DREX_STRAIGHT_PROMPT=progress
    out=$runtime/runs/$DREX_STRAIGHT_NAME
    original=$runtime/runs/navsim-straight-train-v1/observations.json
    [[ -s "$original" ]] || { echo 'Original straight-scene observations are missing' >&2; exit 2; }
    mkdir -p "$out"
    exec 8>"$out/.progress-test.lock"
    flock -n 8 || { echo 'The prompt comparison is already running' >&2; exit 1; }
    if [[ ! -e "$out/observations.json" ]]; then
      cp -- "$original" "$out/observations.json"
    fi
    cmp -s "$original" "$out/observations.json" || { echo 'Original observations changed' >&2; exit 2; }
    script="$repo/scripts/drex_drive/straight.sh"
    bash "$script" connect
    if [[ ! -s "$out/predictions.json" ]]; then
      if ! bash "$script" submit --wait; then
        [[ -s "$out/predictions.json" ]] || exit 1
      fi
    fi
    bash "$script" score ;;
  connect)
    if ! ssh -S "$control" -O check -p "$port" -- "$host"; then
      ssh -M -S "$control" -o ControlPersist=12h -o ServerAliveInterval=30 \
        -o ServerAliveCountMax=6 -fNT -p "$port" -- "$host"
    fi ;;
  prepare)
    python3 -m scripts.drex_drive.navsim_straight bridge-prepare --out "$out" --limit "${DREX_STRAIGHT_SCENES:-8}" ;;
  submit)
    [[ -s "$out/observations.json" ]] || { echo 'Run prepare first' >&2; exit 2; }
    [[ ! -e "$out/predictions.json" ]] || { echo 'This run already has predictions' >&2; exit 2; }
    mkdir -p "$runtime/logs"
    wait_args=()
    if [[ ${2:-} == --wait ]]; then
      wait_args=(--wait)
    fi
    exec sbatch --job-name=drex-navsim-straight --partition="${DREX_PARTITION:-batch}" \
      --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=4 --mem=96G --time=00:30:00 \
      --chdir="$repo" --output="$runtime/logs/straight-%j.log" \
      "${wait_args[@]}" "$repo/scripts/drex_drive/straight.sh" --worker "$runtime" "$out" \
      "${DREX_STRAIGHT_PROMPT:-baseline}" ;;
  score)
    python3 -m scripts.drex_drive.navsim_straight bridge-score --out "$out" ;;
  *) printf 'Usage: %s connect|prepare|submit|score|progress-test\n' "$0" >&2; exit 2 ;;
esac
