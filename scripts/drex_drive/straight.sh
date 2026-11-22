#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == --worker ]]; then
  runtime=$2
  out=$3
  exec 9>"$out/.predict.lock"
  flock -n 9 || { echo 'Prediction is already running for this run' >&2; exit 1; }
  export PYTHONPATH="$runtime/upstream"
  exec srun --ntasks=1 "$runtime/venv/bin/python" -m scripts.drex_drive.navsim_straight predict \
    --out "$out" --weights "$runtime/models/drex-max-v1"
fi

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo"
runtime=${DREX_RUNTIME_ROOT:-/scratch/pr0201/drex_runtime}
out=$runtime/runs/${DREX_STRAIGHT_NAME:-navsim-straight-train-v1}
control=${DREX_TRANSFER_CONTROL:-$HOME/.ssh/drex-astra-transfer.sock}
host=${DREX_ASTRA_HOST:-egorov@93.180.62.99}
port=${DREX_ASTRA_PORT:-27434}

case "${1:-}" in
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
    exec sbatch --job-name=drex-navsim-straight --partition="${DREX_PARTITION:-batch}" \
      --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=4 --mem=96G --time=00:30:00 \
      --chdir="$repo" --output="$runtime/logs/straight-%j.log" \
      "$repo/scripts/drex_drive/straight.sh" --worker "$runtime" "$out" ;;
  score)
    python3 -m scripts.drex_drive.navsim_straight bridge-score --out "$out" ;;
  *) printf 'Usage: %s connect|prepare|submit|score\n' "$0" >&2; exit 2 ;;
esac
