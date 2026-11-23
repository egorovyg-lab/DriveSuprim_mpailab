#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo"
script="$repo/scripts/drex_drive/night.sh"
module=scripts.drex_drive.navsim_night
runtime=${DREX_RUNTIME_ROOT:-/scratch/pr0201/drex_runtime}
out=$runtime/runs/navsim-straight-night-v1
pilot=$runtime/runs/navsim-straight-train-v1/observations.json
log=$out/night.log

case "${1:-}" in
  start)
    [[ -s "$pilot" ]] || { echo 'Original TRAIN observations are missing' >&2; exit 2; }
    mkdir -p "$out" "$runtime/logs"
    bash "$repo/scripts/drex_drive/straight.sh" connect
    nohup bash "$script" run </dev/null >>"$log" 2>&1 &
    printf 'Night run PID %s\nLog %s\n' "$!" "$log" ;;
  run)
    exec 7>"$out/.run.lock"
    flock -n 7 || { echo 'The nightly comparison is already running' >&2; exit 1; }
    stage=prepare
    trap 'code=$?; if (( code )); then printf "%s failed (%s)\n" "$stage" "$code" >"$out/status.txt"; fi' EXIT
    printf '%s\n' "$stage" >"$out/status.txt"
    python3 -m "$module" prepare --out "$out" --pilot "$pilot"
    if [[ ! -s "$out/gpu-complete.txt" ]]; then
      stage=gpu
      printf '%s\n' "$stage" >"$out/status.txt"
      sbatch --job-name=drex-night --wait --partition="${DREX_PARTITION:-batch}" --nodes=1 --ntasks=1 \
        --gres=gpu:1 --cpus-per-task=4 --mem=96G --time="${DREX_NIGHT_GPU_TIME:-02:00:00}" --chdir="$repo" \
        --output="$runtime/logs/night-%j.log" "$script" worker
      printf 'Predictions complete\n' >"$out/gpu-complete.txt"
    fi
    if [[ ! -s "$out/suite.json" ]]; then
      stage=score
      printf '%s\n' "$stage" >"$out/status.txt"
      python3 -m "$module" score --out "$out"
    fi
    if [[ ! -s "$out/single-edits.json" ]]; then
      stage=probes
      printf '%s\n' "$stage" >"$out/status.txt"
      python3 -m "$module" probes --out "$out"
    fi
    printf 'complete\n' >"$out/status.txt" ;;
  worker)
    exec 8>"$out/.predict.lock"
    flock -n 8 || { echo 'Night predictions are already running' >&2; exit 1; }
    export PYTHONPATH="$runtime/upstream"
    exec srun --ntasks=1 "$runtime/venv/bin/python" -m "$module" predict \
      --out "$out" --weights "$runtime/models/drex-max-v1" ;;
  status)
    if [[ -s "$out/status.txt" ]]; then
      cat "$out/status.txt"
    fi
    squeue -u "$USER" -n drex-night -o "%.18i %.22j %.10T %.10M %R"
    if [[ -s "$log" ]]; then
      tail -n 18 "$log"
    fi ;;
  *) printf 'Usage: %s start|status\n' "$0" >&2; exit 2 ;;
esac
