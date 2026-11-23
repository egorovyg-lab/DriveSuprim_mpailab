#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == --suite-worker ]]; then
  runtime=$2
  out=$3
  exec 9>"$out/.predict.lock"
  flock -n 9 || { echo 'This suite is already running' >&2; exit 1; }
  export PYTHONPATH="$runtime/upstream"
  exec srun --ntasks=1 "$runtime/venv/bin/python" -m scripts.drex_drive.navsim_straight predict-suite \
    --out "$out" --weights "$runtime/models/drex-max-v1"
fi
if [[ ${1:-} == --worker ]]; then
  runtime=$2
  out=$3
  prompt=${4:-baseline}
  observation=${5:-radar}
  exec 9>"$out/.predict.lock"
  flock -n 9 || { echo 'Prediction is already running for this run' >&2; exit 1; }
  export PYTHONPATH="$runtime/upstream"
  exec srun --ntasks=1 "$runtime/venv/bin/python" -m scripts.drex_drive.navsim_straight predict \
    --out "$out" --weights "$runtime/models/drex-max-v1" --prompt-variant "$prompt" \
    --observation-variant "$observation"
fi

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo"
runtime=${DREX_RUNTIME_ROOT:-/scratch/pr0201/drex_runtime}
out=$runtime/runs/${DREX_STRAIGHT_NAME:-navsim-straight-train-v1}
control=${DREX_TRANSFER_CONTROL:-$HOME/.ssh/drex-astra-transfer.sock}
host=${DREX_ASTRA_HOST:-egorov@93.180.62.99}
port=${DREX_ASTRA_PORT:-27434}
slurm_args=(--partition="${DREX_PARTITION:-batch}" --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=4
            --mem=96G --time=00:30:00 --chdir="$repo" --output="$runtime/logs/straight-%j.log")

case "${1:-}" in
  next-tests)
    out=$runtime/runs/${DREX_STRAIGHT_NAME:-navsim-straight-extra-inputs-v1}
    original=$runtime/runs/navsim-straight-train-v1/observations.json
    reference=$runtime/runs/navsim-straight-gap-forecast-v1/report.json
    [[ -s "$original" && -s "$reference" ]] || { echo 'Original observations or gap-forecast report missing' >&2; exit 2; }
    mkdir -p "$out" "$runtime/logs"
    exec 8>"$out/.comparison.lock"
    flock -n 8 || { echo 'This comparison is already running' >&2; exit 1; }
    variants=(leader-history action-effects)
    missing=0
    for variant in "${variants[@]}"; do
      run=$out/$variant
      mkdir -p "$run"
      if [[ ! -e "$run/observations.json" ]]; then
        cp -- "$original" "$run/observations.json"
      fi
      cmp -s "$original" "$run/observations.json" || { echo 'Original observations changed' >&2; exit 2; }
      if [[ ! -s "$run/predictions.json" ]]; then
        missing=1
      fi
    done
    script="$repo/scripts/drex_drive/straight.sh"
    bash "$script" connect
    module=scripts.drex_drive.navsim_straight
    python3 -m "$module" bridge-history --out "$out/leader-history"
    if (( missing )); then
      if ! sbatch --job-name=drex-navsim-inputs --wait "${slurm_args[@]}" "$script" --suite-worker "$runtime" "$out"; then
        echo 'Slurm returned a nonzero status. Checking available predictions.' >&2
      fi
    fi
    for variant in "${variants[@]}"; do
      run=$out/$variant
      [[ -s "$run/predictions.json" ]] || { echo "Missing predictions in $run" >&2; exit 1; }
      if ! python3 -m "$module" bridge-score --out "$run" --prompt-variant baseline --observation-variant "$variant"; then
        [[ -s "$run/report.json" ]] || exit 1
      fi
    done
    python3 -m "$module" summarize-inputs --out "$out" --reference-report "$reference" ;;
  progress-test|forecast-test)
    if [[ $1 == forecast-test ]]; then
      export DREX_STRAIGHT_NAME=${DREX_STRAIGHT_NAME:-navsim-straight-gap-forecast-v1}
      export DREX_STRAIGHT_PROMPT=baseline DREX_STRAIGHT_OBSERVATION=gap-forecast
    else
      export DREX_STRAIGHT_NAME=${DREX_STRAIGHT_NAME:-navsim-straight-progress-v1}
      export DREX_STRAIGHT_PROMPT=progress DREX_STRAIGHT_OBSERVATION=radar
    fi
    out=$runtime/runs/$DREX_STRAIGHT_NAME
    original=$runtime/runs/navsim-straight-train-v1/observations.json
    [[ -s "$original" ]] || { echo 'Original straight-scene observations are missing' >&2; exit 2; }
    mkdir -p "$out"
    exec 8>"$out/.comparison.lock"
    flock -n 8 || { echo 'This comparison is already running' >&2; exit 1; }
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
    exec sbatch --job-name=drex-navsim-straight "${slurm_args[@]}" "${wait_args[@]}" \
      "$repo/scripts/drex_drive/straight.sh" --worker "$runtime" "$out" \
      "${DREX_STRAIGHT_PROMPT:-baseline}" "${DREX_STRAIGHT_OBSERVATION:-radar}" ;;
  score)
    score_args=(--out "$out")
    if [[ -n ${DREX_STRAIGHT_PROMPT:-} ]]; then
      score_args+=(--prompt-variant "$DREX_STRAIGHT_PROMPT")
    fi
    if [[ -n ${DREX_STRAIGHT_OBSERVATION:-} ]]; then
      score_args+=(--observation-variant "$DREX_STRAIGHT_OBSERVATION")
    fi
    python3 -m scripts.drex_drive.navsim_straight bridge-score "${score_args[@]}" ;;
  *) printf 'Usage: %s connect|prepare|submit|score|progress-test|forecast-test|next-tests\n' "$0" >&2; exit 2 ;;
esac
