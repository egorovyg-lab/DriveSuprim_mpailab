#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == --job ]]; then
  runtime=$2
  shift 2
  export PYTHONPATH="$runtime/upstream"
  exec srun --ntasks=1 "$runtime/venv/bin/python" -m scripts.drex_drive \
    --mode realtime --model "$runtime/models/drex-max-v1" \
    --out "$runtime/runs/drive-rt-$SLURM_JOB_ID" "$@"
fi

variant=${1:-warm}
if (( $# )); then
  shift
fi
case "$variant" in
  warm) args=(--warmup) ;;
  cold) args=() ;;
  suite) args=(--warmup --suite) ;;
  *) printf 'Usage: %s [warm|cold|suite] [route|red_light|lead_brakes] [OPTIONS]\n' "$0" >&2; exit 2 ;;
esac
if [[ $variant != suite ]]; then
  scenario=${1:-route}
  case "$scenario" in
    route|red_light|lead_brakes) ;;
    *) printf 'Unknown scenario: %s\n' "$scenario" >&2; exit 2 ;;
  esac
  args+=(--scenario "$scenario")
  if (( $# )); then
    shift
  fi
fi

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
runtime=${DREX_RUNTIME_ROOT:-/scratch/pr0201/drex_runtime}
mkdir -p "$runtime/logs"
runtime=$(cd "$runtime" && pwd)

exec sbatch --job-name="drex-rt-$variant" --partition="${DREX_PARTITION:-batch}" \
  --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=4 --mem=96G --time=00:30:00 \
  --chdir="$repo" --output="$runtime/logs/drive-rt-%j.log" \
  "$repo/scripts/drex_drive/submit.sh" --job "$runtime" "${args[@]}" "$@"
