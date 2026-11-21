#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-status} == --inside ]]; then
  [[ $# == 7 && -n ${SLURM_JOB_ID:-} ]] || exit 2
  phase=$2; repo=$3; root=$4; source=$5; scores=$6; out=$7
else
  repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
  root=${DREX_RUNTIME_ROOT:-/scratch/pr0201/drex_runtime}
  source=$root/runs/${DREX_PROBE_NAME:-rollout_audit_drex_probe_v1}
  scores=$root/runs/${DREX_POINTWISE_NAME:-pointwise_gpu4_v1}
  out=$root/runs/${DREX_POINTWISE_VALUE_NAME:-pointwise_value_v1}
fi

case "${1:-status}" in
  submit-pilot|submit-full)
    phase=${1#submit-}
    source "$repo/scripts/drex/slurm_options.sh"
    for path in "$source/audit_request.json" "$scores/report.json" "$scores/scores"; do
      [[ -e "$path" ]] || { echo "Missing CPU input: $path" >&2; exit 1; }
    done
    if [[ "$phase" == full && ! -f "$out/pilot_report.json" ]]; then
      echo 'Complete the pilot fold first' >&2; exit 1
    fi
    (cd "$root" && mkdir -p logs)
    opts=(
      --partition="${DREX_PARTITION:-batch}" --nodes=1 "${DREX_NODE_ARGS[@]}"
      --ntasks=1 --cpus-per-task=2 --mem=8G --gres=none --time=01:00:00 --export=ALL
      --job-name="drex-value-$phase" --chdir="$repo"
      --output="$root/logs/pointwise-value-$phase-%j.log"
    )
    exec sbatch "${opts[@]}" "$repo/scripts/drex/run_pointwise_value.sh" --inside "$phase" "$repo" "$root" "$source" "$scores" "$out" ;;
  --inside)
    [[ "$phase" == pilot || "$phase" == full ]] || exit 2
    source "$repo/scripts/drex/slurm_options.sh"
    drex_require_python "$root"
    mkdir -p "$out"
    exec 9>"$out/.execution.lock"
    flock -n 9 || { echo 'Another pointwise value job is active' >&2; exit 1; }
    export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
    cd "$repo"
    "$root/venv/bin/python" -B -m unittest discover -s scripts/drex -p test_pointwise_value.py -v
    exec "$root/venv/bin/python" -u scripts/drex/pointwise_value.py --source "$source" --scores "$scores" --out "$out" --stage "$phase" ;;
  status)
    printf 'Output: %s\n' "$out"
    for name in pilot_report.json report.json; do
      file="$out/$name"
      if [[ -f "$file" ]]; then
        python3 -c 'import json, sys; r=json.load(open(sys.argv[1])); print(sys.argv[1], r["status"], "scenes:", r["scenes"], "sessions:", r["sessions"])' "$file"
      fi
    done ;;
  *) echo 'Usage: run_pointwise_value.sh submit-pilot|submit-full|status' >&2; exit 2 ;;
esac
