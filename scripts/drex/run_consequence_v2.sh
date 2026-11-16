#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-status} == --inside ]]; then
  # sbatch executes a spool copy, so BASH_SOURCE cannot locate the checkout.
  [[ $# == 5 && -n ${SLURM_JOB_ID:-} ]] || { echo 'Missing explicit batch paths/job ID' >&2; exit 2; }
  repo=$(cd "$2" && pwd)
  root=$(cd "$3" && pwd)
  name=$4
  source=$5
else
  repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
  root=${DREX_RUNTIME_ROOT:-/scratch/pr0201/drex_runtime}
  root=$(cd "$root" && pwd)
  name=${DREX_CONSEQUENCE_V2_NAME:-consequence_nested_v2}
  source=$root/runs/${DREX_PROBE_NAME:-rollout_audit_drex_probe_v1}
fi
[[ -f "$repo/scripts/drex/consequence_v2.py" && -d "$repo/scripts/drex" ]] || { echo "Invalid repository: $repo" >&2; exit 2; }
[[ "$name" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
out=$root/runs/$name
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
case "${1:-status}" in
  submit|resume)
    node=$(cat "$root/setup_node.txt"); [[ "$node" =~ ^[A-Za-z0-9_.-]+$ ]] || exit 2
    mkdir -p "$root/logs"
    exec sbatch --partition="${DREX_PARTITION:-batch}" --nodelist="$node" --nodes=1 --ntasks=1 \
      --cpus-per-task=2 --mem=16G --gres=none --time=08:00:00 --export=ALL \
      --job-name=drex-consequence-v2 --chdir="$repo" --output="$root/logs/consequence-v2-%j.log" \
      "$repo/scripts/drex/run_consequence_v2.sh" --inside "$repo" "$root" "$name" "$source" ;;
  --inside)
    [[ -n ${SLURM_JOB_ID:-} ]] || exit 2
    cd "$repo"; mkdir -p "$out"; exec 9>"$out/.execution.lock"
    flock -n 9 || { echo 'Another v2 worker is running' >&2; exit 1; }
    "$root/venv/bin/python" -B -m unittest discover -s scripts/drex -p test_consequence_v2.py -v
    for stage in pilot full; do
      "$root/venv/bin/python" -u "$repo/scripts/drex/consequence_v2.py" --source "$source" --out "$out" --stage "$stage"
    done ;;
  status)
    echo "Output: $out"
    for file in progress.json report.json; do
      [[ ! -f "$out/$file" ]] || python3 - "$out/$file" <<'PY'
import json,sys
r=json.load(open(sys.argv[1])); print(json.dumps({k:v for k,v in r.items() if k not in ('selections','quality','metrics','config','note','primary_contrasts')},indent=2))
PY
    done ;;
  *) echo 'Usage: run_consequence_v2.sh submit|resume|status' >&2; exit 2 ;;
esac
