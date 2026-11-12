#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
root=${DREX_RUNTIME_ROOT:-$HOME/drex_runtime}
stage=${1:-status}; inside=false
if [[ "$stage" == --inside ]]; then inside=true; stage=${2:-}; fi
source_name=${DREX_TABLE_AUDIT_SOURCE:-table_topk_night_gpu8_v1}
audit_name=${DREX_TABLE_AUDIT_NAME:-table_cases_night_v1}
[[ "$source_name" =~ ^table_[A-Za-z0-9_-]{1,80}$ && "$audit_name" =~ ^table_cases_[A-Za-z0-9_-]{1,64}$ ]] || exit 2
[[ "$source_name" != "$audit_name" ]] || exit 2
export DREX_TABLE_AUDIT_SOURCE="$source_name" DREX_TABLE_AUDIT_NAME="$audit_name"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p "$root/logs"; root=$(cd "$root" && pwd); export DREX_RUNTIME_ROOT="$root"
out=$root/runs/$audit_name
if [[ "$stage" == status ]]; then
  if [[ ! -f "$out/report.json" ]]; then echo "Audit report not ready: $out/report.json"; exit; fi
  python3 -B - "$out/report.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
keys=('status','scenes','sessions','source_csv_rows_verified','source_queries_verified','case_counts','image_effect_decomposition','pool_expansion_effects','order_effects','seconds')
print(json.dumps({k:r[k] for k in keys},indent=2))
PY
  exit
fi
case "$stage" in submit|check|run) ;; *) echo 'Usage: run_table_case_audit.sh check|submit|status' >&2; exit 2 ;; esac
node=$(cat "$root/setup_node.txt"); [[ "$node" =~ ^[A-Za-z0-9_.-]+$ ]] || exit 2
if [[ "$inside" == false ]]; then
  if [[ "$stage" == check ]]; then
    exec srun --partition="${DREX_PARTITION:-batch}" --nodes=1 --ntasks=1 --nodelist="$node" \
      --cpus-per-task=2 --mem=16G --gres=none --time=00:15:00 --export=ALL \
      bash --login "$repo/scripts/drex/run_table_case_audit.sh" --inside check
  fi
  [[ "$stage" == submit ]] || exit 2
  printf -v command '%q ' srun --nodes=1 --ntasks=1 --cpus-per-task=2 --gres=none --kill-on-bad-exit=1 \
    bash --login "$repo/scripts/drex/run_table_case_audit.sh" --inside run
  printf '#!/usr/bin/env bash\nset -euo pipefail\nexec %s\n' "$command" | sbatch \
    --partition="${DREX_PARTITION:-batch}" --nodes=1 --ntasks=1 --nodelist="$node" \
    --cpus-per-task=2 --mem=16G --gres=none --time=01:00:00 --export=ALL \
    --job-name=drex-table-case-audit --output="$root/logs/table-case-audit-%j.log"
  echo "Source: $root/runs/$source_name"; echo "Audit output: $out"; exit
fi
[[ -n ${SLURM_JOB_ID:-} && -n ${SLURM_STEP_ID:-} ]] || { echo 'Requires a real Slurm task step' >&2; exit 2; }
venv=$root/venv; [[ -x "$venv/bin/python" ]] || exit 1
cd "$repo"; "$venv/bin/python" -c 'import numpy,PIL'
"$venv/bin/python" -m unittest discover -s scripts/drex -p test_table_cases.py -v
[[ "$stage" != check ]] || exit
mkdir -p "$out"; exec 9>"$out/.audit.lock"
flock -n 9 || { echo 'Another case audit is running' >&2; exit 1; }
exec "$venv/bin/python" -u "$repo/scripts/drex/audit_table_cases.py" \
  --source "$root/runs/$source_name" --out "$out" --max-cases 64
