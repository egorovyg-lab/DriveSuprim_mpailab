#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
root=${DREX_RUNTIME_ROOT:-$HOME/drex_runtime}
stage=${1:-submit}
inside=false
if [[ "$stage" == --inside ]]; then inside=true; stage=${2:-}; shift; fi
count=${2:-${DREX_GPUS:-2}}
[[ "$count" =~ ^[1-8]$ ]] || { echo 'Replica count must be 1..8' >&2; exit 2; }
default_name=real_train64_gpu${count}_v1
if [[ "$stage" == audit ]]; then default_name=real_train64_v1; fi
export DREX_REAL_NAME=${DREX_REAL_NAME:-$default_name}
[[ "$DREX_REAL_NAME" =~ ^[A-Za-z0-9_-]{1,80}$ ]] || { echo 'Invalid output name' >&2; exit 2; }
[[ "$stage" == audit || "$DREX_REAL_NAME" != real_train64_v1 ]] || {
  echo 'Use a separate DREX_REAL_NAME for parallel execution' >&2; exit 2;
}
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
case "$stage" in submit|plan|worker|merge|check|audit) ;; *) echo 'Usage: run_real_parallel.sh submit|plan|merge|check|audit [1..8]' >&2; exit 2 ;; esac
mkdir -p "$root/logs"
root=$(cd "$root" && pwd)
export DREX_RUNTIME_ROOT="$root"
out=$root/runs/$DREX_REAL_NAME
if [[ "$inside" == false ]]; then
  [[ "$stage" != worker ]] || { echo 'Worker is internal to Slurm array' >&2; exit 2; }
  [[ -s "$root/setup_node.txt" ]] || { echo 'Run runtime prepare first' >&2; exit 1; }
  node=$(cat "$root/setup_node.txt")
  [[ "$node" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo 'Invalid saved setup node' >&2; exit 1; }
  if [[ "$stage" != submit ]]; then
    exec srun --partition="${DREX_PARTITION:-batch}" --nodes=1 --ntasks=1 --nodelist="$node" \
      --cpus-per-task=2 --mem=8G --gres=none --time=00:15:00 --export=ALL \
      bash --login "$repo/scripts/drex/run_real_parallel.sh" --inside "$stage" "$count"
  fi
  submit_stage() {
    local which=$1 dependency=$2 worker args memory cpus limit gres raw id
    memory=8G; cpus=2; limit=00:15:00; gres=none
    args=(--parsable --partition="${DREX_PARTITION:-batch}" --nodes=1 --ntasks=1 --nodelist="$node" --export=ALL)
    if [[ -n "$dependency" ]]; then args+=(--dependency="afterok:$dependency"); fi
    if [[ "$which" == worker ]]; then
      memory=96G; cpus=4; limit=04:00:00; gres=gpu:1
      args+=(--array="0-$((count-1))%$count" --output="$root/logs/parallel-worker-%A_%a.log")
    else
      args+=(--output="$root/logs/parallel-$which-%j.log")
    fi
    printf -v worker '%q ' srun --ntasks=1 --cpus-per-task="$cpus" --gres="$gres" \
      bash --login "$repo/scripts/drex/run_real_parallel.sh" --inside "$which" "$count"
    raw=$(printf '#!/usr/bin/env bash\nset -euo pipefail\nexec %s\n' "$worker" | \
      sbatch "${args[@]}" --cpus-per-task="$cpus" --mem="$memory" --gres="$gres" --time="$limit" --job-name="drex-par-$which")
    id=${raw%%;*}
    [[ "$id" =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch job id: $raw" >&2; return 1; }
    printf '%s\n' "$id"
  }
  plan_id=$(submit_stage plan '')
  echo "CPU plan job: $plan_id"
  array_id=$(submit_stage worker "$plan_id")
  echo "GPU array: $array_id; replicas=$count; each task requests 1 GPU, 4 CPU, 96G RAM"
  merge_id=$(submit_stage merge "$array_id")
  echo "CPU merge/report job: $merge_id (after all array tasks succeed)"
  echo "Output: $out"
  exit
fi
[[ -n ${SLURM_JOB_ID:-} && -n ${SLURM_STEP_ID:-} ]] || { echo 'Internal mode requires an actual Slurm step' >&2; exit 2; }
venv=$root/venv; upstream=$root/upstream
[[ -x "$venv/bin/python" ]] || { echo 'Run runtime prepare first' >&2; exit 1; }
[[ $(git -C "$upstream" rev-parse HEAD) == b932571901f531b2378b356fcab7d176048c2bcb ]] || { echo 'Wrong upstream commit' >&2; exit 1; }
[[ -z $(git -C "$upstream" status --porcelain) ]] || { echo 'Upstream has local changes' >&2; exit 1; }
export PYTHONPATH="$upstream${PYTHONPATH:+:$PYTHONPATH}"
echo "Parallel stage=$stage replicas=$count host=$(hostname) job=$SLURM_JOB_ID step=$SLURM_STEP_ID"
if [[ "$stage" == check ]]; then
  cd "$repo"
  "$venv/bin/python" -c 'import numpy, PIL'
  "$venv/bin/python" -m unittest discover -s scripts/drex -p 'test_parallel.py' -v
  "$venv/bin/python" -m unittest discover -s scripts/drex -p 'test_audit.py' -v
  exit
fi
if [[ "$stage" == audit ]]; then
  exec 9>"$root/.real-audit.lock"
  flock -n 9 || { echo 'Another real audit is running' >&2; exit 1; }
  "$venv/bin/python" -u "$repo/scripts/drex/audit_real.py" --root "$root" --run "$out"
  exit
fi
mkdir -p "$out"
args=("$stage" --root "$root" --shards "$count")
if [[ "$stage" == worker ]]; then
  index=${SLURM_ARRAY_TASK_ID:-}
  [[ "$index" =~ ^[0-7]$ && "$index" -lt "$count" ]] || { echo 'Invalid array task index' >&2; exit 2; }
  mkdir -p "$out/shards/$(printf '%02d' "$index")"
  exec 9>"$out/shards/$(printf '%02d' "$index")/.worker.lock"
  args+=(--index "$index")
else
  exec 9>"$out/.$stage.lock"
fi
flock -n 9 || { echo "Another $stage is running for this output" >&2; exit 1; }
"$venv/bin/python" -u "$repo/scripts/drex/real_parallel.py" "${args[@]}"
