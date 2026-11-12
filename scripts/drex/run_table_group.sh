#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
root=${DREX_RUNTIME_ROOT:-$HOME/drex_runtime}
stage=${1:-status}; inside=false
if [[ "$stage" == --inside ]]; then inside=true; stage=${2:-}; shift; fi
count=${2:-2}
[[ "$count" =~ ^(1|2|4|8)$ ]] || { echo 'Use 1, 2, 4 or 8 GPU ranks' >&2; exit 2; }
export DREX_TABLE_COHORT=${DREX_TABLE_COHORT:-all}
export DREX_TABLE_NAME=${DREX_TABLE_NAME:-table_group_${DREX_TABLE_COHORT}_gpu${count}_v1}
[[ "$DREX_TABLE_NAME" =~ ^table_group_[A-Za-z0-9_-]{1,64}$ ]] || { echo 'Use a NEW table_group_* run name' >&2; exit 2; }
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
case "$stage" in check|submit-bind|submit-pilot|submit-night|submit|status|check-cpu|plan|bind|gpu-pilot|gpu-night|gpu-full|worker|worker-pilot|merge) ;; *) echo 'Usage: run_table_group.sh check|submit-bind|submit-pilot|submit-night|submit|status [1|2|4|8]' >&2; exit 2 ;; esac
mkdir -p "$root/logs"; root=$(cd "$root" && pwd); export DREX_RUNTIME_ROOT="$root"
out=$root/runs/$DREX_TABLE_NAME
if [[ "$stage" == status ]]; then
  echo "Output: $out"
  for f in "$out/group_execution.json" "$out/report.json" "$out"/shards/*/inference_status.json "$out"/shards/*/failure.json; do
    [[ ! -f "$f" ]] || { echo "$f"; cat "$f"; }
  done
  exit
fi
node=$(cat "$root/setup_node.txt"); [[ "$node" =~ ^[A-Za-z0-9_.-]+$ ]] || exit 2
if [[ "$inside" == false ]]; then
  submit_one() {
    local action=$1 dep=$2 raw id command gpu=none memory=16G tasks=1 cpus=2 limit=00:45:00
    local args=(--parsable --partition="${DREX_PARTITION:-batch}" --nodes=1 --nodelist="$node" --export=ALL)
    [[ -z "$dep" ]] || args+=(--dependency="afterok:$dep" --kill-on-invalid-dep=yes)
    if [[ "$action" == gpu-* || "$action" == bind ]]; then
      gpu=gpu:$count; tasks=$count; cpus=4; memory=$((96*count))G; limit=${DREX_TABLE_TIME:-08:00:00}
      [[ "$action" != gpu-pilot ]] || limit=00:45:00
      [[ "$action" != bind ]] || limit=00:10:00
    fi
    args+=(--ntasks="$tasks" --cpus-per-task="$cpus" --mem="$memory" --gres="$gpu" --time="$limit" \
      --job-name="drex-table-group-$action" --output="$root/logs/table-group-$action-%j.log")
    # Batch coordinator launches the srun steps inside this ONE allocation.
    printf -v command '%q ' bash --login "$repo/scripts/drex/run_table_group.sh" --inside "$action" "$count"
    raw=$(printf '#!/usr/bin/env bash\nset -euo pipefail\nexec %s\n' "$command" | sbatch "${args[@]}")
    id=${raw%%;*}; [[ "$id" =~ ^[0-9]+$ ]] || { echo "Bad job ID: $raw" >&2; return 1; }; echo "$id"
  }
  case "$stage" in
    check) id=$(submit_one check-cpu ''); echo "CPU checks: $id" ;;
    submit-bind) id=$(submit_one bind ''); echo "Single allocation GPU binding check: $id; GPUs=$count" ;;
    submit|submit-pilot|submit-night)
      dep=''
      if [[ "$stage" == submit-night ]]; then dep=$(submit_one check-cpu ''); echo "CPU checks: $dep"; fi
      plan=$(submit_one plan "$dep"); echo "CPU plan: $plan"
      action=gpu-full
      [[ "$stage" != submit-pilot ]] || action=gpu-pilot
      [[ "$stage" != submit-night ]] || action=gpu-night
      job=$(submit_one "$action" "$plan"); echo "ONE GPU allocation: $job; GPUs=$count; ranks=$count; RAM=$((96*count))G"
      if [[ "$stage" != submit-pilot ]]; then report=$(submit_one merge "$job"); echo "CPU report: $report"; fi ;;
    *) echo 'Internal stage; use submit-*' >&2; exit 2 ;;
  esac
  echo "Output: $out"; exit
fi
[[ -n ${SLURM_JOB_ID:-} ]] || { echo 'Requires a real Slurm allocation' >&2; exit 2; }
venv=$root/venv; [[ -x "$venv/bin/python" ]] || exit 1
gpu_step() {
  local phase=$1
  export DREX_GROUP_STEP_TOKEN="$(date -u +%s%N)-${BASHPID}"
  srun --nodes=1 --ntasks="$count" --cpus-per-task=4 --gres="gpu:$count" \
    --gpus-per-task=1 --gpu-bind=single:1 --kill-on-bad-exit=1 \
    --output="$root/logs/table-group-$phase-%j-%t.log" \
    --error="$root/logs/table-group-$phase-%j-%t.log" \
    bash --login "$repo/scripts/drex/run_table_group.sh" --inside "$phase" "$count"
}
case "$stage" in
  gpu-night) gpu_step worker-pilot; gpu_step worker; exit ;;
  gpu-pilot) gpu_step worker-pilot; exit ;;
  gpu-full) gpu_step worker; exit ;;
  bind)
    if [[ ! ${SLURM_STEP_ID:-} =~ ^[0-9]+$ ]]; then gpu_step bind; exit; fi ;;
  check-cpu)
    exec srun --nodes=1 --ntasks=1 --cpus-per-task=2 --gres=none --kill-on-bad-exit=1 \
      "$venv/bin/python" -B -m unittest discover -s "$repo/scripts/drex" -p 'test_table*.py' -v ;;
  plan|merge)
    if [[ ! ${SLURM_STEP_ID:-} =~ ^[0-9]+$ ]]; then
      exec srun --nodes=1 --ntasks=1 --cpus-per-task=2 --gres=none --kill-on-bad-exit=1 \
        bash --login "$repo/scripts/drex/run_table_group.sh" --inside "$stage" "$count"
    fi ;;
esac
[[ ${SLURM_STEP_ID:-} =~ ^[0-9]+$ ]] || { echo 'Requires a real numbered Slurm task step' >&2; exit 2; }
upstream=$root/upstream
[[ $(git -C "$upstream" rev-parse HEAD) == b932571901f531b2378b356fcab7d176048c2bcb ]] || exit 1
[[ -z $(git -C "$upstream" status --porcelain) ]] || { echo 'Upstream has changes' >&2; exit 1; }
export PYTHONPATH="$upstream${PYTHONPATH:+:$PYTHONPATH}"
exec "$venv/bin/python" -u "$repo/scripts/drex/table_group.py" "$stage" --root "$root" --replicas "$count"
