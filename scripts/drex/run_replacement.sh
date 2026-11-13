#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
root=${DREX_RUNTIME_ROOT:-$HOME/drex_runtime}
stage=${1:-status}; inside=false
if [[ "$stage" == --inside ]]; then inside=true; stage=${2:-}; shift; fi
count=${2:-8}
[[ "$count" =~ ^(1|2|4|8)$ ]] || { echo 'Use 1, 2, 4 or 8 GPUs' >&2; exit 2; }
export DREX_REPLACEMENT_NAME=${DREX_REPLACEMENT_NAME:-replacement_gpu${count}_v1}
export DREX_REPLACEMENT_SOURCE=${DREX_REPLACEMENT_SOURCE:-table_topk_night_gpu8_v1}
[[ "$DREX_REPLACEMENT_NAME" =~ ^replacement_[A-Za-z0-9_-]{1,64}$ ]] || exit 2
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p "$root/logs"; root=$(cd "$root" && pwd); export DREX_RUNTIME_ROOT="$root"
out=$root/runs/$DREX_REPLACEMENT_NAME
if [[ "$stage" == status ]]; then
  echo "Output: $out"
  for f in "$out/report.json" "$out/failure.json" "$out"/shards/*/inference_status.json "$out"/shards/*/failure.json; do
    [[ ! -f "$f" ]] || { echo "$f"; cat "$f"; }
  done
  exit
fi
case "$stage" in check|submit-bind|submit-pilot|submit-night|resume|report|check-cpu|plan|gpu-night|gpu-pilot|gpu-full|bind|worker|worker-pilot) ;; *) echo 'Usage: run_replacement.sh check|submit-bind|submit-pilot|submit-night|resume|report|status [1|2|4|8]' >&2; exit 2 ;; esac
node=$(cat "$root/setup_node.txt"); [[ "$node" =~ ^[A-Za-z0-9_.-]+$ ]] || exit 2
if [[ "$inside" == false ]]; then
  submit_one() {
    local action=$1 dep=$2 command raw id tasks=1 cpus=2 memory=16G gpu=none limit=01:00:00
    [[ "$action" != report ]] || limit=02:00:00
    local args=(--parsable --partition="${DREX_PARTITION:-batch}" --nodes=1 --nodelist="$node" --export=ALL)
    [[ -z "$dep" ]] || args+=(--dependency="afterok:$dep" --kill-on-invalid-dep=yes)
    if [[ "$action" == gpu-* || "$action" == bind ]]; then
      tasks=$count; cpus=4; memory=$((96*count))G; gpu=gpu:$count; limit=${DREX_REPLACEMENT_TIME:-08:00:00}
      [[ "$action" != gpu-pilot ]] || limit=00:45:00
      [[ "$action" != bind ]] || limit=00:10:00
    fi
    printf -v command '%q ' bash --login "$repo/scripts/drex/run_replacement.sh" --inside "$action" "$count"
    raw=$(printf '#!/usr/bin/env bash\nset -euo pipefail\nexec %s\n' "$command" | sbatch "${args[@]}" \
      --ntasks="$tasks" --cpus-per-task="$cpus" --mem="$memory" --gres="$gpu" --time="$limit" \
      --job-name="drex-replacement-$action" --output="$root/logs/replacement-$action-%j.log")
    id=${raw%%;*}; [[ "$id" =~ ^[0-9]+$ ]] || return 1; echo "$id"
  }
  case "$stage" in
    check) id=$(submit_one check-cpu ''); echo "CPU checks: $id" ;;
    submit-bind) id=$(submit_one bind ''); echo "GPU binding check: $id; GPUs=$count" ;;
    submit-pilot|submit-night)
      check=$(submit_one check-cpu ''); plan=$(submit_one plan "$check")
      action=gpu-night; [[ "$stage" != submit-pilot ]] || action=gpu-pilot
      job=$(submit_one "$action" "$plan")
      echo "CPU checks: $check; CPU plan: $plan"
      echo "ONE GPU allocation: $job; GPUs=$count; RAM=$((96*count))G"
      if [[ "$stage" == submit-night ]]; then merge=$(submit_one report "$job"); echo "CPU report: $merge"; fi ;;
    resume)
      job=$(submit_one gpu-full ''); merge=$(submit_one report "$job")
      echo "Resume GPU allocation: $job; CPU report: $merge" ;;
    report) job=$(submit_one report ''); echo "CPU report: $job" ;;
    *) echo 'Use a submit-* stage' >&2; exit 2 ;;
  esac
  echo "Output: $out"; exit
fi
[[ -n ${SLURM_JOB_ID:-} ]] || { echo 'Requires Slurm' >&2; exit 2; }
venv=$root/venv; [[ -x "$venv/bin/python" ]] || exit 1
gpu_step() {
  local phase=$1
  export DREX_GROUP_STEP_TOKEN="$(date -u +%s%N)-${BASHPID}"
  srun --nodes=1 --ntasks="$count" --cpus-per-task=4 --gres="gpu:$count" \
    --gpus-per-task=1 --gpu-bind=single:1 --kill-on-bad-exit=1 \
    --output="$root/logs/replacement-$phase-%j-%t.log" --error="$root/logs/replacement-$phase-%j-%t.log" \
    bash --login "$repo/scripts/drex/run_replacement.sh" --inside "$phase" "$count"
}
case "$stage" in
  gpu-night) gpu_step worker-pilot; gpu_step worker; exit ;;
  gpu-pilot) gpu_step worker-pilot; exit ;;
  gpu-full) gpu_step worker; exit ;;
  bind) if [[ ! ${SLURM_STEP_ID:-} =~ ^[0-9]+$ ]]; then gpu_step bind; exit; fi ;;
  check-cpu|plan|report)
    if [[ ! ${SLURM_STEP_ID:-} =~ ^[0-9]+$ ]]; then
      exec srun --nodes=1 --ntasks=1 --cpus-per-task=2 --gres=none --kill-on-bad-exit=1 \
        bash --login "$repo/scripts/drex/run_replacement.sh" --inside "$stage" "$count"
    fi ;;
esac
[[ ${SLURM_STEP_ID:-} =~ ^[0-9]+$ ]] || exit 2
cd "$repo"
if [[ "$stage" == check-cpu ]]; then
  exec "$venv/bin/python" -B -m unittest discover -s scripts/drex -p 'test_replacement.py' -v
fi
upstream=$root/upstream
[[ $(git -C "$upstream" rev-parse HEAD) == b932571901f531b2378b356fcab7d176048c2bcb ]] || exit 1
[[ -z $(git -C "$upstream" status --porcelain) ]] || { echo 'Upstream has changes' >&2; exit 1; }
export PYTHONPATH="$upstream${PYTHONPATH:+:$PYTHONPATH}"
exec "$venv/bin/python" -u "$repo/scripts/drex/replacement.py" "$stage" --root "$root" --replicas "$count"
