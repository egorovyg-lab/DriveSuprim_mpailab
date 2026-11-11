#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
root=${DREX_RUNTIME_ROOT:-$HOME/drex_runtime}
stage=${1:-plan}; inside=false
if [[ "$stage" == --inside ]]; then inside=true; stage=${2:-}; shift; fi
count=${2:-${DREX_GPUS:-2}}
[[ "$count" =~ ^(1|2|4|8)$ ]] || { echo 'Use 1, 2, 4 or 8 GPU replicas; default is 2' >&2; exit 2; }
export DREX_OVERLAY_V2_NAME=${DREX_OVERLAY_V2_NAME:-overlay_v2_top4_gpu${count}_v1}
[[ "$DREX_OVERLAY_V2_NAME" =~ ^overlay_v2_[A-Za-z0-9_-]{1,64}$ ]] || { echo 'Use a separate overlay_v2_* output name' >&2; exit 2; }
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
case "$stage" in plan|submit-plan|submit-pilot|submit|worker|worker-pilot|merge|check|status) ;; *) echo 'Usage: run_overlay_v2.sh plan|submit-plan|check|submit-pilot|submit|status [1|2|4|8]' >&2; exit 2 ;; esac
mkdir -p "$root/logs"; root=$(cd "$root" && pwd); export DREX_RUNTIME_ROOT="$root"
out=$root/runs/$DREX_OVERLAY_V2_NAME
if [[ "$stage" == status ]]; then
  [[ ! -f "$out/report.json" ]] || cat "$out/report.json"
  for f in "$out"/shards/*/inference_status.json; do [[ ! -f "$f" ]] || { echo "$f"; cat "$f"; }; done
  exit
fi
if [[ "$inside" == false ]]; then
  [[ -s "$root/setup_node.txt" ]] || { echo 'Run runtime prepare first' >&2; exit 1; }
  node=$(cat "$root/setup_node.txt"); [[ "$node" =~ ^[A-Za-z0-9_.-]+$ ]] || exit 2
  if [[ "$stage" == plan || "$stage" == check ]]; then
    exec srun --partition="${DREX_PARTITION:-batch}" --nodes=1 --ntasks=1 --nodelist="$node" \
      --cpus-per-task=2 --mem=8G --gres=none --time=00:20:00 --export=ALL \
      bash --login "$repo/scripts/drex/run_overlay_v2.sh" --inside "$stage" "$count"
  fi
  [[ "$stage" == submit || "$stage" == submit-pilot || "$stage" == submit-plan ]] || { echo 'Worker/merge stages are internal' >&2; exit 2; }
  submit_one() {
    local action=$1 dep=$2 args memory=8G cpus=2 limit=00:20:00 gres=none command raw id
    args=(--parsable --partition="${DREX_PARTITION:-batch}" --nodes=1 --ntasks=1 --nodelist="$node" --export=ALL)
    [[ -z "$dep" ]] || args+=(--dependency="afterok:$dep")
    if [[ "$action" == worker* ]]; then
      memory=96G; cpus=4; limit=04:00:00; gres=gpu:1
      [[ "$action" != worker-pilot ]] || limit=00:30:00
      args+=(--array="0-$((count-1))%$count" --output="$root/logs/overlay-v2-$action-%A_%a.log")
    else args+=(--output="$root/logs/overlay-v2-$action-%j.log"); fi
    printf -v command '%q ' srun --ntasks=1 --cpus-per-task="$cpus" --gres="$gres" \
      bash --login "$repo/scripts/drex/run_overlay_v2.sh" --inside "$action" "$count"
    raw=$(printf '#!/usr/bin/env bash\nset -euo pipefail\nexec %s\n' "$command" | sbatch "${args[@]}" \
      --cpus-per-task="$cpus" --mem="$memory" --gres="$gres" --time="$limit" --job-name="drex-overlay-v2-$action")
    id=${raw%%;*}; [[ "$id" =~ ^[0-9]+$ ]] || { echo "Unexpected job id: $raw" >&2; return 1; }; echo "$id"
  }
  plan=$(submit_one plan ''); echo "CPU plan job: $plan"
  if [[ "$stage" == submit-plan ]]; then echo "Output: $out"; exit; fi
  action=worker; [[ "$stage" != submit-pilot ]] || action=worker-pilot
  workers=$(submit_one "$action" "$plan"); echo "GPU array: $workers; replicas=$count; 1 GPU per worker"
  if [[ "$stage" == submit ]]; then merge=$(submit_one merge "$workers"); echo "CPU merge/report job: $merge"; fi
  echo "Output: $out"; exit
fi
[[ -n ${SLURM_JOB_ID:-} && -n ${SLURM_STEP_ID:-} ]] || { echo 'Requires an actual Slurm job step' >&2; exit 2; }
venv=$root/venv; upstream=$root/upstream
[[ -x "$venv/bin/python" ]] || exit 1
[[ $(git -C "$upstream" rev-parse HEAD) == b932571901f531b2378b356fcab7d176048c2bcb ]] || exit 1
[[ -z $(git -C "$upstream" status --porcelain) ]] || { echo 'Upstream has changes' >&2; exit 1; }
export PYTHONPATH="$upstream${PYTHONPATH:+:$PYTHONPATH}"
if [[ "$stage" == check ]]; then
  cd "$repo"; "$venv/bin/python" -c 'import numpy,PIL'
  "$venv/bin/python" -m unittest discover -s scripts/drex -p 'test_overlay_v2.py' -v; exit
fi
mkdir -p "$out"; args=("$stage" --root "$root" --replicas "$count")
if [[ "$stage" == worker* ]]; then
  index=${SLURM_ARRAY_TASK_ID:-}; [[ "$index" =~ ^[0-7]$ && "$index" -lt "$count" ]] || exit 2
  shard=$out/shards/$(printf '%02d' "$index"); mkdir -p "$shard"; exec 9>"$shard/.worker.lock"
  args=(worker --root "$root" --replicas "$count" --index "$index")
  [[ "$stage" != worker-pilot ]] || args+=(--pilot)
else exec 9>"$out/.$stage.lock"; fi
flock -n 9 || { echo "Another overlay $stage is running" >&2; exit 1; }
"$venv/bin/python" -u "$repo/scripts/drex/overlay_v2_pilot.py" "${args[@]}"
