#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
root=${DREX_RUNTIME_ROOT:-$HOME/drex_runtime}
stage=${1:-status}; inside=false
if [[ "$stage" == --inside ]]; then inside=true; stage=${2:-}; shift; fi
count=${2:-${DREX_GPUS:-2}}
[[ "$count" =~ ^(1|2|4|8)$ ]] || { echo 'Use 1, 2, 4 or 8 replicas' >&2; exit 2; }
export DREX_TABLE_COHORT=${DREX_TABLE_COHORT:-all}
[[ "$DREX_TABLE_COHORT" == all || "$DREX_TABLE_COHORT" == balanced64 ]] || exit 2
export DREX_TABLE_NAME=${DREX_TABLE_NAME:-table_topk_${DREX_TABLE_COHORT}_gpu${count}_v1}
[[ "$DREX_TABLE_NAME" =~ ^table_[A-Za-z0-9_-]{1,80}$ ]] || { echo 'Use a separate table_* run name' >&2; exit 2; }
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
case "$stage" in tests|check|plan|submit-plan|submit-pilot|submit|submit-night|worker|worker-pilot|merge|report|status|submit-inventory|inventory) ;; *) echo 'Usage: run_table.sh tests|check|plan|submit-plan|submit-pilot|submit|submit-night|report|status|submit-inventory [1|2|4|8]' >&2; exit 2 ;; esac
if [[ "$stage" == tests ]]; then
  cd "$repo"; "${PYTHON:-python3}" -c 'import numpy,PIL'
  exec "${PYTHON:-python3}" -m unittest discover -s scripts/drex -p 'test_table.py' -v
fi
mkdir -p "$root/logs"; root=$(cd "$root" && pwd); export DREX_RUNTIME_ROOT="$root"
out=$root/runs/$DREX_TABLE_NAME
if [[ "$stage" == status ]]; then
  if [[ -f "$out/report.json" ]]; then cat "$out/report.json"; fi
  for f in "$out"/shards/*/inference_status.json; do [[ ! -f "$f" ]] || { echo "$f"; cat "$f"; }; done
  for f in "$out/failure.json" "$out"/shards/*/failure.json; do [[ ! -f "$f" ]] || { echo "$f"; cat "$f"; }; done
  if [[ -f "$root/data/astra_trainval_inventory_v1/report.json" ]]; then cat "$root/data/astra_trainval_inventory_v1/report.json"; fi
  echo "Output: $out"; exit
fi
if [[ "$inside" == false ]]; then
  [[ -s "$root/setup_node.txt" ]] || { echo 'Run runtime prepare first' >&2; exit 1; }
  node=$(cat "$root/setup_node.txt"); [[ "$node" =~ ^[A-Za-z0-9_.-]+$ ]] || exit 2
  if [[ "$stage" == check || "$stage" == plan ]]; then
    exec srun --partition="${DREX_PARTITION:-batch}" --nodes=1 --ntasks=1 --nodelist="$node" \
      --cpus-per-task=2 --mem=16G --gres=none --time=00:45:00 --export=ALL \
      bash --login "$repo/scripts/drex/run_table.sh" --inside "$stage" "$count"
  fi
  submit_one() {
    local action=$1 dependency=$2 memory=16G cpus=2 limit=00:45:00 gres=none raw id command
    local args=(--parsable --partition="${DREX_PARTITION:-batch}" --nodes=1 --ntasks=1 --nodelist="$node" --export=ALL)
    [[ -z "$dependency" ]] || args+=(--dependency="afterok:$dependency" --kill-on-invalid-dep=yes)
    if [[ "$action" == worker* ]]; then
      memory=96G; cpus=4; limit=${DREX_TABLE_TIME:-08:00:00}; gres=gpu:1
      [[ "$action" != worker-pilot ]] || limit=00:45:00
      args+=(--array="0-$((count-1))%$count" --output="$root/logs/table-$action-%A_%a.log")
    else
      args+=(--output="$root/logs/table-$action-%j.log")
      [[ "$action" != inventory ]] || limit=02:00:00
    fi
    printf -v command '%q ' srun --ntasks=1 --cpus-per-task="$cpus" --gres="$gres" \
      bash --login "$repo/scripts/drex/run_table.sh" --inside "$action" "$count"
    raw=$(printf '#!/usr/bin/env bash\nset -euo pipefail\nexec %s\n' "$command" | sbatch "${args[@]}" \
      --cpus-per-task="$cpus" --mem="$memory" --gres="$gres" --time="$limit" --job-name="drex-table-$action")
    id=${raw%%;*}; [[ "$id" =~ ^[0-9]+$ ]] || { echo "Unexpected job ID: $raw" >&2; return 1; }; echo "$id"
  }
  case "$stage" in
    submit-inventory) id=$(submit_one inventory ''); echo "CPU inventory job: $id" ;;
    report) id=$(submit_one merge ''); echo "CPU report job: $id" ;;
    submit|submit-plan|submit-pilot|submit-night)
      dependency=''
      if [[ "$stage" == submit-night ]]; then dependency=$(submit_one check ''); echo "CPU checks: $dependency"; fi
      id=$(submit_one plan "$dependency"); echo "CPU plan job: $id"
      if [[ "$stage" != submit-plan ]]; then
        if [[ "$stage" == submit-night ]]; then id=$(submit_one worker-pilot "$id"); echo "GPU mini-pilot array: $id"; fi
        action=worker; [[ "$stage" != submit-pilot ]] || action=worker-pilot
        workers=$(submit_one "$action" "$id"); echo "GPU array: $workers; replicas=$count; one GPU per worker"
        if [[ "$stage" == submit || "$stage" == submit-night ]]; then merged=$(submit_one merge "$workers"); echo "CPU report job: $merged"; fi
      fi ;;
    *) echo 'Internal stage; use submit or submit-inventory' >&2; exit 2 ;;
  esac
  echo "Output: $out"; exit
fi
[[ -n ${SLURM_JOB_ID:-} && -n ${SLURM_STEP_ID:-} ]] || { echo 'Requires an actual Slurm step' >&2; exit 2; }
venv=$root/venv; upstream=$root/upstream
[[ -x "$venv/bin/python" ]] || exit 1
if [[ "$stage" == check ]]; then
  cd "$repo"; "$venv/bin/python" -c 'import numpy,PIL'
  exec "$venv/bin/python" -m unittest discover -s scripts/drex -p 'test_table.py' -v
fi
if [[ "$stage" == inventory ]]; then
  exec 9>"$root/.table_inventory.lock"; flock -n 9 || exit 1
  exec "$venv/bin/python" -u "$repo/scripts/drex/inventory_trainval.py" \
    --raw "${DREX_RAW_DATA:-$root/data/astra_trainval_raw_v1}" \
    --package "${DREX_TABLE_DATA:-$root/data/navsim_train_v1/package}" \
    --transfer-log "$root/logs/astra_trainval_transfer.log" --out "$root/data/astra_trainval_inventory_v1"
fi
[[ $(git -C "$upstream" rev-parse HEAD) == b932571901f531b2378b356fcab7d176048c2bcb ]] || exit 1
[[ -z $(git -C "$upstream" status --porcelain) ]] || { echo 'Upstream has changes' >&2; exit 1; }
export PYTHONPATH="$upstream${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$out"
args=("$stage" --root "$root" --replicas "$count" --cohort "$DREX_TABLE_COHORT")
if [[ "$stage" == worker* ]]; then
  index=${SLURM_ARRAY_TASK_ID:-}; [[ "$index" =~ ^[0-7]$ && "$index" -lt "$count" ]] || exit 2
  shard=$out/shards/$(printf '%02d' "$index"); mkdir -p "$shard"; exec 9>"$shard/.worker.lock"
  args=(worker --root "$root" --replicas "$count" --index "$index" --cohort "$DREX_TABLE_COHORT")
  [[ "$stage" != worker-pilot ]] || args+=(--pilot)
else
  [[ "$stage" != merge ]] || stage=report
  exec 9>"$out/.$stage.lock"
fi
flock -n 9 || { echo "Another table $stage is running" >&2; exit 1; }
exec "$venv/bin/python" -u "$repo/scripts/drex/table_pilot.py" "${args[@]}"
