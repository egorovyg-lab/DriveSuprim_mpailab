#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
root=${DREX_RUNTIME_ROOT:-$HOME/drex_runtime}
stage=${1:-status}; inside=false
if [[ "$stage" == --inside ]]; then inside=true; stage=${2:-}; shift; fi
count=${2:-${DREX_GPUS:-2}}
[[ "$count" =~ ^(1|2|4|8)$ ]] || { echo 'Use 1,2,4 or 8 GPU replicas' >&2; exit 2; }
export DREX_PAIR_DIAG_NAME=${DREX_PAIR_DIAG_NAME:-pair_diag_gpu${count}_v1}
export DREX_PAIR_SOURCE=${DREX_PAIR_SOURCE:-overlay_v2_top4_gpu4_v1}
[[ "$DREX_PAIR_DIAG_NAME" =~ ^pair_diag_[A-Za-z0-9_-]{1,64}$ ]] || exit 2
[[ "$DREX_PAIR_SOURCE" =~ ^overlay_v2_[A-Za-z0-9_-]{1,64}$ ]] || exit 2
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
case "$stage" in plan|submit-plan|submit|worker|merge|check|status) ;; *) echo 'Usage: run_pair_diagnostic.sh check|plan|submit-plan|submit|status [1|2|4|8]' >&2; exit 2 ;; esac
mkdir -p "$root/logs"; root=$(cd "$root" && pwd); export DREX_RUNTIME_ROOT="$root"
out=$root/runs/$DREX_PAIR_DIAG_NAME
if [[ "$stage" == status ]]; then
  [[ ! -f "$out/plan_summary.json" ]] || cat "$out/plan_summary.json"
  [[ ! -f "$out/report.json" ]] || cat "$out/report.json"
  for f in "$out"/shards/*/inference_status.json; do [[ ! -f "$f" ]] || { echo "$f"; cat "$f"; }; done
  exit
fi
if [[ "$inside" == false ]]; then
  [[ -s "$root/setup_node.txt" ]] || { echo 'Run runtime prepare first' >&2; exit 1; }
  node=$(cat "$root/setup_node.txt"); [[ "$node" =~ ^[A-Za-z0-9_.-]+$ ]] || exit 2
  if [[ "$stage" == check || "$stage" == plan ]]; then
    exec srun --partition="${DREX_PARTITION:-batch}" --nodes=1 --ntasks=1 --nodelist="$node" \
      --cpus-per-task=2 --mem=8G --gres=none --time=00:20:00 --export=ALL \
      bash --login "$repo/scripts/drex/run_pair_diagnostic.sh" --inside "$stage" "$count"
  fi
  [[ "$stage" == submit || "$stage" == submit-plan ]] || { echo 'Internal stage' >&2; exit 2; }
  submit_one() {
    local action=$1 dep=$2 raw command memory=8G cpus=2 limit=00:20:00 gres=none id
    local args=(--parsable --partition="${DREX_PARTITION:-batch}" --nodes=1 --ntasks=1 --nodelist="$node" --export=ALL)
    [[ -z "$dep" ]] || args+=(--dependency="afterok:$dep")
    if [[ "$action" == worker ]]; then
      memory=96G; cpus=4; limit=01:00:00; gres=gpu:1
      args+=(--array="0-$((count-1))%$count" --output="$root/logs/pair-diag-worker-%A_%a.log")
    else args+=(--output="$root/logs/pair-diag-$action-%j.log"); fi
    printf -v command '%q ' srun --ntasks=1 --cpus-per-task="$cpus" --gres="$gres" \
      bash --login "$repo/scripts/drex/run_pair_diagnostic.sh" --inside "$action" "$count"
    raw=$(printf '#!/usr/bin/env bash\nset -euo pipefail\nexec %s\n' "$command" | sbatch "${args[@]}" \
      --cpus-per-task="$cpus" --mem="$memory" --gres="$gres" --time="$limit" --job-name="drex-pair-diag-$action")
    id=${raw%%;*}; [[ "$id" =~ ^[0-9]+$ ]] || { echo "Unexpected job id: $raw" >&2; return 1; }; echo "$id"
  }
  plan=$(submit_one plan ''); echo "CPU plan: $plan"
  if [[ "$stage" == submit-plan ]]; then echo "Output: $out"; exit; fi
  workers=$(submit_one worker "$plan"); echo "GPU array: $workers; replicas=$count; one GPU each"
  report=$(submit_one merge "$workers"); echo "CPU merge/report: $report"
  echo "Output: $out"; exit
fi
[[ -n ${SLURM_JOB_ID:-} && -n ${SLURM_STEP_ID:-} ]] || { echo 'Requires an actual Slurm job step' >&2; exit 2; }
venv=$root/venv; upstream=$root/upstream
[[ -x "$venv/bin/python" ]] || exit 1
if [[ "$stage" == check ]]; then
  cd "$repo"; "$venv/bin/python" -m unittest discover -s scripts/drex -p 'test_pair_diagnostic.py' -v; exit
fi
[[ $(git -C "$upstream" rev-parse HEAD) == b932571901f531b2378b356fcab7d176048c2bcb ]] || exit 1
[[ -z $(git -C "$upstream" status --porcelain) ]] || { echo 'Upstream has changes' >&2; exit 1; }
export PYTHONPATH="$upstream${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$out"; args=("$stage" --root "$root" --replicas "$count")
if [[ "$stage" == worker ]]; then
  index=${SLURM_ARRAY_TASK_ID:-}; [[ "$index" =~ ^[0-7]$ && "$index" -lt "$count" ]] || exit 2
  shard=$out/shards/$(printf '%02d' "$index"); mkdir -p "$shard"; exec 9>"$shard/.worker.lock"
  args+=(--index "$index")
else exec 9>"$out/.$stage.lock"; fi
flock -n 9 || { echo "Another pair diagnostic $stage is running" >&2; exit 1; }
"$venv/bin/python" -u "$repo/scripts/drex/pair_diagnostic.py" "${args[@]}"
