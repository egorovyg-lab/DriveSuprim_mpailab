#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-status} == --inside ]]; then
  [[ $# == 6 && -n ${SLURM_JOB_ID:-} ]] || exit 2
  action=$2; repo=$(cd "$3" && pwd); root=$(cd "$4" && pwd); replicas=$5; name=$6
else
  action=${1:-status}; repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
  root=${DREX_RUNTIME_ROOT:-/scratch/pr0201/drex_runtime}; replicas=${2:-2}
  name=${DREX_REASON_NAME:-reason_replacement_gpu${replicas}_v1}
fi
source "$repo/scripts/drex/slurm_options.sh"
[[ "$replicas" =~ ^(1|2|4|8)$ && "$name" =~ ^reason_[A-Za-z0-9_-]+$ ]] || exit 2
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
launcher="$repo/scripts/drex/run_reason_replacement.sh"
module="$repo/scripts/drex/reason_replacement.py"
out="$root/runs/$name"
args=(--root "$root" --out "$out" --replicas "$replicas")

case "$action" in
  status)
    echo "Output: $out"
    for f in "$out/pilot_report.json" "$out/report.json" "$out"/workers/*/status.json; do
      [[ ! -f "$f" ]] || cat "$f"
    done
    exit ;;
  submit-pilot|submit-full|submit-night)
    phase=${action#submit-}
    limit=02:00:00
    [[ "$phase" != pilot ]] || limit=00:30:00
    [[ "$phase" != full || -f "$out/pilot_report.json" ]] || { echo 'Run the pilot first' >&2; exit 1; }
    mkdir -p "$root/logs"
    exec sbatch --partition="${DREX_PARTITION:-batch}" --nodes=1 "${DREX_NODE_ARGS[@]}" \
      --ntasks="$replicas" --cpus-per-task=4 --mem="$((96*replicas))G" --gres="gpu:$replicas" \
      --time="$limit" --export=ALL --chdir="$repo" --job-name="drex-reason-$phase" \
      --output="$root/logs/reason-$phase-%j.log" \
      "$launcher" --inside "coordinator-$phase" "$repo" "$root" "$replicas" "$name" ;;
esac

[[ -n ${SLURM_JOB_ID:-} ]] || exit 2
upstream="$root/upstream"
[[ $(git -C "$upstream" rev-parse HEAD) == b932571901f531b2378b356fcab7d176048c2bcb && -z $(git -C "$upstream" status --porcelain) ]] || exit 1
export PYTHONPATH="$upstream${PYTHONPATH:+:$PYTHONPATH}"
python="$root/venv/bin/python"
drex_require_python "$root"
case "$action" in
  coordinator-pilot|coordinator-full|coordinator-night)
    mkdir -p "$out"
    exec 9>"$out/.execution.lock"
    flock -n 9 || { echo 'Another reason experiment is active in this directory' >&2; exit 1; }
    cd "$repo"
    "$python" -B -m unittest discover -s "$repo/scripts/drex" -p test_reason_replacement.py -v
    "$python" -u "$module" plan "${args[@]}"
    phases=("${action#coordinator-}")
    [[ "$action" != coordinator-night ]] || phases=(pilot full)
    for phase in "${phases[@]}"; do
      export DREX_GROUP_STEP_TOKEN="$(date -u +%s%N)-${BASHPID}"
      srun --nodes=1 --ntasks="$replicas" --cpus-per-task=4 --gres="gpu:$replicas" --gpus-per-task=1 \
        --gpu-bind=single:1 --kill-on-bad-exit=1 --output="$root/logs/reason-worker-$phase-%j-%t.log" \
        bash "$launcher" --inside "worker-$phase" "$repo" "$root" "$replicas" "$name"
      "$python" -u "$module" collect "${args[@]}" --phase "$phase"
    done ;;
  worker-pilot|worker-full)
    exec "$python" -u "$module" worker "${args[@]}" --phase "${action#worker-}" ;;
  *) echo 'Usage: run_reason_replacement.sh submit-pilot|submit-full|submit-night|status [1|2|4|8]' >&2; exit 2 ;;
esac
