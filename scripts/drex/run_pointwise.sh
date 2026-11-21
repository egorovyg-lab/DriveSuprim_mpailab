#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-status} == --inside ]]; then
  [[ $# == 6 && -n ${SLURM_JOB_ID:-} ]] || exit 2
  action=$2; repo=$3; root=$4; replicas=$5; name=$6
else
  action=${1:-status}; repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
  root=${DREX_RUNTIME_ROOT:-/scratch/pr0201/drex_runtime}; replicas=${2:-2}
  name=${DREX_POINTWISE_NAME:-pointwise_gpu${replicas}_v1}
fi
source "$repo/scripts/drex/slurm_options.sh"
[[ "$replicas" =~ ^(1|2|4|8)$ && "$name" =~ ^pointwise_[A-Za-z0-9_-]+$ ]] || exit 2
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
launcher="$repo/scripts/drex/run_pointwise.sh"
module="$repo/scripts/drex/pointwise.py"
out="$root/runs/$name"
args=(--root "$root" --out "$out" --replicas "$replicas")

case "$action" in
  status)
    echo "Output: $out"
    for f in "$out/pilot_report.json" "$out/report.json" "$out"/workers/*/status.json; do
      if [[ -f "$f" ]]; then printf '\n%s\n' "${f#"$out/"}"; cat "$f"; fi
    done
    exit ;;
  submit-pilot|submit-night|resume)
    phase=${action#submit-}
    limit=${DREX_POINTWISE_TIME:-08:00:00}
    if [[ "$phase" == pilot ]]; then limit=00:30:00; fi
    if [[ "$phase" == resume ]]; then
      [[ -f "$out/pilot_report.json" ]] || { echo 'Complete the pilot first' >&2; exit 1; }
      phase=full
    fi
    root=$(cd "$root" && pwd)
    (cd "$root" && mkdir -p logs)
    exec sbatch --partition="${DREX_PARTITION:-batch}" --nodes=1 "${DREX_NODE_ARGS[@]}" +      --ntasks="$replicas" --cpus-per-task=4 --mem="$((96*replicas))G" --gres="gpu:$replicas" +      --time="$limit" --export=ALL --chdir="$repo" --job-name="drex-pointwise-$phase" +      --output="$root/logs/pointwise-$phase-%j.log" +      "$launcher" --inside "coordinator-$phase" "$repo" "$root" "$replicas" "$name" ;;
esac
[[ -n ${SLURM_JOB_ID:-} ]] || exit 2
upstream="$root/upstream"
[[ $(git -C "$upstream" rev-parse HEAD) == b932571901f531b2378b356fcab7d176048c2bcb && -z $(git -C "$upstream" status --porcelain) ]] || exit 1
export PYTHONPATH="$upstream${PYTHONPATH:+:$PYTHONPATH}"
drex_require_python "$root"
python="$root/venv/bin/python"
case "$action" in
  coordinator-pilot|coordinator-full|coordinator-night)
    (cd "$root" && mkdir -p "runs/$name")
    exec 9>"$out/.execution.lock"
    flock -n 9 || { echo 'Another pointwise job is active for this run' >&2; exit 1; }
    cd "$repo"
    "$python" -B -m unittest discover -s "$repo/scripts/drex" -p test_pointwise.py -v
    if [[ ! -f "$out/plan.json" ]]; then
      "$python" -u "$module" plan "${args[@]}"
    fi
    phases=("${action#coordinator-}")
    if [[ "$action" == coordinator-night ]]; then phases=(pilot full); fi
    for phase in "${phases[@]}"; do
      export DREX_GROUP_STEP_TOKEN="$(date -u +%s%N)-${BASHPID}"
      srun --nodes=1 --ntasks="$replicas" --cpus-per-task=4 --gres="gpu:$replicas" --gpus-per-task=1 +        --gpu-bind=single:1 --kill-on-bad-exit=1 --output="$root/logs/pointwise-worker-$phase-%j-%t.log" +        bash "$launcher" --inside "worker-$phase" "$repo" "$root" "$replicas" "$name"
      "$python" -u "$module" collect "${args[@]}" --phase "$phase"
    done ;;
  worker-pilot|worker-full)
    exec "$python" -u "$module" worker "${args[@]}" --phase "${action#worker-}" ;;
  *) echo 'Usage: run_pointwise.sh submit-pilot|submit-night|resume|status [1|2|4|8]' >&2; exit 2 ;;
esac
