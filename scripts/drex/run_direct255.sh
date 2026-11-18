#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-status} == --inside ]]; then
  [[ $# == 6 && -n ${SLURM_JOB_ID:-} ]] || exit 2
  action=$2; repo=$(cd "$3" && pwd); root=$(cd "$4" && pwd); count=$5; name=$6
else
  action=${1:-status}; repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
  root=${DREX_RUNTIME_ROOT:-/scratch/pr0201/drex_runtime}; count=${2:-2}
  name=${DREX_DIRECT255_NAME:-direct255_absolute_gpu${count}_v1}
fi
source "$repo/scripts/drex/slurm_options.sh"
[[ "$count" =~ ^(1|2|4|8)$ && "$name" =~ ^direct255_[A-Za-z0-9_-]{1,64}$ ]] || exit 2
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
out=$root/runs/$name
args=(--root "$root" --name "$name" --count "$count")
case "$action" in
  status)
    echo "Output: $out"
    for f in "$out/pilot_report.json" "$out/report.json" "$out"/workers/*/status.json; do
      [[ ! -f "$f" ]] || cat "$f"
    done; exit ;;
  submit-pilot|submit-full|submit-night)
    phase=pilot; [[ "$action" != submit-full ]] || phase=full
    [[ "$action" != submit-night ]] || phase=night
    limit=01:00:00; [[ "$phase" == pilot ]] || limit=08:00:00
    [[ "$phase" != full || -f "$out/pilot_report.json" ]] || { echo 'Successful pilot required' >&2; exit 1; }
    mkdir -p "$root/logs"
    exec sbatch --partition="${DREX_PARTITION:-batch}" --nodes=1 "${DREX_NODE_ARGS[@]}" --ntasks="$count" \
      --cpus-per-task=4 --mem="$((96*count))G" --gres="gpu:$count" --time="$limit" --export=ALL \
      --chdir="$repo" --job-name="drex-direct255-$phase" --output="$root/logs/direct255-$phase-%j.log" \
      "$repo/scripts/drex/run_direct255.sh" --inside "coordinator-$phase" "$repo" "$root" "$count" "$name" ;;
esac
[[ -n ${SLURM_JOB_ID:-} ]] || exit 2
upstream=$root/upstream
[[ $(git -C "$upstream" rev-parse HEAD) == b932571901f531b2378b356fcab7d176048c2bcb && -z $(git -C "$upstream" status --porcelain) ]] || exit 1
export PYTHONPATH="$upstream${PYTHONPATH:+:$PYTHONPATH}"
python=$root/venv/bin/python
drex_require_python "$root"
case "$action" in
  coordinator-pilot|coordinator-full|coordinator-night)
    phase=${action#coordinator-}; mkdir -p "$out"; exec 9>"$out/.execution.lock"
    flock -n 9 || { echo 'Another direct255 coordinator is active' >&2; exit 1; }
    cd "$repo"
    "$python" -B -m unittest discover -s "$repo/scripts/drex" -p test_direct255.py -v
    "$python" -u "$repo/scripts/drex/direct255.py" plan "${args[@]}"
    phases=("$phase"); [[ "$phase" != night ]] || phases=(pilot full)
    for phase in "${phases[@]}"; do
      export DREX_GROUP_STEP_TOKEN="$(date -u +%s%N)-${BASHPID}"
      srun --nodes=1 --ntasks="$count" --cpus-per-task=4 --gres="gpu:$count" --gpus-per-task=1 \
        --gpu-bind=single:1 --kill-on-bad-exit=1 --output="$root/logs/direct255-worker-$phase-%j-%t.log" \
        bash "$repo/scripts/drex/run_direct255.sh" --inside "worker-$phase" "$repo" "$root" "$count" "$name"
      "$python" -u "$repo/scripts/drex/direct255.py" report "${args[@]}" --phase "$phase"
    done ;;
  worker-pilot|worker-full)
    exec "$python" -u "$repo/scripts/drex/direct255.py" worker "${args[@]}" --phase "${action#worker-}" ;;
  *) echo 'Usage: run_direct255.sh submit-pilot|submit-full|submit-night|status [1|2|4|8]' >&2; exit 2 ;;
esac
