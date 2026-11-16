#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-status} == --inside ]]; then
  [[ $# == 6 && -n ${SLURM_JOB_ID:-} ]] || exit 2
  action=$2; repo=$(cd "$3" && pwd); root=$(cd "$4" && pwd); count=$5; name=$6
else
  action=${1:-status}; repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
  root=${DREX_RUNTIME_ROOT:-/scratch/pr0201/drex_runtime}; count=${2:-2}
  name=${DREX_DIRECT256_NAME:-direct256_absolute_gpu${count}_v1}
fi
[[ "$count" =~ ^(1|2|4|8)$ && "$name" =~ ^direct256_[A-Za-z0-9_-]{1,64}$ ]] || exit 2
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
out=$root/runs/$name
args=(--root "$root" --name "$name" --count "$count")
case "$action" in
  status)
    echo "Output: $out"
    for f in "$out/pilot_report.json" "$out/report.json" "$out"/workers/*/failure.json; do
      [[ ! -f "$f" ]] || cat "$f"
    done; exit ;;
  submit-pilot|submit-full)
    phase=pilot; [[ "$action" != submit-full ]] || phase=full
    limit=01:00:00; [[ "$phase" != full ]] || limit=08:00:00
    [[ "$phase" != full || -f "$out/pilot_report.json" ]] || { echo 'Successful pilot required' >&2; exit 1; }
    node=$(cat "$root/setup_node.txt"); [[ "$node" =~ ^[A-Za-z0-9_.-]+$ ]] || exit 2
    mkdir -p "$root/logs"
    exec sbatch --partition="${DREX_PARTITION:-batch}" --nodes=1 --nodelist="$node" --ntasks="$count" \
      --cpus-per-task=4 --mem="$((96*count))G" --gres="gpu:$count" --time="$limit" --export=ALL \
      --chdir="$repo" --job-name="drex-direct256-$phase" --output="$root/logs/direct256-$phase-%j.log" \
      "$repo/scripts/drex/run_direct256.sh" --inside "coordinator-$phase" "$repo" "$root" "$count" "$name" ;;
esac
[[ -n ${SLURM_JOB_ID:-} ]] || exit 2
upstream=$root/upstream
[[ $(git -C "$upstream" rev-parse HEAD) == b932571901f531b2378b356fcab7d176048c2bcb && -z $(git -C "$upstream" status --porcelain) ]] || exit 1
export PYTHONPATH="$upstream${PYTHONPATH:+:$PYTHONPATH}"
python=$root/venv/bin/python
case "$action" in
  coordinator-pilot|coordinator-full)
    phase=${action#coordinator-}; mkdir -p "$out"; exec 9>"$out/.execution.lock"
    flock -n 9 || { echo 'Another direct256 coordinator is active' >&2; exit 1; }
    cd "$repo"
    "$python" -B -m unittest discover -s "$repo/scripts/drex" -p test_direct256.py -v
    "$python" -u "$repo/scripts/drex/direct256.py" plan "${args[@]}"
    export DREX_GROUP_STEP_TOKEN="$(date -u +%s%N)-${BASHPID}"
    srun --nodes=1 --ntasks="$count" --cpus-per-task=4 --gres="gpu:$count" --gpus-per-task=1 \
      --gpu-bind=single:1 --kill-on-bad-exit=1 --output="$root/logs/direct256-worker-%j-%t.log" \
      bash "$repo/scripts/drex/run_direct256.sh" --inside "worker-$phase" "$repo" "$root" "$count" "$name"
    "$python" -u "$repo/scripts/drex/direct256.py" report "${args[@]}" --phase "$phase" ;;
  worker-pilot|worker-full)
    exec "$python" -u "$repo/scripts/drex/direct256.py" worker "${args[@]}" --phase "${action#worker-}" ;;
  *) echo 'Usage: run_direct256.sh submit-pilot|submit-full|status [1|2|4|8]' >&2; exit 2 ;;
esac
