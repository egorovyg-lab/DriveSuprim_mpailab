#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source "$repo/scripts/drex/slurm_options.sh"
root=${DREX_RUNTIME_ROOT:-/scratch/pr0201/drex_runtime}
stage=${1:-status}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export DREX_PROBE_NAME=${DREX_PROBE_NAME:-rollout_audit_drex_probe_v1}
[[ "$DREX_PROBE_NAME" =~ ^rollout_audit_[A-Za-z0-9_-]{1,64}$ ]] || exit 2
args=(--root "$root" --name "$DREX_PROBE_NAME"
  --source "${DREX_PROBE_SOURCE:-replacement_gpu2_v1}" --pool "${DREX_PROBE_POOL:-rollout_audit_pool64_v1}"
  --host "${DREX_ASTRA_HOST:-egorov@93.180.62.99}" --port "${DREX_ASTRA_PORT:-27434}"
  --container "${DREX_ASTRA_CONTAINER:-egorov_drivesuprim}" --repo "${DREX_ASTRA_REPO:-/workspace/DriveSuprim}"
  --control "${DREX_TRANSFER_CONTROL:-$HOME/.ssh/drex-astra-transfer.sock}")
case "$stage" in
  connect) exec bash "$repo/scripts/drex/run_transfer.sh" connect ;;
  diagnose)
    exec "${PYTHON:-python3}" -B "$repo/scripts/drex/diagnose_consequence_probe.py" \
      --run "$root/runs/$DREX_PROBE_NAME" ;;
  check|prepare)
    exec srun --partition="${DREX_PARTITION:-batch}" --nodes=1 --ntasks=1 "${DREX_NODE_ARGS[@]}" \
      --cpus-per-task=2 --mem=16G --gres=none --time=02:00:00 --export=ALL \
      bash --login "$repo/scripts/drex/run_consequence_probe.sh" --inside "$stage" ;;
  --inside)
    [[ -n ${SLURM_JOB_ID:-} && -n ${SLURM_STEP_ID:-} ]] || exit 2
    drex_require_python "$root"
    cd "$repo"
    "$root/venv/bin/python" -B -m unittest discover -s scripts/drex -p test_consequence_probe.py -v
    [[ ${2:-} != check ]] || exit 0
    mkdir -p "$root/runs/$DREX_PROBE_NAME"
    exec 9>"$root/runs/$DREX_PROBE_NAME/.prepare.lock"
    flock -n 9 || { echo 'Probe preparation already running' >&2; exit 1; }
    exec "$root/venv/bin/python" -u "$repo/scripts/drex/consequence_cli.py" prepare "${args[@]}" ;;
  stage|start|status|fetch)
    exec "${PYTHON:-python3}" -u "$repo/scripts/drex/consequence_cli.py" "$stage" "${args[@]}" ;;
  *) echo 'Usage: run_consequence_probe.sh check|prepare|connect|stage|start|status|fetch|diagnose' >&2; exit 2 ;;
esac
