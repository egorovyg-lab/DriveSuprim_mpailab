#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
root=${DREX_RUNTIME_ROOT:-/scratch/pr0201/drex_runtime}
export DREX_CONTINUOUS_NAME=${DREX_CONTINUOUS_NAME:-rollout_audit_continuous_v1}
export DREX_CONTINUOUS_SOURCE=${DREX_CONTINUOUS_SOURCE:-replacement_gpu2_v1}
[[ "$DREX_CONTINUOUS_NAME" =~ ^rollout_audit_[A-Za-z0-9_-]{1,64}$ ]] || exit 2
[[ "$DREX_CONTINUOUS_SOURCE" =~ ^replacement_[A-Za-z0-9_-]{1,64}$ ]] || exit 2
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p "$root/logs"; root=$(cd "$root" && pwd); export DREX_RUNTIME_ROOT="$root"
out=$root/runs/$DREX_CONTINUOUS_NAME
args=(--root "$root" --name "$DREX_CONTINUOUS_NAME" --source "$DREX_CONTINUOUS_SOURCE"
  --host "${DREX_ASTRA_HOST:-egorov@93.180.62.99}" --port "${DREX_ASTRA_PORT:-27434}"
  --container "${DREX_ASTRA_CONTAINER:-egorov_drivesuprim}"
  --control "${DREX_TRANSFER_CONTROL:-$HOME/.ssh/drex-astra-transfer.sock}"
  --remote-repo "${DREX_ASTRA_REPO:-/workspace/DriveSuprim}"
  --remote-package "${DREX_ASTRA_PACKAGE:-/workspace/exp_v2/drex_transfer/navsim_train_v1}")
stage=${1:-status}
case "$stage" in
  connect) exec bash "$repo/scripts/drex/run_transfer.sh" connect ;;
  check|prepare)
    node=$(cat "$root/setup_node.txt"); [[ "$node" =~ ^[A-Za-z0-9_.-]+$ ]] || exit 2
    exec srun --partition="${DREX_PARTITION:-batch}" --nodes=1 --ntasks=1 --nodelist="$node" \
      --cpus-per-task=2 --mem=16G --gres=none --time=02:00:00 --export=ALL \
      bash --login "$repo/scripts/drex/run_continuous_audit.sh" --inside "$stage" ;;
  --inside)
    [[ -n ${SLURM_JOB_ID:-} && -n ${SLURM_STEP_ID:-} ]] || exit 2
    cd "$repo"
    "$root/venv/bin/python" -B -m unittest discover -s scripts/drex -p test_continuous_audit.py -v
    [[ ${2:-} != check ]] || exit 0
    mkdir -p "$out"; exec 9>"$out/.prepare.lock"; flock -n 9 || { echo 'Preparation already running' >&2; exit 1; }
    exec "$root/venv/bin/python" -u "$repo/scripts/drex/continuous_audit.py" prepare "${args[@]}" ;;
  pilot|start)
    [[ -f "$out/audit_request.json" ]] || { echo 'Run prepare first' >&2; exit 1; }
    ssh -S "${DREX_TRANSFER_CONTROL:-$HOME/.ssh/drex-astra-transfer.sock}" -O check \
      -p "${DREX_ASTRA_PORT:-27434}" "${DREX_ASTRA_HOST:-egorov@93.180.62.99}"
    log=$(mktemp "$root/logs/continuous-audit-XXXXXXXX.log")
    nohup bash "$repo/scripts/drex/run_continuous_audit.sh" --worker "$stage" > "$log" 2>&1 < /dev/null &
    printf '%s\n' "$!" > "$out/bridge.pid"; printf '%s\n' "$log" > "$out/bridge_log.txt"
    echo "Astra CPU audit started: pid=$! log=$log"
    echo 'start runs the small pilot first, then full replay only if the pilot passes. No GPU requested.' ;;
  --worker)
    mkdir -p "$out"; exec 9>"$out/.bridge.lock"
    flock -n 9 || { echo 'Another continuous audit bridge is running' >&2; exit 1; }
    "${PYTHON:-python3}" -u "$repo/scripts/drex/continuous_audit.py" pilot "${args[@]}"
    [[ ${2:-} != start ]] || "${PYTHON:-python3}" -u "$repo/scripts/drex/continuous_audit.py" run "${args[@]}" ;;
  status)
    echo "Output: $out"
    [[ ! -f "$out/bridge_log.txt" ]] || tail -n 30 "$(cat "$out/bridge_log.txt")"
    if [[ -f "$out/latest_result.json" ]]; then
      python3 -B - "$out/latest_result.json" <<'PY'
import json, sys
from pathlib import Path
p = json.load(open(sys.argv[1])); result = Path(p['result'])
r = json.loads((result/'report.json').read_text())
print('Retrieved snapshot:', result)
print(json.dumps({k:v for k,v in r.items() if k != 'comparisons_vs_DS'}, indent=2))
print('Full report:', result/'report.json')
print('A retrieved pilot snapshot can coexist with an ongoing full CPU replay; inspect the log above.')
PY
    else echo 'No retrieved result yet; see log for live parity/progress/ETA.'; fi ;;
  *) echo 'Usage: run_continuous_audit.sh check|prepare|connect|pilot|start|status' >&2; exit 2 ;;
esac
