#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
root=${DREX_RUNTIME_ROOT:-/scratch/pr0201/drex_runtime}
export DREX_POOL_NAME=${DREX_POOL_NAME:-rollout_audit_pool64_v1}
export DREX_POOL_SOURCE=${DREX_POOL_SOURCE:-table_topk_night_gpu8_v1}
export DREX_POOL_REUSE=${DREX_POOL_REUSE:-rollout_audit_continuous_v1}
[[ "$DREX_POOL_NAME" =~ ^rollout_audit_[A-Za-z0-9_-]{1,64}$ && "$DREX_POOL_NAME" != "$DREX_POOL_REUSE" ]] || exit 2
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p "$root/logs"; root=$(cd "$root" && pwd); export DREX_RUNTIME_ROOT="$root"
out=$root/runs/$DREX_POOL_NAME
args=(--root "$root" --name "$DREX_POOL_NAME" --source "$DREX_POOL_SOURCE" --reuse "$DREX_POOL_REUSE"
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
      bash --login "$repo/scripts/drex/run_pool_audit.sh" --inside "$stage" ;;
  --inside)
    [[ -n ${SLURM_JOB_ID:-} && -n ${SLURM_STEP_ID:-} ]] || exit 2
    cd "$repo"
    "$root/venv/bin/python" -B -m unittest discover -s scripts/drex -p test_pool_audit.py -v
    [[ ${2:-} != check ]] || exit 0
    mkdir -p "$out"; exec 9>"$out/.prepare.lock"; flock -n 9 || exit 1
    exec "$root/venv/bin/python" -u "$repo/scripts/drex/pool_audit.py" prepare "${args[@]}" ;;
  pilot|start)
    [[ -f "$out/audit_request.json" ]] || { echo 'Run prepare first' >&2; exit 1; }
    ssh -S "${DREX_TRANSFER_CONTROL:-$HOME/.ssh/drex-astra-transfer.sock}" -O check \
      -p "${DREX_ASTRA_PORT:-27434}" "${DREX_ASTRA_HOST:-egorov@93.180.62.99}"
    log=$(mktemp "$root/logs/pool-audit-XXXXXXXX.log")
    nohup bash "$repo/scripts/drex/run_pool_audit.sh" --worker "$stage" > "$log" 2>&1 < /dev/null &
    printf '%s\n' "$!" > "$out/bridge.pid"; printf '%s\n' "$log" > "$out/bridge_log.txt"
    echo "Astra CPU top64 audit: pid=$! log=$log"
    echo 'Pilot first; full replay only after success. No GPU, fixed seven-hour budget or model calls.' ;;
  --worker)
    mkdir -p "$out"; exec 9>"$out/.bridge.lock"; flock -n 9 || { echo 'Another pool bridge is running' >&2; exit 1; }
    "${PYTHON:-python3}" -u "$repo/scripts/drex/pool_audit.py" pilot "${args[@]}"
    [[ ${2:-} != start ]] || "${PYTHON:-python3}" -u "$repo/scripts/drex/pool_audit.py" run "${args[@]}" ;;
  status)
    echo "Output: $out"
    [[ ! -f "$out/bridge_log.txt" ]] || tail -n 22 "$(cat "$out/bridge_log.txt")"
    if [[ -f "$out/latest_result.json" ]]; then
      python3 -B - "$out/latest_result.json" <<'PY'
import json,sys
from pathlib import Path
p=json.load(open(sys.argv[1])); result=Path(p['result']); r=json.loads((result/'report.json').read_text())
print('Retrieved snapshot:',result)
print(json.dumps({k:v for k,v in r.items() if k not in ('oracles','config','note','statistics_note')},indent=2))
print('Full report:',result/'report.json')
print('A pilot snapshot may remain while full replay runs; live progress/ETA is in the log above.')
PY
    else echo 'No retrieved report yet; see bridge log.'; fi ;;
  *) echo 'Usage: run_pool_audit.sh check|prepare|connect|pilot|start|status' >&2; exit 2 ;;
esac
