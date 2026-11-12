#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
root=${DREX_RUNTIME_ROOT:-$HOME/drex_runtime}
export DREX_TABLE_PAIR_NAME=${DREX_TABLE_PAIR_NAME:-rollout_audit_table_pairs_v1}
export DREX_TABLE_PAIR_SOURCE=${DREX_TABLE_PAIR_SOURCE:-table_topk_night_gpu8_v1}
[[ "$DREX_TABLE_PAIR_NAME" =~ ^rollout_audit_[A-Za-z0-9_-]{1,64}$ ]] || exit 2
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p "$root/logs"; root=$(cd "$root" && pwd); export DREX_RUNTIME_ROOT="$root"
out=$root/runs/$DREX_TABLE_PAIR_NAME
stage=${1:-status}
args=(--root "$root" --name "$DREX_TABLE_PAIR_NAME" --source "$DREX_TABLE_PAIR_SOURCE"
  --host "${DREX_ASTRA_HOST:-egorov@93.180.62.99}" --port "${DREX_ASTRA_PORT:-27434}"
  --container "${DREX_ASTRA_CONTAINER:-egorov_drivesuprim}"
  --control "${DREX_TRANSFER_CONTROL:-$HOME/.ssh/drex-astra-transfer.sock}"
  --remote-repo "${DREX_ASTRA_REPO:-/workspace/DriveSuprim}"
  --remote-package "${DREX_ASTRA_PACKAGE:-/workspace/exp_v2/drex_transfer/navsim_train_v1}")
case "$stage" in
  connect) exec bash "$repo/scripts/drex/run_transfer.sh" connect ;;
  prepare|check)
    node=$(cat "$root/setup_node.txt"); [[ "$node" =~ ^[A-Za-z0-9_.-]+$ ]] || exit 2
    exec srun --partition="${DREX_PARTITION:-batch}" --nodes=1 --ntasks=1 --nodelist="$node" \
      --cpus-per-task=2 --mem=16G --gres=none --time=01:00:00 --export=ALL \
      bash --login "$repo/scripts/drex/run_table_pair_audit.sh" --inside "$stage" ;;
  --inside)
    [[ -n ${SLURM_JOB_ID:-} && -n ${SLURM_STEP_ID:-} ]] || exit 2
    cd "$repo"
    "$root/venv/bin/python" -B -m unittest discover -s scripts/drex -p test_table_factorial.py -v
    [[ ${2:-} != check ]] || exit
    mkdir -p "$out"; exec 9>"$out/.prepare.lock"; flock -n 9 || exit 1
    exec "$root/venv/bin/python" -u "$repo/scripts/drex/table_pair_bridge.py" prepare "${args[@]}" ;;
  start)
    [[ -f "$out/audit_request.json" ]] || { echo 'Run prepare first' >&2; exit 1; }
    ssh -S "${DREX_TRANSFER_CONTROL:-$HOME/.ssh/drex-astra-transfer.sock}" -O check \
      -p "${DREX_ASTRA_PORT:-27434}" "${DREX_ASTRA_HOST:-egorov@93.180.62.99}"
    log=$(mktemp "$root/logs/table-pair-audit-XXXXXXXX.log")
    nohup bash "$repo/scripts/drex/run_table_pair_audit.sh" --worker > "$log" 2>&1 < /dev/null &
    printf '%s\n' "$!" > "$out/bridge.pid"; printf '%s\n' "$log" > "$out/bridge_log.txt"
    echo "Astra CPU replay started: pid=$! log=$log" ;;
  --worker)
    mkdir -p "$out"; exec 9>"$out/.bridge.lock"; flock -n 9 || { echo 'Another bridge is running' >&2; exit 1; }
    exec "${PYTHON:-python3}" -u "$repo/scripts/drex/table_pair_bridge.py" run "${args[@]}" ;;
  status)
    [[ ! -f "$out/bridge_log.txt" ]] || tail -n 40 "$(cat "$out/bridge_log.txt")"
    if [[ -f "$out/latest_result.json" ]]; then
      python3 -B - "$out/latest_result.json" <<'PY'
import json,sys
from pathlib import Path
p=Path(json.load(open(sys.argv[1]))['result'])
print('Artifacts:',p)
print((p/'report.json').read_text())
PY
    else echo "No retrieved report yet: $out"; fi ;;
  *) echo 'Usage: run_table_pair_audit.sh prepare|connect|start|status|check' >&2; exit 2 ;;
esac
