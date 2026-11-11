#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
root=${DREX_RUNTIME_ROOT:-$HOME/drex_runtime}
export DREX_CLEARANCE_NAME=${DREX_CLEARANCE_NAME:-rollout_audit_clearance_pair6_v1}
export DREX_CLEARANCE_SOURCE=${DREX_CLEARANCE_SOURCE:-rollout_audit_pair6_v2}
[[ "$DREX_CLEARANCE_NAME" =~ ^rollout_audit_[A-Za-z0-9_-]{1,64}$ && "$DREX_CLEARANCE_SOURCE" =~ ^rollout_audit_[A-Za-z0-9_-]{1,64}$ && "$DREX_CLEARANCE_NAME" != "$DREX_CLEARANCE_SOURCE" ]] || exit 2
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$root/logs"; root=$(cd "$root" && pwd); export DREX_RUNTIME_ROOT="$root"
out=$root/runs/$DREX_CLEARANCE_NAME
case "${1:-status}" in
  check) cd "$repo"; exec "${PYTHON:-python3}" -m unittest discover -s scripts/drex -p 'test_clearance.py' -v ;;
  connect) exec bash "$repo/scripts/drex/run_transfer.sh" connect ;;
  start)
    mkdir -p "$out"
    control=${DREX_TRANSFER_CONTROL:-$HOME/.ssh/drex-astra-transfer.sock}
    ssh -S "$control" -O check -p "${DREX_ASTRA_PORT:-27434}" "${DREX_ASTRA_HOST:-egorov@93.180.62.99}"
    log=$(mktemp "$root/logs/clearance-audit-XXXXXXXX.log")
    nohup bash "$repo/scripts/drex/run_clearance_audit.sh" --worker > "$log" 2>&1 < /dev/null &
    printf '%s\n' "$!" > "$out/bridge.pid"; printf '%s\n' "$log" > "$out/bridge_log.txt"
    echo "CPU clearance audit started: pid=$! log=$log"
    ;;
  --worker)
    mkdir -p "$out"; exec 9>"$out/.bridge.lock"; flock -n 9 || { echo 'Another clearance bridge is running'; exit 1; }
    "${PYTHON:-python3}" -u "$repo/scripts/drex/clearance_bridge.py" --root "$root" \
      --name "$DREX_CLEARANCE_NAME" --source "$DREX_CLEARANCE_SOURCE" \
      --host "${DREX_ASTRA_HOST:-egorov@93.180.62.99}" --port "${DREX_ASTRA_PORT:-27434}" \
      --container "${DREX_ASTRA_CONTAINER:-egorov_drivesuprim}" \
      --control "${DREX_TRANSFER_CONTROL:-$HOME/.ssh/drex-astra-transfer.sock}" \
      --remote-repo "${DREX_ASTRA_REPO:-/workspace/DriveSuprim}"
    ;;
  status)
    [[ ! -f "$out/bridge_log.txt" ]] || tail -n 20 "$(cat "$out/bridge_log.txt")"
    if [[ -f "$out/latest_result.json" ]]; then
      "${PYTHON:-python3}" "$repo/scripts/drex/clearance_summary.py" "$out/latest_result.json"
    fi
    ;;
  *) echo 'Usage: run_clearance_audit.sh check|connect|start|status' >&2; exit 2 ;;
esac
