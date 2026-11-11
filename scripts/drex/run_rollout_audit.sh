#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
root=${DREX_RUNTIME_ROOT:-$HOME/drex_runtime}
export DREX_ROLLOUT_AUDIT_NAME=${DREX_ROLLOUT_AUDIT_NAME:-rollout_audit_pair6_v1}
export DREX_ROLLOUT_PAIR_SOURCE=${DREX_ROLLOUT_PAIR_SOURCE:-pair_diag_gpu4_v1}
[[ "$DREX_ROLLOUT_AUDIT_NAME" =~ ^rollout_audit_[A-Za-z0-9_-]{1,64}$ ]] || exit 2
[[ "$DREX_ROLLOUT_PAIR_SOURCE" =~ ^pair_diag_[A-Za-z0-9_-]{1,64}$ ]] || exit 2
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$root/logs"; root=$(cd "$root" && pwd); export DREX_RUNTIME_ROOT="$root"
out=$root/runs/$DREX_ROLLOUT_AUDIT_NAME
stage=${1:-status}
case "$stage" in
  connect) exec bash "$repo/scripts/drex/run_transfer.sh" connect ;;
  start)
    mkdir -p "$out"
    control=${DREX_TRANSFER_CONTROL:-$HOME/.ssh/drex-astra-transfer.sock}
    ssh -S "$control" -O check -p "${DREX_ASTRA_PORT:-27434}" "${DREX_ASTRA_HOST:-egorov@93.180.62.99}"
    log=$(mktemp "$root/logs/rollout-audit-XXXXXXXX.log")
    nohup bash "$repo/scripts/drex/run_rollout_audit.sh" --worker > "$log" 2>&1 < /dev/null &
    printf '%s\n' "$!" > "$out/bridge.pid"
    printf '%s\n' "$log" > "$out/bridge_log.txt"
    echo "CPU audit started: pid=$! log=$log"
    ;;
  --worker)
    mkdir -p "$out"; exec 9>"$out/.bridge.lock"
    flock -n 9 || { echo 'Another rollout bridge is running'; exit 1; }
    "${PYTHON:-python3}" -u "$repo/scripts/drex/rollout_bridge.py" \
      --root "$root" --name "$DREX_ROLLOUT_AUDIT_NAME" --pair "$DREX_ROLLOUT_PAIR_SOURCE" \
      --host "${DREX_ASTRA_HOST:-egorov@93.180.62.99}" --port "${DREX_ASTRA_PORT:-27434}" \
      --container "${DREX_ASTRA_CONTAINER:-egorov_drivesuprim}" \
      --control "${DREX_TRANSFER_CONTROL:-$HOME/.ssh/drex-astra-transfer.sock}" \
      --remote-repo "${DREX_ASTRA_REPO:-/workspace/DriveSuprim}" \
      --remote-package "${DREX_ASTRA_PACKAGE:-/workspace/exp_v2/drex_transfer/navsim_train_v1}"
    ;;
  status)
    [[ ! -f "$out/bridge_log.txt" ]] || tail -n 45 "$(cat "$out/bridge_log.txt")"
    if [[ -f "$out/latest_result.json" ]]; then
      "${PYTHON:-python3}" - "$out/latest_result.json" <<'PY'
import json,sys
from pathlib import Path
value=json.loads(Path(sys.argv[1]).read_text())
root=Path(value['result'])
print('Artifacts:',root)
print((root/'report.json').read_text())
PY
    fi
    ;;
  *) echo 'Usage: run_rollout_audit.sh connect|start|status' >&2; exit 2 ;;
esac
