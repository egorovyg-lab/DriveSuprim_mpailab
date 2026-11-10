#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
root=${DREX_RUNTIME_ROOT:-$HOME/drex_runtime}
control=${DREX_TRANSFER_CONTROL:-$HOME/.ssh/drex-astra-transfer.sock}
host=${DREX_ASTRA_HOST:-egorov@93.180.62.99}
port=${DREX_ASTRA_PORT:-27434}
container=${DREX_ASTRA_CONTAINER:-egorov_drivesuprim}
dest=$root/data/navsim_train_v1
stage=${1:-status}
export PYTHONDONTWRITEBYTECODE=1
case "$stage" in
  connect)
    mkdir -p "$(dirname "$control")"
    if ssh -S "$control" -O check -p "$port" "$host" 2>/dev/null; then
      echo 'Transfer SSH connection is already open'; exit
    fi
    echo 'Authenticate to Astra in this terminal. No password is stored by the script.'
    ssh -M -S "$control" -o ControlPersist=12h -o ServerAliveInterval=30 \
      -o ServerAliveCountMax=6 -fNT -p "$port" "$host"
    ;;
  start)
    ssh -S "$control" -O check -p "$port" "$host"
    mkdir -p "$root/logs"
    log=$root/logs/transfer-$(date -u +%Y%m%dT%H%M%SZ).log
    nohup bash "$repo/scripts/drex/run_transfer.sh" --worker > "$log" 2>&1 < /dev/null &
    printf '%s\n' "$!" > "$root/transfer.pid"
    printf '%s\n' "$log" > "$root/transfer_log.txt"
    echo "Transfer started: pid=$! log=$log"
    ;;
  --worker)
    mkdir -p "$dest"
    exec 9>"$root/.transfer.lock"
    flock -n 9 || { echo 'Another transfer is running'; exit 1; }
    args=(--host "$host" --port "$port" --container "$container" --control "$control")
    "${PYTHON:-python3}" -u "$repo/scripts/drex/transfer.py" prepare-remote "${args[@]}"
    "${PYTHON:-python3}" -u "$repo/scripts/drex/transfer.py" pull "${args[@]}" --dest "$dest"
    ;;
  status)
    if [[ -f "$root/transfer_log.txt" ]]; then tail -n 30 "$(cat "$root/transfer_log.txt")"; fi
    if [[ -f "$dest/transfer_report.json" ]]; then cat "$dest/transfer_report.json"; fi
    ;;
  verify) "${PYTHON:-python3}" "$repo/scripts/drex/transfer.py" verify "$dest/package" ;;
  *) echo "Usage: $0 connect|start|status|verify" >&2; exit 2 ;;
esac
