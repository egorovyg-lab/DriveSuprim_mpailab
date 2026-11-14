#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
root=${DREX_RUNTIME_ROOT:-/scratch/pr0201/drex_runtime}
stage=${1:-status}
case "$stage" in
  prepare|connect) exec bash "$repo/scripts/drex/run_pool_audit.sh" "$stage" ;;
  check) cd "$repo"; exec "${PYTHON:-python3}" -B -m unittest discover -s scripts/drex -p test_pool_direct.py -v ;;
  stage|start|status|fetch)
    exec "${PYTHON:-python3}" -u "$repo/scripts/drex/pool_direct.py" "$stage" --root "$root" \
      --name "${DREX_POOL_NAME:-rollout_audit_pool64_v1}" \
      --host "${DREX_ASTRA_HOST:-egorov@93.180.62.99}" --port "${DREX_ASTRA_PORT:-27434}" \
      --container "${DREX_ASTRA_CONTAINER:-egorov_drivesuprim}" \
      --repo "${DREX_ASTRA_REPO:-/workspace/DriveSuprim}" \
      --control "${DREX_TRANSFER_CONTROL:-$HOME/.ssh/drex-astra-transfer.sock}" ;;
  *) echo 'Usage: run_pool_direct.sh prepare|connect|check|stage|start|status|fetch' >&2; exit 2 ;;
esac
