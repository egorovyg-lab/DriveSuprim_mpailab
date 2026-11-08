#!/usr/bin/env bash
set -euo pipefail
repo=${NAVSIM_DEVKIT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
export PYTHONPATH="${repo}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
cd "$repo"
stage=${1:-plan}
if [[ $# -gt 0 ]]; then shift; fi
case "$stage" in
  check)
    "${PYTHON:-python}" -m unittest discover -s scripts/drivesuprim/evaluation -p 'test_selector_dataset.py' -v
    echo "Selector dataset checks: OK"
    ;;
  plan|fetch|status)
    base=${NAVSIM_EXP_ROOT:-/workspace/exp_v2}/drivemem_memory
    "${PYTHON:-python}" -u download/prepare_selector_dataset.py "$stage" \
      --source "${SELECTOR_DATA_SOURCE:-${base}/budget20gb_v1}" \
      --addition "${SELECTOR_DATA_ADDITION:-${base}/mf_new4_shard174_v1}" \
      --inventory "${SELECTOR_DATA_INVENTORY:-${base}/pilot_v1}" \
      --validation "${SELECTOR_DATA_VALIDATION:-${base}/validation_cached4_v1}" \
      --archive-dir "${SELECTOR_DATA_ARCHIVES:-/workspace/exp_v2/drivemem_data/.downloads/drivemem_camera}" \
      --out "${SELECTOR_DATA_ROOT:-${base}/selector_data_expansion_v2}" \
      --max-extra-gb "${SELECTOR_DATA_BUDGET_GB:-30}" "$@"
    ;;
  *) echo "Usage: $0 check|plan|fetch|status [Python options]" >&2; exit 2 ;;
esac
