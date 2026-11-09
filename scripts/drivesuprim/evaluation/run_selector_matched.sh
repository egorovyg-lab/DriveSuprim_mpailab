#!/usr/bin/env bash
set -euo pipefail
repo=${NAVSIM_DEVKIT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
export PYTHONPATH="${repo}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
cd "$repo"
stage=${1:-compare}
if [[ $# -gt 0 ]]; then shift; fi
case "$stage" in
  check)
    "${PYTHON:-python}" -m unittest discover -s scripts/drivesuprim/evaluation -p 'test_selector_matched.py' -v
    echo "Matched selector comparison checks: OK"
    ;;
  compare)
    "${PYTHON:-python}" -u scripts/drivesuprim/evaluation/selector_matched.py "$@"
    ;;
  *) echo "Usage: $0 check|compare [Python report options]" >&2; exit 2 ;;
esac
