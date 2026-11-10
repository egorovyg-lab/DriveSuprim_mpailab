#!/usr/bin/env bash
set -euo pipefail
repo=${NAVSIM_DEVKIT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
export PYTHONPATH="${repo}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
cd "$repo"
stage=${1:-audit}
if [[ $# -gt 0 ]]; then shift; fi
case "$stage" in
  check)
    "${PYTHON:-python}" -m unittest discover -s scripts/drivesuprim/evaluation -p 'test_selector_veto_audit.py' -v
    echo "Teacher-veto audit checks: OK"
    ;;
  audit) "${PYTHON:-python}" -u scripts/drivesuprim/evaluation/audit_selector_teacher_veto.py "$@" ;;
  *) echo "Usage: $0 check|audit [Python options]" >&2; exit 2 ;;
esac
