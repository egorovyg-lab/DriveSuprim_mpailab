#!/usr/bin/env bash
set -euo pipefail
repo=${NAVSIM_DEVKIT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
export NAVSIM_DEVKIT_ROOT="$repo"
export NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT:-/workspace/exp_v2}
export PYTHONPATH="${repo}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
cd "$repo"
stage=${1:-plan}
if [[ $# -gt 0 ]]; then shift; fi
case "$stage" in
  check)
    "${PYTHON:-python}" -m unittest discover -s scripts/drivesuprim/evaluation -p 'test_corridor_projection.py' -v
    echo "Corridor projection checks: OK"
    ;;
  plan|pilot|all|report)
    "${PYTHON:-python}" -u navsim/planning/script/run_corridor_projection.py "$stage" \
      --geometry "${CORRIDOR_GEOMETRY_ROOT:-${NAVSIM_EXP_ROOT}/drivemem_memory/experiments/corridor_geometry_v2}" \
      --out "${CORRIDOR_PROJECTION_ROOT:-${NAVSIM_EXP_ROOT}/drivemem_memory/experiments/corridor_projection_v1}" "$@"
    ;;
  *) echo "Usage: $0 check|plan|pilot|all|report [Python options]" >&2; exit 2 ;;
esac
