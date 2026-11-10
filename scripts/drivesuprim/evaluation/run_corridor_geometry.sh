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
    "${PYTHON:-python}" -m unittest discover -s scripts/drivesuprim/evaluation -p 'test_corridor_geometry.py' -v
    echo "Corridor geometry checks: OK"
    ;;
  pilot)
    "${PYTHON:-python}" -u navsim/planning/script/run_corridor_geometry.py prepare-pilot \
      --out "${CORRIDOR_ROOT:-${NAVSIM_EXP_ROOT}/drivemem_memory/experiments/corridor_geometry_pilot_v2}" "$@"
    "${PYTHON:-python}" -u navsim/planning/script/run_corridor_geometry.py infer \
      --out "${CORRIDOR_ROOT:-${NAVSIM_EXP_ROOT}/drivemem_memory/experiments/corridor_geometry_pilot_v2}" "$@"
    "${PYTHON:-python}" -u navsim/planning/script/run_corridor_geometry.py preview \
      --out "${CORRIDOR_ROOT:-${NAVSIM_EXP_ROOT}/drivemem_memory/experiments/corridor_geometry_pilot_v2}" "$@"
    "${PYTHON:-python}" -u navsim/planning/script/run_corridor_geometry.py evaluate \
      --out "${CORRIDOR_ROOT:-${NAVSIM_EXP_ROOT}/drivemem_memory/experiments/corridor_geometry_pilot_v2}" "$@"
    ;;
  plan|prepare-pilot|prepare|infer|evaluate|preview|report|all)
    "${PYTHON:-python}" -u navsim/planning/script/run_corridor_geometry.py "$stage" \
      --out "${CORRIDOR_ROOT:-${NAVSIM_EXP_ROOT}/drivemem_memory/experiments/corridor_geometry_v2}" "$@"
    ;;
  *) echo "Usage: $0 check|plan|pilot|prepare|infer|evaluate|preview|report|all [Python options]" >&2; exit 2 ;;
esac
