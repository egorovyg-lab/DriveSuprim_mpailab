#!/usr/bin/env bash
set -euo pipefail
repo=${NAVSIM_DEVKIT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
export NAVSIM_DEVKIT_ROOT="$repo"
export NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT:-/workspace/exp_v2}
export PYTHONPATH="${repo}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${SELECTOR_COMPONENT_THREADS:-2}
export MKL_NUM_THREADS=${SELECTOR_COMPONENT_THREADS:-2}
export OPENBLAS_NUM_THREADS=${SELECTOR_COMPONENT_THREADS:-2}
cd "$repo"
stage=${1:-plan}
if [[ $# -gt 0 ]]; then shift; fi
case "$stage" in
  check)
    "${PYTHON:-python}" -m unittest discover -s scripts/drivesuprim/evaluation -p 'test_selector_component_critic.py' -v
    echo "Component critic checks: OK"
    ;;
  night)
    bash scripts/drivesuprim/evaluation/run_selector_component_critic.sh check
    bash scripts/drivesuprim/evaluation/run_selector_component_critic.sh smoke
    bash scripts/drivesuprim/evaluation/run_selector_component_critic.sh all "$@"
    ;;
  formula|plan|all|report|status|smoke)
    base=${NAVSIM_EXP_ROOT}/drivemem_memory
    "${PYTHON:-python}" -u navsim/planning/script/run_selector_component_critic.py "$stage" \
      --bundle "${SELECTOR_COMPONENT_BUNDLE:-${base}/selector_training_bundle_v3_30gb}" \
      --night "${SELECTOR_COMPONENT_REFERENCE:-${base}/experiments/residual_critic_night_v1}" \
      --out "${SELECTOR_COMPONENT_ROOT:-${base}/experiments/component_critic_v1}" \
      --epochs "${SELECTOR_COMPONENT_EPOCHS:-120}" --threads "${SELECTOR_COMPONENT_THREADS:-2}" "$@"
    ;;
  *) echo "Usage: $0 check|smoke|formula|plan|night|all|status|report [Python options]" >&2; exit 2 ;;
esac
