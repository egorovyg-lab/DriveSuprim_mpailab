#!/usr/bin/env bash
set -euo pipefail
repo=${NAVSIM_DEVKIT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
export NAVSIM_DEVKIT_ROOT="$repo"
export NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT:-/workspace/exp_v2}
export PYTHONPATH="${repo}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${SELECTOR_CRITIC_THREADS:-2}
export MKL_NUM_THREADS=${SELECTOR_CRITIC_THREADS:-2}
export OPENBLAS_NUM_THREADS=${SELECTOR_CRITIC_THREADS:-2}
cd "$repo"
stage=${1:-plan}
if [[ $# -gt 0 ]]; then shift; fi
case "$stage" in
  check)
    "${PYTHON:-python}" -m unittest discover -s scripts/drivesuprim/evaluation -p 'test_selector_residual_critic.py' -v
    echo "Residual critic checks: OK"
    ;;
  night)
    bash scripts/drivesuprim/evaluation/run_selector_residual_night.sh check
    bash scripts/drivesuprim/evaluation/run_selector_residual_night.sh smoke
    bash scripts/drivesuprim/evaluation/run_selector_residual_night.sh all "$@"
    ;;
  plan|all|report|status|smoke)
    base=${NAVSIM_EXP_ROOT}/drivemem_memory
    "${PYTHON:-python}" -u navsim/planning/script/run_selector_residual_night.py "$stage" \
      --bundle "${SELECTOR_CRITIC_BUNDLE:-${base}/selector_training_bundle_v3_30gb}" \
      --reference "${SELECTOR_CRITIC_REFERENCE:-${base}/experiments/trajectory_selector_v3_30gb_e120}" \
      --addition "${SELECTOR_CRITIC_ADDITION:-${base}/mf_new4_shard174_v1}" \
      --out "${SELECTOR_CRITIC_ROOT:-${base}/experiments/residual_critic_night_v1}" \
      --epochs "${SELECTOR_CRITIC_EPOCHS:-120}" --threads "${SELECTOR_CRITIC_THREADS:-2}" "$@"
    ;;
  *) echo "Usage: $0 check|smoke|plan|night|all|status|report [Python options]" >&2; exit 2 ;;
esac
