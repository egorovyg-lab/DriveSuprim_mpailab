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
base=${NAVSIM_EXP_ROOT}/drivemem_memory
sweep=${SELECTOR_SCALE_ROOT:-${base}/experiments/selector_duration_v1}
case "$stage" in
  check)
    "${PYTHON:-python}" -m unittest discover -s scripts/drivesuprim/evaluation -p 'test_selector_scale.py' -v
    echo "Selector scaling checks: OK"
    ;;
  plan|capacity|compare)
    "${PYTHON:-python}" -u scripts/drivesuprim/evaluation/selector_scale.py "$stage" \
      --data "${SELECTOR_DATA_ROOT:-${base}/selector_data_expansion_v2}" \
      --bundle "${SELECTOR_BUNDLE_ROOT:-${base}/selector_training_bundle_v2}" \
      --archive-dir "${SELECTOR_DATA_ARCHIVES:-/workspace/exp_v2/drivemem_data/.downloads/drivemem_camera}" \
      --reference "${SELECTOR_SCALE_REFERENCE:-${base}/experiments/trajectory_selector_v2}" \
      --sweep-root "$sweep" "$@"
    ;;
  long-fit)
    # Only epochs and output root change. Do not run old development evaluation
    # or touch the fresh holdout while comparing training durations.
    for epochs in 360 1000; do
      echo "DURATION FIT epochs=$epochs root=$sweep/epochs$epochs"
      SELECTOR_V2_ROOT="$sweep/epochs$epochs" SELECTOR_V2_EPOCHS="$epochs" \
        bash scripts/drivesuprim/evaluation/run_trajectory_selector_v2.sh fit
    done
    bash scripts/drivesuprim/evaluation/run_selector_scale.sh compare
    ;;
  *) echo "Usage: $0 check|plan|capacity|long-fit|compare [report options]" >&2; exit 2 ;;
esac
