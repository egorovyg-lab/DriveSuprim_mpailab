#!/usr/bin/env bash
set -euo pipefail
repo=${NAVSIM_DEVKIT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
export NAVSIM_DEVKIT_ROOT="$repo"
export NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT:-/workspace/exp_v2}
export PYTHONPATH="${repo}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=${SELECTOR_V2_THREADS:-2}
export MKL_NUM_THREADS=${SELECTOR_V2_THREADS:-2}
export OPENBLAS_NUM_THREADS=${SELECTOR_V2_THREADS:-2}
export PYTHONDONTWRITEBYTECODE=1
cd "$repo"
stage=${1:-audit}
if [[ $# -gt 0 ]]; then shift; fi
case "$stage" in
  check)
    "${PYTHON:-python}" -m unittest discover -s scripts/drivesuprim/evaluation -p 'test_trajectory_selector_v2.py' -v
    echo "Trajectory selector v2 checks: OK"
    ;;
  audit|fit|historical|report|all)
    base=${NAVSIM_EXP_ROOT}/drivemem_memory
    "${PYTHON:-python}" -u navsim/planning/script/run_trajectory_selector_v2.py "$stage" \
      --bundle "${SELECTOR_BUNDLE_ROOT:-${base}/selector_training_bundle_v2}" \
      --addition "${SELECTOR_V2_ADDITION:-${base}/mf_new4_shard174_v1}" \
      --prior "${SELECTOR_V2_PRIOR:-${base}/experiments/pair_utility_v1}" \
      --decoder "${SELECTOR_V2_DECODER:-${base}/decoder_features_functional_v1}" \
      --out "${SELECTOR_V2_ROOT:-${base}/experiments/trajectory_selector_v2}" \
      --epochs "${SELECTOR_V2_EPOCHS:-120}" --threads "${SELECTOR_V2_THREADS:-2}" "$@"
    ;;
  *) echo "Usage: $0 check|audit|fit|historical|report|all [Python options]" >&2; exit 2 ;;
esac
