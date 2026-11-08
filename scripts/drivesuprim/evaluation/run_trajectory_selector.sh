#!/usr/bin/env bash
set -euo pipefail
repo=${NAVSIM_DEVKIT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
export NAVSIM_DEVKIT_ROOT="$repo"
export NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT:-/workspace/exp_v2}
export PYTHONPATH="${repo}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=${DRIVEMEM_SELECTOR_THREADS:-2}
export MKL_NUM_THREADS=${DRIVEMEM_SELECTOR_THREADS:-2}
export OPENBLAS_NUM_THREADS=${DRIVEMEM_SELECTOR_THREADS:-2}
export PYTHONDONTWRITEBYTECODE=1
cd "$repo"
stage=${1:-audit}
if [[ $# -gt 0 ]]; then shift; fi
case "$stage" in
  check)
    "${PYTHON:-python}" -m unittest discover -s scripts/drivesuprim/evaluation -p 'test_trajectory_selector.py' -v
    echo "Trajectory selector checks: OK"
    ;;
  audit|fit|predict|evaluate|report|all)
    base=${NAVSIM_EXP_ROOT}/drivemem_memory
    "${PYTHON:-python}" -u navsim/planning/script/run_trajectory_selector.py "$stage" \
      --prior "${DRIVEMEM_SELECTOR_PRIOR:-${base}/experiments/pair_utility_v1}" \
      --addition "${DRIVEMEM_SELECTOR_ADDITION:-${base}/mf_new4_shard174_v1}" \
      --decoder "${DRIVEMEM_SELECTOR_DECODER:-${base}/decoder_features_functional_v1}" \
      --out "${DRIVEMEM_SELECTOR_ROOT:-${base}/experiments/trajectory_selector_v1}" \
      --epochs "${DRIVEMEM_SELECTOR_EPOCHS:-120}" \
      --threads "${DRIVEMEM_SELECTOR_THREADS:-2}" "$@"
    ;;
  *) echo "Usage: $0 check|audit|fit|predict|evaluate|report|all [Python options]" >&2; exit 2 ;;
esac
