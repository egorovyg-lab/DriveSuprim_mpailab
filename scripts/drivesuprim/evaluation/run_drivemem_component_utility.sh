#!/usr/bin/env bash
set -euo pipefail
: "${NAVSIM_DEVKIT_ROOT:?}"
: "${NAVSIM_EXP_ROOT:?}"
cd "$NAVSIM_DEVKIT_ROOT"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=${DRIVEMEM_CPU_THREADS:-4}
export OPENBLAS_NUM_THREADS=${DRIVEMEM_CPU_THREADS:-4}
export MKL_NUM_THREADS=${DRIVEMEM_CPU_THREADS:-4}
base=${NAVSIM_EXP_ROOT}/drivemem_memory
stage=${1:-plan}
case "$stage" in
  check)
    bash scripts/drivesuprim/evaluation/test_drivemem_component_launchers.sh
    python -m unittest discover -s scripts/drivesuprim/evaluation -p test_drivemem_component_utility.py -v
    ;;
  plan|fit|evaluate|report|all)
    python -u navsim/planning/script/run_drivemem_component_utility.py "$stage" \
      --prior "${DRIVEMEM_PRIOR_UTILITY_ROOT:-${base}/experiments/pair_utility_v1}" \
      --addition "${DRIVEMEM_NEW_MF_ROOT:-${base}/mf_new4_shard174_v1}" \
      --out "${DRIVEMEM_COMPONENT_ROOT:-${base}/experiments/component_utility_v1}"
    ;;
  *) echo "Usage: $0 check|plan|fit|evaluate|report|all" >&2; exit 2 ;;
esac
