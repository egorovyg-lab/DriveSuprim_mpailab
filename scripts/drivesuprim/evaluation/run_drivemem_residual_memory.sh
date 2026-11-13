#!/usr/bin/env bash
set -euo pipefail
repo=${NAVSIM_DEVKIT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
export PYTHONPATH="${repo}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=20261113
export CUDA_VISIBLE_DEVICES=""
threads=${DRIVEMEM_RESIDUAL_THREADS:-2}
[[ "$threads" =~ ^[1-9][0-9]*$ ]] || { echo 'threads must be positive' >&2; exit 2; }
export OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads" OPENBLAS_NUM_THREADS="$threads"
export NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT:-/workspace/exp_v2}
cd "$repo"
stage=${1:-status}
if [[ $# -gt 0 ]]; then shift; fi
case "$stage" in
  check)
    [[ $# -eq 0 ]] || { echo 'check takes no arguments' >&2; exit 2; }
    "${PYTHON:-python}" -c 'import numpy,pandas,torch; print("CPU dependencies: OK; torch=" + torch.__version__)'
    "${PYTHON:-python}" -m unittest discover -s scripts/drivesuprim/evaluation -p 'test_drivemem_residual_memory.py' -v
    echo 'DriveMem residual memory checks: OK'
    ;;
  night)
    bash scripts/drivesuprim/evaluation/run_drivemem_residual_memory.sh check
    bash scripts/drivesuprim/evaluation/run_drivemem_residual_memory.sh smoke
    bash scripts/drivesuprim/evaluation/run_drivemem_residual_memory.sh plan "$@"
    bash scripts/drivesuprim/evaluation/run_drivemem_residual_memory.sh pilot "$@"
    bash scripts/drivesuprim/evaluation/run_drivemem_residual_memory.sh all "$@"
    ;;
  inventory|plan|pilot|all|report|status|smoke)
    args=("$stage" --threads "$threads")
    [[ -z ${DRIVEMEM_RESIDUAL_BUNDLE:-} ]] || args+=(--bundle "$DRIVEMEM_RESIDUAL_BUNDLE")
    [[ -z ${DRIVEMEM_RESIDUAL_LABELS:-} ]] || args+=(--labels "$DRIVEMEM_RESIDUAL_LABELS")
    [[ -z ${DRIVEMEM_RESIDUAL_REQUEST:-} ]] || args+=(--label-request "$DRIVEMEM_RESIDUAL_REQUEST")
    [[ -z ${DRIVEMEM_RESIDUAL_PACKAGE_MANIFEST:-} ]] || args+=(--package-manifest "$DRIVEMEM_RESIDUAL_PACKAGE_MANIFEST")
    [[ -z ${DRIVEMEM_RESIDUAL_ROOT:-} ]] || args+=(--out "$DRIVEMEM_RESIDUAL_ROOT")
    exec "${PYTHON:-python}" -u navsim/planning/script/run_drivemem_residual_memory.py "${args[@]}" "$@"
    ;;
  *) echo "Usage: $0 check|smoke|inventory|plan|pilot|all|night|report|status [Python options]" >&2; exit 2 ;;
esac
