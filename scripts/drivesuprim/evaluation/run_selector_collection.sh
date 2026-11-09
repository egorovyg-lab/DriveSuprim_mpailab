#!/usr/bin/env bash
set -euo pipefail
repo=${NAVSIM_DEVKIT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
export NAVSIM_DEVKIT_ROOT="$repo"
export NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT:-/workspace/exp_v2}
export PYTHONPATH="${repo}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${SELECTOR_COLLECTION_THREADS:-4}
export OPENBLAS_NUM_THREADS=${SELECTOR_COLLECTION_THREADS:-4}
export MKL_NUM_THREADS=${SELECTOR_COLLECTION_THREADS:-4}
# Before Python/CUDA initialization; same prospective profile as accepted sidecar.
export PYTHONHASHSEED=20261027
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=${SELECTOR_COLLECTION_GPU:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}
cd "$repo"
stage=${1:-status}
case "$stage" in
  check)
    "${PYTHON:-python}" -m unittest discover -s scripts/drivesuprim/evaluation -p 'test_selector_collection.py' -v
    echo "Selector collection checks: OK"
    ;;
  init|cache|export-pilot|export|label-pilot|label|status|report)
    base=${NAVSIM_EXP_ROOT}/drivemem_memory
    "${PYTHON:-python}" -u navsim/planning/script/run_selector_collection.py "$stage" \
      --data "${SELECTOR_DATA_ROOT:-${base}/selector_data_expansion_v2}" \
      --out "${SELECTOR_COLLECTION_ROOT:-${base}/selector_collection_v2}"
    ;;
  *) echo "Usage: $0 check|init|cache|export-pilot|export|label-pilot|label|status|report" >&2; exit 2 ;;
esac
