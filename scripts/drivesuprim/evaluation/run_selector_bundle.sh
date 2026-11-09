#!/usr/bin/env bash
set -euo pipefail
repo=${NAVSIM_DEVKIT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
export NAVSIM_DEVKIT_ROOT="$repo"
export NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT:-/workspace/exp_v2}
export PYTHONPATH="${repo}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
cd "$repo"
stage=${1:-audit}
case "$stage" in
  check)
    "${PYTHON:-python}" -m unittest discover -s scripts/drivesuprim/evaluation -p 'test_selector_bundle.py' -v
    echo "Selector training bundle checks: OK"
    ;;
  build|audit)
    base=${NAVSIM_EXP_ROOT}/drivemem_memory
    "${PYTHON:-python}" -u navsim/planning/script/run_selector_bundle.py "$stage" \
      --prior "${SELECTOR_BUNDLE_PRIOR:-${base}/experiments/pair_utility_v1}" \
      --addition "${SELECTOR_BUNDLE_ADDITION:-${base}/mf_new4_shard174_v1}" \
      --decoder "${SELECTOR_BUNDLE_DECODER:-${base}/decoder_features_functional_v1}" \
      --collection "${SELECTOR_COLLECTION_ROOT:-${base}/selector_collection_v2}" \
      --out "${SELECTOR_BUNDLE_ROOT:-${base}/selector_training_bundle_v2}"
    ;;
  *) echo "Usage: $0 check|build|audit" >&2; exit 2 ;;
esac
