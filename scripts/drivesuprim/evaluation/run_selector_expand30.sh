#!/usr/bin/env bash
set -euo pipefail
repo=${NAVSIM_DEVKIT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
export NAVSIM_DEVKIT_ROOT="$repo"
export NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT:-/workspace/exp_v2}
export PYTHONPATH="${repo}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
cd "$repo"
base=${NAVSIM_EXP_ROOT}/drivemem_memory
data=${SELECTOR_EXPAND30_DATA:-${base}/selector_data_expansion_v3_30gb}
collection=${SELECTOR_EXPAND30_COLLECTION:-${base}/selector_collection_v3_30gb}
bundle=${SELECTOR_EXPAND30_BUNDLE:-${base}/selector_training_bundle_v3_30gb}
parent=${SELECTOR_EXPAND30_PARENT_BUNDLE:-${base}/selector_training_bundle_v2}
stage=${1:-plan}
images() {
  "${PYTHON:-python}" -u download/prepare_selector_expand30.py "$1" \
    --previous-data "${SELECTOR_EXPAND30_PARENT_DATA:-${base}/selector_data_expansion_v2}" \
    --previous-bundle "$parent" --out "$data" \
    --archive-dir "${SELECTOR_DATA_ARCHIVES:-/workspace/exp_v2/drivemem_data/.downloads/drivemem_camera}" \
    --max-extra-gb "${SELECTOR_EXPAND30_GB:-30}"
}
collect() {
  SELECTOR_DATA_ROOT="$data" SELECTOR_COLLECTION_ROOT="$collection" \
    bash scripts/drivesuprim/evaluation/run_selector_collection.sh "$1"
}
make_bundle() {
  "${PYTHON:-python}" -u navsim/planning/script/run_selector_extend_bundle.py "$1" \
    --parent "$parent" --collection "$collection" --out "$bundle"
}
case "$stage" in
  check)
    "${PYTHON:-python}" -m unittest discover -s scripts/drivesuprim/evaluation -p 'test_selector_expand30.py' -v
    echo "Selector second expansion checks: OK"
    ;;
  plan|fetch) images "$stage" ;;
  images-status) images status ;;
  init|cache|export-pilot|export|label-pilot|label|status|report) collect "$stage" ;;
  prepare) images plan; images fetch; collect init; collect cache ;;
  collect) collect init; collect cache; collect export; collect label ;;
  bundle) make_bundle build ;;
  bundle-audit) make_bundle audit ;;
  fit)
    epochs=${SELECTOR_V2_EPOCHS:-120}
    SELECTOR_BUNDLE_ROOT="$bundle" \
      SELECTOR_V2_ROOT="${SELECTOR_EXPAND30_FIT_ROOT:-${base}/experiments/trajectory_selector_v3_30gb_e${epochs}}" \
      bash scripts/drivesuprim/evaluation/run_trajectory_selector_v2.sh fit
    ;;
  all)
    images plan; images fetch
    collect init; collect cache; collect export; collect label
    make_bundle build
    ;;
  *) echo "Usage: $0 check|plan|fetch|images-status|prepare|init|cache|export-pilot|export|label-pilot|label|collect|status|report|bundle|bundle-audit|fit|all" >&2; exit 2 ;;
esac
