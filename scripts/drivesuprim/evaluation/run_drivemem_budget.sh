#!/usr/bin/env bash
set -euo pipefail

: "${NAVSIM_DEVKIT_ROOT:?NAVSIM_DEVKIT_ROOT is not set}"
: "${NAVSIM_EXP_ROOT:?NAVSIM_EXP_ROOT is not set}"
export OPENSCENE_DATA_ROOT=${OPENSCENE_DATA_ROOT:-${NAVSIM_EXP_ROOT}/drivemem_data}
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "$NAVSIM_DEVKIT_ROOT"

stage=${1:-plan}
source_root=${DRIVEMEM_BUDGET_SOURCE:-${NAVSIM_EXP_ROOT}/drivemem_memory/pilot_v1}
export DRIVEMEM_MEMORY_ROOT=${DRIVEMEM_MEMORY_ROOT:-${NAVSIM_EXP_ROOT}/drivemem_memory/budget20gb_v1}
export DRIVEMEM_MEMORY_SESSIONS=${DRIVEMEM_MEMORY_SESSIONS:-6}
export DRIVEMEM_FIT_SESSIONS=${DRIVEMEM_FIT_SESSIONS:-4}
export DRIVEMEM_DEV_SESSIONS=${DRIVEMEM_DEV_SESSIONS:-3}
export DRIVEMEM_CAL_SESSIONS=${DRIVEMEM_CAL_SESSIONS:-2}
export DRIVEMEM_TEST_SESSIONS=${DRIVEMEM_TEST_SESSIONS:-2}
count=${DRIVEMEM_BUDGET_SCENES:-8}
export DRIVEMEM_DONOR_SCENES=$count
export DRIVEMEM_FIT_SCENES=$count
export DRIVEMEM_QUERY_SCENES=$count
budget=${DRIVEMEM_MAX_DOWNLOAD_GB:-20}
image_args=(--max-gb "$budget" --archive-dir "${DRIVEMEM_IMAGE_ARCHIVE_DIR:-${OPENSCENE_DATA_ROOT}/.downloads/drivemem_camera}")
if [[ "${DRIVEMEM_DISCARD_ARCHIVES:-0}" == 1 ]]; then
  image_args+=(--discard-archives)
fi

case "$stage" in
  check)
    python -m unittest discover -s scripts/drivesuprim/evaluation -p test_drivemem_budget.py -v
    python -m unittest discover -s scripts/drivesuprim/evaluation -p test_drivemem_images.py -v
    bash scripts/drivesuprim/evaluation/run_drivemem_memory.sh check
    ;;
  plan)
    python -u download/prepare_drivemem_budget.py "$source_root" "$DRIVEMEM_MEMORY_ROOT" \
      --max-gb "$budget" --sessions "$DRIVEMEM_MEMORY_SESSIONS" "$DRIVEMEM_FIT_SESSIONS" \
      "$DRIVEMEM_DEV_SESSIONS" "$DRIVEMEM_CAL_SESSIONS" "$DRIVEMEM_TEST_SESSIONS" \
      --scenes-per-session "$count"
    # Verify the generated identity with the unchanged main experiment code.
    bash scripts/drivesuprim/evaluation/run_drivemem_memory.sh inventory
    ;;
  images|images-status)
    action=fetch
    [[ "$stage" != images-status ]] || action=status
    python -u download/fetch_drivemem_budget_images.py "$action" "$DRIVEMEM_MEMORY_ROOT" "${image_args[@]}"
    ;;
  all|export)
    python -u download/fetch_drivemem_budget_images.py status "$DRIVEMEM_MEMORY_ROOT" "${image_args[@]}"
    bash scripts/drivesuprim/evaluation/run_drivemem_memory.sh "$stage"
    ;;
  cache|cache-status|label|fit|evaluate|test|report|paths)
    bash scripts/drivesuprim/evaluation/run_drivemem_memory.sh "$stage"
    ;;
  *) echo "Usage: $0 plan|images|images-status|cache|cache-status|all|export|label|fit|evaluate|test|report|check|paths" >&2; exit 2 ;;
esac
