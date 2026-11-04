#!/usr/bin/env bash
set -euo pipefail
: "${NAVSIM_DEVKIT_ROOT:?}"
: "${NAVSIM_EXP_ROOT:?}"
cd "$NAVSIM_DEVKIT_ROOT"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OPENSCENE_DATA_ROOT=${OPENSCENE_DATA_ROOT:-${NAVSIM_EXP_ROOT}/drivemem_data}
export DRIVEMEM_MEMORY_ROOT=${DRIVEMEM_EXPANSION_ROOT:-${NAVSIM_EXP_ROOT}/drivemem_memory/mf_expansion_v1}
source_root=${DRIVEMEM_FROZEN_SOURCE_ROOT:-${NAVSIM_EXP_ROOT}/drivemem_memory/budget20gb_v1}
inventory=${DRIVEMEM_INVENTORY_ROOT:-${NAVSIM_EXP_ROOT}/drivemem_memory/pilot_v1}
validation=${DRIVEMEM_VALIDATION_ROOT:-${NAVSIM_EXP_ROOT}/drivemem_memory/validation_cached4_v1}
archives=${DRIVEMEM_IMAGE_ARCHIVE_DIR:-${OPENSCENE_DATA_ROOT}/.downloads/drivemem_camera}
export DRIVEMEM_MEMORY_SESSIONS=6 DRIVEMEM_FIT_SESSIONS=4 DRIVEMEM_DEV_SESSIONS=0
export DRIVEMEM_CAL_SESSIONS=0 DRIVEMEM_TEST_SESSIONS=0
export DRIVEMEM_DONOR_SCENES=16 DRIVEMEM_FIT_SCENES=16 DRIVEMEM_QUERY_SCENES=16
export DRIVEMEM_SEED=20261027 DRIVEMEM_MEMORY_MODE=enriched
export OMP_NUM_THREADS=${DRIVEMEM_CPU_THREADS:-4}
export OPENBLAS_NUM_THREADS=${DRIVEMEM_CPU_THREADS:-4}
export MKL_NUM_THREADS=${DRIVEMEM_CPU_THREADS:-4}
stage=${1:-plan}
case "$stage" in
  capacity)
    python -u download/plan_drivemem_mf_capacity.py "$source_root" --inventory "$inventory" \
      --validation "$validation" --archive-dir "$archives" --max-extra-gb "${DRIVEMEM_CAPACITY_BUDGET_GB:-5}"
    ;;
  check)
    bash scripts/drivesuprim/evaluation/test_drivemem_safety_launchers.sh
    python -m unittest discover -s scripts/drivesuprim/evaluation -p test_drivemem_safety_data.py -v
    python navsim/planning/script/run_drivemem_safety_data.py --cfg job --resolve >/dev/null
    ;;
  plan)
    python -u download/prepare_drivemem_mf_expansion.py "$source_root" "$DRIVEMEM_MEMORY_ROOT" \
      --inventory "$inventory" --validation "$validation" --archive-dir "$archives"
    bash scripts/drivesuprim/evaluation/run_drivemem_memory.sh data-validate
    ;;
  images-offline|images-status)
    action=fetch
    [[ "$stage" != images-status ]] || action=status
    python -u download/fetch_drivemem_budget_images.py "$action" "$DRIVEMEM_MEMORY_ROOT" \
      --archive-dir "$archives" --max-gb 20 --offline
    ;;
  cache|cache-status)
    bash scripts/drivesuprim/evaluation/run_drivemem_memory.sh "$stage"
    ;;
  export)
    bash "$0" images-status
    bash scripts/drivesuprim/evaluation/run_drivemem_memory.sh export
    ;;
  screen|support-report)
    bash scripts/drivesuprim/evaluation/run_drivemem_memory.sh "data-$stage"
    ;;
  label-pilot|label)
    export DRIVEMEM_LABEL_LIMIT=${DRIVEMEM_LABEL_LIMIT:-80}
    [[ "$stage" != label-pilot ]] || export DRIVEMEM_LABEL_LIMIT=8
    bash scripts/drivesuprim/evaluation/run_drivemem_memory.sh data-label
    ;;
  prepare)
    for step in plan images-offline cache export screen; do bash "$0" "$step"; done
    ;;
  *) echo "Usage: $0 check|capacity|plan|prepare|images-offline|images-status|cache|cache-status|export|screen|label-pilot|label|support-report" >&2; exit 2 ;;
esac
