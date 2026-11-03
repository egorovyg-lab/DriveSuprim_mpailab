#!/usr/bin/env bash
set -euo pipefail
: "${NAVSIM_DEVKIT_ROOT:?}"
: "${NAVSIM_EXP_ROOT:?}"
cd "$NAVSIM_DEVKIT_ROOT"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OPENSCENE_DATA_ROOT=${OPENSCENE_DATA_ROOT:-${NAVSIM_EXP_ROOT}/drivemem_data}
export DRIVEMEM_MEMORY_ROOT=${DRIVEMEM_VALIDATION_ROOT:-${NAVSIM_EXP_ROOT}/drivemem_memory/validation_cached4_v1}
model_root=${DRIVEMEM_FROZEN_MODEL_ROOT:-${NAVSIM_EXP_ROOT}/drivemem_memory/experiments/action_transfer_v2}
source_root=${DRIVEMEM_FROZEN_SOURCE_ROOT:-${NAVSIM_EXP_ROOT}/drivemem_memory/budget20gb_v1}
archives=${DRIVEMEM_IMAGE_ARCHIVE_DIR:-${OPENSCENE_DATA_ROOT}/.downloads/drivemem_camera}
export DRIVEMEM_MEMORY_SESSIONS=0 DRIVEMEM_FIT_SESSIONS=0 DRIVEMEM_DEV_SESSIONS=4
export DRIVEMEM_CAL_SESSIONS=0 DRIVEMEM_TEST_SESSIONS=0
export DRIVEMEM_DONOR_SCENES=4 DRIVEMEM_FIT_SCENES=4 DRIVEMEM_QUERY_SCENES=4
export DRIVEMEM_SEED=20261027 DRIVEMEM_MEMORY_MODE=enriched
export OMP_NUM_THREADS=${DRIVEMEM_CPU_THREADS:-4}
export OPENBLAS_NUM_THREADS=${DRIVEMEM_CPU_THREADS:-4}
export MKL_NUM_THREADS=${DRIVEMEM_CPU_THREADS:-4}
stage=${1:-plan}
case "$stage" in
  check)
    python -m unittest discover -s scripts/drivesuprim/evaluation -p test_drivemem_frozen_validation.py -v
    python navsim/planning/script/run_drivemem_frozen_validation.py --cfg job --resolve >/dev/null
    ;;
  plan)
    python -u download/prepare_drivemem_cached_validation.py "$source_root" "$model_root" "$DRIVEMEM_MEMORY_ROOT" --archive-dir "$archives"
    bash scripts/drivesuprim/evaluation/run_drivemem_memory.sh inventory
    ;;
  images|images-status)
    action=fetch
    [[ "$stage" != images-status ]] || action=status
    python -u download/fetch_drivemem_budget_images.py "$action" "$DRIVEMEM_MEMORY_ROOT" --archive-dir "$archives" --max-gb 20 --offline
    ;;
  cache|cache-status)
    bash scripts/drivesuprim/evaluation/run_drivemem_memory.sh "$stage"
    ;;
  export)
    bash "$0" images-status
    bash scripts/drivesuprim/evaluation/run_drivemem_memory.sh export
    ;;
  evaluate)
    bash scripts/drivesuprim/evaluation/run_drivemem_memory.sh frozen-validation
    ;;
  all)
    bash "$0" images
    bash "$0" cache
    bash "$0" export
    bash "$0" evaluate
    ;;
  report)
    python -c 'import pathlib,json,sys; print(json.dumps(json.loads((pathlib.Path(sys.argv[1])/"frozen_evaluation/report.json").read_text()),indent=2))' "$DRIVEMEM_MEMORY_ROOT"
    ;;
  *) echo "Usage: $0 check|plan|images|images-status|cache|cache-status|export|evaluate|all|report" >&2; exit 2 ;;
esac
