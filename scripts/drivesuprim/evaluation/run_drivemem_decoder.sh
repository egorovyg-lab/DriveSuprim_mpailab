#!/usr/bin/env bash
set -euo pipefail
: "${NAVSIM_DEVKIT_ROOT:?}"
: "${NAVSIM_EXP_ROOT:?}"
cd "$NAVSIM_DEVKIT_ROOT"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OPENSCENE_DATA_ROOT=${OPENSCENE_DATA_ROOT:-${NAVSIM_EXP_ROOT}/drivemem_data}
base=${NAVSIM_EXP_ROOT}/drivemem_memory
export DRIVEMEM_MEMORY_ROOT=${DRIVEMEM_FROZEN_SOURCE_ROOT:-${base}/budget20gb_v1}
export DRIVEMEM_PRIOR_UTILITY_ROOT=${DRIVEMEM_PRIOR_UTILITY_ROOT:-${base}/experiments/pair_utility_v1}
export DRIVEMEM_NEW_MF_ROOT=${DRIVEMEM_NEW_MF_ROOT:-${base}/mf_new4_shard174_v1}
export DRIVEMEM_DECODER_FEATURE_ROOT=${DRIVEMEM_DECODER_FEATURE_ROOT:-${base}/decoder_features_v1}
export DRIVEMEM_MEMORY_SESSIONS=6 DRIVEMEM_FIT_SESSIONS=4 DRIVEMEM_DEV_SESSIONS=3
export DRIVEMEM_CAL_SESSIONS=2 DRIVEMEM_TEST_SESSIONS=2
export DRIVEMEM_DONOR_SCENES=8 DRIVEMEM_FIT_SCENES=8 DRIVEMEM_QUERY_SCENES=8
export DRIVEMEM_SEED=20261027 DRIVEMEM_MEMORY_MODE=enriched
export OMP_NUM_THREADS=${DRIVEMEM_CPU_THREADS:-4}
export OPENBLAS_NUM_THREADS=${DRIVEMEM_CPU_THREADS:-4}
export MKL_NUM_THREADS=${DRIVEMEM_CPU_THREADS:-4}
stage=${1:-plan}
case "$stage" in
  check)
    bash scripts/drivesuprim/evaluation/test_drivemem_decoder_launchers.sh
    python -m unittest discover -s scripts/drivesuprim/evaluation -p test_drivemem_decoder.py -v
    python navsim/planning/script/run_drivemem_decoder_export.py --cfg job --resolve >/dev/null
    ;;
  plan|export-status)
    target=decoder-plan
    [[ "$stage" != export-status ]] || target=decoder-status
    bash scripts/drivesuprim/evaluation/run_drivemem_memory.sh "$target"
    ;;
  export-pilot|export)
    export DRIVEMEM_DECODER_EXPORT_LIMIT=${DRIVEMEM_DECODER_EXPORT_LIMIT:-152}
    [[ "$stage" != export-pilot ]] || export DRIVEMEM_DECODER_EXPORT_LIMIT=4
    bash scripts/drivesuprim/evaluation/run_drivemem_memory.sh decoder-export
    ;;
  fit|evaluate|report|all)
    python -u navsim/planning/script/run_drivemem_decoder_utility.py "$stage" \
      --prior "$DRIVEMEM_PRIOR_UTILITY_ROOT" --addition "$DRIVEMEM_NEW_MF_ROOT" \
      --decoder "$DRIVEMEM_DECODER_FEATURE_ROOT" \
      --out "${DRIVEMEM_DECODER_MODEL_ROOT:-${base}/experiments/component_decoder_v1}"
    ;;
  *) echo "Usage: $0 check|plan|export-pilot|export|export-status|fit|evaluate|report|all" >&2; exit 2 ;;
esac
