#!/usr/bin/env bash
set -euo pipefail

stage=${1:-inventory}
case "$stage" in inventory|cache|export|label|fit|evaluate|test|report|all|check) ;; *)
  echo "Usage: $0 inventory|cache|export|label|fit|evaluate|test|report|all|check" >&2; exit 2 ;;
esac
: "${NAVSIM_DEVKIT_ROOT:?NAVSIM_DEVKIT_ROOT is not set}"
: "${NAVSIM_EXP_ROOT:?NAVSIM_EXP_ROOT is not set}"
: "${OPENSCENE_DATA_ROOT:?OPENSCENE_DATA_ROOT is not set}"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"
cd "$NAVSIM_DEVKIT_ROOT"

if [[ "$stage" == check ]]; then
    python -m unittest discover -s scripts/drivesuprim/evaluation -p test_drivemem_memory.py -v
    python navsim/planning/script/run_drivemem_memory.py --cfg job --resolve >/dev/null
    echo "Tests and Hydra configuration: OK"
    exit 0
fi

root=${DRIVEMEM_MEMORY_ROOT:-${NAVSIM_EXP_ROOT}/drivemem_memory/pilot_v1}
checkpoint=${DRIVEMEM_CHECKPOINT:-${NAVSIM_EXP_ROOT}/model_ckpt/drivesuprim_r34.ckpt}
[[ -f "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2; exit 1; }
[[ -n "${NUPLAN_MAPS_ROOT:-}" ]] || { echo "NUPLAN_MAPS_ROOT is not set" >&2; exit 1; }

cmd=(python -u navsim/planning/script/run_drivemem_memory.py
    "drivemem.stage=$stage" "drivemem.root=$root" "agent.checkpoint_path=$checkpoint"
    "drivemem.seed=${DRIVEMEM_SEED:-20261027}"
    "drivemem.sessions.memory=${DRIVEMEM_MEMORY_SESSIONS:-24}"
    "drivemem.sessions.fit=${DRIVEMEM_FIT_SESSIONS:-16}"
    "drivemem.sessions.development=${DRIVEMEM_DEV_SESSIONS:-8}"
    "drivemem.sessions.calibration=${DRIVEMEM_CAL_SESSIONS:-8}"
    "drivemem.sessions.test=${DRIVEMEM_TEST_SESSIONS:-8}"
    "drivemem.scenes.memory=${DRIVEMEM_DONOR_SCENES:-8}"
    "drivemem.scenes.fit=${DRIVEMEM_FIT_SCENES:-4}"
    "drivemem.scenes.development=${DRIVEMEM_QUERY_SCENES:-4}"
    "drivemem.scenes.calibration=${DRIVEMEM_QUERY_SCENES:-4}"
    "drivemem.scenes.test=${DRIVEMEM_QUERY_SCENES:-4}"
    "drivemem.memory_mode=${DRIVEMEM_MEMORY_MODE:-enriched}"
)
echo "Memory pilot stage=$stage root=$root"
env -u WORLD_SIZE -u RANK -u LOCAL_RANK -u MASTER_ADDR -u MASTER_PORT \
    -u TORCHELASTIC_RUN_ID -u TORCHELASTIC_RESTART_COUNT -u PL_TORCH_DISTRIBUTED_BACKEND \
    "${cmd[@]}"
