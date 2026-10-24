#!/usr/bin/env bash
set -euo pipefail

if (( $# < 5 || $# > 7 )); then
    echo "Usage: $0 DIR FILE NUM_REFINEMENT_STAGE STAGE_LAYERS TOPKS [AGENT] [INFERENCE_MODEL]" >&2
    exit 2
fi
: "${NAVSIM_DEVKIT_ROOT:?Set NAVSIM_DEVKIT_ROOT}"
: "${NAVSIM_EXP_ROOT:?Set NAVSIM_EXP_ROOT}"
: "${OPENSCENE_DATA_ROOT:?Set OPENSCENE_DATA_ROOT}"
: "${NUPLAN_MAPS_ROOT:?Set NUPLAN_MAPS_ROOT}"

dir=$1
file=$2
num_refinement_stage=$3
stage_layers=$4
topks=$5
agent=${6:-drivesuprim_agent_r34}
inference_model=${7:-teacher}
checkpoint="${NAVSIM_EXP_ROOT}/${dir}/${file}.ckpt"
cache="${NAVSIM_V1_CACHE:-${NAVSIM_EXP_ROOT}/metric_cache/navsim_v1/test/ori}"
mode=${NAVSIM_V1_MODE:-all}
if [[ ! -f "$checkpoint" ]]; then
    echo "Checkpoint not found: $checkpoint" >&2
    exit 1
fi
case "$mode" in all|cache|evaluate) ;; *) echo "NAVSIM_V1_MODE must be all, cache or evaluate" >&2; exit 2 ;; esac

export TMPDIR="${TMPDIR:-/tmp}"
export RAY_TMPDIR="${RAY_TMPDIR:-${TMPDIR}/ray}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "$TMPDIR" "$RAY_TMPDIR"
unset WORLD_SIZE RANK LOCAL_RANK MASTER_ADDR MASTER_PORT
unset TORCHELASTIC_RUN_ID TORCHELASTIC_RESTART_COUNT PL_TORCH_DISTRIBUTED_BACKEND

cmd=(
    python3
    "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_drivesuprim_navsim_v1.py"
    "agent=${agent}"
    "train_test_split=navtest"
    "scorer=navsim_v1"
    "++trainer.params.devices=${GPUS_PER_NODE:-2}"
    "dataloader.params.batch_size=${BATCH_SIZE:-1}"
    "worker.threads_per_node=${THREADS_PER_NODE:-4}"
    "agent.checkpoint_path=${checkpoint}"
    "agent.config.training=false"
    "agent.config.only_ori_input=true"
    "+agent.config.inference.protocol=navsim_v1"
    "agent.config.inference.model=${inference_model}"
    "agent.config.inference.save_pickle=true"
    "agent.config.refinement.use_multi_stage=true"
    "agent.config.refinement.num_refinement_stage=${num_refinement_stage}"
    "agent.config.refinement.stage_layers=${stage_layers}"
    "agent.config.refinement.topks=${topks}"
    "experiment_name=${dir}/test-${file}-pdms-v1-${inference_model}"
    "metric_cache_path=${cache}"
)
if [[ "$mode" == "all" || "$mode" == "cache" ]]; then
    echo "--- Preparing separate NAVSIM v1 cache (5 s observations) ---"
    printf '%q ' "${cmd[@]}" "+navsim_v1_mode=cache"; printf '\n'
    "${cmd[@]}" "+navsim_v1_mode=cache"
fi
if [[ "$mode" == "all" || "$mode" == "evaluate" ]]; then
    echo "--- Evaluating NAVSIM v1 / PDMS ---"
    printf '%q ' "${cmd[@]}" "+navsim_v1_mode=evaluate"; printf '\n'
    "${cmd[@]}" "+navsim_v1_mode=evaluate"
fi
