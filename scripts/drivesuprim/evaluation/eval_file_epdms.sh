#!/usr/bin/env bash
set -euo pipefail

if (( $# < 5 || $# > 7 )); then
    echo "Использование: $0 DIR FILE NUM_REFINEMENT_STAGE STAGE_LAYERS TOPKS [AGENT] [INFERENCE_MODEL]" >&2
    exit 2
fi

: "${NAVSIM_DEVKIT_ROOT:?Не задана переменная NAVSIM_DEVKIT_ROOT}"
: "${NAVSIM_EXP_ROOT:?Не задана переменная NAVSIM_EXP_ROOT}"

dir=$1
file=$2
num_refinement_stage=$3
stage_layers=$4
topks=$5
agent=${6:-drivesuprim_agent_r34}
inference_model=${7:-teacher}
threads=${THREADS_PER_NODE:-4}

export TMPDIR="${TMPDIR:-/tmp}"
export RAY_TMPDIR="${RAY_TMPDIR:-${TMPDIR}/ray}"
mkdir -p "$TMPDIR" "$RAY_TMPDIR"

# CUDA allocator setting for PyTorch 2.0.1.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"

export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

metric_cache_path="${NAVSIM_EXP_ROOT}/metric_cache/test/ori"
checkpoint_path="${NAVSIM_EXP_ROOT}/${dir}/${file}.ckpt"

if [[ ! -f "$checkpoint_path" ]]; then
    echo "Не найден checkpoint: $checkpoint_path" >&2
    exit 1
fi

if [[ "$inference_model" == "teacher" ]]; then
    experiment_name="${dir}/test-${file}-epdms"
else
    experiment_name="${dir}/test-${file}-epdms-${inference_model}"
fi

cmd=(
    python3
    "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_one_stage_gpu_ssl.py"
    "+debug=false"
    "+use_pdm_closed=false"
    "++trainer.params.devices=${GPUS_PER_NODE:-2}"
    "agent=${agent}"
    "train_test_split=navtest"
    "worker.threads_per_node=${threads}"
    "dataloader.params.batch_size=${BATCH_SIZE:-1}"
    "agent.checkpoint_path=${checkpoint_path}"
    "agent.config.training=false"
    "agent.config.only_ori_input=true"
    "agent.config.inference.model=${inference_model}"
    "agent.config.inference.save_pickle=false"
    "agent.config.refinement.use_multi_stage=true"
    "agent.config.refinement.num_refinement_stage=${num_refinement_stage}"
    "agent.config.refinement.stage_layers=${stage_layers}"
    "agent.config.refinement.topks=${topks}"
    "experiment_name=${experiment_name}"
    "+cache_path=null"
    "metric_cache_path=${metric_cache_path}"
    "scorer.config.human_penalty_filter=true"
    "scorer.config.lane_keeping_weight=2.0"
    "scorer.config.history_comfort_weight=2.0"
    "scorer.config.two_frame_extended_comfort_weight=2.0"
    "scorer.config.progress_weight=5.0"
    "scorer.config.ttc_weight=5.0"
)

echo "--- COMMAND ---"
printf '%q ' "${cmd[@]}"
printf '\n'

env \
    -u WORLD_SIZE \
    -u RANK \
    -u LOCAL_RANK \
    -u MASTER_ADDR \
    -u MASTER_PORT \
    -u TORCHELASTIC_RUN_ID \
    -u TORCHELASTIC_RESTART_COUNT \
    -u PL_TORCH_DISTRIBUTED_BACKEND \
    "${cmd[@]}"