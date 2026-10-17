#!/bin/bash

dir=$1
file=$2
num_refinement_stage=$3
stage_layers=$4
topks=$5
agent=${6:-"drivesuprim_agent_r34"}
inference_model=${7:-"teacher"}
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}"
threads=${THREADS_PER_NODE:-16}

metric_cache_path="${NAVSIM_EXP_ROOT}/metric_cache/test/ori"

if [ "$inference_model" = "teacher" ]; then
    experiment_name="${dir}/test-${file}"
else
    experiment_name="${dir}/test-${file}-${inference_model}"
fi

command_string="${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_one_stage_gpu_ssl.py \
    +debug=true \
    +use_pdm_closed=false \
    agent=$agent \
    train_test_split=navtest_debug \
    worker.threads_per_node=$threads \
    agent.checkpoint_path='${NAVSIM_EXP_ROOT}/${dir}/${file}.ckpt' \
    agent.config.training=false \
    agent.config.only_ori_input=true \
    agent.config.inference.model=${inference_model} \
    agent.config.inference.save_pickle=true \
    agent.config.refinement.use_multi_stage=true \
    agent.config.refinement.num_refinement_stage=$num_refinement_stage \
    agent.config.refinement.stage_layers=$stage_layers \
    agent.config.refinement.topks=$topks \
    experiment_name=${experiment_name} \
    +cache_path=null \
    metric_cache_path=${metric_cache_path}
"

echo "--- COMMAND ---"
echo $command_string
echo -e "\n\n"
