#!/usr/bin/env bash
set -euo pipefail

stage=${1:-}
if [[ "$stage" != export && "$stage" != score ]]; then
    echo "Usage: bash $0 export|score" >&2
    exit 2
fi
: "${NAVSIM_DEVKIT_ROOT:?NAVSIM_DEVKIT_ROOT is not set}"
: "${NAVSIM_EXP_ROOT:?NAVSIM_EXP_ROOT is not set}"
: "${OPENSCENE_DATA_ROOT:?OPENSCENE_DATA_ROOT is not set}"

checkpoint=${DRIVEMEM_CHECKPOINT:-${NAVSIM_EXP_ROOT}/model_ckpt/drivesuprim_r34.ckpt}
metric_cache=${DRIVEMEM_METRIC_CACHE:-${NAVSIM_EXP_ROOT}/metric_cache/test/ori}
output=${DRIVEMEM_OUTPUT:-${NAVSIM_EXP_ROOT}/drivemem_oracle/pilot}
reference_csv=${DRIVEMEM_REFERENCE_CSV:-}
scene_count=${DRIVEMEM_SCENES:-8}
candidate_limit=${DRIVEMEM_CANDIDATES:-16}
split=${DRIVEMEM_SPLIT:-navtest_debug}
sampling=${DRIVEMEM_SAMPLING:-first}

[[ -f "$checkpoint" ]] || { echo "Checkpoint not found: $checkpoint" >&2; exit 1; }
[[ -d "$metric_cache/metadata" ]] || { echo "Metric cache metadata not found: $metric_cache" >&2; exit 1; }
if [[ -n "$reference_csv" && ! -f "$reference_csv" ]]; then
    echo "Reference CSV not found: $reference_csv" >&2
    exit 1
fi
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"
cd "$NAVSIM_DEVKIT_ROOT"

cmd=(python -u navsim/planning/script/run_drivemem_oracle_pilot.py
    agent=drivesuprim_agent_r34
    "train_test_split=$split"
    "train_test_split.scene_filter.max_scenes=$scene_count"
    "agent.checkpoint_path=$checkpoint"
    agent.config.training=false
    agent.config.only_ori_input=true
    agent.config.inference.model=teacher
    +agent.config.inference.export_candidates=true
    agent.config.refinement.use_multi_stage=true
    agent.config.refinement.num_refinement_stage=1
    agent.config.refinement.stage_layers=3
    agent.config.refinement.topks=256
    "metric_cache_path=$metric_cache"
    scorer.config.human_penalty_filter=true
    scorer.config.progress_weight=5.0
    scorer.config.ttc_weight=5.0
    scorer.config.lane_keeping_weight=2.0
    scorer.config.history_comfort_weight=2.0
    scorer.config.two_frame_extended_comfort_weight=2.0
    "experiment_name=drivemem_oracle/hydra_${stage}"
    "+drivemem.stage=$stage"
    "+drivemem.output_dir=$output"
    "+drivemem.candidate_limit=$candidate_limit"
    "+drivemem.reference_csv=$reference_csv"
    "+drivemem.sampling=$sampling"
    "+drivemem.n_logs=${DRIVEMEM_LOGS:-32}"
    "+drivemem.scenes_per_log=${DRIVEMEM_SCENES_PER_LOG:-2}"
    "+drivemem.min_gap_s=${DRIVEMEM_MIN_GAP_S:-10}"
    "+drivemem.seed=${DRIVEMEM_SEED:-20261027}"
    "+drivemem.excluded_logs=${DRIVEMEM_EXCLUDED_LOGS:-[]}"
)

echo "Stage: $stage; output: $output; split: $split; scenes: $scene_count; candidates: $candidate_limit"
printf 'Command: '; printf '%q ' "${cmd[@]}"; printf '\n'
env -u WORLD_SIZE -u RANK -u LOCAL_RANK -u MASTER_ADDR -u MASTER_PORT \
    -u TORCHELASTIC_RUN_ID -u TORCHELASTIC_RESTART_COUNT -u PL_TORCH_DISTRIBUTED_BACKEND \
    "${cmd[@]}"
