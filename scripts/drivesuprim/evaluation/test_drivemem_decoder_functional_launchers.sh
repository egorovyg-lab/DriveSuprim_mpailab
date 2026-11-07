#!/usr/bin/env bash
set -euo pipefail
trap 'printf "Launcher test failed: %s stage=%s line=%s\n" "${BASH_SOURCE[0]}" "${stage:-setup}" "$LINENO" >&2' ERR
# Test defaults in a child shell, independently of the caller's experiment.
for task_env_name in "${!DRIVEMEM_@}"; do
    unset "$task_env_name"
done
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
scratch=$(mktemp -d)
trap 'rm -f -- "$scratch/python" "$scratch/trace" "$scratch/checkpoint" "$scratch/output"; rmdir -- "$scratch"' EXIT
cat > "$scratch/python" <<'STUB'
#!/usr/bin/env bash
printf 'ENV=%s,%s,%s LIMIT=%s %s\n' "$CUBLAS_WORKSPACE_CONFIG" "$PYTHONHASHSEED" "$CUDA_VISIBLE_DEVICES" "${DRIVEMEM_DECODER_EXPORT_LIMIT:-unset}" "$*" > "$DRIVEMEM_TEST_TRACE"
STUB
chmod +x "$scratch/python"
touch "$scratch/checkpoint"
for stage in plan export-pilot export export-status fit evaluate report all; do
    env PATH="$scratch:$PATH" NAVSIM_DEVKIT_ROOT="$repo" NAVSIM_EXP_ROOT="$scratch" \
      OPENSCENE_DATA_ROOT="$scratch/data" NUPLAN_MAPS_ROOT="$scratch/maps" \
      DRIVEMEM_CHECKPOINT="$scratch/checkpoint" DRIVEMEM_TEST_TRACE="$scratch/trace" \
      bash "$repo/scripts/drivesuprim/evaluation/run_drivemem_decoder_functional.sh" "$stage" > "$scratch/output" 2>&1 \
      || { cat "$scratch/output" >&2; exit 1; }
    trace=$(< "$scratch/trace")
    [[ "$trace" == "ENV=:4096:8,20261027,0 "* ]]
    case "$stage" in
      plan) [[ "$trace" == *"run_drivemem_decoder_functional_export.py"* && "$trace" == *"drivemem.stage=decoder-functional-plan"* ]] ;;
      export-pilot|export) [[ "$trace" == *"run_drivemem_decoder_functional_export.py"* && "$trace" == *"drivemem.stage=decoder-functional-export"* ]] ;;
      export-status) [[ "$trace" == *"run_drivemem_decoder_functional_export.py"* && "$trace" == *"drivemem.stage=decoder-functional-status"* ]] ;;
      *) [[ "$trace" == *"run_drivemem_decoder_utility.py $stage"* && "$trace" == *"--decoder $scratch/drivemem_memory/decoder_features_functional_v1"* && "$trace" == *"--out $scratch/drivemem_memory/experiments/component_decoder_functional_v1"* ]] ;;
    esac
    if [[ "$stage" == export-pilot ]]; then [[ "$trace" == *"LIMIT=4"* ]]; fi
    if [[ "$trace" == *"drivemem.stage="* ]]; then
        [[ "$trace" == *"drivemem.sessions.memory=6"* && "$trace" == *"drivemem.scenes.memory=8"* ]]
        [[ "$trace" == *"drivemem.root=$scratch/drivemem_memory/budget20gb_v1"* ]]
    fi
done
echo "Functional decoder launcher: all eight stages OK"
