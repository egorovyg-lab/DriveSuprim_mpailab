#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
scratch=$(mktemp -d)
trap 'rm -f -- "$scratch/python" "$scratch/checkpoint" "$scratch/trace" "$scratch/output"; rmdir -- "$scratch"' EXIT
cat > "$scratch/python" <<'STUB'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$DRIVEMEM_TEST_TRACE"
STUB
chmod +x "$scratch/python"
touch "$scratch/checkpoint"
cd "$repo"
for stage in plan images images-status cache cache-status export screen label-pilot label support-report prepare; do
    : > "$scratch/trace"
    env PATH="$scratch:$PATH" NAVSIM_DEVKIT_ROOT="$repo" NAVSIM_EXP_ROOT="$scratch" \
        OPENSCENE_DATA_ROOT="$scratch/data" NUPLAN_MAPS_ROOT="$scratch/maps" \
        DRIVEMEM_CHECKPOINT="$scratch/checkpoint" DRIVEMEM_TEST_TRACE="$scratch/trace" \
        DRIVEMEM_NEW_MF_ROOT="$scratch/new_mf" \
        bash scripts/drivesuprim/evaluation/run_drivemem_new_mf.sh "$stage" > "$scratch/output" 2>&1 \
        || { cat "$scratch/output" >&2; exit 1; }
    trace=$(< "$scratch/trace")
    case "$stage" in
      plan) [[ "$trace" == *"download/prepare_drivemem_new_mf.py"* && "$trace" == *"drivemem.stage=data-validate"* ]] ;;
      images|images-status) [[ "$trace" == *"fetch_drivemem_budget_images.py"* && "$trace" == *"--max-gb 8.5"* ]] ;;
      cache|export) [[ "$trace" == *"drivemem.stage=$stage"* ]] ;;
      cache-status) [[ "$trace" == *"check_drivemem_cache.py"* ]] ;;
      screen|support-report) [[ "$trace" == *"drivemem.stage=data-$stage"* ]] ;;
      label*) [[ "$trace" == *"drivemem.stage=data-label"* ]] ;;
      prepare) [[ "$trace" == *"drivemem.stage=data-screen"* && "$trace" == *"--max-gb 8.5"* ]] ;;
    esac
    if [[ "$trace" == *"drivemem.stage="* ]]; then
        [[ "$trace" == *"drivemem.sessions.memory=2"* && "$trace" == *"drivemem.sessions.fit=2"* ]]
        [[ "$trace" == *"drivemem.sessions.development=0"* && "$trace" == *"drivemem.sessions.test=0"* ]]
        [[ "$trace" == *"drivemem.root=$scratch/new_mf"* && "$trace" == *"drivemem.scenes.fit=16"* ]]
    fi
done
echo "New-MF launcher routing: all eleven stages OK"
