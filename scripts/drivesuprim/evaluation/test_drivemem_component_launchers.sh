#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
scratch=$(mktemp -d)
trap 'rm -f -- "$scratch/python" "$scratch/trace"; rmdir -- "$scratch"' EXIT
cat > "$scratch/python" <<'STUB'
#!/usr/bin/env bash
printf '%s\n' "$*" > "$DRIVEMEM_TEST_TRACE"
STUB
chmod +x "$scratch/python"
for stage in plan fit evaluate report all; do
    env PATH="$scratch:$PATH" NAVSIM_DEVKIT_ROOT="$repo" NAVSIM_EXP_ROOT="$scratch" \
      DRIVEMEM_TEST_TRACE="$scratch/trace" \
      bash "$repo/scripts/drivesuprim/evaluation/run_drivemem_component_utility.sh" "$stage"
    trace=$(< "$scratch/trace")
    [[ "$trace" == *"run_drivemem_component_utility.py $stage"* ]]
    [[ "$trace" == *"--prior $scratch/drivemem_memory/experiments/pair_utility_v1"* ]]
    [[ "$trace" == *"--addition $scratch/drivemem_memory/mf_new4_shard174_v1"* ]]
    [[ "$trace" == *"--out $scratch/drivemem_memory/experiments/component_utility_v1"* ]]
done
echo "Component utility launcher: all five stages OK"
