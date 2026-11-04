#!/usr/bin/env bash
# Exercise both real wrappers; stub only Python, without opening any experiment.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
scratch=$(mktemp -d)
trap 'rm -f -- "$scratch/python" "$scratch/checkpoint" "$scratch/args" "$scratch/output"; rmdir -- "$scratch"' EXIT
cat > "$scratch/python" <<'STUB'
#!/usr/bin/env bash
printf '%s\n' "$@" > "$DRIVEMEM_TEST_ARGS"
STUB
chmod +x "$scratch/python"
touch "$scratch/checkpoint"
cd "$repo"
has_arg() {
    local expected=$1 item
    for item in "${args[@]}"; do
        [[ "$item" != "$expected" ]] || return 0
    done
    echo "Missing Python argument: $expected" >&2
    return 1
}
for stage in utility-plan utility-fit utility-evaluate utility-report utility-all; do
    env PATH="$scratch:$PATH" NAVSIM_DEVKIT_ROOT="$repo" NAVSIM_EXP_ROOT="$scratch" \
        OPENSCENE_DATA_ROOT="$scratch/data" NUPLAN_MAPS_ROOT="$scratch/maps" \
        DRIVEMEM_CHECKPOINT="$scratch/checkpoint" DRIVEMEM_TEST_ARGS="$scratch/args" \
        DRIVEMEM_MEMORY_ROOT="$scratch/source" DRIVEMEM_MEMORY_SESSIONS=6 \
        DRIVEMEM_FIT_SESSIONS=4 DRIVEMEM_DEV_SESSIONS=3 \
        DRIVEMEM_CAL_SESSIONS=2 DRIVEMEM_TEST_SESSIONS=2 DRIVEMEM_BUDGET_SCENES=8 \
        bash scripts/drivesuprim/evaluation/run_drivemem_budget.sh "$stage" > "$scratch/output" 2>&1 \
        || { cat "$scratch/output" >&2; exit 1; }
    mapfile -t args < "$scratch/args"
    [[ "${args[0]}" == -u && "${args[1]}" == navsim/planning/script/run_drivemem_pair_utility.py ]]
    has_arg "drivemem.stage=$stage"
    has_arg "drivemem.root=$scratch/source"
    has_arg "drivemem.sessions.memory=6"
    has_arg "drivemem.sessions.fit=4"
    has_arg "drivemem.sessions.development=3"
    has_arg "drivemem.scenes.memory=8"
    has_arg "drivemem.scenes.fit=8"
done
echo "Utility launcher routing: all five stages OK"
