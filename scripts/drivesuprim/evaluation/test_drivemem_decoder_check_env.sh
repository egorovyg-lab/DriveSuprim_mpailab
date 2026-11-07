#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
scratch=$(mktemp -d)
trap 'rm -f -- "$scratch/python" "$scratch/output"; rmdir -- "$scratch"' EXIT
cat > "$scratch/python" <<'STUB'
#!/usr/bin/env bash
exit 0
STUB
chmod +x "$scratch/python"
for variant in decoder decoder_replay decoder_deterministic decoder_numeric decoder_functional; do
    if ! env PATH="$scratch:$PATH" NAVSIM_DEVKIT_ROOT="$repo" NAVSIM_EXP_ROOT="$scratch" \
        DRIVEMEM_DECODER_FEATURE_ROOT="$scratch/inherited-features" \
        DRIVEMEM_DECODER_MODEL_ROOT="$scratch/inherited-model" \
        DRIVEMEM_FROZEN_SOURCE_ROOT="$scratch/inherited-source" \
        DRIVEMEM_PRIOR_UTILITY_ROOT="$scratch/inherited-prior" \
        DRIVEMEM_NEW_MF_ROOT="$scratch/inherited-addition" \
        DRIVEMEM_DECODER_EXPORT_LIMIT=7 DRIVEMEM_DECODER_GPU=1 \
        bash "$repo/scripts/drivesuprim/evaluation/run_drivemem_${variant}.sh" check > "$scratch/output" 2>&1; then
        printf 'Nested check failed with inherited experiment settings: %s\n' "$variant" >&2
        cat "$scratch/output" >&2
        exit 1
    fi
    printf 'Nested %s check: inherited settings isolated OK\n' "$variant"
done
