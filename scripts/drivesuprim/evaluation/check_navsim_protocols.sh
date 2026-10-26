#!/usr/bin/env bash
set -euo pipefail

# Run inside the project container. Override paths explicitly if another run is intended.
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
v1_run="${NAVSIM_V1_RUN:-/workspace/exp_v2/model_ckpt/test-drivesuprim_r34-pdms-v1-teacher/2026.10.25.20.06.06}"
v2_csv="${NAVSIM_V2_CSV:-/workspace/exp_v2/model_ckpt/test-drivesuprim_r34-epdms/2026.10.24.15.13.06/2026.10.24.16.41.17.csv}"
export TMPDIR="${TMPDIR:-/tmp}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray}"
mkdir -p "$TMPDIR" "$RAY_TMPDIR"
cd "$project_root"
exec python3 -u scripts/drivesuprim/evaluation/diagnose_navsim_protocols.py \
    "$v1_run" --v2-csv "$v2_csv" "$@"
