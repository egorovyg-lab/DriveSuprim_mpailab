#!/usr/bin/env bash
set -euo pipefail

stage=${1:-all}
case "$stage" in all|export|score|report) ;; *) echo "Usage: $0 all|export|score|report" >&2; exit 2 ;; esac
: "${NAVSIM_DEVKIT_ROOT:?NAVSIM_DEVKIT_ROOT is not set}"
: "${NAVSIM_EXP_ROOT:?NAVSIM_EXP_ROOT is not set}"

export DRIVEMEM_SAMPLING=logs
export DRIVEMEM_SPLIT="${DRIVEMEM_SPLIT:-navtest}"
export DRIVEMEM_SCENES=null
export DRIVEMEM_LOGS="${DRIVEMEM_LOGS:-32}"
export DRIVEMEM_SCENES_PER_LOG="${DRIVEMEM_SCENES_PER_LOG:-2}"
export DRIVEMEM_MIN_GAP_S="${DRIVEMEM_MIN_GAP_S:-10}"
export DRIVEMEM_SEED="${DRIVEMEM_SEED:-20261027}"
export DRIVEMEM_CANDIDATES="${DRIVEMEM_CANDIDATES:-256}"
export DRIVEMEM_EXCLUDED_LOGS="${DRIVEMEM_EXCLUDED_LOGS:-[2021.06.03.12.02.06_veh-35_01100_01227]}"
export DRIVEMEM_OUTPUT="${DRIVEMEM_OUTPUT:-${NAVSIM_EXP_ROOT}/drivemem_oracle/multilog32x2_seed20261027}"

# Explicitly set empty string to disable reference comparison; missing default is an error.
export DRIVEMEM_REFERENCE_CSV="${DRIVEMEM_REFERENCE_CSV-/workspace/exp_v2/model_ckpt/test-drivesuprim_r34-epdms/2026.10.24.15.13.06/2026.10.24.16.41.17.csv}"

cd "$NAVSIM_DEVKIT_ROOT"
launcher=scripts/drivesuprim/evaluation/run_drivemem_oracle_pilot.sh
if [[ "$stage" == all || "$stage" == export ]]; then
    bash "$launcher" export
fi
if [[ "$stage" == all || "$stage" == score ]]; then
    bash "$launcher" score
fi
if [[ "$stage" == all || "$stage" == report ]]; then
    python - "$DRIVEMEM_OUTPUT/scoring_k$DRIVEMEM_CANDIDATES" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
report = json.loads((path / 'report.json').read_text())
print(json.dumps(report, indent=2))
print('\nResults per log:')
print((path / 'log_summary.csv').read_text())
print('Full per-scene results:', path / 'scene_summary.csv')
PY
fi
