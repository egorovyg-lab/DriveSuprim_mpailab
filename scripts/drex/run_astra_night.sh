#!/usr/bin/env bash
set -euo pipefail
tools=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export ASTRA_CPU256_ROOT=${ASTRA_CPU256_ROOT:-/workspace/exp_v2/drex_astra_cpu256/oracle651_v2}
export ASTRA_CPU256_REUSE=${ASTRA_CPU256_REUSE:-/workspace/exp_v2/drex_astra_cpu256/controls_oracle64_v1}
export ASTRA_CPU256_ORACLE_ALL=1
export ASTRA_CPU256_SOURCE=${ASTRA_CPU256_SOURCE:-/workspace/exp_v2/drex_rollout_audit/rollout_audit_pool64_v1}
export PYTHONPATH="$ASTRA_CPU256_SOURCE/tools${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
[[ -z ${ASTRA_CPU256_DECISIONS:-} ]] || { echo 'Oracle-all cannot consume model decisions' >&2; exit 2; }
case "${1:-status}" in
  tests)
    python -B -m unittest discover -s "$tools" -p test_astra_cpu256.py -v
    python -B -m unittest discover -s "$tools" -p test_astra_night.py -v ;;
  probe)
    # Exit 2 is an explicit capacity block, NOT a successful GPU smoke test.
    exec python -B "$tools/astra_drex_probe.py" --out "$ASTRA_CPU256_ROOT/drex_probe" ;;
  start)
    [[ "$ASTRA_CPU256_ROOT" != "$ASTRA_CPU256_REUSE" ]] || { echo 'Output must differ from completed source' >&2; exit 2; }
    exec bash "$tools/run_astra_cpu256.sh" start ;;
  status) exec bash "$tools/run_astra_cpu256.sh" status ;;
  *) echo 'Usage: run_astra_night.sh tests|probe|start|status' >&2; exit 2 ;;
esac
