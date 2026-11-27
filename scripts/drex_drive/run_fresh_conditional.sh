#!/usr/bin/env bash
# Explicit operator entry point. Nothing is submitted or transferred by this script.
set -euo pipefail
umask 077

usage() {
  cat <<'EOF'
Usage: bash scripts/drex_drive/run_fresh_conditional.sh MODE [ARGS...]

  synthetic                          Local or MSU270 CPU: synthetic A/C and archived B/C tests.
  astra-check                        Astra CPU: import and synthetic replay under Python 3.9.23.
  archive-predict [runner options]    MSU270 GPU allocation: one archived A/C pair, no fresh data.
  archive-score [runner options]      Astra CPU: separately verify the archived smoke.
  freeze [runner options]             Approved data zone: attest and freeze 120 eligible drives.
  plan-check MANIFEST SHA256          Offline: check a pinned manifest and print counts and time caps.
  predict [runner options]            MSU270 GPU allocation: complete pinned fresh A/C inference.
  score [runner options]              Astra CPU: score prediction pairs on matching caches.
  validate [runner options]           Approved CPU zone: final paired decision, without caches.

Set DREX_PYTHON to an approved absolute interpreter path. GPU roles also need the
pinned upstream on PYTHONPATH; Astra scoring needs the pinned tools and NAVSIM.
Pass MODE --help for the underlying runner's options. See FRESH_CONDITIONAL.md
for exact inputs and stage order. There is no all/auto mode and no automatic
GPU submission, connection to Astra, or transmission of any artifact.
EOF
}

fail() {
  printf '%s\n' "$*" >&2
  exit 2
}

mode=${1:-help}
case "$mode" in
  -h|--help|help) usage; exit 0 ;;
  synthetic|astra-check)
    [[ $# == 1 ]] || fail "$mode takes no arguments" ;;
  plan-check)
    [[ $# == 3 ]] || fail 'Usage: plan-check MANIFEST SHA256'
    [[ $2 == /* && $3 =~ ^[0-9a-f]{64}$ ]] || fail 'plan-check requires an absolute manifest and a pinned SHA-256' ;;
  archive-predict|predict|archive-score|score|freeze|validate)
    [[ $# -ge 2 ]] || fail "Missing $mode inputs; see $mode --help and FRESH_CONDITIONAL.md" ;;
  *) fail "Unknown mode: $mode; see --help" ;;
esac
shift

checkout=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd "$checkout"
[[ ${DREX_PYTHON:-} == /* && -x ${DREX_PYTHON:-} ]] ||
  fail 'Set DREX_PYTHON to the approved existing absolute Python executable'
export PYTHONDONTWRITEBYTECODE=1

runner="$checkout/scripts/drex_drive/fresh_conditional.sh"
if [[ $# == 1 && ( $1 == --help || $1 == -h ) ]]; then
  exec bash "$runner" "$mode" "$@"
fi

case "$mode" in
  synthetic)
    export CUDA_VISIBLE_DEVICES=""
    exec "$DREX_PYTHON" -B -m unittest \
      scripts.drex_drive.test_fresh_conditional scripts.drex_drive.test_lead_brake_gap_bc -v ;;
  astra-check)
    export CUDA_VISIBLE_DEVICES=""
    "$DREX_PYTHON" -B -c 'import platform; import sys; sys.exit(0 if platform.python_version() == "3.9.23" else "Expected the pinned Astra Python 3.9.23")' ||
      fail 'Astra interpreter does not match the archived scorer'
    exec "$DREX_PYTHON" -B -m unittest \
      scripts.drex_drive.test_fresh_conditional.FreshConditional.test_astra_replay_imports_do_not_evaluate_union_annotations -v ;;
  plan-check)
    exec "$DREX_PYTHON" -B - "$@" <<'PY'
from pathlib import Path
import sys
from scripts.drex_drive import fresh_conditional as fc

try:
    plan, _ = fc.load_plan(fc.canonical(Path(sys.argv[1])), sys.argv[2])
except fc.ERRORS as exc:
    print(f"Not ready: {exc}", file=sys.stderr)
    sys.exit(2)
print(f"attempt={plan['attempt']} drives={len(plan['session_ids'])} scenes={len(plan['scenes'])}")
print(f"msu_wall_seconds={plan['limits']['msu_wall_seconds']} astra_wall_seconds={plan['limits']['astra_wall_seconds']}")
print("Manifest checks only; no inference or scorer ran.")
PY
    ;;
  archive-predict|predict)
    [[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]] ||
      fail "$mode requires a separately approved MSU270 Slurm GPU allocation"
    exec bash "$runner" "$mode" "$@" ;;
  archive-score|score)
    export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
    exec bash "$runner" "$mode" "$@" ;;
  freeze|validate)
    exec bash "$runner" "$mode" "$@" ;;
esac
