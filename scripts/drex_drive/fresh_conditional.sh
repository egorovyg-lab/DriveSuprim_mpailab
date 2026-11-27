#!/usr/bin/env bash
# Manual worker entry point; never creates a job or connects to another host.
set -euo pipefail

checkout=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd "$checkout"
: "${DREX_PYTHON:?set DREX_PYTHON to the approved existing absolute Python executable}"
[[ "$DREX_PYTHON" == /* && -x "$DREX_PYTHON" ]] || { echo 'Invalid DREX_PYTHON' >&2; exit 2; }
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=20261124
exec "$DREX_PYTHON" -B -m scripts.drex_drive.fresh_conditional "$@"
