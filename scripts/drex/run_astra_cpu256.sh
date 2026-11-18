#!/usr/bin/env bash
set -euo pipefail
tools=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export ASTRA_CPU256_ROOT=${ASTRA_CPU256_ROOT:-/workspace/exp_v2/drex_astra_cpu256/controls_oracle64_v1}
export ASTRA_CPU256_SOURCE=${ASTRA_CPU256_SOURCE:-/workspace/exp_v2/drex_rollout_audit/rollout_audit_pool64_v1}
export ASTRA_CPU256_PACKAGE=${ASTRA_CPU256_PACKAGE:-/workspace/exp_v2/drex_transfer/navsim_train_v1}
export ASTRA_CPU256_REPO=${ASTRA_CPU256_REPO:-/workspace/DriveSuprim}
export PYTHONPATH="$ASTRA_CPU256_SOURCE/tools${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
case "${1:-status}" in
  start)
    mkdir -p "$ASTRA_CPU256_ROOT"
    exec python - "$tools/run_astra_cpu256.sh" "$ASTRA_CPU256_ROOT" <<'PY'
import fcntl,json,subprocess,sys,time
from pathlib import Path
script,folder=sys.argv[1:]; root=Path(folder)
with (root/'.launch.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    with (root/'.pipeline.lock').open('a') as running:
        try:fcntl.flock(running,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:print('Already running: '+folder);raise SystemExit(0)
        fcntl.flock(running,fcntl.LOCK_UN)
    p=subprocess.Popen(['bash',script,'--worker'],start_new_session=True,stdin=subprocess.DEVNULL,
                       stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,close_fds=True)
    time.sleep(.3)
    if p.poll() is not None:raise SystemExit('Worker exited; inspect '+str(root/'run.log'))
    print(json.dumps(dict(status='launched',pid=p.pid,log=str(root/'run.log'))))
PY
    ;;
  --worker)
    exec >>"$ASTRA_CPU256_ROOT/run.log" 2>&1
    exec 9>"$ASTRA_CPU256_ROOT/.pipeline.lock"; flock -n 9 || exit 0
    python -B -m unittest discover -s "$tools" -p test_astra_cpu256.py -v
    if [[ -f "$tools/test_astra_night.py" ]]; then
      python -B -m unittest discover -s "$tools" -p test_astra_night.py -v
    fi
    args=(--package "$ASTRA_CPU256_PACKAGE" --source "$ASTRA_CPU256_SOURCE" --out "$ASTRA_CPU256_ROOT" --repo "$ASTRA_CPU256_REPO")
    [[ -z ${ASTRA_CPU256_DECISIONS:-} ]] || args+=(--decisions "$ASTRA_CPU256_DECISIONS")
    [[ ${ASTRA_CPU256_ORACLE_ALL:-0} != 1 ]] || args+=(--oracle-all)
    [[ -z ${ASTRA_CPU256_REUSE:-} ]] || args+=(--reuse "$ASTRA_CPU256_REUSE")
    python -u "$tools/astra_cpu256.py" pilot "${args[@]}"
    python -u "$tools/astra_cpu256.py" full "${args[@]}" ;;
  status)
    echo "Output: $ASTRA_CPU256_ROOT"
    if [[ -d "$ASTRA_CPU256_ROOT" ]]; then
      python - "$ASTRA_CPU256_ROOT" <<'PY'
import fcntl,json,sys
from pathlib import Path
with (Path(sys.argv[1])/'.pipeline.lock').open('a') as lock:
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state='idle'
    except BlockingIOError:state='running'
print(json.dumps(dict(pipeline_state=state)))
PY
    fi
    [[ ! -f "$ASTRA_CPU256_ROOT/run.log" ]] || tail -n 12 "$ASTRA_CPU256_ROOT/run.log"
    for file in progress.json report.json failure.json; do
      [[ ! -f "$ASTRA_CPU256_ROOT/$file" ]] || python - "$ASTRA_CPU256_ROOT/$file" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]));print(json.dumps({k:v for k,v in r.items() if k not in ('methods','note')},indent=2))
PY
    done ;;
  *) echo 'Usage: run_astra_cpu256.sh start|status' >&2;exit 2 ;;
esac
