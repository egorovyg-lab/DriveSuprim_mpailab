#!/usr/bin/env bash
set -euo pipefail
umask 077

module=scripts.drex_drive.navsim_night

check_attempt_path() {
  local parent="${out%/*}"
  if [[ "$parent" != "$runtime/runs/corridor-preview-lead-brake-gap-v1" ||
        ! ${out##*/} =~ ^attempt-[0-9]{3,}$ ||
        "$out" != "$(realpath -m -- "$out")" ||
        "$archive" != "$(realpath -e -- "$archive")" ||
        "$archive" != "$(realpath -e -- "$runtime/runs/navsim-straight-night-v1")" ||
        "$out" == "$archive" || "$out" == "$archive/"* || "$archive" == "$out/"* ]]; then
    echo 'Refusing noncanonical attempt or overlap with the read-only archive' >&2
    return 2
  fi
}

lock_attempt() {
  # mkdir does not dereference an existing lock symlink or truncate a file.
  mkdir -- "$out/.lead-brake-gap.lock" || { echo "Attempt busy or interrupted: $out" >&2; return 2; }
  trap 'rmdir -- "$out/.lead-brake-gap.lock"' EXIT
}

run_logged_worker() {
  # Slurm itself only writes to /dev/null; open the real log here, after
  # validation. Walk the absolute path using directory descriptors so replacing
  # an ancestor with a symlink cannot redirect the log into the archive.
  "$runtime/venv/bin/python" -B -c '
import os
from pathlib import Path
import sys

log_dir, job, *command = sys.argv[1:]
path = Path(log_dir)
if not path.is_absolute() or not job.isascii() or not job.isdecimal():
    raise SystemExit("Invalid log directory or Slurm job ID")
flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
parent = os.open("/", flags)
for name in path.parts[1:]:
    if name == "..":
        raise SystemExit("Noncanonical worker log directory")
    child = os.open(name, flags, dir_fd=parent)
    os.close(parent)
    parent = child
info = os.fstat(parent)
if info.st_uid != os.geteuid() or info.st_mode & 0o077:
    raise SystemExit("Worker log directory is not private")
log = os.open(f"slurm-{job}.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
os.close(parent)
os.dup2(log, 1)
os.dup2(log, 2)
os.close(log)
os.execvp(command[0], command)
' "$out/$stage-slurm" "$SLURM_JOB_ID" "$@"
}

if [[ ${1:-} == worker ]]; then
  stage=$2 repo=$3 runtime=$4 archive=$5 out=$6 provenance=$7 head=$8 diff_sha=$9
  case "$stage" in brake-gap-gpu-smoke|brake-gap-predict) ;; *) exit 2 ;; esac
  check_attempt_path
  [[ -n ${SLURM_JOB_ID:-} && -d "$out/$stage-slurm" && ! -L "$out/$stage-slurm" ]] || exit 2
  cd "$repo"
  lock_attempt
  export PYTHONPATH="$runtime/upstream"
  run_logged_worker srun --ntasks=1 "$runtime/venv/bin/python" -B -m "$module" "$stage" \
    --out "$out" --archive "$archive" --provenance "$provenance" \
    --weights "$runtime/models/drex-max-v1" --upstream "$runtime/upstream" \
    --git-head "$head" --git-diff-sha256 "$diff_sha"
  exit 0
fi

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd "$repo"
script="$repo/scripts/drex_drive/lead_brake_gap.sh"
runtime=$(realpath -e -- "${DREX_RUNTIME_ROOT:-/scratch/pr0201/drex_runtime}")
archive=$(realpath -e -- "$runtime/runs/navsim-straight-night-v1")
python="$runtime/venv/bin/python"
attempt=${2:-attempt-001}
[[ $attempt =~ ^attempt-[0-9]{3,}$ ]] || { echo 'Use attempt-NNN' >&2; exit 2; }
out="$runtime/runs/corridor-preview-lead-brake-gap-v1/$attempt"
check_attempt_path

case "${1:-}" in
  check)
    for path in observations.json corridor-preview-input.json corridor-preview.json \
                corridor-preview/identity.json control/identity.json; do
      [[ -s "$archive/$path" ]] || { echo "Missing archive input: $path" >&2; exit 2; }
    done
    [[ -x "$python" && -f "$runtime/upstream/drex_max/model.py" \
       && -d "$runtime/models/drex-max-v1" ]] || { echo 'Missing Drex runtime or weights' >&2; exit 2; }
    "$python" --version
    "$python" -B -c 'import numpy; print("NumPy", numpy.__version__)'
    printf 'Read-only archive: %s\nNew attempt: %s\n' "$archive" "$out"
    df -h "$runtime"
    nproc
    free -h
    sinfo -p "${DREX_PARTITION:-batch}" -o '%P %a %l %D %t %c %m %G'
    squeue -u "$USER" -o '%.18i %.24j %.10T %.6D %R'
    exit 0 ;;
  verify|baseline-score|submit-smoke|submit-pilot|score) ;;
  *) echo "Usage: $0 {check|verify|baseline-score|submit-smoke|submit-pilot|score} [attempt-NNN]" >&2; exit 2 ;;
esac

head=$(git rev-parse HEAD)
diff_sha=$(git diff --no-ext-diff HEAD -- scripts/drex_drive | sha256sum | cut -d ' ' -f 1)
args=(--out "$out" --archive "$archive" --git-head "$head" --git-diff-sha256 "$diff_sha")
if [[ $1 == verify ]]; then
  exec "$python" -B -m "$module" brake-gap-verify "${args[@]}"
fi
[[ -O "$out" && -s "$out/manifest.json" ]] || { echo 'Run verify first' >&2; exit 2; }

case "$1" in
  submit-smoke|submit-pilot)
    provenance=${DREX_BRAKE_GAP_PROVENANCE:?Set path to independently archived Slurm 298185 runtime provenance}
    [[ -s "$provenance" && -s "$out/baseline-score.json" ]] || { echo 'Missing provenance or baseline scoring' >&2; exit 2; }
    provenance=$(realpath "$provenance")
    stage=brake-gap-gpu-smoke
    wall=00:10:00
    if [[ $1 == submit-pilot ]]; then
      [[ -s "$out/gpu-smoke.json" && ! -e "$out/predictions" ]] || { echo 'GPU smoke missing or pilot already attempted' >&2; exit 2; }
      stage=brake-gap-predict
      wall=00:30:00
    fi
    # Repeat the same runtime and artifact gates in the worker before loading weights.
    PYTHONPATH="$runtime/upstream" "$python" -B -m "$module" "brake-gap-preflight-${stage#brake-gap-}" "${args[@]}" \
      --provenance "$provenance" --weights "$runtime/models/drex-max-v1" --upstream "$runtime/upstream"
    # A freshly created private logging directory is also the write-once reservation.
    check_attempt_path
    mkdir -- "$out/$stage-slurm"
    exec sbatch --wait --parsable --job-name="$stage" --partition="${DREX_PARTITION:-batch}" \
      --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=4 --mem=96G --time="$wall" \
      --chdir="$repo" --output=/dev/null --error=/dev/null "$script" worker "$stage" "$repo" "$runtime" \
      "$archive" "$out" "$provenance" "$head" "$diff_sha" ;;
  baseline-score|score)
    lock_attempt
    mkdir -- "$out/brake-gap-$1-started"
    "$python" -B -m "$module" "brake-gap-$1" "${args[@]}" ;;
esac
