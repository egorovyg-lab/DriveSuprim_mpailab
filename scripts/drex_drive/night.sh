#!/usr/bin/env bash
set -euo pipefail

# Slurm runs a copy of this script from its spool directory.
if [[ ${1:-} == worker ]]; then
  repo=$2
  runtime=$3
  out=$4
  task=${5:-predict}
  cd "$repo"
  exec 8>"$out/.predict.lock"
  flock -n 8 || { echo 'Night predictions are already running' >&2; exit 1; }
  export PYTHONPATH="$runtime/upstream"
  exec srun --ntasks=1 "$runtime/venv/bin/python" -m scripts.drex_drive.navsim_night "$task" \
    --out "$out" --weights "$runtime/models/drex-max-v1"
fi

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo"
script="$repo/scripts/drex_drive/night.sh"
module=scripts.drex_drive.navsim_night
runtime=${DREX_RUNTIME_ROOT:-/scratch/pr0201/drex_runtime}
out=$runtime/runs/navsim-straight-night-v1
pilot=$runtime/runs/navsim-straight-train-v1/observations.json
log=$out/night.log

case "${1:-}" in
  preview-check)
    for input in observations.json suite.json corridor.json control/identity.json; do
      [[ -s "$out/$input" ]] || { echo "Required input missing: $out/$input" >&2; exit 2; }
    done
    if [[ ! -s "$out/corridor-preview-gpu-complete.txt" ]]; then
      [[ -x "$runtime/venv/bin/python" && -f "$runtime/upstream/drex_max/model.py" &&
        -d "$runtime/models/drex-max-v1" ]] ||
        { echo "Drex runtime, upstream loader or weights missing in $runtime" >&2; exit 2; }
      "$runtime/venv/bin/python" --version
      echo 'GPU request: 1 GPU, 4 CPUs, 96G RAM, 30 minutes'
    else
      echo 'GPU predictions already complete'
    fi
    printf '\nScratch space\n'
    df -h "$out"
    printf '\nLogin node resources\n'
    nproc
    free -h
    printf '\nSlurm partition\n'
    sinfo -p "${DREX_PARTITION:-batch}" -o '%P %a %l %D %t %c %m %G'
    printf '\nJobs for %s\n' "$USER"
    squeue -u "$USER" -o '%.18i %.24j %.10T %.10M %R' ;;
  corridor|corridor-preview)
    [[ -s "$out/suite.json" ]] || { echo 'Complete nightly scoring before running the corridor test' >&2; exit 2; }
    exec 7>"$out/.corridor.lock"
    flock -n 7 || { echo 'Corridor test is already running' >&2; exit 1; }
    stage=corridor
    if [[ $1 == corridor-preview ]]; then
      [[ -s "$out/corridor.json" ]] || { echo 'Complete the corridor object test before running its preview' >&2; exit 2; }
      stage=preview
    fi
    bash "$repo/scripts/drex_drive/straight.sh" connect
    python3 -m "$module" "$stage-prepare" --out "$out"
    if [[ ! -s "$out/$1-gpu-complete.txt" ]]; then
      sbatch --job-name="drex-$1" --wait --partition="${DREX_PARTITION:-batch}" --nodes=1 --ntasks=1 \
        --gres=gpu:1 --cpus-per-task=4 --mem=96G --time=00:30:00 --chdir="$repo" \
        --output="$runtime/logs/$1-%j.log" "$script" worker "$repo" "$runtime" "$out" "$stage-predict"
      printf 'Predictions complete\n' >"$out/$1-gpu-complete.txt"
    fi
    python3 -m "$module" "$stage-score" --out "$out" ;;
  collision)
    [[ -s "$out/suite.json" ]] || { echo 'Complete nightly scoring before reviewing the collision' >&2; exit 2; }
    exec 7>"$out/.collision.lock"
    flock -n 7 || { echo 'Collision review is already running' >&2; exit 1; }
    bash "$repo/scripts/drex_drive/straight.sh" connect
    python3 -m "$module" collision --out "$out" ;;
  start)
    [[ -s "$pilot" ]] || { echo 'Original TRAIN observations are missing' >&2; exit 2; }
    mkdir -p "$out" "$runtime/logs"
    bash "$repo/scripts/drex_drive/straight.sh" connect
    nohup bash "$script" run </dev/null >>"$log" 2>&1 &
    printf 'Night run PID %s\nLog %s\n' "$!" "$log" ;;
  run)
    exec 7>"$out/.run.lock"
    flock -n 7 || { echo 'The nightly comparison is already running' >&2; exit 1; }
    stage=prepare
    trap 'code=$?; if (( code )); then printf "%s failed (%s)\n" "$stage" "$code" >"$out/status.txt"; fi' EXIT
    printf '%s\n' "$stage" >"$out/status.txt"
    python3 -m "$module" prepare --out "$out" --pilot "$pilot"
    if [[ ! -s "$out/gpu-complete.txt" ]]; then
      stage=gpu
      printf '%s\n' "$stage" >"$out/status.txt"
      sbatch --parsable --job-name=drex-night --wait --partition="${DREX_PARTITION:-batch}" --nodes=1 --ntasks=1 \
        --gres=gpu:1 --cpus-per-task=4 --mem=96G --time="${DREX_NIGHT_GPU_TIME:-02:00:00}" --chdir="$repo" \
        --output="$runtime/logs/night-%j.log" "$script" worker "$repo" "$runtime" "$out" | tee "$out/last-job.txt"
      printf 'Predictions complete\n' >"$out/gpu-complete.txt"
    fi
    if [[ ! -s "$out/suite.json" ]]; then
      stage=score
      printf '%s\n' "$stage" >"$out/status.txt"
      python3 -m "$module" score --out "$out"
    fi
    if [[ ! -s "$out/single-edits.json" ]]; then
      stage=probes
      printf '%s\n' "$stage" >"$out/status.txt"
      python3 -m "$module" probes --out "$out"
    fi
    printf 'complete\n' >"$out/status.txt" ;;
  status)
    if [[ -s "$out/status.txt" ]]; then
      cat "$out/status.txt"
    fi
    squeue -u "$USER" -n drex-night -o "%.18i %.22j %.10T %.10M %R"
    if [[ -s "$log" ]]; then
      tail -n 18 "$log"
    fi
    if [[ -s "$out/last-job.txt" ]]; then
      read -r job <"$out/last-job.txt"
      job=${job%%;*}
      if [[ $job =~ ^[0-9]+$ ]]; then
        printf '\nSlurm job %s\n' "$job"
        sacct -X -j "$job" --format=JobID,State,Elapsed,ExitCode || true
        if [[ -s "$runtime/logs/night-$job.log" ]]; then
          tail -n 18 "$runtime/logs/night-$job.log"
        fi
      fi
    fi ;;
  *) printf 'Usage: %s start|status|collision|corridor|preview-check|corridor-preview\n' "$0" >&2; exit 2 ;;
esac
