#!/bin/bash -l
set -euo pipefail

repo=${DAC_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
runtime=${DREX_RUNTIME_ROOT:?Set DREX_RUNTIME_ROOT to the existing MSU runtime}
data=${DAC_DATA:-$runtime/data/drivemem_spatial_dac_v1}
images=${DAC_IMAGES:-$runtime/data/navsim_train_v1/package/images}
out=${DAC_OUT:-$runtime/runs/drivemem_spatial_dac_v1}
py=${DAC_PYTHON:-$runtime/venv/bin/python}
partition=${DAC_PARTITION:-${DREX_PARTITION:-batch}}
cd "$repo"
case "${1:-submit}" in
  submit)
    [[ -f "$data/manifest.json" && -d "$images" ]] || {
      echo 'Missing portable manifest or images' >&2
      exit 1
    }
    export DAC_DATA="$data" DAC_IMAGES="$images" DAC_OUT="$out" DAC_PYTHON="$py" DAC_REPO="$repo"
    mkdir -p "$runtime/logs"
    unset SBATCH_NODELIST
    runner="$repo/scripts/drivesuprim/evaluation/run_drivemem_spatial_dac_msu.sh"
    gpu=$(sbatch --parsable --export=ALL --partition="$partition" --gres=gpu:1 --cpus-per-task=4 \
      --mem=24G --time=01:30:00 --job-name=dac-r34 --output="$runtime/logs/dac-r34-%j.log" "$runner" export)
    gpu=${gpu%%;*}
    [[ "$gpu" =~ ^[0-9]+$ ]]
    fit=$(sbatch --parsable --export=ALL --partition="$partition" --dependency="afterok:$gpu" --kill-on-invalid-dep=yes \
      --gres=none --cpus-per-task=8 --mem=24G --time=03:00:00 --job-name=dac-probe \
      --output="$runtime/logs/dac-probe-%j.log" "$runner" fit)
    printf 'GPU export %s\nCPU fit    %s\nOutput     %s\n' "$gpu" "$fit" "$out"
    ;;
  export|fit)
    [[ -n ${SLURM_JOB_ID:-} ]] || { echo 'Use submit or a Slurm allocation' >&2; exit 2; }
    [[ -x "$py" ]] || { echo "Python is unavailable on $(hostname): $py" >&2; exit 2; }
    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
    export MKL_NUM_THREADS="$OMP_NUM_THREADS" PYTHONDONTWRITEBYTECODE=1
    if [[ $1 == export ]]; then
      "$py" -u -m navsim.planning.script.run_drivemem_spatial_dac smoke \
        --data "$data" --images "$images" --out "$out"
      exec "$py" -u -m navsim.planning.script.run_drivemem_spatial_dac export \
        --data "$data" --images "$images" --out "$out"
    fi
    exec "$py" -u -m navsim.planning.script.run_drivemem_spatial_dac fit --data "$data" --out "$out"
    ;;
  *) echo 'Usage: run_drivemem_spatial_dac_msu.sh submit|export|fit' >&2; exit 2 ;;
esac
