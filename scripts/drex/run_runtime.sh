#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
root=${DREX_RUNTIME_ROOT:-$HOME/drex_runtime}
upstream=$root/upstream
venv=$root/venv
commit=b932571901f531b2378b356fcab7d176048c2bcb
stage=${1:-check}
inside=false
background=false
if [[ "$stage" == submit-smoke || "$stage" == submit-night || "$stage" == submit-night-pilot || "$stage" == submit-real || "$stage" == submit-real-pilot ]]; then
  background=true
  stage=${stage#submit-}
fi
if [[ "$stage" == --inside ]]; then
  inside=true
  stage=${2:-}
fi
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
if [[ "$stage" == tests ]]; then
  cd "$repo"
  "${PYTHON:-python3}" -m unittest discover -s scripts/drex -p 'test_*.py' -v
  exit
fi
case "$stage" in
  prepare|setup|check|gpu-check|model-plan|download|smoke|night-plan|night-pilot|night|night-report|real-check|real-plan|real-pilot|real|real-report) ;;
  *) echo "Usage: $0 tests|prepare|setup|check|gpu-check|model-plan|download|smoke|submit-smoke|night-plan|night-pilot|night|submit-night-pilot|submit-night|night-report|real-check|real-plan|real-pilot|real|submit-real-pilot|submit-real|real-report" >&2; exit 2 ;;
esac
mkdir -p "$root"
root=$(cd "$root" && pwd)
export DREX_RUNTIME_ROOT="$root"
if [[ "$inside" == false ]]; then
  # Never inspect/execute the compute-node venv on the login node: its Python
  # symlink may be deliberately unavailable there (3.10 vs 3.12).
  memory=8G; limit=00:05:00; cpus=2
  resources=(--gres=gpu:1)
  if [[ "$stage" == setup || "$stage" == prepare ]]; then
    memory=16G; limit=00:30:00; cpus=4
  elif [[ "$stage" == smoke || "$stage" == night-pilot ]]; then
    memory=96G; limit=00:20:00; cpus=4
  elif [[ "$stage" == night ]]; then
    memory=96G; limit=08:00:00; cpus=4
  elif [[ "$stage" == real ]]; then
    memory=96G; limit=04:00:00; cpus=4
  elif [[ "$stage" == real-pilot ]]; then
    memory=96G; limit=00:30:00; cpus=4
  elif [[ "$stage" == model-plan || "$stage" == download || "$stage" == night-plan || "$stage" == night-report || "$stage" == real-check || "$stage" == real-plan || "$stage" == real-report ]]; then
    [[ -s "$root/setup_node.txt" ]] || { echo 'Run setup or prepare first' >&2; exit 1; }
    node=$(cat "$root/setup_node.txt")
    [[ "$node" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo 'Invalid saved setup node' >&2; exit 1; }
    resources=(--nodelist="$node" --gres=none)
    if [[ "$stage" == download ]]; then limit=00:45:00; fi
    if [[ "$stage" == real-check || "$stage" == real-plan || "$stage" == real-report ]]; then limit=00:15:00; fi
  fi
  echo "Slurm stage=$stage memory=$memory limit=$limit"
  if [[ "$background" == true ]]; then
    mkdir -p "$root/logs"
    # A Bash batch script with shell-escaped arguments; no eval or nested allocation.
    printf -v worker '%q ' srun --ntasks=1 --cpus-per-task="$cpus" --gres=gpu:1 \
      bash --login "$repo/scripts/drex/run_runtime.sh" --inside "$stage"
    printf '#!/usr/bin/env bash\nset -euo pipefail\nexec %s\n' "$worker" | \
      sbatch --partition="${DREX_PARTITION:-batch}" --nodes=1 --ntasks=1 \
        --cpus-per-task="$cpus" --mem="$memory" --gres=gpu:1 --time="$limit" \
        --export=ALL --job-name="drex-$stage" --output="$root/logs/$stage-%j.log"
    exit
  fi
  exec srun --partition="${DREX_PARTITION:-batch}" --nodes=1 --ntasks=1 \
    --cpus-per-task="$cpus" --mem="$memory" "${resources[@]}" --time="$limit" \
    --export=ALL bash --login "$repo/scripts/drex/run_runtime.sh" --inside "$stage"
fi
[[ -n ${SLURM_JOB_ID:-} && -n ${SLURM_STEP_ID:-} ]] || {
  echo '--inside is internal and requires an actual Slurm job step' >&2; exit 2;
}
echo "Compute stage=$stage host=$(hostname) job=$SLURM_JOB_ID step=$SLURM_STEP_ID"
if [[ "$stage" == prepare ]]; then
  # One allocation, no nested srun. A failure prevents the following stage.
  for next in setup gpu-check model-plan; do
    bash "$repo/scripts/drex/run_runtime.sh" --inside "$next"
  done
  exit
fi
if [[ "$stage" == setup ]]; then
  command -v flock >/dev/null || { echo 'flock (util-linux) is required for setup locking' >&2; exit 1; }
  exec 9>"$root/.setup.lock"
  flock -n 9 || { echo 'Another setup is running' >&2; exit 1; }
  bootstrap=${DREX_BOOTSTRAP_PYTHON:-python3}
  "$bootstrap" -c 'import sys; assert sys.version_info >= (3,12), "Compute-node Python >=3.12 required"'
  if [[ ! -d "$upstream" ]]; then
    git clone --no-checkout https://github.com/nace-ai/drex-decision-models.git "$upstream"
    git -C "$upstream" checkout --detach "$commit"
  fi
  [[ $(git -C "$upstream" rev-parse HEAD) == "$commit" ]] || { echo 'Wrong upstream commit' >&2; exit 1; }
  [[ -z $(git -C "$upstream" status --porcelain) ]] || { echo 'Upstream has local changes' >&2; exit 1; }
  if [[ ! -x "$venv/bin/python" ]]; then
    "$bootstrap" -m venv "$venv"
  fi
  "$venv/bin/python" -c 'import sys; assert sys.version_info >= (3,12), "Existing venv needs Python >=3.12; choose a new DREX_RUNTIME_ROOT"'
  if [[ ! -f "$root/requirements.lock.txt" ]]; then
    "$venv/bin/python" -m pip install -r "$upstream/models/drex-max-v1/requirements.txt"
    "$venv/bin/python" -m pip check
    "$venv/bin/python" -m pip freeze > "$root/requirements.lock.txt.tmp"
    mv "$root/requirements.lock.txt.tmp" "$root/requirements.lock.txt"
  fi
  export PYTHONPATH="$upstream${PYTHONPATH:+:$PYTHONPATH}"
  "$venv/bin/python" "$repo/scripts/drex/runtime.py" check --root "$root"
  [[ ${SLURMD_NODENAME:-} =~ ^[A-Za-z0-9_.-]+$ ]] || { echo 'No valid Slurm node name' >&2; exit 1; }
  printf '%s\n' "$SLURMD_NODENAME" > "$root/setup_node.txt.tmp"
  mv "$root/setup_node.txt.tmp" "$root/setup_node.txt"
  exit
fi
[[ -x "$venv/bin/python" ]] || { echo 'Run setup first' >&2; exit 1; }
[[ $(git -C "$upstream" rev-parse HEAD) == "$commit" ]] || { echo 'Wrong upstream commit' >&2; exit 1; }
[[ -z $(git -C "$upstream" status --porcelain) ]] || { echo 'Upstream has local changes' >&2; exit 1; }
export PYTHONPATH="$upstream${PYTHONPATH:+:$PYTHONPATH}"
if [[ "$stage" == real-check ]]; then
  cd "$repo"
  "$venv/bin/python" -c 'import numpy, PIL'
  "$venv/bin/python" -m unittest discover -s scripts/drex -p 'test_real.py' -v
  exit
fi
if [[ "$stage" == smoke ]]; then
  exec 9>"$root/.smoke.lock"
  flock -n 9 || { echo 'Another smoke is running' >&2; exit 1; }
  "$venv/bin/python" -u "$repo/scripts/drex/smoke.py" --root "$root"
  exit
fi
if [[ "$stage" == night || "$stage" == night-pilot || "$stage" == night-plan || "$stage" == night-report ]]; then
  exec 9>"$root/.night.lock"
  flock -n 9 || { echo 'Another night benchmark command is running' >&2; exit 1; }
  args=(run)
  if [[ "$stage" == night-pilot ]]; then args+=(--max-new 28); fi
  if [[ "$stage" == night-plan ]]; then args=(plan); fi
  if [[ "$stage" == night-report ]]; then args=(report); fi
  "$venv/bin/python" -u "$repo/scripts/drex/night.py" "${args[@]}" --root "$root"
  exit
fi
if [[ "$stage" == real || "$stage" == real-pilot || "$stage" == real-plan || "$stage" == real-report ]]; then
  exec 9>"$root/.real.lock"
  flock -n 9 || { echo 'Another real-pilot command is running' >&2; exit 1; }
  args=(all)
  if [[ "$stage" == real-pilot ]]; then args=(infer --max-new 16); fi
  if [[ "$stage" == real-plan ]]; then args=(plan); fi
  if [[ "$stage" == real-report ]]; then args=(report); fi
  "$venv/bin/python" -u "$repo/scripts/drex/real_pilot.py" "${args[@]}" --root "$root"
  exit
fi
case "$stage" in
  check|gpu-check|model-plan|download)
    "$venv/bin/python" "$repo/scripts/drex/runtime.py" "$stage" --root "$root"
    ;;
esac
