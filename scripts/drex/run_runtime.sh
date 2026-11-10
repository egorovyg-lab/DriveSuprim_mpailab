#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
root=${DREX_RUNTIME_ROOT:-$HOME/drex_runtime}
upstream=$root/upstream
venv=$root/venv
commit=b932571901f531b2378b356fcab7d176048c2bcb
stage=${1:-check}
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
if [[ "$stage" == tests ]]; then
  cd "$repo"
  "${PYTHON:-python3}" -m unittest discover -s scripts/drex -p 'test_runtime.py' -v
  exit
fi
mkdir -p "$root"
if [[ "$stage" == setup ]]; then
  exec 9>"$root/.setup.lock"
  flock -n 9 || { echo 'Another setup is running' >&2; exit 1; }
  "${PYTHON:-python3}" -c 'import sys; assert sys.version_info >= (3,12), "Python >=3.12 required"'
  if [[ ! -d "$upstream" ]]; then
    git clone --no-checkout https://github.com/nace-ai/drex-decision-models.git "$upstream"
    git -C "$upstream" checkout --detach "$commit"
  fi
  [[ $(git -C "$upstream" rev-parse HEAD) == "$commit" ]] || { echo 'Wrong upstream commit' >&2; exit 1; }
  [[ -z $(git -C "$upstream" status --porcelain) ]] || { echo 'Upstream has local changes' >&2; exit 1; }
  if [[ ! -x "$venv/bin/python" ]]; then
    "${PYTHON:-python3}" -m venv "$venv"
  fi
  if [[ ! -f "$root/requirements.lock.txt" ]]; then
    "$venv/bin/python" -m pip install -r "$upstream/models/drex-max-v1/requirements.txt"
    "$venv/bin/python" -m pip check
    "$venv/bin/python" -m pip freeze > "$root/requirements.lock.txt.tmp"
    mv "$root/requirements.lock.txt.tmp" "$root/requirements.lock.txt"
  fi
  stage=check
fi
[[ -x "$venv/bin/python" ]] || { echo 'Run setup first' >&2; exit 1; }
[[ $(git -C "$upstream" rev-parse HEAD) == "$commit" ]] || { echo 'Wrong upstream commit' >&2; exit 1; }
[[ -z $(git -C "$upstream" status --porcelain) ]] || { echo 'Upstream has local changes' >&2; exit 1; }
export PYTHONPATH="$upstream${PYTHONPATH:+:$PYTHONPATH}"
case "$stage" in
  check|model-plan|download)
    "$venv/bin/python" "$repo/scripts/drex/runtime.py" "$stage" --root "$root"
    ;;
  gpu-check)
    srun --partition="${DREX_PARTITION:-batch}" --nodes=1 --ntasks=1 \
      --cpus-per-task=2 --mem=8G --gres=gpu:1 --time=00:05:00 --immediate=60 \
      "$venv/bin/python" "$repo/scripts/drex/runtime.py" gpu-check --root "$root"
    ;;
  *) echo "Usage: $0 tests|setup|check|gpu-check|model-plan|download" >&2; exit 2 ;;
esac
