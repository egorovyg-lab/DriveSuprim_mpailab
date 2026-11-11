#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
root=${DREX_RUNTIME_ROOT:-$HOME/drex_runtime}
export DREX_OVERLAY_NAME=${DREX_OVERLAY_NAME:-overlay_top4_v1}
[[ "$DREX_OVERLAY_NAME" =~ ^overlay_[A-Za-z0-9_-]{1,64}$ ]] || exit 2
stage=${1:-submit}
if [[ "$stage" != --inside ]]; then
  [[ "$stage" == submit ]] || { echo 'Usage: run_overlay_audit.sh submit' >&2; exit 2; }
  [[ -s "$root/setup_node.txt" ]] || { echo 'Run runtime prepare first' >&2; exit 1; }
  node=$(cat "$root/setup_node.txt")
  [[ "$node" =~ ^[A-Za-z0-9_.-]+$ ]] || exit 2
  mkdir -p "$root/logs"
  root=$(cd "$root" && pwd); export DREX_RUNTIME_ROOT="$root"
  printf -v command '%q ' srun --ntasks=1 --cpus-per-task=2 --gres=none \
    bash --login "$repo/scripts/drex/run_overlay_audit.sh" --inside
  printf '#!/usr/bin/env bash\nset -euo pipefail\nexec %s\n' "$command" | \
    sbatch --partition="${DREX_PARTITION:-batch}" --nodes=1 --ntasks=1 --nodelist="$node" \
      --cpus-per-task=2 --mem=8G --gres=none --time=00:15:00 --export=ALL \
      --job-name=drex-overlay-audit --output="$root/logs/overlay-audit-%j.log"
  exit
fi
[[ -n ${SLURM_JOB_ID:-} && -n ${SLURM_STEP_ID:-} ]] || { echo 'Requires an actual Slurm job step' >&2; exit 2; }
[[ -x "$root/venv/bin/python" ]] || { echo 'Missing compute-node venv' >&2; exit 1; }
out=$root/runs/$DREX_OVERLAY_NAME
[[ -f "$out/plan.json" ]] || { echo 'Missing existing overlay plan' >&2; exit 1; }
exec 9>"$out/.projection_audit.lock"
flock -n 9 || { echo 'Another projection audit is running' >&2; exit 1; }
export PYTHONDONTWRITEBYTECODE=1
"$root/venv/bin/python" -u "$repo/scripts/drex/audit_overlay_projection.py" --root "$root"
