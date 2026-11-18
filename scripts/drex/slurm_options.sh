# Source only after the launcher has resolved its checkout path (including sbatch spool mode).
# setup_node.txt is provenance, never a scheduling constraint.
export DREX_NODE=${DREX_NODE:-auto}
DREX_NODE_ARGS=()
if [[ "$DREX_NODE" != auto ]]; then
  [[ "$DREX_NODE" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]*$ ]] || {
    echo 'DREX_NODE must be empty/auto or one explicit node name, e.g. cn29' >&2
    exit 2
  }
  DREX_NODE_ARGS=(--nodelist="$DREX_NODE")
fi
# An inherited sbatch environment option must not silently reinstate an old pin.
unset SBATCH_NODELIST

drex_require_python() {
  [[ -x "$1/venv/bin/python" ]] || {
    echo "Compute Python unavailable on $(hostname): $1/venv/bin/python. Check shared paths and Python >=3.12 on this node; setup_node.txt is diagnostic only. No packages were changed." >&2
    return 1
  }
}
