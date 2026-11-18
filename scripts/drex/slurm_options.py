"""Standard-library scheduling policy shared with slurm_options.sh."""
import os
import re


def node_options(environ=None):
    value=(os.environ if environ is None else environ).get('DREX_NODE') or 'auto'
    if value=='auto':return []
    if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*',value):
        raise ValueError('DREX_NODE must be empty/auto or one explicit node name')
    return ['--nodelist='+value]
