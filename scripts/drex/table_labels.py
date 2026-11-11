"""Stdlib bridge: upload the frozen pool and scorer tools, retrieve checked labels."""
import argparse
import os
from pathlib import Path
import re
import subprocess
import time

import clearance_bridge as bridge
from transfer import read,sha,freeze


def wait_for(path,seconds):
    deadline = time.monotonic()+seconds
    if not path.exists(): print('Waiting for '+str(path),flush=True)
    while not path.exists():
        if time.monotonic()>=deadline: raise RuntimeError('Wait timed out: '+str(path))
        time.sleep(min(10,max(.1,deadline-time.monotonic())))


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True); p.add_argument('--table-run',required=True)
    p.add_argument('--host',default='egorov@93.180.62.99'); p.add_argument('--port',type=int,default=27434)
    p.add_argument('--container',default='egorov_drivesuprim')
    p.add_argument('--control',type=Path,default=Path.home()/'.ssh/drex-astra-transfer.sock')
    p.add_argument('--remote-repo',default='/workspace/DriveSuprim')
    p.add_argument('--wait-seconds',type=int,default=0)
    p.add_argument('--submit-report',action='store_true')
    args = p.parse_args(); root = args.root.expanduser().resolve()
    if not re.fullmatch(r'table_[A-Za-z0-9_-]{1,80}',args.table_run): raise ValueError('Invalid table run name')
    source = root/'runs'/args.table_run
    source.mkdir(parents=True,exist_ok=True)
    import fcntl
    label_lock = (source/'.labels.lock').open('a')
    try: fcntl.flock(label_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError('Another label bridge is running for this table experiment')
    if args.wait_seconds<0: raise ValueError('Wait budget must be nonnegative')
    wait_for(source/'label_request.json',args.wait_seconds)
    request = read(source/'label_request.json')
    if request['plan_sha256']!=sha(source/'plan.json'): raise RuntimeError('Label request/plan mismatch')
    # Short deterministic remote output name; never overwrite a changed experiment.
    args.source = args.table_run
    args.name = 'rollout_audit_table_'+sha(source/'label_request.json')[:20]
    tools = tuple(dict.fromkeys(('table_score.py','table_data.py',*bridge.TOOLS)))
    upload = root/'runs'/args.name
    freeze(upload/'audit_request.json',request)
    if len(bridge.input_archive(upload,tools))>4*1024**2:
        raise RuntimeError('Label upload exceeds bounded 4 MiB protocol; split the cohort in a new protocol')
    code = bridge.run(args,request=request,tools=tools,remote_tool='table_score.py')
    if code: raise SystemExit(code)
    pointer = read(root/'runs'/args.name/'latest_result.json')
    result = Path(pointer['result']); summary = read(result/'report.json')
    if summary['status']!='complete' or sha(result/'labels.json')!=summary['labels_sha256']:
        raise RuntimeError('Verified complete scoring results required')
    freeze(source/'label_source.json',dict(labels=str(result/'labels.json'),sha256=sha(result/'labels.json'),
        request_sha256=sha(source/'label_request.json')))
    if args.submit_report:
        wait_for(source/'decision_commit.json',args.wait_seconds)
        wait_for(source/'report.json',args.wait_seconds)
        if read(source/'report.json')['status']!='complete':
            identity = read(source/'identity.json')
            env = dict(os.environ,DREX_RUNTIME_ROOT=str(root),DREX_TABLE_NAME=args.table_run,
                       DREX_TABLE_COHORT=identity['cohort'])
            subprocess.run(['bash',str(Path(__file__).with_name('run_table.sh')),'report',str(identity['replicas'])],
                           env=env,check=True)
        else: print('Complete report already exists.',flush=True)
    else: print('Labels retrieved. Run run_table.sh report with the SAME cohort and GPU count.')
