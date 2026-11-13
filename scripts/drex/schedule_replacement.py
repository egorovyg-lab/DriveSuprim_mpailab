"""Finite Slurm continuation chain. Does not modify the frozen experiment code."""
import argparse
from contextlib import contextmanager
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

LIMIT = 'RuntimeError: Worker time budget reached; resume SAME name/count before report'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8'); temp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(args, env=None):
    result = subprocess.run(args, env=env, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError('Command failed: '+shlex.join(map(str,args))+'\n'+result.stdout+result.stderr)
    return result.stdout


def check_budget_logs(root, parent, count):
    """Check every rank, not just the presence of one expected failure."""
    saw_limit = False
    for rank in range(count):
        path = root/'logs'/f'replacement-worker-{parent}-{rank}.log'
        text = path.read_text(encoding='utf-8', errors='replace')
        errors = re.findall(r'^[A-Za-z_][\w.]*(?:Error|Exception|Interrupt|Exit):.*$',text,re.M)
        if any(line != LIMIT for line in errors):
            raise RuntimeError('Unexpected exception in '+str(path)+': '+repr(errors))
        if 'Traceback (most recent call last):' in text and LIMIT not in errors:
            raise RuntimeError('Unrecognized traceback in '+str(path))
        if re.search(r'oom[_ -]?kill|out.of.memory|TIME LIMIT|NODE_FAIL',text,re.I):
            raise RuntimeError('Infrastructure/time-limit failure in '+str(path))
        saw_limit |= LIMIT in errors
    if not saw_limit:
        raise RuntimeError('No confirmed worker-budget stop; inspect predecessor logs before resuming')


class Chain:
    def __init__(self, root, name, count, call=command):
        if not re.fullmatch(r'replacement_[A-Za-z0-9_-]{1,64}',name) or count not in (1,2,4,8):
            raise ValueError('Invalid run/GPU count')
        self.root = Path(root).expanduser().resolve(); self.name = name; self.count = count; self.call = call
        self.out = self.root/'runs'/name; self.script = Path(__file__).resolve()
        self.launcher = self.script.with_name('run_replacement.sh')
        identity = read(self.out/'identity.json'); plan = read(self.out/'plan.json')
        if (identity['replicas'] != count or plan['identity_sha256'] != sha(self.out/'identity.json')
                or identity['config']['protocol'] != 'replacement_star_v1'):
            raise RuntimeError('Existing frozen plan does not match GPU count/protocol')
        self.binding = dict(plan_sha256=sha(self.out/'plan.json'), helper_sha256=sha(self.script),
            launcher_sha256=sha(self.launcher), name=name, gpus=count)
        self.node = (self.root/'setup_node.txt').read_text().strip()
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',self.node): raise ValueError('Invalid node')

    def path(self,parent):
        if not re.fullmatch(r'\d+',str(parent)): raise ValueError('Invalid predecessor job ID')
        return self.out/'scheduler'/f'after-{parent}.json'

    def checked(self,parent):
        record = read(self.path(parent))
        current = dict(self.binding,plan_sha256=sha(self.out/'plan.json'),
            helper_sha256=sha(self.script),launcher_sha256=sha(self.launcher))
        if record['binding'] != current or current != self.binding:
            raise RuntimeError('Scheduler helper, launcher or scientific plan changed')
        return record

    def submit(self,parent,passes,check_queue=False):
        path = self.path(parent)
        if not 1 <= passes <= 4: raise ValueError('Use 1..4 finite continuations')
        if path.exists():
            record = self.checked(parent)
            if record['passes'] != passes: raise RuntimeError('A different chain already owns this predecessor')
            if record.get('callback_job'):
                print('Already scheduled: '+str(record['callback_job']),flush=True); return record
            raise RuntimeError('Previous submission is unresolved; inspect '+str(path))
        if check_queue:
            text = self.call(['squeue','-h','-u',getpass.getuser(),'-o','%i|%j|%T'])
            rows = [[part.strip() for part in line.split('|')] for line in text.splitlines() if line.strip()]
            target = [r for r in rows if r[0] == str(parent)]
            if len(target)!=1 or target[0][1] not in ('drex-replacement-gpu-full','drex-replacement-gpu-night') or target[0][2]!='RUNNING':
                raise RuntimeError('Expected running predecessor GPU job owned by this user')
            if any(r[1].startswith(('drex-delayed-resume','drex-resume-chain','drex-final-resume','drex-chain-')) for r in rows):
                raise RuntimeError('Another delayed continuation exists; do not create a second chain')
        record = dict(binding=self.binding,parent=str(parent),passes=passes,state='submitting')
        write(path,record)  # Crash here deliberately blocks a blind duplicate submission.
        argv = ['python3',str(self.script),'advance','--root',str(self.root),'--name',self.name,
            '--gpus',str(self.count),'--after',str(parent)]
        script = '#!/usr/bin/env bash\nset -euo pipefail\nexec '+shlex.join(argv)+'\n'
        args = ['sbatch','--parsable','--partition='+os.environ.get('DREX_PARTITION','batch'),
            '--nodes=1','--ntasks=1','--cpus-per-task=1','--mem=512M','--gres=none','--time=00:05:00',
            '--nodelist='+self.node,'--dependency=afterany:'+str(parent),'--kill-on-invalid-dep=yes',
            '--job-name=drex-chain-'+str(parent),'--export=ALL',
            '--output='+str(self.root/'logs'/'resume-chain-%j.log')]
        # sbatch reads the small callback from stdin, not from a mutable shell variable.
        if self.call is command:
            result = subprocess.run(args,input=script,text=True,capture_output=True)
            if result.returncode: raise RuntimeError('sbatch rejected callback: '+result.stderr)
            output = result.stdout
        else: output = self.call(args+['__SCRIPT__',script])
        job = output.strip().split(';')[0]
        if not re.fullmatch(r'\d+',job): raise RuntimeError('Unrecognized sbatch reply: '+output)
        record.update(state='queued',callback_job=job); write(path,record)
        print('Scheduled callback '+job+' after GPU '+str(parent)+'; remaining GPU continuations='+str(passes),flush=True)
        return record

    def advance(self,parent):
        record = self.checked(parent); path = self.path(parent)
        if record['state'] in ('done','already_complete'):
            print('Callback already handled; no duplicate GPU submission.',flush=True); return
        if record['state'] not in ('queued','gpu_submitted'):
            raise RuntimeError('Unresolved previous callback action; inspect '+str(path))
        if record['state']=='queued':
            if all((self.out/'shards'/f'{i:02d}'/'commit.json').is_file() for i in range(self.count)):
                record['state']='already_complete'; write(path,record)
                print('All workers committed; no further GPU work needed.',flush=True); return
            check_budget_logs(self.root,parent,self.count)
            record['state']='resuming'; write(path,record)
            env = dict(os.environ,DREX_RUNTIME_ROOT=str(self.root),DREX_REPLACEMENT_NAME=self.name)
            output = self.call(['bash',str(self.launcher),'resume',str(self.count)],env=env)
            print(output,flush=True)
            matches = re.findall(r'^Resume GPU allocation: (\d+); CPU report: (\d+)\s*$',output,re.M)
            if len(matches)!=1: raise RuntimeError('Cannot identify actual new GPU/report job; inspect '+str(path))
            gpu,report = matches[0]
            record.update(state='gpu_submitted',gpu_job=gpu,report_job=report); write(path,record)
        if record['passes'] > 1:
            child = self.submit(record['gpu_job'],record['passes']-1)
            record['next_callback_job']=child['callback_job']
        record['state']='done'; write(path,record)


@contextmanager
def lock(out):
    import fcntl
    folder = out/'scheduler'; folder.mkdir(parents=True,exist_ok=True)
    with (folder/'.lock').open('a') as stream:
        try: fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Another scheduler action is running')
        yield


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('submit','advance','status'))
    p.add_argument('--root',type=Path,default=Path(os.environ.get('DREX_RUNTIME_ROOT',str(Path.home()/'drex_runtime'))))
    p.add_argument('--name',default=os.environ.get('DREX_REPLACEMENT_NAME','replacement_gpu2_v1'))
    p.add_argument('--gpus',type=int,default=2); p.add_argument('--after',type=int); p.add_argument('--passes',type=int,default=2)
    a = p.parse_args(); chain = Chain(a.root,a.name,a.gpus)
    if a.stage=='status':
        for path in sorted((chain.out/'scheduler').glob('after-*.json')): print(path.name+'\n'+path.read_text())
        return
    if a.after is None: p.error('--after is required')
    if a.stage=='advance' and not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Callback requires Slurm')
    with lock(chain.out):
        if a.stage=='submit': chain.submit(a.after,a.passes,check_queue=True)
        else: chain.advance(a.after)


if __name__=='__main__': main()
