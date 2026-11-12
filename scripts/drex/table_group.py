"""One-allocation execution adapter; the frozen table experiment is unchanged."""
import argparse
import json
import os
from pathlib import Path
import re
import socket
import sys
import time

from runtime import read,write
from transfer import sha,freeze

SOURCES = ('table_group.py','run_table_group.sh')


def validate_bindings(rows,count):
    if len(rows)!=count or sorted(r['rank'] for r in rows)!=list(range(count)):
        raise RuntimeError('Incomplete/duplicate GPU ranks')
    if any(r['visible_devices']!=1 or not r['uuid'] for r in rows):
        raise RuntimeError('Each rank must see exactly one physical GPU with a UUID')
    if len({r['uuid'] for r in rows})!=count:
        raise RuntimeError('Two workers share a GPU; refusing model inference')
    if len({r['hostname'] for r in rows})!=1:
        raise RuntimeError('This launcher requires one node')
    if len({(r['job_id'],r['step_id']) for r in rows})!=1:
        raise RuntimeError('GPU bindings come from different Slurm steps')
    if any(not r.get('launch_id') for r in rows) or len({r['launch_id'] for r in rows})!=1:
        raise RuntimeError('GPU bindings come from different launch attempts')


def gpu_binding(out,count,phase):
    import torch
    job,step = os.environ.get('SLURM_JOB_ID',''),os.environ.get('SLURM_STEP_ID','')
    rank_text = os.environ.get('SLURM_PROCID','')
    if not re.fullmatch(r'\d+',job) or not re.fullmatch(r'\d+',step) or not re.fullmatch(r'\d+',rank_text):
        raise RuntimeError('Requires a real numbered Slurm task/step')
    rank = int(rank_text)
    launch_id = os.environ.get('DREX_GROUP_STEP_TOKEN','')
    if not re.fullmatch(r'\d{1,24}-\d{1,16}',launch_id):
        raise RuntimeError('Missing group launch token; use the grouped batch coordinator')
    if not 0<=rank<count or int(os.environ.get('SLURM_NTASKS','0'))!=count:
        raise RuntimeError('Slurm task count/rank differs from requested GPU count')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('GPU binding failed: each process must see exactly one allocated GPU')
    props = torch.cuda.get_device_properties(0)
    raw_uuid = getattr(props,'uuid',None)
    uuid = '' if raw_uuid is None else str(raw_uuid)
    if not uuid or props.total_memory<70*1024**3 or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Requires a UUID-identifiable BF16 GPU with >=70 GiB VRAM')
    directory = out/'execution'/f'{job}-{step}-{phase}-{launch_id}'
    row = dict(rank=rank,hostname=socket.gethostname(),job_id=job,step_id=step,uuid=uuid,
        visible_devices=torch.cuda.device_count(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        gpu=props.name,vram_bytes=props.total_memory,launch_id=launch_id,observed_unix_seconds=time.time())
    write(directory/f'rank-{rank:02d}.json',row)
    deadline = time.monotonic()+120
    paths = [directory/f'rank-{i:02d}.json' for i in range(count)]
    while not all(p.exists() for p in paths):
        if time.monotonic()>deadline: raise RuntimeError('Other GPU ranks did not reach the binding barrier')
        time.sleep(.25)
    rows = [read(p) for p in paths]
    validate_bindings(rows,count)
    freeze(directory/f'verified-{rank:02d}.json',dict(bindings={p.name:sha(p) for p in paths}))
    print(json.dumps(dict(status='gpu_binding_passed',**row),indent=2),flush=True)
    return rank


def prepare(root,package,out,count,cohort):
    import table_pilot as pilot
    result = pilot.plan(root,package,out,count,cohort)
    freeze(out/'group_execution.json',dict(protocol='single_allocation_replicas_v1',replicas=count,
        scientific_plan_sha256=sha(out/'plan.json'),
        sources={n:sha(Path(__file__).with_name(n)) for n in SOURCES},
        cpus_per_rank=4,memory_gib_per_rank=96,
        note='One Slurm allocation, N concurrent ranks, one GPU per rank. Same fixed scientific table protocol.'))
    return result


def check_execution(out,count):
    value = read(out/'group_execution.json')
    if (value['protocol']!='single_allocation_replicas_v1' or value['replicas']!=count
            or value['scientific_plan_sha256']!=sha(out/'plan.json')):
        raise RuntimeError('Frozen grouped execution plan changed')
    for name,digest in value['sources'].items():
        if sha(Path(__file__).with_name(name))!=digest:
            raise RuntimeError('Grouped launcher changed; use a new table_group_* run name')
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('plan','bind','worker','worker-pilot','merge'))
    p.add_argument('--root',type=Path,required=True); p.add_argument('--replicas',type=int,required=True)
    a = p.parse_args(); root = a.root.expanduser().resolve()
    if a.replicas not in (1,2,4,8): raise ValueError('Use 1/2/4/8 GPU ranks')
    name = os.environ['DREX_TABLE_NAME']
    if not re.fullmatch(r'table_group_[A-Za-z0-9_-]{1,64}',name):
        raise ValueError('Use a new table_group_* output; old array experiments are preserved')
    out = root/'runs'/name; phase = a.stage
    if phase=='bind':
        gpu_binding(out,a.replicas,'bind'); return
    if phase=='plan':
        import fcntl
        out.mkdir(parents=True,exist_ok=True)
        plan_lock = (out/'.plan.lock').open('a')
        try: fcntl.flock(plan_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Another grouped plan is running')
        package = Path(os.environ.get('DREX_TABLE_DATA',str(root/'data/navsim_train_v1/package'))).resolve()
        result = prepare(root,package,out,a.replicas,os.environ.get('DREX_TABLE_COHORT','all'))
        print(json.dumps(result,indent=2)); return
    check_execution(out,a.replicas)
    import table_pilot as pilot
    args = ['table_pilot.py','merge' if phase=='merge' else 'worker','--root',str(root),
            '--replicas',str(a.replicas),'--cohort',os.environ.get('DREX_TABLE_COHORT','all')]
    if phase!='merge':
        rank = gpu_binding(out,a.replicas,phase)
        import fcntl
        directory = out/'shards'/f'{rank:02d}'; directory.mkdir(parents=True,exist_ok=True)
        lock = (directory/'.worker.lock').open('a')
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('A worker already owns this shard')
        args += ['--index',str(rank)]
        if phase=='worker-pilot': args += ['--pilot']
    else:
        import fcntl
        lock = (out/'.report.lock').open('a')
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Another report job is running')
    sys.argv = args
    pilot.main()


if __name__=='__main__': main()
