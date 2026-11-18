"""Bounded, standard-library-only Drex feasibility check. Never downloads weights."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import urllib.request

UPSTREAM='b932571901f531b2378b356fcab7d176048c2bcb'
REVISION='81cb1111d9ceb6fb54eb2e300890ec5380f175c0'
MODEL_BLOB='f3f50697a5125cdb0c7b929ffa1440010b2586f9'
DECISION_SHA256='37870bfacfd9bdddf30f191d3cb4d30e908a57a436d557189dd65bc652826125'
LIMIT=256*1024


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(path)


def fetch(url,path):
    if path.exists():
        if path.stat().st_size>LIMIT:raise ValueError('Cached metadata exceeds bounded size')
        return path.read_bytes()
    request=urllib.request.Request(url,headers={'User-Agent':'DriveSuprim-Astra-feasibility/1'})
    with urllib.request.urlopen(request,timeout=20) as response:
        content=response.read(LIMIT+1)
    if len(content)>LIMIT:raise ValueError('Metadata exceeds bounded download size')
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_bytes(content);tmp.replace(path)
    return content


def source_capacity(source,verify_blob=True):
    if verify_blob:
        digest=hashlib.sha1(b'blob '+str(len(source)).encode()+b'\0'+source).hexdigest()
        if digest!=MODEL_BLOB:raise RuntimeError('Pinned upstream model.py Git blob changed')
    tree=ast.parse(source)
    caps=[n.value.value for n in tree.body if isinstance(n,ast.Assign) and
          any(isinstance(t,ast.Name) and t.id=='MAX_OPTIONS' for t in n.targets) and
          isinstance(n.value,ast.Constant) and type(n.value.value) is int]
    readouts=[n.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and
              any(isinstance(t,ast.Attribute) and t.attr=='readout' for t in n.targets)]
    if len(caps)!=1 or not any(isinstance(v,ast.Call) and len(v.args)>1 and
                              isinstance(v.args[1],ast.Name) and v.args[1].id=='MAX_OPTIONS' for v in readouts):
        raise RuntimeError('Cannot establish fixed readout capacity; manual review required')
    return caps[0]


def contract(source,decision,requested=256):
    capacity=source_capacity(source)
    codes=decision['codes']; ids=decision['token_ids']
    if (decision.get('format_version')!=1 or
        decision.get('attention_mode','noncausal_full_attention')!='noncausal_full_attention' or
        len(codes)!=capacity or len(ids)!=capacity or len(set(codes))!=capacity or len(set(ids))!=capacity or
        not all(isinstance(c,str) and c for c in codes) or not all(type(i) is int and i>=0 for i in ids)):
        raise RuntimeError('Checkpoint decision configuration disagrees with pinned runtime')
    return dict(requested_choices=requested,official_readout_outputs=capacity,
                checkpoint_codes=len(codes),exact_request_supported=requested<=capacity,
                status='contract_passed_not_inference_tested' if requested<=capacity else 'blocked_choice_capacity',
                weight_download_bytes=0,inference_calls=0,quantization_tested=False,
                candidate_dropped=False,architecture_modified=False)


def hardware():
    info=dict(python=sys.version,python_executable=sys.executable)
    mem=Path('/proc/meminfo')
    if mem.exists():
        values={line.split(':')[0]:line.split(':')[1].strip() for line in mem.read_text().splitlines()}
        info['host_memory']={k:values.get(k) for k in ('MemTotal','MemAvailable','SwapFree')}
    info['cgroup_memory_limits']={str(p):p.read_text().strip() for p in (
        Path('/sys/fs/cgroup/memory.max'),Path('/sys/fs/cgroup/memory/memory.limit_in_bytes')) if p.exists()}
    command=shutil.which('nvidia-smi')
    if command:
        try:
            r=subprocess.run([command,'--query-gpu=index,name,memory.total,memory.free,driver_version',
                              '--format=csv,noheader'],capture_output=True,text=True,timeout=15)
            info['nvidia_smi']=dict(returncode=r.returncode,stdout=r.stdout.strip(),stderr=r.stderr.strip())
        except subprocess.TimeoutExpired:info['nvidia_smi']=dict(error='timeout')
    else:info['nvidia_smi']=dict(error='not found')
    return info


def run(out):
    out.mkdir(parents=True,exist_ok=True)
    report=dict(status='checking',upstream_revision=UPSTREAM,model_revision=REVISION,
                hardware=hardware(),weight_download_bytes=0,inference_calls=0,
                note='Technical contract check only. No installation, weights, CUDA allocations, model inference or dataset outcomes. '
                     'A 4-bit runtime would be a separate unverified model variant; it cannot add a trained readout output.')
    try:
        source=fetch('https://raw.githubusercontent.com/nace-ai/drex-decision-models/'+UPSTREAM+'/drex_max/model.py',out/'metadata/model.py')
        source_capacity(source)  # Verify before accepting any claimed model capability.
        config=fetch('https://huggingface.co/nace-ai/drex-max-v1/resolve/'+REVISION+'/decision_config.json',out/'metadata/decision_config.json')
        if hashlib.sha256(config).hexdigest()!=DECISION_SHA256:raise RuntimeError('Pinned decision_config.json changed')
        report.update(contract(source,json.loads(config)))
        report['metadata_sha256']={n:hashlib.sha256(b).hexdigest() for n,b in [('model.py',source),('decision_config.json',config)]}
        report['next_step']='Do not load weights for exact256. A 255-way or hierarchical experiment requires a separately defined protocol.'
    except Exception as exc:
        report.update(status='technical_failure',error=repr(exc),traceback=traceback.format_exc())
    write(out/'report.json',report);print(json.dumps(report,indent=2),flush=True)
    return 2 if report['status']=='blocked_choice_capacity' else (0 if report['status']=='contract_passed_not_inference_tested' else 1)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    raise SystemExit(run(parser.parse_args().out.resolve()))
