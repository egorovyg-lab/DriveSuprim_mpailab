"""Read-only post-hoc diagnosis of a completed table/image experiment. No inference."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps

import real_data as base
import table_data as data
import table_pilot as pilot
from night import load_result
from runtime import read, write
from transfer import checked, sha, freeze

EPS = 1e-8
NOTE = ('Post-hoc outcome-selected diagnosis of already evaluated TRAIN scenes. No model calls, '
        'new scoring, fitting, prompt changes or holdout. All original files are read-only. '
        'Images are audit material and must not be reused as model inputs: they include outcomes '
        'and an oracle. Nominal path diagrams are not maps, camera projections or controller rollouts. '
        'Probability margins are descriptive, not calibrated risk. Session exclusions are '
        'sensitivity analyses, not a replacement benchmark. No conclusion about visual understanding '
        'can be made without a matched blank-image control.')


def validate_source(source):
    identity, plan = pilot.context(source)
    report, dc = read(source/'report.json'), read(source/'decision_commit.json')
    if (report['status']!='complete' or report['missing_pool_labels']
            or report['decision_commit_sha256']!=sha(source/'decision_commit.json')
            or dc['plan_sha256']!=sha(source/'plan.json')
            or dc['decisions_sha256']!=sha(source/'decisions.json')):
        raise RuntimeError('Complete, unchanged full-cohort decisions/report required')
    decisions = read(source/'decisions.json')
    expected_pairs = {(r['token'],k) for r in identity['records'] for k in data.KS}
    keys = [(d['metadata']['token'],d['k']) for d in decisions]
    if len(keys)!=len(set(keys)) or set(keys)!=expected_pairs:
        raise RuntimeError('Decision cohort/pools incomplete or duplicated')
    if set(dc['workers'])!={str(i) for i in range(identity['replicas'])}:
        raise RuntimeError('Worker commitment membership changed')
    results = {}
    for i,tokens in enumerate(plan['assignments']):
        folder = source/'shards'/f'{i:02d}'; cp = folder/'commit.json'
        if sha(cp)!=dc['workers'][str(i)]: raise RuntimeError('Worker commitment changed')
        commit = read(cp)
        records = [r for r in identity['records'] if r['token'] in set(tokens)]
        tasks = data.tasks(records)
        expected = {data.task_name(*t) for t in tasks}
        if (commit['plan_sha256']!=sha(source/'plan.json') or set(commit['results'])!=expected
                or {p.name for p in (folder/'results').glob('*.json')}!=expected
                or sha(folder/'environment.json')!=commit['environment_sha256']):
            raise RuntimeError('Worker files/environment changed')
        for token,k,mode,kind in tasks:
            name = data.task_name(token,k,mode,kind); path = folder/'results'/name
            if sha(path)!=commit['results'][name] or name in results:
                raise RuntimeError('Changed/duplicate frozen result')
            value = load_result(path)
            pilot.validate_result(value,token,k,mode,kind,data.order(token,k,kind))
            results[name] = value
    if report['requests']!=len(results) or report['scenes']!=len(identity['records']):
        raise RuntimeError('Report/cohort counts changed')
    # Only AFTER every worker's complete commitment has been checked, load outcomes.
    pointer = read(source/'label_source.json'); labels_path = Path(pointer['labels'])
    if (pointer['request_sha256']!=sha(source/'label_request.json')
            or sha(labels_path)!=pointer['sha256'] or sha(labels_path)!=report['external_labels_sha256']):
        raise RuntimeError('Measured label source changed')
    external = read(labels_path); request = read(source/'label_request.json')
    if (external['status']!='complete' or external['request_digest']!=base.digest(request)
            or external['metrics']!=['pdm_score',*base.METRICS]):
        raise RuntimeError('Invalid measured label identity/schema')
    actual = [(r['token'],r['candidate_local'],r['candidate_vocab']) for r in external['rows']]
    wanted = [(r['token'],r['candidate_local'],r['candidate_vocab']) for r in data.requested_candidates(request)]
    if len(actual)!=len(set(actual)) or set(actual)!=set(wanted):
        raise RuntimeError('Measured labels do not cover exactly the frozen pools')
    package = Path(identity['package'])
    if sha(package/'package_manifest.json')!=identity['package_manifest_sha256']:
        raise RuntimeError('Package changed')
    manifest,_,_,vocab = base.package_metadata(package)
    labels = pilot.label_lookup(package,manifest,decisions,external['rows'])
    samples = {}
    for r in identity['records']:
        token = r['token']; entry = plan['entries'][token]; path = source/'inputs'/(token+'.json')
        if sha(path)!=entry['inputs_sha256']: raise RuntimeError('Frozen numeric inputs changed')
        sample = read(path)
        if sample['candidates']!=entry['candidates'] or sample['baseline_local']!=entry['baseline_local']:
            raise RuntimeError('Frozen input alignment changed')
        samples[token] = sample
    references = {r['token']:r for r in identity['records']}
    for d in decisions:
        token,k = d['metadata']['token'],d['k']; sample = samples[token]
        pool = sample['candidates'][:k]
        if (d['metadata']!=references[token] or d['baseline_local']!=sample['baseline_local']
                or d['pool']!=[dict(local=c['local'],vocab=c['vocab']) for c in pool]):
            raise RuntimeError('Decision metadata/candidate alignment changed')
        methods = data.rules(sample,k)
        for mode in data.MODES:
            probabilities = []
            for kind in data.ORDERS:
                r = results[data.task_name(token,k,mode,kind)]
                p = np.empty(k); p[r['pool_positions']] = r['probabilities']; probabilities.append(p)
            methods[mode+'/single'] = data.pick(probabilities[0],sample,k)
            methods[mode+'/mean3'] = data.pick(np.mean(probabilities,axis=0),sample,k)
        if methods!=d['methods']: raise RuntimeError('Decisions disagree with frozen probabilities')
    return identity,plan,report,decisions,results,labels,samples,vocab


def candidate(sample,labels,j):
    rank = next(i for i,c in enumerate(sample['candidates']) if c['local']==j)
    c = sample['candidates'][rank]; raw = labels[j]
    return dict(candidate_local=j,candidate_vocab=c['vocab'],teacher_rank=rank+1,
        raw_pdm=float(raw[0]),raw_metrics=dict(zip(base.METRICS,map(float,raw[1:]))),
        input_features=dict(zip(data.COLUMNS[1:],c['values'])))


def image_category(t,c,b,camera_keeps_ds):
    if c>t+EPS/100:
        if t<b-EPS and abs(c-b)<=EPS:
            return 'avoided_table_loss_by_keeping_ds' if camera_keeps_ds else 'recovered_to_ds_pdm_with_other_candidate'
        if c<b-EPS: return 'reduced_table_loss'
        if c>b+EPS and t<=b+EPS: return 'new_gain_over_ds'
        return 'larger_gain_over_ds'
    if c<t-EPS/100:
        if t>b+EPS and abs(c-b)<=EPS:
            return 'missed_table_gain_by_keeping_ds' if camera_keeps_ds else 'missed_table_gain_with_ds_equal_pdm_alternative'
        if c<b-EPS and t>=b-EPS: return 'new_camera_loss_vs_ds'
        if c<b-EPS and t<b-EPS: return 'larger_loss_vs_ds'
        return 'smaller_gain_over_ds'
    return 'equal_measured_pdm'


def describe(d,sample,labels):
    b = d['baseline_local']; k = d['k']; pool = sample['candidates'][:k]
    maximum = max(labels[c['local']][0] for c in pool)
    oracle = b if labels[b][0]==maximum else next(c['local'] for c in pool if labels[c['local']][0]==maximum)
    chosen = dict(ds=b,table=d['methods']['table/mean3'],camera=d['methods']['table_camera/mean3'],oracle=oracle)
    descriptions = {name:candidate(sample,labels,j) for name,j in chosen.items()}
    baseline = descriptions['ds']['raw_pdm']
    for value in descriptions.values():
        value['gain_pp'] = 100*(value['raw_pdm']-baseline)
        delta = {m:value['raw_metrics'][m]-descriptions['ds']['raw_metrics'][m] for m in base.METRICS}
        value['deltas_vs_ds'] = delta
        value['degraded_components'] = [m for m,v in delta.items() if v<-EPS]
        value['relative_safety_harm'] = any(delta[m]<-EPS for m in base.SAFETY)
    t,c = descriptions['table']['raw_pdm'],descriptions['camera']['raw_pdm']
    table_gain,camera_gain = t-baseline,c-baseline
    return dict(token=d['metadata']['token'],session_id=d['metadata']['session_id'],k=k,
        choices=descriptions,image_minus_table_pp=100*(c-t),image_category=image_category(t,c,baseline,chosen['camera']==b),
        camera_minus_table_components={m:descriptions['camera']['raw_metrics'][m]-descriptions['table']['raw_metrics'][m] for m in base.METRICS},
        image_loss_avoidance_pp=100*(max(0.,-table_gain)-max(0.,-camera_gain)),
        image_gain_capture_pp=100*(max(0.,camera_gain)-max(0.,table_gain)))


def contribution(rows,key):
    sessions = defaultdict(list)
    for r in rows: sessions[r['session_id']].append(r[key])
    return dict(micro_pp=float(np.mean([r[key] for r in rows])),
        macro_pp=float(np.mean([np.mean(v) for v in sessions.values()])))


def pool_effects(cases):
    by_key = {(r['token'],r['k']):r for r in cases}; result = {}
    tokens = sorted({r['token'] for r in cases})
    for mode in ('table','camera'):
        for old_k in (16,32):
            rows = []
            for token in tokens:
                old,new = by_key[token,old_k],by_key[token,64]
                a,b = old['choices'][mode],new['choices'][mode]
                category = ('same_candidate' if a['candidate_local']==b['candidate_local'] else
                            'selected_new_tail' if b['teacher_rank']>old_k else 'changed_within_old_pool')
                rows.append(dict(token=token,session_id=new['session_id'],category=category,
                                 delta=100*(b['raw_pdm']-a['raw_pdm'])))
            groups = {}
            for name in ('same_candidate','selected_new_tail','changed_within_old_pool'):
                selected = [r for r in rows if r['category']==name]
                # Zero-fill other scenes: these contributions add to the FULL-cohort effect.
                zero_filled = [dict(r,delta=r['delta'] if r['category']==name else 0.) for r in rows]
                groups[name] = dict(scenes=len(selected),**contribution(zero_filled,'delta'))
            result[f'{mode}/{old_k}_to_64'] = dict(total=contribution(rows,'delta'),categories=groups)
    return result


def order_effects(decisions,samples,results,labels):
    rows = []
    for d in decisions:
        token,k = d['metadata']['token'],d['k']; sample = samples[token]; lab = labels[token]
        safety = [1+base.METRICS.index(m) for m in base.SAFETY]
        for mode in data.MODES:
            selected = []
            for kind in data.ORDERS:
                value = results[data.task_name(token,k,mode,kind)]
                p = np.empty(k); p[value['pool_positions']] = value['probabilities']
                selected.append(data.pick(p,sample,k))
            canonical = lab[selected[0]]
            for i,kind in enumerate(data.ORDERS[1:],1):
                delta = lab[selected[i]]-canonical
                rows.append(dict(token=token,session_id=d['metadata']['session_id'],k=k,mode=mode,ordering=kind,
                    canonical_local=selected[0],other_local=selected[i],changed=selected[0]!=selected[i],
                    pdm_delta_pp=float(100*delta[0]),same_pdm=bool(abs(delta[0])<=EPS),
                    meaningful_pdm_difference=bool(abs(delta[0])>=.005),
                    components_changed=bool(np.any(np.abs(delta[1:])>EPS)),
                    safety_worse_than_canonical=bool(np.any(delta[safety]<-EPS)),
                    **{'delta_vs_canonical_'+m:float(delta[1+i]) for i,m in enumerate(base.METRICS)}))
    summary = {}
    for key in sorted({(r['k'],r['mode'],r['ordering']) for r in rows}):
        group = [r for r in rows if (r['k'],r['mode'],r['ordering'])==key]
        summary[f'{key[1]}/k{key[0]}/{key[2]}'] = dict(scenes=len(group),
            choice_flips=sum(r['changed'] for r in group),
            flips_with_equal_pdm=sum(r['changed'] and r['same_pdm'] for r in group),
            equal_pdm_with_component_changes=sum(r['changed'] and r['same_pdm'] and r['components_changed'] for r in group),
            meaningful_pdm_differences=sum(r['meaningful_pdm_difference'] for r in group),
            pdm_worse=sum(r['pdm_delta_pp']<-EPS for r in group),
            safety_worse_than_canonical=sum(r['safety_worse_than_canonical'] for r in group),
            component_degradations={m:sum(r['delta_vs_canonical_'+m]<-EPS for r in group) for m in base.METRICS})
    return rows,summary


def check_csv(source,decisions,labels,samples):
    expected = {}
    for d in decisions:
        token,k = d['metadata']['token'],d['k']; lab = labels[token]; baseline = lab[d['baseline_local']]
        methods = dict(d['methods'])
        maximum = max(lab[c['local']][0] for c in d['pool'])
        methods['oracle_upper_bound'] = d['baseline_local'] if baseline[0]==maximum else next(c['local'] for c in d['pool'] if lab[c['local']][0]==maximum)
        for method,j in methods.items(): expected[token,k,method] = (d,j,lab[j],baseline)
    seen = set()
    with (source/'per_scene.csv').open(encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):
            key = r['token'],int(r['k']),r['method']
            if key in seen or key not in expected: raise RuntimeError('Unexpected/duplicate CSV row')
            seen.add(key); d,j,raw,baseline = expected[key]
            cv = next(c['vocab'] for c in samples[r['token']]['candidates'] if c['local']==j)
            safety = [1+base.METRICS.index(m) for m in base.SAFETY]
            checks = dict(intervention=j!=d['baseline_local'],relative_component_harm=bool(np.any((raw-baseline)[safety]<-EPS)),
                          pdm_decreased=bool(raw[0]<baseline[0]-EPS))
            if (r['session_id']!=d['metadata']['session_id'] or int(r['candidate_local'])!=j
                    or int(r['candidate_vocab'])!=cv or any(r[k]!=str(v) for k,v in checks.items())):
                raise RuntimeError('CSV candidate/flag mismatch')
            numeric = dict(raw_pdm=raw[0],gain_pp=100*(raw[0]-baseline[0]),
                           **{'delta_'+m:raw[1+i]-baseline[1+i] for i,m in enumerate(base.METRICS)})
            if any(not np.isfinite(float(r[k])) or abs(float(r[k])-v)>EPS for k,v in numeric.items()):
                raise RuntimeError('CSV numeric metrics mismatch')
    if seen!=set(expected): raise RuntimeError('Incomplete CSV cohort')
    return len(seen)


def sensitivity(cases):
    sessions = defaultdict(list)
    for r in cases: sessions[r['session_id']].append(r)
    result = {}
    for mode in ('table','camera'):
        gains = {s:float(np.mean([r['choices'][mode]['gain_pp'] for r in group])) for s,group in sessions.items()}
        ranked = sorted(gains,key=lambda s:(-gains[s],s)); total = sum(gains.values()); n = len(gains)
        result[mode] = dict(gain_macro_pp=total/n,
            sessions_positive=sum(v>EPS for v in gains.values()),sessions_negative=sum(v<-EPS for v in gains.values()),
            sessions_zero=sum(abs(v)<=EPS for v in gains.values()),
            without_best_session_macro_pp=(total-gains[ranked[0]])/(n-1) if n>1 else None,
            without_best_two_macro_pp=(total-sum(gains[s] for s in ranked[:2]))/(n-2) if n>2 else None,
            per_session=[dict(session_id=s,scenes=len(sessions[s]),gain_pp=gains[s],contribution_macro_pp=gains[s]/n,
                pdm_losses=sum(r['choices'][mode]['gain_pp']<-100*EPS for r in sessions[s]),
                interventions=sum(r['choices'][mode]['candidate_local']!=r['choices']['ds']['candidate_local'] for r in sessions[s])) for s in ranked])
    return result


def write_csv(path,rows):
    if not rows: return
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix('.csv.tmp')
    with tmp.open('w',encoding='utf-8',newline='') as f:
        w = csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    tmp.replace(path)


def preview(case,photo,vocab,path):
    names = ('ds','table','camera','oracle')
    trajectories = [vocab[case['choices'][name]['candidate_vocab']] for name in names]
    all_xy = np.vstack((np.zeros((1,2)),np.asarray(trajectories)[:,:,:2].reshape(-1,2)))
    lo,hi = all_xy.min(axis=0)-2,all_xy.max(axis=0)+2
    scale = min(260/max(hi[0]-lo[0],1),320/max(hi[1]-lo[1],1))
    board = Image.new('RGB',(1536,1100),'white'); draw = ImageDraw.Draw(board)
    draw.text((12,6),f'AUDIT ONLY | {case["token"]} | outcome-selected case',fill='black',font=base.font(20))
    fitted = ImageOps.contain(photo,(1536,640)); board.paste(fitted,((1536-fitted.width)//2,34))
    for j,(name,traj,color) in enumerate(zip(names,trajectories,('#0057a8','#d07800','#00834a','#b30042'))):
        ox = 384*j; value = case['choices'][name]
        draw.text((ox+8,685),f'{name.upper()} | DS rank {value["teacher_rank"]}',fill=color,font=base.font(18))
        draw.text((ox+8,709),f'Raw PDM {value["raw_pdm"]:.4f} | gain {value["gain_pp"]:+.2f} pp',fill='black',font=base.font(15))
        def pixel(x,y): return ox+192-(y-(lo[1]+hi[1])/2)*scale, 880-(x-(lo[0]+hi[0])/2)*scale
        points = [pixel(0,0)]+[pixel(x,y) for x,y,_ in traj]
        draw.line(points,fill=color,width=3)
        x,y = pixel(0,0); draw.ellipse((x-4,y-4,x+4,y+4),fill='black')
        for second in (1,2,3,4):
            x,y = pixel(*traj[10*second-1,:2]); draw.ellipse((x-3,y-3,x+3,y+3),fill=color)
        m = value['raw_metrics']
        draw.text((ox+8,1035),f'NC {m["no_at_fault_collisions"]:.2f} DAC {m["drivable_area_compliance"]:.2f} TTC {m["time_to_collision_within_bound"]:.2f}',fill='black',font=base.font(15))
        draw.line((ox,680,ox,1060),fill='#aaaaaa')
    draw.text((12,1070),'Nominal ego-local paths only. Common metric scale; up=forward. NOT a map, camera projection or simulated rollout.',fill='black',font=base.font(17))
    board.save(path)


def run(source,out,max_cases=64,previews=True):
    source,out = source.resolve(),out.resolve()
    if source==out or source in out.parents or out in source.parents:
        raise ValueError('Audit must use a separate sibling output directory')
    started = time.monotonic()
    identity,plan,original,decisions,results,labels,samples,vocab = validate_source(source)
    package = Path(identity['package']).resolve()
    if out==package or package in out.parents or out in package.parents:
        raise ValueError('Audit output must not overlap the original package')
    if max_cases<1: raise ValueError('max_cases must be positive')
    freeze(out/'identity.json',dict(protocol='table_case_audit_v1',source=str(source),
        decision_commit_sha256=sha(source/'decision_commit.json'),report_sha256=sha(source/'report.json'),
        labels_sha256=original['external_labels_sha256'],per_scene_sha256=sha(source/'per_scene.csv'),
        sources={n:sha(Path(__file__).with_name(n)) for n in ('audit_table_cases.py','run_table_case_audit.sh')},
        max_cases=max_cases,previews=previews,note=NOTE))
    validated_csv = check_csv(source,decisions,labels,samples)
    cases = [describe(d,samples[d['metadata']['token']],labels[d['metadata']['token']]) for d in decisions]
    primary = [r for r in cases if r['k']==64]
    for k in data.KS:
        group = [r for r in cases if r['k']==k]
        paired = original['paired_image_ablation'][str(k)]
        if (sum(r['image_minus_table_pp']>EPS for r in group)!=paired['camera_better_scenes']
                or sum(r['image_minus_table_pp']<-EPS for r in group)!=paired['camera_worse_scenes']
                or abs(contribution(group,'image_minus_table_pp')['macro_pp']-paired['camera_minus_table_gain_macro_pp'])>EPS):
            raise RuntimeError('Recomputed image pairing differs from original report')
        for mode,method in (('table','table'),('camera','table_camera')):
            means = sensitivity(group)[mode]['gain_macro_pp']
            previous = original['methods'][f'{method}/mean3/k{k}']
            if (abs(means-previous['gain_macro_pp'])>EPS
                    or sum(r['choices'][mode]['gain_pp']<-100*EPS for r in group)!=previous['pdm_decreased']
                    or sum(r['choices'][mode]['relative_safety_harm'] for r in group)!=previous['relative_component_harm']
                    or sum(r['choices'][mode]['candidate_local']!=r['choices']['ds']['candidate_local'] for r in group)!=previous['interventions']):
                raise RuntimeError('Recomputed macro gain differs from original report')
    order_rows,order_summary = order_effects(decisions,samples,results,labels)
    for key,value in order_summary.items():
        if value['choice_flips']!=original['order_sensitivity'][key]['choice_flips']:
            raise RuntimeError('Recomputed order effects differ from original report')
    categories = dict(
        image_worse=sorted([r for r in primary if r['image_minus_table_pp']<-EPS],key=lambda r:r['image_minus_table_pp']),
        camera_pdm_losses=sorted([r for r in primary if r['choices']['camera']['gain_pp']<-100*EPS],key=lambda r:r['choices']['camera']['gain_pp']),
        table_safety_losses=sorted([r for r in primary if r['choices']['table']['relative_safety_harm']],key=lambda r:r['choices']['table']['gain_pp']),
        camera_safety_losses=sorted([r for r in primary if r['choices']['camera']['relative_safety_harm']],key=lambda r:r['choices']['camera']['gain_pp']),
        image_better=sorted([r for r in primary if r['image_minus_table_pp']>EPS],key=lambda r:-r['image_minus_table_pp']),
        table_largest_gains=sorted([r for r in primary if r['choices']['table']['gain_pp']>EPS],key=lambda r:-r['choices']['table']['gain_pp'])[:10],
        camera_largest_gains=sorted([r for r in primary if r['choices']['camera']['gain_pp']>EPS],key=lambda r:-r['choices']['camera']['gain_pp'])[:10])
    selected = []
    for rows in categories.values():
        for r in rows:
            if r['token'] not in {s['token'] for s in selected}: selected.append(r)
    selected = selected[:max_cases]
    index_lines = ['# Post-hoc table/image case audit','',NOTE,'',
        '| Case | Token | Categories | DS PDM | Table PDM | Camera PDM | Photo minus table, pp | Preview |',
        '|---|---|---|---:|---:|---:|---:|---|']
    write(out/'all_scene_details.json',cases)
    flat = [dict(token=r['token'],session_id=r['session_id'],k=r['k'],image_category=r['image_category'],
        image_minus_table_pp=r['image_minus_table_pp'],image_loss_avoidance_pp=r['image_loss_avoidance_pp'],
        image_gain_capture_pp=r['image_gain_capture_pp'],
        **{name+'_'+key:v[key] for name,v in r['choices'].items() for key in ('candidate_local','candidate_vocab','teacher_rank','raw_pdm','gain_pp')}) for r in cases]
    write_csv(out/'image_pairs.csv',flat); write_csv(out/'order_effects.csv',order_rows)
    for i,r in enumerate(selected,1):
        folder = out/'cases'/f'{i:03d}-{r["token"]}'; folder.mkdir(parents=True,exist_ok=True)
        detail = dict(r,categories=[name for name,group in categories.items() if r['token'] in {g['token'] for g in group}],
                      measured_outcomes_used_to_select_case=True,note=NOTE)
        link = f'[image](cases/{folder.name}/preview.png)' if previews else 'disabled'
        index_lines.append(f'| {i:03d} | {r["token"]} | {", ".join(detail["categories"])} | '
            f'{r["choices"]["ds"]["raw_pdm"]:.4f} | {r["choices"]["table"]["raw_pdm"]:.4f} | '
            f'{r["choices"]["camera"]["raw_pdm"]:.4f} | {r["image_minus_table_pp"]:+.4f} | {link} |')
        sample = samples[r['token']]
        candidate_rows = []
        for c in sample['candidates']:
            desc = candidate(sample,labels[r['token']],c['local'])
            candidate_rows.append(dict(candidate_local=c['local'],candidate_vocab=c['vocab'],teacher_rank=desc['teacher_rank'],
                raw_pdm=desc['raw_pdm'],**{'raw_'+m:v for m,v in desc['raw_metrics'].items()},**desc['input_features']))
        write(folder/'summary.json',detail); write_csv(folder/'candidate_metrics.csv',candidate_rows)
        if previews:
            photo_path = source/'photos'/(r['token']+'.png')
            if sha(photo_path)!=plan['entries'][r['token']]['photo_sha256']: raise RuntimeError('Audit photo changed')
            with Image.open(photo_path) as image: photo = image.convert('RGB')
            # Recreate the six primary requests ONLY for selected cases, as an extra input check.
            for mode in data.MODES:
                for kind in data.ORDERS:
                    _,_,digest = data.payload(r['token'],sample,64,mode,kind,photo)
                    if digest!=results[data.task_name(r['token'],64,mode,kind)]['request_sha256']:
                        raise RuntimeError('Selected case request no longer matches original pixels/table')
            preview(r,photo,vocab,folder/'preview.png')
        print(f'CASE AUDIT {i}/{len(selected)} {r["token"]}',flush=True)
    (out/'index.md').write_text('\n'.join(index_lines)+'\n',encoding='utf-8')
    decomposition = {}
    for k in data.KS:
        group = [r for r in cases if r['k']==k]
        decomposition[str(k)] = dict(image_categories=dict(Counter(r['image_category'] for r in group)),
            equal_pdm_with_component_changes=sum(abs(r['image_minus_table_pp'])<=EPS and any(abs(v)>EPS for v in r['camera_minus_table_components'].values()) for r in group),
            net=contribution(group,'image_minus_table_pp'),
            loss_avoidance=contribution(group,'image_loss_avoidance_pp'),gain_capture=contribution(group,'image_gain_capture_pp'))
    summary = dict(status='complete',scenes=len(primary),sessions=len({r['session_id'] for r in primary}),
        source=str(source),source_csv_rows_verified=validated_csv,source_queries_verified=len(results),
        case_counts={name:len(rows) for name,rows in categories.items()},case_previews=len(selected) if previews else 0,
        selected_cases=[dict(token=r['token'],image_delta_pp=r['image_minus_table_pp'],camera_gain_pp=r['choices']['camera']['gain_pp']) for r in selected],
        image_effect_decomposition=decomposition,pool_expansion_effects=pool_effects(cases),
        order_effects=order_summary,session_sensitivity=sensitivity(primary),
        seconds=time.monotonic()-started,note=NOTE)
    write(out/'report.json',summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--max-cases',type=int,default=64); p.add_argument('--no-previews',action='store_true')
    a = p.parse_args()
    result = run(a.source,a.out,a.max_cases,not a.no_previews)
    compact = {k:result[k] for k in ('status','scenes','sessions','source_csv_rows_verified','source_queries_verified','case_counts','image_effect_decomposition','pool_expansion_effects','seconds')}
    print(json.dumps(compact,indent=2),flush=True)


if __name__=='__main__': main()
