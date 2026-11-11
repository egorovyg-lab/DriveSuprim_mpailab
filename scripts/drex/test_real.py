from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

AVAILABLE=bool(importlib.util.find_spec('numpy') and importlib.util.find_spec('PIL'))
if AVAILABLE:
    import numpy as np
    from PIL import Image
    import real_data as data
    import real_pilot as pilot
    from runtime import read, write
    from transfer import sha


@unittest.skipUnless(AVAILABLE,'NumPy and Pillow required; use the compute-node runtime for real-pilot checks')
class RealPilotTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=str(Path.cwd()))
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name); self.package=self.root/'package'; self.out=self.root/'run'
        self.package.mkdir()
        self.capture=redirect_stdout(io.StringIO()); self.capture.__enter__()
        self.addCleanup(self.capture.__exit__,None,None,None)
        self.vocab=np.zeros((300,40,3),dtype=np.float32)
        for i in range(300):
            self.vocab[i,:,0]=np.linspace(.1,4,40)*(i%20+1)
            self.vocab[i,:,1]=np.linspace(.1,4,40)*(i%7-3)
        self.sample=dict(candidate_ids=np.arange(256),selection_scores=np.arange(256,dtype=float),
                         selected_local=np.asarray(37),ego_status=np.asarray([1,0,0,0,4,0,.1,0]*2,dtype=float))
        self.sample['selection_scores'][37]=1000.
        self.inputs=data.inputs_from_arrays(self.sample,self.vocab)
        self.index={}; self.scenes=[]
        for i in range(4):
            token=f'token{i}'; meta=dict(token=token,role='train',session_id=f's{i//2}',origin='fixture')
            raw=np.ones((16,9)); raw[:,0]=.8; raw[5,0]=.9; raw[3,0]=.123; raw[3,2]=0
            path=self.package/f'bundle/examples/{token}.npz'; path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,**self.sample,label_indices=self.inputs['pool_local'],label_raw=raw,label_filtered=raw)
            self.index[token]=dict(metadata=meta,sha256=sha(path))
            history=[]
            for h in range(4):
                cams={}
                for c in ('cam_l0','cam_f0','cam_r0','cam_l1','cam_l2','cam_r1','cam_r2','cam_b0'):
                    name=f'images/{token}/{h}-{c}.jpg'; image=self.package/name; image.parent.mkdir(parents=True,exist_ok=True)
                    Image.new('RGB',(64,32),(20+h*40,30,70)).save(image)
                    cams[c]=name
                history.append(dict(frame_token=token if h==3 else f'{token}-past{h}',cameras=cams))
            self.scenes.append(dict(metadata=meta,history=history))
        bundle=self.package/'bundle'
        np.savez_compressed(bundle/'vocab.npz',vocab=self.vocab)
        counts=dict(scenes=4,sessions=2)
        write(bundle/'identity.json',dict(counts=counts,protected_sessions=['reserved']))
        write(bundle/'examples_index.json',self.index)
        m={key:sha(bundle/name) for name,key in (('identity.json','identity_sha256'),
            ('examples_index.json','index_sha256'),('vocab.npz','vocab_sha256'))}
        m.update(counts=counts,label_keys=['pdm_score',*data.METRICS])
        write(bundle/'bundle_manifest.json',m)
        write(self.package/'scene_inputs.json',self.scenes)
        files={p.relative_to(self.package).as_posix():dict(size=p.stat().st_size,sha256=sha(p))
               for p in self.package.rglob('*') if p.is_file()}
        write(self.package/'package_manifest.json',dict(protocol='portable_train_v1',holdout_scenes=0,
            counts=counts,files=files,images=128,total_bytes=sum(e['size'] for e in files.values())))
        write(self.root/'model_plan.json',dict(revision='a'*40,files=[]))
        (self.root/'requirements.lock.txt').write_text('fixture==1\n')
        self.manifest,self.records,self.scene_map,self.loaded_vocab=data.prepare(
            self.package,self.root,self.out,sessions=2,per_session=2)

    def infer(self,row):
        self.assertEqual(set(row),{'state','question','images'})
        self.assertEqual(len(row['question']['criteria']),16)
        self.assertEqual(len(row['images']),2)
        # Deliberately use model-slot preference, not expected labels or physical IDs.
        return dict(probabilities=[1.]+[0.]*15,seconds=.1)

    def run_infer(self,**kwargs):
        return pilot.infer_cohort(self.package,self.out,self.manifest,self.records,self.scene_map,
                                  self.loaded_vocab,self.infer,**kwargs)

    def test_selection_is_session_balanced_deterministic_and_ignores_extra_metadata(self):
        a=data.choose_records(self.index,2,2)
        self.assertEqual(a,data.choose_records(dict(reversed(list(self.index.items()))),2,2))
        self.assertEqual(len({r['session_id'] for r in a[:2]}),2)
        changed={t:dict(e,unused_outcome=999) for t,e in self.index.items()}
        self.assertEqual(a,data.choose_records(changed,2,2))
        with self.assertRaisesRegex(ValueError,'declared sessions'):
            data.choose_records(self.index,3,2)

    def test_input_allowlist_never_deserializes_labels_or_teacher_heads(self):
        class Guard:
            def __getitem__(_,key):
                if key not in data.INPUT_KEYS: raise AssertionError('Outcome read: '+key)
                return self.sample[key]
        inputs=data.inputs_from_arrays(Guard(),self.vocab)
        token=self.records[0]['token']
        row,_,_=data.payload(inputs,self.scene_map[token],self.package,self.manifest,'camera','canonical')
        text=str({k:v for k,v in row.items() if k!='images'})
        for forbidden in ('pdm_score','label_raw','selection_scores','pool_local',token,'head_'):
            self.assertNotIn(forbidden,text)
        self.assertEqual(inputs['baseline_local'],37)
        self.assertEqual(inputs['pool_local'][0],37)

    def test_ensemble_aligns_physical_candidates_and_preserves_baseline_on_exact_ties(self):
        results=[]; samples={r['token']:self.inputs for r in self.records}
        for r in self.records:
            for mode,kind in data.TASKS:
                order=data.permutations(r['token'],kind); probabilities=[0.]*16
                probabilities[order.index(5)]=1.
                results.append(dict(token=r['token'],mode=mode,ordering=kind,pool_positions=order,probabilities=probabilities))
        decisions=pilot.decisions_from_results(self.records,samples,results)
        self.assertTrue(all(d['methods']['drex_mean3']['candidate_local']==self.inputs['pool_local'][5] for d in decisions))
        self.assertEqual(pilot.choice([1/16]*16,self.inputs),37)

    def test_vocabulary_ids_are_independent_of_local_pool_indices(self):
        changed=dict(self.sample,candidate_ids=np.roll(np.arange(256),73))
        inputs=data.inputs_from_arrays(changed,self.vocab)
        ids=changed['candidate_ids'][inputs['pool_local']]
        self.assertNotEqual(inputs['pool_vocab'][0],inputs['baseline_local'])
        self.assertEqual(inputs['pool_vocab'],ids.tolist())
        np.testing.assert_array_equal(inputs['trajectories'],self.vocab[ids])

    def test_camera_ablation_changes_only_camera_pixels_not_trajectories_or_state(self):
        r=self.records[0]; token=r['token']
        a,ids,ha=data.payload(self.inputs,self.scene_map[token],self.package,self.manifest,'camera','canonical')
        b,ids2,hb=data.payload(self.inputs,self.scene_map[token],self.package,self.manifest,'blank','canonical')
        self.assertEqual(ids,ids2)
        self.assertEqual(a['state'],b['state']); self.assertEqual(a['question'],b['question'])
        self.assertEqual(a['images'][1].tobytes(),b['images'][1].tobytes())
        self.assertNotEqual(a['images'][0].tobytes(),b['images'][0].tobytes())
        self.assertNotEqual(ha,hb)

    def test_partial_run_cannot_read_outcomes_and_resume_finishes_whole_cohort(self):
        with patch.object(pilot,'load_labels',side_effect=AssertionError('Premature outcome read')) as labels:
            partial=self.run_infer(max_new=5)
            self.assertEqual(partial['completed_requests'],5)
            self.assertFalse((self.out/'decision_commit.json').exists())
            with self.assertRaises(FileNotFoundError):
                pilot.evaluate(self.package,self.out,self.manifest,self.records,self.loaded_vocab)
            labels.assert_not_called()
            complete=self.run_infer()
            self.assertEqual(complete['status'],'choices_frozen')
            self.assertEqual(complete['new_requests'],11)
            labels.assert_not_called()
        report=pilot.evaluate(self.package,self.out,self.manifest,self.records,self.loaded_vocab)
        self.assertEqual(report['status'],'complete')
        self.assertEqual(report['scenes'],4)
        self.assertEqual(report['requests'],16)
        self.assertEqual(report['methods']['baseline']['gain_macro_pp'],0)

    def test_stored_raw_score_used_without_component_reconstruction(self):
        decisions=[]; labels={}
        for r in self.records:
            raw=np.ones((16,9)); raw[:,0]=.8; raw[3,0]=.123; raw[3,2]=0
            labels[r['token']]=raw
            decisions.append(dict(metadata=r,pool_local=self.inputs['pool_local'],pool_vocab=self.inputs['pool_vocab'],
                baseline_local=37,methods=dict(baseline=dict(candidate_local=37,candidate_vocab=37),
                drex_mean3=dict(candidate_local=self.inputs['pool_local'][3],candidate_vocab=self.inputs['pool_vocab'][3]))))
        rows=pilot.score_decisions(decisions,labels)
        chosen=[r for r in rows if r['method']=='drex_mean3']
        self.assertTrue(all(r['raw_pdm']==.123 and r['raw_harm'] and r['pdm_decreased'] for r in chosen))
        self.assertAlmostEqual(chosen[0]['raw_gain_pp'],-67.7)

    def test_changed_result_and_incomplete_commit_block_label_lookup(self):
        self.run_infer()
        path=next((self.out/'results').glob('*.json'))
        path.write_text(path.read_text()+' ')
        with patch.object(pilot,'load_labels',side_effect=AssertionError('Premature outcome read')) as labels:
            with self.assertRaisesRegex(RuntimeError,'Committed result changed'):
                pilot.evaluate(self.package,self.out,self.manifest,self.records,self.loaded_vocab)
            labels.assert_not_called()

    def test_technical_failure_is_saved_and_not_retried(self):
        calls=[]
        def fail(row): calls.append(row); raise RuntimeError('GPU test failure')
        args=(self.package,self.out,self.manifest,self.records,self.scene_map,self.loaded_vocab,fail)
        with self.assertRaisesRegex(RuntimeError,'GPU test failure'):
            pilot.infer_cohort(*args)
        with self.assertRaisesRegex(RuntimeError,'Recorded technical failure'):
            pilot.infer_cohort(*args)
        self.assertEqual(len(calls),1)


if __name__=='__main__': unittest.main()
