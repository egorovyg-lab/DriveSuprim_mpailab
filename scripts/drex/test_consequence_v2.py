import ast
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import consequence_v2 as v2
import consequence_model as old
import consequence_run as storage
from test_consequence_probe import fixture


class V2Tests(unittest.TestCase):
    def test_logistic_constant_classes_and_train_only_scaling(self):
        rng=np.random.default_rng(3); x=rng.normal(size=(8,63,4)); y=rng.normal(size=(8,63,7))
        b=np.zeros((8,63,5)); b[:,:,0]=(x[:,:,0]>0)
        heads=v2.fit_heads(x,y,b,np.array(['a','a','b','b','c','c','d','d']))
        _, logits=v2.predict_heads(heads,x)
        self.assertTrue(np.isfinite(logits).all()); self.assertTrue(heads[2][1:].all())
        loss=np.mean(np.logaddexp(0.,logits[:,:,0])-b[:,:,0]*logits[:,:,0])
        self.assertLess(loss,.4)
        self.assertLess(v2.sigmoid(logits[:,:,1]).max(),1e-5)
        mean=heads[0]['x_mean'].copy(); v2.predict_heads(heads,x*1e6)
        np.testing.assert_array_equal(mean,heads[0]['x_mean'])

    def test_targets_separate_progress_from_safety_and_geometry(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); fixture(root); o=storage.load_arrays(root/'labels.npz')['outcomes']
            ix={k:i for i,k in enumerate(v2.data.OUTCOMES)}
            o[:,1,ix['route_progress_m']]=5
            c,b,_=v2.targets(o)
            self.assertTrue((b[:,0,0]==0).all()); self.assertTrue((b[:,0,1]==0).all())
            self.assertTrue((b[:,0,2]==1).all())

    def test_gate_tuning_no_harm_and_minimum_session_support(self):
        scores=np.zeros((4,63,5)); scores[:,0,4]=.8; pred=np.zeros((4,63,7))
        binary=np.zeros((4,63,5)); binary[:,0,4]=1
        sessions=np.array(['a','b','c','d'])
        selected=v2.tune_gate(scores,pred,pred,binary,sessions)
        self.assertEqual(selected['support_sessions'],4)
        binary[:,0,0]=1
        self.assertIn('fallback',v2.tune_gate(scores,pred,pred,binary,sessions))

    def test_nested_training_receives_only_outer_training_sessions(self):
        rng=np.random.default_rng(7); x=rng.normal(size=(12,63,3)); c=rng.normal(size=(12,63,7))
        b=np.zeros((12,63,5)); b[:,:,0]=(x[:,:,0]>0); b[:,:,4]=(x[:,:,1]>1)
        sessions=np.array(['s'+str(i//2) for i in range(12)])
        called=[]; original=v2.fit_heads
        def tracked(a, y, labels, sid):
            called.append(set(sid)); return original(a,y,labels,sid)
        with patch.object(v2,'fit_heads',side_effect=tracked):
            heads, calibration, selection=v2.fit_outer(x[:8],c[:8],b[:8],sessions[:8])
        self.assertTrue(all(not (s & set(sessions[8:])) for s in called))
        _, logits=v2.predict_heads(heads,x[8:]); probabilities=v2.probabilities(logits,calibration)
        self.assertTrue(((probabilities>=0)&(probabilities<=1)).all())

    def test_pipeline_pilot_resume_binding_and_no_outer_target_tuning(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); fixture(root); out=root/'v2'
            def fake_fit(x,c,b,s):
                return v2.fit_heads(x,c,b,s),np.column_stack((np.zeros(5),np.ones(5))),dict(gate=[0.,1.01,0.])
            with patch.object(v2,'fit_outer',side_effect=fake_fit), patch('builtins.print'):
                pilot=v2.run(root,out,'pilot'); before=storage.sha(out/'folds/0/ds/predictions.npz')
                full=v2.run(root,out,'full')
            self.assertEqual(pilot['status'],'pilot_complete'); self.assertEqual(full['status'],'complete')
            self.assertEqual(full['scenes'],16); self.assertEqual(before,storage.sha(out/'folds/0/ds/predictions.npz'))
            with (out/'folds/0/ds/predictions.npz').open('ab') as f:f.write(b'changed')
            with self.assertRaisesRegex(RuntimeError,'Checkpoint'):
                v2.run(root,out,'full')

    def test_python39_and_cpu_launcher(self):
        ast.parse(Path(v2.__file__).read_text(),feature_version=(3,9))
        import test_runtime as fixtures
        f=fixtures.SlurmRoutingTests('test_internal_mode_requires_a_real_job_step'); f.setUp(); self.addCleanup(f.doCleanups)
        f.script=Path(v2.__file__).with_name('run_consequence_v2.sh')
        (f.envroot/'setup_node.txt').write_text('cn29\n')
        f.stub('sbatch','printf "%s\\n" "$*" >> "$DREX_TEST_LOG"\n')
        r=f.run_stage('submit'); self.assertEqual(r.returncode,0,r.stderr)
        text=f.log.read_text(); self.assertIn('--gres=none',text); self.assertNotIn('--gres=gpu',text)
        self.assertIn('--chdir=',text); self.assertIn('--inside ',text)

    def test_batch_spool_copy_uses_explicit_checkout_and_runtime(self):
        import test_runtime as fixtures
        f=fixtures.SlurmRoutingTests('test_internal_mode_requires_a_real_job_step'); f.setUp(); self.addCleanup(f.doCleanups)
        repo=f.root/'checkout with spaces'; scripts=repo/'scripts/drex'; scripts.mkdir(parents=True)
        (scripts/'consequence_v2.py').write_text('# fixture\n')
        spool=f.root/'spool/job123'; spool.mkdir(parents=True)
        copy=spool/'slurm_script'
        copy.write_bytes(Path(v2.__file__).with_name('run_consequence_v2.sh').read_bytes())
        f.script=copy; f.env['SLURM_JOB_ID']='123'
        venv=f.envroot/'venv/bin'; venv.mkdir(parents=True)
        python=venv/'python'
        python.write_text('#!/usr/bin/env bash\nprintf "%s|%s\\n" "$PWD" "$*" >> "$DREX_TEST_CALLS"\n',newline='\n')
        python.chmod(0o755); f.stub('flock','exit 0\n')
        source=f.envroot/'runs/original_probe'
        result=f.run_stage('--inside',f.posix(repo),f.posix(f.envroot),'v2_test',f.posix(source))
        self.assertEqual(result.returncode,0,result.stderr)
        calls=f.calls.read_text().splitlines(); self.assertEqual(len(calls),3)
        self.assertTrue(all(line.startswith(f.posix(repo)+'|') for line in calls))
        self.assertIn('--stage pilot',calls[1]); self.assertIn('--stage full',calls[2])
        self.assertIn('--source '+f.posix(source),calls[2])
        result=f.run_stage('--inside'); self.assertNotEqual(result.returncode,0)


if __name__=='__main__':unittest.main()
