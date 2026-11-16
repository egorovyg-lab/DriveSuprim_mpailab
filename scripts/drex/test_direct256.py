import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import direct256 as d


def fixture():
    vocab=np.zeros((256,40,3)); vocab[:,:,0]=np.arange(1,41)[None,:]*.2
    vocab[:,:,1]=np.arange(256)[:,None]*.001
    arrays=dict(candidate_ids=np.arange(256),ego_status=np.array([0,0,0,0,2.,0.,.1,0.]))
    arrays.update({'head_'+m:np.zeros(256) for m in d.base.METRICS})
    return arrays,vocab


class DirectTests(unittest.TestCase):
    def test_only_allowed_arrays_and_256_complete_rows(self):
        a,v=fixture()
        class Whitelist(dict):
            def __getitem__(self,key):
                if key not in a: raise AssertionError('Forbidden array read: '+key)
                return a[key]
        s=d.inputs(Whitelist(a),v)
        self.assertEqual(len(s['candidates']),256)
        self.assertEqual(len(s['candidates'][0]['values']),len(d.COLUMNS)-1)
        self.assertEqual(s['candidates'][0]['values'][:8],[.5]*8)
        self.assertEqual(s['candidates'][0]['values'][8],2.)
        self.assertEqual(s['candidates'][0]['values'][10],4.)
        self.assertEqual(s['candidates'][0]['values'][12],8.)

    def test_payload_no_teacher_rank_score_imi_baseline_or_images(self):
        a,v=fixture(); s=d.inputs(a,v)
        s.update(baseline_local=19,ds_score_gap=99,imitation_log_probability_256=100,images=['forbidden'])
        row,ids,_=d.payload('token',s,'primary')
        self.assertEqual(len(row['question']['criteria']),256)
        self.assertEqual(set(ids),set(range(256))); self.assertNotEqual(ids,list(range(256)))
        text=json.dumps(row)
        for forbidden in ('baseline','ds_score_gap','imitation_log','images','teacher_rank','change_if_replaced'):
            self.assertNotIn(forbidden,text)
        self.assertEqual(len(row['state']['candidate_table'].splitlines()),257)

    def test_reversed_order_preserves_physical_aliases_and_values(self):
        a,v=fixture(); s=d.inputs(a,v)
        p,ids,_=d.payload('token',s,'primary'); q,reverse,_=d.payload('token',s,'reversed')
        self.assertEqual(ids,list(reversed(reverse)))
        self.assertEqual(list(p['question']['criteria']),list(reversed(q['question']['criteria'])))
        self.assertEqual(p['state']['candidate_table'].splitlines()[1:],list(reversed(q['state']['candidate_table'].splitlines()[1:])))

    def test_rejects_invalid_candidate_and_head_alignment(self):
        a,v=fixture(); a['candidate_ids'][1]=0
        with self.assertRaises(ValueError):d.inputs(a,v)
        a,v=fixture(); a['head_ego_progress'][1]=np.nan
        with self.assertRaises(ValueError):d.inputs(a,v)

    def test_pilot_eight_queries_four_distinct_sessions_and_python39(self):
        identity=dict(records=[dict(token=str(i)) for i in range(7)],pilot_tokens=['0','1','2','3'])
        self.assertEqual(len(d.tasks(identity,'pilot')),8)
        self.assertEqual(len(d.tasks(identity,'full')),14)
        ast.parse(Path(d.__file__).read_text(),feature_version=(3,9))

    def test_slurm_one_allocation_and_explicit_spool_paths(self):
        import test_runtime as fixture
        f=fixture.SlurmRoutingTests('test_internal_mode_requires_a_real_job_step');f.setUp();self.addCleanup(f.doCleanups)
        f.script=Path(d.__file__).with_name('run_direct256.sh')
        (f.envroot/'setup_node.txt').write_text('cn29\n')
        f.stub('sbatch','printf "%s\\n" "$@" > "$DREX_TEST_LOG"\n')
        r=f.run_stage('submit-pilot','4');self.assertEqual(r.returncode,0,r.stderr)
        text=f.log.read_text();self.assertIn('--gres=gpu:4',text);self.assertIn('--ntasks=4',text)
        self.assertIn('--chdir=',text);self.assertIn('coordinator-pilot',text)

    def test_report_preserves_physical_choice_and_does_not_invent_scores(self):
        a,v=fixture();sample=d.inputs(a,v)
        identity=dict(records=[dict(token='one'),dict(token='two')],pilot_tokens=['one','two'])
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            for i,(token,order) in enumerate(d.tasks(identity,'pilot')):
                _,ids,digest=d.payload(token,sample,order); probabilities=[0.]*256
                probabilities[ids.index(200)]=1.
                d.write(out/'workers'/str(i%2)/'results'/(token+'-'+order+'.json'),
                        dict(request_sha256=digest,local_ids=ids,probabilities=probabilities))
            with patch.object(d,'context',return_value=(identity,{})),patch.object(d,'checked_sample',return_value=sample),patch('builtins.print'):
                d.report(out,out,2,'pilot')
            r=d.read(out/'pilot_report.json'); decisions=d.read(out/'pilot_decisions.json')
            self.assertEqual(r['status'],'pilot_complete');self.assertFalse(r['performance_evaluated'])
            self.assertFalse(r['outcomes_read']);self.assertEqual(r['ordering_flips'],0)
            self.assertTrue(all(row['candidate_local']==200 for row in decisions))


if __name__=='__main__':unittest.main()
