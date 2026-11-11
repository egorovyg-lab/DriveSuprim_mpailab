import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import test_runtime as slurm_fixtures
AVAILABLE=bool(importlib.util.find_spec('numpy') and importlib.util.find_spec('PIL'))
if AVAILABLE:
    import real_data as data
    import real_pilot as pilot
    import real_parallel as parallel
    from runtime import read
    import test_real as real_fixtures


class ParallelSlurmTests(unittest.TestCase):
    def setUp(self):
        self.fixture=slurm_fixtures.SlurmRoutingTests('test_internal_mode_requires_a_real_job_step')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.script=Path(__file__).with_name('run_real_parallel.sh').resolve()
        (self.fixture.envroot/'setup_node.txt').write_text('cn29\n')
        self.fixture.env.pop('DREX_REAL_NAME',None)
        self.fixture.env['DREX_TEST_PREFIX']=self.fixture.posix(self.fixture.root/'batch')
        self.fixture.stub('sbatch',
            'n=0; if [[ -f "$DREX_TEST_PREFIX.count" ]]; then n=$(cat "$DREX_TEST_PREFIX.count"); fi\n'
            'n=$((n+1)); printf "%s" "$n" > "$DREX_TEST_PREFIX.count"\n'
            'printf "%s\\n" "$@" > "$DREX_TEST_PREFIX-$n.args"\n'
            'cat > "$DREX_TEST_PREFIX-$n.sh"\necho "$((700+n))"\n')

    def test_submission_has_cpu_plan_one_gpu_per_array_task_and_cpu_afterok_merge(self):
        f=self.fixture
        result=f.run_stage('submit','2')
        self.assertEqual(result.returncode,0,result.stderr)
        plan=(f.root/'batch-1.args').read_text().splitlines()
        worker=(f.root/'batch-2.args').read_text().splitlines()
        merge=(f.root/'batch-3.args').read_text().splitlines()
        self.assertIn('--gres=none',plan)
        self.assertIn('--gres=gpu:1',worker)
        self.assertIn('--array=0-1%2',worker)
        self.assertIn('--mem=96G',worker)
        self.assertIn('--dependency=afterok:701',worker)
        self.assertIn('--dependency=afterok:702',merge)
        self.assertIn('--gres=none',merge)
        self.assertIn('--inside worker 2',(f.root/'batch-2.sh').read_text())
        for args in (plan,worker,merge):
            self.assertFalse(any('--exclusive' in x or '--immediate' in x for x in args))
        self.assertFalse((f.envroot/'venv').exists())

    def test_invalid_count_and_current_single_gpu_output_are_rejected_before_submission(self):
        f=self.fixture
        for n in ('0','9','2;echo bad'):
            self.assertNotEqual(f.run_stage('submit',n).returncode,0)
        f.env['DREX_REAL_NAME']='real_train64_v1'
        self.assertNotEqual(f.run_stage('submit','2').returncode,0)
        self.assertFalse((f.root/'batch.count').exists())

    def test_audit_uses_cpu_and_can_read_the_finished_single_gpu_output(self):
        f=self.fixture
        f.env['DREX_REAL_NAME']='real_train64_v1'
        result=f.run_stage('audit')
        self.assertEqual(result.returncode,0,result.stderr)
        args=f.log.read_text().splitlines()
        self.assertIn('--gres=none',args)
        self.assertNotIn('--gres=gpu:1',args)
        self.assertEqual(args[-3:],['--inside','audit','2'])


@unittest.skipUnless(AVAILABLE,'NumPy/Pillow needed for parallel protocol tests')
class ParallelProtocolTests(unittest.TestCase):
    def setUp(self):
        self.fixture=real_fixtures.RealPilotTests('test_partial_run_cannot_read_outcomes_and_resume_finishes_whole_cohort')
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        self.f=self.fixture
        original=data.prepare
        def prepare(package,root,out): return original(package,root,out,sessions=2,per_session=2)
        self.mock=patch.object(data,'prepare',side_effect=prepare); self.mock.start(); self.addCleanup(self.mock.stop)
        self.plan,*_=parallel.freeze_plan(self.f.root,self.f.package,self.f.out,2)
        self.calls=0

    def replica(self,root,package,out,manifest,records,scenes,vocab):
        def infer(row): self.calls+=1; return self.f.infer(row)
        status=pilot.infer_cohort(package,out,manifest,records,scenes,vocab,infer)
        data.freeze(out/'inference_environment.json',dict(gpu='fixture',numpy='fixture'))
        return status

    def workers(self):
        for i in range(2):
            parallel.worker(self.f.root,self.f.package,self.f.out,i,replica=self.replica)

    def test_assignment_is_disjoint_balanced_and_exhaustive(self):
        parts=parallel.assignment(self.f.records,2)
        self.assertEqual([len(x) for x in parts],[2,2])
        self.assertFalse(set(parts[0])&set(parts[1]))
        self.assertEqual(set(sum(parts,[])),{r['token'] for r in self.f.records})
        with self.assertRaises(ValueError): parallel.assignment(self.f.records,0)

    def test_missing_worker_blocks_all_label_reads(self):
        parallel.worker(self.f.root,self.f.package,self.f.out,0,replica=self.replica)
        with patch.object(pilot,'load_labels',side_effect=AssertionError('Premature label access')) as labels:
            with self.assertRaises(FileNotFoundError): parallel.merge(self.f.root,self.f.package,self.f.out)
            labels.assert_not_called()

    def test_parallel_decisions_match_serial_and_finished_workers_do_not_repeat(self):
        self.workers(); self.assertEqual(self.calls,16)
        self.workers(); self.assertEqual(self.calls,16)
        report=parallel.merge(self.f.root,self.f.package,self.f.out)
        serial=self.f.root/'serial'
        manifest,records,scenes,vocab=data.prepare(self.f.package,self.f.root,serial)
        pilot.infer_cohort(self.f.package,serial,manifest,records,scenes,vocab,self.f.infer)
        expected=pilot.evaluate(self.f.package,serial,manifest,records,vocab)
        self.assertEqual(read(self.f.out/'decisions.json'),read(serial/'decisions.json'))
        self.assertEqual(report['methods'],expected['methods'])

    def test_changed_worker_result_blocks_labels(self):
        self.workers()
        path=next((self.f.out/'shards/00/results').glob('*.json'))
        path.write_text(path.read_text()+' ')
        with patch.object(pilot,'load_labels',side_effect=AssertionError('Premature label access')) as labels:
            with self.assertRaisesRegex(RuntimeError,'Committed result changed'):
                parallel.merge(self.f.root,self.f.package,self.f.out)
            labels.assert_not_called()


if __name__=='__main__': unittest.main()
