from pathlib import Path
import tempfile
import unittest

import table_group as group
import test_runtime as fixtures
from runtime import read,write
from transfer import sha


class BindingTests(unittest.TestCase):
    def rows(self,n):
        return [dict(rank=i,uuid=f'GPU-{i}',visible_devices=1,hostname='cn29',job_id='100',step_id='0',launch_id='fixture') for i in range(n)]

    def test_all_supported_counts_have_distinct_physical_gpus(self):
        for n in (1,2,4,8): group.validate_bindings(self.rows(n),n)

    def test_shared_gpu_extra_visibility_missing_and_duplicate_ranks_are_rejected(self):
        rows = self.rows(2); rows[1]['uuid']=rows[0]['uuid']
        with self.assertRaisesRegex(RuntimeError,'share'): group.validate_bindings(rows,2)
        rows = self.rows(2); rows[0]['visible_devices']=2
        with self.assertRaisesRegex(RuntimeError,'exactly one'): group.validate_bindings(rows,2)
        with self.assertRaisesRegex(RuntimeError,'Incomplete'): group.validate_bindings(self.rows(1),2)
        rows = self.rows(2); rows[1]['rank']=0
        with self.assertRaisesRegex(RuntimeError,'duplicate'): group.validate_bindings(rows,2)

    def test_different_nodes_and_stale_steps_cannot_pass_the_barrier(self):
        rows = self.rows(2); rows[1]['hostname']='cn30'
        with self.assertRaisesRegex(RuntimeError,'one node'): group.validate_bindings(rows,2)
        rows = self.rows(2); rows[1]['step_id']='1'
        with self.assertRaisesRegex(RuntimeError,'different Slurm'): group.validate_bindings(rows,2)
        rows = self.rows(2); rows[1]['launch_id']='old-attempt'
        with self.assertRaisesRegex(RuntimeError,'different launch'): group.validate_bindings(rows,2)

    def test_execution_manifest_binds_plan_and_launcher_code(self):
        with tempfile.TemporaryDirectory() as name:
            out = Path(name); write(out/'plan.json',dict(scientific='fixture'))
            value = dict(protocol='single_allocation_replicas_v1',replicas=4,
                scientific_plan_sha256=sha(out/'plan.json'),
                sources={n:sha(Path(group.__file__).with_name(n)) for n in group.SOURCES})
            write(out/'group_execution.json',value)
            self.assertEqual(group.check_execution(out,4),value)
            with self.assertRaisesRegex(RuntimeError,'plan changed'): group.check_execution(out,8)
            write(out/'plan.json',dict(scientific='changed'))
            with self.assertRaisesRegex(RuntimeError,'plan changed'): group.check_execution(out,4)


class GroupSlurmTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.SlurmRoutingTests('test_internal_mode_requires_a_real_job_step')
        self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.f.script = Path(__file__).with_name('run_table_group.sh').resolve()
        (self.f.envroot/'setup_node.txt').write_text('cn29\n')
        for name in ('DREX_TABLE_NAME','DREX_TABLE_COHORT','DREX_TABLE_TIME'): self.f.env.pop(name,None)
        self.f.env['DREX_TEST_PREFIX']=self.f.posix(self.f.root/'batch')
        self.f.stub('sbatch','n=0; [[ ! -f "$DREX_TEST_PREFIX.count" ]] || n=$(cat "$DREX_TEST_PREFIX.count")\n'
            'n=$((n+1)); echo "$n" > "$DREX_TEST_PREFIX.count"\n'
            'printf "%s\\n" "$@" > "$DREX_TEST_PREFIX-$n.args"\n'
            'cat > "$DREX_TEST_PREFIX-$n.sh"\necho "$((900+n))"\n')

    def test_single_gpu_allocation_requests_all_ranks_and_total_ram(self):
        for i,n in enumerate((1,2,4,8),1):
            result = self.f.run_stage('submit-bind',str(n))
            self.assertEqual(result.returncode,0,result.stderr)
            args = (self.f.root/f'batch-{i}.args').read_text().splitlines()
            self.assertIn(f'--gres=gpu:{n}',args); self.assertIn(f'--ntasks={n}',args)
            self.assertIn(f'--mem={96*n}G',args); self.assertIn('--cpus-per-task=4',args)
            self.assertFalse(any(a.startswith('--array') for a in args))
        self.assertNotEqual(self.f.run_stage('submit-bind','3').returncode,0)

    def test_night_has_one_gpu_job_for_both_pilot_and_full(self):
        result = self.f.run_stage('submit-night','8')
        self.assertEqual(result.returncode,0,result.stderr)
        for i in (1,2,4):
            self.assertIn('--gres=none',(self.f.root/f'batch-{i}.args').read_text().splitlines())
        args = (self.f.root/'batch-3.args').read_text().splitlines()
        self.assertIn('--gres=gpu:8',args); self.assertIn('--ntasks=8',args)
        self.assertIn('--dependency=afterok:902',args)
        self.assertIn('--inside gpu-night 8',(self.f.root/'batch-3.sh').read_text())
        self.assertIn('--dependency=afterok:903',(self.f.root/'batch-4.args').read_text().splitlines())
        self.assertFalse((self.f.root/'batch-5.args').exists())

    def test_srun_binds_one_gpu_per_rank_and_runs_pilot_before_full(self):
        self.f.fake_setup()
        self.f.env['SLURM_STEP_ID']='batch'
        self.f.stub('srun','printf "%s\\n" "$@" >> "$DREX_TEST_LOG"\n')
        result = self.f.run_stage('--inside','gpu-night','4')
        self.assertEqual(result.returncode,0,result.stderr)
        log = self.f.log.read_text()
        self.assertEqual(log.count('--gpus-per-task=1'),2)
        self.assertEqual(log.count('--gpu-bind=single:1'),2)
        self.assertEqual(log.count('--kill-on-bad-exit=1'),2)
        self.assertLess(log.index('worker-pilot'),log.rindex('worker'))
        self.assertNotIn('--overlap',log)

    def test_binding_probe_uses_srun_even_when_batch_step_variable_exists(self):
        self.f.fake_setup(); self.f.env['SLURM_STEP_ID']='batch'
        result = self.f.run_stage('--inside','bind','2')
        self.assertEqual(result.returncode,0,result.stderr)
        log = self.f.log.read_text()
        self.assertIn('--gpus-per-task=1',log); self.assertIn('--ntasks=2',log)
        self.assertNotIn('table_pilot.py',self.f.calls.read_text() if self.f.calls.exists() else '')

    def test_old_scientific_outputs_cannot_be_used_as_group_outputs(self):
        self.f.env['DREX_TABLE_NAME']='table_topk_night_gpu8_v1'
        result = self.f.run_stage('submit-night','8')
        self.assertNotEqual(result.returncode,0)
        self.assertFalse((self.f.root/'batch-1.args').exists())


if __name__=='__main__': unittest.main()
