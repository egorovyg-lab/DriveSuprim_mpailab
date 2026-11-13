import ast
from pathlib import Path
import tempfile
import shutil
import subprocess
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import schedule_replacement as scheduler


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.name = 'replacement_gpu2_v1'
        self.out = self.root/'runs'/self.name
        scheduler.write(self.out/'identity.json',dict(replicas=2,config=dict(protocol='replacement_star_v1')))
        scheduler.write(self.out/'plan.json',dict(identity_sha256=scheduler.sha(self.out/'identity.json')))
        (self.root/'setup_node.txt').write_text('cn29\n')
        (self.root/'logs').mkdir()
        self.calls = []; self.batch = 8000; self.resumes = 0
        self.chain = scheduler.Chain(self.root,self.name,2,call=self.call)

    def call(self,args,env=None):
        self.calls.append((args,env))
        if args[0]=='squeue':
            return '288789|drex-replacement-gpu-full|RUNNING\n288790|drex-replacement-report|PENDING\n'
        if args[0]=='sbatch':
            self.batch += 1; return str(self.batch)+';cluster\n'
        if args[0]=='bash':
            self.resumes += 1
            return f'Resume GPU allocation: {9000+self.resumes}; CPU report: {9500+self.resumes}\nOutput: fixture\n'
        raise AssertionError(args)

    def logs(self,parent,second=None):
        (self.root/'logs'/f'replacement-worker-{parent}-0.log').write_text('Traceback (most recent call last):\n'+scheduler.LIMIT+'\n')
        (self.root/'logs'/f'replacement-worker-{parent}-1.log').write_text(second or 'REPLACEMENT 75000/123228\nslurmstepd: CANCELLED\n')

    def test_full_finite_chain_follows_real_gpu_jobs_not_callbacks(self):
        first = self.chain.submit(288789,2,check_queue=True)
        self.assertEqual(first['callback_job'],'8001'); self.assertEqual(self.resumes,0)
        submit = self.calls[-1][0]
        self.assertIn('--dependency=afterany:288789',submit)
        self.assertIn('--gres=none',submit); self.assertIn('--nodelist=cn29',submit)
        self.assertIn('advance',submit[-1]); self.assertNotIn('resume 2',submit[-1])
        self.logs(288789); self.chain.advance(288789)
        child = self.chain.checked(9001)
        self.assertEqual(child['passes'],1); self.assertEqual(child['callback_job'],'8002')
        self.assertIn('--dependency=afterany:9001',self.calls[-1][0])
        self.assertEqual(self.resumes,1)
        self.logs(9001); self.chain.advance(9001)
        self.assertEqual(self.resumes,2)
        self.assertEqual(len([a for a,_ in self.calls if a[0]=='sbatch']),2)

    def test_repeated_submission_and_callback_do_not_duplicate_gpu_work(self):
        first = self.chain.submit(288789,2)
        self.assertEqual(self.chain.submit(288789,2),first)
        self.logs(288789); self.chain.advance(288789); self.chain.advance(288789)
        self.assertEqual(self.resumes,1)
        self.assertEqual(len([a for a,_ in self.calls if a[0]=='sbatch']),2)

    def test_both_worker_budget_messages_are_accepted(self):
        self.chain.submit(288789,1); self.logs(288789,scheduler.LIMIT+'\n')
        self.chain.advance(288789); self.assertEqual(self.resumes,1)

    def test_unexpected_error_on_other_rank_blocks_continuation(self):
        self.chain.submit(288789,2)
        self.logs(288789,'Traceback (most recent call last):\nRuntimeError: CUDA out of memory\n')
        with self.assertRaisesRegex(RuntimeError,'Unexpected exception'): self.chain.advance(288789)
        self.assertEqual(self.resumes,0)

    def test_missing_logs_or_no_budget_stop_do_not_trigger_resume(self):
        self.chain.submit(288789,2)
        with self.assertRaises(FileNotFoundError): self.chain.advance(288789)
        for rank in (0,1): (self.root/'logs'/f'replacement-worker-288789-{rank}.log').write_text('node exited\n')
        with self.assertRaisesRegex(RuntimeError,'No confirmed'): self.chain.advance(288789)
        self.assertEqual(self.resumes,0)

    def test_infrastructure_failure_even_with_budget_marker_blocks_resume(self):
        self.chain.submit(288789,2); self.logs(288789,'slurmstepd: oom-kill event\n')
        with self.assertRaisesRegex(RuntimeError,'Infrastructure'): self.chain.advance(288789)
        self.assertEqual(self.resumes,0)

    def test_early_completion_stops_chain_without_new_gpu(self):
        self.chain.submit(288789,2)
        for rank in (0,1): scheduler.write(self.out/'shards'/f'{rank:02d}'/'commit.json',{})
        self.chain.advance(288789)
        self.assertEqual(self.chain.checked(288789)['state'],'already_complete')
        self.assertEqual(self.resumes,0)

    def test_unknown_resume_reply_never_causes_blind_duplicate_submission(self):
        self.chain.submit(288789,2); self.logs(288789)
        original = self.chain.call
        self.chain.call = lambda args,env=None: 'unknown reply' if args[0]=='bash' else original(args,env)
        with self.assertRaisesRegex(RuntimeError,'Cannot identify'): self.chain.advance(288789)
        with self.assertRaisesRegex(RuntimeError,'Unresolved'): self.chain.advance(288789)

    def test_recovery_after_gpu_submission_only_finishes_scheduling(self):
        self.chain.submit(288789,2)
        record = self.chain.checked(288789)
        record.update(state='gpu_submitted',gpu_job='9001',report_job='9501'); scheduler.write(self.chain.path(288789),record)
        self.chain.advance(288789); self.assertEqual(self.resumes,0)
        self.assertEqual(self.chain.checked(9001)['passes'],1)

    def test_existing_legacy_delayed_job_blocks_second_chain(self):
        self.chain.call = lambda args,env=None: self.call(args,env)+'8002|drex-delayed-resume|PENDING\n'
        with self.assertRaisesRegex(RuntimeError,'Another delayed'): self.chain.submit(288789,2,check_queue=True)
        self.assertFalse(self.chain.path(288789).exists())

    def test_wrong_plan_gpu_count_and_mutated_source_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'plan'): scheduler.Chain(self.root,self.name,4)
        self.chain.submit(288789,2)
        scheduler.write(self.out/'plan.json',dict(changed=True))
        with self.assertRaisesRegex(RuntimeError,'plan changed'): self.chain.advance(288789)

    def test_helper_is_stdlib_and_compatible_with_server_python310(self):
        path = Path(scheduler.__file__)
        tree = ast.parse(path.read_text(),feature_version=(3,10))
        modules = {n.names[0].name for n in ast.walk(tree) if isinstance(n,ast.Import)}
        self.assertFalse({'numpy','torch','PIL'} & modules)

    def test_real_sbatch_call_receives_callback_on_stdin_and_exact_dependency(self):
        chain = scheduler.Chain(self.root,self.name,2)
        result = SimpleNamespace(returncode=0,stdout='8001;cluster\n',stderr='')
        with patch.object(scheduler.subprocess,'run',return_value=result) as run:
            chain.submit(288789,2)
        args = run.call_args.args[0]; script = run.call_args.kwargs['input']
        self.assertIn('--dependency=afterany:288789',args)
        self.assertIn('--gres=none',args); self.assertIn('--mem=512M',args)
        self.assertTrue(script.startswith('#!/usr/bin/env bash\nset -euo pipefail\nexec python3 '))
        self.assertIn(' advance ',script); self.assertIn('--after 288789',script)
        bash = str(Path(r'C:\Program Files\Git\bin\bash.exe')) if Path(r'C:\Program Files\Git\bin\bash.exe').exists() else shutil.which('bash')
        if bash:
            checked = subprocess.run([bash,'-n'],input=script,text=True,capture_output=True)
            self.assertEqual(checked.returncode,0,checked.stderr)


if __name__=='__main__': unittest.main()
