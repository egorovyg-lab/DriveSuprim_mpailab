import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from slurm_options import node_options
import test_runtime as fixture


class NodeRoutingTests(unittest.TestCase):
    def fixture(self):
        f=fixture.SlurmRoutingTests('test_internal_mode_requires_a_real_job_step');f.setUp();self.addCleanup(f.doCleanups)
        f.stub('sbatch','printf "%s\\0" "$@" >> "$DREX_TEST_LOG"\ncat >/dev/null\nprintf "12345\\n"\n')
        f.stub('srun','printf "%s\\0" "$@" >> "$DREX_TEST_LOG"\n')
        return f

    def test_all_launchers_auto_and_explicit_without_setup_file(self):
        f=self.fixture()
        cases={
            'run_runtime.sh':('gpu-check',),
            'run_direct256.sh':('submit-pilot','2'),
            'run_consequence_v2.sh':('submit',),
            'run_consequence_probe.sh':('prepare',),
            'run_continuous_audit.sh':('prepare',),
            'run_pool_audit.sh':('prepare',),
            'run_table_pair_audit.sh':('prepare',),
            'run_overlay.sh':('submit-pilot','2'),
            'run_overlay_v2.sh':('submit-pilot','2'),
            'run_overlay_audit.sh':('submit',),
            'run_pair_diagnostic.sh':('submit-plan','2'),
            'run_real_parallel.sh':('submit','2'),
            'run_replacement.sh':('submit-night','2'),
            'run_rollout_choice.sh':('submit','2'),
            'run_table.sh':('submit-night','2'),
            'run_table_group.sh':('submit-bind','2'),
            'run_table_factorial.sh':('submit-night','2'),
            'run_table_case_audit.sh':('submit',),
        }
        for node in ('','cn03'):
            f.env['DREX_NODE']=node
            for name,args in cases.items():
                with self.subTest(node=node,launcher=name):
                    f.script=Path(__file__).with_name(name).resolve();f.log.write_bytes(b'')
                    r=f.run_stage(*args);self.assertEqual(r.returncode,0,r.stderr)
                    argv=f.log.read_bytes().decode().split('\0');pins=[a for a in argv if a.startswith('--nodelist')]
                    self.assertTrue(argv and argv[0])
                    if node:self.assertTrue(pins);self.assertEqual(set(pins),{'--nodelist='+node})
                    else:self.assertEqual(pins,[])
                    self.assertNotIn('',argv[:-1])  # Empty arrays must produce zero arguments.

    def test_old_setup_file_and_sbatch_env_do_not_pin_auto(self):
        f=self.fixture();(f.envroot/'setup_node.txt').write_text('cn29\n')
        f.env['SBATCH_NODELIST']='cn29'
        f.stub('srun','[[ -z ${SBATCH_NODELIST+x} ]] || exit 17\nprintf "%s\\n" "$@" > "$DREX_TEST_LOG"\n')
        for stage in ('gpu-check','model-plan'):
            r=f.run_stage(stage);self.assertEqual(r.returncode,0,r.stderr)
            self.assertNotIn('--nodelist',f.log.read_text())

    def test_empty_auto_invalid_and_explicit_policies_match_shell(self):
        f=self.fixture();helper=Path(__file__).with_name('slurm_options.sh').resolve()
        for node in ('','auto','cn29','cn03.sc.test','cn[03-05]','-wcn29','cn03 cn04','cn03;id'):
            env=dict(f.env,DREX_NODE=node)
            r=subprocess.run([f.bash,'-c','set -euo pipefail; source "$1"; printf "%s\\0" "${DREX_NODE_ARGS[@]}"','policy',f.posix(helper)],
                             env=env,capture_output=True,timeout=15)
            try:expected=node_options(env)
            except ValueError:self.assertNotEqual(r.returncode,0);continue
            self.assertEqual(r.returncode,0,r.stderr)
            actual=[v for v in r.stdout.decode().split('\0') if v]
            self.assertEqual(actual,expected)

    def test_unavailable_shared_python_has_actionable_error(self):
        f=self.fixture();helper=Path(__file__).with_name('slurm_options.sh').resolve()
        r=subprocess.run([f.bash,'-c','set -euo pipefail; source "$1"; drex_require_python "$2"','check',f.posix(helper),f.posix(f.envroot)],
                         env=f.env,text=True,capture_output=True,timeout=15)
        self.assertNotEqual(r.returncode,0);self.assertIn('Compute Python unavailable',r.stderr)

    def test_scheduler_auto_and_pin_without_setup_file(self):
        from test_schedule_replacement import SchedulerTests
        import schedule_replacement as s
        f=SchedulerTests('test_repeated_submission_and_callback_do_not_duplicate_gpu_work');f.setUp();self.addCleanup(f.doCleanups)
        (f.root/'setup_node.txt').unlink()
        with patch.dict(os.environ,DREX_NODE='cn04'):
            chain=s.Chain(f.root,f.name,2,call=f.call);chain.submit(100,1)
            self.assertIn('--nodelist=cn04',f.calls[-1][0])
        chain=s.Chain(f.root,f.name,2,call=f.call);chain.submit(101,1)
        self.assertFalse(any(a.startswith('--nodelist') for a in f.calls[-1][0]))


if __name__=='__main__':unittest.main()
