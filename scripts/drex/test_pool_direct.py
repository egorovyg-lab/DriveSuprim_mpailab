import ast
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pool_direct as direct


class DirectTests(unittest.TestCase):
    def test_standalone_controller_pilot_then_full(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); job=root/'rollout_audit_test'; (job/'tools').mkdir(parents=True)
            (job/'audit_request.json').write_text('{}')
            (job/'tools/pool_remote.py').write_text('''import sys,json,hashlib
from pathlib import Path
r=Path(sys.argv[sys.argv.index('--out')+1]);r.mkdir(exist_ok=True)
p=sys.argv[sys.argv.index('--stage')+1]
with (r/'phases.txt').open('a') as f:f.write(p+'\\n')
(r/'report.json').write_text(json.dumps({'status':'pilot_complete' if p=='pilot' else 'complete'}))
q=Path(sys.argv[sys.argv.index('--input')+1])
(r/'identity.json').write_text(json.dumps({'request_sha256':hashlib.sha256(q.read_bytes()).hexdigest()}))
''')
            # Run redirection in a child process; emulate locks only on Windows.
            preface=''
            if sys.platform=='win32':
                preface='import sys,types\nf=types.SimpleNamespace(flock=lambda *a:None,LOCK_EX=1,LOCK_NB=2)\nsys.modules["fcntl"]=f\n'
            source=inspect.getsource(direct.supervise).replace("Path('/workspace/exp_v2/drex_rollout_audit').resolve()",f'Path({str(root)!r}).resolve()')
            value=preface+source+f'\nraise SystemExit(supervise({str(job)!r},{str(root)!r}))'
            p=subprocess.run([sys.executable,'-B','-c',value],capture_output=True,text=True,timeout=15)
            self.assertEqual(p.returncode,0,p.stderr)
            self.assertEqual((job/'artifacts/phases.txt').read_text().splitlines(),['pilot','full'])
            self.assertEqual(json.loads((job/'direct_job.json').read_text())['state'],'complete')
            self.assertIn('DIRECT POOL full',(job/'direct.log').read_text())

    def test_detachment_and_no_shell_or_scientific_changes(self):
        source=inspect.getsource(direct.start_job)
        self.assertIn('start_new_session=True',source);self.assertIn('close_fds=True',source)
        self.assertIn('stdin=subprocess.DEVNULL',source);self.assertNotIn('shell=True',source)
        self.assertNotIn('pool_direct.py',direct.pool_audit.TOOLS)

    def test_self_contained_remote_helpers_python39(self):
        for fn in (direct.supervise,direct.install_worker,direct.start_job,direct.remote_status):
            source=direct.code(fn,'fixture','fixture')
            ast.parse(source,feature_version=(3,9))
        ast.parse(Path(direct.__file__).read_text(),feature_version=(3,9))


if __name__=='__main__':unittest.main()
