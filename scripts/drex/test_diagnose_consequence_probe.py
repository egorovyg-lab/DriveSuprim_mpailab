import ast
import contextlib
import csv
import gzip
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import consequence_model as model
import diagnose_consequence_probe as diagnostic


def fixture(folder):
    methods = ('random_features/ds', 'random_features/ds_drex_road')
    accuracy = {name: {k: dict(mae_session_macro=.2, average_precision_session_macro=.1)
                       for k in diagnostic.TARGETS} for name in (*methods, 'train_mean')}
    policies = {name+'/t0.1': dict(intervention=0, actual_benefit=0, actual_harm=0) for name in methods}
    report = dict(status='complete', config=model.CONFIG, targets=list(diagnostic.TARGETS),
                  completed_folds=[0, 1, 2, 3], held_out_pairs=4*63, scenes=4, sessions=4,
                  accuracy=accuracy, policies=policies, continuous_mae_road_minus_control={})
    (folder/'report.json').write_text(json.dumps(report), encoding='utf-8')
    with gzip.open(folder/'per_candidate.csv.gz', 'wt', newline='', encoding='utf-8') as f:
        fields = ['token', 'session_id', 'method', 'candidate_local', 'teacher_rank',
                  *('actual_'+k for k in diagnostic.TARGETS), *('predicted_'+k for k in diagnostic.TARGETS)]
        writer=csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for name in methods:
            for i in range(4):
                for j in range(63):
                    prediction = [.3, .1, 0., 0., -.1, .8 if name.endswith('/ds') else .01,
                                  .8 if name.endswith('/ds') else .01]
                    actual = [.5, .1, 0., 0., -.1, 0., float(j == 0)]
                    writer.writerow(dict(token=f'{i:016x}', session_id='session'+str(i), method=name,
                                         candidate_local=j+1, teacher_rank=j+2,
                                         **{'actual_'+k: v for k, v in zip(diagnostic.TARGETS, actual)},
                                         **{'predicted_'+k: v for k, v in zip(diagnostic.TARGETS, prediction)}))
    return report


class DiagnosisTests(unittest.TestCase):
    def test_gate_reproduction_matches_frozen_selector_and_boundaries(self):
        p = np.random.default_rng(88).uniform(-1., 1., size=(400, 7))
        expected = model.choose(p[:, None, :], model.CONFIG['primary_threshold']) >= 0
        np.testing.assert_array_equal(expected, [all(diagnostic.gates(row, model.CONFIG)) for row in p])
        p = [.3, .1, -1., 0., 0., .05, .1]
        self.assertFalse(all(diagnostic.gates(p, model.CONFIG)))  # Strict benefit threshold.
        p[6] += 1e-12; self.assertTrue(all(diagnostic.gates(p, model.CONFIG)))
        for index, value in ((2, -1.-1e-12), (4, 1e-12), (5, .05+1e-12)):
            row=p.copy(); row[index]=value
            self.assertFalse(all(diagnostic.gates(row, model.CONFIG)))

    def test_distinguishes_risk_from_benefit_blocker_and_is_read_only(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d); fixture(folder)
            before={p.name: p.read_bytes() for p in folder.iterdir()}
            result=diagnostic.diagnose(folder)
            self.assertEqual(result['actual_benefit_pairs'], 4)
            ds=result['gate_diagnostics']['random_features/ds']
            road=result['gate_diagnostics']['random_features/ds_drex_road']
            self.assertEqual(ds['standalone']['benefit']['candidates'], 252)
            self.assertEqual(ds['single_blocker']['risk']['candidates'], 252)
            self.assertEqual(road['single_blocker']['benefit']['candidates'], 252)
            self.assertEqual(road['all_gates']['scenes'], 0)
            with contextlib.redirect_stdout(io.StringIO()) as output: diagnostic.render(result)
            self.assertIn('none at any predeclared threshold/family', output.getvalue())
            self.assertEqual(before, {p.name: p.read_bytes() for p in folder.iterdir()})

    def test_manifest_and_policy_mismatch_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d); report=fixture(folder)
            manifest={p.name: dict(size=p.stat().st_size, sha256=diagnostic.sha(p)) for p in folder.iterdir()}
            marker=folder/'artifact_manifest.json'; marker.write_text(json.dumps(manifest))
            self.assertTrue(diagnostic.diagnose(folder)['artifact_manifest_verified'])
            report['policies']['random_features/ds/t0.1']['intervention']=1
            (folder/'report.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(RuntimeError, 'checksum'): diagnostic.diagnose(folder)
            marker.unlink()
            with self.assertRaisesRegex(RuntimeError, 'reproduction'): diagnostic.diagnose(folder)

    def test_duplicate_or_missing_prediction_rows_are_rejected(self):
        for duplicate in (False, True):
            with self.subTest(duplicate=duplicate), tempfile.TemporaryDirectory() as d:
                folder=Path(d); fixture(folder); path=folder/'per_candidate.csv.gz'
                with gzip.open(path, 'rt') as f: lines=f.readlines()
                with gzip.open(path, 'wt') as f: f.writelines(lines + lines[1:2] if duplicate else lines[:-1])
                with self.assertRaises(ValueError): diagnostic.diagnose(folder)

    def test_standalone_python39_and_no_fit_or_network_dependency(self):
        source=Path(diagnostic.__file__).read_text(encoding='utf-8')
        tree=ast.parse(source, feature_version=(3, 9))
        imports=[n.names[0].name for n in tree.body if isinstance(n, ast.Import)]
        self.assertNotIn('numpy', imports); self.assertNotIn('torch', imports)
        self.assertNotIn('subprocess', imports); self.assertNotIn('consequence_run', imports)

    def test_standalone_cli_pointer_binding_and_export_location(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); archive=root/'archive.tar'; archive.write_bytes(b'fixture archive')
            digest=diagnostic.sha(archive); archive.rename(root/('audit-'+digest+'.tar'))
            folder=root/('result_'+digest[:12]); folder.mkdir(); fixture(folder)
            (root/'latest_result.json').write_text(json.dumps(dict(archive_sha256=digest, result=str(folder), remote_exit_code=0)))
            command=[sys.executable, '-I', '-B', str(Path(diagnostic.__file__)), '--run', str(root)]
            result=subprocess.run(command, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr); self.assertIn('COMPLETE: scenes=4', result.stdout)
            result=subprocess.run(command+['--json', str(folder/'new.json')], capture_output=True, text=True, timeout=15)
            self.assertNotEqual(result.returncode, 0); self.assertFalse((folder/'new.json').exists())
            (root/('audit-'+digest+'.tar')).write_bytes(b'tampered')
            result=subprocess.run(command, capture_output=True, text=True, timeout=15)
            self.assertNotEqual(result.returncode, 0); self.assertIn('SHA-verified', result.stderr)


if __name__ == '__main__': unittest.main()
