import importlib.util
import io
from contextlib import redirect_stdout
from pathlib import Path
import unittest
from unittest.mock import patch

import test_runtime as shell_fixtures

AVAILABLE = bool(importlib.util.find_spec('numpy') and importlib.util.find_spec('PIL'))
if AVAILABLE:
    import numpy as np
    import audit_table_cases as audit
    import table_data as data
    import table_pilot as pilot
    import test_table as fixtures
    from runtime import read,write
    from transfer import sha


@unittest.skipUnless(AVAILABLE,'NumPy/Pillow required; use run_table_case_audit.sh check')
class CaseAuditTests(unittest.TestCase):
    def setUp(self):
        self.capture=redirect_stdout(io.StringIO()); self.capture.__enter__()
        self.addCleanup(self.capture.__exit__,None,None,None)
        self.f=fixtures.TableTests(); self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.source=self.f.out; self.out=self.f.root/'case_audit'
        plan=self.f.fixture()
        for i,r in enumerate(self.f.records):
            token=r['token']; w=self.source/'shards'/f'{i:02d}'
            def infer(row,token=token):
                k=len(row['question']['criteria']); camera='images' in row
                chosen=21 if camera or token=='token1' else 20
                if token=='token1' and camera: chosen=0
                if k==16: chosen=0
                # Introduce a real PDM-changing order flip in one text condition.
                names=list(row['question']['criteria']); aliases=data.aliases(token,self.f.sample)
                reversed_names=[aliases[self.f.sample['candidates'][j]['local']] for j in data.order(token,k,'reversed')]
                if token=='token0' and not camera and k>16 and names==reversed_names: chosen=21
                local=self.f.sample['candidates'][chosen]['local']; alias=aliases[local]
                p=[0.]*k; p[names.index(alias)]=1.
                return dict(probabilities=p,seconds=.1)
            pilot.infer_tasks(self.source,plan,[r],w,infer)
        rows=[]
        for r in self.f.records:
            for i,c in enumerate(self.f.sample['candidates']):
                raw=[1.]*9; raw[0]=.95 if i==20 else 0. if i==21 else .8
                if i==21: raw[2]=0.
                rows.append(dict(token=r['token'],candidate_local=c['local'],candidate_vocab=c['vocab'],raw=raw))
        self.labels=self.f.root/'labels.json'
        write(self.labels,dict(status='complete',request_digest=audit.base.digest(read(self.source/'label_request.json')),
            metrics=['pdm_score',*audit.base.METRICS],rows=rows))
        pilot.report(self.source,self.labels)
        write(self.source/'label_source.json',dict(labels=str(self.labels),sha256=sha(self.labels),
            request_sha256=sha(self.source/'label_request.json')))
        manifest=read(self.f.package/'package_manifest.json')
        self.metadata_patch=patch.object(audit.base,'package_metadata',return_value=(manifest,None,None,self.f.vocab))
        self.metadata_patch.start(); self.addCleanup(self.metadata_patch.stop)

    def test_audit_is_read_only_and_identifies_camera_loss_and_loss_avoidance(self):
        before={p.relative_to(self.source).as_posix():sha(p) for p in self.source.rglob('*') if p.is_file()}
        with patch.object(pilot,'run_gpu',side_effect=AssertionError('Unexpected inference')):
            report=audit.run(self.source,self.out,previews=True)
        after={p.relative_to(self.source).as_posix():sha(p) for p in self.source.rglob('*') if p.is_file()}
        self.assertEqual(before,after)
        self.assertEqual(report['case_counts']['image_worse'],1)
        self.assertEqual(report['case_counts']['image_better'],1)
        self.assertEqual(report['case_counts']['camera_pdm_losses'],1)
        self.assertEqual(report['case_counts']['table_safety_losses'],1)
        self.assertEqual(report['source_queries_verified'],36)
        self.assertEqual(report['source_csv_rows_verified'],48)
        self.assertTrue(list((self.out/'cases').glob('*/preview.png')))
        d=report['image_effect_decomposition']['64']
        self.assertAlmostEqual(d['loss_avoidance']['macro_pp']+d['gain_capture']['macro_pp'],d['net']['macro_pp'])

    def test_corrupt_csv_is_not_treated_as_evidence(self):
        path=self.source/'per_scene.csv'
        path.write_text(path.read_text().replace('token0','tampered',1))
        with self.assertRaisesRegex(RuntimeError,'CSV row'): audit.run(self.source,self.out,previews=False)

    def test_changed_query_blocks_outcome_lookup(self):
        p=next((self.source/'shards/00/results').glob('*.json')); p.write_text(p.read_text()+' ')
        with patch.object(pilot,'label_lookup',side_effect=AssertionError('Early labels')) as lookup:
            with self.assertRaisesRegex(RuntimeError,'Changed/duplicate'): audit.run(self.source,self.out,previews=False)
            lookup.assert_not_called()

    def test_incomplete_or_changed_report_is_rejected(self):
        path=self.source/'report.json'; report=read(path); report['status']='awaiting_top64_labels'; write(path,report)
        with self.assertRaisesRegex(RuntimeError,'Complete'): audit.run(self.source,self.out,previews=False)

    def test_source_or_package_cannot_be_used_as_audit_output(self):
        with self.assertRaisesRegex(ValueError,'separate'): audit.run(self.source,self.source,previews=False)
        with self.assertRaisesRegex(ValueError,'package'): audit.run(self.source,self.f.package/'audit',previews=False)

    def test_order_analysis_reports_actual_quality_changes_not_just_ids(self):
        report=audit.run(self.source,self.out,previews=False)
        result=report['order_effects']['table/k64/reversed']
        self.assertEqual(result['choice_flips'],1)
        self.assertEqual(result['meaningful_pdm_differences'],1)
        self.assertEqual(result['pdm_worse'],1)
        self.assertEqual(result['safety_worse_than_canonical'],1)
        self.assertEqual(result['component_degradations']['drivable_area_compliance'],1)

    def test_conservatism_classification_and_pool_contributions(self):
        self.assertEqual(audit.image_category(.2,.8,.8,True),'avoided_table_loss_by_keeping_ds')
        self.assertEqual(audit.image_category(.95,.8,.8,True),'missed_table_gain_by_keeping_ds')
        self.assertEqual(audit.image_category(.2,.8,.8,False),'recovered_to_ds_pdm_with_other_candidate')
        self.assertEqual(audit.image_category(.95,.8,.8,False),'missed_table_gain_with_ds_equal_pdm_alternative')
        report=audit.run(self.source,self.out,previews=False)
        for value in report['pool_expansion_effects'].values():
            self.assertAlmostEqual(sum(c['macro_pp'] for c in value['categories'].values()),value['total']['macro_pp'])
            self.assertAlmostEqual(sum(c['micro_pp'] for c in value['categories'].values()),value['total']['micro_pp'])
        self.assertEqual(report['pool_expansion_effects']['camera/16_to_64']['categories']['selected_new_tail']['scenes'],1)


class CaseAuditSlurmTests(unittest.TestCase):
    def setUp(self):
        self.f=shell_fixtures.SlurmRoutingTests('test_internal_mode_requires_a_real_job_step')
        self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.f.script=Path(__file__).with_name('run_table_case_audit.sh').resolve()
        (self.f.envroot/'setup_node.txt').write_text('cn29\n')
        self.f.env['DREX_TABLE_NAME']='table_group_bind_gpu2_v1'  # Must not redirect the scientific audit.
        for k in ('DREX_TABLE_AUDIT_NAME','DREX_TABLE_AUDIT_SOURCE'): self.f.env.pop(k,None)
        self.f.env['DREX_TEST_BATCH']=self.f.posix(self.f.root/'batch.sh')
        self.f.stub('sbatch','printf "%s\\n" "$@" > "$DREX_TEST_LOG"\ncat > "$DREX_TEST_BATCH"\necho 999\n')

    def test_one_cpu_job_and_source_independent_of_current_gpu_name(self):
        result=self.f.run_stage('submit')
        self.assertEqual(result.returncode,0,result.stderr)
        args=self.f.log.read_text().splitlines()
        self.assertIn('--gres=none',args); self.assertIn('--mem=16G',args)
        self.assertFalse(any('--array' in s for s in args))
        self.assertIn('table_topk_night_gpu8_v1',result.stdout)
        self.assertNotIn('Source: '+self.f.env['DREX_TABLE_NAME'],result.stdout)
        self.assertIn('--gres=none',(self.f.root/'batch.sh').read_text())
        self.assertFalse((self.f.envroot/'venv').exists())

    def test_output_cannot_overwrite_source(self):
        self.f.env['DREX_TABLE_AUDIT_SOURCE']='table_cases_shared'
        self.f.env['DREX_TABLE_AUDIT_NAME']='table_cases_shared'
        self.assertNotEqual(self.f.run_stage('submit').returncode,0)
        self.assertFalse(self.f.log.exists())


if __name__=='__main__': unittest.main()
