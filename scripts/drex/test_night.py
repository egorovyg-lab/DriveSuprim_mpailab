import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

import night
import synthetic as task


class GeometryTests(unittest.TestCase):
    def test_continuous_sweep_and_closed_contacts(self):
        self.assertTrue(task.intersects((0, 0), (0, 20), (-1, 9, 1, 10)))
        self.assertTrue(task.intersects((-2, 10), (0, 10), (0, 9, 1, 11)))
        self.assertFalse(task.intersects((-2, 8), (-2, 12), (0, 9, 1, 11)))
        world = dict(road=[-3, 3], obstacles=[[-1, 9, 1, 10]])
        self.assertFalse(task.legal(world, dict(points=[[0, 0], [0, 20]])))
        self.assertFalse(task.legal(dict(road=[-3, 3], obstacles=[]), dict(points=[[2.3, 0], [2.3, 20]])))
        # The centre misses, but the footprint touches the obstacle.
        self.assertFalse(task.legal(world, dict(points=[[1.7, 0], [1.7, 20]])))

    def test_all_worlds_have_safe_stop_and_balanced_cases(self):
        from collections import Counter
        worlds = [task.scene(i) for i in range(384)]
        self.assertTrue(all(task.legal(w, w['candidates'][0]) for w in worlds))
        self.assertEqual(set(Counter((len(w['candidates']), w['family']) for w in worlds).values()), {24})
        self.assertEqual(worlds, [task.scene(i) for i in range(384)])

    def test_ties_and_stop_are_graded_by_utility(self):
        w = dict(road=[-3, 3], obstacles=[], candidates=[
            dict(points=[[0, 0], [0, 0]], progress=0),
            dict(points=[[0, 0], [0, 20]], progress=20),
            dict(points=[[1, 0], [1, 20]], progress=20)])
        self.assertTrue(task.grade(w, 1)['optimal'])
        self.assertTrue(task.grade(w, 2)['optimal'])
        self.assertFalse(task.grade(w, 0)['optimal'])

    def test_input_allowlist_and_permutation_mapping(self):
        w = task.scene(3)
        w['expected_answer'] = 'POISON'
        ids = task.order(w, 'canonical')
        self.assertEqual(task.order(w, 'repeat'), ids)
        self.assertEqual(task.order(w, 'reversed'), ids[::-1])
        self.assertEqual(sorted(task.order(w, 'permuted')), sorted(ids))
        row = task.request(w, 'image', ids)
        self.assertEqual(set(row), {'state', 'question'})
        self.assertNotIn('POISON', str(row))
        self.assertNotIn('obstacle_rectangles_xyxy', row['state'])
        text = task.request(w, 'text', ids)
        self.assertEqual(text['state']['candidates'][0]['points'], w['candidates'][ids[0]]['points'])

    def test_constant_stop_is_safe_but_not_optimal_or_progress(self):
        w = task.scene(0)  # Clear road, forward alternative is valid.
        r = dict(scene=0, mode='image', ordering='canonical', k=2, family='clear',
                 chosen_id=0, probabilities=[1.,0.], seconds=.1, grade=task.grade(w,0))
        s = night.summarize([r],1)['by_condition']['image/canonical']
        self.assertEqual(s['unsafe'],0)
        self.assertEqual(s['optimal'],0)
        self.assertEqual(s['stopped'],1)
        self.assertEqual(s['forward_progress_capture'],0.)


@unittest.skipUnless(importlib.util.find_spec('PIL'), 'Pillow needed for diagram/cache tests')
class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=str(Path.cwd()))
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)

    def infer(self, row):
        n = len(row['question']['criteria'])
        return dict(probabilities=[1/n]*n, seconds=.1)

    def test_resume_skips_finished_requests_and_detects_corruption(self):
        worlds = [task.scene(0), task.scene(1)]
        r = night.exercise(worlds, self.out, self.infer, max_new=4)
        self.assertEqual(r['completed_requests'], 4)
        self.assertEqual(r['status'], 'partial')
        calls = []
        def infer(row):
            calls.append(row)
            return self.infer(row)
        r = night.exercise(worlds, self.out, infer)
        self.assertEqual(len(calls), 10)
        self.assertEqual(r['status'], 'complete')
        self.assertEqual(r['order_and_repeat']['repeat']['max_probability_abs_diff'], 0)
        path = next((self.out/'results').glob('*.json'))
        value = night.read(path)
        value['chosen_id'] = 999
        night.write(path, value)
        with self.assertRaisesRegex(RuntimeError, 'Changed result'):
            night.exercise(worlds, self.out, self.infer)

    def test_changed_pixels_or_scene_refuse_resume(self):
        w = task.scene(1)
        night.exercise([w], self.out, self.infer)
        changed = copy.deepcopy(w)
        changed['road'][0] -= .1
        with self.assertRaisesRegex(RuntimeError, 'Changed request'):
            night.exercise([changed], self.out, self.infer)

    def test_failure_is_saved_and_not_silently_retried(self):
        calls = []
        def fail(row):
            calls.append(row)
            raise RuntimeError('OOM for test')
        with self.assertRaisesRegex(RuntimeError, 'OOM'):
            night.exercise([task.scene(0)], self.out, fail)
        with self.assertRaisesRegex(RuntimeError, 'Recorded technical failure'):
            night.exercise([task.scene(0)], self.out, fail)
        self.assertEqual(len(calls), 1)

    def test_repeat_exact_input_and_blank_image_ablation(self):
        w = task.scene(3)
        _, _, a = night.payload(w, 'image', 'canonical')
        _, _, b = night.payload(w, 'image', 'repeat')
        row, _, c = night.payload(w, 'blank', 'canonical')
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(row['images'][0].getextrema(), ((255,255),)*3)
        self.assertLessEqual(row['images'][0].width*row['images'][0].height, night.CONFIG['max_image_pixels'])


if __name__ == '__main__':
    unittest.main()
