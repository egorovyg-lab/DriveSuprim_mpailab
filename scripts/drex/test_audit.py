import importlib.util
import unittest

AVAILABLE=bool(importlib.util.find_spec('numpy'))
if AVAILABLE:
    import numpy as np
    import audit_real as audit
    import real_data as data


@unittest.skipUnless(AVAILABLE,'NumPy needed for paired CPU audit')
class AuditTests(unittest.TestCase):
    def test_session_sensitivity_does_not_drop_the_hero_from_main_estimate(self):
        r=audit.session_summary(dict(hero=[30,30],other=[-1,-1],third=[-2,-2]))
        self.assertEqual(r['mean_pp'],9)
        self.assertEqual(r['leave_one_session_out_pp']['hero'],-1.5)
        self.assertEqual(set(r['by_session']),{'hero','other','third'})

    def test_paired_camera_ablation_uses_the_same_scenes(self):
        rows=[]
        for token,sid,a,b in [('t0','s0',5,1),('t1','s0',3,1),('t2','s1',-1,1),('t3','s1',0,2)]:
            rows.extend([dict(token=token,session_id=sid,method='camera',raw_gain_pp=a),
                         dict(token=token,session_id=sid,method='blank',raw_gain_pp=b)])
        r=audit.paired_contrast(rows,'camera','blank')
        self.assertEqual(r['by_session'],dict(s0=3,s1=-2))
        self.assertEqual(r['mean_pp'],.5)

    def test_order_flips_distinguish_equal_utility_from_geometric_change(self):
        inputs=dict(pool_local=list(range(16)),baseline_local=0,trajectories=np.zeros((16,40,3)))
        inputs['trajectories'][1,:,0]=2
        labels=np.ones((16,9)); labels[0,0]=.8; labels[1,0]=.8
        records=[dict(token='token',session_id='session')]; results=[]
        for kind,target in [('canonical',0),('reversed',1),('permuted',0)]:
            order=data.permutations('token',kind); p=[0.]*16; p[order.index(target)]=1
            results.append(dict(token='token',mode='camera',ordering=kind,pool_positions=order,probabilities=p))
        r=audit.geometric_order_audit(records,dict(token=inputs),results,dict(token=labels))
        self.assertEqual(r['reversed']['choice_flips'],1)
        self.assertEqual(r['reversed']['changed_choices_with_mean_distance_over_1m'],1)
        self.assertEqual(r['reversed']['changed_choices_with_pdm_delta_over_half_pp'],0)
        self.assertEqual(r['permuted']['choice_flips'],0)


if __name__=='__main__': unittest.main()
