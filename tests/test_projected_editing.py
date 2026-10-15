import unittest
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from gbhaa_experiments.post_tuning_protocol import make_jobs,get_profile
from gbhaa_experiments.projected_editing import word_plan, adapt_candidate, project, choose_candidate

class EditingTests(unittest.TestCase):
    def test_factorial_is_twelve_matched_trials(self):
        jobs=make_jobs(['citeseer'],[15],.1,60,preset='editing',matched_controls=True)
        self.assertEqual(len(jobs),12)
        self.assertEqual({j['profile_id'] for j in jobs},{'edit_control','edit_gain','edit_words','edit_both'})
        for j in jobs:self.assertEqual(j['config']['editor_candidates'],3)
    def test_gradient_word_directions_and_enforcement(self):
        v=CountVectorizer(vocabulary=['apple','berry','cherry','dates'],binary=True)
        x=np.array([1,1,0,0],dtype=np.float32);g=np.array([-2,1,3,-4])
        plan=word_plan(x,g,[3,2],1,True,80,4)
        self.assertEqual(plan['add'],[2]);self.assertEqual(plan['drop'],[0]);self.assertEqual(plan['keep'],[1])
        actual,info=adapt_candidate('apple dates',plan,x,v,['apple','berry','cherry','dates'],70,.35)
        self.assertEqual(actual[0],0);self.assertGreater(actual[2],0)
        self.assertEqual(info['actual_dropped'],1)
        self.assertEqual(actual[3],0)  # Incidental negative-gradient novel words cannot leak back in.
    def test_projection_and_choice_use_real_vectors(self):
        x=np.array([1.,0,0],np.float32)
        a,_=project(x,np.array([0,1,1],np.float32),70,.9,[])
        self.assertGreaterEqual(float(a@x/np.linalg.norm(a)),.9-1e-6)
        # Raw candidate has larger linear gain; projection reverses that order.
        b,_=project(x,np.array([1,.4,0],np.float32),70,.9,[])
        self.assertGreater(2.,.4);self.assertLess(float(a[1]+a[2]),float(b[1])*2)
        self.assertEqual(choose_candidate([.3,.7,.7],True,15,0,2),1)
        self.assertEqual(choose_candidate([1,9,3],False,15,0,2),choose_candidate([9,1,3],False,15,0,2))
    def test_required_additions_survive_cap_and_partial_tail(self):
        x=np.array([1,0,0,0,1],np.float32)
        y,info=project(x,np.array([1,1,1,1,0],np.float32),1,.35,[3])
        self.assertEqual(y[3],1);self.assertEqual(y[1],0);self.assertEqual(info['added'],1)
    def test_zero_gradient_and_nonfinite(self):
        p=word_plan(np.array([1.,0]),np.zeros(2),[1],2,True,80,2)
        self.assertEqual(p['add'],[]);self.assertEqual(p['drop'],[])
        with self.assertRaises(ValueError):choose_candidate([0,float('nan')],True,1,0,0)

    def test_zero_vector_projection_does_not_break_cosine_constraint(self):
        x=np.array([1.,0],np.float32)
        y,info=project(x,np.zeros(2,np.float32),70,.35,[])
        np.testing.assert_array_equal(y,x)
        self.assertEqual(info['projection_fraction'],0.)
