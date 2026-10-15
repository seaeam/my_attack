import unittest
from gbhaa_experiments.post_editing_summary import factorial_effects
class FactorialTests(unittest.TestCase):
    def rows(self):
        return {(p,arm):{seed:{'evaluation':{'metrics':{'validation_accuracy':a+offset}}}
                           for seed,offset in ((15,0),(16,.01))}
                for p,a in (('edit_control',.8),('edit_gain',.7),('edit_words',.6),('edit_both',.4))
                for arm in ('endpoint_post','endpoint_pre','parallel')}
    def test_effect_signs_and_interaction(self):
        rows=factorial_effects(self.rows(),[15,16],('citeseer',.1,60),True)
        self.assertEqual(len(rows),9)
        for r in rows:
            self.assertAlmostEqual(r['mean_pp'],{'gain':15,'words':25,'interaction':10}[r['effect']])
            self.assertEqual(r['n'],2);self.assertEqual(r['status'],'complete');self.assertIsNotNone(r['ci95_pp'])
    def test_missing_cell_suppresses_complete_effects(self):
        d=self.rows();del d['edit_both','endpoint_post'][16]
        rows=factorial_effects(d,[15,16],('citeseer',.1,60),False)
        for r in rows:self.assertIsNone(r['mean_pp']);self.assertIsNone(r['ci95_pp']);self.assertEqual(r['valid_n'],0)
    def test_single_seed_has_no_sd_or_ci(self):
        rows=factorial_effects(self.rows(),[15],('citeseer',.1,60),True)
        for r in rows:self.assertEqual(r['n'],1);self.assertIsNone(r['sd_pp']);self.assertIsNone(r['ci95_pp'])
