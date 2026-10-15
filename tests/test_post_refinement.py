"""Focused profile, paired uncertainty and safe-launcher regression tests."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from gbhaa_experiments import REPO_ROOT
from gbhaa_experiments.post_tuning_protocol import build_trials, get_profile, make_jobs
from gbhaa_experiments.summarize_post_tuning import _compare


class RefinementTests(unittest.TestCase):
    def test_ten_profiles_have_matched_arms_and_fixed_generation_budgets(self):
        jobs = make_jobs(['citeseer'], [15], .1, 60, preset='refine', matched_controls=True)
        self.assertEqual(len(jobs), 30)
        profiles = {job['profile_id'] for job in jobs}
        self.assertEqual(profiles, {'baseline', 'alpha050', 'alpha040', 'alpha060',
            'a50_f01', 'a50_f03', 'a50_f05', 'a50_diffused', 'a50_f03_diffused', 'a50_coverage025'})
        baseline = get_profile('baseline')
        for name in profiles:
            rows = [job for job in jobs if job['profile_id'] == name]
            self.assertEqual({r['arm'] for r in rows}, {'endpoint_post', 'endpoint_pre', 'parallel'})
            self.assertTrue(all(r['config'] == rows[0]['config'] for r in rows))
            self.assertEqual(rows[0]['config']['iterations'], 60)
            for key in baseline:
                if key.startswith('text_') or key == 'generation_max_tokens':
                    self.assertEqual(rows[0]['config'][key], baseline[key])
        self.assertEqual([len(build_trials(preset=p)) for p in ('tiny', 'quick', 'extended')], [6, 12, 22])

    def test_local_refinements_change_only_declared_scoring_parameters(self):
        changes = {'alpha040': {'alpha': .4}, 'alpha060': {'alpha': .6},
            'a50_f01': {'alpha': .5, 'weight_floor': .01},
            'a50_f03': {'alpha': .5, 'weight_floor': .03},
            'a50_f05': {'alpha': .5, 'weight_floor': .05},
            'a50_diffused': {'alpha': .5, 'readout': 'diffused'},
            'a50_f03_diffused': {'alpha': .5, 'weight_floor': .03, 'readout': 'diffused'},
            'a50_coverage025': {'alpha': .5, 'visit_gamma': .25}}
        for name, expected in changes.items():
            base, actual = get_profile('baseline'), get_profile(name)
            changed = {k: v for k, v in actual.items() if k not in ('id', 'family', 'description') and v != base[k]}
            self.assertEqual(changed, expected)

    @staticmethod
    def comparison(differences, nll_gain=0.):
        post = {i: {'evaluation': {'metrics': {'validation_accuracy': .5, 'validation_nll': 1. + nll_gain}}}
                for i in range(len(differences))}
        controls = {i: {'evaluation': {'metrics': {'validation_accuracy': .5 + d/100., 'validation_nll': 1.}}}
                    for i, d in enumerate(differences)}
        return _compare(post, controls, 'alpha050', 'alpha050', 'endpoint_pre',
                        'matched configuration', ('citeseer', .1, 60))

    def test_paired_ci_uses_seed_differences_and_sample_variance(self):
        row = self.comparison([1., 2., 3., 4., 5.])
        self.assertIn('control_minus_post_accuracy_pp_ci95', row)
        low, high = row['control_minus_post_accuracy_pp_ci95']
        self.assertAlmostEqual(low, 1.0367568385, places=8)
        self.assertAlmostEqual(high, 4.9632431615, places=8)
        self.assertEqual((row['accuracy_wins'], row['accuracy_losses'], row['accuracy_ties']), (5, 0, 0))

    def test_one_seed_nll_tie_break_does_not_become_accuracy_win(self):
        row = self.comparison([0.], nll_gain=.3)
        self.assertIn('accuracy_wins', row)
        self.assertIsNone(row['control_minus_post_accuracy_pp_ci95'])
        self.assertEqual((row['accuracy_wins'], row['accuracy_losses'], row['accuracy_ties']), (0, 0, 1))
        self.assertEqual(row['wins'], 1)  # Retain the historical ACC-then-NLL field.
        self.assertEqual(row['paired_differences'][0]['accuracy_outcome'], 'tie')

    def test_mixed_seed_outcomes_and_constant_difference_interval(self):
        row = self.comparison([1., -1., 0.])
        self.assertIn('accuracy_losses', row)
        self.assertEqual((row['accuracy_wins'], row['accuracy_losses'], row['accuracy_ties']), (1, 1, 1))
        self.assertLess(row['control_minus_post_accuracy_pp_ci95'][0], 0)
        self.assertGreater(row['control_minus_post_accuracy_pp_ci95'][1], 0)
        row = self.comparison([2., 2.])
        self.assertAlmostEqual(row['control_minus_post_accuracy_pp_ci95'][0], 2.)
        self.assertAlmostEqual(row['control_minus_post_accuracy_pp_ci95'][1], 2.)

    def launcher(self, *args):
        script = REPO_ROOT/'gbhaa_experiments/launchers/run_post_refinement.sh'
        self.assertTrue(script.is_file(), 'refinement launcher missing')
        with tempfile.TemporaryDirectory(prefix='refine launcher ') as tmp:
            python = Path(tmp)/'python'
            python.write_text(f'#!{sys.executable}\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
            python.chmod(0o755)
            result = subprocess.run(['/bin/bash', str(script), *args], capture_output=True,
                text=True, check=True, env={**os.environ, 'PYTHON_BIN': str(python)})
            return json.loads(result.stdout)

    def test_launcher_screen_is_matched_preview_with_separate_root(self):
        args = self.launcher()
        self.assertEqual(args[:2], ['-m', 'gbhaa_experiments.run_post_tuning'])
        self.assertEqual(args[args.index('--preset')+1], 'refine')
        self.assertIn('--matched-controls', args)
        self.assertNotIn('--execute', args)
        self.assertEqual(args[args.index('--seeds')+1], '15')
        self.assertTrue(args[args.index('--output-root')+1].endswith('post_refinement_results'))
        self.assertIn('--execute', self.launcher('--execute'))

    def test_launcher_confirmation_freezes_recommendation_and_changes_seeds(self):
        args = self.launcher('confirm')
        self.assertIn('--recommendation', args)
        i = args.index('--seeds')
        self.assertEqual(args[i+1:i+5], ['16', '17', '18', '19'])
        self.assertEqual(args[args.index('--max-actions')+1], '120')
        self.assertTrue(args[args.index('--output-root')+1].endswith('post_refinement_confirm_results'))
        self.assertNotIn('--execute', args)

    def test_launcher_summary_modes_use_corresponding_roots(self):
        for mode, root in [('summary-screen', 'post_refinement_results'),
                           ('summary-confirm', 'post_refinement_confirm_results')]:
            args = self.launcher(mode)
            self.assertEqual(args[:2], ['-m', 'gbhaa_experiments.summarize_post_tuning'])
            self.assertTrue(args[args.index('--root')+1].endswith(root))
            self.assertIn('--output-dir', args)


if __name__ == '__main__':
    unittest.main()
