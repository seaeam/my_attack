import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gbhaa_experiments import run_action_state as runner


class RunnerTests(unittest.TestCase):
    def test_default_plan_has_twenty_shared_trajectories_and_sixty_arms(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'new_results'
            output=io.StringIO()
            with patch.object(runner,'run_command',side_effect=AssertionError('must not run')), \
                    patch.object(runner,'python_environment',side_effect=AssertionError('no environment probes')), \
                    contextlib.redirect_stdout(output):
                self.assertEqual(runner.main(['--output-root',str(root)]),0)
            self.assertFalse(root.exists())
            self.assertEqual(output.getvalue().count('--action-state-phase structure'),20)
            for arm in runner.ARMS:
                self.assertEqual(output.getvalue().count('--action-state-phase '+arm),20)

    def test_arm_commands_replay_one_plan_and_use_fixed_degree_source(self):
        root=Path('/tmp/action_state_test')
        jobs=runner.make_jobs(['cora'],[15],[.05])
        for job in jobs:
            command=runner.build_command(job,'python',root,'http://localhost:11434/v1',root/'cache')
            self.assertEqual(command[command.index('--global_seed_strategy')+1],'degree')
            self.assertEqual(command[command.index('--action-state-trajectory')+1],
                str(root/'cora/ptb_0.05/seed_15/structure/artifacts/trajectory.json'))
            self.assertEqual('--use_text_attack' in command,job['variant']!='structure')

    def test_pubmed_is_opt_in_and_bad_matrix_is_rejected(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(['--datasets','pubmed','--seeds','15','--budgets','.05']),0)
        for options in (['--budgets','nan'],['--seeds','15','15'],['--datasets','cora','cora']):
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                runner.main(options)

    def test_complete_changed_fingerprint_cannot_be_reused_or_retried(self):
        self.assertTrue(runner.can_resume({'status':'complete','fingerprint':'same'},'same'))
        self.assertFalse(runner.can_resume({'status':'complete','fingerprint':'old'},'new'))
        self.assertFalse(runner.can_resume({'status':'failed','fingerprint':'same'},'same'))

    def test_input_changes_are_checked_with_source_changes(self):
        sources = {'code.py': 'a'}
        inputs = {'bow_cache/cora.pkl': 'b'}
        with patch.object(runner, 'source_manifest', return_value=sources), \
                patch.object(runner, 'input_manifest', return_value=inputs):
            runner.check_unchanged_inputs(sources, inputs)
        with patch.object(runner, 'source_manifest', return_value=sources), \
                patch.object(runner, 'input_manifest', return_value={'bow_cache/cora.pkl': 'changed'}):
            with self.assertRaisesRegex(ValueError, 'input'):
                runner.check_unchanged_inputs(sources, inputs)


if __name__=='__main__':unittest.main()
