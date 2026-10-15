import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'gbhaa_experiments/launchers/run_post_mechanism.sh'

class MechanismLauncherTests(unittest.TestCase):
    def test_preview_default_and_single_seed_create_no_results(self):
        self.assertTrue(SCRIPT.exists())
        for seeds,expected in (([],24),(['--seeds','15'],12)):
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory)/'untouched'
                result=subprocess.run(['bash',str(SCRIPT),*seeds,'--output-root',str(root)],
                    cwd=ROOT,env={**os.environ,'PYTHON_BIN':sys.executable},text=True,capture_output=True,check=True)
                self.assertIn(f'PLAN ONLY: {expected} short trials, 60 actions each',result.stdout)
                self.assertEqual(result.stdout.count('-m gbhaa_experiments.entrypoints.meta_post_tuning'),expected)
                self.assertFalse(root.exists())
                self.assertIn('--tune-profile frontier_editgain --tune-arm parallel',result.stdout)

    def test_summary_dispatches_to_new_report_directory(self):
        self.assertTrue(SCRIPT.exists())
        result=subprocess.run(['bash',str(SCRIPT),'summary','--help'],cwd=ROOT,
            env={**os.environ,'PYTHON_BIN':'/bin/echo'},text=True,capture_output=True,check=True)
        self.assertIn('-m gbhaa_experiments.summarize_post_tuning',result.stdout)
        self.assertIn('post_mechanism_results',result.stdout)
        self.assertIn('followup_reports/post_mechanism',result.stdout)

if __name__=='__main__':unittest.main()
