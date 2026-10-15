import os,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'gbhaa_experiments/launchers/run_post_editing.sh'
class EditingLauncherTests(unittest.TestCase):
    def test_preview_matrices_create_no_result(self):
        for seeds,count in (([],12),(['--seeds','15','16'],24)):
            with tempfile.TemporaryDirectory() as directory:
                target=Path(directory)/'untouched'
                result=subprocess.run(['bash',str(SCRIPT),*seeds,'--output-root',str(target)],
                    cwd=ROOT,env={**os.environ,'PYTHON_BIN':sys.executable},text=True,capture_output=True,check=True)
                self.assertIn(f'PLAN ONLY: {count} short trials, 60 actions each',result.stdout)
                self.assertEqual(result.stdout.count('-m gbhaa_experiments.entrypoints.meta_post_tuning'),count)
                self.assertFalse(target.exists())
                for p in ('edit_control','edit_gain','edit_words','edit_both'):
                    for arm in ('endpoint_post','endpoint_pre','parallel'):
                        self.assertIn(f'--tune-profile {p} --tune-arm {arm}',result.stdout)
    def test_summary_destination(self):
        result=subprocess.run(['bash',str(SCRIPT),'summary','--help'],cwd=ROOT,
            env={**os.environ,'PYTHON_BIN':'/bin/echo'},text=True,capture_output=True,check=True)
        self.assertIn('post_editing_results',result.stdout)
        self.assertIn('followup_reports/post_editing',result.stdout)
