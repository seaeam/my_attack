"""Real short subprocesses only: no models, data or network requests."""
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from gbhaa_experiments.runner import run_command


class StreamingRunnerTests(unittest.TestCase):
    def test_legacy_capture_mode_is_still_quiet(self):
        with tempfile.TemporaryDirectory() as tmp:
            console = io.StringIO()
            with redirect_stdout(console):
                result = run_command([sys.executable,"-c","print('captured')"],
                    cwd=Path(tmp),log_path=Path(tmp)/"stdout.log")
            self.assertEqual(console.getvalue(), "")
            self.assertIn("captured",result.output)

    def test_interrupt_keeps_partial_log_and_stops_own_child_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); marker=root/"terminated"; log=root/"stdout.log"
            code="""import signal,time,sys
from pathlib import Path
def stop(*_):
    Path(sys.argv[1]).write_text('stopped')
    sys.exit(0)
signal.signal(signal.SIGTERM,stop)
print('READY')
time.sleep(10)
"""
            class InterruptedConsole(io.StringIO):
                def write(self,text):
                    super().write(text)
                    if "READY" in self.getvalue():
                        raise KeyboardInterrupt
                    return len(text)
            with redirect_stdout(InterruptedConsole()),self.assertRaises(KeyboardInterrupt):
                run_command([sys.executable,"-c",code,str(marker)],cwd=root,
                            log_path=log,stream_output=True)
            import time
            deadline=time.monotonic()+1
            while not marker.exists() and time.monotonic()<deadline:
                time.sleep(.01)
            self.assertTrue(marker.exists())
            self.assertIn("READY",log.read_text())

    def test_followup_enables_live_output_and_reports_job_completion(self):
        from gbhaa_experiments import run_followup
        from gbhaa_experiments.runner import CommandResult
        with tempfile.TemporaryDirectory() as tmp:
            console=io.StringIO()
            with patch.dict("os.environ", {"OPENAI_API_KEY":"test-key", "OLLAMA_MODEL":"test-model"}), \
                 patch.object(run_followup,"source_manifest",return_value={"source":"sha"}), \
                 patch.object(run_followup,"input_manifest",return_value={}), \
                 patch.object(run_followup,"python_environment",return_value={}), \
                 patch.object(run_followup,"load_completed_artifacts",return_value=(
                     {"generation_model":"test-model"},{"metrics":{"combined_accuracy":.5}})), \
                 patch.object(run_followup,"run_command",return_value=CommandResult([],0,.1,None,"output")) as call, \
                 redirect_stdout(console):
                code=run_followup.main(["--suite","selectors","--seeds","15","--budgets",".05",
                    "--variants","ae_pr","--output-root",str(Path(tmp)/"results"),
                    "--api-base-url","http://localhost:11434/v1","--execute"])
            self.assertEqual(code,0)
            self.assertTrue(call.call_args.kwargs["stream_output"])
            self.assertEqual(call.call_args.kwargs["log_path"].name,"stdout.log")
            self.assertEqual(call.call_args.kwargs["redact_values"],["test-key"])
            self.assertIn("RUN [1/1]",console.getvalue())
            self.assertIn("DONE [1/1]",console.getvalue())

    def test_progress_and_log_are_visible_before_child_finishes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            release = root / "release"
            log = root / "stdout.log"

            class Console(io.StringIO):
                def write(self, text):
                    count = super().write(text)
                    if "step 1\r" in self.getvalue() and log.exists() and "step 1" in log.read_text():
                        release.touch()
                    return count

            code = """import sys,time
from pathlib import Path
sys.stdout.write('step 1\\r')
deadline=time.monotonic()+3
while not Path(sys.argv[1]).exists() and time.monotonic()<deadline:
    time.sleep(.01)
if not Path(sys.argv[1]).exists():
    sys.exit(9)
print('finished')
"""
            console = Console()
            with redirect_stdout(console):
                result = run_command([sys.executable, "-c", code, str(release)],
                    cwd=root, log_path=log, stream_output=True)
            self.assertEqual(result.returncode, 0, result.output)
            self.assertIn("step 1\r", console.getvalue())
            self.assertIn("finished", result.output)

    def test_fragmented_utf8_and_secret_are_streamed_safely(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); log = root / "stdout.log"
            code = """import os,time
data='进度'.encode()
os.write(1,data[:2]);time.sleep(.03)
os.write(1,data[2:]+b' token=my-sec');time.sleep(.03)
os.write(2,b'ret-value done\\n')
"""
            console = io.StringIO()
            with redirect_stdout(console):
                result = run_command([sys.executable, "-c", code], cwd=root,
                    log_path=log, stream_output=True, redact_values=["my-secret-value"])
            for value in (console.getvalue(), result.output, log.read_text()):
                self.assertIn("进度 token=<redacted> done", value)
                self.assertNotIn("my-secret-value", value)

    def test_silent_child_has_heartbeat_and_error_output_is_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            console = io.StringIO()
            with redirect_stdout(console):
                result = run_command([sys.executable, "-c",
                    "import time,sys; time.sleep(.15); print('failed',file=sys.stderr);sys.exit(7)"],
                    cwd=Path(tmp), log_path=Path(tmp)/"stdout.log",
                    stream_output=True, heartbeat_seconds=.03)
            self.assertEqual(result.returncode, 7)
            self.assertIn("RUNNING", console.getvalue())
            self.assertIn("failed", result.output)
            self.assertNotIn("RUNNING", result.output)


if __name__ == "__main__":
    unittest.main()
