import tempfile
import unittest
import contextlib
import io
import json
import os
from pathlib import Path
from unittest.mock import patch

from gbhaa_experiments import run_followup as runner
from gbhaa_experiments.followup_config import SELECTOR_PROTOCOL
from gbhaa_experiments.runner import CommandResult


class CompletionIdentityTests(unittest.TestCase):
    """Offline subprocess-boundary fixtures; no model training or API requests."""

    def write_artifacts(self, directory, job):
        artifacts = directory / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)
        attack = {key: job[key] for key in (
            "seed", "dataset", "selector", "local_graph", "global_transition", "global_source")}
        attack.update(protocol=SELECTOR_PROTOCOL, budget_complete=True,
                      generation_model="llama3.2:1b-instruct-fp16")
        evaluation = {"protocol": SELECTOR_PROTOCOL,
                      "metrics": {"combined_accuracy": .5}}
        (artifacts / "attack.json").write_text(json.dumps(attack))
        (artifacts / "evaluation.json").write_text(json.dumps(evaluation))
        # This layer only checks existence; summary tests validate NPZ contents.
        (artifacts / "combined_predictions.npz").touch()
        return attack, evaluation

    def test_selector_completion_rejects_embedded_artifact_disagreement(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            job = runner.make_jobs("selectors", [16], [.05])[0]
            attack, evaluation = self.write_artifacts(directory, job)
            (directory / "result.json").write_text(json.dumps({
                **job, "seed": 17, "status": "complete",
                "attack": {**attack, "seed": 17}, "evaluation": evaluation}))
            with self.assertRaisesRegex(ValueError, "differs from saved artifacts"):
                runner.load_completed_artifacts(directory, SELECTOR_PROTOCOL)

    def test_selector_completion_rejects_wrong_seed_before_marking_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            job = runner.make_jobs("selectors", [16], [.05])[0]
            self.write_artifacts(directory, job)
            (directory / "result.json").write_text(json.dumps({
                **job, "seed": 17, "status": "running"}))
            with self.assertRaisesRegex(ValueError, r"attack.seed.*17.*16"):
                runner.load_completed_artifacts(directory, SELECTOR_PROTOCOL)

    def run_offline_matrix(self, root, child):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "offline-placeholder",
                                    "OLLAMA_MODEL": "llama3.2:1b-instruct-fp16"}), \
                patch.object(runner, "source_manifest", return_value={"offline.py": "a" * 64}), \
                patch.object(runner, "input_manifest", return_value={"offline.npz": "b" * 64}), \
                patch.object(runner, "python_environment", return_value={"offline": True}), \
                patch.object(runner, "run_command", side_effect=child), \
                contextlib.redirect_stdout(io.StringIO()):
            return runner.main(["--suite", "selectors", "--seeds", "17", "18", "19",
                "--budgets", ".05", "--variants", "ae_pr", "--execute",
                "--api-base-url", "http://localhost:11434/v1", "--output-root", str(root)])

    def test_runner_preserves_seed_and_output_for_each_child(self):
        seen = []
        def child(command, **kwargs):
            seed = int(command[command.index("--seed") + 1])
            directory = Path(command[command.index("--followup-artifacts") + 1]).parent
            self.assertEqual(directory.parent.name, f"seed_{seed}")
            job = runner.make_jobs("selectors", [seed], [.05])[0]
            self.write_artifacts(directory, job)
            seen.append(seed)
            return CommandResult(command, 0, .01, None, "")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "results"
            self.assertEqual(self.run_offline_matrix(root, child), 0)
            self.assertEqual(seen, [17, 18, 19])
            for seed in seen:
                path = root / f"selectors/citeseer/ptb_0.05/seed_{seed}/ae_pr/result.json"
                row = json.loads(path.read_text())
                self.assertEqual(row["status"], "complete")
                self.assertEqual(row["attack"]["seed"], seed)
                self.assertEqual(row["command"][row["command"].index("--seed") + 1], str(seed))

    def test_runner_records_command_and_interrupt_type_before_child_returns(self):
        def child(command, **kwargs):
            directory = Path(command[command.index("--followup-artifacts") + 1]).parent
            row = json.loads((directory / "result.json").read_text())
            self.assertEqual(row.get("command"), command)
            raise KeyboardInterrupt()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "results"
            with self.assertRaises(KeyboardInterrupt):
                self.run_offline_matrix(root, child)
            path = root / "selectors/citeseer/ptb_0.05/seed_17/ae_pr/result.json"
            row = json.loads(path.read_text())
            self.assertEqual(row["status"], "failed")
            self.assertEqual(row.get("error_type"), "KeyboardInterrupt")
            self.assertEqual(row["error"], "KeyboardInterrupt")
            self.assertFalse((path.parent / ".running").exists())

    def test_runner_stops_instead_of_completing_a_child_with_wrong_seed(self):
        def child(command, **kwargs):
            directory = Path(command[command.index("--followup-artifacts") + 1]).parent
            old_job = runner.make_jobs("selectors", [16], [.05])[0]
            self.write_artifacts(directory, old_job)
            return CommandResult(command, 0, .01, None, "")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "results"
            with self.assertRaisesRegex(ValueError, r"attack.seed.*17.*16"):
                self.run_offline_matrix(root, child)
            row = json.loads((root / "selectors/citeseer/ptb_0.05/seed_17/ae_pr/result.json").read_text())
            self.assertEqual(row["status"], "failed")
            self.assertIn("attack.seed", row["error"])
            self.assertFalse((root / "selectors/citeseer/ptb_0.05/seed_18").exists())


class FollowupRunnerTests(unittest.TestCase):
    def test_model_provenance_matches_backend_selection(self):
        from gbhaa_experiments.run_followup import generation_model
        self.assertEqual(generation_model("http://localhost:11434/v1", "local-model"), "local-model")
        self.assertEqual(generation_model("https://api.openai.com/v1", "local-model"), "gpt-3.5-turbo")

    def test_factorial_full_matrix_and_degree_configuration(self):
        from gbhaa_experiments.run_followup import make_jobs
        jobs = make_jobs("factorial", [15,16,17,18,19], [.05,.1])
        self.assertEqual(len(jobs), 40)
        self.assertEqual({j["global_source"] for j in jobs}, {"degree"})
        self.assertEqual({j["selector"] for j in jobs}, {"ae_pr", "ppr"})

    def test_selectors_have_same_config_except_selector(self):
        from gbhaa_experiments.run_followup import make_jobs
        jobs = make_jobs("selectors", [15], [.1])
        reduced = [{k:v for k,v in job.items() if k not in ("variant", "selector")} for job in jobs]
        self.assertTrue(all(item == reduced[0] for item in reduced))

    def test_command_has_no_secret_and_requires_explicit_protocol(self):
        from gbhaa_experiments.run_followup import make_jobs, build_command
        job = make_jobs("selectors", [15], [.05])[0]
        cmd = build_command(job, "python", Path("/tmp/no-training"), "http://localhost:11434/v1")
        self.assertNotIn("--openai_api_key", cmd)
        self.assertIn("--followup-artifacts", cmd)
        self.assertEqual(cmd[cmd.index("--global_seed_strategy")+1], "degree")

    def test_resume_rejects_changed_fingerprint_and_partial_output(self):
        from gbhaa_experiments.run_followup import resumable
        self.assertFalse(resumable({"fingerprint": "a", "status":"complete"}, "b"))
        self.assertFalse(resumable({"fingerprint": "a", "status":"failed"}, "a"))
        self.assertTrue(resumable({"fingerprint": "a", "status":"complete"}, "a"))

    def test_dry_run_does_not_create_result_tree(self):
        from gbhaa_experiments.run_followup import main
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/"results"
            code = main(["--suite", "selectors", "--seeds", "15", "--budgets", ".05", "--output-root", str(root)])
            self.assertEqual(code, 0)
            self.assertFalse(root.exists())


if __name__ == "__main__":
    unittest.main()
