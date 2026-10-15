"""Bounded, offline fixtures for the historical result audit."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from gbhaa_experiments.specs import EXPERIMENTS


class FollowupAuditTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(
            importlib.util.find_spec("gbhaa_experiments.followup_audit"),
            "The offline audit module must exist",
        )
        from gbhaa_experiments import followup_audit

        self.audit = followup_audit
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "results"
        self.root.mkdir()

    def payload(self, variant="ppr_ppr", seed=15, rate=0.05,
                suite="ae_ppr", dataset="citeseer"):
        spec = next(v for v in EXPERIMENTS[suite] if v.name == variant)
        command = ["python", "-m", spec.target] if spec.target_kind == "module" else ["python", spec.target]
        command += [
            "--global_seed_strategy", "degree", "--global_ppr_alpha", "0.15",
            "--global_ppr_iters", "30", "--text_ppr_alpha", "0.28",
            "--text_ppr_iters", "60", "--freeze_structure_features",
            "--model", "gcn", "--dataset", dataset, "--seed", str(seed),
            "--ptb_rate", str(rate), *spec.args,
        ]
        metrics = {"clean_accuracy": 0.8, spec.primary_metric: 0.6,
                   spec.primary_metric.replace("accuracy", "drop"): 0.2}
        if spec.primary_metric == "combined_accuracy":
            metrics["misclassification"] = 0.4
        return dict(experiment=suite, dataset=dataset, variant=variant,
                    seed=seed, ptb_rate=rate, command=command, returncode=0,
                    metrics=metrics, primary_metric=spec.primary_metric,
                    primary_value=0.6, wall_time_seconds=10, peak_rss_mib=20,
                    runtime_environment={"machine": "fixture", "cpu_count": 4})

    def write(self, payload, prefix=""):
        path = self.root / payload["experiment"] / prefix / payload["dataset"] / (
            f"ptb_{payload['ptb_rate']:g}/seed_{payload['seed']}/{payload['variant']}/result.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def block(self, seed=15):
        return [self.write(self.payload(v.name, seed)) for v in EXPERIMENTS["ae_ppr"]]

    def codes(self, report):
        return {issue["code"] for record in report["records"] for issue in record["issues"]}

    def test_empty_tree_has_explicit_expected_grid_including_cora_node_level(self):
        report = self.audit.audit_results(self.root)
        self.assertEqual(len(report["expected_cells"]), 210)
        self.assertEqual(report["missing_cells"], report["expected_cells"])
        self.assertIn(dict(experiment="efficiency", dataset="cora", ptb_rate=0.1,
                           seed=19, variant="node_level"), report["missing_cells"])
        self.assertEqual(report["usable_complete_seed_blocks"], [])

    def test_exact_missing_cell_and_runs_jsonl_is_never_counted(self):
        paths = self.block()
        paths[-1].unlink()
        (self.root / "ae_ppr" / "runs.jsonl").write_text(json.dumps(self.payload("ae_ppr_ae_ppr")))
        report = self.audit.audit_results(self.root)
        missing = [c for c in report["missing_cells"] if c["experiment"] == "ae_ppr"
                   and c["seed"] == 15 and c["ptb_rate"] == 0.05]
        self.assertEqual([c["variant"] for c in missing], ["ae_ppr_ae_ppr"])
        self.assertEqual(report["summary"]["records"], 3)
        self.assertFalse(report["usable_complete_seed_blocks"])

    def test_complete_factorial_allows_only_transition_differences(self):
        self.block()
        report = self.audit.audit_results(self.root)
        self.assertEqual(len(report["usable_complete_seed_blocks"]), 1)
        self.assertFalse(report["factorial_settings_mismatches"])
        config = report["records"][0]["config"]
        self.assertEqual(config["global_seed_strategy"], "degree")
        self.assertEqual(config["global_ppr_alpha"], 0.15)
        self.assertEqual(config["global_ppr_iters"], 30)
        self.assertEqual(config["text_ppr_alpha"], 0.28)
        self.assertEqual(config["text_ppr_iters"], 60)
        self.assertEqual(config["source"], "recorded_command")

    def test_config_split_within_arms_blocks_factorial_use(self):
        self.block()
        changed = self.payload("ae_ppr_ppr")
        changed["command"] += ["--global_seed_strategy=uniform", "--epochs=250"]
        self.write(changed)
        report = self.audit.audit_results(self.root)
        fields = report["factorial_settings_mismatches"][0]["differences"]
        self.assertIn("global_seed_strategy", fields)
        self.assertIn("epochs", fields)
        self.assertFalse(report["usable_complete_seed_blocks"])

    def test_config_profiles_do_not_pool_different_settings_across_seeds(self):
        self.block(15)
        for variant in EXPERIMENTS["ae_ppr"]:
            payload = self.payload(variant.name, seed=16)
            payload["command"] += ["--global_ppr_iters=120"]
            self.write(payload)
        report = self.audit.audit_results(self.root)
        blocks = report["usable_complete_seed_blocks"]
        self.assertEqual(len(blocks), 2)
        self.assertNotEqual(blocks[0]["configuration_group"], blocks[1]["configuration_group"])

    def test_metric_complements_drops_and_primary_value_are_checked(self):
        payload = self.payload()
        payload["metrics"].update(misclassification=0.1, combined_drop=0.05)
        payload["primary_value"] = 0.2
        self.write(payload)
        report = self.audit.audit_results(self.root)
        self.assertTrue({"misclassification_disagreement", "drop_disagreement",
                         "primary_value_disagreement"}.issubset(self.codes(report)))
        self.assertEqual(len(report["unusable_cells"]), 1)

    def test_rounding_tolerance_accepts_boundary_but_rejects_larger_error(self):
        payload = self.payload()
        payload["metrics"].update(misclassification=0.4003, combined_drop=0.2003)
        self.write(payload)
        report = self.audit.audit_results(self.root)
        self.assertNotIn("misclassification_disagreement", self.codes(report))
        self.assertNotIn("drop_disagreement", self.codes(report))
        payload["metrics"]["combined_drop"] = 0.20031
        self.write(payload)
        self.assertIn("drop_disagreement", self.codes(self.audit.audit_results(self.root)))

    def test_corrupt_returncode_missing_metrics_and_nonfinite_are_distinct(self):
        paths = self.block()
        paths[0].write_text('{"openai_api_key": "do-not-echo", BROKEN')
        second = self.payload("ae_ppr_ppr")
        second["returncode"] = 1
        self.write(second)
        third = self.payload("ppr_ae_ppr")
        third["metrics"].pop("combined_accuracy")
        self.write(third)
        fourth = self.payload("ae_ppr_ae_ppr")
        fourth["metrics"]["combined_accuracy"] = float("nan")
        fourth["wall_time_seconds"] = float("inf")
        self.write(fourth)
        report = self.audit.audit_results(self.root)
        self.assertTrue({"corrupt_json", "returncode", "missing_metrics", "nonfinite"}
                        .issubset(self.codes(report)))
        self.assertEqual(len(report["missing_cells"]), 206)
        self.assertEqual(len(report["unusable_cells"]), 4)
        self.assertNotIn("do-not-echo", json.dumps(report, allow_nan=False))

    def test_secret_values_are_redacted_in_both_reports_including_equals_syntax(self):
        payload = self.payload()
        secrets = ["secret-one", "secret-two", "secret-three", "secret-four", "secret-five"]
        payload["command"] += ["--openai_api_key", secrets[0], "--token=" + secrets[1],
                               "--api-key=" + secrets[2], "--client_secret", secrets[3]]
        payload["runtime_environment"]["password"] = secrets[4]
        payload["runtime_environment"]["note"] = "copied secret-one and --token=secret-two"
        self.write(payload)
        report = self.audit.audit_results(self.root)
        output = Path(self.temp.name) / "reports"
        self.audit.write_reports(report, output)
        for name in ("audit.json", "audit.md"):
            text = (output / name).read_text()
            for secret in secrets:
                self.assertNotIn(secret, text)
            self.assertIn("<redacted>", text)

    def test_string_commands_and_missing_historical_defaults_are_not_invented(self):
        payload = self.payload()
        payload["command"] = "python -m gbhaa_experiments.entrypoints.meta_ae_ppr_ablation --token='hidden token'"
        self.write(payload)
        report = self.audit.audit_results(self.root)
        self.assertIsNone(report["records"][0]["config"]["global_ppr_alpha"])
        self.assertIn("transition_mapping_unresolved", self.codes(report))
        self.assertNotIn("hidden token", json.dumps(report))
        self.assertTrue(report["unresolved_mapping"])

    def test_conflicting_metadata_and_command_are_unresolved_and_not_usable(self):
        payload = self.payload()
        payload["command"] += ["--seed=19"]
        path = self.write(payload)
        payload["variant"] = "ae_ppr_ppr"
        path.write_text(json.dumps(payload))
        report = self.audit.audit_results(self.root)
        self.assertIn("cell_mapping_conflict", self.codes(report))
        self.assertIn("command_metadata_mismatch", self.codes(report))
        self.assertFalse(report["usable_complete_seed_blocks"])
        self.assertTrue(any(m.get("path") for m in report["unresolved_mapping"]))

    def test_duplicate_result_files_are_not_arbitrarily_selected(self):
        self.block()
        self.write(self.payload(), prefix="copy")
        report = self.audit.audit_results(self.root)
        self.assertIn("duplicate_cell", self.codes(report))
        self.assertFalse(report["usable_complete_seed_blocks"])

    def test_current_order_and_global_operator_are_only_cautions(self):
        payload = self.payload()
        payload["command"] += ["--global_seed_strategy=uniform"]
        self.write(payload)
        report = self.audit.audit_results(self.root)
        record = report["records"][0]
        self.assertEqual(record["provenance"]["historical_order"], "unknown")
        self.assertIn("uniform_global_invariance", self.codes(report))
        cautions = " ".join(report["cautions"])
        self.assertIn("current default is fixed to edge-first", cautions)
        self.assertIn("historical execution-order provenance remains unknown", cautions)
        self.assertIn("stationary PageRank", cautions)
        self.assertIn("manuscript", cautions)

    def test_cli_writes_only_reports_and_preserves_legacy_bytes(self):
        self.write(self.payload())
        before = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        output = Path(self.temp.name) / "reports"
        result = subprocess.run(
            [sys.executable, "-B", "-m", "gbhaa_experiments.followup_audit",
             "--result-root", str(self.root), "--output-dir", str(output)],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual({p.name for p in output.iterdir()}, {"audit.json", "audit.md"})
        self.assertEqual(before, {p.relative_to(self.root): p.read_bytes()
                                 for p in self.root.rglob("*") if p.is_file()})

    def test_output_directory_cannot_be_inside_legacy_results(self):
        report = self.audit.audit_results(self.root)
        with self.assertRaises(ValueError):
            self.audit.write_reports(report, self.root / "reports")

    def test_nested_seed_directory_is_not_silently_assigned_to_outer_seed(self):
        payload = self.payload()
        nested = self.root / "ae_ppr/citeseer/ptb_0.05/seed_17/seed_15/ppr_ppr/result.json"
        nested.parent.mkdir(parents=True)
        nested.write_text(json.dumps(payload))
        report = self.audit.audit_results(self.root)
        self.assertIn("path_mapping_unresolved", self.codes(report))
        self.assertTrue(any(c["seed"] == 17 and c["variant"] == "ppr_ppr"
                            for c in report["missing_cells"]))
        self.assertFalse(report["records"][0]["usable_record"])

    def test_invalid_primary_schema_and_missing_returncode_do_not_crash(self):
        payload = self.payload()
        payload["primary_metric"] = ["combined_accuracy"]
        payload.pop("returncode")
        payload["variant"] = "unknown"
        self.write(payload)
        report = self.audit.audit_results(self.root)
        self.assertIn("returncode", self.codes(report))
        self.assertIn("primary_metric_mismatch", self.codes(report))
        json.dumps(report, allow_nan=False)

    def test_nonfinite_metric_strings_are_flagged(self):
        payload = self.payload()
        payload["metrics"]["combined_accuracy"] = "Infinity"
        self.write(payload)
        self.assertIn("nonfinite", self.codes(self.audit.audit_results(self.root)))

    def test_report_symlink_cannot_overwrite_a_legacy_file(self):
        legacy = self.write(self.payload())
        original = legacy.read_bytes()
        output = Path(self.temp.name) / "reports"
        output.mkdir()
        (output / "audit.json").symlink_to(legacy)
        with self.assertRaises(ValueError):
            self.audit.write_reports(self.audit.audit_results(self.root), output)
        self.assertEqual(legacy.read_bytes(), original)

    def test_runtime_mismatch_is_not_hidden_by_equal_command_settings(self):
        self.block()
        payload = self.payload("ae_ppr_ppr")
        payload["runtime_environment"]["cpu_count"] = 32
        self.write(payload)
        report = self.audit.audit_results(self.root)
        self.assertIn("runtime_environment", report["factorial_settings_mismatches"][0]["differences"])
        self.assertFalse(report["usable_complete_seed_blocks"])


if __name__ == "__main__":
    unittest.main()
