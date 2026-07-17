from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RunSmallDatasetScriptTest(unittest.TestCase):
    def setUp(self):
        self.source = (ROOT / "run_small_datasets.sh").read_text(encoding="utf-8")

    def test_selects_supported_datasets_and_all_target_entrypoints(self):
        self.assertIn("cora_ml|acm|polblogs", self.source)
        self.assertIn('TARGET_MODEL="${2:-${TARGET_MODEL:-gcn}}"', self.source)
        self.assertIn('ENTRYPOINT="meta.py"', self.source)
        self.assertIn('ENTRYPOINT="meta_gin.py"', self.source)
        self.assertIn('ENTRYPOINT="meta_gsage.py"', self.source)

    def test_has_fast_smoke_mode_and_unbuffered_python(self):
        self.assertIn('RUN_MODE="${RUN_MODE:-full}"', self.source)
        self.assertIn('if [[ "$RUN_MODE" == "smoke" ]]', self.source)
        self.assertIn('exec "$PYTHON_BIN" -u "$ENTRYPOINT"', self.source)

    def test_prepares_data_before_selecting_text_attack_mode(self):
        self.assertIn("prepare_small_datasets.py", self.source)
        self.assertLess(
            self.source.index("prepare_small_datasets.py"),
            self.source.index("EFFECTIVE_USE_TEXT_ATTACK"),
        )
        self.assertIn('USE_TEXT_ATTACK="${USE_TEXT_ATTACK:-auto}"', self.source)
        self.assertIn("ALLOW_FALLBACK_VOCAB", self.source)

    def test_preflights_text_attack_dependencies_and_installed_ollama_model(self):
        self.assertIn("importlib.util.find_spec", self.source)
        self.assertIn('"$OLLAMA_ROOT/api/tags" "$OLLAMA_MODEL"', self.source)
        self.assertIn("ollama pull", self.source)
        self.assertIn("127.0.0.1:11434", self.source)
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        self.assertIn("httpx", requirements.splitlines())

    def test_dataset_launchers_embed_independent_experiment_configs(self):
        for dataset in ("cora_ml", "acm", "polblogs"):
            launcher = (ROOT / f"run_{dataset}.sh").read_text(encoding="utf-8")
            self.assertTrue(launcher.startswith("#!/bin/bash\n"))
            self.assertIn("ARGS=(", launcher)
            self.assertIn(f"--dataset {dataset}", launcher)
            self.assertIn('# python meta.py "${ARGS[@]}"', launcher)
            self.assertIn('# python meta_gin.py "${ARGS[@]}"', launcher)
            self.assertIn('python meta_gsage.py "${ARGS[@]}"', launcher)
            self.assertNotIn("run_small_datasets.sh", launcher)

    def test_fallback_vocabulary_is_explicit_only_for_unaligned_datasets(self):
        cora_ml = (ROOT / "run_cora_ml.sh").read_text(encoding="utf-8")
        self.assertNotIn("--allow_fallback_vocabulary", cora_ml)
        for dataset in ("acm", "polblogs"):
            launcher = (ROOT / f"run_{dataset}.sh").read_text(encoding="utf-8")
            self.assertIn("--allow_fallback_vocabulary", launcher)

    def test_rejects_unknown_dataset_without_starting_an_experiment(self):
        result = subprocess.run(
            ["bash", str(ROOT / "run_small_datasets.sh"), "unknown"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("Unsupported small dataset", result.stderr)


if __name__ == "__main__":
    unittest.main()
