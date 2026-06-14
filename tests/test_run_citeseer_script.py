from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RunCiteseerScriptTest(unittest.TestCase):
    def setUp(self):
        self.source = (ROOT / "run_citeseer_gb.sh").read_text(encoding="utf-8")

    def test_selects_all_three_target_entrypoints(self):
        self.assertIn('TARGET_MODEL="${1:-${TARGET_MODEL:-gcn}}"', self.source)
        self.assertIn('ENTRYPOINT="meta.py"', self.source)
        self.assertIn('ENTRYPOINT="meta_gin.py"', self.source)
        self.assertIn('ENTRYPOINT="meta_gsage.py"', self.source)

    def test_has_fast_smoke_mode_and_unbuffered_python(self):
        self.assertIn('RUN_MODE="${RUN_MODE:-full}"', self.source)
        self.assertIn('if [[ "$RUN_MODE" == "smoke" ]]', self.source)
        self.assertIn('"$PYTHON_BIN" -u "$ENTRYPOINT"', self.source)

    def test_preflights_text_attack_dependencies_and_ollama(self):
        self.assertIn("importlib.util.find_spec", self.source)
        self.assertIn("localhost:11434/api/tags", self.source)
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        self.assertIn("httpx", requirements.splitlines())


if __name__ == "__main__":
    unittest.main()
