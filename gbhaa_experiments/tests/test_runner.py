import unittest

from gbhaa_experiments.runner import parse_attack_output, redact_command


class RunnerTest(unittest.TestCase):
    def test_parses_attack_summary_and_accumulates_llm_calls(self):
        output = """
          Clean accuracy:         0.8000
          Edge attack accuracy:   0.7000 (drop: 0.1000)
          Feature attack accuracy: 0.6500 (drop: 0.1500)
          Combined accuracy:      0.5000 (drop: 0.3000)
          Misclassification:      0.5000
        Cluster Attack Done. Success: 2/2, LLM Calls: 3, Cache Hits: 1
        Cluster Attack Done. Success: 1/1, LLM Calls: 2, Cache Hits: 0
        Attack completed: Structure perturbations=7, feature attack=enabled
        """
        metrics = parse_attack_output(output)
        self.assertEqual(metrics["clean_accuracy"], 0.8)
        self.assertEqual(metrics["combined_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["combined_drop"], 0.3)
        self.assertEqual(metrics["llm_calls"], 5)
        self.assertEqual(metrics["cache_hits"], 1)
        self.assertEqual(metrics["structure_perturbations"], 7)

    def test_redacts_api_key_value(self):
        command = ["python", "meta.py", "--openai_api_key", "secret", "--seed", "15"]
        self.assertEqual(
            redact_command(command),
            ["python", "meta.py", "--openai_api_key", "<redacted>", "--seed", "15"],
        )

    def test_explicit_external_call_count_overrides_template_count(self):
        output = """
        Cluster Attack Done. Success: 1/1, LLM Calls: 1, Cache Hits: 0
        External LLM Calls: 0 (deterministic local templates)
        """
        metrics = parse_attack_output(output)
        self.assertEqual(metrics["llm_calls"], 0)


if __name__ == "__main__":
    unittest.main()
