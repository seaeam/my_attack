import unittest

from gbhaa_experiments.run_matrix import build_command
from gbhaa_experiments.specs import get_variants


class SpecTest(unittest.TestCase):
    def test_all_required_experiment_matrices_exist(self):
        for experiment in (
            "efficiency",
            "gb_ablation",
            "hybrid",
            "llm_ablation",
            "ae_ppr",
        ):
            self.assertTrue(get_variants(experiment))

    def test_ae_ppr_is_full_two_by_two_factorial(self):
        names = {variant.name for variant in get_variants("ae_ppr")}
        self.assertEqual(
            names,
            {"ppr_ppr", "ae_ppr_ppr", "ppr_ae_ppr", "ae_ppr_ae_ppr"},
        )

    def test_feature_only_uses_feature_metric(self):
        variant = get_variants("hybrid", ["feature_only"])[0]
        self.assertEqual(variant.primary_metric, "feature_accuracy")
        self.assertTrue(variant.needs_text_attack)

    def test_build_command_keeps_matched_run_fields_last(self):
        variant = get_variants("gb_ablation", ["gb"])[0]
        command = build_command(
            python="python",
            variant=variant,
            dataset="citeseer",
            seed=19,
            ptb_rate=0.1,
            base_args=["--seed", "1", "--level", "2"],
            llm_type="gpt",
            api_key="unused",
            api_base_url="",
        )
        self.assertEqual(command[-6:], ["--dataset", "citeseer", "--seed", "19", "--ptb_rate", "0.1"])

    def test_edge_only_command_drops_text_only_freeze_flag(self):
        variant = get_variants("hybrid", ["edge_only"])[0]
        command = build_command(
            python="python",
            variant=variant,
            dataset="citeseer",
            seed=15,
            ptb_rate=0.05,
            base_args=[
                "--level",
                "2",
                "--freeze_structure_features",
                "--text_retries",
                "0",
                "--text_similarity_min",
                "0.85",
                "--text_retries=1",
            ],
            llm_type="gpt",
            api_key="unused",
            api_base_url="",
        )
        self.assertNotIn("--freeze_structure_features", command)
        self.assertNotIn("--text_retries", command)
        self.assertNotIn("--text_similarity_min", command)
        self.assertNotIn("0.85", command)
        self.assertNotIn("--text_retries=1", command)

    def test_deterministic_control_does_not_require_external_llm(self):
        variant = get_variants("llm_ablation", ["deterministic_no_llm"])[0]
        self.assertTrue(variant.needs_text_attack)
        self.assertFalse(variant.uses_external_llm)
        command = build_command(
            python="python",
            variant=variant,
            dataset="citeseer",
            seed=15,
            ptb_rate=0.05,
            base_args=[],
            llm_type="gpt",
            api_key="",
            api_base_url="",
        )
        self.assertIn("--use_text_attack", command)
        self.assertNotIn("--openai_api_key", command)


if __name__ == "__main__":
    unittest.main()
