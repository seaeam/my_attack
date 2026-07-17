import unittest

from gbhaa_experiments.aggregate_results import aggregate


class AggregateTest(unittest.TestCase):
    def test_aggregates_mean_and_sample_std_by_variant(self):
        records = [
            {
                "experiment": "gb_ablation",
                "dataset": "citeseer",
                "variant": "gb",
                "ptb_rate": 0.1,
                "seed": 1,
                "metrics": {"edge_accuracy": 0.5},
            },
            {
                "experiment": "gb_ablation",
                "dataset": "citeseer",
                "variant": "gb",
                "ptb_rate": 0.1,
                "seed": 2,
                "metrics": {"edge_accuracy": 0.7},
            },
        ]
        rows = aggregate(records)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["metrics.edge_accuracy_mean"], 0.6)
        self.assertGreater(rows[0]["metrics.edge_accuracy_std"], 0.0)

    def test_ignores_failures_and_keeps_latest_success_per_seed(self):
        base = {
            "experiment": "hybrid",
            "dataset": "citeseer",
            "variant": "serial",
            "ptb_rate": 0.05,
        }
        records = [
            {
                **base,
                "seed": 15,
                "returncode": 0,
                "metrics": {"combined_accuracy": 0.7},
            },
            {
                **base,
                "seed": 15,
                "returncode": 0,
                "metrics": {"combined_accuracy": 0.6},
            },
            {
                **base,
                "seed": 16,
                "returncode": 1,
                "metrics": {"combined_accuracy": 0.1},
            },
        ]
        row = aggregate(records)[0]
        self.assertEqual(row["runs"], 1)
        self.assertAlmostEqual(row["metrics.combined_accuracy_mean"], 0.6)


if __name__ == "__main__":
    unittest.main()
