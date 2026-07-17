from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp

from dataset_support import (
    DEEPROBUST_DATASETS,
    SMALL_EXPERIMENT_DATASETS,
    normalize_dataset_name,
)
from split_test import get_deeproubust_dataset


class DatasetSupportTest(unittest.TestCase):
    def test_requested_small_datasets_are_registered(self):
        self.assertEqual(SMALL_EXPERIMENT_DATASETS, ("cora_ml", "acm", "polblogs"))
        self.assertTrue(
            set(SMALL_EXPERIMENT_DATASETS).issubset(DEEPROBUST_DATASETS)
        )

    def test_cli_dataset_aliases_are_normalized(self):
        self.assertEqual(normalize_dataset_name("Cora-ML"), "cora_ml")
        self.assertEqual(normalize_dataset_name("POL-BLOGS"), "polblogs")
        self.assertEqual(normalize_dataset_name("citesser"), "citeseer")

    def test_unknown_hyphenated_dataset_name_is_preserved(self):
        self.assertEqual(normalize_dataset_name("ogbn-arxiv"), "ogbn-arxiv")

    def test_deeprobust_loader_forwards_seed(self):
        fake = SimpleNamespace(
            adj=sp.csr_matrix(np.eye(5, dtype=np.float32)),
            features=sp.csr_matrix(np.eye(5, dtype=np.float32)),
            labels=np.array([0, 1, 0, 1, 0], dtype=np.int64),
        )
        with patch("split_test.Dataset", return_value=fake) as dataset_cls:
            loaded = get_deeproubust_dataset("acm", split="normal", seed=23)

        dataset_cls.assert_called_once_with(root="./Data/", name="acm", seed=23)
        self.assertIs(loaded, fake)
        self.assertEqual(len(loaded.idx_train), 3)
        self.assertEqual(len(loaded.idx_val), 1)
        self.assertEqual(len(loaded.idx_test), 1)

    def test_deeprobust_split_is_reproducible_for_the_same_seed(self):
        def make_dataset():
            return SimpleNamespace(
                adj=sp.csr_matrix(np.eye(10, dtype=np.float32)),
                features=sp.csr_matrix(np.eye(10, dtype=np.float32)),
                labels=np.arange(10, dtype=np.int64) % 2,
            )

        with patch("split_test.Dataset", side_effect=[make_dataset(), make_dataset()]):
            first = get_deeproubust_dataset("acm", split="normal", seed=23)
            second = get_deeproubust_dataset("acm", split="normal", seed=23)

        self.assertEqual(first.idx_train.tolist(), second.idx_train.tolist())
        self.assertEqual(first.idx_val.tolist(), second.idx_val.tolist())
        self.assertEqual(first.idx_test.tolist(), second.idx_test.tolist())


if __name__ == "__main__":
    unittest.main()
