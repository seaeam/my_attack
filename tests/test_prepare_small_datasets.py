from pathlib import Path
from types import SimpleNamespace
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp

from prepare_small_datasets import prepare_dataset


class PrepareSmallDatasetsTest(unittest.TestCase):
    def test_cora_ml_download_prepares_feature_aligned_vocabulary(self):
        fake = SimpleNamespace(features=sp.csr_matrix(np.eye(3, dtype=np.float32)))
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp) / "Data"
            cache_root = Path(tmp) / "bow_cache"
            data_root.mkdir()
            np.savez(
                data_root / "cora_ml.npz",
                attr_names=np.array(["alpha", "beta", "gamma"]),
            )

            with patch("prepare_small_datasets.Dataset", return_value=fake) as loader:
                prepare_dataset("cora-ml", data_root, cache_root, seed=23)

            loader.assert_called_once_with(
                root=str(data_root) + "/", name="cora_ml", seed=23
            )
            with (cache_root / "cora_ml.pkl").open("rb") as stream:
                vectorizer = pickle.load(stream)

        self.assertEqual(
            vectorizer.get_feature_names_out().tolist(),
            ["alpha", "beta", "gamma"],
        )
        encoded = vectorizer.transform(["gamma alpha"]).toarray()[0]
        self.assertEqual(encoded.tolist(), [1, 0, 1])

    def test_acm_download_does_not_invent_vocabulary(self):
        fake = SimpleNamespace(
            features=sp.csr_matrix(np.zeros((4, 2), dtype=np.float32))
        )
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp) / "Data"
            cache_root = Path(tmp) / "bow_cache"
            with patch("prepare_small_datasets.Dataset", return_value=fake):
                prepare_dataset("acm", data_root, cache_root)

            self.assertFalse((cache_root / "acm.pkl").exists())


if __name__ == "__main__":
    unittest.main()
