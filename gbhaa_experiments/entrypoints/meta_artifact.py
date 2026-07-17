"""Run an attack variant and persist original/attacked graph artifacts."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch

import heir as heir_module


REPO_ROOT = Path(__file__).resolve().parents[2]
_ARTIFACT_DIR: Optional[Path] = None
_BASE_NAME = "serial"


def _as_csr(value) -> sp.csr_matrix:
    if sp.issparse(value):
        return value.tocsr()
    if torch.is_tensor(value):
        tensor = value.detach().cpu()
        if tensor.is_sparse:
            tensor = tensor.to_dense()
        value = tensor.numpy()
    return sp.csr_matrix(np.asarray(value))


def _artifact_subclass(base_class):
    class ArtifactAttack(base_class):
        def __init__(self, *args, **kwargs):
            patched_class = heir_module.Heirattack
            heir_module.Heirattack = base_class
            try:
                super().__init__(*args, **kwargs)
            finally:
                heir_module.Heirattack = patched_class

        def meta_attack_multi_step(self, features, ori_adj, labels, *args, **kwargs):
            result = super().meta_attack_multi_step(
                features, ori_adj, labels, *args, **kwargs
            )
            assert _ARTIFACT_DIR is not None
            _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
            sp.save_npz(_ARTIFACT_DIR / "original_adj.npz", _as_csr(ori_adj))
            sp.save_npz(
                _ARTIFACT_DIR / "modified_adj.npz", _as_csr(self.modified_adj)
            )
            sp.save_npz(
                _ARTIFACT_DIR / "original_features.npz", _as_csr(features)
            )
            sp.save_npz(
                _ARTIFACT_DIR / "modified_features.npz",
                _as_csr(self.modified_features),
            )
            label_values = (
                labels.detach().cpu().numpy() if torch.is_tensor(labels) else labels
            )
            np.save(_ARTIFACT_DIR / "labels.npy", np.asarray(label_values))
            metadata = {
                "base": _BASE_NAME,
                "dataset": getattr(self.args, "dataset", None),
                "seed": getattr(self.args, "seed", None),
                "ptb_rate": getattr(self.args, "ptb_rate", None),
                "coarsen_method": getattr(self.args, "coarsen_method", None),
            }
            (_ARTIFACT_DIR / "metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"Experiment artifacts saved to {_ARTIFACT_DIR}")
            return result

    ArtifactAttack.__name__ = f"Artifact{base_class.__name__}"
    return ArtifactAttack


def _consume_args() -> None:
    global _ARTIFACT_DIR, _BASE_NAME
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--experiment_base",
        choices=["serial", "independent", "feature", "edge"],
        required=True,
    )
    parser.add_argument("--experiment_artifact_dir", type=Path, required=True)
    parsed, remaining = parser.parse_known_args(sys.argv[1:])
    _BASE_NAME = parsed.experiment_base
    _ARTIFACT_DIR = parsed.experiment_artifact_dir.resolve()
    sys.argv = [sys.argv[0], *remaining]


def main() -> None:
    _consume_args()
    if _BASE_NAME == "independent":
        import meta_independent

        meta_independent._consume_independent_args(sys.argv)
        base_class = meta_independent.IndependentHeirattack
        script = REPO_ROOT / "meta.py"
    elif _BASE_NAME == "feature":
        import meta_independent

        from gbhaa_experiments.entrypoints.meta_feature_only import FeatureOnlyAttack

        meta_independent._consume_independent_args(sys.argv)
        base_class = FeatureOnlyAttack
        script = REPO_ROOT / "meta.py"
    elif _BASE_NAME == "edge":
        base_class = heir_module.Heirattack
        script = REPO_ROOT / "meta_edge_only.py"
    else:
        base_class = heir_module.Heirattack
        script = REPO_ROOT / "meta.py"

    heir_module.Heirattack = _artifact_subclass(base_class)
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
