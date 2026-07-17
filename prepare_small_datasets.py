#!/usr/bin/env python3
"""Download the requested DeepRobust datasets and prepare safe local metadata."""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
from deeprobust.graph.data import Dataset
from sklearn.feature_extraction.text import CountVectorizer

from dataset_support import SMALL_EXPERIMENT_DATASETS, normalize_dataset_name


def _decode_feature_name(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def prepare_cora_ml_vocabulary(data_root, bow_cache_dir, feature_dim):
    """Build a vocabulary whose indices exactly match Cora-ML feature columns."""
    data_path = Path(data_root) / "cora_ml.npz"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Cora-ML archive was not created at {data_path} after download"
        )

    with np.load(data_path, allow_pickle=True) as archive:
        if "attr_names" not in archive.files:
            raise ValueError(f"{data_path} does not contain attr_names")
        feature_names = [_decode_feature_name(v) for v in archive["attr_names"]]

    if len(feature_names) != int(feature_dim):
        raise ValueError(
            "Cora-ML attr_names are not aligned with the feature matrix: "
            f"{len(feature_names)} names for {feature_dim} columns"
        )
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("Cora-ML attr_names contain duplicate feature names")

    bow_cache_dir = Path(bow_cache_dir)
    bow_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = bow_cache_dir / "cora_ml.pkl"

    if cache_path.exists():
        with cache_path.open("rb") as stream:
            cached_vectorizer = pickle.load(stream)
        cached_names = list(cached_vectorizer.get_feature_names_out())
        if cached_names != feature_names:
            raise ValueError(
                f"Existing {cache_path} is not aligned with Data/cora_ml.npz; "
                "remove it and rerun preparation"
            )
        print(f"Vocabulary ready: {cache_path} ({len(cached_names)} aligned tokens)")
        return cache_path

    vectorizer = CountVectorizer(
        vocabulary={name: idx for idx, name in enumerate(feature_names)},
        token_pattern=r"(?u)\b\w+\b",
    )
    with cache_path.open("wb") as stream:
        pickle.dump(vectorizer, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Created vocabulary: {cache_path} ({len(feature_names)} aligned tokens)")
    return cache_path


def prepare_dataset(dataset_name, data_root="./Data", bow_cache_dir="./bow_cache", seed=15):
    """Download/load one dataset and prepare any recoverable semantic vocabulary."""
    dataset_name = normalize_dataset_name(dataset_name)
    if dataset_name not in SMALL_EXPERIMENT_DATASETS:
        raise ValueError(
            f"Unsupported small dataset: {dataset_name}; "
            f"choose from {', '.join(SMALL_EXPERIMENT_DATASETS)}"
        )

    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    dataset = Dataset(
        root=str(data_root) + os.sep,
        name=dataset_name,
        seed=int(seed),
    )
    num_nodes, feature_dim = dataset.features.shape
    print(
        f"Dataset ready: {dataset_name} "
        f"({num_nodes} nodes, {feature_dim} features) in {data_root}"
    )

    if dataset_name == "cora_ml":
        prepare_cora_ml_vocabulary(data_root, bow_cache_dir, feature_dim)
    elif dataset_name == "acm":
        print(
            "Vocabulary unavailable: the DeepRobust ACM package contains numeric "
            "feature columns but no aligned feature names."
        )
    else:
        print(
            "Vocabulary unavailable: DeepRobust PolBlogs has no node attributes; "
            "its feature matrix is an identity matrix."
        )
    return dataset


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        nargs="+",
        type=normalize_dataset_name,
        choices=SMALL_EXPERIMENT_DATASETS,
        default=list(SMALL_EXPERIMENT_DATASETS),
        help="Datasets to download/prepare (default: all three)",
    )
    parser.add_argument("--data-root", default="./Data")
    parser.add_argument("--bow-cache-dir", default="./bow_cache")
    parser.add_argument("--seed", type=int, default=15)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    for dataset_name in args.dataset:
        prepare_dataset(
            dataset_name,
            data_root=args.data_root,
            bow_cache_dir=args.bow_cache_dir,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
