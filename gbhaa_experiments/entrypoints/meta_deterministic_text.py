"""Non-LLM, feature-aligned template control for the LLM ablation."""

from __future__ import annotations

import pickle
import runpy
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

import heir as heir_module


REPO_ROOT = Path(__file__).resolve().parents[2]
_BASE_ATTACK = heir_module.Heirattack


class DeterministicTextGenerator:
    """Use an aligned real or explicit fallback vocabulary without an LLM."""

    def __init__(
        self,
        dataset_name: str,
        bow_cache_dir: str = "./bow_cache",
        feature_dim: Optional[int] = None,
        allow_fallback_vocabulary: bool = False,
        **_: object,
    ):
        self.dataset_name = dataset_name
        self.feature_dim = feature_dim
        cache_path = Path(bow_cache_dir) / f"{dataset_name}.pkl"
        self.uses_fallback_vocabulary = False
        if cache_path.exists():
            with cache_path.open("rb") as handle:
                self.vectorizer = pickle.load(handle)
            self.vocab = self.vectorizer.get_feature_names_out()
            if feature_dim is not None and len(self.vocab) != int(feature_dim):
                if not allow_fallback_vocabulary:
                    raise ValueError(
                        f"Vocabulary/feature mismatch: {len(self.vocab)} != {feature_dim}"
                    )
                self._install_fallback(int(feature_dim))
        elif allow_fallback_vocabulary and feature_dim:
            self._install_fallback(int(feature_dim))
        else:
            raise FileNotFoundError(
                f"Aligned vocabulary not found at {cache_path}; deterministic text "
                "control refuses placeholder tokens unless --allow_fallback_vocabulary is explicit."
            )
        if feature_dim is not None and len(self.vocab) != int(feature_dim):
            raise ValueError(
                f"Vocabulary/feature mismatch: {len(self.vocab)} != {feature_dim}"
            )
        self.vocab_size = len(self.vocab)
        self.num_retries = 0
        print(
            "Deterministic non-LLM text control initialized with "
            f"{self.vocab_size} feature-aligned tokens "
            f"(fallback={self.uses_fallback_vocabulary})"
        )

    def _install_fallback(self, feature_dim: int) -> None:
        fallback = [f"feature_{index}" for index in range(feature_dim)]
        self.vectorizer = CountVectorizer(
            vocabulary=fallback,
            token_pattern=r"(?u)\b\w+\b",
        )
        self.vocab = self.vectorizer.get_feature_names_out()
        self.uses_fallback_vocabulary = True

    def extract_words_from_bow_vector(self, bow_vector):
        if hasattr(bow_vector, "detach"):
            bow_vector = bow_vector.detach().cpu().numpy()
        values = np.asarray(bow_vector).reshape(-1)
        if np.all((values == 0) | (values == 1)):
            active = values == 1
        else:
            active = values > 0.1
        used_words = [self.vocab[index] for index in np.flatnonzero(active)]
        not_used_words = [self.vocab[index] for index in np.flatnonzero(~active)]
        return used_words, not_used_words

    def generate_cluster_template(
        self,
        cluster_attributes: List[str],
        discriminative_words: List[str],
        style_constraints: str = "",
        num_candidates: int = 3,
    ) -> List[str]:
        words = list(dict.fromkeys([*cluster_attributes, *discriminative_words]))
        if not words:
            words = list(self.vocab[:1])
        candidates: List[str] = []
        for offset in range(max(1, num_candidates)):
            rotated = words[offset:] + words[:offset]
            candidates.append(" ".join(rotated))
        return candidates


class DeterministicTextAttack(_BASE_ATTACK):
    """Label template generation accurately as local rather than an API call."""

    def __init__(self, *args, **kwargs):
        patched_class = heir_module.Heirattack
        heir_module.Heirattack = _BASE_ATTACK
        try:
            super().__init__(*args, **kwargs)
        finally:
            heir_module.Heirattack = patched_class

    def attack_features_with_text(self, *args, **kwargs):
        result = super().attack_features_with_text(*args, **kwargs)
        print("External LLM Calls: 0 (deterministic local templates)")
        return result


def main() -> None:
    heir_module.TextAttackGenerator = DeterministicTextGenerator
    heir_module.TEXT_ATTACK_AVAILABLE = True
    heir_module.Heirattack = DeterministicTextAttack
    # The canonical constructor validates provider credentials before it
    # instantiates the patched generator. Supply a non-secret local sentinel;
    # DeterministicTextGenerator never creates a client or makes a request.
    sys.argv.extend(
        ["--llm_type", "gpt", "--openai_api_key", "deterministic-local"]
    )
    runpy.run_path(str(REPO_ROOT / "meta.py"), run_name="__main__")


if __name__ == "__main__":
    main()
