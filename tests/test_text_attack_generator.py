import importlib
import pickle
import sys
import tempfile
import types
import unittest


def _install_optional_dependency_stubs():
    httpx_module = types.ModuleType("httpx")

    class Client:
        def __init__(self, *args, **kwargs):
            pass

    httpx_module.Client = Client
    sys.modules.setdefault("httpx", httpx_module)

    openai_module = types.ModuleType("openai")

    class OpenAI:
        def __init__(self, *args, **kwargs):
            pass

    openai_module.OpenAI = OpenAI
    sys.modules.setdefault("openai", openai_module)

    transformers_module = types.ModuleType("transformers")

    class LogitsProcessor:
        pass

    transformers_module.AutoTokenizer = object
    transformers_module.AutoModelForCausalLM = object
    transformers_module.LogitsProcessorList = list
    transformers_module.LogitsProcessor = LogitsProcessor
    sys.modules.setdefault("transformers", transformers_module)


class TextAttackGeneratorTest(unittest.TestCase):
    def test_missing_bow_cache_is_rejected_by_default(self):
        _install_optional_dependency_stubs()
        module = importlib.import_module("text_attack_generator")

        with tempfile.TemporaryDirectory() as bow_cache_dir:
            with self.assertRaisesRegex(FileNotFoundError, "disable text attack"):
                module.TextAttackGenerator(
                    dataset_name="polblogs",
                    bow_cache_dir=bow_cache_dir,
                    api_key="dummy",
                    base_url="http://localhost:11434/v1",
                    device="cpu",
                    llm_type="gpt",
                    feature_dim=4,
                )

    def test_explicit_opt_in_uses_feature_aligned_fallback_vocabulary(self):
        _install_optional_dependency_stubs()
        module = importlib.import_module("text_attack_generator")

        with tempfile.TemporaryDirectory() as bow_cache_dir:
            generator = module.TextAttackGenerator(
                dataset_name="polblogs",
                bow_cache_dir=bow_cache_dir,
                api_key="dummy",
                base_url="http://127.0.0.1:11434/v1",
                device="cpu",
                llm_type="gpt",
                feature_dim=4,
                allow_fallback_vocabulary=True,
            )

        self.assertEqual(generator.vocab_size, 4)
        self.assertEqual(
            list(generator.vocab),
            ["feature_0", "feature_1", "feature_2", "feature_3"],
        )
        encoded = generator.vectorizer.transform(["feature_1 feature_3"]).toarray()[0]
        self.assertEqual(encoded.tolist(), [0, 1, 0, 1])
        self.assertEqual(generator.model_name, "llama3.2:1b-instruct-fp16")

    def test_explicit_opt_in_replaces_dimension_mismatched_cache(self):
        _install_optional_dependency_stubs()
        module = importlib.import_module("text_attack_generator")

        with tempfile.TemporaryDirectory() as bow_cache_dir:
            vectorizer = module.CountVectorizer(vocabulary=["old_0", "old_1"])
            with open(f"{bow_cache_dir}/citeseer.pkl", "wb") as handle:
                pickle.dump(vectorizer, handle)
            generator = module.TextAttackGenerator(
                dataset_name="citeseer",
                bow_cache_dir=bow_cache_dir,
                api_key="dummy",
                base_url="http://127.0.0.1:11434/v1",
                device="cpu",
                llm_type="gpt",
                feature_dim=4,
                allow_fallback_vocabulary=True,
            )

        self.assertTrue(generator.uses_fallback_vocabulary)
        self.assertEqual(
            list(generator.vocab),
            ["feature_0", "feature_1", "feature_2", "feature_3"],
        )

    def test_dimension_mismatched_cache_is_rejected_without_opt_in(self):
        _install_optional_dependency_stubs()
        module = importlib.import_module("text_attack_generator")

        with tempfile.TemporaryDirectory() as bow_cache_dir:
            vectorizer = module.CountVectorizer(vocabulary=["old_0", "old_1"])
            with open(f"{bow_cache_dir}/citeseer.pkl", "wb") as handle:
                pickle.dump(vectorizer, handle)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                module.TextAttackGenerator(
                    dataset_name="citeseer",
                    bow_cache_dir=bow_cache_dir,
                    api_key="dummy",
                    base_url="http://127.0.0.1:11434/v1",
                    device="cpu",
                    llm_type="gpt",
                    feature_dim=4,
                )


if __name__ == "__main__":
    unittest.main()
