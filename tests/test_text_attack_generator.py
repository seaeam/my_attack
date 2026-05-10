import importlib
import sys
import tempfile
import types
import unittest


def _install_optional_dependency_stubs():
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
    def test_missing_bow_cache_uses_feature_aligned_fallback_vocabulary(self):
        _install_optional_dependency_stubs()
        module = importlib.import_module("text_attack_generator")

        with tempfile.TemporaryDirectory() as bow_cache_dir:
            generator = module.TextAttackGenerator(
                dataset_name="polblogs",
                bow_cache_dir=bow_cache_dir,
                api_key="dummy",
                base_url="http://localhost:11434/v1",
                device="cpu",
                llm_type="gpt",
                feature_dim=4,
            )

        self.assertEqual(generator.vocab_size, 4)
        self.assertEqual(
            list(generator.vocab),
            ["feature_0", "feature_1", "feature_2", "feature_3"],
        )
        encoded = generator.vectorizer.transform(["feature_1 feature_3"]).toarray()[0]
        self.assertEqual(encoded.tolist(), [0, 1, 0, 1])


if __name__ == "__main__":
    unittest.main()
