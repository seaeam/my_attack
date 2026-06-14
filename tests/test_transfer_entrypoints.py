import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TransferEntrypointTest(unittest.TestCase):
    def _read_tree(self, filename):
        path = ROOT / filename
        self.assertTrue(path.exists(), f"{filename} must exist")
        source = path.read_text(encoding="utf-8")
        return source, ast.parse(source)

    def test_gin_entrypoint_keeps_gcn_surrogate_and_uses_gin_target(self):
        source, tree = self._read_tree("meta_gin.py")

        self.assertIn("surrogate = GCN(", source)
        self.assertIn('target_model_name="gin"', source)
        self.assertTrue(
            any(
                isinstance(node, ast.FunctionDef) and node.name == "test"
                for node in ast.walk(tree)
            )
        )

    def test_gsage_entrypoint_keeps_gcn_surrogate_and_uses_gsage_target(self):
        source, tree = self._read_tree("meta_gsage.py")

        self.assertIn("surrogate = GCN(", source)
        self.assertIn('target_model_name="gsage"', source)
        self.assertTrue(
            any(
                isinstance(node, ast.FunctionDef) and node.name == "test"
                for node in ast.walk(tree)
            )
        )


if __name__ == "__main__":
    unittest.main()
