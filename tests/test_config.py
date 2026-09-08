import tempfile
import unittest
from pathlib import Path

from droidblend.config import load_config
from droidblend.errors import ConfigurationError
from droidblend.data import PromptExample, require_disjoint


class ConfigTests(unittest.TestCase):
    def test_placeholder_paths_are_rejected_only_when_execution_starts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs").mkdir()
            config = root / "configs" / "x.yaml"
            config.write_text(
                """experiment: {name: x, seed: 1, output_dir: ./outputs/x}\nmodels: {sender_path: __SET_A__, receiver_path: __SET_B__, tokenizer_path: __SET_T__}\ndata: {calibration_jsonl: a, evaluation_jsonl: b}\ncache: {root: c}\nprofiling: {}\nevaluation: {}\n""",
                encoding="utf-8",
            )
            loaded = load_config(config)
            self.assertEqual(loaded.output_dir, root / "outputs" / "x")
            with self.assertRaises(ConfigurationError):
                loaded.assert_model_paths_configured()

    def test_held_out_split_rejects_shared_ids_and_prompts(self):
        calibration = [PromptExample("a", "prompt one", ("x",), {})]
        with self.assertRaises(ConfigurationError):
            require_disjoint(calibration, [PromptExample("a", "prompt two", ("y",), {})], "cal", "eval")
        with self.assertRaises(ConfigurationError):
            require_disjoint(calibration, [PromptExample("b", "prompt one", ("y",), {})], "cal", "eval")


if __name__ == "__main__":
    unittest.main()
