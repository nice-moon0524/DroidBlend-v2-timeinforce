"""A no-download integration test for the complete DroidSpeak experiment chain."""

import json
import importlib.util
import tempfile
import unittest
from pathlib import Path


try:
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    AVAILABLE = True
except ImportError:  # pragma: no cover
    AVAILABLE = False


@unittest.skipUnless(AVAILABLE, "torch, tokenizers and transformers are required")
class FullSyntheticExperimentTests(unittest.TestCase):
    def _make_tokenizer(self, path: Path) -> None:
        vocabulary = {
            "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
            "what": 4, "is": 5, "one": 6, "two": 7, "answer": 8, "please": 9,
        }
        raw = Tokenizer(WordLevel(vocab=vocabulary, unk_token="<unk>"))
        raw.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=raw, bos_token="<s>", eos_token="</s>",
            pad_token="<pad>", unk_token="<unk>",
        )
        tokenizer.save_pretrained(path)

    def _make_model(self, path: Path, seed: int) -> None:
        torch.manual_seed(seed)
        config = LlamaConfig(
            vocab_size=10, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=64, bos_token_id=1, eos_token_id=2, pad_token_id=0,
        )
        LlamaForCausalLM(config).save_pretrained(path, safe_serialization=True)

    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    def test_capture_sensitivity_profile_select_evaluate_and_report(self):
        from droidblend.cli import main
        from droidblend.config import load_config

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sender_dir, receiver_dir, tokenizer_dir = root / "sender", root / "receiver", root / "tokenizer"
            sender_dir.mkdir()
            receiver_dir.mkdir()
            tokenizer_dir.mkdir()
            self._make_tokenizer(tokenizer_dir)
            self._make_model(sender_dir, seed=7)
            self._make_model(receiver_dir, seed=11)
            script = Path(__file__).resolve().parents[1] / "scripts" / "compare_llama31_models.py"
            spec = importlib.util.spec_from_file_location("compare_llama31_models", script)
            compare_module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(compare_module)
            weight_report = compare_module.safetensor_shape_probe(str(sender_dir), str(receiver_dir))
            self.assertTrue(weight_report["available"])
            self.assertTrue(weight_report["compatible"])
            calibration = root / "calibration.jsonl"
            evaluation = root / "evaluation.jsonl"
            self._write_jsonl(calibration, [{"id": "cal-1", "prompt": "what is one", "answers": ["one"]}])
            self._write_jsonl(evaluation, [{"id": "eval-1", "prompt": "what is two", "answers": ["two"]}])
            config_path = root / "experiment.yaml"
            config_payload = {
                "experiment": {"name": "synthetic", "seed": 9, "output_dir": str(root / "outputs")},
                "models": {
                    "sender_path": str(sender_dir), "receiver_path": str(receiver_dir), "tokenizer_path": str(tokenizer_dir),
                    "torch_dtype": "float32", "attention_implementation": "sdpa", "device_sender": "cpu", "device_receiver": "cpu",
                },
                "data": {
                    "calibration_jsonl": str(calibration), "evaluation_jsonl": str(evaluation),
                    "max_prompt_tokens": 32, "max_new_tokens": 1, "prompt_suffix_tokens": 1,
                },
                "cache": {"root": str(root / "cache"), "storage_dtype": "float32"},
                "profiling": {
                    "recompute_lengths": [1], "start_layers": [0], "measure_latency_repetitions": 1,
                    "warmup_repetitions": 0, "quality_metric": "qa_f1", "quality_tolerance_relative": 0.05,
                },
                "evaluation": {},
            }
            config_path.write_text(json.dumps(config_payload), encoding="utf-8")
            common = ["--config", str(config_path)]
            self.assertEqual(main([*common, "validate-pair"]), 0)
            self.assertEqual(main([*common, "baseline", "--split", "calibration"]), 0)
            self.assertEqual(main([*common, "capture-sender", "--split", "calibration"]), 0)
            self.assertEqual(main([*common, "sensitivity"]), 0)
            self.assertEqual(main([*common, "profile"]), 0)
            self.assertEqual(main([*common, "select-pareto"]), 0)
            self.assertEqual(main([*common, "capture-sender", "--split", "evaluation"]), 0)
            self.assertEqual(main([*common, "evaluate"]), 0)
            self.assertEqual(main([*common, "report"]), 0)
            config = load_config(config_path)
            self.assertTrue((config.output_dir / "run_manifest.json").is_file())
            self.assertTrue((config.output_dir / "baseline" / "native_pair_calibration.json").is_file())
            self.assertTrue((config.output_dir / "evaluation" / "layers_000_000.json").is_file())
            self.assertTrue((config.output_dir / "reports" / "sensitivity.png").is_file())
            self.assertTrue((config.output_dir / "reports" / "pareto.png").is_file())


if __name__ == "__main__":
    unittest.main()
