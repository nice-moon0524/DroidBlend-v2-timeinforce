import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class PlotDroidSpeakMetricsTests(unittest.TestCase):
    def _write(self, path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_renders_all_offline_figures(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "plot_droidspeak_metrics.py"
        spec = importlib.util.spec_from_file_location("plot_droidspeak_metrics", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "outputs"
            self._write(root / "baseline" / "native_pair_calibration.json", {
                "sender_A_native": {"quality": {"qa_f1": 0.2}, "mean_prefill_ms": 30},
                "receiver_B_native": {"quality": {"qa_f1": 0.3}, "mean_prefill_ms": 40},
                "droid_direction_gate_passed": True,
            })
            self._write(root / "sensitivity" / "single_layer_kv_swap.json", {
                "native": {"qa_f1": 0.3},
                "layers": [
                    {"layer": 0, "scores": {"qa_f1": 0.29}, "qa_f1_drop_absolute": 0.01},
                    {"layer": 1, "scores": {"qa_f1": 0.2}, "qa_f1_drop_absolute": 0.1},
                ],
            })
            first = {"group": {"start": 0, "end": 1, "identifier": "layers_000_000"}, "quality": 0.29, "native_quality": 0.3, "hybrid_prefill_ms": 10, "native_prefill_ms": 40}
            second = {"group": {"start": 1, "end": 3, "identifier": "layers_001_002"}, "quality": 0.3, "native_quality": 0.3, "hybrid_prefill_ms": 20, "native_prefill_ms": 40}
            self._write(root / "profiling" / "points.json", {"native_quality": 0.3, "points": [first, second]})
            self._write(root / "profiling" / "selected.json", {"selected": first})
            self._write(root / "evaluation" / "layers_000_000.json", {
                "native": {"qa_f1": 0.3}, "raw_sender_kv": {"qa_f1": 0.1}, "droidspeak_hybrid": {"qa_f1": 0.29},
                "local_prefill_performance": {"native_receiver_prefill_ms": 40, "droidspeak_hybrid_prefill_ms": 10},
            })
            with patch("sys.argv", [str(script), "--output-dir", str(root)]):
                self.assertEqual(module.main(), 0)
            figures = root / "paper_figures"
            self.assertTrue((figures / "01_sender_receiver_baseline.png").is_file())
            self.assertTrue((figures / "02_layer_sensitivity.png").is_file())
            self.assertTrue((figures / "03_quality_prefill_pareto.png").is_file())
            self.assertTrue((figures / "04_heldout_quality_and_prefill.png").is_file())


if __name__ == "__main__":
    unittest.main()
