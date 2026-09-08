import json
import tempfile
import unittest
from pathlib import Path

from droidblend.data import load_jsonl, require_disjoint, split_jsonl


class JsonlSplitTests(unittest.TestCase):
    def test_split_is_deterministic_and_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "all.jsonl"
            source.write_text(
                "\n".join(
                    json.dumps({"id": f"id-{index}", "prompt": f"prompt {index}", "answers": [str(index)]})
                    for index in range(5)
                )
                + "\n",
                encoding="utf-8",
            )
            first = split_jsonl(source, root / "calibration.jsonl", root / "evaluation.jsonl", 2, 7)
            second = split_jsonl(source, root / "calibration-2.jsonl", root / "evaluation-2.jsonl", 2, 7)
            calibration = load_jsonl(first["calibration_path"])
            evaluation = load_jsonl(first["evaluation_path"])
            self.assertEqual([item.identifier for item in calibration], [item.identifier for item in load_jsonl(second["calibration_path"])])
            self.assertEqual(len(calibration), 2)
            self.assertEqual(len(evaluation), 3)
            require_disjoint(calibration, evaluation, "calibration", "evaluation")
