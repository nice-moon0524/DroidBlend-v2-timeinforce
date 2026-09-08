"""Build a tiny mixed QA dataset from the three DroidBlend corpora.

The script keeps the existing file names expected by the experiment runner and
selects the shortest rows from each source split so the resulting run stays
small and fast.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable, Iterator


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


DATASETS = {
    "hotpotqa": "hotpotqa_test.jsonl",
    "2wikimqa": "2wikimqa_test.jsonl",
    "multifieldqa_en": "multifieldqa_en_test.jsonl",
}


def _score(record: dict) -> tuple[int, str]:
    context = str(record.get("context", ""))
    question = str(record.get("question", ""))
    return len(context) + len(question), str(record.get("id", ""))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a small 3xN DroidBlend evaluation set")
    parser.add_argument("--source-dir", default="data/processed", help="Directory with the full JSONL corpora")
    parser.add_argument("--output-dir", default="data/mini_mixed", help="Where to write the reduced dataset")
    parser.add_argument("--samples-per-dataset", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples_per_dataset <= 0:
        raise SystemExit("--samples-per-dataset must be positive")

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Output directory already exists: {output_dir}. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "samples_per_dataset": args.samples_per_dataset,
        "datasets": {},
    }

    for dataset_name, filename in DATASETS.items():
        path = source_dir / filename
        records = sorted(read_jsonl(path), key=_score)
        if len(records) < args.samples_per_dataset:
            raise SystemExit(
                f"{path} only contains {len(records)} rows; need {args.samples_per_dataset}."
            )
        selected = records[: args.samples_per_dataset]
        write_jsonl(output_dir / filename, selected)
        manifest["datasets"][dataset_name] = {
            "source_file": filename,
            "selected_ids": [str(record.get("id", "")) for record in selected],
            "selected_lengths": [len(str(record.get("context", ""))) + len(str(record.get("question", ""))) for record in selected],
        }

    (output_dir / "selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
