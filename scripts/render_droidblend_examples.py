"""Render the per-example DroidBlend output JSON into Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CORE_METHODS = [
    "full_prefill",
    "full_kv_reuse",
    "droidspeak",
    "cacheblend_reference",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else _repo_root() / candidate


def _excerpt(text: str, limit: int = 1200) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)].rstrip() + "..."


def _code_block(text: str) -> list[str]:
    return ["```text", text, "```"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render DroidBlend example predictions to Markdown")
    parser.add_argument("--input", default="results/droidblend_examples.json")
    parser.add_argument("--output", default="results/droidblend_examples.md")
    parser.add_argument("--limit", type=int, default=None, help="Optional global sample cap")
    parser.add_argument("--show-hybrid", action="store_true", help="Also show droidblend_reference")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw = json.loads(_resolve(args.input).read_text(encoding="utf-8"))
    lines: list[str] = []
    lines.append("# DroidBlend examples")
    lines.append("")
    lines.append("Three datasets, two samples each, with four core experiment outputs grouped by sample.")
    lines.append("")

    methods = list(CORE_METHODS)
    if args.show_hybrid:
        methods.append("droidblend_reference")

    count = 0
    for dataset_name, bucket in raw.get("datasets", {}).items():
        records = bucket.get("records", [])
        if not records:
            continue
        lines.append(f"## {dataset_name}")
        lines.append("")
        for index, record in enumerate(records, 1):
            if args.limit is not None and count >= args.limit:
                break
            count += 1
            lines.append(f"### Sample {index}")
            lines.append(f"- ID: `{record.get('id')}`")
            lines.append(f"- Question: {record.get('question', '')}")
            answer = record.get("answer", [])
            if isinstance(answer, list):
                answer_text = ", ".join(map(str, answer))
            else:
                answer_text = str(answer)
            lines.append(f"- Gold: {answer_text}")
            lines.append("")
            lines.append("Prompt excerpt:")
            lines.append("")
            lines.extend(_code_block(_excerpt(record.get("prompt", ""), 1000)))
            lines.append("")
            lines.append("Outputs:")
            lines.append("")
            for method_name in methods:
                method = record.get("methods", {}).get(method_name)
                if not method:
                    continue
                lines.append(f"- {method_name} ({method.get('latency_s', 0):.2f}s)")
                lines.extend(_code_block(_excerpt(method.get("prediction", ""), 1000)))
                selection = method.get("selection")
                if selection:
                    lines.append(
                        f"  selection: {selection.get('selected_count')}/{selection.get('candidate_count')} "
                        f"({selection.get('selected_fraction', 0):.2%})"
                    )
            lines.append("")
        if args.limit is not None and count >= args.limit:
            break

    text = "\n".join(lines).rstrip() + "\n"
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
