"""Render a short human-readable comparison from per-example result JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_jsonl_prompts(path: Path) -> dict[str, str]:
    prompts: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        prompts[str(row["id"])] = str(row["prompt"])
    return prompts


def _load_predictions(path: Path, bucket_key: str, prediction_key: str) -> dict[str, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if bucket_key not in raw:
        raise KeyError(f"{path} does not contain {bucket_key!r}")
    bucket = raw[bucket_key]
    if not isinstance(bucket, dict) or "predictions" not in bucket:
        raise KeyError(f"{path} does not contain {bucket_key}.predictions")
    return {str(item["id"]): item for item in bucket["predictions"]}


def _load_direct_reuse(path: Path) -> dict[str, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if "per_example" not in raw:
        raise KeyError(f"{path} does not contain per_example")
    return {str(item["id"]): item for item in raw["per_example"]}


def _load_droidblend_result(root: Path, explicit: str | None) -> dict[str, dict[str, Any]] | None:
    if explicit:
        path = _resolve(explicit)
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {str(item["id"]): item for item in raw.get("per_example", [])}

    selected_path = root / "profiling" / "selected.json"
    if selected_path.is_file():
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        group = selected.get("selected", {}).get("group", {})
        identifier = group.get("identifier")
        if identifier:
            path = root / "evaluation" / f"{identifier}.json"
            if path.is_file():
                raw = json.loads(path.read_text(encoding="utf-8"))
                return {str(item["id"]): item for item in raw.get("per_example", [])}

    evaluation_dir = root / "evaluation"
    candidates = list(evaluation_dir.glob("*.json"))
    if len(candidates) == 1:
        raw = json.loads(candidates[0].read_text(encoding="utf-8"))
        return {str(item["id"]): item for item in raw.get("per_example", [])}
    return None


def _split_prompt(prompt: str) -> tuple[str, str]:
    marker = "\nQuestion:"
    if marker not in prompt:
        return prompt, ""
    body, question = prompt.rsplit(marker, 1)
    return body.strip(), question.strip()


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else _repo_root() / candidate


def _render_value(value: str | None) -> str:
    return value if value else "N/A"


def _code_block(text: str) -> list[str]:
    return ["```text", text, "```"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="data/evaluation.jsonl",
        help="Prompt source JSONL",
    )
    parser.add_argument(
        "--baseline",
        default="outputs/llama31_8b_base_to_instruct/baseline/native_pair_evaluation.json",
        help="Baseline JSON with sender_A_native and receiver_B_native",
    )
    parser.add_argument(
        "--direct-reuse",
        default="outputs/llama31_8b_base_to_instruct/diagnostics/direct_reuse_evaluation.json",
        help="Direct reuse diagnostic JSON",
    )
    parser.add_argument(
        "--droidblend",
        default=None,
        help="Optional DroidBlend evaluation JSON. If omitted, try to infer it from profiling/selected.json.",
    )
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--output", default=None, help="Optional markdown output path")
    args = parser.parse_args()

    base = _resolve(args.dataset)
    baseline_path = _resolve(args.baseline)
    direct_path = _resolve(args.direct_reuse)
    prompts = _load_jsonl_prompts(base)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    sender = _load_predictions(baseline_path, "sender_A_native", "prediction")
    receiver = _load_predictions(baseline_path, "receiver_B_native", "prediction")
    direct = _load_direct_reuse(direct_path)
    droidblend = _load_droidblend_result(baseline_path.parents[1], args.droidblend)

    sample_ids = [
        item["id"]
        for item in baseline["receiver_B_native"]["predictions"]
        if str(item["id"]) in direct and (droidblend is None or str(item["id"]) in droidblend)
    ][: args.limit]
    lines: list[str] = []
    lines.append("# DroidBlend sample comparison")
    lines.append("")
    lines.append("Four-way view, using three concrete samples from the hotpotqa evaluation export.")
    lines.append("The fourth slot is filled from the DroidBlend hybrid evaluation output when available.")
    lines.append("")

    for index, sample_id in enumerate(sample_ids, 1):
        prompt = prompts[sample_id]
        prompt_body, question = _split_prompt(prompt)
        answer = ", ".join(map(str, direct[sample_id].get("answers", [])))
        lines.append(f"## Sample {index}")
        lines.append(f"- ID: `{sample_id}`")
        lines.append(f"- Question: {question}")
        lines.append(f"- Gold: {answer}")
        lines.append("")
        lines.append("Prompt excerpt:")
        lines.append("")
        lines.extend(_code_block(_truncate(prompt_body, 900)))
        lines.append("")
        lines.append("Sender A native:")
        lines.extend(_code_block(sender[sample_id]["prediction"]))
        lines.append("")
        lines.append("Receiver B native:")
        lines.extend(_code_block(receiver[sample_id]["prediction"]))
        lines.append("")
        lines.append("Direct A-KV reuse:")
        lines.extend(_code_block(direct[sample_id]["direct_sender_kv_prediction"]))
        lines.append("")
        if droidblend is not None and sample_id in droidblend:
            lines.append("DroidBlend hybrid:")
            lines.extend(_code_block(droidblend[sample_id]["hybrid_prediction"]))
        else:
            lines.append("DroidBlend hybrid: N/A (run profile/select-pareto/evaluate first)")
        lines.append("")

    text = "\n".join(lines)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
