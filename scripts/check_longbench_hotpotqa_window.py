from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from transformers import AutoTokenizer

from core.model_loading import DEFAULT_RECEIVER_PATH, make_prompt, resolve_model_path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else _repo_root() / candidate


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether LongBench HotpotQA prompts exceed the Mistral sliding window.")
    parser.add_argument(
        "--data-file",
        default="../DroidSpeak-new/data/processed/longbench_hotpotqa_test_50.jsonl",
        help="Relative path from the DroidBlend-v2-timeinforce repo root, or an absolute path.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=resolve_model_path("DROIDBLEND_RECEIVER", DEFAULT_RECEIVER_PATH),
        help="Receiver tokenizer path used to count prompt tokens.",
    )
    parser.add_argument("--sliding-window", type=int, default=4096, help="Mistral sliding-window limit to compare against.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional sample cap for a quick check.")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--fail-on-over-window", action="store_true")
    args = parser.parse_args()

    data_file = _resolve(args.data_file)
    if not data_file.is_file():
        raise FileNotFoundError(data_file)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        use_fast=True,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = _read_jsonl(data_file)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    token_lengths: list[int] = []
    over_window: list[dict] = []
    for row in rows:
        prompt = make_prompt(row)
        token_ids = tokenizer(prompt, add_special_tokens=False, truncation=False)["input_ids"]
        token_len = len(token_ids)
        token_lengths.append(token_len)
        if token_len > args.sliding_window:
            over_window.append(
                {
                    "id": row.get("id"),
                    "token_len": token_len,
                    "over_by": token_len - args.sliding_window,
                }
            )

    summary = {
        "data_file": str(data_file),
        "samples": len(rows),
        "sliding_window": args.sliding_window,
        "min_prompt_tokens": min(token_lengths) if token_lengths else 0,
        "mean_prompt_tokens": mean(token_lengths) if token_lengths else 0.0,
        "max_prompt_tokens": max(token_lengths) if token_lengths else 0,
        "over_window_count": len(over_window),
        "over_window_examples": over_window[:10],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if over_window and args.fail_on_over_window:
        raise SystemExit(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
