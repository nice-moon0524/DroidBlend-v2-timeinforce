from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer

from core.model_loading import make_prompt


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_tokenizer(path: str | None, local_files_only: bool, trust_remote_code: bool):
    if not path:
        return None
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            path,
            use_fast=True,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except Exception as exc:
        warnings.warn(f"Failed to load tokenizer from {path!r}; falling back to approximate counting. Error: {exc}")
        return None


def _count_prompt_tokens(tokenizer, prompt: str) -> tuple[int, str]:
    if tokenizer is not None:
        token_ids = tokenizer(prompt, add_special_tokens=False, truncation=False)["input_ids"]
        return len(token_ids), "exact"
    return max(1, round(len(prompt) / 4)), "approximate_char4"


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether LongBench HotpotQA prompts exceed the Mistral sliding window.")
    parser.add_argument(
        "--data-file",
        default="../DroidSpeak-new/data/processed/longbench_hotpotqa_test_50.jsonl",
        help="Relative path from the DroidBlend-v2-timeinforce repo root, or an absolute path.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Optional tokenizer path for exact token counts. If omitted, a rough character-based estimate is used.",
    )
    parser.add_argument("--sliding-window", type=int, default=4096)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--fail-on-over-window", action="store_true")
    args = parser.parse_args()

    data_file = _resolve(args.data_file)
    if not data_file.is_file():
        raise FileNotFoundError(data_file)

    tokenizer = _load_tokenizer(args.tokenizer_path, args.local_files_only, args.trust_remote_code)
    rows = _read_jsonl(data_file)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    token_lengths: list[int] = []
    over_window: list[dict] = []
    counting_mode = "exact" if tokenizer is not None else "approximate_char4"
    for row in rows:
        prompt = make_prompt(row)
        token_len, counting_mode = _count_prompt_tokens(tokenizer, prompt)
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
        "counting_mode": counting_mode,
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
