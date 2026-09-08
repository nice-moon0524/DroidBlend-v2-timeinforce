#!/usr/bin/env python
"""Download LongBench HotpotQA and build a no-truncation 16K experiment set.

The filtering tokenizer and chat template must be identical to the receiver used
on the server.  This script therefore counts the fully rendered receiver-chat
prompt with no truncation before selecting any example.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer


SYSTEM_INSTRUCTION = (
    "Answer the question based on the given passages. Only give me the answer "
    "and do not output any other words. The following are the passages:"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tokenizer_vocab_digest(tokenizer: Any) -> str:
    rows = sorted(tokenizer.get_vocab().items())
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def render_prompt(tokenizer: Any, context: str, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": f"{context.strip()}\n\nQuestion: {question.strip()}"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def token_count(tokenizer: Any, context: str, question: str) -> int:
    rendered = render_prompt(tokenizer, context, question)
    encoded = tokenizer(rendered, add_special_tokens=False, truncation=False)
    return len(encoded["input_ids"])


def plain_prompt(context: str, question: str) -> str:
    """Compatibility field for cache fingerprints and plain-mode diagnostics.

    The formal experiment uses ``receiver_chat`` and therefore does not feed this
    text directly to either model.  It must nevertheless be stable and answer-free.
    """
    return f"Context:\n{context.strip()}\n\nQuestion: {question.strip()}\n\nAnswer:"


def load_hotpotqa_from_zip(path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        candidates = [name for name in names if name.replace("\\", "/").endswith("data/hotpotqa.jsonl")]
        if len(candidates) != 1:
            raise RuntimeError(f"Cannot find a unique data/hotpotqa.jsonl in {path}: {candidates}")
        with archive.open(candidates[0]) as handle:
            return [json.loads(line) for line in handle if line.strip()]


def normalize_row(row: dict[str, Any], source_index: int, count: int) -> dict[str, Any]:
    context = row.get("context")
    question = row.get("input", row.get("question"))
    answers = row.get("answers", row.get("answer"))
    if not isinstance(context, str) or not context.strip():
        raise ValueError("missing context")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("missing question")
    if isinstance(answers, str):
        answers = [answers]
    if not isinstance(answers, list) or not all(isinstance(answer, str) and answer.strip() for answer in answers):
        raise ValueError("missing answers")
    source_id = str(row.get("_id", row.get("id", source_index)))
    return {
        "id": f"longbench-hotpotqa-{source_index:03d}-{source_id[:12]}",
        "source_id": source_id,
        "source_index": source_index,
        "prompt": plain_prompt(context, question),
        "context": context,
        "question": question,
        "answers": answers,
        "token_count": count,
        "source": "zai-org/LongBench:hotpotqa",
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a 16K-safe LongBench HotpotQA split for DroidBlend")
    parser.add_argument("--output-dir", type=Path, default=Path("data/longbench_hotpotqa_16k"))
    parser.add_argument(
        "--tokenizer-path",
        default="unsloth/Meta-Llama-3.1-8B-Instruct",
        help="Receiver tokenizer directory or Hugging Face id; it must match the server receiver tokenizer.",
    )
    parser.add_argument(
        "--tokenizer-reference",
        default="unsloth/Meta-Llama-3.1-8B-Instruct",
        help="Stable provenance label recorded in the manifest when --tokenizer-path is a local cache path.",
    )
    # The server may use a different Transformers tokenizer implementation from
    # the local builder. Keep a small headroom below the 16,384-token runtime cap.
    parser.add_argument("--max-prompt-tokens", type=int, default=16000)
    parser.add_argument("--num-examples", type=int, default=50)
    parser.add_argument("--calibration-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/longbench"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.calibration_size <= 0 or args.calibration_size >= args.num_examples:
        raise SystemExit("--calibration-size must be positive and smaller than --num-examples")
    if args.max_prompt_tokens <= 0:
        raise SystemExit("--max-prompt-tokens must be positive")
    if args.output_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Output directory already exists: {args.output_dir}; use --overwrite after checking it")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    zip_path = args.raw_dir / "data.zip"
    if zip_path.is_file():
        print(f"Reusing existing LongBench archive: {zip_path}")
    else:
        try:
            zip_name = hf_hub_download(
                repo_id="zai-org/LongBench",
                repo_type="dataset",
                filename="data.zip",
                local_dir=str(args.raw_dir),
            )
        except Exception as exc:
            raise SystemExit(f"Failed to download zai-org/LongBench data.zip: {exc}") from exc
        zip_path = Path(zip_name)
        print(f"Downloaded LongBench archive: {zip_path}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    except Exception as exc:
        raise SystemExit(
            "Failed to load the receiver tokenizer. Provide the exact local tokenizer directory with "
            f"--tokenizer-path. Current value: {args.tokenizer_path!r}. Original error: {exc}"
        ) from exc
    if not getattr(tokenizer, "chat_template", None):
        raise SystemExit("The selected tokenizer has no chat_template; it is not suitable for receiver_chat filtering")

    source_rows = load_hotpotqa_from_zip(zip_path)
    eligible: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    over_limit: list[dict[str, Any]] = []
    for index, source_row in enumerate(source_rows):
        try:
            count = token_count(tokenizer, source_row["context"], source_row.get("input", source_row.get("question", "")))
            row = normalize_row(source_row, index, count)
        except (KeyError, TypeError, ValueError) as exc:
            invalid_rows.append({"source_index": index, "reason": str(exc)})
            continue
        if count <= args.max_prompt_tokens:
            eligible.append(row)
        else:
            over_limit.append({"id": row["id"], "source_index": index, "token_count": count})
    if len(eligible) < args.num_examples:
        raise SystemExit(
            f"Only {len(eligible)} of {len(source_rows)} rows fit within {args.max_prompt_tokens} tokens; "
            f"need {args.num_examples}. No dataset was selected."
        )

    rng = random.Random(args.seed)
    rng.shuffle(eligible)
    selected = eligible[: args.num_examples]
    calibration = selected[: args.calibration_size]
    evaluation = selected[args.calibration_size :]
    write_jsonl(args.output_dir / "calibration.jsonl", calibration)
    write_jsonl(args.output_dir / "evaluation.jsonl", evaluation)
    manifest = {
        "source_repo": "zai-org/LongBench",
        "source_file": "data.zip:data/hotpotqa.jsonl",
        "archive_path": str(zip_path),
        "archive_sha256": sha256(zip_path),
        "receiver_tokenizer": args.tokenizer_reference,
        "receiver_tokenizer_path_used": args.tokenizer_path,
        "receiver_tokenizer_vocab_digest": tokenizer_vocab_digest(tokenizer),
        "receiver_chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode("utf-8")).hexdigest(),
        "selection_prompt_format": "receiver_chat",
        "max_prompt_tokens": args.max_prompt_tokens,
        "seed": args.seed,
        "source_examples": len(source_rows),
        "invalid_examples": invalid_rows,
        "over_limit_examples": over_limit,
        "eligible_examples": len(eligible),
        "selected_examples": len(selected),
        "calibration_examples": len(calibration),
        "evaluation_examples": len(evaluation),
        "selected_ids": [row["id"] for row in selected],
    }
    (args.output_dir / "selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: manifest[key] for key in (
        "source_examples", "eligible_examples", "selected_examples", "calibration_examples", "evaluation_examples"
    )}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
