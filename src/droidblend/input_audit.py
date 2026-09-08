"""Preflight audit for the exact shared A/B token sequence."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from .artifacts import atomic_json_write
from .config import ExperimentConfig
from .data import load_jsonl
from .inputs import encode_example
from .runtime import load_tokenizer


def _vocabulary_digest(tokenizer: Any) -> str:
    rows = sorted(tokenizer.get_vocab().items())
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def audit_inputs(config: ExperimentConfig, split: Literal["calibration", "evaluation"]) -> dict[str, Any]:
    """Report true receiver-tokenizer lengths without silently truncating any prompt."""
    config.assert_model_paths_configured()
    source = config.data.calibration_jsonl if split == "calibration" else config.data.evaluation_jsonl
    examples = load_jsonl(config.resolve(source))
    tokenizer = load_tokenizer(config.models.tokenizer_path, config.models.trust_remote_code)
    rows = []
    for example in examples:
        encoded = encode_example(tokenizer, example, config.data.input_format)
        ids = encoded.input_ids[0]
        tail_ids = ids[-min(96, ids.shape[0]) :]
        rows.append(
            {
                "id": example.identifier,
                "token_count": encoded.token_count,
                "max_prompt_tokens": config.data.max_prompt_tokens,
                "within_limit": encoded.token_count <= config.data.max_prompt_tokens,
                "rendered_tail": tokenizer.decode(tail_ids, skip_special_tokens=False),
            }
        )
    counts = [row["token_count"] for row in rows]
    result = {
        "split": split,
        "input_format": config.data.input_format,
        "tokenizer_vocab_digest": _vocabulary_digest(tokenizer),
        "chat_template_sha256": hashlib.sha256((tokenizer.chat_template or "").encode("utf-8")).hexdigest(),
        "max_prompt_tokens": config.data.max_prompt_tokens,
        "samples": len(rows),
        "min_tokens": min(counts),
        "mean_tokens": sum(counts) / len(counts),
        "max_tokens": max(counts),
        "over_limit_examples": [row["id"] for row in rows if not row["within_limit"]],
        "all_within_limit": all(row["within_limit"] for row in rows),
        "examples": rows,
    }
    path = config.output_dir / "input_audit" / f"{split}.json"
    atomic_json_write(path, result)
    return {"input_audit_path": str(path), **result}
