"""Conversion of a local Hugging Face HotpotQA dataset into DroidBlend JSONL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import ConfigurationError


def hotpotqa_prompt(context: str, question: str) -> str:
    """The concise QA template used by the prior DroidSpeak experiment."""
    return (
        "You are a helpful assistant. Answer the question based strictly on the given context. "
        "Provide a very short, concise answer containing ONLY the precise entity name or phrase, "
        "without writing full sentences or repeating the question.\n\n"
        f"Context:\n{context.strip()}\n\nQuestion: {question.strip()}\n\nAnswer:"
    )


def _answers(raw: Any) -> list[str]:
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, dict):
        for name in ("text", "answers", "answer"):
            if name in raw:
                return _answers(raw[name])
    if isinstance(raw, (tuple, list)):
        return [item for item in raw if isinstance(item, str) and item.strip()]
    return []


def convert_hotpotqa_dataset(source: str | Path, destination: str | Path, split: str = "test", limit: int | None = None) -> dict[str, Any]:
    try:
        from datasets import DatasetDict, load_from_disk
    except ImportError as exc:  # pragma: no cover
        raise ConfigurationError("HotpotQA conversion requires datasets; install the [data] extra") from exc
    source = Path(source)
    if not source.is_dir():
        raise ConfigurationError(f"Local Hugging Face dataset does not exist: {source}")
    # datasets/fsspec 在 Windows 下不能可靠地直接接收 pathlib.Path。
    dataset = load_from_disk(str(source))
    if isinstance(dataset, DatasetDict) or hasattr(dataset, "keys"):
        if split not in dataset:
            raise ConfigurationError(f"Dataset has no split {split!r}; available: {list(dataset.keys())}")
        dataset = dataset[split]
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(dataset):
        context = item.get("context", "")
        question = item.get("input", item.get("question", ""))
        answers = _answers(item.get("answers", item.get("answer", [])))
        identifier = str(item.get("id", f"hotpotqa-{split}-{index}"))
        if not isinstance(context, str) or not isinstance(question, str) or not context.strip() or not question.strip() or not answers:
            continue
        if identifier in seen:
            raise ConfigurationError(f"Duplicate source example id: {identifier}")
        seen.add(identifier)
        rows.append(
            {
                "id": identifier,
                "prompt": hotpotqa_prompt(context, question),
                "answers": answers,
                # Keep raw fields for receiver chat-template rendering and audit.
                # ``prompt`` is retained only as a human-readable/plain-mode fallback.
                "context": context,
                "question": question,
                "source": "hotpotqa",
                "source_index": index,
            }
        )
        if limit is not None and len(rows) >= limit:
            break
    if not rows:
        raise ConfigurationError("No usable HotpotQA examples were found")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(destination)
    return {"source": str(source), "destination": str(destination), "split": split, "examples": len(rows)}
