from __future__ import annotations

import json
from random import Random
from dataclasses import dataclass
from pathlib import Path

from .errors import ConfigurationError


@dataclass(frozen=True)
class PromptExample:
    identifier: str
    prompt: str
    answers: tuple[str, ...]
    metadata: dict[str, object]


def load_jsonl(path: str | Path, limit: int | None = None) -> list[PromptExample]:
    path = Path(path)
    if not path.is_file():
        raise ConfigurationError(f"Dataset JSONL does not exist: {path}")
    items: list[PromptExample] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ConfigurationError(f"Invalid JSONL at {path}:{lineno}: {exc}") from exc
            identifier = str(raw.get("id", ""))
            prompt = raw.get("prompt")
            answers = raw.get("answers")
            if not identifier or not isinstance(prompt, str) or not isinstance(answers, list):
                raise ConfigurationError(
                    f"{path}:{lineno} must contain non-empty id, string prompt and list answers"
                )
            if identifier in seen:
                raise ConfigurationError(f"Duplicate example id in {path}: {identifier}")
            if not all(isinstance(answer, str) for answer in answers):
                raise ConfigurationError(f"{path}:{lineno} answers must be strings")
            seen.add(identifier)
            metadata = {key: value for key, value in raw.items() if key not in {"id", "prompt", "answers"}}
            items.append(PromptExample(identifier, prompt, tuple(answers), metadata))
            if limit is not None and len(items) >= limit:
                break
    if not items:
        raise ConfigurationError(f"Dataset is empty: {path}")
    return items


def require_disjoint(left: list[PromptExample], right: list[PromptExample], left_name: str, right_name: str) -> None:
    """Prevent calibration prompts from silently reappearing in held-out evaluation."""
    left_ids = {item.identifier for item in left}
    shared_ids = left_ids & {item.identifier for item in right}
    if shared_ids:
        raise ConfigurationError(
            f"{left_name} and {right_name} share example id(s), e.g. {sorted(shared_ids)[:5]}"
        )
    left_prompts = {item.prompt for item in left}
    shared_prompts = left_prompts & {item.prompt for item in right}
    if shared_prompts:
        raise ConfigurationError(f"{left_name} and {right_name} share prompt text; held-out evaluation is invalid")


def split_jsonl(
    source: str | Path,
    calibration_destination: str | Path,
    evaluation_destination: str | Path,
    calibration_size: int,
    seed: int,
) -> dict[str, object]:
    """固定随机种子地切分 JSONL，避免标定集泄漏到最终评测集。"""
    source_path = Path(source)
    calibration_path = Path(calibration_destination)
    evaluation_path = Path(evaluation_destination)
    if calibration_path.resolve() == evaluation_path.resolve():
        raise ConfigurationError("Calibration and evaluation destinations must be different files")
    examples = load_jsonl(source_path)
    if calibration_size <= 0 or calibration_size >= len(examples):
        raise ConfigurationError(
            f"calibration_size must be in [1, {len(examples) - 1}], got {calibration_size}"
        )
    indices = list(range(len(examples)))
    Random(seed).shuffle(indices)
    calibration_indices = set(indices[:calibration_size])
    calibration = [item for index, item in enumerate(examples) if index in calibration_indices]
    evaluation = [item for index, item in enumerate(examples) if index not in calibration_indices]
    require_disjoint(calibration, evaluation, "calibration", "evaluation")

    def write(destination: Path, rows: list[PromptExample]) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for item in rows:
                row = {"id": item.identifier, "prompt": item.prompt, "answers": list(item.answers), **item.metadata}
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(destination)

    write(calibration_path, calibration)
    write(evaluation_path, evaluation)
    return {
        "source": str(source_path),
        "calibration_path": str(calibration_path),
        "evaluation_path": str(evaluation_path),
        "seed": seed,
        "calibration_examples": len(calibration),
        "evaluation_examples": len(evaluation),
    }
