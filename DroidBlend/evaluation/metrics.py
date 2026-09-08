"""SQuAD-style token F1 for short-form QA."""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable, Sequence


def normalize_answer(text: str) -> str:
    """Lowercase, normalize numeric forms, remove punctuation, and collapse space."""
    text = str(text).lower()
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _tokens(text: str) -> list[str]:
    return normalize_answer(text).split()


def f1_score(prediction: str, ground_truth: str) -> float:
    """Compute multiset token F1, including well-defined empty-answer behavior."""
    prediction_tokens = _tokens(prediction)
    reference_tokens = _tokens(ground_truth)
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)
    overlap = Counter(prediction_tokens) & Counter(reference_tokens)
    matches = sum(overlap.values())
    if not matches:
        return 0.0
    precision = matches / len(prediction_tokens)
    recall = matches / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def max_f1_score(prediction: str, ground_truths: str | Sequence[str]) -> float:
    """Score against the best accepted reference answer."""
    references = [ground_truths] if isinstance(ground_truths, str) else list(ground_truths)
    return max((f1_score(prediction, answer) for answer in references), default=0.0)


def batch_f1_score(predictions: Iterable[str], ground_truths: Iterable[str | Sequence[str]]) -> float:
    scores = [max_f1_score(prediction, answer) for prediction, answer in zip(predictions, ground_truths)]
    return sum(scores) / len(scores) if scores else 0.0
