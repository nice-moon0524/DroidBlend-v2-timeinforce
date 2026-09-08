from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Iterable


def normalize_qa_text(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def qa_f1(prediction: str, answers: Iterable[str]) -> float:
    predicted = normalize_qa_text(prediction).split()
    if not predicted:
        return 0.0
    best = 0.0
    for answer in answers:
        target = normalize_qa_text(answer).split()
        if not target:
            continue
        overlap = sum((Counter(predicted) & Counter(target)).values())
        if overlap == 0:
            continue
        precision = overlap / len(predicted)
        recall = overlap / len(target)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def exact_match(prediction: str, answers: Iterable[str]) -> float:
    value = normalize_qa_text(prediction)
    return float(any(value == normalize_qa_text(answer) for answer in answers))


def aggregate(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def aggregate_scores(predictions: Iterable[str], answers: Iterable[Iterable[str]]) -> dict[str, float]:
    """Task quality used consistently by profiling and held-out evaluation."""
    pairs = list(zip(predictions, answers, strict=True))
    return {
        "qa_f1": aggregate(qa_f1(prediction, reference) for prediction, reference in pairs),
        "exact_match": aggregate(exact_match(prediction, reference) for prediction, reference in pairs),
    }
