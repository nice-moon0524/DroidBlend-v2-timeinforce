"""Focused diagnostics for validating DroidSpeak benchmark assumptions."""

from __future__ import annotations

import random
from typing import Any, Iterable, Literal

from .artifacts import CacheManifest, atomic_json_write, require_same_fingerprint
from .cache_store import SenderCacheStore
from .capture import cache_root_for_split
from .compat import inspect_model
from .config import ExperimentConfig
from .data import PromptExample, load_jsonl
from .inputs import tokenize_example
from .metrics import aggregate_scores, exact_match, qa_f1
from .runner import native_generate, native_prefix_kv, raw_sender_kv_run
from .runtime import HybridPrefill, append_suffix_and_generate, dynamic_cache_from_legacy, load_causal_lm, load_tokenizer


def _prepared(
    config: ExperimentConfig, split: Literal["calibration", "evaluation"]
) -> tuple[list[PromptExample], Any, Any, SenderCacheStore, CacheManifest, list[Any]]:
    source = config.data.calibration_jsonl if split == "calibration" else config.data.evaluation_jsonl
    examples = load_jsonl(config.resolve(source))
    tokenizer = load_tokenizer(config.models.tokenizer_path, config.models.trust_remote_code)
    store = SenderCacheStore(cache_root_for_split(config, split))
    if not store.manifest_path.is_file():
        raise RuntimeError(f"Sender cache for {split} is missing; run capture-sender --split {split} first.")
    manifest = CacheManifest.read(store.manifest_path)
    receiver = load_causal_lm(
        config.models.receiver_path,
        config.models.torch_dtype,
        config.models.device_receiver,
        config.models.trust_remote_code,
        config.models.attention_implementation,
    )
    current = inspect_model(config.models.receiver_path, config.models.tokenizer_path, config.models.trust_remote_code)
    require_same_fingerprint(manifest.receiver, current, "Receiver")
    tensors = [
        tokenize_example(
            tokenizer,
            example,
            config.data.max_prompt_tokens,
            config.data.prompt_suffix_tokens,
            config.data.input_format,
        )
        for example in examples
    ]
    return examples, tokenizer, receiver, store, manifest, tensors


def direct_reuse_diagnostic(config: ExperimentConfig, split: Literal["calibration", "evaluation"]) -> dict[str, Any]:
    """Compare native B against decoding from the complete A KV cache (Figure 3)."""
    examples, tokenizer, receiver, store, _manifest, tensors = _prepared(config, split)
    native_predictions: list[str] = []
    direct_predictions: list[str] = []
    direct_lengths: list[int] = []
    transfer_bytes: list[int] = []
    per_example: list[dict[str, Any]] = []
    for example, tensor in zip(examples, tensors, strict=True):
        native_ids, _ = native_generate(receiver, tensor, config.data.max_new_tokens, config.models.device_receiver)
        native = tokenizer.decode(native_ids, skip_special_tokens=True)
        cache = store.load(example.identifier, layers=[])
        direct = raw_sender_kv_run(receiver, cache, tensor, config.models.device_receiver, config.data.max_new_tokens)
        direct_text = tokenizer.decode(direct.generated_ids, skip_special_tokens=True)
        native_predictions.append(native)
        direct_predictions.append(direct_text)
        direct_lengths.append(len(direct.generated_ids))
        transfer_bytes.append(direct.transfer_bytes)
        per_example.append(
            {
                "id": example.identifier,
                "answers": list(example.answers),
                "native_prediction": native,
                "direct_sender_kv_prediction": direct_text,
                "native_qa_f1": qa_f1(native, example.answers),
                "direct_sender_kv_qa_f1": qa_f1(direct_text, example.answers),
            }
        )
    answers = [example.answers for example in examples]
    native_quality = aggregate_scores(native_predictions, answers)
    direct_quality = aggregate_scores(direct_predictions, answers)
    result = {
        "method": "receiver B decodes from the complete sender A KV cache; no B prefix layer is recomputed",
        "split": split,
        "samples": len(examples),
        "native_receiver": native_quality,
        "direct_sender_kv": {
            "quality": direct_quality,
            "mean_generated_tokens": sum(direct_lengths) / max(1, len(direct_lengths)),
            "mean_transfer_bytes": sum(transfer_bytes) / max(1, len(transfer_bytes)),
        },
        "receiver_minus_direct_sender_kv": {
            key: native_quality[key] - direct_quality[key] for key in native_quality
        },
        "relative_qa_f1_loss": (native_quality["qa_f1"] - direct_quality["qa_f1"])
        / max(abs(native_quality["qa_f1"]), 1e-12),
        "per_example": per_example,
    }
    path = config.output_dir / "diagnostics" / f"direct_reuse_{split}.json"
    atomic_json_write(path, result)
    return {
        "direct_reuse_path": str(path),
        "split": split,
        "samples": len(examples),
        "native_receiver": native_quality,
        "direct_sender_kv": result["direct_sender_kv"],
        "receiver_minus_direct_sender_kv": result["receiver_minus_direct_sender_kv"],
        "relative_qa_f1_loss": result["relative_qa_f1_loss"],
    }


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[max(0, min(index, len(ordered) - 1))]


def targeted_sensitivity_stability(
    config: ExperimentConfig, layers: Iterable[int], bootstrap_samples: int
) -> dict[str, Any]:
    """Measure per-example uncertainty for a small set of suspicious swap layers."""
    checked_layers = sorted({int(layer) for layer in layers})
    if not checked_layers:
        raise ValueError("At least one layer is required")
    examples, tokenizer, receiver, store, _manifest, tensors = _prepared(config, "calibration")
    depth = int(receiver.config.num_hidden_layers)
    if any(layer < 0 or layer >= depth for layer in checked_layers):
        raise ValueError(f"Layers must be within [0, {depth - 1}]")
    native_predictions: list[str] = []
    native_prefixes = []
    for tensor in tensors:
        ids, _ = native_generate(receiver, tensor, config.data.max_new_tokens, config.models.device_receiver)
        native_predictions.append(tokenizer.decode(ids, skip_special_tokens=True))
        native_prefixes.append(native_prefix_kv(receiver, tensor, config.models.device_receiver)[0])
    native_f1 = [qa_f1(prediction, example.answers) for prediction, example in zip(native_predictions, examples, strict=True)]
    native_quality = aggregate_scores(native_predictions, [example.answers for example in examples])
    sender_caches = [store.load(example.identifier, layers=[]) for example in examples]
    rng = random.Random(config.experiment.seed)
    layer_rows = []
    for layer in checked_layers:
        predictions: list[str] = []
        scores: list[float] = []
        details: list[dict[str, Any]] = []
        for example, tensor, sender_cache, native_kv, baseline_text, baseline_f1 in zip(
            examples, tensors, sender_caches, native_prefixes, native_predictions, native_f1, strict=True
        ):
            mixed = list(native_kv)
            mixed[layer] = sender_cache.legacy_kv[layer]
            cache = dynamic_cache_from_legacy(receiver, mixed, config.models.device_receiver)
            _, ids, _ = append_suffix_and_generate(
                receiver,
                HybridPrefill(cache=cache, last_hidden_state=None, elapsed_ms=0.0),
                tensor.suffix_input_ids,
                tensor.full_attention_mask,
                config.data.max_new_tokens,
            )
            prediction = tokenizer.decode(ids, skip_special_tokens=True)
            score = qa_f1(prediction, example.answers)
            predictions.append(prediction)
            scores.append(score)
            details.append(
                {
                    "id": example.identifier,
                    "native_prediction": baseline_text,
                    "swap_prediction": prediction,
                    "native_qa_f1": baseline_f1,
                    "swap_qa_f1": score,
                    "native_minus_swap_qa_f1": baseline_f1 - score,
                }
            )
        deltas = [left - right for left, right in zip(native_f1, scores, strict=True)]
        bootstrap = [
            sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
            for _ in range(bootstrap_samples)
        ]
        layer_rows.append(
            {
                "layer": layer,
                "scores": aggregate_scores(predictions, [example.answers for example in examples]),
                "mean_native_minus_swap_qa_f1": sum(deltas) / len(deltas),
                "bootstrap_95pct_ci_native_minus_swap_qa_f1": [_percentile(bootstrap, 0.025), _percentile(bootstrap, 0.975)],
                "examples_worse_than_native": sum(delta > 0 for delta in deltas),
                "examples_better_than_native": sum(delta < 0 for delta in deltas),
                "examples_tied": sum(delta == 0 for delta in deltas),
                "per_example": details,
            }
        )
    result = {
        "method": "targeted single-layer A-KV swaps with paired per-example QA-F1 and bootstrap confidence intervals",
        "samples": len(examples),
        "bootstrap_samples": bootstrap_samples,
        "native_receiver": native_quality,
        "layers": layer_rows,
    }
    path = config.output_dir / "diagnostics" / "targeted_sensitivity_stability.json"
    atomic_json_write(path, result)
    return {"stability_path": str(path), "samples": len(examples), "layers": checked_layers}
