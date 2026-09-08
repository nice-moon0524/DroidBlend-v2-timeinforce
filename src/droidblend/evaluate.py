from __future__ import annotations

from statistics import mean
from typing import Any

from .artifacts import CacheManifest, atomic_json_write, require_same_fingerprint
from .cache_store import SenderCacheStore
from .capture import cache_root_for_split
from .compat import inspect_model
from .config import ExperimentConfig
from .data import load_jsonl, require_disjoint
from .errors import ArtifactError
from .inputs import tokenize_example
from .metrics import aggregate_scores
from .runner import hybrid_run, native_generate, native_prefix_kv, raw_sender_kv_run
from .runtime import load_causal_lm, load_tokenizer
from .selection import resolve_selected_group


def evaluate_selected_profile(config: ExperimentConfig, profile: str | None = None) -> dict[str, Any]:
    """Evaluate a calibration-selected layer group once on held-out prompts."""
    config.assert_model_paths_configured()
    group = resolve_selected_group(config, profile)
    examples = load_jsonl(config.resolve(config.data.evaluation_jsonl))
    require_disjoint(
        load_jsonl(config.resolve(config.data.calibration_jsonl)), examples, "calibration", "evaluation"
    )
    tokenizer = load_tokenizer(config.models.tokenizer_path, config.models.trust_remote_code)
    store = SenderCacheStore(cache_root_for_split(config, "evaluation"))
    if not store.manifest_path.is_file():
        raise ArtifactError("Evaluation sender cache is missing. Run capture-sender --split evaluation first.")
    manifest = CacheManifest.read(store.manifest_path)
    receiver = load_causal_lm(
        config.models.receiver_path,
        config.models.torch_dtype,
        config.models.device_receiver,
        config.models.trust_remote_code,
        config.models.attention_implementation,
    )
    current_receiver = inspect_model(
        config.models.receiver_path, config.models.tokenizer_path, config.models.trust_remote_code
    )
    require_same_fingerprint(manifest.receiver, current_receiver, "Receiver")
    if group.start not in manifest.transition_layers:
        raise ArtifactError(f"Evaluation cache lacks E cache for selected group start {group.start}")
    native_predictions: list[str] = []
    raw_predictions: list[str] = []
    hybrid_predictions: list[str] = []
    answers: list[tuple[str, ...]] = []
    rows: list[dict[str, Any]] = []
    for example in examples:
        tensors = tokenize_example(tokenizer, example, config.data.max_prompt_tokens, config.data.prompt_suffix_tokens, config.data.input_format)
        cache = store.load(example.identifier, layers=[group.start])
        native_ids, native_ms = native_generate(receiver, tensors, config.data.max_new_tokens, config.models.device_receiver)
        raw = raw_sender_kv_run(receiver, cache, tensors, config.models.device_receiver, config.data.max_new_tokens)
        reference_kv = None
        native_prefill_ms = None
        if config.evaluation.collect_cache_error:
            reference_kv, native_prefill_ms = native_prefix_kv(receiver, tensors, config.models.device_receiver)
        run = hybrid_run(
            receiver,
            cache,
            tensors,
            group,
            config.models.device_receiver,
            config.data.max_new_tokens,
            reference_kv,
        )
        native_text = tokenizer.decode(native_ids, skip_special_tokens=True)
        raw_text = tokenizer.decode(raw.generated_ids, skip_special_tokens=True)
        hybrid_text = tokenizer.decode(run.generated_ids, skip_special_tokens=True)
        native_predictions.append(native_text)
        raw_predictions.append(raw_text)
        hybrid_predictions.append(hybrid_text)
        answers.append(example.answers)
        rows.append(
            {
                "id": example.identifier,
                "answers": list(example.answers),
                "native_prediction": native_text,
                "raw_sender_kv_prediction": raw_text,
                "hybrid_prediction": hybrid_text,
                "native_total_ms": native_ms,
                "native_prefix_ms": native_prefill_ms,
                "hybrid_selected_blocks_ms": run.prefill_ms,
                "raw_sender_kv_suffix_and_decode_ms": raw.suffix_and_decode_ms,
                "hybrid_suffix_and_decode_ms": run.suffix_and_decode_ms,
                "cache_relative_l2": run.cache_error,
                "transfer_bytes": run.transfer_bytes if config.evaluation.collect_transfer_bytes else None,
                "raw_sender_kv_transfer_bytes": raw.transfer_bytes if config.evaluation.collect_transfer_bytes else None,
            }
        )
    def mean_present(field: str) -> float | None:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        return mean(values) if values else None

    native_prefill_ms = mean_present("native_prefix_ms")
    hybrid_prefill_ms = mean_present("hybrid_selected_blocks_ms")
    summary = {
        "selected_group": {"start": group.start, "end": group.end, "identifier": group.identifier},
        "samples": len(examples),
        "native": aggregate_scores(native_predictions, answers),
        "raw_sender_kv": aggregate_scores(raw_predictions, answers),
        "droidspeak_hybrid": aggregate_scores(hybrid_predictions, answers),
        "local_prefill_performance": {
            "measurement_scope": "single-GPU local cache materialisation plus selected B-layer recomputation; excludes network transfer and queueing",
            "native_receiver_prefill_ms": native_prefill_ms,
            "droidspeak_hybrid_prefill_ms": hybrid_prefill_ms,
            "prefill_speedup": (
                native_prefill_ms / hybrid_prefill_ms
                if native_prefill_ms is not None and hybrid_prefill_ms is not None and hybrid_prefill_ms > 0
                else None
            ),
            "mean_cache_relative_l2": mean_present("cache_relative_l2"),
            "mean_transfer_mib": (
                mean_present("transfer_bytes") / (1024 * 1024)
                if mean_present("transfer_bytes") is not None
                else None
            ),
        },
        "per_example": rows if config.evaluation.report_per_example else None,
    }
    output = config.output_dir / "evaluation" / f"{group.identifier}.json"
    atomic_json_write(output, summary)
    return {"evaluation_path": str(output), "group": group.identifier, "samples": len(examples)}
