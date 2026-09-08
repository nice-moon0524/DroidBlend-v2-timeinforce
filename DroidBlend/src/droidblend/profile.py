from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from .artifacts import CacheManifest, atomic_json_write, require_same_fingerprint
from .cache_store import SenderCacheStore
from .capture import cache_root_for_split
from .compat import inspect_model
from .config import ExperimentConfig
from .data import load_jsonl
from .errors import ArtifactError
from .inputs import tokenize_example
from .metrics import aggregate_scores
from .pareto import ProfilePoint
from .runner import hybrid_run, native_generate, native_prefix_kv
from .runtime import load_causal_lm, load_tokenizer
from .schedules import LayerGroup, enumerate_layer_groups


def _score(metric: str, predictions: list[str], answers: list[tuple[str, ...]]) -> float:
    result = aggregate_scores(predictions, answers)
    if metric not in result:
        raise ArtifactError(f"Unsupported quality metric {metric!r}; available: {sorted(result)}")
    return float(result[metric])


def _write_points(path: Path, points: list[ProfilePoint], native_quality: float) -> None:
    atomic_json_write(
        path,
        {
            "native_quality": native_quality,
            "points": [point.to_dict() for point in points],
        },
    )


def profile_droidspeak(config: ExperimentConfig) -> dict[str, Any]:
    """Exhaustively assess legal continuous B-layer groups on calibration data."""
    config.assert_model_paths_configured()
    examples = load_jsonl(config.resolve(config.data.calibration_jsonl))
    tokenizer = load_tokenizer(config.models.tokenizer_path, config.models.trust_remote_code)
    store = SenderCacheStore(cache_root_for_split(config, "calibration"))
    if not store.manifest_path.is_file():
        raise ArtifactError("Calibration sender cache is missing. Run capture-sender --split calibration first.")
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
    groups = enumerate_layer_groups(
        receiver.config.num_hidden_layers,
        config.profiling.recompute_lengths,
        config.profiling.min_recompute_layers,
        config.profiling.max_recompute_layers,
        config.profiling.start_layers,
    )
    if config.profiling.max_profiles_per_run is not None:
        groups = groups[: config.profiling.max_profiles_per_run]
    missing_starts = {group.start for group in groups} - set(manifest.transition_layers)
    if missing_starts:
        raise ArtifactError(f"Calibration cache lacks E caches for selected group starts: {sorted(missing_starts)}")
    tensors = [
        tokenize_example(tokenizer, example, config.data.max_prompt_tokens, config.data.prompt_suffix_tokens, config.data.input_format)
        for example in examples
    ]

    native_predictions: list[str] = []
    native_latencies: list[float] = []
    exact_prefixes = []
    for tensor in tensors:
        generated, _ = native_generate(receiver, tensor, config.data.max_new_tokens, config.models.device_receiver)
        native_predictions.append(tokenizer.decode(generated, skip_special_tokens=True))
        for _ in range(config.profiling.warmup_repetitions):
            native_prefix_kv(receiver, tensor, config.models.device_receiver)
        repeats = [
            native_prefix_kv(receiver, tensor, config.models.device_receiver)
            for _ in range(config.profiling.measure_latency_repetitions)
        ]
        exact_prefixes.append(repeats[-1][0])
        native_latencies.append(mean(item[1] for item in repeats))
    native_quality = _score(config.profiling.quality_metric, native_predictions, [item.answers for item in examples])

    points: list[ProfilePoint] = []
    for group in groups:
        predictions: list[str] = []
        latencies: list[float] = []
        errors: list[float] = []
        transfer_bytes = 0
        for example, tensor, exact in zip(examples, tensors, exact_prefixes, strict=True):
            # Only the transition activation E^s needed by this candidate is
            # transferred; keeping every captured E cache in memory would make
            # the reported communication cost meaningless.
            cached = store.load(example.identifier, layers=[group.start])
            # Warm-up uses exactly the same hybrid execution, never native B prefill.
            for _ in range(config.profiling.warmup_repetitions):
                hybrid_run(receiver, cached, tensor, group, config.models.device_receiver, 0, None)
            repeats = [
                hybrid_run(
                    receiver,
                    cached,
                    tensor,
                    group,
                    config.models.device_receiver,
                    0,
                    None,
                )
                for _ in range(config.profiling.measure_latency_repetitions)
            ]
            selected = hybrid_run(
                receiver,
                cached,
                tensor,
                group,
                config.models.device_receiver,
                config.data.max_new_tokens,
                exact,
            )
            predictions.append(tokenizer.decode(selected.generated_ids, skip_special_tokens=True))
            latencies.append(mean(item.prefill_ms for item in repeats))
            if selected.cache_error is not None:
                errors.append(selected.cache_error)
            transfer_bytes += selected.transfer_bytes
        points.append(
            ProfilePoint(
                group=group,
                quality=_score(config.profiling.quality_metric, predictions, [item.answers for item in examples]),
                native_quality=native_quality,
                quality_metric=config.profiling.quality_metric,
                hybrid_prefill_ms=mean(latencies),
                native_prefill_ms=mean(native_latencies),
                transfer_bytes=transfer_bytes // max(len(examples), 1),
                cache_error=mean(errors) if errors else None,
                samples=len(examples),
            )
        )
    output = config.output_dir / "profiling" / "points.json"
    _write_points(output, points, native_quality)
    return {"points_path": str(output), "native_quality": native_quality, "profiles": len(points)}
