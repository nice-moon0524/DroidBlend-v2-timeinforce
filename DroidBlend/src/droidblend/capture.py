from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Literal

from .artifacts import SCHEMA_VERSION, CacheManifest, stable_hash
from .cache_store import SenderCacheStore, SenderPrefixCache
from .compat import inspect_model, validate_pair
from .config import ExperimentConfig
from .data import load_jsonl, require_disjoint
from .errors import ArtifactError
from .inputs import tokenize_example
from .runtime import capture_layer_inputs, legacy_kv_from_cache, load_causal_lm, load_tokenizer
from .schedules import enumerate_layer_groups, transition_layers


def cache_root_for_split(config: ExperimentConfig, split: Literal["calibration", "evaluation"]) -> Path:
    return config.cache_dir / split


def _cast_cache_tensor(tensor, dtype_name: str):
    """Apply storage policy without ever casting token IDs/masks."""
    import torch

    try:
        dtype = getattr(torch, dtype_name)
    except AttributeError as exc:
        raise ArtifactError(f"Unsupported cache.storage_dtype: {dtype_name}") from exc
    return tensor.detach().to(device="cpu", dtype=dtype).contiguous()


def capture_sender_caches(config: ExperimentConfig, split: Literal["calibration", "evaluation"]) -> dict:
    """Run the sender exactly once per independent prompt and persist its E/KV state."""
    config.assert_model_paths_configured()
    sender = inspect_model(config.models.sender_path, config.models.tokenizer_path, config.models.trust_remote_code)
    receiver = inspect_model(config.models.receiver_path, config.models.tokenizer_path, config.models.trust_remote_code)
    validate_pair(sender, receiver).require()
    groups = enumerate_layer_groups(
        receiver.num_layers,
        config.profiling.recompute_lengths,
        config.profiling.min_recompute_layers,
        config.profiling.max_recompute_layers,
        config.profiling.start_layers,
    )
    layers = transition_layers(groups)
    if config.cache.capture_transition_layers != "all":
        layers = sorted({int(item) for item in config.cache.capture_transition_layers})
        missing = set(transition_layers(groups)) - set(layers)
        if missing:
            raise ArtifactError(f"cache.capture_transition_layers omits required group starts: {sorted(missing)}")
    data_path = config.resolve(config.data.calibration_jsonl if split == "calibration" else config.data.evaluation_jsonl)
    examples = load_jsonl(data_path)
    if split == "evaluation":
        require_disjoint(
            load_jsonl(config.resolve(config.data.calibration_jsonl)), examples, "calibration", "evaluation"
        )
    tokenizer = load_tokenizer(config.models.tokenizer_path, config.models.trust_remote_code)
    manifest_examples = []
    prepared = []
    for example in examples:
        tensors = tokenize_example(tokenizer, example, config.data.max_prompt_tokens, config.data.prompt_suffix_tokens, config.data.input_format)
        manifest_examples.append(
            {
                "id": example.identifier,
                "prompt_hash": stable_hash(example.prompt),
                "prefix_tokens": int(tensors.prefix_attention_mask.sum().item()),
                "full_tokens": int(tensors.full_attention_mask.sum().item()),
            }
        )
        prepared.append((example, tensors))
    manifest = CacheManifest(
        schema_version=SCHEMA_VERSION,
        sender=sender,
        receiver=receiver,
        common_tokenizer_path=str(config.resolve(config.models.tokenizer_path)),
        transition_layers=tuple(layers),
        prompt_suffix_tokens=config.data.prompt_suffix_tokens,
        storage_dtype=config.cache.storage_dtype,
        examples=tuple(manifest_examples),
    )
    store = SenderCacheStore(cache_root_for_split(config, split), overwrite=config.cache.overwrite)
    store.initialize(manifest)
    model = load_causal_lm(
        config.models.sender_path,
        config.models.torch_dtype,
        config.models.device_sender,
        config.models.trust_remote_code,
        config.models.attention_implementation,
    )
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ArtifactError("torch is required to capture sender cache") from exc
    captured_count = 0
    for example, tensors in prepared:
        if store.exists(example.identifier) and not config.cache.overwrite:
            continue
        ids = tensors.prefix_input_ids.to(config.models.device_sender)
        mask = tensors.prefix_attention_mask.to(config.models.device_sender)
        with capture_layer_inputs(model, layers) as e_caches:
            with torch.inference_mode():
                output = model(input_ids=ids, attention_mask=mask, use_cache=True, return_dict=True)
        absent = set(layers) - set(e_caches)
        if absent:
            raise ArtifactError(f"Sender failed to capture E caches at layers {sorted(absent)}")
        store.save(
            example.identifier,
            SenderPrefixCache(
                input_ids=tensors.prefix_input_ids.cpu(),
                attention_mask=tensors.prefix_attention_mask.cpu(),
                legacy_kv=[
                    (_cast_cache_tensor(key, config.cache.storage_dtype), _cast_cache_tensor(value, config.cache.storage_dtype))
                    for key, value in legacy_kv_from_cache(output.past_key_values)
                ],
                e_caches={layer: _cast_cache_tensor(hidden, config.cache.storage_dtype) for layer, hidden in e_caches.items()},
            ),
            metadata={
                "id": example.identifier,
                "prompt_hash": stable_hash(example.prompt),
                "answers": list(example.answers),
                "metadata": example.metadata,
                "split": split,
            },
        )
        captured_count += 1
    del model
    return {
        "split": split,
        "cache_root": str(store.root),
        "manifest_digest": manifest.digest,
        "examples": len(examples),
        "captured_now": captured_count,
        "transition_layers": layers,
    }
