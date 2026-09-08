from __future__ import annotations

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
from .runner import native_generate, native_prefix_kv
from .runtime import HybridPrefill, append_suffix_and_generate, dynamic_cache_from_legacy, load_causal_lm, load_tokenizer


def layer_swap_sensitivity(config: ExperimentConfig) -> dict[str, Any]:
    """Measure how harmful A->B K/V replacement is at each individual layer.

    This is the paper-style diagnostic: start with a completely native B cache,
    replace exactly one B layer's K/V by A's layer, and decode with B.  It is
    not partial recomputation and it does not select a group by itself.
    """
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
    num_layers = int(receiver.config.num_hidden_layers)
    tensors = [
        tokenize_example(tokenizer, example, config.data.max_prompt_tokens, config.data.prompt_suffix_tokens, config.data.input_format)
        for example in examples
    ]
    sender_caches = [store.load(example.identifier, layers=[]) for example in examples]
    native_prefixes = []
    native_predictions = []
    for tensor in tensors:
        generated, _ = native_generate(receiver, tensor, config.data.max_new_tokens, config.models.device_receiver)
        native_predictions.append(tokenizer.decode(generated, skip_special_tokens=True))
        native_prefixes.append(native_prefix_kv(receiver, tensor, config.models.device_receiver)[0])
    native_quality = aggregate_scores(native_predictions, [item.answers for item in examples])
    layers = []
    for layer_idx in range(num_layers):
        predictions = []
        for tensor, sender_cache, native_kv in zip(tensors, sender_caches, native_prefixes, strict=True):
            mixed = list(native_kv)
            mixed[layer_idx] = sender_cache.legacy_kv[layer_idx]
            cache = dynamic_cache_from_legacy(receiver, mixed, config.models.device_receiver)
            _, ids, _ = append_suffix_and_generate(
                receiver,
                HybridPrefill(cache=cache, last_hidden_state=None, elapsed_ms=0.0),
                tensor.suffix_input_ids,
                tensor.full_attention_mask,
                config.data.max_new_tokens,
            )
            predictions.append(tokenizer.decode(ids, skip_special_tokens=True))
        scores = aggregate_scores(predictions, [item.answers for item in examples])
        layers.append(
            {
                "layer": layer_idx,
                "scores": scores,
                "qa_f1_drop_absolute": native_quality["qa_f1"] - scores["qa_f1"],
                "exact_match_drop_absolute": native_quality["exact_match"] - scores["exact_match"],
            }
        )
    output = config.output_dir / "sensitivity" / "single_layer_kv_swap.json"
    atomic_json_write(
        output,
        {
            "method": "receiver-native cache with exactly one layer replaced by sender K/V",
            "native": native_quality,
            "layers": layers,
        },
    )
    return {"sensitivity_path": str(output), "layers": num_layers, "native": native_quality}
