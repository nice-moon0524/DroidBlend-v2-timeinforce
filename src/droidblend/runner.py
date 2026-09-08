from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .cache_store import SenderPrefixCache
from .errors import ArtifactError
from .inputs import PromptTensors
from .runtime import (
    HybridPrefill,
    append_suffix_and_generate,
    dynamic_cache_from_legacy,
    legacy_kv_from_cache,
    partial_prefill,
    synchronize,
    tensor_nbytes,
)
from .schedules import LayerGroup


@dataclass
class HybridRun:
    generated_ids: list[int]
    prefill_ms: float
    suffix_and_decode_ms: float
    cache_error: float | None
    transfer_bytes: int


def raw_sender_kv_run(receiver: Any, cache: SenderPrefixCache, tensors: PromptTensors, device: str, max_new_tokens: int) -> HybridRun:
    """Zero-recomputation baseline: B decodes directly from the complete A cache."""
    assert_cached_prefix_matches(cache, tensors)
    dynamic_cache = dynamic_cache_from_legacy(receiver, cache.legacy_kv, device)
    _, generated, suffix_ms = append_suffix_and_generate(
        receiver,
        HybridPrefill(cache=dynamic_cache, last_hidden_state=None, elapsed_ms=0.0),
        tensors.suffix_input_ids,
        tensors.full_attention_mask,
        max_new_tokens,
    )
    return HybridRun(
        generated_ids=generated,
        prefill_ms=0.0,
        suffix_and_decode_ms=suffix_ms,
        cache_error=None,
        transfer_bytes=tensor_nbytes(cache.legacy_kv),
    )


def _cache_relative_l2(observed: list[tuple[Any, Any]], reference: list[tuple[Any, Any]]) -> float:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ArtifactError("torch is required for cache error computation") from exc
    numerator = 0.0
    denominator = 0.0
    for (ok, ov), (rk, rv) in zip(observed, reference, strict=True):
        numerator += float((ok.float() - rk.float()).square().sum().item())
        numerator += float((ov.float() - rv.float()).square().sum().item())
        denominator += float(rk.float().square().sum().item() + rv.float().square().sum().item())
    return (numerator / max(denominator, 1e-12)) ** 0.5


def assert_cached_prefix_matches(cache: SenderPrefixCache, tensors: PromptTensors) -> None:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ArtifactError("torch is required to validate cached prefixes") from exc
    if not torch.equal(cache.input_ids, tensors.prefix_input_ids) or not torch.equal(cache.attention_mask, tensors.prefix_attention_mask):
        raise ArtifactError(
            "Cached sender prefix does not equal the current tokenizer result. "
            "Use the same prompt, tokenizer, truncation and prompt_suffix_tokens as capture."
        )


def hybrid_run(
    receiver: Any,
    cache: SenderPrefixCache,
    tensors: PromptTensors,
    group: LayerGroup,
    device: str,
    max_new_tokens: int,
    reference_receiver_prefix_kv: list[tuple[Any, Any]] | None = None,
) -> HybridRun:
    assert_cached_prefix_matches(cache, tensors)
    if group.start not in cache.e_caches:
        raise ArtifactError(f"Sender cache does not contain E cache for required start layer {group.start}")
    synchronize(device)
    prefill = partial_prefill(
        receiver,
        cache.legacy_kv,
        cache.e_caches[group.start],
        tensors.prefix_attention_mask,
        group,
        device,
    )
    synchronize(device)
    hybrid_kv = legacy_kv_from_cache(prefill.cache)
    error = _cache_relative_l2(hybrid_kv, reference_receiver_prefix_kv) if reference_receiver_prefix_kv else None
    _, generated, suffix_ms = append_suffix_and_generate(
        receiver,
        prefill,
        tensors.suffix_input_ids,
        tensors.full_attention_mask,
        max_new_tokens,
    )
    synchronize(device)
    return HybridRun(
        generated_ids=generated,
        prefill_ms=prefill.elapsed_ms,
        suffix_and_decode_ms=suffix_ms,
        cache_error=error,
        transfer_bytes=tensor_nbytes(cache.legacy_kv) + sum(
            hidden.numel() * hidden.element_size() for hidden in cache.e_caches.values()
        ),
    )


def native_prefix_kv(receiver: Any, tensors: PromptTensors, device: str) -> tuple[list[tuple[Any, Any]], float]:
    synchronize(device)
    started = time.perf_counter()
    with __import__("torch").inference_mode():
        output = receiver(
            input_ids=tensors.prefix_input_ids.to(device),
            attention_mask=tensors.prefix_attention_mask.to(device),
            use_cache=True,
            return_dict=True,
        )
    synchronize(device)
    return legacy_kv_from_cache(output.past_key_values), (time.perf_counter() - started) * 1000


def native_generate(receiver: Any, tensors: PromptTensors, max_new_tokens: int, device: str) -> tuple[list[int], float]:
    synchronize(device)
    started = time.perf_counter()
    with __import__("torch").inference_mode():
        output = receiver.generate(
            input_ids=tensors.full_input_ids.to(device),
            attention_mask=tensors.full_attention_mask.to(device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=getattr(receiver.config, "pad_token_id", None),
        )
    synchronize(device)
    return output[0, tensors.full_input_ids.shape[1] :].tolist(), (time.perf_counter() - started) * 1000
