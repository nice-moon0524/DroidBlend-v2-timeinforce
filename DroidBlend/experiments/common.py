"""Shared execution primitives for all reported experiment scripts."""

from __future__ import annotations

from time import perf_counter
from typing import Mapping

import torch

from core.partial_prefill import PartialPrefillResult, finalize_prompt_logits, greedy_decode
from kv_cache.cache_utils import LayerKV, cache_to_layer_kv, clone_layer_kv
from kv_cache.layer_kv_injector import build_partial_cache


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _shared_suffix(left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Align two cache tensors on their common trailing prompt span.

    Sender and receiver caches can legitimately differ in total cached length
    on long contexts (for example when one side uses a shorter sliding-window
    span). CacheBlend only needs a comparable suffix to measure per-token
    differences, so we compare the overlap that both caches actually share.
    """
    shared_length = min(left.shape[-2], right.shape[-2])
    if shared_length <= 0:
        raise ValueError("Cache tensors must contain at least one cached token")
    return left[..., -shared_length:, :], right[..., -shared_length:, :]


@torch.inference_mode()
def receiver_prefill(receiver_model, model_inputs: dict[str, torch.Tensor]) -> PartialPrefillResult:
    output = receiver_model(**model_inputs, use_cache=True, return_dict=True)
    return PartialPrefillResult(output.past_key_values, output.logits[:, -1, :].float())


@torch.inference_mode()
def full_reuse_prefill(receiver_model, sender_kv: Mapping[int, LayerKV], sender_e: Mapping[int, torch.Tensor], input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> PartialPrefillResult:
    """All-layer cache reuse with a one-token receiver decode transition."""
    cache = build_partial_cache(sender_kv, sorted(sender_kv))
    return finalize_prompt_logits(receiver_model, cache, input_ids, attention_mask)


@torch.inference_mode()
def cacheblend_prefill(receiver_model, sender_kv: Mapping[int, LayerKV], receiver_prefill_result: PartialPrefillResult, important_fraction: float, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> PartialPrefillResult:
    """Blend per-token sender/receiver KV based on first-layer cache distance.

    Stock Transformers cannot sparse-prefill arbitrary positions, so receiver
    cache is computed first. The reported latency therefore includes that full
    prefill, which prevents an artificial CacheBlend speed claim.
    """
    receiver_cache = receiver_prefill_result.past_key_values
    receiver_layer_kv = cache_to_layer_kv(receiver_cache)
    sender_first, receiver_first = _shared_suffix(sender_kv[0][0], receiver_layer_kv[0][0])
    distances = (sender_first - receiver_first).float().square().mean(dim=(0, 1, 3))
    count = max(1, int(distances.numel() * important_fraction))
    important = torch.zeros_like(distances, dtype=torch.bool)
    important[torch.topk(distances, count).indices] = True
    mixed = clone_layer_kv(sender_kv)
    for layer_id in mixed:
        sender_key, sender_value = mixed[layer_id]
        receiver_key, receiver_value = receiver_layer_kv[layer_id]
        shared_length = min(sender_key.shape[-2], receiver_key.shape[-2], important.numel())
        if shared_length <= 0:
            raise ValueError("Cache tensors must contain at least one cached token")
        layer_mask = important[-shared_length:]
        sender_key = sender_key[..., -shared_length:, :]
        receiver_key = receiver_key[..., -shared_length:, :]
        sender_value = sender_value[..., -shared_length:, :]
        receiver_value = receiver_value[..., -shared_length:, :]
        sender_key[:, :, layer_mask, :] = receiver_key[:, :, layer_mask, :]
        sender_value[:, :, layer_mask, :] = receiver_value[:, :, layer_mask, :]
    return finalize_prompt_logits(receiver_model, build_partial_cache(mixed, sorted(mixed)), input_ids, attention_mask)


def run_method(method, receiver_model, receiver_tokenizer, max_new_tokens: int, device: str) -> tuple[str, float]:
    synchronize(device)
    started = perf_counter()
    result = method()
    synchronize(device)
    latency = perf_counter() - started
    return greedy_decode(receiver_model, receiver_tokenizer, result, max_new_tokens), latency
