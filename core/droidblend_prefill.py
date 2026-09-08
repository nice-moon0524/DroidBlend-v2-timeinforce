"""Reference DroidBlend prefill: DroidSpeak layers plus CacheBlend tokens.

This module intentionally implements a correctness/reference path.  Stock
Transformers does not expose an efficient arbitrary-token sparse prefill kernel,
so selected receiver-token KV values are obtained from a full receiver prefill
and then written back into the mixed cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from core.partial_prefill import PartialPrefillResult, finalize_prompt_logits, partial_prefill
from experiments.common import receiver_prefill
from kv_cache.cache_utils import LayerKV, cache_to_layer_kv, clone_layer_kv
from kv_cache.layer_kv_injector import build_partial_cache


@dataclass
class TokenSelection:
    """High-KV-deviation token selection metadata."""

    layer: int
    important_fraction: float
    selected_count: int
    candidate_count: int
    indices: list[int]


@dataclass
class DroidBlendPrefillResult(PartialPrefillResult):
    """Prompt cache plus token-selection metadata."""

    selection: TokenSelection
    recompute_layers: tuple[int, int]
    reference_mode: bool = True


def _shared_token_span(left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shared_length = min(left.shape[-2], right.shape[-2])
    if shared_length <= 0:
        raise ValueError("Cache tensors must contain at least one cached token")
    return left[..., -shared_length:, :], right[..., -shared_length:, :]


def token_kv_distances(sender_layer: LayerKV, receiver_layer: LayerKV) -> torch.Tensor:
    """Compute per-token KV deviation on the common trailing prompt span."""
    sender_key, sender_value = sender_layer
    receiver_key, receiver_value = receiver_layer
    sender_key, receiver_key = _shared_token_span(sender_key, receiver_key)
    sender_value, receiver_value = _shared_token_span(sender_value, receiver_value)
    key_dist = (sender_key - receiver_key).float().square().mean(dim=(0, 1, 3))
    value_dist = (sender_value - receiver_value).float().square().mean(dim=(0, 1, 3))
    return (key_dist + value_dist) / 2


def select_high_deviation_tokens(
    sender_kv: Mapping[int, LayerKV],
    receiver_kv: Mapping[int, LayerKV],
    important_fraction: float = 0.2,
    top_k: int | None = None,
    selection_layer: int = 0,
) -> tuple[torch.Tensor, TokenSelection]:
    """Select HKVD tokens using first-layer sender/receiver KV deviation."""
    if not 0 < important_fraction <= 1:
        raise ValueError("important_fraction must be in (0, 1]")
    if selection_layer not in sender_kv or selection_layer not in receiver_kv:
        raise ValueError(f"Layer {selection_layer} is missing from sender or receiver KV cache")
    distances = token_kv_distances(sender_kv[selection_layer], receiver_kv[selection_layer])
    candidate_count = int(distances.numel())
    count = int(top_k) if top_k is not None else max(1, int(candidate_count * important_fraction))
    count = max(1, min(candidate_count, count))
    indices = torch.topk(distances, count).indices.sort().values
    mask = torch.zeros(candidate_count, dtype=torch.bool, device=distances.device)
    mask[indices] = True
    return mask, TokenSelection(
        layer=selection_layer,
        important_fraction=important_fraction,
        selected_count=count,
        candidate_count=candidate_count,
        indices=[int(item) for item in indices.detach().cpu().tolist()],
    )


def _patch_selected_tokens(base: LayerKV, receiver: LayerKV, token_mask: torch.Tensor) -> LayerKV:
    base_key, base_value = base
    receiver_key, receiver_value = receiver
    shared_length = min(base_key.shape[-2], receiver_key.shape[-2], token_mask.numel())
    if shared_length <= 0:
        raise ValueError("Cache tensors must contain at least one cached token")
    layer_mask = token_mask[-shared_length:].to(device=base_key.device)
    base_start = base_key.shape[-2] - shared_length
    receiver_start = receiver_key.shape[-2] - shared_length
    mixed_key = base_key.clone()
    mixed_value = base_value.clone()
    mixed_key[:, :, base_start:, :][:, :, layer_mask, :] = receiver_key[:, :, receiver_start:, :][:, :, layer_mask, :]
    mixed_value[:, :, base_start:, :][:, :, layer_mask, :] = receiver_value[:, :, receiver_start:, :][:, :, layer_mask, :]
    return mixed_key, mixed_value


def build_droidblend_layer_kv(
    droidspeak_kv: Mapping[int, LayerKV],
    receiver_kv: Mapping[int, LayerKV],
    token_mask: torch.Tensor,
    recompute_layers: tuple[int, int],
) -> dict[int, LayerKV]:
    """Patch selected receiver-token KV into DroidSpeak's mixed cache."""
    start, end = recompute_layers
    mixed = clone_layer_kv(droidspeak_kv)
    for layer_id in sorted(mixed):
        if start <= layer_id <= end:
            continue
        if layer_id not in receiver_kv:
            raise ValueError(f"Receiver KV cache is missing layer {layer_id}")
        mixed[layer_id] = _patch_selected_tokens(mixed[layer_id], receiver_kv[layer_id], token_mask)
    return mixed


@torch.inference_mode()
def droidblend_reference_prefill(
    receiver_model,
    sender_kv_cache: Mapping[int, LayerKV],
    sender_e_cache: Mapping[int, torch.Tensor],
    recompute_layers: tuple[int, int],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    important_fraction: float = 0.2,
    top_k: int | None = None,
    selection_layer: int = 0,
) -> DroidBlendPrefillResult:
    """Build a DroidBlend mixed cache with reference receiver KV.

    The method first builds DroidSpeak's layer-selective cache, then uses
    receiver full prefill to select and patch high-deviation tokens outside the
    fully recomputed DroidSpeak layer group.
    """
    model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    droidspeak = partial_prefill(receiver_model, sender_kv_cache, sender_e_cache, recompute_layers, input_ids, attention_mask)
    receiver_reference = receiver_prefill(receiver_model, model_inputs)
    receiver_kv = cache_to_layer_kv(receiver_reference.past_key_values)
    droidspeak_kv = cache_to_layer_kv(droidspeak.past_key_values)
    token_mask, selection = select_high_deviation_tokens(
        sender_kv_cache,
        receiver_kv,
        important_fraction=important_fraction,
        top_k=top_k,
        selection_layer=selection_layer,
    )
    mixed_kv = build_droidblend_layer_kv(droidspeak_kv, receiver_kv, token_mask, recompute_layers)
    mixed_cache = build_partial_cache(mixed_kv, sorted(mixed_kv))
    finalized = finalize_prompt_logits(receiver_model, mixed_cache, input_ids, attention_mask)
    return DroidBlendPrefillResult(
        past_key_values=finalized.past_key_values,
        next_token_logits=finalized.next_token_logits,
        selection=selection,
        recompute_layers=recompute_layers,
    )
