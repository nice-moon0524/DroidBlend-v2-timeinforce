"""Compatibility helpers for the mutable HuggingFace DynamicCache API."""

from __future__ import annotations

from typing import Mapping

import torch
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer


LayerKV = tuple[torch.Tensor, torch.Tensor]


class PartialDynamicCache(DynamicCache):
    """DynamicCache that permits unpopulated decoder-layer slots.

    HuggingFace's stock DynamicCache assumes every layer is updated in order;
    partial prefill needs populated sender entries above and below a hole.
    """

    def __init__(self):
        Cache.__init__(self, layers=[])

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        while len(self.layers) <= layer_idx:
            self.layers.append(DynamicLayer())
        layer = self.layers[layer_idx]
        if not layer.is_initialized:
            layer.lazy_initialization(key_states)
            layer.keys = key_states
            layer.values = value_states
        else:
            layer.keys = torch.cat([layer.keys, key_states], dim=-2)
            layer.values = torch.cat([layer.values, value_states], dim=-2)
        return layer.keys, layer.values

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if len(self.layers) <= layer_idx:
            return 0
        layer = self.layers[layer_idx]
        if not layer.is_initialized or layer.keys is None:
            return 0
        return layer.keys.shape[-2]


def _layer_slot(cache: DynamicCache, layer_id: int):
    while len(cache.layers) <= layer_id:
        cache.layers.append(DynamicLayer())
    return cache.layers[layer_id]


def empty_cache() -> PartialDynamicCache:
    return PartialDynamicCache()


def set_layer(cache: DynamicCache, layer_id: int, key: torch.Tensor, value: torch.Tensor) -> None:
    """Set, rather than append, one layer's complete prompt cache."""
    layer = _layer_slot(cache, layer_id)
    layer.keys = key
    layer.values = value
    layer.dtype = key.dtype
    layer.device = key.device
    layer.is_initialized = True


def get_layer(cache: DynamicCache, layer_id: int) -> LayerKV | None:
    if len(cache.layers) <= layer_id:
        return None
    layer = cache.layers[layer_id]
    if not getattr(layer, "is_initialized", False):
        return None
    return layer.keys, layer.values


def inject_layers(cache: DynamicCache, layer_kv: Mapping[int, LayerKV], layer_ids: range | list[int]) -> DynamicCache:
    for layer_id in layer_ids:
        key, value = layer_kv[layer_id]
        set_layer(cache, layer_id, key, value)
    return cache


def clone_layer_kv(layer_kv: Mapping[int, LayerKV]) -> dict[int, LayerKV]:
    return {index: (key.clone(), value.clone()) for index, (key, value) in layer_kv.items()}


def cache_to_layer_kv(cache: DynamicCache) -> dict[int, LayerKV]:
    layer_kv: dict[int, LayerKV] = {}
    layers = getattr(cache, "layers", None)
    if layers is not None:
        for layer_id, layer in enumerate(layers):
            if getattr(layer, "is_initialized", False) and layer.keys is not None and layer.values is not None:
                layer_kv[layer_id] = (layer.keys, layer.values)
        return layer_kv
    key_cache = getattr(cache, "key_cache", None)
    value_cache = getattr(cache, "value_cache", None)
    if key_cache is not None and value_cache is not None:
        for layer_id, (key, value) in enumerate(zip(key_cache, value_cache, strict=False)):
            if key is not None and value is not None:
                layer_kv[layer_id] = (key, value)
    return layer_kv
