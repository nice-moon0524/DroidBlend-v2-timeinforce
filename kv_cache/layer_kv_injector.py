"""Inject selected completed layers into an otherwise-empty prompt cache."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from kv_cache.cache_utils import LayerKV, empty_cache, inject_layers


def build_partial_cache(sender_kv_cache: Mapping[int, LayerKV], inject_layer_ids: Iterable[int]):
    cache = empty_cache()
    return inject_layers(cache, sender_kv_cache, list(inject_layer_ids))
