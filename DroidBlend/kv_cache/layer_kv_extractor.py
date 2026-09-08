"""Extract sender KV cache and hidden states at every possible transition."""

from __future__ import annotations

import torch

from kv_cache.cache_utils import LayerKV, cache_to_layer_kv


@torch.inference_mode()
def extract_layer_caches(model, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> tuple[dict[int, LayerKV], dict[int, torch.Tensor]]:
    """Return prompt KV and the input E cache for each decoder layer.

    E[layer] is `hidden_states[layer]`, namely the representation entering that
    zero-indexed decoder layer.  Tensors remain on the model device.
    """
    outputs = model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    past = outputs.past_key_values
    layer_kv = {layer_id: (key.detach(), value.detach()) for layer_id, (key, value) in cache_to_layer_kv(past).items()}
    e_cache = {layer_id: outputs.hidden_states[layer_id].detach() for layer_id in range(len(model.model.layers))}
    # This normalized final state is useful for the all-layer-reuse baseline.
    e_cache[len(model.model.layers)] = outputs.hidden_states[-1].detach()
    return layer_kv, e_cache
