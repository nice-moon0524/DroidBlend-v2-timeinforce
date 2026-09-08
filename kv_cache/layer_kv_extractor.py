"""Extract sender KV cache and optional hidden-state transition caches."""

from __future__ import annotations

import torch

from kv_cache.cache_utils import LayerKV, cache_to_layer_kv


@torch.inference_mode()
def extract_layer_caches(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    include_e_cache: bool = True,
) -> tuple[dict[int, LayerKV], dict[int, torch.Tensor]]:
    """Return prompt KV and optionally the input E cache for each decoder layer.

    ``include_e_cache=False`` is recommended for token-only sweeps because the
    all-layer hidden-state copy can be large for long prompts.
    """
    outputs = model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        output_hidden_states=include_e_cache,
        return_dict=True,
    )
    past = outputs.past_key_values
    layer_kv = {
        layer_id: (key.detach(), value.detach())
        for layer_id, (key, value) in cache_to_layer_kv(past).items()
    }
    if not include_e_cache:
        return layer_kv, {}
    e_cache = {
        layer_id: outputs.hidden_states[layer_id].detach()
        for layer_id in range(len(model.model.layers))
    }
    e_cache[len(model.model.layers)] = outputs.hidden_states[-1].detach()
    return layer_kv, e_cache
