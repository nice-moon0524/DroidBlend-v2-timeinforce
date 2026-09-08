"""Layer-selective KV reuse for same-architecture Mistral checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

from kv_cache.cache_utils import LayerKV, cache_to_layer_kv
from kv_cache.layer_kv_injector import build_partial_cache


@dataclass
class PartialPrefillResult:
    """Prompt cache plus next-token logits, ready for incremental decode."""

    past_key_values: object
    next_token_logits: torch.Tensor


def _causal_mask(model, attention_mask: torch.Tensor | None, inputs_embeds: torch.Tensor, cache_position: torch.Tensor, past_key_values, position_ids: torch.Tensor):
    """Use the same causal-mask path as the current Mistral forward implementation."""
    mask_function = create_causal_mask if model.model.config.sliding_window is None else create_sliding_window_causal_mask
    return mask_function(
        config=model.model.config,
        input_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )


def _hidden_from_layer_output(layer_output):
    return layer_output[0] if isinstance(layer_output, (tuple, list)) else layer_output


@torch.inference_mode()
def finalize_prompt_logits(receiver_model, cache, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> PartialPrefillResult:
    """Evaluate the final prompt token against a cache for its preceding tokens.

    KV reuse alone does not contain the last-layer hidden state required for
    the first decode logit. Replaying *one* final prompt token through the
    receiver provides it and restores the complete cache. This is O(layers),
    not O(prompt_length * layers), and is included in measured prefill time.
    """
    prefix_layers = {
        layer_id: (key[:, :, :-1, :].contiguous(), value[:, :, :-1, :].contiguous())
        for layer_id, (key, value) in cache_to_layer_kv(cache).items()
    }
    prefix_cache = build_partial_cache(prefix_layers, list(prefix_layers))
    outputs = receiver_model(
        input_ids=input_ids[:, -1:],
        attention_mask=attention_mask,
        past_key_values=prefix_cache,
        use_cache=True,
        return_dict=True,
    )
    return PartialPrefillResult(outputs.past_key_values, outputs.logits[:, -1, :].float())


@torch.inference_mode()
def partial_prefill(
    receiver_model,
    sender_kv_cache: Mapping[int, LayerKV],
    sender_e_cache: Mapping[int, torch.Tensor],
    recompute_layers: tuple[int, int],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> PartialPrefillResult:
    """Build the receiver prompt cache by recomputing one contiguous layer group.

    The function intentionally operates on decoder layers instead of calling
    `receiver_model.forward`: this lets layers outside `[start, end]` retain
    sender cache entries.  The two checkpoints must have identical Mistral v0.1
    dimensions and cache layout.
    """
    start, end = recompute_layers
    layers = receiver_model.model.layers
    if not (0 <= start <= end < len(layers)):
        raise ValueError(f"Invalid recompute range {(start, end)} for {len(layers)} layers")
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")

    reused = list(range(0, start)) + list(range(end + 1, len(layers)))
    cache = build_partial_cache(sender_kv_cache, reused)
    # Sender E is already the state entering `start`; embedding lookup is not needed.
    hidden_states = sender_e_cache[start].to(device=input_ids.device, dtype=receiver_model.dtype)
    sequence_length = hidden_states.shape[1]
    cache_position = torch.arange(sequence_length, device=hidden_states.device)
    position_ids = cache_position.unsqueeze(0)
    causal_mask = _causal_mask(receiver_model, attention_mask, hidden_states, cache_position, cache, position_ids)
    position_embeddings = receiver_model.model.rotary_emb(hidden_states, position_ids)

    for layer_id in range(start, end + 1):
        layer_output = layers[layer_id](
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=cache,
            output_attentions=False,
            use_cache=True,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = _hidden_from_layer_output(layer_output)

    return finalize_prompt_logits(receiver_model, cache, input_ids, attention_mask)


@torch.inference_mode()
def greedy_decode(receiver_model, receiver_tokenizer, prefill: PartialPrefillResult, max_new_tokens: int = 64) -> str:
    """Greedily decode from a prompt cache without re-prefilling the prompt."""
    cache = prefill.past_key_values
    logits = prefill.next_token_logits
    generated: list[int] = []
    eos_token_id = receiver_tokenizer.eos_token_id
    for _ in range(max_new_tokens):
        token = int(torch.argmax(logits, dim=-1).item())
        if token == eos_token_id:
            break
        generated.append(token)
        step = torch.tensor([[token]], device=logits.device)
        outputs = receiver_model(input_ids=step, past_key_values=cache, use_cache=True, return_dict=True)
        cache = outputs.past_key_values
        logits = outputs.logits[:, -1, :].float()
    return receiver_tokenizer.decode(generated, skip_special_tokens=True).strip()


@torch.inference_mode()
def full_prefill_and_decode(receiver_model, receiver_tokenizer, model_inputs: dict[str, torch.Tensor], max_new_tokens: int = 64) -> str:
    """HF baseline used for quality comparison."""
    generated = receiver_model.generate(**model_inputs, do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=receiver_tokenizer.eos_token_id)
    prompt_length = model_inputs["input_ids"].shape[1]
    return receiver_tokenizer.decode(generated[0, prompt_length:], skip_special_tokens=True).strip()


@torch.inference_mode()
def single_layer_reuse_prefill(receiver_model, sender_kv_cache: Mapping[int, LayerKV], sender_e_cache: Mapping[int, torch.Tensor], layer_id: int, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> PartialPrefillResult:
    """Normal receiver prefill except for one sender-KV layer.

    The sender E cache immediately after the reused layer provides the valid
    transition input for the following receiver layer.
    """
    layers = receiver_model.model.layers
    if not 0 <= layer_id < len(layers):
        raise ValueError(f"Invalid layer {layer_id}")
    cache = build_partial_cache(sender_kv_cache, [layer_id])
    hidden_states = receiver_model.model.embed_tokens(input_ids)
    sequence_length = input_ids.shape[1]
    cache_position = torch.arange(sequence_length, device=input_ids.device)
    position_ids = cache_position.unsqueeze(0)
    causal_mask = _causal_mask(receiver_model, attention_mask, hidden_states, cache_position, cache, position_ids)
    position_embeddings = receiver_model.model.rotary_emb(hidden_states, position_ids)
    for current_id, layer in enumerate(layers):
        if current_id == layer_id:
            hidden_states = sender_e_cache[layer_id + 1].to(input_ids.device, receiver_model.dtype)
            continue
        output = layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=cache,
            output_attentions=False,
            use_cache=True,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = _hidden_from_layer_output(output)
    return finalize_prompt_logits(receiver_model, cache, input_ids, attention_mask)
