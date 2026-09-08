"""Independent token-anchor ablation prefill for DroidSpeak + CacheBlend.

This module intentionally does not modify or import the historical
``core.sparse_droidblend`` implementation.  It keeps a DroidSpeak contiguous
span dense for every valid prompt token, then sparsely recomputes the selected
tokens on the remaining receiver layers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb, eager_attention_forward

from core.partial_prefill import PartialPrefillResult, finalize_prompt_logits
from kv_cache.cache_utils import LayerKV, cache_to_layer_kv, clone_layer_kv
from kv_cache.layer_kv_injector import build_partial_cache


@dataclass
class TokenAnchorSelection:
    """Selection metadata for one prompt and one anchor/scorer configuration."""

    selection_anchor_layer: int
    selection_score_mode: str
    token_recompute_ratio: float
    selected_count: int
    candidate_count: int
    selected_indices: list[int]
    candidate_indices: list[int]
    score_k: list[float]
    score_v: list[float] | None


@dataclass
class TokenAnchorPrefillResult(PartialPrefillResult):
    selection: TokenAnchorSelection
    recompute_layers: tuple[int, int]
    full_recompute_layer_count: int
    sparse_recompute_layer_count: int
    recompute_work_ratio: float
    receiver_full_dependency: bool = False


def _layer_output(output):
    return output[0] if isinstance(output, (tuple, list)) else output


def _causal_mask(hidden_states: torch.Tensor) -> torch.Tensor:
    length = hidden_states.shape[1]
    mask = torch.zeros((1, 1, length, length), dtype=hidden_states.dtype, device=hidden_states.device)
    keys = torch.arange(length, device=hidden_states.device).view(1, 1, 1, length)
    queries = torch.arange(length, device=hidden_states.device).view(1, 1, length, 1)
    return mask.masked_fill(keys > queries, torch.finfo(hidden_states.dtype).min)


def _project_anchor_kv(model, layer_id: int, hidden_states: torch.Tensor, positions, include_value: bool) -> LayerKV:
    """Project receiver K, and V only when the scorer needs it.

    This is the full-token anchor calculation.  It is deliberately limited to
    the anchor projection and never invokes receiver full prefill.
    """
    layer = model.model.layers[layer_id]
    normed = layer.input_layernorm(hidden_states)
    shape = (*normed.shape[:-1], -1, layer.self_attn.head_dim)
    key = layer.self_attn.k_proj(normed).view(shape).transpose(1, 2)
    value = layer.self_attn.v_proj(normed).view(shape).transpose(1, 2) if include_value else None
    cos, sin = positions
    _, key = apply_rotary_pos_emb(torch.zeros_like(key), key, cos, sin)
    if value is None:
        return key, torch.empty(0, device=key.device, dtype=key.dtype)
    return key, value


def _project_sparse_qkv(attn, hidden_states: torch.Tensor, positions):
    shape = (*hidden_states.shape[:-1], -1, attn.head_dim)
    query = attn.q_proj(hidden_states).view(shape).transpose(1, 2)
    key = attn.k_proj(hidden_states).view(shape).transpose(1, 2)
    value = attn.v_proj(hidden_states).view(shape).transpose(1, 2)
    cos, sin = positions
    query, key = apply_rotary_pos_emb(query, key, cos, sin)
    return query, key, value


def _tail(layer: LayerKV, token_length: int) -> LayerKV:
    key, value = layer
    if key.shape[-2] <= token_length:
        return key, value
    return key[:, :, -token_length:, :].contiguous(), value[:, :, -token_length:, :].contiguous()


def _distance(receiver: torch.Tensor, sender: torch.Tensor) -> torch.Tensor:
    """Per-token mean-head relative L2 difference."""
    if receiver.shape != sender.shape:
        raise ValueError(f"anchor shapes differ: receiver={tuple(receiver.shape)}, sender={tuple(sender.shape)}")
    numerator = (receiver.float() - sender.float()).norm(p=2, dim=-1)
    denominator = sender.float().norm(p=2, dim=-1).clamp_min(1e-12)
    return (numerator / denominator).mean(dim=(0, 1))


def select_token_indices(
    sender_layer: LayerKV,
    receiver_key: torch.Tensor,
    receiver_value: torch.Tensor | None,
    candidate_mask: torch.Tensor,
    token_recompute_ratio: float,
    selection_anchor_layer: int,
    selection_score_mode: str,
) -> tuple[torch.Tensor, TokenAnchorSelection]:
    """Rank valid tokens, with stable position-order tie breaking."""
    if not 0 < token_recompute_ratio <= 1:
        raise ValueError("token_recompute_ratio must be in (0, 1]")
    if selection_score_mode not in {"k", "kv"}:
        raise ValueError("selection_score_mode must be 'k' or 'kv'")
    sender_key, sender_value = sender_layer
    score_k = _distance(receiver_key, sender_key)
    score_v = None
    if selection_score_mode == "kv":
        if receiver_value is None:
            raise ValueError("receiver value projection is required for selection_score_mode='kv'")
        score_v = _distance(receiver_value, sender_value)
        scores = 0.5 * (score_k + score_v)
    else:
        scores = score_k
    candidates = torch.nonzero(candidate_mask, as_tuple=False).flatten()
    if candidates.numel() == 0:
        raise ValueError("no valid tokens are available for selection")
    selected_count = min(candidates.numel(), max(1, math.ceil(candidates.numel() * token_recompute_ratio)))
    # torch.argsort(stable=True) preserves candidate index order for score ties.
    ranked = torch.argsort(scores[candidates], descending=True, stable=True)
    selected = candidates[ranked[:selected_count]].sort().values
    selection = TokenAnchorSelection(
        selection_anchor_layer=selection_anchor_layer,
        selection_score_mode=selection_score_mode,
        token_recompute_ratio=token_recompute_ratio,
        selected_count=int(selected_count),
        candidate_count=int(candidates.numel()),
        selected_indices=[int(index) for index in selected.detach().cpu().tolist()],
        candidate_indices=[int(index) for index in candidates.detach().cpu().tolist()],
        score_k=[float(score_k[index].item()) for index in candidates],
        score_v=None if score_v is None else [float(score_v[index].item()) for index in candidates],
    )
    return selected, selection


def _run_dense_layer(model, layer_id: int, hidden_states, attention_mask, cache, position_ids, positions):
    output = model.model.layers[layer_id](
        hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=cache,
        output_attentions=False,
        use_cache=True,
        cache_position=position_ids[0],
        position_embeddings=positions,
    )
    return _layer_output(output)


def _patch_rows(layer: LayerKV, row_key: torch.Tensor, row_value: torch.Tensor, indices: torch.Tensor) -> LayerKV:
    old_key, old_value = layer
    key, value = old_key.clone(), old_value.clone()
    key[:, :, indices, :] = row_key
    value[:, :, indices, :] = row_value
    return key, value


def _run_sparse_layer(model, layer_id: int, hidden_rows, cache_layer: LayerKV, token_indices, positions, row_position_indices):
    layer = model.model.layers[layer_id]
    residual = hidden_rows
    normed = layer.input_layernorm(hidden_rows)
    cos, sin = positions
    row_positions = (cos[:, row_position_indices, :], sin[:, row_position_indices, :])
    query, row_key, row_value = _project_sparse_qkv(layer.self_attn, normed, row_positions)
    key_cache, value_cache = _patch_rows(cache_layer, row_key, row_value, token_indices)
    full_length = key_cache.shape[-2]
    mask = torch.zeros((1, 1, token_indices.numel(), full_length), dtype=query.dtype, device=query.device)
    columns = torch.arange(full_length, device=query.device).view(1, 1, 1, full_length)
    rows = token_indices.view(1, 1, token_indices.numel(), 1)
    mask = mask.masked_fill(columns > rows, torch.finfo(query.dtype).min)
    attention, _ = eager_attention_forward(
        layer.self_attn, query, key_cache, value_cache, mask, dropout=0.0,
        scaling=layer.self_attn.scaling, sliding_window=getattr(model.config, "sliding_window", None),
    )
    attention = layer.self_attn.o_proj(attention.reshape(1, token_indices.numel(), -1).contiguous())
    hidden_rows = residual + attention
    return hidden_rows + layer.mlp(layer.post_attention_layernorm(hidden_rows)), (key_cache, value_cache)


def _normalise_span(recompute_layers: tuple[int, int], layer_count: int) -> tuple[int, int]:
    start, end = map(int, recompute_layers)
    if not 0 <= start <= end < layer_count:
        raise ValueError(f"recompute_layers must be within [0, {layer_count - 1}]")
    return start, end


@torch.inference_mode()
def token_anchor_sparse_prefill(
    receiver_model,
    sender_kv_cache: Mapping[int, LayerKV],
    sender_e_cache: Mapping[int, torch.Tensor],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    recompute_layers: tuple[int, int],
    selection_anchor_layer: int,
    selection_score_mode: str,
    token_recompute_ratio: float = 0.20,
) -> TokenAnchorPrefillResult:
    """Run one independent token-anchor ablation configuration.

    ``recompute_layers`` is the dense DroidSpeak span.  Outside that span only
    selected tokens receive receiver-layer recomputation and their K/V rows are
    merged into the sender cache.  No receiver full prefill is invoked.
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("token_anchor_sparse_prefill expects input_ids with shape [1, sequence]")
    layer_count = len(receiver_model.model.layers)
    start, end = _normalise_span(recompute_layers, layer_count)
    if selection_anchor_layer not in {0, start}:
        raise ValueError("selection_anchor_layer must be 0 or recompute_layers[0]")
    if selection_anchor_layer not in sender_kv_cache:
        raise ValueError("sender_kv_cache does not contain the selection anchor layer")

    embeddings = receiver_model.model.embed_tokens(input_ids)
    sequence_length = input_ids.shape[1]
    position_ids = torch.arange(sequence_length, device=input_ids.device).unsqueeze(0)
    positions = receiver_model.model.rotary_emb(embeddings, position_ids)
    anchor_hidden = embeddings if selection_anchor_layer == 0 else sender_e_cache[selection_anchor_layer].to(input_ids.device, receiver_model.dtype)
    cache_length = sender_kv_cache[selection_anchor_layer][0].shape[-2]
    offset = sequence_length - cache_length
    if offset < 0:
        raise ValueError("sender cache is longer than the input prompt")
    candidate_mask = torch.ones(cache_length, dtype=torch.bool, device=input_ids.device)
    if attention_mask is not None:
        candidate_mask = attention_mask[0, offset:].to(device=input_ids.device, dtype=torch.bool)

    receiver_key, receiver_value_or_empty = _project_anchor_kv(
        receiver_model, selection_anchor_layer, anchor_hidden, positions,
        include_value=selection_score_mode == "kv",
    )
    receiver_key = _tail((receiver_key, receiver_key), cache_length)[0]
    receiver_value = None
    if selection_score_mode == "kv":
        receiver_value = _tail((receiver_value_or_empty, receiver_value_or_empty), cache_length)[0]
    token_indices, selection = select_token_indices(
        sender_kv_cache[selection_anchor_layer], receiver_key, receiver_value, candidate_mask,
        token_recompute_ratio, selection_anchor_layer, selection_score_mode,
    )
    token_indices = token_indices.to(input_ids.device)
    prompt_token_indices = token_indices + offset

    mixed_kv = clone_layer_kv(sender_kv_cache)
    dense_cache = build_partial_cache(mixed_kv, [layer_id for layer_id in mixed_kv if not start <= layer_id <= end])
    dense_hidden = sender_e_cache[start].to(input_ids.device, receiver_model.dtype)
    dense_mask = _causal_mask(dense_hidden)
    for layer_id in range(start, end + 1):
        dense_hidden = _run_dense_layer(receiver_model, layer_id, dense_hidden, dense_mask, dense_cache, position_ids, positions)
    dense_layers = cache_to_layer_kv(dense_cache)
    mixed_kv.update({layer_id: _tail(dense_layers[layer_id], cache_length) for layer_id in range(start, end + 1)})

    sparse_hidden = embeddings[:, prompt_token_indices, :]
    for layer_id in range(start):
        sparse_hidden, mixed_kv[layer_id] = _run_sparse_layer(
            receiver_model, layer_id, sparse_hidden, mixed_kv[layer_id], token_indices, positions, prompt_token_indices,
        )
    # Dense DroidSpeak layers own the transition into the post-span sparse tail.
    sparse_hidden = dense_hidden[:, prompt_token_indices, :]
    for layer_id in range(end + 1, layer_count):
        sparse_hidden, mixed_kv[layer_id] = _run_sparse_layer(
            receiver_model, layer_id, sparse_hidden, mixed_kv[layer_id], token_indices, positions, prompt_token_indices,
        )

    finalized = finalize_prompt_logits(receiver_model, build_partial_cache(mixed_kv, sorted(mixed_kv)), input_ids, attention_mask)
    full_count = end - start + 1
    sparse_count = layer_count - full_count
    work_ratio = (selection.candidate_count * full_count + selection.selected_count * sparse_count) / (selection.candidate_count * layer_count)
    return TokenAnchorPrefillResult(
        past_key_values=finalized.past_key_values,
        next_token_logits=finalized.next_token_logits,
        selection=selection,
        recompute_layers=(start, end),
        full_recompute_layer_count=full_count,
        sparse_recompute_layer_count=sparse_count,
        recompute_work_ratio=work_ratio,
    )
