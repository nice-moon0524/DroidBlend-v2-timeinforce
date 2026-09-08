"""Sparse DroidBlend prefill without a receiver full-KV oracle."""

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
class SparseTokenSelection:
    """Per-example metadata for anchor-based token selection."""

    selection_anchor_layer: int
    selection_score_mode: str
    token_recompute_ratio: float
    selected_count: int
    candidate_count: int
    indices: list[int]
    candidate_indices: list[int]

    @property
    def check_layer(self) -> int:
        """Compatibility alias for historical results."""
        return self.selection_anchor_layer


@dataclass
class SparseDroidBlendResult(PartialPrefillResult):
    selection: SparseTokenSelection
    recompute_layers: tuple[int, int]
    span_mode: str
    full_recompute_layer_count: int
    recompute_work_ratio: float
    receiver_full_dependency: bool = False
    reference_mode: bool = False

    @property
    def dense_layer_count(self) -> int:
        return self.full_recompute_layer_count


def _hidden_from_layer_output(layer_output):
    return layer_output[0] if isinstance(layer_output, (tuple, list)) else layer_output


def _patch_rows(base: LayerKV, row_k: torch.Tensor, row_v: torch.Tensor, indices: torch.Tensor) -> LayerKV:
    key, value = base
    patched_key = key.clone()
    patched_value = value.clone()
    patched_key[:, :, indices, :] = row_k
    patched_value[:, :, indices, :] = row_v
    return patched_key, patched_value


def _project_qkv(attn, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor]):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, attn.head_dim)
    query_states = attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    return query_states, key_states, value_states


def _project_anchor_kv(model, layer_id: int, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor]) -> LayerKV:
    """Compute only K/V projections; attention and MLP are intentionally skipped."""
    layer = model.model.layers[layer_id]
    normed = layer.input_layernorm(hidden_states)
    _, key_states, value_states = _project_qkv(layer.self_attn, normed, position_embeddings)
    return key_states, value_states


def _full_causal_mask(hidden_states: torch.Tensor) -> torch.Tensor:
    seq_len = hidden_states.shape[1]
    mask = torch.zeros((1, 1, seq_len, seq_len), dtype=hidden_states.dtype, device=hidden_states.device)
    return mask.masked_fill(
        torch.arange(seq_len, device=hidden_states.device).view(1, 1, 1, seq_len)
        > torch.arange(seq_len, device=hidden_states.device).view(1, 1, seq_len, 1),
        torch.finfo(hidden_states.dtype).min,
    )


def _run_full_layer(model, layer_id: int, hidden_states: torch.Tensor, attention_mask: torch.Tensor, cache, position_ids: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor]):
    output = model.model.layers[layer_id](
        hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=cache,
        output_attentions=False,
        use_cache=True,
        cache_position=position_ids[0],
        position_embeddings=position_embeddings,
    )
    return _hidden_from_layer_output(output)


def _run_full_layer_with_projected_kv(
    model,
    layer_id: int,
    hidden_states: torch.Tensor,
    projected_kv: LayerKV,
    attention_mask: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Run an anchor block while reusing selection-time K/V projections."""
    layer = model.model.layers[layer_id]
    residual = hidden_states
    normed = layer.input_layernorm(hidden_states)
    input_shape = normed.shape[:-1]
    query = layer.self_attn.q_proj(normed).view(*input_shape, -1, layer.self_attn.head_dim).transpose(1, 2)
    cos, sin = position_embeddings
    query, _ = apply_rotary_pos_emb(query, torch.zeros_like(query), cos, sin)
    key_states, value_states = projected_kv
    attn_output, _ = eager_attention_forward(
        layer.self_attn, query, key_states, value_states, attention_mask, dropout=0.0,
        scaling=layer.self_attn.scaling, sliding_window=getattr(model.config, "sliding_window", None),
    )
    attn_output = layer.self_attn.o_proj(attn_output.reshape(1, hidden_states.shape[1], -1).contiguous())
    hidden_states = residual + attn_output
    return hidden_states + layer.mlp(layer.post_attention_layernorm(hidden_states))

def _run_sparse_layer(
    model,
    layer_id: int,
    hidden_rows: torch.Tensor,
    old_kv: LayerKV,
    token_indices: torch.Tensor,
    position_token_indices: torch.Tensor,
    full_position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, LayerKV]:
    layer = model.model.layers[layer_id]
    residual = hidden_rows
    normed = layer.input_layernorm(hidden_rows)
    cos, sin = full_position_embeddings
    row_positions = (cos[:, position_token_indices, :], sin[:, position_token_indices, :])
    q, k_rows, v_rows = _project_qkv(layer.self_attn, normed, row_positions)
    key_cache, value_cache = _patch_rows(old_kv, k_rows, v_rows, token_indices)
    full_len = key_cache.shape[-2]
    mask = torch.zeros((1, 1, token_indices.numel(), full_len), dtype=q.dtype, device=q.device)
    columns = torch.arange(full_len, device=q.device).view(1, 1, 1, full_len)
    rows = token_indices.view(1, 1, token_indices.numel(), 1)
    mask = mask.masked_fill(columns > rows, torch.finfo(q.dtype).min)
    attn_output, _ = eager_attention_forward(
        layer.self_attn, q, key_cache, value_cache, mask, dropout=0.0,
        scaling=layer.self_attn.scaling, sliding_window=getattr(model.config, "sliding_window", None),
    )
    attn_output = layer.self_attn.o_proj(attn_output.reshape(1, token_indices.numel(), -1).contiguous())
    hidden_rows = residual + attn_output
    return hidden_rows + layer.mlp(layer.post_attention_layernorm(hidden_rows)), (key_cache, value_cache)


def _score_distance(receiver: torch.Tensor, sender: torch.Tensor) -> torch.Tensor:
    """mean_h(||receiver-sender||_2 / (||sender||_2 + 1e-12)) per token."""
    if receiver.shape != sender.shape:
        raise ValueError(f"sender/receiver anchor KV shapes differ: {tuple(sender.shape)} != {tuple(receiver.shape)}")
    numerator = (receiver.float() - sender.float()).norm(p=2, dim=-1)
    denominator = sender.float().norm(p=2, dim=-1).clamp_min(1e-12)
    return (numerator / denominator).mean(dim=(0, 1))


def _select_tokens(
    sender_layer: LayerKV,
    receiver_layer: LayerKV,
    token_recompute_ratio: float,
    selection_anchor_layer: int,
    selection_score_mode: str,
    candidate_mask: torch.Tensor,
) -> tuple[torch.Tensor, SparseTokenSelection]:
    if not 0 < token_recompute_ratio <= 1:
        raise ValueError("token_recompute_ratio must be in (0, 1]")
    if selection_score_mode not in {"k", "kv"}:
        raise ValueError("selection_score_mode must be 'k' or 'kv'")
    sender_key, sender_value = sender_layer
    receiver_key, receiver_value = receiver_layer
    score_k = _score_distance(receiver_key, sender_key)
    scores = score_k if selection_score_mode == "k" else 0.5 * (score_k + _score_distance(receiver_value, sender_value))
    candidates = torch.nonzero(candidate_mask, as_tuple=False).flatten()
    if candidates.numel() == 0:
        raise ValueError("token selection has no valid non-padding candidates")
    selected_count = min(candidates.numel(), max(1, math.ceil(candidates.numel() * token_recompute_ratio)))
    ranked = torch.argsort(scores[candidates], descending=True, stable=True)
    indices = candidates[ranked[:selected_count]].sort().values
    return indices, SparseTokenSelection(
        selection_anchor_layer=selection_anchor_layer,
        selection_score_mode=selection_score_mode,
        token_recompute_ratio=token_recompute_ratio,
        selected_count=int(selected_count),
        candidate_count=int(candidates.numel()),
        indices=[int(item) for item in indices.detach().cpu().tolist()],
        candidate_indices=[int(item) for item in candidates.detach().cpu().tolist()],
    )


def _tail_crop_layer(layer: LayerKV, token_length: int) -> LayerKV:
    key, value = layer
    if key.shape[-2] <= token_length:
        return key, value
    return key[:, :, -token_length:, :].contiguous(), value[:, :, -token_length:, :].contiguous()


def _normalize_recompute_layers(recompute_layers: tuple[int, int] | None, dense_layer_count: int | None, num_layers: int) -> tuple[int, int]:
    if recompute_layers is None:
        count = 1 if dense_layer_count is None else int(dense_layer_count)
        recompute_layers = (0, count - 1)
    start, end = int(recompute_layers[0]), int(recompute_layers[1])
    if not 0 <= start <= end < num_layers:
        raise ValueError(f"recompute_layers must be an inclusive range within [0, {num_layers - 1}]")
    return start, end


@torch.inference_mode()
def sparse_droidblend_prefill(
    receiver_model,
    sender_kv_cache: Mapping[int, LayerKV],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    dense_layer_count: int | None = 1,
    token_recompute_ratio: float = 0.15,
    sender_e_cache: Mapping[int, torch.Tensor] | None = None,
    recompute_layers: tuple[int, int] | None = None,
    selection_anchor_layer: int = 0,
    selection_score_mode: str = "k",
    span_mode: str = "contiguous",
) -> SparseDroidBlendResult:
    """Build mixed KV using only sender caches and receiver local computation."""
    if span_mode != "contiguous":
        raise NotImplementedError("span_mode='disjoint' is reserved for a later experiment phase")
    num_layers = len(receiver_model.model.layers)
    start, end = _normalize_recompute_layers(recompute_layers, dense_layer_count, num_layers)
    if selection_anchor_layer not in {0, start}:
        raise ValueError("selection_anchor_layer must be 0 or recompute_layers[0]")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("sparse_droidblend_prefill expects input_ids shape [1, sequence]")

    embed_hidden = receiver_model.model.embed_tokens(input_ids)
    seq_len = input_ids.shape[1]
    positions = torch.arange(seq_len, device=input_ids.device)
    position_ids = positions.unsqueeze(0)
    position_embeddings = receiver_model.model.rotary_emb(embed_hidden, position_ids)
    full_mask = _full_causal_mask(embed_hidden)
    cache_length = sender_kv_cache[selection_anchor_layer][0].shape[-2]
    offset = seq_len - cache_length
    if offset < 0:
        raise ValueError("sender KV is longer than input_ids")

    if selection_anchor_layer == 0:
        anchor_hidden = embed_hidden
    else:
        if sender_e_cache is None or selection_anchor_layer not in sender_e_cache:
            raise ValueError("sender_e_cache must contain selection_anchor_layer")
        anchor_hidden = sender_e_cache[selection_anchor_layer].to(device=input_ids.device, dtype=receiver_model.dtype)
    receiver_anchor_full = _project_anchor_kv(receiver_model, selection_anchor_layer, anchor_hidden, position_embeddings)
    receiver_anchor = _tail_crop_layer(receiver_anchor_full, cache_length)
    candidate_mask = torch.ones(cache_length, dtype=torch.bool, device=input_ids.device)
    if attention_mask is not None:
        candidate_mask = attention_mask[0, offset:].to(device=input_ids.device, dtype=torch.bool)
    token_indices, selection = _select_tokens(
        sender_kv_cache[selection_anchor_layer], receiver_anchor, token_recompute_ratio,
        selection_anchor_layer, selection_score_mode, candidate_mask,
    )
    token_indices = token_indices.to(device=input_ids.device)
    position_token_indices = token_indices + offset

    mixed_kv = clone_layer_kv(sender_kv_cache)
    sparse_hidden = embed_hidden[:, position_token_indices, :]
    for layer_id in range(start):
        sparse_hidden, mixed_kv[layer_id] = _run_sparse_layer(
            receiver_model, layer_id, sparse_hidden, mixed_kv[layer_id], token_indices,
            position_token_indices, position_embeddings,
        )

    injected_layers = [layer_id for layer_id in mixed_kv if not start <= layer_id <= end]
    span_cache = build_partial_cache(mixed_kv, injected_layers)
    if start == 0:
        span_hidden = embed_hidden
    else:
        if sender_e_cache is None:
            raise ValueError("sender_e_cache is required when recompute_layers starts after layer 0")
        span_hidden = sender_e_cache[start].to(device=input_ids.device, dtype=receiver_model.dtype)
    span_kv: dict[int, LayerKV] = {}
    for layer_id in range(start, end + 1):
        if layer_id == selection_anchor_layer:
            span_hidden = _run_full_layer_with_projected_kv(
                receiver_model, layer_id, span_hidden, receiver_anchor_full, full_mask, position_embeddings
            )
            span_kv[layer_id] = receiver_anchor_full
        else:
            span_hidden = _run_full_layer(receiver_model, layer_id, span_hidden, full_mask, span_cache, position_ids, position_embeddings)
    cached_span_kv = cache_to_layer_kv(span_cache)
    for layer_id in range(start, end + 1):
        if layer_id not in span_kv:
            span_kv[layer_id] = cached_span_kv[layer_id]
    mixed_kv.update({layer_id: _tail_crop_layer(span_kv[layer_id], cache_length) for layer_id in range(start, end + 1)})
    sparse_hidden = span_hidden[:, position_token_indices, :]

    for layer_id in range(end + 1, num_layers):
        sparse_hidden, mixed_kv[layer_id] = _run_sparse_layer(
            receiver_model, layer_id, sparse_hidden, mixed_kv[layer_id], token_indices,
            position_token_indices, position_embeddings,
        )

    finalized = finalize_prompt_logits(receiver_model, build_partial_cache(mixed_kv, sorted(mixed_kv)), input_ids, attention_mask)
    full_count = end - start + 1
    work_ratio = selection.selected_count * full_count / (selection.candidate_count * num_layers)
    return SparseDroidBlendResult(
        past_key_values=finalized.past_key_values,
        next_token_logits=finalized.next_token_logits,
        selection=selection,
        recompute_layers=(start, end),
        span_mode=span_mode,
        full_recompute_layer_count=full_count,
        recompute_work_ratio=work_ratio,
    )

