"""Minimal, explicit runtime for faithful DroidSpeak-style KV reuse.

The implementation deliberately uses the receiver's *real* decoder blocks for
the selected continuous group.  It does not approximate a block with a linear
projection or silently fall back to native prefill: an unsupported Transformers
layout raises an error instead.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from .errors import CompatibilityError
from .schedules import LayerGroup


def torch_module() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise CompatibilityError("torch is required for model execution") from exc
    return torch


def dtype_from_name(name: str) -> Any:
    torch = torch_module()
    try:
        return getattr(torch, name)
    except AttributeError as exc:
        raise CompatibilityError(f"Unknown torch dtype: {name}") from exc


def load_causal_lm(path: str, dtype: str, device: str, trust_remote_code: bool, attention_implementation: str) -> Any:
    try:
        import transformers
        from transformers import AutoModelForCausalLM
    except ImportError as exc:  # pragma: no cover
        raise CompatibilityError("transformers is required for model execution") from exc
    # Transformers 5 renamed ``torch_dtype`` to ``dtype``; retain 4.x support
    # because many research clusters pin a 4.x version for FlashAttention.
    dtype_key = "dtype" if int(transformers.__version__.split(".", 1)[0]) >= 5 else "torch_dtype"
    kwargs: dict[str, Any] = {dtype_key: dtype_from_name(dtype), "trust_remote_code": trust_remote_code}
    if attention_implementation:
        kwargs["attn_implementation"] = attention_implementation
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, **kwargs)
    return model.to(device).eval()


def load_tokenizer(path: str, trust_remote_code: bool) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise CompatibilityError("transformers is required for model execution") from exc
    return AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code, use_fast=True, local_files_only=True)


def legacy_kv_from_cache(cache: Any) -> list[tuple[Any, Any]]:
    """Detach a Transformers Cache into portable layer-indexed (K,V) tensors."""
    if cache is None:
        raise CompatibilityError("Sender forward pass did not return past_key_values")
    if hasattr(cache, "to_legacy_cache"):
        raw = cache.to_legacy_cache()
        return [(key.detach().cpu(), value.detach().cpu()) for key, value in raw]
    if hasattr(cache, "layers"):
        pairs = []
        for layer in cache.layers:
            key, value = getattr(layer, "keys", None), getattr(layer, "values", None)
            if key is None or value is None:
                raise CompatibilityError("Sender cache contains an uninitialized decoder layer")
            pairs.append((key.detach().cpu(), value.detach().cpu()))
        return pairs
    try:
        return [(key.detach().cpu(), value.detach().cpu()) for key, value in cache]
    except TypeError as exc:
        raise CompatibilityError(f"Unsupported Transformers cache type: {type(cache)!r}") from exc


def _base_model(causal_lm: Any) -> Any:
    base = getattr(causal_lm, "model", None)
    if base is None or not all(hasattr(base, attr) for attr in ("layers", "embed_tokens", "norm", "rotary_emb")):
        raise CompatibilityError(
            "DroidBlend currently supports Llama/Mistral-style causal LMs exposing "
            "model.layers, embed_tokens, norm and rotary_emb."
        )
    return base


@contextmanager
def capture_layer_inputs(causal_lm: Any, capture_layers: list[int]) -> Iterator[dict[int, Any]]:
    """Capture sender E^l: input activation to every requested decoder layer."""
    base = _base_model(causal_lm)
    captured: dict[int, Any] = {}
    handles = []

    def hook(layer: int):
        def record(_module: Any, args: tuple[Any, ...]) -> None:
            if not args:
                raise CompatibilityError(f"Could not capture sender E cache at layer {layer}")
            captured[layer] = args[0].detach().cpu().clone()
        return record

    for layer in capture_layers:
        if layer < 0 or layer >= len(base.layers):
            raise CompatibilityError(f"Requested E cache layer {layer} outside model range")
        handles.append(base.layers[layer].register_forward_pre_hook(hook(layer)))
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def _dynamic_cache_with_sender_kv(receiver: Any, sender_kv: list[tuple[Any, Any]], group: LayerGroup, device: str) -> Any:
    try:
        from transformers.cache_utils import DynamicCache
    except ImportError as exc:  # pragma: no cover
        raise CompatibilityError("This Transformers version lacks DynamicCache") from exc
    base = _base_model(receiver)
    if len(sender_kv) != len(base.layers):
        raise CompatibilityError(
            f"Sender cache has {len(sender_kv)} layers but receiver has {len(base.layers)} layers"
        )
    try:
        cache = DynamicCache(config=receiver.config)
    except TypeError:  # Transformers 4.x did not accept config in the constructor.
        cache = DynamicCache()
    # Insert only skipped layers. Recomputed layers must begin with an empty
    # cache, otherwise DynamicCache would append B's keys to A's old keys.
    for layer_idx, (key, value) in enumerate(sender_kv):
        if group.start <= layer_idx < group.end:
            continue
        cache.update(key.to(device), value.to(device), layer_idx, {})
    return cache


def dynamic_cache_from_legacy(receiver: Any, legacy_kv: list[tuple[Any, Any]], device: str) -> Any:
    """Build a receiver cache from portable K/V tensors (for swap sensitivity)."""
    try:
        from transformers.cache_utils import DynamicCache
    except ImportError as exc:  # pragma: no cover
        raise CompatibilityError("This Transformers version lacks DynamicCache") from exc
    moved = [(key.to(device), value.to(device)) for key, value in legacy_kv]
    if hasattr(DynamicCache, "from_legacy_cache"):
        return DynamicCache.from_legacy_cache(tuple(moved))
    try:
        return DynamicCache(ddp_cache_data=moved, config=receiver.config)
    except TypeError:
        try:
            return DynamicCache(ddp_cache_data=moved)
        except TypeError:
            cache = DynamicCache()
            for layer_idx, (key, value) in enumerate(moved):
                cache.update(key, value, layer_idx, {})
            return cache


def _causal_mask(base: Any, hidden: Any, attention_mask: Any, cache_position: Any, cache: Any, position_ids: Any) -> Any:
    # Transformers 4.x used LlamaModel._update_causal_mask; 5.x centralised it.
    if hasattr(base, "_update_causal_mask"):
        return base._update_causal_mask(attention_mask, hidden, cache_position, cache, False)
    try:
        from transformers.masking_utils import create_causal_mask
    except ImportError:
        try:
            from transformers.models.llama.modeling_llama import create_causal_mask
        except ImportError as exc:  # pragma: no cover
            raise CompatibilityError("Cannot construct a causal mask for this Transformers version") from exc
    return create_causal_mask(
        config=base.config,
        input_embeds=hidden,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=cache,
        position_ids=position_ids,
    )


@dataclass
class HybridPrefill:
    cache: Any
    last_hidden_state: Any
    elapsed_ms: float


def partial_prefill(receiver: Any, sender_kv: list[tuple[Any, Any]], sender_e_at_start: Any, attention_mask: Any, group: LayerGroup, device: str) -> HybridPrefill:
    """Run B exactly on ``group`` and reuse A K/V on every other layer.

    ``sender_e_at_start`` is A's activation *entering* the first B layer.  This
    is the essential DroidSpeak handoff; it is not the final hidden state.
    """
    torch = torch_module()
    # CUDA kernels are asynchronous.  This is the actual local hybrid-prefill
    # latency (cache materialisation plus B's selected blocks), not merely the
    # CPU-side kernel-launch overhead.
    synchronize(device)
    started = time.perf_counter()
    base = _base_model(receiver)
    if group.end > len(base.layers):
        raise CompatibilityError(f"{group.identifier} exceeds receiver depth {len(base.layers)}")
    cache = _dynamic_cache_with_sender_kv(receiver, sender_kv, group, device)
    hidden = sender_e_at_start.to(device=device, dtype=next(receiver.parameters()).dtype)
    mask = attention_mask.to(device)
    seq_len = hidden.shape[1]
    cache_position = torch.arange(seq_len, device=device)
    position_ids = cache_position.unsqueeze(0)
    # The hybrid cache contains full-prefix A K/V at skipped layers.  A generic
    # mask builder would interpret those as *past* tokens and create a 2N mask
    # for the N-token recomputation.  The selected B group instead recomputes
    # the whole prefix from E_A[s], so its attention has zero past length.
    try:
        from transformers.cache_utils import DynamicCache
        try:
            mask_cache = DynamicCache(config=receiver.config)
        except TypeError:  # Transformers 4.x
            mask_cache = DynamicCache()
    except ImportError as exc:  # pragma: no cover
        raise CompatibilityError("This Transformers version lacks DynamicCache") from exc
    causal_mask = _causal_mask(base, hidden, mask, cache_position, mask_cache, position_ids)
    position_embeddings = base.rotary_emb(hidden, position_ids=position_ids)

    with torch.inference_mode():
        for layer_idx in range(group.start, group.end):
            layer_output = base.layers[layer_idx](
                hidden,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                cache_position=cache_position,
            )
            hidden = layer_output[0] if isinstance(layer_output, (tuple, list)) else layer_output
    final_hidden = base.norm(hidden)
    synchronize(device)
    return HybridPrefill(cache=cache, last_hidden_state=final_hidden, elapsed_ms=(time.perf_counter() - started) * 1000)


def append_suffix_and_generate(receiver: Any, prefill: HybridPrefill, suffix_ids: Any, suffix_attention_mask: Any, max_new_tokens: int) -> tuple[str, list[int], float]:
    """Finish prompt and greedily decode. All suffix/decode blocks run in B."""
    torch = torch_module()
    device = next(receiver.parameters()).device
    generated: list[int] = []
    started = time.perf_counter()
    with torch.inference_mode():
        outputs = receiver(
            input_ids=suffix_ids.to(device),
            attention_mask=suffix_attention_mask.to(device),
            past_key_values=prefill.cache,
            use_cache=True,
            return_dict=True,
        )
        cache = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
        eos = getattr(receiver.generation_config, "eos_token_id", getattr(receiver.config, "eos_token_id", None))
        terminators = {int(item) for item in eos} if isinstance(eos, (tuple, list, set)) else ({int(eos)} if eos is not None else set())
        for _ in range(max_new_tokens):
            next_id = int(torch.argmax(logits, dim=-1).item())
            generated.append(next_id)
            if next_id in terminators:
                break
            outputs = receiver(
                input_ids=torch.tensor([[next_id]], device=device),
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
    return "", generated, (time.perf_counter() - started) * 1000


def tensor_nbytes(kv: list[tuple[Any, Any]]) -> int:
    return sum(key.numel() * key.element_size() + value.numel() * value.element_size() for key, value in kv)


def synchronize(device: str) -> None:
    """Make CUDA latency measurements honest while remaining CPU-safe."""
    torch = torch_module()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)
