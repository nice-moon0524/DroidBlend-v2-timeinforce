import torch
from transformers import MistralConfig, MistralForCausalLM

from core.partial_prefill import greedy_decode, partial_prefill, single_layer_reuse_prefill
from experiments.common import cacheblend_prefill, full_reuse_prefill, receiver_prefill
from kv_cache.layer_kv_extractor import extract_layer_caches


class _TinyTokenizer:
    eos_token_id = 0

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(map(str, token_ids))


def test_experiment_methods_run_on_tiny_mistral():
    config = MistralConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    model = MistralForCausalLM(config).eval()
    tokenizer = _TinyTokenizer()
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    attention_mask = torch.ones_like(input_ids)
    model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

    sender_kv, sender_e = extract_layer_caches(model, **model_inputs)
    baseline = receiver_prefill(model, model_inputs)
    reuse = full_reuse_prefill(model, sender_kv, sender_e, input_ids, attention_mask)
    partial = partial_prefill(model, sender_kv, sender_e, (1, 2), input_ids, attention_mask)
    single = single_layer_reuse_prefill(model, sender_kv, sender_e, 1, input_ids, attention_mask)
    blended = cacheblend_prefill(model, sender_kv, baseline, 0.25, input_ids, attention_mask)

    assert baseline.next_token_logits.shape == (1, config.vocab_size)
    assert reuse.next_token_logits.shape == (1, config.vocab_size)
    assert partial.next_token_logits.shape == (1, config.vocab_size)
    assert single.next_token_logits.shape == (1, config.vocab_size)
    assert blended.next_token_logits.shape == (1, config.vocab_size)
    assert isinstance(greedy_decode(model, tokenizer, partial, max_new_tokens=2), str)


def test_cacheblend_handles_mismatched_cache_lengths():
    config = MistralConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    model = MistralForCausalLM(config).eval()
    sender_ids = torch.randint(0, config.vocab_size, (1, 8))
    receiver_ids = torch.randint(0, config.vocab_size, (1, 12))
    sender_mask = torch.ones_like(sender_ids)
    receiver_mask = torch.ones_like(receiver_ids)

    sender_kv, _ = extract_layer_caches(model, input_ids=sender_ids, attention_mask=sender_mask)
    baseline = receiver_prefill(model, {"input_ids": receiver_ids, "attention_mask": receiver_mask})
    blended = cacheblend_prefill(model, sender_kv, baseline, 0.25, sender_ids, sender_mask)

    assert blended.next_token_logits.shape == (1, config.vocab_size)
