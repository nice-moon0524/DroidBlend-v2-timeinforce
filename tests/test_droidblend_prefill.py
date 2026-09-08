import torch
from transformers import MistralConfig, MistralForCausalLM

from core.droidblend_prefill import (
    build_droidblend_layer_kv,
    droidblend_reference_prefill,
    select_high_deviation_tokens,
)
from core.partial_prefill import greedy_decode, partial_prefill
from kv_cache.cache_utils import cache_to_layer_kv
from kv_cache.layer_kv_extractor import extract_layer_caches


class _TinyTokenizer:
    eos_token_id = 0

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(map(str, token_ids))


def _tiny_mistral():
    config = MistralConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    return MistralForCausalLM(config).eval()


def test_select_high_deviation_tokens_uses_first_layer_kv_distance():
    sender = {
        0: (
            torch.zeros(1, 2, 5, 4),
            torch.zeros(1, 2, 5, 4),
        )
    }
    receiver = {
        0: (
            torch.zeros(1, 2, 5, 4),
            torch.zeros(1, 2, 5, 4),
        )
    }
    receiver[0][0][:, :, 3, :] = 10
    receiver[0][1][:, :, 1, :] = 8

    mask, selection = select_high_deviation_tokens(sender, receiver, important_fraction=0.4)

    assert selection.selected_count == 2
    assert selection.candidate_count == 5
    assert set(selection.indices) == {1, 3}
    assert mask.tolist() == [False, True, False, True, False]


def test_droidblend_reference_prefill_runs_on_tiny_mistral():
    model = _tiny_mistral()
    tokenizer = _TinyTokenizer()
    input_ids = torch.randint(0, model.config.vocab_size, (1, 8))
    attention_mask = torch.ones_like(input_ids)

    sender_kv, sender_e = extract_layer_caches(model, input_ids=input_ids, attention_mask=attention_mask)
    result = droidblend_reference_prefill(
        model,
        sender_kv,
        sender_e,
        (1, 2),
        input_ids,
        attention_mask,
        important_fraction=0.25,
    )

    assert result.next_token_logits.shape == (1, model.config.vocab_size)
    assert result.selection.selected_count == 2
    assert result.selection.candidate_count == input_ids.shape[1]
    assert result.past_key_values.get_seq_length(0) == input_ids.shape[1]
    assert isinstance(greedy_decode(model, tokenizer, result, max_new_tokens=2), str)


def test_token_patch_skips_droidspeak_recomputed_layers():
    model = _tiny_mistral()
    input_ids = torch.randint(0, model.config.vocab_size, (1, 8))
    attention_mask = torch.ones_like(input_ids)
    sender_kv, sender_e = extract_layer_caches(model, input_ids=input_ids, attention_mask=attention_mask)
    droidspeak = partial_prefill(model, sender_kv, sender_e, (1, 2), input_ids, attention_mask)
    receiver_kv = cache_to_layer_kv(model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True).past_key_values)
    droidspeak_kv = cache_to_layer_kv(droidspeak.past_key_values)
    mask = torch.zeros(input_ids.shape[1], dtype=torch.bool)
    mask[0] = True

    mixed = build_droidblend_layer_kv(droidspeak_kv, receiver_kv, mask, (1, 2))

    assert torch.equal(mixed[1][0], droidspeak_kv[1][0])
    assert torch.equal(mixed[2][1], droidspeak_kv[2][1])


def test_sparse_droidblend_accepts_nonzero_recompute_span():
    from core.sparse_droidblend import sparse_droidblend_prefill

    model = _tiny_mistral()
    input_ids = torch.randint(0, model.config.vocab_size, (1, 8))
    attention_mask = torch.ones_like(input_ids)
    sender_kv, sender_e = extract_layer_caches(model, input_ids=input_ids, attention_mask=attention_mask)

    result = sparse_droidblend_prefill(
        model,
        sender_kv,
        input_ids=input_ids,
        attention_mask=attention_mask,
        sender_e_cache=sender_e,
        recompute_layers=(2, 3),
        token_recompute_ratio=0.25,
    )

    assert result.next_token_logits.shape == (1, model.config.vocab_size)
    assert result.recompute_layers == (2, 3)
    assert result.full_recompute_layer_count == 3
    assert result.selection.check_layer == 0
    assert result.selection.selected_count == 2
