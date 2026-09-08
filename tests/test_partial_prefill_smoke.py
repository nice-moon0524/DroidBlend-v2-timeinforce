import torch
from transformers import MistralConfig, MistralForCausalLM

from core.partial_prefill import partial_prefill
from kv_cache.layer_kv_extractor import extract_layer_caches


def test_partial_prefill_runs_on_tiny_mistral():
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
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    attention_mask = torch.ones_like(input_ids)

    full_logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    ).logits[:, -1, :].float()
    sender_kv, sender_e = extract_layer_caches(model, input_ids=input_ids, attention_mask=attention_mask)
    result = partial_prefill(model, sender_kv, sender_e, (1, 2), input_ids, attention_mask)

    assert len(sender_kv) == config.num_hidden_layers
    assert len(sender_e) == config.num_hidden_layers + 1
    assert result.next_token_logits.shape == (1, config.vocab_size)
    assert result.past_key_values.get_seq_length(0) == input_ids.shape[1]
    assert torch.allclose(result.next_token_logits, full_logits, atol=1e-5)
