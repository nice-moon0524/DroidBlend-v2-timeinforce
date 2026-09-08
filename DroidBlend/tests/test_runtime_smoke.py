import unittest


try:
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    AVAILABLE = True
except ImportError:  # pragma: no cover
    AVAILABLE = False


@unittest.skipUnless(AVAILABLE, "torch and transformers are required for runtime smoke tests")
class RuntimeSmokeTests(unittest.TestCase):
    def _model(self):
        config = LlamaConfig(
            vocab_size=97,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=64,
        )
        return LlamaForCausalLM(config).eval()

    def test_sender_e_and_kv_can_drive_receiver_partial_prefill(self):
        from droidblend.runtime import (
            HybridPrefill,
            append_suffix_and_generate,
            capture_layer_inputs,
            dynamic_cache_from_legacy,
            legacy_kv_from_cache,
            partial_prefill,
        )
        from droidblend.schedules import LayerGroup

        sender, receiver = self._model(), self._model()
        prefix_ids = torch.tensor([[1, 2, 3, 4, 5]])
        prefix_mask = torch.ones_like(prefix_ids)
        with capture_layer_inputs(sender, [1]) as e_caches:
            sender_output = sender(input_ids=prefix_ids, attention_mask=prefix_mask, use_cache=True, return_dict=True)
        sender_kv = legacy_kv_from_cache(sender_output.past_key_values)
        prefill = partial_prefill(receiver, sender_kv, e_caches[1], prefix_mask, LayerGroup(1, 3), "cpu")
        hybrid_kv = legacy_kv_from_cache(prefill.cache)
        self.assertEqual(len(hybrid_kv), 4)
        self.assertTrue(all(key.shape[-2] == prefix_ids.shape[1] for key, _ in hybrid_kv))

        _, tokens, _ = append_suffix_and_generate(
            receiver,
            prefill,
            torch.tensor([[6]]),
            torch.ones((1, 6), dtype=torch.long),
            2,
        )
        self.assertGreaterEqual(len(tokens), 1)

        # The same adapter also supports the single-layer swap sensitivity path.
        cache = dynamic_cache_from_legacy(receiver, sender_kv, "cpu")
        _, swapped_tokens, _ = append_suffix_and_generate(
            receiver,
            HybridPrefill(cache=cache, last_hidden_state=None, elapsed_ms=0.0),
            torch.tensor([[6]]),
            torch.ones((1, 6), dtype=torch.long),
            1,
        )
        self.assertGreaterEqual(len(swapped_tokens), 1)


if __name__ == "__main__":
    unittest.main()
