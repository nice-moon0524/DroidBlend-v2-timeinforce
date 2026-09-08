import torch

from kv_cache.cache_utils import cache_to_layer_kv
from kv_cache.layer_kv_injector import build_partial_cache


def test_partial_cache_supports_sparse_layers():
    layer_kv = {
        0: (torch.ones(1, 2, 3, 4), torch.zeros(1, 2, 3, 4)),
        2: (torch.full((1, 2, 3, 4), 2.0), torch.full((1, 2, 3, 4), 3.0)),
    }

    cache = build_partial_cache(layer_kv, [0, 2])
    extracted = cache_to_layer_kv(cache)

    assert cache.get_seq_length(0) == 3
    assert cache.get_seq_length(1) == 0
    assert cache.get_seq_length(2) == 3
    assert sorted(extracted) == [0, 2]
    assert torch.equal(extracted[2][0], layer_kv[2][0])
