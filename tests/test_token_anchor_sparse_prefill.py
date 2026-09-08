"""CPU-only tests for independent token-anchor selection logic."""

from __future__ import annotations

import torch

from core.token_anchor_sparse_prefill import select_token_indices


def _layer(key_values: list[float], value_values: list[float]):
    key = torch.tensor(key_values, dtype=torch.float32).view(1, 1, -1, 1)
    value = torch.tensor(value_values, dtype=torch.float32).view(1, 1, -1, 1)
    return key, value


def test_k_only_excludes_padding_and_uses_stable_tie_breaking():
    sender = _layer([1, 1, 1, 1, 1], [1, 1, 1, 1, 1])
    receiver_key = torch.tensor([1, 3, 3, 10, 2], dtype=torch.float32).view(1, 1, -1, 1)
    selected, metadata = select_token_indices(
        sender, receiver_key, None, torch.tensor([True, True, True, False, True]),
        token_recompute_ratio=0.50, selection_anchor_layer=0, selection_score_mode="k",
    )
    assert selected.tolist() == [1, 2]
    assert metadata.candidate_indices == [0, 1, 2, 4]
    assert metadata.selected_count == 2
    assert metadata.candidate_count == 4
    assert metadata.score_v is None


def test_kv_can_change_the_k_only_ranking():
    sender = _layer([1, 1, 1], [1, 1, 1])
    receiver_key = torch.tensor([1, 3, 2], dtype=torch.float32).view(1, 1, -1, 1)
    receiver_value = torch.tensor([1, 1, 10], dtype=torch.float32).view(1, 1, -1, 1)
    candidates = torch.tensor([True, True, True])
    k_selected, _ = select_token_indices(sender, receiver_key, None, candidates, 1 / 3, 12, "k")
    kv_selected, metadata = select_token_indices(sender, receiver_key, receiver_value, candidates, 1 / 3, 12, "kv")
    assert k_selected.tolist() == [1]
    assert kv_selected.tolist() == [2]
    assert metadata.score_v is not None


def test_kv_matches_k_when_value_distances_are_equal():
    sender = _layer([1, 1, 1], [1, 1, 1])
    receiver_key = torch.tensor([1, 4, 2], dtype=torch.float32).view(1, 1, -1, 1)
    receiver_value = torch.tensor([2, 2, 2], dtype=torch.float32).view(1, 1, -1, 1)
    candidates = torch.tensor([True, True, True])
    k_selected, _ = select_token_indices(sender, receiver_key, None, candidates, 1 / 3, 0, "k")
    kv_selected, _ = select_token_indices(sender, receiver_key, receiver_value, candidates, 1 / 3, 0, "kv")
    assert k_selected.tolist() == kv_selected.tolist() == [1]
