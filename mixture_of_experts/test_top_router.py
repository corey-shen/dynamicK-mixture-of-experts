import torch
import pytest

from mixture_of_experts import Top2Gating

# NOTE: Compile via `pytest test_top_router.py`

def test_top_router_shape():
    gating = Top2Gating(dim=8, num_gates=16)
    logits = torch.randn(4,16)
    select_idx, select_probs = gating.top_router(logits, tau = 0.5, max_k=4)
    assert select_idx.shape == (4, 4)
    assert select_probs.shape == (4, 4)

    batch_size, seq_len, num_experts = 2, 3, 8
    max_k = 4
    tau = 0.8
    
    # mock logits
    logits = torch.randn(batch_size, seq_len, num_experts)
    
    probabilities = torch.softmax(logits, dim=-1)
    p_sorted, idx_sorted = torch.sort(probabilities, descending=True, dim=-1)
    
    cumulative_sum = torch.cumsum(p_sorted, dim=-1)
    mask_keep = (cumulative_sum < tau).type(torch.int32)
    mask_keep[..., 0] = 1
    k_star = mask_keep.sum(dim=-1).clamp(max=max_k)
    
    size_range = torch.arange(max_k, device=logits.device)
    select_the_mask = size_range[None, :].expand(*k_star.shape, max_k)
    select_the_mask = select_the_mask < k_star.unsqueeze(-1)
    
    pad_values = torch.full_like(idx_sorted[..., :max_k], -1)
    select_idx = torch.where(select_the_mask, idx_sorted[..., :max_k], pad_values)
    topk_all_probs = p_sorted[..., :max_k]
    pad_probs = torch.zeros_like(topk_all_probs)
    select_probs = torch.where(select_the_mask, topk_all_probs, pad_probs)
    
    renormalization = select_probs.sum(dim=-1, keepdim=True).clamp(min=0.0000001)
    select_probs = select_probs / renormalization
    
    # === TEST ASSERTIONS ===
    
    # Test output shapes
    assert select_idx.shape == (batch_size, seq_len, max_k), f"Expected select_idx shape {(batch_size, seq_len, max_k)}, got {select_idx.shape}"
    assert select_probs.shape == (batch_size, seq_len, max_k), f"Expected select_probs shape {(batch_size, seq_len, max_k)}, got {select_probs.shape}"
    
    # Test that probabilities are properly normalized (sum to 1 for each position)
    prob_sums = select_probs.sum(dim=-1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-6), "Selected probabilities should sum to 1"
    
    # Test that all probabilities are non-negative
    assert torch.all(select_probs >= 0), "All probabilities should be non-negative"
    
    # Test that selected indices are valid (either -1 for padding or valid expert indices)
    valid_indices = (select_idx >= 0) & (select_idx < num_experts)
    padding_indices = (select_idx == -1)
    assert torch.all(valid_indices | padding_indices), "Indices should be either valid expert indices or -1 for padding"
    
    # Test that k_star is within bounds
    assert torch.all(k_star >= 1), "k_star should be at least 1 (due to mask_keep[..., 0] = 1)"
    assert torch.all(k_star <= max_k), f"k_star should not exceed max_k={max_k}"
    
    # Test that the first position is always selected (due to mask_keep[..., 0] = 1)
    assert torch.all(select_idx[..., 0] >= 0), "First position should always be selected (not padded)"
    assert torch.all(select_probs[..., 0] > 0), "First position should always have non-zero probability"
    
    # Test that padding is consistent: if position i is padded, all positions j > i should also be padded
    for b in range(batch_size):
        for s in range(seq_len):
            padded_positions = (select_idx[b, s] == -1)
            if torch.any(padded_positions):
                first_pad_idx = torch.nonzero(padded_positions)[0].item()
                assert torch.all(select_idx[b, s, first_pad_idx:] == -1), "Padding should be contiguous from first padded position"
                assert torch.all(select_probs[b, s, first_pad_idx:] == 0), "Padded positions should have zero probability"
    
    # Test that selected indices are sorted by probability (highest first, ignoring padding)
    for b in range(batch_size):
        for s in range(seq_len):
            non_padded = select_idx[b, s] != -1
            if torch.sum(non_padded) > 1:  # Only test if we have multiple selected experts
                selected_probs = select_probs[b, s][non_padded]
                # Should be in descending order (with small tolerance for numerical precision)
                assert torch.all(selected_probs[:-1] >= selected_probs[1:] - 1e-6), "Selected probabilities should be in descending order"
    
    # Test cumulative probability constraint
    # The cumulative sum of original probabilities should respect the tau threshold
    for b in range(batch_size):
        for s in range(seq_len):
            k = k_star[b, s].item()
            cumsum_at_k = cumulative_sum[b, s, k-1].item()
            if k < num_experts:  # Only test if we didn't select all experts
                cumsum_at_k_plus_1 = cumulative_sum[b, s, k].item()
                # Either we hit max_k limit or the cumulative sum constraint
                assert k == max_k or cumsum_at_k_plus_1 >= tau, f"Should stop when cumsum >= tau or when k reaches max_k"
    
    # Test edge case: tau = 0 should select only 1 expert (the top one)
    logits_edge = torch.randn(2, 2, 5)
    probabilities_edge = torch.softmax(logits_edge, dim=-1)
    p_sorted_edge, idx_sorted_edge = torch.sort(probabilities_edge, descending=True, dim=-1)
    
    cumulative_sum_edge = torch.cumsum(p_sorted_edge, dim=-1)
    mask_keep_edge = (cumulative_sum_edge < 0.0).type(torch.int32)  # tau = 0
    mask_keep_edge[..., 0] = 1
    k_star_edge = mask_keep_edge.sum(dim=-1).clamp(max=max_k)
    
    assert torch.all(k_star_edge == 1), "With tau=0, should select exactly 1 expert"
    
    # Test edge case: tau = 1.0 should potentially select more experts
    mask_keep_full = (cumulative_sum < 1.0).type(torch.int32)
    mask_keep_full[..., 0] = 1
    k_star_full = mask_keep_full.sum(dim=-1).clamp(max=max_k)
    
    # Should select at least 1 expert (due to mask_keep[..., 0] = 1)
    assert torch.all(k_star_full >= 1), "Should always select at least 1 expert"
    
    print("All tests passed!")

# Additional specific test cases
def test_deterministic_case():
    """Test with a deterministic probability distribution"""
    # Create logits that will result in a clear probability ordering
    logits = torch.tensor([[[10.0, 5.0, 1.0, 0.1, 0.01]]])  # shape [1, 1, 5]
    tau = 0.95
    max_k = 3
    
    probabilities = torch.softmax(logits, dim=-1)
    p_sorted, idx_sorted = torch.sort(probabilities, descending=True, dim=-1)
    
    cumulative_sum = torch.cumsum(p_sorted, dim=-1)
    mask_keep = (cumulative_sum < tau).type(torch.int32)
    mask_keep[..., 0] = 1
    k_star = mask_keep.sum(dim=-1).clamp(max=max_k)
    
    size_range = torch.arange(max_k, device=logits.device)
    select_the_mask = size_range[None, :].expand(*k_star.shape, max_k)
    select_the_mask = select_the_mask < k_star.unsqueeze(-1)
    
    pad_values = torch.full_like(idx_sorted[..., :max_k], -1)
    select_idx = torch.where(select_the_mask, idx_sorted[..., :max_k], pad_values)
    topk_all_probs = p_sorted[..., :max_k]
    pad_probs = torch.zeros_like(topk_all_probs)
    select_probs = torch.where(select_the_mask, topk_all_probs, pad_probs)
    
    renormalization = select_probs.sum(dim=-1, keepdim=True).clamp(min=0.0000001)
    select_probs = select_probs / renormalization
    
    # The highest logit (10.0) corresponds to index 0, should be selected first
    assert select_idx[0, 0, 0] == 0, "Highest probability expert should be selected first"
    
    # Probabilities should sum to 1
    assert torch.allclose(select_probs.sum(dim=-1), torch.ones(1, 1)), "Probabilities should sum to 1"
    
    print("Deterministic case test passed!")

if __name__ == "__main__":
    test_top_router_shape()
    test_deterministic_case()