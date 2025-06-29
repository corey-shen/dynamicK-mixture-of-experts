import torch
from torch import nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import math
from inspect import isfunction
import os

# constants
MIN_EXPERT_CAPACITY = 4

# helper functions

def default(val, default_val):
    default_val = default_val() if isfunction(default_val) else default_val
    return val if val is not None else default_val

def cast_tuple(el):
    return el if isinstance(el, tuple) else (el,)

# tensor related helper functions

def top1(t):
    values, index = t.topk(k=1, dim=-1)
    values, index = map(lambda x: x.squeeze(dim=-1), (values, index))
    return values, index

def cumsum_exclusive(t, dim=-1):
    num_dims = len(t.shape)
    num_pad_dims = - dim - 1
    pre_padding = (0, 0) * num_pad_dims
    pre_slice   = (slice(None),) * num_pad_dims
    padded_t = F.pad(t, (*pre_padding, 1, 0)).cumsum(dim=dim)
    return padded_t[(..., slice(None, -1), *pre_slice)]

# pytorch one hot throws an error if there are out of bound indices.
# tensorflow, in contrast, does not throw an error
def safe_one_hot(indexes, max_length):
    max_index = indexes.max() + 1
    return F.one_hot(indexes, max(max_index + 1, max_length))[..., :max_length]

def init_(t):
    dim = t.shape[-1]
    std = 1 / math.sqrt(dim)
    return t.uniform_(-std, std)

# activations

class GELU_(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_

# expert class

class Experts(nn.Module):
    def __init__(self,
        dim,
        num_experts = 16,
        hidden_dim = None,
        activation = GELU):
        super().__init__()

        hidden_dim = default(hidden_dim, dim * 4)
        num_experts = cast_tuple(num_experts)

        w1 = torch.zeros(*num_experts, dim, hidden_dim)
        w2 = torch.zeros(*num_experts, hidden_dim, dim)

        w1 = init_(w1)
        w2 = init_(w2)

        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        self.act = activation()

    def forward(self, x):
        hidden = torch.einsum('...nd,...dh->...nh', x, self.w1)
        hidden = self.act(hidden)
        out    = torch.einsum('...nh,...hd->...nd', hidden, self.w2)
        return out

# the below code is almost all transcribed from the official tensorflow version, from which the papers are written
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/research/moe.py

# gating network

class DynamicKGating(nn.Module):
    def __init__(
        self,
        dim,
        num_gates,
        eps = 1e-9,
        outer_expert_dims = tuple(),
        second_policy_train = 'random',
        second_policy_eval = 'random',
        second_threshold_train = 0.2,
        second_threshold_eval = 0.2,
        capacity_factor_train = 1.25,
        capacity_factor_eval = 2.,
        tau = 0.7,
        max_k = 8
        ):
        super().__init__()

        self.eps = eps
        self.num_gates = num_gates
        self.w_gating = nn.Parameter(torch.randn(*outer_expert_dims, dim, num_gates))

        self.second_policy_train = second_policy_train
        self.second_policy_eval = second_policy_eval
        self.second_threshold_train = second_threshold_train
        self.second_threshold_eval = second_threshold_eval
        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval
        self.tau = tau
        self.max_k = max_k

    def forward(self, x, importance=None):
        *_, b, group_size, dim = x.shape
        num_gates = self.num_gates

        if self.training:
            capacity_factor = self.capacity_factor_train
        else:
            capacity_factor = self.capacity_factor_eval

        raw_gates = torch.einsum('...bnd,...de->...bne', x, self.w_gating)
        
        # USE DYNAMIC-K ROUTING
        selected_experts, selected_probs = self.top_router(raw_gates, self.tau, self.max_k)
        
        # Calculate how many tokens each expert can handle
        expert_capacity = min(group_size, int((group_size * capacity_factor) / num_gates))
        expert_capacity = max(expert_capacity, MIN_EXPERT_CAPACITY)
        
        # Initialize result tensors
        combine_tensor = torch.zeros(*x.shape[:-1], num_gates, expert_capacity, device=x.device)
        dispatch_tensor = torch.zeros_like(combine_tensor)
        
        # Track current position for each expert across all tokens
        expert_positions = torch.zeros(num_gates, dtype=torch.long, device=x.device)
        
        # Process each token in the batch
        for batch_idx in range(b):
            for seq_idx in range(group_size):
                # Get the experts and probabilities for this token
                token_experts = selected_experts[batch_idx, seq_idx]  # [max_k]
                token_probs = selected_probs[batch_idx, seq_idx]     # [max_k]
                
                # Process each selected expert for this token
                for k in range(self.max_k):
                    expert_id = token_experts[k].item()
                    prob = token_probs[k].item()
                    
                    # Skip padded entries
                    if expert_id < 0:
                        continue
                    
                    # Check if this expert has capacity
                    current_pos = expert_positions[expert_id].item()
                    if current_pos < expert_capacity:
                        # Assign this token to this expert at the current position
                        dispatch_tensor[batch_idx, seq_idx, expert_id, current_pos] = 1.0
                        combine_tensor[batch_idx, seq_idx, expert_id, current_pos] = prob
                        
                        # Increment the position for this expert
                        expert_positions[expert_id] += 1
        
        # Calculate load balancing loss
        expert_usage = dispatch_tensor.sum(dim=(0, 1, -1))  # [num_gates] - total tokens per expert
        if expert_usage.sum() > 0:
            # Variance of expert usage
            mean_usage = expert_usage.float().mean()
            loss = ((expert_usage.float() - mean_usage) ** 2).mean() / (mean_usage + 1e-8)
        else:
            loss = torch.tensor(0.0, device=x.device)
        
        return dispatch_tensor, combine_tensor, loss
    
    def top_router(self, logits, tau, max_k=4): 
        probabilities = torch.softmax(logits, dim=-1)
        p_sorted, idx_sorted = torch.sort(probabilities, descending=True, dim=-1)

        cumulative_sum = torch.cumsum(p_sorted, dim=-1)

        #we want to create a boolean mask to figure out the true "crossing" point!
        mask_keep = (cumulative_sum < tau).type(torch.int32)
        mask_keep[..., 0] = 1 #this ellipsis notation represents all the leading dimensions 
        k_star = mask_keep.sum(dim = -1).clamp(max = max_k)

        #need to start building up a matrix up to our max_k value in our function
        size_range = torch.arange(max_k, device=logits.device)
        select_the_mask = size_range[None, :].expand(*k_star.shape, max_k)
        #unsqueeze(-1) adds a size 1 dimension to the end of the tensor btw
        select_the_mask = select_the_mask < k_star.unsqueeze(-1)

        #at this point we pad our values as needed, specifically when select_the_mask is False
        pad_values = torch.full_like(idx_sorted[..., :max_k], -1)
        select_idx = torch.where(select_the_mask, idx_sorted[..., :max_k], pad_values)
        # select_probs = torch.where(select_the_mask, p_sorted[..., max_k], torch.zeros(1, dtype=logits.dtype, device=logits.device))
        '''
        The line above only selects the single probability at index max_k, not the first max_k probability
        '''
        topk_all_probs = p_sorted[..., :max_k]                     # shape [..., max_k] | Creates a tensor of the top max_k probabilities
        pad_probs = torch.zeros_like(topk_all_probs)              # shape [..., max_k]  | Create a tensor of zeros (same shape)
        select_probs = torch.where(select_the_mask, topk_all_probs, pad_probs)  # Apply mask - keep topk where True, pad where False

        renormalization = select_probs.sum(dim=-1, keepdim=True).clamp(min = 0.0000001)
        select_probs = select_probs / renormalization

        return select_idx, select_probs

class DynamicMoE(nn.Module):
    def __init__(self,
        dim,
        num_experts = 16,
        hidden_dim = None,
        activation = nn.ReLU,
        capacity_factor_train = 1.25,
        capacity_factor_eval = 2.,
        loss_coef = 1e-2,
        experts = None,
        tau = 0.95,
        max_k = 4):
        super().__init__()

        self.num_experts = num_experts

        # Use the new Dynamic-K Gating
        self.gate = DynamicKGating(
            dim, 
            num_gates=num_experts, 
            capacity_factor_train=capacity_factor_train,
            capacity_factor_eval=capacity_factor_eval,
            tau=tau,
            max_k=max_k
        )
        
        self.experts = default(experts, lambda: Experts(dim, num_experts = num_experts, hidden_dim = hidden_dim, activation = activation))
        self.loss_coef = loss_coef

    def forward(self, inputs, **kwargs):
        b, n, d, e = *inputs.shape, self.num_experts
        dispatch_tensor, combine_tensor, loss = self.gate(inputs)
        expert_inputs = torch.einsum('bnd,bnec->ebcd', inputs, dispatch_tensor)

        # Now feed the expert inputs through the experts.
        orig_shape = expert_inputs.shape
        expert_inputs = expert_inputs.reshape(e, -1, d)
        expert_outputs = self.experts(expert_inputs)
        expert_outputs = expert_outputs.reshape(*orig_shape)

        output = torch.einsum('ebcd,bnec->bnd', expert_outputs, combine_tensor)
        return output, loss * self.loss_coef


# plain mixture of experts

class MoE(nn.Module):
    def __init__(self,
        dim,
        num_experts = 16,
        hidden_dim = None,
        activation = nn.ReLU,
        second_policy_train = 'random',
        second_policy_eval = 'random',
        second_threshold_train = 0.2,
        second_threshold_eval = 0.2,
        capacity_factor_train = 1.25,
        capacity_factor_eval = 2.,
        loss_coef = 1e-2,
        experts = None):
        super().__init__()

        self.num_experts = num_experts

        gating_kwargs = {'second_policy_train': second_policy_train, 'second_policy_eval': second_policy_eval, 'second_threshold_train': second_threshold_train, 'second_threshold_eval': second_threshold_eval, 'capacity_factor_train': capacity_factor_train, 'capacity_factor_eval': capacity_factor_eval}
        self.gate = DynamicKGating(dim, num_gates = num_experts, **gating_kwargs)
        self.experts = default(experts, lambda: Experts(dim, num_experts = num_experts, hidden_dim = hidden_dim, activation = activation))
        self.loss_coef = loss_coef

    def forward(self, inputs, **kwargs):
        b, n, d, e = *inputs.shape, self.num_experts
        dispatch_tensor, combine_tensor, loss = self.gate(inputs)
        expert_inputs = torch.einsum('bnd,bnec->ebcd', inputs, dispatch_tensor)

        # Now feed the expert inputs through the experts.
        orig_shape = expert_inputs.shape
        expert_inputs = expert_inputs.reshape(e, -1, d)
        expert_outputs = self.experts(expert_inputs)
        expert_outputs = expert_outputs.reshape(*orig_shape)

        output = torch.einsum('ebcd,bnec->bnd', expert_outputs, combine_tensor)
        return output, loss * self.loss_coef

# 2-level heirarchical mixture of experts

class HeirarchicalMoE(nn.Module):
    def __init__(self,
        dim,
        num_experts = (4, 4),
        hidden_dim = None,
        activation = nn.ReLU,
        second_policy_train = 'random',
        second_policy_eval = 'random',
        second_threshold_train = 0.2,
        second_threshold_eval = 0.2,
        capacity_factor_train = 1.25,
        capacity_factor_eval = 2.,
        loss_coef = 1e-2,
        experts = None):
        super().__init__()

        assert len(num_experts) == 2, 'only 2 levels of heirarchy for experts allowed for now'
        num_experts_outer, num_experts_inner = num_experts
        self.num_experts_outer = num_experts_outer
        self.num_experts_inner = num_experts_inner

        gating_kwargs = {'second_policy_train': second_policy_train, 'second_policy_eval': second_policy_eval, 'second_threshold_train': second_threshold_train, 'second_threshold_eval': second_threshold_eval, 'capacity_factor_train': capacity_factor_train, 'capacity_factor_eval': capacity_factor_eval}

        self.gate_outer = DynamicKGating(dim, num_gates = num_experts_outer, **gating_kwargs)
        self.gate_inner = DynamicKGating(dim, num_gates = num_experts_inner, outer_expert_dims = (num_experts_outer,), **gating_kwargs)

        self.experts = default(experts, lambda: Experts(dim, num_experts = num_experts, hidden_dim = hidden_dim, activation = activation))
        self.loss_coef = loss_coef

    def forward(self, inputs, **kwargs):
        b, n, d, eo, ei = *inputs.shape, self.num_experts_outer, self.num_experts_inner
        dispatch_tensor_outer, combine_tensor_outer, loss_outer = self.gate_outer(inputs)
        expert_inputs_outer = torch.einsum('bnd,bnec->ebcd', inputs, dispatch_tensor_outer)

        # we construct an "importance" Tensor for the inputs to the second-level
        # gating.  The importance of an input is 1.0 if it represents the
        # first-choice expert-group and 0.5 if it represents the second-choice expert
        # group.  This is used by the second-level gating.
        importance = combine_tensor_outer.permute(2, 0, 3, 1).sum(dim=-1)
        importance = 0.5 * ((importance > 0.5).float() + (importance > 0.).float())

        dispatch_tensor_inner, combine_tensor_inner, loss_inner = self.gate_inner(expert_inputs_outer, importance = importance)
        expert_inputs = torch.einsum('ebnd,ebnfc->efbcd', expert_inputs_outer, dispatch_tensor_inner)

        # Now feed the expert inputs through the experts.
        orig_shape = expert_inputs.shape
        expert_inputs = expert_inputs.reshape(eo, ei, -1, d)
        expert_outputs = self.experts(expert_inputs)
        expert_outputs = expert_outputs.reshape(*orig_shape)

        # NOW COMBINE EXPERT OUTPUTS (reversing everything we have done)
        # expert_output has shape [y0, x1, h, d, n]

        expert_outputs_outer = torch.einsum('efbcd,ebnfc->ebnd', expert_outputs, combine_tensor_inner)
        output = torch.einsum('ebcd,bnec->bnd', expert_outputs_outer, combine_tensor_outer)
        return output, (loss_outer + loss_inner) * self.loss_coef
    

def test_current_code():
    """Test that the current code runs without the dynamic-k changes"""
    print("Testing current implementation...")
    
    try:
        # Test DynamicMoE (though it's not actually using dynamic-k yet)
        print("Testing Forward Pass")
        model = DynamicMoE(dim=128, num_experts=8)
        x = torch.randn(2, 10, 128)
        print(f"   Input shape: {x.shape}")
        output, loss = model(x) # forward pass is implemented here
        
        print(f"   ✓ Forward pass successful!")
        print(f"   Output shape: {output.shape}")
        print(f"   Loss: {loss.item():.6f}")
        
        # Test that top_router function exists and works
        print("Testing top_router function")
        gate = model.gate
        test_logits = torch.randn(2, 10, 8)  # batch=2, seq=10, experts=8
        
        selected_experts, selected_probs = gate.top_router(test_logits, tau=0.8, max_k=3)
        
        print(f"   Selected experts shape: {selected_experts.shape}")
        print(f"   Selected probs shape: {selected_probs.shape}")
        print(f"   Sample selected experts: {selected_experts[0, 0, :]}")  # First position
        print(f"   Sample selected probs: {selected_probs[0, 0, :]}")
        
        print("\nAll tests passed! Ready to implement dynamic-k changes.")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please fix the current code before implementing dynamic-k changes.")
        return False

import torch
import matplotlib.pyplot as plt
import numpy as np

def test_dynamic_k_behavior():
    """Test that dynamic-k actually varies the number of experts selected"""
    print("🧪 Testing Dynamic-K Behavior...")
    
    model = DynamicMoE(dim=128, num_experts=8, tau=0.7, max_k=6)
    x = torch.randn(4, 20, 128)  # Larger batch for better statistics
    
    # Hook into the gating to capture routing decisions
    routing_stats = []
    
    def capture_routing(module, input, output):
        dispatch_tensor, combine_tensor, loss = output
        # Count how many experts each token uses
        experts_per_token = (dispatch_tensor.sum(dim=-1) > 0).sum(dim=-1)  # [batch, seq]
        routing_stats.extend(experts_per_token.flatten().tolist())
    
    # Register hook
    handle = model.gate.register_forward_hook(capture_routing)
    
    # Run forward pass
    output, loss = model(x)
    
    # Remove hook
    handle.remove()
    
    # Analyze results
    routing_stats = np.array(routing_stats)
    unique_counts, frequencies = np.unique(routing_stats, return_counts=True)
    
    print(f"   Expert usage distribution:")
    for count, freq in zip(unique_counts, frequencies):
        percentage = (freq / len(routing_stats)) * 100
        print(f"   {int(count)} experts: {freq:3d} tokens ({percentage:5.1f}%)")
    
    print(f"   Average experts per token: {routing_stats.mean():.2f}")
    print(f"   Min/Max experts per token: {routing_stats.min()}/{routing_stats.max()}")
    
    # Test passes if we see variation in expert counts
    has_variation = len(unique_counts) > 1
    print(f"   ✓ Dynamic behavior: {'YES' if has_variation else 'NO'}")
    return has_variation

def test_tau_sensitivity():
    """Test that different tau values affect expert selection"""
    print("\n🎯 Testing Tau Sensitivity...")
    
    tau_values = [0.5, 0.7, 0.8, 0.9, 0.95]
    x = torch.randn(2, 10, 128)
    
    results = {}
    
    for tau in tau_values:
        model = DynamicMoE(dim=128, num_experts=8, tau=tau, max_k=8)
        
        # Get routing info directly from top_router
        raw_gates = torch.randn(2, 10, 8)  # Simulate gate logits
        selected_experts, selected_probs = model.gate.top_router(raw_gates, tau, max_k=8)
        
        # Count valid experts (not -1)
        valid_experts = (selected_experts >= 0).sum(dim=-1).float()
        avg_experts = valid_experts.mean().item()
        
        results[tau] = avg_experts
        print(f"   τ={tau:4.2f}: {avg_experts:.2f} experts per token on average")
    
    # Check that lower tau = more experts
    tau_sorted = sorted(results.items())
    is_monotonic = all(tau_sorted[i][1] >= tau_sorted[i+1][1] for i in range(len(tau_sorted)-1))
    print(f"   ✓ Lower τ → more experts: {'YES' if is_monotonic else 'NO'}")
    
    return is_monotonic

def test_vs_traditional_moe():
    """Compare Dynamic-K vs traditional top-2 MoE"""
    print("\n⚖️  Comparing Dynamic-K vs Traditional MoE...")
    
    x = torch.randn(4, 15, 128)
    
    # Traditional MoE (should always use exactly 2 experts)
    traditional_moe = MoE(dim=128, num_experts=8)
    
    # Dynamic-K MoE 
    dynamic_moe = DynamicMoE(dim=128, num_experts=8, tau=0.8, max_k=4)
    
    # Test traditional MoE
    output_trad, loss_trad = traditional_moe(x)
    
    # Test dynamic MoE
    output_dyn, loss_dyn = dynamic_moe(x)
    
    print(f"   Traditional MoE loss: {loss_trad.item():.6f}")
    print(f"   Dynamic-K MoE loss:  {loss_dyn.item():.6f}")
    print(f"   Output shapes match: {output_trad.shape == output_dyn.shape}")
    
    # Both should produce valid outputs
    both_valid = (torch.isfinite(output_trad).all() and torch.isfinite(output_dyn).all())
    print(f"   ✓ Both produce valid outputs: {'YES' if both_valid else 'NO'}")
    
    return both_valid

def test_expert_load_balancing():
    """Test that experts are being used somewhat evenly"""
    print("\n⚖️  Testing Expert Load Balancing...")
    
    model = DynamicMoE(dim=128, num_experts=8, tau=0.85, max_k=3)
    x = torch.randn(8, 25, 128)  # More tokens for better statistics
    
    # Track expert usage
    expert_usage = torch.zeros(8)
    
    def track_expert_usage(module, input, output):
        dispatch_tensor, combine_tensor, loss = output
        # Count tokens sent to each expert
        tokens_per_expert = dispatch_tensor.sum(dim=(0, 1, -1))  # Sum over batch, seq, capacity
        expert_usage.add_(tokens_per_expert)
    
    handle = model.gate.register_forward_hook(track_expert_usage)
    output, loss = model(x)
    handle.remove()
    
    print(f"   Expert usage: {expert_usage.tolist()}")
    
    # Check if usage is reasonably balanced (no expert gets >50% of tokens)
    max_usage = expert_usage.max().item()
    total_usage = expert_usage.sum().item()
    max_percentage = (max_usage / total_usage) * 100 if total_usage > 0 else 0
    
    is_balanced = max_percentage < 50  # No single expert dominates
    print(f"   Max expert usage: {max_percentage:.1f}% of all tokens")
    print(f"   ✓ Reasonably balanced: {'YES' if is_balanced else 'NO'}")
    
    return is_balanced

def test_capacity_constraints():
    """Test that expert capacity constraints are respected"""
    print("\n🏗️  Testing Capacity Constraints...")
    
    model = DynamicMoE(dim=64, num_experts=4, tau=0.6, max_k=3)  # Low tau = more experts
    x = torch.randn(2, 8, 64)  # Small input
    
    output, loss = model(x)
    
    # Get the dispatch tensor to check capacity
    dispatch_tensor, combine_tensor, gate_loss = model.gate(x)
    
    # Check that no expert receives more tokens than its capacity
    tokens_per_expert_per_slot = dispatch_tensor.sum(dim=(0, 1))  # [num_experts, expert_capacity]
    max_tokens_per_slot = tokens_per_expert_per_slot.max(dim=0)[0]  # Max tokens in any slot
    
    capacity_respected = (max_tokens_per_slot <= 1.0).all()  # Each slot should have at most 1 token
    
    print(f"   Dispatch tensor shape: {dispatch_tensor.shape}")
    print(f"   Max tokens per slot: {max_tokens_per_slot.max().item():.3f}")
    print(f"   ✓ Capacity constraints respected: {'YES' if capacity_respected else 'NO'}")
    
    return capacity_respected

def visualize_routing_pattern():
    """Visualize how tokens are routed to experts"""
    print("\n📊 Visualizing Routing Patterns...")
    
    model = DynamicMoE(dim=32, num_experts=6, tau=0.8, max_k=4)
    x = torch.randn(1, 12, 32)  # Single batch for easy visualization
    
    # Get routing information
    dispatch_tensor, combine_tensor, loss = model.gate(x)
    
    # Create routing matrix: [seq_pos, expert] = probability
    routing_matrix = combine_tensor[0].sum(dim=-1)  # Sum over capacity dim
    
    plt.figure(figsize=(10, 6))
    plt.imshow(routing_matrix.T.detach().numpy(), aspect='auto', cmap='Blues')
    plt.colorbar(label='Routing Probability')
    plt.xlabel('Sequence Position')
    plt.ylabel('Expert ID')
    plt.title('Dynamic-K MoE Routing Pattern')
    plt.tight_layout()
    plt.show()
    
    print(f"   Routing matrix shape: {routing_matrix.shape}")
    print(f"   Non-zero routings: {(routing_matrix > 0).sum().item()}")

def run_all_tests():
    """Run all tests and provide summary"""
    print("🚀 Running Dynamic-K MoE Test Suite\n")
    print("=" * 50)
    
    tests = [
        ("Dynamic-K Behavior", test_dynamic_k_behavior),
        ("Tau Sensitivity", test_tau_sensitivity), 
        ("vs Traditional MoE", test_vs_traditional_moe),
        ("Load Balancing", test_expert_load_balancing),
        ("Capacity Constraints", test_capacity_constraints),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"   ❌ Test failed with error: {e}")
            results[test_name] = False
    
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
    
    all_passed = all(results.values())
    print(f"\nOverall: {'🎉 ALL TESTS PASSED' if all_passed else '⚠️  SOME TESTS FAILED'}")
    
    # Optional: visualize routing
    try:
        visualize_routing_pattern()
    except:
        print("Note: Visualization requires matplotlib")

def debug_top_router():
    """Debug the top_router to understand tau behavior"""
    print("🔍 Debugging top_router tau behavior...")
    
    # Create a simple test case
    logits = torch.tensor([[[1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.02]]])  # 1 batch, 1 seq, 8 experts
    
    print("Input logits:", logits[0, 0])
    
    model = DynamicMoE(dim=128, num_experts=8)
    
    for tau in [0.5, 0.7, 0.9, 0.95]:
        selected_experts, selected_probs = model.gate.top_router(logits, tau, max_k=8)
        
        print(f"\nτ={tau}:")
        print(f"  Selected experts: {selected_experts[0, 0]}")
        print(f"  Selected probs: {selected_probs[0, 0]}")
        
        # Check cumulative sum
        probs = torch.softmax(logits, dim=-1)
        p_sorted, _ = torch.sort(probs, descending=True, dim=-1)
        cumsum = torch.cumsum(p_sorted, dim=-1)
        
        print(f"  Sorted probs: {p_sorted[0, 0]}")
        print(f"  Cumulative sum: {cumsum[0, 0]}")
        print(f"  Mask (cumsum < tau): {(cumsum < tau)[0, 0]}")
        
        # Find crossing point
        crossing = (cumsum >= tau).float().argmax(dim=-1)
        print(f"  Crossing point: {crossing[0, 0].item()}")

class SimpleTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead=8, use_moe=True, moe_config=None):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        if use_moe:
            self.ffn = DynamicMoE(**moe_config)
            self.use_moe = True
        else:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model),
                torch.nn.SELU()
            )
            self.use_moe = False
    
    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feedforward
        if self.use_moe:
            ffn_out, moe_loss = self.ffn(x)
            x = self.norm2(x + ffn_out)
            return x, moe_loss
        else:
            ffn_out = self.ffn(x)
            x = self.norm2(x + ffn_out)
            return x, torch.tensor(0.0, device=x.device)

class SimpleLM(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=4, max_seq_len=512, moe_config=None):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(max_seq_len, d_model))
        
        # Create layers (only middle layers use MoE)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            use_moe = (i in [1, 2]) if num_layers >= 4 else (i == 1)  # Middle layers
            layer_moe_config = moe_config if use_moe else None
            self.layers.append(SimpleTransformerLayer(d_model, nhead, use_moe, layer_moe_config))
        
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
    
    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        
        # Embeddings
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = x + self.pos_embedding[:seq_len]
        
        # Transformer layers
        total_moe_loss = 0
        for layer in self.layers:
            x, moe_loss = layer(x)
            total_moe_loss = total_moe_loss + moe_loss
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits, total_moe_loss

if __name__ == "__main__":
    '''
    NOTE: Replaced top-2 expert selection with self.top_router() instead of top1(raw_gates)
    '''
    #test_current_code()
    run_all_tests()
    debug_top_router()
    # test_perplexity()

class testClass():
    print("hello world")