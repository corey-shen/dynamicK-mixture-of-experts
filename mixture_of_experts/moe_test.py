# NOTE: These are test functions originally in mixture_of_experts.py
# They are purely for testing purposes

import torch
import matplotlib.pyplot as plt
import numpy as np

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