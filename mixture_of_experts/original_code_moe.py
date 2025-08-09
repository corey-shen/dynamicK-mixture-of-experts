import torch
from torch import nn
import torch.nn.functional as F
import math
from inspect import isfunction

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

# Dynamic-K Gating Network
class DynamicKGating(nn.Module):
    """
    Dynamic-K gating that selects experts based on cumulative probability threshold.
    Compatible with the original MoE interface.
    """
    def __init__(
        self,
        dim,
        num_gates,
        eps = 1e-9,
        outer_expert_dims = tuple(),
        threshold = 0.8,
        capacity_factor_train = 1.25,
        capacity_factor_eval = 2.):
        super().__init__()
        
        self.eps = eps
        self.num_gates = num_gates
        self.threshold = threshold
        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval
        
        # Gating network
        self.w_gating = nn.Parameter(torch.randn(*outer_expert_dims, dim, num_gates))
        
    def forward(self, x, importance = None):
        # Handle both 3D and 4D inputs (for hierarchical MoE)
        if x.dim() == 3:
            b, group_size, dim = x.shape
            x = x.unsqueeze(0)  # Add dummy dimension
            squeeze_output = True
        else:
            *outer_dims, b, group_size, dim = x.shape
            squeeze_output = False
            
        num_gates = self.num_gates
        
        if self.training:
            capacity_factor = self.capacity_factor_train
        else:
            capacity_factor = self.capacity_factor_eval
        
        # Compute gate scores
        raw_gates = torch.einsum('...bnd,...de->...bne', x, self.w_gating)
        raw_gates = raw_gates.softmax(dim=-1)
        
        # Sort probabilities in descending order
        sorted_probs, sorted_indices = torch.sort(raw_gates, dim=-1, descending=True)
        
        # Find k for each token using threshold
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        threshold_mask = cumulative_probs >= self.threshold
        
        # Get k values (number of experts to select per token)
        k_values = torch.argmax(threshold_mask.float(), dim=-1) + 1
        never_reached = ~threshold_mask.any(dim=-1)
        k_values = torch.where(never_reached, num_gates, k_values)
        
        # Find maximum k for tensor allocation
        max_k = min(k_values.max().item(), num_gates)
        
        # Create masks for selected experts
        expert_masks = torch.zeros_like(raw_gates)
        expert_weights = torch.zeros_like(raw_gates)
        
        # Handle different input dimensions
        if raw_gates.dim() == 3:
            # Standard 3D case
            for i in range(b):
                for j in range(group_size):
                    k = min(k_values[i, j].item(), max_k)
                    selected = sorted_indices[i, j, :k]
                    selected_probs = sorted_probs[i, j, :k]
                    
                    # Normalize selected probabilities
                    normalized_probs = selected_probs / selected_probs.sum()
                    
                    for idx, expert_idx in enumerate(selected):
                        expert_masks[i, j, expert_idx] = 1.0
                        expert_weights[i, j, expert_idx] = normalized_probs[idx]
        else:
            # 4D case for hierarchical MoE
            outer_dim = raw_gates.shape[0]
            for o in range(outer_dim):
                for i in range(b):
                    for j in range(group_size):
                        k = min(k_values[o, i, j].item(), max_k)
                        selected = sorted_indices[o, i, j, :k]
                        selected_probs = sorted_probs[o, i, j, :k]
                        
                        # Normalize selected probabilities
                        normalized_probs = selected_probs / selected_probs.sum()
                        
                        for idx, expert_idx in enumerate(selected):
                            expert_masks[o, i, j, expert_idx] = 1.0
                            expert_weights[o, i, j, expert_idx] = normalized_probs[idx]
        
        # Calculate capacity
        expert_capacity = min(group_size, int((group_size * capacity_factor) / num_gates))
        expert_capacity = max(expert_capacity, MIN_EXPERT_CAPACITY)
        expert_capacity_f = float(expert_capacity)
        
        # Compute assignment positions
        position_in_expert = cumsum_exclusive(expert_masks, dim=-2) * expert_masks
        
        # Apply capacity constraint
        expert_masks *= (position_in_expert < expert_capacity_f).float()
        position_in_expert *= expert_masks
        position_in_expert = position_in_expert.sum(dim=-1).long()
        
        # Create dispatch and combine tensors based on dimensions
        if raw_gates.dim() == 3:
            # Standard 3D case
            dispatch_tensor = torch.zeros(b, group_size, num_gates, expert_capacity, device=x.device)
            combine_tensor = torch.zeros(b, group_size, num_gates, expert_capacity, device=x.device)
            
            # Fill dispatch and combine tensors
            valid_mask = expert_masks.bool()
            valid_positions = position_in_expert.unsqueeze(-1).expand(-1, -1, num_gates)
            
            for i in range(b):
                for j in range(group_size):
                    for e in range(num_gates):
                        if valid_mask[i, j, e]:
                            pos = valid_positions[i, j, e].item()
                            if pos < expert_capacity:
                                dispatch_tensor[i, j, e, pos] = 1.0
                                combine_tensor[i, j, e, pos] = expert_weights[i, j, e]
        else:
            # 4D case for hierarchical MoE
            outer_dim = raw_gates.shape[0]
            dispatch_tensor = torch.zeros(outer_dim, b, group_size, num_gates, expert_capacity, device=x.device)
            combine_tensor = torch.zeros(outer_dim, b, group_size, num_gates, expert_capacity, device=x.device)
            
            # Fill dispatch and combine tensors
            valid_mask = expert_masks.bool()
            valid_positions = position_in_expert.unsqueeze(-1).expand(outer_dim, b, group_size, num_gates)
            
            for o in range(outer_dim):
                for i in range(b):
                    for j in range(group_size):
                        for e in range(num_gates):
                            if valid_mask[o, i, j, e]:
                                pos = valid_positions[o, i, j, e].item()
                                if pos < expert_capacity:
                                    dispatch_tensor[o, i, j, e, pos] = 1.0
                                    combine_tensor[o, i, j, e, pos] = expert_weights[o, i, j, e]
        
        # Calculate load balancing loss
        density = expert_masks.mean(dim=-2)
        density_proxy = raw_gates.mean(dim=-2)
        loss = (density_proxy * density).mean() * float(num_gates ** 2)
        
        # Add dynamic-k statistics to loss (for monitoring)
        avg_k = k_values.float().mean()
        loss = loss + 0.0 * avg_k  # Add without affecting gradient
        
        # Remove dummy dimension if added
        if squeeze_output and dispatch_tensor.dim() == 5:
            dispatch_tensor = dispatch_tensor.squeeze(0)
            combine_tensor = combine_tensor.squeeze(0)
        
        return dispatch_tensor, combine_tensor, loss

# Original Top2 Gating (kept for compatibility)
class Top2Gating(nn.Module):
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
        capacity_factor_eval = 2.):
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

    def forward(self, x, importance = None):
        *_, b, group_size, dim = x.shape
        num_gates = self.num_gates

        if self.training:
            policy = self.second_policy_train
            threshold = self.second_threshold_train
            capacity_factor = self.capacity_factor_train
        else:
            policy = self.second_policy_eval
            threshold = self.second_threshold_eval
            capacity_factor = self.capacity_factor_eval

        raw_gates = torch.einsum('...bnd,...de->...bne', x, self.w_gating)
        raw_gates = raw_gates.softmax(dim=-1)

        gate_1, index_1 = top1(raw_gates)
        mask_1 = F.one_hot(index_1, num_gates).float()
        density_1_proxy = raw_gates

        if importance is not None:
            equals_one_mask = (importance == 1.).float()
            mask_1 *= equals_one_mask[..., None]
            gate_1 *= equals_one_mask
            density_1_proxy = density_1_proxy * equals_one_mask[..., None]
            del equals_one_mask

        gates_without_top_1 = raw_gates * (1. - mask_1)

        gate_2, index_2 = top1(gates_without_top_1)
        mask_2 = F.one_hot(index_2, num_gates).float()

        if importance is not None:
            greater_zero_mask = (importance > 0.).float()
            mask_2 *= greater_zero_mask[..., None]
            del greater_zero_mask

        denom = gate_1 + gate_2 + self.eps
        gate_1 /= denom
        gate_2 /= denom

        density_1 = mask_1.mean(dim=-2)
        density_1_proxy = density_1_proxy.mean(dim=-2)
        loss = (density_1_proxy * density_1).mean() * float(num_gates ** 2)

        if policy == "all":
            pass
        elif policy == "none":
            mask_2 = torch.zeros_like(mask_2)
        elif policy == "threshold":
            mask_2 *= (gate_2 > threshold).float()
        elif policy == "random":
            probs = torch.zeros_like(gate_2).uniform_(0., 1.)
            mask_2 *= (probs < (gate_2 / max(threshold, self.eps))).float().unsqueeze(-1)
        else:
            raise ValueError(f"Unknown policy {policy}")

        expert_capacity = min(group_size, int((group_size * capacity_factor) / num_gates))
        expert_capacity = max(expert_capacity, MIN_EXPERT_CAPACITY)
        expert_capacity_f = float(expert_capacity)

        position_in_expert_1 = cumsum_exclusive(mask_1, dim=-2) * mask_1
        mask_1 *= (position_in_expert_1 < expert_capacity_f).float()
        mask_1_count = mask_1.sum(dim=-2, keepdim=True)
        mask_1_flat = mask_1.sum(dim=-1)
        position_in_expert_1 = position_in_expert_1.sum(dim=-1)
        gate_1 *= mask_1_flat

        position_in_expert_2 = cumsum_exclusive(mask_2, dim=-2) + mask_1_count
        position_in_expert_2 *= mask_2
        mask_2 *= (position_in_expert_2 < expert_capacity_f).float()
        mask_2_flat = mask_2.sum(dim=-1)

        position_in_expert_2 = position_in_expert_2.sum(dim=-1)
        gate_2 *= mask_2_flat
        
        combine_tensor = (
            gate_1[..., None, None]
            * mask_1_flat[..., None, None]
            * F.one_hot(index_1, num_gates)[..., None]
            * safe_one_hot(position_in_expert_1.long(), expert_capacity)[..., None, :] +
            gate_2[..., None, None]
            * mask_2_flat[..., None, None]
            * F.one_hot(index_2, num_gates)[..., None]
            * safe_one_hot(position_in_expert_2.long(), expert_capacity)[..., None, :]
        )

        dispatch_tensor = combine_tensor.bool().to(combine_tensor)
        return dispatch_tensor, combine_tensor, loss

# MoE with configurable gating
class MoE(nn.Module):
    def __init__(self,
        dim,
        num_experts = 16,
        hidden_dim = None,
        activation = nn.ReLU,
        gating_type = 'dynamic_k',  # 'dynamic_k' or 'top2'
        threshold = 0.8,  # For dynamic-k
        second_policy_train = 'random',  # For top2
        second_policy_eval = 'random',  # For top2
        second_threshold_train = 0.2,  # For top2
        second_threshold_eval = 0.2,  # For top2
        capacity_factor_train = 1.25,
        capacity_factor_eval = 2.,
        loss_coef = 1e-2,
        experts = None):
        super().__init__()

        self.num_experts = num_experts
        self.gating_type = gating_type
        
        # Choose gating mechanism
        if gating_type == 'dynamic_k':
            self.gate = DynamicKGating(
                dim, 
                num_gates = num_experts,
                threshold = threshold,
                capacity_factor_train = capacity_factor_train,
                capacity_factor_eval = capacity_factor_eval
            )
        else:  # top2
            gating_kwargs = {
                'second_policy_train': second_policy_train,
                'second_policy_eval': second_policy_eval,
                'second_threshold_train': second_threshold_train,
                'second_threshold_eval': second_threshold_eval,
                'capacity_factor_train': capacity_factor_train,
                'capacity_factor_eval': capacity_factor_eval
            }
            self.gate = Top2Gating(dim, num_gates = num_experts, **gating_kwargs)
        
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

# Hierarchical MoE with configurable gating
class HeirarchicalMoE(nn.Module):
    def __init__(self,
        dim,
        num_experts = (4, 4),
        hidden_dim = None,
        activation = nn.ReLU,
        gating_type = 'dynamic_k',  # 'dynamic_k' or 'top2'
        threshold = 0.8,  # For dynamic-k
        second_policy_train = 'random',  # For top2
        second_policy_eval = 'random',  # For top2
        second_threshold_train = 0.2,  # For top2
        second_threshold_eval = 0.2,  # For top2
        capacity_factor_train = 1.25,
        capacity_factor_eval = 2.,
        loss_coef = 1e-2,
        experts = None):
        super().__init__()

        assert len(num_experts) == 2, 'only 2 levels of hierarchy for experts allowed for now'
        num_experts_outer, num_experts_inner = num_experts
        self.num_experts_outer = num_experts_outer
        self.num_experts_inner = num_experts_inner
        self.gating_type = gating_type
        
        # Create gating networks based on type
        if gating_type == 'dynamic_k':
            self.gate_outer = DynamicKGating(
                dim, 
                num_gates = num_experts_outer,
                threshold = threshold,
                capacity_factor_train = capacity_factor_train,
                capacity_factor_eval = capacity_factor_eval
            )
            self.gate_inner = DynamicKGating(
                dim,
                num_gates = num_experts_inner,
                outer_expert_dims = (num_experts_outer,),
                threshold = threshold,
                capacity_factor_train = capacity_factor_train,
                capacity_factor_eval = capacity_factor_eval
            )
        else:  # top2
            gating_kwargs = {
                'second_policy_train': second_policy_train,
                'second_policy_eval': second_policy_eval,
                'second_threshold_train': second_threshold_train,
                'second_threshold_eval': second_threshold_eval,
                'capacity_factor_train': capacity_factor_train,
                'capacity_factor_eval': capacity_factor_eval
            }
            self.gate_outer = Top2Gating(dim, num_gates = num_experts_outer, **gating_kwargs)
            self.gate_inner = Top2Gating(dim, num_gates = num_experts_inner, outer_expert_dims = (num_experts_outer,), **gating_kwargs)

        self.experts = default(experts, lambda: Experts(dim, num_experts = num_experts, hidden_dim = hidden_dim, activation = activation))
        self.loss_coef = loss_coef

    def forward(self, inputs, **kwargs):
        b, n, d, eo, ei = *inputs.shape, self.num_experts_outer, self.num_experts_inner
        dispatch_tensor_outer, combine_tensor_outer, loss_outer = self.gate_outer(inputs)
        expert_inputs_outer = torch.einsum('bnd,bnec->ebcd', inputs, dispatch_tensor_outer)

        importance = combine_tensor_outer.permute(2, 0, 3, 1).sum(dim=-1)
        importance = 0.5 * ((importance > 0.5).float() + (importance > 0.).float())

        dispatch_tensor_inner, combine_tensor_inner, loss_inner = self.gate_inner(expert_inputs_outer, importance = importance)
        expert_inputs = torch.einsum('ebnd,ebnfc->efbcd', expert_inputs_outer, dispatch_tensor_inner)

        orig_shape = expert_inputs.shape
        expert_inputs = expert_inputs.reshape(eo, ei, -1, d)
        expert_outputs = self.experts(expert_inputs)
        expert_outputs = expert_outputs.reshape(*orig_shape)

        expert_outputs_outer = torch.einsum('efbcd,ebnfc->ebnd', expert_outputs, combine_tensor_inner)
        output = torch.einsum('ebcd,bnec->bnd', expert_outputs_outer, combine_tensor_outer)
        return output, (loss_outer + loss_inner) * self.loss_coef

# Example usage
if __name__ == "__main__":
    # Test Dynamic-K MoE
    batch_size = 2
    seq_len = 10
    dim = 512
    
    # Create input
    x = torch.randn(batch_size, seq_len, dim)
    
    # Test standard MoE with Dynamic-K gating
    print("Testing MoE with Dynamic-K gating...")
    dynamic_k_moe = MoE(
        dim=dim,
        num_experts=8,
        hidden_dim=2048,
        gating_type='dynamic_k',
        threshold=0.8,
        loss_coef=0.01
    )
    
    output, loss = dynamic_k_moe(x)
    print(f"Output shape: {output.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # Test standard MoE with Top2 gating (original)
    print("\nTesting MoE with Top2 gating...")
    top2_moe = MoE(
        dim=dim,
        num_experts=8,
        hidden_dim=2048,
        gating_type='top2',
        second_policy_train='random',
        loss_coef=0.01
    )
    
    output, loss = top2_moe(x)
    print(f"Output shape: {output.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # Test Hierarchical MoE with Dynamic-K
    print("\nTesting Hierarchical MoE with Dynamic-K gating...")
    hierarchical_moe = HeirarchicalMoE(
        dim=dim,
        num_experts=(4, 4),
        hidden_dim=2048,
        gating_type='dynamic_k',
        threshold=0.75,
        loss_coef=0.01
    )
    
    output, loss = hierarchical_moe(x)
    print(f"Output shape: {output.shape}")
    print(f"Loss: {loss.item():.4f}")