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
def cumsum_exclusive(t, dim=-1):
    num_dims = len(t.shape)
    num_pad_dims = - dim - 1
    pre_padding = (0, 0) * num_pad_dims
    pre_slice   = (slice(None),) * num_pad_dims
    padded_t = F.pad(t, (*pre_padding, 1, 0)).cumsum(dim=dim)
    return padded_t[(..., slice(None, -1), *pre_slice)]

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
    Selects as many experts as needed until cumulative probability exceeds threshold.
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
        probs = raw_gates.softmax(dim=-1)       # raw_gates is not defined
        p_sorted, idx_sorted = torch.sort(probs, -1, descending=True)
        cumsum = torch.cumsum(p_sorted, dim=-1)
        keep = (cumsum < self.threshold)
        keep[..., 0] = True
        k_star = keep.sum(dim=-1).clamp(max=self.num_gates)
        range_e = torch.arange(self.num_gates, device=probs.device)
        slot_mask = range_e.view(1, 1, -1) < k_star.unsqueeze(-1)
        sel_idx_sorted = torch.where(slot_mask, idx_sorted, torch.full_like(idx_sorted, -1))
        sel_p_sorted   = torch.where(slot_mask, p_sorted, 0.0)
        renorm = sel_p_sorted.sum(dim=-1, keepdim=True).clamp_(min=1e-9)
        sel_p_sorted = sel_p_sorted / renorm
        B, T, E = probs.shape
        expert_masks = torch.zeros(B, T, E, device=probs.device, dtype=probs.dtype)
        expert_weights = torch.zeros_like(expert_masks)
        valid = sel_idx_sorted != -1
        expert_masks.scatter_add_(
            dim=-1,
            index=sel_idx_sorted.clamp(min=0),
            src=valid.to(probs.dtype)
        )
        expert_weights.scatter_add_(
            dim=-1,
            index=sel_idx_sorted.clamp(min=0),
            src=sel_p_sorted
        )
        expert_weights = expert_weights * (expert_masks > 0)
        if self.training:
            capacity_factor = self.capacity_factor_train
        else:
            capacity_factor = self.capacity_factor_eval
        expert_capacity = min(
            T,
            math.ceil((T * capacity_factor) / self.num_gates)
        )
        expert_capacity = max(expert_capacity, MIN_EXPERT_CAPACITY)
        C = expert_capacity
        pos = cumsum_exclusive(expert_masks.transpose(1, 2), dim=-1)
        pos = pos.transpose(1, 2)
        keep_cap = (pos < float(C)) & (expert_masks > 0)
        pos = pos.clamp(max=C-1).long()
        dispatch = torch.zeros(B, T, E, C, device=probs.device, dtype=probs.dtype)
        combine  = torch.zeros_like(dispatch)
        t_idx = torch.arange(T, device=probs.device).view(1, T, 1).expand(B, T, E)
        b_idx = torch.arange(B, device=probs.device).view(B, 1, 1).expand(B, T, E)
        e_idx = torch.arange(E, device=probs.device).view(1, 1, E).expand(B, T, E)
        b_sel = b_idx[keep_cap]
        t_sel = t_idx[keep_cap]
        e_sel = e_idx[keep_cap]
        c_sel = pos[keep_cap]
        dispatch[b_sel, t_sel, e_sel, c_sel] = 1.0
        combine [b_sel, t_sel, e_sel, c_sel] = expert_weights[keep_cap]
        density       = expert_masks.mean(dim=1)
        density_proxy = probs.mean(dim=1)
        aux_loss = (density * density_proxy).mean() * (self.num_gates ** 2)
        return dispatch, combine, aux_loss

# MoE with Dynamic-K gating
class MoE(nn.Module):
    def __init__(self,
        dim,
        num_experts = 16,
        hidden_dim = None,
        activation = nn.ReLU,
        threshold = 0.8,
        capacity_factor_train = 1.25,
        capacity_factor_eval = 2.,
        loss_coef = 1e-2,
        experts = None):
        super().__init__()

        self.num_experts = num_experts
        
        # Dynamic-K gating
        self.gate = DynamicKGating(
            dim, 
            num_gates = num_experts,
            threshold = threshold,
            capacity_factor_train = capacity_factor_train,
            capacity_factor_eval = capacity_factor_eval
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

# Hierarchical MoE with Dynamic-K gating
class HierarchicalMoE(nn.Module):
    def __init__(self,
        dim,
        num_experts = (4, 4),
        hidden_dim = None,
        activation = nn.ReLU,
        threshold = 0.8,
        capacity_factor_train = 1.25,
        capacity_factor_eval = 2.,
        loss_coef = 1e-2,
        experts = None):
        super().__init__()

        assert len(num_experts) == 2, 'only 2 levels of hierarchy for experts allowed for now'
        num_experts_outer, num_experts_inner = num_experts
        self.num_experts_outer = num_experts_outer
        self.num_experts_inner = num_experts_inner
        
        # Dynamic-K gating for both levels
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

        self.experts = default(experts, lambda: Experts(dim, num_experts = num_experts, hidden_dim = hidden_dim, activation = activation))
        self.loss_coef = loss_coef

    def forward(self, inputs, **kwargs):
        b, n, d, eo, ei = *inputs.shape, self.num_experts_outer, self.num_experts_inner
        dispatch_tensor_outer, combine_tensor_outer, loss_outer = self.gate_outer(inputs)
        expert_inputs_outer = torch.einsum('bnd,bnec->ebcd', inputs, dispatch_tensor_outer)

        # Compute importance scores for inner routing
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
    moe = MoE(
        dim=dim,
        num_experts=8,
        hidden_dim=2048,
        threshold=0.8,
        loss_coef=0.01
    )
    
    output, loss = moe(x)
    print(f"Output shape: {output.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # Test with different threshold (more selective)
    print("\nTesting MoE with higher threshold (0.95)...")
    moe_selective = MoE(
        dim=dim,
        num_experts=8,
        hidden_dim=2048,
        threshold=0.95,  # Will select fewer experts on average
        loss_coef=0.01
    )
    
    output, loss = moe_selective(x)
    print(f"Output shape: {output.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # Test Hierarchical MoE with Dynamic-K
    print("\nTesting Hierarchical MoE with Dynamic-K gating...")
    hierarchical_moe = HierarchicalMoE(
        dim=dim,
        num_experts=(4, 4),
        hidden_dim=2048,
        threshold=0.75,
        loss_coef=0.01
    )
    
    output, loss = hierarchical_moe(x)
    print(f"Output shape: {output.shape}")
    print(f"Loss: {loss.item():.4f}")