import torch
from torch import nn
import math
from inspect import isfunction
from wikitext_loader import get_wikitext103
from transformers import AutoTokenizer
from datasets import load_from_disk
from torch.utils.data import DataLoader

MIN_EXPERT_CAPACITY = 4

def default(val, default_val):
    default_val = default_val() if isfunction(default_val) else default_val
    return val if val is not None else default_val

def cast_tuple(el):
    return el if isinstance(el, tuple) else (el,)

def init_(t):
    dim = t.shape[-1]
    std = 1 / math.sqrt(dim)
    return t.uniform_(-std, std)

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

        w1 = torch.zeros(*num_experts, dim, hidden_dim)     # raw weight tensors
        w2 = torch.zeros(*num_experts, hidden_dim, dim)     

        w1 = init_(w1)
        w2 = init_(w2)

        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        self.act = activation()

    def forward(self, x):
        hidden = torch.einsum('...nd,...dh->...nh', x, self.w1)    # Get each token's hidden representation/expert
        hidden = self.act(hidden)
        out    = torch.einsum('...nh,...hd->...nd', hidden, self.w2)    # Project back to the original embedded size
        return out

# the below code is almost all transcribed from the official tensorflow version, from which the papers are written
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/research/moe.py

# gating network

class DynamicKGating(nn.Module):
    def __init__(
        self,
        dim,
        num_gates,
        capacity_factor_train = 1.25,
        capacity_factor_eval = 2.,
        tau = 0.7,
        max_k = 8,
        model_name = ""
        ):
        super().__init__()

        self.num_gates = num_gates
        self.w_gating = nn.Parameter(torch.randn(dim, num_gates))

        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval
        self.tau = tau
        self.max_k = max_k

    def forward(self, x):
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
            loss = ((expert_usage.float() - mean_usage) ** 2).mean() / (mean_usage + 1e-8) # formula for variance
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
    

def load_pretrained(cls, map_location='cpu'):  
    """
    Loads a pretrained DynamicMoE model from a checkpoint.
    Assumes the checkpoint was saved using:
        torch.save({
            'model_args': {...},
            'state_dict': model.state_dict()
        }, path)
    """
    dataset_path = "tokenized_wikitext103"
    dataset = load_from_disk(dataset_path)
    dataloader = DataLoader(dataset, batch_size=4)
    # checkpoint = torch.load("tokenized_wikitext103", map_location=map_location)  # add torch.save()
    # Extract model constructor arguments
    # model_args = checkpoint.get("model_args", {})
    # model = cls(**model_args)
    # # Load the weights
    # model.load_state_dict(checkpoint["state_dict"])
    # return model
    return dataloader

if __name__ == "__main__":
    model_id  = "Qwen/Qwen3-4B"

    '''
    def __init__(
            self,
            dim,
            num_gates,
            capacity_factor_train = 1.25,
            capacity_factor_eval = 2.,
            tau = 0.7,
            max_k = 8,
            model_name = ""
            ):
    '''
    tokenizer = load_from_disk("tokenized_wikitext103")
    # model = DynamicMoE.load_pretrained(model_id).cuda().eval()
    model = DynamicKGating(4, 4, 1.25, 2, 0.7, 8, model_id)
    #print(f"Model shape: {model.shape[-1]}")
    print(f"Model dimension: {model.dim}")

    #def calculate_wikitext():
    total_loss, n_tokens = 0.0, 0
    ce = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")

    forward_pass = model.forward()  # returns a tuple of length 3

    with torch.grad():
        for batch in tokenizer:
            ids = batch["input_ids"].cuda()
            labels = ids.clone()
            logits = model(ids).logits          # (b, t, vocab)   model[ids].logits
            loss   = ce(logits[:, :-1].reshape(-1, logits.size(-1)),
                        labels[:, 1:].reshape(-1))
            total_loss += loss.item()
            n_tokens   += ids[:, 1:].numel()

    perplexity = math.exp(total_loss / n_tokens)
    print(f"Perplexity Score on WikiText-103: {perplexity:8.2f}")