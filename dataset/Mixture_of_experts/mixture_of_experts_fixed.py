# Clean copy of mixture_of_experts.py

import torch
from torch import nn
import math
from inspect import isfunction
from wikitext_loader import get_wikitext103
from transformers import AutoTokenizer, AutoModel
from datasets import load_from_disk
from torch.utils.data import DataLoader

MIN_EXPERT_CAPACITY = 4     # Used in line 101 of mixture_of_experts.py as part of expert_capacity calculation 

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

class DynamicKGating(nn.Module):
    def __init__(
        self,
        num_gates,  # Used for expert_capacity (benchmark),
        training=True,  # Used for self.training as a boolean value
        capacity_factor_train = 1.25,   # arbitrary values? | Used for calculating expert_capacity
        capacity_factor_eval = 2.,      # arbitrary values? | Used for calculating expert_capacity
        tau = 0.7,  # Hyperparameter we can adjust
        max_k = 8,  # Max number of experts 
        model_name = ""     # Name of the model we're running
        ):

        self.num_gates = num_gates
        self.w_gating = nn.Parameter(torch.randn(dim, num_gates))   # Calculation for logits

        self.training = training    # Boolean value | True by default
        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval
        self.tau = tau
        self.max_k = max_k
        self.model_name = model_name

        embed_model = AutoModel.from_pretrained(model_name)
        self.model = embed_model
        dim         = embed_model.config.hidden_size
        print(f"Dimension: {dim}")


if __name__ == "__main__":
    model_id  = "Qwen/Qwen3-4B"
