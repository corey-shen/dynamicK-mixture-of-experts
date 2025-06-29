import torch
from torch import nn
from mixture_of_experts import MoE

print("Defining custom Experts class...")
# a 3 layered MLP as the experts
class Experts(nn.Module):
    def __init__(self, dim, num_experts = 16):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(num_experts, dim, dim * 4))
        self.w2 = nn.Parameter(torch.randn(num_experts, dim * 4, dim * 4))
        self.w3 = nn.Parameter(torch.randn(num_experts, dim * 4, dim))
        self.act = nn.LeakyReLU(inplace = True)
    
    def forward(self, x):
        hidden1 = self.act(torch.einsum('end,edh->enh', x, self.w1))
        hidden2 = self.act(torch.einsum('end,edh->enh', hidden1, self.w2))
        out = torch.einsum('end,edh->enh', hidden2, self.w3)
        return out

print("Creating custom experts...")
experts = Experts(512, num_experts = 16)
print(f"Custom experts created with {sum(p.numel() for p in experts.parameters())} parameters")

print("Creating MoE with custom experts...")
moe = MoE(
    dim = 512,
    num_experts = 16,
    experts = experts
)
print(f"Total MoE parameters: {sum(p.numel() for p in moe.parameters())}")

print("Creating sample input...")
inputs = torch.randn(4, 1024, 512)

print("Running forward pass...")
out, aux_loss = moe(inputs)
print(f"Output shape: {out.shape}")
print(f"Auxiliary loss: {aux_loss.item()}")
print(f"Output mean: {out.mean().item()}")
print(f"Output std: {out.std().item()}")
