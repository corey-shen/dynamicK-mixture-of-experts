import torch
from torch import nn
import torch.nn.functional as F
import math
from inspect import isfunction
import os
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
import numpy as np
from tqdm import tqdm
import glob
import random

# ============ Your existing MoE code (unchanged) ============
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
        # Compute gate scores
        raw_gates = torch.einsum('...bnd,...de->...bne', x, self.w_gating)
        probs = raw_gates.softmax(dim=-1)       
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

        orig_shape = expert_inputs.shape
        expert_inputs = expert_inputs.reshape(e, -1, d)
        expert_outputs = self.experts(expert_inputs)
        expert_outputs = expert_outputs.reshape(*orig_shape)

        output = torch.einsum('ebcd,bnec->bnd', expert_outputs, combine_tensor)
        return output, loss * self.loss_coef

# ============ New code for 1B Word Benchmark integration ============

class OneBillionWordDataset(Dataset):
    """Dataset for 1 Billion Word Benchmark"""
    def __init__(self, data_path, tokenizer, max_length=512, split='train'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        
        # Load data files
        self.file_paths = self._get_file_paths(data_path, split)
        self.current_file_idx = 0
        self.current_file_data = []
        self.current_position = 0
        
        # Load first file
        if self.file_paths:
            self._load_next_file()
    
    def _get_file_paths(self, data_path, split):
        """Get all file paths for the split"""
        if split == 'train':
            pattern = os.path.join(data_path, 'training-monolingual.tokenized.shuffled', 'news.en-*')
        elif split == 'valid':
            pattern = os.path.join(data_path, 'heldout-monolingual.tokenized.shuffled', 'news.en.heldout-*')
        else:  # test
            # For test, you might want to use a held-out portion
            pattern = os.path.join(data_path, 'heldout-monolingual.tokenized.shuffled', 'news.en.heldout-*')
        
        files = glob.glob(pattern)
        return sorted(files)
    
    def _load_next_file(self):
        """Load the next file in sequence"""
        if self.current_file_idx < len(self.file_paths):
            file_path = self.file_paths[self.current_file_idx]
            print(f"Loading file: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Tokenize all lines
            self.current_file_data = []
            for line in tqdm(lines[:10000], desc="Tokenizing"):  # Limit for demo
                line = line.strip()
                if line:
                    tokens = self.tokenizer.encode(line, max_length=self.max_length, 
                                                  truncation=True)
                    if len(tokens) > 1:  # Skip empty tokenizations
                        self.current_file_data.append(tokens)
            
            self.current_position = 0
            self.current_file_idx += 1
    
    def __len__(self):
        # Approximate length (for DataLoader)
        return len(self.current_file_data) * len(self.file_paths)
    
    def __getitem__(self, idx):
        """Get a single example"""
        if self.current_position >= len(self.current_file_data):
            self._load_next_file()
            if self.current_position >= len(self.current_file_data):
                # Wrap around to first file if we've exhausted all files
                self.current_file_idx = 0
                self._load_next_file()
        
        tokens = self.current_file_data[self.current_position]
        self.current_position += 1
        
        # Pad or truncate to max_length
        if len(tokens) < self.max_length:
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]
        
        tokens = torch.tensor(tokens, dtype=torch.long)
        
        # For language modeling, input is tokens[:-1], target is tokens[1:]
        return tokens[:-1], tokens[1:]


class MoELanguageModel(nn.Module):
    """Language Model with MoE layers"""
    def __init__(self, vocab_size, dim=512, num_layers=6, num_experts=8, 
                 num_heads=8, max_seq_len=512, threshold=0.8):
        super().__init__()
        
        self.dim = dim
        self.num_layers = num_layers
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_seq_len, dim)
        
        # Transformer layers with MoE
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                'attention': nn.MultiheadAttention(dim, num_heads, batch_first=True),
                'norm1': nn.LayerNorm(dim),
                'moe': MoE(dim, num_experts=num_experts, hidden_dim=dim*4, 
                          activation=GELU, threshold=threshold),
                'norm2': nn.LayerNorm(dim)
            })
            self.layers.append(layer)
        
        # Output projection
        self.output_proj = nn.Linear(dim, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Get embeddings
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(pos_ids)
        x = token_emb + pos_emb
        
        # Create causal mask for attention
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1)
        
        total_aux_loss = 0
        
        # Pass through transformer layers
        for layer in self.layers:
            # Self-attention
            attn_out, _ = layer['attention'](x, x, x, attn_mask=causal_mask, 
                                            key_padding_mask=attention_mask)
            x = layer['norm1'](x + attn_out)
            
            # MoE feedforward
            moe_out, aux_loss = layer['moe'](x)
            x = layer['norm2'](x + moe_out)
            total_aux_loss += aux_loss
        
        # Project to vocabulary
        logits = self.output_proj(x)
        
        return logits, total_aux_loss


def calculate_perplexity(model, dataloader, device):
    """Calculate perplexity on a dataset"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch_idx, (input_ids, targets) in enumerate(tqdm(dataloader, desc="Calculating perplexity")):
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            
            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = (input_ids != dataloader.dataset.tokenizer.pad_token_id).float()
            
            # Forward pass
            logits, _ = model(input_ids, attention_mask)
            
            # Calculate loss
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=dataloader.dataset.tokenizer.pad_token_id,
                reduction='sum'
            )
            
            # Count non-padding tokens
            num_tokens = (targets != dataloader.dataset.tokenizer.pad_token_id).sum()
            
            total_loss += loss.item()
            total_tokens += num_tokens.item()
            
            # Limit evaluation for demo
            if batch_idx >= 100:
                break
    
    # Calculate perplexity
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return perplexity, avg_loss


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_aux_loss = 0
    total_tokens = 0
    
    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch}")
    for batch_idx, (input_ids, targets) in enumerate(progress_bar):
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        
        # Create attention mask
        attention_mask = (input_ids != dataloader.dataset.tokenizer.pad_token_id).float()
        
        # Forward pass
        logits, aux_loss = model(input_ids, attention_mask)
        
        # Calculate language modeling loss
        lm_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=dataloader.dataset.tokenizer.pad_token_id
        )
        
        # Total loss
        loss = lm_loss + aux_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Statistics
        num_tokens = (targets != dataloader.dataset.tokenizer.pad_token_id).sum()
        total_loss += lm_loss.item() * num_tokens.item()
        total_aux_loss += aux_loss.item() * num_tokens.item()
        total_tokens += num_tokens.item()
        
        # Update progress bar
        if batch_idx % 10 == 0:
            avg_loss = total_loss / max(total_tokens, 1)
            avg_aux = total_aux_loss / max(total_tokens, 1)
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'aux_loss': f'{avg_aux:.6f}',
                'ppl': f'{math.exp(avg_loss):.2f}'
            })
        
        # Limit training for demo
        if batch_idx >= 500:
            break
    
    avg_loss = total_loss / total_tokens
    return avg_loss, total_aux_loss / total_tokens


def main():
    # Configuration
    config = {
        'data_path': '/path/to/1-billion-word-language-modeling-benchmark',  # Update this path
        'batch_size': 16,
        'learning_rate': 1e-4,
        'num_epochs': 3,
        'max_seq_len': 128,  # Reduced for demo
        'vocab_size': 50257,  # GPT-2 tokenizer vocab size
        'dim': 512,
        'num_layers': 4,
        'num_experts': 8,
        'num_heads': 8,
        'threshold': 0.8,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"Using device: {config['device']}")
    
    # Initialize tokenizer
    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = OneBillionWordDataset(
        config['data_path'], 
        tokenizer, 
        max_length=config['max_seq_len'],
        split='train'
    )
    
    valid_dataset = OneBillionWordDataset(
        config['data_path'],
        tokenizer,
        max_length=config['max_seq_len'],
        split='valid'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Initialize model
    print("Initializing model...")
    model = MoELanguageModel(
        vocab_size=config['vocab_size'],
        dim=config['dim'],
        num_layers=config['num_layers'],
        num_experts=config['num_experts'],
        num_heads=config['num_heads'],
        max_seq_len=config['max_seq_len'],
        threshold=config['threshold']
    ).to(config['device'])
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(1, config['num_epochs'] + 1):
        # Train
        train_loss, train_aux_loss = train_epoch(
            model, train_loader, optimizer, config['device'], epoch
        )
        print(f"\nEpoch {epoch} - Train Loss: {train_loss:.4f}, "
              f"Train Perplexity: {math.exp(train_loss):.2f}, "
              f"Aux Loss: {train_aux_loss:.6f}")
        
        # Evaluate
        print(f"\nEvaluating epoch {epoch}...")
        val_perplexity, val_loss = calculate_perplexity(
            model, valid_loader, config['device']
        )
        print(f"Validation Loss: {val_loss:.4f}, "
              f"Validation Perplexity: {val_perplexity:.2f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_perplexity': val_perplexity,
            'config': config
        }
        torch.save(checkpoint, f'moe_lm_checkpoint_epoch_{epoch}.pt')
        print(f"Saved checkpoint for epoch {epoch}")
    
    print("\nTraining completed!")
    
    # Final evaluation
    print("\nFinal evaluation on validation set...")
    final_perplexity, final_loss = calculate_perplexity(
        model, valid_loader, config['device']
    )
    print(f"Final Validation Loss: {final_loss:.4f}")
    print(f"Final Validation Perplexity: {final_perplexity:.2f}")
    
    # Test expert activation patterns
    print("\nAnalyzing expert activation patterns...")
    analyze_expert_usage(model, valid_loader, config['device'])


def analyze_expert_usage(model, dataloader, device):
    """Analyze which experts are being used and how often"""
    model.eval()
    
    expert_counts = {i: torch.zeros(model.layers[i].moe.num_experts) 
                    for i in range(len(model.layers))}
    
    with torch.no_grad():
        for batch_idx, (input_ids, _) in enumerate(dataloader):
            if batch_idx >= 10:  # Analyze first 10 batches
                break
                
            input_ids = input_ids.to(device)
            batch_size, seq_len = input_ids.shape
            
            # Get embeddings
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            token_emb = model.token_embedding(input_ids)
            pos_emb = model.position_embedding(pos_ids)
            x = token_emb + pos_emb
            
            # Pass through layers and track expert usage
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1)
            
            for layer_idx, layer in enumerate(model.layers):
                # Self-attention
                attn_out, _ = layer['attention'](x, x, x, attn_mask=causal_mask)
                x = layer['norm1'](x + attn_out)
                
                # Track expert usage in MoE
                dispatch_tensor, combine_tensor, _ = layer['moe'].gate(x)
                
                # Count expert activations
                expert_usage = dispatch_tensor.sum(dim=(0, 1, 3))  # Sum over batch, sequence, capacity
                expert_counts[layer_idx] += expert_usage.cpu()
                
                # MoE forward
                moe_out, _ = layer['moe'](x)
                x = layer['norm2'](x + moe_out)
    
    # Print expert usage statistics
    print("\nExpert Usage Statistics:")
    print("-" * 50)
    for layer_idx in range(len(model.layers)):
        counts = expert_counts[layer_idx]
        total = counts.sum()
        percentages = (counts / total * 100).numpy()
        
        print(f"\nLayer {layer_idx}:")
        for expert_idx, pct in enumerate(percentages):
            bar = '█' * int(pct / 2)  # Simple bar chart
            print(f"  Expert {expert_idx:2d}: {pct:5.2f}% {bar}")
        
        # Calculate load balancing metrics
        std_dev = percentages.std()
        cv = std_dev / percentages.mean() * 100  # Coefficient of variation
        print(f"  Load Balance CV: {cv:.2f}% (lower is better)")


if __name__ == "__main__":
    main()