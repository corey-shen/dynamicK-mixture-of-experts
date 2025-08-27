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

MIN_EXPERT_CAPACITY = 4

def default(val, default_val):
    default_val = default_val() if isfunction(default_val) else default_val
    return val if val is not None else default_val

def cast_tuple(el):
    return el if isinstance(el, tuple) else (el,)

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

class GELU_(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_

class Experts(nn.Module):
    def __init__(self, dim, num_experts=16, hidden_dim=None, activation=GELU):
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
        out = torch.einsum('...nh,...hd->...nd', hidden, self.w2)
        return out

class DynamicKGating(nn.Module):
    def __init__(self, dim, num_gates, eps=1e-9, outer_expert_dims=tuple(),
                 threshold=0.8, capacity_factor_train=1.25, capacity_factor_eval=2.):
        super().__init__()
        self.eps = eps
        self.num_gates = num_gates
        self.threshold = threshold
        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval
        self.w_gating = nn.Parameter(torch.randn(*outer_expert_dims, dim, num_gates))
        
    def forward(self, x, importance=None):
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
        sel_p_sorted = torch.where(slot_mask, p_sorted, 0.0)
        renorm = sel_p_sorted.sum(dim=-1, keepdim=True).clamp_(min=1e-9)
        sel_p_sorted = sel_p_sorted / renorm
        B, T, E = probs.shape
        expert_masks = torch.zeros(B, T, E, device=probs.device, dtype=probs.dtype)
        expert_weights = torch.zeros_like(expert_masks)
        valid = sel_idx_sorted != -1
        expert_masks.scatter_add_(dim=-1, index=sel_idx_sorted.clamp(min=0), src=valid.to(probs.dtype))
        expert_weights.scatter_add_(dim=-1, index=sel_idx_sorted.clamp(min=0), src=sel_p_sorted)
        expert_weights = expert_weights * (expert_masks > 0)
        capacity_factor = self.capacity_factor_train if self.training else self.capacity_factor_eval
        expert_capacity = min(T, math.ceil((T * capacity_factor) / self.num_gates))
        expert_capacity = max(expert_capacity, MIN_EXPERT_CAPACITY)
        C = expert_capacity
        pos = cumsum_exclusive(expert_masks.transpose(1, 2), dim=-1)
        pos = pos.transpose(1, 2)
        keep_cap = (pos < float(C)) & (expert_masks > 0)
        pos = pos.clamp(max=C-1).long()
        dispatch = torch.zeros(B, T, E, C, device=probs.device, dtype=probs.dtype)  
        combine = torch.zeros_like(dispatch)
        t_idx = torch.arange(T, device=probs.device).view(1, T, 1).expand(B, T, E)
        b_idx = torch.arange(B, device=probs.device).view(B, 1, 1).expand(B, T, E)
        e_idx = torch.arange(E, device=probs.device).view(1, 1, E).expand(B, T, E)
        b_sel = b_idx[keep_cap]
        t_sel = t_idx[keep_cap]
        e_sel = e_idx[keep_cap]
        c_sel = pos[keep_cap]
        dispatch[b_sel, t_sel, e_sel, c_sel] = 1.0
        combine[b_sel, t_sel, e_sel, c_sel] = expert_weights[keep_cap]
        density = expert_masks.mean(dim=1)
        density_proxy = probs.mean(dim=1)
        aux_loss = (density * density_proxy).mean() * (self.num_gates ** 2)
        return dispatch, combine, aux_loss

class MoE(nn.Module):
    def __init__(self, dim, num_experts=16, hidden_dim=None, activation=nn.ReLU,
                 threshold=0.8, capacity_factor_train=1.25, capacity_factor_eval=2.,
                 loss_coef=1e-2, experts=None):
        super().__init__()
        self.num_experts = num_experts
        self.gate = DynamicKGating(dim, num_gates=num_experts, threshold=threshold,
                                  capacity_factor_train=capacity_factor_train,
                                  capacity_factor_eval=capacity_factor_eval)
        self.experts = default(experts, lambda: Experts(dim, num_experts=num_experts,
                                                       hidden_dim=hidden_dim, activation=activation))
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

class OneBillionWordDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512, split='train', max_samples=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.data = []
        file_paths = self._get_file_paths(data_path, split)
        if not file_paths:
            print(f"Warning: No files found for {split} split")
            print("Generating dummy data for demonstration...")
            self._generate_dummy_data()
        else:
            self._load_files(file_paths, max_samples)
    
    def _get_file_paths(self, data_path, split):
        if split == 'train':
            pattern = os.path.join(data_path, 'training-monolingual.tokenized.shuffled', 'news.en-*')
        else:
            pattern = os.path.join(data_path, 'heldout-monolingual.tokenized.shuffled', 'news.en.heldout-*')
        files = glob.glob(pattern)
        if files:
            print(f"Found {len(files)} files for {split} split")
        return sorted(files)
    
    def _load_files(self, file_paths, max_samples):
        total_loaded = 0
        max_samples = max_samples or float('inf')
        for file_path in file_paths:
            if total_loaded >= max_samples:
                break
            print(f"Loading: {os.path.basename(file_path)}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                batch_size = min(1000, max_samples - total_loaded)
                for line in tqdm(lines[:batch_size], desc="Tokenizing", leave=False):
                    line = line.strip()
                    if line:
                        tokens = self.tokenizer.encode(line, max_length=self.max_length, truncation=True)
                        if len(tokens) > 1:
                            self.data.append(tokens)
                            total_loaded += 1
                            if total_loaded >= max_samples:
                                break
                print(f"Loaded {len(self.data)} samples so far")
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")
                continue
        if not self.data:
            print("No data loaded, generating dummy data...")
            self._generate_dummy_data()
        else:
            print(f"Total samples loaded: {len(self.data)}")
    
    def _generate_dummy_data(self):
        dummy_sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning models require large amounts of data.",
            "Natural language processing has advanced significantly.",
            "Deep learning revolutionized artificial intelligence.",
            "Transformers are powerful neural network architectures.",
        ] * 200
        for sentence in dummy_sentences:
            tokens = self.tokenizer.encode(sentence, max_length=self.max_length, truncation=True)
            if len(tokens) > 1:
                self.data.append(tokens)
        print(f"Generated {len(self.data)} dummy samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tokens = self.data[idx].copy()
        if len(tokens) < self.max_length:
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]
        tokens = torch.tensor(tokens, dtype=torch.long)
        return tokens[:-1], tokens[1:]

class MoELanguageModel(nn.Module):
    def __init__(self, vocab_size, dim=512, num_layers=6, num_experts=8,
                 num_heads=8, max_seq_len=512, threshold=0.8):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_seq_len, dim)
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
        self.output_proj = nn.Linear(dim, vocab_size)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(pos_ids)
        x = token_emb + pos_emb
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1)
        total_aux_loss = 0
        for layer in self.layers:
            attn_out, _ = layer['attention'](x, x, x, attn_mask=causal_mask, key_padding_mask=attention_mask)
            x = layer['norm1'](x + attn_out)
            moe_out, aux_loss = layer['moe'](x)
            x = layer['norm2'](x + moe_out)
            total_aux_loss += aux_loss
        logits = self.output_proj(x)
        return logits, total_aux_loss

def calculate_perplexity(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for batch_idx, (input_ids, targets) in enumerate(tqdm(dataloader, desc="Calculating perplexity")):
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            attention_mask = (input_ids != dataloader.dataset.tokenizer.pad_token_id).float()
            logits, _ = model(input_ids, attention_mask)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                                  ignore_index=dataloader.dataset.tokenizer.pad_token_id, reduction='sum')
            num_tokens = (targets != dataloader.dataset.tokenizer.pad_token_id).sum()
            total_loss += loss.item()
            total_tokens += num_tokens.item()
            if batch_idx >= 50:
                break
    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 20))
    return perplexity, avg_loss

def train_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    total_aux_loss = 0
    total_tokens = 0
    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch}")
    for batch_idx, (input_ids, targets) in enumerate(progress_bar):
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        attention_mask = (input_ids != dataloader.dataset.tokenizer.pad_token_id).float()
        logits, aux_loss = model(input_ids, attention_mask)
        lm_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                                 ignore_index=dataloader.dataset.tokenizer.pad_token_id)
        loss = lm_loss + aux_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        num_tokens = (targets != dataloader.dataset.tokenizer.pad_token_id).sum()
        total_loss += lm_loss.item() * num_tokens.item()
        total_aux_loss += aux_loss.item() * num_tokens.item()
        total_tokens += num_tokens.item()
        if batch_idx % 10 == 0 and total_tokens > 0:
            avg_loss = total_loss / total_tokens
            avg_aux = total_aux_loss / total_tokens
            current_ppl = math.exp(min(avg_loss, 20))
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}', 'aux_loss': f'{avg_aux:.6f}', 'ppl': f'{current_ppl:.2f}'})
        if batch_idx >= 100:
            break
    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss, total_aux_loss / max(total_tokens, 1)

def main():
    config = {
        'data_path': '1-billion-word-language-modeling-benchmark',
        'batch_size': 8,
        'learning_rate': 1e-3,
        'num_epochs': 10,
        'max_seq_len': 64,
        'vocab_size': 50257,
        'dim': 256,
        'num_layers': 2,
        'num_experts': 4,
        'num_heads': 4,
        'threshold': 0.8,
        'max_train_samples': 100000,
        'max_test_samples': 100000,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"Using device: {config['device']}")
    
    if not os.path.exists(config['data_path']):
        print(f"\nError: Data path '{config['data_path']}' does not exist.")
        print("Please ensure the dataset is in the current directory.")
        return
    
    print("\nLoading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    print("\nCreating datasets...")
    train_dataset = OneBillionWordDataset(config['data_path'], tokenizer,
                                         max_length=config['max_seq_len'],
                                         split='train', max_samples=config['max_train_samples'])
    num_samples = len(train_dataset)
    print(f"Number of training samples: {num_samples}")

    valid_dataset = OneBillionWordDataset(config['data_path'], tokenizer,
                                         max_length=config['max_seq_len'],
                                         split='valid', max_samples=config['max_test_samples'])
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                            shuffle=True, num_workers=0, pin_memory=False)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'],
                            shuffle=False, num_workers=0, pin_memory=False)
    
    print("\nInitializing model...")
    model = MoELanguageModel(vocab_size=config['vocab_size'], dim=config['dim'],
                           num_layers=config['num_layers'], num_experts=config['num_experts'],
                           num_heads=config['num_heads'], max_seq_len=config['max_seq_len'],
                           threshold=config['threshold']).to(config['device'])
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    
    for epoch in range(1, config['num_epochs'] + 1):
        train_loss, train_aux_loss = train_epoch(model, train_loader, optimizer, config['device'], epoch)
        train_ppl = math.exp(min(train_loss, 20))
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Train Perplexity: {train_ppl:.2f}")
        print(f"  Aux Loss: {train_aux_loss:.6f}")
        
        print(f"\nEvaluating epoch {epoch}...")
        test_perplexity, test_loss = calculate_perplexity(model, valid_loader, config['device'])
        print(f"  Testing Loss: {test_loss:.4f}")
        print(f"  Testing Perplexity: {test_perplexity:.2f}")
    
    print("\n" + "="*50)
    print("Training completed!")
    print("="*50)
    
    print("\nFinal evaluation...")
    final_perplexity, final_loss = calculate_perplexity(model, valid_loader, config['device'])
    print(f"Final Testing Loss: {final_loss:.4f}")
    print(f"Final Testing Perplexity: {final_perplexity:.2f}")

if __name__ == "__main__":
    main()