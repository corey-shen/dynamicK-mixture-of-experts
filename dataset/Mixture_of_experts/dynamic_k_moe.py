import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import inspect
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os
import re
from collections import Counter, defaultdict
import pickle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def get_current_line_number():
    """
    Returns the line number where this function is called.
    """
    frame = inspect.currentframe()
    if frame and frame.f_back:
        return frame.f_back.f_lineno
    return None

class DynamicKRouter(nn.Module):
    """
    Dynamic-K routing based on probability threshold.
    Selects experts until cumulative probability exceeds threshold τ.
    """
    def __init__(self, hidden_dim, num_experts, threshold=0.8):
        super().__init__()
        self.num_experts = num_experts
        self.threshold = threshold
        
        # Gating network
        self.gate = nn.Linear(hidden_dim, num_experts)
        
        # Initialize with small weights for stable training
        nn.init.normal_(self.gate.weight, 0, 0.1)
        nn.init.zeros_(self.gate.bias)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
        
        Returns:
            selected_experts: Indices of selected experts [batch_size, seq_len, max_k_found]
            selected_probs: Probabilities of selected experts [batch_size, seq_len, max_k_found]
            k_values: Number of experts selected per token [batch_size, seq_len]
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # Compute unnormalized routing logits z = (z1, ..., zn)
        logits = self.gate(x)  # [batch_size, seq_len, num_experts]
        
        # Apply softmax to get probability distribution pi = softmax(z)
        probs = F.softmax(logits, dim=-1)  # [batch_size, seq_len, num_experts]
        
        # Sort probabilities in descending order: p1 >= p2 >= ... >= pn
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        
        # Find optimal k for each token using threshold τ
        # k = min{k' : Σ(i=1 to k') pi >= τ}
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)  # [batch_size, seq_len, num_experts]
        
        # Find first position where cumulative probability >= threshold
        threshold_mask = cumulative_probs >= self.threshold  # [batch_size, seq_len, num_experts]
        
        # Get k values (number of experts to select per token)
        # Add 1 because we want the position where threshold is first exceeded
        k_values = torch.argmax(threshold_mask.float(), dim=-1) + 1  # [batch_size, seq_len]
        
        # Handle case where threshold is never reached (use all experts)
        never_reached = ~threshold_mask.any(dim=-1)  # [batch_size, seq_len]
        k_values = torch.where(never_reached, self.num_experts, k_values)
        
        # Find maximum k across all tokens for tensor allocation
        max_k = k_values.max().item()
        max_k = min(max_k, self.num_experts)  # Cap at number of experts
        
        # Create output tensors
        selected_experts = torch.full((batch_size, seq_len, max_k), -1, 
                                    dtype=torch.long, device=x.device)
        selected_probs = torch.zeros((batch_size, seq_len, max_k), 
                                   dtype=torch.float, device=x.device)
        
        # Fill in selected experts and probabilities
        for b in range(batch_size):
            for s in range(seq_len):
                k = k_values[b, s].item()
                k = min(k, max_k)  # Ensure we don't exceed tensor bounds
                
                # Select top-k experts for this token
                selected_experts[b, s, :k] = sorted_indices[b, s, :k]
                selected_probs[b, s, :k] = sorted_probs[b, s, :k]
        
        return selected_experts, selected_probs, k_values

class ExpertNetwork(nn.Module):
    """Individual expert network (feed-forward network)"""
    def __init__(self, hidden_dim, expert_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, expert_dim)
        self.fc2 = nn.Linear(expert_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x):
        return self.fc2(self.dropout(self.activation(self.fc1(x))))

class DynamicKMoELayer(nn.Module):
    """
    Dynamic-K Mixture of Experts Layer with threshold-based expert selection
    """
    def __init__(self, hidden_dim, num_experts, expert_dim, threshold=0.8, 
                 capacity_factor=1.25, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.threshold = threshold
        self.capacity_factor = capacity_factor
        
        # Dynamic-K router
        self.router = DynamicKRouter(hidden_dim, num_experts, threshold)
        
        # Expert networks
        self.experts = nn.ModuleList([
            ExpertNetwork(hidden_dim, expert_dim, dropout)
            for _ in range(num_experts)
        ])
        
        # Layer normalization for output
        self.output_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
        
        Returns:
            output: Expert-processed output [batch_size, seq_len, hidden_dim]
            aux_loss: Load balancing auxiliary loss
            routing_stats: Dictionary with routing statistics
        """
        batch_size, seq_len, hidden_dim = x.shape
        original_shape = x.shape
        
        # Get dynamic expert selection
        selected_experts, selected_probs, k_values = self.router(x)
        max_k = selected_experts.shape[-1]
        
        # Initialize output
        output = torch.zeros_like(x)
        
        # Track expert usage for load balancing
        expert_usage = torch.zeros(self.num_experts, device=x.device)
        total_tokens_routed = 0
        
        # Process tokens through selected experts
        for b in range(batch_size):
            #print(f"Iterating through batch {b} of batch size {batch_size} | line 143")
            for s in range(seq_len):
                #print(f"Iterating through sequence {s} of sequence length {seq_len}, batch {b} batch size {batch_size} | line 145")
                token_input = x[b, s]  # [hidden_dim]
                token_output = torch.zeros_like(token_input)
                
                k = k_values[b, s].item()
                k = min(k, max_k)
                
                # Process through each selected expert
                for i in range(k):
                    # print(f"Iterating through expert; iteration number {i} | line 154")
                    expert_idx = selected_experts[b, s, i].item()
                    if expert_idx >= 0:  # Valid expert (not padding)
                        prob = selected_probs[b, s, i]
                        
                        # Process token through expert
                        expert_output = self.experts[expert_idx](token_input.unsqueeze(0))
                        expert_output = expert_output.squeeze(0)
                        
                        # Weight by probability and add to output
                        token_output += prob * expert_output
                        
                        # Track expert usage
                        expert_usage[expert_idx] += 1
                        total_tokens_routed += 1
                
                output[b, s] = token_output
        
        # Apply output normalization
        output = self.output_norm(output)
        
        # Calculate load balancing loss (coefficient of variation)
        if total_tokens_routed > 0:
            expert_usage_normalized = expert_usage / total_tokens_routed
            mean_usage = expert_usage_normalized.mean()
            variance = ((expert_usage_normalized - mean_usage) ** 2).mean()
            aux_loss = variance / (mean_usage + 1e-8)  # Coefficient of variation
        else:
            aux_loss = torch.tensor(0.0, device=x.device)
        
        # Routing statistics
        routing_stats = {
            'avg_k': k_values.float().mean().item(),
            'max_k': k_values.max().item(),
            'min_k': k_values.min().item(),
            'expert_usage': expert_usage.cpu().numpy(),
            'total_tokens_routed': total_tokens_routed,
            'threshold': self.threshold
        }
        
        return output, aux_loss, routing_stats

class DynamicKTransformerBlock(nn.Module):
    """Transformer block with Dynamic-K MoE"""
    def __init__(self, hidden_dim, num_heads, num_experts, expert_dim, 
                 threshold=0.8, dropout=0.1):
        print(f"Reached line {get_current_line_number()}")
        super().__init__()
        print(f"Reached line {get_current_line_number()}")
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        print(f"Reached line {get_current_line_number()}")
        
        # Dynamic-K MoE layer
        self.moe = DynamicKMoELayer(
            hidden_dim, num_experts, expert_dim, threshold, dropout=dropout
        )
        print(f"Reached line {get_current_line_number()}")
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attn_mask=None):
        # Self-attention with residual connection
        print(f"Reached line {get_current_line_number()}")
        attn_out, _ = self.attention(x, x, x, attn_mask=attn_mask)
        print(f"Reached line {get_current_line_number()}")
        x = self.norm1(x + self.dropout(attn_out))
        
        # Dynamic-K MoE with residual connection
        moe_out, aux_loss, routing_stats = self.moe(x)
        print(f"Reached line {get_current_line_number()}")
        x = self.norm2(x + self.dropout(moe_out))
        
        return x, aux_loss, routing_stats

class DynamicKMoETransformer(nn.Module):
    """Complete Transformer model with Dynamic-K MoE"""
    def __init__(self, vocab_size, hidden_dim, num_layers, num_heads, num_experts,
                 expert_dim, max_seq_len=512, threshold=0.8, dropout=0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        
        # Transformer blocks with Dynamic-K MoE
        self.blocks = nn.ModuleList([
            DynamicKTransformerBlock(
                hidden_dim, num_heads, num_experts, expert_dim, threshold, dropout
            ) for _ in range(num_layers)
        ])
        
        # Output layers
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, attention_mask=None):
        print(f"Reached line {get_current_line_number()}")
        batch_size, seq_len = input_ids.shape
        
        # Create position indices
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        print(f"Reached line {get_current_line_number()}")
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(positions)
        x = self.embedding_dropout(token_embeds + position_embeds)
        print(f"Reached line {get_current_line_number()}")
        
        # Process through transformer blocks
        total_aux_loss = 0.0
        all_routing_stats = []
        
        print(f"Length of self.blocks: {len(self.blocks)}")
        for block in self.blocks:
            x, aux_loss, routing_stats = block(x, attention_mask)
            total_aux_loss += aux_loss
            all_routing_stats.append(routing_stats)
        print(f"Reached line {get_current_line_number()}")
        
        # Final processing
        x = self.final_norm(x)
        logits = self.output_projection(x)
        print(f"Reached line {get_current_line_number()}")
        
        # Average auxiliary loss across layers
        avg_aux_loss = total_aux_loss / len(self.blocks)
        
        return {
            'logits': logits,
            'aux_loss': avg_aux_loss,
            'routing_stats': all_routing_stats
        }
# === Below is original wikitext_runner.py ===
class WikiTextTokenizer:
    """Custom tokenizer for Wikitext dataset with vocabulary building"""
    
    def __init__(self, vocab_size=50000, min_freq=2):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_built = False
        
        # Special tokens
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
        }
        
        # Initialize with special tokens
        for token, idx in self.special_tokens.items():
            self.word_to_id[token] = idx
            self.id_to_word[idx] = token
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        # Remove special wiki markup
        text = re.sub(r' = [^=]+ = ', ' ', text)  # Remove section headers
        text = re.sub(r' = = [^=]+ = = ', ' ', text)  # Remove subsection headers
        text = re.sub(r'@-@ ', '', text)  # Remove @-@ markers
        text = re.sub(r' +', ' ', text)  # Multiple spaces to single space
        
        # Basic tokenization (split on whitespace and punctuation)
        tokens = re.findall(r'\b\w+\b|[.,!?;]', text.lower())
        return tokens
    
    def build_vocab(self, texts):
        """Build vocabulary from text corpus"""
        print("Building vocabulary...")
        word_counts = Counter()
        
        for text in tqdm(texts, desc="Processing texts"):
            tokens = self.preprocess_text(text)
            word_counts.update(tokens)
        
        # Select most frequent words
        most_common = word_counts.most_common(self.vocab_size - len(self.special_tokens))
        
        # Add words to vocabulary
        next_id = len(self.special_tokens)
        for word, count in most_common:
            if count >= self.min_freq:
                self.word_to_id[word] = next_id
                self.id_to_word[next_id] = word
                next_id += 1
        
        self.vocab_built = True
        print(f"Vocabulary built with {len(self.word_to_id)} tokens")
        
        # Print some statistics
        print(f"Most common words: {list(word_counts.most_common(10))}")
        print(f"Total unique words in corpus: {len(word_counts)}")
    
    def encode(self, text):
        """Convert text to token IDs"""
        if not self.vocab_built:
            raise ValueError("Vocabulary not built. Call build_vocab() first.")
        
        tokens = self.preprocess_text(text)
        token_ids = []
        
        for token in tokens:
            token_id = self.word_to_id.get(token, self.special_tokens['<UNK>'])
            token_ids.append(token_id)
        
        return token_ids
    
    def decode(self, token_ids):
        """Convert token IDs back to text"""
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_word:
                tokens.append(self.id_to_word[token_id])
        return ' '.join(tokens)
    
    def save(self, filepath):
        """Save tokenizer to file"""
        tokenizer_data = {
            'word_to_id': self.word_to_id,
            'id_to_word': self.id_to_word,
            'vocab_size': self.vocab_size,
            'vocab_built': self.vocab_built,
            'special_tokens': self.special_tokens
        }
        with open(filepath, 'wb') as f:
            pickle.dump(tokenizer_data, f)
        print(f"Tokenizer saved to {filepath}")
    
    def load(self, filepath):
        """Load tokenizer from file"""
        with open(filepath, 'rb') as f:
            tokenizer_data = pickle.load(f)
        
        self.word_to_id = tokenizer_data['word_to_id']
        self.id_to_word = tokenizer_data['id_to_word']
        self.vocab_size = tokenizer_data['vocab_size']
        self.vocab_built = tokenizer_data['vocab_built']
        self.special_tokens = tokenizer_data['special_tokens']
        print(f"Tokenizer loaded from {filepath}")

class WikiTextDataset(Dataset):
    """Dataset class for Wikitext data with proper tokenization"""
    
    def __init__(self, texts, tokenizer, max_length=512, stride=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        
        print("Tokenizing dataset...")
        self.examples = []

        print(f"Total length: {len(texts)}")
        for text in tqdm(texts, total=len(texts), desc="Creating examples"):
            if len(text.strip()) < 50:  # Skip very short texts
                continue
                
            # Tokenize the text
            token_ids = tokenizer.encode(text)
            
            # Create overlapping windows
            for i in range(0, len(token_ids) - max_length + 1, stride):
                window = token_ids[i:i + max_length]
                if len(window) == max_length:
                    self.examples.append(window)
        
        print(f"Created {len(self.examples)} training examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        tokens = self.examples[idx]
        
        # Create input and target (shifted by 1 for language modeling)
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        
        return input_ids, target_ids

def load_wikitext_data(data_dir):
    """Load Wikitext-103 dataset from directory"""
    
    def read_file(filepath):
        """Read and clean a single file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split into articles (separated by double newlines)
            articles = content.split('\n')
            
            # Filter out empty articles and headers
            cleaned_articles = []
            print(f"Length of articles: {len(articles)}")
            for article in articles:
                article = article.strip()
                if len(article) > 100 and not article.startswith('='):
                    cleaned_articles.append(article)
            
            return cleaned_articles
        
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return []
    
    # File paths for Wikitext-103
    train_file = os.path.join(data_dir, 'wiki.train.tokens')
    #train_file = "dataset/Mixture_of_experts/wikitext-103/wiki.train.tokens"
    valid_file = os.path.join(data_dir, 'wiki.valid.tokens')
    test_file = os.path.join(data_dir, 'wiki.test.tokens')
    
    print("Loading Wikitext-103 dataset...")
    
    # Load files
    train_texts = read_file(train_file) if os.path.exists(train_file) else []
    valid_texts = read_file(valid_file) if os.path.exists(valid_file) else []
    test_texts = read_file(test_file) if os.path.exists(test_file) else []
    print(f"Train file exists: {os.path.exists(train_file)}")  
    print(f"Valid file exists: {os.path.exists(valid_file)}")  
    print(f"Test file exists: {os.path.exists(test_file)}")    

    # print(f"Train texts: {train_texts}")
    # print(f"Valid texts: {valid_texts}")
    # print(f"Test texts: {test_texts}")
    
    print(f"Loaded {len(train_texts)} training articles")
    print(f"Loaded {len(valid_texts)} validation articles")
    print(f"Loaded {len(test_texts)} test articles")
    
    return train_texts, valid_texts, test_texts

def calculate_perplexity(model, dataloader, device, tokenizer):
    """Calculate perplexity on a dataset"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.special_tokens['<PAD>'], reduction='sum')
    
    with torch.no_grad():
        #print(f"Length of tqdm: {len(tqdm(dataloader, desc="Calculating perplexity"))} | line 225")
        
        
        for input_ids, target_ids in tqdm(dataloader, desc="Calculating perplexity"):
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)
            
            outputs = model(input_ids)
            
            #print(f"Iteration number: {count2} of {count}")
            loss = criterion(outputs['logits'].view(-1, outputs['logits'].size(-1)), 
                           target_ids.view(-1))
            
            #print(f"target_ids: {target_ids}")
            # Count non-padding tokens
            valid_tokens = (target_ids != tokenizer.special_tokens['<PAD>']).sum().item()
            print(f"valid_tokens: {valid_tokens} | line 560")
            
            total_loss += loss.item()
            print(f"total_loss: {total_loss}")

            total_tokens += valid_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity

def train_wikitext_model(model, train_loader, val_loader, tokenizer, num_epochs=5, device = device):
    """Training loop optimized for Wikitext dataset"""
    
    # Setup optimizer with different learning rates for different components
    optimizer = optim.AdamW([
        {'params': model.token_embedding.parameters(), 'lr': 1e-4},
        {'params': model.position_embedding.parameters(), 'lr': 1e-4},
        {'params': [p for n, p in model.named_parameters() if 'gate' in n], 'lr': 5e-5},  # Lower LR for gates
        {'params': [p for n, p in model.named_parameters() if 'expert' in n], 'lr': 1e-4},  # Normal LR for experts
        {'params': [p for n, p in model.named_parameters() if not any(x in n for x in ['gate', 'expert', 'embedding'])], 'lr': 1e-4}
    ], weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=[1e-4, 1e-4, 5e-5, 1e-4, 1e-4],
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.special_tokens['<PAD>'])
    
    # Training history
    history = {
        'train_loss': [],
        'train_perplexity': [],
        'val_loss': [],
        'val_perplexity': [],
        'routing_stats': []
    }
    
    model.to(device)
    best_val_perplexity = float('inf')
    
    num_epochs = 1 #for testing, to make sure the run is successful
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        epoch_code(model, train_loader, val_loader, tokenizer, epoch, num_epochs, optimizer, scheduler, criterion, history, best_val_perplexity,device = device)
        print(f"Reached line {get_current_line_number()} (finished an epoch of training)")

    
    return history

def epoch_code(model, train_loader, val_loader, tokenizer,epoch, num_epochs,  optimizer, scheduler, criterion, history, best_val_perplexity, device = device):
    
    # Training phase
    model.train()
    train_loss = 0.0
    train_tokens = 0
    epoch_routing_stats = defaultdict(list)
    
    num_batches = len(train_loader)
    print(f"Number of iterations (batches) this epoch: {num_batches}")
    train_pbar = tqdm(train_loader, desc="Training")
    print(f"Total iterations according to tqdm: {train_pbar.total}")
    for batch_idx, (input_ids, target_ids) in enumerate(train_pbar):
        print(f"Reached line {get_current_line_number()}")
        input_ids, target_ids = input_ids.to(device), target_ids.to(device)
        
        # Forward pass
        outputs = model(input_ids)
        print(f"Reached line {get_current_line_number()} (after forward pass)")
        
        # Calculate losses
        
        lm_loss = criterion(outputs['logits'].view(-1, outputs['logits'].size(-1)), 
                            target_ids.view(-1))
        aux_loss = outputs['aux_loss']
        print(f"Reached line {get_current_line_number()} (after calculating losses)")
        
        # Adaptive auxiliary loss weight (higher early in training)
        aux_weight = 0.1 * (1.0 - epoch / num_epochs)
        total_loss = lm_loss + aux_weight * aux_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        print(f"Reached line {get_current_line_number()} (after backward pass)")
        
        # Track metrics
        valid_tokens = (target_ids != tokenizer.special_tokens['<PAD>']).sum().item()
        train_loss += lm_loss.item() * valid_tokens
        train_tokens += valid_tokens
        
        # Collect routing statistics
        for layer_idx, stats in enumerate(outputs['routing_stats']):
            epoch_routing_stats[f'layer_{layer_idx}_avg_k'].append(stats['avg_k'])
        print(f"Reached line {get_current_line_number()} (after collecting routing stats)")
        
        # Update progress bar
        current_ppl = math.exp(lm_loss.item())
        train_pbar.set_postfix({
            'Loss': f'{lm_loss.item():.4f}',
            'PPL': f'{current_ppl:.2f}',
            'Aux': f'{aux_loss.item():.4f}',
            'LR': f'{scheduler.get_last_lr()[0]:.6f}'
        })
        
        # Periodic validation during training
        if batch_idx % 1000 == 0 and batch_idx > 0:
            model.eval()
            val_sample_loss = 0.0
            val_sample_tokens = 0
            
            with torch.no_grad():
                print(f"Reached line {get_current_line_number()} (within peiodic validation)")
                for i, (val_input, val_target) in enumerate(val_loader):
                    if i >= 10:  # Quick validation sample
                        break
                    val_input, val_target = val_input.to(device), val_target.to(device)
                    val_outputs = model(val_input)
                    print(f"Reached line {get_current_line_number()} (after getting val outputs)")
                    val_loss = criterion(val_outputs['logits'].view(-1, val_outputs['logits'].size(-1)), 
                                        val_target.view(-1))
                    val_tokens = (val_target != tokenizer.special_tokens['<PAD>']).sum().item()
                    val_sample_loss += val_loss.item() * val_tokens
                    val_sample_tokens += val_tokens
                print(f"Reached line {get_current_line_number()} (after validation iteration)")
            
            if val_sample_tokens > 0:
                sample_ppl = math.exp(val_sample_loss / val_sample_tokens)
                print(f"\n  Batch {batch_idx}: Sample Val PPL = {sample_ppl:.2f}")
            
            model.train()
    
    # Calculate epoch training metrics
    avg_train_loss = train_loss / train_tokens
    train_perplexity = math.exp(avg_train_loss)
    
    # Validation phase
    print("Running full validation...")
    val_perplexity = calculate_perplexity(model, val_loader, device, tokenizer)
    
    # Save metrics
    history['train_loss'].append(avg_train_loss)
    history['train_perplexity'].append(train_perplexity)
    history['val_perplexity'].append(val_perplexity)
    
    # Average routing stats
    epoch_avg_routing = {}
    for key, values in epoch_routing_stats.items():
        epoch_avg_routing[key] = np.mean(values)
    print(f"Reached line {get_current_line_number()} (after calculating average routing stats)")
    history['routing_stats'].append(epoch_avg_routing)
    
    # Print epoch summary
    print(f"Train Loss: {avg_train_loss:.4f} | Train PPL: {train_perplexity:.2f}")
    print(f"Val PPL: {val_perplexity:.2f}")
    
    if epoch_avg_routing:
        avg_k_all_layers = np.mean([v for k, v in epoch_avg_routing.items() if 'avg_k' in k])
        print(f"Average experts per token: {avg_k_all_layers:.2f}")
    
    # Save best model
    if val_perplexity < best_val_perplexity:
        best_val_perplexity = val_perplexity
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_perplexity': val_perplexity,
            'tokenizer': tokenizer,
            'config': model.config if hasattr(model, 'config') else None
        }, 'best_wikitext_dynamic_k_moe.pt')
        print(f"✓ New best model saved (Val PPL: {val_perplexity:.2f})")
    

def generate_wikitext_sample(model, tokenizer, prompt="The", max_length=200, temperature=0.8, device= device):
    """Generate text sample using the trained model"""
    model.eval()
    model.to(device)
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    generated_ids = input_ids.copy()
    
    with torch.no_grad():
        for _ in range(max_length):
            # Prepare input (use last 512 tokens if too long)
            if len(generated_ids) > 512:
                input_tensor = torch.tensor([generated_ids[-512:]], dtype=torch.long).to(device)
            else:
                input_tensor = torch.tensor([generated_ids], dtype=torch.long).to(device)
            
            # Forward pass
            outputs = model(input_tensor)
            
            # Get next token probabilities
            next_token_logits = outputs['logits'][0, -1, :] / temperature
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            
            # Sample next token (avoid special tokens)
            valid_tokens = torch.arange(len(tokenizer.special_tokens), len(tokenizer.word_to_id), device=device)
            valid_probs = next_token_probs[valid_tokens]
            valid_probs = valid_probs / valid_probs.sum()
            
            sampled_idx = torch.multinomial(valid_probs, 1)
            next_token = valid_tokens[sampled_idx].item()
            
            generated_ids.append(next_token)
            
            # Stop at end of sentence tokens
            if tokenizer.id_to_word.get(next_token, '') in ['.', '!', '?']:
                break
    
    # Decode generated text
    generated_text = tokenizer.decode(generated_ids)
    return generated_text

def main():
    """Main function to train Dynamic-K MoE on Wikitext-103"""
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model configuration
    config = {
        'vocab_size': 50000,
        'hidden_dim': 768,
        'num_layers': 12, #changed from 12 to 2 for testing
        'num_heads': 12,
        'num_experts': 16, #changed from 32 to 16 for testing
        'expert_dim': 3072,
        'threshold': 0.75,
        'max_seq_len': 1024,
        'dropout': 0.1
    }
    
    print("Model Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Data directory (update this path)
    print(f"Test.tokens: {os.path.exists('wikitext-103/wiki.test.tokens')}")
    print(f"Train.tokens: {os.path.exists('wikitext-103/wiki.train.tokens')}")
    print(f"Valid.tokens: {os.path.exists('wikitext-103/wiki.valid.tokens')}")
    data_dir = "./wikitext-103"  # Update this to your Wikitext-103 directory
    tokenizer_path = "wikitext_tokenizer.pkl"
    
    # Load or create tokenizer
    tokenizer = WikiTextTokenizer(vocab_size=config['vocab_size'])
    
    if os.path.exists(tokenizer_path):
        print("Loading existing tokenizer...")
        tokenizer.load(tokenizer_path)
    else:
        print("Creating new tokenizer...")
        # Load data for vocabulary building
        train_texts, val_texts, test_texts = load_wikitext_data(data_dir)
        
        if not train_texts:
            print("ERROR: Could not load training data. Please check the data directory path.")
            return
        
        # Build vocabulary on training data
        tokenizer.build_vocab(train_texts[:10000])  # Use subset for faster vocab building
        tokenizer.save(tokenizer_path)
    
    # Update vocab size based on actual tokenizer
    config['vocab_size'] = len(tokenizer.word_to_id)
    print(f"Actual vocabulary size: {config['vocab_size']}")
    
    # Create model
    model = DynamicKMoETransformer(**config)
    model.config = config
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Load data if not already loaded
    if 'train_texts' not in locals():
        train_texts, val_texts, test_texts = load_wikitext_data(data_dir)
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = WikiTextDataset(train_texts, tokenizer, max_length=1024, stride=128)
    val_dataset = WikiTextDataset(val_texts, tokenizer, max_length=256, stride=64)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")
    
    # Train model
    print("\nStarting training...")
    history = train_wikitext_model(model, train_loader, val_loader, tokenizer, 
                                 num_epochs=3, device=device)
    
 
    # Generate samples
    # print("\nGenerating sample text...")
    # sample_prompts = [
    #     "The history of artificial intelligence",
    #     "In the field of machine learning",
    #     "The concept of neural networks"
    # ]
    

    # for prompt in sample_prompts:
    #     generated = generate_wikitext_sample(model, tokenizer, prompt, max_length=100, device=device)
    #     print(f"\nPrompt: '{prompt}'")
    #     print(f"Generated: {generated}")

    
    # Save final results
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'history': history,
        'tokenizer': tokenizer
    }, 'final_wikitext_dynamic_k_moe.pt')
    
    print(f"\nFinal validation perplexity: {history['val_perplexity'][-1]:.2f}")
    print("Training completed successfully!")

if __name__ == "__main__":
    main()