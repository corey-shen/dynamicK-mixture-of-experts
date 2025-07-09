import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import inspect

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
            print(f"Iterating through batch {b} of batch size {batch_size} | line 143")
            for s in range(seq_len):
                print(f"Iterating through sequence {s} of sequence length {seq_len}, batch {b} batch size {batch_size} | line 145")
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
        print("Within DynamicKTransformerBlock class | line 197")
        super().__init__()
        print("Finished super().__init__() | line 199")
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        print("Finished Multi-head attention | line 205")
        
        # Dynamic-K MoE layer
        self.moe = DynamicKMoELayer(
            hidden_dim, num_experts, expert_dim, threshold, dropout=dropout
        )
        print("Finished Dynamic-K MoE layer | line 211")
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attn_mask=None):
        # Self-attention with residual connection
        print("Reached forward() of DynamicKTransformerBlock class | line 222")
        attn_out, _ = self.attention(x, x, x, attn_mask=attn_mask)
        print("Reached line 224")
        x = self.norm1(x + self.dropout(attn_out))
        
        # Dynamic-K MoE with residual connection
        moe_out, aux_loss, routing_stats = self.moe(x)
        print("Reached line 229")
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
        print("Reaches forward pass within DynamicKMoETransformer class | Line 267 /dynamic_k_moe.py")
        batch_size, seq_len = input_ids.shape
        
        # Create position indices
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        print("Create position indices | line 272")
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(positions)
        x = self.embedding_dropout(token_embeds + position_embeds)
        print("Create embeddings | line 278")
        
        # Process through transformer blocks
        total_aux_loss = 0.0
        all_routing_stats = []
        
        print(f"Length of self.blocks: {len(self.blocks)}")
        for block in self.blocks:
            x, aux_loss, routing_stats = block(x, attention_mask)
            total_aux_loss += aux_loss
            all_routing_stats.append(routing_stats)
        print("Finished iterating in self.blocks | line 288")
        
        # Final processing
        x = self.final_norm(x)
        logits = self.output_projection(x)
        print("Final processing | line 293")
        
        # Average auxiliary loss across layers
        avg_aux_loss = total_aux_loss / len(self.blocks)
        
        return {
            'logits': logits,
            'aux_loss': avg_aux_loss,
            'routing_stats': all_routing_stats
        }

# Example usage and testing
if __name__ == "__main__":
    # Model configuration
    config = {
        'vocab_size': 10000,
        'hidden_dim': 512,
        'num_layers': 6,
        'num_heads': 8,
        'num_experts': 16,
        'expert_dim': 2048,
        'threshold': 0.8,  # Dynamic-K threshold
        'max_seq_len': 512,
        'dropout': 0.1
    }
    
    # Create model
    model = DynamicKMoETransformer(**config)
    
    # Sample input
    batch_size = 4
    seq_len = 128
    input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(input_ids)
    
    print("=== Dynamic-K MoE Model Results ===")
    print(f"Input shape: {input_ids.shape}")
    print(f"Output logits shape: {output['logits'].shape}")
    print(f"Auxiliary loss: {output['aux_loss']:.6f}")
    
    # Analyze routing statistics
    print("\n=== Routing Statistics ===")
    for layer_idx, stats in enumerate(output['routing_stats']):
        print(f"\nLayer {layer_idx + 1}:")
        print(f"  Average k per token: {stats['avg_k']:.2f}")
        print(f"  Max k: {stats['max_k']}")
        print(f"  Min k: {stats['min_k']}")
        print(f"  Threshold: {stats['threshold']}")
        print(f"  Total tokens routed: {stats['total_tokens_routed']}")
        print(f"  Expert usage distribution: {stats['expert_usage'][:8]}...")  # Show first 8 experts
    
    # Compare with different thresholds
    print("\n=== Threshold Comparison ===")
    thresholds = [0.5, 0.7, 0.8, 0.9, 0.95]
    
    for threshold in thresholds:
        # Create router with different threshold
        router = DynamicKRouter(config['hidden_dim'], config['num_experts'], threshold)
        
        # Test with sample input
        x = torch.randn(2, 10, config['hidden_dim'])
        selected_experts, selected_probs, k_values = router(x)
        
        avg_k = k_values.float().mean().item()
        print(f"Threshold {threshold}: Average k = {avg_k:.2f}")
    
    # Training example
    print("\n=== Training Example ===")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Generate target (shifted input for language modeling)
    target = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    
    # Forward pass
    output = model(input_ids)
    
    # Calculate losses
    lm_loss = criterion(output['logits'].view(-1, config['vocab_size']), target.view(-1))
    total_loss = lm_loss + 0.01 * output['aux_loss']  # Weight auxiliary loss
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    print(f"Language modeling loss: {lm_loss.item():.4f}")
    print(f"Auxiliary loss: {output['aux_loss'].item():.6f}")
    print(f"Total loss: {total_loss.item():.4f}")
    
    # Model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")