import torch
import torch.nn as nn
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
import math

# Import the Dynamic-K MoE model (assuming it's available)
# from dynamic_k_moe import DynamicKMoETransformer

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
        
        for text in tqdm(texts, desc="Creating examples"):
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
            articles = content.split('\n\n')
            
            # Filter out empty articles and headers
            cleaned_articles = []
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
    valid_file = os.path.join(data_dir, 'wiki.valid.tokens')
    test_file = os.path.join(data_dir, 'wiki.test.tokens')
    
    print("Loading Wikitext-103 dataset...")
    
    # Load files
    train_texts = read_file(train_file) if os.path.exists(train_file) else []
    valid_texts = read_file(valid_file) if os.path.exists(valid_file) else []
    test_texts = read_file(test_file) if os.path.exists(test_file) else []
    
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
        for input_ids, target_ids in tqdm(dataloader, desc="Calculating perplexity"):
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)
            
            outputs = model(input_ids)
            loss = criterion(outputs['logits'].view(-1, outputs['logits'].size(-1)), 
                           target_ids.view(-1))
            
            # Count non-padding tokens
            valid_tokens = (target_ids != tokenizer.special_tokens['<PAD>']).sum().item()
            
            total_loss += loss.item()
            total_tokens += valid_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity

def train_wikitext_model(model, train_loader, val_loader, tokenizer, num_epochs=5, device='cpu'):
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
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 60)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_tokens = 0
        epoch_routing_stats = defaultdict(list)
        
        train_pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (input_ids, target_ids) in enumerate(train_pbar):
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)
            
            # Forward pass
            outputs = model(input_ids)
            
            # Calculate losses
            lm_loss = criterion(outputs['logits'].view(-1, outputs['logits'].size(-1)), 
                              target_ids.view(-1))
            aux_loss = outputs['aux_loss']
            
            # Adaptive auxiliary loss weight (higher early in training)
            aux_weight = 0.1 * (1.0 - epoch / num_epochs)
            total_loss = lm_loss + aux_weight * aux_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # Track metrics
            valid_tokens = (target_ids != tokenizer.special_tokens['<PAD>']).sum().item()
            train_loss += lm_loss.item() * valid_tokens
            train_tokens += valid_tokens
            
            # Collect routing statistics
            for layer_idx, stats in enumerate(outputs['routing_stats']):
                epoch_routing_stats[f'layer_{layer_idx}_avg_k'].append(stats['avg_k'])
            
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
                    for i, (val_input, val_target) in enumerate(val_loader):
                        if i >= 10:  # Quick validation sample
                            break
                        val_input, val_target = val_input.to(device), val_target.to(device)
                        val_outputs = model(val_input)
                        val_loss = criterion(val_outputs['logits'].view(-1, val_outputs['logits'].size(-1)), 
                                           val_target.view(-1))
                        val_tokens = (val_target != tokenizer.special_tokens['<PAD>']).sum().item()
                        val_sample_loss += val_loss.item() * val_tokens
                        val_sample_tokens += val_tokens
                
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
    
    return history

def generate_wikitext_sample(model, tokenizer, prompt="The", max_length=200, temperature=0.8, device='cpu'):
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
        'num_layers': 12,
        'num_heads': 12,
        'num_experts': 32,
        'expert_dim': 3072,
        'threshold': 0.85,
        'max_seq_len': 512,
        'dropout': 0.1
    }
    
    print("Model Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Data directory (update this path)
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
    train_dataset = WikiTextDataset(train_texts, tokenizer, max_length=config['max_seq_len'], stride=256)
    val_dataset = WikiTextDataset(val_texts, tokenizer, max_length=config['max_seq_len'], stride=512)
    
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
    print("\nGenerating sample text...")
    sample_prompts = [
        "The history of artificial intelligence",
        "In the field of machine learning",
        "The concept of neural networks"
    ]
    
    for prompt in sample_prompts:
        generated = generate_wikitext_sample(model, tokenizer, prompt, max_length=100, device=device)
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: {generated}")
    
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