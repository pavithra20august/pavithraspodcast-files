import re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import List, Optional
import numpy as np

# ============================================================================
# 1. TOKENIZATION
# ============================================================================

class SimpleTokenizer:
    def __init__(self):
        self.vocab = {}
        self.token_to_id = {}
        self.id_to_token = {}
        self.unk_token = '[UNK]'
        self.pad_token = '[PAD]'
        self.eos_token = '[EOS]'
        
        # Add special tokens
        self.special_tokens = [self.pad_token, self.unk_token, self.eos_token]
        self.vocab_size = 0
        
    def build_vocab(self, texts: List[str]):
        """Build vocabulary from texts"""
        word_freq = {}
        
        for text in texts:
            tokens = self.tokenize(text)
            for token in tokens:
                word_freq[token] = word_freq.get(token, 0) + 1
        
        # Start with special tokens
        vocab_list = self.special_tokens.copy()
        
        # Add words by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        for word, freq in sorted_words:
            if word not in vocab_list:
                vocab_list.append(word)
        
        # Create mappings
        self.token_to_id = {token: idx for idx, token in enumerate(vocab_list)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        self.vocab_size = len(vocab_list)
        
        print(f"Vocabulary built with {self.vocab_size} tokens")
        
    def tokenize(self, text: str) -> List[str]:
        """Basic tokenization splitting on whitespace and punctuation"""
        pattern = r'\w+|[^\w\s]'
        tokens = re.findall(pattern, text.lower())
        return tokens
    
    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs"""
        tokens = self.tokenize(text)
        return [self.token_to_id.get(token, self.token_to_id[self.unk_token]) for token in tokens]
    
    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text"""
        tokens = [self.id_to_token.get(id, self.unk_token) for id in token_ids]
        return ' '.join(tokens)
    
    @property
    def pad_token_id(self):
        return self.token_to_id[self.pad_token]
    
    @property
    def eos_token_id(self):
        return self.token_to_id[self.eos_token]

# ============================================================================
# 2. POSITIONAL ENCODING
# ============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_seq_len: int = 1024):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        
        # Create division term for sinusoidal pattern
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           -(math.log(10000.0) / embed_dim))
        
        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to embeddings"""
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

# ============================================================================
# 3. MULTI-HEAD ATTENTION
# ============================================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.output = nn.Linear(embed_dim, embed_dim)
        
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, x: torch.Tensor, mask=None) -> tuple:
        batch_size, seq_len, embed_dim = x.shape
        
        # Generate Q, K, V
        Q = self.query(x)  # (batch, seq_len, embed_dim)
        K = self.key(x)
        V = self.value(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply causal mask for GPT (prevent looking at future tokens)
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        # Reshape back
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        
        # Final linear projection
        output = self.output(attended)
        
        return output, attention_weights

# ============================================================================
# 4. FEED FORWARD NETWORK
# ============================================================================

class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, ff_dim: int = None, dropout: float = 0.1):
        super().__init__()
        if ff_dim is None:
            ff_dim = 4 * embed_dim  # Standard is 4x the embedding dimension
            
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward transformation"""
        # First linear layer with ReLU activation
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        
        # Second linear layer
        x = self.linear2(x)
        
        return x

# ============================================================================
# 5. TRANSFORMER BLOCK
# ============================================================================

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int = None, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Multi-head attention
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        
        # Feed-forward network
        if ff_dim is None:
            ff_dim = 4 * embed_dim
        self.feed_forward = FeedForward(embed_dim, ff_dim, dropout)
        
        # Layer normalizations
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """Forward pass through transformer block"""
        
        # Self-attention with residual connection and layer norm
        # GPT uses pre-norm: LayerNorm -> Attention -> Residual
        attn_input = self.ln1(x)
        attn_output, _ = self.attention(attn_input, mask)
        attn_output = self.dropout(attn_output)
        x = x + attn_output  # Residual connection
        
        # Feed-forward with residual connection and layer norm
        ff_input = self.ln2(x)
        ff_output = self.feed_forward(ff_input)
        ff_output = self.dropout(ff_output)
        x = x + ff_output  # Residual connection
        
        return x

# ============================================================================
# 6. COMPLETE GPT MODEL
# ============================================================================

class GPTModel(nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        embed_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        max_seq_len: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Positional embeddings (learned, not sinusoidal in GPT)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, 4 * embed_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.ln_final = nn.LayerNorm(embed_dim)
        
        # Language modeling head
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights following GPT-2 paper"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
                
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through GPT model"""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create position indices
        pos_ids = torch.arange(0, seq_len, dtype=torch.long, device=device)
        pos_ids = pos_ids.unsqueeze(0).expand(batch_size, seq_len)
        
        # Get embeddings
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.pos_embedding(pos_ids)
        x = token_embeds + pos_embeds
        
        # Create causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        mask = mask.unsqueeze(0).unsqueeze(0)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, mask)
            
        # Final layer norm
        x = self.ln_final(x)
        
        # Language modeling head
        logits = self.lm_head(x)
        
        return logits

# ============================================================================
# 7. TEXT GENERATION
# ============================================================================

class TextGenerator:
    def __init__(self, model: GPTModel, tokenizer: SimpleTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
        
    def generate(
        self, 
        prompt: str, 
        max_length: int = 50, 
        temperature: float = 1.0,
        top_k: int = 50,
        do_sample: bool = True
    ) -> str:
        """Generate text using the GPT model"""
        
        # Tokenize input
        input_ids = torch.tensor([self.tokenizer.encode(prompt)])
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                logits = self.model(input_ids)
                
                # Get logits for last token
                next_token_logits = logits[0, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, -float('inf'))
                    next_token_logits[top_k_indices] = top_k_logits
                
                # Convert to probabilities
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Sample next token
                if do_sample:
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(probs, keepdim=True)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                
                # Stop if EOS token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                    
                # Prevent infinite generation
                if input_ids.shape[1] > 1024:
                    break
        
        # Decode generated text
        generated_text = self.tokenizer.decode(input_ids[0].tolist())
        return generated_text

# ============================================================================
# 8. TRAINING
# ============================================================================

class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer: SimpleTokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize and truncate
        tokens = self.tokenizer.encode(text)[:self.max_length]
        
        # Pad if necessary
        if len(tokens) < self.max_length:
            tokens.extend([self.tokenizer.pad_token_id] * (self.max_length - len(tokens)))
        
        return {'input_ids': torch.tensor(tokens, dtype=torch.long)}

class GPTTrainer:
    def __init__(self, model: GPTModel, tokenizer: SimpleTokenizer, device='cpu'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        
    def train_step(self, batch, optimizer):
        """Single training step"""
        self.model.train()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        
        # Create targets (shift input_ids by one position)
        targets = input_ids[:, 1:].contiguous()
        input_ids = input_ids[:, :-1]
        
        # Forward pass
        logits = self.model(input_ids)
        
        # Reshape for loss calculation
        batch_size, seq_len, vocab_size = logits.shape
        logits = logits.view(-1, vocab_size)
        targets = targets.view(-1)
        
        # Calculate loss
        loss = self.criterion(logits, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (important for transformer training)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        return loss.item()
    
    def train_epoch(self, dataloader, optimizer):
        """Train for one epoch"""
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            loss = self.train_step(batch, optimizer)
            total_loss += loss
            num_batches += 1
            
            if num_batches % 100 == 0:
                avg_loss = total_loss / num_batches
                print(f"Batch {num_batches}, Average Loss: {avg_loss:.4f}")
        
        return total_loss / num_batches

# ============================================================================
# 9. DEMONSTRATION FUNCTIONS
# ============================================================================

def demonstrate_tokenization():
    """Demonstrate tokenization process"""
    print("=== TOKENIZATION DEMO ===")
    
    # Sample texts
    texts = [
        "Hello world! How are you today?",
        "The quick brown fox jumps over the lazy dog.",
        "GPT models use transformer architecture."
    ]
    
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(texts)
    
    for text in texts:
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        
        print(f"Original: {text}")
        print(f"Tokens: {tokens}")
        print(f"IDs: {ids}")
        print(f"Decoded: {decoded}")
        print("-" * 50)

def demonstrate_attention():
    """Demonstrate attention mechanism"""
    print("=== ATTENTION DEMO ===")
    
    embed_dim = 128
    num_heads = 8
    seq_len = 10
    batch_size = 2
    
    attention = MultiHeadAttention(embed_dim, num_heads)
    
    # Create sample input
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # Create causal mask
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    
    # Forward pass
    output, attention_weights = attention(x, mask)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"Attention weights sum (should be ~1.0): {attention_weights[0, 0, 0, :].sum():.4f}")

def demonstrate_model():
    """Demonstrate complete GPT model"""
    print("=== GPT MODEL DEMO ===")
    
    # Model configuration (small for demo)
    vocab_size = 1000
    embed_dim = 256
    num_layers = 4
    num_heads = 8
    max_seq_len = 128
    
    # Create model
    model = GPTModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=max_seq_len
    )
    
    # Sample input
    batch_size, seq_len = 2, 20
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass
    with torch.no_grad():
        logits = model(input_ids)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input shape: {input_ids.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Memory usage: ~{sum(p.numel() * 4 for p in model.parameters()) / 1024**2:.1f} MB")

def demonstrate_sampling():
    """Demonstrate different sampling strategies"""
    print("=== SAMPLING STRATEGIES DEMO ===")
    
    # Simulate model logits for vocabulary of size 10
    logits = torch.tensor([2.0, 1.5, 1.0, 0.5, 0.2, 0.1, 0.0, -0.5, -1.0, -2.0])
    vocab = [f"token_{i}" for i in range(len(logits))]
    
    print(f"Logits: {logits.tolist()}")
    print(f"Vocabulary: {vocab}")
    print()
    
    # 1. Greedy decoding
    greedy = torch.argmax(logits)
    print(f"Greedy: {vocab[greedy]} (always picks highest)")
    
    # 2. Temperature sampling
    for temp in [0.5, 1.0, 2.0]:
        probs = F.softmax(logits / temp, dim=-1)
        sample = torch.multinomial(probs, 1)
        print(f"Temperature {temp}: {vocab[sample.item()]}, top 3 probs: {probs[:3].tolist()}")
    
    # 3. Top-k sampling
    top_k = 3
    top_k_logits, top_k_indices = torch.topk(logits, top_k)
    filtered_logits = torch.full_like(logits, -float('inf'))
    filtered_logits[top_k_indices] = top_k_logits
    probs = F.softmax(filtered_logits, dim=-1)
    sample = torch.multinomial(probs, 1)
    print(f"Top-k ({top_k}): {vocab[sample.item()]}, only considers top {top_k} tokens")

# ============================================================================
# 10. MAIN EXECUTION
# ============================================================================

def main():
    """Run all demonstrations"""
    print("GPT ALGORITHM STEP-BY-STEP DEMONSTRATION")
    print("=" * 60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Run demonstrations
    demonstrate_tokenization()
    print()
    
    demonstrate_attention()
    print()
    
    demonstrate_model()
    print()
    
    demonstrate_sampling()
    print()
    
    print("=" * 60)
    print("All demonstrations completed!")
    print("You can now use these components to build and train your own GPT model.")

if __name__ == "__main__":
    main()