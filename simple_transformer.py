import numpy as np
import math

class SimpleTransformer:
    """
    A minimal transformer implementation for educational purposes.
    This implements the core components: attention, feed-forward, and layer norm.
    """
    
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_seq_len):
        self.vocab_size = vocab_size
        self.d_model = d_model  # embedding dimension
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        
        # Initialize parameters (normally would use proper initialization)
        self.embedding = np.random.randn(vocab_size, d_model) * 0.1
        self.pos_encoding = self._create_positional_encoding()
        
        # Transformer blocks
        self.layers = []
        for _ in range(n_layers):
            layer = {
                'attention': MultiHeadAttention(d_model, n_heads),
                'feed_forward': FeedForward(d_model),
                'ln1': LayerNorm(d_model),
                'ln2': LayerNorm(d_model)
            }
            self.layers.append(layer)
        
        # Output projection
        self.output_proj = np.random.randn(d_model, vocab_size) * 0.1
    
    def _create_positional_encoding(self):
        """Create sinusoidal positional encodings"""
        pos_enc = np.zeros((self.max_seq_len, self.d_model))
        
        for pos in range(self.max_seq_len):
            for i in range(0, self.d_model, 2):
                pos_enc[pos, i] = math.sin(pos / (10000 ** (i / self.d_model)))
                if i + 1 < self.d_model:
                    pos_enc[pos, i + 1] = math.cos(pos / (10000 ** (i / self.d_model)))
        
        return pos_enc
    
    def forward(self, input_ids):
        """Forward pass through the transformer"""
        seq_len = len(input_ids)
        
        # Embedding + positional encoding
        x = self.embedding[input_ids] + self.pos_encoding[:seq_len]
        
        # Pass through transformer layers
        for layer in self.layers:
            # Multi-head attention with residual connection
            attn_out = layer['attention'].forward(x)
            x = layer['ln1'].forward(x + attn_out)
            
            # Feed-forward with residual connection
            ff_out = layer['feed_forward'].forward(x)
            x = layer['ln2'].forward(x + ff_out)
        
        # Output projection to vocabulary
        logits = np.dot(x, self.output_proj)
        return logits

class MultiHeadAttention:
    """Multi-head self-attention mechanism"""
    
    def __init__(self, d_model, n_heads):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # dimension per head
        
        # Linear projections for Q, K, V
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1
        self.W_o = np.random.randn(d_model, d_model) * 0.1
    
    def forward(self, x):
        """Forward pass for multi-head attention"""
        seq_len, d_model = x.shape
        
        # Linear projections
        Q = np.dot(x, self.W_q)  # (seq_len, d_model)
        K = np.dot(x, self.W_k)  # (seq_len, d_model)
        V = np.dot(x, self.W_v)  # (seq_len, d_model)
        
        # Reshape for multi-head attention
        Q = Q.reshape(seq_len, self.n_heads, self.d_k).transpose(1, 0, 2)  # (n_heads, seq_len, d_k)
        K = K.reshape(seq_len, self.n_heads, self.d_k).transpose(1, 0, 2)
        V = V.reshape(seq_len, self.n_heads, self.d_k).transpose(1, 0, 2)
        
        # Scaled dot-product attention for each head
        attention_outputs = []
        for i in range(self.n_heads):
            attn_output = self._scaled_dot_product_attention(Q[i], K[i], V[i])
            attention_outputs.append(attn_output)
        
        # Concatenate heads
        concat_output = np.concatenate(attention_outputs, axis=-1)  # (seq_len, d_model)
        
        # Final linear projection
        output = np.dot(concat_output, self.W_o)
        return output
    
    def _scaled_dot_product_attention(self, Q, K, V):
        """Compute scaled dot-product attention"""
        # Attention scores
        scores = np.dot(Q, K.T) / math.sqrt(self.d_k)  # (seq_len, seq_len)
        
        # Apply causal mask (for autoregressive generation)
        seq_len = scores.shape[0]
        mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
        scores += mask
        
        # Softmax
        attention_weights = self._softmax(scores)
        
        # Apply attention to values
        output = np.dot(attention_weights, V)  # (seq_len, d_k)
        return output
    
    def _softmax(self, x):
        """Numerically stable softmax"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class FeedForward:
    """Position-wise feed-forward network"""
    
    def __init__(self, d_model, d_ff=None):
        if d_ff is None:
            d_ff = 4 * d_model  # Common practice: 4x the model dimension
        
        self.W1 = np.random.randn(d_model, d_ff) * 0.1
        self.W2 = np.random.randn(d_ff, d_model) * 0.1
        self.b1 = np.zeros(d_ff)
        self.b2 = np.zeros(d_model)
    
    def forward(self, x):
        """Forward pass: Linear -> ReLU -> Linear"""
        # First linear transformation
        hidden = np.dot(x, self.W1) + self.b1
        
        # ReLU activation
        hidden = np.maximum(0, hidden)
        
        # Second linear transformation
        output = np.dot(hidden, self.W2) + self.b2
        return output

class LayerNorm:
    """Layer normalization"""
    
    def __init__(self, d_model, eps=1e-6):
        self.eps = eps
        self.gamma = np.ones(d_model)  # learnable scale
        self.beta = np.zeros(d_model)  # learnable shift
    
    def forward(self, x):
        """Apply layer normalization"""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        
        # Normalize
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        
        # Scale and shift
        output = self.gamma * x_norm + self.beta
        return output


# Example usage and demonstration
def demonstrate_transformer():
    """Demonstrate the transformer with a simple example"""
    print("=== Simple Transformer Demonstration ===\n")
    
    # Setup
    vocab_size = 1000
    d_model = 128
    n_heads = 8
    n_layers = 2
    max_seq_len = 50
    
    # Create transformer
    transformer = SimpleTransformer(vocab_size, d_model, n_heads, n_layers, max_seq_len)
    
    # Example input (token IDs)
    input_ids = [1, 15, 42, 7, 99]  # Example token sequence
    print(f"Input token IDs: {input_ids}")
    print(f"Sequence length: {len(input_ids)}")
    print(f"Model dimension: {d_model}")
    print(f"Number of attention heads: {n_heads}")
    print(f"Number of layers: {n_layers}")
    
    # Forward pass
    logits = transformer.forward(input_ids)
    print(f"\nOutput shape: {logits.shape}")  # (seq_len, vocab_size)
    
    # Show predictions for the last token (next token prediction)
    last_token_logits = logits[-1]  # Last position predictions
    predicted_token = np.argmax(last_token_logits)
    print(f"Predicted next token ID: {predicted_token}")
    
    # Show attention mechanism working
    print("\n=== Attention Mechanism Demo ===")
    attention_layer = transformer.layers[0]['attention']
    
    # Get embeddings for visualization
    embeddings = transformer.embedding[input_ids] + transformer.pos_encoding[:len(input_ids)]
    print(f"Input embeddings shape: {embeddings.shape}")
    
    # Show how attention scores would look (simplified)
    Q = np.dot(embeddings, attention_layer.W_q)
    K = np.dot(embeddings, attention_layer.W_k)
    
    # Compute attention scores for first head
    d_k = transformer.d_model // transformer.n_heads
    Q_head = Q[:, :d_k]  # First head
    K_head = K[:, :d_k]
    
    scores = np.dot(Q_head, K_head.T) / math.sqrt(d_k)
    
    print(f"\nAttention scores shape: {scores.shape}")
    print("Attention scores (how much each position attends to others):")
    print(scores.round(3))
    
    print("\n=== Key Transformer Components ===")
    print("1. Token Embeddings: Convert tokens to dense vectors")
    print("2. Positional Encoding: Add position information")
    print("3. Multi-Head Attention: Let tokens attend to each other")
    print("4. Feed-Forward: Process each position independently")
    print("5. Layer Norm: Stabilize training")
    print("6. Residual Connections: Help with gradient flow")
    print("7. Output Projection: Convert to vocabulary probabilities")

if __name__ == "__main__":
    demonstrate_transformer()
