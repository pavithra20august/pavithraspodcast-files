import React, { useState, useEffect } from 'react';
import { ChevronRight, ChevronDown, Play, Brain, Layers, Eye, Calculator, ArrowRight } from 'lucide-react';

const TransformerEducationalTool = () => {
  const [activeStep, setActiveStep] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [currentDemo, setCurrentDemo] = useState('embeddings');

  // Sample data for demonstrations
  const sampleText = ["The", "cat", "sat", "on", "mat"];
  const vocabSize = 1000;
  const dModel = 64;
  const nHeads = 4;

  const steps = [
    {
      id: 'embeddings',
      title: '1. Token Embeddings & Positional Encoding',
      description: 'Convert words to vectors and add position information',
      code: `# Token Embeddings
class TokenEmbedding:
    def __init__(self, vocab_size, d_model):
        # Random initialization (normally learned)
        self.embedding_matrix = np.random.randn(vocab_size, d_model) * 0.1
    
    def forward(self, token_ids):
        return self.embedding_matrix[token_ids]

# Positional Encoding
class PositionalEncoding:
    def __init__(self, d_model, max_seq_len=512):
        pos_enc = np.zeros((max_seq_len, d_model))
        
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pos_enc[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
                if i + 1 < d_model:
                    pos_enc[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))
        
        self.pos_enc = pos_enc
    
    def forward(self, embeddings):
        seq_len = embeddings.shape[0]
        return embeddings + self.pos_enc[:seq_len]`
    },
    {
      id: 'attention',
      title: '2. Self-Attention Mechanism',
      description: 'How tokens attend to each other',
      code: `class SelfAttention:
    def __init__(self, d_model):
        self.d_model = d_model
        
        # Query, Key, Value projection matrices
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1
    
    def forward(self, x):
        # Step 1: Create Q, K, V matrices
        Q = np.dot(x, self.W_q)  # Queries: "what am I looking for?"
        K = np.dot(x, self.W_k)  # Keys: "what information do I have?"
        V = np.dot(x, self.W_v)  # Values: "what information do I give?"
        
        # Step 2: Compute attention scores
        scores = np.dot(Q, K.T) / np.sqrt(self.d_model)
        
        # Step 3: Apply softmax to get attention weights
        attention_weights = self.softmax(scores)
        
        # Step 4: Apply attention to values
        output = np.dot(attention_weights, V)
        
        return output, attention_weights
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)`
    },
    {
      id: 'multihead',
      title: '3. Multi-Head Attention',
      description: 'Multiple attention heads for different relationships',
      code: `class MultiHeadAttention:
    def __init__(self, d_model, n_heads):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # dimension per head
        
        # Linear projections for all heads
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1
        self.W_o = np.random.randn(d_model, d_model) * 0.1  # output projection
    
    def forward(self, x):
        seq_len, d_model = x.shape
        
        # Create Q, K, V for all heads
        Q = np.dot(x, self.W_q).reshape(seq_len, self.n_heads, self.d_k)
        K = np.dot(x, self.W_k).reshape(seq_len, self.n_heads, self.d_k)
        V = np.dot(x, self.W_v).reshape(seq_len, self.n_heads, self.d_k)
        
        # Transpose for easier computation: (n_heads, seq_len, d_k)
        Q = Q.transpose(1, 0, 2)
        K = K.transpose(1, 0, 2)
        V = V.transpose(1, 0, 2)
        
        # Compute attention for each head
        head_outputs = []
        attention_maps = []
        
        for i in range(self.n_heads):
            # Scaled dot-product attention
            scores = np.dot(Q[i], K[i].T) / np.sqrt(self.d_k)
            attention_weights = self.softmax(scores)
            head_output = np.dot(attention_weights, V[i])
            
            head_outputs.append(head_output)
            attention_maps.append(attention_weights)
        
        # Concatenate all heads
        concat_output = np.concatenate(head_outputs, axis=-1)
        
        # Final linear projection
        output = np.dot(concat_output, self.W_o)
        
        return output, attention_maps`
    },
    {
      id: 'feedforward',
      title: '4. Feed-Forward Network',
      description: 'Position-wise processing after attention',
      code: `class FeedForward:
    def __init__(self, d_model, d_ff=None):
        if d_ff is None:
            d_ff = 4 * d_model  # Common practice: 4x expansion
        
        self.W1 = np.random.randn(d_model, d_ff) * 0.1
        self.W2 = np.random.randn(d_ff, d_model) * 0.1
        self.b1 = np.zeros(d_ff)
        self.b2 = np.zeros(d_model)
    
    def forward(self, x):
        # First linear layer + ReLU activation
        hidden = np.maximum(0, np.dot(x, self.W1) + self.b1)
        
        # Second linear layer (back to original dimension)
        output = np.dot(hidden, self.W2) + self.b2
        
        return output

# Why Feed-Forward?
# 1. Attention is about relationships between positions
# 2. Feed-forward processes each position independently
# 3. Adds non-linearity and computational power
# 4. Expansion then compression allows complex transformations`
    },
    {
      id: 'layernorm',
      title: '5. Layer Normalization & Residuals',
      description: 'Stabilizing training with normalization and skip connections',
      code: `class LayerNorm:
    def __init__(self, d_model, eps=1e-6):
        self.eps = eps
        self.gamma = np.ones(d_model)   # learnable scale parameter
        self.beta = np.zeros(d_model)   # learnable shift parameter
    
    def forward(self, x):
        # Compute mean and variance along the feature dimension
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        
        # Normalize
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        
        # Scale and shift
        return self.gamma * x_norm + self.beta

# Transformer Block with Residual Connections
class TransformerBlock:
    def __init__(self, d_model, n_heads):
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
    
    def forward(self, x):
        # Multi-head attention with residual connection
        attn_output, attn_weights = self.attention.forward(x)
        x = self.ln1.forward(x + attn_output)  # Residual + LayerNorm
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward.forward(x)
        x = self.ln2.forward(x + ff_output)    # Residual + LayerNorm
        
        return x, attn_weights`
    },
    {
      id: 'complete',
      title: '6. Complete Transformer',
      description: 'Putting it all together',
      code: `class Transformer:
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_seq_len):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # Components
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer blocks
        self.blocks = []
        for _ in range(n_layers):
            self.blocks.append(TransformerBlock(d_model, n_heads))
        
        # Output layer
        self.ln_final = LayerNorm(d_model)
        self.output_projection = np.random.randn(d_model, vocab_size) * 0.1
    
    def forward(self, token_ids):
        # Step 1: Token embeddings + positional encoding
        x = self.token_embedding.forward(token_ids)
        x = self.pos_encoding.forward(x)
        
        # Step 2: Pass through transformer blocks
        all_attention_weights = []
        for block in self.blocks:
            x, attn_weights = block.forward(x)
            all_attention_weights.append(attn_weights)
        
        # Step 3: Final layer norm
        x = self.ln_final.forward(x)
        
        # Step 4: Output projection to vocabulary
        logits = np.dot(x, self.output_projection)
        
        return logits, all_attention_weights
    
    def predict_next_token(self, token_ids):
        logits, _ = self.forward(token_ids)
        # Get the prediction for the last token
        last_token_logits = logits[-1]
        return np.argmax(last_token_logits)`
    }
  ];

  const AttentionVisualization = () => {
    const attentionMatrix = [
      [0.8, 0.1, 0.05, 0.03, 0.02],
      [0.2, 0.6, 0.15, 0.03, 0.02],
      [0.1, 0.3, 0.5, 0.08, 0.02],
      [0.05, 0.1, 0.2, 0.6, 0.05],
      [0.02, 0.05, 0.1, 0.3, 0.53]
    ];

    return (
      <div className="bg-gray-50 p-4 rounded-lg">
        <h4 className="font-semibold mb-2">Attention Weights Visualization</h4>
        <div className="grid grid-cols-6 gap-1 text-xs">
          <div></div>
          {sampleText.map((word, i) => (
            <div key={i} className="p-1 text-center font-medium">{word}</div>
          ))}
          {sampleText.map((word, i) => (
            <React.Fragment key={i}>
              <div className="p-1 font-medium">{word}</div>
              {attentionMatrix[i].map((weight, j) => (
                <div
                  key={j}
                  className="p-1 text-center rounded"
                  style={{
                    backgroundColor: `rgba(59, 130, 246, ${weight})`,
                    color: weight > 0.5 ? 'white' : 'black'
                  }}
                >
                  {weight.toFixed(2)}
                </div>
              ))}
            </React.Fragment>
          ))}
        </div>
        <p className="text-sm text-gray-600 mt-2">
          Darker blue = higher attention. Each row shows how much that word attends to others.
        </p>
      </div>
    );
  };

  const EmbeddingVisualization = () => {
    return (
      <div className="bg-gray-50 p-4 rounded-lg">
        <h4 className="font-semibold mb-2">Token Embeddings + Positional Encoding</h4>
        <div className="space-y-2">
          {sampleText.map((word, i) => (
            <div key={i} className="flex items-center space-x-2">
              <div className="w-12 text-sm font-medium">{word}</div>
              <div className="text-xs text-gray-600">→</div>
              <div className="flex space-x-1">
                {Array.from({length: 8}, (_, j) => (
                  <div
                    key={j}
                    className="w-4 h-4 rounded text-xs flex items-center justify-center"
                    style={{
                      backgroundColor: `hsl(${(i * 50 + j * 30) % 360}, 50%, 70%)`,
                      color: 'white'
                    }}
                  >
                    {(Math.random() * 2 - 1).toFixed(1)}
                  </div>
                ))}
                <div className="text-xs text-gray-600">+ pos</div>
              </div>
            </div>
          ))}
        </div>
        <p className="text-sm text-gray-600 mt-2">
          Each word becomes a vector of numbers. Position encoding is added to preserve word order.
        </p>
      </div>
    );
  };

  const MultiHeadVisualization = () => {
    return (
      <div className="bg-gray-50 p-4 rounded-lg">
        <h4 className="font-semibold mb-2">Multi-Head Attention (4 heads)</h4>
        <div className="grid grid-cols-2 gap-4">
          {[1, 2, 3, 4].map(head => (
            <div key={head} className="border rounded p-2 bg-white">
              <div className="text-xs font-medium mb-1">Head {head}</div>
              <div className="grid grid-cols-5 gap-0.5 text-xs">
                {Array.from({length: 25}, (_, i) => (
                  <div
                    key={i}
                    className="w-3 h-3 rounded"
                    style={{
                      backgroundColor: `hsl(${head * 90}, 60%, ${60 + Math.random() * 30}%)`,
                      opacity: 0.3 + Math.random() * 0.7
                    }}
                  />
                ))}
              </div>
            </div>
          ))}
        </div>
        <p className="text-sm text-gray-600 mt-2">
          Each head learns different types of relationships (syntax, semantics, etc.)
        </p>
      </div>
    );
  };

  const currentStep = steps[activeStep];

  return (
    <div className="max-w-6xl mx-auto p-6 bg-white">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-800 mb-4 flex items-center">
          <Brain className="mr-3 text-blue-600" size={32} />
          Interactive Transformer Builder
        </h1>
        <p className="text-gray-600">
          Learn how transformers work by building one step-by-step with detailed explanations and visualizations.
        </p>
      </div>

      {/* Progress Steps */}
      <div className="mb-8">
        <div className="flex flex-wrap gap-2 mb-4">
          {steps.map((step, index) => (
            <button
              key={step.id}
              onClick={() => setActiveStep(index)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                index === activeStep
                  ? 'bg-blue-600 text-white'
                  : index < activeStep
                  ? 'bg-green-100 text-green-800'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              {step.title}
            </button>
          ))}
        </div>
        
        {/* Progress bar */}
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div
            className="bg-blue-600 h-2 rounded-full transition-all duration-500"
            style={{ width: `${((activeStep + 1) / steps.length) * 100}%` }}
          />
        </div>
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left Column - Code and Explanation */}
        <div className="space-y-6">
          <div className="bg-white border rounded-lg p-6">
            <h2 className="text-xl font-bold text-gray-800 mb-2 flex items-center">
              <Layers className="mr-2 text-blue-600" size={20} />
              {currentStep.title}
            </h2>
            <p className="text-gray-600 mb-4">{currentStep.description}</p>
            
            {/* Code Block */}
            <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
              <pre className="text-sm">
                <code>{currentStep.code}</code>
              </pre>
            </div>
          </div>

          {/* Key Concepts */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
            <h3 className="font-semibold text-blue-800 mb-2">Key Concepts:</h3>
            {activeStep === 0 && (
              <ul className="space-y-2 text-sm text-blue-700">
                <li>• <strong>Token Embeddings:</strong> Convert discrete tokens to continuous vectors</li>
                <li>• <strong>Positional Encoding:</strong> Add position information using sin/cos functions</li>
                <li>• <strong>Why Both:</strong> Embeddings capture meaning, positions capture order</li>
              </ul>
            )}
            {activeStep === 1 && (
              <ul className="space-y-2 text-sm text-blue-700">
                <li>• <strong>Queries (Q):</strong> "What am I looking for?"</li>
                <li>• <strong>Keys (K):</strong> "What information do I contain?"</li>
                <li>• <strong>Values (V):</strong> "What information do I provide?"</li>
                <li>• <strong>Attention:</strong> softmax(QK^T/√d_k)V</li>
              </ul>
            )}
            {activeStep === 2 && (
              <ul className="space-y-2 text-sm text-blue-700">
                <li>• <strong>Multiple Heads:</strong> Learn different types of relationships</li>
                <li>• <strong>Parallel Processing:</strong> Each head works independently</li>
                <li>• <strong>Concatenation:</strong> Combine all head outputs</li>
                <li>• <strong>Final Projection:</strong> Mix information from all heads</li>
              </ul>
            )}
            {activeStep === 3 && (
              <ul className="space-y-2 text-sm text-blue-700">
                <li>• <strong>Position-wise:</strong> Same network applied to each position</li>
                <li>• <strong>Expansion:</strong> Hidden layer is typically 4x larger</li>
                <li>• <strong>Non-linearity:</strong> ReLU activation adds expressiveness</li>
                <li>• <strong>Compression:</strong> Back to original dimension</li>
              </ul>
            )}
            {activeStep === 4 && (
              <ul className="space-y-2 text-sm text-blue-700">
                <li>• <strong>Residual Connections:</strong> x + f(x) helps gradient flow</li>
                <li>• <strong>Layer Normalization:</strong> Stabilizes training</li>
                <li>• <strong>Order:</strong> Attention → Add&Norm → FFN → Add&Norm</li>
              </ul>
            )}
            {activeStep === 5 && (
              <ul className="space-y-2 text-sm text-blue-700">
                <li>• <strong>Stacked Blocks:</strong> Multiple transformer layers</li>
                <li>• <strong>Deep Representations:</strong> Each layer builds more complex features</li>
                <li>• <strong>Output Projection:</strong> Map to vocabulary for next token prediction</li>
              </ul>
            )}
          </div>
        </div>

        {/* Right Column - Visualizations */}
        <div className="space-y-6">
          {/* Interactive Demo */}
          <div className="bg-white border rounded-lg p-6">
            <h3 className="font-semibold text-gray-800 mb-4 flex items-center">
              <Eye className="mr-2 text-green-600" size={20} />
              Interactive Visualization
            </h3>
            
            {activeStep === 0 && <EmbeddingVisualization />}
            {activeStep === 1 && <AttentionVisualization />}
            {activeStep === 2 && <MultiHeadVisualization />}
            {activeStep === 3 && (
              <div className="bg-gray-50 p-4 rounded-lg">
                <h4 className="font-semibold mb-2">Feed-Forward Processing</h4>
                <div className="space-y-2">
                  {sampleText.map((word, i) => (
                    <div key={i} className="flex items-center space-x-2">
                      <div className="w-12 text-sm">{word}</div>
                      <div className="flex items-center space-x-1">
                        <div className="w-8 h-4 bg-blue-300 rounded"></div>
                        <ArrowRight size={12} />
                        <div className="w-16 h-4 bg-green-300 rounded"></div>
                        <ArrowRight size={12} />
                        <div className="w-8 h-4 bg-blue-300 rounded"></div>
                      </div>
                      <div className="text-xs text-gray-600">d_model → 4×d_model → d_model</div>
                    </div>
                  ))}
                </div>
                <p className="text-sm text-gray-600 mt-2">
                  Each position processed independently through expand → ReLU → compress
                </p>
              </div>
            )}
            {activeStep === 4 && (
              <div className="bg-gray-50 p-4 rounded-lg">
                <h4 className="font-semibold mb-2">Residual Connections & Layer Norm</h4>
                <div className="space-y-3">
                  <div className="flex items-center space-x-2">
                    <div className="w-16 h-8 bg-blue-200 rounded flex items-center justify-center text-xs">Input</div>
                    <div className="text-xs">+</div>
                    <div className="w-20 h-8 bg-green-200 rounded flex items-center justify-center text-xs">Attention</div>
                    <ArrowRight size={16} />
                    <div className="w-20 h-8 bg-purple-200 rounded flex items-center justify-center text-xs">LayerNorm</div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-16 h-8 bg-purple-200 rounded flex items-center justify-center text-xs">Result</div>
                    <div className="text-xs">+</div>
                    <div className="w-20 h-8 bg-orange-200 rounded flex items-center justify-center text-xs">Feed-Fwd</div>
                    <ArrowRight size={16} />
                    <div className="w-20 h-8 bg-purple-200 rounded flex items-center justify-center text-xs">LayerNorm</div>
                  </div>
                </div>
                <p className="text-sm text-gray-600 mt-2">
                  Residual connections help gradients flow, LayerNorm stabilizes training
                </p>
              </div>
            )}
            {activeStep === 5 && (
              <div className="bg-gray-50 p-4 rounded-lg">
                <h4 className="font-semibold mb-2">Complete Architecture</h4>
                <div className="space-y-2">
                  <div className="bg-blue-100 p-2 rounded text-sm">Token Embeddings + Positional Encoding</div>
                  <div className="bg-green-100 p-2 rounded text-sm">Transformer Block 1 (Attention + FFN)</div>
                  <div className="bg-green-100 p-2 rounded text-sm">Transformer Block 2 (Attention + FFN)</div>
                  <div className="bg-green-100 p-2 rounded text-sm">... (N blocks)</div>
                  <div className="bg-yellow-100 p-2 rounded text-sm">Final Layer Norm</div>
                  <div className="bg-red-100 p-2 rounded text-sm">Output Projection → Vocabulary</div>
                </div>
                <div className="mt-4 p-3 bg-white rounded border">
                  <div className="text-sm font-medium">Sample Output:</div>
                  <div className="text-xs font-mono mt-1">
                    Input: "The cat sat on"<br />
                    Predicted next: "the" (probability: 0.89)
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Architecture Overview */}
          <div className="bg-white border rounded-lg p-6">
            <h3 className="font-semibold text-gray-800 mb-4 flex items-center">
              <Calculator className="mr-2 text-purple-600" size={20} />
              Architecture Summary
            </h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span>Vocabulary Size:</span>
                <span className="font-mono">{vocabSize.toLocaleString()}</span>
              </div>
              <div className="flex justify-between">
                <span>Model Dimension:</span>
                <span className="font-mono">{dModel}</span>
              </div>
              <div className="flex justify-between">
                <span>Attention Heads:</span>
                <span className="font-mono">{nHeads}</span>
              </div>
              <div className="flex justify-between">
                <span>Current Step:</span>
                <span className="font-mono">{activeStep + 1}/6</span>
              </div>
            </div>
            
            <div className="mt-4 pt-4 border-t">
              <div className="text-xs text-gray-600">
                <strong>Parameters so far:</strong><br />
                Embeddings: {(vocabSize * dModel).toLocaleString()}<br />
                {activeStep >= 1 && `Attention: ${(dModel * dModel * 3).toLocaleString()}`}<br />
                {activeStep >= 3 && `Feed-Forward: ${(dModel * dModel * 8).toLocaleString()}`}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <div className="flex justify-between mt-8 pt-6 border-t">
        <button
          onClick={() => setActiveStep(Math.max(0, activeStep - 1))}
          disabled={activeStep === 0}
          className="px-6 py-2 bg-gray-100 text-gray-600 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-200"
        >
          Previous Step
        </button>
        
        <div className="text-sm text-gray-500 flex items-center">
          Step {activeStep + 1} of {steps.length}
        </div>
        
        <button
          onClick={() => setActiveStep(Math.min(steps.length - 1, activeStep + 1))}
          disabled={activeStep === steps.length - 1}
          className="px-6 py-2 bg-blue-600 text-white rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-blue-700 flex items-center"
        >
          Next Step
          <ChevronRight className="ml-1" size={16} />
        </button>
      </div>
    </div>
  );
};

export default TransformerEducationalTool;