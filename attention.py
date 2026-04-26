import torch
import torch.nn as nn
import math
from components import ManualLayerNorm, FeedForward


class MultiHeadAttention(nn.Module):
    """
    Implementation of the Multi-Head Attention mechanism from scratch.
    
    This is the "engine" of the Transformer. It allows the model to focus on 
    different parts of the input sequence simultaneously by splitting the 
    embedding dimension into multiple "heads".

    Args:
        embedding_dim (int): Total dimension of the input vector (d_model).
        num_heads (int): Number of parallel attention heads.
        dropout (float): Dropout probability for attention weights and output.
    """

    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        # Ensure the dimension can be evenly split across heads
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # Linear projections for Query, Key, and Value
        # We use a single large matrix for each and split them later for efficiency.
        self.w_query = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.w_key = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.w_value = nn.Linear(embedding_dim, embedding_dim, bias=True)

        # Final projection to merge all heads back together
        self.w_out = nn.Linear(embedding_dim, embedding_dim, bias=True)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        """
        Initializes weights using Xavier Uniform and biases to zero.
        
        This setup is critical for industrial-grade stability, ensuring that 
        gradients neither explode nor vanish at the start of training.
        """
        nn.init.xavier_uniform_(self.w_query.weight)
        nn.init.xavier_uniform_(self.w_key.weight)
        nn.init.xavier_uniform_(self.w_value.weight)
        nn.init.xavier_uniform_(self.w_out.weight)
        
        if self.w_query.bias is not None:
            nn.init.constant_(self.w_query.bias, 0.)
            nn.init.constant_(self.w_key.bias, 0.)
            nn.init.constant_(self.w_value.bias, 0.)
            nn.init.constant_(self.w_out.bias, 0.)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Executes the forward pass of the Multi-Head Attention module.
        
        Note: In PyTorch, you call this by executing 'model(x)', which 
        internally triggers this method.

        The process follows these stages:
        1. Projection: Create Q, K, V matrices from the input.
        2. Splitting & Transposing: Reshape into the "God Shape" (Batch, Heads, Seq, Dim).
        3. Attention Calculation: Compute similarity scores (Q @ K^T).
        4. Softmax & Weighting: Normalize scores and multiply by V.
        5. Recombination: Concatenate all heads and apply final projection.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Seq_Length, embedding_dim).
            mask (torch.Tensor, optional): Mask to ignore specific tokens (e.g., future tokens).
                Mask values of 1 indicate tokens to be ignored.
        """
        batch_size, seq_length, _ = x.shape
        
        # 1. Projections
        q = self.w_query(x)
        k = self.w_key(x)
        v = self.w_value(x)

        # 2. Transforming to "God Shape" (B, H, S, d_k)
        # We split the embedding_dim into num_heads and move 'H' to the second dimension.
        # This allows PyTorch to process all heads in parallel as if they were a batch.
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Scaled Dot-Product Attention
        # Calculate how much each token 'resembles' every other token.
        # Shape results in (B, H, S, S)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            # Apply mask by setting ignored positions to a very large negative value.
            # After Softmax, these positions will become zero.
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 1, float("-1e9"))

        # 4. Softmax Upcasting (Stability Trick)
        # We convert to float32 before Softmax to avoid numerical overflow in FP16 contexts,
        # then cast back to the original type.
        attn_weights = torch.softmax(scores.float(), dim=-1).to(x.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Multiply weights by Values to get the contextualized representation
        context = torch.matmul(attn_weights, v)

        # 5. Recombine Heads
        # We undo the "God Shape" by transposing back and merging heads into a single vector.
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embedding_dim)
        
        # Final projection to mix information between different heads
        output = self.resid_dropout(self.w_out(context))

        return output, attn_weights


class TransformerBlock(nn.Module):
    """
    Full Transformer Encoder Block with Pre-Normalization.
    
    This is the standard building block of modern Transformer architectures. 
    It combines Multi-Head Attention and a FeedForward network, using 
    Residual Connections (Add) and Normalization (Norm) for stability.

    Architecture (Pre-Norm style):
    x -> Norm -> Attention -> Add(x) -> Norm -> FeedForward -> Add(new_x)
    """

    def __init__(self, embedding_dim: int, num_heads: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.norm1 = ManualLayerNorm(embedding_dim)
        self.attn = MultiHeadAttention(embedding_dim, num_heads, dropout=dropout)
        
        self.norm2 = ManualLayerNorm(embedding_dim)
        self.ffn = FeedForward(embedding_dim, d_ff, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Applies attention and feed-forward sub-layers with residual connections.
        
        Residual connections (x + sublayer(x)) allow the original signal to flow 
        through the deep network, preventing the "forgetting" of information.
        """
        # 1. Attention Sub-layer (with Pre-Normalization)
        attn_out, weights = self.attn(self.norm1(x), mask=mask)
        x = x + attn_out
        
        # 2. Feed-Forward Sub-layer (with Pre-Normalization)
        x = x + self.ffn(self.norm2(x))
        
        return x, weights
