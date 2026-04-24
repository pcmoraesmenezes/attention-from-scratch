import torch
import torch.nn as nn
import math
from components import ManualLayerNorm, FeedForward



class MultiHeadAttention(nn.Module):
    """Implementation of the Multi-Head Attention mechanism from scratch.

    Args:
        embedding_dim (int): Total dimension of the input vector (d_model).
        num_heads (int): Number of parallel attention heads.
        dropout (float): Dropout probability for attention weights and output.
    """

    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.w_query = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.w_key = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.w_value = nn.Linear(embedding_dim, embedding_dim, bias=True)

        self.w_out = nn.Linear(embedding_dim, embedding_dim, bias=True)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initializes weights using Xavier Uniform and biases to zero."""
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
        """Executes the forward pass of the Multi-Head Attention module.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Seq_Length, embedding_dim).
            mask (torch.Tensor, optional): Mask to ignore specific tokens. Shape (B, S, S) or (B, 1, S, S).
                Mask values of 1 indicate tokens to be ignored.

        Returns:
            tuple: (Output tensor of shape (B, S, D), Attention weights of shape (B, H, S, S)).
        """
        batch_size, seq_length, _ = x.shape
        q = self.w_query(x)
        k = self.w_key(x)
        v = self.w_value(x)

        # Proj -> Split -> Transpose to God Shape (B, H, S, d_k)
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 1, float("-1e9"))

        # Stability Trick: Upcast to float32 before Softmax in FP16 contexts
        attn_weights = torch.softmax(scores.float(), dim=-1).to(x.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.matmul(attn_weights, v)

        # Recombine heads and project
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embedding_dim)
        output = self.resid_dropout(self.w_out(context))

        return output, attn_weights

class TransformerBlock(nn.Module):
    """Full Transformer Encoder Block with Pre-Normalization.

    Args:
        embedding_dim (int): Input and output dimension.
        num_heads (int): Number of attention heads.
        d_ff (int): Hidden dimension of the FeedForward network.
        dropout (float): Dropout probability.
    """

    def __init__(self, embedding_dim: int, num_heads: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.norm1 = ManualLayerNorm(embedding_dim)
        self.attn = MultiHeadAttention(embedding_dim, num_heads, dropout=dropout)
        
        self.norm2 = ManualLayerNorm(embedding_dim)
        self.ffn = FeedForward(embedding_dim, d_ff, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """Applies attention and feed-forward sub-layers with residual connections.

        Args:
            x (torch.Tensor): Input tensor.
            mask (torch.Tensor, optional): Attention mask.

        Returns:
            tuple: (Processed tensor, Attention weights).
        """
        # 1. Attention (Pre-Norm)
        attn_out, weights = self.attn(self.norm1(x), mask=mask)
        x = x + attn_out
        
        # 2. Feed-Forward (Pre-Norm)
        x = x + self.ffn(self.norm2(x))
        
        return x, weights

if __name__ == "__main__":
    embedding_dim = 512
    num_heads = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MultiHeadAttention(embedding_dim, num_heads).to(device).to(torch.float16)
    
    x = torch.randn(1, 10, embedding_dim).to(device).to(torch.float16)
    output, _ = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Dtype: {output.dtype}")
