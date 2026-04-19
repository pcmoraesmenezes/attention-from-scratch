import torch
import torch.nn as nn
import math



class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.1):
        """
        This class handles the implementation of a Multi Head Attention.

        Args:
            embedding_dim: Total dimension of entry vector (ex: 512)
            num_heads: Number of parallel heads (ex: 8)
            dropout: Dropout probability (default: 0.1)
        """

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
        """Standard industrial initialization."""
        nn.init.xavier_uniform_(self.w_query.weight)
        nn.init.xavier_uniform_(self.w_key.weight)
        nn.init.xavier_uniform_(self.w_value.weight)
        nn.init.xavier_uniform_(self.w_out.weight)
        
        if self.w_query.bias is not None:
            nn.init.constant_(self.w_query.bias, 0.)
            nn.init.constant_(self.w_key.bias, 0.)
            nn.init.constant_(self.w_value.bias, 0.)
            nn.init.constant_(self.w_out.bias, 0.)

    def forward(self, x, mask=None):
        """
        This method handles the forward pass of the MultiHeadAttention.

        Args:
            x: Entry shape (Batch, Seq_Length, embedding_dim)
            mask: Optional mask (B, 1, S, S) or (B, S, S). Mask values of 1 indicate tokens to ignore.
        """

        batch_size, seq_length, _ = x.shape
        q = self.w_query(x)
        k = self.w_key(x)
        v = self.w_value(x)

        q = q.view(
            batch_size,
            seq_length,
            self.num_heads,
            self.head_dim
        ).transpose(1,2)

        k = k.view(
            batch_size,
            seq_length,
            self.num_heads,
            self.head_dim
        ).transpose(1,2)

        v = v.view(
            batch_size,
            seq_length,
            self.num_heads,
            self.head_dim
        ).transpose(1,2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 1, float("-1e9"))

        attn_weights = torch.softmax(scores.float(), dim=-1).to(x.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.matmul(attn_weights, v)

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embedding_dim)
        output = self.resid_dropout(self.w_out(context))

        return output, attn_weights

if __name__ == "__main__":
    embedding_dim = 512
    num_heads = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MultiHeadAttention(embedding_dim, num_heads).to(device).to(torch.float16)
    
    x = torch.randn(1, 10, embedding_dim).to(device).to(torch.float16)
    output, _ = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Dtype: {output.dtype}")
