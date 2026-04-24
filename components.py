import torch
import torch.nn as nn

class ManualLayerNorm(nn.Module):
    """Implementation of Layer Normalization from scratch.

    Args:
        d_model (int): The dimension of the input embeddings.
        eps (float, optional): Small value to avoid division by zero. Defaults to 1e-6.
    """
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies layer normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, S, D).

        Returns:
            torch.Tensor: Normalized tensor.
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class FeedForward(nn.Module):
    """Standard Feed Forward Network used in Transformer blocks.

    Args:
        d_model (int): Input and output dimension.
        d_ff (int, optional): Hidden layer dimension. Defaults to 2048.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
    """
    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the feed-forward network to the input."""
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class PositionalEncoding(nn.Module):
    """Implements Sinusoidal Positional Encoding for sequence awareness.

    Args:
        d_model (int): Embedding dimension.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
        max_len (int, optional): Maximum sequence length supported. Defaults to 5000.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        
        # position (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Precompute frequencies in log space for numerical stability
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Shape: (1, max_len, d_model) for broadcasting
        pe = pe.unsqueeze(0)
        
        # Register as buffer to ensure it moves with the model device but is not trainable
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Injects positional information into the input embeddings.

        Args:
            x (torch.Tensor): Input embeddings of shape (Batch, Seq, Dim).

        Returns:
            torch.Tensor: Embeddings with positional information added.
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
