import torch
import torch.nn as nn

class ManualLayerNorm(nn.Module):
    """
    Implementation of Layer Normalization from scratch.
    
    This component ensures that the data flowing through the network stays within a stable 
    mathematical range. It calculates the mean and standard deviation of each token's 
    features and normalizes them.
    
    Why use this? 
    Without normalization, values could explode or shrink too much during deep 
    calculations, making the model impossible to train.

    Args:
        d_model (int): The dimension of the input embeddings (e.g., 512).
        eps (float, optional): A tiny value added to the denominator to prevent 
            division by zero. Defaults to 1e-6.
    """
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        # gamma and beta are learnable parameters (nn.Parameter). 
        # They allow the model to re-scale and re-shift the normalized data 
        # to find the optimal representation for learning.
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the input tensor across its last dimension.

        Process:
        1. Calculate the mean and standard deviation of the features for each token.
        2. Subtract the mean and divide by the standard deviation (Standardization).
        3. Multiply by gamma (scale) and add beta (shift).

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Sequence, Features).

        Returns:
            torch.Tensor: The stabilized, normalized tensor.
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class FeedForward(nn.Module):
    """
    Standard Feed Forward Network (FFN) used in Transformer blocks.
    
    Think of this as a "mini-brain" that processes each word individually after 
    the Attention mechanism has gathered context from the whole sentence. 
    It expands the data to a higher dimension to allow for more complex 
    reasoning before projecting it back.

    Args:
        d_model (int): Input and output dimension (e.g., 512).
        d_ff (int, optional): Hidden layer dimension, usually 4x larger than d_model. 
            Defaults to 2048.
        dropout (float, optional): Probability of temporarily "turning off" 
            neurons to prevent the model from memorizing the training data. 
            Defaults to 0.1.
    """
    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        
        # GELU (Gaussian Error Linear Unit) is a smooth activation function that helps gradients flow 
        # better than the traditional ReLU, especially in deep Transformers.
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies two linear projections with a non-linear activation in between.

        The sequence is: Project Up -> Activate -> Regularize (Dropout) -> Project Down.
        """
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class PositionalEncoding(nn.Module):
    """
    Injects information about the order of tokens into the sequence.
    
    Since the Attention mechanism treats all tokens as a set (ignoring their order), 
    we must "mark" each token with its position in the sentence. We do this BEFORE 
    the data enters the Transformer blocks.
    
    How it works:
    It creates a unique pattern using sine and cosine waves of different frequencies. 
    This pattern is added (summed) directly to the word embeddings.

    Args:
        d_model (int): Embedding dimension (must match the input).
        dropout (float, optional): Dropout probability applied after adding 
            the positions. Defaults to 0.1.
        max_len (int, optional): Maximum sentence length supported by this 
            pre-calculated pattern. Defaults to 5000.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Pre-calculating the positional pattern (pe)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Log-space frequency calculation for better numerical precision
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        # Sine for even indices (0, 2, 4...) and Cosine for odd indices (1, 3, 5...)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        
        # register_buffer ensures 'pe' is saved with the model but NOT updated 
        # by the optimizer (it is a fixed mathematical pattern, not a learnable weight).
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds the pre-calculated positional pattern to the input embeddings.

        Args:
            x (torch.Tensor): Input embeddings (Batch, Sequence, Dimensions).

        Returns:
            torch.Tensor: Embeddings now "aware" of their position in the sequence.
        """
        # Summing the positional pattern to the input data
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
