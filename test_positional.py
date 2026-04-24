import torch
import matplotlib.pyplot as plt
from components import PositionalEncoding

def test_positional_encoding():
    d_model = 512
    max_len = 100
    dropout = 0.0
    
    pos_enc = PositionalEncoding(d_model, dropout, max_len)
    
    x = torch.zeros(1, max_len, d_model)
    
    # forward
    pe_output = pos_enc(x).squeeze(0)
    
    print(f"PE Output Shape: {pe_output.shape}")
    
    plt.figure(figsize=(12, 8))
    plt.pcolormesh(pe_output.numpy(), cmap='RdBu')
    plt.xlabel('Embedding Dimension (d_model)')
    plt.ylabel('Sequence Position (pos)')
    plt.colorbar(label='PE Value')
    plt.title('Sinusoidal Positional Encoding Pattern')
    
    save_path = "positional_encoding_pattern.png"
    plt.savefig(save_path)
    print(f"Pattern visualization saved to {save_path}")

    # Mathematical Verification
    # Check if sine and cosine are alternating
    # pe[:, 0::2] is sin, pe[:, 1::2] is cos
    # At pos 0, sin(0) = 0, cos(0) = 1
    assert torch.allclose(pe_output[0, 0], torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(pe_output[0, 1], torch.tensor(1.0), atol=1e-6)
    print("Logic Check: pos=0 (sin=0, cos=1) passed.")

if __name__ == "__main__":
    test_positional_encoding()
