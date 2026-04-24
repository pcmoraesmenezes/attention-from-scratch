import torch
from vision_layers import PatchEmbedding
from attention import TransformerBlock
from components import PositionalEncoding
from visualize import plot_attention_heatmap

def test_full_pipeline():
    # 1. Config
    img_size = 224
    patch_size = 16
    embed_dim = 768
    num_heads = 12
    batch_size = 1
    
    print("--- Testing Full Vision Transformer Pipeline (Bottom-Up) ---")
    
    # 2. Vision Bridge: Patch Embedding
    patch_embed = PatchEmbedding(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
    fake_img = torch.randn(batch_size, 3, img_size, img_size)
    
    patches = patch_embed(fake_img)
    print(f"1. Patch Embedding Output Shape: {patches.shape}") # Expect (1, 196, 768)
    
    # 3. Positional Encoding
    pos_enc = PositionalEncoding(d_model=embed_dim, max_len=patch_embed.num_patches)
    patches = pos_enc(patches)
    print(f"2. Positional Encoding Added (Shape maintained: {patches.shape})")
    
    # 4. Transformer Block
    block = TransformerBlock(embedding_dim=embed_dim, num_heads=num_heads)
    output, weights = block(patches)
    
    print(f"3. Transformer Block Output Shape: {output.shape}") # Expect (1, 196, 768)
    print(f"4. Attention Weights Shape: {weights.shape}")     # Expect (1, 12, 196, 196)
    
    # 5. Visualization
    print("5. Generating Heatmap...")
    plot_attention_heatmap(weights, head_idx=0, save_path="vision_attention_heatmap.png")
    
if __name__ == "__main__":
    test_full_pipeline()
