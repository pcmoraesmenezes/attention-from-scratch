import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_attention_heatmap(attn_weights: torch.Tensor, head_idx: int = 0, save_path: str = None):
    """
    Plots a heatmap for a specific head's attention weights.
    
    Args:
        attn_weights: Tensor of shape (Batch, Heads, Seq_Len, Seq_Len)
        head_idx: The index of the head to visualize.
        save_path: If provided, saves the plot to this path.
    """

    weights = attn_weights[0, head_idx].detach().cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(weights, cmap='viridis', xticklabels=False, yticklabels=False)
    plt.title(f"Attention Heatmap - Head {head_idx}")
    plt.xlabel("Key Positions")
    plt.ylabel("Query Positions")
    
    if save_path:
        plt.savefig(save_path)
        print(f"Heatmap saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    S = 20
    H = 8
    dummy_weights = torch.softmax(torch.randn(1, H, S, S), dim=-1)
    plot_attention_heatmap(dummy_weights, head_idx=0, save_path="demo_heatmap.png")
