import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """Translates images into sequences of patches.

    Args:
        img_size (int, optional): Height/Width of the square input image. Defaults to 224.
        patch_size (int, optional): Height/Width of each square patch. Defaults to 16.
        in_channels (int, optional): Number of input channels (RGB). Defaults to 3.
        embed_dim (int, optional): The projection dimension (d_model). Defaults to 768.
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # We use a convolution to partition the image and project it
        self.projection = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projects and flattens the image into patches.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Tensor of shape (B, num_patches, embed_dim).
        """
        # (B, C, H, W) -> (B, E, H/P, W/P)
        x = self.projection(x)
        
        # (B, E, H/P, W/P) -> (B, E, num_patches)
        x = x.flatten(2)
        
        # (B, E, num_patches) -> (B, num_patches, E)
        x = x.transpose(1, 2)
        
        return x

if __name__ == "__main__":
    # Smoke test
    p_embed = PatchEmbedding(img_size=224, patch_size=16, embed_dim=768)
    fake_img = torch.randn(1, 3, 224, 224)
    out = p_embed(fake_img)
    print(f"Vision Input shape: {fake_img.shape}")
    print(f"Vision Embedded shape: {out.shape}") # Expect (1, 196, 768)
