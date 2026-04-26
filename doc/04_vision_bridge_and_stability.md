# 04. Vision Bridge, Stability, and Explainability

This document bridges the gap between raw attention mechanics and practical applications in Computer Vision, while addressing industrial stability requirements.

## 1. The Vision Bridge: Patch Embedding

In Computer Vision, images are 2D grids of pixels $(B, C, H, W)$. Transformers, however, are designed to process **sequences** of tokens $(B, S, D)$. To bridge this gap, we use the **Patch Embedding** technique.

### The Convolutional Trick
Instead of manually slicing images into squares, we use a **2D Convolution** with:
- `kernel_size = patch_size`
- `stride = patch_size`

This operation simultaneously partitions the image into patches and projects each patch into the model dimension ($d_{model}$).

### Transformation Flow:
1.  **Input Image**: $(B, 3, 224, 224)$
2.  **Projection**: $(B, d_{model}, 14, 14)$ - *Each pixel in the 14x14 grid represents a 16x16 patch.*
3.  **Flattening**: $(B, d_{model}, 196)$ - *Converting the 2D grid into a 1D sequence.*
4.  **Transposition**: $(B, 196, d_{model})$ - *Reaching the standard sequence format for Attention.*

---

## 2. Industrial Stability: Softmax Upcasting

When training in **Mixed Precision (FP16)**, numerical stability becomes a critical challenge.

### The Problem
The Softmax function involves calculating exponents: $\exp(x)$. In `float16`, the maximum representable value is approximately **65,504**. If the attention scores before Softmax are high, $\exp(x)$ will easily overflow to `inf`, resulting in `NaN` gradients and a collapsed model.

### The Solution: Upcasting
We perform the Softmax calculation in **`float32`** and then cast the result back to the original dtype:

```python
# Upcast to float32 for numerical safety during exponentiation
attn_weights = torch.softmax(scores.float(), dim=-1).to(x.dtype)
```

This "Stability Trick" ensures that the model can handle high-variance scores without losing precision or crashing, a requirement for any industrial-grade Transformer implementation.

---

## 3. Explainable AI (XAI): Interpreting Heatmaps

Attention is not just a mathematical tool; it is an **explainability window**. By visualizing the attention weights, we can see what the model is "looking at."

### How to Read the Map
An Attention Map is an $S \times S$ matrix (for each head).
- **Rows**: The **Query** tokens (what is looking).
- **Columns**: The **Key** tokens (what is being looked at).

### Key Patterns:
- **Diagonal Dominance**: The model is focusing mostly on the token itself (Self-identity).
- **Vertical Lines**: Multiple tokens are focusing on one specific "anchor" token (e.g., a dominant object in an image or the subject of a sentence).
- **Localized Clusters**: Tokens are focusing on their neighbors (similar to a convolutional behavior).

### Multi-Head Specialization
In a Multi-Head setup, different heads learn to focus on different features:
- **Head A**: May focus on global structure (Low frequency).
- **Head B**: May focus on edges or specific textures (High frequency).
- **Head C**: May learn to identify the separation between objects.

---
> [!TIP]
> Use the provided `visualize.py` script to generate heatmaps. In a Vision Transformer, these maps will show you exactly which patches of the image were most relevant for a specific prediction.
