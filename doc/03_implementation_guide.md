# 03. Implementation Guide: Phases 1 and 2 (Foundation and Mechanics)

This guide consolidates the technical instructions for the manual implementation of the Multi-Head Attention mechanism, focusing on stability for consumer hardware (**FP16**) and adherence to **Industrial Standards**.

## 🎯 Technical Objectives
- **Zero Loops**: All tensor manipulation must be done via axes (view/transpose).
- **Parity**: Implementation compatible with `nn.MultiheadAttention` (bias=True).
- **Efficiency**: Optimized for `torch.float16`.

---

## 🏗️ Phase 1: Matrix Foundation (The Blueprint)

In this phase, we transform the input $X$ into Query, Key, and Value spaces.

### 1. Initialization and Weights
- **DType**: Use `torch.float16` to save VRAM.
- **Bias**: Enabled (`bias=True`) to ensure absolute parity with PyTorch.
- **Xavier/Glorot Initialization**: Essential for gradient stability in deep networks. Initialize weights using `nn.init.xavier_uniform_` and biases to zero.
- **Dropout**: Include `attn_dropout` (applied to attention weights) and `resid_dropout` (applied after the output projection).
- **Scaling Factor**: $\sqrt{d_k}$ where $d_k = d_{model} / h$.

### 2. The "God Shape" (Split Heads)
The crucial transformation for parallelism:
1. `(B, S, d_model)` → Linear Projection → `(B, S, d_model)`
2. `(B, S, d_model)` → Reshape → `(B, S, h, d_k)`
3. `(B, S, h, d_k)` → Transpose(1, 2) → **`(B, h, S, d_k)`**

---

## ⚙️ Phase 2: Scaled Dot-Product (The Engine)

### 1. Similarity (Dot Product)
Multiply $Q$ by $K^T$. In PyTorch:
```python
# q: (B, h, S, d_k)
# k: (B, h, S, d_k)
attn_scores = torch.matmul(q, k.transpose(-2, -1)) # Result: (B, h, S, S)
```

### 2. FP16 Stability and Masking
To prevent the Softmax from losing precision or causing NaNs, and to ignore unwanted tokens:
- **Causal/Padding Mask**: Apply before Softmax by replacing values with $-\infty$ (or `-1e9`).
- **Softmax Upcasting**: Always perform Softmax in `float32` to avoid numerical instability in `float16`.

```python
# q, k: (B, h, S, d_k)
scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

if mask is not None:
    scores = scores.masked_fill(mask == 1, float("-1e9"))

# Precision Trick: .float() before softmax
attn_weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
attn_weights = self.attn_dropout(attn_weights)
```

### 3. Contextualization and Fusion
```python
# Multiply by Values: (B, h, S, S) @ (B, h, S, d_k) -> (B, h, S, d_k)
context = torch.matmul(attn_weights, v)

# Merge Heads: (B, h, S, d_k) -> (B, S, h, d_k) -> (B, S, d_model)
context = context.transpose(1, 2).contiguous().view(B, S, d_model)

# Final Projection + Residual Dropout
output = self.resid_dropout(self.w_out(context))
```

---

## 🏁 Definition of Done (DoD)
1.  **Shape Integrity**: The final tensor must return to the original shape `(B, S, d_model)`.
2.  **Zero Python Loops**: No `for` loops in head processing.
3.  **Half Precision**: The entire module must operate in `self.half()`.

---
> [!TIP]
> **Parity Sniper**: After implementation, create a script that instantiates both your module and PyTorch's native one, copies the weights (`state_dict`) from one to the other, and compares the output. The Mean Squared Error (MSE) should be near zero.
