# 03.1 Deep Dive: The Mathematics of Attention

This document explains the Multi-Head Attention implementation from the perspective of Linear Algebra and Tensor Geometry.

## 1. The Pieces on the Board (Initialization)

### `nn.Linear` and its Matrix Nature
When we define `self.w_query = nn.Linear(embed_dim, embed_dim)`, we are creating a **Weight Matrix** $W_Q$.

**Mathematical Representation:**
$$Q = X \cdot W_Q^T + b$$

Where:
- $X$: Input matrix with shape $(Batch, Seq, Embed\_Dim)$.
- $W_Q$: Learnable projection matrix with shape $(Embed\_Dim, Embed\_Dim)$.
- $b$: Bias vector, which allows the network to shift the function in space.

**Why project?**
The input $X$ is the same for all three layers ($Q, K, V$). However, by multiplying it by different weight matrices ($W_Q, W_K, W_V$), we extract different "features" from the same data:
- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What do I have to offer?"
- **Value (V)**: "The actual information to be transmitted."

---

## 2. Tensor Management (The God Shape)

### `view()`: Geometric Reorganization
The `view` function does not change the data in memory; it only changes how PyTorch interprets the dimensions.

**Logic:**
We treat the vector of size 512 as a block of $8 \times 64$.
$$(Batch, Seq, 512) \xrightarrow{\text{view}} (Batch, Seq, 8, 64)$$

### `transpose(1, 2)`: The Leap to Parallelism
Here, we swap the Sequence axis ($S$) with the Heads axis ($H$).

**Visual Representation:**
- **Before (1, 10, 8, 64)**: 10 words, each with 8 pieces.
- **After (1, 8, 10, 64)**: 8 heads, each seeing the entire sentence (10 words).

This is vital because PyTorch processes the batch dimension and the subsequent dimension in parallel. By moving the 8 to the front, each head becomes an independent "mini-batch."

---

## 3. The Engine (Scaled Dot-Product)

### `matmul()` (Matrix Multiplication)
The `matmul` function performs the dot product between matrices. In our case:
$$\text{Scores} = Q \times K^T$$

$$(B, H, S, d_k) \times (B, H, d_k, S) = (B, H, S, S)$$

**Why transpose $K$?**
In linear algebra, to multiply a matrix $A(n \times m)$ by $B$, matrix $B$ must have shape $(m \times p)$. Since $Q$ and $K$ both have shape $(S \times d_k)$, we must rotate $K$ so the internal dimensions ($d_k$) match.

The result is an $S \times S$ matrix (Sequence by Sequence). Each cell $(i, j)$ in this matrix represents: **"How much does word $i$ resemble word $j$?"**

### Scaling: $1 / \sqrt{d_k}$
If $d_k$ is large, the result of the dot product can explode. This pushes the Softmax into regions where the derivative is zero (**Vanishing Gradient**). Dividing by the square root of $d_k$ keeps the data variance near 1, keeping learning active.

---

## 4. Recombination and Output

### `context.transpose(1, 2).contiguous().view(...)`
We reverse the process. The 8 heads have finished their work and now need to "reunite."
1.  **`transpose`**: Returns to $(B, S, H, d_k)$.
2.  **`contiguous`**: Since we rotated the matrix, the data in memory is no longer sequential. `contiguous` creates an organized copy so that `view` can merge the axes.
3.  **`view`**: Merges $H \times d_k$ back into $Embed\_Dim$ (512).

### The Role of `self.w_out`
Finally, we have `output = self.w_out(context)`. 
Even though we've concatenated the heads, they are still "islands" of information. The $W_O$ matrix acts as a diplomat: it multiplies the final result so that information from all heads is mixed and refined before leaving the module.

---

## 5. Spatial Awareness: Positional Encoding

Since the attention mechanism is agnostic to order (permutation invariant), we must inject the notion of position.

### The Thesis of Frequencies
We use sinusoidal functions of different frequencies for each dimension of the embedding. This creates a unique pattern for each position that the model can learn to decode.

**Frequency Vector (Div Term):**
$$temp = \exp(i \times -\log(10000) / d_{model})$$

Where $i$ iterates through only the even indices of the model dimension. This generates a geometric progression of wavelengths, ranging from $2\pi$ to $10000 \cdot 2\pi$.

### Implementation via `register_buffer`
Unlike the $W_Q, W_K, W_V$ matrices, the Positional Encoding is not trained via backpropagation (in the original paper). In PyTorch, we use `self.register_buffer('pe', pe)` so that:
1.  The tensor is not considered a parameter (no gradient is computed).
2.  The tensor is automatically moved between CPU/GPU along with the model.
3.  The tensor is saved in the model's `state_dict`.
