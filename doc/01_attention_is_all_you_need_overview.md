# 01. Attention Is All You Need: Architecture Overview

## Context

Previously, the State of the Art (SOTA) in the market was dominated by **Recurrent Neural Networks (RNNs)**. These networks often incorporated an attention layer, forming an **RNN + Attention** architecture.

The fundamental limitation of this architecture was its sequential nature: predictions depended on the previous input. For instance, to compute hidden state $h_3$, you needed $h_2$, which in turn required $h_1$. This sequential dependency makes parallelization impossible, leading to high computational costs.

This chain makes training expensive and introduces the **Vanishing Gradient** problem, which causes RNNs to "forget" information in long sequences. For example, when training on a 100-word sentence, the GPU is forced to perform 100 sequential steps. By the time it reaches word 100, the signal from word 1 has often degraded into noise.

Thus, we faced two major challenges:
1.  **Computational Efficiency**: Lack of parallelization.
2.  **Long-Range Degradation**: Difficulty in maintaining context over long chains.

---

The goal of reducing sequential processing costs was also pursued in other architectures, such as **Convolutional Neural Networks (CNNs)**. While CNNs allow for parallelization, they require stacking many layers to capture long-range dependencies because each convolutional kernel only sees a local neighborhood. This makes capturing relationships between distant tokens linearly or logarithmically difficult, often leading to vanishing gradients and information loss.

---

## 1. The Mechanism: Vector Similarity

Before the architecture comes the mechanics. **Attention** is essentially a **Vector Similarity** mechanism. By calculating the dot product $Q \cdot K^T$, the model measures how close one word is to another in the vector space. The result is a "score" that determines how much of each **Value (V)** will be absorbed. In simple terms: Attention allows each word to "rewrite" itself based on the most important information around it, acting like an intelligent search in a dynamic database.

## 2. The Thesis: The End of Recurrence

The breakthrough of the "Attention Is All You Need" paper is the proposal that we don't need RNNs or CNNs. The key insight is replacing sequential dependencies with pure attention layers. This solves the forgetting problem in long chains and allows the GPU to process all tokens simultaneously.

## 3. The Major Shift: Cross-Attention

The architecture is organized into an **Encoder** and a **Decoder**, and the connection between them is the secret to its success:

*   **The Encoder (The Understander)**: Processes the input and generates a "Meaning Map" (Key/Value pairs). It uses **Self-Attention** to understand how the words in the input relate to each other.
*   **The Decoder (The Generator)**: This is where **Cross-Attention** occurs. The Decoder uses its **Query (Q)**—what it is currently trying to generate—to "search" the map built by the Encoder.
*   **The "Magic"**: This interaction allows the Decoder to "point a laser" at the exact word in the input needed to generate the next word in the output. It’s a constant dialogue between the component that understood the context and the one creating the response.

## 4. The Final Anatomy

The flow continues with **Positional Encodings** (to maintain order in the parallel chaos), **Add & Norm** connections (to prevent the network from forgetting the original signal), and **Feed Forward** networks (to refine the final result). The process concludes with a **Softmax** layer, converting raw mathematics into probabilities for real words.

---

## Summary of the Transformer Architecture

The Transformer architecture removes sequential dependencies by using global connections through the Attention mechanism. Unlike older models, the Transformer can look at all words at once, achieving a constant path cost of $O(1)$.

The process works as follows:

1.  **Embeddings & Positional Encoding**: Input embeddings are summed with Positional Encodings to indicate the exact position of each token.
2.  **Multi-Head Self-Attention**: This layer fragments processing into smaller sub-spaces (heads), allowing the model to learn different types of relationships (e.g., grammar and meaning) simultaneously.
3.  **Add & Norm**: The "Add" is a residual connection (a shortcut) that ensures the original signal is preserved, while "Norm" (Layer Normalization) stabilizes the mathematical magnitude.
4.  **Feed Forward Network**: Refines the information of each word individually.
5.  **Encoder-Decoder Interaction**: The Encoder generates **Keys (K)** and **Values (V)**, which are fed into the **Cross-Attention** layer of the Decoder.
6.  **Causal Masking (Shifted Right)**: The Decoder uses a masking technique to ensure it only learns to predict the next token based on the past, preventing it from "cheating" by looking at future tokens.
7.  **Final Output**: Through a Linear layer and a Softmax function, the results are converted into probabilities, defining which word will be chosen.

This structure is typically repeated for $N=6$ layers in both the Encoder and Decoder.