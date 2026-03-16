# Deep Learning Interview Q&A

Concise, precise answers to the most commonly asked DL interview questions.
For ML fundamentals (bias-variance, regularization, evaluation), see [ml-mastery interview_qa.md](https://github.com/himanshu231204/ml-mastery/blob/main/extras/interview_qa.md).

---

## 👨‍💻 Author — Himanshu Kumar
- 🌐 GitHub: [@himanshu231204](https://github.com/himanshu231204)
- 💼 LinkedIn: [himanshu231204](https://www.linkedin.com/in/himanshu231204)

---

## Neural Network Fundamentals

**Q: What is a neural network? How does a single neuron compute its output?**
A neural network is a stack of parameterized layers that transform an input into an output through repeated application of linear transformations followed by non-linear activations. A single neuron computes z = w·x + b (linear combination), then applies an activation: a = f(z). Without the non-linearity f, any number of stacked layers collapses to a single linear transformation — the activation is what gives neural networks their expressive power.

**Q: What is backpropagation? Derive the weight update rule.**
Backpropagation applies the chain rule of calculus to compute ∂L/∂w for every weight in the network. Forward pass: compute all activations. Backward pass: starting from the loss, propagate the error signal δ^l = (∂L/∂z^l) backward through each layer using δ^{l-1} = (W^l)^T δ^l ⊙ f'(z^{l-1}). Weight gradient: ∂L/∂W^l = δ^l (a^{l-1})^T. Update: W := W - α × ∂L/∂W. The key insight is that gradients from earlier layers are computed by multiplying gradients from later layers — which creates the vanishing/exploding gradient problem.

**Q: What is the vanishing gradient problem? How do you fix it?**
In deep networks, backpropagation multiplies gradients through many layers. If activations like sigmoid have derivatives bounded in (0, 0.25), multiplying 100 such values gives a gradient of ~10^{-60} — vanishes to zero. Early layers stop learning. Fixes: (1) ReLU activations — gradient is 1 for positive inputs, doesn't saturate. (2) Batch normalization — normalizes activations, prevents them from entering saturation zones. (3) Residual connections (skip connections) — provide a gradient highway that bypasses deep layers. (4) Careful weight initialization (Xavier/He).

**Q: What is the exploding gradient problem? How do you fix it?**
When weights are large or many layers have large gradients, the backward signal grows exponentially and causes training instability (NaN loss, oscillating training curves). Fix: gradient clipping — rescale the gradient vector when its norm exceeds a threshold: if ||g|| > threshold, g = g × (threshold / ||g||). This preserves the direction of the gradient but bounds its magnitude.

**Q: Why do we need non-linear activations?**
Without non-linearity, stacking any number of linear layers W_n × ... × W_2 × W_1 × x is equivalent to a single linear transformation (W_n × ... × W_1) × x — no matter how deep the network, it can only represent linear functions. Non-linear activations allow the network to learn and compose non-linear transformations, which is what enables neural networks to approximate any continuous function (Universal Approximation Theorem).

---

## Activation Functions

**Q: Compare ReLU, Leaky ReLU, and GELU.**
ReLU: f(x) = max(0, x) — fast, sparse activations, but "dying ReLU" problem (neurons permanently output 0 if input is always negative). Leaky ReLU: f(x) = x if x > 0 else αx (α ≈ 0.01) — allows small gradient for negative inputs, prevents dying neurons. GELU: f(x) = x × Φ(x) where Φ is the Gaussian CDF — smooth, stochastic interpretation, used in BERT/GPT because it performs better on NLP tasks. In practice: use ReLU for CNNs, GELU for transformers.

**Q: What is the dying ReLU problem and how do you detect it?**
If a neuron consistently receives negative inputs (e.g., due to large negative bias or large learning rate), its output is always 0 and its gradient is always 0 — the neuron never updates and is "dead". Detection: inspect activation distributions after training. If a large fraction of activations are exactly 0 across many inputs, neurons may be dying. Fix: use Leaky ReLU, initialize biases to small positive values, reduce learning rate.

**Q: Why is softmax used for multi-class output?**
Softmax converts a vector of raw scores (logits) into probabilities that sum to 1: softmax(z)_k = exp(z_k) / Σ exp(z_j). It preserves the ranking of logits (monotonic transformation) while producing a valid probability distribution. The exponential function amplifies differences between scores — a small difference in logits becomes a large difference in probabilities, making the output more decisive.

---

## Optimization

**Q: What is the difference between SGD, SGD with momentum, and Adam?**
SGD: w := w - α × g. Updates are noisy (single batch gradient), may oscillate in narrow valleys. SGD + Momentum: accumulates a velocity v = βv + g, then w := w - α × v. Dampens oscillations across high-curvature directions, accelerates along low-curvature directions. Adam: maintains per-parameter running estimates of the first moment (mean gradient) m and second moment (uncentered variance) v. Update: w := w - α × m̂ / (√v̂ + ε). Adapts the learning rate per parameter — large gradients get smaller effective learning rates, sparse parameters get larger updates.

**Q: Derive the Adam update rule.**
At step t: m_t = β₁m_{t-1} + (1-β₁)g_t (first moment). v_t = β₂v_{t-1} + (1-β₂)g_t² (second moment). Bias-corrected: m̂_t = m_t/(1-β₁^t), v̂_t = v_t/(1-β₂^t) (corrects initialization bias when m,v start at 0). Update: w_t = w_{t-1} - α × m̂_t / (√v̂_t + ε). Default values: β₁=0.9, β₂=0.999, ε=1e-8, α=1e-3.

**Q: What is learning rate warmup and why is it used?**
Warmup linearly increases the learning rate from 0 (or a small value) to the target learning rate over the first N steps. Without warmup, Adam can make large updates in early training when the second moment estimate v is unreliable (close to 0, giving a very large effective step size). Warmup prevents these destabilizing early updates. Especially important for transformers — the original "Attention Is All You Need" paper used a warmup schedule.

**Q: What is weight decay and how does it relate to L2 regularization?**
Weight decay directly subtracts a fraction of the weights at each step: w := w - α × (g + λw) = w(1-αλ) - α×g. This is identical to L2 regularization in standard SGD. However in Adam, L2 regularization applies the weight penalty through the gradient (scaled by the second moment), which is not the same as decoupled weight decay (AdamW). AdamW decouples the weight decay from the gradient update: w := w(1-αλ) - α × m̂/√v̂. Use AdamW for transformers.

---

## Regularization

**Q: How does batch normalization work? What does it fix?**
BatchNorm normalizes the pre-activation outputs across the mini-batch: x̂ = (x - μ_B) / √(σ_B² + ε), then applies learnable scale and shift: y = γx̂ + β. During inference, uses running statistics collected during training. Fixes: (1) internal covariate shift — distribution of activations changes during training, making optimization harder. (2) Allows higher learning rates. (3) Reduces sensitivity to initialization. (4) Mild regularization effect. Note: LayerNorm normalizes across features (not batch) — used in transformers because batch statistics are unstable for variable-length sequences.

**Q: How does dropout work during training vs inference?**
During training: randomly zero out each neuron's activation with probability p. Remaining activations are scaled by 1/(1-p) to maintain expected output magnitude. During inference: no dropout applied, all neurons active. Dropout effectively trains an ensemble of 2^n sub-networks (where n = number of dropout-eligible neurons) by sampling a different sub-network each forward pass. Regularizes by preventing co-adaptation (neurons cannot rely on specific other neurons always being present).

---

## CNNs

**Q: What does a convolutional layer compute?**
A 2D convolution slides a filter (kernel) W of shape (k, k, C_in) across the input feature map of shape (H, W, C_in) and computes the dot product at each position: output[i,j] = Σ_c Σ_m Σ_n input[i+m, j+n, c] × W[m, n, c] + b. The output has shape (H', W', C_out) where H' = (H - k + 2p)/s + 1. Each filter learns to detect a specific feature (edge, texture). Multiple filters produce multiple channels in the output.

**Q: What is the receptive field and why does it matter?**
The receptive field of a neuron is the region of the input image that influences its output. In layer 1 with a 3×3 filter, each neuron sees a 3×3 patch. In layer 2, each neuron sees a 5×5 region of the original input. Receptive field grows with depth. Two 3×3 layers have the same receptive field as one 5×5 layer but fewer parameters (2×9×C² vs 25×C²) and more non-linearity. This is why VGGNet stacked small filters.

**Q: What is the residual connection in ResNet and why does it work?**
A residual block computes F(x) + x instead of just F(x). The skip connection adds the input directly to the output. Two reasons it works: (1) Gradient flow — during backprop, the gradient flows through the skip connection without passing through the block's layers, providing a "gradient highway" to early layers. (2) Optimization — it is easier to learn a residual mapping F(x) = H(x) - x than the full function H(x). If a block is not needed, F(x)≈0 is easier to achieve than F(x)≈x.

**Q: What is transfer learning? When does fine-tuning outperform feature extraction?**
Feature extraction: freeze all pretrained layers, train only a new output head. Fine-tuning: update all (or some) layers with a small learning rate. Fine-tuning outperforms feature extraction when: your dataset is large enough to benefit from it (>10k samples), your domain is different from the pretraining domain (ImageNet features may not generalize well to medical images), and you have sufficient compute. Feature extraction is preferred when: dataset is small, or target domain is similar to pretraining domain.

---

## RNNs and LSTMs

**Q: What is the vanishing gradient problem in RNNs specifically?**
In an RNN unrolled over T time steps, the gradient of the loss at step T with respect to the hidden state at step t involves the product of T-t Jacobians of the hidden state transition: ∂h_T/∂h_t = Π_{i=t}^{T-1} ∂h_{i+1}/∂h_i. If the spectral radius of the recurrent weight matrix W_h is less than 1, this product shrinks exponentially with T-t. Gradients from early time steps vanish — the network cannot learn long-range dependencies.

**Q: How does the LSTM cell state solve the vanishing gradient problem?**
The LSTM maintains a cell state c_t that flows through time with only element-wise multiplication by the forget gate f_t ∈ (0,1). The gradient of c_t with respect to c_{t-1} is just f_t — a single multiplication instead of a full matrix product. When f_t ≈ 1 (forget gate open), gradients flow through time steps without shrinking. The cell state acts as a gradient highway, similar to the residual connection in ResNets.

**Q: What is the difference between LSTM and GRU?**
LSTM: has three gates (forget, input, output) and a separate cell state c_t — more expressive, more parameters. GRU: merges cell state and hidden state, uses only two gates (reset, update) — simpler, fewer parameters, faster. In practice, GRU performs similarly to LSTM on most tasks but with fewer parameters. Use LSTM when the sequence is very long or the problem requires fine-grained memory control. Use GRU for faster training on moderate-length sequences.

---

## Transformers and Attention

**Q: Derive the scaled dot-product attention formula.**
Given queries Q ∈ R^{n×d_k}, keys K ∈ R^{m×d_k}, values V ∈ R^{m×d_v}: Attention(Q,K,V) = softmax(QK^T / √d_k) V. Step by step: (1) Compute similarity scores: QK^T ∈ R^{n×m} — each query-key dot product. (2) Scale by √d_k — without scaling, dot products grow large with d_k, pushing softmax into saturation zones with near-zero gradients. (3) Softmax — convert scores to probabilities (attention weights). (4) Weighted sum of values — output is a weighted combination of value vectors.

**Q: What is multi-head attention? Why use multiple heads?**
Multi-head attention runs h independent attention operations in parallel, each with different learned projections: head_i = Attention(QW_i^Q, KW_i^K, VW_i^V). Outputs are concatenated and projected: MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O. Multiple heads allow the model to jointly attend to information from different representation subspaces at different positions simultaneously. One head might attend to syntactic relationships while another attends to semantic relationships.

**Q: What is positional encoding and why is it needed?**
Self-attention is permutation-equivariant — if you shuffle the input tokens, the output shuffles the same way. Unlike RNNs, attention has no inherent notion of position. Positional encoding adds position information to the token embeddings. The original transformer uses sinusoidal encodings: PE(pos, 2i) = sin(pos/10000^{2i/d_model}), PE(pos, 2i+1) = cos(pos/10000^{2i/d_model}). Different frequencies allow the model to learn both absolute and relative position. Modern models often use learned positional embeddings or RoPE (Rotary Position Embedding).

**Q: What is the difference between BERT and GPT?**
BERT: bidirectional transformer encoder. Trained with masked language modeling (predict randomly masked tokens using both left and right context). Good for understanding tasks (classification, NER, QA). Cannot generate text autoregressively. GPT: unidirectional transformer decoder. Trained with causal language modeling (predict next token using only left context, future tokens masked). Good for generation tasks. Can be prompted for many tasks without task-specific fine-tuning. Key difference: BERT sees full context, GPT sees only past context.

---

## Generative Models

**Q: What is the ELBO in VAEs and why do we maximize it?**
The VAE maximizes the Evidence Lower BOund: ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z)). The true objective is to maximize log p(x) = log ∫p(x|z)p(z)dz, which is intractable (integral over all latent codes). The ELBO is a lower bound: log p(x) ≥ ELBO. Maximizing the ELBO is equivalent to: (1) reconstruction term E_q[log p(x|z)] — decoder should reconstruct x well, (2) KL divergence term — encoder posterior q(z|x) should be close to the prior p(z)=N(0,I).

**Q: What is the reparameterization trick and why is it needed?**
To train the VAE encoder with backpropagation, we need to compute gradients through the sampling step z ~ q(z|x) = N(μ, σ²). Direct sampling is not differentiable — we cannot backpropagate through a random node. The reparameterization trick rewrites z = μ + σ⊙ε where ε ~ N(0,I). Now the randomness (ε) is separate from the parameters (μ, σ). Gradients flow through μ and σ normally, while ε is treated as a fixed (though randomly sampled) constant.

**Q: What is the GAN training objective? What is mode collapse?**
GAN trains a generator G and discriminator D with a minimax game: min_G max_D E[log D(x)] + E[log(1-D(G(z)))]. D tries to distinguish real from fake; G tries to fool D. Mode collapse: the generator learns to produce only a few (or one) types of outputs that fool the discriminator, ignoring most of the data distribution. The generator finds a "mode" that always gets a high score and gets stuck there. Fixes: Wasserstein GAN (different loss), minibatch discrimination, spectral normalization, progressive growing.

---

## Debugging Deep Learning

**Q: Your training loss is NaN from the first step. What do you check?**
In order: (1) Learning rate too large — the first update overshoots catastrophically. Try 10x smaller. (2) Input data contains NaN or Inf — check with `torch.isnan(X).any()`. (3) Log of zero — if using log loss with 0 probabilities. Add ε clamp. (4) Division by zero — batch normalization with zero variance on a single-sample batch. (5) Exploding gradients — add gradient clipping. (6) Wrong loss function — e.g., using BCE on multi-class logits instead of CrossEntropyLoss.

**Q: Training loss decreases but validation loss increases. What do you do?**
Classic overfitting. In order: (1) Add dropout (start with p=0.3-0.5). (2) Increase weight decay. (3) Reduce model capacity (fewer layers/units). (4) Add data augmentation. (5) Collect more training data. (6) Use early stopping — stop training when val loss stops improving. (7) Check for data leakage — verify train/val splits are clean.

**Q: Loss is decreasing but accuracy is not improving. What is happening?**
The model is improving its probability estimates but not changing the predicted class. Usually means: (1) model stuck predicting one class with moderate confidence (class imbalance), (2) learning rate too small — loss decreases slowly, predictions barely change, (3) wrong metric — loss decreasing for a multi-class problem but accuracy computed on the wrong dimension.
