# DL Mastery

A structured, math-first, code-deep Deep Learning repository built for engineers who want to **truly understand** what is happening — not just run training loops.

Every concept is built in three layers: **NumPy from scratch → raw PyTorch tensors → `nn.Module` API**. You see exactly what the framework is hiding at every step.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshu231204/dl-mastery)

> **Prerequisites:** Classical ML foundations (linear/logistic regression, gradient descent, numpy). Complete [ml-mastery](https://github.com/himanshu231204/ml-mastery) first if needed.

---

## 👨‍💻 Author

**Himanshu Kumar**

- 🌐 GitHub: [@himanshu231204](https://github.com/himanshu231204)
- 💼 LinkedIn: [himanshu231204](https://www.linkedin.com/in/himanshu231204)
- 🐦 Twitter/X: [@himanshu231204](https://twitter.com/himanshu231204)
- 📧 Email: himanshu231204@gmail.com

---

## What Makes This Different

| Feature | ml-mastery | dl-mastery |
|---|---|---|
| Code layers | 2 (scratch + sklearn) | 3 (NumPy → raw PyTorch → nn.Module) |
| Visualizations | Static matplotlib | Animated interactive diagrams |
| Math coverage | Per-algorithm | Dedicated `00_math/` module |
| Training | N/A | Full diagnostic dashboard per run |
| Papers | None | Paper → Code section per topic |
| Debugging | 3 gotchas per notebook | Full debugging section with broken code |

---

## Repository Structure

```
dl-mastery/
│
├── 00_math/                          # Mathematical prerequisites
│   ├── linear_algebra.ipynb          # Vectors, matrices, eigendecomposition, SVD
│   ├── calculus_chain_rule.ipynb     # Derivatives, partial derivatives, chain rule
│   ├── probability_distributions.ipynb  # Gaussian, KL divergence, entropy
│   └── matrix_calculus.ipynb        # Jacobians, Hessians, vectorized gradients
│
├── 01_foundations/                   # Deep Learning core
│   ├── perceptron_and_ann.ipynb      # From single neuron to multi-layer network
│   ├── backpropagation.ipynb         # Full derivation + animated computation graph
│   ├── activation_functions.ipynb    # Sigmoid, ReLU, GELU — math + vanishing gradient
│   ├── optimizers.ipynb              # SGD → Momentum → RMSProp → Adam — full derivations
│   ├── regularization.ipynb         # Dropout, BatchNorm, weight decay, early stopping
│   └── pytorch_fundamentals.ipynb   # Tensors, autograd, training loop, GPU
│
├── 02_cnn/                           # Computer Vision
│   ├── convolution_operation.ipynb   # Conv math, padding, stride, receptive field
│   ├── cnn_architectures.ipynb       # LeNet → VGG → ResNet — implement residual block
│   ├── transfer_learning.ipynb       # Feature extraction vs fine-tuning, practical guide
│   └── object_detection_intro.ipynb  # Sliding window → anchor boxes → YOLO concept
│
├── 03_sequences/                     # Sequence Models
│   ├── rnn.ipynb                     # Vanishing gradient problem, BPTT derivation
│   ├── lstm_gru.ipynb                # Gate equations, cell state, forget gate math
│   └── seq2seq.ipynb                 # Encoder-decoder, beam search, teacher forcing
│
├── 04_transformers/                  # Attention and Transformers
│   ├── attention_mechanism.ipynb     # Scaled dot-product attention — full derivation
│   ├── transformer_architecture.ipynb  # Multi-head attention, positional encoding, FFN
│   └── bert_and_gpt.ipynb           # BERT masked LM vs GPT causal LM, fine-tuning
│
├── 05_generative/                    # Generative Models
│   ├── vae.ipynb                     # ELBO derivation, reparameterization trick
│   ├── gan.ipynb                     # Minimax objective, mode collapse, training tricks
│   └── dcgan.ipynb                   # Deep Convolutional GAN — implement from paper
│
├── 06_projects/                      # End-to-End Projects
│   ├── sentiment_analysis.ipynb      # LSTM → Transformer — IMDb dataset
│   ├── image_classifier.ipynb        # CNN + transfer learning — real image dataset
│   └── text_generator.ipynb          # Character-level GPT from scratch
│
├── extras/
│   ├── pytorch_cheatsheet.md        # Every PyTorch operation with examples
│   ├── dl_interview_qa.md           # 100+ DL interview questions with answers
│   └── papers_reading_list.md       # Curated paper list with reading order
│
├── requirements.txt
├── CONTRIBUTING.md
└── README.md
```

---

## Learning Path.

```
00_math/ (all 4 notebooks)
        ↓
pytorch_fundamentals → perceptron_and_ann → backpropagation
        ↓
activation_functions → optimizers → regularization
        ↓
        ├── 02_cnn/ (Vision track)
        ├── 03_sequences/ → 04_transformers/ (NLP track)
        └── Both tracks → 05_generative/
        ↓
06_projects/ (apply everything)
```


---

## Every Notebook Structure

Each notebook follows a strict 7-section format:

| Section | What you get |
|---|---|
| **Math prerequisites** | Which equations from `00_math/` this notebook uses |
| **Concept + animated diagram** | Interactive visualization of what the algorithm does |
| **NumPy from scratch** | Core computation with no framework — see every operation |
| **Raw PyTorch tensors** | Same thing with autograd — no `nn.Module` yet |
| **`nn.Module` implementation** | Production-style PyTorch code |
| **Training dashboard** | Loss, accuracy, gradient flow, weight distributions |
| **Paper → Code + Debugging + Exercises** | Implement from original paper, fix broken code |

---

## Key Design Decisions

**Why PyTorch only?** PyTorch is the standard in research and increasingly in production. One framework, learned deeply, is worth more than two frameworks learned superficially.

**Why three code layers?** Most people use `nn.Linear` without knowing it is just `x @ W.T + b`. The three-layer structure makes this explicit — you implement the same thing three times with increasing abstraction.

**Why `00_math/` first?** Every DL tutorial assumes you know matrix calculus and the chain rule in vectorized form. Most people don't. Without this, you can run code but you cannot debug gradients or read a paper.

**Why animated diagrams?** Backpropagation, attention, and convolution are dynamic processes. A static plot cannot show how a gradient signal flows backward or how a query attends to different keys. Animation is not decoration — it is the explanation.

---

## Dependencies

| Library | Version | Purpose |
|---|---|---|
| torch | ≥ 2.0 | Core deep learning framework |
| torchvision | ≥ 0.15 | Datasets, transforms, pretrained models |
| numpy | ≥ 1.24 | Scratch implementations |
| matplotlib | ≥ 3.7 | Visualization |
| tqdm | ≥ 4.65 | Training progress bars |
| scikit-learn | ≥ 1.3 | Evaluation metrics, datasets |
| transformers | ≥ 4.30 | HuggingFace — BERT, GPT notebooks |
| datasets | ≥ 2.12 | HuggingFace datasets |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for standards on adding notebooks, fixing derivations, or improving visualizations.

---

## License

MIT License — free to use, share, and build on with attribution.
