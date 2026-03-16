# DL Papers Reading List

Curated reading list ordered by topic and difficulty. For each paper: what it introduces, why it matters, and which notebook in this repo implements it.

---

## 👨‍💻 Author — Himanshu Kumar
- 🌐 GitHub: [@himanshu231204](https://github.com/himanshu231204)
- 💼 LinkedIn: [himanshu231204](https://www.linkedin.com/in/himanshu231204)

---

## How to Read a Paper

1. Read the abstract and conclusion first — understand what they claim
2. Look at all figures — they usually contain the key idea visually
3. Read the introduction — understand the problem and motivation
4. Read the method section — understand the technical contribution
5. Skip the proofs on first pass — come back to them later
6. Implement the key equation — you don't understand it until you code it

---

## Foundations

| Paper | Year | What it introduces | Notebook |
|---|---|---|---|
| Learning Representations by Back-propagating Errors — Rumelhart et al. | 1986 | Backpropagation algorithm | `01_foundations/backpropagation.ipynb` |
| Dropout: A Simple Way to Prevent Overfitting — Srivastava et al. | 2014 | Dropout regularization | `01_foundations/regularization.ipynb` |
| Batch Normalization — Ioffe & Szegedy | 2015 | Batch normalization | `01_foundations/regularization.ipynb` |
| Adam: A Method for Stochastic Optimization — Kingma & Ba | 2014 | Adam optimizer | `01_foundations/optimizers.ipynb` |

---

## Computer Vision

| Paper | Year | What it introduces | Notebook |
|---|---|---|---|
| Gradient-Based Learning Applied to Document Recognition — LeCun et al. | 1998 | LeNet, convolutional networks | `02_cnn/convolution_operation.ipynb` |
| ImageNet Classification with Deep CNNs (AlexNet) — Krizhevsky et al. | 2012 | Deep CNNs, ReLU, dropout in CV | `02_cnn/cnn_architectures.ipynb` |
| Very Deep Convolutional Networks (VGGNet) — Simonyan & Zisserman | 2014 | Small 3×3 filters, depth | `02_cnn/cnn_architectures.ipynb` |
| Deep Residual Learning for Image Recognition (ResNet) — He et al. | 2015 | Residual connections, skip connections | `02_cnn/cnn_architectures.ipynb` |
| How transferable are features in deep neural networks? — Yosinski et al. | 2014 | Transfer learning theory | `02_cnn/transfer_learning.ipynb` |

---

## Sequence Models

| Paper | Year | What it introduces | Notebook |
|---|---|---|---|
| Learning Long-Term Dependencies with Gradient Descent is Difficult — Bengio et al. | 1994 | Vanishing gradient problem in RNNs | `03_sequences/rnn.ipynb` |
| Long Short-Term Memory — Hochreiter & Schmidhuber | 1997 | LSTM architecture and gates | `03_sequences/lstm_gru.ipynb` |
| Empirical Evaluation of Gated RNNs — Chung et al. | 2014 | GRU — simpler alternative to LSTM | `03_sequences/lstm_gru.ipynb` |
| Sequence to Sequence Learning with Neural Networks — Sutskever et al. | 2014 | Encoder-decoder, seq2seq | `03_sequences/seq2seq.ipynb` |
| Neural Machine Translation by Jointly Learning to Align and Translate — Bahdanau et al. | 2014 | Attention mechanism (original) | `04_transformers/attention_mechanism.ipynb` |

---

## Transformers

| Paper | Year | What it introduces | Notebook |
|---|---|---|---|
| Attention Is All You Need — Vaswani et al. | 2017 | Transformer architecture, multi-head attention | `04_transformers/transformer_architecture.ipynb` |
| BERT: Pre-training of Deep Bidirectional Transformers — Devlin et al. | 2018 | Masked language modeling, fine-tuning | `04_transformers/bert_and_gpt.ipynb` |
| Language Models are Unsupervised Multitask Learners (GPT-2) — Radford et al. | 2019 | Causal language modeling, few-shot | `04_transformers/bert_and_gpt.ipynb` |
| An Image is Worth 16×16 Words (ViT) — Dosovitskiy et al. | 2020 | Transformers for vision | `02_cnn/cnn_architectures.ipynb` |

---

## Generative Models

| Paper | Year | What it introduces | Notebook |
|---|---|---|---|
| Auto-Encoding Variational Bayes — Kingma & Welling | 2013 | VAE, ELBO, reparameterization trick | `05_generative/vae.ipynb` |
| Generative Adversarial Networks — Goodfellow et al. | 2014 | GAN framework, minimax objective | `05_generative/gan.ipynb` |
| Unsupervised Representation Learning with DCGANs — Radford et al. | 2015 | Deep Convolutional GAN | `05_generative/dcgan.ipynb` |
| Denoising Diffusion Probabilistic Models — Ho et al. | 2020 | Diffusion models (read after VAE+GAN) | `05_generative/` (bonus) |

---

## How to Access Papers

All papers above are freely available:

- **arXiv:** https://arxiv.org — the standard for AI/ML preprints
- **Papers With Code:** https://paperswithcode.com — paper + code + benchmarks
- **Semantic Scholar:** https://www.semanticscholar.org — paper search with citations
- **Google Scholar:** https://scholar.google.com — broad academic search

---

## Recommended Reading Order for a Fresher

```
1. Backprop (Rumelhart 1986) — understand the core algorithm
2. Dropout (Srivastava 2014) — why regularization works
3. Adam (Kingma 2014) — how modern optimizers work
4. ResNet (He 2015) — the residual connection idea changed everything
5. Attention (Bahdanau 2014) — original attention, simpler than transformer
6. Attention Is All You Need (Vaswani 2017) — read this multiple times
7. BERT (Devlin 2018) — understand the pretraining paradigm
8. VAE (Kingma 2013) — generative models, probabilistic thinking
9. GAN (Goodfellow 2014) — adversarial training
```
