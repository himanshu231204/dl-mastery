# PyTorch Cheatsheet

Complete reference for PyTorch operations used throughout dl-mastery.

---

## 👨‍💻 Author — Himanshu Kumar
- 🌐 GitHub: [@himanshu231204](https://github.com/himanshu231204)
- 💼 LinkedIn: [himanshu231204](https://www.linkedin.com/in/himanshu231204)

---

## Tensors — Creation

```python
import torch
import numpy as np

# From data
torch.tensor([1, 2, 3])                        # int64
torch.tensor([1.0, 2.0, 3.0])                  # float32
torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)

# Built-in constructors
torch.zeros(3, 4)                               # all zeros
torch.ones(2, 3)                                # all ones
torch.full((2, 2), 7.0)                        # filled with value
torch.eye(3)                                    # identity matrix
torch.arange(0, 10, 2)                         # [0, 2, 4, 6, 8]
torch.linspace(0, 1, 5)                        # [0.00, 0.25, 0.50, 0.75, 1.00]
torch.rand(3, 4)                                # uniform [0, 1)
torch.randn(3, 4)                               # standard normal
torch.randint(0, 10, (3, 4))                   # random integers

# From NumPy
a = np.array([1.0, 2.0, 3.0])
t = torch.from_numpy(a)                        # shares memory!
t = torch.tensor(a)                            # copies data

# Device management — always do this
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
t = torch.randn(3, 4).to(device)
```

---

## Tensors — Attributes and Inspection

```python
t = torch.randn(3, 4)

t.shape          # torch.Size([3, 4])
t.size()         # torch.Size([3, 4])  — same as shape
t.ndim           # 2
t.numel()        # 12 — total number of elements
t.dtype          # torch.float32
t.device         # device(type='cpu')
t.requires_grad  # False
t.is_cuda        # False

# Always print shapes during debugging
print(f"x: {t.shape} | dtype: {t.dtype} | device: {t.device}")
```

---

## Tensors — Indexing and Slicing

```python
t = torch.arange(12).reshape(3, 4)
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])

t[0]           # first row — tensor([0, 1, 2, 3])
t[:, 0]        # first col — tensor([0, 4, 8])
t[1:3, 1:3]    # submatrix — tensor([[5, 6], [9, 10]])
t[-1]          # last row
t[t > 5]       # boolean indexing — tensor([ 6,  7,  8,  9, 10, 11])

# Fancy indexing
idx = torch.tensor([0, 2])
t[idx]         # rows 0 and 2
```

---

## Tensors — Reshaping

```python
t = torch.arange(12)

t.reshape(3, 4)      # returns view when possible
t.view(3, 4)         # always returns view — fails if non-contiguous
t.reshape(3, -1)     # -1 = infer this dimension
t.flatten()          # always 1D
t.squeeze()          # removes all dim-1 axes
t.unsqueeze(0)       # adds dim at position 0  — shape (1, 12)
t.unsqueeze(-1)      # adds dim at last position — shape (12, 1)

# Contiguous check — needed before .view()
t2 = t.T             # transpose is non-contiguous
t2.contiguous().view(-1)  # make contiguous first
```

---

## Tensors — Math Operations

```python
a = torch.tensor([[1., 2.], [3., 4.]])
b = torch.tensor([[5., 6.], [7., 8.]])

# Element-wise
a + b;  a - b;  a * b;  a / b;  a ** 2

# Matrix multiplication
a @ b                      # (2,2) @ (2,2) → (2,2)
torch.mm(a, b)             # same, 2D only
torch.matmul(a, b)         # general — handles batches too
torch.bmm(A, B)            # batched matmul — (B,n,m) @ (B,m,p) → (B,n,p)

# Reductions
a.sum()                    # scalar
a.sum(dim=0)               # sum along rows — shape (2,)
a.sum(dim=1)               # sum along cols — shape (2,)
a.sum(dim=1, keepdim=True) # shape (2,1) — keeps dimension
a.mean(); a.std(); a.max(); a.min()
a.argmax(); a.argmax(dim=1)
```

---

## Autograd — Gradient Computation

```python
# requires_grad=True tells PyTorch to track operations
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()       # y = x0² + x1² = 4 + 9 = 13

y.backward()             # compute dy/dx
print(x.grad)            # tensor([4., 6.])  — dy/dx = 2x

# Zero gradients before next backward pass — CRITICAL
x.grad.zero_()           # in-place zero

# Detach from computation graph
z = x.detach()           # z is a tensor with no grad tracking
with torch.no_grad():    # context manager — no grad tracking inside
    pred = model(x)
```

---

## Building Models — nn.Module

```python
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = MLP(784, 256, 10).to(device)
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable   = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total: {total_params:,}  Trainable: {trainable:,}")
```

---

## Common Layers

```python
nn.Linear(in, out)                             # y = xW^T + b
nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
nn.MaxPool2d(kernel_size, stride)
nn.AvgPool2d(kernel_size)
nn.BatchNorm1d(num_features)                   # for 2D input
nn.BatchNorm2d(num_features)                   # for 4D input (images)
nn.LayerNorm(normalized_shape)                 # used in transformers
nn.Dropout(p=0.5)                              # training only
nn.Embedding(num_embeddings, embedding_dim)    # lookup table
nn.LSTM(input_size, hidden_size, num_layers)
nn.MultiheadAttention(embed_dim, num_heads)
nn.TransformerEncoderLayer(d_model, nhead)

# Activation functions
F.relu(x)
F.gelu(x)
F.sigmoid(x)
F.tanh(x)
F.softmax(x, dim=-1)
F.log_softmax(x, dim=-1)
```

---

## Loss Functions

```python
nn.CrossEntropyLoss()       # classification — combines log_softmax + NLLLoss
                             # expects raw logits, not softmax output
nn.BCEWithLogitsLoss()      # binary classification — numerically stable
nn.MSELoss()                # regression
nn.L1Loss()                 # MAE
nn.NLLLoss()                # negative log likelihood
nn.KLDivLoss(reduction='batchmean')  # KL divergence — used in VAE

# Usage
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, targets)   # logits: (batch, classes), targets: (batch,) — int
```

---

## Optimizers

```python
import torch.optim as optim

optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
optim.RMSprop(model.parameters(), lr=1e-3, alpha=0.99)

# Learning rate schedulers
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=100, epochs=10)
```

---

## Standard Training Loop

```python
from tqdm import tqdm

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for X, y in tqdm(loader, desc='Training'):
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()          # 1. zero gradients
        logits = model(X)              # 2. forward pass
        loss   = criterion(logits, y)  # 3. compute loss
        loss.backward()                # 4. backward pass
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 5. clip grads
        optimizer.step()               # 6. update weights

        total_loss += loss.item() * X.size(0)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += X.size(0)

    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss   = criterion(logits, y)
            total_loss += loss.item() * X.size(0)
            correct    += (logits.argmax(1) == y).sum().item()
            total      += X.size(0)

    return total_loss / total, correct / total
```

---

## Training Dashboard — 4-Panel Diagnostic

```python
import matplotlib.pyplot as plt

def plot_training_dashboard(history):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Loss curves
    axes[0,0].plot(history['train_loss'], color='steelblue', linewidth=2, label='Train')
    axes[0,0].plot(history['val_loss'],   color='coral',     linewidth=2, label='Val')
    axes[0,0].set_title('Loss'); axes[0,0].legend()

    # Accuracy curves
    axes[0,1].plot(history['train_acc'], color='steelblue', linewidth=2, label='Train')
    axes[0,1].plot(history['val_acc'],   color='coral',     linewidth=2, label='Val')
    axes[0,1].set_title('Accuracy'); axes[0,1].legend()

    # Gradient flow — mean grad norm per layer
    axes[1,0].bar(range(len(history['grad_norms'])),
                  history['grad_norms'], color='seagreen', alpha=0.8)
    axes[1,0].set_title('Gradient Flow (mean norm per layer)')
    axes[1,0].set_xlabel('Layer index')

    # Weight distribution of first layer
    axes[1,1].hist(history['weight_sample'], bins=50,
                   color='steelblue', edgecolor='white', alpha=0.85)
    axes[1,1].set_title('Weight Distribution (layer 1)')

    plt.suptitle('Training Dashboard', fontsize=14)
    plt.tight_layout()
    plt.show()
```

---

## Useful Utilities

```python
# Save and load checkpoints
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pt')

checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Freeze layers (for transfer learning)
for param in model.features.parameters():
    param.requires_grad = False

# DataLoader
from torch.utils.data import DataLoader, TensorDataset, random_split

dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
val_size   = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=64, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_set,   batch_size=64, shuffle=False, num_workers=2)

# Reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
```
