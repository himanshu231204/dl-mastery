# Contributing to DL Mastery

Thank you for improving this repo. These standards are strict — they exist to keep quality high.

---

## Notebook Standards

Every notebook must follow this exact 7-section order:

1. **Header** — Title, one-line description, Open in Colab badge, prerequisites list
2. **Math prerequisites** — Which notebook in `00_math/` contains the required background
3. **Concept + animated diagram** — Interactive HTML/SVG visualization of the mechanism
4. **NumPy from scratch** — Core computation using only NumPy — no framework
5. **Raw PyTorch tensors** — Same computation using `torch.Tensor` and manual autograd
6. **`nn.Module` implementation** — Production-style PyTorch with training loop
7. **Training dashboard** — 4-panel: loss curve, accuracy, gradient flow, weight histogram
8. **Paper → Code** — Key equation from the original paper, implemented in PyTorch
9. **Debugging section** — At least 3 reproducible failure modes with explanations and fixes
10. **Exercises** — At least 3 tasks with solutions in collapsed cells

---

## Code Standards

- All code must run top to bottom without errors on CPU (GPU optional)
- Every code cell must have a markdown explanation above it
- Print all intermediate tensor shapes — `print(f"x shape: {x.shape}")`
- Use `torch.manual_seed(42)` and `np.random.seed(42)` at the top of every notebook
- Variable names must be descriptive — no `x1`, `a`, `tmp`
- All training loops must use `tqdm` for progress bars
- Never leave a `.cuda()` call that breaks CPU-only execution — use `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`

## Math Standards

- Use LaTeX in markdown: `$inline$` and `$$block$$`
- Every variable in every equation must be defined immediately below it
- Derive from first principles — never state a result without showing the derivation
- Dimensions must be annotated: `W ∈ R^{d_out × d_in}`

## Diagram Standards

- Every major concept must have an animated or interactive diagram
- Diagrams must work in both light and dark mode
- Labels must be readable at 12px minimum
- Animation must be purposeful — it should show a process, not just move for aesthetics

---

## Branch Naming

| Type | Format | Example |
|---|---|---|
| New notebook | `add/description` | `add/attention-mechanism` |
| Bug fix | `fix/description` | `fix/backprop-gradient-sign` |
| Improvement | `improve/description` | `improve/lstm-animated-diagram` |

---

## Pull Request Checklist

- [ ] All cells run top to bottom without errors
- [ ] All 7 sections present in correct order
- [ ] Tensor shapes printed at key steps
- [ ] Paper reference included with DOI or arXiv link
- [ ] Debugging section has at least 3 failure modes
- [ ] Exercises have solutions in collapsed cells
- [ ] Author info is in the final cell
