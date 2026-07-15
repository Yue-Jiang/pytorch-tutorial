# PyTorch Fundamentals — a 10-day typing bootcamp

Course #2 in the series. Where `transformer-from-scratch/` built one machine,
this course drills the *toolbox*: tensors, strides, indexing, einsum,
autograd, the nn.Module system, data/optim machinery, devices — ending in a
bug-forensics capstone.

Same rules as course #1: **type everything by hand**, banned shortcuts are
enforced by the checker, 30 minutes a day, ask for hints not answers.

## Setup

Reuses course #1's conda env (torch + pytest, nothing new):

```bash
conda activate transformer-tutorial
cd pytorch-fundamentals
```

## Daily flow

```bash
python check.py day01          # one part at a time, in order
python check.py day01 <part>   # verbose detail for one part
python check.py reset day01    # restore a blank skeleton (asks first)
pytest day01 -v                # raw pytest if you prefer
```

When a day is green, tell Claude — review, quiz, reflection, commit.

## Syllabus

| Day | Topic                     | You build / drill                                      |
|-----|---------------------------|--------------------------------------------------------|
| 1   | Tensor anatomy            | creation gallery, dtype rules, bytes accounting        |
| 2   | Storage & strides         | view-vs-copy detection, stride prediction, `expand` vs `repeat`, sliding windows |
| 3   | The indexing family       | `gather` from scratch, top-k masks, `where`, scatter histograms |
| 4   | einsum                    | six classic contractions, one call each                |
| 5   | Reductions & elementwise  | `logsumexp`, masked mean, standardize, pairwise distances |
| 6   | Autograd                  | grad plumbing, finite-difference checking, `detach` semantics, a custom `autograd.Function` |
| 7   | nn.Module system          | `Dropout` and `BatchNorm1d` from scratch (train/eval, buffers) |
| 8   | Data & optim machinery    | padding collate, SGD+momentum from scratch, grad clipping |
| 9   | Devices & performance     | device resolution, tree-moves, a timing harness        |
| 10  | Capstone: bug forensics   | six broken snippets — diagnose and repair each         |

Days are independent (unlike course #1) — but do them in order anyway;
later tests occasionally reuse earlier ideas.

## Scope

In: the core tensor/autograd/module machinery every PyTorch codebase leans
on, drilled to muscle memory with reference-checked exercises. Out:
distributed training, CUDA kernel writing, torch.compile internals,
domain libraries (torchvision etc.).
