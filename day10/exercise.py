"""Day 10 — bug forensics. Six broken functions. Diagnose, then repair
IN PLACE with the smallest correct change. Docstrings state the INTENDED
behavior; the code disagrees somewhere.
"""
import torch
import torch.nn.functional as F


def stable_softmax(x: torch.Tensor) -> torch.Tensor:
    """Softmax over the last dim. Must survive inputs like [10000, 9999]."""
    e = torch.exp(x)
    return e / e.sum(dim=-1, keepdim=True)


def causal_scores(scores: torch.Tensor) -> torch.Tensor:
    """Given (T, T) attention scores, return them with all FUTURE positions
    (column > row) set to -inf, ready for softmax."""
    mask = torch.tril(torch.ones_like(scores, dtype=torch.bool))
    return scores.masked_fill(mask, -torch.inf)


def merge_heads(y: torch.Tensor) -> torch.Tensor:
    """y: (B, H, T, D_head) -> (B, T, H * D_head), the day-4 merge."""
    B, H, T, Dh = y.shape
    return y.transpose(1, 2).view(B, T, H * Dh)


def standardize_rows(x: torch.Tensor) -> torch.Tensor:
    """x: (N, C). Return x with each ROW standardized: every row of the
    output has mean 0 and (biased) std 1."""
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True, correction=0)
    return (x - mean) / std


def sgd_update(p: torch.Tensor, lr: float) -> None:
    """One in-place SGD step on parameter p (a leaf with requires_grad=True
    and .grad already populated): p <- p - lr * p.grad."""
    p -= lr * p.grad


def sequence_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy. logits: (N, C) RAW SCORES; targets: (N,) long."""
    probs = torch.softmax(logits, dim=-1)
    return F.cross_entropy(probs, targets)
