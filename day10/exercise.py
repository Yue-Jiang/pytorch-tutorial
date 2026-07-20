"""Day 10 — bug forensics.

This file is different from every other exercise file: the six functions
below are ALREADY WRITTEN, and every one of them contains exactly one
bug. Each docstring states the INTENDED behavior truthfully; somewhere,
the code disagrees with it. Your job, for each function: run the
checker, read the failure, diagnose which bug class you are looking at,
and repair it IN PLACE with the smallest correct change — do not rewrite
the function. Bring your six one-line diagnoses to the review; naming
the bug class is the skill this day exists to test.
"""
import torch
import torch.nn.functional as F


def stable_softmax(x: torch.Tensor) -> torch.Tensor:
    """Compute the softmax over the last dimension of x.

    The result must be a valid probability distribution along the last
    dimension, and the function must survive inputs with very large
    values — for example a row like [10000, 9999] — without producing
    infinities or NaNs.
    """
    e = torch.exp(x)
    return e / e.sum(dim=-1, keepdim=True)


def causal_scores(scores: torch.Tensor) -> torch.Tensor:
    """Given a square (T, T) matrix of attention scores, return a copy in
    which every FUTURE position is replaced by negative infinity, ready
    for a softmax.

    "Future" means the entries above the main diagonal — positions where
    the column index is greater than the row index. Every entry on or
    below the diagonal (the present and the past) must come through with
    its original value unchanged.
    """
    mask = torch.tril(torch.ones_like(scores, dtype=torch.bool))
    return scores.masked_fill(mask, -torch.inf)


def merge_heads(y: torch.Tensor) -> torch.Tensor:
    """Merge per-head attention outputs back into one feature dimension.

    The input y has shape (B, H, T, D_head): batch, heads, sequence
    positions, and per-head features. The output must have shape
    (B, T, H * D_head), where each position's H per-head vectors are
    laid side by side — the merge step from course #1, day 4.
    """
    B, H, T, Dh = y.shape
    return y.transpose(1, 2).view(B, T, H * Dh)


def standardize_rows(x: torch.Tensor) -> torch.Tensor:
    """Standardize each ROW of a 2-D tensor.

    The input x has shape (N, C). In the output, every row must have
    mean zero and standard deviation one, where the standard deviation
    is the biased one (computed with correction=0).
    """
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True, correction=0)
    return (x - mean) / std


def sgd_update(p: torch.Tensor, lr: float) -> None:
    """Apply one plain SGD step to a parameter, in place.

    The argument p is a leaf tensor with requires_grad=True whose .grad
    mailbox has already been filled by a backward call. This function
    must change p in place to p minus lr times p.grad, must leave p as
    the same tensor object (no rebinding, no copy), and must leave
    p.requires_grad set to True. It returns nothing.
    """
    p -= lr * p.grad


def sequence_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute the mean cross-entropy loss for a batch of predictions.

    The argument logits has shape (N, C) and contains RAW SCORES — the
    unnormalized outputs of a model, exactly what F.cross_entropy is
    designed to receive. The argument targets has shape (N,) and holds
    the correct class index for each row. Return the mean loss as a
    scalar tensor.
    """
    probs = torch.softmax(logits, dim=-1)
    return F.cross_entropy(probs, targets)
