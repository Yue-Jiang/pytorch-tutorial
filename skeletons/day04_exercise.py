"""Day 4 — einsum. Every function: ONE torch.einsum call, nothing else."""
import torch


def attn_scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """q: (B, H, T, D); k: (B, H, S, D)  ->  (B, H, T, S).
    Your transformer's q @ k.transpose(-2, -1), as one einsum."""
    raise NotImplementedError


def outer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a: (M,); b: (N,)  ->  (M, N) with out[i, j] = a[i] * b[j]."""
    raise NotImplementedError


def batch_trace(x: torch.Tensor) -> torch.Tensor:
    """x: (B, N, N)  ->  (B,) with out[b] = sum of x[b]'s diagonal."""
    raise NotImplementedError


def weighted_sum(w: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """w: (B, T); v: (B, T, D)  ->  (B, D): each batch's rows of v,
    combined with its weights. (Attention's mixing step, one head.)"""
    raise NotImplementedError


def bilinear(x: torch.Tensor, A: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """x: (B, D1); A: (D1, D2); y: (B, D2)  ->  (B,) with
    out[b] = x[b] @ A @ y[b]. Three inputs, one einsum."""
    raise NotImplementedError
