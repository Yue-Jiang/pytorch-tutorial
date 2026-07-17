"""Day 5 — Reductions & elementwise."""
import torch


def logsumexp(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable log(sum(exp(x))) along dim. No keepdim in the
    output (reduce the dim away).

    Banned: torch.logsumexp / torch.special. You built the algebra on
    course #1 day 1: shift by the max, exp, sum, log, shift back.
    Must survive rows like [1000.0, 1000.1].
    """
    xmax = x.amax(dim=dim)
    x = x - xmax.unsqueeze(dim)
    return xmax + torch.log(torch.exp(x).sum(dim=dim))


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Mean of x over `dim`, counting ONLY entries where mask is True.

    x float, mask bool, same shape. Rows whose mask is all-False return 0.0
    (not nan!). Divide by the count of True entries, not the dim size.
    """
    return (x * mask).sum(dim) / mask.sum(dim).clamp(min=1)


def standardize(x: torch.Tensor, dim: int = -1, eps: float = 1e-5) -> torch.Tensor:
    """(x - mean) / sqrt(var + eps) along dim — LayerNorm's core without the
    affine. Biased variance (correction=0), keepdim on the stats.
    """
    return (x - x.mean(dim=dim, keepdim=True)) / torch.sqrt(x.var(dim=dim, keepdim=True, correction=0) + eps)


def pairwise_sq_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a: (M, D); b: (N, D)  ->  (M, N) of SQUARED euclidean distances:
    out[i, j] = ||a[i] - b[j]||^2.

    Banned: torch.cdist, loops. Broadcast complementary unsqueezes.
    """
    return torch.einsum('mnd->mn', (a.unsqueeze(1) - b.unsqueeze(0))**2)
