"""Day 5 — Reductions & elementwise."""
import torch


def logsumexp(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable log(sum(exp(x))) along dim. No keepdim in the
    output (reduce the dim away).

    Banned: torch.logsumexp / torch.special. You built the algebra on
    course #1 day 1: shift by the max, exp, sum, log, shift back.
    Must survive rows like [1000.0, 1000.1].
    """
    raise NotImplementedError


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Mean of x over `dim`, counting ONLY entries where mask is True.

    x float, mask bool, same shape. Rows whose mask is all-False return 0.0
    (not nan!). Divide by the count of True entries, not the dim size.
    """
    raise NotImplementedError


def standardize(x: torch.Tensor, dim: int = -1, eps: float = 1e-5) -> torch.Tensor:
    """(x - mean) / sqrt(var + eps) along dim — LayerNorm's core without the
    affine. Biased variance (correction=0), keepdim on the stats.
    """
    raise NotImplementedError


def pairwise_sq_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a: (M, D); b: (N, D)  ->  (M, N) of SQUARED euclidean distances:
    out[i, j] = ||a[i] - b[j]||^2.

    Banned: torch.cdist, loops. Broadcast complementary unsqueezes.
    """
    raise NotImplementedError
