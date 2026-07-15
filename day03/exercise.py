"""Day 3 — The indexing family."""
import torch


def my_gather(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Reimplement torch.gather for the 2-D, dim=1 case.

    x: (N, C); index: (N, K) long with values in [0, C).
    Returns out (N, K) with out[i, j] = x[i, index[i, j]].

    Banned: torch.gather / .gather / take_along_dim. Use advanced indexing
    with a broadcast row-index column (course #1 day 6's grid trick).
    """
    raise NotImplementedError


def pick_per_row(x: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    """x: (N, C); cols: (N,) long. Return (N,) with out[i] = x[i, cols[i]].

    (The cross-entropy indexing move — one line.)
    """
    raise NotImplementedError


def top2_mask(x: torch.Tensor) -> torch.Tensor:
    """x: (N, C) with C >= 2. Return a BOOL mask of shape (N, C), True at
    each row's two largest entries, False elsewhere.

    Assume no ties among each row's top values. (topk + scatter, or
    topk + comparison — your pick.)
    """
    raise NotImplementedError


def histogram(ids: torch.Tensor, n_bins: int) -> torch.Tensor:
    """ids: (M,) long with values in [0, n_bins). Return (n_bins,) long
    where out[v] = number of times v appears in ids.

    Banned: torch.bincount, torch.histc, loops over ids. Use scatter_add_
    (or index_add_) into a zeros tensor.
    """
    raise NotImplementedError
