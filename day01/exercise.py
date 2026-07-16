"""Day 1 — Tensor anatomy."""
import torch


def creation_gallery() -> dict[str, torch.Tensor]:
    """Return a dict with exactly these keys, each built with a creation op:

    "evens":    the even numbers 0..18 inclusive, dtype int64, shape (10,)
    "grid":     11 evenly spaced floats from 0.0 to 1.0 INCLUSIVE, shape (11,)
    "checker":  3x3 float32 tensor of alternating 1s and 0s, 1 at [0,0]
                (hint: build from arange sums % 2, or spell it out)
    "eye4":     the 4x4 identity matrix, float32
    "sevens":   shape (2, 3) filled with 7.0, float32
    """
    evens = torch.arange(0, 19, 2, dtype=torch.long)
    grid = torch.linspace(0, 1, 11)
    checker = torch.arange(1., 10., 1).reshape(3, 3) % 2
    eye4 = torch.eye(4, dtype=torch.float32)
    sevens = torch.full((2, 3), 7.)
    return {'evens':evens, 'grid':grid, 'checker':checker, 'eye4':eye4, 'sevens':sevens}


def cast_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Return x converted to ref's dtype AND device (one call does both)."""
    return x.to(ref)


def safe_mean(x: torch.Tensor) -> torch.Tensor:
    """Mean of ANY numeric tensor, including integer dtypes.

    torch.mean refuses int tensors — cast to float32 first, then mean.
    Returns a float32 scalar tensor.
    """
    return torch.mean(x.float())


def tensor_bytes(x: torch.Tensor) -> int:
    """Bytes of storage x's data occupies: numel * element_size.

    Compute it from x's properties (numel(), element_size()) — no
    hardcoded dtype tables.
    """
    return x.numel() * x.element_size()
