"""Day 2 — Storage & strides."""
import torch


def shares_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    """True iff a and b are views into the same underlying storage buffer.

    Compare the tensors' untyped storages (`untyped_storage().data_ptr()`).
    """
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


def transpose_strides(shape: tuple[int, int, int]) -> tuple[int, int, int]:
    """PREDICT (compute, don't measure) the strides of
    `x.transpose(0, 2)` where x is a fresh CONTIGUOUS tensor of `shape`.

    Rules: contiguous strides are suffix products of the shape;
    transpose(0, 2) swaps the first and last stride. Return a plain tuple
    of ints, computed from `shape` by arithmetic — creating a tensor and
    measuring it is banned (the checker reads your source).
    """
    # original_strides = (shape[1] * shape[2], shape[2], 1)
    new_strides = (1, shape[2], shape[1] * shape[2])
    return new_strides


def tile_rows_view(x: torch.Tensor, m: int) -> torch.Tensor:
    """Given x of shape (n,), return an (m, n) tensor whose every row is x —
    WITHOUT copying memory (use expand; result must share storage with x).
    """
    return x.expand(m, x.shape[-1])


def tile_rows_copy(x: torch.Tensor, m: int) -> torch.Tensor:
    """Same (m, n) result, but as an independent COPY (use repeat or
    equivalent; result must NOT share storage with x)."""
    return x.expand(m, x.shape[-1]).clone()
    # or x.unsqueeze(0).repeat(m, 1)


def sliding_windows(x: torch.Tensor, size: int, step: int) -> torch.Tensor:
    """All sliding windows of a 1-D tensor as a zero-copy strided view.

    x: (N,) -> result: (num_windows, size), row i = x[i*step : i*step+size].
    Use `unfold`. Must share storage with x.
    """
    # idx_first_row = torch.arange(size).unsqueeze(0)
    # idx_col_offsets = torch.arange(0, x.shape[-1], step).unsqueeze(1)
    # idx = idx_first_row + idx_col_offsets
    # return x[idx]
    return x.unfold(dimension=0, size=size, step=step)
