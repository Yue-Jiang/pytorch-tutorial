"""Day 8 — Data & optim machinery."""
import torch


def pad_collate(
    seqs: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack variable-length 1-D long tensors into a padded batch.

    seqs: list of B tensors, lengths L_i, dtype long.
    Returns:
        batch: (B, L_max) long — each row is a sequence, zero-padded at
               the END.
        mask:  (B, L_max) bool — True where the entry is real data.
    """
    raise NotImplementedError


class SGDMomentum:
    """Plain SGD with classical momentum, matching
    torch.optim.SGD(params, lr, momentum) semantics:

        buf <- momentum * buf + grad        (first step: buf = grad copy)
        p   <- p - lr * buf

    Implement: __init__(params, lr, momentum), step(), zero_grad().
    params is a list of tensors (with .grad filled by backward).
    All updates under torch.no_grad(), in-place.
    Banned: torch.optim.
    """

    def __init__(self, params: list[torch.Tensor], lr: float, momentum: float = 0.9):
        raise NotImplementedError

    def step(self) -> None:
        raise NotImplementedError

    def zero_grad(self) -> None:
        """Set every param's .grad to None (or zero it — None is cleaner)."""
        raise NotImplementedError


def clip_grad_norm(params: list[torch.Tensor], max_norm: float) -> float:
    """Clip gradients by GLOBAL norm, matching
    torch.nn.utils.clip_grad_norm_ semantics.

    total_norm = sqrt(sum of ||p.grad||^2 over all params). If it exceeds
    max_norm, multiply EVERY grad in-place by max_norm / total_norm.
    Returns the total_norm (as a plain float) measured BEFORE clipping.
    Banned: torch.nn.utils.
    """
    raise NotImplementedError
