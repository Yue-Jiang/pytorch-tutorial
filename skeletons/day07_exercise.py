"""Day 7 — nn.Module system: Dropout and BatchNorm1d from scratch."""
import torch
import torch.nn as nn


class MyDropout(nn.Module):
    """Inverted dropout.

    Training: zero each element independently with probability p; scale
    the survivors by 1/(1-p). Eval: identity. Use self.training.
    Banned: nn.Dropout, F.dropout.
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        assert 0.0 <= p < 1.0
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MyBatchNorm1d(nn.Module):
    """BatchNorm over dim 0 for (N, C) inputs, matching nn.BatchNorm1d.

    Parameters: weight (ones), bias (zeros), both (C,).
    Buffers (register_buffer!): running_mean (zeros), running_var (ones).

    forward, training mode:
        batch mean/var over dim 0 (var BIASED for normalizing);
        update running stats with momentum (default 0.1):
            running = (1 - momentum) * running + momentum * batch_stat
            (the batch var stored into running_var is UNBIASED — that's
             what nn.BatchNorm1d does; N > 1 guaranteed in tests)
        normalize x with the BATCH stats, then affine.
    forward, eval mode:
        normalize with the RUNNING stats, then affine. No updates.

    Banned: nn.BatchNorm1d, F.batch_norm.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        # TODO: create weight/bias Parameters and the two buffers.
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
