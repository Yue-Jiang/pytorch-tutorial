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
        if not self.training:
            return x
        probs = torch.full_like(x, self.p)
        mask = torch.bernoulli(probs).to(dtype=torch.bool)
        return x.masked_fill(mask, 0) / (1 - self.p)


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
        self.weight = nn.Parameter(torch.ones((num_features,)))
        self.bias = nn.Parameter(torch.zeros((num_features,)))
        self.register_buffer("running_mean", torch.zeros((num_features,)))
        self.register_buffer("running_var", torch.ones((num_features,)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, correction=1)
        batch_var_biased = x.var(dim=0, keepdim=True, correction=0)
        if self.training:
            x = (x - batch_mean) / torch.sqrt(batch_var_biased + self.eps)
            self.running_mean.mul_(1 - self.momentum).add_(self.momentum * batch_mean.detach())
            self.running_var.mul_(1 - self.momentum).add_(self.momentum * batch_var.detach())
        else:
            x = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        return x * self.weight + self.bias
