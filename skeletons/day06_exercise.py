"""Day 6 — Autograd."""
from collections.abc import Callable

import torch


def grad_of(f: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """Gradient of scalar-valued f at x, via autograd.

    Must NOT modify the caller's x (no requires_grad flag left behind on
    it): work on a detached clone. Returns a tensor of x's shape.
    """
    raise NotImplementedError


def numeric_grad(
    f: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, eps: float = 1e-3
) -> torch.Tensor:
    """Central finite-difference gradient of scalar-valued f at x:

        g[i] = (f(x + eps*e_i) - f(x - eps*e_i)) / (2*eps)

    x is 1-D. A Python loop over x's entries is fine here (this is the
    slow reference — that's its job). Use float64 x for accuracy (the
    tests do).
    """
    raise NotImplementedError


def half_stopped(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return (a * b_const).sum() where b_const is b treated as a CONSTANT:
    gradients must flow to a but NOT to b. (One detach, well placed.)"""
    raise NotImplementedError


class MyCube(torch.autograd.Function):
    """x -> x**3 with a hand-written backward (d/dx = 3x^2).

    Implement both static methods. In forward, compute WITHOUT autograd's
    help (it's off inside Functions anyway) and save x for backward.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
