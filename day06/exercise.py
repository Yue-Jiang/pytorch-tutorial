"""Day 6 — Autograd."""
from collections.abc import Callable

import torch


def grad_of(f: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """Gradient of scalar-valued f at x, via autograd.

    Must NOT modify the caller's x (no requires_grad flag left behind on
    it): work on a detached clone. Returns a tensor of x's shape.
    """
    x = x.detach().clone().requires_grad_(True)
    y = f(x)
    y.backward()
    return x.grad


def numeric_grad(
    f: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, eps: float = 1e-3
) -> torch.Tensor:
    """Central finite-difference gradient of scalar-valued f at x:

        g[i] = (f(x + eps*e_i) - f(x - eps*e_i)) / (2*eps)

    x is 1-D. A Python loop over x's entries is fine here (this is the
    slow reference — that's its job). Use float64 x for accuracy (the
    tests do).
    """
    g = torch.zeros_like(x, dtype=torch.float64)
    for i in torch.arange(x.shape[-1]):
        xp = x.clone()
        xp[i] = xp[i] + eps
        xm = x.clone()
        xm[i] = xm[i] - eps
        g[i] = (f(xp) - f(xm)) / (2 *eps)
    return g
    


def half_stopped(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return (a * b_const).sum() where b_const is b treated as a CONSTANT:
    gradients must flow to a but NOT to b. (One detach, well placed.)"""
    b_const = b.detach()
    return (a*b_const).sum()


class MyCube(torch.autograd.Function):
    """x -> x**3 with a hand-written backward (d/dx = 3x^2).

    Implement both static methods. In forward, compute WITHOUT autograd's
    help (it's off inside Functions anyway) and save x for backward.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return x**3

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        return grad_out * 3 * x**2
