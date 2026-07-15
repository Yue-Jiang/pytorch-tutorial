import inspect

import torch

from day04 import exercise
from helpers import assert_used

torch.manual_seed(0)

BANNED = ["@", "matmul", "bmm", ".sum(", "torch.outer", "torch.trace",
          ".mul(", " * "]


def check_pure_einsum(fn):
    src = inspect.getsource(fn)
    if fn.__doc__:  # docstrings may name the ops being replaced
        src = src.replace(fn.__doc__, "")
    assert "einsum" in src, f"{fn.__name__} must use torch.einsum"
    for b in BANNED:
        assert b not in src, (
            f"{fn.__name__}: `{b}` is banned — express it inside the einsum."
        )


class TestScores:
    def test_pure(self):
        check_pure_einsum(exercise.attn_scores)

    def test_matches(self):
        q = torch.randn(2, 3, 5, 8)
        k = torch.randn(2, 3, 7, 8)
        torch.testing.assert_close(
            exercise.attn_scores(q, k), q @ k.transpose(-2, -1))


class TestOuter:
    def test_pure(self):
        check_pure_einsum(exercise.outer)

    def test_matches(self):
        a, b = torch.randn(4), torch.randn(6)
        torch.testing.assert_close(exercise.outer(a, b), torch.outer(a, b))


class TestTrace:
    def test_pure(self):
        check_pure_einsum(exercise.batch_trace)

    def test_matches(self):
        x = torch.randn(5, 4, 4)
        want = torch.stack([torch.trace(x[b]) for b in range(5)])
        torch.testing.assert_close(exercise.batch_trace(x), want)


class TestWeightedSum:
    def test_pure(self):
        check_pure_einsum(exercise.weighted_sum)

    def test_matches(self):
        w = torch.randn(3, 6)
        v = torch.randn(3, 6, 10)
        want = (w.unsqueeze(-1) * v).sum(dim=1)
        torch.testing.assert_close(exercise.weighted_sum(w, v), want)


class TestBilinear:
    def test_pure(self):
        check_pure_einsum(exercise.bilinear)

    def test_matches(self):
        x = torch.randn(4, 5)
        A = torch.randn(5, 7)
        y = torch.randn(4, 7)
        want = torch.stack([x[b] @ A @ y[b] for b in range(4)])
        torch.testing.assert_close(exercise.bilinear(x, A, y), want)
