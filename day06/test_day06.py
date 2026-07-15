import torch

from day06 import exercise

torch.manual_seed(0)


class TestGradPlumbing:
    def test_simple_grad(self):
        x = torch.randn(5)
        g = exercise.grad_of(lambda t: (t ** 2).sum(), x)
        torch.testing.assert_close(g, 2 * x)

    def test_caller_tensor_untouched(self):
        x = torch.randn(4)
        exercise.grad_of(lambda t: t.sum(), x)
        assert x.requires_grad is False, "don't mutate the caller's tensor"
        assert x.grad is None

    def test_chain(self):
        x = torch.randn(3)
        g = exercise.grad_of(lambda t: (t.exp() + t).sum(), x)
        torch.testing.assert_close(g, x.exp() + 1)


class TestNumericGrad:
    def test_matches_autograd(self):
        x = torch.randn(6, dtype=torch.float64)
        f = lambda t: (t ** 3 - 2 * t).sum()
        num = exercise.numeric_grad(f, x)
        ana = exercise.grad_of(f, x)
        torch.testing.assert_close(num, ana, atol=1e-4, rtol=1e-4)

    def test_catches_a_wrong_gradient(self):
        # numeric grad of x^2 is 2x — NOT 3x; the tool must distinguish.
        x = torch.randn(4, dtype=torch.float64)
        num = exercise.numeric_grad(lambda t: (t ** 2).sum(), x)
        assert not torch.allclose(num, 3 * x, atol=1e-2), \
            "numeric_grad should disagree with a wrong analytic formula"


class TestDetachSemantics:
    def test_grad_flows_to_a_only(self):
        a = torch.randn(5, requires_grad=True)
        b = torch.randn(5, requires_grad=True)
        out = exercise.half_stopped(a, b)
        out.backward()
        torch.testing.assert_close(a.grad, b.detach())
        assert b.grad is None or torch.all(b.grad == 0), \
            "gradients must NOT reach b — detach it"

    def test_value_unchanged(self):
        a, b = torch.randn(3, requires_grad=True), torch.randn(3, requires_grad=True)
        torch.testing.assert_close(
            exercise.half_stopped(a, b).detach(), (a * b).sum().detach())


class TestCustomFunction:
    def test_forward(self):
        x = torch.randn(4)
        torch.testing.assert_close(exercise.MyCube.apply(x), x ** 3)

    def test_backward_formula(self):
        x = torch.randn(5, requires_grad=True)
        exercise.MyCube.apply(x).sum().backward()
        torch.testing.assert_close(x.grad, 3 * x.detach() ** 2)

    def test_gradcheck(self):
        x = torch.randn(4, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(exercise.MyCube.apply, (x,)), \
            "gradcheck compares your backward against finite differences"
