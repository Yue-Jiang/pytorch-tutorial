import torch
import torch.nn.functional as F

from day10 import exercise

torch.manual_seed(0)


class TestSoftmaxBug:
    def test_normal_inputs(self):
        x = torch.randn(4, 7)
        torch.testing.assert_close(
            exercise.stable_softmax(x), torch.softmax(x, dim=-1))

    def test_huge_inputs(self):
        x = torch.tensor([[10000.0, 9999.0]])
        got = exercise.stable_softmax(x)
        assert torch.isfinite(got).all(), (
            "nan/inf on large inputs — which day-1 protection is missing?"
        )
        torch.testing.assert_close(got, torch.softmax(x, dim=-1))


class TestMaskBug:
    def test_future_is_blocked(self):
        s = torch.randn(5, 5)
        out = exercise.causal_scores(s)
        upper = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
        assert (out[upper] == -torch.inf).all(), (
            "future positions (above the diagonal) must be -inf"
        )

    def test_past_survives(self):
        s = torch.randn(5, 5)
        out = exercise.causal_scores(s)
        keep = torch.tril(torch.ones(5, 5, dtype=torch.bool))
        torch.testing.assert_close(out[keep], s[keep])


class TestViewBug:
    def test_runs_and_matches(self):
        y = torch.randn(2, 4, 7, 16)
        out = exercise.merge_heads(y)          # crashes until fixed
        assert out.shape == (2, 7, 64)
        want = y.permute(0, 2, 1, 3).reshape(2, 7, 64)
        torch.testing.assert_close(out, want)


class TestMeanBug:
    def test_rows_standardized_nonsquare(self):
        x = torch.randn(3, 8) * 4 + 2          # NOT square — no hiding
        out = exercise.standardize_rows(x)
        torch.testing.assert_close(
            out.mean(dim=1), torch.zeros(3), atol=1e-5, rtol=0)
        torch.testing.assert_close(
            out.std(dim=1, correction=0), torch.ones(3), atol=1e-4, rtol=0)

    def test_values(self):
        x = torch.randn(4, 6)
        want = (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True, correction=0)
        torch.testing.assert_close(exercise.standardize_rows(x), want)


class TestUpdateBug:
    def test_step_runs_and_updates(self):
        p = torch.randn(5, requires_grad=True)
        (p ** 2).sum().backward()
        before = p.detach().clone()
        grad = p.grad.clone()
        exercise.sgd_update(p, lr=0.1)          # errors until fixed
        torch.testing.assert_close(p.detach(), before - 0.1 * grad)
        assert p.requires_grad, "p must remain a trainable leaf"

    def test_in_place(self):
        p = torch.randn(3, requires_grad=True)
        p.sum().backward()
        ptr = p.data_ptr()
        exercise.sgd_update(p, lr=0.01)
        assert p.data_ptr() == ptr, "update must be in-place on the same tensor"


class TestLossBug:
    def test_matches_reference(self):
        logits = torch.randn(32, 10) * 4
        targets = torch.randint(0, 10, (32,))
        torch.testing.assert_close(
            exercise.sequence_loss(logits, targets),
            F.cross_entropy(logits, targets),
            atol=1e-5, rtol=1e-5,
        )

    def test_confident_correct_is_near_zero(self):
        logits = torch.full((4, 5), -20.0)
        targets = torch.tensor([0, 1, 2, 3])
        logits[torch.arange(4), targets] = 20.0
        loss = exercise.sequence_loss(logits, targets)
        assert loss.item() < 1e-3, (
            f"perfectly confident correct predictions should give ~0 loss, "
            f"got {loss.item():.3f} — what does cross_entropy expect as input?"
        )
