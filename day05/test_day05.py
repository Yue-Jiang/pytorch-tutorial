import torch

from day05 import exercise
from helpers import assert_not_used

torch.manual_seed(0)


class TestLogsumexp:
    def test_no_banned(self):
        assert_not_used(exercise.logsumexp, ["torch.logsumexp", "special"])

    def test_matches_torch(self):
        x = torch.randn(4, 9) * 5
        torch.testing.assert_close(
            exercise.logsumexp(x, dim=-1), torch.logsumexp(x, dim=-1))
        torch.testing.assert_close(
            exercise.logsumexp(x, dim=0), torch.logsumexp(x, dim=0))

    def test_stable(self):
        x = torch.tensor([[1000.0, 1000.1], [-2000.0, -2000.5]])
        got = exercise.logsumexp(x, dim=-1)
        assert torch.isfinite(got).all(), "overflow/underflow — shift by the max"
        torch.testing.assert_close(got, torch.logsumexp(x, dim=-1))


class TestMaskedMean:
    def test_basic(self):
        x = torch.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
        m = torch.tensor([[True, True, False], [False, False, True]])
        torch.testing.assert_close(
            exercise.masked_mean(x, m), torch.tensor([1.5, 30.0]))

    def test_empty_row_is_zero_not_nan(self):
        x = torch.randn(3, 4)
        m = torch.zeros(3, 4, dtype=torch.bool)
        got = exercise.masked_mean(x, m)
        assert torch.isfinite(got).all(), "all-False rows must give 0, not nan"
        torch.testing.assert_close(got, torch.zeros(3))

    def test_full_mask_is_plain_mean(self):
        x = torch.randn(5, 7)
        m = torch.ones(5, 7, dtype=torch.bool)
        torch.testing.assert_close(exercise.masked_mean(x, m), x.mean(dim=-1))


class TestStandardize:
    def test_matches_layernorm_core(self):
        x = torch.randn(4, 6, 32) * 3 + 1
        ref = torch.nn.functional.layer_norm(x, (32,))
        torch.testing.assert_close(
            exercise.standardize(x), ref, atol=1e-5, rtol=1e-5)

    def test_other_dim(self):
        x = torch.randn(8, 5)
        got = exercise.standardize(x, dim=0)
        torch.testing.assert_close(got.mean(dim=0), torch.zeros(5), atol=1e-5, rtol=0)


class TestPairwise:
    def test_no_banned(self):
        assert_not_used(exercise.pairwise_sq_dist, ["cdist", "for ", "while "])

    def test_matches_cdist(self):
        a, b = torch.randn(6, 4), torch.randn(9, 4)
        want = torch.cdist(a, b) ** 2
        torch.testing.assert_close(
            exercise.pairwise_sq_dist(a, b), want, atol=1e-4, rtol=1e-4)

    def test_self_distance_zero(self):
        a = torch.randn(5, 3)
        d = exercise.pairwise_sq_dist(a, a)
        torch.testing.assert_close(d.diagonal(), torch.zeros(5), atol=1e-5, rtol=0)
