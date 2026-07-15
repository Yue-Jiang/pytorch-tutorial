import torch

from day03 import exercise
from helpers import assert_not_used

torch.manual_seed(0)


class TestGather:
    def test_no_banned(self):
        assert_not_used(exercise.my_gather,
                        ["torch.gather", ".gather(", "take_along_dim"])

    def test_matches_torch(self):
        x = torch.randn(4, 7)
        index = torch.randint(0, 7, (4, 3))
        torch.testing.assert_close(
            exercise.my_gather(x, index), torch.gather(x, 1, index))

    def test_repeats_allowed(self):
        x = torch.arange(12.0).view(3, 4)
        index = torch.tensor([[0, 0, 3], [2, 2, 2], [1, 0, 1]])
        torch.testing.assert_close(
            exercise.my_gather(x, index), torch.gather(x, 1, index))


class TestRowPicks:
    def test_values(self):
        x = torch.randn(6, 9)
        cols = torch.randint(0, 9, (6,))
        want = torch.stack([x[i, cols[i]] for i in range(6)])
        torch.testing.assert_close(exercise.pick_per_row(x, cols), want)


class TestTopkMask:
    def test_mask(self):
        x = torch.tensor([[1.0, 5.0, 3.0, 2.0],
                          [9.0, 0.0, 8.0, 7.0]])
        want = torch.tensor([[False, True, True, False],
                             [True, False, True, False]])
        got = exercise.top2_mask(x)
        assert got.dtype == torch.bool
        assert torch.equal(got, want)

    def test_row_counts(self):
        x = torch.randn(10, 6)
        got = exercise.top2_mask(x)
        assert (got.sum(dim=-1) == 2).all(), "each row must have exactly two Trues"

    def test_masks_the_max(self):
        x = torch.randn(10, 6)
        got = exercise.top2_mask(x)
        assert got[torch.arange(10), x.argmax(dim=-1)].all()


class TestScatterHistogram:
    def test_no_banned(self):
        assert_not_used(exercise.histogram,
                        ["bincount", "histc", "for ", "while "])

    def test_matches_bincount(self):
        ids = torch.randint(0, 12, (300,))
        got = exercise.histogram(ids, 12)
        torch.testing.assert_close(got, torch.bincount(ids, minlength=12))

    def test_empty_bins(self):
        ids = torch.tensor([1, 1, 4])
        got = exercise.histogram(ids, 6)
        assert got.tolist() == [0, 2, 0, 0, 1, 0]
