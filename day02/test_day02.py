import torch

from day02 import exercise
from helpers import assert_not_used

torch.manual_seed(0)


class TestViewDetective:
    def test_views_detected(self):
        x = torch.randn(4, 6)
        assert exercise.shares_storage(x, x.view(24))
        assert exercise.shares_storage(x, x.t())
        assert exercise.shares_storage(x, x[1:3])

    def test_copies_detected(self):
        x = torch.randn(4, 6)
        assert not exercise.shares_storage(x, x.clone())
        assert not exercise.shares_storage(x, x.t().contiguous())
        assert not exercise.shares_storage(x, torch.randn(4, 6))


class TestStridePrediction:
    def test_no_measuring(self):
        assert_not_used(
            exercise.transpose_strides, [".stride()", "torch."],
            hint="Compute the strides by arithmetic on the shape tuple.",
        )

    def test_matches_reality(self):
        for shape in [(2, 3, 4), (5, 1, 7), (3, 3, 3)]:
            want = torch.empty(shape).transpose(0, 2).stride()
            got = exercise.transpose_strides(shape)
            assert tuple(got) == tuple(want), (
                f"For contiguous {shape}, transpose(0,2) strides are {want}"
            )


class TestExpandVsRepeat:
    def test_view_is_free(self):
        x = torch.arange(5.0)
        out = exercise.tile_rows_view(x, 3)
        assert out.shape == (3, 5)
        torch.testing.assert_close(out, x.unsqueeze(0).repeat(3, 1))
        assert exercise.shares_storage(x, out) if hasattr(exercise, "shares_storage") else True
        assert out.untyped_storage().data_ptr() == x.untyped_storage().data_ptr(), \
            "tile_rows_view must not copy — use expand"
        assert out.stride(0) == 0, (
            "The tiled dim of an expand has stride 0 — that's the whole trick."
        )

    def test_copy_is_independent(self):
        x = torch.arange(5.0)
        out = exercise.tile_rows_copy(x, 3)
        assert out.shape == (3, 5)
        assert out.untyped_storage().data_ptr() != x.untyped_storage().data_ptr(), \
            "tile_rows_copy must be an independent copy"
        x_backup = x.clone()
        out[0, 0] = 99.0
        torch.testing.assert_close(x, x_backup)  # writing the copy leaves x alone


class TestWindows:
    def test_values(self):
        x = torch.arange(10)
        out = exercise.sliding_windows(x, size=4, step=2)
        want = torch.stack([x[0:4], x[2:6], x[4:8], x[6:10]])
        torch.testing.assert_close(out, want)

    def test_zero_copy(self):
        x = torch.arange(10)
        out = exercise.sliding_windows(x, size=3, step=1)
        assert out.untyped_storage().data_ptr() == x.untyped_storage().data_ptr(), \
            "sliding_windows must be a strided view (unfold), not a copy"
