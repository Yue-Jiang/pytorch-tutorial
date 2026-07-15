import torch

from day01 import exercise

torch.manual_seed(0)


class TestCreation:
    def test_gallery(self):
        g = exercise.creation_gallery()
        torch.testing.assert_close(g["evens"], torch.arange(0, 20, 2))
        assert g["evens"].dtype == torch.int64
        torch.testing.assert_close(g["grid"], torch.linspace(0, 1, 11))
        assert g["grid"][-1].item() == 1.0, "grid must INCLUDE 1.0 (linspace, not arange)"
        want_checker = torch.tensor(
            [[1., 0., 1.], [0., 1., 0.], [1., 0., 1.]])
        torch.testing.assert_close(g["checker"], want_checker)
        torch.testing.assert_close(g["eye4"], torch.eye(4))
        torch.testing.assert_close(g["sevens"], torch.full((2, 3), 7.0))


class TestDtypes:
    def test_cast_like_dtype(self):
        x = torch.arange(5)                     # int64
        ref = torch.randn(2, dtype=torch.float64)
        out = exercise.cast_like(x, ref)
        assert out.dtype == torch.float64
        torch.testing.assert_close(out, torch.arange(5, dtype=torch.float64))

    def test_cast_like_bool(self):
        x = torch.tensor([0.0, 1.5, -2.0])
        ref = torch.tensor([True])
        out = exercise.cast_like(x, ref)
        assert out.dtype == torch.bool
        assert out.tolist() == [False, True, True]

    def test_safe_mean_int(self):
        x = torch.arange(10)  # int64 — torch.mean would raise
        out = exercise.safe_mean(x)
        assert out.dtype == torch.float32
        torch.testing.assert_close(out, torch.tensor(4.5))

    def test_safe_mean_float(self):
        x = torch.randn(4, 5)
        torch.testing.assert_close(
            exercise.safe_mean(x), x.mean().to(torch.float32))


class TestBytes:
    def test_bytes_by_dtype(self):
        assert exercise.tensor_bytes(torch.zeros(10)) == 40          # float32
        assert exercise.tensor_bytes(torch.zeros(10, dtype=torch.int64)) == 80
        assert exercise.tensor_bytes(torch.zeros(10, dtype=torch.bfloat16)) == 20
        assert exercise.tensor_bytes(torch.zeros(10, dtype=torch.bool)) == 10

    def test_bytes_shape(self):
        assert exercise.tensor_bytes(torch.zeros(3, 4, 5)) == 3 * 4 * 5 * 4

    def test_no_hardcoding(self):
        from helpers import assert_used
        assert_used(exercise.tensor_bytes, ["element_size"])
