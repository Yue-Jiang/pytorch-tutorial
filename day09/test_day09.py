import time

import torch

from day09 import exercise


class TestDevice:
    def test_returns_device_object(self):
        d = exercise.best_device()
        assert isinstance(d, torch.device)

    def test_resolution_order(self):
        d = exercise.best_device()
        if torch.cuda.is_available():
            assert d.type == "cuda"
        elif torch.backends.mps.is_available():
            assert d.type == "mps"
        else:
            assert d.type == "cpu"


class TestTreeMove:
    def test_flat_tensor(self):
        x = torch.randn(3)
        out = exercise.move_tree(x, torch.device("cpu"))
        assert out.device.type == "cpu"

    def test_nested_structures(self):
        tree = {
            "a": torch.randn(2),
            "b": [torch.randn(3), {"c": torch.randn(4)}],
            "d": (torch.randn(5), "hello", 42),
            "e": None,
        }
        out = exercise.move_tree(tree, torch.device("cpu"))
        assert out["a"].device.type == "cpu"
        assert out["b"][1]["c"].device.type == "cpu"
        assert isinstance(out["d"], tuple), "tuples must stay tuples"
        assert out["d"][1] == "hello" and out["d"][2] == 42
        assert out["e"] is None

    def test_values_preserved(self):
        x = torch.randn(4)
        out = exercise.move_tree({"x": x}, torch.device("cpu"))
        torch.testing.assert_close(out["x"], x)

    def test_real_device_roundtrip(self):
        d = exercise.best_device()
        x = exercise.move_tree(torch.randn(3), d)
        assert x.device.type == d.type


class TestTiming:
    def test_returns_reasonable_scale(self):
        t = exercise.time_op(lambda: time.sleep(0.005), iters=5, warmup=1)
        assert 0.004 < t < 0.05, f"a 5ms sleep should time ~5ms, got {t*1000:.1f}ms"

    def test_fast_op_is_fast(self):
        t = exercise.time_op(lambda: 1 + 1, iters=10, warmup=2)
        assert t < 1e-3

    def test_median_ignores_outliers(self):
        calls = {"n": 0}

        def spiky():
            calls["n"] += 1
            if calls["n"] == 4:          # one slow outlier mid-run
                time.sleep(0.05)

        t = exercise.time_op(spiky, iters=9, warmup=0)
        assert t < 0.01, "median timing must shrug off a single outlier"
