import torch
import torch.nn as nn

from day07 import exercise

torch.manual_seed(0)


class TestDropout:
    def test_eval_is_identity(self):
        d = exercise.MyDropout(0.5).eval()
        x = torch.randn(100)
        torch.testing.assert_close(d(x), x)

    def test_train_zeros_and_scales(self):
        torch.manual_seed(1)
        d = exercise.MyDropout(0.25).train()
        x = torch.ones(10000)
        out = d(x)
        zeros = (out == 0).float().mean().item()
        assert 0.2 < zeros < 0.3, f"~25% of entries should be zeroed, got {zeros:.0%}"
        survivors = out[out != 0]
        torch.testing.assert_close(
            survivors, torch.full_like(survivors, 1 / 0.75),
            msg="survivors must be scaled by 1/(1-p) — inverted dropout",
        )

    def test_expectation_preserved(self):
        torch.manual_seed(2)
        d = exercise.MyDropout(0.5).train()
        x = torch.ones(100000)
        assert abs(d(x).mean().item() - 1.0) < 0.02

    def test_uses_training_flag(self):
        d = exercise.MyDropout(0.5)
        x = torch.randn(1000)
        d.train()
        assert not torch.equal(d(x), x)
        d.eval()
        assert torch.equal(d(x), x)


class TestBatchNorm:
    def make_pair(self):
        torch.manual_seed(3)
        mine = exercise.MyBatchNorm1d(8)
        ref = nn.BatchNorm1d(8)
        return mine, ref

    def test_buffers_not_parameters(self):
        m = exercise.MyBatchNorm1d(4)
        param_names = {n for n, _ in m.named_parameters()}
        buffer_names = {n for n, _ in m.named_buffers()}
        assert param_names == {"weight", "bias"}
        assert {"running_mean", "running_var"} <= buffer_names, \
            "running stats must be register_buffer'd, not Parameters"

    def test_train_matches_torch(self):
        mine, ref = self.make_pair()
        mine.train(), ref.train()
        for _ in range(3):
            x = torch.randn(16, 8) * 2 + 1
            torch.testing.assert_close(mine(x), ref(x), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            mine.running_mean, ref.running_mean, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            mine.running_var, ref.running_var, atol=1e-5, rtol=1e-5)

    def test_eval_uses_running_stats(self):
        mine, ref = self.make_pair()
        mine.train(), ref.train()
        for _ in range(5):
            x = torch.randn(32, 8)
            mine(x), ref(x)
        mine.eval(), ref.eval()
        x = torch.randn(4, 8) + 10  # wildly off-distribution batch
        torch.testing.assert_close(mine(x), ref(x), atol=1e-5, rtol=1e-5)

    def test_eval_does_not_update(self):
        mine, _ = self.make_pair()
        mine.train()
        mine(torch.randn(16, 8))
        mine.eval()
        before = mine.running_mean.clone()
        mine(torch.randn(16, 8) + 5)
        torch.testing.assert_close(mine.running_mean, before)
