import torch

from day08 import exercise
from helpers import assert_not_used

torch.manual_seed(0)


class TestCollate:
    def test_shapes_and_values(self):
        seqs = [torch.tensor([1, 2, 3]), torch.tensor([4]), torch.tensor([5, 6])]
        batch, mask = exercise.pad_collate(seqs)
        assert batch.shape == (3, 3) and mask.shape == (3, 3)
        assert batch.dtype == torch.long and mask.dtype == torch.bool
        torch.testing.assert_close(
            batch, torch.tensor([[1, 2, 3], [4, 0, 0], [5, 6, 0]]))
        assert mask.tolist() == [
            [True, True, True], [True, False, False], [True, True, False]]

    def test_mask_selects_originals(self):
        seqs = [torch.randint(1, 50, (n,)) for n in [5, 2, 7, 1]]
        batch, mask = exercise.pad_collate(seqs)
        recovered = batch[mask]
        torch.testing.assert_close(recovered, torch.cat(seqs))


class TestSGDMomentum:
    def test_no_banned(self):
        assert_not_used(exercise.SGDMomentum, ["torch.optim"])

    def test_matches_torch_sgd(self):
        torch.manual_seed(1)
        p_mine = torch.randn(6, requires_grad=True)
        p_ref = p_mine.detach().clone().requires_grad_(True)
        mine = exercise.SGDMomentum([p_mine], lr=0.1, momentum=0.9)
        ref = torch.optim.SGD([p_ref], lr=0.1, momentum=0.9)
        target = torch.randn(6)
        for _ in range(10):
            mine.zero_grad(), ref.zero_grad()
            ((p_mine - target) ** 2).sum().backward()
            ((p_ref - target) ** 2).sum().backward()
            mine.step(), ref.step()
            torch.testing.assert_close(p_mine, p_ref, atol=1e-6, rtol=1e-6)

    def test_zero_grad(self):
        p = torch.randn(3, requires_grad=True)
        opt = exercise.SGDMomentum([p], lr=0.1)
        (p ** 2).sum().backward()
        opt.zero_grad()
        assert p.grad is None or torch.all(p.grad == 0)


class TestClip:
    def test_no_banned(self):
        assert_not_used(exercise.clip_grad_norm, ["torch.nn.utils"])

    def test_matches_torch(self):
        def fresh_grads():
            torch.manual_seed(2)
            ps = [torch.randn(4, requires_grad=True), torch.randn(3, requires_grad=True)]
            (sum((p ** 3).sum() for p in ps)).backward()
            return ps

        mine = fresh_grads()
        got_norm = exercise.clip_grad_norm(mine, max_norm=1.0)
        ref = fresh_grads()
        want_norm = torch.nn.utils.clip_grad_norm_(ref, max_norm=1.0)
        assert abs(got_norm - want_norm.item()) < 1e-5
        for a, b in zip(mine, ref):
            torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-6)

    def test_small_grads_untouched(self):
        p = torch.zeros(3, requires_grad=True)
        (0.001 * p.sum()).backward()
        before = p.grad.clone()
        exercise.clip_grad_norm([p], max_norm=10.0)
        torch.testing.assert_close(p.grad, before)
