"""Day 8 — Data & optim machinery."""
import torch


def pad_collate(
    seqs: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack variable-length 1-D long tensors into a padded batch.

    seqs: list of B tensors, lengths L_i, dtype long.
    Returns:
        batch: (B, L_max) long — each row is a sequence, zero-padded at
               the END.
        mask:  (B, L_max) bool — True where the entry is real data.
    """
    b = len(seqs)
    lmax = max([seq.shape[-1] for seq in seqs])
    batch = torch.zeros((b, lmax), dtype=torch.long)
    mask = torch.full((b, lmax), dtype=torch.bool, fill_value=False)
    for i in torch.arange(b):
        batch[i, :seqs[i].shape[-1]] = seqs[i]
        mask[i, :seqs[i].shape[-1]] = True
    return batch, mask


class SGDMomentum:
    """Stochastic gradient descent with classical momentum, built by hand.

    This class must behave exactly like torch.optim.SGD(params, lr,
    momentum): the checker runs both optimizers side by side for ten
    steps from identical starting points and requires the parameter
    trajectories to match to six decimal places. Using torch.optim
    anywhere in this class is banned.

    WHAT THE CONSTRUCTOR STORES. It receives a list of parameter tensors
    (each will have its .grad mailbox filled by someone else's backward
    call), a learning rate, and a momentum coefficient. It should store
    all three, plus one momentum BUFFER per parameter. A buffer is a
    tensor that accumulates a running blend of past gradients — it is the
    optimizer's own private state, the same species of thing as
    BatchNorm's running statistics. Before the first step, no gradients
    have been seen yet, so initialize the buffers as a list of None
    values, one slot per parameter.

    WHAT step() DOES. Everything happens inside a `with torch.no_grad():`
    block, because updating parameters is bookkeeping, not a computation
    the tape should record. For each parameter p with gradient g and
    buffer slot i, there are two cases. If this is the first step (the
    buffer slot still holds None), the buffer becomes a detached COPY of
    the gradient: bufs[i] = g.detach().clone(). On every later step, the
    buffer is updated IN PLACE by the momentum formula: multiply the
    buffer by the momentum coefficient, then add the gradient (the
    in-place spelling is bufs[i].mul_(momentum).add_(g)). In both cases,
    the parameter then takes one downhill step, in place:
    p.sub_(lr * bufs[i]), which means p becomes p minus lr times the
    buffer. Note that the step direction is the BUFFER (the smoothed
    gradient history), not the raw gradient.

    WHAT zero_grad() DOES. It visits every parameter and sets its .grad
    attribute to None, so that the next backward call starts from clean
    mailboxes. It does NOT touch the momentum buffers — those persist
    across steps by design; that persistence is what momentum IS.
    """

    def __init__(self, params: list[torch.Tensor], lr: float, momentum: float = 0.9):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.momentum_buffer = [None for p in params]

    def step(self) -> None:
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if self.momentum_buffer[i] is None:
                    self.momentum_buffer[i] = p.grad.detach().clone()
                else:
                    self.momentum_buffer[i].mul_(self.momentum).add_(p.grad)
                p.sub_(self.lr * self.momentum_buffer[i])

    def zero_grad(self) -> None:
        """Set every param's .grad to None (or zero it — None is cleaner)."""
        for p in self.params:
            p.grad = None


def clip_grad_norm(params: list[torch.Tensor], max_norm: float) -> float:
    """Clip all gradients by their single GLOBAL norm, matching the
    behavior of torch.nn.utils.clip_grad_norm_ (which is banned here —
    you are building it).

    WHY THIS EXISTS. Occasionally a training step produces enormous
    gradients (a bad batch, an unstable moment), and one such step can
    fling the parameters far from anything sensible. Clipping puts a
    ceiling on how far a single step can move, while leaving ordinary
    steps completely untouched.

    WHAT TO COMPUTE. Imagine concatenating every parameter's gradient
    into one long vector. Compute that vector's Euclidean norm: square
    each gradient tensor's norm, sum those squares over all parameters,
    and take the square root of the total. This is ONE number for the
    whole model, called the total norm.

    WHAT TO DO WITH IT. Compare the total norm against max_norm. If the
    total is larger, every gradient is too big by the same proportion, so
    scale EVERY gradient in place by the single shared factor
    max_norm / (total_norm + 1e-6). The tiny 1e-6 in the denominator
    guards against dividing by zero and matches PyTorch's own formula,
    which the checker compares you against. If the total norm does not
    exceed max_norm, change nothing at all.

    WHAT TO RETURN. Return the total norm as a plain Python float, and
    make sure it is the value measured BEFORE any clipping happened.
    """
    total_norm = torch.sqrt(sum([(p.grad**2).sum() for p in params]))
    if total_norm > max_norm:
        for p in params:
            p.grad.mul_(max_norm / (total_norm + 1e-6))
    return float(total_norm)

