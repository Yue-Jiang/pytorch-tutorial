# Day 8 reflection — data & optim machinery (2026-07-20)

## pad_collate

- The batch matrix is filled by the idiomatic "loop over the batch,
  vectorize within each row" shape: raggedness means no rectangular op
  can write differently-sized pieces, so a B-iteration loop of
  slice-assignments (`batch[i, :len(s)] = s`) is professional code —
  the anti-pattern is looping over ELEMENTS, not rows. The mask can skip
  the loop entirely via the comparison grid
  `torch.arange(L) < lengths.unsqueeze(1)`.
- Mistakes: `torch.max(python_list)` — torch functions want tensors; the
  builtin `max` is the tool for Python lists. `torch.fill` is not a
  creation op — `torch.full(shape, value)` creates; `fill_` mutates an
  existing tensor; for an all-False canvas, `zeros(..., dtype=bool)` is
  plainest. And one glorious typo: `torch[i, ...] = False` — assigning
  into the MODULE ("'module' object does not support item assignment"),
  with the mask value inverted on the same line.

## SGDMomentum

- `register_buffer` is nn.Module machinery; an optimizer is a plain
  Python class and stores its state as ordinary attributes. The analogy
  to BatchNorm's buffers is conceptual (persistent, non-gradient state),
  not mechanical. Real torch.optim optimizers keep state in `self.state`
  with their OWN state_dict — which is why checkpointing training saves
  two state dicts.
- **The reference-versus-values law**: when you KEEP a tensor someone
  else owns, clone it (assignment stores a reference; backward's `+=`
  into `.grad` would corrupt an aliased buffer) and detach it (no_grad
  only stops NEW recording — it does not sever history already attached
  to a stored tensor). When you merely READ a tensor's values in an
  operation (`buf.mul_(m).add_(g)`), nothing is retained and neither
  call is needed.
- The out-of-place spelling (`buf = buf * m + g`) is CORRECT here —
  fresh result, no alias, no tape under no_grad, same values — but
  wasteful: allocation churn every step, new object identity, and full
  reliance on the no_grad envelope. Classified: the first-step clone is
  a RULE; in-place updating is idiom + efficiency. Worth knowing which
  disciplines are rules and which are rituals.

## clip_grad_norm

- The serious bug: operating on `p` (the WEIGHTS) instead of `p.grad` —
  in both the norm and the scaling. The interpreter itself confessed the
  identity: "a leaf Variable that requires grad is being used in an
  in-place operation" fires precisely BECAUSE parameters require grad
  and gradients do not. Fixed the norm site, missed the scaling site —
  the half-edit tally grows again.
- `p.grad**2.sum()` — the tokenizer reads `2.` as a float literal and
  chokes ("invalid decimal literal"); exponentiation has sharp elbows;
  parenthesize: `(p.grad ** 2).sum()`.
- **Container-versus-elements refinement**: torch.sum/torch.max demand a
  TENSOR argument and refuse lists even of tensors; builtins iterate any
  container and combine elements with Python operators, so builtin `sum`
  over a list of scalar tensors works — and its RESULT IS A TENSOR
  (result type follows the elements, not the tool), so torch.sqrt (or
  `.sqrt()`, which dodges the namespace question) applies after, and
  `.item()`/float() crosses back to Python-land for the return.
- Invented a `/ len(params)` — norms do not average; the statistician's
  reflex toward means, caught by the spec.
- Bare-name tally #5: `sqrt`.

## The door toll (global vs per-tensor clipping)

All gradients together are ONE ARROW in the big parameter space. One
global factor changes only the arrow's LENGTH; its direction — the
relative movement of every layer against every other — is exactly
preserved. Per-tensor clipping scales segments differently and BENDS the
arrow: layers with large gradients get suppressed relative to neighbors,
and the step heads somewhere else. Clipping's contract: bound how far a
bad step can go while completely trusting the gradient about which way.

## Carry into day 9

Devices and timing. Door question for `time_op`: the spec demands a
warmup phase and the MEDIAN of many timed iterations. What one-time
costs does the warmup absorb, and why is the median the right summary
rather than the mean? (Day 5's outlier knowledge applies.)
