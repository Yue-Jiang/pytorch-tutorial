# Day 7 reflection — the nn.Module system (2026-07-15)

## Dropout

- The scaling preserves expectation: with keep-probability (1−p), the
  unscaled output has E[out] = (1−p)·x; dividing survivors by (1−p)
  restores E[out] = x, so downstream layers train against the same
  expected signal strength they will see at eval.
- Without train-time scaling, eval breaks silently: downstream weights
  spend training calibrated to a dimmed (1−p)·x signal, then eval's
  identity path suddenly delivers full-strength x — every activation
  1/(1−p) times hotter than anything the weights were fitted to. Nothing
  crashes; the model is just worse, only in production.
- "Inverted" dropout: the original 2012 version patched the EVAL side
  (multiply by (1−p) at test time, forever taxing inference); the modern
  version patches the TRAIN side so eval is a free no-op. Same agreement,
  opposite side of the train/eval divide.
- Implementation notes: the eval gate (`if not self.training: return x`)
  must come BEFORE any drop machinery — my crash was the eval test
  reaching the bernoulli line. The bernoulli route
  (`torch.bernoulli(full_like(x, p))` as a drop-mask) and the comparison
  route (`rand_like(x) >= p` as a keep-mask) are both valid.

## BatchNorm — the bug anthology (7 in one function)

1. `self.bias == ...` — the `==`-for-`=` scar, exact recurrence of
   course-1 day-8. Fails as AttributeError at construction because the
   comparison evaluates the not-yet-existing attribute.
2. Running stats as plain attributes instead of `register_buffer` (see
   blind spot below).
3. `biased=True` is not a kwarg — the spelling is `correction=0`.
4. One variance doing two jobs — normalize needs biased (correction=0),
   the running update stores unbiased (correction=1).
5. Mode inversion — I normalized with running stats during training;
   training normalizes with the CURRENT batch's own statistics.
6. `if not self.eval:` — `eval` is a bound METHOD, always truthy, so the
   condition was permanently False and updates never ran. The flag is
   `self.training` (used correctly in my own Dropout minutes earlier).
7. Rebinding `self.running_mean = new` instead of in-place mutation, with
   a (1, C) keepdim shape and no detach.
   Also: a missing close-parenthesis took the whole FILE down —
   a SyntaxError fails at import, so even the Dropout tests went red.

## The three blind spots, resolved

- **register_buffer — the three categories of module state.** Parameters
  (trained, saved, moved), buffers (NOT trained, but saved and moved),
  plain attributes (none of the three). Buffers are for state updated by
  something other than gradient descent. Concrete failures of plain
  attributes: absent from state_dict → a reloaded model evals against
  running_mean=0/var=1 (silent garbage); left behind by .to(device) →
  device-mismatch crash.
- **Descriptive N vs inferential N−1.** Normalizing THIS batch treats the
  batch as the entire population (describe: correction=0). The running
  estimate treats the batch as a SAMPLE of the data distribution
  (infer: Bessel, correction=1). The asymmetry is the first-year
  statistics rule applied twice in one function.
- **The write/read split of running stats.** Training WRITES the buffers
  (EMA blend of training batches, momentum=0.1 ≈ memory of ~10 recent
  batches) but normalizes with fresh batch stats; eval READS the buffers
  and writes nothing. Why training doesn't normalize with the smoother
  running stats: (1) gradient flow — batch stats are functions of x, so
  the normalization is differentiable and the optimizer accounts for it
  (the original paper's mechanism); (2) cold start — buffers begin at
  0/1 fantasy values; (3) feedback lag — stats trail the shifting
  activations (Batch Renormalization exists to manage that loop).
- **The graph-chain leak, with the honest nuance.** Batch stats carry the
  autograd tape only if x does — recording triggers on requires_grad of
  any input (the infection rule), not on "computed from x." In the
  checker, x is cold data, so no leak; in real use mid-network, x is
  always hot (descended from parameters), and un-severed buffer updates
  accumulate an unbounded chain of dead graphs — memory climbs every
  step, nothing crashes. detach() the stats or update under no_grad().

## Process note

Asked Claude to stop using abbreviated English after "both (C,)" cost
three exchanges to decode ("weight (ones), bias (zeros), both (C,)"
means: weight initialized to ones, bias to zeros, and both tensors have
shape (C,)). Complete sentences from here on — recorded in memory.

## Carry into day 8

Data and optimizer machinery. Door question: `clip_grad_norm` computes
ONE total norm across ALL parameters' gradients and scales every gradient
by the same single factor. Why one global factor rather than clipping
each tensor separately? What property of the overall update does global
clipping preserve that per-tensor clipping would destroy?
