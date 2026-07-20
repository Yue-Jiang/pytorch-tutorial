# Day 10 reflection — bug forensics finale (2026-07-20) 🎓

## The exam transcript: six diagnoses

1. **stable_softmax — missing stability shift.** No max-subtraction
   before exp; large inputs overflow to inf/nan. Repair:
   `x - x.amax(dim=-1, keepdim=True)`. (My first repair attempt
   resurrected two day-1 scars: `x.max(dim=-1)` returns a (values,
   indices) namedtuple — amax returns just values — and keepdim was
   missing. The final exam opened exactly where course #1 began.)
2. **causal_scores — mask polarity inversion.** masked_fill writes where
   True; the code built a KEEP-mask and filled exactly there, nuking the
   past and keeping the future: an anti-causal attention. My own
   course-1 day-3 bug, resurrected. Repair: fill where `~mask`.
3. **merge_heads — non-contiguous view refusal.** Transpose swaps
   strides so the walking order no longer matches physical box order;
   `view` only re-dices memory order and refuses. Repair: insert
   `.contiguous()` before the view (or use reshape and accept a silent
   copy). Note: not "fragmented" — the buffer is intact; the NOTE
   disagrees with the boxes.
4. **standardize_rows — wrong-axis reduction.** Stats over dim 0
   standardize COLUMNS (an accidental BatchNorm); the spec says rows;
   the non-square test input is what exposed it (square shapes let this
   family hide — my day-5 finding). Repair: dim=1 on both stat lines.
   Restraint lesson: I wanted to add an eps guard — right instinct for
   production, wrong move in a repair. The docstring's contract has no
   eps, the test compares against the eps-free formula, and "while I
   was in there" improvements during bug fixes are their own failure
   mode. Smallest correct change means MATCH THE STATED INTENT.
5. **sgd_update — in-place update of a grad-requiring leaf outside
   no_grad.** My first diagnosis was REJECTED: I claimed `p -=` doesn't
   mutate in place / wouldn't persist across the call — backwards on a
   day-1 fact (augmented assignment on tensors IS in-place and DOES
   persist; the non-persistence intuition belongs to immutable types
   and rebinding). I then ran the controlled experiment by accident:
   swapping to `p.sub_` produced the byte-identical error, proving the
   spelling was never the issue. The real class: autograd REFUSES
   in-place ops on trainable leaves outside a `with torch.no_grad():`
   context; the operation and target were always correct (updating
   parameters is an optimizer's legitimate business); only the
   bookkeeping-declaration envelope was missing. **Same error text does
   not mean same bug**: in my day-8 clip function this exact message
   meant "wrong tensor" (weights instead of grads); here it meant
   "right tensor, missing context."
6. **sequence_loss — double softmax (probabilities fed to a
   logits-expecting loss).** F.cross_entropy applies log-softmax
   internally; pre-softmaxing normalizes twice. The crown jewel because
   it is SILENT: post-softmax values live in [0,1], so the internal
   softmax never sees a gap larger than 1.0 — confidence is capped
   forever (confident-correct predictions floor at ~0.5 loss instead of
   ~0), gradients are crushed, training plateaus early, and every line
   looks reasonable. Day 1's vaccine sentence: losses take logits so
   the exp-log plumbing stays fused. Repair: delete the softmax line.

## Meta-lessons of the finale

- **Read the failure before the code.** Every diagnosis started from
  the error text or the failing assertion; the one time I led with a
  mechanism-guess (patient five) was the one rejection.
- **Same symptom, different diseases.** The leaf-variable error meant
  wrong-tensor on day 8 and missing-context on day 10.
- **Repair is not improvement.** The eps restraint at patient four.
- **Silent bugs outrank loud ones.** Five patients announced
  themselves; the sixth would have shipped.

## Course #2, complete

All ten days green. The toolbox: tensor anatomy, storage and strides,
the indexing family, einsum, reductions, autograd down to a hand-built
ctx, the module system with Dropout and BatchNorm from scratch, collate
and a lockstep-verified optimizer, devices and honest timing, and a
forensics exam passed on six planted bug classes — most of which I had
already met, committed, and repaired in my own work across 21 days of
these two courses.
