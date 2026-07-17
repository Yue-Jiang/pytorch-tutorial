# Day 5 reflection — reductions & elementwise (2026-07-15)

## Mistakes I made (and what they taught)

1. **The keepdim trap, five days later** — logsumexp's `amax(dim)` without
   restoring the axis: crashes for dim=-1, *accidentally works* for dim=0,
   and would be silently wrong-axis on square inputs. My oldest adversary,
   reproduced from memory. Settled on a personal style for
   reduce-and-restore — three equivalent bookkeepings:
   (a) keepdim throughout + final squeeze; (b) keepdim only for the
   subtraction; (c) **drop-and-unsqueeze-on-demand** (my default:
   `xmax.unsqueeze(dim)` at the one line that needs the axis back —
   everything else stays reduced, which is what the final `xmax + ...`
   wants anyway). Shape-at-use, not shape-at-birth.
2. **Missed `correction=0`** — kneejerk `var`/`std` takes the DEFAULT,
   and the default is the statistician's unbiased estimator, not the
   normalizer's. Diagnosis lesson was the real prize: **uniform relative
   error = one global scale factor = ask what number it is.**
   1.6% on a size-32 dim = √(32/31) = Bessel. Catalog of self-naming
   constants: √(N/(N−1)) correction bugs, √d missing attention scaling,
   1/(1−p) dropout scaling, −N sign+reduction, ×N sum-vs-mean.
3. **Identical failure after an edit is itself a diagnostic** — byte-same
   mismatch numbers meant my edit never touched the causal path (my "fix"
   was aimed at a different function entirely — two patients, two
   diseases; say which file/function a diagnosis refers to).

## Core learnings

- **Why `x[mask].mean()` can't be masked_mean**: boolean indexing flattens
  — row structure destroyed, one anonymous pile, one scalar. Per-row
  selections are RAGGED (rows keep different counts) and no rectangular
  tensor holds them. The pattern that replaces it:
  **neutralize, don't remove** — `(x·mask).sum(dim) / mask.sum(dim).clamp(min=1)`
  stays rectangular, batches fully, divides by true counts, and clamp
  makes empty rows a clean 0 instead of nan. Same pattern family as
  padding masks and attention's −inf: pick the neutral element that makes
  excluded entries invisible to the reduction — **0 for sum, −inf for
  max/softmax, 1 for product.**
- einsum-as-plain-sum (`'mnd->mn'`) is legal but `.sum(-1)` says it
  plainer (day-4 note, second sighting). The einsum-native spelling of
  square-and-sum is the self-inner-product: `einsum('mnd,mnd->mn', diff,
  diff)` — Frobenius per slot.
- logsumexp now exists as MY named function — course-#1 day-1's algebra,
  third build, this time legally.

## Carry into day 6

Autograd. Door question: `grad_of(f, x)` must not touch the caller's
tensor — the recipe is `x.detach().clone().requires_grad_(True)`. Arrive
able to say what each of the three calls contributes, and what would go
wrong with just `x.requires_grad_(True)` (two distinct problems — one
about mutation, one about tape hygiene).
