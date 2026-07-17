# Day 3 reflection — the indexing family (2026-07-15)

## The coordinate-arrays model (advanced indexing, complete)

- **A single tensor index consumes dim 0 only**: `x[index]` picks WHOLE
  ROWS arranged in index's shape — `(N,C)[(N,K)] → (N,K,C)`. My near-miss:
  my_gather's column numbers would have been spent as row numbers.
- **One index tensor per dim; they broadcast WITH EACH OTHER** to a common
  shape S; output shape = S + un-indexed trailing dims; each slot s reads
  `x[A[s], B[s]]`. The three shape rules: (1) values bounded by the dim
  each tensor indexes (runtime-checked); (2) A,B must broadcast (the ONLY
  shape constraint — "A + B wouldn't raise"; S = torch.broadcast_shapes);
  (3) shapes unrelated to x's shape entirely.
- **Zip vs product are the same rule**: same shapes → elementwise pairs;
  complementary 1s → all combinations (the old grid trick is an input
  pattern, not the mechanism); (N,1) vs (N,K) → per-row zip = gather.
  The "expansion" is day-2's stride-0 expand — visible via
  broadcast_tensors (strides (1,0) and (0,1)).
- Dragon flagged, not fought: advanced indices mixed with slices
  (`x[A, :, B]`) relocate S to the front. Look up when needed.

## The toll theorem (why x[idx] always copies)

A view is a triple; every reachable element sits at
`offset + Σ index·stride` — an **affine walk** (fixed start, fixed jumps).
A tensor index encodes an arbitrary, data-dependent itinerary — revisits
([0,0]), reversals ([2,1]), teleports — that no (offset, strides) can
express. **Views are formulas; gathers are lists.** Formally: views are
exactly the affine reindexings of storage; affine∘affine = affine is why
views-of-views collapse to one triple.

- Affine, formally: f(x) = Ax + b — linear map (A: stretch/rotate/shear,
  fixes origin) plus translation (b). Preserves lines, parallelism, ratios
  along lines; not lengths/angles/origin. nn.Linear is affine (bias = b);
  LayerNorm's weight/bias is PyTorch's "elementwise_affine". History:
  Euler named it (*affinis*, 1748); Möbius built its coordinates (1827);
  Klein's Erlangen Program (1872) placed it: a geometry = a transformation
  group + its invariants.

## scatter_add_ (gather's write-direction mirror)

- `out.scatter_add_(0, ids, src)`: for each i, `out[ids[i]] += src[i]` —
  deposits at named coordinates, **collisions accumulate**.
- Why it must exist: `out[ids] += 1` is silently broken under repeats —
  desugars to gather→add→plain write, last-write-wins, collisions dropped
  (verified: [1,1,1] vs correct [1,1,3]). scatter_add is the += that means
  += under repetition; atomic adds make it GPU-parallel-safe.
- Contract: src pairs ELEMENTWISE with index (same shape); dtypes of out
  and src must match (both sides bit me — see mistakes); ids values must
  be legal indices into out (bounds, NOT uniqueness: repeats are the
  point, empty bins fine). n_bins is the caller's declaration of the value
  universe, not derived from data (dice need 6 bins even if no one rolls 3).
- **I've used this for 10 years without seeing it**: `table(ids)` =
  src-ones scatter_add; `rowsum(x, g)` / `group_by |> summarise(sum(x))` =
  arbitrary-src. src=ones is n(), arbitrary src is sum(x). Weighted
  survey tables = src-weights. Group means = two scatters + divide.
  Embedding backward = scatter_add of gradients (repeated lookups sum).
- **Tidyverse ↔ tensor Rosetta stone**: summarise(n())→scatter_add(ones);
  summarise(sum(x))→scatter_add(x); filter→bool mask; mutate→elementwise;
  select→slice; arrange→argsort+gather; left_join→gather by key;
  pivot_wider→scatter to 2-D canvas. Tidyverse organizes by data
  semantics; tensors by memory coordinates. Same computations, different
  altitude.

## Mistakes I made (and what they taught)

1. **Bare `topk`** — third `torch.`-namespace tally (eye, full, topk).
   Escape hatch: method spelling `x.topk(...)` dodges the namespace.
2. **Positional argument after keyword** — `f(dim=0, ids, ...)` is a
   SyntaxError (pytest died at COLLECTION — the file couldn't import).
   Once you name an argument, everything after must be named.
3. **src sized to the bins, not the ballots** — `ones(n_bins)` vs
   `ones_like(ids)`: src zips with index, one deposit per element.
   ones_like fixes length AND dtype in one call — that's why the idiom.
4. **Ledger dtype** — `torch.zeros(n)` is float32 by default (day-1 fact);
   scatter demands self.dtype == src.dtype. Same line, three successive
   errors, each strictly later in the pipeline: parser → argument
   contract → operation. Fixing bugs in dependency order.

## TODO

- [ ] Retype `top2_mask` via the scatter route:
  `zeros_bool.scatter_(1, topk.indices, True)` — compare with my threshold
  route (tie behavior differs: threshold keeps ties, scatter keeps exactly
  k). Checker accepts both.

## Carry into day 4

einsum. Arrive able to answer: in `"bhqd,bhkd->bhqk"`, what happens to an
index that appears in both inputs but NOT the output (d)? One that appears
in both inputs AND the output (b, h)? One that appears in one input and
the output (q, k)? Those three cases are the entire notation.
