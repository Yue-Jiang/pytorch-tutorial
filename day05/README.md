# Day 5 — Reductions & elementwise (~30 min)

Systematizing the reduce-then-broadcast pattern you used all through
course #1 — plus the classic numerically-careful compositions.

## Concepts

- Reductions take `dim` (possibly a tuple!) and `keepdim`. You know why.
- **`logsumexp`** — the stable log-of-sum-of-exps you built inside
  log_softmax; now it's legal to build once and name.
- **Masked reductions**: mean over only the True entries — divide by the
  *count*, not the size, and guard the empty-mask nan.
- **Pairwise distances by broadcasting**: `(M,1,D) - (1,N,D)` → the
  square-of-difference grid — the outer-product idiom in 3-D.

## Your task

In `day05/exercise.py`: `logsumexp` (no torch.logsumexp), `masked_mean`,
`standardize`, `pairwise_sq_dist` (no torch.cdist, no loops).

## Check

```bash
python check.py day05
```

## Hints

<details><summary>Hint — masked_mean</summary>
`(x * mask).sum(dim) / mask.sum(dim).clamp(min=1)` — the clamp guards
empty rows (spec says they return 0, which the zero numerator handles).
Watch dtypes: bool must become float before multiplying.
</details>

<details><summary>Hint — pairwise</summary>
`diff = a.unsqueeze(1) - b.unsqueeze(0)` gives (M, N, D); square, sum the
last dim. Complementary unsqueezes — course #1 day 6.
</details>
