# Day 4 reflection — einsum (2026-07-15)

## The notation, complete (index citizenship)

| Pattern | Meaning | Example |
|---|---|---|
| both inputs, NOT output | contract: multiply & sum away | `d` in attention scores |
| both inputs AND output | batch: matched slices, independent problems | `b,h` — "the head axis is a batch axis," now as notation |
| one input + output | free: enumerates the output grid | `q,k` |
| twice in ONE input | diagonal constraint | `bii->b` batched trace; `ii->i` diagonal; `ii->` trace |

Mnemonic: in the output = keep a slot per value; missing = sum it away;
shared between inputs = slices must agree. Contrasts that lock it:
`bij,bjk->bik` = bmm; drop b from output → sum of all per-b matmuls;
different letters (`bij,cjk->bcik`) → every-slice-with-every-slice.

## einsum's exact boundary

einsum speaks ONE algebra: multiply matched entries, sum omitted indices
(the (+,×) semiring). `out[i,j] = a[i] + b[j]` is NOT expressible — that's
the broadcasting grid `a[:,None] + b[None,:]` (my day-9 window grid was an
additive outer all along). Division of labor: elementwise-combine across
axes → broadcasting; multiply-then-reduce → einsum. Footnote: other
semirings exist — (max,+) gives shortest paths, (logsumexp,+) gives stable
probabilistic contractions; einsum is one algebra among several.

## Why multiply-and-sum is THE pattern (intuitions)

1. **Accountant**: total = Σ quantity × rate — proportional, independent
   contributions accumulated. Linearity FORCES this form.
2. **Statistician (my decade)**: covariance, OLS's XᵀX, and every
   expectation E[X] = Σ value × probability are dot products. Attention's
   `weights @ v` = E[value] under the attention distribution.
3. **Paths**: (AB)[i,k] = Σⱼ A[i,j]B[j,k] = multiply along path k→j→i, sum
   over stopovers j. Matmul = all paths through the hidden middle.
4. **Probabilist (deepest)**: Σⱼ P(i|j)P(j|k) = Chapman–Kolmogorov —
   **contraction is marginalization**; einsum indices are variables,
   omission integrates them out.

One sentence: multiply pairs the parts that interact; sum totals the
independent ways it can happen.

## Mistakes I made (and what they taught)

1. **Bare `diagonal`** — namespace tally #4 (eye, full, topk, diagonal).
   Standing fix: method spelling (`x.diagonal(...)`) or type `torch.` first.
2. **Half-edit** — swapped in `'bii->b'` but left the dead variable `d` as
   the argument. Third tally (across courses) on: an edit isn't done until
   the whole statement is re-read.
3. My diagonal-then-sum route would have PASSED the checker (ban-list gap
   — checker humility, same lesson as course-1 day 4's docstring bug) but
   violated the one-einsum spec and dodged the repeated-letter lesson.
   Credit where due: torch.diagonal is the day-2 catalog's funky-stride
   view (stride s₀+s₁), so the route was respectable engineering — just
   not this exercise. Also: einsum-as-plain-sum (`bn->b`) is legal but
   `.sum(-1)` says it plainer; einsum earns its keep on pairing and
   contraction.

## Carry into day 5

Reductions & elementwise. Warm-up reads to answer at the door (reading
fluency, no typing): what do `'ij->ji'`, `'ij->'`, and `'ij,ij->'` each
compute — and which of the three could have been a view?
