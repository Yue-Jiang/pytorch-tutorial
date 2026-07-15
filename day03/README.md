# Day 3 — The indexing family (~30 min)

You own slicing, bool masks, and index grids from course #1. Today adds the
batched per-row pair you only half-met: **gather** (read by index table)
and **scatter** (write by index table).

## Concepts

- **`gather(dim, index)`**: `out[i][j] = x[i][index[i][j]]` (dim=1) — every
  row picks its own columns. It's the batched generalization of
  `x[arange(N), cols]`.
- **`scatter_add_(dim, index, src)`**: the inverse — route values TO
  indexed slots, summing collisions. Histograms, one-hots, segment sums.
- **`where(cond, a, b)`**: elementwise select — the tensor `if`.
- Reading `topk`/`argsort` output as *coordinates* to feed back into
  gather/indexing.

## Your task

In `day03/exercise.py`: `my_gather` (no torch.gather!), `pick_per_row`,
`top2_mask`, `histogram`.

## Check

```bash
python check.py day03
```

## Hints

<details><summary>Hint — my_gather</summary>
For 2-D, dim=1: rows are `arange(N).unsqueeze(1)` (a column), columns are
`index` itself — advanced indexing `x[rows, index]` broadcasts the pair.
That IS gather.
</details>

<details><summary>Hint — histogram</summary>
Start from `torch.zeros(n_bins)`, then `scatter_add_` along dim 0 with
`src=torch.ones_like(ids, dtype=...)`. (`bincount` exists but is banned —
build it once yourself.)
</details>
