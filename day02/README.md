# Day 2 — Storage & strides (~30 min)

A tensor is a **flat storage buffer + shape + strides + offset**. You met
this on transformer day 4 (`view`/`transpose`/`contiguous`); today it
becomes a calculation instead of a vibe.

## Concepts

- **Strides** = memory slots to jump when each index increments. Contiguous
  `(r, c)` has strides `(c, 1)`. `transpose`/`permute` just permute the
  stride tuple — zero copy.
- **View vs copy**: views share storage (`untyped_storage().data_ptr()`
  equal); copies don't. `view`, `transpose`, `expand`, slicing → views.
  `repeat`, `contiguous` (on non-contiguous), `clone` → copies.
- **`expand` vs `repeat`**: expand stretches size-1 dims with stride 0 —
  free, read-only-ish; repeat physically tiles memory. The broadcasting
  you know is implicit expand.
- **`unfold`**: sliding windows as a strided view — day 9's batching trick,
  zero-copy edition.

## Your task

In `day02/exercise.py`: `shares_storage`, `transpose_strides`,
`tile_rows_view`, `tile_rows_copy`, `sliding_windows`.

## Check

```bash
python check.py day02
```

## Hints

<details><summary>Hint — stride arithmetic</summary>
Contiguous strides are suffix products of the shape: for (a, b, c) they're
(b*c, c, 1). A permutation of dims permutes those numbers identically.
</details>

<details><summary>Hint — tile via view</summary>
`x.unsqueeze(0)` is (1, n); `expand(m, n)` stretches the size-1 dim for
free. Check `.stride()` — the stretched dim has stride 0!
</details>
