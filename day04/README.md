# Day 4 — einsum (~30 min)

One function that subsumes matmul, bmm, outer products, traces, transposes,
and weighted sums: **`torch.einsum`**. It's the lingua franca of research
code — today you learn to read and write it.

## The notation

`einsum("bhqd,bhkd->bhqk", q, k)` reads: inputs have these named axes;
**an index appearing in both inputs but NOT the output gets multiplied and
summed over** (contracted); indices in the output survive.

- repeated + omitted = contract:  `"ij,jk->ik"` is matmul (j contracted)
- shared + kept = batch:          `b` in `"bij,bjk->bik"` (bmm)
- omitted from output = sum out:  `"ij->i"` sums over j
- no contraction at all = outer:  `"i,j->ij"`

Your transformer's attention, in one line each:
`scores = einsum("bhqd,bhkd->bhqk", q, k)`,
`out = einsum("bhqk,bhkd->bhqd", w, v)`.

## Your task

In `day04/exercise.py` — each function must be **a single `torch.einsum`
call** (the checker inspects your source: `@`, `matmul`, `bmm`, `.sum`,
`outer`, `trace` are banned): `attn_scores`, `outer`, `batch_trace`,
`weighted_sum`, `bilinear`.

## Check

```bash
python check.py day04
```

## Hints

<details><summary>Hint — trace</summary>
The diagonal is where both indices are equal: reuse ONE letter twice in a
single input: `"...ii->..."`-style thinking (spell the batch dim
explicitly: `"bii->b"`).
</details>

<details><summary>Hint — bilinear</summary>
`x A y` with batch: `"bi,ij,bj->b"` — einsum happily takes three inputs.
</details>
