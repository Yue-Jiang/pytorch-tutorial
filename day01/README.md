# Day 1 — Tensor anatomy (~30 min)

Every tensor is four facts: **shape, dtype, device, layout**. Today: the
first two, cold.

## Concepts

- **Creation ops**: `arange` (half-open!), `linspace` (inclusive!), `zeros/
  ones/full/empty`, `rand/randn/randint`, `eye` — and the `_like` family
  (`zeros_like(x)` = zeros with x's shape+dtype+device).
- **Dtype defaults**: floats default `float32`, Python ints → `int64`,
  bools are their own world. Integer tensors can't do calculus: `mean`,
  `var`, `softmax` all refuse them — cast first (`.float()`, `.to(dtype)`).
- **Type promotion**: mixing dtypes promotes to the "bigger" one
  (`int64 + float32 → float32`). Silent, usually fine, occasionally the bug.
- **Memory accounting**: bytes = `numel × element_size`. float32 = 4 bytes,
  int64 = 8, bfloat16 = 2, bool = 1. Know your tensor's rent.

## Your task

In `day01/exercise.py`: `creation_gallery`, `cast_like`, `safe_mean`,
`tensor_bytes`.

## Check

```bash
python check.py day01
```

## Hints

<details><summary>Hint — evens</summary>
`torch.arange` takes a step argument, same as your day-6 sinusoidal table.
</details>

<details><summary>Hint — cast_like</summary>
`.to()` accepts a tensor: `x.to(ref)` copies ref's dtype AND device in one
call.
</details>
