# Day 8 — Data & optim machinery (~35 min)

The two engine-room pieces course #1 let you skip: **collating
variable-length data** (why DataLoaders have a `collate_fn`) and **what an
optimizer actually does per step** (build SGD-with-momentum, verify against
torch's).

## Concepts

- **Padding + mask**: variable-length sequences can't stack; pad to the
  batch max and return a bool mask saying which entries are real. (Day 5's
  masked_mean is who consumes such masks; course #1 day 3's nan row is
  what happens when a mask row is all-False.)
- **SGD + momentum**, the actual update:
      buf = momentum * buf + grad          (buf starts as a copy of grad)
      p  -= lr * buf
  Mutations under no_grad, in-place on state you own — day-1-of-course-#1
  rules.
- **Gradient clipping by GLOBAL norm**: total_norm = norm of ALL grads
  concatenated; if it exceeds max_norm, scale every grad by
  max_norm/total_norm. One scale factor for the whole model, not per-tensor.

## Your task

In `day08/exercise.py`: `pad_collate`, `SGDMomentum`, `clip_grad_norm`.

## Check

```bash
python check.py day08
```

## Hints

<details><summary>Hint — collate</summary>
Find the max length, `torch.zeros((B, L), dtype=...)` as canvas, loop
sequences writing each into its row prefix (a Python loop over B is fine
and normal here), build the mask the same way — or from lengths with
`arange(L) < lengths.unsqueeze(1)` (the grid trick).
</details>

<details><summary>Hint — momentum buffers</summary>
Keep `self.bufs` as a list parallel to params, initialized lazily on the
first step (`clone().detach()` of each grad). All updates inside
`torch.no_grad()`, using in-place ops (`mul_`, `add_`, `sub_`).
</details>
