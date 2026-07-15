# Day 9 — Devices & performance (~30 min)

Course #1 ran everything on CPU. Your Mac has a GPU (`mps` — Metal), and
device discipline is what makes code portable to it and to CUDA machines.

## Concepts

- **Resolution order**: prefer `cuda` if available, else `mps` (Apple),
  else `cpu`. `torch.cuda.is_available()` /
  `torch.backends.mps.is_available()`.
- **Everything must agree**: an op with inputs on two devices errors.
  Models move with `.to(device)` (recursive, thanks to registration);
  batches must be moved *per batch* — the classic training-loop line.
- **Nested state**: checkpoints and batches are often dicts/lists of
  tensors; moving them takes a small recursive function. (This is also
  how `map_location` thinks.)
- **Honest timing**: warm up first (first calls pay one-time costs), time
  many iterations, take the best-of-k or the median — never a single cold
  measurement. (On GPU you'd also need to synchronize; on CPU/mps-free
  code, wall clock is fine.)

## Your task

In `day09/exercise.py`: `best_device`, `move_tree`, `time_op`.

Afterwards, for fun (not tested): port course #1's capstone to
`best_device()` and see what the robot's training time does.

## Check

```bash
python check.py day09
```

## Hints

<details><summary>Hint — move_tree recursion</summary>
Three cases: Tensor -> `.to(device)`; dict -> rebuild with moved values;
list/tuple -> rebuild with moved items (preserve the type!); anything
else -> return unchanged.
</details>

<details><summary>Hint — time_op</summary>
`time.perf_counter()`, not `time.time()` — it's the high-resolution
monotonic clock made for this.
</details>
