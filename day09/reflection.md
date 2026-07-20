# Day 9 reflection — devices & performance (2026-07-20)

## Devices

- **mps = Metal Performance Shaders**: Metal is Apple's low-level
  graphics-and-compute API (their CUDA analog); MPS is the library of
  optimized GPU kernels on top of it; PyTorch's mps backend routes tensor
  ops through it to the Apple-silicon GPU. Two facts: M-series Macs have
  UNIFIED memory (CPU and GPU share one pool, so cpu↔mps transfers are
  cheaper than CUDA's over-the-bus copies), and the backend is younger
  than CUDA's — some ops are unsupported ("works on CUDA, complains on
  mps" is a real bug category). Resolution order: cuda, then mps, then
  cpu. best_device() returned mps on this machine — first time either
  course addressed my GPU.

## move_tree

- **There is no in-place device move.** `.to(device)` always returns a
  different tensor (crossing devices always buys new boxes — day 2), and
  tuples are immutable anyway, so the function must RECONSTRUCT: build a
  mirror of the nesting (same keys, same order, same container types,
  non-tensors passed through) with moved tensors as the new leaves. The
  caller's structure is untouched afterward. Recursion produces this
  naturally because every branch RETURNS something built from recursive
  calls. Polite no-op: a tensor already on the target device comes back
  as the same object.
- **The pairing-tool taxonomy** (my dict bug: `enumerate(obj)` over a
  dict pairs a COUNTER with each KEY — I built a dict with integer keys
  and string values, hence KeyError 'a'):
  `enumerate(sequence)` → (position, item), for when indices are needed;
  `dict.items()` → (key, value), the only correct walk of a dict;
  `zip(a, b)` → pairs from two parallel sequences.
  Also: iterating a bare dict yields KEYS only.
- **Recursion threads its own arguments**: each recursive call is a
  fresh invocation with no memory of the outer call — `move_tree(v)`
  forgot the device; the fix had THREE call sites to sweep (swept all
  three; denied the half-edit tally).
- `isinstance(obj, torch.Tensor)` is the documented-preferred spelling
  over `torch.is_tensor(obj)`; using isinstance throughout makes the
  five branches uniform.

## time_op

- **fn is a first-class function value**: calling it is `fn()` — the
  same mechanism as day 6's `f(xx)` in grad_of, just zero-argument.
- **perf_counter returns SECONDS** (float, sub-microsecond resolution);
  differences of two readings are elapsed seconds directly. Its absolute
  value is meaningless (arbitrary reference point): it is a STOPWATCH,
  not a clock — monotonic, never adjusted backward, unlike time.time.
- **The toll, both halves.** (1) Warmup absorbs one-time costs: lazy
  initialization (GPU driver context on first op; allocator pools
  grabbed once then recycled), kernel compilation and autotuning (cuDNN
  races algorithms on first call; torch.compile moves ALL cost into the
  first call), cold caches at every layer (OS page faults on first
  touch, CPU data/instruction caches, branch predictors, lazy imports),
  and clock ramp-up. The first call measures building-the-machinery
  PLUS using it; steady state — what a training loop actually
  experiences — is using it alone. (2) Timing noise is ONE-SIDED:
  interference only ever makes calls slower, so the distribution is
  right-skewed and the mean is systematically biased upward by the
  tail; the median (or even the minimum, in some harnesses) is the
  honest summary. Stronger than the generic outlier argument: here the
  direction of every outlier is known.
- **Name-error taxonomy, third species**: correctly qualified name,
  module never imported (`statistics.median` without
  `import statistics`). The genus now: forgot the prefix (eye, sqrt);
  wrong namespace for the operand's type (torch.max on a list); right
  namespace, missing import. All cured by the same question: where does
  this name live, and is that place in scope?

## Not done (optional, for fun later)

Port course #1's capstone to best_device() and measure what mps does to
the robot's training time.

## Carry into day 10

The finale. Six pre-written functions, each with exactly one planted
bug; all parts start RED, not blank. The job per function: read the
failure, NAME the bug class, repair with the smallest correct change.
Six one-line diagnoses are due at the review — that list is the final
exam of both courses.
