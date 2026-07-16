# Day 2 reflection — storage & strides (2026-07-15)

## The model (boxes and sticky notes)

Memory is one flat row of boxes; a tensor is a NOTE: (storage pointer,
shape, strides, offset). Strides answer "when this index +1, jump how many
boxes?" Contiguous strides = suffix products of the shape (shape (a,b,c) →
strides (b·c, c, 1)); address = Σ index·stride. `transpose(i, j)` swaps
positions i,j in BOTH shape and stride tuples — O(1), no data motion;
`permute` = arbitrary reorder; slicing bakes fixed indices into the offset
(x[1, 1:] → offset 1·12 + 1·4 = 16). **An op allocates only if the result
can't be expressed as a triple over the existing boxes** — there's no
third case.

| Note-ops (free, share storage) | Box-buyers (allocate + copy) |
|---|---|
| view, transpose, permute, squeeze/unsqueeze, slice, expand, unfold | clone, repeat, contiguous (on non-contig), cat/stack, all arithmetic |

(`reshape` = two-faced: view when possible, silent copy when not.)

## Mistakes I made (and what they taught)

1. **`x.size[-1]`** — `size` is a method; the day-1 attr-vs-method seam,
   mirrored. Family: `x.size(-1)` ≡ `x.shape[-1]` ≡ `x.size()[-1]`; the
   error text is the decoder ("not callable" → drop parens; "not
   subscriptable" → needs parens). Then typo'd the fix (`shpae`).
2. **Quiz: unfold strides swapped.** Correct: `(step, 1)` — next window
   jumps STEP (window starts are step apart; that's what sliding means),
   within-window jumps 1 (a window is a contiguous slice). `stride(0) <
   size(1)` ⇒ rows share boxes: overlap readable off the metadata.
3. My gather-based sliding_windows draft (index grid + `x[idx]`) produces
   right values but **advanced indexing always copies** — can't be a
   view because arbitrary index patterns aren't expressible as one triple.
   That's exactly why `unfold` exists: overlapping windows ARE a valid
   triple `(num_windows, size) / (step, 1)`, but unreachable by composing
   polite note-ops — only unfold/as_strided mint notes that revisit boxes.
   (Uses: im2col convolution, audio framing, ViT patches. Course-1
   get_batch copies DELIBERATELY — training wants independent windows.)

## Write-through-view semantics (the toll)

`out = x.expand(3, 5); out[0, 0] = 99` → box 0 written: `x[0]` becomes 99
AND out's whole first column reads 99 (stride-0 row axis: every row's col 0
IS box 0). But `out += 1` → RuntimeError. **The overlap guardrail is
partial**: it fires only when the written-to view has INTERNAL overlap;
single-element writes sail through and corrupt quietly. Expanded tensors
are read-only by discipline, not enforcement — `.clone()` before writing.
Write-through-view generally: core idiom on canvases you own (pe[:, 0::2],
pad_collate rows, KV caches), bug when aliasing is accidental — clone at
ownership boundaries; in-place never on the autograd tape. `.contiguous()`
returns a new tensor (in-place is IMPOSSIBLE — rearranging shared boxes
would betray other notes); no-op returning self if already contiguous;
consume it in a chain or rebind.

## Other new knowledge

- **1-D tensors have NO orientation** — row/column are 2-D concepts. Who
  assigns orientation: broadcasting right-aligns → behaves as ROW (serves
  the `x + bias` feature-vector case); matmul assigns BY SIDE (`A @ v`:
  column; `v @ A`: row — the stats convention survives there). Discipline:
  treat (n,) as an orientation-free list; assign orientation explicitly at
  the use site.
- **Unsqueeze rule**: write the target `out[i, j] = f(a[i], b[j])`; each
  ingredient's index slot gets its values, all other slots get 1s
  (`a[:, None]`, `b[None, :]`). Load-bearing one is almost always
  `unsqueeze(1)` — bare vectors already act like rows.
- **`(D,)` demystified**: commas make tuples, parens just group; `(5)` is
  an int, `(5,)` a 1-tuple, `()` the empty tuple (no ambiguity with
  grouping, so no comma needed; `(,)` is a SyntaxError — a comma needs an
  element). Shape ladder: `()` scalar → `(D,)` vector → `(1,D)`/`(D,1)`.
  Retroactively explains the day-9 course-1 randint size tuple.
- **size(0) appears in no contiguous stride** (suffix products only use
  dims to the right) → batch dim is addressing-free: same walk for B=1 or
  B=512, appending rows shifts nothing (memmap corpora), `view(-1, ...)`
  can infer it.
- **Lineage**: address polynomial = mixed-radix numerals; Fortran baked it
  into compilation (1950s); Algol's dope vectors = the runtime triple
  (~1960s); APL made shape-manipulation an algebra (Iverson); BLAS carries
  strides as "leading dimension"; NumPy (Numeric '95, Oliphant '05) made
  zero-copy strided views the user-facing standard; Torch/PyTorch
  inherited. Broadcasting IS stride-0 views.

## Carry into day 3

The indexing family. Open thread from today: advanced indexing always
copies (gather-class ops) vs basic slicing views — day 3 builds gather and
scatter by hand, so arrive with the question: why CAN'T `x[idx]` for a
tensor idx ever be a view? (Answer's in today's notes — say it in one
sentence at the door.)
